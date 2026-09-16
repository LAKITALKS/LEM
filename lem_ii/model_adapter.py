"""Pinned Qwen pre-answer state extraction; caches never survive an answer."""
from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import time
from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

TEMPLATE_SHA256 = "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f"
PREFIX_IDS = [151644, 77091, 198]
REQUIRED_VERSIONS = {"torch": "2.9.1", "transformers": "4.57.6", "tokenizers": "0.22.2",
                     "huggingface-hub": "0.36.2", "numpy": "2.4.6", "safetensors": "0.7.0"}


def runtime_versions():
    versions = {name: importlib.metadata.version(name) for name in REQUIRED_VERSIONS}
    if any(versions[name].split("+")[0] != expected for name, expected in REQUIRED_VERSIONS.items()):
        raise RuntimeError("Model runtime differs from the pinned tested dependencies")
    return versions


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def visible_messages(messages, system_prompt):
    if not messages or len(messages) % 2 != 0:
        raise ValueError("Context must end with a current user message")
    for index, message in enumerate(messages):
        expected = "system" if index == 0 else ("user" if index % 2 else "assistant")
        if set(message) != {"role", "content"} or message["role"] != expected or not isinstance(message["content"], str):
            raise ValueError("Only alternating visible role/content messages are allowed")
    if messages[0]["content"] != system_prompt:
        raise ValueError("System prompt differs from the uniform protocol prompt")


def last_nonpadding_indices(mask):
    if mask.ndim != 2 or not torch.all((mask == 0) | (mask == 1)) or torch.any(mask.sum(1) == 0):
        raise ValueError("Invalid attention mask")
    positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    return torch.where(mask.bool(), positions, -1).max(dim=1).values


def selected_states(hidden_states, mask, layers):
    position = last_nonpadding_indices(mask)
    if any(layer < 0 or layer >= len(hidden_states) for layer in layers):
        raise ValueError("Requested layer unavailable; no substitute states")
    rows = torch.arange(mask.shape[0], device=mask.device)
    result = torch.stack([hidden_states[layer][rows, position] for layer in layers], dim=1)
    result = result.detach().float().cpu().numpy()
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite extracted states")
    return result


@dataclass
class TurnMeasurement:
    record: dict
    states: np.ndarray  # layers x hidden dimension, float32 storage


class ModelAdapter:
    def __init__(self, model, tokenizer, config, *, tiny_fixture=False):
        self.model, self.tokenizer, self.config = model.eval(), tokenizer, config
        self.tiny_fixture = tiny_fixture
        self.device = next(model.parameters()).device
        template_hash = digest(tokenizer.chat_template.encode())
        if template_hash != TEMPLATE_SHA256:
            raise ValueError("Pinned chat template mismatch")
        if not tiny_fixture:
            if (model.config.num_hidden_layers, model.config.hidden_size) != (36, 2048):
                raise ValueError("Unexpected Qwen architecture")
            if next(model.parameters()).dtype != torch.bfloat16:
                raise ValueError("Production requires unquantized BF16 weights")
            if getattr(model.config, "_commit_hash", None) != config["model"]["revision"]:
                raise ValueError("Loaded model revision mismatch")
        self.metadata = {
            "model_revision": config["model"]["revision"],
            "tokenizer_revision": config["model"]["tokenizer_revision"],
            "chat_template_sha256": template_hash, "prefix_token_ids": PREFIX_IDS,
            "tiny_random_fixture": tiny_fixture, "parameter_count": sum(p.numel() for p in model.parameters()),
            "weight_dtype": str(next(model.parameters()).dtype),
            "attention_implementation": model.config._attn_implementation,
            "software": runtime_versions(),
            "python": platform.python_version(), "platform": platform.platform(),
            "hardware": {"device": str(self.device), "cuda": torch.version.cuda,
                         "gpu": torch.cuda.get_device_name() if self.device.type == "cuda" else None,
                         "gpu_total_memory_bytes": torch.cuda.get_device_properties(self.device).total_memory if self.device.type == "cuda" else None},
        }

    @classmethod
    def load(cls, config, *, cache_dir=None):
        m = config["model"]
        runtime_versions()
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Real collection requires a CUDA GPU with native BF16 support")
        tokenizer = AutoTokenizer.from_pretrained(m["tokenizer_id"], revision=m["tokenizer_revision"],
                                                  trust_remote_code=False, cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(
            m["id"], revision=m["revision"], dtype=torch.bfloat16,
            trust_remote_code=False, attn_implementation=m["attention_implementation"], cache_dir=cache_dir)
        return cls(model.to("cuda"), tokenizer, config)

    def prepare(self, messages):
        visible_messages(messages, self.config["system_prompt"])
        context = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        no_prefix = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        if ids != no_prefix + PREFIX_IDS:
            raise ValueError("Actual generation prefix token boundaries differ from pinned protocol")
        maximum = self.config["model"]["max_context_tokens"]
        if len(ids) + self.config["model"]["generation"]["max_new_tokens"] > maximum:
            raise ValueError("Context plus reserved answer exceeds cap; no truncation")
        return context, ids, len(no_prefix) - 1

    def measure_reply(self, messages, guard=None):
        context, ids, user_message_end = self.prepare(messages)
        generation = dict(self.config["model"]["generation"])
        if guard:
            guard.reserve_turn(len(ids), generation["max_new_tokens"])
        inputs = torch.tensor([ids], dtype=torch.long, device=self.device)
        mask = torch.ones_like(inputs)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model(input_ids=inputs, attention_mask=mask, output_hidden_states=True,
                                 use_cache=False, logits_to_keep=1)
            states = selected_states(outputs.hidden_states, mask, self.config["extraction"]["layers"])[0]
            del outputs
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            extraction_seconds = time.perf_counter() - started
            if guard:
                guard.check_deadline()
            generated_at = time.perf_counter()
            # Explicit generation config avoids inheriting sampling/repetition defaults.
            settings = GenerationConfig(**generation)
            sequences = self.model.generate(input_ids=inputs, attention_mask=mask, generation_config=settings)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            generation_seconds = time.perf_counter() - generated_at
        output_ids = sequences[0, len(ids):].tolist()
        if not output_ids or len(output_ids) > generation["max_new_tokens"]:
            raise ValueError("Empty or over-limit token generation")
        if guard:
            guard.finish_turn(len(output_ids))
        answer = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        record = {"context": context, "input_token_ids": ids, "input_token_sha256": digest(np.asarray(ids, dtype="<i8").tobytes()),
                  "input_tokens": len(ids), "generation_prefix_position": len(ids) - 1,
                  "generation_prefix_token_id": ids[-1], "user_message_end_position": user_message_end,
                  "user_message_end_token_id": ids[user_message_end],
                  "answer": answer, "output_token_ids": output_ids, "output_tokens": len(output_ids),
                  "stop_reason": "eos" if output_ids[-1] in generation["eos_token_id"] else "max_new_tokens",
                  "extraction_seconds": extraction_seconds, "generation_seconds": generation_seconds,
                  "prefill_passes": 2, "prefill_tokens_including_generation": 2 * len(ids),
                  "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if self.device.type == "cuda" else None,
                  "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved() if self.device.type == "cuda" else None,
                  "assistant_characters": len(answer), "assistant_words": len(answer.split())}
        return TurnMeasurement(record, states)
