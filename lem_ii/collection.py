"""Session-level collection and validated replay of completed prefixes."""
from __future__ import annotations

from dataclasses import replace
import random
import numpy as np
import torch

from .design import SessionSpec, build_sessions
from .simulator import UserSimulator
from .storage import SessionStore, canonical_hash


def smoke_sessions(config):
    topic = next(t for t in config["simulator"]["topics"] if t["id"] == "community_garden")
    if topic["split"] != "development":
        raise ValueError("Smoke topic must belong to development")
    first = SessionSpec("smoke-full-24", "p07", "advance_when_addressed", "smoke",
                        topic["id"], topic["formulation_family"], config["seeds"]["smoke"], 24)
    # Same first prompt/seed in the second session provides a direct reset check.
    seed = config["seeds"]["smoke"]
    return [replace(first, session_id="smoke-full-24", split="smoke", seed=seed),
            replace(first, session_id="smoke-reset-02", split="smoke", seed=seed, turns=2)]


def collect_session(config, spec, adapter, root, run_manifest, *, guard=None, resume=False, checkpoint_hook=None):
    smoke = spec.split == "smoke"
    if not smoke:
        if not config["execution"]["study_collection_allowed"] or config["execution"]["smoke_only"]:
            raise PermissionError("Main study collection is not authorized")
        if spec not in build_sessions(config):
            raise ValueError("Study session is not the prespecified assignment")
        if spec.split == "confirmation" and not config["execution"]["confirmation_generation_allowed"]:
            raise PermissionError("Confirmation collection is not authorized")
    elif spec not in smoke_sessions(config):
        raise ValueError("Only the two prespecified smoke-only sessions may use the technical path")
    data_kind = "synthetic_fixture" if adapter.tiny_fixture else ("real_model_smoke" if smoke else "real_model_study")
    binding = {"run": run_manifest, "config_sha256": canonical_hash(config), "adapter": adapter.metadata,
               "max_new_tokens": config["model"]["generation"]["max_new_tokens"],
               "eos_token_ids": config["model"]["generation"]["eos_token_id"],
               "state_shape": [len(config["extraction"]["layers"]), adapter.model.config.hidden_size],
               "data_kind": data_kind, "eligible_for_scientific_analysis": data_kind == "real_model_study"}
    store = SessionStore(root, spec, binding, resume=resume)
    simulator = UserSimulator(config, spec)
    history = [{"role": "system", "content": config["system_prompt"]}]
    previous = None
    # Reconstruct policy and exact visible history, never load hidden state or KV caches.
    for record in store.records():
        user = simulator.next_message(previous)
        if user != record["user"]:
            raise ValueError("Resume simulator replay differs from saved user text")
        history.append({"role": "user", "content": user})
        context, ids, _ = adapter.prepare(history)
        if ids != record["input_token_ids"] or context != record["context"]:
            raise ValueError("Resume context/tokenization differs from checkpoint")
        previous = record["answer"]
        history.append({"role": "assistant", "content": previous})
    if store.manifest["status"] == "complete":
        return store
    if guard:
        guard.start_session(spec.session_id, spec.turns)
    random.seed(spec.seed)
    np.random.seed(spec.seed)
    torch.manual_seed(spec.seed)
    try:
        while store.manifest["completed_turns"] < spec.turns:
            user = simulator.next_message(previous)
            history.append({"role": "user", "content": user})
            measurement = adapter.measure_reply(history, guard)
            store.append(measurement, user, simulator.last_audit)
            if checkpoint_hook:
                checkpoint_hook(store)
            previous = measurement.record["answer"]
            history.append({"role": "assistant", "content": previous})
            print(f"{spec.session_id} turn {store.manifest['completed_turns']}/{spec.turns}: "
                  f"input={measurement.record['input_tokens']} output={measurement.record['output_tokens']}", flush=True)
    except Exception as error:
        store.fail(error)
        raise
    store.validate()
    return store
