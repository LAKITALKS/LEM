"""Real local Qwen2 mechanics with random tiny weights: NOT LLM study data.

The tokenizer must already be cached at the exact pinned revision. Tests do not
download assets and never load the real 3B weights or invoke remote execution.
"""
from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from lem_ii.collection import collect_session, smoke_sessions
from lem_ii.design import load_config
from lem_ii.model_adapter import ModelAdapter, PREFIX_IDS, TurnMeasurement, selected_states, last_nonpadding_indices
from lem_ii.storage import SessionStore, atomic_json


class ModelStorageTests(unittest.TestCase):
    def test_partial_prefix_replay_and_persistence_failure_preserve_complete_data(self):
        from lem_ii.study import StudyRuntimeGuard
        class StopAfterFive:
            count = 0
            def check_deadline(self): pass
            def start_session(self, *args): pass
            def reserve_turn(self, *args):
                if self.count == 5: raise RuntimeError("fixture interruption")
                self.count += 1
            def finish_turn(self, *args): pass
        first = smoke_sessions(self.config)[0]
        checkpoints = []
        run = {"run_id": "partial-persistence-fixture"}
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError):
                collect_session(self.config, first, self.adapter, directory, run,
                                guard=StopAfterFive(), checkpoint_hook=lambda s: checkpoints.append(s.manifest["completed_turns"]))
            manifest = json.loads((Path(directory) / first.session_id / "manifest.json").read_text())
            self.assertEqual(manifest["completed_turns"], 5)
            original_hashes = manifest["turn_hashes"]
            guard = StudyRuntimeGuard({"deadline_epoch": 100, "max_sessions": 1,
                                       "max_prefill_tokens": 1000000, "max_generation_tokens": 1000,
                                       "completed_prefixes": {first.session_id: 5}},
                                      Path(directory) / "guard.json", clock=lambda: 0)
            def persist(store):
                checkpoints.append(store.manifest["completed_turns"])
                if store.manifest["completed_turns"] == 24:
                    raise OSError("fixture persistence failure after final valid append")
            with self.assertRaises(OSError):
                collect_session(self.config, first, self.adapter, directory, run,
                                guard=guard, resume=True, checkpoint_hook=persist)
            saved = SessionStore(directory, first, manifest["binding"], resume=True)
            self.assertEqual(saved.manifest["status"], "complete")
            self.assertEqual(saved.manifest["turn_hashes"][:5], original_hashes)
            self.assertEqual(checkpoints, list(range(1, 25)))
            self.assertEqual(len(guard.state["turns"]), 19)
            complete = collect_session(self.config, first, self.adapter, directory, run, resume=True)
            self.assertEqual(complete.manifest["turn_hashes"], saved.manifest["turn_hashes"])

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(1706)
        cls.config = deepcopy(load_config())
        cls.config["extraction"]["layers"] = [0, 1, 2]
        cls.config["extraction"]["primary_layer"] = 2
        cls.config["model"]["generation"]["max_new_tokens"] = 2
        root = Path(__file__).resolve().parents[2]
        snapshot = (root / ".runtime" / "hf-cache" / "models--Qwen--Qwen2.5-3B-Instruct" /
                    "snapshots" / cls.config["model"]["tokenizer_revision"])
        if not (snapshot / "tokenizer.json").is_file():
            raise unittest.SkipTest("Exact pinned tokenizer not cached: local model extraction remains unverified")
        # Snapshot path avoids an upstream tokenizer-version check making a network
        # request even when a Hub model ID is passed with local_files_only=True.
        cls.tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
        model_config = Qwen2Config(
            vocab_size=len(cls.tokenizer), hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            max_position_embeddings=8192, attention_dropout=0.0,
            bos_token_id=151643, eos_token_id=151645, pad_token_id=151643,
            tie_word_embeddings=True,
        )
        model_config._attn_implementation = "sdpa"
        cls.model = Qwen2ForCausalLM(model_config)
        cls.adapter = ModelAdapter(cls.model, cls.tokenizer, cls.config, tiny_fixture=True)
        cls.messages = [{"role": "system", "content": cls.config["system_prompt"]},
                        {"role": "user", "content": "Explain how to grow tomatoes in one sentence."}]
        cls.measurement = cls.adapter.measure_reply(cls.messages)

    def test_actual_prefill_states_equal_direct_qwen_forward_and_layer_zero_embedding(self):
        measurement = self.measurement
        ids = torch.tensor([measurement.record["input_token_ids"]])
        mask = torch.ones_like(ids)
        with torch.inference_mode():
            outputs = self.model(input_ids=ids, attention_mask=mask, output_hidden_states=True,
                                 use_cache=False, logits_to_keep=1)
        expected = np.stack([outputs.hidden_states[layer][0, -1].float().numpy() for layer in (0, 1, 2)])
        np.testing.assert_allclose(measurement.states, expected, rtol=1e-5, atol=1e-6)
        with torch.inference_mode():
            embedding = self.model.model.embed_tokens(ids)[0, -1].float().numpy()
        np.testing.assert_array_equal(measurement.states[0], embedding)
        self.assertEqual(measurement.states.shape, (3, 16))
        self.assertEqual(measurement.states.dtype, np.float32)
        self.assertTrue(self.adapter.metadata["tiny_random_fixture"])

    def test_left_and_right_padding_select_actual_last_nonpadding_forward_state(self):
        ids = torch.tensor([[0, 0, 11, 12, 13], [21, 22, 23, 0, 0]])
        mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 0, 0]])
        self.assertEqual(last_nonpadding_indices(mask).tolist(), [4, 2])
        with torch.inference_mode():
            outputs = self.model(input_ids=ids, attention_mask=mask, output_hidden_states=True,
                                 use_cache=False, logits_to_keep=1)
        extracted = selected_states(outputs.hidden_states, mask, [0, 1, 2])
        for row, position in ((0, 4), (1, 2)):
            for layer in (0, 1, 2):
                np.testing.assert_array_equal(extracted[row, layer], outputs.hidden_states[layer][row, position].float().numpy())
        with self.assertRaises(ValueError):
            last_nonpadding_indices(torch.zeros((1, 3), dtype=torch.long))
        with self.assertRaises(ValueError):
            last_nonpadding_indices(torch.tensor([[1, 2]]))
        with self.assertRaises(ValueError):
            selected_states(outputs.hidden_states, mask, [3])

    def test_pinned_generation_prefix_is_measured_not_user_end_or_current_answer(self):
        record = self.measurement.record
        self.assertEqual(record["input_token_ids"][-3:], PREFIX_IDS)
        self.assertEqual(record["generation_prefix_token_id"], 198)
        self.assertEqual(record["generation_prefix_position"], record["input_tokens"] - 1)
        self.assertEqual(record["user_message_end_position"], record["generation_prefix_position"] - 3)
        rendered = self.tokenizer.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
        self.assertEqual(record["context"], rendered)
        self.assertTrue(record["context"].endswith("<|im_start|>assistant\n"))
        self.assertEqual(record["output_tokens"], len(record["output_token_ids"]))
        self.assertLessEqual(record["output_tokens"], 2)
        # Current assistant answers are invalid measurement contexts by role order.
        with self.assertRaises(ValueError):
            self.adapter.prepare(self.messages + [{"role": "assistant", "content": "FUTURE_SENTINEL"}])

    def test_metadata_role_fields_and_nonuniform_system_are_rejected(self):
        cases = []
        bad = deepcopy(self.messages); bad[-1]["profile_id"] = "p01"; cases.append(bad)
        bad = deepcopy(self.messages); bad[-1]["role"] = "metadata"; cases.append(bad)
        bad = deepcopy(self.messages); bad[0]["content"] += " profile p01"; cases.append(bad)
        bad = deepcopy(self.messages); bad[-1]["content"] = {"regime": "retain_priority"}; cases.append(bad)
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                self.adapter.prepare(case)

    def test_context_limit_reserves_answer_and_fails_without_truncation(self):
        config = deepcopy(self.config)
        config["model"]["max_context_tokens"] = self.measurement.record["input_tokens"] + 1
        adapter = ModelAdapter(self.model, self.tokenizer, config, tiny_fixture=True)
        with self.assertRaisesRegex(ValueError, "no truncation"):
            adapter.prepare(self.messages)

    def make_store(self, directory, *, turns=1):
        spec = replace(smoke_sessions(self.config)[0], session_id="smoke-storage-fixture", turns=turns)
        binding = {"state_shape": [3, 16], "model_revision": self.config["model"]["revision"],
                   "config_sha256": "tiny-fixture-binding", "data_kind": "synthetic_fixture",
                   "eligible_for_scientific_analysis": False,
                   "max_new_tokens": self.config["model"]["generation"]["max_new_tokens"],
                   "eos_token_ids": self.config["model"]["generation"]["eos_token_id"]}
        return SessionStore(directory, spec, binding)

    def test_saved_visible_data_and_hidden_audit_stay_separate_and_complete_turn_is_not_duplicated(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.make_store(directory)
            store.append(self.measurement, self.messages[-1]["content"], {"profile_id": "p07", "action": "test_only"})
            store.validate()
            visible = json.loads((store.path / "turn-01" / "visible.json").read_text())
            self.assertNotIn("profile_id", visible)
            self.assertNotIn("action", visible)
            self.assertEqual(store.manifest["completed_turns"], 1)
            with self.assertRaises(ValueError):
                store.append(self.measurement, "duplicate", {})
            resumed = SessionStore(directory, store.spec, store.binding, resume=True)
            self.assertEqual(len(resumed.records()), 1)
            record = resumed.analysis_record(2)
            self.assertEqual(record["states"].shape, (1, 16))
            self.assertFalse(record["eligible_for_scientific_analysis"])
            self.assertEqual(record["data_kind"], "synthetic_fixture")

    def test_resume_rejects_binding_mismatch_and_checkpoint_checksum_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.make_store(directory)
            store.append(self.measurement, self.messages[-1]["content"], {})
            changed = {**store.binding, "config_sha256": "different-config"}
            with self.assertRaisesRegex(ValueError, "binding mismatch"):
                SessionStore(directory, store.spec, changed, resume=True)
            with (store.path / "turn-01" / "states.npy").open("ab") as handle:
                handle.write(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                SessionStore(directory, store.spec, store.binding, resume=True)

    def test_resume_rejects_miscount_incomplete_directory_and_label_changes(self):
        for corruption in ("count", "pending", "labels"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as directory:
                store = self.make_store(directory, turns=2)
                store.append(self.measurement, self.messages[-1]["content"], {})
                if corruption == "count":
                    store.manifest["completed_turns"] = 2
                    atomic_json(store.path / "manifest.json", store.manifest)
                elif corruption == "pending":
                    (store.path / "pending-uncommitted").mkdir()
                else:
                    labels = json.loads((store.path / "labels.json").read_text())
                    labels["profile_id"] = "p01"
                    atomic_json(store.path / "labels.json", labels)
                with self.assertRaises(ValueError):
                    SessionStore(directory, store.spec, store.binding, resume=True)

    def test_resume_rejects_correctly_hashed_but_inconsistent_generation_records(self):
        # Integrity hashes alone do not show that writer metadata was correct.
        # append computes valid hashes; resume must still check semantic invariants.
        cases = {
            "output_count": {"output_tokens": 128},
            "prefix_position": {"generation_prefix_position": 0},
            "prefix_token": {"generation_prefix_token_id": 0},
            "user_boundary": {"user_message_end_position": 0},
            "stop_reason": {"stop_reason": "unrecognized_stop"},
            "output_limit": {"output_token_ids": [1, 2, 3], "output_tokens": 3},
            "negative_id": {"output_token_ids": [-1], "output_tokens": 1},
            "empty_output": {"output_token_ids": [], "output_tokens": 0},
        }
        for name, changes in cases.items():
            with self.subTest(corruption=name), tempfile.TemporaryDirectory() as directory:
                store = self.make_store(directory)
                record = {**deepcopy(self.measurement.record), **changes}
                store.append(TurnMeasurement(record, self.measurement.states.copy()), "synthetic fixture", {})
                with self.assertRaises(ValueError):
                    SessionStore(directory, store.spec, store.binding, resume=True)

    def test_collection_24_turns_session_reset_and_completed_resume_do_not_regenerate(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            first, second = smoke_sessions(self.config)
            run = {"run_id": "unit-tiny-only", "code_reference": "synthetic-test-code"}
            full = collect_session(self.config, first, self.adapter, directory, run)
            reset = collect_session(self.config, second, self.adapter, directory, run)
            self.assertEqual(full.manifest["completed_turns"], 24)
            self.assertEqual(reset.manifest["completed_turns"], 2)
            self.assertEqual(full.records()[0]["input_token_ids"], reset.records()[0]["input_token_ids"])
            self.assertEqual(full.records()[0]["output_token_ids"], reset.records()[0]["output_token_ids"])
            np.testing.assert_array_equal(np.load(full.path / "turn-01" / "states.npy"),
                                          np.load(reset.path / "turn-01" / "states.npy"))
            self.assertGreater(full.records()[-1]["input_tokens"], full.records()[0]["input_tokens"])
            self.assertNotIn("smoke-full-24", reset.records()[0]["context"])
            hashes = deepcopy(full.manifest["turn_hashes"])
            resumed = collect_session(self.config, first, self.adapter, directory, run, resume=True)
            self.assertEqual(resumed.manifest["turn_hashes"], hashes)
            self.assertEqual(resumed.analysis_record(2)["states"].shape, (24, 16))

    def test_collection_failure_is_explicit_and_no_extracted_zero_turn_is_committed(self):
        class RejectBeforeForward:
            def start_session(self, *_):
                pass
            def reserve_turn(self, *_):
                raise RuntimeError("synthetic-token-budget-exhausted")
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            spec = smoke_sessions(self.config)[1]
            with self.assertRaisesRegex(RuntimeError, "synthetic-token-budget-exhausted"):
                collect_session(self.config, spec, self.adapter, directory, {"run_id": "failure-fixture"}, guard=RejectBeforeForward())
            manifest = json.loads((Path(directory) / spec.session_id / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["completed_turns"], 0)
            self.assertEqual(manifest["turn_hashes"], [])
            self.assertEqual(manifest["errors"][0]["type"], "RuntimeError")


if __name__ == "__main__":
    unittest.main()
