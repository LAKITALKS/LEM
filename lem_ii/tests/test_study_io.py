"""Future study gate/IO tests; no study model or confirmation dialogue is made.

Metadata-only development/validation inventories mock already validated stores.
Actual state/checksum validation has separate tiny-random-model production tests.
"""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from lem_ii.design import build_sessions, load_config
from lem_ii.model_adapter import TEMPLATE_SHA256
from lem_ii.storage import atomic_json, canonical_hash, file_hash
from lem_ii import study


LIMITS = {"max_seconds": 100, "max_sessions": 1, "max_prefill_tokens": 20000,
          "max_generation_tokens": 3072, "gross_cost_cap_usd": 1.0,
          "inclusive_resource_usd_per_second": 0.0003, "noncompute_reserve_usd": 0.1,
          "prior_technical_cost_usd": 0.0, "external_hard_timeout_seconds": 160}


class StudyIOTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()

    def enabled_analysis(self):
        config = deepcopy(self.config)
        config["execution"]["allow_scientific_analysis"] = True
        return config

    def metadata_inventory(self, root, config):
        """Only metadata headers for 72 dev/val sessions; NO dialogue/state files."""
        collection_config = deepcopy(config)
        collection_config["execution"].update(study_collection_allowed=True, smoke_only=False,
                                               allow_scientific_analysis=False)
        specs = [s for s in build_sessions(config) if s.split in ("development", "validation")]
        sources = study.source_hashes()
        binding = {
            "data_kind": "real_model_study", "eligible_for_scientific_analysis": True,
            "config_sha256": canonical_hash(collection_config), "state_shape": [3, 2048],
            "max_new_tokens": 128, "eos_token_ids": [151645, 151643],
            "adapter": {"tiny_random_fixture": False, "model_revision": config["model"]["revision"],
                        "tokenizer_revision": config["model"]["tokenizer_revision"],
                        "chat_template_sha256": TEMPLATE_SHA256, "weight_dtype": "torch.bfloat16",
                        "attention_implementation": "sdpa", "software": dict(study.PINNED_COLLECTION_SOFTWARE)},
            "run": {"schema": "lem-ii.study-run.v1", "config_snapshot": collection_config,
                    "scientific_config_sha256": study._scientific_hash(config), "source_sha256": sources,
                    "source_set_sha256": canonical_hash(sources), "planned_session_ids": [s.session_id for s in specs]},
        }
        for spec in specs:
            directory = Path(root) / spec.session_id
            directory.mkdir()
            atomic_json(directory / "manifest.json", {
                "test_metadata_only_not_collected_data": True, "spec": asdict(spec), "status": "complete",
                "completed_turns": 24, "binding": binding,
            })
        return specs

    def test_plan_is_108_hashed_metadata_assignments_without_dialogue_generation(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(study, "_load_adapter") as load:
            output = Path(directory) / "plan.json"
            result = study.plan(self.config, output)
            self.assertEqual(result["planned_sessions"], 108)
            self.assertEqual(result["planned_assistant_responses"], 2592)
            self.assertEqual(result["assignment_sha256"], canonical_hash(result["assignments"]))
            self.assertTrue(result["metadata_only"])
            self.assertEqual(result["dialogues_generated"], 0)
            self.assertFalse(any("context" in row or "answer" in row for row in result["assignments"]))
            load.assert_not_called()
            with self.assertRaises(FileExistsError):
                study.plan(self.config, output)

    def test_current_collect_gate_stops_before_download_or_output_creation(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(study, "_load_adapter") as load:
            root = Path(directory) / "not-created"
            with self.assertRaises(PermissionError):
                study.collect(root, self.config, "development", **LIMITS)
            self.assertFalse(root.exists())
            load.assert_not_called()

    def test_current_analysis_and_confirmation_gates_stop_before_input_loading(self):
        with patch.object(study, "load_completed_study_records") as load:
            with self.assertRaises(PermissionError):
                study.analyze_development_validation("not-read", self.config, "not-written")
            with self.assertRaises(PermissionError):
                study.confirm("not-read", self.config, "not-read", "not-written")
            load.assert_not_called()
        config = self.enabled_analysis()
        with patch.object(study, "load_completed_study_records") as load:
            with self.assertRaises(PermissionError):
                study.confirm("not-read", config, "not-read", "not-written")
            load.assert_not_called()

    def test_gate_only_change_preserves_scientific_hash_but_model_change_does_not(self):
        enabled = deepcopy(self.config)
        for key in ("study_collection_allowed", "confirmation_generation_allowed", "confirmation_evaluation_allowed",
                    "allow_scientific_analysis", "allow_confirmation_analysis"):
            enabled["execution"][key] = True
        enabled["execution"]["smoke_only"] = False
        self.assertEqual(study._scientific_hash(enabled), study._scientific_hash(self.config))
        enabled["model"]["generation"]["max_new_tokens"] = 127
        self.assertNotEqual(study._scientific_hash(enabled), study._scientific_hash(self.config))

    def test_loader_requires_all_72_before_opening_any_store(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(study, "SessionStore") as store:
            with self.assertRaisesRegex(ValueError, "72"):
                study.load_completed_study_records(directory, self.enabled_analysis())
            store.assert_not_called()

    def test_loader_checks_all_complete_development_stores_before_returning_records(self):
        config = self.enabled_analysis()
        with tempfile.TemporaryDirectory() as directory:
            specs = self.metadata_inventory(directory, config)
            mocks = []
            def verified_store(root, spec, binding, resume):
                self.assertTrue(resume)
                fake = MagicMock()
                # Inert return marker: no fake scientific dialogue or activation.
                fake.analysis_record.return_value = {"dialogue_id": spec.session_id, "data_kind": "real_model_study",
                                                     "eligible_for_scientific_analysis": True, "mock_only": True}
                mocks.append(fake)
                return fake
            original_hash = file_hash
            def checksum(path):
                path = Path(path)
                return original_hash(path) if path.is_file() else "mock-checksum-no-dialogue-file"
            with patch.object(study, "SessionStore", side_effect=verified_store), patch.object(study, "file_hash", side_effect=checksum), \
                    patch("lem_ii.features.validate_records") as validate:
                rows, provenance = study.load_completed_study_records(directory, config)
            self.assertEqual(len(rows), 72)
            self.assertEqual({r["dialogue_id"] for r in rows}, {s.session_id for s in specs})
            self.assertEqual(provenance["session_count"], 72)
            self.assertTrue(provenance["all_session_stores_validated"])
            self.assertEqual(provenance["input_set_sha256"], canonical_hash(provenance["input_file_sha256"]))
            for fake in mocks:
                fake.validate.assert_called_once()
                fake.analysis_record.assert_called_once_with(2)
            validate.assert_called_once()

    def test_loader_rejects_changed_science_source_or_fixture_binding_before_states(self):
        for change in ("science", "source", "fixture", "assignment", "software"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                config = self.enabled_analysis()
                specs = self.metadata_inventory(directory, config)
                path = Path(directory) / specs[0].session_id / "manifest.json"
                manifest = json.loads(path.read_text())
                if change == "science":
                    manifest["binding"]["run"]["scientific_config_sha256"] = "different"
                elif change == "source":
                    manifest["binding"]["run"]["source_sha256"]["simulator.py"] = "changed"
                elif change == "fixture":
                    manifest["binding"]["adapter"]["tiny_random_fixture"] = True
                elif change == "software":
                    manifest["binding"]["adapter"]["software"]["transformers"] = "different-version"
                else:
                    manifest["binding"]["run"]["planned_session_ids"] = []
                atomic_json(path, manifest)
                with patch.object(study, "SessionStore") as store, self.assertRaises(ValueError):
                    study.load_completed_study_records(directory, config)
                store.assert_not_called()

    def test_corrupt_real_store_validation_propagates_without_loading_analysis_vector(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.enabled_analysis()
            self.metadata_inventory(directory, config)
            failed = MagicMock()
            failed.validate.side_effect = ValueError("checkpoint checksum mismatch")
            with patch.object(study, "SessionStore", return_value=failed), self.assertRaisesRegex(ValueError, "checksum"):
                study.load_completed_study_records(directory, config)
            failed.analysis_record.assert_not_called()

    def test_enabled_collection_rejects_missing_limits_or_uncovered_cost_before_load(self):
        config = deepcopy(self.config)
        config["execution"].update(study_collection_allowed=True, smoke_only=False)
        for changes in ({"max_seconds": None}, {"inclusive_resource_usd_per_second": 1.0}, {"max_sessions": 37},
                        {"external_hard_timeout_seconds": None}, {"external_hard_timeout_seconds": 161}):
            with patch.object(study, "_load_adapter") as load, self.assertRaises(ValueError):
                study.collect("not-created", config, "development", **{**LIMITS, **changes})
            load.assert_not_called()

    def test_future_guard_accounts_two_prefills_pending_failures_deadline_and_session_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            limits = {"deadline_epoch": 100, "max_sessions": 1, "max_prefill_tokens": 100,
                      "max_generation_tokens": 4}
            guard = study.StudyRuntimeGuard(limits, Path(directory) / "runtime.json", clock=lambda: 1)
            guard.start_session("study-test-metadata", 24)
            guard.reserve_turn(20, 2)
            with self.assertRaises(RuntimeError):
                guard.reserve_turn(20, 2)
            guard.finish_turn(1)
            with self.assertRaises(RuntimeError):
                guard.start_session("study-test-another", 24)
            guard.reserve_turn(20, 2)
            guard.finish_turn(2)
            with self.assertRaises(RuntimeError):
                guard.reserve_turn(20, 1)
            self.assertEqual(sum(t["input_tokens"] * 2 for t in guard.state["turns"]), 80)
            self.assertEqual(sum(t["max_new_tokens"] for t in guard.state["turns"]), 4)
            guard.clock = lambda: 101
            with self.assertRaises(RuntimeError):
                guard.check_deadline()
            existing = (Path(directory) / "runtime.json").read_bytes()
            with self.assertRaises(FileExistsError):
                study.StudyRuntimeGuard(limits, Path(directory) / "runtime.json", clock=lambda: 1)
            self.assertEqual((Path(directory) / "runtime.json").read_bytes(), existing)

    def test_source_reference_rejects_untracked_or_dirty_sources_before_claiming_commit(self):
        with patch.object(study.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "git ls-files")):
            with self.assertRaisesRegex(ValueError, "tracked"):
                study._code_reference()
        tracked = subprocess.CompletedProcess([], 0, stdout="tracked sources", stderr="")
        dirty = subprocess.CompletedProcess([], 0, stdout=" M study.py\n", stderr="")
        with patch.object(study.subprocess, "run", side_effect=[tracked, dirty]) as run:
            with self.assertRaisesRegex(ValueError, "uncommitted"):
                study._code_reference()
            self.assertEqual(run.call_count, 2)
        clean = subprocess.CompletedProcess([], 0, stdout="", stderr="")
        head = subprocess.CompletedProcess([], 0, stdout="test-commit\n", stderr="")
        with patch.object(study.subprocess, "run", side_effect=[tracked, clean, head]):
            self.assertEqual(study._code_reference(), "test-commit")

    def test_failed_future_load_keeps_full_reservation_and_next_attempt_cannot_reset_budget(self):
        config = deepcopy(self.config)
        config["execution"].update(study_collection_allowed=True, smoke_only=False)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            ledger = Path(directory) / "shared-study-ledger.json"
            with patch.object(study, "STUDY_LEDGER", ledger), patch.object(study, "_code_reference", return_value="test-no-model-run"), \
                    patch.object(study, "_load_adapter", side_effect=RuntimeError("invented-load-failure")) as load:
                with self.assertRaisesRegex(RuntimeError, "invented-load-failure"):
                    study.collect(root, config, "development", **LIMITS)
                saved = json.loads(ledger.read_text())
                self.assertEqual(saved["attempts"][0]["status"], "failed")
                self.assertEqual(saved["attempts"][0]["reserved_usd"], 1.0)
                self.assertEqual(len(list(root.glob("study-p*"))), 0)
                with self.assertRaisesRegex(ValueError, "allowance exhausted"):
                    study.collect(root, config, "development", **{**LIMITS, "gross_cost_cap_usd": 25.0})
                load.assert_called_once()
                self.assertEqual(json.loads(ledger.read_text()), saved)


if __name__ == "__main__":
    unittest.main()
