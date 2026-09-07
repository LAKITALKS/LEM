"""Offline launch-boundary checks. Every Modal network entry is intercepted."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from lem_ii import launch_smoke
from lem_ii.budget import BudgetError
from lem_ii.design import load_config
from lem_ii.storage import canonical_hash, file_hash


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "preflight.json"
        self.proof = {
            "verified_at_utc": datetime.now(timezone.utc).isoformat(),
            "workspace": "workspace-fixture", "environment": "main",
            "workspace_gross_usage_limit_usd": 5,
            "workspace_usage_before_usd": 0,
            "rates": {"L4": .000222, "CPU_core": .0000131, "RAM_GiB": .00000222},
        }

    def validate(self, change=None, *, profile="workspace-fixture", apps=None):
        proof = deepcopy(self.proof)
        if change:
            proof.update(change)
        self.path.write_text(json.dumps(proof))
        def response(*args):
            if args[1:3] == ("profile", "current"):
                return profile
            if args[1:3] == ("app", "list"):
                return json.dumps(apps or [])
            raise AssertionError(f"Unexpected command in offline fixture: {args}")
        with patch.object(launch_smoke, "command", side_effect=response):
            return launch_smoke.validate_preflight(self.path)

    def test_valid_current_gross_usage_evidence_accepts_without_launch(self):
        self.assertEqual(self.validate()["workspace_gross_usage_limit_usd"], 5)
        self.assertEqual(self.validate({"workspace": "alternate-workspace-fixture"},
                                       profile="alternate-workspace-fixture")["workspace"],
                         "alternate-workspace-fixture")

    def test_stale_wrong_scope_rate_usage_and_profile_reject_before_launch(self):
        for change in [
            {"verified_at_utc": "2020-01-01T00:00:00+00:00"},
            {"verified_at_utc": "2099-01-01T00:00:00+00:00"},
            {"workspace": "other"}, {"environment": "other"},
            {"workspace_gross_usage_limit_usd": 30},
            {"workspace_usage_before_usd": 1.01},
            {"rates": {"L4": .1, "CPU_core": .0000131, "RAM_GiB": .00000222}},
        ]:
            with self.subTest(change=change), self.assertRaises((RuntimeError, ValueError)):
                self.validate(change)
        with self.assertRaises(RuntimeError):
            self.validate(profile="wrong-profile")
        with self.assertRaises(RuntimeError):
            self.validate(apps=[{"App ID": "ap-active", "Description": "unrelated-work", "State": "ephemeral", "Tasks": "1"}])

    def test_nonfinite_or_negative_usage_cannot_pass_comparison_with_cap(self):
        for value in [-1, float("nan"), float("inf"), float("-inf")]:
            with self.subTest(value=value), self.assertRaises((RuntimeError, ValueError)):
                self.validate({"workspace_usage_before_usd": value})

    def test_verified_stopped_own_app_does_not_preclude_authorized_second_attempt(self):
        for row in ({"App ID": "ap-owned", "State": "stopped", "Tasks": "0"},
                    {"app_id": "ap-owned", "state": "stopped", "tasks": "0"}):
            result = self.validate(apps=[row])
            self.assertEqual(result["workspace"], "workspace-fixture")
        with self.assertRaises(RuntimeError):
            self.validate(apps=[{"app_id": "ap-owned", "state": "ephemeral", "tasks": "1"}])


class StopOwnershipTests(unittest.TestCase):
    def test_already_stopped_exit_one_is_idempotent_and_scoped_to_exact_app(self):
        result = subprocess.CompletedProcess([], 1, stdout="", stderr="App is already stopped.\n")
        rows = [{"App ID": "ap-owned", "State": "stopped", "Tasks": "0"},
                {"App ID": "ap-other", "State": "ephemeral", "Tasks": "1"}]
        with patch.object(launch_smoke.subprocess, "run", return_value=result) as stop, \
                patch.object(launch_smoke, "command", return_value=json.dumps(rows)):
            response = launch_smoke.stop_owned_app("ap-owned")
        self.assertEqual(response["app_id"], "ap-owned")
        self.assertEqual(stop.call_args.args[0][1:4], ["app", "stop", "ap-owned"])
        self.assertNotIn("ap-other", stop.call_args.args[0])

    def test_other_stop_failure_is_not_treated_as_terminal_success(self):
        result = subprocess.CompletedProcess([], 1, stdout="", stderr="fixture authentication error")
        with patch.object(launch_smoke.subprocess, "run", return_value=result), \
                patch.object(launch_smoke, "command") as inspect:
            with self.assertRaises(RuntimeError):
                launch_smoke.stop_owned_app("ap-owned")
        inspect.assert_not_called()

    def test_stop_ack_waits_for_zero_tasks_and_rejects_unverified_termination(self):
        result = subprocess.CompletedProcess([], 0, stdout="", stderr="")
        running = json.dumps([{"App ID": "ap-owned", "State": "stopping...", "Tasks": "1"}])
        stopped = json.dumps([{"App ID": "ap-owned", "State": "stopped", "Tasks": "0"}])
        with patch.object(launch_smoke.subprocess, "run", return_value=result), \
                patch.object(launch_smoke.time, "sleep"), \
                patch.object(launch_smoke, "command", side_effect=[running, stopped]) as inspect:
            launch_smoke.stop_owned_app("ap-owned")
        self.assertEqual(inspect.call_count, 2)
        for unverified in [running, "[]"]:
            with self.subTest(state=unverified), patch.object(launch_smoke.subprocess, "run", return_value=result), \
                    patch.object(launch_smoke.time, "sleep"), \
                    patch.object(launch_smoke, "command", return_value=unverified):
                with self.assertRaises(RuntimeError):
                    launch_smoke.stop_owned_app("ap-owned")


class SourceBindingTests(unittest.TestCase):
    def test_only_committed_current_executable_inputs_can_be_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "lem_ii/config").mkdir(parents=True)
            for name, contents in [("runner.py", "# fixture executable\n"),
                                   ("requirements.txt", "numpy==2.4.6\n"),
                                   ("config/study.json", '{"fixture": true}\n')]:
                (root / "lem_ii" / name).write_text(contents)
            def git(*args):
                if args[1] == "rev-parse":
                    return "f" * 40
                return ""
            with patch.object(launch_smoke, "ROOT", root), patch.object(launch_smoke, "command", side_effect=git):
                bound = launch_smoke.source_binding()
            self.assertEqual(bound["commit"], "f" * 40)
            self.assertEqual(set(bound["files"]),
                             {"lem_ii/runner.py", "lem_ii/requirements.txt", "lem_ii/config/study.json"})
            for name, checksum in bound["files"].items():
                self.assertEqual(checksum, file_hash(root / name))
            def dirty(*args):
                return " M lem_ii/runner.py" if args[1] == "status" else ""
            with patch.object(launch_smoke, "ROOT", root), patch.object(launch_smoke, "command", side_effect=dirty):
                with self.assertRaises(RuntimeError):
                    launch_smoke.source_binding()
            def untracked(*args):
                if args[1] == "ls-files":
                    raise subprocess.CalledProcessError(1, args)
                return ""
            with patch.object(launch_smoke, "ROOT", root), patch.object(launch_smoke, "command", side_effect=untracked):
                with self.assertRaises(subprocess.CalledProcessError):
                    launch_smoke.source_binding()


class RemoteEntryBoundaryTests(unittest.TestCase):
    def setUp(self):
        # Importing the declarative app constructs local objects; .remote/.run are never called.
        from lem_ii.modal_entry import smoke_worker
        self.worker = smoke_worker.local
        self.config = load_config()
        self.attempt = {"attempt_id": "offline-fixture", "deadline_epoch": time.time() + 100,
                        "config_hash": canonical_hash(self.config)}
        self.source = {"files": {}}

    def test_expired_or_mismatched_attempt_rejects_before_cloud_claim_or_model_load(self):
        cases = [(self.config, {**self.attempt, "deadline_epoch": time.time() - 1}, BudgetError),
                 (self.config, {**self.attempt, "config_hash": "0" * 64}, ValueError)]
        forbidden = deepcopy(self.config)
        forbidden["execution"]["confirmation_generation_allowed"] = True
        cases.append((forbidden, {**self.attempt, "config_hash": canonical_hash(forbidden)}, PermissionError))
        for config, attempt, error in cases:
            with self.subTest(error=error), patch("modal.Dict.from_name") as claim, \
                    patch("lem_ii.model_adapter.ModelAdapter.load") as load:
                with self.assertRaises(error):
                    self.worker(config, attempt, self.source)
                claim.assert_not_called()
                load.assert_not_called()

    def test_duplicate_preemption_claim_rejects_before_model_loading(self):
        claims = Mock()
        claims.put.return_value = False
        with patch("modal.Dict.from_name", return_value=claims), patch("lem_ii.model_adapter.ModelAdapter.load") as load:
            with self.assertRaisesRegex(RuntimeError, "already consumed"):
                self.worker(self.config, self.attempt, self.source)
            load.assert_not_called()
        self.assertTrue(claims.put.call_args.kwargs["skip_if_exists"])
        self.assertEqual(claims.put.call_args.args[0], self.attempt["attempt_id"])


if __name__ == "__main__":
    unittest.main()
