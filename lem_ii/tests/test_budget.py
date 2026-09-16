"""Technical fixtures exercise shared accounting before any real cloud work."""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from lem_ii.budget import (BudgetError, BudgetLedger, RuntimeGuard, DEADLINE_SECONDS,
                           TASK_ID, cost_proof)

HASH = "a" * 64


def _reserve_in_process(path):
    try:
        return BudgetLedger(path).reserve(HASH)["attempt_id"]
    except BudgetError:
        return None


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "task-budget.json"
        self.ledger = BudgetLedger(self.path, clock=lambda: 10000.0)

    def test_cost_proof_accounts_for_load_idle_build_and_uncertainty(self):
        proof = cost_proof()
        self.assertAlmostEqual(proof["resource_rate_usd_per_second"], .00028372)
        self.assertEqual(proof["resource_seconds"], 2702)
        self.assertAlmostEqual(proof["total_envelope_usd"], 1.76661144)
        self.assertLessEqual(proof["total_envelope_usd"], proof["reservation_usd"])
        with patch("lem_ii.budget.ATTEMPT_RESERVATION_USD", 1.0):
            with self.assertRaises(BudgetError):
                self.ledger.reserve(HASH)
        self.assertFalse(self.path.exists())

    def test_failures_and_cancellations_permanently_consume_two_attempts(self):
        first = self.ledger.reserve(HASH)
        self.assertEqual(first["deadline_epoch"], 10000 + DEADLINE_SECONDS)
        self.ledger.finish(first["attempt_id"], "failed", {"stage": "model_load"})
        # A fresh controller process does not grant a fresh task budget.
        second = BudgetLedger(self.path).reserve(HASH)
        self.ledger.finish(second["attempt_id"], "cancelled")
        self.assertEqual(self.ledger.snapshot()["reserved_total_usd"], 4.0)
        self.assertEqual(self.ledger.snapshot()["unreserved_task_usd"], 1.0)
        with self.assertRaises(BudgetError):
            BudgetLedger(self.path).reserve(HASH)
        self.assertEqual(len(self.ledger.snapshot()["attempts"]), 2)

    def test_duplicate_active_config_and_task_mismatch_fail_without_changes(self):
        first = self.ledger.reserve(HASH, attempt_id="exact-start")
        old = self.path.read_bytes()
        for ledger, digest, attempt_id in [
            (self.ledger, HASH, "exact-start"),
            (self.ledger, HASH, "second-active"),
            (self.ledger, "b" * 64, "different-config"),
            (BudgetLedger(self.path, task_id="different-task"), HASH, "different-task"),
        ]:
            with self.subTest(attempt=attempt_id), self.assertRaises(BudgetError):
                ledger.reserve(digest, attempt_id=attempt_id)
            self.assertEqual(self.path.read_bytes(), old)
        self.ledger.finish(first["attempt_id"], "failed")
        with self.assertRaises(BudgetError):
            self.ledger.reserve("b" * 64)
        with self.assertRaises(BudgetError):
            self.ledger.finish(first["attempt_id"], "completed")

    def test_expiry_does_not_silently_clear_active_or_refund(self):
        self.ledger.reserve(HASH)
        expired = BudgetLedger(self.path, clock=lambda: 1e9)
        with self.assertRaises(BudgetError):
            expired.reserve(HASH)
        self.assertEqual(expired.snapshot()["reserved_total_usd"], 2)

    def test_invalid_ledger_is_never_reset(self):
        for content in ["{incomplete", "[]", json.dumps({"schema_version": 1})]:
            self.path.write_text(content)
            with self.assertRaises(BudgetError):
                self.ledger.reserve(HASH)
            self.assertEqual(self.path.read_text(), content)

    def test_actual_separate_process_launch_race_allows_only_one(self):
        with ProcessPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(_reserve_in_process, [str(self.path)] * 2))
        self.assertEqual(sum(outcome is not None for outcome in outcomes), 1)
        self.assertEqual(self.ledger.snapshot()["reserved_total_usd"], 2)
        self.assertEqual(json.loads(self.path.read_text())["task_id"], TASK_ID)

    def test_atomic_replace_failure_preserves_previous_ledger(self):
        first = self.ledger.reserve(HASH)
        before = self.path.read_bytes()
        with patch("lem_ii.budget.os.replace", side_effect=OSError("fixture disk error")):
            with self.assertRaises(OSError):
                self.ledger.finish(first["attempt_id"], "failed")
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(sorted(p.name for p in self.path.parent.iterdir()),
                         ["task-budget.json", "task-budget.json.lock"])


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.now = 10000.0
        self.guard = RuntimeGuard(self.now + DEADLINE_SECONDS, clock=lambda: self.now)

    def turn(self, input_tokens=100, output_tokens=1):
        self.guard.reserve_turn(input_tokens, 128)
        self.guard.finish_turn(output_tokens)

    def test_full_24_and_separate_2_turn_session_reach_exact_scope(self):
        self.guard.start_session("smoke-long", 24)
        for _ in range(24):
            self.turn()
        self.guard.start_session("smoke-reset", 2)
        for _ in range(2):
            self.turn()
        snapshot = self.guard.snapshot()
        self.assertEqual(snapshot["generation_tokens_reserved"], 3328)
        self.assertEqual(snapshot["prefill_tokens_reserved"], 5200)
        self.assertEqual(snapshot["actual_output_tokens"], 26)
        with self.assertRaises(BudgetError):
            self.turn()
        with self.assertRaises(BudgetError):
            self.guard.start_session("third-session", 1)

    def test_failures_keep_pending_reservations_and_cannot_be_retried(self):
        self.guard.start_session("smoke-long", 24)
        self.guard.reserve_turn(250, 128)
        before = self.guard.snapshot()
        # An extraction failure leaves a pending turn and consumes its allowance.
        restored = RuntimeGuard(self.guard.deadline_epoch, state=before, clock=lambda: self.now)
        with self.assertRaises(BudgetError):
            restored.reserve_turn(250, 128)
        with self.assertRaises(BudgetError):
            restored.start_session("second", 2)
        self.assertEqual(restored.snapshot(), before)
        with self.assertRaises(BudgetError):
            restored.finish_turn(129)
        self.assertEqual(restored.snapshot(), before)

    def test_input_generation_and_aggregate_prefill_limits_before_operation(self):
        self.guard.start_session("smoke-long", 24)
        for invalid_input, invalid_output in [(8192, 128), (8065, 128), (100, 129),
                                              (0, 128), (100, 0), (True, 128)]:
            with self.subTest(input=invalid_input, output=invalid_output), self.assertRaises(BudgetError):
                self.guard.reserve_turn(invalid_input, invalid_output)
        self.assertFalse(self.guard.turns)
        for _ in range(24):
            self.turn(input_tokens=8064)
        self.guard.start_session("smoke-reset", 2)
        self.guard.reserve_turn(6464, 128)  # Exactly 400,000, including TWO prefills.
        self.guard.finish_turn(1)
        self.assertEqual(self.guard.prefill_tokens_reserved, 400000)
        before = self.guard.snapshot()
        with self.assertRaises(BudgetError):
            self.guard.reserve_turn(1, 128)
        self.assertEqual(self.guard.snapshot(), before)

    def test_aggregate_generation_guard_independently_rejects_before_operation(self):
        self.guard.start_session("fixture", 2)
        self.turn()
        before = self.guard.snapshot()
        with patch("lem_ii.budget.MAX_GENERATION_TOKENS", 128):
            with self.assertRaises(BudgetError):
                self.guard.reserve_turn(100, 128)
        self.assertEqual(self.guard.snapshot(), before)

    def test_deadline_covers_initialization_and_each_operation(self):
        self.guard.start_session("fixture", 2)
        self.guard.reserve_turn(100, 128)
        self.now += DEADLINE_SECONDS
        for operation in [self.guard.check_deadline, lambda: self.guard.finish_turn(1),
                          lambda: self.guard.reserve_turn(100, 128),
                          lambda: self.guard.start_session("later", 1)]:
            with self.assertRaises(BudgetError):
                operation()
        restored = RuntimeGuard(self.guard.deadline_epoch, self.guard.snapshot(), clock=lambda: self.now)
        with self.assertRaises(BudgetError):
            restored.check_deadline()

    def test_sessions_cannot_be_duplicate_or_silently_abandoned(self):
        for invalid in (25, 0, -1, True):
            with self.assertRaises(BudgetError):
                self.guard.start_session("bad", invalid)
        self.guard.start_session("one", 24)
        with self.assertRaises(BudgetError):
            self.guard.start_session("one", 2)
        with self.assertRaises(BudgetError):
            self.guard.start_session("other", 2)
        for _ in range(24):
            self.turn()
        with self.assertRaises(BudgetError):
            self.guard.start_session("too-many", 3)
        self.guard.start_session("two", 2)
        with self.assertRaises(BudgetError):
            self.guard.finish_turn(0)

    def test_resume_validates_deadline_counters_turn_order_and_pending_status(self):
        self.guard.start_session("one", 24)
        self.turn()
        snapshot = self.guard.snapshot()
        self.assertEqual(RuntimeGuard(self.guard.deadline_epoch, snapshot).snapshot(), snapshot)
        for change in (lambda s: s.update(deadline_epoch=s["deadline_epoch"] + 2700),
                       lambda s: s.update(generation_tokens_reserved=0),
                       lambda s: s["turns"][0].update(turn=2),
                       lambda s: s["turns"][0].update(actual_output_tokens=129),
                       lambda s: s["turns"][0].update(session_id="unknown")):
            corrupted = deepcopy(snapshot)
            change(corrupted)
            with self.assertRaises(BudgetError):
                RuntimeGuard(self.guard.deadline_epoch, corrupted)


if __name__ == "__main__":
    unittest.main()
