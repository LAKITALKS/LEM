"""Fail-closed smoke limits; all figures are gross usage, never credits/net spend.

The local ledger is shared by *every* launch of this task. A $2 reservation is
permanently consumed even when a build, invocation, or extraction fails. It is
not provider billing, and its file must not be reset to obtain another attempt.
RuntimeGuard is also used in the remote production path before expensive work.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from decimal import Decimal
import fcntl
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Callable
from uuid import uuid4

TASK_ID = "lem-ii-light-technical-2026-09-07"
TASK_LIMIT_USD = 5.0
ATTEMPT_RESERVATION_USD = 2.0
MAX_ATTEMPTS = 2
MAX_PARALLEL = 1
STARTUP_SECONDS = 900
EXECUTION_SECONDS = 1800
IDLE_SECONDS = 2
DEADLINE_SECONDS = STARTUP_SECONDS + EXECUTION_SECONDS
MAX_SESSIONS = 2
MAX_TURNS = 26
MAX_SESSION_TURNS = 24
MAX_CONTEXT_TOKENS = 8192
MAX_OUTPUT_TOKENS = 128
MAX_GENERATION_TOKENS = 3328
MAX_PREFILL_TOKENS = 400000
PREFILL_PASSES = 2
TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled", "aborted", "blocked"})


class BudgetError(RuntimeError):
    """A launch or operation would violate a prespecified technical limit."""


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise BudgetError(f"{name} must be an integer >= {minimum}")
    return value


def _timestamp(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise BudgetError(f"{name} must be a finite positive timestamp")
    return float(value)


def cost_proof() -> dict[str, Any]:
    """Conservative prespecified charge envelope; not a billing guarantee.

    CPU and RAM MUST be hard-capped in the Modal decorator at 2 physical cores
    and 16 GiB. One L4, no region premium, no GPU fallback, no automatic retries.
    $0.50 separately covers CPU image construction. Another $0.50 is retained
    for price/termination uncertainty and small associated storage charges.
    """
    gpu, cpu, memory = map(Decimal, ("0.000222", "0.0000131", "0.00000222"))
    rate = gpu + 2 * cpu + 16 * memory
    resource_seconds = STARTUP_SECONDS + EXECUTION_SECONDS + IDLE_SECONDS
    compute = rate * resource_seconds
    total = compute + Decimal("0.50") + Decimal("0.50")
    if total > Decimal(str(ATTEMPT_RESERVATION_USD)):
        raise BudgetError("Prespecified costs exceed the permanent attempt reservation")
    if MAX_ATTEMPTS * ATTEMPT_RESERVATION_USD > TASK_LIMIT_USD:
        raise BudgetError("Attempt reservations exceed the task usage cap")
    return {
        "currency": "USD", "gross_usage_before_credits": True,
        "gpu": "L4", "gpu_count": 1, "cpu_limit_physical_cores": 2,
        "memory_limit_gib": 16, "gpu_usd_per_second": float(gpu),
        "cpu_usd_per_core_second": float(cpu), "memory_usd_per_gib_second": float(memory),
        "resource_rate_usd_per_second": float(rate),
        "startup_seconds": STARTUP_SECONDS, "execution_seconds": EXECUTION_SECONDS,
        "idle_seconds": IDLE_SECONDS, "resource_seconds": resource_seconds,
        "compute_envelope_usd": float(compute), "cpu_build_reserve_usd": 0.5,
        "price_termination_storage_reserve_usd": 0.5,
        "total_envelope_usd": float(total), "reservation_usd": ATTEMPT_RESERVATION_USD,
        "provider_bill_guaranteed": False,
        "pricing_source": "https://modal.com/pricing", "rates_verified_on": "2026-09-07",
    }


class BudgetLedger:
    """Atomic persistent reservations guarded by a separate stable flock file.

    All controllers must use the SAME path. A terminal status never refunds a
    reservation. Expired active entries also remain active until the controller
    establishes that the remote app stopped and explicitly records its finish.
    A supplied attempt_id makes an accidental repeated launch detectable.
    """

    def __init__(self, path: str | Path, task_id: str = TASK_ID,
                 clock: Callable[[], float] = time.time):
        self.path = Path(path).resolve()
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        if not isinstance(task_id, str) or not task_id.strip():
            raise BudgetError("A stable nonempty task ID is required")
        self.task_id, self.clock = task_id, clock

    @contextmanager
    def _locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schema_version": 1, "task_id": self.task_id,
                    "task_limit_usd": TASK_LIMIT_USD, "config_hash": None, "attempts": []}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if (data["schema_version"] != 1 or data["task_id"] != self.task_id
                    or data["task_limit_usd"] != TASK_LIMIT_USD):
                raise BudgetError("Ledger task/schema/limit mismatch; do not reset this ledger")
            attempts = data["attempts"]
            if not isinstance(attempts, list) or len(attempts) > MAX_ATTEMPTS:
                raise BudgetError("Malformed or over-limit attempt ledger")
            if attempts and not re.fullmatch(r"[0-9a-f]{64}", data["config_hash"] or ""):
                raise BudgetError("Invalid ledger configuration hash")
            seen = set()
            for attempt in attempts:
                if (not isinstance(attempt["attempt_id"], str) or not attempt["attempt_id"]
                        or attempt["attempt_id"] in seen
                        or attempt["reserved_usd"] != ATTEMPT_RESERVATION_USD
                        or attempt["config_hash"] != data["config_hash"]
                        or attempt["status"] not in TERMINAL_STATUSES | {"active"}):
                    raise BudgetError("Invalid or duplicate persisted attempt")
                seen.add(attempt["attempt_id"])
                start = _timestamp(attempt["started_epoch"], "started_epoch")
                deadline = _timestamp(attempt["deadline_epoch"], "deadline_epoch")
                if deadline != start + DEADLINE_SECONDS:
                    raise BudgetError("Persisted deadline does not match the fixed wall-time cap")
            if sum(a["status"] == "active" for a in attempts) > MAX_PARALLEL:
                raise BudgetError("Persisted parallelism exceeds the task cap")
            return data
        except (OSError, ValueError, TypeError, KeyError) as error:
            raise BudgetError("Invalid budget ledger; refused to replace or reset it") from error

    def _write(self, data: dict[str, Any]) -> None:
        encoded = json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n"
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent,
                                             prefix=self.path.name + ".", delete=False) as stream:
                temporary = stream.name
                os.chmod(temporary, 0o600)
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            temporary = None
            directory = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if temporary is not None:
                os.unlink(temporary)

    def snapshot(self) -> dict[str, Any]:
        with self._locked():
            data = self._read()
            data["reserved_total_usd"] = sum(a["reserved_usd"] for a in data["attempts"])
            data["unreserved_task_usd"] = TASK_LIMIT_USD - data["reserved_total_usd"]
            return deepcopy(data)

    def reserve(self, config_hash: str, attempt_id: str | None = None) -> dict[str, Any]:
        proof = cost_proof()
        if not isinstance(config_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", config_hash):
            raise BudgetError("A complete lowercase SHA256 configuration hash is required")
        if attempt_id is not None and (not isinstance(attempt_id, str) or not attempt_id.strip()):
            raise BudgetError("Invalid attempt ID")
        with self._locked():
            data = self._read()
            if data["config_hash"] not in (None, config_hash):
                raise BudgetError("Configuration mismatch; a new file must not bypass the task budget")
            attempts = data["attempts"]
            if any(a["attempt_id"] == attempt_id for a in attempts):
                raise BudgetError("Attempt already reserved; refused duplicate launch")
            if any(a["status"] == "active" for a in attempts):
                raise BudgetError("Another task attempt is active; parallel launches are prohibited")
            if len(attempts) >= MAX_ATTEMPTS:
                raise BudgetError("The two-attempt task allowance is exhausted, including failed attempts")
            if sum(a["reserved_usd"] for a in attempts) + ATTEMPT_RESERVATION_USD > TASK_LIMIT_USD:
                raise BudgetError("The shared gross-usage budget is exhausted")
            started = _timestamp(self.clock(), "launch time")
            attempt = {"attempt_id": attempt_id or "smoke-" + uuid4().hex,
                       "task_id": self.task_id, "config_hash": config_hash,
                       "started_epoch": started, "deadline_epoch": started + DEADLINE_SECONDS,
                       "reserved_usd": ATTEMPT_RESERVATION_USD, "status": "active",
                       "cost_proof": proof}
            data["config_hash"] = config_hash
            attempts.append(attempt)
            self._write(data)
            return deepcopy(attempt)

    def finish(self, attempt_id: str, status: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
        if status not in TERMINAL_STATUSES:
            raise BudgetError("Finish requires an explicit terminal status")
        if details is not None and not isinstance(details, dict):
            raise BudgetError("Finish details must be an object")
        with self._locked():
            data = self._read()
            attempt = next((a for a in data["attempts"] if a["attempt_id"] == attempt_id), None)
            if attempt is None:
                raise BudgetError("Unknown attempt; cannot account for an unreserved launch")
            if attempt["status"] != "active":
                raise BudgetError("Attempt already finished; terminal history cannot be rewritten")
            attempt.update(status=status, finished_epoch=_timestamp(self.clock(), "finish time"),
                           details=deepcopy(details or {}))
            # reserved_usd deliberately never changes, including failure and cancellation.
            self._write(data)
            return deepcopy(attempt)


class RuntimeGuard:
    """Bound one smoke attempt before prefill/generation and retain failed work.

    Each turn reserves its maximum output and TWO complete input prefills
    (measurement and generation). Unused output capacity is not refunded.
    Sessions and pending turns are checkpointable; restore preserves the same
    absolute deadline. No automatic retry API is provided.
    """

    def __init__(self, deadline_epoch: float, state: dict[str, Any] | None = None,
                 clock: Callable[[], float] = time.time):
        self.deadline_epoch = _timestamp(deadline_epoch, "deadline_epoch")
        self.clock = clock
        self.sessions: list[dict[str, Any]] = []
        self.turns: list[dict[str, Any]] = []
        if state is not None:
            self._restore(state)

    def check_deadline(self) -> None:
        if _timestamp(self.clock(), "current time") >= self.deadline_epoch:
            raise BudgetError("Absolute task-attempt deadline reached; no further operation is permitted")

    @property
    def prefill_tokens_reserved(self) -> int:
        return sum(t["input_tokens"] * PREFILL_PASSES for t in self.turns)

    @property
    def generation_tokens_reserved(self) -> int:
        return sum(t["max_new_tokens"] for t in self.turns)

    @property
    def actual_output_tokens(self) -> int:
        return sum(t["actual_output_tokens"] or 0 for t in self.turns)

    def _pending(self) -> bool:
        return bool(self.turns and self.turns[-1]["actual_output_tokens"] is None)

    def start_session(self, session_id: str, planned_turns: int) -> None:
        self.check_deadline()
        _integer(planned_turns, "planned_turns", 1)
        if not isinstance(session_id, str) or not session_id.strip():
            raise BudgetError("A unique nonempty smoke session ID is required")
        if any(s["session_id"] == session_id for s in self.sessions):
            raise BudgetError("Session already started; duplicate sessions are prohibited")
        if len(self.sessions) >= MAX_SESSIONS or planned_turns > MAX_SESSION_TURNS:
            raise BudgetError("Smoke session count or per-session turn cap exceeded")
        if sum(s["planned_turns"] for s in self.sessions) + planned_turns > MAX_TURNS:
            raise BudgetError("Total planned smoke turns exceed 26")
        if self.sessions:
            previous = self.sessions[-1]
            completed = sum(t["session_id"] == previous["session_id"] and t["actual_output_tokens"] is not None
                            for t in self.turns)
            if self._pending() or completed != previous["planned_turns"]:
                raise BudgetError("Previous session has unfinished turns; do not silently discard them")
        self.sessions.append({"session_id": session_id, "planned_turns": planned_turns})

    def reserve_turn(self, input_tokens: int, max_new_tokens: int = MAX_OUTPUT_TOKENS) -> None:
        self.check_deadline()
        _integer(input_tokens, "input_tokens", 1)
        _integer(max_new_tokens, "max_new_tokens", 1)
        if not self.sessions:
            raise BudgetError("Start an explicitly bounded session before model operations")
        if self._pending():
            raise BudgetError("Previous turn has no validated response; a retry cannot bypass accounting")
        session = self.sessions[-1]
        count = sum(t["session_id"] == session["session_id"] for t in self.turns)
        if count >= session["planned_turns"] or len(self.turns) >= MAX_TURNS:
            raise BudgetError("Session or aggregate turn limit reached")
        if max_new_tokens > MAX_OUTPUT_TOKENS or input_tokens + max_new_tokens > MAX_CONTEXT_TOKENS:
            raise BudgetError("Output or full input-plus-generation context cap exceeded; no truncation")
        if self.prefill_tokens_reserved + PREFILL_PASSES * input_tokens > MAX_PREFILL_TOKENS:
            raise BudgetError("Two-pass aggregate prefill token cap exceeded")
        if self.generation_tokens_reserved + max_new_tokens > MAX_GENERATION_TOKENS:
            raise BudgetError("Aggregate generation token reservation cap exceeded")
        self.turns.append({"session_id": session["session_id"], "turn": count + 1,
                           "input_tokens": input_tokens, "max_new_tokens": max_new_tokens,
                           "actual_output_tokens": None})

    def finish_turn(self, actual_output_tokens: int) -> None:
        self.check_deadline()
        _integer(actual_output_tokens, "actual_output_tokens")
        if not self._pending():
            raise BudgetError("No unfinished reserved turn; duplicate completion is prohibited")
        if actual_output_tokens > self.turns[-1]["max_new_tokens"]:
            raise BudgetError("Model output exceeded its reserved hard token cap")
        self.turns[-1]["actual_output_tokens"] = actual_output_tokens

    def snapshot(self) -> dict[str, Any]:
        return {"schema_version": 1, "deadline_epoch": self.deadline_epoch,
                "sessions": deepcopy(self.sessions), "turns": deepcopy(self.turns),
                "prefill_tokens_reserved": self.prefill_tokens_reserved,
                "generation_tokens_reserved": self.generation_tokens_reserved,
                "actual_output_tokens": self.actual_output_tokens}

    def _restore(self, state: dict[str, Any]) -> None:
        try:
            if state["schema_version"] != 1 or state["deadline_epoch"] != self.deadline_epoch:
                raise BudgetError("Runtime checkpoint schema/deadline mismatch")
            # Replay guard decisions without granting a new deadline or refunds. An
            # expired snapshot can be inspected but still fails check_deadline().
            replay = RuntimeGuard(self.deadline_epoch, clock=lambda: self.deadline_epoch - 1)
            sessions, turns = state["sessions"], state["turns"]
            if not isinstance(sessions, list) or not isinstance(turns, list):
                raise BudgetError("Malformed runtime checkpoint")
            index = 0
            for session in sessions:
                replay.start_session(session["session_id"], session["planned_turns"])
                while index < len(turns) and turns[index]["session_id"] == session["session_id"]:
                    turn = turns[index]
                    replay.reserve_turn(turn["input_tokens"], turn["max_new_tokens"])
                    if turn["turn"] != replay.turns[-1]["turn"]:
                        raise BudgetError("Duplicate or missing runtime checkpoint turn")
                    if turn["actual_output_tokens"] is not None:
                        replay.finish_turn(turn["actual_output_tokens"])
                    index += 1
            if index != len(turns) or replay.snapshot() != state:
                raise BudgetError("Runtime checkpoint counters or session order are inconsistent")
            self.sessions, self.turns = replay.sessions, replay.turns
        except (ValueError, TypeError, KeyError) as error:
            raise BudgetError("Invalid runtime budget checkpoint") from error
