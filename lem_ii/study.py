"""Future-only genuine-study CLI. Current configuration permits metadata planning.

No command provisions cloud resources. Collection requires an already provisioned
CUDA/BF16 machine and separately enabled execution gates. Every attempt reserves
its full conservative envelope in a shared ledger, including failed attempts.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid

from .design import SessionSpec, build_sessions, load_config
from .storage import SessionStore, atomic_json, canonical_hash, file_hash

PACKAGE = Path(__file__).resolve().parent
SOURCE_FILES = ("__init__.py", "design.py", "simulator.py", "model_adapter.py", "storage.py",
                "collection.py", "features.py", "evaluation.py", "study.py", "requirements.txt")
STUDY_LEDGER = PACKAGE.parent / ".runtime" / "study-usage-ledger.json"
PINNED_COLLECTION_SOFTWARE = {"torch": "2.9.1", "transformers": "4.57.6", "tokenizers": "0.22.2",
                              "huggingface-hub": "0.36.2", "safetensors": "0.7.0", "numpy": "2.4.6"}


def _scientific_hash(config):
    from .evaluation import analysis_configuration_hash
    return analysis_configuration_hash(config)


def source_hashes():
    return {name: file_hash(PACKAGE / name) for name in SOURCE_FILES}


def _code_reference():
    try:
        subprocess.run(["git", "ls-files", "--error-unmatch", "--", *SOURCE_FILES], cwd=PACKAGE,
                       text=True, capture_output=True, check=True)
    except subprocess.CalledProcessError as error:
        raise ValueError("Every collection source file must be tracked and committed before execution") from error
    status = subprocess.run(["git", "status", "--porcelain=v1", "--untracked-files=all", "--", *SOURCE_FILES],
                            cwd=PACKAGE, text=True, capture_output=True, check=True)
    if status.stdout.strip():
        raise ValueError("Collection source files have uncommitted changes; commit the reviewed source first")
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PACKAGE,
                            text=True, capture_output=True, check=True)
    return result.stdout.strip()


def _require(config, operation, splits=()):
    gates = config["execution"]
    if operation == "collect":
        if not gates["study_collection_allowed"] or gates["smoke_only"]:
            raise PermissionError("Study collection is not authorized; configuration remains smoke-only")
        if "confirmation" in splits and not gates["confirmation_generation_allowed"]:
            raise PermissionError("Confirmation generation is not authorized")
    else:
        if not gates.get("allow_scientific_analysis", False):
            raise PermissionError("Scientific analysis is not authorized")
        if "confirmation" in splits and not gates.get("allow_confirmation_analysis", False):
            raise PermissionError("Confirmation analysis is not authorized")


def plan(config, output):
    """Write only prespecified assignment metadata; no simulated dialogue text."""
    assignments = [asdict(spec) for spec in build_sessions(config)]
    payload = {"schema": "lem-ii.study-plan.v1", "metadata_only": True,
               "dialogues_generated": 0, "status": config["status"],
               "config_sha256": canonical_hash(config), "scientific_config_sha256": _scientific_hash(config),
               "assignment_sha256": canonical_hash(assignments), "planned_sessions": len(assignments),
               "planned_assistant_responses": sum(row["turns"] for row in assignments), "assignments": assignments}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(payload, indent=2) + "\n")
    return payload


def _validate_binding(binding, config):
    """Validate captured configuration and exact collection-code identity."""
    from .model_adapter import TEMPLATE_SHA256
    if binding.get("data_kind") != "real_model_study" or binding.get("eligible_for_scientific_analysis") is not True:
        raise ValueError("Smoke, fixtures or ineligible records cannot enter study analysis")
    run = binding["run"]
    if run.get("schema") != "lem-ii.study-run.v1":
        raise ValueError("A genuine study-run manifest is required")
    snapshot = run["config_snapshot"]
    if canonical_hash(snapshot) != binding["config_sha256"]:
        raise ValueError("Captured execution configuration digest mismatch")
    if not snapshot["execution"]["study_collection_allowed"] or snapshot["execution"]["smoke_only"]:
        raise ValueError("Stored data were not collected under a study-enabled configuration")
    scientific = _scientific_hash(config)
    if _scientific_hash(snapshot) != scientific or run["scientific_config_sha256"] != scientific:
        raise ValueError("Stored and requested scientific configurations disagree")
    current_sources = source_hashes()
    if run["source_sha256"] != current_sources or run["source_set_sha256"] != canonical_hash(current_sources):
        raise ValueError("Collection source files changed; no silent rebinding of study data")
    adapter = binding["adapter"]
    if any(adapter.get("software", {}).get(name) != version for name, version in PINNED_COLLECTION_SOFTWARE.items()):
        raise ValueError("Stored collection software differs from the pinned tested versions")
    if (adapter.get("tiny_random_fixture") is not False
            or adapter.get("model_revision") != config["model"]["revision"]
            or adapter.get("tokenizer_revision") != config["model"]["tokenizer_revision"]
            or adapter.get("chat_template_sha256") != TEMPLATE_SHA256
            or adapter.get("weight_dtype") != "torch.bfloat16"
            or adapter.get("attention_implementation") != config["model"]["attention_implementation"]):
        raise ValueError("Stored model, tokenizer, precision or extraction identity mismatch")
    if (binding["state_shape"] != [len(config["extraction"]["layers"]), 2048]
            or binding["max_new_tokens"] != config["model"]["generation"]["max_new_tokens"]
            or binding["eos_token_ids"] != config["model"]["generation"]["eos_token_id"]):
        raise ValueError("Stored extraction shape or generation limits changed")


def load_completed_study_records(root, config, splits=("development", "validation")):
    """Read exactly 72 development/validation or 36 confirmation study sessions.

    Validate each complete SessionStore before loading vectors, reject any
    out-of-plan study folder, and return source checksums with the feature records.
    This never calls a model or generates confirmation fixtures.
    """
    splits = tuple(splits)
    if splits not in (("development", "validation"), ("confirmation",)):
        raise ValueError("Analysis requires the complete prespecified phase set")
    _require(config, "analyze", splits)
    root = Path(root)
    assignment = build_sessions(config)
    expected = {spec.session_id: spec for spec in assignment if spec.split in splits}
    all_planned = {spec.session_id for spec in assignment}
    observed = {path.name for path in root.glob("study-*") if path.is_dir()}
    if observed - all_planned:
        raise ValueError("Unplanned or duplicate-named study session folders found")
    if not set(expected) <= observed:
        raise ValueError(f"Incomplete phase: require {len(expected)} complete planned sessions")
    rows, source_files = [], {}
    layer_index = config["extraction"]["layers"].index(config["extraction"]["primary_layer"])
    for identifier, spec in expected.items():
        directory = root / identifier
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest["spec"] != asdict(spec) or manifest["status"] != "complete" or manifest["completed_turns"] != 24:
            raise ValueError("Stored session does not match its complete planned assignment")
        _validate_binding(manifest["binding"], config)
        if spec.session_id not in manifest["binding"]["run"]["planned_session_ids"]:
            raise ValueError("Session is absent from its captured run assignment")
        if spec.split == "confirmation" and not manifest["binding"]["run"]["config_snapshot"]["execution"]["confirmation_generation_allowed"]:
            raise ValueError("Confirmation was generated without its stored execution gate")
        store = SessionStore(root, spec, manifest["binding"], resume=True)
        store.validate()
        row = store.analysis_record(layer_index)
        if row["data_kind"] != "real_model_study" or not row["eligible_for_scientific_analysis"]:
            raise ValueError("Analysis record kind differs from verified source")
        rows.append(row)
        paths = [manifest_path, directory / "labels.json"]
        paths.extend(directory / f"turn-{turn:02d}" / name for turn in range(1, 25)
                     for name in ("visible.json", "audit.json", "states.npy"))
        source_files[identifier] = {str(path.relative_to(root)): file_hash(path) for path in paths}
    from .features import validate_records
    validate_records(rows, config=config)
    provenance = {"input_root": str(root.resolve()), "session_count": len(rows),
                  "splits": list(splits), "scientific_config_sha256": _scientific_hash(config),
                  "source_code_sha256": source_hashes(), "input_file_sha256": source_files,
                  "input_set_sha256": canonical_hash(source_files), "all_session_stores_validated": True}
    return rows, provenance


def analyze_development_validation(root, config, output):
    _require(config, "analyze", ("development", "validation"))
    from sklearn.metrics import balanced_accuracy_score
    from .evaluation import (METHODS, analysis_provenance, earliest_recognition, fit_and_tune,
                             freeze_artifact, label_countercheck, labels_for, paired_cluster_intervals,
                             predict, simultaneous_grid_intervals, split_records, time_countercheck)
    rows, provenance = load_completed_study_records(root, config)
    output = Path(output)
    # A single output directory freezes one analysis, rather than permitting a
    # succession of favorable classifier fits to replace an earlier freeze.
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "verified_inputs.json", provenance)
    atomic_json(output / "analysis_started.json", {"status": "started_once", "input_set_sha256": provenance["input_set_sha256"]})
    development, validation = split_records(rows, config)
    report = {"data_kind": "real_model_study", "scope": "development_validation_descriptive",
              "confirmation_generated_or_evaluated": False, "scientific_config_sha256": _scientific_hash(config),
              "input_set_sha256": provenance["input_set_sha256"], "endpoints": {}, **analysis_provenance()}
    grid, primary = {}, None
    for endpoint in config["features"]["endpoints"]:
        fitted = fit_and_tune(rows, config, endpoint)
        grid[endpoint] = predict(fitted, validation)
        development_predictions = predict(fitted, development)
        frozen = freeze_artifact(fitted, output)
        report["endpoints"][endpoint] = {
            "validation_balanced_accuracy": {m: float(balanced_accuracy_score(labels_for(validation), p)) for m, p in grid[endpoint].items()},
            "development_balanced_accuracy": {m: float(balanced_accuracy_score(labels_for(development), p)) for m, p in development_predictions.items()},
            "tuning": fitted["tuning"], "artifact_sha256": frozen["artifact_sha256"]}
        if endpoint == 24:
            primary = fitted
    report["primary_paired_differences_descriptive_validation"] = paired_cluster_intervals(validation, grid[24], config)
    report["simultaneous_grid_contrast_intervals"] = simultaneous_grid_intervals(validation, grid, config)
    report["time_countercheck"] = time_countercheck(primary, validation)
    report["label_countercheck"] = label_countercheck(primary, rows, config)
    report["recognition_descriptive"] = {
        method: earliest_recognition({point: values["validation_balanced_accuracy"][method] for point, values in report["endpoints"].items()},
                                     config["analysis"]["recognition_threshold"], config["analysis"]["recognition_consecutive"])
        for method in METHODS}
    report["topic_population_inference"] = False
    atomic_json(output / "study_development_validation.json", report)
    return report


def confirm(root, config, artifacts, output):
    _require(config, "analyze", ("confirmation",))
    from .evaluation import predict_confirmation_grid_once
    output = Path(output)
    if output.exists() or output.with_suffix(output.suffix + ".inputs.json").exists():
        raise FileExistsError("Confirmation output or input provenance already exists")
    rows, provenance = load_completed_study_records(root, config, ("confirmation",))
    output.parent.mkdir(parents=True, exist_ok=True)
    # This provenance contains no model predictions. The analysis API reserves
    # its permanent shared grid lock before any confirmation prediction.
    atomic_json(output.with_suffix(output.suffix + ".inputs.json"), provenance)
    return predict_confirmation_grid_once(Path(artifacts), rows, config, output)


class StudyRuntimeGuard:
    """Explicit future run limits, persisted before every costly turn.

    Failed or pending turn reservations are never refunded. A failed attempt has
    no automatic retry or new-deadline resume path in this CLI.
    """
    def __init__(self, limits, path, *, clock=time.time):
        self.limits, self.path, self.clock = deepcopy(limits), Path(path), clock
        self.state = {"deadline_epoch": limits["deadline_epoch"], "sessions": [], "turns": []}
        # Exclusive creation also prevents a second controller from replacing
        # existing reservations between an existence check and the first write.
        with self.path.open("x") as stream:
            json.dump(self.state, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _save(self):
        atomic_json(self.path, self.state)

    def check_deadline(self):
        if self.clock() >= self.limits["deadline_epoch"]:
            raise RuntimeError("Absolute study run deadline reached")

    def start_session(self, session_id, planned_turns):
        self.check_deadline()
        if planned_turns != 24 or session_id in self.state["sessions"] or len(self.state["sessions"]) >= self.limits["max_sessions"]:
            raise RuntimeError("Study session count, identity or turn limit exceeded")
        if self.state["turns"] and self.state["turns"][-1]["actual_output_tokens"] is None:
            raise RuntimeError("Failed previous turn cannot be silently retried")
        if self.state["sessions"]:
            previous = self.state["sessions"][-1]
            if sum(t["session_id"] == previous for t in self.state["turns"]) != 24:
                raise RuntimeError("Previous study session is incomplete")
        self.state["sessions"].append(session_id)
        self._save()

    def reserve_turn(self, input_tokens, max_new_tokens):
        self.check_deadline()
        if any(type(n) is not int or n < 1 for n in (input_tokens, max_new_tokens)):
            raise RuntimeError("Positive integer token reservations required")
        if not self.state["sessions"] or (self.state["turns"] and self.state["turns"][-1]["actual_output_tokens"] is None):
            raise RuntimeError("Start a session and finish the previous turn before reserving")
        session = self.state["sessions"][-1]
        if sum(t["session_id"] == session for t in self.state["turns"]) >= 24:
            raise RuntimeError("Per-session 24-turn limit reached")
        if max_new_tokens > 128 or input_tokens + max_new_tokens > 8192:
            raise RuntimeError("Context or output reservation cap exceeded")
        if (sum(t["input_tokens"] * 2 for t in self.state["turns"]) + 2 * input_tokens > self.limits["max_prefill_tokens"]
                or sum(t["max_new_tokens"] for t in self.state["turns"]) + max_new_tokens > self.limits["max_generation_tokens"]):
            raise RuntimeError("Aggregate two-prefill or generation-token limit exceeded")
        self.state["turns"].append({"session_id": session, "input_tokens": input_tokens,
                                     "max_new_tokens": max_new_tokens, "actual_output_tokens": None})
        self._save()

    def finish_turn(self, actual_output_tokens):
        self.check_deadline()
        if (type(actual_output_tokens) is not int or actual_output_tokens < 1 or not self.state["turns"]
                or self.state["turns"][-1]["actual_output_tokens"] is not None
                or actual_output_tokens > self.state["turns"][-1]["max_new_tokens"]):
            raise RuntimeError("Invalid, duplicate or over-limit model completion")
        self.state["turns"][-1]["actual_output_tokens"] = actual_output_tokens
        self._save()


@contextmanager
def _absolute_alarm(seconds):
    if not hasattr(signal, "SIGALRM"):
        raise RuntimeError("This command requires Unix SIGALRM for its absolute runtime guard")
    def expired(signum, frame):
        raise TimeoutError("Absolute collection runtime exceeded, including model loading")
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _load_adapter(config, cache_dir):
    from .model_adapter import ModelAdapter
    return ModelAdapter.load(config, cache_dir=cache_dir)


def collect(root, config, split, *, max_seconds, max_sessions, max_prefill_tokens,
            max_generation_tokens, gross_cost_cap_usd, inclusive_resource_usd_per_second,
            noncompute_reserve_usd, prior_technical_cost_usd, external_hard_timeout_seconds=None, cache_dir=None):
    """Collect on an existing CUDA machine; never provision or start Modal.

    Resource rate must include GPU + CPU/RAM + associated running resources.
    Reservation covers full max_seconds + 60 seconds termination allowance and
    caller-specified startup/storage reserve. Provider billing is not guaranteed.
    The fixed shared ledger counts reservations permanently across all attempts.
    A separately verified provider resource timeout is mandatory. SIGALRM is a
    local process guard, not a guarantee that a GPU or paid resource terminates.
    """
    _require(config, "collect", (split,))
    if split not in {"development", "validation", "confirmation"}:
        raise ValueError("One planned split is required")
    for name, value, ceiling in (("max_seconds", max_seconds, 86400), ("max_sessions", max_sessions, 36),
                                  ("max_prefill_tokens", max_prefill_tokens, 2 * 36 * 24 * 8192),
                                  ("max_generation_tokens", max_generation_tokens, 36 * 24 * 128)):
        if type(value) is not int or not 1 <= value <= ceiling:
            raise ValueError(f"Explicit {name} must be a positive integer no greater than {ceiling}")
    if type(external_hard_timeout_seconds) is not int or not 0 < external_hard_timeout_seconds <= max_seconds + 60:
        raise ValueError("Independently verified external resource hard timeout is required and must fit max_seconds + 60")
    numbers = (gross_cost_cap_usd, inclusive_resource_usd_per_second, noncompute_reserve_usd, prior_technical_cost_usd)
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in numbers):
        raise ValueError("Finite explicit gross-cost inputs are required")
    if not (0 < gross_cost_cap_usd <= 25 and inclusive_resource_usd_per_second > 0
            and noncompute_reserve_usd > 0 and 0 <= prior_technical_cost_usd <= 5):
        raise ValueError("Study cost cap/reserves exceed the separately releasable $25 scope")
    envelope = inclusive_resource_usd_per_second * (max_seconds + 60) + noncompute_reserve_usd
    if envelope > gross_cost_cap_usd or prior_technical_cost_usd + gross_cost_cap_usd > 30:
        raise ValueError("Resource-time envelope does not fit the stated gross-use budget")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    STUDY_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    lock_path = STUDY_LEDGER.with_suffix(".lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        scientific = _scientific_hash(config)
        if STUDY_LEDGER.exists():
            ledger = json.loads(STUDY_LEDGER.read_text())
            if (ledger.get("schema") != "lem-ii.study-budget.v1" or ledger["scientific_config_sha256"] != scientific
                    or ledger["prior_technical_cost_usd"] != prior_technical_cost_usd):
                raise ValueError("Shared study ledger binding changed; do not reset it")
            attempts = ledger["attempts"]
            if (not isinstance(attempts, list)
                    or len({a["run_id"] for a in attempts}) != len(attempts)
                    or any(a["status"] not in {"active", "complete", "failed"}
                           or not isinstance(a["reserved_usd"], (int, float))
                           or not math.isfinite(a["reserved_usd"]) or not 0 < a["reserved_usd"] <= 25
                           for a in attempts)):
                raise ValueError("Invalid shared budget reservations; refuse to replace ledger")
            if any(a["status"] == "active" for a in ledger["attempts"]):
                raise ValueError("An unresolved prior attempt remains active; inspect it before another run")
        else:
            ledger = {"schema": "lem-ii.study-budget.v1", "scientific_config_sha256": scientific,
                      "prior_technical_cost_usd": prior_technical_cost_usd, "attempts": []}
        reserved = sum(a["reserved_usd"] for a in ledger["attempts"])
        if reserved + gross_cost_cap_usd > min(25.0, 30.0 - prior_technical_cost_usd):
            raise ValueError("Shared future-study reservation allowance exhausted; failed runs are not refunded")
        candidates = [s for s in build_sessions(config) if s.split == split]
        selected = []
        # Complete compatible sessions are skipped without opening a model. Partial
        # sessions need explicit investigation, not an automatic replay/refund.
        for spec in candidates:
            path = root / spec.session_id
            if path.exists():
                manifest = json.loads((path / "manifest.json").read_text())
                _validate_binding(manifest["binding"], config)
                store = SessionStore(root, spec, manifest["binding"], resume=True)
                if store.manifest["status"] != "complete":
                    raise ValueError("Partial/failed study session needs explicit checkpoint and budget review")
            else:
                selected.append(spec)
        selected = selected[:max_sessions]
        if not selected:
            return {"status": "already_complete", "split": split, "new_model_calls": 0}
        started = time.time()
        run_id = "study-run-" + uuid.uuid4().hex
        limits = {"deadline_epoch": started + max_seconds, "max_seconds": max_seconds, "max_sessions": max_sessions,
                  "max_prefill_tokens": max_prefill_tokens, "max_generation_tokens": max_generation_tokens,
                  "external_hard_timeout_seconds": external_hard_timeout_seconds,
                  "resource_termination_contract": "provider resource termination must be independently verified; CLI does not manage resource"}
        sources = source_hashes()
        run = {"schema": "lem-ii.study-run.v1", "run_id": run_id, "created_epoch": started,
               "config_snapshot": deepcopy(config), "scientific_config_sha256": scientific,
               "source_sha256": sources, "source_set_sha256": canonical_hash(sources), "code_reference": _code_reference(),
               "assignment_sha256": canonical_hash([asdict(s) for s in build_sessions(config)]),
               "planned_session_ids": [s.session_id for s in selected], "limits": limits,
               "cost_envelope": {"reserved_usd": gross_cost_cap_usd, "inclusive_resource_usd_per_second": inclusive_resource_usd_per_second,
                                 "termination_allowance_seconds": 60, "noncompute_reserve_usd": noncompute_reserve_usd,
                                 "envelope_usd": envelope, "provider_bill_guaranteed": False}}
        run_directory = root / "runs" / run_id
        run_directory.mkdir(parents=True)
        atomic_json(run_directory / "manifest.json", run)
        attempt = {"run_id": run_id, "reserved_usd": gross_cost_cap_usd, "status": "active", "input_root": str(root)}
        ledger["attempts"].append(attempt)
        atomic_json(STUDY_LEDGER, ledger)
        guard = StudyRuntimeGuard(limits, run_directory / "runtime.json")
        from .collection import collect_session
        try:
            with _absolute_alarm(max_seconds):
                guard.check_deadline()
                adapter = _load_adapter(config, cache_dir)
                guard.check_deadline()
                if adapter.tiny_fixture:
                    raise ValueError("Random tiny fixtures cannot be collected as study records")
                for spec in selected:
                    collect_session(config, spec, adapter, root, run, guard=guard)
            attempt["status"] = "complete"
        except BaseException as error:
            attempt["status"] = "failed"
            attempt["error_type"] = type(error).__name__
            raise
        finally:
            attempt["finished_epoch"] = time.time()
            attempt["elapsed_seconds"] = attempt["finished_epoch"] - started
            atomic_json(STUDY_LEDGER, ledger)
            atomic_json(run_directory / "result.json", attempt)
        return {**attempt, "session_ids": [s.session_id for s in selected], "runtime": guard.state}


def main(argv=None):
    parser = argparse.ArgumentParser(description="LEM-II-Light planned study; all execution remains gated")
    parser.add_argument("--config", type=Path, default=PACKAGE / "config" / "study.json")
    commands = parser.add_subparsers(dest="command", required=True)
    planning = commands.add_parser("plan", help="Write all 108 assignments and hashes; no dialogues")
    planning.add_argument("--output", type=Path, required=True)
    collecting = commands.add_parser("collect", help="Future-only collection on already provisioned CUDA/BF16 hardware")
    collecting.add_argument("--root", type=Path, required=True)
    collecting.add_argument("--split", choices=("development", "validation", "confirmation"), required=True)
    collecting.add_argument("--cache-dir", type=Path)
    for flag in ("max-seconds", "max-sessions", "max-prefill-tokens", "max-generation-tokens", "external-hard-timeout-seconds"):
        collecting.add_argument("--" + flag, type=int)
    for flag in ("gross-cost-cap-usd", "inclusive-resource-usd-per-second", "noncompute-reserve-usd", "prior-technical-cost-usd"):
        collecting.add_argument("--" + flag, type=float)
    analysis = commands.add_parser("analyze-development-validation", help="Future-only complete 72-session fit/tune/freeze")
    analysis.add_argument("--root", type=Path, required=True)
    analysis.add_argument("--output", type=Path, required=True)
    confirmation = commands.add_parser("confirm", help="Future-only one-time full-grid frozen confirmation")
    confirmation.add_argument("--root", type=Path, required=True)
    confirmation.add_argument("--artifacts", type=Path, required=True)
    confirmation.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.command == "plan":
        result = plan(config, args.output)
        summary = {k: result[k] for k in ("metadata_only", "planned_sessions", "planned_assistant_responses", "assignment_sha256")}
    elif args.command == "collect":
        values = vars(args).copy()
        for key in ("command", "config"):
            values.pop(key)
        summary = collect(config=config, **values)
    elif args.command == "analyze-development-validation":
        result = analyze_development_validation(args.root, config, args.output)
        summary = {"scope": result["scope"], "output": str(args.output.resolve()), "confirmation_generated_or_evaluated": False}
    else:
        result = confirm(args.root, config, args.artifacts, args.output)
        summary = {"scope": result["scope"], "output": str(args.output.resolve())}
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
