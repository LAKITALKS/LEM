"""Durable serial study orchestration; scientific settings are never edited here.

The source manifest is produced from a clean Git checkout by the local launcher.
A container without Git validates the exact shipped bytes against that manifest.
Its commit field is an attestation by that launcher, not a remote Git lookup.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import importlib.metadata
import json
import math
from pathlib import Path
import re
import time
import zipfile

from .design import build_sessions, validate_config
from .storage import SessionStore, atomic_json, canonical_hash, file_hash
from . import study

PHASES = ("development", "validation", "confirmation")
PHASE_RESERVATION_USD = 4.5
MAX_ATTEMPTS = 5  # Three phases and at most two explicitly investigated continuations.
WORKER_SECONDS = 10800
STARTUP_SECONDS = 900
TERMINATION_SECONDS = 60
RESOURCE_RATE = 0.000222 + 2 * 0.0000131 + 16 * 0.00000222
NONCOMPUTE_PER_ATTEMPT_USD = 1.0
TOTAL_NEW_USAGE_CEILING_USD = 24.0  # 22.5 attempt reservations + 1.5 storage/other reserve.


def source_manifest():
    from .launch_smoke import source_binding
    manifest = source_binding()
    manifest.update(schema="lem-ii.source-attestation.v1", created_epoch=time.time(),
                    source_sha256=study.source_hashes())
    manifest["source_set_sha256"] = canonical_hash(manifest["source_sha256"])
    return manifest


def verify_source(manifest, root=None):
    root = Path(root or study.PACKAGE.parent)
    if (manifest.get("schema") != "lem-ii.source-attestation.v1"
            or not re.fullmatch(r"[a-f0-9]{40}", manifest.get("commit", ""))
            or manifest.get("source_set_sha256") != canonical_hash(manifest.get("source_sha256"))):
        raise ValueError("Invalid frozen source attestation")
    expected_names = {f"lem_ii/{name}" for name in study.SOURCE_FILES}
    if not expected_names <= set(manifest["files"]):
        raise ValueError("Source attestation omits an executed production file")
    for name, digest in manifest["files"].items():
        path = root / name
        if not name.startswith("lem_ii/") or ".." in Path(name).parts or file_hash(path) != digest:
            raise ValueError("Container bytes differ from frozen source attestation")
    actual = {name: file_hash(root / "lem_ii" / name) for name in study.SOURCE_FILES}
    if actual != manifest["source_sha256"]:
        raise ValueError("Executed study source set differs from attestation")
    return manifest["commit"]


def phase_config(base, phase):
    validate_config(base)
    if phase not in PHASES:
        raise ValueError("Unknown study phase")
    result = deepcopy(base)
    result["execution"].update(study_collection_allowed=True, smoke_only=False,
                               allow_scientific_analysis=True,
                               confirmation_generation_allowed=phase == "confirmation",
                               confirmation_evaluation_allowed=phase == "confirmation",
                               allow_confirmation_analysis=phase == "confirmation")
    if study._scientific_hash(result) != study._scientific_hash(base):
        raise ValueError("Execution gates changed the scientific configuration")
    return result


def session_inventory(root, config, phase, *, allow_partial=False):
    """Validate complete stores before selecting any model work; no model loading."""
    root = Path(root)
    specs = build_sessions(config)
    known = {s.session_id for s in specs}
    if {p.name for p in root.glob("study-*") if p.is_dir()} - known:
        raise ValueError("Unplanned study directory")
    complete, pending = [], []
    for spec in (s for s in specs if s.split == phase):
        path = root / spec.session_id
        if not path.exists():
            pending.append((spec, None))
            continue
        manifest = json.loads((path / "manifest.json").read_text())
        study._validate_binding(manifest["binding"], config)
        store = SessionStore(root, spec, manifest["binding"], resume=True)
        if store.manifest["status"] == "complete":
            complete.append(spec.session_id)
        elif allow_partial:
            pending.append((spec, manifest))
        else:
            raise ValueError("Partial session requires an explicit inspected continuation")
    return complete, pending


def verify_freeze(directory, config, source=None):
    directory = Path(directory)
    seal = json.loads((directory / "freeze.json").read_text())
    if seal["schema"] != "lem-ii.freeze.v1" or seal["scientific_config_sha256"] != study._scientific_hash(config):
        raise ValueError("Freeze configuration mismatch")
    if seal["source_sha256"] != study.source_hashes():
        raise ValueError("Freeze source files changed")
    if source is not None and seal["source_manifest_sha256"] != canonical_hash(source):
        raise ValueError("Freeze belongs to a different execution snapshot")
    required = {f"study_endpoint_{point}.{suffix}" for point in (12, 16, 20, 24)
                for suffix in ("joblib", "manifest.json")}
    required |= {"verified_inputs.json", "study_development_validation.json", "analysis_started.json"}
    if not required <= set(seal["files"]):
        raise ValueError("Incomplete four-endpoint freeze")
    if any(file_hash(directory / name) != digest for name, digest in seal["files"].items()):
        raise ValueError("Frozen file changed")
    report = json.loads((directory / "study_development_validation.json").read_text())
    if (report["label_countercheck"]["replicates"] != 199
            or report["simultaneous_grid_contrast_intervals"]["bootstrap_replicates"] != 2000):
        raise ValueError("Freeze used reduced fixture repetition counts")
    inputs = json.loads((directory / "verified_inputs.json").read_text())
    if inputs["session_count"] != 72 or inputs["splits"] != ["development", "validation"]:
        raise ValueError("Freeze requires exactly the 72 development/validation inputs")
    return seal


def seal_freeze(directory, config, source):
    from .evaluation import analysis_provenance, load_frozen_artifact
    directory = Path(directory)
    if (directory / "freeze.json").exists():
        return verify_freeze(directory, config, source)
    for point in (12, 16, 20, 24):
        fitted = load_frozen_artifact(directory / f"study_endpoint_{point}.joblib",
                                      directory / f"study_endpoint_{point}.manifest.json", config)
        if fitted["endpoint"] != point or len(fitted["development_ids"]) != 36 or len(fitted["validation_ids"]) != 36:
            raise ValueError("Unexpected fitted assignment")
    files = {p.name: file_hash(p) for p in directory.iterdir() if p.is_file()}
    seal = {"schema": "lem-ii.freeze.v1", "sealed_epoch": time.time(), "files": files,
            "scientific_config_sha256": study._scientific_hash(config),
            "source_sha256": study.source_hashes(), "source_manifest_sha256": canonical_hash(source),
            "source_commit": source["commit"], **analysis_provenance(),
            "confirmation_generated": False}
    atomic_json(directory / "freeze.json", seal)
    verify_freeze(directory, config, source)
    for name in (*files, "freeze.json"):
        (directory / name).chmod(0o444)
    return seal


def reserve_attempt(ledger, attempt_id, phase):
    """Never refund failed reservations; retain the remaining phases' allocations."""
    if ledger["schema"] != "lem-ii.production-ledger.v1" or phase not in PHASES:
        raise ValueError("Invalid production ledger")
    attempts = ledger["attempts"]
    if len({a["attempt_id"] for a in attempts}) != len(attempts):
        raise ValueError("Duplicate ledger attempts")
    if any(a["status"] == "active" for a in attempts):
        raise ValueError("Unresolved active attempt; verify provider termination first")
    if any(a["attempt_id"] == attempt_id for a in attempts):
        raise ValueError("Attempt already consumed; infrastructure replay is forbidden")
    if len(attempts) >= MAX_ATTEMPTS:
        raise ValueError("Shared attempt allowance exhausted")
    if any(a["reserved_usd"] != PHASE_RESERVATION_USD for a in attempts):
        raise ValueError("Unexpected permanent budget reservation")
    finished = set(ledger["completed_phases"])
    if any(previous not in finished for previous in PHASES[:PHASES.index(phase)]):
        raise ValueError("Earlier phase is incomplete")
    remaining = set(PHASES) - finished - {phase}
    committed = sum(a["reserved_usd"] for a in attempts)
    if committed + PHASE_RESERVATION_USD * (1 + len(remaining)) > 22.5:
        raise ValueError("Must retain budget for every outstanding phase")
    envelope = RESOURCE_RATE * (WORKER_SECONDS + STARTUP_SECONDS + TERMINATION_SECONDS) + NONCOMPUTE_PER_ATTEMPT_USD
    if envelope > PHASE_RESERVATION_USD:
        raise ValueError("Provider runtime envelope exceeds phase reservation")
    attempt = {"attempt_id": attempt_id, "phase": phase, "status": "active",
               "reserved_usd": PHASE_RESERVATION_USD, "provider_envelope_usd": envelope,
               "started_epoch": time.time()}
    attempts.append(attempt)
    return attempt


def collect_phase(root, base, phase, attempt_id, source, *, persist=lambda: None,
                  allow_partial=False, adapter_loader=None, pre_reserved=False):
    """Existing extraction path, one model load, durable per-turn and session commits."""
    from .collection import collect_session
    from .model_adapter import ModelAdapter
    root = Path(root)
    config = phase_config(base, phase)
    commit = verify_source(source)
    root.mkdir(parents=True, exist_ok=True)
    if phase == "confirmation":
        seal = verify_freeze(root / "freeze", config, source)
        if seal["sealed_epoch"] >= time.time():
            raise ValueError("Freeze must precede confirmation")
    for previous_phase in PHASES[:PHASES.index(phase)]:
        previous_complete, previous_pending = session_inventory(root / "data", config, previous_phase)
        if len(previous_complete) != 36 or previous_pending:
            raise ValueError("Prior phase does not have 36 complete validated sessions")
    complete, pending = session_inventory(root / "data", config, phase, allow_partial=allow_partial)
    if not pending:
        return {"status": "already_complete", "phase": phase, "new_model_calls": 0, "complete_sessions": complete}
    ledger_path = root / "study-usage-ledger.json"
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text())
    else:
        ledger = {"schema": "lem-ii.production-ledger.v1", "study_id": root.name,
                  "scientific_config_sha256": study._scientific_hash(config),
                  "source_manifest_sha256": canonical_hash(source), "completed_phases": [], "attempts": []}
    if (ledger["scientific_config_sha256"] != study._scientific_hash(config)
            or ledger["source_manifest_sha256"] != canonical_hash(source)):
        raise ValueError("Ledger source/configuration binding changed")
    if pre_reserved:
        matches = [a for a in ledger["attempts"] if a["attempt_id"] == attempt_id]
        if (len(matches) != 1 or matches[0]["status"] != "active" or matches[0]["phase"] != phase
                or matches[0]["reserved_usd"] != PHASE_RESERVATION_USD
                or sum(a["reserved_usd"] for a in ledger["attempts"]) > 22.5):
            raise ValueError("Remote invocation lacks its precommitted shared reservation")
        attempt = matches[0]
    else:
        attempt = reserve_attempt(ledger, attempt_id, phase)
    atomic_json(ledger_path, ledger)
    run_dir = root / "data" / "runs" / attempt_id
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    limits = {"deadline_epoch": started + WORKER_SECONDS - 30, "max_seconds": WORKER_SECONDS - 30,
              "max_sessions": 36, "max_prefill_tokens": 2 * 36 * 24 * 8192,
              "max_generation_tokens": 36 * 24 * 128, "external_hard_timeout_seconds": WORKER_SECONDS,
              "completed_prefixes": {spec.session_id: old["completed_turns"] for spec, old in pending if old}}
    sources = study.source_hashes()
    run = {"schema": "lem-ii.study-run.v1", "run_id": attempt_id, "created_epoch": started,
           "config_snapshot": config, "scientific_config_sha256": study._scientific_hash(config),
           "source_sha256": sources, "source_set_sha256": canonical_hash(sources), "code_reference": commit,
           "source_attestation": source, "source_verification": "container_bytes_verified_against_clean_git_launcher_attestation",
           "assignment_sha256": canonical_hash([asdict(s) for s in build_sessions(config)]),
           "planned_session_ids": [s.session_id for s, _ in pending], "limits": limits,
           "cost_envelope": deepcopy(attempt), "skipped_complete_session_ids": complete,
           "continuations": {s.session_id: {"original_run_id": old["binding"]["run"]["run_id"],
                                             "validated_prefix_turns": old["completed_turns"]}
                             for s, old in pending if old}}
    if phase == "confirmation":
        run["freeze_sha256"] = file_hash(root / "freeze" / "freeze.json")
        run["freeze_sealed_epoch"] = seal["sealed_epoch"]
    atomic_json(run_dir / "manifest.json", run)
    guard = study.StudyRuntimeGuard(limits, run_dir / "runtime.json")
    persist()  # Reservation survives a container loss before the first model call.
    try:
        adapter = (adapter_loader or ModelAdapter.load)(config)
        if adapter.tiny_fixture:
            raise ValueError("Fixture adapter is ineligible for scientific collection")
        attempt["model_load_seconds"] = time.time() - started
        atomic_json(run_dir / "software.json", {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()})
        for spec, previous in pending:
            guard.check_deadline()
            original_run = previous["binding"]["run"] if previous else run
            original_config = original_run["config_snapshot"] if previous else config
            store = collect_session(original_config, spec, adapter, root / "data", original_run,
                                    guard=guard, resume=previous is not None,
                                    checkpoint_hook=lambda _: persist())
            store.validate()
            archive_dir = root / "session-archives"
            archive_dir.mkdir(exist_ok=True)
            archive = archive_dir / f"{spec.session_id}.zip"
            if archive.exists():
                raise ValueError("Do not overwrite an existing complete-session archive")
            with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as handle:
                for path in store.path.rglob("*"):
                    if path.is_file():
                        handle.write(path, path.relative_to(root / "data"))
            atomic_json(root / "progress.json", {"phase": phase, "latest_complete_session": spec.session_id,
                                               "updated_epoch": time.time(), "attempt_id": attempt_id})
            persist()
        checked, rest = session_inventory(root / "data", config, phase)
        if len(checked) != 36 or rest:
            raise ValueError("Phase is incomplete")
        attempt["status"] = "complete"
        ledger["completed_phases"].append(phase)
    except BaseException as error:
        attempt["status"] = "failed"
        attempt["error_type"] = type(error).__name__
        attempt["error"] = str(error)[:1000]
        raise
    finally:
        attempt["finished_epoch"] = time.time()
        attempt["elapsed_seconds"] = attempt["finished_epoch"] - started
        atomic_json(ledger_path, ledger)
        atomic_json(run_dir / "result.json", attempt)
        persist()
    return deepcopy(attempt)
