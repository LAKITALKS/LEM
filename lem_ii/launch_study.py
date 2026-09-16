"""One durable full study, three serial phases; no phase-specific fresh ledgers."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import uuid
import zipfile

from .design import load_config
from .launch_smoke import command, MODAL, stop_owned_app
from .production import (PHASES, PHASE_RESERVATION_USD, TOTAL_NEW_USAGE_CEILING_USD,
                         WORKER_SECONDS, STARTUP_SECONDS, TERMINATION_SECONDS,
                         phase_config, reserve_attempt, session_inventory,
                         source_manifest, verify_source, verify_freeze)
from .storage import atomic_json, canonical_hash, file_hash
from .study import _scientific_hash

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / ".runtime" / "full-study-20260908"
STUDY_ID = "lem-ii-light-full-20260908"


def read_preflight(path, ledger=None):
    proof = json.loads(Path(path).read_text())
    age = time.time() - datetime.fromisoformat(proof["verified_at_utc"]).timestamp()
    usage, cap = proof["workspace_usage_upper_usd"], proof["workspace_gross_usage_limit_usd"]
    if (not 0 <= age <= 7200 or isinstance(usage, bool) or not isinstance(usage, (int, float))
            or not math.isfinite(usage) or usage < 0 or cap != 27 or proof["environment"] != "main"):
        raise ValueError("Fresh evidence of the authorized $27 gross workspace limit is required")
    if proof["rates"] != {"L4": 0.000222, "CPU_core": 0.0000131, "RAM_GiB": 0.00000222}:
        raise ValueError("Tariffs changed; recalculate every phase before proceeding")
    if command(MODAL, "profile", "current") != proof["workspace"]:
        raise ValueError("Modal profile does not match verified workspace")
    apps = json.loads(command(MODAL, "app", "list", "--env", "main", "--json"))
    if any(str(r.get("state", r.get("State", ""))).lower() != "stopped"
           or str(r.get("tasks", r.get("Tasks"))) != "0" for r in apps):
        raise ValueError("Another live app exists; do not start competing study work")
    committed = sum(a["reserved_usd"] for a in (ledger or {}).get("attempts", []))
    if usage + TOTAL_NEW_USAGE_CEILING_USD - committed > cap - 0.1:
        raise ValueError("Current workspace usage cannot cover remaining reserved phases and safety margin")
    return proof


def initialize(proof_path):
    PRIVATE.mkdir(parents=True, exist_ok=True)
    state_path = PRIVATE / "study-usage-ledger.json"
    if state_path.exists():
        existing = json.loads((PRIVATE / "source.json").read_text())
        verify_source(existing)
        return json.loads(state_path.read_text())
    proof = read_preflight(proof_path)
    config = load_config()
    source = source_manifest()
    verify_source(source)
    state = {"schema": "lem-ii.production-ledger.v1", "study_id": STUDY_ID,
             "scientific_config_sha256": _scientific_hash(config),
             "source_manifest_sha256": canonical_hash(source), "completed_phases": [], "attempts": [],
             "workspace_limit_usd": 27, "new_usage_ceiling_usd": TOTAL_NEW_USAGE_CEILING_USD,
             "phase_reservations_usd": {phase: PHASE_RESERVATION_USD for phase in PHASES},
             "additional_attempt_reserve_usd": 9.0, "storage_and_other_reserve_usd": 1.5,
             "existing_technical_reservation_untouched": True,
             "initial_workspace_usage_upper_usd": proof["workspace_usage_upper_usd"]}
    for name, value in (("source.json", source), ("base-config.json", config), ("initial-preflight.json", proof),
                        ("study-usage-ledger.json", state)):
        with (PRIVATE / name).open("x") as handle:
            json.dump(value, handle, indent=2)
            handle.flush()
            import os
            os.fsync(handle.fileno())
    from .study import plan
    plan(config, PRIVATE / "plan.json")
    return state


def import_export(archive_path, destination, expected_digest):
    """Validate before copying; immutable complete sessions cannot be replaced."""
    if file_hash(archive_path) != expected_digest:
        raise ValueError("Downloaded export digest mismatch")
    destination = Path(destination)
    stage = archive_path.with_suffix(".unpacked")
    stage.mkdir(exist_ok=False)
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            if (not (stage / info.filename).resolve().is_relative_to(stage.resolve())
                    or (info.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError("Unsafe export member")
        archive.extractall(stage)
    old_ledger_path = destination / "study-usage-ledger.json"
    new_ledger_path = stage / "study-usage-ledger.json"
    if old_ledger_path.exists():
        old_ledger = json.loads(old_ledger_path.read_text())
        new_ledger = json.loads(new_ledger_path.read_text())
        if (old_ledger["source_manifest_sha256"] != new_ledger["source_manifest_sha256"]
                or old_ledger["scientific_config_sha256"] != new_ledger["scientific_config_sha256"]
                or [(a["attempt_id"], a["reserved_usd"]) for a in old_ledger["attempts"]]
                   != [(a["attempt_id"], a["reserved_usd"]) for a in new_ledger["attempts"]]):
            raise ValueError("Downloaded ledger changes the preserved reservations or source binding")
    for new_dir in (stage / "data").glob("study-*"):
        old_dir = destination / "data" / new_dir.name
        if old_dir.exists():
            old = json.loads((old_dir / "manifest.json").read_text())
            new = json.loads((new_dir / "manifest.json").read_text())
            count = old["completed_turns"]
            if (old["binding"] != new["binding"] or new["completed_turns"] < count
                    or new["turn_hashes"][:count] != old["turn_hashes"]):
                raise ValueError("Downloaded continuation changes the existing prefix")
            for path in old_dir.rglob("*"):
                if path.is_file() and (old["status"] == "complete" or "turn-" in str(path.relative_to(old_dir))):
                    if file_hash(path) != file_hash(new_dir / path.relative_to(old_dir)):
                        raise ValueError("Existing dialogue or activation would be overwritten")
    for path in stage.rglob("*"):
        if path.is_file():
            target = destination / path.relative_to(stage)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    return stage


def reconcile_stopped_attempt(attempt_id):
    """Explicit technical recovery: preserve all files/reservations, release only active lock."""
    with (PRIVATE / "controller.lock").open("a+") as process_lock:
        fcntl.flock(process_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = json.loads((PRIVATE / "study-usage-ledger.json").read_text())
        attempt = next(a for a in ledger["attempts"] if a["attempt_id"] == attempt_id)
        local = PRIVATE / "attempts" / attempt_id
        app_record = local / "app.json"
        if not app_record.exists():
            raise ValueError("No owned app ID recorded; inspect provider lifecycle before reconciliation")
        receipt = stop_owned_app(json.loads(app_record.read_text())["app_id"])
        atomic_json(local / "recovery-provider-stopped.json", {**receipt, "verified_epoch": time.time()})
        from .modal_study import VOLUME_NAME, volume, claims
        recovery = local / ("recovery-" + uuid.uuid4().hex)
        subprocess.run([MODAL, "volume", "get", VOLUME_NAME, f"/{STUDY_ID}", str(recovery),
                        "--env", "main"], check=True, timeout=1800)
        # Preserve the downloaded provider snapshot permanently before changing status.
        snapshot = local / (recovery.name + ".zip")
        with zipfile.ZipFile(snapshot, "x", zipfile.ZIP_DEFLATED) as handle:
            for name in ("study-usage-ledger.json", "progress.json"):
                if (recovery / name).exists():
                    handle.write(recovery / name, name)
            for path in (recovery / "data").rglob("*"):
                if path.is_file():
                    handle.write(path, path.relative_to(recovery))
        import_export(snapshot, PRIVATE, file_hash(snapshot))
        state = json.loads((PRIVATE / "study-usage-ledger.json").read_text())
        recovered = next(a for a in state["attempts"] if a["attempt_id"] == attempt_id)
        if recovered["status"] == "active":
            recovered.update(status="failed", error_type="InterruptedAfterReservation",
                             reconciled_after_verified_stop_epoch=time.time())
        atomic_json(PRIVATE / "study-usage-ledger.json", state)
        with volume.batch_upload(force=True) as upload:
            upload.put_file(PRIVATE / "study-usage-ledger.json", f"/{STUDY_ID}/study-usage-ledger.json")
        active = claims.get("active-study", None)
        if active is not None:
            if active["attempt_id"] != attempt_id:
                raise ValueError("Another attempt owns the active-study claim")
            claims.pop("active-study")
        # Permanent attempt claim is NEVER removed. Future work has its own paid reservation.
        return recovered


def run_phase(phase, proof_path, *, allow_partial=False):
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE / "controller.lock").open("a+") as process_lock:
        fcntl.flock(process_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = initialize(proof_path)
        source = json.loads((PRIVATE / "source.json").read_text())
        base = json.loads((PRIVATE / "base-config.json").read_text())
        verify_source(source)
        config = phase_config(base, phase)
        if phase == "confirmation":
            verify_freeze(PRIVATE / "freeze", config, source)
        complete, pending = session_inventory(PRIVATE / "data", config, phase, allow_partial=allow_partial)
        if not pending:
            # A transport error after the 36th valid dialogue can leave the run
            # failed while its phase is complete. Keep its failed reservation.
            if phase not in ledger["completed_phases"]:
                ledger["completed_phases"].append(phase)
                ledger.setdefault("completion_reconciliations", []).append(
                    {"phase": phase, "validated_sessions": len(complete), "epoch": time.time(), "new_model_calls": 0})
                atomic_json(PRIVATE / "study-usage-ledger.json", ledger)
            return {"status": "already_complete", "new_model_calls": 0, "session_count": len(complete)}
        proof = read_preflight(proof_path, ledger)
        attempt_id = "study-run-" + uuid.uuid4().hex
        attempt = reserve_attempt(ledger, attempt_id, phase)
        attempt["preflight_usage_upper_usd"] = proof["workspace_usage_upper_usd"]
        atomic_json(PRIVATE / "study-usage-ledger.json", ledger)
        local = PRIVATE / "attempts" / attempt_id
        local.mkdir(parents=True)
        atomic_json(local / "preflight.json", proof)
        from .modal_study import app, phase_worker, volume, VOLUME_NAME
        import modal
        # Only this controller holds the study lock, and preflight proves no live job.
        # The shared ledger snapshot, including all failed reservations, crosses containers.
        with volume.batch_upload(force=True) as upload:
            upload.put_file(PRIVATE / "study-usage-ledger.json", f"/{STUDY_ID}/study-usage-ledger.json")
            if phase == "confirmation":
                for path in (PRIVATE / "freeze").iterdir():
                    if path.is_file():
                        upload.put_file(path, f"/{STUDY_ID}/freeze/{path.name}")
        started = time.time()
        absolute_deadline = started + STARTUP_SECONDS + WORKER_SECONDS + TERMINATION_SECONDS
        done = threading.Event()
        state = {"phase": "build", "expired": False, "app_id": None}

        def watchdog():
            while not done.wait(1):
                if app.app_id and state["app_id"] is None:
                    state["app_id"] = app.app_id
                    atomic_json(local / "app.json", {"app_id": app.app_id})
                if time.time() >= absolute_deadline or (state["phase"] == "build" and time.time() - started >= STARTUP_SECONDS):
                    state["expired"] = True
                    if app.app_id:
                        try:
                            stop_owned_app(app.app_id)
                            return
                        except Exception as error:
                            atomic_json(local / "stop-error.json", {"error": str(error)[:500]})
                            done.wait(5)

        watcher = threading.Thread(target=watchdog, daemon=True)
        watcher.start()
        result = None
        try:
            with modal.enable_output(), app.run(environment_name="main"):
                state["phase"] = "execute"
                state["app_id"] = app.app_id
                atomic_json(local / "app.json", {"app_id": app.app_id})
                if state["expired"]:
                    raise TimeoutError("Image construction exhausted its independent deadline")
                result = phase_worker.remote(base, phase, attempt_id, source, allow_partial)
                atomic_json(local / "worker-result.json", result)
        except BaseException as error:
            atomic_json(local / "controller-error.json", {"error_type": type(error).__name__, "error": str(error)[:1000]})
            raise
        finally:
            done.set()
            watcher.join(timeout=2)
            if state["app_id"]:
                receipt = stop_owned_app(state["app_id"])
                atomic_json(local / "provider-stopped.json", {**receipt, "verified_epoch": time.time()})
            # An uncertain/failed remote attempt stays active until its durable state
            # is reconciled after verified provider termination; reservations remain.
        archive = local / "export.zip"
        subprocess.run([MODAL, "volume", "get", VOLUME_NAME, f"/{STUDY_ID}/exports/{attempt_id}.zip",
                        str(archive), "--env", "main"], check=True, timeout=600)
        import_export(archive, PRIVATE, result["export_sha256"])
        updated = json.loads((PRIVATE / "study-usage-ledger.json").read_text())
        if ([(a["attempt_id"], a["reserved_usd"]) for a in updated["attempts"]]
                != [(a["attempt_id"], a["reserved_usd"]) for a in ledger["attempts"]]):
            raise ValueError("Remote export changed permanent budget reservations")
        if result["status"] != "complete":
            raise RuntimeError("Phase failed; preserve data and investigate this attempt before continuation")
        checked, pending = session_inventory(PRIVATE / "data", config, phase)
        if len(checked) != 36 or pending:
            raise ValueError("Downloaded phase failed its complete-session inventory")
        atomic_json(local / "verified-local.json", {"phase": phase, "sessions": len(checked), "turns": 24 * len(checked),
                                                   "source_commit": source["commit"], "verified_epoch": time.time()})
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=PHASES)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--resume-inspected-partial", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_phase(args.phase, args.preflight, allow_partial=args.resume_inspected_partial), indent=2))


if __name__ == "__main__":
    main()
