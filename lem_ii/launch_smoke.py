"""Explicit single-launch controller; no automatic retries or study entry point."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import re
from pathlib import Path
import subprocess
import sys
import threading
import time
import zipfile

from .budget import BudgetLedger, cost_proof
from .design import load_config
from .storage import atomic_json, canonical_hash, file_hash

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / ".runtime/technical-budget.json"
MODAL = str(Path(sys.executable).with_name("modal"))


def command(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True, timeout=60).strip()


def stop_owned_app(app_id):
    result = subprocess.run([MODAL, "app", "stop", app_id, "--yes", "--env", "main"],
                            capture_output=True, text=True, timeout=60)
    # Modal1.5.5 reports a server-verified STOPPED lifecycle as exit1.
    already_stopped = result.returncode == 1 and "App is already stopped." in result.stderr
    if result.returncode != 0 and not already_stopped:
        raise RuntimeError(f"Could not establish owned app termination: {result.stderr[:300]}")
    if already_stopped:
        return {"app_id": app_id, "server_lifecycle_stopped": True}
    for _ in range(10):
        rows = json.loads(command(MODAL, "app", "list", "--json", "--env", "main"))
        matches = [row for row in rows if row.get("app_id", row.get("App ID")) == app_id]
        if matches and all(str(row.get("state", row.get("State", ""))).lower() == "stopped"
                           and str(row.get("tasks", row.get("Tasks"))) == "0" for row in matches):
            break
        time.sleep(1)
    else:
        raise RuntimeError("Owned app has not reached a stopped, zero-task state")
    return {"app_id": app_id, "stop_requested_or_already_stopped": True}


def source_binding():
    # Every uploaded executable input must already have a reviewable commit.
    paths = sorted(list((ROOT / "lem_ii").glob("*.py")) + list((ROOT / "lem_ii/config").glob("*.json"))
                   + [ROOT / "lem_ii/requirements.txt"])
    for path in paths:
        rel = str(path.relative_to(ROOT))
        command("git", "ls-files", "--error-unmatch", rel)
        if command("git", "status", "--porcelain", "--", rel):
            raise RuntimeError("Commit executable inputs before cloud execution")
    return {"commit": command("git", "rev-parse", "HEAD"),
            "files": {str(p.relative_to(ROOT)): file_hash(p) for p in paths}}


def validate_preflight(path):
    proof = json.loads(Path(path).read_text())
    age = time.time() - datetime.fromisoformat(proof["verified_at_utc"]).timestamp()
    if not 0 <= age <= 7200:
        raise RuntimeError("Refresh read-only access, prices and budget evidence within two hours of launch")
    workspace = proof["workspace"]
    if not isinstance(workspace, str) or not workspace.strip() or proof["environment"] != "main":
        raise RuntimeError("Invalid preflight workspace/environment")
    usage = proof["workspace_usage_before_usd"]
    if (not isinstance(usage, (int, float)) or not math.isfinite(usage) or usage < 0
            or proof["workspace_gross_usage_limit_usd"] != 5 or usage + 4 > 5):
        raise RuntimeError("Workspace cap cannot cover conservative total task reservations")
    if proof["rates"] != {"L4": 0.000222, "CPU_core": 0.0000131, "RAM_GiB": 0.00000222}:
        raise RuntimeError("Rates changed: do not run with a stale cost envelope")
    if command(MODAL, "profile", "current") != workspace:
        raise RuntimeError("Active Modal profile differs")
    apps = json.loads(command(MODAL, "app", "list", "--env", "main", "--json"))
    if any(str(row.get("state", row.get("State", ""))).lower() != "stopped"
           or str(row.get("tasks", row.get("Tasks"))) != "0" for row in apps):
        raise RuntimeError("Active or recently live apps require a fresh concurrency/billing review")
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", default=str(ROOT / ".runtime/private-preflight.json"))
    args = parser.parse_args()
    config = load_config()
    if not config["execution"]["smoke_only"] or any(config["execution"][key] for key in
            ("study_collection_allowed", "confirmation_generation_allowed", "allow_scientific_analysis", "allow_confirmation_analysis")):
        raise PermissionError("This controller only accepts the unmodified smoke-only authorization")
    proof, source = validate_preflight(args.preflight), source_binding()
    cost_proof()
    ledger = BudgetLedger(LEDGER)
    attempt = ledger.reserve(canonical_hash(config))
    destination = ROOT / ".runtime" / attempt["attempt_id"]
    destination.mkdir()
    atomic_json(destination / "launch.json", {"attempt": attempt, "preflight": proof, "source": source})
    from .modal_entry import app, smoke_worker
    phase = {"value": "build", "app_id": None, "expired": False, "stop_error": None}
    completed = threading.Event()

    def watchdog():
        # Includes image build time. No abandoned detached job if the client waits.
        build_deadline = time.time() + 900
        while not completed.wait(1):
            app_id = app.app_id
            if app_id and phase["app_id"] is None:
                phase["app_id"] = app_id
                atomic_json(destination / "app.json", {"app_id": app_id})
            if time.time() >= attempt["deadline_epoch"] or (phase["value"] == "build" and time.time() >= build_deadline):
                phase["expired"] = True
                if app_id:
                    try:
                        stop_owned_app(app_id)
                        return
                    except Exception as error:
                        phase["stop_error"] = str(error)[:300]
                        atomic_json(destination / "stop-error.json", {"message": phase["stop_error"]})
                        # Retry termination only, never the billable model invocation.
                        if completed.wait(5):
                            return

    watcher = threading.Thread(target=watchdog, daemon=True)
    watcher.start()
    status = "failed"
    try:
        import modal
        with modal.enable_output(), app.run(environment_name="main"):
            phase["value"] = "execute"
            phase["app_id"] = app.app_id
            atomic_json(destination / "app.json", {"app_id": app.app_id})
            if phase["expired"] or time.time() >= attempt["deadline_epoch"]:
                raise RuntimeError("Launch deadline reached during image construction")
            payload = smoke_worker.remote(config, attempt, source)
            (destination / "results.zip").write_bytes(payload)
            with zipfile.ZipFile(destination / "results.zip") as archive:
                for name in archive.namelist():
                    target = (destination / "results" / name).resolve()
                    if not target.is_relative_to((destination / "results").resolve()):
                        raise ValueError("Invalid result archive path")
                archive.extractall(destination / "results")
            summary = json.loads((destination / "results/summary.json").read_text())
            status = "completed" if summary["status"] == "PASS_TECHNICAL_SMOKE" else "failed"
    except Exception as error:
        message = re.sub(r"(?:hf_[A-Za-z0-9]+|a[ks]-[A-Za-z0-9_-]+)", "[REDACTED]", str(error))
        atomic_json(destination / "launch-error.json", {"phase": phase["value"],
                    "type": type(error).__name__, "message": message[:500]})
        raise
    finally:
        completed.set()
        watcher.join(timeout=2)
        # Explicit stop of this one task-owned ephemeral app; never touch ASCR apps.
        if phase["app_id"]:
            stop_owned_app(phase["app_id"])
        ledger.finish(attempt["attempt_id"], status, {"app_id": phase["app_id"], "result_directory": str(destination)})
    print(json.dumps({"status": status, "attempt_id": attempt["attempt_id"], "results": str(destination)}, indent=2))
    if status != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
