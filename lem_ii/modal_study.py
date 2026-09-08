"""Bounded production collection on one L4; never use the technical smoke app."""
from pathlib import Path
import modal

from .production import WORKER_SECONDS, STARTUP_SECONDS

PACKAGE = Path(__file__).resolve().parent
VOLUME_NAME = "lem-ii-light-study-20260908"
VOLUME_ROOT = Path("/study")
app = modal.App("lem-ii-light-study-20260908")
volume = modal.Volume.from_name(VOLUME_NAME, environment_name="main", create_if_missing=True)
claims = modal.Dict.from_name("lem-ii-light-study-20260908-claims", environment_name="main", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install_from_requirements(str(PACKAGE / "requirements.txt"))
         .env({"TOKENIZERS_PARALLELISM": "false", "HF_HUB_DISABLE_TELEMETRY": "1"})
         .add_local_dir(PACKAGE, "/root/lem_ii", ignore=["tests/", "evidence/", "examples/", "__pycache__/", "*.md"]))


@app.function(image=image, gpu="L4", cpu=(2.0, 2.0), memory=(16384, 16384),
              volumes={str(VOLUME_ROOT): volume}, max_containers=1, min_containers=0,
              buffer_containers=0, scaledown_window=2, timeout=WORKER_SECONDS,
              startup_timeout=STARTUP_SECONDS, retries=0, single_use_containers=True)
def phase_worker(base, phase, attempt_id, source, allow_partial=False):
    import json
    import time
    import zipfile
    from .production import collect_phase, verify_source
    from .storage import atomic_json, file_hash

    verify_source(source)
    if not claims.put(attempt_id, {"consumed_epoch": time.time()}, skip_if_exists=True):
        raise RuntimeError("Attempt already consumed; provider replay cannot repeat model work")
    if not claims.put("active-study", {"attempt_id": attempt_id}, skip_if_exists=True):
        raise RuntimeError("A study worker already owns the durable serial lock")
    root = VOLUME_ROOT / "lem-ii-light-full-20260908"
    result = {"attempt_id": attempt_id, "phase": phase, "status": "failed"}
    try:
        result = collect_phase(root, base, phase, attempt_id, source, persist=volume.commit,
                               allow_partial=allow_partial, pre_reserved=True)
    except BaseException as error:
        result.update(error_type=type(error).__name__, error=str(error)[:1000])
    finally:
        atomic_json(root / f"{attempt_id}.worker.json", result)
        exports = root / "exports"
        exports.mkdir(exist_ok=True)
        archive = exports / f"{attempt_id}.zip"
        with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as handle:
            # One shared ledger plus all complete/partial data from this phase.
            for name in ("study-usage-ledger.json", "progress.json", f"{attempt_id}.worker.json"):
                path = root / name
                if path.exists():
                    handle.write(path, name)
            from .design import build_sessions
            for spec in (s for s in build_sessions(base) if s.split == phase):
                directory = root / "data" / spec.session_id
                if directory.exists():
                    for path in directory.rglob("*"):
                        if path.is_file():
                            handle.write(path, path.relative_to(root))
            for path in (root / "data" / "runs").rglob("*"):
                if path.is_file():
                    handle.write(path, path.relative_to(root))
        volume.commit()
        result["export_sha256"] = file_hash(archive)
        claims.pop("active-study")
    return result
