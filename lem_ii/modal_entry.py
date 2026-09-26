"""One ephemeral, serial GPU invocation. Use launch_smoke, never deploy this app."""
from pathlib import Path
import modal

PACKAGE = Path(__file__).resolve().parent
app = modal.App("lem-ii-light-technical-20260907")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install_from_requirements(str(PACKAGE / "requirements.txt"))
         .env({"TOKENIZERS_PARALLELISM": "false", "HF_HUB_DISABLE_TELEMETRY": "1"})
         .add_local_dir(PACKAGE, "/root/lem_ii", ignore=["tests/", "evidence/", "examples/", "__pycache__/", "*.md"]))


@app.function(image=image, gpu="L4", cpu=(2.0, 2.0), memory=(16384, 16384),
              max_containers=1, min_containers=0, buffer_containers=0, scaledown_window=2,
              timeout=1800, startup_timeout=900, retries=0, single_use_containers=True)
def smoke_worker(config, attempt, source_manifest):
    import io
    import importlib.metadata
    import json
    import resource
    import tempfile
    import time
    import zipfile
    import numpy as np
    from lem_ii.budget import RuntimeGuard, cost_proof
    from lem_ii.collection import collect_session, smoke_sessions
    from lem_ii.design import validate_config
    from lem_ii.features import compute_technical_features
    from lem_ii.model_adapter import ModelAdapter
    from lem_ii.storage import atomic_json, canonical_hash, file_hash

    started = time.time()
    guard = RuntimeGuard(attempt["deadline_epoch"])
    guard.check_deadline()
    validate_config(config)
    if any(config["execution"][key] for key in ("study_collection_allowed", "confirmation_generation_allowed",
                                                "allow_scientific_analysis", "allow_confirmation_analysis")):
        raise PermissionError("Technical entry point rejects study/confirmation authorization flags")
    if canonical_hash(config) != attempt["config_hash"]:
        raise ValueError("Remote configuration differs from reserved attempt")
    for name, expected in source_manifest["files"].items():
        if file_hash(Path("/root") / name) != expected:
            raise ValueError("Uploaded source checksum mismatch")
    # Atomic, durable claim prevents infrastructure preemption from repeating work.
    claims = modal.Dict.from_name("lem-ii-light-technical-20260907-claims", environment_name="main", create_if_missing=True)
    if not claims.put(attempt["attempt_id"], {"started_epoch": started}, skip_if_exists=True):
        raise RuntimeError("Attempt was already consumed; automatic replay is forbidden")
    root = Path(tempfile.mkdtemp(prefix="lem-ii-smoke-"))
    summary = {"status": "started", "data_kind": "real_model_smoke", "eligible_for_scientific_analysis": False,
               "attempt_id": attempt["attempt_id"], "started_epoch": started,
               "config_sha256": canonical_hash(config), "source": source_manifest, "cost_proof": cost_proof()}
    try:
        adapter = ModelAdapter.load(config)
        summary["model_load_seconds"] = time.time() - started
        summary["adapter"] = adapter.metadata
        summary["software_full"] = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
        guard.check_deadline()
        atomic_json(root / "run.json", summary)
        stores = []
        for spec in smoke_sessions(config):
            stores.append(collect_session(config, spec, adapter, root, source_manifest, guard=guard))
        a, b = [store.records()[0] for store in stores]
        states_a = np.load(stores[0].path / "turn-01/states.npy", allow_pickle=False)
        states_b = np.load(stores[1].path / "turn-01/states.npy", allow_pickle=False)
        summary["reset_check"] = {"identical_first_context": a["input_token_ids"] == b["input_token_ids"],
                                  "identical_first_answer": a["output_token_ids"] == b["output_token_ids"],
                                  "identical_first_states": bool(np.array_equal(states_a, states_b))}
        if not all(summary["reset_check"].values()):
            raise ValueError("Exact same-session-start replay failed")
        primary_index = config["extraction"]["layers"].index(config["extraction"]["primary_layer"])
        primary = stores[0].analysis_record(primary_index)["states"]
        technical = compute_technical_features(primary, config)
        if not technical["finite_features"] or not technical["point_cloud_permutation_invariant"]:
            raise ValueError("Real-state feature validation failed")
        atomic_json(root / "technical_features.json", technical)
        summary["technical_features"] = {key: technical[key] for key in
            ("finite_features", "point_cloud_permutation_invariant", "delay_point_count", "representation")}
        summary["sessions"] = [{"session_id": s.spec.session_id, "turns": len(s.records()),
                                "turn_measurements": [{k: v for k, v in r.items() if k in
                                    {"turn", "input_tokens", "output_tokens", "extraction_seconds", "generation_seconds",
                                     "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes", "stop_reason",
                                     "prefill_passes", "prefill_tokens_including_generation"}} for r in s.records()]}
                               for s in stores]
        summary["activation_bytes"] = sum(p.stat().st_size for p in root.rglob("states.npy"))
        summary["status"] = "PASS_TECHNICAL_SMOKE"
    except Exception as error:
        summary["status"] = "FAILED_TECHNICAL_SMOKE"
        summary["error"] = {"type": type(error).__name__, "message": str(error)[:500]}
    finally:
        summary["finished_epoch"] = time.time()
        summary["worker_seconds"] = summary["finished_epoch"] - started
        summary["peak_host_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        summary["process_cpu_seconds"] = resource.getrusage(resource.RUSAGE_SELF).ru_utime + resource.getrusage(resource.RUSAGE_SELF).ru_stime
        summary["guard"] = guard.snapshot()
        atomic_json(root / "summary.json", summary)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(root))
    return buffer.getvalue()
