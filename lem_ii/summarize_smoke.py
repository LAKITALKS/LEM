"""Verify returned real checkpoints and estimate the unexecuted 108-dialog scope."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np
from .budget import cost_proof
from .collection import smoke_sessions
from .design import load_config
from .features import compute_technical_features
from .storage import SessionStore, atomic_json, canonical_hash, file_hash


def billing_costs(rows, app_id, worker_seconds):
    """Use only this app's gross provider rows; keep metering provisional."""
    selected = [row for row in rows if row["object_id"] == app_id]
    if not selected:
        raise ValueError("Provider report has no rows for this technical app")
    costs = {}
    for row in selected:
        cost = float(row["cost"])
        if not math.isfinite(cost) or cost < 0:
            raise ValueError("Invalid provider charge")
        costs[row["resource"]] = costs.get(row["resource"], 0.0) + cost
    if costs.get("L4", 0) <= 0:
        raise ValueError("Provider report does not yet include positive L4 usage")
    gpu_seconds = costs["L4"] / cost_proof()["gpu_usd_per_second"]
    return {"app_id": app_id, "resource_costs_usd": costs, "reported_gross_usage_usd": sum(costs.values()),
            "gpu_billable_seconds_inferred_from_rate": gpu_seconds,
            "outside_worker_gpu_seconds": max(2.0, gpu_seconds - worker_seconds),
            "status": "provider_reported_usage_snapshot_not_final_invoice",
            "metering_can_lag": True, "credits_not_subtracted": True}


def summarize(result_directory, config, billing_rows=None):
    root = Path(result_directory)
    summary = json.loads((root / "summary.json").read_text())
    if summary["status"] != "PASS_TECHNICAL_SMOKE" or summary["config_sha256"] != canonical_hash(config):
        raise ValueError("Only a complete, configuration-matched real smoke may be summarized")
    stores = []
    for spec in smoke_sessions(config):
        manifest = json.loads((root / spec.session_id / "manifest.json").read_text())
        if (manifest["binding"]["data_kind"] != "real_model_smoke"
                or manifest["binding"]["config_sha256"] != canonical_hash(config)
                or manifest["binding"]["run"] != summary["source"]):
            raise ValueError("Returned session source/configuration/data-kind binding mismatch")
        stores.append(SessionStore(root, spec, manifest["binding"], resume=True))
    index = config["extraction"]["layers"].index(config["extraction"]["primary_layer"])
    features = compute_technical_features(stores[0].analysis_record(index)["states"], config)
    if not features["finite_features"] or not features["point_cloud_permutation_invariant"]:
        raise ValueError("Local recomputation of real-state features failed")
    remote_features = json.loads((root / "technical_features.json").read_text())
    for block, values in features["features"].items():
        np.testing.assert_allclose(values, remote_features["features"][block], rtol=1e-5, atol=1e-7)
    rows = stores[0].records()
    full_seconds = sum(r["extraction_seconds"] + r["generation_seconds"] for r in rows)
    resource_rate = cost_proof()["resource_rate_usd_per_second"]
    # All states are collected with the answers. Label/order controls reuse them;
    # zero additional planned GPU dialogues, not zero postprocessing CPU work.
    cold = summary["model_load_seconds"]
    billing = None
    overhead = 2.0
    if billing_rows is not None:
        app_id = json.loads((root.parent / "app.json").read_text())["app_id"]
        billing = billing_costs(billing_rows, app_id, summary["worker_seconds"])
        overhead = billing["outside_worker_gpu_seconds"]
    per_dialogue_estimate = (full_seconds + cold + overhead) * resource_rate
    base = 108 * per_dialogue_estimate
    conservative = base * 2 + 1.0  # 2x observed compute + explicit CPU-analysis/build reserve
    report = {"status": summary["status"], "scientific_results": False,
              "confirmation_generated_or_evaluated": False,
              "source_commit": summary["source"]["commit"], "config_sha256": summary["config_sha256"],
              "model_revision": summary["adapter"]["model_revision"],
              "session_turns": [len(s.records()) for s in stores],
              "full_24_turn_measured_seconds": full_seconds, "model_load_seconds": cold,
              "worker_seconds": summary["worker_seconds"],
              "input_tokens_first_last": [rows[0]["input_tokens"], rows[-1]["input_tokens"]],
              "input_token_growth": [r["input_tokens"] for r in rows],
              "full_dialogue_output_tokens": sum(r["output_tokens"] for r in rows),
              "full_dialogue_prefill_tokens_two_passes": sum(r["prefill_tokens_including_generation"] for r in rows),
              "full_dialogue_generation_tokens_per_second": sum(r["output_tokens"] for r in rows) / sum(r["generation_seconds"] for r in rows),
              "peak_gpu_allocated_bytes": max(r["peak_cuda_allocated_bytes"] for r in rows),
              "peak_gpu_reserved_bytes": max(r["peak_cuda_reserved_bytes"] for r in rows),
              "peak_host_rss_bytes": summary["peak_host_rss_bytes"], "activation_bytes": summary["activation_bytes"],
              "reset_check": summary["reset_check"], "local_feature_recomputation_matches": True,
              "technical_resource_time_cost_estimate_usd": (summary["worker_seconds"] + 2) * resource_rate,
              "provider_billing_status": billing["status"] if billing else "pending_separate_provider_report_not_a_measured_invoice",
              "provider_billing": billing,
              "study_projection": {"dialogs": 108, "responses": 2592, "separate_extraction_prefills": 2592,
                  "generation_prefills": 2592, "additional_gpu_control_dialogues": 0,
                  "control_scope": "199 profile-label refits and order counterchecks reuse stored contexts/states; local CPU analysis",
                  "assumed_cold_model_loads": 108, "base_resource_cost_usd": base,
                  "outside_worker_overhead_seconds_per_dialogue": overhead,
                  "overhead_basis": "provider_L4_billable_seconds_minus_worker_seconds" if billing else "prespecified_2_second_idle_only_pending_provider_data",
                  "each_36_dialogue_phase_base_usd": base / 3,
                  "conservative_compute_multiplier": 2, "cpu_analysis_and_build_reserve_usd": 1.0,
                  "conservative_remaining_study_estimate_usd": conservative,
                  "within_unreleased_25_usd": conservative <= 25,
                  "total_with_full_5_usd_technical_reserve": conservative + 5,
                  "uncertainty": "One full24-turn developmental-topic smoke, no across-topic throughput distribution. No guarantee; recheck before each phase.",
                  "authorization": "not_authorized_to_execute"},
              "evidence_hashes": {str(p.relative_to(root)): file_hash(p) for p in root.rglob("*") if p.is_file()}}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--billing-report", type=Path)
    args = parser.parse_args()
    rows = json.loads(args.billing_report.read_text()) if args.billing_report else None
    report = summarize(args.result_directory, load_config(), rows)
    atomic_json(args.output, report)
    print(json.dumps({"output": str(args.output), "status": report["status"], "projection": report["study_projection"]}, indent=2))


if __name__ == "__main__":
    main()
