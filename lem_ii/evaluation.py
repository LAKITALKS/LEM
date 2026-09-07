"""Executable fixture fit/tune/freeze/predict and paired diagnostic evaluation.

This command deliberately has no confirmation or scientific-data input option.
All reported validation numbers demonstrate software behavior only.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any
import uuid

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from .features import FeaturePipeline, LeakageError, METHODS, validate_records
from .fixtures import make_fixture_records


CONTRASTS = (
    ("activation_added", METHODS[0], METHODS[1]),
    ("time_geometry_added", METHODS[1], METHODS[2]),
    ("topology_added", METHODS[2], METHODS[3]),
)


def configuration_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def analysis_provenance() -> dict:
    root = Path(__file__).parent
    return {"code_file_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                  for name in ("features.py", "evaluation.py", "fixtures.py")},
            "software_versions": {name: importlib.metadata.version(name)
                                  for name in ("numpy", "scipy", "scikit-learn", "ripser", "joblib")}}


def analysis_configuration_hash(config: dict) -> str:
    """All scientific settings frozen; only explicit future authorization may change."""
    frozen = copy.deepcopy(config)
    for gate in ("study_collection_allowed", "confirmation_generation_allowed",
                 "confirmation_evaluation_allowed", "smoke_only",
                 "allow_scientific_analysis", "allow_confirmation_analysis"):
        frozen.setdefault("execution", {}).pop(gate, None)
    return configuration_hash(frozen)


def join_record_inputs(dialogue_rows: list[dict], label_rows: list[dict],
                       states: dict[str, np.ndarray], config: dict | None = None) -> list[dict]:
    """Explicit one-to-one join: feature records are never built by dict merging.

    Dialogue rows: dialogue_id, contexts. Label rows: dialogue_id, profile_id,
    regime, split, topic_family, formulation_family, data_kind, eligibility.
    No metadata is appended to visible contexts.
    """
    dialogue_ids = [row["dialogue_id"] for row in dialogue_rows]
    label_ids = [row["dialogue_id"] for row in label_rows]
    if len(set(dialogue_ids)) != len(dialogue_ids) or len(set(label_ids)) != len(label_ids):
        raise LeakageError("Duplicate IDs in persisted dialogue/label join")
    if set(dialogue_ids) != set(label_ids) or set(dialogue_ids) != set(states):
        raise LeakageError("Dialogues, labels and state files require exact matching IDs")
    labels = {row["dialogue_id"]: row for row in label_rows}
    records = []
    for row in dialogue_rows:
        identifier = row["dialogue_id"]
        label = labels[identifier]
        records.append({"dialogue_id": identifier, "contexts": row["contexts"],
                        "states": states[identifier],
                        **{key: label[key] for key in (
                            "profile_id", "regime", "split", "topic_family", "formulation_family",
                            "eligible_for_scientific_analysis", "data_kind")}})
    validate_records(records, config=config)
    return records


def split_records(records: list[dict], config: dict | None = None) -> tuple[list[dict], list[dict]]:
    validate_records(records, config=config)
    if any(row["data_kind"] == "real_model_smoke" for row in records):
        raise LeakageError("Technical smoke is never classified")
    development = [row for row in records if row["split"] == "development"]
    validation = [row for row in records if row["split"] == "validation"]
    if len(development) + len(validation) != len(records):
        raise LeakageError("Only development and validation records may enter this analysis")
    if not development or not validation:
        raise LeakageError("Both development and validation are required")
    if records[0]["data_kind"] == "real_model_study":
        expected = int(config["design"]["profiles"]) * 2
        if len(development) != expected or len(validation) != expected:
            raise LeakageError("Complete prespecified development and validation session sets are required")
    if {r["profile_id"] for r in development} != {r["profile_id"] for r in validation}:
        raise LeakageError("This design repeats the same known profile bundles across phases")
    if {r["regime"] for r in development} != {r["regime"] for r in validation}:
        raise LeakageError("Regime classes must match across phases")
    for phase in (development, validation):
        profiles = {r["profile_id"] for r in phase}
        for profile in profiles:
            bundle = [r for r in phase if r["profile_id"] == profile]
            if len(bundle) != 2 or len({r["topic_family"] for r in bundle}) != 2:
                raise LeakageError("Each profile must contribute both genuine topic sessions in each phase")
    return development, validation


def labels_for(records: list[dict], mapping: dict[str, str] | None = None) -> np.ndarray:
    return np.asarray([mapping[r["profile_id"]] if mapping is not None else r["regime"] for r in records])


def fit_tune_matrices(train_matrices: dict, validation_matrices: dict,
                      train_labels: np.ndarray, validation_labels: np.ndarray,
                      config: dict) -> tuple[dict, dict]:
    """Each method gets exactly the same three C candidates; ties choose smallest."""
    options = config.get("analysis", {})
    candidates = sorted(float(value) for value in options.get("C_grid", [0.1, 1.0, 10.0]))
    if candidates != [0.1, 1.0, 10.0]:
        raise ValueError("The fixed equal tuning budget is C=[0.1,1,10]")
    seed = config.get("seeds", {}).get("classifier", 1702)
    models, tuning = {}, {}
    for method in METHODS:
        best_score = -np.inf
        trials = []
        for c_value in candidates:
            model = LogisticRegression(C=c_value, solver="lbfgs",
                                       max_iter=2000, class_weight="balanced", random_state=seed)
            model.fit(train_matrices[method], train_labels)
            if int(model.n_iter_.max()) >= model.max_iter:
                raise RuntimeError(f"Classifier failed convergence for {method}, C={c_value}")
            prediction = model.predict(validation_matrices[method])
            score = float(balanced_accuracy_score(validation_labels, prediction))
            trials.append({"C": c_value, "validation_balanced_accuracy": score})
            if score > best_score + 1e-12:
                best_score = score
                models[method] = model
        tuning[method] = {"trials": trials, "selected_C": float(models[method].C),
                          "selection_split": "validation", "tie_rule": "smallest_C"}
    return models, tuning


def fit_and_tune(records: list[dict], config: dict, endpoint: int = 24) -> dict:
    development, validation = split_records(records, config)
    pipeline = FeaturePipeline(config).fit(development, endpoint)
    train_matrices = pipeline.transform(development)
    validation_matrices = pipeline.transform(validation)
    models, tuning = fit_tune_matrices(train_matrices, validation_matrices,
                                      labels_for(development), labels_for(validation), config)
    return {"pipeline": pipeline, "models": models, "tuning": tuning,
            "endpoint": endpoint, "config_hash": configuration_hash(config),
            "analysis_config_hash": analysis_configuration_hash(config),
            "development_ids": [r["dialogue_id"] for r in development],
            "validation_ids": [r["dialogue_id"] for r in validation],
            "fitted_topic_families": sorted({r["topic_family"] for r in records}),
            "fitted_formulation_families": sorted({r["formulation_family"] for r in records}),
            "profile_regimes": {r["profile_id"]: r["regime"] for r in records},
            "data_kind": records[0]["data_kind"],
            "eligible_for_scientific_analysis": records[0]["eligible_for_scientific_analysis"]}


def predict(fitted: dict, records: list[dict], *, time_shuffle=False) -> dict[str, np.ndarray]:
    matrices = fitted["pipeline"].transform(records, time_shuffle=time_shuffle)
    return {method: fitted["models"][method].predict(matrices[method]) for method in METHODS}


def clustered_indices(records: list[dict], rng: np.random.Generator) -> np.ndarray:
    """Stratified profile bootstrap: resample bundles, keep both sessions together."""
    bundles = {}
    for index, record in enumerate(records):
        bundles.setdefault(record["profile_id"], []).append(index)
    by_regime: dict[str, list[str]] = {}
    for profile, indices in bundles.items():
        regimes = {records[index]["regime"] for index in indices}
        if len(regimes) != 1 or len(indices) != 2:
            raise LeakageError("Bootstrap units must be two-session stable-regime profile bundles")
        by_regime.setdefault(next(iter(regimes)), []).append(profile)
    output = []
    for regime in sorted(by_regime):
        profiles = sorted(by_regime[regime])
        for profile in rng.choice(profiles, size=len(profiles), replace=True):
            output.extend(bundles[profile])
    return np.asarray(output, dtype=int)


def paired_cluster_intervals(records: list[dict], predictions: dict[str, np.ndarray],
                             config: dict, *, replicates: int | None = None) -> dict:
    labels = labels_for(records)
    options = config.get("analysis", {})
    n_bootstrap = int(replicates or options.get("bootstrap_replicates", 2000))
    if n_bootstrap < 2:
        raise ValueError("At least two bootstrap resamples required")
    alpha = float(options.get("alpha", 0.05))
    tail = alpha / (2 * len(CONTRASTS))
    rng = np.random.default_rng(config.get("seeds", {}).get("bootstrap", 1703))
    samples = {name: [] for name, _, _ in CONTRASTS}
    for _ in range(n_bootstrap):
        indices = clustered_indices(records, rng)
        scores = {method: balanced_accuracy_score(labels[indices], values[indices])
                  for method, values in predictions.items()}
        for name, base, added in CONTRASTS:
            samples[name].append(scores[added] - scores[base])
    result = {}
    for name, base, added in CONTRASTS:
        delta = balanced_accuracy_score(labels, predictions[added]) - balanced_accuracy_score(labels, predictions[base])
        bounds = np.quantile(samples[name], [tail, 1 - tail])
        result[name] = {"base": base, "added": added, "balanced_accuracy_difference": float(delta),
                        "interval": bounds.tolist(), "confidence_level": 1 - alpha / len(CONTRASTS),
                        "bootstrap_replicates": n_bootstrap,
                        "resampling_unit": "profile_bundle_with_both_sessions_stratified_by_regime",
                        "scope": "descriptive_validation_conditional_on_fixed_topic_families"}
    return result


def profile_label_mapping(records: list[dict], rng: np.random.Generator) -> dict[str, str]:
    """One permuted regime assignment per entire profile across all phases."""
    original = {}
    for row in records:
        if row["profile_id"] in original and original[row["profile_id"]] != row["regime"]:
            raise LeakageError("Inconsistent regime within a profile")
        original[row["profile_id"]] = row["regime"]
    profiles = sorted(original)
    shuffled = rng.permutation([original[profile] for profile in profiles])
    return dict(zip(profiles, shuffled.tolist()))


def simultaneous_grid_intervals(records: list[dict], predictions_by_endpoint: dict[int, dict],
                                config: dict, *, replicates: int | None = None) -> dict:
    """Conservative Bonferroni family across all 3 contrasts x 4 fixed endpoints."""
    endpoints = sorted(predictions_by_endpoint)
    declared = sorted(config.get("features", {}).get("endpoints", [12, 16, 20, 24]))
    if endpoints != declared:
        raise ValueError("Intervals require every predeclared endpoint; no favorable grid selection")
    options = config.get("analysis", {})
    count = int(replicates or options.get("bootstrap_replicates", 2000))
    alpha = float(options.get("alpha", 0.05))
    family = len(endpoints) * len(CONTRASTS)
    tail = alpha / (2 * family)
    samples = {(point, name): [] for point in endpoints for name, _, _ in CONTRASTS}
    labels = labels_for(records)
    rng = np.random.default_rng(config.get("seeds", {}).get("bootstrap", 1703))
    for _ in range(count):
        indices = clustered_indices(records, rng)
        for point in endpoints:
            scores = {method: balanced_accuracy_score(labels[indices], values[indices])
                      for method, values in predictions_by_endpoint[point].items()}
            for name, base, added in CONTRASTS:
                samples[point, name].append(scores[added] - scores[base])
    return {"scope": "descriptive_validation_contrast_grid", "family_size": family,
            "confidence_level_each": 1 - alpha / family, "bootstrap_replicates": count,
            "intervals": {str(point): {name: np.quantile(samples[point, name], [tail, 1 - tail]).tolist()
                                        for name, _, _ in CONTRASTS} for point in endpoints}}


def label_countercheck(fitted: dict, records: list[dict], config: dict,
                      *, replicates: int | None = None) -> dict:
    development, validation = split_records(records, config)
    pipeline = fitted["pipeline"]
    train_matrices, val_matrices = pipeline.transform(development), pipeline.transform(validation)
    options = config.get("analysis", {})
    declared = int(options.get("permutation_replicates", 199))
    count = int(declared if replicates is None else replicates)
    if count < 1:
        raise ValueError("At least one diagnostic label randomization required")
    rng = np.random.default_rng(config.get("seeds", {}).get("permutation", 1704))
    scores = {method: [] for method in METHODS}
    for _ in range(count):
        mapping = profile_label_mapping(records, rng)
        train_labels, val_labels = labels_for(development, mapping), labels_for(validation, mapping)
        # Transformations are label independent and stay fitted on development;
        # every classifier and its equal C-selection budget is rerun each time.
        models, _ = fit_tune_matrices(train_matrices, val_matrices, train_labels, val_labels, config)
        for method in METHODS:
            scores[method].append(float(balanced_accuracy_score(val_labels, models[method].predict(val_matrices[method]))))
    return {"kind": "diagnostic_profile_bundle_label_randomization",
            "replicates": count, "predeclared_replicates": declared,
            "diagnostic_only_override": count != declared,
            "scope": "validation_tuning_is_repeated;not_a_confirmation_p_value",
            "balanced_accuracy_samples": scores}


def time_countercheck(fitted: dict, records: list[dict]) -> dict:
    normal_matrices = fitted["pipeline"].transform(records)
    shuffled_matrices = fitted["pipeline"].transform(records, time_shuffle=True)
    difference = normal_matrices[METHODS[1]] - shuffled_matrices[METHODS[1]]
    maximum_difference = float(np.max(np.abs(difference.data))) if difference.nnz else 0.0
    if maximum_difference > 1e-9:
        raise LeakageError("The time countercheck changed the common static baseline")
    normal = {method: fitted["models"][method].predict(normal_matrices[method]) for method in METHODS}
    shuffled = {method: fitted["models"][method].predict(shuffled_matrices[method]) for method in METHODS}
    labels = labels_for(records)
    return {"kind": "inference_time_predecessor_order_shuffle_endpoint_fixed_no_refit", "delay_embedding_rebuilt": True,
            "current_state_fixed": True, "entire_static_basis_invariant": True,
            "static_max_absolute_difference": maximum_difference,
            "scope": "sensitivity_of_frozen_classifier;distribution_shift_is_possible",
            "methods": {method: {"normal_balanced_accuracy": float(balanced_accuracy_score(labels, normal[method])),
                                  "shuffled_balanced_accuracy": float(balanced_accuracy_score(labels, shuffled[method])),
                                  "changed_predictions": int(np.sum(normal[method] != shuffled[method]))}
                        for method in METHODS}}


def earliest_recognition(scores: dict[int, float], threshold=0.70, consecutive=2) -> dict:
    endpoints = sorted(scores)
    for start in range(len(endpoints) - consecutive + 1):
        chosen = endpoints[start:start + consecutive]
        if all(scores[point] >= threshold for point in chosen):
            return {"earliest_endpoint": chosen[0], "stability_confirmed_at": chosen[-1],
                    "status": "reached", "threshold": threshold, "consecutive": consecutive}
    return {"earliest_endpoint": None, "stability_confirmed_at": None,
            "status": "not_reached_in_observed_grid", "threshold": threshold, "consecutive": consecutive}


def freeze_artifact(fitted: dict, output: Path) -> dict:
    """Write a reusable fitted artifact and separate integrity manifest."""
    output.mkdir(parents=True, exist_ok=True)
    prefix = "fixture" if fitted["data_kind"] == "synthetic_fixture" else "study"
    path = output / f"{prefix}_endpoint_{fitted['endpoint']}.joblib"
    if prefix == "study" and (path.exists() or (output / "confirmation_grid.lock").exists()):
        raise LeakageError("A frozen scientific artifact or reserved confirmation grid cannot be overwritten")
    joblib.dump(fitted, path, compress=3)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {key: fitted[key] for key in ("endpoint", "config_hash", "analysis_config_hash", "development_ids", "validation_ids",
                                           "fitted_topic_families", "fitted_formulation_families", "profile_regimes",
                                           "tuning", "data_kind", "eligible_for_scientific_analysis")}
    manifest.update({"artifact_file": path.name, "artifact_sha256": digest,
                     "confirmation_accessed": False, "prediction_uses_frozen_transformations": True})
    manifest.update(analysis_provenance())
    (output / f"{prefix}_endpoint_{fitted['endpoint']}.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_frozen_artifact(path: Path, manifest_path: Path, config: dict) -> dict:
    """Only load a trusted locally-produced joblib after validating its digest."""
    manifest = json.loads(manifest_path.read_text())
    if manifest["artifact_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
        raise LeakageError("Frozen artifact digest mismatch")
    if manifest["analysis_config_hash"] != analysis_configuration_hash(config):
        raise LeakageError("Frozen artifact configuration changed")
    provenance = analysis_provenance()
    if any(manifest.get(field) != provenance[field] for field in ("code_file_sha256", "software_versions")):
        raise LeakageError("Frozen analysis code or software versions changed")
    # joblib is executable serialization; this function is not an untrusted-file loader.
    fitted = joblib.load(path)
    if (fitted["analysis_config_hash"] != manifest["analysis_config_hash"] or
            analysis_configuration_hash(fitted["pipeline"].config) != manifest["analysis_config_hash"]):
        raise LeakageError("Frozen payload and integrity manifest disagree")
    return fitted


def _validate_confirmation_input(records: list[dict], config: dict) -> None:
    gates = config.get("execution", {})
    if not (gates.get("allow_scientific_analysis", False) and gates.get("allow_confirmation_analysis", False)):
        raise LeakageError("Independent confirmation requires separate future authorization")
    validate_records(records, config=config)
    if not records or any(r["split"] != "confirmation" or r["data_kind"] != "real_model_study" for r in records):
        raise LeakageError("Confirmation requires eligible genuine reserved study records")
    if len(records) != int(config["design"]["profiles"]) * 2:
        raise LeakageError("The complete 36-dialogue confirmation set is required")
    clustered_indices(records, np.random.default_rng(0))  # Validate genuine two-session bundles before reservation.


def _validate_confirmation_artifact(fitted: dict, records: list[dict]) -> None:
    if fitted["data_kind"] != "real_model_study" or not fitted["eligible_for_scientific_analysis"]:
        raise LeakageError("Fixture/smoke artifacts cannot be promoted to scientific confirmation")
    used = set(fitted["development_ids"]) | set(fitted["validation_ids"])
    if used & {r["dialogue_id"] for r in records}:
        raise LeakageError("Confirmation dialogue overlaps fitted data")
    if set(fitted["fitted_topic_families"]) & {r["topic_family"] for r in records}:
        raise LeakageError("Confirmation topic overlaps fitted data")
    if set(fitted["fitted_formulation_families"]) & {r["formulation_family"] for r in records}:
        raise LeakageError("Confirmation formulation overlaps fitted data")
    if fitted["profile_regimes"] != {r["profile_id"]: r["regime"] for r in records}:
        raise LeakageError("Confirmation must preserve known profile-regime assignments")


def _reserve_confirmation_once(artifact_directory: Path, output_path: Path) -> None:
    if output_path.exists():
        raise LeakageError("Confirmation output already exists; do not rerun or overwrite")
    lock_path = artifact_directory / "confirmation_grid.lock"
    try:
        with lock_path.open("x") as lock:
            lock.write(json.dumps({"output": str(output_path.resolve()), "status": "reserved_once"}) + "\n")
    except FileExistsError as exc:
        raise LeakageError("This frozen study grid already has a reserved confirmation attempt") from exc


def predict_confirmation_grid_once(artifact_directory: Path, records: list[dict], config: dict,
                                   output_path: Path) -> dict:
    """Future-only full-grid confirmation using four previously frozen artifacts.

    Directory must contain study_endpoint_{12,16,20,24}.joblib and the matching
    .manifest.json files written by freeze_artifact. One shared grid lock is
    reserved before prediction; no fit/tune step exists in this path.
    """
    _validate_confirmation_input(records, config)
    endpoints = config["features"]["endpoints"]
    if endpoints != [12, 16, 20, 24]:
        raise LeakageError("Full confirmation requires the exact prespecified grid")
    fitted_grid, hashes = {}, {}
    common_fit = None
    for endpoint in endpoints:
        path = artifact_directory / f"study_endpoint_{endpoint}.joblib"
        manifest_path = artifact_directory / f"study_endpoint_{endpoint}.manifest.json"
        fitted = load_frozen_artifact(path, manifest_path, config)
        _validate_confirmation_artifact(fitted, records)
        if fitted["endpoint"] != endpoint:
            raise LeakageError("Frozen endpoint disagrees with its grid position")
        fit_ids = (tuple(fitted["development_ids"]), tuple(fitted["validation_ids"]))
        if common_fit is not None and common_fit != fit_ids:
            raise LeakageError("The endpoint grid used different development/validation cases")
        common_fit = fit_ids
        fitted_grid[endpoint] = fitted
        hashes[endpoint] = hashlib.sha256(path.read_bytes()).hexdigest()
    _reserve_confirmation_once(artifact_directory, output_path)
    predictions, scores = {}, {}
    labels = labels_for(records)
    for endpoint in endpoints:
        fitted_grid[endpoint]["pipeline"].config = copy.deepcopy(config)
        predictions[endpoint] = predict(fitted_grid[endpoint], records)
        scores[endpoint] = {method: float(balanced_accuracy_score(labels, values))
                            for method, values in predictions[endpoint].items()}
    options = config["analysis"]
    report = {
        "data_kind": "real_model_study", "scope": "independent_frozen_full_grid_confirmation",
        "artifact_sha256_by_endpoint": hashes, "execution_config_hash": configuration_hash(config),
        "analysis_config_hash": analysis_configuration_hash(config),
        "dialogue_ids": [r["dialogue_id"] for r in records],
        "predictions": {endpoint: {method: values.tolist() for method, values in predicted.items()}
                        for endpoint, predicted in predictions.items()},
        "balanced_accuracy_by_endpoint": scores,
        "primary_paired_differences": paired_cluster_intervals(records, predictions[24], config),
        "simultaneous_grid_contrast_intervals": simultaneous_grid_intervals(records, predictions, config),
        "recognition_prespecified_point_estimate_summary": {
            method: earliest_recognition({endpoint: scores[endpoint][method] for endpoint in endpoints},
                                         options["recognition_threshold"], options["recognition_consecutive"])
            for method in METHODS},
        "fitting_or_tuning_performed": False, "topic_population_inference": False,
    }
    for result in report["primary_paired_differences"].values():
        result["scope"] = "confirmation_conditional_on_fixed_topic_families"
    report["simultaneous_grid_contrast_intervals"]["scope"] = "confirmation_secondary_fixed_grid_contrast_family"
    with output_path.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    return report


def predict_confirmation_once(artifact_path: Path, manifest_path: Path, records: list[dict],
                              config: dict, output_path: Path) -> dict:
    """Future-only locked prediction: no fitting, tuning, threshold choice or retry.

    Both execution gates must be explicitly enabled by a separately authorized
    future operator. This turn's configuration disables both and CLI cannot call it.
    """
    _validate_confirmation_input(records, config)
    if output_path.exists():
        raise LeakageError("Confirmation output already exists; do not rerun or overwrite")
    fitted = load_frozen_artifact(artifact_path, manifest_path, config)
    _validate_confirmation_artifact(fitted, records)
    if fitted["endpoint"] != 24:
        raise LeakageError("This locked confirmation entry point implements the primary endpoint only")
    # One lock per frozen artifact, independent of the caller's output filename.
    # A failed locked attempt needs human investigation, never an automatic retry.
    _reserve_confirmation_once(artifact_path.parent, output_path)
    fitted["pipeline"].config = copy.deepcopy(config)
    predictions = predict(fitted, records)
    report = {"data_kind": "real_model_study", "scope": "primary_endpoint_24_confirmation",
              "artifact_sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
              "execution_config_hash": configuration_hash(config),
              "analysis_config_hash": analysis_configuration_hash(config),
              "predictions": {method: values.tolist() for method, values in predictions.items()},
              "dialogue_ids": [r["dialogue_id"] for r in records],
              "primary_paired_differences": paired_cluster_intervals(records, predictions, config)}
    for result in report["primary_paired_differences"].values():
        result["scope"] = "confirmation_conditional_on_fixed_topic_families"
    # Exclusive creation means parallel duplicate attempts cannot replace results.
    with output_path.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    return report


def run_fixture_analysis(config: dict, output: Path, *, diagnostic_permutations: int | None = None,
                         diagnostic_bootstrap: int | None = None) -> dict[str, Any]:
    records = make_fixture_records(seed=config.get("seeds", {}).get("fixtures", 1706))
    development, validation = split_records(records, config)
    endpoints = config.get("features", {}).get("endpoints", [12, 16, 20, 24])
    report: dict[str, Any] = {"data_kind": "synthetic_fixture", "eligible_for_scientific_analysis": False,
                              "scientific_results": False, "confirmation_generated_or_evaluated": False,
                              "config_hash": configuration_hash(config), "endpoints": {},
                              "interpretation": "Software exercise only; validation is used for tuning and is descriptive."}
    report.update({"run_id": f"fixture-analysis-{uuid.uuid4()}", "model_invoked": False,
                   "fixture_seed": config.get("seeds", {}).get("fixtures", 1706), **analysis_provenance()})
    fitted_at_primary = None
    grid_predictions = {}
    for endpoint in endpoints:
        fitted = fit_and_tune(records, config, endpoint)
        validation_predictions = predict(fitted, validation)
        grid_predictions[endpoint] = validation_predictions
        development_predictions = predict(fitted, development)
        manifest = freeze_artifact(fitted, output)
        report["endpoints"][endpoint] = {
            "validation_balanced_accuracy": {method: float(balanced_accuracy_score(labels_for(validation), values))
                                             for method, values in validation_predictions.items()},
            "development_balanced_accuracy": {method: float(balanced_accuracy_score(labels_for(development), values))
                                              for method, values in development_predictions.items()},
            "tuning": fitted["tuning"], "artifact_sha256": manifest["artifact_sha256"]}
        if endpoint == 24:
            fitted_at_primary = fitted
            report["primary_paired_differences"] = paired_cluster_intervals(validation, validation_predictions, config,
                                                                           replicates=diagnostic_bootstrap)
    if fitted_at_primary is None:
        raise ValueError("The primary endpoint 24 must be present in the fixed grid")
    options = config.get("analysis", {})
    report["recognition_descriptive"] = {
        method: earliest_recognition({int(endpoint): row["validation_balanced_accuracy"][method]
                                      for endpoint, row in report["endpoints"].items()},
                                     options.get("recognition_threshold", 0.70), options.get("recognition_consecutive", 2))
        for method in METHODS}
    report["time_countercheck"] = time_countercheck(fitted_at_primary, validation)
    report["simultaneous_grid_contrast_intervals"] = simultaneous_grid_intervals(
        validation, grid_predictions, config, replicates=diagnostic_bootstrap)
    report["label_countercheck"] = label_countercheck(fitted_at_primary, records, config,
                                                      replicates=diagnostic_permutations)
    report["bootstrap_diagnostic_only_override"] = diagnostic_bootstrap is not None
    report["topic_population_inference"] = False
    (output / "fixture_analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="SYNTHETIC FIXTURE ONLY: executable LEM-II Light analysis")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config" / "study.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostic-permutations", type=int, help="Logged fixture-only reduction of the predeclared repetition count")
    parser.add_argument("--diagnostic-bootstrap", type=int, help="Logged fixture-only reduction of bootstrap repetitions")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    report = run_fixture_analysis(config, args.output, diagnostic_permutations=args.diagnostic_permutations,
                                  diagnostic_bootstrap=args.diagnostic_bootstrap)
    print(json.dumps({"output": str(args.output.resolve()), "data_kind": report["data_kind"],
                      "scientific_results": False, "confirmation_generated_or_evaluated": False}))


if __name__ == "__main__":
    main()
