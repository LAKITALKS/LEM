"""Focused production-path checks; all data are synthetic software fixtures."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from lem_ii.evaluation import (
    clustered_indices, earliest_recognition, fit_and_tune, freeze_artifact,
    join_record_inputs, label_countercheck, load_frozen_artifact,
    paired_cluster_intervals, predict, predict_confirmation_once, predict_confirmation_grid_once,
    profile_label_mapping, split_records,
)
from lem_ii.features import (
    FeaturePipeline, LeakageError, METHODS, NotEvaluable, activation_blocks,
    compute_technical_features, delay_embedding, persistence_features, validate_records,
)
from lem_ii.fixtures import make_fixture_records


class FeatureEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((Path(__file__).parents[1] / "config" / "study.json").read_text())
        cls.records = make_fixture_records(profiles_per_regime=2, hidden_dimension=12)
        cls.development, cls.validation = split_records(cls.records)
        cls.fitted = fit_and_tune(cls.records, cls.config, 24)

    def test_split_and_metadata_leakage_guards(self):
        validate_records(self.records)
        for mutation in ("dialogue", "topic", "formulation", "metadata", "context"):
            records = copy.deepcopy(self.records)
            dev = next(r for r in records if r["split"] == "development")
            val = next(r for r in records if r["split"] == "validation")
            if mutation == "dialogue":
                val["dialogue_id"] = dev["dialogue_id"]
            elif mutation == "topic":
                val["topic_family"] = dev["topic_family"]
            elif mutation == "formulation":
                val["formulation_family"] = dev["formulation_family"]
            elif mutation == "metadata":
                val["generator_metadata"] = {"answer": val["regime"]}
            else:
                val["contexts"][0] += val["profile_id"]
            with self.subTest(mutation=mutation), self.assertRaises(LeakageError):
                validate_records(records)
        with self.assertRaises(LeakageError):
            FeaturePipeline(self.config).fit(self.validation)

    def test_no_future_context_or_states_and_no_validation_fit(self):
        pipeline = FeaturePipeline(self.config).fit(self.development, endpoint=12)
        baseline = pipeline.transform(self.validation)
        changed = copy.deepcopy(self.validation)
        for row in changed:
            row["contexts"][12:] = ["future-only token FUTURE_SECRET"] * 12
            row["states"][12:] = 1e20
        actual = pipeline.transform(changed)
        for method in METHODS:
            np.testing.assert_array_equal(actual[method].toarray(), baseline[method].toarray())
        before = (pipeline.pca.mean_.copy(), pipeline.state_scaler.mean_.copy(), dict(pipeline.word.vocabulary_))
        altered = copy.deepcopy(self.validation)
        for row in altered:
            row["contexts"][11] += " validationexclusiveword"
            row["states"] += 1000
        pipeline.transform(altered)
        np.testing.assert_array_equal(before[0], pipeline.pca.mean_)
        np.testing.assert_array_equal(before[1], pipeline.state_scaler.mean_)
        self.assertEqual(before[2], pipeline.word.vocabulary_)
        self.assertNotIn("validationexclusiveword", pipeline.word.vocabulary_)

    def test_ablation_exact_common_prefix_and_tuning_budget(self):
        matrices = self.fitted["pipeline"].transform(self.validation)
        for base, augmented in zip(METHODS, METHODS[1:]):
            width = matrices[base].shape[1]
            self.assertEqual((matrices[base] != matrices[augmented][:, :width]).nnz, 0)
        self.assertEqual(matrices[METHODS[-1]].shape[1] - matrices[METHODS[-2]].shape[1], 8)
        for tuning in self.fitted["tuning"].values():
            self.assertEqual([r["C"] for r in tuning["trials"]], [0.1, 1.0, 10.0])
            maximum = max(r["validation_balanced_accuracy"] for r in tuning["trials"])
            expected = min(r["C"] for r in tuning["trials"] if r["validation_balanced_accuracy"] == maximum)
            self.assertEqual(tuning["selected_C"], expected)

    def test_delay_order_control_and_point_cloud_invariance(self):
        rng = np.random.default_rng(184)
        states = np.cumsum(rng.normal(size=(24, 8)), axis=0)
        window = states[-12:]
        embedded = delay_embedding(window, lag=1, dimension=3)
        self.assertEqual(embedded.shape, (10, 24))
        np.testing.assert_array_equal(embedded[0], window[:3].reshape(-1))
        permutation = rng.permutation(12)
        np.testing.assert_allclose(persistence_features(window), persistence_features(window[permutation]), rtol=1e-6)
        normal = activation_blocks(states, self.config, endpoint=24)
        shuffled = activation_blocks(states, self.config, endpoint=24, shuffle_seed=145)
        np.testing.assert_allclose(normal["static"], shuffled["static"], atol=1e-12)
        self.assertFalse(np.allclose(normal["geometry_time"], shuffled["geometry_time"]))
        self.assertFalse(np.allclose(normal["topology"], shuffled["topology"]))
        shuffled_window = window[np.append(np.random.default_rng(145).permutation(11), 11)]
        np.testing.assert_allclose(shuffled["topology"], persistence_features(delay_embedding(shuffled_window)))

    def test_smoke_feature_path_and_short_failure(self):
        states = self.records[0]["states"]
        report = compute_technical_features(states, self.config)
        self.assertFalse(report["eligible_for_scientific_analysis"])
        self.assertTrue(report["finite_features"])
        self.assertTrue(report["point_cloud_permutation_invariant"])
        self.assertEqual(report["delay_point_count"], 10)
        self.assertEqual(len(report["features"]["topology"]), 8)
        with self.assertRaises(NotEvaluable):
            compute_technical_features(states[:2], self.config, endpoint=2)
        with self.assertRaises(NotEvaluable):
            activation_blocks(states[:11], self.config, endpoint=12)
        with self.assertRaises(NotEvaluable):
            compute_technical_features(np.zeros_like(states), self.config)

    def test_profile_cluster_bootstrap_and_group_label_randomization(self):
        indices = clustered_indices(self.validation, np.random.default_rng(23))
        for profile in {row["profile_id"] for row in self.validation}:
            members = [i for i, row in enumerate(self.validation) if row["profile_id"] == profile]
            self.assertEqual(int(np.sum(indices == members[0])), int(np.sum(indices == members[1])))
        mapping = profile_label_mapping(self.records, np.random.default_rng(8))
        self.assertEqual(sorted(mapping.values()), sorted({row["profile_id"]: row["regime"] for row in self.records}.values()))
        predictions = predict(self.fitted, self.validation)
        intervals = paired_cluster_intervals(self.validation, predictions, self.config, replicates=12)
        for row in intervals.values():
            self.assertAlmostEqual(row["confidence_level"], 1 - .05 / 3)
            self.assertEqual(row["bootstrap_replicates"], 12)
        check = label_countercheck(self.fitted, self.records, self.config, replicates=1)
        self.assertTrue(check["diagnostic_only_override"])
        self.assertEqual(check["replicates"], 1)

    def test_frozen_prediction_hash_and_confirmation_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            manifest = freeze_artifact(self.fitted, output)
            path = output / manifest["artifact_file"]
            manifest_path = output / "fixture_endpoint_24.manifest.json"
            loaded = load_frozen_artifact(path, manifest_path, self.config)
            expected = predict(self.fitted, self.validation)
            actual = predict(loaded, self.validation)
            for method in METHODS:
                np.testing.assert_array_equal(actual[method], expected[method])
            altered = copy.deepcopy(self.config)
            altered["features"]["window"] = 11
            with self.assertRaises(LeakageError):
                load_frozen_artifact(path, manifest_path, altered)
            with self.assertRaises(LeakageError):
                predict_confirmation_once(path, manifest_path, [], self.config, output / "confirmation.json")
            with self.assertRaises(LeakageError):
                predict_confirmation_grid_once(output, [], self.config, output / "confirmation_grid.json")
            path.write_bytes(path.read_bytes() + b"tamper")
            with self.assertRaises(LeakageError):
                load_frozen_artifact(path, manifest_path, self.config)

    def test_real_study_requires_explicit_future_gate_and_never_mixes(self):
        records = copy.deepcopy(self.records)
        for row in records:
            row["data_kind"] = "real_model_study"
            row["eligible_for_scientific_analysis"] = True
        with self.assertRaises(LeakageError):
            split_records(records, self.config)
        # A gate cannot promote fictitious session IDs to prespecified study data.
        # No scientific fit or evaluation follows this guard test.
        config = copy.deepcopy(self.config)
        config["execution"]["allow_scientific_analysis"] = True
        with self.assertRaises(LeakageError):
            validate_records(records, config=config)
        with self.assertRaises(LeakageError):
            validate_records([records[0], self.records[1]], config=config)

    def test_explicit_input_join_and_early_endpoint_rule(self):
        rows = self.records[:2]
        dialogues = [{"dialogue_id": r["dialogue_id"], "contexts": r["contexts"]} for r in rows]
        metadata = [{key: value for key, value in r.items() if key not in {"contexts", "states"}} for r in rows]
        states = {r["dialogue_id"]: r["states"] for r in rows}
        self.assertEqual(len(join_record_inputs(dialogues, metadata, states)), 2)
        with self.assertRaises(LeakageError):
            join_record_inputs(dialogues, metadata[:1], states)
        self.assertEqual(earliest_recognition({12: .69, 16: .71, 20: .72, 24: .73})["earliest_endpoint"], 16)
        self.assertIsNone(earliest_recognition({12: .69, 16: .71, 20: .69, 24: .75})["earliest_endpoint"])


if __name__ == "__main__":
    unittest.main()
