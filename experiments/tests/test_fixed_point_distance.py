"""Analytical and production-path regressions; run with stdlib unittest.

Integration fixtures compile the actual experiment bodies with ONLY their local
configuration constants reduced for speed. Metric statements, simulation loops,
and return/export paths are untouched. Deterministic normal draws embed s=(1,0),
d=(0,1), zero initial states and zero noise in all three experiment functions.
LEM_SIMULATIONS_SOURCE selects an isolated mathematical mutant for the red check.
"""

import ast
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

EXPERIMENTS = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get("LEM_SIMULATIONS_SOURCE", EXPERIMENTS / "lem_simulations.py"))
sys.path.insert(0, str(EXPERIMENTS))


@contextlib.contextmanager
def output_directory():
    previous = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="lem-test-") as folder:
        os.chdir(folder)
        Path("assets").mkdir()
        try:
            yield
        finally:
            os.chdir(previous)


with output_directory():
    import run_all  # unchanged headless display setup
    spec = importlib.util.spec_from_file_location("simulation_under_test", SOURCE)
    sim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sim)


def controlled_experiment(name):
    tree = ast.parse(SOURCE.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    fixture = {"DIM": 2, "N_USERS": 3, "N_TESTS": 1, "T_STEPS": 100,
               "LAST_K": 20, "SIGMA": 0.0, "SEEDS": [42], "SIGMA_VALUES": [0.0]}
    for node in function.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in fixture:
                node.value = ast.parse(repr(fixture[target.id]), mode="eval").body
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = vars(sim).copy()
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace[name]


class FixedPointDistanceTests(unittest.TestCase):
    def test_noiseless_converged_scalar(self):
        s, d = np.array([1.0, 0.0]), np.array([0.0, 1.0])
        x = np.array([-2.0, 3.0])
        states = []
        for _ in range(100):
            x = 0.3 * x + 0.4 * s + 0.3 * d
            states.append(x.copy())
        mean = np.mean(states[-20:], axis=0)
        # Independent analytical expectation, not computed by the helper.
        np.testing.assert_allclose(mean, [4 / 7, 3 / 7], rtol=0, atol=1e-12)
        self.assertLess(sim.fixed_point_distance(mean, s, d, 0.4, 0.3), 1e-12)
        # Old metric has a nonzero deterministic bias, exactly 3/14 here.
        self.assertAlmostEqual(np.linalg.norm(mean - np.array([0.4, 0.3])), 3 / 14)

    def test_vectorized_distances_match_independent_values_and_scalar(self):
        users = np.array([[1.0, 0.0], [0.0, 1.0]])
        domain = np.array([0.0, 1.0])
        means = np.array([[4 / 7, 3 / 7], [3.0, 5.0]])
        actual = sim.fixed_point_distance(means, users, domain[None, :], 0.4, 0.3)
        np.testing.assert_allclose(actual, [0.0, 5.0], rtol=0, atol=1e-12)
        for i in range(2):
            self.assertAlmostEqual(actual[i], sim.fixed_point_distance(means[i], users[i], domain, 0.4, 0.3))

    def assert_production_distance_zero(self, name):
        calls = 0

        def normal(*args, size=None, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return np.tile([1.0, 0.0], (3, 1))
            if calls == 2:
                return np.tile([0.0, 1.0], (2, 1))
            return np.zeros(size)

        with output_directory(), patch.object(np.random, "normal", side_effect=normal), contextlib.redirect_stdout(io.StringIO()):
            result = controlled_experiment(name)()
        if name == "run_experiment_1":
            distance = result["mean_attractor_distance"]
        elif name == "run_experiment_1_scaled":
            distance = result[0]["mean_attractor_distance"].iloc[0]
        else:
            distance = result["attractor_distance"].iloc[0]
        # Production exports round at 4/6 decimals; the analytical error is <1e-12.
        self.assertEqual(distance, 0.0, f"{name} must measure the actual fixed point")

    def test_pilot_production_path(self):
        self.assert_production_distance_zero("run_experiment_1")

    def test_scaled_production_path(self):
        self.assert_production_distance_zero("run_experiment_1_scaled")

    def test_v1b_production_path(self):
        self.assert_production_distance_zero("run_experiment_1b")


if __name__ == "__main__":
    unittest.main()
