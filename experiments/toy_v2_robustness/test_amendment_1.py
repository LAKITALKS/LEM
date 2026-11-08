"""Amendment-specific invariants and preservation of the original version."""
import hashlib
import json
import subprocess

import numpy as np
import pytest

from . import run_toy_v2_robustness as baseline
from .run_amendment_1 import (ORIGINAL_DIR, category, fit_arm,
                              supplemental_unit, uniform_control)


@pytest.mark.parametrize('mode', ['common', 'independent'])
def test_supplement_three_seeds_bitwise_and_shuffle(mode):
    for seed in (0, 1, 2):
        a = supplemental_unit((mode, .05, seed))
        b = supplemental_unit((mode, .05, seed))
        assert json.dumps(a, sort_keys=True).encode() == json.dumps(b, sort_keys=True).encode()
        assert a[3]['passed']


def test_uniform_exact_moments():
    _, cycle = baseline.base_trajectories()
    for seed in (0, 1, 199):
        uniform, diagnostic = uniform_control(cycle, seed)
        assert len(uniform) == len(cycle)
        np.testing.assert_allclose(uniform.mean(0), cycle.mean(0), atol=1e-13, rtol=0)
        np.testing.assert_allclose(np.cov(uniform.T), np.cov(cycle.T), atol=1e-12, rtol=0)
        assert max(diagnostic.values()) < 1e-12


def test_calibration_fit_is_independent_of_evaluation_data():
    rng = np.random.RandomState(23)
    calibration = rng.randn(40, 4)
    labels = np.r_[np.zeros(20), np.ones(20)]
    first, fitted_first = fit_arm(calibration, labels, np.ones((6, 4)))
    repeated, fitted_repeated = fit_arm(calibration, labels, np.ones((6, 4)))
    _, fitted_other = fit_arm(calibration, labels, np.full((6, 4), 1e6))
    assert first.tobytes() == repeated.tobytes()
    assert fitted_first == fitted_repeated == fitted_other


def test_category_boundaries():
    def metrics(h1=1, g=.65, delta=.05, lower=.001):
        return {'H1': {'auc': h1}, 'G': {'auc': g},
                'delta_auc': {'estimate': delta, 'ci95': [lower, .1]}}
    assert category(metrics()) == 'EINDEUTIGER VORTEIL'
    assert category(metrics(g=.66)) == 'ZUSATZNUTZEN'
    assert category(metrics(g=.66, lower=0)) == 'GLEICHSTAND'
    assert category(metrics(g=.66, delta=.049)) == 'GLEICHSTAND'
    assert category(metrics(h1=.949)) == 'NICHT ROBUST'


def test_original_outputs_unchanged_from_original_results_commit():
    for name in ('per_seed.csv', 'summary.json', 'toy_v2_robustness.png',
                 'environment.txt', 'benchmark.json'):
        original = subprocess.check_output(['git', 'show',
            f'ee90b9f:experiments/results/toy_v2_robustness/{name}'], cwd=baseline.ROOT)
        assert hashlib.sha256(original).digest() == hashlib.sha256((ORIGINAL_DIR / name).read_bytes()).digest()
