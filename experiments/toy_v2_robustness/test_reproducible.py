"""Independent provenance, deterministic output, and matched-control checks."""
import ast
import json
import textwrap
from pathlib import Path

import numpy as np
import pytest
from . import original_helpers
from .run_toy_v2_robustness import (base_trajectories, calibrate, classify,
    construct_controls, evaluate, run_unit)


@pytest.mark.parametrize('mode', ['common', 'independent'])
def test_three_seeds_bitwise(mode):
    for seed in (0, 1, 2):
        a = run_unit((mode, .05, seed))
        b = run_unit((mode, .05, seed))
        assert json.dumps(a, sort_keys=True).encode() == json.dumps(b, sort_keys=True).encode()
        for left, right in zip(a, b):
            floats = [key for key in left if isinstance(left[key], float)]
            assert np.array([left[k] for k in floats]).tobytes() == np.array([right[k] for k in floats]).tobytes()


def test_original_seed_1337():
    rows = run_unit(('independent', .05, 1337))
    assert round(rows[0]['H1'], 4) == .0326
    assert round(rows[1]['H1'], 4) == 5.2649


def test_verbatim_original_helpers():
    source = Path(__file__).resolve().parents[1] / 'lem_simulations.py'
    original = next(n for n in ast.parse(source.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'run_experiment_2')
    copied_source = Path(original_helpers.__file__).read_text()
    copied = ast.parse(copied_source)
    names = ('system_fixed_point', 'system_van_der_pol', 'embed_high_dim',
             'compute_persistence', 'persistence_stats')
    for name in names:
        a = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == name)
        b = next(n for n in copied.body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert ast.dump(a) == ast.dump(b)
        assert textwrap.dedent('\n'.join(source.read_text().splitlines()[a.lineno - 1:a.end_lineno])) == ast.get_source_segment(copied_source, b)


def test_exact_control_moments():
    _, cycle = base_trajectories()
    for seed in (0, 1, 199):
        disk, line, diagnostics = construct_controls(cycle, seed)
        np.testing.assert_allclose(disk.mean(0), cycle.mean(0), atol=1e-13, rtol=0)
        np.testing.assert_allclose(np.cov(disk.T), np.cov(cycle.T), atol=1e-12, rtol=0)
        np.testing.assert_allclose(line.mean(0), cycle.mean(0), atol=1e-13, rtol=0)
        assert diagnostics['line_variance_residual'] < 1e-12
        assert np.linalg.matrix_rank(line - line.mean(0), tol=1e-10) == 1


def test_frozen_direction_and_threshold():
    calibration = calibrate(np.array([3., 4., 5.]), np.array([0., 1., 2.]))
    assert calibration['sign'] == -1
    metrics = evaluate(np.array([0., 1., 2.]), np.array([3., 4., 5.]), calibration)
    assert metrics['auc'] == 0  # evaluation must not reorient the score
    assert metrics['accuracy'] == 0


def test_r2_ignores_vc():
    primary = {'Va': {'H1': {'auc': 1}}, 'Vb': {'H1': {'auc': 1},
               'g1': {'auc': .5}, 'g2': {'auc': .5}, 'g3': {'auc': .65}},
               'Vc': {'H1': {'auc': 0}, 'g2': {'auc': 1}}}
    assert classify(primary)['R2'] == 'EINDEUTIGER VORTEIL'
    primary['Vb']['g3']['auc'] = .95
    assert classify(primary)['R2'] == 'KEIN EINDEUTIGER VORTEIL'
    primary['Vb']['H1']['auc'] = .949
    assert classify(primary)['R2'] == 'NICHT ROBUST'
