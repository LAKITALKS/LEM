"""Audit exported predictions independently of the fitting/evaluation runner."""
import csv
import json

import numpy as np
import pytest
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from .run_amendment_1 import OUTPUT


@pytest.mark.skipif(not (OUTPUT / 'summary.json').exists(), reason='Supplemental run not exported yet')
def test_reported_predictions_scalers_and_paired_intervals():
    summary = json.loads((OUTPUT / 'summary.json').read_text())
    features = list(csv.DictReader((OUTPUT / 'per_seed.csv').open()))
    predictions = list(csv.DictReader((OUTPUT / 'evaluation_predictions.csv').open()))
    feature_lookup = {(r['projection'], float(r['noise']), r['class'], int(r['seed'])): r for r in features}
    assert len(features) == 4800
    assert summary['original_outputs_unchanged']
    assert summary['original_outputs_sha256_before'] == summary['original_outputs_sha256_after']
    assert summary['shuffle_checks']['all_passed'] and summary['shuffle_checks']['count'] == 800
    for mode, levels in summary['metrics'].items():
        for noise, controls in levels.items():
            n = 100 if float(noise) == .05 else 20
            sampled = np.random.RandomState(2300).randint(n, size=(1000, n))
            for control, metrics in controls.items():
                selected = [r for r in predictions if r['projection'] == mode and
                            float(r['noise']) == float(noise) and r['control'] == control]
                assert len(selected) == 2 * n
                labels = np.array([int(r['positive_label']) for r in selected])
                assert np.array_equal(labels, np.r_[np.zeros(n), np.ones(n)])
                for label in ('CYCLE', control):
                    assert [int(r['seed']) for r in selected if r['class'] == label] == list(range(n, 2*n))
                bootstraps = {}
                for arm, column in (('H1', 'H1_signed'), ('G', 'G_probability'),
                                     ('G+H1', 'G_plus_H1_probability')):
                    scores = np.array([float(r[column]) for r in selected])
                    assert roc_auc_score(labels, scores) == metrics[arm]['auc']
                    bootstrap = np.c_[scores[:n][sampled], scores[n:][sampled]]
                    rank = rankdata(bootstrap, axis=1)
                    auc = (rank[:, n:].sum(1) - n*(n+1)/2) / n**2
                    bootstraps[arm] = auc
                    np.testing.assert_allclose(np.percentile(auc, [2.5,97.5]),
                                               metrics[arm]['auc_ci95'], atol=1e-15, rtol=0)
                    if arm == 'H1':
                        continue
                    fitted = metrics[arm]['fitted_model']
                    x = np.array([[float(feature_lookup[(mode, float(noise), r['class'], int(r['seed']))][name]) for name in fitted['features']] for r in selected])
                    predicted = expit(((x-np.array(fitted['scaler_mean']))/np.array(fitted['scaler_scale'])) @
                                      np.array(fitted['coefficients']) + fitted['intercept'])
                    np.testing.assert_allclose(predicted, scores, atol=1e-14, rtol=0)
                    calibration = np.array([[float(r[f]) for f in fitted['features']]
                        for label in ('CYCLE', control) for r in features if r['projection'] == mode and
                        float(r['noise']) == float(noise) and r['class'] == label and r['split'] == 'calibration'])
                    assert calibration.shape[0] == 2*n
                    np.testing.assert_allclose(calibration.mean(0), fitted['scaler_mean'], atol=1e-12, rtol=1e-12)
                    np.testing.assert_allclose(calibration.std(0), fitted['scaler_scale'], atol=1e-12, rtol=1e-12)
                delta = bootstraps['G+H1'] - bootstraps['G']
                np.testing.assert_allclose(np.percentile(delta, [2.5,97.5]), metrics['delta_auc']['ci95'], atol=1e-15, rtol=0)
                assert metrics['delta_auc']['estimate'] == metrics['G+H1']['auc']-metrics['G']['auc']
