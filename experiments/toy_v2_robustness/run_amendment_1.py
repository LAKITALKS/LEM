"""Amendment 1 only. Original results are immutable. Author: Lazaros Varvatis."""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

if __package__:
    from . import original_helpers as original
    from . import run_toy_v2_robustness as baseline
else:
    import original_helpers as original
    import run_toy_v2_robustness as baseline

ROOT = baseline.ROOT
ORIGINAL_DIR = baseline.DEFAULT_OUTPUT
OUTPUT = ORIGINAL_DIR / 'amendment_1'
CONTROLS = ('GAUSS', 'UNIFORM', 'LINE')
GEOMETRY = ('g1', 'g2', 'g3')


def original_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(ORIGINAL_DIR.iterdir()) if p.is_file()}


def uniform_control(cycle, seed):
    rng = np.random.RandomState(seed + 1000000)
    n = len(cycle)
    angles = rng.uniform(0, 2 * np.pi, n)
    radii = np.sqrt(rng.uniform(0, 1, n))
    mean = cycle.mean(axis=0)
    covariance = np.cov(cycle, rowvar=False, ddof=1)
    root = baseline.covariance_power(covariance, .5)
    points = np.column_stack((radii * np.cos(angles), radii * np.sin(angles))) @ root + mean
    points = ((points - points.mean(axis=0)) @
              baseline.covariance_power(np.cov(points, rowvar=False, ddof=1), -.5) @ root + mean)
    return points, {
        'uniform_mean_residual': float(np.max(np.abs(points.mean(axis=0) - mean))),
        'uniform_cov_residual': float(np.max(np.abs(np.cov(points, rowvar=False) - covariance))),
    }


def geometry(points):
    centered = points - points.mean(axis=0)
    _, singular, vectors = np.linalg.svd(centered, full_matrices=False)
    radii = np.linalg.norm(centered @ vectors[:2].T, axis=1)
    return {'g1': baseline.radius(points), 'g2': float((singular[0] / singular[1]) ** 2),
            'g3': float(radii.std(ddof=0) / radii.mean()),
            'g1_second_half': baseline.radius(points[len(points) // 2:])}


def diagram_difference(left, right):
    if left.shape != right.shape:
        return float('inf')
    a = left[np.lexsort((left[:, 1], left[:, 0]))]
    b = right[np.lexsort((right[:, 1], right[:, 0]))]
    if not np.array_equal(np.isposinf(a), np.isposinf(b)):
        return float('inf')
    finite = np.isfinite(a) & np.isfinite(b)
    return float(np.max(np.abs(a[finite] - b[finite]), initial=0))


def supplemental_unit(task):
    mode, sigma, seed = task
    with threadpool_limits(limits=1):
        sink, cycle = baseline.base_trajectories()
        gauss, line, _ = baseline.construct_controls(cycle, seed)
        uniform, diagnostics = uniform_control(cycle, seed)
        np.random.seed(seed)
        original.NOISE_LEVEL = sigma
        matrix = np.random.randn(2, 16) if mode == 'common' else None
        cycle_hd = None
        uniform_hd = None
        for label, points in zip(('SINK', 'CYCLE', 'GAUSS', 'LINE', 'UNIFORM'),
                                 (sink, cycle, gauss, line, uniform)):
            if mode == 'independent':
                embedded = original.embed_high_dim(points)
            else:
                embedded = points @ matrix + np.random.normal(scale=sigma, size=(len(points), 16))
            if label == 'CYCLE':
                cycle_hd = embedded
            elif label == 'UNIFORM':
                uniform_hd = embedded
        diagrams = original.compute_persistence(cycle_hd)
        shuffled = cycle_hd[np.random.RandomState(seed + 2000000).permutation(len(cycle_hd))]
        shuffled_diagrams = original.compute_persistence(shuffled)
        cycle_stats = {'H1': original.persistence_stats(diagrams[1])[0], **geometry(cycle_hd)}
        shuffle_stats = {'H1': original.persistence_stats(shuffled_diagrams[1])[0], **geometry(shuffled)}
        checks = {'projection': mode, 'noise': sigma, 'seed': seed,
                  'H0_diagram_max_abs': diagram_difference(diagrams[0], shuffled_diagrams[0]),
                  'H1_diagram_max_abs': diagram_difference(diagrams[1], shuffled_diagrams[1]),
                  **{f'{s}_max_abs': float(abs(cycle_stats[s] - shuffle_stats[s])) for s in GEOMETRY},
                  **diagnostics}
        checks['passed'] = all(checks[k] <= 1e-9 for k in
            ('H0_diagram_max_abs', 'H1_diagram_max_abs', 'g1_max_abs', 'g2_max_abs', 'g3_max_abs'))
        return (baseline.statistics(uniform_hd), cycle_stats, shuffle_stats, checks)


def auc_bootstrap(negative, positive, indices):
    n = len(negative)
    boot = np.concatenate([negative[indices], positive[indices]], axis=1)
    ranks = rankdata(boot, axis=1, method='average')
    return (ranks[:, n:].sum(axis=1) - n * (n + 1) / 2) / n ** 2


def fit_arm(calibration_x, calibration_y, evaluation_x):
    scaler = StandardScaler().fit(calibration_x)
    model = LogisticRegression(C=1.0, l1_ratio=0.0, solver='lbfgs', max_iter=10000, tol=1e-10,
                               fit_intercept=True, class_weight=None, random_state=2301)
    # sklearn's default L2 penalty is specified by the fixed l1_ratio=0 default.
    model.fit(scaler.transform(calibration_x), calibration_y)
    if int(model.n_iter_.max()) >= model.max_iter:
        raise RuntimeError('STOP: logistic regression did not converge')
    probabilities = model.predict_proba(scaler.transform(evaluation_x))[:, 1]
    fitted = {'features': GEOMETRY if calibration_x.shape[1] == 3 else (*GEOMETRY, 'H1'),
              'C': 1.0, 'penalty': 'L2', 'solver': 'lbfgs', 'tol': 1e-10,
              'scaler_mean': scaler.mean_.tolist(), 'scaler_scale': scaler.scale_.tolist(),
              'coefficients': model.coef_[0].tolist(), 'intercept': float(model.intercept_[0]),
              'iterations': int(model.n_iter_[0]), 'threshold': .5}
    return probabilities, fitted


def category(metrics):
    if metrics['H1']['auc'] < .95:
        return 'NICHT ROBUST'
    if metrics['G']['auc'] <= .65:
        return 'EINDEUTIGER VORTEIL'
    if metrics['delta_auc']['estimate'] >= .05 and metrics['delta_auc']['ci95'][0] > 0:
        return 'ZUSATZNUTZEN'
    return 'GLEICHSTAND'


def analyze(rows):
    metrics = {}
    prediction_rows = []
    for mode in baseline.MODES:
        metrics[mode] = {}
        for sigma in baseline.NOISES:
            n = 100 if sigma == .05 else 20
            subset = [r for r in rows if r['projection'] == mode and r['noise'] == sigma]
            metrics[mode][str(sigma)] = {}
            indices = np.random.RandomState(2300).randint(n, size=(1000, n))
            for control in CONTROLS:
                classes = ('CYCLE', control)
                def extract(split, features):
                    return np.array([[r[f] for f in features] for label in classes for r in subset
                                     if r['class'] == label and r['split'] == split])
                calibration_y = np.r_[np.zeros(n), np.ones(n)]
                evaluation_y = calibration_y.copy()
                h1_cal = extract('calibration', ('H1',)).ravel()
                h1_eval = extract('evaluation', ('H1',)).ravel()
                h1_frozen = baseline.calibrate(h1_cal[:n], h1_cal[n:])
                result = {'H1': baseline.evaluate(h1_eval[:n], h1_eval[n:], h1_frozen)}
                probability_arms = {}
                bootstrap_arms = {}
                for arm, features in (('G', GEOMETRY), ('G+H1', (*GEOMETRY, 'H1'))):
                    probability, fitted = fit_arm(extract('calibration', features), calibration_y,
                                                   extract('evaluation', features))
                    probability_arms[arm] = probability
                    boot_auc = auc_bootstrap(probability[:n], probability[n:], indices)
                    bootstrap_arms[arm] = boot_auc
                    boot_accuracy = np.mean(np.c_[probability[:n][indices] < .5,
                                                   probability[n:][indices] >= .5], axis=1)
                    result[arm] = {
                        'auc': float(roc_auc_score(evaluation_y, probability)),
                        'auc_ci95': np.percentile(boot_auc, [2.5, 97.5]).tolist(),
                        'accuracy': float(np.mean((probability >= .5) == evaluation_y)),
                        'accuracy_ci95': np.percentile(boot_accuracy, [2.5, 97.5]).tolist(),
                        'fitted_model': fitted, 'calibration_seeds': list(range(n)),
                        'evaluation_seeds': list(range(n, 2 * n)), 'evaluation_n_per_class': n}
                delta = bootstrap_arms['G+H1'] - bootstrap_arms['G']
                result['delta_auc'] = {
                    'estimate': result['G+H1']['auc'] - result['G']['auc'],
                    'ci95': np.percentile(delta, [2.5, 97.5]).tolist()}
                for arm in GEOMETRY + ('g1_second_half',):
                    cal = extract('calibration', (arm,)).ravel()
                    ev = extract('evaluation', (arm,)).ravel()
                    result[arm] = baseline.evaluate(ev[:n], ev[n:], baseline.calibrate(cal[:n], cal[n:]))
                for index in range(2 * n):
                    prediction_rows.append({'projection': mode, 'noise': sigma, 'control': control,
                        'seed': n + index % n, 'class': classes[index // n],
                        'positive_label': index // n, 'H1_signed': float(h1_eval[index] * h1_frozen['sign']),
                        'G_probability': float(probability_arms['G'][index]),
                        'G_plus_H1_probability': float(probability_arms['G+H1'][index])})
                metrics[mode][str(sigma)][control] = result
    primary = metrics['common']['0.05']
    categories = {control: category(primary[control]) for control in ('GAUSS', 'UNIFORM')}
    order = ('EINDEUTIGER VORTEIL', 'ZUSATZNUTZEN', 'GLEICHSTAND', 'NICHT ROBUST')
    categories['overall'] = max(categories.values(), key=order.index)
    return metrics, categories, prediction_rows


def save_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def plot(metrics, path):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    for i, mode in enumerate(baseline.MODES):
        for j, control in enumerate(CONTROLS):
            ax = axes[i, j]
            for arm, color in (('H1', '#234d85'), ('G', '#b65470'), ('G+H1', '#37816a')):
                values = [metrics[mode][str(s)][control][arm] for s in baseline.NOISES]
                ax.plot(baseline.NOISES, [v['auc'] for v in values], 'o-', color=color, label=arm, markersize=4)
                ax.fill_between(baseline.NOISES, [v['auc_ci95'][0] for v in values],
                                [v['auc_ci95'][1] for v in values], color=color, alpha=.09)
            ax.axhline(.95, color='grey', linestyle=':', linewidth=.8)
            ax.axhline(.5, color='grey', linestyle='--', linewidth=.8)
            ax.axvline(.05, color='grey', linestyle=':', linewidth=.8)
            ax.set_xscale('log'); ax.set_ylim(-.03, 1.03)
            ax.set_title(f'CYCLE vs {control} / {mode}')
            ax.grid(alpha=.15)
            if i == 1:
                ax.set_xlabel('Observation noise sigma')
            if j == 0:
                ax.set_ylabel('Evaluation AUC')
    axes[0, 0].legend(fontsize=8, loc='lower left')
    fig.suptitle('Toy V2 Amendment 1: frozen predictors, paired 95% bootstrap intervals')
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark-only', action='store_true')
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT / 'summary.json').exists():
        raise FileExistsError('Supplemental summary already exists; do not overwrite a reported run')
    before = original_hashes()
    started = time.perf_counter()
    for mode in baseline.MODES:
        for seed in range(5):
            unit = supplemental_unit((mode, .05, seed))
            if not unit[3]['passed']:
                raise AssertionError(f'Shuffle invariance failed during benchmark: {unit[3]}')
    elapsed = time.perf_counter() - started
    workers = max(1, args.workers)
    if elapsed / 10 * 800 > 3600:
        workers = max(2, workers, min(4, os.cpu_count() or 2))
    benchmark = {'seeds': list(range(5)), 'seconds': elapsed,
                 'projected_serial_seconds': elapsed / 10 * 800, 'workers': workers}
    print(json.dumps({'benchmark': benchmark}), flush=True)
    if args.benchmark_only:
        assert before == original_hashes()
        return
    original_rows = list(csv.DictReader((ORIGINAL_DIR / 'per_seed.csv').open()))
    groups = {}
    for r in original_rows:
        row = {key: (int(value) if key == 'seed' else value if key in ('projection', 'split', 'class') else float(value))
               for key, value in r.items()}
        if row['class'] == 'DISK':
            row['class'] = 'GAUSS'
        key = (row['projection'], row['noise'], row['seed'])
        groups.setdefault(key, []).append(row)
    tasks = list(groups)
    rows, checks = [], []
    computation_start = time.perf_counter()
    def collect(task, result):
        mode, sigma, seed = task
        uniform_stats, cycle_stats, shuffle_stats, diagnostic = result
        group = groups[task]
        cycle_row = next(r for r in group if r['class'] == 'CYCLE')
        for stat in baseline.STATS:
            if cycle_row[stat] != cycle_stats[stat]:
                raise AssertionError(f'Regenerated CYCLE differs from original: {task} {stat}')
        if not diagnostic['passed']:
            raise AssertionError(f'Shuffle invariance failed: {diagnostic}')
        metadata = {key: cycle_row[key] for key in ('projection', 'noise', 'seed', 'split')}
        group.append({**metadata, 'class': 'UNIFORM', **uniform_stats,
                      'uniform_mean_residual': diagnostic['uniform_mean_residual'],
                      'uniform_cov_residual': diagnostic['uniform_cov_residual']})
        group.append({**metadata, 'class': 'CYCLE_SHUF', **shuffle_stats})
        rows.extend(group); checks.append(diagnostic)
    if workers == 1:
        for i, task in enumerate(tasks, 1):
            collect(task, supplemental_unit(task))
            if i % 80 == 0:
                print(f'{i}/800 supplemental units; {time.perf_counter()-computation_start:.1f}s', flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for i, (task, unit) in enumerate(zip(tasks, executor.map(supplemental_unit, tasks)), 1):
                collect(task, unit)
                if i % 80 == 0:
                    print(f'{i}/800 supplemental units', flush=True)
    computation_seconds = time.perf_counter() - computation_start
    metrics, categories, predictions = analyze(rows)
    plot(metrics, OUTPUT / 'toy_v2_robustness.png')
    after = original_hashes()
    if before != after:
        raise AssertionError('Original output hashes changed')
    save_csv(OUTPUT / 'per_seed.csv', rows)
    save_csv(OUTPUT / 'shuffle_checks.csv', checks)
    save_csv(OUTPUT / 'evaluation_predictions.csv', predictions)
    original_summary = json.loads((ORIGINAL_DIR / 'summary.json').read_text())
    amendment_commit = subprocess.check_output(['git', 'log', '-1', '--format=%H', '--',
        'experiments/toy_v2_robustness/PROTOCOL.md'], cwd=ROOT, text=True).strip()
    maxima = {key: max(row[key] for row in checks) for key in checks[0]
              if key not in ('projection', 'noise', 'seed', 'passed')}
    summary = {'amendment_commit': amendment_commit, 'original_protocol_commit': original_summary['protocol_commit'],
               'original_results_commit': 'ee90b9f', 'original_categories': original_summary['categories'],
               'original_outputs_sha256_before': before, 'original_outputs_sha256_after': after,
               'original_outputs_unchanged': before == after, 'metrics': metrics, 'categories': categories,
               'shuffle_checks': {'count': len(checks), 'all_passed': all(r['passed'] for r in checks),
                                  'tolerance': 1e-9, 'maxima': maxima},
               'benchmark': benchmark, 'computation_seconds': computation_seconds,
               'total_seconds': time.perf_counter() - started, 'row_count': len(rows),
               'bootstrap': {'draws': 1000, 'seed': 2300, 'unit': 'whole paired evaluation seed',
                             'interval': 'percentile 95%', 'refit': False},
               'scope': 'Synthetic pipeline checks only; point-cloud H1 measures loop geometry, not temporal return.'}
    (OUTPUT / 'summary.json').write_text(json.dumps(baseline.json_safe(summary), indent=2, allow_nan=False) + '\n')
    packages = sorted({f'{d.metadata["Name"]}=={d.version}' for d in importlib.metadata.distributions()})
    (OUTPUT / 'environment.txt').write_text(f'Python: {sys.version}\nPlatform: {platform.platform()}\n'
        'CPU only; one numerical thread per worker\n' + '\n'.join(packages) + '\n')
    print(json.dumps({'categories': categories, 'total_seconds': summary['total_seconds'],
                      'shuffle_checks': summary['shuffle_checks']}), flush=True)


if __name__ == '__main__':
    main()
