"""CPU-only execution of the committed PROTOCOL.md. Author: Lazaros Varvatis."""
import os
# Set before importing numerical libraries; a worker owns its whole RNG stream.
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
import csv
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
from scipy.integrate import odeint
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, roc_curve
from threadpoolctl import threadpool_limits

if __package__:
    from . import original_helpers as original
else:
    import original_helpers as original

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / 'experiments/results/toy_v2_robustness'
CLASSES = ('SINK', 'CYCLE', 'DISK', 'LINE')
STATS = ('H1', 'g1', 'g2', 'g3', 'g1_second_half')
COMPARISONS = {'Va': ('SINK', 'CYCLE'), 'Vb': ('CYCLE', 'DISK'),
               'Vc': ('CYCLE', 'LINE')}
MODES = ('common', 'independent')
NOISES = (.01, .025, .05, .1, .2, .5)


def pooled_phi(points):
    centered = points - points.mean(axis=0)
    return float(np.sum(centered[1:] * centered[:-1]) /
                 np.sum(centered[:-1] ** 2))


def covariance_power(covariance, power):
    values, vectors = np.linalg.eigh(covariance)
    if np.any(values <= 0):
        raise ValueError('STOP: covariance is not positive definite')
    return (vectors * values ** power) @ vectors.T


def construct_controls(cycle, seed):
    rng = np.random.RandomState(seed)
    n = len(cycle)
    mean = cycle.mean(axis=0)
    covariance = np.cov(cycle, rowvar=False, ddof=1)
    phi = pooled_phi(cycle)
    if not abs(phi) < 1:
        raise ValueError('STOP: stationary AR(1) requires abs(phi)<1')
    root = covariance_power(covariance, .5)
    disk = np.empty_like(cycle)
    disk[0] = mean + rng.randn(2) @ root
    innovations = rng.randn(n - 1, 2) @ root * np.sqrt(1 - phi ** 2)
    for t in range(1, n):
        disk[t] = mean + phi * (disk[t - 1] - mean) + innovations[t - 1]
    phi_before = pooled_phi(disk)
    disk = ((disk - disk.mean(axis=0)) @
            covariance_power(np.cov(disk, rowvar=False, ddof=1), -.5) @ root + mean)
    centered = cycle - mean
    power = np.sum(np.abs(np.fft.rfft(centered, axis=0)) ** 2, axis=1)
    index = int(np.argmax(power[1:]) + 1)
    frequency = np.fft.rfftfreq(n, d=.1)[index]
    sine = np.sin(2 * np.pi * frequency * np.arange(n) * .1)
    sine -= sine.mean()
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    amplitude = np.sqrt(np.trace(covariance) / np.var(sine, ddof=1))
    line = mean + amplitude * sine[:, None] * eigenvectors[:, -1]
    diagnostics = {
        'phi': phi, 'disk_phi_before': phi_before, 'disk_phi_after': pooled_phi(disk),
        'disk_mean_residual': float(np.max(np.abs(disk.mean(axis=0) - mean))),
        'disk_cov_residual': float(np.max(np.abs(np.cov(disk, rowvar=False) - covariance))),
        'line_mean_residual': float(np.max(np.abs(line.mean(axis=0) - mean))),
        'line_variance_residual': float(abs(np.trace(np.cov(line, rowvar=False)) - np.trace(covariance))),
        'frequency': float(frequency),
    }
    return disk, line, diagnostics


def base_trajectories():
    t = np.linspace(0., 40., 401)
    return (odeint(original.system_fixed_point, [2., 1.5], t, args=(.5,)),
            odeint(original.system_van_der_pol, [.1, .1], t, args=(1.5,)))


def radius(points):
    return float(np.sqrt(np.mean(np.sum((points - points.mean(axis=0)) ** 2, axis=1))))


def statistics(points):
    h1 = original.persistence_stats(original.compute_persistence(points)[1])[0]
    centered = points - points.mean(axis=0)
    _, singular, vectors = np.linalg.svd(centered, full_matrices=False)
    radii = np.linalg.norm(centered @ vectors[:2].T, axis=1)
    return {'H1': h1, 'g1': radius(points),
            'g2': float((singular[0] / singular[1]) ** 2),
            'g3': float(radii.std(ddof=0) / radii.mean()),
            'g1_second_half': radius(points[len(points) // 2:])}


def run_unit(task):
    mode, sigma, seed = task
    with threadpool_limits(limits=1):
        sink, cycle = base_trajectories()
        disk, line, diagnostics = construct_controls(cycle, seed)
        np.random.seed(seed)
        original.NOISE_LEVEL = sigma
        matrix = np.random.randn(2, 16) if mode == 'common' else None
        rows = []
        for label, points in zip(CLASSES, (sink, cycle, disk, line)):
            if mode == 'independent':
                embedded = original.embed_high_dim(points)
            else:
                embedded = points @ matrix + np.random.normal(scale=sigma, size=(len(points), 16))
            row = {'projection': mode, 'noise': sigma, 'seed': seed,
                   'split': 'calibration' if seed < (100 if sigma == .05 else 20) else 'evaluation',
                   'class': label, **statistics(embedded), **diagnostics}
            rows.append(row)
        return rows


def calibrate(negative, positive):
    values = np.r_[negative, positive]
    labels = np.r_[np.zeros(len(negative)), np.ones(len(positive))]
    best = None
    for sign in (1, -1):
        fpr, tpr, thresholds = roc_curve(labels, sign * values, drop_intermediate=False)
        j = tpr - fpr
        index = int(np.flatnonzero(j == j.max())[0])  # descending thresholds
        candidate = {'sign': sign, 'threshold': float(thresholds[index]), 'youden_j': float(j[index])}
        if best is None or candidate['youden_j'] > best['youden_j']:
            best = candidate
    return best


def evaluate(negative, positive, calibration):
    sign, threshold = calibration['sign'], calibration['threshold']
    negative, positive = sign * negative, sign * positive
    n = len(negative)
    labels = np.r_[np.zeros(n), np.ones(n)]
    scores = np.r_[negative, positive]
    auc = float(roc_auc_score(labels, scores))
    accuracy = float(np.mean(np.r_[negative < threshold, positive >= threshold]))
    indices = np.random.RandomState(2300).randint(n, size=(1000, n))
    boot = np.concatenate([negative[indices], positive[indices]], axis=1)
    ranks = rankdata(boot, axis=1, method='average')
    aucs = (ranks[:, n:].sum(axis=1) - n * (n + 1) / 2) / n ** 2
    accuracies = np.mean(np.concatenate([negative[indices] < threshold,
                                       positive[indices] >= threshold], axis=1), axis=1)
    return {'auc': auc, 'auc_ci95': np.percentile(aucs, [2.5, 97.5]).tolist(),
            'accuracy': accuracy, 'accuracy_ci95': np.percentile(accuracies, [2.5, 97.5]).tolist(),
            'calibration': calibration, 'evaluation_n_per_class': n,
            'calibration_seeds': list(range(n)),
            'evaluation_seeds': list(range(n, 2 * n))}


def classify(primary):
    r1 = ('H1 trennt Senke und Zyklus' if primary['Va']['H1']['auc'] >= .95
          else 'R1 nicht erfüllt')
    vb = primary['Vb']
    best_g = max(vb[s]['auc'] for s in ('g1', 'g2', 'g3'))
    if vb['H1']['auc'] < .95:
        r2 = 'NICHT ROBUST'
    elif best_g <= .65:
        r2 = 'EINDEUTIGER VORTEIL'
    elif best_g < .95:
        r2 = 'TEILWEISE'
    else:
        r2 = 'KEIN EINDEUTIGER VORTEIL'
    return {'R1': r1, 'R2': r2, 'best_g_vb_auc': best_g}


def summarize(rows):
    summary = {}
    for mode in MODES:
        summary[mode] = {}
        for sigma in NOISES:
            subset = [r for r in rows if r['projection'] == mode and r['noise'] == sigma]
            comparisons = {}
            for comparison, (a, b) in COMPARISONS.items():
                comparisons[comparison] = {}
                for stat in STATS:
                    def extract(label, split):
                        return np.array([r[stat] for r in subset if r['class'] == label and r['split'] == split])
                    calibration = calibrate(extract(a, 'calibration'), extract(b, 'calibration'))
                    comparisons[comparison][stat] = evaluate(extract(a, 'evaluation'), extract(b, 'evaluation'), calibration)
            summary[mode][str(sigma)] = comparisons
    return summary


def draw_plot(summary, path):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    colors = {'H1': '#234d85', 'g1': '#c67816', 'g2': '#4d8d61', 'g3': '#a94363'}
    for i, mode in enumerate(MODES):
        for j, comparison in enumerate(COMPARISONS):
            ax = axes[i, j]
            for stat, color in colors.items():
                metrics = [summary[mode][str(sigma)][comparison][stat] for sigma in NOISES]
                values = np.array([m['auc'] for m in metrics])
                intervals = np.array([m['auc_ci95'] for m in metrics])
                ax.plot(NOISES, values, 'o-', label=stat, color=color, markersize=4)
                ax.fill_between(NOISES, intervals[:, 0], intervals[:, 1], color=color, alpha=.09)
            ax.axhline(.95, color='grey', linestyle=':', linewidth=.8)
            ax.axhline(.5, color='grey', linestyle='--', linewidth=.8)
            ax.axvline(.05, color='grey', linestyle=':', linewidth=.8)
            ax.set_xscale('log'); ax.set_ylim(-.03, 1.03)
            ax.set_title(f'{comparison}: {" vs ".join(COMPARISONS[comparison])} / {mode}')
            ax.grid(alpha=.15)
            if i == 1:
                ax.set_xlabel('Observation noise sigma')
            if j == 0:
                ax.set_ylabel('Evaluation AUC (frozen sign)')
    axes[0, 0].legend(ncol=2, fontsize=8, loc='lower left')
    fig.suptitle('Toy V2 robustness: held-out seeds, paired 95% bootstrap intervals')
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return 'Infinity' if value > 0 else '-Infinity'
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--benchmark-only', action='store_true')
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    benchmark_start = time.perf_counter()
    for mode in MODES:
        for seed in range(5):
            run_unit((mode, .05, seed))
    benchmark_seconds = time.perf_counter() - benchmark_start
    projected_seconds = benchmark_seconds / 10 * 800
    workers = max(1, args.workers)
    if projected_seconds > 3600:
        workers = max(2, workers, min(4, os.cpu_count() or 2))
    benchmark = {'seeds': list(range(5)), 'units': 10, 'seconds': benchmark_seconds,
                 'projected_serial_seconds': projected_seconds, 'workers': workers}
    print(json.dumps({'benchmark': benchmark}), flush=True)
    (output / 'benchmark.json').write_text(json.dumps(benchmark, indent=2) + '\n')
    if args.benchmark_only:
        return
    tasks = [(mode, sigma, seed) for mode in MODES for sigma in NOISES
             for seed in range(200 if sigma == .05 else 40)]
    computation_start = time.perf_counter()
    rows = []
    if workers == 1:
        units = map(run_unit, tasks)
        for i, unit in enumerate(units, 1):
            rows.extend(unit)
            if i % 40 == 0:
                print(f'{i}/{len(tasks)} units; {time.perf_counter() - computation_start:.1f}s', flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for i, unit in enumerate(executor.map(run_unit, tasks), 1):
                rows.extend(unit)
                if i % 40 == 0:
                    print(f'{i}/{len(tasks)} units; {time.perf_counter() - computation_start:.1f}s', flush=True)
    computation_seconds = time.perf_counter() - computation_start
    rows.sort(key=lambda r: (MODES.index(r['projection']), r['noise'], r['seed'], CLASSES.index(r['class'])))
    with (output / 'per_seed.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    metrics = summarize(rows)
    diagnostics = {key: {'min': min(r[key] for r in rows), 'max': max(r[key] for r in rows)}
                   for key in ('phi', 'disk_phi_before', 'disk_phi_after', 'disk_mean_residual',
                               'disk_cov_residual', 'line_mean_residual', 'line_variance_residual', 'frequency')}
    protocol_commit = subprocess.check_output(['git', 'log', '-1', '--format=%H', '--',
        'experiments/toy_v2_robustness/PROTOCOL.md'], cwd=ROOT, text=True).strip()
    draw_plot(metrics, output / 'toy_v2_robustness.png')
    summary = {'protocol_commit': protocol_commit, 'metrics': metrics,
               'categories': classify(metrics['common']['0.05']), 'diagnostics': diagnostics,
               'benchmark': benchmark, 'computation_seconds': computation_seconds,
               'total_seconds': time.perf_counter() - started, 'row_count': len(rows),
               'bootstrap': {'seed': 2300, 'draws': 1000, 'unit': 'paired seed', 'interval': 'percentile 95%'},
               'scope': 'Synthetic pipeline checks only; no inference about LLMs or users.'}
    (output / 'summary.json').write_text(json.dumps(json_safe(summary), indent=2, allow_nan=False) + '\n')
    packages = ('numpy', 'scipy', 'scikit-learn', 'ripser', 'matplotlib', 'threadpoolctl', 'pytest', 'pypdf')
    (output / 'environment.txt').write_text(
        f'Python: {sys.version}\nPlatform: {platform.platform()}\nCPU only; one numerical thread per worker\n' +
        '\n'.join(f'{p}=={importlib.metadata.version(p)}' for p in packages) + '\n')
    print(json.dumps({'categories': summary['categories'], 'total_seconds': summary['total_seconds'],
                      'diagnostics': diagnostics}), flush=True)


if __name__ == '__main__':
    main()
