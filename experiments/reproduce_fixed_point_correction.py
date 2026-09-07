"""Reproduce only V1, with separate original/corrected sources and output folders.

Run from the repository root. No V2, model API, or cloud workload is invoked.
"""

import argparse
import ast
import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "f15073f2cab3fb68857ffb16035d6675957c2e06"
SOURCE_FILES = ["experiments/lem_simulations.py", "experiments/run_all.py"]
FUNCTIONS = ["run_experiment_1", "run_experiment_1_scaled", "run_experiment_1b"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def configurations(source):
    configs = {}
    for node in ast.parse(source.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS:
            configs[node.name] = {}
            for statement in node.body:
                if isinstance(statement, ast.Assign):
                    for target in statement.targets:
                        if isinstance(target, ast.Name) and target.id.isupper():
                            try:
                                value = ast.literal_eval(statement.value)
                            except (ValueError, TypeError):
                                # SEEDS_LIST is a derived plotting list, not a configuration.
                                continue
                            configs[node.name][target.id] = value
    return configs


def worker(source_root, output):
    parameters = configurations(source_root / SOURCE_FILES[0])
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(output)
    sys.path.insert(0, str(source_root / "experiments"))
    # Use the unchanged standalone import shim, without invoking its main/V2 path.
    import run_all
    import numpy as np

    rng_states = {}
    for name in FUNCTIONS:
        print(f"\n--- {name} ---", flush=True)
        getattr(run_all, name)()
        state = np.random.get_state()
        digest = hashlib.sha256(state[1].tobytes())
        digest.update(repr((state[0], *state[2:])).encode())
        rng_states[name] = digest.hexdigest()
        write_json(output / "rng_states.json", rng_states)

    config_info = io.StringIO()
    with contextlib.redirect_stdout(config_info):
        np.show_config()
    write_json(output / "execution.json", {
        "source_sha256": {p: sha256(source_root / p) for p in SOURCE_FILES},
        "parameters": parameters,
        "rng_state_sha256_after_each_experiment": rng_states,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": {p: importlib.metadata.version(p) for p in
                     ["numpy", "scipy", "matplotlib", "scikit-learn", "pandas"]},
        "numpy_build_configuration": config_info.getvalue(),
        "thread_environment": {k: os.environ.get(k) for k in
                               ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]},
        "worker_command": sys.argv,
        "outputs_sha256": {str(p.relative_to(output)): sha256(p)
                           for p in sorted(output.rglob("*"))
                           if p.is_file() and p.suffix in {".csv", ".json", ".png"}
                           and p.name != "execution.json"},
    })


def run_phase(phase, source_root, output):
    folder = output / phase
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "execution.json").exists():
        raise SystemExit(f"Completed results already exist: {folder}. Use a fresh --output-dir.")
    command = [sys.executable, str(Path(__file__).resolve()), "--worker",
               "--source-root", str(source_root), "--output-dir", str(folder)]
    with (folder / "run.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)


def compare(output, baseline):
    import pandas as pd
    original = json.loads((output / "original/execution.json").read_text())
    corrected = json.loads((output / "corrected/execution.json").read_text())
    for field in ["parameters", "packages", "python", "numpy_build_configuration",
                  "thread_environment", "rng_state_sha256_after_each_experiment"]:
        if original[field] != corrected[field]:
            raise AssertionError(f"Unexpected before/after difference: {field}")
    for filename in SOURCE_FILES:
        expected = hashlib.sha256(git("show", f"{baseline}:{filename}")).hexdigest()
        assert original["source_sha256"][filename] == expected, filename
        assert corrected["source_sha256"][filename] == sha256(ROOT / filename), filename
    assert original["source_sha256"][SOURCE_FILES[1]] == corrected["source_sha256"][SOURCE_FILES[1]], "Standalone runner changed"
    before_tree = ast.parse(git("show", f"{baseline}:{SOURCE_FILES[0]}"))
    after_tree = ast.parse((ROOT / SOURCE_FILES[0]).read_text())
    method_checks = {}
    for name in FUNCTIONS + ["run_experiment_2"]:
        before = next(n for n in before_tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        after = next(n for n in after_tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        old_nested = {n.name: ast.dump(n) for n in before.body if isinstance(n, ast.FunctionDef)}
        new_nested = {n.name: ast.dump(n) for n in after.body if isinstance(n, ast.FunctionDef)}
        assert old_nested == new_nested, f"Simulation/estimator/classifier helper changed: {name}"
        method_checks[name] = "Nested simulation, estimator, normalization and classifier functions AST-identical"
        if name == "run_experiment_2":
            assert ast.dump(before) == ast.dump(after), "V2 changed"
            method_checks[name] = "Entire function AST-identical; not rerun"

    rows = []
    old_pilot = json.loads((output / "original/toy_v1_metrics.json").read_text())
    new_pilot = json.loads((output / "corrected/toy_v1_metrics.json").read_text())
    for metric, old in old_pilot.items():
        new = new_pilot[metric]
        if metric != "mean_attractor_distance":
            assert old == new, metric
        rows.append({"experiment": "pilot", "condition": "seed=42", "metric": metric,
                     "original": old, "corrected": new, "delta": new - old})
    for experiment, filename, key, distance in [
        ("scaled", "toy_v1_scaled_results.csv", "seed", "mean_attractor_distance"),
        ("v1b", "toy_v1b_robustness.csv", "sigma", "attractor_distance"),
    ]:
        old = pd.read_csv(output / "original" / filename).set_index(key)
        new = pd.read_csv(output / "corrected" / filename).set_index(key)
        pd.testing.assert_frame_equal(old.drop(columns=distance), new.drop(columns=distance),
                                      check_exact=True)
        for condition in old.index:
            for metric in old.columns:
                a, b = float(old.loc[condition, metric]), float(new.loc[condition, metric])
                rows.append({"experiment": experiment, "condition": f"{key}={condition}",
                             "metric": metric, "original": a, "corrected": b, "delta": b - a})
    pd.DataFrame(rows).to_csv(output / "comparison.csv", index=False)

    patch = git("diff", baseline, "--", *SOURCE_FILES)
    (output / "production.patch").write_bytes(patch)
    archive_files = [p for p in git("ls-tree", "-r", "--name-only", baseline, "paper").decode().splitlines()]
    archive_hashes = {}
    for p in archive_files:
        digest = hashlib.sha256(git("show", f"{baseline}:{p}")).hexdigest()
        assert sha256(ROOT / p) == digest, f"Archived file changed: {p}"
        archive_hashes[p] = digest
    write_json(output / "manifest.json", {
        "baseline_commit": baseline,
        "corrected_source_base_commit": git("rev-parse", "HEAD").decode().strip(),
        "corrected_source_reference": "Apply production.patch to baseline_commit; source hashes in corrected/execution.json",
        "production_patch_sha256": sha256(output / "production.patch"),
        "reproduction_script_sha256": sha256(Path(__file__)),
        "reproduce_command": f"python experiments/reproduce_fixed_point_correction.py --baseline-ref {baseline} --output-dir /tmp/lem-correction-reproduction",
        "comparison": "All reported NN/baseline/cosine values exactly equal; RNG states exactly equal; only distances change.",
        "method_integrity": method_checks,
        "standard_deviation": "Sample SD across the 10 exported (6-decimal) seed values; pandas ddof=1. No seed SD for pilot or single-seed V1b.",
        "method_details": {
            "signature": "mean of last K stored states - beta * domain_vector (unchanged)",
            "domain_construction": "Normalized Gaussian user/domain vectors; second domain replaced with normalize(first domain + 0.4 * Gaussian noise). Original draws/order preserved.",
            "v1b_reseeding": "Seed 42 for fixed setup, then reset to 42 before EVERY sigma.",
            "time_indexing": "Pilot/train and all scalar test queries store x_0,...,x_(T-1). Vectorized Scaled/V1b training updates T times and retains x_(T-K+1),...,x_T. All estimators average exactly K stored states.",
            "classification": "Euclidean nearest neighbour within each domain; mean accuracy across two domains; fresh queries (pilot 200, Scaled 500, V1b 300 per domain).",
            "cosine_pairs": "Same user across domains; different user: random nonself offset in pilot, fixed np.roll(...,1) in Scaled/V1b.",
            "distance": "Euclidean norm of each tail mean minus fixed point, averaged over users and domains; alpha=0.4, beta=0.3.",
            "rounding": "Pilot/V1b outputs 4 decimals; Scaled per-seed outputs 6 decimals before summary, as in baseline.",
        },
        "v2": "Not rerun: run_experiment_2 and run_all.py unchanged; archived V2 figure/table retained.",
        "archived_paper_sha256_unchanged": archive_hashes,
    })
    print("PASS: exact equality of unaffected metrics, configurations and RNG states; archive hashes preserved.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["all", "original", "corrected", "compare"], default="all")
    parser.add_argument("--baseline-ref", default=BASELINE)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/results/fixed_point_correction")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--source-root", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.worker:
        worker(args.source_root.resolve(), output)
        return
    baseline = git("rev-parse", args.baseline_ref).decode().strip()
    output.mkdir(parents=True, exist_ok=True)
    if args.phase in {"all", "original"}:
        with tempfile.TemporaryDirectory(prefix="lem-original-") as folder:
            source_root = Path(folder)
            for filename in SOURCE_FILES:
                dest = source_root / filename
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(git("show", f"{baseline}:{filename}"))
            run_phase("original", source_root, output)
    if args.phase in {"all", "corrected"}:
        run_phase("corrected", ROOT, output)
    if args.phase in {"all", "compare"}:
        compare(output, baseline)


if __name__ == "__main__":
    main()
