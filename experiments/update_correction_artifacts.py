"""Build correction-only tables/figures from verified V1 result files."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "experiments/results/fixed_point_correction")
    parser.add_argument("--paper-dir", type=Path, default=ROOT / "paper")
    args = parser.parse_args()
    manifest = json.loads((args.results_dir / "manifest.json").read_text())
    execution = json.loads((args.results_dir / "corrected/execution.json").read_text())
    source = args.results_dir / "corrected"
    files = ["toy_v1_scaled_results.csv", "toy_v1_scaled_summary.csv", "toy_v1b_robustness.csv",
             "assets/toy_v1_scaled_results.png", "assets/toy_v1b_robustness.png"]
    for filename in files:
        assert hashlib.sha256((source / filename).read_bytes()).hexdigest() == execution["outputs_sha256"][filename], filename
    figures = args.paper_dir / "figures_correction"
    tables = args.paper_dir / "tables_correction"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    for filename in files[-2:]:
        shutil.copyfile(source / filename, figures / Path(filename).name)
    summary = pd.read_csv(source / "toy_v1_scaled_summary.csv", index_col=0)
    seeds = pd.read_csv(source / "toy_v1_scaled_results.csv")
    columns = list(summary.index)
    pd.testing.assert_frame_equal(summary, seeds[columns].agg(["mean", "std", "min", "max"]).T,
                                  check_exact=False, atol=1e-14, rtol=1e-14)
    rows = [r"Random baseline accuracy & 0.0020 & --- & --- & --- \\"]
    for metric, label in [("nn_reidentification_accuracy", "NN re-id accuracy"),
                          ("mean_attractor_distance", "Mean attractor distance"),
                          ("same_user_cosine_similarity", "Cosine (same user)"),
                          ("different_user_cosine_similarity", "Cosine (different user)")]:
        rows.append(label + " & " + " & ".join(f"{summary.loc[metric, c]:.4f}" for c in ["mean", "std", "min", "max"]) + r" \\")
    (tables / "toy_v1_scaled.tex").write_text(
        "\\begin{tabular}{lrrrr}\n\\toprule\n"
        + r"\textbf{Metric} & \textbf{Mean} & \textbf{Std} & \textbf{Min} & \textbf{Max} \\" + "\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    robustness = pd.read_csv(source / "toy_v1b_robustness.csv")
    rows = []
    for row in robustness.itertuples(index=False):
        rows.append(f"{row.sigma:.2f} & {row.nn_accuracy:.4f} & {row.attractor_distance:.4f} & {row.cosine_same:.4f} & {row.cosine_diff:.4f}" + r" \\")
    (tables / "toy_v1b.tex").write_text(
        "\\begin{tabular}{rrrrr}\n\\toprule\n"
        + r"$\sigma$ & \textbf{NN Acc.} & \textbf{Attr. Dist.} & \textbf{Cos. (same)} & \textbf{Cos. (diff)} \\" + "\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    print(f"Generated two tables and two figures from baseline {manifest['baseline_commit']} and its verified correction.")


if __name__ == "__main__":
    main()
