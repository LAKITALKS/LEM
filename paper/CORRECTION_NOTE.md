# LEM I — Fixed-point distance correction

**Status:** Corrected working draft, 7 September 2026. Not a new Zenodo version; no new DOI.
**Author:** Lazaros Varvatis.
**Baseline:** `f15073f2cab3fb68857ffb16035d6675957c2e06` (verified `origin/main` at checkout).
**Branch:** `codex/lem-i-fixed-point-correction`. The corrected numerical source is identified by [`production.patch`](../experiments/results/fixed_point_correction/production.patch) against this baseline and SHA-256 values in the [manifest](../experiments/results/fixed_point_correction/manifest.json) and execution records; the enclosing Git commit identifies the complete draft.

## Error and scope

The dynamics already imply `x_star = (alpha*s + beta*d)/(alpha+beta)`, as stated in the paper. Pilot, Scaled and V1b instead measured the distance to the unnormalised numerator. All three production paths now call `fixed_point_distance`, which includes the denominator. This changes only the reported distance; no simulation update, seed, random-call order, window, grid, query count, classifier or signature estimator was changed.

## Recomputed results

The table compares actual reruns of the original and corrected code in the same Python environment. Values are rounded here to four decimals; the CSVs retain the original export precision.

| Metric / condition | Original | Corrected |
|---|---:|---:|
| Pilot mean distance | 0.2166 | 0.0917 |
| Scaled mean distance, ten seeds | 0.7893 | 0.7597 |
| Scaled distance sample SD (`ddof=1`) | 0.0010 | 0.0008 |
| Scaled NN accuracy | 1.0000 | 1.0000 |
| Scaled same-user cosine | 0.3570 | 0.3570 |
| Scaled different-user cosine | 0.0022 | 0.0022 |
| V1b distance, sigma=0.05 | 0.3319 | 0.2533 |
| V1b distance, sigma=0.15 | 0.7895 | 0.7599 |
| V1b distance, sigma=0.30 | 1.5348 | 1.5198 |
| V1b distance, sigma=0.50 | 2.5419 | 2.5329 |
| V1b distance, sigma=0.75 | 3.8054 | 3.7994 |
| V1b distance, sigma=1.00 | 5.0703 | 5.0659 |
| V1b distance, sigma=1.50 | 7.6017 | 7.5988 |
| V1b distance, sigma=2.00 | 10.1339 | 10.1317 |

All exported non-distance metrics agree **exactly** for all ten Scaled seeds, all eight V1b noise levels and the pilot. Final NumPy RNG-state hashes also agree after each experiment. The scaled distance mean changes from 0.7892870 to 0.7596518; its sample SD changes from 0.0009725116 to 0.0008297591. Scaled cosine means and SDs are unchanged by the metric correction. The table now consistently uses the Pandas sample-SD convention: different-user cosine SD is 0.0018 at four decimals (the archived paper printed 0.0019), and the original distance SD would round to 0.0010 (the archived paper printed 0.0009). These manuscript values were aligned to the actual aggregates rather than changing the calculations.

Toy V2 was not rerun: its entire function and the standalone runner are unchanged. Its archived figure and reported SNR 161.6 are retained; no new V2 result is claimed.

## Descriptions aligned with the executed method

- The noise grid has measurements at 0.15 and 0.30, not 0.25. Accuracy is 1.0000 then 0.2300; no precise phase-transition location is inferred. At 0.50, accuracy 0.0200 is ten times the 0.0020 random baseline.
- Each signature averages exactly K stored states. Vectorized training ends at `x_T`; scalar test trajectories and the pilot end at `x_(T-1)`. V1b uses 300 queries per model versus 500 in Scaled, and resets seed 42 before every sigma.
- The second domain is constructed by normalizing the first domain plus `0.4 * Gaussian noise`. It is not an independent final domain draw.
- The published estimator `mean_last_K_states - beta*domain_vector` is retained. Its noiseless limit is `(4/7)*s + (9/70)*d`, so cross-domain cosines include a residual domain component.
- Distance is an average of norms of temporal-mean errors. The manuscript distinguishes the stationary per-coordinate state variance `sigma²/(1-rho²)` from the correlated K-state mean variance. For `rho=0.3`, `K=40`, `d=512`, the corresponding RMS norm is about `5.069*sigma`; a mean norm is not exactly that RMS. No extra sigma sweep was run.

## Reproduction and verification

From the repository root:

```bash
python3 -m venv .venv-correction
.venv-correction/bin/python -m pip install -r experiments/requirements-v1-correction.lock
.venv-correction/bin/python experiments/reproduce_fixed_point_correction.py --output-dir /tmp/lem-correction-reproduction
.venv-correction/bin/python experiments/verify_fixed_point_regression.py --output-dir /tmp/lem-correction-tests
.venv-correction/bin/python experiments/update_correction_artifacts.py --results-dir /tmp/lem-correction-reproduction
cd paper
tectonic --keep-logs lem_paper_correction_draft.tex
```

Use a fresh output directory. The original numerical source is extracted from the baseline into a temporary working copy; original and corrected outputs are separated. The [result package](../experiments/results/fixed_point_correction/) records full experiment constants, Python 3.11.7, package versions, NumPy build details, execution commands, output hashes, and `ddof=1`. The lock file fixes all installed V1 packages. A first original run completed its numerical outputs but failed during metadata extraction of a derived plotting list; that exporter was corrected and the complete original run repeated. Its numerical files were verified identical to the first run before inclusion. Only the complete, verified comparison is included.

**Regression evidence:** Five corrected tests pass. Reinstating only the old target in the existing helper produces five mathematical assertion failures. Separately, all three actual archived experiment bodies fail the production-path tests. The analytical two-dimensional case has fixed point `(4/7,3/7)`: corrected distance is below `1e-12`; the old target produces `3/14 = 0.214285714...`. Logs are included in [`tests/`](../experiments/results/fixed_point_correction/tests/). The integration fixtures change only test configuration constants and deterministic random inputs, preserving production metric expressions.

## Archive and manuscript

Every pre-existing file under `paper/`, including both published PDFs, v4/legacy sources, bibliography, changelog and all three original figures, is SHA-256-identical to the baseline. The current folder’s original standalone archive was not edited. The correction is separate: [`lem_paper_correction_draft.tex`](lem_paper_correction_draft.tex), [`lem_paper_correction_draft.pdf`](lem_paper_correction_draft.pdf), `figures_correction/`, and `tables_correction/`. Tables and V1 figures are regenerated directly from checked numerical files. The archive DOI [10.5281/zenodo.21268033](https://doi.org/10.5281/zenodo.21268033) continues to identify v2.1.0, not this draft.

The build and visual-check results are recorded in [`CORRECTION_VALIDATION.md`](CORRECTION_VALIDATION.md).

No LEM-II pipeline, paid model calls, Modal jobs, main merge, tag, release, Zenodo publication, DOI creation or third-party messages are part of this correction.
