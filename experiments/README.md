# LEM I — toy experiments and fixed-point correction

The active V1 code measures the Euclidean distance from each last-K-state mean to
`(alpha * user_signature + beta * domain_vector) / (alpha + beta)`.
The archived code omitted the denominator. Only the distance metric changes;
trajectories, random draws, signature estimation, NN classification and cosine
comparisons retain their original definitions.

The corrected paper is published as [Zenodo v2.2.0](https://doi.org/10.5281/zenodo.22973258).
The [GitHub working draft](../paper/lem_paper_correction_draft.pdf) is now the
unpublished v2.3.0 draft; use the Zenodo version for the current published paper.
See the [correction note](../paper/CORRECTION_NOTE.md) for before/after values.
The v3/v4 sources, PDFs and their original images are preserved in `paper/`.

## Scope of Toy V2

The [pre-specified multi-seed Toy V2 check and Amendment 1](toy_v2_robustness/PROTOCOL.md) retain the original sink/cycle generator and add exactly moment-matched GAUSS (originally DISK), UNIFORM and LINE controls, with calibration and evaluation seeds kept separate.
At noise 0.05 with common projection, the amended overall category is **incremental value (ZUSATZNUTZEN)**: GAUSS gives incremental value (H1 AUC 0.9942; G 0.8276; G+H1 0.9933; paired ΔAUC 0.1657, 95% interval [0.1087, 0.2347]), and UNIFORM gives clear advantage (EINDEUTIGER VORTEIL).
The [original results](results/toy_v2_robustness/) remain unchanged with R1 **H1 separates sink and cycle** and R2 **partial (TEILWEISE)**; the dated amendment was committed after those outputs already existed, before supplemental computation, and its [tables and results](results/toy_v2_robustness/amendment_1/main_tables.md) are reported as the new main analysis.
LINE remains a plausibility probe, and all 800 CYCLE/CYCLE_SHUF checks retain identical H0/H1 diagrams and g1–g3 within 1e-9: point-cloud H1 measures loop geometry, not temporal return.
These are synthetic pipeline checks of these generators and fixed predictors; they establish no LLM, user-identity or dynamical-attractor result.

## Reproduce Toy V2 and Amendment 1

Use a new local environment and the recorded package versions:

```bash
python3 -m venv .venv-toy-v2
.venv-toy-v2/bin/python -m pip install -r experiments/toy_v2_robustness/requirements.lock
.venv-toy-v2/bin/python -m pytest -q
# Reproduce original calculations to a fresh directory:
.venv-toy-v2/bin/python experiments/toy_v2_robustness/run_toy_v2_robustness.py --output-dir tmp/toy-v2-original-reproduction
# Read unchanged committed original per-seed values; compute new controls,
# shuffle checks and frozen joint predictors into another fresh directory:
.venv-toy-v2/bin/python experiments/toy_v2_robustness/run_amendment_1.py --output-dir tmp/toy-v2-amendment-reproduction
```

Each runner times five seeds in both projection modes first. CPU execution uses
one numerical thread per worker; process parallelism is selected if the serial
estimate exceeds 60 minutes. The recorded full times, including each benchmark
and reporting, were about 211 seconds originally and 279 seconds for Amendment 1.
The supplemental runner rejects an existing summary and the original output
directory. It verifies original output hashes before and after execution.

The original package contains `per_seed.csv`, `summary.json`, the original noise
plot, `environment.txt` and `benchmark.json`. The supplemental `amendment_1/`
package additionally contains evaluation predictions, all 800 shuffle checks,
model coefficients/scalers, paired Delta-AUC intervals, both versions' categories
and the original hashes. Every noise/projection/comparison has its own frozen
calibration; levels are never pooled. The primary 0.05 point uses evaluation
seeds 100–199; other noise levels use 20–39, so curves also reflect different seed
sets. An interval [1,1] is a finite-bootstrap outcome, not certainty elsewhere.

To regenerate manuscript tables from the two saved summaries and rebuild only
the working-draft PDF:

```bash
.venv-toy-v2/bin/python experiments/toy_v2_robustness/make_report_tables.py
cd paper
tectonic --keep-logs lem_paper_correction_draft.tex
pdftotext lem_paper_correction_draft.pdf -
```

The current published version remains v2.2.0, DOI 10.5281/zenodo.22973258.
The v2.3.0 working draft is not published; no historical PDF or Zenodo record is
replaced. The remaining sections document the unchanged V1 correction.

## Reproduce the matched V1 comparison

Run from the repository root, with Python 3.11 (the recorded run used 3.11.7):

```bash
python3 -m venv .venv-correction
.venv-correction/bin/python -m pip install -r experiments/requirements-v1-correction.lock
.venv-correction/bin/python experiments/reproduce_fixed_point_correction.py --output-dir /tmp/lem-correction-reproduction
```

Choose a fresh output directory. The default baseline is
`f15073f2cab3fb68857ffb16035d6675957c2e06`; `--baseline-ref` can select an explicit
alternative. The runner extracts the original source files into a temporary copy,
runs Pilot, Scaled and V1b in separate original/corrected output directories, and
compares all unaffected reported values exactly. It also checks parameter and
package equality, the final NumPy RNG state after each experiment, and SHA-256
hashes of the immutable archived PDFs, original figures and legacy v3 source.
Editorially corrected TeX, bibliography and changelog files are intentionally
allowed to differ; the test checks V2's scientific AST after excluding only
the named display and asset-directory changes. No V2, LLM call or cloud job is invoked.
`--phase original`, `--phase corrected` and `--phase compare` support separate
execution and a comparison-only recheck without simulations.

The checked-in [result package](results/fixed_point_correction/) contains:

- `original/` and `corrected/`: numerical JSON/CSV, execution metadata, RNG-state
  hashes and run logs. Generated PNGs in these folders are local outputs.
- `comparison.csv`: before/after values per seed or noise level, including deltas.
- `manifest.json`: source references, parameters/method conventions, reproduction
  command, patch hash and immutable historical-artifact hashes; each `execution.json` records
  Python/package versions, actual worker command and output hashes.
- `production.patch`: the exact simulation/standalone-runner diff from the baseline.
- `tests/`: the green run and two mathematical red checks.

`requirements-v1-correction.lock` pins the packages used for this CPU comparison;
`requirements.txt` remains the broader original full-suite dependency list.

## Regression tests and mathematical countercheck

```bash
.venv-correction/bin/python experiments/verify_fixed_point_regression.py --output-dir /tmp/lem-correction-tests
```

Five tests cover the scalar helper, vectorized/broadcast distances and all three
actual experiment bodies. Small, noiseless analytical fixtures use `s=(1,0)` and
`d=(0,1)`, so the independently derived fixed point is `(4/7,3/7)`. The helper
must return a distance below `1e-12`; the old target gives `3/14` instead.
Integration fixtures reduce only configuration constants and control random draws;
they leave the production dynamics, metric expressions and return paths intact.

The verifier keeps the helper interface and production call sites while replacing
only the denominator to produce five assertion failures, then runs the actual
archived experiment bodies and obtains three distance assertion failures. The
corrected source passes all five tests. No import/interface failure counts as red.

## Regenerate and compile the correction manuscript

```bash
.venv-correction/bin/python experiments/update_correction_artifacts.py --results-dir /tmp/lem-correction-reproduction
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error lem_paper_correction_draft.tex
```

The generator verifies input hashes and recreates only `figures_correction/` and
`tables_correction/`. The correction TeX includes those tables and images; the
unchanged V2 image is still read from the archive. Render the PDF pages with
`pdftoppm -r 120 -png lem_paper_correction_draft.pdf /tmp/lem-correction-page`
and inspect the changed text, tables, figures and version notice.

## Original experiment settings retained

| Setting | Pilot | V1 Scaled | V1b |
|---|---|---|---|
| Dimension / users / domains | 4 / 20 / 2 | 512 / 500 / 2 | 512 / 500 / 2 |
| T / last K | 60 / 20 | 200 / 40 | 200 / 40 |
| Alpha / beta | 0.4 / 0.3 | 0.4 / 0.3 | 0.4 / 0.3 |
| Noise sigma | 0.15 | 0.15 | 0.05, 0.15, 0.30, 0.50, 0.75, 1.00, 1.50, 2.00 |
| Test queries per domain | 200 | 500 | 300 |
| Seeds | 42 | 42, 123, 999, 2024, 7, 314, 1337, 77, 256, 88 | 42, reset before each sigma |

Scaled/V1b vectorized training stores the last K of T updates (`x_1` to `x_T`).
Pilot trajectories and all scalar test queries store `x_0` to `x_(T-1)` and use
the last K stored states. Both conventions are preserved, including random-call
order. The second domain is the normalized first domain plus `0.4 * Gaussian noise`,
not an independent final domain draw.

The signature remains `mean_last_K_states - beta * domain_vector`. At the noiseless
fixed point this is `(4/7)*user_signature + (9/70)*domain_vector`, so cross-domain
cosines include residual domain dependence. Within each domain, NN classification
uses Euclidean distances to training signatures from fresh query trajectories.

Scaled reports mean, **sample SD (`ddof=1`)**, min and max across ten six-decimal
per-seed exports. Pilot and V1b values use four decimals; no seed SD is claimed for
single-seed V1b. V1b accuracy falls from 1.0000 at sigma=0.15 to 0.2300 at 0.30;
there is no measurement at 0.25. Accuracy 0.0200 at 0.50 exceeds baseline 0.0020.
Tail-mean distance scaling depends on dimension, window length and temporal
correlation; it is not the standard deviation of an individual state.

## Separate full-suite runner

The existing standalone runner remains available. The numerical source also
handles a missing `IPython` display import when run outside a notebook. From
an output directory, after installing `experiments/requirements.txt`, run the
repository's `experiments/run_all.py`
(use `--skip-pilot` to omit the pilot). It executes corrected V1 plus Toy V2
with the unchanged scientific AST and optional display setup; it writes PNGs
to `assets/` and JSON/CSV into the current working directory.
Use a separate output directory to protect manuscript assets. V2's archived
ratio is 161.6 for that one constructed pair; V2 was not rerun for the original
V1 metric-only correction.
