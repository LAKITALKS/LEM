# Pre-specified multi-seed check of Toy V2

Author and sole contributor: Lazaros Varvatis.
Working draft for v2.3.0; no publication, release, or Zenodo action.
This protocol must be committed before any simulation, benchmark, or numerical
result inspection. The user's clarification supersedes the initial R2 and g1
specifications. Report outcomes as obtained; do not change classes, statistics,
seeds, directions, or thresholds after inspecting evaluation results. Stop before
computation if a specification cannot be implemented.

## Original generator and window

Source: `experiments/lem_simulations.py`, commit
`e2c0487151ab1341ead26dba60ec44c72bc62af2`, `run_experiment_2`, lines
566–651 (parameters 578–588; numerical helper functions 591–617;
ODE integration and embedding 629–634). Copy the numerical helper functions
verbatim into a new module, preserving their bodies and random-call order for
the original reproduction path. Do not modify the original file or V1 files.

SINK: fixed-point sink with lambda=0.5, initial state (2, 1.5).
CYCLE: Van der Pol limit cycle with mu=1.5, initial state (0.1, 0.1).
Use scipy.integrate.odeint, T=40, dt=0.1, linspace with 401 points.
The original has **no burn-in removal and no subsampling**. W is precisely these
401 points, including transients and the initial state, the point set entering
persistence. Embed in 16 dimensions by an unnormalized 2-by-16 standard-normal
matrix and add independent coordinatewise Gaussian noise with the stated sigma.
Use ripser(maxdim=1), Euclidean distance, maximum finite H1 death-minus-birth;
empty finite H1 gives zero. No rescaling of embedded clouds.

## Four classes for each seed

SINK and CYCLE are exactly the original deterministic 2-D trajectories.
Construct the following controls in 2-D before projection and observation noise.
Use sample covariance with ddof=1 throughout.

**DISK:** a stationary 2-D OU process discretized as AR(1), with a common
coefficient phi pooled over the two centered coordinates of CYCLE on W:
phi = sum_t dot(c[t], c[t-1]) / sum_t dot(c[t-1], c[t-1]).
Use CYCLE's sample mean m and covariance C as the stationary parameters.
Draw x[0] from N(m,C); evolve x[t]=m+phi*(x[t-1]-m)+epsilon[t], with
independent epsilon distributed N(0,(1-phi^2)*C). Length equals |W|.
If abs(phi)>=1 or either required covariance is not positive definite, stop;
do not clip phi or silently substitute a different generator.

Additionally, exactly affine-match the realized DISK on W: center on its own
sample mean, whiten using its own symmetric inverse covariance square root,
color using CYCLE's symmetric covariance square root, and shift to m.
Report maximum absolute residuals of sample means and sample covariances before
projection, expected at floating-point rounding level. The purpose is to prevent
finite-sample scale mismatch: near phi=1 the approximate variance factor
1-(1+phi)/((1-phi)*N) may be far below one. That expression is a motivating
approximation, not an exact correction formula.
A fixed affine map commutes with the scalar AR(1) transition and retains phi
algebraically. Because this map is estimated from the same realized path, the
corrected ensemble need not itself be a stationary Gaussian OU process with
independent innovations, nor must its empirical lag-1 coefficient equal phi.
Report empirical pooled lag-1 values before and after correction as diagnostics.

**LINE:** a centered finite sinusoid along CYCLE's first principal axis.
Estimate the dominant nonzero frequency using the largest summed periodogram
power of both centered CYCLE coordinates (rfft on W, dt=0.1; first frequency in
a tie). Use phase zero, subtract the finite sinusoid's sample mean, and choose
amplitude from its actual ddof=1 variance so that LINE's total sample variance
exactly equals trace(C). Shift to CYCLE's sample mean. Do not use A^2/2.
Report mean and total-variance matching residuals before projection.

## Projection, randomness, and seeds

Primary: one common random projection per seed for all four classes.
Secondary: an independent projection per trajectory, as in the original.
Observation noise is independent across classes. In each mode use
numpy.random.seed(seed), embed SINK then CYCLE in original order; the independent
mode must preserve the original SINK/CYCLE random sequence exactly. In the common
mode draw the matrix once, then draw class-specific noise in class order.
Controls use a separate RandomState seeded by seed, so their construction does
not consume the embedding RNG stream. Classes are ordered SINK,CYCLE,DISK,LINE.
Reuse seed identities across noise levels and projection modes; each run resets
the RNG. CPU only, one BLAS/OpenMP thread per worker. Parallel workers, if used,
handle whole seed/mode/noise units and results are sorted before export.

Primary sigma=0.05: 200 seeds per class, calibration 0–99, evaluation 100–199.
Noise sweep sigma=0.01,0.025,0.1,0.2,0.5: 40 seeds per class and level,
calibration 0–19, evaluation 20–39. Both projection modes use all these seeds.

## Statistics and calibration

Compute on the noisy 16-D point set W unless stated otherwise:

- H1: maximum finite persistent H1 lifetime.
- g1: radius of gyration sqrt(mean squared distance from the centroid), on all W.
- g1_second_half: same statistic on indices len(W)//2 onward, separately reported
  to retain comparability with the September Va definition; not used in R2.
- g2: largest divided by second-largest sample PCA covariance eigenvalue.
- g3: population SD divided by mean of distances to the centroid in the first
  two principal components. Use full deterministic SVD, not randomized PCA.

Before projection, g1 and g2 of DISK are matched by construction. They are
plausibility probes, with AUC expected about 0.5 under common projection;
independent observation noise can perturb the equalities in 16-D. The actual
geometric competitor in Vb is g3. Independent projections can change geometry.
No numerical jitter or rounding is added to force ties or improve AUCs.

Comparisons: Va SINK vs CYCLE; Vb CYCLE vs DISK; Vc CYCLE vs LINE.
The second listed class is the positive label. For each statistic/comparison/
noise/projection, select both sign (+1 or -1) and threshold solely on calibration
seeds by maximizing Youden J=TPR-FPR. Candidate thresholds are the unique signed
calibration values plus positive infinity, prediction is signed score >= threshold.
Break ties by sign +1 first, then the highest threshold. Freeze sign and threshold
and apply unchanged to evaluation seeds; AUC also uses that frozen sign. No
orientation re-selection on evaluation data. Record calibration J, sign, threshold,
and calibration seed IDs. An infinite threshold is exported as the string
"Infinity" in strict JSON.

Report evaluation AUC and accuracy, each with percentile 95% bootstrap intervals
(2.5th and 97.5th percentiles), 1000 draws, RandomState(2300). Resample paired seed
IDs with replacement within each comparison, retaining both classes for every
drawn seed. Use the same bootstrap index sequence across statistics and conditions;
do not refit thresholds or signs in bootstrap samples. Classes are balanced.

## Interpretation rules (evaluation, sigma=0.05, common projection only)

R1: report exactly **H1 trennt Senke und Zyklus** if AUC(H1,Va)>=0.95;
otherwise report **R1 nicht erfüllt**.

R2 is decided **only on Vb** using H1 and g1–g3:

- **EINDEUTIGER VORTEIL**: H1 AUC>=0.95 and all g AUCs<=0.65.
- **TEILWEISE**: H1 AUC>=0.95 and best g AUC>0.65 and <0.95.
- **KEIN EINDEUTIGER VORTEIL**: H1 AUC>=0.95 and any g AUC>=0.95.
- **NICHT ROBUST**: H1 AUC<0.95.

The boundary 0.65 belongs to EINDEUTIGER VORTEIL, 0.95 to KEIN EINDEUTIGER
VORTEIL. Decisions use point estimates without rounding. Vc is a reported
plausibility probe only; g2 AUC>=0.95 is expected because geometry suffices to
separate a line from a cycle. Vc does not determine R2. No inference about LLMs,
users, identity, or dynamical attractors follows from this toy.

## Execution and artifacts

After committing this protocol, implement the CPU runner and tests. First time
five seeds (0–4, sigma=0.05, both modes, all classes), extrapolate to 800
seed/mode/noise units and report the estimate. If over 60 minutes, use deterministic
process parallelism; otherwise serial execution is allowed. Benchmark does not
alter the design. Reproducibility tests: three seeds twice bitwise; original
independent-projection seed 1337 must round to SINK 0.0326 and CYCLE 5.2649.

Export to experiments/results/toy_v2_robustness/: per_seed.csv, summary.json
(all statistics, AUCs, accuracies, intervals, thresholds, categories, diagnostics,
protocol commit, benchmark and actual runtime), toy_v2_robustness.png (noise sweep),
and environment.txt (Python, platform and package versions). Record all 3200
class observations; keep calibration and evaluation labels. Commit results as
obtained, then update the working paper, PDF, README files and v2.3.0 changelog.

## Amendment 1 (8 Nov 2026, after external review; original B2 already computed)

**9 Nov 2026.** Clarification: the amendment was drafted by reviewers who had not seen the results; the original results had already been computed and reported in the execution chat before it arrived (see paper, Section 4.3).

Author and sole contributor: Lazaros Varvatis. The requester supplied this
amendment following external review attributed to Eudora and stated that it was
fixed before Lazaros or Claude saw results. This is **not** a preregistration of
the original run: B2 had already completed, its outputs were committed in
`ee90b9f`, and the executing assistant had inspected and reported its outcome in
the chat before this amendment arrived. The amendment is committed before any
supplemental fitting or computation. Preserve all original result files byte for
byte; report the original version alongside the amended main analysis. No
claim is made here about when other analysts read the chat.

### E1: joint geometric predictor

G is logistic regression on standardized g1, g2, g3, with L2 regularization,
C=1.0 fixed. G+H1 uses the same model with H1 as its fourth feature. Fit both
StandardScaler and logistic regression only on calibration seeds, separately for
each comparison, projection and noise level. Use StandardScaler's population
variance convention, solver lbfgs, intercept enabled, class_weight=None,
max_iter=10000, tol=1e-10, random_state=2301. The second class (the control) is
positive, as in the original protocol. Evaluate fixed predicted probabilities
on evaluation seeds without refitting, reorientation or hyperparameter tuning.
Report AUC(H1), AUC(G), AUC(G+H1), and paired
Delta AUC = AUC(G+H1)-AUC(G), each with 95% bootstrap intervals. H1 retains its
calibration-selected sign and threshold; for GAUSS these are the original DISK
values. For G and G+H1, also report accuracy at the fixed probability cutoff 0.5;
this cutoff is descriptive and does not enter categorization.

### E2: controls and preservation of the original version

In the amended analysis, rename DISK to **GAUSS**. Its construction stays exactly
as specified above: an affine-matched Gaussian OU cloud, not a uniform disk.
Keep DISK as the historical name in the unchanged original outputs.

Add **UNIFORM**: N=|W| independent uniform points in an ellipse, then exactly
affine-match sample mean and sample covariance on W to CYCLE. Draw independent
angles uniformly on [0,2*pi) and radii sqrt(U), U uniform on [0,1), forming uniform
points in the unit disk. Color with the symmetric square root of CYCLE's
covariance to obtain an ellipse; then center, whiten by its own ddof=1 sample
covariance, color by CYCLE's covariance and shift to its mean. Use a separate
RandomState(seed+1000000). Report matching residuals. As for GAUSS, affine matching
couples the final finite sample, so independence/uniformity describes the draws
before the sample-dependent exact matching step.

Use exactly the original seeds, windows, projection matrices and observation
noise for all existing classes. In the common-projection mode, use the same
matrix for UNIFORM, with new independent observation noise. In the independent
mode, draw a new projection and noise for UNIFORM. To preserve the original RNG
stream, append UNIFORM **after** SINK,CYCLE,GAUSS,LINE in both modes. Do not alter
any original per-seed value. LINE remains a plausibility probe.

### E3: shuffled cycle

CYCLE_SHUF is the exact same **noisy embedded** CYCLE point cloud in random order,
using a permutation from RandomState(seed+2000000). Do not regenerate projection
or noise for it. Check both persistence diagrams H0/H1 after lexicographic sorting
of (birth,death), and g1–g3, for equality to CYCLE with absolute tolerance 1e-9,
relative tolerance zero (equal infinite H0 endpoints are allowed). Report the
maximum differences and failures; do not change tolerance after inspection.
The second-half g1 is order-dependent and is excluded from this invariance test.
The paper must state: point-cloud H1 measures loop geometry, not temporal return.

### E4: evaluation and bootstrap

Report CYCLE against GAUSS, UNIFORM, and LINE separately, retaining Va for
continuity. Each noise level and projection mode is evaluated separately; never
pool across noise levels. Use the original calibration/evaluation seed splits.
Bootstrap whole evaluation seed IDs, retaining all classes and predictions of a
seed together, 1000 draws with RandomState(2300). Use the same seed-index draws
across controls and feature arms within a condition. Compute percentile 95%
intervals for AUCs and paired Delta AUC from those joint resamples; never refit
models, scalers, H1 signs or thresholds in a resample.

### E5: amended categories

Decide separately for GAUSS and UNIFORM at sigma=0.05, common projection,
on evaluation seeds, using point estimates without rounding:

- **EINDEUTIGER VORTEIL**: AUC(H1)>=0.95 and AUC(G)<=0.65.
- **ZUSATZNUTZEN**: AUC(H1)>=0.95, AUC(G)>0.65, Delta AUC>=0.05,
  and the lower 95% bootstrap limit of Delta AUC is strictly positive.
- **GLEICHSTAND**: AUC(H1)>=0.95, but neither of the two conditions above holds.
- **NICHT ROBUST**: AUC(H1)<0.95. This means H1 alone does not recognize the
  constructed loop reliably; it does not mean "not significant".

Overall category is the weaker of GAUSS and UNIFORM, ordered from strongest to
weakest: EINDEUTIGER VORTEIL > ZUSATZNUTZEN > GLEICHSTAND > NICHT ROBUST.
A tie is a valid reportable result. LINE does not enter the category.
The original R1/R2 and all original tables remain reportable as the original
committed analysis, not relabeled as amended outcomes.

### Supplemental artifacts and reporting

Implement supplemental computation in a separate runner, preserving the original
runner's scientific logic. Write new outputs only under
`experiments/results/toy_v2_robustness/amendment_1/` (per-seed statistics,
summary with fitted model coefficients/scalers, AUCs, intervals, Delta AUCs,
thresholds, both versions' categories, diagnostic checks, runtime, environment,
and a separate noise-sweep plot). Record this amendment's commit hash and SHA-256
hashes of all original outputs before and after execution. Any mismatch is a
failure. Supplemental files do not overwrite original outputs. The amended
analysis becomes the paper's main result; state the amendment date, external-review
reason, and original results' prior existence, and additionally report the
original committed R2 category. Keep interpretation limited to this toy.

## Label mapping (9 Nov 2026)

This is an editorial display mapping for the paper and README files. It does not
change either frozen analysis, its decision rules, or the stored result labels.
All frozen result files (including summary.json and per_seed.csv) and their
SHA-256 hashes remain unchanged.

| English name | Label in the frozen result files |
| --- | --- |
| clear advantage | `EINDEUTIGER VORTEIL` |
| incremental value | `ZUSATZNUTZEN` |
| tie | `GLEICHSTAND` |
| not robust | `NICHT ROBUST` |
| partial | `TEILWEISE` |
| no clear advantage | `KEIN EINDEUTIGER VORTEIL` |
| H1 separates sink and cycle | `H1 trennt Senke und Zyklus` |
