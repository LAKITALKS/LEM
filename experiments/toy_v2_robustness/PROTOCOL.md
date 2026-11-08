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
