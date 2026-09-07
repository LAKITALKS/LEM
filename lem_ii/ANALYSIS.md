# LEM-II Light analysis implementation

This is a **prespecified protocol**, not a public preregistration. The checked-in
execution configuration disables real study analysis and confirmation. The command
below generates only clearly marked synthetic software fixtures. Their labels are
independent of the random activation trajectories; their scores are not LLM
research findings, and no method is required to win.

```sh
.venv-lem-ii/bin/python -m lem_ii.evaluation \
  --config lem_ii/config/study.json \
  --output /tmp/lem-ii-fixture-analysis \
  --diagnostic-permutations 3 --diagnostic-bootstrap 100

.venv-lem-ii/bin/python -m unittest lem_ii.tests.test_features_evaluation -v
```

The small repetition counts above are explicit, recorded **fixture-only diagnostic
overrides**. Omitting them exercises the configured 199 label randomizations and
2,000 profile-cluster resamples. Output contains four frozen endpoint artifacts,
their JSON manifests and hashes, and `fixture_analysis.json`. Fitted artifacts and
raw fixtures belong outside Git. Joblib artifacts are executable serialization:
load only trusted artifacts produced locally by this pipeline, after digest checks.

## Data and information boundary

`join_record_inputs(dialogue_rows, label_rows, state_arrays, config)` explicitly
joins separately stored labels by unique dialogue ID. Missing/duplicate IDs fail.
The feature record has exactly these keys:

```text
dialogue_id, profile_id, regime, split, topic_family, formulation_family,
contexts, states, eligible_for_scientific_analysis, data_kind
```

`states` is an aligned T × H array from the primary layer 36; `contexts[t]` is
exactly the visible pre-answer input corresponding to `states[t]`. It includes
previous assistant answers, but excludes the current answer and every later turn.
Only `contexts[endpoint-1]` and `states[:endpoint]` become features. The collector
is responsible for verifying actual chat-template token positions before saving;
analysis cannot reconstruct that guarantee from arbitrary supplied strings.

Metadata are inspected for split and provenance checks, never vectorized. Unknown
feature record fields, profile IDs in text, named generator metadata fields,
duplicate dialogue IDs, cross-split topic/formulation families, changing labels
inside a profile, nonfinite states and misaligned context/state arrays fail.
Each development/validation profile contributes both topic sessions. Real study
records additionally must match the complete prespecified session assignment.
All prefixes/windows inherit their dialogue's phase; no turn is an independent
training example in the endpoint classifiers.

## Fixed transformations and paired feature paths

For each endpoint in **12, 16, 20, 24**, fit a distinct preprocessing/classifier
artifact using developmental prefixes through that endpoint. Deterministic full
SVD PCA with 8 components is fitted to those raw primary-layer developmental
states; no whitening is applied. Coordinate standardization is then fitted to
their projected coordinates. Neither validation nor confirmation is used to fit
PCA, scaling or a vocabulary.

The text block concatenates word (1,2)-gram and character (3,5)-gram TF-IDF, each
capped at 12,000 learned terms, with sublinear term frequency and `min_df=1`.
Scikit-learn's lowercase, smoothed-IDF and L2 vector normalization defaults apply.
Eight developmental-row-standardized style features count characters, words and
lines, mean word length, unique lowercase word fraction, and question marks,
exclamation marks and digits per character. Word parsing uses `\b\w+\b`.

Every activation method uses the trailing **12 genuine states**, giving constant
window size across endpoints. Static features are the current PCA8 state, window
mean and population standard deviation, plus the four time-invariant
pairwise-distance mean, population standard deviation, maximum and radius of
gyration (28 columns total). The added geometry/time block contains only
step-norm mean/std/max; adjacent-step-cosine mean/std; lag-1/2/3
displacement mean/std; and non-topological summaries of the delay representation
(17 additional columns). Thus the second direct comparison isolates the addition
of explicitly temporal and delay-representation features to the identical static
and unordered-geometric base.
An undefined cosine involving a zero-length step is explicitly 0.

Delay rows concatenate `[x_t,x_(t+1),x_(t+2)]`: lag 1, dimension 3,
**10 points in 24 coordinates**. Their pairwise-distance mean/std/max, radius of
gyration and coordinate-standard-deviation mean/max are part of the common
non-topological base. Euclidean Vietoris–Rips persistence uses H0/H1, coefficient
field 2 and no filtration cutoff on those same points. For each dimension, the
strictly positive finite bars provide count, total persistence, maximum
persistence and natural-log persistence entropy. Infinite bars are excluded.
A genuinely empty finite diagram has zero summary statistics. A missing/short
sequence instead raises `NotEvaluable`; it is never zero-filled or interpolated.

The executable paths are exactly:

1. `text`
2. `text_static`
3. `text_static_geometry_time`
4. `text_static_geometry_time_topology`

Each next matrix contains the previous matrix as an exactly equal column prefix.
Dense activation blocks have separate scalers fitted only to developmental rows.
The topology ablation adds exactly eight columns and leaves all shared columns,
training cases and classifier choices unchanged.

Every method fits the same L2 logistic classifier (`lbfgs`, balanced class weights,
2,000 maximum iterations) at **C = 0.1, 1, 10**. Validation Balanced Accuracy selects
C separately per method, with ties choosing the smallest C. Failed convergence
raises an error. Developmental training is not refitted on validation after
selection. The frozen artifact holds the exact preprocessing and selected models;
its manifest records tuning trials, fit IDs, topic/formulation families, profile
assignments and configuration/artifact hashes.

## Paired uncertainty and controls

At primary endpoint 24, report Balanced Accuracy differences for the three adjacent
paths. Bootstrap **profile bundles**, stratified by regime, retaining both topic
sessions whenever a profile is sampled. The same indices are used for every
method. The three differences use Bonferroni-adjusted **98.3333% percentile
intervals** with 2,000 resamples. Validation scores and intervals are descriptive
because validation selected C. They are never reported as independent confirmation.

An additional utility computes simultaneous conservative intervals for all
3 contrasts × 4 fixed endpoints, at **99.5833% per contrast/endpoint**. This is a
separate secondary family. Raw developmental/validation method-BA curves are
unbanded and descriptive. The earliest recognition summary is the first grid
point beginning two consecutive point estimates at least 0.70; turn 24 alone
cannot qualify. Otherwise the result is `not_reached_in_observed_grid`. The
threshold and grid are never selected using confirmation results.

These intervals are conditional on the fixed small set of topic/formulation
families. They resample profiles, not a topic population. Two families per phase
cannot support broad claims about new topics in general; repeated sessions or
many turns do not remove that limit. Regime prediction concerns controlled
synthetic behavior in the same known profile set, not new people, personality,
identity, hidden cognition, post-reset memory or additional information in an
information-theoretic sense. Light makes no attractor claim.

Label counterchecks permute the regime assignment once per whole profile bundle,
using the same mapping in development and validation and preserving class counts.
Every classifier and its three-choice tuning are rerun. Label-independent
developmental transformations can be reused. No turn-label shuffle occurs; the
countercheck is diagnostic, not a confirmation p-value.

Time controls seed a permutation of the preceding 11 states while keeping the
current, 12th window state fixed, **then rebuild its delay embedding**. This keeps
the entire static baseline, including the current state, invariant. The frozen
classifier is evaluated without refitting; a changed score
can also reflect distribution shift. Window means/stds and plain point-cloud
persistence should be invariant under reorderings. Ordinary point-cloud
persistence itself is not temporal. Degenerate trajectories need not
change after rebuilding. Controls diagnose sensitivity, not causal temporal
mechanisms.

## Real-state smoke and future study gates

`compute_technical_features(states, config, endpoint=24)` exercises every static,
geometry/time and topology calculation on saved genuine smoke states. This
technical path unit-normalizes each full hidden vector without fitting PCA or a
scaler; the report explicitly identifies this different, **smoke-only**
representation. It also reports a rebuilt shuffled embedding and the expected
point-cloud permutation invariance. It never trains a classifier. A short second
session is explicitly not evaluable at the 12-state feature window.

Future genuine data use `data_kind="real_model_study"` and scientific eligibility
true. `execution.allow_scientific_analysis` must be separately authorized before
the fit/tune APIs accept them. Fixtures, smoke and study records cannot be mixed
or promoted through a fitted artifact. Current checked-in flags remain false.

`predict_confirmation_grid_once(artifact_directory, records, config, output_path)`
is the future complete-grid API requiring both analysis gates. It verifies all
four trusted frozen genuine-study artifacts, the immutable analysis configuration,
the complete 36 reserved confirmation sessions and disjoint
dialogue/topic/formulation families. It predicts at all four endpoints without
fitting/tuning, and reports the primary differences, simultaneous 12-contrast
interval family and fixed recognition summary. `predict_confirmation_once(...)`
also supports a primary-only report but consumes the same study-grid attempt.

Only six operational booleans are excluded from the scientific configuration hash:
`study_collection_allowed`, `confirmation_generation_allowed`,
`confirmation_evaluation_allowed`, `smoke_only`, `allow_scientific_analysis`, and
`allow_confirmation_analysis`. Model, seeds, generation parameters, feature
definitions, classifier choices and other configuration remain hash-bound. The
full execution configuration is also recorded separately. Freezing cannot
overwrite a scientific artifact. An exclusive `confirmation_grid.lock` in the
artifact directory reserves one shared attempt across all endpoints, and output
creation is exclusive. A failed locked attempt requires investigation, never
automatic retry. No confirmation dialogue is generated, loaded or evaluated by
the current fixture CLI.
