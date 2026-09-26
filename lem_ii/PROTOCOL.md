# LEM-II-Light: prespecified protocol, version 1.0.0

Author: Lazaros Varvatis. Date: 2026-09-07.

This is a **prespecified protocol**, not a publicly preregistered study. Its
executable specification is `config/study.json` (schema
`lem-ii-light.study.v1`). Freeze its SHA-256, code commit and the complete session
assignment before collecting study data. No study dialogue or scientific finding
is supplied by this implementation. Synthetic fixtures and any technical smoke
dialogues are explicitly ineligible for scientific inference.

The current authorization covers local implementation and a separately guarded
technical run costing at most **$5 in total**, including failed attempts. The
planned full study has a **$30 total additional-use ceiling**, of which the other
$25 is not released by this task. Study collection and confirmation generation
and analysis are disabled in the supplied configuration. A future authorized
execution must explicitly open the corresponding gates after reviewing measured
costs; merely constructing and hashing the 108-session assignment generates no
dialogues. There is no automatic next stage. No changes to LEM-I, archived
manuscripts, publication metadata, or ASCR resources are part of this protocol.

## Questions and scope of inference

The primary target is the three-class **interaction regime**, determined by the
user simulator's observable decision rule, not by measured geometry. The primary
endpoint is balanced accuracy (unweighted mean of the three class recalls) after
the user contribution of **turn 24**, before its assistant response. One turn is
one user contribution followed by one assistant response; a collected session
therefore contains exactly 24 of each, even though the final answer is not used
to predict its own pre-answer state.

The three paired primary comparisons are:

| Question | Baseline | Added block |
|---|---|---|
| Do activations improve prediction over a strong text representation? | `text` | `text_static`: current state, window mean/standard deviation and unordered state geometry |
| Do explicit time features improve over static summaries? | `text_static` | `text_static_geometry_time`: temporal features and delay-space geometry |
| Does topology add predictive value to that full common basis? | `text_static_geometry_time` | `text_static_geometry_time_topology`: delay-space persistence summaries |

The second comparison adds only explicitly temporal and delay-representation
features to an identical static and unordered-geometric base. This operational
feature comparison does not establish a causal temporal mechanism. The third comparison retains
**every** common feature, including text and geometry from the same delay
representation, and adds only the topology block. All paired methods use the same
sessions, information cutoff, classifier family and three-value tuning budget.

A positive finding supports predictive performance for these synthetic
interactions and this fixed model. It does not establish human identity,
personality, hidden cognition, or memory after a full context reset. Activations
are computed from the visible context; a predictive advantage is not evidence
of additional information in the information-theoretic sense. A null result is
valid. No attractor, cycle or convergence label is defined, and no attractor
claim is tested. Such a claim would require a separate intervention, comparator
and return-measurement module. Recognition of the 18 individual known profiles is
not implemented as an additional outcome in this version.

## Assignment, crossing and holdouts

There are 3 regimes × 6 variants = **18 profiles**, each crossed with all six
topic families: **108 sessions and 2,592 assistant responses** before controls or
retries. Each profile has one session per topic. All 18 profiles occur in every
phase, twice. Six topics are distinct practical domains, not paraphrases of one
underlying problem.

| Phase | Topic family | Facets | Formulation family |
|---|---|---|---|
| Development | Neighborhood vegetable garden | planting; access/shared work; material costs | `d_direct` |
| Development | Museum exhibition about early navigation | historical content; visitor experience; object preservation | `d_scenario` |
| Validation | Temporary storm shelter operation | space/occupancy; daily supplies; arrival/coordination | `v_request` |
| Validation | Historical newspaper digitization | image capture; search/description; storage/access | `v_consult` |
| Confirmation | Volunteer river water monitoring | sampling location/timing; measurement quality; records/reporting | `c_workshop` |
| Confirmation | Local food festival | vendors/menus; site operation; attendance/publicity | `c_brief` |

Each phase contains **36 dialogues**, with 12 per regime. Profiles and regimes
are crossed with topics. Topic, formulation family and phase change **jointly**:
the design cannot separate topic transfer from formulation transfer. Each
formulation family supplies three interchangeable sentence templates; their exact
English text, topic descriptions, facet requests and lexical scoring dictionaries
are fixed in `study.json`. Templates and topic assignments are identical across
regimes for corresponding variants. No phrase is selected by the regime name.

The estimand is generalization to the two held-out topic/formulation pairs within
the **same 18 synthetic profiles**. It is not generalization to unseen profiles,
new people or the population of all possible topics. All turns, prefixes and
overlapping windows inherit their dialogue's split. Do not split these objects
individually. Neither validation nor confirmation states fit PCA, scaling,
vocabularies or other transformations. Confirmation is never a tuning set.

`design.build_sessions` constructs deterministic `study-pNN-topic` assignments.
It validates unique IDs and profile/topic pairs, 24 turns, phase-disjoint topics
and formulations, the complete profile crossing and within-topic class balance.
Technical IDs begin with `smoke-` and are excluded from those 108 planned IDs.

## Rule-based user simulator

The visible system prompt is exactly:

> You are a helpful assistant. Help the user work through the practical task in the conversation. Give clear, specific answers and explain uncertainty when relevant.

Profile IDs, regime names, control rules, seeds and audit records are never
rendered in that prompt or passed as textual classifier inputs. The only visible
user content is an ordinary task request formed from a topic, chosen facet,
shared sentence template and shared follow-up request. The fixed follow-up
requests are either to explain practical details with a concrete example or to
apply details to a specific example and explain the next step. Both are available
to all regimes. Initial user prompts are byte-identical across the three regimes
for each matched variant/topic pair.

Every topic has three facets. The six variants within **each** regime are:

| Variant index | Initial/priority facet index | Sensitivity |
|---:|---:|---|
| 0 | 0 | lenient |
| 1 | 1 | lenient |
| 2 | 2 | lenient |
| 3 | 0 | strict |
| 4 | 1 | strict |
| 5 | 2 | strict |

Profiles `p01`–`p06` retain priority, `p07`–`p12` advance when addressed, and
`p13`–`p18` revise from the answer. The cyclic tie order starts at the priority
facet and proceeds modulo three. Each first turn uses the priority facet.

Only the preceding assistant response controls the next action. Its measured
features are character count; word count under `\b[\w']+\b`; the number of
digit groups matched by `\d+(?:[.,]\d+)?`; counts of `maybe`, `perhaps`,
`possibly`, `uncertain`, and `unclear`; and the number of **distinct exact,
case-folded word matches** to each facet's five fixed keywords. There is no
stemming, embedding similarity, semantic judge or second LLM.

| Threshold | Lenient | Strict |
|---|---:|---:|
| Distinct keywords for adequate coverage of the current facet | 1 | 2 |
| Minimum answer words | 20 | 35 |
| Maximum hedge occurrences | 3 | 1 |
| Minimum digit groups for a concrete answer | 1 | 2 |

“Adequate” means the keyword, minimum-length and maximum-hedge conditions all
hold for the previous selected facet. “Concrete” means the length, digit-group
and hedge conditions all hold. The policies are:

1. **`retain_priority`** always selects its priority facet. Adequate previous
   coverage changes the follow-up from clarification to application; otherwise
   it requests clarification. Thus previous answers affect visible user action
   even when facet selection remains fixed.
2. **`advance_when_addressed`** advances the focus by one facet modulo three
   when the answer adequately covers the previous focus; otherwise it retains
   that focus. Follow-up mode uses the same adequacy rule as all other regimes.
3. **`revise_from_answer`** selects the least-covered facet after a concrete
   answer, or the most-covered facet after a nonconcrete answer. Ties follow the
   fixed cyclic priority order. Follow-up mode again uses previous-focus adequacy.

This is answer-responsive in the literal operational sense: changing the previous
answer can change the next user message. It is a bounded lexical state machine,
not a model of a human. Exact keywords can miss synonyms and negation; digits need
not be useful quantities; hedging can be appropriate; length is only a proxy.
Those limitations are part of the fixed intervention, not reasons to retune the
simulator after observing a favorable or unfavorable result.

`UserSimulator(config, spec).next_message(previous_assistant)` returns only a
string. Turn 1 requires `None`; later turns require an assistant string. An empty
answer is logged as empty and scored as such, never silently replaced with a
successful answer. More than the session turn limit raises an error. User text
is limited to 100 words and fails instead of truncating. The separate `audit_log`
records response features, previous/selected focus, action, request mode, template
index, sensitivity variant, seeds, user lengths and empty-answer status. Collection
also records assistant lengths, token counts, finish status and aborts. These
records support descriptive checks for length, template and execution confounds;
they are not extra prediction features.

`examples/simulator_examples.json` contains **invented test responses**, paired
with actual simulator outputs for each policy. These are not generated research
dialogues or evidence of class separability. Templates, policies and features may
not be changed to manufacture a desired difference after a smoke run.

## Model, context and extraction

Use only `Qwen/Qwen2.5-3B-Instruct`, model and tokenizer revision
`aa8e72537993ba99e69dfaafa59ed015b17504d1`, with unchanged weights in BF16,
no quantization, no remote model code and no training. The adapter uses SDPA
attention, selected explicitly before data collection. Generation is greedy
(`do_sample=false`, `num_beams=1`, `repetition_penalty=1.0`), bounded to
128 new tokens per answer, with EOS token IDs `[151645, 151643]` and BOS/pad ID
`151643`. The separate extraction prefill has `use_cache=false`; generation has
`use_cache=true` only inside one assistant answer. No cache is returned for reuse
across turns or sessions. Dependency pins and
the verified chat template are recorded in the runtime manifest.

Within a session, retain the complete system/user/assistant history. Between
sessions reset messages and every session-level state; do not reuse caches,
hidden states or simulator history. The model weights may remain loaded.
The **8,192-token context cap includes a reservation for the next 128 output
tokens**. Any overrun fails explicitly; no silent truncation or rolling context
is permitted. A complete 24-turn technical dialogue is required to test growth.

At each turn, append the current user message and render the pinned model's
chat template with its assistant generation prefix. Measure the **last
non-padding token of that prefix before the current assistant answer**. This is
not described as the last user token. Verify the actual token IDs and generation
prefix boundaries against the pinned tokenizer; the manifest records the template
hash and extraction position. Use the attention mask to exclude padding; no
token pooling is performed. Save only the three selected vectors per turn.

Transformers `hidden_states[0]` is the embedding output. `hidden_states[k]` is
the output after transformer block `k`, with the final model normalization in
the final layer output. Save indices **0, 18 and 36**. Layer **36** is the sole
primary feature source; layer 18 is reserved for a separately labelled descriptive
sensitivity calculation and must not select the primary layer. Index 0 is an
embedding diagnostic. Raw stored vectors receive no normalization. For this
model each saved vector has 2,048 coordinates.

The text baseline receives exactly the visible messages available at that
measurement: the system prompt, prior user/assistant turns and current user
message, in order, without label metadata or the current/future assistant answer.
The text representation may omit template control tokens but must contain the
same visible content. Stored current assistant answers are not an input to their
own measurement. Full-precision saved extraction arrays, token positions,
attention-mask checks and the adapter's actual output shape provide the audit
trail; errors are statuses, never zero substitute vectors.

## Information cutoffs and feature construction

Evaluate the prespecified grid **12, 16, 20, 24**. Each endpoint uses a trailing
**12-state window**, so its delay representation always has **10 points**. This
separates the planned length comparison from increasing window size or point
count. The visible text context necessarily grows because it is the full context
seen by the model; therefore performance-by-turn is not a pure causal time effect.
No expanding-window result is part of the current analysis.

At each endpoint separately, fit PCA with **8 components**, full SVD and `whiten=false`, on
raw layer-36 development states observed up to that endpoint. Then fit
per-coordinate standardization on those development projected states. Apply the
frozen transform to validation and, later, confirmation. Standard deviations of
constant coordinates use the scaler's unit-scale convention. This is
**PCA first, projected-coordinate scaling second**, without raw-state L2
normalization in scientific analysis. Do not fit on later development turns
when evaluating an earlier endpoint.

The fixed feature blocks are:

* **Text:** word TF-IDF n-grams 1–2 and character TF-IDF n-grams 3–5, each limited
  to 12,000 vocabulary items, `min_df=1`, sublinear term frequency, lowercase
  conversion, no accent stripping, smooth IDF and separate L2 row normalization.
  Word tokenization is `(?u)\b\w\w+\b`; the character analyzer is `char`,
  including spaces. Terms are not binarized. The eight
  additional style features are character count, word count under `\b\w+\b`,
  number of split lines, mean word length, fraction of distinct lowercase words,
  question marks per character, exclamation marks per character and digits per
  character. Fit vocabulary/IDF and style scaling on development endpoint rows.
* **Static:** current projected state (8), trailing-window coordinate mean (8),
  coordinate population standard deviation (`ddof=0`, 8), and the four unordered
  state-geometry summaries below, for **28 columns** in the shared static block.
* **Unordered state geometry (included in static):** Euclidean pairwise distance
  mean, population standard deviation and maximum, plus radius of gyration of the
  projected state cloud. These are invariant under state reordering.
* **Time:** adjacent step-norm mean/standard deviation/maximum, adjacent step
  cosine mean/standard deviation, and lag-1, lag-2 and lag-3 displacement
  mean/standard deviation. A cosine involving a zero-length step is assigned 0;
  this explicit directional convention does not replace a missing extracted
  state. All reported feature standard deviations use `ddof=0`.
* **Delay geometry:** construct
  `z_i = [x_i, x_(i+1), x_(i+2)]`, lag 1 and embedding dimension 3, giving
  `12 - 1*(3-1) = 10` points in **24 dimensions**. Compute their pairwise
  distance mean/standard deviation/maximum, mean/maximum coordinate standard
  deviation and radius of gyration. This block remains present on both sides of
  the topology ablation.
* **Topology:** Euclidean Vietoris–Rips persistence on those exact 10 delay
  points, coefficient field 2, dimensions H0 and H1, no finite filtration cutoff.
  For each dimension, summarize finite bars with strictly positive lifetimes by
  count, total persistence, maximum
  persistence and persistence entropy using natural logarithms. Exclude essential
  infinite bars. An empty finite diagram has valid zero summaries; this must not
  be confused with a missing or too-short state sequence.

Dense feature blocks are scaled using development endpoint feature rows only;
the same learned common blocks are used in paired ablations. A sequence shorter
than 12 actual states is **not evaluable**. No interpolation, duplicated states
or all-zero substitute features manufacture a sufficient sequence.

Smoke-only feature exercise may run these geometry, time, delay and topology
functions on all 2,048 extracted coordinates after explicitly labelled per-vector
L2 normalization, because no development-fitted PCA exists in one technical
dialogue. This is a distinct **technical feature-path exercise**, not the
scientific PCA pipeline or a model comparison. The full scientific fitted path
is exercised separately on clearly marked synthetic fixtures.

## Fitting, uncertainty and length analysis

Use the same L2-regularized logistic regression family for all methods:
`solver=lbfgs`, `class_weight=balanced`, `max_iter=2000`. For each endpoint and
method, fit exactly **C ∈ {0.1, 1, 10}** on development, select the highest
validation balanced accuracy, and break exact ties toward the smallest C. Freeze
the selected development-fitted model and transformations; do **not** refit on
validation. No additional feature, vocabulary, layer, threshold or model search
is allowed. Convergence failure is reported and handled as an analysis failure,
not hidden by silently expanding the tuning budget.

Report paired differences in balanced accuracy, with the more complete method
minus its baseline, on the same dialogues. The three turn-24 differences form
one primary family. For each difference use **2,000 paired cluster bootstrap
replicates**, sampling six profiles with replacement within each regime and
retaining all sessions for each sampled profile. The same sampled profile
multiplicities are used for both methods. Use percentile intervals at **98⅓%**
coverage (`1 − 0.05/3`), the prespecified Bonferroni allocation for three primary
contrasts. These are small-sample bootstrap approximations, not guarantees of
nominal coverage. Do not bootstrap turns or treat windows as independent subjects.

This profile bootstrap is conditional on the fixed two topic/formulation pairs
in the evaluated phase (six across the complete design). It does not estimate
uncertainty over a population of themes; only two held-out themes materially
limit that claim. Development and validation reports are descriptive. The primary
inferential report requires the independently collected, still blocked
confirmation set. No confirmation performance is available in this task.

Earlier-point comparisons form a separate secondary family: all three paired
contrasts at all four grid points receive conservative simultaneous bootstrap
intervals with **99.583333%** coverage (`1 − 0.05/12`) per interval. Report the whole
curve, including turn 24 in that family, without selecting a favorable point.
These are Bonferroni bands for the 12 paired contrasts over the discrete grid,
not continuous-time bands. Raw method balanced-accuracy curves have no confidence
bands and remain descriptive. Development/validation comparisons are descriptive;
the corresponding confirmation contrast curve is prespecified and may be reported
simultaneously by the same utility only under future analysis authorization.

For each method, the earliest recognition summary is the first grid point whose
**point-estimate** balanced accuracy is at least **0.70**, with the immediately
following grid point also at least 0.70. Possible first points are 12, 16 or 20;
24 alone cannot qualify. If none qualifies, report **“not reached in the observed
range.”** This fixed discrete threshold summary does not assert statistical
certainty of a detection time and is never calibrated on confirmation data.

## Counterchecks and what they test

Label counterchecks permute regime labels at the **profile block**, keeping all
six sessions of each profile together and applying the same mapping in every
phase. A permutation reassigns the 18 profile labels while preserving six
profiles per class. Refit allowed development/validation analysis under the
permuted labels. Use **199** permutations with the prespecified permutation seed;
any permutation p-value uses the plus-one correction. A smaller count is allowed
only for a logged `diagnostic_only` fixture test, never scientific inference.
This checks target dependence under the assumed exchangeable synthetic-profile
null; individual-turn label shuffles are invalid. It does not establish external
validity or compensate for only two themes per phase.

For the temporal order control, apply a seeded permutation to the **11 preceding
states while keeping the current, 12th window state fixed**. Recompute all
features, **including a fresh delay embedding**, from the reordered sequence.
The complete static basis, including the current state, means/standard deviations
and unordered geometry, remains invariant. The production countercheck verifies
this invariance before comparing frozen-classifier predictions. Separately,
reverse a plain point cloud to check persistence invariance. Ordinary point-cloud persistence is order-invariant
and cannot by itself establish temporal sensitivity. Delay geometry/topology and
explicit temporal features can change after rebuilding, but need not change on
constant, symmetric or otherwise degenerate examples. Analytic fixtures verify
the invariants and at least one nondegenerate temporal change. These checks test
order dependence, not whether any order-dependent feature adds predictive value.

Leakage checks reject overlap of dialogue IDs, topic families and formulation
families across phases, verify all profiles and balanced regimes within every
topic, reject duplicate profile/topic sessions, and keep hidden metadata outside
feature input. Identical initial wording across matched regimes and shared
template selection guard against an unnecessary direct wording label. Topic and
formulation cannot serve as deterministic class labels because every such pair
contains all three classes equally. These checks cannot prove that a classifier
has learned a desired psychological mechanism; inspect recorded lengths and
policy-execution frequencies descriptively without retroactive optimization.

## Seeds, records, resumability and completion criteria

Fixed seeds are design **20260907**, PCA **1701**, classifier **1702**, bootstrap
**1703**, label permutation **1704**, time shuffle **1705**, synthetic fixtures
**1706**, and smoke **20260908**. A session seed is the first eight SHA-256 digest
bytes of `design_seed|session_id`, interpreted big-endian and reduced modulo
2^32. Template RNG seeds use
`design_seed|wording|topic_family|variant_index` by the same rule, deliberately
matched across the three regimes. Session seeds and actual template seeds are
both recorded. Full-SVD PCA is deterministic and does not consume the reserved
PCA seed. Python's process-dependent built-in `hash()` is not a seed source.

Every run has a unique ID and a manifest with code reference, configuration hash,
model/tokenizer IDs and revisions, chat-template hash, software versions, seeds,
generation parameters, selected layers and token position, hardware, and
technical/scientific eligibility. Store visible dialogues, labels/policy metadata,
state arrays, per-turn completion status and errors separately and cross-reference
their IDs. No credentials are recorded. Raw dialogues and large model/activation
files are excluded from Git; small technical summaries and hashes may be tracked.

Resume validates configuration, model/template/extraction identity, code binding,
session IDs, expected turns, array shape/dtype, finite values and integrity before
reuse. Do not duplicate completed turns, accept partial records as complete, or
combine different protocols. Explicitly record interrupted or failed sessions and
their consumed budget; retry is not a new budget allocation.

Controls operating on already saved states/text incur no new model responses;
the 199 label permutations are offline classifier fits. The planned primary study
requires one prefilling extraction plus one answer generation per turn, with
three vectors saved from the same extraction pass. Any additional session,
generation retry or extra extraction pass is counted and costed separately.
No layer sensitivity result requires new generation because indices 18 and 36
are both saved during the original extraction.

The technical gate requires one separately labelled **24-turn smoke-only
dialogue**, optionally a short second session to check isolation, under a shared
conservative $5 ledger covering GPU, CPU/RAM, startup/loading, idle time, storage,
failures and retries. Fixed prepared user messages are allowed for this gate and
must be labelled as such; they do not demonstrate execution of the adaptive study.
Parallelism is one and automatic retries are zero. The runtime documentation and
manifest give live-verified tariffs, reservation, time/token/session caps and
measured costs; an estimated watchdog is not a guarantee of cent-exact provider
billing. Existing Modal limits may not be raised or removed.

The smoke report records throughput, maximum context length, duration, peak GPU
memory, saved activation size, billed or estimated usage and every failure. Real
states must pass extraction integrity and all feature blocks, including topology.
No multiclass accuracy or topological advantage is required to accept a technical
run. A small locally initialized compatible model tests extraction mechanics but
does not substitute for that pinned-model check. If remote access or a credible
cost bound is absent, finish and test locally and explicitly report the remote
gate as outstanding.

After the full-length smoke, estimate all 108 dialogues plus distinct controls,
startup/storage and contingency using measured context growth, not an unchecked
linear projection from a few short turns. Separate measured cost from estimates
and preserve the $30 total/$5 technical ceilings. If the estimate is too high,
propose a concrete design/budget revision for later authorization; do not silently
remove text, topology or confirmation. Even a favorable estimate does not start
the main study in this task.
