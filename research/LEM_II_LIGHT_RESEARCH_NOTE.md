# LEM-II-Light: research history and scope of the findings

**Author:** Lazaros Varvatis

**Study completed:** 8 September 2026

**Research note:** 16 September 2026

## What was tested

LEM's broader question is whether repeated user–LLM interaction produces reproducible, user-related representation signatures and, under suitable tests, attractor-like dynamics. **LEM-II-Light tested a narrower question:** whether activation, temporal and topological features improve classification of three rule-defined interaction regimes over their respective baselines.

The [prespecified protocol](https://github.com/LAKITALKS/LEM/blob/3817eda79de44c0135d668a9b71d20122b9fd586/lem_ii/PROTOCOL.md) explicitly excluded an attractor claim and did not implement individual-profile recognition. This distinction was part of the original design, not a reinterpretation introduced after the outcome. The protocol was **not publicly preregistered**.

The study used unchanged BF16 `Qwen/Qwen2.5-3B-Instruct` weights and a rule-based, answer-responsive user simulator: 18 synthetic profiles, six sessions per profile, 24 user/assistant rounds per session. Development, validation and confirmation each contained 36 dialogues. All 18 profiles occurred in every phase; topics and formulation families were separated between phases.

## What was observed

The completed collection contains **108 dialogues, 2,592 rounds and 7,776 saved activation vectors**. Those counts describe collected material, not independent participants. The following results concern the 36 confirmation dialogues at the primary endpoint, round 24:

| Method | Balanced accuracy |
|---|---:|
| Text | 41.7% |
| Text + static activation/geometry | 47.2% |
| Same static basis + time/delay features | 47.2% |
| Same full basis + topology | 52.8% |

| Prespecified nested increment | Difference (percentage points) | Bonferroni-adjusted 98.33% bootstrap interval |
|---|---:|---:|
| Static activation/geometry over text | +5.6 | [-16.7, +25.0] |
| Time/delay features over the static basis | 0.0 | [0.0, 0.0] |
| Topology over the full non-topological basis | +5.6 | [0.0, +13.9] |

**None of these intervals establishes positive added value.** The zero-width time interval arises from identical predictions at this endpoint; it is not evidence of general equivalence. The broad static-feature interval leaves substantial uncertainty. With 12 dialogues per class, a net improvement of two correct classifications is approximately 5.6 percentage points. The prespecified recognition threshold was not reached.

The 199 label permutations in the published aggregate report are **validation diagnostics**, not confirmation p-values. They must not be treated as a confirmation null distribution. These results establish neither that every method is at chance nor that all possible activation methods lack signal.

## What the study contributes, and its limits

The execution record documents real-model state collection, complete dialogue storage, frozen transformations/classifiers before confirmation, a single confirmation analysis and explicit counterchecks. This is reusable experimental infrastructure. Its procedural integrity does not establish the broader LEM hypothesis.

Interpretation is limited by the following design choices:

- The same known synthetic profiles recur in all phases, and only two topic/formulation families were reserved for confirmation. Profile-bundle bootstrap intervals do not measure uncertainty over a general population of topics or people.
- **2,591 of 2,592 answers reached the 128-token cap.** This can interrupt answers and influence the simulator's next action. The keyword-based simulator is a narrow behavioral construction, not a validated human model.
- The primary analysis used a particular readout, layer and reduced representation. Every prediction arm included the text basis. Thus, the outcome concerns these incremental feature comparisons, not a standalone test of all information readable from activations.
- No controlled perturbation-and-return experiment or individual-profile recognition endpoint was included. Human identity, post-reset memory and user-specific attractors remain untested.

The finding is retained without rerunning confirmation to seek a favorable outcome. It neither confirms nor directly refutes the broader attractor hypothesis.

## Why the next experiment changes

The planned character study brings the operational questions closer to the original motivation:

1. **A — Recognition:** distinguish known synthetic characters across new sessions and topics, comparing structured text, static activation summaries and trajectories, including the direct topology ablation.
2. **B — Persistence:** measure whether character-related differences remain when subsequent user inputs are shared and neutral. Earlier history remains in context, so this tests context-carried persistence.
3. **C — Perturbation response:** compare perturbed and control continuations of the same history. Test recovery toward the appropriate control while checking common drift, fading history and loss of differences between characters.

Longer dialogue alone does not establish an attractor. Role quality, answer completeness, measurement sensitivity and actual costs require separate calibration; sufficient statistical precision must not be inferred from software tests. This next study is planned and paused, not a result reported here. The broader signature and cross-model research questions remain open.

## Evidence and publication boundary

This note summarizes the existing public [study report](https://github.com/LAKITALKS/LEM/blob/3817eda79de44c0135d668a9b71d20122b9fd586/lem_ii/RESULTS_2026-09-08.md) and [aggregate JSON](https://github.com/LAKITALKS/LEM/blob/3817eda79de44c0135d668a9b71d20122b9fd586/lem_ii/results/2026-09-08/summary.json). Scientific execution used [commit 358e76a0](https://github.com/LAKITALKS/LEM/commit/358e76a0b69fa4fa8ff10af2feb09ecb91bd99c1); the linked documentation snapshot is `3817eda79de44c0135d668a9b71d20122b9fd586`. These pinned links preserve the distinction between executed code and later reporting.

The public evidence remains in [Draft PR #2](https://github.com/LAKITALKS/LEM/pull/2), which this update does not merge. Full raw dialogues, activation arrays and frozen fitted artifacts are not included in the public repository; the aggregate evidence alone does not permit complete independent reproduction. This documentation update adds no new analysis or data-access claim and exposes no private operational records.

This is a **GitHub documentation update only**. Existing LEM-I archives and the reported Light results are preserved. No new experiment, tag, release, Zenodo deposit or DOI is created; the existing archival DOI does not identify this note or the Light study.
