# The Lazaros–Eudora Method (LEM)

**Author:** Lazaros Varvatis · **Status:** research program and synthetic
pipeline checks; LEM-II-Light pilot completed with a narrow null result.
The name “Eudora” refers to AI collaboration; it does not name a second
scientific author. AI tools assisted with drafting, code, and review;
Lazaros Varvatis is responsible for the claims and released materials.

> **Read this correction first:** [LEM I archival erratum](paper/ARCHIVE_ERRATUM.md).
> The existing Zenodo [v2.1.0 archival edition](https://doi.org/10.5281/zenodo.21268033)
> contains known scientific and bibliographic errors. Its PDF remains an unchanged
> historical artifact. Read and cite the [corrected edition v2.2.0](https://doi.org/10.5281/zenodo.22973258)
> ([current PDF](https://zenodo.org/records/22973258/files/LEM-I-v2.2.0-corrected.pdf?download=1)).
> Its [source commit](https://github.com/LAKITALKS/LEM/commit/854849765c8ee01356e3ea520de5fe9ed2169eb2)
> and deposited correction note document the publication build.

## Research question and evidence

LEM asks whether repeated interaction with a person produces reproducible
patterns in a language model's internal representations, across fresh sessions,
topics and controlled perturbations. A frozen model's activations depend on its
current input and available history. This hypothesis does **not** imply that a
person changes the model's weights during ordinary inference or that an old
conversation survives an empty-context reset.

| Proposed measurement | Current status |
|---|---|
| Repeatable user region, single-vector and cluster signatures | Hypotheses; no real-user confirmation. |
| User-specific trajectory shapes across independent sessions | Untested as an identity endpoint in the completed Light study. |
| User-specific attractor with return after perturbation | Untested; a cluster or long-lived loop alone does not establish an attractor. |
| Cross-model correspondence and controlled reactivation | Future work; representation spaces require explicit alignment and causal intervention. |
| Cooperation/affinity correlated with a signature | Separate exploratory hypothesis, not a claim about a model's feelings. |

Two **synthetic pipeline checks** appear in LEM I:

| Example | Observation | Boundary |
|---|---|---|
| Toy V1: 500 randomly separated synthetic user vectors in 512 dimensions | Nearest-neighbor accuracy 1.000 across ten seeds at σ = 0.15; corrected mean distance to the prescribed fixed point 0.7597. | The user signals were built into the simulator; no LLM users or identity effects are demonstrated. Accuracy at σ = 0.30 was 0.2300, at σ = 0.50 it was 0.0200 versus random 0.0020. |
| Toy V2: pre-specified multi-seed check with matched controls; Amendment 1 | At σ = 0.05 with common projection and evaluation seeds 100–199: against GAUSS, AUC(H₁) = 0.9942, AUC(G) = 0.8276, AUC(G+H₁) = 0.9933; paired ΔAUC = 0.1657 [0.1087, 0.2347]. Amended overall category: **ZUSATZNUTZEN** (GAUSS: ZUSATZNUTZEN; UNIFORM: EINDEUTIGER VORTEIL). | Only these synthetic generators and fixed predictors. Amendment 1 followed the already computed original analysis (R2: TEILWEISE), whose outputs remain unchanged. LINE is a plausibility probe; shuffled CYCLE has identical H₁ diagrams. Point-cloud H₁ measures loop geometry, not temporal return, LLM behavior or user identity. |

The later [LEM-II-Light research note](research/LEM_II_LIGHT_RESEARCH_NOTE.md)
documents a limited study with **108 synthetic dialogues and Qwen2.5-3B-Instruct
activations**. At the prespecified confirmation turn, the four classifiers had
41.7%, 47.2%, 47.2% and 52.8% balanced accuracy; no nested increment had an
adjusted interval strictly above zero. This tested **rule-defined regime
classification**, not user identity or attractors. All classifier arms included
text, the same 18 profiles appeared in all phases, and nearly every answer hit a
128-token cap. The available public aggregates do not enable a complete independent
reconstruction of the dialogues or activation arrays. The protocol and aggregate
evidence remain in [Draft PR #2](https://github.com/LAKITALKS/LEM/pull/2).

A further private study (a role-swap design with Ministral-3-14B) was completed in November 2026; its result is under review and not yet published here. Other runs in private planning are technical calibrations or instrument checks.

## Files and reproduction

- [Corrected manuscript and archive](paper/) · [erratum](paper/ARCHIVE_ERRATUM.md)
  · [V1 numerical correction and provenance](paper/CORRECTION_NOTE.md).
- [Toy simulation code and instructions](experiments/README.md) ·
  [checked V1 result package](experiments/results/fixed_point_correction/).
- [Historical v1 topological framework](LEM_v1.0_Topological_Framework.md),
  with editorial corrections; the conceptual drawings in [assets/](assets/) are
  illustrations, not evidence.
- [LEM-II-Light research note](research/LEM_II_LIGHT_RESEARCH_NOTE.md) and the
  linked public evidence in Draft PR #2.

From the repository root, with the documented Python environment:

```bash
python3 -m venv .venv-correction
.venv-correction/bin/python -m pip install -r experiments/requirements-v1-correction.lock
.venv-correction/bin/python experiments/reproduce_fixed_point_correction.py --output-dir /tmp/lem-v1-comparison
.venv-correction/bin/python experiments/verify_fixed_point_regression.py --output-dir /tmp/lem-v1-regression
```

The [experiment instructions](experiments/README.md) explain the reproduction
scope and PDF build. The corrected tables and V1 figures derive from the
checked-in numerical result package. The archived Toy V2 figure is an
illustration of its original single-pair result. The separate
[Toy V2 protocol and Amendment 1](experiments/toy_v2_robustness/PROTOCOL.md),
[original outputs](experiments/results/toy_v2_robustness/) and
[amended main tables](experiments/results/toy_v2_robustness/amendment_1/main_tables.md)
document the new multi-seed check.

The [v2.3.0 working draft](paper/lem_paper_correction_draft.pdf) is **not yet
published**. It reports the amended main analysis and the original committed
analysis separately; the current published version remains v2.2.0.

## Publication, credit, and reuse

The corrected paper is published as [v2.2.0](https://doi.org/10.5281/zenodo.22973258)
with version DOI **10.5281/zenodo.22973258**. The prior DOI
[10.5281/zenodo.21268033](https://doi.org/10.5281/zenodo.21268033) still identifies
the historical **v2.1.0** edition. The [Concept DOI](https://doi.org/10.5281/zenodo.16541978)
represents the version family; cite the version DOI for the exact corrected paper.
Neither DOI identifies LEM-II-Light. No historical PDF was replaced.

The publication PDF is `LEM-I-v2.2.0-corrected.pdf`, SHA-256
`65f6334315784570d76f6c6c2ee5068ba6a6067b11a4f789d2a0a992de0d7c61`.
The GitHub working-draft PDF remains a provenance artifact; the Zenodo PDF
updates its publication notices and title metadata only. The deposited
[correction note](https://zenodo.org/records/22973258/files/LEM-I-v2.2.0-CORRECTION-NOTE.txt?download=1)
includes the exact source-status diff and build instructions. That published correction added no new experiment
or confirmation result. No GitHub release was created; automatic
GitHub–Zenodo synchronization remains disabled.

Code: MIT (see LICENSE). Texts, figures and the paper: CC BY 4.0. For research questions and collaboration, use
[GitHub Issues](https://github.com/LAKITALKS/LEM/issues).
