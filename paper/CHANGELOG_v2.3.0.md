# v2.3.0 working draft — not yet published

Author and sole contributor: Lazaros Varvatis. Date: 8 November 2026.
The current published version remains v2.2.0, DOI 10.5281/zenodo.22973258.

- Corrected the README status: the completed private November 2026
  Ministral-3-14B role-swap study is under review; no private result was imported.
  Other private runs remain calibrations or instrument checks.
- Added the MIT code license (copyright 2026 Lazaros Varvatis); specified
  CC BY 4.0 for texts, figures and paper. Updated GitHub About to the author's
  research-program wording.
- Committed the original Toy V2 protocol as `c7d44d1` before computation.
  Added a deterministic CPU runner, verbatim source-provenanced numerical helpers,
  pinned dependencies and reproducibility tests. Original seed 1337 reproduces
  0.0326 and 5.2649; repeated three-seed results are bitwise identical.
- Added exactly moment-matched DISK/GAUSS and finite-sinusoid LINE controls;
  common/independent projections, the specified seeds and noise levels,
  H1, g1–g3 and supplementary second-half g1, frozen Youden signs/thresholds,
  paired-seed bootstrap intervals and five-seed runtime estimates.
- Preserved the original result package `ee90b9f` byte-for-byte (3200 class rows):
  R1 **H1 separates sink and cycle**, original R2 **partial (TEILWEISE)**.
- Committed Amendment 1 (`cf66e01`) after external review on 8 November 2026,
  after the original results had been computed, inspected and reported in chat,
  before supplemental computation. The review requested a joint geometric
  comparator, UNIFORM and a temporal-shuffling probe. This chronology is explicit;
  the amendment is not presented as preceding access to the original results.
- In the amended version, renamed DISK to GAUSS without changing construction;
  added exactly moment-matched UNIFORM and same-point-set CYCLE_SHUF.
  Added calibration-only standardization and L2 logistic regression (C=1.0 fixed)
  for G and G+H1; whole-seed paired Delta-AUC intervals are evaluated separately
  at every noise level and projection mode, with no refitting or tuning.
- Wrote supplemental outputs only to the new `amendment_1/` subdirectory:
  per-seed data, coefficients/scalers, evaluation predictions, shuffle diagnostics,
  bootstrap summaries, both versions' categories, original SHA-256 hashes,
  environment, runtime and a new noise plot. Original output hashes are unchanged.
- Amended main category: **incremental value (ZUSATZNUTZEN)** overall; **incremental value** for GAUSS,
  **clear advantage (EINDEUTIGER VORTEIL)** for UNIFORM. Against GAUSS, H1 AUC is 0.9942,
  G is 0.8276, G+H1 is 0.9933, Delta AUC is 0.1657 [0.1087, 0.2347].
  LINE remains a plausibility probe and does not enter categorization.
- All 800 shuffle checks pass 1e-9; H0/H1 diagram differences are zero.
  Moment residuals are at rounding level. Reported the realized lag-1 changes
  after sample-dependent affine correction and the distinction between initial
  independent control draws and exactly matched finite samples.
- Updated the working-draft notice, abstract, contributions, methods, both versions'
  primary AUC/accuracy tables, noise figures, limits and conclusion. The result
  replaces the open-question sentence only for this toy and these fixed predictors.
  Point-cloud H1 measures loop geometry, not temporal return; no LLM/user result
  or universal advantage over geometric methods is inferred.
- Added reproducible table formatting and independent exported-prediction/scaler/
  bootstrap audits; updated both README files and reproduction commands. Rebuilt
  the working-draft PDF with Tectonic, checking its text layer and page layout.
  Used OpenType text fonts without common ligatures under XeTeX so that
  identifiability and Vietoris appear correctly in pdftotext output.
- Kept the original simulation, all V1 files, final v3/v4 materials, `paper/archive/`,
  published v2.2.0 and Zenodo records unchanged. No merge, tag, release or Zenodo
  publication is part of this branch.

## Editorial follow-up — 9 November 2026

- Replaced category labels in the paper and both README files with English names,
  retaining each original protocol identifier in parentheses at its first occurrence;
  translated R1 to "H1 separates sink and cycle".
- Introduced both synthetic checks before the abstract's V2 result and limited
  that result to two sentences with the requested three-decimal values and the
  original rule's partial outcome. Detailed V2 numbers remain in Section 4.3.
- Added the dated reviewer-blindness clarification immediately below Amendment 1
  and made the same chronology explicit in Section 4.3.
- Rebuilt the working-draft PDF with Tectonic and verified its text layer and
  changed pages. Numerical outputs, fitted models and decision rules are unchanged.
