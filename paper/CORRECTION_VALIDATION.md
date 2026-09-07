# Correction validation — 7 September 2026

The correction is complete as an unpublished working draft. Numerical execution,
source integrity and rendering were checked locally; no publication step is implied.

| Check | Result |
|---|---|
| Original / corrected V1 | Pilot, ten Scaled seeds and all eight original V1b sigma levels completed in the same environment |
| Non-distance results | Every reported NN/baseline/cosine value exactly equal; RNG-state hashes equal after each experiment |
| Method preservation | Nested simulation, signature, normalization and classification functions unchanged; parameters and reseeding unchanged |
| Regression green | 5/5 passed, including all three production experiment bodies |
| Mathematical red | 5 assertion failures with only the old target restored; 3 assertion failures in the actual archived production bodies |
| Toy V2 | Entire function and standalone runner unchanged; not rerun |
| Archive | Every pre-existing paper source, PDF, figure, bibliography and note matches baseline SHA-256 |
| Manuscript | 14 pages compiled with Tectonic 0.16.9; 0 LaTeX errors, undefined references/citations or overfull boxes |
| Visual inspection | All 14 final rendered pages inspected; tables, figures, equations, references, margins and version/DOI notice are legible, without clipping or overlap |
| Independent diff review | No actionable findings; results and integrity checks independently confirmed |

Build: `cd paper && tectonic --keep-logs lem_paper_correction_draft.tex`.
Render: `pdftoppm -r 105 -png paper/lem_paper_correction_draft.pdf tmp/pdf-review/corrected`
(from the repository root). The backend reports that an ICC 4.3 color profile was
not embedded in PDF 1.5; rendered figures were visually checked. Original images
were preserved. This metadata warning does not affect the numerical results.

The [machine-readable validation](../experiments/results/fixed_point_correction/manuscript_validation.json)
records the exact PDF/source/table/figure SHA-256 values, inspected pages and tools.
The [build log](../experiments/results/fixed_point_correction/pdf_build.log),
[regression logs](../experiments/results/fixed_point_correction/tests/),
[before/after CSV](../experiments/results/fixed_point_correction/comparison.csv) and
[source/experiment manifest](../experiments/results/fixed_point_correction/manifest.json)
provide the supporting evidence.

No remaining implementation, numerical or manuscript-review blocker was found.
