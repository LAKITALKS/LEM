# Corrected working draft: validation on 26 September 2026

**Author:** Lazaros Varvatis · **Scope:** Repository documentation, source and PDF
corrections. This validation snapshot predates the separate
[Zenodo v2.2.0 publication](https://doi.org/10.5281/zenodo.22973258). It is not a newly executed LEM study.

- `latexmk -pdf -interaction=nonstopmode -halt-on-error lem_paper_correction_draft.tex`
  completed. The final PDF has **15 pages**; the final LaTeX log has no errors,
  undefined citations/references, warnings or overfull boxes.
- `pdftotext -layout` confirmed the title/date, the V2 single-pair limitation,
  the Gardinazzi wording, Ko–Geiping entry, Templeton's full 26-name list and
  the correctly rendered **Schäfl**. Pages 1, 9 and 14 were visually inspected.
- `python3 experiments/verify_fixed_point_regression.py --output-dir /tmp/lem-v1-regression-20260926`
  passed: green run 0 failed assertions; mathematical red checks produced the
  expected 5 old-target and 3 archived-production assertion failures.
- The complete documented CPU command
  `python3 experiments/reproduce_fixed_point_correction.py --output-dir /tmp/lem-v1-full-20260926`
  **passed**. It ran original and corrected Pilot, all ten Scaled seeds and the
  eight V1b noise levels in the same environment; unaffected metrics and RNG
  states were exactly equal. All **seven** numerical corrected result files,
  including RNG-state hashes, were byte-identical to the checked-in V1 result
  package. The comparison checks V2's entire scientific AST with an explicit
  whitelist for display and output-directory setup; it never executes Toy V2.
  It verifies the baseline hashes for both archival PDFs, all original paper
  figures, and the legacy v3 source. Bibliography, v4 source and changelog
  are deliberately editable editorial records and are not misrepresented as
  unchanged archive assets.
- `python3 -m py_compile experiments/lem_simulations.py experiments/run_all.py`
  passed; importing `lem_simulations.py` in an empty temporary directory did
  not create an `assets/` folder, and its fixed-point scalar fixture returned
  less than `1e-12`. The historical Toy V2 figure and numbers were **not rerun**.
- Original archival PDFs were not rebuilt or replaced. Their SHA-256 values are
  documented in [ARCHIVE_ERRATUM.md](ARCHIVE_ERRATUM.md) and can be checked with
  `sha256sum paper/lem_paper_final_v3.pdf paper/lem_paper_final_v4.pdf`.

New GitHub working PDF SHA-256:
`e8217766fc311fef77f544ad193e72f89aa3ee7349130e8d537bb5c7ef9efa5b`.
The archived v4 PDF is still bibliographically incorrect. The separate v2.2.0
Zenodo publication now provides the corrected paper with updated publication
notices; its file and SHA-256 are documented in [ARCHIVE_ERRATUM.md](ARCHIVE_ERRATUM.md).
The GitHub draft and the hash above describe the pre-publication artifact.
No statistical confirmation of user-specific signatures or attractor dynamics
follows from these checks or the publication.
