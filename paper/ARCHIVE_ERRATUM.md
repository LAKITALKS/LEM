# LEM I — archival erratum and reading order

**26 September 2026 · Lazaros Varvatis**

The existing Zenodo version [v2.1.0](https://doi.org/10.5281/zenodo.21268033)
and its GitHub PDF are **historical artifacts**. They have not been replaced or
silently edited. The currently corrected manuscript is the
[working PDF](lem_paper_correction_draft.pdf) and its
[LaTeX source](lem_paper_correction_draft.tex), published here on GitHub without
a new Zenodo deposit or DOI. Cite the DOI only when referring to the archival
version; identify the GitHub commit when citing the revised draft.

| Archived statement or record | What the evidence supports |
|---|---|
| Toy V1 distance to the fixed point, reported mean 0.7893 | The implementation omitted the denominator `alpha + beta` in the *measured target*. The corrected scaled mean is 0.7597. Trajectories, seeds, classifications, and cosine values are unchanged; see [matched reanalysis](CORRECTION_NOTE.md). |
| Precise noise transition at σ ≈ 0.25 / chance level at σ = 0.50 | The sampled grid has σ = 0.15 and 0.30, but not 0.25. Accuracy at 0.50 was 0.0200 versus a 0.0020 random baseline. |
| Toy V2 “central validation”, “reliably distinguishes”, and “zero error” | One deliberately constructed sink and one deliberately constructed limit cycle were compared at seed 1337. Maximum H₁ lifetimes were 0.0326 and 5.2649; their ratio, historically called “SNR”, was 161.6 **for that pair only**. The threshold 0.1 was evaluated on those same two examples, so it yields no estimate of a future error rate. Independent random projections, no matched-size noncycle controls, no held-out seeds, and no comparison to geometric spread prevent conclusions about topological advantage, user-specific signatures, or actual model attractors. |
| “Bibliography-corrected” archival v4 PDF | Its embedded references still contain the superseded Bricken author list and `Sch"afl`; the later `references.bib` repairs were not reflected in that PDF. The present corrected draft rebuilds and checks its own bibliography. The archived PDF remains unchanged. |
| v4 described as bibliography-only and content-identical to the published v3 | This was inaccurate. Section 5.3 of v4 expands the single sentence printed in the v3 PDF, adds Chia et al., and attributes to Gardinazzi et al. a comparison with linear probes that the cited study does not establish. The v4 changelog also misstates the v3 reference count (11, not 12). The corrected draft describes Gardinazzi's actual zigzag-persistence study and identifies subsequent Ko–Geiping work explicitly. |
| “Homotopies” guarantee persistence stability across model updates in the v1 framework | Homotopy of a model or latent space does not, by itself, control the metric distances of an activation point cloud or its persistence diagram. Cross-model correspondence, alignment, an explicit metric and measured distance bounds would be needed. The framework is now marked as a historical proposal with this correction. |

**A remaining bibliographic source discrepancy:** The
[Bricken et al. (2023) publisher page](https://transformer-circuits.pub/2023/monosemantic-features/index.html)
lists a different visible author line from the BibTeX that the same publisher
offers on that page: its offered BibTeX includes Zac Hatfield-Dodds and lists
Nick Turner. The corrected draft uses that publisher-supplied BibTeX rather
than inventing a reconciled list. A reader should consult the publisher's
record if exact credit across its two versions matters. The current
bibliography also includes all 26 Templeton et al. names and renders Schäfl
correctly; this does not retroactively repair the archival PDF.

Both archival PDFs remain byte-identical to the `main` snapshot
`94e17e6f7d470f82b869d9d3c0c904e832ee9666` before these corrections:

| PDF | SHA-256 |
|---|---|
| `lem_paper_final_v3.pdf` | `3909aa11f230bb82358b813c5d7459c3725d46bc4b3db5004f81ad05c19b4547` |
| `lem_paper_final_v4.pdf` | `2c864e516bf53d5cd45657e15b2db3669ec64e9df8a5a177ebbdae53a9f91848` |

The [original v4 changelog](CHANGELOG_v4.md) and
[original source](lem_paper_final_v4.tex) remain available for historical
accountability; their claims of content parity, complete bibliography correction,
and exclusive bibliographic changes are superseded **by this erratum**. Compiling
v4 now against the corrected shared bibliography would create a *new* PDF and
must not be mistaken for the historical archived artifact. The corrected draft
has its own version notice and PDF text layer; the v4 text layer also has
search/extraction problems that cannot be fixed retroactively on Zenodo here.

This correction does not add a new experiment or a confirmation result. The
[LEM-II-Light research note](../research/LEM_II_LIGHT_RESEARCH_NOTE.md) covers a
later, narrower real-model study and its limitations; it is not evidence for a
user-specific attractor. New synthetic-control and independent real-model tests
remain future work.
