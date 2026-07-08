# LEM I — Quellen-Rekonstruktion v3 → v4

**Datum:** 2026-07-08
**Anlass:** Externer Review-Befund zur `ripser`-Zitation in Abschnitt 4.2.1.

## Ausgangslage

`paper/lem_paper_final_v3.tex` im Repo war **nicht** die Quelle der
veröffentlichten PDF (`lem_paper_final_v3.pdf`, Zenodo DOI 10.5281/zenodo.19266201):

- Sie kompilierte nicht (`! Undefined control sequence`, Zeile 250).
- Zeile 250 war ein 25.680-Zeichen-Block mit literalen `\n`-Escapes,
  doppelt eingefügt, aus einer älteren Fassung.
- Sie enthielt eine tote `thebibliography`-Umgebung mit acht Quellen,
  die weder in der PDF vorkommen noch verifizierbar sind.
- Sie enthielt kein `\bibliography{references}` — `references.bib` war unverknüpft.

## Was v4 ist

`lem_paper_final_v4.tex` ist eine aus der veröffentlichten PDF rekonstruierte
LaTeX-Quelle. Inhaltliche Parität wurde geprüft (96,8 % Volltextähnlichkeit;
Restdifferenz = Ligatur-/Silbentrennungsartefakte der Textextraktion plus die
unten genannten Zitationskorrekturen). Alle 126 numerischen Werte in Fließtext
und Tabellen sind identisch, ohne Abweichung.

Vollständig enthalten: 8 Sections, 12 Subsections, 7 Subsubsections,
3 Tabellen, 3 Abbildungen, 5 nummerierte Gleichungen.

## Änderungen gegenüber v3 — ausschließlich bibliographisch

1. **Abschnitt 4.2.1, `ripser`.** War: `[7]` = Gardinazzi et al. (falsch).
   Ist: Bauer (2021), JACT 5(3):391–423, DOI 10.1007/s41468-021-00071-5,
   zusammen mit Tralie et al. (2018), JOSS 3(29):925 — der Python-Bindung,
   die `experiments/lem_simulations.py` per `from ripser import ripser` nutzt.
   Beide Zitationen entsprechen der Bitte der ripser.py-Maintainer.

2. **Abschnitt 5.3, `chia2025probing`.** War: „Chia and Pan".
   Ist: „Chia et al." — das Paper hat drei Autoren
   (Xin Wei Chia, Swee Liang Wong, Jonathan Pan).
   Der Eintrag fehlte zudem komplett in `references.bib`.

3. **`references.bib`, `wang2025attractorcycles`.** Die Autorenliste war frei
   erfunden („Wang, Xigui; Zhao, Tian; Sun, Zhangang; Gao, Yang"), der Titel
   abgeschnitten. Korrekt: Zhilin Wang, Yafu Li, Jianhao Yan, Yu Cheng, Yue Zhang,
   „Unveiling Attractor Cycles in Large Language Models: A Dynamical Systems View
   of Successive Paraphrasing", ACL 2025, S. 12740–12755. Venue, Seiten, DOI und
   arXiv-ID waren bereits korrekt. Der Fließtext („Wang et al.") bleibt unverändert
   und war inhaltlich zutreffend.

4. **`references.bib`, `gardinazzi2024persistent`.** Autorenreihenfolge auf die
   aktuelle arXiv-Fassung korrigiert (Gardinazzi, Viswanathan, Panerai, …
   statt der v1-Reihenfolge). Hinweis „Accepted as poster at ICML 2025" ergänzt.

Keine inhaltlichen, strukturellen oder theoretischen Änderungen.

## Nummerierung

5. **`references.bib`, `bricken2023monosemanticity`.** Vier erfundene Autorennamen
   („Leask, Matthew", „Lieberum, Tom", „Tseng, Cathy", „McDougall, Callum") gegen die
   reale Autorenliste ersetzt (Conerly, Turner, Anil, Denison, Askell, Lasenby, Wu,
   Kravec, Schiefer, Maxwell, Joseph, Hatfield-Dodds, Tamkin, Nguyen, McLean, Burke,
   Hume, Carter, Henighan, Olah). URL auf die kanonische Form gesetzt.

6. **`references.bib`, `templeton2024scaling`.** „Templeton, Adly and others" durch
   die vollständige Autorenliste ersetzt. Kein Fehler, aber bibliographisch dünn.

7. **`references.bib`, Key-Umbenennung.** `erdogan2025attributiongraphs` →
   `ameisen2025circuittracing`. Es gibt keinen Autor namens Erdogan. Autorenliste
   vervollständigt. `\cite`-Aufruf in der `.tex` nachgezogen.

## Verifikationsstand aller 14 Einträge

Alle Einträge wurden gegen arXiv, ACL Anthology, DOI-Resolver oder die offizielle
Publikationsseite geprüft. Stand nach v4: 14 von 14 verifiziert.

Gefundene Metadatenfehler insgesamt: 4 (ripser-Fehlverweis, Chia unvollständig,
Wang erfundene Autoren, Bricken erfundene Autoren) plus 1 unpräziser Eintrag
(Templeton) und 1 irreführender Key (erdogan).

## Nummerierung

Durch die drei neuen Einträge verschiebt sich die Referenznummerierung
(`plain`, alphabetisch): jetzt 14 statt 12 Einträge. Bauer = [3],
Chia et al. = [7], Tralie et al. = [13].

## Build

```
latexmk -pdf lem_paper_final_v4.tex
```

Finaler Durchlauf: 0 Fehler, 0 LaTeX-Warnungen, 0 undefined references,
0 undefined citations. 3 Overfull-hboxes (kosmetisch, auch in v3 vorhanden).
13 Seiten (v3: 12 — Differenz durch zwei zusätzliche Literatureinträge).

Benötigte Dateien im selben Verzeichnis:
`references.bib`, `toy_v1_scaled_results.png`, `toy_v1b_robustness.png`,
`toy_v2_moneyplot.png`.
