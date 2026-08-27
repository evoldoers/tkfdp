# neurips-paper2

`neurips_workshop_paper2.tex` — "Variational inference in coupled models of amino
acid substitution", formatted for a **NeurIPS 2026 workshop** submission.

---

Note: after generating this file, I removed lumpable/half-lumpable models and the
entire section about TKF-DP. Some notes might not be relevant anymore.

## ATTENTION — six things need your decision before submitting

**1. `\workshoptitle{WORKSHOP TITLE}` is a placeholder.**
`neurips_workshop_paper2.tex`, near the top. The style file requires this for workshop
tracks; it prints in the first-page footer of the camera-ready copy. Replace it
with the workshop name.

**2. Content is 9 pages; that is probably over the limit.**
The template states nine pages for the main track, and workshop limits are
usually shorter still — often four or eight. Acknowledgments, references and
the checklist do not count, so the 19-page total is not the number that
matters; the References heading landing on page 9 is. **Reformatting cannot
recover this — cutting content is the remaining work.** `./build.sh --check`
reports the current content page count on every run.

**3. All 16 checklist answers are still `\answerTODO{}`,** rendered in red in
`checklist.tex`. `checklist.tex` says papers omitting the checklist are desk
rejected, so it is included; the answers are yours to write. Replace each
`\answerTODO{}` with `\answerYes{}`, `\answerNo{}` or `\answerNA{}` plus a
one-or-two sentence justification.

**4. One thing in the typeset submission identifies the authors,** which
matters because the submission is double-blind: the `https://tkfdp.net/` link
at the end of the abstract. I left it alone rather than editing your content.
The second occurrence, in the acknowledgments, is not typeset — the style
file's `ack` environment hides that whole section under `[dblblindworkshop]`,
and I confirmed it is absent from the submission PDF.

The self-citation `LargeHolmes2026` needs no action: it is a public bioRxiv
preprint and both citations of it are already in the third person, which is
what the template asks for.

**5. Section 5 ends "The algorithm, the phylogenetic fixed-bridge Potts ELBO,
is given in the supplementary appendix."** That appendix is
`../tkf-dp/psb-paper/paper2-coupled-appendices.tex`, which the PSB build does
not include either — the PSB `build.sh` lists it as supporting material and
builds only `paper2-coupled.tex`. So the sentence currently points at nothing a
reviewer can open. Either drop the sentence, or add the appendix as
supplementary material. I did not port that file, since porting
`paper2-coupled.tex` was the request; note that it uses ws-procs11x85's
`\appendix{...}` (a macro taking a title argument), which `article` does not
have, so it needs its own conversion.

**6. Track is set to submission, not camera-ready.**
`\usepackage[dblblindworkshop]{neurips_2026}` gives the anonymous version with
review line numbers and the acknowledgments hidden. For the accepted version,
change that one line to `\usepackage[dblblindworkshop, final]{neurips_2026}`.
Both paths were built and checked; the camera-ready build is also free of
overfull boxes, and its content runs to 10 pages because the author block and
the acknowledgments reappear. If the target workshop is single-blind instead,
use `sglblindworkshop`.

### One placement artifact, which needs no action now

Figure 4 (`holmes_msa_triangle_PF00053_K8.pdf`) floats onto page 10, after the
References heading. There is simply more float material in Section 6 than fits
before the text ends: Table 4 takes the last top-of-page slot on page 9. I
tried four alternatives — declaring the figure earlier, `[!t]`, and shrinking
it to 0.62 and 0.55 `\textwidth` — and each either left the figure where it is
or pushed Table 4 or the References heading onto page 10 instead. It resolves
by itself once the content is cut to the workshop limit (item 2), so the figure
is left at the width the PSB version used.

---

## Build

```
./build.sh            # pdflatex -> bibtex -> pdflatex x2
./build.sh --check    # + compliance report; read this before submitting
./build.sh --open     # + open the PDF
./build.sh --clean
```

Requires `pdflatex` and `bibtex`. `--check` additionally uses `pdfinfo`,
`pdffonts` or `mutool` if any are installed.

Verified state of the current build: **19 pages** total, References beginning
on page 9, checklist pages 13--19. Zero overfull boxes, no undefined references
or citations, no LaTeX warnings, US Letter (612 × 792 pt), 23 embedded Type 1
fonts and no Type 3 fonts. The three underfull `\hbox` messages are in the
Figure 2 caption and are cosmetic — loose interword spacing on three lines, no
text missing or protruding.

## This directory is self-contained

Nothing outside it is needed to build. No `BIBINPUTS`, no `TEXINPUTS`, no
sibling directories. The `.bib` file and all three figures are local copies, so
the directory can be zipped and uploaded as-is. It was verified by copying it
to an unrelated location with no other project directories reachable, unsetting
`BIBINPUTS` and `TEXINPUTS`, and building there; the page count and the
compliance report matched, and the log contains no path outside the directory.

| File | Notes |
|---|---|
| `neurips_workshop_paper2.tex` | the paper |
| `neurips_2026.sty` | official template file, **byte-identical, do not edit** |
| `checklist.tex` | official template file, byte-identical; answers are `\answerTODO{}` |
| `refs.bib` | 156 entries, 34 of them cited here; the sole bibliography |
| `figures/mixture_components_K8.pdf` | Figure 2 |
| `figures/elbo_vs_expm.pdf` | Figure 3 |
| `figures/holmes_msa_triangle_PF00053_K8.pdf` | Figure 4 |
| `build.sh` | build driver and compliance check |
| `port-from-psb.py` | regenerates the `.tex` from the PSB source; see below |

Figure 1 is TikZ, drawn inline in the `.tex`, so it has no file here.

`build.sh --check` verifies `neurips_2026.sty` and `checklist.tex` against
SHA-256 values recorded in the script itself, so the check stays local.
Tweaking the style files is grounds for desk rejection.

`refs.bib` is `../tkf-dp/math-paper/refs.bib` with the 45 entries of
`../tkf-dp/psb-paper/paper2-refs.bib` appended; the two files share no keys, so
nothing was renamed or dropped. It carries 122 entries this paper does not
cite. That is harmless — BibTeX emits only what is cited — but you may prefer
to prune it before uploading.

## Provenance

Ported from `../tkf-dp/psb-paper/paper2-coupled.tex` (World Scientific
`ws-procs11x85`, for PSB). The copies here were taken from:

| Copied here | Taken from |
|---|---|
| `neurips_2026.sty`, `checklist.tex` | `../Formatting_Instructions_For_NeurIPS_2026/` |
| `refs.bib` | `../tkf-dp/math-paper/refs.bib` + `../tkf-dp/psb-paper/paper2-refs.bib` |
| `figures/mixture_components_K8.pdf`, `figures/elbo_vs_expm.pdf` | `../tkf-dp/psb-paper/figures/` |
| `figures/holmes_msa_triangle_PF00053_K8.pdf` | `../tkf-dp/math-paper/figures/` |

Nothing in `../tkf-dp/` or `../Formatting_Instructions_For_NeurIPS_2026/` was
modified. Those originals are no longer consulted at build time. If the PSB
version changes and you want to re-port, run `python3 port-from-psb.py`, which
reads the PSB source and rewrites `neurips_workshop_paper2.tex`. Every edit it
makes asserts its expected match count and the script aborts on any mismatch,
so an upstream change fails loudly instead of producing a half-converted file.
**It overwrites `neurips_workshop_paper2.tex`**, so do not run it after editing
the paper here.

## What changed from the PSB version

The body text, mathematics, tables and figures are the same content. Only the
class, title block, float markup and bibliography changed.

**Class and preamble.** `ws-procs11x85` → `article` +
`\usepackage[dblblindworkshop]{neurips_2026}`. The packages the template loads
are kept in template order, followed by this paper's own additions (`amsmath`,
`amssymb`, `bm`, `graphicx`, `tikz`, `cleveref`). `ws-procs-thm` and `placeins`
were dropped — the former is the PSB class's theorem package, and this paper
never used `\FloatBarrier`. The `defn`/`rem` theorem environments and their
`\crefname` lines are kept, declared with `\newtheorem`, though the paper
currently uses neither. `cleveref` must load after `hyperref`.
`\raggedbottom` was dropped because the style file sets `\flushbottom` and
overriding it changes the template's vertical layout. `\graphicspath` now
points only at `./figures/`, since the second PSB entry
(`../math-paper/figures/`) is outside this directory.

**Numeric citations.** `\PassOptionsToPackage{numbers,compress}{natbib}` before
loading the style file. The bibliography is numeric; the template's natbib
default is author-year and the build fails outright without this.

**Title block.** `\address` and `\copyrightinfo` do not exist in the NeurIPS
template. The address moved into `\author`, in the template's `\And` form, with
the author order of the PSB version kept; the PSB copyright footnote was
removed, since the style file prints its own footer. The template has no
keywords element, so the keyword list is commented out in the source rather
than deleted.

**Acknowledgments.** `\section*{Acknowledgments}` → `\begin{ack}...\end{ack}`,
the environment the style file provides, which hides the section in the
anonymised submission and titles it "Acknowledgments and Disclosure of Funding"
in the camera-ready copy. Note the template also requires a competing-interests
declaration there; the current text declares funding only.

**Bibliography.** `ws-procs11x85.bst` → `plainnat`, reading one local
`refs.bib` in place of the two paths the PSB source used. All 34 cited keys
resolve.

**Tables.** All four tables had the same ws-procs11x85 shape — `\tbl{caption}`
followed by a `{\footnotesize \begin{minipage}{\textwidth}...}` wrapper with
`\label` at the end. They are now plain `table` + `booktabs`, with `\caption`
above the tabular, `\label` immediately after it, and `\centering`, as the
template requires; the `minipage` wrapper is gone. The rules were already
`booktabs` in the PSB source and no table had vertical rules or `\hline`, so
nothing there needed changing. One regex converts all four and the script
aborts unless exactly four match.

**Figure 1.** Was a 0.58/0.40 `minipage` pair with the caption beside the TikZ
diagram; now a single centred `tikzpicture` with the caption below, per the
template. Figures 2–4 were already `\includegraphics` with the caption below
and needed no structural change.

**Width, 6.6in → 5.5in.** Three constructs were adjusted. The Figure 1 legend
row ran past the right edge once the diagram was no longer confined to a
minipage, so its last swatch was pulled in by 0.4 TikZ units. `\tabcolsep` in
Table 2 went 5pt → 4pt (it was 2.9pt overfull). Equation (9) was reflowed onto
two lines inside `gathered` — same content, still one equation number — where
it had been 18.6pt overfull. Freed of its minipage, the Figure 1 diagram
occupied under half the text width, so it is wrapped in
`\resizebox{0.75\textwidth}{!}{...}`, which scales the labels with the cells
and leaves every internal proportion as drawn.
