#!/bin/bash
# Build the Molecular Biology and Evolution submission of paper 1:
#   mbe_paper1.pdf     the single PDF MBE asks you to upload
#   cover_letter.pdf   the accompanying cover letter
#
# There is no separate supplementary file: MBE sets no page limit, so the
# former appendices are subsections of Materials and Methods.
#
# This directory is self-contained.  Nothing outside it is needed to build:
# no BIBINPUTS, no TEXINPUTS, no sibling directories.  refs.bib and the figure
# are local copies.  Regenerating the .tex from the NeurIPS source is a
# separate step -- see port-from-neurips.py.
#
# Usage:
#   ./build.sh                # build the PDFs
#   ./build.sh --check        # build + MBE compliance report (read before submitting)
#   ./build.sh --open         # build + open the manuscript PDF
#   ./build.sh --clean        # remove intermediates and PDFs
#
# Requires: pdflatex, bibtex.

set -euo pipefail
cd "$(dirname "$0")"

PAPER=mbe_paper1

build_one() {
    local base="$1"
    : > "${base}.build.log"
    echo "==> pdflatex ${base} (1/3)"
    pdflatex -interaction=nonstopmode -halt-on-error "${base}.tex" \
        >> "${base}.build.log" 2>&1 || { tail -40 "${base}.build.log" >&2; exit 1; }
    if [ -f "${base}.aux" ] && grep -q '\\bibdata' "${base}.aux"; then
        echo "==> bibtex ${base}"
        bibtex "${base}" >> "${base}.build.log" 2>&1 \
            || { tail -40 "${base}.build.log" >&2; exit 1; }
    fi
    echo "==> pdflatex ${base} (2/3)"
    pdflatex -interaction=nonstopmode -halt-on-error "${base}.tex" \
        >> "${base}.build.log" 2>&1 || { tail -40 "${base}.build.log" >&2; exit 1; }
    echo "==> pdflatex ${base} (3/3)"
    pdflatex -interaction=nonstopmode -halt-on-error "${base}.tex" \
        >> "${base}.build.log" 2>&1 || { tail -40 "${base}.build.log" >&2; exit 1; }
    local pages
    pages=$(pdfinfo "${base}.pdf" 2>/dev/null | awk '/^Pages:/ {print $2}' || true)
    if [ -z "${pages:-}" ] && command -v mutool >/dev/null 2>&1; then
        pages=$(mutool info "${base}.pdf" 2>/dev/null | awk '/^Pages:/ {print $2}' || true)
    fi
    echo "==> ${base}.pdf produced ($(wc -c < "${base}.pdf") bytes, ${pages:-?} pages)"
}

build_letter() {
    if [ -f cover_letter.tex ]; then
        echo "==> pdflatex cover_letter"
        pdflatex -interaction=nonstopmode -halt-on-error cover_letter.tex \
            > cover_letter.build.log 2>&1 \
            || { tail -40 cover_letter.build.log >&2; exit 1; }
        echo "==> cover_letter.pdf produced"
    fi
}

# Report what the MBE author guidelines ask for at initial submission.
check_one() {
    local base="$1"
    echo
    echo "============ MBE compliance check (paper 1) ============"

    echo "-- page size and margins --"
    if command -v pdfinfo >/dev/null 2>&1; then
        pdfinfo "${base}.pdf" | grep 'Page size' | sed 's/^/   /'
    fi
    echo "   geometry: 25 mm all round (MBE asks for ~25 mm)"

    echo "-- required elements --"
    for needle in "Article type: Methods" "Abstract" "Key words" \
                  "section{Introduction}" "section{Materials and Methods}" \
                  "section*{Data availability}" "section*{Acknowledgements}" \
                  "subsection*{Funding}" "section*{Conflict of interest}"; do
        if grep -qF "$needle" "${base}.tex"; then
            echo "   present: ${needle}"
        else
            echo "   MISSING: ${needle}" >&2
        fi
    done
    # Matched exactly so a "Results:" subheading elsewhere cannot mask a
    # missing top-level section.
    for needle in '^\\section\{Results\}$' '^\\section\{Discussion\}$'; do
        if grep -qE "$needle" "${base}.tex"; then
            echo "   present: ${needle}"
        else
            echo "   MISSING: ${needle}" >&2
        fi
    done

    echo "-- top-level section order (MBE permits Methods before Results) --"
    grep -nE '^\\section\*?\{' "${base}.tex" | sed 's/^/   /'

    echo "-- abstract length (limit 250 words) --"
    python3 - "${base}.tex" <<'PY'
import re, sys
s = open(sys.argv[1]).read()
a = s.split(r'\noindent\textbf{Abstract}')[1].split(r'\vspace{1em}')[0]
a = re.sub(r'(?m)^\s*%.*$', '', a)
a = re.sub(r'\$[^$]*\$', ' Xmath ', a)
a = re.sub(r'\\[a-zA-Z@]+\*?', ' ', a)
a = a.replace('{', ' ').replace('}', ' ').replace('\\', ' ')
n = sum(1 for w in a.split() if re.search(r'[A-Za-z0-9]', w))
print("   abstract: %d words%s"
      % (n, "" if n <= 250 else "   OVER THE LIMIT -- trim %d" % (n - 250)))
PY

    echo "-- line numbers and double spacing --"
    grep -q '\\linenumbers' "${base}.tex" && echo "   line numbers: on" \
        || echo "   line numbers: OFF" >&2
    grep -q '\\doublespacing' "${base}.tex" && echo "   spacing: double" \
        || echo "   spacing: NOT double" >&2

    echo "-- citations must be author-year --"
    echo "   bare \\cite{} left (should be 0): $(grep -c '\\cite{' "${base}.tex" || true)"

    echo "-- alt text (MBE requires one per figure, under the legend) --"
    local nfig nalt
    # sed strips commented-out lines so retired figures are not counted
    nfig=$(sed 's/^[[:space:]]*%.*$//' "${base}.tex" | grep -c '\\begin{figure}' || true)
    nalt=$(grep -c 'Alt text:' "${base}.tex" || true)
    echo "   figures: ${nfig}; alt-text blocks: ${nalt}"
    [ "${nfig}" -eq "${nalt}" ] || echo "   ATTENTION: one 'Alt text:' line per figure" >&2

    echo "-- unresolved cross-references and citations --"
    if grep -qiE 'Citation .* undefined|Reference .* undefined' "${base}.log"; then
        grep -iE 'Citation .* undefined|Reference .* undefined' "${base}.log" \
            | sort -u | head -20 | sed 's/^/   /'
    else
        echo "   none"
    fi

    echo "-- overfull boxes --"
    if grep -qE 'Overfull \\(h|v)box' "${base}.log"; then
        echo "   count: $(grep -cE 'Overfull \\(h|v)box' "${base}.log")"
    else
        echo "   none"
    fi

    echo "-- still to do before you submit --"
    echo "   * ORCID iD for the submitting author (entered in the portal, not the PDF)"
    echo "   * confirm the repository URL in Data availability is public"
    echo "   * references are abbrvnat, i.e. roughly MBE style; MBE requires exact"
    echo "     MBE reference format only at revision"
    echo "   * cover letter: cover_letter.pdf"
    echo "========================================================"
}

clean_one() {
    rm -f "${PAPER}".{aux,bbl,blg,log,out,toc,fls,fdb_latexmk,build.log} "${PAPER}.pdf" \
          cover_letter.{aux,log,out,build.log,pdf}
}

open_pdf() {
    if [ ! -f "$1" ]; then echo "WARN: $1 not found." >&2; return 0; fi
    if command -v open >/dev/null 2>&1; then open "$1"
    elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$1" >/dev/null 2>&1 &
    else echo "WARN: no opener available." >&2; fi
}

MODE=build
OPEN=0
CHECK=0
for arg in "$@"; do
    case "$arg" in
        --open)  OPEN=1 ;;
        --check) CHECK=1 ;;
        --clean) MODE=clean ;;
        -h|--help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        "") ;;
        *) echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done

case "$MODE" in
    build)
        build_one "${PAPER}"
        build_letter
        if [ "$CHECK" -eq 1 ]; then check_one "${PAPER}"; fi
        if [ "$OPEN" -eq 1 ]; then open_pdf "${PAPER}.pdf"; fi
        ;;
    clean) clean_one; echo "==> cleaned" ;;
esac
