#!/bin/bash
# Build the NeurIPS 2026 workshop version of the coupled-substitution paper:
#   neurips_workshop_paper2.tex  (coupled models of amino acid substitution)
#
# This directory is self-contained.  Nothing outside it is needed to build:
# no BIBINPUTS, no TEXINPUTS, no sibling directories.  The .bib file and the
# three figures are local copies.
#
# neurips_2026.sty and checklist.tex are the official template files and must
# not be modified -- tweaking the style files is grounds for desk rejection.
# --check verifies their SHA-256 against the values recorded below.
#
# Usage:
#   ./build.sh                # build neurips_workshop_paper2.pdf
#   ./build.sh --check        # build + compliance report (read this before submitting)
#   ./build.sh --open         # build + open the resulting PDF
#   ./build.sh --clean        # remove intermediates and the PDF
#
# Requires: pdflatex, bibtex.

set -euo pipefail
cd "$(dirname "$0")"

PAPER=neurips_workshop_paper2

# SHA-256 of the official NeurIPS 2026 template files, as shipped.
STY_SHA=c3fc2894e83d2517ca18b66741d6c595986d97957dc08ec08bb2125a7ec4555a
CHK_SHA=780ba13c480f652dcc42e69ed61a752ce0ea270f15d332d4a45b059dabad84f6

build_one() {
    local base="$1"
    echo "==> pdflatex ${base} (1/3)"
    pdflatex -interaction=nonstopmode -halt-on-error "${base}.tex" \
        > "${base}.build.log" 2>&1 || \
        { tail -40 "${base}.build.log" >&2; exit 1; }
    if [ -f "${base}.aux" ] && grep -q "\\\\bibdata" "${base}.aux"; then
        echo "==> bibtex ${base}"
        bibtex "${base}" >> "${base}.build.log" 2>&1 || \
            { tail -40 "${base}.build.log" >&2; exit 1; }
    fi
    echo "==> pdflatex ${base} (2/3)"
    pdflatex -interaction=nonstopmode -halt-on-error "${base}.tex" \
        >> "${base}.build.log" 2>&1 || \
        { tail -40 "${base}.build.log" >&2; exit 1; }
    echo "==> pdflatex ${base} (3/3)"
    pdflatex -interaction=nonstopmode -halt-on-error "${base}.tex" \
        >> "${base}.build.log" 2>&1 || \
        { tail -40 "${base}.build.log" >&2; exit 1; }
    grep -E "Warning|Underfull|Overfull|undefined" "${base}.log" | head -20 || true
    PAGES=$(pdfinfo "${base}.pdf" 2>/dev/null | awk '/^Pages:/ {print $2}' || true)
    if [ -z "${PAGES:-}" ] && command -v mutool >/dev/null 2>&1; then
        PAGES=$(mutool info "${base}.pdf" 2>/dev/null | awk '/^Pages:/ {print $2}' || true)
    fi
    echo "==> ${base}.pdf produced ($(wc -c < "${base}.pdf") bytes, ${PAGES:-?} pages)"
}

# Report what NeurIPS desk-checks: unmodified style files, page size, content
# pages against the limit, embedded font types, and boxes overflowing the
# 5.5in x 9in text block.  Uses pdfinfo/pdffonts when present, else mutool.
check_one() {
    local base="$1"
    local pdf="${base}.pdf"
    echo
    echo "================ NeurIPS compliance check ================"

    echo "-- template files must be unmodified --"
    local got
    got=$(shasum -a 256 neurips_2026.sty | awk '{print $1}')
    if [ "$got" = "$STY_SHA" ]; then
        echo "   neurips_2026.sty: unmodified"
    else
        echo "   WARN: neurips_2026.sty has been MODIFIED -- desk-rejection risk" >&2
    fi
    got=$(shasum -a 256 checklist.tex | awk '{print $1}')
    if [ "$got" = "$CHK_SHA" ]; then
        echo "   checklist.tex:    unmodified (answers are filled in separately)"
    else
        echo "   note: checklist.tex differs from the shipped copy"
        echo "         (expected once you replace the \\answerTODO{} macros)"
    fi

    local pages="" dump=""
    if command -v pdfinfo >/dev/null 2>&1; then
        pages=$(pdfinfo "$pdf" | awk '/^Pages:/ {print $2}')
        echo "-- page size (must be US Letter, 612 x 792 pt) --"
        pdfinfo "$pdf" | grep 'Page size' | sed 's/^/   /'
    elif command -v mutool >/dev/null 2>&1; then
        pages=$(mutool info "$pdf" 2>/dev/null | awk '/^Pages:/ {print $2}')
        echo "-- page size (must be US Letter, 612 x 792 pt) --"
        mutool pages "$pdf" 1 2>/dev/null | grep -i MediaBox | head -1 | sed 's/^/   /'
    fi
    echo "   total pages: ${pages:-?}"

    echo "-- content pages (ack, references and checklist do not count) --"
    if command -v pdftotext >/dev/null 2>&1; then
        dump=$(pdftotext -layout "$pdf" - 2>/dev/null)
    elif command -v mutool >/dev/null 2>&1; then
        dump=$(mutool draw -F txt -o - "$pdf" 2>/dev/null)
    fi
    if [ -n "$dump" ]; then
        local refpage
        refpage=$(printf '%s' "$dump" | awk '
            /\f/ {page++}
            /^[[:space:]]*References[[:space:]]*$/ && !found {print page+1; found=1}' | head -1)
        if [ -n "${refpage:-}" ]; then
            echo "   References begins on page ${refpage}"
            echo "   => ${refpage} content pages"
            if [ "${refpage}" -gt 9 ]; then
                echo "   ATTENTION: over the nine pages the template states for the"
                echo "   main track; workshop limits are usually shorter still." >&2
            fi
        else
            echo "   (could not locate the References heading; check by eye)"
        fi
    else
        echo "   (no pdftotext or mutool; check by eye)"
    fi

    echo "-- font types (Type 1 or embedded TrueType only; no Type 3) --"
    if command -v pdffonts >/dev/null 2>&1; then
        local nt3
        nt3=$(pdffonts "$pdf" | grep -c 'Type 3' || true)
        pdffonts "$pdf" | awk 'NR<=2' | sed 's/^/   /'
        echo "   Type 3 fonts: ${nt3}"
        if [ "${nt3}" -gt 0 ]; then echo "   WARN: Type 3 fonts present." >&2; fi
    elif command -v mutool >/dev/null 2>&1; then
        mutool info -F "$pdf" 2>/dev/null \
            | grep -oE 'Type1|TrueType|Type3|Type0' | sort | uniq -c | sed 's/^/   /'
        mutool info -F "$pdf" 2>/dev/null | grep -q Type3 && \
            echo "   WARN: Type 3 fonts present." >&2 || true
    else
        echo "   (neither pdffonts nor mutool installed; skipping)"
    fi

    echo "-- overfull boxes (text must stay inside 5.5in x 9in) --"
    if grep -qE 'Overfull \\(h|v)box' "${base}.log"; then
        grep -E 'Overfull \\(h|v)box' "${base}.log" | head -20 | sed 's/^/   /'
    else
        echo "   none"
    fi

    echo "-- unresolved cross-references and citations --"
    if grep -qiE 'Citation .* undefined|Reference .* undefined' "${base}.log"; then
        grep -iE 'Citation .* undefined|Reference .* undefined' "${base}.log" \
            | head -10 | sed 's/^/   /'
    else
        echo "   none"
    fi

    echo "-- items still needing your attention --"
    local todo
    todo=$(grep -c 'Answer: .answerTODO' checklist.tex 2>/dev/null || true)
    echo "   checklist questions still unanswered: ${todo:-?}"
    if grep -q 'workshoptitle{WORKSHOP TITLE}' "${base}.tex"; then
        echo "   \\workshoptitle is still the placeholder WORKSHOP TITLE"
    fi
    # match the active \usepackage line, not the comment describing it
    if ! grep -qE '^\\usepackage\[dblblindworkshop, *final\]' "${base}.tex"; then
        echo "   building in submission mode (anonymous, line numbers);"
        echo "     add \"final\" to the style options for the camera-ready copy"
    fi
    echo "   see README.md for the full list"
    echo "=========================================================="
}

clean_one() {
    rm -f "${PAPER}".{aux,bbl,blg,log,out,toc,bcf,fls,fdb_latexmk,build.log,run.xml} \
          "${PAPER}".pdf
}

open_pdf() {
    local pdf="$1"
    if [ ! -f "$pdf" ]; then
        echo "WARN: ${pdf} not found; cannot open." >&2
        return 0
    fi
    if command -v open >/dev/null 2>&1; then
        open "$pdf"
    elif command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$pdf" >/dev/null 2>&1 &
    else
        echo "WARN: neither 'open' nor 'xdg-open' available; not opening." >&2
    fi
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
        if [ "$CHECK" -eq 1 ]; then check_one "${PAPER}"; fi
        if [ "$OPEN" -eq 1 ]; then open_pdf "${PAPER}.pdf"; fi
        ;;
    clean) clean_one; echo "==> cleaned" ;;
esac
