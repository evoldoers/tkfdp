#!/bin/bash
# Build the math paper (and supplement, if it exists).
#
# Usage:
#   ./build.sh                       # build main paper only
#   ./build.sh --all                 # build main + supplement (two PDFs)
#   ./build.sh --supp                # build supplement only
#   ./build.sh --combined            # build both, concatenate into
#                                    # combined.pdf
#   ./build.sh --clean               # remove intermediates and PDFs
#   ./build.sh [mode] --open         # also open the resulting PDF
#                                    # (macOS `open`, fallback xdg-open)
#
# Requires: pdflatex, bibtex; the tkf-mixdom git submodule must be
# initialised (`git submodule update --init --recursive` from ~/tkf-dp/).
# --combined additionally requires pdfunite (poppler-utils) or
# ghostscript.

set -euo pipefail

cd "$(dirname "$0")"

require_submodule() {
    if [ ! -e tkf-mixdom/tkf/tkf.tex ]; then
        echo "ERROR: tkf-mixdom submodule not initialised." >&2
        echo "From ~/tkf-dp/: git submodule update --init --recursive" >&2
        exit 1
    fi
}

build_one() {
    # Run pdflatex+bibtex+pdflatex+pdflatex on a single document.
    # $1 = base name (without .tex).
    local base="$1"
    if [ ! -f "${base}.tex" ]; then
        echo "WARN: ${base}.tex not found; skipping." >&2
        return 0
    fi
    require_submodule
    # TEXINPUTS lets pdflatex find files \input'd by submodule bodies
    # using relative paths (e.g. lhopital-limits.tex \input{joint-vs-conditional}).
    # Leading colon means "search this in addition to the default path".
    export TEXINPUTS=".:./tkf-mixdom/tkf:${TEXINPUTS:-}"
    export BIBINPUTS=".:./tkf-mixdom/tkf:${BIBINPUTS:-}"
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
    grep -E "Warning|Underfull|Overfull|undefined" "${base}.log" | \
        head -20 || true
    echo "==> ${base}.pdf produced ($(wc -c < "${base}.pdf") bytes, \
$(pdfinfo "${base}.pdf" 2>/dev/null | awk '/Pages:/ {print $2}') pages)"
}

clean_one() {
    local base="$1"
    rm -f "${base}".{aux,bbl,blg,log,out,toc,bcf,fls,fdb_latexmk,build.log,run.xml,nav,snm} \
          "${base}".pdf
}

clean_combined() {
    rm -f combined.pdf
}

open_pdf() {
    # Open a PDF in the system default viewer. macOS `open` -> Preview.
    # Linux fallback: xdg-open. No-op if neither available.
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

# Parse args: first positional is the mode; --open is a modifier.
MODE="--main"
OPEN=0
for arg in "$@"; do
    case "$arg" in
        --open) OPEN=1 ;;
        -h|--help|--main|--supp|--supplement|--all|--combined|--clean)
            MODE="$arg" ;;
        "") ;;
        *)
            echo "unknown arg: $arg" >&2
            exit 2
            ;;
    esac
done

case "$MODE" in
    --main)
        build_one main
        [ "$OPEN" -eq 1 ] && open_pdf main.pdf
        ;;
    --supp|--supplement)
        build_one supplement
        [ "$OPEN" -eq 1 ] && open_pdf supplement.pdf
        ;;
    --all)
        build_one main
        build_one supplement
        [ "$OPEN" -eq 1 ] && { open_pdf main.pdf; open_pdf supplement.pdf; }
        ;;
    --combined)
        build_one main
        build_one supplement
        if command -v pdfunite >/dev/null 2>&1; then
            echo "==> pdfunite main.pdf + supplement.pdf -> combined.pdf"
            pdfunite main.pdf supplement.pdf combined.pdf
        elif command -v gs >/dev/null 2>&1; then
            echo "==> ghostscript merge -> combined.pdf"
            gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite \
               -sOutputFile=combined.pdf main.pdf supplement.pdf
        else
            echo "ERROR: need pdfunite (poppler-utils) or ghostscript" >&2
            exit 1
        fi
        echo "==> combined.pdf produced ($(wc -c < combined.pdf) bytes, \
$(pdfinfo combined.pdf 2>/dev/null | awk '/Pages:/ {print $2}') pages)"
        [ "$OPEN" -eq 1 ] && open_pdf combined.pdf
        ;;
    --clean)
        clean_one main
        clean_one supplement
        clean_combined
        echo "==> cleaned"
        ;;
    -h|--help)
        head -n 17 "$0" | sed -n '2,$p'
        ;;
esac
