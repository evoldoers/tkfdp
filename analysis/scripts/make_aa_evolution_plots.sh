#!/usr/bin/env bash
# Regenerate all 10 AA-evolution bridge figures (5 transitions x 2 styles).
# See math-paper/figures/AA_EVOLUTION_RECIPE.md for documentation.

set -euo pipefail

PY=${PY:-$HOME/tkf-mixdom/python/.venv/bin/python}
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== pixel-density rendering ==="
"$PY" "$SCRIPTS_DIR/plot_aa_evolution_k4.py" --batch

echo
echo "=== stacked-area rendering ==="
"$PY" "$SCRIPTS_DIR/plot_aa_evolution_stacked.py" --batch

echo
echo "Done. Figures in math-paper/figures/aa_evolution_*.{pdf,png}"
