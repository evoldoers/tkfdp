#!/bin/bash
# publish_drop.sh — publish the dev repos to their public evoldoers/*
# counterparts in one command. Standalone: no manual steps, no agent.
#
#   ~/tkf-dp       (${REPO_OWNER}/tkfdp)        -> evoldoers/tkfdp        (paper + code; submodules)
#   ~/tkf-mixdom   (${REPO_OWNER}/tkf-mixdom)    -> evoldoers/tkf-mixdom   (supporting library)
#   ~/bio-datasets (${REPO_OWNER}/bio-datasets)  -> evoldoers/bio-datasets (fetch scripts)
#   + refreshes evoldoers/tkfdp.net    (committed supplement.pdf; PDF-only)
#
# Pipeline:
#   1. stage_clean_drop.sh           scrub + subset dev trees -> ~/staging/tkfdp
#   2. per target: refresh a working clone under ~/.cache/tkfdp-publish,
#      rsync the staged subset in (ADDITIVE — never --delete, so released
#      checkpoints / result JSONs are preserved), force-add dev-tracked
#      figure PDFs that the published `*.pdf` .gitignore would otherwise
#      drop, re-pin the tkfdp submodule to the freshly-pushed tkf-mixdom
#      commit, BUILD-VERIFY the papers (abort if they don't compile),
#      then commit + push.
#
# Why each guard exists (all learned the hard way, see CLAUDE.md):
#   * GNU sed         — stage_clean_drop.sh uses `sed -i -E`; macOS sed is
#                       BSD, so we shim `gsed` onto PATH.
#   * force-add PDFs  — figure PDFs are `*.pdf` build artifacts force-added
#                       (`git add -f`) in dev; a plain `git add` in the
#                       published clone silently drops them and the paper
#                       won't build (e.g. tkf_trajectory_uber.pdf,
#                       pfam_closedform.pdf).
#   * additive rsync  — a `--delete` mirror would remove released
#                       checkpoints (results/K4-*) that live only on the
#                       published side.
#   * exclude README/.gitmodules — hand-tuned on the published side
#                       (bioRxiv DOI link; concrete evoldoers submodule URL).
#   * build-verify    — never push a drop whose paper does not compile.
#
# Usage:  bash scripts/publish_drop.sh [--no-push]
# Needs:  GNU sed (or gsed), git + gh (auth'd to evoldoers), rsync, pdflatex+bibtex.
set -euo pipefail

OWNER=evoldoers
STAGING="$HOME/staging/tkfdp"
WORK="$HOME/.cache/tkfdp-publish"
HERE="$(cd "$(dirname "$0")" && pwd)"
PUSH=1; [ "${1:-}" = "--no-push" ] && PUSH=0
mkdir -p "$WORK"

# 0. Ensure GNU sed for stage_clean_drop.sh's `sed -i -E`.
if ! sed --version >/dev/null 2>&1; then
  command -v gsed >/dev/null || { echo "ERROR: need GNU sed (brew install gnu-sed)" >&2; exit 1; }
  shimdir="$(mktemp -d)"; ln -sf "$(command -v gsed)" "$shimdir/sed"; export PATH="$shimdir:$PATH"
  echo "==> shimmed gsed -> sed (macOS BSD sed detected)"
fi

# 1. Stage the scrubbed clean drop.
echo "==> staging clean drop into $STAGING"
bash "$HERE/stage_clean_drop.sh" >/dev/null

# Refresh (or create) a working clone at $WORK/<repo> == origin/main.
refresh_clone() {                       # $1 = repo name on github.com/$OWNER
  local repo="$1" c="$WORK/$1"
  if [ -d "$c/.git" ]; then
    git -C "$c" fetch -q origin
    git -C "$c" reset -q --hard origin/main
    git -C "$c" clean -qfd
  else
    git clone -q "git@github.com:$OWNER/$repo.git" "$c"
  fi
}

# Re-add figure PDFs tracked in the dev source but dropped by *.pdf ignore.
readd_pdfs() {                          # $1 = clone dir   $2 = dev source dir
  ( cd "$2" && git ls-files '*.pdf' ) | while read -r f; do
    [ -f "$1/$f" ] && git -C "$1" add -f "$f" || true
  done
}

commit_push() {                         # $1 = clone dir   $2 = commit message
  git -C "$1" add -A
  if git -C "$1" diff --cached --quiet; then
    echo "    (no changes)"
  else
    git -C "$1" commit -q -m "$2" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
    if [ "$PUSH" = 1 ]; then
      git -C "$1" push -q origin HEAD:main
      echo "    pushed $(git -C "$1" rev-parse --short HEAD)"
    fi
  fi
}

# 2. tkf-mixdom (supporting library) — must go first; tkfdp pins it.
echo "==> $OWNER/tkf-mixdom"
refresh_clone tkf-mixdom
rsync -a --exclude='.git/' "$STAGING/tkf-mixdom/" "$WORK/tkf-mixdom/"
readd_pdfs "$WORK/tkf-mixdom" "$HOME/tkf-mixdom"
commit_push "$WORK/tkf-mixdom" "Refresh published supporting library from dev"
MIX_SHA="$(git -C "$WORK/tkf-mixdom" rev-parse HEAD)"

# 3. tkfdp (paper + code). Preserve hand-tuned README + .gitmodules and the
#    real submodules; re-pin tkf-mixdom; build-verify before pushing.
echo "==> $OWNER/tkfdp"
refresh_clone tkfdp
rsync -a --exclude='.git/' --exclude='.gitmodules' --exclude='README.md' \
      --exclude='math-paper/tkf-mixdom/' --exclude='bio-datasets/' \
      "$STAGING/tkf-dp/" "$WORK/tkfdp/"
git -C "$WORK/tkfdp" submodule update --init -q math-paper/tkf-mixdom 2>/dev/null || true
git -C "$WORK/tkfdp/math-paper/tkf-mixdom" fetch -q origin
git -C "$WORK/tkfdp/math-paper/tkf-mixdom" checkout -q "$MIX_SHA"
readd_pdfs "$WORK/tkfdp" "$HOME/tkf-dp"
echo "    build-verifying main + supplement..."
( cd "$WORK/tkfdp/math-paper"
  rm -f main.pdf supplement.pdf
  bash build.sh        >/dev/null 2>&1 || true
  bash build.sh --supp >/dev/null 2>&1 || true
  [ -s main.pdf ] && [ -s supplement.pdf ] ) \
  || { echo "ERROR: tkfdp paper build failed — not committing/pushing." >&2; exit 1; }
commit_push "$WORK/tkfdp" "Refresh published drop from dev (re-pin tkf-mixdom @ ${MIX_SHA:0:9})"

# 4. bio-datasets (fetch scripts).
echo "==> $OWNER/bio-datasets"
refresh_clone bio-datasets
rsync -a --exclude='.git/' --exclude='README.md' "$STAGING/bio-datasets/" "$WORK/bio-datasets/"
commit_push "$WORK/bio-datasets" "Refresh from dev"

# 5. tkfdp.net — refresh the committed supplement.pdf (PDF-only site; the
#    LaTeX->HTML conversion was removed).
echo "==> $OWNER/tkfdp.net"
refresh_clone tkfdp.net
cp "$WORK/tkfdp/math-paper/supplement.pdf" "$WORK/tkfdp.net/supplement.pdf"
commit_push "$WORK/tkfdp.net" "Refresh supplement.pdf"

echo "==> publish complete (push=$PUSH)."
