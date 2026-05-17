#!/bin/bash
# Project the canonical tkf-dp + tkf-mixdom + bio-datasets trees into a
# clean drop under $STAGING (default ~/staging/tkfdp/), applying the
# exclusion patterns from REPRODUCTION_MANIFEST.md Section 3 and the
# URL parameterisation from Section "GitHub release URLs".
#
# Single source of truth: edit code/paper in the canonical trees; re-run
# this script to refresh staging. Never edit staging directly.
#
# Usage: scripts/stage_clean_drop.sh [STAGING_DIR]
set -euo pipefail

STAGING=${1:-$HOME/staging/tkfdp}
SRC_TKF_DP=$HOME/tkf-dp
SRC_TKF_MIXDOM=$HOME/tkf-mixdom
SRC_BIO_DATASETS=$HOME/bio-datasets
# Local scrub literals (NOT shipped — sed -e expands these so the staged
# script only contains the placeholder, never the literal).
MAINTAINER_EMAIL_LITERAL="${MAINTAINER_EMAIL}"

echo "Staging into: $STAGING"
mkdir -p "$STAGING"/{tkf-dp,tkf-mixdom,bio-datasets/fetch}

# Common exclude flags
RSYNC_COMMON_EXCL=(
  --exclude='.git/' --exclude='.claude/' --exclude='__pycache__/' --exclude='*.pyc'
  --exclude='.venv/' --exclude='.pytest_cache/'
  --exclude='*.aux' --exclude='*.bbl' --exclude='*.blg' --exclude='*.log'
  --exclude='*.toc' --exclude='*.out' --exclude='*.build.log'
  --exclude='_old_*' --exclude='_aborted_*' --exclude='_partial_*'
  --exclude='_v[0-9]*_*'
)

# ============== tkf-dp ==============
rsync -a --delete --delete-excluded "${RSYNC_COMMON_EXCL[@]}" \
  --exclude='logs/' --exclude='data/pdb_cache/' \
  --exclude='combined.pdf' --exclude='math-paper/main.pdf' \
  --exclude='math-paper/supplement.pdf' \
  --exclude='math-paper/figures/aa_evolution_*' \
  --exclude='math-paper/.review/' \
  --exclude='refs/' \
  --exclude='math-paper/tkf-mixdom/' \
  --exclude='CLAUDE.md' --exclude='handoff_*.md' \
  --exclude='results/legacy_pre_*' \
  --exclude='analysis/artifacts_inv*' --exclude='analysis/k4_kh4/' \
  --exclude='analysis/k4_pdb*/' --exclude='analysis/re_diag/' \
  --exclude='psb-paper/' \
  "$SRC_TKF_DP/" "$STAGING/tkf-dp/"

# Drop dev-only exp* / preprocess scripts
for pat in 'exp1_*' 'exp2_compare_*' 'exp2_dca_*' 'exp2_heldout_*' \
           'exp2_pf00027_*' 'exp2_pfam_alpha_sweep.py' 'exp2_pfam_l2_sweep.py' \
           'exp2_pfam_K.py' 'exp2_potts_dp_synthetic.py' 'exp2_synthetic.py' \
           'exp2_val_compare*' 'exp3_*' 'em_stochastic_compare.py' \
           'build_pfam_full_corpus.py' 'preprocess_pfam_full_stress.py'; do
  find "$STAGING/tkf-dp/experiments" -maxdepth 2 -name "$pat" -delete 2>/dev/null || true
done

# Drop research-notes analysis MDs
for pat in '*_evaluation.md' 'inv1*' 'inv2*' 'k4_pdb*' 'k4_pdbrestrict*' 're_diag'; do
  find "$STAGING/tkf-dp/analysis" -maxdepth 2 -name "$pat" -exec rm -rf {} + 2>/dev/null || true
done

# Drop top-level handoff / planning docs
for f in gravestone_evaluation.md gravestone_implementation.md \
         implementation_notes.md pfam_evaluation.md \
         variational_quality_evaluation.md handoff_*.md; do
  rm -f "$STAGING/tkf-dp/$f" 2>/dev/null || true
done

# Stub the submodule path with a README; full content lives in $STAGING/tkf-mixdom
rm -rf "$STAGING/tkf-dp/math-paper/tkf-mixdom"
mkdir -p "$STAGING/tkf-dp/math-paper/tkf-mixdom"
cat > "$STAGING/tkf-dp/math-paper/tkf-mixdom/README.md" << 'EOF'
This directory is the tkf-mixdom git submodule. After `git clone`:
  cd tkf-dp && git submodule update --init --recursive
to populate it. Same content available at $STAGING/tkf-mixdom/.
EOF

# ============== tkf-mixdom ==============
rsync -a --delete --delete-excluded "${RSYNC_COMMON_EXCL[@]}" \
  --exclude='python/tests/' \
  --exclude='misc/' \
  --exclude='python/pfam/*.npz.tmp' \
  --exclude='python/.venv/' --exclude='python/.cache/' \
  --exclude='python/tkfmixdom.egg-info/' \
  --exclude='data/' \
  --exclude='CLAUDE.md' \
  --exclude='python/experiments/AUTONOMOUS_PLAN.md' \
  --exclude='scripts/aws_resumption.md' \
  "$SRC_TKF_MIXDOM/" "$STAGING/tkf-mixdom/"

# Trim heavyweight derived artefacts (regen via fetch + train; release-hosted)
rm -rf "$STAGING/tkf-mixdom/python/pfam/jax_cache" \
       "$STAGING/tkf-mixdom/python/pfam/precompiled" \
       "$STAGING/tkf-mixdom/python/pfam/precompiled_v"[0-9]* \
       "$STAGING/tkf-mixdom/python/pfam/precompiled_5k" \
       "$STAGING/tkf-mixdom/python/pfam/cherries_tkf92" \
       "$STAGING/tkf-mixdom/python/gamma_labels" \
       "$STAGING/tkf-mixdom/python/gamma_labels_G6" \
       "$STAGING/tkf-mixdom/python/data"
find "$STAGING/tkf-mixdom/python/pfam" -maxdepth 1 -name "PF*.sto" -delete 2>/dev/null || true
find "$STAGING/tkf-mixdom/python/pfam" -maxdepth 1 -name "PF*.fasta" -delete 2>/dev/null || true

# Trim dev-only experiments json + scratch dirs
rm -rf "$STAGING/tkf-mixdom/python/experiments/varanc_vanilla_tkf_quarantine" \
       "$STAGING/tkf-mixdom/python/experiments/contaminated_val_split_triage_for_deletion"
rm -f "$STAGING/tkf-mixdom/python/experiments/treefam_pfam_mapping.json"

# Keep only the 7 canonical *_withsps.json files in expected_balibase/
EB="$STAGING/tkf-mixdom/python/experiments/expected_balibase"
if [ -d "$EB" ]; then
  find "$EB" -maxdepth 1 -type f \
    ! -name 'expected_balibase_tkf92_withsps.json' \
    ! -name 'expected_balibase_mixdom_d3f1_withsps.json' \
    ! -name 'expected_balibase_tkf92_K20_withsps.json' \
    ! -name 'expected_balibase_cherryml_C20_withsps.json' \
    ! -name 'expected_balibase_mafft_withsps.json' \
    ! -name 'expected_balibase_muscle_withsps.json' \
    ! -name 'expected_balibase_inf_phmm_K4_l150_withsps.json' \
    -delete
fi

# ============== bio-datasets (fetch scripts only) ==============
mkdir -p "$STAGING/bio-datasets/fetch"
for ds in pfam balibase treefam rfam oxbench common.py; do
  if [ -e "$SRC_BIO_DATASETS/fetch/$ds" ]; then
    rsync -a --delete "$SRC_BIO_DATASETS/fetch/$ds" "$STAGING/bio-datasets/fetch/"
  fi
done
[ -f "$SRC_BIO_DATASETS/README.md" ] && cp "$SRC_BIO_DATASETS/README.md" "$STAGING/bio-datasets/" || true

# ============== Root MANIFEST + README ==============
# Canonical manifest IS the parameterised one — see top of REPRODUCTION_MANIFEST.md
cp "$SRC_TKF_DP/REPRODUCTION_MANIFEST.md" "$STAGING/REPRODUCTION_MANIFEST.md"

cat > "$STAGING/README.md" << 'EOF'
# tkfdp.net reproduction drop

Three top-level directories (sister repos):
- `tkf-dp/` — paper, K=4 sampler + figure scripts, AWS infra
- `tkf-mixdom/` — JAX TKF / MixDom inference library
- `bio-datasets/` — dataset fetch scripts (no data; fetch on demand)

See `REPRODUCTION_MANIFEST.md` for the full file list + step-by-step recipe.
EOF

# ============== Credential + ihh-path scrub (idempotent, run every time) ==============
# 1. Replace ihh/{tkf-dp,tkf-mixdom,bio-datasets,tkfdp} -> ${REPO_OWNER}/... in
#    .md/.sh/.py/.bib/.gitmodules and gmail email. Skip .tex (author email is
#    real) and binary files. Skip sibling-project refs (ihh/machineboss,
#    ihh/subby — real other-project URLs, not the tkfdp drop).
find "$STAGING" -type f \
  \( -name '*.md' -o -name '*.sh' -o -name '*.py' -o -name '*.bib' -o -name '.gitmodules' -o -name '*.tex' \) \
  ! -path '*/.git/*' -print0 \
  | xargs -0 sed -i -E \
    -e 's|${REPO_OWNER}/tkfdp|${REPO_OWNER}/tkfdp|g' \
    -e 's|${REPO_OWNER}/tkf-mixdom|${REPO_OWNER}/tkf-mixdom|g' \
    -e 's|${REPO_OWNER}/bio-datasets|${REPO_OWNER}/bio-datasets|g' \
    -e 's|${REPO_OWNER}/tkfdp|${REPO_OWNER}/tkfdp|g' \
    -e 's|github\.com:${REPO_OWNER}/tkfdp|github.com:${REPO_OWNER}/tkfdp|g' \
    -e 's|github\.com:${REPO_OWNER}/tkf-mixdom|github.com:${REPO_OWNER}/tkf-mixdom|g' \
    -e 's|github\.com:${REPO_OWNER}/bio-datasets|github.com:${REPO_OWNER}/bio-datasets|g' \
    -e 's|github\.com:${REPO_OWNER}/tkfdp|github.com:${REPO_OWNER}/tkfdp|g' \
    -e 's|github\.com/${REPO_OWNER}/tkfdp|github.com/${REPO_OWNER}/tkfdp|g' \
    -e 's|github\.com/${REPO_OWNER}/tkf-mixdom|github.com/${REPO_OWNER}/tkf-mixdom|g' \
    -e 's|github\.com/${REPO_OWNER}/bio-datasets|github.com/${REPO_OWNER}/bio-datasets|g' \
    -e 's|github\.com/${REPO_OWNER}/tkfdp|github.com/${REPO_OWNER}/tkfdp|g' \
    -e "s|${MAINTAINER_EMAIL_LITERAL}|\${MAINTAINER_EMAIL}|g" \
    -e 's|/home/yam/\.claude/[^\" ]*|${CLAUDE_LOCAL_PATH}|g' \
    -e 's|\.claude/agents/[^ ]*\.md|${CLAUDE_AGENTS_PATH}|g'

# 2. Refuse to ship if any credential pattern leaks. Audit + abort if hit.
#    Skip this script itself (the regex literals look like leaks).
LEAKS=$(grep -rlE \
  'AWS_ACCESS_KEY_ID|AWS_SECRET_ACCESS_KEY|aws_access_key_id|aws_secret_access_key|AKIA[0-9A-Z]{16}|ANTHROPIC_API_KEY|sk-ant-[A-Za-z0-9_-]{20,}|sk-proj-[A-Za-z0-9_-]{20,}|GITHUB_TOKEN|gho_[A-Za-z0-9_]{30,}|ghp_[A-Za-z0-9_]{30,}|bearer [A-Za-z0-9_-]{30,}' \
  "$STAGING" 2>/dev/null | grep -vE '\.git/|stage_clean_drop\.sh$' || true)
if [ -n "$LEAKS" ]; then
  echo "FATAL: credential leak(s) detected in staging — refusing to ship:" >&2
  echo "$LEAKS" >&2
  exit 1
fi

# 3. Audit AWS bucket name + account number specifically (project-specific).
ACCOUNT_LEAKS=$(grep -rlE '618647024028|tkf-mixdom-gpu-618647024028' "$STAGING" \
  --include='*.sh' --include='*.py' 2>/dev/null | grep -v '\.git/' || true)
if [ -n "$ACCOUNT_LEAKS" ]; then
  echo "WARN: AWS account/bucket references in scripts (consider parameterising):" >&2
  echo "$ACCOUNT_LEAKS" >&2
fi

# 4. Stray .claude/ or .anthropic/ directories shouldn't exist (rsync excludes
#    them) but double-check.
STRAY=$(find "$STAGING" -type d \( -name '.claude' -o -name '.anthropic' \) 2>/dev/null)
if [ -n "$STRAY" ]; then
  echo "FATAL: stray Claude/Anthropic directories — refusing to ship:" >&2
  echo "$STRAY" >&2
  exit 1
fi

echo
echo "=== Final state ==="
du -sh "$STAGING"/{tkf-dp,tkf-mixdom,bio-datasets} 2>/dev/null || true
du -sh "$STAGING" 2>/dev/null
echo "File count: $(find "$STAGING" -type f | wc -l)"
echo
echo "Done. Inspect $STAGING/ then commit/push the desired branch state."
