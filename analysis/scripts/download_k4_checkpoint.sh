#!/bin/bash
# Download the K=4 EM-warmup checkpoint from the public ${REPO_OWNER}/tkfdp release.
# Requires gh CLI authenticated to a user with read access to the repo.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p results
gh release download results/K4-emwarm-top1000-2026-05-09 \
    --repo ${REPO_OWNER}/tkfdp \
    -D results/ \
    -p "K4-emwarm-top1000-2026-05-09.tar.gz"
cd results
tar -xzf K4-emwarm-top1000-2026-05-09.tar.gz
echo "Unpacked to results/K4-emwarm-top1000-2026-05-09/_best_chkpt/state.npz"
