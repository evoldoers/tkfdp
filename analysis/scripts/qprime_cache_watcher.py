#!/usr/bin/env python3
"""Background watcher that mirrors Q'-cache updates into the tracked repo
directory and commits per-family.

Polls ``~/.cache/tkf-mixdom-balibase/<method>/`` every POLL_SECONDS for
new or updated `<family>.{npz,json}` files. For each detected change,
copies the file into ``math-paper/results/qprime_cache/<method>/`` and
commits a single per-family commit. Pushes at the end of each polling
cycle (with `git pull --rebase --autostash` to avoid conflicts with
concurrent manual work on the branch).

Stateless (uses mtime + filesize to detect changes). Robust to:
  - concurrent writes by multiple sampler workers (different families)
  - manual rebases on main
  - intermittent network failure (pull/push retried next cycle)

Usage:
    nohup python3 analysis/scripts/qprime_cache_watcher.py \
        > /tmp/qprime_cache_watcher.log 2>&1 &
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

CACHE_ROOT = Path.home() / ".cache" / "tkf-mixdom-balibase"
REPO_ROOT = Path.home() / "tkf-dp"
REPO_MIRROR = REPO_ROOT / "math-paper" / "results" / "qprime_cache"
POLL_SECONDS = 60

# Methods we want to mirror. Add new ones here.
METHODS = [
    "tkf92_lg08", "mixdom_d3f1", "cherryml_C20", "tkf92_K20",
    "mafft_auto", "muscle",
    "infinite_phmm_mcmc_K4_coupled_RE",
    "infinite_phmm_mcmc_K4_pdbanchor_RE",
]


def needs_copy(src: Path, dst: Path) -> bool:
    if not dst.exists():
        return True
    if src.stat().st_size != dst.stat().st_size:
        return True
    if src.stat().st_mtime > dst.stat().st_mtime + 0.5:
        return True
    return False


def scan_and_copy() -> list[tuple[str, str]]:
    """Return [(method, family)] of newly-copied files."""
    updated: list[tuple[str, str]] = []
    for method in METHODS:
        src_dir = CACHE_ROOT / method
        if not src_dir.is_dir():
            continue
        dst_dir = REPO_MIRROR / method
        dst_dir.mkdir(parents=True, exist_ok=True)
        for npz in sorted(src_dir.glob("*.npz")):
            fam = npz.stem
            json_src = src_dir / (fam + ".json")
            npz_dst = dst_dir / npz.name
            json_dst = dst_dir / (fam + ".json")
            changed = False
            if needs_copy(npz, npz_dst):
                shutil.copy2(npz, npz_dst)
                changed = True
            if json_src.exists() and needs_copy(json_src, json_dst):
                shutil.copy2(json_src, json_dst)
                changed = True
            if changed:
                updated.append((method, fam))
    return updated


def _git(*args, capture=True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        capture_output=capture, text=True)


def commit_and_push(updates: list[tuple[str, str]]) -> None:
    if not updates:
        return
    # Pull-rebase first to avoid push conflicts with concurrent commits.
    _git("pull", "--rebase", "--autostash")
    n_committed = 0
    for method, fam in updates:
        rel = f"math-paper/results/qprime_cache/{method}/{fam}"
        _git("add", f"{rel}.npz", f"{rel}.json")
        msg = f"qprime_cache: update {method}/{fam}"
        r = _git("commit", "-m", msg)
        if r.returncode == 0:
            n_committed += 1
            print(f"  {time.strftime('%H:%M:%S')} committed: {method}/{fam}",
                   flush=True)
        else:
            err = (r.stdout + r.stderr).strip()[:200]
            if "nothing to commit" not in err:
                print(f"  WARN: commit failed for {method}/{fam}: {err}",
                       flush=True)
    if n_committed:
        push = _git("push", "origin", "main")
        if push.returncode != 0:
            print(f"  WARN: push failed: "
                   f"{(push.stdout + push.stderr).strip()[:200]}", flush=True)
        else:
            print(f"  {time.strftime('%H:%M:%S')} pushed {n_committed} commit(s)",
                   flush=True)


def main() -> int:
    print(f"qprime_cache_watcher: polling {CACHE_ROOT} every {POLL_SECONDS}s",
           flush=True)
    print(f"  mirror -> {REPO_MIRROR}", flush=True)
    print(f"  tracking methods: {METHODS}", flush=True)
    while True:
        try:
            updates = scan_and_copy()
            commit_and_push(updates)
        except Exception as e:
            print(f"[watcher] error: {e}", flush=True)
        time.sleep(POLL_SECONDS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
