#!/usr/bin/env python3
"""Coverage guard for the BAliBASE L<150 evaluation.

The failure mode this exists to stop is silent, not loud. A sweep that
loses families, or a downstream stitch that runs over a partially-synced
cache, still produces a well-formed result JSON whose aggregate is
numerically correct -- just over a smaller denominator. Commit 72055202
is the worked example: the same K=8 posteriors scored SP 0.7301 over 187
pairs and SP 0.6622 over 155, and nothing in the output distinguished the
two except a field nobody compared.

So the expected family set is written down as data
(``analysis/balibase_l150_coverage.json``, cross-checked against four
independent sources) and compared against, rather than being re-derived
and trusted each run.

Two entry points:

    assert_coverage(observed, label="sweep preflight")
        raise CoverageError unless `observed` matches the manifest exactly.

    check_coverage(observed) -> list[str]
        the same comparison, returning problems instead of raising.

`observed` maps family name -> pair count. Comparing against the stored
per-family counts rather than a bare total also catches the nastier case:
right total, wrong families.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO / "analysis" / "balibase_l150_coverage.json"


class CoverageError(RuntimeError):
    """Raised when observed coverage does not match the manifest."""


def load_manifest(path: str | Path | None = None) -> dict:
    """Load the expected-coverage manifest."""
    p = Path(path) if path is not None else DEFAULT_MANIFEST
    if not p.exists():
        raise CoverageError(
            f"coverage manifest not found at {p}. It is required; if you "
            f"are intentionally running a different subset, pass the "
            f"explicit opt-out flag rather than deleting the manifest.")
    with open(p) as fh:
        return json.load(fh)


def check_coverage(observed: dict[str, int],
                   manifest: dict | None = None,
                   *,
                   manifest_path: str | Path | None = None) -> list[str]:
    """Compare observed {family: n_pairs} against the manifest.

    Returns a list of human-readable problems; empty means exact match.
    """
    man = manifest if manifest is not None else load_manifest(manifest_path)
    expected = {k: v["n_pairs"] for k, v in man["families"].items()}

    problems: list[str] = []

    missing = sorted(set(expected) - set(observed))
    if missing:
        lost = sum(expected[f] for f in missing)
        problems.append(
            f"{len(missing)} expected famil{'y' if len(missing) == 1 else 'ies'} "
            f"absent ({lost} pairs): {', '.join(missing)}")

    extra = sorted(set(observed) - set(expected))
    if extra:
        problems.append(
            f"{len(extra)} unexpected famil{'y' if len(extra) == 1 else 'ies'} "
            f"present: {', '.join(extra)}")

    short = []
    for fam in sorted(set(expected) & set(observed)):
        if observed[fam] != expected[fam]:
            short.append(f"{fam} {observed[fam]}/{expected[fam]}")
    if short:
        problems.append(
            f"{len(short)} famil{'y' if len(short) == 1 else 'ies'} with wrong "
            f"pair count (got/expected): {', '.join(short)}")

    got_pairs = sum(observed.values())
    want_pairs = man["n_pairs"]
    if got_pairs != want_pairs or len(observed) != man["n_families"]:
        problems.append(
            f"totals: {len(observed)} families / {got_pairs} pairs, "
            f"expected {man['n_families']} / {want_pairs}")

    return problems


def assert_coverage(observed: dict[str, int],
                    manifest: dict | None = None,
                    *,
                    manifest_path: str | Path | None = None,
                    label: str = "coverage check") -> None:
    """Raise CoverageError unless `observed` matches the manifest exactly."""
    man = manifest if manifest is not None else load_manifest(manifest_path)
    problems = check_coverage(observed, man)
    if not problems:
        print(f"[{label}] OK: {man['n_families']} families / "
              f"{man['n_pairs']} pairs, matching "
              f"{DEFAULT_MANIFEST.relative_to(REPO)}", flush=True)
        return

    detail = "\n".join(f"  - {p}" for p in problems)
    raise CoverageError(
        f"[{label}] BAliBASE coverage does not match the expected set.\n"
        f"{detail}\n"
        f"Expected set: {man['n_families']} families / {man['n_pairs']} pairs "
        f"(max_len {man['max_len']}), from "
        f"{DEFAULT_MANIFEST.relative_to(REPO)}.\n"
        f"A partial run still produces a valid-looking aggregate over a "
        f"smaller denominator, which is why this is fatal rather than a "
        f"warning. Complete the run, or pass the explicit opt-out flag if a "
        f"different subset is genuinely intended.")


def observed_from_per_family(per_family: list[dict],
                             *,
                             family_key: str = "family",
                             pairs_key: str = "n_pairs") -> dict[str, int]:
    """Build the observed map from a result JSON's ``per_family`` list.

    Two result schemas are in circulation and they count pairs
    differently: the ``expected_pairwise_balibase`` outputs carry an
    explicit ``n_pairs``, while the ``downstream_fsa_on_cached_qprime``
    outputs carry a ``per_pair`` list instead. Prefer the explicit count,
    fall back to the list length, and refuse to guess beyond that -- a
    row we cannot count is reported as a mismatch rather than silently
    treated as zero or as complete.
    """
    out: dict[str, int] = {}
    for row in per_family:
        fam = row.get(family_key)
        if fam is None:
            continue
        if row.get(pairs_key) is not None:
            out[fam] = int(row[pairs_key])
        elif isinstance(row.get("per_pair"), list):
            out[fam] = len(row["per_pair"])
        else:
            raise CoverageError(
                f"cannot determine the pair count for family {fam}: the row "
                f"has neither '{pairs_key}' nor a 'per_pair' list. Counting "
                f"it as zero or as complete would both be guesses, so this "
                f"is fatal; teach observed_from_per_family the schema.")
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Check a result JSON's coverage against the manifest.")
    ap.add_argument("result_json", help="path to a result JSON with per_family")
    ap.add_argument("--manifest", default=None)
    args = ap.parse_args()

    with open(args.result_json) as fh:
        res = json.load(fh)
    obs = observed_from_per_family(res.get("per_family", []))
    try:
        assert_coverage(obs, manifest_path=args.manifest,
                        label=Path(args.result_json).name)
    except CoverageError as exc:
        print(exc)
        raise SystemExit(1)
