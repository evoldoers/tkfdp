#!/usr/bin/env python3
"""Update the K=4 RE row in math-paper/main.tex Table 1 with the latest
corpus statistics from the cached Q' arrays + downstream FSA results.

Inputs:
  - Sampler outputs (any of):
      math-paper/results/infinite_phmm_balibase_k4_replicaexchange.json
      math-paper/results/infinite_phmm_balibase_k4_replicaexchange_gpu0.json
      math-paper/results/infinite_phmm_balibase_k4_replicaexchange_gpu1.json
    For per-pair Q' suff stats (e_tp, total_mass, gold).
  - Downstream FSA:
      math-paper/results/k4_re_downstream_fsa.json
    For corpus opt-acc F1, MSA F1 / SP / TC at gap_factor=1.

Reports F1[gap=1] only (not gap=0) -- per user, the two were near-identical
and we have collapsed to the single default column.

Patches the K=4 row of Table 1 in main.tex and the corresponding caption.
Idempotent: safe to re-run as more families complete.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MAIN_TEX = REPO / "math-paper" / "main.tex"
DOWNSTREAM_JSON = REPO / "math-paper" / "results" / "k4_re_downstream_fsa.json"


def pool_qprime_from_jsons(json_paths):
    """Pool per-pair Q' suff stats across one or more sampler-output JSONs.
    Deduplicate by (family, name_i, name_j)."""
    seen = set()
    e_tp = total_mass = gold = 0.0
    n_pairs = 0
    fam_set = set()
    for p in json_paths:
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        for fam in d.get('per_family', []):
            fname = fam.get('family')
            for r in fam.get('per_pair', []):
                key = (fname, r.get('name_i'), r.get('name_j'))
                if key in seen:
                    continue
                seen.add(key)
                e_tp += r.get('e_tp', 0)
                total_mass += r.get('total_mass', 0)
                gold += r.get('gold', 0)
                n_pairs += 1
                fam_set.add(fname)
    f1 = (2 * e_tp / (total_mass + gold)) if (total_mass + gold) > 0 else 0.0
    return {
        'F1': f1, 'e_tp': e_tp, 'total_mass': total_mass, 'gold': gold,
        'n_pairs': n_pairs, 'n_families': len(fam_set),
        'families': sorted(fam_set),
    }


def parse_downstream(path: Path):
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    fam_records = d.get('per_family', [])
    # Pool opt-acc + MSA F1 (gap=1) across families.
    opt_etp = opt_mass = gold = 0.0
    msa_f1_sum = msa_sp_sum = msa_tc_sum = 0.0
    n_msa = 0
    for fr in fam_records:
        for pr in fr.get('per_pair', []):
            opt_etp += pr.get('opt_acc_e_tp', 0)
            opt_mass += pr.get('opt_acc_total_mass', 0)
            gold += pr.get('gold', 0)
        if 'msa_f1_g1' in fr:
            msa_f1_sum += fr['msa_f1_g1']
            n_msa += 1
        if 'msa_sp_g1' in fr:
            msa_sp_sum += fr['msa_sp_g1']
        if 'msa_tc_g1' in fr:
            msa_tc_sum += fr['msa_tc_g1']
    f1_opt = (2 * opt_etp / (opt_mass + gold)) if (opt_mass + gold) > 0 else 0.0
    return {
        'opt_F1': f1_opt, 'opt_e_tp': opt_etp,
        'msa_F1': msa_f1_sum / max(1, n_msa),
        'msa_SP': msa_sp_sum / max(1, n_msa),
        'msa_TC': msa_tc_sum / max(1, n_msa),
        'n_families': n_msa,
    }


def fmt_num(x, sig=3):
    if x == 0 or x is None:
        return "---"
    return f"{x:.{sig}f}"


def patch_main_tex(qpool, dpool):
    text = MAIN_TEX.read_text()

    # The K=4 row marker is "inf-PHMM-$K{=}4$" near the bottom of Table 1.
    # We rewrite the WHOLE row (single line of \\-terminated values).
    # Old patterns to match (any of):
    #   inf-PHMM-$K{=}4$ % (single-chain, legacy rates)\textsuperscript{$\ast$}
    #    & 0.454 &  4827 & 0.462 &  4953 &   --- &   --- &   --- \\
    #   inf-PHMM-$K{=}4$ (RE, ...) & ... \\

    f1 = fmt_num(qpool['F1'])
    etp = f"{int(round(qpool['e_tp']))}"
    if dpool is not None:
        f1_opt = fmt_num(dpool['opt_F1'])
        etp_opt = f"{int(round(dpool['opt_e_tp']))}"
        f1_msa = fmt_num(dpool['msa_F1'])
        sp = fmt_num(dpool['msa_SP'])
        tc = fmt_num(dpool['msa_TC'])
    else:
        f1_opt = etp_opt = f1_msa = sp = tc = "---"
    n_fams = qpool['n_families']

    new_row = (f"inf-PHMM-$K{{=}}4$ (RE, {n_fams}/22 fams cached)"
               f"\\textsuperscript{{$\\ast$}}\n"
               f" & {f1} & {etp} & {f1_opt} & {etp_opt} & "
               f"{f1_msa} & {sp} & {tc} \\\\")

    # Match any existing inf-PHMM-$K{=}4$ row + the next \\ line.
    pattern = re.compile(
        r"inf-PHMM-\$K\{=\}4\$.*?\\\\",
        re.DOTALL)
    if pattern.search(text):
        # Use a lambda replacement so backslashes in new_row are not
        # interpreted as regex back-references (\1, \2, ...).
        text2 = pattern.sub(lambda _m: new_row, text, count=1)
        if text2 != text:
            MAIN_TEX.write_text(text2)
            print(f"main.tex K=4 row updated: F1={f1} eTP={etp} "
                  f"opt_F1={f1_opt} MSA_F1={f1_msa} SP={sp} TC={tc} "
                  f"({n_fams}/22 fams)")
            return True
        else:
            print(f"main.tex K=4 row already up-to-date "
                  f"({n_fams}/22 fams)")
            return True
    print("WARN: no K=4 row found in main.tex to patch")
    return False


def main():
    json_paths = [
        REPO / "math-paper" / "results"
        / "infinite_phmm_balibase_k4_replicaexchange.json",
        REPO / "math-paper" / "results"
        / "infinite_phmm_balibase_k4_replicaexchange_gpu0.json",
        REPO / "math-paper" / "results"
        / "infinite_phmm_balibase_k4_replicaexchange_gpu1.json",
    ]
    qpool = pool_qprime_from_jsons(json_paths)
    print(f"Q' pool: {qpool['n_pairs']} pairs across {qpool['n_families']} "
          f"families, F1 = {qpool['F1']:.4f}")
    dpool = parse_downstream(DOWNSTREAM_JSON)
    if dpool is not None:
        print(f"Downstream FSA: opt-F1 = {dpool['opt_F1']:.4f}, "
              f"MSA F1 = {dpool['msa_F1']:.4f}, "
              f"SP = {dpool['msa_SP']:.4f}, TC = {dpool['msa_TC']:.4f} "
              f"({dpool['n_families']} fams)")
    else:
        print(f"Downstream FSA results not yet available "
              f"({DOWNSTREAM_JSON})")
    patch_main_tex(qpool, dpool)
    return 0


if __name__ == "__main__":
    sys.exit(main())
