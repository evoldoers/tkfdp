#!/usr/bin/env python3
"""Patch the K=4 row in main.tex Table 1 from the replica-exchange
output JSON.

Reads:
  ~/tkf-dp/math-paper/results/infinite_phmm_balibase_k4_replicaexchange.json
  ~/tkf-dp/math-paper/main.tex

Replaces the placeholder row:
  inf-PHMM-$K{=}4$ (single-chain, legacy rates)\textsuperscript{$\ast$}
    & 0.454 &  4827 & 0.462 &  4953 & --- & --- & --- \\

with:
  inf-PHMM-$K{=}4$ (RE, 4-chain mean)\textsuperscript{$\ast$}
    & <F1> & <e_tp> & <F1_opt> & <e_tp_opt> & --- & --- & --- \\

and updates the table footnote with ESS / r-hat / swap-acc summary.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
MAIN_TEX = REPO / "math-paper" / "main.tex"
RE_JSON = REPO / "math-paper" / "results" / "infinite_phmm_balibase_k4_replicaexchange.json"


def pool_corpus(per_family):
    pp = []
    pp_opt = []
    for f in per_family:
        for r in f.get('per_pair', []):
            pp.append(r)
            if 'opt_acc_e_tp' in r and 'opt_acc_total_mass' in r:
                pp_opt.append({
                    'e_tp': r['opt_acc_e_tp'],
                    'total_mass': r['opt_acc_total_mass'],
                    'gold': r['gold'],
                })
    e_tp = sum(r['e_tp'] for r in pp)
    mass = sum(r['total_mass'] for r in pp)
    gold = sum(r['gold'] for r in pp)
    f1 = 2 * e_tp / (mass + gold) if (mass + gold) > 0 else float('nan')
    e_tp_opt = sum(r['e_tp'] for r in pp_opt)
    mass_opt = sum(r['total_mass'] for r in pp_opt)
    f1_opt = (2 * e_tp_opt / (mass_opt + gold)) if pp_opt and (mass_opt + gold) > 0 else float('nan')
    return f1, e_tp, f1_opt, e_tp_opt, len(pp)


def summarise_diagnostics(per_family):
    ess_n_match = []
    ess_log_pi = []
    swap_acc = []
    for f in per_family:
        for r in f.get('per_pair', []):
            diag = r.get('mcmc_diag') or {}
            for c in diag.get('per_chain', []):
                if c.get('ess_n_match') is not None:
                    ess_n_match.append(c['ess_n_match'])
                if c.get('ess_log_pi') is not None:
                    ess_log_pi.append(c['ess_log_pi'])
            for s in (diag.get('swap_acc_rates') or []):
                swap_acc.append(s)
    return {
        'ess_n_match_median': (float(np.median(ess_n_match))
                                 if ess_n_match else None),
        'ess_log_pi_median': (float(np.median(ess_log_pi))
                                if ess_log_pi else None),
        'swap_acc_median': (float(np.median(swap_acc))
                              if swap_acc else None),
    }


def main():
    if not RE_JSON.exists():
        print(f"ERR: {RE_JSON} not present; run TASK A step 6 first.",
              file=sys.stderr)
        sys.exit(1)
    d = json.loads(RE_JSON.read_text())
    if isinstance(d, list):
        per_family = d
    else:
        per_family = d.get('per_family', [])
    f1, e_tp, f1_opt, e_tp_opt, n_pairs = pool_corpus(per_family)
    diag = summarise_diagnostics(per_family)

    new_row = (
        f'inf-PHMM-$K{{=}}4$ (RE, 4-chain mean)\\textsuperscript{{$\\ast$}} '
        f'& {f1:.3f} & {e_tp: 5.0f} & {f1_opt:.3f} & {e_tp_opt: 5.0f} & '
        f'  --- &   --- &   --- \\\\'
    )
    print(f'New row: {new_row}')
    print(f'Diagnostics: {diag}')

    txt = MAIN_TEX.read_text()
    # Match the legacy single-chain row exactly.
    pattern = re.compile(
        r'inf-PHMM-\$K\{=\}4\$ \(single-chain, legacy rates\)\\textsuperscript'
        r'\{\$\\ast\$\}[^\\]*\\\\')
    new_txt, n = pattern.subn(new_row, txt, count=1)
    if n == 0:
        print('ERR: could not find the inf-PHMM row to patch.',
              file=sys.stderr)
        sys.exit(2)
    MAIN_TEX.write_text(new_txt)
    print(f'Patched {MAIN_TEX} (replaced {n} row).')
    print('Remember to update the footnote with the new ESS / r-hat / swap-acc summary.')


if __name__ == '__main__':
    main()
