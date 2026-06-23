"""
Score a trained TKF-DP Potts H matrix against a fixed list of textbook
amino-acid interactions.

Each textbook entry has an expected sign:
  +  = favored / over-represented vs independent baseline
  -  = disfavored / under-represented

For a given checkpoint:
  log_enrich(x,y) = log p_joint(x,y) - log [pi(x) * pi(y)]
where p_joint is the model's implied amino-acid pair stationary obtained by
mixing exp(-H) by the trained empirical class proportions rho and the
per-class pi:
  p_xy = sum_{c,c'} rho_c * rho_c' * (pi_c * pi_c' * exp(-H)) / Z_{cc'}

A textbook pair PASSes if sign(log_enrich) matches the expected sign AND
|log_enrich| > THRESH (in nats). Otherwise FAIL (wrong sign) or neutral
(below threshold).

Usage:
    python3 analysis/scripts/textbook_score.py CHKPT_DIR [CHKPT_DIR ...]
    python3 analysis/scripts/textbook_score.py CHKPT_DIR --thresh 1.0 --verbose

Each CHKPT_DIR should contain state.npz with keys: potts_atoms, pi_class,
cls_N (for N=0,1,...) used to compute empirical class proportions.
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path

import numpy as np

AA = 'ACDEFGHIKLMNPQRSTVWY'
A = 20

# === FIXED TEXTBOOK INTERACTION LIST ===
# (residue_a, residue_b, expected_sign, category, note)
TEXTBOOK: list[tuple[str, str, int, str, str]] = [
    # Disulfide
    ('C', 'C', +1, 'disulfide', 'covalent S-S bond'),
    # Salt bridges (opposite charge)
    ('D', 'K', +1, 'salt_bridge', 'Asp-Lys'),
    ('D', 'R', +1, 'salt_bridge', 'Asp-Arg'),
    ('E', 'K', +1, 'salt_bridge', 'Glu-Lys'),
    ('E', 'R', +1, 'salt_bridge', 'Glu-Arg'),
    # Aromatic stacking (pi-pi)
    ('F', 'W', +1, 'aromatic_stack', 'pi-pi'),
    ('F', 'Y', +1, 'aromatic_stack', 'pi-pi'),
    ('W', 'Y', +1, 'aromatic_stack', 'pi-pi'),
    ('F', 'F', +1, 'aromatic_stack', 'aromatic self'),
    ('W', 'W', +1, 'aromatic_stack', 'aromatic self'),
    ('Y', 'Y', +1, 'aromatic_stack', 'aromatic self'),
    # Cation-pi (basic on aromatic)
    ('K', 'W', +1, 'cation_pi', 'Lys-Trp'),
    ('K', 'F', +1, 'cation_pi', 'Lys-Phe'),
    ('K', 'Y', +1, 'cation_pi', 'Lys-Tyr'),
    ('R', 'W', +1, 'cation_pi', 'Arg-Trp'),
    ('R', 'F', +1, 'cation_pi', 'Arg-Phe'),
    ('R', 'Y', +1, 'cation_pi', 'Arg-Tyr'),
    # Hydrophobic core
    ('I', 'V', +1, 'hydrophobic', 'aliphatic'),
    ('I', 'L', +1, 'hydrophobic', 'aliphatic'),
    ('L', 'V', +1, 'hydrophobic', 'aliphatic'),
    ('A', 'L', +1, 'hydrophobic', 'aliphatic'),
    ('A', 'V', +1, 'hydrophobic', 'aliphatic'),
    ('A', 'I', +1, 'hydrophobic', 'aliphatic'),
    ('M', 'L', +1, 'hydrophobic', 'aliphatic-Met'),
    # Polar H-bonds (sidechain-sidechain)
    ('N', 'D', +1, 'polar_hbond', 'amide-acid'),
    ('Q', 'E', +1, 'polar_hbond', 'amide-acid (long)'),
    ('S', 'T', +1, 'polar_hbond', 'hydroxyl pair'),
    ('N', 'S', +1, 'polar_hbond', 'amide-hydroxyl'),
    ('D', 'S', +1, 'polar_hbond', 'acid-hydroxyl'),
    ('N', 'T', +1, 'polar_hbond', 'amide-hydroxyl'),
    # Histidine metal coordination
    ('H', 'C', +1, 'metal_bind', 'Zn-finger / His-Cys'),
    ('H', 'D', +1, 'metal_bind', 'metal coordination'),
    ('H', 'H', +1, 'metal_bind', 'Zn-finger'),
    # Same-charge repulsion (disfavored)
    ('K', 'R', -1, 'same_charge', 'basic-basic'),
    ('D', 'E', -1, 'same_charge', 'acidic-acidic'),
    # Proline disrupts buried hydrophobics (disfavored as co-conserved pair)
    ('P', 'W', -1, 'pro_bulky', 'Pro disrupts'),
    ('P', 'L', -1, 'pro_bulky', 'Pro in helix bad'),
    ('P', 'I', -1, 'pro_bulky', 'Pro disrupts'),
    ('P', 'F', -1, 'pro_bulky', 'Pro disrupts'),
]


def load_model_enrich(state_path: Path) -> np.ndarray:
    """Compute the (A, A) log_enrich matrix for a given checkpoint state.npz."""
    data = np.load(str(state_path), allow_pickle=True)
    H = data['potts_atoms'][0]
    H = 0.5 * (H + H.T)
    pi_class = data['pi_class']
    K_c = pi_class.shape[0]
    # Prefer an explicitly stored class-prior rho of shape (K_c,) — that's
    # the supervised-fit convention. If rho has a different shape (the
    # unsupervised SVI checkpoint reuses the same field name for the
    # TSB stick-breaking weights), fall back to accumulating cls_N
    # partition assignments.
    rho = None
    if 'rho' in data.files:
        cand = np.asarray(data['rho'], dtype=np.float64)
        if cand.shape == (K_c,):
            rho = cand / cand.sum()
    if rho is None:
        total = np.zeros(K_c, dtype=np.int64)
        for k in data.files:
            if re.match(r'cls_\d+$', k):
                total += np.bincount(data[k].astype(int), minlength=K_c)
        if total.sum() == 0:
            rho = np.ones(K_c) / K_c
        else:
            rho = total / total.sum()
    expH = np.exp(-H)
    p_xy = np.zeros((A, A))
    for c in range(K_c):
        for cp in range(K_c):
            outer = np.outer(pi_class[c], pi_class[cp])
            unnorm = outer * expH
            p_xy += rho[c] * rho[cp] * unnorm / unnorm.sum()
    p_xy = 0.5 * (p_xy + p_xy.T)
    pm = p_xy.sum(axis=1)
    enrich = np.log(np.maximum(p_xy, 1e-30) / np.maximum(np.outer(pm, pm), 1e-30))
    return enrich


def score_textbook(enrich: np.ndarray, thresh: float = 0.3) -> dict:
    """Score the model against the textbook list at a given nat threshold."""
    cat_stats: dict[str, list[str]] = {}
    overall: list[str] = []
    rows: list[tuple] = []
    for a, b, exp_sign, cat, note in TEXTBOOK:
        i, j = AA.index(a), AA.index(b)
        le = float(enrich[i, j])
        if abs(le) < thresh:
            verdict = 'neutral'
        elif np.sign(le) == exp_sign:
            verdict = 'PASS'
        else:
            verdict = 'FAIL'
        rows.append((a, b, exp_sign, cat, note, le, verdict))
        overall.append(verdict)
        cat_stats.setdefault(cat, []).append(verdict)
    p = overall.count('PASS')
    f = overall.count('FAIL')
    n = overall.count('neutral')
    return dict(rows=rows, cat_stats=cat_stats,
                n_pass=p, n_fail=f, n_neutral=n,
                f_textbook=p / max(1, p + f + n))


def print_report(label: str, scored: dict, verbose: bool):
    if verbose:
        print(f'=== {label}  (THRESH applied) ===')
        print(f"{'Pair':<5} {'Exp':>3} {'log_enr':>8}  {'Verdict':<8}  {'Category':<18}  Note")
        print('-' * 80)
        for a, b, exp_sign, cat, note, le, verdict in scored['rows']:
            pair = a + b
            es = '+' if exp_sign > 0 else '-'
            print(f'{pair:<5} {es:>3} {le:>+8.2f}  {verdict:<8}  {cat:<18}  {note}')
        print()
        print(f"{'Category':<18}  {'PASS':>5} {'FAIL':>5} {'neutr':>5} {'total':>5}  {'%PASS':>6}")
        print('-' * 60)
        for cat in sorted(scored['cat_stats'].keys()):
            v = scored['cat_stats'][cat]
            pp = v.count('PASS'); ff = v.count('FAIL'); nn = v.count('neutral')
            tot = pp + ff + nn
            print(f'  {cat:<16}  {pp:>5} {ff:>5} {nn:>5} {tot:>5}  {100*pp/tot:>5.1f}%')
        print()
    print(f"{label:<40}  PASS={scored['n_pass']:>3}  FAIL={scored['n_fail']:>3}  "
          f"neut={scored['n_neutral']:>3}  f_textbook={scored['f_textbook']*100:.1f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('chkpts', nargs='+', type=Path,
                    help='One or more _best_chkpt/ directories.')
    ap.add_argument('--thresh', type=float, default=0.3,
                    help='|log_enrich| threshold in nats for non-neutral (default 0.3). '
                         'Try 1.0 for a stricter "e-fold or more" criterion.')
    ap.add_argument('--verbose', action='store_true',
                    help='Print full per-pair table and per-category breakdown.')
    args = ap.parse_args()

    for chkpt in args.chkpts:
        state_path = chkpt / 'state.npz' if chkpt.is_dir() else chkpt
        if not state_path.exists():
            print(f'SKIP {chkpt} (no state.npz)')
            continue
        enrich = load_model_enrich(state_path)
        scored = score_textbook(enrich, thresh=args.thresh)
        print_report(str(chkpt), scored, args.verbose)


if __name__ == '__main__':
    main()
