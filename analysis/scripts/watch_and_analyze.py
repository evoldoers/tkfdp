"""Watch a dynfield training log; at each SWEEP BOUNDARY run dynfield_metrics on
the current checkpoint and append a merged row (log fields + computed metrics) to
a trajectory JSONL. Decoupled from training (no GPU, no coupling into the loop).

Matches full sweep lines '# sweep=  N corpus_ll=...' (not '#   v checkpoint:
...~mid'), so it fires once per completed sweep. Exits when --stop-file appears
or the log has been idle past --max-idle-sec after the target run is gone.

Usage:
  PYTHONPATH=src:$HOME/tkf-mixdom/python python analysis/scripts/watch_and_analyze.py \
    --log logs/<run>.log --ckpt results/<run>/_chkpt.npz \
    --out results/<run>/metrics_trajectory.jsonl
"""
from __future__ import annotations
import argparse, json, re, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dynfield_metrics as dm

SWEEP = re.compile(r"^# sweep=\s*(\d+) corpus_ll=([-\d.]+)")
def _grab(pat, line):
    m = re.search(pat, line)
    return float(m.group(1)) if m else None


def analyze_sweep(sweep, logline, args):
    ck = __import__("numpy").load(args.ckpt, allow_pickle=False)
    fam_ids = list(ck["family_ids"].tolist()); fam_set = set(fam_ids)
    arch = __import__("numpy").asarray(ck["arch_assignment"])
    disc = dm._discovered_pairs(ck)
    row = {
        "sweep": sweep, "ckpt_step": int(ck["step"]),
        "corpus_ll": _grab(r"corpus_ll=([-\d.]+)", logline),
        "n_pairs_log": _grab(r"n_pairs=(\d+)", logline),
        "t_sec": _grab(r"t=(\d+)s", logline),
        "flip_conf": _grab(r"flip\[conf\]=([\d.]+)", logline),
        "flip_other": _grab(r"flip\[other\]=([\d.]+)", logline),
        "n_pairs": sum(len(s) for s in disc.values()),
        "flip_recall": dm.flip_recall(disc, args.confirmed_flips, fam_set),
        "contact_enrichment": dm.contact_enrichment(disc, args.pdb_dir, fam_ids),
        "arch_charge_swap": dm.arch_charge_swap(arch, args.archetypes,
                                                n_perm=args.n_perm),
    }
    with open(args.out, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    e = row["contact_enrichment"]; fr = row["flip_recall"]; a = row["arch_charge_swap"]
    phi = ("" if row["flip_conf"] is None
           else f" flip[c/o]={row['flip_conf']}/{row['flip_other']}")
    print(f"[analyze] sweep {sweep}: pairs={row['n_pairs']} "
          f"recall={fr['discovered']}/{fr['confirmed_in_fams']} "
          f"enrich={e['fold']}x(p={e['poisson_p']:.1e}) "
          f"archX={a['charge_crossing_obs']}/{a['classes']}(p={a['p_ge_obs']:.2f}){phi}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pdb-dir", default="data/pdb_partition_clv_top1000_sifts")
    ap.add_argument("--confirmed-flips",
                    default="data/pdb_partition_clv_top1000_sifts/confirmed_flips.json")
    ap.add_argument("--archetypes", default="c20", choices=["c10", "c20"])
    ap.add_argument("--n-perm", type=int, default=20000)
    ap.add_argument("--poll-sec", type=int, default=60)
    ap.add_argument("--stop-file", default="")
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    # pick up any sweeps already logged (so a late-started watcher backfills)
    print(f"[analyze] watching {args.log} -> {args.out}", flush=True)
    while True:
        try:
            lines = Path(args.log).read_text(errors="ignore").splitlines()
        except FileNotFoundError:
            lines = []
        for ln in lines:
            m = SWEEP.match(ln)
            if not m:
                continue
            sw = int(m.group(1))
            if sw in seen:
                continue
            seen.add(sw)
            try:
                analyze_sweep(sw, ln, args)
            except Exception as e:                       # never die on one sweep
                print(f"[analyze] sweep {sw} failed: {e}", flush=True)
        if args.stop_file and Path(args.stop_file).exists():
            print("[analyze] stop-file present, exiting", flush=True)
            return
        time.sleep(args.poll_sec)


if __name__ == "__main__":
    main()
