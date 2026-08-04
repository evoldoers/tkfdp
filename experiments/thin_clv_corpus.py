"""Thin each family's tree to at most --max-leaves leaves and re-peel a new
CLV corpus, to bound pad_N (and the JIT shape / tau-bin explosion) on a
large-tree corpus. Mirrors the setup used on lambda.biowiki.org's smaller GPU.

Design decisions that matter for the supervised coevolution run:

  * ROWS ONLY, never columns. Thinning removes leaves (sequences); it keeps
    EVERY alignment column, so L is unchanged and the PDB size-2 partition
    (build_pdb_partition.py, indexed by raw alignment column) still maps onto
    the thinned corpus 1:1.

  * DIVERSITY-PRESERVING selection, not random. We keep the --max-leaves most
    phylogenetically-spread leaves via farthest-point sampling on the tree
    metric (greedily add the leaf whose minimum patristic distance to the
    already-chosen set is largest). Divergent lineages are exactly the ones
    that carry the substitution / acid-base-flip variation we want to model,
    so max-min spread preserves covariation far better than uniform down-
    sampling.

  * SIGNAL CHECK. With --partition-dir we measure, per contact pair, how much
    of the column-pair covariation survives thinning (joint coverage + number
    of distinct joint residue states + acid<->base flip arrangements), full
    vs thinned, and write it to <out-dir>/retention.json. Guards against
    silently thinning the signal away.

Output: same schema as preprocess_pfam_pswm.py (family, L, n_nodes, n_leaves,
root_id, parent, tau, clv, log_scale, log_p_lg_per_site, leaf_msa) so the
phylo-ELBO trainer consumes it unchanged via --clv-dir.

Usage:
    python experiments/thin_clv_corpus.py \
        --src-clv-dir data/pfam_processed_clv_top1000 \
        --out-dir data/pfam_processed_clv_top1000_thin128 \
        --max-leaves 128 \
        --partition-dir data/pdb_partition_clv_top1000 \
        --n-workers 8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tkfdp.bio import Node, load_family  # noqa: E402
from tkfdp.lg08 import ALPHA_ORDER  # noqa: E402
from tkfdp.pswm_peeling import compute_pswm_family  # noqa: E402

ALPHA = "".join(list(ALPHA_ORDER))
ACIDIC, BASIC = set("DE"), set("KRH")


# ------------------------------------------------------- tree thinning

def _leaves(root: Node) -> 'list[Node]':
    out = []
    stack = [root]
    while stack:
        v = stack.pop()
        if v.is_leaf():
            out.append(v)
        else:
            stack.extend(v.children)
    return out


def _dist_from_source(src: Node, root: Node) -> 'dict[int, float]':
    """Patristic distance from src to every node, over the unrooted tree
    (edges weighted by child branch_length). Returns {id(node): dist}."""
    # adjacency: each node <-> parent with weight = node.branch_length
    adj: 'dict[int, list[tuple[Node, float]]]' = {}
    stack = [root]
    nodes = []
    while stack:
        v = stack.pop()
        nodes.append(v)
        adj.setdefault(id(v), [])
        for c in v.children:
            w = float(c.branch_length or 0.0)
            adj.setdefault(id(c), []).append((v, w))
            adj[id(v)].append((c, w))
            stack.append(c)
    id_to_node = {id(v): v for v in nodes}
    dist = {id(src): 0.0}
    # Dijkstra-free: tree is acyclic, so a simple stack relaxation works.
    order = [src]
    seen = {id(src)}
    while order:
        u = order.pop()
        du = dist[id(u)]
        for nb, w in adj[id(u)]:
            if id(nb) not in seen:
                seen.add(id(nb))
                dist[id(nb)] = du + w
                order.append(nb)
    return dist


def select_diverse_leaves(root: Node, k: int) -> 'list[Node]':
    """Farthest-point sampling on the tree metric: return up to k leaves that
    maximise the minimum pairwise patristic distance (greedy max-min)."""
    leaves = _leaves(root)
    n = len(leaves)
    if n <= k:
        return leaves
    # Seed with the leaf farthest from an arbitrary leaf (a good diameter
    # endpoint), then greedily add farthest-from-selected.
    d0 = _dist_from_source(leaves[0], root)
    first = max(leaves, key=lambda lv: d0.get(id(lv), 0.0))
    selected = [first]
    min_dist = {id(lv): _dist_from_source(first, root).get(id(lv), 0.0)
                for lv in leaves}
    while len(selected) < k:
        nxt = max(leaves, key=lambda lv: min_dist[id(lv)]
                  if id(lv) not in {id(s) for s in selected} else -1.0)
        selected.append(nxt)
        dn = _dist_from_source(nxt, root)
        for lv in leaves:
            dv = dn.get(id(lv), 0.0)
            if dv < min_dist[id(lv)]:
                min_dist[id(lv)] = dv
    return selected


def prune_to_leaves(root: Node, keep: 'set[int]') -> 'Node | None':
    """Return a pruned copy retaining only leaves whose id() is in `keep`;
    collapse resulting degree-1 internal nodes, summing branch lengths."""
    if root.is_leaf():
        if id(root) in keep:
            c = Node(root.name, root.branch_length)
            return c
        return None
    kept = [prune_to_leaves(c, keep) for c in root.children]
    kept = [c for c in kept if c is not None]
    if not kept:
        return None
    if len(kept) == 1:
        child = kept[0]
        child.branch_length = (child.branch_length or 0.0) + (root.branch_length or 0.0)
        return child
    new = Node(root.name, root.branch_length)
    for c in kept:
        c.parent = new
    new.children = kept
    return new


# ------------------------------------------------------- covariation check

def _pair_stats(msa_rows: np.ndarray, i: int, j: int) -> dict:
    """Coverage + joint diversity + acid/base flip arrangement counts for a
    column pair over the given leaf rows (gap = 20)."""
    a, b = msa_rows[:, i], msa_rows[:, j]
    both = (a < 20) & (b < 20)
    n = int(both.sum())
    if n == 0:
        return {"n": 0, "n_joint_states": 0, "ab": 0, "ba": 0}
    aa = a[both]; bb = b[both]
    joint = set(zip(aa.tolist(), bb.tolist()))
    ab = ba = 0
    for x, y in joint:
        cx, cy = ALPHA[x], ALPHA[y]
        if cx in ACIDIC and cy in BASIC:
            ab += 1
        elif cx in BASIC and cy in ACIDIC:
            ba += 1
    return {"n": n, "n_joint_states": len(joint), "ab": ab, "ba": ba}


# ------------------------------------------------------- per-family worker

def thin_family(fam: str, out_dir: Path, max_leaves: int,
                partition_dir: 'Path | None') -> dict:
    try:
        fd = load_family(fam)
    except Exception as e:
        return {"family": fam, "ok": False, "err": f"load: {e!r}"}
    if fd.msa.size == 0 or fd.tree is None:
        return {"family": fam, "ok": False, "err": "empty msa/tree"}
    L = int(fd.msa.shape[1])
    full_leaves = _leaves(fd.tree)
    n_full = len(full_leaves)

    if n_full > max_leaves:
        sel = select_diverse_leaves(fd.tree, max_leaves)
        keep_ids = {id(lv) for lv in sel}
        pruned = prune_to_leaves(fd.tree, keep_ids)
    else:
        sel = full_leaves
        pruned = fd.tree

    try:
        peel = compute_pswm_family(fd.msa, fd.names, pruned)
    except Exception as e:
        import traceback
        return {"family": fam, "ok": False,
                "err": f"peel: {e!r}\n{traceback.format_exc()}"}

    n_leaves = peel["n_leaves"]
    leaf_msa = np.full((n_leaves, L), 20, dtype=np.int8)
    for r in range(n_leaves):
        row = int(peel["leaf_msa_row"][r])
        if row >= 0:
            leaf_msa[r] = fd.msa[row].astype(np.int8)

    np.savez(out_dir / f"{fam}.npz",
             family=fam, L=np.int32(L),
             n_nodes=np.int64(peel["n_nodes"]),
             n_leaves=np.int64(n_leaves),
             root_id=np.int64(peel["root_id"]),
             parent=peel["parent"].astype(np.int32),
             tau=peel["tau"].astype(np.float64),
             clv=peel["clv"].astype(np.float32),
             log_scale=peel["log_scale"].astype(np.float32),
             log_p_lg_per_site=peel["log_p_lg_per_site"].astype(np.float64),
             leaf_msa=leaf_msa)

    result = {"family": fam, "ok": True, "L": L,
              "n_leaves_full": n_full, "n_leaves_thin": int(n_leaves)}

    # Signal-retention check against the supervised partition.
    if partition_dir is not None:
        pf = partition_dir / f"{fam}.npz"
        if pf.exists():
            pz = np.load(pf, allow_pickle=True)
            if int(pz["L"]) == L:
                # Fair comparison: full = rows of the ORIGINAL tree's leaves
                # (the untrimmed training data), thin = retained leaf rows.
                name_to_row = {nm: i for i, nm in enumerate(fd.names)}
                full_rows = [name_to_row[lv.name] for lv in full_leaves
                             if lv.name in name_to_row]
                full_msa = (fd.msa[np.asarray(full_rows, dtype=np.int64)]
                            if full_rows else fd.msa)
                thin_msa = leaf_msa
                pairs = np.asarray(pz["pairs"], np.int32).reshape(-1, 2)
                kinds = (np.asarray(pz["kind"]).tolist()
                         if "kind" in pz.files else ["nn"] * len(pairs))
                per = []
                for (i, j), k in zip(pairs, kinds):
                    fu = _pair_stats(full_msa, int(i), int(j))
                    th = _pair_stats(thin_msa, int(i), int(j))
                    per.append({"i": int(i), "j": int(j), "kind": k,
                                "full": fu, "thin": th})
                result["pairs"] = per
    return result


# ------------------------------------------------------- driver

def _worker(args):
    fam, out_dir, max_leaves, partition_dir = args
    return thin_family(fam, Path(out_dir), max_leaves,
                       Path(partition_dir) if partition_dir else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src-clv-dir", default="data/pfam_processed_clv_top1000",
                    help="Defines the family set (uses its index.json).")
    ap.add_argument("--out-dir", default="data/pfam_processed_clv_top1000_thin128")
    ap.add_argument("--max-leaves", type=int, default=128)
    ap.add_argument("--partition-dir", default="data/pdb_partition_clv_top1000",
                    help="For the covariation-retention check ('' to skip).")
    ap.add_argument("--families", default="",
                    help="Comma-separated subset (default: all in src corpus).")
    ap.add_argument("--n-workers", type=int, default=8)
    args = ap.parse_args()

    src = Path(args.src_clv_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = json.loads((src / "index.json").read_text())
    if args.families:
        families = [f.strip() for f in args.families.split(",") if f.strip()]
    else:
        families = [f for f in index["families"] if (src / f"{f}.npz").exists()]
    print(f"# thinning {len(families)} families to <= {args.max_leaves} "
          f"leaves -> {out_dir}", flush=True)

    tasks = [(f, str(out_dir), args.max_leaves, args.partition_dir or "")
             for f in families]
    results = []
    if args.n_workers > 1:
        import multiprocessing as mp
        with mp.Pool(args.n_workers) as pool:
            for i, r in enumerate(pool.imap_unordered(_worker, tasks)):
                results.append(r)
                if (i + 1) % 100 == 0:
                    ok = sum(1 for x in results if x.get("ok"))
                    print(f"  [{i+1}/{len(tasks)}] ok={ok}", flush=True)
    else:
        for i, t in enumerate(tasks):
            results.append(_worker(t))
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(tasks)}]", flush=True)

    ok = [r for r in results if r.get("ok")]
    fam_L = {r["family"]: r["L"] for r in ok}
    (out_dir / "index.json").write_text(json.dumps({
        "families": sorted(fam_L),
        "L": {k: fam_L[k] for k in sorted(fam_L)},
        "src_clv_dir": str(src), "max_leaves": args.max_leaves,
        "n_families": len(ok),
    }, indent=2))

    # Aggregate retention report.
    thinned = [r for r in ok if r["n_leaves_full"] > r["n_leaves_thin"]]
    print(f"# {len(ok)} families thinned/copied; "
          f"{len(thinned)} actually thinned "
          f"(median full leaves "
          f"{int(np.median([r['n_leaves_full'] for r in ok]))} -> "
          f"{int(np.median([r['n_leaves_thin'] for r in ok]))})")
    _write_retention(out_dir, ok)
    errs = [r for r in results if not r.get("ok")]
    if errs:
        print(f"# {len(errs)} failures, e.g. {errs[0]['family']}: "
              f"{errs[0]['err'][:120]}")
    print(f"# wrote {out_dir}")


def _write_retention(out_dir: Path, ok: 'list[dict]'):
    """Summarise covariation retention across all partition pairs, by kind."""
    by_kind: 'dict[str, dict]' = {}
    detail = []
    for r in ok:
        for p in r.get("pairs", []):
            k = p["kind"]
            fu, th = p["full"], p["thin"]
            if fu["n"] == 0:
                continue
            cov = th["n"] / fu["n"]
            js = (th["n_joint_states"] / fu["n_joint_states"]
                  if fu["n_joint_states"] else 1.0)
            flip_full = fu["ab"] > 0 and fu["ba"] > 0
            flip_thin = th["ab"] > 0 and th["ba"] > 0
            d = by_kind.setdefault(k, {"n": 0, "cov": [], "js": [],
                                       "flip_full": 0, "flip_kept": 0})
            d["n"] += 1
            d["cov"].append(cov); d["js"].append(js)
            if flip_full:
                d["flip_full"] += 1
                if flip_thin:
                    d["flip_kept"] += 1
            detail.append({"family": r["family"], **p,
                           "cov": cov, "js_ratio": js,
                           "flip_full": flip_full, "flip_thin": flip_thin})
    summary = {}
    print("# --- covariation retention (thinned / full) by contact kind ---")
    for k, d in sorted(by_kind.items()):
        med_cov = float(np.median(d["cov"])) if d["cov"] else 0.0
        med_js = float(np.median(d["js"])) if d["js"] else 0.0
        flip_ret = (d["flip_kept"] / d["flip_full"]) if d["flip_full"] else None
        summary[k] = {"n_pairs": d["n"], "median_coverage": med_cov,
                      "median_joint_state_ratio": med_js,
                      "n_pairs_with_flip_full": d["flip_full"],
                      "flip_retention": flip_ret}
        fr = f"{flip_ret:.2f}" if flip_ret is not None else "n/a"
        print(f"#   {k:10s} n={d['n']:5d}  med_cov={med_cov:.2f}  "
              f"med_joint_states={med_js:.2f}  "
              f"flip_retained={fr} ({d['flip_kept']}/{d['flip_full']})")
    (out_dir / "retention.json").write_text(json.dumps(
        {"by_kind": summary, "detail": detail}, indent=2))


if __name__ == "__main__":
    main()
