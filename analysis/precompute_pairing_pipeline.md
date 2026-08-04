# Precompute pairing pipeline (miz prefilter → exact peel → z-move lookup)

Goal: replace the discovery z-move's per-move GPU scoring with a **precomputed
pairing-evidence cache**, so the sampler does lookups + cheap prior terms. In
**enum400** the arch map is frozen, so the expensive cache is built **once** and
never invalidated — the entire discovery run then becomes table lookups.

No ELBO: for cap-2 the exact 2×20×20 peel (`exact_pair_ll_tree`) is affordable,
so we score pairs exactly. The de-Finetti ELBO is deferred to the m>2 side quest.

## Three stages

### Stage A — miz prefilter (class-free, one-time, never invalidated)

For each family, score every candidate column pair by the **bias-corrected MI
z-score** and keep a shortlist. This is computed from raw leaf residues only — no
model, no field, no classes — so it is arch/class-independent and never needs
recomputing.

- Input: `data/pfam_processed_clv_top1000_thin128/<fam>.npz` → `leaf_msa`.
- Metric: `miz(i,j)` = (MI − mean_null)/std_null. Corpus-wide use the **analytic
  bias correction (APC or Miller–Madow)**, O(1)/pair (~60 s corpus-wide); refine
  the top pairs with the permutation z (200 shuffles) if wanted.
- Shortlist rule: per family keep pairs with `miz > z*` (start z*≈3) OR top-K per
  family; union with any PDB-contact pairs. Expected size ~10²–10³ pairs/family,
  ~10⁵ corpus-wide (vs ~6×10⁶ all-C(L,2)).
- Output: `data/pairing_cache/<fam>.miz.npz` — arrays `(i, j, miz)` for the
  shortlist.
- Reuses: the miz machinery already in `experiments/build_pdb_partition.py`
  (generalize from PDB pairs to all candidate pairs).

### Stage B — exact pair-evidence cache (arch-dependent; one-time in enum400)

For each shortlisted pair, compute the **exact class-marginalized** pairing
evidence and store the pieces DM-reweighting needs.

Per pair (i,j):
1. Rank top-N=12 classes per column by singleton peel (cheap, batched); the
   convergence check showed N=12 reproduces the full 400-class sum to <0.1 nat.
2. For each of the 12² class combos (c_i,c_j): `exact_pair_ll_tree` at each
   field-rate bin → rate-marginalize → `LL_pair[c_i,c_j]` (12×12).
3. Singleton marginals `logZ_s(i)`, `logZ_s(j)` over all 400 classes (cheap).

Store per pair: `{top12_i, top12_j, LL_pair[12,12], logZ_s_i, logZ_s_j}`. Keep
`LL_pair` **unsummed over classes** so the DM reweight is applied at lookup time.

- Output: `data/pairing_cache/<fam>.pairev.npz`.
- Cost: ~10⁵ pairs × 144 combos × 4 rate bins exact peels. Numpy ~hours; the
  800-state peel is O(n_nodes·A³) per call and **batches on GPU** over
  pairs×classes×rates (the "highly batched precompute") → minutes.
- Reuses: `scratchpad/ll_classmarg.py` (already does exactly this per pair) —
  promote to a batched precompute script, add a shared expm(Q_arch,τ) table.
- **Invalidation:** depends only on `arch_assignment`, rho, rho_chain, rate
  bins. In enum400 arch is frozen → build once. If arch is later *learned*, an
  arch move on class c invalidates only pairs whose top-12 touches archetype c →
  partial refresh, not a full rebuild.

### Stage C — z-move integration (lookup + DM + Ewens)

Replace `_score_specs_rates` in `field_rate_discovery.z_move` with a cache
lookup for shortlisted pairs:

- Merge log-odds for pairing s with partner t:
  `logZ_pair(s,t) − logZ_s(s) − logZ_s(t) − log(alpha_z)`
  where `logZ_pair = logsumexp_{c_i,c_j}( LL_pair[c_i,c_j] + logP_DM(c_i,c_j|h) )`
  — the DM reweight over the cached 12×12 grid (cheap, recomputed as DM updates).
- Pairs **not** in the shortlist are non-candidates (miz below threshold ⇒ treat
  as won't-pair). This bounds the z-move candidate set to the shortlist.
- cn-move / singletons: the singleton peels are cached over all 400 classes, so
  cn is also a lookup; for a *paired* column restrict cn to the cached top-12
  (captures the mass) with a live-peel fallback on the rare out-of-grid draw.

Net: no per-move tree forward; the sampler is lookups + DM reweight + Ewens.

## Storage / cost summary

| stage | depends on | frequency | size | cost |
|---|---|---|---|---|
| A miz | raw leaves | once, never invalid | ~10⁵ pairs | ~min (analytic) |
| B pair-ev | arch (frozen in enum400) | once in enum400 | 10⁵ × (144 f64 + idx) ≈ 100s MB | GPU minutes |
| C lookup | DM (reweight only) | per move | — | O(144) per move |

## Validation plan (each stage gated)

1. Stage A: confirmed-flip pairs must survive the shortlist at z*≈3 (they were
   selected by miz, so this is a sanity floor); report shortlist recall of the
   confirmed flips and shortlist size.
2. Stage B: cache `LL_pair` must match a live `ll_classmarg` call to ~1e-6 on a
   sample of pairs; class-marg log-odds must match the earlier direct
   computation on the confirmed flips.
3. Stage C: run the cached z-move vs the current `_score_specs_rates` z-move on a
   small corpus (e.g. 20 families, 1 sweep) — the pairing decisions/log-odds
   must agree within the mm-vs-exact gap (~0.1 nat) except where exact is
   deliberately more accurate. Behind a `--cached-zmove` flag; default off until
   validated.

## Build order (reuse vs new)

1. **[new]** `experiments/precompute_pairing.py`: Stage A (analytic miz over all
   pairs) → shortlist; Stage B (batched exact peel over shortlist × top-12²) →
   `data/pairing_cache/`. Reuses `ll_classmarg` + a shared expm table; JAX/GPU
   batch the peel.
2. **[new]** cache loader + DM-reweight lookup in a small module.
3. **[modify]** `field_rate_discovery.z_move`: `--cached-zmove` path that reads
   the cache instead of scoring live; keep the live path as the default/fallback.
4. Validate (gate 3), then flip the default for enum400 runs.

## Open questions / risks

- **Shortlist recall vs size** — z*≈3 is a guess; sweep it against confirmed-flip
  recall to pick the operating point. A too-tight prefilter silently drops real
  pairs (log it — no silent caps).
- **cn out-of-grid draws** — how often does cn want a class outside the cached
  top-12 for a paired column? If rare, the fallback is cheap; if common, widen N.
- **Non-enum400 (learned arch)** — the partial-refresh story needs the
  pair→archetype dependency index; fine for enum400 now, design later.
- The pairing score becomes **exact** (not the +0.089-nat mm forward), so expect
  small systematic shifts vs the current sampler — a feature, but note it when
  comparing runs.
