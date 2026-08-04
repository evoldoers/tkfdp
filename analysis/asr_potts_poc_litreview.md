# Potts + phylogenetics ASR: proof-of-concept candidates (literature review)

> Generated 2026-07-31 by a multi-agent literature-review workflow (18 agents; 5-angle web sweep → per-candidate assessment → synthesis). Citations are agent-gathered from live web search and should be verified before use in a paper.

Confirmed the load-bearing codebase facts. Writing the briefing now.

---

# Briefing: an "easy" POC where coupling-aware ASR should beat independent-site Felsenstein

## Bottom line

**Go to RNA. Simulate first, then real Rfam stems.** Our protein null was not a bug in the decoder — it was the data: per-contact coupling ~0.1 nat (~0.14 bit), diffuse across residue-pair combinations, and washed out by dominant single-site phylogenetic signal. The Weigt group reached the same conclusion independently (Zeinaty et al. 2026; Rodríguez-Horta & Weigt 2021), and the Thornton lab stated it as a general result (Muniz-Trejo, Park & Thornton 2025, *MBE* msaf084: "no apparent need for ASR models to explicitly incorporate epistatic interactions" on real protein alignments). Do **not** expect any real-protein-family win.

RNA Watson–Crick base pairs are the opposite regime on both axes that matter: coupling is **~1–2 bits per covarying pair (5–15× our protein number) and near-deterministic/low-dimensional** (of 16 doublets only ~6 carry mass), and it lives in the **stationary** distribution where our machinery can see it cleanly, not only in dynamics. A base pair is a size-2 cluster with a 16-state doublet emission — a near-verbatim fit to our reversible paired-CTMC-on-a-tree.

**Ranked shortlist**

| Rank | Candidate | Verdict | One-line reason |
|---|---|---|---|
| **1** | **Simulated 16-state doublet CTMC on a fixed RNA tree** | **strong POC — do first** | Own the coupling strength *and* the branch lengths; exact ground-truth ancestors; guaranteed, tunable win that validates the decoder before touching biology |
| **2** | **Real RNA stems: 5S rRNA (RF00001) / single-isotype tRNA / RNase P (RF00010)** | **strong POC** | SS_cons hands you the pairing partition for free; ~1–2 bit coupling, R-scape-validated; deep divergence gives real ancestral uncertainty |
| 3 | Lattice-protein / synthetic Potts-on-tree with tunable β·J | strong POC (control) | Exact known Hamiltonian + true posterior; maps the (coupling × divergence) phase boundary; robustness check that a win isn't RNA-specific |
| 4 | HK–RR two-component interface | plausible | Strongest *concentrated* protein coupling, but inter-protein (needs paralog matching) and framed on interface columns only |
| — | Protein bmDCA forward-sim (β-lactamase/PF00072) | **trap** | Self-consistency + our reversible generator sits in De Leonardis's ~2% Potts-evolver regime → risks reproducing the null |
| — | Netti–Weigt evolutionary-intermediate landscapes | **trap** | It *is* our failing protein landscape; the "coupling helps" claim is ensemble faithfulness, **not** point accuracy |
| — | Structure/stability (Williams–Goldstein) evolver | **trap** | Many-body threshold, not pairwise; the one headroom (over-stability) is already closed by independent-site *Bayesian* sampling |
| — | Steroid-receptor GR/MR resurrection | **trap** | No independent ground-truth ancestor (ancestors are themselves ML-ASR); epistasis is higher-order. Cite for motivation only |
| — | Viral RNA (HIV RRE / HCV IRES) | **trap for now** | Recombination breaks the single-tree CTMC assumption; run only as a late stress-test |
| — | Salt bridges / our own PDB flips | **trap (negative control)** | This is our documented null (~0.003–0.004 nat/count); keep it as the "not enough coupling" anchor |

---

## The win condition (why RNA flips the null)

Coupling-aware ASR beats independent-site reconstruction at a node **iff both**: (a) the per-site marginal posterior there is **diffuse** — branch lengths long enough that a leaf residue doesn't by itself pin the ancestor — **and** (b) the pairwise mutual information **exceeds** that per-site information deficit, so the partner site supplies what the marginal lost. Proteins fail (a)+(b) simultaneously. RNA supplies (b) intrinsically (WC complementarity) and lets you tune (a) with branch length. The failure mode therefore *inverts*: the risk is no longer weak coupling, it is **insufficient phylogenetic uncertainty** (canonical stems are conserved), which is why the effect must be scored **conditional on a compensatory event having occurred**, not as a whole-molecule average.

---

## TOP RECOMMENDATION — RNA base-pair doublet CTMC ASR

### Stage 0 (mandatory, first): controlled simulation

This is the airtight POC and the gate the real data is judged against. It answers the one thing the protein experiment could not: *does our decoder exploit coupling when coupling is unambiguously present?*

- **Generator:** a reversible 16-state doublet rate matrix. Two options: (i) the empirical **RNA16A / RNA16C** matrices of Smith, Lui & Tillier 2004 (*MBE* 21:419), which have a **nonzero double-substitution rate** — matrices at `https://wwwlabs.uhnresearch.ca/labs/tillier/rRNA/rna.html`; or (ii) construct a Savill–Hoyle–Higgs 2001 S,π with stationary mass on the 6 canonical+wobble doublets and a **tunable compensatory exchangeability λ**.
- **Tree:** a real 5S/tRNA topology (or a Yule tree), branch lengths swept as the uncertainty knob.
- **Ground truth:** exact ancestor at every internal node.
- **Sweep:** compensatory rate λ (or off-canonical stationary mass) × branch length. Predicted curve: at short branches independent-site is already near-perfect (no room); at very long branches both fail (signal erased); in the moderate band the **joint/paired decoder wins by tens of points of pair-accuracy**, concentrated exactly on branches carrying a compensatory double substitution. Reproducing that curve *is* the headline result.

### Stage 1: real data

Use Rfam seed alignments — every family ships a Stockholm file whose `#=GC SS_cons` line **gives the base-pair partition for free** (no contact inference, no discovery, no flip-recall confound — this is the structure-supervised case, the analogue of `apply_pdb_partition`).

**Primary family — 5S rRNA, RF00001:** `https://rfam.org/family/RF00001` (compact ~120 nt, ~30–37 pairs, spans Bacteria/Archaea/Eukarya = deep divergence, classic paired-model benchmark; exact 16-state decoder is cheap at this size). Documented signal: 299 compensatory substitutions on 5S, terminal:intermediate clustering ~5.8:1 vs ~1:1.17 neutral, ~89% of coevolving groups on known base pairs (PLoS ONE 2012 e0044376).

**Orthology-clean alternative:** a **single tRNA isotype** (e.g. tRNA-Phe) pulled across a clean species set from **GtRNAdb** (`https://gtrnadb.ucsc.edu`, Chan & Lowe 2016). This avoids the RF00005 multigene/paralog trap (40–60 paralogous tRNA genes/genome → a concatenated full-family "tree" is a functional dendrogram, not a phylogeny).

**"More room" follow-up:** **RNase P RF00010** (`https://rfam.org/family/RF00010`) or riboswitch aptamers (FMN RF00050, TPP RF00059, cobalamin RF00174) — deep bacterial divergence with stems that stay paired, i.e. strong coupling *and* genuine node uncertainty. LSU eukaryotic expansion segments (RF02543) have the best room-of-all but bring size + alignment pain; SSU RF00177 is the cleaner large-rRNA cut if you want it.

Run **R-scape** (Rivas, Clements & Eddy 2017, *Nat Methods* 14:45; `http://eddylab.org/R-scape/`) on the seed and **evaluate ASR gain only on the R-scape-significant, actively-substituting columns** — ultra-conserved core stems have strong coupling but no reconstruction uncertainty, and averaging them in will dilute a real win into a false "modest" number.

### Why the coupling is strong enough — with the contrast

| Quantity | Our protein contacts | RNA WC stem pair |
|---|---|---|
| Mutual information / pair | ~0.14 bit (~0.1 nat) | ~1 bit typical, up to 2 bit ceiling |
| Genuine coupling over a well-fit null | ~0.003–0.004 nat/count | near-hard constraint; survives R-scape phylogeny correction |
| Dimensionality | diffuse across many residue pairs | ~6 of 16 doublets carry mass |
| Where it lives | dynamics only (stationary counts show a composition continuum) | **stationary** distribution — directly visible |

Tillier & Collins 1998 (*Genetics* 148:1993) measured the double:single substitution-rate ratio as **high** in core rRNA — compensatory changes fix so fast they "appear instantaneous," which is precisely the event a site-factored model misreads as two improbable independent substitutions and a paired model scores as one cheap step.

### Why the phylogeny leaves room

Room is controllable (simulation) and, on real data, present but **anti-correlated with conservation**. Score it directly: the fraction of true ancestral pairs whose *independent-site* MAP is non-canonical is the exact headroom; the paired decoder recovers most of it by snapping to a valid pair. This is the RNA analogue of our composite-ELBO N_eff finding — the win is set by (coupling strength) × (single-site ambiguity), not by node count.

### The specific model variant to run

**Not** the amino-acid composite Potts. Run a **paired-nucleotide doublet CTMC**: alphabet A=4, coupled model = a single **16-state (doublet) Felsenstein chain**, independent baseline = **two 4-state GTR+Γ Felsenstein chains**. Use the full 16-state (RNA16A/C) form, **not** a 6-state canonical-only model, so GU wobble and mismatch intermediates exist and are correctly priced (otherwise you mis-score the very intermediates the decoder must reject). Drop the LG-C10/C20 archetype / de-Finetti latent-field layer entirely — RNA coupling is stationary and position-specific, so a plain doublet generator is both sufficient and the literature standard (Muse 1995; Savill et al. 2001; Jow et al. 2002 / PHASE).

### Expected win

- **Simulation:** clean, large, monotone in λ and branch length — near-100% vs sharply worse pair-accuracy on compensatory-substitution branches. Guaranteed if the machinery is correct.
- **Real 5S/tRNA:** reliably positive but modest in aggregate (conserved columns contribute no gap), **large when stratified** on compensatory branches / R-scape-significant columns. Sharpest single readout: the paired decoder **near-eliminates reconstructed non-canonical ancestral pairs** that the independent decoder produces — a clean binary plot. (The one published probabilistic-ish precedent, structure-aware parsimony ASR, recovered only ~1.5–3% more canonical pairs — PMC5123390 — but that is parsimony and is a *floor*, not a ceiling, for a proper paired CTMC.)

### Minimal implementation path (given what we already have)

Verified against the tree:

1. **Reuse the alphabet-agnostic forward.** `src/tkfdp/coupling/dynfield/phylo_elbo/exact_cap2_jax.py` already takes the alphabet dimension `A` as a parameter throughout (`_leaf_msg_pair(obs, L, A)`, `exact_pair_tree_ll(...)`) — it is not hardwired to 20. At A=4 the joint pair state space is 16, so the **exact** cap-2 path is trivially cheap; **no Lumpable / class-marginal / responsibility-mixture approximation is needed** (those exist only to tame A=400). For RNA the simplest correct thing is a plain 16-state Felsenstein — you don't even need the field/`J`-boost term, which exists to couple two otherwise-independent chains via the latent field.
2. **Single-site baseline is nearly free.** `src/tkfdp/lg08.py::build_single_site_Q(S, pi)` is a general GTR builder — feed it a 4-state nucleotide S and π (HKY/GTR+Γ). Get `get_lg08()` out of the loop; it's amino-acid-specific.
3. **What must be generalized:** `A_DIM = 20` is hardwired in `cluster_hr_exact.py:48` and `cluster_hr_jax.py:46`, and `supervised_trainer.py` assumes the `ACDEFGHIKLMNPQRSTVWY` alphabet (`aa_a < 20`, line 519). The Holmes–Rubin bridge E-step (`cluster_hr_*.py`) is only needed if you want to **fit** the doublet rate matrix; for the first POC you can **fix** the generator (RNA16A for real data, the known matrix in simulation) and skip fitting entirely — then only the alphabet-4 leaf/emission layer and the two decoders are new work.
4. **Decoders:** the marginal and joint-MAP-over-the-pair decoders we already ran transfer verbatim; only the alphabet changes.
5. **Generator form, if you do fit:** `src/tkfdp/generator.py` is the F81/square-root-Metropolis reversible joint generator, reversible w.r.t. `π_joint(x,y) ∝ π₁(x)π₂(y)exp(−H(x,y))` — the algebra is alphabet-general; instantiate at A=4 with H strongly negative on the 6 WC+wobble doublets.

Net: a bounded new leaf/alphabet + nucleotide-baseline layer reusing the tree-forward unchanged; the small alphabet *removes* the scalability approximations that complicate the protein pipeline rather than adding to them.

### Discipline (don't re-derive a false null)

- **Baseline must be a genuinely well-fit GTR+Γ** nucleotide Felsenstein, not a strawman — same lesson as the protein null (a bad null inflates apparent coupling; here a bad null could also *hide* a real win).
- **Pre-register the stratification** (paired ∧ R-scape-significant ∧ compensatory-branch) or the headline number looks unimpressive.
- **Report both legs.** The simulation win proves the *method*; it is **not** evidence that real RNA ASR is coupling-limited, and it does **not** rescue the protein result. Frame the whole thing as "coupling-aware ASR *can* win when coupling is strong and low-dimensional — a controlled positive control contrasting our protein null," not as a new method (PHASE has done paired-model marginal ASR since 2002).

---

## Runners-up (brief)

- **Lattice protein / synthetic Potts-on-tree (rank 3).** Jacquin et al. 2016 (*PLoS Comput Biol* 12:e1004889) gives an exactly-solvable Hamiltonian with true couplings and a computable true posterior; independent-site models provably cannot fold designed sequences. Rodríguez-Horta, Lage-Castellanos & Mulet 2021 (*J Stat Mech* 093502; arXiv:2108.03801) already **prove** coupling-aware Bayesian ASR beats independent-site over a wide (J × branch-length) range — reproduce their decisive regime as a second, protein-flavored positive control and to map the phase boundary. `adabmDCA 2.0` (Rosset/Muntoni et al. 2025, arXiv:2501.18456) and `SISSI/SISSIz` (Gesell & von Haeseler 2006) are the turnkey simulators.
- **HK–RR interface (rank 4).** Strongest *concentrated* protein coupling (Weigt et al. 2009 *PNAS* 106:67; Bitbol et al. 2016 *PNAS* 113:12180), but inter-protein — the ASR benchmark needs correct paralog matching, and the natural task is partner-matching, not per-site ancestral accuracy. A bridge system, not a first POC.

## Traps (looked promising, aren't)

- **Protein bmDCA forward-sim** (β-lactamase PF13354/PSE-1, PF00072). Full DCA J is all-to-all; our cap-2 model structurally can't represent it (model-mismatch), and De Leonardis, Pagnani & Barrat-Charlaix 2025 (*MBE* 42:msaf070) show a **reversible** Potts evolver yields only ~2% aggregate Hamming gain — our detailed-balance generator lives in exactly that weak regime. The dramatic literature gains (0.4→0.3) come from *irreversible* arDCA, which we are not. Risks reproducing the null under a self-consistency halo. (A *matched sparse-pair* self-simulation with our own generator would win, but has zero external validity — it's just Stage 0 with extra steps.)
- **Netti & Weigt 2026 evolutionary intermediates** (arXiv:2606.27983). Mechanically a beautiful fit (reversible CTMC on a Potts landscape; midpoint = cherry ancestor). But it *is* the chorismate-mutase/β-lactamase bmDCA landscape — our failing ~0.1-nat regime — and the paper's own Fig 2A shows the coupling gain is in **ensemble faithfulness, not point accuracy** (their baseline is a crude direct-path heuristic weaker than a proper independent-site Felsenstein). You'd reproduce our null in cleaner clothes. Excellent *validation substrate* for "does the coupled decoder recover the ensemble," not a POC for an accuracy win.
- **Structure/stability evolver** (Williams, Pollock, Blackburne & Goldstein 2006, *PLoS Comput Biol* 2:e69). Fitness is a nonlinear threshold on ΔG_fold → genuinely **many-body**, not pairwise; a reversible pairwise CTMC is doubly misspecified. The celebrated over-stability bias is already erased by independent-site **Bayesian** posterior sampling (BI ΔΔG ≈ −0.05), so a coupling-aware decoder would have to beat independent-Bayesian, which the literature never shows on this system. Robustness check *after* a POC lands, not the POC.
- **Steroid-receptor GR/MR** (Bridgham/Ortlund/Thornton 2009 *Nature* 461:515; Harms & Thornton 2014; Starr et al. 2017 *Nature* 549:409). The field's best *existence proof* that epistasis controls ancestral states — but the ancestors are themselves ML-ASR inferences (resurrection validates function, not sequence), so there is **no independent ground truth to score Hamming against**, and the epistasis is higher-order/sparse on ~5–10 sites along one lineage. Cite for relevance; never score decoders on it.
- **Viral RNA (HIV RRE / HCV IRES).** Strong WC coupling and high within-species uncertainty, but HIV within-host recombination (~10⁻⁵–10⁻⁴/base/gen) is a first-order violation of the single-tree CTMC — it will muddy toward a null for a *tree* reason, not a coupling reason. Late stress-test only, and only after GARD/3SEQ recombination filtering.
- **Salt bridges / our own PDB flips.** This is our documented null (~0.003–0.004 nat/count; 0/72 confirmed flips reached off-mass > 0.3). Keep as the "not enough coupling" negative-control anchor against which the RNA win is measured.

---

## Key references

- Zeinaty, di Bari, Rossi, Barrat-Charlaix, Zamponi, Weigt 2026, "Towards coevolution-aware ancestral sequence reconstruction," arXiv:2606.27942 — protein coupling helps only for weakly-mutable roots at intermediate divergence; washes out otherwise.
- De Leonardis, Pagnani, Barrat-Charlaix 2025, *MBE* 42(4):msaf070 — reversible Potts evolver ~2% gain vs irreversible arDCA 0.4→0.3; the reversibility-ceiling warning.
- Muniz-Trejo, Park & Thornton 2025, *MBE* msaf084 (bioRxiv 2024.12.20.629812) — the Thornton-lab statement of our exact null on real protein alignments.
- Tillier & Collins 1998, *Genetics* 148:1993 — high apparent rate of simultaneous compensatory substitution in rRNA.
- Savill, Hoyle & Higgs 2001, *Genetics* 157:399 — 6/7/16-state paired RNA CTMCs; nonzero double-substitution rate fits significantly better.
- Smith, Lui & Tillier 2004, *MBE* 21:419 — empirical RNA16A/C doublet matrices (the "LG08 of base pairs").
- Muse 1995, *Genetics* 139:1429; Schöniger & von Haeseler 1994, *MPE* 3:240 — origin of the paired-site CTMC. Jow, Hudelot, Rattray & Higgs 2002, *MBE* 19:1591 — PHASE, Bayesian paired-model ASR on a tree.
- Rivas, Clements & Eddy 2017, *Nat Methods* 14:45 (R-scape); Tavares, Rivas & Eddy 2020, *Bioinformatics* 36:3072 — covariation significance / power.
- Rodríguez-Horta, Lage-Castellanos & Mulet 2021, *J Stat Mech* 093502 (arXiv:2108.03801) — proof that coupling-aware ASR beats independent-site over a wide regime (the mirror image of our null). Jacquin et al. 2016, *PLoS Comput Biol* 12:e1004889 — exactly-solvable lattice-protein benchmark.
- Data: Rfam RF00001 (5S), RF00005 (tRNA), RF00010 (RNase P), RF00177 (SSU) — `https://rfam.org`; GtRNAdb `https://gtrnadb.ucsc.edu`; RNA16 matrices `https://wwwlabs.uhnresearch.ca/labs/tillier/rRNA/`; R-scape `http://eddylab.org/R-scape/`.

**One-sentence recommendation to the group:** implement the 16-state doublet emission as an A=4 pair on the existing `exact_cap2_jax` forward, run the simulated-tree sweep first as the guaranteed positive control, then reconstruct held-out ancestors on 5S rRNA (RF00001) / single-isotype tRNA scored on R-scape-significant compensatory columns — a clean, honestly-framed demonstration that coupling-aware ASR wins where coupling is strong and low-dimensional, which our protein data structurally never was.