# Vendored Cohn-et-al CTBN mean-field implementation

`ctbn.py` and `bounded_while_loop.py` are copied from the **evolsnake** repo
(`git@github.com:evoldoers/evolsnake.git`), path `python/`, at commit:

    dcef5954de0303bbd447900301341184eb681b16   (2026-08-01)

They implement the endpoint-conditioned mean-field variational approximation for
continuous-time Bayesian networks of **Cohn et al. (2010)**, *Mean Field
Variational Approximation for Continuous-Time Bayesian Networks*, JMLR 11:93,
specialised to protein Potts models parameterised by contact + coupling matrices
(single-site-move Glauber rates `q(a->c | neighbours b) = S[a,c] exp(h[c] + 2 sum_b J[c,b])`).

Used here purely as a **comparison baseline** for the closed-form path ELBO
(`src/tkfdp/variational_hr.py`, `experiments/elbo_vs_expm.py`) and the exact
matrix exponential — see `experiments/cohn_vs_elbo_pair.py`.

## Files

- `ctbn.py`, `bounded_while_loop.py` — the runtime code.
- `ctbn_test.py`, `optimize.py` — evolsnake's unit-test suite (verbatim) + its
  `optimize` dependency, vendored so the algorithm's regression tests (the
  `q_bar_cond`/`q_tilde_cond` marginalization-consistency checks and the
  `psi`-matches-Cohn tests that guard the sign / padding / psi fixes) travel with
  the vendored code. All 24 tests pass against this (modified) `ctbn.py`; run from
  this directory with `JAX_PLATFORMS=cpu python3 -m unittest ctbn_test`.

## Local modifications (kept minimal, for reproducibility)

`ctbn.py` is verbatim from evolsnake except for two **non-algorithmic** changes
(the mean-field math -- `q_bar`, `q_tilde`, `psi`, `gamma`, the rho/mu ODEs, `F` --
is untouched; a 2026-08-01 audit confirmed all three upstream fix commits (sign
`0bb5729`, padding `29a13d1`, psi `cefb4c6`) are present, and the vendored commit
`dcef595` is HEAD with no later fixes):

- **`CTBN_MAX_STEPS = 1 << 14` (16384)** passed as `max_steps=` to the three
  `diffrax.diffeqsolve` calls (`solve_F`, `solve_rho`, `solve_mu`). evolsnake's
  tests only exercise N=2, which converges inside diffrax's default 4096-step cap;
  the 400-state (N=20) two-site case has stiffer rho/mu ODEs on off-diagonal
  (substitution) endpoint pairs that exceed it. (Must stay modest:
  `SaveAt(dense=True)` preallocates buffers of length `max_steps` per solve, so an
  astronomically large cap OOMs XLA compilation.)
- **`rtol`/`atol` added to the `ctbn_variational_log_cond` signature** (defaults
  1e-3 / 1e-6, matching the upstream hardcoded values) and threaded to every
  `solve_rho`/`solve_mu`/`solve_F` call, so the ODE tolerance -- which governs the
  accuracy of the log-singular `F` integral -- is tunable for the bound-validity
  tolerance sweep (`experiments/cohn_bound_tolerance_diag.py`,
  `cohn_tight_tol.py`).

Do NOT edit these files to "fix" behaviour for tkf-dp's own model — that is what
`variational_cohn.py` / `variational_hr.py` are for. Re-sync from evolsnake by
re-copying and re-applying the `max_steps` change if the upstream algorithm changes.
