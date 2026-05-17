"""TKF92-DP: simplified composite-likelihood pipeline + gravestone sampler.

Modules:
- lg08: LG08 exchangeabilities and stationary frequencies
- bio: Pfam loader (bio-datasets)
- cherries: tree -> cherry list with distances
- generator: 400-state joint Q(H), symmetrize, eigh, exact P(t)
- composite: composite log-likelihood
- sim: forward simulator (uniformization on the 400-state CTMC)
- partition: per-MSA matching MCMC (cluster size <= 2)
- train: Adam SGD on H
- riccati: bivariate PGF for the BDI/gravestone sampler
- branch_sampler: gravestone-augmented per-branch sampler
- bdi_reference: closed-form TKF91 P_ij(T), E[B], E[D], E[S]
- variational: N=0/N=1 mean-field ELBO on a 2-site cluster
"""

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"  # alphabetical order
A = len(AMINO_ACIDS)  # 20
A2 = A * A  # 400
