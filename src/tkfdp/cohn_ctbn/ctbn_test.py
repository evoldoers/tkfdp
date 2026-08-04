import unittest

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm

import ctbn
import optimize

jax.config.update('jax_platform_name', 'cpu')


# ---- helpers for the parameter-recovery tests ----
# We test fitting by recovering GAUGE-INVARIANT quantities (the induced stationary
# distribution and the induced transition matrix) rather than the raw h/J/S, which
# carry the model's gauge freedoms and (for endpoint pairs) rate/time confounds.

def _blankets (C):
    seq_mask, nbr_idx, nbr_mask, *_rest = ctbn.get_Markov_blankets(C)
    return seq_mask, nbr_idx, nbr_mask

def _canonical_states (N, K):
    # states indexed by the canonical seq_to_idx / idx_to_seq (little-endian) order used
    # by q_joint and expm -- NOT ctbn.all_seqs, which is np.ndindex (big-endian).
    return jnp.stack ([ctbn.idx_to_seq(i, N, K) for i in range(N**K)])  # (N^K, K)

def _exact_stationary (seq_mask, nbr_idx, nbr_mask, params, N, K):
    # Exact Potts stationary over all N^K sequences (canonical order); differentiable in params.
    seqs = _canonical_states (N, K)
    E = jax.vmap (lambda X: ctbn.ctbn_log_marg_unnorm(X, seq_mask, nbr_idx, nbr_mask, params))(seqs)
    return jax.nn.softmax(E)

def _sim_stationary (C, params, n, rng):
    seq_mask, nbr_idx, nbr_mask = _blankets(C)
    K, N = C.shape[0], params['S'].shape[0]
    p = np.asarray (_exact_stationary(seq_mask, nbr_idx, nbr_mask, params, N, K))
    p = np.clip(p, 0, None); p /= p.sum()
    idx = rng.choice (N**K, size=n, p=p)
    return jnp.asarray(idx), (seq_mask, nbr_idx, nbr_mask, N, K)

def _sim_pairs (C, params, T, n, rng):
    seq_mask, nbr_idx, nbr_mask = _blankets(C)
    K, N = C.shape[0], params['S'].shape[0]
    pn = ctbn.normalise_ctbn_params(params)
    p0 = np.asarray (_exact_stationary(seq_mask, nbr_idx, nbr_mask, params, N, K))
    p0 = np.clip(p0, 0, None); p0 /= p0.sum()
    q = np.asarray (ctbn.q_joint(nbr_idx, nbr_mask, pn))
    P = np.asarray (expm(T * q)); P = np.clip(P, 0, None); P /= P.sum(-1, keepdims=True)
    x0 = rng.choice (N**K, size=n, p=p0)
    xT = np.empty (n, dtype=int)
    for s in range(N**K):
        m = x0 == s
        if m.any():
            xT[m] = rng.choice (N**K, size=int(m.sum()), p=P[s])
    return jnp.asarray(x0), jnp.asarray(xT), (seq_mask, nbr_idx, nbr_mask, N, K)

def _neutral_params (N):
    S = jnp.array([[0., 1.], [1., 0.]]) if N == 2 else (jnp.ones((N, N)) - jnp.eye(N))
    return {'S': S, 'J': jnp.zeros((N, N)), 'h': jnp.zeros(N)}

def _fit (loss, init_params, lr=0.05, steps=4000, patience=120):
    vg = jax.jit (jax.value_and_grad(loss))
    best, best_loss = optimize.optimize (vg, init_params, init_lr=lr, max_iter=steps,
                                         min_inc=1e-9, patience=patience, verbose=False)
    return best, best_loss

# Tests for continuous-time Bayes network inference algorithms
# For the two-component chain, use Example 1 from Cohn et al (2010)
def ising2 (q_same = 10, q_diff = 1):
    J_same = jnp.log(q_same) / 2
    J_diff = jnp.log(q_diff) / 2
    C = jnp.ones((2,2)) - jnp.eye(2)
    S = jnp.array ([[0, 1], [1, 0]])
    J = jnp.array ([[J_same, J_diff], [J_diff, J_same]])
    h = jnp.zeros(2)
    params = { 'S': S, 'J': J, 'h': h }
    return C, params

# For single-component and independent K-component examples, use the telegraph process
def telegraph (K = 1, lambda1 = 1, lambda2 = 2):
    C = jnp.zeros((K, K))
    S = jnp.array ([[0, 1], [1, 0]])
    J = jnp.zeros((2,2))
    h = jnp.log(jnp.array([lambda2, lambda1]))  # lambda2 is the rate from state #2 -> #1, h[0] is the bias of state #1
    params = { 'S': S, 'J': J, 'h': h }
    return C, params

class TestCTBN (unittest.TestCase):

    # EQUILIBRIUM STATISTICS (single sequence)

    # For a single component, the partition function and its variational bound should be sum(exp(h))
    def test_telegraph1_partition (self):
        self.do_test_telegraph_partition(1)

    def do_test_telegraph_partition (self, K):
        C, params = telegraph(K=K)
        seq_mask, nbr_idx, nbr_mask, *_rest = ctbn.get_Markov_blankets(C)
        logZ_expected = K * ctbn.logsumexp(params['h'])
        logZ_exact = ctbn.ctbn_exact_log_Z(seq_mask, nbr_idx, nbr_mask, params)
        logZ_variational, theta = ctbn.ctbn_variational_log_Z(seq_mask, nbr_idx, nbr_mask, params)
        self.assertTrue (jnp.allclose(logZ_exact, logZ_expected))
        self.assertTrue (jnp.allclose(logZ_variational, logZ_expected))

    # For two components that are not in contact, the partition function and its variational bound should be sum(exp(-h))^2
    def test_telegraph2_partition (self):
        self.do_test_telegraph_partition(2)

    # For two components that are in contact, the exact partition function should be greater than its variational lower bound
    def test_ising2_partition (self):
        C, params = ising2()
        seq_mask, nbr_idx, nbr_mask, *_rest = ctbn.get_Markov_blankets(C)
        logZ_exact = ctbn.ctbn_exact_log_Z(seq_mask, nbr_idx, nbr_mask, params)
        logZ_variational, theta = ctbn.ctbn_variational_log_Z(seq_mask, nbr_idx, nbr_mask, params)
        self.assertTrue (jnp.all(logZ_exact > logZ_variational))

    # For a single component, the log-pseudolikelihood should be equal to the log-likelihood
    def test_telegraph1_pseudo (self):
        self.do_test_telegraph_pseudo ([0])
        self.do_test_telegraph_pseudo ([1])

    def do_test_telegraph_pseudo (self, xs):
        N = 2
        K = len(xs)
        C, params = telegraph(K=K)
        seq_mask, nbr_idx, nbr_mask, *_rest = ctbn.get_Markov_blankets(C)
        xs = jnp.array(xs)
        xidx = ctbn.seq_to_idx(xs,N)
        q_joint = ctbn.q_joint(nbr_idx, nbr_mask, params)
        q_eqm = ctbn.exact_eqm (q_joint)
        ll_exact_from_eqm = jnp.log(q_eqm[xidx])
        ll_exact = ctbn.ctbn_exact_log_marg (xs, seq_mask, nbr_idx, nbr_mask, params)
        ll_pseudo = ctbn.ctbn_pseudo_log_marg (xs, seq_mask, nbr_idx, nbr_mask, params)
        self.assertTrue (jnp.isclose(ll_exact, ll_exact_from_eqm))
        self.assertTrue (jnp.isclose(ll_exact, ll_pseudo))

    # For two components that are not in contact, the log-pseudolikelihood should be equal to the log-likelihood
    def test_telegraph2_pseudo (self):
        self.do_test_telegraph_pseudo ([0,1])
        self.do_test_telegraph_pseudo ([1,1])

    # MEAN-FIELD RATE CONSISTENCY

    # q_bar_cond, marginalized over the conditioned neighbor's state, should reduce to q_bar.
    # This is a tight consistency check on the exp(2 C_ij J) factor in q_bar_cond.
    def test_ising2_q_bar_cond_consistency (self):
        self.do_test_q_bar_cond_consistency (*ising2())

    def do_test_q_bar_cond_consistency (self, C, params):
        seq_mask, nbr_idx, nbr_mask, *_rest = ctbn.get_Markov_blankets(C)
        params = ctbn.normalise_ctbn_params(params)
        K, M = nbr_idx.shape
        N = params['S'].shape[0]
        prng = jax.random.PRNGKey(0)
        mu = jax.nn.softmax (jax.random.normal (prng, (K, N)), axis=-1)  # arbitrary non-uniform mu
        for i in range(K):
            qbar = ctbn.q_bar(jnp.array([i]), nbr_idx, nbr_mask, params, mu)[0]
            qbar_cond = ctbn.q_bar_cond(i, nbr_idx, nbr_mask, params, mu)
            for j in range(M):
                if int(nbr_mask[i, j]) == 0:
                    continue
                avg = jnp.einsum('x,xab->ab', mu[nbr_idx[i, j]], qbar_cond[j])
                self.assertTrue (jnp.allclose(avg, qbar, atol=1e-5))

    # q_tilde_cond, geometrically averaged over the conditioned neighbor's state, should reduce to q_tilde.
    def test_ising2_q_tilde_cond_consistency (self):
        self.do_test_q_tilde_cond_consistency (*ising2())

    def do_test_q_tilde_cond_consistency (self, C, params):
        seq_mask, nbr_idx, nbr_mask, *_rest = ctbn.get_Markov_blankets(C)
        params = ctbn.normalise_ctbn_params(params)
        K, M = nbr_idx.shape
        N = params['S'].shape[0]
        prng = jax.random.PRNGKey(0)
        mu = jax.nn.softmax (jax.random.normal (prng, (K, N)), axis=-1)
        offdiag = ctbn.offdiag_mask(N)
        for i in range(K):
            qtilde = ctbn.q_tilde(jnp.array([i]), nbr_idx, nbr_mask, params, mu)[0]
            qtilde_cond = ctbn.q_tilde_cond(i, nbr_idx, nbr_mask, params, mu)
            for j in range(M):
                if int(nbr_mask[i, j]) == 0:
                    continue
                log_qtc = jnp.where (offdiag[None,:,:] > 0, ctbn.safe_log(qtilde_cond[j]), 0)
                geom_avg = jnp.exp (jnp.einsum('x,xab->ab', mu[nbr_idx[i, j]], log_qtc)) * offdiag
                self.assertTrue (jnp.allclose(geom_avg, qtilde * offdiag, atol=1e-5))

    # K=3 chain (1-2-3): exposes any handling bug for components with M > #real-neighbors padding.
    def test_chain3_q_bar_cond_consistency (self):
        C = jnp.array([[0,1,0],[1,0,1],[0,1,0]], dtype=jnp.float32)
        _, params = ising2()
        self.do_test_q_bar_cond_consistency (C, params)

    def test_chain3_q_tilde_cond_consistency (self):
        C = jnp.array([[0,1,0],[1,0,1],[0,1,0]], dtype=jnp.float32)
        _, params = ising2()
        self.do_test_q_tilde_cond_consistency (C, params)

    # psi must match Cohn (2010) eq for psi: sum_{j in Children_i} sum_{x_j} [mu^j_{x_j} qbar^j_{x_j,x_j|x_i} + sum_{y_j} gamma^j_{x_j,y_j} ln qtilde^j_{x_j,y_j|x_i}]
    # ising2 satisfies this trivially because nbr(i)\{j}=nbr(j)\{i}=empty; chains expose the structural difference.
    def test_ising2_psi_matches_cohn (self):
        self.do_test_psi_matches_cohn (*ising2())

    def test_chain3_psi_matches_cohn (self):
        C = jnp.array([[0,1,0],[1,0,1],[0,1,0]], dtype=jnp.float32)
        _, params = ising2()
        self.do_test_psi_matches_cohn (C, params)

    def test_chain4_psi_matches_cohn (self):
        C = jnp.array([[0,1,0,0],[1,0,1,0],[0,1,0,1],[0,0,1,0]], dtype=jnp.float32)
        _, params = ising2()
        self.do_test_psi_matches_cohn (C, params)

    def do_test_psi_matches_cohn (self, C, params):
        seq_mask, nbr_idx, nbr_mask, *_rest = ctbn.get_Markov_blankets(C)
        params = ctbn.normalise_ctbn_params(params)
        K, N = nbr_idx.shape[0], params['S'].shape[0]
        prng = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(prng)
        mu = jax.nn.softmax (jax.random.normal(k1, (K, N)), axis=-1)
        rho = jax.nn.softmax (jax.random.normal(k2, (K, N)), axis=-1) + 0.1
        for i in range(K):
            if int(seq_mask[i]) == 0:
                continue
            psi_py = ctbn.psi (i, nbr_idx, nbr_mask, params, mu, rho)
            psi_ref = self._cohn_psi (i, nbr_idx, nbr_mask, params, mu, rho)
            self.assertTrue (jnp.allclose(psi_py, psi_ref, atol=1e-4),
                             msg=f"i={i}: python={psi_py} cohn={psi_ref}")

    # Direct transcription of Cohn (2010) eq for psi (for use as ground truth in tests).
    def _cohn_psi (self, i, nbr_idx, nbr_mask, params, mu, rho):
        K, N = mu.shape
        M = nbr_idx.shape[1]
        result = jnp.zeros(N)
        for jj in range(M):
            if int(nbr_mask[i, jj]) == 0:
                continue
            j = int(nbr_idx[i, jj])
            slots = jnp.where (nbr_idx[j] == i)[0]
            if slots.shape[0] == 0:
                continue
            i_slot = int(slots[0])
            qbar_j = ctbn.q_bar_cond (j, nbr_idx, nbr_mask, params, mu)
            qtilde_j = ctbn.q_tilde_cond (j, nbr_idx, nbr_mask, params, mu)
            log_qtilde_j = ctbn.safe_log (jnp.where(qtilde_j < 0, 1, qtilde_j))
            gamma_j = ctbn.gamma (jnp.array([j]), nbr_idx, nbr_mask, params, mu, rho)[0]
            term1 = -jnp.einsum('y,xyz->x', mu[j], qbar_j[i_slot])
            term2 = jnp.einsum('yz,xyz->x', gamma_j, log_qtilde_j[i_slot])
            result = result + term1 + term2
        return result

    # TIME-DEPENDENT STATISTICS (two sequences)

    # For a single component, the ODEs for rho and mu should equal the exact results, and F should equal the log-likelihood
    def test_telegraph1_rho_mu (self):
        self.do_test_telegraph1_rho_mu (0, 0)
        self.do_test_telegraph1_rho_mu (0, 1)
        self.do_test_telegraph1_rho_mu (1, 0)
        self.do_test_telegraph1_rho_mu (1, 1)

    def do_test_telegraph1_rho_mu (self, x, y):
        C, params = telegraph(K=1)
        seq_mask, nbr_idx, nbr_mask, *_rest = ctbn.get_Markov_blankets(C)
        T = 1.
        q = ctbn.q_joint(nbr_idx, nbr_mask, params)
        zero = [ctbn.ZeroSolution(2)]
        rho_ode = ctbn.solve_rho (0, jax.nn.one_hot(y,2), seq_mask, nbr_idx, nbr_mask, params, zero, zero, T)
        rho_exact = ctbn.ExactRho (q, T, x, y)
        self.assertTrue (self.close_over_domain (rho_exact, rho_ode, T))
        mu_ode = ctbn.solve_mu (0, jax.nn.one_hot(x,2), seq_mask, nbr_idx, nbr_mask, params, zero, [rho_ode], T)
        mu_exact = ctbn.ExactMu (q, T, x, y)
        self.assertTrue (self.close_over_domain (mu_exact, mu_ode, T))
        ll_exact = jnp.log(expm(T * q)[x, y])
        F = ctbn.solve_F (seq_mask, nbr_idx, nbr_mask, params, [mu_ode], [rho_ode], T)
#        print(f"F={F} ll_exact={ll_exact}")
        self.assertTrue (jnp.isclose (ll_exact, F, rtol=1e-2, atol=1e-1))

    def close_over_domain (self, a, b, T, a_label="exact", b_label="bound", rtol=1e-3, atol=1e-3, steps=10):
        ts = jnp.arange(steps+1) * T / steps
        a_t = jnp.array ([a.evaluate(t) for t in ts])
        b_t = jnp.array ([b.evaluate(t) for t in ts])
        pred = jnp.isclose (a_t, b_t, rtol=rtol, atol=atol)
        if not jnp.all(pred):
            for t, at, bt, p in zip(ts, a_t, b_t, pred):
                print(f"t={t}: {a_label}={at} {b_label}={bt} close={p}")
        return jnp.all(pred)

    # For a single component, the variational lower bound should be equal to the log-likelihood
    def test_telegraph1_variational (self):
        self.do_test_telegraph_variational ([0], [1], 1.0)
    
    def do_test_telegraph_variational (self, xs, ys, T):
        N = 2
        K = len(xs)
        C, params = telegraph(K=K)
        seq_mask, nbr_idx, nbr_mask, *_rest = ctbn.get_Markov_blankets(C)
        xs = jnp.array(xs)
        ys = jnp.array(ys)
        xidx = ctbn.seq_to_idx(xs,N)
        yidx = ctbn.seq_to_idx(ys,N)
        q_joint = ctbn.q_joint(nbr_idx, nbr_mask, params)
        ll_exact = jnp.log(expm(T * q_joint)[xidx, yidx])
        prng = jax.random.PRNGKey(42)
        log_elbo, (mu_elbo, rho_elbo) = ctbn.ctbn_variational_log_cond (prng, xs, ys, seq_mask, nbr_idx, nbr_mask, params, T)
        self.assertTrue (jnp.allclose(log_elbo, ll_exact, rtol=1e-2, atol=1e-1))
        q1 = ctbn.q_single (params)
        for k in range(K):
            mu_exact = ctbn.ExactMu(q1, T, xs[k], ys[k])
            rho_exact = ctbn.ExactRho(q1, T, xs[k], ys[k])
            self.assertTrue (self.close_over_domain(mu_exact, mu_elbo[k], T))
            self.assertTrue (self.close_over_domain(rho_exact, rho_elbo[k], T))

    # For two components that are not in contact, the variational lower bound should be equal to the log-likelihood
    def test_telegraph2_variational (self):
        self.do_test_telegraph_variational ([0,0], [1,1], 1.0)

    # For two components that are in contact, F should be a reasonably close lower bound for the log-likelihood
    def test_ising2_variational (self):
        self.do_test_ising2_variational ([0,1], [1,0], 1.0)

    def do_test_ising2_variational (self, xs, ys, T):
        C, params = ising2()
        seq_mask, nbr_idx, nbr_mask, *_rest = ctbn.get_Markov_blankets(C)
        xs = jnp.array(xs)
        ys = jnp.array(ys)
        xidx = ctbn.seq_to_idx(xs,2)
        yidx = ctbn.seq_to_idx(ys,2)
        q_joint = ctbn.q_joint(nbr_idx, nbr_mask, params)
        ll_exact = jnp.log(expm(T * q_joint)[xidx, yidx])
        prng = jax.random.PRNGKey(42)
        log_elbo, (mu_elbo, rho_elbo) = ctbn.ctbn_variational_log_cond (prng, xs, ys, seq_mask, nbr_idx, nbr_mask, params, T, min_inc=1e-6)
        self.assertTrue (ll_exact > log_elbo)
        dt_steps = 100
        ts = jnp.linspace(0,T,dt_steps+1)
        mu_t = jnp.array ([[mu.evaluate(t) for mu in mu_elbo] for t in ts])
        for t, mu in zip(ts,mu_t):
            print(f"t={t}",*list(f" P(x{i})={mu_i[1].item()}" for i,mu_i in enumerate(mu)))

    # For two components that are in contact, reproduce Figure 3(b)-(d) from Cohn et al (2010):
    # the mean-field marginals P(x_i(t)=1 | x_0, x_T) interpolate between the conditioned
    # endpoints and track the exact forward-backward marginals -- closely at weak/moderate
    # coupling, more loosely (but still a valid lower bound) at strong coupling.
    def test_figure3_marginals_moderate (self):
        self._check_figure3 (q_same=3.0, track_atol=0.1)    # MF is a good approximation here

    def test_figure3_marginals_strong (self):
        self._check_figure3 (q_same=10.0, track_atol=None)  # MF degrades; endpoints + bound still hold

    def _check_figure3 (self, q_same, track_atol):
        C, params = ising2(q_same)
        seq_mask, nbr_idx, nbr_mask, *_rest = ctbn.get_Markov_blankets(C)
        xs, ys, T = jnp.array([0, 1]), jnp.array([1, 0]), 1.0
        # exact time-marginals via forward-backward on the 4-state joint chain
        q = ctbn.q_joint (nbr_idx, nbr_mask, ctbn.normalise_ctbn_params(params))
        xidx, yidx = ctbn.seq_to_idx(xs, 2), ctbn.seq_to_idx(ys, 2)
        Z = expm(T * q)[xidx, yidx]
        ll_exact = jnp.log(Z)
        states = _canonical_states (2, 2)  # (4,2), canonical order matching q/expm
        def exact_marg (t):
            fwd = expm(t * q)[xidx, :]            # (4,)
            bwd = expm((T - t) * q)[:, yidx]      # (4,)
            joint = fwd * bwd / Z                 # posterior over full state at t
            return jnp.array ([jnp.sum(joint * (states[:, i] == 1)) for i in range(2)])
        prng = jax.random.PRNGKey(0)
        log_elbo, (mu_elbo, _rho) = ctbn.ctbn_variational_log_cond (
            prng, xs, ys, seq_mask, nbr_idx, nbr_mask, params, T, min_inc=1e-7)
        # (i) the variational objective is a valid lower bound on the exact log-likelihood
        self.assertTrue (log_elbo <= ll_exact + 1e-2, msg=f"elbo={log_elbo} > exact={ll_exact}")
        # (ii) endpoint conditions are met exactly
        mu0 = jnp.array ([mu_elbo[i].evaluate(0.0)[1] for i in range(2)])
        muT = jnp.array ([mu_elbo[i].evaluate(T)[1] for i in range(2)])
        self.assertTrue (jnp.allclose(mu0, (xs == 1).astype(float), atol=1e-2), msg=f"mu(0)={mu0}")
        self.assertTrue (jnp.allclose(muT, (ys == 1).astype(float), atol=1e-2), msg=f"mu(T)={muT}")
        # (iii) marginals stay valid, and (moderate coupling) track the exact marginals
        for t in jnp.linspace(0.1, 0.9, 9):
            var = jnp.array ([mu_elbo[i].evaluate(t)[1] for i in range(2)])
            self.assertTrue (jnp.all((var >= -1e-3) & (var <= 1 + 1e-3)), msg=f"t={t} var={var}")
            if track_atol is not None:
                ex = exact_marg(t)
                self.assertTrue (jnp.allclose(ex, var, atol=track_atol),
                                 msg=f"t={t} exact={ex} var={var}")

    # PARAMETER FITTING
    # We recover the model by maximizing a likelihood over a simulated dataset, then check
    # gauge-invariant induced quantities (stationary / transition) against the empirical
    # sufficient statistics (the MLE optimality condition) and the truth.

    # -- Stationary log-marginal: recover h (and J for contacts) --

    # For a single component, we should be able to recover h by maximizing the log-marginal.
    def test_fit_marginal_h_single (self):
        self._check_marginal_recovery (*telegraph(K=1), desc="h (single)")

    # For two components not in contact, we should be able to recover h by maximizing the log-marginal.
    def test_fit_marginal_h_noncontact (self):
        self._check_marginal_recovery (*telegraph(K=2), desc="h (2 independent)")

    # For two components in contact, we should be able to recover h and J by maximizing the log-marginal.
    def test_fit_marginal_hJ_contact (self):
        self._check_marginal_recovery (*ising2(), desc="h,J (2 in contact)")

    def _check_marginal_recovery (self, C, params, desc, n=20000, seed=0):
        rng = np.random.default_rng(seed)
        data_idx, (seq_mask, nbr_idx, nbr_mask, N, K) = _sim_stationary(C, params, n, rng)
        counts = jnp.asarray (np.bincount(np.asarray(data_idx), minlength=N**K)).astype(float)
        tot = counts.sum()
        emp = counts / tot
        def loss (p):  # exact negative log-marginal (exact log-Z) -> true MLE
            logp = ctbn.safe_log (_exact_stationary(seq_mask, nbr_idx, nbr_mask, p, N, K))
            return -jnp.sum(counts * logp) / tot
        fitted, _ = _fit (loss, _neutral_params(N))
        induced = _exact_stationary (seq_mask, nbr_idx, nbr_mask, fitted, N, K)
        p_true = _exact_stationary (seq_mask, nbr_idx, nbr_mask, params, N, K)
        # exact MLE reproduces the empirical distribution
        self.assertTrue (jnp.allclose(induced, emp, atol=1e-2),
                         msg=f"{desc}: induced={induced} empirical={emp}")
        # and hence recovers the true stationary at this sample size
        self.assertTrue (jnp.allclose(induced, p_true, atol=2.5e-2),
                         msg=f"{desc}: induced={induced} true={p_true}")

    # -- Endpoint-pair log-joint: recover h and S (and J for contacts) --

    # For a single component, we should be able to recover h and S by maximizing the log-joint.
    def test_fit_joint_hS_single (self):
        self._check_joint_recovery (*telegraph(K=1), desc="h,S (single)", identifiable=True)

    # For two components not in contact, we should be able to recover h and S by maximizing the log-joint.
    def test_fit_joint_hS_noncontact (self):
        self._check_joint_recovery (*telegraph(K=2), desc="h,S (2 independent)", identifiable=True)

    # For two components in contact, we should be able to recover h, J and S by maximizing the log-joint.
    def test_fit_joint_hJS_contact (self):
        self._check_joint_recovery (*ising2(), desc="h,J,S (2 in contact)", identifiable=False)

    def _check_joint_recovery (self, C, params, desc, identifiable, T=1.0, n=20000, seed=1):
        rng = np.random.default_rng(seed)
        x0, xT, (seq_mask, nbr_idx, nbr_mask, N, K) = _sim_pairs(C, params, T, n, rng)
        def neg_lj (p):  # negative log-joint of the endpoint pair (x0, xT)
            pn = ctbn.normalise_ctbn_params(p)
            P = expm (T * ctbn.q_joint(nbr_idx, nbr_mask, pn))
            p0 = _exact_stationary (seq_mask, nbr_idx, nbr_mask, p, N, K)
            lj = ctbn.safe_log(p0)[x0] + ctbn.safe_log(P)[x0, xT]
            return -jnp.mean(lj)
        fitted, fit_loss = _fit (neg_lj, _neutral_params(N))
        true_loss = neg_lj(params)
        # MLE optimality: the fit explains the data at least as well as the true params
        # (holds regardless of the h/J/S gauge and rate-vs-time confound)
        self.assertTrue (fit_loss <= true_loss + 1e-2,
                         msg=f"{desc}: fit_loss={fit_loss} true_loss={true_loss}")
        # induced stationary matches the empirical start-state distribution
        emp0 = jnp.asarray (np.bincount(np.asarray(x0), minlength=N**K)).astype(float)
        emp0 = emp0 / emp0.sum()
        ind0 = _exact_stationary (seq_mask, nbr_idx, nbr_mask, fitted, N, K)
        self.assertTrue (jnp.allclose(ind0, emp0, atol=3e-2),
                         msg=f"{desc}: induced_stationary={ind0} empirical={emp0}")
        if identifiable:
            # induced transition recovers the truth -> the S rate-scale was recovered
            Pf = expm (T * ctbn.q_joint(nbr_idx, nbr_mask, ctbn.normalise_ctbn_params(fitted)))
            Pt = expm (T * ctbn.q_joint(nbr_idx, nbr_mask, ctbn.normalise_ctbn_params(params)))
            self.assertTrue (jnp.allclose(Pf, Pt, atol=5e-2),
                             msg=f"{desc}: P_fit={Pf} P_true={Pt}")

if __name__ == '__main__':
    unittest.main()