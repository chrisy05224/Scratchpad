"""
Magnetic susceptibility of a 1D Ising chain with an OPEN-boundary,
site-dependent external field h_i, estimated from an observed TIME SERIES
of spin configurations.

    H(s) = - sum_{i=0}^{N-2} J_i s_i s_{i+1}  -  sum_{i=0}^{N-1} h_i s_i

Open boundary conditions: site 0 and site N-1 each touch exactly one bond;
there is no wraparound coupling between the two ends of the chain.

Inputs
------
h : (N,) array -- external field at each site (index-aligned to the chain,
    h[i] is the field felt by spin i). Fixed across time.
s : (N,) or (T, N) array of +/-1 -- the observed chain. If 2D, each ROW is
    one full chain snapshot and rows are timestamps (t = 0..T-1); a plain
    (N,) array is treated as a single-timestamp chain (T=1).

The bond couplings J_i are not given directly -- they are estimated by
maximum pseudo-likelihood from `s` and the known `h`. When `s` has many
rows, all timestamps are pooled into one pseudo-likelihood fit, which is
what makes the estimate of J solid: a single snapshot only weakly pins
down N-1 couplings, but dozens/hundreds of independent-in-time snapshots
of the *same* chain (same J, same h) sharpen the fit a lot without
changing the model at all.

Once J is estimated, the full susceptibility matrix

    chi[k, l] = beta * ( <s_k s_l> - <s_k><s_l> )

is computed exactly (no sampling, no approximation) via the "site factor /
bond factor" transfer-matrix construction:

    F_i = diag(exp(beta*h_i), exp(-beta*h_i))                    (site factor)
    B_i = [[exp(beta*J_i), exp(-beta*J_i)],
           [exp(-beta*J_i), exp(beta*J_i)]]                      (bond factor)

    Z = u^T  (F_0 B_0)(F_1 B_1) ... (F_{N-2} B_{N-2}) F_{N-1}  u,   u = (1, 1)

Every site contributes its own field factor exactly once and every bond
contributes its own coupling factor exactly once, so there is no field
"splitting" and no risk of double-counting or dropping a boundary term.
<s_k> and <s_k s_l> follow by inserting sigma_z = diag(1, -1) into this
product at the position(s) of site k (and l). All products are kept
numerically stable via running max-abs rescaling (every factor here is
strictly positive, so rescaling only avoids overflow/underflow -- it never
changes a sign). Building the full (N, N) matrix is O(N^2) and vectorized
with NumPy, so it comfortably handles chains of several thousand sites.
"""

import numpy as np
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# 1. MLE estimation of J from an observed spin time series
# ---------------------------------------------------------------------------

def estimate_J_mle(s, h, beta=1.0, l2_reg=1e-8):
    """
    s : (N,) or (T, N) array of +/-1. Rows (if 2D) are timestamps; all rows
        are assumed to be draws from the same chain (same J, same h).
    h : (N,) array, known external field.
    beta : inverse temperature.

    Returns J, shape (N-1,): J[i] is the coupling on bond (i, i+1), fit by
    maximizing the pooled pseudo-likelihood

        sum_t sum_i log P(s[t, i] | neighbors, J, h)

    with the OPEN-chain conditional local field

        lf[t, 0]     = J[0]   * s[t, 1]                    + h[0]
        lf[t, i]     = J[i-1] * s[t, i-1] + J[i] * s[t, i+1] + h[i]   (0 < i < N-1)
        lf[t, N-1]   = J[N-2] * s[t, N-2]                   + h[N-1]

        P(s_i = +1 | neighbors) = sigmoid(2 * beta * lf_i)

    A small L2 ridge (`l2_reg`) is added purely for conditioning (e.g. the
    T=1, N=2 corner case); it has negligible effect on the optimum once
    there are a handful of timestamps.
    """
    s = np.atleast_2d(np.asarray(s, dtype=np.float64))
    h = np.asarray(h, dtype=np.float64)
    T, N = s.shape
    if h.size != N:
        raise ValueError("s and h must have the same number of sites N")
    if N < 2:
        raise ValueError("chain needs at least 2 sites")
    if not np.all((s == 1.0) | (s == -1.0)):
        raise ValueError("s must contain only +1/-1 values")
    if not np.all(np.isfinite(h)):
        raise ValueError("h must contain only finite values")

    s_prev = np.zeros_like(s)
    s_prev[:, 1:] = s[:, :-1]          # s_prev[t, i] = s[t, i-1], 0 at i=0
    s_next = np.zeros_like(s)
    s_next[:, :-1] = s[:, 1:]          # s_next[t, i] = s[t, i+1], 0 at i=N-1

    def neg_pll_and_grad(J):
        Jleft = np.concatenate(([0.0], J))     # Jleft[i]  = J[i-1], i = 1..N-1
        Jright = np.concatenate((J, [0.0]))    # Jright[i] = J[i],   i = 0..N-2

        lf = Jleft[None, :] * s_prev + Jright[None, :] * s_next + h[None, :]
        x = beta * lf
        ax = np.abs(x)
        log2cosh = ax + np.log1p(np.exp(-2.0 * ax))            # stable log(2 cosh x)
        pll = np.sum(s * x - log2cosh) - l2_reg * np.sum(J * J)

        r = s - np.tanh(x)                                      # (T, N) residual
        grad_pll = beta * np.sum(
            r[:, :-1] * s_next[:, :-1] + r[:, 1:] * s_prev[:, 1:], axis=0
        )
        grad_pll -= 2.0 * l2_reg * J
        return -pll, -grad_pll

    J0 = np.zeros(N - 1)
    result = minimize(neg_pll_and_grad, J0, jac=True, method="L-BFGS-B")
    return result.x


# ---------------------------------------------------------------------------
# 2. Transfer-matrix machinery, open boundary conditions, per-bond J
# ---------------------------------------------------------------------------

def _renorm_vec(v):
    """Rescale a batch of 2-vectors (..., 2) by their max-abs entry."""
    scale = np.abs(v).max(axis=-1)
    scale = np.where(scale == 0, 1.0, scale)
    return v / scale[..., None], np.log(scale)


def build_factors(J, h, beta):
    """
    J : (N-1,) per-bond coupling.
    h : (N,) per-site field.

    Returns:
      M[i]      : (2,2), i=0..N-2, = diag(F_i) @ B_i  ("site i field, then
                  bond i->i+1"), rescaled so the max entry is 1
      SigmaM[i] : (2,2), i=0..N-2, = diag(sigma_z) @ diag(F_i) @ B_i
      logscaleM[i] : log of the rescaling factor removed from M[i]/SigmaM[i]
      F_last, SigmaF_last : (2,) vectors for the final, bond-less site N-1
    """
    J = np.asarray(J, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    N = h.size
    if J.size != N - 1:
        raise ValueError("J must have length N-1 for an N-site open chain")

    eh, emh = np.exp(beta * h), np.exp(-beta * h)
    F = np.stack([eh, emh], axis=1)            # (N, 2)
    SigmaF = np.stack([eh, -emh], axis=1)      # (N, 2)  (sigma_z applied)

    eJ, emJ = np.exp(beta * J), np.exp(-beta * J)
    B = np.empty((N - 1, 2, 2))
    B[:, 0, 0] = eJ
    B[:, 0, 1] = emJ
    B[:, 1, 0] = emJ
    B[:, 1, 1] = eJ

    M_raw = F[:-1, :, None] * B                 # diag(F_i) @ B_i, i=0..N-2
    SigmaM_raw = SigmaF[:-1, :, None] * B

    c = np.maximum(M_raw.reshape(N - 1, 4).max(axis=1), 1e-300)
    logscaleM = np.log(c)
    M = M_raw / c[:, None, None]
    SigmaM = SigmaM_raw / c[:, None, None]     # same rescale as M (sigma_z only flips signs)

    return M, SigmaM, logscaleM, F[-1], SigmaF[-1]


def prefix_vectors(M, logscaleM):
    """P[k], logP[k] for k=0..N-1: normalized/rescaled u_row @ M[0]@...@M[k-1]."""
    Nb = M.shape[0]  # N-1 bonds -> N sites, k ranges 0..N-1
    P = np.empty((Nb + 1, 2))
    logP = np.empty(Nb + 1)
    P[0] = np.array([1.0, 1.0])
    logP[0] = 0.0
    for i in range(Nb):
        v = P[i] @ M[i]
        vn, dlog = _renorm_vec(v)
        P[i + 1] = vn
        logP[i + 1] = logP[i] + logscaleM[i] + dlog
    return P, logP


def suffix_vectors(M, logscaleM, F_last):
    """
    S[k], logS[k] for k=0..N-1: normalized/rescaled M[k]@...@M[N-2]@diag(F_last)@u_col.
    S[N-1] is the base case (just F_last @ u_col, i.e. F_last summed).
    """
    Nb = M.shape[0]
    S = np.empty((Nb + 1, 2))
    logS = np.empty(Nb + 1)
    S[Nb] = F_last.copy()
    logS[Nb] = 0.0
    for i in range(Nb - 1, -1, -1):
        v = M[i] @ S[i + 1]
        vn, dlog = _renorm_vec(v)
        S[i] = vn
        logS[i] = logS[i + 1] + logscaleM[i] + dlog
    return S, logS


def log_partition_function(P, logP, F_last):
    """Z = P[N-1] . F_last (P[N-1] has NOT yet absorbed the last site's own
    field factor -- that only happens here, or inside suffix_vectors)."""
    Nb = P.shape[0] - 1
    tot = P[Nb] @ F_last
    return logP[Nb] + np.log(tot)


def magnetization_all(P, logP, S, logS, SigmaM, F_last, SigmaF_last, logscaleM, logZ):
    """<s_k> for every site k, O(N) vectorized."""
    N = P.shape[0]
    Nb = N - 1

    # k = 0..N-2: sigma inserted via SigmaM[k] instead of M[k]
    vec = np.einsum('ka,kab->kb', P[:Nb], SigmaM)          # (N-1, 2)
    val = np.einsum('kb,kb->k', vec, S[1:Nb + 1])
    logfac = logP[:Nb] + logscaleM + logS[1:Nb + 1] - logZ
    m = np.empty(N)
    m[:Nb] = np.exp(logfac) * val

    # k = N-1: last site has no trailing bond, use SigmaF_last directly
    valN = P[Nb] @ SigmaF_last
    m[Nb] = np.exp(logP[Nb] - logZ) * valN
    return m


def correlation_matrix(M, SigmaM, logscaleM, P, logP, S, logS, F_last, SigmaF_last, logZ):
    """
    Full <s_k s_l> matrix (diagonal exactly 1). O(N^2), vectorized over the
    gap g = l - k so the Python-level loop runs O(N) times while each step
    is a batched (n, 2, 2)/(n, 2) operation over every valid k at once.
    """
    N = P.shape[0]
    corr = np.eye(N)
    Nb = N - 1  # number of bonds

    # A[k] = P[k] @ SigmaM[k], valid for k = 0..N-2 (site k with sigma inserted,
    # already advanced across bond k)
    A = np.einsum('ka,kab->kb', P[:Nb], SigmaM)             # (N-1, 2)
    logA = logP[:Nb] + logscaleM

    Svec, logSvec = A, logA          # "S_1[k]" carrying forward from site k
    for g in range(1, N):
        n_valid = N - g              # number of (k, k+g) pairs at this gap
        # For every gap g, l = k+g ranges over g..N-1 as k ranges 0..n_valid-1.
        # l = N-1 (the last, bond-less site) is ALWAYS hit -- at the single
        # largest k in this batch (k = n_valid-1) -- regardless of g, since
        # SigmaM/S only cover interior/left sites 0..N-2, that one pair needs
        # the SigmaF_last special case; every other pair uses the general path.
        n_general = n_valid - 1      # k = 0..n_general-1, l = g..N-2

        if n_general > 0:
            Ml = SigmaM[g: g + n_general]                    # bond/site factor at l, sigma inserted
            vals = np.einsum('kb,kbc->kc', Svec[:n_general], Ml)
            logvals = logSvec[:n_general] + logscaleM[g: g + n_general]
            Rl = S[g + 1: g + 1 + n_general]
            logRl = logS[g + 1: g + 1 + n_general]
            corrvals_gen = np.exp(logvals + logRl - logZ) * np.einsum('kb,kb->k', vals, Rl)
            k_idx = np.arange(n_general)
            corr[k_idx, k_idx + g] = corrvals_gen
            corr[k_idx + g, k_idx] = corrvals_gen

        # special pair: k = n_valid-1, l = N-1 (last, bond-less site)
        k_last = n_valid - 1
        val_special = np.exp(logSvec[k_last] - logZ) * (Svec[k_last] @ SigmaF_last)
        corr[k_last, N - 1] = val_special
        corr[N - 1, k_last] = val_special

        # advance Svec through bond g (site k -> ... -> site k+g, via plain M[g])
        # for the next gap; only the n_general entries survive (next gap's
        # max k is one less than this gap's). Must be rescaled every step --
        # left un-rescaled this overflows float64 within a few hundred sites.
        if n_general > 0:
            Mg = M[g: g + n_general]
            Svec_next = np.einsum('kb,kbc->kc', Svec[:n_general], Mg)
            Svec_next, dlog = _renorm_vec(Svec_next)
            logSvec = logSvec[:n_general] + logscaleM[g: g + n_general] + dlog
            Svec = Svec_next

    return corr


# ---------------------------------------------------------------------------
# 3. Top-level entry point
# ---------------------------------------------------------------------------

def magnetic_susceptibility(h, s, beta=1.0, return_details=False):
    """
    h : (N,) array -- known external field, one value per site (site i's
        field is h[i]; static across time).
    s : (N,) or (T, N) array of +/-1 -- observed chain. If 2D, each row is
        a full snapshot and rows are timestamps; a 1D array is one
        snapshot (T=1).
    beta : inverse temperature (default 1.0).

    J is unknown and is first estimated via pooled maximum pseudo-
    likelihood from every timestamp in `s` (see estimate_J_mle). The
    (N, N) susceptibility matrix

        chi[k, l] = beta * (<s_k s_l> - <s_k><s_l>)

    is then computed exactly, under OPEN boundary conditions (the two end
    spins do not interact with each other), via the transfer-matrix method.

    If return_details=True, returns (chi, J, magnetization) instead.
    """
    s_arr = np.atleast_2d(np.asarray(s, dtype=np.float64))
    h = np.asarray(h, dtype=np.float64)
    N = h.size

    if s_arr.shape[1] != N:
        raise ValueError("s and h must have the same number of sites N")
    if N < 2:
        raise ValueError("chain needs at least 2 sites")
    if not np.all((s_arr == 1.0) | (s_arr == -1.0)):
        raise ValueError("s must contain only +1/-1 values")
    if not np.all(np.isfinite(h)):
        raise ValueError("h must contain only finite values")

    J = estimate_J_mle(s_arr, h, beta=beta)

    M, SigmaM, logscaleM, F_last, SigmaF_last = build_factors(J, h, beta)
    P, logP = prefix_vectors(M, logscaleM)
    S, logS = suffix_vectors(M, logscaleM, F_last)
    logZ = log_partition_function(P, logP, F_last)

    m = magnetization_all(P, logP, S, logS, SigmaM, F_last, SigmaF_last, logscaleM, logZ)
    corr = correlation_matrix(M, SigmaM, logscaleM, P, logP, S, logS, F_last, SigmaF_last, logZ)

    chi = beta * (corr - np.outer(m, m))
    chi = 0.5 * (chi + chi.T)   # symmetrize away float round-off

    if return_details:
        return chi, J, m
    return chi


def bulk_susceptibility(chi):
    return chi.sum() / chi.shape[0]


def local_susceptibility(chi):
    return np.diag(chi)


# ---------------------------------------------------------------------------
# 4. Self-checks
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time
    from itertools import product as iproduct

    rng = np.random.default_rng(0)

    # --- 4.1 brute-force validation of the transfer-matrix physics (known J, h) ---
    def brute_force(J, h, beta):
        N = h.size
        Z = 0.0
        m_num = np.zeros(N)
        corr_num = np.zeros((N, N))
        for config in iproduct([1.0, -1.0], repeat=N):
            sv = np.array(config)
            E = -np.sum(J * sv[:-1] * sv[1:]) - np.sum(h * sv)
            w = np.exp(-beta * E)
            Z += w
            m_num += w * sv
            corr_num += w * np.outer(sv, sv)
        m = m_num / Z
        corr = corr_num / Z
        chi = beta * (corr - np.outer(m, m))
        return np.log(Z), m, corr, chi

    n_trials = 30
    for t in range(n_trials):
        N = rng.integers(2, 8)
        J = rng.uniform(-2.0, 2.0, N - 1)
        h = rng.uniform(-2.0, 2.0, N)
        beta_t = rng.uniform(0.2, 3.0)

        M, SigmaM, logscaleM, F_last, SigmaF_last = build_factors(J, h, beta_t)
        P, logP = prefix_vectors(M, logscaleM)
        S, logS = suffix_vectors(M, logscaleM, F_last)
        logZ = log_partition_function(P, logP, F_last)
        m = magnetization_all(P, logP, S, logS, SigmaM, F_last, SigmaF_last, logscaleM, logZ)
        corr = correlation_matrix(M, SigmaM, logscaleM, P, logP, S, logS, F_last, SigmaF_last, logZ)
        chi = beta_t * (corr - np.outer(m, m))

        logZ_bf, m_bf, corr_bf, chi_bf = brute_force(J, h, beta_t)

        assert np.isclose(logZ, logZ_bf, atol=1e-8), (t, N, "logZ", logZ, logZ_bf)
        assert np.allclose(m, m_bf, atol=1e-8), (t, N, "magnetization")
        assert np.allclose(corr, corr_bf, atol=1e-8), (t, N, "correlation")
        assert np.allclose(chi, chi_bf, atol=1e-6), (t, N, "chi")
    print(f"[brute force] {n_trials} random trials (N=2..7, non-constant J,h) match exact enumeration.")

    # --- 4.2 finite-difference cross-check of chi_kl (open BC, non-uniform J,h) ---
    N = 12
    J = rng.uniform(-1.0, 1.0, N - 1)
    h = rng.uniform(-0.5, 0.5, N)
    beta_t = 1.1
    eps = 1e-6

    def logZ_of(h_):
        M_, SM_, ls_, Fl_, SFl_ = build_factors(J, h_, beta_t)
        P_, lp_ = prefix_vectors(M_, ls_)
        return log_partition_function(P_, lp_, Fl_)

    def m_of(h_):
        M_, SM_, ls_, Fl_, SFl_ = build_factors(J, h_, beta_t)
        P_, lp_ = prefix_vectors(M_, ls_)
        S_, ls2_ = suffix_vectors(M_, ls_, Fl_)
        lZ_ = log_partition_function(P_, lp_, Fl_)
        return magnetization_all(P_, lp_, S_, ls2_, SM_, Fl_, SFl_, ls_, lZ_)

    M0, SM0, ls0, Fl0, SFl0 = build_factors(J, h, beta_t)
    P0, lp0 = prefix_vectors(M0, ls0)
    S0, ls20 = suffix_vectors(M0, ls0, Fl0)
    lZ0 = log_partition_function(P0, lp0, Fl0)
    m0 = magnetization_all(P0, lp0, S0, ls20, SM0, Fl0, SFl0, ls0, lZ0)
    corr0 = correlation_matrix(M0, SM0, ls0, P0, lp0, S0, ls20, Fl0, SFl0, lZ0)
    chi0 = beta_t * (corr0 - np.outer(m0, m0))

    for l in [0, 5, N - 1]:
        h_plus = h.copy(); h_plus[l] += eps
        h_minus = h.copy(); h_minus[l] -= eps
        dlogZ = (logZ_of(h_plus) - logZ_of(h_minus)) / (2 * eps)
        assert abs(dlogZ - beta_t * m0[l]) < 1e-4, (l, dlogZ, beta_t * m0[l])

        dchi = (m_of(h_plus) - m_of(h_minus)) / (2 * eps)
        err = np.max(np.abs(dchi - chi0[:, l]))
        assert err < 1e-4, (l, err)
    print("[FD check] d(lnZ)/dh_l and d<s_k>/dh_l match chi_kl at l = 0, 5, N-1.")

    # --- 4.3 open-BC sanity: end spins don't interact with each other ------
    N = 6
    J = np.array([1.0, 1.0, 1.0, 1.0, 1.0])   # would-be-ring bond N-1<->0 simply absent
    h = np.zeros(N)
    M, SigmaM, logscaleM, F_last, SigmaF_last = build_factors(J, h, 1.0)
    P, logP = prefix_vectors(M, logscaleM)
    S, logS = suffix_vectors(M, logscaleM, F_last)
    logZ = log_partition_function(P, logP, F_last)
    corr = correlation_matrix(M, SigmaM, logscaleM, P, logP, S, logS, F_last, SigmaF_last, logZ)
    # site 0 and site N-1 are the farthest apart on an open chain -> weakest correlation
    assert corr[0, N - 1] == corr.min() - 0.0 or corr[0, N - 1] <= corr[0, 1:].min() + 1e-9
    print(f"[open BC] end-to-end correlation <s_0 s_{N-1}> = {corr[0, N-1]:.4f} "
          f"(weakest in row 0, as expected with no wraparound bond).")

    # --- 4.4 exact sequential sampler + MLE recovery from a time series ----
    def sample_chain(J, h, beta, T, rng):
        """
        Exact i.i.d. sampling of T chain configurations via sequential
        conditionals built from the same suffix vectors S used above.

        S[i] = M[i] @ ... @ M[N-2] @ F_last already includes site i's own
        field factor F_i (since M[i] = diag(F_i) @ B_i), so:
          - P(s_0 = s)       propto S[0][s]                    (no bond yet)
          - P(s_i = s | s_{i-1})  propto B[i-1](s_{i-1}, s) * S[i][s]   (i >= 1)
        with S[N-1] = F_last handling the last, bond-less site automatically.
        """
        N = h.size
        eJ, emJ = np.exp(beta * J), np.exp(-beta * J)
        B = np.empty((N - 1, 2, 2))
        B[:, 0, 0] = eJ; B[:, 0, 1] = emJ
        B[:, 1, 0] = emJ; B[:, 1, 1] = eJ

        M_, SM_, ls_, Fl_, SFl_ = build_factors(J, h, beta)
        S_, _ = suffix_vectors(M_, ls_, Fl_)   # log-scales cancel in the normalized conditionals

        samples = np.empty((T, N))
        for t in range(T):
            p0 = S_[0] / S_[0].sum()
            s_prev = rng.choice([1.0, -1.0], p=p0)
            samples[t, 0] = s_prev
            for i in range(1, N):
                row = 0 if s_prev == 1.0 else 1
                w = B[i - 1, row, :] * S_[i]
                p = w / w.sum()
                s_i = rng.choice([1.0, -1.0], p=p)
                samples[t, i] = s_i
                s_prev = s_i
        return samples

    N = 15
    J_true = rng.uniform(-1.2, 1.2, N - 1)
    h_true = rng.uniform(-0.8, 0.8, N)
    beta_t = 1.0
    T = 4000
    samples = sample_chain(J_true, h_true, beta_t, T, rng)
    J_hat = estimate_J_mle(samples, h_true, beta=beta_t)
    mae = np.mean(np.abs(J_hat - J_true))
    print(f"[MLE recovery] N={N}, T={T}: mean|J_hat - J_true| = {mae:.4f} "
          f"(J_true range [{J_true.min():.2f}, {J_true.max():.2f}])")
    assert mae < 0.25

    chi, J_fit, m_fit = magnetic_susceptibility(h_true, samples, beta=beta_t, return_details=True)
    assert chi.shape == (N, N)
    assert np.allclose(chi, chi.T)
    assert np.all(np.isfinite(chi))
    print(f"[pipeline] N={N}, T={T}: chi shape={chi.shape}, symmetric & finite OK.")

    # --- 4.5 single-snapshot (T=1) and 1D-input convenience path -----------
    N = 20
    s1 = rng.choice([1.0, -1.0], size=N)
    h1 = rng.uniform(-1.0, 1.0, N)
    chi1 = magnetic_susceptibility(h1, s1, beta=1.0)
    assert chi1.shape == (N, N) and np.allclose(chi1, chi1.T)
    print(f"[1D input] N={N}, T=1 (plain 1D s): chi shape={chi1.shape} OK.")

    # --- 4.6 performance at N=5000, T=200 -----------------------------------
    N = 5000
    T = 200
    s_big = rng.choice([1.0, -1.0], size=(T, N))
    h_big = rng.uniform(-1.0, 1.0, N)
    t0 = time.time()
    chi_big, J_big, m_big = magnetic_susceptibility(h_big, s_big, beta=1.0, return_details=True)
    dt = time.time() - t0
    print(f"[perf] N={N}, T={T}: took {dt:.2f}s, chi shape={chi_big.shape}, "
          f"finite={np.all(np.isfinite(chi_big))}, symmetric={np.allclose(chi_big, chi_big.T)}")
    assert np.all(np.isfinite(chi_big))
    assert np.allclose(chi_big, chi_big.T)

    print("All checks passed.")
