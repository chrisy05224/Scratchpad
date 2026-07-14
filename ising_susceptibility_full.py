"""
Magnetic susceptibility of a 1D Ising chain with site-dependent field h_i.

The bond couplings J_i are not given: they are first estimated from the
observed spin configuration and the known field via maximum (pseudo-)
likelihood. The susceptibility matrix is then computed exactly with the
symmetrized transfer-matrix method (ring boundary conditions), following
`inhomogeneous_ising_susceptibility_spec.md`:

    H(s) = - sum_i J_i s_i s_{(i+1) mod N}  -  sum_i h_i s_i

    chi[k, l] = beta * ( <s_k s_l> - <s_k><s_l> )

Both stages are O(N) / O(N^2) and vectorized with NumPy so the chain can be
several thousand sites long. All transfer-matrix products are kept
numerically stable via running rescaling (every T_i has strictly positive
entries, so this never changes signs, only avoids overflow/underflow).
"""

import numpy as np
from scipy.optimize import minimize

SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]])


# ---------------------------------------------------------------------------
# 1. MLE estimation of J_i from a single observed configuration
# ---------------------------------------------------------------------------

def estimate_J_mle(spins, h, beta=1.0, l2_reg=1e-8):
    """
    Estimate bond couplings J (J[i] on the bond between site i and site
    (i+1) % N) from one observed configuration `spins` and known field `h`,
    by maximizing the pseudo-likelihood

        sum_i log P(s_i | s_{i-1}, s_{i+1})

    with local field  lf_i = J_{i-1} s_{i-1} + J_i s_{i+1} + h_i  and

        P(s_i = +1 | neighbors) = sigmoid(2 * beta * lf_i).

    A single configuration only weakly constrains N couplings (the problem
    is convex but can be flat), so a small L2 ridge (`l2_reg`) is added
    purely for conditioning; it has negligible effect on the optimum.

    Returns J as an (N,) float64 array.
    """
    s = np.asarray(spins, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    N = s.size
    if h.size != N:
        raise ValueError("spins and h must have the same length")
    if N < 2:
        raise ValueError("chain needs at least 2 sites")

    s_next = np.roll(s, -1)   # s_{i+1}
    s_prev = np.roll(s, 1)    # s_{i-1}

    def neg_pll_and_grad(J):
        lf = np.roll(J, 1) * s_prev + J * s_next + h
        x = beta * lf
        ax = np.abs(x)
        log2cosh = ax + np.log1p(np.exp(-2.0 * ax))          # stable log(2 cosh x)
        pll = np.sum(s * x - log2cosh) - l2_reg * np.sum(J * J)

        r = s - np.tanh(x)                                    # s_i - <s_i>_cond
        grad_pll = beta * (r * s_next + np.roll(r, -1) * s) - 2.0 * l2_reg * J
        return -pll, -grad_pll

    J0 = np.zeros(N)
    result = minimize(neg_pll_and_grad, J0, jac=True, method="L-BFGS-B")
    return result.x


# ---------------------------------------------------------------------------
# 2. Symmetrized transfer matrices (Section 2 of the spec)
# ---------------------------------------------------------------------------

def build_transfer_matrices(J, h, beta):
    """
    Returns (T, logscale):
      T[i]        : (2,2) matrix, rescaled so its largest entry is 1
      logscale[i] : log of the rescaling factor, so the true matrix is
                    exp(logscale[i]) * T[i]

    Row/col order is (s=+1, s=-1). Ring boundary conditions: h_{i+1} wraps
    with (i+1) % N.
    """
    J = np.asarray(J, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    N = J.size
    h_next = np.roll(h, -1)

    Jb = beta * J
    hb = 0.5 * beta * h
    hb_next = 0.5 * beta * h_next

    E = np.empty((N, 2, 2), dtype=np.float64)
    E[:, 0, 0] = Jb + hb + hb_next
    E[:, 0, 1] = -Jb + hb - hb_next
    E[:, 1, 0] = -Jb - hb + hb_next
    E[:, 1, 1] = Jb - hb - hb_next

    c = E.max(axis=(1, 2))               # per-bond stabilization
    T = np.exp(E - c[:, None, None])     # entries in (0, 1]
    return T, c


def _renorm(M):
    """Rescale a batch of matrices (..., 2, 2) so the max |entry| is 1."""
    scale = np.abs(M).reshape(M.shape[:-2] + (4,)).max(axis=-1)
    scale = np.where(scale == 0, 1.0, scale)
    return M / scale[..., None, None], np.log(scale)


def prefix_products(T, logscale):
    """P[k], logP[k] = normalized/rescaled product T[0]@...@T[k-1], k=0..N."""
    N = T.shape[0]
    P = np.empty((N + 1, 2, 2))
    logP = np.empty(N + 1)
    P[0] = np.eye(2)
    logP[0] = 0.0
    for i in range(N):
        M = P[i] @ T[i]
        Mn, dlog = _renorm(M)
        P[i + 1] = Mn
        logP[i + 1] = logP[i] + logscale[i] + dlog
    return P, logP


def suffix_products(T, logscale):
    """R[k], logR[k] = normalized/rescaled product T[k]@...@T[N-1], k=0..N."""
    N = T.shape[0]
    R = np.empty((N + 1, 2, 2))
    logR = np.empty(N + 1)
    R[N] = np.eye(2)
    logR[N] = 0.0
    for i in range(N - 1, -1, -1):
        M = T[i] @ R[i + 1]
        Mn, dlog = _renorm(M)
        R[i] = Mn
        logR[i] = logR[i + 1] + logscale[i] + dlog
    return R, logR


def log_partition_function(P, logP):
    N = P.shape[0] - 1
    tr = P[N, 0, 0] + P[N, 1, 1]
    return logP[N] + np.log(tr)


def _trace_sigmaz_sandwich(A, B):
    """trace(A @ diag(1,-1) @ B) for batched (...,2,2) arrays A, B."""
    return (A[..., 0, 0] * B[..., 0, 0] + A[..., 1, 0] * B[..., 0, 1]
            - A[..., 0, 1] * B[..., 1, 0] - A[..., 1, 1] * B[..., 1, 1])


def magnetization_all(P, logP, R, logR, logZ):
    """<s_k> for every site k, O(N) vectorized (Section 4)."""
    N = P.shape[0] - 1
    logfac = logP[:N] + logR[:N] - logZ
    tr = _trace_sigmaz_sandwich(P[:N], R[:N])
    return np.exp(logfac) * tr


def correlation_matrix(T, logscaleT, P, logP, R, logR, logZ):
    """
    Full <s_k s_l> matrix (diagonal is exactly 1). O(N^2), vectorized over
    the "gap" g = l - k so the Python-level loop only runs O(N) times while
    each step does a batched (n, 2, 2) matmul over all valid k at once
    (Section 4 + the prefix/suffix optimization noted in Section 6).
    """
    N = T.shape[0]
    corr = np.eye(N)

    A, dlogA = _renorm(P[:N] @ SIGMA_Z)     # A_k = Pnorm[k] @ sigma_z
    logA = logP[:N] + dlogA

    S, logS = A, logA                        # S_0[k] = A_k  (Mid_0 = I)
    for g in range(1, N):
        n_valid = N - g
        Tg = T[g - 1: g - 1 + n_valid]
        logTg = logscaleT[g - 1: g - 1 + n_valid]

        M = S[:n_valid] @ Tg
        Mn, dlog = _renorm(M)
        S = Mn
        logS = logS[:n_valid] + logTg + dlog

        Rl = R[g: g + n_valid]
        logRl = logR[g: g + n_valid]
        vals = np.exp(logS + logRl - logZ) * _trace_sigmaz_sandwich(S, Rl)

        k_idx = np.arange(n_valid)
        corr[k_idx, k_idx + g] = vals
        corr[k_idx + g, k_idx] = vals

    return corr


# ---------------------------------------------------------------------------
# 3. Top-level entry point
# ---------------------------------------------------------------------------

def magnetic_susceptibility(model, h, beta=1.0, return_details=False):
    """
    model : sequence of +1/-1, length N            (observed spin configuration)
    h     : sequence of floats, length N            (external field per site)
    beta  : inverse temperature (default 1.0)

    J_i is unknown and is first estimated via MLE (pseudo-likelihood) from
    `model` and `h`. Returns the (N, N) susceptibility matrix

        chi[k, l] = beta * (<s_k s_l> - <s_k><s_l>)

    If return_details=True, returns (chi, J, magnetization) instead.
    """
    s = np.asarray(model, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    N = s.size

    if h.size != N:
        raise ValueError("model and h must have the same length")
    if N < 2:
        raise ValueError("chain needs at least 2 sites")
    if not np.all((s == 1.0) | (s == -1.0)):
        raise ValueError("model must contain only +1/-1 values")
    if not np.all(np.isfinite(h)):
        raise ValueError("h must contain only finite values")

    J = estimate_J_mle(s, h, beta=beta)

    T, logscaleT = build_transfer_matrices(J, h, beta)
    P, logP = prefix_products(T, logscaleT)
    R, logR = suffix_products(T, logscaleT)
    logZ = log_partition_function(P, logP)

    m = magnetization_all(P, logP, R, logR, logZ)
    corr = correlation_matrix(T, logscaleT, P, logP, R, logR, logZ)

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
# 4. Self-checks (Sections 5.3 / 5.4 of the spec)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    rng = np.random.default_rng(0)

    # --- 5.4 known closed form: constant J, h=0 ---------------------------
    N, beta, Jc = 40, 0.8, 0.6
    J = np.full(N, Jc)
    h = np.zeros(N)
    T, logscaleT = build_transfer_matrices(J, h, beta)
    P, logP = prefix_products(T, logscaleT)
    R, logR = suffix_products(T, logscaleT)
    logZ = log_partition_function(P, logP)
    m = magnetization_all(P, logP, R, logR, logZ)
    corr = correlation_matrix(T, logscaleT, P, logP, R, logR, logZ)
    chi = beta * (corr - np.outer(m, m))
    expected = beta * np.exp(2 * beta * Jc)
    got = bulk_susceptibility(chi)
    print(f"[constant J,h=0] chi(0) expected={expected:.6f} got={got:.6f}")
    assert abs(got - expected) < 1e-3

    # --- 5.4 periodic-impurity closed form, k=2 (alternating sublattice) ---
    # chi_imp(0) = (1/N) * sum_{k,l in sublattice {0,k,2k,...}} chi_kl
    # (the response of the impurity sublattice's own magnetization to its
    # own field, normalized per total site) reduces to the k=1 bulk formula
    # at k=1 and to beta/2 * cosh(2 beta J) at k=2, which we check here.
    N = 200  # large enough that ring finite-size effects are negligible
    J = np.full(N, Jc)
    h = np.zeros(N)
    T, logscaleT = build_transfer_matrices(J, h, beta)
    P, logP = prefix_products(T, logscaleT)
    R, logR = suffix_products(T, logscaleT)
    logZ = log_partition_function(P, logP)
    m = magnetization_all(P, logP, R, logR, logZ)
    corr = correlation_matrix(T, logscaleT, P, logP, R, logR, logZ)
    chi = beta * (corr - np.outer(m, m))
    sub = np.arange(0, N, 2)
    expected_k2 = 0.5 * beta * np.cosh(2 * beta * Jc)
    got_k2 = chi[np.ix_(sub, sub)].sum() / N
    print(f"[k=2 impurity]    chi(0) expected={expected_k2:.6f} got={got_k2:.6f}")
    assert abs(got_k2 - expected_k2) < 1e-3

    # --- 5.3 finite-difference cross-check of chi_kl (non-uniform J,h) -----
    N = 12
    J = rng.uniform(-1.0, 1.0, N)
    h = rng.uniform(-0.5, 0.5, N)
    beta = 1.1
    eps = 1e-6

    def logZ_of(h_):
        T_, ls_ = build_transfer_matrices(J, h_, beta)
        P_, lp_ = prefix_products(T_, ls_)
        return log_partition_function(P_, lp_)

    T0, ls0 = build_transfer_matrices(J, h, beta)
    P0, lp0 = prefix_products(T0, ls0)
    R0, lr0 = suffix_products(T0, ls0)
    lZ0 = log_partition_function(P0, lp0)
    m0 = magnetization_all(P0, lp0, R0, lr0, lZ0)
    corr0 = correlation_matrix(T0, ls0, P0, lp0, R0, lr0, lZ0)
    chi0 = beta * (corr0 - np.outer(m0, m0))

    l = 3
    h_plus = h.copy(); h_plus[l] += eps
    h_minus = h.copy(); h_minus[l] -= eps
    dlogZ = (logZ_of(h_plus) - logZ_of(h_minus)) / (2 * eps)
    print(f"[FD check] d(lnZ)/dh_{l}: analytic={beta*m0[l]:.8f} numeric={dlogZ:.8f}")
    assert abs(dlogZ - beta * m0[l]) < 1e-4

    def magnetization_of(h_):
        T_, ls_ = build_transfer_matrices(J, h_, beta)
        P_, lp_ = prefix_products(T_, ls_)
        R_, lr_ = suffix_products(T_, ls_)
        lZ_ = log_partition_function(P_, lp_)
        return magnetization_all(P_, lp_, R_, lr_, lZ_)

    m_plus = magnetization_of(h_plus)
    m_minus = magnetization_of(h_minus)
    dchi = (m_plus - m_minus) / (2 * eps)
    err = np.max(np.abs(dchi - chi0[:, l]))
    print(f"[FD check] max|dchi/dh - chi[:,l]| = {err:.3e}")
    assert err < 1e-4

    # --- MLE + full pipeline smoke test -------------------------------------
    N = 25
    s = rng.choice([-1.0, 1.0], size=N)
    h = rng.uniform(-1.0, 1.0, N)
    chi, J_hat, m_hat = magnetic_susceptibility(s, h, beta=1.0, return_details=True)
    print(f"[pipeline] N={N}, chi shape={chi.shape}, symmetric={np.allclose(chi, chi.T)}")
    assert chi.shape == (N, N)
    assert np.allclose(chi, chi.T)

    # --- performance at N=5000 ----------------------------------------------
    N = 5000
    s = rng.choice([-1.0, 1.0], size=N)
    h = rng.uniform(-1.0, 1.0, N)
    t0 = time.time()
    chi, J_hat, m_hat = magnetic_susceptibility(s, h, beta=1.0, return_details=True)
    dt = time.time() - t0
    print(f"[perf] N={N} took {dt:.2f}s, chi shape={chi.shape}, "
          f"finite={np.all(np.isfinite(chi))}, symmetric={np.allclose(chi, chi.T)}")
    assert np.all(np.isfinite(chi))
    assert np.allclose(chi, chi.T)

    print("All checks passed.")
