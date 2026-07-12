"""
1D Ising chain ("one-column" lattice) with a spatially varying magnetic
field, solved exactly via the transfer-matrix method.

Hamiltonian (open chain, N spins, sigma_i = +-1):

    H = -J * sum_{i=1}^{N-1} sigma_i * sigma_{i+1}  -  sum_{i=1}^{N} h_i * sigma_i

h is an array of length N (one field value per site) -- it does NOT need to
be uniform.

Transfer-matrix construction
-----------------------------
Every site contributes its own field factor exactly once and every bond
contributes its own coupling factor exactly once:

    F_i = diag(exp(beta*h_i), exp(-beta*h_i))                 (site factor)
    B   = [[exp(beta*J), exp(-beta*J)],
           [exp(-beta*J), exp(beta*J)]]                       (bond factor, same for every bond)

    Z = u^T  F_1 B F_2 B F_3 B ... B F_N  u ,   u = [1, 1]^T

This "site factor, then bond factor" grouping (as opposed to splitting each
h_i asymmetrically onto a single transfer matrix) avoids double counting or
dropping the field at the chain ends, which is a common source of bugs in
transfer-matrix code for inhomogeneous fields.

Observables
-----------
<sigma_i>            is obtained by inserting Sigma = diag(1, -1) in front of
                      F_i (i.e. replacing F_i -> Sigma @ F_i) in the product.
<sigma_i * sigma_j>   is obtained by inserting Sigma in front of both F_i and F_j.

chi_ij = beta * (<sigma_i sigma_j> - <sigma_i><sigma_j>)   (fluctuation-dissipation)
chi_total = sum_{i,j} chi_ij

Numerical stability
--------------------
For long chains / low temperature the raw matrix entries grow like
exp(beta*J*N) and can overflow float64. All chain products below are
renormalized at every step (divide by the running vector's max magnitude)
while the discarded scale is tracked as a running log; the log-scales cancel
between numerator and Z when the ratio is finally taken. This keeps the
result accurate for arbitrarily long chains.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product as iproduct

import numpy as np


@dataclass
class IsingChainResult:
    N: int
    beta: float
    Z: float
    log_Z: float
    magnetization: np.ndarray          # shape (N,)   <sigma_i>
    correlation: np.ndarray            # shape (N,N)  <sigma_i sigma_j>
    susceptibility: np.ndarray         # shape (N,N)  chi_ij
    chi_total: float


def _normalize(vec: np.ndarray) -> tuple[np.ndarray, float]:
    """Rescale vec by its max-abs entry; return (rescaled vec, log of the scale removed)."""
    scale = np.max(np.abs(vec))
    if scale == 0.0 or not np.isfinite(scale):
        raise FloatingPointError("Degenerate transfer-matrix vector (all zeros or overflow).")
    return vec / scale, np.log(scale)


def _ratio(val: float, log_scale: float, log_Z: float) -> float:
    """Compute val * exp(log_scale - log_Z) without a log(0) warning when val == 0."""
    if val == 0.0:
        return 0.0
    return float(np.sign(val) * np.exp(np.log(np.abs(val)) + log_scale - log_Z))


def solve_ising_chain(J: float, h: np.ndarray, T: float, kB: float = 1.0) -> IsingChainResult:
    """
    Exactly solve the open 1D Ising chain with site-dependent field h (length N)
    via the transfer-matrix method, returning magnetization, spin-spin
    correlations and the full susceptibility matrix chi_ij = d<sigma_i>/dh_j.

    Parameters
    ----------
    J : nearest-neighbour coupling (scalar, same for every bond)
    h : 1D array of length N, external field at each site
    T : temperature (T > 0)
    kB : Boltzmann constant (default 1, i.e. natural units)
    """
    h = np.asarray(h, dtype=np.float64)
    if h.ndim != 1 or h.size == 0:
        raise ValueError("h must be a non-empty 1D array (one field value per site).")
    if T <= 0:
        raise ValueError("Temperature T must be positive.")

    N = h.size
    beta = 1.0 / (kB * T)

    # Bond factor (same 2x2 matrix for every bond since J is constant).
    eJ, emJ = np.exp(beta * J), np.exp(-beta * J)
    B = np.array([[eJ, emJ],
                  [emJ, eJ]])

    # Per-site field factors and their "sigma inserted" counterparts.
    eh = np.exp(beta * h)
    emh = np.exp(-beta * h)
    F = [np.diag([eh[i], emh[i]]) for i in range(N)]          # F[i] = F_{i+1} (0-indexed)
    SigmaF = [np.diag([eh[i], -emh[i]]) for i in range(N)]    # Sigma @ F_i

    # M[i] = "site i factor, then bond to site i+1" for i = 0..N-2 ; M[N-1] = F_N (no trailing bond)
    M = [F[i] @ B for i in range(N - 1)] + [F[N - 1]]
    SigmaM = [SigmaF[i] @ B for i in range(N - 1)] + [SigmaF[N - 1]]

    u_row = np.array([1.0, 1.0])
    u_col = np.array([1.0, 1.0])

    # --- normalized prefix products: P[k] ~ u^T M_1 ... M_k, k = 0..N ---
    P = [u_row.copy()]
    P_scale = [0.0]
    for k in range(N):
        nxt = P[-1] @ M[k]
        nxt, ds = _normalize(nxt)
        P.append(nxt)
        P_scale.append(P_scale[-1] + ds)

    # --- normalized suffix products: Q[k] ~ M_k ... M_N u, indexed k = N+1 down to 1 ---
    # store as dict keyed by k for readability; Q[N+1] = u (trivial, empty product)
    Q = {N + 1: u_col.copy()}
    Q_scale = {N + 1: 0.0}
    for k in range(N, 0, -1):
        nxt = M[k - 1] @ Q[k + 1]
        nxt, ds = _normalize(nxt)
        Q[k] = nxt
        Q_scale[k] = Q_scale[k + 1] + ds

    log_Z = np.log(P[N] @ u_col) + P_scale[N]
    with np.errstate(over="ignore"):
        Z = np.exp(log_Z)  # may legitimately overflow to inf for long/cold chains; log_Z stays finite

    # sanity cross-check: same Z obtained purely from the suffix side
    log_Z_alt = np.log(u_row @ Q[1]) + Q_scale[1]
    if not np.isclose(log_Z, log_Z_alt, rtol=1e-8, atol=1e-8):
        raise FloatingPointError(
            f"Internal inconsistency: log Z from prefix ({log_Z}) != from suffix ({log_Z_alt})."
        )

    # ---------------- magnetization <sigma_i>, i = 1..N (1-indexed) ----------------
    magnetization = np.empty(N)
    for i in range(1, N + 1):  # 1-indexed site
        vec = P[i - 1] @ SigmaM[i - 1]
        val = vec @ Q[i + 1]
        log_scale = P_scale[i - 1] + Q_scale[i + 1]
        magnetization[i - 1] = np.clip(_ratio(val, log_scale, log_Z), -1.0, 1.0)

    # ---------------- correlations <sigma_i sigma_j>, all i,j (1-indexed internally) ----------------
    correlation = np.eye(N)  # diagonal: <sigma_i^2> = 1 exactly
    for i in range(1, N + 1):
        V = P[i - 1] @ SigmaM[i - 1]
        run_scale = P_scale[i - 1]
        for j in range(i + 1, N + 1):
            if j > i + 1:
                V = V @ M[j - 2]  # incorporate site (j-1)'s factor + bond to j
                V, ds = _normalize(V)
                run_scale += ds
            Vj = V @ SigmaM[j - 1]
            val = Vj @ Q[j + 1]
            log_scale = run_scale + Q_scale[j + 1]
            corr = np.clip(_ratio(val, log_scale, log_Z), -1.0, 1.0)
            correlation[i - 1, j - 1] = corr
            correlation[j - 1, i - 1] = corr

    m_outer = np.outer(magnetization, magnetization)
    susceptibility = beta * (correlation - m_outer)
    np.fill_diagonal(susceptibility, beta * (1.0 - magnetization ** 2))
    chi_total = float(np.sum(susceptibility))

    return IsingChainResult(
        N=N, beta=beta, Z=float(Z), log_Z=float(log_Z),
        magnetization=magnetization, correlation=correlation,
        susceptibility=susceptibility, chi_total=chi_total,
    )


# --------------------------------------------------------------------------
# Brute-force exact reference (full enumeration over 2^N configurations).
# Used only to validate solve_ising_chain on small chains; O(2^N), N <~ 16.
# --------------------------------------------------------------------------
def _brute_force(J: float, h: np.ndarray, T: float, kB: float = 1.0) -> IsingChainResult:
    h = np.asarray(h, dtype=np.float64)
    N = h.size
    beta = 1.0 / (kB * T)

    Z = 0.0
    m_num = np.zeros(N)
    corr_num = np.zeros((N, N))
    for config in iproduct([1, -1], repeat=N):
        s = np.array(config, dtype=np.float64)
        E = -J * np.sum(s[:-1] * s[1:]) - np.sum(h * s)
        w = np.exp(-beta * E)
        Z += w
        m_num += w * s
        corr_num += w * np.outer(s, s)

    magnetization = m_num / Z
    correlation = corr_num / Z
    susceptibility = beta * (correlation - np.outer(magnetization, magnetization))
    np.fill_diagonal(susceptibility, beta * (1.0 - magnetization ** 2))
    chi_total = float(np.sum(susceptibility))

    return IsingChainResult(
        N=N, beta=beta, Z=float(Z), log_Z=float(np.log(Z)),
        magnetization=magnetization, correlation=correlation,
        susceptibility=susceptibility, chi_total=chi_total,
    )


def _self_test(seed: int = 0, trials: int = 25, max_N: int = 8) -> None:
    """Cross-check the transfer-matrix solver against brute-force enumeration."""
    rng = np.random.default_rng(seed)
    for t in range(trials):
        N = rng.integers(1, max_N + 1)
        J = rng.uniform(-2.0, 2.0)
        h = rng.uniform(-2.0, 2.0, size=N)
        T = rng.uniform(0.2, 5.0)

        exact = _brute_force(J, h, T)
        tm = solve_ising_chain(J, h, T)

        assert np.isclose(exact.Z, tm.Z, rtol=1e-8), (t, N, J, h, T, exact.Z, tm.Z)
        assert np.allclose(exact.magnetization, tm.magnetization, atol=1e-8), (t, "magnetization")
        assert np.allclose(exact.correlation, tm.correlation, atol=1e-8), (t, "correlation")
        assert np.allclose(exact.susceptibility, tm.susceptibility, atol=1e-6), (t, "susceptibility")
        assert np.isclose(exact.chi_total, tm.chi_total, atol=1e-6), (t, "chi_total")
    print(f"Self-test passed: {trials} random trials (N<={max_N}) match brute force exactly.")


if __name__ == "__main__":
    _self_test()

    # Demo: a domain-wall-like field profile on a 10-site chain.
    N = 10
    J = 1.0
    T = 2.0
    h = np.array([2.0, 1.5, 1.0, 0.5, 0.0, 0.0, -0.5, -1.0, -1.5, -2.0])

    result = solve_ising_chain(J, h, T)
    print("\nField profile h_i:      ", np.array2string(h, precision=3))
    print("Local magnetization <s_i>:", np.array2string(result.magnetization, precision=4))
    print("Diagonal chi_ii:          ", np.array2string(np.diag(result.susceptibility), precision=4))
    print("Total susceptibility chi_total =", result.chi_total)
