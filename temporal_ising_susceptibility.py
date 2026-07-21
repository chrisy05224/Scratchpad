"""
Temporal 1D Ising Susceptibility: Signal Sensitivity Matrix.

Fits an open-boundary, site-inhomogeneous 1D Ising chain (one site per
timestamp) to an observed +/-1 spin time series driven by a real-valued
signal time series, then returns the exact susceptibility matrix

    chi[k, l] = d<s_k>/d(signal_l) | K fixed

solved via the transfer-matrix method (no periodic wraparound: this chain
represents time, and time does not wrap around).

Important interpretive note: the transfer-matrix construction is an
equilibrium/undirected model, so <s_k s_l> = <s_l s_k> always, and the
returned chi is exactly symmetric (chi[k, l] == chi[l, k]), including
entries with l > k (signal *after* time k). chi[k, l] should NOT be read
as "signal at l causes a move in the target at k" -- it reflects the
model's equilibrium joint structure, not a causal/lead-lag relationship.
"""

import math
import warnings

import numpy as np
from scipy.optimize import minimize

_SIGN = np.array([1.0, -1.0])
_PARAM_BOUND = 30.0


# ---------------------------------------------------------------------------
# Section 2: model construction
# ---------------------------------------------------------------------------

def build_bond_weights(N):
    """w_L, w_R for bonds i = 0..N-2 (each length N-1)."""
    n_bonds = N - 1
    w_L = np.full(n_bonds, 0.5)
    w_R = np.full(n_bonds, 0.5)
    w_L[0] = 1.0
    w_R[-1] = 1.0
    return w_L, w_R


def build_transfer_matrices(K, B, w_L, w_R):
    """
    T[i](s, s') = exp[K_i s s' + w_L(i) B_i s + w_R(i) B_{i+1} s'],
    rows/cols ordered (s=+1, s=-1).

    K: shape (N-1,), B: shape (N,), w_L/w_R: shape (N-1,).
    Returns T: shape (N-1, 2, 2).
    """
    n_bonds = K.shape[0]
    T = np.empty((n_bonds, 2, 2))
    Bl = w_L * B[:-1]
    Br = w_R * B[1:]
    # Clip exponent arguments to stay within float64 range. A single bond's
    # weight saturating this hard is already an effectively-deterministic
    # bond (exp(700) vs exp(709) are both "certain" in any product), so this
    # only guards against overflow -- it does not change fitted behavior in
    # any regime the MLE would actually reach.
    clip = 700.0
    T[:, 0, 0] = np.exp(np.clip(K + Bl + Br, -clip, clip))
    T[:, 0, 1] = np.exp(np.clip(-K + Bl - Br, -clip, clip))
    T[:, 1, 0] = np.exp(np.clip(-K - Bl + Br, -clip, clip))
    T[:, 1, 1] = np.exp(np.clip(K - Bl - Br, -clip, clip))
    return T


# ---------------------------------------------------------------------------
# Section 3 / 8: prefix-suffix products with log-scale tracking
# ---------------------------------------------------------------------------

def prefix_suffix_products(T):
    """
    L[k] = T[0]...T[k-1] (L[0] = I), R[k] = T[k]...T[N-2] (R[N-1] = I).

    Both arrays are periodically rescaled (divided by their max entry) to
    avoid overflow over long chains; the corresponding cumulative log-scale
    is tracked separately (log-sum-exp-style stabilization). All entries of
    T, and hence of L and R, are strictly positive (they are exponentials),
    so no sign bookkeeping is needed at this stage.
    """
    n_bonds = T.shape[0]
    N = n_bonds + 1

    L = np.empty((N, 2, 2))
    logL_scale = np.zeros(N)
    L[0] = np.eye(2)
    for k in range(1, N):
        M = L[k - 1] @ T[k - 1]
        s = M.max()
        if not np.isfinite(s) or s <= 0:
            raise FloatingPointError(
                "Non-finite or non-positive value encountered while building "
                "prefix products; check K/B magnitudes."
            )
        L[k] = M / s
        logL_scale[k] = logL_scale[k - 1] + math.log(s)

    R = np.empty((N, 2, 2))
    logR_scale = np.zeros(N)
    R[N - 1] = np.eye(2)
    for k in range(N - 2, -1, -1):
        M = T[k] @ R[k + 1]
        s = M.max()
        if not np.isfinite(s) or s <= 0:
            raise FloatingPointError(
                "Non-finite or non-positive value encountered while building "
                "suffix products; check K/B magnitudes."
            )
        R[k] = M / s
        logR_scale[k] = logR_scale[k + 1] + math.log(s)

    return L, logL_scale, R, logR_scale


def log_partition_function(L, logL_scale):
    """
    Z = e^T L[N-1] e = sum of all entries of L[N-1] (already the full
    product T[0]@...@T[N-2] -- do not multiply by T[N-2] again).
    Returns log(Z).
    """
    total = L[-1].sum()
    if not np.isfinite(total) or total <= 0:
        raise FloatingPointError(
            "Partition function is zero/non-finite; the fitted parameters "
            "produced a degenerate model."
        )
    return math.log(total) + logL_scale[-1]


# ---------------------------------------------------------------------------
# Section 3: magnetization and correlations
# ---------------------------------------------------------------------------

def _col_sums(L):
    """e^T L[k] for every k -> shape (N, 2)."""
    return L.sum(axis=1)


def _row_sums(R):
    """R[k] e for every k -> shape (N, 2)."""
    return R.sum(axis=2)


def magnetization_all(L, logL_scale, R, logR_scale, logZ):
    """<s_k> for every k, shape (N,). Vectorized, log-domain stable."""
    cL = _col_sums(L)              # e^T L[k]
    cR = _row_sums(R)              # R[k] e
    v = cL * _SIGN                 # e^T L[k] sigma_z

    numer = v[:, 0] * cR[:, 0] + v[:, 1] * cR[:, 1]
    sign = np.sign(numer)
    with np.errstate(divide="ignore"):
        lognum = np.log(np.abs(numer)) + logL_scale + logR_scale
    mag = sign * np.exp(lognum - logZ)
    return np.clip(mag, -1.0, 1.0)


def adjacent_correlations(T, L, logL_scale, R, logR_scale, logZ):
    """<s_i s_{i+1}> for i = 0..N-2, shape (N-1,). Vectorized."""
    cL = _col_sums(L)
    cR = _row_sums(R)
    v = (cL * _SIGN)[:-1]          # i = 0..N-2
    u = (cR * _SIGN)[1:]           # i+1 = 1..N-1

    temp = np.einsum("ij,ijk->ik", v, T)
    numer = (temp * u).sum(axis=1)
    sign = np.sign(numer)
    with np.errstate(divide="ignore"):
        lognum = np.log(np.abs(numer)) + logL_scale[:-1] + logR_scale[1:]
    corr = sign * np.exp(lognum - logZ)
    return np.clip(corr, -1.0, 1.0)


def full_correlation_matrix(T, L, logL_scale, R, logR_scale, logZ):
    """
    <s_k s_l> for every (k, l), shape (N, N), exactly symmetric, diagonal
    == 1. O(N) Python-level steps, each numpy-vectorized across k, giving
    O(N^2) total work (Section 8's scheme, vectorized across the k index
    instead of looped).
    """
    N = L.shape[0]
    cL = _col_sums(L)
    cR = _row_sums(R)
    v = cL * _SIGN                 # shape (N, 2), row-vector side
    u = cR * _SIGN                 # shape (N, 2), column-vector side

    corr = np.eye(N)

    if N == 1:
        return corr

    # Running product M_{k,l} = T[k] @ ... @ T[l-1], maintained for all
    # active k simultaneously (rows), rescaled each step; row_logscale[k]
    # tracks the *total* log-scale of row k (base logL_scale[k] plus every
    # rescale increment applied since the row was created).
    Rmat = np.zeros((N, 2))
    row_logscale = np.zeros(N)
    m = 0

    for l in range(1, N):
        k_new = l - 1
        Rmat[m] = v[k_new] @ T[k_new]
        row_logscale[m] = logL_scale[k_new]
        m += 1

        active = Rmat[:m]
        numer = active[:, 0] * u[l, 0] + active[:, 1] * u[l, 1]
        sign = np.sign(numer)
        with np.errstate(divide="ignore"):
            lognum = np.log(np.abs(numer)) + row_logscale[:m] + logR_scale[l]
        vals = sign * np.exp(lognum - logZ)
        vals = np.clip(vals, -1.0, 1.0)
        corr[:m, l] = vals
        corr[l, :m] = vals

        if l < N - 1:
            newvals = active @ T[l]
            maxabs = np.max(np.abs(newvals))
            if maxabs > 0 and np.isfinite(maxabs):
                row_logscale[:m] += math.log(maxabs)
                newvals = newvals / maxabs
            elif not np.isfinite(maxabs):
                raise FloatingPointError(
                    "Non-finite value encountered while building the full "
                    "correlation matrix; check K/B magnitudes."
                )
            Rmat[:m] = newvals

    return corr


# ---------------------------------------------------------------------------
# Section 6: MLE
# ---------------------------------------------------------------------------

def _k_array(theta, j_model, sig_avg_bonds):
    if j_model == "constant":
        K, B_scale = theta
        K_arr = np.full(sig_avg_bonds.shape[0], K)
    else:
        K0, K1, B_scale = theta
        K_arr = K0 + K1 * sig_avg_bonds
    return K_arr, B_scale


def negative_log_likelihood_and_grad(theta, spins, signal, sig_avg_bonds,
                                      w_L, w_R, j_model):
    K_arr, B_scale = _k_array(theta, j_model, sig_avg_bonds)
    B_arr = B_scale * signal

    T = build_transfer_matrices(K_arr, B_arr, w_L, w_R)
    L, logL_scale, R, logR_scale = prefix_suffix_products(T)
    logZ = log_partition_function(L, logL_scale)

    mag = magnetization_all(L, logL_scale, R, logR_scale, logZ)
    adj_corr = adjacent_correlations(T, L, logL_scale, R, logR_scale, logZ)

    obs_ss = spins[:-1] * spins[1:]
    ell = np.sum(K_arr * obs_ss) + np.sum(B_arr * spins) - logZ

    dK_terms = obs_ss - adj_corr
    dB_scale = np.sum(signal * (spins - mag))

    if j_model == "constant":
        grad = np.array([np.sum(dK_terms), dB_scale])
    else:
        grad = np.array([
            np.sum(dK_terms),
            np.sum(sig_avg_bonds * dK_terms),
            dB_scale,
        ])

    return -ell, -grad


def fit(spins, signal, sig_avg_bonds, w_L, w_R, j_model):
    """Maximize the log-likelihood for the given j_model in {"constant",
    "signal_modulated"}. Returns (theta_hat, loglik, bic, n_params)."""
    N = spins.shape[0]

    obs_ss = spins[:-1] * spins[1:]
    mean_ss = np.clip(obs_ss.mean(), -0.999, 0.999)
    K_init = math.atanh(mean_ss)

    sig_std = signal.std()
    if sig_std > 1e-12:
        c = np.corrcoef(signal, spins)[0, 1]
        B_init = 0.0 if not np.isfinite(c) else float(np.clip(c, -3.0, 3.0))
    else:
        B_init = 0.0

    if j_model == "constant":
        theta0 = np.array([K_init, B_init])
        bounds = [(-_PARAM_BOUND, _PARAM_BOUND)] * 2
        n_params = 2
    else:
        theta0 = np.array([K_init, 0.0, B_init])
        bounds = [(-_PARAM_BOUND, _PARAM_BOUND)] * 3
        n_params = 3

    res = minimize(
        negative_log_likelihood_and_grad,
        theta0,
        args=(spins, signal, sig_avg_bonds, w_L, w_R, j_model),
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
    )

    loglik = -res.fun
    bic = -2.0 * loglik + n_params * math.log(N)
    return res.x, loglik, bic, n_params


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------

def susceptibility_matrix(spins, signal, standardize_signal=True,
                           j_model="auto", return_diagnostics=False):
    """
    See module docstring. chi[k, l] = d<s_k>/d(signal_l), at fixed fitted
    coupling K. chi is exactly symmetric regardless of j_model (Section 7);
    it is a correlational, not causal, sensitivity.

    j_model: "constant" | "signal_modulated" | "auto" (fit both, pick by BIC)
    """
    spins = np.asarray(spins, dtype=np.float64)
    signal = np.asarray(signal, dtype=np.float64)

    if spins.ndim != 1 or signal.ndim != 1:
        raise ValueError("spins and signal must both be 1D arrays.")
    if spins.shape[0] != signal.shape[0]:
        raise ValueError("spins and signal must have the same length.")
    N = spins.shape[0]
    if N < 2:
        raise ValueError("N must be >= 2.")
    if not np.all((spins == 1.0) | (spins == -1.0)):
        raise ValueError("spins must contain only +1.0 or -1.0 values.")
    if not np.all(np.isfinite(signal)):
        raise ValueError("signal must not contain NaN or Inf values.")
    if j_model not in ("auto", "constant", "signal_modulated"):
        raise ValueError('j_model must be "auto", "constant", or "signal_modulated".')
    if N < 20:
        warnings.warn(
            f"N={N} is small; the coupling fit will be noisy.",
            stacklevel=2,
        )

    signal_mean = float(signal.mean())
    signal_std = float(signal.std())
    if standardize_signal:
        if signal_std < 1e-12:
            warnings.warn(
                "signal has ~zero standard deviation; skipping standardization.",
                stacklevel=2,
            )
            signal_used = signal - signal_mean
        else:
            signal_used = (signal - signal_mean) / signal_std
    else:
        signal_used = signal

    w_L, w_R = build_bond_weights(N)
    sig_avg_bonds = 0.5 * (signal_used[:-1] + signal_used[1:])

    diagnostics = {
        "signal_mean": signal_mean,
        "signal_std": signal_std,
        "standardize_signal": standardize_signal,
        "symmetric_note": (
            "chi is symmetric by construction (equilibrium model); chi[k, l] "
            "for l > k is not a causal 'signal at l affects target at k' "
            "statement."
        ),
    }

    fits = {}
    if j_model in ("constant", "auto"):
        fits["constant"] = fit(spins, signal_used, sig_avg_bonds, w_L, w_R, "constant")
    if j_model in ("signal_modulated", "auto"):
        fits["signal_modulated"] = fit(spins, signal_used, sig_avg_bonds, w_L, w_R, "signal_modulated")

    if j_model == "auto":
        bic_const = fits["constant"][2]
        bic_mod = fits["signal_modulated"][2]
        selected = "constant" if bic_const <= bic_mod else "signal_modulated"
        diagnostics["bic_constant"] = bic_const
        diagnostics["bic_signal_modulated"] = bic_mod
    else:
        selected = j_model

    theta_hat, loglik, bic, n_params = fits[selected]
    K_arr, B_scale = _k_array(theta_hat, selected, sig_avg_bonds)
    B_arr = B_scale * signal_used

    diagnostics["j_model_selected"] = selected
    diagnostics["loglik"] = loglik
    diagnostics["bic_selected"] = bic
    diagnostics["n_params"] = n_params
    diagnostics["B_scale"] = float(B_scale)
    if selected == "constant":
        diagnostics["K"] = float(theta_hat[0])
    else:
        diagnostics["K0"] = float(theta_hat[0])
        diagnostics["K1"] = float(theta_hat[1])

    T = build_transfer_matrices(K_arr, B_arr, w_L, w_R)
    L, logL_scale, R, logR_scale = prefix_suffix_products(T)
    logZ = log_partition_function(L, logL_scale)

    mag = magnetization_all(L, logL_scale, R, logR_scale, logZ)
    corr = full_correlation_matrix(T, L, logL_scale, R, logR_scale, logZ)

    chi_cov = corr - np.outer(mag, mag)
    chi = B_scale * chi_cov
    chi = 0.5 * (chi + chi.T)  # enforce exact symmetry against fp roundoff

    if return_diagnostics:
        return chi, diagnostics
    return chi
