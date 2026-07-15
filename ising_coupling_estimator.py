"""
Inverse Ising inference with automatic estimator selection.

Given a 2D array of observed spin configurations (n_rows samples x n_cols
sites), estimate the symmetric column-to-column coupling matrix J and the
local fields h of the Ising model

    P(s) = (1/Z) * exp( sum_{i<j} J_ij s_i s_j + sum_i h_i s_i )

Two estimators are implemented:

  * Exact maximum likelihood (MLE), via brute-force enumeration of all
    2**n_cols configurations to get the partition function Z and its
    gradient in closed form, optimized with L-BFGS-B. This is exact but
    its cost is exponential in n_cols, so it is only used for small
    lattices.

  * Pseudolikelihood (Besag 1975; Ravikumar/Wainwright/Lafferty 2010;
    Aurell & Ekeberg 2012). For each site i, s_i is regressed against
    every other spin with a logistic model; the two one-sided estimates
    J_ij and J_ji are averaged into a symmetric matrix. Cost is linear in
    n_rows and polynomial in n_cols, so it scales to large lattices.

Method selection (as specified):
  - n_cols > 15                -> pseudolikelihood
  - n_rows >= 250               -> pseudolikelihood
  - otherwise (n_cols <= 15 and n_rows < 250) -> exact MLE

Core API
--------
estimate_couplings(spins, method="auto") -> (J, h, method_used)
    spins : (n_rows, n_cols) array of +/-1 (or 0/1) spin values
    J     : (n_cols, n_cols) symmetric coupling matrix, zero diagonal
    h     : (n_cols,) local fields
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

# ----------------------------------------------------------------------
# Input handling
# ----------------------------------------------------------------------


def _sanitize_spins(spins: np.ndarray) -> np.ndarray:
    spins = np.asarray(spins)
    if spins.ndim != 2:
        raise ValueError("spins must be a 2D array of shape (n_rows, n_cols)")

    spins = spins.astype(np.float64, copy=True)
    unique_vals = np.unique(spins)

    if unique_vals.size == 0:
        raise ValueError("spins array is empty")
    if np.all(np.isin(unique_vals, [0.0, 1.0])):
        spins = 2.0 * spins - 1.0
    elif not np.all(np.isin(unique_vals, [-1.0, 1.0])):
        raise ValueError("spins must contain only -1/+1 (or 0/1) values")

    return np.ascontiguousarray(spins)


def choose_method(n_rows: int, n_cols: int) -> str:
    """Method-selection rule.

    Pseudolikelihood is used whenever exact enumeration would be too
    costly: either because there are too many sites (2**n_cols blows up)
    or too many samples (n_rows >= 250 is treated as "large enough that
    the cheaper estimator is preferred"). Exact MLE is only used for the
    small regime: n_cols <= 15 and n_rows < 250.
    """
    if n_cols > 15 or n_rows >= 250:
        return "pseudolikelihood"
    return "mle"


# ----------------------------------------------------------------------
# Exact MLE via brute-force enumeration (small n_cols only)
# ----------------------------------------------------------------------

_MAX_MLE_COLS = 20  # hard safety cap: 2**20 ~ 1e6 configs is already a lot


def _all_configs(n: int) -> np.ndarray:
    """All 2**n spin configurations as a (2**n, n) array of +/-1."""
    codes = np.arange(2 ** n, dtype=np.int64)[:, None]
    bits = (codes >> np.arange(n, dtype=np.int64)[None, :]) & 1
    return (bits * 2 - 1).astype(np.float64)


def _neg_log_likelihood_and_grad(
    theta: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    n_cols: int,
    S_all: np.ndarray,
    empirical_ss: np.ndarray,
    empirical_s: np.ndarray,
    n_rows: int,
) -> Tuple[float, np.ndarray]:
    n_pairs = pair_i.shape[0]
    J_vals = theta[:n_pairs]
    h = theta[n_pairs:]

    J_full = np.zeros((n_cols, n_cols), dtype=np.float64)
    J_full[pair_i, pair_j] = J_vals
    J_full[pair_j, pair_i] = J_vals

    # energies[s] = sum_{i<j} J_ij si sj + h . s = 0.5 * s^T J_full s + h.s
    energies = 0.5 * np.sum((S_all @ J_full) * S_all, axis=1) + S_all @ h
    log_z = logsumexp(energies)
    weights = np.exp(energies - log_z)  # model probabilities, sum to 1

    e_model_s = weights @ S_all  # (n_cols,)
    e_model_ss = S_all.T @ (S_all * weights[:, None])  # (n_cols, n_cols)

    data_term = np.sum(J_vals * empirical_ss[pair_i, pair_j]) + h @ empirical_s
    log_likelihood = data_term - n_rows * log_z

    grad_J = empirical_ss[pair_i, pair_j] - n_rows * e_model_ss[pair_i, pair_j]
    grad_h = empirical_s - n_rows * e_model_s
    grad = np.concatenate([grad_J, grad_h])

    return -log_likelihood, -grad


def fit_mle(
    spins: np.ndarray, tol: float = 1e-8, max_iter: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    """Exact maximum-likelihood estimate of J and h via full enumeration.

    Only tractable for small n_cols (2**n_cols configurations are
    enumerated explicitly); raises if n_cols exceeds a safety cap.
    """
    spins = _sanitize_spins(spins)
    n_rows, n_cols = spins.shape
    if n_cols > _MAX_MLE_COLS:
        raise ValueError(
            f"fit_mle: n_cols={n_cols} is too large for brute-force "
            f"enumeration (cap is {_MAX_MLE_COLS}); use fit_pseudolikelihood instead"
        )

    pair_i, pair_j = np.triu_indices(n_cols, k=1)
    S_all = _all_configs(n_cols)

    empirical_ss = spins.T @ spins
    empirical_s = spins.sum(axis=0)

    n_params = pair_i.shape[0] + n_cols
    theta0 = np.zeros(n_params, dtype=np.float64)

    result = minimize(
        _neg_log_likelihood_and_grad,
        theta0,
        args=(pair_i, pair_j, n_cols, S_all, empirical_ss, empirical_s, n_rows),
        jac=True,
        method="L-BFGS-B",
        tol=tol,
        options={"maxiter": max_iter},
    )

    n_pairs = pair_i.shape[0]
    J = np.zeros((n_cols, n_cols), dtype=np.float64)
    J[pair_i, pair_j] = result.x[:n_pairs]
    J[pair_j, pair_i] = result.x[:n_pairs]
    h = result.x[n_pairs:]
    return J, h


# ----------------------------------------------------------------------
# Pseudolikelihood (scales to large n_rows / n_cols)
# ----------------------------------------------------------------------


def _neg_log_pseudo_likelihood_and_grad(
    params: np.ndarray, s_i: np.ndarray, s_other: np.ndarray, l2_reg: float
) -> Tuple[float, np.ndarray]:
    """Value and gradient of the (L2-regularized) negative pseudo
    log-likelihood of one site.

    params = [h, J_0, J_1, ..., J_{n-2}]
    s_i     : (n_rows,)              spin of the target site
    s_other : (n_rows, n_cols - 1)   spins of all other sites
    """
    h = params[0]
    J = params[1:]

    local_field = h + s_other @ J
    x = -2.0 * s_i * local_field

    nll = np.logaddexp(0.0, x).mean()
    obj = nll + l2_reg * np.dot(J, J)

    r = expit(x)
    common = -2.0 * s_i * r

    n_rows = s_i.shape[0]
    grad_h = common.mean()
    grad_J = (s_other.T @ common) / n_rows + 2.0 * l2_reg * J

    grad = np.empty_like(params)
    grad[0] = grad_h
    grad[1:] = grad_J
    return obj, grad


def _fit_single_site(
    spins: np.ndarray, site: int, l2_reg: float, tol: float, max_iter: int
) -> Tuple[float, np.ndarray, np.ndarray]:
    n_cols = spins.shape[1]
    mask = np.ones(n_cols, dtype=bool)
    mask[site] = False

    s_i = spins[:, site]
    s_other = np.ascontiguousarray(spins[:, mask])

    x0 = np.zeros(n_cols, dtype=np.float64)
    result = minimize(
        _neg_log_pseudo_likelihood_and_grad,
        x0,
        args=(s_i, s_other, l2_reg),
        jac=True,
        method="L-BFGS-B",
        tol=tol,
        options={"maxiter": max_iter},
    )
    return result.x[0], result.x[1:], mask


def fit_pseudolikelihood(
    spins: np.ndarray,
    l2_reg: float = 0.01,
    tol: float = 1e-6,
    max_iter: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pseudolikelihood estimate of J and h.

    Fits an independent L2-regularized logistic regression per site
    (each site's spin regressed on every other site's spin), then
    symmetrizes the two one-sided estimates of every pair. Cost is
    O(n_cols) independent fits, each O(n_rows * n_cols) per iteration,
    so this scales to large n_rows and moderately large n_cols.
    """
    spins = _sanitize_spins(spins)
    n_rows, n_cols = spins.shape
    if n_rows < 2:
        raise ValueError("need at least 2 rows to estimate couplings")

    J_rows = np.zeros((n_cols, n_cols), dtype=np.float64)
    h = np.zeros(n_cols, dtype=np.float64)

    for site in range(n_cols):
        h_i, J_i, mask = _fit_single_site(spins, site, l2_reg, tol, max_iter)
        h[site] = h_i
        J_rows[site, mask] = J_i

    J = 0.5 * (J_rows + J_rows.T)
    np.fill_diagonal(J, 0.0)
    return J, h


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------


def estimate_couplings(
    spins: np.ndarray,
    method: str = "auto",
    mle_kwargs: Optional[dict] = None,
    pseudo_kwargs: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Estimate the column:column coupling matrix J (and local fields h)
    of a 2D Ising model from observed spin configurations.

    Parameters
    ----------
    spins : array, shape (n_rows, n_cols)
        Observed spin configurations. Values may be +/-1 or 0/1.
    method : {"auto", "mle", "pseudolikelihood"}
        "auto" (default) applies the selection rule in `choose_method`:
        exact MLE for n_cols <= 15 and n_rows < 250, pseudolikelihood
        otherwise.
    mle_kwargs, pseudo_kwargs : dict, optional
        Extra keyword arguments forwarded to `fit_mle` /
        `fit_pseudolikelihood` respectively.

    Returns
    -------
    J : array, shape (n_cols, n_cols)
        Symmetric coupling matrix, zero diagonal.
    h : array, shape (n_cols,)
        Estimated local field at each site.
    method_used : str
        Either "mle" or "pseudolikelihood".
    """
    spins = _sanitize_spins(spins)
    n_rows, n_cols = spins.shape

    if method == "auto":
        method_used = choose_method(n_rows, n_cols)
    elif method in ("mle", "pseudolikelihood"):
        method_used = method
    else:
        raise ValueError(f"unknown method: {method!r}")

    if method_used == "mle":
        J, h = fit_mle(spins, **(mle_kwargs or {}))
    else:
        J, h = fit_pseudolikelihood(spins, **(pseudo_kwargs or {}))

    return J, h, method_used


# ----------------------------------------------------------------------
# Demo / sanity check: build a lattice, simulate spins from it with a
# Metropolis sampler, and recover J from the samples for both the small
# (MLE) and large (pseudolikelihood) regimes.
# ----------------------------------------------------------------------


def build_lattice_J(L: int, J0: float = 0.4, periodic: bool = True) -> np.ndarray:
    """Nearest-neighbor coupling matrix for an L x L 2D Ising lattice."""
    n = L * L

    def idx(r: int, c: int) -> int:
        if periodic:
            return (r % L) * L + (c % L)
        return r * L + c

    J = np.zeros((n, n), dtype=np.float64)
    for r in range(L):
        for c in range(L):
            i = idx(r, c)
            for nr, nc in ((r, c + 1), (r + 1, c)):
                if not periodic and (nr >= L or nc >= L):
                    continue
                j = idx(nr, nc)
                J[i, j] = J0
                J[j, i] = J0
    return J


def simulate_ising(
    J: np.ndarray,
    h: Optional[np.ndarray] = None,
    n_samples: int = 4000,
    burn_in: int = 2000,
    thin: int = 10,
    beta: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Metropolis sampler for a general Ising model. Returns
    (n_samples, n_sites) array of +/-1 spins."""
    rng = np.random.default_rng() if rng is None else rng
    n = J.shape[0]
    if h is None:
        h = np.zeros(n)

    spins = rng.choice(np.array([-1.0, 1.0]), size=n)
    samples = np.empty((n_samples, n), dtype=np.float64)

    total_sweeps = burn_in + n_samples * thin
    collected = 0
    for sweep in range(total_sweeps):
        flip_order = rng.integers(0, n, size=n)
        for i in flip_order:
            local_field = h[i] + J[i] @ spins
            dE = 2.0 * spins[i] * local_field
            if dE <= 0.0 or rng.random() < np.exp(-beta * dE):
                spins[i] = -spins[i]
        if sweep >= burn_in and (sweep - burn_in) % thin == 0:
            samples[collected] = spins
            collected += 1

    return samples


def _report(J_true: np.ndarray, J_est: np.ndarray, method_used: str) -> None:
    mask = ~np.eye(J_true.shape[0], dtype=bool)
    err = np.abs(J_est - J_true)[mask]
    corr = np.corrcoef(J_true[mask], J_est[mask])[0, 1]
    print(f"  method used: {method_used}")
    print(f"  mean |J_est - J_true| (off-diagonal): {err.mean():.4f}")
    print(f"  max  |J_est - J_true| (off-diagonal): {err.max():.4f}")
    print(f"  correlation(J_true, J_est) off-diagonal: {corr:.4f}")


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    print("=== Small regime: 3x3 lattice (9 cols), 200 rows -> expect MLE ===")
    L_small = 3
    J_true_small = build_lattice_J(L_small, J0=0.5)
    spins_small = simulate_ising(J_true_small, n_samples=200, burn_in=1000, thin=5, rng=rng)
    J_est_small, h_est_small, method_small = estimate_couplings(spins_small)
    _report(J_true_small, J_est_small, method_small)

    print("\n=== Large regime: 5000 rows, 20 cols -> expect pseudolikelihood ===")
    n_cols_large = 20
    J_true_large = build_lattice_J(int(np.ceil(np.sqrt(n_cols_large))), J0=0.4)[:n_cols_large, :n_cols_large]
    J_true_large = 0.5 * (J_true_large + J_true_large.T)
    np.fill_diagonal(J_true_large, 0.0)
    spins_large = simulate_ising(J_true_large, n_samples=5000, burn_in=1000, thin=3, rng=rng)
    J_est_large, h_est_large, method_large = estimate_couplings(spins_large)
    _report(J_true_large, J_est_large, method_large)
