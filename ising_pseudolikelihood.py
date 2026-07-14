"""
Inverse Ising inference via pseudolikelihood maximization.

Given observed spin configurations sampled from an (unknown) Ising model,
estimate the full pairwise coupling matrix J (every J_ij between every
pair of sites), using the pseudolikelihood method (Besag 1975; applied to
inverse Ising by Ravikumar/Wainwright/Lafferty 2010, Aurell & Ekeberg
2012). For each site i, s_i is regressed against all other spins with an
L2-regularized logistic model; the two one-sided estimates J_ij and J_ji
are then averaged to produce a symmetric matrix.

Core API
--------
estimate_J_pseudolikelihood(spins, l2_reg=0.01, n_jobs=1) -> (J, h)
    spins : (n_samples, n_sites) array of +/-1 (or 0/1) spin values
    J     : (n_sites, n_sites) symmetric coupling matrix, zero diagonal
    h     : (n_sites,) local fields

A small demo at the bottom builds a 2D nearest-neighbor Ising lattice,
simulates spin configurations from it with Metropolis sampling, and
recovers the coupling matrix from the samples to sanity-check the
estimator.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

# ----------------------------------------------------------------------
# Core objective: negative log pseudolikelihood for a single site,
# fully vectorized (no per-sample Python loops) so it scales to large
# numbers of samples/sites.
# ----------------------------------------------------------------------


def _neg_log_pseudo_likelihood_and_grad(
    params: np.ndarray, s_i: np.ndarray, s_other: np.ndarray, l2_reg: float
) -> Tuple[float, np.ndarray]:
    """Value and gradient of the (regularized) negative pseudo log-
    likelihood of one site, as a function of its local field h and its
    couplings J to every other site.

    params = [h, J_0, J_1, ..., J_{n-2}]  (length n_sites)
    s_i     : (n_samples,)              spin of the target site
    s_other : (n_samples, n_sites - 1)  spins of all other sites
    """
    h = params[0]
    J = params[1:]

    local_field = h + s_other @ J  # (n_samples,)
    x = -2.0 * s_i * local_field

    # nll_sample = log(1 + exp(x)) = softplus(x), numerically stable form
    nll = np.logaddexp(0.0, x).mean()
    obj = nll + l2_reg * np.dot(J, J)

    r = expit(x)  # sigmoid(x) = d(nll_sample)/d(local_field) factor
    common = -2.0 * s_i * r  # (n_samples,)

    n_samples = s_i.shape[0]
    grad_h = common.mean()
    grad_J = (s_other.T @ common) / n_samples + 2.0 * l2_reg * J

    grad = np.empty_like(params)
    grad[0] = grad_h
    grad[1:] = grad_J
    return obj, grad


def _fit_single_site(
    spins: np.ndarray, site: int, l2_reg: float, tol: float, max_iter: int
) -> Tuple[int, float, np.ndarray, np.ndarray]:
    n_sites = spins.shape[1]
    mask = np.ones(n_sites, dtype=bool)
    mask[site] = False

    s_i = spins[:, site]
    s_other = np.ascontiguousarray(spins[:, mask])

    x0 = np.zeros(n_sites, dtype=np.float64)  # [h, J_others...]
    res = minimize(
        _neg_log_pseudo_likelihood_and_grad,
        x0,
        args=(s_i, s_other, l2_reg),
        jac=True,
        method="L-BFGS-B",
        tol=tol,
        options={"maxiter": max_iter},
    )
    return site, res.x[0], res.x[1:], mask


# ----------------------------------------------------------------------
# Multiprocessing plumbing: workers hold one copy of `spins` (set once
# via the pool initializer) instead of re-pickling it for every site.
# ----------------------------------------------------------------------

_worker_spins: Optional[np.ndarray] = None


def _init_worker(spins: np.ndarray) -> None:
    global _worker_spins
    _worker_spins = spins


def _fit_site_worker(args):
    site, l2_reg, tol, max_iter = args
    assert _worker_spins is not None
    return _fit_single_site(_worker_spins, site, l2_reg, tol, max_iter)


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def _sanitize_spins(spins: np.ndarray) -> np.ndarray:
    spins = np.asarray(spins)
    if spins.ndim != 2:
        raise ValueError("spins must be a 2D array of shape (n_samples, n_sites)")

    spins = spins.astype(np.float64, copy=True)
    unique_vals = np.unique(spins)

    if np.array_equal(unique_vals, [0.0, 1.0]) or np.array_equal(unique_vals, [0.0]) or np.array_equal(unique_vals, [1.0]):
        spins = 2.0 * spins - 1.0
    elif not np.all(np.isin(unique_vals, [-1.0, 1.0])):
        raise ValueError("spins must contain only -1/+1 (or 0/1) values")

    return np.ascontiguousarray(spins)


def estimate_J_pseudolikelihood(
    spins: np.ndarray,
    l2_reg: float = 0.01,
    tol: float = 1e-6,
    max_iter: int = 500,
    n_jobs: int = 1,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate the full inter-site coupling matrix J of an Ising model
    from observed spin configurations, via L2-regularized pseudolikelihood.

    Parameters
    ----------
    spins : array, shape (n_samples, n_sites)
        Observed spin configurations. Values may be +/-1 or 0/1.
    l2_reg : float
        L2 penalty on the couplings (not on the local fields). Larger
        values shrink J towards zero; helps when n_samples is small
        relative to n_sites.
    tol, max_iter : optimizer settings passed to L-BFGS-B for each site.
    n_jobs : int
        Number of sites to fit in parallel via separate processes.
        n_jobs=1 (default) runs sequentially in-process. Use
        n_jobs=os.cpu_count() for large lattices, since each site's fit
        is independent ("embarrassingly parallel").
    verbose : bool
        Print progress every 10% of sites completed.

    Returns
    -------
    J : array, shape (n_sites, n_sites)
        Symmetric coupling matrix with zero diagonal. J[i, j] is the
        average of the one-sided pseudolikelihood estimates J_ij and J_ji.
    h : array, shape (n_sites,)
        Estimated local field at each site.
    """
    spins = _sanitize_spins(spins)
    n_samples, n_sites = spins.shape
    if n_samples < 2:
        raise ValueError("need at least 2 samples to estimate couplings")

    J_rows = np.zeros((n_sites, n_sites), dtype=np.float64)
    h = np.zeros(n_sites, dtype=np.float64)

    def _store(site, h_i, J_i, mask):
        h[site] = h_i
        J_rows[site, mask] = J_i

    if n_jobs is None or n_jobs <= 1:
        for i in range(n_sites):
            site, h_i, J_i, mask = _fit_single_site(spins, i, l2_reg, tol, max_iter)
            _store(site, h_i, J_i, mask)
            if verbose and n_sites >= 10 and (i + 1) % max(1, n_sites // 10) == 0:
                print(f"  fitted {i + 1}/{n_sites} sites")
    else:
        n_jobs = min(n_jobs, n_sites, os.cpu_count() or n_jobs)
        tasks = [(i, l2_reg, tol, max_iter) for i in range(n_sites)]
        with ProcessPoolExecutor(
            max_workers=n_jobs, initializer=_init_worker, initargs=(spins,)
        ) as pool:
            for count, (site, h_i, J_i, mask) in enumerate(
                pool.map(_fit_site_worker, tasks, chunksize=max(1, n_sites // (4 * n_jobs)))
            ):
                _store(site, h_i, J_i, mask)
                if verbose and n_sites >= 10 and (count + 1) % max(1, n_sites // 10) == 0:
                    print(f"  fitted {count + 1}/{n_sites} sites")

    J = 0.5 * (J_rows + J_rows.T)
    np.fill_diagonal(J, 0.0)
    return J, h


# ----------------------------------------------------------------------
# Demo: build a lattice, simulate spins on it, recover J from the
# samples. This simulator is a plain Metropolis sampler meant only to
# produce demo/test data; it is not part of the (vectorized) estimator
# above, which is the piece intended to scale to large n_sites.
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
            neighbors = [(r, c + 1), (r + 1, c)]
            for nr, nc in neighbors:
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
    """Metropolis sampler for a general Ising model with coupling matrix
    J and local fields h. Returns (n_samples, n_sites) array of +/-1 spins.
    """
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


if __name__ == "__main__":
    L = 5
    J0 = 0.4
    print(f"Building a {L}x{L} periodic nearest-neighbor Ising lattice (J0={J0})")
    J_true = build_lattice_J(L, J0=J0)

    print("Simulating spin configurations with Metropolis sampling...")
    rng = np.random.default_rng(0)
    spins = simulate_ising(J_true, n_samples=4000, burn_in=2000, thin=10, beta=1.0, rng=rng)

    print("Estimating J via L2-regularized pseudolikelihood...")
    J_est, h_est = estimate_J_pseudolikelihood(spins, l2_reg=0.01, n_jobs=1, verbose=True)

    mask_off_diag = ~np.eye(J_true.shape[0], dtype=bool)
    err = np.abs(J_est - J_true)[mask_off_diag]
    corr = np.corrcoef(J_true[mask_off_diag], J_est[mask_off_diag])[0, 1]

    print(f"\nMean |J_est - J_true| (off-diagonal): {err.mean():.4f}")
    print(f"Max  |J_est - J_true| (off-diagonal): {err.max():.4f}")
    print(f"Correlation(J_true, J_est) off-diagonal: {corr:.4f}")
    print("\nEstimated J matrix:")
    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    print(J_est)
