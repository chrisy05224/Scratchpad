import numpy as np


def generate_correlated_lognormal_timeseries(
    correlation,
    upper1, lower1,
    upper2, lower2,
    n=1000,
    sigma=0.5,
    random_state=None,
):
    """
    Generate two lognormally-distributed timeseries, each bounded to its own
    [lower, upper] range, whose Pearson correlation matches `correlation`.

    Method (Gaussian copula / NORTA):
      1. Draw two standard normals Z1, Z2 with a chosen correlation rho_z.
      2. Exponentiate them (lognormal transform): X_i = exp(sigma * Z_i).
      3. Min-max rescale each series independently into its own [lower, upper]
         bound. This is an affine, monotonically-increasing map per series,
         so it preserves Pearson correlation exactly.

    The correlation of two lognormal variables is a nonlinear function of the
    underlying normal correlation rho_z:
        rho_ln = (exp(rho_z * s1 * s2) - 1) / sqrt((exp(s1^2)-1)(exp(s2^2)-1))
    so rho_z is solved for by inverting this relation given the requested
    `correlation`. When s1 == s2 (used here), rho_ln = +1 is exactly
    achievable for any sigma, but strongly negative correlations are only
    achievable with a small sigma (a mathematical property of lognormal
    variables, not a bug: two lognormals derived from perfectly
    anti-correlated normals are never perfectly anti-correlated themselves,
    since exp() is convex). To stay robust for high |correlation|, `sigma`
    is adaptively shrunk until the target becomes solvable; as a last resort
    (target essentially -1) the solved rho_z is clipped to [-1, 1], which
    gets arbitrarily close to -1 but never exactly there.

    Note: larger `sigma` produces heavier-tailed, more visibly skewed data,
    but also increases the sample-to-sample variance of the achieved
    correlation for a fixed `n` (heavy tails => noisier Pearson correlation
    estimates). Increase `n` if you need tight correlation matching with a
    large `sigma`.

    Parameters
    ----------
    correlation : float
        Target Pearson correlation in [-1, 1].
    upper1, lower1 : float
        Bounds for timeseries 1 (upper1 > lower1).
    upper2, lower2 : float
        Bounds for timeseries 2 (upper2 > lower2).
    n : int
        Number of timesteps to generate.
    sigma : float
        Initial lognormal shape parameter (std dev of the underlying normal).
        Larger sigma => more skewed timeseries, but a smaller achievable
        range of negative correlations (auto-shrunk internally if needed).
    random_state : int, np.random.Generator, or None
        Seed / generator for reproducibility.

    Returns
    -------
    ts1, ts2 : np.ndarray
        Two correlated lognormal timeseries of length n, each within its
        specified [lower, upper] bound.
    """

    # ---- validation ----
    if not -1.0 <= correlation <= 1.0:
        raise ValueError(f"correlation must be in [-1, 1], got {correlation}")
    if upper1 <= lower1:
        raise ValueError("upper1 must be strictly greater than lower1")
    if upper2 <= lower2:
        raise ValueError("upper2 must be strictly greater than lower2")
    if n < 2:
        raise ValueError("n must be >= 2")
    if sigma <= 0:
        raise ValueError("sigma must be > 0")

    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)

    def solve_rho_z(rho_target, s):
        # s1 == s2 == s
        denom = np.exp(s ** 2) - 1.0  # = sqrt((exp(s^2)-1)^2)
        arg = 1.0 + rho_target * denom
        if arg <= 0:
            return None
        rz = np.log(arg) / (s * s)
        if abs(rz) > 1.0:
            return None
        return rz

    # Adaptively shrink sigma until the requested correlation is achievable
    # under the equal-sigma lognormal correlation formula.
    s = float(sigma)
    rho_z = solve_rho_z(correlation, s)
    tries = 0
    while rho_z is None and tries < 200 and s > 1e-6:
        s *= 0.85
        rho_z = solve_rho_z(correlation, s)
        tries += 1

    if rho_z is None:
        # Extreme edge case (correlation essentially -1): fall back to the
        # smallest sigma tried and clip rho_z into the valid range. This
        # yields a correlation arbitrarily close to (but never exactly) -1.
        s = max(s, 1e-6)
        denom = np.exp(s ** 2) - 1.0
        arg = max(1.0 + correlation * denom, 1e-15)
        rho_z = np.log(arg) / (s * s)

    rho_z = float(np.clip(rho_z, -1.0, 1.0))
    sigma1 = sigma2 = s

    # ---- correlated standard normals ----
    z1 = rng.standard_normal(n)
    eps = rng.standard_normal(n)
    z2 = rho_z * z1 + np.sqrt(max(0.0, 1.0 - rho_z ** 2)) * eps

    # ---- lognormal transform ----
    x1 = np.exp(sigma1 * z1)
    x2 = np.exp(sigma2 * z2)

    # ---- rescale into [lower, upper] (affine map => preserves correlation) ----
    def rescale(x, lo, hi):
        x_min, x_max = x.min(), x.max()
        if x_max == x_min:
            return np.full_like(x, (lo + hi) / 2.0)
        return lo + (x - x_min) * (hi - lo) / (x_max - x_min)

    ts1 = rescale(x1, lower1, upper1)
    ts2 = rescale(x2, lower2, upper2)

    return ts1, ts2


if __name__ == "__main__":
    # Quick self-test / demo
    targets = [-0.999, -0.99, -0.95, -0.8, -0.5, 0.0, 0.5, 0.8, 0.95, 0.99, 1.0, -1.0]
    for target in targets:
        ts1, ts2 = generate_correlated_lognormal_timeseries(
            target, upper1=100, lower1=10, upper2=50, lower2=5,
            n=20000, random_state=42,
        )
        assert ts1.min() >= 10 - 1e-9 and ts1.max() <= 100 + 1e-9
        assert ts2.min() >= 5 - 1e-9 and ts2.max() <= 50 + 1e-9
        achieved = np.corrcoef(ts1, ts2)[0, 1]
        print(f"target={target:+.3f}  achieved={achieved:+.4f}")
