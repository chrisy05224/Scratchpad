"""
Example usage of temporal_ising_susceptibility.susceptibility_matrix.

Run: python3 example_usage.py
"""

import numpy as np
from temporal_ising_susceptibility import susceptibility_matrix

rng = np.random.default_rng(0)

# --- 1. Your data --------------------------------------------------------
# spins:  1D array of +1/-1, one per timestamp (the observed target series,
#         e.g. "regime up/down", "on/off", a binarized indicator, ...).
# signal: 1D array of the real-valued driving covariate, same length,
#         same timestamps (e.g. a stress index, external field, news score).
#
# Here we just simulate a plausible example: a signal-driven spin path with
# some persistence, so the output has real structure to look at.
N = 200
signal = np.cumsum(rng.normal(0, 1, N)) * 0.05 + rng.normal(0, 0.3, N)
field_strength = 0.7
persistence = 0.5
spins = np.empty(N)
spins[0] = 1.0
for t in range(1, N):
    p_up = 1.0 / (1.0 + np.exp(-2 * (persistence * spins[t - 1] + field_strength * signal[t])))
    spins[t] = 1.0 if rng.random() < p_up else -1.0

# --- 2. Fit + get the susceptibility matrix ------------------------------
chi, diagnostics = susceptibility_matrix(
    spins,
    signal,
    standardize_signal=True,   # z-score signal first (recommended default)
    j_model="auto",            # let BIC choose constant vs signal-modulated K
    return_diagnostics=True,
)

print("chi shape:", chi.shape)
print("selected j_model:", diagnostics["j_model_selected"])
print("B_scale (sensitivity per 1 std-dev of raw signal):", diagnostics["B_scale"])
if diagnostics["j_model_selected"] == "constant":
    print("K (coupling):", diagnostics["K"])
else:
    print("K0, K1:", diagnostics["K0"], diagnostics["K1"])
print("BIC constant vs signal_modulated:",
      diagnostics.get("bic_constant"), "vs", diagnostics.get("bic_signal_modulated"))
print("log-likelihood at optimum:", diagnostics["loglik"])
print("symmetry check ||chi - chi.T||:", np.abs(chi - chi.T).max())

# --- 3. Read off a couple of concrete numbers -----------------------------
k = 100
print(f"\nSensitivity of target at t={k} to signal at various lags:")
for lag in [-20, -5, 0, 5, 20]:
    l = k + lag
    if 0 <= l < N:
        print(f"  signal[t={l:>4} , lag={lag:+d}] -> chi[{k},{l}] = {chi[k, l]:+.4f}")

# --- 4. (Optional) visualize -----------------------------------------------
try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(chi, cmap="RdBu_r", vmin=-np.abs(chi).max(), vmax=np.abs(chi).max())
    ax.set_xlabel("signal timestamp l")
    ax.set_ylabel("target timestamp k")
    ax.set_title("Susceptibility matrix chi[k, l]")
    fig.colorbar(im, ax=ax, label="d<s_k>/d(signal_l)")
    fig.tight_layout()
    fig.savefig("chi_heatmap.png", dpi=150)
    print("\nSaved heatmap to chi_heatmap.png")
except ImportError:
    print("\n(matplotlib not installed; skipping heatmap)")
