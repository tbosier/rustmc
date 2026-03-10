"""
rustmc — ArviZ / InferenceData Integration Example
====================================================

`fit.to_arviz()` returns an `arviz.InferenceData` object, which unlocks
the full ArviZ diagnostics and visualization suite instantly.

Install ArviZ first:
    pip install arviz matplotlib

What you get for free after calling to_arviz():
    az.plot_trace(idata)           — trace plots (stationarity check)
    az.plot_posterior(idata)       — posterior density + HDI
    az.plot_pair(idata)            — pairplot / correlation matrix
    az.plot_forest(idata)          — forest plot across parameters
    az.summary(idata)              — R-hat, ESS, MCSE, HDI
    az.loo(idata)                  — Leave-One-Out cross-validation (needs log_likelihood)
    az.compare({"m1": id1, ...})   — model comparison table
    ... and everything else in ArviZ
"""

import numpy as np
import rustmc as rmc

# ArviZ is an optional dependency — give a clear message if missing
try:
    import arviz as az
except ImportError:
    raise SystemExit("Install ArviZ first:  pip install arviz")

# ── 1. Fit a model ────────────────────────────────────────────────────────────

np.random.seed(42)
N = 500
x = np.random.randn(N)
alpha_true, beta_true, sigma_true = 1.5, 2.5, 1.0
y = alpha_true + beta_true * x + np.random.randn(N) * sigma_true

builder = rmc.ModelBuilder(data={"x": x, "y": y})
alpha = builder.normal_prior("alpha", mu=0.0, sigma=10.0)
beta  = builder.normal_prior("beta",  mu=0.0, sigma=10.0)
sigma = builder.half_normal_prior("sigma", sigma=2.0)
builder.normal_likelihood("obs", mu_expr=alpha + beta * "x", sigma=sigma, observed_key="y")
model = builder.build()

fit = rmc.sample(model_spec=model, chains=4, draws=2000, warmup=1000, seed=42)

print("rustmc built-in summary:")
print(fit.summary())
print()

# ── 2. Convert to InferenceData ───────────────────────────────────────────────

idata = fit.to_arviz()

print(f"InferenceData groups: {list(idata.groups())}")
print(f"Posterior variables:  {list(idata.posterior.data_vars)}")
print(f"Chains × draws:       {idata.posterior.dims['chain']} × {idata.posterior.dims['draw']}")
print()

# ── 3. ArviZ summary (richer than fit.summary()) ──────────────────────────────

print("ArviZ summary:")
print(az.summary(idata, round_to=4))
print()

# ── 4. Convergence diagnostics ────────────────────────────────────────────────

print("Per-parameter R-hat values:")
summary_df = az.summary(idata)
for param, rhat in summary_df["r_hat"].items():
    status = "✓" if rhat < 1.01 else "⚠"
    print(f"  {status}  {param}: R-hat = {rhat:.4f}")
print()

n_div = int(idata.sample_stats.diverging.values.sum())
print(f"Total divergent transitions: {n_div}")
print()

# ── 5. Visualization (save to file so it works headlessly) ────────────────────

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Trace plot — check stationarity and mixing
    ax = az.plot_trace(idata, figsize=(10, 6))
    plt.suptitle("Trace plot", fontsize=14)
    plt.tight_layout()
    plt.savefig("trace_plot.png", dpi=100)
    plt.close()
    print("Saved: trace_plot.png")

    # Posterior density + HDI
    ax = az.plot_posterior(idata, figsize=(10, 3))
    plt.suptitle("Posterior distributions", fontsize=14)
    plt.tight_layout()
    plt.savefig("posterior_plot.png", dpi=100)
    plt.close()
    print("Saved: posterior_plot.png")

    # Pair plot — useful for spotting correlations between parameters
    ax = az.plot_pair(idata, figsize=(6, 6), divergences=True)
    plt.suptitle("Pair plot", fontsize=14)
    plt.tight_layout()
    plt.savefig("pair_plot.png", dpi=100)
    plt.close()
    print("Saved: pair_plot.png")

    print()
except Exception as e:
    print(f"(Matplotlib unavailable, skipping plots: {e})")

# ── 6. Posterior statistics via xarray ───────────────────────────────────────

post = idata.posterior
print("Posterior means via xarray:")
for var in post.data_vars:
    m = float(post[var].mean())
    s = float(post[var].std())
    print(f"  {var}: {m:.4f} ± {s:.4f}")
print()

print(f"True values: alpha={alpha_true}, beta={beta_true}, sigma={sigma_true}")
