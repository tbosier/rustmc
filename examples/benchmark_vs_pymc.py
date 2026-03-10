"""
rustmc vs PyMC + nutpie — 500-parameter linear regression benchmark
===================================================================
Same model, same data, same draws/warmup/chains.
Reports wall time, iterations/s, and (most importantly) ESS/s.
"""
import time
import warnings
import logging
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────
N_OBS    = 2_000
N_PARAMS = 500
DRAWS    = 500
WARMUP   = 500
CHAINS   = 1
SEED     = 42

# ── Data ──────────────────────────────────────────────────────────────────
np.random.seed(SEED)
true_beta  = np.random.randn(N_PARAMS) * 0.1
X          = np.random.randn(N_OBS, N_PARAMS)
y          = X @ true_beta + np.random.randn(N_OBS) * 1.0

print(f"Model : {N_OBS:,} obs, {N_PARAMS:,} params")
print(f"Run   : {WARMUP} warmup + {DRAWS} draws, {CHAINS} chain(s)")
print()

# ── rustmc ────────────────────────────────────────────────────────────────
print("=" * 50)
print("rustmc (faer MatVecMul + Rayon)")
print("=" * 50)

import rustmc as rmc

builder   = rmc.ModelBuilder(data={"X": X, "y": y})
intercept = builder.normal_prior("intercept", mu=0.0, sigma=10.0)
beta      = builder.vector_normal_prior("beta", n=N_PARAMS, mu=0.0, sigma=1.0)
builder.normal_likelihood("obs",
    mu_expr=intercept + beta @ "X",
    sigma=1.0,
    observed_key="y")
model = builder.build()

t0 = time.perf_counter()
rmc_result = rmc.sample(model,
                        draws=DRAWS, warmup=WARMUP,
                        chains=CHAINS, seed=SEED,
                        show_progress=True)
rmc_time = time.perf_counter() - t0

# ESS: average ess_bulk across all beta params
rmc_diag  = rmc_result.diagnostics()
# diagnostics() returns a list of dicts; filter to beta params
rmc_beta_ess = np.mean([
    d["ess_bulk"] for d in rmc_diag
    if d["name"].startswith("beta[")
])

print(f"  Wall time   : {rmc_time:.2f}s")
print(f"  Iters/s     : {(DRAWS + WARMUP) / rmc_time:.1f}")
print(f"  Accept rate : {rmc_result.accept_rates()[0]:.3f}")
print(f"  Divergences : {sum(rmc_result.divergences())}")
print(f"  Beta ESS    : {rmc_beta_ess:.0f}  (mean across {N_PARAMS} params)")
print(f"  ESS/s       : {rmc_beta_ess / rmc_time:.1f}")

# ── PyMC + nutpie ─────────────────────────────────────────────────────────
print()
print("=" * 50)
print("PyMC + nutpie")
print("=" * 50)

logging.getLogger("pymc").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import pymc as pm
import arviz as az

with pm.Model() as pymc_model:
    pm_intercept = pm.Normal("intercept", mu=0.0, sigma=10.0)
    pm_beta      = pm.Normal("beta", mu=0.0, sigma=1.0, shape=N_PARAMS)
    mu           = pm_intercept + pm.math.dot(X, pm_beta)
    pm.Normal("obs", mu=mu, sigma=1.0, observed=y)

    t0 = time.perf_counter()
    trace = pm.sample(
        draws=DRAWS,
        tune=WARMUP,
        chains=CHAINS,
        nuts_sampler="nutpie",
        progressbar=True,
        random_seed=SEED,
    )
    pymc_time = time.perf_counter() - t0

pymc_beta_ess = float(az.ess(trace)["beta"].values.mean())
pymc_divs     = int(trace.sample_stats["diverging"].values.sum())
# nutpie uses mean_tree_accept; fallback to acceptance_rate for other samplers
accept_key = "mean_tree_accept" if "mean_tree_accept" in trace.sample_stats else "acceptance_rate"
pymc_accept   = float(trace.sample_stats[accept_key].values.mean())

print(f"  Wall time   : {pymc_time:.2f}s")
print(f"  Iters/s     : {(DRAWS + WARMUP) / pymc_time:.1f}")
print(f"  Accept rate : {pymc_accept:.3f}")
print(f"  Divergences : {pymc_divs}")
print(f"  Beta ESS    : {pymc_beta_ess:.0f}  (mean across {N_PARAMS} params)")
print(f"  ESS/s       : {pymc_beta_ess / pymc_time:.1f}")

# ── Summary ───────────────────────────────────────────────────────────────
print()
print("=" * 50)
print("Summary")
print("=" * 50)
print(f"{'':20s}  {'rustmc':>10}  {'PyMC+nutpie':>12}")
print(f"{'Wall time (s)':20s}  {rmc_time:>10.2f}  {pymc_time:>12.2f}")
print(f"{'Iters/s':20s}  {(DRAWS+WARMUP)/rmc_time:>10.1f}  {(DRAWS+WARMUP)/pymc_time:>12.1f}")
print(f"{'Beta ESS':20s}  {rmc_beta_ess:>10.0f}  {pymc_beta_ess:>12.0f}")
print(f"{'ESS/s':20s}  {rmc_beta_ess/rmc_time:>10.1f}  {pymc_beta_ess/pymc_time:>12.1f}")

speedup = (rmc_beta_ess / rmc_time) / (pymc_beta_ess / pymc_time)
if speedup >= 1:
    print(f"\nrustmc is {speedup:.2f}x faster by ESS/s")
else:
    print(f"\nPyMC+nutpie is {1/speedup:.2f}x faster by ESS/s")
