"""
rustmc vs PyMC — Heavy Multivariate Regression Benchmark
=========================================================

20-parameter multiple regression with 100K observations and 8 chains.

This workload stresses the inner sampling loop: each gradient evaluation
requires summing over 100K data points for each of 20 parameters, and
the 8 chains highlight thread-based (rustmc) vs process-based (PyMC)
parallelism.
"""

import time

import numpy as np

# ---------------------------------------------------------------------------
# Synthetic data: y = X @ beta + noise
# ---------------------------------------------------------------------------
np.random.seed(42)
N = 10_000
P = 4

X = np.random.randn(N, P)
beta_true = np.linspace(0.5, 3.0, P)
y = X @ beta_true + np.random.randn(N) * 0.5

NUM_CHAINS = 8
NUM_DRAWS = 2000

print(f"Model: {P} parameters, {N:,} observations, {NUM_CHAINS} chains, {NUM_DRAWS} draws")
print(f"True betas: [{beta_true[0]:.2f}, {beta_true[1]:.2f}, ..., {beta_true[-1]:.2f}]")
print()

# ---------------------------------------------------------------------------
# Fit with PyMC
# ---------------------------------------------------------------------------
pymc_time = None
pymc_means = None
try:
    import pymc as pm

    with pm.Model():
        betas = [pm.Normal(f"beta_{i}", 0, 10) for i in range(P)]
        mu = sum(betas[i] * X[:, i] for i in range(P))
        pm.Normal("obs", mu=mu, sigma=0.5, observed=y)

        start = time.time()
        trace = pm.sample(NUM_DRAWS, chains=NUM_CHAINS, cores=NUM_CHAINS)
        pymc_time = time.time() - start

    pymc_means = {f"beta_{i}": float(trace.posterior[f"beta_{i}"].mean()) for i in range(P)}
    print(f"PyMC  — time: {pymc_time:.2f}s")
    for i in [0, 1, P - 1]:
        print(f"  beta_{i}: true={beta_true[i]:.3f}, est={pymc_means[f'beta_{i}']:.3f}")
except ImportError:
    print("PyMC not installed — skipping PyMC benchmark.")
    print("  Install with: pip install pymc")
except Exception as e:
    print(f"PyMC error: {e}")

print()

# ---------------------------------------------------------------------------
# Fit with rustmc
# ---------------------------------------------------------------------------
import rustmc as rmc

builder = rmc.ModelBuilder()
params = [builder.normal_prior(f"beta_{i}", mu=0.0, sigma=10.0) for i in range(P)]

mu_expr = params[0] * "x_0"
for i in range(1, P):
    mu_expr = mu_expr + params[i] * f"x_{i}"
builder.normal_likelihood("obs", mu_expr=mu_expr, sigma=0.5, observed_key="y")
model = builder.build()

data = {f"x_{i}": np.ascontiguousarray(X[:, i]) for i in range(P)}
data["y"] = y

start = time.time()
fit = rmc.sample(
    model_spec=model,
    data=data,
    chains=NUM_CHAINS,
    draws=NUM_DRAWS,
    warmup=1000,
    seed=42,
)
rustmc_time = time.time() - start

rustmc_means = fit.mean()
print(f"rustmc — time: {rustmc_time:.2f}s")
for i in [0, 1, P - 1]:
    name = f"beta_{i}"
    print(f"  {name}: true={beta_true[i]:.3f}, est={rustmc_means[name]:.3f}")

print(f"\nAccept rates: {[f'{r:.2f}' for r in fit.accept_rates()]}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print(f"{'Metric':<25} {'rustmc':>12} {'PyMC':>12}")
print("-" * 60)
print(f"{'Time (s)':<25} {rustmc_time:>12.2f} {(pymc_time or float('nan')):>12.2f}")
print(f"{'Chains':<25} {NUM_CHAINS:>12} {NUM_CHAINS:>12}")
print(f"{'Draws':<25} {NUM_DRAWS:>12} {NUM_DRAWS:>12}")
print(f"{'Parameters':<25} {P:>12} {P:>12}")
print(f"{'Observations':<25} {N:>12,} {(N if pymc_time else 'N/A'):>12}")
if pymc_time:
    speedup = pymc_time / rustmc_time
    print(f"{'Speedup':<25} {f'{speedup:.1f}x':>12}")
print("=" * 60)
