"""
rustmc vs PyMC — Benchmark Comparison
======================================

Fit the same Bayesian linear regression model using both engines
and compare wall-clock time.

Requirements:
    pip install pymc numpy
    pip install maturin  (to build rustmc)
"""

import time

import numpy as np

# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------
np.random.seed(42)
N = 10_000
x = np.random.randn(N)
beta_true = 2.5
y = beta_true * x + np.random.randn(N)
data = {"x": x, "y": y}

NUM_CHAINS = 4
NUM_DRAWS = 1000

# ---------------------------------------------------------------------------
# Fit with PyMC
# ---------------------------------------------------------------------------
pymc_time = None
pymc_mean = None
try:
    import pymc as pm

    with pm.Model():
        beta = pm.Normal("beta", 0, 1)
        mu = beta * x
        obs = pm.Normal("obs", mu=mu, sigma=1, observed=y)

        start = time.time()
        trace = pm.sample(NUM_DRAWS, chains=NUM_CHAINS, progressbar=True)
        pymc_time = time.time() - start

    pymc_mean = float(trace.posterior["beta"].mean())
    print(f"PyMC  — time: {pymc_time:.2f}s, beta mean: {pymc_mean:.4f}")
except ImportError:
    print("PyMC not installed — skipping PyMC benchmark.")
except Exception as e:
    print(f"PyMC failed: {e}")

# ---------------------------------------------------------------------------
# Fit with rustmc
# ---------------------------------------------------------------------------
import rustmc as rmc

builder = rmc.ModelBuilder()
beta = builder.normal_prior("beta", mu=0.0, sigma=1.0)
mu_expr = beta * "x"
builder.normal_likelihood("obs", mu_expr=mu_expr, sigma=1.0, observed_key="y")
model = builder.build()

start = time.time()
fit = rmc.sample(
    model_spec=model,
    data=data,
    chains=NUM_CHAINS,
    draws=NUM_DRAWS,
    warmup=500,
    seed=42,
)
rustmc_time = time.time() - start

rustmc_mean = fit.mean()["beta"]
print(f"rustmc — time: {rustmc_time:.2f}s, beta mean: {rustmc_mean:.4f}")

# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
print()
print("=" * 50)
print(f"True beta: {beta_true}")
print(f"rustmc  beta: {rustmc_mean:.4f} ± {fit.std()['beta']:.4f}")
if pymc_mean is not None:
    print(f"PyMC    beta: {pymc_mean:.4f}")
print()
if pymc_time is not None:
    speedup = pymc_time / rustmc_time
    print(f"rustmc time: {rustmc_time:.2f}s")
    print(f"PyMC   time: {pymc_time:.2f}s")
    print(f"Speedup: {speedup:.1f}x")
else:
    print(f"rustmc time: {rustmc_time:.2f}s")
