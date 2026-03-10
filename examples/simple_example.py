"""
rustmc — Simple Example
========================

Fit a Bayesian linear regression: y ~ Normal(beta * x, sigma)

Both the slope (beta) and the observation noise (sigma) are inferred
from data.  sigma gets a HalfNormal(2) prior — the standard choice for
a positive scale parameter.  Using sigma as a free parameter rather than
a fixed constant is the typical real-world setup.

Note: our NUTS implementation uses a scalar step size (no diagonal mass
matrix).  Models with exp-transformed parameters (HalfNormal, Gamma) may
show a small number of divergent transitions even when estimates are
correct.  Full mass-matrix adaptation is a planned improvement.
"""

import numpy as np
import rustmc as rmc

# --- Generate synthetic data ---
np.random.seed(42)
N = 500
x = np.random.randn(N)
beta_true  = 2.5
sigma_true = 1.5
y = beta_true * x + np.random.normal(0, sigma_true, N)
data = {"x": x, "y": y}

# --- Define model ---
builder = rmc.ModelBuilder(data=data)
beta  = builder.normal_prior("beta", mu=0.0, sigma=5.0)
sigma = builder.half_normal_prior("sigma", sigma=2.0)  # sigma ~ HalfNormal(2)

# Pass sigma as a ParamRef — it will be jointly inferred with beta
builder.normal_likelihood("obs", mu_expr=beta * "x", sigma=sigma, observed_key="y")
model = builder.build()

# --- Sample ---
fit = rmc.sample(
    model_spec=model,
    chains=4,
    draws=2000,
    warmup=1000,
    seed=42,
)

# --- Diagnostics ---
print()
print(fit.summary())
print()
print(f"True beta  = {beta_true},  estimated = {fit.mean()['beta']:.4f} ± {fit.std()['beta']:.4f}")
print(f"True sigma = {sigma_true}, estimated = {fit.mean()['sigma']:.4f} ± {fit.std()['sigma']:.4f}")
print(f"Step sizes: {[round(s, 4) for s in fit.step_sizes()]}")
