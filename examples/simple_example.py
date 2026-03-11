"""
rustmc — Simple Example
========================

Fit a Bayesian linear regression: y ~ Normal(beta * x, sigma)

Demonstrates the full Bayesian workflow:
  1. Prior predictive check  — does the prior produce plausible data?
  2. Posterior sampling       — condition on observed data via NUTS
  3. Posterior predictive     — generate replicated data from the posterior

Both the slope (beta) and the observation noise (sigma) are inferred
from data.  sigma gets a HalfNormal(2) prior — the standard choice for
a positive scale parameter.
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

# --- 1. Prior predictive check ---
print("Prior predictive check …")
prior_pred = rmc.sample_prior_predictive(model, n_samples=500, seed=0)
print(f"  Prior beta  ~ N(0, 5):  mean={prior_pred['beta'].mean():.2f}, std={prior_pred['beta'].std():.2f}")
print(f"  Prior sigma ~ HN(2):    mean={prior_pred['sigma'].mean():.2f}, std={prior_pred['sigma'].std():.2f}")
print(f"  Prior y_hat range:      [{prior_pred['obs'].min():.1f}, {prior_pred['obs'].max():.1f}]")
print()

# --- 2. Posterior sampling ---
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

# --- 3. Posterior predictive check ---
print()
print("Posterior predictive check …")
ppc = fit.posterior_predictive(n_samples=500, seed=42)
y_rep = ppc["obs"]                      # shape: (n_samples, N)
print(f"  y_rep shape:  {y_rep.shape}")
print(f"  y_rep mean:   {y_rep.mean():.4f}  (data mean: {y.mean():.4f})")
print(f"  y_rep std:    {y_rep.std():.4f}   (data std:  {y.std():.4f})")
# Simple posterior predictive p-value: fraction of reps where std > observed std
ppc_p = (y_rep.std(axis=1) > y.std()).mean()
print(f"  PPC p-value (std): {ppc_p:.3f}  (0.5 = perfect calibration)")
