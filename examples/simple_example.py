"""
rustmc — Simple Example
========================

Fit a Bayesian linear regression: y = beta * x + noise
with a Normal(0, 1) prior on beta and Normal(mu, 1) likelihood.
"""

import numpy as np
import rustmc as rmc

# --- Generate synthetic data ---
np.random.seed(42)
N = 1_000
x = np.random.randn(N)
beta_true = 2.5
y = beta_true * x + np.random.randn(N)
data = {"x": x, "y": y}

# --- Define model ---
builder = rmc.ModelBuilder()
beta = builder.normal_prior("beta", mu=0.0, sigma=1.0)
mu_expr = beta * "x"
builder.normal_likelihood("obs", mu_expr=mu_expr, sigma=1.0, observed_key="y")
model = builder.build()

# --- Sample ---
fit = rmc.sample(
    model_spec=model,
    data=data,
    chains=4,
    draws=1000,
    seed=42,
)

# --- Results ---
print(fit)
print()
print("Posterior means:", fit.mean())
print("Posterior stds:", fit.std())
print("Accept rates:", fit.accept_rates())
print()
print(f"True beta = {beta_true}")
print(f"Estimated beta = {fit.mean()['beta']:.4f} ± {fit.std()['beta']:.4f}")
