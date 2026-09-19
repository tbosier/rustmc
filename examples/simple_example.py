"""Linear regression

Fit a Bayesian linear regression and walk the whole workflow: check the prior,
sample the posterior, then check the posterior against the data it was fitted to.

    beta  ~ Normal(0, 5)
    sigma ~ HalfNormal(2)
    y_i   ~ Normal(beta * x_i, sigma)

Both the slope and the observation noise are inferred. The HalfNormal(2) prior on
`sigma` suits this example's units; pick your own for your data.
"""

# %%
import numpy as np
import rustmc as rmc

# %% Generate data
np.random.seed(42)
N = 500
x = np.random.randn(N)
beta_true = 2.5
sigma_true = 1.5
y = beta_true * x + np.random.normal(0, sigma_true, N)
data = {"x": x, "y": y}

print(f"N={N}, beta_true={beta_true}, sigma_true={sigma_true}")

# %% Define the model
builder = rmc.ModelBuilder(data=data)
beta = builder.normal_prior("beta", mu=0.0, sigma=5.0)
sigma = builder.half_normal_prior("sigma", sigma=2.0)

# Passing `sigma` as a ParamRef makes it a parameter, inferred jointly with beta.
builder.normal_likelihood("obs", mu_expr=beta * "x", sigma=sigma, observed_key="y")
model = builder.build()

# %% [markdown]
# ## Prior predictive check
#
# Draw from the priors alone, before seeing the data, and look at what data they
# imply. If the prior predictive range is absurd for your units, the priors are
# wrong and no amount of sampling will fix that.

# %%
prior_pred = rmc.sample_prior_predictive(model, n_samples=500, seed=0)
print(f"  Prior beta  ~ N(0, 5):  mean={prior_pred['beta'].mean():.2f}, std={prior_pred['beta'].std():.2f}")
print(f"  Prior sigma ~ HN(2):    mean={prior_pred['sigma'].mean():.2f}, std={prior_pred['sigma'].std():.2f}")
print(f"  Prior y_hat range:      [{prior_pred['obs'].min():.1f}, {prior_pred['obs'].max():.1f}]")

# %% Sample the posterior
fit = rmc.sample(
    model_spec=model,
    chains=4,
    draws=2000,
    warmup=1000,
    seed=42,
)
print(fit.summary())

# %% [markdown]
# ## Read the diagnostics before the estimates
#
# `r_hat` near 1.0 and `ess_bulk` in the thousands mean the chains agree and the
# draws are close to independent. Divergences would mean the sampler could not
# follow the posterior geometry, and any estimate below them would be suspect.
# Passing diagnostics say the sampler did its job; they do not say the model is
# the right model for the data.

# %%
print(f"True beta  = {beta_true},  estimated = {fit.mean()['beta']:.4f} +/- {fit.std()['beta']:.4f}")
print(f"True sigma = {sigma_true}, estimated = {fit.mean()['sigma']:.4f} +/- {fit.std()['sigma']:.4f}")
print(f"Step sizes: {[round(s, 4) for s in fit.step_sizes()]}")

# %% Posterior predictive check
ppc = fit.posterior_predictive(n_samples=500, seed=42)
y_rep = ppc["obs"]  # shape: (n_samples, N)
print(f"  y_rep shape:  {y_rep.shape}")
print(f"  y_rep mean:   {y_rep.mean():.4f}  (data mean: {y.mean():.4f})")
print(f"  y_rep std:    {y_rep.std():.4f}   (data std:  {y.std():.4f})")

# Fraction of replicated datasets whose spread exceeds the observed spread. Values
# near 0 or 1 mean the model reproduces the data's spread badly; this one is fine.
ppc_p = (y_rep.std(axis=1) > y.std()).mean()
print(f"  PPC p-value (std): {ppc_p:.3f}")

# %% [markdown]
# ## Plots
#
# `examples/arviz_example.py` fits the same shape of model and writes trace,
# posterior and pair plots with ArviZ. It needs `pip install "rustmc[viz]"`.
