# Linear Regression

A Bayesian linear regression with an unknown slope **and** noise level, using the full workflow: prior predictive check → posterior sampling → posterior predictive check.

## Model

$$
\beta \sim \text{Normal}(0, 5)
$$
$$
\sigma \sim \text{HalfNormal}(2)
$$
$$
y_i \sim \text{Normal}(\beta \cdot x_i,\ \sigma)
$$

Both the slope and the observation noise are inferred from data. `sigma` gets a `HalfNormal` prior — the standard choice for a positive scale parameter.

## Code

```python
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
sigma = builder.half_normal_prior("sigma", sigma=2.0)

builder.normal_likelihood("obs", mu_expr=beta * "x", sigma=sigma, observed_key="y")
model = builder.build()

# --- 1. Prior predictive check ---
prior_pred = rmc.sample_prior_predictive(model, n_samples=500, seed=0)
print(f"Prior beta  ~ N(0,5):  mean={prior_pred['beta'].mean():.2f}, std={prior_pred['beta'].std():.2f}")
print(f"Prior sigma ~ HN(2):   mean={prior_pred['sigma'].mean():.2f}, std={prior_pred['sigma'].std():.2f}")
print(f"Prior y_hat range:     [{prior_pred['obs'].min():.1f}, {prior_pred['obs'].max():.1f}]")

# --- 2. Posterior sampling ---
fit = rmc.sample(model_spec=model, chains=4, draws=2000, warmup=1000, seed=42)
print(fit.summary())

print(f"True beta  = {beta_true},  estimated = {fit.mean()['beta']:.4f} ± {fit.std()['beta']:.4f}")
print(f"True sigma = {sigma_true}, estimated = {fit.mean()['sigma']:.4f} ± {fit.std()['sigma']:.4f}")

# --- 3. Posterior predictive check ---
ppc   = fit.posterior_predictive(n_samples=500, seed=42)
y_rep = ppc["obs"]   # shape: (500, N)

ppc_p = (y_rep.std(axis=1) > y.std()).mean()
print(f"PPC p-value (std): {ppc_p:.3f}  (0.5 = perfect calibration)")
```

## Output

```
Prior beta  ~ N(0,5):  mean=0.07, std=4.98
Prior sigma ~ HN(2):   mean=1.57, std=1.22
Prior y_hat range:     [-42.3, 45.1]

4 chains x 2000 draws per chain

Parameter        mean      std     hdi_3%    hdi_97%   ess_bulk   ess_tail    r_hat  mcse_mean
-----------------------------------------------------------------------------------------------
beta           2.4981   0.0670     2.3662     2.6258       3241       3108   1.0003   0.001178
sigma          1.5043   0.0479     1.4142     1.5973       3887       2989   1.0003   0.000768
-----------------------------------------------------------------------------------------------
Mean accept rate: 0.94  |  Divergences: 0

True beta  = 2.5,  estimated = 2.4981 ± 0.0670
True sigma = 1.5,  estimated = 1.5043 ± 0.0479
PPC p-value (std): 0.496  (0.5 = perfect calibration)
```

## Notes

- R-hat near 1.0 and ESS > 400 on both parameters indicate clean convergence.
- The PPC p-value of ~0.5 means the model's replicated datasets have similar spread to the observed data — a well-calibrated fit.
- `sigma` is sampled in log-space internally (HalfNormal uses a log transform) and back-transformed before returning. You don't need to do anything special; this is automatic.
