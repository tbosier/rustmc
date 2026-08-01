# Getting Started

## Installation

### From PyPI (recommended)

```bash
pip install rustmc
```

### From source

Requires Rust (stable) and [maturin](https://github.com/PyO3/maturin).

```bash
git clone https://github.com/tbosier/rustmc.git
cd rustmc
python -m venv .venv && source .venv/bin/activate
pip install maturin numpy
maturin develop --manifest-path python_bindings/Cargo.toml --release
```

## Your First Model

A simple Bayesian linear regression: `y ~ Normal(beta * x, sigma)`.

```python
import numpy as np
import rustmc as rmc

# Generate synthetic data
np.random.seed(42)
N = 500
x = np.random.randn(N)
y = 2.5 * x + np.random.normal(0, 1.5, N)

# Build the model
builder = rmc.ModelBuilder(data={"x": x, "y": y})
beta  = builder.normal_prior("beta",  mu=0.0, sigma=5.0)
sigma = builder.half_normal_prior("sigma", sigma=2.0)
builder.normal_likelihood("obs", mu_expr=beta * "x", sigma=sigma, observed_key="y")
model = builder.build()

# Sample
fit = rmc.sample(model_spec=model, chains=4, draws=2000, warmup=1000, seed=42)
print(fit.summary())
```

## Reading the Output

```
4 chains x 2000 draws per chain

Parameter        mean      std     hdi_3%    hdi_97%   ess_bulk   ess_tail    r_hat  mcse_mean
-----------------------------------------------------------------------------------------------
beta           2.4981   0.0670     2.3662     2.6258       3241       3108   1.0003   0.001178
sigma          1.5043   0.0479     1.4142     1.5973       3887       2989   1.0003   0.000768
-----------------------------------------------------------------------------------------------
Mean accept rate: 0.94  |  Divergences: 0
```

- **mean / std** — posterior mean and standard deviation
- **hdi_3% / hdi_97%** — 94% highest density interval
- **ess_bulk / ess_tail** — effective sample size; aim for > 400
- **r_hat** — raw split-chain convergence diagnostic; values near 1.0 indicate convergence
- **mcse_mean** — Monte Carlo standard error of the mean

## Core API

### `ModelBuilder`

```python
builder = rmc.ModelBuilder(data={"x": x, "y": y})
```

Data can be passed at build time or at sample time via the `data=` argument to `rmc.sample()`.

### Priors

```python
# Scalar priors
beta  = builder.normal_prior("beta",  mu=0.0, sigma=1.0)
sigma = builder.half_normal_prior("sigma", sigma=2.0)
alpha = builder.student_t_prior("alpha", nu=3.0, mu=0.0, sigma=1.0)
p     = builder.beta_prior("p", alpha=1.0, beta=1.0)
lam   = builder.gamma_prior("lam", alpha=2.0, beta=1.0)
```

### Likelihoods

```python
builder.normal_likelihood("obs", mu_expr=beta * "x", sigma=sigma, observed_key="y")
```

`mu_expr` can be:

- A `ParamRef` directly: `mu_expr=beta`
- A scalar multiply: `mu_expr=beta * "x"`
- A linear combination: `mu_expr=alpha + beta * "x"`
- A matrix multiply: `mu_expr=beta @ "X"` (promotes `beta` to a vector)

### `rmc.sample()`

```python
fit = rmc.sample(
    model_spec=model,
    data={"x": x, "y": y},   # optional if passed to ModelBuilder
    chains=4,
    draws=2000,
    warmup=1000,
    seed=42,
    sampler="nuts",           # "nuts" (default) or "hmc"
)
```

### Compile once and bind many datasets

When the model structure is repeated across datasets, compile it once and bind each
payload separately:

```python
builder = rmc.ModelBuilder()
beta = builder.normal_prior("beta", mu=0.0, sigma=1.0)
builder.normal_likelihood("obs", beta * "x", sigma=1.0, observed_key="y")
compiled = builder.compile()

fit = compiled.sample({"x": x, "y": y}, chains=4, draws=1000)
batch = compiled.sample_batch(
    [{"y": store_a_y}, {"y": store_b_y}],
    shared={"x": shared_x},
    ids=["store-a", "store-b"],
)
```

`compiled.bind(data)` validates the schema eagerly and returns a reusable `BoundModel`.
Bindings may have different row counts, but matrix column counts and parameter shapes are
part of the compiled structure.

### Linear Gaussian state-space models

For fixed-system filtering, smoothing, and forecasting:

```python
ssm = rmc.LinearGaussianStateSpace.local_level(
    process_variance=0.2,
    observation_variance=1.0,
)
filtered = ssm.filter(y_with_optional_nan_values)
smoothed = ssm.smooth(y_with_optional_nan_values)
forecast = ssm.forecast(y_with_optional_nan_values, steps=12)
lower_95, upper_95 = forecast.interval(0.95)
```

This standalone API keeps the supplied variances fixed. Its interval includes state,
process, and observation uncertainty, but not parameter uncertainty.

For Bayesian local-level fitting and a parameter-integrated 95% forecast interval:

```python
bayesian = rmc.BayesianLocalLevel(
    process_variance_prior=rmc.InverseGammaPrior(2.5, 0.3),
    observation_variance_prior=rmc.InverseGammaPrior(2.5, 0.6),
    initial_mean=0.0,
    initial_variance=4.0,
)
fit = bayesian.fit(y_with_optional_nan_values, chains=4, draws=1000, warmup=500)
forecast = fit.forecast(steps=12, seed=43)
lower_95, upper_95 = forecast.interval(0.95)

# Latent-state credible interval, excluding future observation noise:
state_lower_95, state_upper_95 = forecast.state_interval(0.95)
```

The inverse-gamma priors are on variances and must be chosen for the units of your
series; rustmc does not silently infer a universal scale.

### `FitResult` methods

```python
fit.summary()            # formatted table
fit.mean()               # dict: param -> float
fit.std()                # dict: param -> float
fit.get_samples()        # dict: param -> flattened samples
fit.get_samples_2d()     # dict: param -> np.ndarray, shape (chains, draws)
fit.diagnostics()        # list of per-parameter diagnostics
fit.accept_rates()       # list of per-chain accept rates
fit.step_sizes()         # list of per-chain step sizes
fit.divergences()        # list of per-chain divergence counts
fit.posterior_predictive(n_samples=500, seed=42)
fit.log_likelihood()     # dict: likelihood -> np.ndarray, shape (chains, draws, obs)
fit.to_arviz(include_ppc=True)
```

### Prior & Posterior Predictive

```python
prior_pred = rmc.sample_prior_predictive(model, n_samples=500, seed=0)
ppc        = fit.posterior_predictive(n_samples=500, seed=42)
```

`sample_prior_predictive()` returns parameter draws plus simulated observations. `posterior_predictive()` returns simulated observations keyed by likelihood name.

If you pass `n_samples`, `posterior_predictive()` randomly subsamples posterior draws without replacement. Use `None` to include all draws.

### Hierarchical Priors

Scalar hierarchical priors are supported today:

```python
mu_global = builder.normal_prior("mu_global", mu=0.0, sigma=5.0)
sigma_group = builder.half_normal_prior("sigma_group", sigma=2.0)
mu_group = builder.normal_prior("mu_group", mu=mu_global, sigma=sigma_group)
```

That works for scalar parameters and scalar `sigma` in the likelihood. Eligible scalar hierarchical normals are automatically compiled through a non-centered parameterization, while still appearing under the logical parameter name in summaries and ArviZ output. Vector-valued hierarchical blocks are not implemented yet.

### Other likelihood families

The same builder also supports:

```python
builder.bernoulli_logit_likelihood("clicked", eta_expr=alpha + beta * "x", observed_key="y")
builder.poisson_log_likelihood("count", eta_expr=alpha + beta * "x", observed_key="y")
builder.exponential_likelihood("time", eta_expr=alpha + beta * "x", observed_key="y")
builder.log_normal_likelihood("sales", mu_expr=alpha + beta * "x", sigma=sigma, observed_key="y")
builder.negative_binomial_likelihood("orders", eta_expr=alpha + beta * "x", alpha=dispersion, observed_key="y")
```

For model comparison with ArviZ:

```python
idata = fit.to_arviz(include_log_likelihood=True)
# az.loo(idata)
# az.waic(idata)
```

## Next Steps

- [Linear Regression example](examples/linear-regression.md) — full workflow with prior/posterior predictive checks
- [Hierarchical Models](examples/hierarchical.md) — partial pooling across groups
- [Batch Inference](examples/batch-inference.md) — fitting many independent models at once
- [High-Dimensional Regression](examples/high-dimensional.md) — faer-backed `X @ beta`
