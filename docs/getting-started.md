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
- **r_hat** — convergence diagnostic; values near 1.0 indicate convergence
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

### `FitResult` methods

```python
fit.summary()            # formatted table
fit.mean()               # dict: param -> float
fit.std()                # dict: param -> float
fit.get_samples("beta")  # np.ndarray, shape (chains, draws)
fit.diagnostics()        # dict with r_hat, ess, etc.
fit.step_sizes()         # list of per-chain step sizes
fit.divergences()        # total divergence count
```

### Prior & Posterior Predictive

```python
prior_pred = rmc.sample_prior_predictive(model, n_samples=500, seed=0)
ppc        = fit.posterior_predictive(n_samples=500, seed=42)
```

Both return `dict[str, np.ndarray]` keyed by likelihood name.

## Next Steps

- [Linear Regression example](examples/linear-regression.md) — full workflow with prior/posterior predictive checks
- [Hierarchical Models](examples/hierarchical.md) — partial pooling across groups
- [Batch Inference](examples/batch-inference.md) — fitting 10,000 models at once
- [High-Dimensional Regression](examples/high-dimensional.md) — faer-backed `X @ beta`
