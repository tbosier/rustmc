# API Reference

## Linear Gaussian state-space models

`LinearGaussianStateSpace` implements a time-homogeneous Kalman filter and
Rauch--Tung--Striebel smoother for an arbitrary-dimensional latent state and a
single scalar observation per time point. Arrays use the conventional model

```text
x[t] = transition @ x[t-1] + process noise
y[t] = observation @ x[t] + observation noise
```

`initial_mean` and `initial_covariance` describe `x[-1]`, immediately before the
first observation. The filter applies one transition/process-noise prediction before
updating on `y[0]`. This convention also applies when forecasting an empty history.

Construct a general model with NumPy arrays, or use the `local_level()`,
`local_linear_trend()`, and zero-mean `stationary_ar1()` constructors. `filter(y)` returns predicted and filtered
state means/covariances plus the observed-data log likelihood; `smooth(y)` adds
smoothed state moments; and `forecast(y, steps)` returns future latent-state and
observation means/variances. A `NaN` observation is treated as missing and
causes a prediction-only step; infinities are rejected.

`forecast.interval(level=0.95)` returns lower and upper pointwise Gaussian
predictive bounds. Its `uncertainty_kind` is `"conditional_fixed_parameters"`:
the interval includes filtered-state, future-process, and observation noise, but
does not include uncertainty about the supplied system parameters. It is therefore
a conditional predictive interval, not yet a parameter-integrated Bayesian credible
interval.

This first state-space API assumes fixed, time-invariant system matrices,
Gaussian noise, and univariate observations. Covariance matrices and scalar
noise variances must be strictly positive definite/positive. It does not yet
estimate system parameters, support multivariate observations, accept
time-varying matrices, or integrate a Kalman likelihood into `ModelBuilder`.
Filtering, smoothing, and forecasting release the Python GIL after converting the
input NumPy array.

## Bayesian local-level forecasting

`BayesianLocalLevel` estimates the two unknown noise variances in the scalar model

```text
x[-1] ~ Normal(initial_mean, initial_variance)
x[t] = x[t-1] + Normal(0, process_variance)
y[t] = x[t] + Normal(0, observation_variance)
```

It uses joint forward-filtering/backward-sampling state draws and conjugate Gibbs
updates for the variances, following the FFBS/data-augmentation approach described by
[Frühwirth-Schnatter (1994)](https://doi.org/10.1111/j.1467-9892.1994.tb00184.x)
and [Carter and Kohn (1994)](https://doi.org/10.1093/biomet/81.3.541).

Priors are deliberately explicit because variance scales depend on the units of the
series:

```python
model = rmc.BayesianLocalLevel(
    process_variance_prior=rmc.InverseGammaPrior(shape=2.5, scale=0.3),
    observation_variance_prior=rmc.InverseGammaPrior(shape=2.5, scale=0.6),
    initial_mean=0.0,
    initial_variance=4.0,
)
fit = model.fit(y, chains=4, draws=1000, warmup=500, seed=42)
forecast = fit.forecast(steps=12, seed=43)

predictive_lower, predictive_upper = forecast.interval(0.95)
state_lower, state_upper = forecast.state_interval(0.95)
```

`InverseGammaPrior(shape, scale)` is a prior on a **variance**, with density
proportional to `x^(-shape-1) exp(-scale/x)`. `fit.get_samples_2d()` returns both
variance and standard-deviation draws plus the terminal latent level, all preserving
`(chain, draw)` shape. `fit.to_arviz()` exports the parameters and observed series for
convergence checks.

`forecast.state_samples` and `forecast.observation_samples` have shape
`(chain, draw, step)` and contain coherent paths. `state_interval()` is a pointwise
equal-tailed latent-state posterior credible interval. `interval()` is a pointwise
equal-tailed posterior-predictive interval for future observations. The latter
integrates parameter uncertainty, terminal-state uncertainty, future process noise,
and observation noise. It is not a simultaneous trajectory band.

`NaN` retains a missing time step and infinities are rejected. Fitting requires at
least two finite observations. This specialized model assumes equally spaced scalar
Gaussian observations; local trend, AR(1), seasonality, covariates, and irregular
timestamps are not yet part of the fitted Bayesian API. Gibbs output remains finite
MCMC output, so inspect multiple-chain convergence and effective sample sizes rather
than treating it as an analytic posterior.

## `ModelBuilder`

```python
builder = rmc.ModelBuilder(data=None)
```

Constructs a model. Data can be bound at build time or passed later to `rmc.sample()`,
`rmc.batch_sample()`, or `rmc.sample_prior_predictive()`.

### Priors

| Method | Distribution | Notes |
|--------|-------------|-------|
| `normal_prior(name, mu, sigma)` | Normal(mu, sigma) | `mu` and `sigma` may be `float` or earlier `ParamRef` values |
| `half_normal_prior(name, sigma)` | HalfNormal(sigma) | `sigma` may be `float` or earlier `ParamRef` |
| `exponential_prior(name, rate)` | Exponential(rate) | `rate` may be `float` or earlier `ParamRef` |
| `log_normal_prior(name, mu, sigma)` | LogNormal(mu, sigma) | `mu` and `sigma` may be `float` or earlier `ParamRef` |
| `student_t_prior(name, nu, mu=0.0, sigma=1.0)` | StudentT(nu, mu, sigma) | scalar only |
| `gamma_prior(name, alpha, beta)` | Gamma(alpha, beta) | scalar only |
| `beta_prior(name, alpha, beta)` | Beta(alpha, beta) | scalar only |
| `uniform_prior(name, lower=0.0, upper=1.0)` | Uniform(lower, upper) | scalar only |
| `vector_normal_prior(name, n, mu=0.0, sigma=1.0)` | Normal(mu, sigma)^n | explicit vector block |
| `bernoulli_prior(name, p=0.5)` | Bernoulli(p) | discrete, not suitable for gradient-based inference |
| `poisson_prior(name, lam)` | Poisson(lam) | discrete, not suitable for gradient-based inference |

All scalar prior methods return a `ParamRef`. `vector_normal_prior()` returns a `VectorParamRef`.

### Hierarchical priors and automatic non-centering

Scalar hierarchical priors are supported for:

- `normal_prior()` with parameter-valued `mu` and/or `sigma`
- `half_normal_prior()` with parameter-valued `sigma`
- `exponential_prior()` with parameter-valued `rate`
- `log_normal_prior()` with parameter-valued `mu` and/or `sigma`

When a scalar `normal_prior()` depends on another parameter through `mu` or `sigma`,
rustmc automatically compiles it as a non-centered latent where appropriate. Users still
see the logical parameter name in summaries, diagnostics, ArviZ export, and predictive
workflows.

Vector-valued hierarchical priors are not yet supported.

### Likelihoods

| Method | Family | Linear predictor | Extra parameter |
|--------|--------|------------------|-----------------|
| `normal_likelihood(name, mu_expr, sigma, observed_key)` | Normal | `mu_expr` | `sigma` is `float` or `ParamRef` |
| `bernoulli_logit_likelihood(name, eta_expr, observed_key)` | Bernoulli with logit link | `eta_expr` | none |
| `poisson_log_likelihood(name, eta_expr, observed_key)` | Poisson with log link | `eta_expr` | none |
| `exponential_likelihood(name, eta_expr, observed_key)` | Exponential with log-rate link | `eta_expr` | none |
| `log_normal_likelihood(name, mu_expr, sigma, observed_key)` | LogNormal | `mu_expr` | `sigma` is `float` or `ParamRef` |
| `negative_binomial_likelihood(name, eta_expr, alpha, observed_key)` | NegativeBinomial with log-mean link | `eta_expr` | `alpha` is `float` or `ParamRef` |

Likelihood expressions accept:

- a bare `ParamRef`
- `beta * "x"`
- `alpha + beta * "x"`
- `beta @ "X"` for matrix-vector regression
- additive constants such as `alpha + beta * "x" + 1.0`

### `build()`

```python
model = builder.build()
```

Returns a `ModelSpec`, the opaque handle passed to the sampling and predictive APIs.

### `compile()`

```python
compiled = builder.compile()
```

Returns a `CompiledModel` containing immutable graph structure and a `DataSchema`, without
concrete observation payloads. Matrix column counts and parameter shapes are structural;
the observation row count belongs to each binding.

## `CompiledModel`

| Member | Description |
|--------|-------------|
| `param_names` | Structural parameter names |
| `required_keys` | Predictor, response, and matrix keys required by the schema |
| `structure_id` | Process-local identity useful for checking structure reuse |
| `bind(data, id="0", strict=True, check_finite=True)` | Validate data and return a `BoundModel` |
| `sample(data_or_binding, **sampler_options)` | Sample one validated dataset and return `FitResult` |
| `sample_batch(datasets, ids=None, shared=None, **sampler_options)` | Sample many datasets and return `BatchFit` in input order |

`BoundModel` exposes `id` and `n_obs` and can be reused only with the `CompiledModel`
that created it. In `sample_batch()`, keys supplied through `shared` are converted once
and cannot be shadowed by a per-dataset dictionary. Dataset IDs must be unique and match
the number of datasets.

`BatchFit.ids` preserves caller order; `len(batch_fit)` returns the dataset count and
`batch_fit[i]` returns a `BatchResult`. This foundational batch path currently fails fast
on binding or sampling errors; partial-failure collection is future work.

### Context-manager contract

`ModelBuilder`, `CompiledModel`, `BoundModel`, and `BatchFit` may be used with
`with` when lexical scoping improves readability:

```python
with rmc.ModelBuilder() as builder:
    beta = builder.normal_prior("beta", 0.0, 1.0)
    builder.normal_likelihood("obs", beta * "x", 1.0, "y")

with builder.compile() as compiled:
    with compiled.bind(data, id="store-17") as bound:
        fit = compiled.sample(bound)
```

Entering returns the same object and exiting propagates exceptions. No object is
closed or invalidated, so the builder, compiled model, binding, and batch result
remain usable afterward. There is no ambient or thread-local current model:
model declarations must always be called on the intended builder. Context syntax
does not compile a builder or bind data automatically.

## `rmc.sample()`

```python
fit = rmc.sample(
    model_spec,
    data=None,
    chains=4,
    draws=1000,
    warmup=500,
    seed=42,
    threads=0,
    step_size=0.0,
    sampler="nuts",
    max_tree_depth=10,
    num_leapfrog_steps=15,
    show_progress=True,
)
```

Returns a `FitResult`.

Notes:

- `sampler` may be `"nuts"` or `"hmc"`.
- `threads=0` uses Rayon defaults.
- `max_tree_depth` applies to NUTS.
- `num_leapfrog_steps` applies to HMC.
- `step_size=0.0` means auto-tune.

## `rmc.batch_sample()`

```python
results = rmc.batch_sample(
    models,  # list[(ModelSpec, data_dict)]
    chains=1,
    draws=500,
    warmup=300,
    seed=42,
    sampler="nuts",
    step_size=0.0,
    max_tree_depth=8,
    num_leapfrog_steps=15,
    show_progress=True,
)
```

Returns a list of `BatchResult`, one per model.

Unlike the original throughput-only path, batch sampling now supports multiple chains per
model and both NUTS and fixed-step HMC. Use `chains > 1` when reliability matters more
than absolute batch throughput.

## `FitResult`

| Method | Returns | Description |
|--------|---------|-------------|
| `summary()` | `str` | Formatted diagnostics table |
| `mean()` | `dict[str, float]` | Posterior mean per parameter |
| `std()` | `dict[str, float]` | Posterior std per parameter |
| `get_samples()` | `dict[str, np.ndarray]` | Flattened samples across chains |
| `get_samples_2d()` | `dict[str, np.ndarray]` | Samples shaped `(chains, draws)` |
| `diagnostics()` | `list[dict]` | Per-parameter diagnostics |
| `accept_rates()` | `list[float]` | Per-chain accept rates |
| `step_sizes()` | `list[float]` | Per-chain adapted step sizes |
| `divergences()` | `list[int]` | Per-chain divergence counts |
| `posterior_predictive(n_samples=None, seed=42)` | `dict[str, np.ndarray]` | Posterior predictive samples shaped `(n_samples, n_obs)` per likelihood |
| `log_likelihood()` | `dict[str, np.ndarray]` | Pointwise log-likelihood shaped `(chain, draw, obs)` per likelihood |
| `to_arviz(include_ppc=False, ppc_samples=None, ppc_seed=42, include_log_likelihood=True)` | ArviZ inference container | Convert to ArviZ's version-native container (`InferenceData` on 0.x, `DataTree` on 1.x) with observed data, optionally including predictive draws and pointwise log-likelihood |

`log_likelihood()` is the intended bridge for `az.loo(...)` and `az.waic(...)`.

## `rmc.sample_prior_predictive()`

```python
prior_pred = rmc.sample_prior_predictive(
    model,
    data=None,
    n_samples=500,
    seed=42,
)
```

Returns `dict[str, np.ndarray]` containing:

- one 1-D array per parameter with `n_samples` prior draws
- one 2-D array per likelihood with shape `(n_samples, n_obs)`

For automatically non-centered scalar hierarchical normals, the returned parameter draws use
the logical parameter name, not the hidden raw latent.

## `BatchResult`

Each element returned by `rmc.batch_sample()` is a `BatchResult`.

| Method / property | Returns | Description |
|--------|---------|-------------|
| `mean()` | `dict[str, float]` | Posterior mean per parameter |
| `std()` | `dict[str, float]` | Posterior std per parameter |
| `get_samples()` | `dict[str, np.ndarray]` | Flattened samples across all chains and draws |
| `get_samples_2d()` | `dict[str, np.ndarray]` | Samples shaped `(chains, draws)` |
| `chains` | `int` | Number of chains run for this model |
| `draws` | `int` | Number of post-warmup draws per chain |
| `accept_rate` | `float` | Mean accept rate across chains |
| `accept_rates` | `list[float]` | Per-chain accept rates |
| `divergences` | `int` | Total divergences across chains |
| `divergences_per_chain` | `list[int]` | Per-chain divergence counts |

## `ParamRef` and `Expr` operators

```python
beta * "x"             # elementwise scalar predictor
alpha + beta * "x"     # additive linear predictor
beta @ "X"             # matrix-vector regression
beta + 1.0             # additive constant
1.0 + beta * "x"       # constant plus expression
```

Direct use as the likelihood expression is also valid:

```python
mu_global = builder.normal_prior("mu_global", mu=0.0, sigma=5.0)
sigma_group = builder.half_normal_prior("sigma_group", sigma=2.0)
mu_group = builder.normal_prior("mu_group", mu=mu_global, sigma=sigma_group)
builder.normal_likelihood("obs", mu_expr=mu_group, sigma=1.0, observed_key="y")
```

That scalar hierarchical pattern is supported today and will automatically use the
non-centered compilation path when eligible.
