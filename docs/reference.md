# API Reference

Per-class arguments, output shapes, and stated limits. This is a lookup page, not a
tutorial: it assumes you have already fitted something in
[Get started](getting-started.md) and now need the exact behaviour of one call.

Coverage is not complete. `StructuralModel`, `BayesianDynamicGLM`, `VarianceParameter`
and `ForecastSession` are public but documented only in their guides —
[structural composition](structural-forecasting.md) and
[dynamic GLMs](dynamic-glm.md).

Each section states what a model does not do as well as what it does. Those limits are
deliberate and current; where a thing is not supported, it says so next to the thing.

## Linear Gaussian state-space models

`LinearGaussianStateSpace` implements a Kalman filter and Rauch--Tung--Striebel
smoother for an arbitrary-dimensional latent state and a single scalar observation per
time point. Arrays use the conventional model

```text
x[t] = transition @ x[t-1] + process noise
y[t] = observation[t] @ x[t] + observation noise
```

The transition is the same at every time step. The observation row is the same at every
time step unless you set per-time rows; see [what may vary with
time](#what-may-vary-with-time) below.

`initial_mean` and `initial_covariance` describe `x[-1]`, immediately before the
first observation. The filter applies one transition/process-noise prediction before
updating on `y[0]`. This convention also applies when forecasting an empty history.

Construct a general model with NumPy arrays, or use the `local_level()`,
`local_linear_trend()`, `seasonal_local_level()`, and zero-mean `stationary_ar1()`
constructors. `filter(y)` returns predicted and filtered state means/covariances plus
the observed-data log likelihood. `smooth(y)` returns the filtered and smoothed state
moments and the same log likelihood; it does not carry the predicted moments, so keep
the `filter(y)` result if you need those. `forecast(y, steps)` returns future
latent-state and observation means/variances. A `NaN` observation is treated as missing
and causes a prediction-only step; infinities are rejected.

`forecast.observation_covariance` contains the joint covariance across future scalar
observations. `cumulative_observation_means`, `cumulative_observation_variances`, and
`cumulative_interval(level=0.95)` summarize prefix totals without discarding
cross-horizon dependence. `forecast.interval(level=0.95)` returns lower and upper pointwise Gaussian
predictive bounds. Its `uncertainty_kind` is `"conditional_fixed_parameters"`:
the interval includes filtered-state, future-process, and observation noise, but
does not include uncertainty about the supplied system parameters. It is therefore
a conditional predictive interval, not yet a parameter-integrated Bayesian credible
interval.

`seasonal_local_level(period=..., ...)` builds a sum-to-zero dummy-seasonal system.
Its supplied seasonal and level variances remain fixed; it does not estimate them.
The optional `initial_seasonal_effects` is one complete cycle in forecast order and
must sum to zero.

### What may vary with time

Some of the system may vary with time and some may not. The observation row may vary:
`with_observation_rows(rows)` supplies one row per training time. A model built that
way then *requires* `forecast(y, steps, future_observation_rows=R)`: calling
`forecast(y, steps)` on it raises rather than silently reusing the constant row, and `R`
must have exactly `steps` rows. See
[time-varying observation rows](regression-forecasting.md#fixed-parameter-time-varying-observation-rows).
The transition matrix, the process covariance, and the initial mean and covariance are
constant for the whole series and horizon; there is no API to vary them. In the Rust
core the per-time observation variance can also vary, through
`with_observation_variances`; that is used internally by the structural models and is
not exposed in Python, where the observation variance is a single constant.

Whether or not the rows vary, the parameters are supplied, not estimated. Noise is
Gaussian and the observation is univariate. Process covariance may be positive
semidefinite so deterministic state shifts are representable; initial covariance must
be positive definite, and observation variance must be positive. This API does not
estimate system parameters, support multivariate observations, or integrate a Kalman
likelihood into `ModelBuilder`.

Filtering, smoothing, and forecasting release the Python GIL after converting the
input NumPy array.

## Joint hierarchical means for ragged series

`BayesianHierarchicalMean` fits all supplied programs in one posterior:

```text
population_mean ~ Normal(population_mean_prior, population_variance_prior)
group_mean[g] ~ Normal(population_mean, group_variance)
program_mean[p] ~ Normal(group_mean[group_index[p]], program_variance)
y[p, t] ~ Normal(program_mean[p], observation_variance)
```

The three variances have explicit `InverseGammaPrior(shape, scale)` priors. The
specialized conjugate Gibbs kernel draws exact full conditionals, so it does not send
NUTS/HMC through the funnel geometry of a centered hierarchy. Centered Gibbs chains can
still be highly autocorrelated when a variance component approaches zero. This remains
finite MCMC: use multiple chains and inspect `fit.summary()` or `fit.diagnostics()` for
rank-normalized R-hat, bulk/tail ESS, and MCSE, especially with few groups or sparse
programs.

```python
model = rmc.BayesianHierarchicalMean(
    group_variance_prior=rmc.InverseGammaPrior(3.0, 20.0),
    program_variance_prior=rmc.InverseGammaPrior(3.0, 10.0),
    observation_variance_prior=rmc.InverseGammaPrior(3.0, 25.0),
    population_mean_prior=100.0,
    population_variance_prior=400.0,
)
fit = model.fit(
    [program_a, program_b, program_c],
    group_index=[0, 0, 1],
    program_names=["a", "b", "c"],
    group_names=["division-a", "division-b"],
    chains=4,
    warmup=500,
    draws=1_000,
    seed=42,
)
forecast = fit.forecast(steps=12, seed=43)
```

Series are genuinely ragged and each may contain one or more finite observations.
`NaN` positions are ignored by the likelihood, infinities are rejected, and input
program order is preserved. Conditional on the variance draws, the program posterior
mean weights its sample mean by
`n * program_variance / (n * program_variance + observation_variance)`. A one-point
program therefore receives more groupward shrinkage than an otherwise comparable long
program. How strongly either program leans on its group depends on the learned ratio of
program to observation variance; the shrinkage strength is inferred rather than manually
assigned.

`fit.get_samples()` returns scalar population/variance/standard-deviation arrays shaped
`(chain, draw)`, group means shaped `(chain, draw, group)`, and program means shaped
`(chain, draw, program)`. `time_counts`, `observed_counts`, `group_index`,
`program_names`, and `group_names` preserve the ragged hierarchy metadata.
Hierarchical validation and numerical failures raise `InferenceError`, which subclasses
`ValueError`.

Forecast `state_samples` and `observation_samples` have shape
`(chain, draw, program, step)`. Selecting `[:, :, program, :]` produces the same
`(chain, draw, step)` layout as other fitted forecast objects. Rollups must be formed
inside each joint draw:

```python
company_draws = forecast.observation_samples.sum(axis=2)
division_draws = forecast.group_observation_samples  # chain, draw, group, step
assert np.array_equal(company_draws, forecast.total_observation_samples)
```

Forecast storage is contiguous per posterior draw and does not retain a redundant copy
of the static state at every step. To fail predictably instead of risking process OOM,
one fit is limited to 50 million retained parameter values and one forecast call to 25
million materialized observation values. Refit with fewer retained chains/draws, or
reduce the forecast horizon, if a guard is reached.

`BayesianHierarchicalMean` pools a static intercept only. Time ordering does not affect
the fit, and future observations are conditionally iid around each program mean. For a
pooled level that moves over time, see `BayesianHierarchicalDynamicRegression` in
[dynamic GLMs](dynamic-glm.md). This model does not
pool trends or seasonal shapes and is not a stochastic local-level model. A common
within-program observation variance and shared group/program variances are assumed;
variance priors can materially influence singleton programs and weakly populated groups.

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
Gaussian observations. Add known covariates with `fit(..., exog=X,
coefficient_prior=GaussianCoefficientPrior(...))` and supply future `exog` to the
returned regression fit. See [regression and Fourier forecasting](regression-forecasting.md).
Irregular timestamps are not modeled automatically. Gibbs output remains finite
MCMC output, so inspect multiple-chain convergence and effective sample sizes rather
than treating it as an analytic posterior.

## Bayesian seasonal local-level forecasting

`BayesianSeasonalLocalLevel` adds one stochastic sum-to-zero dummy-seasonal component
to a stochastic local level. It jointly samples latent states with multivariate FFBS and
updates level, seasonal, and observation variances from explicit inverse-gamma priors.

```python
model = rmc.BayesianSeasonalLocalLevel(
    period=12,
    level_variance_prior=rmc.InverseGammaPrior(3.0, 2.0),
    seasonal_variance_prior=rmc.InverseGammaPrior(3.0, 1.0),
    observation_variance_prior=rmc.InverseGammaPrior(3.0, 8.0),
    initial_level=100.0,
    initial_seasonal_effects=np.zeros(12),
    initial_level_variance=25.0,
    initial_seasonal_variance=9.0,
)
fit = model.fit(y, chains=4, draws=1000, warmup=500, seed=42)
forecast = fit.forecast(steps=12, seed=43)
lower, upper = forecast.interval(0.95)
cumulative_lower, cumulative_upper = forecast.cumulative_interval(0.95)
```

The initial seasonal vector is one cycle in forecast order and must sum to zero.
`forecast.level_samples`, `seasonal_samples`, `observation_samples`, and
`cumulative_observation_samples` have shape `(chain, draw, step)`. Cumulative paths are
formed inside each posterior draw before quantiles are calculated.

The fitted model is equally spaced, scalar, Gaussian, and single-seasonal. Seasonal
innovations preserve structural identification but do not force every realized rolling
cycle to sum exactly to zero. Missing values retain their time positions. Fitting
requires three finite observations, one per inferred variance, with no full-cycle
minimum. Short histories can be
strongly sensitive to initial-state and variance priors. The regression extension and
`fourier_design` provide a smaller harmonic model for long periods.

All specialized fits expose `summary()`, `diagnostics()`, and `sampler_stats`.
Hamiltonian divergences and acceptance are unavailable for Gibbs/FFBS and exact
conjugate sampling. [Independent batches](forecast-batches.md) preserve cell identity
and return per-cell fits, errors, diagnostics, and forecasts.

For [sparse amounts](sparse-amounts.md), use `BayesianHurdleLogNormal`; for
[payment-event triangles](runoff.md), use `DirichletMultinomialRunoff`. Their data,
prior, and inference contracts are documented separately from Gaussian forecasting.

## Bayesian local-linear-trend forecasting

`BayesianLocalLinearTrend` fits the two-state structural model

```text
[level[-1], slope[-1]] ~ Normal(initial_mean, initial_covariance)
level[t] = level[t-1] + slope[t-1] + Normal(0, level_variance)
slope[t] = slope[t-1] + Normal(0, slope_variance)
y[t] = level[t] + Normal(0, observation_variance)
```

All three variance priors are explicit. The initial covariance is configured with level
variance, slope variance, and an optional level/slope covariance and must be strictly
positive definite.

```python
model = rmc.BayesianLocalLinearTrend(
    level_variance_prior=rmc.InverseGammaPrior(3.0, 0.24),
    slope_variance_prior=rmc.InverseGammaPrior(3.0, 0.04),
    observation_variance_prior=rmc.InverseGammaPrior(3.0, 0.70),
    initial_level=0.0,
    initial_slope=0.0,
    initial_level_variance=4.0,
    initial_slope_variance=1.0,
)
fit = model.fit(y, chains=4, draws=1000, warmup=500, thin=1, seed=42)
forecast = fit.forecast(steps=12, seed=43)
predictive_lower, predictive_upper = forecast.interval(0.95)
level_lower, level_upper = forecast.level_interval(0.95)
slope_lower, slope_upper = forecast.slope_interval(0.95)
```

Level, slope, and observation samples each have shape `(chain, draw, step)` and belong
to the same coherent paths. `interval()` is the observation posterior-predictive
interval; `level_interval()` and `slope_interval()` summarize the latent states. The
process innovations are independent (diagonal process covariance). `NaN` retains a
scheduled missing time point. Variance components can be weakly identified near zero,
so inspect multi-chain convergence.

## Bayesian autoregression AR(p)

`BayesianAutoRegression(order=p)` and its shorter alias `BayesianAR` fit any positive
caller-selected order under the conditional Gaussian likelihood

```text
y[t] = intercept + phi[1] y[t-1] + ... + phi[p] y[t-p] + epsilon[t]
epsilon[t] ~ Normal(0, innovation_variance)
```

The explicit conjugate prior is

```text
innovation_variance ~ InverseGamma(variance_shape, variance_scale)
coefficients | innovation_variance
    ~ Normal(coefficient_mean, innovation_variance * coefficient_precision^-1)
```

Coefficients are always `[intercept, lag_1, ..., lag_p]`.

```python
order = 3
prior = rmc.NormalInverseGammaPrior(
    coefficient_mean=np.zeros(order + 1),
    coefficient_precision=np.eye(order + 1) * 0.05,
    variance_shape=2.5,
    variance_scale=0.2,
)
model = rmc.BayesianAutoRegression(order=order, prior=prior)
fit = model.fit(y, chains=4, draws=1000, seed=42)
forecast = fit.forecast(steps=12, seed=43)
lower, upper = forecast.interval(0.95)
```

This is a directly observed AR(p), distinct from the latent
`LinearGaussianStateSpace.stationary_ar1()` model with separate process and measurement
noise. The conjugate posterior draws are independent, so there is no warmup or thinning.
`fit.get_samples()` returns `coefficient` with shape `(chain, draw, p + 1)` and innovation
variance/standard deviation with shape `(chain, draw)`. Forecast observation and recursive
conditional-mean paths have shape `(chain, draw, step)`.

The likelihood conditions on the first `p` observations. Input must be complete, finite,
equally spaced, and longer than `p`. Coefficient draws are not restricted to the
stationary region and are never silently clipped; explosive recursive draws return a
numerical error. `interval()` is a pointwise equal-tailed posterior-predictive interval,
not a simultaneous path band.

## `ModelBuilder`

```python
builder = rmc.ModelBuilder(data=None, dims=None)
```

Constructs a model. Data can be bound at build time or passed later to `rmc.sample()`,
`rmc.batch_sample()`, or `rmc.sample_prior_predictive()`. `dims` maps a data key to a
named population dimension; keys left out use the compatibility dimension `"obs"`.

The builder also has `data(name, dim=None)`, `potential(name, expression)` for a bare
log-density term (the expression must be scalar), and `deterministic(name, expression)`
for a named quantity recorded alongside the draws. See
[custom models](custom-models.md).

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
| `bernoulli_prior(name, p=0.5)` | Bernoulli(p) | discrete; see the note below |
| `poisson_prior(name, lam)` | Poisson(lam) | discrete; see the note below |

All scalar prior methods return a `ParamRef`. `vector_normal_prior()` returns a `VectorParamRef`.

The two discrete priors are not merely a poor fit for gradient-based inference; they
are refused by it. A model declaring a Bernoulli or Poisson prior raises `ValueError`
from `builder.compile()`, `rmc.sample()`, and `rmc.batch_sample()`. They are usable
only with `sample_prior_predictive()`. Posterior inference needs continuous parameters
or an explicit marginalization you write yourself.

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

That list is the fused fast paths, not the whole vocabulary. Expressions also support
`-`, `/`, `**` and unary negation, and the methods `.exp()`, `.log()`, `.sqrt()`,
`.sigmoid()`, `.tanh()`, `.softplus()`, `.sin()`, `.cos()` and `.sum()`.
`beta["group_key"]` selects one element of a vector parameter per observation, keyed
by an integer-valued data column. [Custom models](custom-models.md) has the details.

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
`batch_fit[i]` returns a `BatchResult`.

`sample_batch(..., errors=...)` chooses what happens when one dataset fails to bind or
sample. The default `errors="raise"` propagates the first failure. With
`errors="collect"`, every cell is attempted and `BatchFit.errors` returns a
`{dataset_id: message}` mapping containing only the failed cells, so an empty mapping
means every cell succeeded. Successful cells are retrieved normally; `batch_fit.get(id)`
and `batch_fit[i]` on a failed cell raise `ValueError` carrying that cell's ID and
message rather than returning a placeholder. `len(batch_fit)` still counts every dataset,
failed cells included. Any other value of `errors` raises `ValueError`.

```python
batch = compiled.sample_batch(
    [good_data, malformed_data], ids=["a", "broken"], errors="collect",
)
batch.errors                 # {"broken": "..."}
fit = batch.get("a")         # BatchResult
batch.get("broken")          # raises ValueError: dataset 'broken': ...
```

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
    target_accept=0.8,
    sampler="nuts",
    max_tree_depth=10,
    num_leapfrog_steps=15,
    show_progress=True,
    init=None,
    metric="auto",
)
```

Returns a `FitResult`.

Notes:

- `sampler` may be `"nuts"` or `"hmc"`.
- `init` supplies starting positions. With `None`, each chain starts at its own random
  point, uniform on (-2, 2) in unconstrained coordinates, as Stan does. A start whose
  log density or gradient is not finite is redrawn; if none of 100 draws works, the
  origin is tried, and failing that `sample()` asks for `init`.
- Chain `c` draws from a stream derived from `seed` and `c`, so different seeds never
  share a chain.
- `metric` sets how warmup adapts the metric of vector parameters. `"diag"` is Stan's
  default diagonal metric. `"dense"` estimates a full covariance for each vector
  parameter of at most 512 elements. `"auto"` (the default) stays diagonal unless a
  vector parameter's warmup draws show correlation well beyond their own sampling
  noise, which suits strongly correlated regression coefficients. Scalar parameters
  are always diagonal.
- Warmup follows Stan's windowed schedule: a 75-draw initial buffer, doubling
  metric windows starting at 25 draws, and a 50-draw terminal buffer, shrinking to
  15%, 75% and 10% of warmup when warmup is too short for those. The last window is
  stretched to meet the terminal buffer rather than cut short.
- `threads=0` uses Rayon defaults.
- `max_tree_depth` applies to NUTS.
- `num_leapfrog_steps` applies to HMC.
- `step_size=0.0` means auto-tune.
- `target_accept` controls dual-averaging during warmup and must be between 0 and 1.

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
    target_accept=0.8,
    max_tree_depth=8,
    num_leapfrog_steps=15,
    show_progress=True,
    metric="auto",
)
```

Returns a list of `BatchResult`, one per model.

Unlike the original throughput-only path, batch sampling now supports multiple chains per
model and both NUTS and fixed-trajectory HMC. Use `chains > 1` when reliability matters more
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
| `transition_diagnostics()` | `dict` | Per-chain and aggregate energy, tree-depth, and leapfrog telemetry |
| `accept_rates()` | `list[float]` | Per-chain accept rates |
| `step_sizes()` | `list[float]` | Per-chain adapted step sizes |
| `divergences()` | `list[int]` | Per-chain divergence counts |
| `posterior_predictive(n_samples=None, seed=42, data=None, expected=False, sizes=None)` | `dict[str, np.ndarray]` | Posterior predictive samples shaped `(n_samples, n_obs)` per likelihood. `data` substitutes new predictors; `expected=True` returns conditional means instead of sampled draws |
| `predict(data=None, seed=42, expected=False, sizes=None)` | `dict[str, np.ndarray]` | Predictive draws on `(chain, draw, obs)` axes |
| `deterministics(data=None, sizes=None)` | `dict[str, np.ndarray]` | Draws of each declared `deterministic()`, on `(chain, draw)` or `(chain, draw, obs)` axes |
| `log_likelihood()` | `dict[str, np.ndarray]` | Pointwise log-likelihood shaped `(chain, draw, obs)` per likelihood |
| `to_arviz(include_ppc=False, ppc_samples=None, ppc_seed=42, include_log_likelihood=True)` | ArviZ inference container | Convert to ArviZ's version-native container (`InferenceData` on 0.x, `DataTree` on 1.x) with observed data, optionally including predictive draws and pointwise log-likelihood |

`log_likelihood()` is the intended bridge for `az.loo(...)` and `az.waic(...)`.

`to_arviz(include_ppc=True)` writes the predictive group with real `(chain, draw, obs)`
axes that line up with the posterior group, rather than flattening the draws into a
single synthetic chain. Predictive draw `(c, d)` is the one generated from posterior
draw `(c, d)`. `ppc_samples` thins the draw axis rather than a flattened pool: the same
`ppc_samples // n_chains` draw indices are kept in every chain, and those indices are
written onto the predictive group's `draw` coordinate, so
`idata.posterior.sel(draw=idata.posterior_predictive.draw)` recovers the parameter draws
that produced each predictive draw. `fit.posterior_predictive()` is unaffected and still
returns `(n_samples, n_obs)`.

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
- one array per declared `deterministic()`: 1-D when the expression is scalar, and
  `(n_samples, n)` when it is vector-valued

For automatically non-centered scalar hierarchical normals, the returned parameter draws use
the logical parameter name, not the hidden raw latent.

A `potential()` term is a bare log-density with no generator behind it, so a model
declaring one cannot be simulated forward. This call raises `ValueError` for such a
model. Bernoulli and Poisson priors, by contrast, are supported here and only here.

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

That scalar hierarchical pattern compiles to the non-centered form described under
[hierarchical priors](#hierarchical-priors-and-automatic-non-centering): the sampled
coordinate is a standard normal `raw`, and the value reported under the declared name is
`mu + sigma * raw`. Vector priors are not rewritten this way, because
`vector_normal_prior` takes constant hyperparameters; write a pooled vector block
explicitly as `sigma * z[key]`.

## Exceptions

`rustmc.StateSpaceError` reports invalid model structure or a numerical failure inside
a native kernel. `rustmc.ParameterError` reports an invalid parameter or expression,
including mixing references from two builders. `rustmc.InferenceError` reports invalid
inputs or a numerical failure in a fitted Bayesian model; the hierarchical entry points
raise it. All three subclass `ValueError`.

The forecasting models do not agree on one exception class. The local-level, trend,
seasonal and AR models raise `StateSpaceError`, as does structural fitting; runoff
raises plain `ValueError`. Because every one of these subclasses `ValueError`,
`except ValueError` catches them all, and that — not `except InferenceError` — is the
form to write if you want to catch a forecasting failure.

## Result types

Fitting and forecasting calls return named types, and each guide describes the members
of the types it returns. `FitResult`, `BatchResult`, `BatchFit`,
`StructuralFit`/`StructuralForecast`, `RunoffFit`,
`ForecastBatchFit`/`ForecastBatchForecast`, `DynamicGLMForecast`,
`KalmanFilterResult`/`KalmanSmootherResult`, and one fit type plus one forecast type
per specialized model come from those calls and are not constructed directly. The
forecast type is not reliably named after the fit type:
`BayesianLocalLevelFit.forecast()` returns `BayesianForecastResult`,
`BayesianLocalLinearTrendFit.forecast()` returns `BayesianTrendForecast`,
`BayesianSeasonalLocalLevelFit.forecast()` returns `BayesianSeasonalForecast`, and
`BayesianHierarchicalMeanFit.forecast()` returns `BayesianHierarchicalForecast`. Others
do follow the pattern — `BayesianARFit` → `BayesianARForecast`,
`BayesianHurdleLogNormalFit` → `BayesianHurdleLogNormalForecast`, and
`BayesianRegressionFit` → `BayesianRegressionForecast` — so the name cannot be guessed
either way. Check the fit's own page. `ForecastDraws`, `NamedDesign`, `ScenarioForecast`,
`BacktestResult` and `BacktestFold` are ordinary dataclasses you may also construct
yourself, which is how you score draws that rustmc did not produce.

Not everything returns a wrapper. `fit.predict()`, `fit.deterministics()`,
`fit.get_samples()`, `fit.get_samples_2d()`, `fit.log_likelihood()` and
`fit.posterior_predictive()` return plain dicts of NumPy arrays, and the evaluation
helpers return NumPy arrays or dicts of them.
