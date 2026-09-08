# Sparse nonnegative amounts

`BayesianHurdleLogNormal` fits a point mass at zero and a changing positive-payment
level. It is suitable when each period either has no payment or a positive amount.
It accepts exact zeros, one-positive histories, and all-zero histories; `NaN` marks
missing observations and retains the time position. Negative amounts require a
different model.

```python
import numpy as np
import rustmc as rmc

model = rmc.BayesianHurdleLogNormal(
    process_variance_prior=rmc.InverseGammaPrior(4.0, 0.03),
    observation_variance_prior=rmc.InverseGammaPrior(4.0, 0.3),
    occurrence_alpha=1.0,
    occurrence_beta=1.0,
    initial_log_level=np.log(100.0),
    initial_variance=0.2,
    process_variance_upper=1.0,
    observation_variance_upper=4.0,
)
fit = model.fit(
    np.array([0., 0., 110., np.nan, 0., 90., 0., 0.]),
    chains=4, warmup=500, draws=1000, seed=42,
)
forecast = fit.forecast(steps=12, seed=43)
lower, upper = forecast.interval(0.95)
total_draws = forecast.observation_samples.sum(axis=2)
print(fit.summary())
```

The model is

```text
p ~ Beta(occurrence_alpha, occurrence_beta)
paid[t] ~ Bernoulli(p)
level[-1] ~ Normal(initial_log_level, initial_variance)
level[t] ~ Normal(level[t-1], q)
log(y[t]) | paid[t] = 1 ~ Normal(level[t], r)
y[t] | paid[t] = 0 = 0
```

`q` and `r` have the specified inverse-gamma priors conditioned on
`0 < q <= process_variance_upper` and `0 < r <= observation_variance_upper`.
The upper bounds are part of the statistical model, not clipping of sampled draws.
Their defaults are 1 and 4 in squared log units. Choose priors and bounds to match
plausible period-to-period changes and positive-amount dispersion. Initial level and
all variance settings are on the natural-log scale.

Finite bounds ensure finite predictive amount moments over a finite horizon. Without
bounds, inverse-gamma log-variance mixtures have infinite positive raw moments even
after finite data. The exact conditional sampler rejects variance proposals above
the bounds and reports an error if too little conditional mass falls below a bound.
It never clips variance draws, discards inconvenient posterior draws, or converts an
overflowing payment to zero. Very large log levels/horizons can still overflow machine
arithmetic and are reported as inference failures.

Occurrence probability is static, with posterior
`Beta(alpha + positive_count, beta + observed_zero_count)`. Occurrence and positive
severity have independent priors and factorized likelihoods. Severity is sampled by
Gaussian FFBS and truncated inverse-gamma Gibbs updates. On zero months severity
continues to evolve, but receives no amount observation. Missing months also give
no occurrence observation. All-zero histories use independent prior severity draws;
`fit.severity_informed_by_data` and sampler metadata make that distinction explicit.

Forecast arrays have shape `(chain, draw, step)`:

| Attribute | Meaning |
| --- | --- |
| `observation_samples` | Realized payments, including exact zeros |
| `positive_mean_samples` | `exp(level + r/2)`, conditional mean given payment |
| `mean_samples` | `p * exp(level + r/2)`, conditional mean including zero probability |
| `mean` | Monte Carlo average of conditional means |
| `observation_mean` | Monte Carlo average of realized payment draws |
| `cumulative_observation_samples` | Cumulative payments computed inside each draw |

`interval()` and `cumulative_interval()` return predictive equal-tailed intervals;
`mean_interval()` summarizes conditional mean uncertainty. With high zero probability,
an interval can have zero as its lower bound or both endpoints. Diagnose probability,
variance parameters, and the terminal log level using `diagnostics()`/`summary()`.
`sampler_stats()` reports Hamiltonian divergences and acceptance as unavailable.
`to_arviz()` exports parameter draws and the observed amounts when ArviZ is installed.

The standalone `hurdle_lognormal_logp(y, payment_probability, log_level, log_variance)`
evaluates the mixed point-mass/continuous density. This release exposes a specialized
dynamic fitted model; it does not add a new generic graph-builder likelihood or claim
that arbitrary non-Gaussian observations can use the Gaussian FFBS kernel.

The occurrence process has no calendar covariates or time dependence in this model.
The positive severity component has a local level, without exogenous features or
hierarchical pooling. Compare zero frequency, positive-amount tails, and cumulative
coverage on rolling origins before choosing the model for a particular cell family.

Independent sparse cells use the same [native batch API](forecast-batches.md) as
Gaussian forecasting models:

```python
batch = model.fit_batch(
    [np.zeros(12), np.array([0., np.nan, 110., 0.])],
    ids=["no-payments", "one-payment"],
    chains=4, draws=1000, warmup=500, seed=42,
    threads=4, chunk_size=32, errors="collect",
)
print(batch.errors)
print(batch["no-payments"].severity_informed_by_data)  # False
print(batch.diagnostics())
future = batch.forecast(12, seed=43, threads=4, errors="collect")
paths = future["one-payment"].observation_samples
```

Supply `models=[model_for_first_cell, model_for_second_cell]` to vary occurrence
priors, log-level priors, or variance bounds; `None` uses the calling model. Mixed
batches can include the other supported forecasting models. Hurdle cells explicitly
reject training or future `exog` and coefficient priors; regression cells in a mixed
batch may supply them. All-zero and one-positive histories keep their distinct
severity-information metadata and ordinary fit result types.

Seeds use stable cell IDs for fitting and coherent forecasts, so results are
unchanged by reordering, resuming a subset, chunk size, or worker count. Per-cell
validation, allocation-limit and numerical failures can be collected alongside
successful cells. The shared worker and retained-memory limits apply; use
caller-managed slices to retain a larger workload. Each cell remains independent:
batching introduces no shared occurrence shocks or hierarchical pooling.
