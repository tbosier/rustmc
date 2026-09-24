# Forecasting workflows

Start here for time series. This page covers the shape of a forecasting job — fit,
predict, evaluate against a baseline, and read the diagnostics — and the axis
conventions every forecasting model shares. The pages that follow it choose a
particular model: [structural composition](structural-forecasting.md),
[calendar seasonality](regression-forecasting.md),
[counts and intermittent amounts](dynamic-glm.md),
[sparse amounts](sparse-amounts.md), and [payment runoff](runoff.md).

These models do not use `ModelBuilder` or the NUTS sampler. Each has its own
constructor and its own kernel: Gibbs with FFBS for the Gaussian state-space models,
exact conjugate draws for AR, block elliptical slice sampling for dynamic GLMs, and for
runoff either exact conjugate draws or latent-count Gibbs, depending on whether every
ultimate total is known. A fit's `sampler_stats` names the one that ran. Where the
draws are exact and independent there is no warmup and no convergence period, so read
`sampler_stats` before interpreting a diagnostic.

They return their own fit classes rather than the `FitResult` from
[Get started](getting-started.md). `summary()`, `diagnostics()` and `sampler_stats`
are common to all of them. Beyond that they diverge: most expose `get_samples_2d()` and
`forecast(steps)`, but `RunoffFit` does neither — it exposes `allocation_samples()`,
`ultimate_samples()`, `calendar_samples(steps)` and the other arrays described in
[payment runoff](runoff.md). Check the page for the model you are using.

One difference from a graph fit: none of these samplers has a divergence or an accept
rate, so `sampler_stats` reports both as `None`. R-hat, ESS and MCSE apply as before.

The Python package combines native inference with reusable evaluation and result tools.
Forecasts have leading `(chain, draw)` axes and a final horizon axis. Panel forecasts
insert a series axis before the horizon. Keep these axes together when aggregating:
paired draws carry dependence across horizons and series.

## Fit, predict, and check a model

```python
import numpy as np
import rustmc as r

rng = np.random.default_rng(7)
y = 10 + np.cumsum(rng.normal(0, .15, 30)) + rng.normal(0, .4, 30)
model = r.StructuralModel([
    r.StructuralComponent.level("baseline", r.VarianceParameter.inverse_gamma(3, .1),
                                initial_mean=10., initial_variance=2.),
], observation_variance=r.VarianceParameter.inverse_gamma(3, .3))
fit = model.fit(y, chains=4, draws=1000, warmup=1000, store_states=True)
print(fit.summary())
forecast = r.ForecastDraws.from_result(fit.forecast(6), dates=[
    "2026-10", "2026-11", "2026-12", "2027-01", "2027-02", "2027-03"])
lower, upper = forecast.interval(.9)
total_lower, total_upper = forecast.interval(.9, cumulative=True)
forecast.save("forecast.npz")
restored = r.ForecastDraws.load("forecast.npz")
```

`observation_samples` includes realization noise. `mean_samples`, when exposed,
contains conditional expected responses given each parameter/state draw. These are
different uncertainty questions. Heavy-tailed priors can make unconditional moments
nonexistent even when every conditional mean is finite. Empirical summaries cannot
establish moment existence. Dates are explicit labels; the library does not infer frequency,
fill missing calendar rows, or generate holidays.

Use `model.prior_predict(...)` for structural models and `model.prior_predictive(...)`
for dynamic GLMs before fitting. Historical states/components are optional for structural
fits. The [structural guide](structural-forecasting.md) explains state timing and priors;
the [dynamic GLM guide](dynamic-glm.md) explains panel/exposure shapes and fixed scales.

## Compare at historical forecast origins

```python
result = r.backtest(model, y, horizon=3, initial=18, step=3,
                    fit_kwargs={"chains": 4, "draws": 1000, "warmup": 1000},
                    levels=(.5, .8, .95), errors="collect")
print(result.summary())             # One score per horizon, averaged over folds/series
print(result.summary(baseline=True))
print(result.errors, result.baseline_errors)
```

Each origin refits only the preceding data and scores the following complete horizon.
Pass `exog` as `(time, feature)` or `(series, time, feature)` and exposure with the same
shape as observations; the evaluator slices training and future rows. A callable model
factory receives only a training copy and can learn preprocessing or empirical priors
inside that fold. Other model-specific inputs passed through kwargs are the caller's
responsibility to prepare without future information. Seeds derive from origin IDs.

The default baseline is a random-walk bootstrap; `baseline_period=12` uses seasonal
differences and `baseline_period=None` disables it. Baseline failures are reported
separately without discarding successful model scores. Compare models on the same
successful origins; aggregate scores alone do not reveal omitted failures.

Scores include empirical CRPS, standard WIS, mean bias/error, interval coverage, and
interval width. WIS divides the weighted median/interval numerator by `K + .5` for
`K` intervals. `interval_score` is a single-interval score. Missing realized outcomes
produce NaN scores, which summaries omit. Infinite losses (for example, an
unrepresentably large squared error) remain infinite in summaries rather than
being omitted. Coverage and widths are pointwise, not simultaneous bands.

The [retained calibration study](https://github.com/tbosier/rustmc/blob/main/benchmarks/results/2026-09-09-calibration.md)
uses 32 independently generated datasets per model and reports convergence, coverage,
CRPS, and Monte Carlo uncertainty. Its companion pilot retains short-run convergence
failures. These checks average over generating priors under correct specification;
they do not establish calibration on every dataset or under misspecification.

## Score draws and baselines directly

The scoring functions `backtest` uses are exported, so any set of draws can be scored
without going through a fold loop. Each takes `samples` shaped `(sample, *target)` or
`(chain, draw, *target)` and `actual` shaped `target`, flattens the leading axes, and
returns an array of shape `target`. Draws must be non-empty and finite; `actual` may be
NaN for a missing outcome and the NaN propagates into the score. Lower is better for
every loss below.

`crps(samples, actual)` is the empirical CRPS, `mean|X - y| - mean|X - X'| / 2`, using
the `1/n**2` normalization rather than the unbiased `1/(n*(n-1))` one. It is computed
from sorted draws, so cost is `O(n log n)` and no pairwise matrix is formed.

`weighted_interval_score(samples, actual, levels=(.5, .8, .95))` is the standard WIS:
the median term plus one weighted interval score per level, divided by `K + .5` for `K`
levels. `levels` are central-interval coverages, must be one-dimensional, unique, and
strictly inside `(0, 1)`.

`interval_score(actual, lower, upper, alpha=.05)` scores one interval from precomputed
bounds: `(upper - lower) + (2/alpha) * (lower - y)+ + (2/alpha) * (y - upper)+`.
Arguments broadcast against each other. Note the polarity: `alpha` here is the
non-coverage probability, so `alpha=.05` and `levels=(.95,)` describe the same interval.

`score_forecast(samples, actual, levels=(.5, .8, .95))` returns a dict with `bias`,
`absolute_error`, `squared_error`, `crps`, `wis`, and `coverage_<level>` /
`width_<level>` for each level, every value shaped like `actual`. `bias` is
`mean(draws) - actual`, and `absolute_error` and `squared_error` are that one error's
absolute value and square. They are point scores on the predictive mean, not the mean
absolute or squared error of the draws.

Two baselines produce comparison forecasts from observations shaped `(*series, time)`.
`seasonal_naive(observations, steps, period=1)` repeats the last complete cycle and
returns `(*series, steps)`; it needs `time >= period` and finite values in that final
cycle, but tolerates NaN earlier. `naive_forecast(observations, steps, *, period=1,
draws=1000, seed=42)` is its probabilistic counterpart: the same point path plus a
bootstrap of centred seasonal differences, accumulated so spread grows every `period`
steps. It returns `(1, draws, *series, steps)` — a leading singleton chain axis, so the
result can be passed straight to `score_forecast`. Innovations are resampled as whole
time columns, which preserves empirical cross-series dependence, and a time column is
dropped if any series is non-finite there. It needs `time > period`, one seasonal
difference more than `seasonal_naive`, and it does not integrate uncertainty in the
innovation distribution.

```python
import numpy as np
import rustmc as r

rng = np.random.default_rng(0)
y = 10. + np.tile([0., 3.], 10) + rng.normal(0., 1., 20)   # 20 periods, cycle of 2
baseline = r.naive_forecast(y[:16], 4, period=2, draws=500, seed=9)   # (1, 500, 4)
actual = y[16:]

r.crps(baseline, actual)                       # (4,)
r.weighted_interval_score(baseline, actual)    # (4,)
scores = r.score_forecast(baseline, actual, levels=(.5, .95))
scores["crps"], scores["coverage_0.95"], scores["width_0.95"]

lower, upper = np.quantile(baseline.reshape(-1, 4), [.025, .975], axis=0)
r.interval_score(actual, lower, upper, alpha=.05)
r.seasonal_naive(y[:16], 4, 2)                 # (4,) point forecast
```

`backtest(model, observations, *, horizon, ...)` wraps this loop over rolling origins
and returns a `BacktestResult` whose `.folds` are `BacktestFold` records. Its `errors`
argument defaults to `"raise"`, so one failing fold aborts the run unless
`errors="collect"` is passed.

## Named predictors, scenarios, and updates

`NamedDesign(array, ("price", "holiday"))` records feature identity and order. Wrap a
model in `ForecastSession(model, y, exog=named_design, fit_kwargs=...)`; subsequent
forecast/update designs must have exactly those names in the same order. Unnamed
native model inputs validate shape only.

`forecast_scenarios(session, {"base": base_design, "promotion": promotion_design},
probabilities={"base": .7, "promotion": .3})` retains each conditional forecast.
Its `.mixture(draws=1000, seed=...)` samples whole paths, preserving time dependence.
Scenario weights describe a user-specified distribution independent of posterior
parameters; they are not inferred. Direct native fits accept unnamed scenario arrays.

`session.update(new_y, exog=new_design)` appends observations and performs a complete
parameter-posterior refit. The session changes only after success. It is not an
incremental fixed-parameter filtering operation. Retain data securely if saving fits:
generic and dynamic fits include training observations, while the structural artifact
retains state and system information needed for forecasting. `ForecastDraws` archives
contain samples and metadata without executable pickle objects.

## Custom models and independent batches

Custom graph fits support `.predict(data={"x": future_x}, expected=False)` without
dummy outcomes; outputs retain chain/draw axes. `.deterministics(data=...)` computes
named expressions. Use `.to_json()` / `FitResult.from_json()` to preserve training data,
posterior and predictive replay; compiled-model JSON omits training values.

`compiled.sample_batch(datasets, ids=..., threads=4, chunk_size=64, errors="collect")`
uses one bounded native pool and returns an ordered `BatchFit`. Inspect `.errors`,
retrieve a successful item with `.get(id)`, and use `.diagnostics()`, `.predict()`, or
`.fit` for the full result. With `errors="raise"`, cell errors raise with their ID.
The default `seed_policy="cell_id_v1"` is stable under reordering, thread count and
chunk-size changes. `seed_policy="position_v0"` seeds each cell by its position
instead. Either way, chains within a cell draw different streams than in 0.12, so
seeded draws differ from 0.12. IDs generated by omission are positions; supply
persistent IDs for reorder invariance. Chunks bound execution submission; all supplied
datasets and returned results are retained in memory. The older global `batch_sample`
uses positional cell seeds and the same native batch path.

Single fits accept `init=[[raw_parameter_values_for_chain_1], ...]`. Values are in
the unconstrained coordinates reported by the compiled graph parameter order, before
positive/probability transforms and noncentered display conversions.
Compiled batches accept `init={cell_id: [[chain_1_values], ...]}` with the same
coordinate contract; invalid cell initializations participate in collected errors.

For a Rust model outside the graph language, implement `rustmc_core::target::LogDensity`
and use `sample_target`. Supply the complete unconstrained log density and gradient,
including any Jacobians. `Ok(-inf)` rejects points outside support; `Err` reports an
evaluation failure. This interface supplies inference only; custom predictive simulation
remains a separate model operation.
