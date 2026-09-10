# Forecasting workflows

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
produce NaN scores. Coverage and widths are pointwise, not simultaneous bands.

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
chunk-size changes. `seed_policy="position_v0"` reproduces the previous positional
scheme. IDs generated by omission are positions; supply persistent IDs for reorder
invariance. Chunks bound execution submission; all supplied datasets and returned
results are retained in memory. The older global `batch_sample` keeps its positional
seed contract for compatibility.

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
