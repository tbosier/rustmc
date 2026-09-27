# Interventions and changepoints

This page covers `TimeDesign`, which builds regression columns for events you know
about — a contract change, a price change, an outage, a launch — and for changes in
trend at dates you do not know. Pass its columns as `exog` to the forecasting models
on the [regression page](regression-forecasting.md), to `StructuralModel`, or to the
dynamic GLMs. Read it if your series has a break and you want the forecast to account
for it.

A `TimeDesign` fixes the training window when you build it. Its `future(steps)` rows
come from the same column definitions evaluated at the positions after that window,
so they continue the training rows exactly. You never give the forecast rows their
own start position, which is where hand-built intervention columns usually go wrong.

## Build a design and fit it

```python
import numpy as np
import rustmc as rmc

# 100 training steps and 12 to forecast. At step 60 the level jumps by 3 and the
# slope rises by 0.08 per step; step 30 has a one-off drop of 2.
rng = np.random.default_rng(1)
n, steps = 100, 12
t = np.arange(n + steps)
y = (10 + 0.02 * t + 3.0 * (t >= 60) + 0.08 * np.maximum(0, t - 60)
     - 2.0 * (t == 30) + np.cumsum(rng.normal(0, 0.03, n + steps))
     + rng.normal(0, 0.3, n + steps))

D = rmc.DesignTerm
design = rmc.TimeDesign([
    D.ramp("trend", at=0),            # the ordinary linear trend
    D.step("shift", at=60),           # level shift
    D.ramp("slope_change", at=60),    # change in slope
    D.pulse("outage", at=30),         # one-step effect
], length=n)
print(design.names)   # ['trend', 'shift', 'slope_change', 'outage']

prior = rmc.GaussianCoefficientPrior(np.zeros(4), np.diag([0.01, 25.0, 0.01, 25.0]))
model = rmc.BayesianLocalLevel(rmc.InverseGammaPrior(3, 0.01), rmc.InverseGammaPrior(3, 0.2),
                               initial_mean=10.0, initial_variance=4.0)
fit = model.fit(y[:n], exog=design.training(), coefficient_prior=prior,
                chains=4, draws=500, warmup=500, seed=1)
forecast = fit.forecast(steps, exog=design.future(steps), seed=2)

beta = fit.get_samples_2d()["coefficients"].reshape(-1, design.width)
lower, upper = np.quantile(beta, [0.025, 0.975], axis=0)
for name, low, high in zip(design.names, lower, upper):
    print(f"{name:13s} [{low:6.3f}, {high:6.3f}]")
```

Each coefficient gets its own prior variance, in units of `y` per unit of its column.
Choose them for your series. The coefficient intervals are equal-tailed posterior
intervals; the forecast intervals are posterior-predictive intervals conditional on
the design.

## The terms

Positions are integers on the same calendar as `fourier_design`: training row `i` is
position `start + i`, and `start` defaults to 0. Every term is a function of the
position `t` alone.

| Term | Column at position `t` | The coefficient is |
|---|---|---|
| `DesignTerm.step(name, at)` | 1 if `t >= at`, else 0 | the level shift from `at` on |
| `DesignTerm.pulse(name, at)` | 1 if `t == at`, else 0 | a one-period effect |
| `DesignTerm.ramp(name, at)` | `max(0, t - at)` | the change in slope per step after `at` |
| `DesignTerm.decay(name, at, rate)` | `rate**(t - at)` if `t >= at`, else 0 | the effect at `at`; it shrinks by `rate` each step |
| `DesignTerm.fourier(name, period, harmonics)` | the `fourier_design` columns | seasonal weights |
| `DesignTerm.changepoints(name, ...)` | `max(0, t - s)` for each changepoint `s` | the change in slope at `s` |
| `DesignTerm.regressor(name)` | values you supply | an ordinary regression effect |

A decay term's total effect over all later steps is `coefficient / (1 - rate)`.
`rate` must lie strictly between 0 and 1 and is fixed, not estimated; a pulse is the
`rate = 0` case and a step the `rate = 1` case. A ramp column is 0 at `at` and 1 one
step later, so the trend has no jump; pair it with a step for a jump as well. A ramp at
the first training position is a plain linear trend.

Column names are the term name, except for Fourier terms (`annual_sin1`,
`annual_cos1`, and so on, with only the cosine at the Nyquist harmonic, as in
`fourier_design`) and changepoint terms (`trend@24`, one per position). Term names and
column names must be unique within a design.

### Where the event falls

The columns follow their definitions wherever `at` lies. That has consequences worth
knowing before you fit:

- **Before the training window.** A step is 1 on every training row, so it is
  indistinguishable from the level or an intercept; only its prior separates them. A
  pulse is 0 everywhere and its coefficient is the prior. A ramp keeps counting from
  `at`, and a decay has already decayed by the first row.
- **At the first training position.** A step is again constant over the training
  rows. A ramp at `start` is the linear trend.
- **After the training window, inside the forecast horizon.** Every training value is
  0, so the data say nothing about the effect: the coefficient is its prior, and the
  forecast carries that prior uncertainty from `at` on. Use this for a planned change
  whose size you can state as a prior. Its future rows switch on at the right step.
- **Late in the training window.** Only the few rows after `at` inform the effect, so
  its interval is wide.

### From dates to positions

`TimeDesign` takes integer positions, not dates. Convert a date with the index of your
series:

```python
months = np.arange("2019-01", "2026-01", dtype="datetime64[M]")
contract = int(np.searchsorted(months, np.datetime64("2023-04")))
print(contract)   # 51: the 52nd month of the series
```

Check that the date is actually in the index; `searchsorted` returns the insertion
point for a missing date without complaint. With pandas, `index.get_loc(date)` raises
instead.

## Future rows and your own regressors

`design.training()` returns `(length, width)` rows for positions
`start .. start + length - 1`, and `design.future(steps)` returns `(steps, width)` rows
for the positions right after them. Build the design once and use the same object for
both. A column you measure yourself, such as price or promotions, is a
`DesignTerm.regressor`, and its values are passed by name:

```python
promo = rng.binomial(1, 0.3, n + steps).astype(float)
design = rmc.TimeDesign([
    D.step("shift", at=60),
    D.fourier("annual", period=12, harmonics=2),
    D.regressor("promo"),
], length=n)
X = design.training(regressors={"promo": promo[:n]})
X_future = design.future(steps, regressors={"promo": promo[n:]})
print(X.shape, X_future.shape)   # (100, 6) (12, 6)
```

Regressor values need exactly one finite number per row requested. A missing name, an
unknown name, a wrong length, `NaN` or a boolean array raises a `ValueError` naming the
regressor. Other design errors also raise a plain `ValueError` naming the term or
argument at fault, as `fourier_design` does.

For a rolling-origin evaluation with `backtest`, build the design over every
observation and pass `design.training()` as `exog`; `backtest` slices it at each
origin, and the deterministic columns need no future information. A changepoint grid
is the exception: its default positions come from the whole window, so fix them with
`at=` if each fold must not see the full length of the series.

## Changepoints in the trend

For a trend that may change slope at dates you do not know, `DesignTerm.changepoints`
adds one ramp column per candidate position, as Prophet does. With a base linear trend
`D.ramp("trend", at=0)` the fitted trend is piecewise linear: its slope after step `t`
is the trend coefficient plus every changepoint coefficient at or before `t`.

```python
# The slope is 0.05, then 0.20 from step 35, then -0.05 from step 65.
slope = np.where(t < 35, 0.05, np.where(t < 65, 0.20, -0.05))
y2 = 5 + np.concatenate([[0.0], np.cumsum(slope[:-1])]) + rng.normal(0, 0.3, n + steps)

trend_design = rmc.TimeDesign([D.ramp("trend", at=0), D.changepoints("cp", count=10)], length=n)
print(trend_design.changepoints)   # {'cp': [8, 16, 24, 32, 40, 47, 55, 63, 71, 79]}
k = trend_design.width
trend_fit = rmc.BayesianLocalLevel(rmc.InverseGammaPrior(3, 0.002), rmc.InverseGammaPrior(3, 0.2),
                                   initial_mean=5.0, initial_variance=4.0).fit(
    y2[:n], exog=trend_design.training(),
    coefficient_prior=rmc.GaussianCoefficientPrior(np.zeros(k), 0.01 * np.eye(k)),
    chains=4, draws=300, warmup=300, seed=3)
trend_forecast = trend_fit.forecast(steps, exog=trend_design.future(steps), seed=4)
deltas = trend_fit.get_samples_2d()["coefficients"].reshape(-1, k)[:, 1:].mean(axis=0)
print(dict(zip(trend_design.names[1:], deltas.round(3).tolist())))
```

Give either explicit positions with `at=[...]` or a grid with `count=`. The grid places
`count` evenly spaced positions after `first`, ending at
`first + floor(fraction * (last - first))`, and rounds each to the nearest integer
(halves up). `first` and `last` default to the first and last training positions, and
`fraction` defaults to 0.8, Prophet's default: candidates stop before the last fifth
of the history, where a slope change has few points to estimate it but moves the
forecast most. `first` itself is not a candidate, because a hinge there is the base
trend. `design.changepoints` shows the resolved positions. A grid needs at least
`count` positions in its window.

**This is shrinkage, not detection.** The coefficient priors in these models are
Gaussian, so the changepoint coefficients are shrunk towards zero the way ridge
regression shrinks, not the way a sparse prior does. In the test suite's four seeded
versions of the example above, the slope change of +0.15 at step 35, which lies
between candidates 32 and 40, was split between them (posterior means 0.03–0.14 at 32
and 0.06–0.13 at 40), and candidates where nothing changed still had posterior means
of up to about ±0.07. The fitted slope in each segment was within 0.03 of the truth;
the individual coefficients do not say where a break happened. No part of this
estimates a break date or tests whether a break occurred.

A sparse prior such as the Laplace or horseshoe, which would push most changepoint
coefficients to near zero and leave a few large, is not available. The regression
models sample coefficients jointly with the states from a Gaussian prior, and adding a
scale-mixture prior would change that sampler; it is not part of this feature.

The prior variance sets how far the trend can bend: a larger variance lets the fit
follow noise, a smaller one flattens real changes. Changepoint coefficients are in
units of `y` per step, so a variance that suits monthly data is far too large for
daily data. Assess the choice with a rolling-origin `backtest`.

## Using a design with other models

`StructuralModel` reads regression columns in the order of its regression components.
`design.term_slice(name)` gives the columns of one term, so you can give each term its
own component, prior, and, if you want, time-varying coefficients:

```python
C, V = rmc.StructuralComponent, rmc.VarianceParameter
components = [C.level("level", V.inverse_gamma(3, 0.01), initial_mean=10.0, initial_variance=4.0)]
for term in design.terms:
    width = len(design.names[design.term_slice(term.name)])
    components.append(C.regression(term.name, initial_mean=np.zeros(width),
                                   initial_covariance=9.0 * np.eye(width)))
structural = rmc.StructuralModel(components, observation_variance=V.inverse_gamma(3, 0.2))
structural_fit = structural.fit(y[:n], exog=X, chains=2, draws=300, warmup=300, seed=5)
structural_forecast = structural_fit.forecast(steps, exog=X_future, seed=6)
```

Keep the terms in the design in the same order as the components.

The dynamic GLMs take `exog[group][time][feature]`. Broadcast the same design to every
group:

```python
groups = 3
counts = rng.poisson(np.exp(1.0 + 0.7 * (t[:n] >= 60)), size=(groups, n)).astype(float)
count_design = rmc.TimeDesign([D.step("shift", at=60)], length=n)
glm_fit = rmc.BayesianDynamicPoisson(initial_mean=1.0, process_sd=0.05).fit(
    counts, exog=np.broadcast_to(count_design.training(), (groups, n, 1)).copy(),
    chains=2, draws=300, warmup=300, seed=7)
glm_forecast = glm_fit.forecast(
    steps, exog=np.broadcast_to(count_design.future(steps), (groups, steps, 1)).copy(), seed=8)
```

The dynamic GLMs add their own intercept, so do not include a step that is 1 on every
training row. Their coefficient prior is one `coefficient_sd` for every column, on the
linear-predictor scale (the log rate for Poisson); ramp and changepoint columns grow
with time, so scale them, or prefer steps, pulses and decays there.

## What the tests show

`tests/test_time_design.py` checks the exact matrix of each term, including events
before the window, at position 0, on the last training row and inside the forecast
horizon, and that `future()` rows equal the matching rows of one longer design for
every term type. On a simulated series with a level shift of 3 and a slope change of
0.08 at step 60 (the first example above, with four seeds):

- the 95% intervals for both effects contain the true values and exclude zero, and the
  coefficient R-hat is below 1.01;
- fitted to the same series without the break, both intervals contain zero;
- over eight seeds, the post-break forecast CRPS is 0.20 with the columns and 0.35
  without them, and lower on every seed;
- on one series, a rolling-origin backtest from four origins after the break gives
  CRPS 0.16 with the columns, 0.32 without, and 0.71 for the naive baseline.

For the piecewise-linear trend above, the changepoint basis gives forecast CRPS 0.21
against 0.75 without it, over four seeds. One seeded `StructuralModel` fit and one
`BayesianDynamicPoisson` fit each recover a level shift from the same kind of design,
with 95% intervals that contain the truth and exclude zero.

These are checks on simulated data built to match the model. They do not show that a
real series has the breaks you encode, or that the dates are right.

## Limitations

- You supply the dates. Nothing here detects a break or estimates its date.
- Changepoint shrinkage is Gaussian (ridge-like), with no sparse prior; see above.
- A decay term's rate is fixed, not estimated.
- Positions are integers; convert dates yourself.
- In the changepoint test, the local level's variance mixed slowly (R-hat 1.04–1.19
  at 300 draws) while the coefficients mixed well (R-hat below 1.01). This resembles
  the slow variance mixing the project roadmap describes for the local linear trend;
  check `fit.diagnostics()`.
- An event on or before the first training row, or after the last, is identified only
  by its prior. Check the design before fitting if the date could be wrong: a column
  of zeros is not an error.
