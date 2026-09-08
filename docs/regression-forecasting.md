# Regression and calendar seasonality

`BayesianLocalLevel`, `BayesianLocalLinearTrend`, and
`BayesianSeasonalLocalLevel` accept keyword-only `exog` and `coefficient_prior`
arguments in `fit`. An exogenous fit returns `BayesianRegressionFit`. Fits without
exogenous data retain their original result classes and sampling paths.

```python
import numpy as np
import rustmc as rmc

model = rmc.BayesianLocalLevel(
    rmc.InverseGammaPrior(3.0, 0.08),
    rmc.InverseGammaPrior(3.0, 0.4),
    initial_mean=0.0, initial_variance=4.0,
)
X = np.column_stack([promotion, price_change]).astype(float)
prior = rmc.GaussianCoefficientPrior(
    mean=np.zeros(2), covariance=np.diag([1.0, 0.25]),
)
fit = model.fit(y, exog=X, coefficient_prior=prior,
                chains=4, draws=1000, warmup=500, seed=42)
forecast = fit.forecast(steps=12, exog=X_future, seed=43)
lower, upper = forecast.interval(0.95)
cumulative_lower, cumulative_upper = forecast.cumulative_interval(0.95)
```

Choose coefficient and variance priors for the scales of your observations and
features. Coefficient covariance must be finite, symmetric, and strictly positive
definite. The coefficient prior is independent of the variance priors. Coefficients
remain uncertain and constant through time; the observation design changes.
The model samples coefficients and structural states together with augmented-state
FFBS, then updates each variance from those same sampled states and residuals.
The coefficient block has exactly zero process noise.

Training `exog` must have shape `(len(y), features)` and contain finite numbers,
including at missing observations. `NaN` in `y` preserves the calendar position;
infinite observations are rejected. At least two finite observations are required.
Every forecast requires finite future `exog` with shape `(steps, features)`.
Columns have positional identity: supply them in the same order used for fitting.
The API cannot detect a caller swapping equally shaped columns. Constant or
collinear columns are allowed under proper priors, but their separate effects can
remain weakly identified, especially alongside a structural level or trend.

`get_samples_2d()` retains variance arrays with shape `(chains, draws)`,
`coefficients` with shape `(chains, draws, features)`, and `terminal_state` with
shape `(chains, draws, structural_dimension)`. Every forecast uses its corresponding
joint coefficient, terminal-state, and variance draw. Forecast sample arrays have
shape `(chains, draws, steps)`:

- `regression_samples` is the contribution `X_future @ beta`.
- `level_samples` (also `state_samples`) is the structural level.
- `slope_samples` or `seasonal_samples` exposes the applicable structural component.
- `mean_samples` is the complete conditional mean, including regression.
- `observation_samples` includes future observation noise.
- `cumulative_observation_samples` sums the same observation path over time.

Intervals are pointwise equal-tailed posterior-predictive intervals. Forecasts are
conditional on supplied future features; random future feature scenarios are not
modeled. Static uncertain coefficients do not implement time-varying coefficients.
Dense augmented FFBS costs approximately `O(T * (structural_dimension + features)^3)`
per iteration. Designs with many features therefore need measured runtime planning.

## Fourier seasonality and short histories

`fourier_design(count, period, harmonics, start=0)` produces sine/cosine calendar
columns. Use an explicit small number of harmonics and regularizing coefficient
priors. This is fixed harmonic seasonality; it differs from stochastic dummy
seasonality in `BayesianSeasonalLocalLevel`.

```python
period, harmonics = 12, 2
X = rmc.fourier_design(len(y), period, harmonics)
prior = rmc.GaussianCoefficientPrior(np.zeros(X.shape[1]),
                                     0.5 * np.eye(X.shape[1]))
fit = model.fit(y, exog=X, coefficient_prior=prior)
X_future = rmc.fourier_design(12, period, harmonics, start=len(y))
forecast = fit.forecast(12, exog=X_future)
```

`start` is the integer calendar index of the first row. Continue the training origin
into forecasting, including missing observations. Harmonics must be between one and
`floor(period / 2)`. Column order is `sin(1), cos(1), sin(2), cos(2), ...`; at the
even-period Nyquist harmonic, only its cosine is included. For example, period 12
with 6 harmonics has 11 columns. No intercept is added automatically.

The stochastic seasonal model also accepts short histories with at least two finite
observations; there is no full-cycle or period-dependent finite-count requirement.
This permits 12- and 18-month annual histories and short weekly histories. It does
not establish that the data identify seasonality. Assess sensitivity to initial-state,
coefficient, and variance priors and compare rolling-origin forecasts against a
simple baseline. Truncated Fourier models reduce state dimension for long periods.

## Fixed-parameter time-varying observation rows

`LinearGaussianStateSpace.with_observation_rows(rows)` returns a model with one
finite observation vector per training time. Filtering, smoothing, and FFBS use the
matching row, including across missing observations. Supply future rows explicitly:

```python
varying = fixed_model.with_observation_rows(Z_train)
smoothed = varying.smooth(y)
forecast = varying.forecast(y, steps=12, future_observation_rows=Z_future)
```

The future row count must equal `steps`; every row width must equal state dimension.
Joint forecast covariance uses both corresponding horizon rows. Transition and
process matrices remain constant. These fixed-parameter forecasts still condition on
the supplied variances, unlike the fitted Bayesian regression forecasts.
