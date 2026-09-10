# Composable structural forecasts

`StructuralModel` adds named independent Gaussian state blocks to one observation
mean. It supports a local level, linear or damped trend, multiple harmonic
seasonalities (including fractional periods), static and random-walk regression,
and fixed stable AR(p) residuals. The observation family is Gaussian by default;
`student_df=5` selects Student-t errors with fixed degrees of freedom. The supported
range is finite `student_df > 1`, so every conditional observation mean exists.

```python
import numpy as np
from rustmc import StructuralComponent as C, StructuralModel, VarianceParameter as V

model = StructuralModel([
    C.trend("baseline", V.inverse_gamma(3, .1), V.fixed(.001),
            initial_mean=[10., 0.], initial_covariance=[[4., 0.], [0., .1]],
            damping=.98),
    C.seasonal("weekly", period=7, harmonics=2,
               innovation=V.fixed(0), initial_variance=1.),
    C.seasonal("annual", period=365.25, harmonics=3,
               innovation=V.inverse_gamma(4, .01), initial_variance=.5),
    C.regression("promotion", initial_mean=[0.], initial_covariance=[[4.]]),
    C.regression("price", initial_mean=[-1.], initial_covariance=[[1.]],
                 innovations=[V.inverse_gamma(4, .01)]),
    C.ar("residual", coefficients=[.4], innovation=V.fixed(.05),
         initial_mean=[0.], initial_covariance=[[.2]]),
], observation_variance=V.inverse_gamma(3, .2), student_df=5)

n = 40
exog = np.column_stack([np.arange(n) % 5 == 0, np.linspace(0, 1, n)])
y = 10 + exog[:, 0] - exog[:, 1]
fit = model.fit(y, exog=exog, chains=2, draws=200, warmup=200,
                thin=1, seed=42, store_states=True)
future_exog = np.column_stack([np.zeros(7), np.ones(7)])
forecast = fit.forecast(7, exog=future_exog, seed=43)
interval = np.quantile(forecast.observation_paths, [.05, .5, .95], axis=(0, 1))
```

Use a level or a trend for the baseline, according to the intended model. Each
component needs a unique name. Regression columns follow regression-component
order and then coefficient order inside each component; future exog must contain
exactly the same columns. Static regression is the default; pass one innovation
variance per coefficient to select dynamic coefficients. No intercept or centering
is added. All states have explicit proper Gaussian initial priors describing
`x[-1]`, immediately before the first observation. Each observation first advances
the state once. NaN observations are missing, and still require their exog row.

`VarianceParameter.inverse_gamma(shape, scale)` uses density proportional to
`v**(-shape-1) * exp(-scale/v)` on positive variance. Fixed zero innovations are
allowed and remain exactly zero. Initial covariances are independent of innovation
variances and must be positive definite. The trend transition is
`level[t] = level[t-1] + damping * slope[t-1] + noise` and
`slope[t] = damping * slope[t-1] + noise`; damping is fixed in `(0, 1]`.
Seasonal pairs rotate by `2*pi*k/period`; `2*harmonics < period` avoids aliasing
and the redundant Nyquist pair. Each pair coordinate gets its own independent
innovation variance parameter, even when constructed with the same inverse-gamma
specification. AR coefficients and Student-t degrees of freedom are fixed,
validated inputs. AR initial states use the supplied covariance, which need not
be the stationary covariance. A zero-innovation component can imply singular
state transitions; the Gaussian smoother reports an error if its required
conditional solves become singular.

Inference alternates exact joint state FFBS with inverse-gamma conditional
variance updates. The complete state trajectory includes `x[-1]`, so all observed
and missing-time transitions enter innovation updates. Student-t observations use
an additional Gamma precision update. Thus their observation variance parameter
is the squared Student-t scale; the marginal observation-error variance equals
`scale_squared * df/(df-2)` when `df > 2` and is not finite for smaller df.
Inverse-gamma scale priors can still make parameter-integrated moments infinite;
conditional mean draws are not a promise that every marginal moment exists.

`fit.diagnostics()` and `fit.summary()` report rank-normalized R-hat, bulk/tail ESS,
and MCSE for every innovation/noise variance and terminal state. They do not cover
every historical state or latent Student-t precision. `get_samples_2d()` returns
named `(chain, draw)` arrays, and `get_samples()` flattens these axes.
`sampler_stats` identifies the Gibbs/FFBS kernel and fixed Student-t degrees of freedom.
Forecast `mean_samples` and `observation_samples` alias `mean_paths` and
`observation_paths`; `chains`, `draws`, and `steps` expose their leading axes/horizon.

Oversized working/retained array requests raise a validation error before allocation
at a 25-million-value bound. Predictive Gamma precision underflow/overflow aborts
the request instead of replacing the draw.
FFBS requires positive definite predictive state covariances. A deterministic AR
block with both zero innovation variance and a singular transition can be simulated
but cannot currently be fitted; the sampler returns a factorization error.
No posterior values are clipped. Numerical overflow or invalid conditionals raise
an error. Variance/state mixing can be slow for short or weakly identified series;
inspect convergence across chains and calibrate initial and innovation priors.

`fit.variance_draws` has shape `(chain, draw, dimension + 1)` with labels in
`fit.variance_names`; the final column is observation variance. Fixed innovation
columns remain present. `fit.terminal_states` preserves paired terminal states.
With `store_states=True`, `fit.states` has shape
`(chain, draw, training_count + 1, dimension)` and
`fit.historical_components` has shape `(chain, draw, training_count, component)`.
The historical decomposition excludes `x[-1]` and sums to fitted latent means.

Forecasts retain each joint state/variance draw for the whole future path:

- `state_paths`: `(chain, draw, horizon, dimension)`.
- `component_paths`: `(chain, draw, horizon, component)`, labels in `component_names`.
- `mean_paths`, `observation_paths`, `cumulative_observation_paths`:
  `(chain, draw, horizon)`.

Component contributions sum to the latent observation mean. Cumulative paths sum
future observations, retaining shared parameter uncertainty and all horizon
covariance. `model.prior_predict(steps, exog=..., draws=..., seed=...)` samples
initial states and variances from their actual priors before advancing through
the horizon. It returns the same forecast type with one chain axis. Use it to
check scales before fitting.

Models and fits support `to_json()` and class-level `from_json(text)` with
versioned, validated formats. A fit includes model priors, posterior variances,
paired terminal states, training observation rows, and optional state history.
Loading preserves floating-point values and seeded forecasts exactly. The native
fit intentionally does not store original observed values; retain those in your
application if you want to refit after new data arrive. Existing specialized
forecasting presets remain available.
