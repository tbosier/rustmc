# RustMC forecasting arm

## Protocol

The RustMC runner uses only each series' training split for model selection and fitting.
The test split is read only after the final posterior forecast exists, for metric
calculation.

The base candidate set contains two Bayesian local-level specifications, two Bayesian
local-linear-trend specifications, and conjugate Bayesian AR models. Monthly AR orders
are 1, 2, 3, 6, and 12. Weekly AR orders are 1, 2, 3, 6, 13, 26, and 52. Each fit is
standardized using its own training fold. Variance priors therefore have consistent
meaning across series.

Base candidates are compared by rolling-origin weighted interval score (WIS):

- Monthly origins: 15, 18, and 21, with forecasts of up to 3 months.
- Weekly origins: 65, 78, and 91, with forecasts of up to 13 weeks.
- Selection fits: 2 chains, 250 retained draws, and 250 warmup draws for Gibbs models.
- Final fits: 4 chains and 1,000 retained draws; Gibbs models use 500 warmup draws.

The fitted seasonal local-level API requires two complete seasonal cycles. Exactly 24
monthly or 104 weekly training observations therefore leave no honest rolling holdout
on which that model can be tuned. The runner handles this explicitly:

- Monthly data use the fitted seasonal model only when a quadratic-detrended comparison
  of the two annual cycles has correlation at least 0.35 and seasonal RMS ratio at
  least 0.65. A training-only median paired-cycle drift is removed before fitting and
  restored over the forecast horizon. The initial seasonal pattern is shrunk according
  to the observed cycle correlation.
- Weekly data retain the rolling-selected model. When annual AR(52) is within 5% of the
  best rolling WIS and the two annual cycles have correlation at least 0.60 and seasonal
  RMS ratio at least 0.80, annual AR(52) wins the structural tie-break.

Final fit and forecast times exclude imports and model selection. Selection timing is
preserved separately in `rustmc_results.json`.

## Results

| Series | Selected RustMC model | Fit (s) | Forecast (s) | MAE | RMSE | sMAPE | 95% HDI coverage | Mean HDI width |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Monthly easy | Drift-adjusted Bayesian seasonal local level | 0.7069 | 0.0012 | 1.471 | 1.552 | 1.33% | 100.0% | 12.57 |
| Monthly medium | Drift-adjusted Bayesian seasonal local level | 0.6888 | 0.0008 | 9.355 | 9.866 | 6.45% | 16.7% | 14.66 |
| Monthly hard | Drift-adjusted Bayesian seasonal local level | 0.6963 | 0.0006 | 6.671 | 8.029 | 6.03% | 100.0% | 37.31 |
| Weekly easy | Regularized Bayesian AR(52) | 0.0021 | 0.0061 | 6.122 | 7.015 | 2.48% | 100.0% | 25.17 |
| Weekly medium | Regularized Bayesian AR(52) | 0.0022 | 0.0056 | 10.433 | 12.283 | 4.94% | 100.0% | 51.45 |
| Weekly hard | Regularized Bayesian AR(52) | 0.0022 | 0.0060 | 20.516 | 26.459 | 6.94% | 88.5% | 75.10 |

The HDIs are pointwise shortest 95% intervals computed from 4,000 coherent
future-observation posterior-predictive paths. They are predictive HDIs, not intervals
for a latent mean and not simultaneous bands for an entire future trajectory.

## Interpretation and limitations

The easy monthly result and the two simpler weekly results demonstrate useful current
capability. The medium monthly result is the important failure: only one of six held-out
observations falls inside the nominal 95% predictive HDI. Its accelerating trend and
changing seasonal amplitude require a joint trend-plus-seasonal model. RustMC currently
offers those components as separate fitted models.

The monthly drift adjustment is estimated before the RustMC fit. The reported predictive
HDI conditions on that drift estimate; it does not integrate drift-estimation uncertainty.
This is another likely contributor to the medium-series undercoverage. It should be
replaced by a joint Bayesian structural model rather than hidden behind wider ad hoc
bands.

The annual weekly fitted seasonal local-level model was also tested directly. One
period-52 fit with 4 chains, 1,000 retained draws, and 500 warmup draws took 272.15
seconds with four Rayon threads; forecasting took 0.0036 seconds. The current dense
53-state FFBS implementation scales poorly with long seasonal periods. The conjugate
AR(52) fallback is fast, but it is not a replacement for an efficient structural model
and performs weakly on the hard multi-seasonal, pulse-and-regime series.

The clearest next implementation is a single structural model with local trend,
multiple seasonal components, regression effects, and sparse or specialized seasonal
state transitions. Drift and seasonal regression uncertainty must be propagated inside
the posterior rather than in preprocessing. Robust or heavy-tailed observation models
would address the hard-series outliers, and rolling-origin scoring should become a
first-class library feature.
