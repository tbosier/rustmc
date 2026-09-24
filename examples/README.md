# Examples

Run these from the repository root after installing rustmc:

```bash
python examples/instrument_calibration.py
python examples/site_effects.py
python examples/custom_forecast_workflow.py
```

Every script generates its own data. Their priors describe those examples; adapt them
to your units. Short runs demonstrate the API. Inspect diagnostics before interpreting
a fit.

## Regression, prediction, and reuse

| Example | Use it for |
|---|---|
| `simple_example.py` | Prior and posterior predictive checks. |
| `instrument_calibration.py` | Regression, diagnostics, and prediction at new inputs. |
| `repeated_calibration.py` | Independent fits with one compiled model and stable IDs. |
| `arviz_example.py` | Exporting a fit to ArviZ and using its diagnostics and plots. |
| `large_linear_regression.py` | A 500-coefficient vector parameter on the faer GEMV path. |
| `batch_inference.py` | Many independent fits in one call, by shared schema and by differing structure. |

## Pooling and panels

| Example | Use it for |
|---|---|
| `site_effects.py` | Partial pooling and comparisons from joint posterior draws. |
| `hierarchical_example.py` | Scalar partial pooling with a global mean and a between-group scale. |
| `partial_pooling_template.py` | What the builder's hierarchical priors do and do not support today. |
| `hierarchical_mean.py` | A specialized Gaussian Gibbs model and joint totals. |
| `fixed_effects_panel_forecast.py` | Group indexing with `beta["key"]`, and how to keep a multi-level panel identified. |

## Forecasting

| Example | Use it for |
|---|---|
| `custom_forecast_workflow.py` | Structural forecasts and held-out scoring. |
| `state_space_forecasting.py` | Fixed-parameter local-level, local-trend, and AR(1) forecasts. |
| `custom_state_space_forecasting.py` | A custom latent state-space system built from explicit matrices. |
| `bayesian_local_level_forecasting.py` | Predictive intervals against latent-state intervals. |
| `bayesian_local_linear_trend_forecasting.py` | Level and slope intervals from one trend fit. |
| `bayesian_seasonal_forecasting.py` | A seasonal local level on monthly data. |
| `bayesian_ar_forecasting.py` | A Bayesian AR(3) posterior-predictive forecast. |
| `payment_triangle_runoff.py` | Censored payment-count development with uncertain ultimates. |
| `rebate_accrual_forecast.py` | A deliberately cautious accrual baseline, with its own limitations stated. |

Each script above runs on the released package with NumPy alone, except
`arviz_example.py`, which needs `pip install "rustmc[viz]"`.
`bayesian_local_level_forecasting.py` prints extra ArviZ diagnostics when ArviZ is
installed and says so when it is not.

## Helpers, not examples

`hierarchical_templates.py` is a module rather than a script, and running it directly
does nothing useful. It holds `build_centered_normal_partial_pooling`, the reusable
builder helper that `hierarchical_example.py` and `partial_pooling_template.py` import;
[`partial_pooling_template.py`](partial_pooling_template.py) shows what it covers.

Performance comparisons belong in `benchmarks/`: use `python -m benchmarks.run --help`
for the reference protocol and `benchmarks/README.md` for what may be claimed from a
measurement.
