# Examples

Run these from the repository root after installing rustmc:

```bash
python examples/instrument_calibration.py
python examples/repeated_calibration.py
python examples/site_effects.py
python examples/custom_forecast_workflow.py
```

| Example | Use it for |
|---|---|
| `instrument_calibration.py` | Regression, diagnostics, and prediction at new inputs. |
| `repeated_calibration.py` | Independent fits with one compiled model and stable IDs. |
| `site_effects.py` | Partial pooling and comparisons from joint posterior draws. |
| `custom_forecast_workflow.py` | Structural forecasts and held-out scoring. |
| `simple_example.py` | Prior and posterior predictive checks. |
| `hierarchical_mean.py` | A specialized Gaussian Gibbs model and joint totals. |

These use generated data. Their priors describe those examples; adapt them to your units.
Short runs demonstrate the API. Inspect diagnostics before interpreting a fit.

Performance comparisons belong in `benchmarks/`. The older `benchmark_*.py`,
`compare_with_pymc.py`, and `batch_many_series.py` scripts are exploratory studies,
not retained evidence for speed claims. Use `python -m benchmarks.run --help` for
the reference protocol.
