# rustmc

Bayesian models in Python. Inference in Rust.

rustmc focuses on small, structured models you need to fit repeatedly: regressions,
group comparisons, calibration, and forecasts. Build a model once, fit new datasets,
and keep the posterior draws for prediction and diagnostics.

The project is **alpha**. The Python package is supported; the Rust API is still
changing. Check convergence and model fit on your own data.

```bash
pip install rustmc
```

NumPy is the only required Python dependency. Install `rustmc[viz]` for ArviZ and
Matplotlib. Python 3.9–3.13 are covered by install tests.

## Fit a regression

This example estimates an instrument's offset, gain, and measurement noise.

```python
import numpy as np
import rustmc as rmc

rng = np.random.default_rng(42)
x = np.linspace(-2, 2, 100)
y = 0.3 + 1.2 * x + rng.normal(0, 0.2, x.size)

model = rmc.ModelBuilder()
offset = model.normal_prior("offset", 0.0, 1.0)
gain = model.normal_prior("gain", 1.0, 0.5)
noise = model.half_normal_prior("noise", 0.5)
model.normal_likelihood("reading", offset + gain * "x", noise, "y")
compiled = model.compile()

fit = compiled.sample(
    {"x": x, "y": y}, chains=4, warmup=1000, draws=1000, seed=42,
    show_progress=False,
)
print(fit.summary())

future = fit.predict({"x": np.array([-1.0, 0.0, 1.0])}, seed=43)
print(np.quantile(future["reading"], [0.025, 0.975], axis=(0, 1)))
```

`predict` keeps the `(chain, draw, observation)` axes. Use `expected=True` for the
conditional mean without new observation noise. Priors above are chosen for this
example's units.

## Reuse the model

`compiled.sample()` accepts another dataset with the same columns and a different
number of rows. `compiled.sample_batch()` fits independent datasets with stable IDs:

```python
batch = compiled.sample_batch(
    [{"x": x, "y": y}, {"x": x[:50], "y": y[:50]}],
    ids=["instrument-a", "instrument-b"],
    chains=4, warmup=1000, draws=1000, threads=2, errors="collect",
    show_progress=False,
)
for instrument in batch.ids:
    if instrument not in batch.errors:
        print(instrument, batch.get(instrument).summary())
print(batch.errors)
```

Independent fits do not share information. For related groups, build one
[partial-pooling model](docs/examples/site-effects.md).

## What's included

- NUTS and HMC with autodiff, constrained parameters, and parallel chains.
- Scalar and vector regressions, group indexing, nonlinear expressions, and custom
  log-density terms. See [custom models](docs/custom-models.md).
- Prior and posterior prediction, pointwise log likelihood, R-hat, effective sample
  size, Monte Carlo error, and ArviZ export.
- Exact Gaussian AR regression, Gaussian hierarchical models, and Kalman/FFBS
  algorithms for state-space models.
- [Forecasting workflows](docs/forecasting-workflows.md) for structural, count,
  hurdle, and runoff models, with joint predictive paths and backtests.
- Versioned model and fit artifacts. Compiled model artifacts omit training data;
  fitted artifacts include it. Neither resumes sampler adaptation or RNG state.

The modeling language is deliberately small. PyMC and Stan offer broader model
support. rustmc aims to earn its place through repeated fitting and a few well-tested
specialized algorithms. Performance depends on the workload; see the
[benchmark protocol](benchmarks/README.md).

## Start here

- [Instrument calibration](examples/instrument_calibration.py): regression and new-data prediction.
- [Repeated calibration](examples/repeated_calibration.py): one model, several datasets.
- [Site effects](examples/site_effects.py): partial pooling with unequal group sizes.
- [Forecasting](examples/custom_forecast_workflow.py): fit, predict, and evaluate.
- [Examples guide](examples/README.md) and [API reference](docs/reference.md).

The [roadmap](ROADMAP.md) tracks five priorities: statistical release gates,
representative benchmarks, native model artifacts, bounded batches, and consistent
results and diagnostics. Forecasting remains an application of that shared core.

For source builds and checks, see [Contributing](CONTRIBUTING.md).
MIT licensed.
