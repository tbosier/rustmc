# rustmc

Bayesian models in Python. Inference in Rust.

Use rustmc for regressions, partial pooling, calibration, and forecasts. Compile a
model once, fit new datasets, and carry posterior uncertainty into predictions.

```bash
pip install rustmc
```

The project is alpha. Its modeling language is small, and the Rust API is still changing.
Check diagnostics and model fit before interpreting results.

[Fit your first model](getting-started.md), then try
[repeated calibration](examples/repeated-calibration.md) or
[site effects](examples/site-effects.md). The [custom model guide](custom-models.md)
covers expressions, dimensions, prediction, and persistence.

[Forecasting](forecasting-workflows.md) uses the same Bayesian foundations, with
specialized state-space, count, and sparse-amount models.

The current work focuses on stronger statistical checks, representative benchmarks,
Rust artifact loading, bounded batches, and consistent results.
See the [roadmap](https://github.com/tbosier/rustmc/blob/main/ROADMAP.md).
