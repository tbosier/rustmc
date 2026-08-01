# Validation Plan

This document defines the validation stack rustmc needs before it can be treated as a production Bayesian engine.

## What Exists Today

- Core autodiff unit tests in `rust_core/src/autodiff.rs`
- Diagnostics unit tests in `rust_core/src/diagnostics.rs`
- A sampler regression test in `rust_core/src/sampler.rs`
- End-to-end examples for linear regression, hierarchical models, ArviZ export, batch inference, and large linear models in `examples/`
- User-facing workflow docs and benchmark protocol in `README.md` and `docs/`
- A seeded Rust recovery suite in `rust_core/tests/recovery_suite.rs`
- Python import/API/end-to-end smoke tests in `tests/test_smoke.py`
- Clean-wheel metadata/install checks in `tests/test_packaging.py`
- Reproducible benchmark drivers in `examples/run_benchmarks.py` and a provenance/raw
  output checklist in `benchmarks/RESULTS_TEMPLATE.md`

## What Is Missing

- Statistical recovery tests for every supported model family
- Calibration tests for prior predictive and posterior predictive behavior
- Automated benchmark regression thresholds (the harness records speed, ESS/s,
  divergences, and memory, but CI does not enforce performance budgets)
- Broader wheel-level integration coverage for predictive and optional ArviZ paths
- Failure-mode tests for invalid data, incompatible shapes, thread configuration, and divergence-heavy models

## Priority Workstreams

### 1. Statistical Recovery

Goal: prove the sampler recovers known parameters on synthetic data.

Start with:

- Simple linear regression
- Hierarchical partial pooling
- High-dimensional regression using `MatVecMul`
- Heavy-tailed regression once `StudentT` likelihoods are expanded

Acceptance criteria:

- Posterior means land near the known truth on seeded synthetic data
- Coverage on posterior predictive intervals is reasonable
- Divergences remain bounded on the reference problems
- R-hat and ESS stay within explicit thresholds

### 2. Calibration

Goal: prove the posterior and predictive outputs are well-calibrated.

Add tests for:

- Prior predictive shape checks
- Posterior predictive coverage checks
- Calibration summaries across repeated seeded runs
- Out-of-sample holdout checks for forecasting-style examples

Acceptance criteria:

- Predictive intervals contain observed values at the expected rate
- Calibration does not regress across releases
- Failure cases are easy to diagnose from the returned summaries

### 3. Benchmark Regression — HARNESS IMPLEMENTED

Goal: make the performance claims reproducible and trackable.

Track:

- Single-model wall time
- ESS/s
- Batch throughput
- Memory growth across draws and chains
- Divergence counts on benchmark datasets

Use the existing benchmark examples in:

- `examples/benchmark_vs_pymc.py`
- `examples/benchmark_multivariate.py`
- `examples/batch_10k_skus.py`
- `examples/large_linear_regression.py`

Acceptance criteria:

- Benchmark runs are scripted and repeatable
- Regression thresholds are defined, not just narrative claims
- Results can be compared against previous releases

### 4. Python Integration — SMOKE COVERAGE IMPLEMENTED

Goal: validate the packaged Python API as users actually consume it.

Add tests for:

- `import rustmc`
- `ModelBuilder` construction and data binding
- `sample()`
- `batch_sample()`
- `sample_prior_predictive()`
- `FitResult.posterior_predictive()`
- `FitResult.to_arviz()`

Acceptance criteria:

- Tests run against the built wheel or editable install
- API behavior matches the docs
- Optional dependencies fail cleanly with actionable errors

### 5. Packaging and Release — CLEAN-WHEEL CI IMPLEMENTED

Goal: prevent broken releases from shipping.

Add checks for:

- Version sync across `Cargo.toml`, `pyproject.toml`, and release tags
- Wheel build success across supported Python versions
- Import smoke tests after wheel build
- No stale docs or benchmark claims in the published package

Acceptance criteria:

- A release candidate can be built and imported in a clean environment
- Version mismatches fail the release pipeline
- Packaging artifacts are reproducible

### 6. Failure Modes

Goal: make bad inputs and unstable sampling behavior obvious.

Add tests for:

- Missing or mis-shaped data keys
- Unsupported model combinations
- Divergence-heavy models
- Threading configuration edge cases
- Invalid artifact/schema mismatches once compiled models exist

Acceptance criteria:

- Failures are explicit and descriptive
- No silent fallback masks a bad configuration
- Error paths are covered in CI

## Recommended File Additions

- `tests/` contains Python integration and packaging tests
- `benchmarks/` and `examples/run_benchmarks.py` contain the repeatable benchmark harness
- `validation/` or `tests/fixtures/` for seeded datasets and golden outputs

## Current Code References

- `rust_core/src/autodiff.rs`
- `rust_core/src/diagnostics.rs`
- `rust_core/src/sampler.rs`
- `rust_core/src/nuts.rs`
- `python_bindings/src/lib.rs`
- `examples/simple_example.py`
- `examples/hierarchical_example.py`
- `examples/arviz_example.py`
- `examples/benchmark_vs_pymc.py`
- `examples/benchmark_multivariate.py`
- `examples/batch_10k_skus.py`
