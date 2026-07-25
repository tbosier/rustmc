# Validation Plan

This document defines the validation stack rustmc needs before it can be treated as a production Bayesian engine.

> **Status:** workstreams 1, 2 and 6 are now implemented. See
> [`VALIDATION_RESULTS.md`](VALIDATION_RESULTS.md) for the measured results,
> the tolerance methodology, the defects the suite uncovered, and what is still
> uncovered. Workstreams 3 (benchmark regression) and 5 (packaging) remain open.

## What Exists Today

- Core autodiff unit tests in `rust_core/src/autodiff.rs`
- Diagnostics unit tests in `rust_core/src/diagnostics.rs`
- A sampler regression test in `rust_core/src/sampler.rs`
- End-to-end examples for linear regression, hierarchical models, ArviZ export, batch inference, and large linear models in `examples/`
- User-facing workflow docs and benchmark claims in `README.md` and `docs/`

## What Is Missing

- ~~Statistical recovery tests for each supported model family~~ — done
  (`rust_core/tests/{analytic_posterior,prior_recovery,likelihood_recovery,sbc}.rs`)
- ~~Calibration tests for prior predictive and posterior predictive behavior~~ — done
  (`rust_core/tests/sbc.rs`, `tests/test_statistical_predictive.py`)
- Benchmark suites that track speed, ESS/s, divergences, and memory use over time
- ~~Python integration tests that exercise the actual wheel, not just Rust internals~~ —
  partially done (`tests/test_statistical_*.py`); packaging-level tests still missing
- Packaging and release validation for wheels, version sync, and import compatibility
- ~~Failure-mode tests for invalid data, incompatible shapes, thread configuration,
  and divergence-heavy models~~ — done (`rust_core/tests/numerical_stability.rs`,
  `tests/test_statistical_engine_bugs.py`)
- Statistical validation of `batch_sample` and of `compiled_model` artifact round-trips

## Priority Workstreams

### 1. Statistical Recovery — IMPLEMENTED

Goal: prove the sampler recovers known parameters on synthetic data.

Implemented as analytic-posterior comparison (stronger than recovery: the
posterior mean *and* sd are asserted against exact values), plus recovery for
the families with no conjugate reference, plus simulation-based calibration.
Results and tolerance derivations: `VALIDATION_RESULTS.md`.

Note on acceptance criteria: "divergences remain bounded" cannot be evaluated
against the number the engine reports, because it includes warmup divergences
(DEFECT 3 in `VALIDATION_RESULTS.md`). The suite computes post-warmup
divergences itself from `SampleResult::transitions`.

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

### 2. Calibration — IMPLEMENTED

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

### 3. Benchmark Regression

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

### 4. Python Integration

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

### 5. Packaging and Release

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

### 6. Failure Modes — IMPLEMENTED

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

- ~~`tests/` for integration and regression tests once Python-side test execution is wired up~~ — added
- `benchmarks/` or `scripts/` for repeatable benchmark entrypoints
- `validation/` or `tests/fixtures/` for seeded datasets and golden outputs
  (not needed so far: every dataset in the suite is generated from a seeded
  deterministic RNG defined in `rust_core/tests/common/mod.rs`, so there are no
  fixture files to drift)

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
