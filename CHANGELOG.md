# Changelog

All notable changes to rustmc are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases use semantic
versioning while the public API is stabilized.

## [Unreleased]

## [0.10.0] - 2026-08-04

### Added

- Added `BayesianHierarchicalMean`, a joint population → group → program Gaussian
  partial-pooling model for ragged program series.
- Added a dedicated conjugate Gibbs kernel that draws the hierarchy's exact full
  conditionals and avoids requiring HMC/NUTS to traverse funnel geometry.
- Added explicit group, program, and observation variance priors plus program/group
  names and ragged time/observation-count metadata.
- Added aligned hierarchical posterior-predictive paths shaped
  `(chain, draw, program, step)` and built-in draw-wise group/company rollups.
- Added hierarchical R-hat, bulk/tail ESS, MCSE, and HDI reporting plus a dedicated
  `InferenceError` Python exception.
- Added Rust and Python coverage for singleton adaptive shrinkage, missing values,
  reproducibility, axis alignment, ragged validation, and coherent aggregation.

### Performance

- Precomputed group membership and per-program sufficient statistics so each Gibbs
  sweep is linear in programs/groups rather than rescanning all programs per group.
- Stored predictive observations contiguously, reconstructed static state paths lazily,
  and added checked posterior/forecast allocation guards.

### Documentation

- Documented the static hierarchical-intercept estimand, ragged-series weighting,
  prior sensitivity, and the boundary with dynamic local-level forecasting.
- Added a complete hierarchical mean and rollup example.

## [0.9.0] - 2026-08-02

### Added

- Fitted Bayesian local-level, seasonal local-level, local-linear-trend, and directly
  observed AR(p) forecast models with coherent posterior paths.
- Fixed-matrix linear-Gaussian state-space filtering, smoothing, missing-observation
  handling, and forecasting.
- In-memory compile-once/bind-many model reuse.
- A reproducible cross-engine benchmark harness for rustmc, PyMC, PyMC with nutpie, and
  NumPyro with analytic posterior checks and isolated backend environments.
- A rebate-accrual example that distinguishes latent credible intervals from
  future-observation posterior-predictive intervals and aggregates paths correctly.

### Changed

- Reframed the project as a practical, general Bayesian toolkit; forecasting is one
  application rather than the library's identity.
- Added configurable `target_accept` to generic NUTS/HMC sampling entry points.
- Extended fixed linear-Gaussian state-space forecasts with joint future-observation
  covariance and exact cumulative Gaussian summaries.
- Added a fixed-parameter sum-to-zero seasonal local-level state-space constructor.
- Added fitted Bayesian seasonal local-level inference with Gibbs/FFBS, missing-value
  support, and coherent seasonal and cumulative posterior-predictive paths.
- Exposed generic sampler transition diagnostics through the Python API.
- Moved internal planning, review, and validation notes out of the source repository.
- Removed generated plots and scratch data from version control.

### Fixed

- Corrected the Lanczos `ln_gamma` implementation used by Negative-Binomial log density.
- Corrected NUTS and HMC initial step-size threshold calculations.
- Replaced misleading diagnostic calculations with a genuine 94% HDI,
  rank-normalized folded split R-hat, and corrected bulk/tail ESS estimation.
- Corrected constrained-parameter use in predictors and tightened cross-model parameter
  reference validation.
- Synchronized Rust, Python, wheel, and runtime package versions.

### Packaging

- Prepared Python 3.9+ ABI3 wheel metadata and clean-wheel verification.
- Marked the internal Python extension crate as non-publishable on crates.io, where the
  `rustmc` name belongs to an unrelated package.

## [0.8.0] - 2026-04-25

- Last public PyPI release before the fitted forecasting and 0.9 correctness work.

[Unreleased]: https://github.com/tbosier/rustmc/compare/v0.10.0...HEAD
[0.10.0]: https://github.com/tbosier/rustmc/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/tbosier/rustmc/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/tbosier/rustmc/releases/tag/v0.8.0
