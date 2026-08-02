# Changelog

All notable changes to rustmc are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases use semantic
versioning while the public API is stabilized.

## [Unreleased]

### Changed

- Reframed the project around honest, decision-grade Bayesian inference and forecasting
  rather than unaudited performance comparisons.
- Moved internal planning, review, and validation notes out of the source repository.
- Removed generated plots and scratch data from version control.

## [0.9.0] - Unreleased

### Added

- Fitted Bayesian local-level, local-linear-trend, and directly observed AR(p) forecast
  models with coherent posterior paths.
- Fixed-matrix linear-Gaussian state-space filtering, smoothing, missing-observation
  handling, and forecasting.
- In-memory compile-once/bind-many model reuse.
- A rebate-accrual example that distinguishes latent credible intervals from
  future-observation posterior-predictive intervals and aggregates paths correctly.

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

[Unreleased]: https://github.com/tbosier/rustmc/compare/v0.8.0...HEAD
[0.9.0]: https://github.com/tbosier/rustmc/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/tbosier/rustmc/releases/tag/v0.8.0
