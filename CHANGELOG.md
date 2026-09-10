# Changelog

All notable changes to rustmc are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases use semantic
versioning while the public API is stabilized.

## [Unreleased]

## [0.12.0] - 2026-09-09

### Added

- Scalar/elementwise expression arithmetic and transforms, named deterministics,
  potentials, independent observation dimensions, grouped parameter indexing, and
  posterior prediction on new predictor rows without dummy responses.
- Versioned declarative compiled-model artifacts and validated generic fit artifacts
  that preserve posterior draws, training data, telemetry, and seeded prediction.
- Composable structural level, damped trend, harmonic seasonal, stable fixed AR, and
  static/dynamic regression blocks, with joint Gaussian FFBS/Gibbs, optional Student-t
  observation errors, component histories, prior prediction, and persistence.
- Native dynamic Poisson, negative-binomial, hurdle-lognormal, and pooled Gaussian
  models with exposure, group coefficients, latent trajectories and optional shared
  shocks using elliptical slice sampling. These infer coefficients and states with
  explicitly fixed scale/dispersion inputs.
- Rolling-origin backtests, CRPS/WIS/coverage/width scores, seasonal-naive bootstrap
  baselines, labeled forecast draws, named feature checks, scenario mixtures, portable
  forecast archives, and full-refit update sessions.
- A native `LogDensity`/gradient interface reusing HMC/NUTS and explicit unconstrained
  chain initialization. Generic compiled batches use stable cell IDs, bounded pools,
  collected failures, diagnostics and predictive results.
- A mixed Python/native package with maintained API type information.

### Fixed

- Forecast simulation now propagates the full configured process covariance, including
  fixed regression-state innovations; invalid conjugate covariance updates are rejected.
- Native observation simulation no longer clips exponential rates or probabilities.
- Core sampler configuration rejects empty chains/draws, invalid integration controls,
  nonfinite initialization and overflowing allocation counts.
- Duplicate likelihood/deterministic outputs and incompatible named dimensions are
  rejected instead of silently overwriting or conflating results.
- Demo WIS uses its standard denominator. The separate baseline single-interval metric
  is named `interval_score_95`; retained scores were corrected without rerunning timing.
- Persistence support/shape validation and allocation guards cover new model kernels.

### Changed

- `CompiledModel.sample_batch` defaults to `cell_id_v1` seeds. Use `position_v0` for
  previous positional replay; the legacy global `batch_sample` keeps its old seeds.
- Extracted expression compilation, prediction binding, observation simulation and
  artifact code into focused modules; isolated the legacy Rust compiled-model format
  and reference autodiff while preserving their public import paths.
- Structural Student-t forecasts require `df > 1`, so their conditional means exist.
  Damping/AR/df and dynamic-GLM scales/dispersion remain fixed model inputs.

## [0.11.0] - 2026-09-08

### Added

- Joint Bayesian exogenous regression for local-level, seasonal, and trend forecasts,
  with proper Gaussian coefficient priors and paired coefficient/state/variance draws.
- Time-varying observation rows in fixed Gaussian state-space models and explicit
  future-design validation, including joint and cumulative forecast covariance.
- Native independent forecasting batches with stable cell seeds, ragged histories,
  per-cell models/designs, controlled worker counts, and collected cell errors.
- Sampler-aware parameter diagnostics for forecasting fits, including coefficients
  and retained terminal states; Gibbs telemetry does not invent divergences or
  acceptance rates.
- Fourier calendar designs with explicit phase and Nyquist handling. Stochastic
  seasonal fits now accept short histories with at least two finite observations.
- A hurdle model for nonnegative amounts with static uncertain occurrence
  probability and dynamic lognormal severity under upper-truncated inverse-gamma
  variance priors, including exact zeros and all-zero histories.
- Count-only payment runoff with shared Dirichlet lag probabilities, known or
  inferred ultimate counts, prefix censoring, and an explicit unscheduled tail.
  It does not model continuous currency allocations or couple separate accrual fits.

### Fixed

- Guarded oversized forecasting/runoff allocations and sparse large-count binomial
  draws; strengthened version checks across manifests, dependencies, and the lockfile.

### Packaging

- Prepared synchronized Rust and Python 0.11.0 metadata. Release uploads require
  verification of the built wheel or source archive before artifact publication.

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

[Unreleased]: https://github.com/tbosier/rustmc/compare/v0.11.0...HEAD
[0.11.0]: https://github.com/tbosier/rustmc/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/tbosier/rustmc/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/tbosier/rustmc/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/tbosier/rustmc/releases/tag/v0.8.0
