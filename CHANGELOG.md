# Changelog

All notable changes to rustmc are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases use semantic
versioning while the public API is stabilized.

## [Unreleased]

## [0.13.0] - 2026-09-18

This release closes a repository-wide correctness review. The headline item is a
regression in 0.12.0 that silently transposed Fortran-ordered design matrices; if
you are on 0.12.0 and pass a 2-D `X` that is not C-contiguous, upgrade.

### Added


- `rustmc_core::model::GraphModel::sample_prior` and `GraphModel::prior_predictive`,
  so a loaded model artifact can be simulated from Rust. Model-level prior generation
  moved out of the Python binding crate into `rustmc_core::prior_sampling`; draws are
  bit-identical for a given seed.
- A bare data-key string is accepted anywhere an expression operand is accepted, so
  `beta["group"] * "x"` (random slopes) and `builder.normal_likelihood("obs", "x", ...)`
  work like `beta * "x"` already did. The fused linear-predictor fast path is preserved.
- `rustmc.__all__`, so `from rustmc import *` no longer pulls in the `evaluation` and
  `forecasting` submodules.
- `scripts/run_examples.py`, run in CI: every example documented in
  `examples/README.md` must run inside a time budget, and a script in neither a README
  table nor an excluded section fails the build.

### Changed


- **Breaking (alpha Rust API):** `nuts::run_chain`, `nuts::run_chain_bound`,
  `hmc::run_chain` and `hmc::run_chain_bound` return `Result<ChainResult, String>` and
  reject discrete latent parameters; they previously bypassed every guard.
- **Breaking (alpha Rust API):** new `graph::Op::BoundedSigmoid` variant, which breaks
  an external exhaustive match on `Op`.
- A `potential` or `deterministic` naming a data key is validated when it is declared,
  on the same "only when data is bound" rule the likelihood families use. A builder
  holding part of its data can no longer declare one naming a key that arrives later.
- `examples/fixed_effects_panel_forecast.py` and `examples/large_linear_regression.py`
  were rewritten; they ran for about 57 and 42 minutes and now take 27s and 5s. The
  panel example used 168 dense one-hot indicator columns instead of the library's own
  group indexing, measured at 63x the cost per gradient, and stacked four nested
  intercept blocks that were not identified.

### Fixed


- **Fortran-ordered and otherwise non-C-contiguous 2-D inputs are no longer read as
  row-major.** In 0.12.0 a design matrix in column-major order was reinterpreted
  against its own shape, silently producing a different model and a wrong posterior
  with no error.
- **Structural fits and forecasts no longer share an RNG stream.** `structural.rs` was
  the only fit-then-forecast module without a domain separator, so passing one seed to
  both `fit()` and `forecast()` replayed the fitting draws: each forecast's first
  innovations were the standard normals that built its own terminal state. Measured on
  two fixed-variance levels, step-1 predictive variance was 6.14 against an analytic
  2.75-3.8 depending on the configuration, and the terminal state correlated with its
  first innovation at +0.93. Seeded structural forecast output therefore differs from
  0.12.0 for every seed, not only colliding ones.
- **A bounded prior's density is evaluated at the point it reports as the draw.** The
  logistic transform had three implementations, two of which overflowed for arguments
  below about -709. `Uniform` priors now compile to one fused `BoundedSigmoid` node,
  which also removes the span factor that made the gradient overflow: `Uniform(0, 1e308)`
  at raw -710 gave `-inf` and now gives -19.037138374168897.
- **`a / b` gradients stay representable at extreme denominator scales.** `-a/(b*b)`
  collapsed to zero once `b*b` overflowed and to infinity once it underflowed, so a
  representable derivative was silently replaced.
- **The statistical release gate screens every parameter.** It aggregated with builtin
  `max`/`min`, which drop a NaN that is not first, so a fit whose second or later
  parameter had a NaN R-hat or ESS passed. Diagnostics are now checked per parameter
  against the full range their estimators can produce, a promised reference case that
  did not run fails the run, and the report is written even when a case cannot be built.
- **Posterior-predictive draws keep their chain and draw identity in `to_arviz`.** They
  were flattened into a single fake chain, so a four-chain fit exported posterior
  parameters as `(4, draws, ...)` and predictive draws as `(1, 4*draws, ...)`, leaving
  no way to pair them. `ppc_samples` is now chain-stratified and exports the retained
  draw coordinates.
- Non-finite deterministics are rejected instead of being returned inside an otherwise
  successful `sample_prior_predictive` or `FitResult.deterministics()` result.
- Discrete latent parameters are rejected at the sampler boundary rather than only in
  the Python and artifact layers, including when they reach their density through a
  transform, and including through the raw `nuts`/`hmc` kernels. A discrete prior can
  still be loaded and simulated for prior prediction, which it could not before.
- Artifacts with unknown fields are rejected instead of being silently truncated.
- An unknown data key in a `potential` or `deterministic` is named, with the available
  keys listed, instead of surfacing as a confusing length mismatch.
- The diagnostics tables size themselves to their contents. Parameter names longer than
  12 characters, which the library's own models emit, shifted every later column; both
  horizontal rules were also the wrong length.
- Every native class reports `__module__ == "rustmc"` instead of `"builtins"`.
- Preserve posterior means, standard deviations and Monte Carlo standard errors
  across parameter units without overflow or arbitrary variance cutoffs.
- Allow smoothing and FFBS for valid deterministic state transitions with singular
  predicted covariance, including zero-innovation AR components.
- Keep zero powers and zero-valued predictors with learned positive exponents from
  introducing invalid gradients into otherwise finite custom-model targets.
- Preserve infinite losses in backtest summaries and stabilize CRPS, WIS and point
  error arithmetic for narrow forecasts with large levels or extreme finite scales.
- Preserve logical NumPy matrix order across contiguous, transposed and strided inputs.
- Evaluate Beta and Uniform priors in unconstrained coordinates without truncating
  rounded sigmoid tails; retain exact sampler positions for prediction and fit artifacts.
  Fit artifact version 2 retains these positions and reads existing version 1 artifacts.
- Keep output-only deterministic expressions from contaminating target gradients, and
  preserve positive-scale support when noncentering hierarchical Normal priors.
- Preserve correlated state uncertainty across measurement scales, reject asymmetric
  covariances consistently in the shared state-space implementation, and avoid
  cancellation of small posterior variances in smoothing.
- Count terminating NUTS expansions in tree depth and recenter HMC step-size adaptation
  after changing the mass matrix.
- Preserve native conditional-mean forecast draws, coordinates and metadata in workflow
  adapters; reject empty potential names before creating an unreadable model artifact.
- Preserve Jacobian/potential terms and the new support constraints in legacy graph
  exports, and reject overflowing Uniform ranges or nonfinite sampled outputs.
- Correct Poisson simulation at tiny and large rates across generic observations,
  dynamic count models and payment runoff.
- Preserve Poisson and negative-binomial likelihood curvature at large counts and
  dispersion, sharing stable densities across inference and pointwise diagnostics.
- Preserve Gamma, Exponential and HalfNormal prior tails in unconstrained inference
  and analytic prior draws; avoid scale overflow in Normal and Student-t densities.
- Match pointwise observation likelihoods to the fitted model without arbitrary
  scale or response floors, including very small positive LogNormal observations.
- Apply unit-independent covariance symmetry checks to specialized Gaussian models.

### Removed


- **Breaking (alpha Rust API):** `rustmc_core::compiled_model` and its re-exports
  (`ArtifactError`, `CompiledModelArtifact`, `CompiledModelRuntime`, `ModelMetadata`,
  `ModelStep`, `NodeRef`, `ParameterBlock`, `SerializableObsFamily`,
  `SerializableParamTransform`). 1,866 lines with no consumer: the Python
  `CompiledModel` API never emitted or accepted this format.
- **Breaking (alpha Rust API):** `rustmc_core::autodiff::{reference, eval_logp, forward,
  grad_logp, Value}`. The allocating reference evaluator is a differential-testing
  oracle that panics on unexpected node shapes; it is now `#[cfg(test)]` and still
  checks the optimised evaluator.
- **Breaking (alpha Rust API):** the `Op::Sub`, `Op::Div`, `Op::Neg`, `Op::Log` and
  `Op::Square` IR variants and their `Graph` builders, which only the deleted legacy
  module constructed. `ElementwiseOp` carries all of them.

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
- Regression expression compilation preserves parameter identity for names resembling
  internal constant markers; intercepts use explicit constant/parameter variants.
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
