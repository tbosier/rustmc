# Production Bayesian engine plan

This file describes unfinished work. Implemented behavior belongs in the README and
tests; roadmap items must not be advertised as current features.

## Current state

- Sampling and autodiff run in Rust; graphs are shared read-only across chains.
- Legacy `sample()` and `batch_sample()` still create data-owning graphs, while
  `ModelBuilder.compile()` exposes an in-memory compile-once/bind-many path backed by an
  immutable structure and validated `DataBinding` payloads.
- The re-bindable compiled model is not serializable yet; the existing JSON artifact is
  a separate legacy, data-owning format.
- A standalone time-homogeneous linear Gaussian state-space API now provides Kalman
  filtering, smoothing, forecasting, and missing-observation handling with fixed system
  matrices. It is not yet integrated into `ModelBuilder` parameter inference.
- A specialized Bayesian local-level API now estimates process and observation variances
  with explicit inverse-gamma priors and multi-chain FFBS/Gibbs sampling, returning
  parameter-integrated latent and observation forecast paths.
- A Bayesian local-linear-trend API now fits stochastic level, slope, and observation
  variance with two-state FFBS/Gibbs and returns coherent level/slope/observation paths.
- A directly observed Bayesian AR(p) API now supports any positive caller-selected lag
  order with exact Normal-Inverse-Gamma posterior draws and recursive predictive paths.

## Tier 1: Correctness and trust

- Replace the current raw split R-hat and approximate ESS with rank-normalized,
  folded split R-hat and modern bulk/tail ESS; compute genuine HDIs rather than
  labeling central quantiles as HDIs.
- Stabilize scalar/vector logit transforms in extreme tails and add adversarial
  transform tests.
- Add configurable initialization and `target_accept`, BFMI, tree-depth saturation,
  and clearer chain-level numerical-failure reporting.
- Keep exact post-warmup sampler events and export divergence flags, acceptance
  statistics, energy error, tree depth, leapfrog count, and termination reason.
- Maintain seeded analytic/recovery tests for every transform and likelihood.
- Keep wheel-level Python smoke tests and honest benchmark protocols in CI/docs.

## Tier 2: Compile once, bind many

- Extend the foundational `DataSchema`/`DataBinding` split with named dimensions,
  coordinates, and richer shape constraints.
- Keep the immutable graph-template boundary and remove remaining data-owning legacy
  paths only when compatibility shims can preserve current results.
- Benchmark compile, bind, and sample phases separately before publishing reuse claims.
- Remove the nested one-thread Rayon pool created per compiled batch dataset and define
  one explicit parallelism policy across datasets and chains.
- Add collect-errors batch semantics and bring `BatchResult` diagnostics/predictive
  methods to parity with `FitResult`.
- Add artifact serialization/versioning only after in-memory rebinding is correct.

## Tier 3: State-space time series

- Integrate the existing collapsed linear-Gaussian filter into `ModelBuilder`, rather
  than sampling one latent variable per time step with NUTS.
- Build a moments-free Kalman log-likelihood kernel with reusable workspace and
  parameter sensitivities for gradient-based generic state-space inference; the
  specialized local-level Gibbs path already returns parameter-integrated forecasts.
- Add a fitted noisy-latent stationary AR family over the generic state-space kernel;
  directly observed AR(p), local level, and local linear trend now have specialized
  fitted Bayesian paths, while all fixed-input constructors retain conditional intervals.
- Unify fitted forecasting results, named lag/time coordinates, pointwise likelihood,
  posterior-predictive ArviZ groups, and forecasting-specific diagnostics/calibration.
- Extend the existing filtering/smoothing and forecast outputs with named coordinates.
- Validate log likelihood and filtered moments against a small independent reference.
- Multivariate/non-Gaussian state-space models remain later work.
- Support singular process covariance for deterministic components, then add
  time-varying matrices, covariates/seasonality, and multivariate observations.

## Tier 4: Modeling surface and operations

- Add named dimensions, group indexing, offsets/exposure, vector hierarchical blocks,
  and prediction on newly bound data.
- Add first-class out-of-sample prediction/forecast binding, preserve chain/draw
  provenance in posterior predictive output, and ship `py.typed`/`.pyi` metadata.
- Add a slot-only Rust artifact/runtime API without silently migrating the legacy
  data-owning artifact.
- Keep Python context-manager syntax optional; do not conflate it with reusable
  compiled artifacts or data binding.
- Track wall time, ESS/s, memory, and divergences; never publish extrapolated results.

## Code references

- Sampler state and result surface: `rust_core/src/sampler.rs`
- NUTS adaptation and termination behavior: `rust_core/src/nuts.rs`
- Diagnostics summary layer: `rust_core/src/diagnostics.rs`
- Python API surface and predictive helpers: `python_bindings/src/lib.rs`
- Graph and transform model representation: `rust_core/src/graph.rs`
