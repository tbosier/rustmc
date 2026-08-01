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

## Tier 1: Correctness and trust

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
- Add artifact serialization/versioning only after in-memory rebinding is correct.

## Tier 3: State-space time series

- Integrate the existing collapsed linear-Gaussian filter into `ModelBuilder`, rather
  than sampling one latent variable per time step with NUTS.
- Add parameter-estimation paths for local level, local linear trend, and stationary
  AR(1); all three fixed-input constructors and conditional forecast intervals are now
  available.
- Extend the existing filtering/smoothing and forecast outputs with named coordinates.
- Validate log likelihood and filtered moments against a small independent reference.
- Multivariate/non-Gaussian state-space models remain later work.

## Tier 4: Modeling surface and operations

- Add named dimensions, group indexing, offsets/exposure, vector hierarchical blocks,
  and prediction on newly bound data.
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
