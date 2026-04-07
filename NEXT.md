Production Bayesian engine plan
Ordered by impact, with workstreams that can run in parallel.

## Tier 1: Telemetry and trust

- Preserve exact sampler events per transition: accept prob, divergence flag, energy error, tree depth, leapfrog count, and termination reason.
- Export structured diagnostics instead of only formatted summaries: per-chain stats, per-transition stats, and per-parameter summaries.
- Add log-likelihood export at the observation level so LOO/WAIC and posterior predictive workflows can be implemented cleanly.
- Replace stringly sampler errors with a structured error type that the Python binding can surface unchanged.

## Tier 2: Model artifact and runtime

- Introduce a compiled model artifact that stores the graph, parameter metadata, transforms, observed-data schema, and likelihood metadata.
- Make the runtime able to load that artifact without rebuilding the Python-side model DSL.
- Add a Rust-native public API for loading a compiled model and running inference directly from Rust.
- Make serialization/versioning explicit so artifacts fail fast on schema mismatch.

## Tier 3: Model surface

### First implementation slice
- Generalize the likelihood API so it can express a linear predictor plus a family-specific link function.
- Land Bernoulli-logit regression first. It is the smallest production GLM wedge and covers binary classification, conversion, and event models.
- Land Poisson-log regression second, with exposure/offset support for rate models.
- Add Student-t regression third, as the robust continuous baseline for outlier-heavy data.
- Add Negative Binomial after Poisson, since it is the natural overdispersed count extension.
- Add non-centered hierarchical templates and vector-valued random effects after the core families are in place.
- Add reusable helpers for common production patterns: trend/seasonality, group intercepts, varying slopes, and noise hierarchies.
- Keep the DSL small and opinionated; do not add arbitrary custom likelihoods until the built-in families are stable and covered by tests.

## Tier 4: Validation and operations

- Add statistical recovery tests and calibration tests for each supported model family.
- Add benchmark suites for single-model ESS/s, batch throughput, and memory use.
- Add packaging/release checks for Python wheels, Rust crates, and compiled-model compatibility.
- Add failure-mode tests for divergences, low ESS, invalid data schemas, and incompatible artifacts.

## Current state

- The hot path is already in Rust, with zero-allocation autodiff and parallel chain execution.
- Sampler outputs currently expose aggregate samples, acceptance rates, step sizes, and divergence counts.
- Diagnostics are summary-level only today; there is no per-transition telemetry or log-likelihood export yet.
- The Python DSL is narrow but usable for repeated linear/Normal models.

## Code references

- Sampler state and current result surface: `rust_core/src/sampler.rs`
- NUTS adaptation and termination behavior: `rust_core/src/nuts.rs`
- Diagnostics summary layer: `rust_core/src/diagnostics.rs`
- Python API surface and predictive helpers: `python_bindings/src/lib.rs`
- Graph and transform model representation: `rust_core/src/graph.rs`
