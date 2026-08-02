# rustmc roadmap and capability ideas

This document describes direction, not release commitments. Items are ordered by
dependency and user value. Correctness evidence and a coherent public API take priority
over feature count.

## Product direction

rustmc should grow as a **practical, extensible, general Bayesian toolkit**. It should
complement PyMC and Stan rather than duplicate their complete modeling languages. Its
distinctive combination is:

1. a compact graph, autodiff, and sampling runtime for general supported models;
2. compile-once/bind-many execution for repeated model structures;
3. specialized exact, conjugate, or state-space inference when structure permits;
4. deterministic posterior and predictive outputs that can be audited and deployed; and
5. a stable native core usable without keeping Python in the execution path; and
6. implementations that remain understandable and adaptable for focused work in science,
   engineering, biomedical research, finance, and other domains.

Forecasting is an important proving ground for these ideas, but it is one application.
Regression, GLMs, hierarchical models, experiments, reliability models, and other
repeated Bayesian analyses should use the same foundations.

## 1. Trustworthy 0.9 release

This is the release gate for everything below.

- Keep analytic value and finite-difference gradient tests for every log density,
  transform, and likelihood.
- Add learned-auxiliary-parameter tests wherever a likelihood has parameters beyond its
  linear predictor.
- Pin rank-normalized folded R-hat, bulk/tail ESS, MCSE, and HDI behavior against
  independent references and adversarial chains.
- Add simulation-based calibration and posterior recovery for every supported family at
  a scale appropriate for CI, with higher-power suites runnable on demand.
- Complete the existing energy, tree-depth, and leapfrog telemetry with termination
  reasons and BFMI inputs.
- Make initialization configurable, extend `target_accept` tests across every sampling
  entry point, and test failure reporting at the chain level.
- Publish synchronized Python and Rust versions from one clean tag. Test the actual
  stable-ABI wheels on Python 3.9 through 3.13 outside the checkout.
- Retain raw validation and benchmark artifacts with the exact revision, environment,
  command, seed, workload, and statistical-quality metrics.

**Exit criterion:** a user can trace every supported density, diagnostic, wheel, and
public numerical claim to a reproducible test or retained artifact.

## 2. Complete the general modeling foundation

The next priority is making common Bayesian models natural without trying to implement
an unrestricted tensor language immediately.

- Add named dimensions, coordinates, and validated group indexing.
- Add vector-valued hierarchical priors and non-centered group effects.
- Add offsets, exposure terms, weights where statistically meaningful, and prediction on
  newly bound covariates.
- Expand robust and constrained modeling support: Student-t likelihoods, censored data,
  ordered/simplex transforms, and multivariate Gaussian building blocks.
- Make missing-data behavior explicit per likelihood rather than relying on incidental
  numeric handling.
- Define a stable extension boundary for new distributions and graph operations so each
  feature does not require unrelated Python and Rust surgery.
- Split the Python binding monolith into focused model, inference, diagnostics,
  state-space, and conversion modules, leaving `lib.rs` responsible primarily for module
  registration.
- Unify result shapes, parameter naming, coordinates, and ArviZ groups across scalar,
  vector, hierarchical, and specialized fits.
- Ship `py.typed` and maintained `.pyi` files for the supported Python API.

**Exit criterion:** regression, common GLMs, and partial-pooling models have consistent
construction, prediction, diagnostics, and typed results.

## 3. Turn compile/bind into an operational advantage

The in-memory structural split already exists. The next work should make it useful for
large repeated workloads without making unsupported speed claims.

- Give one compiled structure a first-class `fit`, `predict`, and `sample_batch` lifecycle.
- Preserve dataset identities and deterministic seeds when a batch is reordered or
  resumed.
- Add collect-errors semantics so one invalid or unstable dataset does not discard an
  otherwise successful batch.
- Define one explicit parallelism policy across datasets, chains, and linear algebra.
- Bound retained evaluator memory after unusually large bindings and report peak memory
  in regression tests.
- Add streaming or chunked dataset submission instead of requiring every payload to be
  resident before inference starts.
- Design a portable, versioned, slot-only artifact that never embeds an accidental
  training dataset.
- Provide a stable Rust runtime for binding an artifact and producing predictions or
  posterior draws without Python.

**Exit criterion:** the same audited artifact can be fitted or evaluated across many
validated datasets with stable IDs, bounded memory, isolated failures, and reproducible
results.

## 4. Build a structure-aware inference planner

This is the strongest long-term technical differentiator. A model should not pay for
generic NUTS when its graph admits a safer or more direct method.

- Detect conjugate Gaussian and Normal-Inverse-Gamma subgraphs and dispatch to exact
  posterior kernels.
- Add collapsed linear-Gaussian likelihood evaluation using the existing Kalman core.
- Keep FFBS/Gibbs implementations as reference oracles while adding marginalized
  state-space inference for unknown system parameters.
- Introduce an explicit kernel registry: exact posterior, conjugate Gibbs, Kalman/FFBS,
  Laplace, NUTS, and fixed-trajectory HMC should share a result contract.
- Explain the selected kernel and its assumptions in machine-readable fit metadata.
- Add MAP and Laplace approximation for models where a validated Hessian makes them
  useful; report approximation diagnostics rather than presenting them as equivalent to
  MCMC.
- Investigate automatic reparameterization using graph structure and observed sampler
  geometry.

**Exit criterion:** inference selection is deterministic, inspectable, tested against an
independent reference, and never silently changes the target distribution.

## 5. Develop high-value applications on the shared core

Application layers should validate the architecture without redefining the whole project.

### Time series and forecasting

- Use one dated forecast result for latent credible intervals, observation predictive
  intervals, coherent paths, and draw-wise cumulative summaries.
- Add rolling-origin backtesting with seasonal-naive and classical baselines, interval
  coverage, bias, sharpness, CRPS/WIS, and prior-sensitivity reporting.
- Extend the fitted seasonal component with calendar effects and multiple seasonalities;
  add known future regressors, offsets/exposure, time-varying matrices, and robust or
  positive observation families.
- Add hierarchical pooling and probabilistic reconciliation across related series and
  aggregation levels.
- Add stationarity-aware AR priors or explicit diagnostics rather than clipping unstable
  draws.

Demand, staffing, reliability, sensor monitoring, and rebate accruals are example use
cases. None should be hard-coded into the general inference or result APIs.

### Other promising application layers

- Repeated A/B-test and conversion models with partial pooling.
- Reliability and event-rate models with exposure and censoring.
- Small-area or site-level GLMs fitted across many related datasets.
- Bayesian linear and generalized-linear model artifacts for embedded scoring.

**Exit criterion:** each application ships with a representative dataset or generator,
simple baselines, calibration evidence, and a clear statement of its estimand.

## 6. Community and ecosystem

- Maintain a small gallery of end-to-end, reproducible models rather than many overlapping
  demo scripts.
- Publish validation reports that include negative results and known failure regimes.
- Add a pinned subset of PosteriorDB or comparable independent reference targets.
- Document mathematical parameterizations and transformations at the public API boundary.
- Stabilize `rustmc_core` in stages and publish explicit Rust API compatibility policy.
- Provide issue templates for model requests, correctness reports, and reproducible
  benchmarks.
- Label approachable work in diagnostics, distributions, examples, and documentation for
  first-time contributors.

## Deferred directions

The following may become useful, but should not displace the ordered work above:

- broad arbitrary-PPL feature parity;
- GPU or distributed MCMC;
- Stan-language import;
- browser-first inference;
- a large collection of approximate inference algorithms without calibration evidence.

The bar for adding one of these directions is a concrete workload where the existing
architecture is the right foundation and correctness can be measured.
