What's needed next, ordered by impact

## Tier 1: Credibility and reach

- Expand the modeling surface beyond Normal likelihoods into real GLM families.
- Add explicit support for more link functions and likelihoods that cover common production use cases.
- Strengthen validation tooling with LOO-CV, trace plots, and better predictive checks.

## Tier 2: Production readiness

- Compile a model graph once, serialize it, and load it without Python in deployment.
- Add streaming and online posterior updates for systems that ingest new data continuously.
- Expose a stable C API / FFI so other ecosystems can embed the core engine.

## Tier 3: Competitive moat

- Grow batched multi-model execution into a polished first-class workflow.
- Add WASM compilation for browser and edge deployments.
- Explore GPU-accelerated log-probability evaluation for very large observation sets.
- Add sparse or block-structured mass matrices for large hierarchical models.
- Investigate automatic reparameterization for funnel geometries.

## Current state

- Constrained parameters are now sampled with transforms and Jacobian corrections.
- Scalar hierarchical priors are supported in a limited form.
- Prior predictive and posterior predictive sampling are implemented.
- The remaining gap is breadth of model families and deployment tooling, not basic sampler credibility.
