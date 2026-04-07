# Hierarchical Templates

This page captures the reusable hierarchical template boundary for rustmc.
It is the first step toward a proper template API, but it stays within the
current centered-model capability so it does not depend on the GLM substrate
work.

## What Exists Today

rustmc can already express centered scalar hierarchies:

```python
mu_global = builder.normal_prior("mu_global", mu=0.0, sigma=10.0)
sigma_group = builder.half_normal_prior("sigma_group", sigma=5.0)
mu_j = builder.normal_prior("mu_j", mu=mu_global, sigma=sigma_group)
```

That is enough for partial pooling, but not yet for true non-centered random
effects or vector-valued hierarchical blocks.

## Reusable Helper

The repo now includes a small helper module:

```python
from hierarchical_templates import build_centered_normal_partial_pooling

template = build_centered_normal_partial_pooling(
    builder,
    observed_keys=[f"y_{j}" for j in range(J)],
    sigma_obs=sigma_obs,
)
```

This helper builds the centered partial-pooling pattern and returns the
global hyperprior, group-scale hyperprior, and per-group latent parameters.
It is intentionally narrow: its job is to make the current supported pattern
reusable while the core DSL grows.

## Future Contract

The eventual non-centered API should preserve the same conceptual structure:

```python
template = builder.hierarchical_normal(
    name="mu",
    location=mu_global,
    scale=sigma_group,
    shape=J,
    centered=False,
)
```

That future form is not implemented yet. It requires parameter-to-parameter
transform support in the DSL and the corresponding graph/autodiff changes.

## Ownership Notes

- The helper layer is implemented in Python-facing code only.
- The non-centered runtime path is deferred to the core substrate workstream.
- The GLM expansion should land first because the same DSL machinery will be
  used for both hierarchical templates and family-specific observation models.

## Example

See [Hierarchical Models](hierarchical.md) for the centered
partial-pooling example built on top of this helper.
