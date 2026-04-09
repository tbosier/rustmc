# Hierarchical Templates

This page describes the current reusable-hierarchy boundary in rustmc.

## What Exists Today

rustmc supports scalar hierarchical priors directly in the builder:

```python
mu_global = builder.normal_prior("mu_global", mu=0.0, sigma=10.0)
sigma_group = builder.half_normal_prior("sigma_group", sigma=5.0)
mu_j = builder.normal_prior("mu_j", mu=mu_global, sigma=sigma_group)
```

That pattern works for centered-looking model code, but eligible scalar hierarchical
normals are automatically compiled through a non-centered latent internally. Users still
see `mu_j` in summaries, diagnostics, posterior samples, prior predictive draws, and
ArviZ export.

## Current Scope

Supported today:

- scalar hierarchical `Normal`
- scalar hierarchical `HalfNormal`
- scalar hierarchical `Exponential`
- scalar hierarchical `LogNormal`
- parameter-valued likelihood scale terms such as `sigma` or `alpha`

Not supported yet:

- vector-valued hierarchical random effects
- grouped varying-slope blocks compiled automatically from one declaration
- a dedicated high-level template API such as `builder.hierarchical_normal(...)`

## Practical Pattern

The recommended current pattern is still to write the hierarchy explicitly:

```python
builder = rmc.ModelBuilder(data={"y": y})

mu_global = builder.normal_prior("mu_global", mu=0.0, sigma=5.0)
sigma_group = builder.half_normal_prior("sigma_group", sigma=2.0)
theta = builder.normal_prior("theta", mu=mu_global, sigma=sigma_group)

builder.normal_likelihood("obs", mu_expr=theta, sigma=1.0, observed_key="y")
model = builder.build()
```

This keeps the Python surface simple while letting the compiler apply the safer geometry
internally.

## What Is Next

The next meaningful extension is vector/group random effects, where automatic non-centering
matters most. That will likely require:

- logical-parameter mappings for vector blocks
- grouped latent templates in the builder
- better support for correlated random effects
