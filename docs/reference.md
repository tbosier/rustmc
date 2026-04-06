# API Reference

## `ModelBuilder`

```python
builder = rmc.ModelBuilder(data=None)
```

Constructs a model. `data` can be passed here or via `rmc.sample(data=...)`.

`normal_prior()` supports limited scalar hierarchical priors today: `mu` and `sigma` may each be a `ParamRef`, as long as the referenced parameter is declared earlier. `half_normal_prior()` accepts a `ParamRef` sigma, and `normal_likelihood()` accepts a parameter-valued `sigma` too. Vector-valued hierarchical priors are not yet supported.

### Priors

| Method | Distribution | Notes |
|--------|-------------|-------|
| `normal_prior(name, mu, sigma)` | Normal(mu, σ) | `mu`, `sigma` can be `float` or `ParamRef` |
| `half_normal_prior(name, sigma)` | HalfNormal(σ) | sampled in log-space, back-transformed |
| `student_t_prior(name, nu, mu, sigma)` | StudentT(ν, μ, σ) | |
| `beta_prior(name, alpha, beta)` | Beta(α, β) | sampled via logit transform |
| `gamma_prior(name, alpha, beta)` | Gamma(α, β) | sampled in log-space |
| `uniform_prior(name, lower, upper)` | Uniform(a, b) | sampled via logit transform |
| `vector_normal_prior(name, n, mu, sigma)` | Normal(μ, σ)^n | explicit vector of n parameters |

`bernoulli_prior()` and `poisson_prior()` exist in the API but are limited. They are discrete and do not participate cleanly in gradient-based NUTS/HMC workflows today.

All scalar prior methods return a `ParamRef`.

### Likelihoods

```python
builder.normal_likelihood(name, mu_expr, sigma, observed_key)
```

- `mu_expr` — one of: `ParamRef`, `ParamRef * "key"`, `ParamRef + ParamRef * "key"`, `ParamRef @ "key"`
- `sigma` — `float` or `ParamRef`
- `observed_key` — key into the data dict

### `build()`

```python
model = builder.build()
```

Returns a `ModelSpec` (opaque handle passed to `rmc.sample` or `rmc.batch_sample`).

---

## `rmc.sample()`

```python
fit = rmc.sample(
    model_spec,
    data=None,
    chains=4,
    draws=1000,
    warmup=1000,
    seed=42,
    sampler="nuts",   # "nuts" or "hmc"
)
```

Returns a `FitResult`.

`sample()` fits the model with either NUTS or fixed-step HMC. It returns posterior draws plus diagnostics and predictive helpers.

---

## `rmc.batch_sample()`

```python
results = rmc.batch_sample(
    models,    # list of (ModelSpec, data_dict) tuples
    draws=500,
    warmup=300,
    seed=42,
)
```

Returns a list of `BatchResult`, one per model. Each `BatchResult` exposes `mean()`, `std()`, `get_samples()`, `accept_rate`, and `divergences`.

---

## `FitResult`

| Method | Returns | Description |
|--------|---------|-------------|
| `summary()` | `str` | Formatted table of all parameters |
| `mean()` | `dict[str, float]` | Posterior mean per parameter |
| `std()` | `dict[str, float]` | Posterior std per parameter |
| `get_samples()` | `dict[str, np.ndarray]` | All samples flattened across chains for each parameter |
| `get_samples_2d()` | `dict[str, np.ndarray]` | Samples shaped as `(chains, draws)` for each parameter |
| `diagnostics()` | `list[dict]` | Per-parameter diagnostics: r_hat, ess_bulk, ess_tail, mcse, HDIs |
| `step_sizes()` | `list[float]` | Per-chain adapted step size |
| `divergences()` | `list[int]` | Per-chain divergence count |
| `posterior_predictive(n_samples, seed)` | `dict[str, np.ndarray]` | Samples from posterior predictive; `n_samples=None` uses all draws, otherwise posterior draws are randomly subsampled without replacement |
| `to_arviz(include_ppc=False, ppc_samples=None, ppc_seed=42)` | `arviz.InferenceData` | Convert the fit to ArviZ for plotting and inspection |

---

## `rmc.sample_prior_predictive()`

```python
prior_pred = rmc.sample_prior_predictive(model, n_samples=500, seed=0)
```

Returns `dict[str, np.ndarray]` with samples from the prior predictive distribution. Keys include all parameter names and likelihood names.

For `rmc.batch_sample()`, each result is a `BatchResult`:

| Method / property | Returns | Description |
|--------|---------|-------------|
| `mean()` | `dict[str, float]` | Posterior mean per parameter |
| `std()` | `dict[str, float]` | Posterior std per parameter |
| `get_samples()` | `dict[str, np.ndarray]` | Samples per parameter |
| `accept_rate` | `float` | Mean acceptance rate for the chain |
| `divergences` | `int` | Divergence count for the chain |

---

## `ParamRef` operators

`ParamRef` objects support arithmetic for building `mu_expr`:

```python
beta * "x"             # scalar multiply: beta_i * x_i
alpha + beta * "x"     # linear combination
beta @ "X"             # matrix multiply: X @ beta (auto-promotes to vector)
```

Direct use as `mu_expr` (no data key) is also valid for hierarchical models:

```python
mu_j = builder.normal_prior("mu_j", mu=mu_global, sigma=sigma_group)
builder.normal_likelihood("obs_j", mu_expr=mu_j, sigma=2.0, observed_key="y_j")
```

That pattern works for scalar latent parameters. It does not yet extend to hierarchical vector blocks or arbitrary custom likelihoods.
