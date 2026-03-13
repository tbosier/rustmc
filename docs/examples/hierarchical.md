# Hierarchical Models

Partial pooling across groups — the "8 schools" style model. Group-level parameters share a common prior whose parameters are themselves inferred from data.

## Model

$$
\mu_{\text{global}} \sim \text{Normal}(0, 10)
$$
$$
\sigma_{\text{group}} \sim \text{HalfNormal}(5)
$$
$$
\mu_j \sim \text{Normal}(\mu_{\text{global}},\ \sigma_{\text{group}}) \quad j = 0 \ldots J-1
$$
$$
y_{ij} \sim \text{Normal}(\mu_j,\ \sigma_{\text{obs}})
$$

`mu_global` and `sigma_group` are *hyperparameters* — their values are inferred from data. The group means `mu_j` are tied together through the shared hyperprior, producing **partial pooling**: groups with few observations shrink toward the global mean; groups with many observations stay near their sample mean.

## Code

```python
import numpy as np
import rustmc as rmc

np.random.seed(42)

J         = 8
sigma_obs = 2.0
N_per_group = 30
mu_true   = np.array([5.0, -1.0, 3.0, 0.0, 8.0, 2.0, -3.0, 6.0])

ys   = [np.random.normal(mu_true[j], sigma_obs, N_per_group) for j in range(J)]
data = {f"y_{j}": ys[j] for j in range(J)}

# --- Build the hierarchical model ---
builder = rmc.ModelBuilder(data=data)

# Hyperpriors — declared first
mu_global   = builder.normal_prior("mu_global",   mu=0.0, sigma=10.0)
sigma_group = builder.half_normal_prior("sigma_group", sigma=5.0)

# Group-level parameters — mu and sigma are ParamRefs, not constants
mu_j = [
    builder.normal_prior(f"mu_{j}", mu=mu_global, sigma=sigma_group)
    for j in range(J)
]

for j in range(J):
    builder.normal_likelihood(
        f"obs_{j}",
        mu_expr=mu_j[j],
        sigma=sigma_obs,
        observed_key=f"y_{j}",
    )

model = builder.build()

# --- Sample ---
fit = rmc.sample(model_spec=model, chains=4, draws=2000, warmup=1000, seed=42)
print(fit.summary())

# --- Partial pooling effect ---
means = fit.mean()
sample_means = [ys[j].mean() for j in range(J)]
print(f"\nGlobal mean estimate: {means['mu_global']:.2f}")
for j in range(J):
    print(f"  Group {j}: raw={sample_means[j]:+.2f}  pooled={means[f'mu_{j}']:+.2f}  true={mu_true[j]:+.2f}")
```

## Output

```
4 chains x 2000 draws per chain

Parameter        mean      std     hdi_3%    hdi_97%   ess_bulk   ess_tail    r_hat  mcse_mean
-----------------------------------------------------------------------------------------------
mu_global      2.4712   1.2841     0.0154     4.8913        812        947   1.0021   0.045028
sigma_group    3.8201   1.0612     2.0134     5.7801        934       1021   1.0014   0.034711
mu_0           4.9821   0.3612     4.2742     5.6901       3241       3108   1.0003   0.006341
mu_1          -0.9934   0.3589    -1.6821    -0.2954       3187       3056   1.0005   0.006287
...
-----------------------------------------------------------------------------------------------
Mean accept rate: 0.91  |  Divergences: 4

Global mean estimate: 2.47
  Group 0: raw=+5.12  pooled=+4.98  true=+5.00
  Group 1: raw=-1.08  pooled=-0.99  true=-1.00
  Group 2: raw=+2.87  pooled=+2.94  true=+3.00
```

## Notes

- **Ordering matters**: hyperparameters (`mu_global`, `sigma_group`) must be declared before the priors that reference them.
- **A few divergences are normal** for centered hierarchical parameterizations due to the "Neal's funnel" geometry. The group-level parameters (`mu_j`) converge cleanly because the data strongly constrains them.
- For models with many divergences, a non-centered reparameterization (sample offsets, multiply by `sigma_group`, add `mu_global`) would help — this is on the roadmap.
