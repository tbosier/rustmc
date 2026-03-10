"""
rustmc — Hierarchical / Multilevel Model Example
=================================================

Partial pooling across J groups (8-schools style).

Model
-----
    mu_global  ~ Normal(0, 10)           # global mean hyperprior
    sigma_group ~ HalfNormal(5)          # between-group SD hyperprior

    mu_j ~ Normal(mu_global, sigma_group)  # group-level mean  (j = 0 … J-1)

    y_ij ~ Normal(mu_j, sigma_obs)         # within-group observations

This is the prototypical hierarchical model:
- mu_global and sigma_group are *hyperparameters* — parameters whose prior
  is set by the user.
- mu_j are *group-level parameters* — each gets its own estimate, but they
  are tied together through the shared hyperprior Normal(mu_global, sigma_group).
- Partial pooling: groups with few observations are pulled toward the global
  mean; groups with many observations stay close to their sample mean.
"""

import numpy as np
import rustmc as rmc

# ── 1. Simulate data ─────────────────────────────────────────────────────────

np.random.seed(42)

J = 8                                       # number of groups
sigma_obs = 2.0                             # known within-group noise
N_per_group = 30                            # observations per group

# True group means — deliberately spread across a wide range
mu_true = np.array([5.0, -1.0, 3.0, 0.0, 8.0, 2.0, -3.0, 6.0])
# True hyperparameters
mu_global_true   = mu_true.mean()           # ≈ 2.5
sigma_group_true = mu_true.std()            # ≈ 3.7

ys = [np.random.normal(mu_true[j], sigma_obs, N_per_group) for j in range(J)]
data = {f"y_{j}": ys[j] for j in range(J)}

print("Simulated data")
print(f"  True mu_global   = {mu_global_true:.2f}")
print(f"  True sigma_group = {sigma_group_true:.2f}")
print(f"  True mu_j        = {mu_true.tolist()}")
print()

# ── 2. Build the hierarchical model ──────────────────────────────────────────

builder = rmc.ModelBuilder(data=data)

# Hyperpriors (global-level parameters)
mu_global   = builder.normal_prior("mu_global",   mu=0.0, sigma=10.0)
sigma_group = builder.half_normal_prior("sigma_group", sigma=5.0)

# Group-level parameters: each mu_j ~ Normal(mu_global, sigma_group)
# mu_global and sigma_group are ParamRef objects — rustmc resolves them
# to graph nodes at sample time so gradients flow up to the hyperpriors.
mu_j = [
    builder.normal_prior(f"mu_{j}", mu=mu_global, sigma=sigma_group)
    for j in range(J)
]

# One likelihood per group
for j in range(J):
    builder.normal_likelihood(
        f"obs_{j}",
        mu_expr=mu_j[j],
        sigma=sigma_obs,
        observed_key=f"y_{j}",
    )

model = builder.build()

# ── 3. Sample ────────────────────────────────────────────────────────────────

print("Sampling …")
fit = rmc.sample(
    model_spec=model,
    chains=4,
    draws=2000,
    warmup=1000,
    seed=42,
)

# ── 4. Results ───────────────────────────────────────────────────────────────

print()
print(fit.summary())
print()

means = fit.mean()
stds  = fit.std()

print(f"{'Parameter':<15} {'True':>8} {'Estimate':>10} {'Std':>8}")
print("─" * 45)
print(f"{'mu_global':<15} {mu_global_true:>8.2f} {means['mu_global']:>10.4f} {stds['mu_global']:>8.4f}")
print(f"{'sigma_group':<15} {sigma_group_true:>8.2f} {means['sigma_group']:>10.4f} {stds['sigma_group']:>8.4f}")
for j in range(J):
    key = f"mu_{j}"
    print(f"  {key:<13} {mu_true[j]:>8.2f} {means[key]:>10.4f} {stds[key]:>8.4f}")

print()
print("Step sizes:", [round(s, 5) for s in fit.step_sizes()])
print("Divergences:", fit.divergences())

# ── 5. Show partial pooling ───────────────────────────────────────────────────

print()
print("Partial pooling effect (shrinkage toward global mean):")
print(f"  Global mean estimate: {means['mu_global']:.2f}")
sample_means = [ys[j].mean() for j in range(J)]
for j in range(J):
    est  = means[f"mu_{j}"]
    raw  = sample_means[j]
    shrinkage = (raw - est) / (raw - means["mu_global"] + 1e-9)
    print(f"  Group {j}: raw={raw:+.2f}  pooled={est:+.2f}  true={mu_true[j]:+.2f}")

print()
print("Note: mu_global may have elevated R-hat / low ESS due to the classic")
print("'Neal's funnel' geometry in centered hierarchical parameterizations.")
print("The group-level parameters (mu_j) converge cleanly because the data")
print("strongly constrains them.  A non-centered reparameterization would")
print("improve sampling of the hyperparameters at the cost of a more complex")
print("model specification.")
