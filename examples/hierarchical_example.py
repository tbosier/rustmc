"""Hierarchical models

Partial pooling across J groups, in the shape of the "eight schools" model.

    mu_global   ~ Normal(0, 10)              global mean hyperprior
    sigma_group ~ HalfNormal(5)              between-group scale hyperprior
    mu_j        ~ Normal(mu_global, sigma_group)     group mean, j = 0 .. J-1
    y_ij        ~ Normal(mu_j, sigma_obs)            observations within a group

`mu_global` and `sigma_group` are hyperparameters: the prior on each group mean is
itself estimated. That is what ties the groups together. A group with few
observations is pulled toward the global mean; a group with many stays near its own
sample mean.

The model is written in the conditional, "centered" form above, which is the form
that reads like the mathematics. rustmc compiles eligible scalar hierarchies to
noncentered sampling coordinates internally, so the awkward geometry of the centered
form does not reach the sampler. `mu_j` is still what you see in summaries,
diagnostics and posterior draws.
"""

# %%
import numpy as np
import rustmc as rmc
from hierarchical_templates import build_centered_normal_partial_pooling

# %% Simulate data
rng = np.random.default_rng(42)

J = 8  # number of groups
sigma_obs = 2.0  # known within-group noise
N_per_group = 30  # observations per group

mu_global_true = 2.5
sigma_group_true = 3.0
mu_true = rng.normal(mu_global_true, sigma_group_true, J)
ys = [rng.normal(mu_true[j], sigma_obs, N_per_group) for j in range(J)]
data = {f"y_{j}": ys[j] for j in range(J)}

print("Simulated data")
print(f"  True mu_global   = {mu_global_true:.2f}")
print(f"  True sigma_group = {sigma_group_true:.2f}")
print(f"  True mu_j        = {np.round(mu_true, 3).tolist()}")

# %% Build the hierarchy
# `build_centered_normal_partial_pooling` lives in examples/hierarchical_templates.py
# and does nothing you could not write inline: one `normal_prior` for the global
# mean, one `half_normal_prior` for the between-group scale, then one
# `normal_prior(mu=mu_global, sigma=sigma_group)` and one likelihood per group.
builder = rmc.ModelBuilder(data=data)
template = build_centered_normal_partial_pooling(
    builder,
    observed_keys=[f"y_{j}" for j in range(J)],
    sigma_obs=sigma_obs,
)
mu_global = template.mu_global
sigma_group = template.sigma_group
mu_j = template.group_params

model = builder.build()

# %% Prior predictive check
prior_pred = rmc.sample_prior_predictive(model, n_samples=200, seed=0)
print(f"  mu_global   prior: mean={prior_pred['mu_global'].mean():.2f}, std={prior_pred['mu_global'].std():.2f}")
print(f"  sigma_group prior: mean={prior_pred['sigma_group'].mean():.2f}, std={prior_pred['sigma_group'].std():.2f}")
for j in range(J):
    key = f"obs_{j}"
    if key in prior_pred:
        prior_y = prior_pred[key]
        print(f"  Group {j} prior y range: [{prior_y.min():.1f}, {prior_y.max():.1f}]")

# %% Sample
fit = rmc.sample(
    model_spec=model,
    chains=4,
    draws=2000,
    warmup=1000,
    seed=42,
)
print(fit.summary())

# %% Recover the parameters
means = fit.mean()
stds = fit.std()

print(f"{'Parameter':<15} {'True':>8} {'Estimate':>10} {'Std':>8}")
print("-" * 45)
print(f"{'mu_global':<15} {mu_global_true:>8.2f} {means['mu_global']:>10.4f} {stds['mu_global']:>8.4f}")
print(f"{'sigma_group':<15} {sigma_group_true:>8.2f} {means['sigma_group']:>10.4f} {stds['sigma_group']:>8.4f}")
for j in range(J):
    key = f"mu_{j}"
    print(f"  {key:<13} {mu_true[j]:>8.2f} {means[key]:>10.4f} {stds[key]:>8.4f}")

print()
print("Step sizes:", [round(s, 5) for s in fit.step_sizes()])
print("Divergences:", fit.divergences())

# %% [markdown]
# ## What the hyperparameters can and cannot say
#
# `mu_global` has a wide posterior and it should. Eight group means drawn from a
# distribution carry about as much information about that distribution's mean as
# eight observations do, so the interval stays broad however many draws you take.
# `sigma_group` is estimated from the same eight numbers and is biased upward when
# the groups happen to spread further than the truth.
#
# The centered form of this model is the textbook case of Neal's funnel, where the
# sampler stalls in the neck and reports divergent transitions. Those do not appear
# here because rustmc compiles this hierarchy to noncentered coordinates. If you
# write a hierarchy rustmc cannot recognise and see divergences, reparameterise it
# by hand before trusting the hyperparameter estimates.

# %% Partial pooling
print("Partial pooling effect (shrinkage toward the global mean):")
print(f"  Global mean estimate: {means['mu_global']:.2f}")
sample_means = [ys[j].mean() for j in range(J)]
for j in range(J):
    est = means[f"mu_{j}"]
    raw = sample_means[j]
    print(f"  Group {j}: raw={raw:+.2f}  pooled={est:+.2f}  true={mu_true[j]:+.2f}")

print()
print("Each group has 30 observations and known noise, so the data pins mu_j down")
print("and the pull toward the global mean is small. Shrinkage grows as a group's")
print("sample size falls; examples/site_effects.py shows it with unequal counts.")

# %% Posterior predictive check
ppc = fit.posterior_predictive(n_samples=500, seed=42)
print(f"  Likelihood keys in PPC: {sorted(ppc.keys())}")
for j in range(J):
    key = f"obs_{j}"
    if key in ppc:
        y_rep = ppc[key]  # (n_samples, N_per_group)
        y_obs = ys[j]
        lower = y_rep.mean(axis=0) - 2 * y_rep.std(axis=0)
        upper = y_rep.mean(axis=0) + 2 * y_rep.std(axis=0)
        inside = ((lower < y_obs) & (y_obs < upper)).mean()
        print(f"  Group {j}: obs mean={y_obs.mean():.2f}  ppc mean={y_rep.mean():.2f}  within +/-2 sd={inside:.2%}")
