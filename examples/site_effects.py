"""Site effects

Estimate related site means with partial pooling and unequal sample counts.

A partial-pooling model estimates a shared population mean, a between-site scale,
and each site's deviation from the population. Sites with fewer readings borrow
information from the population, so their estimates are pulled toward it; sites
with many readings stay close to their own data.

The observation noise is known here and fixed at 0.5. A real model can give it a
positive prior instead.
"""

# %%
import numpy as np
import rustmc as rmc

# %% Four sites with 8 to 100 readings each
rng = np.random.default_rng(42)
counts = [8, 20, 50, 100]
site = np.repeat(np.arange(len(counts)), counts).astype(float)
site_means = np.array([-0.3, 0.2, 0.4, -0.1])
y = site_means[site.astype(int)] + rng.normal(0, 0.5, len(site))

for index, count in enumerate(counts):
    observed = y[site == index]
    print(f"site {index}: n={count:3d}  true mean={site_means[index]:+.2f}  sample mean={observed.mean():+.3f}")

# %% [markdown]
# ## A noncentered parameterization
#
# The model is written as `population + between_sites * z[site]`, with `z` standard
# normal, rather than drawing each site mean directly from
# `Normal(population, between_sites)`. The two are the same model; this form gives
# the sampler a posterior whose shape does not change with `between_sites`, which
# is what keeps a hierarchy with few groups out of Neal's funnel.
#
# `deterministic` records `site_mean` so the per-site means are stored alongside the
# parameters instead of being reconstructed afterwards.

# %% Build and fit
model = rmc.ModelBuilder()
population = model.normal_prior("population", 0.0, 1.0)
between_sites = model.half_normal_prior("between_sites", 0.5)
z = model.vector_normal_prior("z", len(counts), 0.0, 1.0)
mean = population + between_sites * z["site"]
model.normal_likelihood("reading", mean, 0.5, "y")
model.deterministic("site_mean", mean)

fit = model.compile().sample(
    {"site": site, "y": y},
    chains=4,
    warmup=1500,
    draws=2000,
    target_accept=0.95,
    seed=42,
    show_progress=False,
)
print(fit.summary())

# %% Site means and intervals
means = fit.predict({"site": np.arange(len(counts), dtype=float)}, expected=True)["reading"]
posterior_mean = means.mean(axis=(0, 1))
lower, upper = np.quantile(means, [0.025, 0.975], axis=(0, 1))
print(f"{'site':>5} {'n':>5} {'mean':>8} {'2.5%':>8} {'97.5%':>8}")
for index, count in enumerate(counts):
    print(f"{index:>5} {count:>5} {posterior_mean[index]:+8.3f} {lower[index]:+8.3f} {upper[index]:+8.3f}")

# %% How far each site moved toward the population
print(f"Population mean estimate: {fit.mean()['population']:+.3f}")
print(f"{'site':>5} {'n':>5} {'sample':>8} {'pooled':>8} {'moved':>8}")
for index, count in enumerate(counts):
    sample_mean = y[site == index].mean()
    moved = posterior_mean[index] - sample_mean
    print(f"{index:>5} {count:>5} {sample_mean:+8.3f} {posterior_mean[index]:+8.3f} {moved:+8.3f}")

# %% [markdown]
# ## Comparisons need paired draws
#
# The probability that one site's mean exceeds another's is computed inside each
# joint draw, so the dependence between the two estimates is preserved. Shuffling
# the sites independently, or fitting them separately and comparing the marginals,
# would throw that dependence away and give a different answer.
#
# These are equal-tailed quantile intervals, not highest-density intervals.

# %% Compare two sites
difference = means[:, :, 0] - means[:, :, 1]
print("P(site 0 mean > site 1 mean):", (difference > 0).mean())
