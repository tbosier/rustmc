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
print("Site posterior means:", np.round(means.mean(axis=(0, 1)), 4))
print("95% credible intervals:")
print(np.round(np.quantile(means, [0.025, 0.975], axis=(0, 1)), 4))

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
