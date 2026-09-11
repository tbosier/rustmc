"""Estimate related site means with partial pooling and unequal sample counts."""
import numpy as np
import rustmc as rmc


def main():
    rng = np.random.default_rng(42)
    counts = [8, 20, 50, 100]
    site = np.repeat(np.arange(len(counts)), counts).astype(float)
    site_means = np.array([-0.3, 0.2, 0.4, -0.1])
    y = site_means[site.astype(int)] + rng.normal(0, 0.5, len(site))
    model = rmc.ModelBuilder()
    population = model.normal_prior("population", 0.0, 1.0)
    between_sites = model.half_normal_prior("between_sites", 0.5)
    z = model.vector_normal_prior("z", len(counts), 0.0, 1.0)
    mean = population + between_sites * z["site"]
    model.normal_likelihood("reading", mean, 0.5, "y")
    model.deterministic("site_mean", mean)
    fit = model.compile().sample(
        {"site": site, "y": y}, chains=4, warmup=1500, draws=2000,
        target_accept=0.95, seed=42, show_progress=False,
    )
    print(fit.summary())
    means = fit.predict({"site": np.arange(len(counts), dtype=float)}, expected=True)["reading"]
    print("Site posterior means:", means.mean(axis=(0, 1)))
    print("95% credible intervals:", np.quantile(means, [0.025, 0.975], axis=(0, 1)))
    # Compute comparisons within each joint draw to preserve dependence.
    difference = means[:, :, 0] - means[:, :, 1]
    print("P(site 0 mean > site 1 mean):", (difference > 0).mean())


if __name__ == "__main__":
    main()
