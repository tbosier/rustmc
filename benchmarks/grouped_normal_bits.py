"""Hash seeded draws, log densities and gradients of grouped models, for cross-build bit checks.

Prints one line per model: a hash of the seeded draws (2 chains, 300 warmup, 300 draws),
a hash of log density and gradient bits at 48 positions with scales up to 800, and, on
builds that have it, how many fused grouped-Normal reverse passes the model ran.
"""
import hashlib
import numpy as np
import rustmc as rmc
from rustmc import _rustmc

rng = np.random.default_rng(5)
rows, groups = 300, 6
group = rng.integers(0, groups, size=rows).astype(float)
x = rng.normal(size=rows)
effects = 0.7 * rng.normal(size=groups)
y = 0.4 + effects[group.astype(int)] + rng.normal(scale=0.6, size=rows)
y2 = -0.2 + effects[group.astype(int)] + rng.normal(scale=0.6, size=rows)
eta = 0.2 + 0.5 * effects[group.astype(int)]
yes = (rng.random(rows) < 1 / (1 + np.exp(-eta))).astype(float)
counts = rng.poisson(np.exp(eta)).astype(float)
schools_y = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
schools_sigma = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])
data = {"group": group, "x": x, "y": y, "y2": y2, "yes": yes, "counts": counts,
        "school": np.arange(8.0), "y_scaled": schools_y / schools_sigma, "inv_sigma": 1 / schools_sigma}


def model(name):
    b = rmc.ModelBuilder()
    mu = b.normal_prior("mu", 0.0, 5.0)
    tau = b.half_normal_prior("tau", 1.0)
    if name == "eight_schools":
        z = b.vector_normal_prior("z", n=8, mu=0.0, sigma=1.0)
        b.normal_likelihood("obs", (mu + tau * z["school"]) * "inv_sigma", 1.0, "y_scaled")
        return b, ["school", "y_scaled", "inv_sigma"]
    z = b.vector_normal_prior("z", n=groups, mu=0.0, sigma=1.0)
    sigma = b.half_normal_prior("sigma", 1.0)
    keys = ["group", "y"]
    if name == "varying_intercept":
        b.normal_likelihood("obs", mu + tau * z["group"], sigma, "y")
    elif name == "constant_scales":
        b.normal_likelihood("obs", mu + 0.8 * z["group"], 0.6, "y")
    elif name == "scale_written_last":
        b.normal_likelihood("obs", mu + z["group"] * tau, sigma, "y")
    elif name == "effect_only":
        b.normal_likelihood("obs", z["group"], sigma, "y")
    elif name == "mean_plus_effect":
        b.normal_likelihood("obs", mu + z["group"], sigma, "y")
    elif name == "two_terms":
        b.normal_likelihood("obs", mu + tau * z["group"], sigma, "y")
        b.normal_likelihood("obs2", mu + tau * z["group"], sigma, "y2")
        keys.append("y2")
    elif name == "slope_on_data":
        b.normal_likelihood("obs", mu + z["group"] * "x", sigma, "y")
        keys.append("x")
    elif name == "logistic":
        b.bernoulli_logit_likelihood("obs", mu + tau * z["group"], "yes")
        keys = ["group", "yes"]
    elif name == "poisson":
        b.poisson_log_likelihood("obs", mu + tau * z["group"], "counts")
        keys = ["group", "counts"]
    return b, keys


def passes():
    f = getattr(_rustmc, "_fused_grouped_normal_passes", None)
    return f()[1] if f else None


names = ["varying_intercept", "constant_scales", "scale_written_last", "effect_only",
         "mean_plus_effect", "two_terms", "slope_on_data", "eight_schools", "logistic", "poisson"]
for name in names:
    b, keys = model(name)
    compiled = b.compile()
    bound = {k: data[k] for k in keys}
    before = passes()
    fit = compiled.sample(bound, chains=2, warmup=300, draws=300, seed=11, show_progress=False)
    draws = np.stack([np.asarray(v) for _, v in sorted(fit.get_samples_2d().items())])
    pos_rng = np.random.default_rng(23)
    h = hashlib.sha256()
    for k in range(48):
        scale = [0.5, 3.0, 40.0, 800.0][k % 4]
        position = pos_rng.normal(scale=scale, size=len(compiled.param_names))
        density, gradient = compiled.log_density(bound, position)
        h.update(np.float64(density).tobytes())
        h.update(np.asarray(gradient, dtype=np.float64).tobytes())
    after = passes()
    fused = None if before is None else after - before
    print(f"{name:20} draws {hashlib.sha256(draws.tobytes()).hexdigest()[:16]} "
          f"densities {h.hexdigest()[:16]} fused reverse passes {fused}", flush=True)
