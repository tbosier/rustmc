"""Cost of ``metric="auto"`` on a range of targets, against ``diag`` and ``dense``.

Run: python -m benchmarks.auto_metric_targets [--seeds N] [--warmup N]
[--metrics auto,diag,dense]
(with the build to measure on PYTHONPATH)

Every target holds at least one vector parameter, so that ``auto`` may choose a dense
block for it. For each target and metric it prints, over the seeds, the median and
worst leapfrog steps per unit of the smallest bulk ESS (warmup steps included), the
fits with any divergence, and the worst R-hat. Fits use 4 chains and 1000 draws,
after 1000 warmup iterations unless ``--warmup`` says otherwise. Running it against two builds compares how they choose the metric; the
statistical-validation page quotes such a comparison.
"""
from __future__ import annotations

import argparse

import numpy as np
import rustmc as rmc

CHAINS = 4
WARMUP = 1000
DRAWS = 1000

SCHOOLS_Y = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
SCHOOLS_SIGMA = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])


def isotropic(d):
    b = rmc.ModelBuilder()
    beta = b.vector_normal_prior("beta", d, 0.0, 5.0)
    b.normal_likelihood("obs", beta @ "X", 1.0, "y")
    return b.compile(), {"X": np.eye(d), "y": np.random.default_rng(0).uniform(-1, 1, d)}


def correlated_regression(d, rho=0.9, n=300):
    rng = np.random.default_rng(0)
    shared = rng.standard_normal((n, 1))
    X = np.sqrt(rho) * shared + np.sqrt(1 - rho) * rng.standard_normal((n, d))
    y = X @ np.linspace(-1, 1, d) + rng.standard_normal(n)
    b = rmc.ModelBuilder()
    beta = b.vector_normal_prior("beta", d, 0.0, 1.0)
    b.normal_likelihood("obs", beta @ "X", 1.0, "y")
    return b.compile(), {"X": X, "y": y}


def scaled_correlated_regression(d=6, rho=0.5, n=200):
    # Predictor scales from 0.01 to 100, so posterior sds span four orders.
    rng = np.random.default_rng(1)
    shared = rng.standard_normal((n, 1))
    Z = np.sqrt(rho) * shared + np.sqrt(1 - rho) * rng.standard_normal((n, d))
    X = Z * np.logspace(-2, 2, d)
    y = Z @ np.ones(d) + rng.standard_normal(n)
    b = rmc.ModelBuilder()
    beta = b.vector_normal_prior("beta", d, 0.0, 1000.0)
    b.normal_likelihood("obs", beta @ "X", 1.0, "y")
    return b.compile(), {"X": X, "y": y}


def logistic_regression(d=8, rho=0.7, n=200):
    rng = np.random.default_rng(2)
    shared = rng.standard_normal((n, 1))
    X = np.sqrt(rho) * shared + np.sqrt(1 - rho) * rng.standard_normal((n, d))
    eta = X @ np.linspace(-1.5, 1.5, d)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(float)
    b = rmc.ModelBuilder()
    beta = b.vector_normal_prior("beta", d, 0.0, 2.5)
    b.bernoulli_logit_likelihood("obs", beta @ "X", "y")
    return b.compile(), {"X": X, "y": y}


def eight_schools_noncentered():
    # y_j / sigma_j ~ Normal((mu + tau z_j) / sigma_j, 1): the same posterior as
    # y_j ~ Normal(mu + tau z_j, sigma_j), since sigma_j is known.
    b = rmc.ModelBuilder()
    mu = b.normal_prior("mu", 0.0, 5.0)
    tau = b.half_normal_prior("tau", 5.0)
    z = b.vector_normal_prior("z", 8, 0.0, 1.0)
    b.normal_likelihood("obs", (mu + tau * z["school"]) * "inv_sigma", 1.0, "y_scaled")
    data = {"school": np.arange(8.0), "inv_sigma": 1 / SCHOOLS_SIGMA,
            "y_scaled": SCHOOLS_Y / SCHOOLS_SIGMA}
    return b.compile(), data


def eight_schools_centered():
    # theta_j ~ Normal(mu, tau) written as a potential over a vector with a
    # prior too broad to matter: Neal's funnel in theta and log tau.
    b = rmc.ModelBuilder()
    mu = b.normal_prior("mu", 0.0, 5.0)
    tau = b.half_normal_prior("tau", 5.0)
    theta = b.vector_normal_prior("theta", 8, 0.0, 1e4)
    t = theta["school"]
    b.potential(
        "population",
        (-0.5 * ((t - mu) / tau) ** 2 - tau.log() + 0.5 * (t / 1e4) ** 2).sum(),
    )
    b.normal_likelihood("obs", t * "inv_sigma", 1.0, "y_scaled")
    data = {"school": np.arange(8.0), "inv_sigma": 1 / SCHOOLS_SIGMA,
            "y_scaled": SCHOOLS_Y / SCHOOLS_SIGMA}
    return b.compile(), data


def banana():
    data = {"i": np.array([0.0]), "j": np.array([1.0])}
    b = rmc.ModelBuilder(data)
    q = b.vector_normal_prior("q", 2, 0.0, 1000.0)
    x, y = q["i"], q["j"]
    b.potential(
        "banana",
        (-0.5 * (x / 5) ** 2 - 0.5 * (y - 0.5 * (x * x - 25)) ** 2
         + 0.5 * (x * x + y * y) / 1e6).sum(),
    )
    return b.compile(), data


TARGETS = {
    "isotropic d=50": lambda: isotropic(50),
    "isotropic d=100": lambda: isotropic(100),
    "correlated regression d=10": lambda: correlated_regression(10),
    "correlated regression d=20": lambda: correlated_regression(20),
    "scaled correlated regression d=6": scaled_correlated_regression,
    "logistic regression d=8": logistic_regression,
    "eight schools noncentered": eight_schools_noncentered,
    "eight schools centered": eight_schools_centered,
    "banana": banana,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--metrics", default="auto,diag,dense")
    parser.add_argument("--targets", default=",".join(TARGETS))
    args = parser.parse_args()
    seeds = range(1, args.seeds + 1)
    print(f"rustmc {rmc.__version__} ({rmc.__file__})")
    print(f"chains={CHAINS} warmup={args.warmup} draws={DRAWS} seeds={list(seeds)}")
    print(f"{'target':34s} {'metric':6s} {'steps/ESS':>9s} {'worst':>9s} "
          f"{'divergent fits':>14s} {'worst R-hat':>11s}")
    for name in args.targets.split(","):
        compiled, data = TARGETS[name]()
        for metric in args.metrics.split(","):
            costs, divergent, r_hats = [], 0, []
            for seed in seeds:
                fit = compiled.sample(data, chains=CHAINS, warmup=args.warmup, draws=DRAWS,
                                      seed=seed, metric=metric, show_progress=False)
                rows = fit.diagnostics()
                report = fit.transition_diagnostics()
                costs.append(report["total_leapfrog_steps"]
                             / min(row["ess_bulk"] for row in rows))
                divergent += report["total_divergences"] > 0
                r_hats.append(max(row["r_hat"] for row in rows))
            print(f"{name:34s} {metric:6s} {np.median(costs):9.1f} {max(costs):9.1f} "
                  f"{divergent:>9d} / {len(seeds):<2d} {max(r_hats):11.4f}", flush=True)


if __name__ == "__main__":
    main()
