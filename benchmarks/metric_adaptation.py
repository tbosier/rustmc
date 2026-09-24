"""Warmup metric and schedule measurements for the graph-model NUTS sampler.

Run: python -m benchmarks.metric_adaptation [--quick]

Prints three tables, each row naming its model, options and seeds, so that the
numbers quoted for the metric and warmup-schedule changes can be reproduced:

1. isotropic vector regressions (d = 50, 100, 300) under each ``metric``, with
   the same target written as d scalar parameters as a reference;
2. equicorrelated (rho = 0.9) vector regressions, d = 10 and 20;
3. a warmup-length sweep on a well-scaled and a badly scaled regression.

``steps/draw`` is leapfrog steps per iteration over warmup and draws together,
the cost the metric decides. Timings depend on the machine and are indicative
only; the step counts and divergences are deterministic for a given build.
"""
from __future__ import annotations

import argparse
import platform
import time

import numpy as np
import rustmc as rmc

CHAINS = 4
DRAWS = 1000
WARMUP = 1000
METRICS = ("auto", "diag", "dense")


def steps_per_draw(fit):
    report = fit.transition_diagnostics()
    return report["total_leapfrog_steps"] / report["total_transitions"]


def min_bulk_ess(fit):
    return min(row["ess_bulk"] for row in fit.diagnostics())


def run(model, data=None, **options):
    start = time.perf_counter()
    fit = rmc.sample(model, data=data, chains=CHAINS, show_progress=False, **options)
    return fit, time.perf_counter() - start


def isotropic_vector(d):
    # Identity design: every coefficient has an independent posterior of sd ~1.
    b = rmc.ModelBuilder()
    beta = b.vector_normal_prior("beta", d, 0.0, 5.0)
    b.normal_likelihood("obs", beta @ "X", 1.0, "y")
    return b.build(), {"X": np.eye(d), "y": np.random.default_rng(0).uniform(-1, 1, d)}


def isotropic_scalars(d):
    y = np.random.default_rng(0).uniform(-1, 1, d)
    b = rmc.ModelBuilder()
    for i in range(d):
        beta = b.normal_prior(f"beta{i}", 0.0, 5.0)
        b.normal_likelihood(f"obs{i}", beta, 1.0, f"y{i}")
    return b.build(), {f"y{i}": y[i:i + 1] for i in range(d)}


def correlated_vector(d, rho=0.9, n=300):
    rng = np.random.default_rng(0)
    shared = rng.standard_normal((n, 1))
    X = np.sqrt(rho) * shared + np.sqrt(1 - rho) * rng.standard_normal((n, d))
    y = X @ np.linspace(-1, 1, d) + rng.standard_normal(n)
    b = rmc.ModelBuilder()
    beta = b.vector_normal_prior("beta", d, 0.0, 1.0)
    b.normal_likelihood("obs", beta @ "X", 1.0, "y")
    return b.build(), {"X": X, "y": y}


def well_scaled_regression():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(30)
    b = rmc.ModelBuilder()
    a = b.normal_prior("a", 0.0, 10.0)
    s = b.normal_prior("s", 0.0, 10.0)
    sigma = b.half_normal_prior("sigma", 5.0)
    b.normal_likelihood("obs", a + s * "x", sigma, "y")
    return b.build(), {"x": x, "y": 1 + 2 * x + rng.standard_normal(30)}


def badly_scaled_regression():
    # Posterior sds of about 7 for the intercept and 0.007 for the slope.
    rng = np.random.default_rng(0)
    x = rng.standard_normal(200) * 1000
    b = rmc.ModelBuilder()
    a = b.normal_prior("a", 0.0, 1000.0)
    s = b.normal_prior("s", 0.0, 10.0)
    b.normal_likelihood("obs", a + s * "x", 100.0, "y")
    return b.build(), {"x": x, "y": 0.5 * x + rng.standard_normal(200) * 100}


def metric_rows(title, cases, seeds):
    print(f"\n## {title}")
    print(f"chains={CHAINS} draws={DRAWS} warmup={WARMUP} seeds={list(seeds)}; "
          "mean over seeds")
    print(f"{'model':28s} {'metric':7s} {'steps/draw':>10s} {'min ESS':>8s} "
          f"{'divergences':>11s} {'seconds':>8s}")
    for label, (model, data), metrics in cases:
        for metric in metrics:
            steps, ess, divergences, seconds = [], [], [], []
            for seed in seeds:
                options = dict(draws=DRAWS, warmup=WARMUP, seed=seed)
                if metric != "-":
                    options["metric"] = metric
                fit, elapsed = run(model, data, **options)
                steps.append(steps_per_draw(fit))
                ess.append(min_bulk_ess(fit))
                divergences.append(sum(fit.divergences()))
                seconds.append(elapsed)
            print(f"{label:28s} {metric:7s} {np.mean(steps):10.1f} {np.mean(ess):8.0f} "
                  f"{np.mean(divergences):11.1f} {np.mean(seconds):8.2f}", flush=True)


def warmup_rows(warmups, seeds):
    print("\n## Warmup length")
    print(f"chains={CHAINS} draws={DRAWS} metric=auto seeds={list(seeds)}; divergences are "
          "per fit (all chains), min ESS the median over seeds")
    print(f"{'model':22s} {'warmup':>6s} {'divergences':>11s} {'steps/draw':>10s} "
          f"{'min ESS':>8s} {'step size':>9s}")
    for label, (model, data) in (("well-scaled regression", well_scaled_regression()),
                                 ("badly scaled regression", badly_scaled_regression())):
        for warmup in warmups:
            divergences, steps, ess, step_sizes = [], [], [], []
            for seed in seeds:
                fit, _ = run(model, data, draws=DRAWS, warmup=warmup, seed=seed)
                divergences.append(sum(fit.divergences()))
                steps.append(steps_per_draw(fit))
                ess.append(min_bulk_ess(fit))
                step_sizes.append(np.mean(fit.step_sizes()))
            print(f"{label:22s} {warmup:6d} {np.mean(divergences):11.1f} "
                  f"{np.mean(steps):10.1f} {np.median(ess):8.0f} "
                  f"{np.mean(step_sizes):9.3f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true",
                        help="one seed and fewer cases, for a smoke run")
    args = parser.parse_args()
    seeds = range(1) if args.quick else range(3)
    dims = (50,) if args.quick else (50, 100, 300)
    print(f"rustmc {rmc.__version__} ({rmc.__file__})")
    print(f"python {platform.python_version()}, numpy {np.__version__}, "
          f"{platform.machine()} {platform.system()}")

    cases = []
    for d in dims:
        cases.append((f"isotropic vector d={d}", isotropic_vector(d), METRICS))
        cases.append((f"isotropic scalars d={d}", isotropic_scalars(d), ("-",)))
    metric_rows("Isotropic regression", cases, seeds)
    metric_rows("Correlated regression (rho = 0.9, n = 300)",
                [(f"correlated vector d={d}", correlated_vector(d), METRICS)
                 for d in ((10,) if args.quick else (10, 20))], seeds)
    warmups = (20, 151, 1000) if args.quick else (20, 25, 100, 149, 150, 151, 155, 500, 1000)
    warmup_rows(warmups, range(1) if args.quick else range(5))


if __name__ == "__main__":
    main()
