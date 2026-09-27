"""Mixing of the local-linear-trend sampler's level, slope and observation variances.

Run: python -m benchmarks.llt_variance_mixing [--quick] [--other-models]

Prints its configuration and seeds, then:

1. for each regime, series length and variance: the worst split R-hat, the smallest
   bulk and tail ESS over the seeds, the median bulk ESS per second, and recovery of
   the true variance (median of the posterior medians, and in how many seeds the
   central 90% interval covers the truth);
2. posterior summaries from long runs on one short series, beside posterior means
   computed by quadrature with the general Kalman filter, for checking that a build
   samples the intended posterior; run the script under two builds to compare them;
3. with ``--other-models``, the same mixing numbers for the forecasting samplers that
   still update their state variances from sampled states: local level, seasonal
   local level, a ``StructuralModel`` trend, and the trend fitted with ``exog``.

To compare builds, run the script once against each: for example against a wheel of
the previous revision on ``PYTHONPATH`` and against this one. Seconds depend on the
machine and the thread count (chains run in parallel) and are indicative only; R-hat,
ESS and the posterior summaries are deterministic for a given build.

The variance priors are the same weakly informative inverse-gamma priors in every
regime. A true variance of 1e-6 is far below where they put mass, so the posterior
for it is pulled up to the prior's scale; that is the prior's doing, and the
recovery columns show it rather than hide it.
"""
from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import time

import numpy as np
import rustmc as rmc

CHAINS = 4
WARMUP = 300
DRAWS = 500
LENGTHS = (200, 1000)
SEEDS = (0, 1, 2)
# True (level, slope, observation) variances.
REGIMES = {
    "balanced": (0.05, 0.005, 0.3),
    "near-zero slope variance": (0.05, 1e-6, 0.3),
    "near-zero level variance": (1e-6, 0.002, 0.3),
    "low signal-to-noise": (0.005, 0.0005, 1.0),
    "near-zero observation variance": (0.05, 0.005, 1e-4),
}
PRIORS = {"level": (2.0, 0.01), "slope": (2.0, 0.001), "observation": (2.0, 0.1)}
NAMES = ("level", "slope", "observation")


def simulate(variances, n, seed):
    """A local linear trend from level 0 and slope 0.02."""
    rng = np.random.default_rng(seed)
    level_sd, slope_sd, observation_sd = np.sqrt(variances)
    level, slope = 0.0, 0.02
    y = np.empty(n)
    for t in range(n):
        level += slope + rng.normal(0.0, level_sd)
        slope += rng.normal(0.0, slope_sd)
        y[t] = level + rng.normal(0.0, observation_sd)
    return y


def prior(name):
    return rmc.InverseGammaPrior(*PRIORS[name])


def trend_model(y):
    return rmc.BayesianLocalLinearTrend(
        level_variance_prior=prior("level"),
        slope_variance_prior=prior("slope"),
        observation_variance_prior=prior("observation"),
        initial_level=float(y[0]),
        initial_slope=0.0,
        initial_level_variance=10.0,
        initial_slope_variance=1.0,
    )


def timed_fit(model, y, seed, **options):
    start = time.perf_counter()
    fit = model.fit(y, chains=CHAINS, warmup=WARMUP, draws=DRAWS, seed=seed, **options)
    return fit, time.perf_counter() - start


def diagnostics_by_name(fit):
    return {row["name"]: row for row in fit.diagnostics()}


def mixing_rows(lengths, seeds):
    print("\n## Local-linear-trend mixing")
    print(f"chains={CHAINS} warmup={WARMUP} draws={DRAWS} seeds={list(seeds)} "
          f"(data seed s, sampler seed 100 + s); priors InverseGamma(shape, scale) "
          f"{PRIORS}; initial level y[0], slope 0, variances 10 and 1")
    print("R-hat is the worst over seeds, ESS the smallest, ESS/s the median; "
          "median is the median over seeds of the posterior median; "
          "cover is how many seeds' central 90% intervals contain the truth")
    print(f"{'regime':31s} {'n':>5s} {'variance':12s} {'R-hat':>6s} {'bulk':>6s} "
          f"{'tail':>6s} {'ESS/s':>8s} {'truth':>9s} {'median':>9s} {'cover':>5s}")
    for regime, truths in REGIMES.items():
        for n in lengths:
            per_seed = []
            for seed in seeds:
                y = simulate(truths, n, seed)
                fit, seconds = timed_fit(trend_model(y), y, 100 + seed)
                per_seed.append((diagnostics_by_name(fit), fit.get_samples_2d(), seconds))
            for name, truth in zip(NAMES, truths):
                key = f"{name}_variance"
                rhat = max(d[key]["r_hat"] for d, _, _ in per_seed)
                bulk = min(d[key]["ess_bulk"] for d, _, _ in per_seed)
                tail = min(d[key]["ess_tail"] for d, _, _ in per_seed)
                rate = np.median([d[key]["ess_bulk"] / s for d, _, s in per_seed])
                medians = [np.median(samples[key]) for _, samples, _ in per_seed]
                cover = sum(
                    np.quantile(samples[key], 0.05) <= truth <= np.quantile(samples[key], 0.95)
                    for _, samples, _ in per_seed
                )
                print(f"{regime:31s} {n:5d} {name:12s} {rhat:6.3f} {bulk:6.0f} {tail:6.0f} "
                      f"{rate:8.0f} {truth:9.2e} {np.median(medians):9.2e} "
                      f"{cover:>2d}/{len(per_seed)}", flush=True)
            seconds = np.median([s for _, _, s in per_seed])
            print(f"{'':31s} {n:5d} {'fit seconds':12s} {seconds:6.3f} (median)")


REFERENCE_PRIORS = ((3.0, 0.1), (3.0, 0.01), (3.0, 2.0))


def reference_series():
    y = simulate((0.3, 0.05, 0.2), 40, 5)
    y[17] = np.nan
    return y


def quadrature_means(y, points=40, lowest=-11.0, highest=2.0):
    """Exact posterior means on a grid in log variance, using the general
    ``LinearGaussianStateSpace`` filter's likelihood, which the trend sampler does
    not use. 40 points a side agree with 80 to about 1e-5 relative."""
    grid = np.linspace(lowest, highest, points)
    log_weights, values = [], []
    for a in grid:
        for b in grid:
            for c in grid:
                variances = np.exp([a, b, c])
                result = rmc.LinearGaussianStateSpace.local_linear_trend(
                    *variances, initial_level=0.0, initial_trend=0.0,
                    initial_level_variance=4.0, initial_trend_variance=1.0,
                ).filter(y)
                # Inverse-gamma densities times the Jacobian of log coordinates.
                log_prior = sum(-shape * np.log(q) - scale / q
                                for (shape, scale), q in zip(REFERENCE_PRIORS, variances))
                log_weights.append(log_prior + result.log_likelihood)
                values.append([*variances, *np.asarray(result.filtered_means)[-1]])
    weights = np.exp(np.array(log_weights) - max(log_weights))
    return dict(zip(REFERENCE_KEYS, weights @ np.array(values) / weights.sum()))


REFERENCE_KEYS = ("level_variance", "slope_variance", "observation_variance",
                  "terminal_level", "terminal_slope")


def reference_rows(draws):
    # A short series on which the old and new samplers both reach a usable ESS, so
    # their posterior summaries can be compared with each other and with quadrature.
    y = reference_series()
    model = rmc.BayesianLocalLinearTrend(
        level_variance_prior=rmc.InverseGammaPrior(*REFERENCE_PRIORS[0]),
        slope_variance_prior=rmc.InverseGammaPrior(*REFERENCE_PRIORS[1]),
        observation_variance_prior=rmc.InverseGammaPrior(*REFERENCE_PRIORS[2]),
        initial_level=0.0,
        initial_slope=0.0,
        initial_level_variance=4.0,
        initial_slope_variance=1.0,
    )
    exact = quadrature_means(y)
    print("\n## Posterior summaries for comparing builds")
    print(f"series: simulate((0.3, 0.05, 0.2), n=40, seed=5) with y[17] missing; priors "
          f"InverseGamma{REFERENCE_PRIORS}; initial state N([0, 0], diag(4, 1)); "
          f"chains={CHAINS} warmup=1000 draws={draws} seeds=(1, 2); z is "
          "(mean - quadrature mean) / MCSE")
    print(f"{'quantity':22s} {'seed':>4s} {'mean':>10s} {'MCSE':>9s} {'quadrature':>10s} "
          f"{'z':>5s} {'sd':>9s} {'q05':>9s} {'q50':>9s} {'q95':>9s} {'bulk ESS':>9s}")
    for seed in (1, 2):
        fit = model.fit(y, chains=CHAINS, warmup=1000, draws=draws, seed=seed)
        rows = diagnostics_by_name(fit)
        samples = fit.get_samples_2d()
        for key in REFERENCE_KEYS:
            values = samples[key]
            mcse = rows[key]["mcse_mean"]
            q05, q50, q95 = np.quantile(values, [0.05, 0.5, 0.95])
            print(f"{key:22s} {seed:4d} {values.mean():10.5f} {mcse:9.5f} {exact[key]:10.5f} "
                  f"{(values.mean() - exact[key]) / mcse:5.1f} {values.std(ddof=1):9.5f} "
                  f"{q05:9.5f} {q50:9.5f} {q95:9.5f} {rows[key]['ess_bulk']:9.0f}",
                  flush=True)


def other_model_fits(y, seed):
    """Fits of the samplers this change leaves alone, with their state-variance names."""
    level, slope, observation = (prior(name) for name in NAMES)
    local_level = rmc.BayesianLocalLevel(
        process_variance_prior=level,
        observation_variance_prior=observation,
        initial_mean=float(y[0]),
        initial_variance=10.0,
    )
    yield "local level", timed_fit(local_level, y, seed)[0], ("process_variance",)
    seasonal = rmc.BayesianSeasonalLocalLevel(
        period=4,
        level_variance_prior=level,
        seasonal_variance_prior=slope,
        observation_variance_prior=observation,
        initial_level=float(y[0]),
        initial_level_variance=10.0,
    )
    yield ("seasonal local level (period 4)", timed_fit(seasonal, y, seed)[0],
           ("level_variance", "seasonal_variance"))
    structural = rmc.StructuralModel(
        [rmc.StructuralComponent.trend(
            "trend", rmc.VarianceParameter.inverse_gamma(*PRIORS["level"]),
            rmc.VarianceParameter.inverse_gamma(*PRIORS["slope"]),
            initial_mean=[float(y[0]), 0.0], initial_covariance=[[10.0, 0.0], [0.0, 1.0]])],
        observation_variance=rmc.VarianceParameter.inverse_gamma(*PRIORS["observation"]),
    )
    fit = structural.fit(y, chains=CHAINS, warmup=WARMUP, draws=DRAWS, seed=seed)
    names = tuple(row["name"] for row in fit.diagnostics()[:2])
    yield "StructuralModel trend", fit, names
    exog = np.linspace(-1.0, 1.0, y.size)[:, None]
    fit, _ = timed_fit(trend_model(y), y, seed, exog=exog,
                       coefficient_prior=rmc.GaussianCoefficientPrior([0.0], [[1.0]]))
    yield "trend with exog", fit, ("level_variance", "slope_variance")


def other_model_rows(seeds):
    print("\n## Samplers not changed here: state variances drawn given sampled states")
    print(f"n=200, chains={CHAINS} warmup={WARMUP} draws={DRAWS} seeds={list(seeds)}; "
          "the same series and priors as above (the seasonal variance takes the slope "
          "prior); worst R-hat and smallest bulk ESS over seeds")
    print(f"{'regime':31s} {'model':32s} {'variance':22s} {'R-hat':>6s} {'bulk':>6s}")
    for regime in ("balanced", "near-zero level variance"):
        results = {}
        for seed in seeds:
            y = simulate(REGIMES[regime], 200, seed)
            for label, fit, names in other_model_fits(y, 100 + seed):
                rows = diagnostics_by_name(fit)
                for name in names:
                    results.setdefault((label, name), []).append(rows[name])
        for (label, name), rows in results.items():
            print(f"{regime:31s} {label:32s} {name:22s} "
                  f"{max(r['r_hat'] for r in rows):6.3f} "
                  f"{min(r['ess_bulk'] for r in rows):6.0f}", flush=True)


def checkout_revision():
    """The git revision of the checkout holding this script, marked dirty if it
    has uncommitted changes. The rustmc build measured is the path printed above,
    which may come from another revision."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        revision = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"],
                                  capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", root, "status", "--porcelain"],
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return revision + (" (with uncommitted changes)" if dirty else "")


def cpu_model():
    try:
        with open("/proc/cpuinfo") as cpuinfo:
            for line in cpuinfo:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true",
                        help="one seed, n=200 only, and a shorter reference run")
    parser.add_argument("--other-models", action="store_true",
                        help="also measure the samplers this change leaves alone")
    args = parser.parse_args()
    seeds = SEEDS[:1] if args.quick else SEEDS
    print(f"rustmc {rmc.__version__} ({rmc.__file__})")
    print(f"script checkout revision {checkout_revision()}")
    print(f"python {platform.python_version()}, numpy {np.__version__}, "
          f"{platform.machine()} {platform.system()}, {cpu_model()}, "
          f"{os.cpu_count()} logical CPUs, RAYON_NUM_THREADS="
          f"{os.environ.get('RAYON_NUM_THREADS', 'unset (one thread per logical CPU)')}")
    print(f"command: python -m benchmarks.llt_variance_mixing {' '.join(sys.argv[1:])}")
    mixing_rows(LENGTHS[:1] if args.quick else LENGTHS, seeds)
    reference_rows(2_000 if args.quick else 25_000)
    if args.other_models:
        other_model_rows(seeds)


if __name__ == "__main__":
    main()
