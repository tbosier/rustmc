"""Worst split R-hat of the state variances of every Gibbs/FFBS forecasting sampler.

Same protocol as benchmarks/llt_variance_mixing.py (commit e9039d8): five regimes,
series of 200 and 1000 points, three data seeds (sampler seed 100 + s), 4 chains,
300 warmup, 500 draws, the same inverse-gamma priors. Each model is fitted to data
simulated from its own process. For every model, setting (regime, length) and
variance it prints the worst R-hat over the seeds, then each variance's range of
those worst values over the ten settings.

Run: python -m benchmarks.state_variance_mixing  (with the build to measure on
PYTHONPATH). A full run takes about two minutes on 4 threads.
"""
import sys

import numpy as np
import rustmc as rmc

CHAINS, WARMUP, DRAWS = 4, 300, 500
LENGTHS = (200, 1000)
SEEDS = (0, 1, 2)
PERIOD = 4
# (first state variance, second state variance, observation variance)
REGIMES = {
    "balanced": (0.05, 0.005, 0.3),
    "near-zero second variance": (0.05, 1e-6, 0.3),
    "near-zero first variance": (1e-6, 0.002, 0.3),
    "low signal-to-noise": (0.005, 0.0005, 1.0),
    "near-zero observation variance": (0.05, 0.005, 1e-4),
}
PRIORS = {"level": (2.0, 0.01), "slope": (2.0, 0.001), "observation": (2.0, 0.1)}


def ig(name):
    return rmc.InverseGammaPrior(*PRIORS[name])


def vp(name):
    return rmc.VarianceParameter.inverse_gamma(*PRIORS[name])


def simulate_trend(v, n, rng, damping=1.0):
    sd = np.sqrt(v)
    level, slope, y = 0.0, 0.02, np.empty(n)
    for t in range(n):
        level += damping * slope + rng.normal(0.0, sd[0])
        slope = damping * slope + rng.normal(0.0, sd[1])
        y[t] = level + rng.normal(0.0, sd[2])
    return y


def simulate_level(v, n, rng):
    sd = np.sqrt(v)
    level, y = 0.0, np.empty(n)
    for t in range(n):
        level += rng.normal(0.0, sd[0])
        y[t] = level + rng.normal(0.0, sd[2])
    return y


def simulate_seasonal(v, n, rng):
    """Local level plus a dummy seasonal of period 4 whose effects sum to zero."""
    sd = np.sqrt(v)
    level, effects, y = 0.0, [0.5, -0.2, -0.4], np.empty(n)
    for t in range(n):
        level += rng.normal(0.0, sd[0])
        new = -sum(effects) + rng.normal(0.0, sd[1])
        effects = [new] + effects[:-1]
        y[t] = level + new + rng.normal(0.0, sd[2])
    return y


def exog(n):
    return dict(exog=np.linspace(-1.0, 1.0, n)[:, None],
                coefficient_prior=rmc.GaussianCoefficientPrior([0.0], [[1.0]]))


def structural_trend(y, damping):
    return rmc.StructuralModel(
        [rmc.StructuralComponent.trend(
            "trend", vp("level"), vp("slope"), initial_mean=[float(y[0]), 0.0],
            initial_covariance=[[10.0, 0.0], [0.0, 1.0]], damping=damping)],
        observation_variance=vp("observation"))


def llt(y):
    return rmc.BayesianLocalLinearTrend(
        level_variance_prior=ig("level"), slope_variance_prior=ig("slope"),
        observation_variance_prior=ig("observation"), initial_level=float(y[0]),
        initial_slope=0.0, initial_level_variance=10.0, initial_slope_variance=1.0)


def seasonal(y):
    return rmc.BayesianSeasonalLocalLevel(
        period=PERIOD, level_variance_prior=ig("level"), seasonal_variance_prior=ig("slope"),
        observation_variance_prior=ig("observation"), initial_level=float(y[0]),
        initial_level_variance=10.0)


def local_level(y):
    return rmc.BayesianLocalLevel(
        process_variance_prior=ig("level"), observation_variance_prior=ig("observation"),
        initial_mean=float(y[0]), initial_variance=10.0)


S = dict(chains=CHAINS, warmup=WARMUP, draws=DRAWS)
# name: (simulate(v, n, rng), fit(y, seed), state-variance names)
MODELS = {
    "BayesianLocalLinearTrend": (simulate_trend, lambda y, s: llt(y).fit(y, seed=s, **S),
                                 ("level_variance", "slope_variance")),
    "BayesianLocalLinearTrend with exog": (
        simulate_trend, lambda y, s: llt(y).fit(y, seed=s, **S, **exog(y.size)),
        ("level_variance", "slope_variance")),
    "StructuralModel trend": (
        simulate_trend, lambda y, s: structural_trend(y, 1.0).fit(y, seed=s, **S),
        ("trend.innovation[0]", "trend.innovation[1]")),
    "StructuralModel trend, damping 0.9": (
        lambda v, n, rng: simulate_trend(v, n, rng, 0.9),
        lambda y, s: structural_trend(y, 0.9).fit(y, seed=s, **S),
        ("trend.innovation[0]", "trend.innovation[1]")),
    "BayesianSeasonalLocalLevel": (
        simulate_seasonal, lambda y, s: seasonal(y).fit(y, seed=s, **S),
        ("level_variance", "seasonal_variance")),
    "BayesianSeasonalLocalLevel with exog": (
        simulate_seasonal, lambda y, s: seasonal(y).fit(y, seed=s, **S, **exog(y.size)),
        ("level_variance", "seasonal_variance")),
    "BayesianLocalLevel": (simulate_level, lambda y, s: local_level(y).fit(y, seed=s, **S),
                           ("process_variance",)),
}


def main():
    print(f"rustmc {rmc.__version__} ({rmc.__file__})")
    print(f"chains={CHAINS} warmup={WARMUP} draws={DRAWS} lengths={LENGTHS} seeds={SEEDS} "
          f"(sampler seed 100 + s) priors={PRIORS}")
    summary = {}
    for model, (simulate, fit, names) in MODELS.items():
        for regime, truths in REGIMES.items():
            for n in LENGTHS:
                worst = {name: 0.0 for name in names}
                for seed in SEEDS:
                    y = simulate(np.array(truths), n, np.random.default_rng(seed))
                    rows = {row["name"]: row for row in fit(y, 100 + seed).diagnostics()}
                    for name in names:
                        worst[name] = max(worst[name], rows[name]["r_hat"])
                for name in names:
                    summary.setdefault((model, name), []).append(worst[name])
                    print(f"{model:38s} {regime:31s} {n:5d} {name:20s} {worst[name]:6.3f}",
                          flush=True)
    print("\nrange over the ten settings of the worst R-hat over seeds")
    for (model, name), values in summary.items():
        print(f"{model:38s} {name:20s} {min(values):6.3f} to {max(values):6.3f}")


if __name__ == "__main__":
    sys.exit(main())
