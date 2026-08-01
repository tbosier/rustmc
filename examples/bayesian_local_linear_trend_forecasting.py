"""Fit a Bayesian local-linear trend and print three 95% forecast intervals.

The observation interval is posterior predictive. The level and slope
intervals are latent-state credible intervals. All are pointwise equal-tailed
intervals and integrate uncertainty in the three fitted variance components.

Run with:
    python examples/bayesian_local_linear_trend_forecasting.py
"""

from __future__ import annotations

import numpy as np
import rustmc as rmc


def simulate_trend(seed: int = 2026, size: int = 100) -> np.ndarray:
    rng = np.random.default_rng(seed)
    level = 0.0
    slope = 0.04
    observations = np.empty(size)
    for time in range(size):
        level += slope + rng.normal(0.0, np.sqrt(0.06))
        slope += rng.normal(0.0, np.sqrt(0.008))
        observations[time] = level + rng.normal(0.0, np.sqrt(0.30))
    observations[[18, 51, 52]] = np.nan
    return observations


def main() -> None:
    observations = simulate_trend()

    # These inverse-gamma priors are on variances. For shape > 1, their means
    # are scale / (shape - 1): 0.06, 0.008, and 0.30 respectively here.
    model = rmc.BayesianLocalLinearTrend(
        level_variance_prior=rmc.InverseGammaPrior(shape=3.0, scale=0.12),
        slope_variance_prior=rmc.InverseGammaPrior(shape=3.0, scale=0.016),
        observation_variance_prior=rmc.InverseGammaPrior(shape=3.0, scale=0.60),
        initial_level=0.0,
        initial_slope=0.0,
        initial_level_variance=4.0,
        initial_slope_variance=0.25,
        initial_level_slope_covariance=0.0,
    )
    fit = model.fit(
        observations,
        chains=4,
        draws=600,
        warmup=400,
        thin=1,
        seed=42,
    )

    horizon = 12
    forecast = fit.forecast(steps=horizon, seed=43)
    observation_lower, observation_upper = forecast.interval(0.95)
    level_lower, level_upper = forecast.level_interval(0.95)
    slope_lower, slope_upper = forecast.slope_interval(0.95)

    assert forecast.observation_samples.shape == (4, 600, horizon)
    assert forecast.level_samples.shape == (4, 600, horizon)
    assert forecast.slope_samples.shape == (4, 600, horizon)

    print("Bayesian local-linear-trend forecast")
    print(f"  observations: {fit.observed_count} observed / {fit.time_count} time steps")
    print(f"  posterior draws: {fit.chains} chains x {fit.draws} draws")
    print(
        "\n  step     observation 95% predictive"
        "          latent level 95% credible       latent slope 95% credible"
    )
    for step in range(horizon):
        print(
            f"  {step + 1:>4}  "
            f"{forecast.observation_mean[step]:>8.3f} "
            f"[{observation_lower[step]:>8.3f}, {observation_upper[step]:>8.3f}]    "
            f"{forecast.level_mean[step]:>8.3f} "
            f"[{level_lower[step]:>8.3f}, {level_upper[step]:>8.3f}]    "
            f"{forecast.slope_mean[step]:>8.3f} "
            f"[{slope_lower[step]:>8.3f}, {slope_upper[step]:>8.3f}]"
        )

    assert forecast.uncertainty_kind == "parameter_integrated_posterior_predictive"
    assert forecast.interval_kind == "pointwise_equal_tailed"


if __name__ == "__main__":
    main()
