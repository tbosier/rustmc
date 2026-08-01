"""Forecast with rustmc's fixed-parameter state-space convenience models.

This example compares the local-level, local-linear-trend, and stationary AR(1)
constructors on deterministic synthetic series.  Their 95% intervals condition on
the supplied system parameters; they do not include parameter uncertainty.

Run with:
    python examples/state_space_forecasting.py
"""

from __future__ import annotations

import numpy as np
import rustmc as rmc

HORIZON = 8


def print_forecast(name: str, model, observations: np.ndarray) -> None:
    """Filter a series, forecast it, and print its pointwise 95% interval."""
    filtered = model.filter(observations)
    forecast = model.forecast(observations, steps=HORIZON)
    lower, upper = forecast.interval(0.95)

    assert forecast.observation_means.shape == (HORIZON,)
    assert forecast.observation_variances.shape == (HORIZON,)
    assert np.all(lower <= forecast.observation_means)
    assert np.all(forecast.observation_means <= upper)
    assert forecast.uncertainty_kind == "conditional_fixed_parameters"

    print(f"\n{name}")
    print(f"  observed-data log likelihood: {filtered.log_likelihood:.2f}")
    print("  step       mean        95% predictive interval")
    for step, (mean, lo, hi) in enumerate(
        zip(forecast.observation_means, lower, upper), start=1
    ):
        print(f"  {step:>4}  {mean:>9.3f}      [{lo:>8.3f}, {hi:>8.3f}]")


def main() -> None:
    rng = np.random.default_rng(2026)
    time = np.arange(60, dtype=float)

    # A drifting level. NaN is a genuine missing time step: the filter performs
    # prediction without an observation update and retains the time spacing.
    local_level_y = np.cumsum(rng.normal(0.0, np.sqrt(0.08), time.size))
    local_level_y += rng.normal(0.0, np.sqrt(0.30), time.size)
    local_level_y[[12, 31]] = np.nan
    local_level = rmc.LinearGaussianStateSpace.local_level(
        process_variance=0.08,
        observation_variance=0.30,
        initial_mean=0.0,
        initial_variance=2.0,
    )

    # A level with a persistent stochastic slope.
    trend_y = 4.0 + 0.18 * time + rng.normal(0.0, np.sqrt(0.25), time.size)
    local_trend = rmc.LinearGaussianStateSpace.local_linear_trend(
        level_variance=0.03,
        trend_variance=0.002,
        observation_variance=0.25,
        initial_level=4.0,
        initial_trend=0.18,
        initial_level_variance=1.0,
        initial_trend_variance=0.05,
    )

    # A stationary, mean-reverting latent AR(1) observed with measurement noise.
    coefficient = 0.82
    ar1_state = np.empty(time.size)
    ar1_state[0] = rng.normal(0.0, np.sqrt(0.16 / (1.0 - coefficient**2)))
    for index in range(1, time.size):
        ar1_state[index] = coefficient * ar1_state[index - 1] + rng.normal(0.0, 0.4)
    ar1_y = ar1_state + rng.normal(0.0, 0.3, time.size)
    ar1 = rmc.LinearGaussianStateSpace.stationary_ar1(
        coefficient=coefficient,
        process_variance=0.16,
        observation_variance=0.09,
    )

    print_forecast("Local level", local_level, local_level_y)
    print_forecast("Local linear trend", local_trend, trend_y)
    print_forecast("Stationary AR(1)", ar1, ar1_y)


if __name__ == "__main__":
    main()
