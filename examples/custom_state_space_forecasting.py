"""Build and forecast a custom multivariate latent state-space model.

The latent state contains a stochastic level and a damped cycle.  This shows how
the generic constructor extends beyond the local-level, local-trend, and AR(1)
convenience constructors while retaining the same filter/smooth/forecast API.

All matrices are fixed here, so forecast intervals remain conditional on them.

Run with:
    python examples/custom_state_space_forecasting.py
"""

from __future__ import annotations

import numpy as np
import rustmc as rmc


def main() -> None:
    period = 12.0
    damping = 0.96
    angle = 2.0 * np.pi / period

    # State = [level, cycle_cosine, cycle_sine]. The two-dimensional rotation
    # advances the seasonal cycle by one time step and damping keeps it stable.
    transition = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, damping * np.cos(angle), -damping * np.sin(angle)],
            [0.0, damping * np.sin(angle), damping * np.cos(angle)],
        ]
    )
    observation = np.array([1.0, 1.0, 0.0])
    process_covariance = np.diag([0.03, 0.01, 0.01])
    observation_variance = 0.16
    initial_mean = np.array([5.0, 1.2, 0.0])
    initial_covariance = np.diag([1.0, 0.5, 0.5])

    model = rmc.LinearGaussianStateSpace(
        transition=transition,
        observation=observation,
        process_covariance=process_covariance,
        observation_variance=observation_variance,
        initial_mean=initial_mean,
        initial_covariance=initial_covariance,
    )

    rng = np.random.default_rng(2026)
    time = np.arange(48, dtype=float)
    observations = 5.0 + 1.2 * damping**time * np.cos(angle * time)
    observations += rng.normal(0.0, np.sqrt(observation_variance), time.size)
    observations[20] = np.nan

    smoothed = model.smooth(observations)
    forecast = model.forecast(observations, steps=12)
    lower, upper = forecast.interval(0.95)

    assert model.dimension == 3
    assert smoothed.smoothed_means.shape == (observations.size, 3)
    assert forecast.state_means.shape == (12, 3)
    assert forecast.state_covariances.shape == (12, 3, 3)

    print("Custom level + damped-cycle forecast")
    print(f"  state dimension: {model.dimension}")
    print(f"  observed-data log likelihood: {smoothed.log_likelihood:.2f}")
    print("\n  step      mean        95% conditional predictive interval")
    for step, (mean, lo, hi) in enumerate(
        zip(forecast.observation_means, lower, upper), start=1
    ):
        print(f"  {step:>4}  {mean:>8.3f}       [{lo:>8.3f}, {hi:>8.3f}]")


if __name__ == "__main__":
    main()
