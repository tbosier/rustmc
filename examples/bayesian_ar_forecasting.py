"""Fit a Bayesian AR(3) and print a deterministic 95% forecast.

The interval for future observations is a pointwise, equal-tailed
posterior-predictive interval. It integrates coefficient, innovation-variance,
and recursively propagated future-innovation uncertainty.

Run with:
    python examples/bayesian_ar_forecasting.py
"""

from __future__ import annotations

import numpy as np
import rustmc as rmc


def simulate_ar3(seed: int = 2026, size: int = 240) -> np.ndarray:
    rng = np.random.default_rng(seed)
    coefficients = np.array([0.35, 0.55, -0.25, 0.12])
    values = np.full(size + 103, 0.5)
    for time in range(3, values.size):
        values[time] = (
            coefficients[0]
            + coefficients[1] * values[time - 1]
            + coefficients[2] * values[time - 2]
            + coefficients[3] * values[time - 3]
            + rng.normal(0.0, 0.30)
        )
    return values[103:]


def main() -> None:
    observations = simulate_ar3()
    order = 3

    # Coefficients are [intercept, lag_1, lag_2, lag_3]. The precision
    # controls the Gaussian prior conditional on the innovation variance.
    prior = rmc.NormalInverseGammaPrior(
        coefficient_mean=np.zeros(order + 1),
        coefficient_precision=np.eye(order + 1) * 0.05,
        variance_shape=2.5,
        variance_scale=0.2,
    )
    model = rmc.BayesianAR(order=order, prior=prior)
    fit = model.fit(observations, chains=4, draws=1_000, seed=42)
    forecast = fit.forecast(steps=12, seed=43)
    lower, upper = forecast.interval(0.95)

    coefficient_mean = fit.get_samples()["coefficient"].mean(axis=(0, 1))
    print("Bayesian AR(3) posterior")
    print("  coefficient order: intercept, lag_1, lag_2, lag_3")
    print("  posterior mean:   ", np.round(coefficient_mean, 3))
    print(f"  posterior draws:   {fit.chains} chains x {fit.draws} draws")
    print("\n  step      mean       95% posterior-predictive interval")
    for step, (mean, low, high) in enumerate(
        zip(forecast.observation_mean, lower, upper), start=1
    ):
        print(f"  {step:>4}  {mean:>8.3f}      [{low:>8.3f}, {high:>8.3f}]")

    assert forecast.observation_samples.shape == (4, 1_000, 12)
    assert forecast.uncertainty_kind == "parameter_integrated_posterior_predictive"
    assert forecast.interval_kind == "pointwise_equal_tailed"


if __name__ == "__main__":
    main()
