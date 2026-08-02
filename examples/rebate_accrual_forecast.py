"""A cautious 24-month rebate-accrual forecasting baseline.

This example demonstrates uncertainty handling, not a production-ready rebate model.
The fitted local-level model is nonseasonal and has no business regressors. Replace the
illustrative values and priors, and validate every forecast horizon with rolling origins
before using the output in a financial process.

Run with:
    python examples/rebate_accrual_forecast.py
"""

from __future__ import annotations

import numpy as np
import rustmc as rmc


# Illustrative positive monthly accruals in thousands of currency units. A log scale
# prevents the Gaussian local-level model from generating negative dollar forecasts.
ACCRUAL_HISTORY = np.asarray(
    [
        101,
        98,
        103,
        105,
        102,
        108,
        111,
        109,
        114,
        116,
        113,
        119,
        121,
        118,
        123,
        126,
        124,
        129,
        131,
        128,
        134,
        136,
        133,
        139,
    ],
    dtype=np.float64,
)


def interval(draws: np.ndarray, probability: float = 0.95) -> tuple[float, float]:
    """Return an equal-tailed interval from posterior or predictive draws."""

    tail = (1.0 - probability) / 2.0
    lower, upper = np.quantile(draws, [tail, 1.0 - tail])
    return float(lower), float(upper)


def main() -> None:
    log_history = np.log(ACCRUAL_HISTORY)

    # Priors are on log-scale variances. Their means are scale / (shape - 1):
    # 0.0025 for process variance and 0.01 for observation variance here.
    model = rmc.BayesianLocalLevel(
        process_variance_prior=rmc.InverseGammaPrior(shape=3.0, scale=0.005),
        observation_variance_prior=rmc.InverseGammaPrior(shape=3.0, scale=0.020),
        initial_mean=float(log_history[0]),
        initial_variance=0.25,
    )
    fit = model.fit(log_history, chains=4, warmup=500, draws=1_000, seed=42)
    forecast = fit.forecast(steps=12, seed=43)

    # Back-transform every coherent path before summarizing. exp(mean(log Y)) is not
    # generally equal to mean(Y), so transforming a summary would be wrong.
    payment_draws = np.exp(forecast.observation_samples)
    expected_level_draws = np.exp(forecast.state_samples)

    print("Monthly forecast (thousands)")
    print("month   payment mean and 95% predictive     expected level and 95% credible")
    for month in range(12):
        payment = payment_draws[:, :, month]
        expected_level = expected_level_draws[:, :, month]
        payment_low, payment_high = interval(payment)
        level_low, level_high = interval(expected_level)
        print(
            f"{month + 1:>5}   "
            f"{payment.mean():>8.2f} [{payment_low:>8.2f}, {payment_high:>8.2f}]   "
            f"{expected_level.mean():>8.2f} [{level_low:>8.2f}, {level_high:>8.2f}]"
        )

    print("\nCumulative payment forecast (thousands)")
    for horizon in (3, 6, 12):
        # Sum inside each posterior-predictive path. Summing monthly endpoints would
        # discard temporal dependence and does not produce an interval for the total.
        total_draws = payment_draws[:, :, :horizon].sum(axis=2)
        lower, upper = interval(total_draws)
        print(
            f"{horizon:>2} months: mean {total_draws.mean():>9.2f}, "
            f"95% predictive [{lower:>9.2f}, {upper:>9.2f}]"
        )

    print(
        "\nThis is a nonseasonal baseline. Compare it with seasonal naive/ETS and the "
        "current finance method using rolling 3-, 6-, and 12-month backtests."
    )


if __name__ == "__main__":
    main()
