"""How much to reserve when falling short costs more than over-reserving.

A monthly rebate liability is forecast with a Bayesian local-level model on the log
scale. Next month's forecast is about $7.5M with a 95% interval of roughly $5M to $10M.
If each dollar under-reserved costs five times what a dollar over-reserved costs, the
reserve with the lowest expected cost is the 5 / (5 + 1) = 83rd percentile of the
forecast, not its mean or its median.

The history and the costs are illustrative, and the model is nonseasonal and has no
business drivers. A decision is only as good as the forecast behind it: check the
forecast on rolling origins before using a decision like this in a financial process.

Run with:
    python examples/reserve_decision.py
"""

from __future__ import annotations

import numpy as np
import rustmc as rmc

# Illustrative monthly rebate liabilities, in millions of dollars.
HISTORY = np.array(
    [
        5.07, 7.27, 6.74, 5.82, 6.97, 6.01, 7.62, 5.85, 6.01, 5.44, 5.82, 7.15,
        6.67, 8.81, 7.67, 5.28, 5.31, 8.72, 8.25, 7.91, 10.08, 7.87, 6.36, 6.14,
        8.82, 7.43, 7.84, 7.69, 6.86, 6.21, 7.10, 8.09, 7.07, 7.23, 7.42, 7.69,
    ]
)


def main() -> None:
    # A local level on the log scale keeps forecast amounts positive.
    model = rmc.BayesianLocalLevel(
        process_variance_prior=rmc.InverseGammaPrior(shape=3.0, scale=0.004),
        observation_variance_prior=rmc.InverseGammaPrior(shape=3.0, scale=0.04),
        initial_mean=float(np.log(HISTORY[0])),
        initial_variance=1.0,
    )
    fit = model.fit(np.log(HISTORY), chains=4, draws=2000, warmup=1000, seed=7)
    diagnostics = fit.diagnostics()
    print(
        "Fit: smallest bulk ESS "
        f"{min(row['ess_bulk'] for row in diagnostics):.0f}, "
        f"largest R-hat {max(row['r_hat'] for row in diagnostics):.3f}"
    )

    # Joint paths for the next three months, in millions of dollars. Each path
    # keeps its months together, so totals over months are computed per path.
    forecast = rmc.ForecastDraws(np.exp(fit.forecast(steps=3, seed=8).observation_samples))
    next_month = forecast.observation_samples[..., 0]
    lower, upper = forecast.interval(0.95)
    print(
        f"\nNext month: mean ${next_month.mean():.2f}M, "
        f"95% equal-tailed interval ${lower[0]:.2f}M to ${upper[0]:.2f}M"
    )

    # Under-reserving costs 5 per dollar, over-reserving 1 per dollar.
    loss = rmc.AsymmetricLinearLoss(under=5, over=1)
    reserve = forecast.optimal_decision(loss, steps=[0])
    print(
        f"Reserve ${reserve.decision:.2f}M: the {reserve.quantile:.3f} quantile, "
        f"Monte Carlo standard error ${reserve.decision_mcse:.3f}M"
    )
    print(
        "  the same rule on each half of every chain gives "
        f"${reserve.batch_decisions.min():.2f}M to ${reserve.batch_decisions.max():.2f}M"
    )

    print("\nExpected cost of other reserves, in the same units as the costs:")
    for label, amount in [
        ("the mean", float(next_month.mean())),
        ("the median", float(np.median(next_month))),
        ("the optimum", reserve.decision),
        ("the 97.5% upper bound", float(upper[0])),
    ]:
        cost = forecast.expected_loss(amount, loss, steps=[0])
        print(f"  {label:>22} ${amount:5.2f}M: {cost.value:.3f} (MCSE {cost.mcse:.3f})")

    # A quarter's reserve is decided on the total of each path, not by adding up
    # three monthly reserves.
    quarter = forecast.optimal_decision(loss, steps=range(3))
    monthly = sum(forecast.optimal_decision(loss, steps=[step]).decision for step in range(3))
    print(
        f"\nThree-month reserve ${quarter.decision:.2f}M "
        f"(MCSE ${quarter.decision_mcse:.3f}M); "
        f"the sum of three monthly reserves is ${monthly:.2f}M"
    )
    quarter_cost = forecast.expected_loss(quarter.decision, loss, steps=range(3)).value
    summed_cost = forecast.expected_loss(monthly, loss, steps=range(3)).value
    print(f"  expected cost {quarter_cost:.3f} against {summed_cost:.3f} for the sum")

    # Falling short by more than $0.5M costs 10 per dollar beyond it; this has
    # no closed-form answer, so it is searched on a grid of candidate reserves.
    tiered = rmc.PiecewiseLinearLoss(shortfall=[-1.0, 0.0, 0.5, 1.5], loss=[1.0, 0.0, 2.5, 12.5])
    grid = np.arange(7.0, 12.0, 0.01)
    tiered_reserve = forecast.optimal_decision(tiered, steps=[0], candidates=grid)
    print(
        f"\nTiered shortfall cost: reserve ${tiered_reserve.decision:.2f}M "
        f"(grid spacing $0.01M, on the grid's edge: {tiered_reserve.on_grid_edge})"
    )

    # Any loss can be written as a function of the decision and the draws.
    # Here a shortfall of more than $1M also carries a fixed cost of 3.
    def with_penalty(decision: float, outcomes: np.ndarray) -> np.ndarray:
        shortfall = outcomes - decision
        linear = np.where(shortfall > 0, 5 * shortfall, -shortfall)
        return linear + 3.0 * (shortfall > 1.0)

    penalised = forecast.optimal_decision(with_penalty, steps=[0], candidates=grid)
    print(
        f"With a fixed penalty for a shortfall over $1M: reserve ${penalised.decision:.2f}M, "
        f"expected cost {penalised.expected_loss:.3f} (MCSE {penalised.expected_loss_mcse:.3f})"
    )


if __name__ == "__main__":
    main()
