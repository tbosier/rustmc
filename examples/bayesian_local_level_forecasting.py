"""Fit a Bayesian local-level model and produce two kinds of 95% intervals.

The observation interval is posterior predictive: it includes variance-parameter,
terminal-state, future-process, and future-observation uncertainty.  The state
interval is a credible interval for the unobserved latent level.  Both are
pointwise equal-tailed intervals, not simultaneous bands for an entire path.

ArviZ diagnostics are printed when the optional ``arviz`` package is installed.

Run with:
    python examples/bayesian_local_level_forecasting.py
"""

from __future__ import annotations

import numpy as np
import rustmc as rmc


def simulate_local_level(seed: int = 2026, size: int = 80) -> np.ndarray:
    rng = np.random.default_rng(seed)
    level = np.cumsum(rng.normal(0.0, np.sqrt(0.08), size))
    observations = level + rng.normal(0.0, np.sqrt(0.40), size)
    observations[[14, 39, 40]] = np.nan
    return observations


def print_optional_arviz_diagnostics(fit) -> None:
    try:
        import arviz as az
    except ImportError:
        print(
            "\nArviZ not installed; run `pip install arviz` for R-hat/ESS diagnostics."
        )
        return

    summary = az.summary(
        fit.to_arviz(),
        var_names=["process_sd", "observation_sd"],
        kind="diagnostics",
        round_to=3,
    )
    print("\nArviZ convergence diagnostics (inspect these on real analyses):")
    print(summary)


def main() -> None:
    observations = simulate_local_level()

    # InverseGammaPrior is a prior on a variance, not a standard deviation.
    # With shape > 1 its mean is scale / (shape - 1), so these prior means are
    # 0.10 for the process variance and 0.50 for the observation variance.
    model = rmc.BayesianLocalLevel(
        process_variance_prior=rmc.InverseGammaPrior(shape=3.0, scale=0.20),
        observation_variance_prior=rmc.InverseGammaPrior(shape=3.0, scale=1.00),
        initial_mean=0.0,
        initial_variance=4.0,
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
    predictive_lower, predictive_upper = forecast.interval(0.95)
    state_lower, state_upper = forecast.state_interval(0.95)

    assert forecast.observation_samples.shape == (4, 600, horizon)
    assert forecast.state_samples.shape == (4, 600, horizon)
    assert forecast.uncertainty_kind == "parameter_integrated_posterior_predictive"
    assert forecast.interval_kind == "pointwise_equal_tailed"

    print("Bayesian local-level forecast")
    print(
        f"  observations: {fit.observed_count} observed / {fit.time_count} time steps"
    )
    print(f"  posterior draws: {fit.chains} chains x {fit.draws} draws")
    print(
        "\n  step      observation forecast            latent-level credible interval"
    )
    for step in range(horizon):
        print(
            f"  {step + 1:>4}  "
            f"{forecast.observation_mean[step]:>8.3f} "
            f"[{predictive_lower[step]:>8.3f}, {predictive_upper[step]:>8.3f}]    "
            f"[{state_lower[step]:>8.3f}, {state_upper[step]:>8.3f}]"
        )

    print_optional_arviz_diagnostics(fit)


if __name__ == "__main__":
    main()
