"""Fit a Bayesian seasonal local-level model and summarize coherent forecasts.

The example uses synthetic monthly data so the seasonal structure is explicit. Priors
must be chosen for the scale and frequency of the real series being modeled.
"""

import numpy as np

import rustmc as rmc


rng = np.random.default_rng(42)
period = 12
seasonal_pattern = np.array(
    [-8.0, -5.0, -2.0, 1.0, 4.0, 7.0, 10.0, 8.0, 4.0, 0.0, -4.0, -5.0]
)
seasonal_pattern -= seasonal_pattern.mean()
assert np.isclose(seasonal_pattern.sum(), 0.0)

level = 100.0
observations = []
for month in range(48):
    level += rng.normal(0.0, 1.0)
    observations.append(
        level + seasonal_pattern[month % period] + rng.normal(0.0, 2.0)
    )
observations = np.asarray(observations)

model = rmc.BayesianSeasonalLocalLevel(
    period=period,
    level_variance_prior=rmc.InverseGammaPrior(3.0, 2.0),
    seasonal_variance_prior=rmc.InverseGammaPrior(3.0, 1.0),
    observation_variance_prior=rmc.InverseGammaPrior(3.0, 8.0),
    initial_level=float(observations[:period].mean()),
    initial_seasonal_effects=np.zeros(period),
    initial_level_variance=25.0,
    initial_seasonal_variance=9.0,
)
fit = model.fit(observations, chains=4, warmup=500, draws=1_000, seed=42)
forecast = fit.forecast(steps=12, seed=43)

lower, upper = forecast.interval(0.95)
cumulative_lower, cumulative_upper = forecast.cumulative_interval(0.95)

print("Future observation posterior-predictive means:")
print(forecast.observation_mean)
print("Future observation 95% posterior-predictive interval:")
print(np.column_stack([lower, upper]))
print("Cumulative 3/6/12-period posterior-predictive summaries:")
for horizon in (3, 6, 12):
    index = horizon - 1
    print(
        horizon,
        forecast.cumulative_observation_mean[index],
        cumulative_lower[index],
        cumulative_upper[index],
    )

# fit.to_arviz() exposes variance and terminal-state draws for convergence checks.
