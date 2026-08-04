"""Joint partial pooling and coherent rollups for ragged program series."""

import numpy as np
import rustmc as rmc


series = [
    np.array([95.0]),
    np.array([101.0, 99.0, 103.0]),
    np.array([198.0, 205.0, 202.0, 207.0, 204.0]),
]

model = rmc.BayesianHierarchicalMean(
    group_variance_prior=rmc.InverseGammaPrior(3.0, 100.0),
    program_variance_prior=rmc.InverseGammaPrior(3.0, 50.0),
    observation_variance_prior=rmc.InverseGammaPrior(3.0, 25.0),
    population_mean_prior=125.0,
    population_variance_prior=2_500.0,
)
fit = model.fit(
    series,
    group_index=[0, 0, 1],
    program_names=["program-a", "program-b", "program-c"],
    group_names=["division-east", "division-west"],
    chains=4,
    warmup=500,
    draws=1_000,
    seed=42,
)
forecast = fit.forecast(steps=6, seed=43)

# Keep shared chain/draw indices intact before reducing.
company_draws = forecast.total_observation_samples
division_draws = forecast.group_observation_samples
company_interval = np.quantile(company_draws, [0.025, 0.975], axis=(0, 1))

print("program posterior means:", fit.get_samples()["program_mean"].mean(axis=(0, 1)))
print("company forecast mean:", company_draws.mean(axis=(0, 1)))
print("company 95% interval:\n", company_interval)
print("division draw shape:", division_draws.shape)
