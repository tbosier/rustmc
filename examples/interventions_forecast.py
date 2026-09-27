"""Forecast a monthly series through a known contract change.

The synthetic series has a trend, annual seasonality, a promotion effect, a one-month
outage, and a contract change that shifts the level and steepens the trend. The dates
of the outage and the contract change are known. One `TimeDesign` builds every
regression column for the training months and the forecast months, so the two sets of
rows cannot drift apart.

The same model without the intervention columns is scored on the same held-out year,
first on one series and then on twelve simulated series, because one year of held-out
data is a noisy comparison.

Priors here describe this synthetic series; choose them for the units of yours.
"""

import numpy as np

import rustmc as rmc

# 72 months of history and 12 held-out months.
months = np.arange("2019-01", "2026-01", dtype="datetime64[M]")
history, horizon = 72, 12

# Calendar positions of the known events: row i of the training data is position i.
contract = int(np.searchsorted(months, np.datetime64("2023-04")))
outage = int(np.searchsorted(months, np.datetime64("2021-08")))


def simulate(seed):
    rng = np.random.default_rng(seed)
    t = np.arange(len(months))
    promotion = rng.binomial(1, 0.25, len(months)).astype(float)
    y = (
        50.0
        + 0.10 * t
        + 6.0 * (t >= contract)
        + 0.40 * np.maximum(0, t - contract)
        - 8.0 * (t == outage)
        + 2.0 * np.sin(2 * np.pi * t / 12)
        + 1.5 * promotion
        + np.cumsum(rng.normal(0.0, 0.2, len(months)))
        + rng.normal(0.0, 0.8, len(months))
    )
    return y, promotion


D = rmc.DesignTerm
baseline_terms = [
    D.ramp("trend", at=0),
    D.fourier("annual", period=12, harmonics=2),
    D.regressor("promotion"),
]
intervention_terms = [
    D.step("contract_level", at=contract),
    D.ramp("contract_slope", at=contract),
    D.pulse("outage", at=outage),
]
# Prior standard deviation of each term's coefficients, in units of y per unit of
# the column. A ramp column grows by one per month, so its coefficient is per month.
prior_sd = {
    "trend": 0.5, "annual": 5.0, "promotion": 5.0,
    "contract_level": 10.0, "contract_slope": 0.5, "outage": 10.0,
}

model = rmc.BayesianLocalLevel(
    rmc.InverseGammaPrior(3.0, 0.1),
    rmc.InverseGammaPrior(3.0, 1.0),
    initial_mean=50.0,
    initial_variance=25.0,
)


def fit_and_forecast(terms, y, promotion, draws=1000):
    design = rmc.TimeDesign(terms, length=history)
    sd = np.concatenate([
        np.full(len(design.names[design.term_slice(term.name)]), prior_sd[term.name])
        for term in design.terms
    ])
    prior = rmc.GaussianCoefficientPrior(np.zeros(design.width), np.diag(sd**2))
    fit = model.fit(
        y[:history],
        exog=design.training(regressors={"promotion": promotion[:history]}),
        coefficient_prior=prior,
        chains=4, draws=draws, warmup=draws, seed=42,
    )
    # future() continues the training calendar; only the promotion plan is supplied.
    future = design.future(horizon, regressors={"promotion": promotion[history:]})
    return design, fit, fit.forecast(horizon, exog=future, seed=43)


y, promotion = simulate(7)
design, fit, forecast = fit_and_forecast(baseline_terms + intervention_terms, y, promotion)
print("Columns:", design.names)

coefficients = fit.get_samples_2d()["coefficients"].reshape(-1, design.width)
r_hat = {d["name"]: d["r_hat"] for d in fit.diagnostics()}
print("\nIntervention effects, posterior mean and 95% equal-tailed interval (truth):")
for name, truth in (("contract_level", 6.0), ("contract_slope", 0.40), ("outage", -8.0)):
    index = design.names.index(name)
    lower, upper = np.quantile(coefficients[:, index], [0.025, 0.975])
    print(f"  {name:15s} {coefficients[:, index].mean():7.3f}  [{lower:7.3f}, {upper:7.3f}]"
          f"  ({truth})  R-hat {r_hat[f'coefficient[{index}]']:.3f}")
print(f"Largest R-hat over all reported parameters: {max(r_hat.values()):.3f}")

actual = y[history:]
lower, upper = forecast.interval(0.9)
print("\nHeld-out year of this series: 90% interval coverage "
      f"{np.mean((lower <= actual) & (actual <= upper)):.2f}")
_, _, without = fit_and_forecast(baseline_terms, y, promotion)
for label, result in (("with intervention columns", forecast), ("without them", without)):
    print(f"  {label:26s} mean CRPS {rmc.crps(result.observation_samples, actual).mean():6.3f}")

# Shorter runs for the repeated comparison; the fit above shows the mixing.
print("\nMean CRPS on the held-out year of twelve simulated series:")
scores = []
for seed in range(12):
    y, promotion = simulate(seed)
    scores.append([
        rmc.crps(fit_and_forecast(terms, y, promotion, draws=300)[2].observation_samples,
                 y[history:]).mean()
        for terms in (baseline_terms + intervention_terms, baseline_terms)
    ])
scores = np.array(scores)
print(f"  with intervention columns  {scores[:, 0].mean():6.3f}")
print(f"  without them               {scores[:, 1].mean():6.3f}")
print(f"  series where the intervention columns scored better: {(scores[:, 0] < scores[:, 1]).sum()} of 12")
