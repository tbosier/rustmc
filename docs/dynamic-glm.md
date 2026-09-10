# Dynamic count, intermittent amount, and pooled regression forecasts

`BayesianDynamicGLM` fits a panel with a shared uncertain regression population,
group deviations, and group random walks. Optional common random-walk innovations
represent shocks affecting all groups. Named constructors select the observation
family: `BayesianDynamicPoisson`, `BayesianDynamicNegativeBinomial`,
`BayesianDynamicHurdleLogNormal`, and `BayesianHierarchicalDynamicRegression`
(Gaussian observations). They return the same `BayesianDynamicGLM` class.

```python
import numpy as np
import rustmc

# Every array includes a group axis, even for one series.
y = [[0., 2., 1., np.nan, 4.], [1., 0., 2., 3., 2.]]
exposure = [[1., 2., 1., 1., 3.], [1., 1., 2., 2., 1.]]
model = rustmc.BayesianDynamicNegativeBinomial(
    dispersion=4., process_sd=.15, coefficient_sd=1., group_sd=.4,
    shared_process_sd=.05,
)
fit = model.fit(y, exposure=exposure, chains=4, warmup=1000, draws=1000)
print(fit.summary())
forecast = fit.forecast(3, exposure=[[1., 1., 2.], [1., 2., 2.]])
# (chain, draw, group, horizon); group draws remain paired.
realizations = forecast.observation_samples
conditional_means = forecast.mean_samples
aggregate = forecast.aggregate_observation_samples
interval = np.quantile(realizations, [.025, .975], axis=(0, 1))
```

For predictors, pass finite `exog[group][time][feature]` to `fit` and `forecast`.
An intercept is added automatically. Future design widths must match the fitted
model, and a future design is required when fitting used predictors. Columns are
positional; callers must preserve their meaning/order. For hurdle models, the same
design enters separate occurrence and positive severity regressions; a zero effect
in either component can be learned independently.

## Statistical specification

For each component, with an implicit intercept in `x`,

```
population_beta ~ Normal(initial_mean * intercept, coefficient_sd^2 I)
beta_g | population_beta ~ Normal(population_beta, group_sd^2 I)
state_g,0 = 0
state_g,t = state_g,t-1 + shared_process_sd * shared_z_t + process_sd * z_g,t
eta_g,t = x_g,t @ beta_g + state_g,t
```

All `z` are independent standard normals. The first observation follows the first
state innovation; missing periods retain their place in this recursion. Setting
`group_sd=0` makes coefficients fully pooled. Setting `process_sd=0` removes group
state drift, and `shared_process_sd=0` removes future common shocks. Population
uncertainty still creates posterior dependence across groups. Group deviation
scales express a fixed amount of pooling; they are not estimated hyperparameters.

* Poisson: `y ~ Poisson(exposure * exp(eta))`.
* Negative binomial: the same mean `mu`, with `Var(y | eta)=mu+mu^2/dispersion`.
  Forecast simulation uses the exact Gamma-Poisson mixture.
* Gaussian: `y ~ Normal(eta, observation_sd^2)`.
* Hurdle: `P(y>0)=logistic(occurrence_eta)` and
  `log(y) | y>0 ~ Normal(severity_eta, observation_sd^2)`.
  Component 0 is severity, component 1 is occurrence; each has independent
  population/group regression priors and state innovations. Occurrence intercept
  prior mean is `occurrence_initial_mean`; severity uses `initial_mean`.

`coefficient_sd`, `group_sd`, `process_sd`, `shared_process_sd`,
`observation_sd`, and NB `dispersion` are **fixed specifications**, not posterior
draws. Only coefficients and trajectories are learned. Forecasts integrate their
posterior uncertainty and future shocks, conditional on these fixed scales.
The existing `BayesianHurdleLogNormal` has a different specification and remains
available unchanged.

## Inference, missingness, and numerical behavior

Inference uses block elliptical slice sampling in independent standard-normal
coordinates. A sweep updates population coefficients, each group's deviations,
the shared innovation path, and each group innovation path for each component.
The slice likelihood is the exact supported observation likelihood; non-Gaussian
models do not use a Gaussian FFBS approximation. Bracket exhaustion aborts with an
error. There is no adaptation, target-changing clipping, or removal of inconvenient
posterior draws. This general kernel can mix slowly with long series or tight
likelihoods; inspect rank R-hat, bulk/tail ESS, and MCSE before using a fit.

`fit.diagnostics()` and `summary()` cover population/group coefficients and terminal
states. `fit.state_samples(component=0)` exposes all historical states, which are
not individually covered by the default report. `sampler_stats` reports likelihood
evaluation counts including warmup/thinning, fixed scales, and joint group draws.
Hamiltonian divergences and acceptance telemetry are inapplicable.

NaN observations contribute no likelihood. Count zeros remain observed; positive
counts at zero exposure are rejected. Zero exposure deterministically produces zero
counts and supplies no information about the log rate when the observation is zero.
Exposure must be finite and nonnegative at every time, including missing periods.
It is supported only by the two count families. Count observations must be exact
nonnegative integers below `2^53`. All-missing panels generate a prior-target MCMC
fit; `prior_predictive` draws independent prior parameters directly.

Hurdle zeros inform occurrence only. Positive amounts inform both components.
All-zero histories are supported: the severity posterior stays equal to its prior.
`occurrence_samples` contains occurrence probabilities and `positive_mean_samples`
contains `exp(severity_eta + observation_sd^2/2)`; these are available only for hurdle
forecasts. `mean_samples` includes the probability of zero.

Fixed finite Gaussian scales imply finite positive predictive moments at every
finite horizon and finite deterministic design. For observed finite data, each
likelihood is bounded as a function of the Gaussian latent variables; therefore
the posterior remains dominated by a constant times the Gaussian prior. There is
no inverse-gamma variance mixture whose exponential moments could fail to exist.
Mathematical finiteness does not guarantee floating-point representability: overflow,
positive-severity underflow, and unsupported Poisson rates raise an error for the
whole request. Forecast paths are never filtered to manufacture finite summaries.

## Prior predictive checks and persistence

```python
prior = model.prior_predictive(12, groups=2, chains=1, draws=2000, seed=14)
saved = fit.to_json()
restored = rustmc.DynamicGLMFit.from_json(saved)
assert np.array_equal(
    fit.forecast(3, seed=99).observation_samples,
    restored.forecast(3, seed=99).observation_samples,
)
```

The versioned JSON artifact validates posterior dimensions, finite coefficient and
state values, chain/draw counts, and observations. It retains all paired posterior
draws and missing observations, but not training exog/exposure. Future designs and
exposures are supplied explicitly. `to_arviz()` exports named posterior draws and
observations. These artifacts resume forecasting; they do not resume an ESS chain.

Native validation includes an independent quadrature posterior for Poisson with
exposure, Gaussian hierarchical conditional means/covariance, simulated count
regression and time drift recovery, NB tail/moments, hurdle zero frequency and tail,
and shared-shock horizon covariance. These are targeted checks, not a universal
calibration or mixing guarantee.
