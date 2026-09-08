"""Joint regression/structural inference, design contracts and calendar phase."""
import numpy as np
import pytest


def prior(rmc, count=1, variance=4.0):
    return rmc.GaussianCoefficientPrior(np.zeros(count), np.eye(count) * variance)


def models(rmc):
    ig = rmc.InverseGammaPrior
    return [
        rmc.BayesianLocalLevel(ig(3, 0.04), ig(3, 0.4)),
        rmc.BayesianLocalLinearTrend(ig(3, 0.04), ig(3, 0.004), ig(3, 0.4)),
        rmc.BayesianSeasonalLocalLevel(12, ig(3, 0.04), ig(3, 0.02), ig(3, 0.4)),
    ]


@pytest.mark.parametrize("model_index", range(3))
def test_joint_fit_forecast_components_and_validation(rustmc_module, model_index):
    rmc = rustmc_module
    model = models(rmc)[model_index]
    rng = np.random.default_rng(19)
    x = rng.normal(size=(30, 1))
    y = 2 * x[:, 0] + rng.normal(scale=0.3, size=30)
    y[3] = np.nan
    options = dict(chains=2, draws=70, warmup=50, seed=17, coefficient_prior=prior(rmc))
    fit = model.fit(y, exog=x, **options)
    repeated = model.fit(y, exog=x, **options)
    assert isinstance(fit, rmc.BayesianRegressionFit)
    assert fit.time_count == 30 and fit.observed_count == 29
    for key, value in fit.get_samples_2d().items():
        np.testing.assert_array_equal(value, repeated.get_samples_2d()[key])
    beta = fit.get_samples_2d()["coefficients"]
    assert beta.shape == (2, 70, 1)
    assert abs(beta.mean() - 2.0) < 0.4
    future = np.array([[1.0], [2.0], [-1.0]])
    forecast = fit.forecast(3, exog=future, seed=4)
    np.testing.assert_allclose(forecast.regression_samples, beta * future[:, 0])
    structural = forecast.level_samples
    if model_index == 2:
        structural = structural + forecast.seasonal_samples
    np.testing.assert_allclose(forecast.mean_samples, structural + forecast.regression_samples)
    np.testing.assert_array_equal(forecast.cumulative_observation_samples,
                                  np.cumsum(forecast.observation_samples, axis=2))
    np.testing.assert_allclose(forecast.interval(), np.quantile(forecast.observation_samples, [0.025, 0.975], axis=(0, 1)))
    for invalid in (None, np.zeros((2, 1)), np.zeros((3, 2)), np.full((3, 1), np.nan)):
        with pytest.raises(ValueError):
            fit.forecast(3, exog=invalid)
    with pytest.raises(ValueError, match="prior"):
        model.fit(y, exog=x, draws=1, warmup=1)
    for invalid in (x[:-1], np.zeros((30, 2)), np.full((30, 1), np.inf)):
        with pytest.raises(ValueError):
            model.fit(y, exog=invalid, **options)


def test_collinear_features_and_prior_validation(rustmc_module):
    rmc = rustmc_module
    model = models(rmc)[0]
    fit = model.fit(np.ones(12), exog=np.ones((12, 2)), coefficient_prior=prior(rmc, 2), chains=1, draws=10, warmup=5)
    assert np.isfinite(fit.get_samples_2d()["coefficients"]).all()
    for covariance in (np.zeros((2, 2)), np.array([[1., 2.], [2., 1.]]), np.eye(1), np.full((2, 2), np.nan)):
        with pytest.raises(ValueError):
            rmc.GaussianCoefficientPrior(np.zeros(2), covariance)


def test_fixed_time_varying_rows_against_direct_gaussian_conditioning(rustmc_module):
    rmc = rustmc_module
    z = np.array([[1., -1.], [1., 0.], [1., 1.], [1., 2.]])
    y = np.array([1., 2., np.nan, 4.])
    model = rmc.LinearGaussianStateSpace(np.eye(2), np.ones(2), np.zeros((2, 2)), 0.5, np.zeros(2), np.diag([2., 3.])).with_observation_rows(z)
    keep = np.isfinite(y)
    covariance = np.linalg.inv(np.diag([.5, 1/3]) + z[keep].T @ z[keep] / .5)
    mean = covariance @ z[keep].T @ y[keep] / .5
    smoothed = model.smooth(y)
    np.testing.assert_allclose(smoothed.smoothed_means, np.tile(mean, (4, 1)))
    future = np.array([[1., 3.], [1., -2.]])
    forecast = model.forecast(y, 2, future_observation_rows=future)
    np.testing.assert_allclose(forecast.observation_means, future @ mean)
    expected = future @ covariance @ future.T + .5 * np.eye(2)
    np.testing.assert_allclose(forecast.observation_covariance, expected)
    np.testing.assert_allclose(forecast.cumulative_observation_variances[-1], expected.sum())
    with pytest.raises(ValueError, match="future"):
        model.forecast(y, 2)
    with pytest.raises(ValueError):
        model.filter(y[:3])
    with pytest.raises(ValueError):
        model.with_observation_rows(np.ones((4, 3)))


@pytest.mark.parametrize("period,count", [(12, 12), (12, 18), (52, 9)])
def test_short_seasonal_histories_and_missing_phases(rustmc_module, period, count):
    rmc = rustmc_module
    ig = rmc.InverseGammaPrior
    y = np.sin(2 * np.pi * np.arange(count) / period)
    y[1::3] = np.nan
    model = rmc.BayesianSeasonalLocalLevel(period, ig(3, .04), ig(3, .08), ig(3, .4))
    fit = model.fit(y, chains=1, draws=4, warmup=2)
    assert np.isfinite(fit.forecast(3).observation_samples).all()


def test_fourier_short_history_future_phase_and_shrinkage(rustmc_module):
    rmc = rustmc_module
    complete = rmc.fourier_design(30, 12, 2)
    np.testing.assert_array_equal(rmc.fourier_design(12, 12, 2, start=18), complete[18:])
    assert rmc.fourier_design(5, 12, 6).shape == (5, 11)
    assert rmc.fourier_design(0, 12, 6).shape == (0, 11)
    with pytest.raises(ValueError):
        rmc.fourier_design(5, 12, 7)
    model = models(rmc)[0]
    y = complete[:18] @ np.array([1.5, -.8, .2, .1])
    fit = model.fit(y, exog=complete[:18], coefficient_prior=prior(rmc, 4), chains=2, draws=100, warmup=70)
    future = fit.forecast(12, exog=complete[18:])
    truth = complete[18:] @ np.array([1.5, -.8, .2, .1])
    assert np.mean((future.mean_samples.mean(axis=(0, 1)) - truth)**2) < .12


def test_repeated_short_history_holdouts_and_prior_sensitivity(rustmc_module):
    """A finite seeded calibration smoke check, not a universal coverage guarantee."""
    rmc = rustmc_module
    rng = np.random.default_rng(552)
    x = rmc.fourier_design(24, 12, 1)
    ig = rmc.InverseGammaPrior
    model = rmc.BayesianLocalLevel(ig(3, .04), ig(3, .16), initial_variance=1.)
    covered, squared_errors, baseline_errors = [], [], []
    for repeat in range(12):
        level = np.cumsum(rng.normal(scale=.1, size=24))
        y = level + x @ np.array([1.5, -.8]) + rng.normal(scale=.2, size=24)
        fit = model.fit(y[:18], exog=x[:18], coefficient_prior=prior(rmc, 2),
                        chains=2, draws=70, warmup=50, seed=repeat)
        prediction = fit.forecast(6, exog=x[18:], seed=repeat + 100)
        lower, upper = prediction.interval(.9)
        covered.extend((lower <= y[18:]) & (y[18:] <= upper))
        squared_errors.extend((prediction.mean_samples.mean(axis=(0, 1)) - y[18:])**2)
        baseline_errors.extend((y[6:12] - y[18:])**2)
    assert np.mean(covered) >= .75
    assert np.mean(squared_errors) < np.mean(baseline_errors)
    # Unobserved harmonic phases retain uncertainty controlled by proper priors.
    sparse = np.array([0., 0.] + [np.nan] * 10)
    weak = model.fit(sparse, exog=x[:12], coefficient_prior=prior(rmc, 2, 9.),
                     chains=2, draws=120, warmup=80, seed=23)
    strong = model.fit(sparse, exog=x[:12], coefficient_prior=prior(rmc, 2, .01),
                       chains=2, draws=120, warmup=80, seed=23)
    assert weak.get_samples_2d()["coefficients"].std(axis=(0, 1)).sum() > strong.get_samples_2d()["coefficients"].std(axis=(0, 1)).sum()
