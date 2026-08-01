"""Bayesian AR(p) inference and recursive posterior forecasting coverage."""

import numpy as np
import pytest


TRUE_COEFFICIENTS = np.array([0.35, 0.55, -0.25, 0.12])


def simulate_ar3(seed=2026, size=1_000, burn_in=200):
    """Generate a deterministic, stationary AR(3) series."""
    rng = np.random.default_rng(seed)
    values = np.zeros(size + burn_in + 3)
    values[:3] = 0.5
    for time in range(3, values.size):
        values[time] = (
            TRUE_COEFFICIENTS[0]
            + TRUE_COEFFICIENTS[1] * values[time - 1]
            + TRUE_COEFFICIENTS[2] * values[time - 2]
            + TRUE_COEFFICIENTS[3] * values[time - 3]
            + rng.normal(0.0, 0.30)
        )
    return values[burn_in + 3 :]


def make_prior(rmc, order=3):
    dimension = order + 1
    return rmc.NormalInverseGammaPrior(
        coefficient_mean=np.zeros(dimension),
        coefficient_precision=np.eye(dimension) * 0.05,
        variance_shape=2.5,
        variance_scale=0.2,
    )


def test_prior_and_model_surface_preserve_coefficient_order(rustmc_module):
    rmc = rustmc_module
    mean = np.array([0.1, 0.2, 0.3, 0.4])
    precision = np.diag([1.0, 2.0, 3.0, 4.0])
    prior = rmc.NormalInverseGammaPrior(mean, precision, 3.0, 0.5)
    model = rmc.BayesianAutoRegression(order=3, prior=prior)

    assert rmc.BayesianAR is rmc.BayesianAutoRegression
    assert prior.coefficient_count == 4
    np.testing.assert_array_equal(prior.coefficient_mean, mean)
    np.testing.assert_array_equal(prior.coefficient_precision, precision)
    assert prior.variance_shape == 3.0
    assert prior.variance_scale == 0.5
    assert model.order == 3
    np.testing.assert_array_equal(model.prior.coefficient_mean, mean)


def test_seeded_ar3_fit_shapes_reproducibility_and_recovery(rustmc_module):
    rmc = rustmc_module
    observations = simulate_ar3()
    model = rmc.BayesianAR(order=3, prior=make_prior(rmc))
    first = model.fit(observations, chains=2, draws=500, seed=17)
    second = model.fit(observations, chains=2, draws=500, seed=17)

    assert first.order == 3
    assert first.chains == 2
    assert first.draws == 500
    assert first.time_count == observations.size
    assert first.regression_count == observations.size - 3
    assert first.seed == 17

    first_samples = first.get_samples()
    second_samples = second.get_samples()
    assert set(first_samples) == {
        "coefficient",
        "innovation_variance",
        "innovation_sd",
    }
    assert first_samples["coefficient"].shape == (2, 500, 4)
    assert first_samples["innovation_variance"].shape == (2, 500)
    assert first_samples["innovation_sd"].shape == (2, 500)
    for name in first_samples:
        np.testing.assert_array_equal(first_samples[name], second_samples[name])
    np.testing.assert_allclose(
        first_samples["innovation_sd"] ** 2,
        first_samples["innovation_variance"],
    )
    assert not np.array_equal(
        first_samples["coefficient"][0], first_samples["coefficient"][1]
    )

    # Coefficients are ordered [intercept, lag_1, lag_2, lag_3].
    posterior_mean = first_samples["coefficient"].mean(axis=(0, 1))
    np.testing.assert_allclose(posterior_mean, TRUE_COEFFICIENTS, atol=0.07)


def test_recursive_forecast_paths_and_empirical_intervals(rustmc_module):
    rmc = rustmc_module
    observations = simulate_ar3(size=250)
    fit = rmc.BayesianAR(order=3, prior=make_prior(rmc)).fit(
        observations, chains=2, draws=120, seed=19
    )
    forecast = fit.forecast(steps=5, seed=23)
    repeated = fit.forecast(steps=5, seed=23)

    assert forecast.chains == 2
    assert forecast.draws == 120
    assert forecast.steps == 5
    assert forecast.conditional_mean_samples.shape == (2, 120, 5)
    assert forecast.observation_samples.shape == (2, 120, 5)
    np.testing.assert_array_equal(
        forecast.conditional_mean_samples, repeated.conditional_mean_samples
    )
    np.testing.assert_array_equal(
        forecast.observation_samples, repeated.observation_samples
    )
    np.testing.assert_allclose(
        forecast.conditional_mean,
        forecast.conditional_mean_samples.mean(axis=(0, 1)),
    )
    np.testing.assert_allclose(
        forecast.observation_mean,
        forecast.observation_samples.mean(axis=(0, 1)),
    )

    coefficients = fit.get_samples()["coefficient"]
    expected_first = (
        coefficients[:, :, 0]
        + coefficients[:, :, 1] * observations[-1]
        + coefficients[:, :, 2] * observations[-2]
        + coefficients[:, :, 3] * observations[-3]
    )
    np.testing.assert_allclose(
        forecast.conditional_mean_samples[:, :, 0], expected_first
    )
    # Recursion uses each path's simulated observation, not its conditional mean.
    expected_second = (
        coefficients[:, :, 0]
        + coefficients[:, :, 1] * forecast.observation_samples[:, :, 0]
        + coefficients[:, :, 2] * observations[-1]
        + coefficients[:, :, 3] * observations[-2]
    )
    np.testing.assert_allclose(
        forecast.conditional_mean_samples[:, :, 1], expected_second
    )

    lower, upper = forecast.interval(0.95)
    mean_lower, mean_upper = forecast.conditional_mean_interval(0.95)
    np.testing.assert_allclose(
        lower,
        np.quantile(forecast.observation_samples, 0.025, axis=(0, 1)),
    )
    np.testing.assert_allclose(
        upper,
        np.quantile(forecast.observation_samples, 0.975, axis=(0, 1)),
    )
    np.testing.assert_allclose(
        mean_lower,
        np.quantile(forecast.conditional_mean_samples, 0.025, axis=(0, 1)),
    )
    np.testing.assert_allclose(
        mean_upper,
        np.quantile(forecast.conditional_mean_samples, 0.975, axis=(0, 1)),
    )
    np.testing.assert_allclose(
        forecast.observation_quantile(0.5),
        np.quantile(forecast.observation_samples, 0.5, axis=(0, 1)),
    )
    np.testing.assert_allclose(
        forecast.conditional_mean_quantile(0.5),
        np.quantile(forecast.conditional_mean_samples, 0.5, axis=(0, 1)),
    )
    assert forecast.uncertainty_kind == "parameter_integrated_posterior_predictive"
    assert forecast.interval_kind == "pointwise_equal_tailed"


def test_bayesian_ar_prior_and_model_validation(rustmc_module):
    rmc = rustmc_module
    with pytest.raises(rmc.StateSpaceError, match="intercept"):
        rmc.NormalInverseGammaPrior(np.zeros(1), np.eye(1), 2.0, 1.0)
    with pytest.raises(rmc.StateSpaceError, match="shape"):
        rmc.NormalInverseGammaPrior(np.zeros(2), np.eye(3), 2.0, 1.0)
    with pytest.raises(rmc.StateSpaceError, match="finite"):
        rmc.NormalInverseGammaPrior(np.array([0.0, np.nan]), np.eye(2), 2.0, 1.0)
    with pytest.raises(rmc.StateSpaceError, match="finite"):
        rmc.NormalInverseGammaPrior(
            np.zeros(2), np.array([[1.0, 0.0], [0.0, np.inf]]), 2.0, 1.0
        )
    with pytest.raises(rmc.StateSpaceError, match="symmetric"):
        rmc.NormalInverseGammaPrior(
            np.zeros(2), np.array([[1.0, 0.5], [0.0, 1.0]]), 2.0, 1.0
        )
    with pytest.raises(rmc.StateSpaceError, match="positive definite"):
        rmc.NormalInverseGammaPrior(np.zeros(2), np.diag([1.0, 0.0]), 2.0, 1.0)
    for shape, scale in ((0.0, 1.0), (2.0, 0.0), (np.nan, 1.0)):
        with pytest.raises(rmc.StateSpaceError):
            rmc.NormalInverseGammaPrior(np.zeros(2), np.eye(2), shape, scale)

    ar1_prior = make_prior(rmc, order=1)
    with pytest.raises(rmc.StateSpaceError, match="at least one"):
        rmc.BayesianAR(order=0, prior=ar1_prior)
    with pytest.raises(rmc.StateSpaceError, match="requires 3"):
        rmc.BayesianAR(order=2, prior=ar1_prior)


def test_bayesian_ar_data_sampling_and_forecast_validation(rustmc_module):
    rmc = rustmc_module
    model = rmc.BayesianAR(order=3, prior=make_prior(rmc))
    for observations in (
        np.array([], dtype=float),
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0, 2.0, 3.0, np.nan]),
        np.array([1.0, 2.0, 3.0, np.inf]),
    ):
        with pytest.raises(rmc.StateSpaceError):
            model.fit(observations, chains=1, draws=2)
    observations = simulate_ar3(size=30)
    with pytest.raises(rmc.StateSpaceError, match="chains"):
        model.fit(observations, chains=0, draws=2)
    with pytest.raises(rmc.StateSpaceError, match="draws"):
        model.fit(observations, chains=1, draws=0)

    fit = model.fit(observations, chains=1, draws=8)
    with pytest.raises(rmc.StateSpaceError, match="horizon"):
        fit.forecast(0)
    forecast = fit.forecast(2)
    for level in (0.0, 1.0, -0.1, 1.1, np.nan):
        with pytest.raises(ValueError, match="strictly between"):
            forecast.interval(level)
        with pytest.raises(ValueError, match="strictly between"):
            forecast.conditional_mean_interval(level)
    for probability in (-0.1, 1.1, np.nan):
        with pytest.raises(rmc.StateSpaceError, match="probabilities"):
            forecast.observation_quantile(probability)
        with pytest.raises(rmc.StateSpaceError, match="probabilities"):
            forecast.conditional_mean_quantile(probability)


def test_bayesian_ar_arviz_export_preserves_chain_draw_and_coefficients(
    rustmc_module,
):
    pytest.importorskip("arviz")
    rmc = rustmc_module
    observations = simulate_ar3(size=40)
    fit = rmc.BayesianAR(order=3, prior=make_prior(rmc)).fit(
        observations, chains=2, draws=10, seed=5
    )
    data = fit.to_arviz()
    posterior_group = data.posterior
    posterior = getattr(posterior_group, "dataset", posterior_group)
    observed_group = data.observed_data
    observed = getattr(observed_group, "dataset", observed_group)

    assert posterior["coefficient"].shape == (2, 10, 4)
    assert posterior["innovation_variance"].shape == (2, 10)
    assert posterior["innovation_sd"].shape == (2, 10)
    np.testing.assert_array_equal(
        posterior["coefficient"].values, fit.get_samples()["coefficient"]
    )
    np.testing.assert_array_equal(observed["y"].values, observations)
