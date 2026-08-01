"""Numerical and validation coverage for the linear Gaussian state-space API."""

import math

import numpy as np
import pytest


def test_scalar_filter_matches_closed_form(rustmc_module):
    model = rustmc_module.LinearGaussianStateSpace.local_level(
        process_variance=1.0,
        observation_variance=2.0,
        initial_mean=0.0,
        initial_variance=3.0,
    )
    result = model.filter(np.array([2.0]))

    assert model.dimension == 1
    np.testing.assert_allclose(result.predicted_means, [[0.0]])
    np.testing.assert_allclose(result.predicted_covariances, [[[4.0]]])
    np.testing.assert_allclose(result.filtered_means, [[4.0 / 3.0]])
    np.testing.assert_allclose(result.filtered_covariances, [[[4.0 / 3.0]]])
    expected = -0.5 * (math.log(2.0 * math.pi) + math.log(6.0) + 4.0 / 6.0)
    assert result.log_likelihood == pytest.approx(expected)


def test_smoothing_and_forecasting(rustmc_module):
    model = rustmc_module.LinearGaussianStateSpace.local_level(1.0, 1.0)
    smoothed = model.smooth(np.array([1.0, 2.0]))

    np.testing.assert_allclose(smoothed.smoothed_means[:, 0], [1.0, 1.5])
    np.testing.assert_allclose(smoothed.smoothed_covariances[:, 0, 0], [0.5, 0.625])
    np.testing.assert_allclose(smoothed.filtered_means[-1], smoothed.smoothed_means[-1])

    forecast = model.forecast(np.array([1.0, 2.0]), steps=2)
    assert forecast.state_means.shape == (2, 1)
    assert forecast.state_covariances.shape == (2, 1, 1)
    np.testing.assert_allclose(forecast.observation_means, [1.5, 1.5])
    np.testing.assert_allclose(forecast.observation_variances, [2.625, 3.625])

    lower, upper = forecast.interval()
    critical = 1.959963984540054
    scale = np.sqrt(forecast.observation_variances)
    np.testing.assert_allclose(lower, forecast.observation_means - critical * scale, rtol=2e-4)
    np.testing.assert_allclose(upper, forecast.observation_means + critical * scale, rtol=2e-4)
    assert forecast.uncertainty_kind == "conditional_fixed_parameters"

    for invalid in (0.0, 1.0, -0.1, 1.1, np.nan):
        with pytest.raises(ValueError, match="strictly between"):
            forecast.interval(invalid)


def test_stationary_ar1_constructor_and_forecast(rustmc_module):
    model = rustmc_module.LinearGaussianStateSpace.stationary_ar1(0.8, 0.36, 0.25)
    assert model.dimension == 1
    filtered = model.filter(np.array([1.0]))
    forecast = model.forecast(np.array([1.0]), 2)
    np.testing.assert_allclose(
        forecast.observation_means[1],
        0.8 * forecast.observation_means[0],
    )
    assert np.isfinite(filtered.log_likelihood)

    with pytest.raises(rustmc_module.StateSpaceError, match="strictly between"):
        rustmc_module.LinearGaussianStateSpace.stationary_ar1(1.0, 0.36, 0.25)


def test_missing_value_is_prediction_only(rustmc_module):
    model = rustmc_module.LinearGaussianStateSpace.local_level(1.0, 2.0, 3.0, 4.0)
    result = model.filter(np.array([np.nan, 3.0]))

    np.testing.assert_allclose(result.filtered_means[0], result.predicted_means[0])
    np.testing.assert_allclose(
        result.filtered_covariances[0], result.predicted_covariances[0]
    )
    expected_second_only = -0.5 * (math.log(2.0 * math.pi) + math.log(8.0))
    assert result.log_likelihood == pytest.approx(expected_second_only)


def test_general_two_dimensional_model_and_trend_constructor(rustmc_module):
    general = rustmc_module.LinearGaussianStateSpace(
        transition=np.array([[1.0, 1.0], [0.0, 1.0]]),
        observation=np.array([1.0, 0.0]),
        process_covariance=np.diag([0.2, 0.1]),
        observation_variance=0.5,
        initial_mean=np.array([0.0, 0.1]),
        initial_covariance=np.eye(2),
    )
    trend = rustmc_module.LinearGaussianStateSpace.local_linear_trend(
        0.2, 0.1, 0.5, initial_trend=0.1
    )

    y = np.array([0.0, 0.2, 0.5])
    for model in (general, trend):
        result = model.smooth(y)
        assert result.smoothed_means.shape == (3, 2)
        assert result.smoothed_covariances.shape == (3, 2, 2)
        assert np.all(np.isfinite(result.smoothed_means))
        assert np.all(np.linalg.eigvalsh(result.smoothed_covariances) > 0.0)


@pytest.mark.parametrize(
    "factory, match",
    [
        (
            lambda rmc: rmc.LinearGaussianStateSpace.local_level(0.0, 1.0),
            "positive definite",
        ),
        (
            lambda rmc: rmc.LinearGaussianStateSpace.local_level(1.0, -1.0),
            "observation variance",
        ),
        (
            lambda rmc: rmc.LinearGaussianStateSpace(
                np.eye(2),
                np.ones(2),
                np.array([[1.0, 0.5], [0.0, 1.0]]),
                1.0,
                np.zeros(2),
                np.eye(2),
            ),
            "not symmetric",
        ),
        (
            lambda rmc: rmc.LinearGaussianStateSpace(
                np.eye(2), np.ones(1), np.eye(2), 1.0, np.zeros(2), np.eye(2)
            ),
            "observation has 1 entries",
        ),
    ],
)
def test_constructor_validation_is_a_structured_value_error(
    rustmc_module, factory, match
):
    with pytest.raises(rustmc_module.StateSpaceError, match=match) as error:
        factory(rustmc_module)
    assert isinstance(error.value, ValueError)


def test_observation_infinity_is_rejected_but_empty_series_is_supported(rustmc_module):
    model = rustmc_module.LinearGaussianStateSpace.local_level(1.0, 1.0)
    with pytest.raises(rustmc_module.StateSpaceError, match="not infinity"):
        model.filter(np.array([np.inf]))

    filtered = model.filter(np.array([], dtype=float))
    assert filtered.filtered_means.shape == (0, 1)
    assert filtered.log_likelihood == 0.0
    forecast = model.forecast(np.array([], dtype=float), 0)
    assert forecast.observation_means.shape == (0,)
