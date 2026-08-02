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
    np.testing.assert_allclose(
        forecast.observation_covariance,
        [[2.625, 1.625], [1.625, 3.625]],
    )
    np.testing.assert_allclose(forecast.cumulative_observation_means, [1.5, 3.0])
    np.testing.assert_allclose(forecast.cumulative_observation_variances, [2.625, 9.5])

    lower, upper = forecast.interval()
    critical = 1.959963984540054
    scale = np.sqrt(forecast.observation_variances)
    np.testing.assert_allclose(lower, forecast.observation_means - critical * scale, rtol=2e-4)
    np.testing.assert_allclose(upper, forecast.observation_means + critical * scale, rtol=2e-4)
    cumulative_lower, cumulative_upper = forecast.cumulative_interval()
    cumulative_scale = np.sqrt(forecast.cumulative_observation_variances)
    np.testing.assert_allclose(
        cumulative_lower,
        forecast.cumulative_observation_means - critical * cumulative_scale,
        rtol=2e-4,
    )
    np.testing.assert_allclose(
        cumulative_upper,
        forecast.cumulative_observation_means + critical * cumulative_scale,
        rtol=2e-4,
    )
    assert forecast.cumulative_observation_variances[-1] > np.sum(
        forecast.observation_variances
    )
    assert forecast.uncertainty_kind == "conditional_fixed_parameters"

    for invalid in (0.0, 1.0, -0.1, 1.1, np.nan):
        with pytest.raises(ValueError, match="strictly between"):
            forecast.interval(invalid)
        with pytest.raises(ValueError, match="strictly between"):
            forecast.cumulative_interval(invalid)


def test_seasonal_local_level_repeats_cycle_and_supports_filtering(rustmc_module):
    effects = [1.0, -0.5, -0.25, -0.25]
    model = rustmc_module.LinearGaussianStateSpace.seasonal_local_level(
        period=4,
        level_variance=0.0,
        seasonal_variance=0.0,
        observation_variance=0.1,
        initial_level=10.0,
        initial_seasonal_effects=effects,
        initial_level_variance=1.0,
        initial_seasonal_variance=1.0,
    )

    assert model.dimension == 4
    forecast = model.forecast(np.array([], dtype=float), 8)
    np.testing.assert_allclose(
        forecast.observation_means,
        10.0 + np.resize(np.asarray(effects), 8),
    )
    assert forecast.observation_covariance.shape == (8, 8)
    np.testing.assert_allclose(
        forecast.observation_covariance,
        forecast.observation_covariance.T,
    )
    assert np.all(np.linalg.eigvalsh(forecast.observation_covariance) > 0.0)

    filtered = model.filter(10.0 + np.resize(np.asarray(effects), 12))
    assert filtered.filtered_means.shape == (12, 4)
    assert np.isfinite(filtered.log_likelihood)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"period": 1}, "period must be at least 2"),
        (
            {"period": 4, "initial_seasonal_effects": [1.0, 0.0, 0.0, 0.0]},
            "must sum to zero",
        ),
        ({"period": 4, "initial_seasonal_effects": [0.0, 0.0]}, "expected 4"),
        ({"period": 4, "seasonal_variance": -0.1}, "non-negative"),
    ],
)
def test_seasonal_local_level_validation(rustmc_module, kwargs, match):
    arguments = {
        "period": 4,
        "level_variance": 0.1,
        "seasonal_variance": 0.1,
        "observation_variance": 0.2,
    }
    arguments.update(kwargs)
    with pytest.raises(rustmc_module.StateSpaceError, match=match):
        rustmc_module.LinearGaussianStateSpace.seasonal_local_level(**arguments)


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
            lambda rmc: rmc.LinearGaussianStateSpace.local_level(-1.0, 1.0),
            "positive semidefinite",
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
    assert forecast.observation_covariance.shape == (0, 0)
    assert forecast.cumulative_observation_means.shape == (0,)
    assert forecast.cumulative_observation_variances.shape == (0,)
