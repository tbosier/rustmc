"""Bayesian structural seasonal local-level inference and forecasting."""

import numpy as np
import pytest


def make_model(rmc):
    return rmc.BayesianSeasonalLocalLevel(
        period=4,
        level_variance_prior=rmc.InverseGammaPrior(3.0, 0.16),
        seasonal_variance_prior=rmc.InverseGammaPrior(3.0, 0.08),
        observation_variance_prior=rmc.InverseGammaPrior(3.0, 0.36),
        initial_level=5.0,
        initial_seasonal_effects=[1.0, -0.5, -0.25, -0.25],
        initial_level_variance=4.0,
        initial_seasonal_variance=2.0,
    )


def seasonal_observations(seed=123, count=40):
    rng = np.random.default_rng(seed)
    period = 4
    state = np.array([5.0, -0.25, -0.25, -0.5])
    values = []
    for _ in range(count):
        next_level = state[0] + rng.normal(0.0, np.sqrt(0.04))
        next_seasonal = -state[1:].sum() + rng.normal(0.0, np.sqrt(0.02))
        state[2:] = state[1:-1]
        state[0] = next_level
        state[1] = next_seasonal
        values.append(next_level + next_seasonal + rng.normal(0.0, 0.3))
    return np.asarray(values)


def test_seeded_fit_infers_all_variances_and_handles_missing(rustmc_module):
    observations = seasonal_observations()
    observations[[7, 22]] = np.nan
    model = make_model(rustmc_module)
    first = model.fit(observations, chains=2, draws=60, warmup=40, seed=17)
    second = model.fit(observations, chains=2, draws=60, warmup=40, seed=17)

    assert model.period == 4
    np.testing.assert_allclose(model.initial_seasonal_effects.sum(), 0.0)
    assert first.period == 4
    assert first.chains == 2
    assert first.draws == 60
    assert first.time_count == 40
    assert first.observed_count == 38
    samples = first.get_samples_2d()
    repeated = second.get_samples_2d()
    assert set(samples) == {
        "level_variance",
        "seasonal_variance",
        "observation_variance",
        "level_sd",
        "seasonal_sd",
        "observation_sd",
        "terminal_level",
        "terminal_seasonal",
    }
    for name, values in samples.items():
        assert values.shape == (2, 60)
        assert np.isfinite(values).all()
        np.testing.assert_array_equal(values, repeated[name])
    for component in ("level", "seasonal", "observation"):
        assert np.all(samples[f"{component}_variance"] > 0.0)
        assert np.std(samples[f"{component}_variance"]) > 0.0
        np.testing.assert_allclose(
            samples[f"{component}_sd"] ** 2,
            samples[f"{component}_variance"],
        )


def test_forecast_paths_and_cumulative_intervals_are_drawwise(rustmc_module):
    fit = make_model(rustmc_module).fit(
        seasonal_observations(), chains=2, draws=80, warmup=50, seed=19
    )
    forecast = fit.forecast(steps=8, seed=23)
    repeated = fit.forecast(steps=8, seed=23)

    assert forecast.chains == 2
    assert forecast.draws == 80
    assert forecast.steps == 8
    for name in ("level", "seasonal", "observation"):
        paths = getattr(forecast, f"{name}_samples")
        assert paths.shape == (2, 80, 8)
        np.testing.assert_array_equal(paths, getattr(repeated, f"{name}_samples"))
        np.testing.assert_allclose(
            getattr(forecast, f"{name}_mean"), paths.mean(axis=(0, 1))
        )
    cumulative = forecast.cumulative_observation_samples
    np.testing.assert_array_equal(cumulative, np.cumsum(forecast.observation_samples, axis=2))
    np.testing.assert_allclose(
        forecast.cumulative_observation_mean, cumulative.mean(axis=(0, 1))
    )

    lower, upper = forecast.interval()
    cumulative_lower, cumulative_upper = forecast.cumulative_interval()
    np.testing.assert_allclose(
        lower, np.quantile(forecast.observation_samples, 0.025, axis=(0, 1))
    )
    np.testing.assert_allclose(
        upper, np.quantile(forecast.observation_samples, 0.975, axis=(0, 1))
    )
    np.testing.assert_allclose(
        cumulative_lower, np.quantile(cumulative, 0.025, axis=(0, 1))
    )
    np.testing.assert_allclose(
        cumulative_upper, np.quantile(cumulative, 0.975, axis=(0, 1))
    )
    assert forecast.uncertainty_kind == "parameter_integrated_posterior_predictive"
    assert forecast.interval_kind == "pointwise_equal_tailed"


def test_synthetic_variance_recovery_is_in_the_right_scale(rustmc_module):
    fit = make_model(rustmc_module).fit(
        seasonal_observations(seed=321, count=100),
        chains=2,
        draws=200,
        warmup=150,
        seed=5,
    )
    samples = fit.get_samples_2d()
    truths = {
        "level_variance": 0.04,
        "seasonal_variance": 0.02,
        "observation_variance": 0.09,
    }
    for name, truth in truths.items():
        posterior_median = np.median(samples[name])
        assert truth / 2.0 < posterior_median < truth * 2.0


def test_validation_rejects_unidentified_or_invalid_seasonal_fits(rustmc_module):
    rmc = rustmc_module
    with pytest.raises(ValueError, match="sum to zero"):
        rmc.BayesianSeasonalLocalLevel(
            4,
            rmc.InverseGammaPrior(3.0, 0.16),
            rmc.InverseGammaPrior(3.0, 0.08),
            rmc.InverseGammaPrior(3.0, 0.36),
            initial_seasonal_effects=[1.0, 0.0, 0.0, 0.0],
        )
    model = make_model(rmc)
    for observations in (
        np.zeros(7),
        np.array([0.0] * 5 + [np.nan] * 3),
        np.array([0.0] * 7 + [np.inf]),
    ):
        with pytest.raises(ValueError):
            model.fit(observations, chains=1, draws=2, warmup=1)
    fit = model.fit(np.zeros(8), chains=1, draws=4, warmup=2)
    with pytest.raises(ValueError, match="horizon"):
        fit.forecast(0)
    forecast = fit.forecast(2)
    for level in (0.0, 1.0, -0.1, 1.1, np.nan):
        with pytest.raises(ValueError, match="strictly between"):
            forecast.interval(level)
        with pytest.raises(ValueError, match="strictly between"):
            forecast.cumulative_interval(level)
