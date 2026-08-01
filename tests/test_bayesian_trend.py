"""Bayesian local-linear-trend inference and forecast coverage."""

import numpy as np
import pytest


def make_model(rmc):
    return rmc.BayesianLocalLinearTrend(
        level_variance_prior=rmc.InverseGammaPrior(shape=3.0, scale=0.24),
        slope_variance_prior=rmc.InverseGammaPrior(shape=3.0, scale=0.04),
        observation_variance_prior=rmc.InverseGammaPrior(shape=3.0, scale=0.70),
        initial_level=0.2,
        initial_slope=0.05,
        initial_level_variance=4.0,
        initial_slope_variance=1.0,
        initial_level_slope_covariance=0.25,
    )


def observations_with_gaps():
    return np.array([0.1, 0.3, np.nan, 0.8, 1.2, np.nan, 1.9, 2.4])


def test_constructor_properties_and_covariance_validation(rustmc_module):
    rmc = rustmc_module
    model = make_model(rmc)
    assert model.initial_level == 0.2
    assert model.initial_slope == 0.05
    np.testing.assert_array_equal(
        model.initial_covariance,
        [[4.0, 0.25], [0.25, 1.0]],
    )
    assert model.level_variance_prior.shape == 3.0
    assert model.slope_variance_prior.scale == 0.04
    assert model.observation_variance_prior.scale == 0.70

    priors = dict(
        level_variance_prior=rmc.InverseGammaPrior(3.0, 0.24),
        slope_variance_prior=rmc.InverseGammaPrior(3.0, 0.04),
        observation_variance_prior=rmc.InverseGammaPrior(3.0, 0.70),
    )
    with pytest.raises(ValueError, match="initial level and slope must be finite"):
        rmc.BayesianLocalLinearTrend(**priors, initial_level=np.nan)
    with pytest.raises(ValueError, match="initial variances must be finite and positive"):
        rmc.BayesianLocalLinearTrend(**priors, initial_slope_variance=0.0)
    with pytest.raises(ValueError, match="positive definite"):
        rmc.BayesianLocalLinearTrend(
            **priors,
            initial_level_variance=1.0,
            initial_slope_variance=1.0,
            initial_level_slope_covariance=1.0,
        )
    extreme = rmc.BayesianLocalLinearTrend(
        **priors,
        initial_level_variance=1e200,
        initial_slope_variance=1e200,
    )
    assert np.isfinite(extreme.initial_covariance).all()


def test_seeded_fit_preserves_provenance_missing_schedule_and_samples(rustmc_module):
    observations = observations_with_gaps()
    model = make_model(rustmc_module)
    first = model.fit(
        observations,
        chains=2,
        draws=80,
        warmup=50,
        thin=2,
        seed=17,
    )
    second = model.fit(
        observations,
        chains=2,
        draws=80,
        warmup=50,
        thin=2,
        seed=17,
    )

    assert first.chains == 2
    assert first.draws == 80
    assert first.time_count == observations.size
    assert first.observed_count == 6
    assert first.warmup == 50
    assert first.thin == 2
    assert "time_count=8" in repr(first)

    samples = first.get_samples_2d()
    repeated = second.get_samples_2d()
    assert set(samples) == {
        "level_variance",
        "slope_variance",
        "observation_variance",
        "level_sd",
        "slope_sd",
        "observation_sd",
        "terminal_level",
        "terminal_slope",
    }
    for name, values in samples.items():
        assert values.shape == (2, 80)
        assert np.isfinite(values).all()
        np.testing.assert_array_equal(values, repeated[name])
    for component in ("level", "slope", "observation"):
        np.testing.assert_allclose(
            samples[f"{component}_sd"] ** 2,
            samples[f"{component}_variance"],
        )
    assert not np.array_equal(samples["level_variance"][0], samples["level_variance"][1])


def test_forecast_paths_summaries_quantiles_and_95_percent_intervals(rustmc_module):
    fit = make_model(rustmc_module).fit(
        observations_with_gaps(),
        chains=2,
        draws=100,
        warmup=50,
        seed=19,
    )
    forecast = fit.forecast(steps=6, seed=23)
    repeated = fit.forecast(steps=6, seed=23)

    assert forecast.chains == 2
    assert forecast.draws == 100
    assert forecast.steps == 6
    assert "steps=6" in repr(forecast)
    for name in ("level", "slope", "observation"):
        paths = getattr(forecast, f"{name}_samples")
        assert paths.shape == (2, 100, 6)
        np.testing.assert_array_equal(
            paths,
            getattr(repeated, f"{name}_samples"),
        )
        np.testing.assert_allclose(
            getattr(forecast, f"{name}_mean"),
            paths.mean(axis=(0, 1)),
        )
        for probability in (0.1, 0.5, 0.9):
            np.testing.assert_allclose(
                getattr(forecast, f"{name}_quantile")(probability),
                np.quantile(paths, probability, axis=(0, 1)),
            )

    interval_methods = {
        "level": forecast.level_interval,
        "slope": forecast.slope_interval,
        "observation": forecast.interval,
    }
    for name, method in interval_methods.items():
        lower, upper = method(0.95)
        paths = getattr(forecast, f"{name}_samples")
        np.testing.assert_allclose(lower, np.quantile(paths, 0.025, axis=(0, 1)))
        np.testing.assert_allclose(upper, np.quantile(paths, 0.975, axis=(0, 1)))

    # Level and slope paths are coupled: the next level uses the preceding
    # slope before receiving its own process shock. This at least guards that
    # forecast state draws were not independently reshuffled by component.
    assert np.corrcoef(
        forecast.level_samples[:, :, -1].ravel(),
        forecast.slope_samples[:, :, -1].ravel(),
    )[0, 1] > 0.05
    assert forecast.uncertainty_kind == "parameter_integrated_posterior_predictive"
    assert forecast.interval_kind == "pointwise_equal_tailed"


def test_invalid_fit_data_sampling_controls_forecast_and_levels(rustmc_module):
    model = make_model(rustmc_module)
    for observations in (
        np.array([], dtype=float),
        np.array([1.0, 2.0]),
        np.array([1.0, 2.0, np.nan]),
        np.array([np.nan, np.nan, np.nan]),
        np.array([1.0, 2.0, np.inf]),
    ):
        with pytest.raises(ValueError):
            model.fit(observations, chains=1, draws=2, warmup=1)

    valid = np.array([0.0, 0.2, 0.5])
    for controls in (
        {"chains": 0},
        {"draws": 0},
        {"thin": 0},
    ):
        with pytest.raises(ValueError, match="positive"):
            model.fit(
                valid,
                **{"chains": 1, "draws": 2, "warmup": 1, "thin": 1, **controls},
            )

    fit = model.fit(valid, chains=1, draws=5, warmup=3, seed=4)
    with pytest.raises(ValueError, match="horizon"):
        fit.forecast(0)
    forecast = fit.forecast(2)
    for probability in (-0.1, 1.1, np.nan):
        for method in (
            forecast.level_quantile,
            forecast.slope_quantile,
            forecast.observation_quantile,
        ):
            with pytest.raises(ValueError, match="between zero and one"):
                method(probability)
    for level in (0.0, 1.0, -0.1, 1.1, np.nan):
        for method in (forecast.level_interval, forecast.slope_interval, forecast.interval):
            with pytest.raises(ValueError, match="strictly between"):
                method(level)


def test_arviz_export_preserves_chain_draw_and_missing_observations(rustmc_module):
    pytest.importorskip("arviz")
    observations = observations_with_gaps()
    fit = make_model(rustmc_module).fit(
        observations,
        chains=2,
        draws=10,
        warmup=5,
        seed=5,
    )
    data = fit.to_arviz()
    posterior_group = data.posterior
    posterior = getattr(posterior_group, "dataset", posterior_group)
    observed_group = data.observed_data
    observed = getattr(observed_group, "dataset", observed_group)

    for name in (
        "level_variance",
        "slope_variance",
        "observation_variance",
        "terminal_level",
        "terminal_slope",
    ):
        assert posterior[name].shape == (2, 10)
    np.testing.assert_array_equal(observed["y"].values, observations)
