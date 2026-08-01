"""Bayesian local-level inference and posterior forecast coverage."""

import numpy as np
import pytest


def make_model(rmc):
    return rmc.BayesianLocalLevel(
        process_variance_prior=rmc.InverseGammaPrior(shape=2.5, scale=0.3),
        observation_variance_prior=rmc.InverseGammaPrior(shape=2.5, scale=0.6),
        initial_mean=0.0,
        initial_variance=4.0,
    )


def test_seeded_multichain_fit_and_parameter_samples(rustmc_module):
    model = make_model(rustmc_module)
    observations = np.array([0.2, -0.1, np.nan, 0.4, 0.3, 0.6])
    first = model.fit(observations, chains=2, draws=80, warmup=40, seed=17)
    second = model.fit(observations, chains=2, draws=80, warmup=40, seed=17)

    assert first.chains == 2
    assert first.draws == 80
    assert first.time_count == 6
    assert first.observed_count == 5
    first_samples = first.get_samples_2d()
    second_samples = second.get_samples_2d()
    assert set(first_samples) == {
        "process_variance",
        "observation_variance",
        "process_sd",
        "observation_sd",
        "terminal_level",
    }
    for name, values in first_samples.items():
        assert values.shape == (2, 80)
        np.testing.assert_array_equal(values, second_samples[name])
    np.testing.assert_allclose(
        first_samples["process_sd"] ** 2,
        first_samples["process_variance"],
    )
    np.testing.assert_allclose(
        first_samples["observation_sd"] ** 2,
        first_samples["observation_variance"],
    )
    assert not np.array_equal(
        first_samples["process_variance"][0],
        first_samples["process_variance"][1],
    )


def test_parameter_integrated_forecast_paths_and_intervals(rustmc_module):
    fit = make_model(rustmc_module).fit(
        np.array([0.2, -0.1, np.nan, 0.4, 0.3, 0.6]),
        chains=2,
        draws=100,
        warmup=50,
        seed=19,
    )
    forecast = fit.forecast(steps=5, seed=23)
    repeated = fit.forecast(steps=5, seed=23)

    assert forecast.state_samples.shape == (2, 100, 5)
    assert forecast.observation_samples.shape == (2, 100, 5)
    np.testing.assert_array_equal(forecast.state_samples, repeated.state_samples)
    np.testing.assert_array_equal(
        forecast.observation_samples,
        repeated.observation_samples,
    )
    np.testing.assert_allclose(forecast.state_mean, forecast.state_samples.mean((0, 1)))
    np.testing.assert_allclose(
        forecast.observation_mean,
        forecast.observation_samples.mean((0, 1)),
    )

    lower, upper = forecast.interval(0.95)
    state_lower, state_upper = forecast.state_interval(0.95)
    np.testing.assert_allclose(
        lower,
        np.quantile(forecast.observation_samples, 0.025, axis=(0, 1)),
    )
    np.testing.assert_allclose(
        upper,
        np.quantile(forecast.observation_samples, 0.975, axis=(0, 1)),
    )
    np.testing.assert_allclose(
        state_lower,
        np.quantile(forecast.state_samples, 0.025, axis=(0, 1)),
    )
    np.testing.assert_allclose(
        state_upper,
        np.quantile(forecast.state_samples, 0.975, axis=(0, 1)),
    )
    assert forecast.uncertainty_kind == "parameter_integrated_posterior_predictive"
    assert forecast.interval_kind == "pointwise_equal_tailed"


def test_bayesian_local_level_validation(rustmc_module):
    with pytest.raises(ValueError, match="shape"):
        rustmc_module.InverseGammaPrior(0.0, 1.0)
    with pytest.raises(ValueError, match="scale"):
        rustmc_module.InverseGammaPrior(2.0, -1.0)

    model = make_model(rustmc_module)
    for observations in (
        np.array([], dtype=float),
        np.array([1.0]),
        np.array([1.0, np.nan]),
        np.array([np.nan, np.nan]),
        np.array([1.0, np.inf]),
    ):
        with pytest.raises(ValueError):
            model.fit(observations, chains=1, draws=2, warmup=1)

    fit = model.fit(np.array([0.0, 0.1]), chains=1, draws=4, warmup=2)
    with pytest.raises(ValueError, match="horizon"):
        fit.forecast(0)
    forecast = fit.forecast(2)
    for level in (0.0, 1.0, -0.1, 1.1, np.nan):
        with pytest.raises(ValueError, match="strictly between"):
            forecast.interval(level)
        with pytest.raises(ValueError, match="strictly between"):
            forecast.state_interval(level)


def test_bayesian_fit_arviz_export_preserves_chain_draw(rustmc_module):
    pytest.importorskip("arviz")
    fit = make_model(rustmc_module).fit(
        np.array([0.0, 0.2, 0.1]), chains=2, draws=10, warmup=5, seed=5
    )
    data = fit.to_arviz()
    posterior_group = data.posterior
    posterior = getattr(posterior_group, "dataset", posterior_group)
    observed_group = data.observed_data
    observed = getattr(observed_group, "dataset", observed_group)
    assert posterior["process_sd"].shape == (2, 10)
    assert posterior["observation_sd"].shape == (2, 10)
    np.testing.assert_array_equal(observed["y"].values, [0.0, 0.2, 0.1])
