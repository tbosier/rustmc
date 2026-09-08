"""Sparse amount inference: support, uncertainty, independent references and paths."""
import numpy as np
import pytest


def model(rmc, **kwargs):
    return rmc.BayesianHurdleLogNormal(
        rmc.InverseGammaPrior(4.0, 0.03),
        rmc.InverseGammaPrior(4.0, 0.3),
        initial_log_level=1.0,
        initial_variance=0.2,
        **kwargs,
    )


def test_zero_history_retains_occurrence_uncertainty_and_prior_severity(rustmc_module):
    fit = model(rustmc_module).fit(np.zeros(18), chains=2, draws=1200, seed=9)
    assert fit.observed_count == 18
    assert fit.positive_count == 0
    assert fit.severity_informed_by_data is False
    samples = fit.get_samples_2d()
    assert samples["payment_probability"].shape == (2, 1200)
    assert abs(samples["payment_probability"].mean() - 1 / 20) < 0.004
    assert np.all((samples["payment_probability"] > 0) & (samples["payment_probability"] < 1))
    assert np.all(samples["process_variance"] <= 1.0)
    assert np.all(samples["observation_variance"] <= 4.0)
    forecast = fit.forecast(steps=6, seed=91)
    y = forecast.observation_samples
    assert y.shape == (2, 1200, 6)
    assert np.isfinite(y).all() and np.all(y >= 0)
    assert 0.935 < (y == 0).mean() < 0.965
    assert np.any(y > 0)
    np.testing.assert_allclose(
        forecast.mean_samples,
        samples["payment_probability"][..., None] * forecast.positive_mean_samples,
    )
    assert fit.sampler_stats()["divergences"] is None
    assert fit.sampler_stats()["acceptance_rate"] is None
    assert "no positive observations" in fit.summary()


def test_positive_amounts_missing_steps_and_cumulative_paths(rustmc_module):
    y = np.array([0.0, 0.0, 2.5, np.nan, 0.0, 4.0, 0.0, 0.0])
    first = model(rustmc_module).fit(y, chains=2, draws=150, warmup=60, seed=8)
    repeat = model(rustmc_module).fit(y, chains=2, draws=150, warmup=60, seed=8)
    assert (first.time_count, first.observed_count, first.positive_count) == (8, 7, 2)
    assert first.severity_informed_by_data
    for key, value in first.get_samples_2d().items():
        np.testing.assert_array_equal(value, repeat.get_samples_2d()[key])
    forecast = first.forecast(5, seed=7)
    np.testing.assert_array_equal(forecast.observation_samples, first.forecast(5, seed=7).observation_samples)
    cumulative = np.cumsum(forecast.observation_samples, axis=2)
    np.testing.assert_array_equal(forecast.cumulative_observation_samples, cumulative)
    np.testing.assert_allclose(forecast.interval(), np.quantile(forecast.observation_samples, [0.025, 0.975], axis=(0, 1)))
    np.testing.assert_allclose(forecast.cumulative_interval(), np.quantile(cumulative, [0.025, 0.975], axis=(0, 1)))
    np.testing.assert_allclose(forecast.mean_interval(), np.quantile(forecast.mean_samples, [0.025, 0.975], axis=(0, 1)))
    np.testing.assert_allclose(forecast.mean, forecast.mean_samples.mean(axis=(0, 1)))
    names = {item["name"] for item in first.diagnostics()}
    assert names == set(first.get_samples_2d())


def test_one_positive_with_weak_history_is_supported(rustmc_module):
    fit = model(rustmc_module).fit(np.array([0.0, np.nan, 3.0, 0.0]), chains=2, draws=30, warmup=20)
    assert np.isfinite(fit.forecast(2).observation_samples).all()


def test_density_matches_independent_lognormal_formula(rustmc_module):
    logp = rustmc_module.hurdle_lognormal_logp
    assert logp(0.0, 0.25, 1.0, 0.4) == pytest.approx(np.log(0.75))
    y, p, mu, var = 3.0, 0.25, 1.0, 0.4
    expected = np.log(p) - np.log(y) - 0.5 * (np.log(2 * np.pi * var) + (np.log(y) - mu)**2 / var)
    assert logp(y, p, mu, var) == pytest.approx(expected)
    assert logp(0.0, 1.0, 0.0, 1.0) == -np.inf
    assert logp(1.0, 0.0, 0.0, 1.0) == -np.inf


@pytest.mark.parametrize("y", [[], [np.nan], [-0.1], [np.inf], [-np.inf]])
def test_invalid_amounts(rustmc_module, y):
    with pytest.raises(ValueError):
        model(rustmc_module).fit(np.asarray(y), draws=2, chains=1)


def test_bounds_and_counts_are_validated_without_panics(rustmc_module):
    for key in ("process_variance_upper", "observation_variance_upper"):
        for value in (0.0, -1.0, np.nan, np.inf):
            with pytest.raises(ValueError):
                model(rustmc_module, **{key: value})
    with pytest.raises(ValueError):
        model(rustmc_module).fit(np.array([0.0]), draws=2**61)
    fit = model(rustmc_module).fit(np.array([0.0]), draws=2, chains=1)
    for horizon in (0, 2**61):
        with pytest.raises(ValueError):
            fit.forecast(horizon)
    assert all(item["r_hat"] is None for item in fit.diagnostics())
    with pytest.raises(ValueError):
        fit.forecast(1).interval(1.0)


def test_bounded_variance_priors_are_used_in_fit(rustmc_module):
    fit = model(rustmc_module, process_variance_upper=0.012, observation_variance_upper=0.12).fit(
        np.array([0.0, 2.0, 0.0, 3.0]), draws=100, chains=2, warmup=30,
    )
    samples = fit.get_samples_2d()
    assert np.all((samples["process_variance"] > 0) & (samples["process_variance"] <= 0.012))
    assert np.all((samples["observation_variance"] > 0) & (samples["observation_variance"] <= 0.12))


def test_arviz_export_when_installed(rustmc_module):
    pytest.importorskip("arviz")
    fit = model(rustmc_module).fit(np.array([0., 2., 0., 3.]), chains=2, draws=20, warmup=10)
    exported = fit.to_arviz()
    np.testing.assert_array_equal(exported.posterior["payment_probability"], fit.get_samples_2d()["payment_probability"])
