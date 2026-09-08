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


def test_hurdle_batches_preserve_sparse_cells_seeds_and_coherent_forecasts(rustmc_module):
    rmc = rustmc_module
    sparse = model(rmc)
    histories = [np.zeros(9), np.array([0., np.nan, 2.5, 0.]), np.zeros(9)]
    ids = ["never/α", "one-positive", "same-data/different-id"]
    kwargs = dict(chains=2, draws=80, warmup=30, thin=2, seed=705)
    batch = sparse.fit_batch(histories, ids, threads=1, chunk_size=3, **kwargs)
    reverse = sparse.fit_batch(histories[::-1], ids[::-1], threads=3, chunk_size=1, **kwargs)
    resumed = sparse.fit_batch(histories[1:2], ids[1:2], threads=2, **kwargs)
    assert batch.ids == ids and batch.errors == {}
    assert not batch[ids[0]].severity_informed_by_data
    assert batch[ids[1]].severity_informed_by_data
    assert (batch[ids[1]].time_count, batch[ids[1]].observed_count) == (4, 3)
    for cell, y in zip(ids, histories):
        single = sparse.fit(y, chains=2, draws=80, warmup=30, thin=2,
                             seed=rmc.forecast_cell_seed(705, cell))
        for name, values in batch[cell].get_samples_2d().items():
            np.testing.assert_array_equal(values, reverse[cell].get_samples_2d()[name])
            np.testing.assert_array_equal(values, single.get_samples_2d()[name])
        assert batch[cell].sampler_stats()["divergences"] is None
    for name, values in resumed[ids[1]].get_samples_2d().items():
        np.testing.assert_array_equal(values, batch[ids[1]].get_samples_2d()[name])
    assert not np.array_equal(batch[ids[0]].get_samples_2d()["payment_probability"],
                              batch[ids[2]].get_samples_2d()["payment_probability"])
    assert set(batch.diagnostics()) == set(ids)
    assert {p["name"] for p in batch.diagnostics()[ids[0]]} == set(batch[ids[0]].get_samples_2d())
    future = batch.forecast(6, seed=871, threads=3, chunk_size=1)
    reverse_future = reverse.forecast(6, seed=871, threads=1)
    resumed_future = resumed.forecast(6, seed=871)
    for cell in ids:
        paths = future[cell].observation_samples
        assert np.isfinite(paths).all() and np.all(paths >= 0)
        np.testing.assert_array_equal(paths, reverse_future[cell].observation_samples)
        np.testing.assert_array_equal(paths, batch[cell].forecast(
            6, seed=rmc.forecast_cell_seed(871, cell, "forecast")).observation_samples)
        np.testing.assert_array_equal(np.cumsum(paths, axis=-1), future[cell].cumulative_observation_samples)
        np.testing.assert_allclose(future[cell].mean_samples,
                                  batch[cell].get_samples_2d()["payment_probability"][..., None]
                                  * future[cell].positive_mean_samples)
    np.testing.assert_array_equal(future[ids[1]].observation_samples,
                                  resumed_future[ids[1]].observation_samples)


def test_hurdle_batches_use_per_cell_priors_and_mix_with_regression(rustmc_module):
    rmc = rustmc_module
    sparse = model(rmc)
    other = model(rmc, occurrence_alpha=3., occurrence_beta=2., process_variance_upper=.02)
    zero_batch = sparse.fit_batch([np.zeros(4), np.zeros(4)], ["default", "other"],
                                   models=[None, other], chains=2, draws=2000, seed=174, threads=2)
    # Independent analytic Beta posterior references establish per-cell prior routing.
    for cell, alpha, beta in [("default", 1., 5.), ("other", 3., 6.)]:
        p = zero_batch[cell].get_samples_2d()["payment_probability"]
        assert p.mean() == pytest.approx(alpha / (alpha + beta), abs=.01)
        assert p.var() == pytest.approx(alpha * beta / ((alpha+beta)**2 * (alpha+beta+1)), rel=.12)
    assert np.all(zero_batch["other"].get_samples_2d()["process_variance"] <= .02)
    prior = rmc.InverseGammaPrior(3., .1)
    gaussian = rmc.BayesianLocalLevel(prior, prior)
    coefficient_prior = rmc.GaussianCoefficientPrior(np.zeros(1), np.eye(1))
    y = [np.zeros(4), np.array([0., .2, .4, .6])]
    x = [None, np.arange(4, dtype=float)[:, None]]
    mixed = sparse.fit_batch(y, ["sparse", "regression"], models=[None, gaussian],
                             exog=x, coefficient_priors=[None, coefficient_prior],
                             chains=2, draws=24, warmup=12, threads=2)
    assert isinstance(mixed["sparse"], rmc.BayesianHurdleLogNormalFit)
    assert isinstance(mixed["regression"], rmc.BayesianRegressionFit)
    prediction = mixed.forecast(2, exog=[None, np.array([[4.], [5.]])], threads=2)
    assert prediction["sparse"].observation_samples.shape == (2, 24, 2)
    assert prediction["regression"].observation_samples.shape == (2, 24, 2)
    # The ordinary model's entry point must also recognize a hurdle per-cell model.
    from_gaussian = gaussian.fit_batch([np.zeros(3)], ["hurdle"], models=[sparse], draws=8, warmup=2)
    assert isinstance(from_gaussian["hurdle"], rmc.BayesianHurdleLogNormalFit)


def test_hurdle_batch_errors_exog_and_allocation_guards(rustmc_module):
    rmc = rustmc_module
    sparse = model(rmc)
    batch = sparse.fit_batch([[0.], [0., -1.], [np.nan]], ["ok", "negative", "missing"],
                             chains=1, draws=8, warmup=2, errors="collect")
    assert set(batch.errors) == {"negative", "missing"}
    assert batch.results[1:] == [None, None]
    assert set(batch.forecast(2, errors="collect").errors) == {"negative", "missing"}
    with pytest.raises(ValueError, match="negative"):
        sparse.fit_batch([[0., -1.]], ["negative"], draws=8)
    assert "allocation limit" in sparse.fit_batch([[0.]], ["big"], draws=2**61,
                                                   errors="collect").errors["big"]
    assert "allocation limit" in batch.forecast(2**61, errors="collect").errors["ok"]
    coefficient_prior = rmc.GaussianCoefficientPrior(np.zeros(1), np.eye(1))
    for kwargs in [dict(exog=[[[1.]]]), dict(coefficient_priors=[coefficient_prior]),
                   dict(exog=[[[1.]]], coefficient_priors=[coefficient_prior])]:
        invalid = sparse.fit_batch([[0.]], ["unsupported"], draws=8, errors="collect", **kwargs)
        assert "hurdle exog is unsupported" in invalid.errors["unsupported"]
    plain = sparse.fit_batch([[0.]], ["plain"], chains=1, draws=8)
    assert "hurdle exog is unsupported" in plain.forecast(1, exog=[[[1.]]], errors="collect").errors["plain"]
    overflow = rmc.BayesianHurdleLogNormal(rmc.InverseGammaPrior(4., .03),
                                         rmc.InverseGammaPrior(4., .3), initial_log_level=1000.)
    numerical = sparse.fit_batch([[0.], [0.]], ["ok", "overflow"], models=[None, overflow],
                                  chains=1, draws=8, threads=2)
    forecasts = numerical.forecast(1, errors="collect")
    assert set(forecasts.errors) == {"overflow"}
    assert "overflow" in forecasts.errors["overflow"].lower()
