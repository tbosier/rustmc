"""Native independent forecasting: identity, isolation and diagnostic semantics."""
import numpy as np
import pytest


def models(rmc):
    prior = rmc.InverseGammaPrior(3.0, 0.4)
    return [
        rmc.BayesianLocalLevel(process_variance_prior=prior, observation_variance_prior=prior),
        rmc.BayesianSeasonalLocalLevel(period=4, level_variance_prior=prior,
                                      seasonal_variance_prior=prior, observation_variance_prior=prior),
        rmc.BayesianLocalLinearTrend(level_variance_prior=prior, slope_variance_prior=prior,
                                    observation_variance_prior=prior),
        rmc.BayesianAR(1, rmc.NormalInverseGammaPrior(np.zeros(2), np.eye(2), 3.0, 0.4)),
    ]


def samples(fit):
    return fit.get_samples() if hasattr(fit, "get_samples") else fit.get_samples_2d()


@pytest.mark.parametrize("model_index", range(4))
def test_batch_identity_reorder_chunk_resume_threads_and_forecasts(rustmc_module, model_index):
    model = models(rustmc_module)[model_index]
    rng = np.random.default_rng(204)
    y = rng.normal(size=16)
    ys = [y, y[:12], y.copy()]
    ids = ["north/α", "south", "same data different ID"]
    kwargs = dict(chains=2, draws=24, warmup=12, seed=451)
    batch = model.fit_batch(ys, ids, threads=1, chunk_size=3, **kwargs)
    reordered = model.fit_batch(ys[::-1], ids[::-1], threads=3, chunk_size=1, **kwargs)
    resumed = model.fit_batch(ys[1:2], ids[1:2], threads=2, **kwargs)
    assert batch.ids == ids
    assert len(batch) == 3
    assert batch.errors == {}
    for cell in ids:
        for name, values in samples(batch[cell]).items():
            np.testing.assert_array_equal(values, samples(reordered[cell])[name])
    for name, values in samples(resumed[ids[1]]).items():
        np.testing.assert_array_equal(values, samples(batch[ids[1]])[name])
    key = next(iter(samples(batch[ids[0]])))
    assert not np.array_equal(samples(batch[ids[0]])[key], samples(batch[ids[2]])[key])
    forecasts = batch.forecast(5, seed=75, threads=3, chunk_size=1)
    other = reordered.forecast(5, seed=75, threads=1, chunk_size=3)
    resumed_fc = resumed.forecast(5, seed=75)
    for cell in ids:
        values = forecasts[cell].observation_samples
        assert values.shape == (2, 24, 5)
        np.testing.assert_array_equal(values, other[cell].observation_samples)
    np.testing.assert_array_equal(forecasts[ids[1]].observation_samples,
                                  resumed_fc[ids[1]].observation_samples)
    assert not np.array_equal(forecasts[ids[0]].observation_samples,
                              forecasts[ids[2]].observation_samples)


def test_ragged_mixed_per_cell_models_errors_and_missing_schedule(rustmc_module):
    all_models = models(rustmc_module)
    y = np.arange(16, dtype=float) / 10
    ragged = [y[:8], y, y[:10], y, [], [1.0, np.inf], "not a series"]
    selected = all_models + [None, None, None]
    ids = ["level", "seasonal", "trend", "ar", "empty", "infinite", "wrong_type"]
    batch = all_models[0].fit_batch(ragged, ids, models=selected, chains=2, draws=12,
                                    warmup=10, errors="collect", threads=2)
    assert len(batch) == 7
    assert set(batch.errors) == set(ids[4:])
    assert batch.results[4:] == [None, None, None]
    assert [batch[cell].time_count for cell in ids[:4]] == [8, 16, 10, 16]
    assert set(batch.diagnostics()) == set(ids)
    assert batch.diagnostics()["empty"] is None
    with pytest.raises(rustmc_module.StateSpaceError, match="empty"):
        batch["empty"]
    with pytest.raises(KeyError):
        batch["unknown"]
    forecast = batch.forecast(3, errors="collect", threads=2)
    assert set(forecast.errors) == set(ids[4:])
    assert forecast.results[4:] == [None, None, None]
    with pytest.raises(rustmc_module.StateSpaceError, match="empty"):
        batch.forecast(3)
    gaps = all_models[0].fit_batch([[0., np.nan, 1., 2.]], ["gaps"], draws=6, warmup=3)
    assert gaps["gaps"].time_count == 4
    assert gaps["gaps"].observed_count == 3


def test_batch_validation_and_cell_numerical_failures(rustmc_module):
    model = models(rustmc_module)[0]
    y = [0., 1., 2.]
    for kwargs, match in [({"threads": 0}, "threads"), ({"chunk_size": 0}, "chunk_size"),
                          ({"errors": "ignore"}, "errors")]:
        with pytest.raises(ValueError, match=match):
            model.fit_batch([y], ["one"], **kwargs)
    with pytest.raises(ValueError, match="unique"):
        model.fit_batch([y, y], ["one", "one"])
    with pytest.raises(ValueError, match="length"):
        model.fit_batch([y], [])
    with pytest.raises(ValueError, match="one entry"):
        model.fit_batch([y], ["one"], models=[])
    with pytest.raises(rustmc_module.StateSpaceError, match="empty"):
        model.fit_batch([[], y], ["empty", "ok"], chunk_size=1)
    batch = model.fit_batch([y, [1e308, -1e308]], ["ok", "overflow"], chains=1,
                            draws=6, warmup=3, errors="collect")
    assert batch["ok"].sampler_stats["numerical_failures"] == 0
    assert "numerical" in batch.errors["overflow"].lower()
    empty = model.fit_batch([], [], draws=6)
    assert len(empty) == 0
    assert empty.errors == {}


@pytest.mark.parametrize("model_index", range(4))
def test_diagnostics_and_sampler_metadata(rustmc_module, model_index):
    model = models(rustmc_module)[model_index]
    y = np.random.default_rng(813).normal(size=16)
    kwargs = dict(chains=2, draws=40, seed=154)
    if model_index != 3:
        kwargs["warmup"] = 20
    fit = model.fit(y, **kwargs)
    diagnostics = {p["name"]: p for p in fit.diagnostics()}
    assert len(diagnostics) == [3, 7, 5, 3][model_index]
    for item in diagnostics.values():
        assert item["ess_bulk"] > 0
        assert item["ess_tail"] > 0
        assert item["r_hat"] > 0
        assert item["mcse_mean"] >= 0
    stats = fit.sampler_stats
    assert stats["divergences"] is None
    assert stats["acceptance_rate"] is None
    assert stats["numerical_failures"] == 0
    assert stats["independent_chain_comparison"]
    assert "ess_bulk" in fit.summary()
    assert "Mean accept rate" not in fit.summary()
    short_kwargs = dict(chains=1, draws=3)
    if model_index != 3:
        short_kwargs["warmup"] = 1
    short = model.fit(y, **short_kwargs)
    assert not short.sampler_stats["independent_chain_comparison"]
    assert not short.sampler_stats["diagnostics_available"]
    for item in short.diagnostics():
        for key in ["r_hat", "ess_bulk", "ess_tail", "mcse_mean"]:
            assert item[key] is None


def test_diagnostics_match_arviz_for_identical_retained_chains(rustmc_module):
    az = pytest.importorskip("arviz")
    y = np.random.default_rng(714).normal(size=40)
    fit = models(rustmc_module)[0].fit(y, chains=4, draws=600, warmup=300, seed=518)
    posterior = fit.get_samples_2d()
    for item in fit.diagnostics():
        values = posterior[item["name"]]
        assert item["r_hat"] == pytest.approx(float(az.rhat(values, method="rank")), rel=1e-7)
        assert item["ess_bulk"] == pytest.approx(float(az.ess(values, method="bulk")), rel=.04)
        assert item["ess_tail"] == pytest.approx(float(az.ess(values, method="tail")), rel=.04)
        assert item["mcse_mean"] == pytest.approx(np.asarray(az.mcse(values, method="mean")).item(), rel=.04)


def test_batch_is_exactly_single_fit_with_documented_cell_seed(rustmc_module):
    rmc = rustmc_module
    model = models(rmc)[0]
    y = np.array([0.1, 0.2, np.nan, 0.4, 0.3])
    cell = "program/α"
    kwargs = dict(chains=2, draws=40, warmup=20)
    batch = model.fit_batch([y], [cell], seed=76, **kwargs)
    single = model.fit(y, seed=rmc.forecast_cell_seed(76, cell), **kwargs)
    for name, values in single.get_samples_2d().items():
        np.testing.assert_array_equal(values, batch[cell].get_samples_2d()[name])
    forecast = single.forecast(4, seed=rmc.forecast_cell_seed(84, cell, "forecast"))
    np.testing.assert_array_equal(forecast.observation_samples,
                                  batch.forecast(4, seed=84)[cell].observation_samples)
    with pytest.raises(ValueError, match="domain"):
        rmc.forecast_cell_seed(1, cell, "unknown")


@pytest.mark.parametrize("model_index", range(3))
def test_regression_batches_ragged_designs_priors_diagnostics_and_paths(rustmc_module, model_index):
    rmc = rustmc_module
    model = models(rmc)[model_index]
    ys = [np.linspace(0.1, 0.8, 8), np.linspace(-0.2, 0.7, 10)]
    xs = [np.arange(8, dtype=float)[:, None] / 8,
          np.column_stack([np.ones(10), np.arange(10, dtype=float) / 10])]
    priors = [rmc.GaussianCoefficientPrior(np.zeros(1), np.eye(1)),
              rmc.GaussianCoefficientPrior(np.zeros(2), np.eye(2) * 2)]
    ids = ["linear", "intercept+linear"]
    kwargs = dict(chains=2, draws=16, warmup=8, seed=64)
    batch = model.fit_batch(ys, ids, exog=xs, coefficient_priors=priors, threads=2, **kwargs)
    reordered = model.fit_batch(ys[::-1], ids[::-1], exog=xs[::-1], coefficient_priors=priors[::-1],
                                  threads=1, chunk_size=1, **kwargs)
    for cell, prior, y, x in zip(ids, priors, ys, xs):
        single = model.fit(y, exog=x, coefficient_prior=prior, chains=2, draws=16, warmup=8,
                           seed=rmc.forecast_cell_seed(64, cell))
        for name, values in batch[cell].get_samples_2d().items():
            np.testing.assert_array_equal(values, reordered[cell].get_samples_2d()[name])
            np.testing.assert_array_equal(values, single.get_samples_2d()[name])
        names = {p["name"] for p in batch[cell].diagnostics()}
        assert "coefficient[0]" in names
        assert "terminal_state[0]" in names
        assert "observation_variance" in names
        assert batch[cell].sampler_stats["divergences"] is None
        assert "joint conjugate Gibbs" in batch[cell].summary()
    future = [np.array([[1.], [1.1], [1.2]]), np.array([[1., 1.], [1., 1.1], [1., 1.2]])]
    forecast = batch.forecast(3, exog=future, seed=87, threads=2)
    reversed_forecast = reordered.forecast(3, exog=future[::-1], seed=87, threads=1)
    for cell in ids:
        np.testing.assert_array_equal(forecast[cell].observation_samples,
                                      reversed_forecast[cell].observation_samples)
        np.testing.assert_allclose(forecast[cell].cumulative_observation_samples,
                                  np.cumsum(forecast[cell].observation_samples, axis=-1))
    missing = batch.forecast(3, errors="collect")
    assert set(missing.errors) == set(ids)
    assert all("future exog" in error for error in missing.errors.values())
    malformed = batch.forecast(3, exog=[future[0], [[1.]]], errors="collect")
    assert set(malformed.errors) == {ids[1]}


def test_regression_batch_validation_is_per_cell(rustmc_module):
    rmc = rustmc_module
    model = models(rmc)[0]
    y = [0., .1, .2]
    x = np.ones((3, 1))
    prior = rmc.GaussianCoefficientPrior(np.zeros(1), np.eye(1))
    batch = model.fit_batch([y]*5, ["ok", "no_prior", "no_x", "bad_shape", "bad_values"],
                            exog=[x, x, None, [[1.]], [[float("nan")]]*3],
                            coefficient_priors=[prior, None, prior, prior, prior],
                            draws=8, warmup=4, errors="collect")
    assert set(batch.errors) == {"no_prior", "no_x", "bad_shape", "bad_values"}
    assert batch["ok"].time_count == 3
    with pytest.raises(ValueError, match="exog must have one"):
        model.fit_batch([y], ["ok"], exog=[])
    with pytest.raises(ValueError, match="coefficient_priors must have one"):
        model.fit_batch([y], ["ok"], coefficient_priors=[])
    plain = model.fit_batch([y], ["plain"], draws=8, warmup=4)
    assert "without regression" in plain.forecast(2, exog=[[[1.], [2.]]], errors="collect").errors["plain"]


def test_huge_batch_allocations_return_errors_instead_of_panicking(rustmc_module):
    model = models(rustmc_module)[0]
    huge = model.fit_batch([[1., 2., 3.]], ["huge"], chains=1, draws=2**61,
                           warmup=0, errors="collect")
    assert "allocation limit" in huge.errors["huge"]
    tiny = model.fit_batch([[1., 2., 3.]], ["tiny"], chains=1, draws=8, warmup=0)
    bad_forecast = tiny.forecast(2**61, errors="collect")
    assert "allocation limit" in bad_forecast.errors["tiny"]
