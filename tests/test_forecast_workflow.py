"""Forecast workflow contracts: proper scores, temporal splits, paths and artifacts."""
from types import SimpleNamespace

import numpy as np
import pytest

import rustmc as r


def test_scores_match_pairwise_crps_and_deterministic_absolute_error():
    rng = np.random.default_rng(501)
    samples = rng.normal(size=(2, 15, 4))
    actual = np.array([1., 2., -1., 0.])
    flat = samples.reshape(-1, 4)
    reference = np.abs(flat - actual).mean(axis=0) - np.abs(flat[:, None] - flat[None]).mean(axis=(0, 1))/2
    np.testing.assert_allclose(r.crps(samples, actual), reference, atol=1e-14)
    np.testing.assert_allclose(r.weighted_interval_score(np.zeros((20, 1)), [1.], levels=(.8, .95)), 1.)
    np.testing.assert_allclose(r.crps(np.zeros((20, 1)), [1.]), 1.)
    np.testing.assert_allclose(r.crps([0., 1.], .5), .25)
    assert np.isnan(r.score_forecast(samples, [1., np.nan, 0., 2.])["wis"][1])
    with pytest.raises(ValueError, match="unique"):
        r.weighted_interval_score(samples, actual, levels=(.9, .9))


def test_naive_preserves_panel_dependence_and_seasonal_recursion():
    y = np.array([0., 1., 0., 2., 0., 4., 0., 5.])
    np.testing.assert_array_equal(r.seasonal_naive(y, 5, period=2), [0., 5., 0., 5., 0.])
    samples = r.naive_forecast(np.stack([y, 2*y]), 8, period=2, draws=100, seed=9)
    assert samples.shape == (1, 100, 2, 8)
    np.testing.assert_allclose(samples[:, :, 1], 2*samples[:, :, 0])
    with pytest.raises(ValueError, match="difference"):
        r.naive_forecast([1.], 3)


def test_backtest_only_exposes_training_data_and_slices_future_design():
    calls = []
    class Model:
        def fit(self, y, exog, seed):
            calls.append((y.copy(), exog.copy(), seed))
            terminal = y[-1]
            class Fit:
                def forecast(self, steps, exog, seed):
                    np.testing.assert_array_equal(exog[:, 0], np.arange(len(y), len(y)+steps))
                    return SimpleNamespace(observation_samples=np.full((2, 3, steps), terminal))
                def diagnostics(self):
                    return [{"name": "test", "r_hat": 1.0}]
            return Fit()
    factory_inputs = []
    def factory(train):
        factory_inputs.append(train.copy())
        return Model()
    y = np.arange(10, dtype=float)
    first = r.backtest(factory, y, horizon=2, origins=[4, 6], exog=y[:, None])
    repeat = r.backtest(factory, y, horizon=2, origins=[6, 4], exog=y[:, None])
    assert [len(v) for v in factory_inputs] == [4, 6, 6, 4]
    assert calls[0][2] == calls[3][2]
    np.testing.assert_array_equal(first.folds[0].samples, repeat.folds[1].samples)
    np.testing.assert_array_equal(first.summary()["bias"], [-1., -2.])
    assert first.folds[0].diagnostics[0]["r_hat"] == 1.


def test_backtest_collects_failures_and_rejects_leaky_kwargs():
    class Bad:
        def fit(self, y, **kwargs):
            raise ValueError("deliberate failure")
    result = r.backtest(Bad(), np.arange(8.), horizon=2, origins=[3, 5], errors="collect")
    assert set(result.errors) == {3, 5}
    assert result.summary() == {}
    with pytest.raises(ValueError, match="sliced"):
        r.backtest(Bad(), np.arange(8.), horizon=2, origins=[3], fit_kwargs={"exog": np.ones((8, 1))})


def test_draw_archive_and_aggregation_preserve_joint_dependence(tmp_path):
    rng = np.random.default_rng(78)
    x = rng.normal(size=(2, 20, 4))
    panel = np.stack([x, -x], axis=2)
    f = r.ForecastDraws(panel, panel/2, dates=("a", "b", "c", "d"), series=("one", "two"))
    total = f.aggregate()
    np.testing.assert_allclose(total.observation_samples, 0.)
    np.testing.assert_allclose(total.interval(cumulative=True), 0.)
    path = tmp_path / "forecast.npz"
    f.save(path)
    restored = r.ForecastDraws.load(path)
    np.testing.assert_array_equal(restored.observation_samples, panel)
    assert restored.dates == f.dates
    assert restored.series == f.series
    assert not restored.observation_samples.flags.writeable
    with pytest.raises(ValueError, match="conditional mean"):
        r.ForecastDraws(x).interval(kind="mean")


def test_scenarios_mix_whole_paths_and_validate_weights():
    low = r.ForecastDraws(np.ones((1, 10, 3)))
    high = r.ForecastDraws(np.full((1, 10, 3), 10.))
    mix = r.ScenarioForecast({"low": low, "high": high}, {"low": .7, "high": .3}).mixture(draws=10000, seed=67)
    samples = mix.observation_samples
    assert (samples == samples[..., :1]).all()
    assert abs((samples[..., 0] == 10.).mean() - .3) < .02
    with pytest.raises(ValueError, match="sum"):
        r.ScenarioForecast({"a": low}, {"a": .5}).mixture()


def test_session_feature_validation_and_transactional_refit():
    class Model:
        def fit(self, y, exog, **kwargs):
            if not np.isfinite(y).all():
                raise ValueError("bad data")
            return SimpleNamespace(forecast=lambda steps, **kw: SimpleNamespace(
                observation_samples=np.full((1, 5, steps), y.mean())))
    design = r.NamedDesign(np.ones((3, 2)), ("price", "holiday"))
    session = r.ForecastSession(Model(), [1., 2., 3.], exog=design)
    with pytest.raises(ValueError, match="training order"):
        session.forecast(2, exog=r.NamedDesign(np.ones((2, 2)), ("holiday", "price")))
    session.update([4.], exog=r.NamedDesign(np.ones((1, 2)), design.features))
    assert len(session.observations) == 4
    with pytest.raises(ValueError, match="bad data"):
        session.update([np.inf], exog=r.NamedDesign(np.ones((1, 2)), design.features))
    assert len(session.observations) == 4
    f = session.forecast(2, exog=r.NamedDesign(np.ones((2, 2)), design.features), dates=("Jan", "Feb"))
    np.testing.assert_allclose(f.observation_mean, [2.5, 2.5])
    scenarios = r.forecast_scenarios(session, {"base": r.NamedDesign(np.ones((2, 2)), design.features)})
    assert scenarios.forecasts["base"].observation_samples.shape == (1, 5, 2)
    with pytest.raises(ValueError, match="directly"):
        r.ForecastSession(Model(), [1., 2.], fit_kwargs={"exog": np.ones((2, 1))})


def test_missing_baseline_does_not_discard_model_fold():
    class Model:
        def fit(self, y, **kwargs):
            return SimpleNamespace(forecast=lambda steps, **kw: SimpleNamespace(
                observation_samples=np.zeros((1, 3, steps))))
    result = r.backtest(Model(), [0., 1., np.nan, 2., 3.], horizon=2, origins=[3])
    assert result.errors == {}
    assert 3 in result.baseline_errors
    assert "crps" in result.summary()


def test_forecast_arithmetic_checks_range():
    f = r.ForecastDraws(np.full((2, 20, 3), 1e308))
    np.testing.assert_allclose(f.observation_mean, 1e308)
    with pytest.raises(ValueError, match="floating-point"):
        _ = f.cumulative_observation_samples
