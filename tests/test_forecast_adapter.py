"""Forecast conversion retains each native family's conditional response draws."""
from types import SimpleNamespace

import numpy as np
import pytest

import rustmc as r


def native_forecast(family):
    ig = r.InverseGammaPrior
    y = np.array([1., 2., 1., 3., 2., 1.])
    options = dict(chains=2, draws=12, warmup=5, seed=70)
    if family == "local":
        result = r.BayesianLocalLevel(ig(3., .1), ig(3., .2)).fit(y, **options).forecast(3)
        expected = result.state_samples
    elif family == "trend":
        result = r.BayesianLocalLinearTrend(ig(3., .1), ig(3., .01), ig(3., .2)).fit(y, **options).forecast(3)
        expected = result.level_samples
    elif family == "seasonal":
        result = r.BayesianSeasonalLocalLevel(3, ig(3., .1), ig(3., .1), ig(3., .2)).fit(y, **options).forecast(3)
        expected = result.level_samples + result.seasonal_samples
    elif family == "ar":
        prior = r.NormalInverseGammaPrior(np.zeros(2), np.eye(2), 3., 1.)
        result = r.BayesianAR(1, prior).fit(y, chains=2, draws=12, seed=70).forecast(3)
        expected = result.conditional_mean_samples
    elif family == "hierarchical":
        model = r.BayesianHierarchicalMean(ig(3., .1), ig(3., .1), ig(3., .2),
                                         population_mean_prior=0., population_variance_prior=4.)
        result = model.fit([y, y+1], [0, 0], program_names=["a", "b"], **options).forecast(3)
        expected = result.state_samples
    else:
        model = r.StructuralModel([r.StructuralComponent.level("level", r.VarianceParameter.fixed(.1), 0., 1.)],
                                  r.VarianceParameter.fixed(.2))
        result = model.fit(y, **options).forecast(3)
        expected = result.mean_samples
    return result, expected


@pytest.mark.parametrize("family", ["local", "trend", "seasonal", "ar", "hierarchical", "structural"])
def test_native_forecast_conditional_means_survive_conversion(family, tmp_path):
    result, expected = native_forecast(family)
    wrapped = r.ForecastDraws.from_result(result, dates=["Jan", "Feb", "Mar"])
    np.testing.assert_array_equal(wrapped.observation_samples, result.observation_samples)
    np.testing.assert_array_equal(wrapped.mean_samples, expected)
    np.testing.assert_allclose(wrapped.interval(.9, kind="mean"), np.quantile(expected, [.05, .95], axis=(0, 1)))
    np.testing.assert_allclose(wrapped.interval(.9, kind="mean", cumulative=True),
                               np.quantile(expected.cumsum(-1), [.05, .95], axis=(0, 1)))
    if family == "hierarchical":
        assert wrapped.series == ("a", "b")
    path = tmp_path / "forecast.npz"
    wrapped.save(path)
    np.testing.assert_array_equal(r.ForecastDraws.load(path).mean_samples, expected)


def test_converting_forecast_draws_preserves_coordinates_and_metadata():
    original = r.ForecastDraws(np.ones((1, 2, 1, 3)), dates=("a", "b", "c"), series=("cell",),
                               metadata={"uncertainty_kind": "test", "custom": 17})
    converted = r.ForecastDraws.from_result(original)
    assert converted.dates == original.dates
    assert converted.series == original.series
    assert converted.metadata == original.metadata
    renamed = r.ForecastDraws.from_result(original, dates=("d", "e", "f"), series=("renamed",))
    assert renamed.dates == ("d", "e", "f")
    assert renamed.series == ("renamed",)


def test_unknown_state_arrays_are_not_assumed_to_be_response_means():
    result = SimpleNamespace(observation_samples=np.ones((1, 3, 2)), state_samples=np.zeros((1, 3, 2)))
    assert r.ForecastDraws.from_result(result).mean_samples is None


def test_session_retains_native_mean_draws():
    model = r.BayesianLocalLevel(r.InverseGammaPrior(3., .1), r.InverseGammaPrior(3., .2))
    session = r.ForecastSession(model, np.arange(5.), fit_kwargs={"chains": 1, "draws": 10, "warmup": 5})
    np.testing.assert_array_equal(session.forecast(3, seed=17).mean_samples,
                                  session.fit.forecast(3, seed=17).state_samples)
