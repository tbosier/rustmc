"""Score invariants across location/scale and explicit missing-value semantics."""
from types import SimpleNamespace

import numpy as np
import pytest

import rustmc as r


@pytest.mark.parametrize("draws", [13, 33, 100, 1000])
def test_crps_preserves_narrow_forecasts_at_large_offsets(draws):
    samples = 1e12 + np.random.default_rng(94).normal(scale=.001, size=draws)
    actual = 1e12
    reference = np.abs(samples - actual).mean() - np.abs(samples[:, None] - samples).mean() / 2
    np.testing.assert_allclose(r.crps(samples, actual), reference, rtol=1e-14)
    np.testing.assert_allclose(r.crps(samples - actual, 0.), reference, rtol=1e-14)


def test_scores_of_exact_large_forecast_are_zero():
    scores = r.score_forecast(np.full((2, 10, 1), 1e308), [1e308])
    for name, value in scores.items():
        np.testing.assert_array_equal(value, [1. if name.startswith("coverage") else 0.])


def test_scores_scale_without_overflowing_intermediate_penalties():
    samples = np.full((10, 1), 5e307)
    scores = r.score_forecast(samples, [-5e307])
    for name in ("bias", "absolute_error", "crps", "wis"):
        np.testing.assert_allclose(scores[name], [1e308], rtol=1e-14)
    assert np.isposinf(scores["squared_error"]).all()


def test_quantiles_and_crps_handle_opposite_extreme_draws():
    # The full sample span overflows, while both scores remain representable.
    samples = np.array([-1., 1.])
    np.testing.assert_allclose(r.crps(samples * 1e308, 0.), 5e307)
    np.testing.assert_allclose(
        r.weighted_interval_score(samples * 1e308, 0.),
        r.weighted_interval_score(samples, 0.) * 1e308,
    )


def test_quantiles_are_bitwise_numpy_linear_quantiles():
    from rustmc.evaluation import _quantiles

    rng = np.random.default_rng(20260924)
    probabilities = np.concatenate([
        [0.0, 0.025, 0.1, 0.25, 0.5, 0.69, 0.75, 0.9, 0.975, 1.0],
        rng.uniform(size=40),
    ])
    for trial in range(300):
        n = int(rng.integers(1, 60))
        scale = 10.0 ** rng.uniform(-300, 300)
        draws = rng.normal(size=(n, 3)) * scale
        if trial % 3 == 0:
            # Many ties, and large draws either side of values near zero.
            draws = np.round(draws / scale) * scale
            draws[: n // 2, 1] = rng.choice([-1e16, 1e16 + 2, 1e-12, 0.0], size=n // 2)
        # Adding zero merges -0.0 into 0.0: NumPy's partition leaves equal
        # signed zeros in no particular order, so only their sign may differ.
        expected = np.quantile(draws, probabilities, axis=0, method="linear") + 0.0
        actual = _quantiles(draws, probabilities) + 0.0
        assert actual.tobytes() == expected.tobytes(), (trial, np.flatnonzero(actual != expected))
    assert _quantiles(np.array([-1e16, 1e16 + 2]), 0.5) == 2.0


@pytest.mark.parametrize("baseline", [False, True])
def test_summary_preserves_infinite_losses_and_omits_only_missing_outcomes(baseline):
    scores = [
        {"loss": np.array([0., 1e308, np.nan])},
        {"loss": np.array([np.inf, 1e308, np.nan])},
    ]
    result = r.BacktestResult(tuple(
        r.BacktestFold(i, np.zeros(3), None, score, score) for i, score in enumerate(scores)
    ))
    np.testing.assert_array_equal(result.summary(baseline=baseline)["loss"], [np.inf, 1e308, np.nan])
    assert np.isposinf(result.summary(baseline=baseline, by_horizon=False)["loss"])


def test_backtest_does_not_hide_finite_forecasts_with_overflowed_squared_loss():
    class Model:
        def fit(self, y, **kwargs):
            value = 0. if len(y) == 2 else 1e160
            return SimpleNamespace(forecast=lambda steps, **kw: SimpleNamespace(
                observation_samples=np.full((1, 10, steps), value)))

    result = r.backtest(Model(), np.zeros(5), horizon=1, origins=[2, 3], baseline_period=None)
    assert not result.errors
    np.testing.assert_array_equal(result.folds[0].scores["squared_error"], [0.])
    assert np.isposinf(result.folds[1].scores["squared_error"]).all()
    assert np.isposinf(result.summary()["squared_error"]).all()


def test_summary_keeps_smallest_representable_scores_and_missing_horizons():
    tiny = np.nextafter(0., 1.)
    score = {"loss": np.array([[tiny, np.nan], [tiny, np.nan]])}
    result = r.BacktestResult(tuple(r.BacktestFold(i, np.zeros((2, 2)), None, score, None) for i in range(3)))
    np.testing.assert_array_equal(result.summary()["loss"], [tiny, np.nan])
    assert result.summary(by_horizon=False)["loss"] == tiny
