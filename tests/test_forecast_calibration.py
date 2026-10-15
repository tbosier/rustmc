"""Calibration diagnostics: PIT values, coverage tables, sharpness and point accuracy.

Every distributional claim is checked on seeded draws with a known predictive
distribution, with the tolerance stated as a number of binomial standard errors.
"""
from types import SimpleNamespace

import numpy as np
import pytest

import rustmc as r
from rustmc.evaluation import CALIBRATION_LEVELS


def _bin_bound(total, bins, sigmas=5.0):
    """Largest deviation of one bin from total/bins allowed at `sigmas` binomial SEs."""
    p = 1.0 / bins
    return sigmas * np.sqrt(total * p * (1 - p))


def _gaussian_forecasts(n_targets, forecast_sd, truth_sd, draws=200, seed=0):
    rng = np.random.default_rng(seed)
    samples = rng.normal(0.0, forecast_sd, size=(draws, n_targets))
    actual = rng.normal(0.0, truth_sd, size=n_targets)
    return samples, actual


def test_pit_of_the_true_predictive_is_uniform():
    targets, bins = 20_000, 10
    samples, actual = _gaussian_forecasts(targets, 1.0, 1.0, seed=11)
    values = r.pit(samples, actual)
    assert values.shape == (targets,) and np.all((values >= 0) & (values < 1))
    counts = r.pit_histogram(values, bins)
    assert counts.sum() == targets
    # Signed, so that counts - expected cannot wrap around.
    assert np.issubdtype(counts.dtype, np.signedinteger)
    assert np.abs(counts - targets / bins).max() < _bin_bound(targets, bins)
    # The PIT of U(0, 1) has mean 1/2 and variance 1/12.
    assert abs(values.mean() - 0.5) < 5 * np.sqrt(1 / 12 / targets)
    assert abs(values.var() - 1 / 12) < 0.003
    # With continuous draws the plain PIT is uniform on the grid k/n too.
    plain = r.pit_histogram(r.pit(samples, actual, randomized=False), bins)
    assert np.abs(plain - targets / bins).max() < _bin_bound(targets, bins) + targets / 201


@pytest.mark.parametrize("forecast_sd, truth_sd", [(1.0, 2.0), (2.0, 1.0)])
def test_miscalibrated_spread_shows_in_pit_and_every_coverage_level(forecast_sd, truth_sd):
    targets, bins = 20_000, 10
    samples, actual = _gaussian_forecasts(targets, forecast_sd, truth_sd, seed=12)
    counts = r.pit_histogram(r.pit(samples, actual), bins)
    table = r.calibration_table(samples, actual, by_horizon=False)
    edges, middle = counts[[0, -1]], counts[4:6]
    expected = targets / bins
    if forecast_sd < truth_sd:
        # Overconfident: outcomes land in the tails too often, a U shape.
        assert np.all(edges > 2 * expected) and np.all(middle < 0.7 * expected)
        assert np.all(table["coverage"] < table["level"])
        assert np.all(table["coverage_upper"] < table["level"])
    else:
        # Underconfident: outcomes crowd the centre, a hump.
        assert np.all(edges < 0.3 * expected) and np.all(middle > 1.5 * expected)
        assert np.all(table["coverage"] > table["level"])
        assert np.all(table["coverage_lower"] > table["level"])
    # Sharpness: the 90% interval is about 2 * 1.645 forecast standard deviations.
    ninety = list(CALIBRATION_LEVELS).index(0.9)
    np.testing.assert_allclose(table["width"][ninety], 2 * 1.6448536 * forecast_sd, rtol=0.02)
    assert table["count"] == targets


def test_randomized_pit_is_uniform_for_poisson_counts_and_plain_pit_is_not():
    targets, bins, rate = 20_000, 10, 3.0
    rng = np.random.default_rng(13)
    samples = rng.poisson(rate, size=(100, targets)).astype(float)
    actual = rng.poisson(rate, size=targets).astype(float)
    randomized = r.pit_histogram(r.pit(samples, actual, seed=5), bins)
    plain = r.pit_histogram(r.pit(samples, actual, randomized=False), bins)
    bound = _bin_bound(targets, bins)
    assert np.abs(randomized - targets / bins).max() < bound
    # The fraction of draws at or below a count jumps by the probability of
    # each count, so the plain PIT piles onto a few values and its histogram is
    # far from flat.
    assert np.abs(plain - targets / bins).max() > 4 * bound


def test_randomized_pit_is_seeded_and_ignores_missing_outcomes():
    samples = np.tile(np.arange(5.0), (2, 4, 1)).transpose(0, 2, 1)  # (chain, draw, 4)
    actual = np.array([2.0, np.nan, 2.0, 7.0])
    first = r.pit(samples, actual, seed=3)
    np.testing.assert_array_equal(first, r.pit(samples, actual, seed=3))
    assert not np.array_equal(first[[0, 2]], r.pit(samples, actual, seed=4)[[0, 2]])
    assert np.isnan(first[1])
    # 10 draws, 4 below 2 and 2 equal: the value lies in [4/11, 7/11).
    assert 4 / 11 <= first[0] < 7 / 11
    np.testing.assert_array_equal(r.pit(samples, actual, randomized=False)[[0, 3]], [0.6, 1.0])
    assert r.pit_histogram(first, 2).sum() == 3


def _numpy_calibration(flat, actual, levels):
    """Coverage, width and Wilson bounds computed independently with NumPy."""
    observed = ~np.isnan(actual)
    coverage, width, lower, upper = [], [], [], []
    z = 1.959963984540054
    for level in levels:
        lo, hi = np.quantile(flat, [(1 - level) / 2, (1 + level) / 2], axis=0)
        hits = ((actual >= lo) & (actual <= hi))[observed]
        n, p = hits.size, hits.mean()
        coverage.append(p)
        width.append((hi - lo)[observed].mean())
        centre = (p + z * z / (2 * n)) / (1 + z * z / n)
        half = z / (1 + z * z / n) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        lower.append(max(centre - half, 0.0))
        upper.append(min(centre + half, 1.0))
    return np.array(coverage), np.array(width), np.array(lower), np.array(upper)


def test_coverage_and_width_match_an_independent_numpy_computation():
    rng = np.random.default_rng(14)
    samples = rng.gamma(2.0, 1.5, size=(2, 60, 3, 4))  # (chain, draw, series, horizon)
    actual = rng.gamma(2.0, 1.5, size=(3, 4))
    actual[1, 2] = np.nan
    levels = (0.1, 0.5, 0.8, 0.95)
    table = r.calibration_table(samples, actual, levels, by_horizon=True)
    flat = samples.reshape(-1, 3, 4)
    for h in range(4):
        coverage, width, lower, upper = _numpy_calibration(flat[..., h], actual[:, h], levels)
        np.testing.assert_array_equal(table["coverage"][h], coverage)
        np.testing.assert_allclose(table["width"][h], width, rtol=1e-12)
        np.testing.assert_allclose(table["coverage_lower"][h], lower, rtol=1e-12)
        np.testing.assert_allclose(table["coverage_upper"][h], upper, rtol=1e-12)
    np.testing.assert_array_equal(table["count"], [3, 3, 2, 3])
    pooled = r.calibration_table(samples, actual, levels, by_horizon=False)
    coverage, width, _, _ = _numpy_calibration(flat.reshape(120, -1), actual.ravel(), levels)
    np.testing.assert_array_equal(pooled["coverage"], coverage)
    np.testing.assert_allclose(pooled["width"], width, rtol=1e-12)
    assert pooled["count"] == 11


def test_point_accuracy_matches_hand_computed_values():
    actual = np.array([10.0, 0.0, 5.0])
    point = np.array([12.0, 1.0, 5.0])  # errors 2, 1, 0
    samples = np.repeat(point[None], 3, axis=0)
    training = np.array([1.0, 3.0, 2.0, 6.0])  # differences 2, 1, 4; two apart 1, 3
    one = r.point_accuracy(samples, actual, training=training, by_horizon=False)
    assert one["count"] == 3
    np.testing.assert_allclose(one["mae"], 1.0, rtol=1e-15)
    np.testing.assert_allclose(one["rmse"], np.sqrt(5 / 3), rtol=1e-15)
    np.testing.assert_allclose(one["wape"], 3 / 15, rtol=1e-15)
    np.testing.assert_allclose(one["mase"], 1.0 / (7 / 3), rtol=1e-15)
    two = r.point_accuracy(samples, actual, training=training, period=2, by_horizon=False)
    np.testing.assert_allclose(two["mase"], 1.0 / 2.0, rtol=1e-15)
    by_step = r.point_accuracy(samples, actual, training=training)
    np.testing.assert_allclose(by_step["mae"], [2.0, 1.0, 0.0])
    np.testing.assert_allclose(by_step["wape"], [0.2, np.inf, 0.0])
    assert "mase" not in r.point_accuracy(samples, actual)
    # The median of 0, 0, 9 is 0 and the mean is 3.
    skewed = np.array([[0.0], [0.0], [9.0]])
    assert r.point_accuracy(skewed, [0.0], point="median", by_horizon=False)["mae"] == 0.0
    assert r.point_accuracy(skewed, [0.0], by_horizon=False)["mae"] == 3.0


def test_panel_mase_scales_each_series_by_its_own_history():
    training = np.array([[0.0, 2.0, 4.0], [0.0, 10.0, 20.0]])  # scales 2 and 10
    actual = np.array([[1.0, np.nan], [1.0, 1.0]])
    samples = np.broadcast_to(np.array([[3.0, 0.0], [11.0, 21.0]]), (5, 2, 2))
    result = r.point_accuracy(samples, actual, training=training, by_horizon=False)
    # Scaled errors 2/2, 10/10, 20/10; the NaN outcome is skipped.
    np.testing.assert_allclose(result["mase"], (1.0 + 1.0 + 2.0) / 3, rtol=1e-15)
    assert result["count"] == 3


def test_point_errors_agree_with_score_forecast_where_summarising_first_would_round():
    # The mean draw 1e16 + 1 is not representable; the mean error 1 is.
    samples = np.array([[1e16], [1e16 + 2.0]])
    assert r.score_forecast(samples, [1e16])["absolute_error"][0] == 1.0
    for point in ("mean", "median"):
        result = r.point_accuracy(samples, [1e16], point=point, by_horizon=False)
        assert (result["mae"], result["rmse"]) == (1.0, 1.0)


def test_coverage_of_a_forecast_with_atoms_exceeds_the_level_even_when_exact():
    # The documented limitation: a forecast that is always zero, of outcomes
    # that are always zero, covers at every level; its randomised PIT does not
    # pile up.
    zeros = np.zeros((100, 500))
    table = r.calibration_table(zeros, np.zeros(500), by_horizon=False)
    np.testing.assert_array_equal(table["coverage"], 1.0)
    counts = r.pit_histogram(r.pit(zeros, np.zeros(500), seed=8))
    assert np.abs(counts - 50).max() < _bin_bound(500, 10)


def test_zero_denominators_are_zero_or_infinite_and_missing_scales_are_nan():
    zeros = np.zeros((4, 2))
    exact = r.point_accuracy(zeros, [0.0, 0.0], training=[5.0, 5.0], by_horizon=False)
    assert (exact["wape"], exact["mase"]) == (0.0, 0.0)
    wrong = r.point_accuracy(zeros + 1.0, [0.0, 0.0], training=[5.0, 5.0], by_horizon=False)
    assert np.isposinf(wrong["wape"]) and np.isposinf(wrong["mase"])
    unscaled = r.point_accuracy(zeros, [1.0, 1.0], training=[5.0], by_horizon=False)
    assert np.isnan(unscaled["mase"]) and unscaled["mae"] == 1.0
    missing = r.point_accuracy(zeros, [np.nan, np.nan], by_horizon=False)
    assert missing["count"] == 0 and np.isnan(missing["mae"])


def test_inputs_follow_the_exact_conversion_rule_and_are_validated():
    samples = np.zeros((10, 3))
    with pytest.raises(ValueError, match="bool"):
        r.pit(samples, np.array([True, False, True]))
    with pytest.raises(ValueError, match="finite"):
        r.pit(samples, [0.0, np.inf, 0.0])
    with pytest.raises(ValueError, match="finite"):
        r.calibration_table(samples + np.nan, [0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="actual shape"):
        r.calibration_table(samples, [0.0, 0.0])
    with pytest.raises(ValueError, match="unique"):
        r.calibration_table(samples, [0.0] * 3, levels=(0.5, 0.5))
    with pytest.raises(ValueError, match="between zero and one"):
        r.calibration_table(samples, [0.0] * 3, levels=(0.5, 1.0))
    with pytest.raises(ValueError, match="bins"):
        r.pit_histogram([0.5], 0)
    with pytest.raises(ValueError, match="between zero and one"):
        r.pit_histogram([1.5])
    with pytest.raises(ValueError, match="seed"):
        r.pit(samples, [0.0] * 3, seed=-1)
    with pytest.raises(ValueError, match="training"):
        r.point_accuracy(samples, [0.0] * 3, training=np.zeros((2, 5)))
    with pytest.raises(ValueError, match="point"):
        r.point_accuracy(samples, [0.0] * 3, point="mode")
    with pytest.raises(ValueError, match="period"):
        r.point_accuracy(samples, [0.0] * 3, period=0)
    # Levels follow the exact conversion rule too.
    for bad in (["0.5"], np.array([0.5 + 1j]), np.ma.masked_array([0.5, 0.8], mask=[False, True])):
        with pytest.raises(ValueError, match="levels"):
            r.calibration_table(samples, [0.0] * 3, levels=bad)
    # Sizes too large to allocate, or to pass to the native layer, are refused.
    for bins in (2**62, 2**64):
        with pytest.raises(ValueError, match="bins|safety limit"):
            r.pit_histogram([0.5], bins)
    with pytest.raises(ValueError, match="period"):
        r.point_accuracy(samples, [0.0] * 3, period=2**64)


class _RandomWalkModel:
    """Forecasts a Gaussian random walk from its last value with a chosen step sd.

    With step_sd equal to the data's, these are draws from the true predictive
    distribution, so any miscalibration a test finds belongs to the diagnostics.
    """
    def __init__(self, step_sd, draws=400):
        self.step_sd, self.draws = step_sd, draws

    def fit(self, y, seed):
        rng = np.random.default_rng(seed)
        def forecast(steps, seed):
            steps_drawn = rng.normal(0.0, self.step_sd, size=(1, self.draws, steps))
            return SimpleNamespace(observation_samples=y[-1] + np.cumsum(steps_drawn, axis=-1))
        return SimpleNamespace(forecast=forecast)


def _random_walk(length=600, seed=15):
    return np.cumsum(np.random.default_rng(seed).normal(size=length))


def test_backtest_calibration_by_horizon_separates_true_and_overconfident_forecasts():
    y = _random_walk()
    honest = r.backtest(_RandomWalkModel(1.0), y, horizon=3, initial=20)
    narrow = r.backtest(_RandomWalkModel(0.5), y, horizon=3, initial=20)
    honest_table, narrow_table = honest.calibration(), narrow.calibration()
    assert honest_table["coverage"].shape == (3, len(CALIBRATION_LEVELS))
    np.testing.assert_array_equal(honest_table["count"], [578] * 3)
    # Folds overlap, so allow more than the independent-trial interval would.
    assert np.abs(honest_table["coverage"] - honest_table["level"]).max() < 0.08
    assert np.all(narrow_table["coverage"] < narrow_table["level"])
    # Half the spread gives about half the width at every level and step.
    np.testing.assert_allclose(narrow_table["width"] / honest_table["width"], 0.5, rtol=0.1)
    overall = narrow.calibration(by_horizon=False)
    assert overall["coverage"].shape == (len(CALIBRATION_LEVELS),) and overall["count"] == 3 * 578
    by_step = narrow.pit_histogram(by_horizon=True)
    assert by_step.shape == (3, 10) and np.all(by_step[:, [0, -1]] > 2 * 578 / 10)


def test_backtest_pit_values_are_dependent_across_steps_and_overlapping_folds():
    # Why no uniformity p-value is offered: under the true predictive, the PIT
    # at step 2 shares a shock with step 1 of the same fold and with step 2 of
    # the next fold. For a Gaussian random walk the correlations are about
    # 0.68 and 0.48, not zero.
    result = r.backtest(_RandomWalkModel(1.0, draws=1000), _random_walk(2000), horizon=2, initial=20)
    values = result.pit()
    assert values.shape == (len(result.folds), 2)
    across_steps = np.corrcoef(values[:, 0], values[:, 1])[0, 1]
    across_folds = np.corrcoef(values[:-1, 1], values[1:, 1])[0, 1]
    assert 0.55 < across_steps < 0.8, across_steps
    assert 0.35 < across_folds < 0.6, across_folds


def test_backtest_keeps_its_scores_and_the_new_tables_agree_with_them():
    # Internal consistency only: that scores are unchanged from before this
    # feature rests on the pre-existing tests in test_forecast_workflow.py and
    # test_evaluation_numerics.py, which are not modified.
    y = _random_walk(80)
    result = r.backtest(_RandomWalkModel(1.0), y, horizon=2, initial=40, levels=(0.5, 0.8))
    for fold in result.folds:
        expected = r.score_forecast(fold.samples, fold.actual, (0.5, 0.8))
        assert fold.scores.keys() == expected.keys()
        for key in expected:
            np.testing.assert_array_equal(fold.scores[key], expected[key])
        baseline = r.score_forecast(fold.baseline_samples, fold.actual, (0.5, 0.8))
        for key in baseline:
            np.testing.assert_array_equal(fold.baseline_scores[key], baseline[key])
    table = result.calibration((0.5, 0.8))
    summary = result.summary()
    np.testing.assert_allclose(table["coverage"][:, 0], summary["coverage_0.5"], rtol=1e-14)
    np.testing.assert_allclose(table["width"][:, 1], summary["width_0.8"], rtol=1e-12)
    np.testing.assert_allclose(result.accuracy()["mae"], summary["absolute_error"], rtol=1e-12)
    np.testing.assert_array_equal(result.observations, y)
    assert not result.observations.flags.writeable
    view = result.baseline()
    np.testing.assert_array_equal(view.summary()["crps"], result.summary(baseline=True)["crps"])


def test_backtest_mase_uses_only_data_before_each_origin():
    y = np.array([0.0, 1.0, 3.0, 6.0, 10.0, 15.0])
    class Last:
        def fit(self, train, seed):
            return SimpleNamespace(forecast=lambda steps, seed: SimpleNamespace(
                observation_samples=np.full((1, 2, steps), train[-1])))
    result = r.backtest(Last(), y, horizon=1, origins=[3, 5], baseline_period=None)
    # Origin 3: error 3, scale mean(1, 2) = 1.5. Origin 5: error 5, scale 2.5.
    np.testing.assert_allclose(result.accuracy(by_horizon=False)["mase"], (3 / 1.5 + 5 / 2.5) / 2)
    stripped = r.BacktestResult(result.folds)
    assert "mase" not in stripped.accuracy()


def test_compare_backtests_scores_common_origins_side_by_side():
    y = _random_walk(300)
    honest = r.backtest(_RandomWalkModel(1.0), y, horizon=2, initial=20)
    narrow = r.backtest(_RandomWalkModel(0.5), y, horizon=2, initial=20)
    failing = [f if f.origin != 25 else r.BacktestFold(25, f.actual, None, None, None, error="boom")
               for f in narrow.folds]
    narrow = r.BacktestResult(tuple(failing), narrow.observations)
    comparison = r.compare_backtests({"honest": honest, "narrow": narrow, "naive": honest.baseline()})
    assert 25 not in comparison.origins and len(comparison.origins) == len(honest.folds) - 1
    assert comparison.excluded == {"honest": (25,), "narrow": (25,), "naive": (25,)}
    honest_row, narrow_row = comparison.metrics["honest"], comparison.metrics["narrow"]
    for key in ("mae", "rmse", "wape", "mase", "crps", "wis", "coverage_0.8", "width_0.95", "coverage_error"):
        assert key in honest_row and np.ndim(honest_row[key]) == 0
    # The same point forecast in expectation; the spread is what differs.
    assert narrow_row["coverage_0.8"] < 0.65 < honest_row["coverage_0.8"]
    assert narrow_row["coverage_error"] > 3 * honest_row["coverage_error"]
    assert narrow_row["width_0.95"] < honest_row["width_0.95"]
    text = str(comparison)
    assert "honest" in text and "narrow" in text and "coverage_error" in text
    by_step = r.compare_backtests({"honest": honest, "narrow": narrow}, by_horizon=True)
    assert by_step.metrics["honest"]["mae"].shape == (2,)
    assert "mae h=2" in str(by_step)
    # A result without observations has no MASE; it is reported as NaN, not dropped.
    mixed = r.compare_backtests({"with": honest, "without": r.BacktestResult(honest.folds)})
    assert np.isnan(mixed.metrics["without"]["mase"]) and np.isfinite(mixed.metrics["with"]["mase"])
    assert list(mixed.metrics["with"]) == list(mixed.metrics["without"]) and "nan" in str(mixed)
    with pytest.raises(ValueError, match="BacktestResult"):
        r.compare_backtests({"a": honest.folds})
    other = r.backtest(_RandomWalkModel(1.0), y + 1.0, horizon=2, initial=20)
    with pytest.raises(ValueError, match="different outcomes"):
        r.compare_backtests({"a": honest, "b": other})
    nothing = r.BacktestResult(tuple(r.BacktestFold(f.origin, f.actual, None, None, None, error="x")
                                     for f in honest.folds))
    with pytest.raises(ValueError, match="every result"):
        r.compare_backtests({"a": honest, "b": nothing})
    assert nothing.calibration() == {} and nothing.accuracy() == {}
    assert np.isnan(nothing.pit()).all()


def test_compare_backtests_refuses_or_omits_what_would_not_compare_like_with_like():
    y = _random_walk(60)
    wide = r.backtest(_RandomWalkModel(1.0), y, horizon=2, initial=40, levels=(0.5,))
    narrow_levels = r.backtest(_RandomWalkModel(1.0), y, horizon=2, initial=40, levels=(0.95,))
    # WIS scored at different interval levels is a different score.
    mixed = r.compare_backtests({"a": wide, "b": narrow_levels})
    assert "wis" not in mixed.metrics["a"] and "crps" in mixed.metrics["a"]
    # A fold without stored scores would drop out of CRPS but not of the rest.
    unscored = r.BacktestResult(tuple(
        r.BacktestFold(f.origin, f.actual, f.samples, None if i == 0 else f.scores, None)
        for i, f in enumerate(wide.folds)), wide.observations)
    partial = r.compare_backtests({"a": wide, "b": unscored})
    assert "crps" not in partial.metrics["a"] and "mae" in partial.metrics["b"]
    with pytest.raises(ValueError, match="unique"):
        r.compare_backtests({"a": wide}, levels=(0.5, 0.5, 0.8))
    duplicated = r.BacktestResult(wide.folds + wide.folds[:1], wide.observations)
    with pytest.raises(ValueError, match="same origin"):
        r.compare_backtests({"a": duplicated})
    shifted = r.BacktestResult(tuple(
        r.BacktestFold(-1 if i == 0 else f.origin, f.actual, f.samples, f.scores, None)
        for i, f in enumerate(wide.folds)), wide.observations)
    with pytest.raises(ValueError, match="origin"):
        shifted.accuracy()
    ragged = r.BacktestResult((wide.folds[0], r.BacktestFold(99, np.zeros(1), np.zeros((5, 1)), None, None)))
    with pytest.raises(ValueError, match="target shape"):
        ragged.pit()
    # Scalar targets have no horizon axis; every metric is one number.
    scalar = r.BacktestResult(tuple(
        r.BacktestFold(f.origin, f.actual[0], f.samples[..., 0], r.score_forecast(f.samples[..., 0], f.actual[0]), None)
        for f in wide.folds))
    by_step = r.compare_backtests({"s": scalar}, by_horizon=True)
    assert all(np.ndim(value) == 0 for value in by_step.metrics["s"].values())


def test_baseline_view_reports_why_a_baseline_is_missing():
    y = np.arange(6.0)
    result = r.backtest(_RandomWalkModel(1.0), y, horizon=1, origins=[1, 4], baseline_period=2)
    view = result.baseline()
    assert set(view.errors) == {1} and "terminal cycle" in view.errors[1]
    assert view.pit().shape == (2, 1) and np.isnan(view.pit()[0]).all()
    disabled = r.backtest(_RandomWalkModel(1.0), y, horizon=1, origins=[4], baseline_period=None)
    assert disabled.baseline().errors == {4: "no baseline forecast"}


def test_dynamic_poisson_backtest_gives_randomized_pit_for_counts():
    rng = np.random.default_rng(16)
    counts = rng.poisson(np.exp(1.0 + np.cumsum(rng.normal(0.0, 0.05, 40))))[None, :].astype(float)
    model = r.BayesianDynamicPoisson(initial_mean=1.0, process_sd=0.05)
    result = r.backtest(model, counts, horizon=2, origins=[34, 36, 38],
                        fit_kwargs={"chains": 2, "draws": 200, "warmup": 200}, baseline_period=None)
    samples = result.folds[0].samples
    assert np.array_equal(samples, np.round(samples))  # integer-valued draws, so ties
    randomized, plain = result.pit(seed=1), result.pit(randomized=False)
    assert randomized.shape == (3, 1, 2)
    assert np.all((randomized >= 0) & (randomized < 1))
    assert not np.array_equal(randomized, plain)
    np.testing.assert_array_equal(randomized, result.pit(seed=1))
    assert not np.array_equal(randomized, result.pit(seed=2))
    assert result.calibration()["coverage"].shape == (2, len(CALIBRATION_LEVELS))
