"""Probabilistic forecast scores and evaluation at historical forecast origins."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


def _draws(samples: Any, actual: Any) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(actual, dtype=float)
    samples = np.asarray(samples, dtype=float)
    # Accept (sample, *target_shape) or (chain, draw, *target_shape).
    n_axes = samples.ndim - actual.ndim
    if n_axes not in (1, 2) or samples.shape[n_axes:] != actual.shape:
        raise ValueError("samples must have one sample axis or chain/draw axes followed by the actual shape")
    if not samples.size or not np.isfinite(samples).all():
        raise ValueError("predictive draws must be nonempty and finite")
    if np.isinf(actual).any():
        raise ValueError("actual values must be finite or NaN for missing outcomes")
    return samples.reshape((-1,) + actual.shape), actual


def _weighted_difference(a: np.ndarray, b: np.ndarray, weight: Any) -> np.ndarray:
    """Subtract before weighting for precision; scale first only on overflow."""
    with np.errstate(over="ignore", invalid="ignore"):
        difference = a - b
        return np.where(np.isinf(difference), a * weight - b * weight, difference * weight)


def _mean_error(draws: np.ndarray, actual: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        errors = draws - actual
        overflow = np.isinf(errors).any(axis=0)
        errors = np.where(overflow, draws * 0.5 - actual * 0.5, errors)
        scale = np.max(np.abs(errors), axis=0)
        normalized = np.divide(errors, scale, out=np.zeros_like(errors), where=scale != 0)
        return normalized.mean(axis=0) * scale * np.where(overflow, 2.0, 1.0)


def _quantiles(draws: np.ndarray, probabilities: Any) -> np.ndarray:
    """Quantiles over the leading sample axis, shaped ``probabilities.shape + draws.shape[1:]``.

    Uses the native rule behind every forecast interval (linear between order
    statistics, exact on ties, overflow-safe), so a score and a native
    ``interval()`` of the same draws agree exactly.
    """
    from ._rustmc import _empirical_quantiles

    draws = np.asarray(draws, dtype=float)
    positions = np.asarray(probabilities, dtype=float)
    values = _empirical_quantiles(draws.reshape(len(draws), -1), positions.ravel().tolist())
    return values.reshape(positions.shape + draws.shape[1:])


def interval_score(actual: Any, lower: Any, upper: Any, alpha: float = 0.05) -> np.ndarray:
    """Proper score for a central (1-alpha) prediction interval; lower is better."""
    actual, lower, upper = np.broadcast_arrays(
        np.asarray(actual, dtype=float), np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    )
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    if np.any(lower > upper) or not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError("interval bounds must be finite and ordered")
    if np.isinf(actual).any():
        raise ValueError("actual values must be finite or NaN")
    return upper - lower + (2 / alpha) * np.maximum(lower - actual, 0) + (2 / alpha) * np.maximum(actual - upper, 0)


def crps(samples: Any, actual: Any) -> np.ndarray:
    """CRPS of the empirical predictive distribution, with no quadratic pair matrix."""
    draws, actual = _draws(samples, actual)
    ordered = np.sort(draws, axis=0)
    n = len(ordered)
    ranks = np.arange(1, n + 1).reshape((n,) + (1,) * actual.ndim)
    # Equivalent to E|X-y| - E|X-X'|/2, expressed as nonnegative terms.
    # Subtracting two large expectations or summing weighted uncentered draws
    # can erase the score when the forecast is narrow relative to its level.
    weights = np.where(ordered <= actual, 2 * ranks - 1, 2 * (n - ranks) + 1) / (n * n)
    return np.sum(np.abs(_weighted_difference(ordered, actual, weights)), axis=0)


def weighted_interval_score(samples: Any, actual: Any, levels: Sequence[float] = (0.5, 0.8, 0.95)) -> np.ndarray:
    """Standard WIS: median error plus alpha/2-weighted interval scores, divided by K+0.5."""
    draws, actual = _draws(samples, actual)
    levels = np.asarray(levels, dtype=float)
    if levels.ndim != 1 or not levels.size or not np.isfinite(levels).all() or np.any((levels <= 0) | (levels >= 1)):
        raise ValueError("levels must be a nonempty sequence strictly between zero and one")
    if len(np.unique(levels)) != len(levels):
        raise ValueError("interval levels must be unique")
    denominator = len(levels) + 0.5
    score = np.abs(_weighted_difference(actual, _quantiles(draws, 0.5), 0.5 / denominator))
    for level in levels:
        alpha = 1 - level
        lower, upper = _quantiles(draws, (alpha / 2, 1 - alpha / 2))
        # Apply the WIS weights before forming interval penalties: a component
        # interval score may overflow even when the final WIS is representable.
        score += _weighted_difference(upper, lower, alpha / (2 * denominator))
        score += np.maximum(_weighted_difference(lower, actual, 1 / denominator), 0)
        score += np.maximum(_weighted_difference(actual, upper, 1 / denominator), 0)
    return score


def score_forecast(samples: Any, actual: Any, levels: Sequence[float] = (0.5, 0.8, 0.95)) -> dict[str, np.ndarray]:
    """Scores retain target dimensions, including horizon. Missing outcomes score NaN."""
    draws, actual = _draws(samples, actual)
    bias = _mean_error(draws, actual)
    with np.errstate(over="ignore"):
        squared_error = bias**2
    scores = {
        "bias": bias,
        "absolute_error": np.abs(bias),
        "squared_error": squared_error,
        "crps": crps(draws, actual),
        "wis": weighted_interval_score(draws, actual, levels),
    }
    for level in levels:
        lower, upper = _quantiles(draws, ((1-level)/2, (1+level)/2))
        suffix = format(float(level), ".12g")
        scores[f"coverage_{suffix}"] = np.where(np.isnan(actual), np.nan, (actual >= lower) & (actual <= upper)).astype(float)
        scores[f"width_{suffix}"] = np.where(np.isnan(actual), np.nan, upper - lower)
    return scores


def seasonal_naive(observations: Any, steps: int, period: int = 1) -> np.ndarray:
    """Repeat the latest complete seasonal cycle along the final time axis."""
    y = np.asarray(observations, dtype=float)
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if isinstance(period, bool) or not isinstance(period, (int, np.integer)) or period < 1:
        raise ValueError("period must be a positive integer")
    if y.ndim < 1 or y.shape[-1] < period or not np.isfinite(y[..., -period:]).all():
        raise ValueError("seasonal naive requires a complete finite terminal cycle")
    return np.take(y[..., -period:], np.arange(steps) % period, axis=-1)


def naive_forecast(observations: Any, steps: int, *, period: int = 1, draws: int = 1000, seed: int = 42) -> np.ndarray:
    """Seasonal random-walk paths using a bootstrap of observed seasonal differences.

    Returns (1, draw, *series_shape, horizon). Innovations are centered and sampled
    by complete time columns to preserve empirical dependence across panel series.
    This baseline does not integrate uncertainty in the innovation distribution.
    """
    y = np.asarray(observations, dtype=float)
    point = seasonal_naive(y, steps, period)
    if isinstance(draws, bool) or not isinstance(draws, (int, np.integer)) or draws < 1:
        raise ValueError("draws must be a positive integer")
    if y.shape[-1] <= period:
        raise ValueError("probabilistic naive requires at least one seasonal difference")
    residuals = y[..., period:] - y[..., :-period]
    valid = np.isfinite(residuals).reshape((-1, residuals.shape[-1])).all(axis=0)
    residuals = residuals[..., valid]
    if residuals.shape[-1] == 0:
        raise ValueError("no observed seasonal differences")
    residuals = residuals - residuals.mean(axis=-1, keepdims=True)
    rng = np.random.default_rng(seed)
    paths = np.empty((draws,) + point.shape)
    for h in range(steps):
        innovation = np.moveaxis(np.take(residuals, rng.integers(residuals.shape[-1], size=draws), axis=-1), -1, 0)
        paths[..., h] = (point[..., h] if h < period else paths[..., h-period]) + innovation
    return paths[np.newaxis]


# Central-interval levels of a calibration table: 0.05, 0.10, ..., 0.95.
CALIBRATION_LEVELS = tuple(round(0.05 * k, 2) for k in range(1, 20))


def _check_flag(name: str, value: Any) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be True or False")
    return bool(value)


def _check_count(name: str, value: Any, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not minimum <= value < 2**63:
        raise ValueError(f"{name} must be an integer from {minimum} to 2**63 - 1")
    return int(value)


def _check_seed(seed: Any) -> int:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**64:
        raise ValueError("seed must be an integer between 0 and 2**64 - 1")
    return int(seed)


def pit(samples: Any, actual: Any, *, randomized: bool = True, seed: int = 42) -> np.ndarray:
    """Probability integral transform (PIT) of each outcome under its predictive draws.

    Returns an array shaped like ``actual``, NaN where the outcome is missing. If
    the forecasts are calibrated, PIT values are uniform on [0, 1]: too many near
    0 and 1 means intervals are too narrow, too many near 0.5 means too wide.

    ``randomized=False`` gives the fraction of draws at or below the outcome. For a
    count forecast that is not uniform even when the forecast is exactly right,
    because a count has atoms. The default breaks ties at random: with ``n`` draws,
    of which ``below`` are smaller than the outcome and ``ties`` equal to it, the
    value is ``(below + u * (ties + 1)) / (n + 1)`` for a uniform ``u`` drawn from
    ``seed``. That is exactly uniform when the outcome and the draws come from the
    same distribution independently, for counts, continuous amounts and mixtures
    such as hurdle forecasts alike. MCMC draws are correlated with each other, so
    even for a correct model the uniformity is approximate.
    """
    from ._rustmc import _pit_values

    return _pit_values(samples, actual, _check_flag("randomized", randomized), _check_seed(seed), 0)


def pit_histogram(values: Any, bins: int = 10) -> np.ndarray:
    """Counts of PIT values in ``bins`` equal-width bins on [0, 1], ignoring NaN.

    The last bin includes 1. A calibrated forecast gives roughly equal counts.
    """
    from ._rustmc import _pit_histogram

    return _pit_histogram(values, _check_count("bins", bins))


def calibration_table(samples: Any, actual: Any, levels: Sequence[float] = CALIBRATION_LEVELS, *,
    by_horizon: bool = True) -> dict[str, np.ndarray]:
    """Empirical coverage and mean width of central predictive intervals at each level.

    ``samples`` and ``actual`` are shaped as for ``score_forecast``. Every target
    with an observed outcome counts; with ``by_horizon`` they are grouped by the
    last axis of ``actual``. Returns ``level`` (L,), and ``coverage``,
    ``coverage_lower``, ``coverage_upper`` and ``width`` shaped (horizon, L), or
    (L,) without ``by_horizon``, plus ``count``, the outcomes in each group.

    ``coverage_lower`` and ``coverage_upper`` are the 95% Wilson interval for a
    binomial proportion. It treats every outcome as an independent trial. Outcomes
    across the horizon steps of one path and across overlapping backtest folds are
    not independent, and their errors are usually positively correlated, which
    makes the interval too narrow.

    Bounds count as inside. For a forecast with atoms, such as a count or a
    hurdle forecast's zeros, an interval that ends on an atom holds more than its
    nominal probability, so coverage exceeds the level even when the forecast is
    exactly right (a forecast that is always zero covers at every level). Use the
    randomised PIT to check calibration of such forecasts.
    """
    from ._rustmc import _calibration_table

    return _calibration_table([(samples, actual)], levels, _check_flag("by_horizon", by_horizon))


def point_accuracy(samples: Any, actual: Any, *, training: Any = None, period: int = 1,
    point: str = "mean", by_horizon: bool = True) -> dict[str, np.ndarray | float]:
    """MAE, RMSE, WAPE and, given ``training``, MASE of a point forecast from draws.

    The point forecast is the predictive ``"mean"`` or ``"median"`` of the draws.
    WAPE is ``sum |error| / sum |actual|``. MASE divides each absolute error by its
    series' in-sample mean absolute difference between observations ``period``
    steps apart in ``training``, shaped ``(*series, time)`` for ``actual`` shaped
    ``(*series, horizon)``. A missing outcome is skipped. A zero denominator gives
    zero for a zero error and infinity otherwise, and MASE is NaN if a series has
    no observed pair ``period`` apart.
    """
    from ._rustmc import _point_accuracy

    return _point_accuracy([(samples, actual, training)], point, _check_count("period", period),
        _check_flag("by_horizon", by_horizon))


@dataclass(frozen=True)
class BacktestFold:
    origin: int
    actual: np.ndarray
    samples: np.ndarray | None
    scores: Mapping[str, np.ndarray] | None
    baseline_scores: Mapping[str, np.ndarray] | None
    diagnostics: Any = None
    error: str | None = None
    baseline_error: str | None = None
    baseline_samples: np.ndarray | None = None


@dataclass(frozen=True)
class BacktestResult:
    folds: tuple[BacktestFold, ...]
    observations: np.ndarray | None = None

    def summary(self, *, baseline: bool = False, by_horizon: bool = True) -> dict[str, np.ndarray | float]:
        """Average successful fold scores; report failures separately in .errors."""
        scores = [f.baseline_scores if baseline else f.scores for f in self.folds if f.error is None]
        scores = [s for s in scores if s is not None]
        if not scores:
            return {}
        result = {}
        for key in scores[0]:
            values = np.stack([s[key] for s in scores])
            axes = tuple(range(values.ndim - 1)) if by_horizon else None
            present = ~np.isnan(values)
            count = np.sum(present, axis=axes)
            denominator = np.sum(present, axis=axes, keepdims=True)
            # NaN represents a missing outcome; infinity is a real loss and
            # must not make a failed forecast disappear from the average.
            scale = np.max(np.where(np.isfinite(values), np.abs(values), 0), axis=axes, keepdims=True)
            scale = np.where(scale == 0, 1, scale)
            with np.errstate(over="ignore", invalid="ignore"):
                normalized = np.where(present, values, 0) / scale
                total = np.sum(normalized, axis=axes, keepdims=True)
                mean = np.divide(total, denominator, out=np.zeros_like(total), where=denominator > 0) * scale
            mean = np.squeeze(mean, axis=axes)
            result[key] = np.where(count > 0, mean, np.nan)
        return result

    @property
    def errors(self) -> dict[int, str]:
        return {f.origin: f.error for f in self.folds if f.error is not None}

    @property
    def baseline_errors(self) -> dict[int, str]:
        return {f.origin: f.baseline_error for f in self.folds if f.baseline_error is not None}

    def _forecast_folds(self) -> list[BacktestFold]:
        return [f for f in self.folds if f.error is None and f.samples is not None]

    def baseline(self) -> BacktestResult:
        """The baseline's forecasts as a result of their own, for the same methods.

        A fold whose baseline was not computed carries the reason as its error.
        """
        folds = []
        for f in self.folds:
            missing = f.baseline_samples is None and f.baseline_scores is None
            error = (f.baseline_error or f.error or "no baseline forecast") if missing else None
            folds.append(BacktestFold(f.origin, f.actual, f.baseline_samples, f.baseline_scores, None, error=error))
        return BacktestResult(tuple(folds), self.observations)

    def pit(self, *, randomized: bool = True, seed: int = 42) -> np.ndarray:
        """PIT values shaped ``(fold, *target)``, one row per entry of ``folds``.

        A failed fold's row is NaN, as is a missing outcome. Randomised values
        draw from a stream keyed by ``seed`` and the fold's origin. See ``pit``.

        The values are not independent. Folds that overlap share outcomes and most
        of their training data, and the horizon steps of one path share its errors.
        A uniformity test that assumes independence does not describe them, so read
        the histogram as a description, not as a test.
        """
        from ._rustmc import _pit_values

        randomized, seed = _check_flag("randomized", randomized), _check_seed(seed)
        if not self.folds:
            return np.empty((0,))
        shape = np.shape(self.folds[0].actual)
        if any(np.shape(fold.actual) != shape for fold in self.folds):
            raise ValueError("every fold must have the same target shape")
        values = np.full((len(self.folds),) + shape, np.nan)
        for index, fold in enumerate(self.folds):
            if fold.error is None and fold.samples is not None:
                values[index] = _pit_values(fold.samples, fold.actual, randomized, seed,
                    _check_count("fold origin", fold.origin, 0))
        return values

    def pit_histogram(self, bins: int = 10, *, randomized: bool = True, seed: int = 42,
        by_horizon: bool = False) -> np.ndarray:
        """PIT counts over all folds and series, shaped (bins,) or (horizon, bins)."""
        values = self.pit(randomized=randomized, seed=seed)
        if not _check_flag("by_horizon", by_horizon) or values.ndim < 2:
            return pit_histogram(values, bins)
        return np.stack([pit_histogram(values[..., h], bins) for h in range(values.shape[-1])])

    def calibration(self, levels: Sequence[float] = CALIBRATION_LEVELS, *,
        by_horizon: bool = True) -> dict[str, np.ndarray]:
        """Coverage and interval width per level, pooled over successful folds and series.

        See ``calibration_table`` for the keys and their limitations. The Wilson
        interval treats every outcome as independent. Outcomes from overlapping
        folds and from the steps of one path are not; their errors are usually
        positively correlated, which makes the interval too narrow by an amount
        this cannot estimate. Returns an empty dict when no fold succeeded.
        """
        from ._rustmc import _calibration_table

        folds = self._forecast_folds()
        if not folds:
            return {}
        return _calibration_table([(f.samples, f.actual) for f in folds], levels,
            _check_flag("by_horizon", by_horizon))

    def accuracy(self, *, period: int = 1, point: str = "mean", by_horizon: bool = True) -> dict[str, np.ndarray | float]:
        """MAE, RMSE, WAPE and MASE pooled over successful folds and series.

        MASE scales each fold by the training data before its origin, using
        ``observations``; without them it is omitted. See ``point_accuracy``.
        Returns an empty dict when no fold succeeded.
        """
        from ._rustmc import _point_accuracy

        folds = self._forecast_folds()
        if not folds:
            return {}
        y = self.observations
        if y is not None:
            for fold in folds:
                if _check_count("fold origin", fold.origin, 0) > np.shape(y)[-1]:
                    raise ValueError(f"fold origin {fold.origin} is beyond the observations")
        forecasts = [(f.samples, f.actual, None if y is None else y[..., :f.origin]) for f in folds]
        return _point_accuracy(forecasts, point, _check_count("period", period), _check_flag("by_horizon", by_horizon))


@dataclass(frozen=True)
class BacktestComparison:
    """Accuracy and calibration of several backtests over the origins they share.

    ``metrics[name]`` maps each metric to a value, or to one value per horizon
    step. ``origins`` are the origins every result forecast successfully, the
    only ones scored; ``excluded[name]`` lists that result's other origins.
    """
    metrics: Mapping[str, Mapping[str, np.ndarray | float]]
    origins: tuple[int, ...]
    excluded: Mapping[str, tuple[int, ...]]

    def __str__(self) -> str:
        names = list(self.metrics)
        rows = []
        for metric, first in self.metrics[names[0]].items():
            values = [np.atleast_1d(self.metrics[name][metric]) for name in names]
            for h in range(len(values[0])):
                label = metric if np.ndim(first) == 0 else f"{metric} h={h + 1}"
                rows.append([label] + [format(float(v[h]), ".4g") for v in values])
        header = ["metric"] + names
        widths = [max(len(row[i]) for row in [header] + rows) for i in range(len(header))]
        lines = [f"{len(self.origins)} common origins"]
        for row in [header] + rows:
            lines.append("  ".join([row[0].ljust(widths[0])] + [c.rjust(w) for c, w in zip(row[1:], widths[1:])]))
        return "\n".join(lines)


def compare_backtests(results: Mapping[str, BacktestResult], *, levels: Sequence[float] = (0.5, 0.8, 0.95),
    period: int = 1, point: str = "mean", by_horizon: bool = False) -> BacktestComparison:
    """Point accuracy and calibration of several backtests, side by side.

    Pass each forecaster's ``BacktestResult`` under a name, for example
    ``{"model": result, "seasonal naive": result.baseline()}``. Only origins every
    result forecast successfully are scored, and all results must hold the same
    outcomes there. Metrics are ``mae``, ``rmse``, ``wape``, ``mase`` (NaN for a
    result without its observations), ``crps`` (when every scored fold of every
    result holds its scores), ``wis`` (when, in addition, all were scored at the
    same interval levels), ``coverage_<level>`` and ``width_<level>`` for each of
    ``levels``, ``coverage_error``, the mean absolute gap between empirical and
    nominal coverage over ``CALIBRATION_LEVELS``, and ``count``, the outcomes
    scored. See ``calibration_table`` for when coverage can exceed the nominal
    level even for a correct forecast.
    """
    from ._rustmc import _interval_levels

    names = list(results)
    if not names or any(not isinstance(name, str) or not name or not isinstance(results[name], BacktestResult)
                        for name in names):
        raise ValueError("results must be a nonempty mapping from names to BacktestResult")
    succeeded = {}
    for name in names:
        folds = results[name]._forecast_folds()
        succeeded[name] = {f.origin: f for f in folds}
        if len(succeeded[name]) != len(folds):
            raise ValueError(f"'{name}' has more than one successful fold at the same origin")
    common = set.intersection(*(set(folds) for folds in succeeded.values()))
    if not common:
        raise ValueError("no origin was forecast successfully by every result")
    origins = tuple(sorted(common))
    reference = succeeded[names[0]]
    for name in names[1:]:
        for origin in origins:
            if not np.array_equal(succeeded[name][origin].actual, reference[origin].actual, equal_nan=True):
                raise ValueError(f"'{name}' and '{names[0]}' hold different outcomes at origin {origin}")
    levels = _interval_levels(levels)
    table_levels = list(dict.fromkeys(levels + list(CALIBRATION_LEVELS)))
    # A scalar target has no horizon axis to group by.
    by_horizon = _check_flag("by_horizon", by_horizon) and np.ndim(reference[origins[0]].actual) > 0
    # CRPS and WIS come from each fold's stored scores. Averaging them over a
    # subset of the origins, or WIS computed at different levels, would not
    # compare like with like, so they are left out unless neither happens.
    subsets = {name: BacktestResult(tuple(succeeded[name][origin] for origin in origins), results[name].observations)
               for name in names}
    scored = all(f.scores is not None for subset in subsets.values() for f in subset.folds)
    wis_levels = {frozenset(key for key in f.scores if key.startswith("coverage_"))
                  for subset in subsets.values() for f in subset.folds} if scored else set()
    shared_scores = ("crps", "wis") if len(wis_levels) == 1 else ("crps",) if scored else ()
    metrics = {}
    for name, subset in subsets.items():
        accuracy = subset.accuracy(period=period, point=point, by_horizon=by_horizon)
        table = subset.calibration(table_levels, by_horizon=by_horizon)
        row = {key: accuracy[key] for key in ("mae", "rmse", "wape", "mase") if key in accuracy}
        scores = subset.summary(by_horizon=by_horizon)
        row.update({key: scores[key] for key in shared_scores if key in scores})
        for index, level in enumerate(levels):
            suffix = format(float(level), ".12g")
            row[f"coverage_{suffix}"] = table["coverage"][..., index]
            row[f"width_{suffix}"] = table["width"][..., index]
        grid = [table_levels.index(level) for level in CALIBRATION_LEVELS]
        row["coverage_error"] = np.abs(table["coverage"][..., grid] - np.asarray(CALIBRATION_LEVELS)).mean(axis=-1)
        row["count"] = accuracy["count"]
        metrics[name] = row
    # A metric one result lacks (MASE without observations) is NaN for it, so
    # every result reports the same metrics.
    keys = list(dict.fromkeys(key for row in metrics.values() for key in row))
    for name, row in metrics.items():
        missing = np.full(np.shape(row["count"]), np.nan)
        metrics[name] = {key: row.get(key, missing if missing.ndim else float("nan")) for key in keys}
    excluded = {name: tuple(sorted(f.origin for f in results[name].folds if f.origin not in common)) for name in names}
    return BacktestComparison(metrics, origins, excluded)


def backtest(
    model: Any,
    observations: Any,
    *,
    horizon: int,
    initial: int | None = None,
    origins: Sequence[int] | None = None,
    step: int = 1,
    exog: Any = None,
    exposure: Any = None,
    fit_kwargs: Mapping[str, Any] | None = None,
    forecast_kwargs: Mapping[str, Any] | None = None,
    levels: Sequence[float] = (0.5, 0.8, 0.95),
    baseline_period: int | None = 1,
    seed: int = 42,
    errors: str = "raise",
) -> BacktestResult:
    """Refit at each historical origin using only the preceding observations.

    model is a fitted-model constructor with .fit, or a callable receiving a copy
    of training observations and returning such a model. The callable is the place
    to estimate fold-specific priors/preprocessing. Exogenous data use (..., time,
    feature); observations/exposure use (..., time). Only complete test horizons
    are evaluated. Seeds depend on origin and survive fold reordering.
    """
    from ._rustmc import forecast_cell_seed
    y = np.asarray(observations, dtype=float)
    if y.ndim not in (1, 2) or np.isinf(y).any():
        raise ValueError("observations must be a series or panel, finite or NaN")
    for name, value in (("horizon", horizon), ("step", step)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if errors not in ("raise", "collect"):
        raise ValueError("errors must be 'raise' or 'collect'")
    if origins is None:
        if isinstance(initial, bool) or not isinstance(initial, (int, np.integer)) or initial < 1:
            raise ValueError("supply a positive initial training length or explicit origins")
        origins = list(range(initial, y.shape[-1] - horizon + 1, step))
    origins = tuple(origins)
    if not origins or len(set(origins)) != len(origins) or any(
        isinstance(o, bool) or not isinstance(o, (int, np.integer)) or o < 1 or o + horizon > y.shape[-1] for o in origins
    ):
        raise ValueError("origins must be unique positive training lengths with complete test horizons")
    x = None if exog is None else np.asarray(exog, dtype=float)
    e = None if exposure is None else np.asarray(exposure, dtype=float)
    if x is not None and (x.ndim != y.ndim + 1 or x.shape[:-2] != y.shape[:-1] or x.shape[-2] != y.shape[-1] or not np.isfinite(x).all()):
        raise ValueError("exog must align with observations and add a final feature axis")
    if e is not None and (e.shape != y.shape or not np.isfinite(e).all()):
        raise ValueError("exposure must be finite and match observations")
    if any(k in (fit_kwargs or {}) or k in (forecast_kwargs or {}) for k in ("exog", "exposure")):
        raise ValueError("pass exog/exposure to backtest so they are sliced at the origin")
    folds = []
    for origin in origins:
        truth = y[..., origin:origin+horizon].copy()
        try:
            train = y[..., :origin].copy()
            candidate = model if hasattr(model, "fit") else model(train.copy())
            fit_options = dict(fit_kwargs or {})
            forecast_options = dict(forecast_kwargs or {})
            fit_options.setdefault("seed", forecast_cell_seed(seed, str(origin), "fit"))
            forecast_options.setdefault("seed", forecast_cell_seed(seed, str(origin), "forecast"))
            if x is not None:
                fit_options["exog"] = x[..., :origin, :].copy()
                forecast_options["exog"] = x[..., origin:origin+horizon, :].copy()
            if e is not None:
                fit_options["exposure"] = e[..., :origin].copy()
                forecast_options["exposure"] = e[..., origin:origin+horizon].copy()
            fit = candidate.fit(train, **fit_options)
            forecast = fit.forecast(horizon, **forecast_options)
            samples = np.asarray(forecast.observation_samples)
            scores = score_forecast(samples, truth, levels)
            baseline = None
            baseline_samples = None
            baseline_error = None
            if baseline_period is not None:
                try:
                    baseline_samples = naive_forecast(train, horizon, period=baseline_period,
                        seed=forecast_cell_seed(seed, f"baseline:{origin}", "forecast"))
                    baseline = score_forecast(baseline_samples, truth, levels)
                except ValueError as error:
                    baseline, baseline_samples, baseline_error = None, None, str(error)
            diagnostics = fit.diagnostics() if hasattr(fit, "diagnostics") else None
            folds.append(BacktestFold(origin, truth, samples, scores, baseline, diagnostics,
                baseline_error=baseline_error, baseline_samples=baseline_samples))
        except Exception as error:
            if errors == "raise":
                raise
            folds.append(BacktestFold(origin, truth, None, None, None, error=f"{type(error).__name__}: {error}"))
    observations = y.copy()
    observations.setflags(write=False)
    return BacktestResult(tuple(folds), observations)
