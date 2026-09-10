"""Probabilistic forecast scores and evaluation at historical forecast origins."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

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
    weights = (2 * np.arange(1, n + 1) - n - 1).reshape((n,) + (1,) * actual.ndim)
    return np.mean(np.abs(draws - actual), axis=0) - np.sum(weights * ordered, axis=0) / (n * n)


def weighted_interval_score(samples: Any, actual: Any, levels: Sequence[float] = (0.5, 0.8, 0.95)) -> np.ndarray:
    """Standard WIS: median error plus alpha/2-weighted interval scores, divided by K+0.5."""
    draws, actual = _draws(samples, actual)
    levels = np.asarray(levels, dtype=float)
    if levels.ndim != 1 or not levels.size or not np.isfinite(levels).all() or np.any((levels <= 0) | (levels >= 1)):
        raise ValueError("levels must be a nonempty sequence strictly between zero and one")
    if len(np.unique(levels)) != len(levels):
        raise ValueError("interval levels must be unique")
    score = 0.5 * np.abs(actual - np.median(draws, axis=0))
    for level in levels:
        alpha = 1 - level
        lower, upper = np.quantile(draws, (alpha / 2, 1 - alpha / 2), axis=0)
        score += (alpha / 2) * interval_score(actual, lower, upper, alpha)
    return score / (len(levels) + 0.5)


def score_forecast(samples: Any, actual: Any, levels: Sequence[float] = (0.5, 0.8, 0.95)) -> dict[str, np.ndarray]:
    """Scores retain target dimensions, including horizon. Missing outcomes score NaN."""
    draws, actual = _draws(samples, actual)
    bias = np.mean(draws, axis=0) - actual
    scores = {
        "bias": bias,
        "absolute_error": np.abs(bias),
        "squared_error": bias**2,
        "crps": crps(draws, actual),
        "wis": weighted_interval_score(draws, actual, levels),
    }
    for level in levels:
        lower, upper = np.quantile(draws, ((1-level)/2, (1+level)/2), axis=0)
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


@dataclass(frozen=True)
class BacktestResult:
    folds: tuple[BacktestFold, ...]

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
            finite = np.isfinite(values)
            count = np.sum(finite, axis=axes)
            total = np.sum(np.where(finite, values, 0), axis=axes)
            result[key] = np.divide(total, count, out=np.full_like(total, np.nan, dtype=float), where=count > 0)
        return result

    @property
    def errors(self) -> dict[int, str]:
        return {f.origin: f.error for f in self.folds if f.error is not None}

    @property
    def baseline_errors(self) -> dict[int, str]:
        return {f.origin: f.baseline_error for f in self.folds if f.baseline_error is not None}


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
            baseline_error = None
            if baseline_period is not None:
                try:
                    baseline = score_forecast(naive_forecast(train, horizon, period=baseline_period,
                        seed=forecast_cell_seed(seed, f"baseline:{origin}", "forecast")), truth, levels)
                except ValueError as error:
                    baseline_error = str(error)
            diagnostics = fit.diagnostics() if hasattr(fit, "diagnostics") else None
            folds.append(BacktestFold(origin, truth, samples, scores, baseline, diagnostics, baseline_error=baseline_error))
        except Exception as error:
            if errors == "raise":
                raise
            folds.append(BacktestFold(origin, truth, None, None, None, error=f"{type(error).__name__}: {error}"))
    return BacktestResult(tuple(folds))
