"""Leakage-free, deterministic LightGBM baseline for short univariate series."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from time import perf_counter
from typing import Iterable

import lightgbm as lgb
import numpy as np


@dataclass
class LightGBMForecast:
    mean: np.ndarray
    fit_seconds: float
    forecast_seconds: float
    selected_params: dict
    validation_mae: float
    training_rows: int

    def as_dict(self) -> dict:
        value = asdict(self)
        value["mean"] = self.mean.tolist()
        return value


def _slope(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    x = np.arange(values.size, dtype=np.float64)
    return float(np.polyfit(x, values, 1)[0])


def _baseline(
    y: np.ndarray,
    cutoff: int,
    horizon: int,
    period: int,
    seasonal_drift_weight: float,
) -> float:
    """Training-only seasonal-drift baseline that permits extrapolation."""
    available = y[: cutoff + 1]
    width = min(max(8, period), available.size)
    trend = _slope(available[-width:])
    trend_projection = float(y[cutoff] + horizon * trend)
    target = cutoff + horizon
    seasonal_idx = target - period
    if seasonal_idx < 0 or seasonal_idx > cutoff:
        return trend_projection
    # A previous-season analogue adjusted by the estimated full-cycle drift.
    if available.size > period:
        cycle_drift = float(
            np.median(available[period:] - available[:-period])
        )
    else:
        cycle_drift = period * trend
    drift = (
        seasonal_drift_weight * cycle_drift
        + (1.0 - seasonal_drift_weight) * period * trend
    )
    seasonal_projection = float(y[seasonal_idx] + drift)
    return 0.7 * seasonal_projection + 0.3 * trend_projection


def _features(
    y: np.ndarray,
    cutoff: int,
    horizon: int,
    period: int,
    seasonal_drift_weight: float,
) -> list[float]:
    """Features known at cutoff for target cutoff + horizon."""
    target = cutoff + horizon
    available = y[: cutoff + 1]
    recent_mean = float(np.mean(available[-min(6, available.size) :]))
    row: list[float] = [
        float(horizon),
        float(target),
        float(y[cutoff]),
        float(y[cutoff] - y[cutoff - 1]) if cutoff else 0.0,
    ]
    for lag in (1, 2, 3, 4, 6, 12, 26, 52):
        idx = cutoff - lag + 1
        row.extend([float(y[idx]) if idx >= 0 else recent_mean, float(idx >= 0)])
    for width in (3, 6, 12, 26, 52):
        window = available[-min(width, available.size) :]
        row.extend([float(np.mean(window)), float(np.std(window)), _slope(window)])
    seasonal_idx = target - period
    has_seasonal = 0 <= seasonal_idx <= cutoff
    row.extend([
        float(y[seasonal_idx]) if has_seasonal else recent_mean,
        float(has_seasonal),
        _baseline(y, cutoff, horizon, period, seasonal_drift_weight),
    ])
    angle = 2.0 * np.pi * target / period
    row.extend([
        float(np.sin(angle)),
        float(np.cos(angle)),
        float(np.sin(2.0 * angle)),
        float(np.cos(2.0 * angle)),
    ])
    return row


def _supervised(
    y: np.ndarray,
    max_horizon: int,
    period: int,
    seasonal_drift_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    # Starting at cutoff 5 ensures every row has a minimally useful local history.
    rows: list[list[float]] = []
    targets: list[float] = []
    for cutoff in range(5, y.size - 1):
        for horizon in range(1, min(max_horizon, y.size - cutoff - 1) + 1):
            rows.append(_features(y, cutoff, horizon, period, seasonal_drift_weight))
            targets.append(float(
                y[cutoff + horizon]
                - _baseline(y, cutoff, horizon, period, seasonal_drift_weight)
            ))
    return np.asarray(rows, dtype=np.float64), np.asarray(targets, dtype=np.float64)


def _future_matrix(
    y: np.ndarray, horizon: int, period: int, seasonal_drift_weight: float
) -> np.ndarray:
    cutoff = y.size - 1
    return np.asarray(
        [
            _features(y, cutoff, step, period, seasonal_drift_weight)
            for step in range(1, horizon + 1)
        ],
        dtype=np.float64,
    )


def _future_baseline(
    y: np.ndarray, horizon: int, period: int, seasonal_drift_weight: float
) -> np.ndarray:
    cutoff = y.size - 1
    return np.asarray(
        [
            _baseline(y, cutoff, step, period, seasonal_drift_weight)
            for step in range(1, horizon + 1)
        ],
        dtype=np.float64,
    )


def _model(params: dict, seed: int) -> lgb.LGBMRegressor:
    model_params = {key: value for key, value in params.items() if key != "seasonal_drift_weight"}
    return lgb.LGBMRegressor(
        objective="regression_l1",
        random_state=seed,
        deterministic=True,
        force_col_wise=True,
        n_jobs=1,
        verbosity=-1,
        **model_params,
    )


def _candidate_params() -> Iterable[dict]:
    # Deliberately compact: enough regularization choices for short series without
    # turning the comparison into a large hyperparameter search.
    model_params = [
        dict(n_estimators=160, learning_rate=0.035, num_leaves=7,
             max_depth=3, min_child_samples=3, reg_lambda=1.0,
             feature_fraction=0.9),
        dict(n_estimators=260, learning_rate=0.025, num_leaves=7,
             max_depth=3, min_child_samples=5, reg_lambda=3.0,
             feature_fraction=1.0),
        dict(n_estimators=180, learning_rate=0.04, num_leaves=15,
             max_depth=4, min_child_samples=5, reg_lambda=2.0,
             feature_fraction=0.85),
        dict(n_estimators=300, learning_rate=0.02, num_leaves=15,
             max_depth=4, min_child_samples=10, reg_lambda=5.0,
             feature_fraction=1.0),
    ]
    for drift_weight in (0.0, 0.5, 1.0):
        for params in model_params:
            yield {**params, "seasonal_drift_weight": drift_weight}


def _validation_origins(n: int, horizon: int, period: int) -> list[int]:
    """Return training lengths for two expanding-window validation folds."""
    earliest = max(12, period)
    candidates = [n - 2 * horizon, n - horizon]
    origins = sorted({max(earliest, value) for value in candidates})
    return [value for value in origins if value >= 7 and value + horizon <= n]


def forecast_lightgbm(
    y_train: np.ndarray,
    horizon: int,
    period: int,
    seed: int = 20260802,
) -> LightGBMForecast:
    """Tune on expanding training-only folds, refit, and forecast directly."""
    y = np.asarray(y_train, dtype=np.float64)
    if y.ndim != 1 or not np.all(np.isfinite(y)):
        raise ValueError("y_train must be a finite one-dimensional array")
    if y.size < max(18, period):
        raise ValueError("training series is too short for this seasonal baseline")

    origins = _validation_origins(y.size, horizon, period)
    best_score = np.inf
    best_params: dict | None = None
    for candidate_idx, params in enumerate(_candidate_params()):
        errors: list[float] = []
        drift_weight = float(params["seasonal_drift_weight"])
        for train_length in origins:
            fold_y = y[:train_length]
            x_fit, y_fit = _supervised(fold_y, horizon, period, drift_weight)
            model = _model(params, seed + candidate_idx)
            model.fit(x_fit, y_fit)
            prediction = (
                _future_baseline(fold_y, horizon, period, drift_weight)
                + model.predict(_future_matrix(fold_y, horizon, period, drift_weight))
            )
            actual = y[train_length : train_length + horizon]
            errors.extend(np.abs(prediction - actual).tolist())
        score = float(np.mean(errors)) if errors else np.inf
        if score < best_score:
            best_score = score
            best_params = params

    assert best_params is not None
    drift_weight = float(best_params["seasonal_drift_weight"])
    x_fit, y_fit = _supervised(y, horizon, period, drift_weight)
    final_model = _model(best_params, seed)
    fit_started = perf_counter()
    final_model.fit(x_fit, y_fit)
    fit_seconds = perf_counter() - fit_started

    forecast_started = perf_counter()
    mean = _future_baseline(y, horizon, period, drift_weight) + final_model.predict(
        _future_matrix(y, horizon, period, drift_weight)
    )
    forecast_seconds = perf_counter() - forecast_started
    return LightGBMForecast(
        mean=np.asarray(mean, dtype=np.float64),
        fit_seconds=fit_seconds,
        forecast_seconds=forecast_seconds,
        selected_params=best_params,
        validation_mae=best_score,
        training_rows=int(x_fit.shape[0]),
    )


def accuracy(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    error = predicted - actual
    denominator = np.abs(actual) + np.abs(predicted)
    smape = np.mean(np.where(denominator > 0, 2.0 * np.abs(error) / denominator, 0.0))
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "smape": float(smape),
    }
