"""GP and Prophet adapters for the deterministic short-series comparison.

Model selection uses training observations only. Final held-out observations are never
passed to fitting, hyperparameter optimization, or candidate selection.
"""

from __future__ import annotations

import json
import logging
import time
from argparse import ArgumentParser
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

Z_95 = 1.959963984540054
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@dataclass(frozen=True)
class ForecastSeries:
    name: str
    frequency: Literal["monthly", "weekly"]
    dates: pd.DatetimeIndex
    train: np.ndarray
    test: np.ndarray
    seasonal_periods: tuple[float, ...]

    def validate(self) -> None:
        expected_train = 24 if self.frequency == "monthly" else 104
        expected_test = 6 if self.frequency == "monthly" else 26
        if len(self.train) != expected_train or len(self.test) != expected_test:
            raise ValueError(
                f"{self.name}: expected {expected_train}/{expected_test} train/test "
                f"points for {self.frequency}, got {len(self.train)}/{len(self.test)}"
            )
        if len(self.dates) != expected_train + expected_test:
            raise ValueError(
                f"{self.name}: dates must cover train and held-out periods"
            )
        if not np.all(np.isfinite(self.train)) or not np.all(np.isfinite(self.test)):
            raise ValueError(f"{self.name}: values must be finite")


@dataclass
class ForecastOutput:
    series: str
    engine: str
    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    fit_forecast_seconds: float
    selection_seconds: float
    selected_model: dict[str, Any]
    selection_score: float
    interval_kind: str
    notes: list[str]

    def to_dict(self, observed: np.ndarray) -> dict[str, Any]:
        return {
            "series": self.series,
            "engine": self.engine,
            "mean": self.mean.tolist(),
            "lower": self.lower.tolist(),
            "upper": self.upper.tolist(),
            "fit_forecast_seconds": self.fit_forecast_seconds,
            "selection_seconds": self.selection_seconds,
            "selected_model": self.selected_model,
            "selection_score": self.selection_score,
            "interval_kind": self.interval_kind,
            "notes": self.notes,
            "metrics": forecast_metrics(observed, self.mean, self.lower, self.upper),
        }


def forecast_metrics(
    observed: np.ndarray,
    mean: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float = 0.05,
) -> dict[str, float]:
    observed = np.asarray(observed, dtype=float)
    mean = np.asarray(mean, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if not (observed.shape == mean.shape == lower.shape == upper.shape):
        raise ValueError("observed, mean, lower, and upper must have identical shapes")
    if np.any(lower > upper):
        raise ValueError("lower interval bound exceeds upper bound")
    error = mean - observed
    denominator = np.maximum(np.abs(observed) + np.abs(mean), 1e-12)
    interval_score = (
        upper
        - lower
        + (2.0 / alpha) * (lower - observed) * (observed < lower)
        + (2.0 / alpha) * (observed - upper) * (observed > upper)
    )
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "smape": float(200.0 * np.mean(np.abs(error) / denominator)),
        "coverage_95": float(np.mean((observed >= lower) & (observed <= upper))),
        "mean_width_95": float(np.mean(upper - lower)),
        "wis_95": float(np.mean(interval_score)),
    }


def _origins(n: int, frequency: str) -> list[int]:
    minimum = 12 if frequency == "monthly" else 52
    horizon = 3 if frequency == "monthly" else 13
    candidates = [
        minimum,
        minimum + (n - minimum) // 3,
        minimum + 2 * (n - minimum) // 3,
    ]
    return sorted({origin for origin in candidates if origin + horizon <= n})


def _validation_nlpd(y: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> float:
    variance = np.maximum(np.asarray(sd, dtype=float) ** 2, 1e-10)
    return float(
        np.mean(0.5 * (np.log(2 * np.pi * variance) + (y - mean) ** 2 / variance))
    )


def _detrend(
    t: np.ndarray, y: np.ndarray, mean_kind: str
) -> tuple[np.ndarray, np.ndarray]:
    if mean_kind == "linear":
        design = np.column_stack([np.ones(len(t)), t])
    elif mean_kind == "constant":
        design = np.ones((len(t), 1))
    else:
        raise ValueError(f"unknown mean kind: {mean_kind}")
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    return coefficients, y - design @ coefficients


def _mean_predict(
    t: np.ndarray, coefficients: np.ndarray, mean_kind: str
) -> np.ndarray:
    if mean_kind == "linear":
        return coefficients[0] + coefficients[1] * t
    return np.full(len(t), coefficients[0])


def _seasonal_kernel(period: float):
    from sklearn.gaussian_process.kernels import (
        ConstantKernel,
        ExpSineSquared,
    )

    return ConstantKernel(1.0, (0.05, 20.0)) * ExpSineSquared(
        length_scale=1.0,
        periodicity=period,
        length_scale_bounds=(0.2, 10.0),
        periodicity_bounds="fixed",
    )


def _gp_kernel(kind: str, seasonal_periods: tuple[float, ...]):
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

    longest_period = max(seasonal_periods)
    smooth = ConstantKernel(1.0, (0.05, 20.0)) * RBF(
        longest_period / 2.0, (1.0, longest_period * 5.0)
    )
    seasonal = _seasonal_kernel(longest_period)
    noise = WhiteKernel(0.1, (1e-4, 5.0))
    if kind == "smooth":
        return smooth + noise
    if kind == "additive_seasonal":
        return smooth + seasonal + noise
    if kind == "quasi_periodic":
        return smooth * seasonal + noise
    if kind == "multi_seasonal":
        kernel = smooth
        for period in seasonal_periods:
            kernel += _seasonal_kernel(period)
        return kernel + noise
    raise ValueError(f"unknown GP kernel: {kind}")


def _fit_gp_candidate(
    y: np.ndarray,
    future_steps: int,
    seasonal_periods: tuple[float, ...],
    mean_kind: str,
    kernel_kind: str,
    seed: int,
    restarts: int,
) -> tuple[np.ndarray, np.ndarray, Any]:
    from sklearn.gaussian_process import GaussianProcessRegressor

    t = np.arange(len(y), dtype=float)
    future_t = np.arange(len(y), len(y) + future_steps, dtype=float)
    coefficients, residual = _detrend(t, y, mean_kind)
    residual_scale = max(float(np.std(residual, ddof=1)), 1e-8)
    normalized = residual / residual_scale
    model = GaussianProcessRegressor(
        kernel=_gp_kernel(kernel_kind, seasonal_periods),
        alpha=0.0,
        normalize_y=False,
        n_restarts_optimizer=restarts,
        random_state=seed,
    )
    model.fit(t[:, None], normalized)
    residual_mean, residual_sd = model.predict(future_t[:, None], return_std=True)
    mean = (
        _mean_predict(future_t, coefficients, mean_kind)
        + residual_scale * residual_mean
    )
    sd = residual_scale * residual_sd
    return mean, sd, model


def forecast_gp(series: ForecastSeries, seed: int = 20260802) -> ForecastOutput:
    """Training-only CV over mean and kernel structure, then a final GP fit."""

    series.validate()
    selection_started = time.perf_counter()
    horizon = 3 if series.frequency == "monthly" else 13
    kernel_kinds = ["smooth", "additive_seasonal", "quasi_periodic"]
    if len(series.seasonal_periods) > 1:
        kernel_kinds.append("multi_seasonal")
    candidates = [
        (mean_kind, kernel_kind)
        for mean_kind in ("constant", "linear")
        for kernel_kind in kernel_kinds
    ]
    scored = []
    for candidate_index, (mean_kind, kernel_kind) in enumerate(candidates):
        fold_scores = []
        for origin in _origins(len(series.train), series.frequency):
            validation = series.train[origin : origin + horizon]
            mean, sd, _ = _fit_gp_candidate(
                series.train[:origin],
                len(validation),
                series.seasonal_periods,
                mean_kind,
                kernel_kind,
                seed + candidate_index,
                restarts=0,
            )
            fold_scores.append(_validation_nlpd(validation, mean, sd))
        scored.append((float(np.mean(fold_scores)), mean_kind, kernel_kind))
    selection_score, mean_kind, kernel_kind = min(scored, key=lambda row: row[0])
    selection_seconds = time.perf_counter() - selection_started
    final_started = time.perf_counter()
    mean, sd, model = _fit_gp_candidate(
        series.train,
        len(series.test),
        series.seasonal_periods,
        mean_kind,
        kernel_kind,
        seed,
        restarts=1,
    )
    return ForecastOutput(
        series=series.name,
        engine="gaussian_process",
        mean=mean,
        lower=mean - Z_95 * sd,
        upper=mean + Z_95 * sd,
        fit_forecast_seconds=time.perf_counter() - final_started,
        selection_seconds=selection_seconds,
        selected_model={
            "mean": mean_kind,
            "kernel_family": kernel_kind,
            "declared_seasonal_periods": list(series.seasonal_periods),
            "optimized_kernel": str(model.kernel_),
            "cv_origins": _origins(len(series.train), series.frequency),
        },
        selection_score=selection_score,
        interval_kind="empirical-Bayes Gaussian posterior predictive (plug-in hyperparameters)",
        notes=[
            "kernel and mean structure selected by rolling-origin NLPD on training data only",
            "hyperparameter uncertainty is not integrated, so intervals may be optimistic",
        ],
    )


def _prophet_candidate(
    dates: pd.DatetimeIndex,
    y: np.ndarray,
    future_dates: pd.DatetimeIndex,
    frequency: str,
    seasonal_periods: tuple[float, ...],
    mode: str,
    changepoint_prior_scale: float,
    seed: int,
    uncertainty_samples: int,
):
    from prophet import Prophet

    np.random.seed(seed)
    model = Prophet(
        growth="linear",
        seasonality_mode=mode,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=5.0,
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=0.95,
        uncertainty_samples=uncertainty_samples,
        mcmc_samples=0,
    )
    days_per_step = 365.25 / 12.0 if frequency == "monthly" else 7.0
    for period in seasonal_periods:
        model.add_seasonality(
            name=f"period_{period:g}",
            period=period * days_per_step,
            fourier_order=3
            if frequency == "monthly"
            else max(2, min(6, int(period // 4))),
            mode=mode,
        )
    model.fit(pd.DataFrame({"ds": dates, "y": y}))
    prediction = model.predict(pd.DataFrame({"ds": future_dates}))
    return prediction, model


def forecast_prophet(series: ForecastSeries, seed: int = 20260802) -> ForecastOutput:
    """Training-only rolling-origin selection for a deterministic Prophet baseline."""

    series.validate()
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    logging.getLogger("prophet").setLevel(logging.ERROR)
    selection_started = time.perf_counter()
    horizon = 3 if series.frequency == "monthly" else 13
    candidates = [
        (mode, changepoint)
        for mode in ("additive", "multiplicative")
        for changepoint in (0.01, 0.05, 0.2)
    ]
    scored = []
    for candidate_index, (mode, changepoint) in enumerate(candidates):
        fold_scores = []
        for fold_index, origin in enumerate(
            _origins(len(series.train), series.frequency)
        ):
            validation = series.train[origin : origin + horizon]
            prediction, _ = _prophet_candidate(
                series.dates[:origin],
                series.train[:origin],
                series.dates[origin : origin + horizon],
                series.frequency,
                series.seasonal_periods,
                mode,
                changepoint,
                seed + 100 * candidate_index + fold_index,
                uncertainty_samples=200,
            )
            mean = prediction["yhat"].to_numpy()
            sd = (
                prediction["yhat_upper"].to_numpy()
                - prediction["yhat_lower"].to_numpy()
            ) / (2 * Z_95)
            fold_scores.append(_validation_nlpd(validation, mean, sd))
        scored.append((float(np.mean(fold_scores)), mode, changepoint))
    selection_score, mode, changepoint = min(scored, key=lambda row: row[0])
    selection_seconds = time.perf_counter() - selection_started
    final_started = time.perf_counter()
    prediction, _ = _prophet_candidate(
        series.dates[: len(series.train)],
        series.train,
        series.dates[len(series.train) :],
        series.frequency,
        series.seasonal_periods,
        mode,
        changepoint,
        seed,
        uncertainty_samples=1_000,
    )
    return ForecastOutput(
        series=series.name,
        engine="prophet",
        mean=prediction["yhat"].to_numpy(),
        lower=prediction["yhat_lower"].to_numpy(),
        upper=prediction["yhat_upper"].to_numpy(),
        fit_forecast_seconds=time.perf_counter() - final_started,
        selection_seconds=selection_seconds,
        selected_model={
            "seasonality_mode": mode,
            "changepoint_prior_scale": changepoint,
            "declared_seasonal_periods": list(series.seasonal_periods),
            "cv_origins": _origins(len(series.train), series.frequency),
        },
        selection_score=selection_score,
        interval_kind="Prophet 95% uncertainty interval (MAP trend/seasonality plus simulation)",
        notes=[
            "mode and changepoint prior selected by rolling-origin NLPD on training data only",
            "this is Prophet's uncertainty interval, not a full Bayesian credible interval",
        ],
    )


def load_common_six(data_dir: str | Path) -> list[ForecastSeries]:
    """Load the repository's deterministic common protocol without using held-out data."""

    data_dir = Path(data_dir)
    output = []
    names = [
        "monthly_easy",
        "monthly_medium",
        "monthly_hard",
        "weekly_easy",
        "weekly_medium",
        "weekly_hard",
    ]
    for name in names:
        path = data_dir / f"{name}.csv"
        if not path.is_file():
            raise ValueError(f"missing common protocol file: {path}")
        frame = pd.read_csv(path)
        dates = pd.to_datetime(frame["date"])
        train_mask = frame["split"].eq("train").to_numpy()
        test_mask = frame["split"].eq("test").to_numpy()
        frequency = "monthly" if name.startswith("monthly_") else "weekly"
        periods = (12.0,) if frequency == "monthly" else (52.0,)
        if name == "weekly_hard":
            periods = (52.0, 26.0, 13.0)
        series = ForecastSeries(
            name=name,
            frequency=frequency,
            dates=pd.DatetimeIndex(dates),
            train=frame.loc[train_mask, "value"].to_numpy(dtype=float),
            test=frame.loc[test_mask, "value"].to_numpy(dtype=float),
            seasonal_periods=periods,
        )
        series.validate()
        output.append(series)
    return output


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = []
    data_dir = args.data_dir
    for series in load_common_six(data_dir):
        for adapter in (forecast_gp, forecast_prophet):
            try:
                result = adapter(series)
                results.append(result.to_dict(series.test))
            except ModuleNotFoundError as exc:
                results.append(
                    {
                        "series": series.name,
                        "engine": adapter.__name__.removeprefix("forecast_"),
                        "status": "unavailable",
                        "reason": f"missing optional dependency: {exc.name}",
                    }
                )
    payload = {
        "seed": 20260802,
        "data_dir": str(data_dir),
        "package_versions": {
            package: version(package)
            for package in ("numpy", "pandas", "scikit-learn", "prophet")
        },
        "timing_definition": "final selected fit plus held-out forecast; excludes imports and selection",
        "selection_definition": "rolling-origin Gaussian NLPD on training data only",
        "results": results,
    }
    encoded = json.dumps(payload, indent=2)
    if args.output is None:
        print(encoded)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
