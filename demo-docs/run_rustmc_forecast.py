"""Run the RustMC arm of the deterministic synthetic forecasting study.

Model selection uses training observations only. The reported 95% bands are
pointwise highest-density intervals of future-observation posterior-predictive draws.
They are not latent-state credible intervals and not simultaneous trajectory bands.
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rustmc as rmc


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
SEED = 20_260_802


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    order: int = 0
    scales: tuple[float, ...] = ()


def load_dataset(path: Path) -> tuple[list[dict[str, str]], np.ndarray, np.ndarray]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    train = np.asarray([float(row["value"]) for row in rows if row["split"] == "train"])
    test = np.asarray([float(row["value"]) for row in rows if row["split"] == "test"])
    return rows, train, test


def standardize(y: np.ndarray) -> tuple[np.ndarray, float, float]:
    center = float(y.mean())
    scale = float(y.std(ddof=1))
    if not math.isfinite(scale) or scale < 1e-8:
        scale = max(abs(center) * 0.01, 1.0)
    return (y - center) / scale, center, scale


def base_candidates(period: int) -> list[Candidate]:
    candidates = [
        Candidate("local_level_smooth", "local_level", scales=(0.04, 0.70)),
        Candidate("local_level_responsive", "local_level", scales=(0.20, 0.45)),
        Candidate("local_trend_smooth", "trend", scales=(0.04, 0.004, 0.70)),
        Candidate("local_trend_responsive", "trend", scales=(0.12, 0.018, 0.45)),
        Candidate("ar1", "ar", order=1),
        Candidate("ar2", "ar", order=2),
        Candidate("ar3", "ar", order=3),
        Candidate("ar6", "ar", order=6),
    ]
    if period == 12:
        candidates.append(Candidate("seasonal_ar12", "ar", order=12))
    else:
        candidates.extend(
            [
                Candidate("seasonal_ar13", "ar", order=13),
                Candidate("seasonal_ar26", "ar", order=26),
                Candidate("seasonal_ar52", "ar", order=52),
            ]
        )
    return candidates


def make_model(candidate: Candidate, z: np.ndarray):
    if candidate.family == "local_level":
        process, observation = candidate.scales
        return rmc.BayesianLocalLevel(
            process_variance_prior=rmc.InverseGammaPrior(3.0, process),
            observation_variance_prior=rmc.InverseGammaPrior(3.0, observation),
            initial_mean=float(z[0]),
            initial_variance=4.0,
        )
    if candidate.family == "trend":
        level, slope, observation = candidate.scales
        return rmc.BayesianLocalLinearTrend(
            level_variance_prior=rmc.InverseGammaPrior(3.0, level),
            slope_variance_prior=rmc.InverseGammaPrior(3.0, slope),
            observation_variance_prior=rmc.InverseGammaPrior(3.0, observation),
            initial_level=float(z[0]),
            initial_slope=0.0,
            initial_level_variance=4.0,
            initial_slope_variance=0.25,
        )
    order = candidate.order
    mean = np.zeros(order + 1)
    precision = np.eye(order + 1) * (10.0 if order >= 12 else 1.5)
    precision[0, 0] = 0.5
    if order >= 12:
        # A weakly persistent prior at the declared seasonal lag stabilizes the
        # high-order conditional regression without fixing the coefficient.
        mean[-1] = 0.5
        precision[-1, -1] = 3.0
    prior = rmc.NormalInverseGammaPrior(mean, precision, 3.0, 0.70)
    return rmc.BayesianAR(order=order, prior=prior)


def fit_and_forecast(
    candidate: Candidate,
    y: np.ndarray,
    horizon: int,
    *,
    chains: int,
    draws: int,
    warmup: int,
    seed: int,
) -> tuple[np.ndarray, float, float]:
    z, center, scale = standardize(y)
    model = make_model(candidate, z)
    started = time.perf_counter()
    if candidate.family == "ar":
        fit = model.fit(z, chains=chains, draws=draws, seed=seed)
    else:
        fit = model.fit(z, chains=chains, draws=draws, warmup=warmup, seed=seed)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    forecast = fit.forecast(steps=horizon, seed=seed + 1)
    paths = np.asarray(forecast.observation_samples).reshape(-1, horizon)
    paths = paths * scale + center
    forecast_seconds = time.perf_counter() - started
    return paths, fit_seconds, forecast_seconds


def interval_score(
    truth: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float
) -> np.ndarray:
    return (
        upper
        - lower
        + (2.0 / alpha) * (lower - truth) * (truth < lower)
        + (2.0 / alpha) * (truth - upper) * (truth > upper)
    )


def weighted_interval_score(paths: np.ndarray, truth: np.ndarray) -> float:
    median = np.median(paths, axis=0)
    total = 0.5 * np.abs(truth - median)
    denominator = 0.5
    for alpha in (0.20, 0.05):
        lower, upper = np.quantile(paths, [alpha / 2.0, 1.0 - alpha / 2.0], axis=0)
        total += (alpha / 2.0) * interval_score(truth, lower, upper, alpha)
        denominator += alpha / 2.0
    return float(np.mean(total / denominator))


def rolling_select(y: np.ndarray, period: int, seed: int) -> tuple[Candidate, dict[str, float]]:
    origins = (15, 18, 21) if period == 12 else (65, 78, 91)
    horizon = period // 4
    scores: dict[str, float] = {}
    for candidate_index, candidate in enumerate(base_candidates(period)):
        fold_scores = []
        for fold_index, origin in enumerate(origins):
            fold_horizon = min(horizon, y.size - origin)
            try:
                paths, _, _ = fit_and_forecast(
                    candidate,
                    y[:origin],
                    fold_horizon,
                    chains=2,
                    draws=250,
                    warmup=250,
                    seed=seed + candidate_index * 100 + fold_index * 10,
                )
                fold_scores.append(weighted_interval_score(paths, y[origin : origin + fold_horizon]))
            except Exception:
                fold_scores.append(float("inf"))
        scores[candidate.name] = float(np.mean(fold_scores))
    selected_name = min(scores, key=scores.get)
    return next(item for item in base_candidates(period) if item.name == selected_name), scores


def seasonal_diagnostic(y: np.ndarray, period: int) -> dict[str, float]:
    t = np.arange(y.size, dtype=float)
    design = np.column_stack([np.ones(y.size), t, t * t])
    residual = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    first = residual[:period]
    second = residual[period : 2 * period]
    correlation = float(np.corrcoef(first, second)[0, 1])
    phase_mean = (first + second) / 2.0
    amplitude_ratio = float(np.sqrt(np.mean(phase_mean**2)) / np.sqrt(np.mean(residual**2)))
    return {"detrended_cycle_correlation": correlation, "seasonal_amplitude_ratio": amplitude_ratio}


def seasonal_drift_forecast(
    y: np.ndarray, horizon: int, period: int, diagnostic: dict[str, float], seed: int
) -> tuple[np.ndarray, float, float, dict[str, float]]:
    paired_slopes = (y[period : 2 * period] - y[:period]) / period
    drift = float(np.median(paired_slopes))
    time_index = np.arange(y.size, dtype=float)
    detrended = y - drift * time_index
    z, center, scale = standardize(detrended)
    phase = np.asarray([z[np.arange(z.size) % period == index].mean() for index in range(period)])
    phase -= phase.mean()
    shrinkage = float(np.clip(diagnostic["detrended_cycle_correlation"], 0.25, 1.0))
    initial_effects = phase * shrinkage
    responsive = diagnostic["detrended_cycle_correlation"] < 0.60
    variance_scales = (0.12, 0.05, 0.40) if responsive else (0.04, 0.015, 0.65)
    model = rmc.BayesianSeasonalLocalLevel(
        period=period,
        level_variance_prior=rmc.InverseGammaPrior(3.0, variance_scales[0]),
        seasonal_variance_prior=rmc.InverseGammaPrior(3.0, variance_scales[1]),
        observation_variance_prior=rmc.InverseGammaPrior(3.0, variance_scales[2]),
        initial_level=float(np.mean(z[:period])),
        initial_seasonal_effects=initial_effects,
        initial_level_variance=4.0,
        initial_seasonal_variance=0.25,
    )
    started = time.perf_counter()
    fit = model.fit(z, chains=4, draws=1_000, warmup=500, seed=seed)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    forecast = fit.forecast(steps=horizon, seed=seed + 1)
    paths = np.asarray(forecast.observation_samples).reshape(-1, horizon) * scale + center
    paths += drift * np.arange(y.size, y.size + horizon)
    forecast_seconds = time.perf_counter() - started
    details = {
        "paired_cycle_median_drift_per_step": drift,
        "initial_seasonal_shrinkage": shrinkage,
        "level_variance_prior_scale": variance_scales[0],
        "seasonal_variance_prior_scale": variance_scales[1],
        "observation_variance_prior_scale": variance_scales[2],
    }
    return paths, fit_seconds, forecast_seconds, details


def pointwise_hdi(paths: np.ndarray, probability: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    sorted_paths = np.sort(paths, axis=0)
    sample_count = sorted_paths.shape[0]
    included = int(np.floor(probability * sample_count))
    widths = sorted_paths[included:, :] - sorted_paths[: sample_count - included, :]
    starts = np.argmin(widths, axis=0)
    columns = np.arange(sorted_paths.shape[1])
    return sorted_paths[starts, columns], sorted_paths[starts + included, columns]


def accuracy(mean: np.ndarray, truth: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> dict[str, float]:
    error = mean - truth
    denominator = np.maximum(np.abs(mean) + np.abs(truth), 1e-12)
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "smape_percent": float(100.0 * np.mean(2.0 * np.abs(error) / denominator)),
        "hdi_coverage": float(np.mean((truth >= lower) & (truth <= upper))),
        "mean_hdi_width": float(np.mean(upper - lower)),
    }


def run() -> dict[str, dict]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    forecast_rows: list[dict[str, str | float]] = []
    for dataset_index, path in enumerate(sorted(DATA_DIR.glob("*.csv"))):
        rows, train, truth = load_dataset(path)
        period = 12 if path.stem.startswith("monthly") else 52
        horizon = truth.size
        seed = SEED + dataset_index * 100_000
        selection_started = time.perf_counter()
        rolling_candidate, rolling_scores = rolling_select(train, period, seed)
        diagnostic = seasonal_diagnostic(train, period)
        # Prefer the declared annual recurrence when it is essentially tied on
        # rolling WIS and the two observed cycles agree. This is a structural
        # one-standard-error-style tie break, not a held-out-test decision.
        if (
            period == 52
            and diagnostic["detrended_cycle_correlation"] >= 0.60
            and diagnostic["seasonal_amplitude_ratio"] >= 0.80
            and rolling_scores["seasonal_ar52"] <= 1.05 * min(rolling_scores.values())
        ):
            rolling_candidate = next(
                item for item in base_candidates(period) if item.name == "seasonal_ar52"
            )
        use_fitted_seasonal = (
            period == 12
            and diagnostic["detrended_cycle_correlation"] >= 0.35
            and diagnostic["seasonal_amplitude_ratio"] >= 0.65
        )
        selection_seconds = time.perf_counter() - selection_started
        if use_fitted_seasonal:
            paths, fit_seconds, forecast_seconds, details = seasonal_drift_forecast(
                train, horizon, period, diagnostic, seed + 50_000
            )
            selected = "drift_adjusted_bayesian_seasonal_local_level"
        else:
            paths, fit_seconds, forecast_seconds = fit_and_forecast(
                rolling_candidate,
                train,
                horizon,
                chains=4,
                draws=1_000,
                warmup=500,
                seed=seed + 50_000,
            )
            selected = rolling_candidate.name
            details = {}
        mean = paths.mean(axis=0)
        lower, upper = pointwise_hdi(paths, 0.95)
        test_rows = [row for row in rows if row["split"] == "test"]
        for row, forecast_mean, low, high in zip(test_rows, mean, lower, upper, strict=True):
            forecast_rows.append(
                {
                    "dataset": path.stem,
                    "date": row["date"],
                    "mean": float(forecast_mean),
                    "lower_95_hdi": float(low),
                    "upper_95_hdi": float(high),
                }
            )
        results[path.stem] = {
            "selected_model": selected,
            "selection_seconds_excluded_from_reported_fit": selection_seconds,
            "fit_seconds": fit_seconds,
            "forecast_seconds": forecast_seconds,
            "posterior_predictive_draws": int(paths.shape[0]),
            "interval": "pointwise 95% HDI of future-observation posterior-predictive draws",
            "training_only_seasonal_diagnostic": diagnostic,
            "rolling_candidate": rolling_candidate.name,
            "rolling_candidate_wis": rolling_scores,
            "model_details": details,
            "metrics": accuracy(mean, truth, lower, upper),
            "mean": mean.tolist(),
            "lower_95_hdi": lower.tolist(),
            "upper_95_hdi": upper.tolist(),
        }
        print(
            f"{path.stem}: {selected}; fit={fit_seconds:.6f}s; "
            f"forecast={forecast_seconds:.6f}s; MAE={results[path.stem]['metrics']['mae']:.3f}"
        )

    result_path = RESULTS_DIR / "rustmc_results.json"
    result_path.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    csv_path = RESULTS_DIR / "rustmc_forecasts.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(forecast_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(forecast_rows)
    return results


if __name__ == "__main__":
    run()
