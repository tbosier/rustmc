from pathlib import Path

import numpy as np
from gp_prophet_baselines import (
    forecast_gp,
    forecast_metrics,
    forecast_prophet,
    load_common_six,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def test_synthetic_protocol_shapes_are_exact():
    series = load_common_six(DATA_DIR)
    assert [item.name for item in series] == [
        "monthly_easy",
        "monthly_medium",
        "monthly_hard",
        "weekly_easy",
        "weekly_medium",
        "weekly_hard",
    ]
    for item in series:
        item.validate()


def test_metrics_reward_exact_covered_forecast():
    observed = np.array([1.0, 2.0, 3.0])
    metrics = forecast_metrics(observed, observed, observed - 1, observed + 1)
    assert metrics["rmse"] == 0
    assert metrics["coverage_95"] == 1
    assert metrics["wis_95"] == 2


def test_gp_returns_ordered_predictive_interval():
    series = load_common_six(DATA_DIR)[0]
    result = forecast_gp(series, seed=7)
    assert result.mean.shape == series.test.shape
    assert np.all(result.lower <= result.mean)
    assert np.all(result.mean <= result.upper)
    assert result.fit_forecast_seconds > 0


def test_prophet_returns_ordered_uncertainty_interval():
    series = load_common_six(DATA_DIR)[0]
    result = forecast_prophet(series, seed=7)
    assert result.mean.shape == series.test.shape
    assert np.all(result.lower <= result.mean)
    assert np.all(result.mean <= result.upper)
    assert result.fit_forecast_seconds > 0
