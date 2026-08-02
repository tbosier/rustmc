"""Render the six RustMC-only posterior forecast figures."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_PATH = ROOT / "results" / "rustmc_results.json"
IMAGES_DIR = ROOT / "images"


def parse_date(value: str) -> datetime:
    date_format = "%Y-%m-%d" if len(value) == 10 else "%Y-%m"
    return datetime.strptime(value, date_format)


def render() -> list[Path]:
    results = json.loads(RESULTS_PATH.read_text())
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "axes.edgecolor": "#334155",
            "axes.linewidth": 0.8,
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "text.color": "#0f172a",
        }
    )
    for data_path in sorted(DATA_DIR.glob("*.csv")):
        with data_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        train_rows = [row for row in rows if row["split"] == "train"]
        test_rows = [row for row in rows if row["split"] == "test"]
        weekly = data_path.stem.startswith("weekly")
        trailing_count = 52 if weekly else 12
        trailing_rows = train_rows[-trailing_count:]
        historical_dates = [parse_date(row["date"]) for row in trailing_rows]
        historical_values = np.asarray([float(row["value"]) for row in trailing_rows])
        future_dates = [parse_date(row["date"]) for row in test_rows]
        actual = np.asarray([float(row["value"]) for row in test_rows])
        forecast = results[data_path.stem]
        mean = np.asarray(forecast["mean"])
        lower = np.asarray(forecast["lower_95_hdi"])
        upper = np.asarray(forecast["upper_95_hdi"])

        figure, axis = plt.subplots(figsize=(12, 6.5), constrained_layout=True)
        axis.plot(
            historical_dates,
            historical_values,
            color="#334155",
            linewidth=2.0,
            marker="o",
            markersize=3.5,
            label=f"Observed history ({trailing_count} points)",
            zorder=4,
        )
        connector_dates = [historical_dates[-1], *future_dates]
        connector_mean = np.concatenate([[historical_values[-1]], mean])
        axis.plot(
            connector_dates,
            connector_mean,
            color="#2563eb",
            linewidth=2.5,
            label="RustMC posterior mean",
            zorder=5,
        )
        axis.fill_between(
            future_dates,
            lower,
            upper,
            color="#60a5fa",
            alpha=0.28,
            linewidth=0,
            label="Pointwise 95% posterior-predictive HDI",
            zorder=1,
        )
        axis.plot(
            future_dates,
            actual,
            color="#dc2626",
            linewidth=2.0,
            linestyle="--",
            marker="o",
            markersize=4.0,
            label="Held-out actual",
            zorder=6,
        )
        split_date = future_dates[0]
        axis.axvline(split_date, color="#64748b", linewidth=1.2, linestyle=":", zorder=2)
        title = data_path.stem.replace("_", " ").title()
        model_label = forecast["selected_model"].replace("_", " ")
        axis.set_title(f"RustMC Forecast: {title}\nSelected model: {model_label}", loc="left")
        axis.set_xlabel("Date")
        axis.set_ylabel("Synthetic value")
        axis.grid(axis="y", color="#cbd5e1", linewidth=0.8, alpha=0.6)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        if weekly:
            axis.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            axis.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        else:
            axis.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            axis.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        axis.legend(
            loc="upper left",
            frameon=True,
            framealpha=0.95,
            facecolor="white",
            edgecolor="#cbd5e1",
            ncols=2,
        )
        image_path = IMAGES_DIR / f"{data_path.stem}_rustmc_forecast.png"
        figure.savefig(image_path, dpi=180, facecolor="white")
        plt.close(figure)
        paths.append(image_path)
    return paths


if __name__ == "__main__":
    for path in render():
        print(path.relative_to(ROOT))
