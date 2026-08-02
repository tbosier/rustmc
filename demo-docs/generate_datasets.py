"""Generate deterministic synthetic data for the rustmc forecasting study.

The complete series is generated once, then split into a training window and a held-out
future. Model selection must use only rows whose split is ``train``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MASTER_SEED = 20_260_802


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    frequency: str
    train_size: int
    horizon: int
    start: np.datetime64
    step: np.timedelta64


SPECS = (
    DatasetSpec("monthly_easy", "monthly", 24, 6, np.datetime64("2022-01", "M"), np.timedelta64(1, "M")),
    DatasetSpec("monthly_medium", "monthly", 24, 6, np.datetime64("2022-01", "M"), np.timedelta64(1, "M")),
    DatasetSpec("monthly_hard", "monthly", 24, 6, np.datetime64("2022-01", "M"), np.timedelta64(1, "M")),
    DatasetSpec("weekly_easy", "weekly", 104, 26, np.datetime64("2022-01-03"), np.timedelta64(7, "D")),
    DatasetSpec("weekly_medium", "weekly", 104, 26, np.datetime64("2022-01-03"), np.timedelta64(7, "D")),
    DatasetSpec("weekly_hard", "weekly", 104, 26, np.datetime64("2022-01-03"), np.timedelta64(7, "D")),
)


def _ar1_noise(rng: np.random.Generator, size: int, phi: float, sigma: float) -> np.ndarray:
    innovations = rng.normal(0.0, sigma, size=size)
    values = np.empty(size, dtype=np.float64)
    values[0] = innovations[0] / np.sqrt(1.0 - phi**2)
    for index in range(1, size):
        values[index] = phi * values[index - 1] + innovations[index]
    return values


def _monthly(level: str, rng: np.random.Generator, size: int) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(size, dtype=np.float64)
    annual = 2.0 * np.pi * (t - 2.0) / 12.0
    if level == "easy":
        mean = 80.0 + 1.10 * t + 8.0 * np.sin(annual)
        noise = rng.normal(0.0, 1.5, size=size)
    elif level == "medium":
        mean = (
            120.0
            + 0.35 * t
            + 0.025 * t**2
            + (7.0 + 0.10 * t) * np.sin(annual)
            + 3.0 * np.cos(2.0 * annual)
        )
        noise = _ar1_noise(rng, size, phi=0.45, sigma=3.5)
    elif level == "hard":
        mean = (
            95.0
            + 0.45 * t
            + 8.0 * np.tanh((t - 12.0) / 3.0)
            + 12.0 * np.sin(annual)
            + 5.0 * np.sin(2.0 * annual + 0.7)
        )
        noise = rng.normal(0.0, 4.0 + 0.10 * t, size=size)
        noise[[7, 19]] += np.asarray([12.0, -15.0])
    else:  # pragma: no cover - internal specification error
        raise ValueError(level)
    return mean, mean + noise


def _weekly(level: str, rng: np.random.Generator, size: int) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(size, dtype=np.float64)
    annual = 2.0 * np.pi * (t - 5.0) / 52.0
    quarterly = 2.0 * np.pi * t / 13.0
    if level == "easy":
        mean = 200.0 + 0.35 * t + 18.0 * np.sin(annual)
        noise = rng.normal(0.0, 3.0, size=size)
    elif level == "medium":
        mean = (
            160.0
            + 0.15 * t
            + 0.002 * t**2
            + 15.0 * np.sin(annual)
            + 7.0 * np.cos(quarterly + 0.4)
        )
        noise = _ar1_noise(rng, size, phi=0.60, sigma=5.0)
    elif level == "hard":
        recurring_pulse = 18.0 * np.exp(-0.5 * ((np.mod(t, 26.0) - 2.0) / 1.5) ** 2)
        mean = (
            240.0
            + 0.10 * t
            + 10.0 * np.tanh((t - 60.0) / 8.0)
            + 22.0 * np.sin(annual + 0.3)
            + 10.0 * np.cos(quarterly - 0.6)
            + recurring_pulse
        )
        noise = _ar1_noise(rng, size, phi=0.35, sigma=7.0)
        noise *= 1.0 + 0.003 * t
        noise[[31, 82]] += np.asarray([-20.0, 24.0])
    else:  # pragma: no cover - internal specification error
        raise ValueError(level)
    return mean, mean + noise


def generate() -> list[Path]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    seed_sequence = np.random.SeedSequence(MASTER_SEED)
    paths: list[Path] = []
    for spec, child_seed in zip(SPECS, seed_sequence.spawn(len(SPECS)), strict=True):
        size = spec.train_size + spec.horizon
        rng = np.random.default_rng(child_seed)
        difficulty = spec.name.rsplit("_", 1)[1]
        if spec.frequency == "monthly":
            latent_mean, observed = _monthly(difficulty, rng, size)
            dates = np.arange(size).astype("timedelta64[M]") + spec.start
        else:
            latent_mean, observed = _weekly(difficulty, rng, size)
            dates = np.arange(size) * spec.step + spec.start

        path = DATA_DIR / f"{spec.name}.csv"
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(("date", "value", "latent_mean", "split"))
            for index, (date, value, mean) in enumerate(
                zip(dates, observed, latent_mean, strict=True)
            ):
                writer.writerow(
                    (
                        str(date),
                        f"{value:.12f}",
                        f"{mean:.12f}",
                        "train" if index < spec.train_size else "test",
                    )
                )
        paths.append(path)
    return paths


if __name__ == "__main__":
    for generated_path in generate():
        print(generated_path.relative_to(ROOT))
