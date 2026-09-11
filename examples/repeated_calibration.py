"""Reuse one calibration model for independent instruments with different sample counts."""
import numpy as np
from instrument_calibration import calibration_model


def main():
    rng = np.random.default_rng(42)
    datasets = []
    ids = []
    for i, rows in enumerate([40, 75, 120]):
        x = np.linspace(-2, 2, rows)
        y = 0.1 * i + (1.0 + 0.05 * i) * x + rng.normal(0, 0.2, rows)
        ids.append(f"instrument-{i}")
        datasets.append({"x": x, "y": y})
    batch = calibration_model().sample_batch(
        datasets, ids=ids, chains=4, warmup=1000, draws=1000,
        threads=2, seed=42, errors="collect", show_progress=False,
    )
    for name in batch.ids:
        if name in batch.errors:
            print(name, batch.errors[name])
        else:
            print(name, batch.get(name).summary())


if __name__ == "__main__":
    main()
