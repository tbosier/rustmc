"""Stream independent instrument fits and keep only diagnostic summaries."""
import numpy as np
from instrument_calibration import calibration_model


def observations():
    rng = np.random.default_rng(42)
    for i in range(100):
        x = np.linspace(-2, 2, 40 + i % 30)
        y = .1 + 1.1*x + rng.normal(0, .2, len(x))
        yield f"instrument-{i}", {"x": x, "y": y}


def main():
    with calibration_model().sample_iter(
        observations(), retention="summary", parameters=["gain"], chunk_size=8,
        errors="collect", chains=4, warmup=1000, draws=1000, threads=4,
    ) as results:
        for item in results:
            print(item.id, item.error or item.diagnostics)


if __name__ == "__main__":
    main()
