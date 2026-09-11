"""Estimate an instrument's offset, gain, and noise; predict new readings."""
import numpy as np
import rustmc as rmc


def calibration_model():
    model = rmc.ModelBuilder()
    offset = model.normal_prior("offset", 0.0, 1.0)
    gain = model.normal_prior("gain", 1.0, 0.5)
    noise = model.half_normal_prior("noise", 0.5)
    model.normal_likelihood("reading", offset + gain * "x", noise, "y")
    return model.compile()


def main():
    rng = np.random.default_rng(42)
    x = np.linspace(-2, 2, 100)
    y = 0.3 + 1.2 * x + rng.normal(0, 0.2, len(x))
    fit = calibration_model().sample(
        {"x": x, "y": y}, chains=4, warmup=1000, draws=1000,
        seed=42, show_progress=False,
    )
    print(fit.summary())
    prediction = fit.predict({"x": np.array([-1.0, 0.0, 1.0])}, seed=43)
    print("95% predictive intervals:")
    print(np.quantile(prediction["reading"], [0.025, 0.975], axis=(0, 1)))


if __name__ == "__main__":
    main()
