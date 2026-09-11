# Get started

Install the Python package with `pip install rustmc`. NumPy is required; ArviZ and
Matplotlib are optional through `pip install "rustmc[viz]"`.

## Estimate an instrument's gain

```python
import numpy as np
import rustmc as rmc

rng = np.random.default_rng(42)
x = np.linspace(-2, 2, 100)
y = 0.3 + 1.2 * x + rng.normal(0, 0.2, len(x))

model = rmc.ModelBuilder()
offset = model.normal_prior("offset", 0.0, 1.0)
gain = model.normal_prior("gain", 1.0, 0.5)
noise = model.half_normal_prior("noise", 0.5)
model.normal_likelihood("reading", offset + gain * "x", noise, "y")
compiled = model.compile()
fit = compiled.sample(
    {"x": x, "y": y}, chains=4, warmup=1000, draws=1000,
    seed=42, show_progress=False,
)
print(fit.summary())
```

The posterior estimates the offset, gain, and measurement noise together. Choose
priors in the units of your measurements.

Inspect R-hat, bulk/tail effective sample size, Monte Carlo error, and divergences.
R-hat near one and enough effective samples are necessary checks, not proof that the
model describes your data. Compare predictive draws with observations too.

```python
prediction = fit.predict({"x": np.array([-1.0, 0.0, 1.0])}, seed=43)
interval = np.quantile(prediction["reading"], [0.025, 0.975], axis=(0, 1))
print(interval)
```

Draws have `(chain, draw, observation)` axes. `expected=True` returns conditional
means; the default also samples observation noise.

Reuse `compiled` for another instrument. Use a [batch](examples/repeated-calibration.md)
for independent fits or one [partial-pooling model](examples/site-effects.md) for
related groups. See [custom models](custom-models.md) for the full construction API.

For source builds, follow the repository's
[contributing guide](https://github.com/tbosier/rustmc/blob/main/CONTRIBUTING.md).
