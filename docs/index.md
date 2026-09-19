# rustmc

Bayesian models in Python. Inference in Rust.

rustmc is for small, structured models that you fit again and again: regressions,
calibration curves, group comparisons, and time-series forecasts. You describe the
model once, compile it once, and then fit it to one dataset after another — a new
instrument, a new site, a new week of data — keeping the posterior draws for
prediction and diagnostics.

```bash
pip install rustmc
```

NumPy is the only required dependency. Python 3.9–3.13 are covered by install tests.

## Is this the right library for you?

It probably suits you if you fit the same small model shape repeatedly, in a service
or a scheduled job; if you want posterior uncertainty carried through to predictions
rather than a point estimate; or if you need inference to run inside a Rust process.

It probably does not suit you if you are exploring model space. The modeling language
is deliberately small — a handful of distributions, elementwise arithmetic, group
indexing, and matrix products. [PyMC](https://www.pymc.io) and
[Stan](https://mc-stan.org) cover far more, have much larger ecosystems, and are the
right default for a model you have not written before. rustmc is not trying to replace
them. Come here once you know the model and the bottleneck is fitting it often.

The project is alpha. The Python package is supported; the Rust API is still changing.

## A complete example

This estimates an instrument's offset, gain, and measurement noise, then reuses the
same compiled model on a shorter dataset from a second instrument.

```python
import numpy as np
import rustmc as rmc

rng = np.random.default_rng(42)
x = np.linspace(-2, 2, 100)
y = 0.3 + 1.2 * x + rng.normal(0, 0.2, x.size)

model = rmc.ModelBuilder()
offset = model.normal_prior("offset", 0.0, 1.0)
gain = model.normal_prior("gain", 1.0, 0.5)
noise = model.half_normal_prior("noise", 0.5)
model.normal_likelihood("reading", offset + gain * "x", noise, "y")
compiled = model.compile()

fit = compiled.sample(
    {"x": x, "y": y}, chains=4, warmup=1000, draws=1000, seed=42,
    show_progress=False,
)
print(fit.summary())

# Same compiled model, a different instrument with fewer readings.
second = compiled.sample(
    {"x": x[:40], "y": y[:40]}, chains=4, warmup=1000, draws=1000, seed=7,
    show_progress=False,
)
print(second.summary())
```

```text
4 chains × 1000 draws per chain

Parameter        mean      std     hdi_3%    hdi_97%   ess_bulk   ess_tail    r_hat  mcse_mean
────────────────────────────────────────────────────────────────────────────────────────────────
offset         0.2899   0.0156     0.2581     0.3166       3805       2649   1.0007   0.000253
gain           1.1793   0.0136     1.1526     1.2038       4695       2901   1.0009   0.000198
noise          0.1560   0.0112     0.1347     0.1759       3037       3029   1.0012   0.000202
────────────────────────────────────────────────────────────────────────────────────────────────
Mean accept rate: 0.91  │  Divergences: 0

4 chains × 1000 draws per chain

Parameter        mean      std     hdi_3%    hdi_97%   ess_bulk   ess_tail    r_hat  mcse_mean
────────────────────────────────────────────────────────────────────────────────────────────────
offset         0.3474   0.0749     0.2056     0.4854       1605       1589   1.0039   0.001878
gain           1.2329   0.0573     1.1189     1.3350       1553       1595   1.0047   0.001457
noise          0.1714   0.0201     0.1349     0.2080       1729       1723   1.0035   0.000484
────────────────────────────────────────────────────────────────────────────────────────────────
Mean accept rate: 0.93  │  Divergences: 0
```

The `hdi_3%` and `hdi_97%` columns are the shortest intervals containing 94% of the
draws, not equal-tailed quantiles. R-hat near one, ample effective sample size, and
zero divergences say the sampler explored this posterior; they do not say the model
describes your data. Compare predictive draws against observations as well.

The data were generated with offset 0.3 and gain 1.2. The second fit, with 40 readings
instead of 100, recovers the same values with visibly wider intervals — that is the
point of keeping the posterior.

## Two engines, one result surface

Models you write with `ModelBuilder` compile to a differentiable graph and are fitted
by NUTS or HMC. The forecasting models — structural, seasonal, AR, dynamic GLM,
hurdle, runoff, and the Gaussian hierarchy — are separate hand-written samplers that do
not use that graph at all. Each exploits structure the general sampler cannot: Gibbs
with FFBS, exact conjugate draws, latent-count Gibbs, or block elliptical slice
sampling, depending on the model.

They share everything around inference: the state-space primitives, the diagnostics,
forecast evaluation, the batch executor, and one result and prediction API. So
`fit.summary()` reads the same way either way, but a change to NUTS does not change a
forecast. The one difference worth knowing: these samplers have no notion of a
divergence or an accept rate, so those fields are `None` on a forecasting fit. R-hat,
ESS and MCSE mean what they always did.

## Where to go next

- **[Get started](getting-started.md)** — install, fit the model above, and read the
  diagnostics.
- **[Custom graph models](custom-models.md)** — the full expression language,
  dimensions, prediction, and saving a model.
- **[Forecasting workflows](forecasting-workflows.md)** — fitting, predicting, and
  backtesting time series.
- **[API reference](reference.md)** — every public class and its stated limits.
