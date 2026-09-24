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

NumPy is the only required dependency. Python 3.9–3.14 are covered by install tests.

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

This estimates an instrument's offset, gain, and measurement noise, then refits the
same compiled model to a shorter dataset without rebuilding it.

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

# Same compiled model, fewer readings. A real second instrument would supply
# its own x and y here; this reuses the first 40 rows to keep the example short.
second = compiled.sample(
    {"x": x[:40], "y": y[:40]}, chains=4, warmup=1000, draws=1000, seed=7,
    show_progress=False,
)
print(second.summary())
```

```text
4 chains × 1000 draws per chain

Parameter        mean      std     hdi_3%    hdi_97%   ess_bulk   ess_tail    r_hat  mcse_mean
──────────────────────────────────────────────────────────────────────────────────────────────
offset         0.2896   0.0155     0.2599     0.3181       3901       2953   1.0006   0.000249
gain           1.1789   0.0132     1.1534     1.2033       4053       2940   1.0001   0.000209
noise          0.1559   0.0111     0.1351     0.1762       3453       2648   1.0012   0.000191
──────────────────────────────────────────────────────────────────────────────────────────────
Mean accept rate: 0.92  │  Divergences: 0
4 chains × 1000 draws per chain

Parameter        mean      std     hdi_3%    hdi_97%   ess_bulk   ess_tail    r_hat  mcse_mean
──────────────────────────────────────────────────────────────────────────────────────────────
offset         0.3503   0.0752     0.2023     0.4827       1292       1260   1.0051   0.002115
gain           1.2352   0.0582     1.1178     1.3369       1272       1462   1.0035   0.001645
noise          0.1716   0.0204     0.1336     0.2081       1661       1572   1.0030   0.000501
──────────────────────────────────────────────────────────────────────────────────────────────
Mean accept rate: 0.93  │  Divergences: 0
```

That output is from running exactly that script against rustmc 0.13.0 on Linux
x86-64. Sampling is deterministic for a given seed, build, and platform; digits in
the last places can differ elsewhere.

The `hdi_3%` and `hdi_97%` columns are the shortest intervals containing 94% of the
draws, not equal-tailed quantiles.

R-hat near one, ample effective sample size and zero divergences mean the chains that
ran agree with each other and show no obvious pathology. They cannot show that a region
of the posterior was never visited — four chains that all miss the same mode agree
perfectly — and they say nothing about whether the model describes your data. Treat
them as necessary, not sufficient, and compare predictive draws against observations
as well.

The data were generated with offset 0.3 and gain 1.2, and both fits cover those values.
The second is much less certain: the posterior standard deviation on `offset` goes from
0.0155 to 0.0752. Some of that is the smaller sample and some is the narrower predictor
range, since `x[:40]` spans only part of the original sweep — the two are not separable
here. The point is that the width is reported rather than assumed.

## Two inference paths

Models you write with `ModelBuilder` compile to a differentiable graph and are fitted
by NUTS or HMC. The forecasting models — structural, seasonal, AR, dynamic GLM,
hurdle, runoff, and the Gaussian hierarchy — are separate hand-written samplers that do
not use that graph at all. Each exploits structure the general sampler cannot: Gibbs
with FFBS, exact conjugate draws, block elliptical slice sampling, or a choice between
two kernels, depending on the model.

`diagnostics` is shared, so R-hat, ESS and MCSE are computed by the same code for every
fit, and `summary()` and `diagnostics()` read the same way wherever you find them. The
objects around them are not uniform: a graph fit's `predict()` returns a dict of arrays,
while a forecasting fit's `forecast(steps)` returns a forecast object, and `RunoffFit`
has neither. The guide for each model says what it returns.

Two things to know about a forecasting fit: it has no notion of a divergence or an
accept rate, so those fields are `None`; and where its draws are exact and independent
there is no warmup and no convergence period at all. `sampler_stats` says which kernel
ran.

## Where to go next

- **[Get started](getting-started.md)** — install, fit the model above, and read the
  diagnostics.
- **[Custom graph models](custom-models.md)** — the full expression language,
  dimensions, prediction, and saving a model.
- **[Forecasting workflows](forecasting-workflows.md)** — fitting, predicting, and
  backtesting time series.
- **[API reference](reference.md)** — per-class arguments, output shapes, and stated
  limits. It does not yet cover every public class; `StructuralModel` and
  `BayesianDynamicGLM` are documented in their guides instead.
