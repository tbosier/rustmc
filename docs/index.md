# rustmc

**Fast Bayesian inference in Rust with a Python API.**

rustmc runs the entire sampling loop in compiled Rust — no Python in the inner loop. Chains are parallelized across threads via Rayon. The result is fast enough to fit thousands of independent Bayesian models in a single call.

```
10,000 Bayesian demand models in 70 seconds, with full posterior uncertainty.
```

## Why rustmc?

PyMC, Stan, and other Bayesian frameworks are built for single-model workflows. You define one model, fit it, analyze it. This works well for research but falls apart when you need to fit the same model structure to thousands of datasets — per-store demand models, per-SKU pricing models, per-patient dosing models.

rustmc is designed for that use case. Its batch inference API runs 10,000 independent NUTS chains through a single Rayon thread pool, sharing compute across all available cores with zero serialization overhead.

## Benchmarks

**Single model** — 10 parameters, 100,000 observations, 8 chains, 2,000 draws:

| Method | Time | Speedup |
|--------|------|---------|
| rustmc (NUTS) | 72s | **5.3x** |
| PyMC (NUTS) | 383s | 1.0x |

**Batch inference** — 10,000 independent 3-parameter models:

| Method | Total time | Per model | Uncertainty |
|--------|-----------|-----------|-------------|
| rustmc (batch NUTS) | 70s | 7ms | Yes (full posterior) |
| ARIMA (sequential) | 160s | 16ms | No |
| Prophet (sequential) | 28min | 170ms | Partial |

## Install

```bash
pip install rustmc
```

## Quick Example

```python
import numpy as np
import rustmc as rmc

x = np.random.randn(1000)
y = 2.5 * x + np.random.randn(1000)

builder = rmc.ModelBuilder()
beta = builder.normal_prior("beta", mu=0.0, sigma=1.0)
builder.normal_likelihood("obs", mu_expr=beta * "x", sigma=1.0, observed_key="y")
model = builder.build()

fit = rmc.sample(model_spec=model, data={"x": x, "y": y}, chains=4, draws=1000)
print(fit.summary())
```

```
4 chains x 1000 draws per chain

Parameter        mean      std     hdi_3%    hdi_97%   ess_bulk   ess_tail    r_hat  mcse_mean
-----------------------------------------------------------------------------------------------
beta           2.4575   0.0313     2.3982     2.5133       2638       2966   1.0055   0.000610
-----------------------------------------------------------------------------------------------
Mean accept rate: 0.94  |  Divergences: 0
```

## What's Implemented

**Sampling:** NUTS with multinomial candidate selection, diagonal mass matrix adaptation, dual-averaging step size, multi-chain parallelism via Rayon.

**Distributions:** Normal, StudentT, HalfNormal, Gamma, Beta, Uniform, Bernoulli, Poisson. Constrained distributions are automatically sampled in unconstrained space.

**Diagnostics:** Split R-hat (Vehtari et al. 2021), bulk/tail ESS, MCSE, 94% HDI, divergence detection, per-chain acceptance rates.

**High-dimensional regression:** faer-backed `MatVecMul` op — `beta @ "X"` dispatches to a BLAS-level GEMV rather than N scalar graph nodes.
