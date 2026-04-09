# rustmc

**Fast Bayesian inference in Rust with a Python API.**

rustmc runs the entire sampling loop in compiled Rust - no Python in the inner loop. Chains are parallelized across threads via Rayon. The result is designed for fitting many independent Bayesian models in a single call.

```
10,000 Bayesian demand models in 70 seconds, with full posterior uncertainty.
```

## Why rustmc?

PyMC, Stan, and other Bayesian frameworks are built for general single-model workflows. That is useful for research, but it becomes expensive when you need to fit the same model structure to thousands of datasets - per-store demand models, per-SKU pricing models, per-patient dosing models.

rustmc is designed for that repeated-model setting. Its batch inference API runs many independent models through a single Rayon thread pool, sharing compute across all available cores with low orchestration overhead.

## Differentiation

- Rust-native inference loop with no Python in the hot path.
- Batch execution tuned for repeated independent models, not just single-model sampling.
- Graph-based execution with cached buffers, transforms, and Jacobians.
- Fast paths for regression-style models and high-dimensional `X @ beta` workloads.
- Built-in diagnostics, predictive checks, pointwise log-likelihood, and ArviZ export aimed at production validation.

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

**Sampling:** NUTS with multinomial candidate selection, block-structured mass matrix adaptation, dual-averaging step size, fixed-step HMC fallback, and multi-chain parallelism via Rayon.

**Priors:** Normal, HalfNormal, Exponential, LogNormal, StudentT, Gamma, Beta, Uniform, Bernoulli, Poisson. Constrained distributions are automatically sampled in unconstrained space.

**Likelihoods:** Normal, Bernoulli-logit, Poisson-log, Exponential-log, LogNormal, and NegativeBinomial-log.

**Modeling surface:** scalar hierarchical priors are supported, and scalar hierarchical normals are automatically compiled through a non-centered path where appropriate. Vector-valued hierarchical blocks are still on the roadmap.

**Predictive workflow:** prior predictive sampling, posterior predictive sampling, pointwise log-likelihood, and ArviZ export are implemented for the current likelihood families.

**Diagnostics:** Split R-hat (Vehtari et al. 2021), bulk/tail ESS, MCSE, 94% HDI, divergence detection, per-chain acceptance rates.

**High-dimensional regression:** faer-backed `MatVecMul` op — `beta @ "X"` dispatches to a BLAS-level GEMV rather than N scalar graph nodes.

## What Is Still Missing

- Vector-valued hierarchical models and richer automatic reparameterization support.
- Compiled model artifacts and a Python-free deployment story in the public API.
- Benchmark regression gates and packaging/release automation.
- Higher-level production templates for repeated forecasting and panel workflows.
