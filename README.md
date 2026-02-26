# rustmc

Bayesian inference engine written in Rust. Python API via PyO3.

rustmc runs the entire sampling loop in compiled Rust, with no Python in the inner loop. Chains are parallelized across threads using Rayon. The result is fast enough to fit thousands of independent Bayesian models in a single call.

## Why rustmc

PyMC, Stan, and other Bayesian frameworks are built for single-model workflows. You define one model, fit it, and analyze it. This works well for research but falls apart when you need to fit the same model structure to thousands of datasets -- per-store demand models, per-SKU pricing models, per-patient dosing models.

rustmc is designed for that use case. It provides a batch inference API that runs 10,000 independent NUTS chains through a single Rayon thread pool, sharing compute across all available cores with zero serialization overhead.

**10,000 Bayesian demand models in 70 seconds, with full posterior uncertainty.**

Fitting those same 10,000 models sequentially with ARIMA takes ~160 seconds. With Prophet, ~28 minutes. Neither gives you credible intervals for free.

## Benchmark

10 parameters, 100,000 observations, 8 chains, 2,000 draws:

| Method | Time | Speedup |
|--------|------|---------|
| rustmc (NUTS) | 72s | 5.3x |
| PyMC (NUTS) | 383s | 1.0x |

Batch inference, 10,000 independent 3-parameter models:

| Method | Total time | Per model | Uncertainty |
|--------|-----------|-----------|-------------|
| rustmc (batch NUTS) | 70s | 7ms | Yes (full posterior) |
| ARIMA (sequential) | 160s | 16ms | No |
| Prophet (sequential) | 28min | 170ms | Partial |

## Quick start

```bash
pip install maturin
git clone https://github.com/your-username/rustmc.git
cd rustmc
python -m venv .venv && source .venv/bin/activate
pip install numpy maturin
maturin develop --manifest-path python_bindings/Cargo.toml --release
```

### Single model

```python
import numpy as np
import rustmc as rmc

np.random.seed(42)
x = np.random.randn(1000)
y = 2.5 * x + np.random.randn(1000)

builder = rmc.ModelBuilder()
beta = builder.normal_prior("beta", mu=0.0, sigma=1.0)
mu_expr = beta * "x"
builder.normal_likelihood("obs", mu_expr=mu_expr, sigma=1.0, observed_key="y")
model = builder.build()

fit = rmc.sample(model_spec=model, data={"x": x, "y": y}, chains=4, draws=1000)
print(fit.summary())
```

Output:

```
4 chains x 1000 draws per chain

Parameter        mean      std     hdi_3%    hdi_97%   ess_bulk   ess_tail    r_hat  mcse_mean
-----------------------------------------------------------------------------------------------
beta           2.4575   0.0313     2.3982     2.5133       2638       2966   1.0055   0.000610
-----------------------------------------------------------------------------------------------
Mean accept rate: 0.94  |  Divergences: 0
```

### Batch inference (10,000 models)

```python
import rustmc as rmc
import numpy as np

models = []
for i in range(10_000):
    builder = rmc.ModelBuilder()
    intercept = builder.normal_prior("intercept", mu=0.0, sigma=200.0)
    trend = builder.normal_prior("trend", mu=0.0, sigma=20.0)
    mu_expr = intercept + trend * "t"
    builder.normal_likelihood("obs", mu_expr=mu_expr, sigma=5.0, observed_key="y")
    model = builder.build()

    t = np.arange(52, dtype=np.float64) / 52
    y = some_data[i]  # your per-SKU time series
    models.append((model, {"t": t, "y": y}))

results = rmc.batch_sample(models, draws=500, warmup=300)

# Each result has .mean(), .std(), .get_samples()
for r in results[:5]:
    print(r)
```

## What is implemented

### Sampling

- NUTS (No-U-Turn Sampler) with multinomial candidate selection, generalized U-turn criterion, and divergence detection. Follows Hoffman and Gelman (2014) and Betancourt (2017).
- HMC with fixed leapfrog steps, available as a fallback via `sampler="hmc"`.
- Diagonal mass matrix adaptation with 3-phase warmup (step-size only, mass matrix estimation, final step-size tuning).
- Auto step-size initialization via binary search.
- Deterministic per-chain RNG (ChaCha8) for reproducible results.
- Multithreaded chains via Rayon. Batch inference shares the thread pool across all models.

### Distributions

| Distribution | Support | Transform | Status |
|-------------|---------|-----------|--------|
| Normal | (-inf, inf) | None | Working |
| StudentT | (-inf, inf) | None | Working |
| HalfNormal | (0, inf) | log | Working |
| Gamma | (0, inf) | log | Working |
| Beta | (0, 1) | logit | Working |
| Uniform | (a, b) | logit | Working |
| Bernoulli | {0, 1} | None | Discrete, limited |
| Poisson | {0, 1, 2, ...} | None | Discrete, limited |

Constrained distributions are automatically sampled in unconstrained space via log/logit transforms with Jacobian corrections. Samples are back-transformed before being returned to the user.

### Computation

- Computational graph with reverse-mode automatic differentiation.
- Fused linear combination op for regression models. Replaces N separate multiply-add passes with a single cache-friendly loop over the data.
- Zero-allocation evaluator. All vector intermediates are pre-allocated in a flat buffer and reused across gradient evaluations. No heap allocation in the sampling loop.

### Diagnostics

- Split R-hat with rank normalization (Vehtari et al. 2021).
- Bulk and tail effective sample size (ESS).
- Monte Carlo standard error (MCSE).
- 94% highest density interval.
- Per-chain acceptance rates, step sizes, and divergence counts.
- Automatic warnings for convergence issues.

Available via `fit.summary()` for a formatted table or `fit.diagnostics()` for programmatic access.

### Progress reporting

Live progress bar rendered from Rust at 10 Hz using atomic counters, with no GIL involvement:

```
Sampling 8 chains ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% | 24.0k/24.0k | 0 divergences | 384.0k grad evals | 6.7s
```

## Architecture

```
Python (orchestration only)
  |
  v  GIL released
Rust Core
  +-- Graph         Computational DAG, nodes, ops, data storage
  +-- Autodiff      Forward evaluation + reverse-mode gradient
  +-- Distributions  8 distributions with automatic transforms
  +-- NUTS          Multinomial tree-building, U-turn detection
  +-- HMC           Fixed-step leapfrog (fallback)
  +-- Sampler       Multi-chain parallel runner, batch inference
  +-- Diagnostics   R-hat, ESS, MCSE, HDI
  +-- Progress      Atomic counters, background render thread
```

Design principles:

- Model graph is built once and shared read-only across chains.
- Sampler accepts any log-probability + gradient function derived from a Graph.
- No global state. All state is explicit and owned.
- Deterministic RNG per chain (ChaCha8 seeded from base_seed + chain_index).
- Parameter transforms and Jacobian corrections are handled in the graph, not the sampler.

## Roadmap

Near term:

- Hierarchical priors (parameter as hyperparameter of another parameter's prior)
- Link functions and GLMs
- Custom likelihood functions
- Prior and posterior predictive sampling
- LOO-CV (Pareto-smoothed importance sampling)
- Trace plots and visual diagnostics
- PyPI package (`pip install rustmc`)

Medium term:

- Sufficient statistics optimization for linear-Normal models
- MAP estimation (L-BFGS)
- Laplace approximation
- Sparse indicator variable support
- Stochastic gradient MCMC (SGLD/SGHMC) for large datasets
- Model serialization (compile once, deploy without Python)

Long term:

- Variational inference (ADVI)
- GPU-accelerated log-probability via wgpu
- WASM compilation for browser/edge inference
- Distributed posterior aggregation
- Automatic reparameterization for funnel geometries
- C FFI for embedding in non-Python systems

## License

MIT
