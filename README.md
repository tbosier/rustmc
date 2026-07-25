# rustmc

Bayesian inference engine written in Rust. Python API via PyO3.

## Why rustmc

rustmc is built for production workloads where the same model structure is fit repeatedly:

- Rust-native inference loop with no Python in the hot path.
- Rayon-parallel chains and batch inference for repeated-model throughput.
- Graph-based execution with cached buffers, transforms, and Jacobians.
- Fast paths for linear regression and high-dimensional `X @ beta` models.
- Built-in diagnostics, predictive checks, pointwise log-likelihood, and ArviZ export.

It is a strong fit for repeated Bayesian regression, forecasting, and hierarchical workflows on CPU. It is not yet a full arbitrary-PPL replacement for PyMC or Stan.

What sets rustmc apart is the execution model: it shares one compiled Rust core across chains and across many independent models, so throughput stays high when the same structure is applied to thousands of datasets.

PyMC and Stan are excellent general-purpose tools, but they are optimized around a broader single-model workflow. rustmc is optimized for the repeated-model setting where Python orchestration, per-model overhead, and deployment friction start to dominate.

## Benchmark

**These numbers replace an earlier, unverified set of claims** ("10,000
Bayesian demand models in 70 seconds", "rustmc (NUTS) 72s / 5.3x speedup
vs. PyMC (NUTS) 383s") that this repo could not reproduce. The earlier
comparisons ran PyMC with an unspecified/default configuration, no fixed
seed on the PyMC side, and — in the batch case — 4x less sampling work on
the rustmc side (1 chain vs. PyMC's 4). See `examples/run_benchmarks.py`
and `benchmarks/RESULTS_TEMPLATE.md` for the corrected, reproducible
protocol: identical chains/warmup/draws/seed on every engine, phase-split
timing, and R-hat/ESS/divergences reported alongside wall time.

Every number below is from an actual run in this repo's environment (AMD
Ryzen 9 5900X, 24 logical CPUs, Linux, rustmc 0.8.0, PyMC 5.28.0, nutpie
0.16.6) — run `python examples/run_benchmarks.py` to reproduce on your own
hardware. Results will vary with core count and problem size; **do not
extrapolate these to other model sizes** — see the 500-parameter result
below, where rustmc loses.

**Simple regression** (`compare_with_pymc.py`) — 1 parameter, 10,000 obs, 4 chains, 500 warmup + 1000 draws, seed=42 on every engine:

| Engine | Time (s) | ESS_bulk | ESS/s | Max R-hat | Divergences |
|---|---|---|---|---|---|
| rustmc | 2.05 | 3998 | 1954 | 1.000 | 22 |
| PyMC (default NUTS) | 13.68 | 1759 | 129 | 1.001 | 0 |
| PyMC (nutpie) | 6.94 | 1726 | 249 | 1.002 | 0 |

rustmc wins on ESS/s here, but note it also produced 22 divergences where
both PyMC backends produced 0 on the identical model/data/seed — see
"Known limitations" below.

**High-dimensional regression** (`benchmark_vs_pymc.py`) — 500 parameters, 2,000 obs, 2 chains, 500 warmup + 500 draws, seed=42 on every engine, sigma fixed (not estimated) on every engine:

| Engine | Time (s) | ESS_bulk | ESS/s | Max R-hat | Divergences |
|---|---|---|---|---|---|
| rustmc | 39.2 | 996 | 25.4 | 1.020 | 7 |
| PyMC (default NUTS) | 18.7 | 1531 | 82.0 | 1.014 | 0 |
| PyMC (nutpie) | 9.2 | 1476 | 161.0 | 1.011 | 0 |

**rustmc is 3-6x slower than PyMC here, not faster.** This is the honest
result for a 500-parameter `MatVecMul`-backed model once chains, warmup,
draws, and seed are matched — the faer GEMV path does not currently
outperform PyMC/nutpie's autodiff at this parameter count on this
hardware. Do not read the small-model win above as evidence this holds at
scale.

**Many-chain regression** (`benchmark_multivariate.py`) — 10 parameters, 100,000 observations, 8 chains, 1000 warmup + 2000 draws, seed=42 on every engine, sigma fixed (not estimated) on every engine (an earlier version of this script ran PyMC with its unspecified default sampler and no explicit seed; both are fixed below and both PyMC backends are reported):

| Engine | Time (s) | ESS_bulk | ESS/s | Max R-hat | Divergences |
|---|---|---|---|---|---|
| rustmc | 52.2 | 15998 | 307 | 1.000 | 67 |
| PyMC (default NUTS) | 358.4 | 21098 | 58.9 | 1.001 | 0 |
| PyMC (nutpie) | 78.1 | 21674 | 278 | 1.001 | 0 |

Here rustmc is genuinely 5.2x faster than PyMC's default NUTS and roughly at parity
(1.1x) with PyMC+nutpie — a real win on this many-chain, low-parameter-count workload.
As above, rustmc's divergence count (67) is nonzero where both PyMC backends show 0 on
the identical model/data/seed.

**Batch inference** (`batch_10k_skus.py`) — 100 independent 3-parameter SKU models, 4 chains, 500 warmup + 1000 draws, seed=42 on every engine (an earlier version of this script mislabeled itself "10,000 SKUs" while actually running 100, and gave rustmc 1 chain vs. PyMC+nutpie's 4 — both are fixed below):

| Engine | Time (s) | Max R-hat | Mean ESS | ESS/s (summed) | Divergences | Forecast MAE (SKU #42) |
|---|---|---|---|---|---|---|
| rustmc (batch NUTS) | 1.24 | 1.066 | 241 | 125,885 | 1317 | 5.52 |
| PyMC + nutpie (100 sequential compiles) | 388.9 | 1.015 | 881 | 679 | 0 | 5.56 |

rustmc is genuinely ~300x faster in wall time here because it amortizes one shared
Rayon thread pool across all 400 chains with zero per-model compilation, while nutpie
pays a full model compile for each of the 100 SKUs in sequence — this compile cost is
the actual mechanism behind the speedup, not something mysterious, and it shrinks
proportionally as the model count grows. But the max R-hat (1.066) is above the
conventional 1.01 convergence threshold and rustmc logged 1317 divergences where nutpie
logged 0 on the identical per-SKU model/data/seed — the speed advantage here comes with
a real convergence-quality cost that a wall-time-only comparison would hide. There is no
benchmark run in this repo (or elsewhere) supporting a 10,000-SKU wall-time claim; do not
restate one.

See `benchmarks/results/` for the full logs and environment/statistical-quality detail
behind every table above, and `benchmarks/RESULTS_TEMPLATE.md` for the format used to
record future runs.

## Quick start

```bash
pip install maturin
git clone https://github.com/tbosier/rustmc.git
cd rustmc
python -m venv .venv && source .venv/bin/activate
pip install numpy maturin
maturin develop --manifest-path python_bindings/Cargo.toml --release
```

or if you prefer, install the published wheel from PyPI:

```bash
pip install rustmc
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

### Batch inference (many independent models)

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

# Each result is a BatchResult with .mean(), .std(), .get_samples()
for r in results[:5]:
    print(r)
```

### Vector parameter model (high-dimensional regression)

For models where the parameter count is large — e.g. a regression with thousands of features — use `normal_prior` with `@` to dispatch `X @ beta` via faer. rustmc automatically detects that `beta` is used in a matrix multiply, infers the number of parameters from the matrix dimensions, and promotes it to a contiguous vector parameter block:

```python
import numpy as np
import rustmc as rmc

N, P = 10_000, 500
X = np.random.randn(N, P)           # 2-D array → stored as faer matrix
beta_true = np.random.randn(P)
y = X @ beta_true + np.random.randn(N)

builder = rmc.ModelBuilder()
beta = builder.normal_prior("beta", mu=0.0, sigma=1.0)
mu_expr = beta @ "X"                # auto-promoted to faer GEMV
builder.normal_likelihood("obs", mu_expr=mu_expr, sigma=1.0, observed_key="y")
model = builder.build()

fit = rmc.sample(model_spec=model, data={"X": X, "y": y}, chains=4, draws=500)
print(fit.summary())
```

Instead of 500 separate scalar graph nodes (one per coefficient), rustmc allocates a single `MatVecMul` op backed by faer. The entire `X @ beta` forward pass and its gradient are computed with a single BLAS-level call, giving cache-efficient performance regardless of how many parameters are in the vector.

For explicit control over the vector size, `vector_normal_prior("beta", n=P)` is also available.

The builder supports scalar hierarchical priors today. For `normal_prior`, both `mu` and `sigma` can be other parameters; for `half_normal_prior`, `sigma` can be a parameter; `exponential_prior` and `log_normal_prior` also accept parameter-valued hyperparameters; and likelihood `sigma` or `alpha` can be parameter-valued as well. Scalar hierarchical normals are automatically compiled through a non-centered path where appropriate. Vector-valued hierarchical priors are not yet supported.

## What is implemented

### Sampling

- NUTS (No-U-Turn Sampler) with multinomial candidate selection, generalized U-turn criterion, and divergence detection. Follows Hoffman and Gelman (2014) and Betancourt (2017).
- HMC with fixed leapfrog steps, available as a fallback via `sampler="hmc"`.
- Block-structured mass matrix adaptation with 3-phase warmup (step-size only, mass matrix estimation, final step-size tuning).
- Auto step-size initialization via binary search.
- Deterministic per-chain RNG (ChaCha8) for reproducible results.
- Multithreaded chains via Rayon. Batch inference shares the thread pool across all models.

### Distributions

| Distribution | Support | Transform | Status |
|-------------|---------|-----------|--------|
| Normal | (-inf, inf) | None | Working |
| StudentT | (-inf, inf) | None | Working |
| HalfNormal | (0, inf) | log | Working |
| Exponential | (0, inf) | log | Working |
| LogNormal | (0, inf) | log | Working |
| Gamma | (0, inf) | log | Working |
| Beta | (0, 1) | logit | Working |
| Uniform | (a, b) | logit | Working |
| Bernoulli | {0, 1} | None | Discrete, limited |
| Poisson | {0, 1, 2, ...} | None | Discrete, limited |

Constrained distributions are automatically sampled in unconstrained space via log/logit transforms with Jacobian corrections. Samples are back-transformed before being returned to the user.

Discrete priors are exposed for completeness, but they are not differentiable and are not suitable for gradient-based sampling in their current form. In practice, use the continuous relaxations or a model structure that keeps the latent parameters continuous.

### Likelihood families

- `normal_likelihood(name, mu_expr, sigma, observed_key)`
- `bernoulli_logit_likelihood(name, eta_expr, observed_key)`
- `poisson_log_likelihood(name, eta_expr, observed_key)`
- `exponential_likelihood(name, eta_expr, observed_key)`
- `log_normal_likelihood(name, mu_expr, sigma, observed_key)`
- `negative_binomial_likelihood(name, eta_expr, alpha, observed_key)`

All GLM-style families use the same expression surface: bare parameters, `beta * "x"`, additive expressions, matrix multiplies via `beta @ "X"`, and additive constants.

### Computation

- Computational graph with reverse-mode automatic differentiation.
- Fused linear combination op for regression models. Replaces N separate multiply-add passes with a single cache-friendly loop over the data.
- Zero-allocation evaluator. All vector intermediates are pre-allocated in a flat buffer and reused across gradient evaluations. No heap allocation in the sampling loop.
- faer-backed matrix-vector multiply (`MatVecMul`). When a `normal_prior` parameter is used with `@` (e.g. `beta @ "X"`), rustmc automatically promotes it to a contiguous vector parameter block and dispatches the multiply to faer's GEMV routine. This replaces thousands of individual scalar multiply-add graph ops with a single BLAS-level call. Rayon threads are used for matrices above 100K elements. Explicit `vector_normal_prior` is also available for manual control.
- Vectorized Normal prior (`VectorNormalLogP`). A single graph op evaluates the log-probability of an entire parameter vector under `Normal(mu, sigma)`, replacing one graph node per parameter with a single tight loop. Gradients for all vector parameters accumulate directly into the gradient buffer in one backward pass.
- 2-D NumPy arrays in the data dict are automatically detected and stored as row-major matrices for use with `MatVecMul`.

### Diagnostics

- Split R-hat with rank normalization (Vehtari et al. 2021).
- Bulk and tail effective sample size (ESS).
- Monte Carlo standard error (MCSE).
- 94% highest density interval.
- Per-chain acceptance rates, step sizes, and divergence counts.
- Automatic warnings for convergence issues.
- Recovery suite covering canonical synthetic models in CI.

Available via `fit.summary()` for a formatted table or `fit.diagnostics()` for programmatic access.

### Predictive checks

- `sample_prior_predictive()` returns prior draws plus simulated observations.
- `FitResult.posterior_predictive()` returns simulated observations from posterior draws.
- `FitResult.log_likelihood()` returns pointwise log-likelihood arrays with shape `(chain, draw, obs)`.
- `FitResult.to_arviz()` exports posterior, sample stats, posterior predictive, and pointwise log-likelihood for downstream ArviZ/LOO/WAIC workflows.

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
  +-- Graph         Computational DAG, nodes, ops, data + matrix storage
  +-- Autodiff      Forward evaluation + reverse-mode gradient
  +-- Distributions  Scalar priors, GLM likelihood families, automatic transforms
  +-- NUTS          Multinomial tree-building, U-turn detection
  +-- HMC           Fixed-step leapfrog (fallback)
  +-- Sampler       Multi-chain parallel runner, batch inference
  +-- Diagnostics   R-hat, ESS, MCSE, HDI
  +-- Progress      Atomic counters, background render thread
  +-- faer          BLAS-level MatVecMul for high-dimensional parameter vectors
```

Design principles:

- Model graph is built once and shared read-only across chains.
- Sampler accepts any log-probability + gradient function derived from a Graph.
- No global state. All state is explicit and owned.
- Deterministic RNG per chain (ChaCha8 seeded from base_seed + chain_index).
- Parameter transforms and Jacobian corrections are handled in the graph, not the sampler.

### Data structures (Rust vs JAX)

The hot path uses plain Rust types only: the graph is `Vec<Node>` and `Vec<Op>`, parameters and gradients are `Vec<f64>`, and the autodiff evaluator uses contiguous `vec_buf` / `adj_vec_buf` (flat `Vec<f64>`) for all vector intermediates. For high-dimensional parameter vectors, data matrices are stored row-major as `Vec<f64>` inside the graph and handed to faer's `matmul` kernel as zero-copy views. `ndarray` appears only in the Python bindings for converting incoming 2-D NumPy arrays; it is not present in the inner loop. Benefits of this layout:

- **Cache-friendly**: One pass over the graph touches sequential memory; vector slots are in a single allocation.
- **Zero allocation in the loop**: Buffers are allocated once per chain and reused for every gradient evaluation.
- **No Python or FFI in the inner loop**: The entire NUTS/HMC step runs in Rust; Python is only used to build the model and consume results.
- **Fixed graph traversal**: The same DAG is walked every time; there is no tracing or recompilation per model or per step.
- **BLAS-level throughput for large parameter vectors**: `MatVecMul` calls faer's GEMV, which uses SIMD intrinsics and can optionally spawn Rayon threads for matrices above 100K elements. A 5,000-parameter vector prior that previously required 5,000 individual scalar multiply-add nodes in the graph is now a single op.

JAX, by contrast, traces Python and compiles to XLA. That gives flexibility and GPU support but adds per-model compilation and dispatch overhead. For many small, independent models, rustmc's "compile once, run fixed graph over contiguous buffers" approach can win on CPU because there is no per-model JAX trace/compile and no Python in the inner loop. See `examples/batch_10k_skus.py` and the Benchmark section above for the actual measured comparison against PyMC+nutpie run in a loop over the same number of models with matched chains/warmup/draws/seed — this is a real but narrower and smaller-scale result than "10,000 SKUs" might suggest; see the caveats in that section.

## Known limitations

These are observations from the benchmark runs above, not hand-wavy caveats:

- **rustmc produced nonzero divergence counts in every benchmark in this suite** (22 on
  a 1-parameter model, 7 on a 500-parameter model, 67 on a 10-parameter/100k-obs model,
  1317 across 100 SKUs) where PyMC's default NUTS and nutpie backends produced **zero**
  divergences on the identical model, data, and seed every time. On the 100-SKU batch
  run, rustmc's max R-hat (1.066) also exceeded the conventional 1.01 threshold where
  nutpie's did not (1.015). This is consistent enough across unrelated model shapes that
  it looks like a real property of the current step-size/mass-matrix adaptation or
  divergence-detection logic, not benchmark noise — worth investigating before trusting
  rustmc's posteriors on a model where PyMC shows zero divergences and rustmc doesn't.
- **rustmc is 3-6x slower than PyMC by ESS/s on a 500-parameter regression** (see
  "High-dimensional regression" above). The faer-backed `MatVecMul` path does not
  currently outperform PyMC/nutpie's autodiff at this parameter count on this hardware.
  It wins clearly on low-parameter/many-chain and many-small-model workloads instead —
  don't assume the "faer is faster for high-dimensional regression" framing elsewhere in
  this README holds at 500+ parameters without re-benchmarking.
- Benchmark numbers in this README are from one machine (see each table) and one run
  each, not repeated trials with reported variance. Treat them as directional, and
  reproduce with `python examples/run_benchmarks.py` before relying on them for a
  procurement or architecture decision.

## Roadmap

Near term:

- Expose compiled model artifacts as a first-class public workflow in Python and Rust.
- Extend automatic non-centering beyond scalar hierarchical normals to grouped/vector random effects.
- Investigate the divergence-count and R-hat gap vs. PyMC/nutpie noted in "Known limitations" above.
- Expand the modeling layer with production helpers such as offsets, exposure terms, and panel/hierarchical templates.

Medium term:

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
