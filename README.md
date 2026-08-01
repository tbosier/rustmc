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

What sets rustmc apart today is a Rust-native sampling loop, a shared Rayon pool for chains and batches, and an in-memory compile-once/bind-many path. Use `ModelBuilder.compile()` for graph reuse; the legacy `batch_sample()` entry point still builds concrete data-owning graphs.

PyMC and Stan are excellent general-purpose tools, but they are optimized around a broader single-model workflow. rustmc is optimized for the repeated-model setting where Python orchestration, per-model overhead, and deployment friction start to dominate.

## Benchmark

The repository includes benchmark drivers for small, high-dimensional, many-chain, and
batch regression workloads. Run `python examples/run_benchmarks.py` to collect results
on your hardware. The drivers match chains, warmup, draws, and seeds across engines and
report statistical-quality metrics alongside wall time.

No numeric result is published here because the repository does not retain the raw
command output needed to audit one. Before adding a claim, retain the environment, exact
revision, commands, and unedited raw output; use `benchmarks/RESULTS_TEMPLATE.md` as the
checklist. Do not extrapolate batch throughput or a single workload to other model sizes.

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

- NUTS (No-U-Turn Sampler) with multinomial candidate selection, endpoint-momentum U-turn checks, and divergence detection. Follows Hoffman and Gelman (2014) and Betancourt (2017).
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
- Reusable evaluator buffers. Vector intermediates are pre-allocated in a flat buffer
  and reused across gradient evaluations. This is scoped to the evaluator; other sampler
  operations may allocate.
- faer-backed matrix-vector multiply (`MatVecMul`). When a `normal_prior` parameter is used with `@` (e.g. `beta @ "X"`), rustmc automatically promotes it to a contiguous vector parameter block and dispatches the multiply to faer's GEMV routine. This replaces thousands of individual scalar multiply-add graph ops with a single BLAS-level call. Rayon threads are used for matrices above 100K elements. Explicit `vector_normal_prior` is also available for manual control.
- Vectorized Normal prior (`VectorNormalLogP`). A single graph op evaluates the log-probability of an entire parameter vector under `Normal(mu, sigma)`, replacing one graph node per parameter with a single tight loop. Gradients for all vector parameters accumulate directly into the gradient buffer in one backward pass.
- 2-D NumPy arrays in the data dict are automatically detected and stored as row-major matrices for use with `MatVecMul`.

### Diagnostics

- Raw split R-hat, plus rank-normalized bulk and tail ESS.
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

### Compile once, bind many

`ModelBuilder.compile()` returns an immutable `CompiledModel`. Its validated `bind()`,
`sample()`, and `sample_batch()` paths reuse one shared graph structure across datasets
with different row counts and preserve caller-supplied dataset IDs. The legacy
`sample()` and `batch_sample()` entry points remain available.

### Linear Gaussian state space

`LinearGaussianStateSpace` provides a time-homogeneous Kalman filter,
Rauch--Tung--Striebel smoother, and forecast API for arbitrary latent-state dimension
and scalar observations. `NaN` observations are handled as prediction-only steps;
convenience constructors cover local-level, local-linear-trend, and stationary AR(1)
models. Forecast results expose explicitly labeled pointwise conditional intervals,
including 95% bounds.

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
  +-- Graph         Computational DAG, nodes, ops, and structural data schema
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

The hot path uses plain Rust types only: the graph is `Vec<Node>` and `Vec<Op>`, parameters and gradients are `Vec<f64>`, and the autodiff evaluator uses contiguous `vec_buf` / `adj_vec_buf` (flat `Vec<f64>`) for all vector intermediates. For high-dimensional parameter vectors, data matrices are stored row-major in the active data binding and handed to faer's `matmul` kernel as zero-copy views. Legacy graphs may still own these buffers. `ndarray` appears only in the Python bindings for converting incoming 2-D NumPy arrays; it is not present in the inner loop. Benefits of this layout:

- **Cache-friendly**: One pass over the graph touches sequential memory; vector slots are in a single allocation.
- **Evaluator buffer reuse**: Value and adjoint buffers are allocated once per evaluator
  and reused for gradient evaluations; other sampler operations may allocate.
- **No Python or FFI in the inner loop**: The entire NUTS/HMC step runs in Rust; Python is only used to build the model and consume results.
- **Fixed graph traversal**: The same DAG is walked every time; there is no tracing or recompilation per model or per step.
- **BLAS-level throughput for large parameter vectors**: `MatVecMul` calls faer's GEMV, which uses SIMD intrinsics and can optionally spawn Rayon threads for matrices above 100K elements. A 5,000-parameter vector prior that previously required 5,000 individual scalar multiply-add nodes in the graph is now a single op.

JAX, by contrast, traces Python and compiles to XLA. rustmc walks a fixed Rust graph during sampling, so there is no Python or FFI in the inner loop. The `CompiledModel` API can reuse one immutable structure across validated dataset bindings; the legacy `sample()` and `batch_sample()` shims retain their data-owning behavior. `examples/batch_10k_skus.py` provides a benchmark driver; do not generalize an unrecorded run into a throughput result.

## Known limitations

- **Compiled artifacts are not yet portable.** In-memory compile/bind reuse is
  implemented, but the legacy JSON artifact still owns data and there is no slot-only
  serialized format yet.
- **State-space parameters are fixed inputs.** Filtering, smoothing, missing-value
  updates, and forecasting are implemented for time-homogeneous linear Gaussian models,
  but they are not yet a likelihood inside `ModelBuilder` and do not estimate system
  matrices.
- **Builder context-manager syntax is only lifecycle sugar.** `ModelBuilder` supports
  `with`, returning itself and never suppressing exceptions. The architectural "context"
  in compile/bind design means a model/data binding context; `with ModelBuilder()` does
  not provide compiled-artifact reuse.

- No audited numeric performance baseline is checked in. Run and retain the benchmark
  protocol before making performance, convergence, or cross-engine quality claims.

## Roadmap

Near term:

- Finish the compiled-model path with portable slot-only artifacts, partial-failure
  batches, and prediction on new bindings.
- Extend automatic non-centering beyond scalar hierarchical normals to grouped/vector random effects.
- Rerun the comparison suite after diagnostic fixes and investigate any remaining
  post-warmup divergence or R-hat gap vs. PyMC/nutpie.
- Expand the modeling layer with production helpers such as offsets, exposure terms, and panel/hierarchical templates.

Medium term:

- Integrate the existing collapsed linear-Gaussian filter as a `ModelBuilder` likelihood,
  add an explicit stationary AR(1) constructor, and estimate system parameters.
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
