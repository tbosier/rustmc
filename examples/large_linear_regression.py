"""
rustmc — vector-parameter linear regression
===========================================
Demonstrates the faer-backed MatVecMul path: the 500 coefficients become one
contiguous vector parameter and one graph node. Each gradient evaluation then costs
two faer GEMV calls — `X @ beta` on the forward pass and `X.T @ adjoint` on the
backward pass — instead of 500 scalar axpy loops in each direction.

`normal_prior` combined with `@` auto-promotes `beta` to that vector parameter
block, so `vector_normal_prior` need not be called explicitly.

Runtime note: every leapfrog step streams the whole design matrix twice, so cost grows
with `N_OBS * N_PARAMS`. At 4,000 x 500 the matrix is 16 MB per GEMV, and 1 chain of
200 warmup + 200 draws finished in under 10 seconds when this was written, on a
24-core machine. That is a rough expectation, not retained benchmark evidence; the
script prints its own elapsed time, and `benchmarks/` is where a measurement with
provenance belongs. Raise N_OBS/N_PARAMS to reach a larger regime: 6,000 x 5,000
moves 240 MB per GEMV, 15 times this script's traffic, and takes far longer.
"""
import time
import numpy as np
import rustmc as rmc

N_OBS    = 4_000
N_PARAMS = 500

np.random.seed(42)
true_beta = np.random.randn(N_PARAMS) * 0.1
X = np.random.randn(N_OBS, N_PARAMS)          # row-major C order
y = X @ true_beta + np.random.randn(N_OBS) * 1.0

print(f"Dataset: {N_OBS:,} obs × {N_PARAMS:,} params")

# ── Auto-promoted vector-param model (faer MatVecMul path) ───────────────
t0 = time.time()
builder   = rmc.ModelBuilder(data={"X": X, "y": y})
intercept = builder.normal_prior("intercept", mu=0.0, sigma=10.0)
beta      = builder.normal_prior("beta", mu=0.0, sigma=1.0)
mu_expr   = intercept + beta @ "X"
builder.normal_likelihood("obs", mu_expr=mu_expr, sigma=1.0, observed_key="y")
model     = builder.build()
build_time = time.time() - t0
print(f"Model built in {build_time:.3f}s")

DRAWS, WARMUP = 200, 200
print(f"\nSampling: NUTS, 1 chain, {WARMUP} warmup + {DRAWS} draws ...")
t0 = time.time()
result = rmc.sample(model, draws=DRAWS, warmup=WARMUP, chains=1, seed=42,
                    show_progress=True)
elapsed = time.time() - t0

print(f"\nElapsed : {elapsed:.2f}s")
print(f"Iters/s : {(DRAWS + WARMUP) / elapsed:.1f}")
print(f"Accept  : {result.accept_rates()[0]:.3f}")
print(f"Diverge : {sum(result.divergences())}")

samples = result.get_samples()
beta_means = np.array([samples[f"beta[{k}]"].mean() for k in range(N_PARAMS)])
rmse = np.sqrt(np.mean((beta_means - true_beta) ** 2))
print(f"\nbeta recovery RMSE : {rmse:.4f}  "
      f"(generating coefficient sd {true_beta.std():.4f})")
