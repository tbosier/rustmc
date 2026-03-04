"""
rustmc — large-parameter linear regression benchmark
=====================================================
Demonstrates the faer-backed MatVecMul path: 1001 graph nodes instead of
~15,000, and a single SIMD-vectorized GEMV instead of 5000 scalar axpy loops.

normal_prior + @ auto-promotes beta to a contiguous vector parameter block
backed by faer GEMV — no need to call vector_normal_prior explicitly.

Runtime note: NUTS gradient evaluation streams the full design matrix from RAM
on every leapfrog step. For N_OBS=6000, N_PARAMS=5000 that is ~240 MB per
GEMV. Expect ~30–90 min for 400 samples depending on core count and NUTS tree
depth. Reduce N_OBS/N_PARAMS for a quicker test.
"""
import time
import numpy as np
import rustmc as rmc

N_OBS    = 6_000
N_PARAMS = 5_000

np.random.seed(42)
true_beta = np.random.randn(N_PARAMS) * 0.1
X = np.random.randn(N_OBS, N_PARAMS)          # row-major C order
y = X @ true_beta + np.random.randn(N_OBS) * 1.0

print(f"Dataset: {N_OBS:,} obs × {N_PARAMS:,} params")

# ── Auto-promoted vector-param model (faer MatVecMul path) ───────────────
t0 = time.time()
builder = rmc.ModelBuilder()
intercept = builder.normal_prior("intercept", mu=0.0, sigma=10.0)
beta      = builder.normal_prior("beta", mu=0.0, sigma=1.0)
mu_expr   = intercept + beta @ "X"
builder.normal_likelihood("obs", mu_expr=mu_expr, sigma=1.0, observed_key="y")
model = builder.build()
data  = {"X": X, "y": y}
build_time = time.time() - t0
print(f"Model built in {build_time:.3f}s")

DRAWS, WARMUP = 200, 200
print(f"\nSampling: NUTS, 1 chain, {WARMUP} warmup + {DRAWS} draws ...")
t0 = time.time()
result = rmc.sample(model, data, draws=DRAWS, warmup=WARMUP, chains=1, seed=42,
                    show_progress=True)
elapsed = time.time() - t0

print(f"\nElapsed : {elapsed:.2f}s")
print(f"Iters/s : {(DRAWS + WARMUP) / elapsed:.1f}")
print(f"Accept  : {result.accept_rates()[0]:.3f}")
print(f"Diverge : {sum(result.divergences())}")

samples = result.get_samples()
beta_means = np.array([samples[f"beta[{k}]"].mean() for k in range(N_PARAMS)])
rmse = np.sqrt(np.mean((beta_means - true_beta) ** 2))
print(f"\nbeta recovery RMSE : {rmse:.4f}  (expect small with enough draws)")
