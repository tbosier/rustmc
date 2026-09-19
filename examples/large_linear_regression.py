"""High-dimensional regression

A regression with 500 coefficients, fitted through the faer-backed `MatVecMul`
path. The 500 coefficients become one contiguous vector parameter and one graph
node, so a gradient evaluation costs two faer GEMV calls -- `X @ beta` forward and
`X.T @ adjoint` backward -- instead of 500 scalar loops in each direction.

Runtime note: every leapfrog step streams the whole design matrix twice, so cost
grows with `N_OBS * N_PARAMS`. At 4,000 x 500 the matrix is 16 MB per GEMV, and one
chain of 200 warmup + 200 draws finished in under 10 seconds when this was written,
on a 24-core machine. That is a rough expectation, not retained benchmark evidence.
`benchmarks/` is where a measurement with provenance belongs. Raise the dimensions
to reach a larger regime: 6,000 x 5,000 moves 240 MB per GEMV, 15 times this
script's traffic, and takes far longer.
"""

# %%
import time

import numpy as np
import rustmc as rmc

# %% Generate a 4,000 x 500 design
N_OBS = 4_000
N_PARAMS = 500

np.random.seed(42)
true_beta = np.random.randn(N_PARAMS) * 0.1
X = np.random.randn(N_OBS, N_PARAMS)  # row-major C order
y = X @ true_beta + np.random.randn(N_OBS) * 1.0

print(f"Dataset: {N_OBS:,} obs x {N_PARAMS:,} params")

# %% [markdown]
# ## What `beta @ "X"` does
#
# Writing `beta @ "X"` instead of a sum of scalar products makes rustmc:
#
# 1. detect that `beta` is used in a matrix multiply;
# 2. infer the number of coefficients from the matrix's column count;
# 3. promote `beta` to a contiguous vector parameter block;
# 4. replace N scalar multiply-add nodes with a single `MatVecMul` op;
# 5. compute the forward pass and the gradient through
#    [faer](https://github.com/sarah-ek/faer-rs)'s GEMV, which uses Rayon for
#    matrices above roughly 100,000 elements.
#
# A 2D NumPy array in the data dict is detected and stored as a row-major matrix;
# passing `"X": X` where `X.ndim == 2` is all that is needed. `normal_prior`
# combined with `@` auto-promotes `beta`, so `vector_normal_prior("beta", n=P)` is
# only needed when you want to set the coefficient count yourself rather than infer
# it from the matrix.
#
# The `@` form pays off when `P` is large -- above roughly 50 coefficients. For a
# small regression, scalar `beta * "x"` is fine; the crossover is where walking the
# individual graph nodes costs more than one GEMV dispatch.

# %% Build the model
t0 = time.time()
builder = rmc.ModelBuilder(data={"X": X, "y": y})
intercept = builder.normal_prior("intercept", mu=0.0, sigma=10.0)
beta = builder.normal_prior("beta", mu=0.0, sigma=1.0)
mu_expr = intercept + beta @ "X"
builder.normal_likelihood("obs", mu_expr=mu_expr, sigma=1.0, observed_key="y")
model = builder.build()
build_time = time.time() - t0
print(f"Model built in {build_time:.3f}s")

# %% Sample
DRAWS, WARMUP = 200, 200
print(f"Sampling: NUTS, 1 chain, {WARMUP} warmup + {DRAWS} draws ...")
t0 = time.time()
result = rmc.sample(model, draws=DRAWS, warmup=WARMUP, chains=1, seed=42, show_progress=True)
elapsed = time.time() - t0

print(f"Elapsed : {elapsed:.2f}s")
print(f"Iters/s : {(DRAWS + WARMUP) / elapsed:.1f}")
print(f"Accept  : {result.accept_rates()[0]:.3f}")
print(f"Diverge : {sum(result.divergences())}")

# %% Recover the coefficients
# A vector parameter is reported one entry per coordinate -- beta[0], beta[1], ... --
# not as a nested array.
samples = result.get_samples()
beta_means = np.array([samples[f"beta[{k}]"].mean() for k in range(N_PARAMS)])
rmse = np.sqrt(np.mean((beta_means - true_beta) ** 2))
print(f"beta recovery RMSE : {rmse:.4f}  (generating coefficient sd {true_beta.std():.4f})")
for k in range(5):
    print(f"  beta[{k}]: true={true_beta[k]:+.4f}  estimated={beta_means[k]:+.4f}")

# %% [markdown]
# ## Limitations of this run
#
# One chain of 200 draws is enough to show the API and to recover coefficients whose
# generating scale is known. It is not enough for convergence diagnostics: R-hat
# needs several chains, and 200 draws leaves a small effective sample size per
# coordinate. Raise `chains` and `draws` before reading anything into an individual
# coefficient.
#
# The elapsed time and rate printed above are replaced with placeholders on this
# page, because they depend on the machine and would otherwise change on every
# regeneration. Run the script to see real numbers for your hardware, and use
# `benchmarks/` for a measurement with provenance.
