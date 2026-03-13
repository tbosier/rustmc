# High-Dimensional Regression

For models with many parameters — regressions with hundreds or thousands of features — rustmc uses a `MatVecMul` op backed by [faer](https://github.com/sarah-ek/faer-rs) to compute `X @ beta` at BLAS level rather than walking thousands of scalar graph nodes.

## How It Works

When you write `beta @ "X"`, rustmc:

1. Detects that `beta` is used in a matrix multiply
2. Infers the number of parameters from the matrix's column count
3. Promotes `beta` to a contiguous vector parameter block
4. Replaces N scalar multiply-add nodes with a single `MatVecMul` op
5. Computes the forward pass and gradient via faer's GEMV (uses Rayon for matrices > 100K elements)

## Code

```python
import numpy as np
import rustmc as rmc

N, P = 10_000, 500        # 10k observations, 500 features
np.random.seed(42)

X        = np.random.randn(N, P)
beta_true = np.random.randn(P)
y        = X @ beta_true + np.random.randn(N)

builder = rmc.ModelBuilder()
beta    = builder.normal_prior("beta", mu=0.0, sigma=1.0)
mu_expr = beta @ "X"      # auto-promoted to faer GEMV

builder.normal_likelihood("obs", mu_expr=mu_expr, sigma=1.0, observed_key="y")
model = builder.build()

fit = rmc.sample(
    model_spec=model,
    data={"X": X, "y": y},
    chains=4,
    draws=500,
    warmup=500,
    seed=42,
)
print(fit.summary())

# Compare a few estimates to the true values
means = fit.mean()
for i in range(5):
    print(f"beta[{i}]: true={beta_true[i]:.4f}  estimated={means['beta'][i]:.4f}")
```

## Notes

- **2D NumPy arrays** in the data dict are automatically detected and stored as row-major matrices. You don't need to do anything special — just pass `"X": X` where `X.ndim == 2`.
- **Explicit vector prior**: `vector_normal_prior("beta", n=P)` is available if you want to control the parameter count explicitly rather than inferring it from the matrix.
- **Gradient accumulation**: vector parameter gradients accumulate directly into the gradient buffer in a single backward pass — no per-node overhead.
- **Rayon parallelism**: matrices above 100,000 elements (e.g. N×P > 100k) automatically use multiple Rayon threads for the matrix multiply.

## When to Use This

The `@` syntax is most beneficial when `P` is large (say, > 50 features). For small regressions, scalar `beta * "x"` is fine. The crossover point is roughly when the overhead of walking 50+ individual graph nodes exceeds the cost of a faer GEMV dispatch.
