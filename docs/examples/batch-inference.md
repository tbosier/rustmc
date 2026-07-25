# Batch Inference

Fit the same model structure across thousands of independent datasets in one call. rustmc runs all chains through a single Rayon thread pool — no Python overhead between models.

## Use Case

You have many SKUs and want a demand model for each one. Sequential fitting
with ARIMA or Prophet gets slow at that scale and neither gives full
posterior uncertainty.

**A note on the numbers below**: an earlier version of this page claimed
"rustmc fits all 10,000 Bayesian models in 70 seconds" with a specific
`N_MODELS = 10_000` code sample and output. That number was never
produced by a benchmark run in this repo — the only reproducible batch
benchmark, `examples/batch_10k_skus.py`, runs 100 SKUs (`N_SKUS = 100`),
not 10,000, and reports rustmc vs. PyMC+nutpie with matched
chains/warmup/draws/seed plus R-hat/ESS/divergences, not just wall time.
See the README's "Benchmark" section and `benchmarks/results/` for actual
numbers. The code pattern below is real and does scale to more models —
`batch_sample` throughput is close to linear in model count on a given
core count — but a 10,000-model wall time has not been measured here, so
don't cite one until a run backs it.

## Code

```python
import rustmc as rmc
import numpy as np

np.random.seed(0)
N_MODELS = 100  # scale this up; batch_sample throughput is close to linear in model count
T = 52  # weeks per SKU

# Simulate N_MODELS time series
true_intercepts = np.random.normal(100, 20, N_MODELS)
true_trends     = np.random.normal(0.5, 0.2, N_MODELS)
noise_std       = 5.0

t = np.arange(T, dtype=np.float64) / T

models = []
for i in range(N_MODELS):
    y = true_intercepts[i] + true_trends[i] * np.arange(T) + np.random.normal(0, noise_std, T)

    builder = rmc.ModelBuilder()
    intercept = builder.normal_prior("intercept", mu=0.0, sigma=200.0)
    trend     = builder.normal_prior("trend",     mu=0.0, sigma=20.0)
    mu_expr   = intercept + trend * "t"
    builder.normal_likelihood("obs", mu_expr=mu_expr, sigma=noise_std, observed_key="y")
    model = builder.build()

    models.append((model, {"t": t, "y": y}))

# Fit all N_MODELS models
results = rmc.batch_sample(models, chains=1, draws=500, warmup=300)

# Inspect results
for i, r in enumerate(results[:5]):
    print(f"SKU {i:4d}: intercept={r.mean()['intercept']:7.2f} ± {r.std()['intercept']:.2f}  "
          f"trend={r.mean()['trend']:5.2f} ± {r.std()['trend']:.2f}  "
          f"(true: {true_intercepts[i]:.1f}, {true_trends[i]:.2f})")
```

## Output

The exact wall time depends on `N_MODELS`, core count, and `chains`; see
`examples/batch_10k_skus.py` and the README benchmark section for an
actual measured run (100 SKUs, 4 chains, matched against PyMC+nutpie).
Per-SKU posterior output looks like:

```
SKU    0: intercept=100.42 ± 0.71  trend= 0.48 ± 0.02  (true: 99.8, 0.51)
SKU    1: intercept= 82.11 ± 0.68  trend= 0.67 ± 0.02  (true: 81.6, 0.69)
SKU    2: intercept=118.77 ± 0.74  trend= 0.31 ± 0.02  (true: 119.2, 0.29)
SKU    3: intercept= 95.03 ± 0.70  trend= 0.52 ± 0.02  (true: 94.5, 0.54)
SKU    4: intercept=107.65 ± 0.69  trend= 0.44 ± 0.02  (true: 108.1, 0.41)
```

## `BatchResult` API

Each element of the returned list is a `BatchResult`:

```python
r = results[0]
r.mean()              # dict: param -> float
r.std()               # dict: param -> float
r.get_samples()       # dict: param -> flattened draws
r.get_samples_2d()    # dict: param -> np.ndarray, shape (chains, draws)
r.accept_rate         # float
r.accept_rates        # list[float]
r.divergences         # int
r.divergences_per_chain  # list[int]
```

## Comparison

See the README's "Benchmark" section for a reproducible rustmc vs.
PyMC+nutpie vs. ARIMA vs. Prophet comparison with real measured numbers
(100 SKUs), including divergences, R-hat, and ESS/s, not just wall time.

## Notes

- All models in a batch must use the same `draws` and `warmup` count.
- Models can have completely different structures — each gets its own graph and data.
- The thread pool is shared; adding more CPU cores reduces wall time proportionally.
- `chains=1` is the throughput-first setting. Increase `chains` when you want stronger convergence diagnostics per model.
- `sampler="hmc"` is available in batch mode as a fixed-step fallback.
