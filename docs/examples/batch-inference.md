# Batch Inference

Fit related independent models in one call. rustmc runs their chains through a shared
Rayon thread pool, although each current batch entry still owns a concrete graph and
dataset; compile-once/bind-many graph reuse is planned, not implemented.

## Use Case

You have many SKUs and want a demand model for each one. Repeated per-series
fits can become costly at that scale. Whether rustmc, ARIMA, or Prophet is
faster depends on the model, configuration, data, and hardware; their default
uncertainty outputs are also not directly comparable.

The example uses 100 models so it remains practical to run locally. It is an API example,
not a throughput claim. Scaling depends on model shape, draws, chains, and hardware; use
the matched-protocol benchmark driver and retain raw output before publishing a result.

## Code

```python
import rustmc as rmc
import numpy as np

np.random.seed(0)
N_MODELS = 100  # benchmark larger runs on your own model and hardware
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

Use `examples/batch_10k_skus.py` for a matched rustmc/PyMC+nutpie comparison that reports
divergences, R-hat, and ESS/s alongside wall time. Results are intentionally not checked
in without their raw output and environment provenance.

## Notes

- All models in a batch must use the same `draws` and `warmup` count.
- Models can have completely different structures — each gets its own graph and data.
- The thread pool is shared across chains and models. Additional cores may reduce wall
  time, but scaling depends on model size, batch size, memory bandwidth, and scheduling;
  measure it on the target workload rather than assuming proportional speedup.
- `chains=1` is the throughput-first setting. Increase `chains` when you want stronger convergence diagnostics per model.
- `sampler="hmc"` is available in batch mode as a fixed-step fallback.
