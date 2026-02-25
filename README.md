# rustmc

**High-performance probabilistic inference engine written in Rust.**

rustmc is a Rust-native MCMC engine designed for industrial-scale Bayesian inference. It exposes a Python API via PyO3 while keeping the entire sampling loop in Rust — no Python in the inner loop.

## Why rustmc?

Current Python-first frameworks (PyMC, Stan via PyStan, etc.) face performance ceilings:

| Limitation | PyMC / Python-first | rustmc |
|---|---|---|
| Sampling loop | Python + Aesara/PyTensor | Pure Rust |
| Chain parallelism | Process-based (multiprocessing) | Thread-based (Rayon) |
| Serialization overhead | Pickle between processes | Zero — shared memory |
| Large models (10k+ params) | Slow graph compilation | Compiled once, reused |
| Latency | High (JIT, Python overhead) | Low (native binary) |
| Embeddability | Python only | Rust library or Python extension |

rustmc is **not** trying to replace PyMC's research ergonomics. It exists for workloads where **performance, latency, and throughput** are the bottleneck.

## Architecture

```
┌─────────────────────────────────────────────┐
│  Python (orchestration only)                │
│  ┌───────────────────────────────────────┐  │
│  │  rustmc Python API (PyO3)             │  │
│  │  - ModelBuilder, sample(), FitResult  │  │
│  └──────────────┬────────────────────────┘  │
│                 │ GIL released              │
├─────────────────┼───────────────────────────┤
│  Rust Core      │                           │
│  ┌──────────────▼────────────────────────┐  │
│  │  Graph (computational DAG)            │  │
│  ├───────────────────────────────────────┤  │
│  │  Autodiff (reverse-mode)              │  │
│  ├───────────────────────────────────────┤  │
│  │  HMC Sampler (leapfrog integration)   │  │
│  ├───────────────────────────────────────┤  │
│  │  Parallel Chains (Rayon threads)      │  │
│  │  - Deterministic RNG per chain        │  │
│  │  - Read-only shared graph             │  │
│  └───────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

### Module breakdown

| Module | Purpose |
|---|---|
| `graph.rs` | Computational DAG — nodes, ops, data storage |
| `autodiff.rs` | Forward evaluation + reverse-mode gradient |
| `distributions.rs` | Trait-based distributions (Normal) |
| `hmc.rs` | HMC with leapfrog integrator + dual-averaging step-size adaptation |
| `sampler.rs` | Multi-chain parallel runner via Rayon |
| `python_bindings/` | PyO3 bridge — ModelBuilder, sample(), FitResult |

### Design principles

- **Separate model graph from sampler** — the graph is built once and shared read-only across chains.
- **Generic sampler interface** — the HMC sampler accepts any `(logp, grad)` function derived from a Graph.
- **Trait-based distributions** — new distributions plug in without modifying the sampler.
- **No global state** — all state is explicit and owned.
- **Minimal allocation in the hot loop** — parameter vectors are reused, gradient vectors are stack-allocated where possible.
- **Deterministic RNG** — each chain gets `ChaCha8Rng` seeded from `base_seed + chain_index`.

## Quick start

### Build

```bash
# Install maturin
pip install maturin

# Build and install in development mode
maturin develop --manifest-path python_bindings/Cargo.toml --release
```

### Usage

```python
import numpy as np
import rustmc as rmc

# Generate data
np.random.seed(42)
N = 1000
x = np.random.randn(N)
y = 2.5 * x + np.random.randn(N)

# Define model
builder = rmc.ModelBuilder()
beta = builder.normal_prior("beta", mu=0.0, sigma=1.0)
mu_expr = beta * "x"
builder.normal_likelihood("obs", mu_expr=mu_expr, sigma=1.0, observed_key="y")
model = builder.build()

# Sample (all MCMC runs in Rust, GIL released)
fit = rmc.sample(
    model_spec=model,
    data={"x": x, "y": y},
    chains=8,
    draws=1000,
    warmup=500,
)

print(fit)
print("Posterior mean:", fit.mean())
```

## Performance goals

- **Sampling throughput**: 10-100x faster than PyMC for equivalent models on multi-core machines.
- **Latency**: Sub-second for small models (< 100 params, 1000 draws).
- **Scalability**: Linear scaling with number of chains up to available cores.
- **Memory**: O(params × draws × chains) — no unnecessary copies.

## Roadmap

| Priority | Feature | Status |
|---|---|---|
| MVP | HMC sampler | Done |
| MVP | Reverse-mode autodiff | Done |
| MVP | Normal distribution | Done |
| MVP | Multithreaded chains (Rayon) | Done |
| MVP | PyO3 Python bindings | Done |
| Next | NUTS (No-U-Turn Sampler) | Planned |
| Next | Step-size + mass matrix adaptation | Planned |
| Next | More distributions (HalfNormal, Uniform, StudentT, Bernoulli) | Planned |
| Future | Online / streaming posterior updates | Planned |
| Future | Particle filtering | Planned |
| Future | Large hierarchical model optimizations | Planned |
| Future | GPU integration via wgpu | Planned |
| Future | Distributed posterior aggregation | Planned |

## Key differentiators vs PyMC

1. **Fully Rust sampling loop** — zero Python overhead during MCMC.
2. **True multithreaded chains** — Rayon thread pool, not multiprocessing.
3. **No Python inside inner sampling loop** — GIL is released for the entire sampling run.
4. **Designed for simulation throughput** — minimal allocation, cache-friendly data layout.
5. **Embeddable in Rust systems** — use `rustmc_core` as a standalone Rust library.
6. **Deterministic, low-latency inference** — reproducible results with ChaCha8 RNG per chain.

## License

MIT
