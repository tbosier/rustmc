# rustmc

**Bayesian inference powered by Rust, with a Python API.**

rustmc is a general Bayesian inference library with a deliberately scoped modeling
surface. It combines graph-based automatic differentiation and NUTS/HMC with exact,
conjugate, and state-space algorithms for models whose structure permits a more direct
approach.

## Why rustmc?

The project is exploring a structure-aware inference runtime rather than a smaller clone
of an existing probabilistic programming system. Generic models use the Rust autodiff and
sampling engine. Repeated models can reuse a compiled structure across validated data
bindings. Specialized algorithms remain available through the same package when they are
a better match than sampling every latent variable with NUTS.

## Differentiation

- Generic NUTS/HMC and reverse-mode autodiff execute in Rust outside the Python hot path.
- `ModelBuilder.compile()` separates immutable model structure from validated datasets.
- Independent chains and repeated-model workloads share one Rayon thread pool.
- Exact conjugate and state-space algorithms complement the generic sampler.
- Diagnostics, predictive checks, pointwise log likelihood, and ArviZ export support a
  complete Bayesian workflow for the current modeling surface.

## Benchmarks

The repository provides matched-protocol benchmark drivers; run `python
examples/run_benchmarks.py` on the hardware you care about. No numeric performance result
is published without retained raw command output, environment details, and an exact
revision. See the top-level README and `benchmarks/RESULTS_TEMPLATE.md` for the protocol.

## Install

```bash
pip install rustmc
```

## Quick Example

```python
import numpy as np
import rustmc as rmc

x = np.random.randn(1000)
y = 2.5 * x + np.random.randn(1000)

builder = rmc.ModelBuilder()
beta = builder.normal_prior("beta", mu=0.0, sigma=1.0)
builder.normal_likelihood("obs", mu_expr=beta * "x", sigma=1.0, observed_key="y")
model = builder.build()

fit = rmc.sample(model_spec=model, data={"x": x, "y": y}, chains=4, draws=1000)
print(fit.summary())
```

## What's Implemented

**Sampling:** NUTS with multinomial candidate selection, block-structured mass matrix
adaptation, dual-averaging step size, configurable `target_accept`, fixed-trajectory HMC
fallback, and multi-chain parallelism via Rayon.

**Priors:** Normal, HalfNormal, Exponential, LogNormal, StudentT, Gamma, Beta, Uniform, Bernoulli, Poisson. Constrained distributions are automatically sampled in unconstrained space.

**Likelihoods:** Normal, Bernoulli-logit, Poisson-log, Exponential-log, LogNormal, and NegativeBinomial-log.

**Modeling surface:** scalar hierarchical priors are supported, and scalar hierarchical normals are automatically compiled through a non-centered path where appropriate. Vector-valued hierarchical blocks are still on the roadmap.

**Predictive workflow:** prior predictive sampling, posterior predictive sampling, pointwise log-likelihood, and ArviZ export are implemented for the current likelihood families.

**Diagnostics:** Rank-normalized folded split R-hat, rank-normalized bulk/tail ESS,
MCSE, empirical 94% HDI, divergence detection, and per-chain acceptance rates.

**High-dimensional regression:** faer-backed `MatVecMul` op — `beta @ "X"` dispatches to a BLAS-level GEMV rather than N scalar graph nodes.

**Compile once, bind many:** `ModelBuilder.compile()` returns an immutable
`CompiledModel` whose validated bindings can have different row counts while sharing one
graph structure.

**Linear Gaussian state space:** `LinearGaussianStateSpace` provides fixed-system
Kalman filtering, smoothing, missing-observation handling, and forecasting.

**Forecasting application:** fitted local-level, seasonal local-level, and local-linear-trend structural models
provide FFBS/Gibbs posterior prediction, while `BayesianAutoRegression(order=p)` supports
directly observed Gaussian AR(p) at any positive lag order. All return coherent paths and
parameter-integrated pointwise intervals.

## What Is Still Missing

- A broader general modeling surface: named dimensions, group indexing, vector-valued
  hierarchical models, robust likelihoods, and stable extension points.
- Richer automatic reparameterization support.
- Portable serialization and prediction-on-new-data for the in-memory re-bindable
  compiled model; the legacy JSON artifact remains data-owning.
- Generic Bayesian state-space estimation and collapsed Kalman-likelihood integration in
  `ModelBuilder`; specialized local level, local trend, and direct AR(p) are implemented.
- Consistent result coordinates, typed Python APIs, and richer sampler telemetry.
- Higher-level applications for repeated inference, forecasting, and panel workflows.

See the
[project roadmap](https://github.com/tbosier/rustmc/blob/main/ROADMAP.md)
for the ordered plan. Forecasting is one application of the shared inference core; it is
not the boundary of the project.
