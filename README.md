# rustmc

Bayesian inference powered by Rust, with a Python API.

> **Project status: alpha.** rustmc is suitable for research, evaluation, and
> controlled internal workflows. Its supported modeling surface is useful but still
> intentionally smaller than mature probabilistic programming systems. Validate every
> model on representative data before using its output for consequential decisions.

rustmc is a general Bayesian inference library built around a simple idea: use a generic
sampler when a model needs one, but exploit model structure when a more direct algorithm
is available. The same package therefore contains graph-based automatic differentiation
and NUTS/HMC, reusable compiled models, exact conjugate inference, and specialized
state-space algorithms.

It is not intended to be a smaller clone of PyMC or Stan. The long-term opportunity is a
compact, auditable inference runtime that can recognize useful structure, fit one model
or many related datasets, and return posterior and predictive draws with their provenance
intact.

## Why rustmc

- **General and specialized inference in one runtime.** The model builder uses
  reverse-mode automatic differentiation with NUTS or HMC. Local-level, seasonal, and
  trend models use FFBS/Gibbs, while Gaussian AR(p) uses an exact
  Normal-Inverse-Gamma posterior.
- **Compile once, bind many.** `ModelBuilder.compile()` separates immutable model
  structure from validated datasets, including datasets with different row counts.
- **Native execution.** Sampling, state-space operations, and chain coordination execute
  in Rust outside the Python hot path.
- **Deterministic parallelism.** Chains and repeated-model workloads use Rayon with
  stable per-chain seed derivation and ordered results.
- **Bayesian workflow support.** Prior predictive checks, posterior predictive draws,
  pointwise log likelihood, convergence diagnostics, and ArviZ export are available for
  the generic inference path.
- **Coherent uncertainty.** Specialized forecasting APIs retain complete
  `(chain, draw, horizon)` paths so derived totals and other nonlinear quantities can be
  calculated draw by draw.

These are implementation capabilities, not a universal speed or accuracy claim.
Performance and statistical quality depend on the model, data, tuning, and hardware.

## Installation

Install the latest published Python package with:

```bash
pip install rustmc
```

The source tree is prepared as version 0.9.0. Until a 0.9.0 wheel is published, the
package on PyPI may not contain the APIs and correctness changes described on `main`.
To build the current source:

```bash
git clone https://github.com/tbosier/rustmc.git
cd rustmc
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip maturin numpy
maturin develop --manifest-path python_bindings/Cargo.toml --release
```

Python 3.9 through 3.13 are covered by source-install and wheel-install CI. NumPy is the
only required Python runtime dependency. ArviZ and Matplotlib are optional:

```bash
pip install "rustmc[viz]"
```

The Python extension is the supported public package today. `rustmc_core` contains the
Rust implementation, but its public API should still be considered unstable.

## Quick start

This example fits a Bayesian linear regression with NUTS:

```python
import numpy as np
import rustmc as rmc

rng = np.random.default_rng(42)
x = rng.normal(size=1_000)
y = 2.5 * x + rng.normal(size=1_000)

builder = rmc.ModelBuilder()
beta = builder.normal_prior("beta", mu=0.0, sigma=1.0)
builder.normal_likelihood(
    "obs",
    mu_expr=beta * "x",
    sigma=1.0,
    observed_key="y",
)

fit = rmc.sample(
    model_spec=builder.build(),
    data={"x": x, "y": y},
    chains=4,
    warmup=1_000,
    draws=1_000,
    seed=42,
)
print(fit.summary())
```

The same modeling surface supports scalar hierarchical priors, GLM-style expressions,
and a vectorized `beta @ "X"` path backed by faer.

### Reuse one model structure

When the structure is shared across datasets, compile it once and bind new data:

```python
builder = rmc.ModelBuilder()
intercept = builder.normal_prior("intercept", mu=0.0, sigma=5.0)
slope = builder.normal_prior("slope", mu=0.0, sigma=2.0)
builder.normal_likelihood(
    "obs",
    mu_expr=intercept + slope * "x",
    sigma=1.0,
    observed_key="y",
)

compiled = builder.compile()
batch = compiled.sample_batch(
    [
        {"x": x_a, "y": y_a},
        {"x": x_b, "y": y_b},
    ],
    ids=["dataset-a", "dataset-b"],
    chains=4,
    warmup=500,
    draws=1_000,
    seed=42,
)
```

`CompiledModel` validates each binding against the same structural schema. The legacy
`sample()` and `batch_sample()` entry points remain available.

## Forecasting as an application

Forecasting is one application of rustmc's structure-aware inference rather than the
definition of the library. Current specialized models include Bayesian local level,
seasonal local level, local linear trend, and directly observed Gaussian AR(p), plus
fixed-parameter linear Gaussian state-space filtering, smoothing, and a sum-to-zero
seasonal constructor.

```python
values = np.asarray(
    [101, 98, 103, 105, 102, 108, 111, 109, 114, 116, 113, 119,
     121, 118, 123, 126, 124, 129, 131, 128, 134, 136, 133, 139],
    dtype=float,
)

model = rmc.BayesianLocalLevel(
    process_variance_prior=rmc.InverseGammaPrior(shape=3.0, scale=20.0),
    observation_variance_prior=rmc.InverseGammaPrior(shape=3.0, scale=50.0),
    initial_mean=float(values[0]),
    initial_variance=100.0,
)
fit = model.fit(values, chains=4, warmup=500, draws=1_000, seed=42)
forecast = fit.forecast(steps=12, seed=43)

predictive_lower, predictive_upper = forecast.interval(0.95)
level_lower, level_upper = forecast.state_interval(0.95)

# Derived quantities are summarized after calculation within each joint draw.
six_period_totals = forecast.observation_samples[:, :, :6].sum(axis=2)
total_mean = six_period_totals.mean()
total_interval = np.quantile(six_period_totals, [0.025, 0.975])
```

The observation interval is posterior predictive; the latent-level interval is a
credible interval for the expected level. Applications include demand, operations,
sensor data, and financial series such as rebate accruals. Rebate payments are only an
example: seasonal settlement timing, zeros, contract drivers, and positive support need
careful priors and may need calendar, covariate, hurdle, or positive-valued models beyond
the current Gaussian fitted APIs.

Forecasting examples:

- [`examples/rebate_accrual_forecast.py`](examples/rebate_accrual_forecast.py)
- [`examples/bayesian_local_level_forecasting.py`](examples/bayesian_local_level_forecasting.py)
- [`examples/bayesian_seasonal_forecasting.py`](examples/bayesian_seasonal_forecasting.py)
- [`examples/bayesian_local_linear_trend_forecasting.py`](examples/bayesian_local_linear_trend_forecasting.py)
- [`examples/bayesian_ar_forecasting.py`](examples/bayesian_ar_forecasting.py)
- [`examples/custom_state_space_forecasting.py`](examples/custom_state_space_forecasting.py)

## Implemented surface

| Area | Current support |
|---|---|
| Generic inference | NUTS with configurable `target_accept`, fixed-trajectory HMC, transformed continuous parameters, parallel chains |
| Continuous priors | Normal, Student-t, HalfNormal, Exponential, LogNormal, Gamma, Beta, Uniform |
| Likelihoods | Normal, Bernoulli-logit, Poisson-log, Exponential, LogNormal, Negative Binomial |
| Model structure | Scalar hierarchical priors, scalar/vector regression expressions, automatic non-centering for supported scalar hierarchies |
| Diagnostics | Rank-normalized folded split R-hat, rank-normalized bulk/tail ESS, MCSE, empirical 94% HDI, divergences and acceptance summaries |
| Predictive workflow | Prior predictive, posterior predictive, pointwise log likelihood, ArviZ export |
| Repeated models | In-memory compile/bind reuse and parallel batch sampling |
| Fixed state space | Time-homogeneous linear-Gaussian models, Kalman filter, RTS smoother, missing observations, a seasonal constructor, joint and cumulative conditional forecasts |
| Specialized inference | Bayesian local level, seasonal local level, local linear trend, and directly observed Gaussian AR(p) |

Bernoulli and Poisson are exposed for prior-predictive use, but discrete latent
parameters are not suitable for the current gradient-based samplers. Fitted AR(p)
coefficient draws are not constrained to the stationary region; explosive draws are
possible and are not silently discarded.

## Validation and benchmarks

The repository includes finite-difference autodiff checks, analytic and synthetic
posterior recovery, state-space reference tests, cross-thread determinism checks, Python
API tests, and clean-wheel verification.

Run the core verification with:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --release
python -m pytest -q
```

Run `python examples/run_benchmarks.py --help` for the benchmark harness. This README
does not publish a numeric cross-engine result because the repository does not retain a
complete raw output, environment, and revision for one. Use
[`benchmarks/RESULTS_TEMPLATE.md`](benchmarks/RESULTS_TEMPLATE.md) when publishing a
result, and report statistical quality together with wall time.

Tests establish behavior on their stated reference cases. They do not prove that a new
model is appropriate for a user's data or that its intervals are calibrated under
misspecification.

## Current limitations

- The expression and distribution surface is deliberately finite; arbitrary user-defined
  probability functions and broad tensor algebra are not yet supported.
- Vector-valued hierarchical priors, group indexing, named dimensions, and coordinates
  are incomplete.
- Compile/bind artifacts are in-memory only and are not portable or versioned.
- Initialization controls remain limited; BFMI and explicit termination reasons are not
  yet reported.
- The generic state-space API accepts fixed system matrices rather than inferring them.
- Specialized forecasting lacks covariates/calendar interventions, multiple
  seasonalities, positive/robust observations, hierarchical pooling, dated outputs,
  and rolling backtests.
- Performance has not been established on a representative, retained benchmark corpus.

See [`ROADMAP.md`](ROADMAP.md) for the ordered engineering plan and differentiated
capability ideas.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for development and evidence requirements. Bug
reports are most useful when they include a minimal model, seed, environment,
diagnostics, and expected result.

## License

MIT
