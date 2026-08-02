# rustmc

Bayesian inference and state-space forecasting powered by Rust, with a Python API.

> **Project status: alpha.** rustmc is suitable for research, evaluation, and controlled
> internal workflows. It is not yet a sole source for audited financial forecasts or a
> general-purpose replacement for PyMC or Stan.

rustmc is exploring a focused product thesis: make coherent Bayesian forecasts for many
short, related business time series fast to fit, easy to aggregate, and straightforward
to deploy. The current release contains useful foundations for that goal, but seasonality,
regressors, hierarchical fleet models, and forecasting-specific validation are still on
the roadmap.

## Why this project exists

The project is built around workloads where a model structure is reused across many
datasets or where a specialized inference method is preferable to applying a generic
sampler to every latent state.

- Generic NUTS and HMC execute in Rust, outside the Python hot path.
- Independent chains and batch workloads use a shared Rayon thread pool.
- `ModelBuilder.compile()` provides an in-memory compile-once/bind-many path.
- Specialized local-level and local-linear-trend models use FFBS/Gibbs inference.
- Directly observed Gaussian AR(p) models use an exact Normal-Inverse-Gamma posterior.
- Forecasts retain `(chain, draw, horizon)` paths so users can aggregate uncertainty
  coherently across time.
- Fixed linear-Gaussian state-space models provide Kalman filtering, smoothing, missing
  observation handling, and forecasting.

These are implementation facts, not a claim that rustmc is faster or more accurate than
another engine on every workload. Benchmark results depend on the model, sampler,
hardware, tuning, and statistical quality of the draws.

## Installation

Install the latest published Python package with:

```bash
pip install rustmc
```

The source tree is currently prepared as version 0.9.0. Until a 0.9.0 wheel is published,
the PyPI package may not contain the forecasting and correctness changes described on
`main`. To build the current source:

```bash
git clone https://github.com/tbosier/rustmc.git
cd rustmc
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip maturin numpy
maturin develop --manifest-path python_bindings/Cargo.toml --release
```

Python 3.9 through 3.13 are covered by the repository's source-install and wheel-install
CI matrix. NumPy is the only required Python runtime dependency; ArviZ and Matplotlib are
optional:

```bash
pip install "rustmc[viz]"
```

The Python extension is the supported public package today. `rustmc_core` contains the
Rust implementation, but its API should still be considered unstable.

## Forecasting quick start

The local-level model is a nonseasonal baseline for a noisy latent level. It estimates
process and observation variances and returns both latent-state credible intervals and
future-observation posterior-predictive intervals.

```python
import numpy as np
import rustmc as rmc

monthly_values = np.asarray(
    [101, 98, 103, 105, 102, 108, 111, 109, 114, 116, 113, 119,
     121, 118, 123, 126, 124, 129, 131, 128, 134, 136, 133, 139],
    dtype=float,
)

model = rmc.BayesianLocalLevel(
    process_variance_prior=rmc.InverseGammaPrior(shape=3.0, scale=20.0),
    observation_variance_prior=rmc.InverseGammaPrior(shape=3.0, scale=50.0),
    initial_mean=float(monthly_values[0]),
    initial_variance=100.0,
)
fit = model.fit(monthly_values, chains=4, warmup=500, draws=1000, seed=42)
forecast = fit.forecast(steps=12, seed=43)

predictive_lower, predictive_upper = forecast.interval(0.95)
level_lower, level_upper = forecast.state_interval(0.95)

print(forecast.observation_mean)  # expected future observations
print(predictive_lower, predictive_upper)
print(level_lower, level_upper)   # uncertainty in the latent expected level

# Aggregate the paths first, then summarize. Do not sum marginal interval bounds.
six_month_totals = forecast.observation_samples[:, :, :6].sum(axis=2)
print(six_month_totals.mean())
print(np.quantile(six_month_totals, [0.025, 0.975]))
```

For rebate accrual work, distinguish the estimand:

- A 95% **credible interval** describes uncertainty in a latent expected accrual or level.
- A 95% **posterior-predictive interval** describes uncertainty in a future realized
  payment or observation and normally includes more variation.
- A multi-month total must be computed within each posterior path before taking
  quantiles, which preserves dependence across months.

The current fitted forecast models are equally spaced, univariate, Gaussian, and
nonseasonal. Twenty-four monthly observations contain only two annual cycles. Do not use
the example above as a production model for seasonal, zero-heavy, or quarter-end rebate
payments without rolling-origin validation and suitable business drivers.

Complete examples:

- [`examples/rebate_accrual_forecast.py`](examples/rebate_accrual_forecast.py)
- [`examples/bayesian_local_level_forecasting.py`](examples/bayesian_local_level_forecasting.py)
- [`examples/bayesian_local_linear_trend_forecasting.py`](examples/bayesian_local_linear_trend_forecasting.py)
- [`examples/bayesian_ar_forecasting.py`](examples/bayesian_ar_forecasting.py)
- [`examples/custom_state_space_forecasting.py`](examples/custom_state_space_forecasting.py)

## General inference quick start

```python
import numpy as np
import rustmc as rmc

rng = np.random.default_rng(42)
x = rng.normal(size=1000)
y = 2.5 * x + rng.normal(size=1000)

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
    warmup=1000,
    draws=1000,
    seed=42,
)
print(fit.summary())
```

The model builder supports scalar hierarchical priors, GLM-style expressions, and a
vectorized `beta @ "X"` path backed by faer. `ModelBuilder.compile()` can reuse one
validated graph structure across multiple data bindings. The older `sample()` and
`batch_sample()` entry points remain available.

## Implemented surface

| Area | Current support |
|---|---|
| Generic inference | NUTS, fixed-step HMC, transformed continuous parameters, parallel chains |
| Continuous priors | Normal, Student-t, HalfNormal, Exponential, LogNormal, Gamma, Beta, Uniform |
| Likelihoods | Normal, Bernoulli-logit, Poisson-log, Exponential, LogNormal, Negative Binomial |
| Diagnostics | Rank-normalized folded split R-hat, rank-normalized bulk/tail ESS, MCSE, 94% HDI, divergences and acceptance summaries |
| Predictive workflow | Prior predictive, posterior predictive, pointwise log likelihood, ArviZ export |
| Repeated models | In-memory compile/bind reuse and batch sampling |
| Fixed state space | General time-homogeneous linear-Gaussian models, Kalman filter, RTS smoother, missing values, conditional forecasts |
| Fitted forecasting | Bayesian local level, local linear trend, and directly observed Gaussian AR(p) |

Bernoulli and Poisson are available as prior-predictive distributions, but discrete
latent parameters are not suitable for the current gradient-based samplers. The fitted
AR(p) coefficient posterior is not constrained to the stationary region; explosive
draws are possible and must not be silently discarded.

## Validation and benchmarks

The repository includes:

- finite-difference autodiff checks;
- analytic and synthetic posterior-recovery tests;
- deterministic cross-thread tests for specialized forecasting models;
- state-space filter, smoother, missing-value, and forecast tests;
- Python API, packaging, clean-wheel, and supported-version checks; and
- benchmark drivers that record timing and statistical-quality metrics.

Run the core verification with:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --release
python -m pytest -q
```

Run `python examples/run_benchmarks.py --help` for the benchmark harness. No numeric
cross-engine result is claimed in this README because the repository does not currently
retain the complete raw output, environment, and revision needed to audit one. Use
[`benchmarks/RESULTS_TEMPLATE.md`](benchmarks/RESULTS_TEMPLATE.md) when publishing a
result. Report wall time together with R-hat, ESS, divergences, posterior error, memory,
seeds, and matched work across engines.

Tests demonstrate correctness on their stated reference cases. They do not establish
accuracy or interval calibration for a new user's data-generating process. Forecasting
claims require rolling-origin evaluation on representative historical data.

## Current limitations and non-goals

Important current limitations:

- No native fitted seasonal, calendar, holiday, or dynamic-regression forecast model.
- No positive or hurdle observation model in the specialized forecasting API.
- No hierarchical pooling across related time series.
- No unified dated forecast result with built-in cumulative summaries and backtesting.
- Generic fixed state-space matrices are inputs, not inferred parameters.
- Compile/bind artifacts are in-memory only and are not portable or versioned.
- Vector-valued hierarchical priors, named dimensions, and coordinates are incomplete.
- Specialized forecast fits do not yet expose the full generic diagnostic surface.
- Performance has not been established across a representative, reproducible corpus.

rustmc is not currently pursuing arbitrary-PPL feature parity, a Stan-language parser,
GPU inference, or distributed MCMC. Those directions should not displace correctness,
decision-grade forecasting, and a stable deployment artifact.

## Roadmap, in order

### 1. Trust and release integrity

- Publish 0.9.0 from a clean tag with the tested wheel matrix and synchronized Rust and
  Python versions.
- Maintain reference tests for every log density and gradient, plus recovery tests for
  every supported likelihood.
- Extend sampler telemetry with energy/BFMI, tree-depth saturation, leapfrog counts,
  termination reasons, configurable initialization, and `target_accept`.
- Check in raw, reproducible validation and benchmark artifacts before making numeric
  performance or calibration claims.

### 2. Decision-grade univariate forecasting

- Unify forecast results with dates, named dimensions, latent and observation summaries,
  and draw-wise cumulative 3/6/12-period totals.
- Ship a maintained Python stub surface (`.pyi` and `py.typed`) for the public API.
- Add rolling-origin backtests, seasonal-naive and ETS baselines, coverage, bias,
  sharpness, CRPS/WIS, and prior-sensitivity reports.
- Add fitted seasonality, calendar effects, regressors, exposure/offset terms, and
  time-varying matrices.
- Add positive, robust, censored, and zero-heavy observation families suitable for
  revenue, demand, and rebate payments.

### 3. Forecast fleets

- Add hierarchical partial pooling across customers, programs, products, or locations.
- Provide group indexing, panel data, coherent aggregation/reconciliation, partial
  failure handling, and incremental updates.
- Establish a reproducible short-series corpus with calibration and throughput gates.

### 4. Deployment

- Define a portable, versioned, slot-only model artifact.
- Support prediction on new bindings from Rust and Python without rebuilding the graph.
- Provide a stable Rust runtime and then consider C, WASM, or service interfaces.

### 5. Broader inference methods

- MAP estimation and Laplace approximation.
- Variational inference where its approximation quality can be measured.
- Automatic reparameterization and broader hierarchical vector support.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for development and evidence requirements. Bug
reports that include a minimal model, seed, environment, diagnostics, and expected result
are especially valuable.

## License

MIT
