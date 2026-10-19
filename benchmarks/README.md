# Reference benchmark protocol

`benchmarks/run.py` is the reference comparison for rustmc, native PyMC NUTS,
PyMC model compilation with nutpie, and direct NumPyro NUTS. Scripts under
`benchmarks/comparisons/` are exploratory (see [below](#exploratory-comparisons)) and
are not evidence for a public performance claim.

The reference workload is conjugate Gaussian linear regression with known observation
variance. Every engine receives byte-identical float64 `X` and `y`, the same Normal prior,
chain count, warmup count, retained draws, target acceptance probability, maximum tree
depth, data seed, and sampler seed. The exact posterior is calculated analytically, so the
report measures sampler error against the posterior as well as error against the value
that generated the data.

## Install and run

Build rustmc, then install comparison engines as an optional extra:

```bash
maturin develop --release --manifest-path python_bindings/Cargo.toml
python -m pip install -e ".[benchmark]"
python -m benchmarks.run --config benchmarks/configs/quick.json
python -m benchmarks.run --config benchmarks/configs/standard.json \
  --repetitions 3 --randomize-order --order-seed 20260802 \
  --output /tmp/rustmc-standard.json
```

An engine whose optional package is not installed is reported as `unavailable`; the other
engines still run. Use `--engine rustmc --engine numpyro` to select engines, `--dry-run` to
validate the config and deterministic data hash without importing an engine, and
`--list-engines` to inspect the adapter names.

Each engine runs in a fresh subprocess. The controller fixes requested chain concurrency
and sets BLAS/OpenMP threads to one to limit nested oversubscription. The JSON output
records those environment values, package versions, platform, data SHA-256, full config,
phase timings, diagnostics, and stderr tails.

Backend-specific environment is isolated: PyTensor's compilation directory is set only
for PyMC/nutpie, and JAX/XLA host-device flags are set only for NumPyro. Applying JAX
runtime flags to unrelated engines can materially contaminate timings.

## Timing interpretation

The harness reports only phases exposed truthfully by an engine:

| Engine | Separately observable phases |
|---|---|
| rustmc | import, model build, graph compile, data bind, warmup+sample, postprocess |
| PyMC | import, model build, compile+warmup+sample, postprocess |
| PyMC+nutpie | import, model build, compile, warmup+sample, postprocess |
| NumPyro | import, model build, compile+warmup, sample, postprocess |

Calling a throwaway fit merely to populate a cache would make later sampling look cheaper,
so the harness does not do that. JAX work is explicitly blocked before NumPyro phase timers
stop. `fit_seconds` includes build through retained sampling but excludes imports and common
postprocessing. `cold_fit_seconds` adds imports. `total_seconds` also adds common diagnostics.

The primary throughput metric is mean bulk ESS divided by `fit_seconds`. Sampling-only
ESS/s is emitted only when an engine exposes retained sampling separately; comparing that
number against another engine's combined warmup time is invalid.

## Quality gates and interpretation

All engines are summarized with the same ArviZ rank-normalized R-hat and bulk ESS
implementation. Reports also include divergences, minimum ESS, posterior-mean RMSE against
the exact analytic posterior, posterior-mean RMSE against the generating coefficient, and
relative posterior-SD RMSE against the analytic posterior.

Every successful engine row also contains a machine-readable `quality_gate` with the exact
thresholds and failed metrics. `status: ok` means the adapter executed and returned the
requested draws; it does **not** mean statistical quality passed. A failed quality gate
invalidates interpretation of that timing row. Passing is necessary but not sufficient for
publishing a comparison.

Each gated metric is screened against a domain before it is compared with its threshold,
and `quality_gate.domains` publishes those domains alongside `quality_gate.thresholds`.
A bare `>` or `<` reports false for a NaN, so an unscreened comparison appended nothing to
the failure list and the gate that authorises a speed claim passed on a fit whose
diagnostics were not numbers. The gate now fails closed on a metric that is non-finite,
outside the range its estimator can produce, or absent, and on a divergence count that is
negative or fractional. Screening happens per parameter, before the R-hats are reduced with
`max` and the ESS values with `min`: aggregating first hid exactly the half of each domain
the gate cares about, so per-parameter R-hats of `[-100, 1]` reported a healthy maximum of
`1`. Failures are listed twice — once under the bare metric name, which is what this
document and downstream consumers match on, and once as `metric[reason]` so the report says
why a value was rejected.

A domain is derived from the requested chain and draw counts, so it bounds what an estimator
could have returned for a fit of that shape. It does not authenticate a payload: a value
inside the domain is not thereby a value that came from this fit.

A speed ratio should not be published unless:

- all compared engine rows have `status: ok` and the same data SHA-256;
- divergence counts are zero or the result explicitly explains otherwise;
- maximum rank-normalized R-hat is below the declared acceptance threshold;
- posterior error is comparable and small relative to analytic posterior uncertainty;
- raw JSON, exact revision, config, command, and hardware metadata are retained; and
- the conclusion is limited to this model, configuration, and machine.

The quick config checks plumbing only. It has too few draws for a performance or convergence
claim. Use `--repetitions 3 --randomize-order` (or more repetitions) with the standard
config before treating timing differences as stable; record all repetitions rather than
selecting the fastest. The chosen engine order and order seed are retained in the JSON.

## Exploratory comparisons

`benchmarks/comparisons/batch_many_series.py` fits 100 independent demand regressions
with `rustmc.batch_sample` and PyMC+nutpie, then forecasts one of them against Prophet and
ARIMA. It needs optional packages that rustmc does not depend on — ArviZ, PyMC, nutpie,
Prophet and statsmodels, plus matplotlib for its plot — and CI never runs it, so it can
fall behind the API between releases. It does not follow the protocol above, and nothing
it prints may be quoted as a performance claim. `bench_common.py` beside it records the
environment and phase-separated timings.

Three single-model comparisons that used to live in `examples/`
(`compare_with_pymc.py`, `benchmark_vs_pymc.py` and `benchmark_multivariate.py`) were
removed rather than kept unverified. They were Gaussian regressions with known noise at
1, 500 and 10 coefficients, which `benchmarks/run.py` measures under the protocol above
when `observations`, `parameters` and `chains` are changed in a config. They are not
identical workloads: `run.py` declares the coefficients as one vector parameter, while
two of the removed scripts used separate scalar priors and one added an intercept. The
removed scripts remain in the repository history.

`benchmarks/state_variance_mixing.py` measures how slowly seven Gibbs/FFBS forecasting
fits (local level, local linear trend and seasonal local level, the last two also with
`exog`, and a `StructuralModel` trend at damping 1 and 0.9) mix their state variances: the worst split R-hat over three seeds in ten settings
(five regimes, 200 and 1000 points) at 4 chains, 300 warmup and 500 draws, with data
simulated from each model's own process. The R-hat ranges in the `known_limitation` fit
warnings and in the reference's slow state-variance mixing section come from it.

Four scripts measure how `metric="auto"` chooses a dense block; the
statistical-validation page quotes their records. `curved_metric_choice.py` fits a curved
two-dimensional target, written as one vector parameter, under each `metric` (five seeds
unless `--first-seed` and `--seeds` say otherwise) and prints the smallest bulk ESS, the
largest R-hat and the divergences. `auto_metric_targets.py` fits nine targets and prints
leapfrog steps per unit of the smallest bulk ESS. With `--json`, both save every fit and
the build it ran on (`build_provenance.py`), and `compare_metric_records.py` compares two
such records fit by fit. `auto_metric_noise.py` applies NumPy copies of 0.14.0's rule and
the current one to simulated warmup windows, with no sampler.

## Repeated inference

`benchmarks/repeated.py` fits many Gaussian linear regressions with known noise through
one compiled model per engine: rustmc one dataset at a time (`rustmc`) and in one
`sample_batch` call (`rustmc_batch`), nutpie with `CompiledModel.with_data`, NumPyro with
one `MCMC` object, and PyMC with `pm.Data` and `pm.set_data`. Each dataset is generated
from its own seed, with a fixed row count or, for a ragged workload, one drawn per
dataset, and every engine gets the same data. After each fit the engine predicts the
expected response at new inputs: rustmc with `predict(expected=True)`, the others by
multiplying the draws with the inputs in NumPy.

Each engine runs in its own subprocess with the environment `benchmarks.run` uses.
Setup (imports, model building, compilation) is timed apart from each fit; importing
ArviZ, which computes every engine's diagnostics, is not timed. Every fit goes through
the quality gate above against its exact posterior. A fit that fails the gate or raises
is kept with its reason, its time counts, and its ESS does not: `useful_ess_per_second`
is the bulk ESS of passing fits divided by setup plus every fit's time. The JSON keeps
every fit's phases, quality, gate result and resident memory after the fit, plus the git
revision, with the home directory written as `~`.

```bash
PYTHONPATH=.pybuild python -m benchmarks.repeated --config benchmarks/configs/repeated-fixed.json \
  --output repeated-fixed.json
```

`benchmarks/results/2026-10-19-repeated-fixed.json` and `-ragged.json` hold one run each
on a 24-thread AMD Ryzen 9 5900X, with 30 datasets of 8 coefficients, 4 chains, 500
warmup, 1000 draws and 4 threads: 500 rows each, or 50 to 2000 rows drawn per dataset.
Every fit of every engine passed the gate.

| Engine | Setup s | Fits s, fixed | Fits s, ragged | Useful ESS/s, fixed | Useful ESS/s, ragged |
|---|---|---|---|---|---|
| rustmc, one dataset at a time | 0.01 | 3.11 | 6.81 | 62056 | 28761 |
| rustmc, `sample_batch` | 0.01 | 2.98 | 6.37 | 66335 | 30623 |
| nutpie | 7.85, 7.86 | 1.92 | 2.90 | 16874 | 15715 |
| NumPyro | 0.28 | 24.87 | 25.47 | 7728 | 7623 |
| PyMC | 0.46, 0.44 | 26.16 | 27.98 | 5697 | 5402 |

nutpie fits each dataset faster than rustmc: 1.92 s against 3.11 s for the 30 fixed
datasets, and 2.90 s against 6.81 s for the ragged ones, whose largest datasets have 2000
rows. Its useful ESS per second is lower over 30 datasets because of its 7.9 s of
compilation and a lower ESS per fit: a mean bulk ESS of 5491 and 5636 per fit against
rustmc's 6439 and 6530. At these per-fit times nutpie's total time would fall below
rustmc's after about 200 fixed or 60 ragged datasets of this size. PyMC's
first fit took 6.7 s, most of it compilation, and later fits a median of 0.65 s (fixed
workload). NumPyro's
time is mostly sampling: about 0.8 s per fit in both workloads, although the ragged one
compiles again for each new data shape, and in a separate check four runs of one
`MCMC` object on same-shaped data took 0.95 s and then 0.79 to 0.81 s. Peak resident memory was 344 to 369 MB for the
two rustmc runs, 546 to 553 MB for nutpie, 1232 to 1241 MB for NumPyro and 423 MB for
PyMC. One run on one machine is not a stable ranking; rerun it before quoting a ratio.

