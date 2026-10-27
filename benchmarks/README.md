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
one model per engine: rustmc one dataset at a time (`rustmc`) and in one `sample_batch`
call (`rustmc_batch`), nutpie with `CompiledModel.with_data`, NumPyro with one `MCMC`
object and parallel chains (`numpyro`) or vectorized ones (`numpyro_vectorized`), and
PyMC with `pm.Data` and `pm.set_data`. NumPyro documents that `jit_model_args` does not
take effect for several parallel chains, so `numpyro` compiles again for every fit;
`numpyro_vectorized` reuses its compiled model for every dataset of a shape it has seen,
and keeps one `MCMC` object per shape because rerunning vectorized chains on a new shape
fails in NumPyro 0.21. Each dataset is generated from its own seed, with a fixed row
count or, for a ragged workload, one drawn per dataset, and every engine gets the same
data. After each fit the engine predicts the expected response at new inputs: rustmc
with `predict(expected=True)`, the others by multiplying the draws with the inputs in
NumPy.

Each engine runs in its own subprocess with the environment `benchmarks.run` uses.
Setup (imports, model building, compilation) is timed apart from each fit; importing
ArviZ, which computes every engine's diagnostics, is not timed. Every fit goes through
the quality gate above against its exact posterior, and through checks that gate does
not make. The eigenvalues of the draws' covariance whitened by the exact one must lie
within `quality_max_sd_relative_rmse` plus `2 sqrt(d / ESS)` of 1, with the smallest
bulk ESS, the Marchenko-Pastur allowance for sampling noise, but never more than 0.5
away, so draws with the right marginals but the wrong correlations fail, and so does a
fit with too little ESS for its dimension to verify. Predictions must have the shape
`(chain, draw, row)`, their mean must be within `quality_max_mean_error_sd_units` of the
exact expected response in units of its exact posterior standard deviation, and their
standard deviation within `quality_max_sd_relative_rmse` of that one, relative. A fit that fails or
raises is kept with its reason, its time up to the failure counts, and its ESS does not:
`useful_ess_per_second` is the bulk ESS of passing fits divided by setup plus every
fit's time. The JSON keeps every fit's phases, quality, gate result and resident memory
after the fit, the largest child process that exited, plus the git revision, with the
home directory written as `~`.

```bash
PYTHONPATH=.pybuild python -m benchmarks.repeated --config benchmarks/configs/repeated-fixed.json \
  --output repeated-fixed.json
```

`benchmarks/results/2026-10-24-repeated-fixed.json` and `-ragged.json` hold one run each
on a 24-thread AMD Ryzen 9 5900X, with 30 datasets of 8 coefficients, 4 chains, 500
warmup, 1000 draws and 4 threads: 500 rows each, or a row count drawn per dataset
between 50 and 2000 (103 to 1949 in this run). Every fit of every engine passed.

| Engine | Setup s | Fits s, fixed | Fits s, ragged | Useful ESS/s, fixed | Useful ESS/s, ragged |
|---|---|---|---|---|---|
| rustmc, one dataset at a time | 0.01 | 1.53 | 3.36 | 126267 | 57716 |
| rustmc, `sample_batch` | 0.01 | 1.42 | 3.13 | 136180 | 61819 |
| nutpie | 7.83, 7.88 | 1.93 | 2.85 | 16869 | 15760 |
| NumPyro, parallel chains | 0.27 | 25.36 | 26.17 | 7583 | 7427 |
| NumPyro, vectorized chains | 0.27 | 31.31 | 54.76 | 6154 | 3549 |
| PyMC | 0.45 | 25.95 | 27.84 | 5744 | 5428 |

rustmc's 30 fixed-size fits took 1.53 s against nutpie's 1.93 s, and it was the faster
of the two on all 30; nutpie also spends about 8 s compiling first. On the ragged
datasets nutpie's fits took 2.85 s against rustmc's 3.36 s, and it was faster on 24 of
the 30, so at these times its total would fall below rustmc's after about 460 of them.
rustmc's mean bulk ESS per fit is 6469 and 6480 against nutpie's 5491 and 5636. PyMC's
first fit took 6.7 to 6.8 s and its fits a median of 0.65 s (fixed) and 0.73 s
(ragged). NumPyro with parallel chains took a median of 0.81 and 0.84 s per fit,
compiling each time. With vectorized chains its first fit took 3.0 to 3.1 s and the
later fixed-size ones a median of 0.98 s: reusing the compiled model did not make up
for sampling the four chains on one device, and in the ragged workload, where every
dataset has a new shape, the median was 1.8 s. Peak resident memory of the engine's
own process was 345 to 370 MB for the two rustmc runs, 549 to 553 MB for nutpie, 1237
to 1462 MB for NumPyro and 423 to 424 MB for PyMC. Child processes, which hold PyMC's
chains and anything else an engine or its imports start, are recorded apart
(`peak_children_rss_mb`, the largest that exited): 394 to 402 MB for PyMC and nutpie
and 62 to 64 MB for the others. The runs do not record which child reached each peak.
The runs were made on a machine shared with other work, each started only once the
one-minute load average was below 2.5; the records do not store the load. One
run on one machine is not a stable ranking; rerun it before quoting a ratio.

The 2026-10-19 records, from before narrow design matrices were multiplied with fused
loops instead of faer, show rustmc at 3.11 s and 6.81 s for the same fits, with the
other engines within a few percent of the times above.

## rustmc against nutpie

`benchmarks/head_to_head.py` fits the same models with rustmc and nutpie, at one and
four threads: Gaussian regressions with known and inferred noise, logistic and Poisson
regressions at five sizes from 100 rows by 4 columns to 10,000 by 16 and 1,000 by 128, a
non-centered varying-intercept model at 500 rows in 20 groups and 5,000 in 100, and
non-centered eight schools. Both engines get the same priors, data, chains, warmup and
draws, and their own adaptation. The script's docstring describes the timed spans and
what each fit records; `benchmarks/head_to_head_summary.py` turns a record into
verdicts: a sign test on the per-ESS time ratio (seconds per smallest bulk ESS, rustmc
over nutpie) of the datasets where both fits pass the quality gate (R-hat at most 1.01,
no post-warmup divergence, smallest bulk ESS at least 400), Holm-adjusted within each
model class.

```bash
PYTHONPATH=.pybuild python -m benchmarks.head_to_head --datasets 10 --lock /tmp/bench.lock \
  --output head-to-head.json
python benchmarks/head_to_head_summary.py head-to-head.json
```

`benchmarks/results/2026-10-27-head-to-head.json` holds one run of all 46 cells, with 10
datasets each (eight schools has one dataset, so its 10 fits differ only in the sampler
seed), 4 chains, 1000 warmup and 1000 draws, on a 24-thread AMD Ryzen 9 5900X, at
revision e1ec4ac (main at 18ede80); the `.txt` beside it is the run's log and both
analyses. Every fit finished, each started at a one-minute load below 1.5, and the
regressions also ran nutpie at its default 300 warmup iterations (`nutpie300`). The
table's per-ESS ratios run to an InferenceData holding each engine's posterior draws
and post-warmup sample statistics. That work is close but not identical: rustmc also
writes a small `observed_data` group, and each engine converts its own set of sample
statistics, more of them for nutpie. Medians and observed ranges over the 50 passing
datasets of each class:

| Model class | Threads | vs nutpie | Cells rustmc faster | vs nutpie300 | Cells rustmc faster |
|---|---|---|---|---|---|
| Normal, known noise | 1 | 0.41 [0.15, 0.73] | 5 of 5 | 0.60 [0.24, 0.88] | 5 of 5 |
| Normal, known noise | 4 | 0.40 [0.15, 0.62] | 5 of 5 | 0.52 [0.25, 1.01] | 5 of 5 |
| Normal, inferred noise | 1 | 0.40 [0.16, 0.54] | 5 of 5 | 0.55 [0.23, 0.77] | 5 of 5 |
| Normal, inferred noise | 4 | 0.38 [0.14, 0.53] | 5 of 5 | 0.50 [0.20, 0.90] | 5 of 5 |
| Logistic | 1 | 0.43 [0.19, 0.75] | 5 of 5 | 0.69 [0.27, 1.09] | 3 of 5 |
| Logistic | 4 | 0.43 [0.19, 0.80] | 5 of 5 | 0.65 [0.27, 1.18] | 4 of 5 |
| Poisson | 1 | 0.47 [0.16, 0.61] | 5 of 5 | 0.69 [0.25, 0.99] | 5 of 5 |
| Poisson | 4 | 0.35 [0.18, 0.56] | 5 of 5 | 0.51 [0.27, 0.92] | 5 of 5 |

No regression cell went to nutpie. Timing sampling alone, which leaves both conversions
out, moves every class median against nutpie by 0.031 or less and keeps all 40 cells
rustmc's; against `nutpie300` the medians rise by 0.019 to 0.102, and rustmc wins 34 of
the 40 cells instead of 37. In the table's three logistic cells without a winner
against `nutpie300` (100 by 4 at one thread, and 1,000 by 64 at both) rustmc was faster
on 9 of 10 datasets at 100 by 4 and on 8 of 10 at 1,000 by 64, short of significance
after adjustment. The gap is widest at 128 columns, where the cell medians against
nutpie are 0.16 to 0.23; at the other sizes they are 0.28 to 0.68. rustmc takes 1.08
to 1.40 times nutpie's leapfrog steps on these models, and each step costs 0.20 to 0.50
times nutpie's. nutpie also compiles each model first, 2.4
to 8.2 s per cell here; rustmc builds its model in under a millisecond, and neither is
in the ratios.

The varying-intercept model goes the other way. At 500 rows in 20 groups nutpie misses
the gate on 5 of 10 datasets (R-hat from 1.0103 to 1.0132, or a smallest bulk ESS of
391), rustmc on none, and on the 5 that pass rustmc takes 1.46 times nutpie's time per
ESS at one thread and 1.38 at four (1.60 and 1.85 counting sampling alone), which is no
verdict at 5 pairs. At 5,000 rows in 100 groups both engines miss the gate on every dataset, so there
is no verdict, but rustmc's sampling takes 5.3 s against nutpie's 0.93 s at one thread.
The `Where the sampling time goes` table in the record splits the gap: rustmc takes
2.66 and 2.24 times nutpie's leapfrog steps at the two sizes and spends 1.5 to 1.6 and
2.3 to 2.6 times as long per step; at 500 by 20 the extra steps return 2.6 times nutpie's
smallest bulk ESS, and at 5,000 by 100 they return 1.05 times. On eight schools, 5 of
rustmc's 10 fits and 7 of nutpie's diverge at least once, so only one pair passes;
rustmc's ratio on all 10 finished pairs is 0.52 to 0.81 at one thread and 0.42 to 0.64
at four, which is not a verdict either.

`benchmarks/results/2026-10-25-head-to-head.json` is the run before #47 and #48 fused the
varying-intercept model's gather, scale and Normal term, with 3 datasets per cell. Its
rustmc sampling took 2.87 s and 22.5 s at the two varying-intercept sizes on one thread,
against 0.73 s and 5.3 s here. Its InferenceData spans held different groups for the two
engines; read it with `--endpoint sampling`. One run on one machine is not a stable ranking.

