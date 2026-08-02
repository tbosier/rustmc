# Reference benchmark protocol

`benchmarks/run.py` is the reference comparison for rustmc, native PyMC NUTS,
PyMC model compilation with nutpie, and direct NumPyro NUTS. Scripts under
`examples/benchmark_*.py` are exploratory examples and are not evidence for a public
performance claim.

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
