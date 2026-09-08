# Independent forecasting batch probe

These are single-run operational probes of revision `0474d4669920c2b52d9ddbd4d1143037d8e7897d`,
not a stable performance comparison. Both processes fitted the same seeded 128
ragged local-level series (48–60 observations), four chains, 100 warmup iterations,
200 retained draws, then 12-step forecasts. The simulated process SD was 0.2 and
observation SD 0.4; both variances were inferred under IG(3, 0.3) priors. CPU,
platform, versions, exact commands, seeds and all controls are retained in the raw
[one-worker](2026-09-08-forecast-batch-threads1.json) and
[four-worker](2026-09-08-forecast-batch-threads4.json) output. Builds used
`maturin develop --release --offline`, `CARGO_BUILD_JOBS=2`, and an isolated target
and Python environment. The private pool's explicit worker count controls Rayon;
no BLAS operations are involved in this scalar kernel.

| Probe | Fit including warmup | Forecast | Peak process RSS |
|---|---:|---:|---:|
| 1 worker | 0.20349 s | 0.03727 s | 73,384 KiB |
| 4 workers | 0.06077 s | 0.01163 s | 72,928 KiB |

Compilation/import/data generation and diagnostics are outside the measured fit
and forecast phases. RSS includes both retained fits and forecasts, Python imports,
and subsequent diagnostic calculation. Sampling and warmup are not timed separately.
There is no comparison to another inference engine or posterior-recovery claim.

The maximum rank-normalized folded split R-hat was **1.115**, minimum bulk ESS
**27.9**, and minimum tail ESS **47.2** in both runs. This short sampler schedule
**fails convergence-quality criteria** and is not suitable for final inference.
Gibbs has no Hamiltonian divergences or Metropolis acceptance statistic. Retained
streams were independently tested to be identical under thread, ordering, and
chunk changes. A repeated, longer schedule with acceptable diagnostics is needed
before claiming useful inference throughput or a stable speedup.
