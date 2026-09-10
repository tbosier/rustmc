Repository review, 9 September 2026 — rustmc 0.11.0

This is the pre-implementation review. Source links are pinned to the reviewed commit.
Version 0.12 addresses the correctness findings and prioritized model/workflow work;
see [the workflow guide](forecasting-workflows.md) for the implemented APIs and boundaries.

Reviewed revision: `7e471842607a4391232cb10e7e50c83a12a5979e`.

**Recommendation:** make custom models and forecasting share a coherent construction,
prediction, and evaluation workflow. The existing native samplers, Kalman/FFBS code,
compile/bind separation, and joint forecast draws provide a useful foundation. The
largest development constraint is the separation between a narrow Python expression
builder and individually implemented forecasting models. Adding more standalone model
classes will increase the integration and validation work for each subsequent feature.

This is a source review with targeted reproductions, not a complete statistical audit.
No implementation changes were made. Existing untracked review documents were preserved.

**Confirmed correctness findings, in priority order**

1. **High: custom Rust regression forecasts can omit process uncertainty.**

   [RegressionConfig](https://github.com/tbosier/rustmc/blob/7e471842607a4391232cb10e7e50c83a12a5979e/rust_core/src/bayesian_regression.rs#L20) accepts a general
   `LinearGaussianStateSpace` plus a list of innovation coordinates whose variances are
   learned. Fitting preserves process covariance supplied on other coordinates.
   [Forecasting](https://github.com/tbosier/rustmc/blob/7e471842607a4391232cb10e7e50c83a12a5979e/rust_core/src/bayesian_regression.rs#L370), however, adds innovations
   only for the listed learned variances. Fixed process noise in the supplied model
   disappears from future paths.

   Reproduction: fit a scalar model with fixed `Q=9`, empty `innovation_indices`, empty
   variance-prior/name lists, a proper coefficient prior, and all-zero exogenous design.
   This is accepted. Across 200 forecast paths, the maximum absolute change between
   consecutive future latent levels was exactly zero. The specified random walk requires
   increment variance 9. The terminal posterior remains uncertain, but additional
   uncertainty stops accumulating with the forecast horizon.

   The built-in Python regression presets do not trigger this case: their nonzero
   process variances are all represented in the learned list. This matters directly for
   users constructing custom models through the published Rust API.

   Fix: build each draw's complete process covariance and use a shared state simulation
   routine. Until supported, reject configurations outside the actual conjugate kernel's
   assumptions. Validate unique innovation coordinates and the covariance structure;
   independent inverse-gamma updates are not a general covariance-learning algorithm.
   Verify fixed, learned, and mixed covariance cases against conditional Kalman moments.

2. **High: Exponential predictive simulation silently changes the distribution.**

   Both [posterior prediction](https://github.com/tbosier/rustmc/blob/7e471842607a4391232cb10e7e50c83a12a5979e/python_bindings/src/lib.rs#L2782) and
   [prior prediction](https://github.com/tbosier/rustmc/blob/7e471842607a4391232cb10e7e50c83a12a5979e/python_bindings/src/lib.rs#L3732) clamp the rate and generated
   amounts to at least `1e-12`. This is not part of the Exponential likelihood.

   With constant log-rate 40, the true conditional mean is about `4.248e-18`; all 1,000
   prior draws and all 1,000 posterior predictive draws equaled `1e-12`. With log-rate
   -40, the true mean is about `2.354e17`, while generated sample means were about
   `9.85e11` and `9.88e11`. These values are representable in floating point.

   Fix: centralize native observation simulation, preserve the specified distribution,
   and explicitly reject unrepresentable results when necessary. Use an appropriate
   open-interval uniform or native Exponential generator instead of flooring valid
   amounts. Test distributional moments at small and large scales on both predictive
   entry points. Audit the other family's numerical shortcuts at the same boundary.

3. **Medium: Rust sampler configuration validation is weaker than Python's.**

   [sample_bound](https://github.com/tbosier/rustmc/blob/7e471842607a4391232cb10e7e50c83a12a5979e/rust_core/src/sampler.rs#L212) validates `target_accept`, binding,
   and the initial target, but does not enforce the full sampling configuration contract.
   Direct Rust calls with zero chains returned success and a NaN mean. HMC with zero
   leapfrog steps returned five identical zero draws for a standard-Normal target and
   reported acceptance 1.0. The Python wrapper rejects these configurations.

   Fix: validate configurations in the core and use the same validation from every
   entry point. Cover iteration arithmetic, chain/draw counts, step size, HMC trajectory
   length, and NUTS depth. Return resource errors from thread-pool construction rather
   than using `expect`. Expose per-chain initialization with validation; all public
   generic Python chains currently start at the same raw zero vector.

4. **Medium: duplicate likelihood names overwrite predictive results.**

   Two calls to `normal_likelihood("obs", ...)` are accepted by `build()` and `compile()`.
   Both likelihood terms enter the model, but [dictionary packaging](https://github.com/tbosier/rustmc/blob/7e471842607a4391232cb10e7e50c83a12a5979e/python_bindings/src/lib.rs#L3788)
   puts both output arrays under the same name. The second overwrites the first. The same
   naming issue affects posterior predictive and pointwise likelihood output. Prior
   predictive output also shares one dictionary namespace between parameters and responses.

   Reproduction: give two Normal likelihoods the name `obs`, with means `a` and `a+100`.
   Compilation succeeds and prior prediction returns only one `obs` array.

   Fix: enforce unique observation names during construction, and distinguish posterior
   parameter variables from predictive variables in structured result groups. Preserve
   existing flat output compatibility through an explicit collision policy.

5. **Medium: the forecasting study's score is not standard WIS.**

   [weighted_interval_score](https://github.com/tbosier/rustmc/blob/7e471842607a4391232cb10e7e50c83a12a5979e/demo-docs/run_rustmc_forecast.py#L145) divides by
   `0.5 + sum(alpha/2)`, which is 0.625 for its two intervals. With its numerator,
   standard WIS divides by `K + 0.5`, which is 2.5. A deterministic forecast of zero
   against an actual value of one returns 4.0; standard WIS returns 1.0.
   See the [scoringutils definition](https://github.com/epiforecasts/scoringutils/blob/main/vignettes/scoring-rules.Rmd).

   This fixed factor does not change candidate rankings when every candidate uses the
   same interval grid. It does make reported magnitudes incompatible with standard WIS.
   Separately, the GP/Prophet baseline's [wis_95](https://github.com/tbosier/rustmc/blob/7e471842607a4391232cb10e7e50c83a12a5979e/demo-docs/baselines/gp_prophet_baselines.py#L110)
   is an unweighted single-interval score. Those names should not imply interchangeable
   metrics. Correct the scorer, its labels, and affected retained score artifacts.

**Dead code and cleanup candidates**

Repository-wide call searches found the following. Public Rust symbols may have external
users; “unused in this repository” is not proof that deleting a published API is safe.

| Item | Evidence | Recommended treatment |
|---|---|---|
| Unused core dependency | `ndarray` in [rust_core/Cargo.toml](https://github.com/tbosier/rustmc/blob/7e471842607a4391232cb10e7e50c83a12a5979e/rust_core/Cargo.toml#L19), with no use in core source, examples, or tests | Remove the core's direct dependency; keep the binding dependency, which is used. Transitive dependencies may still bring ndarray into the build. |
| Unused abstraction | [distributions::Distribution](https://github.com/tbosier/rustmc/blob/7e471842607a4391232cb10e7e50c83a12a5979e/rust_core/src/distributions.rs#L3) has no implementations or consumers in the repository | Deprecate/remove it or replace it with a deliberately designed extension interface. It does not currently enable custom distributions. |
| Dead argument | [batch_sample](https://github.com/tbosier/rustmc/blob/7e471842607a4391232cb10e7e50c83a12a5979e/rust_core/src/sampler.rs#L448) accepts `(Graph, Vec<f64>)` and explicitly discards `obs_y` at line 483; Python supplies an empty vector | Introduce an internal graph-only path and retire the old argument through a compatibility wrapper. |
| Obsolete helper | [Graph::normal_obs_predictors](https://github.com/tbosier/rustmc/blob/7e471842607a4391232cb10e7e50c83a12a5979e/rust_core/src/graph.rs#L615) has no repository callers and still describes a Normal-only surface | Retire in favor of `observation_heads`. |
| Unused convenience helper | [Evaluator::vec_to_owned](https://github.com/tbosier/rustmc/blob/7e471842607a4391232cb10e7e50c83a12a5979e/rust_core/src/autodiff.rs#L191) has no repository callers | Retain only if intended public API; otherwise deprecate. |
| Legacy artifact subsystem | [compiled_model.rs](https://github.com/tbosier/rustmc/blob/7e471842607a4391232cb10e7e50c83a12a5979e/rust_core/src/compiled_model.rs) is about 1,550 lines; its data-owning JSON format is separate from Python `CompiledModel` | It is tested and publicly exported, so it is not simply dead code. Isolate it as legacy and design one versioned replacement if serialization is a priority. |
| Duplicate evaluator | [forward/grad_logp](https://github.com/tbosier/rustmc/blob/7e471842607a4391232cb10e7e50c83a12a5979e/rust_core/src/autodiff.rs#L1076) duplicate much of the production `Evaluator`; gradient checks use them | Preserve the reference value while moving it to a clearly named reference module or feature. Maintain independent numerical checks before retiring anything. |
| Stale product documentation | [README limitations](https://github.com/tbosier/rustmc/blob/7e471842607a4391232cb10e7e50c83a12a5979e/README.md#L285) still say forecasting lacks covariates and positive observations; [ROADMAP](https://github.com/tbosier/rustmc/blob/7e471842607a4391232cb10e7e50c83a12a5979e/ROADMAP.md#L25) starts with a future 0.9 gate | Update against the shipped 0.11 capabilities so future work is not planned from obsolete gaps. |

The larger cleanup opportunity is duplicated live code: inverse-gamma utilities,
seed derivation, path summaries, quantiles, validation, predictive family switches,
and NumPy/ArviZ conversion. Consolidate those contracts first. The dedicated scalar
and two-state FFBS implementations have a useful performance purpose and reference
tests; replacing them all with a dense generic matrix implementation would need evidence.

**Architectural changes that unlock custom models**

The Python binding [lib.rs](https://github.com/tbosier/rustmc/blob/7e471842607a4391232cb10e7e50c83a12a5979e/python_bindings/src/lib.rs) has 7,387 lines. It includes
model syntax, compilation, validation, prior sampling, observation simulation,
diagnostics, array conversion, and most forecasting classes. The recently extracted
modules still use `super::*`, so their dependencies remain implicit.

Split by responsibility while preserving public imports: model expressions/specifications,
compile/bind, inference, predictive simulation, results/conversion, and forecast presets.
Statistical kernels should live in the Rust core. Moving code into files helps, but the
important boundary is that Python conversion no longer owns a second implementation of
the observation model.

The current [MuExpr](https://github.com/tbosier/rustmc/blob/7e471842607a4391232cb10e7e50c83a12a5979e/python_bindings/src/lib.rs#L431) supports constants, parameters,
parameter-times-data, addition, and matrix-vector products. Live probes confirmed that
`a*b`, `a*2.0`, and `a-1.0` fail. There is no public builder operation for a custom
log-density contribution, group indexing, named deterministic outputs, or a temporal
recurrence. The Rust graph supports more scalar arithmetic than Python exposes.

Build custom modeling in layers:

1. Complete scalar expression arithmetic and useful transforms; preserve shape checking
   and parameter ownership. Add named deterministic outputs and a `potential` operation
   over supported expressions, with gradient tests.
2. Introduce explicit dimensions and validated indexing. Currently
   [all vector-like data share one length](https://github.com/tbosier/rustmc/blob/7e471842607a4391232cb10e7e50c83a12a5979e/python_bindings/src/lib.rs#L1391), and
   [DataBinding](https://github.com/tbosier/rustmc/blob/7e471842607a4391232cb10e7e50c83a12a5979e/rust_core/src/data.rs#L65) stores one global `n_obs`. This blocks a
   natural generic model with separate occurrence observations, positive severities,
   ragged groups, or multiple response populations. Named dimensions must reach the
   evaluator and observation metadata, not just output labels.
3. Define a native log-density/gradient boundary for models that cannot be expressed
   through the built-in graph. Specify dimension, initialization, transforms, and failure
   behavior. Keep custom density evaluation separate from predictive generation: an
   arbitrary log density does not automatically supply a random-number generator.
4. Add a state-space component specification for structured time series. It should
   declare state slices, initial priors, transition and observation schedules, innovation
   covariance, and learned parameters. Compile supported Gaussian models into Kalman/
   FFBS kernels, with explicit checks on inference assumptions.

Do not start with an unrestricted tensor language or automatic inference planner.
An explicit kernel choice and a small validated set of model structures are enough to
make substantial progress. The [Stan custom-probability interface](https://mc-stan.org/docs/stan-users-guide/custom-probability.html)
is a useful reference for separating a density contribution from a distribution API.

**Make prediction a first-class operation**

[FitResult::posterior_predictive](https://github.com/tbosier/rustmc/blob/7e471842607a4391232cb10e7e50c83a12a5979e/python_bindings/src/lib.rs#L2716) only uses the
training graph and takes `n_samples` and `seed`. Passing new data raises `TypeError`.
`CompiledModel.bind()` can bind another training dataset, but there is no operation to
apply an existing posterior to new predictors. This is the most direct missing feature
for custom regression forecasting.

Add a prediction binding that requires future predictors and their dimensions, without
requiring fabricated response values. Preserve the fitted parameter identities and
chain/draw axes. Distinguish an expected-response draw from a realized observation draw.
For dynamic models, forecasting must recursively propagate the state and retain the
same parameter draw throughout each path. Generic regression prediction alone does not
provide that recursion.

Use a common result protocol across kernels: named parameter draws, sampler metadata,
observed data, optional latent states, predictive draws, and coordinates. Today a
Gaussian fit changes its concrete result class when `exog` is supplied, while generic
batch results omit `diagnostics()` and predictive methods entirely. Consistent protocols,
maintained `.pyi` files, and a `py.typed` marker would make the package much easier to
learn and extend without removing existing classes.

The generic and forecasting batch implementations also have different contracts.
[Generic batches](https://github.com/tbosier/rustmc/blob/7e471842607a4391232cb10e7e50c83a12a5979e/rust_core/src/sampler.rs#L348) seed by position, discard successful
results when another cell fails, and construct a one-thread pool inside each cell's
sampling call. Live reordering changed draws for the same supplied ID. The newer
[forecast executor](https://github.com/tbosier/rustmc/blob/7e471842607a4391232cb10e7e50c83a12a5979e/rust_core/src/forecast_batch.rs#L29) already provides stable IDs,
one controlled pool, and collected failures. Reuse that executor for custom models,
with an explicit versioned seed-policy migration. Add streaming submission/retention
when needed; dispatching chunks currently still retains all returned results.

**Forecasting capabilities to prioritize**

| Order | Capability | Why it matters | Implementation boundary |
|---|---|---|---|
| 1 | Rolling-origin backtests and probabilistic scoring | Makes model and prior choices measurable at the horizons users need | Extract a reusable evaluation API from the demo, correct WIS, add sample-based CRPS, bias, interval coverage/width, and naive/seasonal-naive baselines. Fit preprocessing only on each training fold. |
| 2 | Composable Gaussian structural models | Lets users combine level/trend, regression, seasonal components, and AR residuals | Assemble a validated state system; share fit, forecast, diagnostics, and component outputs. Preserve optimized kernels where appropriate. |
| 3 | Dynamic regression coefficients | Effects can evolve as behavior or operating conditions change | Existing exog uses static uncertain beta. Add coefficient state innovations and regularizing drift priors; return coefficient paths. |
| 4 | Student-t observation noise | Reduces sensitivity of level/trend estimates to isolated outliers | Start with fixed degrees of freedom and a validated Normal-scale-mixture sampler. This requires time-varying observation variances and latent-scale updates. A new likelihood enum alone is insufficient. |
| 5 | Hierarchical dynamic forecasts | Pools short or sparse related series while learning shared effects | Existing hierarchy pools static means only. Start with group-level priors on regression coefficients and innovation scales; retain aligned joint draws and add shared shocks when the model calls for them. |
| 6 | Dynamic Poisson/negative-binomial models with exposure | Forecasts event counts with suitable support and overdispersion | The generic NB likelihood already exists; the missing work is temporal structure, offsets/exposure ergonomics, recursive prediction, and a valid non-Gaussian inference kernel. |
| 7 | Richer intermittent-amount models | Models changing zero frequency as well as positive magnitude | Extend hurdle occurrence with covariates/time dependence and severity with regressors/pooling. Consider compound Poisson-Gamma when observations are sums of event amounts. Validate all-zero histories and predictive moment existence. |
| 8 | Damped trends and stochastic harmonic seasonality | Controls long-horizon extrapolation and handles multiple seasonal cycles | Static multiple Fourier designs can already be concatenated manually. Add declarative component names, evolving amplitudes, time indexing, and fractional periods where meaningful. Learned damping parameters require an appropriate parameter kernel. |

The component approach has established precedents in
[PyMC Extras structural models](https://www.pymc.io/projects/extras/en/latest/statespace/models/structural.html).
[TFP dynamic regression](https://www.tensorflow.org/probability/api_docs/python/tfp/sts/DynamicLinearRegression)
explicitly represents evolving coefficients as latent random walks. These are design
references; the ordering above is my assessment of this repository and your goals.

Rolling evaluation should measure the intended forecast task. A generic pointwise
log-likelihood/ArviZ export does not by itself establish future forecasting accuracy.
Use historical cutoffs and preserve the information available at each cutoff; see
[rolling-origin evaluation](https://otexts.com/fpp3/tscv.html) and
[Bayesian leave-future-out validation](https://mc-stan.org/loo/articles/loo2-lfo.html).
CRPS evaluates the predictive distribution, while coverage and width help diagnose why
a model scores poorly; see [distributional forecast evaluation](https://otexts.com/fpp3/distaccuracy.html).

After those foundations, promising extensions are shared-factor multivariate models,
stochastic volatility, changepoints/regime switches, and calendar/cohort-dependent runoff.
The current runoff model is count-based with shared stationary lag probabilities;
continuous amounts need a severity/compositional model with explicit handling of zeros,
partially observed cohorts, and the tail. For AR models, report the fraction of unstable
coefficient draws and offer a separately specified stationarity-aware prior/kernel.
Filtering out explosive draws silently would change the fitted posterior.

Useful workflow additions alongside new models: optional historical state draws and
decompositions, prior predictive simulation for specialized models, dated forecast
coordinates, named future feature validation, scenario draws for uncertain future
regressors, versioned save/load, and an update API. Distinguish filtering at fixed
parameters from updating the parameter posterior when implementing online learning.
Named future features can catch swapped columns that shape-only validation cannot.

**Suggested sequence of reviewable releases**

| Release slice | Scope | Evidence to require |
|---|---|---|
| Correctness patch, e.g. 0.11.1 | Fixed-process covariance forecast bug; predictive clipping; core config checks; output naming; score normalization and stale docs | Reproductions above converted into targeted regression tests; conditional forecast moments; predictive distribution checks |
| Modeling foundation, e.g. 0.12 | Prediction on new bindings, expression arithmetic, explicit dimensions/indexing, shared result protocol, core observation simulation, binding modularization | Nonlinear scalar model, grouped regression, and prediction on a changed row count against independent references |
| Forecast composition, e.g. 0.13 | Gaussian component builder, dynamic regression, named multiple seasonality, historical-state option, reusable rolling evaluation | Equivalence with existing presets; analytic covariance checks; simulated recovery and horizon-specific calibration |
| Robust and pooled forecasting, e.g. 0.14 | Student-t observations and hierarchical dynamic regressions, followed by count/hurdle extensions | Independent target checks, simulation-based calibration, sparse-series recovery, calibrated aggregate and tail forecasts |

These are proposed scopes, not release commitments. Extend validation from a few
successful synthetic recoveries to repeated simulation-based calibration and realistic
misspecification cases. The retained [batch benchmark](https://github.com/tbosier/rustmc/blob/7e471842607a4391232cb10e7e50c83a12a5979e/benchmarks/results/2026-09-08-forecast-batch.md)
already reports that its short-chain schedule fails convergence criteria; require
acceptable diagnostics before presenting useful-inference throughput improvements.

**Validation performed for this review**

- `cargo test --workspace --release --offline`: 132 unit tests plus 12 recovery tests passed.
- `.venv/bin/python -m pytest -q`: 206 passed, 5 skipped, 1 network test deselected.
- `cargo clippy --workspace --all-targets --offline -- -D warnings`: passed.
- `cargo fmt --all -- --check`: passed.
- Python extension reports 0.11.0. Targeted live probes established expression limits,
  lack of new-data prediction, predictive clipping, duplicate-name overwrites,
  position-dependent generic batch seeds, missing generic batch methods, and WIS scaling.
- A separate Rust probe package at `/tmp/rustmc-review-20260909` reproduced zero-chain
  acceptance, zero-step HMC acceptance, and omitted fixed process innovations. It uses
  this checkout as its core dependency and was built offline in release mode.

Existing test success does not cover the newly reproduced cases. No claim of a complete
proof of sampler correctness, universal calibration, or representative performance is
made by this review.
