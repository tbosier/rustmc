# Changelog

All notable changes to rustmc are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases use semantic
versioning while the public API is stabilized.

## [Unreleased]

### Performance

- **Design matrices up to 64 columns are multiplied with fused multiply-add loops
  instead of faer.** faer's general `matmul` is slow on a narrow row-major product. A
  gradient of a 500-row, 8-column normal regression took 9.7 µs and takes 4.5 µs now;
  at 2 to 64 columns the loops are 1.1 to 2.5 times faster across the normal,
  Bernoulli-logit and Poisson likelihoods, at about 96 columns the two are even, and
  beyond that faer is faster, so wider designs keep it
  (`benchmarks/results/2026-10-24-gradient-cost.txt`, from
  `rust_core/examples/gradient_cost.rs`). The repeated benchmark's 30 fixed-size
  regressions fit in 1.53 s against 3.11 s, now ahead of nutpie's 1.93 s on every
  dataset, and its 30 ragged ones in 3.36 s against 6.81 s, where nutpie takes 2.85 s. The loops run under `pulp`'s AVX2 and FMA
  dispatch on x86-64 and natively on aarch64, and fall back to faer on x86-64 CPUs
  without FMA. **Seeded draws change** for graph models with a design matrix of up to
  64 columns, since the products sum in a different order. At extreme scales the order
  also decides whether a sum that cancels overflows first, so a log density there can be
  NaN where it was finite, or finite where it was NaN.

### Tests

- `default_metric_keeps_the_dense_benefit_for_correlated_coefficients` checked one seed
  at 50 dimensions, where whether the auto rule keeps the dense benefit turns on
  rounding, and the new products' rounding changed that seed's outcome. It now checks
  10 datasets, by count and by total leapfrog steps against dense's. The ignored
  `correlated_coefficients_study` prints the numbers for any range of datasets; on 60,
  auto kept the benefit on 52 before this change and 51 after.
- A gradient whose sum is representable must stay finite when a product along the way
  overflows, which the fused accumulation guarantees and separate multiply-and-add
  did not.

## [0.15.0] - 2026-10-19

This release changes how the default `metric="auto"` chooses a dense metric, gives
every fit and forecast the same names and shapes for its draws, records why each NUTS
tree stopped, and adds a repeated-inference benchmark.

### Breaking changes

- **Seeded draws of vector parameters can change** under the default `metric="auto"`:
  wherever the new rule below chooses differently from 0.14.0's in some warmup window,
  and, by rounding only, wherever a window's eligibility for a dense estimate changed,
  since a block that may become dense computes its diagonal estimate by a different
  sequence of operations. A model with both scalar and vector parameters changes all
  its draws, since its trajectories are joint. Models without a vector parameter, and
  fits with `metric="diag"` or `metric="dense"`, draw as before.
- `BayesianARFit.get_samples()` and `BayesianHierarchicalMeanFit.get_samples()`
  return draws with the chain and draw axes merged, chains concatenated, as every
  other fit's `get_samples()` does: `coefficient` is `(chain * draw, p + 1)` instead
  of `(chain, draw, p + 1)`. The `(chain, draw, ...)` arrays are now
  `get_samples_2d()`. Code that averaged with `mean(axis=(0, 1))` gets a scalar
  instead of an error, so check those calls.
- Rust struct literals of these types need the new fields:
  `hmc::TransitionStats::termination`,
  `diagnostics::ChainTransitionDiagnostics::max_tree_depth_hits` and
  `diagnostics::TransitionDiagnosticsReport::total_max_tree_depth_hits`.
- A saved fit is written as version 4 of the `rustmc.graph-fit` artifact, which
  0.14.0 and earlier cannot load. Versions 1 to 3 still load, and a NUTS fit loaded
  from version 3 is saved as version 3.

### Added

- Each NUTS transition records why its tree stopped
  (`TransitionStats::termination`, a `TreeTermination` of `UTurn`, `Divergence` or
  `MaxTreeDepth`). Recording it reads the tree builder's existing branches and draws
  nothing, so seeded draws do not change. `transition_diagnostics()` reports
  `max_tree_depth_hits` per chain and `total_max_tree_depth_hits`, the draw
  transitions whose tree stopped at the depth limit rather than turning or
  diverging in its last doubling, and `to_arviz()` exports the per-draw flag as
  `reached_max_treedepth`. The count can be far below `tree_depth_saturation`: on
  the informative eight-schools test data with seed 1 and `max_tree_depth=3`, 3648
  of 4000 draw transitions were saturated and none hit the limit. The
  `tree_depth_saturation` warning is unchanged. Version 4 artifacts store the
  reasons and refuse ones that disagree with their trees.

- `get_samples()` on `BayesianLocalLevelFit`, `BayesianLocalLinearTrendFit`,
  `BayesianSeasonalLocalLevelFit`, `BayesianHurdleLogNormalFit` and
  `BayesianRegressionFit`, which had only `get_samples_2d()`, and `get_samples_2d()`
  on `BayesianARFit` and `BayesianHierarchicalMeanFit`. Every fit class that had one
  of the two now has both.
- `mean_samples` on every Bayesian forecast: the draws of the expected observation
  given each parameter and state draw, `(chain, draw, step)`, or with a series axis
  before `step` for the hierarchical mean (`(chain, draw, program, step)`) and the
  dynamic GLM (`(chain, draw, group, step)`, as before). It is new on the local level
  (the state), the local linear trend (the level), the seasonal local level (level
  plus seasonal effect), the hierarchical mean (the program mean) and the AR
  forecast (the conditional mean). `ForecastDraws.from_result` reads it from every
  forecast instead of mapping each class.
- `StructuralForecast.state_samples`, `component_samples` and
  `cumulative_observation_samples`, so the structural forecast uses the
  `*_samples` names of the other forecasts.
- `benchmarks/repeated.py`, a repeated-inference benchmark: Gaussian linear
  regressions, fixed-size or ragged, fitted through one model per engine in rustmc
  (one fit per dataset, and one `sample_batch`), nutpie, NumPyro (parallel and
  vectorized chains) and PyMC. It times setup, each fit and each prediction, records
  memory, gates every fit against its exact posterior, including the draws' joint
  covariance and the predictions, keeps failed and erroring fits with the time they
  took, and counts only passing fits' ESS. In the recorded run of 30 datasets every
  fit passed. nutpie fitted each dataset faster than rustmc (1.94 s against 3.11 s for
  30 datasets of 500 rows), and its compilation of about 8 s left rustmc ahead over 30
  datasets. See benchmarks/README.md.
- Rust: `MassMatrixAccumulator::for_warmup_window`, which takes whether the window is
  the last one, since `auto` asks a dense estimate to win by more there.
  `for_window` treats its window as the last.
- `benchmarks/auto_metric_targets.py`, `auto_metric_noise.py`,
  `compare_metric_records.py` and options for `curved_metric_choice.py`, which measure
  the `auto` metric's choices, with saved records that name the build they ran on.

### Changed

- **`metric="auto"` decides by cross-validation.** Each half of a warmup window
  estimates a diagonal and a dense metric, each estimate is scored by the condition
  number it leaves the other half's sample covariance, and the dense metric is used when
  the geometric mean of its two scores is lower: by any amount in the early windows and
  by a factor of 2 in the last window, whose metric the draws are sampled with. The
  whole-window diagonal and dense metrics that would be installed, regularization
  included, are then scored on each half, and the block stays diagonal unless the
  dense one still conditions them better on average; that catches a window whose
  halves drifted apart, where the whole window's covariance fits neither. Each
  half must hold more draws than the vector has elements, so at the default
  `warmup=500` vectors of 100 or more elements stay diagonal (0.14.0's rule allowed up to
  100). 0.14.0's rule needed the window's effective sample size, and its estimate from
  the average autocorrelation left the 50-dimensional 0.95 correlated exactness target
  diagonal. There `auto` now spends 36.4 to 38.5 leapfrog steps per unit of the
  smallest bulk ESS at the gate seeds, against 317 to 683, and forced `dense` 25.2 to
  26.7. Its gate fits' smallest bulk ESS rose from 606 to 4148 at seed 2211, and in a
  2000-seed study the gate's false alarms on it fell from 31 to 15, none of them now
  from the ESS floor. On nine targets in `benchmarks/auto_metric_targets.py` (ten
  seeds, 1000 warmup) the median paired cost ratio against 0.14.0 was 0.82 to 1.02
  on the regressions and isotropic targets; the 20-coefficient correlated regression
  at 500 warmup was the clearly worse case, 10% dearer. On a curved two-dimensional
  target over 40 seeds the median smallest bulk ESS rose from 86.7 to 123.8 and
  divergences fell from 451 to 348, but single fits moved both ways, one falling from
  226.3 to 24.5. In simulated warmup windows the rule at the last window's margin
  stayed diagonal on uncorrelated noise, including a 40-dimensional case where
  0.14.0's chose dense in 30 of 30. It does not protect a direction the chain has
  hardly explored: in two dimensions with identity covariance and one rotated
  coordinate mixing at 0.99, it chose dense in 17 of 30 windows where 0.14.0's chose
  none. At the early windows' margin, short uncorrelated windows went dense in up to
  28 of 30. The
  measurements, records and commands are in `docs/statistical-validation.md`.

### Deprecated

- `BayesianARForecast.conditional_mean_samples`, now `mean_samples`, and
  `StructuralForecast.state_paths`, `component_paths`, `mean_paths`,
  `observation_paths` and `cumulative_observation_paths`, now the matching
  `*_samples`. Each returns the same array as before with a `DeprecationWarning`.

## [0.14.0] - 2026-10-18

This release collects a repository-wide cleanup review; the first item three judges
chose from seven experts' proposals for where rustmc goes next: test the sampler at
the settings users run it with, and tell a user fitting many models which fits to
distrust; a forecasting toolbox of intervention designs, decisions, calibration
diagnostics and reconciliation, with a new local-linear-trend sampler; and streaming
batches. It is grouped by piece of work, newest first.

The tier-1 work does not change seeded draws. The draws of eight seeded fits (graph
NUTS with a diagonal and a dense metric, graph HMC, a `ModelBuilder` spec through
`rustmc.sample`, local level, local linear trend with and without `exog`, and the
conjugate autoregression) hash identically before and after it. The new
local-linear-trend sampler changes the seeded draws of that model without `exog`,
and the cleanup changes seeded draws only where its entries say so.

### Breaking changes

- Rust struct literals of these types need the new fields:
  `hmc::TransitionStats::energy`, `sampler::SampleResult::max_tree_depth`,
  `diagnostics::ChainTransitionDiagnostics::{e_bfmi, tree_depth_saturation}` and
  `diagnostics::TransitionDiagnosticsReport::{max_tree_depth_limit,
  total_tree_depth_saturation}`.
- A saved fit is written as version 3 of the `rustmc.graph-fit` artifact, which
  releases before 0.14.0 cannot load. Versions 1 and 2 still load, and a fit loaded
  from one, which has no energies to store, is saved as version 2.
- The Rust API removals listed under the cleanup review below.
- Python: a `fit_batch` whose cells are all `BayesianAR` now raises `ValueError` for
  any explicit `warmup` or `thin`, including `warmup=500` and `thin=1`, which 0.13.0
  accepted and ignored. Omit them, or pass `None`.

### Streaming batches

#### Added

- `CompiledModel.sample_iter(datasets, ...)` fits a lazy stream of `(stable_id, data)`
  pairs one chunk at a time and yields a `BatchItem` for each dataset that is not
  skipped as completed, in input order, with its ID, fit (or `None` under
  `retention="summary"`), diagnostics, error and fit warnings. The previous chunk's
  data, initial positions and fits are released before the next chunk is read, apart
  from an `init` mapping, which is held whole. Datasets and initial positions are
  held as given until their chunk is sampled, so a source must not change or reuse
  their buffers before then. Duplicate IDs are detected through a SQLite ledger in
  the temporary directory rather than a set in memory. `completed_ids` skips jobs
  finished in an earlier run, and IDs with the base seed reproduce the draws of
  `sample_batch` across chunk sizes and input order, given an `init` that depends
  only on the ID and data. A restart reruns jobs from their seeds; it does not
  resume partially sampled chains. Each chunk is sampled before any of its items is
  yielded, so an exception mid-chunk means rerunning that chunk's unyielded jobs.
  See `docs/streaming-batches.md` and `examples/stream_calibration.py`.

### Forecasting toolbox

#### Added

- `TimeDesign` and `DesignTerm`, in Python and as `rustmc_core::time_design`, build
  named regression columns for a training window and the forecast horizon after it:
  level shifts (`step`), one-period effects (`pulse`), slope changes (`ramp`),
  temporary effects with a fixed geometric decay rate (`decay`), Fourier terms,
  piecewise-linear trend changepoints at given positions or on a Prophet-style grid
  over the first `fraction` of a window, and named columns the caller supplies.
  `future(steps)` evaluates the same columns at the positions after the training
  window, so the generated forecast columns continue the training columns without a
  separate start position; values the caller supplies are only checked for length
  and finiteness. The columns work as
  `exog` for the Bayesian regression forecasters, `StructuralModel` and the dynamic
  GLMs. Changepoint coefficients get the models' Gaussian priors, which shrink like
  ridge regression; there is no sparse prior and no break detection. See
  `docs/interventions.md` and `examples/interventions_forecast.py`.
- Decisions from predictive draws. `rustmc.decide(draws, loss)` returns the decision
  with the lowest expected loss, estimated from the draws, and
  `rustmc.expected_loss(draws, decision, loss)` the average loss of a given one;
  `ForecastDraws.optimal_decision` and `ForecastDraws.expected_loss` do the same for a
  forecast. Built-in losses are `AsymmetricLinearLoss(under=, over=)` (pinball; its
  optimum is the `under / (under + over)` quantile by the rule behind every interval,
  which can sit up to one gap between draws from the exact minimiser of the average),
  `SquaredLoss` (the mean), `AbsoluteLoss` (the median) and `PiecewiseLinearLoss`; a
  Python function of the decision and each draw also works. Losses without a known
  optimum are minimised over candidate decisions only, up to 201 points across the
  draws by default. With `(chain, draw, horizon)` draws, `steps=` decides on the total
  over those steps within each path. The result reports the expected-loss curve, a
  Monte Carlo standard error of the expected loss from the split-chain effective
  sample size, and a batch-means standard error of the decision from contiguous
  blocks of each chain. The latter is `nan` for grid searches, where batch means
  understated the spread of the decision threefold on a coarse grid. In repeated
  simulation the first matched the observed spread of the estimate to within 7% for
  independent and AR(1) φ = 0.8 draws (300 and 400 replicates, in
  `tests/test_decision.py` and the `rustmc_core::decision` tests). The second ran 3%
  to 7% below the observed spread of the decision in 3,000-replicate runs of the
  pinball optimum (the ignored test
  `decision_mcse_calibration_over_3000_replicates`); the default tests hold it to
  within 20%. The math is in `rustmc_core::decision`. See `docs/decisions.md` and
  `examples/reserve_decision.py`.
- Calibration diagnostics for forecasts and rolling backtests. `pit` gives PIT values
  from predictive draws, with ties broken at random by default, so that the values
  for a correct count or hurdle forecast are close to uniform rather than piled on a
  few points; `pit_histogram` counts them in equal bins. `calibration_table` gives
  empirical coverage of central intervals at 19 levels from 5% to 95%, a 95% Wilson
  interval for each, and the mean interval width, per horizon step or pooled.
  `point_accuracy` gives MAE, RMSE, WAPE and MASE of the predictive mean or median.
  `BacktestResult` gains `pit()`, `pit_histogram()`, `calibration()`, `accuracy()`
  and `baseline()`, and `compare_backtests` reports accuracy and calibration of
  several results side by side over the origins they all completed. The statistics
  are computed in the new `rustmc_core::forecast_evaluation`. Limitations, stated in
  the documentation where each is described: PIT values from overlapping folds and
  from the steps of one path are dependent, so no uniformity p-value is provided and
  the Wilson intervals, which assume independence, are typically too narrow for
  backtests; and interval coverage of a count forecast exceeds the nominal level even
  when the forecast is right. See `examples/forecast_calibration.py`.
- **Forecast reconciliation.** `rustmc.reconcile(draws, hierarchy, method=...)` makes
  the forecasts of a hierarchy add up draw by draw, keeping chain, draw and step, and
  returns a `ReconciledForecast` with the draws, node names, means, quantiles and
  intervals. `rustmc.Hierarchy` is built from an explicit 0/1 summing matrix, which
  may describe a grouped structure, or from each bottom series' labels, which gives
  a nested hierarchy with a documented node order and path names such as
  `BU1/IDN3/cust17`. The methods are bottom-up, top-down by average historical
  proportions or by forecast proportions, OLS, WLS with structural or variance
  scaling, and MinT with Schäfer–Strimmer shrinkage of one-step residuals.
  `reconcile` accepts `(chain, draw, node, step)` arrays, per-node dicts,
  `ForecastDraws`, `ForecastBatchForecast` and `BayesianHierarchicalForecast`. The
  arithmetic is in `rustmc_core::reconciliation`. On a simulated hierarchy with
  correlated errors, over 1,000 seeded replications, MinT's mean squared error was
  0.38 of the incoherent base forecasts' and its CRPS 0.66; WLS with structural
  scaling was worse than the base there (1.12 on both). Draws must be joint draws of
  all nodes: draws generated independently per node still add up, but their
  intervals ignore the correlation between node errors. See
  `docs/reconciliation.md` and `examples/forecast_reconciliation.py`.
- `ForecastDraws.from_result` labels the series of a `ReconciledForecast` with its
  node names.

#### Changed

- `backtest` keeps each fold's baseline draws in the new
  `BacktestFold.baseline_samples` and a read-only copy of the observations in the new
  `BacktestResult.observations`, so a backtest also holds 1,000 baseline paths per
  fold in memory. Both fields default to `None`, so existing constructor calls still work.
  Scores and `summary()` are unchanged.
- **The local-linear-trend sampler integrates the states out, and its seeded
  output changes.** `BayesianLocalLinearTrend.fit` without `exog`, and trend cells
  of `fit_batch` without `exog`, used to draw the state path by FFBS and then each
  variance given the states. That mixed slowly. On simulated series in five regimes
  and two lengths, fitted with 4 chains, 300 warmup and 500 draws, the level
  variance's worst R-hat in each setting was 1.06 to 2.06 and its smallest bulk ESS
  6 to 58 out of 2000 draws. Each variance now takes one slice-sampling step on its
  conditional given the other two, with the likelihood computed exactly by a Kalman
  filter, and retained draws sample the terminal state from its filtered
  distribution. On the same series no variance's R-hat exceeded 1.008, and the
  smallest bulk ESS in each setting, over the three variances, was 388 to 1538.
  The target posterior is the same. In the benchmark, on a short series, both
  samplers' means lie within 1.8 Monte Carlo standard errors of means computed by
  quadrature, and a unit test holds the new sampler to within 4 of them.
  A fit took 2.0 to 2.3 times as long in the recorded run, and the level variance's
  bulk ESS per second rose 3.7 to 96 times. In the low signal-to-noise regime, where
  the old sampler already mixed the observation variance, that variance's ESS per
  second fell to 0.57 to 0.68 of its old value. An interweaving (ASIS) update added
  to the old sampler was tried first, as a prototype that was not kept. On the
  twelve series of the new mixing test it raised the level variance's bulk ESS from
  6–15 only to 13–181. The other figures are from `benchmarks/llt_variance_mixing.py`,
  with raw output in `benchmarks/results/2026-09-27-llt-variance-mixing*`.
  `sampler_stats["sampler"]` for these fits now reads
  `"Gibbs over variances with states integrated out (slice steps)"`.
- The trend fitted with `exog`, `StructuralModel` trends, the seasonal model and the
  local level still draw their variances given sampled states. The same benchmark
  shows them mixing slowly; see ROADMAP.md.
- `BayesianLocalLinearTrend` fits without `exog` carry no `known_limitation` record
  (see fit warnings below), so `flagged()` lists their cells only for a failed check.
  On `benchmarks/state_variance_mixing.py` the level and slope variances' worst R-hat
  is 1.001 to 1.008 in all ten settings, against 1.051 to 2.056 with the states
  sampled. A trend fitted with `exog` carries the record. Fit warnings name the
  sampler as `sampler_stats["sampler"]` does.
- The new sampler's Kalman filter carries each state covariance as the level's
  variance, the slope's regression on the level and the slope's variance given the
  level (an `L D L'` factorization) instead of as the covariance matrix. With an
  initial covariance many orders of magnitude above the variances, forming the
  covariances lost the slope variance to cancellation: in a first version that
  formed them, with an initial covariance of diag(1.6e16, 4e15) and five
  observations, the log likelihood the slice sampler targets was off by 0.16 (the
  likelihood by 15%), and the slope variance after the second observation came out
  as 1.0 instead of 0.606. In the new form an observation
  only rescales the level's variance, and the predicted level variance and slope
  variance given the level are sums of non-negative terms. The initial slope variance
  given the level is the determinant over the level variance, with the determinant
  from Kahan's fused multiply-add algorithm on the matrix scaled by a power of two, and
  products whose intermediate values would leave the range of normal doubles are
  combined as logarithms. Unit tests hold the log likelihood to within 1e-12 of its
  value against the same recursion in exact rational arithmetic in twelve cases, and
  in ten of them the terminal filtered means and covariance as well: that one, a
  correlated diffuse prior with missing observations, a level variance 32 orders of
  magnitude below a correlated slope's, state variances 39 orders of magnitude above
  the observation variance, four at scales from 1e-200 to 1e300, and two initial
  covariances within rounding of singular. The sampler's check that the initial
  covariance is positive definite uses the same determinant; the Python constructor
  and the trend fitted with `exog` keep their own checks. At ordinary scales the log
  likelihood agrees with the general state-space filter's to within 1e-10 of its
  magnitude.

### Fit warnings and batch triage

#### Added

- `warnings()` on `FitResult`, on each `BatchResult` of a `BatchFit`, on every
  forecasting fit, and on `ForecastBatchFit` (a dict by cell ID, `None` for a failed
  cell). Each record has a
  `code`, a `status` of `"fail"` or `"unavailable"`, the `parameter` or `chain` it is
  about, the `value`, the `threshold` and a one-line `message` that states the
  threshold. The checks are R-hat above 1.01, bulk or tail ESS below 400, any
  divergence, a chain's E-BFMI below 0.3, any tree that reached `max_tree_depth`,
  and diagnostics that could not be computed. A check a sampler has no statistic for
  (divergences, E-BFMI and tree depth for every forecasting fit) is reported once as
  `"unavailable"`, never as a pass. The rule is implemented once, in
  `rustmc_core::diagnostics::fit_warnings`, and every binding calls it. The codes,
  thresholds and their sources are in the fit warnings section of the reference.
- `flagged(ignore=None)` on `BatchFit` and `ForecastBatchFit`: the IDs of
  successful cells with at least one failed check, in batch order. Failed cells are
  in `errors`. An unknown code in `ignore` raises `ValueError`.
- A quantity whose draws do not vary fails `diagnostics_unavailable` unless the model
  declares it can hold it constant: a fixed `StructuralModel` variance, a structural
  or exog-regression terminal state that no innovation or initial variance can reach,
  a dynamic GLM's terminal states when both process scales are 0, or a runoff
  `ultimate`. So a stuck chain, or draws that collapsed below floating-point
  resolution, flag a forecasting fit as they flag an HMC or NUTS fit.
  `SamplerTelemetry::NonHamiltonian` takes the declared names as `constant`.
- A `known_limitation` record with status `"fail"` on every fit of a sampler that
  mixes its state variances slowly: `BayesianLocalLinearTrend` with `exog`,
  `BayesianSeasonalLocalLevel` with or without it, and a `StructuralModel`
  trend component in the configuration measured: both state variances and the
  observation variance inferred, Gaussian observations, and damping from 0.9 to 1.
  `flagged()` lists those cells until `ignore=["known_limitation"]` leaves the record
  out. The record goes on a sampler whose worst R-hat was above 1.01 in all ten
  settings of the new `benchmarks/state_variance_mixing.py`; each record states its
  sampler's range. The local level was above 1.01 in seven of its eight distinct
  settings and carries no record. The reference's slow state-variance mixing section
  has the table and the command.
- Each HMC and NUTS transition records the Hamiltonian of the state it ended in
  (`TransitionStats::energy`). It is read off values the sampler already computed,
  with no new evaluation and no RNG draw.
- `transition_diagnostics()` reports per-chain E-BFMI, computed as ArviZ's `bfmi`
  computes it, and per-chain and total tree-depth saturation. `to_arviz()` exports
  `energy` and `tree_depth` in `sample_stats`, so `az.bfmi(fit.to_arviz())` agrees
  with them.
- Rust: `diagnostics::{fit_warnings, has_failure, e_bfmi, FitWarning, WarningCode,
  WarningStatus, SamplerTelemetry, R_HAT_THRESHOLD, ESS_THRESHOLD,
  E_BFMI_THRESHOLD, compute_transition_diagnostics_with_max_tree_depth}`,
  `SampleResult::warnings`, `warnings()` on the local-linear-trend, seasonal
  local-level, regression, structural and dynamic GLM posteriors,
  `RegressionConfig::is_local_linear_trend` and `structural::Component::is_trend`.

#### Changed

- Loading a version-3 artifact also refuses telemetry no sampler could have
  produced: a tree depth of 0, a NUTS tree whose leapfrog step count does not fit
  its depth (between 2^(d-1) and 2^d - 1 steps for depth d), a draw energy below the
  potential energy at its draw, and a tree-depth limit that disagrees with the
  recorded depths.

### Exactness gates at the shipped defaults

#### Added

- `rust_core/tests/exactness.rs` (it began in the cleanup as
  `analytic_posteriors.rs`): NUTS at the shipped defaults must reproduce exact
  posterior moments within 4 Monte Carlo standard errors per coordinate, plus a
  squared-norm check, in six cases: a 100-dimensional independent normal through
  both `sample_target` and the graph path, a 50-dimensional normal with correlation
  0.8 through `sample_target`, one with correlation 0.95 through both, and a
  conjugate regression through the graph path.
- The test-only cargo feature `fault-injection` and the doc-hidden
  `rustmc_core::fault_injection` module, off by default. CI checks that the exactness
  gate fails under a biased multinomial merge and under a step size 30% too large
  with no Metropolis correction.

#### Changed

- The Rust recovery suite runs at the shipped `target_accept` of 0.80 instead of
  0.999. `RUSTMC_RECOVERY_TARGET_ACCEPT` and `RUSTMC_RECOVERY_SEED_OFFSET` rerun it
  at other settings.
- The statistical-validation page documents the exactness gate's false-alarm rate,
  that the default `metric="auto"` does not choose a dense metric on an
  ill-conditioned target, and static HMC's resonance at 15 leapfrog steps.

### Cleanup review (2026-10-14)

Eight reviewers read the codebase for dead code, duplicated logic, unsupported
claims and Rust practice, and a second reviewer checked each finding against the
code before it was acted on. The fixes went in as five workstreams (sampler engine;
autodiff, graph and numerics; forecasting samplers; Python bindings; tests, CI and
docs), and each workstream was reviewed again before merging. **Seeded draws are
unchanged** except where an entry below says otherwise.

#### Changed

- **HMC rejects a trajectory that leaves the support.** A leapfrog step that lands
  where the log density is not finite now ends the trajectory as a divergence: the
  proposal is rejected, `energy_error` is +inf, `accept_prob` is 0 and
  `num_leapfrog_steps` counts the steps taken. Before, only the endpoint was judged,
  so a trajectory could pass through such a point and still be accepted. Draws
  change only for chains where this happened. NUTS already ended the tree at such a
  leaf; that is now tested.
- **Custom targets treat a non-finite density as outside the support.** A NaN or
  +inf density, or a non-finite or unwritten gradient, at a point reached during
  sampling (including the initial step-size search) is handled like `-inf`: a NUTS
  leaf there diverges and an HMC trajectory that reaches it is rejected. Before, it
  failed the whole fit. An `Err` from the target still ends the fit, and an explicit
  `init` must still evaluate finitely.
- Transition telemetry (`mean_energy_error`, `max_abs_energy_error`,
  `mean_step_size`) now describes post-warmup transitions only, so `mean_step_size`
  is the adapted step size. Before, early-warmup blow-ups were averaged in. Each
  chain's `max_abs_energy_error` is now its own; it used to be the running maximum
  over that chain and every chain before it.
- Dynamic GLM fits skip elliptical-slice updates on blocks whose prior scale is zero,
  including the default `shared_process_sd=0`. The posterior is the same, but seeded
  draws, forecasts and `likelihood_evaluations` change for those configurations.
- Dynamic GLM forecasts sample Poisson rates up to 2^53 − 1 instead of failing
  above 1e15.
- Forecasting-sampler diagnostics (R-hat, bulk and tail ESS, MCSE) are reported from
  four retained draws per chain, the rule HMC and NUTS fits already used, instead of
  six. `sampler_stats['diagnostics_available']` follows the same rule.
- When several chains fail in a Gibbs, AR, hurdle or forecast-simulation run, the
  error reported is always the first failing chain's in chain order, whatever the
  thread count. All chains run to completion before the error is returned.
- `LinearGaussianStateSpace` stores a process covariance that is asymmetric within
  the symmetry tolerance with its lower triangle mirrored, which is the matrix the
  constructor validated. `filter`, `smooth` and FFBS now succeed on such inputs where
  they could fail with "square-root Kalman recursion failed". Exactly symmetric
  inputs are unchanged.
- A scalar Student-t density whose `nu` or `sigma` is outside its support now has a
  zero gradient, matching its constant −inf density, instead of NaN.
- An unexpected data key suggests the closest required key (within two edits,
  counted in characters) instead of the first one alphabetically.
- Hand-built graphs whose observation nodes or parameter metadata disagree are
  refused with `GraphShapeError` instead of panicking inside a sampler worker.
- `fit_batch` on `BayesianLocalLevel`, `BayesianSeasonalLocalLevel`,
  `BayesianLocalLinearTrend`, `BayesianAR` and `BayesianHurdleLogNormal`: `warmup`
  and `thin` default to `None` (500 and 1 for Gibbs cells). A batch whose cells are
  all `BayesianAR` refuses any explicit `warmup` or `thin`, including `500` and `1`,
  which it used to accept and ignore. A mixed batch whose Gibbs cell has bad data is
  no longer mistaken for an all-AR batch.
- Sampler option checks that depend only on the call's arguments, including the
  draw and transition size limits, run before the model is compiled or bound. In a
  call with two problems, an oversized request is now reported ahead of a model
  error such as `ParameterError`.
- Indexing a forecast batch or reading `.results` no longer copies each cell's
  posterior and paths.
- Every class the native extension defines, except `ModelBuilder`, is frozen
  (immutable). None had setters. Pure-Python classes such as `ForecastSession` are
  unchanged.
- Rust: `SamplerConfig::validate` and `BatchSampleConfig::validate` use the Python
  bindings' messages ("chains must be >= 1" and so on). `hmc::run_chain` and
  `nuts::run_chain` return `Err` for settings they cannot run with instead of running
  or aborting on allocation. `forecast_common::sorted_quantile` returns NaN for an
  empty slice or a probability outside [0, 1] instead of panicking or extrapolating.

#### Performance (no change in output)

- Posterior prediction, log-likelihood and deterministics no longer copy the
  dataset for every 64-draw chunk, and `GraphModel::log_density` no longer copies it
  twice per call. Fits no longer copy the whole posterior while assembling results.
- LogNormal likelihoods cache ln(y) once per binding. Vector Student-t, Gamma and
  half-normal priors compute their constant terms once. Reverse mode reuses forward
  values for exp, sqrt and pow.
- A correlated process covariance is factored once instead of at every forecast
  step.

#### Added

- `prior_sampling::prior_predictive_seeded`, which reproduces Python's
  `sample_prior_predictive` for the same seed from Rust.
- `impl FromStr for MetricKind`, `HmcConfig::validate`, `NutsConfig::validate`,
  `BatchSampleConfig::to_sampler_config`, `SampleResult::mean_accept_rate`,
  `std::error::Error` for `GraphShapeError`, and
  `diagnostics::MIN_DIAGNOSTIC_DRAWS_PER_CHAIN`.
- `forecast_common::{BayesianForecastError, InverseGammaPrior, ForecastQuantile,
  PosteriorPredictiveForecast}`; the `bayesian_forecast::` paths still work.

#### Removed (Rust API; nothing outside `rustmc_core` used these)

- `sampler::batch_sample_graphs` and `sampler::BatchModelResult`. Use
  `sample_batch_cells` with `BatchSeedPolicy::PositionV0`.
- `MassMatrix::from_graph` (use `MassMatrix::identity`) and
  `MassMatrixAccumulator::from_graph`. `MassMatrix::uturn` no longer takes a scratch
  slice.
- `Graph::node_by_name` and `Graph::add_node_as_logp` (use `add_logp_term`).
  `Evaluator::vec_elem` drops its unused `&Graph` argument.
- The `forecast_diagnostics` and `progress` modules are crate-private.
  `rustmc_core::model` re-exports its API by name; the lowering and validation
  helpers it used to export are private or crate-private. So are
  `prior_sampling::{log_gamma, logit_beta, log_half_normal, sample_prior_raw}`.
- The Python extension no longer depends on serde, serde_json, rand or rand_chacha.
- The unused `test` extra and `scripts/verify_wheel_install.py`.

#### Tests, CI and docs

- The fit-artifact fixtures live in `rust_core/tests/fixtures/`, so the published
  crate's tests build from the crate archive. CI builds them from the packaged crate.
- CI builds the abi3 Linux wheel once and tests it on Python 3.9 to 3.14. Editable
  source installs run on 3.11 and 3.12. The separate release-version job is folded
  into the test job.
- The NUTS module header states that rustmc omits the extra U-turn checks across
  merged subtrees that Stan (since 2.21.0) and PyMC apply.
- The statistical-validation page states the coverage band the calibration check
  accepts at 32, 64 and 256 replicates and the `target_accept` each gate runs at.

## [0.13.0] - 2026-09-24

An earlier revision of this file dated 0.13.0 to 2026-09-18, but nothing was tagged
or uploaded then; this is the first published 0.13.0. It collects the changes from
three repository-wide reviews since 0.12.0, grouped by review, newest first.

### Third review (2026-09-24)

This review found the NUTS sampler adapting a dense metric it could not estimate,
chains that were neither independently seeded nor independently started, and a
release that was dated but never published. **Seeded output changes for every model**:
the sampler, every Gibbs kernel, dynamic GLM and runoff now draw from different
streams and start from different points than in 0.12.0.

#### Added

- `metric="auto" | "diag" | "dense"` on `sample`, `CompiledModel.sample`,
  `sample_batch` and `batch_sample`, and a `metric` field on `SamplerConfig`,
  `NutsConfig`, `HmcConfig` and `BatchSampleConfig`. Rust struct literals of those
  configs need the new field or `..Default::default()`.
- `rustmc_core::forecast_common`, holding what the forecasting samplers had each
  copied: the Gibbs schedule and chain driver, forecast path orchestration and
  summaries, the inverse-gamma draw, Cholesky, and the size guard
  `checked_value_count` / `AllocationLimitError` / `MAX_MATERIALIZED_VALUES`.
- `Evaluator::forward`, a forward-only evaluation used by prediction, and
  `sampler::MAX_RETAINED_VALUES`, the bound on draws a graph fit or prediction may
  hold.
- `benchmarks/metric_adaptation.py`, which reproduces the metric and warmup
  measurements quoted below and prints the configuration and seeds it used.
- `HurdleLogNormalPosterior::diagnostics()` and
  `HurdleLogNormalForecast::expected_value_{paths,means,quantiles}`.
- `rust-version = "1.87"` for `rustmc_core`, checked by a CI job.
- Python 3.14 in the install-test matrices and classifiers.
- CI publishes `rustmc_core` to crates.io before the PyPI upload on release tags,
  behind a `crates-io` environment, and runs the ignored prediction-stream test and
  the network packaging test. See `docs/releasing.md` for the one-time setup.
- A fit saved from Python loads in Rust, and the reverse. The `rustmc.graph-fit`
  format, its validation and prediction now live in `rustmc_core::model`:
  `ModelFit::{to_json, from_json, training_data, log_likelihood, deterministics,
  posterior_predictive}` and `GraphModel::sample_batch`. The format is unchanged
  except that keys are written in a fixed order, so one fit always saves to the same
  bytes; older files still load.
- `forecast_common::{sorted_quantile, cumulative_paths}` and public path summaries,
  observation intervals on `state_space::ForecastResult`, hierarchical forecast
  summaries and roll-ups, `GaussianCoefficientPrior::new` and
  `bayesian_regression::fourier_width`, replacing statistics the Python bindings
  computed themselves.
- Hierarchical forecast `interval()` and `state_interval()` default to
  `level=0.95`, like every other forecast.

#### Changed

- **Vector parameters no longer always get a dense metric.** Every vector
  parameter of 2 to 512 elements used to get a dense metric block estimated from one
  warmup window — 200 draws at the default `warmup=500` — which is noisy at 50
  dimensions and singular above the window's length. On an
  isotropic 300-element regression 0.12 took about 500 leapfrog steps per draw; the
  default now takes 15, the same as scalar parameters. The default `"auto"` goes
  dense only when a window's correlation clearly exceeds its own sampling noise, which
  keeps the gain on correlated coefficients: on a 20-coefficient ρ = 0.9 regression it
  takes 11.8 steps per draw against 56.3 for `"diag"`. It considers a dense block
  only when the window has at least two draws per element, so at the default
  `warmup=500` a vector of more than 100 elements stays diagonal however correlated
  it is (0.12 used a dense block there); raise `warmup` or pass `metric="dense"` for
  such a vector. An explicit `"dense"` shrinks
  correlations toward zero when a window has fewer than two draws per element, so it
  stays well conditioned (22.7 steps at 300 isotropic elements). Figures are from
  `benchmarks/metric_adaptation.py`, which prints its configuration and seeds.
- **The warmup schedule is Stan's from 500 iterations, and no metric window is cut
  short.** At `warmup=1000` the final metric came from 100 draws (850–950) and
  discarded the 400-draw estimate; it now comes from draws 450–950. The default
  `warmup=500` was unaffected. Below 500 iterations the schedule deliberately differs
  from Stan: buffers scale down (initial `min(75, 15%)`, terminal `min(50, 10%)` but at
  least 25), every metric update is followed by at least 25 step-size iterations, no
  window after the first is shorter than 25 draws, and below 41 iterations only the
  step size adapts, as Stan does below 150. On a badly scaled regression this keeps
  warmups of 100–155 at about 16 steps per draw; warmups of 40 or fewer, which adapt
  no metric, take about 420, and 41–45 cost 22–78 against about 15 in 0.12. HMC now
  uses the same windowed schedule instead of one 15–90% window.
- **Chains are seeded independently.** The graph sampler seeded chain `c` with
  `seed + c`, so seed 42's second chain was seed 43's first. Every sampler now keys
  chains through `seeding::chain_seed`, which mixes seed, domain and chain index in
  separate rounds so that no fixed offset of the seed reproduces another chain.
- **Chains start from different points.** Without `init`, graph-model chains start
  uniformly on (−2, 2) in unconstrained coordinates, redrawing non-finite starts. The
  local-level, trend, seasonal, regression, structural, hurdle and hierarchical
  Gibbs samplers start each chain from its own overdispersed point instead of the
  prior mode or the data means. Split R-hat assumes dispersed starts.
- Forward-filtering backward-sampling runs one square-root pass instead of two,
  about 25–30% faster at typical state sizes. On ordinary models draws differ in the
  last bits. On diffuse, nearly collinear designs the old means were off by up to
  about 0.1 posterior standard deviation, and the new ones agree with exact
  conditioning. Hurdle uses the scalar version.
- Dynamic GLM block updates evaluate only the groups a block touches: a sweep is
  O(G·T) rather than O(G²·T), with the same target. For a given stream the draws
  match the old algorithm's up to floating-point rounding; they were bit-identical in
  every comparison run. Two measurements on different runs put the speed-up at 13–20× for 50
  groups and 41–90× for 200 groups (T = 100).
- Dynamic GLM forecasts and prior predictions, and runoff fits, run chains in
  parallel with results independent of thread count.
- Effective sample size uses an FFT for the autocovariance beyond the first lags,
  and diagnostics are computed in parallel over parameters.
- Seasonal fits and trend or seasonal regressions need three finite observations;
  structural fits need one per inverse-gamma variance. Structural fits previously
  accepted a series with no finite value.
- Structural fits with `thin > 1` keep the last sweep of each thinning block, like
  every other sampler.
- Runoff errors raise `InferenceError`, a `ValueError` subclass. `fit_runoff`,
  `PaymentTriangle::validate`, `elliptical_slice::update` and
  `observation::{log_density, mean, sample}` return typed errors;
  `elliptical_slice::update` accepts `FnMut`. Messages are unchanged except the
  runoff size refusal, which now uses the shared size-guard wording.
- The documentation site deploys after a release publishes instead of on every push
  to `main`, so it no longer describes behaviour `pip install rustmc` does not have.
- The third-party actions on the release path (`maturin-action`,
  `gh-action-pypi-publish`, `rust-cache`, and `rust-toolchain` where it builds release
  artifacts) are pinned to commit SHAs; GitHub's own `actions/*` still use version
  tags. The PyPI publish job runs in a `pypi` environment, and CI caches Rust builds.
- `scripts/dev_pytest.sh` honours `RUSTMC_VENV` and finds the main checkout's
  virtual environment instead of a hard-coded home directory.
- **Forecasting exceptions follow one rule.** Errors from the local-level,
  seasonal, trend, AR, hurdle, regression, dynamic GLM and structural models now
  raise `InferenceError` where many raised `StateSpaceError`: their priors, `fit`,
  `forecast` and `prior_predict`, `from_json` of their fits, `fit_batch` cells and
  batch forecasts, and the accessors of the results. `StateSpaceError` is kept for
  `LinearGaussianStateSpace` and structural specifications; shared argument checks
  raise plain `ValueError`. All three subclass `ValueError`, so code catching
  `StateSpaceError` from any of the calls above must catch `InferenceError` or
  `ValueError` instead.
- One exact converter now reads `ModelBuilder` data and every forecasting array.
  Forecasting models accept integer arrays and lists as well as float arrays, and a
  bad input is reported by argument name. Several inputs that raised `TypeError`
  from PyO3's conversion now raise `ValueError`: a wrongly dimensioned array for a
  forecasting fit (including a 1-D `y` for a dynamic GLM), a data value with more
  than two dimensions, and a dict as a data value.
- **Object-dtype arrays are refused**, including pandas object columns, which 0.12
  accepted when their elements were numbers. The error says to convert with
  `.astype(float)`. Python integers beyond 64 bits are refused for the same reason.
- `LinearGaussianStateSpace.filter()` and `smooth()` use the same square-root pass
  as FFBS, so the three agree. On ordinary models results change in the last bits;
  `filter()` is somewhat slower (0.73 s against 0.53 s at d = 52, T = 1000).
- Dynamic GLM `prior_predictive` draws its noise from its own seed domain, so its
  paths change for every seed and no longer replay a posterior forecast made with
  the same seed.
- Forecast quantiles and intervals use one rule, defined once in Rust: NumPy's
  `linear` method, which they now match bit for bit on finite draws. Most values
  move in the last digits; near zero the old formula could be off by thousands of
  ulps. Ties are exact, a constant forecast's interval is that constant, and NaN
  draws are refused. `ForecastDraws.interval` and the evaluation scores use the same
  rule.
- Graph-fit `diagnostics()` reports an unavailable value as `None`, as the
  forecasting fits do, instead of NaN.
- `log_likelihood`, `predict`, `posterior_predictive`, `deterministics`,
  `to_arviz`, `sample_prior_predictive`, `to_json` and `from_json` release the GIL,
  and `log_likelihood` no longer builds an evaluator per draw.
- `batch_sample` runs on the same native path as `sample_batch`, with positional
  cell seeds; its errors name the failing `dataset '<i>'`.
  `sample_batch(show_progress=True)` now shows progress instead of ignoring the flag.
- The Python extension's 6,271-line `lib.rs` is split into one module per surface.

#### Fixed

- Effective sample size now follows ArviZ's algorithm: Geyer's final positive term
  was dropped, bulk ESS ranked before splitting, and the tail indicator used `>=`
  rather than `<=`. The old values could be off in either direction, by up to about
  20% for short chains, and were NaN at four or five draws per chain. Two
  differences from ArviZ are deliberate: a constant parameter's ESS is NaN (ArviZ
  reports the draw count), and a single chain gets a split R-hat (ArviZ reports NaN).
- Invalid input reaching the Rust API returned panics instead of errors: an
  out-of-range group index, an empty data vector, a graph node referring to a later
  node, a vector used as a log-density term or a scale, a wrong-length `init`, and
  the raw `nuts::run_chain` / `hmc::run_chain` entry points. The Python API was
  already guarded.
- Requests too large to hold in memory aborted the Python interpreter instead of
  raising: forecasts and draw counts for the local-level, trend, seasonal and AR
  models, `sample()` draws, warmup or chains, prior-predictive `n_samples`,
  prediction `sizes`, `LinearGaussianStateSpace.forecast` horizons, seasonal periods
  and `fourier_design`. They now raise `ValueError` (or the model's error class)
  before allocating. State-space forecast horizons have no fixed cap; the rest are
  bounded by `forecast_common::MAX_MATERIALIZED_VALUES` or the new
  `sampler::MAX_RETAINED_VALUES`.
- `digamma` never returned for arguments at or below −2⁵³ or −∞.
- Normal and log-normal observation terms returned NaN rather than −∞ for a scale
  at or below zero.
- The step-size search could return a non-finite step, repeated its first probe and
  reused one momentum draw for every probe.
- Runoff chain 0 ran on the raw seed.
- The NUTS leapfrog step allocated nine vectors per step, plus nine per
  transition, despite being documented as allocation-free; a counting-allocator test
  now holds it to that.
- An AR forecast overflowing on an explosive draw names the chain, draw and step.
- State simulation errors name the covariance that failed.
- `hierarchical.rs` no longer claims the conjugate Gibbs sampler avoids funnel
  geometry; with weak data it can stick near a group variance of zero.
- Data dictionaries and forecasting inputs silently coerced what they could not
  represent: a scalar became a length-1 vector, booleans (including `True` inside a
  list) became 0/1, the string `'1.5'` became a number, complex values lost their
  imaginary part, a masked array's hidden values were used as data, long doubles
  and integers above 2⁵³ were rounded. These now raise an error naming the input.
- On diffuse, nearly collinear state-space designs, `LinearGaussianStateSpace`'s
  filter, smoother and log likelihood were inaccurate: in one test the log likelihood
  was off by 4.5e-3 and a coefficient mean by 4e-4 posterior standard deviations.
  They now agree with a 60-digit reference to about 1e-11.
- Under `errors="collect"`, a batch cell whose binding or reported draws failed
  aborted the whole `sample_batch` call instead of recording that cell's error.
- A `fit_batch` of only AR models silently ignored `warmup` and `thin`; it now
  refuses non-default values, since exact AR draws have neither.
- Runoff result accessors could panic on inconsistent shapes; they raise instead.

#### Removed

- `sampler::{batch_sample, sample_batch_bound, sample_batch_bound_with_options,
  BoundBatchResult}`, `distributions::Distribution`, `Normal::observed`,
  `MassMatrix::{accumulator, dim}` and `MassMatrixAccumulator::reset`. None had a
  caller outside tests; `batch_sample` also silently dropped part of its input.
  `progress::spawn_progress_thread` and `runoff::known_total_hazard_posterior` are
  no longer public.
- `demo-docs/` (1.3 MB of internal material measured on 0.9.0),
  `docs/forecasting-extension-review.md` (a stale internal review note),
  `scripts/verify_version.sh`, and the example scripts `compare_with_pymc.py`,
  `benchmark_vs_pymc.py`, `benchmark_multivariate.py` and `run_benchmarks.py`, which
  `benchmarks/run.py` supersedes. `batch_many_series.py` moved to
  `benchmarks/comparisons/`.

### Second review (2026-09-19)

This entry closes a second repository-wide review. Two themes dominate: claims —
in documentation, in test names, and in a release gate — that the code did not
support, and predictive draws sharing an RNG stream with the fit that produced them.

#### Added

- `rustmc_core::seeding`, the single definition of the RNG stream-separation
  primitive that eight modules each carried a private copy of — seven named
  `chain_seed` and one `seed_for`, six of them byte-identical and two combining their
  arguments differently.
- `scripts/build_example_docs.py`. Every page under `docs/examples/` is generated
  from the example of the same name and shows that example's real captured output.
  CI fails if a committed page stops matching the code that produces it. It also
  checks that each guide page's code blocks run when read in order, and that the
  output shown on the landing page is what the code above it prints.

#### Changed

- **Seeded predictive draws change.** `posterior_predictive`, `predict`,
  `to_arviz(include_ppc=True)`, `sample_prior_predictive` and
  `rustmc_core::model::ModelFit::predict` now derive their stream from a domain
  constant instead of using the caller's seed directly. Draws remain deterministic
  and reproducible; a given seed produces different values than in 0.12.0. See Fixed.
- **Seeded hurdle fits change.** Consolidating the seven private copies of the
  stream-separation primitive settled on the additive form six of them used; `hurdle`
  combined its arguments with XOR. Its fitting chains therefore draw different streams
  for the same seed from chain one onward, so a seeded hurdle posterior differs from
  0.12.0. Nothing about the model changed.
- The documentation nav is grouped into sections, and `mkdocs.yml` sets
  `strict: true` so an orphan page or a nav entry pointing at a renamed file fails
  the build rather than shipping.
- `benchmarks/README.md` describes the screened quality gate, including the
  `quality_gate.domains` the report now publishes and what a domain does not
  establish.

#### Fixed

- **The benchmark quality gate passed on diagnostics that were not numbers.** Each
  metric was compared with a bare `>` or `<`, which reports false for a NaN, so
  nothing was appended to the failure list and the gate that authorises publishing
  a speed claim reported success. It now fails closed on non-finite, out-of-domain,
  absent, negative and fractional values, and screens each parameter before the
  R-hats are reduced with `max` and the ESS values with `min` — aggregating first
  hid exactly the half of each domain the gate cares about.
- **Predictive draws replayed a fitting chain's stream.** `sampler::run` seeds chain
  `c` as `seed + c`, and every predictive entry point defaulted to the same seed
  `sample()` defaults to, so a default four-chain fit held streams 42..=45 and
  prediction under seed 42 reproduced chain 0 exactly. Measured at four chains,
  1000 warmup and 1000 draws, the standardised predictive residuals under a
  prediction seed equal to the fit seed were not distinguishable from unrelated
  seeds; the exposure is short runs, where a chain's tail and the prediction noise
  overlap over a much smaller sample.
- `validate_shapes()` never counted the binding indices of `Op::BroadcastObservation`
  or `Op::FusedLinearMu`, so a model using either could panic out of `sample()`.
- `KalmanFilterResult.log_likelihood` and `KalmanSmootherResult.log_likelihood` were
  annotated `dict[str, _FloatArray]` and return `float`. The stub check that should
  have caught this was scoped to the classes one branch had reworked.
- **The Bernoulli-logit density and gradient lost their saturated tail.** Both were
  written as a difference of nearly equal numbers, so at `y = 1, eta = 40` each
  returned exactly zero against a true magnitude of `4.2483542552915889e-18`. A
  saturated observation contributed no gradient at all, and a large predictor scale
  multiplies that zero rather than a small number. `observation.rs` already avoided
  this; the graph evaluator, its reference and the shared density helper each had
  their own expression and none of them did.
- **The local-level filter's variance update left the representable range.** Variances
  around `3e-162` were up to 9.8% wrong and silently positive, and a well-scaled
  problem was rejected outright below about `1e-170` and above about `1e155`. Both
  single orderings of `a b / (a + b)` fail, in opposite directions; the update now
  divides by the sum whichever factor is larger, which keeps every intermediate in
  range by an interval argument rather than an empirical bound.
- Forecast mean accessors on all four specialised results no longer report an infinity
  for a forecast whose draws are finite and whose mean is representable. Their values
  shift in the last bits. This is a trade rather than a strict accuracy win: centring
  the draws before summing is better for draws sharing a large offset and worse for
  draws that cancel to near zero, and the sum is still uncompensated, so its error
  grows with the number of draws — 100,000 draws of mostly `0.1` land about 849 ulp of
  the draws' range from the correctly rounded mean. The reason to take the trade is
  that an overflow is a failure and this is a rounding.
- A benchmark config with a non-finite quality threshold was accepted, and every metric
  then compared false against it, so the gate passed with no failures. Screening the
  metrics had closed only one side of that.
- A published claim that a compiled model is "validated and laid out once rather than
  per instrument". Every chain of every fit revalidates its binding and rebuilds its
  evaluator layout; what is shared is the graph structure.
- An unsupported "63x the cost per gradient" figure in the first-review entry, which no
  retained measurement in the repository supports.
- **Recovery tests a prior-only sampler would have passed.** Across the recovery
  suite, the seasonal, trend and forecast tests, the hurdle, regression, structural
  and diagnostics modules, and the Python smoke test, acceptance windows contained
  the prior mean they claimed to beat — in two cases the prior mean was the truth
  exactly, and two funnel tests carried no data at all while asserting recovery.
  Windows now have to clear the prior by at least their own width, and vector claims
  are stated as a fraction of the error a named data-blind estimator would score. The
  margin is asserted at run time — in the recovery suite on every scalar assertion
  through a shared helper, and in the trend, regression and hurdle modules by their
  own guards — so widening a window back onto a prior turns the test red rather than
  passing quietly. With the likelihood terms stripped
  so the sampler draws from the prior alone, all 28 assertions across the 9 positive
  cases now fail; the 3 tests that still pass are the ones documented as geometry
  checks and negative controls rather than recovery claims.
- Documentation claims the code did not support, including committed example output
  that advertised 128 divergent transitions for a model that now has none.
- The opening code block on the regression-and-seasonality guide used four names it
  never defined, so a reader copying the page's first example got a `NameError`; a
  later block on that page did the same. Both now build their own data.
- Every link in `README.md` was relative. README.md is the package's long description,
  so on the PyPI project page all twelve resolved against `pypi.org` and 404'd,
  including every "Start here" entry.

#### Removed

- Four `Op` variants no callable path could reach, three public items with no caller,
  and the `Option` around a batch cell's fit, which could not be `None`.
- Two committed executed notebooks and their rendered image directories, 1.4 MB in
  all, replaced by the generated example pages.
- `docs/repo-review-2026-09-09.local.md` is no longer tracked. Both `.gitignore` and
  the site build already treated it as local scratch, and it reviewed a revision two
  releases back. The copy on disk is untouched.

### First review (2026-09-18)

This part closes a repository-wide correctness review. The headline item is a
regression in 0.12.0 that silently transposed Fortran-ordered design matrices; if
you are on 0.12.0 and pass a 2-D `X` that is not C-contiguous, upgrade.

#### Added


- `rustmc_core::model::GraphModel::sample_prior` and `GraphModel::prior_predictive`,
  so a loaded model artifact can be simulated from Rust. Model-level prior generation
  moved out of the Python binding crate into `rustmc_core::prior_sampling`. The move
  itself does not change any draw; six of the eight prior families are bit-identical to
  0.12.x for a given seed. `Uniform` and `Beta` draws differ in the last bit, because
  the bounded transform they share was corrected in this same release (see Fixed).
- A bare data-key string is accepted anywhere an expression operand is accepted, so
  `beta["group"] * "x"` (random slopes) and `builder.normal_likelihood("obs", "x", ...)`
  work like `beta * "x"` already did. The fused linear-predictor fast path is preserved.
- `rustmc.__all__`, so `from rustmc import *` no longer pulls in the `evaluation` and
  `forecasting` submodules.
- `scripts/run_examples.py`, run in CI: every example documented in
  `examples/README.md` must run inside a time budget, and a script in neither a README
  table nor an excluded section fails the build.

#### Changed


- **Breaking (alpha Rust API):** `nuts::run_chain`, `nuts::run_chain_bound`,
  `hmc::run_chain` and `hmc::run_chain_bound` return `Result<ChainResult, String>` and
  reject discrete latent parameters; they previously bypassed every guard.
- **Breaking (alpha Rust API):** new `graph::Op::BoundedSigmoid` variant, which breaks
  an external exhaustive match on `Op`.
- `FitResult.std()` and `BatchResult.std()` return the sample standard deviation
  (`n - 1`), which is what `summary()` has always reported. The two paths previously
  disagreed by `sqrt(n/(n-1))`, about 0.0125% at 4000 draws. `mean()` moves by under one
  ulp at ordinary scales.
- **Breaking (alpha Rust API):** new `graph::Op::BoundedSigmoid` forward and backward
  helpers, and `ElementwiseOp::adjoints`, which reverse mode now calls instead of
  `derivatives`. `derivatives` remains as the local-derivative API.
- A `potential` or `deterministic` naming a data key is validated when it is declared,
  on the same "only when data is bound" rule the likelihood families use. A builder
  holding part of its data can no longer declare one naming a key that arrives later.
- `examples/fixed_effects_panel_forecast.py` and `examples/large_linear_regression.py`
  were rewritten; they ran for about 57 and 42 minutes and now take 27s and 5s. The
  panel example used 168 dense one-hot indicator columns instead of the library's own
  group indexing, and stacked four nested intercept blocks that were not identified.

#### Fixed


- **Fortran-ordered and otherwise non-C-contiguous 2-D inputs are no longer read as
  row-major.** In 0.12.0 a design matrix in column-major order was reinterpreted
  against its own shape, silently producing a different model and a wrong posterior
  with no error.
- **Structural fits and forecasts no longer share an RNG stream.** `structural.rs` was
  the only fit-then-forecast module without a domain separator, so passing one seed to
  both `fit()` and `forecast()` replayed the fitting draws: each forecast's first
  innovations were the standard normals that built its own terminal state. Measured on
  two fixed-variance levels, step-1 predictive variance was 6.14 against an analytic
  2.75-3.8 depending on the configuration, and the terminal state correlated with its
  first innovation at +0.93. Seeded structural forecast output therefore differs from
  0.12.0 for every seed, not only colliding ones.
- **A bounded prior's density is evaluated at the point it reports as the draw.** The
  logistic transform had three implementations, two of which overflowed for arguments
  below about -709. `Uniform` priors now compile to one fused `BoundedSigmoid` node,
  which also removes the span factor that made the gradient overflow: `Uniform(0, 1e308)`
  at raw -710 gave `-inf` and now gives -19.037138374168897.
- **`tanh` lost its gradient entirely from |x| ~ 19.** `1 - tanh(x)^2` cancels to exactly
  zero once `tanh` rounds to 1, so a saturated `tanh` reported a flat direction where the
  density is not flat. One step before the cancellation the derivative was already 77%
  high. The true slope stays representable to |x| = 372. This one needs no extreme
  scales to reach.
- **A representable gradient is no longer lost to an unrepresentable intermediate.**
  Reverse mode computed each local derivative and multiplied by the upstream adjoint
  afterwards, so `Div`, `Log` and `Pow` could overflow or underflow on their own while
  the composed result was ordinary. `(1/b) * 1e-200` at `b = 1e-200` gave `inf` for a
  true `1e200`.
- **The Bernoulli and Poisson densities now have the support they claim.** The Bernoulli
  log density was finite at `x = 0.5` -- constant over the whole real line at `p = 0.5` --
  and scored the impossible `x = 1, p = 0` as `-27.63` because of a probability clamp.
  Both are `-inf` off the support. The Poisson score also cancelled: `x/lam - 1` was
  twice the correct value at `x = 1, lam = nextafter(1, 0)`.
- **A wide bounded interval keeps its tail.** The fused transform applied the span after
  materialising the sigmoid, so `Uniform(0, 1e308)` returned 0 below raw `-745` where the
  constrained value is `1.04e-16`, and was already 75% high at `-745`. Folding the span
  into the exponent extends full precision 670 units of raw further out.
- **`FitResult.mean()` and `.std()` share one implementation with the summary table.**
  They accumulated naively and overflowed where the summary did not, so the same fit
  reported an infinite standard deviation from one accessor and a finite one from another.
- **A fit artifact's `training` entries are checked against their own namespace.** The
  schema's key set was flattened across observations, vectors and matrices, so a matrix
  supplied under the name of a required vector passed as a known key and was dropped on
  the next save.
- **`a / b` gradients stay representable at extreme denominator scales.** `-a/(b*b)`
  collapsed to zero once `b*b` overflowed and to infinity once it underflowed, so a
  representable derivative was silently replaced.
- **The statistical release gate screens every parameter.** It aggregated with builtin
  `max`/`min`, which drop a NaN that is not first, so a fit whose second or later
  parameter had a NaN R-hat or ESS passed. Diagnostics are now checked per parameter
  against the full range their estimators can produce, a promised reference case that
  did not run fails the run, and the report is written even when a case cannot be built.
- **Posterior-predictive draws keep their chain and draw identity in `to_arviz`.** They
  were flattened into a single fake chain, so a four-chain fit exported posterior
  parameters as `(4, draws, ...)` and predictive draws as `(1, 4*draws, ...)`, leaving
  no way to pair them. `ppc_samples` is now chain-stratified and exports the retained
  draw coordinates.
- Non-finite deterministics are rejected instead of being returned inside an otherwise
  successful `sample_prior_predictive` or `FitResult.deterministics()` result.
- Discrete latent parameters are rejected at the sampler boundary rather than only in
  the Python and artifact layers, including when they reach their density through a
  transform, and including through the raw `nuts`/`hmc` kernels. A discrete prior can
  still be loaded and simulated for prior prediction, which it could not before.
- Artifacts with unknown fields are rejected instead of being silently truncated,
  across every field of every struct and named enum variant reachable from the five
  `from_json` loaders. A graph fit's `training` entries are matched against the model
  schema per namespace, so a matrix supplied under the name of a required vector is
  refused rather than accepted and dropped on the next save.
- An unknown data key in a `potential` or `deterministic` is named, with the available
  keys listed, instead of surfacing as a confusing length mismatch.
- The diagnostics tables size themselves to their contents. Parameter names longer than
  12 characters, which the library's own models emit, shifted every later column; both
  horizontal rules were also the wrong length.
- Every native class reports `__module__ == "rustmc"` instead of `"builtins"`.
- Preserve posterior means, standard deviations and Monte Carlo standard errors
  across parameter units without overflow or arbitrary variance cutoffs.
- Allow smoothing and FFBS for valid deterministic state transitions with singular
  predicted covariance, including zero-innovation AR components.
- Keep zero powers and zero-valued predictors with learned positive exponents from
  introducing invalid gradients into otherwise finite custom-model targets.
- Preserve infinite losses in backtest summaries and stabilize CRPS, WIS and point
  error arithmetic for narrow forecasts with large levels or extreme finite scales.
- Preserve logical NumPy matrix order across contiguous, transposed and strided inputs.
- Evaluate Beta and Uniform priors in unconstrained coordinates without truncating
  rounded sigmoid tails; retain exact sampler positions for prediction and fit artifacts.
  Fit artifact version 2 retains these positions and reads existing version 1 artifacts.
- Keep output-only deterministic expressions from contaminating target gradients, and
  preserve positive-scale support when noncentering hierarchical Normal priors.
- Preserve correlated state uncertainty across measurement scales, reject asymmetric
  covariances consistently in the shared state-space implementation, and avoid
  cancellation of small posterior variances in smoothing.
- Count terminating NUTS expansions in tree depth and recenter HMC step-size adaptation
  after changing the mass matrix.
- Preserve native conditional-mean forecast draws, coordinates and metadata in workflow
  adapters; reject empty potential names before creating an unreadable model artifact.
- Reject overflowing Uniform ranges and nonfinite sampled outputs. (An earlier fix in
  this cycle also preserved Jacobian/potential terms in legacy graph exports; that
  format is removed below, so only the Uniform and output guards remain.)
- Correct Poisson simulation at tiny and large rates across generic observations,
  dynamic count models and payment runoff.
- Preserve Poisson and negative-binomial likelihood curvature at large counts and
  dispersion, sharing stable densities across inference and pointwise diagnostics.
- Preserve Gamma, Exponential and HalfNormal prior tails in unconstrained inference
  and analytic prior draws; avoid scale overflow in Normal and Student-t densities.
- Match pointwise observation likelihoods to the fitted model without arbitrary
  scale or response floors, including very small positive LogNormal observations.
- Apply unit-independent covariance symmetry checks to specialized Gaussian models.

#### Removed


- **Breaking (alpha Rust API):** `rustmc_core::compiled_model` and its re-exports
  (`ArtifactError`, `CompiledModelArtifact`, `CompiledModelRuntime`, `ModelMetadata`,
  `ModelStep`, `NodeRef`, `ParameterBlock`, `SerializableObsFamily`,
  `SerializableParamTransform`). 1,866 lines with no consumer: the Python
  `CompiledModel` API never emitted or accepted this format.
- **Breaking (alpha Rust API):** `rustmc_core::autodiff::{reference, eval_logp, forward,
  grad_logp, Value}`. The allocating reference evaluator is a differential-testing
  oracle that panics on unexpected node shapes; it is now `#[cfg(test)]` and still
  checks the optimised evaluator.
- **Breaking (alpha Rust API):** the `Op::Sub`, `Op::Div`, `Op::Neg`, `Op::Log` and
  `Op::Square` IR variants and their `Graph` builders, which only the deleted legacy
  module constructed. `ElementwiseOp` carries all of them.

## [0.12.0] - 2026-09-09

### Added

- Scalar/elementwise expression arithmetic and transforms, named deterministics,
  potentials, independent observation dimensions, grouped parameter indexing, and
  posterior prediction on new predictor rows without dummy responses.
- Versioned declarative compiled-model artifacts and validated generic fit artifacts
  that preserve posterior draws, training data, telemetry, and seeded prediction.
- Composable structural level, damped trend, harmonic seasonal, stable fixed AR, and
  static/dynamic regression blocks, with joint Gaussian FFBS/Gibbs, optional Student-t
  observation errors, component histories, prior prediction, and persistence.
- Native dynamic Poisson, negative-binomial, hurdle-lognormal, and pooled Gaussian
  models with exposure, group coefficients, latent trajectories and optional shared
  shocks using elliptical slice sampling. These infer coefficients and states with
  explicitly fixed scale/dispersion inputs.
- Rolling-origin backtests, CRPS/WIS/coverage/width scores, seasonal-naive bootstrap
  baselines, labeled forecast draws, named feature checks, scenario mixtures, portable
  forecast archives, and full-refit update sessions.
- A native `LogDensity`/gradient interface reusing HMC/NUTS and explicit unconstrained
  chain initialization. Generic compiled batches use stable cell IDs, bounded pools,
  collected failures, diagnostics and predictive results.
- A mixed Python/native package with maintained API type information.

### Fixed

- Forecast simulation now propagates the full configured process covariance, including
  fixed regression-state innovations; invalid conjugate covariance updates are rejected.
- Native observation simulation no longer clips exponential rates or probabilities.
- Core sampler configuration rejects empty chains/draws, invalid integration controls,
  nonfinite initialization and overflowing allocation counts.
- Duplicate likelihood/deterministic outputs and incompatible named dimensions are
  rejected instead of silently overwriting or conflating results.
- Regression expression compilation preserves parameter identity for names resembling
  internal constant markers; intercepts use explicit constant/parameter variants.
- Demo WIS uses its standard denominator. The separate baseline single-interval metric
  is named `interval_score_95`; retained scores were corrected without rerunning timing.
- Persistence support/shape validation and allocation guards cover new model kernels.

### Changed

- `CompiledModel.sample_batch` defaults to `cell_id_v1` seeds. Use `position_v0` for
  previous positional replay; the legacy global `batch_sample` keeps its old seeds.
- Extracted expression compilation, prediction binding, observation simulation and
  artifact code into focused modules; isolated the legacy Rust compiled-model format
  and reference autodiff while preserving their public import paths.
- Structural Student-t forecasts require `df > 1`, so their conditional means exist.
  Damping/AR/df and dynamic-GLM scales/dispersion remain fixed model inputs.

## [0.11.0] - 2026-09-08

### Added

- Joint Bayesian exogenous regression for local-level, seasonal, and trend forecasts,
  with proper Gaussian coefficient priors and paired coefficient/state/variance draws.
- Time-varying observation rows in fixed Gaussian state-space models and explicit
  future-design validation, including joint and cumulative forecast covariance.
- Native independent forecasting batches with stable cell seeds, ragged histories,
  per-cell models/designs, controlled worker counts, and collected cell errors.
- Sampler-aware parameter diagnostics for forecasting fits, including coefficients
  and retained terminal states; Gibbs telemetry does not invent divergences or
  acceptance rates.
- Fourier calendar designs with explicit phase and Nyquist handling. Stochastic
  seasonal fits now accept short histories with at least two finite observations.
- A hurdle model for nonnegative amounts with static uncertain occurrence
  probability and dynamic lognormal severity under upper-truncated inverse-gamma
  variance priors, including exact zeros and all-zero histories.
- Count-only payment runoff with shared Dirichlet lag probabilities, known or
  inferred ultimate counts, prefix censoring, and an explicit unscheduled tail.
  It does not model continuous currency allocations or couple separate accrual fits.

### Fixed

- Guarded oversized forecasting/runoff allocations and sparse large-count binomial
  draws; strengthened version checks across manifests, dependencies, and the lockfile.

### Packaging

- Prepared synchronized Rust and Python 0.11.0 metadata. Release uploads require
  verification of the built wheel or source archive before artifact publication.

## [0.10.0] - 2026-08-04

### Added

- Added `BayesianHierarchicalMean`, a joint population → group → program Gaussian
  partial-pooling model for ragged program series.
- Added a dedicated conjugate Gibbs kernel that draws the hierarchy's exact full
  conditionals and avoids requiring HMC/NUTS to traverse funnel geometry.
- Added explicit group, program, and observation variance priors plus program/group
  names and ragged time/observation-count metadata.
- Added aligned hierarchical posterior-predictive paths shaped
  `(chain, draw, program, step)` and built-in draw-wise group/company rollups.
- Added hierarchical R-hat, bulk/tail ESS, MCSE, and HDI reporting plus a dedicated
  `InferenceError` Python exception.
- Added Rust and Python coverage for singleton adaptive shrinkage, missing values,
  reproducibility, axis alignment, ragged validation, and coherent aggregation.

### Performance

- Precomputed group membership and per-program sufficient statistics so each Gibbs
  sweep is linear in programs/groups rather than rescanning all programs per group.
- Stored predictive observations contiguously, reconstructed static state paths lazily,
  and added checked posterior/forecast allocation guards.

### Documentation

- Documented the static hierarchical-intercept estimand, ragged-series weighting,
  prior sensitivity, and the boundary with dynamic local-level forecasting.
- Added a complete hierarchical mean and rollup example.

## [0.9.0] - 2026-08-02

### Added

- Fitted Bayesian local-level, seasonal local-level, local-linear-trend, and directly
  observed AR(p) forecast models with coherent posterior paths.
- Fixed-matrix linear-Gaussian state-space filtering, smoothing, missing-observation
  handling, and forecasting.
- In-memory compile-once/bind-many model reuse.
- A reproducible cross-engine benchmark harness for rustmc, PyMC, PyMC with nutpie, and
  NumPyro with analytic posterior checks and isolated backend environments.
- A rebate-accrual example that distinguishes latent credible intervals from
  future-observation posterior-predictive intervals and aggregates paths correctly.

### Changed

- Reframed the project as a practical, general Bayesian toolkit; forecasting is one
  application rather than the library's identity.
- Added configurable `target_accept` to generic NUTS/HMC sampling entry points.
- Extended fixed linear-Gaussian state-space forecasts with joint future-observation
  covariance and exact cumulative Gaussian summaries.
- Added a fixed-parameter sum-to-zero seasonal local-level state-space constructor.
- Added fitted Bayesian seasonal local-level inference with Gibbs/FFBS, missing-value
  support, and coherent seasonal and cumulative posterior-predictive paths.
- Exposed generic sampler transition diagnostics through the Python API.
- Moved internal planning, review, and validation notes out of the source repository.
- Removed generated plots and scratch data from version control.

### Fixed

- Corrected the Lanczos `ln_gamma` implementation used by Negative-Binomial log density.
- Corrected NUTS and HMC initial step-size threshold calculations.
- Replaced misleading diagnostic calculations with a genuine 94% HDI,
  rank-normalized folded split R-hat, and corrected bulk/tail ESS estimation.
- Corrected constrained-parameter use in predictors and tightened cross-model parameter
  reference validation.
- Synchronized Rust, Python, wheel, and runtime package versions.

### Packaging

- Prepared Python 3.9+ ABI3 wheel metadata and clean-wheel verification.
- Marked the internal Python extension crate as non-publishable on crates.io, where the
  `rustmc` name belongs to an unrelated package.

## [0.8.0] - 2026-04-25

- Last public PyPI release before the fitted forecasting and 0.9 correctness work.

[Unreleased]: https://github.com/tbosier/rustmc/compare/v0.14.0...HEAD
[0.14.0]: https://github.com/tbosier/rustmc/compare/v0.13.0...v0.14.0
[0.13.0]: https://github.com/tbosier/rustmc/compare/v0.12.0...v0.13.0
[0.12.0]: https://github.com/tbosier/rustmc/compare/v0.11.0...v0.12.0
[0.11.0]: https://github.com/tbosier/rustmc/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/tbosier/rustmc/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/tbosier/rustmc/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/tbosier/rustmc/releases/tag/v0.8.0
