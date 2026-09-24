# Changelog

All notable changes to rustmc are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases use semantic
versioning while the public API is stabilized.

## [Unreleased]

Nothing since 0.12.0 has been published. The manifests already say 0.13.0, and the
changes below will ship under that number. An earlier revision of this file dated
0.13.0 to 2026-09-18, but no tag, PyPI upload or crates.io upload was made for it.
The changes came out of three repository-wide reviews and are grouped by review,
newest first.

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
- `Evaluator::forward`, a forward-only evaluation used by prediction.
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
  parameter of 2 to 512 elements used to get a dense metric block estimated from at
  most 200 warmup draws, which is noisy at 50 dimensions and singular above 200. An
  isotropic 300-element posterior took 431 leapfrog steps per iteration and 31
  seconds; it now takes 15 steps and 0.26 seconds, the same as scalar parameters.
  The default `"auto"` goes dense only when a window's correlation clearly exceeds
  its own sampling noise, which keeps the gain on correlated regression coefficients
  (ESS per second on a 20-coefficient ρ=0.9 regression rose from about 32k to 60k).
- **The last warmup window is stretched, not cut short**, following Stan. At
  `warmup=1000` the final metric came from 100 draws (850–950) and discarded the
  400-draw estimate; it now comes from draws 450–950. The default `warmup=500` was
  unaffected. HMC now uses the same windowed schedule instead of one 15–90% window.
- **Chains are seeded independently.** The graph sampler seeded chain `c` with
  `seed + c`, so seed 42's second chain was seed 43's first. Every sampler now keys
  chains through `seeding::chain_seed`, which mixes seed, domain and chain index in
  separate rounds so that no fixed offset of the seed reproduces another chain.
- **Chains start from different points.** Without `init`, graph-model chains start
  uniformly on (−2, 2) in unconstrained coordinates, redrawing non-finite starts. The
  local-level, trend, seasonal, regression, structural, hurdle and hierarchical
  Gibbs samplers start each chain from its own overdispersed point instead of the
  prior mode or the data means. Split R-hat assumes dispersed starts.
- Forward-filtering backward-sampling runs one filter pass instead of two, 25–45%
  faster; draws differ in the last bits. Hurdle uses the scalar version.
- Dynamic GLM block updates evaluate only the groups a block touches: a sweep is
  O(G·T) rather than O(G²·T), with the same target and, for a given stream, the same
  draws. Two measurements on different runs put the speed-up at 13–20× for 50
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
- Forecasting models accept integer arrays and lists as well as float arrays, and
  report a bad input by argument name instead of PyO3's conversion message. A 1-D
  `y` for a dynamic GLM raises `ValueError` rather than `TypeError`.
- Forecast quantiles and intervals use one interpolation rule, defined once in Rust.
  Values move by at most a few ulps; ties are now exact and a constant forecast's
  interval is that constant. `ForecastDraws.interval` and the evaluation scores use
  the same rule.
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
- Oversized forecasts or draw counts for the local-level, trend, seasonal and AR
  models aborted the process; they now raise, like the other models.
- `digamma` never returned for arguments at or below −2⁵³ or −∞.
- Normal and log-normal observation terms returned NaN rather than −∞ for a scale
  at or below zero.
- The step-size search could return a non-finite step, repeated its first probe and
  reused one momentum draw for every probe.
- Runoff chain 0 ran on the raw seed.
- The NUTS leapfrog step allocated about five times per step despite being
  documented as allocation-free; a counting-allocator test now holds it to that.
- An AR forecast overflowing on an explosive draw names the chain, draw and step.
- State simulation errors name the covariance that failed.
- `hierarchical.rs` no longer claims the conjugate Gibbs sampler avoids funnel
  geometry; with weak data it can stick near a group variance of zero.
- Data dictionaries silently coerced what they could not represent: a scalar
  became a length-1 vector, booleans became 0/1, the string `'1.5'` became a number,
  complex values lost their imaginary part, and integers above 2⁵³ were rounded.
  These, and arrays with more than two dimensions, now raise an error naming the
  key.
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

[Unreleased]: https://github.com/tbosier/rustmc/compare/v0.12.0...HEAD
[0.12.0]: https://github.com/tbosier/rustmc/compare/v0.11.0...v0.12.0
[0.11.0]: https://github.com/tbosier/rustmc/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/tbosier/rustmc/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/tbosier/rustmc/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/tbosier/rustmc/releases/tag/v0.8.0
