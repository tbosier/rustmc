# rustmc: deep review and a direction worth pursuing

_Review date: 2 August 2026_

## Executive conclusion

rustmc should not try to become another Stan, PyMC, or NumPyro. It should become the
best **structure-aware Bayesian execution runtime for repeated, production-shaped
workloads**.

That means:

- a deliberately finite set of well-supported models and components;
- exact, conjugate, Kalman, FFBS, Laplace, or MCMC inference selected from model
  structure rather than forcing every problem through NUTS;
- exceptional performance across the _whole job_: import, bind, fit, update, forecast,
  backtest, serialize, reload, and run across hundreds or thousands of datasets;
- deterministic, thread-safe native execution with a small Python footprint;
- coherent posterior-predictive paths, not just point forecasts and marginal bands;
- versioned, data-safe artifacts that can run without Python; and
- unusually visible evidence about correctness, calibration, failure modes, and speed.

The project already contains the beginnings of this idea: a compact graph and sampler,
compile/bind separation, deterministic Rayon parallelism, exact and state-space methods,
joint forecast paths, ArviZ export, and honest documentation. The most impressive part
of the repository is not an algorithm; it is that the README and demo disclose limits
and negative results instead of hiding them.

The project is not work-ready yet. The generic inference engine has not been validated
on a broad independent corpus. The forecasting surface consists of separate model
silos rather than composable components. A dense seasonal implementation takes about
272 seconds for a period-52 example. One synthetic monthly case gives only 16.7%
coverage for a nominal 95% predictive interval. There is no stable portable artifact,
online update contract, named time/panel schema, robust observation layer, or public
retained cross-engine benchmark. The Python binding is a 6,155-line monolith.

Those are normal facts for a young, mostly single-author alpha. They are also a reason
not to market the package as a general Bayesian toolkit yet. The best near-term move is
to narrow the promise, deepen the implementation, and earn trust in one demanding
operational niche.

The proposed one-sentence position is:

> **rustmc turns supported Bayesian models into fast, repeatable operational jobs—especially many-series forecasts and repeated inference—without requiring users to assemble a compiler, tensor backend, sampler backend, backtest loop, and deployment wrapper.**

The path to the “pantheon” is not feature parity. It is becoming the tool that serious
users instinctively reach for when they have a supported model and need to run it
reliably at work.

## Scope and method

This review covers the repository at commit `a7536b5`, the current [README](README.md),
[roadmap](ROADMAP.md), [demo artifacts](demo-docs/README.md), the Rust and Python test
suites, and the public positioning and user communities of Stan, PyMC, NumPyro, Orbit,
and the related Rust sampler nutpie.

Local checks performed for this review:

- `cargo fmt --all -- --check`: passed;
- `cargo clippy --workspace --all-targets -- -D warnings`: passed;
- `cargo test --workspace --release`: 102 unit tests and 12 recovery tests passed;
- `python -m pytest -q`: 138 passed, 1 skipped, 1 expected failure, 1 deselected;
- the quick Rust-only benchmark completed and passed its deliberately relaxed quick
  gate, but its 100-draw result is not publication-quality performance evidence.

This is a product, architecture, and evidence review—not a formal proof or an
independent line-by-line audit of every density and sampler transition. Before the
software controls consequential decisions, its numerical core still needs review by
experienced statistical-computing contributors and validation against independent
reference implementations.

## What the neighboring projects are actually trying to achieve

The four projects overlap, but they optimize for different things.

| Project | Primary objective | Defining strength | Cost users accept |
|---|---|---|---|
| Stan | A portable, rigorous language and runtime for differentiable statistical models | Trusted HMC/NUTS, static semantics, math library, diagnostics, scientific depth | A DSL, C++ compilation, explicit model engineering |
| PyMC | Make flexible Bayesian modeling natural inside scientific Python | Expressive Python API, broad distributions, rich workflow and ecosystem | Symbolic graphs, shapes/dims complexity, backend and compilation knowledge |
| NumPyro | A lightweight, composable PPL on JAX | JIT/vmap/accelerators, effect handlers, broad inference research surface | JAX’s functional, static-shape, PRNG, compilation, and deployment model |
| Orbit | An approachable end-to-end Bayesian forecasting package | Opinionated models, `fit`/`predict`, decomposition, backtesting | A narrower model catalog and backend/toolchain constraints |
| rustmc opportunity | Operational execution for supported structured models | Predictable CPU latency, batch/update/artifacts, specialized inference, coherent paths | Less arbitrary modeling freedom in exchange for a strong contract |

### Stan: trusted language and inference machinery

Stan is a statically typed probabilistic-programming language that defines a
conditioned log density. `stanc3` translates a model into C++, which uses Stan Math’s
reverse-mode automatic differentiation and a mature inference runtime. The current
manual covers HMC/NUTS, optimization, ADVI, Pathfinder, and Laplace approximation, and
the same language is available through several host-language interfaces. See the
[Stan Reference Manual](https://mc-stan.org/docs/reference-manual/index.html),
[Stan Math](https://mc-stan.org/math/), and
[BridgeStan](https://roualdes.us/bridgestan/latest/index.html).

What sets Stan apart is not merely NUTS. It is the combination of language semantics,
parameter transforms, numerical methods, error messages, documentation, diagnostics,
and a community that treats posterior geometry and model criticism as first-class
work. Its [about page](https://mc-stan.org/about/) describes a global, NumFOCUS-backed
project built around open code and reproducible science. The forum contains thousands
of modeling discussions and a great deal of accumulated statistical judgment.

The recurring costs are visible:

- C++ toolchains and compile latency remain user concerns. Recent forum threads still
  report network-sensitive compilation, very long `brms` compilation, and difficult
  Windows setup
  ([2024](https://discourse.mc-stan.org/t/compilation-speed-and-internet-connection-cmdstanr/35297),
  [2025](https://discourse.mc-stan.org/t/why-does-this-brms-generated-model-not-compile-after-15-minutes/39799),
  [2026](https://discourse.mc-stan.org/t/issue-installing-cmdstan/40990)).
- Discrete unknowns must usually be marginalized, and some discrete structures are not
  tractable in Stan’s continuous-parameter inference model
  ([forum explanation](https://discourse.mc-stan.org/t/a-question-on-stan-cannot-deal-with-discrete-parameters/3308/8)).
- Within-chain parallelism is powerful but requires model rewrites, compile flags, and
  careful resource choices; GPU acceleration is selective rather than automatic
  ([parallelization guide](https://mc-stan.org/docs/cmdstan-guide/parallelization.html)).
- Warmup and difficult posterior geometry can dominate run time. That is not a Stan
  defect so much as a fact of general HMC, but it makes repeated small jobs expensive
  ([example discussion](https://discourse.mc-stan.org/t/any-way-to-speed-up-warmup/13956)).
- Stan supplies modeling and inference, not an opinionated operational forecast system.
  Rolling-origin evaluation, artifact lifecycle, serving, and online updates live in
  interfaces or application code. Forum guidance for real-time updating has generally
  been to refit periodically or implement a specialized filter
  ([online inference](https://discourse.mc-stan.org/t/online-inference/5001),
  [real-time state space](https://discourse.mc-stan.org/t/real-time-online-bayesian-inference/16786)).

Stan already compiles native code and reuses a compiled program with new data. Therefore
“native” and “compile once” are not differentiators by themselves. rustmc should use
Stan as a reference implementation and source of workflow discipline, not as a feature
checklist.

### PyMC: the native language of exploratory Bayesian work in Python

PyMC aims to make comprehensive Bayesian modeling readable and interactive inside
Python. Users construct random-variable graphs with a friendly API; PyTensor rewrites
and compiles log-probability graphs; PyMC orchestrates sampling and prediction; ArviZ
provides labeled inference data, diagnostics, plots, and model comparison. Its own
[architecture document](https://github.com/pymc-devs/pymc/blob/main/ARCHITECTURE.md)
explicitly treats PyTensor and ArviZ as separate foundational projects.

PyMC’s strengths are breadth and approachability: a large distribution catalog,
arbitrary deterministics, custom distributions and operations, coordinates, Gaussian
processes, time-series tooling, SMC and variational inference, mixed samplers for
discrete and continuous variables, notebooks, books, conferences, office hours, and
an active Discourse. Its
[governance](https://github.com/pymc-devs/pymc/blob/main/GOVERNANCE.md) emphasizes
openness, institutional neutrality, and low barriers between users and contributors.

The pain points seen repeatedly on Discourse are instructive:

- The symbolic layer leaks. NumPy code is not automatically PyTensor code, and a Python
  black-box likelihood does not automatically have gradients suitable for NUTS
  ([official overview](https://www.pymc.io/projects/docs/en/latest/learn/core_notebooks/pymc_overview.html)).
- Shapes, dimensions, coordinates, and positional broadcasting remain a significant
  cognitive burden
  ([dims discussion](https://discourse.pymc.io/t/understanding-dimensions-shapes-of-variables/12821),
  [systematic dims thread](https://discourse.pymc.io/t/systematic-introduction-of-coords-dims-and-shapes/16057)).
- Fast execution can require knowing when to choose C, Numba, JAX, nutpie, NumPyro, or
  BlackJAX. Backend gains can be large, but compilation and compatibility costs vary
  ([fast-sampling documentation](https://www.pymc.io/projects/examples/en/latest/samplers/fast_sampling_with_jax_and_numba.html)).
- Repeated same-structure inference is a real use case. Users ask how to avoid repeated
  compilation and how to run a model across hundreds or thousands of datasets
  ([model reuse](https://discourse.pymc.io/t/model-compilation-avoidance-when-reusing-a-model/13863),
  [parallel model selection](https://discourse.pymc.io/t/best-practice-for-parallel-model-selection-especially-avoidance-of-recompilation/16904)).
- Model persistence and deployment are not a single standardized core path. PyMC’s
  `ModelBuilder` answer is useful but experimental and requires Python code alongside
  stored inference data
  ([ModelBuilder tutorial](https://www.pymc.io/projects/examples/en/latest/howto/model_builder.html)).
- Forecasting has historically involved careful mutable-data, coordinate, recursion,
  and posterior-predictive plumbing
  ([forecasting thread](https://discourse.pymc.io/t/best-practices-for-time-series-forecasting/12232)).

The gap is getting narrower. In July 2026 PyMC Labs released `pymc_forecast` 0.0.1. It
already supplies labeled forecasts, rolling-origin backtests, probabilistic scores,
hierarchical batch dimensions, multiple inference backends, and state-space/Kalman
interoperation. Its API is early, but its scope is a warning against building “PyMC, but
with a forecast method”
([announcement](https://discourse.pymc.io/t/pymc-forecast-a-new-bayesian-time-series-forecasting-toolkit-for-pymc/17873)).

There is an even more direct warning: PyMC’s preferred fast path can use nutpie, whose
sampler is implemented in Rust. nutpie advertises a Rust NUTS implementation, compiled
PyMC and Stan models, reusable data binding, richer adaptation, and roughly 2× average
speed over Stan on its selected PosteriorDB comparison
([nutpie documentation](https://pymc-devs.github.io/nutpie/)). “NUTS written in Rust”
is therefore occupied territory.

rustmc should complement PyMC by being the compact operational runtime for supported
models, and should continue emitting excellent ArviZ-compatible results. It should not
try to win an API-expressiveness contest.

### NumPyro: composable, accelerator-native probabilistic programming

NumPyro combines Pyro-style embedded Python models and effect handlers with JAX. A model
is an ordinary function containing primitives such as `sample`, `param`, `plate`, and
`factor`. Handlers trace, seed, condition, substitute, mask, replay, and reparameterize
those primitive calls. JAX then supplies `grad`, `jit`, `vmap`, and accelerator
execution. See the [project overview](https://github.com/pyro-ppl/numpyro),
[getting-started guide](https://num.pyro.ai/en/stable/getting_started.html), and
[effect-handler paper](https://arxiv.org/abs/1912.11554).

This architecture is genuinely different from Stan and PyMC. It permits end-to-end JIT
compilation of iterative NUTS, functional transformation of models, vectorization over
chains and posterior draws, and close integration with the JAX scientific and neural
ecosystem. Its inference surface is broad: HMC/NUTS, mixed and discrete kernels, SVI,
autoguides, flows, enumeration, reparameterizers, and third-party JAX samplers.

The tradeoffs are the JAX execution model:

- explicit random keys, immutable arrays, pytrees, tracing, static shapes, `scan`, and
  batch/event dimension rules become part of everyday modeling;
- JIT startup can dominate a small or one-shot fit, while a changed shape can trigger
  recompilation;
- fully vectorized chains can exhaust accelerator memory;
- deployment and cross-process compilation reuse are not packaged as one simple,
  portable Bayesian artifact; and
- runtime behavior is coupled to JAX/XLA versions. A current issue reports a dramatic
  per-step regression after dependency upgrades on one large GPU model; it is not proof
  of general slowness, but it illustrates the operational risk
  ([issue #2225](https://github.com/pyro-ppl/numpyro/issues/2225)).

The recurring community requests align closely with rustmc’s opportunity: amortizing
compilation across many small fits, working around equal-shape requirements, controlling
vectorized-chain memory, and simplifying out-of-sample prediction
([repeated fits](https://forum.pyro.ai/t/parallelising-numpyro/2442),
[batching issue](https://github.com/pyro-ppl/numpyro/issues/2204),
[prediction issue](https://github.com/pyro-ppl/numpyro/issues/2158)).

NumPyro is active and sophisticated. rustmc should not compete for enormous flexible
GPU models or effect-handler elegance. It can win predictable cold and warm CPU latency,
variable-length batch execution, small deployable artifacts, and specialized algorithms
that eliminate generic sampling work.

### Orbit: a forecasting product layer, not a general PPL

Orbit’s original thesis was that business forecasting needed more than a PPL. It placed
an object-oriented, scikit-learn-like API over Stan and Pyro estimators and added
forecast models, decomposition, diagnostics, backtesting, plotting, and tuning. Its
catalog includes ETS, LGT, DLT, and kernel-based time-varying regression. See the
[Uber introduction](https://www.uber.com/ca/en/blog/orbit/),
[repository](https://github.com/uber/orbit), and
[paper](https://arxiv.org/abs/2004.08492).

Orbit proves that an opinionated application layer is valuable. Its model/estimator/
forecaster separation and `fit`/`predict` workflow are more approachable than asking
every analyst to implement recursive prediction in a PPL. DLT’s robust noise, trend
choices, signed regressors, and priors are especially relevant to business data.

The durable gaps in Orbit’s issue tracker are almost a specification for rustmc:

- multiple independent series and multivariate series have remained open requests
  ([#6](https://github.com/uber/orbit/issues/6),
  [#7](https://github.com/uber/orbit/issues/7));
- a request to learn jointly across nearly 12,000 store/brand series remains open
  ([#648](https://github.com/uber/orbit/issues/648));
- persistence for deployment is unresolved
  ([#500](https://github.com/uber/orbit/issues/500));
- prediction has had a reported thread-safety problem
  ([#782](https://github.com/uber/orbit/issues/782));
- users want rolling state updates without a full refit
  ([#764](https://github.com/uber/orbit/issues/764)); and
- uncertainty in future regressor scenarios is not propagated automatically
  ([#703](https://github.com/uber/orbit/issues/703)).

Orbit was updated in May 2026 and should not be called abandoned. A fair description is
compatibility-maintained, with less visible product momentum and a quieter community
than the major PPLs. rustmc should learn from Orbit’s approachable workflow without
copying its model zoo or depending on Stan/Pyro behind the scenes.

## What rustmc is today

### The good foundation

Several choices in the repository are unusually sensible for an alpha.

1. **The docs tell the truth.** The README says alpha, avoids universal speed claims,
   distinguishes posterior-predictive from latent intervals, notes that tests do not
   prove model appropriateness, and documents explosive AR draws. The demo publishes a
   severe undercoverage case and a slow seasonal path. Preserve this culture.

2. **The dependency surface is small.** NumPy is the only required Python runtime
   dependency. The Rust core depends on a compact set of numerical, random, parallel,
   and serialization crates. That supports the low-friction runtime thesis.

3. **Compiled structure and data binding are conceptually clean.** A structural graph
   can be shared while variable-row-count bindings are validated independently. Stable
   dataset identities and explicit schemas are good foundations for repeated jobs.

4. **Determinism is treated as a feature.** Per-chain seed derivation, ordered parallel
   collection, and cross-thread reproducibility tests matter for auditability and work
   operations.

5. **Joint forecast paths are retained.** `(chain, draw, horizon)` outputs support
   cumulative totals, threshold probabilities, and nonlinear downstream quantities
   without destroying dependence. Many forecasting APIs get this wrong.

6. **Specialized inference already exists.** Exact Normal-Inverse-Gamma AR inference,
   FFBS/Gibbs state-space models, Kalman filtering/smoothing, and general NUTS/HMC can
   become a real planner rather than unrelated features.

7. **Packaging discipline is better than the project’s age suggests.** Source and wheel
   installs are tested across Python 3.9–3.13 on Linux, release wheels are built for the
   major desktop targets, version consistency is checked, and strict Clippy is enabled.

8. **The benchmark harness understands statistical quality.** It records environment,
   separates phases, checks analytic posterior moments, computes ESS and R-hat through
   ArviZ, and refuses to interpret timing without a quality gate. This is the correct
   instinct even though the retained corpus is currently too small.

The existing [roadmap](ROADMAP.md) already contains most of the right ingredients. The
main recommendation is to change their order and concentration. After the trust gate,
move the structural IR, common result contract, state-space marginalization, artifact,
and batch lifecycle ahead of broad modeling-surface expansion. Named dimensions and
group indexing are necessary, but a larger generic graph API would consume years while
making the package easier to compare directly with mature PPLs.

### The current limitations that matter most

#### 1. The public identity is broader than the evidence

The README calls rustmc a “practical, general-purpose Bayesian toolkit,” but the generic
surface is still a finite graph builder with a small expression language, limited
vector hierarchies, incomplete dimensions/indexing, limited initialization controls,
and no broad independent target corpus. “General-purpose” invites comparison with Stan
and PyMC on their strongest axis.

For now, describe it as a **structure-aware Bayesian runtime with a supported model
catalog and a generic continuous-model escape hatch**. Broaden the wording only when the
evidence broadens.

#### 2. There are three architectures hiding in one package

The graph/NUTS path, fixed linear-Gaussian state-space path, and specialized fitted
forecast models have different builders, configurations, result types, diagnostics,
and prediction lifecycles. The roadmap proposes a common kernel registry, but users
cannot yet compose a trend, two seasonalities, regressors, and an observation family and
have the runtime derive an inference plan.

The next major abstraction should unify these paths. Adding more stand-alone model
classes first will deepen the split.

#### 3. The Python boundary is a maintenance risk

[`python_bindings/src/lib.rs`](python_bindings/src/lib.rs) is 6,155 lines and contains
model specifications, expression handling, validation, sampling orchestration, result
conversion, predictive simulation, state-space wrappers, and module registration. This
makes every new feature touch a large, difficult-to-reason-about boundary.

Split it before expanding the model surface. Add `.pyi` stubs and `py.typed`; treat the
Python API as a product rather than an incidental PyO3 exposure.

#### 4. The generic sampler is an expensive correctness obligation

The NUTS implementation includes iterative multinomial tree building, windowed mass
adaptation, dual averaging, divergence telemetry, and block metrics. The tests are
thoughtful, but recovery on a dozen small targets is not enough to establish a general
sampler.

Maintaining a home-grown NUTS implementation is not a product differentiator now that
`nuts-rs`/nutpie exists. Evaluate replacing the default generic fallback with
[`nuts-rs`](https://docs.rs/nuts-rs/latest/nuts_rs/) or retaining the current sampler
only as a documented experimental/reference engine. Compare both on PosteriorDB and
adversarial geometry before deciding. If rustmc keeps its sampler, that must be because
measured requirements cannot be met by the shared implementation—not because writing a
sampler is interesting.

The same principle applies to diagnostics: keep native health checks for deployment,
but continually pin them to ArviZ and independent implementations.

#### 5. The state-space implementation is not yet shaped for the proposed niche

The fixed state-space core is useful, but the fitted APIs lack known future regressors,
calendar and intervention effects, multiple seasonalities, positive/count/robust
observations, hierarchical pooling, and online continuation. The seasonal FFBS path
uses a dense 53-state representation for period 52 and took 272.15 seconds in the
repository’s own [demo report](demo-docs/results/rustmc_method.md).

The highest-value technical work is structured linear algebra:

- square-root or UD filtering for numerical stability;
- block-banded/sparse transitions;
- direct recurrences for dummy seasonal states;
- low-dimensional trigonometric seasonal states;
- disturbance smoothing where it avoids dense state covariance work; and
- marginalization of latent Gaussian states while sampling only low-dimensional
  hyperparameters.

This can create orders-of-magnitude value without making any universal Rust claim.

#### 6. Forecast quality is promising but plainly incomplete

The [six-series synthetic study](demo-docs/README.md) is a good diagnostic, not a
product benchmark. It shows fast conjugate AR fits and useful results on several cases.
It also shows:

- only one of six observations inside a nominal 95% interval on `monthly_medium`;
- the need for a joint trend-plus-seasonal model rather than an external drift
  adjustment;
- weak performance on a hard multi-seasonal, pulse-and-regime weekly series; and
- a specialized seasonal model too slow to serve as the honest structural alternative.

Do not “fix” these results by widening intervals or adding hidden preprocessing. Build
the missing generative structure and propagate all uncertainty through the posterior.

#### 7. The deployment story is still a design note

The legacy JSON artifact owns data, while the compile/bind Python object is in-memory
only, as the [compiled-model design note](docs/architecture/compiled-model.md) explains.
There is no portable slot-only artifact, compatibility policy, thread-safe serving
contract, checkpoint, or Rust runtime example that loads an audited fitted artifact and
produces a forecast.

Until artifact v2 exists, “deployable” is a direction, not a feature.

#### 8. Community risk is larger than code risk

The public repository was created in February 2026, has 70 commits from one person under
two author identities, 14 stars, one fork, and no public issues as of this review. There
is no `CITATION.cff`, security policy, code of conduct, governance document, or named
maintainer/reviewer structure. A rapid one-author build can be excellent, but nobody at
work should trust consequential inference because the author and an AI agreed that the
tests look good.

Independent statistical review, real user reports, replication, and time are not
optional extras. They are the moat.

#### 9. The name has collision costs

The crate named `rustmc` is unrelated, which already requires `rustmc_core` in Rust.
“RustMC” is also used by a Rust stateless model-checking project. This will continue to
hurt searchability and confuse Rust users. A rename is disruptive, but before 1.0 is
the least costly time to decide. At minimum, adopt a distinctive subtitle everywhere;
ideally evaluate a name centered on the runtime’s actual promise rather than its
implementation language.

## The product territory rustmc should own

### The boundary

rustmc should be broader than a forecasting model zoo but narrower than a universal
PPL. The stable generality should come from **composable structural components and a
common inference/runtime contract**, not arbitrary tensor syntax.

The supported center:

- Bayesian structural time series;
- dynamic and static regression/GLMs;
- repeated small-area, site, experiment, reliability, and rate models;
- many structurally identical datasets;
- online filtering plus periodic parameter refresh;
- embedded or service-side posterior prediction; and
- exact, conjugate, marginalized, or low-dimensional approximate inference.

The escape hatch:

- a documented value-and-gradient target interface for advanced continuous models;
- generic NUTS through a well-validated shared sampler; and
- clear advice to use Stan, PyMC, or NumPyro when a model falls outside the supported
  structure.

This is “general enough for other things” without claiming to be a general modeling
language.

### A north-star workflow

The ideal user should be able to write something conceptually like:

```python
model = (
    rmc.StructuralModel(time="date", target="demand")
    .local_linear_trend()
    .seasonality(period=7, kind="trigonometric", harmonics=3)
    .seasonality(period=365.25, kind="trigonometric", harmonics=8)
    .regression(["price", "promotion", "holiday"], dynamic=False)
    .student_t_observation()
)

artifact = model.compile()
fit = artifact.fit(train, id="store-104", seed=42)
forecast = fit.forecast(future_covariates, horizon=28, seed=43)
updated = fit.update(new_observations)
report = artifact.backtest(panel, horizon=28, origins=8)
```

The important feature is not the chaining syntax. It is that compilation produces an
explicit plan such as:

```text
model schema
   -> structural analysis
   -> marginalized Gaussian state likelihood
   -> NUTS over 6 static hyperparameters
   -> simulation smoother for requested state draws
   -> coherent forecast paths and fit health report
```

Users must be able to inspect, pin, and override this plan. Automatic inference that
silently changes the target distribution would destroy trust.

### The architecture that supports the boundary

#### 1. A typed model IR

Define a small, serializable intermediate representation containing:

- named parameters, data slots, dimensions, transforms, priors, and observation heads;
- structural components with explicit state and parameter contributions;
- missing-data and support semantics;
- time index, panel keys, aggregation relationships, and future-covariate schema;
- versioned mathematical parameterizations; and
- no embedded observations unless the caller explicitly requests a fitted-data bundle.

This IR should be stricter than Python and more stable than internal evaluator nodes.

#### 2. A structure analyzer and kernel registry

The analyzer should produce an explainable `InferencePlan` from the IR. Initial kernels:

- exact conjugate posterior;
- Kalman filter/smoother and simulation smoother;
- conjugate Gibbs/FFBS;
- marginalized state-space likelihood plus NUTS;
- MAP plus Laplace with approximation diagnostics; and
- general NUTS as the final continuous-model fallback.

Later kernels are justified only by a real workload and an evidence plan. Pathfinder is
more compelling than accumulating several weak VI implementations because it can also
initialize MCMC and has established diagnostics in Stan
([Stan Pathfinder](https://mc-stan.org/docs/reference-manual/pathfinder.html)).

#### 3. One fit/result contract

Every kernel should return the same conceptual groups:

- posterior parameters;
- optional latent states;
- prior and posterior predictive draws;
- pointwise log likelihood where meaningful;
- sampler or approximation diagnostics;
- inference-plan metadata;
- data schema, coordinates, and provenance; and
- machine-readable health status with remediation.

ArviZ `InferenceData` should remain the canonical Python interchange format. Native Rust
results can be leaner, but names and dimensions should map without guessing.

#### 4. A real artifact and runtime

Artifact v2 should be immutable, versioned, and safe by default. It should contain:

- model IR and validated slot schema;
- selected/pinned inference plan;
- fitted parameter draws or approximation state when requested;
- optional filter terminal state and adaptation state;
- package/ABI/version metadata;
- mathematical parameterization identifiers;
- seed derivation policy and provenance;
- resource limits and compatibility checks; and
- a manifest declaring whether any training data or derived sensitive values are stored.

It must support concurrent read-only prediction and return deterministic results for a
given artifact, input, and seed. Load/save round trips need golden compatibility tests.

#### 5. An explicit execution policy

One runtime must decide how threads are divided across datasets, chains, state-space
operations, and linear algebra. Nested Rayon/BLAS oversubscription should be impossible
by default. Batch calls need bounded queues, streaming submission, cancellation, per-job
timeouts, stable IDs, and collect-errors semantics.

## The forecasting wedge

Forecasting is the right proving ground because it exercises model structure,
prediction, repeated fits, coherent uncertainty, online state, and operational
deployment. It should not become the only identity of the core.

### First model family to make excellent

Build one composable structural family rather than five more stand-alone classes:

- local level and local linear trend;
- one or more dummy or trigonometric seasonal components;
- static known-future regression;
- interventions and calendar indicators;
- Gaussian and Student-t observations;
- lognormal/positive observations next;
- count and hurdle/intermittent demand after the Gaussian/robust path is calibrated;
- missing observations and irregular-but-declared schedules; and
- stationarity-aware AR errors where required.

For Gaussian models with unknown static hyperparameters, marginalize the latent state
sequence in the Kalman likelihood and sample only the small parameter vector. Draw
states afterward with a simulation smoother. This is typically a much more important
speed win than optimizing generic NUTS over every latent time step.

Use structured seasonal recurrences or low-rank harmonic states so period 52 and daily/
annual combinations do not create dense cubic work. Add square-root filtering before
claiming numerical robustness over long production series.

### Work-facing capabilities

The application layer should provide:

- real datetime coordinates and frequency validation;
- panels with stable series IDs and variable history lengths;
- known-future covariates and scenario sets for uncertain future regressors;
- `.update()` for filtering new observations without refitting static parameters;
- explicit policies for when to refresh parameters;
- rolling and expanding backtests with no future leakage;
- naive, seasonal-naive, ETS/ARIMA, and suitable ML baselines;
- CRPS/WIS, log score where stable, coverage, bias, sharpness, PIT/rank diagnostics,
  RMSE/MAE, and decision-specific quantities;
- pathwise totals, peaks, stockout probabilities, threshold events, and scenario
  comparisons; and
- probabilistic reconciliation for aggregate hierarchies after the single-level path is
  trustworthy.

The standard of a probabilistic forecast is not merely a low RMSE. It is sharpness
subject to calibration, evaluated out of sample. WIS is a useful quantile approximation
to CRPS, but coverage across many origins and series must be visible rather than reduced
to one average score.

### General uses beyond forecasting

The same runtime and artifact can support:

- thousands of repeated A/B or conversion analyses;
- site- or region-level GLMs with a shared structure;
- reliability and event-rate models with exposure and censoring;
- embedded Bayesian linear/GLM scoring;
- sensor calibration and anomaly probabilities;
- small-area estimation; and
- repeated biomedical or engineering fits where startup, schema safety, and
  reproducibility matter.

These are good extensions because they reuse compile/bind, batch execution, constrained
parameters, predictive paths, and exact/low-dimensional inference. They should arrive
through user evidence, not a desire to lengthen the distribution table.

## A roadmap with gates

### Phase 0: choose and state the boundary (now)

Deliverables:

- replace “general-purpose Bayesian toolkit” with the narrower operational-runtime
  position until evidence supports broader wording;
- decide whether to rename before 1.0;
- mark the generic NUTS path experimental unless and until it passes the independent
  validation gate;
- rename adversarial examples such as `benchmark_vs_pymc.py` toward “cross-engine
  validation” language;
- publish a compatibility matrix that says exactly what Python/Rust APIs are stable;
  and
- write a one-page good-neighbor policy: credit upstream methods, use peers as oracles,
  report matched work, and never imply that native Rust makes statistical inference
  automatically better.

Exit gate: a new user can explain in one sentence when to use rustmc and when to use
Stan/PyMC/NumPyro.

### Phase 1: make the core auditable (0–3 months)

Deliverables:

- split the PyO3 monolith into model, inference, diagnostics, results, state-space, and
  conversion modules;
- add type stubs, `py.typed`, stable errors, termination reasons, BFMI, initialization
  controls, max-treedepth reporting, and unified result shapes;
- resolve the default-sampler decision after a `nuts-rs` evaluation;
- implement a CI-sized SBC suite plus higher-power scheduled runs;
- add a curated PosteriorDB subset covering benign, correlated, constrained,
  hierarchical, funnel, heavy-tail, and failure targets
  ([PosteriorDB](https://github.com/stan-dev/posteriordb));
- independently pin every density, transform Jacobian, prediction family, R-hat, ESS,
  MCSE, and HDI implementation;
- test source and installed wheels on Linux, macOS, and Windows, not only build release
  wheels for the latter two; and
- add fuzz/property tests for schemas, artifacts, missing data, and numerical extremes.

Exit gate: an external reviewer can trace every generic inference claim to an
independent target, and every supported wheel runs the same validation bundle.

### Phase 2: build the structural compiler (2–6 months)

Deliverables:

- typed structural IR and inspectable inference plans;
- composable local level/trend, multiple-seasonal, regression, intervention, and
  Gaussian/Student-t observation components;
- marginalized Gaussian state inference over unknown hyperparameters;
- simulation smoothing for coherent latent/posterior paths;
- sparse/structured and square-root state-space computation;
- stationarity-safe AR parameterization, preferably through partial autocorrelations;
  and
- one result and diagnostics contract across all kernels.

Exit gate: the repository’s six synthetic cases are expressed without external drift
preprocessing; period-52 execution is no longer pathologically dense; and interval
calibration improves because the model, not an ad hoc band, represents the missing
uncertainty.

### Phase 3: make repeated work delightful (5–9 months)

Deliverables:

- first-class panel/batch API with variable lengths, stable IDs, streaming input,
  bounded memory, partial failures, cancellation, and explicit thread policy;
- `.fit`, `.forecast`, `.update`, `.backtest`, and `.score` lifecycles;
- dates, panels, known-future covariates, and scenario propagation;
- artifact v2 with Rust and Python load/run examples;
- thread-safe serving and checkpoint/restart tests; and
- a CLI suitable for scheduled batch jobs.

Exit gate: one artifact runs the same audited model across at least a thousand varied
datasets, isolates bad jobs, resumes safely, and reproduces outputs after serialization.

### Phase 4: earn the workplace claim (8–15 months)

Deliverables:

- a retained benchmark corpus with public data, generated data, exact references, and
  known failure regimes;
- cold, warm, batch, update, forecast, memory, and throughput comparisons against
  appropriate alternatives—not only sampler-loop timing;
- probabilistic forecast evaluation over many origins and series;
- at least three real shadow-mode workplace pilots in different domains;
- security, governance, citation, deprecation, and support policies;
- two or more maintainers who can review statistical core changes; and
- an independent methods review or paper/preprint describing the inference planner and
  validation evidence.

Exit gate: users outside the author’s organization reproduce the claims and operate the
tool for months without private fixes.

### Phase 5: expand only from evidence

Good candidates after the prior gates:

- positive, count, hurdle, and intermittent-demand state-space observations;
- hierarchical/panel pooling and probabilistic reconciliation;
- censored reliability and exposure models;
- portable tuned adaptation/checkpoint state;
- Arrow/Polars input adapters that preserve the small core; and
- a stable custom value-and-gradient ABI.

GPU inference, distributed MCMC, arbitrary discrete latent variables, a universal tensor
language, Stan import, and a large sampler collection should stay deferred until a
specific workload proves that this architecture is the right home.

## The evidence program

### Separate four questions

Every benchmark or validation report should answer four different questions:

1. **Target correctness:** does the engine evaluate the intended density and transforms?
2. **Inference correctness:** do repeated simulated fits recover calibrated parameters
   and reference posteriors?
3. **Predictive validity:** are forecasts calibrated and useful under rolling real or
   realistic held-out data?
4. **Operational performance:** how long and how much memory does the whole workflow use
   at cold start and steady state?

A fast wrong posterior must fail before timing is interpreted. A correct posterior under
the assumed model can still forecast badly under misspecification. A well-calibrated
model can still be operationally unusable. Keep the layers separate.

### Required suites

#### Density and gradient suite

- analytic values and finite differences for every operation and support boundary;
- cross-language checks against Stan Math, SciPy, PyMC/PyTensor, or carefully chosen
  references;
- extreme-tail and transform-Jacobian cases;
- property tests for vectorization, rebinding, and missingness; and
- NaN/Inf/overflow behavior that returns typed errors rather than plausible output.

#### Inference suite

- SBC for every advertised model family
  ([Stan SBC guide](https://mc-stan.org/docs/stan-users-guide/simulation-based-calibration.html));
- PosteriorDB reference comparisons over a labeled subset designed for the intended
  algorithms, not cherry-picked easy targets;
- repeated seeds and adversarial geometries;
- posterior means, scales, quantiles, ESS, MCSE, divergence behavior, and rank tests;
- cross-engine comparison through the same ArviZ representation; and
- explicit expected failures for models the engine should reject.

#### Forecast suite

- synthetic generators where calibration is known;
- M4/M5-style public corpora and domain-specific public datasets with licenses recorded;
- multiple origins, horizons, frequencies, missingness patterns, and regime changes;
- seasonal-naive and strong classical/ML baselines;
- CRPS/WIS, coverage by nominal level, bias, sharpness, PIT/ranks, RMSE/MAE, and
  path-dependent event scores; and
- selection time included in end-to-end results whenever selection is part of the
  workflow.

#### Operational suite

Measure and retain:

- import and cold compile time;
- bind/validation time;
- warm repeated fit time;
- ESS per wall-second and per gradient evaluation for sampled models;
- batch throughput and p50/p95 latency;
- peak and steady-state memory;
- artifact size and load latency;
- update versus full-refit latency;
- one, several, and many concurrent callers; and
- exact revision, packages, CPU/GPU, thread policy, seeds, raw samples or sufficient
  digests, and all failures.

Use at least three workload shapes: one large model, many small equal-shape jobs, and
many variable-length jobs. The last two are where rustmc’s claimed advantage should be
strongest.

## How to become trusted rather than merely impressive

Stan and PyMC reached their position through accumulated human judgment: papers,
reference manuals, thousands of forum answers, adversarial models, independent users,
governance, conferences, teaching materials, citations, and years of bugs. Code volume
and benchmark wins cannot substitute for that history.

rustmc can build trust faster by being unusually disciplined:

1. **Make every claim reproducible.** Link a release claim to a retained manifest and
   artifact. Keep negative results visible.
2. **Invite expert review early.** Ask statistical-computing researchers to attack the
   sampler, transforms, filters, and calibration—not to endorse the project.
3. **Use mature projects as oracles and credit them.** Publish equivalent Stan/PyMC/
   NumPyro models where licensing permits. Submit discrepancies upstream respectfully.
4. **Make examples executable and few.** Prefer ten maintained end-to-end cases over
   fifty overlapping scripts.
5. **Turn support into documentation.** A forum or discussions area becomes valuable
   when every difficult user question improves a guide, test, or error message.
6. **Create a contributor ladder.** Add a code of conduct, security policy,
   `CITATION.cff`, mathematical review checklist, good-first issues, and review ownership.
7. **Do not hide AI assistance.** Generated code is not disqualifying. Unreviewed
   statistical code is. Document provenance where useful and require the same evidence
   for human- and AI-written changes.
8. **Protect scope.** Politely tell users when Stan/PyMC/NumPyro is the better tool. This
   makes the supported rustmc surface more credible.

The realistic prestige target is not “as many users as PyMC.” It is:

> For supported operational models, rustmc is the reference implementation people trust for speed, reproducibility, and deployment—and its results are easy to validate in the broader Bayesian ecosystem.

## Good-neighbor positioning

To avoid stepping on toes:

- say “complements” and show the boundary with examples;
- describe Stan as the reference for flexible, rigorous model development;
- describe PyMC as the reference for exploratory Python modeling and rich workflows;
- describe NumPyro as the reference for flexible JAX/accelerator modeling;
- describe Orbit and `pymc_forecast` as evidence that forecasting workflow matters;
- keep ArviZ as the output lingua franca rather than inventing a closed analysis format;
- consider depending on or contributing to `nuts-rs` instead of competing on a generic
  Rust sampler;
- use cross-engine benchmarks as correctness and workload studies, never horse races;
- never imply that a failure to converge makes another engine slow; and
- never call a specialized exact method a universally faster replacement for NUTS.

Suggested website copy:

> rustmc is a native runtime for supported Bayesian models that need to run repeatedly and predictably. It combines reusable model structure with exact and structure-aware inference, deterministic batch execution, coherent forecast paths, and portable artifacts. Use a full probabilistic programming system when you need arbitrary model flexibility; use rustmc when a validated supported model has to become an operational job.

## What to stop doing

- Stop leading with Rust. Lead with the workflow outcome.
- Stop using generic NUTS speed as the main proof of value.
- Stop adding unrelated distributions before the structural IR exists.
- Stop adding stand-alone forecast classes that cannot compose.
- Stop treating an in-memory compiled object as a deployment artifact.
- Stop publishing single synthetic wins without the failures beside them.
- Stop excluding selection, import, compilation, or post-processing time when those are
  part of the user’s job.
- Stop calling the project general-purpose until its independent validation matches that
  phrase.
- Stop versioning toward 1.0 based on feature count. Version toward an evidence and
  stability gate.

## The next ten decisions

In order:

1. Adopt the operational structure-aware runtime thesis.
2. Decide the name and pre-1.0 compatibility promise.
3. Decide whether generic NUTS is delegated to `nuts-rs`, experimental, or a core
   research commitment.
4. Split the Python binding and establish one typed result schema.
5. Design the model IR, inference-plan contract, and slot-only artifact together.
6. Implement the marginalized, structured, square-root state-space foundation.
7. Express trend, multiple seasonality, regression, and robust observations as
   components on that foundation.
8. Ship batch, update, and backtest as first-class lifecycles.
9. Build the SBC, PosteriorDB, forecast-calibration, and operational benchmark program.
10. Recruit independent reviewers and real shadow-mode users before making a workplace
    reliability claim.

If these decisions are executed well, rustmc will be different enough not to compete
head-on, general enough to serve more than forecasting, and useful enough to earn a
place in serious work. The essential move is to make **structure, execution, and trust**
the product—not Rust, not NUTS, and not a longer list of probability distributions.
