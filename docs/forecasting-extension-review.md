# Review of the six forecasting extensions

Reviewed on 2026-09-08 at commit `bb9d3cc` (version 0.10.0). This is a source-backed
review and implementation plan; the proposed APIs below are not implemented.

The highest-value first change is joint Bayesian regression inside the forecasting
model. The existing Gaussian state-space machinery can support it. Batch forecasting,
diagnostics, and Fourier seasonality can then reuse that work. Sparse observations and
payment development need explicit new likelihoods and data contracts.

| Requested addition | Current implementation | Recommended scope |
| --- | --- | --- |
| Exogenous regressors / time-varying design | Fixed observation vector; fitted APIs accept only a univariate observation array | Time-varying observation rows, proper Gaussian coefficient priors, joint posterior and future-design prediction |
| Independent batch fits | Generic NUTS/HMC batching exists; forecasting only parallelizes chains | Native `fit_batch`, ragged series, stable cell IDs, bounded parallelism, per-cell errors |
| Payment triangle | No cohort/lag model or compositional likelihood | Dedicated runoff model with censoring, uncertain totals, and draws aligned with accrual forecasts |
| Sparse observations | Generic Poisson, negative binomial, and lognormal exist; fitted state-space models are Gaussian | Hurdle lognormal/Gamma or compound Poisson-Gamma, chosen for the observation units and mechanism |
| Gibbs/FFBS diagnostics | Local-level, seasonal, and trend fits lack methods; hierarchical Gibbs already exposes them | Reuse parameter diagnostics; represent sampler-specific statistics as applicable or unavailable |
| Short seasonal histories | Requires both `T >= 2 * period` and `finite_count >= period + 2` | Remove the cycle-based gate with proper priors and validation; add regularized Fourier seasonality |

**1. Exogenous regression should be the first implementation.**

[The state-space model](../rust_core/src/state_space.rs) stores a single
`observation: Vec<f64>` (line 88); filtering reuses it at every time step (lines
385–409). The Python constructors also require a one-dimensional observation vector.
The fitted local-level, seasonal, and trend signatures at lines 5339, 5695, and 6163
of [the bindings](../python_bindings/src/lib.rs) have no design input. Forecast methods
also lack future covariates.

Use the model

```text
y[t] = Z[t] alpha[t] + epsilon[t]
alpha[t] = (level/trend/seasonal state, beta)
Z[t] = (structural observation row, X[t, :])
beta ~ Normal(prior_mean, prior_covariance)
```

Start with coefficients that are constant through time but uncertain. A varying
design does not require coefficients to follow a random walk. Augmenting the state
with beta uses an identity transition and exactly zero process noise for that block.
The current core already accepts positive-semidefinite process covariance and samples
degenerate Gaussian conditionals. Verify this particular augmentation against an
analytic regression posterior before relying on it.

Extend the observation-row representation to constant or per-time rows. Filtering,
smoothing, FFBS, forecasts, and cross-horizon covariance must all use the same indexing
contract. Initially keep transition and process matrices constant; arbitrary variation
in every system matrix is a separate extension. Time-varying design is a standard
state-space representation supported by [statsmodels](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.representation.Representation.html).

Proposed public calls are `model.fit(y, exog=X)` and
`fit.forecast(steps=H, exog=X_future)`, with explicit coefficient priors at construction.
Validate row counts, feature order, finite designs, and future horizon. Missing y values
keep their time positions. A missing future design must fail instead of being filled
with zeros or the last row. Preserve existing behavior when exogenous data is absent.

Retain beta, variance parameters, and the terminal structural state in each joint draw.
Use that same beta draw throughout a forecast path. Expose `regression_samples` and
`mean_samples` for the regression contribution and complete conditional mean; preserve
the existing meaning of structural component samples. Compute observation and cumulative
intervals from the resulting joint paths. Plugging in posterior mean beta, or sampling
beta independently of the fitted state, would still discard covariance and misrepresent
uncertainty. Forecasts are conditional on supplied future X; uncertain business inputs
require an explicit future scenario-draw contract. Calendar values themselves are known.

For larger designs, dense augmented FFBS costs roughly `O(T * (d + p)^3)` per iteration.
Keep the existing scalar and 2-state fast paths for models without X. A blocked Gaussian
beta update alternating with structural FFBS targets the same posterior and is a useful
optimization if measured batch workloads justify it; it must update beta inside every
Gibbs iteration, including observation-variance residual calculations.

Acceptance evidence: analytic joint moments with fixed variances; simulated coefficient
and variance recovery; forecast and cumulative moments including beta-state covariance;
missing observations; future feature validation; collinear/constant calendar columns
under proper priors; and deterministic chain ordering. Check interval calibration over
repeated datasets, rather than requiring every fitted interval to exceed an OLS interval.

**2. Batch forecasting is an extension of an existing capability.**

[`sample_batch_bound`](../rust_core/src/sampler.rs) already parallelizes independent
datasets at lines 348–394. It consumes a generic graph and cannot accept the specialized
forecasting models. Their current `.fit()` implementations parallelize chains instead.

Two existing batch behaviors should not become the new operational contract: the RNG
seed depends on the input position (line 372), so reordering or subsetting a batch
changes a cell's fit; and collecting `Result` values into one result (line 393) means
one failure prevents returning the successful results. Python binding validation also
collects inputs into a single fallible result before sampling.

Add a Rust batch executor around reusable single-chain forecasting kernels, with one
controlled pool across cells and chains. Release the Python GIL for the whole run.
Derive deterministic seeds from a documented stable encoding of `(seed, cell_id,
chain_id, fit_or_forecast)`; do not use a randomized hasher or list position. Preserve
input order in returned results and allow per-cell priors and ragged histories/designs.
Provide `errors="collect"` alongside fail-fast behavior, plus a thread limit. Independent
batching is distinct from the existing joint hierarchical mean model.

Result retention matters: 1,000 cells × 4 chains × 1,000 draws × 12 horizons × 8 bytes
is 384 MB for observation paths alone. Add chunked processing and an explicit choice
of retained draws versus summaries. Keep multiple chains available per cell for
diagnostics. Independence also means batch aggregation does not introduce shared shocks
or hierarchical dependence.

Acceptance evidence: same cell results when reordered, chunked, resumed, or run with
different thread counts; distinct cell RNG streams; correct ragged alignment; isolated
validation/numerical errors; and retained throughput/peak-memory measurements on a
representative many-cell workload.

**3. Native runoff needs a triangle model, not just a distribution.**

There is no payment-triangle implementation. For integer event counts, a
Dirichlet-multinomial is an appropriate candidate for overdispersed allocations across
lags. It is a distribution over integer counts with a total, as documented in
[Stan's distribution reference](https://mc-stan.org/docs/functions-reference/multivariate_discrete_distributions.html).
Dollar amounts should not be treated as multinomial counts, including by treating cents
as independent events. Use a continuous compositional allocation model, or count and
severity components if the underlying events are available.

Define origin/cohort dates, development lags, incremental versus cumulative input,
the valuation date, a future/unobserved mask, and an explicit tail bucket. Future triangle
cells are unobserved, whereas observed zero payments are data. A Dirichlet or logistic-normal
composition alone also needs an explicit treatment of exact observed zeros.

Incomplete rows do not reveal their ultimate total. Normalizing the observed part to
one would force all liability into elapsed lags. Jointly infer the remaining total and
lag allocation, or condition on caller-supplied totals when those are known. Specify
whether lag proportions vary by cohort, calendar, and cell, and how they are pooled.
These are statistical choices that a bare `dirichlet_multinomial_likelihood()` cannot make.

For a continuous allocation draw, coherent aggregation has the form

```text
payment[t, draw] = sum_cohorts ultimate[cohort, draw]
                              * share[cohort, t - cohort, draw]
```

Use the same joint draw for ultimate totals and allocations, and condition future
allocations on payments already observed. A counts model additionally draws the actual
multinomial allocations. If inferred historical accrual states drive unpaid cohorts,
retain or conditionally regenerate those states: [local-level fitting](../rust_core/src/bayesian_forecast.rs)
currently saves only the terminal level (lines 334–338), with a similar terminal-only
contract in the other dynamic fits.

Keep the domain API in a separate runoff module while sharing predictors, posterior
draw conventions, diagnostics, and batching. Acceptance evidence should include a small
complete triangle with an analytic reference, immature cohorts, zeros versus missing
cells, a nonzero tail, total conservation, and rolling valuation-date holdouts.

**4. Sparse observations require a new inference path as well as a density.**

The exhaustive [`ObsFamily` enum](../rust_core/src/graph.rs) at lines 48–54 has neither
a hurdle family nor compound Poisson-Gamma. Negative binomial covers counts; it does
not provide a continuous payment-amount likelihood. All three fitted dynamic models
currently use conjugate Gaussian observation updates.

For nonnegative amounts, a hurdle lognormal/Gamma is a practical first candidate:
one predictor governs payment occurrence and another governs positive size. Both
parts need priors and posterior predictive draws. A Poisson-Gamma compound sum, also
represented by a Tweedie distribution with `1 < p < 2`, is a good candidate when a
month's payment is a sum of random positive event amounts. This is different from
mixing a Poisson rate with a Gamma distribution, which produces negative-binomial
counts. The [mgcv Tweedie documentation](https://www.stat.ethz.ch/R-manual/R-devel/library/mgcv/html/Tweedie.html)
describes the compound construction and numerical density evaluation.

A static hurdle regression could initially reuse the existing Bernoulli-logit and
lognormal graph components. For a dynamic predictor, the occurrence and positive-size
likelihoods generally break exact Gaussian FFBS conjugacy. Choose an explicit validated
kernel or augmentation for that model; a new enum value alone cannot make the current
Gibbs loop correct. Marginalizing discrete indicators keeps them out of HMC, following
the [finite-mixture approach](https://mc-stan.org/docs/stan-users-guide/finite-mixtures.html).

Generic integration touches densities/gradients, graph metadata, compiled-model
serialization, binding support validation, prior/posterior predictive simulation, and
pointwise log likelihood. Acceptance evidence: exact zero probability, positive support,
analytic moments, finite-difference gradients for learned auxiliary parameters,
all-zero and one-positive histories, and calibration of zero frequency and positive tails.
For an all-zero cell, severity remains governed by its prior or shared information;
the API should report this rather than failing solely because no positive value occurred.

**5. Reuse the existing diagnostics, with accurate sampler semantics.**

[`compute_diagnostics`](../rust_core/src/diagnostics.rs) already calculates rank-normalized
folded split R-hat, bulk/tail ESS, MCSE, and HDIs. [Hierarchical Gibbs](../rust_core/src/hierarchical.rs)
adapts its draws to that engine at lines 118–149, and the Python fit exposes `summary()`
and `diagnostics()` at lines 4963–4991 of the bindings. Local-level, seasonal, and trend
fits only expose samples and ArviZ conversion today. This is relatively small work and
should accompany the regression feature.

Include all learned variance and coefficient parameters and the retained terminal
states; label the coverage explicitly if full historical states are not retained.
Keep chain axes intact. Return an unavailable diagnostic for insufficient draws and
distinguish a split diagnostic from evidence across independent chains.

Hamiltonian divergences measure failures of numerical Hamiltonian integration;
they are not defined for conjugate Gibbs/FFBS. See [Stan's explanation](https://mc-stan.org/docs/reference-manual/mcmc.html#divergent-transitions).
Use unavailable/null divergence and acceptance fields for Gibbs, with numerical
failures reported separately. Avoid extending the current hierarchical summary's
string replacement of a synthetic `acceptance=1, divergences=0` report. Parameter
diagnostics and sampler-specific telemetry should be separate structures.

Acceptance evidence: comparison with ArviZ on identical retained chains, chains with
different locations/scales, constant traces, short traces, and documented sampler metadata.

**6. The 24-month rule is removable; Fourier terms also reduce model size.**

[Seasonal validation](../rust_core/src/bayesian_seasonal.rs) enforces `T >= 2 * period`
at lines 350–356 and `finite_count >= period + 2` at lines 363–368. Both checks must be
addressed. The current tests explicitly lock in this restriction. The fixed-parameter
Kalman core has no corresponding cycle rule, and the fitted model has proper priors on
initial states and variances.

Replace the arbitrary cycle threshold with a documented finite-data requirement and
calibration/prior-sensitivity checks for short histories. This enables fitting; it does
not guarantee the data identify the seasonal pattern. Avoid prescribing one universal
calendar-length minimum for every choice of prior and seasonal dimension.

Fixed Fourier coefficients can reuse the Bayesian regression design from item 1:

```text
sin(2*pi*k*t/period), cos(2*pi*k*t/period), k = 1..K
```

Choose a small explicit K with shrinkage priors, preserve the time origin between
training and forecasting, and handle the even-period Nyquist term without an identically
zero sine column. Static harmonic seasonality is a different model from the existing
stochastic seasonal component; evolving Fourier amplitudes would need a later dynamic
state option. [Forecasting: Principles and Practice](https://otexts.com/fpp3/dhr.html)
explains how the number of harmonics controls seasonal smoothness.

The present dummy-seasonal state has dimension `period`, and its dense matrix
multiplications and FFBS factorizations scale cubically in that dimension. A truncated
Fourier model avoids tying state size to the full seasonal period. Validate 12- and
18-month annual histories, short period-52 histories, missing seasonal phases, known
calendar phase, confounding with intercept/trend, and interval calibration under weak data.

**One related correctness issue in the current example.**

At line 76 of [the rebate example](../examples/rebate_accrual_forecast.py),
`exp(forecast.state_samples)` is labeled an expected level. Under the example's model,
this is the conditional median/geometric level of payment; the conditional arithmetic
mean is `exp(level_draw + observation_variance_draw / 2)`. The observation predictive
draws are correctly back-transformed. Correct the label or the calculation before
using this example to validate mean payment or runoff amounts.

**Suggested implementation sequence.**

1. Time-varying observation design, joint Bayesian exogenous regression, and diagnostics
   on the affected fits. Extract touched binding code into focused modules as needed;
   a rewrite of the entire 6,937-line binding file is not a prerequisite.
2. Native independent batch fitting and forecasting, stable cell seeds, collected errors,
   and measured memory/throughput.
3. Short-history seasonal validation and regularized Fourier regressors, using the new
   regression contract. This can follow immediately after step 1 if short histories
   block the first representative dataset.
4. A sparse amount family and its explicit inference kernel, selected against observed
   zero frequency and positive-amount behavior.
5. Native cohort/lag runoff, connecting sparse observations, uncertain totals, and aligned
   posterior paths. Specify and validate the triangle likelihood before adding its API.

**Validation performed for this review.**

- `cargo test --workspace --release --offline`: 107 core unit tests and 12 recovery tests
  passed; workspace binding and documentation test targets also passed.
- Rebuilt and installed the current source into the existing project `.venv` with
  `.venv/bin/maturin develop --release --offline`.
- `.venv/bin/python -m pytest -q`: 140 passed, 5 skipped, 1 deselected.
- `cargo fmt --all -- --check`: passed.
- Probed all three fitted dynamic APIs: each rejects `exog`, lacks `fit_batch`, and its
  fit lacks `summary`/`diagnostics`. Period-12 fitting rejects lengths 12, 18, and 23;
  it accepts 24. Eighteen finite values followed by six NaNs also pass the existing
  length gate. Padding changes the terminal time, so it is not a valid workaround.

These checks validate the current baseline and the reported API gaps. They do not
establish calibration, performance, or correctness of the proposed extensions.
