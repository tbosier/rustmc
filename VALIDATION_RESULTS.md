# Statistical Validation Results

Results of the statistical validation suite added in `Task 3: statistical
validation suite`. Everything below was measured on this branch; nothing is
projected or assumed.

**Bottom line.** The inference engine is statistically correct on every model
family it supports, verified against closed-form posteriors and by
simulation-based calibration. Two real defects were found in the *Python
bindings* (one of which makes hierarchical models unusable) and one genuine
sampling miscalibration was reproduced and quantified (the centered
hierarchical parameterisation). No bias was found in the autodiff, in any
log-density, in any constraining transform, or in the NUTS kernel on
well-conditioned targets.

---

## 1. What the suite is

| File | Tests | What it establishes |
|---|---|---|
| `rust_core/tests/gradient_check.rs` | 8 | `grad_logp` equals the finite-difference gradient of `eval_logp` for every op, prior, likelihood and transform |
| `rust_core/tests/prior_recovery.rs` | 15 | prior-only sampling reproduces each family's closed-form mean and sd — i.e. every constraining transform's Jacobian is right |
| `rust_core/tests/analytic_posterior.rs` | 6 | posterior mean **and** sd match exact conjugate / linear-Gaussian references |
| `rust_core/tests/sbc.rs` | 3 fast + 6 opt-in | simulation-based calibration; posterior credible-interval coverage; a live negative control |
| `rust_core/tests/likelihood_recovery.rs` | 5 | parameter recovery for Exponential / LogNormal / NegativeBinomial GLMs, high-dimensional `MatVecMul`, hierarchical shrinkage |
| `rust_core/tests/numerical_stability.rs` | 12 | determinism, thread-independence, extreme scales, `n = 1`, large `n`, collinearity, shape-error handling |
| `tests/test_statistical_conjugate.py` | 6 | the same analytic references reached through the public Python DSL |
| `tests/test_statistical_predictive.py` | 7 | prior-predictive moments, posterior-predictive interval coverage, analytic predictive checks |
| `tests/test_statistical_engine_bugs.py` | 10 | pinned defects (xfail) plus failure-mode assertions |

## 2. Tolerance methodology

No tolerance in this suite was chosen by running the test and widening until it
passed. Every one is derived:

* **Posterior mean.** `|mean_hat - exact| <= 4 * mcse_mean`, using the MCSE the
  engine itself reports (`posterior sd / sqrt(ess_bulk)`). A single such
  assertion has a ~6e-5 two-sided false-alarm rate.
* **Posterior sd.** `|sd_hat - exact| <= 5 * exact * sqrt((kurtosis - 1) / (4 * ess_bulk))`,
  the delta-method standard error of a sample standard deviation. For a
  Gaussian (`kurtosis = 3`) this is `5 * exact / sqrt(2 * ess)`. The true
  kurtosis is supplied for skewed families (LogNormal, Exponential, Gamma,
  Beta, Uniform, StudentT, HalfNormal) so heavy tails widen the bound honestly
  rather than by fiat. **Detectable effect size:** at `ess = 2000` this is a
  ~7.9% relative sd error, so a `sqrt(2)`- or `2x`-style scale bug is caught
  and a 1% one is not.
* **Every MCSE-based assertion also gates on `ess_bulk`**, because a tolerance
  derived from an unreliable MCSE would be vacuous.
* **Finite differences.** Relative `1e-5` plus an absolute floor plus an
  explicit cancellation term `8 * eps * |logp| / h`, which dominates at extreme
  parameter values where `|logp|` reaches `1e8`.
* **SBC.** Chi-square against uniformity, `df = 7`, rejecting below `p = 1e-3`.
* **Coverage.** 3.5 binomial standard errors, or (for the Python
  posterior-predictive test) 4 *empirical* standard errors measured across
  independent replicates.
* **Recovery tests.** `|mean - truth| <= 3 * posterior_sd + 4 * mcse`: three
  posterior sds is the sampling variability of the estimate given one seeded
  dataset.

## 3. Results

### 3.1 Autodiff — clean

All 8 gradient-check tests pass. Every `Op`, every prior family, every
`Vector*LogP`, every observation family, `MatVecMul` with and without an
intercept, hierarchical wiring, and extreme parameter values (linear predictors
at ±25, `sigma` from 1.2e-4 to 8100) agree with central differences.

### 3.2 Constraining transforms — clean

All 15 prior-recovery tests pass. Sampling from a prior-only model reproduces
the closed-form mean and sd for Normal, HalfNormal, LogNormal, Exponential,
Gamma, Beta, Uniform and StudentT, and for all six `Vector*LogP` ops. This is
the test that catches a wrong or missing Jacobian; none is wrong.

### 3.3 Analytic posteriors — clean

All 6 pass with **zero post-warmup divergences**:

* conjugate Normal-Normal (data-dominated and prior-dominated),
* linear Gaussian regression through `FusedLinearMu`,
* the same posterior through faer `MatVecMul` + `VectorNormalLogP`,
* a two-level linear-Gaussian hierarchy with known variance components
  (7 parameters, exact joint Gaussian, every marginal mean and sd asserted),
* near-collinear predictors (empirical correlation ~0.999).

### 3.4 Simulation-based calibration

Full run: `cargo test -p rustmc_core --test sbc --release -- --ignored --nocapture`
(512 replicates, `L = 63` draws thinned by 8, 8 rank bins, `df = 7`).

```
model                              param     chi2    p        coverage (nom 75% / 91%)
location N(0,1) / N(mu,1)          mu        9.00    0.2527   0.770 / 0.893
scale HalfNormal(1) / N(0,sigma)   sigma     9.31    0.2310   0.723 / 0.912
regression a + b x, learned sigma  a         6.72    0.4587   0.742 / 0.898
                                   b        12.56    0.0835   0.762 / 0.906
                                   sigma     5.44    0.6067   0.766 / 0.896
hierarchical NON-centered          mu        4.19    0.7579   0.752 / 0.908
                                   tau       3.56    0.8286   0.744 / 0.904
                                   z_0       9.78    0.2013   0.732 / 0.916
matvec X @ beta                    beta[0]   2.47    0.9294   0.758 / 0.916
                                   beta[1]   7.41    0.3878   0.764 / 0.914
hierarchical CENTERED (funnel)     mu       13.16    0.0684   0.719 / 0.896
                                   tau      51.75    1.5e-8   0.656 / 0.805   <-- MISCALIBRATED
                                   theta_0   3.53    0.8319   0.734 / 0.918
```

Everything except the centered funnel is calibrated. Minimum p-value among the
calibrated models is 0.068.

**SBC power.** The chi-square non-centrality is
`N * sum_k (p_k - 1/8)^2 / (1/8)`; rejection at `p < 1e-3` needs `chi2 > 24.3`:

| perturbation | detected at 128 reps | at 512 reps |
|---|---|---|
| posterior mean shifted by `d` sd | `d >= 0.45` | `d >= 0.22` |
| posterior sd inflated by factor `c` | `c >= 1.5` | `c >= 1.22` |

So SBC here rules out gross bias, not a 5% scale error. That limit is stated in
the test module and is a property of the replicate count, not of the assertion.

### 3.5 Predictive calibration — clean

* Prior draws for all eight scalar families match closed-form mean and sd
  (40,000 draws each). `sample_prior_raw` is a *separate* implementation from
  the sampler, so this is an independent check.
* The prior predictive of `y` for `mu ~ N(m0, s0)`, `y ~ N(mu, sigma)` matches
  `N(m0, sqrt(s0^2 + sigma^2))`.
* Bernoulli / Poisson / Exponential prior-predictive draws respect their
  support.
* Posterior-predictive central 50% and 90% intervals attain nominal coverage
  across 16 replicates whose true coefficients are drawn from the prior (so
  nominal coverage is exact).
* Posterior-predictive means and sds match the analytic predictive
  `N(x_i' m, sigma^2 + x_i' C x_i)` observation by observation.

### 3.6 Numerical stability and determinism — clean

All 12 pass: bit-identical draws for a fixed seed; results independent of the
Rayon thread count; observation sd from `1e-4` to `1e4`; `n = 1`; `n = 6000`;
perfectly collinear predictors (posterior still exactly Gaussian and matched);
predictors of magnitude `1e6`; shape errors rejected with actionable messages
before sampling.

---

## 4. Defects found (pinned, not fixed)

This worktree owns tests only. Each defect has a test that reproduces it.

### DEFECT 1 — hierarchical models cannot be fitted through the Python DSL (blocking)

`ValueError: Unknown param: mu_0`

`build_prior_into_graph`'s auto-non-centering path (`should_auto_noncenter`,
`python_bindings/src/lib.rs`) registers the sampled parameter as
`"<name>__raw"` and stores the derived value node only in `value_node_map`.
`build_mu_expr` resolves parameter names through `Graph::node_by_name`, which
only sees graph-level parameter nodes, so any attempt to use a hierarchical
parameter in a likelihood fails.

Consequence: **every** hierarchical model expressible in the DSL is broken,
including `examples/hierarchical_example.py` and the shipped
`examples/hierarchical_templates.py` template. Both the centered
(`normal_prior(name, mu=ParamRef, sigma=ParamRef)`) and the constant-sigma
(`normal_prior(name, mu=ParamRef, sigma=1.0)`) forms fail.

Pinned by:
* `tests/test_statistical_engine_bugs.py::test_hierarchical_parameter_can_be_used_in_a_likelihood`
* `tests/test_statistical_conjugate.py::test_hierarchical_known_variances_matches_analytic_posterior`
  (the analytic reference the fix should be validated against)

This is Task 1's parameter-resolution area.

### DEFECT 2 — `sample_prior_predictive` panics on auto-promoted vector parameters

`pyo3_runtime.PanicException: index out of bounds: the len is 1 but the index is 1`

`sample_prior_raw` pushes exactly one raw value for `PriorSpec::Normal`,
`HalfNormal`, `StudentT`, `Uniform`, `Gamma` and `Beta` regardless of
`auto_vector_params`, so when a scalar prior has been promoted to length `n` by
the `@` operator the raw vector is `n - 1` values short and
`derive_display_draw` indexes past the end. Only the `Exponential` and
`LogNormal` branches loop over `n`; `PriorSpec::VectorNormal` (the explicit
`vector_normal_prior`) is correct.

It surfaces as a Rust panic rather than a Python exception.

Pinned by `tests/test_statistical_engine_bugs.py::test_prior_predictive_supports_auto_promoted_vector_parameters`,
with `test_prior_predictive_works_with_an_explicit_vector_prior` as the
positive control that localises it.

### DEFECT 3 — the reported divergence count includes warmup

`SampleResult::divergences` (and hence `DiagnosticsReport::divergences`,
`FitResult.divergences()`, and the `summary()` warning) accumulates
`n_divergences` over **all** iterations in `nuts::run_chain`, warmup included.
Stan and PyMC report post-warmup divergences only.

Measured impact: on the 1-parameter conjugate Normal-Normal model the reported
count is 21 while the true post-warmup count is **0**. On the HalfNormal
prior-only model it is ~230 warmup vs ~17 post-warmup. Users following the
documented "divergences indicate unreliable results" advice will chase
non-problems, and the existing `recovery_suite.rs` thresholds (up to 130
allowed divergences) are inflated for the same reason.

The per-transition flags needed to split the count already exist in
`SampleResult::transitions` (`TransitionStats { is_warmup, divergent }`); the
new `common::divergence_split` helper does exactly this.

Pinned by `tests/test_statistical_engine_bugs.py::test_divergence_count_excludes_warmup`.

### DEFECT 4 — `to_arviz()` discards per-draw divergence flags

`FitResult::to_arviz` warns that "exact per-draw divergence flags are not
stored" and omits `sample_stats["diverging"]`. They *are* stored:
`self.raw_result.transitions[chain][i].divergent` with `is_warmup` to select
the post-warmup subset. As shipped, `az.plot_pair(divergences=True)` and every
ArviZ divergence diagnostic are unavailable.

Pinned by `tests/test_statistical_engine_bugs.py::test_arviz_export_includes_per_draw_divergence_flags`.

### OBSERVATION 5 — NUTS rejects the whole transition on divergence

`nuts::run_chain`:

```rust
if !tree_stats.diverging {
    current.q.copy_from_slice(&proposal.q);
    ...
}
```

When any subtree diverges, `build_tree_iterative` breaks *and* the caller
discards the proposal accumulated from the already-valid subtrees, leaving the
chain at its starting point. Stan stops the doubling but keeps the proposal
from the valid part of the trajectory. Whether a trajectory "diverges" depends
on `h0 = H(initial)`, so this rejection is not symmetric across the trajectory
and is not obviously reversible.

**Evidence bound:** SBC at 512 replicates finds no calibration failure on any
model with a low divergence rate, and the analytic-posterior tests run at zero
post-warmup divergences. So if this introduces bias, it is below the ~0.22 sd /
~1.22x detection threshold on those models. It is called out because it is a
deviation from the reference algorithm, not because a bias has been measured
here.

### OBSERVATION 6 — the accept statistic fed to dual averaging is non-standard

`build_tree_iterative` accumulates
`sum_accept_stat += subtree.log_sum_weight.exp().min(n_leaves)`, i.e.
`min(sum_i w_i, n_leaves)` rather than Stan's `sum_i min(1, w_i)`. Leaves from
a divergent subtree contribute to neither sum. The two statistics differ
whenever the trajectory has both very high and very low weight leaves, so the
step size dual averaging converges to is targeting a slightly different
quantity than `target_accept`. This is consistent with the observed warmup
divergence rates (10-20% of warmup iterations on log-scale parameters) and the
wide spread of adapted step sizes across chains (0.38 to 1.06 on an identical
1-parameter model).

Not a correctness bug — the post-warmup chains are calibrated — but it is the
most likely explanation for the divergence counts and for the modest ESS/step
sizes.

---

## 5. Reproduced miscalibration: the centered hierarchical parameterisation

The centered form `theta_j ~ N(mu, tau)` with `tau ~ HalfNormal(1)` is the
classic Neal funnel. SBC at 512 replicates:

```
tau: chi2 = 51.75, df = 7, p = 1.5e-8
     rank histogram [113, 50, 49, 46, 60, 57, 74, 63]  (uniform would be ~64 each)
     central 75% credible interval covers 65.6%  (4.9 binomial se low)
     central 91% credible interval covers 80.5%
```

The excess in the lowest rank bin means the true `tau` falls *below* the
sampled posterior far more often than it should: the sampler cannot reach the
neck of the funnel, so `tau`'s posterior is truncated from below and its
credible intervals under-cover. `mu` and `theta_0` are calibrated.

The **non-centered** version of the same model is fully calibrated (see the
table above), which localises this to the parameterisation rather than to the
engine. Stan has the same limitation with the same model.

Why it matters here: `examples/hierarchical_templates.py` builds the *centered*
form, and the DSL's automatic non-centering — which would avoid this — is
unreachable because of DEFECT 1. So today the only hierarchical
parameterisation rustmc documents is the miscalibrated one, and it does not run
at all.

Characterisation test:
`rust_core/tests/sbc.rs::sbc_centered_hierarchical_model_is_known_miscalibrated`
(opt-in). It asserts the under-coverage is *still present*; if it starts
passing, the geometry handling improved and the test plus the hierarchical docs
should be updated.

---

## 6. Negative-control verification

To confirm the suite can actually detect the failures it targets, a 15% scale
error was injected into `Normal::prior` in `rust_core/src/distributions.rs`
(`add_constant(sigma)` -> `add_constant(sigma * 1.15)`) and the suite re-run:

```
prior_recovery::normal_prior_reproduces_its_own_distribution        FAILED
  x: posterior sd 1.599142 differs from analytic 1.400000 by 0.199142
     (14.22% relative), tolerance 0.063912 (4.57% relative, ess_bulk 5998)

prior_recovery::hierarchical_prior_marginal_matches_closed_form     FAILED
  mu: posterior sd 1.770815 differs from analytic 1.500000 by 0.270815
      (18.05% relative), tolerance 0.157605 (10.51% relative, ess_bulk 1132)

analytic_posterior::conjugate_normal_normal_prior_dominated_...     FAILED
```

The change was then reverted (`git diff rust_core/src/` is empty). SBC at 128
replicates did **not** flag the 15% error, exactly as the documented power
table predicts — which is itself a useful confirmation that the stated power
limits are real rather than optimistic.

Separately, `sbc.rs::sbc_detects_an_injected_bias` is a permanent, always-on
negative control: it ranks the true value against posterior draws shifted by
0.7 posterior sd and requires SBC to reject (measured `chi2 = 65.88`,
`p = 1.0e-11`).

---

## 7. Runtime

Debug (`cargo test --all`, measured on this branch):

| target | tests | time |
|---|---|---|
| `rustmc_core` unit tests | 27 | 0.03 s |
| `gradient_check` | 8 | < 0.01 s |
| `prior_recovery` | 15 | 1.1 s |
| `sbc` (fast tests only) | 3 | 9.7 s |
| `numerical_stability` | 12 | 26.0 s |
| `analytic_posterior` | 6 | 41.6 s |
| `likelihood_recovery` | 5 | 44.0 s |
| `recovery_suite` (pre-existing) | 12 | 132.0 s |
| **total** | **88** | **~254 s** |

Before this branch: 39 tests, ~132 s. After: 88 tests, ~254 s — so the suite
roughly doubled in both count and wall time, and 49 of the 51 new tests
complete in under 45 s each.

The six opt-in SBC tests add ~40 s in release and are excluded by default:

```
cargo test -p rustmc_core --test sbc --release -- --ignored --nocapture
```

Python: `pytest tests/` is **18 passed, 5 xfailed in 4.8 s** against a
`maturin develop --release` build.

---

## 8. What is still not covered

* **Hierarchical models through the Python DSL** — blocked by DEFECT 1. The
  Rust-level hierarchical tests (analytic posterior, SBC, shrinkage) cover the
  engine; the binding path has an xfail placeholder waiting on the fix.
* **Hierarchical vector parameters** — `normal_prior` with a `ParamRef`
  hyperparameter combined with `@` is explicitly rejected by the bindings, so
  there is nothing to test.
* **`batch_sample`** — no statistical validation. It shares `build_prior_into_graph`
  and the sampler with `sample()`, but the per-model result assembly is
  separate code.
* **`compiled_model.rs`** — serialization round-trips are not statistically
  validated (does a reloaded artifact sample the same posterior?).
* **StudentT and Uniform likelihoods** — the DSL has StudentT/Uniform *priors*
  but no corresponding observation families, so `VALIDATION.md`'s
  "heavy-tailed regression" item is not yet testable.
* **SBC precision** — the current replicate counts detect ~0.22 sd location
  bias and ~1.22x scale bias. Detecting a 5% scale error needs ~10,000
  replicates, which is a nightly job rather than a CI job.
* **Multi-likelihood models** — models with several observation heads are
  exercised incidentally but have no dedicated calibration test.
* **`ess_tail` and HDI** — asserted to be finite but not validated against
  reference implementations.

---

## 9. DSL support: spec versus reality

`VALIDATION.md` and the task brief assume a prior-family x likelihood-family x
scalar/vector/matrix/hierarchical cross-product. What the bindings actually
support today:

**Priors** (`ModelBuilder`): `normal`, `half_normal`, `exponential`,
`log_normal`, `student_t`, `uniform`, `bernoulli`, `poisson`, `gamma`, `beta`,
`vector_normal`.

**Hyperparameters as `ParamRef`**: only `normal_prior(mu, sigma)`,
`half_normal_prior(sigma)`, `exponential_prior(rate)` and
`log_normal_prior(mu, sigma)`. `student_t`, `uniform`, `gamma` and `beta` take
`f64` only — no hierarchical variants.

**Likelihoods**: `normal`, `bernoulli_logit`, `poisson_log`, `exponential`,
`log_normal`, `negative_binomial`. There is **no** StudentT, Uniform, Gamma,
Beta or Binomial likelihood, so those prior families can only ever appear as
priors.

**`sigma` as a `ParamRef`**: supported for `normal_likelihood`,
`log_normal_likelihood`, and `negative_binomial_likelihood` (as `alpha`).

**Vector parameters**: explicit via `vector_normal_prior`, or auto-promoted
from any continuous scalar prior used with `@`. Auto-promotion requires
constant hyperparameters — a hierarchical vector parameter is rejected with an
explicit error.

**Expression algebra**: `ParamRef * "key"`, `ParamRef @ "key"`,
`VectorParamRef @ "key"`, `+` between expressions / params / floats. There is
no subtraction, no scalar multiplication of a parameter by a number, no
interaction terms, and no elementwise product of two parameters.

**Not supported at all**: multivariate normal priors, covariance/correlation
parameters, LKJ, simplex or ordered constraints, censoring/truncation, mixtures,
missing data, offsets/exposures (a Poisson offset must be folded into the data),
non-identity links beyond the fixed per-family link, and time-series structures
(an AR(1) has to be written as a lagged regression by hand).

The single largest gap between spec and reality is that **hierarchical models
are advertised and documented but do not run** (DEFECT 1).
