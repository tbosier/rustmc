# Statistical release checks

This page describes the gate that has to pass before a release: a fixed set of models
whose exact posteriors are known analytically, refitted and compared against them. It
is for contributors and for anyone judging how much the samplers are checked. You do
not need it to use the library.

Run `python -m benchmarks.validate_posteriors --output /tmp/posteriors.json` after
building rustmc. The command returns a nonzero status if a fit or gate fails.

Four fixed datasets have analytic posterior moments: Normal location, correlated
Gaussian regression, Beta–Bernoulli, and Gamma–Poisson. Each runs under three seeds.
The run fails if any of those cases does not produce its three records, or if a case
outside that list is attempted; the report's `case_coverage` block records what ran
against what was required.

Checks require R-hat <= 1.01, bulk and tail ESS >= 400, no divergences, and bounded
errors in posterior means and covariance. Every parameter is checked separately,
against the full range each diagnostic can take rather than only against its threshold.
A missing, non-finite, non-numeric, or out-of-range R-hat or ESS for any single
parameter fails the run on its own and names that parameter in the report; the bounds
are derived from the estimators themselves, so a value outside them did not come from a
fit. Divergence telemetry must be one whole non-negative count per chain, since a sum of
counts can cancel and cannot show a missing chain. Failed values are kept in the report,
written as JSON null where they are not finite. These checks compare with the posterior
for the observed dataset, rather than only with the parameter that generated the data.

A separate experiment draws regression coefficients from the stated prior, simulates
data with NumPy, and refits each dataset. It retains parameter coverage and rank
histograms, every attempted fit, and numerical or convergence failures. Coverage uses
independent replicate counts and a stated conservative finite-sample bound. Rank
histograms are exploratory because the retained MCMC draws are autocorrelated.
A small experiment has limited power; passing is not proof of calibration for every model.

The JSON records source revision, native module hash, versions, command, seeds, and
reference values. CI runs 32 replicates and uploads the report even if a gate fails,
including failures while building a case, failures while simulating a replicate, and
diagnostics that cannot be serialized. No report is written if importing `rustmc`
fails, or if the trailing `git rev-parse` and native-hash calls raise.
The default local run uses 64. Increase the replicate count for a more precise study.

The coverage check passes when each coefficient's 90% interval coverage lies within
`sqrt(ln(4000) / (2n))` of the nominal 0.90, where `n` is the replicate count. At the
32 replicates CI runs, that accepts any coverage from 0.54 to 1.00, so the per-PR run
is a smoke check that catches only gross miscalibration. At the local default of 64
the accepted band is about 0.65 to 1.00, and 256 replicates narrow it to about 0.77 to
1.00. The band reaches 1.00 at every one of these counts, so over-wide intervals are
not caught at all. The report records the tolerance it used as `coverage_tolerance`.

The fixed reference cases above run at `target_accept=0.95`, not the shipped 0.80.
The exactness gate below runs at the shipped defaults. The Rust recovery suite runs at
the shipped `target_accept=0.8`, with its own draws, warmup and tree depths (2000 or
4000 draws, 1000 to 1500 warmup, tree depth 8 to 12), so it does not test the shipped draw and
warmup counts.

## Exactness at the shipped defaults

`rust_core/tests/exactness.rs` fits targets whose moments are known in closed form,
so the only error left is Monte Carlo error. Every fit uses `SamplerConfig::default()`
apart from the seed: NUTS, 4 chains, 500 warmup, 1000 draws, `target_accept=0.8`,
metric `auto`, tree depth 10. A fit passes when it has no divergences, the bulk ESS of
every series whose Monte Carlo standard error (MCSE) is used below is at least 400, and
these standardized errors are all below 4 MCSE:

- each coordinate's mean against its exact mean, with the MCSE the diagnostics report;
- each coordinate's variance: the draws of `(x - exact mean)^2` have expectation exactly
  the exact variance, so their mean is held to the same bound, with an MCSE from those
  squared draws' own effective sample size;
- once per fit, the squared norm of the whitened draw. Each target is
  `w = A x - c ~ Normal(0, I)`, so `|w|^2` has expectation exactly the dimension. A bias
  spread thinly over many directions adds up here while the noise grows only as the
  square root of the dimension.

| Target | Path | Gate seed | Worst mean error | Worst variance error | Squared norm error | Min bulk ESS | Max R-hat |
|---|---|---|---|---|---|---|---|
| 100 independent normals, scales 0.5 to 2 | custom target | 2101 | 2.22 | 2.56 | 1.69 | 4180 | 1.0053 |
| 100 independent normals | graph | 2111 | 3.93 | 2.34 | -0.15 | 4095 | 1.0050 |
| 50-d Gaussian, AR(1) correlation 0.95 | custom target | 2201 | 0.92 | 1.18 | -0.33 | 696 | 1.0056 |
| 50-d Gaussian, AR(1) correlation 0.95 | graph | 2211 | 1.63 | 2.11 | -0.50 | 606 | 1.0120 |
| 10-coefficient regression, known noise | graph | 2301 | 1.00 | 1.21 | 0.60 | 4070 | 1.0024 |

Errors are in MCSE. The correlated target's correlation matrix has condition number
989 and its covariance 1481. The custom-target path (`target::sample_target`) gives
every parameter a diagonal metric; the graph path holds one vector parameter, for which
`auto` may choose a dense block. The regression's posterior is computed in closed form
and its graph is written the way a user writes one (`vector_normal_logp` prior,
`mat_vec_mul`, Normal likelihood with a constant scale). All five pass. The gate does
not check R-hat: the correlated graph fit's 1.012 is above the 1.01 the release checks
use, and the R-hat a fit reports is a warning, not a measure of its exactness.

```text
cargo test --release -p rustmc_core --test exactness -- --nocapture
```

### Power controls

The `fault-injection` cargo feature (off by default, compiled out otherwise) adds two
faults to the NUTS kernel, and `rust_core/tests/exactness_power.rs` requires the gate
to fail a moment check under each, on every gate case at its gate seed and the two
seeds after it. CI runs it.

- Biased multinomial merge: every merge of two subtrees, and every doubling, takes the
  newer subtree's proposal whatever the weights say.
- Step size 30% too large without the Metropolis correction: after warmup the kernel
  integrates with 1.3 times the adapted step size and weights every non-divergent leaf
  equally.

```text
cargo test --release -p rustmc_core --features fault-injection --test exactness_power
```

Over 200 more seeds per case (from 20,000), the gate caught:

| Case | Biased merge: caught | Per-coordinate checks alone | Overstep: caught | Per-coordinate checks alone |
|---|---|---|---|---|
| 100 independent normals, custom target | 188/200 | 18/200 | 200/200 | 157/200 |
| 100 independent normals, graph | 187/200 | 27/200 | 200/200 | 168/200 |
| 50-d correlated, custom target | 200/200 | 182/200 | 200/200 | 0/200 |
| 50-d correlated, graph | 200/200 | 11/200 | 200/200 | 1/200 |
| 10-coefficient regression, graph | 195/200 | 49/200 | 200/200 | 179/200 |

The per-coordinate checks alone missed most biased merges and almost every overstep on
the correlated target: on that target the too-large step distorts the stiff whitened
directions, which barely move any one coordinate's variance. The squared-norm check was
added for that reason; with it the median squared-norm error under the overstep was 13.7
to 21.5 MCSE across cases, and under the biased merge 5.8 to 11.6 MCSE. The biased
merge is a mild fault on Gaussians, and the gate misses it in up to 13 fits in 200.

```text
RUSTMC_EXACTNESS_SEEDS=200 cargo test --release -p rustmc_core --features fault-injection \
    --test exactness_power -- --ignored --nocapture detection_rate_over_seeds
```

### False-alarm rate

The correct sampler was run on every case for 2000 seeds (10,000 to 11,999). The
nominal rate is what the moment checks would give if every standardized error were an
independent standard normal: two checks per coordinate and the squared norm, each
beyond 4 with probability 6.3e-5.

| Case | Fits failing the gate | Failing a moment check | Nominal | Failing the ESS floor |
|---|---|---|---|---|
| 100 independent normals, custom target | 53 (2.65%) | 53 (2.65%) | 1.27% | 0 |
| 100 independent normals, graph | 47 (2.35%) | 47 (2.35%) | 1.27% | 0 |
| 50-d correlated, custom target | 47 (2.35%) | 21 (1.05%) | 0.64% | 26 |
| 50-d correlated, graph | 31 (1.55%) | 18 (0.90%) | 0.64% | 13 |
| 10-coefficient regression, graph | 5 (0.25%) | 5 (0.25%) | 0.13% | 0 |

No fit diverged. The gate fails a correct sampler about twice as often as nominal, and
the excess is in the variance checks. Pooled over coordinates and seeds, the mean
errors behave as a correct MCSE predicts on the independent normals and the regression
(4.5% beyond 2 MCSE against a nominal 4.55%, standard deviation 1.00; on the
independent normals 0.0055% beyond 4 against 0.0063%), and only slightly wider on the correlated target (4.9% beyond
2, standard deviation 1.004 and 1.014). The variance errors have nearly the right
spread (standard deviation 1.006 to 1.037) but a heavier lower tail: 0.018% to 0.063%
beyond 4 MCSE, three to ten times nominal, and all 111 variance failures were below the
exact value (the 29 mean failures split 18 below, 11 above). The squared deviations are
right-skewed, so their standardized mean is left-skewed, more so where the ESS is
smaller. The ESS floor also fails on its own on the correlated target, whose smallest
bulk ESS at the defaults ranged from 410 to 942 over the ten seeds of the cost record
below.

Taking the five cases together, a correct sampler fails at least one of them about 9%
of the time. The gate seeds are fixed, so CI is deterministic, but a change that moves
the draws has about that chance of turning the gate red with nothing wrong. Before
treating a red gate as a bug, run the study on the change and compare it with this
table.

```text
RUSTMC_EXACTNESS_SEEDS=2000 cargo test --release -p rustmc_core --test exactness \
    -- --ignored --nocapture false_alarm_rate_over_seeds
```

The 10,000 fits took 440 s on a 24-thread AMD Ryzen 9 5900X.

### Cost

`benchmarks/results/2026-10-14-exactness.json` records, for each case over ten seeds,
the leapfrog steps in warmup and draw transitions of all four chains divided by the
smallest bulk ESS. The step-size searches warmup runs are not transitions and are not
counted. Medians, warmup included: 17.6 and 17.3 for the independent normals (custom,
graph), 553 and 519 for the correlated Gaussian, 13.4 for the regression.

```text
RUSTMC_EXACTNESS_JSON=$PWD/benchmarks/results/2026-10-14-exactness.json \
    cargo test --release -p rustmc_core --test exactness -- --ignored --nocapture \
    exactness_cost_report
```

### Static HMC: an open finding

The same checks on static HMC at its default 15 leapfrog steps are ignored tests,
because HMC fails them. On these well-conditioned targets the adapted step size puts 15
steps near a full period of a standardized Gaussian coordinate, so a transition returns
almost to where it started. At seed 2102 on the independent normals the worst
coordinate's bulk ESS is 6 and the largest R-hat 1.78; at seed 2202 on the correlated
target the worst bulk ESS is 86.

```text
cargo test --release -p rustmc_core --test exactness -- --ignored --nocapture hmc_
```

## Recovery suite at the shipped target acceptance

`rust_core/tests/recovery_suite.rs` ran every case at `target_accept=0.999` from when
the release gate began demanding zero divergences. Rerun at 0.80 with the same seeds,
draws, warmup and assertions, every case passed, so the suite now runs at 0.80:

| Case | Divergences | Max R-hat | Min bulk ESS | Min tail ESS |
|---|---|---|---|---|
| Intercept-only Gaussian | 0 | 1.0007 | 6077 | 5211 |
| Linear regression | 0 | 1.0009 | 7739 | 5559 |
| Logistic regression | 0 | 1.0011 | 3554 | 4013 |
| Poisson GLM | 0 | 1.0009 | 3619 | 4140 |
| AR(1)-style regression | 0 | 1.0020 | 3237 | 3308 |
| Ridge regression | 0 | 1.0007 | 10395 | 5997 |
| Non-centred partial pooling panel | 0 | 1.0056 | 921 | 1626 |
| Non-centred hierarchical Poisson | 0 | 1.0022 | 1303 | 2189 |
| Non-centred eight schools | 0 | 1.0011 | 2242 | 1879 |
| Non-centred funnel | 0 | 1.0002 | 14068 | 10308 |
| Centred eight schools (negative control, must be flagged) | 219 | 1.0178 | 310 | 155 |
| Centred funnel (negative control, must be flagged) | 1666 | 1.0356 | 97 | 29 |

That is one seeded run. With the sampler seed moved by 1 to 20
(`RUSTMC_RECOVERY_SEED_OFFSET`; the simulated data stay the same), at 0.80 two positive
cases failed some runs, each time on the zero-divergence requirement with exactly one
divergence: non-centred eight schools in 6 of 20 and the non-centred hierarchical
Poisson in 2 of 20. Every other positive case passed all 20, and both negative controls
were flagged in all 20 (at least 61 divergences each). At 0.999 every positive case
passed all 20, but the negative controls missed: centred eight schools in 6 of 20 (5
with no divergences at all) and the centred funnel in 11 of 20 (divergences present,
R-hat and ESS within the healthy standard). The suite takes 6.3 s at 0.80 and 28.8 s
at 0.999 on one test thread.

```text
RUSTMC_RECOVERY_TARGET_ACCEPT=0.999 RUSTMC_RECOVERY_SEED_OFFSET=1 \
    cargo test --release -p rustmc_core --test recovery_suite -- --nocapture --test-threads 1
```

Rust recovery tests enforce the same convergence thresholds for positive cases.
The Gaussian partial-pooling case uses noncentered effects. Centered funnel and
centered eight-schools cases remain negative controls: poor recovery must produce a
diagnostic signal. None of these checks establishes calibration under misspecification.

## Known limitation: local linear trend

None of these gates covers the forecasting samplers' mixing. The local-linear-trend
Gibbs sampler is known to mix slowly on its level and slope variances. ROADMAP.md
reports R-hat from 1.05 to 1.9 for them on a synthetic series with 4 chains, 300 warmup
and 500 draws; that figure is ROADMAP's, not a run of this page. The
`BayesianLocalLinearTrend` documentation describes the limitation.
