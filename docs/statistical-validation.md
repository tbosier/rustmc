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
| 50-d Gaussian, AR(1) correlation 0.8 | custom target | 2201 | 0.93 | 2.22 | 1.66 | 1742 | 1.0031 |
| 50-d Gaussian, AR(1) correlation 0.95 | custom target | 2201 | 0.92 | 1.18 | -0.33 | 696 | 1.0056 |
| 50-d Gaussian, AR(1) correlation 0.95 | graph | 2211 | 1.74 | 1.85 | 0.46 | 4204 | 1.0028 |
| 10-coefficient regression, known noise | graph | 2301 | 1.00 | 1.21 | 0.60 | 4070 | 1.0024 |

Errors are in MCSE; the mean and variance columns give the largest magnitude over
coordinates. The 0.95 correlated target's correlation matrix has condition number 989
and its covariance 1481 (77 and 116 at correlation 0.8, the case this gate started
with). The custom-target path (`target::sample_target`) gives every parameter a
diagonal metric; the graph path holds one vector parameter, for which `auto` may choose
a dense block. The regression's posterior is computed in closed form and its graph is
written the way a user writes one (`vector_normal_logp` prior, `mat_vec_mul`, Normal
likelihood with a constant scale). All six pass. The independent-normals graph case
passes with little room: its worst mean error is 3.93 against the bound of 4. The gate
does not check R-hat, which is a warning about a fit, not a measure of its exactness.

```text
cargo test --release -p rustmc_core --test exactness -- --nocapture
```

On the 0.95 correlated target the default `auto` metric chooses a dense block on the
graph path. At gate seeds 2211 to 2213, `auto` spent 32.5 to 35.2 leapfrog steps
(warmup included) per unit of the smallest bulk ESS, `diagonal` 455 to 841 and `dense`
25 to 27, with a smallest bulk ESS of 4204 to 4414 for `auto` and 381 to 793 for
`diagonal`. All nine fits but one diagonal fit (bulk ESS 381) pass the gate.

```text
cargo test --release -p rustmc_core --test exactness -- --ignored --nocapture \
    metric_choice_on_the_ill_conditioned_graph_target
```

Up to 0.14.0, `auto` stayed diagonal here, at 317 to 683 steps per unit of bulk ESS. It
counts a warmup window's effective draws to judge how noisy the window's correlation
estimate is, and it counted them as for a mean, `n (1 - rho) / (1 + rho)`, which on
this target left fewer effective draws than dimensions, so the dense estimate was never
trusted. It now counts them as for a covariance, `n (1 - rho^2) / (1 + rho^2)`.

That count was first tried for 0.14.0 and not kept, because on a curved
two-dimensional target, `x ~ Normal(0, 25)` and `y | x ~ Normal(0.5 (x^2 - 25), 1)`
written as one vector parameter, one of five seeds (12; 4 chains, 500 warmup, 1000
draws) went dense in its first window and fell from a smallest bulk ESS of 90 with 1
divergence to 16 with 66. Over 40 seeds (100 to 139) the two counts do no differently
there: the median smallest bulk ESS was 86.7 with the old count and 85.7 with the new,
the tenth-smallest 10.6 for both, and 12 against 13 fits had more than 10 divergences.
Every metric samples this target poorly: 21 of the 40 fits had an R-hat above 1.05
under either count, and 16 under `diagonal`. Which seed does worst moves with any
change to warmup.
`benchmarks/curved_metric_choice.py` prints the five-seed comparison.

```text
PYTHONPATH=.pybuild python -m benchmarks.curved_metric_choice
```

`benchmarks/auto_metric_targets.py` runs nine targets with a vector parameter, ten seeds
each, 4 chains and 1000 draws: isotropic regressions (50 and 100 dimensions),
regressions with predictors at correlation 0.9 (10 and 20) and with scales from 0.01 to
100 (6), a logistic regression with correlated predictors (8), the eight schools in
noncentered and centered form, and the curved target above. The table gives the median
leapfrog steps per unit of the smallest bulk ESS, warmup included, under `auto` with
each count, after 1000 and after 500 warmup iterations.

| Target | 1000 warmup: old count | new count | 500 warmup: old count | new count |
|---|---|---|---|---|
| Isotropic, d = 50 | 10.7 | 10.7 | 9.6 | 9.6 |
| Isotropic, d = 100 | 20.3 | 20.3 | 16.4 | 16.4 |
| Correlated regression, d = 10 | 10.1 | 10.2 | 8.8 | 8.4 |
| Correlated regression, d = 20 | 13.7 | 13.7 | 13.9 | 13.9 |
| Scaled correlated regression, d = 6 | 115.0 | 113.1 | 125.1 | 127.8 |
| Logistic regression, d = 8 | 10.4 | 10.4 | 9.2 | 8.9 |
| Eight schools, noncentered | 34.6 | 34.6 | 28.2 | 28.2 |
| Eight schools, centered | 920.0 | 1341.7 | 2183.8 | 1292.1 |
| Curved target | 1568.2 | 1668.6 | 1851.3 | 1807.3 |

Where the fits are sound the two counts cost the same to within a few percent; with
1000 warmup iterations the old count already cost what `dense` does on the correlated
and logistic regressions. The centered eight schools and the curved target fail under every metric
(divergences in most fits, R-hat up to 1.08 and 1.48 under `auto`), so their costs per
unit of ESS rank nothing; the centered model's median moves in opposite directions at
the two warmup lengths.

```text
PYTHONPATH=.pybuild python -m benchmarks.auto_metric_targets [--warmup 500]
```

### Power controls

The `fault-injection` cargo feature (off by default, compiled out otherwise) adds two
faults to the NUTS kernel. CI runs `rust_core/tests/exactness_power.rs`, which fits
every gate case at its gate seed and the two seeds after it (18 fits) under each fault
and requires a moment check to fail, meaning a finite standardized error of 4 MCSE or
more; failing only the ESS floor or a divergence check does not count.

- Biased multinomial merge: every merge of two subtrees, and every doubling, takes the
  newer subtree's proposal whatever the weights say. Required in at least 15 of the 18
  fits.
- Step size 30% too large without the Metropolis correction: after warmup the kernel
  integrates with 1.3 times the adapted step size and weights every non-divergent leaf
  equally. Required in all 18.

```text
cargo test --release -p rustmc_core --features fault-injection --test exactness_power
```

Both faults were caught in all 18 fits. Over 200 more seeds per case (from 20,000), the
gate caught:

| Case | Biased merge: caught | Per-coordinate checks alone | Overstep: caught | Per-coordinate checks alone |
|---|---|---|---|---|
| 100 independent normals, custom target | 188/200 | 18/200 | 200/200 | 157/200 |
| 100 independent normals, graph | 187/200 | 27/200 | 200/200 | 168/200 |
| 50-d correlation 0.8, custom target | 200/200 | 20/200 | 200/200 | 0/200 |
| 50-d correlation 0.95, custom target | 200/200 | 182/200 | 200/200 | 0/200 |
| 50-d correlation 0.95, graph | 198/200 | 11/200 | 200/200 | 158/200 |
| 10-coefficient regression, graph | 195/200 | 49/200 | 200/200 | 179/200 |

The per-coordinate checks alone missed most biased merges, and almost every overstep
on the correlated targets the custom path samples with a diagonal metric: there the too-large step distorts the stiff whitened
directions, which barely move any one coordinate's variance. The squared-norm check was
added for that reason. With it, the median size of the squared-norm error was 13.7 to
21.5 MCSE under the overstep (smallest 6.2) and 5.8 to 11.6 MCSE under the biased
merge. The biased merge is a mild fault on Gaussians, and the gate misses it in up to
13 fits in 200. That is why the CI test asks for 15 of 18 rather than all 18: from
these rates, all 18 would turn it red on about 37% of changes that move the draws with
nothing wrong, and 15 of 18 on about 0.06%. A correct sampler fails the gate on about
2% of fits, so it cannot reach 15. The run is recorded in
`benchmarks/results/2026-10-15-exactness-detection.log` (106 s).

```text
RUSTMC_EXACTNESS_SEEDS=200 cargo test --release -p rustmc_core --features fault-injection \
    --test exactness_power -- --ignored --nocapture detection_rate_over_seeds
```

### False-alarm rate

The correct sampler was run on every case for 2000 seeds (10,000 to 11,999). The
nominal rate is what the moment checks would give if every standardized error were an
independent standard normal: two checks per coordinate and the squared norm, each
beyond 4 with probability 6.3e-5. Expected nominal counts per 2000 fits are 25.3 for
100 dimensions, 12.8 for 50 and 2.6 for 10.

| Case | Fits failing the gate | Failing a moment check | Nominal | Failing the ESS floor |
|---|---|---|---|---|
| 100 independent normals, custom target | 53 (2.65%) | 53 (2.65%) | 1.27% | 0 |
| 100 independent normals, graph | 47 (2.35%) | 47 (2.35%) | 1.27% | 0 |
| 50-d correlation 0.8, custom target | 35 (1.75%) | 35 (1.75%) | 0.64% | 0 |
| 50-d correlation 0.95, custom target | 47 (2.35%) | 21 (1.05%) | 0.64% | 26 |
| 50-d correlation 0.95, graph | 17 (0.85%) | 17 (0.85%) | 0.64% | 0 |
| 10-coefficient regression, graph | 5 (0.25%) | 5 (0.25%) | 0.13% | 0 |

No fit diverged. The moment checks fail a correct sampler more often than nominal: 2.1,
1.9 and 2.7 times on the independent normals and the 0.8 correlated target, where the
counts are large enough to tell. The excesses of 1.6 and 1.3 times on the 0.95
correlated target and 1.9 times on the regression rest on 21, 17 and 5 failures and are
not clearly outside chance. Most of the excess is in the variance checks. Pooled over
coordinates and seeds, the mean errors are close to what a correct MCSE predicts (4.5%
to 4.9% beyond 2 MCSE against a nominal 4.55%, standard deviation 0.998 to 1.006),
though on the 0.8 correlated target 0.020% were beyond 4 against 0.0063%. The variance
errors have nearly the right spread (standard deviation 1.006 to 1.037) but a heavier
lower tail: 0.018% to 0.048% beyond 4 MCSE, three to eight times nominal, and all 133
variance failures the log reports were below the exact value (the 41 mean failures
split 22 below, 19 above; the log names only a fit's first failing check, and the
tallies count one more variance failure). The likely reason is that the squared deviations are right-skewed: a sample
whose squared deviations happen to be small also has a small estimated MCSE, so the
self-normalised error reaches far below zero more often than above. The ESS floor also
fails on its own on the 0.95 correlated target's custom path, which has a diagonal
metric and whose smallest bulk ESS at the defaults ranged from 410 to 720 over the ten
seeds of the cost record below.

Taking the six cases together, a correct sampler fails at least one of them about 10%
of the time. The gate seeds are fixed, so CI is deterministic, but a change that moves
the draws has about that chance of turning the gate red with nothing wrong. Before
treating a red gate as a bug, run the study on the change and compare it with this
table. The 12,000 fits took 535 s on a 24-thread AMD Ryzen 9 5900X; the tallies are in
`benchmarks/results/2026-10-15-exactness-false-alarm.json` and the output, with every
failing fit, in the `.log` beside it.

```text
RUSTMC_EXACTNESS_SEEDS=2000 RUSTMC_EXACTNESS_JSON=$PWD/false-alarm.json \
    cargo test --release -p rustmc_core --test exactness -- --ignored --nocapture \
    false_alarm_rate_over_seeds
```

### Cost

`benchmarks/results/2026-10-15-exactness.json` records, for each case over ten seeds,
the leapfrog steps in warmup and draw transitions of all four chains divided by the
smallest bulk ESS. The step-size searches warmup runs are not transitions and are not
counted. Medians, warmup included: 17.6 and 17.3 for the independent normals (custom,
graph), 106 for the 0.8 correlated target, 553 and 34.8 for the 0.95 one, and 13.4 for
the regression.

```text
RUSTMC_EXACTNESS_JSON=$PWD/benchmarks/results/2026-10-15-exactness.json \
    cargo test --release -p rustmc_core --test exactness -- --ignored --nocapture \
    exactness_cost_report
```

### Static HMC: an open finding

The same checks on static HMC at its default 15 leapfrog steps are ignored tests,
because HMC fails them. On the independent normals at seed 2102 the worst coordinate's
bulk ESS is 6 and the largest R-hat 1.78: the adapted step size puts 15 steps near a
full period of a standardized coordinate, so a transition returns almost to where it
started. On the 0.8 correlated target at seed 2202 every coordinate's bulk ESS is at
least 627, but the squared norm's is 110, below the floor, and the largest R-hat is
1.016. Whether that second failure is the same resonance has not been shown.

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
(`RUSTMC_RECOVERY_SEED_OFFSET`; the simulated data stay the same), 7 of the 20 runs at
0.80 had a failing case. Two positive cases failed, each time on the zero-divergence
requirement with exactly one divergence: non-centred eight schools in 6 of 20 runs and
the non-centred hierarchical Poisson in 2 of 20. Every other positive case passed all
20, and both negative controls were flagged in all 20 (at least 61 divergences each).
At 0.999, 13 of the 20 runs had a failing case, all of them negative controls missing
their signal: centred eight schools in 6 (5 with no divergences at all) and the centred
funnel in 11 (divergences present, R-hat and ESS within the healthy standard); every
positive case passed all 20. Each run's output is in
`benchmarks/results/2026-10-14-recovery-target-accept.log`. The seeded suite took
10.6 s at 0.80 and 61.5 s at 0.999 on one test thread, with 21 runs sharing the machine.

```text
for ta in 0.8 0.999; do for off in $(seq 0 20); do
  RUSTMC_RECOVERY_TARGET_ACCEPT=$ta RUSTMC_RECOVERY_SEED_OFFSET=$off \
    cargo test --release -p rustmc_core --test recovery_suite -- --nocapture --test-threads 1
done; done
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
