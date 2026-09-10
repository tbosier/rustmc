# Repeated forecasting calibration — 2026-09-09

Each model has 32 independently generated replicates, 24 training periods, and 4 forecast periods. Fits use 2 chains, 5000 warmup sweeps and 20000 retained draws per chain. Seed: 20260909.

Parameters are drawn independently from the stated fitted priors. NumPy generates the latent recursions and observations; rustmc prior/predictive methods are not used to generate data. These are prior-averaged calibration checks under correct specification, not evidence of calibration under arbitrary fixed parameters or misspecification. Every attempted fit, exception, seed, generating parameter, and parameter diagnostic is retained in the JSON companion.

Coverage uses pointwise central 90% predictive intervals. CRPS scores the empirical predictive distribution; lower is better, but scales differ across models. Reported Monte Carlo standard errors use independent replicates. Panel series are averaged within each replicate before computing standard errors. Discrete Poisson intervals can overcover nominal levels. Values are estimates without a calibration pass/fail gate.

| Model | Successful / attempted | Maximum R-hat | Minimum bulk / tail ESS | Fits R-hat > 1.01 | Fits bulk / tail ESS < 400 |
|---|---:|---:|---:|---:|---:|
| structural_gaussian | 32 / 32 | 1.0012 | 3732.2 / 4740.2 | 0 | 0 / 0 |
| structural_student_t | 32 / 32 | 1.0009 | 3346.6 / 5322.9 | 0 | 0 / 0 |
| dynamic_poisson | 32 / 32 | 1.0039 | 969.1 / 2096.2 | 0 | 0 / 0 |
| hierarchical_dynamic_gaussian | 32 / 32 | 1.0035 | 1228.3 / 2324.1 | 0 | 0 / 0 |

Diagnostics cover inferred variances and terminal states for structural models, and population/group coefficients and terminal states for dynamic GLMs. Deterministically constant parameters are excluded from aggregate diagnostic extrema, with their names retained per fit. Historical latent states and Student-t precision variables are not exhaustively diagnosed. The R-hat/ESS flag counts describe this exact schedule; flagged fits remain in the calibration tables and may need longer sampling. The stated diagnostic targets are R-hat <= 1.01 and both bulk/tail ESS >= 400 for every monitored nonconstant parameter; these are precision/convergence screening criteria, not a calibration success test.

## structural_gaussian

Observation forecasts (groups averaged for the panel):

| Horizon | Coverage ± MC SE | CRPS ± MC SE | Mean interval width |
|---:|---:|---:|---:|
| 1 | 0.906 ± 0.052 | 0.312 ± 0.059 | 1.978 |
| 2 | 0.875 ± 0.059 | 0.355 ± 0.051 | 2.093 |
| 3 | 0.906 ± 0.052 | 0.359 ± 0.044 | 2.194 |
| 4 | 0.875 ± 0.059 | 0.474 ± 0.060 | 2.297 |

## structural_student_t

Observation forecasts (groups averaged for the panel):

| Horizon | Coverage ± MC SE | CRPS ± MC SE | Mean interval width |
|---:|---:|---:|---:|
| 1 | 0.938 ± 0.043 | 0.345 ± 0.044 | 2.180 |
| 2 | 0.938 ± 0.043 | 0.375 ± 0.052 | 2.281 |
| 3 | 1.000 ± 0.000 | 0.321 ± 0.027 | 2.375 |
| 4 | 0.938 ± 0.043 | 0.373 ± 0.040 | 2.467 |

## dynamic_poisson

Observation forecasts (groups averaged for the panel):

| Horizon | Coverage ± MC SE | CRPS ± MC SE | Mean interval width |
|---:|---:|---:|---:|
| 1 | 1.000 ± 0.000 | 0.231 ± 0.040 | 2.188 |
| 2 | 1.000 ± 0.000 | 0.652 ± 0.090 | 3.656 |
| 3 | 0.969 ± 0.031 | 1.039 ± 0.147 | 4.688 |
| 4 | 1.000 ± 0.000 | 0.943 ± 0.122 | 5.812 |

## hierarchical_dynamic_gaussian

Observation forecasts (groups averaged for the panel):

| Horizon | Coverage ± MC SE | CRPS ± MC SE | Mean interval width |
|---:|---:|---:|---:|
| 1 | 0.927 ± 0.029 | 0.403 ± 0.026 | 2.388 |
| 2 | 0.948 ± 0.022 | 0.376 ± 0.019 | 2.434 |
| 3 | 0.833 ± 0.040 | 0.446 ± 0.034 | 2.387 |
| 4 | 0.854 ± 0.033 | 0.419 ± 0.031 | 2.352 |

Joint panel sum:

| Horizon | Coverage ± MC SE | CRPS ± MC SE | Mean interval width |
|---:|---:|---:|---:|
| 1 | 0.875 ± 0.059 | 0.831 ± 0.097 | 4.308 |
| 2 | 1.000 ± 0.000 | 0.611 ± 0.046 | 4.421 |
| 3 | 0.906 ± 0.052 | 0.866 ± 0.116 | 4.314 |
| 4 | 0.906 ± 0.052 | 0.689 ± 0.086 | 4.226 |

Series-level forecasts, with missingness labels:

| Series | Horizon | Coverage ± MC SE | CRPS |
|---|---:|---:|---:|
| group_0_dense | 1 | 0.906 ± 0.052 | 0.399 |
| group_0_dense | 2 | 0.906 ± 0.052 | 0.411 |
| group_0_dense | 3 | 0.844 ± 0.065 | 0.419 |
| group_0_dense | 4 | 0.781 ± 0.074 | 0.426 |
| group_1_half_observed | 1 | 0.969 ± 0.031 | 0.409 |
| group_1_half_observed | 2 | 0.969 ± 0.031 | 0.348 |
| group_1_half_observed | 3 | 0.812 ± 0.070 | 0.458 |
| group_1_half_observed | 4 | 0.906 ± 0.052 | 0.426 |
| group_2_quarter_observed | 1 | 0.906 ± 0.052 | 0.402 |
| group_2_quarter_observed | 2 | 0.969 ± 0.031 | 0.370 |
| group_2_quarter_observed | 3 | 0.844 ± 0.065 | 0.463 |
| group_2_quarter_observed | 4 | 0.875 ± 0.059 | 0.405 |

## Interpretation and reproducibility

With 32 independent scalar outcomes per horizon, nominal 90% coverage has approximate binomial MC SE 0.053; a rough two-standard-error band is wide. Horizons share training data and future paths, so their coverage estimates are correlated. No multiple-testing adjustment or universal calibration claim is made. Per-horizon zero estimated SE can occur when this small experiment covers every outcome; it is not certainty of perfect population coverage. CRPS uncertainty also reflects occasional high-count/tail outcomes.

The hierarchical experiment has three series, uncertain shared/group regression coefficients, group state innovations, and shared dynamic shocks. Two groups have 50% and 75% missing training values. Its aggregate intervals sum groups within aligned joint posterior draws. Structural models include two missing training periods; the count model includes one missing period and a zero-exposure training period.

Reproduce against the recorded source revision and rebuilt extension:

```bash
python benchmarks/calibrate_forecasting.py --replicates 32 --chains 2 --draws 20000 --warmup 5000 --history 24 --horizon 4 --seed 20260909 --output benchmarks/results/2026-09-09-calibration
```

Recorded source revision: `8be5d855da5393aa8758084745809a4a23ce1cfa`. Backend version: `0.12.0`; NumPy: `2.5.3`. Runtime figures include scoring/diagnostics and are not throughput benchmarks.
The JSON companion records the imported native module location and checksum. Use the matching package and source revision when reproducing the run.
