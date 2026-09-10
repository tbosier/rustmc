# Repeated forecasting calibration — 2026-09-09

Each model has 32 independently generated replicates, 24 training periods, and 4 forecast periods. Fits use 2 chains, 1000 warmup sweeps and 1000 retained draws per chain. Seed: 20260909.

Parameters are drawn independently from the stated fitted priors. NumPy generates the latent recursions and observations; rustmc prior/predictive methods are not used to generate data. These are prior-averaged calibration checks under correct specification, not evidence of calibration under arbitrary fixed parameters or misspecification. Every attempted fit, exception, seed, generating parameter, and parameter diagnostic is retained in the JSON companion.

Coverage uses pointwise central 90% predictive intervals. CRPS scores the empirical predictive distribution; lower is better, but scales differ across models. Reported Monte Carlo standard errors use independent replicates. Panel series are averaged within each replicate before computing standard errors. Discrete Poisson intervals can overcover nominal levels. Values are estimates without a calibration pass/fail gate.

| Model | Successful / attempted | Maximum R-hat | Minimum bulk / tail ESS | Fits R-hat > 1.01 | Fits bulk / tail ESS < 400 |
|---|---:|---:|---:|---:|---:|
| structural_gaussian | 32 / 32 | 1.0220 | 128.4 / 143.3 | 8 | 24 / 2 |
| structural_student_t | 32 / 32 | 1.0250 | 172.1 / 255.4 | 5 | 28 / 4 |
| dynamic_poisson | 32 / 32 | 1.0518 | 36.6 / 52.7 | 24 | 32 / 26 |
| hierarchical_dynamic_gaussian | 32 / 32 | 1.0539 | 38.6 / 99.6 | 29 | 32 / 32 |

Diagnostics cover inferred variances and terminal states for structural models, and population/group coefficients and terminal states for dynamic GLMs. Deterministically constant parameters are excluded from aggregate diagnostic extrema, with their names retained per fit. Historical latent states and Student-t precision variables are not exhaustively diagnosed. The R-hat/ESS flag counts describe this exact schedule; flagged fits remain in the calibration tables and may need longer sampling. The stated diagnostic targets are R-hat <= 1.01 and both bulk/tail ESS >= 400 for every monitored nonconstant parameter; these are precision/convergence screening criteria, not a calibration success test.

## structural_gaussian

Observation forecasts (groups averaged for the panel):

| Horizon | Coverage ± MC SE | CRPS ± MC SE | Mean interval width |
|---:|---:|---:|---:|
| 1 | 0.906 ± 0.052 | 0.312 ± 0.059 | 1.988 |
| 2 | 0.875 ± 0.059 | 0.357 ± 0.051 | 2.095 |
| 3 | 0.906 ± 0.052 | 0.362 ± 0.045 | 2.205 |
| 4 | 0.875 ± 0.059 | 0.476 ± 0.061 | 2.302 |

## structural_student_t

Observation forecasts (groups averaged for the panel):

| Horizon | Coverage ± MC SE | CRPS ± MC SE | Mean interval width |
|---:|---:|---:|---:|
| 1 | 0.938 ± 0.043 | 0.343 ± 0.044 | 2.176 |
| 2 | 0.938 ± 0.043 | 0.374 ± 0.052 | 2.282 |
| 3 | 1.000 ± 0.000 | 0.321 ± 0.027 | 2.366 |
| 4 | 0.938 ± 0.043 | 0.373 ± 0.040 | 2.478 |

## dynamic_poisson

Observation forecasts (groups averaged for the panel):

| Horizon | Coverage ± MC SE | CRPS ± MC SE | Mean interval width |
|---:|---:|---:|---:|
| 1 | 1.000 ± 0.000 | 0.231 ± 0.040 | 2.250 |
| 2 | 1.000 ± 0.000 | 0.654 ± 0.091 | 3.656 |
| 3 | 0.938 ± 0.043 | 1.039 ± 0.147 | 4.656 |
| 4 | 1.000 ± 0.000 | 0.937 ± 0.120 | 5.750 |

## hierarchical_dynamic_gaussian

Observation forecasts (groups averaged for the panel):

| Horizon | Coverage ± MC SE | CRPS ± MC SE | Mean interval width |
|---:|---:|---:|---:|
| 1 | 0.927 ± 0.029 | 0.404 ± 0.026 | 2.378 |
| 2 | 0.958 ± 0.020 | 0.379 ± 0.019 | 2.424 |
| 3 | 0.823 ± 0.040 | 0.446 ± 0.034 | 2.373 |
| 4 | 0.854 ± 0.033 | 0.419 ± 0.031 | 2.348 |

Joint panel sum:

| Horizon | Coverage ± MC SE | CRPS ± MC SE | Mean interval width |
|---:|---:|---:|---:|
| 1 | 0.875 ± 0.059 | 0.827 ± 0.096 | 4.318 |
| 2 | 1.000 ± 0.000 | 0.618 ± 0.047 | 4.422 |
| 3 | 0.906 ± 0.052 | 0.866 ± 0.115 | 4.306 |
| 4 | 0.906 ± 0.052 | 0.687 ± 0.084 | 4.201 |

Series-level forecasts, with missingness labels:

| Series | Horizon | Coverage ± MC SE | CRPS |
|---|---:|---:|---:|
| group_0_dense | 1 | 0.906 ± 0.052 | 0.400 |
| group_0_dense | 2 | 0.906 ± 0.052 | 0.413 |
| group_0_dense | 3 | 0.812 ± 0.070 | 0.421 |
| group_0_dense | 4 | 0.781 ± 0.074 | 0.425 |
| group_1_half_observed | 1 | 0.969 ± 0.031 | 0.412 |
| group_1_half_observed | 2 | 0.969 ± 0.031 | 0.348 |
| group_1_half_observed | 3 | 0.812 ± 0.070 | 0.456 |
| group_1_half_observed | 4 | 0.906 ± 0.052 | 0.427 |
| group_2_quarter_observed | 1 | 0.906 ± 0.052 | 0.401 |
| group_2_quarter_observed | 2 | 1.000 ± 0.000 | 0.376 |
| group_2_quarter_observed | 3 | 0.844 ± 0.065 | 0.461 |
| group_2_quarter_observed | 4 | 0.875 ± 0.059 | 0.405 |

## Interpretation and reproducibility

With 32 independent scalar outcomes per horizon, nominal 90% coverage has approximate binomial MC SE 0.053; a rough two-standard-error band is wide. Horizons share training data and future paths, so their coverage estimates are correlated. No multiple-testing adjustment or universal calibration claim is made. Per-horizon zero estimated SE can occur when this small experiment covers every outcome; it is not certainty of perfect population coverage. CRPS uncertainty also reflects occasional high-count/tail outcomes.

The hierarchical experiment has three series, uncertain shared/group regression coefficients, group state innovations, and shared dynamic shocks. Two groups have 50% and 75% missing training values. Its aggregate intervals sum groups within aligned joint posterior draws. Structural models include two missing training periods; the count model includes one missing period and a zero-exposure training period.

Reproduce against the recorded source revision and rebuilt extension:

```bash
python benchmarks/calibrate_forecasting.py --replicates 32 --chains 2 --draws 1000 --warmup 1000 --history 24 --horizon 4 --seed 20260909 --output benchmarks/results/2026-09-09-calibration-pilot
```

Recorded source revision: `8be5d855da5393aa8758084745809a4a23ce1cfa`. Backend version: `0.12.0`; NumPy: `2.5.3`. Runtime figures include scoring/diagnostics and are not throughput benchmarks.
The JSON companion records the imported native module location and checksum. Use the matching package and source revision when reproducing the run.
