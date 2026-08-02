# Short-series forecasting study

This directory contains a deterministic comparison of rustmc, LightGBM, a Gaussian
process, and Prophet on six synthetic univariate series. It is a diagnostic exercise,
not a claim that one method is universally more accurate.

## Bottom line

rustmc is already useful for short seasonal series, particularly at monthly frequency.
It produced the lowest MAE on the monthly medium and hard cases and was effectively tied
with Prophet on monthly easy. The fitted Bayesian seasonal model also returns coherent
future-observation draws from which a genuine posterior-predictive HDI can be calculated.

The study also found two important limitations:

1. The monthly medium series accelerated while its seasonal amplitude changed. rustmc's
   drift-adjusted seasonal model had good point accuracy but covered only one of six
   held-out observations. A joint Bayesian trend-plus-seasonality model is needed so
   drift uncertainty is propagated instead of estimated in preprocessing.
2. The current dense period-52 seasonal FFBS implementation took 272.15 seconds to fit a
   53-state weekly model. The exact conjugate AR(52) fallback fits and forecasts in about
   8 milliseconds, but it performed poorly on the hard multi-seasonal weekly series.
   Sparse seasonal state transitions are a high-priority optimization.

Across the six series, Prophet had the lowest unweighted mean MAE at 7.82. rustmc was
second at 9.09, narrowly ahead of the GP at 9.27 and LightGBM at 11.30. Weighting every
held-out observation equally instead of every series equally changes the order because
the weekly horizons contain more points: Prophet 8.18, GP 8.89, LightGBM 10.86, and
rustmc 11.13. One synthetic realization is far too little evidence for a general ranking.

## Experimental protocol

- Master seed: `20260802`.
- Monthly cases: 24 training observations and 6 held-out months.
- Weekly cases: 104 training observations and 26 held-out weeks, approximately six
  months.
- Easy adds a smooth trend, one seasonal component, and low independent noise.
- Medium adds curvature, changing seasonal amplitude or a second seasonal component,
  and autocorrelated noise.
- Hard adds a regime transition, multiple seasonal components, recurring pulses,
  heteroskedastic or autocorrelated noise, and training-window outliers.
- The entire stochastic series is generated before splitting. No surprise is inserted
  only into the test horizon.
- Candidate selection and preprocessing use the training split only. Held-out values are
  used only for the final metrics and plots.
- Reported runtime excludes imports and hyperparameter/model selection. It measures the
  final selected estimator's fit plus forecast. Feature construction outside an
  estimator's fit call is also excluded, so these are warm microbenchmarks rather than
  end-to-end service latency.
- Timings are one local run on an AMD Ryzen 9 5900X with 12 cores/24 threads and Linux
  x86-64. They should not be treated as portable performance claims.

The committed CSV files include `latent_mean` so the generator can be audited, but model
selection and final accuracy use the noisy observed `value` only.

## Point accuracy and final runtime

Each cell is `MAE / fit+forecast seconds`. Bold indicates the lowest held-out MAE in that
row; differences this small are not statistically significant with one generated series.

| Dataset | rustmc | LightGBM | Gaussian process | Prophet | Lowest MAE |
|---|---:|---:|---:|---:|---|
| Monthly easy | 1.471 / 0.7082 | 5.921 / 0.0198 | 2.022 / 0.0626 | **1.465 / 0.0582** | Prophet |
| Monthly medium | **9.355 / 0.6897** | 13.805 / 0.0148 | 20.353 / 0.0158 | 10.284 / 0.0777 | rustmc |
| Monthly hard | **6.671 / 0.6970** | 16.289 / 0.0205 | 7.218 / 0.0420 | 9.950 / 0.0653 | rustmc |
| Weekly easy | 6.122 / 0.0082 | 11.255 / 0.0933 | 3.283 / 0.6933 | **2.391 / 0.0437** | Prophet |
| Weekly medium | 10.433 / 0.0079 | 10.698 / 0.0722 | 12.912 / 0.6300 | **4.363 / 0.0447** | Prophet |
| Weekly hard | 20.516 / 0.0081 | 9.825 / 0.0704 | **9.814 / 1.1233** | 18.439 / 0.0523 | GP |

LightGBM is the fastest monthly point baseline but has very little information from which
to learn trees. The rustmc AR posterior is exceptionally fast weekly because it uses an
exact Normal-Inverse-Gamma calculation rather than iterative sampling. The roughly
0.70-second rustmc monthly fits use four chains with 1,000 retained draws each.

## Interval behavior

Each cell is `observed-value coverage / mean interval width`. LightGBM has no interval in
this study. Coverage is descriptive: six monthly observations provide very little
resolution, and none of these methods should be declared calibrated from this experiment.

| Dataset | rustmc 95% predictive HDI | GP 95% predictive interval | Prophet 95% uncertainty interval |
|---|---:|---:|---:|
| Monthly easy | 100.0% / 12.57 | 100.0% / 8.37 | 100.0% / 5.64 |
| Monthly medium | 16.7% / 14.66 | 16.7% / 26.15 | 16.7% / 10.54 |
| Monthly hard | 100.0% / 37.31 | 83.3% / 31.80 | 66.7% / 21.81 |
| Weekly easy | 100.0% / 25.17 | 100.0% / 14.59 | 100.0% / 11.66 |
| Weekly medium | 100.0% / 51.45 | 76.9% / 35.47 | 88.5% / 19.61 |
| Weekly hard | 88.5% / 75.10 | 88.5% / 40.07 | 42.3% / 27.91 |
| All 96 held-out points | 91.7% | 84.4% | 74.0% |

rustmc's intervals are actual pointwise shortest 95% intervals computed from 4,000
future-observation posterior-predictive draws. They are not latent-state intervals and
not simultaneous whole-path bands. The GP interval is empirical Bayes, conditional on
optimized kernel hyperparameters. Prophet uses a MAP fit plus its native uncertainty
simulation. Calling either comparison interval a fully Bayesian credible interval would
overstate what was fitted.

## Model construction

### rustmc

The training-only candidate set contains two local-level specifications, two local-linear
trend specifications, and Bayesian AR orders appropriate to each frequency. Base
candidates are selected by rolling-origin weighted interval score. Because exactly two
annual cycles leave no honest seasonal-model holdout, a predeclared detrended
cycle-correlation diagnostic decides whether monthly data use the fitted seasonal local
level. A median paired-cycle drift is removed and restored over the future horizon.

All three monthly datasets selected the drift-adjusted Bayesian seasonal local-level
model. All three weekly datasets selected a regularized Bayesian AR(52). Full priors,
selection origins, diagnostics, and candidate scores are recorded in
[`results/rustmc_results.json`](results/rustmc_results.json) and explained in
[`results/rustmc_method.md`](results/rustmc_method.md).

### LightGBM

One direct multi-horizon residual model is fitted per series. Inputs include lags,
rolling summaries, local trends, a seasonal reference, and Fourier calendar terms.
Four small tree configurations and three seasonal-drift blends are selected by
expanding-window training-only MAE. This is intentionally medium effort, but monthly
selection remains unstable with only 24 values.

### Gaussian process

The GP selects a constant or linear mean and smooth, seasonal, quasi-periodic, or
multi-seasonal kernels using rolling-origin Gaussian negative log predictive density.
One optimizer restart is used in the final fit. Several fitted hyperparameters reached
their configured bounds, illustrating how fragile plug-in kernel optimization can be on
two seasonal cycles.

### Prophet

Prophet selects additive or multiplicative custom seasonality and changepoint prior
scales `0.01`, `0.05`, or `0.2` using rolling-origin negative log predictive density.
It uses a MAP fit with 1,000 native uncertainty simulations for the final interval.

The candidate families and tuning budgets are deliberately reasonable rather than
exhaustive. They are not identical across engines because the goal is to compare useful
medium-effort workflows, not only default constructor calls.

## RustMC forecast figures

- [Monthly easy](images/monthly_easy_rustmc_forecast.png)
- [Monthly medium](images/monthly_medium_rustmc_forecast.png)
- [Monthly hard](images/monthly_hard_rustmc_forecast.png)
- [Weekly easy](images/weekly_easy_rustmc_forecast.png)
- [Weekly medium](images/weekly_medium_rustmc_forecast.png)
- [Weekly hard](images/weekly_hard_rustmc_forecast.png)

Each figure shows only the trailing 12 months of training history, followed by the
rustmc posterior mean and pointwise posterior-predictive HDI. The dashed red path is the
held-out future and was not available during model selection or fitting.

## Reproduction

Generate the exact data and rerun the rustmc arm:

```bash
python demo-docs/generate_datasets.py
python demo-docs/run_rustmc_forecast.py
python demo-docs/render_rustmc_plots.py
```

The comparison adapters require their optional dependencies:

```bash
python -m pip install lightgbm scikit-learn pandas prophet
python demo-docs/baselines/run_lightgbm_benchmark.py
python demo-docs/baselines/gp_prophet_baselines.py \
  --output demo-docs/results/gp_prophet_results.json
```

Measured package versions were rustmc 0.9.0, NumPy 2.4.2 for rustmc/GP/Prophet,
Matplotlib 3.10.8, scikit-learn 1.9.0, Prophet 1.3.0, and LightGBM 4.7.0. The LightGBM
environment used NumPy 2.5.1, pandas 3.0.5, and scikit-learn 1.9.0.

Machine-readable outputs live under [`results/`](results/). Source-only selection and
timing results are retained rather than copied into the library README as general product
claims.
