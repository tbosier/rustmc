"""Forecast calibration in a rolling backtest

A backtest that reports only point error says whose forecast was closest, not
whether the uncertainty each forecaster claimed was honest. This example backtests a
Bayesian seasonal local-level model against the seasonal naive baseline on
synthetic monthly data and compares them on point accuracy, interval coverage,
interval width and PIT values.

The data are generated from a seasonal local-level process, so the model is close
to correctly specified here. On real data neither forecaster has that advantage.
"""
import numpy as np

import rustmc as r

# %% Simulate monthly data and backtest at rolling origins
rng = np.random.default_rng(2026)
period, months = 12, 84
pattern = np.array([-8.0, -5.0, -2.0, 1.0, 4.0, 7.0, 10.0, 8.0, 4.0, 0.0, -4.0, -5.0])
pattern -= pattern.mean()
level = 100.0 + np.cumsum(rng.normal(0.0, 1.0, months))
y = level + pattern[np.arange(months) % period] + rng.normal(0.0, 2.0, months)

model = r.BayesianSeasonalLocalLevel(
    period=period,
    level_variance_prior=r.InverseGammaPrior(3.0, 2.0),
    seasonal_variance_prior=r.InverseGammaPrior(3.0, 1.0),
    observation_variance_prior=r.InverseGammaPrior(3.0, 8.0),
    initial_level=100.0,
    initial_seasonal_effects=np.zeros(period),
    initial_level_variance=25.0,
    initial_seasonal_variance=25.0,
)
# One fit per origin, each seeing only the months before it. baseline_period=12
# makes the baseline a seasonal naive forecast with bootstrapped seasonal
# differences, so it has intervals to check as well.
result = r.backtest(model, y, horizon=6, initial=48, step=2, baseline_period=period,
                    fit_kwargs={"chains": 2, "draws": 500, "warmup": 500})
print(f"{len(result.folds)} origins, {len(result.errors)} failed fits")

# %% Accuracy and calibration side by side
# compare_backtests scores only the origins both forecasters completed.
# period=12 scales MASE by the seasonal naive error within each fold's training
# data.
comparison = r.compare_backtests(
    {"seasonal model": result, "seasonal naive": result.baseline()}, period=period)
print(comparison)

# %% [markdown] Reading the comparison
# The first six rows score accuracy. MAE, RMSE, WAPE and MASE score the
# predictive mean; CRPS and WIS score the whole predictive distribution. MASE
# divides each error by the mean absolute 12-month difference in that fold's
# training data and averages the results. That is not the baseline's error on
# these outcomes, so a multi-step forecast can beat the baseline and still have
# a MASE above 1.
#
# `coverage_<level>` is the share of outcomes inside the central interval of that
# nominal level, and `width_<level>` is that interval's mean width. A forecaster
# can reach good coverage with wide, uninformative intervals, so read the two
# together. `coverage_error` averages the gap between empirical and nominal
# coverage over 19 levels from 5% to 95%.
#
# Here the model is more accurate than the baseline and its intervals are
# narrower, and both cover less often than nominal. One series with 16
# overlapping origins is a small sample for a coverage rate, as the next table
# shows.

# %% Coverage by horizon step
table = result.calibration(levels=(0.5, 0.8, 0.95))
for step, (coverage, lower, upper, width) in enumerate(zip(
        table["coverage"], table["coverage_lower"], table["coverage_upper"], table["width"]), start=1):
    cells = "  ".join(f"{c:.2f} [{lo:.2f}, {hi:.2f}] w={w:.1f}"
                      for c, lo, hi, w in zip(coverage, lower, upper, width))
    print(f"step {step}: {cells}")

# %% [markdown] How much the intervals in brackets can be trusted
# The bracketed ranges are 95% Wilson intervals for each coverage rate. They treat
# every outcome as an independent trial, and these outcomes are not independent:
# origins two months apart share most of their training data and four of their six
# target months. Overlapping errors like these are usually positively correlated,
# and then the true uncertainty is wider than shown, by an amount the table cannot
# estimate. Use them to spot large gaps, not as a test.

# %% PIT histogram
# A calibrated forecaster's PIT values are roughly uniform: about equal counts per
# bin. Too many in the outer bins means intervals that are too narrow, too many in
# the middle means intervals that are too wide. The same dependence applies, so
# this is a description, not a test, and rustmc does not compute a p-value for it.
for name, backtest in (("seasonal model", result), ("seasonal naive", result.baseline())):
    print(f"{name:>15}:", backtest.pit_histogram(bins=5))
