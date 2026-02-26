"""
rustmc — 10,000 SKUs in seconds
================================

Fit independent Bayesian demand models for 10,000 SKUs using batch inference.
Each SKU gets a 3-parameter model (intercept + trend + seasonality) fit on
52 weeks of synthetic sales data.

Then pick one random SKU and compare the 4-week-ahead forecast (with 95%
credible interval) against Prophet and ARIMA.

This demonstrates a capability that doesn't exist in Python-first Bayesian
frameworks: thousands of independent posterior inferences in parallel,
using a single Rayon thread pool, with zero Python in the inner loop.
"""

import time
import numpy as np

# ─── Generate 10,000 SKU time series ────────────────────────────────

N_SKUS = 10_000
TRAIN_WEEKS = 48
FORECAST_WEEKS = 4
TOTAL_WEEKS = TRAIN_WEEKS + FORECAST_WEEKS

np.random.seed(42)

true_intercepts = np.random.uniform(50, 500, N_SKUS)
true_trends = np.random.uniform(-2, 5, N_SKUS)
true_seasonality = np.random.uniform(5, 40, N_SKUS)
true_noise = np.random.uniform(3, 15, N_SKUS)

t_all = np.arange(TOTAL_WEEKS, dtype=np.float64)
t_norm = t_all / TOTAL_WEEKS
sin_t = np.sin(2 * np.pi * t_all / 52)
cos_t = np.cos(2 * np.pi * t_all / 52)

all_series = np.zeros((N_SKUS, TOTAL_WEEKS))
for i in range(N_SKUS):
    all_series[i] = (
        true_intercepts[i]
        + true_trends[i] * t_norm
        + true_seasonality[i] * sin_t
        + np.random.randn(TOTAL_WEEKS) * true_noise[i]
    )

print(f"Generated {N_SKUS:,} SKU time series ({TRAIN_WEEKS} train + {FORECAST_WEEKS} forecast weeks)")

# ─── Build rustmc batch models ──────────────────────────────────────

import rustmc as rmc

print("\nBuilding models...")
t_build_start = time.time()

t_train = t_norm[:TRAIN_WEEKS]
sin_train = sin_t[:TRAIN_WEEKS]
cos_train = cos_t[:TRAIN_WEEKS]

models = []
for i in range(N_SKUS):
    builder = rmc.ModelBuilder()
    intercept = builder.normal_prior("intercept", mu=0.0, sigma=200.0)
    trend = builder.normal_prior("trend", mu=0.0, sigma=20.0)
    seas = builder.normal_prior("seasonality", mu=0.0, sigma=50.0)

    mu_expr = intercept + trend * "t" + seas * "sin_t"
    builder.normal_likelihood("obs", mu_expr=mu_expr, sigma=true_noise[i], observed_key="y")
    model = builder.build()

    data = {
        "t": t_train,
        "sin_t": sin_train,
        "y": all_series[i, :TRAIN_WEEKS],
    }
    models.append((model, data))

build_time = time.time() - t_build_start
print(f"Models built in {build_time:.2f}s")

# ─── Batch sample ───────────────────────────────────────────────────

print(f"\nSampling {N_SKUS:,} models (NUTS, 300 warmup + 500 draws each)...")
start = time.time()
results = rmc.batch_sample(models, draws=500, warmup=300, seed=42)
rustmc_time = time.time() - start

total_divs = sum(r.divergences for r in results)
avg_accept = np.mean([r.accept_rate for r in results])
print(f"Done in {rustmc_time:.2f}s  ({N_SKUS / rustmc_time:.0f} models/s)")
print(f"Avg accept rate: {avg_accept:.2f}  |  Total divergences: {total_divs}")

# ─── Pick a random SKU and forecast ─────────────────────────────────

TARGET = 42
actual_train = all_series[TARGET, :TRAIN_WEEKS]
actual_test = all_series[TARGET, TRAIN_WEEKS:]

# rustmc forecast
samples = results[TARGET].get_samples()
n_samples = len(samples["intercept"])
t_future = t_norm[TRAIN_WEEKS:TOTAL_WEEKS]
sin_future = sin_t[TRAIN_WEEKS:TOTAL_WEEKS]

t_future_norm = t_norm[TRAIN_WEEKS:TOTAL_WEEKS]
forecasts = np.zeros((n_samples, FORECAST_WEEKS))
for d in range(n_samples):
    mu_d = (
        samples["intercept"][d]
        + samples["trend"][d] * t_future_norm
        + samples["seasonality"][d] * sin_future
    )
    # Posterior predictive: add observation noise for realistic uncertainty
    forecasts[d] = mu_d + np.random.randn(FORECAST_WEEKS) * true_noise[TARGET]

rustmc_mean = forecasts.mean(axis=0)
rustmc_lo = np.percentile(forecasts, 2.5, axis=0)
rustmc_hi = np.percentile(forecasts, 97.5, axis=0)
rustmc_mae = np.mean(np.abs(rustmc_mean - actual_test))

# ─── ARIMA comparison ───────────────────────────────────────────────

arima_forecast = None
arima_time = None
arima_mae = None
try:
    from statsmodels.tsa.arima.model import ARIMA

    t0 = time.time()
    model_arima = ARIMA(actual_train, order=(1, 1, 1))
    fit_arima = model_arima.fit()
    arima_forecast = fit_arima.forecast(steps=FORECAST_WEEKS)
    arima_time = time.time() - t0
    arima_mae = np.mean(np.abs(arima_forecast - actual_test))
    print(f"\nARIMA(1,1,1) fit in {arima_time:.3f}s  |  MAE: {arima_mae:.2f}")
except Exception as e:
    print(f"\nARIMA failed: {e}")

# ─── Prophet comparison ─────────────────────────────────────────────

prophet_forecast = None
prophet_time = None
prophet_mae = None
try:
    from prophet import Prophet
    import pandas as pd
    import logging
    logging.getLogger("prophet").setLevel(logging.WARNING)
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

    df_train = pd.DataFrame({
        "ds": pd.date_range("2023-01-01", periods=TRAIN_WEEKS, freq="W"),
        "y": actual_train,
    })
    df_future = pd.DataFrame({
        "ds": pd.date_range(df_train["ds"].iloc[-1] + pd.Timedelta(weeks=1),
                            periods=FORECAST_WEEKS, freq="W"),
    })

    t0 = time.time()
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(df_train)
    pred = m.predict(df_future)
    prophet_time = time.time() - t0
    prophet_forecast = pred["yhat"].values
    prophet_lo = pred["yhat_lower"].values
    prophet_hi = pred["yhat_upper"].values
    prophet_mae = np.mean(np.abs(prophet_forecast - actual_test))
    print(f"Prophet fit in {prophet_time:.3f}s  |  MAE: {prophet_mae:.2f}")
except Exception as e:
    print(f"Prophet failed: {e}")

# ─── Summary table ──────────────────────────────────────────────────

print(f"\n{'=' * 60}")
print(f"FORECAST COMPARISON — SKU #{TARGET}")
print(f"{'=' * 60}")
print(f"\n{'Week':<8} {'Actual':>8} {'rustmc':>10} {'ARIMA':>10} {'Prophet':>10}")
print("─" * 50)
for i in range(FORECAST_WEEKS):
    week = TRAIN_WEEKS + i
    a = f"{actual_test[i]:.1f}"
    r = f"{rustmc_mean[i]:.1f}"
    ar = f"{arima_forecast[i]:.1f}" if arima_forecast is not None else "N/A"
    pr = f"{prophet_forecast[i]:.1f}" if prophet_forecast is not None else "N/A"
    print(f"  {week:<6} {a:>8} {r:>10} {ar:>10} {pr:>10}")

print(f"\n{'Method':<25} {'MAE':>8} {'Time':>12} {'Note':>20}")
print("─" * 70)
print(f"{'rustmc (batch NUTS)':<25} {rustmc_mae:>8.2f} {rustmc_time:>11.2f}s {'10K models total':>20}")
if arima_mae is not None:
    est_arima_total = arima_time * N_SKUS
    print(f"{'ARIMA(1,1,1)':<25} {arima_mae:>8.2f} {arima_time:>11.3f}s {f'×{N_SKUS:,}={est_arima_total:.0f}s':>20}")
if prophet_mae is not None:
    est_prophet_total = prophet_time * N_SKUS
    print(f"{'Prophet':<25} {prophet_mae:>8.2f} {prophet_time:>11.3f}s {f'×{N_SKUS:,}={est_prophet_total:.0f}s':>20}")

# ─── Plot ────────────────────────────────────────────────────────────

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))

    weeks_train = np.arange(TRAIN_WEEKS)
    weeks_test = np.arange(TRAIN_WEEKS, TOTAL_WEEKS)

    ax.plot(weeks_train, actual_train, "k-", alpha=0.5, linewidth=1, label="Train")
    ax.plot(weeks_test, actual_test, "ko", markersize=8, zorder=5, label="Actual (test)")

    if prophet_forecast is not None:
        ax.plot(weeks_test, prophet_forecast, "C2:", linewidth=2, label=f"Prophet (MAE={prophet_mae:.1f})")
        ax.fill_between(weeks_test, prophet_lo, prophet_hi, color="C2", alpha=0.2)

    if arima_forecast is not None:
        ax.plot(weeks_test, arima_forecast, "C1--", linewidth=2, label=f"ARIMA (MAE={arima_mae:.1f})")

    # Plot rustmc last so CI band is on top and visible
    ax.fill_between(weeks_test, rustmc_lo, rustmc_hi, color="C0", alpha=0.2, label="rustmc 95% CI", zorder=3)
    ax.plot(weeks_test, rustmc_mean, "C0-", linewidth=2.5, zorder=4, label=f"rustmc (MAE={rustmc_mae:.1f})")

    ax.axvline(TRAIN_WEEKS - 0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Week")
    ax.set_ylabel("Sales")
    ax.set_title(f"Demand Forecast — SKU #{TARGET}  (rustmc: {N_SKUS:,} models in {rustmc_time:.1f}s)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Zoom y-axis to the data range so the CI band is visible
    all_vals = np.concatenate([actual_train, actual_test, rustmc_mean, rustmc_lo, rustmc_hi])
    if arima_forecast is not None:
        all_vals = np.concatenate([all_vals, arima_forecast])
    y_min, y_max = all_vals.min(), all_vals.max()
    y_pad = (y_max - y_min) * 0.15
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    # Only show the last 20 train weeks + forecast for clarity
    ax.set_xlim(TRAIN_WEEKS - 20, TOTAL_WEEKS + 0.5)

    plt.tight_layout()
    plt.savefig("examples/forecast_comparison.png", dpi=150)
    print(f"\nPlot saved to examples/forecast_comparison.png")
except Exception as e:
    print(f"\nPlot failed: {e}")
