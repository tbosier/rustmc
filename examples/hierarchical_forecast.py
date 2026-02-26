"""
rustmc — Hierarchical Time Series Forecasting
================================================

Stress test: forecast weekly sales for a retail hierarchy.

Hierarchy:
    3 states × 20 stores × 5 categories × 20 items = 6,000 time series

Model (Bayesian fixed-effects regression):
    y_{s,st,c,i,t} = α_state[s] + α_store[st] + α_cat[c] + α_item[i]
                    + β_trend · t
                    + β_sin · sin(2πt/52)
                    + β_cos · cos(2πt/52)
                    + ε

    171 parameters, ~132K training observations.

    Train on weeks 0–21, forecast weeks 22–25 with posterior uncertainty.
    Compare one series against ARIMA.
"""

import time
import numpy as np

# ─── Hierarchy ───────────────────────────────────────────────────────

N_STATES = 3
N_STORES_PER_STATE = 20
N_CATEGORIES = 5
N_ITEMS_PER_CAT = 20

N_STORES = N_STATES * N_STORES_PER_STATE          # 60
N_ITEMS = N_CATEGORIES * N_ITEMS_PER_CAT           # 100
N_SERIES = N_STORES * N_ITEMS                      # 6,000

TRAIN_WEEKS = 22
TEST_WEEKS = 4
TOTAL_WEEKS = TRAIN_WEEKS + TEST_WEEKS             # 26

print(f"Hierarchy: {N_STATES} states × {N_STORES_PER_STATE} stores × "
      f"{N_CATEGORIES} categories × {N_ITEMS_PER_CAT} items")
print(f"Series: {N_SERIES:,}  |  Train weeks: {TRAIN_WEEKS}  |  "
      f"Test weeks: {TEST_WEEKS}")

# ─── Generate synthetic data ─────────────────────────────────────────

np.random.seed(42)

state_effects = np.random.normal(100, 20, N_STATES)
store_effects = np.random.normal(0, 10, N_STORES)
cat_effects = np.random.normal(0, 15, N_CATEGORIES)
item_effects = np.random.normal(0, 5, N_ITEMS)
true_trend = 0.3
true_sin_amp = 8.0
true_cos_amp = 5.0
noise_std = 4.0

store_state = np.repeat(np.arange(N_STATES), N_STORES_PER_STATE)
item_cat = np.repeat(np.arange(N_CATEGORIES), N_ITEMS_PER_CAT)

rows_state = []
rows_store = []
rows_cat = []
rows_item = []
rows_t = []
rows_y = []

for st_idx in range(N_STORES):
    s_idx = store_state[st_idx]
    for it_idx in range(N_ITEMS):
        c_idx = item_cat[it_idx]
        for t in range(TOTAL_WEEKS):
            mu = (state_effects[s_idx]
                  + store_effects[st_idx]
                  + cat_effects[c_idx]
                  + item_effects[it_idx]
                  + true_trend * t
                  + true_sin_amp * np.sin(2 * np.pi * t / 52)
                  + true_cos_amp * np.cos(2 * np.pi * t / 52))
            y = mu + np.random.normal(0, noise_std)
            rows_state.append(s_idx)
            rows_store.append(st_idx)
            rows_cat.append(c_idx)
            rows_item.append(it_idx)
            rows_t.append(t)
            rows_y.append(y)

rows_state = np.array(rows_state)
rows_store = np.array(rows_store)
rows_cat = np.array(rows_cat)
rows_item = np.array(rows_item)
rows_t = np.array(rows_t, dtype=np.float64)
rows_y = np.array(rows_y)

N_total = len(rows_y)
train_mask = rows_t < TRAIN_WEEKS
N_train = train_mask.sum()
N_test = N_total - N_train

print(f"Total obs: {N_total:,}  |  Train: {N_train:,}  |  Test: {N_test:,}")

# ─── Build design matrix (indicator columns) ────────────────────────

print("\nBuilding design matrix...")
t_build = time.time()

data = {}
param_specs = []  # (name, data_key, mu, sigma)

# State indicators
for s in range(N_STATES):
    key = f"x_state_{s}"
    data[key] = (rows_state[train_mask] == s).astype(np.float64)
    param_specs.append((f"alpha_state_{s}", key, 0.0, 50.0))

# Store indicators
for st in range(N_STORES):
    key = f"x_store_{st}"
    data[key] = (rows_store[train_mask] == st).astype(np.float64)
    param_specs.append((f"alpha_store_{st}", key, 0.0, 20.0))

# Category indicators
for c in range(N_CATEGORIES):
    key = f"x_cat_{c}"
    data[key] = (rows_cat[train_mask] == c).astype(np.float64)
    param_specs.append((f"alpha_cat_{c}", key, 0.0, 30.0))

# Item indicators
for i in range(N_ITEMS):
    key = f"x_item_{i}"
    data[key] = (rows_item[train_mask] == i).astype(np.float64)
    param_specs.append((f"alpha_item_{i}", key, 0.0, 10.0))

# Trend
t_normalized = rows_t[train_mask] / TOTAL_WEEKS
data["x_trend"] = t_normalized
param_specs.append(("beta_trend", "x_trend", 0.0, 20.0))

# Seasonality
data["x_sin"] = np.sin(2 * np.pi * rows_t[train_mask] / 52)
data["x_cos"] = np.cos(2 * np.pi * rows_t[train_mask] / 52)
param_specs.append(("beta_sin", "x_sin", 0.0, 20.0))
param_specs.append(("beta_cos", "x_cos", 0.0, 20.0))

# Observed
data["y"] = rows_y[train_mask]

N_PARAMS = len(param_specs)
print(f"Parameters: {N_PARAMS}  |  Design matrix: {N_train:,} × {N_PARAMS}")
print(f"Design matrix built in {time.time() - t_build:.1f}s")

# ─── Build rustmc model ─────────────────────────────────────────────

import rustmc as rmc

print("\nBuilding model...")
builder = rmc.ModelBuilder()

params = []
for name, data_key, mu, sigma in param_specs:
    p = builder.normal_prior(name, mu=mu, sigma=sigma)
    params.append((p, data_key))

mu_expr = params[0][0] * params[0][1]
for p, dk in params[1:]:
    mu_expr = mu_expr + p * dk

builder.normal_likelihood("obs", mu_expr=mu_expr, sigma=noise_std, observed_key="y")
model = builder.build()

# ─── Sample with NUTS ────────────────────────────────────────────────

NUM_CHAINS = 4
NUM_DRAWS = 500
NUM_WARMUP = 500

print(f"\nSampling: {NUM_CHAINS} chains × ({NUM_WARMUP} warmup + {NUM_DRAWS} draws)")
print(f"Sampler: NUTS  |  Max tree depth: 10")

start = time.time()
fit = rmc.sample(
    model_spec=model,
    data=data,
    chains=NUM_CHAINS,
    draws=NUM_DRAWS,
    warmup=NUM_WARMUP,
    seed=42,
    sampler="nuts",
)
sampling_time = time.time() - start

print(f"\nSampling completed in {sampling_time:.1f}s")
print(f"\nDiagnostics (first 5 + last 3 parameters):")

diags = fit.diagnostics()
header = f"{'Parameter':<18} {'mean':>8} {'std':>8} {'r_hat':>8} {'ess_bulk':>10}"
print(header)
print("─" * len(header))
for d in diags[:5]:
    print(f"{d['name']:<18} {d['mean']:>8.3f} {d['std']:>8.4f} {d['r_hat']:>8.4f} {d['ess_bulk']:>10.0f}")
print(f"  ... ({N_PARAMS - 8} more parameters) ...")
for d in diags[-3:]:
    print(f"{d['name']:<18} {d['mean']:>8.3f} {d['std']:>8.4f} {d['r_hat']:>8.4f} {d['ess_bulk']:>10.0f}")

total_divs = sum(fit.divergences())
print(f"\nDivergences: {total_divs}  |  Accept rates: "
      f"{[round(r, 2) for r in fit.accept_rates()]}")

# ─── Forecast ────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("FORECAST COMPARISON — Store 0, Item 0")
print("=" * 60)

target_store = 0
target_item = 0
target_state = store_state[target_store]
target_cat = item_cat[target_item]

# Build test features for this series
test_t = np.arange(TRAIN_WEEKS, TOTAL_WEEKS, dtype=np.float64)
n_forecast = len(test_t)

# Get posterior samples
samples = fit.get_samples()
n_samples = len(samples[param_specs[0][0]])

# Compute posterior predictive for each sample
forecasts = np.zeros((n_samples, n_forecast))
for draw in range(n_samples):
    for fi, t in enumerate(test_t):
        mu = 0.0
        mu += samples[f"alpha_state_{target_state}"][draw]
        mu += samples[f"alpha_store_{target_store}"][draw]
        mu += samples[f"alpha_cat_{target_cat}"][draw]
        mu += samples[f"alpha_item_{target_item}"][draw]
        mu += samples["beta_trend"][draw] * (t / TOTAL_WEEKS)
        mu += samples["beta_sin"][draw] * np.sin(2 * np.pi * t / 52)
        mu += samples["beta_cos"][draw] * np.cos(2 * np.pi * t / 52)
        forecasts[draw, fi] = mu

forecast_mean = forecasts.mean(axis=0)
forecast_lo = np.percentile(forecasts, 5, axis=0)
forecast_hi = np.percentile(forecasts, 95, axis=0)

# Actual values for this series
series_mask = ((rows_store == target_store) & (rows_item == target_item)
               & (rows_t >= TRAIN_WEEKS))
actual = rows_y[series_mask]

bayesian_mae = np.mean(np.abs(forecast_mean - actual))

print(f"\n{'Week':<8} {'Actual':>8} {'Bayesian':>10} {'90% CI':>20}")
print("─" * 50)
for i in range(n_forecast):
    print(f"  {int(test_t[i]):<6} {actual[i]:>8.1f} {forecast_mean[i]:>10.1f} "
          f"  [{forecast_lo[i]:>7.1f}, {forecast_hi[i]:>7.1f}]")
print(f"\nBayesian MAE: {bayesian_mae:.2f}")

# ─── ARIMA comparison ────────────────────────────────────────────────

try:
    from statsmodels.tsa.arima.model import ARIMA as ARIMA_Model

    train_series_mask = ((rows_store == target_store) & (rows_item == target_item)
                         & (rows_t < TRAIN_WEEKS))
    train_y = rows_y[train_series_mask]

    arima_start = time.time()
    arima = ARIMA_Model(train_y, order=(2, 1, 1))
    arima_fit = arima.fit()
    arima_forecast = arima_fit.forecast(steps=n_forecast)
    arima_time = time.time() - arima_start

    arima_mae = np.mean(np.abs(arima_forecast - actual))

    print(f"\nARIMA(2,1,1) forecast (fit time: {arima_time:.3f}s):")
    for i in range(n_forecast):
        print(f"  Week {int(test_t[i]):<4}  Actual: {actual[i]:>7.1f}  "
              f"ARIMA: {arima_forecast[i]:>7.1f}")
    print(f"\nARIMA MAE: {arima_mae:.2f}")

    print(f"\n{'Method':<20} {'MAE':>8} {'Time':>10}")
    print("─" * 40)
    print(f"{'rustmc Bayesian':<20} {bayesian_mae:>8.2f} {sampling_time:>9.1f}s")
    print(f"{'ARIMA(2,1,1)':<20} {arima_mae:>8.2f} {arima_time:>9.3f}s")
    print(f"\nNote: Bayesian model fits ALL {N_SERIES:,} series jointly;")
    print(f"      ARIMA fits 1 series. To fit all {N_SERIES:,} with ARIMA: "
          f"~{arima_time * N_SERIES:.0f}s")

except ImportError:
    print("\nstatsmodels not installed — skipping ARIMA comparison.")
    print("  Install with: pip install statsmodels")
except Exception as e:
    print(f"\nARIMA failed: {e}")

# ─── Summary ─────────────────────────────────────────────────────────

print(f"\n{'=' * 60}")
print(f"SUMMARY")
print(f"{'=' * 60}")
print(f"Hierarchy:      {N_STATES} states × {N_STORES_PER_STATE} stores × "
      f"{N_CATEGORIES} cat × {N_ITEMS_PER_CAT} items")
print(f"Time series:    {N_SERIES:,}")
print(f"Parameters:     {N_PARAMS}")
print(f"Training obs:   {N_train:,}")
print(f"Sampler:        NUTS ({NUM_CHAINS} chains × {NUM_DRAWS} draws)")
print(f"Sampling time:  {sampling_time:.1f}s")
print(f"Divergences:    {total_divs}")
print(f"Forecast MAE:   {bayesian_mae:.2f}")
