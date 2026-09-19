"""
rustmc — retail panel forecast with group indexing
==================================================

A weekly retail panel: 3 states x 5 stores x 3 categories x 4 items, so 15
stores, 12 items, and 180 store-item series. The model is fitted on 26 weeks
and forecasts the next 4.

Group indexing, not indicator columns
-------------------------------------
Each store effect is one element of a vector parameter, selected per row with
`store_level["store"]`, where `"store"` is an integer data column. That compiles
to a single gather node: memory and work grow with the number of rows, not with
rows x levels. Building one dense 0/1 indicator column and one scalar parameter
per level instead costs a full pass over `n_rows x n_levels` values on every
gradient evaluation, which is what an earlier version of this script did.

Identification
--------------
Stacking `state + store + category + item` intercepts in one linear predictor,
each with its own wide independent prior, is not identified. Two of those blocks
are redundant twice over: a store determines its state and an item determines its
category, so the state and category columns lie exactly in the span of the store
and item columns; and any constant can be moved between blocks without changing
the predictor. Only the sum is identified, the rest is fixed by the priors alone,
and NUTS pays for the resulting flat directions with saturated tree depth.

This version uses a reference-level encoding, which is identified by
construction:

* `store_level[s]` is the expected level of store `s` for the reference item, at
  the mean of the centred covariates. It absorbs the state effect, because every
  store belongs to exactly one state.
* `item_dev[k]` is item `k + 1` relative to item 0, shared across stores. It
  absorbs the category effect for the same reason. Item 0 rows are excluded from
  this term with an `item_on` mask, so no coefficient is needed for the reference.

Partial pooling was the other candidate and is not used here. `vector_normal_prior`
takes constant hyperparameters, so a pooled block has to be written non-centred as
`sigma * z[key]`. Every store here has 312 training rows and every item 390, which pins
the product `sigma * z` tightly and leaves a strongly curved `sigma`-`z` ridge — the
parameterisation that suits sparse groups, not data-rich ones. State and category
summaries are still reported below, by aggregating the posterior store and item
effects rather than by giving them their own redundant parameters.

Runtime: 4 chains x (500 warmup + 500 draws) finished in around 30 seconds when this
was written, on an otherwise idle 24-core machine. Chains run in parallel, so fewer
cores take longer. That is a rough expectation rather than retained benchmark
evidence; the script prints its own elapsed time.
"""

import time

import numpy as np

import rustmc as rmc

# ── Panel shape ──────────────────────────────────────────────────────────
N_STATES, STORES_PER_STATE = 3, 5
N_CATEGORIES, ITEMS_PER_CATEGORY = 3, 4
N_STORES = N_STATES * STORES_PER_STATE          # 15
N_ITEMS = N_CATEGORIES * ITEMS_PER_CATEGORY     # 12

TRAIN_WEEKS, TEST_WEEKS = 26, 4
TOTAL_WEEKS = TRAIN_WEEKS + TEST_WEEKS
SEASON = 13                                     # quarterly cycle, 2 per training window

store_state = np.repeat(np.arange(N_STATES), STORES_PER_STATE)
item_category = np.repeat(np.arange(N_CATEGORIES), ITEMS_PER_CATEGORY)

# ── Generating parameters ────────────────────────────────────────────────
rng = np.random.default_rng(11)


def centred(n, scale):
    values = rng.normal(0.0, scale, n)
    return values - values.mean()


TRUE_BASE = 100.0
true_state = centred(N_STATES, 12.0)
true_store = centred(N_STORES, 6.0)
true_category = centred(N_CATEGORIES, 9.0)
true_item = centred(N_ITEMS, 4.0)
TRUE_TREND = 0.3          # per week
TRUE_SIN, TRUE_COS = 8.0, 5.0
TRUE_NOISE = 4.0

# Long format: one row per (store, item, week).
store = np.repeat(np.arange(N_STORES), N_ITEMS * TOTAL_WEEKS)
item = np.tile(np.repeat(np.arange(N_ITEMS), TOTAL_WEEKS), N_STORES)
week = np.tile(np.arange(TOTAL_WEEKS, dtype=float), N_STORES * N_ITEMS)
state, category = store_state[store], item_category[item]

angle = 2 * np.pi * week / SEASON
mean = (TRUE_BASE + true_state[state] + true_store[store]
        + true_category[category] + true_item[item]
        + TRUE_TREND * week + TRUE_SIN * np.sin(angle) + TRUE_COS * np.cos(angle))
sales = mean + rng.normal(0.0, TRUE_NOISE, mean.size)

train = week < TRAIN_WEEKS
print(f"Panel: {N_STORES} stores x {N_ITEMS} items = {N_STORES * N_ITEMS} series")
print(f"Rows: {sales.size:,} total, {int(train.sum()):,} training, "
      f"{int((~train).sum()):,} held out")


def design(rows):
    """Model columns for a boolean row selection."""
    a = 2 * np.pi * week[rows] / SEASON
    return {
        "store": store[rows].astype(float),
        # Reference-level encoding: item 0 contributes nothing and needs no
        # coefficient, so `item_dev` has N_ITEMS - 1 elements.
        "item_ref": np.maximum(item[rows] - 1, 0).astype(float),
        "item_on": (item[rows] > 0).astype(float),
        "t": week[rows] / TRAIN_WEEKS,
        "sin": np.sin(a),
        "cos": np.cos(a),
    }


training = design(train)
# Centring the covariates keeps them close to orthogonal to the level columns.
shift = {key: float(training[key].mean()) for key in ("t", "sin", "cos")}
for key, value in shift.items():
    training[key] = training[key] - value
training["y"] = sales[train]

# ── Model ────────────────────────────────────────────────────────────────
builder = rmc.ModelBuilder()
store_level = builder.vector_normal_prior("store_level", N_STORES, 100.0, 30.0)
item_dev = builder.vector_normal_prior("item_dev", N_ITEMS - 1, 0.0, 15.0)
beta_trend = builder.normal_prior("beta_trend", 0.0, 20.0)
beta_sin = builder.normal_prior("beta_sin", 0.0, 20.0)
beta_cos = builder.normal_prior("beta_cos", 0.0, 20.0)
sigma = builder.half_normal_prior("sigma_obs", 20.0)

predictor = (store_level["store"]
             + item_dev["item_ref"] * "item_on"
             + beta_trend * "t" + beta_sin * "sin" + beta_cos * "cos")
builder.normal_likelihood("obs", predictor, sigma, "y")
compiled = builder.compile()
print(f"Parameters: {len(compiled.param_names)}")

CHAINS, DRAWS, WARMUP = 4, 500, 500
print(f"\nSampling: NUTS, {CHAINS} chains x ({WARMUP} warmup + {DRAWS} draws)")
start = time.time()
fit = compiled.sample(training, chains=CHAINS, draws=DRAWS, warmup=WARMUP,
                      seed=7, show_progress=False)
elapsed = time.time() - start
print(f"Sampling completed in {elapsed:.1f}s")

# ── Diagnostics ──────────────────────────────────────────────────────────
diagnostics = fit.diagnostics()
print(f"max R-hat {max(d['r_hat'] for d in diagnostics):.4f}  "
      f"min bulk ESS {min(d['ess_bulk'] for d in diagnostics):.0f}  "
      f"min tail ESS {min(d['ess_tail'] for d in diagnostics):.0f}  "
      f"divergences {sum(fit.divergences())}")
print("Inspect these before reading anything below.")

# ── Parameter recovery ───────────────────────────────────────────────────
draws = fit.get_samples_2d()


def summarise(name):
    values = draws[name].reshape(-1)
    return values.mean(), values.std()


# The covariates were centred, so beta_trend is per TRAIN_WEEKS weeks.
print("\nRecovered vs generating values")
print(f"{'parameter':<14} {'posterior mean':>15} {'posterior sd':>13} {'true':>9}")
for name, truth in (("beta_trend", TRUE_TREND * TRAIN_WEEKS),
                    ("beta_sin", TRUE_SIN),
                    ("beta_cos", TRUE_COS),
                    ("sigma_obs", TRUE_NOISE)):
    mean_, sd_ = summarise(name)
    print(f"{name:<14} {mean_:>15.3f} {sd_:>13.3f} {truth:>9.3f}")

# Store levels and item contrasts are identified relative to the reference item
# and the centred covariates, so compare them on that same scale.
true_store_level = (TRUE_BASE + true_state[store_state] + true_store
                    + true_category[0] + true_item[0]
                    + TRUE_TREND * TRAIN_WEEKS * shift["t"]
                    + TRUE_SIN * shift["sin"] + TRUE_COS * shift["cos"])
fitted_store_level = np.array(
    [draws[f"store_level[{s}]"].mean() for s in range(N_STORES)])
true_item_dev = (true_category[item_category[1:]] + true_item[1:]
                 - true_category[0] - true_item[0])
fitted_item_dev = np.array(
    [draws[f"item_dev[{k}]"].mean() for k in range(N_ITEMS - 1)])

print(f"\nStore levels   max |error| {np.abs(fitted_store_level - true_store_level).max():.3f}"
      f"   (generating spread {true_store_level.std():.2f})")
print(f"Item contrasts max |error| {np.abs(fitted_item_dev - true_item_dev).max():.3f}"
      f"   (generating spread {true_item_dev.std():.2f})")

# State and category effects are recovered by aggregation, not by separate
# redundant parameters.
print("\nState means from aggregated store levels (differences from state 0)")
true_state_mean = np.array([true_state[s] + true_store[store_state == s].mean()
                            for s in range(N_STATES)])
for s in range(N_STATES):
    fitted = (fitted_store_level[store_state == s].mean()
              - fitted_store_level[store_state == 0].mean())
    truth = true_state_mean[s] - true_state_mean[0]
    print(f"  state {s}: {fitted:>7.2f}   true {truth:>7.2f}")

print("Category means from aggregated item effects (differences from category 0)")
fitted_item_effect = np.concatenate([[0.0], fitted_item_dev])   # item 0 is the reference
true_item_effect = (true_category[item_category] + true_item
                    - true_category[0] - true_item[0])
for c in range(N_CATEGORIES):
    inside = item_category == c
    fitted = (fitted_item_effect[inside].mean()
              - fitted_item_effect[item_category == 0].mean())
    truth = (true_item_effect[inside].mean()
             - true_item_effect[item_category == 0].mean())
    print(f"  category {c}: {fitted:>7.2f}   true {truth:>7.2f}")

# ── Forecast the held-out weeks ──────────────────────────────────────────
future = design(~train)
for key, value in shift.items():
    future[key] = future[key] - value
predictive = fit.predict(future, seed=8)["obs"]        # (chain, draw, row)
expected = fit.predict(future, expected=True, seed=8)["obs"]
actual = sales[~train]

flat = predictive.reshape(-1, predictive.shape[-1])
lo, hi = np.quantile(flat, [0.05, 0.95], axis=0)
point = expected.reshape(-1, expected.shape[-1]).mean(axis=0)

print(f"\nHeld-out weeks {TRAIN_WEEKS}-{TOTAL_WEEKS - 1}, {actual.size:,} rows")
print(f"  MAE of the posterior mean          {np.abs(point - actual).mean():.3f}"
      f"   (irreducible noise {TRUE_NOISE * np.sqrt(2 / np.pi):.3f})")
print(f"  RMSE of the posterior mean         {np.sqrt(((point - actual) ** 2).mean()):.3f}"
      f"   (irreducible noise {TRUE_NOISE:.3f})")
print(f"  90% equal-tailed predictive cover  "
      f"{np.mean((actual >= lo) & (actual <= hi)):.3f}")
print("The interval above is a posterior predictive interval, not an HDI and not a"
      " confidence interval.")

print("\nOne series, store 0 item 0")
series = (store[~train] == 0) & (item[~train] == 0)
print(f"{'week':>6} {'actual':>9} {'forecast':>10} {'90% predictive':>22}")
for w, a, p, l, h in zip(week[~train][series], actual[series], point[series],
                         lo[series], hi[series]):
    print(f"{int(w):>6} {a:>9.1f} {p:>10.1f}      [{l:>7.1f}, {h:>7.1f}]")

print("\nData here is generated, and the priors describe this example only."
      " Performance comparisons belong in benchmarks/.")
