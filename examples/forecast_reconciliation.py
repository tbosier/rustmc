"""Coherent store, region and total forecasts from separately fitted series.

Each node of a small region -> store hierarchy gets its own local-level fit, so the
base forecasts do not add up. `rmc.reconcile` makes every forecast draw add up and
compares methods on eight held-out weeks.
"""

import numpy as np
import rustmc as rmc

rng = np.random.default_rng(2026)

# Six stores in two regions. A weekly shock shared by every store makes the
# forecast errors of different nodes correlated, which is what MinT uses.
labels = [("north", "n1"), ("north", "n2"), ("north", "n3"),
          ("south", "s1"), ("south", "s2"), ("south", "s3")]
hierarchy = rmc.Hierarchy.from_labels(labels)
print(hierarchy)
print("nodes:", hierarchy.node_names)

weeks, horizon = 104, 8
levels = np.array([40.0, 60.0, 90.0, 30.0, 120.0, 55.0])
walk = np.cumsum(rng.normal(0.0, 1.0, (weeks, 6)) * np.sqrt(levels) / 4, axis=0)
shared = rng.normal(0.0, 3.0, (weeks, 1))
stores = levels + walk + shared + rng.normal(0.0, 1.0, (weeks, 6)) * np.sqrt(levels) / 2
series = stores @ hierarchy.summing_matrix.T  # (week, node), aggregates first
train, actual = series[:-horizon], series[-horizon:]
names = hierarchy.node_names


def fit_nodes(history, seed):
    """One local-level fit per node, with priors scaled to that node's history."""
    models = []
    for column in history.T:
        spread = np.var(np.diff(column))
        models.append(rmc.BayesianLocalLevel(
            process_variance_prior=rmc.InverseGammaPrior(3.0, spread),
            observation_variance_prior=rmc.InverseGammaPrior(3.0, spread),
        ))
    return models[0].fit_batch(list(history.T), ids=names, models=models,
                               chains=2, draws=300, warmup=300, seed=seed, threads=4)


# One-step forecast errors inside the training window, for WLS and MinT: refit
# at each of the last 26 training weeks and forecast the week after it.
errors = []
for origin in range(len(train) - 26, len(train)):
    step = fit_nodes(train[:origin], seed=origin).forecast(1, seed=origin)
    means = np.array([step[name].observation_samples.mean() for name in names])
    errors.append(train[origin] - means)
residuals = np.array(errors)  # (time, node)

# fit_batch fits and forecasts every node independently, so draw (c, d) of one
# node is unrelated to draw (c, d) of another. Reconciled means are unaffected,
# but reconciled intervals then treat the base forecast errors as independent
# across nodes. docs/reconciliation.md explains the consequences.
base = fit_nodes(train, seed=7).forecast(horizon, seed=8)
base_draws = np.stack([base[name].observation_samples for name in names], axis=2)
gap = np.abs(base_draws[:, :, 0] - base_draws[:, :, 3:].sum(axis=2)).max()
print(f"base forecasts: total minus sum of stores, largest gap over draws {gap:.1f}")

results = {"base (incoherent)": base_draws}
for method in ["bottom_up", "ols", "wls_struct", "wls_var", "mint_shrink"]:
    extra = {"residuals": residuals} if method in ("wls_var", "mint_shrink") else {}
    reconciled = rmc.reconcile(base, hierarchy, method=method, **extra)
    results[method] = reconciled.observation_samples
    if method == "mint_shrink":
        print(f"MinT shrinkage intensity: {reconciled.shrinkage_intensity:.3f}")
        draws = reconciled.observation_samples
        gap = np.abs(draws[:, :, 0] - draws[:, :, 3:].sum(axis=2)).max()
        print(f"reconciled: largest gap {gap:.1e}")
        lower, upper = reconciled.interval(0.9)
        print("total, 90% interval by week:")
        for week in range(horizon):
            print(f"  {week + 1}: {lower[0, week]:7.1f} to {upper[0, week]:7.1f}"
                  f"   actual {actual[week, 0]:7.1f}")

# One held-out window of eight weeks: a sanity check, not evidence about which
# method is better on data like this.
print("held-out mean squared error and CRPS, averaged over nodes and weeks:")
for method, draws in results.items():
    squared = ((draws.mean(axis=(0, 1)) - actual.T) ** 2).mean()
    score = rmc.crps(draws, actual.T).mean()
    print(f"  {method:>18}: MSE {squared:8.1f}  CRPS {score:6.2f}")
