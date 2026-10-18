# Forecast reconciliation

This page covers making the forecasts of a hierarchy add up: business unit, IDN and
customer; region and store; product line and item. Read it if you forecast each level
separately and need the customer forecasts to sum to the IDN forecast, the IDNs to the
business unit, and so on.

Forecasts made independently for each node almost never add up. `rmc.reconcile`
adjusts them so they do. It works on forecast draws rather than on point forecasts,
so every draw adds up and the intervals of every level come from the same draws.

## Describe the hierarchy

Build a `Hierarchy` from the labels of each bottom-level series, top level first:

```python
import numpy as np
import rustmc as rmc

hierarchy = rmc.Hierarchy.from_labels([
    ("BU1", "IDN1", "cust1"),
    ("BU1", "IDN1", "cust2"),
    ("BU1", "IDN2", "cust3"),
    ("BU2", "IDN3", "cust4"),
])
print(hierarchy.node_names)
# ['total', 'BU1', 'BU2', 'BU1/IDN1', 'BU1/IDN2', 'BU2/IDN3',
#  'BU1/IDN1/cust1', 'BU1/IDN1/cust2', 'BU1/IDN2/cust3', 'BU2/IDN3/cust4']
print(hierarchy.summing_matrix.shape)  # (10, 4): one row per node, one column per customer
```

The node order is fixed and is the order every array uses:

1. the total, named `total` (change it with `total=`);
2. every level from the top down, each node in the order it first appears in the labels;
3. the bottom series, in the order you gave them.

A node is named by its path joined with `/` (change it with `separator=`), so the same
label under two parents is two nodes: `BU1/IDN3` and `BU2/IDN3`. Labels must be
strings; convert numeric IDs with `str()` first. A label may not contain the
separator, and every path must have the same length. A customer that sits directly
under a business unit, with no IDN, needs an explicit summing matrix.

`hierarchy.levels` gives each node's depth (0 for the total), `hierarchy.parents` each
node's parent, and `aggregate_names` and `bottom_names` the two halves of the node list.

You can also pass the summing matrix `S` yourself. It holds only 0 and 1, its last rows
are the identity (one per bottom series), and every row above them sums at least one
bottom series:

```python
grid = rmc.Hierarchy(
    [[1, 1, 1, 1],   # total
     [1, 1, 0, 0],   # product A
     [0, 0, 1, 1],   # product B
     [1, 0, 1, 0],   # region north
     [0, 1, 0, 1],   # region south
     [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    node_names=["total", "A", "B", "north", "south",
                "A-north", "A-south", "B-north", "B-south"],
)
print(grid.is_nested)  # False: products and regions cross
```

This is a grouped structure: two groupings of the same bottom series that do not nest.
Every method below accepts it except top-down by forecast proportions, which needs a
tree. `from_labels` only builds nested hierarchies, so grouped structures need an
explicit matrix.

## Reconcile forecast draws

Pass draws shaped `(chain, draw, node, step)` with nodes in `hierarchy.node_names`
order, and name a method:

```python
rng = np.random.default_rng(0)
n = hierarchy.node_count
base = 100 + 10 * rng.standard_normal((4, 250, n, 6))   # incoherent base draws
residuals = rng.standard_normal((52, n))                 # one-step errors, (time, node)

result = rmc.reconcile(base, hierarchy, method="mint_shrink", residuals=residuals)
draws = result.observation_samples                       # (chain, draw, node, step)
total = draws[:, :, 0, :]
customers = draws[:, :, hierarchy.aggregate_count:, :]
assert np.allclose(total, customers.sum(axis=2))

mean = result.observation_mean                           # (node, step)
lower, upper = result.interval(0.9)                      # each (node, step)
bu1 = result.node_samples("BU1")                         # (chain, draw, step)
print(result, result.shrinkage_intensity)
```

Draw `(c, d)` of the result is the reconciliation of draw `(c, d)` of the input, at every
step, so chain, draw and horizon keep their meaning. The bottom series are reconciled
first and every aggregate is then their sum, so each draw adds up to within rounding.
Intervals are pointwise and equal-tailed, and use the same quantile rule as every other
forecast. `rmc.ForecastDraws.from_result(result)` gives a `ForecastDraws` with the node
names as series labels, for saving and scoring.

`reconcile` also accepts, in place of the array:

- a dict from every node name to that node's `(chain, draw, step)` draws;
- a `ForecastDraws` with a series axis, matched by its series labels;
- a `ForecastBatchForecast` whose cell IDs are the node names (see below);
- a `BayesianHierarchicalForecast`, with `hierarchy` omitted: it builds the total, group
  and program hierarchy from the forecast's names. Its group and total draws are
  already sums of its program draws, so every projection method returns them
  unchanged; this adapter puts them in the same layout as other reconciled forecasts.
  If a group is named `total` or a name contains `/`, the default names clash; pass a
  hierarchy built with `Hierarchy.from_labels` from `(group name, program name)` pairs
  with another `total=` or `separator=`. It is matched to the forecast by structure.

`residuals` and `history` may likewise be dicts from node name (or bottom name) to one
series. Any input that is missing a node, names a node twice or names one that does not
exist is refused.

## Choose a method

| `method` | Bottom forecasts come from | Needs |
|---|---|---|
| `bottom_up` | the bottom series' own draws | nothing |
| `top_down_average_proportions` | each draw's total, split by `mean_t(y[t, j] / total[t])` | `history`, `(time, bottom series)` |
| `top_down_forecast_proportions` | each draw's total, split by the base forecast means | a nested hierarchy |
| `ols` | the projection with `W = I` | nothing |
| `wls_struct` | the projection with `W = diag(S 1)`, each node's count of bottom series | nothing |
| `wls_var` | the projection with `W` = the residual variances | `residuals`, `(time, node)` |
| `mint_shrink` | the projection with `W` = the shrunk residual covariance | `residuals`, `(time, node)` |

The projection methods set the bottom forecasts to `P y` with
`P = (S' W⁻¹ S)⁻¹ S' W⁻¹`, and every node to `S P y`. They leave forecasts that already
add up unchanged. Bottom-up is also a projection, with `P = [0 | I]`.

Top-down methods split each draw's total; no other node's draw enters the result
directly. They are not projections: a forecast that already adds up generally changes. Every reconciled
bottom draw is a fixed share of the same total draw, so bottom series are perfectly
correlated within a draw. The average-proportions method (Gross and Sohl's method A)
needs non-negative history with a positive total in every period. The
forecast-proportions method (Athanasopoulos, Ahmed and Hyndman, 2009) computes, at each
step, each node's share of the base means of its siblings and multiplies the shares
down the path from the total. Those shares come from the means of the base draws, not
from each draw, so the map is linear within a step; it needs non-negative base means
with a positive sum under every aggregate.

`wls_struct` assumes every bottom series has the same forecast error variance. When
they differ a lot it can be worse than doing nothing: in the known-process test below
it raised the mean squared error by about 12%.

### Residuals for `wls_var` and `mint_shrink`

`residuals` are one-step-ahead forecast errors for every node, one row per time point,
in node order. The method papers use in-sample one-step residuals from each node's
model; errors from one-step forecasts at rolling origins, as in the
[example](https://github.com/tbosier/rustmc/blob/main/examples/forecast_reconciliation.py),
are a common substitute. Every node's residuals must be finite and not all zero.

The covariance estimate is `W₁ = E'E / T`, uncentred, as in the MinT paper and the
`hts` and `fable` packages. `wls_var` uses its diagonal. `mint_shrink` uses
`λ diag(W₁) + (1 − λ) W₁`, with `λ` the Schäfer–Strimmer intensity computed from the
residuals as those packages compute it, reported as `result.shrinkage_intensity`. It
needs at least two residual rows. The same `W` is used for every step of the horizon,
which assumes the error covariance at step `h` is proportional to the one-step
covariance, the assumption Wickramasuriya, Athanasopoulos and Hyndman (2019) make.

The projection is computed without inverting `W` or forming any matrix with a row per
node on both sides, through the equivalent form in Wickramasuriya et al. (2019): the
only system solved has one row per aggregate, and the work per draw and step grows
with the number of bottom series times the number of aggregates.

## Draws must be joint draws

Reconciling each draw separately gives draws from the reconciled forecast distribution
only if every draw is a joint draw of all nodes: draw `(c, d)` of the total and draw
`(c, d)` of a customer must come from one simulation of the whole hierarchy
(Panagiotelis et al., 2023). The library cannot check this; it is your responsibility.

Draws generated independently per node do not meet it. That includes stacking forecasts
from separate fits, a `ForecastBatchForecast`, and a dict of per-node forecasts from
different models. Reconciling them is still well defined and every reconciled draw
still adds up. What changes is what the spread means:

- Reconciled means are the same either way, because the map is linear.
- A reconciled draw is `S P y`, so its covariance is `S P Σ P' S'`, where `Σ` is the
  covariance of the base draws across nodes. Independent draws make `Σ` diagonal and
  drop whatever correlation the base forecast errors really have. Any node that the
  method builds from several base forecasts, which for the projection methods is
  every aggregate and usually every bottom series, then gets an interval of the wrong
  width. The direction cannot be stated in general, because projection weights can
  be negative: with OLS on a total and two series whose errors all correlate at 0.9,
  dropping the correlation makes the reconciled total's interval narrower and the two
  bottom intervals wider (variances 0.67 against 1.67 and 0.47).

If your intervals matter, generate the base draws jointly, or check the reconciled
intervals' coverage on held-out data with `rmc.backtest` or `rmc.interval_score`
before relying on them.

## Forecasts from a batch of independent fits

`fit_batch` fits and forecasts each series on its own, which is a common way to get a
base forecast for every node. Give each cell the node's name as its ID:

```python
prior = rmc.InverseGammaPrior(3.0, 1.0)
model = rmc.BayesianLocalLevel(process_variance_prior=prior,
                               observation_variance_prior=prior)
small = rmc.Hierarchy.from_labels([("north",), ("south",)])
history = 50 + np.cumsum(rng.normal(0, 1, (40, 2)), axis=0)
series = history @ small.summing_matrix.T                # (time, node): total, north, south
batch = model.fit_batch(list(series.T), ids=small.node_names,
                        chains=2, draws=200, warmup=200, seed=1)
forecast = batch.forecast(4, seed=2)
coherent = rmc.reconcile(forecast, small, method="ols")
print(coherent.observation_mean[:, 0])
```

These draws are independent across nodes, so the previous section applies.

## What the tests check

- `rust_core/tests/reconciliation.rs` and `tests/test_reconciliation.py` compare OLS,
  both WLS variants and MinT with the dense formula `S (S' W⁻¹ S)⁻¹ S' W⁻¹` on a nested
  and a grouped hierarchy, to a relative tolerance of 1e-10, including the shrinkage
  intensity against a transcription of `hts`'s estimator. Every method's draws add up
  to 1e-10 relative, projections return coherent input unchanged, and reversing the
  draw order reverses the output.
- A known process: a total, three regions and twelve stores, with forecast errors
  that share a common and a regional factor and whose scale varies thirtyfold between
  stores. Over 1,000 seeded replications, with 200 joint draws each and 60 residual rows
  estimated afresh each time, the mean squared error relative to the incoherent base
  forecasts was 0.38 for `mint_shrink`, 0.46 for `wls_var`, 0.77 for `ols` and 1.12 for
  `wls_struct`; CRPS was 0.66, 0.70, 1.02 and 1.12 of the base. The best achievable
  with the true covariance is about 0.35 of the base in squared error; `mint_shrink` came
  within 11% of it. The test fails if MinT is not within 20% of that optimum, or if
  `wls_var`, which ignores correlations, is. It does not check the shrinkage
  intensity: a fixed intensity also passes it. The formula tests check that.

These are results for one simulated process. On your data, compare methods on held-out
periods.

## Limitations

- Only 0/1 summing matrices: weighted aggregates, such as currency conversion, are not
  supported.
- `from_labels` builds nested hierarchies with every bottom series at the same depth.
  Grouped or ragged structures need an explicit summing matrix.
- Only observation draws are reconciled. `mean_samples` of a `ForecastDraws` are not.
- MinT uses shrinkage only; the unshrunk sample covariance (`mint_cov` in `fable`) is not
  offered, and neither are the non-negative or immutable-node variants.
- The same `W` is used at every step.
- Draws, residuals, the summing matrix and the method's working matrices are each
  limited to 25 million values, as forecasts are. At 5,500 nodes and 12 steps that is
  about 375 draws; reduce the draws or split the horizon if a request is refused.

## References

- Athanasopoulos, G., Ahmed, R. A. and Hyndman, R. J. (2009). Hierarchical forecasts for
  Australian domestic tourism. *International Journal of Forecasting*, 25(1), 146–166.
- Athanasopoulos, G., Hyndman, R. J., Kourentzes, N. and Petropoulos, F. (2017).
  Forecasting with temporal hierarchies. *European Journal of Operational Research*,
  262(1), 60–74.
- Gross, C. W. and Sohl, J. E. (1990). Disaggregation methods to expedite product line
  forecasting. *Journal of Forecasting*, 9(3), 233–254.
- Panagiotelis, A., Gamakumara, P., Athanasopoulos, G. and Hyndman, R. J. (2023).
  Probabilistic forecast reconciliation: properties, evaluation and score optimisation.
  *European Journal of Operational Research*, 306(2), 693–706.
- Schäfer, J. and Strimmer, K. (2005). A shrinkage approach to large-scale covariance
  matrix estimation and implications for functional genomics. *Statistical Applications
  in Genetics and Molecular Biology*, 4(1), Article 32.
- Wickramasuriya, S. L., Athanasopoulos, G. and Hyndman, R. J. (2019). Optimal forecast
  reconciliation for hierarchical and grouped time series through trace minimization.
  *Journal of the American Statistical Association*, 114(526), 804–819.
