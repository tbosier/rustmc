# Decisions from forecast draws

A forecast gives a range of outcomes. Often you then have to commit to one number: a
reserve, an order quantity, a staffing level. This page shows how to pick the number
with the lowest expected cost, given what being wrong in each direction costs you.

Read it if you already have predictive draws, from a rustmc forecast or anywhere else,
and the cost of a shortfall differs from the cost of a surplus.

## Reserve for next month

Next month's rebate liability is forecast at about $7.5M, with a 95% interval of about
$5M to $10M. Each dollar under-reserved costs five times what a dollar over-reserved
costs. How much should you reserve?

The forecast below comes from a local-level model on the log scale, so amounts stay
positive. The history is illustrative.

```python
import numpy as np
import rustmc as rmc

history = np.array([
    5.07, 7.27, 6.74, 5.82, 6.97, 6.01, 7.62, 5.85, 6.01, 5.44, 5.82, 7.15,
    6.67, 8.81, 7.67, 5.28, 5.31, 8.72, 8.25, 7.91, 10.08, 7.87, 6.36, 6.14,
    8.82, 7.43, 7.84, 7.69, 6.86, 6.21, 7.10, 8.09, 7.07, 7.23, 7.42, 7.69,
])  # millions of dollars
model = rmc.BayesianLocalLevel(
    process_variance_prior=rmc.InverseGammaPrior(shape=3.0, scale=0.004),
    observation_variance_prior=rmc.InverseGammaPrior(shape=3.0, scale=0.04),
    initial_mean=float(np.log(history[0])),
    initial_variance=1.0,
)
fit = model.fit(np.log(history), chains=4, draws=2000, warmup=1000, seed=7)
print(fit.summary())

# Joint paths for the next three months, shaped (chain, draw, month).
forecast = rmc.ForecastDraws(np.exp(fit.forecast(steps=3, seed=8).observation_samples))
lower, upper = forecast.interval(0.95)  # equal-tailed, pointwise

loss = rmc.AsymmetricLinearLoss(under=5, over=1)
reserve = forecast.optimal_decision(loss, steps=[0])
print(reserve.decision)            # about 8.65
print(reserve.quantile)            # 5 / (5 + 1) = 0.833
print(reserve.expected_loss, reserve.expected_loss_mcse)
print(reserve.decision_mcse)       # Monte Carlo standard error of the reserve
```

The answer is about $8.65M: the 83rd percentile of the forecast. With these costs,
reserving the mean ($7.49M) has an expected cost of about 2.96 against 2.01 at the
optimum. With `over=1` read as one dollar of cost per dollar over-reserved, these are
expected costs in millions of dollars.

This is the newsvendor result. With a cost of `under` per unit short and `over` per
unit in excess, the expected cost is lowest at the `under / (under + over)` quantile of
the outcome. Only the ratio of the two costs matters for the decision.

To see what another reserve would cost:

```python
for amount in [7.49, 8.65, 10.18]:
    cost = forecast.expected_loss(amount, loss, steps=[0])
    print(amount, cost.value, cost.mcse)
```

## Losses

| Loss | Cost of deciding `a` when the outcome is `y` | Best decision |
|---|---|---|
| `AsymmetricLinearLoss(under=u, over=o)` | `u * (y - a)` if `y > a`, else `o * (a - y)` | the `u / (u + o)` quantile |
| `SquaredLoss()` | `(y - a) ** 2` | the mean |
| `AbsoluteLoss()` | `abs(y - a)` | the median |
| `PiecewiseLinearLoss(shortfall, loss)` | linear between knots in `y - a` | best candidate on a grid |
| a function `f(a, outcomes)` | whatever it returns, one value per draw | best candidate on a grid |

For the first three, `decide` uses the known optimum: a quantile, the mean or the
median of the draws. The quantile follows the
same rule as every rustmc interval: linear interpolation between sorted draws, which
matches NumPy's default `np.quantile`. That can sit up to one gap between neighbouring
draws away from the exact minimiser of the average loss, which costs at most
`(u + o) / n` times that gap for `n` draws. The mean and the median are exact.

## Totals over several months

A reserve for the next quarter covers the total of three months. Pass the months to
total with `steps`, numbered from 0:

```python
quarter = forecast.optimal_decision(loss, steps=range(3))
monthly = sum(forecast.optimal_decision(loss, steps=[m]).decision for m in range(3))
print(quarter.decision, monthly)   # about 25.0 against 26.1
```

Each total is summed within one forecast path, so it keeps the dependence between
months. That is why it needs joint paths, and why it differs from adding three monthly
reserves: the 83rd percentile of a total is not the total of the 83rd percentiles.
Here the sum of monthly reserves costs about 4.51 in expectation against 4.29 for the
quarter's own optimum.

`steps=range(3)` gives the same totals as step 2 of
`forecast.cumulative_observation_samples`. Any set of months works, in any order, such
as `steps=[3, 5]`. Draws with a series axis must be combined first, for example with
`ForecastDraws.aggregate()`.

## Your own loss

A piecewise-linear loss describes tiered costs. Here a shortfall beyond $0.5M costs
10 per dollar instead of 5. Its knots are in the shortfall `outcome - decision`, and it
continues along its first and last segments past the outer knots:

```python
tiered = rmc.PiecewiseLinearLoss(shortfall=[-1.0, 0.0, 0.5, 1.5], loss=[1.0, 0.0, 2.5, 12.5])
grid = np.arange(7.0, 12.0, 0.01)
tiered_reserve = forecast.optimal_decision(tiered, steps=[0], candidates=grid)
print(tiered_reserve.decision, tiered_reserve.on_grid_edge)
```

Any other loss can be a Python function of the decision and the draws. It receives
the decision as a float and the draws as a read-only `(chain, draw)` array, and must
return one loss per draw. Each loss should depend only on the decision and its own
draw: the block decisions described below average these losses over parts of each
chain.

```python
def with_penalty(decision, outcomes):
    shortfall = outcomes - decision
    linear = np.where(shortfall > 0, 5 * shortfall, -shortfall)
    return linear + 3.0 * (shortfall > 1.0)  # a fixed cost for missing by over $1M

penalised = forecast.optimal_decision(with_penalty, steps=[0], candidates=grid)
print(penalised.decision)
```

rustmc uses no formula for the minimum of these losses, so the answer is the best of
the `candidates`, and it is no finer than their spacing. Without `candidates` the grid
is 201 evenly spaced points from the smallest draw to the largest, or fewer when the
draws are constant or span too few distinct values. If the best candidate is the first
or last one (or the only one), `on_grid_edge` is `True` and the true minimum may lie
outside the grid; widen it. A Python loss is called once per candidate, so it runs at
Python speed. The built-in losses run in Rust.

`method="grid"` searches the candidates for the built-in losses too. Use it when only
some decisions are possible, such as reserves in whole $0.25M steps.

## Draws from anywhere

`rmc.decide` and `rmc.expected_loss` take a NumPy array shaped `(draw,)`,
`(chain, draw)` or `(chain, draw, horizon)`, a `ForecastDraws`, or any rustmc forecast
result with `observation_samples`:

```python
draws = np.random.default_rng(1).normal(7.5, 1.28, size=(4, 1000))
print(rmc.decide(draws, loss).decision)
print(rmc.expected_loss(draws, 8.7, loss).value)

native = fit.forecast(steps=3, seed=8)  # log scale here; decide on its draws directly
print(rmc.decide(native, rmc.SquaredLoss(), steps=[0]).decision)
```

Arrays follow the same exact-conversion rule as model data: booleans, strings, values
float64 would round, and non-finite draws raise a `ValueError` naming the argument.
The rule applies to arrays passed to `decide` and `expected_loss` directly. A
`ForecastDraws` converts its input with NumPy's float conversion when it is built, so
a value float64 would round is rounded there rather than refused.

## How far to trust the answer

Every number here is an average over a finite set of draws, so each carries Monte Carlo
error. The result reports it.

- `expected_loss_mcse` (and `ExpectedLoss.mcse`) is the standard deviation of the
  per-draw losses divided by the square root of their effective sample size. The
  effective sample size comes from split chains, as in the `mcse_mean` column of a fit
  summary, so autocorrelated MCMC draws are not treated as independent. A single array
  of draws is treated as one chain in the order given. With fewer than four draws per
  chain it is `nan`, unless every loss is equal, when it is zero.
- `decision_mcse` is a batch-means standard error of the decision. The best decision is
  recomputed on contiguous blocks of each chain, at least eight blocks in all when the
  chains are long enough (halves of each of four chains, quarters of each of two), and
  `batch_decisions` holds those block decisions, shaped `(chain, block)`. Blocks from
  chains that disagree spread further apart, which shows up here. For the pinball
  loss in 3,000 simulated replicates, this estimate ran 3% to 7% below the observed
  spread of the decision, so read it as approximate. The run is an ignored test,
  `decision_mcse_calibration_over_3000_replicates` in `rust_core/src/decision.rs`.
- `decision_mcse` is `nan` for a grid search. The best candidate jumps between grid
  points rather than moving smoothly with the draws, and on a coarse grid the
  batch-means formula understated the spread of the decision threefold. It is also
  `nan` for a quantile at 0 or 1 (the smallest or largest draw). `batch_decisions` is
  still reported, and shows how much the choice moves.
- The expected loss at the chosen decision is biased low. The decision was picked to
  make that average small on these same draws, and its standard error treats the
  decision as fixed.

None of this covers error in the forecast itself. The decision is exactly as good as
the predictive distribution behind it, so check that distribution on held-out periods
first; see [Forecasting workflows](forecasting-workflows.md). A decision layer also
cannot tell you what the costs are: `under` and `over` are business inputs, and the
answer moves with their ratio.

A complete script is in
[`examples/reserve_decision.py`](https://github.com/tbosier/rustmc/blob/main/examples/reserve_decision.py).
