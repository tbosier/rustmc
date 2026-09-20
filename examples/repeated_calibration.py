"""Repeated calibration

Fit independent instrument datasets with one compiled regression model. Each
instrument can have a different number of readings. Stable IDs keep each fit's
random stream tied to its instrument, so reordering the batch does not change
any instrument's result.

This does not pool instruments: every fit is independent and learns nothing from
its neighbours. Use `examples/site_effects.py` when group estimates should share
information. Chunked batch dispatch currently retains inputs and fits; see the
roadmap for bounded streaming.
"""

# %%
import numpy as np

from instrument_calibration import calibration_model

# %% Three instruments with different sample counts
rng = np.random.default_rng(42)
datasets = []
ids = []
for i, rows in enumerate([40, 75, 120]):
    x = np.linspace(-2, 2, rows)
    y = 0.1 * i + (1.0 + 0.05 * i) * x + rng.normal(0, 0.2, rows)
    ids.append(f"instrument-{i}")
    datasets.append({"x": x, "y": y})

for name, dataset in zip(ids, datasets):
    print(f"{name}: {len(dataset['x'])} readings")

# %% [markdown]
# ## One compiled model, many datasets
#
# `calibration_model()` comes from `examples/instrument_calibration.py`: an
# offset, a gain and a noise scale, fitted to `y ~ Normal(offset + gain * x, noise)`.
# Compiling once and calling `sample_batch` builds the graph structure once and
# reuses it for every dataset. Each fit still validates its own binding and lays
# out its own evaluator buffers, because those depend on the data's shapes: what
# is saved is the model construction, not the per-fit setup.
#
# `errors="collect"` keeps one bad dataset from losing the whole batch: the failing
# ID lands in `batch.errors` and the rest still return fits.

# %% Fit the batch
batch = calibration_model().sample_batch(
    datasets,
    ids=ids,
    chains=4,
    warmup=1000,
    draws=1000,
    threads=2,
    seed=42,
    errors="collect",
    show_progress=False,
)

for name in batch.ids:
    print(f"===== {name}")
    if name in batch.errors:
        print(batch.errors[name])
    else:
        print(batch.get(name).summary())

# %% [markdown]
# ## Reading the result
#
# The true offsets are 0.0, 0.1 and 0.2 and the true gains 1.00, 1.05 and 1.10.
# The instrument with 40 readings has the widest intervals, as it should: fewer
# readings, less information, and no borrowing from the other two.
#
# Interval width does not fall monotonically with sample size here. Instrument 2
# has 120 readings against instrument 1's 75 and still reports slightly wider
# intervals. The counts are fixed at 40, 75 and 120; the noise scale is not, and
# instrument 2's came out higher on this simulated draw. Width falls with the
# count and rises with the noise, so a larger instrument can still be the less
# precisely measured one.
#
# Check `r_hat` and `ess_bulk` per instrument before comparing them. A batch
# reports diagnostics per fit for exactly this reason -- one instrument can fail
# to converge while the others are fine.
