"""Graph NUTS on a curved two-dimensional target under each ``metric``.

Run: python -m benchmarks.curved_metric_choice  (with the build to measure on
PYTHONPATH)

The target is ``x ~ Normal(0, 25)`` and ``y | x ~ Normal(0.5 (x^2 - 25), 1)``, written
as one two-element vector parameter so that ``metric="auto"`` may choose a dense block.
Its broad ``vector_normal_prior`` is cancelled in the potential. For seeds 10 to 14 it
prints, per metric, the smallest bulk ESS, the largest R-hat and the divergences of a
fit with 4 chains, 500 warmup and 1000 draws. A dense metric fitted to this target can
be worse than none; the statistical-validation page uses this script to compare ways of
choosing the metric.
"""
from __future__ import annotations

import numpy as np
import rustmc as rmc

SEEDS = range(10, 15)
DATA = {"i": np.array([0.0]), "j": np.array([1.0])}


def model():
    b = rmc.ModelBuilder(DATA)
    q = b.vector_normal_prior("q", 2, 0.0, 1000.0)
    x, y = q["i"], q["j"]
    b.potential(
        "banana",
        (-0.5 * (x / 5) ** 2 - 0.5 * (y - 0.5 * (x * x - 25)) ** 2
         + 0.5 * (x * x + y * y) / 1e6).sum(),
    )
    return b.compile()


def main():
    print(f"rustmc {rmc.__version__} ({rmc.__file__})")
    print(f"chains=4 warmup=500 draws=1000 seeds={list(SEEDS)}")
    print(f"{'metric':6s} {'seed':>4s} {'min ESS':>8s} {'max R-hat':>9s} {'divergences':>11s}")
    compiled = model()
    for metric in ("auto", "diag", "dense"):
        for seed in SEEDS:
            fit = compiled.sample(DATA, chains=4, warmup=500, draws=1000, seed=seed,
                                  metric=metric, show_progress=False)
            rows = fit.diagnostics()
            ess = min(row["ess_bulk"] for row in rows)
            r_hat = max(row["r_hat"] for row in rows)
            divergences = fit.transition_diagnostics()["total_divergences"]
            print(f"{metric:6s} {seed:4d} {ess:8.1f} {r_hat:9.4f} {divergences:11d}")


if __name__ == "__main__":
    main()
