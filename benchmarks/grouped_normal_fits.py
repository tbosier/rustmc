"""Whole-fit time of the non-centered varying-intercept model, as benchmarks/head_to_head.py builds it.

Usage: fit_time.py LABEL ROWS GROUPS. Median over 3 datasets of seconds per fit (4 chains of
1000 warmup and 1000 draws on one thread), total leapfrog steps, and, on builds that have the
counter, the fused grouped-Normal reverse passes the fits ran.
"""
import json
import sys
import time

import numpy as np
import rustmc as rmc
from rustmc import _rustmc

label, rows, groups = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
b = rmc.ModelBuilder()
mu = b.normal_prior("mu", 0.0, 5.0)
tau = b.half_normal_prior("tau", 1.0)
z = b.vector_normal_prior("z", n=groups, mu=0.0, sigma=1.0)
sigma = b.half_normal_prior("sigma", 1.0)
b.normal_likelihood("obs", mu + tau * z["group"], sigma, "y")
compiled = b.compile()
counter = getattr(_rustmc, "_fused_grouped_normal_passes", None)
seconds, steps, fused = [], [], 0
for seed in range(3):
    rng = np.random.default_rng([seed, rows, groups, 0])
    group = rng.integers(0, groups, size=rows)
    y = 0.4 + 0.7 * rng.normal(size=groups)[group] + rng.normal(size=rows)
    before = counter()[1] if counter else 0
    start = time.perf_counter()
    fit = compiled.sample({"group": group.astype(float), "y": y}, chains=4, warmup=1000,
                          draws=1000, seed=seed, threads=1, show_progress=False)
    seconds.append(time.perf_counter() - start)
    fused += (counter()[1] - before) if counter else 0
    steps.append(fit.transition_diagnostics()["total_leapfrog_steps"])
print(json.dumps({"build": label, "rows": rows, "groups": groups,
                  "seconds": [round(s, 3) for s in seconds], "median_seconds": round(float(np.median(seconds)), 3),
                  "leapfrog_steps": steps, "fused_reverse_passes": fused if counter else None}), flush=True)
