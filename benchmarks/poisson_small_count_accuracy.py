"""Error in ulps of the old and new small-count Poisson log mass against mpmath at 60 digits.

`log_mass_from_log_rate_given_rate` in `rust_core/src/count_sampling.rs` used to compute,
for a count from 1 to 15, `count * ln(exp(eta)) - exp(eta) - ln(count!)`, and now computes
`count * eta - exp(eta) - ln(count!)`. Both are evaluated here in the same order with
Python's `math.exp` and `math.log`, which on Linux are the glibc functions the Rust build
calls, and with the table of `ln(count!)` summed as the Rust table sums it. The eta grid
samples the range of normal, finite rates, from `ln(f64::MIN_POSITIVE)` to `ln(f64::MAX)`,
with denser points near zero, near both ends and near `ln(count)`. The figures are maxima
and means over that sample, not bounds over every input. For each count the script also
prints the eta and rate where each form's largest error fell.

Needs mpmath, which rustmc does not depend on:

    python benchmarks/poisson_small_count_accuracy.py
"""
import math
import sys
import numpy as np
import mpmath

mpmath.mp.dps = 60
MIN_POSITIVE = 2.2250738585072014e-308
table = []
for count in range(16):
    total = -0.0
    for k in range(2, count + 1):
        total += math.log(k)
    table.append(total)

rng = np.random.default_rng(20261025)
boundary = math.log(MIN_POSITIVE)
top = math.log(sys.float_info.max)
etas = np.concatenate([
    np.linspace(boundary, top, 4000),
    top - np.linspace(0.0, 1e-9, 200),
    np.linspace(-5.0, 5.0, 4000),
    rng.uniform(-40.0, 40.0, 4000),
    boundary + np.linspace(0.0, 1e-9, 200),
    (np.log(np.arange(1, 16))[None, :] + rng.normal(0, 1e-6, size=(50, 15))).ravel(),
])
results = {}
for count in range(1, 16):
    worst = {"old": 0.0, "new": 0.0}
    where = {"old": None, "new": None}
    total = {"old": 0.0, "new": 0.0}
    n = 0
    for eta in etas:
        eta = float(eta)
        rate = math.exp(eta)
        if not math.isfinite(rate) or rate < MIN_POSITIVE:
            continue
        old = count * math.log(rate) - rate - table[count]
        new = count * eta - rate - table[count]
        reference = (count * mpmath.mpf(eta) - mpmath.exp(mpmath.mpf(eta))
                     - mpmath.loggamma(count + 1))
        ulp = math.ulp(float(reference))
        for name, value in (("old", old), ("new", new)):
            err = float(abs(mpmath.mpf(value) - reference)) / ulp
            if err > worst[name]:
                worst[name] = err
                where[name] = (eta, rate)
            total[name] += err
        n += 1
    results[count] = (n, worst, {k: v / n for k, v in total.items()})
    print(f"count {count:2d}: {n} etas; max ulps old {worst['old']:.3g} new {worst['new']:.3g}; "
          f"mean old {total['old']/n:.3g} new {total['new']/n:.3g}; "
          f"old max at eta {where['old'][0]:.6g} (rate {where['old'][1]:.4g}), "
          f"new at eta {where['new'][0]:.6g} (rate {where['new'][1]:.4g})", flush=True)
print("overall max old", max(r[1]["old"] for r in results.values()),
      "new", max(r[1]["new"] for r in results.values()))
