"""Compare two saved runs of the auto metric benchmarks, fit by fit.

Run: python -m benchmarks.compare_metric_records OLD.json NEW.json

For ``auto_metric_targets`` records it prints, per target and metric, the median and
worst leapfrog steps per unit of the smallest bulk ESS in each run, the median of the
paired per-seed ratio (new over old), and the fits with divergences. For
``curved_metric_choice`` records it lists, per metric, the seeds whose fit changed.
Fits are paired by target, metric and seed, so records with several metrics compare
each metric with itself.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np


def load(path):
    with open(path) as handle:
        return json.load(handle)


def targets(old, new):
    def by_target(record):
        rows = defaultdict(dict)
        for fit in record["fits"]:
            rows[(fit["target"], fit["metric"])][fit["seed"]] = fit
        return rows

    old_rows, new_rows = by_target(old), by_target(new)
    print(f"{'target':34s} {'metric':6s} {'old med':>8s} {'new med':>8s} {'old worst':>9s} "
          f"{'new worst':>9s} {'paired':>7s} {'div old':>7s} {'div new':>7s}")
    for (target, metric), old_fits in old_rows.items():
        new_fits = new_rows.get((target, metric))
        if new_fits is None:
            continue
        seeds = sorted(set(old_fits) & set(new_fits))
        cost = lambda fit: fit["leapfrog_steps"] / fit["min_bulk_ess"]
        a = np.array([cost(old_fits[s]) for s in seeds])
        b = np.array([cost(new_fits[s]) for s in seeds])
        divergent = lambda fits: sum(fits[s]["divergences"] > 0 for s in seeds)
        print(f"{target:34s} {metric:6s} {np.median(a):8.1f} {np.median(b):8.1f} {a.max():9.1f} "
              f"{b.max():9.1f} {np.median(b / a):7.3f} {divergent(old_fits):7d} "
              f"{divergent(new_fits):7d}")


def curved(old, new):
    old_fits = {(fit["metric"], fit["seed"]): fit for fit in old["fits"]}
    new_fits = {(fit["metric"], fit["seed"]): fit for fit in new["fits"]}
    keys = sorted(set(old_fits) & set(new_fits))
    changed = [k for k in keys if old_fits[k]["min_bulk_ess"] != new_fits[k]["min_bulk_ess"]]
    print(f"{len(changed)} of {len(keys)} fits changed")
    for k in changed:
        o, n = old_fits[k], new_fits[k]
        print(f"{k[0]} seed {k[1]}: bulk ESS {o['min_bulk_ess']:.1f} -> {n['min_bulk_ess']:.1f}, "
              f"divergences {o['divergences']} -> {n['divergences']}, "
              f"R-hat {o['max_r_hat']:.4f} -> {n['max_r_hat']:.4f}")


def main():
    old, new = load(sys.argv[1]), load(sys.argv[2])
    (targets if "target" in old["fits"][0] else curved)(old, new)


if __name__ == "__main__":
    main()
