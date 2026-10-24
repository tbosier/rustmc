"""Panels of ten datasets from the auto-dense study, for the metric test's thresholds.

`default_metric_keeps_the_dense_benefit_for_correlated_coefficients` in
`rust_core/src/nuts.rs` sums auto's and dense's leapfrog steps over datasets 1 to 10.
This draws subsets of ten distinct held-out datasets (seeds 21 to 60) from
`benchmarks/results/2026-10-24-auto-dense-50d.txt`, with a fixed seed, and prints how
the ratio of those sums is distributed for each build, how often it exceeds a few
thresholds, and what it becomes on datasets 1 to 10 when auto's steps on the k
datasets where diagonal costs least extra are replaced by diagonal's.

    python benchmarks/auto_dense_panels.py [record]
"""
from __future__ import annotations

import random
import re
import sys
from pathlib import Path

RECORD = Path(__file__).parent / "results" / "2026-10-24-auto-dense-50d.txt"
LINE = re.compile(r"data (\d+): auto ([\d.]+) diagonal ([\d.]+) dense ([\d.]+) kept (\w+)")
PANELS = 200_000


def read(path: Path) -> dict[str, dict[int, tuple[float, float, float]]]:
    builds: dict[str, dict[int, tuple[float, float, float]]] = {}
    build = None
    for line in path.read_text().splitlines():
        if line.startswith("## "):
            build = line[3:]
            builds[build] = {}
        match = LINE.match(line)
        if match and build is not None:
            builds[build][int(match[1])] = tuple(float(match[i]) for i in (2, 3, 4))
    return builds


def ratio(rows) -> float:
    return sum(auto for auto, _, _ in rows) / sum(dense for _, _, dense in rows)


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else RECORD
    for build, rows in read(path).items():
        held_out = [rows[seed] for seed in range(21, 61)]
        rng = random.Random(20261024)
        ratios = sorted(ratio(rng.sample(held_out, 10)) for _ in range(PANELS))
        quantile = lambda q: ratios[int(q * (PANELS - 1))]
        print(build)
        print(f"  held-out panels of 10: median {quantile(0.5):.2f}, 99% {quantile(0.99):.2f}, "
              f"99.9% {quantile(0.999):.2f}, largest {ratios[-1]:.2f}")
        for threshold in (2.5, 3.0, 3.5):
            share = sum(value > threshold for value in ratios) / PANELS
            print(f"  share above {threshold}: {share:.5f}")
        test = [rows[seed] for seed in range(1, 11)]
        print(f"  datasets 1 to 10: {ratio(test):.2f}")
        cheapest = sorted(range(10), key=lambda i: test[i][1] - test[i][0])
        for k in range(1, 6):
            switched = [(test[i][1] if i in cheapest[:k] else test[i][0], 0.0, test[i][2])
                        for i in range(10)]
            print(f"  with {k} switched to diagonal: {ratio(switched):.2f}")


if __name__ == "__main__":
    main()
