"""Panels of ten datasets from the auto-dense study, for the metric test's thresholds.

`default_metric_keeps_the_dense_benefit_for_correlated_coefficients` in
`rust_core/src/nuts.rs` sums auto's and dense's leapfrog steps over datasets 1 to 10.
This draws subsets of ten distinct held-out datasets (seeds 21 to 60) from
`benchmarks/results/2026-10-24-auto-dense-50d.txt`, with a fixed seed, and prints how
the ratio of those sums is distributed for each build, how many panels fail the test's
assertion (at least 6 of 10 datasets keeping the dense benefit, and the ratio at most
3.5) and each half of it, and what the ratio becomes on datasets 1 to 10 when auto's
steps on the k datasets where diagonal costs least extra are replaced by diagonal's.

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


def read(path: Path) -> dict[str, dict[int, tuple[float, float, float, bool]]]:
    builds: dict[str, dict[int, tuple[float, float, float, bool]]] = {}
    build = None
    for line in path.read_text().splitlines():
        if line.startswith("## "):
            build = line[3:]
            builds[build] = {}
        match = LINE.match(line)
        if match and build is not None:
            builds[build][int(match[1])] = (
                float(match[2]), float(match[3]), float(match[4]), match[5] == "true"
            )
    return builds


MIN_KEPT = 6
MAX_RATIO = 3.5


def ratio(rows) -> float:
    return sum(row[0] for row in rows) / sum(row[2] for row in rows)


def kept(rows) -> int:
    return sum(row[3] for row in rows)


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else RECORD
    for build, rows in read(path).items():
        held_out = [rows[seed] for seed in range(21, 61)]
        rng = random.Random(20261024)
        panels = [rng.sample(held_out, 10) for _ in range(PANELS)]
        ratios = sorted(ratio(panel) for panel in panels)
        quantile = lambda q: ratios[int(q * (PANELS - 1))]
        print(build)
        print(f"  held-out panels of 10: median {quantile(0.5):.2f}, 99% {quantile(0.99):.2f}, "
              f"99.9% {quantile(0.999):.2f}, largest {ratios[-1]:.4f}")
        for threshold in (2.5, 3.0, MAX_RATIO):
            count = sum(value > threshold for value in ratios)
            print(f"  panels with a ratio above {threshold}: {count} of {PANELS}")
        too_few = sum(kept(panel) < MIN_KEPT for panel in panels)
        failing = sum(kept(panel) < MIN_KEPT or ratio(panel) > MAX_RATIO for panel in panels)
        print(f"  panels keeping the benefit on fewer than {MIN_KEPT}: {too_few} of {PANELS}")
        print(f"  panels failing the test's assertion: {failing} of {PANELS}")
        test = [rows[seed] for seed in range(1, 11)]
        always_diagonal = [(row[1], 0.0, row[2], False) for row in test]
        print(f"  datasets 1 to 10: kept {kept(test)}, ratio {ratio(test):.2f}, "
              f"{ratio(always_diagonal):.2f} if auto were always diagonal")
        cheapest = sorted(range(10), key=lambda i: test[i][1] - test[i][0])
        for k in range(1, 6):
            switched = [(test[i][1] if i in cheapest[:k] else test[i][0], 0.0, test[i][2], False)
                        for i in range(10)]
            print(f"  with {k} switched to diagonal: {ratio(switched):.2f}")


if __name__ == "__main__":
    main()
