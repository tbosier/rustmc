"""Reproducible batch throughput/peak-RSS probe; outputs JSON, no speedup claims.

Run each thread count in a separate process so peak RSS is comparable:
    python benchmarks/forecast_batch.py --threads 1
    python benchmarks/forecast_batch.py --threads 4
"""
import argparse
import json
import platform
import resource
import subprocess
import time

import numpy as np
import rustmc as rmc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--cells", type=int, default=128)
    args = parser.parse_args()
    rng = np.random.default_rng(914)
    observations = []
    for cell in range(args.cells):
        count = 48 + cell % 13
        level = np.cumsum(rng.normal(0, 0.2, count))
        observations.append(level + rng.normal(0, 0.4, count))
    ids = [f"cell/{i}" for i in range(args.cells)]
    prior = rmc.InverseGammaPrior(3.0, 0.3)
    model = rmc.BayesianLocalLevel(process_variance_prior=prior,
                                   observation_variance_prior=prior)
    started = time.perf_counter()
    batch = model.fit_batch(observations, ids, chains=4, draws=200, warmup=100,
                            seed=91, threads=args.threads, chunk_size=32)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    forecast = batch.forecast(12, seed=92, threads=args.threads, chunk_size=32)
    forecast_seconds = time.perf_counter() - started
    # Diagnostics are outside throughput timings and use every retained cell.
    report = batch.diagnostics()
    parameters = [parameter for cell in report.values() for parameter in cell]
    print(json.dumps({
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "platform": platform.platform(), "python": platform.python_version(),
        "numpy": np.__version__, "threads": args.threads, "cells": args.cells,
        "chains": 4, "draws": 200, "warmup": 100, "history_range": [48, 60],
        "horizon": 12, "chunk_size": 32, "fit_seed": 91, "forecast_seed": 92,
        "fit_seconds": fit_seconds, "fit_cells_per_second": args.cells / fit_seconds,
        "forecast_seconds": forecast_seconds,
        "peak_rss_kib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "max_rhat": max(parameter["r_hat"] for parameter in parameters),
        "min_bulk_ess": min(parameter["ess_bulk"] for parameter in parameters),
        "min_tail_ess": min(parameter["ess_tail"] for parameter in parameters),
        "retained_fit_cells": len(batch), "retained_forecast_cells": len(forecast),
    }, indent=2))


if __name__ == "__main__":
    main()
