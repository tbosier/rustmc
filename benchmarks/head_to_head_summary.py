"""Verdicts for a ``benchmarks/head_to_head.py`` record, recomputed from its fits.

Lists every cell that ran and every problem in it: an engine or fit that failed,
a fit that cannot enter a verdict (its load wait timed out, a diagnostic is not
finite, R-hat is above 1.01, a post-warmup transition diverged, or its smallest
bulk ESS is under 400), a mean-difference test that crossed its 0.1% family-wise
threshold across every parameter of every pair in the cell, and a rustmc extension
whose SHA-256 differs from the record's.

Then, per cell and comparison, the per-ESS ratios (rustmc over nutpie) of the
pairs where both fits pass, their median and observed range (not an interval),
the distribution-free order-statistic interval for the median with its exact
coverage, and two verdicts:

- unadjusted: a two-sided sign test at 5% for that cell alone, which is
  exploratory, since the study makes many such tests;
- Holm-adjusted, within a family declared before looking: one comparison (rustmc
  against nutpie at matched warmup, or against nutpie at its default warmup) and
  one model class, across all of that class's sizes and both thread counts. Holm
  holds under any dependence, which matters because a cell's two thread settings
  share draws and the two comparisons share rustmc's fits. An inconclusive cell
  stays in its family with a p-value of 1, so rejecting one cell never shrinks
  the correction another gets.

A cell whose mean-difference test fired, or whose extension differs, gets no
winner in either verdict. Class aggregates pool only the passing pairs of cells
that are not inconclusive; every finished pair's ratio is printed besides.

    python benchmarks/head_to_head_summary.py RECORD.json [--endpoint sampling|native]

``--endpoint`` picks the span the per-ESS ratios use: the default InferenceData
(``seconds``, posterior draws and post-warmup sample statistics), the end of
sampling, or each engine's native default output. Records written before schema 2
should be read with ``sampling``: their InferenceData spans held different groups
for the two engines.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.head_to_head import (
    Cell,
    compare,
    fit_problems,
    holm,
    paired_by_index,
    scalar_names,
    winner,
)

ENDPOINTS = {"inferencedata": "seconds", "sampling": "sampling_seconds", "native": "native_seconds"}
COMPARISONS = ("nutpie", "nutpie_default_tune")


def normalized(result: dict) -> dict:
    """Older records kept no per-fit status: their fits all finished."""
    fits = [dict(fit, status=fit.get("status", "ok")) for fit in result.get("fits", [])]
    return dict(result, fits=fits)


def fmt(values, digits=2):
    if not values:
        return "-"
    return f"{np.median(values):.{digits}f} [{min(values):.{digits}f}, {max(values):.{digits}f}]"


def adjusted_verdict(c: dict) -> str:
    if c["verdict"].startswith("inconclusive"):
        return c["verdict"]
    if c["holm_p"] is not None and c["holm_p"] < 0.05:
        return winner(c["ratio_seconds_per_ess"])
    return "no clear winner"


def analyze(record: dict, time_key: str) -> dict:
    """Every comparison the record holds, with its problems and the nutpie compile
    notes. A comparison is made for every engine the cell ran, even one with no
    fits, so a failed engine stays in its Holm family as an inconclusive cell."""
    expected_sha = record.get("extension_sha256")
    notes, problems, comparisons = [], [], []
    for row in record["cells"]:
        cell = Cell(**row["config"])
        results = {engine: normalized(result) for engine, result in row["results"].items()}
        if row.get("load_wait_timed_out"):
            # Schema 1 recorded one load wait per cell.
            for result in results.values():
                for fit in result["fits"]:
                    fit["load_wait_timed_out"] = True
        sha = results.get("rustmc", {}).get("provenance", {}).get("extension_sha256")
        provenance_ok = expected_sha is None or sha == expected_sha
        if not provenance_ok:
            problems.append(f"{row['cell']}: rustmc loaded extension {sha}, not {expected_sha}")
        for engine, result in results.items():
            if result.get("status") not in ("ok", "partial"):
                problems.append(f"{row['cell']}: {engine} {result.get('status')}: {result.get('reason')}")
            if result.get("compile_load_wait_timed_out"):
                problems.append(f"{row['cell']}: {engine} compiled after a timed-out load wait")
            if engine.startswith("nutpie") and result.get("compile_cache_files_before") is not None:
                before = result["compile_cache_files_before"]
                notes.append(f"{row['cell']}: {engine} compile {result.get('compile_seconds', 0):.2f} s, "
                             f"{'empty cache directory' if before == 0 else f'non-empty cache, {before} files'}")
            for fit in result["fits"]:
                reasons = fit_problems(fit)
                if reasons:
                    problems.append(f"{row['cell']}: {engine} dataset {fit['index']}: " + ", ".join(reasons))
        for other in COMPARISONS:
            if other not in results or "rustmc" not in results:
                continue
            c = compare(results["rustmc"], results[other], len(scalar_names(cell)), time_key,
                        provenance_ok)
            if c["mean_difference_flag"]:
                problems.append(f"{row['cell']}: rustmc and {other} means differ or cannot be "
                                f"compared, max z {c['max_mean_difference_z']:.1f} over "
                                f"{c['mean_difference_tests']} tests (threshold "
                                f"{c['mean_difference_threshold']:.1f})")
            comparisons.append(dict(c, cell=row["cell"], model=cell.model, threads=cell.threads,
                                    other=other))
    adjust_within_families(comparisons)
    return {"notes": notes, "problems": problems, "comparisons": comparisons}


def adjust_within_families(comparisons: list[dict]) -> None:
    """Holm within each declared family, one comparison and one model class across
    its sizes and thread counts. An inconclusive member counts with p = 1."""
    families: dict[tuple[str, str], list[dict]] = {}
    for c in comparisons:
        families.setdefault((c["other"], c["model"]), []).append(c)
    for members in families.values():
        inconclusive = [c["verdict"].startswith("inconclusive") for c in members]
        p_values = [1.0 if skip else c["sign_test_p"] for c, skip in zip(members, inconclusive)]
        for c, skip, adjusted in zip(members, inconclusive, holm(p_values)):
            c["holm_p"] = None if skip else adjusted
            c["family_size"] = len(members)
            c["adjusted_verdict"] = adjusted_verdict(c)


def class_aggregates(comparisons: list[dict]) -> list[dict]:
    """Per comparison, model class and thread count: the passing pairs' ratios of the
    cells that are not inconclusive, and the count of each adjusted verdict."""
    rows = []
    for other in COMPARISONS:
        for model in dict.fromkeys(c["model"] for c in comparisons):
            for threads in (1, 4):
                chosen = [c for c in comparisons
                          if c["model"] == model and c["other"] == other and c["threads"] == threads]
                if not chosen:
                    continue
                values = [x for c in chosen if not c["verdict"].startswith("inconclusive")
                          for x in c["ratio_seconds_per_ess"]]
                counts: dict[str, int] = {}
                for c in chosen:
                    counts[c["adjusted_verdict"]] = counts.get(c["adjusted_verdict"], 0) + 1
                rows.append({"other": other, "model": model, "threads": threads,
                             "values": values, "verdicts": counts})
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("record", type=Path)
    parser.add_argument("--endpoint", choices=tuple(ENDPOINTS), default="inferencedata")
    args = parser.parse_args(argv)
    time_key = ENDPOINTS[args.endpoint]
    record = json.loads(args.record.read_text())
    expected_sha = record.get("extension_sha256")
    analysis = analyze(record, time_key)

    print(f"revision {record['revision']}, schema {record.get('schema', 1)}, versions {record['versions']}")
    print(f"extension sha256 {expected_sha or 'not recorded'}")
    print(f"compile cache directory at start: {record.get('compile_cache', 'not recorded')}")
    print(f"per-ESS ratios use {time_key}")
    print(f"cells run: {len(record['cells'])}")
    for note in analysis["notes"]:
        print("  " + note)

    print()
    print(f"{'cell':<36} {'vs':<9} {'pass':>5} {'per-ESS ratio, passing pairs':<28} "
          f"{'order-stat interval (coverage)':<32} {'sign p':>7} {'Holm p':>7} "
          f"{'unadjusted':<24} {'Holm-adjusted':<24} {'max z / thr':<12} all finished pairs")
    for c in analysis["comparisons"]:
        vs = "nutpie" if c["other"] == "nutpie" else "nutpie300"
        p = "-" if c["sign_test_p"] is None else f"{c['sign_test_p']:.3f}"
        hp = "-" if c["holm_p"] is None else f"{c['holm_p']:.3f}"
        interval = c["median_interval"]
        iv = ("-" if interval is None else
              f"[{interval['low']:.2f}, {interval['high']:.2f}] ({100 * interval['coverage']:.1f}%)")
        ratios = " ".join(f"{x:.2f}" for x in c["all_ratio_seconds_per_ess"])
        print(f"{c['cell']:<36} {vs:<9} {c['passing_pairs']:>2}/{c['pairs']:<2} "
              f"{fmt(c['ratio_seconds_per_ess']):<28} {iv:<32} {p:>7} {hp:>7} "
              f"{c['verdict']:<24} {c['adjusted_verdict']:<24} "
              f"{c['max_mean_difference_z']:.1f}/{c['mean_difference_threshold']:.1f}    {ratios}")

    print()
    print(f"Problems ({len(analysis['problems'])}):")
    for line in analysis["problems"]:
        print("  " + line)

    print()
    print("By model class: per-ESS ratios of every passing pair in the class, median and observed "
          "range (n values; a cell's two thread settings share draws), and verdict counts over "
          "cells, Holm-adjusted within the class.")
    for row in class_aggregates(analysis["comparisons"]):
        verdicts = ", ".join(f"{name} {count}" for name, count in row["verdicts"].items())
        print(f"  vs {row['other']:<20} {row['model']:<18} threads={row['threads']}: per ESS "
              f"{fmt(row['values'])} (n={len(row['values'])}); cells: {verdicts}")

    print_step_split(record)
    return 0


def step_split(record: dict) -> list[dict]:
    """Split rustmc's sampling time over nutpie's (matched warmup) into leapfrog steps
    taken and sampling seconds per step, with the ratio of smallest bulk ESS beside.
    Fits pair by dataset index. Per pair, steps ratio times per-step ratio is the
    sampling-time ratio; the medians are taken separately, so they need not multiply
    exactly. Uses every pair where both fits finished, passing or not, so the slow
    cells are explained too."""
    rows = []
    for row in record["cells"]:
        results = {engine: normalized(result) for engine, result in row["results"].items()}
        if "rustmc" not in results or "nutpie" not in results:
            continue
        pairs = paired_by_index(results["rustmc"]["fits"], results["nutpie"]["fits"])
        pairs = [(a, b) for a, b in pairs if a["status"] == "ok" and b["status"] == "ok"
                 and a.get("leapfrog_steps") and b.get("leapfrog_steps")]
        if not pairs:
            continue

        def per_step(fit):
            return fit["sampling_seconds"] / fit["leapfrog_steps"]

        rows.append({
            "cell": row["cell"],
            "pairs": len(pairs),
            "steps": float(np.median([a["leapfrog_steps"] / b["leapfrog_steps"]
                                      for a, b in pairs])),
            "per_step": float(np.median([per_step(a) / per_step(b) for a, b in pairs])),
            "min_ess": float(np.median([a["min_bulk_ess"] / b["min_bulk_ess"] for a, b in pairs])),
            "rustmc_us": 1e6 * float(np.median([per_step(a) for a, _ in pairs])),
            "nutpie_us": 1e6 * float(np.median([per_step(b) for _, b in pairs])),
        })
    return rows


def print_step_split(record: dict) -> None:
    rows = step_split(record)
    if not rows:
        return
    print()
    print("Where the sampling time goes, rustmc over nutpie at matched warmup, every finished "
          "pair: median ratios of leapfrog steps, of sampling seconds per step, and of smallest "
          "bulk ESS, with each engine's median microseconds per step (amortized: at 4 threads "
          "it divides concurrent chains' wall time by their total steps).")
    print(f"  {'cell':<36} {'pairs':>5} {'steps':>6} {'per step':>8} {'min ESS':>8} "
          f"{'rustmc us':>10} {'nutpie us':>10}")
    for r in rows:
        print(f"  {r['cell']:<36} {r['pairs']:>5} {r['steps']:>6.2f} {r['per_step']:>8.2f} "
              f"{r['min_ess']:>8.2f} {r['rustmc_us']:>10.2f} {r['nutpie_us']:>10.2f}")


if __name__ == "__main__":
    raise SystemExit(main())
