"""Verdicts for a ``benchmarks/head_to_head.py`` record, recomputed from its fits.

Lists every cell that ran and every problem in it: an engine or fit that failed,
a fit that fails the quality checks (a non-finite diagnostic, R-hat above 1.01,
any post-warmup divergence, a smallest bulk ESS under 400), a load wait that timed
out, a mean-difference test that crossed its 0.1% family-wise threshold, and an
extension hash that differs between children. Then, per cell and comparison, the
per-ESS ratios (rustmc over nutpie) over the pairs where both fits pass, their
median and observed range, a two-sided sign test and a bootstrap interval of the
median, and the verdict: a winner only when the sign test is significant at 5%.
Every ratio of a finished pair is printed too, passing or not.

Ranges are the lowest and highest observed values, not intervals. A cell's two
thread settings time the same draws, so they are not replicates of each other.

    python benchmarks/head_to_head_summary.py RECORD.json [--endpoint sampling]

``--endpoint sampling`` uses the time to the end of sampling for the per-ESS
ratios instead of the time to an InferenceData. Records written before the
InferenceData groups were matched (2026-10-25) should be read that way.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.head_to_head import Cell, compare, fit_problems, scalar_names


def normalized(result: dict) -> dict:
    """Older records kept no per-fit status: their fits all finished."""
    fits = [dict(fit, status=fit.get("status", "ok")) for fit in result.get("fits", [])]
    return dict(result, fits=fits)


def fmt(values, digits=2):
    if not values:
        return "-"
    return f"{np.median(values):.{digits}f} [{min(values):.{digits}f}, {max(values):.{digits}f}]"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("record", type=Path)
    parser.add_argument("--endpoint", choices=("inferencedata", "sampling"), default="inferencedata")
    args = parser.parse_args(argv)
    time_key = "seconds" if args.endpoint == "inferencedata" else "sampling_seconds"
    record = json.loads(args.record.read_text())

    print(f"revision {record['revision']}, versions {record['versions']}")
    print(f"extension sha256 {record.get('extension_sha256', 'not recorded')}")
    print(f"compile cache at start: {record.get('compile_cache', 'not recorded')}")
    print(f"per-ESS ratios use the time to "
          f"{'an InferenceData' if time_key == 'seconds' else 'the end of sampling'}")
    print(f"cells run: {len(record['cells'])}")
    problems = []
    comparisons = []
    for row in record["cells"]:
        cell = Cell(**row["config"])
        results = {engine: normalized(result) for engine, result in row["results"].items()}
        load_ok = not row.get("load_wait_timed_out", False)
        if not load_ok:
            problems.append(f"{row['cell']}: load wait timed out at {row['load_at_start']:.2f}")
        for engine, result in results.items():
            if result.get("status") not in ("ok", "partial"):
                problems.append(f"{row['cell']}: {engine} {result.get('status')}: {result.get('reason')}")
            sha = result.get("provenance", {}).get("extension_sha256")
            if sha and record.get("extension_sha256") and sha != record["extension_sha256"]:
                problems.append(f"{row['cell']}: {engine} loaded a different extension, {sha}")
            if engine.startswith("nutpie") and "compile_cache_files_before" in result:
                before = result["compile_cache_files_before"]
                kind = "cold" if not before else "warm"
                result["compile_label"] = f"{kind} ({before or 0} cache files before)"
            for fit in result["fits"]:
                reasons = fit_problems(fit)
                if reasons:
                    problems.append(f"{row['cell']}: {engine} dataset {fit['index']}: " + ", ".join(reasons))
        for other in ("nutpie", "nutpie_default_tune"):
            if other not in results or not results["rustmc"]["fits"] or not results[other]["fits"]:
                continue
            c = compare(results["rustmc"], results[other], len(scalar_names(cell)), load_ok, time_key)
            if c["mean_difference_flag"]:
                problems.append(f"{row['cell']}: rustmc and {other} means differ, max z "
                                f"{c['max_mean_difference_z']:.1f} over {c['mean_difference_tests']} "
                                f"tests (threshold {c['mean_difference_threshold']:.1f})")
            comparisons.append(dict(c, cell=row["cell"], model=cell.model, threads=cell.threads,
                                    other=other, compile=(results["rustmc"].get("compile_seconds"),
                                                          results[other].get("compile_seconds"),
                                                          results[other].get("compile_label"))))

    print()
    print(f"{'cell':<36} {'vs':<9} {'pass':>5} {'per-ESS ratio, passing pairs':<28} {'sign p':>7} "
          f"{'bootstrap 95% of median':<24} {'verdict':<24} {'max z / threshold':<18} all finished pairs' per-ESS ratios")
    for c in comparisons:
        vs = "nutpie" if c["other"] == "nutpie" else "nutpie300"
        p = "-" if c["sign_test_p"] is None else f"{c['sign_test_p']:.3f}"
        boot = "-" if c["bootstrap_median_95"] is None else "[{:.2f}, {:.2f}]".format(*c["bootstrap_median_95"])
        ratios = " ".join(f"{x:.2f}" for x in c["all_ratio_seconds_per_ess"])
        print(f"{c['cell']:<36} {vs:<9} {c['passing_pairs']:>2}/{c['pairs']:<2} "
              f"{fmt(c['ratio_seconds_per_ess']):<28} {p:>7} {boot:<24} {c['verdict']:<24} "
              f"{c['max_mean_difference_z']:.1f} / {c['mean_difference_threshold']:.1f}       {ratios}")

    print()
    print(f"Problems ({len(problems)}):")
    for line in problems:
        print("  " + line)

    print()
    print("By model class: verdict counts over cells, and the median and observed range of the "
          "per-ESS ratios of every passing pair in the class (n values).")
    for other in ("nutpie", "nutpie_default_tune"):
        for model in dict.fromkeys(c["model"] for c in comparisons):
            for threads in (1, 4):
                chosen = [c for c in comparisons
                          if c["model"] == model and c["other"] == other and c["threads"] == threads]
                if not chosen:
                    continue
                values = [x for c in chosen for x in c["ratio_seconds_per_ess"]]
                counts = {}
                for c in chosen:
                    counts[c["verdict"]] = counts.get(c["verdict"], 0) + 1
                verdicts = ", ".join(f"{name} {count}" for name, count in counts.items())
                print(f"  vs {other:<20} {model:<18} threads={threads}: per ESS {fmt(values)} "
                      f"(n={len(values)}); cells: {verdicts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
