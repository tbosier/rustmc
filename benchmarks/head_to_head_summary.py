"""Flags and paired ratios for a ``benchmarks/head_to_head.py`` record.

Lists every cell that ran: an engine that failed, a fit with R-hat above 1.01, any
post-warmup divergence or a smallest bulk ESS under 400, and a cell where rustmc's
and a nutpie run's means disagree by more than the 0.1% family-wise normal
quantile of their combined Monte Carlo standard errors. Then, per model class and
thread count, the median and range of every paired rustmc/nutpie ratio, and how
many cells' per-ESS ratios lie wholly below 1, wholly above it, or on both sides.

    python benchmarks/head_to_head_summary.py RECORD.json
"""
import json
import sys

import numpy as np
from statistics import NormalDist

# A fit is flagged when R-hat exceeds 1.01, it diverged, or its smallest bulk ESS
# is under 400. Engines disagree when the largest |difference of means| over the
# combined MCSE exceeds the 0.1% family-wise normal quantile for that many
# parameters.
MAX_RHAT, MIN_ESS = 1.01, 400.0


def agreement_threshold(parameters):
    return NormalDist().inv_cdf(1 - 0.001 / (2 * parameters))

record = json.load(open(sys.argv[1]))
rows = []
problems = []
print(f"revision {record['revision']}, versions {record['versions']}")
print(f"cells run: {len(record['cells'])}")
for cell in record["cells"]:
    for engine, result in cell["results"].items():
        if result.get("status") != "ok":
            problems.append(f"{cell['cell']}: {engine} {result.get('status')}: {result.get('reason')}")
            continue
        for fit in result["fits"]:
            flags = []
            if fit["max_rhat"] > MAX_RHAT:
                flags.append(f"R-hat {fit['max_rhat']:.3f}")
            if fit["divergences"]:
                flags.append(f"{fit['divergences']} divergences")
            if fit["min_bulk_ess"] < MIN_ESS:
                flags.append(f"min bulk ESS {fit['min_bulk_ess']:.0f}")
            if flags:
                problems.append(f"{cell['cell']}: {engine} dataset {fit['index']}: " + ", ".join(flags))
    params = None
    if cell["results"].get("rustmc", {}).get("status") == "ok":
        params = len(cell["results"]["rustmc"]["fits"][0]["means"])
    for other in ("nutpie", "nutpie_default_tune"):
        v = cell["summary"].get(f"vs_{other}")
        if v and params and v["max_agreement_z"] > agreement_threshold(params):
            problems.append(f"{cell['cell']}: rustmc and {other} disagree, max z {v['max_agreement_z']:.1f} "
                            f"over {params} parameters (threshold {agreement_threshold(params):.1f})")
    s = cell["summary"]
    c = cell["config"]
    for other in ("nutpie", "nutpie_default_tune"):
        v = s.get(f"vs_{other}")
        if not v:
            continue
        rows.append(dict(model=c["model"], size=f"{c['rows']}x{c['columns']}", threads=c["threads"],
                         other=other, fit=v["ratio_seconds"], sampling=v["ratio_sampling_seconds"],
                         per_ess=v["ratio_seconds_per_ess"], z=v["max_agreement_z"],
                         r_rhat=s["rustmc"]["max_rhat"], n_rhat=s[other]["max_rhat"],
                         r_div=s["rustmc"]["divergences"], n_div=s[other]["divergences"],
                         r_ess=min(s["rustmc"]["min_bulk_ess"]), n_ess=min(s[other]["min_bulk_ess"]),
                         r_compile=s["rustmc"]["compile_seconds"], n_compile=s[other]["compile_seconds"]))
    for engine in ("rustmc", "nutpie", "nutpie_default_tune"):
        if s.get(engine, {}).get("status") not in (None, "ok"):
            print("NOT OK", cell["cell"], engine, s[engine])


def verdict(values):
    lo, hi = min(values), max(values)
    if hi < 1:
        return "rustmc faster"
    if lo > 1:
        return "nutpie faster"
    return "overlap"


def fmt(values):
    return f"{np.median(values):.2f} [{min(values):.2f}, {max(values):.2f}]"


print(f"{'model':<18} {'size':<10} {'thr':>3} {'vs':<10} {'fit ratio':<20} {'sampling ratio':<20} {'per-ESS ratio':<20} verdict(per ESS)  z  rhat r/n  div r/n  minESS r/n")
for r in rows:
    vs = "nutpie" if r["other"] == "nutpie" else "nutpie300"
    print(f"{r['model']:<18} {r['size']:<10} {r['threads']:>3} {vs:<10} {fmt(r['fit']):<20} {fmt(r['sampling']):<20} "
          f"{fmt(r['per_ess']):<20} {verdict(r['per_ess']):<16} {r['z']:.1f} {r['r_rhat']:.3f}/{r['n_rhat']:.3f} "
          f"{r['r_div']}/{r['n_div']} {r['r_ess']:.0f}/{r['n_ess']:.0f}")

print()
print()
print(f"Problems ({len(problems)}):")
for line in problems:
    print("  " + line)
print()
print("By model class and comparison (median and range of every paired ratio in the class):")
for other in ("nutpie", "nutpie_default_tune"):
    for model in dict.fromkeys(r["model"] for r in rows):
        for threads in (1, 4):
            sel = [r for r in rows if r["model"] == model and r["other"] == other and r["threads"] == threads]
            if not sel:
                continue
            fit = [x for r in sel for x in r["fit"]]
            ess = [x for r in sel for x in r["per_ess"]]
            wins = sum(verdict(r["per_ess"]) == "rustmc faster" for r in sel)
            losses = sum(verdict(r["per_ess"]) == "nutpie faster" for r in sel)
            print(f"  vs {other:<20} {model:<18} threads={threads}: fit {fmt(fit)}, per ESS {fmt(ess)}; "
                  f"cells rustmc faster per ESS {wins}, nutpie faster {losses}, overlap {len(sel) - wins - losses} of {len(sel)}")
