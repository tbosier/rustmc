"""The head-to-head summary's verdicts, Holm families, aggregates and step split.

The oracles are hand-computed: exact binomial tails for the sign test, Holm's
step-down rule by hand, and paired medians worked out in the comments."""
import json
import math
import os

import pytest

if os.environ.get("RUSTMC_REQUIRE_SITE_PACKAGES") == "1":
    pytest.skip(
        "the source-only benchmark harness is not part of the runtime wheel",
        allow_module_level=True,
    )

from benchmarks import head_to_head, head_to_head_summary

SHA = "a" * 64
NAMES = ["alpha", "beta[0]", "beta[1]", "beta[2]", "beta[3]"]


def fit(index, seconds, steps=1000, ess=1000.0):
    """One finished fit that passes the quality gate, with means equal across engines."""
    return {
        "index": index, "status": "ok", "seconds": seconds, "sampling_seconds": seconds,
        "native_seconds": seconds, "leapfrog_steps": steps, "min_bulk_ess": ess,
        "mean_bulk_ess": ess, "min_tail_ess": ess, "max_rhat": 1.001, "divergences": 0,
        "means": dict.fromkeys(NAMES, 0.0), "mcse": dict.fromkeys(NAMES, 0.01),
        "load_wait_timed_out": False,
    }


def fits(seconds, steps=None, ess=None):
    n = len(seconds)
    steps = steps or [1000] * n
    ess = ess or [1000.0] * n
    return [fit(i, s, k, e) for i, (s, k, e) in enumerate(zip(seconds, steps, ess))]


def cell(rows, rustmc_seconds, nutpie_seconds, sha=SHA, threads=1, **kwargs):
    config = {"model": "normal", "rows": rows, "columns": 4, "threads": threads,
              "datasets": len(rustmc_seconds), "chains": 4, "warmup": 100, "draws": 100,
              "data_seed": 1}
    return {
        "cell": f"normal {rows}x4 threads={threads}",
        "config": config,
        "results": {
            "rustmc": {"status": "ok", "provenance": {"extension_sha256": sha},
                       "fits": fits(rustmc_seconds, kwargs.get("rustmc_steps"),
                                    kwargs.get("rustmc_ess"))},
            "nutpie": {"status": "ok", "provenance": {},
                       "fits": fits(nutpie_seconds, kwargs.get("nutpie_steps"),
                                    kwargs.get("nutpie_ess"))},
        },
    }


def record(*cells):
    return {"schema": 2, "revision": {"commit": "0" * 40, "dirty": False},
            "versions": {}, "extension_sha256": SHA, "cells": list(cells)}


def comparisons(rec):
    found = head_to_head_summary.analyze(rec, "seconds")["comparisons"]
    return {c["cell"]: c for c in found}


def sign_p(below, above):
    """Two-sided exact sign test, written out from the binomial distribution."""
    n = below + above
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(below, above) + 1)) / 2**n)


def test_sign_test_and_holm_match_hand_computed_values():
    assert head_to_head.sign_test([0.5] * 5) == 2 / 32
    assert head_to_head.sign_test([0.5] * 9 + [2.0]) == 22 / 1024
    # Ties are dropped: 10 below and 11 ties is 10 of 10.
    assert head_to_head.sign_test([0.5] * 10 + [1.0] * 11) == 2 / 1024
    # Holm on (0.01, 0.04, 0.03): 3 * 0.01, then max(0.03, 2 * 0.03), then
    # max(0.06, 1 * 0.04). Bonferroni would give 0.03, 0.12, 0.09.
    assert head_to_head.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert head_to_head.holm([None, 0.02]) == [None, 0.02]


def test_the_winner_is_the_engine_that_won_more_pairs():
    # Ten rustmc wins and eleven exact ties: the median ratio is 1.0, but nutpie
    # never won a pair.
    tied = head_to_head.compare({"fits": fits([0.5] * 10 + [1.0] * 11)},
                                {"fits": fits([1.0] * 21)}, len(NAMES))
    assert tied["sign_test_p"] == sign_p(10, 0)
    assert tied["verdict"] == "rustmc faster"
    # Alone in its family, Holm leaves the same p, and the adjusted verdict must
    # name the same winner.
    (alone,) = comparisons(record(cell(100, [0.5] * 10 + [1.0] * 11, [1.0] * 21))).values()
    assert alone["holm_p"] == sign_p(10, 0)
    assert alone["adjusted_verdict"] == "rustmc faster"
    # One rustmc win and nine losses is significant (p = 22/1024) for nutpie.
    lost = head_to_head.compare({"fits": fits([0.5] + [2.0] * 9)},
                                {"fits": fits([1.0] * 10)}, len(NAMES))
    assert lost["sign_test_p"] == sign_p(1, 9)
    assert lost["verdict"] == "nutpie faster"


def test_holm_families_span_sizes_and_thread_counts():
    # One family: two sizes at one thread and an inconclusive cell at four threads.
    # p for 10 of 10 is 2/1024 and for 9 of 10 is 22/1024. Holm over three members
    # (the inconclusive one at p = 1): 3 * 2/1024, then max(6/1024, 2 * 22/1024),
    # so the second cell is at 44/1024, not its unadjusted 22/1024, nor
    # Bonferroni's 66/1024.
    found = comparisons(record(
        cell(100, [0.5] * 10, [1.0] * 10),
        cell(1000, [0.5] * 9 + [2.0], [1.0] * 10),
        cell(100, [0.5] * 10, [1.0] * 10, sha="b" * 64, threads=4),
    ))
    first, second = found["normal 100x4 threads=1"], found["normal 1000x4 threads=1"]
    third = found["normal 100x4 threads=4"]
    assert first["holm_p"] == 3 * 2 / 1024
    assert second["holm_p"] == 2 * 22 / 1024
    assert first["family_size"] == second["family_size"] == 3
    assert third["verdict"] == third["adjusted_verdict"] == "inconclusive (provenance)"
    assert third["holm_p"] is None
    assert first["adjusted_verdict"] == second["adjusted_verdict"] == "rustmc faster"


@pytest.mark.parametrize("rejection", ["provenance", "failed engine"])
def test_a_rejected_cell_stays_in_its_holm_family(rejection):
    # 6 of 6 wins is p = 2/64; with a second member at p = 1, Holm gives 4/64,
    # which is no winner.
    other = cell(1000, [0.1] * 6, [1.0] * 6, sha="b" * 64 if rejection == "provenance" else SHA)
    if rejection == "failed engine":
        other["results"]["nutpie"].update(status="error", reason="compile failed", fits=[])
    found = comparisons(record(cell(100, [0.5] * 6, [1.0] * 6), other))
    first, second = found["normal 100x4 threads=1"], found["normal 1000x4 threads=1"]
    assert first["sign_test_p"] == 2 / 64
    assert first["holm_p"] == 4 / 64
    assert first["verdict"] == "rustmc faster"
    assert first["adjusted_verdict"] == "no clear winner"
    expected = ("inconclusive (provenance)" if rejection == "provenance"
                else "inconclusive (no finished pairs)")
    assert second["verdict"] == expected
    # The means agree in one case and are absent in the other: neither is a disagreement.
    assert second["mean_difference_flag"] is False


def test_class_aggregates_leave_out_every_rejected_cell():
    # Only the first cell's six 0.5 ratios belong in the class: the second has a
    # different extension and the third disagrees on a mean by 100 against an MCSE
    # of 0.01.
    disagreeing = cell(10000, [0.2] * 6, [1.0] * 6)
    for f in disagreeing["results"]["rustmc"]["fits"]:
        f["means"]["alpha"] = 100.0
    analysis = head_to_head_summary.analyze(record(
        cell(100, [0.5] * 6, [1.0] * 6),
        cell(1000, [0.1] * 6, [1.0] * 6, sha="b" * 64),
        disagreeing,
    ), "seconds")
    verdicts = {c["cell"]: c["verdict"] for c in analysis["comparisons"]}
    assert verdicts["normal 10000x4 threads=1"] == "inconclusive (means differ)"
    (row,) = head_to_head_summary.class_aggregates(analysis["comparisons"])
    assert row["values"] == pytest.approx([0.5] * 6)
    assert row["verdicts"] == {"no clear winner": 1, "inconclusive (provenance)": 1,
                               "inconclusive (means differ)": 1}


def test_printed_table_puts_each_value_under_its_header(tmp_path, capsys):
    # Raw p 2/64 and Holm p 4/64 differ, as do the two verdicts, so a swapped
    # column shows.
    path = tmp_path / "record.json"
    path.write_text(json.dumps(record(cell(100, [0.5] * 6, [1.0] * 6),
                                      cell(1000, [0.1] * 6, [1.0] * 6, sha="b" * 64))))
    assert head_to_head_summary.main([str(path)]) == 0
    lines = capsys.readouterr().out.splitlines()
    header = next(line for line in lines if line.startswith("cell ") and "Holm p" in line)
    row = next(line for line in lines if line.startswith("normal 100x4"))

    def under(name, width):
        start = header.index(name)
        return row[start:start + width].strip()

    assert row[header.index("sign p") - 1:header.index("sign p") + 6].strip() == "0.031"
    assert row[header.index("Holm p") - 1:header.index("Holm p") + 6].strip() == "0.062"
    assert under("unadjusted", 24) == "rustmc faster"
    assert under("Holm-adjusted", 24) == "no clear winner"


def test_step_split_takes_paired_medians_by_dataset():
    # Paired ratios, all sampling times 1 s:
    #   steps    2/1, 3/100, 100/2            -> median 2.0
    #   per step (1/2)/1, (1/3)/(1/100), (1/100)/(1/2) = 0.5, 33.3, 0.02 -> 0.5
    #   ESS      1000/500, 2000/1000, 500/1000 -> median 2.0
    # Medians of each engine's own values instead would give 3/2 and (1/3)/(1/2).
    # Microseconds per step: medians of 1e6/2, 1e6/3, 1e6/100 and of 1e6, 1e4, 1e6/2.
    base = cell(100, [1.0] * 3, [1.0] * 3, rustmc_steps=[2, 3, 100], nutpie_steps=[1, 100, 2],
                rustmc_ess=[1000.0, 2000.0, 500.0], nutpie_ess=[500.0, 1000.0, 1000.0])
    shuffled = json.loads(json.dumps(base))
    shuffled["results"]["nutpie"]["fits"].reverse()
    for rec in (record(base), record(shuffled)):
        (row,) = head_to_head_summary.step_split(rec)
        assert row["pairs"] == 3
        assert row["steps"] == 2.0
        assert row["per_step"] == pytest.approx(0.5)
        assert row["min_ess"] == 2.0
        assert row["rustmc_us"] == pytest.approx(1e6 / 3)
        assert row["nutpie_us"] == pytest.approx(1e6 / 2)


def test_a_repeated_dataset_index_is_refused():
    # Six copies of one dataset must not count as six pairs.
    rustmc, nutpie = fits([0.5] * 6), fits([1.0] * 6)
    for f in rustmc + nutpie:
        f["index"] = 0
    with pytest.raises(ValueError, match="repeated dataset index"):
        head_to_head.compare({"fits": rustmc}, {"fits": nutpie}, len(NAMES))
    duplicated = cell(100, [1.0] * 2, [1.0] * 2)
    duplicated["results"]["nutpie"]["fits"][1]["index"] = 0
    with pytest.raises(ValueError, match="repeated dataset index"):
        head_to_head_summary.step_split(record(duplicated))


@pytest.mark.parametrize("bad", [math.nan, math.inf, 0.0, -1.0])
def test_a_fit_without_a_finite_positive_time_cannot_enter_a_verdict(bad):
    broken = fit(0, bad)
    assert "time not finite and positive" in head_to_head.fit_problems(broken)
    # Six real wins and fifteen fits with a bad time: only the six pass.
    result = head_to_head.compare({"fits": fits([0.5] * 6 + [bad] * 15)},
                                  {"fits": fits([1.0] * 21)}, len(NAMES))
    assert result["passing_pairs"] == 6
    assert result["sign_test_p"] == 2 / 64
    assert result["ratio_seconds_per_ess"] == pytest.approx([0.5] * 6)
