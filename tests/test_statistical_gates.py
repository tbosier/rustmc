"""Release checks must reject plausible-looking but wrong posterior draws."""
import json
import math
import os
import sys
import numpy as np
import pytest

if os.environ.get("RUSTMC_REQUIRE_SITE_PACKAGES") == "1":
    pytest.skip("the validation harness is source-only", allow_module_level=True)

from benchmarks.validate_posteriors import assess_fit, json_safe, reference_cases


# ESS values a real estimator could return for the smallest fixture here (4 x 500
# draws, whose ceiling is ~6602), so the fixtures stay inside their own domain.
HEALTHY = {"r_hat": 1., "ess_bulk": 5000., "ess_tail": 5000.}
# Accurate draws for three independent standard normals, so only the diagnostics differ.
ACCURATE = np.random.default_rng(7).normal(size=(4, 5000, 3))


class ReferenceFit:
    def __init__(self, values):
        self.values = values
    def get_samples_2d(self):
        return {f"p{i}": self.values[..., i] for i in range(self.values.shape[-1])}
    def diagnostics(self):
        # One named row per parameter, as the native bindings emit.
        return [dict(HEALTHY, name=f"p{i}") for i in range(self.values.shape[-1])]
    def divergences(self):
        return [0, 0, 0, 0]


def test_joint_reference_gate_detects_lost_correlation_and_shift():
    covariance = np.array([[1., .8], [.8, 1.]])
    rng = np.random.default_rng(29)
    values = rng.multivariate_normal(np.zeros(2), covariance, size=(4, 5000))
    check = lambda draws: assess_fit(ReferenceFit(draws), ["p0", "p1"], np.zeros(2), covariance)
    assert check(values)["passed"]
    assert "max_mean_error_sd" in check(values + .5)["failed_metrics"]
    shuffled = values.copy()
    shuffled[:, :, 1] = rng.permutation(shuffled[:, :, 1].ravel()).reshape(4, 5000)
    assert "max_covariance_error_sd" in check(shuffled)["failed_metrics"]


def test_nonfinite_metrics_are_failed_and_preserved_as_null():
    fit = ReferenceFit(np.ones((4, 500, 1)))
    fit.diagnostics = lambda: [{"name": "p0", "r_hat": float("nan"), "ess_bulk": 10., "ess_tail": 5.}]
    result = assess_fit(fit, ["p0"], np.array([1.]), np.eye(1))
    assert not result["passed"]
    assert "max_rhat" in result["failed_metrics"]
    assert json_safe(result)["metrics"]["max_rhat"] is None


@pytest.mark.parametrize("key, metric", [("r_hat", "max_rhat"), ("ess_bulk", "min_ess_bulk"),
                                         ("ess_tail", "min_ess_tail")])
@pytest.mark.parametrize("position", [0, 1, 2])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_diagnostic_fails_the_gate_in_any_parameter_position(key, metric, position, bad):
    """A single bad parameter must fail the gate wherever it sits in the list.

    Builtin max/min drop a NaN that is not the first element, so the middle and last
    positions are the ones an aggregate-then-check gate silently accepts.
    """
    names = ["p0", "p1", "p2"]
    diagnostics = [dict(HEALTHY, name=name) for name in names]
    diagnostics[position][key] = bad
    fit = ReferenceFit(ACCURATE)
    fit.diagnostics = lambda: diagnostics
    result = assess_fit(fit, names, np.zeros(3), np.eye(3))
    assert result["passed"] is False
    assert f"{metric}[{names[position]}]" in result["failed_metrics"], result["failed_metrics"]
    assert metric in result["failed_metrics"]
    assert json_safe(result)["metrics"][metric] is None


def test_healthy_multiparameter_diagnostics_do_not_trip_the_finiteness_check():
    names = ["p0", "p1", "p2"]
    fit = ReferenceFit(ACCURATE)
    fit.diagnostics = lambda: [dict(HEALTHY, name=name) for name in names]
    result = assess_fit(fit, names, np.zeros(3), np.eye(3))
    assert result["passed"] is True, result["failed_metrics"]
    assert result["failed_metrics"] == []


def test_unnamed_and_missing_diagnostics_still_fail_closed():
    fit = ReferenceFit(np.zeros((4, 500, 2)))
    fit.diagnostics = lambda: [dict(HEALTHY), {"r_hat": 1., "ess_tail": 5000.}]
    result = assess_fit(fit, ["p0", "p1"], np.zeros(2), np.eye(2))
    assert result["passed"] is False
    assert "min_ess_bulk[1]" in result["failed_metrics"], result["failed_metrics"]
    fit.diagnostics = lambda: []
    empty = assess_fit(fit, ["p0", "p1"], np.zeros(2), np.eye(2))
    assert empty["passed"] is False
    assert "diagnostics_empty" in empty["failed_metrics"]


def test_a_parameter_with_no_diagnostic_row_is_not_silently_unchecked():
    """An absent row is a worse hole than a NaN: nothing about it is ever screened."""
    names = ["p0", "p1", "p2"]
    fit = ReferenceFit(ACCURATE)
    for present in ([0], [0, 1], [0, 0, 0]):
        fit.diagnostics = lambda rows=present: [dict(HEALTHY, name=names[i]) for i in rows]
        result = assess_fit(fit, names, np.zeros(3), np.eye(3))
        assert result["passed"] is False, present
        for name in names:
            if name not in [names[i] for i in present]:
                assert f"diagnostics_missing[{name}]" in result["failed_metrics"]


def test_absent_divergence_telemetry_is_not_read_as_zero_divergences():
    fit = ReferenceFit(ACCURATE)
    fit.diagnostics = lambda: [dict(HEALTHY, name=f"p{i}") for i in range(3)]
    fit.divergences = lambda: []
    result = assess_fit(fit, ["p0", "p1", "p2"], np.zeros(3), np.eye(3))
    assert result["passed"] is False
    assert "divergences" in result["failed_metrics"]
    assert json_safe(result)["metrics"]["divergences"] is None


def test_a_complex_or_oversized_diagnostic_does_not_coerce_to_a_healthy_number():
    fit = ReferenceFit(ACCURATE)
    for bad in (np.complex128(complex(1., np.inf)), 10**1000):
        fit.diagnostics = lambda value=bad: [dict(HEALTHY, name="p0"), dict(HEALTHY, name="p1"),
                                             dict(HEALTHY, name="p2", r_hat=value)]
        result = assess_fit(fit, ["p0", "p1", "p2"], np.zeros(3), np.eye(3))
        assert result["passed"] is False, bad
        assert "max_rhat[p2]" in result["failed_metrics"]


def test_nonfinite_divergence_and_accuracy_metrics_fail_closed():
    fit = ReferenceFit(np.zeros((4, 500, 1)))
    fit.divergences = lambda: [0, float("nan"), 0, 0]
    result = assess_fit(fit, ["p0"], np.array([0.]), np.eye(1))
    assert result["passed"] is False
    assert "divergences" in result["failed_metrics"]
    infinite = ReferenceFit(np.full((4, 500, 1), np.inf))
    accuracy = assess_fit(infinite, ["p0"], np.array([0.]), np.eye(1))
    assert accuracy["passed"] is False
    assert "max_mean_error_sd" in accuracy["failed_metrics"]


@pytest.mark.parametrize("key, metric", [("ess_bulk", "min_ess_bulk"), ("ess_tail", "min_ess_tail")])
@pytest.mark.parametrize("position", [0, 1, 2])
@pytest.mark.parametrize("bad", [1e100, 1e6, 90000.])
def test_an_impossible_ess_cannot_hide_under_a_healthy_minimum(key, metric, position, bad):
    """min() keeps the smallest value, so only an upper bound can catch a huge ESS."""
    names = ["p0", "p1", "p2"]
    diagnostics = [dict(HEALTHY, name=name) for name in names]
    diagnostics[position][key] = bad
    fit = ReferenceFit(ACCURATE)
    fit.diagnostics = lambda: diagnostics
    result = assess_fit(fit, names, np.zeros(3), np.eye(3))
    assert result["passed"] is False
    assert f"{metric}[{names[position]}]" in result["failed_metrics"], result["failed_metrics"]
    assert result["metrics"][metric] == bad


def test_the_ess_ceiling_is_the_one_the_estimator_can_actually_reach():
    """Derived from ess_raw: total / tau with tau >= 1 / log10(total), total = 2*chains*(draws//2)."""
    import benchmarks.validate_posteriors as gate
    for chains, draws in ((4, 2000), (4, 500), (2, 1000)):
        total = 2 * chains * (draws // 2)
        assert gate.ess_ceiling(chains, draws) == pytest.approx(total * math.log10(total))
    assert gate.ess_ceiling(4, 2000) == pytest.approx(31224.719895935552)


def test_a_recorded_ess_value_is_an_order_of_magnitude_below_the_ceiling():
    """Pins one recorded number against the bound; it measures nothing itself.

    Renamed from test_real_ess_values_stay_well_inside_the_ceiling, which read as a
    claim about the ESS values the gate produces. Both sides of the comparison are
    constants: the left is a single figure copied from a past run, not a fit performed
    here, so this only shows that the ceiling is not set so low as to reject it.
    """
    import benchmarks.validate_posteriors as gate
    # Observed for the correlated_regression case at chains=4, draws=2000.
    assert 2928.224634872191 < gate.ess_ceiling(4, 2000) / 10


@pytest.mark.parametrize("counts, reason", [
    ([1, -1, 0, 0], "cancellation"),
    ([0], "three missing chains"),
    ([0, 0, 0], "one missing chain"),
    ([0, 0, 0, 0, 0], "an extra chain"),
    ([0, 0, 0, -3], "a negative count"),
    ([0, 0, 0, 1.5], "a fractional count"),
])
def test_divergence_telemetry_must_be_one_whole_nonnegative_count_per_chain(counts, reason):
    """sum() cancels and cannot see a gap, so the counts are checked before summing."""
    fit = ReferenceFit(ACCURATE)
    fit.divergences = lambda: counts
    result = assess_fit(fit, ["p0", "p1", "p2"], np.zeros(3), np.eye(3))
    assert result["passed"] is False, reason
    assert "divergences" in result["failed_metrics"], reason
    assert json_safe(result)["metrics"]["divergences"] is None


def test_a_clean_four_chain_divergence_count_still_passes():
    fit = ReferenceFit(ACCURATE)
    fit.divergences = lambda: [0, 0, 0, 0]
    result = assess_fit(fit, ["p0", "p1", "p2"], np.zeros(3), np.eye(3))
    assert result["passed"] is True, result["failed_metrics"]
    assert result["metrics"]["divergences"] == 0
    fit.divergences = lambda: [0, 2, 0, 1]
    flagged = assess_fit(fit, ["p0", "p1", "p2"], np.zeros(3), np.eye(3))
    assert flagged["passed"] is False
    assert flagged["metrics"]["divergences"] == 3


@pytest.mark.parametrize("dtype", ["complex64", "complex128", "clongdouble"])
def test_every_complex_width_is_rejected_not_silently_truncated(dtype):
    """float() drops the imaginary part; np.complex64 is not a builtin complex."""
    bad = getattr(np, dtype)(complex(1., np.inf))
    fit = ReferenceFit(ACCURATE)
    fit.diagnostics = lambda: [dict(HEALTHY, name="p0"), dict(HEALTHY, name="p1"),
                               dict(HEALTHY, name="p2", r_hat=bad)]
    result = assess_fit(fit, ["p0", "p1", "p2"], np.zeros(3), np.eye(3))
    assert result["passed"] is False, dtype
    assert "max_rhat[p2]" in result["failed_metrics"], dtype


@pytest.mark.parametrize("position", [0, 1, 2])
@pytest.mark.parametrize("bad", [-100., -1., 0., 0.5, 0.9])
def test_an_out_of_domain_rhat_cannot_hide_under_a_healthy_maximum(position, bad):
    """max() keeps the largest value, so only a lower bound can catch a small R-hat."""
    names = ["p0", "p1", "p2"]
    diagnostics = [dict(HEALTHY, name=name) for name in names]
    diagnostics[position]["r_hat"] = bad
    fit = ReferenceFit(ACCURATE)
    fit.diagnostics = lambda: diagnostics
    result = assess_fit(fit, names, np.zeros(3), np.eye(3))
    assert result["passed"] is False
    assert f"max_rhat[{names[position]}]" in result["failed_metrics"], result["failed_metrics"]
    assert "max_rhat" in result["failed_metrics"]
    assert result["metrics"]["max_rhat"] == bad


@pytest.mark.parametrize("key, metric", [("ess_bulk", "min_ess_bulk"), ("ess_tail", "min_ess_tail")])
@pytest.mark.parametrize("position", [0, 1, 2])
def test_a_negative_ess_names_the_parameter_it_came_from(key, metric, position):
    names = ["p0", "p1", "p2"]
    diagnostics = [dict(HEALTHY, name=name) for name in names]
    diagnostics[position][key] = -5.
    fit = ReferenceFit(ACCURATE)
    fit.diagnostics = lambda: diagnostics
    result = assess_fit(fit, names, np.zeros(3), np.eye(3))
    assert result["passed"] is False
    assert f"{metric}[{names[position]}]" in result["failed_metrics"], result["failed_metrics"]


def test_the_rhat_floor_is_the_one_the_estimator_can_actually_reach():
    """Derived from basic_r_hat: sqrt(var_hat/W) >= sqrt((n-1)/n) with n = draws // 2."""
    import benchmarks.validate_posteriors as gate
    for draws, split in ((500, 250), (1000, 500), (2000, 1000), (5000, 2500)):
        assert gate.rhat_floor(draws) == pytest.approx(math.sqrt((split - 1) / split))
    assert gate.rhat_floor(1000) == pytest.approx(0.9989994995, abs=1e-9)
    assert gate.rhat_floor(2000) == pytest.approx(0.9994998749, abs=1e-9)
    # Real R-hat values dip just below 1. The lowest in this repo's recorded results is
    # 0.9991131788628178 from benchmarks/results/2026-09-09-calibration-pilot.json, whose
    # settings record draws=1000, so it must sit above the floor for that chain length.
    assert 0.9991131788628178 > gate.rhat_floor(1000) - gate.RHAT_FLOOR_SLACK


@pytest.mark.parametrize("r_hat", [0.9991131788628178, 0.9997664331946711, 0.9999823466170655, 1.0])
def test_legitimate_rhat_values_just_below_one_still_pass(r_hat):
    """Values observed from the real estimator must not be rejected as out of domain."""
    values = np.random.default_rng(11).normal(size=(4, 1000, 2))
    fit = ReferenceFit(values)
    fit.diagnostics = lambda: [dict(HEALTHY, name="p0", r_hat=r_hat),
                               dict(HEALTHY, name="p1", r_hat=r_hat)]
    result = assess_fit(fit, ["p0", "p1"], np.zeros(2), np.eye(2))
    assert result["passed"] is True, result["failed_metrics"]
    assert result["metrics"]["max_rhat"] == r_hat


def test_the_gate_fails_when_a_promised_reference_case_did_not_run():
    """all() over no fixed-reference records is vacuously true; the manifest is not."""
    import benchmarks.validate_posteriors as gate
    original = gate.reference_cases
    try:
        gate.reference_cases = lambda: iter(())
        report = gate.run(replicates=2, draws=2000, warmup=1000)
        assert report["passed"] is False
        assert [r for r in report["records"] if r["kind"] == "fixed_reference"] == []
        assert report["case_coverage"]["passed"] is False
        assert report["case_coverage"]["missing"] == sorted(gate.REFERENCE_CASES)

        first = next(iter(original()))
        gate.reference_cases = lambda: iter([first])
        partial = gate.run(replicates=2, draws=2000, warmup=1000)
        assert partial["passed"] is False
        assert first[0] not in partial["case_coverage"]["missing"]
        assert partial["case_coverage"]["attempted"][first[0]] == gate.REFERENCE_REPEATS
        assert len(partial["case_coverage"]["missing"]) == len(gate.REFERENCE_CASES) - 1
    finally:
        gate.reference_cases = original


def test_an_unlisted_reference_case_is_itself_a_failure():
    """The manifest is the source of truth, so an unvetted case cannot be smuggled in."""
    import benchmarks.validate_posteriors as gate
    original = gate.reference_cases
    try:
        cases = list(original())
        gate.reference_cases = lambda: iter(cases + [("smuggled_in",) + tuple(cases[0][1:])])
        report = gate.run(replicates=2, draws=2000, warmup=1000)
        assert report["passed"] is False
        assert report["case_coverage"]["unlisted"] == ["smuggled_in"]
        assert report["case_coverage"]["missing"] == []
    finally:
        gate.reference_cases = original


def test_a_case_construction_failure_still_produces_a_written_report(tmp_path):
    """The report must survive a failure while building a case, not be lost with it."""
    import benchmarks.validate_posteriors as gate
    original = gate.reference_cases

    def one_then_boom():
        yield next(iter(original()))
        raise RuntimeError("model construction blew up")

    try:
        gate.reference_cases = one_then_boom
        report = gate.run(replicates=2, draws=2000, warmup=1000)
    finally:
        gate.reference_cases = original
    assert report["passed"] is False
    broken = [r for r in report["records"] if r["case"] == "reference_case_construction"]
    assert len(broken) == 1
    assert broken[0]["passed"] is False
    assert "RuntimeError: model construction blew up" in broken[0]["error"]
    # The three fits that did complete are still in the report, not thrown away.
    assert len([r for r in report["records"] if r["case"] == gate.REFERENCE_CASES[0]]) == 3
    assert report["case_coverage"]["missing"] == sorted(gate.REFERENCE_CASES[1:])
    out = tmp_path / "report.json"
    out.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n")
    assert json.loads(out.read_text())["passed"] is False


def test_a_malformed_reference_case_is_reported_rather_than_raised():
    import benchmarks.validate_posteriors as gate
    original = gate.reference_cases
    try:
        gate.reference_cases = lambda: iter([("too", "few", "fields")])
        report = gate.run(replicates=2, draws=2000, warmup=1000)
    finally:
        gate.reference_cases = original
    assert report["passed"] is False
    broken = [r for r in report["records"] if r["case"] == "reference_case_construction"]
    assert len(broken) == 1 and broken[0]["passed"] is False


def test_a_calibration_model_failure_is_reported_rather_than_raised(monkeypatch):
    """The calibration model compiled outside every try, the same shape of hole."""
    import benchmarks.validate_posteriors as gate
    import rustmc as mc
    original = gate.reference_cases
    state = {"seen": 0}

    class Design:  # supports `beta @ "X"` so construction reaches compile()
        def __matmul__(self, other):
            return self

    class Exploding:  # the native ModelBuilder is not subclassable
        def vector_normal_prior(self, *args, **kwargs):
            return Design()
        def normal_likelihood(self, *args, **kwargs):
            return None
        def compile(self):
            state["seen"] += 1
            raise RuntimeError("calibration compile blew up")

    try:
        gate.reference_cases = lambda: iter(())
        monkeypatch.setattr(mc, "ModelBuilder", Exploding)
        report = gate.run(replicates=2, draws=2000, warmup=1000)
    finally:
        gate.reference_cases = original
    assert state["seen"] == 1
    assert report["passed"] is False
    assert report["calibration"]["passed"] is False
    failed = [r for r in report["records"] if r["kind"] == "calibration"]
    assert len(failed) == 1
    assert "RuntimeError: calibration compile blew up" in failed[0]["error"]


def test_the_manifest_matches_the_cases_actually_defined():
    import benchmarks.validate_posteriors as gate
    assert [case[0] for case in gate.reference_cases()] == list(gate.REFERENCE_CASES)


# One record's worth of the exact value types the gate produces today: Python floats
# and ints from the bindings, bools, strings, nested lists from .tolist(), and a NaN
# metric that must survive as null. Pinned byte-for-byte so the NumPy hardening in
# json_safe is provably inert on the payload that actually occurs.
TODAYS_PAYLOAD = {
    "format": "rustmc.posterior-validation", "version": 1, "dirty": False,
    "records": [{"case": "correlated_regression", "seed": 20260911, "kind": "fixed_reference",
                 "passed": False, "failed_metrics": ["max_rhat", "max_rhat[beta[1]]"],
                 "metrics": {"max_rhat": float("nan"), "min_ess_bulk": 2928.224634872191,
                             "min_ess_tail": 2670.823398331719, "divergences": 0,
                             "max_mean_error_sd": 0.01366020525234996,
                             "max_covariance_error_sd": 0.009526813215620373},
                 "reference_mean": [0.1, -0.2], "reference_covariance": [[1.0, 0.0], [0.0, 1.0]],
                 "posterior_mean": [0.10001, -0.19998],
                 "diagnostics": [{"name": "beta[0]", "r_hat": 1.0004558411941582,
                                  "ess_bulk": 2928.224634872191, "mcse_mean": 0.001},
                                 {"name": "beta[1]", "r_hat": float("nan"),
                                  "ess_bulk": float("inf"), "mcse_mean": float("-inf")}]}],
    "calibration": {"nominal": 0.9, "coverage": [1.0, 1.0], "replicates": 2,
                    "passed": True, "rank_histograms": [[0, 1], [1, 0]]},
    "seconds": 0.4123,
}
TODAYS_JSON = """{
  "format": "rustmc.posterior-validation",
  "version": 1,
  "dirty": false,
  "records": [
    {
      "case": "correlated_regression",
      "seed": 20260911,
      "kind": "fixed_reference",
      "passed": false,
      "failed_metrics": [
        "max_rhat",
        "max_rhat[beta[1]]"
      ],
      "metrics": {
        "max_rhat": null,
        "min_ess_bulk": 2928.224634872191,
        "min_ess_tail": 2670.823398331719,
        "divergences": 0,
        "max_mean_error_sd": 0.01366020525234996,
        "max_covariance_error_sd": 0.009526813215620373
      },
      "reference_mean": [
        0.1,
        -0.2
      ],
      "reference_covariance": [
        [
          1.0,
          0.0
        ],
        [
          0.0,
          1.0
        ]
      ],
      "posterior_mean": [
        0.10001,
        -0.19998
      ],
      "diagnostics": [
        {
          "name": "beta[0]",
          "r_hat": 1.0004558411941582,
          "ess_bulk": 2928.224634872191,
          "mcse_mean": 0.001
        },
        {
          "name": "beta[1]",
          "r_hat": null,
          "ess_bulk": null,
          "mcse_mean": null
        }
      ]
    }
  ],
  "calibration": {
    "nominal": 0.9,
    "coverage": [
      1.0,
      1.0
    ],
    "replicates": 2,
    "passed": true,
    "rank_histograms": [
      [
        0,
        1
      ],
      [
        1,
        0
      ]
    ]
  },
  "seconds": 0.4123
}"""


def test_json_safe_output_is_unchanged_on_the_payload_that_occurs_today():
    """Pin the serialization so the NumPy hardening is provably inert."""
    assert json.dumps(json_safe(TODAYS_PAYLOAD), indent=2, allow_nan=False) == TODAYS_JSON
    # Types, not just rendering: a bool must not become 1, an int must not become 1.0.
    safe = json_safe(TODAYS_PAYLOAD)
    assert safe["dirty"] is False
    assert isinstance(safe["version"], int) and not isinstance(safe["version"], bool)
    assert isinstance(safe["records"][0]["metrics"]["divergences"], int)
    assert safe["calibration"]["passed"] is True


def test_json_safe_keeps_a_numpy_scalar_from_breaking_the_strict_dump():
    """A non-float numeric in the retained raw diagnostics used to abort the write."""
    payload = {"metrics": {"max_rhat": np.float32("nan")},
               "diagnostics": [{"name": "p0", "r_hat": np.float32("nan"),
                                "ess_bulk": np.float32(10.), "ess_tail": np.int64(7),
                                "hdi": np.array([1.5, np.inf]), "converged": np.bool_(False)}]}
    safe = json_safe(payload)
    assert safe["metrics"]["max_rhat"] is None
    assert safe["diagnostics"][0]["r_hat"] is None
    assert safe["diagnostics"][0]["ess_bulk"] == 10.
    assert safe["diagnostics"][0]["ess_tail"] == 7
    assert safe["diagnostics"][0]["hdi"] == [1.5, None]
    assert safe["diagnostics"][0]["converged"] is False
    json.dumps(safe, allow_nan=False)  # must not raise


def test_json_safe_leaves_an_oversized_integer_alone():
    """math.isfinite raises OverflowError on a huge int; integers are finite anyway."""
    assert json_safe({"n": 10**1000})["n"] == 10**1000


def test_json_safe_terminates_on_a_scalar_that_does_not_unwrap():
    """np.longdouble.item() returns another np.longdouble, so naive recursion hangs."""
    for value in (np.longdouble("nan"), np.longdouble(1.5), np.clongdouble(complex(1., np.inf))):
        safe = json_safe({"x": value})          # must not raise RecursionError
        json.dumps(safe, allow_nan=False)
    assert json_safe({"x": np.longdouble("nan")})["x"] is None
    assert json_safe({"x": np.longdouble(1.5)})["x"] == 1.5
    assert json_safe({"x": np.clongdouble(complex(1., np.inf))})["x"] is None


@pytest.mark.parametrize("dtype", ["complex64", "complex128", "clongdouble"])
def test_a_rejected_complex_diagnostic_does_not_abort_the_dump(dtype):
    """The gate already failed it; serialization must not then throw the report away."""
    payload = {"diagnostics": [{"name": "p0", "r_hat": getattr(np, dtype)(complex(1., np.inf))}]}
    safe = json_safe(payload)
    assert safe["diagnostics"][0]["r_hat"] is None
    json.dumps(safe, allow_nan=False)


def test_main_writes_the_report_even_when_a_case_cannot_be_built(tmp_path, monkeypatch):
    """The CLI guarantee itself, not a stand-in: main() must leave a file behind."""
    import benchmarks.validate_posteriors as gate
    original = gate.reference_cases

    def one_then_boom():
        yield next(iter(original()))
        raise RuntimeError("model construction blew up")

    out = tmp_path / "report.json"
    monkeypatch.setattr(gate, "reference_cases", one_then_boom)
    monkeypatch.setattr(sys, "argv", ["validate_posteriors", "--replicates", "2",
                                      "--draws", "2000", "--warmup", "1000",
                                      "--output", str(out)])
    assert gate.main() == 1
    report = json.loads(out.read_text())
    assert report["passed"] is False
    assert any(r["case"] == "reference_case_construction" for r in report["records"])
    assert report["case_coverage"]["missing"] == sorted(gate.REFERENCE_CASES[1:])


def test_a_factory_that_raises_before_yielding_is_reported(monkeypatch):
    """reference_cases() itself was called outside the handler."""
    import benchmarks.validate_posteriors as gate

    def boom():
        raise RuntimeError("factory blew up")

    monkeypatch.setattr(gate, "reference_cases", boom)
    report = gate.run(replicates=2, draws=2000, warmup=1000)
    assert report["passed"] is False
    broken = [r for r in report["records"] if r["case"] == "reference_case_construction"]
    assert len(broken) == 1
    assert "RuntimeError: factory blew up" in broken[0]["error"]


def test_an_unhashable_case_name_is_reported_rather_than_raised(monkeypatch):
    """The name keys case_coverage, where an unhashable one raised outside every try."""
    import benchmarks.validate_posteriors as gate
    first = next(iter(gate.reference_cases()))
    monkeypatch.setattr(gate, "reference_cases",
                        lambda: iter([(["unhashable"],) + tuple(first[1:])]))
    report = gate.run(replicates=2, draws=2000, warmup=1000)
    assert report["passed"] is False
    broken = [r for r in report["records"] if r["case"] == "reference_case_construction"]
    assert len(broken) == 1 and "TypeError" in broken[0]["error"]


def test_a_failure_simulating_a_calibration_replicate_is_reported(monkeypatch):
    """np.linalg.solve ran outside the per-replicate handler."""
    import benchmarks.validate_posteriors as gate
    real_solve = np.linalg.solve
    state = {"calls": 0}

    # How many solves the reference phase itself consumes, measured rather than assumed.
    def counting(a, b):
        state["calls"] += 1
        return real_solve(a, b)

    monkeypatch.setattr(np.linalg, "solve", counting)
    list(gate.reference_cases())
    baseline = state["calls"]

    def flaky(a, b):
        state["calls"] += 1
        if state["calls"] > 2 * baseline:  # let the reference cases build, then break
            raise np.linalg.LinAlgError("singular design")
        return real_solve(a, b)

    state["calls"] = 0
    monkeypatch.setattr(np.linalg, "solve", flaky)
    report = gate.run(replicates=2, draws=2000, warmup=1000)
    assert report["passed"] is False
    failed = [r for r in report["records"]
              if r["kind"] == "calibration" and "LinAlgError" in r.get("error", "")]
    assert failed, [r.get("error") for r in report["records"] if r["kind"] == "calibration"]
    assert report["calibration"]["passed"] is False


def test_all_analytic_cases_have_a_finite_target_and_gradient():
    for _, model, data, names, mean, covariance in reference_cases():
        assert covariance.shape == (len(names), len(names))
        # The sampler evaluates unconstrained positions; zero is interior here.
        value, gradient = model.log_density(data, [0.] * len(names))
        assert np.isfinite(value)
        assert np.isfinite(gradient).all()
