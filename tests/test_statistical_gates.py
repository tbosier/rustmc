"""Release checks must reject plausible-looking but wrong posterior draws."""
import os
import numpy as np
import pytest

if os.environ.get("RUSTMC_REQUIRE_SITE_PACKAGES") == "1":
    pytest.skip("the validation harness is source-only", allow_module_level=True)

from benchmarks.validate_posteriors import assess_fit, json_safe, reference_cases


HEALTHY = {"r_hat": 1., "ess_bulk": 10000., "ess_tail": 10000.}
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
    fit.diagnostics = lambda: [dict(HEALTHY), {"r_hat": 1., "ess_tail": 10000.}]
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


def test_all_analytic_cases_have_a_finite_target_and_gradient():
    for _, model, data, names, mean, covariance in reference_cases():
        assert covariance.shape == (len(names), len(names))
        # The sampler evaluates unconstrained positions; zero is interior here.
        value, gradient = model.log_density(data, [0.] * len(names))
        assert np.isfinite(value)
        assert np.isfinite(gradient).all()
