"""Release checks must reject plausible-looking but wrong posterior draws."""
import os
import numpy as np
import pytest

if os.environ.get("RUSTMC_REQUIRE_SITE_PACKAGES") == "1":
    pytest.skip("the validation harness is source-only", allow_module_level=True)

from benchmarks.validate_posteriors import assess_fit, json_safe, reference_cases


class ReferenceFit:
    def __init__(self, values):
        self.values = values
    def get_samples_2d(self):
        return {f"p{i}": self.values[..., i] for i in range(self.values.shape[-1])}
    def diagnostics(self):
        return [{"r_hat": 1., "ess_bulk": 10000., "ess_tail": 10000.}]
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
    fit.diagnostics = lambda: [{"r_hat": float("nan"), "ess_bulk": 10., "ess_tail": 5.}]
    result = assess_fit(fit, ["p0"], np.array([1.]), np.eye(1))
    assert not result["passed"]
    assert "max_rhat" in result["failed_metrics"]
    assert json_safe(result)["metrics"]["max_rhat"] is None


def test_all_analytic_cases_have_a_finite_target_and_gradient():
    for _, model, data, names, mean, covariance in reference_cases():
        assert covariance.shape == (len(names), len(names))
        # The sampler evaluates unconstrained positions; zero is interior here.
        value, gradient = model.log_density(data, [0.] * len(names))
        assert np.isfinite(value)
        assert np.isfinite(gradient).all()
