import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

if os.environ.get("RUSTMC_REQUIRE_SITE_PACKAGES") == "1":
    pytest.skip(
        "the source-only benchmark harness is not part of the runtime wheel",
        allow_module_level=True,
    )

from benchmarks import protocol
from benchmarks import run as benchmark_run
from benchmarks.protocol import (
    BenchmarkConfig,
    ess_ceiling,
    evaluate_quality_gate,
    make_linear_regression,
    posterior_quality,
    quality_metric_domains,
    rhat_floor,
    timing_summary,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

#: The config the published comparison is run under, so the gate is exercised with the
#: same chain and draw counts that set its diagnostic domains.
STANDARD = BenchmarkConfig.from_json(REPO_ROOT / "benchmarks" / "configs" / "standard.json")
#: A quality payload a real fit of STANDARD could produce: every metric inside its own
#: domain and inside every published threshold. Each case below damages exactly one key.
HEALTHY_QUALITY = {
    "ess_bulk_mean": 3500.0,
    "ess_bulk_min": 3100.0,
    "rhat_rank_max": 1.0005,
    "divergences": 0,
    "mean_rmse_vs_exact_posterior": 1e-4,
    "mean_rmse_exact_posterior_sd_units": 0.01,
    "mean_rmse_vs_generating_beta": 0.02,
    "sd_relative_rmse_vs_exact_posterior": 0.01,
}
#: Every metric the gate screens; ess_bulk_mean carries no threshold but is the
#: numerator of the published ESS-per-second figure, so it is screened too.
GATED_METRICS = (
    "rhat_rank_max",
    "ess_bulk_mean",
    "ess_bulk_min",
    "mean_rmse_exact_posterior_sd_units",
    "sd_relative_rmse_vs_exact_posterior",
)


def test_problem_is_deterministic_and_has_analytic_posterior():
    config = BenchmarkConfig(observations=40, parameters=3, chains=2, warmup=5, draws=5)
    first = make_linear_regression(config)
    second = make_linear_regression(config)
    assert first.digest == second.digest
    np.testing.assert_array_equal(first.x, second.x)
    np.testing.assert_array_equal(first.y, second.y)
    assert first.x.dtype == np.float64
    assert first.x.flags.c_contiguous
    assert first.posterior_mean.shape == (3,)
    assert np.all(first.posterior_sd > 0)


def test_problem_digest_changes_with_data_seed():
    first = make_linear_regression(BenchmarkConfig(data_seed=1))
    second = make_linear_regression(BenchmarkConfig(data_seed=2))
    assert first.digest != second.digest


@pytest.mark.parametrize(
    ("field", "value"),
    [("observations", 0), ("chains", 0), ("target_accept", 1.0), ("prior_sigma", -1.0)],
)
def test_config_rejects_invalid_values(field, value):
    values = BenchmarkConfig().__dict__ | {field: value}
    with pytest.raises(ValueError):
        BenchmarkConfig(**values).validate()


def test_timing_summary_does_not_double_count_combined_phases():
    summary = timing_summary(
        {
            "import": 1.0,
            "build": 2.0,
            "compile": 3.0,
            "warmup_sample": 4.0,
            "postprocess": 5.0,
        }
    )
    assert summary == {
        "fit_seconds": 9.0,
        "cold_fit_seconds": 10.0,
        "total_seconds": 15.0,
        "sample_seconds": None,
    }


def test_common_quality_metrics_accept_chain_draw_parameter_layout():
    az = pytest.importorskip("arviz")
    config = BenchmarkConfig(observations=40, parameters=2, chains=4, warmup=5, draws=500)
    problem = make_linear_regression(config)
    rng = np.random.default_rng(7)
    samples = rng.normal(
        loc=problem.posterior_mean,
        scale=problem.posterior_sd,
        size=(config.chains, config.draws, config.parameters),
    )
    quality = posterior_quality(
        samples, problem, config, divergences=0, arviz_module=az
    )
    assert quality["divergences"] == 0
    assert quality["rhat_rank_max"] < 1.02
    assert quality["ess_bulk_min"] > 500
    assert quality["mean_rmse_vs_exact_posterior"] < 0.02
    gate = evaluate_quality_gate(quality, config)
    assert set(gate) == {"passed", "failures", "thresholds", "domains", "interpretation"}

    failed_quality = quality | {"rhat_rank_max": 1.2, "divergences": 1}
    failed_gate = evaluate_quality_gate(failed_quality, config)
    assert not failed_gate["passed"]
    assert {"rhat_rank_max", "divergences"} <= set(failed_gate["failures"])

    with pytest.raises(ValueError, match="chain, draw, parameter"):
        posterior_quality(
            samples[:, :-1], problem, config, divergences=0, arviz_module=az
        )


def test_a_healthy_quality_payload_passes_the_gate():
    """The baseline every damaged-payload case below is one edit away from."""
    gate = evaluate_quality_gate(HEALTHY_QUALITY, STANDARD)
    assert gate["passed"] is True, gate["failures"]
    assert gate["failures"] == []


@pytest.mark.parametrize("metric", GATED_METRICS)
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_nonfinite_quality_metric_cannot_pass_the_gate(metric, bad):
    """Bare ``>``/``<`` against NaN is false, so an unscreened gate reported success.

    This gate authorises publishing a speed comparison, so a fit whose diagnostics are
    not numbers has to fail it rather than sail through with an empty failure list.
    """
    gate = evaluate_quality_gate(HEALTHY_QUALITY | {metric: bad}, STANDARD)
    assert gate["passed"] is False, (metric, bad)
    assert metric in gate["failures"]
    assert f"{metric}[non-finite]" in gate["failures"], gate["failures"]


def test_every_diagnostic_nan_at_once_is_still_a_failure():
    damaged = HEALTHY_QUALITY | {metric: float("nan") for metric in GATED_METRICS}
    gate = evaluate_quality_gate(damaged, STANDARD)
    assert gate["passed"] is False
    assert set(GATED_METRICS) <= set(gate["failures"])


@pytest.mark.parametrize(
    "metric, bad",
    [
        # max() over the per-parameter R-hats keeps the largest, so only a lower bound
        # catches a reported maximum below what the estimator can return.
        ("rhat_rank_max", -100.0),
        ("rhat_rank_max", 0.0),
        ("rhat_rank_max", 0.5),
        ("rhat_rank_max", 0.9),
        # min() over the per-parameter ESS keeps the smallest, so only an upper bound
        # catches a reported minimum above what the estimator can return.
        ("ess_bulk_min", 1e100),
        ("ess_bulk_min", 1e6),
        ("ess_bulk_mean", 1e100),
        ("ess_bulk_min", -5.0),
        ("ess_bulk_mean", -5.0),
        # An RMSE takes a square root of a mean of squares and cannot be negative.
        ("mean_rmse_exact_posterior_sd_units", -1.0),
        ("sd_relative_rmse_vs_exact_posterior", -1.0),
    ],
)
def test_an_out_of_domain_metric_is_not_a_slightly_unlucky_one(metric, bad):
    """A value the estimator cannot produce is a broken payload, not a bad fit."""
    gate = evaluate_quality_gate(HEALTHY_QUALITY | {metric: bad}, STANDARD)
    assert gate["passed"] is False, (metric, bad)
    assert f"{metric}[out-of-domain]" in gate["failures"], gate["failures"]


@pytest.mark.parametrize("bad", [None, complex(1.0, 5.0), "not a number", [1.0], object()])
@pytest.mark.parametrize("metric", GATED_METRICS)
def test_a_non_numeric_metric_fails_the_gate_instead_of_raising(metric, bad):
    """float() raised TypeError here, aborting the run instead of failing the gate."""
    gate = evaluate_quality_gate(HEALTHY_QUALITY | {metric: bad}, STANDARD)
    assert gate["passed"] is False, (metric, bad)
    assert metric in gate["failures"]


@pytest.mark.parametrize("dtype", ["complex64", "complex128", "clongdouble"])
def test_every_complex_width_is_rejected_not_silently_truncated(dtype):
    """float() drops the imaginary part; np.complex64 is not a builtin complex."""
    bad = getattr(np, dtype)(complex(1.0, np.inf))
    gate = evaluate_quality_gate(HEALTHY_QUALITY | {"rhat_rank_max": bad}, STANDARD)
    assert gate["passed"] is False, dtype
    assert "rhat_rank_max" in gate["failures"]


@pytest.mark.parametrize("metric", GATED_METRICS)
def test_a_missing_metric_fails_the_gate_instead_of_raising(metric):
    """A KeyError aborted the whole benchmark row rather than failing its gate."""
    gate = evaluate_quality_gate(
        {key: value for key, value in HEALTHY_QUALITY.items() if key != metric}, STANDARD
    )
    assert gate["passed"] is False, metric
    assert f"{metric}[missing]" in gate["failures"], gate["failures"]


def test_an_absent_divergence_count_is_not_read_as_zero_divergences():
    quality = {key: value for key, value in HEALTHY_QUALITY.items() if key != "divergences"}
    gate = evaluate_quality_gate(quality, STANDARD)
    assert gate["passed"] is False
    assert "divergences[missing]" in gate["failures"]
    assert "divergences" in gate["failures"]


@pytest.mark.parametrize(
    "bad, reason",
    [
        (float("nan"), "non-finite"),
        (float("inf"), "non-finite"),
        (-3, "negative"),
        (-0.5, "negative"),
        (2.5, "fractional"),
        (None, "non-finite"),
        (complex(0.0, 1.0), "non-finite"),
    ],
)
def test_a_divergence_count_must_be_a_whole_nonnegative_number(bad, reason):
    """int() truncated a fraction, raised on NaN, and read a negative count as clean."""
    gate = evaluate_quality_gate(HEALTHY_QUALITY | {"divergences": bad}, STANDARD)
    assert gate["passed"] is False, bad
    assert f"divergences[{reason}]" in gate["failures"], gate["failures"]


@pytest.mark.parametrize(
    "metric, bad",
    [("rhat_rank_max", 1.2), ("ess_bulk_min", 10.0), ("divergences", 1),
     ("mean_rmse_exact_posterior_sd_units", 0.9),
     ("sd_relative_rmse_vs_exact_posterior", 0.9)],
)
def test_an_ordinary_threshold_breach_still_fails_by_its_bare_metric_name(metric, bad):
    """Screening must not displace the plain threshold failures the README documents."""
    gate = evaluate_quality_gate(HEALTHY_QUALITY | {metric: bad}, STANDARD)
    assert gate["passed"] is False
    assert gate["failures"] == [metric], gate["failures"]


def test_the_reported_domains_are_the_ones_the_gate_actually_applied():
    gate = evaluate_quality_gate(HEALTHY_QUALITY, STANDARD)
    assert gate["domains"] == {
        name: [None if math.isinf(bound) else bound for bound in bounds]
        for name, bounds in quality_metric_domains(STANDARD).items()
    }
    # Reported bounds must survive a strict JSON dump, which `Infinity` would not.
    json.dumps(gate, allow_nan=False)
    assert gate["domains"]["rhat_rank_max"][0] == pytest.approx(
        rhat_floor(STANDARD.draws) - protocol.RHAT_FLOOR_SLACK
    )
    assert gate["domains"]["ess_bulk_min"][1] == pytest.approx(
        ess_ceiling(STANDARD.chains, STANDARD.draws) * (1 + protocol.ESS_CEILING_SLACK)
    )


def test_both_release_gates_share_one_validation_rule():
    """The duplicated gate is what let this defect survive the last review cycle."""
    validate_posteriors = pytest.importorskip("benchmarks.validate_posteriors")
    for name in ("as_float", "metric_failure", "count_failure", "rhat_floor",
                 "ess_ceiling", "RHAT_FLOOR_SLACK", "ESS_CEILING_SLACK"):
        assert getattr(validate_posteriors, name) is getattr(protocol, name), name


def test_the_rhat_floor_is_the_one_arviz_can_actually_reach():
    """_rhat returns sqrt((B/W + n - 1)/n) with n the split length, and B >= 0.

    arviz.stats.diagnostics._rhat_rank splits each chain in half before calling _rhat,
    so n is ``draws // 2``, and rank-normalizing cannot make B negative. The same bound
    holds for rust_core/src/diagnostics.rs, which is why one helper serves both gates.
    """
    for draws, split in ((500, 250), (1000, 500), (2000, 1000)):
        assert rhat_floor(draws) == pytest.approx(math.sqrt((split - 1) / split))
    assert rhat_floor(1000) == pytest.approx(0.9989994995, abs=1e-9)


def test_the_ess_ceiling_is_the_one_arviz_can_actually_reach():
    """_ess returns total / tau_hat with tau_hat >= 1 / log10(total), total split."""
    for chains, draws in ((4, 1000), (4, 2000), (2, 100)):
        total = 2 * chains * (draws // 2)
        assert ess_ceiling(chains, draws) == pytest.approx(total * math.log10(total))
    assert ess_ceiling(4, 1000) == pytest.approx(14408.2399653, abs=1e-6)


def test_real_arviz_diagnostics_sit_inside_the_screened_domains():
    """The bounds must reject fabrications without rejecting a genuine healthy fit."""
    az = pytest.importorskip("arviz")
    config = BenchmarkConfig(observations=40, parameters=2, chains=4, warmup=5, draws=500)
    problem = make_linear_regression(config)
    rng = np.random.default_rng(7)
    samples = rng.normal(
        loc=problem.posterior_mean,
        scale=problem.posterior_sd,
        size=(config.chains, config.draws, config.parameters),
    )
    quality = posterior_quality(samples, problem, config, divergences=0, arviz_module=az)
    domains = quality_metric_domains(config)
    for metric, (low, high) in domains.items():
        assert low <= float(quality[metric]) <= high, (metric, quality[metric])
    assert evaluate_quality_gate(quality, config)["passed"] is True


def test_missing_optional_engine_is_reported_without_aborting(monkeypatch):
    def missing_adapter(config, problem):
        raise ModuleNotFoundError("No module named 'optional_engine'", name="optional_engine")

    monkeypatch.setitem(benchmark_run.ADAPTERS, "numpyro", missing_adapter)
    config = BenchmarkConfig(observations=20, parameters=2, chains=2, warmup=5, draws=5)
    result = benchmark_run.run_child("numpyro", config)
    assert result["status"] == "unavailable"
    assert result["engine"] == "numpyro"
    assert "optional_engine" in result["reason"]
    assert len(result["data_sha256"]) == 64


def test_child_environment_scopes_backend_flags(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "XLA_FLAGS", "--existing --xla_force_host_platform_device_count=99"
    )
    monkeypatch.setenv("PYTENSOR_FLAGS", "optimizer=fast,base_compiledir=/stale")
    config = BenchmarkConfig(threads=3)

    rustmc_env = benchmark_run._child_environment("rustmc", config, str(tmp_path))
    assert rustmc_env["XLA_FLAGS"] == "--existing"
    assert "PYTENSOR_FLAGS" not in rustmc_env
    assert "NUMBA_CACHE_DIR" not in rustmc_env

    pymc_env = benchmark_run._child_environment("pymc", config, str(tmp_path))
    assert "xla_force_host_platform_device_count" not in pymc_env["XLA_FLAGS"]
    assert f"base_compiledir={tmp_path}" in pymc_env["PYTENSOR_FLAGS"]
    assert "NUMBA_CACHE_DIR" not in pymc_env

    nutpie_env = benchmark_run._child_environment("nutpie", config, str(tmp_path))
    assert nutpie_env["NUMBA_CACHE_DIR"] == str(tmp_path)

    numpyro_env = benchmark_run._child_environment("numpyro", config, str(tmp_path))
    assert "--xla_force_host_platform_device_count=3" in numpyro_env["XLA_FLAGS"]
    assert "PYTENSOR_FLAGS" not in numpyro_env
    assert "NUMBA_CACHE_DIR" not in numpyro_env


def test_quick_config_dry_run_is_machine_readable():
    command = [
        sys.executable,
        str(REPO_ROOT / "benchmarks" / "run.py"),
        "--config",
        str(REPO_ROOT / "benchmarks" / "configs" / "quick.json"),
        "--dry-run",
    ]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    payload = json.loads(completed.stdout)
    assert payload["config"]["name"] == "linear-regression-quick"
    assert len(payload["data_sha256"]) == 64
