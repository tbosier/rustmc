import json
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

from benchmarks import run as benchmark_run
from benchmarks.protocol import (
    BenchmarkConfig,
    evaluate_quality_gate,
    make_linear_regression,
    posterior_quality,
    timing_summary,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


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
    assert set(gate) == {"passed", "failures", "thresholds", "interpretation"}

    failed_quality = quality | {"rhat_rank_max": 1.2, "divergences": 1}
    failed_gate = evaluate_quality_gate(failed_quality, config)
    assert not failed_gate["passed"]
    assert {"rhat_rank_max", "divergences"} <= set(failed_gate["failures"])

    with pytest.raises(ValueError, match="chain, draw, parameter"):
        posterior_quality(
            samples[:, :-1], problem, config, divergences=0, arviz_module=az
        )


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
    assert rustmc_env["PYTENSOR_FLAGS"] == "optimizer=fast"

    pymc_env = benchmark_run._child_environment("pymc", config, str(tmp_path))
    assert "xla_force_host_platform_device_count" not in pymc_env["XLA_FLAGS"]
    assert f"base_compiledir={tmp_path}" in pymc_env["PYTENSOR_FLAGS"]

    numpyro_env = benchmark_run._child_environment("numpyro", config, str(tmp_path))
    assert "--xla_force_host_platform_device_count=3" in numpyro_env["XLA_FLAGS"]
    assert "base_compiledir" not in numpyro_env["PYTENSOR_FLAGS"]


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
