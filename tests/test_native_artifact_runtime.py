"""The same artifact has identical evaluation, posterior, and prediction in Rust and Python."""
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
import rustmc


def test_native_runtime_matches_python(tmp_path):
    executable = os.environ.get("RUSTMC_NATIVE_ARTIFACT_EXAMPLE")
    if not executable:
        pytest.skip("build the native load_model example and set RUSTMC_NATIVE_ARTIFACT_EXAMPLE")
    m = rustmc.ModelBuilder()
    mu = m.normal_prior("mu", 0., 1.)
    scale = m.half_normal_prior("scale", 1.)
    group = m.normal_prior("group", mu, scale)
    beta = m.vector_normal_prior("beta", 2, 0., 1.)
    m.normal_likelihood("obs", group + beta @ "X", .5, "y")
    m.potential("regularizer", -.01 * mu**4)
    compiled = m.compile()
    data = {"X": [[1., -.5], [1., .5], [1., 1.]], "y": [.2, .8, 1.1]}
    future = {"X": [[1., -.25], [1., .75]]}
    q = [.2, -.3, .4, -.1, .3]
    paths = [tmp_path/f"{name}.json" for name in ("model","data","position","future")]
    for path, value in zip(paths, [json.loads(compiled.to_json()), data, q, future]):
        path.write_text(json.dumps(value))
    native = json.loads(subprocess.check_output([executable, *map(str, paths)], text=True))
    lp, gradient = compiled.log_density(data,q)
    np.testing.assert_allclose(native["log_density"],lp,rtol=1e-12)
    np.testing.assert_allclose(native["gradient"],gradient,rtol=1e-12,atol=1e-12)
    fit = compiled.sample(data,chains=2,warmup=500,draws=500,target_accept=.95,seed=42,show_progress=False)
    np.testing.assert_allclose(native["posterior_mean"],[fit.mean()[name] for name in native["param_names"]],rtol=1e-12)
    prediction = fit.predict(future,seed=43,expected=True)
    np.testing.assert_allclose(native["predictions"]["obs"],prediction["obs"],rtol=1e-12)
    restored = rustmc.CompiledModel.from_json(json.dumps(native["artifact"]))
    np.testing.assert_allclose(restored.log_density(data,q)[1],gradient,rtol=1e-12)
