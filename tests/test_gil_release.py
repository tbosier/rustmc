"""Per-draw work on a fitted graph model runs without holding the GIL.

Log-likelihood, prediction, deterministics and loading a saved fit each cost
O(chains x draws x observations) in Rust. Holding the GIL through that
stalled every other Python thread in the process for the whole call.
"""
import json
import threading
import time

import numpy as np
import pytest
import rustmc

N_OBS = 6000
COPIES = 4


@pytest.fixture(scope="module")
def heavy_fit():
    rng = np.random.default_rng(3)
    x = rng.normal(size=N_OBS)
    y = 0.5 + 1.5 * x + rng.normal(scale=0.3, size=N_OBS)
    b = rustmc.ModelBuilder(data={"x": x, "y": y})
    a = b.normal_prior("a", 0.0, 1.0)
    beta = b.normal_prior("beta", 0.0, 1.0)
    s = b.half_normal_prior("s", 1.0)
    b.normal_likelihood("obs", a + beta * "x", s, "y")
    b.deterministic("mu", a + beta * "x")
    fit = rustmc.sample(b.build(), chains=1, draws=500, warmup=30, seed=1, max_tree_depth=3,
                        show_progress=False)
    # Repeating the one chain is a valid multi-chain artifact, and makes the
    # per-draw work long enough to observe without a long sampling run.
    artifact = json.loads(fit.to_json())
    posterior = artifact["posterior"]
    for key in ("samples", "unconstrained_samples", "accept_rates", "step_sizes",
                "divergences", "transitions"):
        posterior[key] = posterior[key] * COPIES
    text = json.dumps(artifact)
    return rustmc.FitResult.from_json(text), text


def other_thread_progress(call):
    """How often the main thread ran Python code while `call` ran on another
    thread, and how long the call took."""
    started, finished = threading.Event(), threading.Event()
    elapsed = []

    def worker():
        started.set()
        begin = time.perf_counter()
        call()
        elapsed.append(time.perf_counter() - begin)
        finished.set()

    thread = threading.Thread(target=worker)
    thread.start()
    started.wait()
    # Let the worker enter the call before counting.
    time.sleep(0.001)
    ticks = 0
    while not finished.is_set():
        ticks += 1
    thread.join()
    return ticks, elapsed[0]


@pytest.mark.parametrize(
    "operation",
    [
        lambda fit, text: fit.log_likelihood(),
        lambda fit, text: fit.posterior_predictive(seed=1),
        lambda fit, text: fit.predict(seed=1),
        lambda fit, text: fit.deterministics(),
        lambda fit, text: rustmc.FitResult.from_json(text),
    ],
    ids=["log_likelihood", "posterior_predictive", "predict", "deterministics", "from_json"],
)
def test_per_draw_work_releases_the_gil(heavy_fit, operation):
    fit, text = heavy_fit
    ticks, elapsed = other_thread_progress(lambda: operation(fit, text))
    if elapsed < 0.01:
        pytest.skip(f"call finished in {elapsed:.3f}s, too quickly to observe")
    # A held GIL leaves the main thread no chance to run until the call
    # returns; released, it spins freely for the whole call.
    assert ticks > 1000, f"main thread ran {ticks} times during a {elapsed:.3f}s call"
