"""Per-draw work on a fitted graph model runs without holding the GIL.

Log-likelihood, prediction, deterministics and loading a saved fit each cost
O(chains x draws x observations) in Rust. Holding the GIL through that
stalled every other Python thread in the process for the whole call.
"""
import copy
import json
import os
import threading
import time

import numpy as np
import pytest
import rustmc

N_OBS = 6000
COPIES = 4
# The workload doubles from COPIES until a call takes MIN_ELAPSED, up to
# MAX_COPIES (whose per-observation outputs are 16 x 500 x N_OBS doubles, about
# 380 MB). On CI a call that is still too fast to observe fails instead of
# skipping, so a fast runner cannot turn the whole test into skips.
MIN_ELAPSED = 0.01
MAX_COPIES = 16


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
    artifact = json.loads(fit.to_json())
    cache = {}

    def replicated(copies):
        # Repeating the one chain is a valid multi-chain artifact, and makes the
        # per-draw work long enough to observe without a long sampling run.
        if copies not in cache:
            scaled = copy.deepcopy(artifact)
            posterior = scaled["posterior"]
            for key in ("samples", "unconstrained_samples", "accept_rates", "step_sizes",
                        "divergences", "transitions"):
                posterior[key] = posterior[key] * copies
            text = json.dumps(scaled)
            cache.clear()
            cache[copies] = rustmc.FitResult.from_json(text), text
        return cache[copies]

    return replicated


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
    copies = COPIES
    while True:
        fit, text = heavy_fit(copies)
        ticks, elapsed = other_thread_progress(lambda: operation(fit, text))
        if elapsed >= MIN_ELAPSED or copies >= MAX_COPIES:
            break
        copies *= 2
    if elapsed < MIN_ELAPSED:
        message = f"call finished in {elapsed:.3f}s at {copies} copies, too quickly to observe"
        if os.environ.get("CI") or os.environ.get("RUSTMC_REQUIRE_SOURCE_ROOT"):
            pytest.fail(message)
        pytest.skip(message)
    # A held GIL leaves the main thread no chance to run until the call
    # returns; released, it spins freely for the whole call.
    assert ticks > 1000, f"main thread ran {ticks} times during a {elapsed:.3f}s call"
