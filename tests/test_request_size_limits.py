"""Requests too large to hold raise instead of aborting the interpreter.

A failed Rust allocation aborts the process, so each of these calls used to
take Python down with "memory allocation ... failed".
"""

import numpy as np
import pytest
import rustmc as r

HUGE = 10**12
DATA = {"y": np.array([0.3, -0.2, 0.9])}


def model():
    b = r.ModelBuilder()
    mu = b.normal_prior("mu", 0.0, 1.0)
    b.normal_likelihood("obs", mu, 1.0, "y")
    return b


@pytest.mark.parametrize("option", ["draws", "warmup", "chains"])
def test_oversized_fits_raise(option):
    options = dict(draws=10, warmup=10, chains=1, show_progress=False)
    options[option] = HUGE
    with pytest.raises(ValueError, match="safety limit"):
        r.sample(model().build(), data=DATA, **options)
    with pytest.raises(ValueError, match="safety limit"):
        model().compile().sample(DATA, **options)


def test_oversized_prior_predictive_raises():
    with pytest.raises(ValueError, match="safety limit"):
        r.sample_prior_predictive(model().build(), DATA, HUGE)
    assert r.sample_prior_predictive(model().build(), DATA, 5)["obs"].shape == (5, 3)


def test_oversized_prediction_raises():
    fit = r.sample(model().build(), data=DATA, chains=2, draws=20, warmup=50,
                   show_progress=False)
    with pytest.raises(ValueError, match="safety limit"):
        fit.predict(sizes={"obs": HUGE})
    assert fit.predict(sizes={"obs": 4})["obs"].shape == (2, 20, 4)
