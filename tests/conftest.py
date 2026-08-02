"""
Shared pytest fixtures for the rustmc test suite.

Nothing here should impose collection-time requirements such as environment
variables or network access on unrelated test modules.
"""
import os
import sys

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "network: test builds a wheel and/or hits the network (deselected by default)"
    )
    config.addinivalue_line(
        "markers", "slow: test is slower than the rest of the suite"
    )


@pytest.fixture(scope="session")
def rustmc_module():
    """
    Import rustmc once per session and hand back the module object.

    Does NOT assert anything about where it was loaded from -- that check
    is specific to packaging validation (see test_packaging.py /
    RUSTMC_REQUIRE_SITE_PACKAGES) and would be the wrong thing to force on
    every consumer of this fixture, since local dev workflows legitimately
    run against an editable `maturin develop` install.
    """
    import rustmc as rmc
    return rmc


@pytest.fixture(scope="session")
def linreg_data():
    """A small, deterministic synthetic linear-regression dataset."""
    import numpy as np

    rng = np.random.default_rng(42)
    n = 200
    x = rng.standard_normal(n)
    alpha_true, beta_true, sigma_true = 1.5, 2.5, 1.0
    y = alpha_true + beta_true * x + rng.standard_normal(n) * sigma_true
    return {
        "x": x,
        "y": y,
        "alpha_true": alpha_true,
        "beta_true": beta_true,
        "sigma_true": sigma_true,
    }


def _module_file_is_under_site_packages(mod) -> bool:
    mod_file = getattr(mod, "__file__", "") or ""
    parts = mod_file.replace(os.sep, "/").split("/")
    return "site-packages" in parts or "dist-packages" in parts


@pytest.fixture(scope="session")
def assert_installed_from_site_packages():
    """
    Helper fixture used by packaging tests that specifically need to prove
    they're exercising an installed wheel and not the repo's source tree
    (there is no pure-Python source tree to leak from today, but this
    guards the invariant going forward).
    """
    def _check(mod):
        assert _module_file_is_under_site_packages(mod), (
            f"expected {mod.__name__} to be importable from a site-packages "
            f"directory, got __file__={getattr(mod, '__file__', None)!r}"
        )
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        mod_file = os.path.abspath(getattr(mod, "__file__", ""))
        assert not mod_file.startswith(repo_root + os.sep), (
            f"{mod.__name__} was loaded from inside the repo checkout ({mod_file}), "
            "not an installed package -- sys.path leakage suspected"
        )
        return mod_file
    return _check
