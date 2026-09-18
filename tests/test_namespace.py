"""``rustmc.__all__`` is the public API, and it must not rot.

The package re-exports the compiled extension with ``from ._rustmc import *``
and then pulls a handful of names out of two pure-Python submodules.  Importing
those submodules binds ``rustmc.evaluation`` and ``rustmc.forecasting`` as
package attributes, so without an explicit ``__all__`` they get dragged in by
``from rustmc import *`` -- advertising module layout that is not part of the
API.  (``dir(rustmc)`` reports the module dict and is unaffected by ``__all__``;
what ``__all__`` governs is ``import *`` and the tooling that honours it.)

Every expectation below is *derived* -- from the extension's own exports, and
from what is importable from the package -- rather than hard-coded, so adding an
export does not require touching this file.  It requires adding the name to
``__all__``, and the failure message says which name that is.
"""

from __future__ import annotations

import inspect

import pytest


@pytest.fixture
def rustmc(rustmc_module):
    return rustmc_module


def _public_attributes(module):
    """Non-private, non-submodule attributes: the intended public surface.

    ``from ._rustmc import *`` honours the extension's own ``__all__``, and the
    submodule imports in ``__init__.py`` are explicit, so anything left after
    dropping underscore names and module objects is something a user is meant
    to reach for.
    """
    return {
        name: value
        for name, value in vars(module).items()
        if not name.startswith("_") and not inspect.ismodule(value)
    }


def _extension_exports():
    """What the compiled extension itself advertises.

    Derived straight from ``rustmc._rustmc`` rather than from the package
    namespace, so a new native export is caught even if a same-named Python
    object would otherwise shadow it in ``vars(rustmc)``.
    """
    import rustmc._rustmc as native

    advertised = getattr(native, "__all__", None) or dir(native)
    return {name for name in advertised if not name.startswith("_")}


def test_all_is_sorted_and_free_of_duplicates(rustmc):
    listed = rustmc.__all__
    assert listed == sorted(listed), (
        "rustmc.__all__ must be sorted; out of order at: "
        + ", ".join(
            f"{a!r} before {b!r}"
            for a, b in zip(listed, listed[1:])
            if a > b
        )
    )
    duplicates = sorted({name for name in listed if listed.count(name) > 1})
    assert not duplicates, f"rustmc.__all__ repeats: {duplicates}"


def test_every_listed_name_resolves(rustmc):
    unresolved = [name for name in rustmc.__all__ if not hasattr(rustmc, name)]
    assert not unresolved, (
        "rustmc.__all__ names attributes that do not exist: " + ", ".join(unresolved)
    )


def test_all_lists_no_private_or_submodule_names(rustmc):
    private = [name for name in rustmc.__all__ if name.startswith("_")]
    assert not private, f"rustmc.__all__ must not export private names: {private}"

    submodules = [
        name
        for name in rustmc.__all__
        if inspect.ismodule(getattr(rustmc, name, None))
    ]
    assert not submodules, (
        "rustmc.__all__ must not export submodule objects (they are layout, not "
        f"API): {submodules}"
    )


def test_every_extension_export_is_listed(rustmc):
    """The load-bearing one: a new native export must be added to ``__all__``.

    Read from the extension, not from ``vars(rustmc)``, so this still fails when
    a Python-level name of the same spelling is bound over the native one.
    """
    unlisted = sorted(_extension_exports() - set(rustmc.__all__))
    assert not unlisted, (
        "the rustmc._rustmc extension exports these, but rustmc.__all__ does not "
        "list them -- add each one to the list in python/rustmc/__init__.py, or "
        "stop exporting it from the extension: " + ", ".join(unlisted)
    )


def test_every_extension_export_is_reachable_from_the_package(rustmc):
    """``__all__`` must not merely *name* an export it has shadowed away."""
    import rustmc._rustmc as native

    wrong = sorted(
        name
        for name in _extension_exports()
        if getattr(rustmc, name, None) is not getattr(native, name)
    )
    assert not wrong, (
        "rustmc re-binds these names to something other than the extension's "
        "object: " + ", ".join(wrong)
    )


def test_nothing_public_is_missing_from_all(rustmc):
    """The same check one level out: anything importable must be listed."""
    unlisted = sorted(set(_public_attributes(rustmc)) - set(rustmc.__all__))
    assert not unlisted, (
        "these names are importable from rustmc but absent from rustmc.__all__ -- "
        "add each one to the list in python/rustmc/__init__.py, or stop exporting "
        "it: " + ", ".join(unlisted)
    )


def test_all_advertises_nothing_beyond_the_public_surface(rustmc):
    """The converse: ``__all__`` must not name things that are not public."""
    public = set(_public_attributes(rustmc))
    extra = sorted(name for name in rustmc.__all__ if name not in public)
    assert not extra, (
        "rustmc.__all__ lists names that are not part of the public surface "
        "(private, a submodule, or no longer exported): " + ", ".join(extra)
    )


def test_star_import_binds_exactly_all(rustmc):
    namespace: dict = {}
    exec("from rustmc import *", namespace)  # noqa: S102 - that is the thing under test
    imported = {name for name in namespace if not name.startswith("__")}
    assert imported == set(rustmc.__all__)


def test_star_import_does_not_leak_submodules(rustmc):
    namespace: dict = {}
    exec("from rustmc import *", namespace)  # noqa: S102
    leaked = sorted(name for name, value in namespace.items() if inspect.ismodule(value))
    assert not leaked, f"`from rustmc import *` leaked submodules: {leaked}"


def test_version_is_still_importable(rustmc):
    from rustmc import __version__

    assert isinstance(__version__, str) and __version__
    assert __version__ == rustmc.__version__
    assert "__version__" not in rustmc.__all__


def test_submodules_remain_reachable_by_explicit_import(rustmc):
    """Hiding them from ``__all__`` must not make them unimportable."""
    import rustmc.evaluation
    import rustmc.forecasting

    assert rustmc.evaluation.backtest is rustmc.backtest
    assert rustmc.forecasting.forecast_scenarios is rustmc.forecast_scenarios
