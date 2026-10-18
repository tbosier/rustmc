"""Forecast reconciliation: adapters from forecast results to the native reconciler.

The arithmetic lives in ``rustmc_core::reconciliation``. This module only turns
the inputs people have - forecast objects, per-node mappings - into the
``(chain, draw, node, step)`` array, ``(time, node)`` residuals and
``(time, bottom series)`` history that the native reconciler takes, placing
every node by name so that no input is matched by position by accident.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from . import _rustmc as _native
from ._rustmc import Hierarchy, ReconciledForecast

METHODS = (
    "bottom_up",
    "top_down_average_proportions",
    "top_down_forecast_proportions",
    "ols",
    "wls_struct",
    "wls_var",
    "mint_shrink",
)


def reconcile(
    draws: Any,
    hierarchy: Hierarchy | None = None,
    *,
    method: str,
    residuals: Any = None,
    history: Any = None,
) -> ReconciledForecast:
    """Make every joint draw of a hierarchy's forecasts add up.

    ``draws`` is one of:

    - an array shaped ``(chain, draw, node, step)``, nodes in
      ``hierarchy.node_names`` order;
    - a mapping from every node name to that node's ``(chain, draw, step)`` draws;
    - a ``ForecastDraws`` with a series axis, matched by its ``series`` labels
      when it has them and taken in node order when it does not;
    - a ``ForecastBatchForecast`` whose cell IDs are the node names;
    - a ``BayesianHierarchicalForecast``, whose hierarchy is built from its
      group and program names when ``hierarchy`` is omitted.

    Draw ``(c, d)`` of every node must come from the same joint draw; see the
    reconciliation guide for what happens when it does not. ``method`` is one
    of ``METHODS``. ``residuals`` (``wls_var``, ``mint_shrink``) are in-sample
    one-step forecast errors shaped ``(time, node)`` or a mapping from node
    name to its series; ``history`` (``top_down_average_proportions``) holds
    the observed bottom series shaped ``(time, bottom series)`` or as a mapping
    from bottom name to series.
    """
    hierarchy, base = _base_draws(draws, hierarchy)
    residuals = _columns(residuals, hierarchy.node_names, "residuals")
    history = _columns(history, hierarchy.bottom_names, "history")
    return _native._reconcile(base, hierarchy, method, residuals, history)


def _base_draws(draws: Any, hierarchy: Hierarchy | None) -> tuple[Hierarchy, Any]:
    from .forecasting import ForecastDraws

    if isinstance(draws, _native.BayesianHierarchicalForecast):
        return _hierarchical_forecast(draws, hierarchy)
    if hierarchy is None:
        raise ValueError(
            "hierarchy is required unless draws is a BayesianHierarchicalForecast"
        )
    if not isinstance(hierarchy, Hierarchy):
        raise ValueError(f"hierarchy must be a rustmc.Hierarchy; got {type(hierarchy).__name__}")
    names = hierarchy.node_names
    if isinstance(draws, _native.ForecastBatchForecast):
        _require_names(draws.ids, names, "ForecastBatchForecast cell IDs")
        return hierarchy, _stack({name: draws[name].observation_samples for name in names}, names)
    if isinstance(draws, ForecastDraws):
        samples = draws.observation_samples
        if samples.ndim != 4:
            raise ValueError(
                "a ForecastDraws to reconcile needs a series axis: (chain, draw, node, step)"
            )
        if draws.series is None:
            return hierarchy, samples
        _require_names(draws.series, names, "ForecastDraws series labels")
        order = [draws.series.index(name) for name in names]
        return hierarchy, samples[:, :, order, :]
    if isinstance(draws, Mapping):
        _require_names(draws, names, "draws keys")
        return hierarchy, _stack(draws, names)
    return hierarchy, draws


def _hierarchical_forecast(forecast: Any, hierarchy: Hierarchy | None) -> tuple[Hierarchy, Any]:
    """Total, group and program draws of a hierarchical-mean forecast.

    Its group and total draws are sums of its program draws, so they are
    coherent already and every projection method returns them unchanged. Nodes
    are placed by structure - the total, the groups in order of first
    appearance among the programs, then the programs in order, which is the
    order ``Hierarchy.from_labels`` gives ``(group, program)`` pairs - so a
    hierarchy built with another ``total`` or ``separator`` also fits.
    """
    groups = forecast.group_names
    programs = forecast.program_names
    membership = list(forecast.group_index)
    order = list(dict.fromkeys(membership))
    structure = np.vstack([
        np.ones((1, len(programs))),
        np.array([[float(g == group) for g in membership] for group in order]),
        np.eye(len(programs)),
    ])
    labels = [(groups[group], program) for program, group in zip(programs, membership)]
    if hierarchy is None:
        try:
            hierarchy = Hierarchy.from_labels(labels)
        except ValueError as error:
            raise ValueError(
                f"cannot name this forecast's nodes by default ({error}); build the hierarchy "
                "with Hierarchy.from_labels from (group name, program name) pairs, choosing "
                "total= and separator=, and pass it as hierarchy"
            ) from error
    elif not isinstance(hierarchy, Hierarchy) or hierarchy.summing_matrix.shape != structure.shape \
            or not np.array_equal(hierarchy.summing_matrix, structure):
        raise ValueError(
            "hierarchy must have the forecast's structure: the total, then its groups in order of "
            "first appearance among the programs, then its programs in order, as "
            "Hierarchy.from_labels builds from (group name, program name) pairs"
        )
    base = np.concatenate([
        forecast.total_observation_samples[:, :, None, :],
        forecast.group_observation_samples[:, :, order, :],
        forecast.observation_samples,
    ], axis=2)
    return hierarchy, base


def _require_names(given: Any, expected: Sequence[str], what: str) -> None:
    given = list(given)
    missing = [name for name in expected if name not in given]
    extra = [name for name in given if name not in set(expected)]
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing nodes {missing}")
        if extra:
            parts.append(f"names that are not nodes {extra}")
        raise ValueError(f"{what} must name every node of the hierarchy exactly once: " + "; ".join(parts))
    if len(set(given)) != len(given):
        raise ValueError(f"{what} repeat a node name")


def _stack(by_name: Mapping[str, Any], names: Sequence[str]) -> np.ndarray:
    """Per-node ``(chain, draw, step)`` draws stacked on a node axis."""
    arrays = [_native._real_array(by_name[name], f"draws for node {name!r}") for name in names]
    shape = arrays[0].shape
    for name, array in zip(names, arrays):
        if array.ndim != 3:
            raise ValueError(
                f"draws for node {name!r} must be (chain, draw, step); got {array.ndim} dimension(s)"
            )
        if array.shape != shape:
            raise ValueError(
                f"draws for node {name!r} have shape {array.shape}; node {names[0]!r} has {shape}, "
                "and every node needs the same chains, draws and steps"
            )
    return np.stack(arrays, axis=2)


def _columns(values: Any, names: Sequence[str], what: str) -> Any:
    """A ``(time, column)`` array from a name-keyed mapping, or ``values`` as given."""
    if values is None or not isinstance(values, Mapping):
        return values
    _require_names(values, names, f"{what} keys")
    columns = [_native._real_array(values[name], f"{what} for node {name!r}") for name in names]
    for name, column in zip(names, columns):
        if column.ndim != 1:
            raise ValueError(f"{what} for node {name!r} must be one series; got shape {column.shape}")
    for name, column in zip(names, columns):
        if len(column) != len(columns[0]):
            raise ValueError(
                f"{what} for node {name!r} has {len(column)} values; node {names[0]!r} has "
                f"{len(columns[0])}, and every node needs the same number"
            )
    return np.stack(columns, axis=1)
