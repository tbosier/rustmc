"""Named forecast draws, scenario forecasts, portable storage, and refit updates."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _cumulative(values: np.ndarray) -> np.ndarray:
    with np.errstate(over="raise", invalid="raise"):
        try:
            return values.cumsum(axis=-1)
        except FloatingPointError as error:
            raise ValueError("cumulative forecast exceeds finite floating-point range") from error


@dataclass(frozen=True)
class NamedDesign:
    values: np.ndarray
    features: tuple[str, ...]

    def __post_init__(self) -> None:
        values = np.array(self.values, dtype=float, copy=True)
        names = tuple(self.features)
        if values.ndim not in (2, 3) or values.shape[-1] != len(names) or not np.isfinite(values).all():
            raise ValueError("design must be finite (..., time, feature) data matching feature names")
        if len(set(names)) != len(names) or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("feature names must be unique nonempty strings")
        values.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "features", names)

    def require_features(self, expected: Sequence[str]) -> np.ndarray:
        if tuple(expected) != self.features:
            raise ValueError(f"future features {self.features!r} must match training order {tuple(expected)!r}")
        return self.values


@dataclass(frozen=True)
class ForecastDraws:
    """Joint paths with leading chain/draw axes and a final horizon axis.

    mean_samples, when present, are conditional expected responses. Observation
    draws also include realization noise. Intervals are pointwise equal-tailed.
    """
    observation_samples: np.ndarray
    mean_samples: np.ndarray | None = None
    dates: tuple[str, ...] | None = None
    series: tuple[str, ...] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        observations = np.array(self.observation_samples, dtype=float, copy=True)
        if observations.ndim not in (3, 4) or any(d == 0 for d in observations.shape) or not np.isfinite(observations).all():
            raise ValueError("draws must be finite (chain, draw, [series,] horizon) arrays")
        observations.setflags(write=False)
        object.__setattr__(self, "observation_samples", observations)
        if self.mean_samples is not None:
            means = np.array(self.mean_samples, dtype=float, copy=True)
            if means.shape != observations.shape or not np.isfinite(means).all():
                raise ValueError("conditional mean draws must match observation draws and be finite")
            means.setflags(write=False)
            object.__setattr__(self, "mean_samples", means)
        if self.dates is not None:
            dates = tuple(str(x) for x in self.dates)
            if len(dates) != observations.shape[-1] or len(set(dates)) != len(dates):
                raise ValueError("dates must provide one unique label per horizon")
            object.__setattr__(self, "dates", dates)
        if self.series is not None:
            series = tuple(self.series)
            if observations.ndim != 4 or len(series) != observations.shape[2] or len(set(series)) != len(series):
                raise ValueError("series labels must uniquely identify the series axis")
            object.__setattr__(self, "series", series)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_result(cls, result: Any, *, dates: Sequence[Any] | None = None, series: Sequence[str] | None = None) -> ForecastDraws:
        return cls(result.observation_samples, getattr(result, "mean_samples", None),
            None if dates is None else tuple(dates), None if series is None else tuple(series),
            {"source": type(result).__name__, "uncertainty_kind": getattr(result, "uncertainty_kind", "posterior_predictive")})

    @property
    def observation_mean(self) -> np.ndarray:
        count = np.prod(self.observation_samples.shape[:2])
        return (self.observation_samples / count).sum(axis=(0, 1))

    @property
    def cumulative_observation_samples(self) -> np.ndarray:
        return _cumulative(self.observation_samples)

    def interval(self, level: float = 0.95, *, kind: str = "observation", cumulative: bool = False) -> tuple[np.ndarray, np.ndarray]:
        if not np.isfinite(level) or not 0 < level < 1:
            raise ValueError("level must lie strictly between zero and one")
        if kind not in ("observation", "mean"):
            raise ValueError("kind must be 'observation' or 'mean'")
        draws = self.observation_samples if kind == "observation" else self.mean_samples
        if draws is None:
            raise ValueError("this forecast does not expose conditional mean draws")
        if cumulative:
            draws = _cumulative(draws)
        lower, upper = np.quantile(draws, ((1-level)/2, (1+level)/2), axis=(0, 1))
        return lower, upper

    def aggregate(self, weights: Sequence[float] | None = None) -> ForecastDraws:
        """Sum within each joint panel draw, preserving cross-series dependence."""
        if self.observation_samples.ndim != 4:
            raise ValueError("aggregation requires a series axis")
        weight = np.ones(self.observation_samples.shape[2]) if weights is None else np.asarray(weights, dtype=float)
        if weight.shape != (self.observation_samples.shape[2],) or not np.isfinite(weight).all():
            raise ValueError("weights must contain one finite value per series")
        mean = None if self.mean_samples is None else np.einsum("cdsh,s->cdh", self.mean_samples, weight)
        return ForecastDraws(np.einsum("cdsh,s->cdh", self.observation_samples, weight), mean, self.dates,
            metadata={**self.metadata, "aggregation_weights": weight.tolist()})

    def save(self, path: str | Path) -> None:
        """Store a versioned NumPy archive without Python pickle payloads."""
        metadata = json.dumps({"format": "rustmc.forecast", "version": 1, "dates": self.dates,
            "series": self.series, "metadata": self.metadata, "has_mean": self.mean_samples is not None}, allow_nan=False)
        arrays = {"observations": self.observation_samples, "metadata": np.array(metadata)}
        if self.mean_samples is not None:
            arrays["means"] = self.mean_samples
        with Path(path).open("wb") as handle:
            np.savez_compressed(handle, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> ForecastDraws:
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
            if metadata.get("format") != "rustmc.forecast" or metadata.get("version") != 1:
                raise ValueError("unsupported forecast artifact format/version")
            return cls(archive["observations"], archive["means"] if metadata["has_mean"] else None,
                metadata["dates"], metadata["series"], metadata["metadata"])


@dataclass(frozen=True)
class ScenarioForecast:
    forecasts: Mapping[str, ForecastDraws]
    probabilities: Mapping[str, float]

    def mixture(self, *, draws: int = 1000, seed: int = 42) -> ForecastDraws:
        """Draw a mixture of whole paths; scenario weights are independent of posterior parameters."""
        if isinstance(draws, bool) or not isinstance(draws, (int, np.integer)) or draws < 1:
            raise ValueError("draws must be a positive integer")
        labels = list(self.forecasts)
        if not labels or set(labels) != set(self.probabilities):
            raise ValueError("scenario probabilities must identify every forecast")
        probabilities = np.array([self.probabilities[label] for label in labels], dtype=float)
        if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1.0, rtol=1e-10, atol=1e-12):
            raise ValueError("scenario probabilities must be nonnegative and sum to one")
        first = self.forecasts[labels[0]]
        for forecast in self.forecasts.values():
            if forecast.observation_samples.shape[2:] != first.observation_samples.shape[2:] or forecast.dates != first.dates or forecast.series != first.series:
                raise ValueError("scenario target shapes and coordinates must agree")
        rng = np.random.default_rng(seed)
        choices = rng.choice(len(labels), draws, p=probabilities/probabilities.sum())
        output = np.empty((draws,) + first.observation_samples.shape[2:])
        has_mean = all(f.mean_samples is not None for f in self.forecasts.values())
        means = np.empty_like(output) if has_mean else None
        for scenario_index, label in enumerate(labels):
            selected = np.flatnonzero(choices == scenario_index)
            forecast = self.forecasts[label]
            flat = forecast.observation_samples.reshape((-1,) + output.shape[1:])
            positions = rng.integers(len(flat), size=len(selected))
            output[selected] = flat[positions]
            if means is not None:
                means[selected] = forecast.mean_samples.reshape(flat.shape)[positions]
        return ForecastDraws(output[np.newaxis], None if means is None else means[np.newaxis], first.dates, first.series,
            {"scenario_labels": labels, "probabilities": probabilities.tolist(), "draw_kind": "scenario_mixture"})


def forecast_scenarios(fit: Any, scenarios: Mapping[str, Any], *, probabilities: Mapping[str, float] | None = None,
    seed: int = 43, dates: Sequence[Any] | None = None, **kwargs: Any) -> ScenarioForecast:
    """Forecast each specified future design and retain conditional scenario results."""
    from ._rustmc import forecast_cell_seed
    if not scenarios or any(not isinstance(label, str) or not label for label in scenarios):
        raise ValueError("scenarios must have nonempty string names")
    outputs = {}
    for label, design in scenarios.items():
        values = design.values if isinstance(design, NamedDesign) else np.asarray(design, dtype=float)
        if values.ndim not in (2, 3):
            raise ValueError("scenario exog must have time and feature axes")
        if isinstance(design, NamedDesign) and not isinstance(fit, ForecastSession):
            raise ValueError("named scenario designs require a ForecastSession to verify training feature identities")
        result = fit.forecast(values.shape[-2], exog=design if isinstance(fit, ForecastSession) else values,
            seed=forecast_cell_seed(seed, label, "forecast"), **kwargs)
        outputs[label] = ForecastDraws.from_result(result, dates=dates)
    probs = dict(probabilities) if probabilities is not None else {key: 1/len(outputs) for key in outputs}
    result = ScenarioForecast(outputs, probs)
    # Validate weights/shape without consuming or changing caller seeds.
    result.mixture(draws=1, seed=seed)
    return result


class ForecastSession:
    """Maintain data for a forecasting model and refit its parameter posterior on update.

    update is an explicit full Bayesian refit, not a fixed-parameter filtering
    approximation. Existing model and fit objects remain available to the caller.
    """
    def __init__(self, model: Any, observations: Any, *, exog: NamedDesign | Any = None,
        exposure: Any = None, fit_kwargs: Mapping[str, Any] | None = None):
        self.model = model
        self.observations = np.array(observations, dtype=float, copy=True)
        self.features = exog.features if isinstance(exog, NamedDesign) else None
        self.exog = None if exog is None else np.array(exog.values if isinstance(exog, NamedDesign) else exog, dtype=float, copy=True)
        self.exposure = None if exposure is None else np.array(exposure, dtype=float, copy=True)
        self.fit_kwargs = dict(fit_kwargs or {})
        if any(key in self.fit_kwargs for key in ("exog", "exposure")):
            raise ValueError("pass exog/exposure directly so the session retains them for updates")
        self.fit = self._fit(self.observations, self.exog, self.exposure, self.fit_kwargs)

    def _fit(self, observations: np.ndarray, exog: np.ndarray | None, exposure: np.ndarray | None, options: Mapping[str, Any]) -> Any:
        options = dict(options)
        if exog is not None:
            options["exog"] = exog
        if exposure is not None:
            options["exposure"] = exposure
        return self.model.fit(observations, **options)

    def update(self, observations: Any, *, exog: NamedDesign | Any = None, exposure: Any = None, **fit_kwargs: Any) -> Any:
        new_y = np.asarray(observations, dtype=float)
        if new_y.ndim != self.observations.ndim or new_y.shape[:-1] != self.observations.shape[:-1] or not new_y.shape[-1]:
            raise ValueError("new observations must preserve series dimensions and add time points")
        if (exog is None) != (self.exog is None) or (exposure is None) != (self.exposure is None):
            raise ValueError("updates must supply the same design/exposure fields used for training")
        new_x = self._future_design(exog)
        if new_x is not None and (new_x.shape[:-2] != new_y.shape[:-1] or new_x.shape[-2] != new_y.shape[-1]):
            raise ValueError("update design must align with new observations")
        new_e = None if exposure is None else np.asarray(exposure, dtype=float)
        if new_e is not None and new_e.shape != new_y.shape:
            raise ValueError("update exposure must align with new observations")
        y = np.concatenate((self.observations, new_y), axis=-1)
        x = None if new_x is None else np.concatenate((self.exog, new_x), axis=-2)
        e = None if new_e is None else np.concatenate((self.exposure, new_e), axis=-1)
        options = {**self.fit_kwargs, **fit_kwargs}
        fitted = self._fit(y, x, e, options)
        # Only mutate session after the complete refit succeeds.
        self.observations, self.exog, self.exposure, self.fit_kwargs, self.fit = y, x, e, options, fitted
        return fitted

    def _future_design(self, exog: Any) -> np.ndarray | None:
        if exog is None:
            return None
        if self.features is not None:
            if not isinstance(exog, NamedDesign):
                raise ValueError("future exog must be NamedDesign to verify training feature identities")
            return exog.require_features(self.features)
        return np.asarray(exog.values if isinstance(exog, NamedDesign) else exog, dtype=float)

    def forecast(self, steps: int, *, exog: Any = None, dates: Sequence[Any] | None = None, **kwargs: Any) -> ForecastDraws:
        if exog is not None:
            kwargs["exog"] = self._future_design(exog)
        return ForecastDraws.from_result(self.fit.forecast(steps, **kwargs), dates=dates)
