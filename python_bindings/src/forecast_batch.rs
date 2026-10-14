//! Native independent-cell forecasting bindings. One pool spans cells and chains.
use super::ar::{PyBayesianArFit, PyBayesianArForecast, PyBayesianAutoRegression};
use super::hurdle::{PyHurdleFit, PyHurdleForecast, PyHurdleLogNormal};
use super::local_level::{PyBayesianForecastResult, PyBayesianLocalLevel, PyBayesianLocalLevelFit};
use super::regression::{
    PyBayesianRegressionFit, PyBayesianRegressionForecast, PyGaussianCoefficientPrior,
};
use super::seasonal::{
    PyBayesianSeasonalForecast, PyBayesianSeasonalLocalLevel, PyBayesianSeasonalLocalLevelFit,
};
use super::trend::{
    PyBayesianLocalLinearTrend, PyBayesianLocalLinearTrendFit, PyBayesianTrendForecast,
};
use crate::forecast_support::{real_matrix, real_vector, DiagnosticsReport, ForecastFit, Schedule};
use crate::{forecast_diagnostics, InferenceError};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rustmc_core::bayesian_ar::{
    fit_bayesian_ar, BayesianArConfig as CoreBayesianArConfig,
    BayesianArForecast as CoreBayesianArForecast,
};
use rustmc_core::bayesian_forecast::{
    fit_bayesian_local_level, BayesianLocalLevelConfig as CoreBayesianLocalLevelConfig,
    PosteriorPredictiveForecast as CorePosteriorPredictiveForecast,
};
use rustmc_core::bayesian_regression::{
    fit_regression, GaussianCoefficientPrior, RegressionConfig, RegressionForecast,
};
use rustmc_core::bayesian_seasonal::{
    fit_bayesian_seasonal_local_level,
    BayesianSeasonalLocalLevelConfig as CoreBayesianSeasonalLocalLevelConfig,
    SeasonalPosteriorPredictiveForecast as CoreSeasonalPosteriorPredictiveForecast,
};
use rustmc_core::bayesian_trend::{
    fit_bayesian_local_linear_trend,
    BayesianLocalLinearTrendConfig as CoreBayesianLocalLinearTrendConfig,
    TrendPosteriorPredictiveForecast as CoreTrendPosteriorPredictiveForecast,
};
use rustmc_core::forecast_batch::{
    execute_batch, execute_batch_fail_fast, stable_cell_seed, BatchError,
};
use rustmc_core::hurdle::{fit_hurdle_lognormal, HurdleLogNormalConfig, HurdleLogNormalForecast};
use std::sync::Arc;

// Limit each call's retained scalar values and each cell's conservative FFBS
// workspace estimate. Larger workloads must use caller-managed chunks.
const MAX_BATCH_VALUES: usize = 25_000_000;
fn allocation_product(values: &[usize]) -> Result<usize, String> {
    values.iter().try_fold(1usize, |product, value| product.checked_mul(*value))
        .filter(|count| *count <= MAX_BATCH_VALUES)
        .ok_or_else(|| "allocation limit exceeded (25,000,000 scalar values); reduce draws, horizon or cell chunk size".into())
}
fn check_total_retention(mut values: impl Iterator<Item = usize>) -> PyResult<()> {
    let total = values.try_fold(0usize, |sum, value| sum.checked_add(value));
    if total.is_none_or(|total| total > MAX_BATCH_VALUES) {
        return Err(PyValueError::new_err("batch retained-value limit exceeded (25,000,000); submit smaller caller-managed chunks"));
    }
    Ok(())
}

#[derive(Clone)]
pub(crate) enum Config {
    Local(CoreBayesianLocalLevelConfig),
    Seasonal(CoreBayesianSeasonalLocalLevelConfig),
    Trend(CoreBayesianLocalLinearTrendConfig),
    Ar(CoreBayesianArConfig),
    Hurdle(HurdleLogNormalConfig),
    Regression(Box<RegressionConfig>, Vec<Vec<f64>>),
}
#[derive(Clone)]
enum CellFit {
    Local(PyBayesianLocalLevelFit),
    Seasonal(PyBayesianSeasonalLocalLevelFit),
    Trend(PyBayesianLocalLinearTrendFit),
    Ar(PyBayesianArFit),
    Hurdle(PyHurdleFit),
    Regression(Box<PyBayesianRegressionFit>),
}
/// Payloads are shared with the Python objects built from them, so handing a
/// cell to Python does not copy its paths.
#[derive(Clone)]
enum CellForecast {
    Local(Arc<CorePosteriorPredictiveForecast>),
    Seasonal(Arc<CoreSeasonalPosteriorPredictiveForecast>),
    Trend(Arc<CoreTrendPosteriorPredictiveForecast>),
    Ar(Arc<CoreBayesianArForecast>),
    Hurdle(Arc<HurdleLogNormalForecast>),
    Regression(Arc<RegressionForecast>, bool, usize),
}
impl Config {
    fn allocation_size(&self, history: usize) -> Result<usize, String> {
        let (chains, draws, parameters, dimension) = match self {
            Self::Hurdle(c) => (c.num_chains, c.num_draws, 4, 1),
            Self::Local(c) => (c.num_chains, c.num_draws, 3, 1),
            Self::Seasonal(c) => (
                c.num_chains,
                c.num_draws,
                c.period.checked_add(3).ok_or("dimension overflow")?,
                c.period,
            ),
            Self::Trend(c) => (c.num_chains, c.num_draws, 5, 2),
            Self::Ar(c) => (
                c.num_chains,
                c.num_draws,
                c.order.checked_add(2).ok_or("dimension overflow")?,
                c.order,
            ),
            Self::Regression(c, _) => {
                let dimension = c
                    .structural_model
                    .dimension()
                    .checked_add(c.coefficient_prior.mean.len())
                    .ok_or("dimension overflow")?;
                let parameters = dimension
                    .checked_add(c.variance_priors.len())
                    .and_then(|n| n.checked_add(1))
                    .ok_or("dimension overflow")?;
                (c.num_chains, c.num_draws, parameters, dimension)
            }
        };
        // Include a pre-observation state and conservative dense filter/smoother buffers.
        allocation_product(&[
            chains.max(1),
            history.checked_add(1).ok_or("history overflow")?,
            dimension,
            dimension,
            8,
        ])?;
        let posterior = allocation_product(&[chains, draws, parameters])?;
        let retained = posterior
            .checked_add(history)
            .ok_or("retained size overflow")?;
        allocation_product(&[retained])
    }
    fn with_regression(
        self,
        design: Vec<Vec<f64>>,
        prior: GaussianCoefficientPrior,
    ) -> Result<Self, String> {
        let config = match self {
            Self::Hurdle(_) => return Err("hurdle exog is unsupported: occurrence is static and severity has no exogenous features".into()),
            Self::Local(config) => RegressionConfig::from_local_level(&config, prior),
            Self::Seasonal(config) => RegressionConfig::from_seasonal(&config, prior),
            Self::Trend(config) => RegressionConfig::from_trend(&config, prior),
            _ => return Err("exog is supported for local-level, seasonal and trend models".into()),
        }
        .map_err(|error| error.to_string())?;
        Ok(Self::Regression(Box::new(config), design))
    }
    fn fit(&self, observations: &[f64], seed: u64) -> Result<CellFit, String> {
        self.allocation_size(observations.len())?;
        match self {
            Self::Hurdle(base) => {
                let mut config = base.clone();
                config.seed = seed;
                let posterior = fit_hurdle_lognormal(observations, &config)
                    .map_err(|error| error.to_string())?;
                Ok(CellFit::Hurdle(PyHurdleFit {
                    posterior: Arc::new(posterior),
                    observations: observations.into(),
                    config,
                }))
            }
            Self::Regression(base, design) => {
                let mut config = (**base).clone();
                config.seed = seed;
                let posterior = fit_regression(observations, design, &config)
                    .map_err(|error| error.to_string())?;
                Ok(CellFit::Regression(Box::new(PyBayesianRegressionFit {
                    posterior: Arc::new(posterior),
                    observations: observations.into(),
                })))
            }
            Self::Local(base) => {
                let mut config = base.clone();
                config.seed = seed;
                let posterior =
                    fit_bayesian_local_level(observations, &config).map_err(|e| e.to_string())?;
                Ok(CellFit::Local(PyBayesianLocalLevelFit {
                    posterior: Arc::new(posterior),
                    observations: observations.into(),
                    config,
                }))
            }
            Self::Seasonal(base) => {
                let mut config = base.clone();
                config.seed = seed;
                let posterior = fit_bayesian_seasonal_local_level(observations, &config)
                    .map_err(|e| e.to_string())?;
                Ok(CellFit::Seasonal(PyBayesianSeasonalLocalLevelFit {
                    posterior: Arc::new(posterior),
                    observations: observations.into(),
                    config,
                }))
            }
            Self::Trend(base) => {
                let mut config = base.clone();
                config.seed = seed;
                let posterior = fit_bayesian_local_linear_trend(observations, &config)
                    .map_err(|e| e.to_string())?;
                Ok(CellFit::Trend(PyBayesianLocalLinearTrendFit {
                    posterior: Arc::new(posterior),
                    observations: observations.into(),
                    config,
                }))
            }
            Self::Ar(base) => {
                let mut config = base.clone();
                config.seed = seed;
                let posterior =
                    fit_bayesian_ar(observations, &config).map_err(|e| e.to_string())?;
                Ok(CellFit::Ar(PyBayesianArFit {
                    posterior: Arc::new(posterior),
                    observations: observations.into(),
                    config,
                }))
            }
        }
    }
}
impl CellFit {
    fn as_fit(&self) -> &dyn ForecastFit {
        match self {
            Self::Local(fit) => fit,
            Self::Seasonal(fit) => fit,
            Self::Trend(fit) => fit,
            Self::Ar(fit) => fit,
            Self::Hurdle(fit) => fit,
            Self::Regression(fit) => &**fit,
        }
    }
    fn forecast_allocation_size(&self, steps: usize) -> Result<usize, String> {
        let components = match self {
            Self::Local(_) | Self::Ar(_) | Self::Hurdle(_) => 3,
            Self::Seasonal(_) | Self::Trend(_) => 4,
            Self::Regression(_) => 6,
        };
        let (chains, draws) = self.as_fit().shape();
        allocation_product(&[chains, draws, steps, components])
    }
    fn to_python(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self {
            Self::Regression(fit) => Ok(Py::new(py, (**fit).clone())?.into_any()),
            Self::Hurdle(fit) => Ok(Py::new(py, fit.clone())?.into_any()),
            Self::Local(fit) => Ok(Py::new(py, fit.clone())?.into_any()),
            Self::Seasonal(fit) => Ok(Py::new(py, fit.clone())?.into_any()),
            Self::Trend(fit) => Ok(Py::new(py, fit.clone())?.into_any()),
            Self::Ar(fit) => Ok(Py::new(py, fit.clone())?.into_any()),
        }
    }
    fn report(&self) -> DiagnosticsReport {
        self.as_fit().report()
    }
    fn forecast(
        &self,
        steps: usize,
        seed: u64,
        exog: Option<&[Vec<f64>]>,
    ) -> Result<CellForecast, String> {
        self.forecast_allocation_size(steps)?;
        if exog.is_some() && matches!(self, Self::Hurdle(_)) {
            return Err("hurdle exog is unsupported: occurrence is static and severity has no exogenous features".into());
        }
        if exog.is_some() && !matches!(self, Self::Regression(_)) {
            return Err("future exog was supplied to a fit without regression".into());
        }
        match self {
            Self::Regression(fit) => {
                let design = exog.ok_or("future exog is required for regression forecasts")?;
                if design.len() != steps {
                    return Err("future exog row count must equal steps".into());
                }
                fit.posterior
                    .forecast(design, seed)
                    .map(|inner| {
                        CellForecast::Regression(
                            Arc::new(inner),
                            fit.posterior.config.seasonal,
                            fit.posterior.config.structural_model.dimension(),
                        )
                    })
                    .map_err(|error| error.to_string())
            }
            Self::Hurdle(fit) => fit
                .posterior
                .forecast(steps, seed)
                .map(|inner| CellForecast::Hurdle(Arc::new(inner)))
                .map_err(|error| error.to_string()),
            Self::Local(fit) => fit
                .posterior
                .forecast(steps, seed)
                .map(|inner| CellForecast::Local(Arc::new(inner)))
                .map_err(|e| e.to_string()),
            Self::Seasonal(fit) => fit
                .posterior
                .forecast(steps, seed)
                .map(|inner| CellForecast::Seasonal(Arc::new(inner)))
                .map_err(|e| e.to_string()),
            Self::Trend(fit) => fit
                .posterior
                .forecast(steps, seed)
                .map(|inner| CellForecast::Trend(Arc::new(inner)))
                .map_err(|e| e.to_string()),
            Self::Ar(fit) => fit
                .posterior
                .forecast(steps, seed)
                .map(|inner| CellForecast::Ar(Arc::new(inner)))
                .map_err(|e| e.to_string()),
        }
    }
}
impl CellForecast {
    fn to_python(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self {
            Self::Regression(inner, seasonal, dimension) => Ok(Py::new(
                py,
                PyBayesianRegressionForecast {
                    inner: inner.clone(),
                    seasonal: *seasonal,
                    dimension: *dimension,
                },
            )?
            .into_any()),
            Self::Hurdle(inner) => Ok(Py::new(
                py,
                PyHurdleForecast {
                    inner: inner.clone(),
                },
            )?
            .into_any()),
            Self::Local(inner) => Ok(Py::new(
                py,
                PyBayesianForecastResult {
                    inner: inner.clone(),
                },
            )?
            .into_any()),
            Self::Seasonal(inner) => Ok(Py::new(
                py,
                PyBayesianSeasonalForecast {
                    inner: inner.clone(),
                },
            )?
            .into_any()),
            Self::Trend(inner) => Ok(Py::new(
                py,
                PyBayesianTrendForecast {
                    inner: inner.clone(),
                },
            )?
            .into_any()),
            Self::Ar(inner) => Ok(Py::new(
                py,
                PyBayesianArForecast {
                    inner: inner.clone(),
                },
            )?
            .into_any()),
        }
    }
}

fn run_cells<T: Sync, R: Send>(
    cells: &[(String, T)],
    seed: u64,
    threads: usize,
    chunk_size: usize,
    errors: &str,
    fit: impl Fn(&T, u64) -> Result<R, String> + Sync,
) -> PyResult<Vec<Result<R, String>>> {
    if errors == "raise" {
        execute_batch_fail_fast(cells, seed, threads, chunk_size, fit)
            .map(|results| results.into_iter().map(Ok).collect())
            .map_err(|error| match error {
                BatchError::Configuration(error) => PyValueError::new_err(error),
                BatchError::Cell { id, error } => {
                    InferenceError::new_err(format!("cell {id:?}: {error}"))
                }
            })
    } else {
        execute_batch(cells, seed, threads, chunk_size, fit).map_err(PyValueError::new_err)
    }
}

/// A cell's converted observations and model, or why it could not be built.
/// The size is `Config::allocation_size` for the cell's history, computed when
/// the cell is prepared and reused by the batch retention check.
type CellInput = Result<(Vec<f64>, Config, usize), String>;

/// A forecasting model that can fit `fit_batch` cells.
pub(crate) trait BatchModel {
    /// This model's cell config under `schedule`. AR ignores warmup and thin.
    fn batch_config(&self, schedule: Schedule) -> Config;
}

/// The config of a `models` entry, if it is a forecasting model.
fn batch_model_config(model: &Bound<'_, PyAny>, schedule: Schedule) -> Option<Config> {
    fn config<T: pyo3::PyClass + BatchModel>(
        model: &Bound<'_, PyAny>,
        schedule: Schedule,
    ) -> Option<Config> {
        let model = model.extract::<PyRef<'_, T>>().ok()?;
        Some(model.batch_config(schedule))
    }
    config::<PyHurdleLogNormal>(model, schedule)
        .or_else(|| config::<PyBayesianLocalLevel>(model, schedule))
        .or_else(|| config::<PyBayesianSeasonalLocalLevel>(model, schedule))
        .or_else(|| config::<PyBayesianLocalLinearTrend>(model, schedule))
        .or_else(|| config::<PyBayesianAutoRegression>(model, schedule))
}

/// Refuse a `warmup` or `thin` that no cell would use.
///
/// AR draws are exact and independent, so an all-AR batch has no warmup or
/// thinning; silently dropping the settings would let a caller believe they
/// had changed the fit. `all_ar` comes from each cell's model kind, which does
/// not depend on whether its observations or exog converted; only a cell with
/// an invalid `models` entry has no known kind, and it counts as possibly Gibbs.
fn reject_unused_schedule(
    all_ar: bool,
    warmup: Option<usize>,
    thin: Option<usize>,
) -> PyResult<()> {
    if all_ar && (warmup.is_some() || thin.is_some()) {
        return Err(PyValueError::new_err(
            "warmup and thin do not apply to BayesianAutoRegression cells, whose posterior \
             draws are exact and independent; leave them unset",
        ));
    }
    Ok(())
}

/// A cell's input conversion error as the message stored for that cell.
fn cell_input<T>(py: Python<'_>, input: PyResult<T>) -> Result<T, String> {
    input.map_err(|error| error.value(py).to_string())
}

fn check_errors(errors: &str) -> PyResult<()> {
    if errors != "raise" && errors != "collect" {
        return Err(PyValueError::new_err("errors must be 'raise' or 'collect'"));
    }
    Ok(())
}
fn collected_errors<'py, T>(
    py: Python<'py>,
    ids: &[String],
    results: &[Result<T, String>],
) -> PyResult<Bound<'py, PyDict>> {
    let errors = PyDict::new(py);
    for (id, result) in ids.iter().zip(results) {
        if let Err(error) = result {
            errors.set_item(id, error)?;
        }
    }
    Ok(errors)
}
fn index_of(ids: &[String], id: &str) -> PyResult<usize> {
    ids.iter()
        .position(|candidate| candidate == id)
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(id.to_string()))
}

fn aligned_optional<'py>(
    values: Option<&Bound<'py, PyAny>>,
    count: usize,
    name: &str,
) -> PyResult<Option<Vec<Bound<'py, PyAny>>>> {
    let values = values
        .map(|values| values.try_iter()?.collect::<PyResult<Vec<_>>>())
        .transpose()?;
    if values.as_ref().is_some_and(|values| values.len() != count) {
        return Err(PyValueError::new_err(format!(
            "{name} must have one entry per cell"
        )));
    }
    Ok(values)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn fit_batch(
    py: Python<'_>,
    model: &dyn BatchModel,
    observations: &Bound<'_, PyAny>,
    ids: Vec<String>,
    models: Option<&Bound<'_, PyAny>>,
    exog: Option<&Bound<'_, PyAny>>,
    coefficient_priors: Option<&Bound<'_, PyAny>>,
    chains: usize,
    draws: usize,
    warmup: Option<usize>,
    thin: Option<usize>,
    seed: u64,
    threads: usize,
    chunk_size: usize,
    errors: &str,
) -> PyResult<PyForecastBatchFit> {
    check_errors(errors)?;
    let schedule = Schedule {
        chains,
        draws,
        warmup: warmup.unwrap_or(Schedule::DEFAULT.warmup),
        thin: thin.unwrap_or(Schedule::DEFAULT.thin),
        seed,
    };
    let default = model.batch_config(schedule);
    let rows = observations.try_iter()?.collect::<PyResult<Vec<_>>>()?;
    if ids.len() != rows.len() {
        return Err(PyValueError::new_err(
            "ids and observations must have the same length",
        ));
    }
    let models = aligned_optional(models, ids.len(), "models")?;
    let designs = aligned_optional(exog, ids.len(), "exog")?;
    let priors = aligned_optional(coefficient_priors, ids.len(), "coefficient_priors")?;
    let mut cells = Vec::with_capacity(ids.len());
    let mut all_ar = !rows.is_empty();
    for (index, row) in rows.iter().enumerate() {
        // The cell's model is resolved before its input, so its kind is known
        // even when the input fails; the observations error still comes first.
        let config = match &models {
            Some(models) if !models[index].is_none() => {
                batch_model_config(&models[index], schedule)
            }
            _ => Some(default.clone()),
        };
        all_ar &= matches!(config, Some(Config::Ar(_)));
        let input = (|| -> CellInput {
            let y = cell_input(py, real_vector(row, "observations"))?;
            let config = config.ok_or(
                "invalid configuration: models entries must be forecasting models or None",
            )?;
            let design = designs
                .as_ref()
                .map(|items| &items[index])
                .filter(|item| !item.is_none());
            let prior = priors
                .as_ref()
                .map(|items| &items[index])
                .filter(|item| !item.is_none());
            if matches!(&config, Config::Hurdle(_)) && (design.is_some() || prior.is_some()) {
                return Err("hurdle exog is unsupported: occurrence is static and severity has no exogenous features".into());
            }
            let config =
                match (design, prior) {
                    (Some(design), Some(prior)) => {
                        let design = cell_input(py, real_matrix(design, "exog"))?;
                        let prior = prior
                            .extract::<PyRef<'_, PyGaussianCoefficientPrior>>()
                            .map_err(|error| format!("invalid coefficient prior: {error}"))?
                            .inner
                            .clone();
                        config.with_regression(design, prior)?
                    }
                    (None, None) => config,
                    _ => return Err(
                        "exog and coefficient_priors must both be supplied for a regression cell"
                            .into(),
                    ),
                };
            let size = config.allocation_size(y.len())?;
            Ok((y, config, size))
        })();
        cells.push((ids[index].clone(), input));
    }
    reject_unused_schedule(all_ar, warmup, thin)?;
    check_total_retention(
        cells
            .iter()
            .filter_map(|(_, input)| input.as_ref().ok())
            .map(|(_, _, size)| *size),
    )?;
    let results = py.allow_threads(|| {
        run_cells(
            &cells,
            seed,
            threads,
            chunk_size,
            errors,
            |input, cell_seed| {
                let (y, config, _) = input.as_ref().map_err(Clone::clone)?;
                config.fit(y, cell_seed)
            },
        )
    })?;
    Ok(PyForecastBatchFit { ids, results })
}

/// Results preserve requested cell order; failed cells have None in `results`.
#[pyclass(name = "ForecastBatchFit", frozen, module = "rustmc")]
pub(crate) struct PyForecastBatchFit {
    ids: Vec<String>,
    results: Vec<Result<CellFit, String>>,
}
#[pymethods]
impl PyForecastBatchFit {
    #[getter]
    fn ids(&self) -> Vec<String> {
        self.ids.clone()
    }
    fn __len__(&self) -> usize {
        self.ids.len()
    }
    fn __getitem__(&self, py: Python<'_>, id: &str) -> PyResult<Py<PyAny>> {
        self.results[index_of(&self.ids, id)?]
            .as_ref()
            .map_err(|error| InferenceError::new_err(format!("cell {id:?}: {error}")))?
            .to_python(py)
    }
    #[getter]
    fn results(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        self.results
            .iter()
            .map(|result| match result {
                Ok(fit) => fit.to_python(py),
                Err(_) => Ok(py.None()),
            })
            .collect()
    }
    #[getter]
    fn errors<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        collected_errors(py, &self.ids, &self.results)
    }
    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let output = PyDict::new(py);
        for (id, result) in self.ids.iter().zip(&self.results) {
            if let Ok(fit) = result {
                output.set_item(
                    id,
                    forecast_diagnostics::diagnostics_list(py, &fit.report())?,
                )?;
            } else {
                output.set_item(id, py.None())?;
            }
        }
        Ok(output)
    }
    #[pyo3(signature = (steps, *, exog=None, seed=43, threads=1, chunk_size=64, errors="raise"))]
    #[allow(clippy::too_many_arguments)]
    fn forecast(
        &self,
        py: Python<'_>,
        steps: usize,
        exog: Option<&Bound<'_, PyAny>>,
        seed: u64,
        threads: usize,
        chunk_size: usize,
        errors: &str,
    ) -> PyResult<PyForecastBatchForecast> {
        check_errors(errors)?;
        let designs = aligned_optional(exog, self.ids.len(), "exog")?;
        let designs = (0..self.ids.len())
            .map(|index| {
                designs
                    .as_ref()
                    .map(|items| &items[index])
                    .filter(|item| !item.is_none())
                    .map(|item| cell_input(py, real_matrix(item, "future exog")))
                    .transpose()
            })
            .collect::<Vec<_>>();
        let cells = self
            .ids
            .iter()
            .zip(&self.results)
            .zip(&designs)
            .map(|((id, result), design)| {
                (
                    id.clone(),
                    (result, design, stable_cell_seed(seed, id, "forecast")),
                )
            })
            .collect::<Vec<_>>();
        check_total_retention(
            self.results
                .iter()
                .filter_map(|result| result.as_ref().ok())
                .filter_map(|fit| fit.forecast_allocation_size(steps).ok()),
        )?;
        let results = py.allow_threads(|| {
            run_cells(
                &cells,
                seed,
                threads,
                chunk_size,
                errors,
                |(result, design, forecast_seed), _| {
                    let fit = result.as_ref().map_err(Clone::clone)?;
                    let design = design.as_ref().map_err(Clone::clone)?;
                    fit.forecast(steps, *forecast_seed, design.as_deref())
                },
            )
        })?;
        Ok(PyForecastBatchForecast {
            ids: self.ids.clone(),
            results,
        })
    }
}
#[pyclass(name = "ForecastBatchForecast", frozen, module = "rustmc")]
pub(crate) struct PyForecastBatchForecast {
    ids: Vec<String>,
    results: Vec<Result<CellForecast, String>>,
}
#[pymethods]
impl PyForecastBatchForecast {
    #[getter]
    fn ids(&self) -> Vec<String> {
        self.ids.clone()
    }
    fn __len__(&self) -> usize {
        self.ids.len()
    }
    fn __getitem__(&self, py: Python<'_>, id: &str) -> PyResult<Py<PyAny>> {
        self.results[index_of(&self.ids, id)?]
            .as_ref()
            .map_err(|error| InferenceError::new_err(format!("cell {id:?}: {error}")))?
            .to_python(py)
    }
    #[getter]
    fn results(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        self.results
            .iter()
            .map(|result| match result {
                Ok(forecast) => forecast.to_python(py),
                Err(_) => Ok(py.None()),
            })
            .collect()
    }
    #[getter]
    fn errors<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        collected_errors(py, &self.ids, &self.results)
    }
}

/// Stable seed for reproducing one independent cell using the ordinary fit API.
#[pyfunction]
#[pyo3(signature = (seed, cell_id, domain="fit"))]
pub(crate) fn forecast_cell_seed(seed: u64, cell_id: &str, domain: &str) -> PyResult<u64> {
    if domain != "fit" && domain != "forecast" {
        return Err(PyValueError::new_err("domain must be 'fit' or 'forecast'"));
    }
    Ok(stable_cell_seed(seed, cell_id, domain))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(forecast_cell_seed, m)?)?;
    m.add_class::<PyForecastBatchFit>()?;
    m.add_class::<PyForecastBatchForecast>()?;
    Ok(())
}
