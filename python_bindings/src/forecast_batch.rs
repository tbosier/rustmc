//! Native independent-cell forecasting bindings. One pool spans cells and chains.
use super::regression::{
    PyBayesianRegressionFit, PyBayesianRegressionForecast, PyGaussianCoefficientPrior,
};
use super::*;
use rustmc_core::bayesian_regression::{
    fit_regression, GaussianCoefficientPrior, RegressionConfig, RegressionForecast,
};
use rustmc_core::diagnostics::DiagnosticsReport;
use rustmc_core::forecast_batch::{
    execute_batch, execute_batch_fail_fast, stable_cell_seed, BatchError,
};

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
    Regression(Box<RegressionConfig>, Vec<Vec<f64>>),
}
#[derive(Clone)]
enum CellFit {
    Local(PyBayesianLocalLevelFit),
    Seasonal(PyBayesianSeasonalLocalLevelFit),
    Trend(PyBayesianLocalLinearTrendFit),
    Ar(PyBayesianArFit),
    Regression(Box<PyBayesianRegressionFit>),
}
#[derive(Clone)]
enum CellForecast {
    Local(CorePosteriorPredictiveForecast),
    Seasonal(CoreSeasonalPosteriorPredictiveForecast),
    Trend(CoreTrendPosteriorPredictiveForecast),
    Ar(CoreBayesianArForecast),
    Regression(RegressionForecast, bool, usize),
}
impl Config {
    fn allocation_size(&self, history: usize) -> Result<usize, String> {
        let (chains, draws, parameters, dimension) = match self {
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
            Self::Regression(base, design) => {
                let mut config = (**base).clone();
                config.seed = seed;
                let posterior = fit_regression(observations, design, &config)
                    .map_err(|error| error.to_string())?;
                Ok(CellFit::Regression(Box::new(PyBayesianRegressionFit {
                    posterior,
                    observations: observations.to_vec(),
                })))
            }
            Self::Local(base) => {
                let mut config = base.clone();
                config.seed = seed;
                let posterior =
                    fit_bayesian_local_level(observations, &config).map_err(|e| e.to_string())?;
                Ok(CellFit::Local(PyBayesianLocalLevelFit {
                    posterior,
                    observations: observations.to_vec(),
                    config,
                }))
            }
            Self::Seasonal(base) => {
                let mut config = base.clone();
                config.seed = seed;
                let posterior = fit_bayesian_seasonal_local_level(observations, &config)
                    .map_err(|e| e.to_string())?;
                Ok(CellFit::Seasonal(PyBayesianSeasonalLocalLevelFit {
                    posterior,
                    observations: observations.to_vec(),
                    config,
                }))
            }
            Self::Trend(base) => {
                let mut config = base.clone();
                config.seed = seed;
                let posterior = fit_bayesian_local_linear_trend(observations, &config)
                    .map_err(|e| e.to_string())?;
                Ok(CellFit::Trend(PyBayesianLocalLinearTrendFit {
                    posterior,
                    observations: observations.to_vec(),
                    config,
                }))
            }
            Self::Ar(base) => {
                let mut config = base.clone();
                config.seed = seed;
                let posterior =
                    fit_bayesian_ar(observations, &config).map_err(|e| e.to_string())?;
                Ok(CellFit::Ar(PyBayesianArFit {
                    posterior,
                    observations: observations.to_vec(),
                    config,
                }))
            }
        }
    }
}
impl CellFit {
    fn forecast_allocation_size(&self, steps: usize) -> Result<usize, String> {
        let (chains, draws, components) = match self {
            Self::Local(fit) => (fit.chains(), fit.draws(), 3),
            Self::Seasonal(fit) => (fit.chains(), fit.draws(), 4),
            Self::Trend(fit) => (fit.chains(), fit.draws(), 4),
            Self::Ar(fit) => (fit.chains(), fit.draws(), 3),
            Self::Regression(fit) => (
                fit.posterior.chains.len(),
                fit.posterior.chains.first().map_or(0, Vec::len),
                6,
            ),
        };
        allocation_product(&[chains, draws, steps, components])
    }
    fn to_python(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self {
            Self::Regression(fit) => Ok(Py::new(py, (**fit).clone())?.into_any()),
            Self::Local(fit) => Ok(Py::new(py, fit.clone())?.into_any()),
            Self::Seasonal(fit) => Ok(Py::new(py, fit.clone())?.into_any()),
            Self::Trend(fit) => Ok(Py::new(py, fit.clone())?.into_any()),
            Self::Ar(fit) => Ok(Py::new(py, fit.clone())?.into_any()),
        }
    }
    fn report(&self) -> DiagnosticsReport {
        match self {
            Self::Regression(fit) => fit.posterior.diagnostics(),
            Self::Local(fit) => fit.posterior.diagnostics(),
            Self::Seasonal(fit) => fit.posterior.diagnostics(),
            Self::Trend(fit) => fit.posterior.diagnostics(),
            Self::Ar(fit) => fit.posterior.diagnostics(),
        }
    }
    fn forecast(
        &self,
        steps: usize,
        seed: u64,
        exog: Option<&[Vec<f64>]>,
    ) -> Result<CellForecast, String> {
        self.forecast_allocation_size(steps)?;
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
                            inner,
                            fit.posterior.config.seasonal,
                            fit.posterior.config.structural_model.dimension(),
                        )
                    })
                    .map_err(|error| error.to_string())
            }
            Self::Local(fit) => fit
                .posterior
                .forecast(steps, seed)
                .map(CellForecast::Local)
                .map_err(|e| e.to_string()),
            Self::Seasonal(fit) => fit
                .posterior
                .forecast(steps, seed)
                .map(CellForecast::Seasonal)
                .map_err(|e| e.to_string()),
            Self::Trend(fit) => fit
                .posterior
                .forecast(steps, seed)
                .map(CellForecast::Trend)
                .map_err(|e| e.to_string()),
            Self::Ar(fit) => fit
                .posterior
                .forecast(steps, seed)
                .map(CellForecast::Ar)
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
                    StateSpaceError::new_err(format!("cell {id:?}: {error}"))
                }
            })
    } else {
        execute_batch(cells, seed, threads, chunk_size, fit).map_err(PyValueError::new_err)
    }
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
    observations: &Bound<'_, PyAny>,
    ids: Vec<String>,
    models: Option<&Bound<'_, PyAny>>,
    exog: Option<&Bound<'_, PyAny>>,
    coefficient_priors: Option<&Bound<'_, PyAny>>,
    default: Config,
    chains: usize,
    draws: usize,
    warmup: usize,
    thin: usize,
    seed: u64,
    threads: usize,
    chunk_size: usize,
    errors: &str,
) -> PyResult<PyForecastBatchFit> {
    check_errors(errors)?;
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
    for (index, row) in rows.iter().enumerate() {
        let input = (|| -> Result<(Vec<f64>, Config), String> {
            let y = row
                .extract::<Vec<f64>>()
                .map_err(|e| format!("invalid observations: {e}"))?;
            let config = if let Some(models) = &models {
                let model = &models[index];
                if model.is_none() {
                    default.clone()
                } else if let Ok(model) = model.extract::<PyRef<'_, PyBayesianLocalLevel>>() {
                    model.batch_config(chains, draws, warmup, thin)
                } else if let Ok(model) = model.extract::<PyRef<'_, PyBayesianSeasonalLocalLevel>>()
                {
                    model.batch_config(chains, draws, warmup, thin)
                } else if let Ok(model) = model.extract::<PyRef<'_, PyBayesianLocalLinearTrend>>() {
                    model.batch_config(chains, draws, warmup, thin)
                } else if let Ok(model) = model.extract::<PyRef<'_, PyBayesianAutoRegression>>() {
                    model.batch_config(chains, draws, warmup, thin)
                } else {
                    return Err(
                        "invalid configuration: models entries must be forecasting models or None"
                            .into(),
                    );
                }
            } else {
                default.clone()
            };
            let design = designs
                .as_ref()
                .map(|items| &items[index])
                .filter(|item| !item.is_none());
            let prior = priors
                .as_ref()
                .map(|items| &items[index])
                .filter(|item| !item.is_none());
            let config =
                match (design, prior) {
                    (Some(design), Some(prior)) => {
                        let design = design
                            .extract::<Vec<Vec<f64>>>()
                            .map_err(|error| format!("invalid exog: {error}"))?;
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
            config.allocation_size(y.len())?;
            Ok((y, config))
        })();
        cells.push((ids[index].clone(), input));
    }
    check_total_retention(
        cells
            .iter()
            .filter_map(|(_, input)| input.as_ref().ok())
            .filter_map(|(y, config)| config.allocation_size(y.len()).ok()),
    )?;
    let results = py.allow_threads(|| {
        run_cells(
            &cells,
            seed,
            threads,
            chunk_size,
            errors,
            |input, cell_seed| {
                let (y, config) = input.as_ref().map_err(Clone::clone)?;
                config.fit(y, cell_seed)
            },
        )
    })?;
    if errors == "raise" {
        for (id, result) in ids.iter().zip(&results) {
            if let Err(error) = result {
                return Err(StateSpaceError::new_err(format!("cell {id:?}: {error}")));
            }
        }
    }
    Ok(PyForecastBatchFit { ids, results })
}

/// Results preserve requested cell order; failed cells have None in `results`.
#[pyclass(name = "ForecastBatchFit")]
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
            .map_err(|error| StateSpaceError::new_err(format!("cell {id:?}: {error}")))?
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
                    .map(|item| {
                        item.extract::<Vec<Vec<f64>>>()
                            .map_err(|error| format!("invalid future exog: {error}"))
                    })
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
        if errors == "raise" {
            for (id, result) in self.ids.iter().zip(&results) {
                if let Err(error) = result {
                    return Err(StateSpaceError::new_err(format!("cell {id:?}: {error}")));
                }
            }
        }
        Ok(PyForecastBatchForecast {
            ids: self.ids.clone(),
            results,
        })
    }
}
#[pyclass(name = "ForecastBatchForecast")]
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
            .map_err(|error| StateSpaceError::new_err(format!("cell {id:?}: {error}")))?
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
