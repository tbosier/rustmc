//! Native independent-cell forecasting bindings. One pool spans cells and chains.
use super::*;
use rustmc_core::diagnostics::DiagnosticsReport;
use rustmc_core::forecast_batch::{
    execute_batch, execute_batch_fail_fast, stable_cell_seed, BatchError,
};

#[derive(Clone)]
pub(crate) enum Config {
    Local(CoreBayesianLocalLevelConfig),
    Seasonal(CoreBayesianSeasonalLocalLevelConfig),
    Trend(CoreBayesianLocalLinearTrendConfig),
    Ar(CoreBayesianArConfig),
}
#[derive(Clone)]
enum CellFit {
    Local(PyBayesianLocalLevelFit),
    Seasonal(PyBayesianSeasonalLocalLevelFit),
    Trend(PyBayesianLocalLinearTrendFit),
    Ar(PyBayesianArFit),
}
#[derive(Clone)]
enum CellForecast {
    Local(CorePosteriorPredictiveForecast),
    Seasonal(CoreSeasonalPosteriorPredictiveForecast),
    Trend(CoreTrendPosteriorPredictiveForecast),
    Ar(CoreBayesianArForecast),
}
impl Config {
    fn fit(&self, observations: &[f64], seed: u64) -> Result<CellFit, String> {
        match self {
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
    fn to_python(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self {
            Self::Local(fit) => Ok(Py::new(py, fit.clone())?.into_any()),
            Self::Seasonal(fit) => Ok(Py::new(py, fit.clone())?.into_any()),
            Self::Trend(fit) => Ok(Py::new(py, fit.clone())?.into_any()),
            Self::Ar(fit) => Ok(Py::new(py, fit.clone())?.into_any()),
        }
    }
    fn report(&self) -> DiagnosticsReport {
        match self {
            Self::Local(fit) => fit.posterior.diagnostics(),
            Self::Seasonal(fit) => fit.posterior.diagnostics(),
            Self::Trend(fit) => fit.posterior.diagnostics(),
            Self::Ar(fit) => fit.posterior.diagnostics(),
        }
    }
    fn forecast(&self, steps: usize, seed: u64) -> Result<CellForecast, String> {
        match self {
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

#[allow(clippy::too_many_arguments)]
pub(crate) fn fit_batch(
    py: Python<'_>,
    observations: &Bound<'_, PyAny>,
    ids: Vec<String>,
    models: Option<&Bound<'_, PyAny>>,
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
    let models = models
        .map(|values| values.try_iter()?.collect::<PyResult<Vec<_>>>())
        .transpose()?;
    if models
        .as_ref()
        .is_some_and(|values| values.len() != ids.len())
    {
        return Err(PyValueError::new_err("models must have one entry per cell"));
    }
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
            Ok((y, config))
        })();
        cells.push((ids[index].clone(), input));
    }
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
    #[pyo3(signature = (steps, *, seed=43, threads=1, chunk_size=64, errors="raise"))]
    fn forecast(
        &self,
        py: Python<'_>,
        steps: usize,
        seed: u64,
        threads: usize,
        chunk_size: usize,
        errors: &str,
    ) -> PyResult<PyForecastBatchForecast> {
        check_errors(errors)?;
        let cells = self
            .ids
            .iter()
            .zip(&self.results)
            .map(|(id, result)| (id.clone(), (result, stable_cell_seed(seed, id, "forecast"))))
            .collect::<Vec<_>>();
        let results = py.allow_threads(|| {
            run_cells(
                &cells,
                seed,
                threads,
                chunk_size,
                errors,
                |(result, forecast_seed), _| {
                    let fit = result.as_ref().map_err(Clone::clone)?;
                    fit.forecast(steps, *forecast_seed)
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
