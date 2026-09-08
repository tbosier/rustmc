type IntervalArrays<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>);
use super::*;
use rustmc_core::bayesian_regression::{
    self as core, GaussianCoefficientPrior, RegressionConfig, RegressionForecast,
    RegressionPosterior,
};

#[pyclass(name = "GaussianCoefficientPrior", frozen)]
#[derive(Clone)]
pub(crate) struct PyGaussianCoefficientPrior {
    pub(crate) inner: GaussianCoefficientPrior,
}
#[pymethods]
impl PyGaussianCoefficientPrior {
    #[new]
    fn new(
        mean: PyReadonlyArray1<'_, f64>,
        covariance: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Self> {
        let mean = state_space_vector(mean);
        let (covariance, p) = state_space_matrix("coefficient covariance", covariance)?;
        if mean.len() != p || p == 0 {
            return Err(StateSpaceError::new_err(
                "coefficient mean and covariance dimensions must agree and be nonempty",
            ));
        }
        CoreLinearGaussianStateSpace::local_level(1.0, 1.0, 0.0, 1.0)
            .map_err(state_space_error)?
            .with_static_regression(&[], &mean, &covariance)
            .map_err(state_space_error)?;
        Ok(Self {
            inner: GaussianCoefficientPrior { mean, covariance },
        })
    }
    #[getter]
    fn mean<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.mean.clone().into_pyarray(py)
    }
    #[getter]
    fn covariance<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let p = self.inner.mean.len();
        Array2::from_shape_fn((p, p), |(i, j)| self.inner.covariance[i * p + j]).into_pyarray(py)
    }
}
pub(crate) fn rows(array: PyReadonlyArray2<'_, f64>) -> Vec<Vec<f64>> {
    array
        .as_array()
        .rows()
        .into_iter()
        .map(|r| r.iter().copied().collect())
        .collect()
}

#[pyfunction]
#[pyo3(signature=(count, period, harmonics, start=0))]
fn fourier_design<'py>(
    py: Python<'py>,
    count: usize,
    period: usize,
    harmonics: usize,
    start: i64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let rows = core::fourier_design(count, period, harmonics, start).map_err(state_space_error)?;
    let p = 2 * harmonics - usize::from(2 * harmonics == period);
    Ok(Array2::from_shape_fn((count, p), |(i, j)| rows[i][j]).into_pyarray(py))
}

pub(crate) fn fit(
    py: Python<'_>,
    observations: Vec<f64>,
    exog: PyReadonlyArray2<'_, f64>,
    prior: Option<PyRef<'_, PyGaussianCoefficientPrior>>,
    mut config: RegressionConfig,
) -> PyResult<PyObject> {
    config.coefficient_prior = prior
        .ok_or_else(|| {
            StateSpaceError::new_err(
                "exog requires an explicit GaussianCoefficientPrior via coefficient_prior",
            )
        })?
        .inner
        .clone();
    let design = rows(exog);
    let posterior = py
        .allow_threads(|| core::fit_regression(&observations, &design, &config))
        .map_err(state_space_error)?;
    Ok(Py::new(
        py,
        PyBayesianRegressionFit {
            posterior,
            observations,
        },
    )?
    .into_any())
}
pub(crate) fn config(
    structural_model: CoreLinearGaussianStateSpace,
    variance_priors: Vec<CoreInverseGammaPrior>,
    variance_names: Vec<&str>,
    observation_variance_prior: CoreInverseGammaPrior,
    seasonal: bool,
    sampling: (usize, usize, usize, usize, u64),
) -> RegressionConfig {
    let (num_chains, num_draws, num_warmup, thinning, seed) = sampling;
    RegressionConfig {
        innovation_indices: (0..variance_priors.len()).collect(),
        structural_model,
        variance_priors,
        variance_names: variance_names.into_iter().map(str::to_string).collect(),
        observation_variance_prior,
        coefficient_prior: GaussianCoefficientPrior {
            mean: vec![],
            covariance: vec![],
        },
        num_chains,
        num_draws,
        num_warmup,
        thinning,
        seed,
        seasonal,
    }
}

#[pyclass(name = "BayesianRegressionFit")]
#[derive(Clone)]
pub(crate) struct PyBayesianRegressionFit {
    pub(crate) posterior: RegressionPosterior,
    pub(crate) observations: Vec<f64>,
}
#[pymethods]
impl PyBayesianRegressionFit {
    fn summary(&self) -> String {
        self.posterior.diagnostics().to_table_with_sampler(Some(
            "Sampler: joint conjugate Gibbs/FFBS; acceptance and divergences unavailable",
        ))
    }
    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        forecast_diagnostics::diagnostics_list(py, &self.posterior.diagnostics())
    }
    #[getter]
    fn sampler_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        forecast_diagnostics::sampler_stats(py, "joint conjugate Gibbs/FFBS", self.chains(), self.draws(),
            "all variance parameters, regression coefficients and terminal structural states; historical states are not retained")
    }
    #[getter]
    fn chains(&self) -> usize {
        self.posterior.chains.len()
    }
    #[getter]
    fn draws(&self) -> usize {
        self.posterior.chains[0].len()
    }
    #[getter]
    fn time_count(&self) -> usize {
        self.observations.len()
    }
    #[getter]
    fn observed_count(&self) -> usize {
        self.observations.iter().filter(|v| v.is_finite()).count()
    }
    #[getter]
    fn warmup(&self) -> usize {
        self.posterior.config.num_warmup
    }
    #[getter]
    fn thin(&self) -> usize {
        self.posterior.config.thinning
    }
    fn get_samples_2d<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let result = PyDict::new(py);
        for (i, name) in self.posterior.config.variance_names.iter().enumerate() {
            result.set_item(
                name,
                Array2::from_shape_fn((self.chains(), self.draws()), |(c, d)| {
                    self.posterior.chains[c][d].variances[i]
                })
                .into_pyarray(py),
            )?;
        }
        result.set_item(
            "observation_variance",
            Array2::from_shape_fn((self.chains(), self.draws()), |(c, d)| {
                self.posterior.chains[c][d].observation_variance
            })
            .into_pyarray(py),
        )?;
        let p = self.posterior.config.coefficient_prior.mean.len();
        result.set_item(
            "coefficients",
            Array3::from_shape_fn((self.chains(), self.draws(), p), |(c, d, p)| {
                self.posterior.chains[c][d].coefficients[p]
            })
            .into_pyarray(py),
        )?;
        let n = self.posterior.config.structural_model.dimension();
        result.set_item(
            "terminal_state",
            Array3::from_shape_fn((self.chains(), self.draws(), n), |(c, d, p)| {
                self.posterior.chains[c][d].terminal_state[p]
            })
            .into_pyarray(py),
        )?;
        Ok(result)
    }
    #[pyo3(signature=(steps, seed=43, *, exog=None))]
    fn forecast(
        &self,
        py: Python<'_>,
        steps: usize,
        seed: u64,
        exog: Option<PyReadonlyArray2<'_, f64>>,
    ) -> PyResult<PyBayesianRegressionForecast> {
        let design = rows(exog.ok_or_else(|| {
            StateSpaceError::new_err("future exog is required for regression forecasts")
        })?);
        if design.len() != steps {
            return Err(StateSpaceError::new_err(
                "future exog row count must equal steps",
            ));
        }
        let inner = py
            .allow_threads(|| self.posterior.forecast(&design, seed))
            .map_err(state_space_error)?;
        Ok(PyBayesianRegressionForecast {
            inner,
            seasonal: self.posterior.config.seasonal,
            dimension: self.posterior.config.structural_model.dimension(),
        })
    }
    fn to_arviz<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let az = py.import("arviz")?;
        let groups = PyDict::new(py);
        groups.set_item("posterior", self.get_samples_2d(py)?)?;
        let observed = PyDict::new(py);
        observed.set_item("y", self.observations.clone().into_pyarray(py))?;
        groups.set_item("observed_data", observed)?;
        arviz_from_groups(&az, groups)
    }
}

#[pyclass(name = "BayesianRegressionForecast")]
pub(crate) struct PyBayesianRegressionForecast {
    pub(crate) inner: RegressionForecast,
    pub(crate) seasonal: bool,
    pub(crate) dimension: usize,
}
#[pymethods]
impl PyBayesianRegressionForecast {
    #[getter]
    fn chains(&self) -> usize {
        self.inner.observation_paths.len()
    }
    #[getter]
    fn draws(&self) -> usize {
        self.inner.observation_paths[0].len()
    }
    #[getter]
    fn steps(&self) -> usize {
        self.inner.observation_paths[0][0].len()
    }
    #[getter]
    fn observation_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        local_level_path_array(py, &self.inner.observation_paths)
    }
    #[getter]
    fn cumulative_observation_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        local_level_path_array(py, &self.inner.cumulative_observation_paths)
    }
    #[getter]
    fn state_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        local_level_path_array(py, &self.inner.level_paths)
    }
    #[getter]
    fn level_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        self.state_samples(py)
    }
    #[getter]
    fn seasonal_samples<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<f64>>> {
        if !self.seasonal {
            return Err(PyValueError::new_err(
                "model has no stochastic seasonal component",
            ));
        }
        Ok(local_level_path_array(py, &self.inner.secondary_paths))
    }
    #[getter]
    fn slope_samples<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<f64>>> {
        if self.seasonal || self.dimension != 2 {
            return Err(PyValueError::new_err("model has no slope component"));
        }
        Ok(local_level_path_array(py, &self.inner.secondary_paths))
    }
    #[getter]
    fn regression_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        local_level_path_array(py, &self.inner.regression_paths)
    }
    #[getter]
    fn mean_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        local_level_path_array(py, &self.inner.mean_paths)
    }
    #[getter]
    fn observation_mean<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        means(&self.inner.observation_paths).into_pyarray(py)
    }
    #[getter]
    fn cumulative_observation_mean<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        means(&self.inner.cumulative_observation_paths).into_pyarray(py)
    }
    #[pyo3(signature=(probability=0.9))]
    fn observation_interval<'py>(
        &self,
        py: Python<'py>,
        probability: f64,
    ) -> PyResult<IntervalArrays<'py>> {
        interval(py, &self.inner.observation_paths, probability)
    }
    #[pyo3(signature=(probability=0.9))]
    fn cumulative_observation_interval<'py>(
        &self,
        py: Python<'py>,
        probability: f64,
    ) -> PyResult<IntervalArrays<'py>> {
        interval(py, &self.inner.cumulative_observation_paths, probability)
    }
    #[pyo3(signature=(level=0.95))]
    fn interval<'py>(&self, py: Python<'py>, level: f64) -> PyResult<IntervalArrays<'py>> {
        if !level.is_finite() || level <= 0.0 || level >= 1.0 {
            return Err(PyValueError::new_err(
                "level must be strictly between zero and one",
            ));
        }
        interval(py, &self.inner.observation_paths, level)
    }
    #[pyo3(signature=(level=0.95))]
    fn cumulative_interval<'py>(
        &self,
        py: Python<'py>,
        level: f64,
    ) -> PyResult<IntervalArrays<'py>> {
        if !level.is_finite() || level <= 0.0 || level >= 1.0 {
            return Err(PyValueError::new_err(
                "level must be strictly between zero and one",
            ));
        }
        interval(py, &self.inner.cumulative_observation_paths, level)
    }
    #[getter]
    fn uncertainty_kind(&self) -> &'static str {
        "parameter_integrated_posterior_predictive"
    }
    #[getter]
    fn interval_kind(&self) -> &'static str {
        "pointwise_equal_tailed"
    }
}
fn means(paths: &core::Paths) -> Vec<f64> {
    (0..paths[0][0].len())
        .map(|i| {
            paths.iter().flatten().map(|p| p[i]).sum::<f64>()
                / (paths.len() * paths[0].len()) as f64
        })
        .collect()
}
fn interval<'py>(
    py: Python<'py>,
    paths: &core::Paths,
    probability: f64,
) -> PyResult<IntervalArrays<'py>> {
    validate_probability(probability)?;
    let mut lower = vec![];
    let mut upper = vec![];
    for i in 0..paths[0][0].len() {
        let mut values: Vec<f64> = paths.iter().flatten().map(|p| p[i]).collect();
        values.sort_by(f64::total_cmp);
        let quantile = |p: f64| {
            let index = p * (values.len() - 1) as f64;
            let lo = index.floor() as usize;
            let hi = index.ceil() as usize;
            values[lo] * (1.0 - index.fract()) + values[hi] * index.fract()
        };
        lower.push(quantile((1.0 - probability) / 2.0));
        upper.push(quantile((1.0 + probability) / 2.0));
    }
    Ok((lower.into_pyarray(py), upper.into_pyarray(py)))
}
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGaussianCoefficientPrior>()?;
    m.add_class::<PyBayesianRegressionFit>()?;
    m.add_class::<PyBayesianRegressionForecast>()?;
    m.add_function(wrap_pyfunction!(fourier_design, m)?)?;
    Ok(())
}
