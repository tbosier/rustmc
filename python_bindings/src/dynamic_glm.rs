//! Python conversion for the native joint dynamic GLM kernel.
use crate::{arviz_from_groups, bayesian_forecast_error, forecast_diagnostics};
use ndarray::{Array2, Array4};
use numpy::{IntoPyArray, PyArray4};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rustmc_core::dynamic_glm::{
    self, Design, DynamicGlmConfig, DynamicGlmForecast, DynamicGlmPosterior, Family, Panel, Paths,
};

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDynamicGLM>()?;
    m.add_class::<PyDynamicGLMFit>()?;
    m.add_class::<PyDynamicGLMForecast>()?;
    m.add_function(wrap_pyfunction!(dynamic_poisson, m)?)?;
    m.add_function(wrap_pyfunction!(dynamic_negative_binomial, m)?)?;
    m.add_function(wrap_pyfunction!(dynamic_hurdle, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchical_regression, m)?)?;
    Ok(())
}

/// Latent-Gaussian dynamic panel model with block elliptical slice inference.
/// Each input keeps a group axis, including a single series: y=[[...]].
/// Scales and NB dispersion are fixed; regression coefficients and state paths
/// are inferred jointly. The design automatically includes an intercept.
#[pyclass(name = "BayesianDynamicGLM", module = "rustmc")]
#[derive(Clone)]
pub(crate) struct PyDynamicGLM {
    config: DynamicGlmConfig,
}

#[pymethods]
impl PyDynamicGLM {
    #[pyo3(signature=(steps, *, groups=1, exog=None, exposure=None, chains=1, draws=1000, seed=42))]
    #[allow(clippy::too_many_arguments)]
    fn prior_predictive(
        &self,
        py: Python<'_>,
        steps: usize,
        groups: usize,
        exog: Option<Design>,
        exposure: Option<Panel>,
        chains: usize,
        draws: usize,
        seed: u64,
    ) -> PyResult<PyDynamicGLMForecast> {
        let config = DynamicGlmConfig {
            chains,
            draws,
            seed,
            ..self.config.clone()
        };
        let inner = py
            .allow_threads(|| {
                dynamic_glm::prior_predictive(
                    &config,
                    groups,
                    steps,
                    exog.as_ref(),
                    exposure.as_ref(),
                )
            })
            .map_err(bayesian_forecast_error)?;
        Ok(PyDynamicGLMForecast { inner })
    }
    #[new]
    #[pyo3(signature=(family="poisson", *, initial_mean=0.0, occurrence_initial_mean=0.0, coefficient_sd=1.0, group_sd=0.5, process_sd=0.1, shared_process_sd=0.0, observation_sd=1.0, dispersion=5.0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        family: &str,
        initial_mean: f64,
        occurrence_initial_mean: f64,
        coefficient_sd: f64,
        group_sd: f64,
        process_sd: f64,
        shared_process_sd: f64,
        observation_sd: f64,
        dispersion: f64,
    ) -> PyResult<Self> {
        let family = match family {
            "poisson" => Family::Poisson,
            "negative_binomial" => Family::NegativeBinomial,
            "hurdle_lognormal" => Family::HurdleLogNormal,
            "gaussian" => Family::Gaussian,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "family must be poisson, negative_binomial, hurdle_lognormal, or gaussian",
                ))
            }
        };
        let config = DynamicGlmConfig {
            family,
            initial_mean,
            occurrence_initial_mean,
            coefficient_sd,
            group_sd,
            process_sd,
            shared_process_sd,
            observation_sd,
            dispersion,
            ..Default::default()
        };
        config.validate().map_err(bayesian_forecast_error)?;
        Ok(Self { config })
    }
    #[pyo3(signature=(y, *, exog=None, exposure=None, chains=4, draws=1000, warmup=1000, thin=1, seed=42))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        &self,
        py: Python<'_>,
        y: Panel,
        exog: Option<Design>,
        exposure: Option<Panel>,
        chains: usize,
        draws: usize,
        warmup: usize,
        thin: usize,
        seed: u64,
    ) -> PyResult<PyDynamicGLMFit> {
        let config = DynamicGlmConfig {
            chains,
            draws,
            warmup,
            thin,
            seed,
            ..self.config.clone()
        };
        let posterior = py
            .allow_threads(|| {
                dynamic_glm::fit_dynamic_glm(&y, exog.as_ref(), exposure.as_ref(), &config)
            })
            .map_err(bayesian_forecast_error)?;
        Ok(PyDynamicGLMFit {
            posterior,
            observations: y,
        })
    }
}

macro_rules! constructor {
    ($rust_name:ident, $python_name:literal, $family:literal) => {
        #[pyfunction(name=$python_name)]
        #[pyo3(signature=(*, initial_mean=0.0, occurrence_initial_mean=0.0, coefficient_sd=1.0, group_sd=0.5, process_sd=0.1, shared_process_sd=0.0, observation_sd=1.0, dispersion=5.0))]
        #[allow(clippy::too_many_arguments)]
        fn $rust_name(initial_mean: f64, occurrence_initial_mean: f64, coefficient_sd: f64, group_sd: f64, process_sd: f64, shared_process_sd: f64, observation_sd: f64, dispersion: f64) -> PyResult<PyDynamicGLM> {
            PyDynamicGLM::new($family, initial_mean, occurrence_initial_mean, coefficient_sd, group_sd, process_sd, shared_process_sd, observation_sd, dispersion)
        }
    };
}
constructor!(dynamic_poisson, "BayesianDynamicPoisson", "poisson");
constructor!(
    dynamic_negative_binomial,
    "BayesianDynamicNegativeBinomial",
    "negative_binomial"
);
constructor!(
    dynamic_hurdle,
    "BayesianDynamicHurdleLogNormal",
    "hurdle_lognormal"
);
constructor!(
    hierarchical_regression,
    "BayesianHierarchicalDynamicRegression",
    "gaussian"
);

#[pyclass(name = "DynamicGLMFit", module = "rustmc")]
#[derive(Clone)]
pub(crate) struct PyDynamicGLMFit {
    posterior: DynamicGlmPosterior,
    observations: Panel,
}
#[pymethods]
impl PyDynamicGLMFit {
    fn to_json(&self) -> PyResult<String> {
        self.posterior
            .to_json(&self.observations)
            .map_err(bayesian_forecast_error)
    }
    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        let (posterior, observations) =
            DynamicGlmPosterior::from_json(json).map_err(bayesian_forecast_error)?;
        Ok(Self {
            posterior,
            observations,
        })
    }
    #[getter]
    fn chains(&self) -> usize {
        self.posterior.chains.len()
    }
    #[getter]
    fn draws(&self) -> usize {
        self.posterior.config.draws
    }
    #[getter]
    fn groups(&self) -> usize {
        self.posterior.groups
    }
    #[getter]
    fn observed_count(&self) -> usize {
        self.posterior.observed_count
    }
    #[getter]
    fn param_names(&self) -> Vec<String> {
        self.posterior.parameter_names()
    }
    fn get_samples_2d<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let result = PyDict::new(py);
        let values = self.posterior.parameter_samples();
        for (j, name) in self.param_names().iter().enumerate() {
            let array =
                Array2::from_shape_fn((self.chains(), self.draws()), |(c, d)| values[c][d][j]);
            result.set_item(name, array.into_pyarray(py))?;
        }
        Ok(result)
    }
    fn get_samples<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let result = PyDict::new(py);
        let values = self.posterior.parameter_samples();
        for (j, name) in self.param_names().iter().enumerate() {
            let flat: Vec<_> = values.iter().flatten().map(|d| d[j]).collect();
            result.set_item(name, flat.into_pyarray(py))?;
        }
        Ok(result)
    }
    /// Training state paths [chain,draw,group,time]. Component 0 is severity or
    /// the count/Gaussian linear predictor; component 1 is hurdle occurrence.
    #[pyo3(signature=(component=0))]
    fn state_samples<'py>(
        &self,
        py: Python<'py>,
        component: usize,
    ) -> PyResult<Bound<'py, PyArray4<f64>>> {
        if component >= self.posterior.chains[0][0].states.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "invalid component index",
            ));
        }
        let paths = self
            .posterior
            .chains
            .iter()
            .map(|c| c.iter().map(|d| d.states[component].clone()).collect())
            .collect();
        Ok(path_array(py, &paths))
    }
    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        forecast_diagnostics::diagnostics_list(py, &self.posterior.diagnostics())
    }
    fn summary(&self) -> String {
        self.posterior.diagnostics().to_table_with_sampler(Some(
            "Sampler: block elliptical slice (latent Gaussian prior); scales and dispersion fixed",
        ))
    }
    #[getter]
    fn sampler_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let stats = forecast_diagnostics::sampler_stats(
            py,
            "block_elliptical_slice",
            self.chains(),
            self.draws(),
            "population and group coefficients and terminal states; not every historical state",
        )?;
        stats.set_item(
            "likelihood_evaluations",
            &self.posterior.likelihood_evaluations,
        )?;
        stats.set_item("likelihood_evaluations_include_warmup", true)?;
        let fixed = PyDict::new(py);
        let cfg = &self.posterior.config;
        for (name, value) in [
            ("coefficient_sd", cfg.coefficient_sd),
            ("group_sd", cfg.group_sd),
            ("process_sd", cfg.process_sd),
            ("shared_process_sd", cfg.shared_process_sd),
            ("observation_sd", cfg.observation_sd),
            ("dispersion", cfg.dispersion),
        ] {
            fixed.set_item(name, value)?;
        }
        stats.set_item("fixed_parameters", fixed)?;
        stats.set_item("joint_group_draws", true)?;
        Ok(stats)
    }
    #[pyo3(signature=(steps, *, exog=None, exposure=None, seed=43))]
    fn forecast(
        &self,
        py: Python<'_>,
        steps: usize,
        exog: Option<Design>,
        exposure: Option<Panel>,
        seed: u64,
    ) -> PyResult<PyDynamicGLMForecast> {
        let inner = py
            .allow_threads(|| {
                self.posterior
                    .forecast(steps, exog.as_ref(), exposure.as_ref(), seed)
            })
            .map_err(bayesian_forecast_error)?;
        Ok(PyDynamicGLMForecast { inner })
    }
    fn to_arviz<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let kwargs = PyDict::new(py);
        kwargs.set_item("posterior", self.get_samples_2d(py)?)?;
        let observed = PyDict::new(py);
        let y = Array2::from_shape_fn((self.groups(), self.posterior.time_count), |(g, t)| {
            self.observations[g][t]
        });
        observed.set_item("y", y.into_pyarray(py))?;
        kwargs.set_item("observed_data", observed)?;
        arviz_from_groups(&py.import("arviz")?, kwargs)
    }
}

#[pyclass(name = "DynamicGLMForecast", module = "rustmc")]
#[derive(Clone)]
pub(crate) struct PyDynamicGLMForecast {
    inner: DynamicGlmForecast,
}
#[pymethods]
impl PyDynamicGLMForecast {
    #[getter]
    fn chains(&self) -> usize {
        self.inner.mean_paths.len()
    }
    #[getter]
    fn draws(&self) -> usize {
        self.inner.mean_paths[0].len()
    }
    #[getter]
    fn groups(&self) -> usize {
        self.inner.mean_paths[0][0].len()
    }
    #[getter]
    fn steps(&self) -> usize {
        self.inner.mean_paths[0][0][0].len()
    }
    #[getter]
    fn mean<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
        path_means(py, &self.inner.mean_paths)
    }
    #[getter]
    fn observation_mean<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
        path_means(py, &self.inner.observation_paths)
    }
    #[pyo3(signature=(level=0.95))]
    fn interval<'py>(&self, py: Python<'py>, level: f64) -> PyResult<PanelInterval<'py>> {
        path_interval(py, &self.inner.observation_paths, level)
    }
    #[pyo3(signature=(level=0.95))]
    fn mean_interval<'py>(&self, py: Python<'py>, level: f64) -> PyResult<PanelInterval<'py>> {
        path_interval(py, &self.inner.mean_paths, level)
    }
    #[getter]
    fn interval_kind(&self) -> &'static str {
        "pointwise_equal_tailed"
    }
    #[getter]
    fn mean_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray4<f64>> {
        path_array(py, &self.inner.mean_paths)
    }
    #[getter]
    fn observation_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray4<f64>> {
        path_array(py, &self.inner.observation_paths)
    }
    #[getter]
    fn occurrence_samples<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray4<f64>>> {
        (!self.inner.occurrence_paths.is_empty())
            .then(|| path_array(py, &self.inner.occurrence_paths))
    }
    #[getter]
    fn positive_mean_samples<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray4<f64>>> {
        (!self.inner.positive_mean_paths.is_empty())
            .then(|| path_array(py, &self.inner.positive_mean_paths))
    }
    /// Sum groups within each aligned posterior draw, preserving dependence.
    #[getter]
    fn aggregate_observation_samples<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, numpy::PyArray3<f64>>> {
        let paths = &self.inner.observation_paths;
        let mut data = ndarray::Array3::zeros((paths.len(), paths[0].len(), paths[0][0][0].len()));
        for ((c, d, t), value) in data.indexed_iter_mut() {
            *value = paths[c][d].iter().map(|g| g[t]).sum::<f64>();
            if !value.is_finite() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "aggregate predictive values overflowed",
                ));
            }
        }
        Ok(data.into_pyarray(py))
    }
    #[getter]
    fn uncertainty_kind(&self) -> &'static str {
        "parameter_integrated_posterior_predictive_conditional_on_fixed_scales"
    }
}
fn path_array<'py>(py: Python<'py>, paths: &Paths) -> Bound<'py, PyArray4<f64>> {
    Array4::from_shape_fn(
        (
            paths.len(),
            paths[0].len(),
            paths[0][0].len(),
            paths[0][0][0].len(),
        ),
        |(c, d, g, t)| paths[c][d][g][t],
    )
    .into_pyarray(py)
}

type PanelInterval<'py> = (
    Bound<'py, numpy::PyArray2<f64>>,
    Bound<'py, numpy::PyArray2<f64>>,
);
fn group_paths(
    paths: &Paths,
    group: usize,
) -> rustmc_core::bayesian_forecast::PosteriorPredictiveForecast {
    rustmc_core::bayesian_forecast::PosteriorPredictiveForecast {
        state_paths: Vec::new(),
        observation_paths: paths
            .iter()
            .map(|c| c.iter().map(|d| d[group].clone()).collect())
            .collect(),
    }
}
fn path_means<'py>(py: Python<'py>, paths: &Paths) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
    let mut result = Array2::zeros((paths[0][0].len(), paths[0][0][0].len()));
    for g in 0..paths[0][0].len() {
        let values = group_paths(paths, g)
            .observation_means()
            .map_err(bayesian_forecast_error)?;
        for (t, value) in values.into_iter().enumerate() {
            result[(g, t)] = value;
        }
    }
    Ok(result.into_pyarray(py))
}
fn path_interval<'py>(py: Python<'py>, paths: &Paths, level: f64) -> PyResult<PanelInterval<'py>> {
    if !level.is_finite() || level <= 0. || level >= 1. {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "level must lie strictly between zero and one",
        ));
    }
    let mut lower = Array2::zeros((paths[0][0].len(), paths[0][0][0].len()));
    let mut upper = lower.clone();
    for g in 0..paths[0][0].len() {
        let quantiles = group_paths(paths, g)
            .observation_quantiles(&[(1. - level) / 2., (1. + level) / 2.])
            .map_err(bayesian_forecast_error)?;
        for t in 0..paths[0][0][0].len() {
            lower[(g, t)] = quantiles[0].values[t];
            upper[(g, t)] = quantiles[1].values[t];
        }
    }
    Ok((lower.into_pyarray(py), upper.into_pyarray(py)))
}
