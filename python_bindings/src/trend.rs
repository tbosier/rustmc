//! Bayesian local-linear-trend bindings.
use crate::forecast_support::*;
use crate::StateSpaceError;
use crate::{arviz_from_groups, forecast_batch, forecast_diagnostics, regression};
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rustmc_core::bayesian_forecast::InverseGammaPrior as CoreInverseGammaPrior;
use rustmc_core::bayesian_trend::{
    fit_bayesian_local_linear_trend,
    BayesianLocalLinearTrendConfig as CoreBayesianLocalLinearTrendConfig,
    LocalLinearTrendPosterior as CoreLocalLinearTrendPosterior,
    LocalLinearTrendPosteriorDraw as CoreLocalLinearTrendPosteriorDraw,
    TrendPosteriorPredictiveForecast as CoreTrendPosteriorPredictiveForecast,
};
use rustmc_core::state_space::LinearGaussianStateSpace as CoreLinearGaussianStateSpace;

pub(crate) fn trend_parameter_array<'py, F>(
    py: Python<'py>,
    posterior: &CoreLocalLinearTrendPosterior,
    value: F,
) -> Bound<'py, PyArray2<f64>>
where
    F: Fn(&CoreLocalLinearTrendPosteriorDraw) -> f64,
{
    let chains = posterior.chains.len();
    let draws = posterior.chains.first().map_or(0, Vec::len);
    Array2::from_shape_fn((chains, draws), |(chain, draw)| {
        value(&posterior.chains[chain][draw])
    })
    .into_pyarray(py)
}

/// Bayesian local-linear-trend model with stochastic level and slope.
#[pyclass(name = "BayesianLocalLinearTrend", frozen, module = "rustmc")]
#[derive(Clone)]
pub(crate) struct PyBayesianLocalLinearTrend {
    pub(crate) initial_mean: [f64; 2],
    pub(crate) initial_covariance: [f64; 4],
    pub(crate) level_variance_prior: CoreInverseGammaPrior,
    pub(crate) slope_variance_prior: CoreInverseGammaPrior,
    pub(crate) observation_variance_prior: CoreInverseGammaPrior,
}

#[pymethods]
impl PyBayesianLocalLinearTrend {
    /// Fit independent ragged cells on one bounded native worker pool.
    #[pyo3(signature = (observations, ids, *, models=None, exog=None, coefficient_priors=None, chains=4, draws=1000, warmup=500, thin=1, seed=42, threads=1, chunk_size=64, errors="raise"))]
    #[allow(clippy::too_many_arguments)]
    fn fit_batch(
        &self,
        py: Python<'_>,
        observations: &Bound<'_, PyAny>,
        ids: Vec<String>,
        models: Option<&Bound<'_, PyAny>>,
        exog: Option<&Bound<'_, PyAny>>,
        coefficient_priors: Option<&Bound<'_, PyAny>>,
        chains: usize,
        draws: usize,
        warmup: usize,
        thin: usize,
        seed: u64,
        threads: usize,
        chunk_size: usize,
        errors: &str,
    ) -> PyResult<forecast_batch::PyForecastBatchFit> {
        forecast_batch::fit_batch(
            py,
            observations,
            ids,
            models,
            exog,
            coefficient_priors,
            self.batch_config(chains, draws, warmup, thin),
            chains,
            draws,
            warmup,
            thin,
            seed,
            threads,
            chunk_size,
            errors,
        )
    }

    #[new]
    #[pyo3(signature = (
        level_variance_prior,
        slope_variance_prior,
        observation_variance_prior,
        initial_level=0.0,
        initial_slope=0.0,
        initial_level_variance=100.0,
        initial_slope_variance=10.0,
        initial_level_slope_covariance=0.0
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        level_variance_prior: PyRef<'_, PyInverseGammaPrior>,
        slope_variance_prior: PyRef<'_, PyInverseGammaPrior>,
        observation_variance_prior: PyRef<'_, PyInverseGammaPrior>,
        initial_level: f64,
        initial_slope: f64,
        initial_level_variance: f64,
        initial_slope_variance: f64,
        initial_level_slope_covariance: f64,
    ) -> PyResult<Self> {
        if !initial_level.is_finite() || !initial_slope.is_finite() {
            return Err(StateSpaceError::new_err(
                "invalid configuration: initial level and slope must be finite",
            ));
        }
        if !initial_level_variance.is_finite()
            || initial_level_variance <= 0.0
            || !initial_slope_variance.is_finite()
            || initial_slope_variance <= 0.0
            || !initial_level_slope_covariance.is_finite()
        {
            return Err(StateSpaceError::new_err(
                "invalid configuration: initial variances must be finite and positive and covariance must be finite",
            ));
        }
        let level_scale = initial_level_variance.sqrt();
        let scaled_covariance = initial_level_slope_covariance / level_scale;
        let slope_remainder = initial_slope_variance - scaled_covariance * scaled_covariance;
        if !slope_remainder.is_finite() || slope_remainder <= 0.0 {
            return Err(StateSpaceError::new_err(
                "invalid configuration: initial state covariance must be positive definite",
            ));
        }
        Ok(Self {
            initial_mean: [initial_level, initial_slope],
            initial_covariance: [
                initial_level_variance,
                initial_level_slope_covariance,
                initial_level_slope_covariance,
                initial_slope_variance,
            ],
            level_variance_prior: level_variance_prior.inner,
            slope_variance_prior: slope_variance_prior.inner,
            observation_variance_prior: observation_variance_prior.inner,
        })
    }

    #[getter]
    fn initial_level(&self) -> f64 {
        self.initial_mean[0]
    }

    #[getter]
    fn initial_slope(&self) -> f64 {
        self.initial_mean[1]
    }

    #[getter]
    fn initial_covariance<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        Array2::from_shape_fn((2, 2), |(row, column)| {
            self.initial_covariance[row * 2 + column]
        })
        .into_pyarray(py)
    }

    #[getter]
    fn level_variance_prior(&self) -> PyInverseGammaPrior {
        PyInverseGammaPrior {
            inner: self.level_variance_prior,
        }
    }

    #[getter]
    fn slope_variance_prior(&self) -> PyInverseGammaPrior {
        PyInverseGammaPrior {
            inner: self.slope_variance_prior,
        }
    }

    #[getter]
    fn observation_variance_prior(&self) -> PyInverseGammaPrior {
        PyInverseGammaPrior {
            inner: self.observation_variance_prior,
        }
    }

    #[pyo3(signature = (observations, chains=4, draws=1000, warmup=500, thin=1, seed=42, *, exog=None, coefficient_prior=None))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        &self,
        py: Python<'_>,
        observations: PyReadonlyArray1<'_, f64>,
        chains: usize,
        draws: usize,
        warmup: usize,
        thin: usize,
        seed: u64,
        exog: Option<PyReadonlyArray2<'_, f64>>,
        coefficient_prior: Option<PyRef<'_, regression::PyGaussianCoefficientPrior>>,
    ) -> PyResult<PyObject> {
        let observations = state_space_vector(observations);
        if let Some(exog) = exog {
            let config = regression::config(
                CoreLinearGaussianStateSpace::new(
                    2,
                    vec![1.0, 1.0, 0.0, 1.0],
                    vec![1.0, 0.0],
                    vec![1.0, 0.0, 0.0, 1.0],
                    1.0,
                    self.initial_mean.to_vec(),
                    self.initial_covariance.to_vec(),
                )
                .map_err(state_space_error)?,
                vec![self.level_variance_prior, self.slope_variance_prior],
                vec!["level_variance", "slope_variance"],
                self.observation_variance_prior,
                false,
                (chains, draws, warmup, thin, seed),
            );
            return regression::fit(py, observations, exog, coefficient_prior, config);
        }
        if coefficient_prior.is_some() {
            return Err(StateSpaceError::new_err("coefficient_prior requires exog"));
        }
        let config = CoreBayesianLocalLinearTrendConfig {
            initial_mean: self.initial_mean,
            initial_covariance: self.initial_covariance,
            level_variance_prior: self.level_variance_prior,
            slope_variance_prior: self.slope_variance_prior,
            observation_variance_prior: self.observation_variance_prior,
            num_chains: chains,
            num_warmup: warmup,
            num_draws: draws,
            thinning: thin,
            seed,
        };
        let posterior = py
            .allow_threads(|| fit_bayesian_local_linear_trend(&observations, &config))
            .map_err(bayesian_forecast_error)?;
        Ok(Py::new(
            py,
            PyBayesianLocalLinearTrendFit {
                posterior,
                observations,
                config,
            },
        )?
        .into_any())
    }

    fn __repr__(&self) -> String {
        format!(
            "BayesianLocalLinearTrend(initial_level={}, initial_slope={})",
            self.initial_mean[0], self.initial_mean[1],
        )
    }
}

#[pyclass(name = "BayesianLocalLinearTrendFit", module = "rustmc")]
#[derive(Clone)]
pub(crate) struct PyBayesianLocalLinearTrendFit {
    pub(crate) posterior: CoreLocalLinearTrendPosterior,
    pub(crate) observations: Vec<f64>,
    pub(crate) config: CoreBayesianLocalLinearTrendConfig,
}

#[pymethods]
impl PyBayesianLocalLinearTrendFit {
    /// Rank-normalized folded split R-hat, bulk/tail ESS, MCSE and HDIs.
    fn summary(&self) -> String {
        self.posterior.diagnostics().to_table_with_sampler(Some(
            "Sampler: conjugate Gibbs/FFBS; acceptance and divergences unavailable",
        ))
    }

    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        forecast_diagnostics::diagnostics_list(py, &self.posterior.diagnostics())
    }

    #[getter]
    fn sampler_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        forecast_diagnostics::sampler_stats(
            py,
            "conjugate Gibbs/FFBS",
            self.chains(),
            self.draws(),
            "variance parameters, terminal level and slope; historical states are not retained",
        )
    }

    #[getter]
    pub(crate) fn chains(&self) -> usize {
        self.posterior.chains.len()
    }

    #[getter]
    pub(crate) fn draws(&self) -> usize {
        self.posterior.chains.first().map_or(0, Vec::len)
    }

    #[getter]
    fn time_count(&self) -> usize {
        self.observations.len()
    }

    #[getter]
    fn observed_count(&self) -> usize {
        self.observations
            .iter()
            .filter(|value| !value.is_nan())
            .count()
    }

    #[getter]
    fn warmup(&self) -> usize {
        self.config.num_warmup
    }

    #[getter]
    fn thin(&self) -> usize {
        self.config.thinning
    }

    fn get_samples_2d<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let samples = PyDict::new(py);
        for (name, values) in [
            (
                "level_variance",
                trend_parameter_array(py, &self.posterior, |draw| draw.level_variance),
            ),
            (
                "slope_variance",
                trend_parameter_array(py, &self.posterior, |draw| draw.slope_variance),
            ),
            (
                "observation_variance",
                trend_parameter_array(py, &self.posterior, |draw| draw.observation_variance),
            ),
            (
                "level_sd",
                trend_parameter_array(py, &self.posterior, |draw| draw.level_variance.sqrt()),
            ),
            (
                "slope_sd",
                trend_parameter_array(py, &self.posterior, |draw| draw.slope_variance.sqrt()),
            ),
            (
                "observation_sd",
                trend_parameter_array(py, &self.posterior, |draw| draw.observation_variance.sqrt()),
            ),
            (
                "terminal_level",
                trend_parameter_array(py, &self.posterior, |draw| draw.terminal_level),
            ),
            (
                "terminal_slope",
                trend_parameter_array(py, &self.posterior, |draw| draw.terminal_slope),
            ),
        ] {
            samples.set_item(name, values)?;
        }
        Ok(samples)
    }

    #[pyo3(signature = (steps, seed=43))]
    fn forecast(
        &self,
        py: Python<'_>,
        steps: usize,
        seed: u64,
    ) -> PyResult<PyBayesianTrendForecast> {
        let forecast = py
            .allow_threads(|| self.posterior.forecast(steps, seed))
            .map_err(bayesian_forecast_error)?;
        Ok(PyBayesianTrendForecast { inner: forecast })
    }

    fn to_arviz<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let az = py.import("arviz")?;
        let groups = PyDict::new(py);
        groups.set_item("posterior", self.get_samples_2d(py)?)?;
        let observed = PyDict::new(py);
        observed.set_item("y", PyArray1::from_vec(py, self.observations.clone()))?;
        groups.set_item("observed_data", observed)?;
        arviz_from_groups(&az, groups)
    }

    fn __repr__(&self) -> String {
        format!(
            "BayesianLocalLinearTrendFit(chains={}, draws={}, time_count={}, observed_count={})",
            self.chains(),
            self.draws(),
            self.time_count(),
            self.observed_count(),
        )
    }
}

#[pyclass(name = "BayesianTrendForecast", module = "rustmc")]
pub(crate) struct PyBayesianTrendForecast {
    pub(crate) inner: CoreTrendPosteriorPredictiveForecast,
}

#[pymethods]
impl PyBayesianTrendForecast {
    #[getter]
    fn chains(&self) -> usize {
        self.inner.observation_paths.len()
    }

    #[getter]
    fn draws(&self) -> usize {
        self.inner.observation_paths.first().map_or(0, Vec::len)
    }

    #[getter]
    fn steps(&self) -> usize {
        self.inner.horizon()
    }

    #[getter]
    fn level_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        local_level_path_array(py, &self.inner.level_paths)
    }

    #[getter]
    fn slope_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        local_level_path_array(py, &self.inner.slope_paths)
    }

    #[getter]
    fn observation_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        local_level_path_array(py, &self.inner.observation_paths)
    }

    #[getter]
    fn level_mean<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self
            .inner
            .level_means()
            .map_err(bayesian_forecast_error)?
            .into_pyarray(py))
    }

    #[getter]
    fn slope_mean<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self
            .inner
            .slope_means()
            .map_err(bayesian_forecast_error)?
            .into_pyarray(py))
    }

    #[getter]
    fn observation_mean<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self
            .inner
            .observation_means()
            .map_err(bayesian_forecast_error)?
            .into_pyarray(py))
    }

    fn level_quantile<'py>(
        &self,
        py: Python<'py>,
        probability: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let quantiles = self
            .inner
            .level_quantiles(&[probability])
            .map_err(bayesian_forecast_error)?;
        Ok(quantiles[0].values.clone().into_pyarray(py))
    }

    fn slope_quantile<'py>(
        &self,
        py: Python<'py>,
        probability: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let quantiles = self
            .inner
            .slope_quantiles(&[probability])
            .map_err(bayesian_forecast_error)?;
        Ok(quantiles[0].values.clone().into_pyarray(py))
    }

    fn observation_quantile<'py>(
        &self,
        py: Python<'py>,
        probability: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let quantiles = self
            .inner
            .observation_quantiles(&[probability])
            .map_err(bayesian_forecast_error)?;
        Ok(quantiles[0].values.clone().into_pyarray(py))
    }

    #[pyo3(signature = (level=0.95))]
    fn level_interval<'py>(&self, py: Python<'py>, level: f64) -> PyResult<PyIntervalArrays<'py>> {
        validate_interval_level(level)?;
        let probabilities = [(1.0 - level) / 2.0, (1.0 + level) / 2.0];
        let quantiles = self
            .inner
            .level_quantiles(&probabilities)
            .map_err(bayesian_forecast_error)?;
        Ok((
            quantiles[0].values.clone().into_pyarray(py),
            quantiles[1].values.clone().into_pyarray(py),
        ))
    }

    #[pyo3(signature = (level=0.95))]
    fn slope_interval<'py>(&self, py: Python<'py>, level: f64) -> PyResult<PyIntervalArrays<'py>> {
        validate_interval_level(level)?;
        let probabilities = [(1.0 - level) / 2.0, (1.0 + level) / 2.0];
        let quantiles = self
            .inner
            .slope_quantiles(&probabilities)
            .map_err(bayesian_forecast_error)?;
        Ok((
            quantiles[0].values.clone().into_pyarray(py),
            quantiles[1].values.clone().into_pyarray(py),
        ))
    }

    #[pyo3(signature = (level=0.95))]
    fn interval<'py>(&self, py: Python<'py>, level: f64) -> PyResult<PyIntervalArrays<'py>> {
        validate_interval_level(level)?;
        let probabilities = [(1.0 - level) / 2.0, (1.0 + level) / 2.0];
        let quantiles = self
            .inner
            .observation_quantiles(&probabilities)
            .map_err(bayesian_forecast_error)?;
        Ok((
            quantiles[0].values.clone().into_pyarray(py),
            quantiles[1].values.clone().into_pyarray(py),
        ))
    }

    #[getter]
    fn uncertainty_kind(&self) -> &'static str {
        "parameter_integrated_posterior_predictive"
    }

    #[getter]
    fn interval_kind(&self) -> &'static str {
        "pointwise_equal_tailed"
    }

    fn __repr__(&self) -> String {
        format!(
            "BayesianTrendForecast(chains={}, draws={}, steps={})",
            self.chains(),
            self.draws(),
            self.steps(),
        )
    }
}

impl PyBayesianLocalLinearTrend {
    pub(crate) fn batch_config(
        &self,
        chains: usize,
        draws: usize,
        warmup: usize,
        thin: usize,
    ) -> forecast_batch::Config {
        let seed = 0;

        forecast_batch::Config::Trend(CoreBayesianLocalLinearTrendConfig {
            initial_mean: self.initial_mean,
            initial_covariance: self.initial_covariance,
            level_variance_prior: self.level_variance_prior,
            slope_variance_prior: self.slope_variance_prior,
            observation_variance_prior: self.observation_variance_prior,
            num_chains: chains,
            num_warmup: warmup,
            num_draws: draws,
            thinning: thin,
            seed,
        })
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBayesianLocalLinearTrend>()?;
    m.add_class::<PyBayesianLocalLinearTrendFit>()?;
    m.add_class::<PyBayesianTrendForecast>()?;
    Ok(())
}
