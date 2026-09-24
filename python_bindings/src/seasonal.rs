//! Bayesian seasonal local-level bindings.
use crate::forecast_support::*;
use crate::StateSpaceError;
use crate::{arviz_from_groups, forecast_batch, forecast_diagnostics, regression};
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rustmc_core::bayesian_forecast::InverseGammaPrior as CoreInverseGammaPrior;
use rustmc_core::bayesian_seasonal::{
    fit_bayesian_seasonal_local_level,
    BayesianSeasonalLocalLevelConfig as CoreBayesianSeasonalLocalLevelConfig,
    SeasonalLocalLevelPosterior as CoreSeasonalLocalLevelPosterior,
    SeasonalLocalLevelPosteriorDraw as CoreSeasonalLocalLevelPosteriorDraw,
    SeasonalPosteriorPredictiveForecast as CoreSeasonalPosteriorPredictiveForecast,
};
use rustmc_core::state_space::LinearGaussianStateSpace as CoreLinearGaussianStateSpace;

pub(crate) fn seasonal_parameter_array<'py, F>(
    py: Python<'py>,
    posterior: &CoreSeasonalLocalLevelPosterior,
    value: F,
) -> Bound<'py, PyArray2<f64>>
where
    F: Fn(&CoreSeasonalLocalLevelPosteriorDraw) -> f64,
{
    let chains = posterior.chains.len();
    let draws = posterior.chains.first().map_or(0, Vec::len);
    Array2::from_shape_fn((chains, draws), |(chain, draw)| {
        value(&posterior.chains[chain][draw])
    })
    .into_pyarray(py)
}

/// Bayesian structural seasonal local-level model using conjugate Gibbs/FFBS.
#[pyclass(name = "BayesianSeasonalLocalLevel", frozen, module = "rustmc")]
#[derive(Clone)]
pub(crate) struct PyBayesianSeasonalLocalLevel {
    pub(crate) period: usize,
    pub(crate) initial_level: f64,
    pub(crate) initial_seasonal_effects: Vec<f64>,
    pub(crate) initial_level_variance: f64,
    pub(crate) initial_seasonal_variance: f64,
    pub(crate) level_variance_prior: CoreInverseGammaPrior,
    pub(crate) seasonal_variance_prior: CoreInverseGammaPrior,
    pub(crate) observation_variance_prior: CoreInverseGammaPrior,
}

#[pymethods]
impl PyBayesianSeasonalLocalLevel {
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
    #[pyo3(signature = (period, level_variance_prior, seasonal_variance_prior, observation_variance_prior, initial_level=0.0, initial_seasonal_effects=None, initial_level_variance=100.0, initial_seasonal_variance=10.0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        period: usize,
        level_variance_prior: PyRef<'_, PyInverseGammaPrior>,
        seasonal_variance_prior: PyRef<'_, PyInverseGammaPrior>,
        observation_variance_prior: PyRef<'_, PyInverseGammaPrior>,
        initial_level: f64,
        initial_seasonal_effects: Option<Vec<f64>>,
        initial_level_variance: f64,
        initial_seasonal_variance: f64,
    ) -> PyResult<Self> {
        let effects = initial_seasonal_effects.unwrap_or_else(|| vec![0.0; period]);
        // Reuse the fixed structural constructor for immediate shape,
        // sum-to-zero, and covariance validation.
        CoreLinearGaussianStateSpace::seasonal_local_level(
            period,
            level_variance_prior.inner.scale / (level_variance_prior.inner.shape + 1.0),
            seasonal_variance_prior.inner.scale / (seasonal_variance_prior.inner.shape + 1.0),
            observation_variance_prior.inner.scale / (observation_variance_prior.inner.shape + 1.0),
            initial_level,
            effects.clone(),
            initial_level_variance,
            initial_seasonal_variance,
        )
        .map_err(state_space_error)?;
        Ok(Self {
            period,
            initial_level,
            initial_seasonal_effects: effects,
            initial_level_variance,
            initial_seasonal_variance,
            level_variance_prior: level_variance_prior.inner,
            seasonal_variance_prior: seasonal_variance_prior.inner,
            observation_variance_prior: observation_variance_prior.inner,
        })
    }

    #[getter]
    fn period(&self) -> usize {
        self.period
    }

    #[getter]
    fn initial_seasonal_effects<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.initial_seasonal_effects.clone().into_pyarray(py)
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
                CoreLinearGaussianStateSpace::seasonal_local_level(
                    self.period,
                    1.0,
                    1.0,
                    1.0,
                    self.initial_level,
                    self.initial_seasonal_effects.clone(),
                    self.initial_level_variance,
                    self.initial_seasonal_variance,
                )
                .map_err(state_space_error)?,
                vec![self.level_variance_prior, self.seasonal_variance_prior],
                vec!["level_variance", "seasonal_variance"],
                self.observation_variance_prior,
                true,
                (chains, draws, warmup, thin, seed),
            );
            return regression::fit(py, observations, exog, coefficient_prior, config);
        }
        if coefficient_prior.is_some() {
            return Err(StateSpaceError::new_err("coefficient_prior requires exog"));
        }
        let config = CoreBayesianSeasonalLocalLevelConfig {
            period: self.period,
            initial_level: self.initial_level,
            initial_seasonal_effects: self.initial_seasonal_effects.clone(),
            initial_level_variance: self.initial_level_variance,
            initial_seasonal_variance: self.initial_seasonal_variance,
            level_variance_prior: self.level_variance_prior,
            seasonal_variance_prior: self.seasonal_variance_prior,
            observation_variance_prior: self.observation_variance_prior,
            num_chains: chains,
            num_warmup: warmup,
            num_draws: draws,
            thinning: thin,
            seed,
        };
        let posterior = py
            .allow_threads(|| fit_bayesian_seasonal_local_level(&observations, &config))
            .map_err(bayesian_forecast_error)?;
        Ok(Py::new(
            py,
            PyBayesianSeasonalLocalLevelFit {
                posterior,
                observations,
                config,
            },
        )?
        .into_any())
    }

    fn __repr__(&self) -> String {
        format!(
            "BayesianSeasonalLocalLevel(period={}, initial_level={})",
            self.period, self.initial_level
        )
    }
}

#[pyclass(name = "BayesianSeasonalLocalLevelFit", module = "rustmc")]
#[derive(Clone)]
pub(crate) struct PyBayesianSeasonalLocalLevelFit {
    pub(crate) posterior: CoreSeasonalLocalLevelPosterior,
    pub(crate) observations: Vec<f64>,
    pub(crate) config: CoreBayesianSeasonalLocalLevelConfig,
}

#[pymethods]
impl PyBayesianSeasonalLocalLevelFit {
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
        forecast_diagnostics::sampler_stats(py, "conjugate Gibbs/FFBS", self.chains(), self.draws(), "variance parameters and all terminal seasonal state coordinates; historical states are not retained")
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
    fn period(&self) -> usize {
        self.posterior.period
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
                seasonal_parameter_array(py, &self.posterior, |draw| draw.level_variance),
            ),
            (
                "seasonal_variance",
                seasonal_parameter_array(py, &self.posterior, |draw| draw.seasonal_variance),
            ),
            (
                "observation_variance",
                seasonal_parameter_array(py, &self.posterior, |draw| draw.observation_variance),
            ),
            (
                "level_sd",
                seasonal_parameter_array(py, &self.posterior, |draw| draw.level_variance.sqrt()),
            ),
            (
                "seasonal_sd",
                seasonal_parameter_array(py, &self.posterior, |draw| draw.seasonal_variance.sqrt()),
            ),
            (
                "observation_sd",
                seasonal_parameter_array(py, &self.posterior, |draw| {
                    draw.observation_variance.sqrt()
                }),
            ),
            (
                "terminal_level",
                seasonal_parameter_array(py, &self.posterior, |draw| draw.terminal_state[0]),
            ),
            (
                "terminal_seasonal",
                seasonal_parameter_array(py, &self.posterior, |draw| draw.terminal_state[1]),
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
    ) -> PyResult<PyBayesianSeasonalForecast> {
        let inner = py
            .allow_threads(|| self.posterior.forecast(steps, seed))
            .map_err(bayesian_forecast_error)?;
        Ok(PyBayesianSeasonalForecast { inner })
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
            "BayesianSeasonalLocalLevelFit(period={}, chains={}, draws={}, time_count={}, observed_count={})",
            self.period(), self.chains(), self.draws(), self.time_count(), self.observed_count()
        )
    }
}

#[pyclass(name = "BayesianSeasonalForecast", module = "rustmc")]
pub(crate) struct PyBayesianSeasonalForecast {
    pub(crate) inner: CoreSeasonalPosteriorPredictiveForecast,
}

#[pymethods]
impl PyBayesianSeasonalForecast {
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
    fn seasonal_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        local_level_path_array(py, &self.inner.seasonal_paths)
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
    fn level_mean<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self
            .inner
            .level_means()
            .map_err(bayesian_forecast_error)?
            .into_pyarray(py))
    }
    #[getter]
    fn seasonal_mean<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self
            .inner
            .seasonal_means()
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
    #[getter]
    fn cumulative_observation_mean<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self
            .inner
            .cumulative_observation_means()
            .map_err(bayesian_forecast_error)?
            .into_pyarray(py))
    }

    fn level_quantile<'py>(
        &self,
        py: Python<'py>,
        probability: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self
            .inner
            .level_quantiles(&[probability])
            .map_err(bayesian_forecast_error)?[0]
            .values
            .clone()
            .into_pyarray(py))
    }
    fn seasonal_quantile<'py>(
        &self,
        py: Python<'py>,
        probability: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self
            .inner
            .seasonal_quantiles(&[probability])
            .map_err(bayesian_forecast_error)?[0]
            .values
            .clone()
            .into_pyarray(py))
    }
    fn observation_quantile<'py>(
        &self,
        py: Python<'py>,
        probability: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self
            .inner
            .observation_quantiles(&[probability])
            .map_err(bayesian_forecast_error)?[0]
            .values
            .clone()
            .into_pyarray(py))
    }
    fn cumulative_observation_quantile<'py>(
        &self,
        py: Python<'py>,
        probability: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self
            .inner
            .cumulative_observation_quantiles(&[probability])
            .map_err(bayesian_forecast_error)?[0]
            .values
            .clone()
            .into_pyarray(py))
    }

    #[pyo3(signature = (level=0.95))]
    fn interval<'py>(&self, py: Python<'py>, level: f64) -> PyResult<PyIntervalArrays<'py>> {
        seasonal_interval(py, level, |probabilities| {
            self.inner.observation_quantiles(probabilities)
        })
    }
    #[pyo3(signature = (level=0.95))]
    fn cumulative_interval<'py>(
        &self,
        py: Python<'py>,
        level: f64,
    ) -> PyResult<PyIntervalArrays<'py>> {
        seasonal_interval(py, level, |probabilities| {
            self.inner.cumulative_observation_quantiles(probabilities)
        })
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
            "BayesianSeasonalForecast(chains={}, draws={}, steps={})",
            self.chains(),
            self.draws(),
            self.steps()
        )
    }
}

pub(crate) fn seasonal_interval<'py, F>(
    py: Python<'py>,
    level: f64,
    quantiles: F,
) -> PyResult<PyIntervalArrays<'py>>
where
    F: FnOnce(
        &[f64],
    ) -> Result<
        Vec<rustmc_core::bayesian_forecast::ForecastQuantile>,
        rustmc_core::bayesian_forecast::BayesianForecastError,
    >,
{
    validate_interval_level(level)?;
    let probabilities = [(1.0 - level) / 2.0, (1.0 + level) / 2.0];
    let values = quantiles(&probabilities).map_err(bayesian_forecast_error)?;
    Ok((
        values[0].values.clone().into_pyarray(py),
        values[1].values.clone().into_pyarray(py),
    ))
}

impl PyBayesianSeasonalLocalLevel {
    pub(crate) fn batch_config(
        &self,
        chains: usize,
        draws: usize,
        warmup: usize,
        thin: usize,
    ) -> forecast_batch::Config {
        let seed = 0;

        forecast_batch::Config::Seasonal(CoreBayesianSeasonalLocalLevelConfig {
            period: self.period,
            initial_level: self.initial_level,
            initial_seasonal_effects: self.initial_seasonal_effects.clone(),
            initial_level_variance: self.initial_level_variance,
            initial_seasonal_variance: self.initial_seasonal_variance,
            level_variance_prior: self.level_variance_prior,
            seasonal_variance_prior: self.seasonal_variance_prior,
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
    m.add_class::<PyBayesianSeasonalLocalLevel>()?;
    m.add_class::<PyBayesianSeasonalLocalLevelFit>()?;
    m.add_class::<PyBayesianSeasonalForecast>()?;
    Ok(())
}
