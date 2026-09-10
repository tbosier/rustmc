use super::*;
use rustmc_core::structural::{
    self as core, Component, SamplingConfig, StructuralConfig, StructuralPaths,
    StructuralPosterior, VarianceParameter,
};

/// An explicit fixed variance or independent inverse-gamma prior (shape, scale).
#[pyclass(name = "VarianceParameter", frozen)]
#[derive(Clone)]
pub(crate) struct PyVarianceParameter {
    inner: VarianceParameter,
}
#[pymethods]
impl PyVarianceParameter {
    #[staticmethod]
    fn fixed(value: f64) -> PyResult<Self> {
        let inner = VarianceParameter::Fixed(value);
        inner.validate().map_err(state_space_error)?;
        Ok(Self { inner })
    }
    #[staticmethod]
    fn inverse_gamma(shape: f64, scale: f64) -> PyResult<Self> {
        let inner = VarianceParameter::InverseGamma { shape, scale };
        inner.validate().map_err(state_space_error)?;
        Ok(Self { inner })
    }
}
/// A named additive component with a proper initial Gaussian state prior.
#[pyclass(name = "StructuralComponent", frozen)]
#[derive(Clone)]
pub(crate) struct PyStructuralComponent {
    inner: Component,
}
fn component(inner: Component) -> PyResult<PyStructuralComponent> {
    StructuralConfig {
        components: vec![inner.clone()],
        observation_variance: VarianceParameter::Fixed(1.0),
        student_df: None,
    }
    .validate()
    .map_err(state_space_error)?;
    Ok(PyStructuralComponent { inner })
}
#[pymethods]
impl PyStructuralComponent {
    #[staticmethod]
    fn level(
        name: String,
        innovation: PyRef<'_, PyVarianceParameter>,
        initial_mean: f64,
        initial_variance: f64,
    ) -> PyResult<Self> {
        component(Component::level(
            name,
            innovation.inner.clone(),
            initial_mean,
            initial_variance,
        ))
    }
    #[staticmethod]
    #[pyo3(signature=(name,level_innovation,slope_innovation,initial_mean,initial_covariance,damping=1.0))]
    fn trend(
        name: String,
        level_innovation: PyRef<'_, PyVarianceParameter>,
        slope_innovation: PyRef<'_, PyVarianceParameter>,
        initial_mean: Vec<f64>,
        initial_covariance: Vec<Vec<f64>>,
        damping: f64,
    ) -> PyResult<Self> {
        let covariance = matrix(initial_covariance, initial_mean.len())?;
        component(
            Component::trend(
                name,
                damping,
                level_innovation.inner.clone(),
                slope_innovation.inner.clone(),
                initial_mean,
                covariance,
            )
            .map_err(state_space_error)?,
        )
    }
    #[staticmethod]
    fn seasonal(
        name: String,
        period: f64,
        harmonics: usize,
        innovation: PyRef<'_, PyVarianceParameter>,
        initial_variance: f64,
    ) -> PyResult<Self> {
        component(
            Component::seasonal(
                name,
                period,
                harmonics,
                innovation.inner.clone(),
                initial_variance,
            )
            .map_err(state_space_error)?,
        )
    }
    /// innovations=None makes all coefficients static; provide one variance per
    /// coefficient to allow dynamic coefficients. Exog columns follow component order.
    #[staticmethod]
    #[pyo3(signature=(name,initial_mean,initial_covariance,innovations=None))]
    fn regression(
        name: String,
        initial_mean: Vec<f64>,
        initial_covariance: Vec<Vec<f64>>,
        innovations: Option<Vec<PyRef<'_, PyVarianceParameter>>>,
    ) -> PyResult<Self> {
        let covariance = matrix(initial_covariance, initial_mean.len())?;
        let q = innovations
            .map(|v| v.iter().map(|q| q.inner.clone()).collect())
            .unwrap_or_else(|| vec![VarianceParameter::Fixed(0.0); initial_mean.len()]);
        component(Component::regression(name, initial_mean, covariance, q))
    }
    #[staticmethod]
    fn ar(
        name: String,
        coefficients: Vec<f64>,
        innovation: PyRef<'_, PyVarianceParameter>,
        initial_mean: Vec<f64>,
        initial_covariance: Vec<Vec<f64>>,
    ) -> PyResult<Self> {
        let covariance = matrix(initial_covariance, initial_mean.len())?;
        component(
            Component::ar(
                name,
                coefficients,
                innovation.inner.clone(),
                initial_mean,
                covariance,
            )
            .map_err(state_space_error)?,
        )
    }
    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }
}
fn matrix(rows: Vec<Vec<f64>>, d: usize) -> PyResult<Vec<f64>> {
    if rows.len() != d || rows.iter().any(|r| r.len() != d) {
        return Err(StateSpaceError::new_err(
            "initial covariance must be square and match initial mean",
        ));
    }
    Ok(rows.into_iter().flatten().collect())
}
#[pyclass(name = "StructuralModel", frozen)]
#[derive(Clone)]
pub(crate) struct PyStructuralModel {
    inner: StructuralConfig,
}
#[pymethods]
impl PyStructuralModel {
    #[new]
    #[pyo3(signature=(components,observation_variance,student_df=None))]
    fn new(
        components: Vec<PyRef<'_, PyStructuralComponent>>,
        observation_variance: PyRef<'_, PyVarianceParameter>,
        student_df: Option<f64>,
    ) -> PyResult<Self> {
        let inner = StructuralConfig {
            components: components.iter().map(|c| c.inner.clone()).collect(),
            observation_variance: observation_variance.inner.clone(),
            student_df,
        };
        inner.validate().map_err(state_space_error)?;
        Ok(Self { inner })
    }
    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(state_space_error)
    }
    #[staticmethod]
    fn from_json(value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: StructuralConfig::from_json(value).map_err(state_space_error)?,
        })
    }
    #[getter]
    fn component_names(&self) -> Vec<String> {
        self.inner
            .components
            .iter()
            .map(|c| c.name.clone())
            .collect()
    }
    #[getter]
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }
    #[pyo3(signature=(observations,*,exog=None,chains=4,draws=1000,warmup=500,thin=1,seed=42,store_states=false))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        &self,
        py: Python<'_>,
        observations: Vec<f64>,
        exog: Option<Vec<Vec<f64>>>,
        chains: usize,
        draws: usize,
        warmup: usize,
        thin: usize,
        seed: u64,
        store_states: bool,
    ) -> PyResult<PyStructuralFit> {
        let config = SamplingConfig {
            chains,
            draws,
            warmup,
            thinning: thin,
            seed,
            store_states,
        };
        let inner = py
            .allow_threads(|| core::fit(&observations, exog.as_deref(), &self.inner, &config))
            .map_err(state_space_error)?;
        Ok(PyStructuralFit { inner })
    }
    #[pyo3(signature=(steps,*,exog=None,draws=1000,seed=43))]
    fn prior_predict(
        &self,
        py: Python<'_>,
        steps: usize,
        exog: Option<Vec<Vec<f64>>>,
        draws: usize,
        seed: u64,
    ) -> PyResult<PyStructuralForecast> {
        let inner = py
            .allow_threads(|| {
                self.inner
                    .prior_predict(steps, exog.as_deref(), draws, seed)
            })
            .map_err(state_space_error)?;
        Ok(PyStructuralForecast {
            inner,
            names: self.component_names(),
        })
    }
}
#[pyclass(name = "StructuralFit", frozen)]
pub(crate) struct PyStructuralFit {
    inner: StructuralPosterior,
}
#[pymethods]
impl PyStructuralFit {
    #[getter]
    fn chains(&self) -> usize {
        self.inner.chains.len()
    }
    #[getter]
    fn draws(&self) -> usize {
        self.inner.chains[0].len()
    }
    #[getter]
    fn param_names(&self) -> Vec<String> {
        self.inner.parameter_names()
    }
    fn get_samples_2d<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let output = PyDict::new(py);
        let samples = self.inner.parameter_samples();
        for (j, name) in self.param_names().iter().enumerate() {
            output.set_item(
                name,
                Array2::from_shape_fn((self.chains(), self.draws()), |(c, d)| samples[c][d][j])
                    .into_pyarray(py),
            )?;
        }
        Ok(output)
    }
    fn get_samples<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let output = PyDict::new(py);
        let samples = self.inner.parameter_samples();
        for (j, name) in self.param_names().iter().enumerate() {
            let flat: Vec<_> = samples.iter().flatten().map(|d| d[j]).collect();
            output.set_item(name, flat.into_pyarray(py))?;
        }
        Ok(output)
    }
    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        forecast_diagnostics::diagnostics_list(py, &self.inner.diagnostics())
    }
    fn summary(&self) -> String {
        self.inner.diagnostics().to_table_with_sampler(Some(
            if self.inner.config.student_df.is_some() {
                "Sampler: Gibbs/FFBS with Student-t Gamma precision updates"
            } else {
                "Sampler: conjugate Gibbs/FFBS"
            },
        ))
    }
    #[getter]
    fn sampler_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let name = if self.inner.config.student_df.is_some() {
            "gibbs_ffbs_student_t"
        } else {
            "gibbs_ffbs_structural"
        };
        let stats = forecast_diagnostics::sampler_stats(py, name, self.chains(), self.draws(), "innovation/noise variances and terminal states; not every historical state or Student-t precision")?;
        stats.set_item("student_df", self.inner.config.student_df)?;
        Ok(stats)
    }
    #[getter]
    fn model(&self) -> PyStructuralModel {
        PyStructuralModel {
            inner: self.inner.config.clone(),
        }
    }
    #[getter]
    fn component_names(&self) -> Vec<String> {
        self.inner
            .config
            .components
            .iter()
            .map(|c| c.name.clone())
            .collect()
    }
    #[getter]
    fn variance_names(&self) -> Vec<String> {
        let mut names = self.inner.config.variance_names();
        names.push("observation_variance".into());
        names
    }
    /// Variance draws, shape (chain, draw, state_dimension+1).
    #[getter]
    fn variance_draws<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        let c = self.inner.chains.len();
        let n = self.inner.chains[0].len();
        let d = self.inner.config.dimension();
        Array3::from_shape_fn((c, n, d + 1), |(i, j, k)| {
            if k == d {
                self.inner.chains[i][j].observation_variance
            } else {
                self.inner.chains[i][j].variances[k]
            }
        })
        .into_pyarray(py)
    }
    #[getter]
    fn terminal_states<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        let c = self.inner.chains.len();
        let n = self.inner.chains[0].len();
        let d = self.inner.config.dimension();
        Array3::from_shape_fn((c, n, d), |(i, j, k)| {
            self.inner.chains[i][j].terminal_state[k]
        })
        .into_pyarray(py)
    }
    /// Historical states include x[-1] at index zero.
    #[getter]
    fn states<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray4<f64>>> {
        let states = self
            .inner
            .chains
            .iter()
            .map(|c| {
                c.iter()
                    .map(|d| {
                        d.states
                            .clone()
                            .ok_or_else(|| StateSpaceError::new_err("fit with store_states=True"))
                    })
                    .collect::<PyResult<Vec<_>>>()
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(array4(py, &states))
    }
    #[getter]
    fn historical_components<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray4<f64>>> {
        Ok(array4(
            py,
            &self
                .inner
                .historical_components()
                .map_err(state_space_error)?,
        ))
    }
    #[pyo3(signature=(steps,*,exog=None,seed=43))]
    fn forecast(
        &self,
        py: Python<'_>,
        steps: usize,
        exog: Option<Vec<Vec<f64>>>,
        seed: u64,
    ) -> PyResult<PyStructuralForecast> {
        let inner = py
            .allow_threads(|| self.inner.forecast(steps, exog.as_deref(), seed))
            .map_err(state_space_error)?;
        Ok(PyStructuralForecast {
            inner,
            names: self.component_names(),
        })
    }
    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(state_space_error)
    }
    #[staticmethod]
    fn from_json(value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: StructuralPosterior::from_json(value).map_err(state_space_error)?,
        })
    }
}
fn array3<'py>(py: Python<'py>, v: &[Vec<Vec<f64>>]) -> Bound<'py, PyArray3<f64>> {
    Array3::from_shape_fn((v.len(), v[0].len(), v[0][0].len()), |(i, j, k)| v[i][j][k])
        .into_pyarray(py)
}
fn array4<'py>(py: Python<'py>, v: &[Vec<Vec<Vec<f64>>>]) -> Bound<'py, PyArray4<f64>> {
    let h = v[0][0].len();
    let d = v[0][0].first().map_or(0, Vec::len);
    Array4::from_shape_fn((v.len(), v[0].len(), h, d), |(i, j, k, l)| v[i][j][k][l])
        .into_pyarray(py)
}
#[pyclass(name = "StructuralForecast", frozen)]
pub(crate) struct PyStructuralForecast {
    inner: StructuralPaths,
    names: Vec<String>,
}
#[pymethods]
impl PyStructuralForecast {
    #[getter]
    fn chains(&self) -> usize {
        self.inner.observations.len()
    }
    #[getter]
    fn draws(&self) -> usize {
        self.inner.observations[0].len()
    }
    #[getter]
    fn steps(&self) -> usize {
        self.inner.observations[0][0].len()
    }
    #[getter]
    fn mean_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        self.mean_paths(py)
    }
    #[getter]
    fn observation_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        self.observation_paths(py)
    }
    #[getter]
    fn uncertainty_kind(&self) -> &'static str {
        "parameter_integrated_posterior_predictive"
    }
    #[getter]
    fn component_names(&self) -> Vec<String> {
        self.names.clone()
    }
    #[getter]
    fn state_paths<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray4<f64>> {
        array4(py, &self.inner.states)
    }
    #[getter]
    fn component_paths<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray4<f64>> {
        array4(py, &self.inner.components)
    }
    #[getter]
    fn mean_paths<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        array3(py, &self.inner.means)
    }
    #[getter]
    fn observation_paths<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        array3(py, &self.inner.observations)
    }
    #[getter]
    fn cumulative_observation_paths<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        array3(py, &self.inner.cumulative)
    }
}
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyVarianceParameter>()?;
    m.add_class::<PyStructuralComponent>()?;
    m.add_class::<PyStructuralModel>()?;
    m.add_class::<PyStructuralFit>()?;
    m.add_class::<PyStructuralForecast>()?;
    Ok(())
}
