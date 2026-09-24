//! Batch results: `BatchFit`, its `BatchResult` cells, and `batch_sample`.
use crate::builder::{
    compile_python_model, reject_discrete_priors_for_gradient_sampling, ModelSpec,
};
use crate::fit_result::FitResult;
use crate::generic_results;
use crate::model_error;
use crate::sampling::SamplerOptions;
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rustmc_core::data::DataBinding as CoreDataBinding;
use rustmc_core::model::{sample_model_batch, ModelBatchCell, ModelFit};
use rustmc_core::sampler::{BatchSeedPolicy, BoundBatchOptions, SampleResult};
use std::collections::HashMap;
use std::sync::Arc;

/// One cell of a batch run: a fit whose posterior is shared, never copied,
/// with the `FitResult` its `fit` property returns.
#[pyclass(module = "rustmc")]
#[derive(Clone)]
pub(crate) struct BatchResult {
    fit: Arc<ModelFit>,
}

impl BatchResult {
    pub(crate) fn new(fit: ModelFit) -> Self {
        Self { fit: Arc::new(fit) }
    }

    /// Display draws for this cell.
    fn display(&self) -> &SampleResult {
        &self.fit.samples
    }
}

#[pymethods]
impl BatchResult {
    /// Internal regression-test hook: compare immutable payload ownership without exposing addresses.
    fn _shares_data(&self, other: &BatchResult, key: &str) -> bool {
        self.fit
            .binding()
            .shares_payload_with(other.fit.binding(), key)
    }
    /// Internal regression-test hook: how many distinct posterior sample trees
    /// this cell retains. One when the display layer passes the raw draws
    /// through unchanged, two when a parameter is genuinely derived.
    fn _posterior_allocations(&self) -> usize {
        if Arc::ptr_eq(self.fit.raw_samples(), &self.fit.samples) {
            1
        } else {
            2
        }
    }

    /// Internal regression-test hook: whether `fit` reuses this cell's retained
    /// posterior rather than holding a copy of it. Compares ownership without
    /// exposing addresses.
    fn _shares_posterior_with(&self, fit: &FitResult) -> bool {
        Arc::ptr_eq(self.fit.raw_samples(), fit.fit.raw_samples())
            && Arc::ptr_eq(&self.fit.samples, &fit.fit.samples)
    }

    #[getter]
    fn fit(&self) -> FitResult {
        FitResult {
            fit: Arc::clone(&self.fit),
        }
    }

    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        generic_results::diagnostics(self.display(), py)
    }

    fn summary(&self) -> String {
        self.display().diagnostics().to_table()
    }

    fn transition_diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        generic_results::transition_diagnostics(self.fit.raw_samples(), py)
    }

    #[pyo3(signature = (data=None, seed=42, expected=false, sizes=None))]
    fn predict<'py>(
        &self,
        py: Python<'py>,
        data: Option<&Bound<'_, PyDict>>,
        seed: u64,
        expected: bool,
        sizes: Option<HashMap<String, usize>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        self.fit().predict(py, data, seed, expected, sizes)
    }

    fn get_samples_2d<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        generic_results::samples_by_chain(self.display(), py)
    }

    fn mean<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let display = self.display();
        generic_results::named_values(py, &display.param_names, display.mean())
    }

    fn std<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let display = self.display();
        generic_results::named_values(py, &display.param_names, display.std())
    }

    fn get_samples<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        generic_results::samples_flat(self.display(), py)
    }

    #[getter]
    fn chains(&self) -> usize {
        self.display().samples.len()
    }

    #[getter]
    fn draws(&self) -> usize {
        self.display().samples.first().map_or(0, Vec::len)
    }

    #[getter]
    fn accept_rate(&self) -> f64 {
        let rates = &self.display().accept_rates;
        if rates.is_empty() {
            0.0
        } else {
            rates.iter().sum::<f64>() / rates.len() as f64
        }
    }

    #[getter]
    fn accept_rates(&self) -> Vec<f64> {
        self.display().accept_rates.clone()
    }

    #[getter]
    fn divergences(&self) -> usize {
        self.display().total_divergences()
    }

    #[getter]
    fn divergences_per_chain(&self) -> Vec<usize> {
        self.display().divergences.clone()
    }

    fn __repr__(&self) -> String {
        let display = self.display();
        let means = display.mean();
        let parts: Vec<String> = display
            .param_names
            .iter()
            .zip(means.iter())
            .map(|(n, m)| format!("{}={:.4}", n, m))
            .collect();
        format!(
            "BatchResult({} chains × {} draws, {})",
            display.samples.len(),
            display.samples.first().map_or(0, Vec::len),
            parts.join(", ")
        )
    }
}

#[pyclass(name = "BatchFit", module = "rustmc")]
pub(crate) struct PyBatchFit {
    pub(crate) ids: Vec<String>,
    pub(crate) results: Vec<Result<BatchResult, String>>,
}

#[pymethods]
impl PyBatchFit {
    #[getter]
    fn ids(&self) -> Vec<String> {
        self.ids.clone()
    }

    #[getter]
    fn errors(&self) -> HashMap<String, String> {
        self.ids
            .iter()
            .zip(&self.results)
            .filter_map(|(id, value)| {
                value
                    .as_ref()
                    .err()
                    .map(|error| (id.clone(), error.clone()))
            })
            .collect()
    }

    fn get(&self, py: Python<'_>, id: &str) -> PyResult<Py<BatchResult>> {
        let index = self
            .ids
            .iter()
            .position(|value| value == id)
            .ok_or_else(|| PyValueError::new_err(format!("unknown dataset ID '{id}'")))?;
        self.__getitem__(py, index as isize)
    }

    fn __len__(&self) -> usize {
        self.results.len()
    }

    fn __getitem__(&self, py: Python<'_>, index: isize) -> PyResult<Py<BatchResult>> {
        let len = self.results.len() as isize;
        let normalized = if index < 0 { len + index } else { index };
        if normalized < 0 || normalized >= len {
            return Err(PyIndexError::new_err("batch index out of range"));
        }
        let result = self
            .results
            .get(normalized as usize)
            .cloned()
            .ok_or_else(|| PyIndexError::new_err("batch index out of range"))?;
        let result = result.map_err(|error| {
            PyValueError::new_err(format!(
                "dataset '{}': {error}",
                self.ids[normalized as usize]
            ))
        })?;
        Py::new(py, result)
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __exit__(
        &self,
        _exc_type: &Bound<'_, PyAny>,
        _exc_value: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) -> bool {
        false
    }

    fn __repr__(&self) -> String {
        format!(
            "BatchFit({} datasets, {} failed)",
            self.results.len(),
            self.errors().len()
        )
    }
}

/// Run thousands of independent models in parallel through Rayon.
///
/// Each entry in `models` is a (ModelSpec, data_dict) pair; the models need
/// not share a structure. By default each gets 1 NUTS chain for throughput,
/// but the batch runner can be configured to use multiple chains or
/// fixed-step HMC when reliability matters more.
///
/// This runs the same batch path as `CompiledModel.sample_batch` with
/// `seed_policy="position_v0"` and `errors="raise"`: model `i` is fitted as a
/// single fit seeded `seed + (i << 32)`, and the first failure raises for the
/// whole batch, naming its dataset index.
#[pyfunction]
#[pyo3(signature = (models, chains=1, draws=500, warmup=300, seed=42, sampler="nuts", step_size=0.0, target_accept=0.8, max_tree_depth=8, num_leapfrog_steps=15, show_progress=true, metric="auto"))]
// The Python API intentionally exposes each sampler option as a named argument.
#[allow(clippy::too_many_arguments)]
pub(crate) fn batch_sample(
    py: Python<'_>,
    models: Vec<(Bound<'_, ModelSpec>, Bound<'_, PyDict>)>,
    chains: usize,
    draws: usize,
    warmup: usize,
    seed: u64,
    sampler: &str,
    step_size: f64,
    target_accept: f64,
    max_tree_depth: usize,
    num_leapfrog_steps: usize,
    show_progress: bool,
    metric: &str,
) -> PyResult<Vec<BatchResult>> {
    let config = SamplerOptions {
        chains,
        draws,
        warmup,
        seed,
        step_size,
        target_accept,
        sampler,
        max_tree_depth,
        num_leapfrog_steps,
        show_progress,
        metric,
    }
    .batch()?;

    let mut cells = Vec::with_capacity(models.len());
    for (index, (spec_bound, data_bound)) in models.iter().enumerate() {
        // Name the dataset in errors raised while preparing it, as the native
        // batch does for errors raised while sampling it, keeping the class.
        let in_dataset = |error: PyErr| {
            PyErr::from_type(
                error.get_type(py),
                format!("dataset '{index}': {}", error.value(py)),
            )
        };
        let spec = spec_bound.borrow();
        reject_discrete_priors_for_gradient_sampling(&spec.priors).map_err(in_dataset)?;
        // Bound data from ModelSpec is the base; call-site dict overrides/extends.
        let (data_map, matrix_map) = spec.data_with(Some(data_bound)).map_err(in_dataset)?;
        let compiled = compile_python_model(&spec, &data_map, &matrix_map).map_err(in_dataset)?;
        let binding =
            CoreDataBinding::from_graph(&compiled.graph).map_err(|error| error.to_string());
        cells.push(ModelBatchCell {
            id: index.to_string(),
            model: compiled.into_model(&spec),
            binding,
            initial: None,
        });
    }
    let options = BoundBatchOptions {
        threads: std::thread::available_parallelism().map_or(1, usize::from),
        chunk_size: cells.len().max(1),
        collect_errors: false,
        seed_policy: BatchSeedPolicy::PositionV0,
    };
    py.allow_threads(|| sample_model_batch(cells, config, options))
        .map_err(model_error)?
        .into_iter()
        .map(|fit| fit.map(BatchResult::new).map_err(PyValueError::new_err))
        .collect()
}
