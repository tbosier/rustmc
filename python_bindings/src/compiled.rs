//! `CompiledModel` and `BoundModel`: structure compiled once, data bound per use.
use crate::batch::{BatchResult, PyBatchFit};
use crate::data_input::{data_inputs_from_maps, parse_data_dict, Data1d, Data2d};
use crate::fit_result::FitResult;
use crate::model_artifact;
use crate::model_error;
use crate::sampling::SamplerOptions;
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rustmc_core::data::{DataBinding as CoreDataBinding, DataInputs, MatrixBinding};
use rustmc_core::graph::Graph;
use rustmc_core::model::GraphModel;
use rustmc_core::sampler::{BatchSeedPolicy, BoundBatchOptions};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

#[pyclass(name = "BoundModel", module = "rustmc")]
#[derive(Clone)]
pub(crate) struct PyBoundModel {
    pub(crate) structure: Arc<Graph>,
    pub(crate) binding: CoreDataBinding,
}

#[pyclass(name = "CompiledModel", module = "rustmc")]
#[derive(Clone)]
pub(crate) struct PyCompiledModel {
    pub(crate) model: GraphModel,
    /// Data bound at build time (or a fit's training data), used for any key
    /// a call does not supply.
    pub(crate) default_data_1d: Data1d,
    pub(crate) default_data_2d: Data2d,
}

impl PyCompiledModel {
    fn structure(&self) -> &Arc<Graph> {
        &self.model.structure
    }

    /// Bind one dataset, given as a dict or a `BoundModel` of this model.
    ///
    /// Dict entries override `base`; a key in `shared` may not be overridden,
    /// because a batch shares that payload across every dataset.
    fn bind_dataset(
        &self,
        value: &Bound<'_, PyAny>,
        id: String,
        base: &DataInputs,
        shared: &HashSet<String>,
    ) -> PyResult<CoreDataBinding> {
        if let Ok(bound) = value.downcast::<PyBoundModel>() {
            let bound = bound.borrow();
            if !Arc::ptr_eq(&bound.structure, self.structure()) {
                return Err(PyValueError::new_err(
                    "BoundModel belongs to a different CompiledModel",
                ));
            }
            let mut binding = bound.binding.clone();
            binding.set_id(id);
            return self.validated(binding);
        }
        let dict = value.downcast::<PyDict>().map_err(|_| {
            PyValueError::new_err("data must be a dict or a BoundModel from this compiled model")
        })?;
        let (extra_1d, extra_2d) = parse_data_dict(dict)?;
        if let Some(key) = extra_1d
            .keys()
            .chain(extra_2d.keys())
            .find(|key| shared.contains(*key))
        {
            return Err(PyValueError::new_err(format!(
                "data key '{}' appears in both shared and per-dataset inputs",
                key
            )));
        }
        // Cloning the base clones Arc handles only, so a default or shared
        // payload stays one allocation however many datasets use it.
        let mut inputs = base.clone();
        for (key, values) in extra_1d {
            inputs.matrices.remove(&key);
            inputs.vectors.insert(key, Arc::from(values));
        }
        for (key, (values, n_rows, n_cols)) in extra_2d {
            inputs.vectors.remove(&key);
            let matrix = MatrixBinding {
                data: Arc::from(values),
                n_rows,
                n_cols,
            };
            inputs.matrices.insert(key, matrix);
        }
        let binding = CoreDataBinding::bind(&self.structure().schema, inputs, id, true, true)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        self.validated(binding)
    }

    fn validated(&self, binding: CoreDataBinding) -> PyResult<CoreDataBinding> {
        binding
            .validate_for(self.structure())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(binding)
    }

    fn default_inputs(&self) -> DataInputs {
        data_inputs_from_maps(&self.default_data_1d, &self.default_data_2d)
    }

    fn bind_any(&self, value: &Bound<'_, PyAny>, id: String) -> PyResult<CoreDataBinding> {
        self.bind_dataset(value, id, &self.default_inputs(), &HashSet::new())
    }
}

#[pymethods]
impl PyBoundModel {
    #[getter]
    fn id(&self) -> &str {
        self.binding.id()
    }

    #[getter]
    fn n_obs(&self) -> usize {
        self.binding.n_obs()
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
}

#[pymethods]
impl PyCompiledModel {
    /// Versioned declarative artifact; excludes all bound training data and defaults.
    fn to_json(&self) -> PyResult<String> {
        model_artifact::encode(self)
    }
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        model_artifact::decode(text)
    }

    /// Evaluate the native graph target and gradient in unconstrained coordinates.
    fn log_density<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'_, PyAny>,
        position: Vec<f64>,
    ) -> PyResult<(f64, Bound<'py, PyArray1<f64>>)> {
        let binding = self.bind_any(data, "0".into())?;
        let (log_density, gradient) = self
            .model
            .log_density(&binding, &position)
            .map_err(model_error)?;
        Ok((log_density, gradient.into_pyarray(py)))
    }
    #[getter]
    fn dimensions(&self) -> HashMap<String, String> {
        let schema = &self.structure().schema;
        schema
            .observations
            .iter()
            .chain(&schema.vectors)
            .chain(&schema.matrices)
            .map(|s| (s.key.clone(), s.dim.clone()))
            .collect()
    }
    #[getter]
    fn param_names(&self) -> Vec<String> {
        self.structure().param_names.clone()
    }

    #[getter]
    fn required_keys(&self) -> Vec<String> {
        self.structure()
            .schema
            .required_keys()
            .into_iter()
            .map(str::to_string)
            .collect()
    }

    /// Stable for this process and useful for verifying Arc structure sharing.
    #[getter]
    fn structure_id(&self) -> usize {
        Arc::as_ptr(self.structure()) as usize
    }

    #[pyo3(signature = (data, id="0", strict=true, check_finite=true))]
    fn bind(
        &self,
        data: &Bound<'_, PyDict>,
        id: &str,
        strict: bool,
        check_finite: bool,
    ) -> PyResult<PyBoundModel> {
        let mut inputs = self.default_inputs();
        let (extra_1d, extra_2d) = parse_data_dict(data)?;
        let extra = data_inputs_from_maps(&extra_1d, &extra_2d);
        for key in extra_1d.keys() {
            inputs.matrices.remove(key);
        }
        for key in extra_2d.keys() {
            inputs.vectors.remove(key);
        }
        inputs.vectors.extend(extra.vectors);
        inputs.matrices.extend(extra.matrices);
        let binding = CoreDataBinding::bind(
            &self.structure().schema,
            inputs,
            id.to_string(),
            strict,
            check_finite,
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyBoundModel {
            structure: Arc::clone(self.structure()),
            binding: self.validated(binding)?,
        })
    }

    #[pyo3(signature = (data, chains=4, draws=1000, warmup=500, seed=42, threads=0, step_size=0.0, target_accept=0.8, sampler="nuts", max_tree_depth=10, num_leapfrog_steps=15, show_progress=true, init=None, metric="auto"))]
    #[allow(clippy::too_many_arguments)]
    fn sample(
        &self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        chains: usize,
        draws: usize,
        warmup: usize,
        seed: u64,
        threads: usize,
        step_size: f64,
        target_accept: f64,
        sampler: &str,
        max_tree_depth: usize,
        num_leapfrog_steps: usize,
        show_progress: bool,
        init: Option<Vec<Vec<f64>>>,
        metric: &str,
    ) -> PyResult<FitResult> {
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
        .single(threads)?;
        let binding = self.bind_any(data, "0".to_string())?;
        let model = &self.model;
        let fit = py
            .allow_threads(|| model.sample(binding, config, init))
            .map_err(model_error)?;
        Ok(FitResult::new(fit))
    }

    #[pyo3(signature = (datasets, ids=None, shared=None, chains=1, draws=500, warmup=300, seed=42, sampler="nuts", step_size=0.0, target_accept=0.8, max_tree_depth=8, num_leapfrog_steps=15, show_progress=true, threads=1, chunk_size=64, errors="raise", seed_policy="cell_id_v1", init=None, metric="auto"))]
    #[allow(clippy::too_many_arguments)]
    fn sample_batch(
        &self,
        py: Python<'_>,
        datasets: Vec<Bound<'_, PyAny>>,
        ids: Option<Vec<String>>,
        shared: Option<&Bound<'_, PyDict>>,
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
        threads: usize,
        chunk_size: usize,
        errors: &str,
        seed_policy: &str,
        init: Option<&Bound<'_, PyDict>>,
        metric: &str,
    ) -> PyResult<PyBatchFit> {
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
        let collect_errors = match errors {
            "raise" => false,
            "collect" => true,
            _ => return Err(PyValueError::new_err("errors must be 'raise' or 'collect'")),
        };
        let seed_policy = match seed_policy {
            "cell_id_v1" => BatchSeedPolicy::CellIdV1,
            "position_v0" => BatchSeedPolicy::PositionV0,
            _ => {
                return Err(PyValueError::new_err(
                    "seed_policy must be 'cell_id_v1' or 'position_v0'",
                ))
            }
        };
        let ids = ids.unwrap_or_else(|| (0..datasets.len()).map(|i| i.to_string()).collect());
        if ids.len() != datasets.len() {
            return Err(PyValueError::new_err(
                "ids length must equal datasets length",
            ));
        }
        let mut unique = HashSet::new();
        if ids.iter().any(|id| !unique.insert(id)) {
            return Err(PyValueError::new_err("dataset ids must be unique"));
        }
        let mut initial_positions = HashMap::new();
        let mut initial_errors = HashMap::new();
        if let Some(initial) = init {
            for (key, value) in initial.iter() {
                let id: String = key.extract()?;
                if !ids.contains(&id) {
                    return Err(PyValueError::new_err(format!(
                        "initialization supplied for unknown dataset ID '{id}'"
                    )));
                }
                match value.extract::<Vec<Vec<f64>>>() {
                    Ok(positions) => {
                        initial_positions.insert(id, positions);
                    }
                    Err(error) => {
                        initial_errors.insert(id, format!("invalid init: {error}"));
                    }
                }
            }
        }
        // Convert defaults and shared payloads once, so every dataset shares
        // them rather than holding its own copy.
        let mut base = self.default_inputs();
        let mut shared_keys = HashSet::new();
        if let Some(shared) = shared {
            let (shared_1d, shared_2d) = parse_data_dict(shared)?;
            for key in shared_1d.keys() {
                base.matrices.remove(key);
            }
            for key in shared_2d.keys() {
                base.vectors.remove(key);
            }
            shared_keys.extend(shared_1d.keys().cloned());
            shared_keys.extend(shared_2d.keys().cloned());
            let shared = data_inputs_from_maps(&shared_1d, &shared_2d);
            base.vectors.extend(shared.vectors);
            base.matrices.extend(shared.matrices);
        }
        let bindings = datasets
            .iter()
            .zip(&ids)
            .map(|(data, id)| {
                let binding = match initial_errors.get(id) {
                    Some(error) => Err(PyValueError::new_err(error.clone())),
                    None => self.bind_dataset(data, id.clone(), &base, &shared_keys),
                };
                (id.clone(), binding.map_err(|error| error.to_string()))
            })
            .collect();
        let options = BoundBatchOptions {
            threads,
            chunk_size,
            collect_errors,
            seed_policy,
        };
        let model = &self.model;
        let results = py
            .allow_threads(|| model.sample_batch(bindings, config, options, initial_positions))
            .map_err(model_error)?
            .into_iter()
            .map(|fit| fit.map(BatchResult::new))
            .collect();
        Ok(PyBatchFit { ids, results })
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
            "CompiledModel(params={}, required_keys={:?})",
            self.structure().param_count,
            self.required_keys()
        )
    }
}
