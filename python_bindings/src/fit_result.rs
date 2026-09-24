//! `FitResult`: the Python view of a core `ModelFit`.
//!
//! Prediction, log-likelihood, deterministics and persistence are computed by
//! `rustmc_core::model::ModelFit`; this module converts arguments and
//! results, and releases the GIL around the computation.
use crate::arviz::{
    arviz_api_generation, arviz_from_groups_versioned, assign_posterior_predictive_draw_coords,
};
use crate::compiled::PyCompiledModel;
use crate::generic_results;
use crate::model_error;
use crate::prediction_binding::prediction_inputs;
use ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rustmc_core::model::{ModelFit, PredictiveDraws};
use rustmc_core::sampler::SampleResult;
use std::collections::HashMap;
use std::sync::Arc;

#[pyclass(module = "rustmc")]
#[derive(Clone)]
pub(crate) struct FitResult {
    /// Shared: cloning a `FitResult`, or taking one from a batch cell, never
    /// copies the posterior.
    pub(crate) fit: Arc<ModelFit>,
}

fn shape_error(error: ndarray::ShapeError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

impl FitResult {
    pub(crate) fn new(fit: ModelFit) -> Self {
        Self { fit: Arc::new(fit) }
    }

    fn display(&self) -> &SampleResult {
        &self.fit.samples
    }

    /// One `(rows, n_obs)` array per likelihood, keyed by likelihood name.
    fn predictive_dict<'py>(
        &self,
        py: Python<'py>,
        draws: PredictiveDraws,
        shape: impl Fn(usize) -> Vec<usize>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for ((name, values), n_obs) in self
            .fit
            .model()
            .likelihood_names
            .iter()
            .zip(draws.values)
            .zip(draws.n_obs)
        {
            let array =
                ndarray::ArrayD::from_shape_vec(shape(n_obs), values).map_err(shape_error)?;
            dict.set_item(name, array.into_pyarray(py))?;
        }
        Ok(dict)
    }
}

#[pymethods]
impl FitResult {
    /// Versioned JSON including bound training data, stored graph draws, and sampler telemetry.
    fn to_json(&self, py: Python<'_>) -> PyResult<String> {
        let fit = &self.fit;
        py.allow_threads(|| fit.to_json()).map_err(model_error)
    }
    #[staticmethod]
    fn from_json(py: Python<'_>, text: &str) -> PyResult<Self> {
        // Loading evaluates the target at every stored draw.
        py.allow_threads(|| ModelFit::from_json(text))
            .map(Self::new)
            .map_err(model_error)
    }
    /// Declarative compiled model with the fitted training data available as bind defaults.
    #[getter]
    fn model(&self) -> PyResult<PyCompiledModel> {
        let (default_data_1d, default_data_2d) = self.fit.training_data().map_err(model_error)?;
        Ok(PyCompiledModel {
            model: self.fit.model().clone(),
            default_data_1d,
            default_data_2d,
        })
    }
    fn get_samples<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        generic_results::samples_flat(self.display(), py)
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

    fn accept_rates<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(py, &self.display().accept_rates)
    }

    /// Print a formatted diagnostics table (R-hat, ESS, MCSE, HDI, divergences).
    fn summary(&self) -> String {
        self.display().diagnostics().to_table()
    }

    /// Return per-parameter diagnostics as a list of dicts.
    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        generic_results::diagnostics(self.display(), py)
    }

    /// Structured sampler telemetry, including integrator work and tree depth.
    fn transition_diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        generic_results::transition_diagnostics(self.fit.raw_samples(), py)
    }

    /// Per-chain adapted step sizes.
    fn step_sizes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(py, &self.display().step_sizes)
    }

    /// Per-chain divergence counts.
    fn divergences<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(py, &self.display().divergences)
    }

    /// Prediction preserving (chain, draw, observation) axes.
    #[pyo3(signature = (data=None, seed=42, expected=false, sizes=None))]
    pub(crate) fn predict<'py>(
        &self,
        py: Python<'py>,
        data: Option<&Bound<'_, PyDict>>,
        seed: u64,
        expected: bool,
        sizes: Option<HashMap<String, usize>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let inputs = prediction_inputs(data)?;
        let fit = &self.fit;
        let draws = py
            .allow_threads(|| {
                let graph = fit.prediction_graph(inputs, sizes)?;
                fit.posterior_predictive(&graph, None, seed, expected)
            })
            .map_err(model_error)?;
        let (chains, n_draws) = (fit.num_chains(), fit.num_draws());
        self.predictive_dict(py, draws, |n_obs| vec![chains, n_draws, n_obs])
    }
    /// Named deterministic draws, with (chain, draw[, observation]) axes.
    #[pyo3(signature = (data=None, sizes=None))]
    fn deterministics<'py>(
        &self,
        py: Python<'py>,
        data: Option<&Bound<'_, PyDict>>,
        sizes: Option<HashMap<String, usize>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let inputs = prediction_inputs(data)?;
        let fit = &self.fit;
        let deterministics = py
            .allow_threads(|| {
                let graph = fit.prediction_graph(inputs, sizes)?;
                fit.deterministics(&graph)
            })
            .map_err(model_error)?;
        let (chains, draws) = (fit.num_chains(), fit.num_draws());
        let result = PyDict::new(py);
        for deterministic in deterministics {
            if deterministic.len == 0 {
                let array = Array2::from_shape_vec((chains, draws), deterministic.values)
                    .map_err(shape_error)?;
                result.set_item(deterministic.name, array.into_pyarray(py))?;
            } else {
                let array = Array3::from_shape_vec(
                    (chains, draws, deterministic.len),
                    deterministic.values,
                )
                .map_err(shape_error)?;
                result.set_item(deterministic.name, array.into_pyarray(py))?;
            }
        }
        Ok(result)
    }
    #[getter]
    fn metadata<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("kernel", "graph_mcmc")?;
        d.set_item("chains", self.fit.num_chains())?;
        d.set_item("draws", self.fit.num_draws())?;
        d.set_item("prediction_axes", ("chain", "draw", "observation"))?;
        let dimensions = PyDict::new(py);
        let graph = self.fit.graph();
        for (slot, obs) in graph.schema.observations.iter().zip(&graph.obs_vectors) {
            dimensions.set_item(&slot.dim, obs.len())?;
        }
        d.set_item("dimensions", dimensions)?;
        Ok(d)
    }

    /// Draw samples from the posterior predictive distribution.
    ///
    /// For each posterior draw (or a random subsample of `n_samples`), runs a
    /// forward pass through the model graph and simulates every likelihood's
    /// observations from its own family (normal, Bernoulli-logit,
    /// Poisson-log, exponential, log-normal or negative binomial).
    ///
    /// Parameters
    /// ----------
    /// n_samples : int or None
    ///     How many posterior draws to use.  None = use all (chains × draws).
    /// seed : int
    ///     RNG seed for the noise draws.
    /// data, sizes : dict or None
    ///     New predictor data or observation-dimension sizes; the training
    ///     data when both are omitted.
    /// expected : bool
    ///     Return each observation's mean instead of a simulated value.
    ///
    /// Returns
    /// -------
    /// dict[str, ndarray(n_samples, n_obs)]
    ///     One key per likelihood, named as it was declared.
    #[pyo3(signature = (n_samples=None, seed=42, data=None, expected=false, sizes=None))]
    fn posterior_predictive<'py>(
        &self,
        py: Python<'py>,
        n_samples: Option<usize>,
        seed: u64,
        data: Option<&Bound<'_, PyDict>>,
        expected: bool,
        sizes: Option<HashMap<String, usize>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let inputs = prediction_inputs(data)?;
        let fit = &self.fit;
        let draws = py
            .allow_threads(|| {
                let graph = fit.prediction_graph(inputs, sizes)?;
                fit.posterior_predictive(&graph, n_samples, seed, expected)
            })
            .map_err(model_error)?;
        let rows = draws.coordinates.len();
        self.predictive_dict(py, draws, |n_obs| vec![rows, n_obs])
    }

    /// Pointwise log-likelihood for each observation in each posterior draw.
    ///
    /// Returns a dict of arrays with shape (chain, draw, obs), one per
    /// likelihood. This is the group ArviZ uses for LOO/WAIC workflows.
    fn log_likelihood<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let fit = &self.fit;
        let blocks = py
            .allow_threads(|| fit.log_likelihood())
            .map_err(model_error)?;
        self.log_likelihood_dict(py, blocks)
    }

    /// Convert to ArviZ's version-native inference container.
    ///
    /// Requires ArviZ: `pip install arviz`
    ///
    /// Returns an `arviz.InferenceData` on ArviZ 0.x or an `xarray.DataTree`
    /// on ArviZ 1.x, with:
    ///   - `posterior`             — (n_chains × n_draws) arrays for every parameter
    ///   - `sample_stats`          — `diverging` (bool) and `step_size` per draw
    ///   - `observed_data`         — the fitted response vector for each likelihood
    ///   - `log_likelihood`        — (n_chains × n_draws × n_obs) pointwise values
    ///   - `posterior_predictive`  — ŷ samples (only when include_ppc=True)
    ///
    /// `posterior_predictive` is exported on the posterior's own
    /// `(chain, draw, obs)` axes, so predictive draw `(c, d)` is the one
    /// generated from posterior draw `(c, d)`. LOO/PSIS and per-chain
    /// predictive diagnostics need that pairing.
    ///
    /// `ppc_samples` thins the draw axis rather than the flattened sample list:
    /// the same `ppc_samples // n_chains` draw indices are retained in every
    /// chain, and the `posterior_predictive` group's `draw` coordinate records
    /// which posterior draws they were, so
    /// `idata.posterior.sel(draw=idata.posterior_predictive.draw)` recovers the
    /// matching parameters. (Before this, `ppc_samples` subsampled a flattened
    /// pool and the export was collapsed to a single fake chain.)
    ///
    /// Example
    /// -------
    ///     idata = fit.to_arviz()
    ///     az.plot_trace(idata)
    ///     az.plot_pair(idata, divergences=True)
    ///     idata = fit.to_arviz(include_ppc=True)
    ///     az.plot_ppc(idata)
    #[pyo3(signature = (include_ppc=false, ppc_samples=None, ppc_seed=42, include_log_likelihood=true))]
    fn to_arviz<'py>(
        &self,
        py: Python<'py>,
        include_ppc: bool,
        ppc_samples: Option<usize>,
        ppc_seed: u64,
        include_log_likelihood: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        // Preserve ArviZ's actual import failure. This distinguishes a missing
        // optional package from a broken transitive dependency or import-time
        // runtime error, all of which previously looked "not installed".
        let az = py.import("arviz")?;
        let fit = &self.fit;
        let has_likelihoods = !fit.model().likelihood_names.is_empty();
        let n_chains = fit.num_chains();
        let n_draws = fit.num_draws();

        // Transitions include warmup for auditability. ArviZ sample_stats is
        // aligned with posterior draws, so export only post-warmup telemetry.
        let raw = fit.raw_samples();
        if raw.transitions.len() != n_chains {
            return Err(PyValueError::new_err(format!(
                "Sampler telemetry has {} chains, but posterior samples have {n_chains} chains",
                raw.transitions.len()
            )));
        }
        let mut step_size_arr = Array2::<f64>::zeros((n_chains, n_draws));
        let mut diverging_arr = Array2::<bool>::from_elem((n_chains, n_draws), false);
        for (ci, transitions) in raw.transitions.iter().enumerate() {
            let post_warmup: Vec<_> = transitions
                .iter()
                .filter(|transition| !transition.is_warmup)
                .collect();
            if post_warmup.len() != n_draws {
                return Err(PyValueError::new_err(format!(
                    "Sampler telemetry for chain {ci} has {} posterior transitions, expected {n_draws}",
                    post_warmup.len()
                )));
            }
            for (di, transition) in post_warmup.into_iter().enumerate() {
                step_size_arr[[ci, di]] = transition.step_size;
                diverging_arr[[ci, di]] = transition.divergent;
            }
        }

        // The per-draw work: pointwise log-likelihood and predictive draws.
        let (log_likelihood, predictive) = py
            .allow_threads(|| {
                let log_likelihood = (include_log_likelihood && has_likelihoods)
                    .then(|| fit.log_likelihood())
                    .transpose()?;
                let predictive = (include_ppc && has_likelihoods)
                    .then(|| fit.posterior_predictive_grid(ppc_samples, ppc_seed))
                    .transpose()?;
                Ok((log_likelihood, predictive))
            })
            .map_err(model_error)?;

        let groups = PyDict::new(py);
        groups.set_item("posterior", self.get_samples_2d(py)?)?;
        let sample_stats = PyDict::new(py);
        sample_stats.set_item("step_size", step_size_arr.into_pyarray(py))?;
        sample_stats.set_item("diverging", diverging_arr.into_pyarray(py))?;
        groups.set_item("sample_stats", sample_stats)?;

        if has_likelihoods {
            let graph = fit.graph();
            let heads = graph.observation_heads();
            let observed_data = PyDict::new(py);
            for (li, name) in fit.model().likelihood_names.iter().enumerate() {
                let head = heads.get(li).ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "observation metadata for likelihood '{}' is unavailable",
                        name
                    ))
                })?;
                let observed = graph.obs_vectors.get(head.obs_data_idx).ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "observed payload for likelihood '{}' is unavailable",
                        name
                    ))
                })?;
                observed_data.set_item(name, PyArray1::from_slice(py, observed))?;
            }
            groups.set_item("observed_data", observed_data)?;
        }

        if let Some(blocks) = log_likelihood {
            groups.set_item("log_likelihood", self.log_likelihood_dict(py, blocks)?)?;
        }

        // Posterior-predictive draws keep the posterior's own (chain, draw)
        // axes so a consumer can pair a predictive draw with the parameters
        // that produced it. `retained_draws` is Some only when `ppc_samples`
        // thinned the draw axis, and then carries the kept draw indices.
        let mut retained_draws = None;
        if let Some((draws, retained)) = predictive {
            let kept = retained.len();
            let ppc = self.predictive_dict(py, draws, |n_obs| vec![n_chains, kept, n_obs])?;
            groups.set_item("posterior_predictive", ppc)?;
            if kept < n_draws {
                retained_draws = Some(retained.into_iter().map(|i| i as i64).collect::<Vec<_>>());
            }
        }

        let arviz_major = arviz_api_generation(&az)?;
        let container = arviz_from_groups_versioned(&az, arviz_major, groups)?;
        if let Some(retained) = retained_draws {
            // Label the thinned axis with the posterior draw indices it came
            // from, so `posterior.sel(draw=ppc.draw)` lines the groups back up.
            assign_posterior_predictive_draw_coords(py, arviz_major, &container, &retained)?;
        }
        Ok(container)
    }

    fn __repr__(&self) -> String {
        let display = self.display();
        let means = display.mean();
        let stds = display.std();
        let parts: Vec<String> = display
            .param_names
            .iter()
            .enumerate()
            .map(|(i, name)| format!("  {}: mean={:.4}, std={:.4}", name, means[i], stds[i]))
            .collect();
        format!(
            "rustmc FitResult ({} chains × {} draws)\n{}",
            display.samples.len(),
            display.samples.first().map_or(0, Vec::len),
            parts.join("\n")
        )
    }
}

impl FitResult {
    fn log_likelihood_dict<'py>(
        &self,
        py: Python<'py>,
        blocks: Vec<Vec<f64>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let (chains, draws) = (self.fit.num_chains(), self.fit.num_draws());
        let dict = PyDict::new(py);
        for (name, values) in self.fit.model().likelihood_names.iter().zip(blocks) {
            let n_obs = values.len() / (chains * draws).max(1);
            let array =
                Array3::from_shape_vec((chains, draws, n_obs), values).map_err(shape_error)?;
            dict.set_item(name, array.into_pyarray(py))?;
        }
        Ok(dict)
    }
}
