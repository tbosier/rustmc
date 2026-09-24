//! `FitResult`: one posterior with its model, for summaries and prediction.
use crate::arviz::{
    arviz_api_generation, arviz_from_groups_versioned, assign_posterior_predictive_draw_coords,
};
use crate::builder::ModelSpec;
use crate::compiled::PyCompiledModel;
use crate::fit_artifact;
use crate::generic_results;
use crate::model_error;
use crate::prediction_binding::prediction_graph;
use ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rustmc_core::autodiff::Evaluator;
use rustmc_core::graph::{Graph, ParamTransform};
use rustmc_core::model::DisplayParamSpec;
use rustmc_core::sampler::SampleResult;
use rustmc_core::seeding::{stream_seed, POSTERIOR_PREDICT_SEED_DOMAIN};
use std::collections::HashMap;
use std::sync::Arc;

pub(crate) fn logit_stable(p: f64) -> f64 {
    rustmc_core::prior_sampling::logit_stable(p)
}

pub(crate) fn invert_param_transform(transform: &ParamTransform, value: f64) -> f64 {
    match transform {
        ParamTransform::Identity => value,
        ParamTransform::Exp => value.ln(),
        ParamTransform::Sigmoid => logit_stable(value),
        ParamTransform::BoundedSigmoid { lower, upper } => {
            let span = upper - lower;
            logit_stable((value - lower) / span)
        }
    }
}

pub(crate) fn constrained_draw_to_raw(draw: &[f64], transforms: &[ParamTransform]) -> Vec<f64> {
    draw.iter()
        .zip(transforms.iter())
        .map(|(&value, transform)| invert_param_transform(transform, value))
        .collect()
}

/// Prefer the sampler's exact position: constrained values may round to a
/// transform boundary, making their inverse infinite or otherwise lossy.
pub(crate) fn posterior_position<'a>(
    result: &'a SampleResult,
    graph: &Graph,
    chain: usize,
    draw: usize,
) -> std::borrow::Cow<'a, [f64]> {
    if let Some(positions) = &result.unconstrained_samples {
        std::borrow::Cow::Borrowed(&positions[chain][draw])
    } else {
        std::borrow::Cow::Owned(constrained_draw_to_raw(
            &result.samples[chain][draw],
            &graph.param_transforms,
        ))
    }
}

pub(crate) fn pointwise_log_likelihood_for_draw(
    graph: &Graph,
    raw_draw: &[f64],
    heads: &[rustmc_core::graph::ObservationHead],
) -> PyResult<Vec<Vec<f64>>> {
    let mut evaluator =
        Evaluator::try_new(graph).map_err(|error| PyValueError::new_err(error.to_string()))?;
    evaluator.forward(graph, raw_draw);

    heads
        .iter()
        .map(|head| {
            let aux = head.aux.map(|node| evaluator.scalar_at(node));
            graph.obs_vectors[head.obs_data_idx]
                .iter()
                .enumerate()
                .map(|(i, &observed)| {
                    rustmc_core::observation::log_density(
                        head.family,
                        observed,
                        evaluator.vec_elem(head.linpred, i, graph),
                        aux,
                    )
                    .map_err(|e| PyValueError::new_err(e.to_string()))
                })
                .collect()
        })
        .collect()
}

pub(crate) fn select_posterior_draw_indices(
    total_draws: usize,
    n_samples: Option<usize>,
    rng: &mut ChaCha8Rng,
) -> Vec<usize> {
    let n = n_samples.unwrap_or(total_draws).min(total_draws);
    if n >= total_draws {
        return (0..total_draws).collect();
    }

    let mut indices: Vec<usize> = (0..total_draws).collect();
    indices.shuffle(rng);
    indices.truncate(n);
    indices.sort_unstable();
    indices
}

pub(crate) fn derive_display_draw(draw: &[f64], specs: &[DisplayParamSpec]) -> PyResult<Vec<f64>> {
    rustmc_core::model::derive_display_draw(draw, specs).map_err(model_error)
}

pub(crate) fn derive_display_sample_result(
    raw_result: &SampleResult,
    specs: &[DisplayParamSpec],
) -> PyResult<SampleResult> {
    rustmc_core::model::derive_display_sample_result(raw_result, specs).map_err(model_error)
}

/// True when the display layer is a pure pass-through of the raw draws: every
/// parameter is reported as sampled, in the order it was sampled.
pub(crate) fn display_specs_are_identity(
    raw_result: &SampleResult,
    specs: &[DisplayParamSpec],
) -> bool {
    specs.len() == raw_result.param_names.len()
        && specs.iter().enumerate().all(|(index, spec)| match spec {
            DisplayParamSpec::Raw { name, raw_index } => {
                *raw_index == index && *name == raw_result.param_names[index]
            }
            DisplayParamSpec::DerivedNonCenteredNormal { .. } => false,
        })
}

/// Display draws for a fit, sharing the raw posterior when nothing is derived.
///
/// `derive_display_sample_result` allocates a second copy of every draw. When
/// no parameter is non-centred, that copy is bit-identical to the raw draws, so
/// a retained batch cell paid for two posteriors to hold one. Sharing the `Arc`
/// keeps the display and raw views distinguishable without duplicating them.
pub(crate) fn display_sample_result(
    raw_result: &Arc<SampleResult>,
    specs: &[DisplayParamSpec],
) -> PyResult<Arc<SampleResult>> {
    if display_specs_are_identity(raw_result, specs) {
        // The copying path rejects nonfinite display values; run the same check
        // so sharing can never accept a fit that copying would have refused.
        for chain in &raw_result.samples {
            for draw in chain {
                if draw.iter().any(|value| !value.is_finite()) {
                    derive_display_draw(draw, specs)?;
                }
            }
        }
        return Ok(Arc::clone(raw_result));
    }
    Ok(Arc::new(derive_display_sample_result(raw_result, specs)?))
}

pub(crate) fn validate_transition_chain_count(
    transition_chains: usize,
    expected_chains: usize,
) -> Result<(), String> {
    if transition_chains == expected_chains {
        Ok(())
    } else {
        Err(format!(
            "Sampler telemetry has {} chains, but posterior samples have {expected_chains} chains",
            transition_chains
        ))
    }
}

#[pyclass(module = "rustmc")]
#[derive(Clone)]
pub(crate) struct FitResult {
    pub(crate) definition: ModelSpec,
    /// The posterior draws dominate a fit's memory, so both views are shared
    /// handles: cloning a `FitResult` never duplicates them, and when no
    /// parameter is derived the two point at the same allocation.
    pub(crate) raw_result: Arc<SampleResult>,
    pub(crate) display_result: Arc<SampleResult>,
    /// A clone of the compiled graph — used for predictive sampling.
    pub(crate) graph: Graph,
    /// Name of each likelihood, in the order they appear in the graph.
    pub(crate) likelihood_names: Vec<String>,
}

impl FitResult {
    /// Forward-simulate the observation model at the given `(chain, draw)`
    /// posterior coordinates.
    ///
    /// Returns one flat `coordinates.len() * n_obs` vector per likelihood, in
    /// the order the coordinates were supplied.
    fn simulate_predictive(
        &self,
        graph: &Graph,
        heads: &[rustmc_core::graph::ObservationHead],
        coordinates: &[(usize, usize)],
        expected: bool,
        rng: &mut ChaCha8Rng,
    ) -> PyResult<Vec<Vec<f64>>> {
        let mut evaluator =
            Evaluator::try_new(graph).map_err(|error| PyValueError::new_err(error.to_string()))?;
        let mut preds: Vec<Vec<f64>> = heads
            .iter()
            .map(|head| Vec::with_capacity(coordinates.len() * head.n_obs))
            .collect();

        for &(chain_idx, draw_idx) in coordinates {
            let position = posterior_position(&self.raw_result, graph, chain_idx, draw_idx);
            evaluator.forward(graph, &position);
            for (li, head) in heads.iter().enumerate() {
                for i in 0..head.n_obs {
                    let eta = evaluator.vec_elem(head.linpred, i, graph);
                    let aux = head.aux.map(|node| evaluator.scalar_at(node));
                    preds[li].push(
                        if expected {
                            rustmc_core::observation::mean(head.family, eta, aux)
                        } else {
                            rustmc_core::observation::sample(head.family, eta, aux, rng)
                        }
                        .map_err(|e| PyValueError::new_err(e.to_string()))?,
                    );
                }
            }
        }
        Ok(preds)
    }

    /// Posterior-predictive draws laid out on the posterior's own
    /// `(chain, draw, obs)` grid, so every predictive draw stays paired with the
    /// parameter draw that produced it.
    ///
    /// When `n_samples` asks for fewer draws than were sampled, the thinning is
    /// chain-stratified: one shared set of per-chain draw indices is retained in
    /// every chain. That keeps the exported block rectangular (ArviZ groups must
    /// be dense arrays), represents every chain equally, and leaves a single
    /// `draw` coordinate vector that identifies exactly which posterior draws
    /// were kept. The second return value holds those retained draw indices, or
    /// `None` when nothing was thinned away.
    fn posterior_predictive_grid<'py>(
        &self,
        py: Python<'py>,
        n_samples: Option<usize>,
        seed: u64,
    ) -> PyResult<(Bound<'py, PyDict>, Option<Vec<i64>>)> {
        let graph = prediction_graph(&self.graph, None, None)?;
        graph
            .validate_shapes()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut rng = ChaCha8Rng::seed_from_u64(stream_seed(seed, POSTERIOR_PREDICT_SEED_DOMAIN));
        let heads = graph.observation_heads();

        let n_chains = self.raw_result.samples.len();
        let n_draws = self.raw_result.samples.first().map_or(0, Vec::len);
        let per_chain = match n_samples {
            // A request of fewer draws than there are chains still keeps one
            // draw per chain: dropping whole chains would be worse than
            // overshooting the budget by a handful of draws.
            Some(requested) if n_chains > 0 => (requested / n_chains).max(1).min(n_draws),
            _ => n_draws,
        };
        let retained = select_posterior_draw_indices(n_draws, Some(per_chain), &mut rng);

        let coordinates: Vec<(usize, usize)> = (0..n_chains)
            .flat_map(|chain_idx| retained.iter().map(move |&draw_idx| (chain_idx, draw_idx)))
            .collect();
        let mut preds = self.simulate_predictive(&graph, &heads, &coordinates, false, &mut rng)?;

        let dict = PyDict::new(py);
        for (li, name) in self.likelihood_names.iter().enumerate() {
            let n_obs = heads[li].n_obs;
            let arr = Array3::from_shape_vec(
                (n_chains, retained.len(), n_obs),
                std::mem::take(&mut preds[li]),
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
            dict.set_item(name, arr.into_pyarray(py))?;
        }

        let thinned = retained.len() < n_draws;
        Ok((
            dict,
            thinned.then(|| retained.iter().map(|&index| index as i64).collect()),
        ))
    }
}

#[pymethods]
impl FitResult {
    /// Versioned JSON including bound training data, stored graph draws, and sampler telemetry.
    fn to_json(&self) -> PyResult<String> {
        fit_artifact::encode(self)
    }
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        fit_artifact::decode(text)
    }
    /// Declarative compiled model with the fitted training data available as bind defaults.
    #[getter]
    fn model(&self) -> PyResult<PyCompiledModel> {
        fit_artifact::model(self)
    }
    fn get_samples<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (pidx, name) in self.display_result.param_names.iter().enumerate() {
            let mut all_samples = Vec::new();
            for chain in &self.display_result.samples {
                for draw in chain {
                    all_samples.push(draw[pidx]);
                }
            }
            let arr = PyArray1::from_vec(py, all_samples);
            dict.set_item(name, arr)?;
        }
        Ok(dict)
    }

    fn get_samples_2d<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (pidx, name) in self.display_result.param_names.iter().enumerate() {
            let n_chains = self.display_result.samples.len();
            let n_draws = self.display_result.samples[0].len();
            let mut arr = Array2::<f64>::zeros((n_chains, n_draws));
            for (ci, chain) in self.display_result.samples.iter().enumerate() {
                for (di, draw) in chain.iter().enumerate() {
                    arr[[ci, di]] = draw[pidx];
                }
            }
            dict.set_item(name, arr.into_pyarray(py))?;
        }
        Ok(dict)
    }

    fn mean<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let means = self.display_result.mean();
        let dict = PyDict::new(py);
        for (name, val) in self.display_result.param_names.iter().zip(means.iter()) {
            dict.set_item(name, val)?;
        }
        Ok(dict)
    }

    fn std<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let stds = self.display_result.std();
        let dict = PyDict::new(py);
        for (name, val) in self.display_result.param_names.iter().zip(stds.iter()) {
            dict.set_item(name, val)?;
        }
        Ok(dict)
    }

    fn accept_rates<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::new(py, &self.display_result.accept_rates)?;
        Ok(list)
    }

    /// Print a formatted diagnostics table (R-hat, ESS, MCSE, HDI, divergences).
    fn summary(&self) -> String {
        self.display_result.diagnostics().to_table()
    }

    /// Return per-parameter diagnostics as a list of dicts.
    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        generic_results::diagnostics(&self.display_result, py)
    }

    /// Structured sampler telemetry, including integrator work and tree depth.
    fn transition_diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        generic_results::transition_diagnostics(&self.raw_result, py)
    }

    /// Per-chain adapted step sizes.
    fn step_sizes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::new(py, &self.display_result.step_sizes)?;
        Ok(list)
    }

    /// Per-chain divergence counts.
    fn divergences<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::new(py, &self.display_result.divergences)?;
        Ok(list)
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
        let flat = self.posterior_predictive(py, None, seed, data, expected, sizes)?;
        let result = PyDict::new(py);
        let chains = self.raw_result.samples.len();
        let draws = self.raw_result.samples.first().map_or(0, Vec::len);
        for (name, value) in flat.iter() {
            let arr = value.downcast::<PyArray2<f64>>()?;
            let n = arr.shape()[1];
            let values = unsafe { arr.as_slice()? }.to_vec();
            result.set_item(
                name,
                Array3::from_shape_vec((chains, draws, n), values)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?
                    .into_pyarray(py),
            )?;
        }
        Ok(result)
    }
    /// Named deterministic draws, with (chain, draw[, observation]) axes.
    #[pyo3(signature = (data=None, sizes=None))]
    fn deterministics<'py>(
        &self,
        py: Python<'py>,
        data: Option<&Bound<'_, PyDict>>,
        sizes: Option<HashMap<String, usize>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let graph = prediction_graph(&self.graph, data, sizes)?;
        let mut evaluator =
            Evaluator::try_new(&graph).map_err(|error| PyValueError::new_err(error.to_string()))?;
        let chains = self.raw_result.samples.len();
        let draws = self.raw_result.samples.first().map_or(0, Vec::len);
        let result = PyDict::new(py);
        for (name, node) in &graph.deterministics {
            let n = evaluator.node_len(*node);
            let mut values = Vec::with_capacity(chains * draws * n.max(1));
            for (chain_idx, chain) in self.raw_result.samples.iter().enumerate() {
                for draw_idx in 0..chain.len() {
                    let position =
                        posterior_position(&self.raw_result, &graph, chain_idx, draw_idx);
                    evaluator.forward(&graph, &position);
                    // Same standard the prior predictive holds deterministics
                    // to, and the same one `sampler` holds the parameters to:
                    // a nonfinite value is a failed computation, not a result.
                    for i in 0..n.max(1) {
                        let value = evaluator.vec_elem(*node, i, &graph);
                        if !value.is_finite() {
                            return Err(PyValueError::new_err(format!(
                                "deterministic '{name}' is nonfinite at chain {chain_idx}, \
                                 draw {draw_idx}"
                            )));
                        }
                        values.push(value);
                    }
                }
            }
            if n == 0 {
                result.set_item(
                    name,
                    Array2::from_shape_vec((chains, draws), values)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?
                        .into_pyarray(py),
                )?;
            } else {
                result.set_item(
                    name,
                    Array3::from_shape_vec((chains, draws, n), values)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?
                        .into_pyarray(py),
                )?;
            }
        }
        Ok(result)
    }
    #[getter]
    fn metadata<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("kernel", "graph_mcmc")?;
        d.set_item("chains", self.raw_result.samples.len())?;
        d.set_item("draws", self.raw_result.samples.first().map_or(0, Vec::len))?;
        d.set_item("prediction_axes", ("chain", "draw", "observation"))?;
        let dimensions = PyDict::new(py);
        for (slot, obs) in self
            .graph
            .schema
            .observations
            .iter()
            .zip(&self.graph.obs_vectors)
        {
            dimensions.set_item(&slot.dim, obs.len())?;
        }
        d.set_item("dimensions", dimensions)?;
        Ok(d)
    }

    /// Draw samples from the posterior predictive distribution.
    ///
    /// For each posterior draw (or a random subsample of `n_samples`), runs a
    /// forward pass through the model graph and samples
    ///     ŷ ~ Normal(mu(params), sigma(params))
    /// for every observation.
    ///
    /// Parameters
    /// ----------
    /// n_samples : int or None
    ///     How many posterior draws to use.  None = use all (chains × draws).
    /// seed : int
    ///     RNG seed for the noise draws.
    ///
    /// Returns
    /// -------
    /// dict[str, ndarray(n_samples, n_obs)]
    ///     One key per likelihood (the name passed to normal_likelihood).
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
        let graph = prediction_graph(&self.graph, data, sizes)?;
        let mut rng = ChaCha8Rng::seed_from_u64(stream_seed(seed, POSTERIOR_PREDICT_SEED_DOMAIN));
        graph
            .validate_shapes()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let heads = graph.observation_heads();

        // Flatten all chain draws in order, then subsample without replacement
        // when the caller requests fewer draws than are available.
        let all_draws: Vec<(usize, usize)> = self
            .raw_result
            .samples
            .iter()
            .enumerate()
            .flat_map(|(chain_idx, chain)| {
                (0..chain.len()).map(move |draw_idx| (chain_idx, draw_idx))
            })
            .collect();
        let chosen_indices = select_posterior_draw_indices(all_draws.len(), n_samples, &mut rng);
        let n = chosen_indices.len();
        let coordinates: Vec<(usize, usize)> = chosen_indices
            .into_iter()
            .map(|index| all_draws[index])
            .collect();

        let mut preds =
            self.simulate_predictive(&graph, &heads, &coordinates, expected, &mut rng)?;

        let dict = PyDict::new(py);
        for (li, name) in self.likelihood_names.iter().enumerate() {
            let n_obs = heads[li].n_obs;
            let arr = Array2::from_shape_vec((n, n_obs), std::mem::take(&mut preds[li]))
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            dict.set_item(name, arr.into_pyarray(py))?;
        }
        Ok(dict)
    }

    /// Pointwise log-likelihood for each observation in each posterior draw.
    ///
    /// Returns a dict of arrays with shape (chain, draw, obs), one per
    /// likelihood. This is the group ArviZ uses for LOO/WAIC workflows.
    fn log_likelihood<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.graph
            .validate_shapes()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let heads = self.graph.observation_heads();
        let n_chains = self.raw_result.samples.len();
        let n_draws = self.raw_result.samples.first().map_or(0, |c| c.len());

        let mut arrays: Vec<Array3<f64>> = heads
            .iter()
            .map(|head| Array3::<f64>::zeros((n_chains, n_draws, head.n_obs)))
            .collect();

        for (chain_idx, chain) in self.raw_result.samples.iter().enumerate() {
            for draw_idx in 0..chain.len() {
                let position =
                    posterior_position(&self.raw_result, &self.graph, chain_idx, draw_idx);
                let per_head = pointwise_log_likelihood_for_draw(&self.graph, &position, &heads)?;
                for (li, values) in per_head.iter().enumerate() {
                    for (obs_idx, &value) in values.iter().enumerate() {
                        arrays[li][[chain_idx, draw_idx, obs_idx]] = value;
                    }
                }
            }
        }

        let dict = PyDict::new(py);
        for (li, name) in self.likelihood_names.iter().enumerate() {
            dict.set_item(name, arrays[li].clone().into_pyarray(py))?;
        }
        Ok(dict)
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

        let n_chains = self.display_result.samples.len();
        let n_draws = self.display_result.samples.first().map_or(0, |c| c.len());

        // ── posterior ────────────────────────────────────────────────────
        let posterior = self.get_samples_2d(py)?;

        // ── sample_stats ─────────────────────────────────────────────────
        // Transitions include warmup for auditability. ArviZ sample_stats is
        // aligned with posterior draws, so export only post-warmup telemetry.
        validate_transition_chain_count(self.raw_result.transitions.len(), n_chains)
            .map_err(PyValueError::new_err)?;
        let sample_stats = PyDict::new(py);
        let mut step_size_arr = Array2::<f64>::zeros((n_chains, n_draws));
        let mut diverging_arr = Array2::<bool>::from_elem((n_chains, n_draws), false);
        for (ci, transitions) in self.raw_result.transitions.iter().enumerate() {
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
        sample_stats.set_item("step_size", step_size_arr.into_pyarray(py))?;
        sample_stats.set_item("diverging", diverging_arr.into_pyarray(py))?;

        // ── posterior predictive (optional) ──────────────────────────────
        let groups = PyDict::new(py);
        groups.set_item("posterior", posterior)?;
        groups.set_item("sample_stats", sample_stats)?;

        if !self.likelihood_names.is_empty() {
            let heads = self.graph.observation_heads();
            let observed_data = PyDict::new(py);
            for (li, name) in self.likelihood_names.iter().enumerate() {
                let head = heads.get(li).ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "observation metadata for likelihood '{}' is unavailable",
                        name
                    ))
                })?;
                let observed = self
                    .graph
                    .obs_vectors
                    .get(head.obs_data_idx)
                    .ok_or_else(|| {
                        PyValueError::new_err(format!(
                            "observed payload for likelihood '{}' is unavailable",
                            name
                        ))
                    })?;
                observed_data.set_item(name, PyArray1::from_vec(py, observed.clone()))?;
            }
            groups.set_item("observed_data", observed_data)?;
        }

        if include_log_likelihood && !self.likelihood_names.is_empty() {
            let log_likelihood = self.log_likelihood(py)?;
            groups.set_item("log_likelihood", log_likelihood)?;
        }

        // Posterior-predictive draws keep the posterior's own (chain, draw)
        // axes so a consumer can pair a predictive draw with the parameters
        // that produced it. `retained_draws` is Some only when `ppc_samples`
        // thinned the draw axis, and then carries the kept draw indices.
        let mut retained_draws = None;
        if include_ppc && !self.likelihood_names.is_empty() {
            let (ppc_dict, retained) = self.posterior_predictive_grid(py, ppc_samples, ppc_seed)?;
            groups.set_item("posterior_predictive", ppc_dict)?;
            retained_draws = retained;
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
        let means = self.display_result.mean();
        let stds = self.display_result.std();
        let mut parts = Vec::new();
        for (i, name) in self.display_result.param_names.iter().enumerate() {
            parts.push(format!(
                "  {}: mean={:.4}, std={:.4}",
                name, means[i], stds[i]
            ));
        }
        let n_chains = self.display_result.samples.len();
        let n_draws = if self.display_result.samples.is_empty() {
            0
        } else {
            self.display_result.samples[0].len()
        };
        format!(
            "rustmc FitResult ({} chains × {} draws)\n{}",
            n_chains,
            n_draws,
            parts.join("\n")
        )
    }
}
