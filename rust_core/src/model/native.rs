//! The versioned model artifact, the Rust-native model built on it, and batch fitting.
use crate::autodiff::Evaluator;
use crate::data::{DataBinding, DataInputs, DataSchema, SlotKind};
use crate::graph::Graph;
use crate::sampler::{self, BatchSampleConfig, BoundBatchOptions, SamplerConfig};
use std::collections::HashMap;
use std::sync::Arc;

use super::*;

/// The existing Python wire format, now owned by the Rust core.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelArtifact {
    pub format: String,
    pub version: u32,
    pub definition: ModelSpec,
    pub schema: DataSchema,
}

#[derive(Clone, Debug)]
pub struct GraphModel {
    pub definition: ModelSpec,
    pub structure: Arc<Graph>,
    pub likelihood_names: Vec<String>,
    pub display_params: Vec<DisplayParamSpec>,
}
impl GraphModel {
    pub fn from_json(text: &str) -> ModelResult<Self> {
        let artifact: ModelArtifact = serde_json::from_str(text)
            .map_err(|e| ModelError::invalid(format!("invalid graph model artifact: {e}")))?;
        Self::from_artifact(artifact)
    }
    pub fn from_artifact(artifact: ModelArtifact) -> ModelResult<Self> {
        if artifact.format != "rustmc.graph-model" || artifact.version != 1 {
            return Err(ModelError::invalid(
                "unsupported graph model artifact format/version",
            ));
        }
        let mut definition = artifact.definition.structure_definition();
        // No discrete-prior rejection here. Bernoulli and Poisson priors are
        // deliberately kept for prior-predictive simulation and refused only
        // for gradient sampling, so refusing them at *load* time made the
        // simulation they are kept for unreachable from an artifact.
        //
        // What covers sampling instead: `GraphModel::sample` goes through
        // `sampler::sample_bound_with_init`, which calls
        // `reject_discrete_latent_parameters`. Every `PriorSpec::Bernoulli` and
        // `PriorSpec::Poisson` compiles to exactly the shape that scan looks
        // for -- a `BernoulliLogP`/`PoissonLogP` term over an `Op::Param` -- so
        // across the priors this loader can carry, the two checks accept and
        // reject the same models.
        //
        // What it does not cover, and did not before either: the `pub`
        // `nuts::run_chain`/`hmc::run_chain` kernels, which take a graph
        // directly, and a discrete density reaching a free parameter
        // indirectly rather than through a bare `Op::Param` -- a gap
        // `reject_discrete_latent_parameters` documents on itself. Neither is
        // reachable *from an artifact*, whose definition is a declarative
        // `ModelSpec` that can only name priors.
        //
        // The Python bindings keep a stricter rule of their own:
        // `CompiledModel` and `FitResult` are posterior-sampling objects, so
        // `model_artifact::from_core` refuses a discrete prior in either.
        for slot in &artifact.schema.matrices {
            let SlotKind::Matrix { n_cols } = slot.kind else {
                return Err(ModelError::invalid("invalid matrix schema"));
            };
            if n_cols == 0 || n_cols > 1_000_000 {
                return Err(ModelError::invalid(
                    "matrix column count must be between 1 and 1000000",
                ));
            }
            definition
                .bound_data_2d
                .insert(slot.key.clone(), (vec![0.; n_cols], 1, n_cols));
        }
        let (one_d, two_d) = template_data_for_spec(&definition)?;
        let compiled = compile(&definition, &one_d, &two_d)?;
        if compiled.graph.schema != artifact.schema {
            return Err(ModelError::invalid(
                "artifact schema disagrees with its model definition",
            ));
        }
        Ok(Self {
            definition: definition.structure_definition(),
            structure: Arc::new(compiled.graph.structure_only()),
            likelihood_names: compiled.likelihood_names,
            display_params: compiled.display_params,
        })
    }
    pub fn to_json(&self) -> ModelResult<String> {
        serde_json::to_string(&self.artifact()).map_err(|e| ModelError::invalid(e.to_string()))
    }
    pub fn artifact(&self) -> ModelArtifact {
        ModelArtifact {
            format: "rustmc.graph-model".into(),
            version: 1,
            definition: self.definition.structure_definition(),
            schema: self.structure.schema.clone(),
        }
    }
    pub fn bind(&self, inputs: DataInputs, id: impl Into<String>) -> ModelResult<DataBinding> {
        let binding = DataBinding::bind(&self.structure.schema, inputs, id.into(), true, true)
            .map_err(|e| ModelError::invalid(e.to_string()))?;
        binding
            .validate_for(&self.structure)
            .map_err(|e| ModelError::invalid(e.to_string()))?;
        Ok(binding)
    }
    /// Positions and returned gradients use the graph's unconstrained parameter order.
    pub fn log_density(
        &self,
        binding: &DataBinding,
        position: &[f64],
    ) -> ModelResult<(f64, Vec<f64>)> {
        binding
            .validate_for(&self.structure)
            .map_err(|e| ModelError::invalid(e.to_string()))?;
        if position.len() != self.structure.param_count || position.iter().any(|x| !x.is_finite()) {
            return Err(ModelError::invalid(
                "position must match the finite unconstrained parameter axis",
            ));
        }
        let graph = self.structure.with_binding(binding);
        let mut evaluator =
            Evaluator::try_new(&graph).map_err(|error| ModelError::invalid(error.to_string()))?;
        evaluator.compute(&graph, position);
        Ok((evaluator.total_logp, evaluator.grad))
    }
    pub fn sample(
        &self,
        binding: DataBinding,
        config: SamplerConfig,
        initial: Option<Vec<Vec<f64>>>,
    ) -> ModelResult<ModelFit> {
        let raw = sampler::sample_bound_with_init(
            Arc::clone(&self.structure),
            binding.clone(),
            config,
            initial,
        )
        .map_err(ModelError::invalid)?;
        ModelFit::new(self.clone(), binding, raw)
    }
    /// Fit this model to many datasets on a bounded pool, as
    /// [`sampler::sample_batch_bound_with_initial`] does, keeping each
    /// successful cell as a [`ModelFit`]. `initial` is keyed by dataset ID.
    pub fn sample_batch(
        &self,
        bindings: Vec<(String, Result<DataBinding, String>)>,
        config: BatchSampleConfig,
        options: BoundBatchOptions,
        mut initial: HashMap<String, Vec<Vec<f64>>>,
    ) -> ModelResult<Vec<Result<ModelFit, String>>> {
        config.validate().map_err(ModelError::invalid)?;
        if let Some(id) = initial
            .keys()
            .find(|id| !bindings.iter().any(|(known, _)| known == *id))
        {
            return Err(ModelError::invalid(format!(
                "initialization supplied for unknown dataset ID '{id}'"
            )));
        }
        let cells = bindings
            .into_iter()
            .map(|(id, binding)| ModelBatchCell {
                initial: initial.remove(&id),
                id,
                model: self.clone(),
                binding,
            })
            .collect();
        sample_model_batch(cells, config, options)
    }
    /// Auto-promoted vector parameters, recovered from the stored schema.
    ///
    /// `collect_matvec_params` only reads each matrix's column count, which the
    /// schema carries, so the artifact alone is enough — no data needed.
    fn auto_vector_params(&self) -> ModelResult<HashMap<String, usize>> {
        let mut matrices: Data2d = HashMap::new();
        for slot in &self.structure.schema.matrices {
            let SlotKind::Matrix { n_cols } = slot.kind else {
                return Err(ModelError::invalid("invalid matrix schema"));
            };
            matrices.insert(slot.key.clone(), (Vec::new(), 0, n_cols));
        }
        collect_matvec_params(&self.definition, &matrices)
    }
    /// One draw from the prior, needing no data and no fit.
    ///
    /// `PriorDraw::raw` uses the same unconstrained axis as `log_density`;
    /// `PriorDraw::display` uses the reported parameter order.
    ///
    /// A potential contributes to the target but has no generator, so for such
    /// a model the priors alone are not the prior and this refuses rather than
    /// returning draws from a distribution the model does not have.
    pub fn sample_prior<R: rand::Rng + ?Sized>(
        &self,
        rng: &mut R,
    ) -> ModelResult<crate::prior_sampling::PriorDraw> {
        reject_potentials_for_prior_predictive(&self.definition.potentials)?;
        crate::prior_sampling::sample_prior_draw(
            &self.structure,
            &self.definition.priors,
            &self.display_params,
            &self.auto_vector_params()?,
            rng,
        )
    }
    /// Prior predictive simulation: parameters drawn from the priors and
    /// observations simulated through the likelihood at the bound covariates.
    ///
    /// `PriorPredictive::params` is indexed like `display_params`, and
    /// `PriorPredictive::predictions` like `likelihood_names`.
    pub fn prior_predictive<R: rand::Rng + ?Sized>(
        &self,
        binding: &DataBinding,
        n_samples: usize,
        rng: &mut R,
    ) -> ModelResult<crate::prior_sampling::PriorPredictive> {
        reject_potentials_for_prior_predictive(&self.definition.potentials)?;
        binding
            .validate_for(&self.structure)
            .map_err(|e| ModelError::invalid(e.to_string()))?;
        let graph = self.structure.with_binding(binding);
        crate::prior_sampling::prior_predictive(
            &graph,
            &self.definition.priors,
            &self.display_params,
            &self.auto_vector_params()?,
            n_samples,
            rng,
        )
    }
}

impl CompiledDefinition {
    /// The immutable model this compilation describes, compiled from
    /// `definition`; the definition's bound data is not kept.
    pub fn into_model(self, definition: &ModelSpec) -> GraphModel {
        GraphModel {
            definition: definition.structure_definition(),
            structure: Arc::new(self.graph.structure_only()),
            likelihood_names: self.likelihood_names,
            display_params: self.display_params,
        }
    }
}

/// One dataset of a model batch: the model, its data (or why it could not be
/// bound), and optionally one start per chain.
#[derive(Clone, Debug)]
pub struct ModelBatchCell {
    pub id: String,
    pub model: GraphModel,
    pub binding: Result<DataBinding, String>,
    pub initial: Option<Vec<Vec<f64>>>,
}

/// Fit independent models, which need not share a structure, through
/// [`sampler::sample_batch_cells`], keeping each successful cell as a
/// [`ModelFit`].
///
/// A cell whose reported draws cannot be derived fails like any other cell:
/// it is that cell's error under `collect_errors`, and otherwise ends the
/// batch as `"dataset '<id>': <error>"`.
pub fn sample_model_batch(
    cells: Vec<ModelBatchCell>,
    config: BatchSampleConfig,
    options: BoundBatchOptions,
) -> ModelResult<Vec<Result<ModelFit, String>>> {
    let collect_errors = options.collect_errors;
    let sampler_cells = cells
        .iter()
        .map(|cell| sampler::BatchCell {
            id: cell.id.clone(),
            graph: Arc::clone(&cell.model.structure),
            binding: cell.binding.clone(),
            initial: cell.initial.clone(),
        })
        .collect();
    let results =
        sampler::sample_batch_cells(sampler_cells, config, options).map_err(ModelError::invalid)?;
    let mut fits = Vec::with_capacity(results.len());
    for (result, cell) in results.into_iter().zip(cells) {
        let fit = result.and_then(|raw| {
            ModelFit::new(cell.model, cell.binding?, raw).map_err(|error| error.to_string())
        });
        match fit {
            Err(error) if !collect_errors => {
                return Err(ModelError::invalid(format!(
                    "dataset '{}': {error}",
                    cell.id
                )))
            }
            fit => fits.push(fit),
        }
    }
    Ok(fits)
}

pub fn bind_prediction(
    graph: &Graph,
    mut inputs: DataInputs,
    mut lengths: HashMap<String, usize>,
) -> ModelResult<Graph> {
    for dimension in lengths.keys() {
        if !graph
            .schema
            .vectors
            .iter()
            .chain(&graph.schema.observations)
            .chain(&graph.schema.matrices)
            .any(|slot| &slot.dim == dimension)
        {
            return Err(ModelError::invalid(format!(
                "unknown prediction dimension '{dimension}'"
            )));
        }
    }
    for slot in graph.schema.vectors.iter().chain(&graph.schema.matrices) {
        let len = inputs
            .vectors
            .get(&slot.key)
            .map(|v| v.len())
            .or_else(|| inputs.matrices.get(&slot.key).map(|m| m.n_rows))
            .ok_or_else(|| {
                ModelError::invalid(format!("missing prediction data key '{}'", slot.key))
            })?;
        if lengths
            .insert(slot.dim.clone(), len)
            .is_some_and(|n| n != len)
        {
            return Err(ModelError::invalid(format!(
                "prediction dimension '{}' has inconsistent lengths",
                slot.dim
            )));
        }
    }
    for (i, slot) in graph.schema.observations.iter().enumerate() {
        if graph.schema.vectors.iter().any(|s| s.key == slot.key) {
            continue;
        }
        let n = lengths
            .get(&slot.dim)
            .copied()
            .or_else(|| graph.obs_vectors.get(i).map(Vec::len))
            .ok_or_else(|| {
                ModelError::invalid(format!(
                    "supply size for prediction dimension '{}'",
                    slot.dim
                ))
            })?;
        if n == 0 {
            return Err(ModelError::invalid(
                "prediction dimensions must be positive",
            ));
        }
        inputs
            .vectors
            .insert(slot.key.clone(), Arc::from(vec![1.; n]));
    }
    let binding = DataBinding::bind(&graph.schema, inputs, "prediction", true, true)
        .map_err(|e| ModelError::invalid(e.to_string()))?;
    let bound = graph.with_binding(&binding);
    bound
        .validate_shapes()
        .map_err(|e| ModelError::invalid(e.to_string()))?;
    Ok(bound)
}

impl ModelArtifact {
    /// Fit payloads already declare the complete parameter axis. Reject impossible
    /// structural sizes before allocating template matrices or parameter vectors.
    pub fn validate_parameter_limit(&self, count: usize) -> ModelResult<()> {
        let mut minimum = 0usize;
        for prior in &self.definition.priors {
            let n = match prior {
                PriorSpec::VectorNormal { n, .. } => *n,
                _ => 1,
            };
            minimum = minimum
                .checked_add(n)
                .ok_or_else(|| ModelError::invalid("artifact parameter count overflow"))?;
            if minimum > count {
                return Err(ModelError::invalid(
                    "artifact model parameter dimensions exceed posterior axis",
                ));
            }
        }
        for slot in &self.schema.matrices {
            if let crate::data::SlotKind::Matrix { n_cols } = slot.kind {
                if n_cols > count {
                    return Err(ModelError::invalid(
                        "artifact matrix width exceeds posterior parameter axis",
                    ));
                }
            }
        }
        Ok(())
    }
}
