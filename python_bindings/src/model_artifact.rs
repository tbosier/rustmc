//! Python adapter for the native graph-model artifact.
use super::{model_error, ModelSpec, PyCompiledModel};
use pyo3::prelude::*;
use rustmc_core::model::GraphModel;
use std::collections::HashMap;
pub(super) type Artifact = rustmc_core::model::ModelArtifact;
pub(super) fn describe(model: &PyCompiledModel) -> Artifact {
    Artifact {
        format: "rustmc.graph-model".into(),
        version: 1,
        definition: model.definition.0.structure_definition(),
        schema: model.structure.schema.clone(),
    }
}
pub(super) fn encode(model: &PyCompiledModel) -> PyResult<String> {
    serde_json::to_string(&describe(model))
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}
pub(super) fn decode(text: &str) -> PyResult<PyCompiledModel> {
    let model = GraphModel::from_json(text).map_err(model_error)?;
    from_core(model)
}
/// A `CompiledModel` exists to be sampled from: `sample`, `sample_batch` and
/// `log_density` are its whole surface, and `ModelBuilder.compile()` refuses a
/// discrete prior for that reason. So this refuses one too, and the two ways of
/// obtaining a `CompiledModel` agree.
///
/// The core's `GraphModel::from_artifact` is deliberately permissive, because a
/// Rust caller loads an artifact to simulate its prior predictive as well as to
/// fit it. The restriction is this crate's, not the format's, so it lives here.
fn from_core(model: GraphModel) -> PyResult<PyCompiledModel> {
    super::reject_discrete_priors_for_gradient_sampling(&model.definition.priors)?;
    Ok(PyCompiledModel {
        definition: ModelSpec(model.definition),
        structure: model.structure,
        likelihood_names: model.likelihood_names,
        display_params: model.display_params,
        default_data_1d: HashMap::new(),
        default_data_2d: HashMap::new(),
    })
}
/// Used by `FitResult.from_json`. A stored fit is a posterior, so a discrete
/// prior in one is a fit that this library's sampler cannot have produced:
/// refusing it here stops fractional draws being restored as Bernoulli or
/// Poisson posterior samples.
pub(super) fn reconstruct(artifact: Artifact) -> PyResult<PyCompiledModel> {
    let model = GraphModel::from_artifact(artifact).map_err(model_error)?;
    from_core(model)
}
