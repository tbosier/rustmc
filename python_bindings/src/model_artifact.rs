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
    Ok(from_core(model))
}
fn from_core(model: GraphModel) -> PyCompiledModel {
    PyCompiledModel {
        definition: ModelSpec(model.definition),
        structure: model.structure,
        likelihood_names: model.likelihood_names,
        display_params: model.display_params,
        default_data_1d: HashMap::new(),
        default_data_2d: HashMap::new(),
    }
}
pub(super) fn reconstruct(artifact: Artifact) -> PyResult<PyCompiledModel> {
    GraphModel::from_artifact(artifact)
        .map(from_core)
        .map_err(model_error)
}
