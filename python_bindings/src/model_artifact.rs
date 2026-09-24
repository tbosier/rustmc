//! Python adapter for the native graph-model artifact.
use crate::builder::reject_discrete_priors_for_gradient_sampling;
use crate::compiled::PyCompiledModel;
use crate::model_error;
use pyo3::prelude::*;
use rustmc_core::model::GraphModel;
use std::collections::HashMap;

pub(crate) fn encode(model: &PyCompiledModel) -> PyResult<String> {
    model.model.to_json().map_err(model_error)
}

pub(crate) fn decode(text: &str) -> PyResult<PyCompiledModel> {
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
/// (A stored *fit* is refused by the core itself: `ModelFit::from_json` rejects
/// a discrete prior, which no gradient sampler can have produced.)
fn from_core(model: GraphModel) -> PyResult<PyCompiledModel> {
    reject_discrete_priors_for_gradient_sampling(&model.definition.priors)?;
    Ok(PyCompiledModel {
        model,
        default_data_1d: HashMap::new(),
        default_data_2d: HashMap::new(),
    })
}
