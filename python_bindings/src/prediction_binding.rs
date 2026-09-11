//! Python input adapter for the shared native prediction binder.
use super::{data_inputs_from_maps, model_error, parse_data_dict};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rustmc_core::graph::Graph;
use std::collections::HashMap;
pub(super) fn prediction_graph(
    graph: &Graph,
    data: Option<&Bound<'_, PyDict>>,
    sizes: Option<HashMap<String, usize>>,
) -> PyResult<Graph> {
    if data.is_none() && sizes.is_none() {
        return Ok(graph.clone());
    }
    let (one_d, two_d) = match data {
        Some(data) => parse_data_dict(data)?,
        None => (HashMap::new(), HashMap::new()),
    };
    rustmc_core::model::bind_prediction(
        graph,
        data_inputs_from_maps(&one_d, &two_d),
        sizes.unwrap_or_default(),
    )
    .map_err(model_error)
}
