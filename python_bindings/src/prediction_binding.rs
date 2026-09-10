//! Prediction-only data preparation for fitted graph models.
use super::{core_binding_from_maps, parse_data_dict};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rustmc_core::graph::Graph;
use std::collections::HashMap;

/// Bind prediction inputs to fitted parameter identities. Response placeholders are
/// internal evaluator storage only and never required from the prediction caller.
pub(super) fn prediction_graph(
    graph: &Graph,
    data: Option<&Bound<'_, PyDict>>,
    sizes: Option<HashMap<String, usize>>,
) -> PyResult<Graph> {
    if data.is_none() && sizes.is_none() {
        return Ok(graph.clone());
    }
    let (mut one_d, two_d) = match data {
        Some(data) => parse_data_dict(data)?,
        None => (HashMap::new(), HashMap::new()),
    };
    let mut lengths = sizes.unwrap_or_default();
    for dimension in lengths.keys() {
        if !graph
            .schema
            .vectors
            .iter()
            .chain(&graph.schema.observations)
            .chain(&graph.schema.matrices)
            .any(|slot| &slot.dim == dimension)
        {
            return Err(PyValueError::new_err(format!(
                "unknown prediction dimension '{dimension}'"
            )));
        }
    }
    for slot in graph.schema.vectors.iter().chain(&graph.schema.matrices) {
        let len = one_d
            .get(&slot.key)
            .map(Vec::len)
            .or_else(|| two_d.get(&slot.key).map(|(_, n, _)| *n))
            .ok_or_else(|| {
                PyValueError::new_err(format!("missing prediction data key '{}'", slot.key))
            })?;
        if let Some(n) = lengths.insert(slot.dim.clone(), len) {
            if n != len {
                return Err(PyValueError::new_err(format!(
                    "prediction dimension '{}' has inconsistent lengths",
                    slot.dim
                )));
            }
        }
    }
    for (i, slot) in graph.schema.observations.iter().enumerate() {
        // A response key that is also a predictor must be supplied as that predictor.
        if graph.schema.vectors.iter().any(|s| s.key == slot.key) {
            continue;
        }
        let n = lengths
            .get(&slot.dim)
            .copied()
            .or_else(|| graph.obs_vectors.get(i).map(Vec::len))
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "supply size for prediction dimension '{}'",
                    slot.dim
                ))
            })?;
        if n == 0 {
            return Err(PyValueError::new_err(
                "prediction dimensions must be positive",
            ));
        }
        one_d.insert(slot.key.clone(), vec![1.0; n]);
    }
    let binding = core_binding_from_maps(
        &graph.schema,
        &one_d,
        &two_d,
        "prediction".into(),
        true,
        true,
    )?;
    let bound = graph.with_binding(&binding);
    bound
        .validate_shapes()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(bound)
}
