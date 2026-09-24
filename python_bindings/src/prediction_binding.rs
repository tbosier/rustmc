//! Python input adapter for the shared native prediction binder.
use crate::data_input::{data_inputs_from_maps, parse_data_dict};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rustmc_core::data::DataInputs;

/// New-data inputs for `ModelFit::prediction_graph`; `None` predicts at the
/// training data.
pub(crate) fn prediction_inputs(data: Option<&Bound<'_, PyDict>>) -> PyResult<Option<DataInputs>> {
    data.map(|data| {
        let (one_d, two_d) = parse_data_dict(data)?;
        Ok(data_inputs_from_maps(&one_d, &two_d))
    })
    .transpose()
}
