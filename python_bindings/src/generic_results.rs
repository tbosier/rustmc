//! Result exports and diagnostics shared by `FitResult` and `BatchResult`.
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rustmc_core::sampler::SampleResult;

/// `{name: value}` for per-parameter summaries such as means.
pub(crate) fn named_values<'py>(
    py: Python<'py>,
    names: &[String],
    values: Vec<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    for (name, value) in names.iter().zip(values) {
        dict.set_item(name, value)?;
    }
    Ok(dict)
}

/// Every parameter's draws, chains concatenated.
pub(crate) fn samples_flat<'py>(
    sample: &SampleResult,
    py: Python<'py>,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    for (index, name) in sample.param_names.iter().enumerate() {
        let values: Vec<f64> = sample
            .samples
            .iter()
            .flatten()
            .map(|draw| draw[index])
            .collect();
        dict.set_item(name, PyArray1::from_vec(py, values))?;
    }
    Ok(dict)
}

/// Every parameter's draws as a `(chain, draw)` array.
pub(crate) fn samples_by_chain<'py>(
    sample: &SampleResult,
    py: Python<'py>,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    let n_chains = sample.samples.len();
    let n_draws = sample.samples.first().map_or(0, Vec::len);
    for (index, name) in sample.param_names.iter().enumerate() {
        let mut values = Array2::<f64>::zeros((n_chains, n_draws));
        for (chain_index, chain) in sample.samples.iter().enumerate() {
            for (draw_index, draw) in chain.iter().enumerate() {
                values[[chain_index, draw_index]] = draw[index];
            }
        }
        dict.set_item(name, values.into_pyarray(py))?;
    }
    Ok(dict)
}

/// Per-parameter diagnostics, one dict per parameter.
///
/// A value the draws cannot support (R-hat or ESS from too few draws, say)
/// is `None`, through the same conversion the forecasting fits use, so
/// every fit reports unavailable diagnostics alike.
pub(crate) fn diagnostics<'py>(
    sample: &SampleResult,
    py: Python<'py>,
) -> PyResult<Bound<'py, PyList>> {
    crate::forecast_diagnostics::diagnostics_list(py, &sample.diagnostics())
}

pub(crate) fn transition_diagnostics<'py>(
    sample: &SampleResult,
    py: Python<'py>,
) -> PyResult<Bound<'py, PyDict>> {
    let report = sample.transition_diagnostics();
    let result = PyDict::new(py);
    result.set_item("total_transitions", report.total_transitions)?;
    result.set_item("total_warmup_transitions", report.total_warmup_transitions)?;
    result.set_item("total_draw_transitions", report.total_draw_transitions)?;
    result.set_item("total_divergences", report.total_divergences)?;
    result.set_item("total_leapfrog_steps", report.total_leapfrog_steps)?;
    result.set_item("mean_accept_prob", report.mean_accept_prob)?;
    result.set_item("mean_energy_error", report.mean_energy_error)?;
    result.set_item("max_abs_energy_error", report.max_abs_energy_error)?;

    let chains = PyList::empty(py);
    for chain in report.chains {
        let item = PyDict::new(py);
        item.set_item("chain", chain.chain_index)?;
        item.set_item("transitions", chain.num_transitions)?;
        item.set_item("warmup_transitions", chain.num_warmup_transitions)?;
        item.set_item("draw_transitions", chain.num_draw_transitions)?;
        item.set_item("divergences", chain.divergences)?;
        item.set_item("accepted_transitions", chain.accepted_transitions)?;
        item.set_item("mean_accept_prob", chain.mean_accept_prob)?;
        item.set_item("mean_energy_error", chain.mean_energy_error)?;
        item.set_item("max_abs_energy_error", chain.max_abs_energy_error)?;
        item.set_item("mean_step_size", chain.mean_step_size)?;
        item.set_item("max_tree_depth", chain.max_tree_depth)?;
        item.set_item("total_leapfrog_steps", chain.total_leapfrog_steps)?;
        chains.append(item)?;
    }
    result.set_item("chains", chains)?;
    Ok(result)
}
