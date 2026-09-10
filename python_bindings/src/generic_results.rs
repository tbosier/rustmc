//! Shared generic-result storage and Python diagnostics conversion.
use super::{FitResult, ModelSpec};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rustmc_core::{data::DataBinding, graph::Graph, sampler::SampleResult};
use std::sync::Arc;

#[derive(Clone)]
pub(super) enum StoredBatchFit {
    Ready(Arc<FitResult>),
    Bound(Arc<BoundBatchFit>),
}
impl From<Arc<FitResult>> for StoredBatchFit {
    fn from(fit: Arc<FitResult>) -> Self {
        Self::Ready(fit)
    }
}
pub(super) struct BoundBatchFit {
    pub(super) structure: Arc<Graph>,
    pub(super) binding: DataBinding,
    pub(super) raw_result: SampleResult,
    pub(super) display_result: SampleResult,
    pub(super) likelihood_names: Vec<String>,
    pub(super) definition: ModelSpec,
}
impl StoredBatchFit {
    pub(super) fn materialize(&self) -> FitResult {
        match self {
            Self::Ready(fit) => (**fit).clone(),
            Self::Bound(fit) => FitResult {
                definition: fit.definition.clone(),
                graph: fit.structure.with_binding(&fit.binding),
                raw_result: fit.raw_result.clone(),
                display_result: fit.display_result.clone(),
                likelihood_names: fit.likelihood_names.clone(),
            },
        }
    }
    pub(super) fn raw(&self) -> &SampleResult {
        match self {
            Self::Ready(fit) => &fit.raw_result,
            Self::Bound(fit) => &fit.raw_result,
        }
    }
    pub(super) fn display(&self) -> &SampleResult {
        match self {
            Self::Ready(fit) => &fit.display_result,
            Self::Bound(fit) => &fit.display_result,
        }
    }
}

pub(super) fn diagnostics<'py>(
    sample: &SampleResult,
    py: Python<'py>,
) -> PyResult<Bound<'py, PyList>> {
    let report = sample.diagnostics();
    let items: Vec<Bound<'py, PyDict>> = report
        .params
        .iter()
        .map(|p| {
            let d = PyDict::new(py);
            d.set_item("name", &p.name).unwrap();
            d.set_item("mean", p.mean).unwrap();
            d.set_item("std", p.std).unwrap();
            d.set_item("hdi_3%", p.hdi_3).unwrap();
            d.set_item("hdi_97%", p.hdi_97).unwrap();
            d.set_item("ess_bulk", p.ess_bulk).unwrap();
            d.set_item("ess_tail", p.ess_tail).unwrap();
            d.set_item("r_hat", p.r_hat).unwrap();
            d.set_item("mcse_mean", p.mcse_mean).unwrap();
            d
        })
        .collect();
    let list = PyList::new(py, &items)?;
    Ok(list)
}

pub(super) fn transition_diagnostics<'py>(
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
