//! Shared non-Hamiltonian diagnostic conversion.
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rustmc_core::diagnostics::DiagnosticsReport;

pub(crate) fn diagnostics_list<'py>(
    py: Python<'py>,
    report: &DiagnosticsReport,
) -> PyResult<Bound<'py, PyList>> {
    let items = PyList::empty(py);
    for parameter in &report.params {
        let item = PyDict::new(py);
        item.set_item("name", &parameter.name)?;
        for (key, value) in [
            ("mean", parameter.mean),
            ("std", parameter.std),
            ("hdi_3%", parameter.hdi_3),
            ("hdi_97%", parameter.hdi_97),
            ("ess_bulk", parameter.ess_bulk),
            ("ess_tail", parameter.ess_tail),
            ("r_hat", parameter.r_hat),
            ("mcse_mean", parameter.mcse_mean),
        ] {
            item.set_item(key, value.is_finite().then_some(value))?;
        }
        // Infinity means detected nonconvergence, not insufficient information.
        if parameter.r_hat.is_infinite() {
            item.set_item("r_hat", parameter.r_hat)?;
        }
        items.append(item)?;
    }
    Ok(items)
}

pub(crate) fn sampler_stats<'py>(
    py: Python<'py>,
    sampler: &str,
    chains: usize,
    draws: usize,
    coverage: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let stats = PyDict::new(py);
    stats.set_item("sampler", sampler)?;
    stats.set_item("chains", chains)?;
    stats.set_item("draws", draws)?;
    stats.set_item("divergences", py.None())?;
    stats.set_item("acceptance_rate", py.None())?;
    stats.set_item("numerical_failures", 0)?;
    stats.set_item("diagnostic_coverage", coverage)?;
    stats.set_item("r_hat_method", "rank-normalized folded split")?;
    stats.set_item("independent_chain_comparison", chains >= 2)?;
    stats.set_item("diagnostics_available", draws >= 6)?;
    Ok(stats)
}
