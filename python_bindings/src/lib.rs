// Graph-model surface: builder, compiled model, fits and batches.
mod arviz;
mod batch;
mod builder;
mod compiled;
mod data_input;
mod errors;
mod expressions;
mod fit_result;
mod generic_results;
mod model_artifact;
mod prediction_binding;
mod sampling;
// Shared with the forecasting modules, which name them from the crate root.
pub(crate) use arviz::arviz_from_groups;
pub(crate) use errors::{
    model_error, param_error, InferenceError, ParameterError, StateSpaceError,
};

// Forecasting surface: one module per model family.
mod ar;
mod decision;
mod dynamic_glm;
mod forecast_batch;
mod forecast_diagnostics;
mod forecast_evaluation;
mod forecast_support;
mod hierarchical;
mod hurdle;
mod local_level;
mod reconciliation;
mod regression;
mod runoff;
mod seasonal;
mod state_space;
mod structural;
mod time_design;
mod trend;

use pyo3::prelude::*;

/// How many fused grouped-Normal forward and reverse passes this process's
/// graph evaluators have run, counted when each evaluator is dropped. Tests
/// and benchmarks read it to show that a fit took the fused pass.
#[pyfunction]
fn _fused_grouped_normal_passes() -> (u64, u64) {
    rustmc_core::autodiff::fused_grouped_normal_passes()
}

/// Allow or forbid fused grouped-Normal chains in every graph evaluator of
/// this process, and return whether they were allowed. Tests use it to
/// compare an output computed through the fused chains with the same output
/// computed through the separate nodes.
#[pyfunction]
fn _set_grouped_normal_fusion(enabled: bool) -> bool {
    rustmc_core::autodiff::set_grouped_normal_fusion(enabled)
}

#[pymodule]
fn _rustmc(m: &Bound<'_, PyModule>) -> PyResult<()> {
    dynamic_glm::register(m)?;
    hurdle::register(m)?;
    regression::register(m)?;
    structural::register(m)?;
    runoff::register(m)?;
    forecast_batch::register(m)?;
    state_space::register(m)?;
    forecast_support::register(m)?;
    forecast_evaluation::register(m)?;
    hierarchical::register(m)?;
    local_level::register(m)?;
    seasonal::register(m)?;
    trend::register(m)?;
    ar::register(m)?;
    time_design::register(m)?;
    decision::register(m)?;
    reconciliation::register(m)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<builder::ModelBuilder>()?;
    m.add_class::<builder::ModelSpec>()?;
    m.add_class::<expressions::ParamRef>()?;
    m.add_class::<expressions::VectorParamRef>()?;
    m.add_class::<expressions::Expr>()?;
    m.add_class::<fit_result::FitResult>()?;
    m.add_class::<batch::BatchResult>()?;
    m.add_class::<compiled::PyCompiledModel>()?;
    m.add_class::<compiled::PyBoundModel>()?;
    m.add_class::<batch::PyBatchFit>()?;
    m.add("ParameterError", m.py().get_type::<ParameterError>())?;
    m.add("StateSpaceError", m.py().get_type::<StateSpaceError>())?;
    m.add("InferenceError", m.py().get_type::<InferenceError>())?;
    m.add_function(wrap_pyfunction!(sampling::sample, m)?)?;
    m.add_function(wrap_pyfunction!(batch::batch_sample, m)?)?;
    m.add_function(wrap_pyfunction!(sampling::sample_prior_predictive, m)?)?;
    m.add_function(wrap_pyfunction!(_fused_grouped_normal_passes, m)?)?;
    m.add_function(wrap_pyfunction!(_set_grouped_normal_fusion, m)?)?;
    Ok(())
}
