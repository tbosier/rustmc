//! Python exception types and the conversions into them.
use pyo3::create_exception;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rustmc_core::param_ref::ParamRefError;

pub(crate) fn model_error(error: rustmc_core::model::ModelError) -> PyErr {
    match error {
        rustmc_core::model::ModelError::Invalid(s) => PyValueError::new_err(s),
        rustmc_core::model::ModelError::Parameter(s) => ParameterError::new_err(s),
    }
}

create_exception!(
    rustmc,
    ParameterError,
    PyValueError,
    "Raised when a parameter reference in a model cannot be resolved.\n\n\
     Subclasses ``ValueError`` for backwards compatibility."
);

create_exception!(
    rustmc,
    StateSpaceError,
    PyValueError,
    "Raised when state-space inputs or numerical updates are invalid."
);

create_exception!(
    rustmc,
    InferenceError,
    PyValueError,
    "Raised when a fitted Bayesian model has invalid inputs or a numerical failure."
);

/// Convert a core parameter-resolution failure into the Python exception.
pub(crate) fn param_error(err: ParamRefError) -> PyErr {
    ParameterError::new_err(err.to_string())
}
