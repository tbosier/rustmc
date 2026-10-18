//! `DesignTerm` and `TimeDesign`: named intervention, changepoint, Fourier
//! and regressor columns whose future rows continue the training rows.
use crate::forecast_support::real_vector;
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySlice};
use rustmc_core::time_design::{
    ChangepointPlacement, DesignError, DesignTerm, TermKind, TimeDesign,
};
use std::collections::BTreeMap;

/// Design arguments are shared argument checks, like `fourier_design`'s.
fn design_error(error: DesignError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

/// One named block of regression columns on the integer calendar.
#[pyclass(name = "DesignTerm", frozen, module = "rustmc")]
#[derive(Clone)]
pub(crate) struct PyDesignTerm {
    inner: DesignTerm,
}

fn term(name: String, kind: TermKind) -> PyResult<PyDesignTerm> {
    Ok(PyDesignTerm {
        inner: DesignTerm::new(name, kind).map_err(design_error)?,
    })
}

#[pymethods]
impl PyDesignTerm {
    /// Level shift: 1 from position `at` onward, 0 before.
    #[staticmethod]
    fn step(name: String, at: i64) -> PyResult<Self> {
        term(name, TermKind::Step { at })
    }
    /// One-period effect: 1 at position `at`, 0 elsewhere.
    #[staticmethod]
    fn pulse(name: String, at: i64) -> PyResult<Self> {
        term(name, TermKind::Pulse { at })
    }
    /// Slope change: max(0, t - at).
    #[staticmethod]
    fn ramp(name: String, at: i64) -> PyResult<Self> {
        term(name, TermKind::Ramp { at })
    }
    /// Temporary effect: rate**(t - at) from `at` onward, 0 before.
    #[staticmethod]
    fn decay(name: String, at: i64, rate: f64) -> PyResult<Self> {
        term(name, TermKind::Decay { at, rate })
    }
    /// Sine/cosine columns, as `fourier_design`.
    #[staticmethod]
    fn fourier(name: String, period: usize, harmonics: usize) -> PyResult<Self> {
        term(name, TermKind::Fourier { period, harmonics })
    }
    /// One ramp column per changepoint, at explicit positions or on a grid.
    /// `fraction` defaults to 0.8, Prophet's default changepoint range.
    #[staticmethod]
    #[pyo3(signature=(name, *, at=None, count=None, first=None, last=None, fraction=None))]
    fn changepoints(
        name: String,
        at: Option<Vec<i64>>,
        count: Option<usize>,
        first: Option<i64>,
        last: Option<i64>,
        fraction: Option<f64>,
    ) -> PyResult<Self> {
        let placement = match (at, count) {
            (Some(positions), None) => {
                if first.is_some() || last.is_some() || fraction.is_some() {
                    return Err(PyValueError::new_err(format!(
                        "design term '{name}': first, last and fraction place a grid; with \
                         explicit positions in `at`, leave them unset"
                    )));
                }
                ChangepointPlacement::At(positions)
            }
            (None, Some(count)) => ChangepointPlacement::Grid {
                count,
                first,
                last,
                fraction: fraction.unwrap_or(0.8),
            },
            _ => {
                return Err(PyValueError::new_err(format!(
                    "design term '{name}': give exactly one of `at` (positions) or `count` \
                     (a grid)"
                )))
            }
        };
        term(name, TermKind::Changepoints(placement))
    }
    /// One column whose values are passed to `training` and `future`.
    #[staticmethod]
    fn regressor(name: String) -> PyResult<Self> {
        term(name, TermKind::Regressor)
    }
    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }
    fn __repr__(&self) -> String {
        let name = &self.inner.name;
        match &self.inner.kind {
            TermKind::Step { at } => format!("DesignTerm.step({name:?}, at={at})"),
            TermKind::Pulse { at } => format!("DesignTerm.pulse({name:?}, at={at})"),
            TermKind::Ramp { at } => format!("DesignTerm.ramp({name:?}, at={at})"),
            TermKind::Decay { at, rate } => {
                format!("DesignTerm.decay({name:?}, at={at}, rate={rate})")
            }
            TermKind::Fourier { period, harmonics } => {
                format!("DesignTerm.fourier({name:?}, period={period}, harmonics={harmonics})")
            }
            TermKind::Changepoints(ChangepointPlacement::At(positions)) => {
                format!("DesignTerm.changepoints({name:?}, at={positions:?})")
            }
            TermKind::Changepoints(ChangepointPlacement::Grid {
                count,
                first,
                last,
                fraction,
            }) => {
                let bound = |value: &Option<i64>| value.map_or("None".into(), |v| v.to_string());
                format!(
                    "DesignTerm.changepoints({name:?}, count={count}, first={}, last={}, \
                     fraction={fraction})",
                    bound(first),
                    bound(last)
                )
            }
            TermKind::Regressor => format!("DesignTerm.regressor({name:?})"),
        }
    }
}

/// Named regression columns for a training window and the horizon after it.
#[pyclass(name = "TimeDesign", frozen, module = "rustmc")]
pub(crate) struct PyTimeDesign {
    inner: TimeDesign,
}

fn regressor_values(
    regressors: Option<&Bound<'_, PyDict>>,
    window: &str,
) -> PyResult<BTreeMap<String, Vec<f64>>> {
    let mut values = BTreeMap::new();
    if let Some(regressors) = regressors {
        for (key, value) in regressors.iter() {
            let key: String = key.extract().map_err(|_| {
                PyValueError::new_err("regressors must be keyed by regressor term name")
            })?;
            let column = real_vector(&value, &format!("{window} values of regressor '{key}'"))?;
            values.insert(key, column);
        }
    }
    Ok(values)
}

fn to_array<'py>(
    py: Python<'py>,
    rows: Vec<Vec<f64>>,
    width: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    Array2::from_shape_vec((rows.len(), width), rows.concat())
        .map(|design| design.into_pyarray(py))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

#[pymethods]
impl PyTimeDesign {
    #[new]
    #[pyo3(signature=(terms, length, *, start=0))]
    fn new(terms: Vec<PyRef<'_, PyDesignTerm>>, length: usize, start: i64) -> PyResult<Self> {
        let terms = terms.iter().map(|term| term.inner.clone()).collect();
        Ok(Self {
            inner: TimeDesign::new(terms, start, length).map_err(design_error)?,
        })
    }
    /// Column names in column order.
    #[getter]
    fn names(&self) -> Vec<String> {
        self.inner.names().to_vec()
    }
    #[getter]
    fn width(&self) -> usize {
        self.inner.width()
    }
    /// Number of training rows.
    #[getter]
    fn length(&self) -> usize {
        self.inner.length()
    }
    /// Calendar position of the first training row.
    #[getter]
    fn start(&self) -> i64 {
        self.inner.start()
    }
    #[getter]
    fn terms(&self) -> Vec<PyDesignTerm> {
        self.inner
            .terms()
            .iter()
            .map(|term| PyDesignTerm {
                inner: term.clone(),
            })
            .collect()
    }
    /// The resolved positions of each changepoint term.
    #[getter]
    fn changepoints(&self) -> BTreeMap<String, Vec<i64>> {
        self.inner
            .changepoints()
            .into_iter()
            .map(|(name, positions)| (name.to_string(), positions.to_vec()))
            .collect()
    }
    /// Names of the terms whose values `training` and `future` need.
    #[getter]
    fn regressor_names(&self) -> Vec<String> {
        self.inner
            .regressor_names()
            .into_iter()
            .map(str::to_string)
            .collect()
    }
    /// The columns of one term, as a slice into a row.
    fn term_slice<'py>(&self, py: Python<'py>, term: &str) -> PyResult<Bound<'py, PySlice>> {
        self.inner
            .term_columns()
            .into_iter()
            .find(|(name, _, _)| *name == term)
            .map(|(_, offset, width)| {
                PySlice::new(py, offset as isize, (offset + width) as isize, 1)
            })
            .ok_or_else(|| PyValueError::new_err(format!("this design has no term named '{term}'")))
    }
    /// The training rows, shaped (length, width).
    #[pyo3(signature=(*, regressors=None))]
    fn training<'py>(
        &self,
        py: Python<'py>,
        regressors: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let values = regressor_values(regressors, "training")?;
        let rows = self.inner.training_rows(&values).map_err(design_error)?;
        to_array(py, rows, self.inner.width())
    }
    /// The `steps` rows right after the training window, shaped (steps, width).
    #[pyo3(signature=(steps, *, regressors=None))]
    fn future<'py>(
        &self,
        py: Python<'py>,
        steps: usize,
        regressors: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let values = regressor_values(regressors, "future")?;
        let rows = self
            .inner
            .future_rows(steps, &values)
            .map_err(design_error)?;
        to_array(py, rows, self.inner.width())
    }
    fn __repr__(&self) -> String {
        format!(
            "TimeDesign(start={}, length={}, names={:?})",
            self.inner.start(),
            self.inner.length(),
            self.inner.names()
        )
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDesignTerm>()?;
    m.add_class::<PyTimeDesign>()?;
    Ok(())
}
