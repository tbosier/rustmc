//! Forecast reconciliation bindings: `Hierarchy`, `ReconciledForecast` and
//! the native entry point behind `rustmc.reconcile`.
//!
//! Every refusal here - a malformed hierarchy, wrongly shaped draws,
//! residuals or history, a method the hierarchy does not support, or weights
//! that cannot be factorized - raises a plain `ValueError`, like the other
//! argument checks forecasting shares. Reconciliation fits no model, so
//! `InferenceError` does not apply.
use crate::data_input::real_numbers;
use crate::forecast_support::{
    interval_probabilities, real_matrix, validate_probability, PyIntervalMatrices,
};
use ndarray::{Array2, Array3, Array4, ArrayD, IxDyn};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyArray4, PyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyString;
use rustmc_core::reconciliation::{
    reconcile as core_reconcile, Hierarchy as CoreHierarchy, NodeDraws,
    ReconciledForecast as CoreReconciledForecast, ReconciliationError, ReconciliationMethod,
};

fn reconciliation_error(error: ReconciliationError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

/// The nodes of a forecast hierarchy and the bottom series each one sums.
///
/// Nodes are ordered aggregates first, then bottom series: the rows of the
/// summing matrix `S`.
#[pyclass(name = "Hierarchy", frozen, module = "rustmc")]
#[derive(Clone)]
pub(crate) struct PyHierarchy {
    pub(crate) inner: CoreHierarchy,
}

#[pymethods]
impl PyHierarchy {
    #[new]
    #[pyo3(signature = (summing_matrix, node_names=None))]
    fn new(summing_matrix: &Bound<'_, PyAny>, node_names: Option<Vec<String>>) -> PyResult<Self> {
        let rows = real_matrix(summing_matrix, "summing_matrix")?;
        CoreHierarchy::from_summing_matrix(&rows, node_names)
            .map(|inner| Self { inner })
            .map_err(reconciliation_error)
    }

    /// A nested hierarchy from each bottom series' labels, top level first.
    #[staticmethod]
    #[pyo3(signature = (labels, *, total="total", separator="/"))]
    fn from_labels(labels: &Bound<'_, PyAny>, total: &str, separator: &str) -> PyResult<Self> {
        if labels.is_instance_of::<PyString>() {
            return Err(PyValueError::new_err(
                "labels must be a sequence of label paths, one per bottom series; got a string",
            ));
        }
        let paths = labels
            .try_iter()
            .map_err(|_| {
                PyValueError::new_err(
                    "labels must be a sequence of label paths, one per bottom series",
                )
            })?
            .enumerate()
            .map(|(index, path)| {
                let path = path?;
                if path.is_instance_of::<PyString>() {
                    return Err(PyValueError::new_err(format!(
                        "labels[{index}] must be a sequence of strings, such as a tuple; got a \
                         single string"
                    )));
                }
                path.extract::<Vec<String>>().map_err(|_| {
                    PyValueError::new_err(format!(
                        "labels[{index}] must be a sequence of strings; convert other labels \
                         with str() first"
                    ))
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        CoreHierarchy::from_labels(&paths, total, separator)
            .map(|inner| Self { inner })
            .map_err(reconciliation_error)
    }

    #[getter]
    fn node_names(&self) -> Vec<String> {
        self.inner.node_names().to_vec()
    }

    #[getter]
    fn aggregate_names(&self) -> Vec<String> {
        self.inner.aggregate_names().to_vec()
    }

    #[getter]
    fn bottom_names(&self) -> Vec<String> {
        self.inner.bottom_names().to_vec()
    }

    #[getter]
    fn node_count(&self) -> usize {
        self.inner.node_count()
    }

    #[getter]
    fn aggregate_count(&self) -> usize {
        self.inner.aggregate_count()
    }

    #[getter]
    fn bottom_count(&self) -> usize {
        self.inner.bottom_count()
    }

    /// The dense `(node, bottom)` summing matrix.
    #[getter]
    fn summing_matrix<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let rows = self.inner.summing_matrix().map_err(reconciliation_error)?;
        let columns = self.inner.bottom_count();
        Ok(
            Array2::from_shape_fn((rows.len(), columns), |(row, column)| rows[row][column])
                .into_pyarray(py),
        )
    }

    #[getter]
    fn is_nested(&self) -> bool {
        self.inner.is_nested()
    }

    /// Each node's depth below the total, or `None` for a grouped structure.
    #[getter]
    fn levels(&self) -> Option<Vec<usize>> {
        self.inner.depths().map(<[usize]>::to_vec)
    }

    /// Each node's parent name (`None` at the top), or `None` for a grouped
    /// structure.
    #[getter]
    fn parents(&self) -> Option<Vec<Option<String>>> {
        let names = self.inner.node_names();
        self.inner.parents().map(|parents| {
            parents
                .iter()
                .map(|parent| parent.map(|index| names[index].clone()))
                .collect()
        })
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .downcast::<Self>()
            .is_ok_and(|other| other.get().inner == self.inner)
    }

    fn __repr__(&self) -> String {
        format!(
            "Hierarchy(nodes={}, aggregates={}, bottom={}, nested={})",
            self.inner.node_count(),
            self.inner.aggregate_count(),
            self.inner.bottom_count(),
            if self.inner.is_nested() {
                "True"
            } else {
                "False"
            }
        )
    }
}

/// Coherent joint draws of every node of a hierarchy.
#[pyclass(name = "ReconciledForecast", frozen, module = "rustmc")]
pub(crate) struct PyReconciledForecast {
    inner: CoreReconciledForecast,
    hierarchy: CoreHierarchy,
}

impl PyReconciledForecast {
    fn node_step_array<'py>(&self, py: Python<'py>, values: Vec<f64>) -> Bound<'py, PyArray2<f64>> {
        let (_, _, nodes, steps) = self.inner.draws.shape();
        Array2::from_shape_vec((nodes, steps), values)
            .expect("core summaries are node-major (node, step)")
            .into_pyarray(py)
    }

    fn quantile_rows<const N: usize>(&self, probabilities: [f64; N]) -> PyResult<[Vec<f64>; N]> {
        for probability in probabilities {
            validate_probability(probability)?;
        }
        let rows: Vec<Vec<f64>> = self
            .inner
            .draws
            .quantiles(&probabilities)
            .map_err(reconciliation_error)?
            .into_iter()
            .map(|quantile| quantile.values)
            .collect();
        Ok(rows
            .try_into()
            .expect("core returns one row per probability"))
    }
}

#[pymethods]
impl PyReconciledForecast {
    #[getter]
    fn chains(&self) -> usize {
        self.inner.draws.shape().0
    }

    #[getter]
    fn draws(&self) -> usize {
        self.inner.draws.shape().1
    }

    #[getter]
    fn node_count(&self) -> usize {
        self.inner.draws.shape().2
    }

    #[getter]
    fn steps(&self) -> usize {
        self.inner.draws.shape().3
    }

    #[getter]
    fn node_names(&self) -> Vec<String> {
        self.hierarchy.node_names().to_vec()
    }

    #[getter]
    fn hierarchy(&self) -> PyHierarchy {
        PyHierarchy {
            inner: self.hierarchy.clone(),
        }
    }

    #[getter]
    fn method(&self) -> &'static str {
        self.inner.method
    }

    /// The MinT shrinkage intensity used, or `None` for other methods.
    #[getter]
    fn shrinkage_intensity(&self) -> Option<f64> {
        self.inner.shrinkage_intensity
    }

    /// Reconciled draws shaped `(chain, draw, node, step)`.
    #[getter]
    fn observation_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray4<f64>> {
        Array4::from_shape_vec(self.inner.draws.shape(), self.inner.draws.values().to_vec())
            .expect("draws are row-major (chain, draw, node, step)")
            .into_pyarray(py)
    }

    /// One node's reconciled draws shaped `(chain, draw, step)`.
    fn node_samples<'py>(
        &self,
        py: Python<'py>,
        name: &str,
    ) -> PyResult<Bound<'py, PyArray3<f64>>> {
        let node = self
            .hierarchy
            .node_names()
            .iter()
            .position(|candidate| candidate == name)
            .ok_or_else(|| {
                PyValueError::new_err(format!("{name:?} is not a node of this hierarchy"))
            })?;
        let (chains, draws, nodes, steps) = self.inner.draws.shape();
        let values = self.inner.draws.values();
        Ok(
            Array3::from_shape_fn((chains, draws, steps), |(chain, draw, step)| {
                values[((chain * draws + draw) * nodes + node) * steps + step]
            })
            .into_pyarray(py),
        )
    }

    /// Mean of every node at every step, shaped `(node, step)`.
    #[getter]
    fn observation_mean<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.node_step_array(py, self.inner.draws.means())
    }

    /// Empirical quantile of every node at every step, shaped `(node, step)`.
    fn observation_quantile<'py>(
        &self,
        py: Python<'py>,
        probability: f64,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let [values] = self.quantile_rows([probability])?;
        Ok(self.node_step_array(py, values))
    }

    /// Pointwise equal-tailed interval, each bound shaped `(node, step)`.
    #[pyo3(signature = (level=0.95))]
    fn interval<'py>(&self, py: Python<'py>, level: f64) -> PyResult<PyIntervalMatrices<'py>> {
        let [lower, upper] = self.quantile_rows(interval_probabilities(level)?)?;
        Ok((
            self.node_step_array(py, lower),
            self.node_step_array(py, upper),
        ))
    }

    #[getter]
    fn interval_kind(&self) -> &'static str {
        "pointwise_equal_tailed"
    }

    fn __repr__(&self) -> String {
        let (chains, draws, nodes, steps) = self.inner.draws.shape();
        format!(
            "ReconciledForecast(method={:?}, chains={chains}, draws={draws}, nodes={nodes}, steps={steps})",
            self.inner.method
        )
    }
}

fn parse_method(
    method: &str,
    residuals: Option<Vec<Vec<f64>>>,
    history: Option<Vec<Vec<f64>>>,
) -> PyResult<ReconciliationMethod> {
    const METHODS: &str = "'bottom_up', 'top_down_average_proportions', \
                           'top_down_forecast_proportions', 'ols', 'wls_struct', 'wls_var' or \
                           'mint_shrink'";
    let uses_residuals = matches!(method, "wls_var" | "mint_shrink");
    let uses_history = method == "top_down_average_proportions";
    if residuals.is_some() && !uses_residuals {
        return Err(PyValueError::new_err(format!(
            "residuals are used only by 'wls_var' and 'mint_shrink', not {method:?}"
        )));
    }
    if history.is_some() && !uses_history {
        return Err(PyValueError::new_err(format!(
            "history is used only by 'top_down_average_proportions', not {method:?}"
        )));
    }
    let residuals = || {
        residuals.clone().ok_or_else(|| {
            PyValueError::new_err(format!(
                "{method:?} needs residuals: in-sample one-step forecast errors shaped \
                 (time, node)"
            ))
        })
    };
    Ok(match method {
        "bottom_up" => ReconciliationMethod::BottomUp,
        "top_down_average_proportions" => ReconciliationMethod::TopDownAverageProportions {
            history: history.ok_or_else(|| {
                PyValueError::new_err(
                    "'top_down_average_proportions' needs history: the observed bottom \
                     series shaped (time, bottom series)",
                )
            })?,
        },
        "top_down_forecast_proportions" => ReconciliationMethod::TopDownForecastProportions,
        "ols" => ReconciliationMethod::Ols,
        "wls_struct" => ReconciliationMethod::WlsStructural,
        "wls_var" => ReconciliationMethod::WlsVariance {
            residuals: residuals()?,
        },
        "mint_shrink" => ReconciliationMethod::MinTShrink {
            residuals: residuals()?,
        },
        _ => {
            return Err(PyValueError::new_err(format!(
                "method must be {METHODS}; got {method:?}"
            )))
        }
    })
}

/// Reconcile `(chain, draw, node, step)` draws. `rustmc.reconcile` adapts
/// forecast objects and name-keyed inputs to this.
#[pyfunction(name = "_reconcile")]
#[pyo3(signature = (draws, hierarchy, method, residuals=None, history=None))]
fn reconcile<'py>(
    py: Python<'py>,
    draws: &Bound<'py, PyAny>,
    hierarchy: PyRef<'py, PyHierarchy>,
    method: &str,
    residuals: Option<&Bound<'py, PyAny>>,
    history: Option<&Bound<'py, PyAny>>,
) -> PyResult<PyReconciledForecast> {
    let residuals = residuals
        .map(|value| real_matrix(value, "residuals"))
        .transpose()?;
    let history = history
        .map(|value| real_matrix(value, "history"))
        .transpose()?;
    let method = parse_method(method, residuals, history)?;
    let (values, shape) = real_numbers(draws, "draws")?;
    let [chains, draw_count, nodes, steps] = shape[..] else {
        return Err(PyValueError::new_err(format!(
            "draws must be four-dimensional (chain, draw, node, step); got {} dimension(s)",
            shape.len()
        )));
    };
    let hierarchy = hierarchy.inner.clone();
    let inner = py
        .allow_threads(|| {
            let base = NodeDraws::new(values, chains, draw_count, nodes, steps)?;
            core_reconcile(&hierarchy, &base, &method)
        })
        .map_err(reconciliation_error)?;
    Ok(PyReconciledForecast { inner, hierarchy })
}

/// A float64 array by the exact conversion every array argument uses, so the
/// Python adapters can stack per-node inputs without NumPy's silent casts.
#[pyfunction(name = "_real_array")]
fn real_array<'py>(
    py: Python<'py>,
    value: &Bound<'py, PyAny>,
    subject: &str,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let (values, shape) = real_numbers(value, subject)?;
    ArrayD::from_shape_vec(IxDyn(&shape), values)
        .map(|array| array.into_pyarray(py))
        .map_err(|error| PyValueError::new_err(format!("{subject}: {error}")))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyHierarchy>()?;
    m.add_class::<PyReconciledForecast>()?;
    m.add_function(wrap_pyfunction!(reconcile, m)?)?;
    m.add_function(wrap_pyfunction!(real_array, m)?)?;
    Ok(())
}
