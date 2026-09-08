use ndarray::{Array2, Array3, Array4};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyArray4, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rustmc_core::runoff::{fit_runoff, PaymentTriangle, RunoffConfig, RunoffPosterior};

/// Pooled Dirichlet lag probabilities for integer payment-event counts.
/// The last alpha entry is an unscheduled tail. Unknown ultimate counts use an
/// independent Gamma(total_shape, rate=total_rate) Poisson intensity per cohort.
/// This model does not accept currency as multinomial event counts.
#[pyclass(name = "DirichletMultinomialRunoff")]
pub struct PyRunoff {
    alpha: Vec<f64>,
    total_shape: f64,
    total_rate: f64,
}

#[pymethods]
impl PyRunoff {
    #[new]
    #[pyo3(signature = (alpha, *, total_shape=2.0, total_rate=0.1))]
    fn new(alpha: Vec<f64>, total_shape: f64, total_rate: f64) -> PyResult<Self> {
        if alpha.len() < 2
            || alpha.iter().any(|a| !a.is_finite() || *a <= 0.0)
            || !alpha.iter().sum::<f64>().is_finite()
            || !total_shape.is_finite()
            || total_shape <= 0.0
            || !total_rate.is_finite()
            || total_rate <= 0.0
        {
            return Err(PyValueError::new_err("alpha requires at least two finite positive entries; total_shape and total_rate must be finite and positive"));
        }
        Ok(Self {
            alpha,
            total_shape,
            total_rate,
        })
    }

    /// Fit incremental counts[cohort, lag], with NaN for future/unobserved cells.
    /// Elapsed regular lags must be observed, including exact zeros. The last column
    /// is an unobserved tail unless explicitly closed. Origins/valuation use integer
    /// periods and lag zero occurs at origin. None totals infer unknown ultimates.
    /// All-known totals use exact independent draws; warmup is ignored in that case.
    #[pyo3(signature = (counts, origins, valuation, totals=None, *, draws=1000, warmup=500, chains=4, seed=42))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        &self,
        py: Python<'_>,
        counts: PyReadonlyArray2<'_, f64>,
        origins: Vec<i64>,
        valuation: i64,
        totals: Option<Vec<Option<u64>>>,
        draws: usize,
        warmup: usize,
        chains: usize,
        seed: u64,
    ) -> PyResult<PyRunoffFit> {
        let counts = counts.as_array().rows().into_iter().map(|row| {
            row.iter().map(|n| {
                if n.is_nan() {
                    Ok(None)
                } else if !n.is_finite() || *n < 0.0 || n.fract() != 0.0 || *n > ((1_u64 << 53) - 1) as f64 {
                    Err(PyValueError::new_err("counts must be nonnegative integers <= 2**53 - 1, or NaN for unobserved cells"))
                } else {
                    Ok(Some(*n as u64))
                }
            }).collect::<PyResult<Vec<_>>>()
        }).collect::<PyResult<Vec<_>>>()?;
        let triangle = PaymentTriangle {
            known_totals: totals.unwrap_or_else(|| vec![None; counts.len()]),
            counts,
            origins,
            valuation,
        };
        let config = RunoffConfig {
            alpha: self.alpha.clone(),
            total_shape: self.total_shape,
            total_rate: self.total_rate,
            draws,
            warmup,
            chains,
            seed,
        };
        let inner = py
            .allow_threads(|| fit_runoff(&triangle, &config))
            .map_err(PyValueError::new_err)?;
        Ok(PyRunoffFit { inner })
    }
}

#[pyclass(name = "RunoffFit")]
pub struct PyRunoffFit {
    inner: RunoffPosterior,
}

fn array3<'py, T: numpy::Element>(
    py: Python<'py>,
    data: Vec<Vec<Vec<T>>>,
) -> Bound<'py, PyArray3<T>> {
    let shape = (data.len(), data[0].len(), data[0][0].len());
    Array3::from_shape_vec(shape, data.into_iter().flatten().flatten().collect())
        .unwrap()
        .into_pyarray(py)
}

#[pymethods]
impl PyRunoffFit {
    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        crate::forecast_diagnostics::diagnostics_list(py, &self.inner.diagnostics())
    }

    fn summary(&self) -> String {
        self.inner
            .diagnostics()
            .to_table_with_sampler(Some(&format!("Sampler: {}", self.sampler())))
    }

    fn sampler_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        crate::forecast_diagnostics::sampler_stats(
            py,
            self.sampler(),
            self.inner.chains.len(),
            self.inner.chains[0].len(),
            "shared lag probabilities; unknown cohort intensities and ultimate counts",
        )
    }

    /// Complete integer allocations, shape (chain, draw, cohort, lag including tail).
    #[getter]
    fn allocation_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray4<u64>> {
        let shape = (
            self.inner.chains.len(),
            self.inner.chains[0].len(),
            self.inner.triangle.counts.len(),
            self.inner.triangle.counts[0].len(),
        );
        let data = self
            .inner
            .chains
            .iter()
            .flatten()
            .flat_map(|d| d.counts.iter().flatten().copied())
            .collect();
        Array4::from_shape_vec(shape, data)
            .unwrap()
            .into_pyarray(py)
    }

    /// Shared lag probabilities, shape (chain, draw, lag including tail).
    #[getter]
    fn lag_probability_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        array3(
            py,
            self.inner
                .chains
                .iter()
                .map(|c| c.iter().map(|d| d.lag_probabilities.clone()).collect())
                .collect(),
        )
    }

    /// Ultimate event counts, shape (chain, draw, cohort).
    #[getter]
    fn ultimate_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<u64>> {
        array3(
            py,
            self.inner
                .chains
                .iter()
                .map(|c| c.iter().map(|d| d.totals.clone()).collect())
                .collect(),
        )
    }

    /// Gamma-Poisson intensity draws; known-total cohorts have NaN, not a parameter.
    #[getter]
    fn intensity_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        array3(
            py,
            self.inner
                .chains
                .iter()
                .map(|c| {
                    c.iter()
                        .map(|d| {
                            d.intensities
                                .iter()
                                .map(|v| v.unwrap_or(f64::NAN))
                                .collect()
                        })
                        .collect()
                })
                .collect(),
        )
    }

    /// Remaining unscheduled tail counts, shape (chain, draw, cohort).
    #[getter]
    fn tail_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<u64>> {
        array3(py, self.inner.tail_samples())
    }

    /// Aggregate future regular-lag paths at valuation+1,...,valuation+steps.
    /// Tail counts and regular cells beyond steps are excluded; use tail_samples
    /// and allocation_samples to account for the full reserve.
    fn calendar_samples<'py>(
        &self,
        py: Python<'py>,
        steps: usize,
    ) -> PyResult<Bound<'py, PyArray3<u64>>> {
        let values = self
            .inner
            .calendar_samples(steps)
            .map_err(PyValueError::new_err)?;
        Ok(array3(py, values))
    }

    #[getter]
    fn observed_mask<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<bool>> {
        let rows = &self.inner.triangle.counts;
        Array2::from_shape_vec(
            (rows.len(), rows[0].len()),
            rows.iter().flatten().map(Option::is_some).collect(),
        )
        .unwrap()
        .into_pyarray(py)
    }

    #[getter]
    fn origins(&self) -> Vec<i64> {
        self.inner.triangle.origins.clone()
    }

    #[getter]
    fn valuation(&self) -> i64 {
        self.inner.triangle.valuation
    }

    #[getter]
    fn sampler(&self) -> &'static str {
        if self.inner.exact {
            "independent_conjugate"
        } else {
            "latent_count_gibbs"
        }
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRunoff>()?;
    m.add_class::<PyRunoffFit>()?;
    Ok(())
}
