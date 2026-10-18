//! Bindings for `rustmc_core::forecast_evaluation`, the calibration and point
//! accuracy statistics behind `rustmc.evaluation`.
//!
//! The functions here are private: `rustmc.evaluation` chooses which forecasts
//! to pool and presents the results, and every statistic is computed in the
//! core. Each forecast is a `(samples, actual)` pair in the layout the other
//! scores take - `samples` shaped `(sample, *target)` or
//! `(chain, draw, *target)` and `actual` shaped `target` - read through the one
//! exact conversion in [`real_numbers`]. Invalid arguments raise plain
//! `ValueError`, like the other argument checks every model shares.
use crate::data_input::real_numbers;
use ndarray::{ArrayD, IxDyn};
use numpy::IntoPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use rustmc_core::forecast_evaluation::{
    mase_scale, pit_histogram as core_pit_histogram, validate_levels, EvaluationError,
    IntervalCalibration, Point, PointAccuracy, TargetDraws,
};

fn value_error(error: EvaluationError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

/// Central-interval levels: a one-dimensional real array-like, nonempty,
/// unique, and strictly between zero and one.
fn read_levels(levels: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    let (values, shape) = real_numbers(levels, "levels")?;
    if shape.len() != 1 {
        return Err(PyValueError::new_err(
            "levels must be a nonempty sequence strictly between zero and one",
        ));
    }
    validate_levels(&values).map_err(value_error)?;
    Ok(values)
}

/// The validated levels as a list, for callers that label results by level.
#[pyfunction(name = "_interval_levels")]
fn interval_levels(levels: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    read_levels(levels)
}

/// One forecast's draws, target by target, with its outcomes and their shape.
struct Forecast {
    draws: TargetDraws,
    actual: Vec<f64>,
    shape: Vec<usize>,
}

impl Forecast {
    fn read(samples: &Bound<'_, PyAny>, actual: &Bound<'_, PyAny>) -> PyResult<Self> {
        let (actual, shape) = real_numbers(actual, "actual")?;
        let (values, sample_shape) = real_numbers(samples, "samples")?;
        let sample_axes = sample_shape.len().wrapping_sub(shape.len());
        if !(sample_axes == 1 || sample_axes == 2) || sample_shape[sample_axes..] != shape[..] {
            return Err(PyValueError::new_err(
                "samples must have one sample axis or chain/draw axes followed by the actual shape",
            ));
        }
        let targets = actual.len();
        let draws: usize = sample_shape[..sample_axes].iter().product();
        if draws == 0 || targets == 0 {
            return Err(PyValueError::new_err(
                "predictive draws must be nonempty and finite",
            ));
        }
        let columns = (0..targets)
            .map(|target| {
                (0..draws)
                    .map(|draw| values[draw * targets + target])
                    .collect()
            })
            .collect();
        Ok(Self {
            draws: TargetDraws::new(columns).map_err(value_error)?,
            actual,
            shape,
        })
    }

    /// The last axis's length: the horizon for a forecast shaped
    /// `(*series, horizon)`, and one for a scalar target.
    fn horizon(&self) -> usize {
        self.shape.last().copied().unwrap_or(1)
    }

    /// Group labels for every target: its horizon step, or zero for all.
    fn groups(&self, by_horizon: bool) -> Vec<usize> {
        let horizon = if by_horizon { self.horizon() } else { 1 };
        (0..self.actual.len())
            .map(|target| target % horizon)
            .collect()
    }

    /// Each target's MASE scale, from `training` shaped `(*series, time)`
    /// where the forecast is `(*series, horizon)`; `NaN` for a series with
    /// no observed pair `period` apart.
    fn mase_scales(&self, training: &Bound<'_, PyAny>, period: usize) -> PyResult<Vec<f64>> {
        let (values, shape) = real_numbers(training, "training")?;
        let series_shape = &self.shape[..self.shape.len().saturating_sub(1)];
        if shape.len() != series_shape.len() + 1 || shape[..series_shape.len()] != *series_shape {
            return Err(PyValueError::new_err(
                "training must have the forecast's series axes followed by a time axis",
            ));
        }
        let time = shape[series_shape.len()];
        let horizon = self.horizon();
        let scales = if time == 0 {
            vec![f64::NAN; self.actual.len() / horizon]
        } else {
            values
                .chunks(time)
                .map(|series| Ok(mase_scale(series, period)?.unwrap_or(f64::NAN)))
                .collect::<Result<Vec<f64>, EvaluationError>>()
                .map_err(value_error)?
        };
        Ok((0..self.actual.len())
            .map(|target| scales[target / horizon])
            .collect())
    }
}

/// Every forecast of a pooled evaluation must describe the same targets.
fn read_forecasts<'py, T>(
    forecasts: &[(Bound<'py, PyAny>, Bound<'py, PyAny>, T)],
) -> PyResult<Vec<Forecast>> {
    let read = forecasts
        .iter()
        .map(|(samples, actual, _)| Forecast::read(samples, actual))
        .collect::<PyResult<Vec<_>>>()?;
    match read.first() {
        None => Err(PyValueError::new_err("there are no forecasts to evaluate")),
        Some(first) if read.iter().any(|f| f.shape != first.shape) => Err(PyValueError::new_err(
            "every forecast must have the same target shape",
        )),
        Some(_) => Ok(read),
    }
}

/// Counts go to NumPy as signed integers, so that subtracting an expected
/// count cannot wrap around. No count of targets held in memory reaches 2**63.
fn count_value(count: usize) -> i64 {
    count as i64
}

fn shaped<'py>(py: Python<'py>, values: Vec<f64>, shape: &[usize]) -> PyResult<Bound<'py, PyAny>> {
    let array = ArrayD::from_shape_vec(IxDyn(shape), values)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(array.into_pyarray(py).into_any())
}

/// PIT values of `actual` under `samples`, shaped like `actual`, `NaN` for a
/// missing outcome. Randomised values use stream `stream` of `seed`.
#[pyfunction(name = "_pit_values")]
fn pit_values<'py>(
    py: Python<'py>,
    samples: &Bound<'py, PyAny>,
    actual: &Bound<'py, PyAny>,
    randomized: bool,
    seed: u64,
    stream: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let forecast = Forecast::read(samples, actual)?;
    let values = py
        .allow_threads(|| {
            if randomized {
                forecast
                    .draws
                    .randomized_pit(&forecast.actual, seed, stream)
            } else {
                forecast.draws.pit(&forecast.actual)
            }
        })
        .map_err(value_error)?;
    shaped(py, values, &forecast.shape)
}

/// Counts of the non-`NaN` values in `bins` equal bins on `[0, 1]`.
#[pyfunction(name = "_pit_histogram")]
fn pit_histogram<'py>(
    py: Python<'py>,
    values: &Bound<'py, PyAny>,
    bins: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let (values, _) = real_numbers(values, "values")?;
    let counts = core_pit_histogram(&values, bins).map_err(value_error)?;
    Ok(counts
        .into_iter()
        .map(count_value)
        .collect::<Vec<_>>()
        .into_pyarray(py)
        .into_any())
}

/// Coverage, its Wilson interval, and mean width of central intervals at
/// each level, pooled over `forecasts` and grouped by horizon step or not.
#[pyfunction(name = "_calibration_table")]
fn calibration_table<'py>(
    py: Python<'py>,
    forecasts: Vec<(Bound<'py, PyAny>, Bound<'py, PyAny>)>,
    levels: &Bound<'py, PyAny>,
    by_horizon: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let levels = read_levels(levels)?;
    let forecasts: Vec<_> = forecasts.into_iter().map(|(s, a)| (s, a, ())).collect();
    let read = read_forecasts(&forecasts)?;
    let groups = if by_horizon { read[0].horizon() } else { 1 };
    let rows = py
        .allow_threads(|| {
            let mut table = IntervalCalibration::new(&levels, groups)?;
            for forecast in &read {
                table.add(
                    &forecast.draws,
                    &forecast.actual,
                    &forecast.groups(by_horizon),
                )?;
            }
            Ok(table.rows())
        })
        .map_err(value_error)?;
    let shape: Vec<usize> = if by_horizon {
        vec![groups, levels.len()]
    } else {
        vec![levels.len()]
    };
    let column = |f: fn(&rustmc_core::forecast_evaluation::CalibrationRow) -> f64| {
        rows.iter().map(f).collect::<Vec<f64>>()
    };
    let result = PyDict::new(py);
    result.set_item("level", levels.clone().into_pyarray(py))?;
    result.set_item("coverage", shaped(py, column(|r| r.coverage), &shape)?)?;
    result.set_item(
        "coverage_lower",
        shaped(py, column(|r| r.coverage_lower), &shape)?,
    )?;
    result.set_item(
        "coverage_upper",
        shaped(py, column(|r| r.coverage_upper), &shape)?,
    )?;
    result.set_item("width", shaped(py, column(|r| r.mean_width), &shape)?)?;
    let counts: Vec<i64> = rows
        .iter()
        .step_by(levels.len())
        .map(|r| count_value(r.count))
        .collect();
    if by_horizon {
        result.set_item("count", counts.into_pyarray(py))?;
    } else {
        result.set_item("count", counts[0])?;
    }
    Ok(result)
}

/// `(samples, actual, training)`, with `training` `None` when MASE is not
/// wanted.
type ScoredForecast<'py> = (
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
    Option<Bound<'py, PyAny>>,
);

/// MAE, RMSE, WAPE and, where `training` is given, MASE of the predictive
/// mean or median, pooled over `forecasts` and grouped by horizon step or not.
#[pyfunction(name = "_point_accuracy")]
fn point_accuracy<'py>(
    py: Python<'py>,
    forecasts: Vec<ScoredForecast<'py>>,
    point: &str,
    period: usize,
    by_horizon: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let point = match point {
        "mean" => Point::Mean,
        "median" => Point::Median,
        _ => return Err(PyValueError::new_err("point must be 'mean' or 'median'")),
    };
    if period == 0 {
        return Err(PyValueError::new_err("period must be a positive integer"));
    }
    let read = read_forecasts(&forecasts)?;
    let with_training = forecasts.iter().filter(|(_, _, t)| t.is_some()).count();
    if with_training != 0 && with_training != forecasts.len() {
        return Err(PyValueError::new_err(
            "give training observations for every forecast or for none",
        ));
    }
    let scales = forecasts
        .iter()
        .zip(&read)
        .map(|((_, _, training), forecast)| match training {
            Some(training) => forecast.mase_scales(training, period),
            None => Ok(vec![f64::NAN; forecast.actual.len()]),
        })
        .collect::<PyResult<Vec<_>>>()?;
    let groups = if by_horizon { read[0].horizon() } else { 1 };
    let rows = py
        .allow_threads(|| {
            let mut accuracy = PointAccuracy::new(groups)?;
            for (forecast, scale) in read.iter().zip(&scales) {
                let errors = forecast.draws.point_errors(&forecast.actual, point)?;
                accuracy.add(
                    &errors,
                    &forecast.actual,
                    scale,
                    &forecast.groups(by_horizon),
                )?;
            }
            Ok(accuracy.rows())
        })
        .map_err(value_error)?;
    let result = PyDict::new(py);
    let set = |name: &str, values: Vec<f64>| -> PyResult<()> {
        if by_horizon {
            result.set_item(name, values.into_pyarray(py))
        } else {
            result.set_item(name, values[0])
        }
    };
    set("mae", rows.iter().map(|r| r.mae).collect())?;
    set("rmse", rows.iter().map(|r| r.rmse).collect())?;
    set("wape", rows.iter().map(|r| r.wape).collect())?;
    if with_training != 0 {
        set("mase", rows.iter().map(|r| r.mase).collect())?;
    }
    let counts: Vec<i64> = rows.iter().map(|r| count_value(r.count)).collect();
    if by_horizon {
        result.set_item("count", counts.into_pyarray(py))?;
    } else {
        result.set_item("count", counts[0])?;
    }
    Ok(result)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pit_values, m)?)?;
    m.add_function(wrap_pyfunction!(pit_histogram, m)?)?;
    m.add_function(wrap_pyfunction!(calibration_table, m)?)?;
    m.add_function(wrap_pyfunction!(point_accuracy, m)?)?;
    m.add_function(wrap_pyfunction!(interval_levels, m)?)?;
    Ok(())
}
