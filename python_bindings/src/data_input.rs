//! Conversion of Python data dictionaries into core data bindings, and the
//! one exact float64 conversion that every array argument goes through.
use numpy::{PyArrayDyn, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyTuple};
use rustmc_core::data::{DataInputs, MatrixBinding};
use rustmc_core::model::MuExpr;
use std::collections::HashMap;
use std::sync::Arc;

pub(crate) type Data1d = HashMap<String, Vec<f64>>;
pub(crate) type Data2d = HashMap<String, (Vec<f64>, usize, usize)>;

pub(crate) fn data_inputs_from_maps(data_1d: &Data1d, data_2d: &Data2d) -> DataInputs {
    DataInputs {
        vectors: data_1d
            .iter()
            .map(|(key, values)| (key.clone(), Arc::<[f64]>::from(values.clone())))
            .collect(),
        matrices: data_2d
            .iter()
            .map(|(key, (values, n_rows, n_cols))| {
                (
                    key.clone(),
                    MatrixBinding {
                        data: Arc::from(values.clone()),
                        n_rows: *n_rows,
                        n_cols: *n_cols,
                    },
                )
            })
            .collect(),
    }
}

/// Extract numeric arrays from a Python dict into typed Rust maps.
///
/// Each value must be a 1-D or 2-D array (or nested list) of real numbers
/// that float64 holds exactly; see [`real_numbers`]. A scalar is refused
/// naming the key rather than silently becoming a length-1 vector.
pub(crate) fn parse_data_dict(data: &Bound<'_, PyDict>) -> PyResult<(Data1d, Data2d)> {
    let mut data_1d = HashMap::new();
    let mut data_2d = HashMap::new();
    for (key, value) in data.iter() {
        let key: String = key.extract()?;
        let (values, shape) = numeric_payload(&key, &value)?;
        ensure_finite_data(&key, &values)?;
        if let [n_rows, n_cols] = shape[..] {
            data_2d.insert(key, (values, n_rows, n_cols));
        } else {
            data_1d.insert(key, values);
        }
    }
    Ok((data_1d, data_2d))
}

/// The row-major float64 values and shape of one data value.
fn numeric_payload(key: &str, value: &Bound<'_, PyAny>) -> PyResult<(Vec<f64>, Vec<usize>)> {
    let (values, shape) = real_numbers(value, &format!("data key '{key}'"))?;
    match shape.len() {
        0 => Err(PyValueError::new_err(format!(
            "data key '{key}' must be a 1-D or 2-D array, got a scalar; wrap a single \
             value in a list"
        ))),
        1 | 2 => Ok((values, shape)),
        ndim => Err(PyValueError::new_err(format!(
            "data key '{key}' must be a 1-D or 2-D array, got {ndim} dimensions"
        ))),
    }
}

/// The row-major float64 values and shape of a real numeric array-like.
///
/// This is the one conversion behind model data dictionaries and every
/// forecasting array argument. It accepts float and integer NumPy arrays of
/// any width, memory order or strides, and (nested) lists and tuples of
/// numbers, but only when float64 holds every value exactly:
///
/// - integers of magnitude at most 2**53, and larger ones only when they are
///   representable (such as 2**60); Python integers beyond the 64-bit range,
///   which NumPy can only store as objects, are refused with object arrays;
/// - float16/32/64 values; a long double only when it is a float64 value.
///
/// Anything NumPy would only coerce is refused with a `ValueError` naming
/// `subject`, rather than silently becoming a different number: booleans
/// (including `True` inside a list, which NumPy would read as 1), strings,
/// bytes, objects, complex values, and masked arrays with a masked entry,
/// whose mask NumPy's conversion would drop. `NaN` and infinities pass
/// through; each caller decides whether it accepts them.
pub(crate) fn real_numbers(
    value: &Bound<'_, PyAny>,
    subject: &str,
) -> PyResult<(Vec<f64>, Vec<usize>)> {
    let numpy = value.py().import("numpy")?;
    let masked = numpy.getattr("ma")?;
    if value.is_instance(&masked.getattr("MaskedArray")?)?
        && masked
            .call_method1("getmaskarray", (value,))?
            .call_method0("any")?
            .is_truthy()?
    {
        return Err(PyValueError::new_err(format!(
            "{subject} is a masked array with masked entries, and the mask would be \
             ignored; fill them explicitly, for example with .filled(np.nan) where \
             missing values are allowed"
        )));
    }
    if value.is_instance_of::<PyList>() || value.is_instance_of::<PyTuple>() {
        refuse_inexact_elements(&numpy, value, subject)?;
    }
    let array = numpy.call_method1("asarray", (value,)).map_err(|error| {
        PyValueError::new_err(format!("{subject} must be a real numeric array: {error}"))
    })?;
    let shape: Vec<usize> = array.getattr("shape")?.extract()?;
    let dtype = array.getattr("dtype")?;
    let kind: String = dtype.getattr("kind")?.extract()?;
    let item_size: usize = dtype.getattr("itemsize")?.extract()?;
    // Row-major, whatever the input's order and strides.
    let contiguous = |target: &str| numpy.call_method1("ascontiguousarray", (&array, target));
    let values = match kind.as_str() {
        "f" => {
            let doubles = contiguous("float64")?;
            if item_size > 8 {
                exact_long_doubles(&numpy, &array, &doubles, subject)?;
            }
            let doubles = doubles.downcast::<PyArrayDyn<f64>>()?.readonly();
            doubles.as_slice()?.to_vec()
        }
        "i" => {
            let array = contiguous("int64")?;
            let array = array.downcast::<PyArrayDyn<i64>>()?.readonly();
            exact_integers(subject, array.as_slice()?.iter().map(|&v| i128::from(v)))?
        }
        "u" => {
            let array = contiguous("uint64")?;
            let array = array.downcast::<PyArrayDyn<u64>>()?.readonly();
            exact_integers(subject, array.as_slice()?.iter().map(|&v| i128::from(v)))?
        }
        _ => {
            let (name, hint) = match kind.as_str() {
                "b" => ("bool".to_string(), ""),
                "U" => ("str".to_string(), ""),
                "S" => ("bytes".to_string(), ""),
                "O" => (
                    "object".to_string(),
                    "; convert an object array of numbers, such as a pandas object \
                     column, with .astype(float) first",
                ),
                "c" => ("complex".to_string(), ""),
                _ => (dtype.str()?.to_string(), ""),
            };
            return Err(PyValueError::new_err(format!(
                "{subject} must hold real numbers (float or integer); got {name} values{hint}"
            )));
        }
    };
    Ok((values, shape))
}

/// Deepest list nesting accepted: NumPy's own dimension limit. It also stops
/// the walk on a list that contains itself.
const MAX_NESTING: usize = 64;

/// Refuse list elements that `numpy.asarray` would silently turn into other
/// numbers: booleans (read as 0 and 1), integers that float64 would round
/// when the list also holds floats, and arrays or NumPy scalars inside the
/// list that [`real_numbers`] would refuse on their own (masked entries,
/// booleans, inexact integers or long doubles).
fn refuse_inexact_elements(
    numpy: &Bound<'_, PyModule>,
    value: &Bound<'_, PyAny>,
    subject: &str,
) -> PyResult<()> {
    let integer_type = numpy.getattr("integer")?;
    let generic = numpy.getattr("generic")?;
    let ndarray = numpy.getattr("ndarray")?;
    let mut pending = vec![(value.clone(), 1)];
    while let Some((sequence, depth)) = pending.pop() {
        if depth > MAX_NESTING {
            return Err(PyValueError::new_err(format!(
                "{subject} is nested more than {MAX_NESTING} lists deep (or contains itself)"
            )));
        }
        for item in sequence.try_iter()? {
            let item = item?;
            // Python floats, including np.float64, are float64 already.
            if item.is_instance_of::<PyFloat>() {
                continue;
            }
            if item.is_instance_of::<PyList>() || item.is_instance_of::<PyTuple>() {
                pending.push((item, depth + 1));
            } else if item.is_instance_of::<PyBool>() {
                return Err(PyValueError::new_err(format!(
                    "{subject} must hold real numbers (float or integer); got bool values \
                     ({item} in a list)"
                )));
            } else if item.is_instance_of::<PyInt>() || item.is_instance(&integer_type)? {
                // `__index__` gives a NumPy integer's exact Python value.
                let integer = item.call_method0("__index__")?;
                let exact = match integer.extract::<i64>() {
                    Ok(small) => (small as f64) as i128 == i128::from(small),
                    // Python compares an int with a float exactly.
                    Err(_) => match integer.call_method0("__float__") {
                        Ok(float) => integer.eq(float)?,
                        Err(_) => false,
                    },
                };
                if !exact {
                    return Err(inexact_integer(subject, &integer.to_string(), None));
                }
            } else if item.is_instance(&ndarray)? || item.is_instance(&generic)? {
                real_numbers(&item, subject)?;
            }
        }
    }
    Ok(())
}

/// Refuse long doubles that float64 would round.
fn exact_long_doubles(
    numpy: &Bound<'_, PyModule>,
    original: &Bound<'_, PyAny>,
    doubles: &Bound<'_, PyAny>,
    subject: &str,
) -> PyResult<()> {
    let back = doubles.call_method1("astype", (original.getattr("dtype")?,))?;
    let same = numpy.call_method1(
        "logical_or",
        (
            numpy.call_method1("equal", (&back, original))?,
            numpy.call_method1("isnan", (original,))?,
        ),
    )?;
    if same.call_method0("all")?.is_truthy()? {
        return Ok(());
    }
    let index: usize = numpy
        .call_method1("argmin", (same.call_method0("ravel")?,))?
        .extract()?;
    Err(PyValueError::new_err(format!(
        "{subject} holds a long double at flat index {index} that float64 cannot \
         represent exactly; convert it with .astype(float) if that rounding is acceptable"
    )))
}

fn inexact_integer(subject: &str, value: &str, index: Option<usize>) -> PyErr {
    let position = index.map_or(String::new(), |index| format!(" at flat index {index}"));
    PyValueError::new_err(format!(
        "{subject} contains integer {value}{position}, which float64 cannot represent \
         exactly (integers beyond 2**53 are rounded); convert it to float explicitly if \
         that rounding is acceptable"
    ))
}

/// Integers as float64, refusing any that float64 would round.
fn exact_integers(subject: &str, values: impl Iterator<Item = i128>) -> PyResult<Vec<f64>> {
    values
        .enumerate()
        .map(|(index, value)| {
            let converted = value as f64;
            // `as i128` is exact for every float64 in the integer ranges here.
            if converted as i128 == value {
                Ok(converted)
            } else {
                Err(inexact_integer(subject, &value.to_string(), Some(index)))
            }
        })
        .collect()
}

fn ensure_finite_data(key: &str, values: &[f64]) -> PyResult<()> {
    if values.is_empty() {
        return Err(PyValueError::new_err(format!(
            "data key '{}' must contain at least one value",
            key
        )));
    }
    if let Some((index, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(PyValueError::new_err(format!(
            "data key '{}' contains non-finite value {} at flat index {}",
            key, value, index
        )));
    }
    Ok(())
}

/// Merge call-site data over bound data while ensuring a key has exactly one
/// dimensional kind. A 1-D override removes a stale 2-D binding and vice versa.
pub(crate) fn merge_data_overrides(
    data_1d: &mut HashMap<String, Vec<f64>>,
    data_2d: &mut HashMap<String, (Vec<f64>, usize, usize)>,
    extra_1d: HashMap<String, Vec<f64>>,
    extra_2d: HashMap<String, (Vec<f64>, usize, usize)>,
) {
    for (key, value) in extra_1d {
        data_2d.remove(&key);
        data_1d.insert(key, value);
    }
    for (key, value) in extra_2d {
        data_1d.remove(&key);
        data_2d.insert(key, value);
    }
}

/// Validate that each matrix's row-major storage matches its shape.
pub(crate) fn validate_matrix_storage(
    data_2d: &HashMap<String, (Vec<f64>, usize, usize)>,
) -> PyResult<()> {
    for (key, (values, rows, cols)) in data_2d {
        if rows.checked_mul(*cols) != Some(values.len()) {
            return Err(PyValueError::new_err(format!(
                "invalid matrix shape for {key}"
            )));
        }
    }
    Ok(())
}

/// Validate that every data key referenced in `expr` and `observed_key` exists in the
/// bound data maps.  Called eagerly at `normal_likelihood()` time when data is bound.
pub(crate) fn validate_data_keys(
    expr: &MuExpr,
    observed_key: &str,
    data_1d: &HashMap<String, Vec<f64>>,
    data_2d: &HashMap<String, (Vec<f64>, usize, usize)>,
) -> PyResult<()> {
    if !data_1d.contains_key(observed_key) {
        let available: Vec<&str> = data_1d.keys().map(String::as_str).collect();
        return Err(PyValueError::new_err(format!(
            "observed key '{}' not found in bound data. Available 1-D keys: [{}]",
            observed_key,
            available.join(", ")
        )));
    }
    validate_expr_keys(expr, data_1d, data_2d)
}

pub(crate) fn validate_expr_keys(
    expr: &MuExpr,
    data_1d: &HashMap<String, Vec<f64>>,
    data_2d: &HashMap<String, (Vec<f64>, usize, usize)>,
) -> PyResult<()> {
    match expr {
        MuExpr::Const(_) => Ok(()),
        MuExpr::Unary(_, a) | MuExpr::Sum(a) => validate_expr_keys(a, data_1d, data_2d),
        MuExpr::ParamTimesData { data_key, .. }
        | MuExpr::Data(data_key)
        | MuExpr::Gather { data_key, .. } => {
            if !data_1d.contains_key(data_key) {
                let available: Vec<&str> = data_1d.keys().map(String::as_str).collect();
                return Err(PyValueError::new_err(format!(
                    "data key '{}' not found in bound data. Available 1-D keys: [{}]",
                    data_key,
                    available.join(", ")
                )));
            }
            Ok(())
        }
        MuExpr::MatVec { data_key, .. } => {
            if !data_2d.contains_key(data_key) {
                let available: Vec<&str> = data_2d.keys().map(String::as_str).collect();
                return Err(PyValueError::new_err(format!(
                    "matrix key '{}' not found in bound data. Available 2-D keys: [{}]",
                    data_key,
                    available.join(", ")
                )));
            }
            Ok(())
        }
        MuExpr::Param(_) => Ok(()),
        MuExpr::Add(a, b) | MuExpr::Binary(_, a, b) => {
            validate_expr_keys(a, data_1d, data_2d)?;
            validate_expr_keys(b, data_1d, data_2d)
        }
    }
}
