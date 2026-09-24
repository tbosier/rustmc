//! Conversion of Python data dictionaries into core data bindings.
use numpy::{PyArrayDyn, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
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
/// that float64 holds exactly: floats, or integers of magnitude at most
/// 2**53 (and larger integers that happen to be representable). Anything
/// NumPy would only coerce -- a scalar, booleans, strings, bytes, objects,
/// complex values -- is refused naming the key, rather than silently
/// becoming a different number.
pub(crate) fn parse_data_dict(data: &Bound<'_, PyDict>) -> PyResult<(Data1d, Data2d)> {
    let numpy = data.py().import("numpy")?;
    let mut data_1d = HashMap::new();
    let mut data_2d = HashMap::new();
    for (key, value) in data.iter() {
        let key: String = key.extract()?;
        let (values, shape) = numeric_payload(&numpy, &key, &value)?;
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
fn numeric_payload(
    numpy: &Bound<'_, PyModule>,
    key: &str,
    value: &Bound<'_, PyAny>,
) -> PyResult<(Vec<f64>, Vec<usize>)> {
    let array = numpy
        .call_method1("asarray", (value,))
        .map_err(|error| PyValueError::new_err(format!("data key '{key}': {error}")))?;
    let ndim: usize = array.getattr("ndim")?.extract()?;
    match ndim {
        0 => {
            return Err(PyValueError::new_err(format!(
                "data key '{key}' must be a 1-D or 2-D array, got a scalar; wrap a single \
                 value in a list"
            )))
        }
        1 | 2 => {}
        _ => {
            return Err(PyValueError::new_err(format!(
                "data key '{key}' must be a 1-D or 2-D array, got {ndim} dimensions"
            )))
        }
    }
    let dtype = array.getattr("dtype")?;
    let kind: String = dtype.getattr("kind")?.extract()?;
    // Core matrices are row-major; `ascontiguousarray` also normalises
    // Fortran-ordered arrays and strided views.
    let contiguous = |target: &str| numpy.call_method1("ascontiguousarray", (&array, target));
    let values = match kind.as_str() {
        "f" => {
            let array = contiguous("float64")?;
            let array = array.downcast::<PyArrayDyn<f64>>()?.readonly();
            array.as_slice()?.to_vec()
        }
        "i" => {
            let array = contiguous("int64")?;
            let array = array.downcast::<PyArrayDyn<i64>>()?.readonly();
            exact_integers(key, array.as_slice()?.iter().map(|&v| i128::from(v)))?
        }
        "u" => {
            let array = contiguous("uint64")?;
            let array = array.downcast::<PyArrayDyn<u64>>()?.readonly();
            exact_integers(key, array.as_slice()?.iter().map(|&v| i128::from(v)))?
        }
        _ => {
            let name = match kind.as_str() {
                "b" => "bool".to_string(),
                "U" => "str".to_string(),
                "S" => "bytes".to_string(),
                "O" => "object".to_string(),
                "c" => "complex".to_string(),
                _ => dtype.str()?.to_string(),
            };
            return Err(PyValueError::new_err(format!(
                "data key '{key}' has {name} values; data must be real numbers (float or \
                 integer arrays, or lists of them)"
            )));
        }
    };
    let shape = array.getattr("shape")?.extract()?;
    Ok((values, shape))
}

/// Integers as float64, refusing any that float64 would round.
fn exact_integers(key: &str, values: impl Iterator<Item = i128>) -> PyResult<Vec<f64>> {
    values
        .enumerate()
        .map(|(index, value)| {
            let converted = value as f64;
            // `as i128` is exact for every float64 in the integer ranges here.
            if converted as i128 == value {
                Ok(converted)
            } else {
                Err(PyValueError::new_err(format!(
                    "data key '{key}' contains integer {value} at flat index {index}, which \
                     float64 cannot represent exactly (integers beyond 2**53 are rounded); \
                     convert it to float explicitly if that rounding is acceptable"
                )))
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
