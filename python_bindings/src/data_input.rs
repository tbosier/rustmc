//! Conversion of Python data dictionaries into core data bindings.
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
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

/// Extract numpy arrays from a Python dict into typed Rust maps.
pub(crate) fn parse_data_dict(data: &Bound<'_, PyDict>) -> PyResult<(Data1d, Data2d)> {
    let mut data_1d = HashMap::new();
    let mut data_2d = HashMap::new();
    for (key, value) in data.iter() {
        let key_str: String = key.extract()?;
        // Core matrices use row-major storage. NumPy's slice API also accepts
        // Fortran-contiguous arrays, so dtype alone cannot justify bypassing
        // this normalization. It also accepts strided views and Python lists.
        let value = data
            .py()
            .import("numpy")?
            .call_method1("ascontiguousarray", (&value, "float64"))?;
        if let Ok(arr) = value.downcast::<PyArray2<f64>>() {
            let shape = arr.shape().to_vec();
            let slice = unsafe { arr.as_slice()? };
            ensure_finite_data(&key_str, slice)?;
            data_2d.insert(key_str, (slice.to_vec(), shape[0], shape[1]));
        } else {
            let arr: &Bound<'_, PyArray1<f64>> = value.downcast()?;
            let vec: Vec<f64> = unsafe { arr.as_slice()?.to_vec() };
            ensure_finite_data(&key_str, &vec)?;
            data_1d.insert(key_str, vec);
        }
    }
    Ok((data_1d, data_2d))
}

pub(crate) fn ensure_finite_data(key: &str, values: &[f64]) -> PyResult<()> {
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
