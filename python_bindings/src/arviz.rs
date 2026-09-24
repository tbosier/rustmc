//! Version-aware conversion into ArviZ containers.
use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Major version of the installed ArviZ, which selects its conversion API.
pub(crate) fn arviz_api_generation(az: &Bound<'_, PyModule>) -> PyResult<u64> {
    let arviz_version: String = az.getattr("__version__")?.extract()?;
    arviz_version
        .split('.')
        .next()
        .and_then(|part| part.parse::<u64>().ok())
        .ok_or_else(|| {
            PyValueError::new_err(format!(
                "cannot determine the ArviZ API generation from version '{arviz_version}'"
            ))
        })
}

/// Call the version-native ArviZ dictionary converter.
pub(crate) fn arviz_from_groups<'py>(
    az: &Bound<'py, PyModule>,
    groups: Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyAny>> {
    let arviz_major = arviz_api_generation(az)?;
    arviz_from_groups_versioned(az, arviz_major, groups)
}

pub(crate) fn arviz_from_groups_versioned<'py>(
    az: &Bound<'py, PyModule>,
    arviz_major: u64,
    groups: Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyAny>> {
    // ArviZ 1.0 moved conversion into arviz-base and changed `from_dict`
    // from one keyword per group to a single nested group dictionary.
    if arviz_major >= 1 {
        az.call_method1("from_dict", (groups,))
    } else {
        az.call_method("from_dict", (), Some(&groups))
    }
}

/// The dataset for one group of an ArviZ container.
///
/// ArviZ 0.x returns an `InferenceData` whose groups are Dataset attributes;
/// 1.x returns an xarray `DataTree` whose children are nodes wrapping one.
pub(crate) fn arviz_group<'py>(
    arviz_major: u64,
    container: &Bound<'py, PyAny>,
    name: &str,
) -> PyResult<Bound<'py, PyAny>> {
    if arviz_major >= 1 {
        container.get_item(name)?.getattr("dataset")
    } else {
        container.getattr(name)
    }
}

/// Relabel the `posterior_predictive` draw axis with the posterior draw labels
/// that survived thinning.
///
/// ArviZ groups are independent datasets, so a shorter predictive draw axis is
/// legal — but without labels it is anonymous, and the pairing between a
/// predictive draw and its parameter draw is lost. Writing the retained draws'
/// labels as the `draw` coordinate restores it: xarray can then align or
/// `.sel()` the posterior down to exactly the draws that were simulated.
///
/// `retained` holds positions along the posterior's draw axis, and the labels
/// are read back off the posterior group rather than assumed: ArviZ's
/// `data.index_origin` decides where the `draw` coordinate starts, so writing
/// bare indices would silently offset the two groups wherever it is not zero.
pub(crate) fn assign_posterior_predictive_draw_coords(
    py: Python<'_>,
    arviz_major: u64,
    container: &Bound<'_, PyAny>,
    retained: &[i64],
) -> PyResult<()> {
    let positions = PyArray1::from_slice(py, retained);
    let labels = arviz_group(arviz_major, container, "posterior")?
        .getattr("draw")?
        .getattr("values")?
        .get_item(positions)?;
    let coords = PyDict::new(py);
    coords.set_item("draw", labels)?;
    let relabelled = arviz_group(arviz_major, container, "posterior_predictive")?.call_method(
        "assign_coords",
        (),
        Some(&coords),
    )?;
    if arviz_major >= 1 {
        container.set_item("posterior_predictive", relabelled)?;
    } else {
        container.setattr("posterior_predictive", relabelled)?;
    }
    Ok(())
}
