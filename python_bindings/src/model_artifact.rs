//! Versioned declarative graph-model artifacts. Training payloads are deliberately excluded.
use super::{compile_python_model, template_data_for_spec, ModelSpec, PyCompiledModel};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rustmc_core::data::DataSchema;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(serde::Serialize, serde::Deserialize)]
pub(super) struct Artifact {
    format: String,
    version: u32,
    definition: ModelSpec,
    schema: DataSchema,
}
pub(super) fn encode(model: &PyCompiledModel) -> PyResult<String> {
    serde_json::to_string(&describe(model)).map_err(|e| PyValueError::new_err(e.to_string()))
}
pub(super) fn describe(model: &PyCompiledModel) -> Artifact {
    Artifact {
        format: "rustmc.graph-model".into(),
        version: 1,
        definition: model.definition.clone(),
        schema: model.structure.schema.clone(),
    }
}
pub(super) fn decode(text: &str) -> PyResult<PyCompiledModel> {
    let artifact: Artifact = serde_json::from_str(text)
        .map_err(|e| PyValueError::new_err(format!("invalid graph model artifact: {e}")))?;
    reconstruct(artifact)
}
pub(super) fn reconstruct(artifact: Artifact) -> PyResult<PyCompiledModel> {
    if artifact.format != "rustmc.graph-model" || artifact.version != 1 {
        return Err(PyValueError::new_err(
            "unsupported graph model artifact format/version",
        ));
    }
    let mut spec = artifact.definition;
    validate_definition(&spec)?;
    super::reject_discrete_priors_for_gradient_sampling(&spec.priors)?;
    // Only column counts and dimension names are structural; templates contain no user data.
    for slot in &artifact.schema.matrices {
        let rustmc_core::data::SlotKind::Matrix { n_cols } = slot.kind else {
            return Err(PyValueError::new_err("invalid matrix schema"));
        };
        if n_cols == 0 {
            return Err(PyValueError::new_err(
                "matrix column count must be positive",
            ));
        }
        spec.bound_data_2d
            .insert(slot.key.clone(), (vec![0.0; n_cols], 1, n_cols));
    }
    let (one_d, two_d) = template_data_for_spec(&spec)?;
    let compiled = compile_python_model(&spec, &one_d, &two_d)?;
    if compiled.graph.schema != artifact.schema {
        return Err(PyValueError::new_err(
            "artifact schema disagrees with its model definition",
        ));
    }
    spec.bound_data_1d.clear();
    spec.bound_data_2d.clear();
    Ok(PyCompiledModel {
        definition: spec,
        structure: Arc::new(compiled.graph.structure_only()),
        likelihood_names: compiled.likelihood_names,
        display_params: compiled.display_params,
        default_data_1d: HashMap::new(),
        default_data_2d: HashMap::new(),
    })
}

fn validate_definition(spec: &ModelSpec) -> PyResult<()> {
    use super::{validate_positive_finite, HyperParam, PriorSpec, SigmaSpec};
    let positive_hyper = |hp: &HyperParam| -> PyResult<()> {
        if let HyperParam::Const(v) = hp {
            validate_positive_finite("prior scale/rate", *v)?;
        }
        Ok(())
    };
    for prior in &spec.priors {
        match prior {
            PriorSpec::Normal { sigma, .. }
            | PriorSpec::HalfNormal { sigma, .. }
            | PriorSpec::LogNormal { sigma, .. } => positive_hyper(sigma)?,
            PriorSpec::Exponential { rate, .. } => positive_hyper(rate)?,
            PriorSpec::StudentT { nu, sigma, .. } => {
                validate_positive_finite("nu", *nu)?;
                validate_positive_finite("sigma", *sigma)?;
            }
            PriorSpec::Uniform { lower, upper, .. } => {
                if lower >= upper {
                    return Err(PyValueError::new_err("uniform bounds must be increasing"));
                }
            }
            PriorSpec::Bernoulli { p, .. } => {
                if !(0.0..=1.0).contains(p) {
                    return Err(PyValueError::new_err(
                        "Bernoulli probability must be in [0,1]",
                    ));
                }
            }
            PriorSpec::Poisson { lam, .. } => validate_positive_finite("Poisson rate", *lam)?,
            PriorSpec::Gamma { alpha, beta, .. } | PriorSpec::Beta { alpha, beta, .. } => {
                validate_positive_finite("alpha", *alpha)?;
                validate_positive_finite("beta", *beta)?;
            }
            PriorSpec::VectorNormal { n, sigma, .. } => {
                if *n == 0 {
                    return Err(PyValueError::new_err(
                        "vector parameter length must be positive",
                    ));
                }
                validate_positive_finite("sigma", *sigma)?;
            }
        }
    }
    for likelihood in &spec.likelihoods {
        if let Some(SigmaSpec::Const(v)) = likelihood.sigma {
            validate_positive_finite("likelihood scale", v)?;
        }
    }
    let mut names = std::collections::HashSet::new();
    for (name, expr) in &spec.potentials {
        if name.is_empty() || !names.insert(name) || !expr.is_scalar() {
            return Err(PyValueError::new_err(
                "potentials must have unique names and scalar expressions",
            ));
        }
    }
    Ok(())
}

impl Artifact {
    /// Fit payloads already declare the complete parameter axis. Reject impossible
    /// structural sizes before allocating template matrices or parameter vectors.
    pub(super) fn validate_parameter_limit(&self, count: usize) -> PyResult<()> {
        let mut minimum = 0usize;
        for prior in &self.definition.priors {
            let n = match prior {
                super::PriorSpec::VectorNormal { n, .. } => *n,
                _ => 1,
            };
            minimum = minimum
                .checked_add(n)
                .ok_or_else(|| PyValueError::new_err("artifact parameter count overflow"))?;
            if minimum > count {
                return Err(PyValueError::new_err(
                    "artifact model parameter dimensions exceed posterior axis",
                ));
            }
        }
        for slot in &self.schema.matrices {
            if let rustmc_core::data::SlotKind::Matrix { n_cols } = slot.kind {
                if n_cols > count {
                    return Err(PyValueError::new_err(
                        "artifact matrix width exceeds posterior parameter axis",
                    ));
                }
            }
        }
        Ok(())
    }
}
