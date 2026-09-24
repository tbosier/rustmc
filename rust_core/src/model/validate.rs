//! Checks a definition must pass before it is compiled: names, references,
//! hyperparameters and observed values.
use crate::graph::{Graph, NodeId};
use crate::param_ref::{validate_param_references, ParamReference};
use std::collections::HashMap;

use super::*;

pub fn template_data_for_spec(spec: &ModelSpec) -> ModelResult<(Data1d, Data2d)> {
    let mut one_d = spec.bound_data_1d.clone();
    let mut two_d = spec.bound_data_2d.clone();
    let vector_sizes: HashMap<&str, usize> = spec
        .priors
        .iter()
        .filter_map(|prior| match prior {
            PriorSpec::VectorNormal { name, n, .. } => Some((name.as_str(), *n)),
            _ => None,
        })
        .collect();
    fn visit(
        expr: &MuExpr,
        one_d: &mut Data1d,
        two_d: &mut Data2d,
        vector_sizes: &HashMap<&str, usize>,
    ) -> ModelResult<()> {
        match expr {
            MuExpr::ParamTimesData { data_key, .. }
            | MuExpr::Data(data_key)
            | MuExpr::Gather { data_key, .. } => {
                one_d.entry(data_key.clone()).or_insert_with(|| vec![0.0]);
            }
            MuExpr::MatVec {
                param_name,
                data_key,
            } => {
                if !two_d.contains_key(data_key) {
                    let n_cols = vector_sizes.get(param_name.as_str()).copied().ok_or_else(|| {
                        ModelError::invalid(format!(
                            "cannot compile matrix '{}' without bound data: declare '{}' with vector_normal_prior so its column count is structural",
                            data_key, param_name
                        ))
                    })?;
                    two_d.insert(data_key.clone(), (vec![1.0; n_cols], 1, n_cols));
                }
            }
            MuExpr::Add(a, b) | MuExpr::Binary(_, a, b) => {
                visit(a, one_d, two_d, vector_sizes)?;
                visit(b, one_d, two_d, vector_sizes)?;
            }
            MuExpr::Unary(_, a) | MuExpr::Sum(a) => visit(a, one_d, two_d, vector_sizes)?,
            MuExpr::Const(_) | MuExpr::Param(_) => {}
        }
        Ok(())
    }
    for likelihood in &spec.likelihoods {
        visit(&likelihood.mu_expr, &mut one_d, &mut two_d, &vector_sizes)?;
        one_d
            .entry(likelihood.observed_key.clone())
            .or_insert_with(|| vec![1.0]);
    }
    for (_, expr) in spec.potentials.iter().chain(&spec.deterministics) {
        visit(expr, &mut one_d, &mut two_d, &vector_sizes)?;
    }
    Ok((one_d, two_d))
}

pub fn prior_name(prior: &PriorSpec) -> &str {
    match prior {
        PriorSpec::Normal { name, .. }
        | PriorSpec::HalfNormal { name, .. }
        | PriorSpec::Exponential { name, .. }
        | PriorSpec::LogNormal { name, .. }
        | PriorSpec::StudentT { name, .. }
        | PriorSpec::Uniform { name, .. }
        | PriorSpec::Bernoulli { name, .. }
        | PriorSpec::Poisson { name, .. }
        | PriorSpec::Gamma { name, .. }
        | PriorSpec::Beta { name, .. }
        | PriorSpec::VectorNormal { name, .. } => name,
    }
}

pub fn prior_hyper_refs(prior: &PriorSpec) -> Vec<(&'static str, &str)> {
    let mut out = Vec::new();
    fn push<'a>(out: &mut Vec<(&'static str, &'a str)>, role: &'static str, hp: &'a HyperParam) {
        if let HyperParam::Param(name) = hp {
            out.push((role, name.as_str()));
        }
    }
    match prior {
        PriorSpec::Normal { mu, sigma, .. } | PriorSpec::LogNormal { mu, sigma, .. } => {
            push(&mut out, "mu", mu);
            push(&mut out, "sigma", sigma);
        }
        PriorSpec::HalfNormal { sigma, .. } => push(&mut out, "sigma", sigma),
        PriorSpec::Exponential { rate, .. } => push(&mut out, "rate", rate),
        PriorSpec::StudentT { .. }
        | PriorSpec::Uniform { .. }
        | PriorSpec::Bernoulli { .. }
        | PriorSpec::Poisson { .. }
        | PriorSpec::Gamma { .. }
        | PriorSpec::Beta { .. }
        | PriorSpec::VectorNormal { .. } => {}
    }
    out
}

pub fn model_reference_set(
    priors: &[PriorSpec],
    likelihoods: &[LikelihoodSpec],
) -> (Vec<String>, Vec<ParamReference>) {
    let declared: Vec<String> = priors.iter().map(|p| prior_name(p).to_string()).collect();
    let mut refs = Vec::new();

    for (idx, prior) in priors.iter().enumerate() {
        for (role, name) in prior_hyper_refs(prior) {
            refs.push(ParamReference::ordered(
                name,
                format!("prior '{}' hyperparameter {}", prior_name(prior), role),
                idx,
            ));
        }
    }

    for lik in likelihoods {
        let mut names = Vec::new();
        collect_expr_param_names(&lik.mu_expr, &mut names);
        for name in names {
            refs.push(ParamReference::unordered(
                name,
                format!("the linear predictor of likelihood '{}'", lik.name),
            ));
        }
        if let Some(SigmaSpec::Param(name)) = &lik.sigma {
            refs.push(ParamReference::unordered(
                name.clone(),
                format!("the scale parameter of likelihood '{}'", lik.name),
            ));
        }
    }

    (declared, refs)
}

pub fn validate_model_references(
    priors: &[PriorSpec],
    likelihoods: &[LikelihoodSpec],
) -> ModelResult<()> {
    let mut names = std::collections::HashSet::new();
    for lik in likelihoods {
        if lik.name.is_empty()
            || !names.insert(&lik.name)
            || priors.iter().any(|p| prior_name(p) == lik.name)
        {
            return Err(ModelError::invalid(format!(
                "observation name '{}' must be unique and distinct from parameter names",
                lik.name
            )));
        }
    }
    let (declared, refs) = model_reference_set(priors, likelihoods);
    validate_param_references(&declared, &refs).map_err(param_error)
}

pub fn reject_discrete_priors_for_gradient_sampling(priors: &[PriorSpec]) -> ModelResult<()> {
    let discrete: Vec<&str> = priors
        .iter()
        .filter_map(|prior| match prior {
            PriorSpec::Bernoulli { name, .. } | PriorSpec::Poisson { name, .. } => {
                Some(name.as_str())
            }
            _ => None,
        })
        .collect();
    if discrete.is_empty() {
        return Ok(());
    }
    Err(ModelError::invalid(format!(
        "Discrete prior parameter(s) [{}] cannot be sampled with HMC/NUTS. \
         Bernoulli and Poisson priors are currently supported only by \
         sample_prior_predictive(); posterior inference requires continuous \
         parameters or explicit marginalisation.",
        discrete.join(", ")
    )))
}

/// A potential is a bare log-density term with no random generator attached,
/// so a model carrying one has no prior that can be simulated forward.
pub fn reject_potentials_for_prior_predictive(potentials: &[(String, MuExpr)]) -> ModelResult<()> {
    if potentials.is_empty() {
        Ok(())
    } else {
        Err(ModelError::invalid(
            "prior predictive simulation is not defined for models with potentials; custom density terms do not supply a prior random generator",
        ))
    }
}

pub fn validate_finite(name: &str, value: f64) -> ModelResult<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(ModelError::invalid(format!("{} must be finite", name)))
    }
}

pub fn validate_positive_finite(name: &str, value: f64) -> ModelResult<()> {
    validate_finite(name, value)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(ModelError::invalid(format!("{} must be > 0", name)))
    }
}

pub fn validate_binary_observations(obs: &[f64], name: &str) -> ModelResult<()> {
    if let Some((idx, value)) = obs
        .iter()
        .copied()
        .enumerate()
        .find(|(_, v)| !v.is_finite() || (*v != 0.0 && *v != 1.0))
    {
        return Err(ModelError::invalid(format!(
            "Bernoulli-logit likelihood '{}' requires binary observed values; found {} at index {}",
            name, value, idx
        )));
    }
    Ok(())
}

pub fn validate_positive_observations(obs: &[f64], name: &str, strict: bool) -> ModelResult<()> {
    if let Some((idx, value)) = obs
        .iter()
        .copied()
        .enumerate()
        .find(|(_, v)| !v.is_finite() || if strict { *v <= 0.0 } else { *v < 0.0 })
    {
        let relation = if strict {
            "strictly positive"
        } else {
            "non-negative"
        };
        return Err(ModelError::invalid(format!(
            "{} likelihood '{}' requires {} observed values; found {} at index {}",
            if strict { "LogNormal" } else { "Exponential" },
            name,
            relation,
            value,
            idx
        )));
    }
    Ok(())
}

pub fn validate_integer_observations(obs: &[f64], name: &str) -> ModelResult<()> {
    if let Some((idx, value)) = obs
        .iter()
        .copied()
        .enumerate()
        .find(|(_, v)| !v.is_finite() || *v < 0.0 || v.fract() != 0.0)
    {
        return Err(ModelError::invalid(format!(
            "NegativeBinomial likelihood '{}' requires non-negative integer observed values; found {} at index {}",
            name, value, idx
        )));
    }
    Ok(())
}

pub fn validate_count_observations(obs: &[f64], name: &str) -> ModelResult<()> {
    if let Some((idx, value)) = obs
        .iter()
        .copied()
        .enumerate()
        .find(|(_, v)| !v.is_finite() || *v < 0.0 || v.fract() != 0.0)
    {
        return Err(ModelError::invalid(format!(
            "Poisson-log likelihood '{}' requires non-negative integer observed values; found {} at index {}",
            name, value, idx
        )));
    }
    Ok(())
}

pub fn resolve_hyper(
    hp: &HyperParam,
    graph: &mut Graph,
    value_node_map: &HashMap<String, NodeId>,
) -> Result<NodeId, ModelError> {
    match hp {
        HyperParam::Const(v) => Ok(graph.add_constant(*v)),
        HyperParam::Param(name) => value_node_map.get(name.as_str()).copied().ok_or_else(|| {
            ModelError::parameter(format!(
                "hyperparameter '{}' has no value node. Declare it before the prior \
                 that references it.",
                name
            ))
        }),
    }
}

pub fn resolve_hyper_value(
    hp: &HyperParam,
    values: &HashMap<String, f64>,
    context: &str,
) -> Result<f64, ModelError> {
    match hp {
        HyperParam::Const(v) => Ok(*v),
        HyperParam::Param(name) => values.get(name).copied().ok_or_else(|| {
            let mut available: Vec<&str> = values.keys().map(String::as_str).collect();
            available.sort_unstable();
            ModelError::parameter(format!(
                "parameter '{}' referenced by {} has no value yet. It must be \
                 declared before the parameter that depends on it. \
                 Available at this point: [{}]",
                name,
                context,
                available.join(", ")
            ))
        }),
    }
}
