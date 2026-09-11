//! Declarative graph models shared by Python and standalone Rust applications.
use crate::autodiff::Evaluator;
use crate::data::{DataBinding, DataInputs, DataSchema, SlotKind};
use crate::distributions::{
    Bernoulli, BetaDist, Exponential, Gamma, HalfNormal, LogNormal, Normal, Poisson, StudentT,
    Uniform,
};
use crate::graph::{Graph, NodeId, ParamTransform};
use crate::param_ref::{validate_param_references, ParamRefError, ParamReference};
use crate::sampler::{self, SampleResult, SamplerConfig};
use std::collections::HashMap;
use std::sync::Arc;

pub type Data1d = HashMap<String, Vec<f64>>;
pub type Data2d = HashMap<String, (Vec<f64>, usize, usize)>;
type LinearTerms = Vec<(String, String)>;
pub type ModelResult<T> = Result<T, ModelError>;
#[derive(Debug, Clone, PartialEq)]
pub enum ModelError {
    Invalid(String),
    Parameter(String),
}
impl ModelError {
    pub fn invalid(message: impl Into<String>) -> Self {
        Self::Invalid(message.into())
    }
    pub fn parameter(message: impl Into<String>) -> Self {
        Self::Parameter(message.into())
    }
}
impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(s) | Self::Parameter(s) => f.write_str(s),
        }
    }
}
impl std::error::Error for ModelError {}
fn param_error(error: ParamRefError) -> ModelError {
    ModelError::parameter(error.to_string())
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelSpec {
    pub dimensions: HashMap<String, String>,
    pub potentials: Vec<(String, MuExpr)>,
    pub deterministics: Vec<(String, MuExpr)>,
    pub priors: Vec<PriorSpec>,
    pub likelihoods: Vec<LikelihoodSpec>,
    #[serde(skip)]
    pub bound_data_1d: HashMap<String, Vec<f64>>,
    #[serde(skip)]
    pub bound_data_2d: HashMap<String, (Vec<f64>, usize, usize)>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum DisplayParamSpec {
    Raw {
        name: String,
        raw_index: usize,
    },
    DerivedNonCenteredNormal {
        name: String,
        raw_index: usize,
        mu: HyperParam,
        sigma: HyperParam,
    },
}

#[derive(Debug, Clone)]
pub struct CompiledDefinition {
    pub graph: Graph,
    pub likelihood_names: Vec<String>,
    pub display_params: Vec<DisplayParamSpec>,
    pub auto_vector_params: HashMap<String, usize>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum HyperParam {
    Const(f64),
    /// Name of a parameter whose value node (post-transform) is used as the hyperparameter.
    Param(String),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum PriorSpec {
    Normal {
        name: String,
        mu: HyperParam,
        sigma: HyperParam,
    },
    HalfNormal {
        name: String,
        sigma: HyperParam,
    },
    Exponential {
        name: String,
        rate: HyperParam,
    },
    LogNormal {
        name: String,
        mu: HyperParam,
        sigma: HyperParam,
    },
    StudentT {
        name: String,
        nu: f64,
        mu: f64,
        sigma: f64,
    },
    Uniform {
        name: String,
        lower: f64,
        upper: f64,
    },
    Bernoulli {
        name: String,
        p: f64,
    },
    Poisson {
        name: String,
        lam: f64,
    },
    Gamma {
        name: String,
        alpha: f64,
        beta: f64,
    },
    Beta {
        name: String,
        alpha: f64,
        beta: f64,
    },
    VectorNormal {
        name: String,
        n: usize,
        mu: f64,
        sigma: f64,
    },
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum SigmaSpec {
    Const(f64),
    Param(String),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum LikelihoodFamily {
    Normal,
    BernoulliLogit,
    PoissonLog,
    ExponentialLog,
    LogNormal,
    NegativeBinomialLog,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LikelihoodSpec {
    pub family: LikelihoodFamily,
    pub name: String,
    pub mu_expr: MuExpr,
    pub sigma: Option<SigmaSpec>,
    pub observed_key: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum MuExpr {
    Data(String),
    Gather {
        param_name: String,
        data_key: String,
    },
    Unary(crate::graph::ElementwiseOp, Box<MuExpr>),
    Binary(crate::graph::ElementwiseOp, Box<MuExpr>, Box<MuExpr>),
    Sum(Box<MuExpr>),
    Const(f64),
    ParamTimesData {
        param_name: String,
        data_key: String,
    },
    /// Element-wise sum of two vector expressions.
    Add(Box<MuExpr>, Box<MuExpr>),
    /// Bare parameter broadcast-added to a vector expression.
    Param(String),
    /// faer-backed matrix-vector multiply: matrix_data_key @ vector_param.
    MatVec {
        param_name: String,
        data_key: String,
    },
}

impl MuExpr {
    pub fn is_scalar(&self) -> bool {
        match self {
            MuExpr::Const(_) | MuExpr::Sum(_) => true,
            MuExpr::Data(_) | MuExpr::Gather { .. } => false,
            MuExpr::Unary(_, a) => a.is_scalar(),
            MuExpr::Binary(_, a, b) => a.is_scalar() && b.is_scalar(),
            MuExpr::Param(_) => true,
            MuExpr::ParamTimesData { .. } => false,
            MuExpr::MatVec { .. } => false,
            MuExpr::Add(a, b) => a.is_scalar() && b.is_scalar(),
        }
    }
}

pub fn collect_expr_param_names(expr: &MuExpr, out: &mut Vec<String>) {
    match expr {
        MuExpr::Const(_) | MuExpr::Data(_) => {}
        MuExpr::Unary(_, a) | MuExpr::Sum(a) => collect_expr_param_names(a, out),
        MuExpr::Param(name) => out.push(name.clone()),
        MuExpr::ParamTimesData { param_name, .. }
        | MuExpr::MatVec { param_name, .. }
        | MuExpr::Gather { param_name, .. } => out.push(param_name.clone()),
        MuExpr::Add(a, b) | MuExpr::Binary(_, a, b) => {
            collect_expr_param_names(a, out);
            collect_expr_param_names(b, out);
        }
    }
}

impl ModelSpec {
    pub fn structure_definition(&self) -> Self {
        let mut definition = self.clone();
        definition.bound_data_1d.clear();
        definition.bound_data_2d.clear();
        definition
    }
}

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

pub fn should_auto_noncenter(
    prior: &PriorSpec,
    auto_vector_params: &HashMap<String, usize>,
) -> bool {
    match prior {
        PriorSpec::Normal { name, mu, sigma } => {
            !auto_vector_params.contains_key(name)
                && (matches!(mu, HyperParam::Param(_)) || matches!(sigma, HyperParam::Param(_)))
        }
        _ => false,
    }
}

pub fn append_raw_display_params(
    display_params: &mut Vec<DisplayParamSpec>,
    graph: &Graph,
    start_idx: usize,
) {
    for raw_index in start_idx..graph.param_count {
        display_params.push(DisplayParamSpec::Raw {
            name: graph.param_names[raw_index].clone(),
            raw_index,
        });
    }
}

pub fn build_likelihood_into_graph(
    graph: &mut Graph,
    lik: &LikelihoodSpec,
    data_map: &HashMap<String, Vec<f64>>,
    matrix_map: &HashMap<String, (Vec<f64>, usize, usize)>,
    vector_param_map: &HashMap<String, (usize, usize)>,
    value_node_map: &HashMap<String, NodeId>,
) -> ModelResult<()> {
    let linpred_node = build_mu_expr(
        graph,
        &lik.mu_expr,
        data_map,
        matrix_map,
        vector_param_map,
        value_node_map,
    )?;

    let obs_vec = data_map
        .get(&lik.observed_key)
        .ok_or_else(|| {
            ModelError::invalid(format!("Missing observed data key: {}", lik.observed_key))
        })?
        .clone();
    let obs_idx = graph.add_named_obs_data(&lik.observed_key, &lik.name, obs_vec.clone());

    let linpred_node = if lik.mu_expr.is_scalar() {
        graph.broadcast_observation(linpred_node, obs_idx)
    } else {
        linpred_node
    };

    match lik.family {
        LikelihoodFamily::Normal => {
            let sigma_spec = lik.sigma.as_ref().ok_or_else(|| {
                ModelError::invalid(format!("Normal likelihood '{}' is missing sigma", lik.name))
            })?;
            let sigma_node = resolve_sigma(sigma_spec, graph, value_node_map)?;
            graph.normal_obs_logp(linpred_node, sigma_node, obs_idx);
        }
        LikelihoodFamily::BernoulliLogit => {
            validate_binary_observations(&obs_vec, &lik.name)?;
            graph.obs_logp_bernoulli_logit(linpred_node, obs_idx);
        }
        LikelihoodFamily::PoissonLog => {
            validate_count_observations(&obs_vec, &lik.name)?;
            graph.obs_logp_poisson_log(linpred_node, obs_idx);
        }
        LikelihoodFamily::ExponentialLog => {
            validate_positive_observations(&obs_vec, &lik.name, false)?;
            graph.obs_logp_exponential_log(linpred_node, obs_idx);
        }
        LikelihoodFamily::LogNormal => {
            validate_positive_observations(&obs_vec, &lik.name, true)?;
            let sigma_spec = lik.sigma.as_ref().ok_or_else(|| {
                ModelError::invalid(format!(
                    "LogNormal likelihood '{}' is missing sigma",
                    lik.name
                ))
            })?;
            let sigma_node = resolve_sigma(sigma_spec, graph, value_node_map)?;
            graph.obs_logp_lognormal(linpred_node, sigma_node, obs_idx);
        }
        LikelihoodFamily::NegativeBinomialLog => {
            validate_integer_observations(&obs_vec, &lik.name)?;
            let alpha_spec = lik.sigma.as_ref().ok_or_else(|| {
                ModelError::invalid(format!(
                    "NegativeBinomial likelihood '{}' is missing alpha",
                    lik.name
                ))
            })?;
            let alpha_node = resolve_sigma(alpha_spec, graph, value_node_map)?;
            graph.obs_logp_negative_binomial_log(linpred_node, alpha_node, obs_idx);
        }
    }
    Ok(())
}

pub fn build_prior_into_graph(
    prior: &PriorSpec,
    graph: &mut Graph,
    vector_param_map: &mut HashMap<String, (usize, usize)>,
    value_node_map: &mut HashMap<String, NodeId>,
    auto_vector_params: &HashMap<String, usize>,
    display_params: &mut Vec<DisplayParamSpec>,
) -> Result<(), ModelError> {
    match prior {
        PriorSpec::Normal { name, mu, sigma } => {
            let start_idx = graph.param_count;
            if should_auto_noncenter(prior, auto_vector_params) {
                let raw_name = format!("{}__raw", name);
                let raw = graph.add_param(&raw_name);
                let zero = graph.add_constant(0.0);
                let one = graph.add_constant(1.0);
                graph.normal_logp(raw, zero, one);
                let mu_node = resolve_hyper(mu, graph, value_node_map)?;
                let sigma_node = resolve_hyper(sigma, graph, value_node_map)?;
                let scaled = graph.mul(sigma_node, raw);
                let v = graph.add(mu_node, scaled);
                value_node_map.insert(name.clone(), v);
                display_params.push(DisplayParamSpec::DerivedNonCenteredNormal {
                    name: name.clone(),
                    raw_index: start_idx,
                    mu: mu.clone(),
                    sigma: sigma.clone(),
                });
            } else if let Some(&n) = auto_vector_params.get(name) {
                // MatVec auto-promotion: constant hyperparams only
                let (mu_f, sigma_f) = match (mu, sigma) {
                    (HyperParam::Const(m), HyperParam::Const(s)) => (*m, *s),
                    _ => {
                        return Err(ModelError::invalid(format!(
                            "Parameter '{}' is used in a matrix multiply (@) but has hierarchical \
                         hyperparameters. Hierarchical vector params are not yet supported.",
                            name
                        )))
                    }
                };
                let param_start = graph.add_vector_params(name, n);
                vector_param_map.insert(name.clone(), (param_start, n));
                graph.vector_normal_logp(param_start, n, mu_f, sigma_f);
                append_raw_display_params(display_params, graph, start_idx);
            } else {
                let mu_node = resolve_hyper(mu, graph, value_node_map)?;
                let sigma_node = resolve_hyper(sigma, graph, value_node_map)?;
                let v = Normal::prior_with_nodes(graph, name, mu_node, sigma_node);
                value_node_map.insert(name.clone(), v);
                append_raw_display_params(display_params, graph, start_idx);
            }
        }
        PriorSpec::HalfNormal { name, sigma } => {
            let start_idx = graph.param_count;
            if let Some(&n) = auto_vector_params.get(name) {
                let sigma_f = match sigma {
                    HyperParam::Const(s) => *s,
                    _ => {
                        return Err(ModelError::invalid(format!(
                        "Parameter '{}' is used in a matrix multiply (@) but has a hierarchical \
                         sigma. Hierarchical vector params are not yet supported.",
                        name
                    )))
                    }
                };
                let param_start =
                    graph.add_vector_params_with_transform(name, n, ParamTransform::Exp);
                vector_param_map.insert(name.clone(), (param_start, n));
                graph.vector_half_normal_logp(param_start, n, sigma_f);
            } else {
                let sigma_node = resolve_hyper(sigma, graph, value_node_map)?;
                let v = HalfNormal::prior_with_node_sigma(graph, name, sigma_node);
                value_node_map.insert(name.clone(), v);
            }
            append_raw_display_params(display_params, graph, start_idx);
        }
        PriorSpec::Exponential { name, rate } => {
            let start_idx = graph.param_count;
            if let Some(&n) = auto_vector_params.get(name) {
                let rate_f = match rate {
                    HyperParam::Const(r) => *r,
                    _ => {
                        return Err(ModelError::invalid(format!(
                        "Parameter '{}' is used in a matrix multiply (@) but has a hierarchical \
                         rate. Hierarchical vector params are not yet supported.",
                        name
                    )))
                    }
                };
                let param_start =
                    graph.add_vector_params_with_transform(name, n, ParamTransform::Exp);
                vector_param_map.insert(name.clone(), (param_start, n));
                graph.vector_gamma_logp(param_start, n, 1.0, rate_f);
            } else {
                let rate_node = resolve_hyper(rate, graph, value_node_map)?;
                let v = Exponential::prior_with_node_rate(graph, name, rate_node);
                value_node_map.insert(name.clone(), v);
            }
            append_raw_display_params(display_params, graph, start_idx);
        }
        PriorSpec::LogNormal { name, mu, sigma } => {
            let start_idx = graph.param_count;
            if let Some(&n) = auto_vector_params.get(name) {
                let (mu_f, sigma_f) = match (mu, sigma) {
                    (HyperParam::Const(m), HyperParam::Const(s)) => (*m, *s),
                    _ => return Err(ModelError::invalid(format!(
                        "Parameter '{}' is used in a matrix multiply (@) but has hierarchical \
                         LogNormal hyperparameters. Hierarchical vector params are not yet supported.",
                        name
                    ))),
                };
                let param_start =
                    graph.add_vector_params_with_transform(name, n, ParamTransform::Exp);
                vector_param_map.insert(name.clone(), (param_start, n));
                graph.vector_normal_logp(param_start, n, mu_f, sigma_f);
            } else {
                let mu_node = resolve_hyper(mu, graph, value_node_map)?;
                let sigma_node = resolve_hyper(sigma, graph, value_node_map)?;
                let v = LogNormal::prior_with_nodes(graph, name, mu_node, sigma_node);
                value_node_map.insert(name.clone(), v);
            }
            append_raw_display_params(display_params, graph, start_idx);
        }
        PriorSpec::StudentT {
            name,
            nu,
            mu,
            sigma,
        } => {
            let start_idx = graph.param_count;
            if let Some(&n) = auto_vector_params.get(name) {
                let param_start = graph.add_vector_params(name, n);
                vector_param_map.insert(name.clone(), (param_start, n));
                graph.vector_student_t_logp(param_start, n, *nu, *mu, *sigma);
            } else {
                let v = StudentT::prior(graph, name, *nu, *mu, *sigma);
                value_node_map.insert(name.clone(), v);
            }
            append_raw_display_params(display_params, graph, start_idx);
        }
        PriorSpec::Uniform { name, lower, upper } => {
            let start_idx = graph.param_count;
            if let Some(&n) = auto_vector_params.get(name) {
                let param_start = graph.add_vector_params_with_transform(
                    name,
                    n,
                    ParamTransform::BoundedSigmoid {
                        lower: *lower,
                        upper: *upper,
                    },
                );
                vector_param_map.insert(name.clone(), (param_start, n));
                graph.vector_uniform_logp(param_start, n, *lower, *upper);
            } else {
                let v = Uniform::prior(graph, name, *lower, *upper);
                value_node_map.insert(name.clone(), v);
            }
            append_raw_display_params(display_params, graph, start_idx);
        }
        PriorSpec::Bernoulli { name, p } => {
            let start_idx = graph.param_count;
            if auto_vector_params.contains_key(name) {
                return Err(ModelError::invalid(format!(
                    "Parameter '{}' is used with @ but has a Bernoulli prior. \
                     Discrete distributions cannot be auto-promoted to vector params.",
                    name
                )));
            }
            let v = Bernoulli::prior(graph, name, *p);
            value_node_map.insert(name.clone(), v);
            append_raw_display_params(display_params, graph, start_idx);
        }
        PriorSpec::Poisson { name, lam } => {
            let start_idx = graph.param_count;
            if auto_vector_params.contains_key(name) {
                return Err(ModelError::invalid(format!(
                    "Parameter '{}' is used with @ but has a Poisson prior. \
                     Discrete distributions cannot be auto-promoted to vector params.",
                    name
                )));
            }
            let v = Poisson::prior(graph, name, *lam);
            value_node_map.insert(name.clone(), v);
            append_raw_display_params(display_params, graph, start_idx);
        }
        PriorSpec::Gamma { name, alpha, beta } => {
            let start_idx = graph.param_count;
            if let Some(&n) = auto_vector_params.get(name) {
                let param_start =
                    graph.add_vector_params_with_transform(name, n, ParamTransform::Exp);
                vector_param_map.insert(name.clone(), (param_start, n));
                graph.vector_gamma_logp(param_start, n, *alpha, *beta);
            } else {
                let v = Gamma::prior(graph, name, *alpha, *beta);
                value_node_map.insert(name.clone(), v);
            }
            append_raw_display_params(display_params, graph, start_idx);
        }
        PriorSpec::Beta { name, alpha, beta } => {
            let start_idx = graph.param_count;
            if let Some(&n) = auto_vector_params.get(name) {
                let param_start =
                    graph.add_vector_params_with_transform(name, n, ParamTransform::Sigmoid);
                vector_param_map.insert(name.clone(), (param_start, n));
                graph.vector_beta_logp(param_start, n, *alpha, *beta);
            } else {
                let v = BetaDist::prior(graph, name, *alpha, *beta);
                value_node_map.insert(name.clone(), v);
            }
            append_raw_display_params(display_params, graph, start_idx);
        }
        PriorSpec::VectorNormal { name, n, mu, sigma } => {
            let start_idx = graph.param_count;
            let param_start = graph.add_vector_params(name, *n);
            vector_param_map.insert(name.clone(), (param_start, *n));
            graph.vector_normal_logp(param_start, *n, *mu, *sigma);
            append_raw_display_params(display_params, graph, start_idx);
        }
    }
    Ok(())
}

pub fn resolve_sigma(
    spec: &SigmaSpec,
    graph: &mut Graph,
    value_node_map: &HashMap<String, NodeId>,
) -> Result<NodeId, ModelError> {
    match spec {
        SigmaSpec::Const(v) => Ok(graph.add_constant(*v)),
        SigmaSpec::Param(name) => value_node_map.get(name.as_str()).copied().ok_or_else(|| {
            let mut available: Vec<&str> = value_node_map.keys().map(String::as_str).collect();
            available.sort_unstable();
            ModelError::parameter(format!(
                "scale parameter '{}' is not a scalar parameter of this model. \
                 Scalar parameters: [{}]",
                name,
                available.join(", ")
            ))
        }),
    }
}

pub fn compile(
    model_spec: &ModelSpec,
    data_map: &HashMap<String, Vec<f64>>,
    matrix_map: &HashMap<String, (Vec<f64>, usize, usize)>,
) -> ModelResult<CompiledDefinition> {
    validate_definition(model_spec)?;
    let mut graph = Graph::new();
    let mut vector_param_map: HashMap<String, (usize, usize)> = HashMap::new();
    let mut value_node_map: HashMap<String, NodeId> = HashMap::new();
    let mut display_params = Vec::new();

    // Defence in depth: `ModelBuilder.build()` already validated these, but a
    // `ModelSpec` can reach here by other routes (pickling, batch_sample).
    // Validating before touching the graph guarantees an unresolvable reference
    // never becomes a silently-defaulted value at sampling time.
    validate_model_references(&model_spec.priors, &model_spec.likelihoods)?;

    let auto_vector_params = collect_matvec_params(model_spec, matrix_map)?;

    for prior in &model_spec.priors {
        build_prior_into_graph(
            prior,
            &mut graph,
            &mut vector_param_map,
            &mut value_node_map,
            &auto_vector_params,
            &mut display_params,
        )?;
    }

    for lik in &model_spec.likelihoods {
        build_likelihood_into_graph(
            &mut graph,
            lik,
            data_map,
            matrix_map,
            &vector_param_map,
            &value_node_map,
        )?;
    }

    for (name, expr) in &model_spec.potentials {
        let node = build_mu_expr(
            &mut graph,
            expr,
            data_map,
            matrix_map,
            &vector_param_map,
            &value_node_map,
        )?;
        graph.add_logp_term(node);
        let _ = name;
    }
    for (name, expr) in &model_spec.deterministics {
        if model_spec.priors.iter().any(|p| prior_name(p) == name)
            || model_spec.likelihoods.iter().any(|l| l.name == *name)
        {
            return Err(ModelError::invalid(
                "deterministic name collides with another output",
            ));
        }
        let node = build_mu_expr(
            &mut graph,
            expr,
            data_map,
            matrix_map,
            &vector_param_map,
            &value_node_map,
        )?;
        graph.deterministics.push((name.clone(), node));
    }
    for slot in graph
        .schema
        .vectors
        .iter_mut()
        .chain(&mut graph.schema.observations)
        .chain(&mut graph.schema.matrices)
    {
        if let Some(dim) = model_spec.dimensions.get(&slot.key) {
            slot.dim = dim.clone();
        }
    }
    let mut output_names = std::collections::HashSet::new();
    for name in graph
        .param_names
        .iter()
        .chain(model_spec.likelihoods.iter().map(|l| &l.name))
        .chain(graph.deterministics.iter().map(|(n, _)| n))
    {
        if !output_names.insert(name) {
            return Err(ModelError::invalid(format!(
                "output name '{name}' is not unique"
            )));
        }
    }
    graph
        .validate_shapes()
        .map_err(|e| ModelError::invalid(e.to_string()))?;

    Ok(CompiledDefinition {
        graph,
        likelihood_names: model_spec
            .likelihoods
            .iter()
            .map(|l| l.name.clone())
            .collect(),
        display_params,
        auto_vector_params,
    })
}

pub fn derive_display_draw(raw_draw: &[f64], specs: &[DisplayParamSpec]) -> ModelResult<Vec<f64>> {
    let mut values = HashMap::new();
    let mut out = Vec::with_capacity(specs.len());
    for spec in specs {
        let value = match spec {
            DisplayParamSpec::Raw { name, raw_index } => {
                let value = raw_draw[*raw_index];
                values.insert(name.clone(), value);
                value
            }
            DisplayParamSpec::DerivedNonCenteredNormal {
                name,
                raw_index,
                mu,
                sigma,
            } => {
                let context = format!("non-centered parameter '{}'", name);
                let mu_v = resolve_hyper_value(mu, &values, &context)?;
                let sigma_v = resolve_hyper_value(sigma, &values, &context)?;
                let value = mu_v + sigma_v * raw_draw[*raw_index];
                values.insert(name.clone(), value);
                value
            }
        };
        out.push(value);
    }
    Ok(out)
}

pub fn derive_display_sample_result(
    raw_result: &SampleResult,
    specs: &[DisplayParamSpec],
) -> ModelResult<SampleResult> {
    let mut samples = Vec::with_capacity(raw_result.samples.len());
    for chain in &raw_result.samples {
        let mut chain_out = Vec::with_capacity(chain.len());
        for draw in chain {
            chain_out.push(derive_display_draw(draw, specs)?);
        }
        samples.push(chain_out);
    }
    let param_names = specs
        .iter()
        .map(|spec| match spec {
            DisplayParamSpec::Raw { name, .. } => name.clone(),
            DisplayParamSpec::DerivedNonCenteredNormal { name, .. } => name.clone(),
        })
        .collect();

    Ok(SampleResult {
        samples,
        accept_rates: raw_result.accept_rates.clone(),
        step_sizes: raw_result.step_sizes.clone(),
        divergences: raw_result.divergences.clone(),
        transitions: raw_result.transitions.clone(),
        param_names,
    })
}

/// A fused intercept retains its expression type independently of parameter names.
enum LinearIntercept {
    Constant(f64),
    Parameter(String),
}

/// Try to decompose an expression into data-weighted terms and a typed intercept.
fn try_extract_linear(expr: &MuExpr) -> Option<(LinearTerms, Option<LinearIntercept>)> {
    let mut terms = Vec::new();
    let mut intercept: Option<LinearIntercept> = None;

    fn walk(
        e: &MuExpr,
        terms: &mut Vec<(String, String)>,
        intercept: &mut Option<LinearIntercept>,
    ) -> bool {
        match e {
            MuExpr::Const(value) if intercept.is_none() => {
                *intercept = Some(LinearIntercept::Constant(*value));
                true
            }
            MuExpr::ParamTimesData {
                param_name,
                data_key,
            } => {
                terms.push((param_name.clone(), data_key.clone()));
                true
            }
            MuExpr::Add(a, b) => walk(a, terms, intercept) && walk(b, terms, intercept),
            MuExpr::Param(name) if intercept.is_none() => {
                *intercept = Some(LinearIntercept::Parameter(name.clone()));
                true
            }
            // MatVec uses faer GEMV — never fuse into scalar linear combination
            _ => false,
        }
    }

    if walk(expr, &mut terms, &mut intercept) && !terms.is_empty() {
        Some((terms, intercept))
    } else {
        None
    }
}

/// Walk all likelihood MuExpr trees and collect param names used in MatVec ops.
/// Returns a set of param names that should be auto-promoted to vector params.
pub fn collect_matvec_params(
    spec: &ModelSpec,
    matrix_map: &HashMap<String, (Vec<f64>, usize, usize)>,
) -> Result<HashMap<String, usize>, ModelError> {
    let mut result = HashMap::new();

    fn walk(
        expr: &MuExpr,
        matrix_map: &HashMap<String, (Vec<f64>, usize, usize)>,
        out: &mut HashMap<String, usize>,
    ) -> Result<(), ModelError> {
        match expr {
            MuExpr::MatVec {
                param_name,
                data_key,
            } => {
                let (_data, _n_rows, n_cols) =
                    matrix_map.get(data_key.as_str()).ok_or_else(|| {
                        ModelError::invalid(format!(
                            "Missing matrix key '{}' in data dict",
                            data_key
                        ))
                    })?;
                out.insert(param_name.clone(), *n_cols);
                Ok(())
            }
            MuExpr::Add(a, b) | MuExpr::Binary(_, a, b) => {
                walk(a, matrix_map, out)?;
                walk(b, matrix_map, out)?;
                Ok(())
            }
            MuExpr::Unary(_, a) | MuExpr::Sum(a) => walk(a, matrix_map, out),
            _ => Ok(()),
        }
    }

    for expr in spec.likelihoods.iter().map(|lik| &lik.mu_expr).chain(
        spec.potentials
            .iter()
            .chain(&spec.deterministics)
            .map(|(_, expr)| expr),
    ) {
        walk(expr, matrix_map, &mut result)?;
    }

    Ok(result)
}

/// Look up the post-transform value node for a scalar parameter.
///
/// Fails loudly — never substitutes a default — when the name is not a scalar
/// parameter of this model.
fn lookup_param_value_node(
    name: &str,
    value_node_map: &HashMap<String, NodeId>,
    context: &str,
) -> Result<NodeId, ModelError> {
    value_node_map.get(name).copied().ok_or_else(|| {
        let mut available: Vec<&str> = value_node_map.keys().map(String::as_str).collect();
        available.sort_unstable();
        ModelError::parameter(format!(
            "parameter '{}' used in {} is not a scalar parameter of this model. \
             Scalar parameters: [{}]",
            name,
            context,
            available.join(", ")
        ))
    })
}

/// Compile a MuExpr tree into graph nodes.
///
/// Parameters are resolved through `value_node_map`, which holds the
/// *post-transform* value node for every scalar parameter. Resolving via
/// `Graph::node_by_name` instead would return the unconstrained raw node for
/// any transformed prior (HalfNormal, Exponential, LogNormal, Uniform, Gamma,
/// Beta), silently putting a log-scale value into the linear predictor.
///
/// When the tree is a pure linear combination (Σ βₖ xₖ + optional intercept),
/// this emits a single FusedLinearMu op instead of individual
/// ScalarMulData / VectorAdd / ScalarBroadcastAdd nodes.
pub fn build_mu_expr(
    graph: &mut Graph,
    expr: &MuExpr,
    data_map: &HashMap<String, Vec<f64>>,
    matrix_map: &HashMap<String, (Vec<f64>, usize, usize)>,
    vector_param_map: &HashMap<String, (usize, usize)>,
    value_node_map: &HashMap<String, NodeId>,
) -> Result<NodeId, ModelError> {
    // Fast path: fuse linear combinations into a single op
    if let Some((terms, intercept)) = try_extract_linear(expr) {
        let mut param_nodes = Vec::with_capacity(terms.len());
        let mut data_indices = Vec::with_capacity(terms.len());

        for (param_name, data_key) in &terms {
            let pn = lookup_param_value_node(param_name, value_node_map, "a linear predictor")?;
            param_nodes.push(pn);

            let data_vec = data_map
                .get(data_key)
                .ok_or_else(|| ModelError::invalid(format!("Missing data key: {}", data_key)))?
                .clone();
            data_indices.push(graph.store_named_data_vec(data_key, data_vec));
        }

        let intercept_node = match intercept {
            Some(LinearIntercept::Constant(value)) => Some(graph.add_constant(value)),
            Some(LinearIntercept::Parameter(name)) => Some(lookup_param_value_node(
                &name,
                value_node_map,
                "the intercept of a linear predictor",
            )?),
            None => None,
        };

        return Ok(graph.fused_linear_mu(param_nodes, data_indices, intercept_node));
    }

    // Fallback: individual ops
    match expr {
        MuExpr::Unary(op, a) => {
            let a = build_mu_expr(
                graph,
                a,
                data_map,
                matrix_map,
                vector_param_map,
                value_node_map,
            )?;
            Ok(graph.elementwise(*op, a, None))
        }
        MuExpr::Sum(a) => {
            let a = build_mu_expr(
                graph,
                a,
                data_map,
                matrix_map,
                vector_param_map,
                value_node_map,
            )?;
            Ok(graph.sum(a))
        }
        MuExpr::Binary(op, a, b) => {
            let a = build_mu_expr(
                graph,
                a,
                data_map,
                matrix_map,
                vector_param_map,
                value_node_map,
            )?;
            let b = build_mu_expr(
                graph,
                b,
                data_map,
                matrix_map,
                vector_param_map,
                value_node_map,
            )?;
            Ok(graph.elementwise(*op, a, Some(b)))
        }
        MuExpr::Data(key) => Ok(graph.add_data(
            key,
            data_map
                .get(key)
                .ok_or_else(|| ModelError::invalid(format!("missing data key {key}")))?
                .clone(),
        )),
        MuExpr::Gather {
            param_name,
            data_key,
        } => {
            let (start, n) = *vector_param_map
                .get(param_name)
                .ok_or_else(|| ModelError::invalid("indexing requires a vector parameter"))?;
            let data = graph.add_data(
                data_key,
                data_map
                    .get(data_key)
                    .ok_or_else(|| ModelError::invalid(format!("missing index key {data_key}")))?
                    .clone(),
            );
            Ok(graph.gather(start, n, data))
        }
        MuExpr::Const(value) => Ok(graph.add_constant(*value)),
        MuExpr::ParamTimesData {
            param_name,
            data_key,
        } => {
            let param_node =
                lookup_param_value_node(param_name, value_node_map, "a linear predictor")?;
            let data_vec = data_map
                .get(data_key)
                .ok_or_else(|| ModelError::invalid(format!("Missing data key: {}", data_key)))?
                .clone();
            let data_node = graph.add_data(data_key, data_vec);
            Ok(graph.scalar_mul_data(param_node, data_node))
        }
        MuExpr::Param(name) => lookup_param_value_node(name, value_node_map, "a linear predictor"),
        MuExpr::MatVec {
            param_name,
            data_key,
        } => {
            let &(param_start, n_params) =
                vector_param_map.get(param_name.as_str()).ok_or_else(|| {
                    ModelError::invalid(format!(
                        "Unknown vector param '{}' — did you call vector_normal_prior?",
                        param_name
                    ))
                })?;
            let (data, n_rows, n_cols) = matrix_map.get(data_key.as_str()).ok_or_else(|| {
                ModelError::invalid(format!("Missing matrix key '{}' in data dict", data_key))
            })?;
            let matrix_idx = graph.store_named_matrix(data_key, data.clone(), *n_rows, *n_cols);
            Ok(graph.mat_vec_mul(matrix_idx, param_start, n_params, None))
        }
        MuExpr::Add(a, b) => {
            let na = build_mu_expr(
                graph,
                a,
                data_map,
                matrix_map,
                vector_param_map,
                value_node_map,
            )?;
            let nb = build_mu_expr(
                graph,
                b,
                data_map,
                matrix_map,
                vector_param_map,
                value_node_map,
            )?;
            let a_scalar = a.is_scalar();
            let b_scalar = b.is_scalar();
            if a_scalar && !b_scalar {
                Ok(graph.scalar_broadcast_add(na, nb))
            } else if !a_scalar && b_scalar {
                Ok(graph.scalar_broadcast_add(nb, na))
            } else if !a_scalar && !b_scalar {
                Ok(graph.vector_add(na, nb))
            } else {
                Ok(graph.add(na, nb))
            }
        }
    }
}

fn validate_definition(spec: &ModelSpec) -> ModelResult<()> {
    let positive_hyper = |hp: &HyperParam| -> ModelResult<()> {
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
                    return Err(ModelError::invalid("uniform bounds must be increasing"));
                }
            }
            PriorSpec::Bernoulli { p, .. } => {
                if !(0.0..=1.0).contains(p) {
                    return Err(ModelError::invalid(
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
                    return Err(ModelError::invalid(
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
            return Err(ModelError::invalid(
                "potentials must have unique names and scalar expressions",
            ));
        }
    }
    Ok(())
}

/// The existing Python wire format, now owned by the Rust core.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelArtifact {
    pub format: String,
    pub version: u32,
    pub definition: ModelSpec,
    pub schema: DataSchema,
}

#[derive(Clone, Debug)]
pub struct GraphModel {
    pub definition: ModelSpec,
    pub structure: Arc<Graph>,
    pub likelihood_names: Vec<String>,
    pub display_params: Vec<DisplayParamSpec>,
}
impl GraphModel {
    pub fn from_json(text: &str) -> ModelResult<Self> {
        let artifact: ModelArtifact = serde_json::from_str(text)
            .map_err(|e| ModelError::invalid(format!("invalid graph model artifact: {e}")))?;
        Self::from_artifact(artifact)
    }
    pub fn from_artifact(artifact: ModelArtifact) -> ModelResult<Self> {
        if artifact.format != "rustmc.graph-model" || artifact.version != 1 {
            return Err(ModelError::invalid(
                "unsupported graph model artifact format/version",
            ));
        }
        let mut definition = artifact.definition.structure_definition();
        reject_discrete_priors_for_gradient_sampling(&definition.priors)?;
        for slot in &artifact.schema.matrices {
            let SlotKind::Matrix { n_cols } = slot.kind else {
                return Err(ModelError::invalid("invalid matrix schema"));
            };
            if n_cols == 0 || n_cols > 1_000_000 {
                return Err(ModelError::invalid(
                    "matrix column count must be between 1 and 1000000",
                ));
            }
            definition
                .bound_data_2d
                .insert(slot.key.clone(), (vec![0.; n_cols], 1, n_cols));
        }
        let (one_d, two_d) = template_data_for_spec(&definition)?;
        let compiled = compile(&definition, &one_d, &two_d)?;
        if compiled.graph.schema != artifact.schema {
            return Err(ModelError::invalid(
                "artifact schema disagrees with its model definition",
            ));
        }
        Ok(Self {
            definition: definition.structure_definition(),
            structure: Arc::new(compiled.graph.structure_only()),
            likelihood_names: compiled.likelihood_names,
            display_params: compiled.display_params,
        })
    }
    pub fn to_json(&self) -> ModelResult<String> {
        serde_json::to_string(&self.artifact()).map_err(|e| ModelError::invalid(e.to_string()))
    }
    pub fn artifact(&self) -> ModelArtifact {
        ModelArtifact {
            format: "rustmc.graph-model".into(),
            version: 1,
            definition: self.definition.structure_definition(),
            schema: self.structure.schema.clone(),
        }
    }
    pub fn bind(&self, inputs: DataInputs, id: impl Into<String>) -> ModelResult<DataBinding> {
        let binding = DataBinding::bind(&self.structure.schema, inputs, id.into(), true, true)
            .map_err(|e| ModelError::invalid(e.to_string()))?;
        binding
            .validate_for(&self.structure)
            .map_err(|e| ModelError::invalid(e.to_string()))?;
        Ok(binding)
    }
    /// Positions and returned gradients use the graph's unconstrained parameter order.
    pub fn log_density(
        &self,
        binding: &DataBinding,
        position: &[f64],
    ) -> ModelResult<(f64, Vec<f64>)> {
        binding
            .validate_for(&self.structure)
            .map_err(|e| ModelError::invalid(e.to_string()))?;
        if position.len() != self.structure.param_count || position.iter().any(|x| !x.is_finite()) {
            return Err(ModelError::invalid(
                "position must match the finite unconstrained parameter axis",
            ));
        }
        let graph = self.structure.with_binding(binding);
        let mut evaluator = Evaluator::new(&graph);
        evaluator.compute(&graph, position);
        Ok((evaluator.total_logp, evaluator.grad))
    }
    pub fn sample(
        &self,
        binding: DataBinding,
        config: SamplerConfig,
        initial: Option<Vec<Vec<f64>>>,
    ) -> ModelResult<ModelFit> {
        let raw = sampler::sample_bound_with_init(
            Arc::clone(&self.structure),
            binding.clone(),
            config,
            initial,
        )
        .map_err(ModelError::invalid)?;
        let samples = derive_display_sample_result(&raw, &self.display_params)?;
        Ok(ModelFit {
            model: self.clone(),
            binding,
            raw,
            samples,
        })
    }
}

/// Posterior predictions indexed by response, chain, draw, and observation.
pub type Prediction = HashMap<String, Vec<Vec<Vec<f64>>>>;
#[derive(Clone, Debug)]
pub struct ModelFit {
    model: GraphModel,
    binding: DataBinding,
    raw: SampleResult,
    pub samples: SampleResult,
}
impl ModelFit {
    /// Predict at new inputs. Response placeholders are constructed internally.
    pub fn predict(
        &self,
        inputs: DataInputs,
        sizes: HashMap<String, usize>,
        seed: u64,
        expected: bool,
    ) -> ModelResult<Prediction> {
        use rand::SeedableRng;
        let graph = self.model.structure.with_binding(&self.binding);
        let prediction_graph = bind_prediction(&graph, inputs, sizes)?;
        let heads = prediction_graph.observation_heads();
        let mut evaluator = Evaluator::new(&prediction_graph);
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let mut output: Prediction = self
            .model
            .likelihood_names
            .iter()
            .map(|name| (name.clone(), Vec::new()))
            .collect();
        for chain in &self.raw.samples {
            let mut values = vec![Vec::with_capacity(chain.len()); heads.len()];
            for draw in chain {
                let position = draw
                    .iter()
                    .zip(&prediction_graph.param_transforms)
                    .map(|(&v, t)| match t {
                        ParamTransform::Identity => v,
                        ParamTransform::Exp => v.ln(),
                        ParamTransform::Sigmoid => v.ln() - (-v).ln_1p(),
                        ParamTransform::BoundedSigmoid { lower, upper } => {
                            let p = (v - lower) / (upper - lower);
                            p.ln() - (-p).ln_1p()
                        }
                    })
                    .collect::<Vec<_>>();
                evaluator.compute(&prediction_graph, &position);
                for (i, head) in heads.iter().enumerate() {
                    let aux = head.aux.map(|n| evaluator.scalar_at(n));
                    let mut observations = Vec::with_capacity(head.n_obs);
                    for j in 0..head.n_obs {
                        let eta = evaluator.vec_elem(head.linpred, j, &prediction_graph);
                        observations.push(
                            if expected {
                                crate::observation::mean(head.family, eta, aux)
                            } else {
                                crate::observation::sample(head.family, eta, aux, &mut rng)
                            }
                            .map_err(ModelError::invalid)?,
                        );
                    }
                    values[i].push(observations);
                }
            }
            for (name, values) in self.model.likelihood_names.iter().zip(values) {
                output
                    .get_mut(name)
                    .expect("model response was initialized")
                    .push(values);
            }
        }
        Ok(output)
    }
}

pub fn bind_prediction(
    graph: &Graph,
    mut inputs: DataInputs,
    mut lengths: HashMap<String, usize>,
) -> ModelResult<Graph> {
    for dimension in lengths.keys() {
        if !graph
            .schema
            .vectors
            .iter()
            .chain(&graph.schema.observations)
            .chain(&graph.schema.matrices)
            .any(|slot| &slot.dim == dimension)
        {
            return Err(ModelError::invalid(format!(
                "unknown prediction dimension '{dimension}'"
            )));
        }
    }
    for slot in graph.schema.vectors.iter().chain(&graph.schema.matrices) {
        let len = inputs
            .vectors
            .get(&slot.key)
            .map(|v| v.len())
            .or_else(|| inputs.matrices.get(&slot.key).map(|m| m.n_rows))
            .ok_or_else(|| {
                ModelError::invalid(format!("missing prediction data key '{}'", slot.key))
            })?;
        if lengths
            .insert(slot.dim.clone(), len)
            .is_some_and(|n| n != len)
        {
            return Err(ModelError::invalid(format!(
                "prediction dimension '{}' has inconsistent lengths",
                slot.dim
            )));
        }
    }
    for (i, slot) in graph.schema.observations.iter().enumerate() {
        if graph.schema.vectors.iter().any(|s| s.key == slot.key) {
            continue;
        }
        let n = lengths
            .get(&slot.dim)
            .copied()
            .or_else(|| graph.obs_vectors.get(i).map(Vec::len))
            .ok_or_else(|| {
                ModelError::invalid(format!(
                    "supply size for prediction dimension '{}'",
                    slot.dim
                ))
            })?;
        if n == 0 {
            return Err(ModelError::invalid(
                "prediction dimensions must be positive",
            ));
        }
        inputs
            .vectors
            .insert(slot.key.clone(), Arc::from(vec![1.; n]));
    }
    let binding = DataBinding::bind(&graph.schema, inputs, "prediction", true, true)
        .map_err(|e| ModelError::invalid(e.to_string()))?;
    let bound = graph.with_binding(&binding);
    bound
        .validate_shapes()
        .map_err(|e| ModelError::invalid(e.to_string()))?;
    Ok(bound)
}

impl ModelArtifact {
    /// Fit payloads already declare the complete parameter axis. Reject impossible
    /// structural sizes before allocating template matrices or parameter vectors.
    pub fn validate_parameter_limit(&self, count: usize) -> ModelResult<()> {
        let mut minimum = 0usize;
        for prior in &self.definition.priors {
            let n = match prior {
                PriorSpec::VectorNormal { n, .. } => *n,
                _ => 1,
            };
            minimum = minimum
                .checked_add(n)
                .ok_or_else(|| ModelError::invalid("artifact parameter count overflow"))?;
            if minimum > count {
                return Err(ModelError::invalid(
                    "artifact model parameter dimensions exceed posterior axis",
                ));
            }
        }
        for slot in &self.schema.matrices {
            if let crate::data::SlotKind::Matrix { n_cols } = slot.kind {
                if n_cols > count {
                    return Err(ModelError::invalid(
                        "artifact matrix width exceeds posterior parameter axis",
                    ));
                }
            }
        }
        Ok(())
    }
}
