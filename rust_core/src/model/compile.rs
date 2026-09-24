//! Compiling a definition into a `Graph`, and mapping raw draws back to the
//! parameters it reports.
use crate::distributions::{
    Bernoulli, BetaDist, Exponential, Gamma, HalfNormal, LogNormal, Normal, Poisson, StudentT,
    Uniform,
};
use crate::graph::{Graph, NodeId, ParamTransform};
use crate::sampler::SampleResult;
use std::collections::HashMap;

use super::*;

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
                // Noncentering cancels the scale in the density and Jacobian,
                // but the conditional Normal still requires a positive scale.
                // Keep that support independently of mu + sigma * raw so large
                // means or tiny scales cannot erase the raw Normal density.
                graph.positive_support(sigma_node);
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
        if !value.is_finite() {
            let name = match spec {
                DisplayParamSpec::Raw { name, .. }
                | DisplayParamSpec::DerivedNonCenteredNormal { name, .. } => name,
            };
            return Err(ModelError::invalid(format!(
                "sampled parameter '{name}' is nonfinite after display transformation"
            )));
        }
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
        unconstrained_samples: raw_result.unconstrained_samples.clone(),
        accept_rates: raw_result.accept_rates.clone(),
        step_sizes: raw_result.step_sizes.clone(),
        divergences: raw_result.divergences.clone(),
        transitions: raw_result.transitions.clone(),
        param_names,
    })
}

/// A fused intercept retains its expression type independently of parameter names.
pub(super) enum LinearIntercept {
    Constant(f64),
    Parameter(String),
}

/// Try to decompose an expression into data-weighted terms and a typed intercept.
pub(super) fn try_extract_linear(expr: &MuExpr) -> Option<(LinearTerms, Option<LinearIntercept>)> {
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
pub(super) fn lookup_param_value_node(
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

pub(super) fn validate_definition(spec: &ModelSpec) -> ModelResult<()> {
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
                validate_positive_finite("uniform width", upper - lower)?;
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
