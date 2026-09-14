use crate::graph::{Graph, NodeId, ParamTransform};

#[deprecated(
    note = "unused abstraction; use graph log-probability terms or the native LogDensity interface"
)]
pub trait Distribution {
    fn logp(&self, graph: &mut Graph) -> NodeId;
}

// ── Normal (unconstrained) ──────────────────────────────────────────

pub struct Normal;

impl Normal {
    pub fn prior(graph: &mut Graph, name: &str, mu: f64, sigma: f64) -> NodeId {
        let param = graph.add_param(name);
        let mu_node = graph.add_constant(mu);
        let sigma_node = graph.add_constant(sigma);
        graph.normal_logp(param, mu_node, sigma_node);
        param
    }

    /// Like `prior`, but accepts pre-built graph nodes for mu and sigma.
    /// Used for hierarchical priors where mu/sigma are themselves parameters.
    pub fn prior_with_nodes(
        graph: &mut Graph,
        name: &str,
        mu_node: NodeId,
        sigma_node: NodeId,
    ) -> NodeId {
        let param = graph.add_param(name);
        graph.normal_logp(param, mu_node, sigma_node);
        param
    }

    pub fn observed(graph: &mut Graph, mu_vec: NodeId, sigma: f64, obs: Vec<f64>) -> NodeId {
        let sigma_node = graph.add_constant(sigma);
        let obs_idx = graph.add_obs_data(obs);
        graph.normal_obs_logp(mu_vec, sigma_node, obs_idx)
    }
}

// ── HalfNormal (x > 0, log-transform) ──────────────────────────────

pub struct HalfNormal;

impl HalfNormal {
    /// Samples raw on (-∞, +∞), transforms via x = exp(raw).
    /// Jacobian: log|dx/draw| = raw = log(x).
    pub fn prior(graph: &mut Graph, name: &str, sigma: f64) -> NodeId {
        let raw = graph.add_param_with_transform(name, ParamTransform::Exp);
        let x = graph.exp(raw);
        let sigma_node = graph.add_constant(sigma);
        graph.log_half_normal_logp(raw, sigma_node);
        x
    }

    /// Like `prior`, but accepts a pre-built graph node for sigma.
    /// Used for hierarchical priors where sigma is itself a parameter.
    pub fn prior_with_node_sigma(graph: &mut Graph, name: &str, sigma_node: NodeId) -> NodeId {
        let raw = graph.add_param_with_transform(name, ParamTransform::Exp);
        let x = graph.exp(raw);
        graph.log_half_normal_logp(raw, sigma_node);
        x
    }
}

// ── StudentT (unconstrained) ────────────────────────────────────────

pub struct StudentT;

impl StudentT {
    pub fn prior(graph: &mut Graph, name: &str, nu: f64, mu: f64, sigma: f64) -> NodeId {
        let param = graph.add_param(name);
        let nu_node = graph.add_constant(nu);
        let mu_node = graph.add_constant(mu);
        let sigma_node = graph.add_constant(sigma);
        graph.student_t_logp(param, nu_node, mu_node, sigma_node);
        param
    }
}

// ── Uniform (lower < x < upper, logit-transform) ───────────────────

pub struct Uniform;

impl Uniform {
    /// Samples raw on (-∞, +∞), transforms via x = lower + (upper-lower) * sigmoid(raw).
    /// Jacobian: log|dx/draw| = log((upper-lower) * sigmoid(raw) * (1-sigmoid(raw)))
    pub fn prior(graph: &mut Graph, name: &str, lower: f64, upper: f64) -> NodeId {
        let param_start = graph.param_count;
        let raw =
            graph.add_param_with_transform(name, ParamTransform::BoundedSigmoid { lower, upper });
        let sig = graph.sigmoid(raw);
        let range_node = graph.add_constant(upper - lower);
        let lower_node = graph.add_constant(lower);
        let scaled = graph.mul(range_node, sig);
        let x = graph.add(lower_node, scaled);

        // Evaluate density and Jacobian together in raw space: the interval
        // width cancels, and rounded sigmoid endpoints must not truncate tails.
        graph.vector_uniform_logp(param_start, 1, lower, upper);
        x
    }
}

// ── Bernoulli (discrete — not differentiable, kept for completeness) ─

pub struct Bernoulli;

impl Bernoulli {
    pub fn prior(graph: &mut Graph, name: &str, p: f64) -> NodeId {
        let param = graph.add_param(name);
        let p_node = graph.add_constant(p);
        graph.bernoulli_logp(param, p_node);
        param
    }
}

// ── Poisson (x > 0, log-transform for rate parameter) ───────────────

pub struct Poisson;

impl Poisson {
    pub fn prior(graph: &mut Graph, name: &str, lam: f64) -> NodeId {
        let param = graph.add_param(name);
        let lam_node = graph.add_constant(lam);
        graph.poisson_logp(param, lam_node);
        param
    }
}

// ── Exponential (x > 0, log-transform) ─────────────────────────────

pub struct Exponential;

impl Exponential {
    pub fn prior(graph: &mut Graph, name: &str, rate: f64) -> NodeId {
        let raw = graph.add_param_with_transform(name, ParamTransform::Exp);
        let x = graph.exp(raw);
        let alpha_node = graph.add_constant(1.0);
        let rate_node = graph.add_constant(rate);
        graph.log_gamma_logp(raw, alpha_node, rate_node);
        x
    }

    pub fn prior_with_node_rate(graph: &mut Graph, name: &str, rate_node: NodeId) -> NodeId {
        let raw = graph.add_param_with_transform(name, ParamTransform::Exp);
        let x = graph.exp(raw);
        let alpha_node = graph.add_constant(1.0);
        graph.log_gamma_logp(raw, alpha_node, rate_node);
        x
    }
}

// ── LogNormal (x > 0, log-transform) ────────────────────────────────

pub struct LogNormal;

impl LogNormal {
    pub fn prior(graph: &mut Graph, name: &str, mu: f64, sigma: f64) -> NodeId {
        let raw = graph.add_param_with_transform(name, ParamTransform::Exp);
        let x = graph.exp(raw);
        let mu_node = graph.add_constant(mu);
        let sigma_node = graph.add_constant(sigma);
        graph.normal_logp(raw, mu_node, sigma_node);
        x
    }

    pub fn prior_with_nodes(
        graph: &mut Graph,
        name: &str,
        mu_node: NodeId,
        sigma_node: NodeId,
    ) -> NodeId {
        let raw = graph.add_param_with_transform(name, ParamTransform::Exp);
        let x = graph.exp(raw);
        graph.normal_logp(raw, mu_node, sigma_node);
        x
    }
}

// ── Gamma (x > 0, log-transform) ───────────────────────────────────

pub struct Gamma;

impl Gamma {
    /// Samples raw on (-∞, +∞), transforms via x = exp(raw).
    /// Jacobian: log|dx/draw| = raw.
    pub fn prior(graph: &mut Graph, name: &str, alpha: f64, beta: f64) -> NodeId {
        let raw = graph.add_param_with_transform(name, ParamTransform::Exp);
        let x = graph.exp(raw);
        let alpha_node = graph.add_constant(alpha);
        let beta_node = graph.add_constant(beta);
        graph.log_gamma_logp(raw, alpha_node, beta_node);
        x
    }
}

// ── Beta (0 < x < 1, logit-transform) ──────────────────────────────

pub struct BetaDist;

impl BetaDist {
    /// Samples raw on (-∞, +∞), transforms via x = sigmoid(raw).
    /// Jacobian: log|dx/draw| = log(sigmoid(raw)) + log(1-sigmoid(raw))
    pub fn prior(graph: &mut Graph, name: &str, alpha: f64, beta: f64) -> NodeId {
        let param_start = graph.param_count;
        let raw = graph.add_param_with_transform(name, ParamTransform::Sigmoid);
        let x = graph.sigmoid(raw);
        // Sharing the raw-space kernel with vector priors avoids both rounded
        // endpoints and cancellation between the density and its Jacobian.
        graph.vector_beta_logp(param_start, 1, alpha, beta);
        x
    }
}
