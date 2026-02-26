use crate::graph::{Graph, NodeId, ParamTransform};

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
        graph.half_normal_logp(x, sigma_node);
        // Jacobian correction: log|det J| = raw (since dx/draw = exp(raw) = x, log = raw)
        let jacobian = graph.add_node_as_logp(raw);
        let _ = jacobian;
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
        let raw = graph.add_param_with_transform(
            name,
            ParamTransform::BoundedSigmoid { lower, upper },
        );
        let sig = graph.sigmoid(raw);
        let range = upper - lower;
        let range_node = graph.add_constant(range);
        let lower_node = graph.add_constant(lower);

        // x = lower + range * sigmoid(raw)
        let scaled = graph.mul(range_node, sig);
        let x = graph.add(lower_node, scaled);

        let upper_node = graph.add_constant(upper);
        graph.uniform_logp(x, lower_node, upper_node);

        // Jacobian: log(range) + log(sigmoid) + log(1 - sigmoid)
        let log_range = graph.add_constant(range.ln());
        let log_sig = graph.log(sig);
        let one = graph.add_constant(1.0);
        let one_minus_sig = graph.sub(one, sig);
        let log_one_minus_sig = graph.log(one_minus_sig);
        let log_sum = graph.add(log_sig, log_one_minus_sig);
        let jac = graph.add(log_range, log_sum);
        graph.add_logp_term(jac);
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
        graph.gamma_logp(x, alpha_node, beta_node);
        graph.add_logp_term(raw);
        x
    }
}

// ── Beta (0 < x < 1, logit-transform) ──────────────────────────────

pub struct BetaDist;

impl BetaDist {
    /// Samples raw on (-∞, +∞), transforms via x = sigmoid(raw).
    /// Jacobian: log|dx/draw| = log(sigmoid(raw)) + log(1-sigmoid(raw))
    pub fn prior(graph: &mut Graph, name: &str, alpha: f64, beta: f64) -> NodeId {
        let raw = graph.add_param_with_transform(name, ParamTransform::Sigmoid);
        let x = graph.sigmoid(raw);
        let alpha_node = graph.add_constant(alpha);
        let beta_node = graph.add_constant(beta);
        graph.beta_logp(x, alpha_node, beta_node);

        // Jacobian: log(x) + log(1-x)
        let log_x = graph.log(x);
        let one = graph.add_constant(1.0);
        let one_minus_x = graph.sub(one, x);
        let log_one_minus_x = graph.log(one_minus_x);
        let jac = graph.add(log_x, log_one_minus_x);
        graph.add_logp_term(jac);
        x
    }
}
