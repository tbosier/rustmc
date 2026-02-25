use crate::graph::{Graph, NodeId};

/// Trait for probability distributions that can contribute log-probability
/// terms to a computational graph.
///
/// Future distributions (HalfNormal, Uniform, StudentT, …) implement this
/// trait to integrate with the graph-based autodiff engine.
pub trait Distribution {
    fn logp(&self, graph: &mut Graph) -> NodeId;
}

/// Univariate Normal distribution.
pub struct Normal {
    pub name: String,
    pub mu: NodeId,
    pub sigma: NodeId,
    pub observed: Option<Observed>,
}

/// Observed data attached to a distribution.
pub struct Observed {
    pub mu_vec: NodeId,
    pub sigma: NodeId,
    pub obs_data_idx: usize,
}

impl Normal {
    /// Create a Normal prior on a free parameter.
    pub fn prior(graph: &mut Graph, name: &str, mu: f64, sigma: f64) -> NodeId {
        let param = graph.add_param(name);
        let mu_node = graph.add_constant(mu);
        let sigma_node = graph.add_constant(sigma);
        graph.normal_logp(param, mu_node, sigma_node);
        param
    }

    /// Create a Normal likelihood for observed data.
    pub fn observed(
        graph: &mut Graph,
        mu_vec: NodeId,
        sigma: f64,
        obs: Vec<f64>,
    ) -> NodeId {
        let sigma_node = graph.add_constant(sigma);
        let obs_idx = graph.add_obs_data(obs);
        graph.normal_obs_logp(mu_vec, sigma_node, obs_idx)
    }
}
