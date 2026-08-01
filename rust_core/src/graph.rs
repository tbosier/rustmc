use std::collections::HashMap;
use std::fmt::{Display, Formatter};

use crate::data::DataBinding;
use crate::data::{DataSchema, DataSlot, SlotKind};

/// Unique identifier for a node in the computation graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

/// Contiguous parameter span in declaration order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParamSpan {
    pub start: usize,
    pub len: usize,
}

/// Shape validation error for graph-level vector data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphShapeError {
    message: String,
}

impl GraphShapeError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl Display for GraphShapeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

/// A 2-D data matrix stored row-major (n_rows × n_cols).
#[derive(Debug, Clone)]
pub struct MatrixData {
    pub data: Vec<f64>,
    pub n_rows: usize,
    pub n_cols: usize,
}

/// Observation families supported by the generic observation op.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObsFamily {
    Normal,
    BernoulliLogit,
    PoissonLog,
    ExponentialLog,
    LogNormal,
    NegativeBinomialLog,
}

/// Metadata for an observation term in the graph.
#[derive(Debug, Clone)]
pub struct ObservationHead {
    pub name: String,
    pub family: ObsFamily,
    pub linpred: NodeId,
    pub aux: Option<NodeId>,
    pub obs_data_idx: usize,
    pub n_obs: usize,
}

/// Operations supported in the computation graph.
#[derive(Debug, Clone)]
pub enum Op {
    /// A free parameter to be sampled (index into the parameter vector).
    Param(usize),
    /// A constant scalar value baked into the graph.
    Constant(f64),
    /// Observed data vector (index into the data table).
    Data(usize),
    Add(NodeId, NodeId),
    Mul(NodeId, NodeId),
    Sub(NodeId, NodeId),
    Div(NodeId, NodeId),
    Neg(NodeId),
    Exp(NodeId),
    Log(NodeId),
    /// 1 / (1 + exp(-x))
    Sigmoid(NodeId),
    Square(NodeId),
    /// Element-wise multiply: scalar * data vector.
    ScalarMulData(NodeId, NodeId),
    /// Element-wise addition of two vectors.
    VectorAdd(NodeId, NodeId),
    /// Broadcast scalar + vector → vector.
    ScalarBroadcastAdd(NodeId, NodeId),
    /// Broadcast scalar → constant vector (every element equals the scalar).
    /// Used when a scalar parameter is the mean for all N observations.
    ScalarBroadcast(NodeId),
    /// Log-probability of a Normal distribution: logp(x | mu, sigma).
    NormalLogP {
        x: NodeId,
        mu: NodeId,
        sigma: NodeId,
    },
    /// Generic observation log-probability term for supported GLM families.
    ObsLogP {
        family: ObsFamily,
        linpred_vec: NodeId,
        aux: Option<NodeId>,
        obs_data_idx: usize,
    },
    /// logp(x | sigma) for x >= 0; HalfNormal
    HalfNormalLogP {
        x: NodeId,
        sigma: NodeId,
    },
    /// logp(x | nu, mu, sigma); StudentT
    StudentTLogP {
        x: NodeId,
        nu: NodeId,
        mu: NodeId,
        sigma: NodeId,
    },
    /// logp(x | lower, upper); Uniform
    UniformLogP {
        x: NodeId,
        lower: NodeId,
        upper: NodeId,
    },
    /// logp(x | p); Bernoulli (x in {0, 1})
    BernoulliLogP {
        x: NodeId,
        p: NodeId,
    },
    /// logp(x | lam); Poisson
    PoissonLogP {
        x: NodeId,
        lam: NodeId,
    },
    /// logp(x | alpha, beta); Gamma
    GammaLogP {
        x: NodeId,
        alpha: NodeId,
        beta: NodeId,
    },
    /// logp(x | alpha, beta); Beta
    BetaLogP {
        x: NodeId,
        alpha: NodeId,
        beta: NodeId,
    },
    /// Fused linear combination: mu[i] = intercept + Σ_k params[k] * data[k][i]
    ///
    /// Replaces a chain of ScalarMulData + VectorAdd + ScalarBroadcastAdd with
    /// a single pass over the data, dramatically improving cache utilization.
    FusedLinearMu {
        param_nodes: Vec<NodeId>,
        data_indices: Vec<usize>,
        intercept: Option<NodeId>,
    },
    /// faer-backed matrix-vector multiply: mu = X @ params[param_start..param_start+n_params]
    /// X is stored row-major in graph.data_matrices[matrix_idx].
    MatVecMul {
        matrix_idx: usize,
        param_start: usize,
        n_params: usize,
        intercept: Option<NodeId>,
    },
    /// Vectorized Normal prior: Σ_k Normal.logp(params[param_start+k], mu, sigma)
    VectorNormalLogP {
        param_start: usize,
        n_params: usize,
        mu: f64,
        sigma: f64,
    },
    /// Vectorized HalfNormal prior on exp-transformed params.
    /// logp(exp(raw)) + raw (Jacobian) summed over n_params.
    VectorHalfNormalLogP {
        param_start: usize,
        n_params: usize,
        sigma: f64,
    },
    /// Vectorized StudentT prior (identity transform).
    VectorStudentTLogP {
        param_start: usize,
        n_params: usize,
        nu: f64,
        mu: f64,
        sigma: f64,
    },
    /// Vectorized Gamma prior on exp-transformed params.
    VectorGammaLogP {
        param_start: usize,
        n_params: usize,
        alpha: f64,
        beta: f64,
    },
    /// Vectorized Beta prior on sigmoid-transformed params.
    VectorBetaLogP {
        param_start: usize,
        n_params: usize,
        alpha: f64,
        beta: f64,
    },
    /// Vectorized Uniform prior on bounded-sigmoid-transformed params.
    VectorUniformLogP {
        param_start: usize,
        n_params: usize,
        lower: f64,
        upper: f64,
    },
}

/// A single node in the computation graph.
#[derive(Debug, Clone)]
pub struct Node {
    pub id: NodeId,
    pub op: Op,
    pub name: Option<String>,
}

/// Transform applied to a parameter so NUTS samples on unconstrained space.
#[derive(Debug, Clone)]
pub enum ParamTransform {
    /// No transform — parameter is unconstrained.
    Identity,
    /// x = exp(raw). For parameters that must be > 0.
    Exp,
    /// x = sigmoid(raw). For parameters in (0, 1).
    Sigmoid,
    /// x = lower + (upper - lower) * sigmoid(raw). For parameters in (lower, upper).
    BoundedSigmoid { lower: f64, upper: f64 },
}

impl ParamTransform {
    pub fn apply(&self, raw: f64) -> f64 {
        match self {
            ParamTransform::Identity => raw,
            ParamTransform::Exp => raw.exp(),
            ParamTransform::Sigmoid => 1.0 / (1.0 + (-raw).exp()),
            ParamTransform::BoundedSigmoid { lower, upper } => {
                let s = 1.0 / (1.0 + (-raw).exp());
                lower + (upper - lower) * s
            }
        }
    }

    /// Derivative of the constrained value with respect to the raw value.
    #[inline]
    pub fn derivative(&self, raw: f64) -> f64 {
        match self {
            ParamTransform::Identity => 1.0,
            ParamTransform::Exp => raw.exp(),
            ParamTransform::Sigmoid => {
                let s = 1.0 / (1.0 + (-raw).exp());
                s * (1.0 - s)
            }
            ParamTransform::BoundedSigmoid { lower, upper } => {
                let s = 1.0 / (1.0 + (-raw).exp());
                (upper - lower) * s * (1.0 - s)
            }
        }
    }
}

/// The computational graph representing a probabilistic model.
///
/// Stores nodes in topological order (each node only references earlier nodes).
/// Data vectors and observed values are stored separately from the graph
/// structure so the graph itself stays lightweight and shareable across threads.
#[derive(Debug, Clone)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub param_count: usize,
    pub data_vectors: Vec<Vec<f64>>,
    pub obs_vectors: Vec<Vec<f64>>,
    pub data_matrices: Vec<MatrixData>,
    /// Structural, user-facing contract for re-bindable dataset payloads.
    pub schema: DataSchema,
    pub param_names: Vec<String>,
    pub param_transforms: Vec<ParamTransform>,
    pub param_spans: Vec<ParamSpan>,
    pub logp_terms: Vec<NodeId>,
    name_to_node: HashMap<String, NodeId>,
}

impl Graph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            param_count: 0,
            data_vectors: Vec::new(),
            obs_vectors: Vec::new(),
            data_matrices: Vec::new(),
            schema: DataSchema::default(),
            param_names: Vec::new(),
            param_transforms: Vec::new(),
            param_spans: Vec::new(),
            logp_terms: Vec::new(),
            name_to_node: HashMap::new(),
        }
    }

    fn add_node(&mut self, op: Op, name: Option<String>) -> NodeId {
        let id = NodeId(self.nodes.len());
        if let Some(ref n) = name {
            self.name_to_node.insert(n.clone(), id);
        }
        self.nodes.push(Node { id, op, name });
        id
    }

    pub fn add_param(&mut self, name: &str) -> NodeId {
        self.add_param_with_transform(name, ParamTransform::Identity)
    }

    pub fn add_param_with_transform(&mut self, name: &str, transform: ParamTransform) -> NodeId {
        let idx = self.param_count;
        self.param_count += 1;
        self.param_names.push(name.to_string());
        self.param_transforms.push(transform);
        self.param_spans.push(ParamSpan { start: idx, len: 1 });
        self.add_node(Op::Param(idx), Some(name.to_string()))
    }

    pub fn add_constant(&mut self, value: f64) -> NodeId {
        self.add_node(Op::Constant(value), None)
    }

    pub fn add_data(&mut self, name: &str, values: Vec<f64>) -> NodeId {
        let idx = self.data_vectors.len();
        self.data_vectors.push(values);
        self.schema.vectors.push(DataSlot {
            key: name.to_string(),
            kind: SlotKind::Vector,
            dim: "obs".to_string(),
        });
        self.add_node(Op::Data(idx), Some(name.to_string()))
    }

    pub fn add_obs_data(&mut self, values: Vec<f64>) -> usize {
        let idx = self.obs_vectors.len();
        self.obs_vectors.push(values);
        idx
    }

    /// Store a named response vector and declare its structural schema slot.
    pub fn add_named_obs_data(&mut self, key: &str, likelihood: &str, values: Vec<f64>) -> usize {
        let idx = self.obs_vectors.len();
        self.obs_vectors.push(values);
        self.schema.observations.push(DataSlot {
            key: key.to_string(),
            kind: SlotKind::Observation {
                likelihood: likelihood.to_string(),
            },
            dim: "obs".to_string(),
        });
        idx
    }

    pub fn add(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.add_node(Op::Add(a, b), None)
    }

    pub fn mul(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.add_node(Op::Mul(a, b), None)
    }

    pub fn sub(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.add_node(Op::Sub(a, b), None)
    }

    pub fn div(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.add_node(Op::Div(a, b), None)
    }

    pub fn neg(&mut self, a: NodeId) -> NodeId {
        self.add_node(Op::Neg(a), None)
    }

    pub fn exp(&mut self, a: NodeId) -> NodeId {
        self.add_node(Op::Exp(a), None)
    }

    pub fn log(&mut self, a: NodeId) -> NodeId {
        self.add_node(Op::Log(a), None)
    }

    pub fn sigmoid(&mut self, a: NodeId) -> NodeId {
        self.add_node(Op::Sigmoid(a), None)
    }

    pub fn square(&mut self, a: NodeId) -> NodeId {
        self.add_node(Op::Square(a), None)
    }

    pub fn scalar_mul_data(&mut self, scalar: NodeId, data: NodeId) -> NodeId {
        self.add_node(Op::ScalarMulData(scalar, data), None)
    }

    pub fn vector_add(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.add_node(Op::VectorAdd(a, b), None)
    }

    pub fn scalar_broadcast_add(&mut self, scalar: NodeId, vec: NodeId) -> NodeId {
        self.add_node(Op::ScalarBroadcastAdd(scalar, vec), None)
    }

    /// Broadcast a scalar node into a vector node (each element = scalar).
    /// Used so a scalar parameter can serve as the linear predictor in ObsLogP.
    pub fn scalar_broadcast(&mut self, scalar: NodeId) -> NodeId {
        self.add_node(Op::ScalarBroadcast(scalar), None)
    }

    pub fn normal_logp(&mut self, x: NodeId, mu: NodeId, sigma: NodeId) -> NodeId {
        let node = self.add_node(Op::NormalLogP { x, mu, sigma }, None);
        self.logp_terms.push(node);
        node
    }

    /// Backward-compatible alias for the Normal observation op.
    pub fn normal_obs_logp(
        &mut self,
        linpred_vec: NodeId,
        sigma: NodeId,
        obs_data_idx: usize,
    ) -> NodeId {
        self.obs_logp_normal(linpred_vec, sigma, obs_data_idx)
    }

    pub fn obs_logp_normal(
        &mut self,
        linpred_vec: NodeId,
        sigma: NodeId,
        obs_data_idx: usize,
    ) -> NodeId {
        let node = self.add_node(
            Op::ObsLogP {
                family: ObsFamily::Normal,
                linpred_vec,
                aux: Some(sigma),
                obs_data_idx,
            },
            None,
        );
        self.logp_terms.push(node);
        node
    }

    pub fn obs_logp_bernoulli_logit(&mut self, linpred_vec: NodeId, obs_data_idx: usize) -> NodeId {
        let node = self.add_node(
            Op::ObsLogP {
                family: ObsFamily::BernoulliLogit,
                linpred_vec,
                aux: None,
                obs_data_idx,
            },
            None,
        );
        self.logp_terms.push(node);
        node
    }

    pub fn obs_logp_poisson_log(&mut self, linpred_vec: NodeId, obs_data_idx: usize) -> NodeId {
        let node = self.add_node(
            Op::ObsLogP {
                family: ObsFamily::PoissonLog,
                linpred_vec,
                aux: None,
                obs_data_idx,
            },
            None,
        );
        self.logp_terms.push(node);
        node
    }

    pub fn obs_logp_exponential_log(&mut self, linpred_vec: NodeId, obs_data_idx: usize) -> NodeId {
        let node = self.add_node(
            Op::ObsLogP {
                family: ObsFamily::ExponentialLog,
                linpred_vec,
                aux: None,
                obs_data_idx,
            },
            None,
        );
        self.logp_terms.push(node);
        node
    }

    pub fn obs_logp_lognormal(
        &mut self,
        linpred_vec: NodeId,
        sigma: NodeId,
        obs_data_idx: usize,
    ) -> NodeId {
        let node = self.add_node(
            Op::ObsLogP {
                family: ObsFamily::LogNormal,
                linpred_vec,
                aux: Some(sigma),
                obs_data_idx,
            },
            None,
        );
        self.logp_terms.push(node);
        node
    }

    pub fn obs_logp_negative_binomial_log(
        &mut self,
        linpred_vec: NodeId,
        alpha: NodeId,
        obs_data_idx: usize,
    ) -> NodeId {
        let node = self.add_node(
            Op::ObsLogP {
                family: ObsFamily::NegativeBinomialLog,
                linpred_vec,
                aux: Some(alpha),
                obs_data_idx,
            },
            None,
        );
        self.logp_terms.push(node);
        node
    }

    pub fn half_normal_logp(&mut self, x: NodeId, sigma: NodeId) -> NodeId {
        let node = self.add_node(Op::HalfNormalLogP { x, sigma }, None);
        self.logp_terms.push(node);
        node
    }

    pub fn student_t_logp(&mut self, x: NodeId, nu: NodeId, mu: NodeId, sigma: NodeId) -> NodeId {
        let node = self.add_node(Op::StudentTLogP { x, nu, mu, sigma }, None);
        self.logp_terms.push(node);
        node
    }

    pub fn uniform_logp(&mut self, x: NodeId, lower: NodeId, upper: NodeId) -> NodeId {
        let node = self.add_node(Op::UniformLogP { x, lower, upper }, None);
        self.logp_terms.push(node);
        node
    }

    pub fn bernoulli_logp(&mut self, x: NodeId, p: NodeId) -> NodeId {
        let node = self.add_node(Op::BernoulliLogP { x, p }, None);
        self.logp_terms.push(node);
        node
    }

    pub fn poisson_logp(&mut self, x: NodeId, lam: NodeId) -> NodeId {
        let node = self.add_node(Op::PoissonLogP { x, lam }, None);
        self.logp_terms.push(node);
        node
    }

    pub fn gamma_logp(&mut self, x: NodeId, alpha: NodeId, beta: NodeId) -> NodeId {
        let node = self.add_node(Op::GammaLogP { x, alpha, beta }, None);
        self.logp_terms.push(node);
        node
    }

    pub fn beta_logp(&mut self, x: NodeId, alpha: NodeId, beta: NodeId) -> NodeId {
        let node = self.add_node(Op::BetaLogP { x, alpha, beta }, None);
        self.logp_terms.push(node);
        node
    }

    /// Mark an existing node as a log-probability term (adds its value to total logp).
    pub fn add_logp_term(&mut self, node: NodeId) {
        self.logp_terms.push(node);
    }

    /// Convenience: add a node's value directly as a logp term (used for Jacobians).
    pub fn add_node_as_logp(&mut self, node: NodeId) -> NodeId {
        self.logp_terms.push(node);
        node
    }

    /// Return observation metadata for every supported observation term.
    pub fn observation_heads(&self) -> Vec<ObservationHead> {
        self.nodes
            .iter()
            .filter_map(|n| {
                if let Op::ObsLogP {
                    family,
                    linpred_vec,
                    aux,
                    obs_data_idx,
                } = &n.op
                {
                    Some(ObservationHead {
                        name: n.name.clone().unwrap_or_default(),
                        family: *family,
                        linpred: *linpred_vec,
                        aux: *aux,
                        obs_data_idx: *obs_data_idx,
                        n_obs: self.obs_vectors[*obs_data_idx].len(),
                    })
                } else {
                    None
                }
            })
            .collect()
    }

    /// Backward-compatible helper for the current Normal-only API surface.
    pub fn normal_obs_predictors(&self) -> Vec<(NodeId, NodeId, usize)> {
        self.observation_heads()
            .into_iter()
            .filter_map(|head| match head.family {
                ObsFamily::Normal => Some((head.linpred, head.aux.unwrap(), head.n_obs)),
                ObsFamily::BernoulliLogit
                | ObsFamily::PoissonLog
                | ObsFamily::ExponentialLog
                | ObsFamily::LogNormal
                | ObsFamily::NegativeBinomialLog => None,
            })
            .collect()
    }

    /// Store a data vector without creating a graph node (used by FusedLinearMu).
    pub fn store_data_vec(&mut self, values: Vec<f64>) -> usize {
        let idx = self.data_vectors.len();
        self.data_vectors.push(values);
        idx
    }

    /// Store a named predictor without creating an explicit data node.
    pub fn store_named_data_vec(&mut self, key: &str, values: Vec<f64>) -> usize {
        let idx = self.data_vectors.len();
        self.data_vectors.push(values);
        self.schema.vectors.push(DataSlot {
            key: key.to_string(),
            kind: SlotKind::Vector,
            dim: "obs".to_string(),
        });
        idx
    }

    pub fn fused_linear_mu(
        &mut self,
        param_nodes: Vec<NodeId>,
        data_indices: Vec<usize>,
        intercept: Option<NodeId>,
    ) -> NodeId {
        self.add_node(
            Op::FusedLinearMu {
                param_nodes,
                data_indices,
                intercept,
            },
            None,
        )
    }

    pub fn node_by_name(&self, name: &str) -> Option<NodeId> {
        self.name_to_node.get(name).copied()
    }

    /// Allocate `n` contiguous parameters with no individual `Param` nodes.
    /// Returns the `param_start` index into the parameter vector.
    pub fn add_vector_params(&mut self, base_name: &str, n: usize) -> usize {
        self.add_vector_params_with_transform(base_name, n, ParamTransform::Identity)
    }

    /// Allocate `n` contiguous parameters with a specific transform.
    /// Returns the `param_start` index into the parameter vector.
    pub fn add_vector_params_with_transform(
        &mut self,
        base_name: &str,
        n: usize,
        transform: ParamTransform,
    ) -> usize {
        let param_start = self.param_count;
        self.param_count += n;
        self.param_spans.push(ParamSpan {
            start: param_start,
            len: n,
        });
        for k in 0..n {
            self.param_names.push(format!("{}[{}]", base_name, k));
            self.param_transforms.push(transform.clone());
        }
        param_start
    }

    /// Store a row-major matrix and return its index in `data_matrices`.
    pub fn store_matrix(&mut self, data: Vec<f64>, n_rows: usize, n_cols: usize) -> usize {
        let idx = self.data_matrices.len();
        self.data_matrices.push(MatrixData {
            data,
            n_rows,
            n_cols,
        });
        idx
    }

    pub fn store_named_matrix(
        &mut self,
        key: &str,
        data: Vec<f64>,
        n_rows: usize,
        n_cols: usize,
    ) -> usize {
        let idx = self.data_matrices.len();
        self.data_matrices.push(MatrixData {
            data,
            n_rows,
            n_cols,
        });
        self.schema.matrices.push(DataSlot {
            key: key.to_string(),
            kind: SlotKind::Matrix { n_cols },
            dim: "obs".to_string(),
        });
        idx
    }

    /// Clone only the immutable structure and schema, dropping dataset payloads.
    pub fn structure_only(&self) -> Self {
        let mut graph = self.clone();
        graph.data_vectors.clear();
        graph.obs_vectors.clear();
        graph.data_matrices.clear();
        graph
    }

    /// Compatibility view for APIs that still consume a data-owning graph.
    pub fn with_binding(&self, binding: &DataBinding) -> Self {
        let mut graph = self.structure_only();
        graph.data_vectors = binding.vectors.iter().map(|v| v.to_vec()).collect();
        graph.obs_vectors = binding.observations.iter().map(|v| v.to_vec()).collect();
        graph.data_matrices = binding
            .matrices
            .iter()
            .map(|m| MatrixData {
                data: m.data.to_vec(),
                n_rows: m.n_rows,
                n_cols: m.n_cols,
            })
            .collect();
        graph
    }

    pub fn mat_vec_mul(
        &mut self,
        matrix_idx: usize,
        param_start: usize,
        n_params: usize,
        intercept: Option<NodeId>,
    ) -> NodeId {
        self.add_node(
            Op::MatVecMul {
                matrix_idx,
                param_start,
                n_params,
                intercept,
            },
            None,
        )
    }

    pub fn vector_normal_logp(
        &mut self,
        param_start: usize,
        n_params: usize,
        mu: f64,
        sigma: f64,
    ) -> NodeId {
        let node = self.add_node(
            Op::VectorNormalLogP {
                param_start,
                n_params,
                mu,
                sigma,
            },
            None,
        );
        self.logp_terms.push(node);
        node
    }

    pub fn vector_half_normal_logp(
        &mut self,
        param_start: usize,
        n_params: usize,
        sigma: f64,
    ) -> NodeId {
        let node = self.add_node(
            Op::VectorHalfNormalLogP {
                param_start,
                n_params,
                sigma,
            },
            None,
        );
        self.logp_terms.push(node);
        node
    }

    pub fn vector_student_t_logp(
        &mut self,
        param_start: usize,
        n_params: usize,
        nu: f64,
        mu: f64,
        sigma: f64,
    ) -> NodeId {
        let node = self.add_node(
            Op::VectorStudentTLogP {
                param_start,
                n_params,
                nu,
                mu,
                sigma,
            },
            None,
        );
        self.logp_terms.push(node);
        node
    }

    pub fn vector_gamma_logp(
        &mut self,
        param_start: usize,
        n_params: usize,
        alpha: f64,
        beta: f64,
    ) -> NodeId {
        let node = self.add_node(
            Op::VectorGammaLogP {
                param_start,
                n_params,
                alpha,
                beta,
            },
            None,
        );
        self.logp_terms.push(node);
        node
    }

    pub fn vector_beta_logp(
        &mut self,
        param_start: usize,
        n_params: usize,
        alpha: f64,
        beta: f64,
    ) -> NodeId {
        let node = self.add_node(
            Op::VectorBetaLogP {
                param_start,
                n_params,
                alpha,
                beta,
            },
            None,
        );
        self.logp_terms.push(node);
        node
    }

    pub fn vector_uniform_logp(
        &mut self,
        param_start: usize,
        n_params: usize,
        lower: f64,
        upper: f64,
    ) -> NodeId {
        let node = self.add_node(
            Op::VectorUniformLogP {
                param_start,
                n_params,
                lower,
                upper,
            },
            None,
        );
        self.logp_terms.push(node);
        node
    }

    /// Validate that all vector-like graph payloads agree on a single length.
    ///
    /// This is the graph-level safety gate for the evaluator. It ensures that
    /// every data vector, observation vector, and matrix row count is
    /// consistent before any sampling or gradient evaluation occurs.
    pub fn validate_shapes(&self) -> Result<usize, GraphShapeError> {
        let mut expected_len: Option<usize> = None;

        let mut set_expected =
            |actual: usize, kind: &str, index: usize| -> Result<(), GraphShapeError> {
                match expected_len {
                    None => {
                        expected_len = Some(actual);
                        Ok(())
                    }
                    Some(expected) if expected == actual => Ok(()),
                    Some(expected) => Err(GraphShapeError::new(format!(
                        "{} {} has length {}, expected {}",
                        kind, index, actual, expected
                    ))),
                }
            };

        for (idx, data) in self.data_vectors.iter().enumerate() {
            set_expected(data.len(), "data vector", idx)?;
        }

        for (idx, obs) in self.obs_vectors.iter().enumerate() {
            set_expected(obs.len(), "observation vector", idx)?;
        }

        for (idx, matrix) in self.data_matrices.iter().enumerate() {
            let payload_len = matrix.data.len();
            let expected_payload_len = matrix.n_rows * matrix.n_cols;
            if payload_len != expected_payload_len {
                return Err(GraphShapeError::new(format!(
                    "matrix {} has shape {}x{} but {} values were provided",
                    idx, matrix.n_rows, matrix.n_cols, payload_len
                )));
            }
            set_expected(matrix.n_rows, "matrix row count", idx)?;
        }

        Ok(expected_len.unwrap_or(0))
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}
