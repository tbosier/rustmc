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
    pub dim: String,
    pub family: ObsFamily,
    pub linpred: NodeId,
    pub aux: Option<NodeId>,
    pub obs_data_idx: usize,
    pub n_obs: usize,
}

pub use crate::numerics::{
    bounded_sigmoid, bounded_sigmoid_adjoint, bounded_sigmoid_derivative, stable_sigmoid,
    stable_sigmoid_derivative,
};
use crate::numerics::{
    div_denominator_adjoint, div_denominator_derivative, pow_adjoints, tanh_slope,
};

/// Arithmetic with scalar broadcasting and elementwise vector semantics.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub enum ElementwiseOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Neg,
    Exp,
    Log,
    Sigmoid,
    Sqrt,
    Tanh,
    Softplus,
    Sin,
    Cos,
}
impl ElementwiseOp {
    pub fn value(self, a: f64, b: f64) -> f64 {
        match self {
            Self::Add => a + b,
            Self::Sub => a - b,
            Self::Mul => a * b,
            Self::Div => a / b,
            Self::Pow => a.powf(b),
            Self::Neg => -a,
            Self::Exp => a.exp(),
            Self::Log => a.ln(),
            Self::Sigmoid => stable_sigmoid(a),
            Self::Sqrt => a.sqrt(),
            Self::Tanh => a.tanh(),
            Self::Softplus => a.max(0.0) + (-a.abs()).exp().ln_1p(),
            Self::Sin => a.sin(),
            Self::Cos => a.cos(),
        }
    }
    pub fn derivatives(self, a: f64, b: f64) -> (f64, f64) {
        match self {
            Self::Add => (1.0, 1.0),
            Self::Sub => (1.0, -1.0),
            Self::Mul => (b, a),
            Self::Div => (1.0 / b, div_denominator_derivative(a, b)),
            Self::Pow => {
                // x^0 is constant even at x=0. For positive exponents, 0^b
                // is also constant in b; forming 0*log(0) gives a false NaN.
                let da = if b == 0.0 { 0.0 } else { b * a.powf(b - 1.0) };
                let db = if a == 0.0 && b > 0.0 {
                    0.0
                } else {
                    a.powf(b) * a.ln()
                };
                (da, db)
            }
            Self::Neg => (-1.0, 0.0),
            Self::Exp => (a.exp(), 0.0),
            Self::Log => (1.0 / a, 0.0),
            Self::Sigmoid => (stable_sigmoid_derivative(a), 0.0),
            Self::Sqrt => (0.5 / a.sqrt(), 0.0),
            Self::Tanh => (tanh_slope(a), 0.0),
            Self::Softplus => (stable_sigmoid(a), 0.0),
            Self::Sin => (a.cos(), 0.0),
            Self::Cos => (-a.sin(), 0.0),
        }
    }

    /// `upstream * derivatives(a, b)`, associated so that the product stays
    /// inside the exponent range wherever the exact product is representable.
    ///
    /// This is what reverse mode calls. [`Self::derivatives`] remains the
    /// local-derivative API, unchanged and still correct on its own terms; it
    /// simply cannot express a composition, because a derivative of `-1e400`
    /// has no representation to hand back however the caller intends to scale
    /// it. See the helpers above for what each operator does about that.
    ///
    /// "Inside the exponent range wherever the exact product is representable"
    /// is the aim, not a proof: `Pow` gives it up for a base that is not
    /// strictly positive, and `Sqrt`, `Exp`, `Sigmoid`, `Softplus` and `Tanh`
    /// are argued below rather than rescued.
    ///
    /// The other eleven operators defer to `derivatives`, and the reason is the
    /// same in each case: their local derivative can only leave the exponent
    /// range where the node's own value already has, so there is no
    /// representable composed gradient left to lose.
    ///
    /// * `Add`, `Sub`, `Neg`, `Sin`, `Cos` — local derivatives bounded by 1.
    /// * `Mul` — the local derivative is one of its own operands, so
    ///   `upstream * b` is one correctly rounded product either way.
    /// * `Sqrt` — `0.5 / sqrt(a)` spans only `[3.7e-155, 2.2e161]` over every
    ///   positive `a`, which cannot leave the range at all. At `a == 0` it is
    ///   infinite, which is the true derivative and not a lost composition.
    /// * `Exp` — the derivative *is* the value, so they overflow together.
    /// * `Sigmoid`, `Softplus`, `Tanh` — the slope decays into the subnormals
    ///   past `|a| = 708` and reaches zero at 745, which is also where the
    ///   value itself is pinned at an endpoint. Between 708 and 745 the slope
    ///   is a subnormal and a very large adjoint would see it with fewer than
    ///   53 bits; that is a precision boundary, not a lost value, and it is not
    ///   rescued here. `Tanh`'s separate cancellation, which *did* lose a
    ///   representable derivative, is fixed in [`tanh_slope`] instead.
    pub fn adjoints(self, upstream: f64, a: f64, b: f64) -> (f64, f64) {
        match self {
            // `upstream / b` is `upstream * (1 / b)` with one rounding instead
            // of two, and it overflows only when the composition does. `1 / b`
            // alone becomes infinite below b = 5.562684646268003e-309 --
            // partway into the subnormals, not at the normal boundary --
            // whatever `a / b` and the adjoint are.
            Self::Div => (upstream / b, div_denominator_adjoint(upstream, a, b)),
            Self::Log => (upstream / a, 0.0),
            Self::Pow => pow_adjoints(upstream, a, b),
            _ => {
                let (da, db) = self.derivatives(a, b);
                (upstream * da, upstream * db)
            }
        }
    }
}

/// Operations supported in the computation graph.
#[derive(Debug, Clone)]
pub enum Op {
    Elementwise {
        operator: ElementwiseOp,
        a: NodeId,
        b: Option<NodeId>,
    },
    Gather {
        param_start: usize,
        n_params: usize,
        indices: NodeId,
    },
    Sum(NodeId),
    BroadcastObservation {
        scalar: NodeId,
        obs_data_idx: usize,
    },
    /// A free parameter to be sampled (index into the parameter vector).
    Param(usize),
    /// A constant scalar value baked into the graph.
    Constant(f64),
    /// Observed data vector (index into the data table).
    Data(usize),
    Add(NodeId, NodeId),
    Mul(NodeId, NodeId),
    Exp(NodeId),
    /// 1 / (1 + exp(-x))
    Sigmoid(NodeId),
    /// `lower + (upper - lower) * sigmoid(raw)` as one node.
    ///
    /// Fused rather than assembled from `Sigmoid`, `Mul` and `Add`, for two
    /// reasons that a three-node chain cannot give:
    ///
    /// * The forward value is [`bounded_sigmoid`], the same call
    ///   [`ParamTransform::apply`] makes, so the point the density is
    ///   evaluated at is by construction the draw reported back to the caller.
    ///   Assembled from nodes it was a second formula, and the two disagreed —
    ///   by one ulp on `(0, 1)`, and by the whole value on `(-1e308, 1)`.
    /// * The span never appears as a separate multiplicative factor, so
    ///   reverse mode never has to materialise the sigmoid's adjoint with it
    ///   already applied. `Uniform(0, 1e308)` at raw -710 under a `sigma = 0.1`
    ///   likelihood needed `-44.76 * 1e308` in that chain and overflowed to
    ///   `-inf`; here the scaling happens inside
    ///   [`bounded_sigmoid_derivative`], where it cannot leave the exponent
    ///   range.
    BoundedSigmoid {
        raw: NodeId,
        lower: f64,
        upper: f64,
    },
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
    /// Combined density and exp-transform Jacobian; x is the raw log value.
    LogHalfNormalLogP {
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
    /// Domain constraint with zero log density for finite x > 0, -infinity otherwise.
    PositiveSupport {
        x: NodeId,
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
    /// Combined density and exp-transform Jacobian; x is the raw log value.
    LogGammaLogP {
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

impl Op {
    /// Everything this op reads: other nodes, and the free-parameter slots it
    /// indexes out of the parameter vector without going through a node.
    ///
    /// The match below is exhaustive and deliberately has no catch-all arm.
    /// Callers use it to decide whether a model is one the samplers can
    /// evaluate at all (see [`Graph::reachable_param`]), so a new `Op` variant
    /// that silently reported no dependencies would reopen a hole rather than
    /// fail a build. Adding a variant must not compile until someone has
    /// written down what it forwards.
    ///
    /// This is a *data* dependency, not the reverse-mode adjoint path: several
    /// ops here read an operand whose adjoint they never propagate to.
    pub(crate) fn visit_dependencies(
        &self,
        visit_node: &mut impl FnMut(NodeId),
        visit_params: &mut impl FnMut(usize, usize),
    ) {
        match self {
            Op::Elementwise { a, b, .. } => {
                visit_node(*a);
                if let Some(b) = b {
                    visit_node(*b);
                }
            }
            Op::Gather {
                param_start,
                n_params,
                indices,
            } => {
                visit_params(*param_start, *n_params);
                visit_node(*indices);
            }
            Op::Sum(a) => visit_node(*a),
            Op::BroadcastObservation { scalar, .. } => visit_node(*scalar),
            Op::Param(index) => visit_params(*index, 1),
            Op::Constant(_) | Op::Data(_) => {}
            Op::Add(a, b)
            | Op::Mul(a, b)
            | Op::ScalarMulData(a, b)
            | Op::VectorAdd(a, b)
            | Op::ScalarBroadcastAdd(a, b) => {
                visit_node(*a);
                visit_node(*b);
            }
            Op::Exp(a) | Op::Sigmoid(a) | Op::ScalarBroadcast(a) => visit_node(*a),
            Op::BoundedSigmoid { raw, .. } => visit_node(*raw),
            Op::NormalLogP { x, mu, sigma } => {
                visit_node(*x);
                visit_node(*mu);
                visit_node(*sigma);
            }
            Op::ObsLogP {
                linpred_vec, aux, ..
            } => {
                visit_node(*linpred_vec);
                if let Some(aux) = aux {
                    visit_node(*aux);
                }
            }
            Op::LogHalfNormalLogP { x, sigma } => {
                visit_node(*x);
                visit_node(*sigma);
            }
            Op::StudentTLogP { x, nu, mu, sigma } => {
                visit_node(*x);
                visit_node(*nu);
                visit_node(*mu);
                visit_node(*sigma);
            }
            Op::PositiveSupport { x } => visit_node(*x),
            Op::BernoulliLogP { x, p } => {
                visit_node(*x);
                visit_node(*p);
            }
            Op::PoissonLogP { x, lam } => {
                visit_node(*x);
                visit_node(*lam);
            }
            Op::LogGammaLogP { x, alpha, beta } => {
                visit_node(*x);
                visit_node(*alpha);
                visit_node(*beta);
            }
            Op::FusedLinearMu {
                param_nodes,
                intercept,
                ..
            } => {
                for node in param_nodes {
                    visit_node(*node);
                }
                if let Some(intercept) = intercept {
                    visit_node(*intercept);
                }
            }
            Op::MatVecMul {
                param_start,
                n_params,
                intercept,
                ..
            } => {
                visit_params(*param_start, *n_params);
                if let Some(intercept) = intercept {
                    visit_node(*intercept);
                }
            }
            Op::VectorNormalLogP {
                param_start,
                n_params,
                ..
            }
            | Op::VectorHalfNormalLogP {
                param_start,
                n_params,
                ..
            }
            | Op::VectorStudentTLogP {
                param_start,
                n_params,
                ..
            }
            | Op::VectorGammaLogP {
                param_start,
                n_params,
                ..
            }
            | Op::VectorBetaLogP {
                param_start,
                n_params,
                ..
            }
            | Op::VectorUniformLogP {
                param_start,
                n_params,
                ..
            } => visit_params(*param_start, *n_params),
        }
    }
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
            ParamTransform::Sigmoid => stable_sigmoid(raw),
            ParamTransform::BoundedSigmoid { lower, upper } => bounded_sigmoid(raw, *lower, *upper),
        }
    }

    /// Derivative of the constrained value with respect to the raw value.
    #[inline]
    pub fn derivative(&self, raw: f64) -> f64 {
        match self {
            ParamTransform::Identity => 1.0,
            ParamTransform::Exp => raw.exp(),
            ParamTransform::Sigmoid => stable_sigmoid_derivative(raw),
            ParamTransform::BoundedSigmoid { lower, upper } => {
                bounded_sigmoid_derivative(raw, *lower, *upper)
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
    pub deterministics: Vec<(String, NodeId)>,
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
            deterministics: Vec::new(),
            name_to_node: HashMap::new(),
        }
    }

    /// The lowest-indexed free parameter `root`'s value depends on, if any.
    ///
    /// Walks operands transitively through [`Op::visit_dependencies`], so a
    /// parameter reached through any number of intervening nodes counts — a
    /// direct `Op::Param` is just the zero-step case. Used by the samplers to
    /// refuse a density they cannot evaluate; the lowest index is taken rather
    /// than the first one found so the error message does not depend on the
    /// traversal order.
    ///
    /// The dependence is syntactic, not semantic. An op that reads a parameter
    /// and ignores it — `theta.powf(0.0)`, which is identically 1 — still
    /// counts as reaching it, and an op that indexes a parameter span reports
    /// the span's first slot rather than the element it will select at runtime,
    /// so a `Gather` over `coef[1]` names `coef[0]`. Both are deliberate: this
    /// backs a safety check, where naming a neighbouring parameter or refusing
    /// a pathological spelling of a constant is the cheap failure, and missing
    /// a real dependence is the expensive one.
    ///
    /// Termination does not rest on the node order: `nodes` is public and a
    /// `NodeId` is an unchecked index, so a caller can build a graph that is
    /// not topologically sorted. The `expanded` set is what bounds the walk —
    /// each node is expanded at most once even under a cycle.
    pub(crate) fn reachable_param(&self, root: NodeId) -> Option<usize> {
        let mut expanded = vec![false; self.nodes.len()];
        let mut stack = vec![root];
        let mut lowest: Option<usize> = None;
        while let Some(id) = stack.pop() {
            let Some(node) = self.nodes.get(id.0) else {
                continue;
            };
            if std::mem::replace(&mut expanded[id.0], true) {
                continue;
            }
            node.op.visit_dependencies(
                &mut |next| stack.push(next),
                &mut |param_start, n_params| {
                    if n_params > 0 {
                        lowest =
                            Some(lowest.map_or(param_start, |seen: usize| seen.min(param_start)));
                    }
                },
            );
        }
        lowest
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

    pub fn elementwise(&mut self, operator: ElementwiseOp, a: NodeId, b: Option<NodeId>) -> NodeId {
        self.add_node(Op::Elementwise { operator, a, b }, None)
    }
    pub fn gather(&mut self, param_start: usize, n_params: usize, indices: NodeId) -> NodeId {
        self.add_node(
            Op::Gather {
                param_start,
                n_params,
                indices,
            },
            None,
        )
    }
    pub fn sum(&mut self, input: NodeId) -> NodeId {
        self.add_node(Op::Sum(input), None)
    }
    pub fn broadcast_observation(&mut self, scalar: NodeId, obs_data_idx: usize) -> NodeId {
        self.add_node(
            Op::BroadcastObservation {
                scalar,
                obs_data_idx,
            },
            None,
        )
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

    pub fn exp(&mut self, a: NodeId) -> NodeId {
        self.add_node(Op::Exp(a), None)
    }

    pub fn sigmoid(&mut self, a: NodeId) -> NodeId {
        self.add_node(Op::Sigmoid(a), None)
    }

    /// The constrained value of a [`ParamTransform::BoundedSigmoid`] parameter.
    ///
    /// Pair it with a parameter carrying the matching transform; see
    /// [`Op::BoundedSigmoid`] for why this is one node rather than three.
    pub fn bounded_sigmoid(&mut self, raw: NodeId, lower: f64, upper: f64) -> NodeId {
        self.add_node(Op::BoundedSigmoid { raw, lower, upper }, None)
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

    /// Density in log-parameter space, including the exp Jacobian.
    pub fn log_half_normal_logp(&mut self, x: NodeId, sigma: NodeId) -> NodeId {
        let node = self.add_node(Op::LogHalfNormalLogP { x, sigma }, None);
        self.logp_terms.push(node);
        node
    }

    pub fn student_t_logp(&mut self, x: NodeId, nu: NodeId, mu: NodeId, sigma: NodeId) -> NodeId {
        let node = self.add_node(Op::StudentTLogP { x, nu, mu, sigma }, None);
        self.logp_terms.push(node);
        node
    }

    /// Retain the positive finite domain of a scale after reparameterization.
    pub fn positive_support(&mut self, x: NodeId) -> NodeId {
        let node = self.add_node(Op::PositiveSupport { x }, None);
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

    /// Density in log-parameter space, including the exp Jacobian.
    pub fn log_gamma_logp(&mut self, x: NodeId, alpha: NodeId, beta: NodeId) -> NodeId {
        let node = self.add_node(Op::LogGammaLogP { x, alpha, beta }, None);
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
    ///
    /// A structure-only graph intentionally has no dataset payload. In that
    /// case `n_obs` is zero; row count belongs to a `DataBinding` and becomes
    /// available again on a hydrated graph returned by `with_binding()`.
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
                        dim: self
                            .schema
                            .observations
                            .get(*obs_data_idx)
                            .map_or_else(|| "obs".into(), |slot| slot.dim.clone()),
                        name: self
                            .schema
                            .observations
                            .get(*obs_data_idx)
                            .and_then(|slot| match &slot.kind {
                                SlotKind::Observation { likelihood } => Some(likelihood.clone()),
                                _ => None,
                            })
                            .unwrap_or_else(|| n.name.clone().unwrap_or_default()),
                        family: *family,
                        linpred: *linpred_vec,
                        aux: *aux,
                        obs_data_idx: *obs_data_idx,
                        n_obs: self
                            .obs_vectors
                            .get(*obs_data_idx)
                            .map_or(0, |values| values.len()),
                    })
                } else {
                    None
                }
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
        let binding =
            DataBinding::from_graph(self).map_err(|e| GraphShapeError::new(e.to_string()))?;
        // The same coverage check the evaluator makes, for the same reason:
        // `validate_node_lengths` indexes the binding at the raw slot indices
        // the graph carries. Without it, `broadcast_observation(p, 0)` on a
        // graph with no observation payload -- both public builders -- indexes
        // an empty vector and panics, and because `sampler::sample` validates
        // shapes before it does anything else, that panic came out of `sample`
        // in place of the `Result` it promises.
        crate::autodiff::validate_slot_coverage(self, &binding)?;
        crate::autodiff::validate_node_lengths(self, &binding)?;
        Ok(binding.n_obs())
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}
