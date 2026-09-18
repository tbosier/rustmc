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

/// Logistic sigmoid, `1 / (1 + exp(-x))`, evaluated without overflow.
///
/// The textbook form overflows `exp(-x)` for `x < -709` and returns 0 where
/// the true value is an ordinary subnormal, so branch on the sign and
/// exponentiate the negative argument.
#[inline]
pub fn stable_sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Derivative of [`stable_sigmoid`].
///
/// Mathematically `s(x) * (1 - s(x))`, but `1 - s(x)` cancels to exactly zero
/// once `s(x)` rounds to 1 (around `x = 37`), discarding a value that stays
/// representable out to `x = 745`. `exp(-|x|) / (1 + exp(-|x|))^2` is the same
/// quantity, is symmetric in `x` by construction, and needs one `exp`. Near
/// `x = 0` the old form was marginally more accurate, because `1 - s` is exact
/// there by Sterbenz; this one stays within about two ulp everywhere instead.
#[inline]
pub fn stable_sigmoid_derivative(x: f64) -> f64 {
    let e = (-x.abs()).exp();
    let d = 1.0 + e;
    e / (d * d)
}

/// `lower + (upper - lower) * sigmoid(raw)`, anchored to whichever endpoint the
/// value is nearest.
///
/// This is the single definition of the bounded transform. Both the
/// constrained draw reported back to a caller ([`ParamTransform::apply`]) and
/// the value the graph evaluates its density at ([`Op::BoundedSigmoid`]) call
/// it, so they cannot drift into two formulas that disagree about where the
/// model was evaluated. That drift is what this function exists to prevent —
/// the formula itself is unchanged.
///
/// `lower + span * s` cancels catastrophically once `s` is near 1 and `lower`
/// is large and negative: with lower = -1e308 and upper = 1, raw = 710 gives
/// -1e308 + 1e308 == 0 instead of 0.552371377432487. Both branches also round
/// towards the endpoint they subtract from, so for any interval whose span is
/// representable the result can never leave `[lower, upper]`. That guarantee is
/// why the branch is here rather than the branch-free convex combination
/// `lower * s(-raw) + upper * s(raw)`: the latter is the same quantity in exact
/// arithmetic, but its two separately rounded products can land outside a
/// narrow interval far from zero — `(1e16, 1e16 + 2)` at raw -34.5 gives
/// 9999999999999998, below `lower` — and can overflow to infinity when both
/// endpoints are near `f64::MAX`.
///
/// Three limits remain, none of them new:
///
/// * The branches can disagree by one ulp at raw == 0, so the mapping is not
///   exactly monotone there (lower = 1, upper = 1e16 steps down by one ulp
///   across zero); that is a rounding artefact of the reported value, and
///   [`bounded_sigmoid_derivative`] does not model it.
/// * Materialising the sigmoid before scaling loses bounded values whose
///   sigmoid underflows: lower = 0, upper = 1e308, raw = -746 gives 0 where the
///   true value is 1.0382848095158282e-16. Removing that needs the scaling
///   folded into the sigmoid.
/// * An interval whose span is not representable — `(-1e308, 1e308)` has
///   `upper - lower == inf` — yields `-inf` or `NaN` here. Such an interval is
///   not a usable `Uniform` prior in any case: both `uniform_bounds_valid` and
///   `model`'s `validate_positive_finite("uniform width", ...)` require a
///   finite width, so its density is `-inf` everywhere regardless of the
///   transform.
#[inline]
pub fn bounded_sigmoid(raw: f64, lower: f64, upper: f64) -> f64 {
    let span = upper - lower;
    if raw >= 0.0 {
        upper - span * stable_sigmoid(-raw)
    } else {
        lower + span * stable_sigmoid(raw)
    }
}

/// d/draw of [`bounded_sigmoid`], i.e. `(upper - lower) * s'(raw)`.
///
/// `span * s'` cannot overflow: `s'` never exceeds 0.25, so the product is at
/// most a quarter of a representable span. Distributing it over the endpoints
/// as `upper * s' - lower * s'` would not be safer — for `(1e16, 1e16 + 2)` at
/// raw 0.176 the two products round to the same f64 and cancel to exactly 0,
/// discarding a derivative of 0.4961479024771348.
#[inline]
pub fn bounded_sigmoid_derivative(raw: f64, lower: f64, upper: f64) -> f64 {
    (upper - lower) * stable_sigmoid_derivative(raw)
}

/// `adjoint * (upper - lower) * s'(raw)`, associated so that the product stays
/// inside the exponent range wherever the exact value is representable.
///
/// Reverse mode has three factors to multiply and only two orders worth
/// considering, and each fails where the other succeeds:
///
/// * Applying the span to the adjoint first is what the old three-node
///   `sigmoid`/`mul`/`add` chain did, and it overflows: `Uniform(0, 1e308)` at
///   raw -710 under a `sigma = 0.1` likelihood has an adjoint of -44.76, and
///   `-44.76 * 1e308` is `-inf` where the true gradient is -19.04.
/// * Applying the span to the slope first — [`bounded_sigmoid_derivative`] —
///   fixes that, but underflows in the mirror case: `(0, 1e-308)` at raw -40
///   has `span * s' == 4.2e-326`, which rounds to zero and erases a gradient of
///   4.248354255291588e-18 under an adjoint of 1e308.
///
/// So take the second order, and fall back to the first only when it lost the
/// value outright. The fallback cannot itself overflow: `span * s'` only
/// underflows when the span is tiny, and multiplying the adjoint by a tiny span
/// moves it towards zero.
///
/// The band where the fallback fires is narrow. `s / s'` is `1 + e^-|raw|`,
/// which never exceeds 2, so `span * s'` and the constrained value's distance
/// from its nearer endpoint underflow within a factor of two of each other:
/// almost everywhere `span * s'` rounds to zero, the value is pinned at an
/// endpoint and a zero gradient is the honest derivative of the rounded map.
/// A subnormal span near raw 0 is the exception — `(0, 1e-323)` has
/// `span * s' == 2.5e-324`, which rounds to zero while the value still moves —
/// and that is the case this order rescues.
#[inline]
pub fn bounded_sigmoid_adjoint(adjoint: f64, raw: f64, lower: f64, upper: f64) -> f64 {
    let slope = stable_sigmoid_derivative(raw);
    let span = upper - lower;
    let scaled = span * slope;
    if scaled == 0.0 && span != 0.0 && slope != 0.0 {
        (adjoint * span) * slope
    } else {
        adjoint * scaled
    }
}

/// d/db of `a / b`, i.e. `-a / b^2`, over the whole representable range.
///
/// `-a / (b * b)` is the accurate form and is used wherever the squared
/// denominator is a normal number: `b * b` is then either exact or off by half
/// an ulp, so the result is within one ulp of the true quotient.
///
/// It fails outside that band, and not gracefully. `b * b` overflows to
/// infinity for `|b| > 1.34e154`, so `-a / inf` returns `-0.0` and erases a
/// derivative that is often perfectly representable (`a = b = 1e200` has
/// derivative `-1e-200`). Going the other way it decays into the subnormals
/// below `|b| = 1.49e-154` and reaches zero below `|b| = 1.58e-162`, so
/// `-a / 0.0` fabricates an infinity where the true derivative is finite
/// (`a = b = 1e-200` has derivative `-1e200`).
///
/// In that band divide twice instead. Two divisions never leave the exponent
/// range, at the cost of a second rounding — which is why this is a fallback
/// and not the only path: double rounding through the subnormals can turn a
/// representable `-5e-324` into `-0.0` (`a = 1.5e-323`, `b = 2.2`), exactly
/// the failure the fallback exists to prevent. Restricting it to the range
/// where `-a / (b * b)` is already broken keeps both forms on the inputs they
/// handle well.
#[inline]
fn div_denominator_derivative(a: f64, b: f64) -> f64 {
    let square = b * b;
    if square.is_normal() {
        -a / square
    } else {
        // Covers b == 0 (the square is +0 either way, so both forms give the
        // same signed infinity or NaN), the two overflow/underflow tails, and
        // the subnormal square band in between.
        -(a / b) / b
    }
}

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
            Self::Tanh => (1.0 - a.tanh().powi(2), 0.0),
            Self::Softplus => (stable_sigmoid(a), 0.0),
            Self::Sin => (a.cos(), 0.0),
            Self::Cos => (-a.sin(), 0.0),
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
    /// Domain constraint with zero log density for finite x > 0, -infinity otherwise.
    PositiveSupport {
        x: NodeId,
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
    /// Combined density and exp-transform Jacobian; x is the raw log value.
    LogGammaLogP {
        x: NodeId,
        alpha: NodeId,
        beta: NodeId,
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
            Op::LogHalfNormalLogP { x, sigma } | Op::HalfNormalLogP { x, sigma } => {
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
            Op::UniformLogP { x, lower, upper } => {
                visit_node(*x);
                visit_node(*lower);
                visit_node(*upper);
            }
            Op::BernoulliLogP { x, p } => {
                visit_node(*x);
                visit_node(*p);
            }
            Op::PoissonLogP { x, lam } => {
                visit_node(*x);
                visit_node(*lam);
            }
            Op::LogGammaLogP { x, alpha, beta }
            | Op::GammaLogP { x, alpha, beta }
            | Op::BetaLogP { x, alpha, beta } => {
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

    /// Retain the positive finite domain of a scale after reparameterization.
    pub fn positive_support(&mut self, x: NodeId) -> NodeId {
        let node = self.add_node(Op::PositiveSupport { x }, None);
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

    /// Density in log-parameter space, including the exp Jacobian.
    pub fn log_gamma_logp(&mut self, x: NodeId, alpha: NodeId, beta: NodeId) -> NodeId {
        let node = self.add_node(Op::LogGammaLogP { x, alpha, beta }, None);
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

    /// Backward-compatible helper for the current Normal-only API surface.
    #[deprecated(note = "use observation_heads for all supported families")]
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
        let binding =
            DataBinding::from_graph(self).map_err(|e| GraphShapeError::new(e.to_string()))?;
        crate::autodiff::validate_node_lengths(self, &binding)?;
        Ok(binding.n_obs())
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}
