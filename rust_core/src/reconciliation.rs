//! Forecast reconciliation: making the forecasts of a hierarchy add up.
//!
//! A hierarchy of `n` series is described by its summing matrix `S`
//! (`n x m`): the last `m` rows are the bottom-level series and form the
//! identity, and each of the first `r = n - m` rows is an aggregate that sums
//! the bottom series marked with a one. Base forecasts `y` made separately for
//! every node rarely satisfy `y = S b` for any `b`. A reconciliation method
//! picks a matrix `P` (`m x n`) and replaces `y` by `S P y`, which is coherent
//! by construction.
//!
//! The methods here are:
//!
//! - bottom-up, `P = [0 | I]`: keep the bottom forecasts and sum them;
//! - top-down by average historical proportions (Gross & Sohl 1990, their
//!   method A) or by forecast proportions (Athanasopoulos, Ahmed & Hyndman
//!   2009): split the total forecast among the bottom series;
//! - the projections `P = (S' W^-1 S)^-1 S' W^-1` with `W` the identity (OLS),
//!   `diag(S 1)` (WLS with structural scaling), the diagonal of the residual
//!   covariance (WLS with variance scaling), or the Schafer-Strimmer shrinkage
//!   of the residual covariance towards its diagonal (MinT with shrinkage,
//!   Wickramasuriya, Athanasopoulos & Hyndman 2019).
//!
//! The projections are computed without inverting `W` or forming any `n x n`
//! matrix, through the equivalent form from Wickramasuriya et al. (2019),
//! `P = J - J W U (U' W U)^-1 U'`, where `U' = [I_r | -A]` and
//! `J = [0 | I_m]`. `U' y` is the incoherence of `y` - each aggregate minus the
//! sum of its bottom series - so a coherent `y` is returned unchanged, and the
//! only system solved is `r x r`, one row per aggregate.
//!
//! # Draws
//!
//! Reconciliation is applied to every joint draw separately (Panagiotelis et
//! al. 2023): draw `(c, d)` at step `h` is the vector of every node's value in
//! that draw, and its reconciled version is `S P` applied to it. When the base
//! draws come from one joint predictive distribution, the reconciled draws are
//! draws from the reconciled distribution, so their spread and dependence stay
//! consistent across the hierarchy. That requires every node's draw `(c, d)`
//! to belong to the same joint draw. Draws generated independently per node
//! can still be reconciled, and each reconciled draw is still coherent, but the
//! result then describes base forecast errors that are independent across
//! nodes: the reconciled means are unaffected (the map is linear), while the
//! spread of every node that combines several base forecasts is not the one a
//! joint predictive would give.

use crate::bayesian_forecast::ForecastQuantile;
use crate::forecast_common::{
    checked_value_count, cholesky, solve_lower, solve_lower_transpose, sorted_quantile,
    AllocationLimitError, MAX_MATERIALIZED_VALUES,
};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt;

/// Why a hierarchy, its inputs or a reconciliation were refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReconciliationError {
    /// The summing matrix, labels or node names do not describe a hierarchy.
    InvalidHierarchy(String),
    /// Draws, residuals or history have the wrong shape or an unusable value.
    InvalidInput(String),
    /// The method does not apply to this hierarchy, such as top-down without
    /// a total node.
    UnsupportedMethod(String),
    /// The weights of a projection could not be factorized.
    NumericalFailure(String),
}

impl fmt::Display for ReconciliationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidHierarchy(message) => write!(f, "invalid hierarchy: {message}"),
            Self::InvalidInput(message) => write!(f, "invalid input: {message}"),
            Self::UnsupportedMethod(message) => write!(f, "unsupported method: {message}"),
            Self::NumericalFailure(message) => write!(f, "numerical failure: {message}"),
        }
    }
}

impl Error for ReconciliationError {}

impl From<AllocationLimitError> for ReconciliationError {
    fn from(error: AllocationLimitError) -> Self {
        Self::InvalidInput(error.to_string())
    }
}

fn invalid_hierarchy(message: impl Into<String>) -> ReconciliationError {
    ReconciliationError::InvalidHierarchy(message.into())
}

fn invalid_input(message: impl Into<String>) -> ReconciliationError {
    ReconciliationError::InvalidInput(message.into())
}

/// The nodes of a hierarchy and which bottom series each aggregate sums.
///
/// Nodes are ordered aggregates first, then the bottom series, matching the
/// rows of the summing matrix `S = [A; I]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hierarchy {
    names: Vec<String>,
    bottom_count: usize,
    /// For each aggregate, the ascending bottom indices it sums.
    supports: Vec<Vec<usize>>,
    /// Parent and depth of every node when the aggregates nest.
    tree: Option<Tree>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Tree {
    parents: Vec<Option<usize>>,
    depths: Vec<usize>,
}

impl Hierarchy {
    /// A hierarchy from an explicit summing matrix, given as its `n` rows.
    ///
    /// Every entry must be exactly 0 or 1, the last `m` rows must be the
    /// `m x m` identity (the bottom series), and at least one row above them
    /// must be an aggregate summing one or more bottom series. Structures
    /// that do not nest, such as a product grouping crossed with a region
    /// grouping, are accepted; only top-down by forecast proportions needs
    /// the aggregates to nest. `names` defaults to `node_0`, `node_1`, ... in
    /// row order.
    pub fn from_summing_matrix(
        rows: &[Vec<f64>],
        names: Option<Vec<String>>,
    ) -> Result<Self, ReconciliationError> {
        let node_count = rows.len();
        let bottom_count = rows.first().map_or(0, Vec::len);
        if bottom_count == 0 {
            return Err(invalid_hierarchy(
                "summing_matrix must have at least one column (bottom series)",
            ));
        }
        if let Some((index, row)) = rows
            .iter()
            .enumerate()
            .find(|(_, row)| row.len() != bottom_count)
        {
            return Err(invalid_hierarchy(format!(
                "summing_matrix rows must all have {bottom_count} columns; row {index} has {}",
                row.len()
            )));
        }
        if node_count <= bottom_count {
            return Err(invalid_hierarchy(format!(
                "summing_matrix has {node_count} rows for {bottom_count} bottom series; it \
                 needs the {bottom_count} identity rows plus at least one aggregate row"
            )));
        }
        for (index, row) in rows.iter().enumerate() {
            if let Some(column) = row.iter().position(|&value| value != 0.0 && value != 1.0) {
                return Err(invalid_hierarchy(format!(
                    "summing_matrix[{index}, {column}] is {}; a summing matrix holds only 0 and 1",
                    row[column]
                )));
            }
        }
        let aggregate_count = node_count - bottom_count;
        for (offset, row) in rows[aggregate_count..].iter().enumerate() {
            if row
                .iter()
                .enumerate()
                .any(|(column, &value)| value != f64::from(u8::from(column == offset)))
            {
                return Err(invalid_hierarchy(format!(
                    "the last {bottom_count} rows of summing_matrix must be the identity \
                     (one row per bottom series, in order); row {} is not",
                    aggregate_count + offset
                )));
            }
        }
        let supports: Vec<Vec<usize>> = rows[..aggregate_count]
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .filter(|(_, &value)| value == 1.0)
                    .map(|(column, _)| column)
                    .collect()
            })
            .collect();
        if let Some(index) = supports.iter().position(Vec::is_empty) {
            return Err(invalid_hierarchy(format!(
                "summing_matrix row {index} sums no bottom series"
            )));
        }
        let names = names.unwrap_or_else(|| (0..node_count).map(|i| format!("node_{i}")).collect());
        Self::new(names, bottom_count, supports)
    }

    /// A nested hierarchy from the labels of each bottom series.
    ///
    /// `labels[j]` is the path of bottom series `j` from the top level down,
    /// such as `["BU1", "IDN3", "cust17"]`; every path has the same length
    /// and no two are equal. The nodes are, in order: the total, named
    /// `total`; then every distinct path prefix of length 1, 2, ... in order
    /// of first appearance in `labels`; then the bottom series in the order
    /// given. A node is named by its path joined with `separator`, so the same
    /// label under two parents names two nodes (`BU1/IDN3`, `BU2/IDN3`).
    /// Labels must be non-empty and must not contain `separator`.
    pub fn from_labels(
        labels: &[Vec<String>],
        total: &str,
        separator: &str,
    ) -> Result<Self, ReconciliationError> {
        if labels.is_empty() {
            return Err(invalid_hierarchy(
                "labels must describe at least one bottom series",
            ));
        }
        if total.is_empty() {
            return Err(invalid_hierarchy("the total node's name must not be empty"));
        }
        if separator.is_empty() {
            return Err(invalid_hierarchy("separator must not be empty"));
        }
        let depth = labels[0].len();
        if depth == 0 {
            return Err(invalid_hierarchy(
                "each bottom series needs at least one label; labels[0] is empty",
            ));
        }
        for (index, path) in labels.iter().enumerate() {
            if path.len() != depth {
                return Err(invalid_hierarchy(format!(
                    "every label path must have the same length; labels[0] has {depth} \
                     labels and labels[{index}] has {}",
                    path.len()
                )));
            }
            for (level, label) in path.iter().enumerate() {
                if label.is_empty() {
                    return Err(invalid_hierarchy(format!(
                        "labels[{index}][{level}] is empty"
                    )));
                }
                if label.contains(separator) {
                    return Err(invalid_hierarchy(format!(
                        "labels[{index}][{level}] ({label:?}) contains the separator \
                         {separator:?}; pass a separator that no label contains"
                    )));
                }
            }
        }
        let bottom_count = labels.len();
        let mut names = vec![total.to_string()];
        let mut supports = vec![(0..bottom_count).collect::<Vec<_>>()];
        for length in 1..depth {
            let mut index_of: HashMap<&[String], usize> = HashMap::new();
            for (bottom, path) in labels.iter().enumerate() {
                let prefix = &path[..length];
                let node = *index_of.entry(prefix).or_insert_with(|| {
                    names.push(prefix.join(separator));
                    supports.push(Vec::new());
                    supports.len() - 1
                });
                supports[node].push(bottom);
            }
        }
        let mut seen = HashSet::new();
        for (index, path) in labels.iter().enumerate() {
            if !seen.insert(path) {
                return Err(invalid_hierarchy(format!(
                    "labels[{index}] repeats an earlier bottom series {path:?}"
                )));
            }
            names.push(path.join(separator));
        }
        Self::new(names, bottom_count, supports)
    }

    fn new(
        names: Vec<String>,
        bottom_count: usize,
        supports: Vec<Vec<usize>>,
    ) -> Result<Self, ReconciliationError> {
        let node_count = supports.len() + bottom_count;
        if names.len() != node_count {
            return Err(invalid_hierarchy(format!(
                "node_names must hold one name per node ({node_count}); got {}",
                names.len()
            )));
        }
        let mut seen = HashSet::new();
        for (index, name) in names.iter().enumerate() {
            if name.is_empty() {
                return Err(invalid_hierarchy(format!("node name {index} is empty")));
            }
            if !seen.insert(name.as_str()) {
                return Err(invalid_hierarchy(format!(
                    "node name {name:?} occurs more than once; node names must be unique"
                )));
            }
        }
        let tree = nest(bottom_count, &supports);
        Ok(Self {
            names,
            bottom_count,
            supports,
            tree,
        })
    }

    pub fn node_count(&self) -> usize {
        self.names.len()
    }

    pub fn bottom_count(&self) -> usize {
        self.bottom_count
    }

    pub fn aggregate_count(&self) -> usize {
        self.supports.len()
    }

    /// Every node's name, aggregates first, in row order of `S`.
    pub fn node_names(&self) -> &[String] {
        &self.names
    }

    pub fn aggregate_names(&self) -> &[String] {
        &self.names[..self.aggregate_count()]
    }

    pub fn bottom_names(&self) -> &[String] {
        &self.names[self.aggregate_count()..]
    }

    /// The ascending bottom indices that aggregate `index` sums.
    pub fn support(&self, index: usize) -> &[usize] {
        &self.supports[index]
    }

    /// The dense summing matrix as its rows.
    pub fn summing_matrix(&self) -> Result<Vec<Vec<f64>>, ReconciliationError> {
        checked_value_count(
            "summing matrix",
            &[self.node_count(), self.bottom_count],
            MAX_MATERIALIZED_VALUES,
        )?;
        let mut rows = vec![vec![0.0; self.bottom_count]; self.node_count()];
        for (row, support) in rows.iter_mut().zip(&self.supports) {
            for &bottom in support {
                row[bottom] = 1.0;
            }
        }
        for bottom in 0..self.bottom_count {
            rows[self.aggregate_count() + bottom][bottom] = 1.0;
        }
        Ok(rows)
    }

    /// Whether every two aggregates either nest or share no bottom series,
    /// so that the nodes form a tree.
    pub fn is_nested(&self) -> bool {
        self.tree.is_some()
    }

    /// The parent of every node in a nested hierarchy; `None` for the root
    /// nodes. Two aggregates summing the same bottom series are a chain, the
    /// earlier row being the parent.
    pub fn parents(&self) -> Option<&[Option<usize>]> {
        self.tree.as_ref().map(|tree| tree.parents.as_slice())
    }

    /// The depth of every node in a nested hierarchy, the total being 0.
    pub fn depths(&self) -> Option<&[usize]> {
        self.tree.as_ref().map(|tree| tree.depths.as_slice())
    }

    /// The first aggregate that sums every bottom series.
    pub fn total(&self) -> Option<usize> {
        self.supports
            .iter()
            .position(|support| support.len() == self.bottom_count)
    }
}

/// Parents and depths of every node when the aggregates form a laminar family
/// (any two are disjoint or nested), or `None`.
///
/// Aggregates are visited largest first, ties in row order. Each bottom series
/// remembers the last visited aggregate containing it, which in a laminar
/// family is the smallest one so far. A visited aggregate is compatible with
/// every earlier one exactly when all its bottom series remember the same
/// aggregate, which is then its parent: any earlier aggregate meeting it is at
/// least as large, so must contain it.
fn nest(bottom_count: usize, supports: &[Vec<usize>]) -> Option<Tree> {
    let aggregates = supports.len();
    let mut order: Vec<usize> = (0..aggregates).collect();
    order.sort_by(|&a, &b| supports[b].len().cmp(&supports[a].len()).then(a.cmp(&b)));
    let mut deepest: Vec<Option<usize>> = vec![None; bottom_count];
    let mut parents = vec![None; aggregates + bottom_count];
    let mut depths = vec![0; aggregates + bottom_count];
    for &node in &order {
        let parent = deepest[supports[node][0]];
        if supports[node]
            .iter()
            .any(|&bottom| deepest[bottom] != parent)
        {
            return None;
        }
        parents[node] = parent;
        depths[node] = parent.map_or(0, |parent| depths[parent] + 1);
        for &bottom in &supports[node] {
            deepest[bottom] = Some(node);
        }
    }
    for (bottom, parent) in deepest.into_iter().enumerate() {
        parents[aggregates + bottom] = parent;
        depths[aggregates + bottom] = parent.map_or(0, |parent| depths[parent] + 1);
    }
    Some(Tree { parents, depths })
}

/// Joint draws of every node, indexed `(chain, draw, node, step)` row-major.
#[derive(Debug, Clone, PartialEq)]
pub struct NodeDraws {
    values: Vec<f64>,
    chains: usize,
    draws: usize,
    nodes: usize,
    steps: usize,
}

impl NodeDraws {
    /// Wrap row-major `(chain, draw, node, step)` values, which must be
    /// non-empty along every axis and finite.
    pub fn new(
        values: Vec<f64>,
        chains: usize,
        draws: usize,
        nodes: usize,
        steps: usize,
    ) -> Result<Self, ReconciliationError> {
        if chains == 0 || draws == 0 || nodes == 0 || steps == 0 {
            return Err(invalid_input(format!(
                "draws must have at least one chain, draw, node and step; got shape \
                 ({chains}, {draws}, {nodes}, {steps})"
            )));
        }
        let expected = checked_value_count(
            "reconciliation draws",
            &[chains, draws, nodes, steps],
            MAX_MATERIALIZED_VALUES,
        )?;
        if values.len() != expected {
            return Err(invalid_input(format!(
                "draws of shape ({chains}, {draws}, {nodes}, {steps}) need {expected} values; \
                 got {}",
                values.len()
            )));
        }
        if let Some(position) = values.iter().position(|value| !value.is_finite()) {
            let step = position % steps;
            let node = position / steps % nodes;
            let draw = position / (steps * nodes) % draws;
            let chain = position / (steps * nodes * draws);
            return Err(invalid_input(format!(
                "draws[{chain}, {draw}, {node}, {step}] is not finite"
            )));
        }
        Ok(Self {
            values,
            chains,
            draws,
            nodes,
            steps,
        })
    }

    /// `(chains, draws, nodes, steps)`.
    pub fn shape(&self) -> (usize, usize, usize, usize) {
        (self.chains, self.draws, self.nodes, self.steps)
    }

    /// The row-major `(chain, draw, node, step)` values.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    pub fn into_values(self) -> Vec<f64> {
        self.values
    }

    fn samples(&self) -> usize {
        self.chains * self.draws
    }

    /// The values of `node` at `step` across every chain and draw.
    fn column(&self, node: usize, step: usize) -> impl Iterator<Item = f64> + '_ {
        let stride = self.nodes * self.steps;
        let offset = node * self.steps + step;
        (0..self.samples()).map(move |sample| self.values[sample * stride + offset])
    }

    /// The mean of every node at every step, node-major, by the scale-aware
    /// mean the forecast summaries use.
    pub fn means(&self) -> Vec<f64> {
        (0..self.nodes * self.steps)
            .into_par_iter()
            .map(|index| {
                let (node, step) = (index / self.steps, index % self.steps);
                crate::diagnostics::scaled_moments(|| self.column(node, step)).0
            })
            .collect()
    }

    /// Empirical quantiles of every node at every step, node-major, by
    /// [`sorted_quantile`].
    pub fn quantiles(
        &self,
        probabilities: &[f64],
    ) -> Result<Vec<ForecastQuantile>, ReconciliationError> {
        if probabilities
            .iter()
            .any(|probability| !probability.is_finite() || !(0.0..=1.0).contains(probability))
        {
            return Err(invalid_input(
                "quantile probabilities must be finite and between zero and one",
            ));
        }
        let per_column: Vec<Vec<f64>> = (0..self.nodes * self.steps)
            .into_par_iter()
            .map(|index| {
                let (node, step) = (index / self.steps, index % self.steps);
                let mut ordered: Vec<f64> = self.column(node, step).collect();
                ordered.sort_by(f64::total_cmp);
                probabilities
                    .iter()
                    .map(|&probability| sorted_quantile(&ordered, probability))
                    .collect()
            })
            .collect();
        Ok(probabilities
            .iter()
            .enumerate()
            .map(|(row, &probability)| ForecastQuantile {
                probability,
                values: per_column.iter().map(|column| column[row]).collect(),
            })
            .collect())
    }
}

/// How to reconcile. Residuals are in-sample one-step forecast errors, one row
/// per time point and one column per node in node order; history holds the
/// observed bottom series, one row per time point and one column per bottom
/// series.
#[derive(Debug, Clone, PartialEq)]
pub enum ReconciliationMethod {
    BottomUp,
    /// Gross & Sohl's method A: bottom series `j` receives the average over
    /// time of its share of the total, `mean_t(y[t, j] / sum_k y[t, k])`.
    TopDownAverageProportions {
        history: Vec<Vec<f64>>,
    },
    /// Athanasopoulos, Ahmed & Hyndman (2009): at each step, each node's
    /// share of its parent is its base mean over the sum of its siblings'
    /// base means, and a bottom series receives the product of the shares on
    /// its path to the total.
    TopDownForecastProportions,
    Ols,
    WlsStructural,
    WlsVariance {
        residuals: Vec<Vec<f64>>,
    },
    MinTShrink {
        residuals: Vec<Vec<f64>>,
    },
}

impl ReconciliationMethod {
    /// The name the Python API uses for this method.
    pub fn name(&self) -> &'static str {
        match self {
            Self::BottomUp => "bottom_up",
            Self::TopDownAverageProportions { .. } => "top_down_average_proportions",
            Self::TopDownForecastProportions => "top_down_forecast_proportions",
            Self::Ols => "ols",
            Self::WlsStructural => "wls_struct",
            Self::WlsVariance { .. } => "wls_var",
            Self::MinTShrink { .. } => "mint_shrink",
        }
    }
}

/// Coherent draws, in the layout and draw order of the base draws.
#[derive(Debug, Clone, PartialEq)]
pub struct ReconciledForecast {
    pub draws: NodeDraws,
    pub method: &'static str,
    /// The Schafer-Strimmer intensity MinT shrinkage used, in `[0, 1]`.
    pub shrinkage_intensity: Option<f64>,
}

/// How each draw's bottom values are formed.
enum Mapping {
    /// `b = y_b - K (y_a - A y_b)`, with `K` (`m x r`, row-major) absent for
    /// bottom-up.
    Projection(Option<Vec<f64>>),
    /// `b = p[step] * y_total`, with `p` step-major `steps x m`.
    TopDown { total: usize, proportions: Vec<f64> },
}

/// Reconcile every joint draw of `base` under `method`.
///
/// Draw `(c, d)` of the result is the reconciliation of draw `(c, d)` of the
/// base, at every step. Bottom values are formed first and every aggregate is
/// then the sum of its bottom series, so each reconciled draw is coherent up
/// to the rounding of that sum.
pub fn reconcile(
    hierarchy: &Hierarchy,
    base: &NodeDraws,
    method: &ReconciliationMethod,
) -> Result<ReconciledForecast, ReconciliationError> {
    let (_, _, nodes, steps) = base.shape();
    if nodes != hierarchy.node_count() {
        return Err(invalid_input(format!(
            "draws have {nodes} nodes on their node axis; the hierarchy has {}",
            hierarchy.node_count()
        )));
    }
    let mut shrinkage_intensity = None;
    let mapping = match method {
        ReconciliationMethod::BottomUp => Mapping::Projection(None),
        ReconciliationMethod::Ols => {
            let weights = vec![1.0; nodes];
            Mapping::Projection(Some(projection_gain(hierarchy, &weights, None)?))
        }
        ReconciliationMethod::WlsStructural => {
            let weights: Vec<f64> = (0..nodes)
                .map(|node| match hierarchy.supports.get(node) {
                    Some(support) => support.len() as f64,
                    None => 1.0,
                })
                .collect();
            Mapping::Projection(Some(projection_gain(hierarchy, &weights, None)?))
        }
        ReconciliationMethod::WlsVariance { residuals } => {
            let residuals = Residuals::new(residuals, hierarchy, 1, method.name())?;
            Mapping::Projection(Some(projection_gain(
                hierarchy,
                &residuals.mean_squares,
                None,
            )?))
        }
        ReconciliationMethod::MinTShrink { residuals } => {
            let residuals = Residuals::new(residuals, hierarchy, 2, method.name())?;
            let lambda = residuals.shrinkage_intensity();
            shrinkage_intensity = Some(lambda);
            let weights: Vec<f64> = residuals
                .mean_squares
                .iter()
                .map(|variance| lambda * variance)
                .collect();
            let low_rank = (lambda < 1.0).then_some((&residuals, 1.0 - lambda));
            Mapping::Projection(Some(projection_gain(hierarchy, &weights, low_rank)?))
        }
        ReconciliationMethod::TopDownAverageProportions { history } => {
            let total = require_total(hierarchy, method.name())?;
            let proportions = average_proportions(history, hierarchy)?;
            Mapping::TopDown {
                total,
                proportions: proportions.repeat(steps),
            }
        }
        ReconciliationMethod::TopDownForecastProportions => {
            let total = require_total(hierarchy, method.name())?;
            let proportions = forecast_proportions(hierarchy, base)?;
            Mapping::TopDown { total, proportions }
        }
    };
    let draws = apply(hierarchy, base, &mapping)?;
    Ok(ReconciledForecast {
        draws,
        method: method.name(),
        shrinkage_intensity,
    })
}

fn require_total(hierarchy: &Hierarchy, method: &str) -> Result<usize, ReconciliationError> {
    hierarchy.total().ok_or_else(|| {
        ReconciliationError::UnsupportedMethod(format!(
            "{method} splits the total forecast, and no node of this hierarchy sums every \
             bottom series"
        ))
    })
}

fn apply(
    hierarchy: &Hierarchy,
    base: &NodeDraws,
    mapping: &Mapping,
) -> Result<NodeDraws, ReconciliationError> {
    let (chains, draws, nodes, steps) = base.shape();
    let aggregates = hierarchy.aggregate_count();
    let bottoms = hierarchy.bottom_count();
    let stride = nodes * steps;
    let mut values = vec![0.0; base.values.len()];
    values
        .par_chunks_mut(stride)
        .zip(base.values.par_chunks(stride))
        .try_for_each(|(output, input)| {
            let mut incoherence = vec![0.0; aggregates];
            let mut bottom = vec![0.0; bottoms];
            for step in 0..steps {
                let value = |node: usize| input[node * steps + step];
                match mapping {
                    Mapping::Projection(gain) => {
                        for (j, target) in bottom.iter_mut().enumerate() {
                            *target = value(aggregates + j);
                        }
                        if let Some(gain) = gain {
                            for (i, target) in incoherence.iter_mut().enumerate() {
                                *target = value(i)
                                    - hierarchy.supports[i]
                                        .iter()
                                        .map(|&j| value(aggregates + j))
                                        .sum::<f64>();
                            }
                            for (j, target) in bottom.iter_mut().enumerate() {
                                let row = &gain[j * aggregates..(j + 1) * aggregates];
                                *target -= row
                                    .iter()
                                    .zip(&incoherence)
                                    .map(|(k, d)| k * d)
                                    .sum::<f64>();
                            }
                        }
                    }
                    Mapping::TopDown { total, proportions } => {
                        let shares = &proportions[step * bottoms..(step + 1) * bottoms];
                        let total = value(*total);
                        for (target, share) in bottom.iter_mut().zip(shares) {
                            *target = share * total;
                        }
                    }
                }
                for (i, support) in hierarchy.supports.iter().enumerate() {
                    output[i * steps + step] = support.iter().map(|&j| bottom[j]).sum();
                }
                for (j, &value) in bottom.iter().enumerate() {
                    output[(aggregates + j) * steps + step] = value;
                }
            }
            if output.iter().all(|value| value.is_finite()) {
                Ok(())
            } else {
                Err(ReconciliationError::NumericalFailure(
                    "a reconciled draw is not finite".into(),
                ))
            }
        })?;
    NodeDraws::new(values, chains, draws, nodes, steps)
}

/// Validated residuals, stored node-major and divided by the largest absolute
/// residual, with their uncentred mean squares on that scale.
struct Residuals {
    /// `columns[node * times + t]`.
    columns: Vec<f64>,
    times: usize,
    nodes: usize,
    mean_squares: Vec<f64>,
}

impl Residuals {
    fn new(
        rows: &[Vec<f64>],
        hierarchy: &Hierarchy,
        min_rows: usize,
        method: &str,
    ) -> Result<Self, ReconciliationError> {
        let nodes = hierarchy.node_count();
        let times = rows.len();
        if times < min_rows {
            return Err(invalid_input(format!(
                "{method} needs residuals with at least {min_rows} row{} (time points); got {times}",
                if min_rows == 1 { "" } else { "s" }
            )));
        }
        if let Some((index, row)) = rows.iter().enumerate().find(|(_, row)| row.len() != nodes) {
            return Err(invalid_input(format!(
                "residuals must have one column per node ({nodes}); row {index} has {}",
                row.len()
            )));
        }
        checked_value_count("residuals", &[times, nodes], MAX_MATERIALIZED_VALUES)?;
        let mut columns = vec![0.0; nodes * times];
        for (t, row) in rows.iter().enumerate() {
            for (node, &value) in row.iter().enumerate() {
                if !value.is_finite() {
                    return Err(invalid_input(format!(
                        "residuals[{t}, {node}] (node {:?}) is not finite",
                        hierarchy.names[node]
                    )));
                }
                columns[node * times + t] = value;
            }
        }
        // A common scale cancels from the projection and from the shrinkage
        // intensity. Dividing by the largest residual first keeps squares of
        // very large or very small residuals representable.
        let largest = columns.iter().fold(0.0, |max: f64, e| max.max(e.abs()));
        if largest > 0.0 {
            for value in &mut columns {
                *value /= largest;
            }
        }
        let mean_squares: Vec<f64> = columns
            .chunks(times)
            .map(|column| column.iter().map(|e| e * e).sum::<f64>() / times as f64)
            .collect();
        for (node, &mean_square) in mean_squares.iter().enumerate() {
            if mean_square <= 0.0 {
                return Err(invalid_input(format!(
                    "the residuals of node {:?} have mean square 0 relative to the largest \
                     residual; {method} needs every node's residuals to vary",
                    hierarchy.names[node]
                )));
            }
        }
        Ok(Self {
            columns,
            times,
            nodes,
            mean_squares,
        })
    }

    fn column(&self, node: usize) -> &[f64] {
        &self.columns[node * self.times..(node + 1) * self.times]
    }

    /// The Schafer-Strimmer intensity for shrinking the residual covariance
    /// towards its diagonal, as the MinT authors' `hts` and `fable` packages
    /// compute it: with `x` the residuals divided by their root mean square
    /// and `w_t = x_ti x_tj`, the sample correlation is `r_ij = mean(w)`, its
    /// variance is estimated by `var(w) / T`, and
    /// `lambda = sum_{i != j} var(r_ij) / sum_{i != j} r_ij^2`, clipped to
    /// `[0, 1]`. With no off-diagonal correlation at all it is 1.
    fn shrinkage_intensity(&self) -> f64 {
        let times = self.times as f64;
        let scaled: Vec<f64> = (0..self.nodes)
            .flat_map(|node| {
                let scale = self.mean_squares[node].sqrt();
                self.column(node).iter().map(move |e| e / scale)
            })
            .collect();
        let column = |node: usize| &scaled[node * self.times..(node + 1) * self.times];
        let (variance, squared) = (0..self.nodes)
            .into_par_iter()
            .map(|i| {
                let (mut variance, mut squared) = (0.0, 0.0);
                for j in i + 1..self.nodes {
                    let (mut sum, mut sum_of_squares) = (0.0, 0.0);
                    for (a, b) in column(i).iter().zip(column(j)) {
                        let product = a * b;
                        sum += product;
                        sum_of_squares += product * product;
                    }
                    variance += (sum_of_squares - sum * sum / times) / (times * (times - 1.0));
                    let correlation = sum / times;
                    squared += correlation * correlation;
                }
                (variance, squared)
            })
            .collect::<Vec<_>>()
            .into_iter()
            .fold((0.0, 0.0), |(v, s), (dv, ds)| (v + dv, s + ds));
        if squared > 0.0 {
            (variance / squared).clamp(0.0, 1.0)
        } else {
            1.0
        }
    }
}

/// The gain `K = J W U (U' W U)^-1`, `m x r` row-major, for
/// `W = diag(weights) + c E'E / T` with `(E, c)` the optional low-rank part.
///
/// With `A` the aggregate rows of `S` and `U' = [I | -A]`:
/// `J W U = -D_b A' + (c/T) E_b' (E U)` and
/// `U' W U = D_a + A D_b A' + (c/T) (E U)' (E U)`, where `E U` is each
/// residual row's incoherence. Nothing `n x n` is formed.
fn projection_gain(
    hierarchy: &Hierarchy,
    weights: &[f64],
    low_rank: Option<(&Residuals, f64)>,
) -> Result<Vec<f64>, ReconciliationError> {
    let aggregates = hierarchy.aggregate_count();
    let bottoms = hierarchy.bottom_count();
    checked_value_count(
        "reconciliation gain",
        &[bottoms, aggregates],
        MAX_MATERIALIZED_VALUES,
    )?;
    checked_value_count(
        "reconciliation system",
        &[aggregates, aggregates],
        MAX_MATERIALIZED_VALUES,
    )?;
    // Aggregates containing each bottom series.
    let mut containing: Vec<Vec<usize>> = vec![Vec::new(); bottoms];
    for (i, support) in hierarchy.supports.iter().enumerate() {
        for &j in support {
            containing[j].push(i);
        }
    }
    let mut system = vec![0.0; aggregates * aggregates];
    let mut cross = vec![0.0; bottoms * aggregates];
    for i in 0..aggregates {
        system[i * aggregates + i] += weights[i];
    }
    for (j, rows) in containing.iter().enumerate() {
        let weight = weights[aggregates + j];
        for &i in rows {
            cross[j * aggregates + i] -= weight;
            for &k in rows {
                system[i * aggregates + k] += weight;
            }
        }
    }
    if let Some((residuals, coefficient)) = low_rank {
        let times = residuals.times;
        checked_value_count(
            "residual incoherence",
            &[times, aggregates],
            MAX_MATERIALIZED_VALUES,
        )?;
        let scale = coefficient / times as f64;
        // incoherence[i * times + t] = e[t, i] - sum over the support of i.
        let incoherence: Vec<f64> = (0..aggregates)
            .into_par_iter()
            .flat_map_iter(|i| {
                let support = &hierarchy.supports[i];
                (0..times).map(move |t| {
                    residuals.column(i)[t]
                        - support
                            .iter()
                            .map(|&j| residuals.column(aggregates + j)[t])
                            .sum::<f64>()
                })
            })
            .collect();
        let row = |i: usize| &incoherence[i * times..(i + 1) * times];
        let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
        system
            .par_chunks_mut(aggregates)
            .enumerate()
            .for_each(|(i, target)| {
                for (k, value) in target.iter_mut().enumerate() {
                    *value += scale * dot(row(i), row(k));
                }
            });
        cross
            .par_chunks_mut(aggregates)
            .enumerate()
            .for_each(|(j, target)| {
                let bottom = residuals.column(aggregates + j);
                for (i, value) in target.iter_mut().enumerate() {
                    *value += scale * dot(bottom, row(i));
                }
            });
    }
    let factor = cholesky(&system, aggregates).map_err(|()| {
        ReconciliationError::NumericalFailure(
            "the reconciliation system U'WU is not positive definite; the weights W are \
             singular or badly scaled"
                .into(),
        )
    })?;
    let gain: Vec<f64> = cross
        .par_chunks(aggregates)
        .flat_map_iter(|row| {
            let half = solve_lower(&factor, aggregates, row);
            solve_lower_transpose(&factor, aggregates, &half)
        })
        .collect();
    if gain.iter().all(|value| value.is_finite()) {
        Ok(gain)
    } else {
        Err(ReconciliationError::NumericalFailure(
            "the reconciliation gain is not finite".into(),
        ))
    }
}

fn average_proportions(
    history: &[Vec<f64>],
    hierarchy: &Hierarchy,
) -> Result<Vec<f64>, ReconciliationError> {
    let bottoms = hierarchy.bottom_count();
    if history.is_empty() {
        return Err(invalid_input(
            "top_down_average_proportions needs history with at least one row (time point)",
        ));
    }
    let mut proportions = vec![0.0; bottoms];
    for (t, row) in history.iter().enumerate() {
        if row.len() != bottoms {
            return Err(invalid_input(format!(
                "history must have one column per bottom series ({bottoms}); row {t} has {}",
                row.len()
            )));
        }
        if let Some(j) = row
            .iter()
            .position(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(invalid_input(format!(
                "history[{t}, {j}] (bottom series {:?}) is {}; historical proportions need \
                 finite, non-negative history",
                hierarchy.bottom_names()[j],
                row[j]
            )));
        }
        let total: f64 = row.iter().sum();
        if !total.is_finite() || total <= 0.0 {
            return Err(invalid_input(format!(
                "history row {t} sums to {total}; historical proportions need every period's \
                 total to be finite and positive"
            )));
        }
        for (proportion, value) in proportions.iter_mut().zip(row) {
            *proportion += value / total;
        }
    }
    let periods = history.len() as f64;
    Ok(proportions.into_iter().map(|sum| sum / periods).collect())
}

/// Step-major `steps x m` proportions from the base means.
fn forecast_proportions(
    hierarchy: &Hierarchy,
    base: &NodeDraws,
) -> Result<Vec<f64>, ReconciliationError> {
    let method = "top_down_forecast_proportions";
    let (Some(parents), Some(depths)) = (hierarchy.parents(), hierarchy.depths()) else {
        return Err(ReconciliationError::UnsupportedMethod(format!(
            "{method} needs a nested hierarchy, and these aggregates overlap without nesting"
        )));
    };
    let (_, _, nodes, steps) = base.shape();
    let aggregates = hierarchy.aggregate_count();
    let root = require_total(hierarchy, method)?;
    let means = base.means();
    let mut proportions = vec![0.0; steps * hierarchy.bottom_count()];
    for step in 0..steps {
        if let Some(node) = (0..nodes).find(|&node| means[node * steps + step] < 0.0) {
            return Err(invalid_input(format!(
                "{method} needs non-negative base means; node {:?} has mean {} at step {step}",
                hierarchy.names[node],
                means[node * steps + step]
            )));
        }
        // Shares do not depend on the scale of the means. Dividing by the
        // largest first keeps a sum of siblings from overflowing.
        let largest = (0..nodes)
            .map(|node| means[node * steps + step])
            .fold(0.0, f64::max);
        let scale = if largest > 0.0 { largest } else { 1.0 };
        let mean = |node: usize| means[node * steps + step] / scale;
        let mut child_sums = vec![0.0; nodes];
        for (node, parent) in parents.iter().enumerate() {
            if let Some(parent) = *parent {
                child_sums[parent] += mean(node);
            }
        }
        for (j, target) in proportions[step * hierarchy.bottom_count()..]
            .iter_mut()
            .take(hierarchy.bottom_count())
            .enumerate()
        {
            let mut share = 1.0;
            let mut node = aggregates + j;
            while let Some(parent) = parents[node] {
                if child_sums[parent] <= 0.0 {
                    return Err(invalid_input(format!(
                        "{method} cannot split node {:?} at step {step}: the base means of \
                         its children sum to zero",
                        hierarchy.names[parent]
                    )));
                }
                share *= mean(node) / child_sums[parent];
                node = parent;
            }
            debug_assert_eq!(node, root);
            debug_assert_eq!(depths[node], 0);
            *target = share;
        }
    }
    Ok(proportions)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn labels(paths: &[&[&str]]) -> Vec<Vec<String>> {
        paths
            .iter()
            .map(|path| path.iter().map(|label| label.to_string()).collect())
            .collect()
    }

    /// total -> {A, B}; A -> {a1, a2}; B -> {b1}.
    fn small() -> Hierarchy {
        Hierarchy::from_labels(
            &labels(&[&["A", "a1"], &["A", "a2"], &["B", "b1"]]),
            "total",
            "/",
        )
        .unwrap()
    }

    #[test]
    fn labels_build_nodes_in_documented_order() {
        let hierarchy = Hierarchy::from_labels(
            &labels(&[
                &["BU2", "IDN1", "c1"],
                &["BU1", "IDN3", "c2"],
                &["BU2", "IDN1", "c3"],
                &["BU1", "IDN1", "c4"],
            ]),
            "total",
            "/",
        )
        .unwrap();
        assert_eq!(
            hierarchy.node_names(),
            [
                "total",
                "BU2",
                "BU1",
                "BU2/IDN1",
                "BU1/IDN3",
                "BU1/IDN1",
                "BU2/IDN1/c1",
                "BU1/IDN3/c2",
                "BU2/IDN1/c3",
                "BU1/IDN1/c4",
            ]
        );
        assert_eq!(hierarchy.support(0), [0, 1, 2, 3]);
        assert_eq!(hierarchy.support(1), [0, 2]);
        assert_eq!(hierarchy.support(3), [0, 2]);
        assert_eq!(hierarchy.support(5), [3]);
        assert_eq!(hierarchy.depths().unwrap(), [0, 1, 1, 2, 2, 2, 3, 3, 3, 3]);
        let parents = hierarchy.parents().unwrap();
        assert_eq!(parents[0], None);
        assert_eq!(parents[3], Some(1));
        assert_eq!(parents[9], Some(5));
        assert_eq!(hierarchy.total(), Some(0));
    }

    #[test]
    fn summing_matrix_round_trips_and_detects_grouping() {
        let hierarchy = small();
        let rows = hierarchy.summing_matrix().unwrap();
        assert_eq!(
            rows,
            vec![
                vec![1.0, 1.0, 1.0],
                vec![1.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
            ]
        );
        let again = Hierarchy::from_summing_matrix(&rows, None).unwrap();
        assert_eq!(again.parents(), hierarchy.parents());
        assert_eq!(again.node_names()[0], "node_0");
        // A 2 x 2 grid: rows and columns overlap without nesting.
        let grouped = vec![
            vec![1.0, 1.0, 1.0, 1.0],
            vec![1.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 1.0],
            vec![1.0, 0.0, 1.0, 0.0],
            vec![0.0, 1.0, 0.0, 1.0],
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];
        let grouped = Hierarchy::from_summing_matrix(&grouped, None).unwrap();
        assert!(!grouped.is_nested());
        assert_eq!(grouped.total(), Some(0));
    }

    #[test]
    fn malformed_hierarchies_are_refused_with_a_reason() {
        let cases: Vec<(Vec<Vec<f64>>, &str)> = vec![
            (
                vec![vec![1.0, 0.0], vec![0.0, 1.0]],
                "at least one aggregate",
            ),
            (
                vec![vec![1.0, 2.0], vec![1.0, 0.0], vec![0.0, 1.0]],
                "only 0 and 1",
            ),
            (
                vec![vec![1.0, 1.0], vec![0.0, 1.0], vec![1.0, 0.0]],
                "identity",
            ),
            (
                vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]],
                "sums no bottom",
            ),
            (vec![vec![1.0, 1.0], vec![1.0]], "columns"),
            (
                vec![vec![f64::NAN, 1.0], vec![1.0, 0.0], vec![0.0, 1.0]],
                "only 0 and 1",
            ),
        ];
        for (rows, expected) in cases {
            let error = Hierarchy::from_summing_matrix(&rows, None).unwrap_err();
            assert!(
                matches!(error, ReconciliationError::InvalidHierarchy(_)),
                "{error}"
            );
            assert!(error.to_string().contains(expected), "{error}");
        }
        let names = Some(vec!["a".into(), "a".into(), "b".into()]);
        let rows = vec![vec![1.0, 1.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        assert!(Hierarchy::from_summing_matrix(&rows, names)
            .unwrap_err()
            .to_string()
            .contains("\"a\" occurs more than once"));
        for (paths, expected) in [
            (labels(&[&["A", "x"], &["A"]]), "same length"),
            (labels(&[&["A", "x"], &["A", "x"]]), "repeats"),
            (labels(&[&["A", "x/y"]]), "separator"),
            (labels(&[&["A", ""]]), "is empty"),
            (labels(&[&["total", "x"]]), "occurs more than once"),
        ] {
            let error = Hierarchy::from_labels(&paths, "total", "/").unwrap_err();
            assert!(error.to_string().contains(expected), "{error}");
        }
    }

    fn draws_from(vectors: &[Vec<f64>]) -> NodeDraws {
        let nodes = vectors[0].len();
        NodeDraws::new(vectors.concat(), 1, vectors.len(), nodes, 1).unwrap()
    }

    #[test]
    fn bottom_up_sums_the_bottom_draws() {
        let hierarchy = small();
        let base = draws_from(&[vec![100.0, 7.0, 9.0, 1.0, 2.0, 3.0]]);
        let result = reconcile(&hierarchy, &base, &ReconciliationMethod::BottomUp).unwrap();
        assert_eq!(result.draws.values(), [6.0, 3.0, 3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn projections_leave_coherent_draws_unchanged() {
        let hierarchy = small();
        let coherent = draws_from(&[vec![6.5, 3.25, 3.25, 1.0, 2.25, 3.25]]);
        let residuals: Vec<Vec<f64>> = (0..8)
            .map(|t| {
                let t = t as f64;
                vec![t.sin(), (1.3 * t).cos(), 0.5 + t, (0.7 * t).sin(), -t, 1.0]
            })
            .collect();
        for method in [
            ReconciliationMethod::BottomUp,
            ReconciliationMethod::Ols,
            ReconciliationMethod::WlsStructural,
            ReconciliationMethod::WlsVariance {
                residuals: residuals.clone(),
            },
            ReconciliationMethod::MinTShrink {
                residuals: residuals.clone(),
            },
        ] {
            let result = reconcile(&hierarchy, &coherent, &method).unwrap();
            assert_eq!(
                result.draws.values(),
                coherent.values(),
                "{}",
                method.name()
            );
        }
    }

    #[test]
    fn ols_matches_the_textbook_projection_on_a_small_case() {
        // S = [[1,1],[1,0],[0,1]]: S'S = [[2,1],[1,2]], (S'S)^-1 S' =
        // [[1,2,-1],[1,-1,2]] / 3.
        let rows = vec![vec![1.0, 1.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        let hierarchy = Hierarchy::from_summing_matrix(&rows, None).unwrap();
        let y = [10.0, 3.0, 4.0];
        let base = draws_from(&[y.to_vec()]);
        let result = reconcile(&hierarchy, &base, &ReconciliationMethod::Ols).unwrap();
        let b1 = (y[0] + 2.0 * y[1] - y[2]) / 3.0;
        let b2 = (y[0] - y[1] + 2.0 * y[2]) / 3.0;
        let expected = [b1 + b2, b1, b2];
        for (actual, expected) in result.draws.values().iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-12, "{actual} {expected}");
        }
    }

    #[test]
    fn shrinkage_intensity_matches_a_direct_evaluation() {
        let hierarchy = small();
        let rows: Vec<Vec<f64>> = (0..12)
            .map(|t| {
                let t = t as f64;
                let common = (0.9 * t).sin();
                vec![
                    common + 0.3 * t.cos(),
                    common + 0.1 * (2.0 * t).sin(),
                    0.2 * common + (1.7 * t).cos(),
                    0.5 * common + 0.4 * (3.1 * t).sin(),
                    (0.4 * t).cos(),
                    common - 0.2 * (1.1 * t).cos(),
                ]
            })
            .collect();
        let residuals = Residuals::new(&rows, &hierarchy, 2, "mint_shrink").unwrap();
        let lambda = residuals.shrinkage_intensity();
        // Direct transcription of hts::shrink.estim.
        let t = rows.len() as f64;
        let n = rows[0].len();
        let sd: Vec<f64> = (0..n)
            .map(|i| (rows.iter().map(|r| r[i] * r[i]).sum::<f64>() / t).sqrt())
            .collect();
        let (mut numerator, mut denominator) = (0.0, 0.0);
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let w: Vec<f64> = rows.iter().map(|r| r[i] / sd[i] * r[j] / sd[j]).collect();
                let mean = w.iter().sum::<f64>() / t;
                let var = w.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (t - 1.0) / t;
                numerator += var;
                denominator += mean * mean;
            }
        }
        let expected = (numerator / denominator).clamp(0.0, 1.0);
        assert!(expected > 0.0 && expected < 1.0, "{expected}");
        assert!((lambda - expected).abs() < 1e-12, "{lambda} {expected}");
    }

    #[test]
    fn input_errors_name_the_input() {
        let hierarchy = small();
        let base = draws_from(&[vec![6.0, 3.0, 3.0, 1.0, 2.0, 3.0]]);
        let wrong_nodes = draws_from(&[vec![1.0, 2.0]]);
        let error = reconcile(&hierarchy, &wrong_nodes, &ReconciliationMethod::Ols).unwrap_err();
        assert!(error.to_string().contains("draws have 2 nodes"), "{error}");
        assert!(NodeDraws::new(vec![1.0, f64::NAN], 1, 1, 2, 1)
            .unwrap_err()
            .to_string()
            .contains("draws[0, 0, 1, 0] is not finite"));
        let short = ReconciliationMethod::MinTShrink {
            residuals: vec![vec![1.0; 6]],
        };
        assert!(reconcile(&hierarchy, &base, &short)
            .unwrap_err()
            .to_string()
            .contains("at least 2 rows"));
        let ragged = ReconciliationMethod::WlsVariance {
            residuals: vec![vec![1.0; 6], vec![1.0; 5]],
        };
        assert!(reconcile(&hierarchy, &base, &ragged)
            .unwrap_err()
            .to_string()
            .contains("row 1 has 5"));
        let mut rows = vec![vec![1.0; 6], vec![-1.0; 6]];
        rows[1][4] = f64::INFINITY;
        let infinite = ReconciliationMethod::WlsVariance { residuals: rows };
        assert!(reconcile(&hierarchy, &base, &infinite)
            .unwrap_err()
            .to_string()
            .contains("residuals[1, 4] (node \"A/a2\") is not finite"));
        let mut rows = vec![vec![1.0; 6], vec![-1.0; 6]];
        rows[0][2] = 0.0;
        rows[1][2] = 0.0;
        let silent = ReconciliationMethod::MinTShrink { residuals: rows };
        assert!(reconcile(&hierarchy, &base, &silent)
            .unwrap_err()
            .to_string()
            .contains("node \"B\""));
        let negative = ReconciliationMethod::TopDownAverageProportions {
            history: vec![vec![1.0, -1.0, 2.0]],
        };
        assert!(reconcile(&hierarchy, &base, &negative)
            .unwrap_err()
            .to_string()
            .contains("history[0, 1]"));
    }

    #[test]
    fn top_down_splits_the_total_draw() {
        let hierarchy = small();
        let history = vec![vec![1.0, 1.0, 2.0], vec![2.0, 0.0, 2.0]];
        // Shares: a1 (0.25 + 0.5) / 2, a2 0.125, b1 0.5.
        let base = draws_from(&[vec![8.0, 1.0, 1.0, 1.0, 1.0, 1.0]]);
        let average = reconcile(
            &hierarchy,
            &base,
            &ReconciliationMethod::TopDownAverageProportions { history },
        )
        .unwrap();
        assert_eq!(average.draws.values(), [8.0, 4.0, 4.0, 3.0, 1.0, 4.0]);
        // Means: A 3, B 1 -> A gets 3/4; within A, a1 1 and a2 3.
        let base = draws_from(&[
            vec![8.0, 2.0, 0.0, 0.0, 2.0, 5.0],
            vec![8.0, 4.0, 2.0, 2.0, 4.0, 5.0],
        ]);
        let forecast = reconcile(
            &hierarchy,
            &base,
            &ReconciliationMethod::TopDownForecastProportions,
        )
        .unwrap();
        assert_eq!(forecast.draws.values()[..6], [8.0, 6.0, 2.0, 1.5, 4.5, 2.0]);
    }

    #[test]
    fn forecast_proportions_survive_huge_means_and_refuse_a_negative_total() {
        let pair = Hierarchy::from_labels(&labels(&[&["a"], &["b"]]), "total", "/").unwrap();
        // The siblings' sum overflows unless the means are rescaled first.
        let base = draws_from(&[vec![1.0, 1e308, 1e308]]);
        let result = reconcile(
            &pair,
            &base,
            &ReconciliationMethod::TopDownForecastProportions,
        )
        .unwrap();
        assert_eq!(result.draws.values(), [1.0, 0.5, 0.5]);
        let negative_total = draws_from(&[vec![-1.0, 1.0, 1.0]]);
        let error = reconcile(
            &pair,
            &negative_total,
            &ReconciliationMethod::TopDownForecastProportions,
        )
        .unwrap_err();
        assert!(
            error.to_string().contains("node \"total\" has mean -1"),
            "{error}"
        );
    }

    #[test]
    fn a_common_residual_scale_does_not_change_the_result() {
        let rows = vec![vec![1.0, 1.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        let hierarchy = Hierarchy::from_summing_matrix(&rows, None).unwrap();
        let base = draws_from(&[vec![10.0, 3.0, 4.0]]);
        let residuals = vec![
            vec![1.0, 2.0, -1.0],
            vec![-1.0, 3.0, 2.0],
            vec![2.0, -1.0, 1.0],
        ];
        let scaled = |factor: f64| -> Vec<Vec<f64>> {
            residuals
                .iter()
                .map(|row| row.iter().map(|e| e * factor).collect())
                .collect()
        };
        for method in [
            |residuals| ReconciliationMethod::WlsVariance { residuals },
            |residuals| ReconciliationMethod::MinTShrink { residuals },
        ] {
            let reference = reconcile(&hierarchy, &base, &method(residuals.clone())).unwrap();
            for factor in [1e-200, 1e-160, 1e154, 1e300] {
                let result = reconcile(&hierarchy, &base, &method(scaled(factor))).unwrap();
                for (a, b) in result.draws.values().iter().zip(reference.draws.values()) {
                    assert!((a - b).abs() <= 1e-12 * b.abs(), "{factor}: {a} {b}");
                }
            }
        }
    }

    #[test]
    fn top_down_needs_a_total_and_forecast_proportions_need_nesting() {
        let rows = vec![
            vec![1.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 1.0],
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];
        let no_total = Hierarchy::from_summing_matrix(&rows, None).unwrap();
        let base = draws_from(&[vec![1.0; 6]]);
        let error = reconcile(
            &no_total,
            &base,
            &ReconciliationMethod::TopDownForecastProportions,
        )
        .unwrap_err();
        assert!(matches!(error, ReconciliationError::UnsupportedMethod(_)));
        let grouped = Hierarchy::from_summing_matrix(
            &[
                vec![1.0, 1.0, 1.0, 1.0],
                vec![1.0, 1.0, 0.0, 0.0],
                vec![1.0, 0.0, 1.0, 0.0],
                vec![1.0, 0.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0, 0.0],
                vec![0.0, 0.0, 1.0, 0.0],
                vec![0.0, 0.0, 0.0, 1.0],
            ],
            None,
        )
        .unwrap();
        let base = draws_from(&[vec![1.0; 7]]);
        let error = reconcile(
            &grouped,
            &base,
            &ReconciliationMethod::TopDownForecastProportions,
        )
        .unwrap_err();
        assert!(error.to_string().contains("nested"), "{error}");
        // Every other method accepts the grouped structure.
        assert!(reconcile(&grouped, &base, &ReconciliationMethod::WlsStructural).is_ok());
    }
}
