//! Decisions from predictive draws.
//!
//! Given draws of an uncertain outcome `y` - a posterior predictive forecast,
//! or any other set of joint draws - and a loss `L(a, y)` for taking decision
//! `a`, this module estimates the expected loss `E[L(a, y)]` as the average
//! over the draws, evaluates it over a grid of candidate decisions, and finds
//! the decision that minimises it.
//!
//! For three built-in losses the optimum is a known summary of the draws,
//! used when asked for:
//!
//! - asymmetric linear (pinball) loss, `under * (y - a)` when the outcome
//!   exceeds the decision and `over * (a - y)` otherwise: the
//!   `under / (under + over)` quantile of the draws, by
//!   [`sorted_quantile`], the rule behind every forecast interval;
//! - squared loss `(y - a)^2`: the mean of the draws;
//! - absolute loss `|y - a|`: the median of the draws.
//!
//! A piecewise-linear loss, or any [`LossFunction`], is minimised over the
//! candidate grid only, so its answer is no finer than that grid.
//!
//! The quantile rule interpolates between order statistics, so for the
//! pinball loss it can sit up to one gap between adjacent draws away from the
//! exact minimiser of the average loss, which costs at most
//! `(under + over) / n` times that gap for `n` draws. The mean and the median
//! are exact minimisers.
//!
//! Every estimate here is a Monte Carlo average, and the draws a sampler
//! returns are autocorrelated within a chain. The Monte Carlo standard error
//! of an expected loss divides the standard deviation of the per-draw losses
//! by the square root of their effective sample size, estimated from split
//! chains exactly as the `mcse_mean` column of a fit summary is. How stable
//! the optimal decision is comes from batch means: the optimal decision is
//! recomputed on contiguous blocks of each chain, and the spread of those
//! block decisions, divided by the square root of their number, estimates its
//! Monte Carlo standard error. That estimate is withheld where it does not
//! hold: for a grid search and for a quantile at 0 or 1.

use crate::diagnostics::{ess_raw, scaled_moments};
use crate::forecast_common::{checked_value_count, sorted_quantile, MAX_MATERIALIZED_VALUES};
use rayon::prelude::*;
use std::error::Error;
use std::fmt;

/// Number of evenly spaced candidates in the default grid, which runs from the
/// smallest draw to the largest.
pub const DEFAULT_CANDIDATE_COUNT: usize = 201;

/// Fewest blocks the batch-means standard error of a decision is computed
/// from, when chains are long enough: with two chains each is cut into four
/// blocks, with four or more chains into halves. Fewer blocks would leave the
/// standard error with too few degrees of freedom to mean much.
pub const MIN_DECISION_BATCHES: usize = 8;

/// A refused input or a failed evaluation. Every message names the argument
/// at fault.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecisionError {
    /// The draws are empty, ragged or not finite, or `steps` does not select
    /// from them.
    InvalidDraws(String),
    /// A loss's parameters are out of range.
    InvalidLoss(String),
    /// The candidate decisions are empty, not finite or not increasing.
    InvalidCandidates(String),
    /// A decision is not finite.
    InvalidDecision(String),
    /// A loss or a total left the finite floating-point range. No draw was
    /// dropped or clipped.
    NumericalFailure(String),
    /// A user-supplied loss function failed or returned the wrong shape.
    LossFunction(String),
}

impl fmt::Display for DecisionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDraws(message)
            | Self::InvalidLoss(message)
            | Self::InvalidCandidates(message)
            | Self::InvalidDecision(message)
            | Self::NumericalFailure(message)
            | Self::LossFunction(message) => f.write_str(message),
        }
    }
}

impl Error for DecisionError {}

/// Draws of one outcome, chain by chain, every chain the same length and
/// every value finite.
///
/// Draws with no chain structure are one chain. Their order is then taken to
/// be the order they were drawn in, which matters only to the Monte Carlo
/// standard errors: independent draws in any order give the right answer.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionDraws {
    chains: Vec<Vec<f64>>,
}

impl DecisionDraws {
    /// Draws indexed `[chain][draw]`.
    pub fn new(chains: Vec<Vec<f64>>) -> Result<Self, DecisionError> {
        let per_chain = chains.first().map_or(0, Vec::len);
        if per_chain == 0 {
            return Err(DecisionError::InvalidDraws(
                "draws must hold at least one draw".into(),
            ));
        }
        if chains.iter().any(|chain| chain.len() != per_chain) {
            return Err(DecisionError::InvalidDraws(
                "draws must hold the same number of draws in every chain".into(),
            ));
        }
        for (chain_index, chain) in chains.iter().enumerate() {
            if let Some((draw, value)) = chain.iter().enumerate().find(|(_, v)| !v.is_finite()) {
                return Err(DecisionError::InvalidDraws(format!(
                    "draws must be finite; chain {chain_index}, draw {draw} is {value}"
                )));
            }
        }
        Ok(Self { chains })
    }

    /// The total over `steps` of every path, from paths indexed
    /// `[chain][draw][step]`.
    ///
    /// Each total is summed within one path, so it keeps the dependence
    /// between steps that a sum of per-step quantiles would lose. Steps are
    /// zero-based and may be given in any order; they are summed in
    /// ascending order, so `steps = 0..k` gives exactly the running total at
    /// step `k - 1` that [`crate::forecast_common::cumulative_paths`]
    /// reports.
    pub fn from_path_totals(
        paths: &[Vec<Vec<f64>>],
        steps: &[usize],
    ) -> Result<Self, DecisionError> {
        let horizon = paths
            .first()
            .and_then(|chain| chain.first())
            .map_or(0, Vec::len);
        if horizon == 0 || paths.iter().any(Vec::is_empty) {
            return Err(DecisionError::InvalidDraws(
                "draws must hold at least one draw and one step".into(),
            ));
        }
        if paths.iter().flatten().any(|path| path.len() != horizon) {
            return Err(DecisionError::InvalidDraws(
                "draws must have the same number of steps in every path".into(),
            ));
        }
        let steps = validate_steps(steps, horizon)?;
        let chains = paths
            .iter()
            .enumerate()
            .map(|(chain_index, chain)| {
                chain
                    .iter()
                    .enumerate()
                    .map(|(draw, path)| {
                        // Every step is checked, chosen or not, as `validate_paths`
                        // checks a whole forecast.
                        if let Some((step, value)) =
                            path.iter().enumerate().find(|(_, v)| !v.is_finite())
                        {
                            return Err(DecisionError::InvalidDraws(format!(
                                "draws must be finite; chain {chain_index}, draw {draw}, \
                                 step {step} is {value}"
                            )));
                        }
                        let total = steps.iter().fold(0.0, |total, &step| total + path[step]);
                        if total.is_finite() {
                            Ok(total)
                        } else {
                            Err(DecisionError::NumericalFailure(format!(
                                "the total over steps of chain {chain_index}, draw {draw} \
                                 is not finite"
                            )))
                        }
                    })
                    .collect()
            })
            .collect::<Result<Vec<Vec<f64>>, _>>()?;
        Self::new(chains)
    }

    /// The draws, indexed `[chain][draw]`.
    pub fn chains(&self) -> &[Vec<f64>] {
        &self.chains
    }

    pub fn chain_count(&self) -> usize {
        self.chains.len()
    }

    pub fn draws_per_chain(&self) -> usize {
        self.chains[0].len()
    }

    /// Number of draws in all.
    pub fn len(&self) -> usize {
        self.chain_count() * self.draws_per_chain()
    }

    /// Always false: construction refuses empty draws.
    pub fn is_empty(&self) -> bool {
        false
    }

    fn values(&self) -> impl Iterator<Item = f64> + Clone + '_ {
        self.chains.iter().flatten().copied()
    }
}

/// Distinct in-range steps, sorted.
fn validate_steps(steps: &[usize], horizon: usize) -> Result<Vec<usize>, DecisionError> {
    if steps.is_empty() {
        return Err(DecisionError::InvalidDraws(
            "steps must name at least one step".into(),
        ));
    }
    if let Some(step) = steps.iter().find(|&&step| step >= horizon) {
        return Err(DecisionError::InvalidDraws(format!(
            "steps holds step {step}, outside a horizon of {horizon} step{} \
             (steps are numbered from 0)",
            if horizon == 1 { "" } else { "s" }
        )));
    }
    let mut sorted = steps.to_vec();
    sorted.sort_unstable();
    if let Some(pair) = sorted.windows(2).find(|pair| pair[0] == pair[1]) {
        return Err(DecisionError::InvalidDraws(format!(
            "steps lists step {} more than once",
            pair[0]
        )));
    }
    Ok(sorted)
}

/// The decision that minimises a loss's average over the draws, when a
/// formula gives it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptimalRule {
    /// The empirical quantile at this probability, by [`sorted_quantile`].
    Quantile(f64),
    Mean,
    Median,
}

impl OptimalRule {
    fn apply(self, draws: impl Iterator<Item = f64> + Clone) -> f64 {
        match self {
            Self::Mean => scaled_moments(|| draws.clone()).0,
            Self::Quantile(probability) => quantile(draws, probability),
            Self::Median => quantile(draws, 0.5),
        }
    }
}

fn quantile(draws: impl Iterator<Item = f64>, probability: f64) -> f64 {
    let mut ordered: Vec<f64> = draws.collect();
    ordered.sort_by(f64::total_cmp);
    sorted_quantile(&ordered, probability)
}

/// A loss of a decision, as a function of the draws of the outcome.
///
/// Implement this for a loss the built-in [`Loss`] does not cover. A loss
/// with no [`OptimalRule`] is minimised over candidate decisions only.
pub trait LossFunction {
    /// The loss of `decision` for every draw, indexed like `draws.chains()`.
    fn losses(&self, decision: f64, draws: &DecisionDraws) -> Result<Vec<Vec<f64>>, DecisionError>;

    /// The formula for the exact minimiser, when there is one.
    fn optimal_rule(&self) -> Option<OptimalRule> {
        None
    }

    /// Refuse parameters out of range before any work is done.
    fn validate(&self) -> Result<(), DecisionError> {
        Ok(())
    }
}

/// A built-in loss of decision `a` when the outcome is `y`.
#[derive(Debug, Clone, PartialEq)]
pub enum Loss {
    /// Pinball loss: `under * (y - a)` when the outcome exceeds the decision,
    /// `over * (a - y)` otherwise. `under` is the cost per unit of falling
    /// short, `over` per unit of overshooting.
    AsymmetricLinear { under: f64, over: f64 },
    /// `(y - a)^2`.
    Squared,
    /// `|y - a|`.
    Absolute,
    /// A piecewise-linear function of the shortfall `y - a`.
    PiecewiseLinear(PiecewiseLinearLoss),
}

impl Loss {
    /// A validated [`Loss::AsymmetricLinear`].
    pub fn asymmetric_linear(under: f64, over: f64) -> Result<Self, DecisionError> {
        let loss = Self::AsymmetricLinear { under, over };
        LossFunction::validate(&loss)?;
        Ok(loss)
    }

    /// The loss of `decision` when the outcome is `outcome`. May overflow to
    /// infinity; [`expected_loss`] and [`optimal_decision`] refuse that.
    pub fn evaluate(&self, decision: f64, outcome: f64) -> f64 {
        let shortfall = outcome - decision;
        match self {
            Self::AsymmetricLinear { under, over } => {
                if outcome > decision {
                    weighted_gap(*under, outcome, decision)
                } else {
                    weighted_gap(*over, decision, outcome)
                }
            }
            Self::Squared => shortfall * shortfall,
            Self::Absolute => shortfall.abs(),
            Self::PiecewiseLinear(loss) => loss.evaluate(shortfall),
        }
    }
}

impl LossFunction for Loss {
    fn losses(&self, decision: f64, draws: &DecisionDraws) -> Result<Vec<Vec<f64>>, DecisionError> {
        Ok(draws
            .chains()
            .par_iter()
            .map(|chain| {
                chain
                    .iter()
                    .map(|&outcome| self.evaluate(decision, outcome))
                    .collect()
            })
            .collect())
    }

    fn optimal_rule(&self) -> Option<OptimalRule> {
        match self {
            Self::AsymmetricLinear { under, over } => {
                // under / (under + over) without overflowing the sum.
                let probability = if *under == 0.0 {
                    0.0
                } else {
                    1.0 / (1.0 + over / under)
                };
                Some(OptimalRule::Quantile(probability))
            }
            Self::Squared => Some(OptimalRule::Mean),
            Self::Absolute => Some(OptimalRule::Median),
            Self::PiecewiseLinear(_) => None,
        }
    }

    fn validate(&self) -> Result<(), DecisionError> {
        match self {
            Self::AsymmetricLinear { under, over } => {
                for (name, value) in [("under", under), ("over", over)] {
                    if !value.is_finite() || *value < 0.0 {
                        return Err(DecisionError::InvalidLoss(format!(
                            "{name} must be finite and nonnegative; got {value}"
                        )));
                    }
                }
                if *under == 0.0 && *over == 0.0 {
                    return Err(DecisionError::InvalidLoss(
                        "under and over cannot both be zero".into(),
                    ));
                }
                Ok(())
            }
            Self::PiecewiseLinear(loss) => {
                PiecewiseLinearLoss::new(loss.shortfall.clone(), loss.loss.clone()).map(|_| ())
            }
            Self::Squared | Self::Absolute => Ok(()),
        }
    }
}

/// `weight * (high - low)` for `high >= low`, scaling before subtracting when
/// the gap alone overflows, and zero for a zero weight, so a representable
/// loss is not lost to its intermediate.
fn weighted_gap(weight: f64, high: f64, low: f64) -> f64 {
    if weight == 0.0 {
        return 0.0;
    }
    let gap = high - low;
    if gap.is_finite() {
        weight * gap
    } else {
        weight * high - weight * low
    }
}

/// A loss that is linear between knots in the shortfall `y - a` - positive
/// when the outcome exceeds the decision, negative when the decision
/// overshoots - and continues along its first and last segments beyond them.
#[derive(Debug, Clone, PartialEq)]
pub struct PiecewiseLinearLoss {
    shortfall: Vec<f64>,
    loss: Vec<f64>,
}

impl PiecewiseLinearLoss {
    /// Knots at strictly increasing `shortfall` values, with the loss at each.
    pub fn new(shortfall: Vec<f64>, loss: Vec<f64>) -> Result<Self, DecisionError> {
        let invalid = |message: String| Err(DecisionError::InvalidLoss(message));
        if shortfall.len() < 2 {
            return invalid(format!(
                "shortfall must hold at least two knots; got {}",
                shortfall.len()
            ));
        }
        if loss.len() != shortfall.len() {
            return invalid(format!(
                "loss must hold one value per shortfall knot ({}); got {}",
                shortfall.len(),
                loss.len()
            ));
        }
        for (name, values) in [("shortfall", &shortfall), ("loss", &loss)] {
            if let Some((index, value)) = values.iter().enumerate().find(|(_, v)| !v.is_finite()) {
                return invalid(format!("{name} must be finite; knot {index} is {value}"));
            }
        }
        for (index, pair) in shortfall.windows(2).enumerate() {
            if pair[1] <= pair[0] {
                return invalid(format!(
                    "shortfall must be strictly increasing; knot {} ({}) does not exceed \
                     knot {index} ({})",
                    index + 1,
                    pair[1],
                    pair[0]
                ));
            }
            if !(pair[1] - pair[0]).is_finite() {
                return invalid(format!(
                    "shortfall knots {index} and {} are too far apart to interpolate",
                    index + 1
                ));
            }
        }
        if let Some(index) = loss
            .windows(2)
            .position(|pair| !(pair[1] - pair[0]).is_finite())
        {
            return invalid(format!(
                "loss values at knots {index} and {} are too far apart to interpolate",
                index + 1
            ));
        }
        Ok(Self { shortfall, loss })
    }

    pub fn shortfall(&self) -> &[f64] {
        &self.shortfall
    }

    pub fn loss(&self) -> &[f64] {
        &self.loss
    }

    /// The loss at `shortfall`, exact at every knot.
    pub fn evaluate(&self, shortfall: f64) -> f64 {
        let knots = &self.shortfall;
        let last = knots.len() - 1;
        // The segment [knots[i], knots[i + 1]] holding the shortfall, or the
        // first or last one beyond the knots.
        let i = knots[1..last].partition_point(|&knot| knot <= shortfall);
        let (left, right) = (knots[i], knots[i + 1]);
        let (low, high) = (self.loss[i], self.loss[i + 1]);
        if shortfall == left {
            return low;
        }
        if shortfall == right {
            return high;
        }
        // Stepping by the slope keeps a far shortfall on a narrow segment,
        // or a near one on a wide segment, from overflowing or underflowing
        // the fraction of the segment covered. A slope too steep to
        // represent falls back to that fraction.
        let slope = (high - low) / (right - left);
        if !slope.is_finite() {
            low + (high - low) * ((shortfall - left) / (right - left))
        } else if shortfall > right {
            high + slope * (shortfall - right)
        } else {
            low + slope * (shortfall - left)
        }
    }
}

/// The losses of `decision`, checked for shape and finiteness.
fn checked_losses<L: LossFunction + ?Sized>(
    loss: &L,
    decision: f64,
    draws: &DecisionDraws,
) -> Result<Vec<Vec<f64>>, DecisionError> {
    let losses = loss.losses(decision, draws)?;
    if losses.len() != draws.chain_count()
        || losses
            .iter()
            .any(|chain| chain.len() != draws.draws_per_chain())
    {
        return Err(DecisionError::LossFunction(format!(
            "the loss function must return one loss per draw, {} chain(s) of {}",
            draws.chain_count(),
            draws.draws_per_chain()
        )));
    }
    for (chain_index, chain) in losses.iter().enumerate() {
        if let Some((draw, value)) = chain.iter().enumerate().find(|(_, v)| !v.is_finite()) {
            return Err(DecisionError::NumericalFailure(format!(
                "the loss of decision {decision} is {value} for chain {chain_index}, draw {draw}; \
                 losses must be finite"
            )));
        }
    }
    Ok(losses)
}

/// The average loss of one decision and its Monte Carlo uncertainty.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExpectedLoss {
    pub decision: f64,
    /// The average of the per-draw losses.
    pub value: f64,
    /// Monte Carlo standard error of `value`: the standard deviation of the
    /// per-draw losses over the square root of `ess`. Zero when every draw
    /// has the same loss, and `NaN` when `ess` is unavailable.
    pub mcse: f64,
    /// Effective sample size of the per-draw losses, from split chains. `NaN`
    /// with fewer than four draws per chain or when every loss is equal.
    pub ess: f64,
}

fn summarize_losses(decision: f64, losses: &[Vec<f64>]) -> ExpectedLoss {
    let (value, deviation) = scaled_moments(|| losses.iter().flatten().copied());
    let ess = ess_raw(losses);
    let mcse = if deviation == 0.0 {
        0.0
    } else {
        deviation / ess.sqrt()
    };
    ExpectedLoss {
        decision,
        value,
        mcse,
        ess,
    }
}

/// The expected loss of `decision`, estimated from `draws`.
pub fn expected_loss<L: LossFunction + ?Sized>(
    draws: &DecisionDraws,
    loss: &L,
    decision: f64,
) -> Result<ExpectedLoss, DecisionError> {
    loss.validate()?;
    if !decision.is_finite() {
        return Err(DecisionError::InvalidDecision(format!(
            "decision must be finite; got {decision}"
        )));
    }
    let losses = checked_losses(loss, decision, draws)?;
    Ok(summarize_losses(decision, &losses))
}

/// Refuse candidates that are empty, not finite, not strictly increasing,
/// or too many to hold.
pub fn validate_candidates(candidates: &[f64]) -> Result<(), DecisionError> {
    let invalid = |message: String| Err(DecisionError::InvalidCandidates(message));
    if candidates.is_empty() {
        return invalid("candidates must hold at least one decision".into());
    }
    if checked_value_count(
        "decision candidates",
        &[candidates.len()],
        MAX_MATERIALIZED_VALUES,
    )
    .is_err()
    {
        return invalid(format!(
            "candidates holds {} decisions, above the limit of {MAX_MATERIALIZED_VALUES}",
            candidates.len()
        ));
    }
    if let Some((index, value)) = candidates.iter().enumerate().find(|(_, v)| !v.is_finite()) {
        return invalid(format!(
            "candidates must be finite; entry {index} is {value}"
        ));
    }
    if let Some(index) = candidates.windows(2).position(|pair| pair[1] <= pair[0]) {
        return invalid(format!(
            "candidates must be strictly increasing; entry {} ({}) does not exceed entry \
             {index} ({})",
            index + 1,
            candidates[index + 1],
            candidates[index]
        ));
    }
    Ok(())
}

/// Up to `count` evenly spaced decisions from the smallest draw to the
/// largest, both included. Constant draws give that one value; a range too
/// narrow to hold `count` distinct floats gives fewer.
pub fn default_candidates(draws: &DecisionDraws, count: usize) -> Vec<f64> {
    let (lowest, highest) = draws
        .values()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), value| {
            (lo.min(value), hi.max(value))
        });
    if lowest == highest || count < 2 {
        return vec![lowest];
    }
    let span = highest - lowest;
    let last = count - 1;
    let mut grid: Vec<f64> = (0..count)
        .map(|index| {
            if index == last {
                return highest;
            }
            let weight = index as f64 / last as f64;
            if span.is_finite() {
                lowest + span * weight
            } else {
                lowest * (1.0 - weight) + highest * weight
            }
        })
        .collect();
    grid.dedup();
    grid
}

/// Average loss at each of `candidates`.
pub fn loss_curve<L: LossFunction + ?Sized>(
    draws: &DecisionDraws,
    loss: &L,
    candidates: &[f64],
) -> Result<Vec<f64>, DecisionError> {
    loss.validate()?;
    validate_candidates(candidates)?;
    candidates
        .iter()
        .map(|&candidate| {
            let losses = checked_losses(loss, candidate, draws)?;
            Ok(scaled_moments(|| losses.iter().flatten().copied()).0)
        })
        .collect()
}

/// How the optimal decision was found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Search {
    /// The closed form when the loss has one, the candidate grid otherwise.
    Auto,
    /// The best of the candidates, whatever the loss.
    Grid,
}

/// Which rule produced a [`Decision`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DecisionMethod {
    Quantile(f64),
    Mean,
    Median,
    /// The candidate with the lowest average loss; the first of any ties.
    Grid,
}

impl From<OptimalRule> for DecisionMethod {
    fn from(rule: OptimalRule) -> Self {
        match rule {
            OptimalRule::Quantile(probability) => Self::Quantile(probability),
            OptimalRule::Mean => Self::Mean,
            OptimalRule::Median => Self::Median,
        }
    }
}

/// The decision that minimises the expected loss, with the evidence behind it.
#[derive(Debug, Clone, PartialEq)]
pub struct Decision {
    pub decision: f64,
    pub method: DecisionMethod,
    /// The expected loss at `decision`. Its standard error treats the
    /// decision as fixed. The decision was chosen to make this average small
    /// on these same draws, so the average is biased low, and the standard
    /// error does not include that bias.
    pub expected_loss: ExpectedLoss,
    pub candidates: Vec<f64>,
    /// Average loss at each candidate.
    pub curve: Vec<f64>,
    /// The optimal decision recomputed on each contiguous block of each chain,
    /// indexed `[chain][block]`. Empty when chains hold a single draw.
    pub batch_decisions: Vec<Vec<f64>>,
    /// Batch-means Monte Carlo standard error of `decision`: the standard
    /// deviation of `batch_decisions` over the square root of their number.
    ///
    /// That formula assumes the decision varies smoothly with the draws, with
    /// a variance that shrinks like one over their number. A grid search
    /// jumps between candidates: on a coarse grid the formula understated the
    /// spread of the decision threefold in simulation. The smallest or
    /// largest draw (a quantile at 0 or 1) does not shrink that way either.
    /// For those this is `NaN`, as it is without batches, and
    /// `batch_decisions` is still reported.
    pub decision_mcse: f64,
    /// True when a grid search chose the first or last candidate (or the
    /// only one), so the minimum may lie outside the grid.
    pub on_grid_edge: bool,
}

/// Contiguous blocks per chain: enough for [`MIN_DECISION_BATCHES`] in all,
/// at least two per chain, and never more than the draws in a chain.
pub fn batches_per_chain(chains: usize, draws_per_chain: usize) -> usize {
    if draws_per_chain < 2 {
        return 0;
    }
    MIN_DECISION_BATCHES
        .div_ceil(chains)
        .max(2)
        .min(draws_per_chain)
}

/// Draw ranges `[start, end)` of the blocks of one chain, as equal as they
/// can be.
fn batch_ranges(draws_per_chain: usize, batches: usize) -> Vec<(usize, usize)> {
    (0..batches)
        .map(|block| {
            (
                block * draws_per_chain / batches,
                (block + 1) * draws_per_chain / batches,
            )
        })
        .collect()
}

fn argmin(values: &[f64]) -> usize {
    values
        .iter()
        .enumerate()
        .fold((0, f64::INFINITY), |(best, lowest), (index, &value)| {
            if value < lowest {
                (index, value)
            } else {
                (best, lowest)
            }
        })
        .0
}

/// The decision that minimises the average loss over `draws`.
///
/// `candidates` defaults to [`default_candidates`] with
/// [`DEFAULT_CANDIDATE_COUNT`] points. With [`Search::Auto`] a loss with an
/// [`OptimalRule`] uses it, and the candidates only trace the curve; any
/// other loss, or [`Search::Grid`], takes the best candidate.
pub fn optimal_decision<L: LossFunction + ?Sized>(
    draws: &DecisionDraws,
    loss: &L,
    candidates: Option<&[f64]>,
    search: Search,
) -> Result<Decision, DecisionError> {
    loss.validate()?;
    let candidates = match candidates {
        Some(candidates) => {
            validate_candidates(candidates)?;
            candidates.to_vec()
        }
        None => default_candidates(draws, DEFAULT_CANDIDATE_COUNT),
    };
    let rule = match search {
        Search::Auto => loss.optimal_rule(),
        Search::Grid => None,
    };
    let per_chain = draws.draws_per_chain();
    let ranges = batch_ranges(per_chain, batches_per_chain(draws.chain_count(), per_chain));

    // The average loss at each candidate, overall and in each block.
    // A grid search keeps the losses of the best candidate so far, so its
    // expected loss needs no second evaluation: a Python loss is called
    // exactly once per candidate.
    let mut curve = Vec::with_capacity(candidates.len());
    let mut block_curves: Vec<Vec<Vec<f64>>> =
        vec![vec![Vec::with_capacity(candidates.len()); ranges.len()]; draws.chain_count()];
    let mut best_losses: Option<(f64, Vec<Vec<f64>>)> = None;
    for &candidate in &candidates {
        let losses = checked_losses(loss, candidate, draws)?;
        let average = scaled_moments(|| losses.iter().flatten().copied()).0;
        curve.push(average);
        if rule.is_none() {
            for (chain, chain_curves) in losses.iter().zip(block_curves.iter_mut()) {
                for (&(start, end), block_curve) in ranges.iter().zip(chain_curves.iter_mut()) {
                    block_curve.push(scaled_moments(|| chain[start..end].iter().copied()).0);
                }
            }
            if best_losses
                .as_ref()
                .is_none_or(|(lowest, _)| average < *lowest)
            {
                best_losses = Some((average, losses));
            }
        }
    }

    let (decision, method, best) = match rule {
        Some(rule) => (rule.apply(draws.values()), DecisionMethod::from(rule), None),
        None => {
            let best = argmin(&curve);
            (candidates[best], DecisionMethod::Grid, Some(best))
        }
    };
    let batch_decisions: Vec<Vec<f64>> = match rule {
        Some(rule) => draws
            .chains()
            .iter()
            .map(|chain| {
                ranges
                    .iter()
                    .map(|&(start, end)| rule.apply(chain[start..end].iter().copied()))
                    .collect()
            })
            .collect(),
        None => block_curves
            .iter()
            .map(|chain_curves| {
                chain_curves
                    .iter()
                    .map(|block_curve| candidates[argmin(block_curve)])
                    .collect()
            })
            .collect(),
    };
    let batch_count = batch_decisions.iter().map(Vec::len).sum::<usize>();
    let smooth = match method {
        DecisionMethod::Mean | DecisionMethod::Median => true,
        DecisionMethod::Quantile(probability) => probability > 0.0 && probability < 1.0,
        DecisionMethod::Grid => false,
    };
    let decision_mcse = if batch_count < 2 || !smooth {
        f64::NAN
    } else {
        scaled_moments(|| batch_decisions.iter().flatten().copied()).1 / (batch_count as f64).sqrt()
    };
    let on_grid_edge = best.is_some_and(|best| best == 0 || best == candidates.len() - 1);
    let expected_loss = match best_losses {
        Some((_, losses)) => summarize_losses(decision, &losses),
        None => expected_loss(draws, loss, decision)?,
    };
    Ok(Decision {
        decision,
        method,
        expected_loss,
        candidates,
        curve,
        batch_decisions,
        decision_mcse,
        on_grid_edge,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diagnostics::inv_normal_cdf;
    use crate::forecast_common::cumulative_paths;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rand_distr::{Distribution, Normal};

    fn normal_chains(seed: u64, chains: usize, draws: usize, mean: f64, sd: f64) -> DecisionDraws {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let normal = Normal::new(mean, sd).unwrap();
        DecisionDraws::new(
            (0..chains)
                .map(|_| (0..draws).map(|_| normal.sample(&mut rng)).collect())
                .collect(),
        )
        .unwrap()
    }

    /// Stationary AR(1) chains with unit marginal variance.
    fn ar1_chains(seed: u64, chains: usize, draws: usize, phi: f64) -> DecisionDraws {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let innovation = Normal::new(0.0, (1.0 - phi * phi).sqrt()).unwrap();
        let start = Normal::new(0.0, 1.0).unwrap();
        DecisionDraws::new(
            (0..chains)
                .map(|_| {
                    let mut value = start.sample(&mut rng);
                    (0..draws)
                        .map(|_| {
                            value = phi * value + innovation.sample(&mut rng);
                            value
                        })
                        .collect()
                })
                .collect(),
        )
        .unwrap()
    }

    fn sorted(draws: &DecisionDraws) -> Vec<f64> {
        let mut values: Vec<f64> = draws.values().collect();
        values.sort_by(f64::total_cmp);
        values
    }

    fn mean_sd(values: &[f64]) -> (f64, f64) {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance =
            values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
        (mean, variance.sqrt())
    }

    #[test]
    fn pinball_optimum_is_the_shared_empirical_quantile() {
        let draws = normal_chains(1, 4, 250, 7.5, 1.25);
        let loss = Loss::asymmetric_linear(5.0, 1.0).unwrap();
        let decision = optimal_decision(&draws, &loss, None, Search::Auto).unwrap();
        assert_eq!(decision.method, DecisionMethod::Quantile(5.0 / 6.0));
        assert_eq!(
            decision.decision,
            sorted_quantile(&sorted(&draws), 5.0 / 6.0)
        );
    }

    #[test]
    fn pinball_optimum_approaches_the_normal_quantile() {
        let (mean, sd) = (7.5, 1.25);
        let probability = 5.0 / 6.0;
        let analytic = mean + sd * inv_normal_cdf(probability);
        let draws = normal_chains(2, 4, 100_000, mean, sd);
        let loss = Loss::asymmetric_linear(5.0, 1.0).unwrap();
        let decision = optimal_decision(&draws, &loss, None, Search::Auto).unwrap();
        // The quantile's standard error is sd * sqrt(p(1-p)) / (phi(z) sqrt(n)),
        // about 0.0033 here; allow four of them.
        assert!(
            (decision.decision - analytic).abs() < 0.014,
            "{} vs {analytic}",
            decision.decision
        );
        assert!(decision.decision_mcse > 0.001 && decision.decision_mcse < 0.008);
    }

    #[test]
    fn squared_loss_gives_the_mean_and_absolute_loss_the_median() {
        let draws = normal_chains(3, 3, 101, 2.0, 3.0);
        let values: Vec<f64> = draws.values().collect();
        let squared = optimal_decision(&draws, &Loss::Squared, None, Search::Auto).unwrap();
        assert_eq!(squared.method, DecisionMethod::Mean);
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        assert!((squared.decision - mean).abs() < 1e-12);
        let absolute = optimal_decision(&draws, &Loss::Absolute, None, Search::Auto).unwrap();
        assert_eq!(absolute.method, DecisionMethod::Median);
        // 303 draws: the median is the 152nd order statistic exactly.
        assert_eq!(absolute.decision, sorted(&draws)[151]);
    }

    #[test]
    fn expected_loss_matches_a_direct_average() {
        let draws = normal_chains(4, 2, 50, 0.0, 1.0);
        let values: Vec<f64> = draws.values().collect();
        for (loss, direct) in [
            (
                Loss::asymmetric_linear(5.0, 1.0).unwrap(),
                values
                    .iter()
                    .map(|&y| if y > 0.3 { 5.0 * (y - 0.3) } else { 0.3 - y })
                    .sum::<f64>(),
            ),
            (
                Loss::Squared,
                values.iter().map(|&y| (y - 0.3) * (y - 0.3)).sum::<f64>(),
            ),
            (
                Loss::Absolute,
                values.iter().map(|&y| (y - 0.3f64).abs()).sum::<f64>(),
            ),
        ] {
            let estimate = expected_loss(&draws, &loss, 0.3).unwrap();
            assert!((estimate.value - direct / 100.0).abs() < 1e-12, "{loss:?}");
            assert!(estimate.mcse > 0.0 && estimate.ess > 10.0);
        }
    }

    #[test]
    fn grid_search_agrees_with_the_closed_forms() {
        let draws = normal_chains(5, 4, 500, 7.5, 1.25);
        let n = draws.len() as f64;
        let ordered = sorted(&draws);
        let widest_gap = ordered
            .windows(2)
            .map(|pair| pair[1] - pair[0])
            .fold(0.0, f64::max);
        for loss in [
            Loss::asymmetric_linear(5.0, 1.0).unwrap(),
            Loss::asymmetric_linear(1.0, 3.0).unwrap(),
            Loss::Squared,
            Loss::Absolute,
        ] {
            let closed = optimal_decision(&draws, &loss, None, Search::Auto).unwrap();
            // Every draw is a candidate, so the grid finds the exact minimiser
            // over the draws for the linear losses.
            let grid = optimal_decision(&draws, &loss, Some(&ordered), Search::Grid).unwrap();
            assert_eq!(grid.method, DecisionMethod::Grid);
            let (closed_value, grid_value) = (closed.expected_loss.value, grid.expected_loss.value);
            match loss {
                // A linear loss's average is minimised at a draw, which the
                // grid holds; the interpolated quantile is within one gap.
                Loss::AsymmetricLinear { under, over } => {
                    assert!(closed_value >= grid_value - 1e-12, "{loss:?}");
                    let bound = (under + over) / n * widest_gap;
                    assert!(closed_value - grid_value <= bound, "{loss:?}");
                }
                // Both medians of an even count are minimisers.
                Loss::Absolute => assert!((closed_value - grid_value).abs() < 1e-12),
                // The mean is the exact minimiser, and no draw beats it.
                _ => assert!(closed_value <= grid_value + 1e-12),
            }
            assert!((closed.decision - grid.decision).abs() < 0.05, "{loss:?}");
        }
    }

    #[test]
    fn piecewise_loss_reproduces_pinball_and_is_exact_at_knots() {
        let piecewise =
            PiecewiseLinearLoss::new(vec![-1.0, 0.0, 1.0], vec![1.0, 0.0, 5.0]).unwrap();
        for (shortfall, expected) in [
            (-3.0, 3.0),
            (-1.0, 1.0),
            (0.0, 0.0),
            (0.5, 2.5),
            (4.0, 20.0),
        ] {
            assert_eq!(piecewise.evaluate(shortfall), expected, "{shortfall}");
        }
        let draws = normal_chains(6, 4, 500, 7.5, 1.25);
        let candidates = default_candidates(&draws, 2001);
        let pinball = Loss::asymmetric_linear(5.0, 1.0).unwrap();
        let loss = Loss::PiecewiseLinear(piecewise);
        let grid = optimal_decision(&draws, &loss, Some(&candidates), Search::Auto).unwrap();
        assert_eq!(grid.method, DecisionMethod::Grid);
        let reference =
            optimal_decision(&draws, &pinball, Some(&candidates), Search::Auto).unwrap();
        for (a, b) in grid.curve.iter().zip(&reference.curve) {
            assert!((a - b).abs() < 1e-12);
        }
        let spacing = candidates[1] - candidates[0];
        assert!((grid.decision - reference.decision).abs() <= 2.0 * spacing + 1e-12);
        assert!(!grid.on_grid_edge);
    }

    #[test]
    fn horizon_totals_match_sums_by_hand_and_cumulative_paths() {
        let paths = vec![
            vec![vec![1.0, 2.0, 3.0, 4.0], vec![0.5, -1.0, 2.5, 1.0]],
            vec![vec![0.1, 0.2, 0.3, 0.4], vec![10.0, 20.0, 30.0, 40.0]],
        ];
        let totals = DecisionDraws::from_path_totals(&paths, &[3, 1]).unwrap();
        assert_eq!(totals.chains(), &[vec![6.0, 0.0], vec![0.2 + 0.4, 60.0]]);
        let cumulative = cumulative_paths(&paths).unwrap();
        let prefix = DecisionDraws::from_path_totals(&paths, &[0, 1, 2]).unwrap();
        for (chain, cumulative_chain) in prefix.chains().iter().zip(&cumulative) {
            for (total, path) in chain.iter().zip(cumulative_chain) {
                assert_eq!(*total, path[2]);
            }
        }
        let error = DecisionDraws::from_path_totals(&paths, &[4]).unwrap_err();
        assert!(error.to_string().contains("steps holds step 4"), "{error}");
        let error = DecisionDraws::from_path_totals(&paths, &[1, 1]).unwrap_err();
        assert!(error.to_string().contains("more than once"), "{error}");
        assert!(DecisionDraws::from_path_totals(&paths, &[]).is_err());
        let overflowing = vec![vec![vec![f64::MAX, f64::MAX]]];
        assert!(matches!(
            DecisionDraws::from_path_totals(&overflowing, &[0, 1]),
            Err(DecisionError::NumericalFailure(_))
        ));
    }

    #[test]
    fn expected_loss_mcse_matches_repeated_simulation() {
        // Independent draws and strongly autocorrelated chains. For each, the
        // spread of the estimate over seeds should match the average reported
        // MCSE; with 400 replicates the spread itself is known to about 4%.
        let loss = Loss::asymmetric_linear(5.0, 1.0).unwrap();
        for (phi, label) in [(0.0, "independent"), (0.8, "AR(1) 0.8")] {
            let (estimates, mcses): (Vec<f64>, Vec<f64>) = (0..400)
                .map(|seed| {
                    let draws = ar1_chains(1000 + seed, 4, 500, phi);
                    let estimate = expected_loss(&draws, &loss, 1.0).unwrap();
                    (estimate.value, estimate.mcse)
                })
                .unzip();
            let (_, spread) = mean_sd(&estimates);
            let (reported, _) = mean_sd(&mcses);
            let ratio = reported / spread;
            assert!(
                (0.85..1.15).contains(&ratio),
                "{label}: reported {reported}, observed {spread}"
            );
        }
    }

    #[test]
    fn autocorrelation_raises_the_mcse_above_the_independent_formula() {
        let draws = ar1_chains(9, 4, 1000, 0.9);
        let loss = Loss::Squared;
        let estimate = expected_loss(&draws, &loss, 0.0).unwrap();
        let losses = loss.losses(0.0, &draws).unwrap();
        let flat: Vec<f64> = losses.into_iter().flatten().collect();
        let naive = mean_sd(&flat).1 / (flat.len() as f64).sqrt();
        assert!(estimate.mcse > 2.0 * naive, "{} vs {naive}", estimate.mcse);
        assert!(estimate.ess < 0.25 * draws.len() as f64);
    }

    #[test]
    fn decision_mcse_matches_repeated_simulation() {
        let loss = Loss::asymmetric_linear(5.0, 1.0).unwrap();
        for (chains, phi) in [(4, 0.0), (1, 0.0), (4, 0.5)] {
            let (decisions, mcses): (Vec<f64>, Vec<f64>) = (0..400)
                .map(|seed| {
                    let draws = ar1_chains(5000 + seed, chains, 2000 / chains, phi);
                    let decision = optimal_decision(&draws, &loss, None, Search::Auto).unwrap();
                    assert_eq!(decision.batch_decisions.len(), chains);
                    (decision.decision, decision.decision_mcse)
                })
                .unzip();
            let (_, spread) = mean_sd(&decisions);
            let (reported, _) = mean_sd(&mcses);
            let ratio = reported / spread;
            // Batch means from eight blocks is a rough estimate: the reported
            // value is noisy, and small blocks bias a tail quantile. Hold it
            // to within 20% on average.
            assert!(
                (0.8..1.2).contains(&ratio),
                "{chains} chain(s), phi {phi}: reported {reported}, observed {spread}"
            );
        }
    }

    #[test]
    fn batches_cover_every_chain() {
        assert_eq!(batches_per_chain(1, 1000), 8);
        assert_eq!(batches_per_chain(2, 1000), 4);
        assert_eq!(batches_per_chain(4, 1000), 2);
        assert_eq!(batches_per_chain(16, 1000), 2);
        assert_eq!(batches_per_chain(1, 3), 3);
        assert_eq!(batches_per_chain(3, 1), 0);
        assert_eq!(batch_ranges(10, 3), vec![(0, 3), (3, 6), (6, 10)]);
        let single = DecisionDraws::new(vec![vec![1.0], vec![2.0], vec![4.0]]).unwrap();
        let decision = optimal_decision(&single, &Loss::Squared, None, Search::Auto).unwrap();
        assert!(decision.batch_decisions.iter().all(Vec::is_empty));
        assert!(decision.decision_mcse.is_nan());
        assert!(decision.expected_loss.ess.is_nan() && decision.expected_loss.mcse.is_nan());
    }

    #[test]
    fn grid_edge_is_reported() {
        let draws = normal_chains(7, 2, 100, 0.0, 1.0);
        let decision =
            optimal_decision(&draws, &Loss::Squared, Some(&[3.0, 4.0, 5.0]), Search::Grid).unwrap();
        assert_eq!(decision.decision, 3.0);
        assert!(decision.on_grid_edge);
        let closed =
            optimal_decision(&draws, &Loss::Squared, Some(&[3.0, 4.0, 5.0]), Search::Auto).unwrap();
        assert!(!closed.on_grid_edge);
        assert_eq!(closed.curve.len(), 3);
    }

    #[test]
    fn constant_draws_have_zero_mcse_and_one_candidate() {
        let draws = DecisionDraws::new(vec![vec![2.0; 10]; 2]).unwrap();
        assert_eq!(default_candidates(&draws, 201), vec![2.0]);
        let decision = optimal_decision(&draws, &Loss::Absolute, None, Search::Auto).unwrap();
        assert_eq!(decision.decision, 2.0);
        assert_eq!(decision.expected_loss.value, 0.0);
        assert_eq!(decision.expected_loss.mcse, 0.0);
        assert_eq!(decision.decision_mcse, 0.0);
    }

    #[test]
    fn default_grid_spans_the_draws_without_overflow() {
        let draws = DecisionDraws::new(vec![vec![-f64::MAX, 0.0, f64::MAX]]).unwrap();
        let grid = default_candidates(&draws, 5);
        assert_eq!(grid.first(), Some(&-f64::MAX));
        assert_eq!(grid.last(), Some(&f64::MAX));
        assert!(grid.windows(2).all(|pair| pair[1] > pair[0]));
        let narrow = DecisionDraws::new(vec![vec![1.0, 1.0 + f64::EPSILON]]).unwrap();
        assert_eq!(
            default_candidates(&narrow, 201),
            vec![1.0, 1.0 + f64::EPSILON]
        );
    }

    #[test]
    fn invalid_inputs_are_refused_by_name() {
        let message = |error: DecisionError| error.to_string();
        assert!(message(DecisionDraws::new(vec![]).unwrap_err()).contains("at least one draw"));
        assert!(
            message(DecisionDraws::new(vec![vec![1.0], vec![]]).unwrap_err())
                .contains("same number of draws")
        );
        assert!(
            message(DecisionDraws::new(vec![vec![1.0, f64::NAN]]).unwrap_err())
                .contains("chain 0, draw 1 is NaN")
        );
        assert!(message(Loss::asymmetric_linear(-1.0, 1.0).unwrap_err()).contains("under"));
        assert!(message(Loss::asymmetric_linear(1.0, f64::INFINITY).unwrap_err()).contains("over"));
        assert!(Loss::asymmetric_linear(0.0, 0.0).is_err());
        assert!(
            message(PiecewiseLinearLoss::new(vec![0.0], vec![0.0]).unwrap_err())
                .contains("at least two knots")
        );
        assert!(
            message(PiecewiseLinearLoss::new(vec![0.0, 0.0], vec![0.0, 1.0]).unwrap_err())
                .contains("strictly increasing")
        );
        assert!(
            message(PiecewiseLinearLoss::new(vec![0.0, 1.0], vec![0.0]).unwrap_err())
                .contains("one value per shortfall knot")
        );
        assert!(PiecewiseLinearLoss::new(vec![-f64::MAX, f64::MAX], vec![0.0, 1.0]).is_err());
        let draws = DecisionDraws::new(vec![vec![1.0, 2.0]]).unwrap();
        assert!(
            message(expected_loss(&draws, &Loss::Squared, f64::NAN).unwrap_err())
                .contains("decision must be finite")
        );
        for (candidates, fragment) in [
            (vec![], "at least one decision"),
            (vec![1.0, f64::INFINITY], "entry 1 is inf"),
            (vec![2.0, 1.0], "strictly increasing"),
        ] {
            let error = optimal_decision(&draws, &Loss::Squared, Some(&candidates), Search::Auto)
                .unwrap_err();
            assert!(matches!(error, DecisionError::InvalidCandidates(_)));
            assert!(message(error).contains(fragment), "{candidates:?}");
        }
        let unvalidated = Loss::AsymmetricLinear {
            under: f64::NAN,
            over: 1.0,
        };
        assert!(matches!(
            expected_loss(&draws, &unvalidated, 1.0),
            Err(DecisionError::InvalidLoss(_))
        ));
        let huge = DecisionDraws::new(vec![vec![1e200, -1e200]]).unwrap();
        assert!(matches!(
            expected_loss(&huge, &Loss::Squared, 0.0),
            Err(DecisionError::NumericalFailure(_))
        ));
    }

    #[test]
    fn decision_mcse_is_unavailable_where_batch_means_does_not_apply() {
        let draws = normal_chains(10, 4, 500, 0.0, 1.0);
        let grid =
            optimal_decision(&draws, &Loss::Squared, Some(&[-1.0, 1.0]), Search::Grid).unwrap();
        assert!(grid.decision_mcse.is_nan());
        assert_eq!(grid.batch_decisions.len(), 4);
        let minimum = Loss::asymmetric_linear(0.0, 1.0).unwrap();
        let lowest = optimal_decision(&draws, &minimum, None, Search::Auto).unwrap();
        assert_eq!(lowest.method, DecisionMethod::Quantile(0.0));
        assert_eq!(lowest.decision, sorted(&draws)[0]);
        assert!(lowest.decision_mcse.is_nan());
    }

    #[test]
    fn grid_expected_loss_reuses_the_chosen_candidates_losses() {
        use std::cell::Cell;
        struct Counting<'a>(&'a Cell<usize>);
        impl LossFunction for Counting<'_> {
            fn losses(
                &self,
                decision: f64,
                draws: &DecisionDraws,
            ) -> Result<Vec<Vec<f64>>, DecisionError> {
                self.0.set(self.0.get() + 1);
                Loss::Squared.losses(decision, draws)
            }
        }
        let calls = Cell::new(0);
        let draws = normal_chains(11, 2, 50, 2.0, 1.0);
        let candidates = [1.0, 2.0, 3.0];
        let decision =
            optimal_decision(&draws, &Counting(&calls), Some(&candidates), Search::Auto).unwrap();
        assert_eq!(calls.get(), 3);
        let direct = expected_loss(&draws, &Loss::Squared, decision.decision).unwrap();
        assert_eq!(decision.expected_loss, direct);
        let only = optimal_decision(&draws, &Counting(&calls), Some(&[5.0]), Search::Auto).unwrap();
        assert!(only.on_grid_edge, "a single candidate is the grid's edge");
    }

    #[test]
    fn extreme_values_do_not_overflow_representable_losses() {
        let pinball = Loss::asymmetric_linear(0.1, 1.0).unwrap();
        let loss = pinball.evaluate(-1e308, 1e308);
        assert!((loss - 2e307).abs() <= 2e307 * 1e-15, "{loss}");
        let free = Loss::asymmetric_linear(0.0, 1.0).unwrap();
        assert_eq!(free.evaluate(-1e308, 1e308), 0.0);
        // The identity on a very narrow first segment, far beyond it.
        let narrow = PiecewiseLinearLoss::new(vec![0.0, 1e-308], vec![0.0, 1e-308]).unwrap();
        assert_eq!(narrow.evaluate(10.0), 10.0);
        let flat = PiecewiseLinearLoss::new(vec![0.0, 1e-308], vec![5.0, 5.0]).unwrap();
        assert_eq!(flat.evaluate(10.0), 5.0);
        // A small shortfall on a very wide segment is not rounded to zero.
        let wide = PiecewiseLinearLoss::new(vec![0.0, 1e308], vec![0.0, 1e308]).unwrap();
        assert_eq!(wide.evaluate(1e-16), 1e-16);
    }

    #[test]
    fn horizon_totals_refuse_non_finite_steps_even_when_not_chosen() {
        let paths = vec![vec![vec![1.0, f64::NAN], vec![2.0, 3.0]]];
        let error = DecisionDraws::from_path_totals(&paths, &[0]).unwrap_err();
        assert!(
            error.to_string().contains("chain 0, draw 0, step 1 is NaN"),
            "{error}"
        );
    }

    /// The calibration run behind the "3% to 7% below" statement in the
    /// changelog and docs/decisions.md: 3,000 replicates for each of two seed
    /// offsets. With `phi = 0` the one-chain and four-chain cases draw almost
    /// the same values from a seed, so they are not independent checks. It
    /// takes about two minutes in release mode:
    ///
    /// `cargo test -p rustmc_core --release --lib decision_mcse_calibration -- --ignored --nocapture`
    #[test]
    #[ignore = "slow: 18,000 decisions; run by hand to reproduce the documented calibration"]
    fn decision_mcse_calibration_over_3000_replicates() {
        let loss = Loss::asymmetric_linear(5.0, 1.0).unwrap();
        for offset in [5000, 100_000] {
            for (chains, phi) in [(4, 0.0), (1, 0.0), (4, 0.5)] {
                let (decisions, mcses): (Vec<f64>, Vec<f64>) = (0..3000)
                    .map(|seed| {
                        let draws = ar1_chains(offset + seed, chains, 2000 / chains, phi);
                        let decision = optimal_decision(&draws, &loss, None, Search::Auto).unwrap();
                        (decision.decision, decision.decision_mcse)
                    })
                    .unzip();
                let (_, spread) = mean_sd(&decisions);
                let (reported, _) = mean_sd(&mcses);
                let ratio = reported / spread;
                println!(
                    "offset {offset}, {chains} chain(s), phi {phi}: reported {reported:.5}, \
                     observed {spread:.5}, ratio {ratio:.3}"
                );
                assert!((0.9..1.02).contains(&ratio));
            }
        }
    }

    #[test]
    fn custom_loss_functions_are_checked_for_shape() {
        struct Broken;
        impl LossFunction for Broken {
            fn losses(&self, _: f64, _: &DecisionDraws) -> Result<Vec<Vec<f64>>, DecisionError> {
                Ok(vec![vec![0.0]])
            }
        }
        let draws = DecisionDraws::new(vec![vec![1.0, 2.0]]).unwrap();
        assert!(matches!(
            expected_loss(&draws, &Broken, 1.0),
            Err(DecisionError::LossFunction(_))
        ));
    }
}
