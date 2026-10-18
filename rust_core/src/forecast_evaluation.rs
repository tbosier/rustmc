//! Calibration and point accuracy of forecasts scored against realised outcomes.
//!
//! `rustmc.evaluation` reports these for a rolling-origin backtest. They are
//! computed here so that a forecast's interval, the coverage counted from it and
//! its PIT value all come from the same sorted draws and the same quantile rule,
//! [`sorted_quantile`].
//!
//! A *target* is one forecast quantity: one series at one horizon step of one
//! forecast. Every function takes predictive draws target by target and treats
//! a `NaN` outcome as missing: it has no PIT value and is left out of every
//! count and mean. An infinite outcome is refused.
//!
//! Nothing here assumes targets are independent, and nothing here tests
//! uniformity or calibration. The Wilson interval that [`IntervalCalibration`]
//! reports is the binomial interval for independent hits. Targets from
//! overlapping rolling origins, from several horizon steps of one path, or from
//! related series are not independent, so the interval is too narrow for them
//! by an amount this module cannot estimate.

use crate::diagnostics::scaled_moments;
use crate::forecast_common::{checked_value_count, sorted_quantile, MAX_MATERIALIZED_VALUES};
use crate::seeding::chain_seed;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

/// Domain separator for the uniforms of a randomised PIT, `"PIT_RAND"`.
const PIT_SEED_DOMAIN: u64 = 0x5049_545F_5241_4E44;

/// The standard normal quantile at 0.975, for a 95% Wilson interval.
const WILSON_Z: f64 = 1.959_963_984_540_054;

/// Why an evaluation was refused. The message is the whole explanation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvaluationError {
    /// The draws, outcomes, levels or grouping are invalid; nothing was
    /// computed.
    InvalidInput(String),
}

impl std::fmt::Display for EvaluationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for EvaluationError {}

fn invalid(message: impl Into<String>) -> EvaluationError {
    EvaluationError::InvalidInput(message.into())
}

/// Where an outcome falls among the draws of its predictive distribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutcomeRank {
    /// Draws strictly below the outcome.
    pub below: usize,
    /// Draws at or below the outcome.
    pub at_or_below: usize,
    /// All draws.
    pub draws: usize,
}

impl OutcomeRank {
    /// The rank of `actual` among `ordered`, which must be ascending and
    /// non-empty.
    pub fn new(ordered: &[f64], actual: f64) -> Self {
        Self {
            below: ordered.partition_point(|&draw| draw < actual),
            at_or_below: ordered.partition_point(|&draw| draw <= actual),
            draws: ordered.len(),
        }
    }

    /// The empirical predictive CDF at the outcome, `P(X <= y)`.
    ///
    /// This is the textbook PIT. For a forecast with atoms, such as a count,
    /// it is not uniform even when the forecast is exactly right: it jumps
    /// over the probability of each atom, so its values pile onto a few
    /// points.
    pub fn pit(&self) -> f64 {
        self.at_or_below as f64 / self.draws as f64
    }

    /// The randomised PIT for a uniform `u` on `[0, 1)`:
    /// `(below + u * (ties + 1)) / (draws + 1)`, where `ties` counts draws
    /// equal to the outcome.
    ///
    /// Placing the outcome uniformly among the `draws + 1` values it would be
    /// ranked with, ties broken at random, makes this exactly uniform on
    /// `[0, 1)` whenever the outcome and the draws are exchangeable, which is
    /// what "drawn from the same distribution, independently" gives. That holds
    /// for counts, for continuous amounts and for a mixture such as a hurdle
    /// model's zeros and positive amounts, with no need to know which. With
    /// many draws it approaches the randomised PIT of Czado, Gneiting and Held
    /// (2009), `F(y-) + u (F(y) - F(y-))`.
    ///
    /// Posterior-predictive draws from a Markov chain are not independent of
    /// each other, so for them "exactly" becomes "approximately".
    pub fn randomized_pit(&self, uniform: f64) -> f64 {
        let ties = (self.at_or_below - self.below) as f64;
        (self.below as f64 + uniform * (ties + 1.0)) / (self.draws as f64 + 1.0)
    }
}

/// Which summary of the predictive draws is the point forecast.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Point {
    Mean,
    /// The better point forecast under absolute error.
    Median,
}

/// Predictive draws for a set of targets, each target's draws sorted.
#[derive(Debug, Clone)]
pub struct TargetDraws {
    ordered: Vec<Vec<f64>>,
}

impl TargetDraws {
    /// Sort each target's draws. Every target needs at least one draw, all
    /// targets the same number, and every draw must be finite.
    pub fn new(mut columns: Vec<Vec<f64>>) -> Result<Self, EvaluationError> {
        let draws = columns.first().map_or(0, Vec::len);
        if draws == 0 || columns.iter().any(|column| column.len() != draws) {
            return Err(invalid(
                "predictive draws must hold the same nonzero number of draws for every target",
            ));
        }
        if columns.iter().flatten().any(|draw| !draw.is_finite()) {
            return Err(invalid("predictive draws must be finite"));
        }
        columns
            .par_iter_mut()
            .for_each(|column| column.sort_by(f64::total_cmp));
        Ok(Self { ordered: columns })
    }

    /// The number of targets.
    pub fn len(&self) -> usize {
        self.ordered.len()
    }

    /// Whether there are no targets.
    pub fn is_empty(&self) -> bool {
        self.ordered.is_empty()
    }

    /// The draws for one target, ascending.
    pub fn ordered(&self, target: usize) -> &[f64] {
        &self.ordered[target]
    }

    /// The equal-tailed central interval holding `level` of one target's
    /// draws: the `(1 - level) / 2` and `(1 + level) / 2` quantiles by
    /// [`sorted_quantile`], the rule behind every forecast interval.
    pub fn central_interval(&self, target: usize, level: f64) -> (f64, f64) {
        let ordered = &self.ordered[target];
        (
            sorted_quantile(ordered, (1.0 - level) / 2.0),
            sorted_quantile(ordered, (1.0 + level) / 2.0),
        )
    }

    /// Each target's point-forecast error, `point - actual`, `NaN` where the
    /// outcome is missing.
    ///
    /// The error is the mean or median of `draw - actual`, as
    /// `score_forecast`'s `bias` is, rather than the mean or median less the
    /// outcome: a summary of large draws rounds to their spacing, and
    /// subtracting afterwards would keep that rounding. Where a difference
    /// overflows, halves are summarised and doubled, so the error is infinite
    /// only when it is unrepresentable.
    pub fn point_errors(&self, actual: &[f64], point: Point) -> Result<Vec<f64>, EvaluationError> {
        self.check_outcomes(actual)?;
        Ok(self
            .ordered
            .iter()
            .zip(actual)
            .map(|(ordered, &y)| {
                if y.is_nan() {
                    return f64::NAN;
                }
                let overflow = ordered.iter().any(|d| !(d - y).is_finite());
                let factor = if overflow { 2.0 } else { 1.0 };
                // Subtracting one value keeps the draws in ascending order.
                let errors: Vec<f64> = ordered
                    .iter()
                    .map(|d| if overflow { d * 0.5 - y * 0.5 } else { d - y })
                    .collect();
                factor
                    * match point {
                        Point::Mean => scaled_moments(|| errors.iter().copied()).0,
                        Point::Median => sorted_quantile(&errors, 0.5),
                    }
            })
            .collect())
    }

    /// The rank of each outcome among its target's draws; `None` for a
    /// missing (`NaN`) outcome.
    pub fn ranks(&self, actual: &[f64]) -> Result<Vec<Option<OutcomeRank>>, EvaluationError> {
        self.check_outcomes(actual)?;
        Ok(self
            .ordered
            .iter()
            .zip(actual)
            .map(|(ordered, &y)| (!y.is_nan()).then(|| OutcomeRank::new(ordered, y)))
            .collect())
    }

    /// [`OutcomeRank::pit`] for each outcome, `NaN` where it is missing.
    pub fn pit(&self, actual: &[f64]) -> Result<Vec<f64>, EvaluationError> {
        Ok(self
            .ranks(actual)?
            .into_iter()
            .map(|rank| rank.map_or(f64::NAN, |rank| rank.pit()))
            .collect())
    }

    /// [`OutcomeRank::randomized_pit`] for each outcome, `NaN` where it is
    /// missing.
    ///
    /// One uniform is drawn per target in target order, missing or not, from
    /// the stream `stream` of `seed`, so a target's value does not depend on
    /// which other outcomes are missing.
    pub fn randomized_pit(
        &self,
        actual: &[f64],
        seed: u64,
        stream: usize,
    ) -> Result<Vec<f64>, EvaluationError> {
        let ranks = self.ranks(actual)?;
        let mut rng = ChaCha8Rng::seed_from_u64(chain_seed(seed, stream, PIT_SEED_DOMAIN));
        Ok(ranks
            .into_iter()
            .map(|rank| {
                let uniform: f64 = rng.gen();
                rank.map_or(f64::NAN, |rank| rank.randomized_pit(uniform))
            })
            .collect())
    }

    fn check_outcomes(&self, actual: &[f64]) -> Result<(), EvaluationError> {
        if actual.len() != self.len() {
            return Err(invalid(format!(
                "{} outcomes for {} forecast targets",
                actual.len(),
                self.len()
            )));
        }
        if actual.iter().any(|y| y.is_infinite()) {
            return Err(invalid(
                "actual values must be finite or NaN for missing outcomes",
            ));
        }
        Ok(())
    }
}

fn check_groups(group: &[usize], targets: usize, groups: usize) -> Result<(), EvaluationError> {
    if group.len() != targets {
        return Err(invalid(format!(
            "{} group labels for {targets} forecast targets",
            group.len()
        )));
    }
    if group.iter().any(|&g| g >= groups) {
        return Err(invalid(format!("group labels must be below {groups}")));
    }
    Ok(())
}

/// Counts of PIT values in `bins` equal-width bins on `[0, 1]`.
///
/// `NaN` values (missing outcomes) are skipped. Bin `k` holds values in
/// `[k / bins, (k + 1) / bins)`, except that the last bin also holds 1.
pub fn pit_histogram(values: &[f64], bins: usize) -> Result<Vec<usize>, EvaluationError> {
    if bins == 0 {
        return Err(invalid("bins must be a positive integer"));
    }
    checked_value_count("PIT histogram", &[bins], MAX_MATERIALIZED_VALUES)
        .map_err(|error| invalid(error.to_string()))?;
    if values
        .iter()
        .any(|v| !v.is_nan() && !(0.0..=1.0).contains(v))
    {
        return Err(invalid(
            "PIT values must lie between zero and one, or be NaN",
        ));
    }
    let width = bins as f64;
    let mut counts = vec![0; bins];
    for &value in values.iter().filter(|v| !v.is_nan()) {
        // `value * bins` can round up across an edge, as 0.8999... * 10 does
        // to 9, so the bin is checked against its edges `k / bins` directly.
        let mut bin = ((value * width) as usize).min(bins - 1);
        if bin > 0 && value < bin as f64 / width {
            bin -= 1;
        } else if bin + 1 < bins && value >= (bin + 1) as f64 / width {
            bin += 1;
        }
        counts[bin] += 1;
    }
    Ok(counts)
}

/// The 95% Wilson score interval for a binomial proportion `hits / count`,
/// or `NaN`s when `count` is zero.
///
/// Unlike `p ± 1.96 sqrt(p (1 - p) / n)`, it does not collapse to a point
/// when every interval, or none, covers. It assumes independent trials.
pub fn wilson_interval(hits: usize, count: usize) -> (f64, f64) {
    if count == 0 {
        return (f64::NAN, f64::NAN);
    }
    let n = count as f64;
    let p = hits as f64 / n;
    let z2 = WILSON_Z * WILSON_Z;
    let denominator = 1.0 + z2 / n;
    let centre = (p + z2 / (2.0 * n)) / denominator;
    let half = WILSON_Z / denominator * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt();
    ((centre - half).max(0.0), (centre + half).min(1.0))
}

/// Validate central-interval coverage levels: nonempty, unique, and strictly
/// between zero and one.
pub fn validate_levels(levels: &[f64]) -> Result<(), EvaluationError> {
    if levels.is_empty()
        || levels
            .iter()
            .any(|l| !(l.is_finite() && *l > 0.0 && *l < 1.0))
    {
        return Err(invalid(
            "levels must be a nonempty sequence strictly between zero and one",
        ));
    }
    for (index, level) in levels.iter().enumerate() {
        if levels[..index].contains(level) {
            return Err(invalid("interval levels must be unique"));
        }
    }
    Ok(())
}

/// One group's coverage and sharpness at one level.
#[derive(Debug, Clone, PartialEq)]
pub struct CalibrationRow {
    pub group: usize,
    /// The nominal coverage of the central interval.
    pub level: f64,
    /// Outcomes counted: targets in the group with an observed outcome.
    pub count: usize,
    /// Outcomes inside their interval, bounds included.
    pub hits: usize,
    /// `hits / count`, or `NaN` when `count` is zero.
    pub coverage: f64,
    /// The 95% Wilson interval for `coverage`; see [`wilson_interval`].
    pub coverage_lower: f64,
    pub coverage_upper: f64,
    /// The mean width of the intervals over the same outcomes.
    pub mean_width: f64,
}

/// Empirical coverage and mean width of central predictive intervals,
/// accumulated over forecasts and reported per group and level.
///
/// A group is whatever the caller labels it: one horizon step, for example,
/// or one group for everything. Only targets with an observed outcome count
/// towards coverage and width alike, so both describe the same forecasts.
#[derive(Debug, Clone)]
pub struct IntervalCalibration {
    levels: Vec<f64>,
    counts: Vec<usize>,
    /// Indexed `group * levels + level`.
    hits: Vec<usize>,
    /// The `(lower, upper)` bounds of every counted interval.
    bounds: Vec<Vec<(f64, f64)>>,
}

impl IntervalCalibration {
    pub fn new(levels: &[f64], groups: usize) -> Result<Self, EvaluationError> {
        validate_levels(levels)?;
        if groups == 0 {
            return Err(invalid("there must be at least one group"));
        }
        Ok(Self {
            levels: levels.to_vec(),
            counts: vec![0; groups],
            hits: vec![0; groups * levels.len()],
            bounds: vec![Vec::new(); groups * levels.len()],
        })
    }

    /// Count one forecast's targets. `group[t]` labels target `t`. Nothing is
    /// counted if any argument is invalid.
    pub fn add(
        &mut self,
        draws: &TargetDraws,
        actual: &[f64],
        group: &[usize],
    ) -> Result<(), EvaluationError> {
        draws.check_outcomes(actual)?;
        check_groups(group, draws.len(), self.counts.len())?;
        let levels = self.levels.len();
        for (target, (&y, &g)) in actual.iter().zip(group).enumerate() {
            if y.is_nan() {
                continue;
            }
            self.counts[g] += 1;
            for (index, &level) in self.levels.iter().enumerate() {
                let (lower, upper) = draws.central_interval(target, level);
                let cell = g * levels + index;
                self.hits[cell] += usize::from(lower <= y && y <= upper);
                self.bounds[cell].push((lower, upper));
            }
        }
        Ok(())
    }

    /// One row per group and level, group-major.
    pub fn rows(&self) -> Vec<CalibrationRow> {
        let levels = self.levels.len();
        (0..self.counts.len())
            .flat_map(|group| {
                self.levels.iter().enumerate().map(move |(index, &level)| {
                    let cell = group * levels + index;
                    let (count, hits) = (self.counts[group], self.hits[cell]);
                    let (coverage_lower, coverage_upper) = wilson_interval(hits, count);
                    CalibrationRow {
                        group,
                        level,
                        count,
                        hits,
                        coverage: if count == 0 {
                            f64::NAN
                        } else {
                            hits as f64 / count as f64
                        },
                        coverage_lower,
                        coverage_upper,
                        mean_width: mean_distance(&self.bounds[cell]),
                    }
                })
            })
            .collect()
    }
}

/// The mean of nonnegative values, `NaN` for none. Finite values cannot
/// overflow into an infinite mean; an infinite value gives an infinite mean.
fn mean_of_nonnegative(values: &[f64]) -> f64 {
    let scale = values.iter().copied().fold(0.0, f64::max);
    if values.is_empty() {
        return f64::NAN;
    }
    if scale == 0.0 || scale.is_infinite() {
        return scale;
    }
    values.iter().map(|v| v / scale).sum::<f64>() / values.len() as f64 * scale
}

/// The mean of `b - a` over pairs of finite values, which overflows only when
/// the mean itself is unrepresentable: if any difference overflows, halves
/// are averaged and doubled. Halving every value would instead lose the last
/// bit of subnormal ones.
fn mean_distance(pairs: &[(f64, f64)]) -> f64 {
    if pairs.iter().all(|(a, b)| (b - a).is_finite()) {
        let distances: Vec<f64> = pairs.iter().map(|(a, b)| (b - a).abs()).collect();
        mean_of_nonnegative(&distances)
    } else {
        let halves: Vec<f64> = pairs
            .iter()
            .map(|(a, b)| (b * 0.5 - a * 0.5).abs())
            .collect();
        2.0 * mean_of_nonnegative(&halves)
    }
}

/// The in-sample MASE scale: the mean absolute difference between
/// observations `period` steps apart, over pairs where both are observed.
///
/// Returns `None` when no such pair exists. `NaN` marks a missing
/// observation; an infinite one is refused.
pub fn mase_scale(training: &[f64], period: usize) -> Result<Option<f64>, EvaluationError> {
    if period == 0 {
        return Err(invalid("period must be a positive integer"));
    }
    if training.iter().any(|y| y.is_infinite()) {
        return Err(invalid("training observations must be finite or NaN"));
    }
    let pairs: Vec<(f64, f64)> = training
        .iter()
        .zip(training.iter().skip(period))
        .filter(|(a, b)| !a.is_nan() && !b.is_nan())
        .map(|(&a, &b)| (a, b))
        .collect();
    Ok((!pairs.is_empty()).then(|| mean_distance(&pairs)))
}

/// Point-forecast accuracy for one group.
#[derive(Debug, Clone, PartialEq)]
pub struct AccuracyRow {
    pub group: usize,
    /// Outcomes counted: targets in the group with an observed outcome.
    pub count: usize,
    /// Mean absolute error.
    pub mae: f64,
    /// Root mean squared error.
    pub rmse: f64,
    /// Weighted absolute percentage error, `sum |e| / sum |y|`.
    pub wape: f64,
    /// Mean absolute scaled error: the mean of `|e| / scale` with each
    /// target's in-sample scale, or `NaN` if any counted target had none.
    pub mase: f64,
}

/// The accuracy of point forecasts, accumulated over forecasts and reported
/// per group.
///
/// A zero denominator follows one rule for WAPE and for each MASE term: zero
/// error over it is zero, and any other error over it is infinite, since the
/// relative error is then unbounded. `NaN` is reserved for missing.
#[derive(Debug, Clone)]
pub struct PointAccuracy {
    /// `(|error|, |actual|, scale)` for each counted target, per group.
    entries: Vec<Vec<(f64, f64, f64)>>,
}

impl PointAccuracy {
    pub fn new(groups: usize) -> Result<Self, EvaluationError> {
        if groups == 0 {
            return Err(invalid("there must be at least one group"));
        }
        Ok(Self {
            entries: vec![Vec::new(); groups],
        })
    }

    /// Count one forecast's targets. `error[t]` is target `t`'s point-forecast
    /// error, as from [`TargetDraws::point_errors`], and may be infinite only
    /// if the error is unrepresentable. `scale[t]` is its MASE scale, `NaN`
    /// when it has none. Nothing is counted if any argument is invalid.
    pub fn add(
        &mut self,
        error: &[f64],
        actual: &[f64],
        scale: &[f64],
        group: &[usize],
    ) -> Result<(), EvaluationError> {
        let targets = error.len();
        if actual.len() != targets || scale.len() != targets {
            return Err(invalid(
                "errors, outcomes and scales must have one value per target",
            ));
        }
        check_groups(group, targets, self.entries.len())?;
        if actual.iter().any(|y| y.is_infinite()) {
            return Err(invalid(
                "actual values must be finite or NaN for missing outcomes",
            ));
        }
        if error
            .iter()
            .zip(actual)
            .any(|(e, y)| e.is_nan() && !y.is_nan())
        {
            return Err(invalid("an observed outcome needs a point-forecast error"));
        }
        if scale.iter().any(|s| *s < 0.0) {
            return Err(invalid("MASE scales must be nonnegative, or NaN"));
        }
        for t in 0..targets {
            if !actual[t].is_nan() {
                self.entries[group[t]].push((error[t].abs(), actual[t].abs(), scale[t]));
            }
        }
        Ok(())
    }

    pub fn rows(&self) -> Vec<AccuracyRow> {
        self.entries
            .iter()
            .enumerate()
            .map(|(group, entries)| accuracy_row(group, entries))
            .collect()
    }
}

fn accuracy_row(group: usize, entries: &[(f64, f64, f64)]) -> AccuracyRow {
    let count = entries.len();
    if count == 0 {
        return AccuracyRow {
            group,
            count,
            mae: f64::NAN,
            rmse: f64::NAN,
            wape: f64::NAN,
            mase: f64::NAN,
        };
    }
    // Every sum below is taken in units of the largest term, so that no
    // partial sum overflows and no mean of tiny values underflows before a
    // ratio is formed.
    let errors: Vec<f64> = entries.iter().map(|&(e, _, _)| e).collect();
    let largest = errors.iter().copied().fold(0.0, f64::max);
    let unrepresentable = largest.is_infinite();
    let rmse = if largest == 0.0 || unrepresentable {
        largest
    } else {
        let mean_square = errors.iter().map(|e| (e / largest).powi(2)).sum::<f64>() / count as f64;
        largest * mean_square.sqrt()
    };
    let wape = if unrepresentable {
        f64::INFINITY
    } else {
        let unit = entries.iter().fold(largest, |unit, &(_, y, _)| unit.max(y));
        if unit == 0.0 {
            0.0
        } else {
            ratio(
                errors.iter().map(|e| e / unit).sum(),
                entries.iter().map(|&(_, y, _)| y / unit).sum(),
            )
        }
    };
    AccuracyRow {
        group,
        count,
        mae: mean_of_nonnegative(&errors),
        rmse,
        wape,
        mase: mean_scaled_error(entries, largest),
    }
}

/// The mean of `|e| / scale`: `NaN` if a scale is missing, and infinite if a
/// term is, including a nonzero error over a zero scale.
fn mean_scaled_error(entries: &[(f64, f64, f64)], largest: f64) -> f64 {
    if entries.iter().any(|&(_, _, s)| s.is_nan()) {
        return f64::NAN;
    }
    let terms: Vec<f64> = entries.iter().map(|&(e, _, s)| ratio(e, s)).collect();
    if terms.iter().all(|t| t.is_finite()) {
        return mean_of_nonnegative(&terms);
    }
    if largest.is_infinite() || entries.iter().any(|&(e, _, s)| s == 0.0 && e > 0.0) {
        return f64::INFINITY;
    }
    // A term overflowed although its error and scale are finite: take the
    // terms in units of the largest error, where they can still be summed.
    let scaled: Vec<f64> = entries
        .iter()
        .map(|&(e, _, s)| ratio(e / largest, s))
        .collect();
    if scaled.iter().any(|t| t.is_infinite()) {
        f64::INFINITY
    } else {
        largest * mean_of_nonnegative(&scaled)
    }
}

/// `numerator / denominator` for nonnegative values, with zero over zero
/// taken as zero.
fn ratio(numerator: f64, denominator: f64) -> f64 {
    if numerator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn draws(columns: &[&[f64]]) -> TargetDraws {
        TargetDraws::new(columns.iter().map(|c| c.to_vec()).collect()).unwrap()
    }

    #[test]
    fn rank_counts_ties_on_both_sides() {
        let rank = OutcomeRank::new(&[0.0, 1.0, 1.0, 1.0, 2.0], 1.0);
        assert_eq!((rank.below, rank.at_or_below, rank.draws), (1, 4, 5));
        assert_eq!(rank.pit(), 0.8);
        // (1 + u * 4) / 6 spans the four positions the outcome can take.
        assert_eq!(rank.randomized_pit(0.0), 1.0 / 6.0);
        assert_eq!(rank.randomized_pit(0.5), 3.0 / 6.0);
        let above = OutcomeRank::new(&[0.0, 1.0], 5.0);
        assert_eq!(above.pit(), 1.0);
        assert!(above.randomized_pit(0.999_999) < 1.0);
    }

    #[test]
    fn randomized_pit_is_uniform_for_exchangeable_ties() {
        // An outcome and 3 draws exchangeable on {0, 1}: each of the four
        // values is equally likely to be the outcome. Pushing a fine grid of
        // uniforms through `randomized_pit` for every case, the share of
        // values at or below q must be q, up to the grid's resolution.
        let values = [0.0, 0.0, 1.0, 1.0];
        let grid = 100_000;
        let mut pits = Vec::with_capacity(4 * grid);
        for outcome_index in 0..4 {
            let mut rest: Vec<f64> = values.to_vec();
            let outcome = rest.remove(outcome_index);
            let rank = OutcomeRank::new(&rest, outcome);
            pits.extend((0..grid).map(|j| rank.randomized_pit((j as f64 + 0.5) / grid as f64)));
        }
        for q in [0.1, 0.3, 0.5, 0.77, 0.9] {
            let share = pits.iter().filter(|&&p| p <= q).count() as f64 / pits.len() as f64;
            assert!((share - q).abs() < 1e-4, "{q}: {share}");
        }
        // The plain PIT of the same cases is not uniform: it takes only the
        // values 1/3 and 1.
        let plain: Vec<f64> = (0..4)
            .map(|i| {
                let mut rest = values.to_vec();
                let outcome = rest.remove(i);
                OutcomeRank::new(&rest, outcome).pit()
            })
            .collect();
        assert_eq!(plain, vec![1.0 / 3.0, 1.0 / 3.0, 1.0, 1.0]);
    }

    #[test]
    fn missing_outcomes_have_no_pit_and_do_not_shift_the_stream() {
        let d = draws(&[&[0.0, 1.0, 2.0], &[0.0, 1.0, 2.0]]);
        let full = d.randomized_pit(&[1.0, 1.0], 7, 3).unwrap();
        let partial = d.randomized_pit(&[f64::NAN, 1.0], 7, 3).unwrap();
        assert!(partial[0].is_nan());
        assert_eq!(full[1], partial[1]);
        assert_ne!(full, d.randomized_pit(&[1.0, 1.0], 7, 4).unwrap());
        assert!(d.pit(&[f64::INFINITY, 0.0]).is_err());
        assert!(d.pit(&[0.0]).is_err());
    }

    #[test]
    fn draws_are_validated() {
        assert!(TargetDraws::new(vec![]).is_err());
        assert!(TargetDraws::new(vec![vec![]]).is_err());
        assert!(TargetDraws::new(vec![vec![1.0], vec![1.0, 2.0]]).is_err());
        assert!(TargetDraws::new(vec![vec![f64::NAN]]).is_err());
    }

    #[test]
    fn histogram_bins_include_the_upper_edge() {
        let counts = pit_histogram(&[0.0, 0.1, 0.5, 0.999, 1.0, f64::NAN], 2).unwrap();
        assert_eq!(counts, vec![2, 3]);
        assert!(pit_histogram(&[1.5], 2).is_err());
        assert!(pit_histogram(&[0.5], 0).is_err());
        // Refused, not an attempted allocation.
        assert!(pit_histogram(&[0.5], usize::MAX).is_err());
    }

    #[test]
    fn histogram_edges_are_exact_where_the_product_rounds() {
        // 0.8999... * 10 rounds to 9.0, but the value is below the edge 0.9.
        let below = 0.9f64.next_down();
        assert_eq!((below * 10.0) as usize, 9);
        let counts = pit_histogram(&[below, 0.9, 0.3, 0.3f64.next_down()], 10).unwrap();
        assert_eq!(counts, vec![0, 0, 1, 1, 0, 0, 0, 0, 1, 1]);
    }

    #[test]
    fn wilson_interval_matches_the_closed_form() {
        // 8 of 10: centre (0.8 + z^2/20) / (1 + z^2/10).
        let (lower, upper) = wilson_interval(8, 10);
        assert!((lower - 0.490_162_2).abs() < 1e-6, "{lower}");
        assert!((upper - 0.943_318_5).abs() < 1e-6, "{upper}");
        // All hits: the interval is [n / (n + z^2), 1], not a point at 1.
        let (lower, upper) = wilson_interval(10, 10);
        assert!(
            (lower - 10.0 / (10.0 + WILSON_Z * WILSON_Z)).abs() < 1e-15,
            "{lower}"
        );
        assert!((upper - 1.0).abs() < 1e-15, "{upper}");
        assert!(wilson_interval(0, 0).0.is_nan());
    }

    #[test]
    fn calibration_counts_inclusive_bounds_and_skips_missing() {
        // Draws 0..=10: the 80% interval is [1, 9].
        let column: Vec<f64> = (0..=10).map(f64::from).collect();
        let d = TargetDraws::new(vec![column.clone(), column.clone(), column]).unwrap();
        let mut table = IntervalCalibration::new(&[0.8], 2).unwrap();
        table.add(&d, &[9.0, 9.5, f64::NAN], &[0, 0, 1]).unwrap();
        let rows = table.rows();
        assert_eq!((rows[0].count, rows[0].hits, rows[0].coverage), (2, 1, 0.5));
        assert_eq!(rows[0].mean_width, 8.0);
        assert_eq!(rows[1].count, 0);
        assert!(rows[1].coverage.is_nan() && rows[1].mean_width.is_nan());
        // A bad group label changes nothing.
        assert!(table.add(&d, &[0.0, 0.0, 0.0], &[0, 0, 2]).is_err());
        assert_eq!(table.rows()[0].count, 2);
        assert!(IntervalCalibration::new(&[0.5, 0.5], 1).is_err());
        assert!(IntervalCalibration::new(&[1.0], 1).is_err());
    }

    #[test]
    fn accuracy_matches_hand_computed_values() {
        // Errors 2, 0, -4 against outcomes 1, 5, 6.
        let mut accuracy = PointAccuracy::new(1).unwrap();
        accuracy
            .add(
                &[2.0, 0.0, -4.0],
                &[1.0, 5.0, 6.0],
                &[2.0, 2.0, 4.0],
                &[0, 0, 0],
            )
            .unwrap();
        let row = &accuracy.rows()[0];
        assert_eq!(row.count, 3);
        assert!((row.mae - 2.0).abs() < 1e-15);
        assert!((row.rmse - (20.0f64 / 3.0).sqrt()).abs() < 1e-15);
        assert!((row.wape - 6.0 / 12.0).abs() < 1e-15);
        assert!((row.mase - (1.0 + 0.0 + 1.0) / 3.0).abs() < 1e-15);
    }

    #[test]
    fn zero_denominators_and_overflow() {
        let mut exact = PointAccuracy::new(1).unwrap();
        exact.add(&[0.0], &[0.0], &[0.0], &[0]).unwrap();
        let row = &exact.rows()[0];
        assert_eq!((row.wape, row.mase), (0.0, 0.0));
        let mut wrong = PointAccuracy::new(1).unwrap();
        wrong.add(&[1.0], &[0.0], &[0.0], &[0]).unwrap();
        assert!(wrong.rows()[0].wape.is_infinite() && wrong.rows()[0].mase.is_infinite());
        let mut unscaled = PointAccuracy::new(1).unwrap();
        unscaled
            .add(&[1.0, 1.0], &[0.0, 2.0], &[1.0, f64::NAN], &[0, 0])
            .unwrap();
        assert!(unscaled.rows()[0].mase.is_nan());
        let mut huge = PointAccuracy::new(1).unwrap();
        huge.add(&[1e308], &[-5e307], &[f64::NAN], &[0]).unwrap();
        let row = &huge.rows()[0];
        assert_eq!((row.mae, row.rmse, row.wape), (1e308, 1e308, 2.0));
        assert!(PointAccuracy::new(1).unwrap().rows()[0].mae.is_nan());
        // A term 1e308 / 0.5 overflows although the mean, 1e308, does not.
        let mut steep = PointAccuracy::new(1).unwrap();
        steep
            .add(&[1e308, 0.0], &[1.0, 1.0], &[0.5, 0.5], &[0, 0])
            .unwrap();
        assert_eq!(steep.rows()[0].mase, 1e308);
        let mut unbounded = PointAccuracy::new(1).unwrap();
        unbounded
            .add(&[f64::INFINITY], &[1.0], &[1.0], &[0])
            .unwrap();
        let row = &unbounded.rows()[0];
        assert!(row.mae.is_infinite() && row.rmse.is_infinite() && row.wape.is_infinite());
        assert!(PointAccuracy::new(1)
            .unwrap()
            .add(&[f64::NAN], &[1.0], &[1.0], &[0])
            .is_err());
    }

    #[test]
    fn ratios_of_subnormal_values_are_not_lost_to_underflow() {
        let tiny = f64::from_bits(1);
        // sum |e| / sum |y| = tiny / tiny, although each mean underflows.
        let mut wape = PointAccuracy::new(1).unwrap();
        wape.add(
            &[-tiny, 0.0, 0.0],
            &[tiny, 0.0, 0.0],
            &[tiny; 3],
            &[0, 0, 0],
        )
        .unwrap();
        assert_eq!(wape.rows()[0].wape, 1.0);
        assert_eq!(mase_scale(&[0.0, tiny], 1).unwrap(), Some(tiny));
    }

    #[test]
    fn point_errors_subtract_before_summarising() {
        // The mean draw, 1e16 + 1, is not representable; the mean error is.
        let d = draws(&[&[1e16, 1e16 + 2.0]]);
        assert_eq!(d.point_errors(&[1e16], Point::Mean).unwrap(), vec![1.0]);
        assert_eq!(d.point_errors(&[1e16], Point::Median).unwrap(), vec![1.0]);
        let skewed = draws(&[&[0.0, 0.0, 9.0]]);
        assert_eq!(skewed.point_errors(&[1.0], Point::Mean).unwrap(), vec![2.0]);
        assert_eq!(
            skewed.point_errors(&[1.0], Point::Median).unwrap(),
            vec![-1.0]
        );
        let wide = draws(&[&[1e308, 1e308]]);
        assert_eq!(
            wide.point_errors(&[-1e308], Point::Mean).unwrap()[0],
            f64::INFINITY
        );
        assert!(wide.point_errors(&[f64::NAN], Point::Mean).unwrap()[0].is_nan());
    }

    #[test]
    fn mean_width_survives_an_overflowing_interval() {
        let d = draws(&[&[-1e308, 1e308], &[0.0, 0.0]]);
        // The 99% interval is [-9.9e307, 9.9e307], wider than the largest
        // double; the mean of it and a zero width is still representable.
        let mut table = IntervalCalibration::new(&[0.99], 1).unwrap();
        table.add(&d, &[0.0, 0.0], &[0, 0]).unwrap();
        let mean = table.rows()[0].mean_width;
        assert!((mean - 9.9e307).abs() <= 1e-15 * 9.9e307, "{mean}");
    }

    #[test]
    fn mase_scale_uses_pairs_a_period_apart() {
        let y = [1.0, 2.0, 4.0, f64::NAN, 8.0];
        // Pairs one apart: (1,2), (2,4); (4,NaN) and (NaN,8) are skipped.
        assert_eq!(mase_scale(&y, 1).unwrap(), Some(1.5));
        // Two apart: (1,4), (4,8); (2,NaN) skipped.
        assert_eq!(mase_scale(&y, 2).unwrap(), Some(3.5));
        assert_eq!(mase_scale(&y, 5).unwrap(), None);
        // The difference 2e308 is unrepresentable, so the scale is infinite
        // and a finite error scaled by it is zero.
        let infinite = mase_scale(&[-1e308, 1e308], 1).unwrap();
        assert_eq!(infinite, Some(f64::INFINITY));
        let mut accuracy = PointAccuracy::new(1).unwrap();
        accuracy
            .add(&[1.0], &[0.0], &[infinite.unwrap()], &[0])
            .unwrap();
        assert_eq!(accuracy.rows()[0].mase, 0.0);
        assert!(mase_scale(&y, 0).is_err());
    }
}
