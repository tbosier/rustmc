//! Regression columns on an integer calendar: known-date interventions,
//! piecewise-linear trend changepoints, Fourier terms, and named columns the
//! caller supplies.
//!
//! Every generated column is a function of the calendar position `t` alone.
//! A [`TimeDesign`] fixes the training window `start..start + length`, and its
//! future rows are the same functions evaluated at `start + length..`, so they
//! continue the training rows exactly. There is no second `start` for a caller
//! to get wrong.
//!
//! The columns are ordinary regressors. Nothing here estimates where a break
//! happened: a changepoint basis places candidate hinges on a grid, and the
//! model's coefficient prior decides how much each one moves.
use crate::bayesian_regression::{fourier_design, fourier_width};
use crate::forecast_common::{checked_value_count, AllocationLimitError, MAX_MATERIALIZED_VALUES};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

/// A term, design or row request that cannot be built.
#[derive(Debug, Clone, PartialEq)]
pub enum DesignError {
    /// One term's own arguments are invalid.
    InvalidTerm { term: String, reason: String },
    /// Two terms, or two generated columns, would share a name.
    DuplicateName(String),
    /// The design as a whole is invalid, such as an empty term list or a
    /// window past the end of the integer calendar.
    InvalidDesign(String),
    /// Values supplied for a regressor term are missing, unexpected, the wrong
    /// length, or not finite.
    RegressorValues { term: String, reason: String },
    /// The rows would exceed the materialization limit.
    TooLarge(AllocationLimitError),
}

impl fmt::Display for DesignError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTerm { term, reason } => write!(f, "design term '{term}': {reason}"),
            Self::DuplicateName(name) => write!(
                f,
                "design name '{name}' is used twice; term names and column names must be unique"
            ),
            Self::InvalidDesign(message) => write!(f, "time design: {message}"),
            Self::RegressorValues { term, reason } => {
                write!(f, "regressor '{term}': {reason}")
            }
            Self::TooLarge(error) => error.fmt(f),
        }
    }
}

impl Error for DesignError {}

impl From<AllocationLimitError> for DesignError {
    fn from(error: AllocationLimitError) -> Self {
        Self::TooLarge(error)
    }
}

/// Where a changepoint basis places its hinges.
#[derive(Debug, Clone, PartialEq)]
pub enum ChangepointPlacement {
    /// These calendar positions, strictly increasing.
    At(Vec<i64>),
    /// `count` evenly spaced positions after `first`, up to
    /// `first + floor(fraction * (last - first))`. `first` and `last` default
    /// to the first and last training positions when the design is built.
    Grid {
        count: usize,
        first: Option<i64>,
        last: Option<i64>,
        fraction: f64,
    },
}

/// What one term contributes.
#[derive(Debug, Clone, PartialEq)]
pub enum TermKind {
    /// Level shift: 1 from `at` onward, 0 before.
    Step { at: i64 },
    /// One-period effect: 1 at `at`, 0 elsewhere.
    Pulse { at: i64 },
    /// Slope change: `max(0, t - at)`, so 0 at `at`, 1 one step later.
    Ramp { at: i64 },
    /// Temporary effect: `rate^(t - at)` from `at` onward, 0 before. The
    /// coefficient is the effect at `at`.
    Decay { at: i64, rate: f64 },
    /// Sine/cosine pairs as in [`fourier_design`].
    Fourier { period: usize, harmonics: usize },
    /// One ramp column `max(0, t - s)` per changepoint `s`.
    Changepoints(ChangepointPlacement),
    /// One column whose values the caller supplies for every row request.
    Regressor,
}

/// A named term of a [`TimeDesign`].
#[derive(Debug, Clone, PartialEq)]
pub struct DesignTerm {
    pub name: String,
    pub kind: TermKind,
}

impl DesignTerm {
    /// A term whose own arguments are valid; a grid's window is checked when
    /// the design is built, because it may default to the training window.
    pub fn new(name: impl Into<String>, kind: TermKind) -> Result<Self, DesignError> {
        let term = Self {
            name: name.into(),
            kind,
        };
        term.validate()?;
        Ok(term)
    }

    pub fn validate(&self) -> Result<(), DesignError> {
        let invalid = |reason: &str| {
            Err(DesignError::InvalidTerm {
                term: self.name.clone(),
                reason: reason.to_string(),
            })
        };
        if self.name.is_empty() {
            return Err(DesignError::InvalidTerm {
                term: String::new(),
                reason: "a term needs a nonempty name".into(),
            });
        }
        match &self.kind {
            TermKind::Decay { rate, .. } if !(rate.is_finite() && *rate > 0.0 && *rate < 1.0) => {
                invalid(
                    "decay rate must lie strictly between 0 and 1; use a pulse for 0 and a \
                     step for 1",
                )
            }
            TermKind::Fourier { period, harmonics } => {
                fourier_width(*period, *harmonics).map(|_| ()).or_else(|_| {
                    invalid(
                        "period must be at least two; harmonics must be between one and \
                         floor(period/2)",
                    )
                })
            }
            TermKind::Changepoints(ChangepointPlacement::At(positions)) => {
                if positions.is_empty() {
                    invalid("give at least one changepoint position")
                } else if positions.windows(2).any(|pair| pair[0] >= pair[1]) {
                    invalid("changepoint positions must be strictly increasing")
                } else {
                    Ok(())
                }
            }
            TermKind::Changepoints(ChangepointPlacement::Grid {
                count,
                first,
                last,
                fraction,
            }) => {
                if *count == 0 {
                    invalid("changepoint count must be at least one")
                } else if !(fraction.is_finite() && *fraction > 0.0 && *fraction <= 1.0) {
                    invalid("fraction must lie in (0, 1]")
                } else if matches!((first, last), (Some(a), Some(b)) if a >= b) {
                    invalid("first must be before last")
                } else {
                    Ok(())
                }
            }
            _ => Ok(()),
        }
    }
}

/// Evenly spaced changepoints after `first`, up to
/// `end = first + floor(fraction * (last - first))`, which is itself the last
/// one. `first` is excluded: a hinge there is a plain linear trend over the
/// window. Positions are `first + round(j * (end - first) / count)` for
/// `j = 1..=count`, rounding halves up, computed in exact integer arithmetic.
fn changepoint_grid(
    first: i64,
    last: i64,
    count: usize,
    fraction: f64,
) -> Result<Vec<i64>, String> {
    if first >= last {
        return Err(format!(
            "the changepoint window needs first < last; got first={first}, last={last}"
        ));
    }
    let span = i128::from(last) - i128::from(first);
    let window = floor_fraction_of(fraction, span);
    let count_wide = count as i128;
    if window < count_wide {
        return Err(format!(
            "{count} changepoints need at least {count} positions after first={first}, but \
             the window first..first + floor(fraction * (last - first)) has {window}; reduce \
             count, raise fraction or widen first..last"
        ));
    }
    Ok((1..=count_wide)
        .map(|j| {
            let offset = (2 * j * window + count_wide) / (2 * count_wide);
            // Between `first` and `last`, so within i64.
            (i128::from(first) + offset) as i64
        })
        .collect())
}

/// `floor(fraction * span)` in exact arithmetic, for `0 < fraction <= 1` and
/// `0 < span < 2^64`. A float64 product would round a span above 2^53 before
/// the floor is taken, moving the grid's end away from `last`.
fn floor_fraction_of(fraction: f64, span: i128) -> i128 {
    // fraction = mantissa * 2^exponent exactly.
    let bits = fraction.to_bits();
    let biased = ((bits >> 52) & 0x7ff) as i32;
    let fraction_bits = (bits & ((1u64 << 52) - 1)) as i128;
    let (mantissa, exponent) = if biased == 0 {
        (fraction_bits, -1074)
    } else {
        (fraction_bits | (1i128 << 52), biased - 1075)
    };
    // mantissa < 2^53 and span < 2^64, so the product fits in i128; the
    // exponent is negative because fraction <= 1.
    let shift = -exponent;
    if shift >= 127 {
        0
    } else {
        (mantissa * span) >> shift
    }
}

/// Number of columns a term will produce, before any are built.
fn planned_width(term: &DesignTerm) -> usize {
    match &term.kind {
        TermKind::Fourier { period, harmonics } => fourier_width(*period, *harmonics).unwrap_or(0),
        TermKind::Changepoints(ChangepointPlacement::At(positions)) => positions.len(),
        TermKind::Changepoints(ChangepointPlacement::Grid { count, .. }) => *count,
        _ => 1,
    }
}

/// A term with its window-dependent arguments resolved.
#[derive(Debug, Clone, PartialEq)]
enum Resolved {
    Step(i64),
    Pulse(i64),
    Ramp(i64),
    Decay(i64, f64),
    Fourier(usize, usize),
    Hinges(Vec<i64>),
    Regressor,
}

/// Level-shift value at calendar position `t`.
pub fn step_value(t: i64, at: i64) -> f64 {
    if t >= at {
        1.0
    } else {
        0.0
    }
}

/// Pulse value at calendar position `t`.
pub fn pulse_value(t: i64, at: i64) -> f64 {
    if t == at {
        1.0
    } else {
        0.0
    }
}

/// Ramp value `max(0, t - at)` at calendar position `t`.
pub fn ramp_value(t: i64, at: i64) -> f64 {
    (i128::from(t) - i128::from(at)).max(0) as f64
}

/// Geometric-decay value `rate^(t - at)` from `at` onward, 0 before.
pub fn decay_value(t: i64, at: i64, rate: f64) -> f64 {
    let elapsed = i128::from(t) - i128::from(at);
    if elapsed < 0 {
        0.0
    } else {
        rate.powf(elapsed as f64)
    }
}

/// Named regression columns over a training window and any horizon after it.
#[derive(Debug, Clone, PartialEq)]
pub struct TimeDesign {
    terms: Vec<DesignTerm>,
    resolved: Vec<Resolved>,
    start: i64,
    length: usize,
    names: Vec<String>,
    widths: Vec<usize>,
}

impl TimeDesign {
    /// A design whose training rows are calendar positions
    /// `start..start + length`.
    pub fn new(terms: Vec<DesignTerm>, start: i64, length: usize) -> Result<Self, DesignError> {
        if terms.is_empty() {
            return Err(DesignError::InvalidDesign("give at least one term".into()));
        }
        if length == 0 {
            return Err(DesignError::InvalidDesign(
                "the training length must be at least one".into(),
            ));
        }
        let end = i64::try_from(length)
            .ok()
            .and_then(|length| start.checked_add(length))
            .ok_or_else(|| {
                DesignError::InvalidDesign(
                    "start + length must not exceed the 64-bit calendar".into(),
                )
            })?;
        let last_training = end - 1;
        // Refuse an oversized design before building any column name or grid.
        for term in &terms {
            term.validate()?;
        }
        let planned = terms
            .iter()
            .try_fold(0usize, |total, term| total.checked_add(planned_width(term)));
        checked_value_count(
            "time design training rows",
            &[length, planned.unwrap_or(usize::MAX)],
            MAX_MATERIALIZED_VALUES,
        )?;
        let mut resolved = Vec::with_capacity(terms.len());
        let mut names = Vec::new();
        let mut widths = Vec::with_capacity(terms.len());
        let mut term_names = BTreeSet::new();
        for term in &terms {
            if !term_names.insert(term.name.as_str()) {
                return Err(DesignError::DuplicateName(term.name.clone()));
            }
            let (item, columns) = match &term.kind {
                TermKind::Step { at } => (Resolved::Step(*at), vec![term.name.clone()]),
                TermKind::Pulse { at } => (Resolved::Pulse(*at), vec![term.name.clone()]),
                TermKind::Ramp { at } => (Resolved::Ramp(*at), vec![term.name.clone()]),
                TermKind::Decay { at, rate } => {
                    (Resolved::Decay(*at, *rate), vec![term.name.clone()])
                }
                TermKind::Regressor => (Resolved::Regressor, vec![term.name.clone()]),
                TermKind::Fourier { period, harmonics } => {
                    let mut columns = Vec::new();
                    for k in 1..=*harmonics {
                        if 2 * k != *period {
                            columns.push(format!("{}_sin{k}", term.name));
                        }
                        columns.push(format!("{}_cos{k}", term.name));
                    }
                    (Resolved::Fourier(*period, *harmonics), columns)
                }
                TermKind::Changepoints(placement) => {
                    let positions = match placement {
                        ChangepointPlacement::At(positions) => positions.clone(),
                        ChangepointPlacement::Grid {
                            count,
                            first,
                            last,
                            fraction,
                        } => changepoint_grid(
                            first.unwrap_or(start),
                            last.unwrap_or(last_training),
                            *count,
                            *fraction,
                        )
                        .map_err(|reason| DesignError::InvalidTerm {
                            term: term.name.clone(),
                            reason,
                        })?,
                    };
                    let columns = positions
                        .iter()
                        .map(|position| format!("{}@{position}", term.name))
                        .collect();
                    (Resolved::Hinges(positions), columns)
                }
            };
            widths.push(columns.len());
            names.extend(columns);
            resolved.push(item);
        }
        let mut seen = BTreeSet::new();
        for name in &names {
            if !seen.insert(name.as_str()) {
                return Err(DesignError::DuplicateName(name.clone()));
            }
        }
        Ok(Self {
            terms,
            resolved,
            start,
            length,
            names,
            widths,
        })
    }

    pub fn terms(&self) -> &[DesignTerm] {
        &self.terms
    }

    /// Calendar position of the first training row.
    pub fn start(&self) -> i64 {
        self.start
    }

    /// Number of training rows.
    pub fn length(&self) -> usize {
        self.length
    }

    /// Column names in row order.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    pub fn width(&self) -> usize {
        self.names.len()
    }

    /// Each term's name, first column and column count, in row order.
    pub fn term_columns(&self) -> Vec<(&str, usize, usize)> {
        let mut offset = 0;
        self.terms
            .iter()
            .zip(&self.widths)
            .map(|(term, &width)| {
                let item = (term.name.as_str(), offset, width);
                offset += width;
                item
            })
            .collect()
    }

    /// The resolved positions of each changepoint term, by term name.
    pub fn changepoints(&self) -> BTreeMap<&str, &[i64]> {
        self.terms
            .iter()
            .zip(&self.resolved)
            .filter_map(|(term, item)| match item {
                Resolved::Hinges(positions) => Some((term.name.as_str(), positions.as_slice())),
                _ => None,
            })
            .collect()
    }

    /// Names of the regressor terms, whose values every row request supplies.
    pub fn regressor_names(&self) -> Vec<&str> {
        self.terms
            .iter()
            .filter(|term| term.kind == TermKind::Regressor)
            .map(|term| term.name.as_str())
            .collect()
    }

    /// The `length` training rows. `regressors` holds one finite value per
    /// training row for every regressor term, and nothing else.
    pub fn training_rows(
        &self,
        regressors: &BTreeMap<String, Vec<f64>>,
    ) -> Result<Vec<Vec<f64>>, DesignError> {
        self.rows(self.start, self.length, regressors, "training")
    }

    /// The `steps` rows immediately after the training window: calendar
    /// positions `start + length..start + length + steps`. `regressors` holds
    /// one finite value per future row for every regressor term.
    pub fn future_rows(
        &self,
        steps: usize,
        regressors: &BTreeMap<String, Vec<f64>>,
    ) -> Result<Vec<Vec<f64>>, DesignError> {
        // `new` checked that this sum fits.
        let first = self.start + self.length as i64;
        self.rows(first, steps, regressors, "future")
    }

    fn rows(
        &self,
        first: i64,
        count: usize,
        regressors: &BTreeMap<String, Vec<f64>>,
        window: &str,
    ) -> Result<Vec<Vec<f64>>, DesignError> {
        let width = self.width();
        checked_value_count("time design rows", &[count, width], MAX_MATERIALIZED_VALUES)?;
        // The last requested position is `first + count - 1`.
        if count > 0
            && i64::try_from(count - 1)
                .ok()
                .and_then(|offset| first.checked_add(offset))
                .is_none()
        {
            return Err(DesignError::InvalidDesign(format!(
                "{count} {window} rows from position {first} run past the 64-bit calendar"
            )));
        }
        self.check_regressors(count, regressors, window)?;
        let mut rows = vec![Vec::with_capacity(width); count];
        for (term, item) in self.terms.iter().zip(&self.resolved) {
            match item {
                Resolved::Fourier(period, harmonics) => {
                    let block =
                        fourier_design(count, *period, *harmonics, first).map_err(|error| {
                            DesignError::InvalidTerm {
                                term: term.name.clone(),
                                reason: error.to_string(),
                            }
                        })?;
                    for (row, values) in rows.iter_mut().zip(block) {
                        row.extend(values);
                    }
                }
                Resolved::Regressor => {
                    for (row, value) in rows.iter_mut().zip(&regressors[&term.name]) {
                        row.push(*value);
                    }
                }
                _ => {
                    for (offset, row) in rows.iter_mut().enumerate() {
                        // Checked above to stay within i64.
                        let t = first + offset as i64;
                        match item {
                            Resolved::Step(at) => row.push(step_value(t, *at)),
                            Resolved::Pulse(at) => row.push(pulse_value(t, *at)),
                            Resolved::Ramp(at) => row.push(ramp_value(t, *at)),
                            Resolved::Decay(at, rate) => row.push(decay_value(t, *at, *rate)),
                            Resolved::Hinges(positions) => {
                                row.extend(positions.iter().map(|&s| ramp_value(t, s)));
                            }
                            Resolved::Fourier(..) | Resolved::Regressor => unreachable!(),
                        }
                    }
                }
            }
        }
        Ok(rows)
    }

    fn check_regressors(
        &self,
        count: usize,
        regressors: &BTreeMap<String, Vec<f64>>,
        window: &str,
    ) -> Result<(), DesignError> {
        let expected = self.regressor_names();
        for name in &expected {
            let values = regressors
                .get(*name)
                .ok_or_else(|| DesignError::RegressorValues {
                    term: name.to_string(),
                    reason: format!("no {window} values were given"),
                })?;
            if values.len() != count {
                return Err(DesignError::RegressorValues {
                    term: name.to_string(),
                    reason: format!(
                        "needs {count} {window} values, one per row, but got {}",
                        values.len()
                    ),
                });
            }
            if let Some(index) = values.iter().position(|value| !value.is_finite()) {
                return Err(DesignError::RegressorValues {
                    term: name.to_string(),
                    reason: format!(
                        "{window} value {} at row {index} is not finite",
                        values[index]
                    ),
                });
            }
        }
        if let Some(extra) = regressors
            .keys()
            .find(|key| !expected.contains(&key.as_str()))
        {
            return Err(DesignError::RegressorValues {
                term: extra.clone(),
                reason: format!(
                    "is not a regressor term of this design; its regressor terms are {expected:?}"
                ),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn term(name: &str, kind: TermKind) -> DesignTerm {
        DesignTerm::new(name, kind).unwrap()
    }

    fn none() -> BTreeMap<String, Vec<f64>> {
        BTreeMap::new()
    }

    fn column(design: &TimeDesign, rows: &[Vec<f64>], name: &str) -> Vec<f64> {
        let index = design.names().iter().position(|n| n == name).unwrap();
        rows.iter().map(|row| row[index]).collect()
    }

    #[test]
    fn intervention_columns_match_their_definitions() {
        let design = TimeDesign::new(
            vec![
                term("step", TermKind::Step { at: 3 }),
                term("pulse", TermKind::Pulse { at: 3 }),
                term("ramp", TermKind::Ramp { at: 3 }),
                term("decay", TermKind::Decay { at: 3, rate: 0.5 }),
            ],
            0,
            6,
        )
        .unwrap();
        assert_eq!(design.names(), ["step", "pulse", "ramp", "decay"]);
        let rows = design.training_rows(&none()).unwrap();
        assert_eq!(
            rows,
            vec![
                vec![0.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0, 0.0],
                vec![1.0, 1.0, 0.0, 1.0],
                vec![1.0, 0.0, 1.0, 0.5],
                vec![1.0, 0.0, 2.0, 0.25],
            ]
        );
        assert_eq!(
            design.future_rows(2, &none()).unwrap(),
            vec![vec![1.0, 0.0, 3.0, 0.125], vec![1.0, 0.0, 4.0, 0.0625]]
        );
    }

    #[test]
    fn edge_positions_before_at_and_after_the_window() {
        // Training positions 10..14, future 14..17.
        let design = TimeDesign::new(
            vec![
                term("before", TermKind::Step { at: 7 }),
                term("at_start", TermKind::Ramp { at: 10 }),
                term("pulse_before", TermKind::Pulse { at: 9 }),
                term("horizon", TermKind::Step { at: 15 }),
                term("horizon_pulse", TermKind::Pulse { at: 14 }),
                term("late", TermKind::Decay { at: 30, rate: 0.5 }),
            ],
            10,
            4,
        )
        .unwrap();
        let rows = design.training_rows(&none()).unwrap();
        assert_eq!(column(&design, &rows, "before"), [1.0; 4]);
        assert_eq!(column(&design, &rows, "at_start"), [0.0, 1.0, 2.0, 3.0]);
        assert_eq!(column(&design, &rows, "pulse_before"), [0.0; 4]);
        assert_eq!(column(&design, &rows, "horizon"), [0.0; 4]);
        assert_eq!(column(&design, &rows, "late"), [0.0; 4]);
        let future = design.future_rows(3, &none()).unwrap();
        assert_eq!(column(&design, &future, "before"), [1.0; 3]);
        assert_eq!(column(&design, &future, "at_start"), [4.0, 5.0, 6.0]);
        assert_eq!(column(&design, &future, "horizon"), [0.0, 1.0, 1.0]);
        assert_eq!(column(&design, &future, "horizon_pulse"), [1.0, 0.0, 0.0]);
        // Ramps before the window keep counting from their own position.
        assert_eq!(ramp_value(10, 7), 3.0);
        assert_eq!(decay_value(10, 7, 0.5), 0.125);
        assert_eq!(decay_value(6, 7, 0.5), 0.0);
    }

    #[test]
    fn future_rows_continue_one_long_design_exactly() {
        let terms = vec![
            term("s", TermKind::Step { at: 13 }),
            term("r", TermKind::Ramp { at: 5 }),
            term("d", TermKind::Decay { at: 11, rate: 0.7 }),
            term(
                "annual",
                TermKind::Fourier {
                    period: 12,
                    harmonics: 6,
                },
            ),
            term(
                "trend",
                TermKind::Changepoints(ChangepointPlacement::At(vec![4, 9, 16])),
            ),
        ];
        let long = TimeDesign::new(terms.clone(), -3, 40).unwrap();
        let all = long.training_rows(&none()).unwrap();
        for length in [1, 7, 16, 29] {
            let short = TimeDesign::new(terms.clone(), -3, length).unwrap();
            assert_eq!(short.training_rows(&none()).unwrap(), all[..length]);
            assert_eq!(
                short.future_rows(40 - length, &none()).unwrap(),
                all[length..]
            );
        }
        // Fourier columns are fourier_design at the same positions.
        let fourier = fourier_design(40, 12, 6, -3).unwrap();
        for (row, expected) in all.iter().zip(&fourier) {
            assert_eq!(row[3..14], expected[..]);
        }
        assert_eq!(long.names()[3], "annual_sin1");
        assert_eq!(long.names()[13], "annual_cos6");
        assert_eq!(&long.names()[14..], ["trend@4", "trend@9", "trend@16"]);
        assert_eq!(
            long.term_columns(),
            [
                ("s", 0, 1),
                ("r", 1, 1),
                ("d", 2, 1),
                ("annual", 3, 11),
                ("trend", 14, 3)
            ]
        );
    }

    #[test]
    fn changepoint_grid_follows_the_window_and_fraction() {
        // Window 0..=99 with fraction 0.8 ends at 79; four evenly spaced hinges.
        assert_eq!(changepoint_grid(0, 99, 4, 0.8).unwrap(), [20, 40, 59, 79]);
        assert_eq!(changepoint_grid(0, 10, 5, 1.0).unwrap(), [2, 4, 6, 8, 10]);
        assert_eq!(changepoint_grid(5, 8, 3, 1.0).unwrap(), [6, 7, 8]);
        // Halves round up: 10 * j / 4 = 2.5, 5, 7.5, 10.
        assert_eq!(changepoint_grid(0, 10, 4, 1.0).unwrap(), [3, 5, 8, 10]);
        assert_eq!(changepoint_grid(-20, -10, 2, 0.5).unwrap(), [-17, -15]);
        let error = changepoint_grid(0, 10, 4, 0.3).unwrap_err();
        assert!(error.contains("4 changepoints need at least 4"), "{error}");
        assert!(changepoint_grid(3, 3, 1, 1.0).is_err());
        assert_eq!(
            changepoint_grid(i64::MIN, i64::MAX, 2, 1.0).unwrap(),
            [0, i64::MAX]
        );
        // The window end is exact beyond float64's integer range.
        for last in [(1i64 << 53) + 1, (1i64 << 62) + 511, i64::MAX] {
            assert_eq!(changepoint_grid(0, last, 1, 1.0).unwrap(), [last]);
        }
        assert_eq!(
            changepoint_grid(0, (1i64 << 62) + 3, 1, 0.5).unwrap(),
            [(1i64 << 61) + 1]
        );
        assert_eq!(floor_fraction_of(0.8, 99), 79);
        assert_eq!(floor_fraction_of(f64::MIN_POSITIVE, 1 << 60), 0);

        // Defaults come from the training window 100..=149.
        let design = TimeDesign::new(
            vec![term(
                "trend",
                TermKind::Changepoints(ChangepointPlacement::Grid {
                    count: 5,
                    first: None,
                    last: None,
                    fraction: 0.8,
                }),
            )],
            100,
            50,
        )
        .unwrap();
        let positions = design.changepoints()["trend"].to_vec();
        assert_eq!(positions, changepoint_grid(100, 149, 5, 0.8).unwrap());
        assert_eq!(positions, [108, 116, 123, 131, 139]);
        let rows = design.training_rows(&none()).unwrap();
        for (offset, row) in rows.iter().enumerate() {
            let t = 100 + offset as i64;
            let expected: Vec<f64> = positions.iter().map(|&s| (t - s).max(0) as f64).collect();
            assert_eq!(*row, expected);
        }
    }

    #[test]
    fn regressor_values_are_checked_by_name_length_and_finiteness() {
        let design = TimeDesign::new(
            vec![
                term("promo", TermKind::Regressor),
                term("shift", TermKind::Step { at: 2 }),
            ],
            0,
            3,
        )
        .unwrap();
        let values = |pairs: &[(&str, Vec<f64>)]| -> BTreeMap<String, Vec<f64>> {
            pairs
                .iter()
                .map(|(name, values)| (name.to_string(), values.clone()))
                .collect()
        };
        let rows = design
            .training_rows(&values(&[("promo", vec![0.5, -1.0, 2.0])]))
            .unwrap();
        assert_eq!(
            rows,
            [vec![0.5, 0.0], vec![-1.0, 0.0], vec![2.0, 1.0]].to_vec()
        );
        let future = design
            .future_rows(2, &values(&[("promo", vec![3.0, 4.0])]))
            .unwrap();
        assert_eq!(future, [vec![3.0, 1.0], vec![4.0, 1.0]].to_vec());
        let refusal = |result: Result<Vec<Vec<f64>>, DesignError>| result.unwrap_err().to_string();
        assert!(refusal(design.training_rows(&none())).contains("no training values"));
        assert!(
            refusal(design.future_rows(3, &values(&[("promo", vec![1.0; 2])])))
                .contains("needs 3 future values")
        );
        assert!(
            refusal(design.training_rows(&values(&[("promo", vec![1.0, f64::NAN, 1.0])])))
                .contains("row 1 is not finite")
        );
        assert!(refusal(
            design.training_rows(&values(&[("promo", vec![1.0; 3]), ("price", vec![1.0; 3])]))
        )
        .contains("regressor 'price': is not a regressor term"));
    }

    #[test]
    fn invalid_terms_and_designs_are_named() {
        let message = |error: DesignError| error.to_string();
        assert!(message(
            DesignTerm::new("fade", TermKind::Decay { at: 0, rate: 1.0 }).unwrap_err()
        )
        .starts_with("design term 'fade': decay rate"));
        assert!(DesignTerm::new("fade", TermKind::Decay { at: 0, rate: 0.0 }).is_err());
        assert!(DesignTerm::new(
            "fade",
            TermKind::Decay {
                at: 0,
                rate: f64::NAN
            }
        )
        .is_err());
        assert!(message(
            DesignTerm::new(
                "annual",
                TermKind::Fourier {
                    period: 12,
                    harmonics: 7
                }
            )
            .unwrap_err()
        )
        .contains("'annual'"));
        assert!(message(
            DesignTerm::new(
                "trend",
                TermKind::Changepoints(ChangepointPlacement::At(vec![5, 5]))
            )
            .unwrap_err()
        )
        .contains("strictly increasing"));
        assert!(DesignTerm::new("", TermKind::Step { at: 0 }).is_err());
        for (count, fraction, first, last) in [
            (0, 0.8, None, None),
            (3, 0.0, None, None),
            (3, 1.5, None, None),
            (3, 0.8, Some(4), Some(4)),
        ] {
            let placement = ChangepointPlacement::Grid {
                count,
                first,
                last,
                fraction,
            };
            assert!(DesignTerm::new("trend", TermKind::Changepoints(placement)).is_err());
        }
        let grid = |count| {
            term(
                "trend",
                TermKind::Changepoints(ChangepointPlacement::Grid {
                    count,
                    first: None,
                    last: None,
                    fraction: 0.8,
                }),
            )
        };
        assert!(
            message(TimeDesign::new(vec![grid(9)], 0, 10).unwrap_err()).starts_with(
                "design term 'trend': 9 changepoints need at least 9 positions after first=0"
            )
        );
        assert!(TimeDesign::new(vec![grid(7)], 0, 10).is_ok());
        let step = term("shift", TermKind::Step { at: 1 });
        assert!(
            message(TimeDesign::new(vec![step.clone(), step.clone()], 0, 5).unwrap_err())
                .contains("'shift' is used twice")
        );
        let clash = term("annual_sin1", TermKind::Regressor);
        let annual = term(
            "annual",
            TermKind::Fourier {
                period: 12,
                harmonics: 1,
            },
        );
        assert!(
            message(TimeDesign::new(vec![annual, clash], 0, 5).unwrap_err())
                .contains("'annual_sin1' is used twice")
        );
        assert!(TimeDesign::new(vec![], 0, 5).is_err());
        assert!(TimeDesign::new(vec![step.clone()], 0, 0).is_err());
        assert!(TimeDesign::new(vec![step.clone()], i64::MAX, 2).is_err());
        // Training positions MAX - 3 and MAX - 2; the future may reach MAX itself.
        let edge = TimeDesign::new(vec![step], i64::MAX - 3, 2).unwrap();
        assert_eq!(edge.future_rows(2, &none()).unwrap(), [[1.0], [1.0]]);
        assert!(message(edge.future_rows(3, &none()).unwrap_err()).contains("64-bit calendar"));
        // Oversized terms are refused before their columns are built.
        let huge_grid = term(
            "trend",
            TermKind::Changepoints(ChangepointPlacement::Grid {
                count: 1 << 40,
                first: Some(0),
                last: Some(i64::MAX),
                fraction: 1.0,
            }),
        );
        let huge_fourier = term(
            "annual",
            TermKind::Fourier {
                period: 1 << 41,
                harmonics: 1 << 40,
            },
        );
        for huge in [huge_grid.clone(), huge_fourier] {
            assert!(
                message(TimeDesign::new(vec![huge], 0, 1).unwrap_err()).contains("safety limit")
            );
        }
        let mut overflow = huge_grid;
        overflow.kind = TermKind::Changepoints(ChangepointPlacement::Grid {
            count: usize::MAX,
            first: Some(0),
            last: Some(i64::MAX),
            fraction: 1.0,
        });
        assert!(
            message(TimeDesign::new(vec![overflow.clone(), overflow], 0, 1).unwrap_err())
                .contains("safety limit")
        );
        assert!(message(edge.future_rows(1 << 40, &none()).unwrap_err()).contains("safety limit"));
    }
}
