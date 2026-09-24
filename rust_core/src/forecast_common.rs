//! Pieces shared by the conjugate Gaussian forecasting samplers.
//!
//! The local-level, local-linear-trend, seasonal, regression, structural,
//! hurdle and hierarchical fits are all seeded Gibbs samplers that run one
//! independent chain per Rayon task, and they and the AR model all forecast by
//! pairing every retained draw with one simulated path. This module holds what
//! they have in common: the size guard in front of every large allocation, the
//! inverse-gamma conditional, the chain driver with its warmup/thinning
//! schedule and overdispersed starting points, the per-chain forecast loop,
//! and the empirical path summaries.

use crate::bayesian_forecast::{BayesianForecastError, ForecastQuantile};
use crate::seeding::chain_seed;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Gamma};
use rayon::prelude::*;
use std::error::Error;
use std::fmt;

/// Largest number of `f64` values one fit, forecast or working buffer may
/// materialize. At eight bytes each this is 200 MB before any Python copy.
pub const MAX_MATERIALIZED_VALUES: usize = 25_000_000;

/// A request whose size overflows or exceeds its safety limit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AllocationLimitError {
    /// What the values were for, e.g. `"local-level forecast"`.
    pub what: &'static str,
    /// The number of values requested, or `None` when the count overflows.
    pub requested: Option<usize>,
    pub limit: usize,
}

impl fmt::Display for AllocationLimitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let limit = if self.limit % 1_000_000 == 0 {
            format!("{} million", self.limit / 1_000_000)
        } else {
            self.limit.to_string()
        };
        match self.requested {
            Some(count) => write!(
                f,
                "{} would materialize {count} values, above the safety limit of {limit}; \
                 reduce chains, draws, series length or forecast horizon",
                self.what
            ),
            None => write!(
                f,
                "{} size overflows and exceeds the safety limit of {limit} values; \
                 reduce chains, draws, series length or forecast horizon",
                self.what
            ),
        }
    }
}

impl Error for AllocationLimitError {}

impl From<AllocationLimitError> for BayesianForecastError {
    fn from(error: AllocationLimitError) -> Self {
        Self::InvalidConfiguration(error.to_string())
    }
}

/// Multiply `factors` and refuse a product above `limit`.
///
/// Call this before reserving storage whose size a caller controls: a Rust
/// allocation that fails aborts the process rather than returning an error,
/// so an oversized `draws` or forecast horizon would otherwise take the
/// Python interpreter down with it.
pub fn checked_value_count(
    what: &'static str,
    factors: &[usize],
    limit: usize,
) -> Result<usize, AllocationLimitError> {
    let requested = factors
        .iter()
        .try_fold(1usize, |product, &factor| product.checked_mul(factor));
    match requested {
        Some(count) if count <= limit => Ok(count),
        _ => Err(AllocationLimitError {
            what,
            requested,
            limit,
        }),
    }
}

/// Guard a forecast that stores `arrays` values per step for every retained
/// draw of possibly ragged `chains`.
pub(crate) fn check_forecast_size<D>(
    what: &'static str,
    chains: &[Vec<D>],
    horizon: usize,
    arrays: usize,
) -> Result<usize, AllocationLimitError> {
    let draws = chains
        .iter()
        .try_fold(0usize, |total, chain| total.checked_add(chain.len()));
    match draws {
        Some(draws) => {
            checked_value_count(what, &[draws, horizon, arrays], MAX_MATERIALIZED_VALUES)
        }
        None => Err(AllocationLimitError {
            what,
            requested: None,
            limit: MAX_MATERIALIZED_VALUES,
        }),
    }
}

/// Require enough finite observations to inform every inferred variance.
///
/// With proper inverse-gamma priors the posterior exists whatever the data,
/// so this is not a condition for the maths to be defined. It is the point
/// below which the likelihood cannot separate the variances it is asked to
/// estimate, so the "fit" would largely hand the priors back; rejecting it
/// catches an all-missing or truncated series instead of returning a
/// posterior that looks informed and is not. Every model needs at least one
/// finite observation. Returns the finite count.
pub(crate) fn require_finite_observations(
    observations: &[f64],
    inferred_variances: usize,
    model: &str,
) -> Result<usize, BayesianForecastError> {
    let finite = observations
        .iter()
        .filter(|value| value.is_finite())
        .count();
    let required = inferred_variances.max(1);
    if finite < required {
        let reason = if inferred_variances > 1 {
            format!(", one per inferred variance ({inferred_variances})")
        } else {
            String::new()
        };
        return Err(BayesianForecastError::InvalidObservations(format!(
            "a {model} fit needs at least {required} finite observation{}{reason}; got {finite}",
            if required == 1 { "" } else { "s" }
        )));
    }
    Ok(finite)
}

/// The inverse-gamma full conditional `InverseGamma(shape, scale)`, as the
/// gamma distribution of its reciprocal.
pub(crate) fn inverse_gamma_conditional(
    shape: f64,
    scale: f64,
) -> Result<Gamma<f64>, BayesianForecastError> {
    if !shape.is_finite() || shape <= 0.0 || !scale.is_finite() || scale <= 0.0 {
        return Err(BayesianForecastError::NumericalFailure(format!(
            "invalid inverse-gamma conditional (shape={shape}, scale={scale})"
        )));
    }
    // A subnormal scale has no representable reciprocal.
    let rate = 1.0 / scale;
    if !rate.is_finite() {
        return Err(BayesianForecastError::NumericalFailure(
            "inverse-gamma reciprocal scale is unrepresentable".into(),
        ));
    }
    Gamma::new(shape, rate).map_err(|error| {
        BayesianForecastError::NumericalFailure(format!(
            "could not construct gamma distribution: {error}"
        ))
    })
}

/// One draw of `InverseGamma(shape, scale)`, rejecting an overflow or
/// underflow rather than passing a zero or infinite variance on.
pub(crate) fn sample_inverse_gamma<R: Rng + ?Sized>(
    shape: f64,
    scale: f64,
    rng: &mut R,
) -> Result<f64, BayesianForecastError> {
    let variance = 1.0 / inverse_gamma_conditional(shape, scale)?.sample(rng);
    if !variance.is_finite() || variance <= 0.0 {
        return Err(BayesianForecastError::NumericalFailure(
            "sampled variance must be finite and strictly positive".into(),
        ));
    }
    Ok(variance)
}

/// Half-width, on the log scale for a positive parameter and in units of a
/// scale for a location, of the window chain starting points are drawn from.
/// `exp(2)` is about 7.4, so a positive start lies anywhere from about a
/// seventh to seven times its reference value.
const START_HALF_WIDTH: f64 = 2.0;

/// Separates a chain's starting-point stream from its sampling stream.
const START_SEED_DOMAIN: u64 = 0x5354_4152_545F_5054;

/// An overdispersed starting value for a positive parameter: `reference`
/// scaled by `exp(U(-2, 2))`.
///
/// Every chain starting from the same point - the prior mode - leaves the
/// between-chain half of split R-hat blind to a sampler that has not left the
/// neighbourhood of its start. A bounded log-uniform factor is used rather than
/// a draw from the prior itself, because an inverse-gamma prior with shape at
/// most 2 has no variance and can put a chain's start many orders of magnitude
/// from anything the data support.
pub(crate) fn overdispersed_positive<R: Rng + ?Sized>(reference: f64, rng: &mut R) -> f64 {
    reference * rng.gen_range(-START_HALF_WIDTH..START_HALF_WIDTH).exp()
}

/// An overdispersed starting value for a location: `center + scale U(-2, 2)`.
pub(crate) fn overdispersed_location<R: Rng + ?Sized>(center: f64, scale: f64, rng: &mut R) -> f64 {
    center + scale * rng.gen_range(-START_HALF_WIDTH..START_HALF_WIDTH)
}

/// Warmup, retention and thinning for one Gibbs fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GibbsSchedule {
    pub(crate) chains: usize,
    pub(crate) warmup: usize,
    pub(crate) draws: usize,
    pub(crate) thinning: usize,
    iterations: usize,
}

impl GibbsSchedule {
    pub(crate) fn new(
        chains: usize,
        warmup: usize,
        draws: usize,
        thinning: usize,
    ) -> Result<Self, BayesianForecastError> {
        let invalid = |message: &str| BayesianForecastError::InvalidConfiguration(message.into());
        if chains == 0 {
            return Err(invalid("number of chains must be positive"));
        }
        if draws == 0 {
            return Err(invalid("number of posterior draws must be positive"));
        }
        if thinning == 0 {
            return Err(invalid("thinning must be positive"));
        }
        let iterations = draws
            .checked_mul(thinning)
            .and_then(|retained| warmup.checked_add(retained))
            .ok_or_else(|| invalid("warmup, draws, and thinning imply too many iterations"))?;
        Ok(Self {
            chains,
            warmup,
            draws,
            thinning,
            iterations,
        })
    }

    /// Whether the state after `iteration` (zero-based) is retained: every
    /// `thinning`-th iteration after warmup, ending on the last one.
    fn retains(&self, iteration: usize) -> bool {
        iteration >= self.warmup && (iteration + 1 - self.warmup).is_multiple_of(self.thinning)
    }
}

/// Run `chains` independent chains on the active Rayon pool, each with its
/// own ChaCha stream keyed by `(seed, chain, domain)`, and collect them in
/// chain order so the result does not depend on the pool size.
pub(crate) fn run_chains<T, E, F>(
    chains: usize,
    seed: u64,
    domain: u64,
    chain: F,
) -> Result<Vec<Vec<T>>, E>
where
    T: Send,
    E: Send,
    F: Fn(u64, &mut ChaCha8Rng) -> Result<Vec<T>, E> + Sync,
{
    (0..chains)
        .into_par_iter()
        .map(|index| {
            let key = chain_seed(seed, index, domain);
            let mut rng = ChaCha8Rng::seed_from_u64(key);
            chain(key, &mut rng)
        })
        .collect()
}

/// Run a seeded Gibbs sampler under `schedule`.
///
/// `start` builds a chain's initial state from a stream of its own, keyed by
/// the chain's sampling key, so every chain - chain 0 included - starts from
/// its own overdispersed point, deterministically given the seed and
/// independently of the Rayon pool.
/// `step` performs one full sweep and, when its `retain` argument is true,
/// returns the draw to keep.
pub(crate) fn run_gibbs_chains<S, T, E, Start, Step>(
    schedule: &GibbsSchedule,
    seed: u64,
    fit_domain: u64,
    start: Start,
    step: Step,
) -> Result<Vec<Vec<T>>, E>
where
    T: Send,
    E: Send,
    Start: Fn(&mut ChaCha8Rng) -> Result<S, E> + Sync,
    Step: Fn(&mut S, &mut ChaCha8Rng, bool) -> Result<Option<T>, E> + Sync,
{
    run_chains(schedule.chains, seed, fit_domain, |key, rng| {
        let mut start_rng = ChaCha8Rng::seed_from_u64(chain_seed(key, 0, START_SEED_DOMAIN));
        let mut state = start(&mut start_rng)?;
        let mut draws = Vec::with_capacity(schedule.draws);
        for iteration in 0..schedule.iterations {
            if let Some(draw) = step(&mut state, rng, schedule.retains(iteration))? {
                draws.push(draw);
            }
        }
        debug_assert_eq!(draws.len(), schedule.draws);
        Ok(draws)
    })
}

/// Pair every posterior draw with one simulated result, one ChaCha stream per
/// chain keyed by `(seed, chain, domain)`, preserving `[chain][draw]` order.
pub(crate) fn simulate_draws<D, P, E, F>(
    chains: &[Vec<D>],
    seed: u64,
    domain: u64,
    simulate: F,
) -> Result<Vec<Vec<P>>, E>
where
    D: Sync,
    P: Send,
    E: Send,
    F: Fn(usize, usize, &D, &mut ChaCha8Rng) -> Result<P, E> + Sync,
{
    chains
        .par_iter()
        .enumerate()
        .map(|(chain_index, chain)| {
            let mut rng = ChaCha8Rng::seed_from_u64(chain_seed(seed, chain_index, domain));
            chain
                .iter()
                .enumerate()
                .map(|(draw_index, draw)| simulate(chain_index, draw_index, draw, &mut rng))
                .collect()
        })
        .collect()
}

/// Paths indexed `[chain][draw][step]`.
pub(crate) type Paths = Vec<Vec<Vec<f64>>>;

/// Turn per-draw groups of `K` paths into `K` separate `[chain][draw][step]`
/// arrays.
pub(crate) fn split_paths<const K: usize>(per_draw: Vec<Vec<[Vec<f64>; K]>>) -> [Paths; K] {
    let mut split: [Paths; K] = std::array::from_fn(|_| Vec::with_capacity(per_draw.len()));
    for chain in per_draw {
        let mut chain_split: [Vec<Vec<f64>>; K] =
            std::array::from_fn(|_| Vec::with_capacity(chain.len()));
        for draw in chain {
            for (target, path) in chain_split.iter_mut().zip(draw) {
                target.push(path);
            }
        }
        for (target, chain_paths) in split.iter_mut().zip(chain_split) {
            target.push(chain_paths);
        }
    }
    split
}

/// Posterior-predictive mean at each horizon, from the scale-aware
/// implementation the sampler's `mean()` accessors already use.
///
/// Accumulating the paths and dividing by their count at the end overflows on
/// input that is entirely finite: two paths holding `1e308` sum to infinity,
/// and the infinity survives the division. See
/// [`crate::diagnostics::scaled_moments`], which centres the draws at one
/// horizon on the first of them and divides by the largest deviation from it
/// before summing, so no partial sum can leave the representable range.
///
/// `validate_paths` has already rejected an empty, ragged or non-finite
/// forecast, so the `NaN` that `scaled_moments` reports for those cases cannot
/// reach a caller from here.
pub(crate) fn path_means(paths: &[Vec<Vec<f64>>]) -> Result<Vec<f64>, BayesianForecastError> {
    let horizon = validate_paths(paths)?;
    Ok((0..horizon)
        .map(|step| {
            crate::diagnostics::scaled_moments(|| paths.iter().flatten().map(|path| path[step])).0
        })
        .collect())
}

/// Empirical quantiles at each horizon, linearly interpolated between order
/// statistics.
pub(crate) fn path_quantiles(
    paths: &[Vec<Vec<f64>>],
    probabilities: &[f64],
) -> Result<Vec<ForecastQuantile>, BayesianForecastError> {
    let horizon = validate_paths(paths)?;
    if probabilities
        .iter()
        .any(|probability| !probability.is_finite() || !(0.0..=1.0).contains(probability))
    {
        return Err(BayesianForecastError::InvalidConfiguration(
            "quantile probabilities must be finite and between zero and one".into(),
        ));
    }
    let ordered_by_step: Vec<Vec<f64>> = (0..horizon)
        .map(|step| {
            let mut values: Vec<f64> = paths.iter().flatten().map(|path| path[step]).collect();
            values.sort_by(f64::total_cmp);
            values
        })
        .collect();
    Ok(probabilities
        .iter()
        .map(|&probability| ForecastQuantile {
            probability,
            values: ordered_by_step
                .iter()
                .map(|ordered| interpolated_quantile(ordered, probability))
                .collect(),
        })
        .collect())
}

/// The common horizon of a non-empty, rectangular, finite forecast.
pub(crate) fn validate_paths(paths: &[Vec<Vec<f64>>]) -> Result<usize, BayesianForecastError> {
    let horizon = paths
        .first()
        .and_then(|chain| chain.first())
        .map_or(0, Vec::len);
    if paths.is_empty() || paths.iter().any(Vec::is_empty) || horizon == 0 {
        return Err(BayesianForecastError::InvalidConfiguration(
            "forecast must contain at least one non-empty path per chain".into(),
        ));
    }
    if paths.iter().flatten().any(|path| path.len() != horizon) {
        return Err(BayesianForecastError::InvalidConfiguration(
            "forecast paths must all have the same horizon".into(),
        ));
    }
    if paths
        .iter()
        .flatten()
        .flatten()
        .any(|value| !value.is_finite())
    {
        return Err(BayesianForecastError::NumericalFailure(
            "forecast contains a non-finite value".into(),
        ));
    }
    Ok(horizon)
}

fn interpolated_quantile(ordered: &[f64], probability: f64) -> f64 {
    let index = probability * (ordered.len() - 1) as f64;
    let lower = index.floor() as usize;
    let upper = index.ceil() as usize;
    if lower == upper {
        ordered[lower]
    } else {
        let weight = index - lower as f64;
        ordered[lower] * (1.0 - weight) + ordered[upper] * weight
    }
}

/// Lower Cholesky factor of a row-major `d x d` matrix, written into `factor`
/// (which must hold `d * d` values and start zeroed above the diagonal).
///
/// Fails on a non-positive or non-finite pivot. Stack-allocated callers such
/// as the two-state trend filter use this directly to stay allocation-free.
pub(crate) fn cholesky_into(matrix: &[f64], d: usize, factor: &mut [f64]) -> Result<(), ()> {
    for i in 0..d {
        for j in 0..=i {
            let mut value = matrix[i * d + j];
            for k in 0..j {
                value -= factor[i * d + k] * factor[j * d + k];
            }
            if i == j {
                if !value.is_finite() || value <= 0.0 {
                    return Err(());
                }
                factor[i * d + j] = value.sqrt();
            } else {
                factor[i * d + j] = value / factor[j * d + j];
            }
        }
    }
    Ok(())
}

/// Lower Cholesky factor of a row-major `d x d` matrix.
pub(crate) fn cholesky(matrix: &[f64], d: usize) -> Result<Vec<f64>, ()> {
    let mut factor = vec![0.0; d * d];
    cholesky_into(matrix, d, &mut factor)?;
    Ok(factor)
}

/// Solve `L x = b` for a row-major lower-triangular `L`.
pub(crate) fn solve_lower(lower: &[f64], d: usize, right_hand_side: &[f64]) -> Vec<f64> {
    let mut solution = vec![0.0; d];
    for row in 0..d {
        let mut value = right_hand_side[row];
        for column in 0..row {
            value -= lower[row * d + column] * solution[column];
        }
        solution[row] = value / lower[row * d + row];
    }
    solution
}

/// Solve `L' x = b` for a row-major lower-triangular `L`.
pub(crate) fn solve_lower_transpose(lower: &[f64], d: usize, right_hand_side: &[f64]) -> Vec<f64> {
    let mut solution = vec![0.0; d];
    for row in (0..d).rev() {
        let mut value = right_hand_side[row];
        for column in row + 1..d {
            value -= lower[column * d + row] * solution[column];
        }
        solution[row] = value / lower[row * d + row];
    }
    solution
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_count_rejects_overflow_and_the_limit_with_a_typed_error() {
        assert_eq!(checked_value_count("x", &[4, 5], 20), Ok(20));
        let error = checked_value_count("local-level forecast", &[4, 6], 20).unwrap_err();
        assert_eq!(error.requested, Some(24));
        assert!(error.to_string().contains("local-level forecast"));
        let overflow = checked_value_count("x", &[usize::MAX, 2], 20).unwrap_err();
        assert_eq!(overflow.requested, None);
        let converted: BayesianForecastError = overflow.into();
        assert!(matches!(
            converted,
            BayesianForecastError::InvalidConfiguration(_)
        ));
        let ragged = vec![vec![0u8; 3], vec![0u8; 2]];
        assert_eq!(check_forecast_size("f", &ragged, 4, 2), Ok(40));
    }

    #[test]
    fn schedule_retains_the_last_iteration_of_every_thinning_block() {
        let schedule = GibbsSchedule::new(1, 3, 4, 2).unwrap();
        let retained: Vec<usize> = (0..schedule.iterations)
            .filter(|&iteration| schedule.retains(iteration))
            .collect();
        assert_eq!(retained, vec![4, 6, 8, 10]);
        assert_eq!(schedule.iterations, 11);
        assert!(GibbsSchedule::new(0, 1, 1, 1).is_err());
        assert!(GibbsSchedule::new(1, 1, 0, 1).is_err());
        assert!(GibbsSchedule::new(1, 1, 1, 0).is_err());
        assert!(GibbsSchedule::new(1, 1, usize::MAX, 2).is_err());
    }

    #[test]
    fn overdispersed_starts_stay_inside_their_window_and_use_it() {
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let starts: Vec<f64> = (0..2000)
            .map(|_| overdispersed_positive(0.5, &mut rng))
            .collect();
        let (lowest, highest) = starts
            .iter()
            .fold((f64::MAX, 0.0f64), |(lo, hi), &x| (lo.min(x), hi.max(x)));
        assert!(lowest >= 0.5 * (-2.0f64).exp() && highest <= 0.5 * 2.0f64.exp());
        assert!(lowest < 0.1 && highest > 2.5, "{lowest} {highest}");
        let locations: Vec<f64> = (0..2000)
            .map(|_| overdispersed_location(10.0, 0.5, &mut rng))
            .collect();
        assert!(locations.iter().all(|x| (9.0..=11.0).contains(x)));
        assert!(locations.iter().any(|x| *x < 9.2) && locations.iter().any(|x| *x > 10.8));
    }

    #[test]
    fn every_chain_starts_from_its_own_seeded_point() {
        let schedule = GibbsSchedule::new(4, 0, 1, 1).unwrap();
        let starts = |seed: u64, threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    run_gibbs_chains(
                        &schedule,
                        seed,
                        0x1234,
                        |rng| Ok::<_, ()>(overdispersed_positive(1.0, rng)),
                        |start: &mut f64, _, retain| Ok(retain.then_some(*start)),
                    )
                    .unwrap()
                })
        };
        let first: Vec<f64> = starts(7, 1).into_iter().flatten().collect();
        assert_eq!(first.len(), 4);
        for (i, a) in first.iter().enumerate() {
            for b in &first[i + 1..] {
                assert_ne!(a, b, "two chains share a starting point");
            }
        }
        let again: Vec<f64> = starts(7, 3).into_iter().flatten().collect();
        assert_eq!(first, again, "starts must depend only on the seed");
        let other: Vec<f64> = starts(8, 1).into_iter().flatten().collect();
        assert_ne!(first[0], other[0], "chain 0's start must follow the seed");
    }

    #[test]
    fn finite_observation_requirement_names_the_model_and_the_shortfall() {
        assert_eq!(
            require_finite_observations(&[1.0, f64::NAN, 2.0], 2, "local-level"),
            Ok(2)
        );
        let error = require_finite_observations(&[1.0, f64::NAN], 2, "local-level").unwrap_err();
        assert!(error.to_string().contains("at least 2 finite observations"));
        assert!(require_finite_observations(&[f64::NAN], 0, "structural").is_err());
        assert_eq!(require_finite_observations(&[0.0], 0, "structural"), Ok(1));
    }

    #[test]
    fn inverse_gamma_rejects_parameters_without_a_representable_rate() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        assert!(sample_inverse_gamma(2.0, f64::from_bits(1), &mut rng).is_err());
        assert!(sample_inverse_gamma(0.0, 1.0, &mut rng).is_err());
        assert!(sample_inverse_gamma(2.0, 1.0, &mut rng).unwrap() > 0.0);
    }
}
