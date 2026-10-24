//! Posterior moments, computed independently, for testing the conjugate Gibbs
//! samplers.
//!
//! Given its variances, each Gaussian forecasting model is linear: every latent
//! mean and every observation is a fixed combination of independent normal
//! terms, such as an initial state, the innovations and the observation errors.
//! A test writes those combinations down from the model's definition, so the
//! observations' distribution given the variances needs no Kalman filter, FFBS
//! or Gibbs step from the code under test. Weighting that Gaussian likelihood by
//! the inverse-gamma priors on a grid of log variances gives the variances'
//! posterior, and conditioning the Gaussian at each grid point gives every
//! latent's.
//!
//! The moments checked are each quantity's mean and variance, every pairwise
//! covariance, and each variance times each latent's squared deviation, which is
//! where a hierarchy's dependence between a spread and the means it governs
//! shows. Matching all of them is necessary for the right joint posterior, not
//! sufficient: higher moments are not compared.

use crate::forecast_common::InverseGammaPrior;

/// One independent normal term. Its variance is either fixed or one of the
/// variances under inference, by index.
#[derive(Clone, Copy)]
pub(crate) enum TermVariance {
    Fixed(f64),
    Inferred(usize),
}

/// A quantity written as `constant + sum_k coefficients[k] * term[k]`, where
/// the terms are independent, zero-mean normals.
#[derive(Clone)]
pub(crate) struct Linear {
    pub constant: f64,
    pub coefficients: Vec<f64>,
}

impl Linear {
    pub fn term(index: usize, terms: usize) -> Self {
        let mut coefficients = vec![0.0; terms];
        coefficients[index] = 1.0;
        Self {
            constant: 0.0,
            coefficients,
        }
    }

    pub fn plus(&self, other: &Self) -> Self {
        self.scaled_plus(1.0, other)
    }

    /// `self + factor * other`.
    pub fn scaled_plus(&self, factor: f64, other: &Self) -> Self {
        Self {
            constant: self.constant + factor * other.constant,
            coefficients: self
                .coefficients
                .iter()
                .zip(&other.coefficients)
                .map(|(a, b)| a + factor * b)
                .collect(),
        }
    }
}

impl InverseGammaPrior {
    /// The prior's own mean and standard deviation, for negative controls.
    pub(crate) fn moments(self) -> Moments {
        assert!(self.shape > 2.0, "the prior needs a finite variance");
        let mean = self.scale / (self.shape - 1.0);
        Moments {
            mean,
            sd: mean / (self.shape - 2.0).sqrt(),
        }
    }
}

/// A model with its variances fixed: independent terms, the observed
/// quantities with their values, and the latent quantities to report.
pub(crate) struct LinearGaussianModel {
    pub terms: Vec<TermVariance>,
    pub observed: Vec<(Linear, f64)>,
    pub latents: Vec<Linear>,
}

/// Posterior mean and standard deviation.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Moments {
    pub mean: f64,
    pub sd: f64,
}

/// Moments of the quantities `q`: the inferred variances, in prior order, then
/// the latents, in the model's order.
pub(crate) struct ExactPosterior {
    variance_count: usize,
    /// `E[q_a]`.
    first: Vec<f64>,
    /// `E[q_a q_b]`, row-major over all quantities.
    product: Vec<f64>,
    /// `E[v_k x_i^2]`, row-major over variances `k` and latents `i`.
    variance_times_square: Vec<f64>,
}

impl ExactPosterior {
    pub fn quantity_count(&self) -> usize {
        self.first.len()
    }

    pub fn moments(&self, a: usize) -> Moments {
        Moments {
            mean: self.first[a],
            sd: self.covariance(a, a).sqrt(),
        }
    }

    pub fn variances(&self) -> Vec<Moments> {
        (0..self.variance_count).map(|k| self.moments(k)).collect()
    }

    /// `E[(q_a - E q_a)(q_b - E q_b)]`.
    fn covariance(&self, a: usize, b: usize) -> f64 {
        self.product[a * self.quantity_count() + b] - self.first[a] * self.first[b]
    }

    /// `E[v_k (x_i - E x_i)^2]` for variance `k` and latent `i`.
    fn variance_times_squared_deviation(&self, k: usize, i: usize) -> f64 {
        let x = self.variance_count + i;
        let mean = self.first[x];
        let latents = self.quantity_count() - self.variance_count;
        self.variance_times_square[k * latents + i]
            - 2.0 * mean * self.product[k * self.quantity_count() + x]
            + mean * mean * self.first[k]
    }

    /// The largest difference between two integrations of the same moments, in
    /// units of the quantities' standard deviations: one for a mean, the
    /// product of two for a covariance, and a variance's mean times a latent's
    /// variance for the mixed moments.
    /// Quantities are numbered as in `ExactPosterior`; the second value names
    /// the moment that moved most.
    fn largest_difference(&self, other: &Self) -> (f64, String) {
        let count = self.quantity_count();
        let sd = |a: usize| self.moments(a).sd;
        let mut worst = (0.0f64, String::new());
        let mut consider = |difference: f64, label: &dyn Fn() -> String| {
            if difference > worst.0 {
                worst = (difference, label());
            }
        };
        for a in 0..count {
            consider((self.first[a] - other.first[a]).abs() / sd(a), &|| {
                format!("E[q{a}]")
            });
            for b in 0..count {
                let scale = sd(a) * sd(b);
                consider(
                    (self.covariance(a, b) - other.covariance(a, b)).abs() / scale,
                    &|| format!("Cov[q{a}, q{b}]"),
                );
            }
        }
        for k in 0..self.variance_count {
            for i in 0..count - self.variance_count {
                let scale = self.first[k] * sd(self.variance_count + i).powi(2);
                consider(
                    (self.variance_times_squared_deviation(k, i)
                        - other.variance_times_squared_deviation(k, i))
                    .abs()
                        / scale,
                    &|| {
                        format!(
                            "E[q{k} (q{} - E q{})^2]",
                            self.variance_count + i,
                            self.variance_count + i
                        )
                    },
                );
            }
        }
        worst
    }
}

/// The largest change, in the units of `largest_difference`, that widening the
/// grid may make.
const WIDENING_TOLERANCE: f64 = 1e-6;
/// The largest change halving the grid's resolution may make. The trapezoid
/// rule's error on these smooth integrands falls exponentially with the
/// spacing, so a grid whose half-resolution answer is this close is closer
/// still.
const RESOLUTION_TOLERANCE: f64 = 1e-4;

/// One variance's grid: `points` equally spaced log variances from `low` to
/// `high`. An odd count keeps every other point a grid of its own.
#[derive(Clone, Copy)]
pub(crate) struct Axis {
    pub low: f64,
    pub high: f64,
    pub points: usize,
}

impl Axis {
    fn step(self) -> f64 {
        (self.high - self.low) / (self.points - 1) as f64
    }

    fn value(self, index: usize) -> f64 {
        self.low + self.step() * index as f64
    }

    /// The same spacing, at least two units further out at each end.
    fn widened(self) -> Self {
        // An even pad keeps the point count odd.
        let pad = 2 * (1.0 / self.step()).ceil() as usize;
        Self {
            low: self.low - pad as f64 * self.step(),
            high: self.high + pad as f64 * self.step(),
            points: self.points + 2 * pad,
        }
    }
}

/// Moments by the trapezoid rule over a grid of log variances, one axis per
/// prior. Two checks guard the grid. Along each axis in turn, every other point
/// is a grid of twice the spacing, and its integral must agree within
/// `RESOLUTION_TOLERANCE`, so an axis too coarse for a narrow posterior fails. A grid two units wider at each
/// end must agree within `WIDENING_TOLERANCE`, so a grid that cuts off a tail
/// fails. Agreement is evidence of convergence, not a bound on the error.
pub(crate) fn exact_posterior(
    model: &LinearGaussianModel,
    priors: &[InverseGammaPrior],
    axes: &[Axis],
) -> ExactPosterior {
    assert_eq!(axes.len(), priors.len(), "one axis per variance");
    assert!(
        axes.iter().all(|axis| axis.points % 2 == 1),
        "an odd number of points holds a grid of twice the spacing"
    );
    let (fine, coarse) = integrate(model, priors, axes);
    for (k, coarse) in coarse.iter().enumerate() {
        let (coarser, moment) = fine.largest_difference(coarse);
        assert!(
            coarser < RESOLUTION_TOLERANCE,
            "halving the grid's resolution along axis {k} moved a moment by {coarser} standard \
             deviations ({moment})"
        );
    }
    let widened = axes.iter().map(|axis| axis.widened()).collect::<Vec<_>>();
    let (wide, _) = integrate(model, priors, &widened);
    let (wider, moment) = fine.largest_difference(&wide);
    if wider >= WIDENING_TOLERANCE {
        // Name the axis whose widening moves the moments most.
        let (axis, change) = (0..axes.len())
            .map(|k| {
                let mut one = axes.to_vec();
                one[k] = axes[k].widened();
                (
                    k,
                    fine.largest_difference(&integrate(model, priors, &one).0).0,
                )
            })
            .fold(
                (0, 0.0),
                |best, next| if next.1 > best.1 { next } else { best },
            );
        panic!(
            "widening the grid moved a moment by {wider} standard deviations ({moment}); \
             widening axis {axis} alone moves one by {change}"
        );
    }
    fine
}

/// Unnormalized trapezoid sums, rescaled whenever a larger log weight appears.
struct Sums {
    largest: f64,
    total: f64,
    first: Vec<f64>,
    product: Vec<f64>,
    variance_times_square: Vec<f64>,
}

impl Sums {
    fn new(count: usize, variance_count: usize) -> Self {
        Self {
            largest: f64::NEG_INFINITY,
            total: 0.0,
            first: vec![0.0; count],
            product: vec![0.0; count * count],
            variance_times_square: vec![0.0; variance_count * (count - variance_count)],
        }
    }

    fn add(&mut self, log_weight: f64, first: &[f64], product: &[f64], times_square: &[f64]) {
        if log_weight > self.largest {
            let rescale = (self.largest - log_weight).exp();
            self.total *= rescale;
            for values in [
                &mut self.first,
                &mut self.product,
                &mut self.variance_times_square,
            ] {
                values.iter_mut().for_each(|value| *value *= rescale);
            }
            self.largest = log_weight;
        }
        let w = (log_weight - self.largest).exp();
        self.total += w;
        for (sum, value) in self.first.iter_mut().zip(first) {
            *sum += w * value;
        }
        for (sum, value) in self.product.iter_mut().zip(product) {
            *sum += w * value;
        }
        for (sum, value) in self.variance_times_square.iter_mut().zip(times_square) {
            *sum += w * value;
        }
    }

    fn finish(self, variance_count: usize) -> ExactPosterior {
        let total = self.total;
        let normalize = |values: Vec<f64>| values.into_iter().map(|v| v / total).collect();
        ExactPosterior {
            variance_count,
            first: normalize(self.first),
            product: normalize(self.product),
            variance_times_square: normalize(self.variance_times_square),
        }
    }
}

/// The trapezoid integral on the grid, and for each axis the integral on the
/// points even along it.
fn integrate(
    model: &LinearGaussianModel,
    priors: &[InverseGammaPrior],
    axes: &[Axis],
) -> (ExactPosterior, Vec<ExactPosterior>) {
    let dimension = priors.len();
    let observed = model.observed.len();
    let latent_count = model.latents.len();
    let count = dimension + latent_count;
    let residuals = model
        .observed
        .iter()
        .map(|(quantity, value)| value - quantity.constant)
        .collect::<Vec<_>>();
    let mut fine = Sums::new(count, dimension);
    let mut coarse = (0..dimension)
        .map(|_| Sums::new(count, dimension))
        .collect::<Vec<_>>();

    let mut index = vec![0usize; dimension];
    loop {
        let log_variances = index
            .iter()
            .zip(axes)
            .map(|(&i, axis)| axis.value(i))
            .collect::<Vec<_>>();
        let variances = log_variances.iter().map(|v| v.exp()).collect::<Vec<_>>();
        let term_variances = model
            .terms
            .iter()
            .map(|term| match *term {
                TermVariance::Fixed(value) => value,
                TermVariance::Inferred(k) => variances[k],
            })
            .collect::<Vec<_>>();
        let covariance = |a: &Linear, b: &Linear| -> f64 {
            a.coefficients
                .iter()
                .zip(&b.coefficients)
                .zip(&term_variances)
                .map(|((x, y), v)| x * y * v)
                .sum()
        };
        let mut matrix = vec![0.0; observed * observed];
        for i in 0..observed {
            for j in 0..=i {
                let value = covariance(&model.observed[i].0, &model.observed[j].0);
                matrix[i * observed + j] = value;
                matrix[j * observed + i] = value;
            }
        }
        let lower = cholesky(&matrix, observed);
        let whitened = forward_solve(&lower, observed, &residuals);
        // Half the log determinant of the observations' covariance.
        let half_log_determinant = (0..observed)
            .map(|i| lower[i * observed + i].ln())
            .sum::<f64>();
        // Inverse-gamma log densities in log coordinates include the Jacobian
        // `v`, which cancels one power: `-shape log v - scale / v`.
        let log_prior = priors
            .iter()
            .zip(&log_variances)
            .zip(&variances)
            .map(|((prior, log_v), v)| -prior.shape * log_v - prior.scale / v)
            .sum::<f64>();
        let log_density =
            log_prior - half_log_determinant - 0.5 * whitened.iter().map(|z| z * z).sum::<f64>();

        // Each latent's conditional mean, and the whitened cross-covariances
        // that give the latents' conditional covariance.
        let whitened_cross = model
            .latents
            .iter()
            .map(|latent| {
                let cross = model
                    .observed
                    .iter()
                    .map(|(quantity, _)| covariance(latent, quantity))
                    .collect::<Vec<_>>();
                forward_solve(&lower, observed, &cross)
            })
            .collect::<Vec<_>>();
        let means = model
            .latents
            .iter()
            .zip(&whitened_cross)
            .map(|(latent, c)| latent.constant + dot(c, &whitened))
            .collect::<Vec<_>>();
        let conditional_covariance = |i: usize, j: usize| {
            covariance(&model.latents[i], &model.latents[j])
                - dot(&whitened_cross[i], &whitened_cross[j])
        };
        // `E[q | variances]`, `E[q_a q_b | variances]` and
        // `E[v_k x_i^2 | variances]`.
        let first = variances.iter().chain(&means).copied().collect::<Vec<_>>();
        let mut product = vec![0.0; count * count];
        for a in 0..count {
            for b in 0..count {
                let mut value = first[a] * first[b];
                if a >= dimension && b >= dimension {
                    value += conditional_covariance(a - dimension, b - dimension);
                }
                product[a * count + b] = value;
            }
        }
        let mut times_square = vec![0.0; dimension * latent_count];
        for k in 0..dimension {
            for i in 0..latent_count {
                let square = conditional_covariance(i, i) + means[i] * means[i];
                times_square[k * latent_count + i] = variances[k] * square;
            }
        }

        // The trapezoid rule halves the weight of each end of each axis; on
        // an axis's even-indexed points the ends are the same two points.
        let ends = index
            .iter()
            .zip(axes)
            .filter(|(&i, axis)| i == 0 || i == axis.points - 1)
            .count() as f64;
        let log_half = 0.5f64.ln();
        fine.add(
            log_density + ends * log_half,
            &first,
            &product,
            &times_square,
        );
        for (k, coarse) in coarse.iter_mut().enumerate() {
            if index[k] % 2 == 0 {
                coarse.add(
                    log_density + ends * log_half,
                    &first,
                    &product,
                    &times_square,
                );
            }
        }

        let mut axis = 0;
        loop {
            if axis == dimension {
                let coarse = coarse
                    .into_iter()
                    .map(|sums| sums.finish(dimension))
                    .collect();
                return (fine.finish(dimension), coarse);
            }
            index[axis] += 1;
            if index[axis] < axes[axis].points {
                break;
            }
            index[axis] = 0;
            axis += 1;
        }
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Written out here rather than shared with the samplers, whose factorization
/// is part of what these references check.
fn cholesky(matrix: &[f64], d: usize) -> Vec<f64> {
    let mut lower = vec![0.0; d * d];
    for i in 0..d {
        for j in 0..=i {
            let mut value = matrix[i * d + j];
            for k in 0..j {
                value -= lower[i * d + k] * lower[j * d + k];
            }
            lower[i * d + j] = if i == j {
                assert!(
                    value > 0.0,
                    "observation covariance is not positive definite"
                );
                value.sqrt()
            } else {
                value / lower[j * d + j]
            };
        }
    }
    lower
}

fn forward_solve(lower: &[f64], d: usize, right: &[f64]) -> Vec<f64> {
    let mut solution = vec![0.0; d];
    for i in 0..d {
        let mut value = right[i];
        for k in 0..i {
            value -= lower[i * d + k] * solution[k];
        }
        solution[i] = value / lower[i * d + i];
    }
    solution
}

/// Added to every test's sampler seed. `scripts/plant_gibbs_bugs.py --seeds`
/// sets it to rerun the tests on other seeds without rebuilding them.
pub(crate) fn seed_offset() -> u64 {
    std::env::var("RUSTMC_EXACT_SEED_OFFSET")
        .ok()
        .map(|value| {
            value
                .parse()
                .expect("RUSTMC_EXACT_SEED_OFFSET is an integer")
        })
        .unwrap_or(0)
}

/// The smallest tolerance in Monte Carlo standard errors, as in the NUTS
/// exactness gate.
const MIN_MCSE_TOLERANCE: f64 = 4.0;
/// The chance a test of a correct sampler should fail by chance, over all its
/// comparisons together.
const FAMILY_FALSE_ALARM: f64 = 0.001;

/// The tolerance for each of `comparisons` estimates: the two-sided normal
/// quantile that holds the chance any one of them misses to
/// `FAMILY_FALSE_ALARM`, and never less than `MIN_MCSE_TOLERANCE`. At 4 MCSE
/// each, the hierarchy's 152 comparisons failed 4 of 100 seeds of a correct
/// sampler, each by a single comparison at 4.1 to 4.3.
fn mcse_tolerance(comparisons: usize) -> f64 {
    let quantile =
        crate::diagnostics::inv_normal_cdf(1.0 - FAMILY_FALSE_ALARM / (2.0 * comparisons as f64));
    quantile.max(MIN_MCSE_TOLERANCE)
}
/// The smallest bulk ESS a checked estimate may rest on, as in the NUTS gate,
/// so stuck chains cannot pass by inflating an MCSE.
const MIN_ESS_BULK: f64 = 400.0;

/// A sampler's draws of every quantity of an `ExactPosterior`, in its order,
/// as `[quantity][chain][draw]` with each quantity's name.
pub(crate) type Draws = Vec<(String, Vec<Vec<f64>>)>;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Check {
    /// A quantity's mean or variance.
    Marginal,
    /// A covariance between two quantities, or a variance times a latent's
    /// squared deviation.
    Dependence,
}

pub(crate) enum Failure {
    /// Too little bulk ESS to rest an estimate on, or none at all.
    LowEss(String),
    /// The estimate missed its exact value.
    Missed(Check, String),
}

impl std::fmt::Debug for Failure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LowEss(message) | Self::Missed(_, message) => formatter.write_str(message),
        }
    }
}

/// Every check of `draws` against `exact`: each quantity's mean and mean
/// squared deviation about the exact mean, the mean of every pairwise product
/// of deviations, and the mean of each variance times each latent's squared
/// deviation. Each must lie within `mcse_tolerance` of the number of
/// comparisons, in Monte Carlo standard errors, of its exact value and rest on a bulk ESS of at least `MIN_ESS_BULK`. A mean's
/// MCSE uses the exact standard deviation, so an outlier cannot widen its own
/// tolerance; the other checks have no exact fourth moments to use and estimate
/// theirs from the draws.
pub(crate) fn check_posterior(draws: &Draws, exact: &ExactPosterior) -> Vec<Failure> {
    let count = exact.quantity_count();
    assert_eq!(draws.len(), count, "one set of draws per quantity");
    let deviation = |a: usize| -> Vec<Vec<f64>> {
        let mean = exact.first[a];
        draws[a]
            .1
            .iter()
            .map(|chain| chain.iter().map(|x| x - mean).collect())
            .collect()
    };
    let deviations = (0..count).map(deviation).collect::<Vec<_>>();
    let comparisons =
        count + count * (count + 1) / 2 + exact.variance_count * (count - exact.variance_count);
    let tolerance = mcse_tolerance(comparisons);
    let mut failures = Vec::new();
    let mut worst = (0.0f64, String::new());
    let mut record = |error: f64, name: &str| {
        if error > worst.0 {
            worst = (error, name.to_string());
        }
    };
    for a in 0..count {
        let name = &draws[a].0;
        let sd = Some(exact.moments(a).sd);
        let label = format!("{name} mean");
        let mean = Estimate::new(label.clone(), Check::Marginal, &draws[a].1);
        record(
            mean.compare(exact.first[a], sd, tolerance, &mut failures),
            &label,
        );
        for b in a..count {
            let products = elementwise(&deviations[a], &deviations[b], |x, y| x * y);
            let (label, check) = if a == b {
                (format!("{name} variance"), Check::Marginal)
            } else {
                (
                    format!("covariance of {name} and {}", draws[b].0),
                    Check::Dependence,
                )
            };
            let error = Estimate::new(label.clone(), check, &products).compare(
                exact.covariance(a, b),
                None,
                tolerance,
                &mut failures,
            );
            record(error, &label);
        }
    }
    for k in 0..exact.variance_count {
        for i in 0..count - exact.variance_count {
            let x = exact.variance_count + i;
            let products = elementwise(&draws[k].1, &deviations[x], |v, d| v * d * d);
            let label = format!("{} times {}'s squared deviation", draws[k].0, draws[x].0);
            let error = Estimate::new(label.clone(), Check::Dependence, &products).compare(
                exact.variance_times_squared_deviation(k, i),
                None,
                tolerance,
                &mut failures,
            );
            record(error, &label);
        }
    }
    // For `scripts/plant_gibbs_bugs.py --seeds`, which tracks the margin.
    if std::env::var_os("RUSTMC_EXACT_REPORT").is_some() {
        eprintln!(
            "worst comparison: {:.3} MCSE ({}), tolerance {tolerance:.3} over {comparisons}",
            worst.0, worst.1
        );
    }
    failures
}

fn elementwise(a: &[Vec<f64>], b: &[Vec<f64>], combine: impl Fn(f64, f64) -> f64) -> Vec<Vec<f64>> {
    a.iter()
        .zip(b)
        .map(|(x, y)| x.iter().zip(y).map(|(p, q)| combine(*p, *q)).collect())
        .collect()
}

/// The mean of one statistic's draws, with what its MCSE needs.
struct Estimate<'a> {
    name: String,
    check: Check,
    chains: &'a [Vec<f64>],
}

impl<'a> Estimate<'a> {
    fn new(name: String, check: Check, chains: &'a [Vec<f64>]) -> Self {
        Self {
            name,
            check,
            chains,
        }
    }

    /// Compares the draws' mean with `exact`, using `exact_sd` for the MCSE
    /// when it is known and the draws' own standard deviation otherwise, and
    /// returns the error in MCSE (infinite when there is too little ESS).
    fn compare(
        self,
        exact: f64,
        exact_sd: Option<f64>,
        tolerance: f64,
        failures: &mut Vec<Failure>,
    ) -> f64 {
        let Self {
            name,
            check,
            chains,
        } = self;
        let all = chains.iter().flatten().copied().collect::<Vec<_>>();
        let count = all.len() as f64;
        let mean = all.iter().sum::<f64>() / count;
        let bulk = crate::diagnostics::ess_bulk_chains(chains);
        // A NaN (draws that never move) fails rather than passing every
        // comparison against it.
        if bulk.is_nan() || bulk < MIN_ESS_BULK {
            failures.push(Failure::LowEss(format!(
                "{name}: bulk ESS {bulk} is below {MIN_ESS_BULK}"
            )));
            return f64::INFINITY;
        }
        let sd = exact_sd.unwrap_or_else(|| {
            (all.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (count - 1.0)).sqrt()
        });
        let mcse = sd / crate::diagnostics::ess_raw(chains).sqrt();
        let error = (mean - exact).abs() / mcse;
        if error.is_nan() || error >= tolerance {
            failures.push(Failure::Missed(
                check,
                format!("{name}: {mean} is {error:.1} MCSE from the exact {exact}"),
            ));
        }
        error
    }
}

/// Asserts that `draws` pass every check against `exact`, and that the checks
/// can fail for the reasons they exist: the variances' draws must miss the
/// prior's moments, and the same draws with their quantities misaligned must
/// miss a dependence check.
pub(crate) fn assert_matches(draws: &Draws, exact: &ExactPosterior, priors: &[InverseGammaPrior]) {
    let failures = check_posterior(draws, exact);
    assert!(failures.is_empty(), "{failures:#?}");
    for (k, prior) in priors.iter().enumerate() {
        let moments = prior.moments();
        let squared = draws[k]
            .1
            .iter()
            .map(|chain| chain.iter().map(|v| (v - moments.mean).powi(2)).collect())
            .collect::<Vec<Vec<f64>>>();
        let mut misses = Vec::new();
        Estimate::new("prior mean".into(), Check::Marginal, &draws[k].1).compare(
            moments.mean,
            Some(moments.sd),
            MIN_MCSE_TOLERANCE,
            &mut misses,
        );
        Estimate::new("prior variance".into(), Check::Marginal, &squared).compare(
            moments.sd.powi(2),
            None,
            MIN_MCSE_TOLERANCE,
            &mut misses,
        );
        assert!(
            misses.iter().any(|f| matches!(f, Failure::Missed(..))),
            "{}: the checks cannot tell its draws from the prior: {misses:?}",
            draws[k].0
        );
    }
    let misaligned = check_posterior(&misaligned(draws), exact);
    assert!(
        misaligned
            .iter()
            .any(|f| matches!(f, Failure::Missed(Check::Dependence, _))),
        "the checks cannot see the dependence between quantities: {misaligned:?}"
    );
}

/// The same draws with each quantity rotated by a different number of
/// iterations within its chain: every marginal keeps its values, and draws of
/// different quantities no longer come from the same iteration.
fn misaligned(draws: &Draws) -> Draws {
    draws
        .iter()
        .enumerate()
        .map(|(a, (name, chains))| {
            let shifted = chains
                .iter()
                .map(|chain| {
                    let mut chain = chain.clone();
                    let len = chain.len();
                    chain.rotate_left((a * 997) % len);
                    chain
                })
                .collect();
            (name.clone(), shifted)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::TermVariance::{Fixed, Inferred};
    use super::*;

    fn axis(low: f64, high: f64, points: usize) -> Axis {
        Axis { low, high, points }
    }

    /// `y[i] ~ Normal(0, v)` with `v ~ InverseGamma(a, b)` has the posterior
    /// `InverseGamma(a + n / 2, b + sum(y^2) / 2)`.
    #[test]
    fn a_conjugate_variance_matches_its_closed_form() {
        let ys = [0.8, -1.9, 0.3, 2.4, -0.6, 1.1];
        let terms = ys.iter().map(|_| Inferred(0)).collect::<Vec<_>>();
        let model = LinearGaussianModel {
            observed: ys
                .iter()
                .enumerate()
                .map(|(i, &y)| (Linear::term(i, terms.len()), y))
                .collect(),
            terms,
            latents: Vec::new(),
        };
        let prior = InverseGammaPrior {
            shape: 3.0,
            scale: 1.5,
        };
        let exact = exact_posterior(&model, &[prior], &[axis(-8.0, 8.0, 121)]).variances()[0];
        let closed = InverseGammaPrior {
            shape: prior.shape + ys.len() as f64 / 2.0,
            scale: prior.scale + ys.iter().map(|y| y * y).sum::<f64>() / 2.0,
        }
        .moments();
        assert!((exact.mean - closed.mean).abs() < 1e-9 * closed.mean);
        assert!((exact.sd - closed.sd).abs() < 1e-9 * closed.sd);
    }

    /// With every term's variance fixed, the latents' posterior is the normal
    /// update: `x ~ Normal(1, 4)` seen twice with unit noise, and `x + d` with
    /// an unobserved `d ~ Normal(0, 2)`. The one inferred variance is unused,
    /// so it keeps its prior.
    #[test]
    fn conditioning_matches_the_normal_update() {
        let terms = vec![Fixed(4.0), Fixed(1.0), Fixed(1.0), Fixed(2.0)];
        let x = Linear {
            constant: 1.0,
            ..Linear::term(0, 4)
        };
        let ys = [2.2, 1.4];
        let model = LinearGaussianModel {
            terms,
            observed: vec![
                (x.plus(&Linear::term(1, 4)), ys[0]),
                (x.plus(&Linear::term(2, 4)), ys[1]),
            ],
            latents: vec![x.clone(), x.plus(&Linear::term(3, 4))],
        };
        let prior = InverseGammaPrior {
            shape: 30.0,
            scale: 29.0,
        };
        let exact = exact_posterior(&model, &[prior], &[axis(-1.5, 1.5, 121)]);
        let variance = 1.0 / (1.0 / 4.0 + 2.0);
        let mean = variance * (1.0 / 4.0 + ys[0] + ys[1]);
        let close = |a: f64, b: f64| (a - b).abs() < 1e-12 * b.abs().max(1.0);
        assert!(close(exact.moments(1).mean, mean));
        assert!(close(exact.moments(1).sd, variance.sqrt()));
        assert!(close(exact.moments(2).mean, mean));
        assert!(close(exact.moments(2).sd, (variance + 2.0).sqrt()));
        assert!(close(exact.covariance(1, 2), variance));
        assert!(close(exact.covariance(0, 1), 0.0));
        let unused = prior.moments();
        assert!((exact.moments(0).mean - unused.mean).abs() < 1e-9);
        assert!((exact.moments(0).sd - unused.sd).abs() < 1e-9);
    }

    /// A variance the data say nothing about keeps its InverseGamma(2.5, 0.3)
    /// prior, whose second moment's integrand decays only as `v^-0.5` in log
    /// coordinates, so a grid ending at `e^7` cuts off part of it.
    #[test]
    #[should_panic(expected = "widening the grid moved a moment")]
    fn a_grid_that_cuts_off_a_tail_is_refused() {
        let model = LinearGaussianModel {
            terms: vec![Inferred(0)],
            observed: vec![(Linear::term(0, 1), 0.4)],
            latents: Vec::new(),
        };
        let prior = InverseGammaPrior {
            shape: 2.5,
            scale: 0.3,
        };
        exact_posterior(&model, &[prior, prior], &[axis(-9.0, 7.0, 151); 2]);
    }

    /// `InverseGamma(100, 100)`, from one zero observation, is about 0.1 wide
    /// in log coordinates, too narrow for points 0.27 apart.
    #[test]
    #[should_panic(expected = "halving the grid's resolution along axis 0")]
    fn a_grid_too_coarse_for_the_posterior_is_refused() {
        let model = LinearGaussianModel {
            terms: vec![Inferred(0)],
            observed: vec![(Linear::term(0, 1), 0.0)],
            latents: Vec::new(),
        };
        let prior = InverseGammaPrior {
            shape: 99.5,
            scale: 100.0,
        };
        exact_posterior(&model, &[prior], &[axis(-10.0, 6.0, 61)]);
    }

    #[test]
    fn draws_that_never_move_stuck_chains_and_an_outlier_fail() {
        let check = |chains: &[Vec<f64>]| {
            let mut failures = Vec::new();
            Estimate::new("x".into(), Check::Marginal, chains).compare(
                0.0,
                Some(1.0),
                MIN_MCSE_TOLERANCE,
                &mut failures,
            );
            failures
        };
        let constant = vec![vec![999.0; 25_000]; 4];
        assert!(matches!(check(&constant)[..], [Failure::LowEss(..)]));
        // Four stuck chains whose pooled mean is one standard deviation off
        // pass a bare MCSE test, since their ESS is about 4.
        let stuck = [-0.5, 0.5, 1.5, 2.5]
            .iter()
            .map(|&value| vec![value; 25_000])
            .collect::<Vec<_>>();
        assert!(matches!(check(&stuck)[..], [Failure::LowEss(..)]));
        // One draw of 1e8 among 100,000 standard normals moves the mean by
        // 1,000, which the draws' own standard deviation would cover.
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(5);
        let mut chains = (0..4)
            .map(|_| {
                (0..25_000)
                    .map(|_| crate::forecast_common::standard_normal(&mut rng))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert!(check(&chains).is_empty());
        chains[2][100] = 1e8;
        assert!(matches!(check(&chains)[..], [Failure::Missed(..)]));
    }
}
