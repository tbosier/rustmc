//! Exact posterior moments for testing the conjugate Gibbs samplers.
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
//! product, and each variance times each latent's squared deviation, which is
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

/// Exact moments of the quantities `q`: the inferred variances, in prior
/// order, then the latents, in the model's order.
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
        let mean = self.first[a];
        Moments {
            mean,
            sd: (self.product[a * self.quantity_count() + a] - mean * mean).sqrt(),
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

    fn largest_difference(&self, other: &Self) -> f64 {
        let count = self.quantity_count();
        let sd = |a: usize| self.moments(a).sd;
        let mut worst = 0.0f64;
        for a in 0..count {
            worst = worst.max((self.first[a] - other.first[a]).abs() / sd(a));
            for b in 0..count {
                let scale = sd(a) * sd(b);
                worst = worst.max((self.covariance(a, b) - other.covariance(a, b)).abs() / scale);
            }
        }
        for k in 0..self.variance_count {
            for i in 0..count - self.variance_count {
                let scale = self.first[k] * sd(self.variance_count + i).powi(2);
                worst = worst.max(
                    (self.variance_times_squared_deviation(k, i)
                        - other.variance_times_squared_deviation(k, i))
                    .abs()
                        / scale,
                );
            }
        }
        worst
    }
}

/// Exact moments, integrating each log variance over `points` equally spaced
/// values from `log_low` to `log_high` by the trapezoid rule. The same integral
/// over a grid two units wider at each end must agree to within `1e-6` of the
/// quantities' standard deviations, so a grid that cuts off a tail fails rather
/// than returning a wrong reference.
pub(crate) fn exact_posterior(
    model: &LinearGaussianModel,
    priors: &[InverseGammaPrior],
    log_low: f64,
    log_high: f64,
    points: usize,
) -> ExactPosterior {
    let step = (log_high - log_low) / (points - 1) as f64;
    let pad = (2.0 / step).ceil() as usize;
    let narrow = integrate(model, priors, log_low, step, points);
    let wide = integrate(
        model,
        priors,
        log_low - pad as f64 * step,
        step,
        points + 2 * pad,
    );
    let difference = narrow.largest_difference(&wide);
    assert!(
        difference < 1e-6,
        "widening the grid moved a moment by {difference} standard deviations"
    );
    narrow
}

fn integrate(
    model: &LinearGaussianModel,
    priors: &[InverseGammaPrior],
    log_low: f64,
    step: f64,
    points: usize,
) -> ExactPosterior {
    let dimension = priors.len();
    let observed = model.observed.len();
    let latent_count = model.latents.len();
    let count = dimension + latent_count;
    let residuals = model
        .observed
        .iter()
        .map(|(quantity, value)| value - quantity.constant)
        .collect::<Vec<_>>();

    // Unnormalized sums, rescaled whenever a larger log weight appears.
    let mut largest = f64::NEG_INFINITY;
    let mut total = 0.0;
    let mut first = vec![0.0; count];
    let mut product = vec![0.0; count * count];
    let mut variance_times_square = vec![0.0; dimension * latent_count];

    let mut index = vec![0usize; dimension];
    loop {
        let log_variances = index
            .iter()
            .map(|&i| log_low + step * i as f64)
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
        // The trapezoid rule halves the weight of each end of each axis.
        let log_trapezoid =
            index.iter().filter(|&&i| i == 0 || i == points - 1).count() as f64 * 0.5f64.ln();
        let log_weight = log_prior + log_trapezoid
            - half_log_determinant
            - 0.5 * whitened.iter().map(|z| z * z).sum::<f64>();

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

        if log_weight > largest {
            let rescale = (largest - log_weight).exp();
            total *= rescale;
            first.iter_mut().for_each(|value| *value *= rescale);
            product.iter_mut().for_each(|value| *value *= rescale);
            variance_times_square
                .iter_mut()
                .for_each(|value| *value *= rescale);
            largest = log_weight;
        }
        let w = (log_weight - largest).exp();
        total += w;
        // `E[q | variances]`, and `E[q_a q_b | variances]` for every pair.
        let conditional_first = variances.iter().chain(&means).copied().collect::<Vec<_>>();
        for a in 0..count {
            first[a] += w * conditional_first[a];
            for b in 0..count {
                let mut value = conditional_first[a] * conditional_first[b];
                if a >= dimension && b >= dimension {
                    value += conditional_covariance(a - dimension, b - dimension);
                }
                product[a * count + b] += w * value;
            }
        }
        for k in 0..dimension {
            for i in 0..latent_count {
                let square = conditional_covariance(i, i) + means[i] * means[i];
                variance_times_square[k * latent_count + i] += w * variances[k] * square;
            }
        }

        let mut axis = 0;
        loop {
            if axis == dimension {
                let normalize = |values: &mut Vec<f64>| values.iter_mut().for_each(|v| *v /= total);
                normalize(&mut first);
                normalize(&mut product);
                normalize(&mut variance_times_square);
                return ExactPosterior {
                    variance_count: dimension,
                    first,
                    product,
                    variance_times_square,
                };
            }
            index[axis] += 1;
            if index[axis] < points {
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

/// The tolerance in Monte Carlo standard errors, as in the NUTS exactness gate.
const MAX_MCSE: f64 = 4.0;
/// The smallest ESS a checked estimate may rest on, as in the NUTS gate, so
/// chains that disagree cannot pass by inflating an MCSE.
const MIN_ESS: f64 = 400.0;

/// A sampler's draws of every quantity of `exact`, in its order, as
/// `[quantity][chain][draw]` with each quantity's name.
pub(crate) type Draws = Vec<(String, Vec<Vec<f64>>)>;

/// Every check of `draws` against `exact`, returning a description of each
/// failure: each quantity's mean and mean squared deviation about the exact
/// mean, the mean of every pairwise product of deviations, and the mean of
/// each variance times each latent's squared deviation. Each must lie within
/// `MAX_MCSE` Monte Carlo standard errors of its exact value and rest on an
/// ESS of at least `MIN_ESS`.
pub(crate) fn check_posterior(draws: &Draws, exact: &ExactPosterior) -> Vec<String> {
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
    let mut failures = Vec::new();
    for a in 0..count {
        let name = &draws[a].0;
        check_mean(
            &format!("{name} mean"),
            &draws[a].1,
            exact.first[a],
            &mut failures,
        );
        for b in a..count {
            let products = elementwise(&deviations[a], &deviations[b], |x, y| x * y);
            let label = if a == b {
                format!("{name} variance")
            } else {
                format!("covariance of {name} and {}", draws[b].0)
            };
            check_mean(&label, &products, exact.covariance(a, b), &mut failures);
        }
    }
    for k in 0..exact.variance_count {
        for i in 0..count - exact.variance_count {
            let x = exact.variance_count + i;
            let products = elementwise(&draws[k].1, &deviations[x], |v, d| v * d * d);
            check_mean(
                &format!("{} times {}'s squared deviation", draws[k].0, draws[x].0),
                &products,
                exact.variance_times_squared_deviation(k, i),
                &mut failures,
            );
        }
    }
    failures
}

fn elementwise(a: &[Vec<f64>], b: &[Vec<f64>], combine: impl Fn(f64, f64) -> f64) -> Vec<Vec<f64>> {
    a.iter()
        .zip(b)
        .map(|(x, y)| x.iter().zip(y).map(|(p, q)| combine(*p, *q)).collect())
        .collect()
}

fn check_mean(name: &str, chains: &[Vec<f64>], exact: f64, failures: &mut Vec<String>) {
    let all = chains.iter().flatten().copied().collect::<Vec<_>>();
    let count = all.len() as f64;
    let mean = all.iter().sum::<f64>() / count;
    let sd = (all.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (count - 1.0)).sqrt();
    let ess = crate::diagnostics::ess_raw(chains);
    // A NaN ESS (draws that never move) or MCSE fails rather than passing
    // every comparison against it.
    if ess.is_nan() || ess < MIN_ESS {
        failures.push(format!("{name}: ESS {ess} is below {MIN_ESS}"));
        return;
    }
    let mcse = sd / ess.sqrt();
    let error = (mean - exact).abs() / mcse;
    if error.is_nan() || error >= MAX_MCSE {
        failures.push(format!(
            "{name}: {mean} is {error:.1} MCSE from the exact {exact}"
        ));
    }
}

/// Asserts that `draws` pass every check against `exact`, and that the checks
/// have the power to fail: they must reject the prior's moments for the
/// variances, and the same draws with their dependence broken.
pub(crate) fn assert_matches(draws: &Draws, exact: &ExactPosterior, priors: &[InverseGammaPrior]) {
    let failures = check_posterior(draws, exact);
    assert!(failures.is_empty(), "{failures:#?}");
    assert!(
        prior_is_rejected(draws, priors),
        "the checks cannot tell these draws from the prior"
    );
    assert!(
        !check_posterior(&decorrelated(draws), exact).is_empty(),
        "the checks cannot see the dependence between quantities"
    );
}

/// Whether `check_posterior` can tell `draws` from the prior: each variance's
/// draws checked against the prior's mean and variance instead of the
/// posterior's. A sampler that ignored the data would draw from the prior.
pub(crate) fn prior_is_rejected(draws: &Draws, priors: &[InverseGammaPrior]) -> bool {
    priors.iter().enumerate().all(|(k, prior)| {
        let moments = prior.moments();
        let mut failures = Vec::new();
        check_mean("prior mean", &draws[k].1, moments.mean, &mut failures);
        let squared = draws[k]
            .1
            .iter()
            .map(|chain| chain.iter().map(|v| (v - moments.mean).powi(2)).collect())
            .collect::<Vec<Vec<f64>>>();
        check_mean(
            "prior variance",
            &squared,
            moments.sd.powi(2),
            &mut failures,
        );
        !failures.is_empty()
    })
}

/// The same draws with each quantity shifted by a different number of draws
/// within its chain, so every marginal is unchanged and every dependence
/// between quantities is broken. `check_posterior` must reject it.
pub(crate) fn decorrelated(draws: &Draws) -> Draws {
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
    use super::TermVariance::Inferred;
    use super::*;

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
        let exact = exact_posterior(&model, &[prior], -8.0, 8.0, 120).variances()[0];
        let closed = InverseGammaPrior {
            shape: prior.shape + ys.len() as f64 / 2.0,
            scale: prior.scale + ys.iter().map(|y| y * y).sum::<f64>() / 2.0,
        }
        .moments();
        assert!((exact.mean - closed.mean).abs() < 1e-9 * closed.mean);
        assert!((exact.sd - closed.sd).abs() < 1e-9 * closed.sd);
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
        exact_posterior(&model, &[prior, prior], -9.0, 7.0, 150);
    }

    #[test]
    fn draws_that_never_move_or_chains_that_disagree_fail() {
        let exact = Moments { mean: 0.0, sd: 1.0 };
        let constant = vec![vec![999.0; 25_000]; 4];
        let mut failures = Vec::new();
        check_mean("constant", &constant, exact.mean, &mut failures);
        assert_eq!(failures.len(), 1, "{failures:?}");
        // Four stuck chains whose pooled mean is one standard deviation off
        // pass a bare MCSE test, since their ESS is about 4.
        let stuck = [-0.5, 0.5, 1.5, 2.5]
            .iter()
            .map(|&value| vec![value; 25_000])
            .collect::<Vec<_>>();
        let mut failures = Vec::new();
        check_mean("stuck", &stuck, exact.mean, &mut failures);
        assert!(failures[0].contains("ESS"), "{failures:?}");
    }
}
