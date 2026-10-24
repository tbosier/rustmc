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
//! latent's. The moments come from the trapezoid rule on that grid, which for a
//! density this smooth and this well inside the grid is accurate to far below
//! the Monte Carlo error it is compared with.

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

pub(crate) struct ExactPosterior {
    pub variances: Vec<Moments>,
    pub latents: Vec<Moments>,
}

/// Posterior moments of every inferred variance and every latent, integrating
/// each variance over `points` log-spaced values from `exp(log_low)` to
/// `exp(log_high)`.
pub(crate) fn exact_posterior(
    model: &LinearGaussianModel,
    priors: &[InverseGammaPrior],
    log_low: f64,
    log_high: f64,
    points: usize,
) -> ExactPosterior {
    let dimension = priors.len();
    let observed = model.observed.len();
    let step = (log_high - log_low) / (points - 1) as f64;
    let grid = (0..points)
        .map(|i| log_low + step * i as f64)
        .collect::<Vec<_>>();
    let residuals = model
        .observed
        .iter()
        .map(|(quantity, value)| value - quantity.constant)
        .collect::<Vec<_>>();

    let mut evaluations = Vec::new();
    let mut index = vec![0usize; dimension];
    loop {
        let log_variances = index.iter().map(|&i| grid[i]).collect::<Vec<_>>();
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
        let log_determinant = (0..observed)
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
        let log_weight =
            log_prior - log_determinant - 0.5 * whitened.iter().map(|z| z * z).sum::<f64>();
        let latent_moments = model
            .latents
            .iter()
            .map(|latent| {
                let cross = model
                    .observed
                    .iter()
                    .map(|(quantity, _)| covariance(latent, quantity))
                    .collect::<Vec<_>>();
                let whitened_cross = forward_solve(&lower, observed, &cross);
                let mean = latent.constant
                    + whitened_cross
                        .iter()
                        .zip(&whitened)
                        .map(|(c, z)| c * z)
                        .sum::<f64>();
                let variance =
                    covariance(latent, latent) - whitened_cross.iter().map(|c| c * c).sum::<f64>();
                (mean, variance)
            })
            .collect::<Vec<_>>();
        let on_edge = index.iter().any(|&i| i == 0 || i == points - 1);
        evaluations.push((log_weight, variances, latent_moments, on_edge));

        let mut axis = 0;
        loop {
            if axis == dimension {
                return summarize(&evaluations, &index_weights(&evaluations), dimension);
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

/// Log weight, variances, each latent's conditional mean and variance, and
/// whether the point lies on the grid's edge.
type Evaluation = (f64, Vec<f64>, Vec<(f64, f64)>, bool);

/// Normalized weights. Equal spacing in every log variance makes the trapezoid
/// rule's interior weights equal, and the grid is wide enough that its edges
/// carry no mass, so the plain average is the trapezoid rule here.
fn index_weights(evaluations: &[Evaluation]) -> Vec<f64> {
    let largest = evaluations
        .iter()
        .map(|(log_weight, ..)| *log_weight)
        .fold(f64::NEG_INFINITY, f64::max);
    let weights = evaluations
        .iter()
        .map(|(log_weight, ..)| (log_weight - largest).exp())
        .collect::<Vec<_>>();
    let total = weights.iter().sum::<f64>();
    weights.into_iter().map(|w| w / total).collect()
}

fn summarize(evaluations: &[Evaluation], weights: &[f64], dimension: usize) -> ExactPosterior {
    let moments = |first: f64, second: f64| Moments {
        mean: first,
        sd: (second - first * first).sqrt(),
    };
    let variances = (0..dimension)
        .map(|k| {
            let (first, second) = evaluations.iter().zip(weights).fold(
                (0.0, 0.0),
                |(first, second), ((_, variances, ..), w)| {
                    (
                        first + w * variances[k],
                        second + w * variances[k] * variances[k],
                    )
                },
            );
            moments(first, second)
        })
        .collect();
    let latent_count = evaluations[0].2.len();
    let latents = (0..latent_count)
        .map(|k| {
            let (first, second) = evaluations.iter().zip(weights).fold(
                (0.0, 0.0),
                |(first, second), ((_, _, latents, _), w)| {
                    let (mean, variance) = latents[k];
                    (first + w * mean, second + w * (variance + mean * mean))
                },
            );
            moments(first, second)
        })
        .collect();
    // The edge weights must be negligible for the plain average to be the
    // trapezoid rule and for the grid to hold the whole posterior.
    let edge_mass = evaluations
        .iter()
        .zip(weights)
        .filter(|((.., on_edge), _)| *on_edge)
        .map(|(_, w)| w)
        .sum::<f64>();
    assert!(
        edge_mass < 1e-9,
        "the grid edges hold {edge_mass} of the posterior"
    );
    ExactPosterior { variances, latents }
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

/// The checks a sampler's draws of one scalar must pass against its exact
/// moments, in the form the NUTS exactness gate uses: the mean within
/// `MAX_MCSE` Monte Carlo standard errors, and the mean squared deviation about
/// the exact mean within `MAX_MCSE` of its own standard errors of the exact
/// variance. Returns a description of each failure.
pub(crate) fn check_draws(name: &str, chains: &[Vec<f64>], exact: Moments) -> Vec<String> {
    const MAX_MCSE: f64 = 4.0;
    let mut failures = Vec::new();
    let all = chains.iter().flatten().copied().collect::<Vec<_>>();
    let count = all.len() as f64;
    let mean = all.iter().sum::<f64>() / count;
    let mean_mcse = standard_deviation(&all) / crate::diagnostics::ess_raw(chains).sqrt();
    if (mean - exact.mean).abs() >= MAX_MCSE * mean_mcse {
        failures.push(format!(
            "{name}: mean {mean} is {:.1} MCSE from the exact {}",
            (mean - exact.mean).abs() / mean_mcse,
            exact.mean
        ));
    }
    let squared = chains
        .iter()
        .map(|chain| {
            chain
                .iter()
                .map(|x| (x - exact.mean).powi(2))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let flat = squared.iter().flatten().copied().collect::<Vec<_>>();
    let second = flat.iter().sum::<f64>() / count;
    let second_mcse = standard_deviation(&flat) / crate::diagnostics::ess_raw(&squared).sqrt();
    let exact_variance = exact.sd * exact.sd;
    if (second - exact_variance).abs() >= MAX_MCSE * second_mcse {
        failures.push(format!(
            "{name}: mean squared deviation {second} is {:.1} MCSE from the exact variance \
             {exact_variance}",
            (second - exact_variance).abs() / second_mcse,
        ));
    }
    failures
}

fn standard_deviation(values: &[f64]) -> f64 {
    let count = values.len() as f64;
    let mean = values.iter().sum::<f64>() / count;
    (values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (count - 1.0)).sqrt()
}
