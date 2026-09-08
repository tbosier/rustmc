//! Integer-event payment development with prefix censoring and an explicit tail.
//!
//! One shared `p ~ Dirichlet(alpha)` pools cohort lag probabilities. Known ultimate
//! counts give multinomial allocations (Dirichlet-multinomial after integrating p).
//! Unknown ultimates use independent `lambda ~ Gamma(shape, rate)` and independent
//! `count[lag] ~ Poisson(lambda * p[lag])`. Currency is not a count observation.
//! The final category is an unscheduled tail, never a dated calendar payment.

use rand::{distributions::Open01, Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Beta, Binomial, Distribution, Gamma, Poisson};

const MAX_EXACT_COUNT: u64 = (1_u64 << 53) - 1;
const MAX_RETAINED_VALUES: usize = 25_000_000;

fn validate_allocation(factors: &[usize]) -> Result<(), String> {
    if factors
        .iter()
        .try_fold(1_usize, |n, factor| n.checked_mul(*factor))
        .is_none_or(|n| n > MAX_RETAINED_VALUES)
    {
        return Err("requested runoff allocation exceeds 25 million values; reduce cohorts, lags, chains, draws, or horizon".into());
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct RunoffConfig {
    pub alpha: Vec<f64>,
    /// Gamma prior shape for each unknown ultimate's Poisson intensity.
    pub total_shape: f64,
    /// Gamma prior rate, not scale.
    pub total_rate: f64,
    pub draws: usize,
    pub warmup: usize,
    pub chains: usize,
    pub seed: u64,
}

#[derive(Debug, Clone)]
pub struct PaymentTriangle {
    /// Incremental counts. None means unobserved; Some(0) is observed zero.
    pub counts: Vec<Vec<Option<u64>>>,
    /// Integer periods on one common calendar; development zero is origin.
    pub origins: Vec<i64>,
    pub valuation: i64,
    /// None infers an uncertain ultimate under the Gamma-Poisson prior.
    pub known_totals: Vec<Option<u64>>,
}

#[derive(Debug, Clone)]
pub struct RunoffDraw {
    pub lag_probabilities: Vec<f64>,
    /// Complete allocations, including the unchanged observed cells.
    pub counts: Vec<Vec<u64>>,
    pub totals: Vec<u64>,
    /// None for known-total cohorts.
    pub intensities: Vec<Option<f64>>,
}

#[derive(Debug, Clone)]
pub struct RunoffPosterior {
    pub triangle: PaymentTriangle,
    pub chains: Vec<Vec<RunoffDraw>>,
    /// Exact independent conjugate draws when every ultimate is known.
    pub exact: bool,
}

impl PaymentTriangle {
    pub fn validate(&self, categories: usize) -> Result<(), String> {
        if categories < 2 {
            return Err("alpha needs at least one regular lag and a final tail category".into());
        }
        if self.counts.is_empty()
            || self.origins.len() != self.counts.len()
            || self.known_totals.len() != self.counts.len()
        {
            return Err(
                "counts, origins, and totals must have the same nonzero cohort count".into(),
            );
        }
        for (i, row) in self.counts.iter().enumerate() {
            if row.len() != categories {
                return Err(format!(
                    "cohort {i}: count columns must match alpha including tail"
                ));
            }
            let mut sum = 0_u64;
            for (lag, count) in row.iter().enumerate() {
                let date = self.origins[i]
                    .checked_add(lag as i64)
                    .ok_or("origin plus development overflows calendar")?;
                if lag + 1 < categories && count.is_some() != (date <= self.valuation) {
                    return Err(format!("cohort {i}, lag {lag}: elapsed regular cells must be observed and future cells unobserved"));
                }
                if lag + 1 == categories && count.is_some() && date > self.valuation {
                    return Err(format!(
                        "cohort {i}: tail cannot close before its first possible period"
                    ));
                }
                if let Some(n) = count {
                    if *n > MAX_EXACT_COUNT {
                        return Err("counts must not exceed 2**53 - 1".into());
                    }
                    sum = sum.checked_add(*n).ok_or("observed count sum overflow")?;
                }
            }
            if sum > MAX_EXACT_COUNT {
                return Err("cohort observed total must not exceed 2**53 - 1".into());
            }
            if let Some(total) = self.known_totals[i] {
                if total > MAX_EXACT_COUNT || sum > total {
                    return Err(format!(
                        "cohort {i}: known total must cover observed counts and be <= 2**53 - 1"
                    ));
                }
                if row.iter().all(Option::is_some) && sum != total {
                    return Err(format!(
                        "cohort {i}: closed row must sum to its known total"
                    ));
                }
            }
        }
        Ok(())
    }
}

fn validate_config(config: &RunoffConfig) -> Result<(), String> {
    if config.alpha.len() < 2
        || config.alpha.iter().any(|a| !a.is_finite() || *a <= 0.0)
        || !config.alpha.iter().sum::<f64>().is_finite()
    {
        return Err("alpha must contain at least two finite positive concentrations".into());
    }
    if !config.total_shape.is_finite()
        || config.total_shape <= 0.0
        || !config.total_rate.is_finite()
        || config.total_rate <= 0.0
    {
        return Err("total_shape and total_rate must be finite and positive".into());
    }
    if config.draws == 0 || config.chains == 0 || config.draws.checked_add(config.warmup).is_none()
    {
        return Err(
            "draws and chains must be positive and draws + warmup must not overflow".into(),
        );
    }
    validate_allocation(&[config.draws, config.chains])?;
    Ok(())
}

fn beta_draw(a: f64, b: f64, rng: &mut ChaCha8Rng) -> Result<f64, String> {
    let q = Beta::new(a, b).map_err(|e| e.to_string())?.sample(rng);
    if !q.is_finite() {
        return Err("nonfinite Beta draw; rescale extreme prior or counts".into());
    }
    Ok(q)
}

fn dirichlet(alpha: &[f64], rng: &mut ChaCha8Rng) -> Result<Vec<f64>, String> {
    let mut remaining = 1.0;
    let mut p = Vec::with_capacity(alpha.len());
    for lag in 0..alpha.len() - 1 {
        let q = beta_draw(alpha[lag], alpha[lag + 1..].iter().sum(), rng)?;
        p.push(remaining * q);
        remaining *= 1.0 - q;
    }
    p.push(remaining);
    Ok(p)
}

fn binomial(n: u64, probability: f64, rng: &mut ChaCha8Rng) -> Result<u64, String> {
    let probability = probability.clamp(0.0, 1.0);
    let p = probability.min(1.0 - probability);
    // rand_distr 0.4's BINV uses powi(i32), so it incorrectly dispatches sparse
    // n > i32::MAX cases to BTPE, whose rejection envelope can be invalid.
    // Exact geometric waiting times handle these rare-event binomials in O(np).
    if n > i32::MAX as u64 && n as f64 * p < 10.0 && p > 0.0 {
        let mut remaining = n;
        let mut events = 0_u64;
        let log_failure = (-p).ln_1p();
        while remaining > 0 {
            let uniform: f64 = rng.sample(Open01);
            let failures = (uniform.ln() / log_failure).floor();
            if failures >= remaining as f64 {
                break;
            }
            remaining -= failures as u64 + 1;
            events += 1;
        }
        return Ok(if probability <= 0.5 {
            events
        } else {
            n - events
        });
    }
    Ok(Binomial::new(n, probability)
        .map_err(|e| e.to_string())?
        .sample(rng))
}

fn poisson(mean: f64, rng: &mut ChaCha8Rng) -> Result<u64, String> {
    if mean == 0.0 {
        return Ok(0);
    }
    if !mean.is_finite() || mean > MAX_EXACT_COUNT as f64 {
        return Err("Poisson mean exceeds the supported exact-count range".into());
    }
    let value = Poisson::new(mean).map_err(|e| e.to_string())?.sample(rng);
    if value > MAX_EXACT_COUNT as f64 {
        return Err("Poisson draw exceeds the supported exact-count range".into());
    }
    Ok(value as u64)
}

/// Independent posterior stick-breaking hazards under known-total prefix censoring.
/// A cohort contributes successes x_l and failures N - sum_{j<=l} x_j only at
/// elapsed lags. The prior hazards are independent Beta(alpha_l, sum_{j>l} alpha_j).
pub fn known_total_hazard_posterior(
    triangle: &PaymentTriangle,
    alpha: &[f64],
) -> Result<Vec<(f64, f64)>, String> {
    triangle.validate(alpha.len())?;
    if alpha.iter().any(|a| !a.is_finite() || *a <= 0.0) || !alpha.iter().sum::<f64>().is_finite() {
        return Err("alpha must be finite and positive".into());
    }
    let mut params = (0..alpha.len() - 1)
        .map(|l| (alpha[l], alpha[l + 1..].iter().sum::<f64>()))
        .collect::<Vec<_>>();
    for (cohort, row) in triangle.counts.iter().enumerate() {
        let mut remaining = triangle.known_totals[cohort].ok_or("all totals must be known")?;
        for (lag, cell) in row.iter().take(alpha.len() - 1).enumerate() {
            if let Some(n) = cell {
                remaining -= n;
                params[lag].0 += *n as f64;
                params[lag].1 += remaining as f64;
            }
        }
    }
    Ok(params)
}

/// Fit native runoff, preserving chain/draw/cohort/lag alignment.
/// Known-only fits use exact conjugate sampling; other fits use latent-count Gibbs.
pub fn fit_runoff(
    triangle: &PaymentTriangle,
    config: &RunoffConfig,
) -> Result<RunoffPosterior, String> {
    validate_config(config)?;
    triangle.validate(config.alpha.len())?;
    validate_allocation(&[
        config.chains,
        config.draws,
        triangle
            .counts
            .len()
            .checked_add(1)
            .ok_or("cohort count overflow")?,
        config
            .alpha
            .len()
            .checked_add(3)
            .ok_or("lag count overflow")?,
    ])?;
    let exact = triangle.known_totals.iter().all(Option::is_some);
    let hazards = if exact {
        Some(known_total_hazard_posterior(triangle, &config.alpha)?)
    } else {
        None
    };
    let categories = config.alpha.len();
    let observed_sums = triangle
        .counts
        .iter()
        .map(|r| r.iter().flatten().sum::<u64>())
        .collect::<Vec<_>>();
    let mut chains = Vec::with_capacity(config.chains);
    for chain in 0..config.chains {
        let mut rng = ChaCha8Rng::seed_from_u64(
            config
                .seed
                .wrapping_add((chain as u64).wrapping_mul(0x9e3779b97f4a7c15)),
        );
        let mut p = dirichlet(&config.alpha, &mut rng)?;
        let warmup = if exact { 0 } else { config.warmup };
        let mut retained = Vec::with_capacity(config.draws);
        for iteration in 0..warmup + config.draws {
            let mut exact_hazards = Vec::new();
            if let Some(params) = &hazards {
                let mut rest = 1.0;
                for (lag, (a, b)) in params.iter().enumerate() {
                    let q = beta_draw(*a, *b, &mut rng)?;
                    exact_hazards.push(q);
                    p[lag] = rest * q;
                    rest *= 1.0 - q;
                }
                p[categories - 1] = rest;
            }
            let mut counts = Vec::with_capacity(triangle.counts.len());
            let mut totals = Vec::with_capacity(triangle.counts.len());
            let mut intensities = Vec::with_capacity(triangle.counts.len());
            for (i, row) in triangle.counts.iter().enumerate() {
                let mut full = row.iter().map(|v| v.unwrap_or(0)).collect::<Vec<_>>();
                if let Some(total) = triangle.known_totals[i] {
                    let mut remaining = total - observed_sums[i];
                    for lag in 0..categories {
                        if row[lag].is_none() {
                            if lag + 1 == categories {
                                full[lag] = remaining;
                            } else {
                                let q = if exact {
                                    exact_hazards[lag]
                                } else {
                                    let mass: f64 = p[lag..].iter().sum();
                                    if mass == 0.0 && remaining > 0 {
                                        return Err("unobserved lag probability underflow; use less extreme priors".into());
                                    }
                                    if mass == 0.0 {
                                        0.0
                                    } else {
                                        p[lag] / mass
                                    }
                                };
                                full[lag] = binomial(remaining, q, &mut rng)?;
                            }
                            remaining -= full[lag];
                        }
                    }
                    intensities.push(None);
                } else {
                    // Marginal lambda conditional on observed cells, then regenerate
                    // missing counts. This blocked update avoids conditioning lambda
                    // on stale latent counts from an immature cohort.
                    let exposure: f64 = row.iter().zip(&p).filter_map(|(n, p)| n.map(|_| p)).sum();
                    let intensity = Gamma::new(
                        config.total_shape + observed_sums[i] as f64,
                        1.0 / (config.total_rate + exposure),
                    )
                    .map_err(|e| e.to_string())?
                    .sample(&mut rng);
                    if !intensity.is_finite() {
                        return Err("nonfinite ultimate intensity draw".into());
                    }
                    for lag in 0..categories {
                        if row[lag].is_none() {
                            full[lag] = poisson(intensity * p[lag], &mut rng)?;
                        }
                    }
                    intensities.push(Some(intensity));
                }
                let total = full
                    .iter()
                    .try_fold(0_u64, |sum, n| sum.checked_add(*n))
                    .ok_or("ultimate count overflow")?;
                if total > MAX_EXACT_COUNT {
                    return Err("ultimate count exceeds 2**53 - 1".into());
                }
                totals.push(total);
                counts.push(full);
            }
            if !exact {
                let mut posterior_alpha = config.alpha.clone();
                for row in &counts {
                    for (a, n) in posterior_alpha.iter_mut().zip(row) {
                        *a += *n as f64;
                    }
                }
                p = dirichlet(&posterior_alpha, &mut rng)?;
            }
            if iteration >= warmup {
                retained.push(RunoffDraw {
                    lag_probabilities: p.clone(),
                    counts,
                    totals,
                    intensities,
                });
            }
        }
        chains.push(retained);
    }
    Ok(RunoffPosterior {
        triangle: triangle.clone(),
        chains,
        exact,
    })
}

impl RunoffPosterior {
    /// Parameter diagnostics cover the shared lag vector and each unknown cohort's
    /// ultimate intensity and count. Observed cells and fixed known totals are excluded.
    pub fn diagnostics(&self) -> crate::diagnostics::DiagnosticsReport {
        let mut names = (0..self.triangle.counts[0].len())
            .map(|lag| format!("lag_probability[{lag}]"))
            .collect::<Vec<_>>();
        for (i, total) in self.triangle.known_totals.iter().enumerate() {
            if total.is_none() {
                names.push(format!("intensity[{i}]"));
                names.push(format!("ultimate[{i}]"));
            }
        }
        let samples = self
            .chains
            .iter()
            .map(|chain| {
                chain
                    .iter()
                    .map(|draw| {
                        let mut values = draw.lag_probabilities.clone();
                        for (i, intensity) in draw.intensities.iter().enumerate() {
                            if let Some(intensity) = intensity {
                                values.push(*intensity);
                                values.push(draw.totals[i] as f64);
                            }
                        }
                        values
                    })
                    .collect()
            })
            .collect::<Vec<_>>();
        crate::forecast_diagnostics::parameter_diagnostics(&samples, &names)
    }

    /// Future calendar counts for valuation+1,...,valuation+steps. This excludes
    /// tail and regular cells beyond the requested horizon; no tail timing is assumed.
    pub fn calendar_samples(&self, steps: usize) -> Result<Vec<Vec<Vec<u64>>>, String> {
        if steps == 0
            || steps > i64::MAX as usize
            || self.triangle.valuation.checked_add(steps as i64).is_none()
        {
            return Err("steps must be positive and fit the integer calendar".into());
        }
        validate_allocation(&[
            self.chains.len(),
            self.chains.first().map_or(0, Vec::len),
            steps,
        ])?;
        self.chains
            .iter()
            .map(|chain| {
                chain
                    .iter()
                    .map(|draw| {
                        let mut calendar = vec![0_u64; steps];
                        for (i, row) in draw.counts.iter().enumerate() {
                            for (lag, n) in row.iter().take(row.len() - 1).enumerate() {
                                let date = self.triangle.origins[i] + lag as i64;
                                let offset = date as i128 - self.triangle.valuation as i128 - 1;
                                if offset >= 0 && offset < steps as i128 {
                                    calendar[offset as usize] = calendar[offset as usize]
                                        .checked_add(*n)
                                        .ok_or("calendar count overflow")?;
                                }
                            }
                        }
                        Ok(calendar)
                    })
                    .collect()
            })
            .collect()
    }

    /// Unobserved tail counts, preserving cohorts. Closed observed tails return zero.
    pub fn tail_samples(&self) -> Vec<Vec<Vec<u64>>> {
        self.chains
            .iter()
            .map(|chain| {
                chain
                    .iter()
                    .map(|draw| {
                        draw.counts
                            .iter()
                            .enumerate()
                            .map(|(i, row)| {
                                if self.triangle.counts[i].last().unwrap().is_none() {
                                    *row.last().unwrap()
                                } else {
                                    0
                                }
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(alpha: Vec<f64>) -> RunoffConfig {
        RunoffConfig {
            alpha,
            total_shape: 4.0,
            total_rate: 0.2,
            draws: 5000,
            warmup: 300,
            chains: 2,
            seed: 182,
        }
    }

    fn mean_var(values: &[f64]) -> (f64, f64) {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        (
            mean,
            values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64,
        )
    }

    #[test]
    fn complete_triangle_matches_dirichlet_moments() {
        let triangle = PaymentTriangle {
            counts: vec![
                vec![Some(8), Some(3), Some(1)],
                vec![Some(4), Some(5), Some(3)],
            ],
            origins: vec![0, 1],
            valuation: 5,
            known_totals: vec![Some(12), Some(12)],
        };
        let fit = fit_runoff(&triangle, &config(vec![2.0, 3.0, 1.0])).unwrap();
        assert!(fit.exact);
        let posterior = [14.0, 11.0, 5.0];
        for (lag, a) in posterior.iter().enumerate() {
            let samples = fit
                .chains
                .iter()
                .flatten()
                .map(|d| d.lag_probabilities[lag])
                .collect::<Vec<_>>();
            let (mean, variance) = mean_var(&samples);
            assert!((mean - a / 30.0).abs() < 0.003);
            assert!((variance - a * (30.0 - a) / (30.0_f64.powi(2) * 31.0)).abs() < 0.0004);
        }
        assert!(fit
            .tail_samples()
            .iter()
            .flatten()
            .flatten()
            .all(|n| *n == 0));
    }

    #[test]
    fn immature_zero_updates_exposure_and_future_is_beta_binomial() {
        let triangle = PaymentTriangle {
            counts: vec![vec![Some(0), None, None]],
            origins: vec![0],
            valuation: 0,
            known_totals: vec![Some(10)],
        };
        let cfg = config(vec![2.0, 3.0, 5.0]);
        assert_eq!(
            known_total_hazard_posterior(&triangle, &cfg.alpha).unwrap(),
            vec![(2.0, 18.0), (3.0, 5.0)]
        );
        let fit = fit_runoff(&triangle, &cfg).unwrap();
        let samples = fit
            .chains
            .iter()
            .flatten()
            .map(|d| d.counts[0][1] as f64)
            .collect::<Vec<_>>();
        let (mean, variance) = mean_var(&samples);
        assert!((mean - 10.0 * 3.0 / 8.0).abs() < 0.07);
        let expected_variance = 10.0 * 3.0 * 5.0 * 18.0 / (64.0 * 9.0);
        assert!((variance - expected_variance).abs() < 0.2);
        let p0 = fit
            .chains
            .iter()
            .flatten()
            .map(|d| d.lag_probabilities[0])
            .sum::<f64>()
            / 10000.0;
        assert!((p0 - 0.1).abs() < 0.003);
    }

    #[test]
    fn masks_totals_seed_and_calendar_alignment() {
        let triangle = PaymentTriangle {
            counts: vec![vec![Some(4), None, None], vec![None, None, None]],
            origins: vec![0, 1],
            valuation: 0,
            known_totals: vec![Some(9), Some(6)],
        };
        let mut cfg = config(vec![3.0, 2.0, 1.0]);
        cfg.draws = 20;
        let fit = fit_runoff(&triangle, &cfg).unwrap();
        let repeat = fit_runoff(&triangle, &cfg).unwrap();
        let calendar = fit.calendar_samples(2).unwrap();
        let tails = fit.tail_samples();
        for c in 0..cfg.chains {
            for d in 0..cfg.draws {
                let draw = &fit.chains[c][d];
                assert_eq!(draw.counts, repeat.chains[c][d].counts);
                assert_eq!(
                    draw.lag_probabilities,
                    repeat.chains[c][d].lag_probabilities
                );
                assert_eq!(draw.totals, vec![9, 6]);
                assert_eq!(draw.counts[0][0], 4);
                assert_eq!(calendar[c][d][0], draw.counts[0][1] + draw.counts[1][0]);
                assert_eq!(calendar[c][d][1], draw.counts[1][1]);
                assert_eq!(
                    calendar[c][d].iter().sum::<u64>() + tails[c][d].iter().sum::<u64>() + 4,
                    15
                );
            }
        }
        assert_ne!(
            fit.chains[0][0].lag_probabilities,
            fit.chains[1][0].lag_probabilities
        );
        let mut invalid = triangle.clone();
        invalid.counts[0][1] = Some(0);
        assert!(fit_runoff(&invalid, &cfg).is_err());
        invalid = triangle.clone();
        invalid.counts[0][0] = None;
        assert!(fit_runoff(&invalid, &cfg).is_err());
        invalid = triangle.clone();
        invalid.known_totals[0] = Some(3);
        assert!(fit_runoff(&invalid, &cfg).is_err());
    }

    #[test]
    fn unknown_unobserved_ultimate_matches_gamma_poisson_prior() {
        let triangle = PaymentTriangle {
            counts: vec![vec![None, None, None]],
            origins: vec![2],
            valuation: 0,
            known_totals: vec![None],
        };
        let fit = fit_runoff(&triangle, &config(vec![2.0, 3.0, 5.0])).unwrap();
        assert!(!fit.exact);
        let (mean, variance) = mean_var(
            &fit.chains
                .iter()
                .flatten()
                .map(|d| d.totals[0] as f64)
                .collect::<Vec<_>>(),
        );
        // Gamma-Poisson mean a/b, variance a/b + a/b^2.
        assert!((mean - 20.0).abs() < 0.4);
        assert!((variance - 120.0).abs() < 7.0);
        let p = fit
            .chains
            .iter()
            .flatten()
            .map(|d| d.lag_probabilities[0])
            .sum::<f64>()
            / 10000.0;
        assert!((p - 0.2).abs() < 0.015);
    }

    #[test]
    fn large_sparse_binomial_is_exact_and_bounded_allocations_reject_oversize() {
        let mut rng = ChaCha8Rng::seed_from_u64(31);
        let samples = (0..20000)
            .map(|_| binomial(10_000_000_000, 2e-10, &mut rng).unwrap() as f64)
            .collect::<Vec<_>>();
        let (mean, variance) = mean_var(&samples);
        assert!((mean - 2.0).abs() < 0.03);
        assert!((variance - 2.0).abs() < 0.07);
        let complements = (0..20000)
            .map(|_| {
                (10_000_000_000 - binomial(10_000_000_000, 1.0 - 2e-10, &mut rng).unwrap()) as f64
            })
            .collect::<Vec<_>>();
        assert!((mean_var(&complements).0 - 2.0).abs() < 0.03);
        let triangle = PaymentTriangle {
            counts: vec![vec![Some(0), None], vec![None, None]],
            origins: vec![0, 1],
            valuation: 0,
            known_totals: vec![Some(10_000_000_000); 2],
        };
        let mut cfg = config(vec![1.0, 1.0]);
        cfg.draws = 10;
        let fit = fit_runoff(&triangle, &cfg).unwrap();
        assert!(fit
            .chains
            .iter()
            .flatten()
            .all(|d| d.totals == vec![10_000_000_000; 2]));
        assert!(fit.calendar_samples(usize::MAX / 8).is_err());
        cfg.chains = usize::MAX;
        assert!(fit_runoff(&triangle, &cfg).is_err());
        cfg.chains = 1;
        cfg.draws = MAX_RETAINED_VALUES;
        assert!(fit_runoff(&triangle, &cfg).is_err());
    }

    #[test]
    fn known_and_unknown_cohorts_recover_shared_lags() {
        let mut rng = ChaCha8Rng::seed_from_u64(612);
        let truth = [0.5, 0.3, 0.15, 0.05];
        let mut triangle = PaymentTriangle {
            counts: vec![],
            origins: vec![],
            valuation: 5,
            known_totals: vec![],
        };
        for i in 0..120 {
            let total = poisson(200.0, &mut rng).unwrap();
            let mut remaining = total;
            let mut row = Vec::new();
            let origin = i % 7;
            for lag in 0..4 {
                let n = if lag == 3 {
                    remaining
                } else {
                    binomial(
                        remaining,
                        truth[lag] / truth[lag..].iter().sum::<f64>(),
                        &mut rng,
                    )
                    .unwrap()
                };
                remaining -= n;
                row.push(if origin + lag <= 5 { Some(n) } else { None });
            }
            triangle.counts.push(row);
            triangle.origins.push(origin as i64);
            triangle
                .known_totals
                .push(if i % 3 == 0 { Some(total) } else { None });
        }
        let mut cfg = config(vec![1.0; 4]);
        cfg.total_shape = 20.0;
        cfg.total_rate = 0.1;
        cfg.draws = 700;
        let fit = fit_runoff(&triangle, &cfg).unwrap();
        for (lag, expected) in truth.iter().enumerate() {
            let mean = fit
                .chains
                .iter()
                .flatten()
                .map(|d| d.lag_probabilities[lag])
                .sum::<f64>()
                / (cfg.chains * cfg.draws) as f64;
            assert!(
                (mean - expected).abs() < 0.015,
                "lag {lag}: {mean} vs {expected}"
            );
        }
        for draw in fit.chains.iter().flatten() {
            for (i, row) in triangle.counts.iter().enumerate() {
                for (lag, n) in row.iter().enumerate() {
                    if let Some(n) = n {
                        assert_eq!(draw.counts[i][lag], *n);
                    }
                }
                assert_eq!(draw.counts[i].iter().sum::<u64>(), draw.totals[i]);
            }
        }
    }
}
