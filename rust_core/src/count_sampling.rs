//! Stable Poisson densities and simulation over the exactly representable count range.
//!
//! Large rates use Hörmann's transformed rejection algorithm PTRS:
//! https://doi.org/10.1016/0167-6687(93)90997-4
//! (also described in NumPy's random_poisson_ptrs implementation).
//! Acceptance uses the Poisson mass in deviance form, avoiding subtraction of
//! terms of order rate * log(rate). No normal approximation or clipping is used.

use rand::{distributions::Open01, Rng};

pub(crate) const MAX_EXACT_COUNT: u64 = (1_u64 << 53) - 1;

/// Why a Poisson draw could not be returned exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CountSamplingError {
    /// The rate is negative, not finite, or above `2^53 - 1`.
    RateOutOfRange,
    /// The draw exceeded `2^53 - 1`, the largest count an `f64` holds exactly.
    DrawOutOfRange,
}

impl std::fmt::Display for CountSamplingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::RateOutOfRange => "Poisson rate outside the supported exact-count range",
            Self::DrawOutOfRange => "Poisson draw exceeds the supported exact-count range",
        })
    }
}

impl std::error::Error for CountSamplingError {}

pub(crate) fn poisson<R: Rng + ?Sized>(rate: f64, rng: &mut R) -> Result<f64, CountSamplingError> {
    if !rate.is_finite() || rate < 0.0 || rate > MAX_EXACT_COUNT as f64 {
        return Err(CountSamplingError::RateOutOfRange);
    }
    if rate == 0.0 {
        return Ok(0.0);
    }
    if rate < 10.0 {
        // Count exponential waiting times in an interval of length rate. This
        // still returns zero correctly when exp(-rate) would round to one.
        let mut count = 0.0;
        let mut elapsed = 0.0;
        loop {
            let uniform: f64 = rng.sample(Open01);
            elapsed -= uniform.ln();
            if elapsed > rate {
                return Ok(count);
            }
            count += 1.0;
        }
    }

    let b = 0.931 + 2.53 * rate.sqrt();
    let a = -0.059 + 0.02483 * b;
    let inverse_alpha = 1.1239 + 1.1328 / (b - 3.4);
    let squeeze = 0.9277 - 3.6224 / (b - 2.0);
    loop {
        let uniform: f64 = rng.sample(Open01);
        let u = uniform - 0.5;
        let v: f64 = rng.sample(Open01);
        let distance = 0.5 - u.abs();
        let candidate = ((2.0 * a / distance + b) * u + rate + 0.43).floor();
        if candidate < 0.0 || (distance < 0.013 && v > distance) {
            continue;
        }
        let accepted = (distance >= 0.07 && v <= squeeze)
            || (v.ln() + inverse_alpha.ln() - (a / distance.powi(2) + b).ln()
                <= log_mass(candidate, rate));
        if accepted {
            if candidate > MAX_EXACT_COUNT as f64 {
                // Reject the operation, not this draw: resampling would
                // silently condition the distribution on the output bound.
                return Err(CountSamplingError::DrawOutOfRange);
            }
            return Ok(candidate);
        }
    }
}

/// Poisson log mass with a log-rate input. Keeping eta for underflowed rates
/// preserves finite log probabilities for positive observations in that tail.
pub(crate) fn log_mass_from_log_rate(count: f64, log_rate: f64) -> f64 {
    log_mass_from_log_rate_given_rate(count, log_rate, log_rate.exp())
}

/// [`log_mass_from_log_rate`] for a caller that already holds
/// `rate = log_rate.exp()`, as the graph evaluator does to reuse it in the
/// gradient. The result is the same to the bit.
pub(crate) fn log_mass_from_log_rate_given_rate(count: f64, log_rate: f64, rate: f64) -> f64 {
    if !log_rate.is_finite() || !count.is_finite() || count < 0.0 || count.fract() != 0.0 {
        return f64::NEG_INFINITY;
    }
    if !rate.is_finite() {
        return f64::NEG_INFINITY;
    }
    if rate < f64::MIN_POSITIVE {
        // Subnormal exp results lose relative accuracy as well. Eta is still
        // precise and there is no large-term cancellation in this tail.
        return count * log_rate - rate - crate::autodiff::ln_gamma(count + 1.0);
    }
    log_mass(count, rate)
}

/// `ln(count!)` for `count` below 16, summed once in the order the log mass
/// used to sum it on every call, so the values are the same to the bit on any
/// platform's `ln`. Counts below 2 take the old empty sum itself, which is
/// `-0.0` rather than `0.0`, and no table lookup.
fn small_log_factorial(count: usize) -> f64 {
    if count < 2 {
        return (2..=count as u32).map(|k| (k as f64).ln()).sum();
    }
    small_log_factorials()[count]
}

/// The table behind [`small_log_factorial`], 105 logarithms built once per
/// process. Building an evaluator builds it, so a fit never pays for it
/// partway through sampling.
pub(crate) fn small_log_factorials() -> &'static [f64; 16] {
    static TABLE: std::sync::OnceLock<[f64; 16]> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        std::array::from_fn(|count| (2..=count as u32).map(|k| (k as f64).ln()).sum())
    })
}

pub(crate) fn log_mass(count: f64, rate: f64) -> f64 {
    if !count.is_finite() || count < 0.0 || count.fract() != 0.0 || !rate.is_finite() || rate < 0.0
    {
        return f64::NEG_INFINITY;
    }
    if rate == 0.0 {
        return if count == 0.0 { 0.0 } else { f64::NEG_INFINITY };
    }
    if count == 0.0 {
        return -rate;
    }
    if count < 16.0 {
        return count * rate.ln() - rate - small_log_factorial(count as usize);
    }
    let relative_delta = (count - rate) / rate;
    let deviance = if relative_delta.abs() < 0.125 {
        // (1+x) log(1+x) - x = sum_{n>=2} (-x)^n / (n(n-1)).
        // The first term is computed from the centered difference, preserving
        // its precision even when count and rate are both very large.
        let mut power = relative_delta * relative_delta;
        let mut sum = 0.5 * power;
        for n in 3..=24 {
            power *= -relative_delta;
            let next = sum + power / (n * (n - 1)) as f64;
            if next == sum {
                break;
            }
            sum = next;
        }
        rate * sum
    } else {
        let ratio = count / rate;
        let log_ratio = if ratio.is_finite() && ratio > 0.0 {
            ratio.ln()
        } else {
            count.ln() - rate.ln()
        };
        count * log_ratio + rate - count
    };
    let inverse = 1.0 / count;
    let inverse_squared = inverse * inverse;
    // Stirling's alternating remainder is below 1.1e-16 at count >= 16.
    let stirling_error = inverse
        * (1.0 / 12.0
            + inverse_squared
                * (-1.0 / 360.0
                    + inverse_squared
                        * (1.0 / 1260.0
                            + inverse_squared * (-1.0 / 1680.0 + inverse_squared / 1188.0))));
    -deviance - 0.5 * (std::f64::consts::TAU.ln() + count.ln()) - stirling_error
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    /// The log mass from a log rate before the table and the given rate, copied
    /// here as it was, so the new forms are compared with the old code rather
    /// than with themselves.
    fn log_mass_from_log_rate_before(count: f64, log_rate: f64) -> f64 {
        if !log_rate.is_finite() || !count.is_finite() || count < 0.0 || count.fract() != 0.0 {
            return f64::NEG_INFINITY;
        }
        let rate = log_rate.exp();
        if !rate.is_finite() {
            return f64::NEG_INFINITY;
        }
        if rate < f64::MIN_POSITIVE {
            return count * log_rate - rate - crate::autodiff::ln_gamma(count + 1.0);
        }
        if count == 0.0 {
            return -rate;
        }
        if count < 16.0 {
            let log_factorial: f64 = (2..=count as u32).map(|k| (k as f64).ln()).sum();
            return count * rate.ln() - rate - log_factorial;
        }
        // From 16 up the log mass is unchanged.
        log_mass(count, rate)
    }

    #[test]
    fn the_table_and_the_given_rate_match_the_old_log_mass_bit_for_bit() {
        for count in 0..16usize {
            let summed: f64 = (2..=count as u32).map(|k| (k as f64).ln()).sum();
            assert_eq!(
                small_log_factorial(count).to_bits(),
                summed.to_bits(),
                "{count}"
            );
        }
        let counts = [
            0.0,
            1.0,
            2.0,
            3.0,
            7.0,
            14.0,
            15.0,
            16.0,
            17.0,
            40.0,
            1e6,
            2.5,
            -1.0,
            f64::NAN,
        ];
        let log_rates = [
            -800.0,
            -745.0,
            -708.0,
            -30.0,
            -1.0,
            -0.0,
            0.0,
            0.7,
            2.0,
            3.0,
            20.0,
            709.0,
            710.0,
            f64::INFINITY,
            f64::NAN,
        ];
        for count in counts {
            for log_rate in log_rates {
                let before = log_mass_from_log_rate_before(count, log_rate);
                let given = log_mass_from_log_rate_given_rate(count, log_rate, log_rate.exp());
                let from_log = log_mass_from_log_rate(count, log_rate);
                let label = format!("count {count}, log rate {log_rate}");
                assert_eq!(given.to_bits(), before.to_bits(), "{label}");
                assert_eq!(from_log.to_bits(), before.to_bits(), "{label}");
            }
        }
    }

    #[test]
    fn out_of_range_rates_are_typed_errors_with_their_message() {
        let mut rng = ChaCha8Rng::seed_from_u64(4);
        for rate in [-1.0, f64::NAN, f64::INFINITY, 1e16] {
            assert_eq!(
                poisson(rate, &mut rng),
                Err(CountSamplingError::RateOutOfRange)
            );
        }
        assert_eq!(
            CountSamplingError::RateOutOfRange.to_string(),
            "Poisson rate outside the supported exact-count range"
        );
    }

    #[test]
    fn log_mass_matches_high_precision_reference_near_large_modes() {
        // 80-digit Decimal evaluation of k*log(rate)-rate-log(k!), with
        // exact factorials below 1000 and high-precision Stirling above.
        for (count, rate, expected) in [
            (16.0, 10.0, -3.830498618175942),
            (100.0, 100.0, -3.2223569567543533),
            (1e14, 1e14, -17.037034184162993),
            (100_000_010_000_000.0, 1e14, -17.537034217496325),
            (99_999_990_000_000.0, 1e14, -17.53703415082966),
            (8_000_000_100_000_000.0, 8e15, -19.853047505145766),
        ] {
            let actual = log_mass(count, rate);
            assert!((actual - expected).abs() < 5e-14, "{actual} != {expected}");
        }
    }

    #[test]
    fn tiny_rates_and_invalid_rates_preserve_count_support() {
        let mut rng = ChaCha8Rng::seed_from_u64(132);
        for rate in [0.0, f64::MIN_POSITIVE, 1e-20, 1e-16] {
            for _ in 0..1000 {
                let draw = poisson(rate, &mut rng).unwrap();
                assert!(draw >= 0.0 && draw.fract() == 0.0);
            }
        }
        for rate in [-1.0, f64::NAN, f64::INFINITY, 1e16] {
            assert!(poisson(rate, &mut rng).is_err());
        }
    }

    #[test]
    fn accepted_out_of_range_draws_error_instead_of_being_clipped_or_resampled() {
        let mut rng = ChaCha8Rng::seed_from_u64(991);
        let mut errors = 0;
        for _ in 0..100 {
            match poisson(MAX_EXACT_COUNT as f64, &mut rng) {
                Ok(value) => assert!(value <= MAX_EXACT_COUNT as f64),
                Err(_) => errors += 1,
            }
        }
        assert!(
            errors > 0,
            "the upper tail must report unrepresentable draws"
        );
    }

    #[test]
    fn poisson_moments_match_at_small_and_large_supported_rates() {
        let draws = 100_000;
        for rate in [0.1, 1.0, 9.9, 10.0, 100.0, 1e6, 1e12, 1e14, 1e15, 8e15] {
            let mut rng = ChaCha8Rng::seed_from_u64(5432);
            let mut sum = 0.0;
            let mut squares = 0.0;
            for _ in 0..draws {
                let draw = poisson(rate, &mut rng).unwrap();
                assert!(draw >= 0.0 && draw.fract() == 0.0);
                let centered = (draw - rate) / rate.sqrt();
                sum += centered;
                squares += centered * centered;
            }
            let mean = sum / draws as f64;
            let variance = squares / draws as f64 - mean * mean;
            assert!(
                mean.abs() < 7.0 / (draws as f64).sqrt(),
                "rate={rate}, mean={mean}"
            );
            let tolerance = 8.0 * ((2.0 + 1.0 / rate) / draws as f64).sqrt();
            assert!(
                (variance - 1.0).abs() < tolerance,
                "rate={rate}, variance={variance}"
            );
        }
    }

    #[test]
    fn rejection_sampler_matches_independent_poisson_probability_recurrence() {
        let rate: f64 = 10.0;
        let draws = 200_000;
        let mut histogram = [0; 31];
        let mut rng = ChaCha8Rng::seed_from_u64(891);
        for _ in 0..draws {
            let count = poisson(rate, &mut rng).unwrap() as usize;
            if count < histogram.len() {
                histogram[count] += 1;
            }
        }
        let mut probability = (-rate).exp();
        for (count, observed) in histogram.iter().enumerate() {
            let expected = probability * draws as f64;
            assert!((*observed as f64 - expected).abs() < 7.0 * expected.sqrt() + 2.0);
            probability *= rate / (count + 1) as f64;
        }
    }
}
