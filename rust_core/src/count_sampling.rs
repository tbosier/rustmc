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
///
/// Below 16, `count * log_rate` stands in for the `count * rate.ln()` of
/// [`log_mass`]. The log rate is exact, where the logarithm of its rounded
/// exponential carries that rounding, about 1e-16, and costs a call.
pub(crate) fn log_mass_from_log_rate_given_rate(count: f64, log_rate: f64, rate: f64) -> f64 {
    if !log_rate.is_finite() || !count.is_finite() || count < 0.0 || !is_whole(count) {
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
    if count == 0.0 {
        return -rate;
    }
    if count < 16.0 {
        return count * log_rate - rate - small_log_factorial(count as usize);
    }
    large_count_log_mass(count, rate)
}

/// Whether `count`, finite and not negative, is a whole number: the same
/// answer as `count.fract() == 0.0`, without the `trunc` call that `fract`
/// compiles to on x86-64 without SSE4.1, once per observation per gradient.
#[inline]
fn is_whole(count: f64) -> bool {
    // Every float from 2^52 up is whole, and below it the conversion to an
    // integer truncates exactly and converts back exactly.
    count >= 4_503_599_627_370_496.0 || (count as u64) as f64 == count
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
/// process. Building a graph evaluator builds it, so a graph fit never pays
/// for it partway through sampling; other callers of the log mass build it
/// on first use.
pub(crate) fn small_log_factorials() -> &'static [f64; 16] {
    static TABLE: std::sync::OnceLock<[f64; 16]> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        std::array::from_fn(|count| (2..=count as u32).map(|k| (k as f64).ln()).sum())
    })
}

pub(crate) fn log_mass(count: f64, rate: f64) -> f64 {
    if !count.is_finite() || count < 0.0 || !is_whole(count) || !rate.is_finite() || rate < 0.0 {
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
    large_count_log_mass(count, rate)
}

/// [`log_mass`] for a whole `count` of at least 16 and a finite, positive
/// `rate`, which the callers have checked.
fn large_count_log_mass(count: f64, rate: f64) -> f64 {
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

    /// The log mass from a log rate before the table, the given rate and the
    /// log rate below 16, copied here as it was, so the new forms are compared
    /// with the old code rather than with themselves.
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
        log_mass_before(count, rate)
    }

    /// `log_mass` as it was before its large-count body moved into
    /// `large_count_log_mass` and it took `is_whole`, copied verbatim, so the
    /// rate-input log mass and the log-rate path's large counts are compared
    /// with the old code rather than with the moved one.
    fn log_mass_before(count: f64, rate: f64) -> f64 {
        if !count.is_finite()
            || count < 0.0
            || count.fract() != 0.0
            || !rate.is_finite()
            || rate < 0.0
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

    /// How many representable doubles lie between two finite values: the
    /// difference of their bit patterns, ordered so that negative values sort
    /// below positive ones and the two zeros coincide.
    fn ulps_apart(a: f64, b: f64) -> u64 {
        let ordered = |x: f64| {
            let bits = x.to_bits() as i64;
            if bits < 0 {
                i64::MIN - bits
            } else {
                bits
            }
        };
        ordered(a).abs_diff(ordered(b))
    }

    /// Only a count from 1 to 15 at a normal, finite rate takes a new form,
    /// `count * log_rate` in place of `count * rate.ln()`. Every other input
    /// keeps the old log mass to the bit, and the new form stays within 4
    /// units in the last place of the old one.
    #[test]
    fn the_log_rate_forms_change_only_small_positive_counts() {
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
        let mut changed = 0;
        for count in counts {
            for log_rate in log_rates
                .iter()
                .copied()
                .chain((-40..=40).map(|k| 0.0137 * k as f64))
            {
                let before = log_mass_from_log_rate_before(count, log_rate);
                let rate = log_rate.exp();
                let given = log_mass_from_log_rate_given_rate(count, log_rate, rate);
                let from_log = log_mass_from_log_rate(count, log_rate);
                let label = format!("count {count}, log rate {log_rate}");
                assert_eq!(from_log.to_bits(), given.to_bits(), "{label}");
                let small = (1.0..16.0).contains(&count) && count.fract() == 0.0;
                if small && rate.is_finite() && rate >= f64::MIN_POSITIVE {
                    let expected = count * log_rate - rate - small_log_factorial(count as usize);
                    assert_eq!(given.to_bits(), expected.to_bits(), "{label}");
                    let apart = ulps_apart(given, before);
                    assert!(apart <= 4, "{label}: {given} {before}, {apart} ulps");
                    changed += (apart > 0) as usize;
                } else {
                    assert_eq!(given.to_bits(), before.to_bits(), "{label}");
                }
            }
        }
        // Rounding in `ln(exp(log_rate))` moves some of these by a bit.
        assert!(changed > 0);
    }

    /// Whether the large-count log mass at `count` and `rate` takes its
    /// deviance series, as `large_count_log_mass` decides.
    fn takes_the_series(count: f64, rate: f64) -> bool {
        count >= 16.0 && rate.is_finite() && rate > 0.0 && ((count - rate) / rate).abs() < 0.125
    }

    /// Large counts at log rates near `ln(count)`, where the large-count log
    /// mass sums its deviance series, keep the old log mass to the bit on the
    /// log-rate path.
    #[test]
    fn large_counts_near_their_rate_keep_the_old_log_mass_bit_for_bit() {
        let counts = [16.0, 17.0, 40.0, 1000.0, 1e6, 1e14];
        let offsets = [-0.11, -0.05, -1e-3, -1e-9, 0.0, 1e-9, 1e-3, 0.05, 0.11];
        let mut series = 0;
        for count in counts {
            for offset in offsets {
                let log_rate = f64::ln(count) + offset;
                let before = log_mass_from_log_rate_before(count, log_rate);
                let given = log_mass_from_log_rate_given_rate(count, log_rate, log_rate.exp());
                let from_log = log_mass_from_log_rate(count, log_rate);
                let label = format!("count {count}, log rate {log_rate}");
                assert_eq!(given.to_bits(), before.to_bits(), "{label}");
                assert_eq!(from_log.to_bits(), before.to_bits(), "{label}");
                series += takes_the_series(count, log_rate.exp()) as usize;
            }
        }
        // Every offset keeps |count / rate - 1| below 0.125.
        assert_eq!(series, counts.len() * offsets.len());
    }

    /// `log_mass` from a rate, which the log-rate path no longer calls, keeps
    /// the old values to the bit across valid and invalid counts and rates,
    /// both of its small-count returns and both large-count branches.
    #[test]
    fn log_mass_from_a_rate_keeps_its_old_values_bit_for_bit() {
        let mut counts = (0..=20).map(f64::from).collect::<Vec<_>>();
        counts.extend([1e6, 1e14, 2.5, -1.0, -0.0, f64::INFINITY, f64::NAN]);
        let mut rates = vec![
            0.0,
            -0.0,
            f64::from_bits(1),
            1e-310,
            f64::MIN_POSITIVE,
            1e-20,
            0.3,
            1.0,
            std::f64::consts::E,
            7.5,
            15.0,
            16.0,
            19.5,
            0.97e6,
            1e6,
            1.03e6,
            1e14,
            1e300,
            f64::MAX,
            f64::INFINITY,
            f64::NAN,
            -1.0,
        ];
        // Rates within the series' reach of every count from 16 to 20.
        rates.extend((16..=20).flat_map(|k| [0.9, 0.999, 1.0, 1.001, 1.1].map(|f| k as f64 * f)));
        let mut series = 0;
        let mut large = 0;
        for &count in &counts {
            for &rate in &rates {
                let before = log_mass_before(count, rate);
                let after = log_mass(count, rate);
                assert_eq!(
                    after.to_bits(),
                    before.to_bits(),
                    "count {count}, rate {rate}"
                );
                if count >= 16.0 && count.is_finite() && rate.is_finite() && rate > 0.0 {
                    large += 1;
                    series += takes_the_series(count, rate) as usize;
                }
            }
        }
        // Of the 294 large-count cases, 94 take the deviance series and 200
        // the logarithm of the ratio, so both branches are compared.
        assert_eq!((series, large), (94, 294));
    }

    /// 60-digit mpmath values of `count * eta - exp(eta) - ln(count!)`, rounded
    /// to the nearest double, for small counts, where the log mass now
    /// multiplies the log rate itself. Each result must lie within 20 doubles
    /// of its reference.
    #[test]
    fn small_count_log_mass_matches_a_high_precision_reference() {
        for (count, log_rate, expected) in [
            (1.0f64, -0.3f64, -1.0408182206817178f64),
            (3.0, 0.7, -1.7055121766985315),
            (7.0, 2.0, -1.9142174599960646),
            (15.0, 2.7, -2.2790031087137255),
            (12.0, -30.0, -379.987214495662),
            (5.0, 6.5, -637.4291247871439),
            (2.0, -700.0, -1400.69314718056),
        ] {
            let actual = log_mass_from_log_rate(count, log_rate);
            let apart = ulps_apart(actual, expected);
            assert!(apart <= 20, "{actual} != {expected}, {apart} ulps");
        }
    }

    /// `is_whole` stands in for `fract() == 0.0` on finite, non-negative
    /// counts, including at the edges of exact integer conversion.
    #[test]
    fn whole_counts_are_the_ones_without_a_fraction() {
        let two_52 = 4_503_599_627_370_496.0f64;
        let mut values = vec![
            -0.0,
            0.0,
            f64::from_bits(1),
            f64::MIN_POSITIVE,
            0.5,
            1.0 - f64::EPSILON / 2.0,
            1.0,
            1.0 + f64::EPSILON,
            15.0,
            15.5,
            16.0,
            1e15 + 0.5,
            two_52 - 0.5,
            two_52,
            two_52 + 1.0,
            2.0 * two_52 + 2.0,
            1.8446744073709552e19,
            1e300,
            f64::MAX,
        ];
        let mut rng = ChaCha8Rng::seed_from_u64(16);
        values.extend((0..10_000).map(|_| {
            let magnitude: f64 = rand::Rng::gen_range(&mut rng, -10.0..60.0);
            let value = 2f64.powf(magnitude) * rand::Rng::gen_range(&mut rng, 0.5..1.0);
            if rand::Rng::gen_bool(&mut rng, 0.5) {
                value.floor()
            } else {
                value
            }
        }));
        for value in values {
            assert_eq!(is_whole(value), value.fract() == 0.0, "{value:e}");
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
