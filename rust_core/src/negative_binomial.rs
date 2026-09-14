//! Negative-binomial mass in mean/log-mean and dispersion coordinates.
//!
//! The generalized-binomial deviance identity separates the O(count) terms
//! from the Stirling remainder, avoiding cancellation between log-gammas.
//! Background: https://svn.r-project.org/R/trunk/src/nmath/dnbinom.c and dbinom.c.
//! The centered residual form and dispersion score below follow directly from
//! that identity and the asymptotic expansion of the digamma function.

use crate::autodiff::{ln_gamma, softplus};

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

// x - log(1+x), with the second-order term retained near zero.
fn log1p_deviance(x: f64) -> f64 {
    if x.abs() >= 0.125 {
        return x - x.ln_1p();
    }
    let mut power = x * x;
    let mut sum = 0.5 * power;
    for n in 3..=24 {
        power *= -x;
        let next = sum + power / n as f64;
        if next == sum {
            break;
        }
        sum = next;
    }
    sum
}

fn stirling_error(x: f64) -> f64 {
    if x < 16.0 {
        return ln_gamma(x) - (x - 0.5) * x.ln() + x - 0.5 * std::f64::consts::TAU.ln();
    }
    let inverse = 1.0 / x;
    let square = inverse * inverse;
    inverse
        * (1.0 / 12.0
            + square
                * (-1.0 / 360.0
                    + square * (1.0 / 1260.0 + square * (-1.0 / 1680.0 + square / 1188.0))))
}

// psi(a+y)-psi(a)-log((a+y)/a), without subtracting nearly equal logs.
fn digamma_remainder_difference(mut a: f64, y: f64) -> f64 {
    if y == 0.0 {
        return 0.0;
    }
    let mut correction = 0.0;
    while a < 16.0 {
        correction += log1p_deviance(1.0 / a) - log1p_deviance(1.0 / (a + y));
        a += 1.0;
    }
    let inverse = 1.0 / a;
    let log_ratio = (y / a).ln_1p();
    let mut result = 0.5 * inverse * (y / (a + y));
    for (power, coefficient) in [
        (2, 1.0 / 12.0),
        (4, -1.0 / 120.0),
        (6, 1.0 / 252.0),
        (8, -1.0 / 240.0),
        (10, 1.0 / 132.0),
        (12, -691.0 / 32760.0),
    ] {
        result += coefficient * inverse.powi(power) * -(-(power as f64) * log_ratio).exp_m1();
    }
    correction + result
}

fn centered_score(count: f64, eta: f64, alpha: f64) -> f64 {
    let log_ratio = eta - alpha.ln();
    let p = sigmoid(-log_ratio);
    let mu = eta.exp();
    if mu.is_finite() {
        p * (count - mu)
    } else {
        p * count - sigmoid(log_ratio) * alpha
    }
}

pub(crate) fn log_mass(count: f64, eta: f64, alpha: f64) -> f64 {
    if !alpha.is_finite()
        || alpha <= 0.0
        || !count.is_finite()
        || count < 0.0
        || count.fract() != 0.0
    {
        return f64::NEG_INFINITY;
    }
    let log_alpha = alpha.ln();
    let log_denom_ratio = softplus(eta - log_alpha);
    if count == 0.0 {
        return -alpha * log_denom_ratio;
    }
    let log_count = count.ln();
    let log_total_ratio = softplus(log_count - log_alpha);
    let log_expected_alpha_ratio = log_total_ratio - log_denom_ratio;
    let log_expected_count_ratio = eta - log_count + log_expected_alpha_ratio;
    let score = centered_score(count, eta, alpha);
    let deviance = |centered: f64, log_expected_ratio: f64| {
        if centered.abs() < 0.125 {
            log1p_deviance(centered)
        } else {
            log_expected_ratio.exp_m1() - log_expected_ratio
        }
    };
    -alpha * deviance(score / alpha, log_expected_alpha_ratio)
        - count * deviance(-score / count, log_expected_count_ratio)
        - 0.5 * (std::f64::consts::TAU.ln() + log_count + log_total_ratio)
        + stirling_error(alpha + count)
        - stirling_error(alpha)
        - stirling_error(count)
}

/// Scores with respect to log mean and the positive dispersion alpha.
pub(crate) fn gradients(count: f64, eta: f64, alpha: f64) -> (f64, f64) {
    // Rejected target states still run the reverse pass. Do not enter the
    // positive-shape recurrence for invalid scales or nonfinite expressions.
    if !alpha.is_finite()
        || alpha <= 0.0
        || !count.is_finite()
        || count < 0.0
        || count.fract() != 0.0
        || !eta.is_finite()
    {
        return (0.0, 0.0);
    }
    let score = centered_score(count, eta, alpha);
    let centered = score / alpha;
    let log_ratio = softplus(count.ln() - alpha.ln()) - softplus(eta - alpha.ln());
    let deviance = if centered.abs() < 0.125 {
        log1p_deviance(centered)
    } else {
        log_ratio.exp_m1() - log_ratio
    };
    (score, digamma_remainder_difference(alpha, count) - deviance)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Independent 80-digit Decimal evaluation of the log-gamma and digamma
    // formula, with recurrence to >=32 and Stirling/Bernoulli series through
    // inverse powers 13/14. Eta values are the exact input f64 values.
    const CASES: &[(f64, f64, f64, f64, f64, f64)] = &[
        (
            1e14,
            32.23619130191664,
            1e14,
            -17.383607774442968,
            -0.06313509366674494,
            2.500000000000006e-15,
        ),
        (
            1e14,
            32.23619140191664,
            1e14,
            -17.633607786599523,
            -5000000.121565577,
            1.2499998975505459e-15,
        ),
        (
            1e14,
            32.23619120191664,
            1e14,
            -17.633607773972507,
            4999999.99529539,
            1.2500000440189765e-15,
        ),
        (
            1e14,
            32.23619130191664,
            1e16,
            -17.042009349589577,
            -0.12501998745890094,
            4.950495049504949e-19,
        ),
        (
            1e14,
            32.23619140191664,
            1e16,
            -17.53705889478735,
            -9900990.824980406,
            4.901408045233952e-21,
        ),
        (
            1.0,
            0.0,
            1e14,
            -1.000000000000005,
            0.0,
            4.999999999999967e-29,
        ),
        (
            1e14,
            32.23619130191664,
            1.0,
            -33.23619130191665,
            -1.2627018733348853e-15,
            0.5772156649015279,
        ),
        (
            3.0,
            std::f64::consts::LN_2,
            0.3,
            -2.930809291455184,
            0.13043478260869565,
            2.0656821753030625,
        ),
        (0.0, 1000.0, 1.0, -1000.0, -1.0, -999.0),
        (3.0, 1000.0, 1.0, -1000.0, -1.0, -997.1666666666666),
        (3.0, -1000.0, 1.0, -3000.0, 3.0, -1.1666666666666667),
    ];

    #[test]
    fn invalid_dispersion_rejects_without_entering_recurrence() {
        use crate::{
            autodiff::{grad_logp, Evaluator},
            graph::Graph,
        };
        let mut graph = Graph::new();
        let eta = graph.add_param("eta");
        let alpha = graph.add_param("alpha");
        let obs = graph.add_obs_data(vec![1.0]);
        let predictor = graph.broadcast_observation(eta, obs);
        graph.obs_logp_negative_binomial_log(predictor, alpha, obs);
        let mut evaluator = Evaluator::new(&graph);
        for a in [-1e20, -1e9, -1.0, 0.0, f64::INFINITY, f64::NAN] {
            evaluator.compute(&graph, &[0.0, a]);
            assert_eq!(evaluator.total_logp, f64::NEG_INFINITY);
            assert_eq!(evaluator.grad, vec![0.0, 0.0]);
            assert_eq!(
                grad_logp(&graph, &[0.0, a]),
                (f64::NEG_INFINITY, vec![0.0, 0.0])
            );
        }
    }

    #[test]
    fn count_domain_and_zero_count_tiny_dispersion_are_safe() {
        for count in [-1.0, 0.5, f64::INFINITY, f64::NAN] {
            assert_eq!(log_mass(count, 0.0, 1.0), f64::NEG_INFINITY);
            assert_eq!(gradients(count, 0.0, 1.0), (0.0, 0.0));
        }
        let alpha = f64::from_bits(1);
        let (_, da) = gradients(0.0, 0.0, alpha);
        assert!((da - (1.0 + alpha.ln())).abs() < 1e-10);
    }

    #[test]
    fn density_and_scores_match_high_precision_oracle() {
        for &(count, eta, alpha, lp, de, da) in CASES {
            let actual = log_mass(count, eta, alpha);
            assert!(
                (actual - lp).abs() < 2e-8,
                "{count}, {eta}, {alpha}: {actual} != {lp}"
            );
            let (actual_e, actual_a) = gradients(count, eta, alpha);
            // Allow only the propagated half-ulp error in exp(eta), not the
            // order-one cancellation error of the old likelihood formula.
            let exp_tolerance = if eta.exp().is_finite() {
                8e-16 * (eta.exp() * sigmoid(alpha.ln() - eta))
            } else {
                0.0
            };
            assert!(
                (actual_e - de).abs() < 1e-11 + exp_tolerance,
                "eta score {actual_e} != {de}"
            );
            assert!(
                (actual_a - da).abs() < 2e-6 * da.abs(),
                "alpha score {actual_a} != {da}"
            );
        }
    }

    #[test]
    fn graph_reference_and_pointwise_paths_preserve_local_curvature() {
        use crate::{
            autodiff::{grad_logp, Evaluator},
            graph::{Graph, ObsFamily},
        };
        let mut graph = Graph::new();
        let eta = graph.add_param("eta");
        let alpha = graph.add_param("alpha");
        let obs = graph.add_obs_data(vec![1e14]);
        let predictor = graph.broadcast_observation(eta, obs);
        graph.obs_logp_negative_binomial_log(predictor, alpha, obs);
        let mut evaluator = Evaluator::new(&graph);
        let mut values = Vec::new();
        for &(count, e, a, lp, _, _) in &CASES[..3] {
            evaluator.compute(&graph, &[e, a]);
            let reference = grad_logp(&graph, &[e, a]);
            let pointwise =
                crate::observation::log_density(ObsFamily::NegativeBinomialLog, count, e, Some(a))
                    .unwrap();
            assert!((evaluator.total_logp - lp).abs() < 2e-8);
            assert_eq!(reference.0, evaluator.total_logp);
            assert_eq!(pointwise, evaluator.total_logp);
            assert_eq!(reference.1, evaluator.grad);
            values.push(evaluator.total_logp);
        }
        // At mu=y=alpha the curvature in eta is -y/2; delta eta=1e-7.
        assert!((values[1] + values[2] - 2.0 * values[0] + 0.5).abs() < 5e-8);
    }

    #[test]
    fn small_counts_match_independent_probability_recurrence() {
        for alpha in [0.001_f64, 0.3, 1.0, 5.0, 1e6, 1e14] {
            for eta in [-20.0, 0.0, 15.0, 32.0] {
                let mut expected = -alpha * softplus(eta - alpha.ln());
                for count in 0..200 {
                    let actual = log_mass(count as f64, eta, alpha);
                    assert!(
                        (actual - expected).abs() < 2e-11 * (1.0 + expected.abs()),
                        "count={count} alpha={alpha} eta={eta}: {actual} != {expected}"
                    );
                    expected += ((count as f64 / alpha).ln_1p() + alpha.ln())
                        - ((count + 1) as f64).ln()
                        - softplus(alpha.ln() - eta);
                }
            }
        }
    }

    #[test]
    fn dispersion_score_agrees_with_finite_differences_at_large_alpha() {
        let eta = 1e14_f64.ln() + 1e-7;
        let alpha = 1e14;
        let h = alpha * 1e-4;
        let numerical =
            (log_mass(1e14, eta, alpha + h) - log_mass(1e14, eta, alpha - h)) / (2.0 * h);
        let analytic = gradients(1e14, eta, alpha).1;
        assert!((numerical / analytic - 1.0).abs() < 1e-6);
    }
}
