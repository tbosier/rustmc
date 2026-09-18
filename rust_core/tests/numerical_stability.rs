//! Numerical-stability regressions for the shared expression kernels.
//!
//! Every expectation here is either a closed form, a central finite
//! difference, or a constant produced by an out-of-crate high-precision
//! evaluation (Python `decimal` at 400 significant digits). Nothing in this
//! file compares one in-repo evaluator against another, because the two
//! evaluators share `ElementwiseOp::derivatives` and would agree on a wrong
//! formula.

use rustmc_core::autodiff::Evaluator;
use rustmc_core::distributions::Normal;
use rustmc_core::graph::{ElementwiseOp, Graph};

fn evaluate(graph: &Graph, params: &[f64]) -> (f64, Vec<f64>) {
    let mut evaluator = Evaluator::new(graph);
    evaluator.compute(graph, params);
    (evaluator.total_logp, evaluator.grad.clone())
}

fn central_difference(graph: &Graph, params: &[f64], index: usize, step: f64) -> f64 {
    let mut plus = params.to_vec();
    let mut minus = params.to_vec();
    plus[index] += step;
    minus[index] -= step;
    (evaluate(graph, &plus).0 - evaluate(graph, &minus).0) / (2.0 * step)
}

/// `a / b` as the whole target, so `total_logp == a / b` and
/// `grad == (1/b, -a/b^2)` in exact arithmetic.
fn division_target() -> Graph {
    let mut graph = Graph::new();
    let a = graph.add_param("a");
    let b = graph.add_param("b");
    let quotient = graph.elementwise(ElementwiseOp::Div, a, Some(b));
    graph.add_logp_term(quotient);
    graph
}

// ---------------------------------------------------------------------------
// Task 1 — Div gradient
// ---------------------------------------------------------------------------

#[test]
fn division_gradient_matches_closed_form_at_ordinary_scale() {
    let graph = division_target();
    for (a, b) in [(3.0, 2.0), (-7.5, 0.25), (1.0, -4.0), (0.125, 1e3)] {
        let (value, grad) = evaluate(&graph, &[a, b]);
        assert_eq!(value, a / b, "value for {a}/{b}");
        assert_eq!(grad[0], 1.0 / b, "d/da for {a}/{b}");
        // Exactly representable at these scales, so equality is the right test.
        assert_eq!(grad[1], -(a / b) / b, "d/db for {a}/{b}");
        for index in 0..2 {
            let numeric = central_difference(&graph, &[a, b], index, 1e-6 * b.abs().max(1.0));
            assert!(
                (grad[index] - numeric).abs() / (1.0 + numeric.abs()) < 1e-6,
                "finite difference mismatch for {a}/{b} index {index}: {} vs {numeric}",
                grad[index]
            );
        }
    }
}

#[test]
fn division_gradient_survives_denominators_whose_square_overflows() {
    let graph = division_target();
    // -a/(b*b) evaluates -1e200/inf == -0.0 and erases a representable
    // derivative of -1e-200.
    for (a, b, expected) in [
        (1e200, 1e200, -1e-200),
        (1e160, 1e160, -1e-160),
        (-2e200, 4e200, 1.25e-201),
        (1e200, -1e200, -1e-200),
    ] {
        let (_, grad) = evaluate(&graph, &[a, b]);
        assert_eq!(grad[0], 1.0 / b, "d/da for {a}/{b}");
        assert!(
            grad[1] != 0.0 && grad[1].is_finite(),
            "d/db for {a}/{b} collapsed to {}",
            grad[1]
        );
        assert!(
            (grad[1] / expected - 1.0).abs() < 1e-14,
            "d/db for {a}/{b}: {} vs {expected}",
            grad[1]
        );
    }
}

#[test]
fn division_gradient_survives_denominators_whose_square_underflows() {
    let graph = division_target();
    // b*b underflows to +0 and -a/0 fabricates an infinity where the true
    // derivative is finite.
    for (a, b, expected) in [
        (1e-200, 1e-200, -1e200),
        (1e-160, 1e-160, -1e160),
        (-2e-200, 4e-200, 1.25e199),
    ] {
        let (_, grad) = evaluate(&graph, &[a, b]);
        assert_eq!(grad[0], 1.0 / b, "d/da for {a}/{b}");
        assert!(
            grad[1].is_finite(),
            "d/db for {a}/{b} blew up to {}",
            grad[1]
        );
        assert!(
            (grad[1] / expected - 1.0).abs() < 1e-14,
            "d/db for {a}/{b}: {} vs {expected}",
            grad[1]
        );
    }
}

/// Guard, not a fix: `-(a/b)/b` must reproduce every degenerate case that
/// `-a/(b*b)` already got right. Asserted on the local derivative rather than
/// on the accumulated gradient, because `(+0.0) + (-0.0) == +0.0` erases the
/// sign of a zero as soon as it enters an adjoint accumulator.
#[test]
fn division_derivative_keeps_zero_denominator_and_signed_zero_behaviour() {
    let d = |a: f64, b: f64| ElementwiseOp::Div.derivatives(a, b);

    // b == +0: 1/b overflows, -a/b^2 overflows with the sign of -a.
    assert_eq!(d(2.0, 0.0), (f64::INFINITY, f64::NEG_INFINITY));
    assert_eq!(d(-2.0, 0.0), (f64::INFINITY, f64::INFINITY));
    // b == -0: 1/b is -inf, but b^2 is still +0, so -a/b^2 is -inf.
    assert_eq!(d(2.0, -0.0), (f64::NEG_INFINITY, f64::NEG_INFINITY));
    // 0/0 stays indeterminate rather than becoming a finite number.
    assert!(d(0.0, 0.0).1.is_nan(), "0/0 gave {}", d(0.0, 0.0).1);

    // Signed zeros: the sign of -a/b^2 is the sign of -a, since b^2 > 0.
    for (a, b, negative) in [
        (0.0, 3.0, true),
        (-0.0, 3.0, false),
        (0.0, -3.0, true),
        (-0.0, -3.0, false),
    ] {
        let (_, db) = d(a, b);
        assert_eq!(db, 0.0, "d/db for {a}/{b}");
        assert_eq!(
            db.is_sign_negative(),
            negative,
            "sign of the zero derivative for {a}/{b}"
        );
    }

    // The degenerate values still reach the accumulated gradient.
    let graph = division_target();
    let (_, grad) = evaluate(&graph, &[2.0, 0.0]);
    assert_eq!(grad[0], f64::INFINITY);
    assert_eq!(grad[1], f64::NEG_INFINITY);
    let (_, grad) = evaluate(&graph, &[0.0, 0.0]);
    assert!(grad[1].is_nan(), "0/0 gave {}", grad[1]);
}

/// The bug as it reaches a user: `theta ~ Normal(log(1e200), 1)` and
/// `0 ~ Normal(1e200 / exp(theta), 1)` evaluated at the prior mean. The
/// analytic gradient used to be exactly 0 while finite differences said 1.
#[test]
fn division_by_a_huge_exponential_keeps_the_user_visible_gradient() {
    let theta0 = 1e200_f64.ln();
    let mut graph = Graph::new();
    let theta = Normal::prior(&mut graph, "theta", theta0, 1.0);
    let numerator = graph.add_constant(1e200);
    let scale = graph.exp(theta);
    let mu = graph.elementwise(ElementwiseOp::Div, numerator, Some(scale));
    let observed = graph.add_constant(0.0);
    let sigma = graph.add_constant(1.0);
    graph.normal_logp(observed, mu, sigma);

    let ratio = 1e200 / theta0.exp();
    let expected_logp = -0.5 * std::f64::consts::TAU.ln() // prior at its mean
        + (-0.5 * std::f64::consts::TAU.ln() - 0.5 * ratio * ratio);
    let expected_grad = ratio * ratio;

    let (logp, grad) = evaluate(&graph, &[theta0]);
    assert!(
        (logp - expected_logp).abs() < 1e-12,
        "logp {logp} vs {expected_logp}"
    );
    assert!(
        grad[0] != 0.0,
        "analytic gradient collapsed to zero at theta0"
    );
    assert!(
        (grad[0] / expected_grad - 1.0).abs() < 1e-12,
        "grad {} vs {expected_grad}",
        grad[0]
    );
    let numeric = central_difference(&graph, &[theta0], 0, 1e-6);
    assert!(
        (grad[0] - numeric).abs() / (1.0 + numeric.abs()) < 1e-7,
        "grad {} vs finite difference {numeric}",
        grad[0]
    );
}
