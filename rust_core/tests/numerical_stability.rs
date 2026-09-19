//! Numerical-stability regressions for the shared expression kernels.
//!
//! Every expectation here is either a closed form, a central finite
//! difference, or a constant produced by an out-of-crate high-precision
//! evaluation (Python `decimal` at 400 significant digits). Nothing in this
//! file compares one in-repo evaluator against another, because the two
//! evaluators share `ElementwiseOp::derivatives` and would agree on a wrong
//! formula.

use rustmc_core::autodiff::Evaluator;
use rustmc_core::distributions::{
    Bernoulli, BetaDist, Exponential, Gamma, HalfNormal, LogNormal, Normal, Poisson, StudentT,
    Uniform,
};
use rustmc_core::graph::{bounded_sigmoid_adjoint, ElementwiseOp, Graph, NodeId, ParamTransform};
use rustmc_core::model::{
    compile, HyperParam, LikelihoodFamily, LikelihoodSpec, ModelSpec, MuExpr, PriorSpec, SigmaSpec,
};
use rustmc_core::sampler::{sample, BatchModelResult, SampleResult, SamplerConfig};
use std::collections::HashMap;

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

/// `-a/b^2` at ordinary scales, against the correctly rounded f64 computed
/// out of crate with Python's `decimal`. Asserting `grad[1] == -(a / b) / b`
/// here would just restate the implementation.
#[test]
fn division_gradient_matches_closed_form_at_ordinary_scale() {
    let graph = division_target();
    for (a, b, expected) in [
        (3.0, 2.0, -0.75),
        (-7.5, 0.25, 120.0),
        (1.0, -4.0, -0.0625),
        (0.125, 1e3, -1.25e-7),
        (1.0, 13.0, -0.005917159763313609),
        (-2.7, 0.3, 30.000000000000004),
        (5.0, 7.0, -0.10204081632653061),
    ] {
        let (value, grad) = evaluate(&graph, &[a, b]);
        assert_eq!(value, a / b, "value for {a}/{b}");
        assert_eq!(grad[0], 1.0 / b, "d/da for {a}/{b}");
        assert_eq!(grad[1], expected, "d/db for {a}/{b}");
        for (index, analytic) in grad.iter().enumerate() {
            let numeric = central_difference(&graph, &[a, b], index, 1e-6 * b.abs().max(1.0));
            assert!(
                (analytic - numeric).abs() / (1.0 + numeric.abs()) < 1e-6,
                "finite difference mismatch for {a}/{b} index {index}: {analytic} vs {numeric}"
            );
        }
    }
}

/// The fallback that rescues the two tails must not be applied where it is not
/// needed: dividing twice rounds twice, and a second rounding inside the
/// subnormals turns a representable derivative into zero and back.
#[test]
fn division_gradient_does_not_double_round_through_the_subnormals() {
    let graph = division_target();
    // (a, b, correctly rounded -a/b^2), computed out of crate at 400 digits.
    for (a, b, expected) in [
        (1.5e-323, 2.2, -5e-324),
        (5e-324, 1.5, -0.0),
        (5e-324, 0.75, -1e-323),
    ] {
        let (_, grad) = evaluate(&graph, &[a, b]);
        assert_eq!(grad[1], expected, "d/db for {a}/{b}");
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

// ---------------------------------------------------------------------------
// Task 2 — sigmoid
// ---------------------------------------------------------------------------

/// `(argument, sigmoid, sigmoid')`, computed with Python's `decimal` module at
/// 400 significant digits and written as the shortest decimal that round-trips
/// to the nearest f64. `sigmoid'` is `exp(-|x|) / (1 + exp(-|x|))^2`, which is
/// `s(1-s)` without the cancellation that destroys the upper tail.
///
/// `assert_close` compares to 1e-12 relative, roughly 4500 ulp. That is a
/// magnitude-and-tail check, not a certification of the last bit: `exp` is not
/// required to be correctly rounded, so demanding equality here would make the
/// suite depend on the host libm. What it does catch is a tail collapsing to
/// zero, which is the whole reason this table exists.
const SIGMOID_REFERENCE: [(f64, f64, f64); 17] = [
    (-800.0, 0.0, 0.0),
    (-745.2, 0.0, 0.0),
    (-710.0, 4.47628622567513e-309, 4.47628622567513e-309),
    (-709.0, 1.216780750623423e-308, 1.216780750623423e-308),
    (-100.0, 3.720075976020836e-44, 3.720075976020836e-44),
    (-37.0, 8.533047625744065e-17, 8.533047625744065e-17),
    (-1.0, 0.2689414213699951, 0.19661193324148185),
    (-1e-8, 0.4999999975, 0.25),
    (0.0, 0.5, 0.25),
    (1e-8, 0.5000000025, 0.25),
    (1.0, 0.7310585786300049, 0.19661193324148185),
    (37.0, 0.9999999999999999, 8.533047625744065e-17),
    (100.0, 1.0, 3.720075976020836e-44),
    (709.0, 1.0, 1.216780750623423e-308),
    (710.0, 1.0, 4.47628622567513e-309),
    (745.2, 1.0, 0.0),
    (800.0, 1.0, 0.0),
];

fn assert_close(actual: f64, expected: f64, label: &str) {
    if expected == 0.0 {
        assert_eq!(actual, 0.0, "{label}: {actual} vs {expected}");
    } else {
        assert!(
            (actual / expected - 1.0).abs() < 1e-12,
            "{label}: {actual} vs {expected}"
        );
    }
}

#[test]
fn elementwise_sigmoid_matches_high_precision_reference() {
    let mut graph = Graph::new();
    let x = graph.add_param("x");
    let s = graph.elementwise(ElementwiseOp::Sigmoid, x, None);
    graph.add_logp_term(s);
    for (raw, value, derivative) in SIGMOID_REFERENCE {
        let (logp, grad) = evaluate(&graph, &[raw]);
        assert_close(logp, value, &format!("sigmoid({raw})"));
        assert_close(grad[0], derivative, &format!("sigmoid'({raw})"));
    }
}

#[test]
fn scalar_sigmoid_op_matches_high_precision_reference() {
    let mut graph = Graph::new();
    let x = graph.add_param("x");
    let s = graph.sigmoid(x);
    graph.add_logp_term(s);
    for (raw, value, derivative) in SIGMOID_REFERENCE {
        let (logp, grad) = evaluate(&graph, &[raw]);
        assert_close(logp, value, &format!("Op::Sigmoid({raw})"));
        assert_close(grad[0], derivative, &format!("Op::Sigmoid'({raw})"));
    }
}

#[test]
fn param_transform_sigmoid_matches_high_precision_reference() {
    for (raw, value, derivative) in SIGMOID_REFERENCE {
        assert_close(
            ParamTransform::Sigmoid.apply(raw),
            value,
            &format!("ParamTransform::Sigmoid.apply({raw})"),
        );
        assert_close(
            ParamTransform::Sigmoid.derivative(raw),
            derivative,
            &format!("ParamTransform::Sigmoid.derivative({raw})"),
        );
    }
}

/// `lower + (upper - lower) * s` loses everything when `s` is near 1 and
/// `lower` is large and negative; the mirror case loses everything when `s`
/// underflows. Both constrained values are ordinary numbers near 0.5.
#[test]
fn bounded_sigmoid_transform_survives_both_tails() {
    let wide_upper = ParamTransform::BoundedSigmoid {
        lower: 0.0,
        upper: 1e308,
    };
    assert_close(
        wide_upper.apply(-710.0),
        0.447628622567513,
        "BoundedSigmoid(0, 1e308).apply(-710)",
    );
    assert_close(
        wide_upper.derivative(-710.0),
        0.447628622567513,
        "BoundedSigmoid(0, 1e308).derivative(-710)",
    );

    let wide_lower = ParamTransform::BoundedSigmoid {
        lower: -1e308,
        upper: 1.0,
    };
    assert_close(
        wide_lower.apply(710.0),
        0.552371377432487,
        "BoundedSigmoid(-1e308, 1).apply(710)",
    );
    assert_close(
        wide_lower.derivative(710.0),
        0.447628622567513,
        "BoundedSigmoid(-1e308, 1).derivative(710)",
    );

    // Ordinary ranges must not move.
    let ordinary = ParamTransform::BoundedSigmoid {
        lower: -2.0,
        upper: 3.0,
    };
    assert_eq!(ordinary.apply(0.0), 0.5);
    assert_close(ordinary.apply(1.0), 1.6552928931500244, "apply(1)");
    assert_close(ordinary.apply(-1.0), -0.6552928931500244, "apply(-1)");
    assert_close(ordinary.derivative(0.0), 1.25, "derivative(0)");
    assert_close(
        ordinary.derivative(1.5),
        0.7457322603516643,
        "derivative(1.5)",
    );
    // Never outside the interval, and strictly inside wherever f64 has the
    // resolution to say so (`-2 + 5 * sigmoid(-40)` rounds to exactly -2).
    for raw in [-800.0, -40.0, -5.0, 0.0, 5.0, 40.0, 800.0] {
        let value = ordinary.apply(raw);
        assert!((-2.0..=3.0).contains(&value), "apply({raw}) = {value}");
    }
    for raw in [-5.0, -1.0, 0.0, 1.0, 5.0] {
        let value = ordinary.apply(raw);
        assert!(value > -2.0 && value < 3.0, "apply({raw}) = {value}");
    }
}

/// Reproduced end to end: `Uniform(0, 1e308)` at raw -710 with a Normal
/// likelihood on the constrained value. The constrained value 0.447629 used to
/// round to 0, so the library evaluated a different density and its finite
/// differences agreed with it.
///
/// The gradient is the second half of the same story. While the constrained
/// value was built as the separate nodes `lower + (upper - lower) * sigmoid`,
/// reverse mode had to materialise the sigmoid node's adjoint with the `1e308`
/// factor already applied: with `sigma = 0.1` that intermediate is `-4.5e309`,
/// it overflowed, and the gradient came back `-inf`. The fused bounded-sigmoid
/// node never forms it, so both scales are now finite and correct to the 1e-9
/// asserted below — not to the last bit, since `exp` is not correctly rounded.
///
/// Expectations below are the analytic derivative of the model density,
/// computed out of crate with Python's `decimal` at 1500 significant digits;
/// `uniform_prior_gradient_matches_high_precision_finite_differences` checks
/// the same numbers against central differences of that reference.
#[test]
fn wide_uniform_prior_density_and_gradient_at_minus_710() {
    let build = |sigma: f64| {
        let mut graph = Graph::new();
        let theta = Uniform::prior(&mut graph, "theta", 0.0, 1e308);
        let observed = graph.add_constant(0.0);
        let sigma = graph.add_constant(sigma);
        graph.normal_logp(observed, theta, sigma);
        graph
    };

    // The headline case: log density is the true -718.6349, not -708.6164.
    let (logp, grad) = evaluate(&build(0.1), &[-710.0]);
    assert!(
        (logp - -718.634922627295).abs() < 1e-9,
        "logp {logp} vs -718.634922627295"
    );
    assert!(
        grad[0].is_finite(),
        "gradient must not overflow to an infinity: {}",
        grad[0]
    );
    assert!(
        (grad[0] - -19.037138374168897).abs() < 1e-9,
        "grad {} vs -19.037138374168897",
        grad[0]
    );

    // Same prior, likelihood scale that keeps every adjoint representable even
    // under the old construction — this arm never regressed and must not move.
    let (logp, grad) = evaluate(&build(1.0), &[-710.0]);
    assert!(
        (logp - -711.0191242250755).abs() < 1e-9,
        "logp {logp} vs -711.0191242250755"
    );
    assert!(
        (grad[0] - 0.7996286162583109).abs() < 1e-9,
        "grad {} vs 0.7996286162583109",
        grad[0]
    );
}

/// `(raw, span * sigmoid(raw))` for `span = 1e308`, from Python `decimal` at
/// 200 significant digits. The slope `span * s'(raw)` is the same number to
/// every digit shown here, since `1 + exp(raw)` is 1 throughout.
///
/// Applying the span to a materialised sigmoid lost this whole table below
/// -708: the sigmoid is a subnormal from there and sheds bits (at -740 the
/// result was 0.26% high, at -745 it was 75% high), and below -745.13 it is
/// zero outright.
const WIDE_SPAN_TAIL: [(f64, f64); 11] = [
    (-710.0, 0.447628622567513),
    (-720.0, 2.0322308024242932e-05),
    (-730.0, 9.226313569122114e-10),
    (-740.0, 4.188739880048049e-14),
    (-745.0, 2.822350730471937e-16),
    (-746.0, 1.0382848095158282e-16),
    (-750.0, 1.9016849634750064e-18),
    (-760.0, 8.633636377213886e-23),
    (-800.0, 3.667874584177687e-40),
    (-1000.0, 5.075958897549457e-127),
    (-1400.0, 9.721322154756662e-301),
];

/// The tail the fused transform used to erase: `Uniform(0, 1e308)` at
/// raw = -746 returned exactly 0 for a constrained value of
/// 1.0382848095158282e-16.
///
/// Folding the span into the exponent needs no logarithm of the span: the
/// sigmoid is `exp(raw)` throughout this range, `exp(raw) == exp(raw / 2)^2`,
/// and `raw / 2` is exact, so the span goes between the two halves.
#[test]
fn bounded_sigmoid_tail_survives_a_sigmoid_that_underflows() {
    let wide = ParamTransform::BoundedSigmoid {
        lower: 0.0,
        upper: 1e308,
    };
    // Each constant is the correctly rounded f64, so the 2 ulp the source
    // claims can be asserted directly rather than through `assert_close`'s
    // 1e-12, which could not tell 2 ulp from 4500 of them.
    let within_two_ulp = |actual: f64, expected: f64, label: &str| {
        assert!(
            (actual / expected - 1.0).abs() <= 2.0 * f64::EPSILON,
            "{label}: {actual} vs {expected} is more than 2 ulp"
        );
    };
    for (raw, expected) in WIDE_SPAN_TAIL {
        within_two_ulp(wide.apply(raw), expected, &format!("apply({raw})"));
        within_two_ulp(
            wide.derivative(raw),
            expected,
            &format!("derivative({raw})"),
        );
        // The mirror interval reaches the same distance from its upper end.
        let mirrored = ParamTransform::BoundedSigmoid {
            lower: -1e308,
            upper: 0.0,
        };
        within_two_ulp(
            -mirrored.apply(-raw),
            expected,
            &format!("mirrored apply({})", -raw),
        );
    }
    // The widest representable span, not just 1e308: `f64::MAX` as an upper
    // bound is the largest interval the transform is ever asked for.
    let widest = ParamTransform::BoundedSigmoid {
        lower: 0.0,
        upper: f64::MAX,
    };
    assert!((f64::MAX - 0.0).is_finite(), "premise: the span is finite");
    assert!(
        widest.apply(-746.0) > 1.8e-16,
        "widest span at -746 gave {}",
        widest.apply(-746.0)
    );
    assert!(
        widest.apply(-1454.0) > 0.0,
        "the widest span should still reach -1454"
    );
}

/// Where the rescue itself stops, pinned so it is a known edge and not a
/// surprise. `exp(raw / 2)` is normal down to `raw = -1416.79`; below that the
/// two halves are subnormals and shed bits like the sigmoid used to, and the
/// value reaches zero near `raw = -1454` for the widest representable span.
///
/// That is 670 units of raw further out than before, and past the point where
/// `f64` can represent the constrained value at all for any narrower interval.
#[test]
fn bounded_sigmoid_tail_is_exact_until_the_halved_exponential_underflows() {
    let wide = ParamTransform::BoundedSigmoid {
        lower: 0.0,
        upper: 1e308,
    };
    // Last raw whose halves are both normal, and the first that is not.
    assert!((2.0 * f64::MIN_POSITIVE.ln() + 1416.7928370645282).abs() < 1e-9);
    assert!((0.5 * -1416.0_f64).exp().is_normal());
    assert!(!(0.5 * -1418.0_f64).exp().is_normal());

    // Still full precision just inside the boundary.
    assert_close(wide.apply(-1416.0), 1.0939906871877455e-307, "apply(-1416)");
    // Degraded but not lost just outside it: the exact value is a subnormal
    // with only a handful of bits left in any case.
    let beyond = wide.apply(-1440.0);
    assert!(
        beyond > 0.0 && (beyond / 4.129964e-318 - 1.0).abs() < 1e-3,
        "apply(-1440) = {beyond}"
    );
    // Nonzero all the way to the last raw whose exact value is representable,
    // so that clamping to zero early would fail here rather than slip through
    // the assertions above.
    for raw in [-1441.0, -1445.0, -1450.0, -1454.0] {
        assert!(
            wide.apply(raw) > 0.0,
            "apply({raw}) reached zero early: {}",
            wide.apply(raw)
        );
    }
    // And zero once the exact value is no longer representable.
    assert_eq!(wide.apply(-1460.0), 0.0, "apply(-1460)");
    assert_eq!(wide.apply(-2000.0), 0.0, "apply(-2000)");
    // Never outside the interval, at any raw.
    for raw in [-2000.0, -1454.0, -746.0, -1.0, 0.0, 1.0, 746.0, 2000.0] {
        let value = wide.apply(raw);
        assert!((0.0..=1e308).contains(&value), "apply({raw}) = {value}");
    }
}

/// The gradient half of the same story, end to end: `Uniform(0, 1e308)` at
/// raw = -746 under a `sigma = 1e-17` likelihood. With the constrained value
/// rounded to 0 the likelihood penalty vanished and the gradient was the
/// Jacobian term alone, exactly `+1`, pointing the wrong way.
///
/// The density and the gradient below are the analytic expressions evaluated
/// out of crate with Python `decimal` at 200 significant digits.
#[test]
fn wide_uniform_prior_gradient_at_minus_746_under_a_tight_likelihood() {
    let mut graph = Graph::new();
    let theta = Uniform::prior(&mut graph, "theta", 0.0, 1e308);
    let observed = graph.add_constant(0.0);
    let sigma = graph.add_constant(1e-17);
    graph.normal_logp(observed, theta, sigma);

    let (logp, grad) = evaluate(&graph, &[-746.0]);
    assert!(
        (logp / -761.6767592358718 - 1.0).abs() < 1e-12,
        "logp {logp} vs -761.6767592358718"
    );
    assert!(
        grad[0] < 0.0,
        "the likelihood penalty vanished again: gradient {}",
        grad[0]
    );
    assert!(
        (grad[0] / -106.80353456713196 - 1.0).abs() < 1e-9,
        "grad {} vs -106.80353456713196",
        grad[0]
    );
}

// ---------------------------------------------------------------------------
// Task 5 — a bounded prior's density point is the draw the caller reads back
// ---------------------------------------------------------------------------

/// Bounded intervals the constrained value has to survive: two ordinary ones,
/// each tail separately, one where both endpoints are large and positive, and
/// two that are narrower than one ulp of their own endpoints — the last two are
/// where a branch-free convex combination of the endpoints fails, returning a
/// value below `lower` or overflowing to infinity.
const BOUNDED_INTERVALS: [(f64, f64); 8] = [
    (0.0, 1.0),
    (-2.0, 3.0),
    (0.0, 1e308),
    (-1e308, 1.0),
    (1.0, 1e16),
    (-1e16, 2e16),
    (1e16, 1e16 + 2.0),
    (1.7976931348623155e308, f64::MAX),
];

/// `Uniform::prior` returns the node every downstream likelihood reads as the
/// parameter's value; `ParamTransform::apply` produces the draw the caller
/// reads back out of the posterior. If those are two different formulas then
/// the posterior a user inspects is not the point the model was evaluated at,
/// and no amount of accuracy in either one repairs that.
///
/// Asserted on the bit pattern rather than a tolerance, because the claim is
/// identity and not agreement: `Uniform(-1e308, 1)` at raw 710 used to report
/// 0.552371377432487 while the graph evaluated its density at 0.0.
///
/// This is the one kind of expectation in this file that is not an independent
/// reference, and it cannot be: after the fix both sides call
/// `graph::bounded_sigmoid`, so a wrong shared helper would satisfy it. It is a
/// tripwire for a second formula being reintroduced, not a check on the
/// formula, and it earns its place only because
/// `uniform_prior_constrained_value_matches_high_precision_reference` below
/// pins the value itself against an out-of-crate reference.
#[test]
fn uniform_prior_density_point_is_the_reported_draw() {
    for (lower, upper) in BOUNDED_INTERVALS {
        let mut graph = Graph::new();
        let theta = Uniform::prior(&mut graph, "theta", lower, upper);
        let transform = ParamTransform::BoundedSigmoid { lower, upper };
        let mut evaluator = Evaluator::new(&graph);
        let mut raw = -800.0;
        while raw <= 800.0 {
            evaluator.compute(&graph, &[raw]);
            let density_point = evaluator.scalar_at(theta);
            let reported = transform.apply(raw);
            assert_eq!(
                density_point.to_bits(),
                reported.to_bits(),
                "Uniform({lower}, {upper}) at raw {raw}: density evaluated at \
                 {density_point} but the reported draw is {reported}"
            );
            raw += 0.25;
        }
    }
}

/// The value both of them agree on has to be the right one. Constants below
/// are the exact constrained value computed out of crate with Python's
/// `decimal` at 1500 significant digits, written as the shortest decimal that
/// round-trips to the nearest f64.
#[test]
fn uniform_prior_constrained_value_matches_high_precision_reference() {
    for (lower, upper, raw, expected) in [
        (0.0, 1.0, 0.0, 0.5),
        (-2.0, 3.0, 1.0, 1.6552928931500244),
        (-2.0, 3.0, -1.0, -0.6552928931500244),
        (0.0, 1e308, -710.0, 0.447628622567513),
        (-1e308, 1.0, 710.0, 0.552371377432487),
        // Narrower than one ulp of its own endpoints: the exact value rounds
        // to `lower`, and the two large endpoint products of a convex
        // combination round to 9999999999999998, below it.
        (1e16, 1e16 + 2.0, -34.5, 1e16),
        (
            1.7976931348623155e308,
            f64::MAX,
            1.625,
            1.7976931348623157e308,
        ),
    ] {
        let mut graph = Graph::new();
        let theta = Uniform::prior(&mut graph, "theta", lower, upper);
        let mut evaluator = Evaluator::new(&graph);
        evaluator.compute(&graph, &[raw]);
        assert_close(
            evaluator.scalar_at(theta),
            expected,
            &format!("Uniform({lower}, {upper}).value({raw})"),
        );
    }
}

/// An interval whose span is not representable is not a bound this transform
/// supports, and the density says so rather than the transform: `(-1e308,
/// 1e308)` has `upper - lower == inf`, and the raw-space Uniform kernel refuses
/// a non-finite width. The transform's own output there is `-inf`/`NaN`, which
/// is why no containment claim is made for it below — the model is already
/// dead at the density.
#[test]
fn a_span_that_is_not_representable_has_no_density_anywhere() {
    let mut graph = Graph::new();
    let theta = Uniform::prior(&mut graph, "theta", -1e308, 1e308);
    let mut evaluator = Evaluator::new(&graph);
    for raw in [-800.0, -1.0, 0.0, 1.0, 800.0] {
        evaluator.compute(&graph, &[raw]);
        assert_eq!(
            evaluator.total_logp,
            f64::NEG_INFINITY,
            "Uniform(-1e308, 1e308) must have no density at raw {raw}"
        );
    }
    // And the value node still reports whatever the transform reports, so the
    // two have not diverged even on a bound neither of them supports.
    for raw in [-800.0, 0.0, 800.0] {
        evaluator.compute(&graph, &[raw]);
        assert_eq!(
            evaluator.scalar_at(theta).to_bits(),
            ParamTransform::BoundedSigmoid {
                lower: -1e308,
                upper: 1e308
            }
            .apply(raw)
            .to_bits()
        );
    }
}

/// A constrained draw outside its own interval is a draw the model's support
/// forbids, and it is finite, so no downstream finiteness check would catch
/// it. Swept densely rather than checked at the tails, since the
/// endpoint-anchored form rounds towards a different endpoint on each side of
/// zero. The two sub-ulp intervals are the load-bearing rows: a convex
/// combination of the endpoints leaves both of them.
#[test]
fn uniform_prior_constrained_value_never_leaves_its_interval() {
    for (lower, upper) in BOUNDED_INTERVALS {
        let transform = ParamTransform::BoundedSigmoid { lower, upper };
        let mut raw = -800.0;
        while raw <= 800.0 {
            let value = transform.apply(raw);
            assert!(
                value >= lower && value <= upper,
                "Uniform({lower}, {upper}).apply({raw}) = {value} is outside the interval"
            );
            raw += 0.125;
        }
    }
}

/// The three-factor product reverse mode forms for a bounded parameter:
/// `adjoint * (upper - lower) * s'(raw)`. Both fixed orders lose values the
/// other keeps, so the association is load-bearing and pinned here against
/// `adjoint * span * s(raw) * (1 - s(raw))` evaluated out of crate with
/// Python's `decimal` at 1500 significant digits.
///
/// Row 2 is the bug the fused node exists to fix: applying the span to the
/// adjoint first, as the old `sigmoid`/`mul`/`add` chain did, gives `-inf`.
/// Row 1 is its mirror: applying the span to the slope first makes `span * s'`
/// 4.2e-326, which rounds to zero. Rows 3 and 4 are the subnormal span near
/// raw 0, the one band where the value still moves after `span * s'` has
/// rounded away.
#[test]
fn bounded_sigmoid_adjoint_matches_high_precision_reference() {
    for (adjoint, raw, lower, upper, expected) in [
        (1e308, -40.0, 0.0, 1e-308, 4.248354255291589e-18),
        (-44.7628622567513, -710.0, 0.0, 1e308, -20.0371383741689),
        (1e300, 0.0, 0.0, 1e-323, 2.470328229206233e-24),
        (1e300, 0.5, 0.0, 1e-323, 2.3221452168794243e-24),
        (1.0, 0.0, -2.0, 3.0, 1.25),
        (3.0, 1.5, -2.0, 3.0, 2.2371967810549926),
        (-2.5, 710.0, -1e308, 1.0, -1.1190715564187825),
        (7.0, -3.25, 0.0, 1.0, 0.2515351357772969),
        // A slope that underflowed on its own, where the span is ordinary and
        // cannot rescue it: `s'(-750)` is zero because `exp(-750)` is below the
        // smallest subnormal, so both the span-first and adjoint-first orders
        // give zero while the exact composition is an ordinary number. The
        // adjoint has to go inside the exponential.
        (1e308, -750.0, 0.0, 1.0, 1.9016849634750064e-18),
        (1e308, 750.0, 0.0, 1.0, 1.9016849634750064e-18),
        (1e308, -800.0, 0.0, 1.0, 3.667874584177687e-40),
        (1e250, -760.0, 0.0, 2.5, 2.1584090943034714e-80),
    ] {
        let actual = bounded_sigmoid_adjoint(adjoint, raw, lower, upper);
        assert!(
            actual.is_finite(),
            "adjoint({adjoint}, {raw}, {lower}, {upper}) = {actual} is not finite"
        );
        assert_close(
            actual,
            expected,
            &format!("bounded_sigmoid_adjoint({adjoint}, {raw}, {lower}, {upper})"),
        );
    }
}

/// The same identity for every other constrained family in `distributions`.
/// `BetaDist::prior` builds its value with `Op::Sigmoid` and the exp-transform
/// families with `Op::Exp`, each of which already calls exactly what its
/// `ParamTransform` calls — this pins that rather than assuming it, since the
/// bounded case shows how quietly the two can diverge.
#[test]
fn every_constrained_prior_evaluates_its_density_at_the_reported_draw() {
    type Build = fn(&mut Graph) -> NodeId;
    let families: [(&str, Build); 5] = [
        ("Beta", |g| BetaDist::prior(g, "x", 2.0, 5.0)),
        ("HalfNormal", |g| HalfNormal::prior(g, "x", 1.5)),
        ("Exponential", |g| Exponential::prior(g, "x", 0.7)),
        ("LogNormal", |g| LogNormal::prior(g, "x", 0.2, 1.1)),
        ("Gamma", |g| Gamma::prior(g, "x", 3.0, 2.0)),
    ];
    for (family, build) in families {
        let mut graph = Graph::new();
        let value = build(&mut graph);
        let transform = graph.param_transforms[0].clone();
        let mut evaluator = Evaluator::new(&graph);
        let mut raw = -750.0;
        while raw <= 750.0 {
            evaluator.compute(&graph, &[raw]);
            let density_point = evaluator.scalar_at(value);
            let reported = transform.apply(raw);
            assert_eq!(
                density_point.to_bits(),
                reported.to_bits(),
                "{family} at raw {raw}: density evaluated at {density_point} \
                 but the reported draw is {reported}"
            );
            raw += 0.25;
        }
    }
}

/// The gradient at both wide tails, against central differences of the same
/// high-precision reference the analytic constants come from — not against the
/// library's own log density, which would agree with itself on a wrong
/// constrained value. Reference central differences at h = 1e-6 are
/// -19.037138374182256 and -25.7257238825765; they converge on the analytic
/// derivatives asserted here as h shrinks.
#[test]
fn uniform_prior_gradient_matches_high_precision_finite_differences() {
    let build = |lower: f64, upper: f64, sigma: f64| {
        let mut graph = Graph::new();
        let theta = Uniform::prior(&mut graph, "theta", lower, upper);
        let observed = graph.add_constant(0.0);
        let sigma = graph.add_constant(sigma);
        graph.normal_logp(observed, theta, sigma);
        graph
    };

    for (lower, upper, sigma, raw, logp_ref, grad_ref) in [
        (
            0.0,
            1e308,
            0.1,
            -710.0,
            -718.634922627295,
            -19.037138374168897,
        ),
        (
            0.0,
            1e308,
            1.0,
            -710.0,
            -711.0191242250755,
            0.7996286162583109,
        ),
        (
            -1e308,
            1.0,
            0.1,
            710.0,
            -723.8720603705437,
            -25.725723882582397,
        ),
        (
            -1e308,
            1.0,
            1.0,
            710.0,
            -711.071495602508,
            -1.247257238825824,
        ),
    ] {
        let (logp, grad) = evaluate(&build(lower, upper, sigma), &[raw]);
        assert!(
            (logp - logp_ref).abs() < 1e-9,
            "Uniform({lower}, {upper}) sigma {sigma}: logp {logp} vs {logp_ref}"
        );
        assert!(
            grad[0].is_finite(),
            "Uniform({lower}, {upper}) sigma {sigma}: gradient {} is not finite",
            grad[0]
        );
        assert!(
            (grad[0] - grad_ref).abs() < 1e-9,
            "Uniform({lower}, {upper}) sigma {sigma}: grad {} vs {grad_ref}",
            grad[0]
        );
    }
}

// ---------------------------------------------------------------------------
// Task 3/4 — nothing in the surviving kernels regressed
// ---------------------------------------------------------------------------

const ALL_ELEMENTWISE: [ElementwiseOp; 14] = [
    ElementwiseOp::Add,
    ElementwiseOp::Sub,
    ElementwiseOp::Mul,
    ElementwiseOp::Div,
    ElementwiseOp::Pow,
    ElementwiseOp::Neg,
    ElementwiseOp::Exp,
    ElementwiseOp::Log,
    ElementwiseOp::Sigmoid,
    ElementwiseOp::Sqrt,
    ElementwiseOp::Tanh,
    ElementwiseOp::Softplus,
    ElementwiseOp::Sin,
    ElementwiseOp::Cos,
];

/// `mu = f(a, b, x)` for each operator, mirroring the end-to-end gradient
/// audit run through the Python bindings.
fn elementwise_case(op: ElementwiseOp) -> Graph {
    let n = 10;
    let xs: Vec<f64> = (0..n)
        .map(|i| 0.5 + 1.5 * i as f64 / (n - 1) as f64)
        .collect();
    let ys: Vec<f64> = (0..n).map(|i| 0.2 + i as f64 / (n - 1) as f64).collect();

    let mut graph = Graph::new();
    let a = Normal::prior(&mut graph, "a", 0.0, 1.0);
    let b = Normal::prior(&mut graph, "b", 0.0, 1.0);
    let xd = graph.add_data("x", xs);
    let obs = graph.add_named_obs_data("y", "obs", ys);

    // Scalar offsets that keep every operator inside its domain.
    let two = graph.add_constant(2.0);
    let three = graph.add_constant(3.0);
    let one_five = graph.add_constant(1.5);

    let mu = match op {
        // Binary operators: combine the two scalar parameters, then scale x.
        ElementwiseOp::Add => {
            let bx = graph.elementwise(ElementwiseOp::Mul, b, Some(xd));
            graph.elementwise(ElementwiseOp::Add, a, Some(bx))
        }
        ElementwiseOp::Sub => {
            let bx = graph.elementwise(ElementwiseOp::Mul, b, Some(xd));
            graph.elementwise(ElementwiseOp::Sub, a, Some(bx))
        }
        ElementwiseOp::Mul => {
            let ab = graph.elementwise(ElementwiseOp::Mul, a, Some(b));
            graph.elementwise(ElementwiseOp::Mul, ab, Some(xd))
        }
        ElementwiseOp::Div => {
            // Both a scalar/scalar and a scalar/vector division.
            let shifted = graph.elementwise(ElementwiseOp::Add, b, Some(three));
            let ratio = graph.elementwise(ElementwiseOp::Div, a, Some(shifted));
            let scaled = graph.elementwise(ElementwiseOp::Mul, ratio, Some(xd));
            // Denominator depends on b, so the vector Div exercises the
            // second-operand gradient and not just the numerator's.
            let bx = graph.elementwise(ElementwiseOp::Mul, b, Some(xd));
            let shifted_vec = graph.elementwise(ElementwiseOp::Add, bx, Some(three));
            let denom = graph.elementwise(ElementwiseOp::Add, shifted_vec, Some(xd));
            let vector_ratio = graph.elementwise(ElementwiseOp::Div, a, Some(denom));
            graph.elementwise(ElementwiseOp::Add, scaled, Some(vector_ratio))
        }
        ElementwiseOp::Pow => {
            let base = graph.elementwise(ElementwiseOp::Add, a, Some(two));
            let exponent = graph.elementwise(ElementwiseOp::Add, b, Some(one_five));
            let powered = graph.elementwise(ElementwiseOp::Pow, base, Some(exponent));
            graph.elementwise(ElementwiseOp::Mul, powered, Some(xd))
        }
        // Unary operators: f(a) * x + b.
        unary => {
            let inner = match unary {
                ElementwiseOp::Log | ElementwiseOp::Sqrt => {
                    graph.elementwise(ElementwiseOp::Add, a, Some(three))
                }
                _ => a,
            };
            let applied = graph.elementwise(unary, inner, None);
            let scaled = graph.elementwise(ElementwiseOp::Mul, applied, Some(xd));
            graph.elementwise(ElementwiseOp::Add, b, Some(scaled))
        }
    };
    let sigma = graph.add_constant(1.0);
    graph.normal_obs_logp(mu, sigma, obs);
    graph
}

fn assert_finite_difference_agrees(graph: &Graph, params: &[f64], label: &str) {
    let (logp, grad) = evaluate(graph, params);
    assert!(logp.is_finite(), "{label}: log density {logp}");
    for (index, analytic) in grad.iter().enumerate() {
        let numeric = central_difference(graph, params, index, 1e-6);
        let relative = (analytic - numeric).abs() / (1.0 + numeric.abs());
        assert!(
            relative < 1e-5,
            "{label}[{index}]: analytic {analytic} vs numeric {numeric} (rel {relative:.3e})"
        );
    }
}

#[test]
fn every_elementwise_operator_agrees_with_finite_differences() {
    for op in ALL_ELEMENTWISE {
        let graph = elementwise_case(op);
        for params in [[0.37, 0.61], [-0.45, 0.2], [1.1, -0.3]] {
            assert_finite_difference_agrees(&graph, &params, &format!("{op:?} at {params:?}"));
        }
    }
}

fn prior_case(prior: PriorSpec, vector: bool) -> Graph {
    let n = 10;
    let xs: Vec<f64> = (0..n)
        .map(|i| -1.0 + 2.0 * i as f64 / (n - 1) as f64)
        .collect();
    let ys: Vec<f64> = (0..n).map(|i| 0.2 + i as f64 / (n - 1) as f64).collect();
    let indices: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();

    let mu_expr = MuExpr::Add(
        Box::new(MuExpr::Param("q".to_string())),
        Box::new(if vector {
            MuExpr::Gather {
                param_name: "p".to_string(),
                data_key: "idx".to_string(),
            }
        } else {
            MuExpr::ParamTimesData {
                param_name: "p".to_string(),
                data_key: "x".to_string(),
            }
        }),
    );
    let spec = ModelSpec {
        dimensions: HashMap::new(),
        potentials: Vec::new(),
        deterministics: Vec::new(),
        priors: vec![
            PriorSpec::Normal {
                name: "q".to_string(),
                mu: HyperParam::Const(0.0),
                sigma: HyperParam::Const(1.0),
            },
            prior,
        ],
        likelihoods: vec![LikelihoodSpec {
            family: LikelihoodFamily::Normal,
            name: "obs".to_string(),
            mu_expr,
            sigma: Some(SigmaSpec::Const(1.0)),
            observed_key: "y".to_string(),
        }],
        bound_data_1d: HashMap::new(),
        bound_data_2d: HashMap::new(),
    };
    let data: HashMap<String, Vec<f64>> = HashMap::from([
        ("x".to_string(), xs),
        ("y".to_string(), ys),
        ("idx".to_string(), indices),
    ]);
    compile(&spec, &data, &HashMap::new())
        .expect("prior sweep model must compile")
        .graph
}

/// Every continuous `PriorSpec` variant `model::compile` can build. The two
/// discrete variants, `PriorSpec::Bernoulli` and `PriorSpec::Poisson`, are
/// deliberately absent: `reject_discrete_priors_for_gradient_sampling` refuses
/// them before a gradient is ever taken, so they have no gradient to check.
#[test]
fn every_prior_family_agrees_with_finite_differences() {
    let families: Vec<(&str, PriorSpec, bool)> = vec![
        (
            "normal",
            PriorSpec::Normal {
                name: "p".to_string(),
                mu: HyperParam::Const(0.5),
                sigma: HyperParam::Const(1.3),
            },
            false,
        ),
        (
            "half_normal",
            PriorSpec::HalfNormal {
                name: "p".to_string(),
                sigma: HyperParam::Const(1.3),
            },
            false,
        ),
        (
            "exponential",
            PriorSpec::Exponential {
                name: "p".to_string(),
                rate: HyperParam::Const(0.7),
            },
            false,
        ),
        (
            "log_normal",
            PriorSpec::LogNormal {
                name: "p".to_string(),
                mu: HyperParam::Const(0.1),
                sigma: HyperParam::Const(0.8),
            },
            false,
        ),
        (
            "student_t",
            PriorSpec::StudentT {
                name: "p".to_string(),
                nu: 4.0,
                mu: 0.2,
                sigma: 1.1,
            },
            false,
        ),
        (
            "uniform",
            PriorSpec::Uniform {
                name: "p".to_string(),
                lower: -2.0,
                upper: 3.0,
            },
            false,
        ),
        (
            "gamma",
            PriorSpec::Gamma {
                name: "p".to_string(),
                alpha: 2.5,
                beta: 1.7,
            },
            false,
        ),
        (
            "beta",
            PriorSpec::Beta {
                name: "p".to_string(),
                alpha: 2.0,
                beta: 3.0,
            },
            false,
        ),
        (
            "vector_normal",
            PriorSpec::VectorNormal {
                name: "p".to_string(),
                n: 2,
                mu: 0.0,
                sigma: 1.5,
            },
            true,
        ),
    ];

    for (label, prior, vector) in families {
        let graph = prior_case(prior, vector);
        let count = graph.param_count;
        for base in [[0.3, -0.45], [-1.2, 0.9], [2.0, -2.0]] {
            let params: Vec<f64> = (0..count).map(|i| base[i % 2]).collect();
            assert_finite_difference_agrees(&graph, &params, &format!("{label} at {params:?}"));
        }
    }
}

/// Every public prior constructor in `rustmc_core::distributions` still builds
/// a graph the `Evaluator` can run, and still scores the parameter it declares.
///
/// (Only the `Evaluator`: the allocating reference evaluator is `#[cfg(test)]`
/// inside the crate and an integration test cannot reach it. The two are
/// cross-checked by the crate's own unit tests.)
///
/// `Op::HalfNormalLogP`, `Op::UniformLogP`, `Op::GammaLogP` and
/// `Op::BetaLogP` — the direct, *constrained*-space scalar densities — were
/// deleted because no production path constructed them: every one of these
/// constructors reaches its family through a transform instead
/// (`LogHalfNormalLogP`, `LogGammaLogP`, `Op::BoundedSigmoid` plus the
/// vectorised raw-space kernels). This test is the standing proof of that: it
/// walks the whole public prior surface, including the hierarchical
/// `*_with_node*` forms, and fails if any of them stopped contributing a
/// density. It asserts reachability, not accuracy — the numbers are audited by
/// `every_prior_family_agrees_with_finite_differences` above.
#[test]
fn every_public_prior_constructor_still_evaluates() {
    // A `Graph` builder, the raw (unconstrained) parameter vector to evaluate it
    // at, and whether the prior is expected to put a non-zero gradient on the
    // parameter it declares. Hyperparameter constructors declare their
    // hyperparameter first, so the prior's own parameter is always the last
    // slot. The flag is false only for the two discrete latents, whose density
    // depends on a constant hyperparameter and so scores nothing back onto the
    // latent itself -- the documented behaviour that makes them
    // prior-predictive only.
    type Build = fn(&mut Graph) -> Vec<f64>;
    let constructors: Vec<(&str, Build, bool)> = vec![
        (
            "Normal::prior",
            |graph| {
                Normal::prior(graph, "p", 0.5, 1.3);
                vec![0.4]
            },
            true,
        ),
        (
            "Normal::prior_with_nodes",
            |graph| {
                let mu = graph.add_param("mu");
                let sigma = graph.add_constant(1.3);
                Normal::prior_with_nodes(graph, "p", mu, sigma);
                vec![0.2, 0.4]
            },
            true,
        ),
        (
            "HalfNormal::prior",
            |graph| {
                HalfNormal::prior(graph, "p", 1.3);
                vec![0.4]
            },
            true,
        ),
        (
            "HalfNormal::prior_with_node_sigma",
            |graph| {
                let sigma = HalfNormal::prior(graph, "s", 1.0);
                HalfNormal::prior_with_node_sigma(graph, "p", sigma);
                vec![0.1, 0.4]
            },
            true,
        ),
        (
            "StudentT::prior",
            |graph| {
                StudentT::prior(graph, "p", 4.0, 0.2, 1.1);
                vec![0.4]
            },
            true,
        ),
        (
            "Uniform::prior",
            |graph| {
                Uniform::prior(graph, "p", -2.0, 3.0);
                vec![0.4]
            },
            true,
        ),
        // Discrete latents: the density is defined only on the support, so the
        // raw value has to sit on it for the term to be finite at all.
        (
            "Bernoulli::prior",
            |graph| {
                Bernoulli::prior(graph, "p", 0.3);
                vec![1.0]
            },
            false,
        ),
        (
            "Poisson::prior",
            |graph| {
                Poisson::prior(graph, "p", 2.5);
                vec![3.0]
            },
            false,
        ),
        (
            "Exponential::prior",
            |graph| {
                Exponential::prior(graph, "p", 0.7);
                vec![0.4]
            },
            true,
        ),
        (
            "Exponential::prior_with_node_rate",
            |graph| {
                let rate = HalfNormal::prior(graph, "r", 1.0);
                Exponential::prior_with_node_rate(graph, "p", rate);
                vec![0.1, 0.4]
            },
            true,
        ),
        (
            "LogNormal::prior",
            |graph| {
                LogNormal::prior(graph, "p", 0.1, 0.8);
                vec![0.4]
            },
            true,
        ),
        (
            "LogNormal::prior_with_nodes",
            |graph| {
                let mu = graph.add_param("mu");
                let sigma = graph.add_constant(0.8);
                LogNormal::prior_with_nodes(graph, "p", mu, sigma);
                vec![0.2, 0.4]
            },
            true,
        ),
        (
            "Gamma::prior",
            |graph| {
                Gamma::prior(graph, "p", 2.5, 1.7);
                vec![0.4]
            },
            true,
        ),
        (
            "BetaDist::prior",
            |graph| {
                BetaDist::prior(graph, "p", 2.0, 5.0);
                vec![0.4]
            },
            true,
        ),
    ];

    assert_eq!(
        constructors.len(),
        14,
        "a prior constructor was added or removed without updating this sweep"
    );

    for (label, build, scores_own_param) in constructors {
        let mut graph = Graph::new();
        let params = build(&mut graph);
        assert_eq!(
            graph.param_count,
            params.len(),
            "{label}: parameter count changed"
        );
        let (logp, grad) = evaluate(&graph, &params);
        assert!(logp.is_finite(), "{label}: logp is {logp}");
        assert!(
            grad.iter().all(|value| value.is_finite()),
            "{label}: gradient is {grad:?}"
        );
        // The part that stops this being a tautology. A `Graph` with no log
        // density term at all evaluates perfectly happily -- `total_logp` is an
        // empty sum, zero, and the gradient is all zeros -- so "it evaluated"
        // proves nothing on its own. Pinning the prior's own score says the
        // density term is still there and still reaches the parameter through
        // reverse mode.
        let own = *grad.last().expect("every case declares a parameter");
        if scores_own_param {
            assert!(
                own.abs() > 1e-9,
                "{label}: the prior contributes no gradient to its own parameter ({own})"
            );
        } else {
            assert_eq!(
                own, 0.0,
                "{label}: a discrete latent must not score its own parameter"
            );
        }
        // A prior whose density were dropped would leave `logp` at exactly the
        // zero of the empty sum.
        assert_ne!(logp, 0.0, "{label}: log density is the empty sum");
    }
}

/// The four scalar arithmetic `Op` variants that survive the IR deletion —
/// `Add`, `Mul`, `Exp`, `Sigmoid` — chained into one target. This is not a
/// sweep of every `Op` variant: the density kernels (`NormalLogP`,
/// `LogGammaLogP`, the vectorised priors, ...) are covered by
/// `every_prior_family_agrees_with_finite_differences` and by the crate's own
/// unit tests.
#[test]
fn surviving_scalar_ops_agree_with_finite_differences() {
    let mut graph = Graph::new();
    let a = Normal::prior(&mut graph, "a", 0.0, 1.0);
    let b = Normal::prior(&mut graph, "b", 0.0, 1.0);
    let sum: NodeId = graph.add(a, b);
    let product = graph.mul(sum, a);
    let exponential = graph.exp(product);
    let squashed = graph.sigmoid(exponential);
    let observed = graph.add_constant(0.75);
    let sigma = graph.add_constant(0.4);
    graph.normal_logp(observed, squashed, sigma);
    for params in [[0.2, -0.3], [-0.9, 0.45], [1.3, 0.8]] {
        assert_finite_difference_agrees(&graph, &params, &format!("scalar ops at {params:?}"));
    }
}

/// `Op::Gather` is the only path that calls `ParamTransform::apply` and
/// `ParamTransform::derivative` while computing a target, so it is the only
/// place the transform rewrite can reach a gradient. The scalar `Uniform`
/// prior does not go through it — it builds sigmoid graph nodes instead — so
/// without this the prior sweep never exercises the changed code.
#[test]
fn gathered_constrained_vector_parameters_agree_with_finite_differences() {
    let indices = vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0];
    let observations = vec![0.4, -0.2, 0.9, 0.1, 0.55, -0.7];

    let transforms = [
        ParamTransform::BoundedSigmoid {
            lower: -2.0,
            upper: 3.0,
        },
        ParamTransform::BoundedSigmoid {
            lower: 0.0,
            upper: 1.0,
        },
        ParamTransform::BoundedSigmoid {
            lower: -50.0,
            upper: 10.0,
        },
        ParamTransform::Sigmoid,
        ParamTransform::Exp,
        ParamTransform::Identity,
    ];

    for transform in transforms {
        let mut graph = Graph::new();
        let start = graph.add_vector_params_with_transform("p", 3, transform.clone());
        match transform {
            ParamTransform::BoundedSigmoid { lower, upper } => {
                graph.vector_uniform_logp(start, 3, lower, upper)
            }
            ParamTransform::Sigmoid => graph.vector_beta_logp(start, 3, 2.0, 3.0),
            ParamTransform::Exp => graph.vector_half_normal_logp(start, 3, 1.3),
            ParamTransform::Identity => graph.vector_normal_logp(start, 3, 0.0, 1.5),
        };
        let index_node = graph.add_data("idx", indices.clone());
        let mu = graph.gather(start, 3, index_node);
        let obs = graph.add_named_obs_data("y", "obs", observations.clone());
        let sigma = graph.add_constant(1.0);
        graph.normal_obs_logp(mu, sigma, obs);

        for params in [
            [0.3, -0.45, 0.9],
            [-1.2, 0.9, -0.1],
            [2.0, -2.0, 0.0],
            [-4.0, 4.0, 1.7],
        ] {
            assert_finite_difference_agrees(
                &graph,
                &params,
                &format!("gathered {transform:?} at {params:?}"),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Task 6 — composing the upstream adjoint with the local derivative
// ---------------------------------------------------------------------------

/// `b ~ Normal(scale, scale)` and `0 ~ Normal((1 / b) * scale, 1)`, evaluated
/// at `b = scale`.
///
/// `1 / b` is representable and so is the composed gradient, but the local
/// derivative between them, `-1 / b^2`, is not, and it is scaled by the `scale`
/// factor one node upstream. The whole-graph gradient is the closed form
/// `scale^2 / b^3`, which at `b = scale` is `1 / scale`; the prior contributes
/// nothing, since `b` sits exactly at its mean.
fn scaled_reciprocal_target(scale: f64) -> Graph {
    let mut graph = Graph::new();
    let b = Normal::prior(&mut graph, "b", scale, scale);
    let one = graph.add_constant(1.0);
    let inverse = graph.elementwise(ElementwiseOp::Div, one, Some(b));
    let factor = graph.add_constant(scale);
    let mu = graph.elementwise(ElementwiseOp::Mul, inverse, Some(factor));
    let observed = graph.add_constant(0.0);
    let sigma = graph.add_constant(1.0);
    graph.normal_logp(observed, mu, sigma);
    graph
}

/// The reported bug: a small upstream factor. `-1/b^2` at `b = 1e-200` is
/// `-1e400`, so it overflowed to `-inf` before reverse mode could multiply it
/// by the upstream adjoint of `-1e-200` — and `inf` was what came out where the
/// composed gradient is the ordinary number `1e200`.
///
/// `1e200` is `scale^2 / b^3` at `b = scale = 1e-200`, evaluated out of crate
/// as an exact rational over the f64 values of those constants.
#[test]
fn composed_division_gradient_survives_a_small_upstream_factor() {
    let graph = scaled_reciprocal_target(1e-200);
    let (logp, grad) = evaluate(&graph, &[1e-200]);
    assert!(logp.is_finite(), "log density {logp}");
    assert!(grad[0].is_finite(), "gradient blew up to {}", grad[0]);
    assert_close(grad[0], 1e200, "d/db of (1/b) * 1e-200 at b = 1e-200");

    let numeric = central_difference(&graph, &[1e-200], 0, 1e-206);
    assert!(
        (grad[0] / numeric - 1.0).abs() < 1e-6,
        "analytic {} vs central difference {numeric}",
        grad[0]
    );
}

/// The mirror: a large upstream factor. `-1/b^2` at `b = 1e200` underflows to
/// `-0.0`, so the composed gradient came out as `+0.0` where it is `1e-200`.
///
/// This is the failure the other way round, and it is the reason the fix cannot
/// simply be "divide twice": `-(a/b)/b` is what produced the zero here.
#[test]
fn composed_division_gradient_survives_a_large_upstream_factor() {
    let graph = scaled_reciprocal_target(1e200);
    let (logp, grad) = evaluate(&graph, &[1e200]);
    assert!(logp.is_finite(), "log density {logp}");
    assert!(grad[0] != 0.0, "gradient collapsed to {}", grad[0]);
    assert_close(grad[0], 1e-200, "d/db of (1/b) * 1e200 at b = 1e200");

    let numeric = central_difference(&graph, &[1e200], 0, 1e194);
    assert!(
        (grad[0] / numeric - 1.0).abs() < 1e-5,
        "analytic {} vs central difference {numeric}",
        grad[0]
    );
}

/// No regression in the ordinary range: the composed adjoints are still
/// `upstream / b` and `-upstream * a / b^2`.
///
/// Expectations are those two closed forms evaluated out of crate as exact
/// rationals over the f64 operands, then rounded once to f64. The numerator
/// adjoint is one division and matches exactly; the denominator adjoint is
/// `upstream * (-a / (b * b))` and rounds three times, so it is checked to
/// within a few ulp — the last row misses the once-rounded value by one
/// (`-2.0000000000000002e-16` against `-2e-16`), and did so before this change
/// as well.
#[test]
fn composed_division_adjoints_are_unchanged_in_the_ordinary_range() {
    for (upstream, a, b, expected_a, expected_b) in [
        (1.0, 3.0, 2.0, 0.5, -0.75),
        (0.5, -7.5, 0.25, 2.0, 60.0),
        (-2.0, 1.0, 13.0, -0.15384615384615385, 0.011834319526627219),
        (3.25, 5.0, 7.0, 0.4642857142857143, -0.33163265306122447),
        (1e-8, 2.0, 1e4, 1e-12, -2e-16),
    ] {
        let (da, db) = ElementwiseOp::Div.adjoints(upstream, a, b);
        assert_eq!(da, expected_a, "d/da for {upstream} * d({a}/{b})");
        assert!(
            (db / expected_b - 1.0).abs() < 1e-15,
            "d/db for {upstream} * d({a}/{b}): {db} vs {expected_b}"
        );
    }
}

/// `Pow` reaches the same wall through a different door: `x^-1` is
/// representable at `x = 1e-200` but its derivative `-x^-2` is `-1e400`, and
/// the `1e-200` factor above it makes the composed gradient `1e200`.
#[test]
fn composed_power_gradient_survives_an_overflowing_local_derivative() {
    let mut graph = Graph::new();
    let x = Normal::prior(&mut graph, "x", 1e-200, 1e-200);
    let minus_one = graph.add_constant(-1.0);
    let inverse = graph.elementwise(ElementwiseOp::Pow, x, Some(minus_one));
    let factor = graph.add_constant(1e-200);
    let mu = graph.elementwise(ElementwiseOp::Mul, inverse, Some(factor));
    let observed = graph.add_constant(0.0);
    let sigma = graph.add_constant(1.0);
    graph.normal_logp(observed, mu, sigma);

    let (logp, grad) = evaluate(&graph, &[1e-200]);
    assert!(logp.is_finite(), "log density {logp}");
    assert!(grad[0].is_finite(), "gradient blew up to {}", grad[0]);
    // Same closed form as the Div case: 1e-200^2 / x^3 at x = 1e-200.
    assert_close(grad[0], 1e200, "d/dx of (x^-1) * 1e-200 at x = 1e-200");
}

/// The rescue must round exactly once. Rescaling an exponent in fixed steps
/// rounds at every step that lands in the subnormals, and two roundings erase
/// the very results the rescue exists to keep: `(1 + 2^-52) * 2^125` over
/// `(2^600)^2` is just above half the smallest subnormal and must round up to
/// it, but stepping down through `2^-512` twice drops the trailing bit first
/// and leaves an exact midpoint, which rounds to even — to zero.
///
/// The companion row is the discipline on the fix: an upstream of `2^125`
/// exactly is strictly *below* the midpoint and must still round to zero, so a
/// rescue that simply rounds everything up would not pass either.
///
/// Both expectations are exact rationals over the f64 operands, out of crate.
#[test]
fn the_rescued_division_adjoint_rounds_once_into_the_subnormals() {
    let above_midpoint = (1.0 + f64::EPSILON) * 2f64.powi(125);
    let at_midpoint = 2f64.powi(125);
    let b = 2f64.powi(600);
    assert_eq!(above_midpoint, 4.253529586511732e37);

    let (_, db) = ElementwiseOp::Div.adjoints(above_midpoint, 1.0, b);
    assert_eq!(db, -5e-324, "just above half the smallest subnormal");
    let (_, db) = ElementwiseOp::Div.adjoints(at_midpoint, 1.0, b);
    assert_eq!(db, 0.0, "exactly at the midpoint, ties to even");
    assert!(db.is_sign_negative(), "the zero keeps its sign");
}

/// A local derivative that is finite but quantized is the same loss with a
/// plausible face on it. `-(a/b)/b` at `a = 1, b = 3.7e161` is `-5e-324`, one
/// bit of a true -7.3e-324; an upstream adjoint of `1e200` then restores the
/// scale and the error together, and the composed gradient comes out 32% low
/// rather than infinite. Rescuing only on an infinity or a zero misses it.
///
/// Expectations are `-upstream * a / b^2` as an exact rational over the f64
/// operands, rounded once, computed out of crate.
#[test]
fn composed_division_gradient_survives_a_quantized_local_derivative() {
    for (b, expected) in [
        (4.5e161, -4.938271604938272e-124),
        (4e161, -6.2499999999999995e-124),
        (3.7e161, -7.304601899196495e-124),
        (3.5e161, -8.16326530612245e-124),
        (1e161, -9.999999999999999e-123),
    ] {
        let local = ElementwiseOp::Div.derivatives(1.0, b).1;
        assert!(
            local != 0.0 && !local.is_normal(),
            "premise: the local derivative for b = {b} must be subnormal, got {local}"
        );
        let (_, db) = ElementwiseOp::Div.adjoints(1e200, 1.0, b);
        assert!(
            (db / expected - 1.0).abs() < 1e-14,
            "d/db for 1e200 * d(1/{b}): {db} vs {expected}"
        );
    }
}

/// `Pow` where `a^b` has left the range too, in both directions. There is no
/// in-range power left to rewrite the derivative through, so the rescue falls
/// back to accumulating the exponent through a logarithm — the only path in
/// the module that does, and the only one accurate to 1e-13 rather than to a
/// few ulp.
///
/// Expectations are `upstream * b * a^(b-1)` as an exact rational over the f64
/// operands (Python `fractions`), rounded once.
#[test]
fn composed_power_gradient_survives_a_value_that_left_the_range() {
    // a^b == 1e-600 rounds to 0 and a^(b-1) == 1e-400 rounds to 0, while the
    // composed gradient 3e-200 is an ordinary number.
    assert_eq!(1e-200_f64.powf(2.0), 0.0, "premise: a^(b-1) underflows");
    let (da, _) = ElementwiseOp::Pow.adjoints(1e200, 1e-200, 3.0);
    assert!(
        (da / 3e-200 - 1.0).abs() < 1e-12,
        "d/da for 1e200 * d(1e-200^3): {da} vs 3e-200"
    );

    // The mirror: a^b == 1e600 and a^(b-1) == 1e400 both overflow, while the
    // composed gradient 3e200 is an ordinary number.
    assert_eq!(
        1e200_f64.powf(2.0),
        f64::INFINITY,
        "premise: a^(b-1) overflows"
    );
    let (da, _) = ElementwiseOp::Pow.adjoints(1e-200, 1e200, 3.0);
    assert!(
        (da / 3e200 - 1.0).abs() < 1e-12,
        "d/da for 1e-200 * d(1e200^3): {da} vs 3e200"
    );

    // A negative base keeps the direct answer: `powf` of one is NaN unless the
    // exponent is an integer, and a logarithm cannot tell the difference.
    let (da, db) = ElementwiseOp::Pow.adjoints(1e-200, -1e200, 3.5);
    assert!(da.is_nan() && db.is_nan(), "negative base gave {da}, {db}");
}

/// `Log`'s local derivative `1 / a` is infinite for every subnormal `a`, while
/// `upstream / a` is an ordinary number whenever the adjoint is small.
///
/// `1e-300 * ln(a)` at `a = 1e-320` has gradient `1e-300 / 1e-320`, which is
/// 1.0000111329412581e20 — the exact rational quotient of the two f64 constants,
/// rounded once, computed out of crate. It is not exactly 1e20 because 1e-320 is
/// a subnormal and carries only about 20 significant bits.
#[test]
fn composed_logarithm_gradient_survives_a_subnormal_argument() {
    let mut graph = Graph::new();
    let a = graph.add_param("a");
    let log_a = graph.elementwise(ElementwiseOp::Log, a, None);
    let factor = graph.add_constant(1e-300);
    let term = graph.elementwise(ElementwiseOp::Mul, log_a, Some(factor));
    graph.add_logp_term(term);

    assert_eq!(1.0 / 1e-320, f64::INFINITY, "1/a must overflow for this a");
    let (logp, grad) = evaluate(&graph, &[1e-320]);
    assert!(logp.is_finite(), "log density {logp}");
    assert_eq!(
        grad[0], 1.0000111329412581e20,
        "d/da of 1e-300 * ln(a) at a = 1e-320"
    );
}

/// `1 - tanh(a)^2` is not `sech^2(a)` in floating point: `tanh` rounds to 1 at
/// `|a| = 19.06` and the difference cancels to exactly zero from there on, while
/// the true derivative stays representable out to `|a| = 372`. One step before
/// the cliff, at `a = 19`, the old form is already 77% too large.
///
/// The reference is `4 e^-2|a| / (1 + e^-2|a|)^2` evaluated out of crate with
/// Python `decimal` at 120 significant digits. As elsewhere in this file the
/// comparison is to 1e-12 relative, because `exp` is not correctly rounded.
#[test]
fn tanh_derivative_does_not_cancel_once_tanh_saturates() {
    for (a, expected) in [
        (0.0, 1.0),
        (0.5, 0.7864477329659274),
        (1.0, 0.4199743416140261),
        (5.0, 0.0001815832309438067),
        (10.0, 8.244614455767397e-09),
        (19.0, 1.2556531168192118e-16),
        (19.1, 1.0280418219381054e-16),
        (20.0, 1.6993417021166355e-17),
        (25.0, 7.714999391855671e-22),
        (40.0, 7.219405551381661e-35),
        (100.0, 5.53558610694695e-87),
        (300.0, 1.0601586212017243e-260),
        // Into the subnormal band, where the factor of four has to go in before
        // the exponential rounds and not after. `4 * s'(2a)` reaches zero at
        // 372.5666, a full unit before the true derivative does.
        (354.0, 1.3230212014553631e-307),
        (360.0, 8.12892320967e-313),
        (370.0, 1.675e-321),
        (372.0, 3e-323),
        (372.5, 1e-323),
        (372.6, 1e-323),
        (373.0, 5e-324),
        (373.2, 5e-324),
        // And zero only once the exact value rounds to zero, near 373.2598.
        (373.3, 0.0),
        (380.0, 0.0),
    ] {
        for signed in [a, -a] {
            let (slope, _) = ElementwiseOp::Tanh.derivatives(signed, 0.0);
            assert_close(slope, expected, &format!("d/da tanh({signed})"));
        }
    }
    // The saturation boundary itself: `tanh(19)` is not 1, `tanh(19.0616)` is.
    assert_eq!(19.0_f64.tanh(), 0.9999999999999999);
    assert_eq!(19.061547465398498_f64.tanh(), 1.0);
}

/// The same cancellation as a user meets it: `0 ~ Normal(tanh(eta), 1)` with
/// `eta` pushed into saturation. The gradient is `-tanh(eta) sech^2(eta)`,
/// which is exactly `-sech^2(eta)` once `tanh` rounds to 1, and it used to be
/// exactly zero for every `|eta| > 19` — a flat direction where the density is
/// not flat.
#[test]
fn saturated_tanh_still_moves_the_gradient() {
    let mut graph = Graph::new();
    let eta = graph.add_param("eta");
    let squashed = graph.elementwise(ElementwiseOp::Tanh, eta, None);
    let observed = graph.add_constant(0.0);
    let sigma = graph.add_constant(1.0);
    graph.normal_logp(observed, squashed, sigma);

    // sech^2 at 120 significant digits, out of crate; tanh is exactly 1 at all
    // three of these, so the gradient is exactly its negation.
    for (eta0, sech_squared) in [
        (25.0, 7.714999391855671e-22),
        (40.0, 7.219405551381661e-35),
        (100.0, 5.53558610694695e-87),
    ] {
        let (logp, grad) = evaluate(&graph, &[eta0]);
        assert!(logp.is_finite(), "log density {logp}");
        assert!(
            grad[0] != 0.0,
            "the gradient collapsed to zero at eta = {eta0}"
        );
        assert_close(grad[0], -sech_squared, &format!("d/deta at {eta0}"));
    }
}

// ---------------------------------------------------------------------------
// Task 7 — discrete supports
// ---------------------------------------------------------------------------

/// `x ~ Bernoulli(p)` with both operands constant, so `total_logp` is the log
/// mass itself.
fn bernoulli_density(x: f64, p: f64) -> f64 {
    let mut graph = Graph::new();
    let x_node = graph.add_constant(x);
    let p_node = graph.add_constant(p);
    graph.bernoulli_logp(x_node, p_node);
    evaluate(&graph, &[]).0
}

/// `x ~ Bernoulli(p)` with `p` free, so `grad[0]` is the score.
fn bernoulli_score(x: f64, p: f64) -> f64 {
    let mut graph = Graph::new();
    let p_node = graph.add_param("p");
    let x_node = graph.add_constant(x);
    graph.bernoulli_logp(x_node, p_node);
    evaluate(&graph, &[p]).1[0]
}

/// The support of a Bernoulli is `{0, 1}`. Nothing checked it, so `x = 0.5` had
/// a finite density and at `p = 0.5` that density was constant over all of R.
///
/// Reachable through `GraphModel::log_density` on a loaded artifact, which is
/// why this is about the density and not about sampling.
#[test]
fn bernoulli_density_is_minus_infinity_off_its_support() {
    for x in [-1.0, -0.5, 0.5, 1.5, 2.0, 1e16, f64::NAN, f64::INFINITY] {
        for p in [0.1, 0.5, 0.9] {
            let value = bernoulli_density(x, p);
            assert_eq!(
                value,
                f64::NEG_INFINITY,
                "Bernoulli({p}) at x = {x} gave {value}"
            );
            assert_eq!(bernoulli_score(x, p), 0.0, "score off support at x = {x}");
        }
    }
    // The support itself stays finite.
    assert_eq!(bernoulli_density(1.0, 0.5), -std::f64::consts::LN_2);
    assert_eq!(bernoulli_density(0.0, 0.5), -std::f64::consts::LN_2);
}

/// An impossible outcome has no density, not a small one. The clamp to
/// `[1e-12, 1 - 1e-12]` scored `x = 1, p = 0` at `ln(1e-12)`, which is
/// -27.631021115928547, and `x = 0, p = 1` at `ln(1 - (1 - 1e-12))`, which is
/// -27.63104323789336 — the two differ because `1 - 1e-12` is not exact.
#[test]
fn impossible_bernoulli_outcomes_have_no_density() {
    assert_eq!(bernoulli_density(1.0, 0.0), f64::NEG_INFINITY);
    assert_eq!(bernoulli_density(0.0, 1.0), f64::NEG_INFINITY);
    // The certain outcomes at the same endpoints have log mass exactly zero.
    assert_eq!(bernoulli_density(0.0, 0.0), 0.0);
    assert_eq!(bernoulli_density(1.0, 1.0), 0.0);
    // A p outside [0, 1] is not a probability at all.
    for p in [-0.5, -1e-300, 1.0000001, 2.0, f64::NAN, f64::INFINITY] {
        for x in [0.0, 1.0] {
            assert_eq!(
                bernoulli_density(x, p),
                f64::NEG_INFINITY,
                "Bernoulli({p}) at x = {x}"
            );
            assert_eq!(bernoulli_score(x, p), 0.0, "score for p = {p}, x = {x}");
        }
    }
}

/// On the support the density is `ln p` and `ln(1 - p)` over the whole of
/// `[0, 1]`, not over `[1e-12, 1 - 1e-12]`. Expectations are those logarithms at
/// 80 significant digits out of crate, rounded once to f64; the last two rows
/// are where the clamp used to replace the answer outright.
#[test]
fn bernoulli_density_matches_closed_form_across_the_unit_interval() {
    for (p, ln_p) in [
        (0.5, -std::f64::consts::LN_2),
        (0.25, -1.3862943611198906),
        (0.7, -0.35667494393873245),
        (1e-12, -27.631021115928547),
        (1e-300, -690.7755278982137),
        (5e-324, -744.4400719213812),
    ] {
        assert_close(bernoulli_density(1.0, p), ln_p, &format!("ln({p})"));
    }
    // `ln_1p(-p)`, not `(1 - p).ln()`: `1 - p` rounds to exactly 1, discarding
    // the whole of -p, for every p at or below 2^-54 = 5.551115123125783e-17
    // (ties-to-even at the boundary itself). The premise asserted per row below
    // is the exact test, not the round figure.
    for (p, ln_1m_p) in [
        (0.5, -std::f64::consts::LN_2),
        (0.25, -0.2876820724517809),
        (0.7, -1.203972804325936),
        (1e-18, -1e-18),
        (1e-300, -1e-300),
    ] {
        assert_close(bernoulli_density(0.0, p), ln_1m_p, &format!("ln(1 - {p})"));
        assert_eq!(
            1.0 - p == 1.0,
            p <= 2f64.powi(-54),
            "premise about (1 - p) for {p}"
        );
    }
}

/// On the support the score is `1/p` and `-1/(1 - p)` exactly, over the whole
/// interval rather than a clamped band, and it agrees with central differences.
#[test]
fn bernoulli_score_matches_the_closed_form_on_its_support() {
    assert_eq!(bernoulli_score(1.0, 0.7), 1.0 / 0.7);
    assert_eq!(bernoulli_score(0.0, 0.7), -1.0 / (1.0 - 0.7));
    // Inside the old clamp band the score was pinned at 1e12.
    assert_eq!(bernoulli_score(1.0, 1e-300), 1.0 / 1e-300);
    assert_eq!(bernoulli_score(1.0, 0.0), f64::INFINITY);
    // A negative zero is a probability of zero, not a probability approached
    // from below: the limit is the same infinity, with the same sign.
    assert_eq!(bernoulli_score(1.0, -0.0), f64::INFINITY);
    assert_eq!(bernoulli_density(1.0, -0.0), f64::NEG_INFINITY);
    assert_eq!(bernoulli_density(0.0, -0.0), 0.0);

    let mut graph = Graph::new();
    let p_node = graph.add_param("p");
    let x_node = graph.add_constant(1.0);
    graph.bernoulli_logp(x_node, p_node);
    for p in [0.1, 0.5, 0.9] {
        let numeric = central_difference(&graph, &[p], 0, 1e-7);
        let analytic = evaluate(&graph, &[p]).1[0];
        assert!(
            (analytic - numeric).abs() / (1.0 + numeric.abs()) < 1e-6,
            "score at p = {p}: {analytic} vs {numeric}"
        );
    }
}

/// The Poisson mass already refused a fractional count, and carries no clamp.
/// Pinned here so it stays that way, together with the score, which did *not*
/// refuse the same inputs before and moved where the mass was flat `-inf`.
#[test]
fn poisson_density_and_score_agree_about_the_support() {
    let density = |x: f64, lam: f64| {
        let mut graph = Graph::new();
        let x_node = graph.add_constant(x);
        let lam_node = graph.add_constant(lam);
        graph.poisson_logp(x_node, lam_node);
        evaluate(&graph, &[]).0
    };
    let score = |x: f64, lam: f64| {
        let mut graph = Graph::new();
        let lam_node = graph.add_param("lam");
        let x_node = graph.add_constant(x);
        graph.poisson_logp(x_node, lam_node);
        evaluate(&graph, &[lam]).1[0]
    };

    for x in [-1.0, 0.5, 2.5, -0.0001, f64::NAN, f64::INFINITY] {
        assert_eq!(density(x, 2.0), f64::NEG_INFINITY, "Poisson(2) at x = {x}");
        assert_eq!(score(x, 2.0), 0.0, "score off support at x = {x}");
    }
    for lam in [-1.0, f64::NAN, f64::INFINITY] {
        assert_eq!(density(3.0, lam), f64::NEG_INFINITY, "Poisson({lam}) at 3");
        assert_eq!(score(3.0, lam), 0.0, "score for lam = {lam}");
    }

    // rate == 0 is the point mass at zero, and its score is -1 either way.
    assert_eq!(density(0.0, 0.0), 0.0);
    assert_eq!(density(1.0, 0.0), f64::NEG_INFINITY);
    assert_eq!(score(0.0, 0.0), -1.0);
    assert_eq!(score(0.0, 4.0), -1.0);
    // A negative zero rate is the same point mass, with the same limits: the
    // density already treats the two zeros identically and the score must too.
    assert_eq!(density(0.0, -0.0), 0.0);
    assert_eq!(density(3.0, -0.0), f64::NEG_INFINITY);
    assert_eq!(score(3.0, -0.0), f64::INFINITY);
    assert_eq!(score(3.0, 0.0), f64::INFINITY);

    // k ln(lam) - lam - ln(k!) at 80 digits, out of crate.
    assert_close(density(3.0, 2.0), -1.712317927548219, "Poisson(2) at 3");
    assert_eq!(score(3.0, 2.0), 0.5);

    // Near the mode, where a count model actually lives, `x / lam - 1` rounds
    // the quotient to something near 1 and then cancels away most of what is
    // left. Expectations are `(x - lam) / lam` as an exact rational over the
    // f64 operands, out of crate.
    let just_below_one = f64::from_bits(1.0_f64.to_bits() - 1);
    assert_eq!(just_below_one, 0.9999999999999999);
    for (x, lam, expected) in [
        (1.0, just_below_one, 1.1102230246251568e-16),
        (100.0, 100.0000000000001, -9.947598300641394e-16),
        (1e14, 100000000000001.0, -9.9999999999999e-15),
        (5.0, 5.0000000001, -2.000000165440742e-11),
    ] {
        assert_eq!(score(x, lam), expected, "score at x = {x}, lam = {lam}");
    }
}

/// Negative control for the whole task: the Bernoulli-logit *observation*
/// likelihood is a different op over observed data, and none of the above
/// touches it. This is the shape the release gate's beta-Bernoulli case fits.
#[test]
fn the_bernoulli_logit_observation_likelihood_is_unaffected() {
    let mut graph = Graph::new();
    let eta = Normal::prior(&mut graph, "eta", 0.0, 2.0);
    let obs = graph.add_obs_data(vec![1.0, 0.0, 1.0, 1.0, 0.0]);
    let linpred = graph.broadcast_observation(eta, obs);
    graph.obs_logp_bernoulli_logit(linpred, obs);

    for eta0 in [-2.0, -0.3, 0.0, 0.8, 3.0] {
        let (logp, grad) = evaluate(&graph, &[eta0]);
        assert!(logp.is_finite(), "log density at eta = {eta0} is {logp}");
        // Closed form: sum over observations of (y - sigmoid(eta)), less the
        // prior score eta / 4.
        let s = 1.0 / (1.0 + (-eta0).exp());
        let expected = (3.0 - 5.0 * s) - eta0 / 4.0;
        assert!(
            (grad[0] - expected).abs() / (1.0 + expected.abs()) < 1e-12,
            "gradient at eta = {eta0}: {} vs {expected}",
            grad[0]
        );
    }
}

// ---------------------------------------------------------------------------
// Task 5 — posterior moments
// ---------------------------------------------------------------------------

/// A `SampleResult` carrying exactly these draws, one parameter, chain-major.
///
/// Built directly rather than fitted, because the point is the arithmetic that
/// turns draws into moments and a fitted posterior cannot be asked to land on
/// chosen draws. `posterior_moments_agree_across_paths_on_a_real_fit` covers
/// the same code reached the way a user reaches it.
fn result_with_draws(chains: &[&[f64]]) -> SampleResult {
    SampleResult {
        samples: chains
            .iter()
            .map(|chain| chain.iter().map(|&v| vec![v]).collect())
            .collect(),
        unconstrained_samples: None,
        accept_rates: vec![1.0; chains.len()],
        step_sizes: vec![0.1; chains.len()],
        divergences: vec![0; chains.len()],
        transitions: chains.iter().map(|_| Vec::new()).collect(),
        param_names: vec!["theta".to_string()],
    }
}

/// Draws near the top of the representable range: both naive accumulators
/// overflow (`9e307 + ... + 2e307` is `inf`, and every `diff * diff` is `inf`),
/// so `mean()` and `std()` both used to report an infinity — every squared
/// deviation from an infinite mean is itself infinite — while `diagnostics()`
/// reported the right numbers for the very same draws.
///
/// Expectations are the exact rational mean and variance of these eight f64
/// values, evaluated out of crate with Python `fractions.Fraction` and the
/// square root taken with `decimal` at 80 significant digits:
///   mean = 5.5000000000000000167258431908505089157e307
///   sd   = 2.4494897427831782313051785197920223420e307
/// The standard deviation is the sample one, `n - 1` in the denominator, which
/// is what the summary table reports.
#[test]
fn posterior_moments_survive_draws_near_the_representable_maximum() {
    const EXPECTED_MEAN: f64 = 5.5e307;
    const EXPECTED_STD: f64 = 2.4494897427831783e307;

    let result = result_with_draws(&[&[9e307, 8e307, 7e307, 6e307], &[5e307, 4e307, 3e307, 2e307]]);

    // The accumulator this replaced, so the test cannot pass by accident.
    assert!(
        !result
            .samples
            .iter()
            .flatten()
            .map(|draw| draw[0])
            .sum::<f64>()
            .is_finite(),
        "the naive running sum must still overflow on these draws"
    );

    let mean = result.mean()[0];
    let std = result.std()[0];
    assert!(
        mean.is_finite() && std.is_finite(),
        "mean {mean}, std {std}"
    );
    assert_close(mean, EXPECTED_MEAN, "mean of draws near f64::MAX");
    assert_close(std, EXPECTED_STD, "std of draws near f64::MAX");

    // The one kind of expectation in this file that is not an independent
    // reference. It cannot be: after the fix both sides call
    // `diagnostics::scaled_moments`, so a wrong shared helper would satisfy it.
    // It is a tripwire for the two paths diverging again, and it earns its
    // place only because the constants above pin the value itself. Same
    // reasoning as `uniform_prior_density_point_is_the_reported_draw`.
    let report = result.diagnostics();
    assert_eq!(mean, report.params[0].mean, "mean() vs diagnostics()");
    assert_eq!(std, report.params[0].std, "std() vs diagnostics()");
}

/// `BatchModelResult` carries its draws flattened rather than per chain, so it
/// is a second implementation of the same traversal and needs its own check.
/// Reverting only its moments would leave every other test here passing.
#[test]
fn batch_posterior_moments_survive_draws_near_the_representable_maximum() {
    let result = BatchModelResult {
        samples: [9e307, 8e307, 7e307, 6e307, 5e307, 4e307, 3e307, 2e307]
            .iter()
            .map(|&v| vec![v])
            .collect(),
        unconstrained_samples: None,
        param_names: vec!["theta".to_string()],
        num_chains: 2,
        num_draws: 4,
        accept_rates: vec![1.0, 1.0],
        step_sizes: vec![0.1, 0.1],
        divergences: vec![0, 0],
        transitions: vec![Vec::new(), Vec::new()],
    };

    // Same eight f64 draws, so the same exact rational moments as above.
    assert_close(result.mean()[0], 5.5e307, "batch mean near f64::MAX");
    assert_close(
        result.std()[0],
        2.4494897427831783e307,
        "batch std near f64::MAX",
    );

    // And the same NaN for a draw too short to carry the parameter.
    let mut ragged = result;
    ragged.samples[2] = Vec::new();
    assert!(ragged.mean()[0].is_nan(), "batch mean for a ragged draw");
    assert!(ragged.std()[0].is_nan(), "batch std for a ragged draw");
}

/// The two paths agree on malformed input too, rather than one panicking and
/// the other reporting. `SampleResult`'s fields are public, so a caller can
/// assemble a ragged `samples` array; `diagnostics()` has always reported NaN
/// for one, and indexing it would panic.
#[test]
fn posterior_moments_report_the_same_nan_diagnostics_does_for_ragged_draws() {
    let mut result = result_with_draws(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
    result.param_names = vec!["a".to_string(), "b".to_string()];
    // Second parameter present in three draws out of six.
    result.samples[0] = vec![vec![1.0, 10.0], vec![2.0], vec![3.0, 30.0]];
    result.samples[1] = vec![vec![4.0], vec![5.0, 50.0], vec![6.0]];

    let report = result.diagnostics();
    for (index, name) in ["a", "b"].iter().enumerate() {
        assert!(
            report.params[index].mean.is_nan(),
            "diagnostics {name} mean"
        );
        assert!(result.mean()[index].is_nan(), "mean() for {name}");
        assert!(result.std()[index].is_nan(), "std() for {name}");
    }
}

/// The same agreement on a fitted posterior, reached the way a caller reaches
/// it. `HalfNormal(1e307)` is sampled on a log scale, so the chain reaches the
/// top of the range from the usual zero initialization, and the draws it
/// reports are of order 1e307: the naive sum over 2000 of them overflows.
///
/// The posterior is the prior, whose moments are closed forms —
/// `sigma * sqrt(2/pi)` and `sigma * sqrt(1 - 2/pi)` — so this checks the
/// reported numbers against something other than the other in-repo path too.
#[test]
fn posterior_moments_agree_across_paths_on_a_real_fit() {
    let sigma = 1e307;
    let mut graph = Graph::new();
    HalfNormal::prior(&mut graph, "theta", sigma);
    let result = sample(
        graph,
        SamplerConfig {
            num_chains: 2,
            num_draws: 1000,
            num_warmup: 1000,
            seed: 20260918,
            show_progress: false,
            ..SamplerConfig::default()
        },
    )
    .expect("a HalfNormal prior at 1e307 must still fit");

    let naive: f64 = result
        .samples
        .iter()
        .flatten()
        .map(|draw| draw[0])
        .sum::<f64>();
    assert!(
        !naive.is_finite(),
        "the naive running sum must overflow for this posterior, got {naive}"
    );

    let mean = result.mean()[0];
    let std = result.std()[0];
    let report = result.diagnostics();
    assert!(
        mean.is_finite() && std.is_finite(),
        "mean {mean}, std {std}"
    );
    assert_eq!(mean, report.params[0].mean, "mean() vs diagnostics()");
    assert_eq!(std, report.params[0].std, "std() vs diagnostics()");

    let expected_mean = sigma * (2.0 / std::f64::consts::PI).sqrt();
    let expected_std = sigma * (1.0 - 2.0 / std::f64::consts::PI).sqrt();
    assert!(
        (mean / expected_mean - 1.0).abs() < 0.2,
        "posterior mean {mean} vs the half-normal mean {expected_mean}"
    );
    assert!(
        (std / expected_std - 1.0).abs() < 0.25,
        "posterior std {std} vs the half-normal sd {expected_std}"
    );
}
