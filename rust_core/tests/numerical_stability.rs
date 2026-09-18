//! Numerical-stability regressions for the shared expression kernels.
//!
//! Every expectation here is either a closed form, a central finite
//! difference, or a constant produced by an out-of-crate high-precision
//! evaluation (Python `decimal` at 400 significant digits). Nothing in this
//! file compares one in-repo evaluator against another, because the two
//! evaluators share `ElementwiseOp::derivatives` and would agree on a wrong
//! formula.

use rustmc_core::autodiff::Evaluator;
use rustmc_core::distributions::{Normal, Uniform};
use rustmc_core::graph::{ElementwiseOp, Graph, NodeId, ParamTransform};
use rustmc_core::model::{
    compile, HyperParam, LikelihoodFamily, LikelihoodSpec, ModelSpec, MuExpr, PriorSpec, SigmaSpec,
};
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

#[test]
fn division_gradient_matches_closed_form_at_ordinary_scale() {
    let graph = division_target();
    for (a, b) in [(3.0, 2.0), (-7.5, 0.25), (1.0, -4.0), (0.125, 1e3)] {
        let (value, grad) = evaluate(&graph, &[a, b]);
        assert_eq!(value, a / b, "value for {a}/{b}");
        assert_eq!(grad[0], 1.0 / b, "d/da for {a}/{b}");
        // Exactly representable at these scales, so equality is the right test.
        assert_eq!(grad[1], -(a / b) / b, "d/db for {a}/{b}");
        for (index, analytic) in grad.iter().enumerate() {
            let numeric = central_difference(&graph, &[a, b], index, 1e-6 * b.abs().max(1.0));
            assert!(
                (analytic - numeric).abs() / (1.0 + numeric.abs()) < 1e-6,
                "finite difference mismatch for {a}/{b} index {index}: {analytic} vs {numeric}"
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

// ---------------------------------------------------------------------------
// Task 2 — sigmoid
// ---------------------------------------------------------------------------

/// `(argument, sigmoid, sigmoid')`, computed with Python's `decimal` module at
/// 400 significant digits and rounded to the nearest f64. `sigmoid'` is
/// `exp(-|x|) / (1 + exp(-|x|))^2`, which is `s(1-s)` without the
/// cancellation that destroys the upper tail.
const SIGMOID_REFERENCE: [(f64, f64, f64); 17] = [
    (-800.0, 0.0, 0.0),
    (-745.2, 0.0, 0.0),
    (-710.0, 4.47628622567513e-309, 4.47628622567513e-309),
    (-709.0, 1.216780750623423e-308, 1.216780750623423e-308),
    (-100.0, 3.720075976020836e-44, 3.720075976020836e-44),
    (-37.0, 8.533047625744064e-17, 8.533047625744064e-17),
    (-1.0, 2.689414213699951e-01, 1.966119332414818e-01),
    (-1e-8, 4.999999975e-01, 2.5e-01),
    (0.0, 5.0e-01, 2.5e-01),
    (1e-8, 5.000000024999999e-01, 2.5e-01),
    (1.0, 7.310585786300049e-01, 1.966119332414818e-01),
    (37.0, 9.999999999999998e-01, 8.533047625744064e-17),
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
/// `Uniform::prior` builds the constrained value as
/// `lower + (upper - lower) * sigmoid(raw)`, so reverse mode must materialise
/// the adjoint of the sigmoid node with the `1e308` factor already applied.
/// With `sigma = 0.1` that intermediate is `-4.5e309` and overflows, which no
/// reassociation inside the evaluator can avoid; the log density is still
/// exact and the gradient becomes a detectable infinity rather than a
/// plausible wrong number. With `sigma = 1.0` the intermediate fits and the
/// gradient is exact.
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

    // The headline case: log density is now the true -718.6349, not -708.6164.
    let (logp, grad) = evaluate(&build(0.1), &[-710.0]);
    assert!(
        (logp - -718.634922627295).abs() < 1e-9,
        "logp {logp} vs -718.634922627295"
    );
    assert!(
        grad[0] != 1.0 && !grad[0].is_nan(),
        "gradient is still the self-consistently wrong 1.0: {}",
        grad[0]
    );

    // Same prior, likelihood scale that keeps every adjoint representable.
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
            let denom = graph.elementwise(ElementwiseOp::Add, xd, Some(three));
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

/// Every scalar `Op` arm the samplers still construct, exercised together so
/// the IR deletion cannot quietly drop one.
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
