//! Floating-point kernels shared by the graph IR and its evaluators.
//!
//! Each of these exists because the textbook formula leaves the exponent
//! range somewhere a sampler can reach: a sigmoid at raw -710, a bounded
//! transform spanning 1e308, a quotient or power whose factors overflow while
//! the product does not. They were defined beside `Op` in `graph.rs`; the
//! public ones are still re-exported from there.

use crate::graph::ElementwiseOp;

/// Logistic sigmoid, `1 / (1 + exp(-x))`, evaluated without overflow.
///
/// The textbook form overflows `exp(-x)` for `x < -709` and returns 0 where
/// the true value is an ordinary subnormal, so branch on the sign and
/// exponentiate the negative argument.
#[inline]
pub fn stable_sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Derivative of [`stable_sigmoid`].
///
/// Mathematically `s(x) * (1 - s(x))`, but `1 - s(x)` cancels to exactly zero
/// once `s(x)` rounds to 1 (around `x = 37`), discarding a value that stays
/// representable out to `x = 745`. `exp(-|x|) / (1 + exp(-|x|))^2` is the same
/// quantity, is symmetric in `x` by construction, and needs one `exp`. Near
/// `x = 0` the old form was marginally more accurate, because `1 - s` is exact
/// there by Sterbenz; this one stays within about two ulp everywhere instead.
#[inline]
pub fn stable_sigmoid_derivative(x: f64) -> f64 {
    let e = (-x.abs()).exp();
    let d = 1.0 + e;
    e / (d * d)
}

/// `lower + (upper - lower) * sigmoid(raw)`, anchored to whichever endpoint the
/// value is nearest.
///
/// This is the single definition of the bounded transform. Both the
/// constrained draw reported back to a caller ([`ParamTransform::apply`]) and
/// the value the graph evaluates its density at ([`Op::BoundedSigmoid`]) call
/// it, so they cannot drift into two formulas that disagree about where the
/// model was evaluated. That drift is what this function exists to prevent —
/// the formula itself is unchanged.
///
/// `lower + span * s` cancels catastrophically once `s` is near 1 and `lower`
/// is large and negative: with lower = -1e308 and upper = 1, raw = 710 gives
/// -1e308 + 1e308 == 0 instead of 0.552371377432487. Both branches also round
/// towards the endpoint they subtract from, so for any interval whose span is
/// representable the result can never leave `[lower, upper]`. That guarantee is
/// why the branch is here rather than the branch-free convex combination
/// `lower * s(-raw) + upper * s(raw)`: the latter is the same quantity in exact
/// arithmetic, but its two separately rounded products can land outside a
/// narrow interval far from zero — `(1e16, 1e16 + 2)` at raw -34.5 gives
/// 9999999999999998, below `lower` — and can overflow to infinity when both
/// endpoints are near `f64::MAX`.
///
/// The distance from the endpoint is [`span_times_sigmoid`], which folds the
/// span into the exponent rather than applying it to a materialised sigmoid.
/// Two limits remain:
///
/// * The branches can disagree by one ulp at raw == 0, so the mapping is not
///   exactly monotone there (lower = 1, upper = 1e16 steps down by one ulp
///   across zero); that is a rounding artefact of the reported value, and
///   [`bounded_sigmoid_derivative`] does not model it.
/// * An interval whose span is not representable — `(-1e308, 1e308)` has
///   `upper - lower == inf` — yields `-inf` or `NaN` here. Such an interval is
///   not a usable `Uniform` prior in any case: both `uniform_bounds_valid` and
///   `model`'s `validate_positive_finite("uniform width", ...)` require a
///   finite width, so its density is `-inf` everywhere regardless of the
///   transform.
#[inline]
pub fn bounded_sigmoid(raw: f64, lower: f64, upper: f64) -> f64 {
    let span = upper - lower;
    if raw >= 0.0 {
        upper - span_times_sigmoid(span, -raw)
    } else {
        lower + span_times_sigmoid(span, raw)
    }
}

/// `span * sigmoid(raw)` for `raw <= 0`, over the whole range where the product
/// is representable rather than only where the sigmoid is.
///
/// Applying the span to a materialised sigmoid loses the tail twice over. Below
/// `raw = -708.4` the sigmoid is a subnormal and sheds bits — at `raw = -740`,
/// `span = 1e308` it is already 0.26% high, and at -745 it is 75% high — and
/// below `raw = -745.13` it is zero outright, so `Uniform(0, 1e308)` at
/// raw = -746 returned 0 for a value of 1.0382848095158282e-16.
///
/// The way out is that the sigmoid *is* `exp(raw)` in that tail: once
/// `exp(raw)` is at or below 2^-53, `1 + exp(raw)` rounds to exactly 1 (at
/// exactly 2^-53 by ties-to-even; `1 + 3*2^-54` does not, which is why the
/// threshold is 2^-53 and not 2^-52). And
/// `exp(raw) == exp(raw / 2)^2` with `raw / 2` exact, so the span goes between
/// the two halves: for `span = 1e308, raw = -746` that is
/// `1e308 * 1.0e-162 * 1.0e-162`. No logarithm of the span is taken, so no
/// precision is lost to one.
///
/// The intermediate `span * exp(raw / 2)` is `sqrt(span * result)`, so it
/// cannot underflow while the result is representable, and cannot overflow
/// either: this branch is only taken for `raw < -708`, which bounds
/// `exp(raw / 2)` below 1.1e-154 against a span of at most 1.8e308. It can be a
/// subnormal, and lose bits, where the span and the result are both near the
/// bottom of the range — `span = 1e-300, raw = -710` returns zero, which is
/// also the exact answer.
///
/// Within **2 ulp** — a measured worst case of 4.44e-16 relative, at
/// `raw = -715.23` for `span = 1e308` — from the gate down to `raw = -1416`,
/// where `exp(raw / 2)` becomes subnormal itself and the halves start shedding
/// bits; the value reaches zero around `raw = -1454` for the widest
/// representable span. Both boundaries are pinned by
/// `bounded_sigmoid_tail_is_exact_until_the_halved_exponential_underflows`, and
/// the 2 ulp by `bounded_sigmoid_tail_survives_a_sigmoid_that_underflows`.
///
/// Two ulp, not zero: `exp(raw / 2)` rounds and then the square rounds again.
/// Against a direct product that loses the value outright below -745.13 and
/// three quarters of it at -745 that is a clear win, but it is not free, and
/// for a span of 1 it can cost an ulp the direct form would have kept —
/// `raw = -708.4` gives 2.2171190816642647e-308 for a correctly rounded
/// 2.217119081664265e-308. It also *gains* one elsewhere at narrow spans
/// (`span = 0.5, raw = -720`), so no gate on the span sorts the two cleanly and
/// none is attempted.
///
/// The direct product is kept wherever the sigmoid is a normal number, so
/// ordinary intervals and ordinary raw values are bit for bit unchanged, and an
/// unrepresentable span keeps returning the infinity or NaN it did before.
#[inline]
fn span_times_sigmoid(span: f64, raw: f64) -> f64 {
    let s = stable_sigmoid(raw);
    if s.is_normal() || !span.is_finite() {
        return span * s;
    }
    let half = (0.5 * raw).exp();
    (span * half) * half
}

/// `span * s'(raw)`, the same rescue as [`span_times_sigmoid`] applied to the
/// slope.
///
/// `s'(raw)` is `exp(-|raw|) / (1 + exp(-|raw|))^2`, and once `exp(-|raw|)` is
/// subnormal the squared denominator rounds to exactly 1, so the slope is
/// `exp(-|raw|)` there and splits into two halves the same way. Without this
/// the derivative of a wide interval went to zero at `|raw| = 745` alongside
/// the value.
#[inline]
fn span_times_sigmoid_slope(span: f64, raw: f64) -> f64 {
    let slope = stable_sigmoid_derivative(raw);
    if slope.is_normal() || !span.is_finite() {
        return span * slope;
    }
    let half = (-0.5 * raw.abs()).exp();
    (span * half) * half
}

/// d/draw of [`bounded_sigmoid`], i.e. `(upper - lower) * s'(raw)`.
///
/// `span * s'` cannot overflow: `s'` never exceeds 0.25, so the product is at
/// most a quarter of a representable span. Distributing it over the endpoints
/// as `upper * s' - lower * s'` would not be safer — for `(1e16, 1e16 + 2)` at
/// raw 0.176 the two products round to the same f64 and cancel to exactly 0,
/// discarding a derivative of 0.4961479024771348.
#[inline]
pub fn bounded_sigmoid_derivative(raw: f64, lower: f64, upper: f64) -> f64 {
    span_times_sigmoid_slope(upper - lower, raw)
}

/// `adjoint * (upper - lower) * s'(raw)`, associated so that the product stays
/// inside the exponent range wherever the exact value is representable.
///
/// Reverse mode has three factors to multiply and only two orders worth
/// considering, and each fails where the other succeeds:
///
/// * Applying the span to the adjoint first is what the old three-node
///   `sigmoid`/`mul`/`add` chain did, and it overflows: `Uniform(0, 1e308)` at
///   raw -710 under a `sigma = 0.1` likelihood has an adjoint of -44.76, and
///   `-44.76 * 1e308` is `-inf` where the true gradient is -19.04.
/// * Applying the span to the slope first — [`bounded_sigmoid_derivative`] —
///   fixes that, but underflows in the mirror case: `(0, 1e-308)` at raw -40
///   has `span * s' == 4.2e-326`, which rounds to zero and erases a gradient of
///   4.248354255291588e-18 under an adjoint of 1e308.
///
/// So take the second order, and fall back to the first only when it lost the
/// value outright. The fallback cannot itself overflow: `span * s'` only
/// underflows when the span is tiny, and multiplying the adjoint by a tiny span
/// moves it towards zero.
///
/// The band where the fallback fires is narrow. `s / s'` is `1 + e^-|raw|`,
/// which never exceeds 2, so `span * s'` and the constrained value's distance
/// from its nearer endpoint underflow within a factor of two of each other:
/// almost everywhere `span * s'` rounds to zero, the value is pinned at an
/// endpoint and a zero gradient is the honest derivative of the rounded map.
/// A subnormal span near raw 0 is the exception — `(0, 1e-323)` has
/// `span * s' == 2.5e-324`, which rounds to zero while the value still moves —
/// and that is the case this order rescues.
///
/// The wide-span tail is no longer one of these cases:
/// [`span_times_sigmoid_slope`] keeps `span * s'` alive past `|raw| = 745`, so
/// that fallback now fires only for a genuinely tiny span.
///
/// There is a third case, which neither order reaches: a slope that has
/// underflowed to zero *on its own*, where the span cannot rescue it because
/// the span is ordinary. `(0, 1)` at raw -750 has `s'(-750) == 0` — `exp(-750)`
/// is below the smallest subnormal — so both `span * s'` and `(adjoint * span)
/// * s'` are zero, while an adjoint of `1e308` makes the exact composition
/// 1.9016849634750064e-18. The adjoint has to go inside the exponential for
/// that one, through the same halving [`span_times_sigmoid`] uses.
///
/// Between the three, this returns a representable composition wherever the
/// exact one is, for every finite span and finite raw.
#[inline]
pub fn bounded_sigmoid_adjoint(adjoint: f64, raw: f64, lower: f64, upper: f64) -> f64 {
    let slope = stable_sigmoid_derivative(raw);
    let span = upper - lower;
    let scaled = span_times_sigmoid_slope(span, raw);
    if scaled != 0.0 || span == 0.0 {
        return adjoint * scaled;
    }
    if slope != 0.0 {
        // A tiny span against a slope that is still representable.
        return (adjoint * span) * slope;
    }
    if !raw.is_finite() || !span.is_finite() || adjoint == 0.0 || !adjoint.is_finite() {
        return adjoint * scaled;
    }
    // The slope itself underflowed: `s'(raw)` is `exp(-|raw|)` here, and
    // `exp(-|raw|) == exp(-|raw| / 2)^2`, so the adjoint goes between the
    // halves and the span is applied last.
    let half = (-0.5 * raw.abs()).exp();
    ((adjoint * half) * half) * span
}

/// d/db of `a / b`, i.e. `-a / b^2`, over the whole representable range.
///
/// `-a / (b * b)` is the accurate form and is used wherever the squared
/// denominator is a normal number: `b * b` is then either exact or off by half
/// an ulp, so the result is within one ulp of the true quotient.
///
/// It fails outside that band, and not gracefully. `b * b` overflows to
/// infinity for `|b| > 1.34e154`, so `-a / inf` returns `-0.0` and erases a
/// derivative that is often perfectly representable (`a = b = 1e200` has
/// derivative `-1e-200`). Going the other way it decays into the subnormals
/// below `|b| = 1.49e-154` and reaches zero below `|b| = 1.58e-162`, so
/// `-a / 0.0` fabricates an infinity where the true derivative is finite
/// (`a = b = 1e-200` has derivative `-1e200`).
///
/// In that band divide twice instead. Two divisions never leave the exponent
/// range, at the cost of a second rounding — which is why this is a fallback
/// and not the only path: double rounding through the subnormals can turn a
/// representable `-5e-324` into `-0.0` (`a = 1.5e-323`, `b = 2.2`), exactly
/// the failure the fallback exists to prevent. Restricting it to the range
/// where `-a / (b * b)` is already broken keeps both forms on the inputs they
/// handle well.
#[inline]
pub(crate) fn div_denominator_derivative(a: f64, b: f64) -> f64 {
    let square = b * b;
    if square.is_normal() {
        -a / square
    } else {
        // Covers b == 0 (the square is +0 either way, so both forms give the
        // same signed infinity or NaN), the two overflow/underflow tails, and
        // the subnormal square band in between.
        -(a / b) / b
    }
}

// ---------------------------------------------------------------------------
// Composing an upstream adjoint with a local derivative
// ---------------------------------------------------------------------------
//
// Reverse mode never wants a local derivative on its own; it wants the local
// derivative times the adjoint flowing in from above. Forming the local factor
// first discards the composition whenever that factor alone leaves the exponent
// range while the product does not — and a local-derivative API has no way to
// express that, because it has nothing to scale against. The helpers below take
// the upstream adjoint as an argument so they can associate the three factors
// in an order that survives.

/// `2^exponent`, for an exponent inside the normal range.
#[inline]
fn two_pow(exponent: i32) -> f64 {
    debug_assert!((-1022..=1023).contains(&exponent));
    f64::from_bits(((exponent + 1023) as u64) << 52)
}

/// Split a finite, nonzero `x` into `(mantissa, exponent)` with the mantissa's
/// *magnitude* in `[0.5, 1)`. Exact: `mantissa * 2^exponent == x`.
///
/// The sign travels with the mantissa — `split_exponent(-8)` is `(-0.5, 4)` —
/// so a product of mantissas carries the sign of the product.
fn split_exponent(x: f64) -> (f64, i32) {
    debug_assert!(x.is_finite() && x != 0.0);
    let bits = x.to_bits();
    let raw = ((bits >> 52) & 0x7ff) as i32;
    if raw == 0 {
        // Subnormal. Scaling by 2^64 is exact and lands every nonzero
        // subnormal in the normals, where the mantissa can be read off.
        let (mantissa, exponent) = split_exponent(x * two_pow(64));
        return (mantissa, exponent - 64);
    }
    let mantissa = f64::from_bits((bits & !(0x7ff_u64 << 52)) | (1022_u64 << 52));
    (mantissa, raw - 1022)
}

/// `mantissa * 2^exponent`, rounded exactly once.
///
/// Rescaling in fixed steps is the obvious way to reach an exponent outside
/// what `two_pow` spans, and it is wrong: each step that lands in the
/// subnormals rounds, and two roundings can erase a representable result. A
/// mantissa of `1 + 2^-52` at exponent -1075 is just above half the smallest
/// subnormal and must round up to it, but stepping down through 2^-512 twice
/// drops the trailing bit first and leaves an exact midpoint, which rounds to
/// even — to zero. That is the failure the whole scaled path exists to prevent.
///
/// So normalise the mantissa, decide the outcome from the total exponent alone,
/// and let exactly one multiplication round. Every multiplication before the
/// last is by a power of two onto a normal number, which is exact.
fn apply_exponent(mantissa: f64, exponent: i32) -> f64 {
    if mantissa == 0.0 || !mantissa.is_finite() {
        return mantissa;
    }
    let (normalized, offset) = split_exponent(mantissa);
    let total = exponent.saturating_add(offset);
    let sign = if normalized < 0.0 { -1.0 } else { 1.0 };

    // `normalized` has magnitude in [0.5, 1), so the result lies in
    // [2^(total-1), 2^total).
    if total > 1024 {
        return sign * f64::INFINITY;
    }
    if total < -1074 {
        // Strictly below 2^-1075, half the smallest subnormal.
        return sign * 0.0;
    }
    if total >= -1021 {
        // Normal, or the very top of the range; `two_pow` spans at most
        // [-1022, 1023], so this may need two exact steps.
        let first = total.clamp(-1022, 1023);
        return (normalized * two_pow(first)) * two_pow(total - first);
    }
    // Subnormal result. Reach 2^-1021 exactly, then round once on the way down.
    (normalized * two_pow(-1021)) * two_pow(total + 1021)
}

/// The product of `numerators` divided by the product of `denominators`, with
/// the exponents accumulated separately so that no intermediate leaves the
/// exponent range while the result is representable.
///
/// Every operand contributes a mantissa whose *magnitude* is in `[0.5, 1)`,
/// carrying its own sign; with at most three numerators and two denominators
/// the running magnitude stays inside `[0.125, 4]`, which is normal, and the
/// exponent is reapplied once at the end. Both bounds are attained — three
/// mantissas of exactly 0.5 give exactly 0.125 — so the interval is closed.
///
/// The cost is one rounding per operand rather than one per product, which is
/// why this is a fallback rather than the only path. It is not, however, true
/// that the direct form is more accurate wherever it works at all: see
/// [`div_denominator_adjoint`], where a direct form that merely went subnormal
/// is 32% out and this one is exact.
///
/// Returns `None` when any operand is zero or not finite. The scaled form has
/// nothing to say there that the direct product has not already said, and it
/// would turn a signed infinity or a NaN into a wrong finite number.
fn scaled_ratio(numerators: &[f64], denominators: &[f64]) -> Option<f64> {
    let mut mantissa = 1.0_f64;
    let mut exponent = 0_i32;
    for &x in numerators {
        if x == 0.0 || !x.is_finite() {
            return None;
        }
        let (m, e) = split_exponent(x);
        mantissa *= m;
        exponent += e;
    }
    for &x in denominators {
        if x == 0.0 || !x.is_finite() {
            return None;
        }
        let (m, e) = split_exponent(x);
        mantissa /= m;
        exponent -= e;
    }
    Some(apply_exponent(mantissa, exponent))
}

/// Whether `direct` has left the exponent range although the exact product of
/// `factors` has not: an infinity, a NaN, or a zero produced from factors that
/// were every one of them finite and nonzero.
///
/// The `factors` list is what makes a legitimate zero legitimate. `Pow`'s
/// derivative with respect to its base is exactly zero when the exponent is
/// zero, and with respect to its exponent when the base is one; passing the
/// exponent and `ln(base)` in this list keeps those from being "rescued" into
/// something else.
#[inline]
fn needs_rescue(direct: f64, factors: &[f64]) -> bool {
    (!direct.is_finite() || direct == 0.0) && factors.iter().all(|f| f.is_finite() && *f != 0.0)
}

/// `upstream * (-a / b^2)` over the whole representable range.
///
/// [`div_denominator_derivative`] already keeps the *local* derivative finite
/// wherever it can be, but that is not enough one node upstream: `1 / b` at
/// `b = 1e-200` has local derivative `-1 / b^2 == -1e400`, which no ordering of
/// a two-argument function can represent, while an upstream adjoint of
/// `-1e-200` makes the composed gradient exactly `1e200`. Multiplying
/// afterwards has already lost it.
///
/// A local derivative that has merely gone *subnormal* is the same loss one
/// step earlier and is rescued as well. `1e200 / b` at `b = 3.7e161` has
/// `-(a/b)/b == -5e-324`, one bit of a true -7.3e-324, and the upstream adjoint
/// then restores the scale and the error together: the composed gradient comes
/// out 32% low, finite and plausible. The rescue is taken only when the
/// composed result is itself a normal number, so the three subnormal results
/// `div_denominator_derivative` already rounds correctly are left exactly as
/// they are.
#[inline]
pub(crate) fn div_denominator_adjoint(upstream: f64, a: f64, b: f64) -> f64 {
    let local = div_denominator_derivative(a, b);
    let direct = upstream * local;
    let quantized = local != 0.0 && !local.is_normal() && direct.is_normal();
    if !quantized && !needs_rescue(direct, &[upstream, a, b]) {
        return direct;
    }
    scaled_ratio(&[upstream, a], &[b, b]).map_or(direct, |value| -value)
}

/// `d/da tanh(a)`, as `4 s'(2a)`.
///
/// Not `1 - tanh(a)^2`: that cancels to exactly zero the moment `tanh` rounds to
/// 1, at `|a| = 19.06`, and is already 77% high one step before it (`x = 19`
/// gives 2.220446e-16 for a true 1.2556531168192118e-16). The true derivative
/// stays representable out to `|a| = 372`. `sech^2(a) == 4 s'(2a)` is the same
/// quantity written through [`stable_sigmoid_derivative`], which was made
/// cancellation-free for exactly this reason, and `2.0 * a` is exact.
///
/// The value and the slope saturate at different points here — `tanh` rounds to
/// exactly 1 at `|a| = 19.061547465398498` while the slope stays representable
/// to `|a| = 373.25975673153056` — so unlike `exp` or `sigmoid` this one loses
/// a derivative the forward pass had every right to. That is what makes it a
/// bug rather than a rounding boundary. (`tanh(19)` is 0.9999999999999999, not
/// 1: the boundary is 19.0615, not 19.)
///
/// The last twenty units are subnormal, and there the factor of four has to go
/// in *before* the exponential rounds rather than after. `4 * s'(2a)` at
/// `a = 372.5` gives 2e-323 for a true 1e-323 and reaches zero at 372.5666,
/// a full unit early, because it scales a subnormal that has already lost its
/// bits. `(2 e^-|a|)^2` is the same quantity with the four folded in, and is
/// within 3e-15 across the whole band. It is used only where the slope has gone
/// subnormal, because it is only there that `(1 + e^-2|a|)^2` rounds to 1.
///
/// One consequence is deliberate and worth naming: from `|a| = 19.06` the
/// computed `tanh` is flat while this slope is not, so a target that amplifies
/// the difference enough will see the gradient disagree with its own density.
/// `1e22 * (tanh(x) - 1)` at `x = 25` has a computed density of exactly 0 and a
/// gradient of 7.71, and one leapfrog step at `epsilon = 0.1` then carries an
/// energy error of 0.275. The alternative is a flat direction where the density
/// is not flat, which a sampler cannot report; this way it diverges and says
/// so. [`stable_sigmoid_derivative`] made the same choice for the same reason.
#[inline]
pub(crate) fn tanh_slope(a: f64) -> f64 {
    let slope = stable_sigmoid_derivative(2.0 * a);
    if slope.is_normal() {
        return 4.0 * slope;
    }
    let half = (-a.abs()).exp();
    (2.0 * half) * (2.0 * half)
}

/// `upstream * (b a^(b-1), a^b ln a)`.
///
/// `a^(b-1)` can leave the range on its own while `a^b` — the value the forward
/// pass already computed and the density already used — does not: `a = 1e-200`
/// with `b = -1` has `a^b = 1e200` and `a^(b-1) = 1e400`. Rewriting the first
/// derivative as `b a^b / a` keeps the whole composition inside the range.
///
/// `a^b` itself can leave the range too — `a = 1e-200, b = 3` has `a^b == 0`
/// and `a^(b-1) == 0` while an adjoint of `1e200` makes the gradient
/// `3e-200` — and then there is no in-range power left to rewrite through.
/// That case goes through [`split_power_magnitude`], which is the one path here
/// that pays a logarithm and is accurate to about 1e-13 rather than to a few
/// ulp. It is the last resort, taken only where the alternative is 0 or an
/// infinity, and only for a positive base: `powf` of a negative base is NaN
/// unless the exponent is an integer, and a logarithm cannot tell the
/// difference.
pub(crate) fn pow_adjoints(upstream: f64, a: f64, b: f64) -> (f64, f64) {
    let (da, db) = ElementwiseOp::Pow.derivatives(a, b);
    let mut adjoint_a = upstream * da;
    let mut adjoint_b = upstream * db;
    let log_a = a.ln();
    let rescue_a = needs_rescue(adjoint_a, &[upstream, a, b]);
    let rescue_b = needs_rescue(adjoint_b, &[upstream, a, log_a]);
    if !rescue_a && !rescue_b {
        return (adjoint_a, adjoint_b);
    }
    let value = a.powf(b);
    if rescue_a {
        adjoint_a = scaled_ratio(&[upstream, b, value], &[a])
            .or_else(|| scaled_power_product(&[upstream, b], a, b - 1.0))
            .unwrap_or(adjoint_a);
    }
    if rescue_b {
        adjoint_b = scaled_ratio(&[upstream, value, log_a], &[])
            .or_else(|| scaled_power_product(&[upstream, log_a], a, b))
            .unwrap_or(adjoint_b);
    }
    (adjoint_a, adjoint_b)
}

/// `|a|^exponent` as `(mantissa, binary exponent)` with the mantissa in
/// `[1, 2)`, for powers far outside what an f64 can hold.
///
/// `exponent * log2(|a|)` carries about one ulp of relative error, so an
/// exponent of order 1000 leaves 1e-13 of absolute error in the logarithm and
/// the same relative error in the result. Every other path in this module is
/// accurate to a few ulp; this one is not, and it is used only where the
/// alternative is zero or an infinity.
fn split_power_magnitude(a: f64, exponent: f64) -> Option<(f64, i32)> {
    let magnitude = a.abs();
    if magnitude == 0.0 || !magnitude.is_finite() || !exponent.is_finite() {
        return None;
    }
    let log2 = exponent * magnitude.log2();
    if !log2.is_finite() || log2.abs() > 1e9 {
        return None;
    }
    let whole = log2.floor();
    Some((f64::powf(2.0, log2 - whole), whole as i32))
}

/// The product of `factors` with `a^exponent`, exponents accumulated
/// separately. `None` for a base that is not strictly positive, or for a factor
/// that is zero or not finite, where the direct product already said what there
/// is to say.
fn scaled_power_product(factors: &[f64], a: f64, exponent: f64) -> Option<f64> {
    if a <= 0.0 {
        return None;
    }
    let (mut mantissa, mut binary_exponent) = split_power_magnitude(a, exponent)?;
    for &x in factors {
        if x == 0.0 || !x.is_finite() {
            return None;
        }
        let (m, e) = split_exponent(x);
        mantissa *= m;
        binary_exponent += e;
    }
    Some(apply_exponent(mantissa, binary_exponent))
}
