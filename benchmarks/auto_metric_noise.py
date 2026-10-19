"""How often ``metric="auto"``'s rule chooses dense on simulated warmup windows.

Run: python -m benchmarks.auto_metric_noise

NumPy reimplementations of two decisions, with exact eigenvalues in place of the
Rust code's 100 power iterations, applied to simulated warmup windows of stationary
Gaussian AR(1) coordinates. ``old`` is 0.14.0's rule: the sample correlation
matrix's condition number against a noise bound from the effective draws of the
average lag-one autocorrelation. ``cv`` is the rule in ``rust_core/src/mass_matrix.rs``
(``dense_beats_diagonal``): each half of the window estimates a diagonal and a dense
metric, each is scored by the condition number it leaves the other half's sample
covariance, and dense must leave a geometric mean over the two folds smaller by a
margin: 1 in early warmup windows and 2 in the last, as shipped. The columns give the
rule at margins of 1, 2 and 4. For each case it prints how many of 30 windows go dense. The isotropic cases
have no correlation, so a dense choice there trusts noise; the correlated cases are
where dense is meant to be chosen. It tests the rules, not NUTS: real warmup draws
are neither stationary nor AR(1).
"""
from __future__ import annotations

import numpy as np

OLD_MARGIN = 8.0
CV_MARGINS = (1.0, 2.0, 4.0)
WINDOWS = 30


def old_rule(window):
    n, d = window.shape
    mean, shift = window.mean(0), window[0]
    variance = window.var(0, ddof=1)
    lag = ((window[1:] - shift) * (window[:-1] - shift)).sum(0) / (n - 1) - (mean - shift) ** 2
    average = np.clip(np.clip(lag / variance, -1, 1).mean(), 0, 0.95)
    effective = n * (1 - average) / (1 + average)
    ratio = d / effective
    if ratio >= 1:
        return False
    edge = np.sqrt(ratio)
    noise_condition = ((1 + edge) / (1 - edge)) ** 2
    eigenvalues = np.linalg.eigvalsh(np.corrcoef(window, rowvar=False))
    return eigenvalues[-1] / eigenvalues[0] > OLD_MARGIN * noise_condition


def shrunk_covariance(draws):
    # `shrink_correlations`: off-diagonal scaled by n / (2d) below two draws per dimension.
    n, d = draws.shape
    covariance = np.cov(draws, rowvar=False)
    weight = max(0.0, 1 - n / (2 * d))
    covariance[~np.eye(d, dtype=bool)] *= 1 - weight
    return covariance


def log_condition(matrix):
    eigenvalues = np.linalg.eigvalsh(matrix)
    return np.log(eigenvalues[-1] / eigenvalues[0]) if eigenvalues[0] > 0 else np.inf


def cv_rule(window, margin):
    n, d = window.shape
    if n // 2 <= d:
        return False
    first, second = window[: n // 2], window[n // 2:]
    diagonal = dense = 0.0
    for train, test in ((first, second), (second, first)):
        test_covariance = np.cov(test, rowvar=False)
        inv_sd = 1 / train.std(0, ddof=1)
        diagonal += log_condition(test_covariance * np.outer(inv_sd, inv_sd))
        lower = np.linalg.cholesky(shrunk_covariance(train))
        half = np.linalg.solve(lower, test_covariance)
        dense += log_condition(np.linalg.solve(lower, half.T))
    return bool(np.isfinite(dense) and dense / 2 < diagonal / 2 - np.log(margin))


def ar1_window(draws, coefficients, rng, mixing=None):
    state = rng.standard_normal(len(coefficients))
    rows = []
    for _ in range(draws):
        state = coefficients * state + np.sqrt(1 - coefficients ** 2) * rng.standard_normal(
            len(coefficients))
        rows.append(state.copy())
    window = np.array(rows)
    return window if mixing is None else window @ mixing.T


def pairwise_rotation(pairs):
    # Coordinate 2k and 2k + 1 are (slow_k +- fast_k) / sqrt(2): orthogonal, so
    # the population covariance is still the identity.
    rotation = np.zeros((2 * pairs, 2 * pairs))
    for k in range(pairs):
        rotation[2 * k, k] = rotation[2 * k + 1, k] = 1 / np.sqrt(2)
        rotation[2 * k, pairs + k] = 1 / np.sqrt(2)
        rotation[2 * k + 1, pairs + k] = -1 / np.sqrt(2)
    return rotation


def correlation_factor(dimension, correlation):
    indices = np.arange(dimension)
    return np.linalg.cholesky(correlation ** np.abs(np.subtract.outer(indices, indices)))


CASES = [
    ("isotropic, all coordinates 0.6", 200, np.full(40, 0.6), None),
    ("isotropic, 20 at 0.99 and 20 at 0", 200, np.r_[np.full(20, 0.99), np.zeros(20)], None),
    ("isotropic, those rotated in pairs", 200, np.r_[np.full(20, 0.99), np.zeros(20)],
     pairwise_rotation(20)),
    ("isotropic, 20 at 0.97 and 20 at 0, rotated", 400, np.r_[np.full(20, 0.97), np.zeros(20)],
     pairwise_rotation(20)),
    ("isotropic, 100 at 0.5", 400, np.full(100, 0.5), None),
    ("isotropic, 10 at 0.5, halves of 11", 23, np.full(10, 0.5), None),
    ("isotropic, 20 at 0.99 and 20 at 0, rotated", 89, np.r_[np.full(20, 0.99), np.zeros(20)],
     pairwise_rotation(20)),
    ("AR(1) correlation 0.95, all at 0.5", 200, np.full(50, 0.5), correlation_factor(50, 0.95)),
    ("AR(1) correlation 0.95, all at 0.8", 200, np.full(50, 0.8), correlation_factor(50, 0.95)),
    ("AR(1) correlation 0.95, one at 0.99, 49 at 0.3", 200, np.r_[0.99, np.full(49, 0.3)],
     correlation_factor(50, 0.95)),
    ("AR(1) correlation 0.8, 20 at 0.3", 80, np.full(20, 0.3), correlation_factor(20, 0.8)),
    ("AR(1) correlation 0.5, all at 0.5", 200, np.full(50, 0.5), correlation_factor(50, 0.5)),
]


def main():
    print(f"{WINDOWS} windows per case, NumPy seeds 0 to {WINDOWS - 1}; dense choices out of {WINDOWS}")
    names = ["cv m1", "cv m2", "cv m4"]
    print(f"{'case':50s} {'draws':>5s} {'old':>4s} " + " ".join(f"{name:>5s}" for name in names))
    for label, draws, coefficients, mixing in CASES:
        old = 0
        cv = [0] * len(CV_MARGINS)
        for seed in range(WINDOWS):
            window = ar1_window(draws, coefficients, np.random.default_rng(seed), mixing)
            old += old_rule(window)
            for index, margin in enumerate(CV_MARGINS):
                cv[index] += cv_rule(window, margin)
        print(f"{label:50s} {draws:5d} {old:4d} " + " ".join(f"{count:5d}" for count in cv))


if __name__ == "__main__":
    main()
