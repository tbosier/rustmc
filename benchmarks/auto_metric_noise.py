"""How often ``metric="auto"``'s rule would choose dense on windows with no correlation.

Run: python -m benchmarks.auto_metric_noise

A NumPy reimplementation of the decision in ``rust_core/src/mass_matrix.rs``
(``effective_draws`` and ``dense_beats_diagonal``, with exact eigenvalues in place of
its 100 power iterations), applied to simulated warmup windows of stationary Gaussian
AR(1) coordinates. For each case it prints how many of 30 windows go dense under three
effective-draw counts: ``old``, the mean's count of the average lag-one
autocorrelation (0.14.0); ``averaged``, the second-moment count of that average; and
``least``, the least per-coordinate second-moment count (now used). The isotropic
cases have no correlation, so a dense choice there trusts noise; the correlated cases
are where dense is meant to be chosen. It tests the rule, not NUTS: real warmup draws
are neither stationary nor AR(1).
"""
from __future__ import annotations

import numpy as np

MARGIN = 8.0
WINDOWS = 30


def effective_draws(window, rule):
    n = len(window)
    mean, shift = window.mean(0), window[0]
    variance = window.var(0, ddof=1)
    lag = ((window[1:] - shift) * (window[:-1] - shift)).sum(0) / (n - 1) - (mean - shift) ** 2
    rho = np.clip(lag / variance, -1, 1)
    if rule == "least":
        rho2 = np.minimum(rho ** 2, 0.999)
        return n * np.min((1 - rho2) / (1 + rho2))
    average = np.clip(rho.mean(), 0, 0.95)
    if rule == "old":
        return n * (1 - average) / (1 + average)
    return n * (1 - average ** 2) / (1 + average ** 2)


def chooses_dense(window, rule):
    ratio = window.shape[1] / effective_draws(window, rule)
    if ratio >= 1:
        return False
    edge = np.sqrt(ratio)
    noise_condition = ((1 + edge) / (1 - edge)) ** 2
    eigenvalues = np.linalg.eigvalsh(np.corrcoef(window, rowvar=False))
    return eigenvalues[-1] / eigenvalues[0] > MARGIN * noise_condition


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
    ("AR(1) correlation 0.95, all at 0.5", 200, np.full(50, 0.5), correlation_factor(50, 0.95)),
    ("AR(1) correlation 0.95, all at 0.8", 200, np.full(50, 0.8), correlation_factor(50, 0.95)),
]


def main():
    print(f"{WINDOWS} windows per case, NumPy seeds 0 to {WINDOWS - 1}; dense choices out of {WINDOWS}")
    print(f"{'case':44s} {'draws':>5s} {'old':>4s} {'averaged':>8s} {'least':>5s}")
    for label, draws, coefficients, mixing in CASES:
        counts = {rule: 0 for rule in ("old", "averaged", "least")}
        for seed in range(WINDOWS):
            window = ar1_window(draws, coefficients, np.random.default_rng(seed), mixing)
            for rule in counts:
                counts[rule] += chooses_dense(window, rule)
        print(f"{label:44s} {draws:5d} {counts['old']:4d} {counts['averaged']:8d} {counts['least']:5d}")


if __name__ == "__main__":
    main()
