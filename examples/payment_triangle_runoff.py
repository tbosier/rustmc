"""Forecast payment-event counts with uncertain ultimates and an undated tail.

Run after installing rustmc: python examples/payment_triangle_runoff.py
These observations count payment events; they are not currency amounts.
"""
import numpy as np
import rustmc


def main():
    counts = np.array([
        [18., 9., 4., 1.],
        [16., 8., 3., np.nan],
        [12., 0., np.nan, np.nan],
        [7., np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan],
    ])
    model = rustmc.DirichletMultinomialRunoff(
        [5., 3., 1.5, .5], total_shape=4., total_rate=.125,
    )
    fit = model.fit(
        counts, [0, 1, 2, 3, 4], valuation=3,
        totals=[32, None, None, None, None],
        draws=1500, warmup=750, chains=4, seed=7,
    )
    future = fit.calendar_samples(3)
    tail = fit.tail_samples.sum(axis=-1)
    observed = int(np.nansum(counts))
    ultimate = fit.ultimate_samples.sum(axis=-1)
    # Horizon covers every remaining regular cell; the tail remains separate.
    np.testing.assert_array_equal(future.sum(axis=-1) + tail + observed, ultimate)
    print("Expected future event counts by period:", future.mean(axis=(0, 1)))
    print("Total regular future events, equal-tailed 95% interval:",
          np.quantile(future.sum(axis=-1), [.025, .975]))
    print("Remaining undated tail events, equal-tailed 95% interval:",
          np.quantile(tail, [.025, .975]))
    print("Ultimate event counts by cohort:", fit.ultimate_samples.mean(axis=(0, 1)))
    print(fit.summary())


if __name__ == "__main__":
    main()
