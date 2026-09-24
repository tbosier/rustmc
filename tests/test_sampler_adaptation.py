"""Seeding, initialization and metric adaptation of the graph-model sampler."""

import numpy as np
import pytest
import rustmc as r


def normal_model():
    b = r.ModelBuilder()
    mu = b.normal_prior("mu", 0.0, 3.0)
    b.normal_likelihood("obs", mu, 1.0, "y")
    return b.build()


DATA = {"y": np.array([0.3, -0.2, 0.9, 0.4])}


def chains(fit, name="mu"):
    return fit.get_samples_2d()[name]


def test_adjacent_seeds_do_not_share_chains():
    # Chain c used to be seeded `seed + c`, so seed 42's chain 1 replayed seed
    # 43's chain 0 draw for draw.
    options = dict(chains=4, draws=50, warmup=50, show_progress=False)
    first = chains(r.sample(normal_model(), data=DATA, seed=42, **options))
    second = chains(r.sample(normal_model(), data=DATA, seed=43, **options))
    for a in first:
        for b in second:
            assert not np.array_equal(a, b)
    again = chains(r.sample(normal_model(), data=DATA, seed=42, **options))
    np.testing.assert_array_equal(first, again)


@pytest.mark.parametrize("sampler", ["nuts", "hmc"])
def test_chains_start_apart_without_init(sampler):
    # A one-iteration warmup from a tiny step leaves the first draw within a
    # hair of the starting point, which is uniform on (-2, 2) per chain.
    fit = r.sample(normal_model(), data=DATA, chains=4, draws=1, warmup=1,
                   step_size=1e-9, max_tree_depth=1, num_leapfrog_steps=1,
                   sampler=sampler, seed=7, show_progress=False)
    starts = chains(fit)[:, 0]
    assert np.all(np.abs(starts) < 2.0 + 1e-4)
    gaps = np.abs(starts[:, None] - starts[None, :])[np.triu_indices(4, 1)]
    assert gaps.min() > 1e-3, starts


def test_init_still_overrides_the_random_start():
    fit = r.sample(normal_model(), data=DATA, chains=2, draws=1, warmup=1,
                   step_size=1e-9, max_tree_depth=1, init=[[1.5], [-0.5]],
                   seed=7, show_progress=False)
    np.testing.assert_allclose(chains(fit)[:, 0], [1.5, -0.5], atol=1e-4)


def test_random_starts_avoid_regions_without_density():
    b = r.ModelBuilder()
    x = b.normal_prior("x", 0.0, 1.0)
    b.potential("positive", x.log())
    fit = b.compile().sample({}, chains=4, draws=20, warmup=20, seed=3, show_progress=False)
    assert (fit.get_samples()["x"] > 0).all()


def isotropic_vector_model(n):
    b = r.ModelBuilder()
    beta = b.vector_normal_prior("b", n, 0.0, 1.0)
    b.normal_likelihood("y", beta @ "X", 1.0, "y")
    data = {"X": np.eye(n), "y": np.random.default_rng(0).normal(size=n)}
    return b.build(), data


def leapfrog_steps_per_iteration(fit):
    report = fit.transition_diagnostics()
    return report["total_leapfrog_steps"] / report["total_transitions"]


def test_default_metric_handles_an_isotropic_vector_parameter():
    # A dense metric estimated from a 200-draw window in 100 dimensions is
    # mostly noise: that default took ~140 leapfrog steps per iteration here,
    # against ~10 for n independent scalar parameters.
    model, data = isotropic_vector_model(100)
    fit = r.sample(model, data=data, chains=1, draws=200, warmup=500, seed=1,
                   show_progress=False)
    assert leapfrog_steps_per_iteration(fit) < 40


def correlated_vector_model(n, rho=0.95):
    # Equicorrelated columns give coefficients whose posterior correlation a
    # diagonal metric cannot remove; a dense one whitens it away.
    rng = np.random.default_rng(0)
    shared = rng.normal(size=(300, 1))
    X = np.sqrt(rho) * shared + np.sqrt(1 - rho) * rng.normal(size=(300, n))
    data = {"X": X, "y": X @ np.linspace(-1, 1, n) + rng.normal(size=300)}
    b = r.ModelBuilder()
    beta = b.vector_normal_prior("b", n, 0.0, 1.0)
    b.normal_likelihood("y", beta @ "X", 1.0, "y")
    return b, data


def isotropic_builder(n):
    b = r.ModelBuilder()
    beta = b.vector_normal_prior("b", n, 0.0, 1.0)
    b.normal_likelihood("y", beta @ "X", 1.0, "y")
    return b, {"X": np.eye(n), "y": np.random.default_rng(0).normal(size=n)}


def sampling_paths(builder, data, options):
    """Leapfrog steps per iteration of each fit, for every sampling entry point."""
    model, compiled = builder.build(), builder.compile()

    def steps(fits):
        return [leapfrog_steps_per_iteration(fit) for fit in fits]

    return {
        "sample": lambda metric: steps([r.sample(model, data=data, metric=metric, **options)]),
        "compiled": lambda metric: steps([compiled.sample(data, metric=metric, **options)]),
        "sample_batch": lambda metric: steps(
            compiled.sample_batch([data, data], metric=metric, **options)),
        "batch_sample": lambda metric: steps(
            r.batch_sample([(model, data), (model, data)], metric=metric, **options)),
    }


def test_metric_option_is_validated_and_forwarded():
    # Every entry point must pass the option through to warmup. On a
    # correlated target a dense metric needs several times fewer leapfrog
    # steps than a diagonal one, so a dropped "diag" shows up there; the
    # default goes dense on it too, so a dropped "dense" shows up instead on
    # an isotropic target, where the default stays diagonal and an explicit
    # dense metric follows a different trajectory.
    options = dict(chains=1, draws=200, warmup=300, seed=2, show_progress=False)
    builder, data = correlated_vector_model(8)
    for name, run in sampling_paths(builder, data, options).items():
        diag, dense = run("diag"), run("dense")
        assert max(dense) * 2.5 < min(diag), (name, diag, dense)
        assert max(run("auto")) * 2.5 < min(diag), name
    for name, run in sampling_paths(*isotropic_builder(20), options).items():
        assert run("dense") != run("auto"), name
        assert run("diag") == run("auto"), name

    builder, data = correlated_vector_model(8)
    model, compiled = builder.build(), builder.compile()
    for call in (
        lambda: r.sample(model, data=data, metric="unit", **options),
        lambda: compiled.sample(data, metric="unit", **options),
        lambda: compiled.sample_batch([data], metric="bogus", **options),
        lambda: r.batch_sample([(model, data)], metric="bogus", **options),
    ):
        with pytest.raises(ValueError, match="metric"):
            call()
