import numpy as np
import pytest

import rustmc


def make_compiled():
    builder = rustmc.ModelBuilder()
    beta = builder.normal_prior("beta", 0.0, 1.0)
    builder.normal_likelihood("obs", beta * "x", 1.0, "y")
    return builder, builder.compile()


def test_bind_accepts_different_row_counts_without_changing_structure():
    _, compiled = make_compiled()
    structure_id = compiled.structure_id
    short = compiled.bind({"x": np.ones(3), "y": np.ones(3)}, id="short")
    long = compiled.bind({"x": np.ones(9), "y": np.ones(9)}, id="long")
    assert (short.n_obs, long.n_obs) == (3, 9)
    assert compiled.structure_id == structure_id
    with compiled as same:
        assert same.structure_id == structure_id
    with short as same_bound:
        assert same_bound.id == "short"


def test_bind_schema_errors_are_actionable():
    _, compiled = make_compiled()
    with pytest.raises(ValueError, match="missing required"):
        compiled.bind({"x": np.ones(3)})
    with pytest.raises(ValueError, match="unexpected data key"):
        compiled.bind({"x": np.ones(3), "y": np.ones(3), "typo": np.ones(3)})
    with pytest.raises(ValueError, match="length"):
        compiled.bind({"x": np.ones(2), "y": np.ones(3)})
    with pytest.raises(ValueError, match="non-finite"):
        compiled.bind({"x": np.array([np.nan]), "y": np.ones(1)})


def _add_scalar_likelihood(builder, family, eta):
    if family == "bernoulli":
        builder.bernoulli_logit_likelihood("obs", eta, "y")
    elif family == "poisson":
        builder.poisson_log_likelihood("obs", eta, "y")
    elif family == "exponential":
        builder.exponential_likelihood("obs", eta, "y")
    elif family == "lognormal":
        builder.log_normal_likelihood("obs", eta, 1.0, "y")
    else:
        builder.negative_binomial_likelihood("obs", eta, 1.0, "y")


@pytest.mark.parametrize(
    ("family", "invalid", "message"),
    [
        ("bernoulli", 0.5, "binary"),
        ("poisson", 1.5, "non-negative integers"),
        ("exponential", -1.0, "non-negative"),
        ("lognormal", 0.0, "strictly positive"),
        ("negative_binomial", 2.5, "non-negative integers"),
    ],
)
def test_bind_revalidates_likelihood_domain(family, invalid, message):
    builder = rustmc.ModelBuilder()
    eta = builder.normal_prior("eta", 0.0, 1.0)
    _add_scalar_likelihood(builder, family, eta)

    compiled = builder.compile()
    with pytest.raises(ValueError, match=message):
        compiled.bind({"y": np.array([invalid])})


@pytest.mark.parametrize(
    ("family", "invalid"),
    [
        ("bernoulli", 1.0 + 1e-12),
        ("poisson", -1e-12),
        ("poisson", 2.0 + 1e-12),
        ("exponential", -1e-12),
        ("negative_binomial", 2.0 + 1e-12),
    ],
)
def test_likelihood_support_boundaries_are_exact(family, invalid):
    bound_builder = rustmc.ModelBuilder({"y": np.array([invalid])})
    eta = bound_builder.normal_prior("eta", 0.0, 1.0)
    _add_scalar_likelihood(bound_builder, family, eta)
    with pytest.raises(ValueError, match="requires"):
        bound_builder.compile()

    template = rustmc.ModelBuilder()
    eta = template.normal_prior("eta", 0.0, 1.0)
    _add_scalar_likelihood(template, family, eta)
    with pytest.raises(ValueError, match="requires"):
        template.compile().bind({"y": np.array([invalid])})


def test_tiny_positive_lognormal_observation_is_valid():
    tiny = np.array([1e-12])
    builder = rustmc.ModelBuilder({"y": tiny})
    eta = builder.normal_prior("eta", 0.0, 1.0)
    _add_scalar_likelihood(builder, "lognormal", eta)
    compiled = builder.compile()
    assert compiled.bind({"y": tiny}).n_obs == 1


def test_bound_model_cannot_be_used_with_a_different_compiled_structure():
    _, first = make_compiled()
    bound = first.bind({"x": np.ones(3), "y": np.ones(3)})

    builder = rustmc.ModelBuilder()
    alpha = builder.normal_prior("alpha", 0.0, 1.0)
    builder.normal_likelihood("other", alpha, 1.0, "z")
    second = builder.compile()

    with pytest.raises(ValueError, match="different CompiledModel"):
        second.sample(
            bound,
            chains=1,
            draws=1,
            warmup=1,
            sampler="hmc",
            num_leapfrog_steps=1,
            show_progress=False,
        )
    with pytest.raises(ValueError, match="different CompiledModel"):
        second.sample_batch(
            [bound],
            chains=1,
            draws=1,
            warmup=1,
            sampler="hmc",
            num_leapfrog_steps=1,
            show_progress=False,
        )


def test_batch_preserves_ids_and_order():
    _, compiled = make_compiled()
    datasets = [
        {"x": np.array([1.0, 2.0]), "y": np.array([1.0, 2.0])},
        {"x": np.array([1.0, 2.0, 3.0]), "y": np.array([1.0, 2.0, 3.0])},
    ]
    fit = compiled.sample_batch(
        datasets,
        ids=["small", "large"],
        chains=1,
        draws=2,
        warmup=2,
        sampler="hmc",
        num_leapfrog_steps=1,
        show_progress=False,
    )
    assert fit.ids == ["small", "large"]
    assert len(fit) == 2
    assert fit[0].draws == 2
    assert fit[-1].draws == 2
    assert len(list(fit)) == 2
    with pytest.raises(IndexError):
        _ = fit[2]


def test_batch_shared_inputs_are_bound_once_and_cannot_be_shadowed():
    _, compiled = make_compiled()
    fit = compiled.sample_batch(
        [{"y": np.ones(2)}, {"y": np.ones(2)}],
        shared={"x": np.ones(2)},
        chains=1,
        draws=1,
        warmup=1,
        sampler="hmc",
        num_leapfrog_steps=1,
        show_progress=False,
    )
    assert len(fit) == 2
    with pytest.raises(ValueError, match="both shared"):
        compiled.sample_batch(
            [{"x": np.ones(2), "y": np.ones(2)}],
            shared={"x": np.ones(2)},
            chains=1,
            draws=1,
            warmup=1,
            sampler="hmc",
            show_progress=False,
        )


def test_compiled_and_legacy_sampling_parity():
    builder, compiled = make_compiled()
    data = {"x": np.array([1.0, 2.0]), "y": np.array([1.0, 2.0])}
    kwargs = dict(chains=1, draws=3, warmup=3, seed=17, sampler="hmc", num_leapfrog_steps=2, show_progress=False)
    modern = compiled.sample(data, **kwargs).get_samples()["beta"]
    legacy = rustmc.sample(builder.build(), data, **kwargs).get_samples()["beta"]
    np.testing.assert_array_equal(modern, legacy)


def test_arviz_ppc_export_includes_observed_data():
    pytest.importorskip("arviz")
    _, compiled = make_compiled()
    data = {"x": np.array([1.0, 2.0]), "y": np.array([1.5, 2.5])}
    fit = compiled.sample(
        data,
        chains=1,
        draws=2,
        warmup=2,
        sampler="hmc",
        num_leapfrog_steps=1,
        show_progress=False,
    )

    idata = fit.to_arviz(include_ppc=True)
    groups = idata.groups() if callable(idata.groups) else idata.groups
    normalized_groups = {group.removeprefix("/") for group in groups}
    assert "observed_data" in normalized_groups
    assert "posterior_predictive" in normalized_groups

    def dataset(group_name):
        group = getattr(idata, group_name)
        return getattr(group, "dataset", group)

    posterior_dataset = dataset("posterior")
    stats_dataset = dataset("sample_stats")
    observed_dataset = dataset("observed_data")
    log_likelihood_dataset = dataset("log_likelihood")
    ppc_dataset = dataset("posterior_predictive")
    assert posterior_dataset["beta"].shape == (1, 2)
    assert stats_dataset["diverging"].shape == (1, 2)
    np.testing.assert_array_equal(observed_dataset["obs"].values, data["y"])
    assert log_likelihood_dataset["obs"].shape == (1, 2, 2)
    assert ppc_dataset["obs"].shape == (1, 2, 2)
