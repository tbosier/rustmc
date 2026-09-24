"""`batch_sample` and `CompiledModel.sample_batch` share one batch path."""
import numpy as np
import pytest
import rustmc

X = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
DATASETS = [
    {"x": X, "y": 1.0 + 0.5 * X},
    {"x": X, "y": -1.0 + 2.0 * X},
    {"x": X[:4], "y": np.array([0.3, 0.1, -0.2, 0.4])},
]
OPTIONS = dict(chains=2, draws=30, warmup=30, seed=17)


def builder():
    b = rustmc.ModelBuilder()
    a = b.normal_prior("a", 0.0, 2.0)
    beta = b.normal_prior("beta", 0.0, 2.0)
    s = b.half_normal_prior("s", 1.0)
    b.normal_likelihood("obs", a + beta * "x", s, "y")
    return b


def test_legacy_batch_sample_is_sample_batch_with_positional_seeds():
    spec = builder().build()
    legacy = rustmc.batch_sample([(spec, data) for data in DATASETS], show_progress=False,
                                 **OPTIONS)
    batch = builder().compile().sample_batch(DATASETS, seed_policy="position_v0",
                                             show_progress=False, **OPTIONS)
    assert len(legacy) == len(batch) == len(DATASETS)
    for old, new in zip(legacy, batch):
        for name, draws in old.get_samples_2d().items():
            np.testing.assert_array_equal(draws, new.get_samples_2d()[name])
        # Legacy cells now keep their data like any other batch cell.
        np.testing.assert_array_equal(old.predict(seed=3)["obs"], new.predict(seed=3)["obs"])


def test_legacy_batch_sample_names_the_failed_dataset():
    # log(z) is NaN for z < 0, so dataset 1 has no finite starting point.
    b = rustmc.ModelBuilder()
    r = b.normal_prior("r", 0.0, 1.0)
    b.potential("p", (r * 0.0 + b.data("z")).log().sum())
    spec = b.build()
    with pytest.raises(ValueError, match="dataset '1': .*no initial point"):
        rustmc.batch_sample([(spec, {"z": np.array([1.0, 2.0])}),
                             (spec, {"z": np.array([-1.0, 2.0])})], show_progress=False,
                            **OPTIONS)


def test_legacy_batch_sample_names_the_dataset_it_could_not_prepare():
    # These fail while binding the data, before any sampling starts.
    spec = builder().build()
    with pytest.raises(ValueError, match="dataset '1': .*at least one value"):
        rustmc.batch_sample([(spec, DATASETS[0]), (spec, {"x": X[:0], "y": X[:0]})],
                            show_progress=False, **OPTIONS)
    with pytest.raises(ValueError, match="dataset '2': .*y"):
        rustmc.batch_sample([(spec, DATASETS[0]), (spec, DATASETS[1]), (spec, {"x": X})],
                            show_progress=False, **OPTIONS)


@pytest.mark.parametrize("show_progress", [True, False])
def test_sample_batch_honours_show_progress(capfd, show_progress):
    builder().compile().sample_batch(DATASETS, show_progress=show_progress, **OPTIONS)
    stderr = capfd.readouterr().err
    assert bool(stderr) is show_progress, stderr
