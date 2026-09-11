"""Bounded streaming, persistent identities, error isolation, and cancellation."""
import gc
import weakref

import numpy as np
import pytest
import rustmc as mc


def model():
    builder = mc.ModelBuilder()
    mu = builder.normal_prior("mu", 0., 1.)
    builder.normal_likelihood("obs", mu, 1., "y")
    return builder.compile()


def jobs(count=7):
    for i in range(count):
        yield f"job-{i}", {"y": np.array([i/10., .2, .3])}


OPTIONS = dict(chains=2, warmup=30, draws=40, threads=2, seed=17, show_progress=False)


def test_stream_is_lazy_closes_source_and_bounds_live_payloads():
    alive = []
    consumed = []
    closed = []
    def source():
        try:
            for i in range(50):
                gc.collect()
                assert sum(ref() is not None for ref in alive) <= 3
                data = np.full(100, .5)
                alive.append(weakref.ref(data))
                consumed.append(i)
                yield str(i), {"y": data}
        finally:
            closed.append(True)
    with model().sample_iter(source(), chunk_size=3, retention="summary", **OPTIONS) as stream:
        assert consumed == []
        first = next(stream)
        assert len(consumed) == 3
        assert first.fit is None and first.diagnostics
        assert len(list(stream)) == 49
    gc.collect()
    assert closed == [True]
    assert all(ref() is None for ref in alive)


def test_draws_match_eager_batch_across_chunks_order_and_resume():
    compiled = model()
    eager = compiled.sample_batch([data for _,data in jobs()],ids=[name for name,_ in jobs()], **OPTIONS)
    for chunk in (1, 3):
        with compiled.sample_iter(reversed(list(jobs())),chunk_size=chunk,completed_ids=iter(["job-2"]), **OPTIONS) as stream:
            results = list(stream)
        assert len(results) == 6
        for result in results:
            np.testing.assert_array_equal(result.fit.get_samples_2d()["mu"],eager.get(result.id).get_samples_2d()["mu"])


def test_errors_and_duplicate_ids_across_chunks():
    compiled = model()
    inputs = [("good", {"y": [0.,1.]}), ("bad", {"missing": [1.]}), ("later", {"y": [2.]})]
    items = list(compiled.sample_iter(inputs,chunk_size=1,errors="collect", **OPTIONS))
    assert items[0].fit is not None
    assert items[1].error and items[1].fit is None
    assert items[2].fit is not None
    with compiled.sample_iter([inputs[0],inputs[0]],chunk_size=1,**OPTIONS) as stream:
        next(stream)
        with pytest.raises(ValueError,match="duplicate"):
            next(stream)


def test_early_close_stops_consumption_and_parameter_selection():
    consumed = []
    def source():
        for name,data in jobs():
            consumed.append(name)
            yield name,data
    stream = mc.sample_iter(model(),source(),chunk_size=2,retention="summary",parameters=["mu"],**OPTIONS)
    assert next(stream).diagnostics[0]["name"] == "mu"
    stream.close()
    assert len(consumed) == 2
    assert list(stream) == []
    with pytest.raises(ValueError,match="retention"):
        mc.sample_iter(model(),jobs(),retention="none")
    with pytest.raises(ValueError,match="cannot be overridden"):
        mc.sample_iter(model(),jobs(),seed_policy="position_v0")


def test_initial_positions_are_selected_by_id_across_chunks():
    compiled = model()
    initial = {name: [[.1], [.2]] for name, _ in jobs()}
    eager = compiled.sample_batch([data for _, data in jobs()],
        ids=[name for name, _ in jobs()], init=initial, **OPTIONS)
    for init in (initial, lambda name, data: initial[name]):
        for item in compiled.sample_iter(jobs(), chunk_size=2, init=init, **OPTIONS):
            np.testing.assert_array_equal(item.fit.get_samples_2d()["mu"],
                                          eager.get(item.id).get_samples_2d()["mu"])
    with pytest.raises(ValueError, match="unknown dataset ID"):
        list(compiled.sample_iter(jobs(1), init={"typo": [[.1], [.2]]}, **OPTIONS))
