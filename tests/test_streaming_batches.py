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


class Tracked(np.ndarray):
    """An array whose deep copies, the ones the stream keeps, are recorded."""
    copies = []

    def __deepcopy__(self, memo):
        duplicate = np.ndarray.copy(self).view(Tracked)
        Tracked.copies.append(weakref.ref(duplicate))
        return duplicate


def live_copies():
    gc.collect()
    return sum(ref() is not None for ref in Tracked.copies)


def test_stream_is_lazy_closes_source_and_bounds_live_payloads():
    Tracked.copies.clear()
    consumed = []
    closed = []
    def source():
        try:
            for i in range(50):
                # While a dataset is read, the stream holds only the copies of
                # those read before it in the same chunk.
                assert live_copies() <= 2
                consumed.append(i)
                yield str(i), {"y": np.full(100, .5).view(Tracked)}
        finally:
            closed.append(True)
    with model().sample_iter(source(), chunk_size=3, retention="summary", **OPTIONS) as stream:
        assert consumed == []
        first = next(stream)
        assert len(consumed) == 3
        assert first.fit is None and first.diagnostics
        assert len(list(stream)) == 49
    assert closed == [True]
    assert live_copies() == 0


def test_draws_match_eager_batch_across_chunks_order_and_resume():
    compiled = model()
    eager = compiled.sample_batch([data for _,data in jobs()],ids=[name for name,_ in jobs()], **OPTIONS)
    for chunk in (1, 3):
        with compiled.sample_iter(reversed(list(jobs())),chunk_size=chunk,completed_ids=iter(["job-2"]), **OPTIONS) as stream:
            results = list(stream)
        assert [result.id for result in results] == [f"job-{i}" for i in (6, 5, 4, 3, 1, 0)]
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
    initial = {name: [[.1 * i], [.2 * i]] for i, (name, _) in enumerate(jobs())}
    eager = compiled.sample_batch([data for _, data in jobs()],
        ids=[name for name, _ in jobs()], init=initial, **OPTIONS)
    for init in (initial, lambda name, data: initial[name]):
        for item in compiled.sample_iter(jobs(), chunk_size=2, init=init, **OPTIONS):
            np.testing.assert_array_equal(item.fit.get_samples_2d()["mu"],
                                          eager.get(item.id).get_samples_2d()["mu"])
    with pytest.raises(ValueError, match="unknown dataset ID"):
        list(compiled.sample_iter(jobs(1), init={"typo": [[.1], [.2]]}, **OPTIONS))


def test_summary_retention_keeps_fit_warnings():
    compiled = model()
    eager = compiled.sample_batch([data for _, data in jobs(3)], ids=[name for name, _ in jobs(3)], **OPTIONS)
    with compiled.sample_iter(jobs(3), chunk_size=2, retention="summary", **OPTIONS) as stream:
        items = list(stream)
    assert [item.id for item in items] == ["job-0", "job-1", "job-2"]
    for item in items:
        assert item.fit is None
        assert list(item.warnings) == eager.get(item.id).warnings()
    # Too few draws for the rank-based checks, so every fit has a warning.
    with compiled.sample_iter(jobs(1), retention="summary", **{**OPTIONS, "draws": 3}) as stream:
        (item,) = list(stream)
    assert item.warnings and item.warnings == tuple(compiled.sample_batch(
        [data for _, data in jobs(1)], ids=["job-0"], **{**OPTIONS, "draws": 3}).get("job-0").warnings())


def test_reused_source_and_init_buffers_keep_each_job_its_own_values():
    compiled = model()
    expected = {name: data for name, data in jobs(4)}
    starts = {name: np.array([[float(i)], [float(i) + 0.5]]) for i, name in enumerate(expected)}
    eager = compiled.sample_batch(list(expected.values()), ids=list(expected),
                                  init=starts, **OPTIONS)

    def reusing_source():
        buffer = np.empty(3)
        for name, data in expected.items():
            buffer[:] = data["y"]
            yield name, {"y": buffer}

    shared_start = np.empty((2, 1))

    def reusing_init(name, data):
        shared_start[:] = starts[name]
        return shared_start

    for chunk in (1, 3):
        with compiled.sample_iter(reusing_source(), chunk_size=chunk, init=reusing_init,
                                  **OPTIONS) as stream:
            for item in stream:
                np.testing.assert_array_equal(item.fit.get_samples_2d()["mu"],
                                              eager.get(item.id).get_samples_2d()["mu"])


def test_init_mapping_keys_must_be_strings():
    with pytest.raises(ValueError, match="init keys must be nonempty strings"):
        model().sample_iter(jobs(2), init={1: [[.1], [.2]]}, **OPTIONS)


def test_previous_chunk_initial_positions_are_released_before_the_next_chunk_is_read():
    Tracked.copies.clear()

    def source():
        # While a dataset is read, only the starts of the datasets read before
        # it in the same chunk are alive: at most one, with chunks of two.
        for name, data in jobs(6):
            assert live_copies() <= 1
            yield name, data

    def start(name, data):
        return np.array([[.1], [.2]]).view(Tracked)

    with model().sample_iter(source(), chunk_size=2, init=start, **OPTIONS) as stream:
        assert len(list(stream)) == 6


def test_close_from_another_thread_closes_the_source():
    import threading
    closed = []
    reading = threading.Event()
    release = threading.Event()

    def source():
        try:
            for index, pair in enumerate(jobs()):
                if index == 2:  # the second chunk's first read
                    reading.set()
                    release.wait(5)
                yield pair
        finally:
            closed.append(True)

    stream = model().sample_iter(source(), chunk_size=2, **OPTIONS)
    next(stream)
    next(stream)
    results = []
    iterating = threading.Thread(target=lambda: results.append(next(stream)))
    iterating.start()
    assert reading.wait(5)
    # close() from here, while the other thread is inside next(): it waits.
    closing = threading.Thread(target=stream.close)
    closing.start()
    closing.join(0.2)
    assert closing.is_alive() and closed == []
    release.set()
    iterating.join(5)
    closing.join(5)
    assert not closing.is_alive()
    assert results and results[0].id == "job-2"
    assert closed == [True]
    assert list(stream) == []


def test_failed_items_carry_an_error_and_no_warnings():
    inputs = [("good", {"y": [0., 1.]}), ("bad", {"missing": [1.]})]
    good, bad = model().sample_iter(inputs, chunk_size=2, errors="collect", **OPTIONS)
    assert good.error is None and good.fit is not None
    assert bad.error and bad.fit is None and bad.warnings == ()


def test_sampling_options_are_checked_when_the_stream_is_created():
    for options in ({"metric": "bogus"}, {"draws": 0}, {"threads": 0}):
        with pytest.raises(ValueError):
            model().sample_iter([], **{**OPTIONS, **options})
    with pytest.raises(TypeError):
        model().sample_iter([], typo=1)
    # Every job already complete: the options are still checked.
    with pytest.raises(ValueError, match="metric"):
        model().sample_iter(jobs(2), completed_ids=["job-0", "job-1"],
                            **{**OPTIONS, "metric": "bogus"})


def test_malformed_and_masked_data_is_validated_as_sample_batch_validates_it():
    compiled = model()
    good = {"y": [0., 1.]}
    for bad in ({"y": [[1., 2.], [3.]]}, {"y": np.ma.array([100., 0.], mask=[True, False])}):
        inputs = [("good", good), ("bad", bad), ("later", good)]
        eager = compiled.sample_batch([data for _, data in inputs], ids=["good", "bad", "later"],
                                      errors="collect", **OPTIONS)
        assert list(eager.errors) == ["bad"]
        items = list(compiled.sample_iter(inputs, chunk_size=3, errors="collect", **OPTIONS))
        assert [item.id for item in items] == ["good", "bad", "later"]
        assert items[1].error == eager.errors["bad"]
        assert items[0].fit is not None and items[2].fit is not None
    # A malformed initial position is the native layer's to report, per cell.
    inputs = [("good", good), ("bad", good)]
    items = list(compiled.sample_iter(inputs, chunk_size=2, errors="collect",
                                      init={"bad": [["oops"]]}, **OPTIONS))
    assert items[0].fit is not None and items[1].error


def test_a_source_that_fails_to_close_is_reported_at_the_end():
    class Source:
        def __init__(self):
            self.items = iter(list(jobs(1)))

        def __iter__(self):
            return self

        def __next__(self):
            return next(self.items)

        def close(self):
            raise RuntimeError("source cleanup failed")

    with pytest.raises(RuntimeError, match="source cleanup failed"):
        list(model().sample_iter(Source(), **OPTIONS))


def test_init_for_an_id_only_listed_as_completed_is_reported():
    with pytest.raises(ValueError, match="unknown dataset ID 'absent'"):
        list(model().sample_iter([], completed_ids=["absent"], init={"absent": [[0.], [0.]]},
                                 **OPTIONS))
    # An ID a dataset had, and that was skipped as completed, is known.
    assert list(model().sample_iter(jobs(1), completed_ids=["job-0"],
                                    init={"job-0": [[0.], [0.]]}, **OPTIONS)) == []
