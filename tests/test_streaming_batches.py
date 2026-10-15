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


def live(references):
    gc.collect()
    return sum(reference() is not None for reference in references)


def test_stream_is_lazy_closes_source_and_bounds_live_payloads():
    alive = []
    consumed = []
    closed = []
    def source():
        try:
            for i in range(50):
                # While a dataset is read, the stream holds only those read
                # before it in the same chunk.
                assert live(alive) <= 2
                data = np.full(100, .5)
                alive.append(weakref.ref(data))
                consumed.append(i)
                yield str(i), {"y": data}
                del data
        finally:
            closed.append(True)
    with model().sample_iter(source(), chunk_size=3, retention="summary", **OPTIONS) as stream:
        assert consumed == []
        first = next(stream)
        assert len(consumed) == 3
        assert first.fit is None and first.diagnostics
        assert len(list(stream)) == 49
    assert closed == [True]
    assert live(alive) == 0


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


def test_init_mapping_keys_must_be_strings():
    with pytest.raises(ValueError, match="init keys must be nonempty strings"):
        model().sample_iter(jobs(2), init={1: [[.1], [.2]]}, **OPTIONS)


def test_previous_chunk_initial_positions_are_released_before_the_next_chunk_is_read():
    alive = []

    def source():
        # While a dataset is read, only the starts of the datasets read before
        # it in the same chunk are alive: at most one, with chunks of two.
        for name, data in jobs(6):
            assert live(alive) <= 1
            yield name, data

    def start(name, data):
        positions = np.array([[.1], [.2]])
        alive.append(weakref.ref(positions))
        return positions

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


def test_bound_models_memoryviews_and_generators_are_passed_as_given():
    compiled = model()
    bound = compiled.bind({"y": np.array([0., 1.])}, id="bound")
    view = memoryview(np.array([0., 1.]))
    items = list(compiled.sample_iter(
        [("bound", bound), ("view", {"y": view}), ("generator", {"y": (x for x in [0.])}),
         ("later", {"y": [1.]})], chunk_size=4, errors="collect", **OPTIONS))
    assert [item.id for item in items] == ["bound", "view", "generator", "later"]
    assert items[0].fit is not None and items[1].fit is not None and items[3].fit is not None
    assert items[2].error


def test_an_explicit_none_start_is_an_error_not_a_random_start():
    for init in ({"a": None}, lambda name, data: None):
        (item,) = model().sample_iter([("a", {"y": [0.]})], errors="collect", init=init,
                                      **OPTIONS)
        assert item.error and "init" in item.error


def test_a_cleanup_failure_after_an_error_is_noted_and_close_retries():
    import sys
    import warnings

    class Source:
        closes = 0

        def __iter__(self):
            return self

        def __next__(self):
            raise RuntimeError("read failed")

        def close(self):
            Source.closes += 1
            if Source.closes == 1:
                raise OSError("cleanup failed")

    stream = model().sample_iter(Source(), **OPTIONS)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(RuntimeError, match="read failed") as raised:
            next(stream)
    if sys.version_info >= (3, 11):
        assert "cleanup failed" in raised.value.__notes__[0]
    else:
        assert any("cleanup failed" in str(warning.message) for warning in caught)
    stream.close()
    assert Source.closes == 2


def test_close_drops_the_input_and_a_failed_chunk_keeps_no_fits():
    payload = [np.full(100, .5) for _ in range(4)]
    watched = weakref.ref(payload[0])
    stream = model().sample_iter(iter([(str(i), {"y": data}) for i, data in enumerate(payload)]),
                                 chunk_size=1, **OPTIONS)
    next(stream)
    del payload
    stream.close()
    gc.collect()
    assert watched() is None

    with pytest.raises(ValueError, match="unknown diagnostic") as raised:
        list(model().sample_iter(jobs(2), retention="summary", parameters=["nope"], **OPTIONS))
    frames = []
    traceback = raised.value.__traceback__
    while traceback is not None:
        frames.append(traceback.tb_frame)
        traceback = traceback.tb_next
    held = [value for frame in frames for value in frame.f_locals.values()]
    assert not any(type(value).__name__ in ("BatchFit", "BatchResult") for value in held)


def test_a_kept_exception_holds_no_inputs_after_an_early_failure():
    alive = []

    def tracked():
        data = np.full(3, .5)
        alive.append(weakref.ref(data))
        return {"y": data}

    def failing_source():
        yield "a", tracked()
        raise RuntimeError("read failed")

    def failing_binding():
        yield "a", tracked()
        yield "b", {"missing": [1.]}

    kept = []
    for source, error in ((failing_source(), RuntimeError), (failing_binding(), ValueError)):
        stream = model().sample_iter(source, chunk_size=2, **OPTIONS)
        with pytest.raises(error) as raised:
            list(stream)
        stream.close()
        kept.append(raised.value)
    assert live(alive) == 0


def test_leaving_a_with_block_on_an_error_keeps_that_error():
    class Source:
        def __iter__(self):
            return self

        def __next__(self):
            raise ValueError("original read error")

        def close(self):
            raise OSError("cleanup failure")

    with pytest.raises(ValueError, match="original read error"):
        with model().sample_iter(Source(), **OPTIONS) as stream:
            next(stream)


def test_batch_item_annotations_resolve_at_runtime():
    import typing
    hints = typing.get_type_hints(mc.BatchItem)
    assert hints["fit"] == typing.Optional[mc.FitResult]


def test_completed_ids_must_not_be_a_single_string():
    with pytest.raises(TypeError, match="single string"):
        model().sample_iter(jobs(2), completed_ids="job-1", **OPTIONS)


def test_a_source_whose_close_raises_stop_iteration_is_reported():
    class Source:
        def __iter__(self):
            return self

        def __next__(self):
            raise StopIteration

        def close(self):
            raise StopIteration("cleanup failed")

    with pytest.raises(RuntimeError, match="closing the input failed"):
        list(model().sample_iter(Source(), **OPTIONS))


def test_a_warnings_filter_that_raises_does_not_replace_the_error():
    import warnings
    from rustmc import batch

    class ErrorWithoutNotes:
        """Stands in for an exception before Python 3.11, which has no notes."""

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        batch._report_failed_close(ErrorWithoutNotes(), OSError("cleanup failure"))


def test_the_ledger_directory_is_removed_when_the_stream_ends(monkeypatch, tmp_path):
    import tempfile
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    assert list(model().sample_iter(jobs(2), **OPTIONS))
    stream = model().sample_iter(jobs(4), chunk_size=1, **OPTIONS)
    next(stream)
    assert list(tmp_path.iterdir())
    stream.close()
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(ValueError):
        list(model().sample_iter([("a", {"missing": [1.]})], **OPTIONS))
    assert list(tmp_path.iterdir()) == []
