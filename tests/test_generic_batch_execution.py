import numpy as np
import pytest
import rustmc as r


def model():
    b = r.ModelBuilder()
    beta = b.normal_prior("beta", 0., 1.)
    b.normal_likelihood("obs", beta * "x", 1., "y")
    return b.compile()


def test_generic_batch_id_seeds_failures_and_prediction():
    compiled = model()
    cells = [{"x": np.ones(5), "y": np.ones(5)}, {"x": np.ones(4), "y": np.zeros(4)}]
    options = dict(chains=2, draws=20, warmup=30, show_progress=False, seed=100)
    first = compiled.sample_batch(cells, ids=["a", "b"], threads=1, chunk_size=1, **options)
    second = compiled.sample_batch(cells[::-1], ids=["b", "a"], threads=2, chunk_size=2, **options)
    np.testing.assert_array_equal(first.get("a").get_samples_2d()["beta"], second.get("a").get_samples_2d()["beta"])
    collected = compiled.sample_batch([cells[0], {"x": [1., 2.]}], ids=["a", "broken"], errors="collect", **options)
    assert set(collected.errors) == {"broken"}
    assert collected[0].diagnostics()[0]["name"] == "beta"
    assert collected[0].transition_diagnostics()["total_transitions"] == 100
    prediction = collected[0].predict({"x": np.arange(3.)}, expected=True)["obs"]
    assert prediction.shape == (2, 20, 3)
    np.testing.assert_allclose(prediction[..., 0], 0.)
    np.testing.assert_allclose(prediction[..., 2], 2*prediction[..., 1])
    with pytest.raises(ValueError, match="broken"):
        collected.get("broken")
    assert collected[0].fit.metadata["chains"] == 2


def test_initial_positions_are_unconstrained_and_checked():
    compiled = model()
    data = {"x": np.ones(3), "y": np.zeros(3)}
    options = dict(chains=2, draws=5, warmup=5, show_progress=False)
    compiled.sample(data, init=[[-2.], [2.]], **options)
    with pytest.raises(ValueError, match="chain"):
        compiled.sample(data, init=[[0.]], **options)
    with pytest.raises(ValueError, match="finite"):
        compiled.sample(data, init=[[np.inf], [0.]], **options)


def test_batch_initialization_is_keyed_by_id_and_collects_cell_errors():
    b=r.ModelBuilder()
    x=b.normal_prior('x',0.,1.)
    # Default zero starts are outside support, so success requires the supplied init.
    b.potential('positive',x.log())
    compiled=b.compile()
    options=dict(chains=2,draws=8,warmup=8,seed=58,show_progress=False)
    init={'left':[[0.5],[1.5]],'right':[[2.],[3.]]}
    one=compiled.sample_batch([{},{}],ids=['left','right'],init=init,threads=1,**options)
    two=compiled.sample_batch([{},{}],ids=['right','left'],init=init,threads=2,**options)
    np.testing.assert_array_equal(one.get('left').get_samples_2d()['x'],two.get('left').get_samples_2d()['x'])
    errors=compiled.sample_batch([{}, {}, {}, {}],ids=['ok','missing','shape','parse'],
        init={'ok':[[1.],[2.]],'shape':[[1.]],'parse':'bad'},errors='collect',**options)
    assert set(errors.errors)=={'missing','shape','parse'}
    assert errors.get('ok').fit.get_samples_2d()['x'].shape==(2,8)
    with pytest.raises(ValueError,match='unknown dataset ID'):
        compiled.sample_batch([{}],ids=['ok'],init={'typo':[[1.],[2.]]},**options)


def test_completed_batches_retain_shared_payloads_without_eager_graph_copies():
    builder=r.ModelBuilder()
    beta=builder.vector_normal_prior('beta',2,0.,1.)
    builder.normal_likelihood('obs',beta @ 'X',1.,'y')
    compiled=builder.compile()
    matrix=np.arange(12.,dtype=float).reshape(6,2)/10
    batch=compiled.sample_batch([{'y':np.ones(6)},{'y':np.zeros(6)}],
        shared={'X':matrix},ids=['a','b'],chains=1,draws=5,warmup=6,show_progress=False)
    a,b=batch.get('a'),batch.get('b')
    assert a._shares_data(b,'X')
    assert not a._shares_data(b,'y')
    a.diagnostics();a.summary();a.transition_diagnostics()
    assert a._shares_data(b,'X')
    future={'X':np.array([[1.,2.],[3.,4.]])}
    np.testing.assert_array_equal(a.predict(future,seed=78)['obs'],a.fit.predict(future,seed=78)['obs'])
    # Explicit fit materialization never replaces the compact retained batch payload.
    assert a._shares_data(b,'X')
