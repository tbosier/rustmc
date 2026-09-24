import os
import sys

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
    # The origin is outside the support; supplied starts inside it must be used as given.
    b.potential('positive',x.log())
    compiled=b.compile()
    options=dict(chains=2,draws=8,warmup=8,seed=58,show_progress=False)
    init={'left':[[0.5],[1.5]],'right':[[2.],[3.]]}
    one=compiled.sample_batch([{},{}],ids=['left','right'],init=init,threads=1,**options)
    two=compiled.sample_batch([{},{}],ids=['right','left'],init=init,threads=2,**options)
    np.testing.assert_array_equal(one.get('left').get_samples_2d()['x'],two.get('left').get_samples_2d()['x'])
    errors=compiled.sample_batch([{}, {}, {}, {}, {}],ids=['ok','missing','outside','shape','parse'],
        init={'ok':[[1.],[2.]],'outside':[[0.],[1.]],'shape':[[1.]],'parse':'bad'},
        errors='collect',**options)
    assert set(errors.errors)=={'outside','shape','parse'}
    assert errors.get('ok').fit.get_samples_2d()['x'].shape==(2,8)
    # Without init, random starts are searched for inside the support.
    assert (errors.get('missing').fit.get_samples_2d()['x']>0).all()
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


def test_identity_display_transform_shares_the_raw_posterior():
    """No parameter is derived here, so the display draws are the raw draws."""
    compiled=model()
    batch=compiled.sample_batch([{'x':np.ones(4),'y':np.ones(4)}],ids=['a'],
        chains=2,draws=10,warmup=10,show_progress=False)
    cell=batch.get('a')
    assert cell._posterior_allocations()==1
    # Retrieval and materialization must hand out the retained posterior itself,
    # not a copy of it -- ownership, not merely within-object aliasing.
    materialized=cell.fit
    assert cell._shares_posterior_with(materialized)
    assert batch[0]._shares_posterior_with(materialized)
    assert batch.get('a')._shares_posterior_with(batch[0].fit)
    assert batch[0]._posterior_allocations()==1


def test_batch_cells_do_not_share_a_posterior_with_each_other():
    """The ownership hook must be able to say no, or it proves nothing."""
    compiled=model()
    batch=compiled.sample_batch([{'x':np.ones(4),'y':np.ones(4)},
                                 {'x':np.ones(4),'y':np.zeros(4)}],
        ids=['a','b'],chains=2,draws=10,warmup=10,show_progress=False)
    assert not batch.get('a')._shares_posterior_with(batch.get('b').fit)


def test_derived_parameters_still_get_their_own_display_posterior():
    builder=r.ModelBuilder()
    scale=builder.half_normal_prior('scale',1.)
    builder.normal_prior('effect',0.,scale)
    builder.normal_likelihood('obs',builder.normal_prior('mu',0.,1.),1.,'y')
    compiled=builder.compile()
    batch=compiled.sample_batch([{'y':np.ones(4)}],ids=['a'],
        chains=2,draws=10,warmup=10,show_progress=False)
    cell=batch.get('a')
    assert cell._posterior_allocations()==2
    # The derived values are genuinely different from the sampled ones.
    assert set(cell.get_samples_2d())=={'scale','effect','mu'}


def resident_mb():
    """Current resident set size, in MB.

    Deliberately not `resource.getrusage(...).ru_maxrss`: that is a
    process-lifetime high-water mark, so an earlier peak makes the difference
    between two readings zero no matter how much the batch retains. `statm`
    reports what is resident right now, which is the thing under test.
    """
    with open("/proc/self/statm") as handle:
        resident_pages=int(handle.read().split()[1])
    return resident_pages*os.sysconf("SC_PAGE_SIZE")/1e6


@pytest.mark.skipif(not sys.platform.startswith("linux"),
                    reason="reads resident set size from /proc/self/statm")
def test_batch_retention_stays_near_a_single_copy_of_the_posterior():
    """A retained batch must not hold several copies of its own draws.

    Generic batches used to keep three: the raw tree, a bit-identical display
    tree, and a flattened `BatchModelResult` of the display tree.
    """
    import gc

    builder=r.ModelBuilder()
    params=40
    beta=builder.vector_normal_prior('beta',params,0.,1.)
    builder.normal_likelihood('obs',beta @ 'X',1.,'y')
    compiled=builder.compile()
    rng=np.random.default_rng(0)
    matrix=rng.normal(size=(60,params))
    cells=[{'X':matrix,'y':rng.normal(size=60)} for _ in range(8)]
    chains,draws=2,3000
    one_copy_mb=8*chains*draws*params*8/1e6
    gc.collect()
    before=resident_mb()
    batch=compiled.sample_batch(cells,ids=[str(i) for i in range(8)],chains=chains,
        warmup=200,draws=draws,seed=1,threads=1,show_progress=False)
    gc.collect()
    retained_mb=resident_mb()-before
    assert len(batch)==8
    # Three copies plus telemetry measured ~3.9x one copy; one copy plus
    # telemetry lands near 1.4x.
    assert retained_mb<2.5*one_copy_mb,f"{retained_mb:.1f} MB for a {one_copy_mb:.1f} MB posterior"


@pytest.mark.skipif(not sys.platform.startswith("linux"),
                    reason="reads resident set size from /proc/self/statm")
def test_many_chains_do_not_inflate_the_regrouped_legacy_batch():
    """Regrouping the legacy flat draw list must not keep per-chain slack.

    Splitting the flat list with `Vec::split_off` leaves every chain holding
    capacity for the whole remaining suffix, which is quadratic in the chain
    count even though the draw values themselves are only moved.
    """
    import gc

    builder=r.ModelBuilder()
    builder.normal_prior('beta',0.,1.)
    builder.normal_likelihood('obs',builder.normal_prior('mu',0.,1.),1.,'y')
    spec=builder.build()
    chains,draws=200,200
    values_mb=chains*draws*2*8/1e6
    gc.collect()
    before=resident_mb()
    results=r.batch_sample([(spec,{'y':np.ones(3)})],chains=chains,draws=draws,
        warmup=20,show_progress=False)
    gc.collect()
    retained_mb=resident_mb()-before
    assert results[0].chains==chains and results[0].draws==draws
    # Quadratic slack measured ~97 MB here for 0.3 MB of parameter values.
    assert retained_mb<20*values_mb,f"{retained_mb:.1f} MB for {values_mb:.2f} MB of draws"
