import numpy as np
import pytest
import rustmc


def check_gradient(model, data, x):
    value, grad = model.log_density(data, x)
    assert np.isfinite(value)
    for k in range(len(x)):
        xp, xm = x.copy(), x.copy()
        xp[k] += 1e-6
        xm[k] -= 1e-6
        fd = (model.log_density(data, xp)[0]-model.log_density(data, xm)[0])/2e-6
        assert grad[k] == pytest.approx(fd, rel=1e-5, abs=1e-5)


def test_nonlinear_potential_and_deterministic_gradient():
    m = rustmc.ModelBuilder()
    a = m.normal_prior('a', 0., 1.)
    b = m.normal_prior('b', 0., 1.)
    f = (a*b + a*2 - 1)/(b.exp()+1) + a.tanh() + b.softplus() + a.sin() - b.cos()
    m.potential('custom', -0.5*(f-0.7)**2)
    m.deterministic('nonlinear', f)
    compiled = m.compile()
    check_gradient(compiled, {}, np.array([0.2, -0.3]))
    fit = compiled.sample({}, chains=2, draws=10, warmup=10, show_progress=False)
    assert fit.deterministics()['nonlinear'].shape == (2, 10)


def test_grouped_ragged_model_and_new_prediction():
    data = {'g': np.array([0., 1., 0.]), 'x': np.array([1., 2., 3.]),
            'y': np.array([1., 2., 1.]), 'z': np.array([0., 1., 0., 1., 1.])}
    m = rustmc.ModelBuilder(data, dims={'g':'severity', 'x':'severity', 'y':'severity', 'z':'occurrence'})
    beta = m.vector_normal_prior('beta', 2, 0., 1.)
    a = m.normal_prior('a', 0., 1.)
    predictor = beta['g'] * m.data('x') + a
    m.normal_likelihood('amount', predictor, 0.7, 'y')
    m.bernoulli_logit_likelihood('occurs', a, 'z')
    m.deterministic('mu', predictor)
    m.potential('shrink', -0.02*(predictor**2).sum())
    compiled = m.compile()
    check_gradient(compiled, data, np.array([0.1, -0.2, 0.3]))
    fit = compiled.sample(data, chains=2, draws=12, warmup=12, show_progress=False)
    future={'g':np.array([1.,0.,1.,1.]), 'x':np.array([2.,1.,3.,0.])}
    means=fit.predict(future, expected=True, sizes={'occurrence':7})
    assert means['amount'].shape==(2,12,4)
    assert means['occurs'].shape==(2,12,7)
    np.testing.assert_allclose(means['amount'],fit.deterministics(future,sizes={'occurrence':7})['mu'])
    assert fit.posterior_predictive(data=future,sizes={'occurrence':7})['amount'].shape==(24,4)
    bad=dict(future,g=np.array([0.,2.,0.,1.]))
    with pytest.raises(ValueError, match='indices'):
        fit.predict(bad)


@pytest.mark.parametrize('eta', [-40., 40.])
def test_exponential_predictive_extreme_rates(eta):
    m=rustmc.ModelBuilder({'y':np.ones(2000)})
    unused=m.normal_prior('a',0.,1.)
    m.exponential_likelihood('obs', unused*0+eta, 'y')
    prior=rustmc.sample_prior_predictive(m.build(),n_samples=20,seed=32)['obs']
    assert prior.mean()/np.exp(-eta)==pytest.approx(1.,rel=0.03)
    fit=rustmc.sample(m.build(),chains=1,draws=20,warmup=10,show_progress=False)
    pp=fit.posterior_predictive(seed=33)['obs']
    assert pp.mean()/np.exp(-eta)==pytest.approx(1.,rel=0.03)


def test_output_name_collisions_rejected():
    m=rustmc.ModelBuilder({'y':np.ones(3)})
    a=m.normal_prior('a',0.,1.)
    m.normal_likelihood('obs',a,1.,'y')
    m.normal_likelihood('obs',a+100,1.,'y')
    with pytest.raises(ValueError,match='unique'):
        m.build()


def test_compiled_artifact_preserves_structure_without_training_data():
    import json
    m=rustmc.ModelBuilder({'x':np.array([913.17,824.19]),'y':np.ones(2)})
    a=m.normal_prior('a',0.,1.)
    m.normal_likelihood('obs',a*m.data('x'),1.,'y')
    m.potential('regularizer',-0.1*a**4)
    m.deterministic('squared',a**2)
    original=m.compile()
    artifact=original.to_json()
    assert '913.17' not in artifact
    restored=rustmc.CompiledModel.from_json(artifact)
    data={'x':np.array([1.,2.,3.]),'y':np.array([0.,1.,0.])}
    for model in [original,restored]:
        check_gradient(model,data,np.array([0.2]))
    assert original.log_density(data,[0.2])[0]==restored.log_density(data,[0.2])[0]
    changed=json.loads(artifact);changed['version']=999
    with pytest.raises(ValueError,match='version'):
        rustmc.CompiledModel.from_json(json.dumps(changed))
    with pytest.raises(ValueError,match='potential'):
        rustmc.sample_prior_predictive(m.build())


def test_prior_deterministic_and_scalar_prediction_sizes():
    m=rustmc.ModelBuilder({'y':np.ones(2)})
    a=m.normal_prior('a',0.,1.)
    m.normal_likelihood('obs',a,1.,'y')
    m.deterministic('square',a*a)
    prior=rustmc.sample_prior_predictive(m.build(),n_samples=8)
    np.testing.assert_allclose(prior['square'],prior['a']**2)
    fit=rustmc.sample(m.build(),chains=1,draws=5,warmup=5,show_progress=False)
    assert fit.predict(sizes={'obs':9})['obs'].shape==(1,5,9)
    with pytest.raises(ValueError,match='unknown prediction dimension'):
        fit.predict({},sizes={'typo':9})


def test_named_dimensions_cannot_align_by_length_alone():
    m=rustmc.ModelBuilder({'x':np.ones(3),'z':np.ones(3),'y':np.ones(3)},
                         dims={'x':'customers','z':'stores','y':'customers'})
    a=m.normal_prior('a',0.,1.)
    m.normal_likelihood('obs',a*m.data('x') + m.data('z'),1.,'y')
    with pytest.raises(ValueError,match='dimension'):
        m.compile()


def test_integer_group_indices():
    data={'g':np.array([0,1,0],dtype=np.int64),'y':np.ones(3)}
    m=rustmc.ModelBuilder(data)
    beta=m.vector_normal_prior('beta',2,0.,1.)
    m.normal_likelihood('obs',beta['g'],1.,'y')
    check_gradient(m.compile(),data,np.array([0.1,0.2]))
