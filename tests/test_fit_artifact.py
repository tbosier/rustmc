import copy
import json

import numpy as np
import pytest
import rustmc


@pytest.fixture(scope='module')
def saved_fit():
    data={'group':np.array([0,1,0]),'x':np.array([1.,2.,3.]),
          'y':np.array([1.,2.,3.]),'occurred':np.array([0.,1.,0.,1.])}
    m=rustmc.ModelBuilder(data,dims={'group':'severity','x':'severity','y':'severity','occurred':'occurrence'})
    mu=m.normal_prior('mu',0.,1.)
    tau=m.half_normal_prior('tau',1.)
    z=m.vector_normal_prior('z',2,0.,1.)
    # This prior also verifies reconstruction of the noncentered display mapping.
    beta=m.normal_prior('beta',mu,tau)
    predictor=mu+tau*z['group']+beta*m.data('x')
    m.normal_likelihood('response',predictor,0.8,'y')
    m.bernoulli_logit_likelihood('occurs',mu,'occurred')
    m.potential('smooth',-0.1*beta**4)
    m.deterministic('mu_response',predictor)
    fit=m.compile().sample(data,chains=2,draws=12,warmup=15,seed=810,show_progress=False)
    return fit,json.loads(fit.to_json())


def test_fit_roundtrip_exact_replay(saved_fit):
    fit,artifact=saved_fit
    restored=rustmc.FitResult.from_json(json.dumps(artifact))
    for accessor in ['get_samples','get_samples_2d','deterministics','log_likelihood']:
        left=getattr(fit,accessor)();right=getattr(restored,accessor)()
        for name in left:
            np.testing.assert_array_equal(left[name],right[name])
    future={'group':np.array([1,0,1,0]),'x':np.array([2.,4.,6.,8.])}
    for expected in [False,True]:
        left=fit.predict(future,seed=47,expected=expected,sizes={'occurrence':5})
        right=restored.predict(future,seed=47,expected=expected,sizes={'occurrence':5})
        for name in left:
            np.testing.assert_array_equal(left[name],right[name])
    assert fit.metadata==restored.metadata
    assert fit.accept_rates()==restored.accept_rates()
    assert fit.step_sizes()==restored.step_sizes()
    assert fit.divergences()==restored.divergences()
    assert artifact==json.loads(restored.to_json())
    assert json.loads(restored.model.to_json())==artifact['model']
    assert 'y' in artifact['training']['vectors']
    assert 'bound_data_1d' not in json.dumps(artifact['model'])
    # Restored model getter has training defaults and remains independently reusable.
    restored.model.bind({})


@pytest.mark.parametrize('corruption',[
    lambda a:a.update(version=999),
    lambda a:a['posterior'].update(coordinate_space='raw'),
    lambda a:a['posterior']['param_names'].reverse(),
    lambda a:a['posterior']['samples'][0].pop(),
    lambda a:a['posterior']['samples'][0][0].pop(),
    lambda a:a['posterior']['samples'][0][0].__setitem__(0,float('nan')),
    lambda a:a['posterior']['samples'][0][0].__setitem__(0,1e300),
    lambda a:a['posterior']['samples'][0][0].__setitem__(1,-1.),
    lambda a:a['posterior']['accept_rates'].pop(),
    lambda a:a['posterior']['step_sizes'].__setitem__(0,-1.),
    lambda a:a['posterior']['transitions'][0].pop(),
    lambda a:a['posterior']['transitions'][0][0].update(is_warmup=False),
    lambda a:a['posterior']['transitions'][0][0].update(accept_prob=1.1),
    lambda a:a['posterior']['transitions'][0][0].update(energy_error='nan',divergent=False),
    lambda a:a['posterior']['divergences'].__setitem__(0,999),
    lambda a:a['training']['vectors']['y'].pop(),
    lambda a:a['training']['vectors']['group'].__setitem__(0,99.),
    lambda a:a['model']['schema']['observations'][0].update(dim='corrupted'),
    lambda a:a['model']['definition']['priors'][2]['VectorNormal'].update(n=2**63),
])
def test_fit_corruption_rejected(saved_fit,corruption):
    artifact=copy.deepcopy(saved_fit[1]);corruption(artifact)
    with pytest.raises(ValueError):
        rustmc.FitResult.from_json(json.dumps(artifact))


def test_global_sample_fit_persistence():
    m=rustmc.ModelBuilder({'y':np.ones(3)})
    a=m.normal_prior('a',0.,1.)
    m.normal_likelihood('obs',a,1.,'y')
    fit=rustmc.sample(m.build(),chains=1,draws=4,warmup=4,show_progress=False)
    restored=rustmc.FitResult.from_json(fit.to_json())
    np.testing.assert_array_equal(fit.predict(seed=21)['obs'],restored.predict(seed=21)['obs'])


def test_divergent_nonfinite_energy_uses_explicit_json_tokens():
    m=rustmc.ModelBuilder()
    m.normal_prior('a',0.,1.)
    fit=m.compile().sample({},chains=1,draws=2,warmup=1,sampler='hmc',
                           step_size=1e100,num_leapfrog_steps=2,show_progress=False)
    artifact=json.loads(fit.to_json())
    assert any(isinstance(t['energy_error'],str)
               for t in artifact['posterior']['transitions'][0])
    restored=rustmc.FitResult.from_json(json.dumps(artifact))
    assert restored.divergences()==fit.divergences()
    assert json.loads(restored.to_json())==artifact


def test_matrix_training_data_and_future_width_roundtrip():
    data={'X':np.array([[1.,2.],[2.,1.],[0.,1.]]),'y':np.array([1.,2.,0.])}
    m=rustmc.ModelBuilder(data)
    beta=m.vector_normal_prior('beta',2,0.,1.)
    m.normal_likelihood('obs',beta @ 'X',1.,'y')
    fit=m.compile().sample(data,chains=1,draws=4,warmup=4,show_progress=False)
    artifact=json.loads(fit.to_json())
    restored=rustmc.FitResult.from_json(json.dumps(artifact))
    future={'X':np.array([[3.,1.],[1.,5.]])}
    np.testing.assert_array_equal(fit.predict(future,seed=27)['obs'],restored.predict(future,seed=27)['obs'])
    artifact['training']['matrices']['X'][2]=3
    with pytest.raises(ValueError):
        rustmc.FitResult.from_json(json.dumps(artifact))

def _discrete_prior_fit_artifact():
    """A saved fit whose definition has been edited to declare a discrete prior.

    The library will not produce one -- `compile()` refuses a discrete prior --
    so it has to be built by hand. That is the point: a hand-edited, corrupted
    or future-version artifact is exactly what a loader has to refuse.
    """
    import json

    n = 8
    data = {"x": np.linspace(-1.0, 1.0, n), "y": np.zeros(n)}
    builder = rustmc.ModelBuilder(data)
    flag = builder.normal_prior("flag", 0.0, 1.0)
    builder.normal_likelihood("obs", flag * "x", 1.0, "y")
    fit = rustmc.sample(
        builder.build(), chains=1, draws=6, warmup=6, seed=2, show_progress=False
    )
    artifact = json.loads(fit.to_json())
    artifact["model"]["definition"]["priors"][0] = {
        "Bernoulli": {"name": "flag", "p": 0.5}
    }
    return artifact


def test_fit_artifact_with_a_discrete_prior_is_refused():
    """Restoring one would present continuous draws as a Bernoulli posterior.

    A stored fit is a posterior. HMC and NUTS cannot have produced one for a
    discrete latent, so an artifact claiming they did is describing a fit that
    does not exist, and its draws are not samples from the prior it names.
    """
    import json

    artifact = _discrete_prior_fit_artifact()
    with pytest.raises(ValueError, match="cannot be sampled with HMC/NUTS"):
        rustmc.FitResult.from_json(json.dumps(artifact))


def test_model_artifact_with_a_discrete_prior_is_refused():
    """`CompiledModel.from_json` must agree with `ModelBuilder.compile()`.

    Both produce the same object, whose whole surface is sampling, so a
    definition one refuses cannot be one the other accepts.
    """
    import json

    artifact = _discrete_prior_fit_artifact()
    model = {
        "format": "rustmc.graph-model",
        "version": 1,
        "definition": artifact["model"]["definition"],
        "schema": artifact["model"]["schema"],
    }
    with pytest.raises(ValueError, match="cannot be sampled with HMC/NUTS"):
        rustmc.CompiledModel.from_json(json.dumps(model))

    builder = rustmc.ModelBuilder()
    builder.bernoulli_prior("flag", 0.5)
    with pytest.raises(ValueError, match="cannot be sampled with HMC/NUTS"):
        builder.compile()


def test_continuous_fit_artifacts_still_round_trip():
    """The guard must not cost an ordinary fit its restore path."""
    n = 8
    data = {"x": np.linspace(-1.0, 1.0, n), "y": np.zeros(n)}
    builder = rustmc.ModelBuilder(data)
    flag = builder.normal_prior("flag", 0.0, 1.0)
    builder.normal_likelihood("obs", flag * "x", 1.0, "y")
    fit = rustmc.sample(
        builder.build(), chains=1, draws=6, warmup=6, seed=2, show_progress=False
    )
    restored = rustmc.FitResult.from_json(fit.to_json())
    np.testing.assert_array_equal(
        restored.get_samples()["flag"], fit.get_samples()["flag"]
    )
    rustmc.CompiledModel.from_json(builder.compile().to_json())
