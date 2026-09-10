# Custom graph models

`ModelBuilder` expressions compile to the native differentiable graph. Parameters,
constants, and data expressions support `+`, `-`, `*`, `/`, power, negation,
`exp()`, `log()`, `sqrt()`, `sigmoid()`, `tanh()`, `softplus()`, `sin()`, and
`cos()`. Vector arithmetic is elementwise, scalars broadcast, and `sum()` reduces
a vector to a scalar. Mixing references from different builders is rejected.
The existing `beta * "x"` and `beta @ "X"` forms retain their fused linear and
matrix multiplication kernels.

Declare population dimensions using `ModelBuilder(..., dims={data_key: dimension})`.
Unspecified keys use the compatibility dimension `"obs"`. Each dimension has an
independent row count, validated whenever data is bound. Equal row counts do not
make distinct named dimensions interchangeable. `m.data("x", dim="customers")`
also declares a data expression's dimension. Parameter indexing uses a data key,
for example `z["group"]`; indices must be finite integers within the vector
parameter's declared bounds.

```python
import numpy as np
import rustmc

observed = {
    "group": np.array([0, 1, 0]),
    "x": np.array([1., 2., 3.]),
    "amount": np.array([2., 4., 3.]),
    "occurred": np.array([0., 1., 0., 1., 1.]),
}
m = rustmc.ModelBuilder(observed, dims={
    "group": "severity", "x": "severity", "amount": "severity",
    "occurred": "occurrence",
})
mu = m.normal_prior("mu", 0., 2.)
tau = m.half_normal_prior("tau", 1.)
z = m.vector_normal_prior("z", 2, 0., 1.)
beta = m.normal_prior("beta", 0., 1.)
predictor = mu + tau*z["group"] + beta*m.data("x")
m.normal_likelihood("severity_response", predictor, 1., "amount")
m.bernoulli_logit_likelihood("occurrence_response", mu, "occurred")
m.deterministic("expected_amount", predictor)
compiled = m.compile()
fit = compiled.sample(observed, chains=2, draws=500, warmup=500,
                      show_progress=False)

future = {"group": np.array([1, 0]), "x": np.array([4., 5.])}
expected = fit.predict(future, expected=True, sizes={"occurrence": 4})
realized = fit.predict(future, seed=123, sizes={"occurrence": 4})
```

`predict` returns named arrays with `(chain, draw, observation)` axes. Expected
responses condition on each parameter draw; realized responses additionally draw
observation noise. Future predictors determine their dimensions' sizes; `sizes`
sets dimensions with no future predictors. No response arrays are required.
`posterior_predictive(data=...)` retains the legacy flattened `(sample, observation)`
shape and optional posterior subsampling. Generic prediction applies a fitted graph
to new data; recursive state propagation belongs to the structural forecasting API.

`fit.deterministics()` returns scalar `(chain, draw)` or vector
`(chain, draw, observation)` arrays. It also accepts future `data` and `sizes`.
Parameter, response, and deterministic output names must be unique, including
expanded vector parameter names. `fit.metadata` reports kernel, axes, chain/draw
counts, and observation dimension sizes.

Use `m.potential("name", scalar_expression)` to add a supported native log-density
term. Reduce vector contributions explicitly with `.sum()`. For example,
`m.potential("penalty", -0.1*(beta**4))` changes the target density. A potential
supplies no random generator, so `sample_prior_predictive` rejects models containing
potentials instead of silently ignoring them. The ordinary prior predictive API
includes named deterministics for models without potentials.

`compiled.log_density(data, position)` returns the scalar target and its gradient;
`position` uses unconstrained coordinates in `compiled.param_names` order. This is
useful for independent finite-difference checks. User-defined native Rust targets
can use the separate `LogDensity` interface.

`compiled.to_json()` produces the `rustmc.graph-model` version 1 declarative
artifact. `rustmc.CompiledModel.from_json(text)` recompiles and validates it. The
artifact stores dimensions, matrix widths, priors, likelihoods, potentials, and
deterministics; training payloads and binding defaults are excluded. A restored
model must be bound to data before fitting. It is separate from the older Rust
`CompiledModelArtifact` JSON format, whose public imports remain available.

## Saving a fitted graph model

`fit.to_json()` creates a `rustmc.graph-fit` version 1 artifact, and
`rustmc.FitResult.from_json(text)` restores it. This fitted artifact includes the
compiled declarative model, keyed training data, every stored chain/draw position,
parameter names/order, and sampler telemetry. Unlike a compiled-model artifact,
it contains observed data. `fit.model` provides the compiled model with those
training defaults, so it can be bound and fitted again independently.

```python
from pathlib import Path

Path("fit.json").write_text(fit.to_json())
restored = rustmc.FitResult.from_json(Path("fit.json").read_text())
replayed = restored.predict(future, seed=123, sizes={"occurrence": 4})
```

Given identical future inputs and prediction seed, restored predictions reproduce
the original arrays exactly. Stored positions use the native sampler result's
**constrained graph parameter** coordinates, including latent parameters used by
noncentered priors; display variables and deterministics are reconstructed from
the model instead of being stored as independent, potentially inconsistent draws.
The sampler does not retain its original unconstrained trajectory or RNG/adaptation
state in a fit, so this format replays predictions and diagnostics rather than
resuming an interrupted chain.

Loading recompiles the declarative model, binds and validates training dimensions,
checks parameter identity, position shape/support, finite target values/gradients,
and diagnostic chain/draw dimensions. Divergent transitions may contain nonfinite
energy errors; these use explicit JSON string tokens instead of nonstandard numeric
literals. Other numeric payloads must be finite. JSON loading executes no pickled
objects or user code. The version 1 compiled-model format remains unchanged.
