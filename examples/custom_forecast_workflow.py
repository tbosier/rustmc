"""Run with `python examples/custom_forecast_workflow.py` after installing rustmc.

Demonstrates prediction on new data and a dynamic structural forecast. Increase
draws/warmup for your model and inspect diagnostics before comparing forecasts.
"""
import numpy as np
import rustmc as r


def main():
    rng = np.random.default_rng(37)
    x = np.linspace(-1, 1, 36)
    holiday = (np.arange(36) % 6 == 0).astype(float)
    y = 10 + .8*x + 1.5*holiday + rng.normal(0, .4, 36)

    builder = r.ModelBuilder()
    intercept = builder.normal_prior("intercept", 10., 2.)
    effect = builder.normal_prior("effect", 0., 1.)
    calendar = builder.normal_prior("calendar", 0., 2.)
    mu = intercept + effect * "x" + calendar * "holiday"
    builder.deterministic("expected_sales", mu)
    builder.normal_likelihood("sales", mu, .4, "y")
    compiled = builder.compile()
    fit = compiled.sample({"x": x, "holiday": holiday, "y": y},
                          chains=4, draws=1000, warmup=1000, show_progress=False)
    future = {"x": np.linspace(1, 1.2, 6), "holiday": np.zeros(6)}
    prediction = fit.predict(future, seed=20)["sales"]
    print("Custom regression prediction (chain, draw, horizon):", prediction.shape)
    print(fit.summary())
    restored = r.FitResult.from_json(fit.to_json())
    np.testing.assert_array_equal(prediction, restored.predict(future, seed=20)["sales"])

    C, V = r.StructuralComponent, r.VarianceParameter
    structural = r.StructuralModel([
        C.level("baseline", V.inverse_gamma(4, .02), initial_mean=10., initial_variance=2.),
        C.regression("effects", initial_mean=[0., 0.],
                     initial_covariance=[[1., 0.], [0., 4.]],
                     innovations=[V.inverse_gamma(4, .001), V.fixed(0.)]),
    ], observation_variance=V.inverse_gamma(4, .4), student_df=5.)
    features = ("x", "holiday")
    design = r.NamedDesign(np.column_stack([x, holiday]), features)
    session = r.ForecastSession(structural, y, exog=design,
                                fit_kwargs={"chains": 4, "draws": 1000, "warmup": 1000})
    scenarios = r.forecast_scenarios(session, {
        "ordinary": r.NamedDesign(np.column_stack([future["x"], np.zeros(6)]), features),
        "holiday": r.NamedDesign(np.column_stack([future["x"], np.ones(6)]), features),
    }, probabilities={"ordinary": .8, "holiday": .2})
    print("Scenario mixture 90% interval:", scenarios.mixture(seed=31).interval(.9))
    comparison = r.backtest(structural, y, exog=design.values, horizon=3, origins=[24, 30],
                            fit_kwargs={"chains": 4, "draws": 1000, "warmup": 1000})
    print("Backtest CRPS by horizon:", comparison.summary()["crps"])
    print("Baseline CRPS by horizon:", comparison.summary(baseline=True)["crps"])


if __name__ == "__main__":
    main()
