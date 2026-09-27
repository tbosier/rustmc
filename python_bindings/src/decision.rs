//! Python bindings for decisions from predictive draws.
//!
//! The losses, averages, quantiles and standard errors are all computed in
//! `rustmc_core::decision`. This module converts the draws and the loss, and
//! calls a Python loss function when one is given.
//!
//! Errors follow the forecasting rule in `forecast_support`: refused
//! arguments raise `ValueError`, a loss or total that leaves the
//! floating-point range raises `InferenceError`, and an exception raised by a
//! user's loss function propagates unchanged.
use crate::data_input::real_numbers;
use crate::forecast_support::real_vector;
use crate::InferenceError;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyInt, PyString};
use rustmc_core::decision::{
    expected_loss as core_expected_loss, optimal_decision, Decision as CoreDecision, DecisionDraws,
    DecisionError, DecisionMethod, ExpectedLoss as CoreExpectedLoss, Loss, LossFunction,
    PiecewiseLinearLoss as CorePiecewiseLinearLoss, Search,
};
use std::cell::RefCell;

/// The Python exception for a core refusal or failure. An exception the
/// user's loss function raised takes precedence over the core's report of it.
fn decision_error(error: DecisionError, raised: Option<PyErr>) -> PyErr {
    if let Some(raised) = raised {
        return raised;
    }
    match error {
        DecisionError::NumericalFailure(message) => InferenceError::new_err(message),
        other => PyValueError::new_err(other.to_string()),
    }
}

/// One real number, by the exact conversion every array argument uses.
fn real_scalar(value: &Bound<'_, PyAny>, name: &str) -> PyResult<f64> {
    let (values, shape) = real_numbers(value, name)?;
    if !shape.is_empty() {
        return Err(PyValueError::new_err(format!(
            "{name} must be a single number; got an array of shape {}",
            shape_text(&shape)
        )));
    }
    Ok(values[0])
}

fn shape_text(shape: &[usize]) -> String {
    match shape {
        [single] => format!("({single},)"),
        _ => format!(
            "({})",
            shape
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// Pinball loss: `under * (y - a)` when the outcome `y` exceeds the decision
/// `a`, `over * (a - y)` otherwise. The expected loss is smallest at the
/// `under / (under + over)` quantile of the outcome.
#[pyclass(name = "AsymmetricLinearLoss", frozen, module = "rustmc")]
#[derive(Clone)]
pub(crate) struct PyAsymmetricLinearLoss {
    under: f64,
    over: f64,
}

#[pymethods]
impl PyAsymmetricLinearLoss {
    /// `under` is the cost per unit the outcome exceeds the decision, `over`
    /// the cost per unit the decision exceeds the outcome. Both must be
    /// finite and nonnegative, and not both zero.
    #[new]
    #[pyo3(signature = (*, under, over))]
    fn new(under: &Bound<'_, PyAny>, over: &Bound<'_, PyAny>) -> PyResult<Self> {
        let (under, over) = (real_scalar(under, "under")?, real_scalar(over, "over")?);
        Loss::asymmetric_linear(under, over).map_err(|error| decision_error(error, None))?;
        Ok(Self { under, over })
    }

    #[getter]
    fn under(&self) -> f64 {
        self.under
    }

    #[getter]
    fn over(&self) -> f64 {
        self.over
    }

    /// The probability `under / (under + over)` whose quantile minimises the
    /// expected loss.
    #[getter]
    fn quantile(&self) -> f64 {
        match self.loss().optimal_rule() {
            Some(rustmc_core::decision::OptimalRule::Quantile(probability)) => probability,
            _ => unreachable!("asymmetric linear loss has a quantile rule"),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "AsymmetricLinearLoss(under={}, over={})",
            self.under, self.over
        )
    }
}

impl PyAsymmetricLinearLoss {
    fn loss(&self) -> Loss {
        Loss::AsymmetricLinear {
            under: self.under,
            over: self.over,
        }
    }
}

/// Squared loss `(y - a)^2`, smallest in expectation at the mean.
#[pyclass(name = "SquaredLoss", frozen, module = "rustmc")]
pub(crate) struct PySquaredLoss;

#[pymethods]
impl PySquaredLoss {
    #[new]
    fn new() -> Self {
        Self
    }

    fn __repr__(&self) -> &'static str {
        "SquaredLoss()"
    }
}

/// Absolute loss `|y - a|`, smallest in expectation at the median.
#[pyclass(name = "AbsoluteLoss", frozen, module = "rustmc")]
pub(crate) struct PyAbsoluteLoss;

#[pymethods]
impl PyAbsoluteLoss {
    #[new]
    fn new() -> Self {
        Self
    }

    fn __repr__(&self) -> &'static str {
        "AbsoluteLoss()"
    }
}

/// A loss linear between knots in the shortfall `y - a`, continued along its
/// first and last segments beyond them. rustmc uses no formula for its
/// optimum, so `decide` searches the candidate grid.
#[pyclass(name = "PiecewiseLinearLoss", frozen, module = "rustmc")]
pub(crate) struct PyPiecewiseLinearLoss {
    inner: CorePiecewiseLinearLoss,
}

#[pymethods]
impl PyPiecewiseLinearLoss {
    /// `shortfall` holds strictly increasing knots of `outcome - decision`
    /// (positive when the decision falls short), and `loss` the loss at each.
    #[new]
    fn new(shortfall: &Bound<'_, PyAny>, loss: &Bound<'_, PyAny>) -> PyResult<Self> {
        let inner = CorePiecewiseLinearLoss::new(
            real_vector(shortfall, "shortfall")?,
            real_vector(loss, "loss")?,
        )
        .map_err(|error| decision_error(error, None))?;
        Ok(Self { inner })
    }

    #[getter]
    fn shortfall<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.shortfall().to_vec().into_pyarray(py)
    }

    #[getter]
    fn loss<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.loss().to_vec().into_pyarray(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "PiecewiseLinearLoss(shortfall={:?}, loss={:?})",
            self.inner.shortfall(),
            self.inner.loss()
        )
    }
}

/// A Python `loss(decision, outcomes)` returning one loss per draw.
struct CallableLoss<'py> {
    function: Bound<'py, PyAny>,
    /// The draws, shaped `(chain, draw)` and read-only, passed on every call.
    outcomes: Bound<'py, PyArray2<f64>>,
    /// The first exception the function raised, re-raised as it was.
    raised: RefCell<Option<PyErr>>,
}

impl<'py> CallableLoss<'py> {
    fn new(function: Bound<'py, PyAny>, draws: &DecisionDraws) -> PyResult<Self> {
        let py = function.py();
        let outcomes = PyArray2::from_vec2(py, draws.chains())?;
        let read_only = PyDict::new(py);
        read_only.set_item("write", false)?;
        outcomes.call_method("setflags", (), Some(&read_only))?;
        Ok(Self {
            function,
            outcomes,
            raised: RefCell::new(None),
        })
    }
}

impl LossFunction for CallableLoss<'_> {
    fn losses(&self, decision: f64, draws: &DecisionDraws) -> Result<Vec<Vec<f64>>, DecisionError> {
        let returned = self
            .function
            .call1((decision, &self.outcomes))
            .and_then(|value| real_numbers(&value, "the loss function's return value"));
        match returned {
            Ok((values, shape)) => {
                let (chains, per_chain) = (draws.chain_count(), draws.draws_per_chain());
                if shape != [chains, per_chain] {
                    return Err(DecisionError::LossFunction(format!(
                        "the loss function must return one loss per draw, shaped like \
                         outcomes ({chains}, {per_chain}); got shape {}",
                        shape_text(&shape)
                    )));
                }
                Ok(values.chunks(per_chain).map(<[f64]>::to_vec).collect())
            }
            Err(error) => {
                self.raised.borrow_mut().get_or_insert(error);
                Err(DecisionError::LossFunction(
                    "the loss function raised an exception".into(),
                ))
            }
        }
    }
}

enum ResolvedLoss<'py> {
    Builtin(Loss),
    Callable(Bound<'py, PyAny>),
}

fn resolve_loss<'py>(loss: &Bound<'py, PyAny>) -> PyResult<ResolvedLoss<'py>> {
    if let Ok(asymmetric) = loss.downcast::<PyAsymmetricLinearLoss>() {
        return Ok(ResolvedLoss::Builtin(asymmetric.get().loss()));
    }
    if loss.is_instance_of::<PySquaredLoss>() {
        return Ok(ResolvedLoss::Builtin(Loss::Squared));
    }
    if loss.is_instance_of::<PyAbsoluteLoss>() {
        return Ok(ResolvedLoss::Builtin(Loss::Absolute));
    }
    if let Ok(piecewise) = loss.downcast::<PyPiecewiseLinearLoss>() {
        return Ok(ResolvedLoss::Builtin(Loss::PiecewiseLinear(
            piecewise.get().inner.clone(),
        )));
    }
    if loss.is_callable() {
        return Ok(ResolvedLoss::Callable(loss.clone()));
    }
    Err(PyValueError::new_err(format!(
        "loss must be an AsymmetricLinearLoss, SquaredLoss, AbsoluteLoss or \
         PiecewiseLinearLoss, or a callable loss(decision, outcomes); got {}",
        loss.get_type().name()?
    )))
}

/// Zero-based step indices from a sequence of integers.
fn step_indices(value: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    let not_a_sequence = || {
        PyValueError::new_err(
            "steps must be a sequence of step indices numbered from 0, such as [0] or range(3)",
        )
    };
    if value.is_instance_of::<PyString>() || value.is_instance_of::<PyInt>() {
        return Err(not_a_sequence());
    }
    let mut steps = Vec::new();
    for item in value.try_iter().map_err(|_| not_a_sequence())? {
        let item = item?;
        let not_an_integer = || {
            PyValueError::new_err(format!(
                "steps must hold integers; got {}",
                item.repr()
                    .map_or_else(|_| "an unprintable value".into(), |r| r.to_string())
            ))
        };
        if item.is_instance_of::<PyBool>() {
            return Err(not_an_integer());
        }
        let index = item
            .call_method0("__index__")
            .map_err(|_| not_an_integer())?;
        let index: i64 = index.extract().map_err(|_| {
            PyValueError::new_err(format!("steps holds {index}, which is not a step index"))
        })?;
        let index = usize::try_from(index).map_err(|_| {
            PyValueError::new_err(format!(
                "steps must be numbered from 0; got {index} (negative indices are not accepted)"
            ))
        })?;
        steps.push(index);
    }
    Ok(steps)
}

/// The outcome draws a decision is made on.
///
/// `draws` is an array - `(draw,)`, `(chain, draw)` or `(chain, draw,
/// horizon)` - or any object with an `observation_samples` array, such as a
/// native forecast or a `ForecastDraws`. With a horizon, the outcome is the
/// total over `steps` within each path.
fn decision_draws(
    draws: &Bound<'_, PyAny>,
    steps: Option<&Bound<'_, PyAny>>,
) -> PyResult<DecisionDraws> {
    let source = if draws.hasattr("observation_samples")? {
        draws.getattr("observation_samples")?
    } else {
        draws.clone()
    };
    let (values, shape) = real_numbers(&source, "draws")?;
    let steps = steps.map(step_indices).transpose()?;
    if shape.is_empty() {
        return Err(PyValueError::new_err(
            "draws must be an array of draws, not a single number",
        ));
    }
    if values.is_empty() {
        return Err(PyValueError::new_err("draws must hold at least one draw"));
    }
    let core = |error| decision_error(error, None);
    match (shape.as_slice(), steps) {
        ([_] | [_, _], Some(_)) => Err(PyValueError::new_err(format!(
            "steps applies only to draws shaped (chain, draw, horizon); these are shaped {}",
            shape_text(&shape)
        ))),
        ([_], None) => DecisionDraws::new(vec![values]).map_err(core),
        ([_, per_chain], None) => {
            DecisionDraws::new(values.chunks(*per_chain).map(<[f64]>::to_vec).collect())
                .map_err(core)
        }
        ([_, per_chain, horizon], steps) => {
            let steps = match steps {
                Some(steps) => steps,
                None if *horizon == 1 => vec![0],
                None => {
                    return Err(PyValueError::new_err(format!(
                        "draws have a horizon of {horizon} steps; pass steps to choose which \
                         to total, for example steps=range({horizon}) for all of them"
                    )))
                }
            };
            let paths: Vec<Vec<Vec<f64>>> = values
                .chunks(per_chain * horizon)
                .map(|chain| chain.chunks(*horizon).map(<[f64]>::to_vec).collect())
                .collect();
            DecisionDraws::from_path_totals(&paths, &steps).map_err(core)
        }
        ([_, _, _, _], _) => Err(PyValueError::new_err(
            "draws with a series axis, shaped (chain, draw, series, horizon), must be reduced \
             to one outcome first: for separate series, ForecastDraws.aggregate(); for a \
             ReconciledForecast, one node with node_samples(name), since its nodes already \
             include their own totals",
        )),
        (_, _) => Err(PyValueError::new_err(format!(
            "draws must be shaped (draw,), (chain, draw) or (chain, draw, horizon); got {} axes",
            shape.len()
        ))),
    }
}

/// The expected loss of one decision and its Monte Carlo uncertainty.
#[pyclass(name = "ExpectedLoss", frozen, module = "rustmc")]
pub(crate) struct PyExpectedLoss {
    inner: CoreExpectedLoss,
}

#[pymethods]
impl PyExpectedLoss {
    #[getter]
    fn decision(&self) -> f64 {
        self.inner.decision
    }

    /// The average loss over the draws.
    #[getter]
    fn value(&self) -> f64 {
        self.inner.value
    }

    /// Monte Carlo standard error of `value`, from the effective sample size
    /// of the per-draw losses. Zero when every loss is equal, otherwise `nan`
    /// with fewer than four draws per chain.
    #[getter]
    fn mcse(&self) -> f64 {
        self.inner.mcse
    }

    /// Effective sample size of the per-draw losses, from split chains.
    #[getter]
    fn ess(&self) -> f64 {
        self.inner.ess
    }

    fn __repr__(&self) -> String {
        format!(
            "ExpectedLoss(decision={}, value={}, mcse={}, ess={})",
            self.inner.decision, self.inner.value, self.inner.mcse, self.inner.ess
        )
    }
}

/// The decision with the lowest expected loss, and the evidence behind it.
#[pyclass(name = "Decision", frozen, module = "rustmc")]
pub(crate) struct PyDecision {
    inner: CoreDecision,
    chains: usize,
}

#[pymethods]
impl PyDecision {
    #[getter]
    fn decision(&self) -> f64 {
        self.inner.decision
    }

    /// `"quantile"`, `"mean"` or `"median"` for a closed form, `"grid"` for
    /// the best candidate.
    #[getter]
    fn method(&self) -> &'static str {
        match self.inner.method {
            DecisionMethod::Quantile(_) => "quantile",
            DecisionMethod::Mean => "mean",
            DecisionMethod::Median => "median",
            DecisionMethod::Grid => "grid",
        }
    }

    /// The quantile probability when `method` is `"quantile"`, else `None`.
    #[getter]
    fn quantile(&self) -> Option<f64> {
        match self.inner.method {
            DecisionMethod::Quantile(probability) => Some(probability),
            _ => None,
        }
    }

    /// The average loss at `decision`. It is biased low: the decision was
    /// chosen from the same draws to make it small.
    #[getter]
    fn expected_loss(&self) -> f64 {
        self.inner.expected_loss.value
    }

    /// Monte Carlo standard error of `expected_loss`, treating the decision
    /// as fixed.
    #[getter]
    fn expected_loss_mcse(&self) -> f64 {
        self.inner.expected_loss.mcse
    }

    #[getter]
    fn expected_loss_ess(&self) -> f64 {
        self.inner.expected_loss.ess
    }

    #[getter]
    fn candidates<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.candidates.clone().into_pyarray(py)
    }

    /// The average loss at each candidate.
    #[getter]
    fn curve<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.curve.clone().into_pyarray(py)
    }

    /// The optimal decision recomputed on contiguous blocks of each chain,
    /// shaped `(chain, block)`.
    #[getter]
    fn batch_decisions<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let blocks = self.inner.batch_decisions.first().map_or(0, Vec::len);
        let flat: Vec<f64> = self
            .inner
            .batch_decisions
            .iter()
            .flatten()
            .copied()
            .collect();
        ndarray::Array2::from_shape_vec((self.chains, blocks), flat)
            .expect("every chain has the same number of blocks")
            .into_pyarray(py)
    }

    /// Batch-means Monte Carlo standard error of `decision`: the standard
    /// deviation of `batch_decisions` over the square root of their number.
    /// `nan` for a grid search and for a quantile at 0 or 1, where that
    /// formula does not hold, and without batches.
    #[getter]
    fn decision_mcse(&self) -> f64 {
        self.inner.decision_mcse
    }

    /// True when a grid search chose the first or last candidate (or the
    /// only one), so the best decision may lie outside the grid.
    #[getter]
    fn on_grid_edge(&self) -> bool {
        self.inner.on_grid_edge
    }

    fn __repr__(&self) -> String {
        let method = match self.inner.method {
            DecisionMethod::Quantile(probability) => format!("quantile {probability}"),
            _ => self.method().to_string(),
        };
        format!(
            "Decision(decision={}, method='{method}', expected_loss={}, expected_loss_mcse={}, \
             decision_mcse={})",
            self.inner.decision,
            self.inner.expected_loss.value,
            self.inner.expected_loss.mcse,
            self.inner.decision_mcse
        )
    }
}

/// The decision with the lowest expected loss, estimated from predictive
/// draws.
///
/// `draws` is an array shaped `(draw,)`, `(chain, draw)` or `(chain, draw,
/// horizon)`, or a forecast with `observation_samples`. With a horizon the
/// outcome is the total over `steps` within each path. `loss` is a built-in
/// loss or a callable `loss(decision, outcomes)` returning one loss per draw,
/// each depending only on the decision and that draw. `candidates` are where
/// the curve is evaluated, up to 201 points from the smallest to the largest
/// draw by default. `method="auto"` uses the quantile, mean or median for the
/// built-in losses that have one; `method="grid"` always takes the best
/// candidate. The quantile interpolates between draws, as every rustmc
/// interval does, so it can sit up to one gap between draws from the exact
/// minimiser of the average loss.
#[pyfunction]
#[pyo3(signature = (draws, loss, *, steps=None, candidates=None, method="auto"))]
fn decide(
    py: Python<'_>,
    draws: &Bound<'_, PyAny>,
    loss: &Bound<'_, PyAny>,
    steps: Option<&Bound<'_, PyAny>>,
    candidates: Option<&Bound<'_, PyAny>>,
    method: &str,
) -> PyResult<PyDecision> {
    let search = match method {
        "auto" => Search::Auto,
        "grid" => Search::Grid,
        other => {
            return Err(PyValueError::new_err(format!(
                "method must be 'auto' or 'grid'; got {other:?}"
            )))
        }
    };
    let draws = decision_draws(draws, steps)?;
    let candidates = candidates
        .map(|candidates| real_vector(candidates, "candidates"))
        .transpose()?;
    let inner = match resolve_loss(loss)? {
        ResolvedLoss::Builtin(loss) => py
            .allow_threads(|| optimal_decision(&draws, &loss, candidates.as_deref(), search))
            .map_err(|error| decision_error(error, None))?,
        ResolvedLoss::Callable(function) => {
            let loss = CallableLoss::new(function, &draws)?;
            optimal_decision(&draws, &loss, candidates.as_deref(), search)
                .map_err(|error| decision_error(error, loss.raised.take()))?
        }
    };
    Ok(PyDecision {
        inner,
        chains: draws.chain_count(),
    })
}

/// The expected loss of `decision`, with its Monte Carlo standard error.
///
/// `draws`, `loss` and `steps` are as for `decide`.
#[pyfunction]
#[pyo3(signature = (draws, decision, loss, *, steps=None))]
fn expected_loss(
    py: Python<'_>,
    draws: &Bound<'_, PyAny>,
    decision: &Bound<'_, PyAny>,
    loss: &Bound<'_, PyAny>,
    steps: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyExpectedLoss> {
    let decision = real_scalar(decision, "decision")?;
    let draws = decision_draws(draws, steps)?;
    let inner = match resolve_loss(loss)? {
        ResolvedLoss::Builtin(loss) => py
            .allow_threads(|| core_expected_loss(&draws, &loss, decision))
            .map_err(|error| decision_error(error, None))?,
        ResolvedLoss::Callable(function) => {
            let loss = CallableLoss::new(function, &draws)?;
            core_expected_loss(&draws, &loss, decision)
                .map_err(|error| decision_error(error, loss.raised.take()))?
        }
    };
    Ok(PyExpectedLoss { inner })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAsymmetricLinearLoss>()?;
    m.add_class::<PySquaredLoss>()?;
    m.add_class::<PyAbsoluteLoss>()?;
    m.add_class::<PyPiecewiseLinearLoss>()?;
    m.add_class::<PyExpectedLoss>()?;
    m.add_class::<PyDecision>()?;
    m.add_function(wrap_pyfunction!(decide, m)?)?;
    m.add_function(wrap_pyfunction!(expected_loss, m)?)?;
    Ok(())
}
