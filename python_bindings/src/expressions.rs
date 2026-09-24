use rustmc_core::graph::ElementwiseOp;
use rustmc_core::model::MuExpr;
// Python expression construction; compilation and evaluation live outside this module.
use crate::builder::validate_finite;
use crate::ParameterError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// A vector parameter, from `ModelBuilder.vector_normal_prior`. It has no
/// scalar value: index it by a group column or multiply it into a matrix.
#[pyclass(module = "rustmc")]
#[derive(Debug, Clone)]
pub(super) struct VectorParamRef {
    pub(super) name: String,
    /// Id of the `ModelBuilder` that created this reference.
    pub(super) owner: u64,
}

#[pymethods]
impl VectorParamRef {
    /// `beta['group']` selects one element per observation, keyed by the
    /// integer-valued data column `group`.
    fn __getitem__(&self, data_key: &str) -> Expr {
        Expr {
            inner: MuExpr::Gather {
                param_name: self.name.clone(),
                data_key: data_key.into(),
            },
            owner: Some(self.owner),
        }
    }
    fn __matmul__(&self, data_key: &str) -> Expr {
        Expr {
            inner: MuExpr::MatVec {
                param_name: self.name.clone(),
                data_key: data_key.to_string(),
            },
            owner: Some(self.owner),
        }
    }

    // The arithmetic dunders below exist only to *reject* an operand this DSL
    // would otherwise have to guess at. A vector parameter is not a scalar
    // expression, and `MuExpr` has no variant standing for "the whole vector",
    // so there is nothing sound they could build. Without them Python reports
    // `unsupported operand type(s)` -- or, for `"x" * vec`, the baffling
    // "can't multiply sequence by non-int" -- which names the wrong problem.
    //
    // They deliberately return `NotImplemented` for anything that is *not* a
    // DSL operand, so Python's reflected-operator protocol still runs and a
    // third-party type (a NumPy array, say) keeps whatever behaviour it had
    // before these methods existed.
    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        self.reject(other)
    }
    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        self.reject(other)
    }
    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        self.reject(other)
    }
    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        self.reject(other)
    }
    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        self.reject(other)
    }
    fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        self.reject(other)
    }
    fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        self.reject(other)
    }
    fn __rtruediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        self.reject(other)
    }
    fn __pow__(
        &self,
        other: &Bound<'_, PyAny>,
        _modulo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        self.reject(other)
    }
    fn __rpow__(
        &self,
        other: &Bound<'_, PyAny>,
        _modulo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        self.reject(other)
    }
    fn __neg__(&self) -> PyResult<Expr> {
        Err(vector_param_arithmetic_error(&self.name))
    }
}

impl VectorParamRef {
    /// Raise the explanatory error for a DSL operand; defer to Python's
    /// reflected-operator protocol for everything else.
    fn reject(&self, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        if is_dsl_operand(other) {
            Err(vector_param_arithmetic_error(&self.name))
        } else {
            Ok(other.py().NotImplemented())
        }
    }
}

/// Is this something the expression DSL would otherwise have accepted?
fn is_dsl_operand(value: &Bound<'_, PyAny>) -> bool {
    value.is_instance_of::<pyo3::types::PyString>()
        || value.is_instance_of::<pyo3::types::PyFloat>()
        || value.is_instance_of::<pyo3::types::PyInt>()
        || value.extract::<PyRef<'_, Expr>>().is_ok()
        || value.extract::<PyRef<'_, ParamRef>>().is_ok()
        || value.extract::<PyRef<'_, VectorParamRef>>().is_ok()
}

/// Combine the owning-model ids of two sub-expressions, rejecting mixtures.
pub(super) fn merge_owners(a: Option<u64>, b: Option<u64>, a_name: &str) -> PyResult<Option<u64>> {
    match (a, b) {
        (Some(x), Some(y)) if x != y => Err(ParameterError::new_err(format!(
            "expression mixes parameters from two different models \
             (offending parameter: '{}'). Build the whole linear predictor \
             from a single ModelBuilder.",
            a_name
        ))),
        (Some(x), _) => Ok(Some(x)),
        (None, other) => Ok(other),
    }
}

/// First parameter name appearing in an expression, for error messages.
pub(super) fn first_param_name(expr: &MuExpr) -> String {
    match expr {
        MuExpr::Const(_) | MuExpr::Data(_) => "<constant>".to_string(),
        MuExpr::Unary(_, a) | MuExpr::Sum(a) => first_param_name(a),
        MuExpr::Param(name) => name.clone(),
        MuExpr::ParamTimesData { param_name, .. }
        | MuExpr::MatVec { param_name, .. }
        | MuExpr::Gather { param_name, .. } => param_name.clone(),
        MuExpr::Add(a, b) | MuExpr::Binary(_, a, b) => {
            let left = first_param_name(a);
            if left == "<constant>" {
                first_param_name(b)
            } else {
                left
            }
        }
    }
}

/// A scalar parameter, as returned by a `ModelBuilder` prior.
#[pyclass(module = "rustmc")]
#[derive(Debug, Clone)]
pub(super) struct ParamRef {
    pub(super) name: String,
    /// Id of the `ModelBuilder` that created this reference.
    pub(super) owner: u64,
}

/// Recursive expression tree built on the Python side, compiled to graph
/// nodes at sampling time.
#[pyclass(module = "rustmc")]
#[derive(Debug, Clone)]
pub(super) struct Expr {
    pub(super) inner: MuExpr,
    /// Id of the `ModelBuilder` whose parameters this expression uses, if any.
    /// `None` for constant-only expressions.
    pub(super) owner: Option<u64>,
}

/// A bare `"x"` builds the same `MuExpr::Data("x")` node that
/// `ModelBuilder.data("x")` returns. It differs in two deliberate ways.
///
/// *Owner.* A string is not tied to any `ModelBuilder`, so `owner` is `None`,
/// exactly as for a numeric constant; `ModelBuilder.data()` records
/// `Some(builder_id)` because it may also register a dimension on that
/// builder. `merge_owners` lets a `None` operand adopt the other side's owner,
/// so an expression that mixes two builders' *parameters* is still rejected,
/// and `ModelBuilder::check_owner` still rejects any expression carrying a
/// foreign owner. The only expressions that stay unowned are those with no
/// parameters at all -- pure data and constants -- which carry no
/// builder-specific state and mean the same thing in any model. The
/// consequence to know: `second.deterministic("d", "x")` is accepted where
/// `second.deterministic("d", first.data("x"))` is rejected.
///
/// *Dimension.* A key introduced this way registers no dimension override.
/// `ModelBuilder.data(name, dim)` only writes into `ModelBuilder::dimensions`,
/// which `model.rs` applies as a post-hoc rename over schema slots that
/// otherwise default to `"obs"`. A bare `"x"` is therefore exactly
/// `builder.data("x")` with `dim=None`; to name a non-`obs` dimension, keep
/// using `builder.data("x", "dim")`.
fn data_expr(data_key: String) -> Expr {
    Expr {
        inner: MuExpr::Data(data_key),
        owner: None,
    }
}

pub(super) fn extract_expr(value: &Bound<'_, PyAny>) -> PyResult<Expr> {
    if let Ok(e) = value.extract::<PyRef<'_, Expr>>() {
        Ok(e.clone())
    } else if let Ok(p) = value.extract::<PyRef<'_, ParamRef>>() {
        Ok(p.as_expr())
    } else if let Ok(v) = value.extract::<PyRef<'_, VectorParamRef>>() {
        Err(vector_param_arithmetic_error(&v.name))
    } else if let Ok(data_key) = value.extract::<String>() {
        // Checked before `f64` so that a string is never coerced to a number.
        Ok(data_expr(data_key))
    } else if let Ok(x) = value.extract::<f64>() {
        validate_finite("expression constant", x)?;
        Ok(Expr {
            inner: MuExpr::Const(x),
            owner: None,
        })
    } else {
        Err(PyValueError::new_err(
            "expected a numeric constant, a parameter, an expression, or a \
             data key string (e.g. beta * 'x')",
        ))
    }
}

/// A whole vector parameter has no scalar value, so it cannot take part in
/// element-wise arithmetic. Name that, rather than claiming it is not an
/// expression.
fn vector_param_arithmetic_error(name: &str) -> PyErr {
    ParameterError::new_err(format!(
        "vector parameter '{name}' has no single value, so it cannot be used \
         directly in arithmetic. Select one element per observation with \
         {name}['group_key'], or take a matrix-vector product with \
         {name} @ 'matrix_key'."
    ))
}
impl ParamRef {
    fn as_expr(&self) -> Expr {
        Expr {
            inner: MuExpr::Param(self.name.clone()),
            owner: Some(self.owner),
        }
    }
}
impl Expr {
    fn as_expr(&self) -> Expr {
        self.clone()
    }
    fn unary(&self, op: ElementwiseOp) -> Expr {
        Expr {
            inner: MuExpr::Unary(op, Box::new(self.inner.clone())),
            owner: self.owner,
        }
    }
    fn binary(&self, other: &Bound<'_, PyAny>, op: ElementwiseOp, reverse: bool) -> PyResult<Expr> {
        let rhs = extract_expr(other)?;
        let owner = merge_owners(self.owner, rhs.owner, &first_param_name(&self.inner))?;
        let (a, b) = if reverse {
            (rhs.inner, self.inner.clone())
        } else {
            (self.inner.clone(), rhs.inner)
        };
        let inner = if matches!(op, ElementwiseOp::Add) {
            MuExpr::Add(Box::new(a), Box::new(b))
        } else {
            MuExpr::Binary(op, Box::new(a), Box::new(b))
        };
        Ok(Expr { inner, owner })
    }
    fn power(
        &self,
        other: &Bound<'_, PyAny>,
        modulo: Option<&Bound<'_, PyAny>>,
        reverse: bool,
    ) -> PyResult<Expr> {
        if modulo.is_some() {
            return Err(PyValueError::new_err("modular power is unsupported"));
        }
        self.binary(other, ElementwiseOp::Pow, reverse)
    }
}

/// The Python methods a scalar operand shares -- `ParamRef` and `Expr` alike:
/// arithmetic, the elementwise functions and `sum`. Each type supplies its
/// own `__mul__` (a `ParamRef` times a data key stays a fused
/// `ParamTimesData`) and `as_expr`, and any methods only it has.
macro_rules! scalar_expression_methods {
    ($ty:ty { $($own:tt)* }) => {
        #[pymethods]
        impl $ty {
            $($own)*
            fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Expr> {
                self.__mul__(other)
            }
            fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<Expr> {
                self.as_expr().binary(other, ElementwiseOp::Add, false)
            }
            fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<Expr> {
                self.__add__(other)
            }
            fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Expr> {
                self.as_expr().binary(other, ElementwiseOp::Sub, false)
            }
            fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Expr> {
                self.as_expr().binary(other, ElementwiseOp::Sub, true)
            }
            fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Expr> {
                self.as_expr().binary(other, ElementwiseOp::Div, false)
            }
            fn __rtruediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Expr> {
                self.as_expr().binary(other, ElementwiseOp::Div, true)
            }
            fn __pow__(
                &self,
                other: &Bound<'_, PyAny>,
                modulo: Option<&Bound<'_, PyAny>>,
            ) -> PyResult<Expr> {
                self.as_expr().power(other, modulo, false)
            }
            fn __rpow__(
                &self,
                other: &Bound<'_, PyAny>,
                modulo: Option<&Bound<'_, PyAny>>,
            ) -> PyResult<Expr> {
                self.as_expr().power(other, modulo, true)
            }
            fn __neg__(&self) -> Expr {
                self.as_expr().unary(ElementwiseOp::Neg)
            }
            fn exp(&self) -> Expr {
                self.as_expr().unary(ElementwiseOp::Exp)
            }
            fn log(&self) -> Expr {
                self.as_expr().unary(ElementwiseOp::Log)
            }
            fn sqrt(&self) -> Expr {
                self.as_expr().unary(ElementwiseOp::Sqrt)
            }
            fn sigmoid(&self) -> Expr {
                self.as_expr().unary(ElementwiseOp::Sigmoid)
            }
            fn tanh(&self) -> Expr {
                self.as_expr().unary(ElementwiseOp::Tanh)
            }
            fn softplus(&self) -> Expr {
                self.as_expr().unary(ElementwiseOp::Softplus)
            }
            fn sin(&self) -> Expr {
                self.as_expr().unary(ElementwiseOp::Sin)
            }
            fn cos(&self) -> Expr {
                self.as_expr().unary(ElementwiseOp::Cos)
            }
            fn sum(&self) -> Expr {
                let expr = self.as_expr();
                Expr {
                    inner: MuExpr::Sum(Box::new(expr.inner)),
                    owner: expr.owner,
                }
            }
        }
    };
}

scalar_expression_methods!(ParamRef {
    /// `beta * "x"` keeps its fused `ParamTimesData` form, which
    /// `model.rs::try_extract_linear` compiles into a single `FusedLinearMu`
    /// op. Every other operand falls through to the generic expression path.
    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Expr> {
        if let Ok(data_key) = other.extract::<String>() {
            Ok(Expr {
                inner: MuExpr::ParamTimesData {
                    param_name: self.name.clone(),
                    data_key,
                },
                owner: Some(self.owner),
            })
        } else {
            self.as_expr().binary(other, ElementwiseOp::Mul, false)
        }
    }
    fn __matmul__(&self, data_key: &str) -> Expr {
        Expr {
            inner: MuExpr::MatVec {
                param_name: self.name.clone(),
                data_key: data_key.into(),
            },
            owner: Some(self.owner),
        }
    }
});

scalar_expression_methods!(Expr {
    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Expr> {
        self.binary(other, ElementwiseOp::Mul, false)
    }
});
