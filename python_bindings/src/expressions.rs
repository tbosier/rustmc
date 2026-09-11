use rustmc_core::model::MuExpr;
// Python expression construction; compilation and evaluation live outside this module.
use super::{validate_finite, ParameterError};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Recursive expression tree built on the Python side, compiled to graph
/// nodes at sampling time.

#[pyclass]
#[derive(Debug, Clone)]
pub(super) struct VectorParamRef {
    pub(super) name: String,
    pub(super) _n: usize,
    /// Id of the `ModelBuilder` that created this reference.
    pub(super) owner: u64,
}

#[pymethods]
impl VectorParamRef {
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

/// Collect every parameter name referenced by an expression tree.

#[pyclass]
#[derive(Debug, Clone)]
pub(super) struct ParamRef {
    pub(super) name: String,
    /// Id of the `ModelBuilder` that created this reference.
    pub(super) owner: u64,
}

#[pyclass]
#[derive(Debug, Clone)]
pub(super) struct Expr {
    pub(super) inner: MuExpr,
    /// Id of the `ModelBuilder` whose parameters this expression uses, if any.
    /// `None` for constant-only expressions.
    pub(super) owner: Option<u64>,
}

pub(super) fn extract_expr(value: &Bound<'_, PyAny>) -> PyResult<Expr> {
    if let Ok(e) = value.extract::<PyRef<'_, Expr>>() {
        Ok(e.clone())
    } else if let Ok(p) = value.extract::<PyRef<'_, ParamRef>>() {
        Ok(p.as_expr())
    } else if let Ok(x) = value.extract::<f64>() {
        validate_finite("expression constant", x)?;
        Ok(Expr {
            inner: MuExpr::Const(x),
            owner: None,
        })
    } else {
        Err(PyValueError::new_err(
            "expected a numeric constant, parameter, or expression",
        ))
    }
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
    fn unary(&self, op: rustmc_core::graph::ElementwiseOp) -> Expr {
        Expr {
            inner: MuExpr::Unary(op, Box::new(self.inner.clone())),
            owner: self.owner,
        }
    }
    fn binary(
        &self,
        other: &Bound<'_, PyAny>,
        op: rustmc_core::graph::ElementwiseOp,
        reverse: bool,
    ) -> PyResult<Expr> {
        let rhs = extract_expr(other)?;
        let owner = merge_owners(self.owner, rhs.owner, &first_param_name(&self.inner))?;
        let (a, b) = if reverse {
            (rhs.inner, self.inner.clone())
        } else {
            (self.inner.clone(), rhs.inner)
        };
        let inner = if matches!(op, rustmc_core::graph::ElementwiseOp::Add) {
            MuExpr::Add(Box::new(a), Box::new(b))
        } else {
            MuExpr::Binary(op, Box::new(a), Box::new(b))
        };
        Ok(Expr { inner, owner })
    }
}
#[pymethods]
impl ParamRef {
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
            self.as_expr()
                .binary(other, rustmc_core::graph::ElementwiseOp::Mul, false)
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

    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<Expr> {
        self.as_expr()
            .binary(other, rustmc_core::graph::ElementwiseOp::Add, false)
    }
    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<Expr> {
        self.__add__(other)
    }
    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Expr> {
        self.as_expr()
            .binary(other, rustmc_core::graph::ElementwiseOp::Sub, false)
    }
    fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Expr> {
        self.as_expr()
            .binary(other, rustmc_core::graph::ElementwiseOp::Sub, true)
    }
    fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Expr> {
        self.as_expr()
            .binary(other, rustmc_core::graph::ElementwiseOp::Div, false)
    }
    fn __rtruediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Expr> {
        self.as_expr()
            .binary(other, rustmc_core::graph::ElementwiseOp::Div, true)
    }
    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Expr> {
        self.__mul__(other)
    }
    fn __pow__(
        &self,
        other: &Bound<'_, PyAny>,
        modulo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Expr> {
        if modulo.is_some() {
            return Err(PyValueError::new_err("modular power is unsupported"));
        }
        self.as_expr()
            .binary(other, rustmc_core::graph::ElementwiseOp::Pow, false)
    }
    fn __rpow__(
        &self,
        other: &Bound<'_, PyAny>,
        modulo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Expr> {
        if modulo.is_some() {
            return Err(PyValueError::new_err("modular power is unsupported"));
        }
        self.as_expr()
            .binary(other, rustmc_core::graph::ElementwiseOp::Pow, true)
    }
    fn __neg__(&self) -> Expr {
        self.as_expr().unary(rustmc_core::graph::ElementwiseOp::Neg)
    }
    fn exp(&self) -> Expr {
        self.as_expr().unary(rustmc_core::graph::ElementwiseOp::Exp)
    }
    fn log(&self) -> Expr {
        self.as_expr().unary(rustmc_core::graph::ElementwiseOp::Log)
    }
    fn sqrt(&self) -> Expr {
        self.as_expr()
            .unary(rustmc_core::graph::ElementwiseOp::Sqrt)
    }
    fn sigmoid(&self) -> Expr {
        self.as_expr()
            .unary(rustmc_core::graph::ElementwiseOp::Sigmoid)
    }
    fn tanh(&self) -> Expr {
        self.as_expr()
            .unary(rustmc_core::graph::ElementwiseOp::Tanh)
    }
    fn softplus(&self) -> Expr {
        self.as_expr()
            .unary(rustmc_core::graph::ElementwiseOp::Softplus)
    }
    fn sin(&self) -> Expr {
        self.as_expr().unary(rustmc_core::graph::ElementwiseOp::Sin)
    }
    fn cos(&self) -> Expr {
        self.as_expr().unary(rustmc_core::graph::ElementwiseOp::Cos)
    }
    fn sum(&self) -> Expr {
        Expr {
            inner: MuExpr::Sum(Box::new(self.as_expr().inner)),
            owner: self.as_expr().owner,
        }
    }
}
#[pymethods]
impl Expr {
    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Expr> {
        self.binary(other, rustmc_core::graph::ElementwiseOp::Mul, false)
    }

    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<Expr> {
        self.as_expr()
            .binary(other, rustmc_core::graph::ElementwiseOp::Add, false)
    }
    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<Expr> {
        self.__add__(other)
    }
    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Expr> {
        self.as_expr()
            .binary(other, rustmc_core::graph::ElementwiseOp::Sub, false)
    }
    fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Expr> {
        self.as_expr()
            .binary(other, rustmc_core::graph::ElementwiseOp::Sub, true)
    }
    fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Expr> {
        self.as_expr()
            .binary(other, rustmc_core::graph::ElementwiseOp::Div, false)
    }
    fn __rtruediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Expr> {
        self.as_expr()
            .binary(other, rustmc_core::graph::ElementwiseOp::Div, true)
    }
    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Expr> {
        self.__mul__(other)
    }
    fn __pow__(
        &self,
        other: &Bound<'_, PyAny>,
        modulo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Expr> {
        if modulo.is_some() {
            return Err(PyValueError::new_err("modular power is unsupported"));
        }
        self.as_expr()
            .binary(other, rustmc_core::graph::ElementwiseOp::Pow, false)
    }
    fn __rpow__(
        &self,
        other: &Bound<'_, PyAny>,
        modulo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Expr> {
        if modulo.is_some() {
            return Err(PyValueError::new_err("modular power is unsupported"));
        }
        self.as_expr()
            .binary(other, rustmc_core::graph::ElementwiseOp::Pow, true)
    }
    fn __neg__(&self) -> Expr {
        self.as_expr().unary(rustmc_core::graph::ElementwiseOp::Neg)
    }
    fn exp(&self) -> Expr {
        self.as_expr().unary(rustmc_core::graph::ElementwiseOp::Exp)
    }
    fn log(&self) -> Expr {
        self.as_expr().unary(rustmc_core::graph::ElementwiseOp::Log)
    }
    fn sqrt(&self) -> Expr {
        self.as_expr()
            .unary(rustmc_core::graph::ElementwiseOp::Sqrt)
    }
    fn sigmoid(&self) -> Expr {
        self.as_expr()
            .unary(rustmc_core::graph::ElementwiseOp::Sigmoid)
    }
    fn tanh(&self) -> Expr {
        self.as_expr()
            .unary(rustmc_core::graph::ElementwiseOp::Tanh)
    }
    fn softplus(&self) -> Expr {
        self.as_expr()
            .unary(rustmc_core::graph::ElementwiseOp::Softplus)
    }
    fn sin(&self) -> Expr {
        self.as_expr().unary(rustmc_core::graph::ElementwiseOp::Sin)
    }
    fn cos(&self) -> Expr {
        self.as_expr().unary(rustmc_core::graph::ElementwiseOp::Cos)
    }
    fn sum(&self) -> Expr {
        Expr {
            inner: MuExpr::Sum(Box::new(self.as_expr().inner)),
            owner: self.as_expr().owner,
        }
    }
}
