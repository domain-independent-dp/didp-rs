use dypdl::expression::*;
use dypdl::prelude::*;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::pyclass::CompareOp;
use std::collections::HashSet;

use crate::ModelPy;

use super::state::StatePy;

pyo3::create_exception!(module, DIDPPyException, pyo3::exceptions::PyException);

#[derive(FromPyObject, Debug, PartialEq, Clone)]
pub enum ExprUnion {
    #[pyo3(transparent, annotation = "ElementExpr")]
    Element(ElementExprPy),
    #[pyo3(transparent, annotation = "SetExpr")]
    Set(SetExprPy),
    #[pyo3(transparent, annotation = "IntExpr")]
    Int(IntExprPy),
    #[pyo3(transparent, annotation = "FloatExpr")]
    Float(FloatExprPy),
    #[pyo3(transparent, annotation = "Condition")]
    Condition(ConditionPy),
}

impl IntoPy<Py<PyAny>> for ExprUnion {
    fn into_py(self, py: Python<'_>) -> Py<PyAny> {
        match self {
            Self::Element(expr) => expr.into_py(py),
            Self::Set(expr) => expr.into_py(py),
            Self::Int(expr) => expr.into_py(py),
            Self::Float(expr) => expr.into_py(py),
            Self::Condition(expr) => expr.into_py(py),
        }
    }
}

#[derive(FromPyObject, Debug, PartialEq, Eq, Clone, Copy)]
pub enum VarUnion {
    #[pyo3(transparent, annotation = "ElementVar")]
    Element(ElementVarPy),
    #[pyo3(transparent, annotation = "ElementResourceVar")]
    ElementResource(ElementResourceVarPy),
    #[pyo3(transparent, annotation = "SetVar")]
    Set(SetVarPy),
    #[pyo3(transparent, annotation = "IntVar")]
    Int(IntVarPy),
    #[pyo3(transparent, annotation = "IntResourceVar")]
    IntResource(IntResourceVarPy),
    #[pyo3(transparent, annotation = "FloatVar")]
    Float(FloatVarPy),
    #[pyo3(transparent, annotation = "FloatResourceVar")]
    FloatResource(FloatResourceVarPy),
}

#[derive(FromPyObject, Debug, PartialEq, Eq, Clone, Copy)]
pub enum ResourceVarUnion {
    #[pyo3(transparent, annotation = "ElementResourceVar")]
    Element(ElementResourceVarPy),
    #[pyo3(transparent, annotation = "IntVar")]
    Int(IntResourceVarPy),
    #[pyo3(transparent, annotation = "FloatResourceVar")]
    Float(FloatResourceVarPy),
}

#[derive(FromPyObject, Debug, PartialEq, Eq, Clone, Copy)]
pub enum ObjectVarUnion {
    #[pyo3(transparent, annotation = "ElementVar")]
    Element(ElementVarPy),
    #[pyo3(transparent, annotation = "ElementResourceVar")]
    ElementResource(ElementResourceVarPy),
    #[pyo3(transparent, annotation = "SetVar")]
    Set(SetVarPy),
}

#[derive(FromPyObject, Debug, PartialEq, Clone)]
pub enum ElementUnion {
    #[pyo3(transparent, annotation = "ElementExpr")]
    Expr(ElementExprPy),
    #[pyo3(transparent, annotation = "ElementVar")]
    Var(ElementVarPy),
    #[pyo3(transparent, annotation = "ElementResourceVar")]
    ResourceVar(ElementResourceVarPy),
    #[pyo3(transparent, annotation = "unsigned int")]
    Const(Element),
}

impl From<ElementUnion> for ElementExpression {
    fn from(element: ElementUnion) -> Self {
        match element {
            ElementUnion::Expr(expr) => ElementExpression::from(expr),
            ElementUnion::Var(var) => ElementExpression::from(var),
            ElementUnion::ResourceVar(var) => ElementExpression::from(var),
            ElementUnion::Const(value) => ElementExpression::from(value),
        }
    }
}

/// Element expression.
///
/// If an arithmetic operator (:code:`+`, :code:`-`, :code:`*`, :code:`/`, :code:`//`, :code:`%`) with an :class:`ElementExpr`, :class:`ElementVar`, :class:`ElementResourceVar`, or :class:`int` is applied, a new :class:`ElementExpr` is returned.
///
/// If a comparison operator (:code:`<`, :code:`<=`, :code:`==`, :code:`!=`, :code:`>`, :code:`>=`) with an :class:`ElementExpr`, :class:`ElementVar`, :class:`ElementResourceVar`, or :class:`int` is applied, a :class:`Condition` is returned.
///
/// Note that :func:`didppy.max` and :func:`didppy.min` should be used instead of :func:`~built_in.max` and :func:`~built_in.min` as comparison operators are overloaded.
///
/// Parameters
/// ----------
/// value : int
///     A non-negative value from which a constant expression is created.
///
/// Raises
/// ------
/// OverflowError
///     If the value is negative.
///
/// Examples
/// --------
/// >>> import didppy as dp
/// >>> model = dp.Model()
/// >>> state = model.target_state
/// >>> expr = dp.ElementExpr(3)
/// >>> expr.eval(state, model)
/// 3
/// >>> (expr + 1).eval(state, model)
/// 4
/// >>> (expr - 1).eval(state, model)
/// 2
/// >>> (expr * 2).eval(state, model)
/// 6
/// >>> (expr / 2).eval(state, model)
/// 1
/// >>> (expr // 2).eval(state, model)
/// 1
/// >>> (expr % 2).eval(state, model)
/// 1
/// >>> (expr < 3).eval(state, model)
/// False
/// >>> (expr <= 3).eval(state, model)
/// True
/// >>> (expr == 3).eval(state, model)
/// True
/// >>> (expr != 3).eval(state, model)
/// False
/// >>> (expr > 3).eval(state, model)
/// False
/// >>> (expr >= 3).eval(state, model)
/// True
#[pyclass(name = "ElementExpr")]
#[derive(Debug, PartialEq, Clone)]
pub struct ElementExprPy(ElementExpression);

impl From<ElementExprPy> for ElementExpression {
    fn from(expression: ElementExprPy) -> Self {
        expression.0
    }
}

impl From<ElementExpression> for ElementExprPy {
    fn from(expression: ElementExpression) -> Self {
        Self(expression)
    }
}

#[pymethods]
impl ElementExprPy {
    #[new]
    #[pyo3(text_signature = "(value)")]
    fn new_py(value: Element) -> Self {
        Self(ElementExpression::from(value))
    }

    fn __richcmp__(&self, other: ElementUnion, op: CompareOp) -> ConditionPy {
        let lhs = self.clone().0;
        let rhs = ElementExpression::from(other);
        let op = match op {
            CompareOp::Lt => ComparisonOperator::Lt,
            CompareOp::Le => ComparisonOperator::Le,
            CompareOp::Eq => ComparisonOperator::Eq,
            CompareOp::Ne => ComparisonOperator::Ne,
            CompareOp::Ge => ComparisonOperator::Ge,
            CompareOp::Gt => ComparisonOperator::Gt,
        };
        ConditionPy(Condition::comparison_e(op, lhs, rhs))
    }

    fn __add__(&self, other: ElementUnion) -> ElementExprPy {
        ElementExprPy(self.clone().0 + ElementExpression::from(other))
    }

    fn __sub__(&self, other: ElementUnion) -> ElementExprPy {
        ElementExprPy(self.clone().0 - ElementExpression::from(other))
    }

    fn __mul__(&self, other: ElementUnion) -> ElementExprPy {
        ElementExprPy(self.clone().0 * ElementExpression::from(other))
    }

    fn __truediv__(&self, other: ElementUnion) -> ElementExprPy {
        self.__floordiv__(other)
    }

    fn __floordiv__(&self, other: ElementUnion) -> ElementExprPy {
        ElementExprPy(self.clone().0 / ElementExpression::from(other))
    }

    fn __mod__(&self, other: ElementUnion) -> ElementExprPy {
        ElementExprPy(self.clone().0 % ElementExpression::from(other))
    }

    fn __radd__(&self, other: ElementUnion) -> ElementExprPy {
        self.__add__(other)
    }

    fn __rsub__(&self, other: ElementUnion) -> ElementExprPy {
        ElementExprPy(ElementExpression::from(other) - self.clone().0)
    }

    fn __rmul__(&self, other: ElementUnion) -> ElementExprPy {
        self.__mul__(other)
    }

    fn __rtruediv__(&self, other: ElementUnion) -> ElementExprPy {
        self.__rfloordiv__(other)
    }

    fn __rfloordiv__(&self, other: ElementUnion) -> ElementExprPy {
        ElementExprPy(ElementExpression::from(other) / self.clone().0)
    }

    fn __rmod__(&self, other: ElementUnion) -> ElementExprPy {
        ElementExprPy(ElementExpression::from(other) % self.clone().0)
    }

    fn __bool__(&self) -> PyResult<bool> {
        Err(DIDPPyException::new_err(
            "ElementExpr cannot be converted to bool",
        ))
    }

    /// eval(state, model)
    ///
    /// Evaluates the expression.
    ///
    /// Parameters
    /// ----------
    /// state : State
    ///     State.
    /// model : Model
    ///     DyPDL Model.
    ///
    /// Returns
    /// -------
    /// int
    ///     Value of the expression.
    ///
    /// Raises
    /// ------
    /// PanicException
    ///     If the expression is not valid.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> obj = model.add_object_type(number=4)
    /// >>> var = model.add_element_var(object_type=obj, target=0)
    /// >>> expr = var + 1
    /// >>> state = model.target_state
    /// >>> expr.eval(state, model)
    /// 1
    #[pyo3(signature = (state, model))]
    fn eval(&self, state: &StatePy, model: &ModelPy) -> Element {
        let mut function_cache =
            StateFunctionCache::new(&model.inner_as_ref().state_functions);

        self.0.eval(
            state.inner_as_ref(),
            &mut function_cache,
            &model.inner_as_ref().state_functions,
            &model.inner_as_ref().table_registry,
        )
    }
}

/// Element variable.
///
/// If an arithmetic operator (:code:`+`, :code:`-`, :code:`*`, :code:`/`, :code:`//`, :code:`%`) with an :class:`ElementExpr`, :class:`ElementVar`, :class:`ElementResourceVar`, or :class:`int` is applied, a new :class:`ElementExpr` is returned.
///
/// If a comparison operator (:code:`<`, :code:`<=`, :code:`==`, :code:`!=`, :code:`>`, :code:`>=`) with an :class:`ElementExpr`, :class:`ElementVar`, :class:`ElementResourceVar`, or :class:`int` is applied, a :class:`Condition` is returned.
///
/// Note that :func:`didppy.max` and :func:`didppy.min` should be used instead of :func:`~built_in.max` and :func:`~built_in.min` as comparison operators are overloaded.
///
/// Examples
/// --------
/// >>> import didppy as dp
/// >>> model = dp.Model()
/// >>> obj = model.add_object_type(number=4)
/// >>> var = model.add_element_var(object_type=obj, target=3)
/// >>> state = model.target_state
/// >>> state[var]
/// 3
/// >>> (var + 1).eval(state, model)
/// 4
/// >>> (var - 1).eval(state, model)
/// 2
/// >>> (var * 2).eval(state, model)
/// 6
/// >>> (var / 2).eval(state, model)
/// 1
/// >>> (var // 2).eval(state, model)
/// 1
/// >>> (var % 2).eval(state, model)
/// 1
/// >>> (var < 3).eval(state, model)
/// False
/// >>> (var <= 3).eval(state, model)
/// True
/// >>> (var == 3).eval(state, model)
/// True
/// >>> (var != 3).eval(state, model)
/// False
/// >>> (var > 3).eval(state, model)
/// False
/// >>> (var >= 3).eval(state, model)
/// True
#[pyclass(name = "ElementVar")]
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct ElementVarPy(ElementVariable);

impl From<ElementVarPy> for ElementVariable {
    fn from(v: ElementVarPy) -> Self {
        v.0
    }
}

impl From<ElementVariable> for ElementVarPy {
    fn from(v: ElementVariable) -> Self {
        Self(v)
    }
}

impl From<ElementVarPy> for ElementExpression {
    fn from(v: ElementVarPy) -> Self {
        v.0.into()
    }
}

#[pymethods]
impl ElementVarPy {
    fn __richcmp__(&self, other: ElementUnion, op: CompareOp) -> ConditionPy {
        let lhs = self.0;
        let rhs = ElementExpression::from(other);
        let op = match op {
            CompareOp::Lt => ComparisonOperator::Lt,
            CompareOp::Le => ComparisonOperator::Le,
            CompareOp::Eq => ComparisonOperator::Eq,
            CompareOp::Ne => ComparisonOperator::Ne,
            CompareOp::Ge => ComparisonOperator::Ge,
            CompareOp::Gt => ComparisonOperator::Gt,
        };
        ConditionPy(Condition::comparison_e(op, lhs, rhs))
    }

    fn __add__(&self, other: ElementUnion) -> ElementExprPy {
        ElementExprPy(self.0 + ElementExpression::from(other))
    }

    fn __sub__(&self, other: ElementUnion) -> ElementExprPy {
        ElementExprPy(self.0 - ElementExpression::from(other))
    }

    fn __mul__(&self, other: ElementUnion) -> ElementExprPy {
        ElementExprPy(self.0 * ElementExpression::from(other))
    }

    fn __truediv__(&self, other: ElementUnion) -> ElementExprPy {
        self.__floordiv__(other)
    }

    fn __floordiv__(&self, other: ElementUnion) -> ElementExprPy {
        ElementExprPy(self.0 / ElementExpression::from(other))
    }

    fn __mod__(&self, other: ElementUnion) -> ElementExprPy {
        ElementExprPy(self.0 % ElementExpression::from(other))
    }

    fn __radd__(&self, other: ElementUnion) -> ElementExprPy {
        self.__add__(other)
    }

    fn __rsub__(&self, other: ElementUnion) -> ElementExprPy {
        ElementExprPy(ElementExpression::from(other) - self.0)
    }

    fn __rmul__(&self, other: ElementUnion) -> ElementExprPy {
        self.__mul__(other)
    }

    fn __rtruediv__(&self, other: ElementUnion) -> ElementExprPy {
        self.__rfloordiv__(other)
    }

    fn __rfloordiv__(&self, other: ElementUnion) -> ElementExprPy {
        ElementExprPy(ElementExpression::from(other) / self.0)
    }

    fn __rmod__(&self, other: ElementUnion) -> ElementExprPy {
        ElementExprPy(ElementExpression::from(other) % self.0)
    }

    fn __bool__(&self) -> PyResult<bool> {
        Err(DIDPPyException::new_err(
            "ElementVar cannot be converted to bool",
        ))
    }
}

/// Element resource variable.
///
/// Intuitively, with :code:`less_is_better=True`/:code:`less_is_better=False`, if everything else is the same, a state having a smaller/greater value is better.
/// Formally, if the values of non-resource variables are the same, a state having equal or better resource variable values must lead to an equal or better solution that has equal or fewer transitions than the other.
///
/// If an arithmetic operator (:code:`+`, :code:`-`, :code:`*`, :code:`/`, :code:`//`, :code:`%`) with an :class:`ElementExpr`, :class:`ElementVar`, :class:`ElementResourceVar`, or :class:`int` is applied, a new :class:`ElementExpr` is returned.
///
/// If a comparison operator (:code:`<`, :code:`<=`, :code:`==`, :code:`!=`, :code:`>`, :code:`>=`) with an :class:`ElementExpr`, :class:`ElementVar`, :class:`ElementResourceVar`, or :class:`int` is applied, a :class:`Condition` is returned.
///
/// Note that :func:`didppy.max` and :func:`didppy.min` should be used instead of :func:`~built_in.max` and :func:`~built_in.min` as comparison operators are overloaded.
///
/// Examples
/// --------
/// >>> import didppy as dp
/// >>> model = dp.Model()
/// >>> obj = model.add_object_type(number=4)
/// >>> var = model.add_element_resource_var(object_type=obj, target=3, less_is_better=True)
/// >>> state = model.target_state
/// >>> state[var]
/// 3
/// >>> (var + 1).eval(state, model)
/// 4
/// >>> (var - 1).eval(state, model)
/// 2
/// >>> (var * 2).eval(state, model)
/// 6
/// >>> (var / 2).eval(state, model)
/// 1
/// >>> (var // 2).eval(state, model)
/// 1
/// >>> (var % 2).eval(state, model)
/// 1
/// >>> (var < 3).eval(state, model)
/// False
/// >>> (var <= 3).eval(state, model)
/// True
/// >>> (var == 3).eval(state, model)
/// True
/// >>> (var != 3).eval(state, model)
/// False
/// >>> (var > 3).eval(state, model)
/// False
/// >>> (var >= 3).eval(state, model)
/// True
#[pyclass(name = "ElementResourceVar")]
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct ElementResourceVarPy(ElementResourceVariable);

impl From<ElementResourceVarPy> for ElementResourceVariable {
    fn from(v: ElementResourceVarPy) -> Self {
        v.0
    }
}

impl From<ElementResourceVariable> for ElementResourceVarPy {
    fn from(v: ElementResourceVariable) -> Self {
        Self(v)
    }
}

impl From<ElementResourceVarPy> for ElementExpression {
    fn from(v: ElementResourceVarPy) -> Self {
        v.0.into()
    }
}

#[pymethods]
impl ElementResourceVarPy {
    fn __richcmp__(&self, other: ElementUnion, op: CompareOp) -> ConditionPy {
        let lhs = self.0;
        let rhs = ElementExpression::from(other);
        let op = match op {
            CompareOp::Lt => ComparisonOperator::Lt,
            CompareOp::Le => ComparisonOperator::Le,
            CompareOp::Eq => ComparisonOperator::Eq,
            CompareOp::Ne => ComparisonOperator::Ne,
            CompareOp::Ge => ComparisonOperator::Ge,
            CompareOp::Gt => ComparisonOperator::Gt,
        };
        ConditionPy(Condition::comparison_e(op, lhs, rhs))
    }

    fn __add__(&self, other: ElementUnion) -> ElementExprPy {
        ElementExprPy(self.0 + ElementExpression::from(other))
    }

    fn __sub__(&self, other: ElementUnion) -> ElementExprPy {
        ElementExprPy(self.0 - ElementExpression::from(other))
    }

    fn __mul__(&self, other: ElementUnion) -> ElementExprPy {
        ElementExprPy(self.0 * ElementExpression::from(other))
    }

    fn __truediv__(&self, other: ElementUnion) -> ElementExprPy {
        self.__floordiv__(other)
    }

    fn __floordiv__(&self, other: ElementUnion) -> ElementExprPy {
        ElementExprPy(self.0 / ElementExpression::from(other))
    }

    fn __mod__(&self, other: ElementUnion) -> ElementExprPy {
        ElementExprPy(self.0 % ElementExpression::from(other))
    }

    fn __radd__(&self, other: ElementUnion) -> ElementExprPy {
        self.__add__(other)
    }

    fn __rsub__(&self, other: ElementUnion) -> ElementExprPy {
        ElementExprPy(ElementExpression::from(other) - self.0)
    }

    fn __rmul__(&self, other: ElementUnion) -> ElementExprPy {
        self.__mul__(other)
    }

    fn __rtruediv__(&self, other: ElementUnion) -> ElementExprPy {
        self.__rfloordiv__(other)
    }

    fn __rfloordiv__(&self, other: ElementUnion) -> ElementExprPy {
        ElementExprPy(ElementExpression::from(other) / self.0)
    }

    fn __rmod__(&self, other: ElementUnion) -> ElementExprPy {
        ElementExprPy(ElementExpression::from(other) % self.0)
    }

    fn __bool__(&self) -> PyResult<bool> {
        Err(DIDPPyException::new_err(
            "ElementResourceVar cannot be converted to bool",
        ))
    }
}

#[derive(FromPyObject, Debug, PartialEq, Clone)]
pub enum SetUnion {
    #[pyo3(transparent, annotation = "SetExpr")]
    Expr(SetExprPy),
    #[pyo3(transparent, annotation = "SetVarPy")]
    Var(SetVarPy),
    #[pyo3(transparent, annotation = "SetConst")]
    Const(SetConstPy),
}

impl From<SetUnion> for SetExpression {
    fn from(set: SetUnion) -> Self {
        match set {
            SetUnion::Expr(expr) => Self::from(expr),
            SetUnion::Var(var) => Self::from(var),
            SetUnion::Const(value) => Self::from(value),
        }
    }
}

/// Set expression.
///
/// If an operator (:code:`-`, :code:`&`, :code:`^`, :code:`|`) with a :class:`SetExpr`, :class:`SetVar`, or :class:`SetConst` is applied, a new :class:`SetExpr` is returned.
///
/// If a comparison operator (:code:`<`, :code:`<=`, :code:`==`, :code:`!=`, :code:`>`, :code:`>=`) with a :class:`SetExpr`, :class:`SetVar`, or :class:`SetConst` is applied, a :class:`Condition` is returned.
///
/// Note that :func:`didppy.max` and :func:`didppy.min` should be used instead of :func:`~built_in.max` and :func:`~built_in.min` as comparison operators are overloaded.
///
/// Parameters
/// ----------
/// value : SetConst
///     A set constant from which a constant expression is created.
///
/// Examples
/// --------
/// >>> import didppy as dp
/// >>> model = dp.Model()
/// >>> state = model.target_state
/// >>> obj = model.add_object_type(number=4)
/// >>> const = model.create_set_const(object_type=obj, value=[0, 1])
/// >>> expr = dp.SetExpr(const)
/// >>> const = model.create_set_const(object_type=obj, value=[1, 2])
/// >>> (expr - const).eval(state, model)
/// {0}
/// >>> (expr & const).eval(state, model)
/// {1}
/// >>> (expr ^ const).eval(state, model)
/// {0, 2}
/// >>> (expr | const).eval(state, model)
/// {0, 1, 2}
/// >>> (expr < const).eval(state, model)
/// False
/// >>> (expr <= const).eval(state, model)
/// False
/// >>> (expr == const).eval(state, model)
/// False
/// >>> (expr != const).eval(state, model)
/// True
/// >>> (expr > const).eval(state, model)
/// False
/// >>> (expr >= const).eval(state, model)
/// False
#[pyclass(name = "SetExpr")]
#[derive(Debug, PartialEq, Clone)]
pub struct SetExprPy(SetExpression);

impl From<SetExprPy> for SetExpression {
    fn from(expression: SetExprPy) -> Self {
        expression.0
    }
}

impl From<SetExpression> for SetExprPy {
    fn from(expression: SetExpression) -> Self {
        SetExprPy(expression)
    }
}

#[pymethods]
impl SetExprPy {
    #[new]
    #[pyo3(text_signature = "(value)")]
    fn new_py(value: SetConstPy) -> Self {
        Self::from(SetExpression::from(value))
    }

    fn __richcmp__(&self, other: SetUnion, op: CompareOp) -> ConditionPy {
        let lhs = self.clone().0;
        let rhs = SetExpression::from(other);
        let condition = match op {
            CompareOp::Lt => lhs.clone().is_subset(rhs.clone()) & !rhs.is_subset(lhs),
            CompareOp::Le => lhs.is_subset(rhs),
            CompareOp::Eq => lhs.is_equal(rhs),
            CompareOp::Ne => lhs.is_not_equal(rhs),
            CompareOp::Ge => rhs.is_subset(lhs),
            CompareOp::Gt => rhs.clone().is_subset(lhs.clone()) & !lhs.is_subset(rhs),
        };
        ConditionPy(condition)
    }

    fn __sub__(&self, other: SetUnion) -> SetExprPy {
        SetExprPy(self.clone().0 - SetExpression::from(other))
    }

    fn __and__(&self, other: SetUnion) -> SetExprPy {
        SetExprPy(self.clone().0 & SetExpression::from(other))
    }

    fn __xor__(&self, other: SetUnion) -> SetExprPy {
        let other = SetExpression::from(other);
        SetExprPy((self.clone().0 - other.clone()) | (other - self.clone().0))
    }

    fn __or__(&self, other: SetUnion) -> SetExprPy {
        SetExprPy(self.clone().0 | SetExpression::from(other))
    }

    fn __rsub__(&self, other: SetUnion) -> SetExprPy {
        SetExprPy(SetExpression::from(other) - self.clone().0)
    }

    fn __rand__(&self, other: SetUnion) -> SetExprPy {
        self.__and__(other)
    }

    fn __rxor__(&self, other: SetUnion) -> SetExprPy {
        self.__xor__(other)
    }

    fn __ror__(&self, other: SetUnion) -> SetExprPy {
        self.__or__(other)
    }

    fn __bool__(&self) -> PyResult<bool> {
        Err(DIDPPyException::new_err(
            "SetExpr cannot be converted to bool",
        ))
    }

    /// isdisjoint(other)
    ///
    /// Checks if two sets are disjoint.
    ///
    /// Parameters
    /// ----------
    /// other: SetExpr, SetVar, or SetConst
    ///    The other set.
    ///
    /// Returns
    /// -------
    /// Condition
    ///    The condition that the two sets are disjoint.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> state = model.target_state
    /// >>> obj = model.add_object_type(number=4)
    /// >>> const = model.create_set_const(object_type=obj, value=[0, 1])
    /// >>> expr = dp.SetExpr(const)
    /// >>> const = model.create_set_const(object_type=obj, value=[2, 3])
    /// >>> expr.isdisjoint(const).eval(state, model)
    /// True
    #[pyo3(signature = (other))]
    fn isdisjoint(&self, other: SetUnion) -> ConditionPy {
        self.__and__(other).is_empty()
    }

    /// issubset(other)
    ///
    /// Checks if this set is a subset of another set.
    ///
    /// Parameters
    /// ----------
    /// other: SetExpr, SetVar, or SetConst
    ///    The other set.
    ///
    /// Returns
    /// -------
    /// Condition
    ///    The condition that the two sets are disjoint.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> state = model.target_state
    /// >>> obj = model.add_object_type(number=4)
    /// >>> const = model.create_set_const(object_type=obj, value=[0, 1])
    /// >>> expr = dp.SetExpr(const)
    /// >>> const = model.create_set_const(object_type=obj, value=[0, 1, 2])
    /// >>> expr.issubset(const).eval(state, model)
    /// True
    #[pyo3(signature = (other))]
    fn issubset(&self, other: SetUnion) -> ConditionPy {
        let lhs = self.clone().0;
        let rhs = SetExpression::from(other);
        ConditionPy(lhs.is_subset(rhs))
    }

    /// issuperset(other)
    ///
    /// Checks if this set is a superset of another set.
    ///
    /// Parameters
    /// ----------
    /// other: SetExpr, SetVar, or SetConst
    ///    The other set.
    ///
    /// Returns
    /// -------
    /// Condition
    ///    The condition that the two sets are disjoint.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> state = model.target_state
    /// >>> obj = model.add_object_type(number=4)
    /// >>> const = model.create_set_const(object_type=obj, value=[0, 1])
    /// >>> expr = dp.SetExpr(const)
    /// >>> const = model.create_set_const(object_type=obj, value=[0])
    /// >>> expr.issuperset(const).eval(state, model)
    /// True
    #[pyo3(signature = (other))]
    fn issuperset(&self, other: SetUnion) -> ConditionPy {
        let lhs = self.clone().0;
        let rhs = SetExpression::from(other);
        ConditionPy(rhs.is_subset(lhs))
    }

    /// add(element)
    ///
    /// Adds an element to a set.
    ///
    /// This method does not change the instance itself.
    ///
    /// Parameters
    /// ----------
    /// element: ElementExpr, ElementVar, ElementResourceVar, or int
    ///     Element added to the set.
    ///
    /// Returns
    /// -------
    /// SetExpr
    ///     The set where the element is added.
    ///
    /// Raises
    /// ------
    /// OverflowError
    ///     If :code:`element` is :class:`int` and negative.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> state = model.target_state
    /// >>> obj = model.add_object_type(number=4)
    /// >>> const = model.create_set_const(object_type=obj, value=[0, 1])
    /// >>> expr = dp.SetExpr(const)
    /// >>> expr.add(2).eval(state, model)
    /// {0, 1, 2}
    #[pyo3(signature = (element))]
    fn add(&self, element: ElementUnion) -> SetExprPy {
        let element = ElementExpression::from(element);
        SetExprPy(self.clone().0.add(element))
    }

    /// remove(element)
    ///
    /// Removes an element from a set.
    ///
    /// This method does not change the instance itself.
    ///
    /// Parameters
    /// ----------
    /// element: ElementExpr, ElementVar, ElementResourceVar, or int
    ///     Element removed from the set.
    ///
    /// Returns
    /// -------
    /// SetExpr
    ///     The set where the element is removed.
    ///
    /// Raises
    /// ------
    /// OverflowError
    ///     If :code:`element` is :class:`int` and negative.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> state = model.target_state
    /// >>> obj = model.add_object_type(number=4)
    /// >>> const = model.create_set_const(object_type=obj, value=[0, 1])
    /// >>> expr = dp.SetExpr(const)
    /// >>> expr.remove(1).eval(state, model)
    /// {0}
    #[pyo3(signature = (element))]
    fn remove(&self, element: ElementUnion) -> SetExprPy {
        self.discard(element)
    }

    /// discard(element)
    ///
    /// Removes an element from a set.
    ///
    /// This method does not change the instance itself.
    ///
    /// Parameters
    /// ----------
    /// element: ElementExpr, ElementVar, ElementResourceVar, or int
    ///     Element removed from the set.
    ///
    /// Returns
    /// -------
    /// SetExpr
    ///     The set where the element is removed.
    ///
    /// Raises
    /// ------
    /// OverflowError
    ///     If :code:`element` is :class:`int` and negative.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> state = model.target_state
    /// >>> obj = model.add_object_type(number=4)
    /// >>> const = model.create_set_const(object_type=obj, value=[0, 1])
    /// >>> expr = dp.SetExpr(const)
    /// >>> expr.discard(1).eval(state, model)
    /// {0}
    #[pyo3(signature = (element))]
    fn discard(&self, element: ElementUnion) -> SetExprPy {
        let element = ElementExpression::from(element);
        SetExprPy(self.clone().0.remove(element))
    }

    /// difference(other)
    ///
    /// Returns a set where all elements in an input set are removed.
    ///
    /// This method is the same as :code:`-` operation.
    /// This method does not change the instance itself.
    ///
    /// Parameters
    /// ----------
    /// other: SetExpr, SetVar, or SetConst
    ///     Set to remove.
    /// Returns
    /// -------
    /// SetExpr
    ///     The set where all elements in :code:`other` are removed.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> state = model.target_state
    /// >>> obj = model.add_object_type(number=4)
    /// >>> const = model.create_set_const(object_type=obj, value=[0, 1])
    /// >>> expr = dp.SetExpr(const)
    /// >>> const = model.create_set_const(object_type=obj, value=[1, 2])
    /// >>> expr.difference(const).eval(state, model)
    /// {0}
    #[pyo3(signature = (other))]
    fn difference(&self, other: SetUnion) -> SetExprPy {
        self.__sub__(other)
    }

    /// intersection(other)
    ///
    /// Returns the intersection with another set.
    ///
    /// This method is the same as :code:`&` operation.
    /// This method does not change the instance itself.
    ///
    /// Parameters
    /// ----------
    /// other: SetExpr, SetVar, or SetConst
    ///     Set to take the intersection with.
    ///
    /// Returns
    /// -------
    /// SetExpr
    ///     The intersection.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> state = model.target_state
    /// >>> obj = model.add_object_type(number=4)
    /// >>> const = model.create_set_const(object_type=obj, value=[0, 1])
    /// >>> expr = dp.SetExpr(const)
    /// >>> const = model.create_set_const(object_type=obj, value=[1, 2])
    /// >>> expr.intersection(const).eval(state, model)
    /// {1}
    #[pyo3(signature = (other))]
    fn intersection(&self, other: SetUnion) -> SetExprPy {
        self.__and__(other)
    }

    /// symmetric_difference(other)
    ///
    /// Returns a set which only contains elements included in either of two sets but not in both.
    ///
    /// This method is the same as :code:`^` operation.
    /// This method does not change the instance itself.
    ///
    /// Parameters
    /// ----------
    /// other: SetExpr, SetVar, or SetConst
    ///     Set to take the symmetric difference with.
    ///
    /// Returns
    /// -------
    /// SetExpr
    ///     The symmetric difference set.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> state = model.target_state
    /// >>> obj = model.add_object_type(number=4)
    /// >>> const = model.create_set_const(object_type=obj, value=[0, 1])
    /// >>> expr = dp.SetExpr(const)
    /// >>> const = model.create_set_const(object_type=obj, value=[1, 2])
    /// >>> expr.symmetric_difference(const).eval(state, model)
    /// {0, 2}
    #[pyo3(signature = (other))]
    fn symmetric_difference(&self, other: SetUnion) -> SetExprPy {
        self.__xor__(other)
    }

    /// union(other)
    ///
    /// Returns the union of two sets.
    ///
    /// This method is the same as :code:`\|` operation.
    /// This method does not change the instance itself.
    ///
    /// Parameters
    /// ----------
    /// other: SetExpr, SetVar, or SetConst
    ///     Set to take the union with.
    ///
    /// Returns
    /// -------
    /// SetExpr
    ///     The union.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> state = model.target_state
    /// >>> obj = model.add_object_type(number=4)
    /// >>> const = model.create_set_const(object_type=obj, value=[0, 1])
    /// >>> expr = dp.SetExpr(const)
    /// >>> const = model.create_set_const(object_type=obj, value=[1, 2])
    /// >>> expr.union(const).eval(state, model)
    /// {0, 1, 2}
    #[pyo3(signature = (other))]
    fn union(&self, other: SetUnion) -> SetExprPy {
        self.__or__(other)
    }

    /// contains(element)
    ///
    /// Returns a condition checking if an element is included.
    ///
    /// Parameters
    /// ----------
    /// element: ElementExpr, ElementVar, ElementResourceVar, or int
    ///     Element to check.
    ///
    /// Returns
    /// -------
    /// Condition
    ///     The condition checking if an element is included in the set.
    ///
    /// Raises
    /// ------
    /// OverflowError
    ///     If :code:`element` is :class:`int` and negative.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> state = model.target_state
    /// >>> obj = model.add_object_type(number=4)
    /// >>> const = model.create_set_const(object_type=obj, value=[0, 1])
    /// >>> expr = dp.SetExpr(const)
    /// >>> expr.contains(0).eval(state, model)
    /// True
    #[pyo3(signature = (element))]
    fn contains(&self, element: ElementUnion) -> ConditionPy {
        let element = ElementExpression::from(element);
        ConditionPy(self.clone().0.contains(element))
    }

    /// Returns the cardinality of a set.
    ///
    /// Returns
    /// -------
    /// IntExpr
    ///     The cardinality.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> state = model.target_state
    /// >>> obj = model.add_object_type(number=4)
    /// >>> const = model.create_set_const(object_type=obj, value=[0, 1])
    /// >>> expr = dp.SetExpr(const)
    /// >>> expr.len().eval(state, model)
    /// 2
    #[pyo3(signature = ())]
    pub fn len(&self) -> IntExprPy {
        IntExprPy(self.clone().0.len())
    }

    /// Returns a condition checking if the set is empty.
    ///
    /// Returns
    /// -------
    /// Condition
    ///     The condition checking if the set is empty.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> state = model.target_state
    /// >>> obj = model.add_object_type(number=4)
    /// >>> const = model.create_set_const(object_type=obj, value=[0, 1])
    /// >>> expr = dp.SetExpr(const)
    /// >>> expr.is_empty().eval(state, model)
    /// False
    #[pyo3(signature = ())]
    pub fn is_empty(&self) -> ConditionPy {
        ConditionPy(self.clone().0.is_empty())
    }

    /// Returns the complement set.
    ///
    /// Returns
    /// -------
    /// SetExpr
    ///     The complement set.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> state = model.target_state
    /// >>> obj = model.add_object_type(number=4)
    /// >>> const = model.create_set_const(object_type=obj, value=[0, 1])
    /// >>> expr = dp.SetExpr(const)
    /// >>> expr.complement().eval(state, model)
    /// {2, 3}
    #[pyo3(signature = ())]
    pub fn complement(&self) -> SetExprPy {
        SetExprPy(!self.clone().0)
    }

    /// eval(state, model)
    ///
    /// Evaluates the expression.
    ///
    /// Parameters
    /// ----------
    /// state : State
    ///     State.
    /// model : Model
    ///     DyPDL Model.
    ///
    /// Returns
    /// -------
    /// set
    ///     Value of the expression.
    ///
    /// Raises
    /// ------
    /// PanicException
    ///     If the expression is not valid.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> obj = model.add_object_type(number=4)
    /// >>> var = model.add_set_var(object_type=obj, target=[0])
    /// >>> expr = var.add(1)
    /// >>> state = model.target_state
    /// >>> expr.eval(state, model)
    /// {0, 1}
    #[pyo3(signature = (state, model))]
    fn eval(&self, state: &StatePy, model: &ModelPy) -> HashSet<usize> {
        let mut function_cache =
            StateFunctionCache::new(&model.inner_as_ref().state_functions);

        HashSet::from_iter(
            self.0
                .eval(
                    state.inner_as_ref(),
                    &mut function_cache,
                    &model.inner_as_ref().state_functions,
                    &model.inner_as_ref().table_registry,
                )
                .ones(),
        )
    }
}

/// Set variable.
///
/// If an operator (:code:`-`, :code:`&`, :code:`^`, :code:`|`) with a :class:`SetExpr`, :class:`SetVar`, or :class:`SetConst` is applied, a new :class:`SetExpr` is returned.
///
/// If a comparison operator (:code:`<`, :code:`<=`, :code:`==`, :code:`!=`, :code:`>`, :code:`>=`) with a :class:`SetExpr`, :class:`SetVar`, or :class:`SetConst` is applied, a :class:`Condition` is returned.
///
/// Note that :func:`didppy.max` and :func:`didppy.min` should be used instead of :func:`~built_in.max` and :func:`~built_in.min` as comparison operators are overloaded.
///
/// Examples
/// --------
/// >>> import didppy as dp
/// >>> model = dp.Model()
/// >>> obj = model.add_object_type(number=4)
/// >>> var = model.add_set_var(object_type=obj, target=[0, 1])
/// >>> const = model.create_set_const(object_type=obj, value=[1, 2])
/// >>> state = model.target_state
/// >>> state[var]
/// {0, 1}
/// >>> (var - const).eval(state, model)
/// {0}
/// >>> (var & const).eval(state, model)
/// {1}
/// >>> (var ^ const).eval(state, model)
/// {0, 2}
/// >>> (var | const).eval(state, model)
/// {0, 1, 2}
/// >>> (var < const).eval(state, model)
/// False
/// >>> (var <= const).eval(state, model)
/// False
/// >>> (var == const).eval(state, model)
/// False
/// >>> (var != const).eval(state, model)
/// True
/// >>> (var > const).eval(state, model)
/// False
/// >>> (var >= const).eval(state, model)
/// False
#[pyclass(name = "SetVar")]
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct SetVarPy(SetVariable);

impl From<SetVarPy> for SetVariable {
    fn from(v: SetVarPy) -> Self {
        v.0
    }
}

impl From<SetVariable> for SetVarPy {
    fn from(v: SetVariable) -> Self {
        Self(v)
    }
}

impl From<SetVarPy> for SetExpression {
    fn from(v: SetVarPy) -> Self {
        v.0.into()
    }
}

#[pymethods]
impl SetVarPy {
    fn __richcmp__(&self, other: SetUnion, op: CompareOp) -> ConditionPy {
        let lhs = self.0;
        let rhs = SetExpression::from(other);
        let condition = match op {
            CompareOp::Lt => lhs.is_subset(rhs.clone()) & !rhs.is_subset(lhs),
            CompareOp::Le => lhs.is_subset(rhs),
            CompareOp::Eq => lhs.is_equal(rhs),
            CompareOp::Ne => lhs.is_not_equal(rhs),
            CompareOp::Ge => rhs.is_subset(lhs),
            CompareOp::Gt => rhs.clone().is_subset(lhs) & !lhs.is_subset(rhs),
        };
        ConditionPy(condition)
    }

    fn __sub__(&self, other: SetUnion) -> SetExprPy {
        SetExprPy(self.0 - SetExpression::from(other))
    }

    fn __and__(&self, other: SetUnion) -> SetExprPy {
        SetExprPy(self.0 & SetExpression::from(other))
    }

    fn __xor__(&self, other: SetUnion) -> SetExprPy {
        let other = SetExpression::from(other);
        SetExprPy((self.0 - other.clone()) | (other - self.0))
    }

    fn __or__(&self, other: SetUnion) -> SetExprPy {
        SetExprPy(self.0 | SetExpression::from(other))
    }

    fn __rsub__(&self, other: SetUnion) -> SetExprPy {
        SetExprPy(SetExpression::from(other) - self.0)
    }

    fn __rand__(&self, other: SetUnion) -> SetExprPy {
        self.__and__(other)
    }

    fn __rxor__(&self, other: SetUnion) -> SetExprPy {
        self.__xor__(other)
    }

    fn __ror__(&self, other: SetUnion) -> SetExprPy {
        self.__or__(other)
    }

    fn __bool__(&self) -> PyResult<bool> {
        Err(DIDPPyException::new_err(
            "SetVar cannot be converted to bool",
        ))
    }

    /// isdisjoint(other)
    ///
    /// Checks if two sets are disjoint.
    ///
    /// Parameters
    /// ----------
    /// other: SetExpr, SetVar, or SetConst
    ///    The other set.
    ///
    /// Returns
    /// -------
    /// Condition
    ///    The condition that the two sets are disjoint.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> obj = model.add_object_type(number=4)
    /// >>> var = model.add_set_var(object_type=obj, target=[0, 1])
    /// >>> const = model.create_set_const(object_type=obj, value=[2, 3])
    /// >>> state = model.target_state
    /// >>> var.isdisjoint(const).eval(state, model)
    /// True
    #[pyo3(signature = (other))]
    fn isdisjoint(&self, other: SetUnion) -> ConditionPy {
        self.__and__(other).is_empty()
    }

    /// issubset(other)
    ///
    /// Checks if this set is a subset of another set.
    ///
    /// Parameters
    /// ----------
    /// other: SetExpr, SetVar, or SetConst
    ///    The other set.
    ///
    /// Returns
    /// -------
    /// Condition
    ///    The condition that the two sets are disjoint.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> obj = model.add_object_type(number=4)
    /// >>> var = model.add_set_var(object_type=obj, target=[0, 1])
    /// >>> const = model.create_set_const(object_type=obj, value=[0, 1, 2])
    /// >>> state = model.target_state
    /// >>> var.issubset(const).eval(state, model)
    /// True
    #[pyo3(signature = (other))]
    fn issubset(&self, other: SetUnion) -> ConditionPy {
        let lhs = self.0;
        let rhs = SetExpression::from(other);
        ConditionPy(lhs.is_subset(rhs))
    }

    /// issuperset(other)
    ///
    /// Checks if this set is a superset of another set.
    ///
    /// Parameters
    /// ----------
    /// other: SetExpr, SetVar, or SetConst
    ///    The other set.
    ///
    /// Returns
    /// -------
    /// Condition
    ///    The condition that the two sets are disjoint.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> obj = model.add_object_type(number=4)
    /// >>> var = model.add_set_var(object_type=obj, target=[0, 1])
    /// >>> const = model.create_set_const(object_type=obj, value=[0])
    /// >>> state = model.target_state
    /// >>> expr.issuperset(const).eval(state, model)
    /// True
    #[pyo3(signature = (other))]
    fn issuperset(&self, other: SetUnion) -> ConditionPy {
        let lhs = self.0;
        let rhs = SetExpression::from(other);
        ConditionPy(rhs.is_subset(lhs))
    }

    /// add(element)
    ///
    /// Adds an element to a set.
    ///
    /// This method does not change the instance itself.
    ///
    /// Parameters
    /// ----------
    /// element: ElementExpr, ElementVar, ElementResourceVar, or int
    ///     Element added to the set.
    ///
    /// Returns
    /// -------
    /// SetExpr
    ///     The set where the element is added.
    ///
    /// Raises
    /// ------
    /// OverflowError
    ///     If :code:`element` is :class:`int` and negative.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> obj = model.add_object_type(number=4)
    /// >>> var = model.add_set_var(object_type=obj, target=[0, 1])
    /// >>> state = model.target_state
    /// >>> var.add(2).eval(state, model)
    /// {0, 1, 2}
    #[pyo3(signature = (element))]
    fn add(&self, element: ElementUnion) -> SetExprPy {
        let element = ElementExpression::from(element);
        SetExprPy(self.0.add(element))
    }

    /// remove(element)
    ///
    /// Removes an element from a set.
    ///
    /// This method does not change the instance itself.
    ///
    /// Parameters
    /// ----------
    /// element: ElementExpr, ElementVar, ElementResourceVar, or int
    ///     Element removed from the set.
    ///
    /// Returns
    /// -------
    /// SetExpr
    ///     The set where the element is removed.
    ///
    /// Raises
    /// ------
    /// OverflowError
    ///     If :code:`element` is :class:`int` and negative.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> obj = model.add_object_type(number=4)
    /// >>> var = model.add_set_var(object_type=obj, target=[0, 1])
    /// >>> state = model.target_state
    /// >>> var.remove(1).eval(state, model)
    /// {0}
    #[pyo3(signature = (element))]
    fn remove(&self, element: ElementUnion) -> SetExprPy {
        self.discard(element)
    }

    /// discard(element)
    ///
    /// Removes an element from a set.
    ///
    /// This method does not change the instance itself.
    ///
    /// Parameters
    /// ----------
    /// element: ElementExpr, ElementVar, ElementResourceVar, or int
    ///     Element removed from the set.
    ///
    /// Returns
    /// -------
    /// SetExpr
    ///     The set where the element is removed.
    ///
    /// Raises
    /// ------
    /// OverflowError
    ///     If :code:`element` is :class:`int` and negative.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> obj = model.add_object_type(number=4)
    /// >>> var = model.add_set_var(object_type=obj, target=[0, 1])
    /// >>> state = model.target_state
    /// >>> var.discard(1).eval(state, model)
    /// {0}
    #[pyo3(signature = (element))]
    fn discard(&self, element: ElementUnion) -> SetExprPy {
        let element = ElementExpression::from(element);
        SetExprPy(self.0.remove(element))
    }

    /// difference(other)
    ///
    /// Returns a set where all elements in an input set are removed.
    ///
    /// This method is the same as :code:`-` operation.
    /// This method does not change the instance itself.
    ///
    /// Parameters
    /// ----------
    /// other: SetExpr, SetVar, or SetConst
    ///     Set to remove.
    ///
    /// Returns
    /// -------
    /// SetExpr
    ///     The set where all elements in :code:`other` are removed.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> obj = model.add_object_type(number=4)
    /// >>> var = model.add_set_var(object_type=obj, target=[0, 1])
    /// >>> const = model.create_set_const(object_type=obj, value=[1, 2])
    /// >>> state = model.target_state
    /// >>> var.difference(const).eval(state, model)
    /// {0}
    #[pyo3(signature = (other))]
    fn difference(&self, other: SetUnion) -> SetExprPy {
        self.__sub__(other)
    }

    /// intersection(other)
    ///
    /// Returns the intersection with another set.
    ///
    /// This method is the same as :code:`&` operation.
    /// This method does not change the instance itself.
    ///
    /// Parameters
    /// ----------
    /// other: SetExpr, SetVar, or SetConst
    ///     Set to take the intersection with.
    ///
    /// Returns
    /// -------
    /// SetExpr
    ///     The intersection.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> obj = model.add_object_type(number=4)
    /// >>> var = model.add_set_var(object_type=obj, target=[0, 1])
    /// >>> const = model.create_set_const(object_type=obj, value=[1, 2])
    /// >>> state = model.target_state
    /// >>> var.intersection(const).eval(state, model)
    /// {1}
    fn intersection(&self, other: SetUnion) -> SetExprPy {
        self.__and__(other)
    }

    /// symmetric_difference(other)
    ///
    /// Returns a set which only contains elements included in either of two sets but not in both.
    ///
    /// This method is the same as :code:`^` operation.
    /// This method does not change the instance itself.
    ///
    /// Parameters
    /// ----------
    /// other: SetExpr, SetVar, or SetConst
    ///     Set to take the symmetric difference with.
    ///
    /// Returns
    /// -------
    /// SetExpr
    ///     The symmetric difference set.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> obj = model.add_object_type(number=4)
    /// >>> var = model.add_set_var(object_type=obj, target=[0, 1])
    /// >>> const = model.create_set_const(object_type=obj, value=[1, 2])
    /// >>> state = model.target_state
    /// >>> var.symmetric_difference(const).eval(state, model)
    /// {0, 2}
    #[pyo3(signature = (other))]
    fn symmetric_difference(&self, other: SetUnion) -> SetExprPy {
        self.__xor__(other)
    }

    /// union(other)
    ///
    /// Returns the union of two sets.
    ///
    /// This method is the same as :code:`\|` operation.
    /// This method does not change the instance itself.
    ///
    /// Parameters
    /// ----------
    /// other: SetExpr, SetVar, or SetConst
    ///     Set to take the union with.
    ///
    /// Returns
    /// -------
    /// SetExpr
    ///     The union.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> obj = model.add_object_type(number=4)
    /// >>> var = model.add_set_var(object_type=obj, target=[0, 1])
    /// >>> const = model.create_set_const(object_type=obj, value=[1, 2])
    /// >>> state = model.target_state
    /// >>> var.union(const).eval(state, model)
    /// {0, 1, 2}
    #[pyo3(signature = (other))]
    fn union(&self, other: SetUnion) -> SetExprPy {
        self.__or__(other)
    }

    /// contains(element)
    ///
    /// Returns a condition checking if an element is included.
    ///
    /// Parameters
    /// ----------
    /// element: ElementExpr, ElementVar, ElementResourceVar, or int
    ///     Element to check.
    ///
    /// Returns
    /// -------
    /// Condition
    ///     The condition checking if an element is included in the set.
    ///
    /// Raises
    /// ------
    /// OverflowError
    ///     If :code:`element` is :class:`int` and negative.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> obj = model.add_object_type(number=4)
    /// >>> var = model.add_set_var(object_type=obj, target=[0, 1])
    /// >>> state = model.target_state
    /// >>> var.contains(0).eval(state, model)
    /// True
    #[pyo3(signature = (element))]
    fn contains(&self, element: ElementUnion) -> ConditionPy {
        let element = ElementExpression::from(element);
        ConditionPy(self.0.contains(element))
    }

    /// Returns the cardinality of a set.
    ///
    /// Returns
    /// -------
    /// IntExpr
    ///     The cardinality.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> obj = model.add_object_type(number=4)
    /// >>> var = model.add_set_var(object_type=obj, target=[0, 1])
    /// >>> state = model.target_state
    /// >>> var.len().eval(state, model)
    /// 2
    #[pyo3(signature = ())]
    fn len(&self) -> IntExprPy {
        IntExprPy(self.0.len())
    }

    /// Returns a condition checking if the set is empty.
    ///
    /// Returns
    /// -------
    /// Condition
    ///     The condition checking if the set is empty.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> obj = model.add_object_type(number=4)
    /// >>> var = model.add_set_var(object_type=obj, target=[0, 1])
    /// >>> state = model.target_state
    /// >>> var.is_empty().eval(state, model)
    /// False
    #[pyo3(signature = ())]
    fn is_empty(&self) -> ConditionPy {
        ConditionPy(self.0.is_empty())
    }

    /// Returns the complement set.
    ///
    /// Returns
    /// -------
    /// SetExpr
    ///     The complement set.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> obj = model.add_object_type(number=4)
    /// >>> var = model.add_set_var(object_type=obj, target=[0, 1])
    /// >>> state = model.target_state
    /// >>> var.complement().eval(state, model)
    /// {2, 3}
    #[pyo3(signature = ())]
    fn complement(&self) -> SetExprPy {
        SetExprPy(!self.0)
    }
}

/// Set constant.
#[pyclass(name = "SetConst")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetConstPy(Set);

impl From<SetConstPy> for Set {
    fn from(set: SetConstPy) -> Self {
        set.0
    }
}

impl From<Set> for SetConstPy {
    fn from(set: Set) -> Self {
        Self(set)
    }
}

impl From<SetConstPy> for SetExpression {
    fn from(set: SetConstPy) -> Self {
        set.0.into()
    }
}

/// Set constant.
///
/// If an operator (:code:`-`, :code:`&`, :code:`^`, :code:`|`) with a :class:`SetExpr`, :class:`SetVar`, or :class:`SetConst` is applied, a new :class:`SetExpr` is returned.
///
/// If a comparison operator (:code:`<`, :code:`<=`, :code:`==`, :code:`!=`, :code:`>`, :code:`>=`) with a :class:`SetExpr`, :class:`SetVar`, or :class:`SetConst` is applied, a :class:`Condition` is returned.
///
/// Note that :func:`didppy.max` and :func:`didppy.min` should be used instead of :func:`~built_in.max` and :func:`~built_in.min` as comparison operators are overloaded.
///
/// Examples
/// --------
/// >>> import didppy as dp
/// >>> model = dp.Model()
/// >>> state = model.target_state
/// >>> obj = model.add_object_type(number=4)
/// >>> const = model.create_set_const(object_type=obj, value=[0, 1])
/// >>> other = model.create_set_const(object_type=obj, value=[1, 2])
/// >>> (const - other).eval(state, model)
/// {0}
/// >>> (const & other).eval(state, model)
/// {1}
/// >>> (const ^ other).eval(state, model)
/// {0, 2}
/// >>> (const | other).eval(state, model)
/// {0, 1, 2}
/// >>> (const < other).eval(state, model)
/// False
/// >>> (const <= other).eval(state, model)
/// False
/// >>> (const == other).eval(state, model)
/// False
/// >>> (const != other).eval(state, model)
/// True
/// >>> (const > other).eval(state, model)
/// False
/// >>> (const >= other).eval(state, model)
/// False
#[pymethods]
impl SetConstPy {
    fn __richcmp__(&self, other: SetUnion, op: CompareOp) -> ConditionPy {
        let lhs = SetExpression::from(self.clone());
        let rhs = SetExpression::from(other);
        let condition = match op {
            CompareOp::Lt => lhs.clone().is_subset(rhs.clone()) & !rhs.is_subset(lhs),
            CompareOp::Le => lhs.is_subset(rhs),
            CompareOp::Eq => lhs.is_equal(rhs),
            CompareOp::Ne => lhs.is_not_equal(rhs),
            CompareOp::Ge => rhs.is_subset(lhs),
            CompareOp::Gt => rhs.clone().is_subset(lhs.clone()) & !lhs.is_subset(rhs),
        };
        ConditionPy(condition)
    }

    fn __sub__(&self, other: SetUnion) -> SetExprPy {
        SetExprPy(self.clone().0 - SetExpression::from(other))
    }

    fn __and__(&self, other: SetUnion) -> SetExprPy {
        SetExprPy(self.clone().0 & SetExpression::from(other))
    }

    fn __xor__(&self, other: SetUnion) -> SetExprPy {
        let other = SetExpression::from(other);
        SetExprPy((self.clone().0 - other.clone()) | (other - self.clone().0))
    }

    fn __or__(&self, other: SetUnion) -> SetExprPy {
        SetExprPy(self.clone().0 | SetExpression::from(other))
    }

    fn __rsub__(&self, other: SetUnion) -> SetExprPy {
        SetExprPy(SetExpression::from(other) - self.clone().0)
    }

    fn __rand__(&self, other: SetUnion) -> SetExprPy {
        self.__and__(other)
    }

    fn __rxor__(&self, other: SetUnion) -> SetExprPy {
        self.__xor__(other)
    }

    fn __ror__(&self, other: SetUnion) -> SetExprPy {
        self.__or__(other)
    }

    fn __bool__(&self) -> PyResult<bool> {
        Err(DIDPPyException::new_err(
            "SetConst cannot be converted to bool",
        ))
    }

    /// isdisjoint(other)
    ///
    /// Checks if two sets are disjoint.
    ///
    /// Parameters
    /// ----------
    /// other: SetExpr, SetVar, or SetConst
    ///    The other set.
    ///
    /// Returns
    /// -------
    /// Condition
    ///    The condition that the two sets are disjoint.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> state = model.target_state
    /// >>> obj = model.add_object_type(number=4)
    /// >>> const = model.create_set_const(object_type=obj, value=[0, 1])
    /// >>> other = model.create_set_const(object_type=obj, value=[2, 3])
    /// >>> const.isdisjoint(other).eval(state, model)
    /// True
    #[pyo3(signature = (other))]
    fn isdisjoint(&self, other: SetUnion) -> ConditionPy {
        self.__and__(other).is_empty()
    }

    /// issubset(other)
    ///
    /// Checks if this set is a subset of another set.
    ///
    /// Parameters
    /// ----------
    /// other: SetExpr, SetVar, or SetConst
    ///    The other set.
    ///
    /// Returns
    /// -------
    /// Condition
    ///    The condition that the two sets are disjoint.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> state = model.target_state
    /// >>> obj = model.add_object_type(number=4)
    /// >>> const = model.create_set_const(object_type=obj, value=[0, 1])
    /// >>> other = model.create_set_const(object_type=obj, value=[0, 1, 2])
    /// >>> const.issubset(other).eval(state, model)
    /// True
    #[pyo3(signature = (other))]
    fn issubset(&self, other: SetUnion) -> ConditionPy {
        let lhs = SetExpression::from(self.clone().0);
        let rhs = SetExpression::from(other);
        ConditionPy(lhs.is_subset(rhs))
    }

    /// issuperset(other)
    ///
    /// Checks if this set is a superset of another set.
    ///
    /// Parameters
    /// ----------
    /// other: SetExpr, SetVar, or SetConst
    ///    The other set.
    ///
    /// Returns
    /// -------
    /// Condition
    ///    The condition that the two sets are disjoint.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> state = model.target_state
    /// >>> obj = model.add_object_type(number=4)
    /// >>> const = model.create_set_const(object_type=obj, value=[0, 1])
    /// >>> other = model.create_set_const(object_type=obj, value=[0])
    /// >>> cost.issuperset(other).eval(state, model)
    /// True
    #[pyo3(signature = (other))]
    fn issuperset(&self, other: SetUnion) -> ConditionPy {
        let lhs = SetExpression::from(self.clone().0);
        let rhs = SetExpression::from(other);
        ConditionPy(rhs.is_subset(lhs))
    }

    /// add(element)
    ///
    /// Adds an element to a set.
    ///
    /// This method does not change the instance itself.
    ///
    /// Parameters
    /// ----------
    /// element: ElementExpr, ElementVar, ElementResourceVar, or int
    ///     Element added to the set.
    /// Returns
    /// -------
    /// SetExpr
    ///     The set where the element is added.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> state = model.target_state
    /// >>> obj = model.add_object_type(number=4)
    /// >>> const = model.create_set_const(object_type=obj, value=[0, 1])
    /// >>> const.add(2).eval(state, model)
    /// {0, 1, 2}
    #[pyo3(signature = (element))]
    fn add(&self, element: ElementUnion) -> SetExprPy {
        let set = SetExpression::from(self.clone());
        let element = ElementExpression::from(element);
        SetExprPy(set.add(element))
    }

    /// remove(element)
    ///
    /// Removes an element from a set.
    ///
    /// This method does not change the instance itself.
    ///
    /// Parameters
    /// ----------
    /// element: ElementExpr, ElementVar, ElementResourceVar, or int
    ///     Element removed from the set.
    ///
    /// Returns
    /// -------
    /// SetExpr
    ///     The set where the element is removed.
    ///
    /// Raises
    /// ------
    /// OverflowError
    ///     If :code:`element` is :class:`int` and negative.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> state = model.target_state
    /// >>> obj = model.add_object_type(number=4)
    /// >>> const = model.create_set_const(object_type=obj, value=[0, 1])
    /// >>> const.remove(1).eval(state, model)
    /// {0}
    #[pyo3(signature = (element))]
    fn remove(&self, element: ElementUnion) -> SetExprPy {
        self.discard(element)
    }

    /// discard(element)
    ///
    /// Removes an element from a set.
    ///
    /// This method does not change the instance itself.
    ///
    /// Parameters
    /// ----------
    /// element: ElementExpr, ElementVar, ElementResourceVar, or int
    ///     Element removed from the set.
    ///
    /// Returns
    /// -------
    /// SetExpr
    ///     The set where the element is removed.
    ///
    /// Raises
    /// ------
    /// OverflowError
    ///     If :code:`element` is :class:`int` and negative.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> state = model.target_state
    /// >>> obj = model.add_object_type(number=4)
    /// >>> const = model.create_set_const(object_type=obj, value=[0, 1])
    /// >>> const.discard(1).eval(state, model)
    /// {0}
    #[pyo3(signature = (element))]
    fn discard(&self, element: ElementUnion) -> SetExprPy {
        let set = SetExpression::from(self.clone());
        let element = ElementExpression::from(element);
        SetExprPy(set.remove(element))
    }

    /// difference(other)
    ///
    /// Returns a set where all elements in an input set are removed.
    ///
    /// This method is the same as :code:`-` operation.
    /// This method does not change the instance itself.
    ///
    /// Parameters
    /// ----------
    /// other: SetExpr, SetVar, or SetConst
    ///     Set to remove.
    ///
    /// Returns
    /// -------
    /// SetExpr
    ///     The set where all elements in :code:`other` are removed.
    ///
    /// Raises
    /// ------
    /// OverflowError
    ///     If :code:`element` is :class:`int` and negative.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> state = model.target_state
    /// >>> obj = model.add_object_type(number=4)
    /// >>> const = model.create_set_const(object_type=obj, value=[0, 1])
    /// >>> other = model.create_set_const(object_type=obj, value=[1, 2])
    /// >>> const.difference(other).eval(state, model)
    /// {0}
    #[pyo3(signature = (other))]
    fn difference(&self, other: SetUnion) -> SetExprPy {
        self.__sub__(other)
    }

    /// intersection(other)
    ///
    /// Returns the intersection with another set.
    ///
    /// This method is the same as :code:`&` operation.
    /// This method does not change the instance itself.
    ///
    /// Parameters
    /// ----------
    /// other: SetExpr, SetVar, or SetConst
    ///     Set to take the intersection with.
    ///
    /// Returns
    /// -------
    /// SetExpr
    ///     The intersection.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> state = model.target_state
    /// >>> obj = model.add_object_type(number=4)
    /// >>> const = model.create_set_const(object_type=obj, value=[0, 1])
    /// >>> other = model.create_set_const(object_type=obj, value=[1, 2])
    /// >>> const.intersection(other).eval(state, model)
    /// {1}
    #[pyo3(signature = (other))]
    fn intersection(&self, other: SetUnion) -> SetExprPy {
        self.__and__(other)
    }

    /// symmetric_difference(other)
    ///
    /// Returns a set which only contains elements included in either of two sets but not in both.
    ///
    /// This method is the same as :code:`^` operation.
    /// This method does not change the instance itself.
    ///
    /// Parameters
    /// ----------
    /// other: SetExpr, SetVar, or SetConst
    ///     Set to take the symmetric difference with.
    ///
    /// Returns
    /// -------
    /// SetExpr
    ///     The symmetric difference set.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> state = model.target_state
    /// >>> obj = model.add_object_type(number=4)
    /// >>> const = model.create_set_const(object_type=obj, value=[0, 1])
    /// >>> other = model.create_set_const(object_type=obj, value=[1, 2])
    /// >>> const.symmetric_difference(other).eval(state, model)
    /// {0, 2}
    #[pyo3(signature = (other))]
    fn symmetric_difference(&self, other: SetUnion) -> SetExprPy {
        self.__xor__(other)
    }

    /// union(other)
    ///
    /// Returns the union of two sets.
    ///
    /// This method is the same as :code:`\|` operation.
    /// This method does not change the instance itself.
    ///
    /// Parameters
    /// ----------
    /// other: SetExpr, SetVar, or SetConst
    ///     Set to take the union with.
    ///
    /// Returns
    /// -------
    /// SetExpr
    ///     The union.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> state = model.target_state
    /// >>> obj = model.add_object_type(number=4)
    /// >>> const = model.create_set_const(object_type=obj, value=[0, 1])
    /// >>> other = model.create_set_const(object_type=obj, value=[1, 2])
    /// >>> const.union(other).eval(state, model)
    /// {0, 1, 2}
    #[pyo3(signature = (other))]
    fn union(&self, other: SetUnion) -> SetExprPy {
        self.__or__(other)
    }

    /// contains(element)
    ///
    /// Returns a condition checking if an element is included.
    ///
    /// Parameters
    /// ----------
    /// element: ElementExpr, ElementVar, ElementResourceVar, or int
    ///     Element to check.
    ///
    /// Returns
    /// -------
    /// Condition
    ///     The condition checking if an element is included in the set.
    ///
    /// Raises
    /// ------
    /// OverflowError
    ///     If :code:`element` is :class:`int` and negative.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> state = model.target_state
    /// >>> obj = model.add_object_type(number=4)
    /// >>> const = model.create_set_const(object_type=obj, value=[0, 1])
    /// >>> const.contains(0).eval(state, model)
    /// True
    #[pyo3(signature = (element))]
    fn contains(&self, element: ElementUnion) -> ConditionPy {
        let set = SetExpression::from(self.clone());
        let element = ElementExpression::from(element);
        ConditionPy(set.contains(element))
    }

    /// Returns the cardinality of a set.
    ///
    /// Returns
    /// -------
    /// IntExpr
    ///     The cardinality.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> state = model.target_state
    /// >>> obj = model.add_object_type(number=4)
    /// >>> const = model.create_set_const(object_type=obj, value=[0, 1])
    /// >>> const.len().eval(state, model)
    /// 2
    #[pyo3(signature = ())]
    fn len(&self) -> IntExprPy {
        let set = SetExpression::from(self.clone());
        IntExprPy(set.len())
    }

    /// Returns a condition checking if the set is empty.
    ///
    /// Returns
    /// -------
    /// Condition
    ///     The condition checking if the set is empty.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> state = model.target_state
    /// >>> obj = model.add_object_type(number=4)
    /// >>> const = model.create_set_const(object_type=obj, value=[0, 1])
    /// >>> const.is_empty().eval(state, model)
    /// False
    #[pyo3(signature = ())]
    fn is_empty(&self) -> ConditionPy {
        let set = SetExpression::from(self.clone());
        ConditionPy(set.is_empty())
    }

    /// Returns the complement set.
    ///
    /// Returns
    /// -------
    /// SetExpr
    ///     The complement set.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> state = model.target_state
    /// >>> obj = model.add_object_type(number=4)
    /// >>> const = model.create_set_const(object_type=obj, value=[0, 1])
    /// >>> const.complement().eval(state, model)
    /// {2, 3}
    #[pyo3(signature = ())]
    fn complement(&self) -> SetExprPy {
        let set = SetExpression::from(self.clone());
        SetExprPy(!set)
    }

    /// Returns the set.
    ///
    /// Returns
    /// -------
    /// set
    ///     The set.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> obj = model.add_object_type(number=4)
    /// >>> const = model.create_set_const(object_type=obj, value=[0, 1])
    /// >>> const.eval()
    /// {0, 1}
    #[pyo3(signature = ())]
    fn eval(&self) -> HashSet<Element> {
        HashSet::from_iter(self.0.ones())
    }
}

#[derive(FromPyObject, Debug, PartialEq, Clone)]
pub enum IntUnion {
    #[pyo3(transparent, annotation = "IntExpr")]
    Expr(IntExprPy),
    #[pyo3(transparent, annotation = "IntVar")]
    Var(IntVarPy),
    #[pyo3(transparent, annotation = "IntResourceVar")]
    ResourceVar(IntResourceVarPy),
    #[pyo3(transparent, annotation = "int")]
    Const(Integer),
}

impl From<IntUnion> for IntegerExpression {
    fn from(int: IntUnion) -> Self {
        match int {
            IntUnion::Expr(expr) => IntegerExpression::from(expr),
            IntUnion::Var(var) => IntegerExpression::from(var),
            IntUnion::ResourceVar(var) => IntegerExpression::from(var),
            IntUnion::Const(value) => IntegerExpression::from(value),
        }
    }
}

#[derive(FromPyObject, Debug, PartialEq, Clone)]
pub enum IntOrFloatUnion {
    #[pyo3(transparent, annotation = "int")]
    Int(IntUnion),
    #[pyo3(transparent, annotation = "float")]
    Float(FloatUnion),
}

#[derive(Debug, PartialEq, Clone)]
pub enum IntOrFloatExpr {
    Int(IntExprPy),
    Float(FloatExprPy),
}

impl IntoPy<Py<PyAny>> for IntOrFloatExpr {
    fn into_py(self, py: Python<'_>) -> Py<PyAny> {
        match self {
            Self::Int(expr) => expr.into_py(py),
            Self::Float(expr) => expr.into_py(py),
        }
    }
}

impl From<CostExpression> for IntOrFloatExpr {
    fn from(expr: CostExpression) -> Self {
        match expr {
            CostExpression::Integer(expr) => Self::Int(IntExprPy::from(expr)),
            CostExpression::Continuous(expr) => Self::Float(FloatExprPy::from(expr)),
        }
    }
}

/// Integer expression.
///
/// If an arithmetic operator (:code:`+`, :code:`-`, :code:`*`, :code:`//`, :code:`%`) with an :class:`IntExpr`, :class:`IntVar`, :class:`IntResourceVar`, or :class:`int` is applied, a new :class:`IntExpr` is returned.
/// For division (`/`) and power (`**`), a :class:`FloatExpr` is returned.
/// If an arithmetic operator with an :class:`FloatExpr`, :class:`FloatVar`, :class:`FloatResourceVar`, or :class:`float` is applied, a :class:`FloatExpr` is returned.
/// If :func:`abs` is applied, a new :class:`IntExpr` is returned.
///
/// If a comparison operator (:code:`<`, :code:`<=`, :code:`==`, :code:`!=`, :code:`>`, :code:`>=`) with an :class:`IntExpr`, :class:`IntVar`, :class:`IntResourceVar`, :class:`FloatExpr`, :class:`FloatVar`, :class:`FloatResourceVar`, :class:`int`, or :class:`float` is applied, a :class:`Condition` is returned.
///
/// Note that :func:`didppy.max` and :func:`didppy.min` should be used instead of :func:`~built_in.max` and :func:`~built_in.min` as comparison operators are overloaded.
///
/// Parameters
/// ----------
/// value : int
///     A value from which a constant expression is created.
///
/// Examples
/// --------
/// >>> import didppy as dp
/// >>> model = dp.Model()
/// >>> state = model.target_state
/// >>> expr = dp.IntExpr(3)
/// >>> expr.eval(state, model)
/// 3
/// >>> (-expr).eval(state, model)
/// -3
/// >>> (expr + 1).eval(state, model)
/// 4
/// >>> (expr + 1.5).eval(state, model)
/// 4.5
/// >>> (expr - 1).eval(state, model)
/// 2
/// >>> (expr * 2).eval(state, model)
/// 6
/// >>> (expr / 2).eval(state, model)
/// 1.5
/// >>> (expr // 2).eval(state, model)
/// 1
/// >>> (expr % 2).eval(state, model)
/// 1
/// >>> abs(expr).eval(state, model)
/// 3
/// >>> (expr ** 2).eval(state, model)
/// 9.0
/// >>> pow(expr, 2).eval(state, model)
/// 9.0
/// >>> (2 ** expr).eval(state, model)
/// 8.0
/// >>> pow(2, expr).eval(state, model)
/// 8.0
/// >>> (expr < 3).eval(state, model)
/// False
/// >>> (expr <= 3).eval(state, model)
/// True
/// >>> (expr == 3).eval(state, model)
/// True
/// >>> (expr != 3).eval(state, model)
/// False
/// >>> (expr > 3).eval(state, model)
/// False
/// >>> (expr >= 3).eval(state, model)
/// True
#[pyclass(name = "IntExpr")]
#[derive(Debug, PartialEq, Clone)]
pub struct IntExprPy(IntegerExpression);

impl From<IntExprPy> for IntegerExpression {
    fn from(expression: IntExprPy) -> Self {
        expression.0
    }
}

impl From<IntegerExpression> for IntExprPy {
    fn from(expression: IntegerExpression) -> Self {
        Self(expression)
    }
}

#[pymethods]
impl IntExprPy {
    #[new]
    #[pyo3(text_signature = "(value)")]
    fn new_py(value: Integer) -> Self {
        Self(IntegerExpression::from(value))
    }

    /// Returns the cost of the transitioned state, which can be used in a cost expression.
    ///
    /// Returns
    /// -------
    /// IntExpr
    ///     The cost of the transitioned state.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> state = model.target_state
    /// >>> cost = dp.IntExpr.state_cost() + 1
    /// >>> cost.eval_cost(1, state, model)
    /// 2
    #[pyo3(signature = ())]
    #[staticmethod]
    fn state_cost() -> IntExprPy {
        Self(IntegerExpression::Cost)
    }

    fn __richcmp__(&self, other: IntOrFloatUnion, op: CompareOp) -> ConditionPy {
        let lhs = self.clone().0;
        let op = match op {
            CompareOp::Lt => ComparisonOperator::Lt,
            CompareOp::Le => ComparisonOperator::Le,
            CompareOp::Eq => ComparisonOperator::Eq,
            CompareOp::Ne => ComparisonOperator::Ne,
            CompareOp::Ge => ComparisonOperator::Ge,
            CompareOp::Gt => ComparisonOperator::Gt,
        };
        match other {
            IntOrFloatUnion::Int(other) => {
                let rhs = IntegerExpression::from(other);
                ConditionPy(Condition::comparison_i(op, lhs, rhs))
            }
            IntOrFloatUnion::Float(other) => {
                let rhs = ContinuousExpression::from(other);
                ConditionPy(Condition::comparison_c(op, lhs, rhs))
            }
        }
    }

    fn __abs__(&self) -> IntExprPy {
        IntExprPy(self.clone().0.abs())
    }

    fn __neg__(&self) -> IntExprPy {
        IntExprPy(-(self.clone().0))
    }

    fn __add__(&self, other: IntOrFloatUnion) -> IntOrFloatExpr {
        match other {
            IntOrFloatUnion::Int(other) => {
                IntOrFloatExpr::Int(IntExprPy(self.clone().0 + IntegerExpression::from(other)))
            }
            IntOrFloatUnion::Float(other) => IntOrFloatExpr::Float(FloatExprPy(
                self.clone().0 + ContinuousExpression::from(other),
            )),
        }
    }

    fn __sub__(&self, other: IntOrFloatUnion) -> IntOrFloatExpr {
        match other {
            IntOrFloatUnion::Int(other) => {
                IntOrFloatExpr::Int(IntExprPy(self.clone().0 - IntegerExpression::from(other)))
            }
            IntOrFloatUnion::Float(other) => IntOrFloatExpr::Float(FloatExprPy(
                self.clone().0 - ContinuousExpression::from(other),
            )),
        }
    }

    fn __mul__(&self, other: IntOrFloatUnion) -> IntOrFloatExpr {
        match other {
            IntOrFloatUnion::Int(other) => {
                IntOrFloatExpr::Int(IntExprPy(self.clone().0 * IntegerExpression::from(other)))
            }
            IntOrFloatUnion::Float(other) => IntOrFloatExpr::Float(FloatExprPy(
                self.clone().0 * ContinuousExpression::from(other),
            )),
        }
    }

    fn __mod__(&self, other: IntOrFloatUnion) -> IntOrFloatExpr {
        match other {
            IntOrFloatUnion::Int(other) => {
                IntOrFloatExpr::Int(IntExprPy(self.clone().0 % IntegerExpression::from(other)))
            }
            IntOrFloatUnion::Float(other) => IntOrFloatExpr::Float(FloatExprPy(
                self.clone().0 % ContinuousExpression::from(other),
            )),
        }
    }

    fn __truediv__(&self, other: FloatUnion) -> FloatExprPy {
        FloatExprPy(self.clone().0 / ContinuousExpression::from(other))
    }

    fn __floordiv__(&self, other: IntOrFloatUnion) -> IntOrFloatExpr {
        match other {
            IntOrFloatUnion::Int(other) => {
                IntOrFloatExpr::Int(IntExprPy(self.clone().0 / IntegerExpression::from(other)))
            }
            IntOrFloatUnion::Float(other) => IntOrFloatExpr::Float(FloatExprPy(
                (self.clone().0 / ContinuousExpression::from(other)).floor(),
            )),
        }
    }

    fn __pow__(&self, other: FloatUnion, modulo: Option<FloatUnion>) -> FloatExprPy {
        let result = self.clone().0.pow(ContinuousExpression::from(other));
        if let Some(modulo) = modulo {
            let modulo = ContinuousExpression::from(modulo);
            FloatExprPy(result % modulo)
        } else {
            FloatExprPy(result)
        }
    }

    fn __radd__(&self, other: IntOrFloatUnion) -> IntOrFloatExpr {
        self.__add__(other)
    }

    fn __rsub__(&self, other: IntOrFloatUnion) -> IntOrFloatExpr {
        match other {
            IntOrFloatUnion::Int(other) => {
                IntOrFloatExpr::Int(IntExprPy(IntegerExpression::from(other) - self.clone().0))
            }
            IntOrFloatUnion::Float(other) => IntOrFloatExpr::Float(FloatExprPy(
                ContinuousExpression::from(other) - self.clone().0,
            )),
        }
    }

    fn __rmul__(&self, other: IntOrFloatUnion) -> IntOrFloatExpr {
        self.__mul__(other)
    }

    fn __rtruediv__(&self, other: FloatUnion) -> FloatExprPy {
        FloatExprPy(ContinuousExpression::from(other) / self.clone().0)
    }

    fn __rfloordiv__(&self, other: IntOrFloatUnion) -> IntOrFloatExpr {
        match other {
            IntOrFloatUnion::Int(other) => {
                IntOrFloatExpr::Int(IntExprPy(IntegerExpression::from(other) / self.clone().0))
            }
            IntOrFloatUnion::Float(other) => IntOrFloatExpr::Float(FloatExprPy(
                (ContinuousExpression::from(other) / self.clone().0).floor(),
            )),
        }
    }

    fn __rmod__(&self, other: IntOrFloatUnion) -> IntOrFloatExpr {
        match other {
            IntOrFloatUnion::Int(other) => {
                IntOrFloatExpr::Int(IntExprPy(IntegerExpression::from(other) % self.clone().0))
            }
            IntOrFloatUnion::Float(other) => IntOrFloatExpr::Float(FloatExprPy(
                ContinuousExpression::from(other) % self.clone().0,
            )),
        }
    }

    fn __rpow__(&self, other: FloatUnion, modulo: Option<FloatUnion>) -> FloatExprPy {
        let result = ContinuousExpression::from(other).pow(self.clone().0);
        if let Some(modulo) = modulo {
            let modulo = ContinuousExpression::from(modulo);
            FloatExprPy(result % modulo)
        } else {
            FloatExprPy(result)
        }
    }

    fn __bool__(&self) -> PyResult<bool> {
        Err(DIDPPyException::new_err(
            "IntExpr cannot be converted to bool",
        ))
    }

    /// eval(state, model)
    ///
    /// Evaluates the expression.
    ///
    /// Parameters
    /// ----------
    /// state : State
    ///     State.
    /// model : Model
    ///     DyPDL Model.
    ///
    /// Returns
    /// -------
    /// int
    ///     Value of the expression.
    ///
    /// Raises
    /// ------
    /// PanicException
    ///     If the expression is not valid.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> var = model.add_int_var(target=0)
    /// >>> expr = var + 1
    /// >>> state = model.target_state
    /// >>> expr.eval(state, model)
    /// 1
    #[pyo3(signature = (state, model))]
    fn eval(&self, state: &StatePy, model: &ModelPy) -> Integer {
        let mut function_cache =
            StateFunctionCache::new(&model.inner_as_ref().state_functions);

        self.0.eval(
            state.inner_as_ref(),
            &mut function_cache,
            &model.inner_as_ref().state_functions,
            &model.inner_as_ref().table_registry,
        )
    }

    /// eval_cost(cost, state, model)
    ///
    /// Evaluates the cost expression.
    ///
    /// Parameters
    /// ----------
    /// cost : int
    ///     State cost.
    /// state : State
    ///     State.
    /// model : Model
    ///     DyPDL Model.
    ///
    /// Returns
    /// -------
    /// int
    ///     Value of the expression.
    ///
    /// Raises
    /// ------
    /// PanicException
    ///     If the expression is not valid.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> var = model.add_int_var(target=0)
    /// >>> expr = var + dp.IntExpr.state_cost()
    /// >>> state = model.target_state
    /// >>> expr.eval_cost(1, state, model)
    /// 1
    #[pyo3(signature = (cost, state, model))]
    fn eval_cost(&self, cost: Integer, state: &StatePy, model: &ModelPy) -> Integer {
        let mut function_cache =
            StateFunctionCache::new(&model.inner_as_ref().state_functions);

        self.0.eval_cost(
            cost,
            state.inner_as_ref(),
            &mut function_cache,
            &model.inner_as_ref().state_functions,
            &model.inner_as_ref().table_registry,
        )
    }
}

/// Integer variable.
///
/// If an arithmetic operator (:code:`+`, :code:`-`, :code:`*`, :code:`//`, :code:`%`) with an :class:`IntExpr`, :class:`IntVar`, :class:`IntResourceVar`, or :class:`int` is applied, a new :class:`IntExpr` is returned.
/// For division (`/`) and power (`**`), a :class:`FloatExpr` is returned.
/// If an arithmetic operator with an :class:`FloatExpr`, :class:`FloatVar`, :class:`FloatResourceVar`, or :class:`float` is applied, a :class:`FloatExpr` is returned.
/// If :func:`abs` is applied, a new :class:`IntExpr` is returned.
///
/// If a comparison operator (:code:`<`, :code:`<=`, :code:`==`, :code:`!=`, :code:`>`, :code:`>=`) with an :class:`IntExpr`, :class:`IntVar`, :class:`IntResourceVar`, :class:`FloatExpr`, :class:`FloatVar`, :class:`FloatResourceVar`, :class:`int`, or :class:`float` is applied, a :class:`Condition` is returned.
///
/// Note that :func:`didppy.max` and :func:`didppy.min` should be used instead of :func:`~built_in.max` and :func:`~built_in.min` as comparison operators are overloaded.
///
///
/// Examples
/// --------
/// >>> import didppy as dp
/// >>> model = dp.Model()
/// >>> var = model.add_int_var(target=3)
/// >>> state = model.target_state
/// >>> state[var]
/// 3
/// >>> (-var).eval(state, model)
/// -3
/// >>> (var + 1).eval(state, model)
/// 4
/// >>> (var + 1.5).eval(state, model)
/// 4.5
/// >>> (var - 1).eval(state, model)
/// 2
/// >>> (var * 2).eval(state, model)
/// 6
/// >>> (var / 2).eval(state, model)
/// 1.5
/// >>> (var // 2).eval(state, model)
/// 1
/// >>> (var % 2).eval(state, model)
/// 1
/// >>> abs(var).eval(state, model)
/// 3
/// >>> (var ** 2).eval(state, model)
/// 9.0
/// >>> pow(var, 2).eval(state, model)
/// 9.0
/// >>> (2 ** var).eval(state, model)
/// 8.0
/// >>> pow(2, var).eval(state, model)
/// 8.0
/// >>> (var < 3).eval(state, model)
/// False
/// >>> (var <= 3).eval(state, model)
/// True
/// >>> (var == 3).eval(state, model)
/// True
/// >>> (var != 3).eval(state, model)
/// False
/// >>> (var > 3).eval(state, model)
/// False
/// >>> (var >= 3).eval(state, model)
/// True
#[pyclass(name = "IntVar")]
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct IntVarPy(IntegerVariable);

impl From<IntVarPy> for IntegerVariable {
    fn from(v: IntVarPy) -> Self {
        v.0
    }
}

impl From<IntegerVariable> for IntVarPy {
    fn from(v: IntegerVariable) -> Self {
        Self(v)
    }
}

impl From<IntVarPy> for IntegerExpression {
    fn from(v: IntVarPy) -> Self {
        v.0.into()
    }
}

#[pymethods]
impl IntVarPy {
    fn __richcmp__(&self, other: IntOrFloatUnion, op: CompareOp) -> ConditionPy {
        let lhs = self.0;
        let op = match op {
            CompareOp::Lt => ComparisonOperator::Lt,
            CompareOp::Le => ComparisonOperator::Le,
            CompareOp::Eq => ComparisonOperator::Eq,
            CompareOp::Ne => ComparisonOperator::Ne,
            CompareOp::Ge => ComparisonOperator::Ge,
            CompareOp::Gt => ComparisonOperator::Gt,
        };
        match other {
            IntOrFloatUnion::Int(other) => {
                let rhs = IntegerExpression::from(other);
                ConditionPy(Condition::comparison_i(op, lhs, rhs))
            }
            IntOrFloatUnion::Float(other) => {
                let rhs = ContinuousExpression::from(other);
                ConditionPy(Condition::comparison_c(op, lhs, rhs))
            }
        }
    }

    fn __abs__(&self) -> IntExprPy {
        IntExprPy(self.0.abs())
    }

    fn __neg__(&self) -> IntExprPy {
        IntExprPy(-(self.0))
    }

    fn __add__(&self, other: IntOrFloatUnion) -> IntOrFloatExpr {
        match other {
            IntOrFloatUnion::Int(other) => {
                IntOrFloatExpr::Int(IntExprPy(self.0 + IntegerExpression::from(other)))
            }
            IntOrFloatUnion::Float(other) => {
                IntOrFloatExpr::Float(FloatExprPy(self.0 + ContinuousExpression::from(other)))
            }
        }
    }

    fn __sub__(&self, other: IntOrFloatUnion) -> IntOrFloatExpr {
        match other {
            IntOrFloatUnion::Int(other) => {
                IntOrFloatExpr::Int(IntExprPy(self.0 - IntegerExpression::from(other)))
            }
            IntOrFloatUnion::Float(other) => {
                IntOrFloatExpr::Float(FloatExprPy(self.0 - ContinuousExpression::from(other)))
            }
        }
    }

    fn __mul__(&self, other: IntOrFloatUnion) -> IntOrFloatExpr {
        match other {
            IntOrFloatUnion::Int(other) => {
                IntOrFloatExpr::Int(IntExprPy(self.0 * IntegerExpression::from(other)))
            }
            IntOrFloatUnion::Float(other) => {
                IntOrFloatExpr::Float(FloatExprPy(self.0 * ContinuousExpression::from(other)))
            }
        }
    }

    fn __truediv__(&self, other: FloatUnion) -> FloatExprPy {
        FloatExprPy(self.0 / ContinuousExpression::from(other))
    }

    fn __floordiv__(&self, other: IntOrFloatUnion) -> IntOrFloatExpr {
        match other {
            IntOrFloatUnion::Int(other) => {
                IntOrFloatExpr::Int(IntExprPy(self.0 / IntegerExpression::from(other)))
            }
            IntOrFloatUnion::Float(other) => IntOrFloatExpr::Float(FloatExprPy(
                (self.0 / ContinuousExpression::from(other)).floor(),
            )),
        }
    }

    fn __mod__(&self, other: IntOrFloatUnion) -> IntOrFloatExpr {
        match other {
            IntOrFloatUnion::Int(other) => {
                IntOrFloatExpr::Int(IntExprPy(self.0 % IntegerExpression::from(other)))
            }
            IntOrFloatUnion::Float(other) => {
                IntOrFloatExpr::Float(FloatExprPy(self.0 % ContinuousExpression::from(other)))
            }
        }
    }

    fn __pow__(&self, other: FloatUnion, modulo: Option<FloatUnion>) -> FloatExprPy {
        let result = self.0.pow(ContinuousExpression::from(other));
        if let Some(modulo) = modulo {
            let modulo = ContinuousExpression::from(modulo);
            FloatExprPy(result % modulo)
        } else {
            FloatExprPy(result)
        }
    }

    fn __radd__(&self, other: IntOrFloatUnion) -> IntOrFloatExpr {
        self.__add__(other)
    }

    fn __rsub__(&self, other: IntOrFloatUnion) -> IntOrFloatExpr {
        match other {
            IntOrFloatUnion::Int(other) => {
                IntOrFloatExpr::Int(IntExprPy(IntegerExpression::from(other) - self.0))
            }
            IntOrFloatUnion::Float(other) => {
                IntOrFloatExpr::Float(FloatExprPy(ContinuousExpression::from(other) - self.0))
            }
        }
    }

    fn __rmul__(&self, other: IntOrFloatUnion) -> IntOrFloatExpr {
        self.__mul__(other)
    }

    fn __rtruediv__(&self, other: FloatUnion) -> FloatExprPy {
        FloatExprPy(ContinuousExpression::from(other) / self.0)
    }

    fn __rfloordiv__(&self, other: IntOrFloatUnion) -> IntOrFloatExpr {
        match other {
            IntOrFloatUnion::Int(other) => {
                IntOrFloatExpr::Int(IntExprPy(IntegerExpression::from(other) / self.0))
            }
            IntOrFloatUnion::Float(other) => IntOrFloatExpr::Float(FloatExprPy(
                (ContinuousExpression::from(other) / self.0).floor(),
            )),
        }
    }

    fn __rmod__(&self, other: IntOrFloatUnion) -> IntOrFloatExpr {
        match other {
            IntOrFloatUnion::Int(other) => {
                IntOrFloatExpr::Int(IntExprPy(IntegerExpression::from(other) % self.0))
            }
            IntOrFloatUnion::Float(other) => {
                IntOrFloatExpr::Float(FloatExprPy(ContinuousExpression::from(other) % self.0))
            }
        }
    }

    fn __rpow__(&self, other: FloatUnion, modulo: Option<FloatUnion>) -> FloatExprPy {
        let result = ContinuousExpression::from(other).pow(self.0);
        if let Some(modulo) = modulo {
            let modulo = ContinuousExpression::from(modulo);
            FloatExprPy(result % modulo)
        } else {
            FloatExprPy(result)
        }
    }

    fn __bool__(&self) -> PyResult<bool> {
        Err(DIDPPyException::new_err(
            "IntVar cannot be converted to bool",
        ))
    }
}

/// Integer resource variable.
///
/// Intuitively, with :code:`less_is_better=True`/:code:`less_is_better=False`, if everything else is the same, a state having a smaller/greater value is better.
/// Formally, if the values of non-resource variables are the same, a state having equal or better resource variable values must lead to an equal or better solution that has equal or fewer transitions than the other.
///
/// If an arithmetic operator (:code:`+`, :code:`-`, :code:`*`, :code:`//`, :code:`%`) with an :class:`IntExpr`, :class:`IntVar`, :class:`IntResourceVar`, or :class:`int` is applied, a new :class:`IntExpr` is returned.
/// For division (`/`) and power (`**`), a :class:`FloatExpr` is returned.
/// If an arithmetic operator with an :class:`FloatExpr`, :class:`FloatVar`, :class:`FloatResourceVar`, or :class:`float` is applied, a :class:`FloatExpr` is returned.
/// If :func:`abs` is applied, a new :class:`IntExpr` is returned.
///
/// If a comparison operator (:code:`<`, :code:`<=`, :code:`==`, :code:`!=`, :code:`>`, :code:`>=`) with an :class:`IntExpr`, :class:`IntVar`, :class:`IntResourceVar`, :class:`FloatExpr`, :class:`FloatVar`, :class:`FloatResourceVar`, :class:`int`, or :class:`float` is applied, a :class:`Condition` is returned.
///
/// Note that :func:`didppy.max` and :func:`didppy.min` should be used instead of :func:`~built_in.max` and :func:`~built_in.min` as comparison operators are overloaded.
///
///
/// Examples
/// --------
/// >>> import didppy as dp
/// >>> model = dp.Model()
/// >>> var = model.add_int_resource_var(target=3, less_is_better=True)
/// >>> state = model.target_state
/// >>> state[var]
/// 3
/// >>> (-var).eval(state, model)
/// -3
/// >>> (var + 1).eval(state, model)
/// 4
/// >>> (var + 1.5).eval(state, model)
/// 4.5
/// >>> (var - 1).eval(state, model)
/// 2
/// >>> (var * 2).eval(state, model)
/// 6
/// >>> (var / 2).eval(state, model)
/// 1.5
/// >>> (var // 2).eval(state, model)
/// 1
/// >>> (var % 2).eval(state, model)
/// 1
/// >>> abs(var).eval(state, model)
/// 3
/// >>> (var ** 2).eval(state, model)
/// 9.0
/// >>> pow(var, 2).eval(state, model)
/// 9.0
/// >>> (2 ** var).eval(state, model)
/// 8.0
/// >>> pow(2, var).eval(state, model)
/// 8.0
/// >>> (var < 3).eval(state, model)
/// False
/// >>> (var <= 3).eval(state, model)
/// True
/// >>> (var == 3).eval(state, model)
/// True
/// >>> (var != 3).eval(state, model)
/// False
/// >>> (var > 3).eval(state, model)
/// False
/// >>> (var >= 3).eval(state, model)
/// True
#[pyclass(name = "IntResourceVar")]
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct IntResourceVarPy(IntegerResourceVariable);

impl From<IntResourceVarPy> for IntegerResourceVariable {
    fn from(v: IntResourceVarPy) -> Self {
        v.0
    }
}

impl From<IntegerResourceVariable> for IntResourceVarPy {
    fn from(v: IntegerResourceVariable) -> Self {
        Self(v)
    }
}

impl From<IntResourceVarPy> for IntegerExpression {
    fn from(v: IntResourceVarPy) -> Self {
        v.0.into()
    }
}

#[pymethods]
impl IntResourceVarPy {
    fn __richcmp__(&self, other: IntOrFloatUnion, op: CompareOp) -> ConditionPy {
        let lhs = self.0;
        let op = match op {
            CompareOp::Lt => ComparisonOperator::Lt,
            CompareOp::Le => ComparisonOperator::Le,
            CompareOp::Eq => ComparisonOperator::Eq,
            CompareOp::Ne => ComparisonOperator::Ne,
            CompareOp::Ge => ComparisonOperator::Ge,
            CompareOp::Gt => ComparisonOperator::Gt,
        };
        match other {
            IntOrFloatUnion::Int(other) => {
                let rhs = IntegerExpression::from(other);
                ConditionPy(Condition::comparison_i(op, lhs, rhs))
            }
            IntOrFloatUnion::Float(other) => {
                let rhs = ContinuousExpression::from(other);
                ConditionPy(Condition::comparison_c(op, lhs, rhs))
            }
        }
    }

    fn __abs__(&self) -> IntExprPy {
        IntExprPy(self.0.abs())
    }

    fn __neg__(&self) -> IntExprPy {
        IntExprPy(-(self.0))
    }

    fn __add__(&self, other: IntOrFloatUnion) -> IntOrFloatExpr {
        match other {
            IntOrFloatUnion::Int(other) => {
                IntOrFloatExpr::Int(IntExprPy(self.0 + IntegerExpression::from(other)))
            }
            IntOrFloatUnion::Float(other) => {
                IntOrFloatExpr::Float(FloatExprPy(self.0 + ContinuousExpression::from(other)))
            }
        }
    }

    fn __sub__(&self, other: IntOrFloatUnion) -> IntOrFloatExpr {
        match other {
            IntOrFloatUnion::Int(other) => {
                IntOrFloatExpr::Int(IntExprPy(self.0 - IntegerExpression::from(other)))
            }
            IntOrFloatUnion::Float(other) => {
                IntOrFloatExpr::Float(FloatExprPy(self.0 - ContinuousExpression::from(other)))
            }
        }
    }

    fn __mul__(&self, other: IntOrFloatUnion) -> IntOrFloatExpr {
        match other {
            IntOrFloatUnion::Int(other) => {
                IntOrFloatExpr::Int(IntExprPy(self.0 * IntegerExpression::from(other)))
            }
            IntOrFloatUnion::Float(other) => {
                IntOrFloatExpr::Float(FloatExprPy(self.0 * ContinuousExpression::from(other)))
            }
        }
    }

    fn __truediv__(&self, other: FloatUnion) -> FloatExprPy {
        FloatExprPy(self.0 / ContinuousExpression::from(other))
    }

    fn __floordiv__(&self, other: IntOrFloatUnion) -> IntOrFloatExpr {
        match other {
            IntOrFloatUnion::Int(other) => {
                IntOrFloatExpr::Int(IntExprPy(self.0 / IntegerExpression::from(other)))
            }
            IntOrFloatUnion::Float(other) => IntOrFloatExpr::Float(FloatExprPy(
                (self.0 / ContinuousExpression::from(other)).floor(),
            )),
        }
    }

    fn __mod__(&self, other: IntOrFloatUnion) -> IntOrFloatExpr {
        match other {
            IntOrFloatUnion::Int(other) => {
                IntOrFloatExpr::Int(IntExprPy(self.0 % IntegerExpression::from(other)))
            }
            IntOrFloatUnion::Float(other) => {
                IntOrFloatExpr::Float(FloatExprPy(self.0 % ContinuousExpression::from(other)))
            }
        }
    }

    fn __pow__(&self, other: FloatUnion, modulo: Option<FloatUnion>) -> FloatExprPy {
        let result = self.0.pow(ContinuousExpression::from(other));
        if let Some(modulo) = modulo {
            let modulo = ContinuousExpression::from(modulo);
            FloatExprPy(result % modulo)
        } else {
            FloatExprPy(result)
        }
    }

    fn __radd__(&self, other: IntOrFloatUnion) -> IntOrFloatExpr {
        self.__add__(other)
    }

    fn __rsub__(&self, other: IntOrFloatUnion) -> IntOrFloatExpr {
        match other {
            IntOrFloatUnion::Int(other) => {
                IntOrFloatExpr::Int(IntExprPy(IntegerExpression::from(other) - self.0))
            }
            IntOrFloatUnion::Float(other) => {
                IntOrFloatExpr::Float(FloatExprPy(ContinuousExpression::from(other) - self.0))
            }
        }
    }

    fn __rmul__(&self, other: IntOrFloatUnion) -> IntOrFloatExpr {
        self.__mul__(other)
    }

    fn __rtruediv__(&self, other: FloatUnion) -> FloatExprPy {
        FloatExprPy(ContinuousExpression::from(other) / self.0)
    }

    fn __rfloordiv__(&self, other: IntOrFloatUnion) -> IntOrFloatExpr {
        match other {
            IntOrFloatUnion::Int(other) => {
                IntOrFloatExpr::Int(IntExprPy(IntegerExpression::from(other) / self.0))
            }
            IntOrFloatUnion::Float(other) => IntOrFloatExpr::Float(FloatExprPy(
                (ContinuousExpression::from(other) / self.0).floor(),
            )),
        }
    }

    fn __rmod__(&self, other: IntOrFloatUnion) -> IntOrFloatExpr {
        match other {
            IntOrFloatUnion::Int(other) => {
                IntOrFloatExpr::Int(IntExprPy(IntegerExpression::from(other) % self.0))
            }
            IntOrFloatUnion::Float(other) => {
                IntOrFloatExpr::Float(FloatExprPy(ContinuousExpression::from(other) % self.0))
            }
        }
    }

    fn __rpow__(&self, other: FloatUnion, modulo: Option<FloatUnion>) -> FloatExprPy {
        let result = ContinuousExpression::from(other).pow(self.0);
        if let Some(modulo) = modulo {
            let modulo = ContinuousExpression::from(modulo);
            FloatExprPy(result % modulo)
        } else {
            FloatExprPy(result)
        }
    }

    fn __bool__(&self) -> PyResult<bool> {
        Err(DIDPPyException::new_err(
            "IntResourceVar cannot be converted to bool",
        ))
    }
}

#[derive(FromPyObject, Debug, PartialEq, Clone)]
pub enum FloatUnion {
    #[pyo3(transparent, annotation = "FloatExpr")]
    Expr(FloatExprPy),
    #[pyo3(transparent, annotation = "IntExpr")]
    IntExpr(IntExprPy),
    #[pyo3(transparent, annotation = "FloatVar")]
    Var(FloatVarPy),
    #[pyo3(transparent, annotation = "FloatResourceVar")]
    ResourceVar(FloatResourceVarPy),
    #[pyo3(transparent, annotation = "IntVar")]
    IntVar(IntVarPy),
    #[pyo3(transparent, annotation = "IntResourceVar")]
    IntResourceVar(IntResourceVarPy),
    #[pyo3(transparent, annotation = "float")]
    Const(Continuous),
    #[pyo3(transparent, annotation = "int")]
    IntConst(Integer),
}

impl From<FloatUnion> for ContinuousExpression {
    fn from(float: FloatUnion) -> Self {
        match float {
            FloatUnion::Expr(expr) => ContinuousExpression::from(expr),
            FloatUnion::IntExpr(expr) => ContinuousExpression::from(expr.0),
            FloatUnion::Var(var) => ContinuousExpression::from(var),
            FloatUnion::ResourceVar(var) => ContinuousExpression::from(var),
            FloatUnion::IntVar(var) => ContinuousExpression::from(var.0),
            FloatUnion::IntResourceVar(var) => ContinuousExpression::from(var.0),
            FloatUnion::Const(value) => ContinuousExpression::from(value),
            FloatUnion::IntConst(value) => ContinuousExpression::from(value),
        }
    }
}

/// Continuous expression.
///
/// If an arithmetic operator (:code:`+`, :code:`-`, :code:`*`, :code:`/`, :code:`//`, :code:`%`, :code:`**`) with an :class:`IntExpr`, :class:`IntVar`, :class:`IntResourceVar`, :class:`FloatExpr`, :class:`FloatVar`, :class:`FloatResourceVar`, :class:`int`, or :class:`float` is applied, a new :class:`FloatExpr` is returned.
/// If :func:`abs` is applied, a new :class:`FloatExpr` is returned.
/// :func:`round`, :func:`trunc`, :func:`floor`, and :func:`ceil` return an :class:`IntExpr`.
///
/// If a comparison operator (:code:`<`, :code:`<=`, :code:`==`, :code:`!=`, :code:`>`, :code:`>=`) with an :class:`IntExpr`, :class:`IntVar`, :class:`IntResourceVar`, :class:`FloatExpr`, :class:`FloatVar`, :class:`FloatResourceVar`, :class:`int`, or :class:`float` is applied, a :class:`Condition` is returned.
///
/// Note that :class:`didppy.max` and :class:`didppy.min` should be used instead of :func:`~built_in.max` and :func:`~built_in.min` as comparison operators are overloaded.
///
/// Parameters
/// ----------
/// value : float
///     A value from which a constant expression is created.
///
/// Examples
/// --------
/// >>> import didppy as dp
/// >>> model = dp.Model()
/// >>> state = model.target_state
/// >>> expr = dp.FloatExpr(3.5)
/// >>> expr.eval(state, model)
/// 3.5
/// >>> (-expr).eval(state, model)
/// -3.5
/// >>> (expr + 1.5).eval(state, model)
/// 5.0
/// >>> (expr + 1).eval(state, model)
/// 4.5
/// >>> (expr - 1.5).eval(state, model)
/// 2.0
/// >>> (expr * 2.5).eval(state, model)
/// 8.75
/// >>> (expr / 2.5).eval(state, model)
/// 1.4
/// >>> (expr // 2.5).eval(state, model)
/// 1.0
/// >>> (expr % 2.5).eval(state, model)
/// 1.0
/// >>> abs(expr).eval(state, model)
/// 3.5
/// >>> (expr ** 2.0).eval(state, model)
/// 12.25
/// >>> pow(expr, 2.0).eval(state, model)
/// 12.25
/// >>> (1.0 ** expr).eval(state, model)
/// 1.0
/// >>> pow(1.0, expr).eval(state, model)
/// 1.0
/// >>> round(expr).eval(state, model)
/// 4
/// >>> import math
/// >>> math.trunc(expr).eval(state, model)
/// 3
/// >>> math.floor(expr).eval(state, model)
/// 3
/// >>> math.ceil(expr).eval(state, model)
/// 4
/// >>> (expr < 3.0).eval(state, model)
/// False
/// >>> (expr > 3.0).eval(state, model)
/// True
#[pyclass(name = "FloatExpr")]
#[derive(Debug, PartialEq, Clone)]
pub struct FloatExprPy(ContinuousExpression);

impl From<FloatExprPy> for ContinuousExpression {
    fn from(expression: FloatExprPy) -> Self {
        expression.0
    }
}

impl From<ContinuousExpression> for FloatExprPy {
    fn from(expression: ContinuousExpression) -> Self {
        Self(expression)
    }
}

#[pymethods]
impl FloatExprPy {
    #[pyo3(text_signature = "(value)")]
    #[new]
    fn new_py(value: Continuous) -> Self {
        Self(ContinuousExpression::from(value))
    }

    /// Returns the cost of the transitioned state, which can be used in a cost expression.
    ///
    /// Returns
    /// -------
    /// FloatExpr
    ///     The cost of the transitioned state.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> state = model.target_state
    /// >>> cost = dp.FloatExpr.state_cost() + 1.5
    /// >>> cost.eval_cost(1.5, state, model)
    /// 3.0
    #[pyo3(signature = ())]
    #[staticmethod]
    fn state_cost() -> FloatExprPy {
        Self(ContinuousExpression::Cost)
    }

    fn __richcmp__(&self, other: FloatUnion, op: CompareOp) -> ConditionPy {
        let lhs = self.clone().0;
        let rhs = ContinuousExpression::from(other);
        let op = match op {
            CompareOp::Lt => ComparisonOperator::Lt,
            CompareOp::Le => ComparisonOperator::Le,
            CompareOp::Eq => ComparisonOperator::Eq,
            CompareOp::Ne => ComparisonOperator::Ne,
            CompareOp::Ge => ComparisonOperator::Ge,
            CompareOp::Gt => ComparisonOperator::Gt,
        };
        ConditionPy(Condition::comparison_c(op, lhs, rhs))
    }

    fn __abs__(&self) -> FloatExprPy {
        FloatExprPy(self.clone().0.abs())
    }

    fn __neg__(&self) -> FloatExprPy {
        FloatExprPy(-(self.clone().0))
    }

    fn __add__(&self, other: FloatUnion) -> FloatExprPy {
        FloatExprPy(self.clone().0 + ContinuousExpression::from(other))
    }

    fn __sub__(&self, other: FloatUnion) -> FloatExprPy {
        FloatExprPy(self.clone().0 - ContinuousExpression::from(other))
    }

    fn __mul__(&self, other: FloatUnion) -> FloatExprPy {
        FloatExprPy(self.clone().0 * ContinuousExpression::from(other))
    }

    fn __truediv__(&self, other: FloatUnion) -> FloatExprPy {
        FloatExprPy(self.clone().0 / ContinuousExpression::from(other))
    }

    fn __floordiv__(&self, other: FloatUnion) -> FloatExprPy {
        FloatExprPy(self.__truediv__(other).0.floor())
    }

    fn __mod__(&self, other: FloatUnion) -> FloatExprPy {
        FloatExprPy(self.clone().0 % ContinuousExpression::from(other))
    }

    fn __pow__(&self, other: FloatUnion, modulo: Option<FloatUnion>) -> FloatExprPy {
        let result = self.clone().0.pow(ContinuousExpression::from(other));
        if let Some(modulo) = modulo {
            let modulo = ContinuousExpression::from(modulo);
            FloatExprPy(result % modulo)
        } else {
            FloatExprPy(result)
        }
    }

    fn __radd__(&self, other: FloatUnion) -> FloatExprPy {
        self.__add__(other)
    }

    fn __rsub__(&self, other: FloatUnion) -> FloatExprPy {
        FloatExprPy(ContinuousExpression::from(other) - self.clone().0)
    }

    fn __rmul__(&self, other: FloatUnion) -> FloatExprPy {
        self.__mul__(other)
    }

    fn __rtruediv__(&self, other: FloatUnion) -> FloatExprPy {
        FloatExprPy(ContinuousExpression::from(other) / self.clone().0)
    }

    fn __rfloordiv__(&self, other: FloatUnion) -> FloatExprPy {
        FloatExprPy(self.__rtruediv__(other).0.floor())
    }

    fn __rmod__(&self, other: FloatUnion) -> FloatExprPy {
        FloatExprPy(ContinuousExpression::from(other) % self.clone().0)
    }

    fn __rpow__(&self, other: FloatUnion, modulo: Option<FloatUnion>) -> FloatExprPy {
        let result = ContinuousExpression::from(other).pow(self.clone().0);
        if let Some(modulo) = modulo {
            let modulo = ContinuousExpression::from(modulo);
            FloatExprPy(result % modulo)
        } else {
            FloatExprPy(result)
        }
    }

    fn __round__(&self) -> IntExprPy {
        IntExprPy(IntegerExpression::round(self.0.clone()))
    }

    fn __trunc__(&self) -> IntExprPy {
        IntExprPy(IntegerExpression::trunc(self.0.clone()))
    }

    fn __floor__(&self) -> IntExprPy {
        IntExprPy(IntegerExpression::floor(self.0.clone()))
    }

    fn __ceil__(&self) -> IntExprPy {
        IntExprPy(IntegerExpression::ceil(self.0.clone()))
    }

    fn __bool__(&self) -> PyResult<bool> {
        Err(DIDPPyException::new_err(
            "FloatExpr cannot be converted to bool",
        ))
    }

    /// eval(state, model)
    ///
    /// Evaluates the expression.
    ///
    /// Parameters
    /// ----------
    /// state : State
    ///     State.
    /// model : Model
    ///     DyPDL Model.
    ///
    /// Returns
    /// -------
    /// float
    ///     Value of the expression.
    ///
    /// Raises
    /// ------
    /// PanicException
    ///     If the expression is not valid.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> var = model.add_float_var(target=1.5)
    /// >>> expr = var + 1.5
    /// >>> state = model.target_state
    /// >>> expr.eval(state, model)
    /// 3.0
    #[pyo3(signature = (state, model))]
    fn eval(&self, state: &StatePy, model: &ModelPy) -> Continuous {
        let mut function_cache =
            StateFunctionCache::new(&model.inner_as_ref().state_functions);

        self.0.eval(
            state.inner_as_ref(),
            &mut function_cache,
            &model.inner_as_ref().state_functions,
            &model.inner_as_ref().table_registry,
        )
    }

    /// eval_cost(cost, state, model)
    ///
    /// Evaluates the cost expression.
    ///
    /// Parameters
    /// ----------
    /// cost : float
    ///     State cost.
    /// state : State
    ///     State.
    /// model : Model
    ///     DyPDL Model.
    ///
    /// Returns
    /// -------
    /// float
    ///     Value of the expression.
    ///
    /// Raises
    /// ------
    /// PanicException
    ///     If the expression is not valid.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> var = model.add_float_var(target=1.5)
    /// >>> expr = var + dp.FloatExpr.state_cost()
    /// >>> state = model.target_state
    /// >>> expr.eval_cost(1.5, state, model)
    /// 3.0
    #[pyo3(signature = (cost, state, model))]
    fn eval_cost(&self, cost: Continuous, state: &StatePy, model: &ModelPy) -> Continuous {
        let mut function_cache =
            StateFunctionCache::new(&model.inner_as_ref().state_functions);

        self.0.eval_cost(
            cost,
            state.inner_as_ref(),
            &mut function_cache,
            &model.inner_as_ref().state_functions,
            &model.inner_as_ref().table_registry,
        )
    }
}

/// Continuous variable.
///
/// If an arithmetic operator (:code:`+`, :code:`-`, :code:`*`, :code:`/`, :code:`//`, :code:`%`, :code:`**`) with an :class:`IntExpr`, :class:`IntVar`, :class:`IntResourceVar`, :class:`FloatExpr`, :class:`FloatVar`, :class:`FloatResourceVar`, :class:`int`, or :class:`float` is applied, a new :class:`FloatExpr` is returned.
/// If :func:`abs` is applied, a new :class:`FloatExpr` is returned.
/// :func:`round`, :func:`trunc`, :func:`floor`, and :func:`ceil` return an :class:`IntExpr`.
///
/// If a comparison operator (:code:`<`, :code:`<=`, :code:`==`, :code:`!=`, :code:`>`, :code:`>=`) with an :class:`IntExpr`, :class:`IntVar`, :class:`IntResourceVar`, :class:`FloatExpr`, :class:`FloatVar`, :class:`FloatResourceVar`, :class:`int`, or :class:`float` is applied, a :class:`Condition` is returned.
///
/// Note that :class:`didppy.max` and :class:`didppy.min` should be used instead of :func:`~built_in.max` and :func:`~built_in.min` as comparison operators are overloaded.
///
/// Examples
/// --------
/// >>> import didppy as dp
/// >>> model = dp.Model()
/// >>> var = model.add_float_var(target=3.5)
/// >>> state = model.target_state
/// >>> state[var]
/// 3.5
/// >>> (-var).eval(state, model)
/// -3.5
/// >>> (var + 1.5).eval(state, model)
/// 5.0
/// >>> (var + 1).eval(state, model)
/// 4.5
/// >>> (var - 1.5).eval(state, model)
/// 2.0
/// >>> (var * 2.5).eval(state, model)
/// 8.75
/// >>> (var / 2.5).eval(state, model)
/// 1.4
/// >>> (var // 2.5).eval(state, model)
/// 1.0
/// >>> (var % 2.5).eval(state, model)
/// 1.0
/// >>> abs(var).eval(state, model)
/// 3.5
/// >>> (var ** 2.0).eval(state, model)
/// 12.25
/// >>> pow(var, 2.0).eval(state, model)
/// 12.25
/// >>> (1.0 ** var).eval(state, model)
/// 1.0
/// >>> pow(1.0, var).eval(state, model)
/// 1.0
/// >>> round(var).eval(state, model)
/// 4
/// >>> import math
/// >>> math.trunc(var).eval(state, model)
/// 3
/// >>> math.floor(var).eval(state, model)
/// 3
/// >>> math.ceil(var).eval(state, model)
/// 4
/// >>> (var < 3.0).eval(state, model)
/// False
/// >>> (var > 3.0).eval(state, model)
/// True
#[pyclass(name = "FloatVar")]
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct FloatVarPy(ContinuousVariable);

impl From<FloatVarPy> for ContinuousVariable {
    fn from(v: FloatVarPy) -> Self {
        v.0
    }
}

impl From<ContinuousVariable> for FloatVarPy {
    fn from(v: ContinuousVariable) -> Self {
        Self(v)
    }
}

impl From<FloatVarPy> for ContinuousExpression {
    fn from(v: FloatVarPy) -> Self {
        v.0.into()
    }
}

#[pymethods]
impl FloatVarPy {
    fn __richcmp__(&self, other: FloatUnion, op: CompareOp) -> ConditionPy {
        let lhs = self.0;
        let rhs = ContinuousExpression::from(other);
        let op = match op {
            CompareOp::Lt => ComparisonOperator::Lt,
            CompareOp::Le => ComparisonOperator::Le,
            CompareOp::Eq => ComparisonOperator::Eq,
            CompareOp::Ne => ComparisonOperator::Ne,
            CompareOp::Ge => ComparisonOperator::Ge,
            CompareOp::Gt => ComparisonOperator::Gt,
        };
        ConditionPy(Condition::comparison_c(op, lhs, rhs))
    }

    fn __abs__(&self) -> FloatExprPy {
        FloatExprPy(self.0.abs())
    }

    fn __neg__(&self) -> FloatExprPy {
        FloatExprPy(-(self.0))
    }

    fn __add__(&self, other: FloatUnion) -> FloatExprPy {
        FloatExprPy(self.0 + ContinuousExpression::from(other))
    }

    fn __sub__(&self, other: FloatUnion) -> FloatExprPy {
        FloatExprPy(self.0 - ContinuousExpression::from(other))
    }

    fn __mul__(&self, other: FloatUnion) -> FloatExprPy {
        FloatExprPy(self.0 * ContinuousExpression::from(other))
    }

    fn __truediv__(&self, other: FloatUnion) -> FloatExprPy {
        FloatExprPy(self.0 / ContinuousExpression::from(other))
    }

    fn __floordiv__(&self, other: FloatUnion) -> FloatExprPy {
        FloatExprPy(self.__truediv__(other).0.floor())
    }

    fn __mod__(&self, other: FloatUnion) -> FloatExprPy {
        FloatExprPy(self.0 % ContinuousExpression::from(other))
    }

    fn __pow__(&self, other: FloatUnion, modulo: Option<FloatUnion>) -> FloatExprPy {
        let result = self.0.pow(ContinuousExpression::from(other));
        if let Some(modulo) = modulo {
            let modulo = ContinuousExpression::from(modulo);
            FloatExprPy(result % modulo)
        } else {
            FloatExprPy(result)
        }
    }

    fn __radd__(&self, other: FloatUnion) -> FloatExprPy {
        self.__add__(other)
    }

    fn __rsub__(&self, other: FloatUnion) -> FloatExprPy {
        FloatExprPy(ContinuousExpression::from(other) - self.0)
    }

    fn __rmul__(&self, other: FloatUnion) -> FloatExprPy {
        self.__mul__(other)
    }

    fn __rtruediv__(&self, other: FloatUnion) -> FloatExprPy {
        FloatExprPy(ContinuousExpression::from(other) / self.0)
    }

    fn __rfloordiv__(&self, other: FloatUnion) -> FloatExprPy {
        FloatExprPy(self.__rtruediv__(other).0.floor())
    }

    fn __rmod__(&self, other: FloatUnion) -> FloatExprPy {
        FloatExprPy(ContinuousExpression::from(other) % self.0)
    }

    fn __rpow__(&self, other: FloatUnion, modulo: Option<FloatUnion>) -> FloatExprPy {
        let result = ContinuousExpression::from(other).pow(self.0);
        if let Some(modulo) = modulo {
            let modulo = ContinuousExpression::from(modulo);
            FloatExprPy(result % modulo)
        } else {
            FloatExprPy(result)
        }
    }

    fn __round__(&self) -> IntExprPy {
        IntExprPy(IntegerExpression::round(self.0))
    }

    fn __trunc__(&self) -> IntExprPy {
        IntExprPy(IntegerExpression::trunc(self.0))
    }

    fn __floor__(&self) -> IntExprPy {
        IntExprPy(IntegerExpression::floor(self.0))
    }

    fn __ceil__(&self) -> IntExprPy {
        IntExprPy(IntegerExpression::ceil(self.0))
    }

    fn __bool__(&self) -> PyResult<bool> {
        Err(DIDPPyException::new_err(
            "FloatVar cannot be converted to bool",
        ))
    }
}

/// Continuous resource variable.
///
/// Intuitively, with :code:`less_is_better=True`/:code:`less_is_better=False`, if everything else is the same, a state having a smaller/greater value is better.
/// Formally, if the values of non-resource variables are the same, a state having equal or better resource variable values must lead to an equal or better solution that has equal or fewer transitions than the other.
///
/// If an arithmetic operator (:code:`+`, :code:`-`, :code:`*`, :code:`/`, :code:`//`, :code:`%`, :code:`**`) with an :class:`IntExpr`, :class:`IntVar`, :class:`IntResourceVar`, :class:`FloatExpr`, :class:`FloatVar`, :class:`FloatResourceVar`, :class:`int`, or :class:`float` is applied, a new :class:`FloatExpr` is returned.
/// If :func:`abs` is applied, a new :class:`FloatExpr` is returned.
/// :func:`round`, :func:`trunc`, :func:`floor`, and :func:`ceil` return an :class:`IntExpr`.
///
/// If a comparison operator (:code:`<`, :code:`<=`, :code:`==`, :code:`!=`, :code:`>`, :code:`>=`) with an :class:`IntExpr`, :class:`IntVar`, :class:`IntResourceVar`, :class:`FloatExpr`, :class:`FloatVar`, :class:`FloatResourceVar`, :class:`int`, or :class:`float` is applied, a :class:`Condition` is returned.
///
/// Note that :class:`didppy.max` and :class:`didppy.min` should be used instead of :func:`~built_in.max` and :func:`~built_in.min` as comparison operators are overloaded.
///
/// Examples
/// --------
/// >>> import didppy as dp
/// >>> model = dp.Model()
/// >>> var = model.add_float_resource_var(target=3.5, less_is_better=True)
/// >>> state = model.target_state
/// >>> state[var]
/// 3.5
/// >>> (-var).eval(state, model)
/// -3.5
/// >>> (var + 1.5).eval(state, model)
/// 5.0
/// >>> (var + 1).eval(state, model)
/// 4.5
/// >>> (var - 1.5).eval(state, model)
/// 2.0
/// >>> (var * 2.5).eval(state, model)
/// 8.75
/// >>> (var / 2.5).eval(state, model)
/// 1.4
/// >>> (var // 2.5).eval(state, model)
/// 1.0
/// >>> (var % 2.5).eval(state, model)
/// 1.0
/// >>> abs(var).eval(state, model)
/// 3.5
/// >>> (var ** 2.0).eval(state, model)
/// 12.25
/// >>> pow(var, 2.0).eval(state, model)
/// 12.25
/// >>> (1.0 ** var).eval(state, model)
/// 1.0
/// >>> pow(1.0, var).eval(state, model)
/// 1.0
/// >>> round(var).eval(state, model)
/// 4
/// >>> import math
/// >>> math.trunc(var).eval(state, model)
/// 3
/// >>> math.floor(var).eval(state, model)
/// 3
/// >>> math.ceil(var).eval(state, model)
/// 4
/// >>> (var < 3.0).eval(state, model)
/// False
/// >>> (var > 3.0).eval(state, model)
/// True
#[pyclass(name = "FloatResourceVar")]
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct FloatResourceVarPy(ContinuousResourceVariable);

impl From<FloatResourceVarPy> for ContinuousResourceVariable {
    fn from(v: FloatResourceVarPy) -> Self {
        v.0
    }
}

impl From<ContinuousResourceVariable> for FloatResourceVarPy {
    fn from(v: ContinuousResourceVariable) -> Self {
        Self(v)
    }
}

impl From<FloatResourceVarPy> for ContinuousExpression {
    fn from(v: FloatResourceVarPy) -> Self {
        v.0.into()
    }
}

#[pymethods]
impl FloatResourceVarPy {
    fn __richcmp__(&self, other: FloatUnion, op: CompareOp) -> ConditionPy {
        let lhs = self.0;
        let rhs = ContinuousExpression::from(other);
        let op = match op {
            CompareOp::Lt => ComparisonOperator::Lt,
            CompareOp::Le => ComparisonOperator::Le,
            CompareOp::Eq => ComparisonOperator::Eq,
            CompareOp::Ne => ComparisonOperator::Ne,
            CompareOp::Ge => ComparisonOperator::Ge,
            CompareOp::Gt => ComparisonOperator::Gt,
        };
        ConditionPy(Condition::comparison_c(op, lhs, rhs))
    }

    fn __abs__(&self) -> FloatExprPy {
        FloatExprPy(self.0.abs())
    }

    fn __neg__(&self) -> FloatExprPy {
        FloatExprPy(-(self.0))
    }

    fn __add__(&self, other: FloatUnion) -> FloatExprPy {
        FloatExprPy(self.0 + ContinuousExpression::from(other))
    }

    fn __sub__(&self, other: FloatUnion) -> FloatExprPy {
        FloatExprPy(self.0 - ContinuousExpression::from(other))
    }

    fn __mul__(&self, other: FloatUnion) -> FloatExprPy {
        FloatExprPy(self.0 * ContinuousExpression::from(other))
    }

    fn __truediv__(&self, other: FloatUnion) -> FloatExprPy {
        FloatExprPy(self.0 / ContinuousExpression::from(other))
    }

    fn __floordiv__(&self, other: FloatUnion) -> FloatExprPy {
        FloatExprPy(self.__truediv__(other).0.floor())
    }

    fn __mod__(&self, other: FloatUnion) -> FloatExprPy {
        FloatExprPy(self.0 % ContinuousExpression::from(other))
    }

    fn __pow__(&self, other: FloatUnion, modulo: Option<FloatUnion>) -> FloatExprPy {
        let result = self.0.pow(ContinuousExpression::from(other));
        if let Some(modulo) = modulo {
            let modulo = ContinuousExpression::from(modulo);
            FloatExprPy(result % modulo)
        } else {
            FloatExprPy(result)
        }
    }

    fn __radd__(&self, other: FloatUnion) -> FloatExprPy {
        self.__add__(other)
    }

    fn __rsub__(&self, other: FloatUnion) -> FloatExprPy {
        FloatExprPy(ContinuousExpression::from(other) - self.0)
    }

    fn __rmul__(&self, other: FloatUnion) -> FloatExprPy {
        self.__mul__(other)
    }

    fn __rtruediv__(&self, other: FloatUnion) -> FloatExprPy {
        FloatExprPy(ContinuousExpression::from(other) / self.0)
    }

    fn __rfloordiv__(&self, other: FloatUnion) -> FloatExprPy {
        FloatExprPy(self.__rtruediv__(other).0.floor())
    }

    fn __rmod__(&self, other: FloatUnion) -> FloatExprPy {
        FloatExprPy(ContinuousExpression::from(other) % self.0)
    }

    fn __rpow__(&self, other: FloatUnion, modulo: Option<FloatUnion>) -> FloatExprPy {
        let result = ContinuousExpression::from(other).pow(self.0);
        if let Some(modulo) = modulo {
            let modulo = ContinuousExpression::from(modulo);
            FloatExprPy(result % modulo)
        } else {
            FloatExprPy(result)
        }
    }

    fn __round__(&self) -> IntExprPy {
        IntExprPy(IntegerExpression::round(self.0))
    }

    fn __trunc__(&self) -> IntExprPy {
        IntExprPy(IntegerExpression::trunc(self.0))
    }

    fn __floor__(&self) -> IntExprPy {
        IntExprPy(IntegerExpression::floor(self.0))
    }

    fn __ceil__(&self) -> IntExprPy {
        IntExprPy(IntegerExpression::ceil(self.0))
    }

    fn __bool__(&self) -> PyResult<bool> {
        Err(DIDPPyException::new_err(
            "FloatResourceVar cannot be converted to bool",
        ))
    }
}

/// Returns an expression representing the square root.
///
/// Parameters
/// ----------
/// x: IntExpr, IntVar, IntResourceVar, FloatExpr, FloatVar, FloatResourceVar, int, or float
///     Input.
///
/// Returns
/// -------
/// FloatExpr
///     The square root.
///
/// Examples
/// --------
/// >>> import didppy as dp
/// >>> model = dp.Model()
/// >>> state = model.target_state
/// >>> expr = dp.FloatExpr(4.0)
/// >>> dp.sqrt(expr).eval(state, model)
/// 2.0
#[pyfunction]
#[pyo3(signature = (x))]
pub fn sqrt(x: FloatUnion) -> FloatExprPy {
    FloatExprPy(ContinuousExpression::from(x).sqrt())
}

/// Returns an expression representing the logarithm of :code:`x` using y :code:`as` a base.
///
/// Parameters
/// ----------
/// x: IntExpr, IntVar, IntResourceVar, FloatExpr, FloatVar, FloatResourceVar, int, or float
///     Input.
/// y: IntExpr, IntVar, IntResourceVar, FloatExpr, FloatVar, FloatResourceVar, int, or float
///     Base.
///
/// Returns
/// -------
/// FloatExpr
///     The logarithm.
///
/// Examples
/// --------
/// >>> import didppy as dp
/// >>> model = dp.Model()
/// >>> state = model.target_state
/// >>> x = dp.FloatExpr(4.0)
/// >>> y = dp.FloatExpr(2.0)
/// >>> dp.log(x, y).eval(state, model)
/// 2.0
#[pyfunction]
#[pyo3(signature = (x, y))]
pub fn log(x: FloatUnion, y: FloatUnion) -> FloatExprPy {
    let x = ContinuousExpression::from(x);
    let y = ContinuousExpression::from(y);
    FloatExprPy(x.log(y))
}

/// Convert an integer expression to a continuous expression.
///
/// Parameters
/// ----------
/// x: IntExpr, IntVar, IntResourceVar, or int
///     Input.
///
/// Returns
/// -------
/// FloatExpr
///     The continuous expression.
///
/// Examples
/// --------
/// >>> import didppy as dp
/// >>> model = dp.Model()
/// >>> state = model.target_state
/// >>> expr = dp.IntExpr(4)
/// >>> dp.float(expr).eval(state, model)
/// 4.0
#[pyfunction]
#[pyo3(signature = (x))]
pub fn float(x: IntUnion) -> FloatExprPy {
    FloatExprPy(ContinuousExpression::from(IntegerExpression::from(x)))
}

/// Returns an expression representing the greater value.
///
/// Parameters
/// ----------
/// x: ElementExpr, ElementVar, ElementResourceVar, IntExpr, IntVar, IntResourceVar, FloatExpr, FloatVar, FloatResourceVar, SetExpr, SetConst, SetVar, int, or float
///     First input.
/// y: ElementExpr, ElementVar, ElementResourceVar, IntExpr, IntVar, IntResourceVar, FloatExpr, FloatVar, FloatResourceVar, SetExpr, SetConst, SetVar, int, or float
///     Second input.
///
/// Returns
/// -------
/// ElementExpr, IntExpr, FloatExpr, or SetExpr
///     The greater value.
///
/// Raises
/// ------
/// TypeError
///     If the types of :code:`x` and :code:`y` mismatch.
///
/// Examples
/// --------
/// >>> import didppy as dp
/// >>> model = dp.Model()
/// >>> var = model.add_int_var(target=2)
/// >>> dp.max(var, 1).eval(model.target_state, model)
/// 2
#[pyfunction]
#[pyo3(signature = (x, y))]
pub fn max(x: Bound<'_, PyAny>, y: Bound<'_, PyAny>) -> PyResult<ExprUnion> {
    let result: (PyResult<IntUnion>, PyResult<IntUnion>) = (x.extract(), y.extract());
    if let (Ok(x), Ok(y)) = result {
        let x = IntegerExpression::from(x);
        let y = IntegerExpression::from(y);
        return Ok(ExprUnion::Int(IntExprPy(x.max(y))));
    }
    let result: (PyResult<FloatUnion>, PyResult<FloatUnion>) = (x.extract(), y.extract());
    if let (Ok(x), Ok(y)) = result {
        let x = ContinuousExpression::from(x);
        let y = ContinuousExpression::from(y);
        return Ok(ExprUnion::Float(FloatExprPy(x.max(y))));
    }
    let result: (PyResult<ElementUnion>, PyResult<ElementUnion>) = (x.extract(), y.extract());
    if let (Ok(x), Ok(y)) = result {
        let x = ElementExpression::from(x);
        let y = ElementExpression::from(y);
        return Ok(ExprUnion::Element(ElementExprPy(x.max(y))));
    }
    let result: (PyResult<SetUnion>, PyResult<SetUnion>) = (x.extract(), y.extract());
    if let (Ok(x), Ok(y)) = result {
        let x = SetExpression::from(x);
        let y = SetExpression::from(y);
        return Ok(ExprUnion::Set(SetExprPy(
            Condition::Set(Box::new(SetCondition::IsSubset(x.clone(), y.clone())))
                .if_then_else(y, x),
        )));
    }
    Err(PyTypeError::new_err("arguments ('x', 'y') failed to extract (IntExpr, IntExpr), (FloatExpr, FloatExpr), (ElementExpr, ElementExpr), or (SetExpr, SetExpr)"))
}

/// Returns an expression representing the smaller value.
///
/// Parameters
/// ----------
/// x: ElementExpr, ElementVar, ElementResourceVar, IntExpr, IntVar, IntResourceVar, FloatExpr, FloatVar, FloatResourceVar, SetExpr, SetConst, SetVar, int, or float
///     First input.
/// y: ElementExpr, ElementVar, ElementResourceVar, IntExpr, IntVar, IntResourceVar, FloatExpr, FloatVar, FloatResourceVar, SetExpr, SetConst, SetVar, int, or float
///     Second input.
///
/// Returns
/// -------
/// ElementExpr, IntExpr, FloatExpr, or SetExpr
///     The smaller value.
///
/// Raises
/// ------
/// TypeError
///     If the types of :code:`x` and :code:`y` mismatch.
///
/// Examples
/// --------
/// >>> import didppy as dp
/// >>> model = dp.Model()
/// >>> var = model.add_int_var(target=2)
/// >>> dp.min(var, 1).eval(model.target_state, model)
/// 1
#[pyfunction]
#[pyo3(signature = (x, y))]
pub fn min(x: Bound<'_, PyAny>, y: Bound<'_, PyAny>) -> PyResult<ExprUnion> {
    let result: (PyResult<IntUnion>, PyResult<IntUnion>) = (x.extract(), y.extract());
    if let (Ok(x), Ok(y)) = result {
        let x = IntegerExpression::from(x);
        let y = IntegerExpression::from(y);
        return Ok(ExprUnion::Int(IntExprPy(x.min(y))));
    }
    let result: (PyResult<FloatUnion>, PyResult<FloatUnion>) = (x.extract(), y.extract());
    if let (Ok(x), Ok(y)) = result {
        let x = ContinuousExpression::from(x);
        let y = ContinuousExpression::from(y);
        return Ok(ExprUnion::Float(FloatExprPy(x.min(y))));
    }
    let result: (PyResult<ElementUnion>, PyResult<ElementUnion>) = (x.extract(), y.extract());
    if let (Ok(x), Ok(y)) = result {
        let x = ElementExpression::from(x);
        let y = ElementExpression::from(y);
        return Ok(ExprUnion::Element(ElementExprPy(x.min(y))));
    }
    let result: (PyResult<SetUnion>, PyResult<SetUnion>) = (x.extract(), y.extract());
    if let (Ok(x), Ok(y)) = result {
        let x = SetExpression::from(x);
        let y = SetExpression::from(y);
        return Ok(ExprUnion::Set(SetExprPy(
            Condition::Set(Box::new(SetCondition::IsSubset(y.clone(), x.clone())))
                .if_then_else(y, x),
        )));
    }
    Err(PyTypeError::new_err("arguments ('x', 'y') failed to extract (IntExpr, IntExpr), (FloatExpr, FloatExpr), (ElementExpr, ElementExpr), or (SetExpr, SetExpr)"))
}

/// Condition.
///
/// The negation of a condition can be crated by :code:`~x`.
/// The conjunction of two conditions can be crated by :code:`x & y`.
/// The disjunction of two conditions can be crated by :code:`x | y`.
///
/// Examples
/// --------
/// >>> import didppy as dp
/// >>> model = dp.Model()
/// >>> var = model.add_int_var(target=4)
/// >>> state = model.target_state
/// >>> condition = var >= 4
/// >>> condition.eval(state, model)
/// True
/// >>> (~condition).eval(state, model)
/// False
/// >>> (condition & (var <= 5)).eval(state, model)
/// True
/// >>> (condition | (var <= 5)).eval(state, model)
/// True
#[pyclass(name = "Condition")]
#[derive(Debug, PartialEq, Clone)]
pub struct ConditionPy(Condition);

impl From<ConditionPy> for Condition {
    fn from(condition: ConditionPy) -> Self {
        condition.0
    }
}

impl From<Condition> for ConditionPy {
    fn from(condition: Condition) -> Self {
        ConditionPy(condition)
    }
}

#[pymethods]
impl ConditionPy {
    fn __invert__(&self) -> ConditionPy {
        ConditionPy(Condition::Not(Box::new(self.0.clone())))
    }

    fn __and__(&self, other: &ConditionPy) -> ConditionPy {
        ConditionPy(self.0.clone() & other.0.clone())
    }

    fn __or__(&self, other: &ConditionPy) -> ConditionPy {
        ConditionPy(self.0.clone() | other.0.clone())
    }

    fn __bool__(&self) -> PyResult<bool> {
        Err(DIDPPyException::new_err(
            "Condition cannot be converted to bool",
        ))
    }

    /// if_then_else(x, y)
    ///
    /// Returns an 'if-then-else' expression, which returns the first expression if the condition holds and the second one otherwise.
    ///
    /// Parameters
    /// ----------
    /// x: ElementExpr, ElementVar, ElementResourceVar, SetExpr, SetVar, SetConst, IntExpr, IntVar, IntResourceVar, FloatExpr, FloatVar, FloatResourceVar, int, or float
    ///     First expression.
    /// y: ElementExpr, ElementVar, ElementResourceVar, SetExpr, SetVar, SetConst, IntExpr, IntVar, IntResourceVar, FloatExpr, FloatVar, FloatResourceVar, int, or float
    ///     Second expression.
    ///
    /// Returns
    /// -------
    /// ElementExpr, SetExpr, IntExpr, or FloatExpr
    ///     The 'if-then-else' expression.
    ///     The type of the return value is determined according to the types of :code:`x` and :code:`y`.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If the types of :code:`x` and :code:`y` mismatch.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> var = model.add_int_var(target=4)
    /// >>> expr = (var >= 0).if_then_else(2, 3)
    /// >>> expr.eval(model.target_state, model)
    /// 2
    #[pyo3(signature = (x, y))]
    fn if_then_else(&self, x: Bound<'_, PyAny>, y: Bound<'_, PyAny>) -> PyResult<ExprUnion> {
        let result: (PyResult<IntUnion>, PyResult<IntUnion>) = (x.extract(), y.extract());
        if let (Ok(x), Ok(y)) = result {
            let x = IntegerExpression::from(x);
            let y = IntegerExpression::from(y);
            return Ok(ExprUnion::Int(IntExprPy(self.clone().0.if_then_else(x, y))));
        }
        let result: (PyResult<FloatUnion>, PyResult<FloatUnion>) = (x.extract(), y.extract());
        if let (Ok(x), Ok(y)) = result {
            let x = ContinuousExpression::from(x);
            let y = ContinuousExpression::from(y);
            return Ok(ExprUnion::Float(FloatExprPy(
                self.clone().0.if_then_else(x, y),
            )));
        }
        let result: (PyResult<ElementUnion>, PyResult<ElementUnion>) = (x.extract(), y.extract());
        if let (Ok(x), Ok(y)) = result {
            let x = ElementExpression::from(x);
            let y = ElementExpression::from(y);
            return Ok(ExprUnion::Element(ElementExprPy(
                self.clone().0.if_then_else(x, y),
            )));
        }
        let result: (PyResult<SetUnion>, PyResult<SetUnion>) = (x.extract(), y.extract());
        if let (Ok(x), Ok(y)) = result {
            let x = SetExpression::from(x);
            let y = SetExpression::from(y);
            return Ok(ExprUnion::Set(SetExprPy(self.clone().0.if_then_else(x, y))));
        }
        Err(PyTypeError::new_err("arguments ('x', 'y') failed to extract (IntExpr, IntExpr), (FloatExpr, FloatExpr), (ElementExpr, ElementExpr), or (SetExpr, SetExpr)"))
    }

    /// eval(state, model)
    ///
    /// Evaluates the condition.
    ///
    /// Parameters
    /// ----------
    /// state : State
    ///     State.
    /// model : Model
    ///     DyPDL Model.
    ///
    /// Returns
    /// -------
    /// bool
    ///     Value of the condition.
    ///
    /// Raises
    /// ------
    /// PanicException
    ///     If the condition is not valid.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> var = model.add_int_var(target=0)
    /// >>> condition = var >= 0
    /// >>> state = model.target_state
    /// >>> condition.eval(state, model)
    /// True
    #[pyo3(signature = (state, model))]
    fn eval(&self, state: &StatePy, model: &ModelPy) -> bool {
        let mut function_cache =
            StateFunctionCache::new(&model.inner_as_ref().state_functions);

        self.0.eval(
            state.inner_as_ref(),
            &mut function_cache,
            &model.inner_as_ref().state_functions,
            &model.inner_as_ref().table_registry,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn element_expression_from_expr() {
        let element = ElementUnion::Expr(ElementExprPy(ElementExpression::Constant(0)));
        assert_eq!(
            ElementExpression::from(element),
            ElementExpression::Constant(0)
        );
    }

    #[test]
    fn element_expression_from_var() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_variable("v", ob, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let element = ElementUnion::Var(ElementVarPy(v));
        assert_eq!(
            ElementExpression::from(element),
            ElementExpression::Variable(v.id())
        );
    }

    #[test]
    fn element_resource_expression_from_var() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_resource_variable("v", ob, false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let element = ElementUnion::ResourceVar(ElementResourceVarPy(v));
        assert_eq!(
            ElementExpression::from(element),
            ElementExpression::ResourceVariable(v.id())
        );
    }

    #[test]
    fn element_expression_from_const() {
        let element = ElementUnion::Const(0);
        assert_eq!(
            ElementExpression::from(element),
            ElementExpression::Constant(0)
        );
    }

    #[test]
    fn element_expr_to_expression() {
        let expression = ElementExprPy(ElementExpression::Constant(0));
        assert_eq!(
            ElementExpression::from(expression),
            ElementExpression::Constant(0)
        );
    }

    #[test]
    fn element_expr_new() {
        assert_eq!(
            ElementExprPy::from(ElementExpression::Constant(0)),
            ElementExprPy(ElementExpression::Constant(0))
        );
    }

    #[test]
    fn element_expr_new_py() {
        assert_eq!(
            ElementExprPy::new_py(0),
            ElementExprPy(ElementExpression::Constant(0))
        );
    }

    #[test]
    fn element_expr_lt() {
        let expression = ElementExprPy(ElementExpression::Constant(0));
        let other = ElementUnion::Const(1);
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Lt),
            ConditionPy(Condition::ComparisonE(
                ComparisonOperator::Lt,
                Box::new(ElementExpression::Constant(0)),
                Box::new(ElementExpression::Constant(1))
            ))
        );
    }

    #[test]
    fn element_expr_le() {
        let expression = ElementExprPy(ElementExpression::Constant(0));
        let other = ElementUnion::Const(1);
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Le),
            ConditionPy(Condition::ComparisonE(
                ComparisonOperator::Le,
                Box::new(ElementExpression::Constant(0)),
                Box::new(ElementExpression::Constant(1))
            ))
        );
    }

    #[test]
    fn element_expr_eq() {
        let expression = ElementExprPy(ElementExpression::Constant(0));
        let other = ElementUnion::Const(1);
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Eq),
            ConditionPy(Condition::ComparisonE(
                ComparisonOperator::Eq,
                Box::new(ElementExpression::Constant(0)),
                Box::new(ElementExpression::Constant(1))
            ))
        );
    }

    #[test]
    fn element_expr_ne() {
        let expression = ElementExprPy(ElementExpression::Constant(0));
        let other = ElementUnion::Const(1);
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Ne),
            ConditionPy(Condition::ComparisonE(
                ComparisonOperator::Ne,
                Box::new(ElementExpression::Constant(0)),
                Box::new(ElementExpression::Constant(1))
            ))
        );
    }

    #[test]
    fn element_expr_gt() {
        let expression = ElementExprPy(ElementExpression::Constant(0));
        let other = ElementUnion::Const(1);
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Gt),
            ConditionPy(Condition::ComparisonE(
                ComparisonOperator::Gt,
                Box::new(ElementExpression::Constant(0)),
                Box::new(ElementExpression::Constant(1))
            ))
        );
    }

    #[test]
    fn element_expr_ge() {
        let expression = ElementExprPy(ElementExpression::Constant(0));
        let other = ElementUnion::Const(1);
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Ge),
            ConditionPy(Condition::ComparisonE(
                ComparisonOperator::Ge,
                Box::new(ElementExpression::Constant(0)),
                Box::new(ElementExpression::Constant(1))
            ))
        );
    }

    #[test]
    fn element_expr_add() {
        let expression = ElementExprPy(ElementExpression::Constant(0));
        let other = ElementUnion::Const(1);
        assert_eq!(
            expression.__add__(other),
            ElementExprPy(ElementExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ElementExpression::Constant(0)),
                Box::new(ElementExpression::Constant(1))
            ))
        );
    }

    #[test]
    fn element_expr_sub() {
        let expression = ElementExprPy(ElementExpression::Constant(0));
        let other = ElementUnion::Const(1);
        assert_eq!(
            expression.__sub__(other),
            ElementExprPy(ElementExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ElementExpression::Constant(0)),
                Box::new(ElementExpression::Constant(1))
            ))
        );
    }

    #[test]
    fn element_expr_mul() {
        let expression = ElementExprPy(ElementExpression::Constant(0));
        let other = ElementUnion::Const(1);
        assert_eq!(
            expression.__mul__(other),
            ElementExprPy(ElementExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ElementExpression::Constant(0)),
                Box::new(ElementExpression::Constant(1))
            ))
        );
    }

    #[test]
    fn element_expr_truediv() {
        let expression = ElementExprPy(ElementExpression::Constant(0));
        let other = ElementUnion::Const(1);
        assert_eq!(
            expression.__truediv__(other),
            ElementExprPy(ElementExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ElementExpression::Constant(0)),
                Box::new(ElementExpression::Constant(1))
            ))
        );
    }

    #[test]
    fn element_expr_floordiv() {
        let expression = ElementExprPy(ElementExpression::Constant(0));
        let other = ElementUnion::Const(1);
        assert_eq!(
            expression.__floordiv__(other),
            ElementExprPy(ElementExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ElementExpression::Constant(0)),
                Box::new(ElementExpression::Constant(1))
            ))
        );
    }

    #[test]
    fn element_expr_mod() {
        let expression = ElementExprPy(ElementExpression::Constant(0));
        let other = ElementUnion::Const(1);
        assert_eq!(
            expression.__mod__(other),
            ElementExprPy(ElementExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ElementExpression::Constant(0)),
                Box::new(ElementExpression::Constant(1))
            ))
        );
    }

    #[test]
    fn element_expr_radd() {
        let expression = ElementExprPy(ElementExpression::Constant(0));
        let other = ElementUnion::Const(1);
        assert_eq!(
            expression.__radd__(other),
            ElementExprPy(ElementExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ElementExpression::Constant(0)),
                Box::new(ElementExpression::Constant(1))
            ))
        );
    }

    #[test]
    fn element_expr_rsub() {
        let expression = ElementExprPy(ElementExpression::Constant(0));
        let other = ElementUnion::Const(1);
        assert_eq!(
            expression.__rsub__(other),
            ElementExprPy(ElementExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ElementExpression::Constant(1)),
                Box::new(ElementExpression::Constant(0)),
            ))
        );
    }

    #[test]
    fn element_expr_rmul() {
        let expression = ElementExprPy(ElementExpression::Constant(0));
        let other = ElementUnion::Const(1);
        assert_eq!(
            expression.__rmul__(other),
            ElementExprPy(ElementExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ElementExpression::Constant(0)),
                Box::new(ElementExpression::Constant(1))
            ))
        );
    }

    #[test]
    fn element_expr_rtruediv() {
        let expression = ElementExprPy(ElementExpression::Constant(0));
        let other = ElementUnion::Const(1);
        assert_eq!(
            expression.__rtruediv__(other),
            ElementExprPy(ElementExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ElementExpression::Constant(1)),
                Box::new(ElementExpression::Constant(0)),
            ))
        );
    }

    #[test]
    fn element_expr_rfloordiv() {
        let expression = ElementExprPy(ElementExpression::Constant(0));
        let other = ElementUnion::Const(1);
        assert_eq!(
            expression.__rfloordiv__(other),
            ElementExprPy(ElementExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ElementExpression::Constant(1)),
                Box::new(ElementExpression::Constant(0)),
            ))
        );
    }

    #[test]
    fn element_expr_rmod() {
        let expression = ElementExprPy(ElementExpression::Constant(0));
        let other = ElementUnion::Const(1);
        assert_eq!(
            expression.__rmod__(other),
            ElementExprPy(ElementExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ElementExpression::Constant(1)),
                Box::new(ElementExpression::Constant(0)),
            ))
        );
    }

    #[test]
    fn element_var_to_variable() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_variable("v", ob, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let v_py = ElementVarPy(v);
        assert_eq!(ElementVariable::from(v_py), v);
    }

    #[test]
    fn element_var_to_expression() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_variable("v", ob, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let v_py = ElementVarPy(v);
        assert_eq!(
            ElementExpression::from(v_py),
            ElementExpression::Variable(v.id())
        );
    }

    #[test]
    fn element_var_new() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_variable("v", ob, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        assert_eq!(ElementVarPy::from(v), ElementVarPy(v));
    }

    #[test]
    fn element_var_lt() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_variable("v", ob, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = ElementVarPy(v);

        let other = ElementUnion::Const(1);
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Lt),
            ConditionPy(Condition::ComparisonE(
                ComparisonOperator::Lt,
                Box::new(ElementExpression::Variable(v.id())),
                Box::new(ElementExpression::Constant(1))
            ))
        );
    }

    #[test]
    fn element_var_le() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_variable("v", ob, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = ElementVarPy(v);

        let other = ElementUnion::Const(1);
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Le),
            ConditionPy(Condition::ComparisonE(
                ComparisonOperator::Le,
                Box::new(ElementExpression::Variable(v.id())),
                Box::new(ElementExpression::Constant(1))
            ))
        );
    }

    #[test]
    fn element_var_eq() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_variable("v", ob, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = ElementVarPy(v);

        let other = ElementUnion::Const(1);
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Eq),
            ConditionPy(Condition::ComparisonE(
                ComparisonOperator::Eq,
                Box::new(ElementExpression::Variable(v.id())),
                Box::new(ElementExpression::Constant(1))
            ))
        );
    }

    #[test]
    fn element_var_ne() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_variable("v", ob, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = ElementVarPy(v);

        let other = ElementUnion::Const(1);
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Ne),
            ConditionPy(Condition::ComparisonE(
                ComparisonOperator::Ne,
                Box::new(ElementExpression::Variable(v.id())),
                Box::new(ElementExpression::Constant(1))
            ))
        );
    }

    #[test]
    fn element_var_gt() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_variable("v", ob, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = ElementVarPy(v);

        let other = ElementUnion::Const(1);
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Gt),
            ConditionPy(Condition::ComparisonE(
                ComparisonOperator::Gt,
                Box::new(ElementExpression::Variable(v.id())),
                Box::new(ElementExpression::Constant(1))
            ))
        );
    }

    #[test]
    fn element_var_ge() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_variable("v", ob, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = ElementVarPy(v);

        let other = ElementUnion::Const(1);
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Ge),
            ConditionPy(Condition::ComparisonE(
                ComparisonOperator::Ge,
                Box::new(ElementExpression::Variable(v.id())),
                Box::new(ElementExpression::Constant(1))
            ))
        );
    }

    #[test]
    fn element_var_add() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_variable("v", ob, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = ElementVarPy(v);

        let other = ElementUnion::Const(1);
        assert_eq!(
            expression.__add__(other),
            ElementExprPy(ElementExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ElementExpression::Variable(v.id())),
                Box::new(ElementExpression::Constant(1))
            ))
        );
    }

    #[test]
    fn element_var_sub() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_variable("v", ob, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = ElementVarPy(v);

        let other = ElementUnion::Const(1);
        assert_eq!(
            expression.__sub__(other),
            ElementExprPy(ElementExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ElementExpression::Variable(v.id())),
                Box::new(ElementExpression::Constant(1))
            ))
        );
    }

    #[test]
    fn element_var_mul() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_variable("v", ob, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = ElementVarPy(v);

        let other = ElementUnion::Const(1);
        assert_eq!(
            expression.__mul__(other),
            ElementExprPy(ElementExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ElementExpression::Variable(v.id())),
                Box::new(ElementExpression::Constant(1))
            ))
        );
    }

    #[test]
    fn element_var_truediv() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_variable("v", ob, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = ElementVarPy(v);

        let other = ElementUnion::Const(1);
        assert_eq!(
            expression.__truediv__(other),
            ElementExprPy(ElementExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ElementExpression::Variable(v.id())),
                Box::new(ElementExpression::Constant(1))
            ))
        );
    }

    #[test]
    fn element_var_floordiv() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_variable("v", ob, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = ElementVarPy(v);

        let other = ElementUnion::Const(1);
        assert_eq!(
            expression.__floordiv__(other),
            ElementExprPy(ElementExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ElementExpression::Variable(v.id())),
                Box::new(ElementExpression::Constant(1))
            ))
        );
    }

    #[test]
    fn element_var_mod() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_variable("v", ob, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = ElementVarPy(v);

        let other = ElementUnion::Const(1);
        assert_eq!(
            expression.__mod__(other),
            ElementExprPy(ElementExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ElementExpression::Variable(v.id())),
                Box::new(ElementExpression::Constant(1))
            ))
        );
    }

    #[test]
    fn element_var_radd() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_variable("v", ob, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = ElementVarPy(v);

        let other = ElementUnion::Const(1);
        assert_eq!(
            expression.__radd__(other),
            ElementExprPy(ElementExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ElementExpression::Variable(v.id())),
                Box::new(ElementExpression::Constant(1))
            ))
        );
    }

    #[test]
    fn element_var_rsub() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_variable("v", ob, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = ElementVarPy(v);

        let other = ElementUnion::Const(1);
        assert_eq!(
            expression.__rsub__(other),
            ElementExprPy(ElementExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ElementExpression::Constant(1)),
                Box::new(ElementExpression::Variable(v.id())),
            ))
        );
    }

    #[test]
    fn element_var_rmul() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_variable("v", ob, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = ElementVarPy(v);

        let other = ElementUnion::Const(1);
        assert_eq!(
            expression.__rmul__(other),
            ElementExprPy(ElementExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ElementExpression::Variable(v.id())),
                Box::new(ElementExpression::Constant(1))
            ))
        );
    }

    #[test]
    fn element_var_rtruediv() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_variable("v", ob, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = ElementVarPy(v);

        let other = ElementUnion::Const(1);
        assert_eq!(
            expression.__rtruediv__(other),
            ElementExprPy(ElementExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ElementExpression::Constant(1)),
                Box::new(ElementExpression::Variable(v.id())),
            ))
        );
    }

    #[test]
    fn element_var_rfloordiv() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_variable("v", ob, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = ElementVarPy(v);

        let other = ElementUnion::Const(1);
        assert_eq!(
            expression.__rfloordiv__(other),
            ElementExprPy(ElementExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ElementExpression::Constant(1)),
                Box::new(ElementExpression::Variable(v.id())),
            ))
        );
    }

    #[test]
    fn element_var_rmod() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_variable("v", ob, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = ElementVarPy(v);

        let other = ElementUnion::Const(1);
        assert_eq!(
            expression.__rmod__(other),
            ElementExprPy(ElementExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ElementExpression::Constant(1)),
                Box::new(ElementExpression::Variable(v.id())),
            ))
        );
    }

    #[test]
    fn element_resource_var_to_variable() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_resource_variable("v", ob, false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let v_py = ElementResourceVarPy(v);
        assert_eq!(ElementResourceVariable::from(v_py), v);
    }

    #[test]
    fn element_resource_var_to_expression() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_resource_variable("v", ob, false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let v_py = ElementResourceVarPy(v);
        assert_eq!(
            ElementExpression::from(v_py),
            ElementExpression::ResourceVariable(v.id())
        );
    }

    #[test]
    fn element_resource_var_new() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_resource_variable("v", ob, false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        assert_eq!(ElementResourceVarPy::from(v), ElementResourceVarPy(v));
    }

    #[test]
    fn element_resource_var_lt() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_resource_variable("v", ob, false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = ElementResourceVarPy(v);

        let other = ElementUnion::Const(1);
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Lt),
            ConditionPy(Condition::ComparisonE(
                ComparisonOperator::Lt,
                Box::new(ElementExpression::ResourceVariable(v.id())),
                Box::new(ElementExpression::Constant(1))
            ))
        );
    }

    #[test]
    fn element_resource_var_le() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_resource_variable("v", ob, false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = ElementResourceVarPy(v);

        let other = ElementUnion::Const(1);
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Le),
            ConditionPy(Condition::ComparisonE(
                ComparisonOperator::Le,
                Box::new(ElementExpression::ResourceVariable(v.id())),
                Box::new(ElementExpression::Constant(1))
            ))
        );
    }

    #[test]
    fn element_resource_var_eq() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_resource_variable("v", ob, false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = ElementResourceVarPy(v);

        let other = ElementUnion::Const(1);
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Eq),
            ConditionPy(Condition::ComparisonE(
                ComparisonOperator::Eq,
                Box::new(ElementExpression::ResourceVariable(v.id())),
                Box::new(ElementExpression::Constant(1))
            ))
        );
    }

    #[test]
    fn element_resource_var_ne() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_resource_variable("v", ob, false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = ElementResourceVarPy(v);

        let other = ElementUnion::Const(1);
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Ne),
            ConditionPy(Condition::ComparisonE(
                ComparisonOperator::Ne,
                Box::new(ElementExpression::ResourceVariable(v.id())),
                Box::new(ElementExpression::Constant(1))
            ))
        );
    }

    #[test]
    fn element_resource_var_gt() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_resource_variable("v", ob, false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = ElementResourceVarPy(v);

        let other = ElementUnion::Const(1);
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Gt),
            ConditionPy(Condition::ComparisonE(
                ComparisonOperator::Gt,
                Box::new(ElementExpression::ResourceVariable(v.id())),
                Box::new(ElementExpression::Constant(1))
            ))
        );
    }

    #[test]
    fn element_resource_var_ge() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_resource_variable("v", ob, false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = ElementResourceVarPy(v);

        let other = ElementUnion::Const(1);
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Ge),
            ConditionPy(Condition::ComparisonE(
                ComparisonOperator::Ge,
                Box::new(ElementExpression::ResourceVariable(v.id())),
                Box::new(ElementExpression::Constant(1))
            ))
        );
    }

    #[test]
    fn element_resource_var_add() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_resource_variable("v", ob, false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = ElementResourceVarPy(v);

        let other = ElementUnion::Const(1);
        assert_eq!(
            expression.__add__(other),
            ElementExprPy(ElementExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ElementExpression::ResourceVariable(v.id())),
                Box::new(ElementExpression::Constant(1))
            ))
        );
    }

    #[test]
    fn element_resource_var_sub() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_resource_variable("v", ob, false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = ElementResourceVarPy(v);

        let other = ElementUnion::Const(1);
        assert_eq!(
            expression.__sub__(other),
            ElementExprPy(ElementExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ElementExpression::ResourceVariable(v.id())),
                Box::new(ElementExpression::Constant(1))
            ))
        );
    }

    #[test]
    fn element_resource_var_mul() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_resource_variable("v", ob, false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = ElementResourceVarPy(v);

        let other = ElementUnion::Const(1);
        assert_eq!(
            expression.__mul__(other),
            ElementExprPy(ElementExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ElementExpression::ResourceVariable(v.id())),
                Box::new(ElementExpression::Constant(1))
            ))
        );
    }

    #[test]
    fn element_resource_var_truediv() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_resource_variable("v", ob, false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = ElementResourceVarPy(v);

        let other = ElementUnion::Const(1);
        assert_eq!(
            expression.__truediv__(other),
            ElementExprPy(ElementExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ElementExpression::ResourceVariable(v.id())),
                Box::new(ElementExpression::Constant(1))
            ))
        );
    }

    #[test]
    fn element_resource_var_floordiv() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_resource_variable("v", ob, false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = ElementResourceVarPy(v);

        let other = ElementUnion::Const(1);
        assert_eq!(
            expression.__floordiv__(other),
            ElementExprPy(ElementExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ElementExpression::ResourceVariable(v.id())),
                Box::new(ElementExpression::Constant(1))
            ))
        );
    }

    #[test]
    fn element_resource_var_mod() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_resource_variable("v", ob, false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = ElementResourceVarPy(v);

        let other = ElementUnion::Const(1);
        assert_eq!(
            expression.__mod__(other),
            ElementExprPy(ElementExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ElementExpression::ResourceVariable(v.id())),
                Box::new(ElementExpression::Constant(1))
            ))
        );
    }

    #[test]
    fn element_resource_var_radd() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_resource_variable("v", ob, false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = ElementResourceVarPy(v);

        let other = ElementUnion::Const(1);
        assert_eq!(
            expression.__radd__(other),
            ElementExprPy(ElementExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ElementExpression::ResourceVariable(v.id())),
                Box::new(ElementExpression::Constant(1))
            ))
        );
    }

    #[test]
    fn element_resource_var_rsub() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_resource_variable("v", ob, false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = ElementResourceVarPy(v);

        let other = ElementUnion::Const(1);
        assert_eq!(
            expression.__rsub__(other),
            ElementExprPy(ElementExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ElementExpression::Constant(1)),
                Box::new(ElementExpression::ResourceVariable(v.id())),
            ))
        );
    }

    #[test]
    fn element_resource_var_rmul() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_resource_variable("v", ob, false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = ElementResourceVarPy(v);

        let other = ElementUnion::Const(1);
        assert_eq!(
            expression.__rmul__(other),
            ElementExprPy(ElementExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ElementExpression::ResourceVariable(v.id())),
                Box::new(ElementExpression::Constant(1))
            ))
        );
    }

    #[test]
    fn element_resource_var_rtruediv() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_resource_variable("v", ob, false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = ElementResourceVarPy(v);

        let other = ElementUnion::Const(1);
        assert_eq!(
            expression.__rtruediv__(other),
            ElementExprPy(ElementExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ElementExpression::Constant(1)),
                Box::new(ElementExpression::ResourceVariable(v.id())),
            ))
        );
    }

    #[test]
    fn element_resource_var_rfloordiv() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_resource_variable("v", ob, false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = ElementResourceVarPy(v);

        let other = ElementUnion::Const(1);
        assert_eq!(
            expression.__rfloordiv__(other),
            ElementExprPy(ElementExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ElementExpression::Constant(1)),
                Box::new(ElementExpression::ResourceVariable(v.id())),
            ))
        );
    }

    #[test]
    fn element_resource_var_rmod() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_resource_variable("v", ob, false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = ElementResourceVarPy(v);

        let other = ElementUnion::Const(1);
        assert_eq!(
            expression.__rmod__(other),
            ElementExprPy(ElementExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ElementExpression::Constant(1)),
                Box::new(ElementExpression::ResourceVariable(v.id())),
            ))
        );
    }

    #[test]
    fn set_expression_from_expr() {
        let set = SetUnion::Expr(SetExprPy(SetExpression::Reference(
            ReferenceExpression::Variable(0),
        )));
        assert_eq!(
            SetExpression::from(set),
            SetExpression::Reference(ReferenceExpression::Variable(0))
        );
    }

    #[test]
    fn set_expression_from_var() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_set_variable("v", ob, Set::with_capacity(10));
        assert!(v.is_ok());
        let v = v.unwrap();
        let set = SetUnion::Var(SetVarPy(v));
        assert_eq!(
            SetExpression::from(set),
            SetExpression::Reference(ReferenceExpression::Variable(v.id()))
        );
    }

    #[test]
    fn set_expression_from_const() {
        let set = SetUnion::Const(SetConstPy(Set::with_capacity(10)));
        assert_eq!(
            SetExpression::from(set),
            SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(10)))
        );
    }

    #[test]
    fn set_expr_to_expression() {
        let expression = SetExprPy(SetExpression::Reference(ReferenceExpression::Variable(0)));
        assert_eq!(
            SetExpression::from(expression),
            SetExpression::Reference(ReferenceExpression::Variable(0))
        );
    }

    #[test]
    fn set_expr_new() {
        assert_eq!(
            SetExprPy::from(SetExpression::Reference(ReferenceExpression::Variable(0))),
            SetExprPy(SetExpression::Reference(ReferenceExpression::Variable(0)))
        );
    }

    #[test]
    fn set_expr_lt() {
        let expression = SetExprPy(SetExpression::Reference(ReferenceExpression::Variable(0)));
        let other = SetUnion::Expr(SetExprPy(SetExpression::Reference(
            ReferenceExpression::Variable(1),
        )));
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Lt),
            ConditionPy(Condition::And(
                Box::new(Condition::Set(Box::new(SetCondition::IsSubset(
                    SetExpression::Reference(ReferenceExpression::Variable(0)),
                    SetExpression::Reference(ReferenceExpression::Variable(1)),
                )))),
                Box::new(Condition::Not(Box::new(Condition::Set(Box::new(
                    SetCondition::IsSubset(
                        SetExpression::Reference(ReferenceExpression::Variable(1)),
                        SetExpression::Reference(ReferenceExpression::Variable(0)),
                    )
                )))),)
            ))
        );
    }

    #[test]
    fn set_expr_le() {
        let expression = SetExprPy(SetExpression::Reference(ReferenceExpression::Variable(0)));
        let other = SetUnion::Expr(SetExprPy(SetExpression::Reference(
            ReferenceExpression::Variable(1),
        )));
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Le),
            ConditionPy(Condition::Set(Box::new(SetCondition::IsSubset(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
                SetExpression::Reference(ReferenceExpression::Variable(1)),
            ))))
        );
    }

    #[test]
    fn set_expr_eq() {
        let expression = SetExprPy(SetExpression::Reference(ReferenceExpression::Variable(0)));
        let other = SetUnion::Expr(SetExprPy(SetExpression::Reference(
            ReferenceExpression::Variable(1),
        )));
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Eq),
            ConditionPy(Condition::Set(Box::new(SetCondition::IsEqual(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
                SetExpression::Reference(ReferenceExpression::Variable(1)),
            ))))
        );
    }

    #[test]
    fn set_expr_ne() {
        let expression = SetExprPy(SetExpression::Reference(ReferenceExpression::Variable(0)));
        let other = SetUnion::Expr(SetExprPy(SetExpression::Reference(
            ReferenceExpression::Variable(1),
        )));
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Ne),
            ConditionPy(Condition::Set(Box::new(SetCondition::IsNotEqual(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
                SetExpression::Reference(ReferenceExpression::Variable(1)),
            ))))
        );
    }

    #[test]
    fn set_expr_gt() {
        let expression = SetExprPy(SetExpression::Reference(ReferenceExpression::Variable(0)));
        let other = SetUnion::Expr(SetExprPy(SetExpression::Reference(
            ReferenceExpression::Variable(1),
        )));
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Gt),
            ConditionPy(Condition::And(
                Box::new(Condition::Set(Box::new(SetCondition::IsSubset(
                    SetExpression::Reference(ReferenceExpression::Variable(1)),
                    SetExpression::Reference(ReferenceExpression::Variable(0)),
                )))),
                Box::new(Condition::Not(Box::new(Condition::Set(Box::new(
                    SetCondition::IsSubset(
                        SetExpression::Reference(ReferenceExpression::Variable(0)),
                        SetExpression::Reference(ReferenceExpression::Variable(1)),
                    )
                )))),)
            ))
        );
    }

    #[test]
    fn set_expr_ge() {
        let expression = SetExprPy(SetExpression::Reference(ReferenceExpression::Variable(0)));
        let other = SetUnion::Expr(SetExprPy(SetExpression::Reference(
            ReferenceExpression::Variable(1),
        )));
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Ge),
            ConditionPy(Condition::Set(Box::new(SetCondition::IsSubset(
                SetExpression::Reference(ReferenceExpression::Variable(1)),
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            ))))
        );
    }

    #[test]
    fn set_expr_sub() {
        let expression = SetExprPy(SetExpression::Reference(ReferenceExpression::Variable(0)));
        let other = SetUnion::Expr(SetExprPy(SetExpression::Reference(
            ReferenceExpression::Variable(1),
        )));
        assert_eq!(
            expression.__sub__(other),
            SetExprPy(SetExpression::SetOperation(
                SetOperator::Difference,
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
            ))
        );
    }

    #[test]
    fn set_expr_and() {
        let expression = SetExprPy(SetExpression::Reference(ReferenceExpression::Variable(0)));
        let other = SetUnion::Expr(SetExprPy(SetExpression::Reference(
            ReferenceExpression::Variable(1),
        )));
        assert_eq!(
            expression.__and__(other),
            SetExprPy(SetExpression::SetOperation(
                SetOperator::Intersection,
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
            ))
        );
    }

    #[test]
    fn set_expr_xor() {
        let expression = SetExprPy(SetExpression::Reference(ReferenceExpression::Variable(0)));
        let other = SetUnion::Expr(SetExprPy(SetExpression::Reference(
            ReferenceExpression::Variable(1),
        )));
        assert_eq!(
            expression.__xor__(other),
            SetExprPy(SetExpression::SetOperation(
                SetOperator::Union,
                Box::new(SetExpression::SetOperation(
                    SetOperator::Difference,
                    Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
                    Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
                )),
                Box::new(SetExpression::SetOperation(
                    SetOperator::Difference,
                    Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
                    Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
                ))
            ))
        );
    }

    #[test]
    fn set_expr_or() {
        let expression = SetExprPy(SetExpression::Reference(ReferenceExpression::Variable(0)));
        let other = SetUnion::Expr(SetExprPy(SetExpression::Reference(
            ReferenceExpression::Variable(1),
        )));
        assert_eq!(
            expression.__or__(other),
            SetExprPy(SetExpression::SetOperation(
                SetOperator::Union,
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
            ))
        );
    }

    #[test]
    fn set_expr_rsub() {
        let expression = SetExprPy(SetExpression::Reference(ReferenceExpression::Variable(0)));
        let other = SetUnion::Expr(SetExprPy(SetExpression::Reference(
            ReferenceExpression::Variable(1),
        )));
        assert_eq!(
            expression.__rsub__(other),
            SetExprPy(SetExpression::SetOperation(
                SetOperator::Difference,
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
            ))
        );
    }

    #[test]
    fn set_expr_rand() {
        let expression = SetExprPy(SetExpression::Reference(ReferenceExpression::Variable(0)));
        let other = SetUnion::Expr(SetExprPy(SetExpression::Reference(
            ReferenceExpression::Variable(1),
        )));
        assert_eq!(
            expression.__rand__(other),
            SetExprPy(SetExpression::SetOperation(
                SetOperator::Intersection,
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
            ))
        );
    }

    #[test]
    fn set_expr_rxor() {
        let expression = SetExprPy(SetExpression::Reference(ReferenceExpression::Variable(0)));
        let other = SetUnion::Expr(SetExprPy(SetExpression::Reference(
            ReferenceExpression::Variable(1),
        )));
        assert_eq!(
            expression.__rxor__(other),
            SetExprPy(SetExpression::SetOperation(
                SetOperator::Union,
                Box::new(SetExpression::SetOperation(
                    SetOperator::Difference,
                    Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
                    Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
                )),
                Box::new(SetExpression::SetOperation(
                    SetOperator::Difference,
                    Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
                    Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
                ))
            ))
        );
    }

    #[test]
    fn set_expr_ror() {
        let expression = SetExprPy(SetExpression::Reference(ReferenceExpression::Variable(0)));
        let other = SetUnion::Expr(SetExprPy(SetExpression::Reference(
            ReferenceExpression::Variable(1),
        )));
        assert_eq!(
            expression.__ror__(other),
            SetExprPy(SetExpression::SetOperation(
                SetOperator::Union,
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
            ))
        );
    }

    #[test]
    fn set_expr_add_element() {
        let expression = SetExprPy(SetExpression::Reference(ReferenceExpression::Variable(0)));
        let element = ElementUnion::Const(0);
        assert_eq!(
            expression.add(element),
            SetExprPy(SetExpression::SetElementOperation(
                SetElementOperator::Add,
                ElementExpression::Constant(0),
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
            ))
        );
    }

    #[test]
    fn set_expr_remove_element() {
        let expression = SetExprPy(SetExpression::Reference(ReferenceExpression::Variable(0)));
        let element = ElementUnion::Const(0);
        assert_eq!(
            expression.remove(element),
            SetExprPy(SetExpression::SetElementOperation(
                SetElementOperator::Remove,
                ElementExpression::Constant(0),
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
            ))
        );
    }

    #[test]
    fn set_expr_difference_ok() {
        let expression = SetExprPy(SetExpression::Reference(ReferenceExpression::Variable(0)));
        let other = SetUnion::Expr(SetExprPy(SetExpression::Reference(
            ReferenceExpression::Variable(1),
        )));
        assert_eq!(
            expression.difference(other),
            SetExprPy(SetExpression::SetOperation(
                SetOperator::Difference,
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
            ))
        );
    }

    #[test]
    fn set_expr_intersection_ok() {
        let expression = SetExprPy(SetExpression::Reference(ReferenceExpression::Variable(0)));
        let other = SetUnion::Expr(SetExprPy(SetExpression::Reference(
            ReferenceExpression::Variable(1),
        )));
        assert_eq!(
            expression.intersection(other),
            SetExprPy(SetExpression::SetOperation(
                SetOperator::Intersection,
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
            ))
        );
    }

    #[test]
    fn set_expr_symmetric_difference_ok() {
        let expression = SetExprPy(SetExpression::Reference(ReferenceExpression::Variable(0)));
        let other = SetUnion::Expr(SetExprPy(SetExpression::Reference(
            ReferenceExpression::Variable(1),
        )));
        assert_eq!(
            expression.symmetric_difference(other),
            SetExprPy(SetExpression::SetOperation(
                SetOperator::Union,
                Box::new(SetExpression::SetOperation(
                    SetOperator::Difference,
                    Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
                    Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
                )),
                Box::new(SetExpression::SetOperation(
                    SetOperator::Difference,
                    Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
                    Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
                ))
            ))
        );
    }

    #[test]
    fn set_expr_union() {
        let expression = SetExprPy(SetExpression::Reference(ReferenceExpression::Variable(0)));
        let other = SetUnion::Expr(SetExprPy(SetExpression::Reference(
            ReferenceExpression::Variable(1),
        )));
        assert_eq!(
            expression.union(other),
            SetExprPy(SetExpression::SetOperation(
                SetOperator::Union,
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
            ))
        );
    }

    #[test]
    fn set_expr_contains() {
        let expression = SetExprPy(SetExpression::Reference(ReferenceExpression::Variable(0)));
        let element = ElementUnion::Const(0);
        assert_eq!(
            expression.contains(element),
            ConditionPy(Condition::Set(Box::new(SetCondition::IsIn(
                ElementExpression::Constant(0),
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            ))))
        );
    }

    #[test]
    fn set_expr_len() {
        let expression = SetExprPy(SetExpression::Reference(ReferenceExpression::Variable(0)));
        assert_eq!(
            expression.len(),
            IntExprPy(IntegerExpression::Cardinality(SetExpression::Reference(
                ReferenceExpression::Variable(0)
            )))
        );
    }

    #[test]
    fn set_expr_is_empty() {
        let expression = SetExprPy(SetExpression::Reference(ReferenceExpression::Variable(0)));
        assert_eq!(
            expression.is_empty(),
            ConditionPy(Condition::Set(Box::new(SetCondition::IsEmpty(
                SetExpression::Reference(ReferenceExpression::Variable(0))
            ))))
        );
    }

    #[test]
    fn set_expr_complement() {
        let expression = SetExprPy(SetExpression::Reference(ReferenceExpression::Variable(0)));
        assert_eq!(
            expression.complement(),
            SetExprPy(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0))
            )))
        );
    }

    #[test]
    fn set_var_to_variable() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_set_variable("v", ob, Set::with_capacity(10));
        assert!(v.is_ok());
        let v = v.unwrap();
        let v_py = SetVarPy(v);
        assert_eq!(SetVariable::from(v_py), v);
    }

    #[test]
    fn set_var_to_expression() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_set_variable("v", ob, Set::with_capacity(10));
        assert!(v.is_ok());
        let v = v.unwrap();
        let v_py = SetVarPy(v);
        assert_eq!(
            SetExpression::from(v_py),
            SetExpression::Reference(ReferenceExpression::Variable(v.id()))
        );
    }

    #[test]
    fn set_var_new() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_set_variable("v", ob, Set::with_capacity(10));
        assert!(v.is_ok());
        let v = v.unwrap();
        assert_eq!(SetVarPy::from(v), SetVarPy(v));
    }

    #[test]
    fn set_var_lt() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_set_variable("v", ob, Set::with_capacity(10));
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = SetVarPy(v);

        let other = SetUnion::Expr(SetExprPy(SetExpression::Reference(
            ReferenceExpression::Variable(1),
        )));
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Lt),
            ConditionPy(Condition::And(
                Box::new(Condition::Set(Box::new(SetCondition::IsSubset(
                    SetExpression::Reference(ReferenceExpression::Variable(v.id())),
                    SetExpression::Reference(ReferenceExpression::Variable(1)),
                )))),
                Box::new(Condition::Not(Box::new(Condition::Set(Box::new(
                    SetCondition::IsSubset(
                        SetExpression::Reference(ReferenceExpression::Variable(1)),
                        SetExpression::Reference(ReferenceExpression::Variable(v.id())),
                    )
                )))),)
            ))
        );
    }

    #[test]
    fn set_var_le() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_set_variable("v", ob, Set::with_capacity(10));
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = SetVarPy(v);

        let other = SetUnion::Expr(SetExprPy(SetExpression::Reference(
            ReferenceExpression::Variable(1),
        )));
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Le),
            ConditionPy(Condition::Set(Box::new(SetCondition::IsSubset(
                SetExpression::Reference(ReferenceExpression::Variable(v.id())),
                SetExpression::Reference(ReferenceExpression::Variable(1)),
            ))))
        );
    }

    #[test]
    fn set_var_eq() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_set_variable("v", ob, Set::with_capacity(10));
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = SetVarPy(v);

        let other = SetUnion::Expr(SetExprPy(SetExpression::Reference(
            ReferenceExpression::Variable(1),
        )));
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Eq),
            ConditionPy(Condition::Set(Box::new(SetCondition::IsEqual(
                SetExpression::Reference(ReferenceExpression::Variable(v.id())),
                SetExpression::Reference(ReferenceExpression::Variable(1)),
            ))))
        );
    }

    #[test]
    fn set_var_ne() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_set_variable("v", ob, Set::with_capacity(10));
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = SetVarPy(v);

        let other = SetUnion::Expr(SetExprPy(SetExpression::Reference(
            ReferenceExpression::Variable(1),
        )));
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Ne),
            ConditionPy(Condition::Set(Box::new(SetCondition::IsNotEqual(
                SetExpression::Reference(ReferenceExpression::Variable(v.id())),
                SetExpression::Reference(ReferenceExpression::Variable(1)),
            ))))
        );
    }

    #[test]
    fn set_var_gt() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_set_variable("v", ob, Set::with_capacity(10));
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = SetVarPy(v);

        let other = SetUnion::Expr(SetExprPy(SetExpression::Reference(
            ReferenceExpression::Variable(1),
        )));
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Gt),
            ConditionPy(Condition::And(
                Box::new(Condition::Set(Box::new(SetCondition::IsSubset(
                    SetExpression::Reference(ReferenceExpression::Variable(1)),
                    SetExpression::Reference(ReferenceExpression::Variable(v.id())),
                )))),
                Box::new(Condition::Not(Box::new(Condition::Set(Box::new(
                    SetCondition::IsSubset(
                        SetExpression::Reference(ReferenceExpression::Variable(v.id())),
                        SetExpression::Reference(ReferenceExpression::Variable(1)),
                    )
                )))),)
            ))
        );
    }

    #[test]
    fn set_var_ge() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_set_variable("v", ob, Set::with_capacity(10));
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = SetVarPy(v);

        let other = SetUnion::Expr(SetExprPy(SetExpression::Reference(
            ReferenceExpression::Variable(1),
        )));
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Ge),
            ConditionPy(Condition::Set(Box::new(SetCondition::IsSubset(
                SetExpression::Reference(ReferenceExpression::Variable(1)),
                SetExpression::Reference(ReferenceExpression::Variable(v.id())),
            ))))
        );
    }

    #[test]
    fn set_var_sub() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_set_variable("v", ob, Set::with_capacity(10));
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = SetVarPy(v);

        let other = SetUnion::Expr(SetExprPy(SetExpression::Reference(
            ReferenceExpression::Variable(1),
        )));
        assert_eq!(
            expression.__sub__(other),
            SetExprPy(SetExpression::SetOperation(
                SetOperator::Difference,
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                    v.id()
                ))),
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
            ))
        );
    }

    #[test]
    fn set_var_and() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_set_variable("v", ob, Set::with_capacity(10));
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = SetVarPy(v);

        let other = SetUnion::Expr(SetExprPy(SetExpression::Reference(
            ReferenceExpression::Variable(1),
        )));
        assert_eq!(
            expression.__and__(other),
            SetExprPy(SetExpression::SetOperation(
                SetOperator::Intersection,
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                    v.id()
                ))),
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
            ))
        );
    }

    #[test]
    fn set_var_xor() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_set_variable("v", ob, Set::with_capacity(10));
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = SetVarPy(v);

        let other = SetUnion::Expr(SetExprPy(SetExpression::Reference(
            ReferenceExpression::Variable(1),
        )));
        assert_eq!(
            expression.__xor__(other),
            SetExprPy(SetExpression::SetOperation(
                SetOperator::Union,
                Box::new(SetExpression::SetOperation(
                    SetOperator::Difference,
                    Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                        v.id()
                    ))),
                    Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
                )),
                Box::new(SetExpression::SetOperation(
                    SetOperator::Difference,
                    Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
                    Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                        v.id()
                    ))),
                ))
            ))
        );
    }

    #[test]
    fn set_var_or() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_set_variable("v", ob, Set::with_capacity(10));
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = SetVarPy(v);

        let other = SetUnion::Expr(SetExprPy(SetExpression::Reference(
            ReferenceExpression::Variable(1),
        )));
        assert_eq!(
            expression.__or__(other),
            SetExprPy(SetExpression::SetOperation(
                SetOperator::Union,
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                    v.id()
                ))),
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
            ))
        );
    }

    #[test]
    fn set_var_rsub() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_set_variable("v", ob, Set::with_capacity(10));
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = SetVarPy(v);

        let other = SetUnion::Expr(SetExprPy(SetExpression::Reference(
            ReferenceExpression::Variable(1),
        )));
        assert_eq!(
            expression.__rsub__(other),
            SetExprPy(SetExpression::SetOperation(
                SetOperator::Difference,
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                    v.id()
                ))),
            ))
        );
    }

    #[test]
    fn set_var_rand() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_set_variable("v", ob, Set::with_capacity(10));
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = SetVarPy(v);

        let other = SetUnion::Expr(SetExprPy(SetExpression::Reference(
            ReferenceExpression::Variable(1),
        )));
        assert_eq!(
            expression.__rand__(other),
            SetExprPy(SetExpression::SetOperation(
                SetOperator::Intersection,
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                    v.id()
                ))),
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
            ))
        );
    }

    #[test]
    fn set_var_rxor() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_set_variable("v", ob, Set::with_capacity(10));
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = SetVarPy(v);

        let other = SetUnion::Expr(SetExprPy(SetExpression::Reference(
            ReferenceExpression::Variable(1),
        )));
        assert_eq!(
            expression.__rxor__(other),
            SetExprPy(SetExpression::SetOperation(
                SetOperator::Union,
                Box::new(SetExpression::SetOperation(
                    SetOperator::Difference,
                    Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                        v.id()
                    ))),
                    Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
                )),
                Box::new(SetExpression::SetOperation(
                    SetOperator::Difference,
                    Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
                    Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                        v.id()
                    ))),
                ))
            ))
        );
    }

    #[test]
    fn set_var_ror() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_set_variable("v", ob, Set::with_capacity(10));
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = SetVarPy(v);

        let other = SetUnion::Expr(SetExprPy(SetExpression::Reference(
            ReferenceExpression::Variable(1),
        )));
        assert_eq!(
            expression.__ror__(other),
            SetExprPy(SetExpression::SetOperation(
                SetOperator::Union,
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                    v.id()
                ))),
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
            ))
        );
    }

    #[test]
    fn set_var_add_element() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_set_variable("v", ob, Set::with_capacity(10));
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = SetVarPy(v);

        let element = ElementUnion::Const(0);
        assert_eq!(
            expression.add(element),
            SetExprPy(SetExpression::SetElementOperation(
                SetElementOperator::Add,
                ElementExpression::Constant(0),
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                    v.id()
                ))),
            ))
        );
    }

    #[test]
    fn set_var_remove_element() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_set_variable("v", ob, Set::with_capacity(10));
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = SetVarPy(v);

        let element = ElementUnion::Const(0);
        assert_eq!(
            expression.remove(element),
            SetExprPy(SetExpression::SetElementOperation(
                SetElementOperator::Remove,
                ElementExpression::Constant(0),
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                    v.id()
                ))),
            ))
        );
    }

    #[test]
    fn set_var_difference() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_set_variable("v", ob, Set::with_capacity(10));
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = SetVarPy(v);

        let other = SetUnion::Expr(SetExprPy(SetExpression::Reference(
            ReferenceExpression::Variable(1),
        )));
        assert_eq!(
            expression.difference(other),
            SetExprPy(SetExpression::SetOperation(
                SetOperator::Difference,
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                    v.id()
                ))),
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
            ))
        );
    }

    #[test]
    fn set_var_intersection() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_set_variable("v", ob, Set::with_capacity(10));
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = SetVarPy(v);

        let other = SetUnion::Expr(SetExprPy(SetExpression::Reference(
            ReferenceExpression::Variable(1),
        )));
        assert_eq!(
            expression.intersection(other),
            SetExprPy(SetExpression::SetOperation(
                SetOperator::Intersection,
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                    v.id()
                ))),
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
            ))
        );
    }

    #[test]
    fn set_var_symmetric_difference() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_set_variable("v", ob, Set::with_capacity(10));
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = SetVarPy(v);

        let other = SetUnion::Expr(SetExprPy(SetExpression::Reference(
            ReferenceExpression::Variable(1),
        )));
        assert_eq!(
            expression.symmetric_difference(other),
            SetExprPy(SetExpression::SetOperation(
                SetOperator::Union,
                Box::new(SetExpression::SetOperation(
                    SetOperator::Difference,
                    Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                        v.id()
                    ))),
                    Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
                )),
                Box::new(SetExpression::SetOperation(
                    SetOperator::Difference,
                    Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
                    Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                        v.id()
                    ))),
                ))
            ))
        );
    }

    #[test]
    fn set_var_union() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_set_variable("v", ob, Set::with_capacity(10));
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = SetVarPy(v);

        let other = SetUnion::Expr(SetExprPy(SetExpression::Reference(
            ReferenceExpression::Variable(1),
        )));
        assert_eq!(
            expression.union(other),
            SetExprPy(SetExpression::SetOperation(
                SetOperator::Union,
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                    v.id()
                ))),
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
            ))
        );
    }

    #[test]
    fn set_var_contains_element() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_set_variable("v", ob, Set::with_capacity(10));
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = SetVarPy(v);

        let element = ElementUnion::Const(0);
        assert_eq!(
            expression.contains(element),
            ConditionPy(Condition::Set(Box::new(SetCondition::IsIn(
                ElementExpression::Constant(0),
                SetExpression::Reference(ReferenceExpression::Variable(v.id())),
            ))))
        );
    }

    #[test]
    fn set_var_len() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_set_variable("v", ob, Set::with_capacity(10));
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = SetVarPy(v);

        assert_eq!(
            expression.len(),
            IntExprPy(IntegerExpression::Cardinality(SetExpression::Reference(
                ReferenceExpression::Variable(v.id())
            )))
        );
    }

    #[test]
    fn set_var_is_empty() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_set_variable("v", ob, Set::with_capacity(10));
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = SetVarPy(v);

        assert_eq!(
            expression.is_empty(),
            ConditionPy(Condition::Set(Box::new(SetCondition::IsEmpty(
                SetExpression::Reference(ReferenceExpression::Variable(v.id()))
            ))))
        );
    }

    #[test]
    fn set_var_complement() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_set_variable("v", ob, Set::with_capacity(10));
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = SetVarPy(v);

        assert_eq!(
            expression.complement(),
            SetExprPy(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(v.id()))
            )))
        );
    }

    #[test]
    fn set_const_to_set() {
        let expression = SetConstPy(Set::with_capacity(10));
        assert_eq!(Set::from(expression), Set::with_capacity(10));
    }

    #[test]
    fn set_const_to_expression() {
        let expression = SetConstPy(Set::with_capacity(10));
        assert_eq!(
            SetExpression::from(expression),
            SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(10)))
        );
    }

    #[test]
    fn set_const_new() {
        assert_eq!(
            SetConstPy::from(Set::with_capacity(10)),
            SetConstPy(Set::with_capacity(10))
        );
    }

    #[test]
    fn set_const_lt() {
        let expression = SetConstPy(Set::with_capacity(10));
        let other = SetUnion::Expr(SetExprPy(SetExpression::Reference(
            ReferenceExpression::Variable(1),
        )));
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Lt),
            ConditionPy(Condition::And(
                Box::new(Condition::Set(Box::new(SetCondition::IsSubset(
                    SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(10))),
                    SetExpression::Reference(ReferenceExpression::Variable(1)),
                )))),
                Box::new(Condition::Not(Box::new(Condition::Set(Box::new(
                    SetCondition::IsSubset(
                        SetExpression::Reference(ReferenceExpression::Variable(1)),
                        SetExpression::Reference(ReferenceExpression::Constant(
                            Set::with_capacity(10)
                        )),
                    )
                )))),)
            ))
        );
    }

    #[test]
    fn set_const_le() {
        let expression = SetConstPy(Set::with_capacity(10));
        let other = SetUnion::Expr(SetExprPy(SetExpression::Reference(
            ReferenceExpression::Variable(1),
        )));
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Le),
            ConditionPy(Condition::Set(Box::new(SetCondition::IsSubset(
                SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(10))),
                SetExpression::Reference(ReferenceExpression::Variable(1)),
            ))))
        );
    }

    #[test]
    fn set_const_eq() {
        let expression = SetConstPy(Set::with_capacity(10));
        let other = SetUnion::Expr(SetExprPy(SetExpression::Reference(
            ReferenceExpression::Variable(1),
        )));
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Eq),
            ConditionPy(Condition::Set(Box::new(SetCondition::IsEqual(
                SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(10))),
                SetExpression::Reference(ReferenceExpression::Variable(1)),
            ))))
        );
    }

    #[test]
    fn set_const_ne() {
        let expression = SetConstPy(Set::with_capacity(10));
        let other = SetUnion::Expr(SetExprPy(SetExpression::Reference(
            ReferenceExpression::Variable(1),
        )));
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Ne),
            ConditionPy(Condition::Set(Box::new(SetCondition::IsNotEqual(
                SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(10))),
                SetExpression::Reference(ReferenceExpression::Variable(1)),
            ))))
        );
    }

    #[test]
    fn set_const_gt() {
        let expression = SetConstPy(Set::with_capacity(10));
        let other = SetUnion::Expr(SetExprPy(SetExpression::Reference(
            ReferenceExpression::Variable(1),
        )));
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Gt),
            ConditionPy(Condition::And(
                Box::new(Condition::Set(Box::new(SetCondition::IsSubset(
                    SetExpression::Reference(ReferenceExpression::Variable(1)),
                    SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(10))),
                )))),
                Box::new(Condition::Not(Box::new(Condition::Set(Box::new(
                    SetCondition::IsSubset(
                        SetExpression::Reference(ReferenceExpression::Constant(
                            Set::with_capacity(10)
                        )),
                        SetExpression::Reference(ReferenceExpression::Variable(1)),
                    )
                )))),)
            ))
        );
    }

    #[test]
    fn set_const_ge() {
        let expression = SetConstPy(Set::with_capacity(10));
        let other = SetUnion::Expr(SetExprPy(SetExpression::Reference(
            ReferenceExpression::Variable(1),
        )));
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Ge),
            ConditionPy(Condition::Set(Box::new(SetCondition::IsSubset(
                SetExpression::Reference(ReferenceExpression::Variable(1)),
                SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(10))),
            ))))
        );
    }

    #[test]
    fn set_const_sub() {
        let expression = SetConstPy(Set::with_capacity(10));
        let other = SetUnion::Expr(SetExprPy(SetExpression::Reference(
            ReferenceExpression::Variable(1),
        )));
        assert_eq!(
            expression.__sub__(other),
            SetExprPy(SetExpression::SetOperation(
                SetOperator::Difference,
                Box::new(SetExpression::Reference(ReferenceExpression::Constant(
                    Set::with_capacity(10)
                ))),
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
            ))
        );
    }

    #[test]
    fn set_const_and() {
        let expression = SetConstPy(Set::with_capacity(10));
        let other = SetUnion::Expr(SetExprPy(SetExpression::Reference(
            ReferenceExpression::Variable(1),
        )));
        assert_eq!(
            expression.__and__(other),
            SetExprPy(SetExpression::SetOperation(
                SetOperator::Intersection,
                Box::new(SetExpression::Reference(ReferenceExpression::Constant(
                    Set::with_capacity(10)
                ))),
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
            ))
        );
    }

    #[test]
    fn set_const_xor() {
        let expression = SetConstPy(Set::with_capacity(10));
        let other = SetUnion::Expr(SetExprPy(SetExpression::Reference(
            ReferenceExpression::Variable(1),
        )));
        assert_eq!(
            expression.__xor__(other),
            SetExprPy(SetExpression::SetOperation(
                SetOperator::Union,
                Box::new(SetExpression::SetOperation(
                    SetOperator::Difference,
                    Box::new(SetExpression::Reference(ReferenceExpression::Constant(
                        Set::with_capacity(10)
                    ))),
                    Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
                )),
                Box::new(SetExpression::SetOperation(
                    SetOperator::Difference,
                    Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
                    Box::new(SetExpression::Reference(ReferenceExpression::Constant(
                        Set::with_capacity(10)
                    ))),
                ))
            ))
        );
    }

    #[test]
    fn set_const_or() {
        let expression = SetConstPy(Set::with_capacity(10));
        let other = SetUnion::Expr(SetExprPy(SetExpression::Reference(
            ReferenceExpression::Variable(1),
        )));
        assert_eq!(
            expression.__or__(other),
            SetExprPy(SetExpression::SetOperation(
                SetOperator::Union,
                Box::new(SetExpression::Reference(ReferenceExpression::Constant(
                    Set::with_capacity(10)
                ))),
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
            ))
        );
    }

    #[test]
    fn set_const_rsub() {
        let expression = SetConstPy(Set::with_capacity(10));
        let other = SetUnion::Expr(SetExprPy(SetExpression::Reference(
            ReferenceExpression::Variable(1),
        )));
        assert_eq!(
            expression.__rsub__(other),
            SetExprPy(SetExpression::SetOperation(
                SetOperator::Difference,
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
                Box::new(SetExpression::Reference(ReferenceExpression::Constant(
                    Set::with_capacity(10)
                ))),
            ))
        );
    }

    #[test]
    fn set_const_rand() {
        let expression = SetConstPy(Set::with_capacity(10));
        let other = SetUnion::Expr(SetExprPy(SetExpression::Reference(
            ReferenceExpression::Variable(1),
        )));
        assert_eq!(
            expression.__rand__(other),
            SetExprPy(SetExpression::SetOperation(
                SetOperator::Intersection,
                Box::new(SetExpression::Reference(ReferenceExpression::Constant(
                    Set::with_capacity(10)
                ))),
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
            ))
        );
    }

    #[test]
    fn set_const_rxor() {
        let expression = SetConstPy(Set::with_capacity(10));
        let other = SetUnion::Expr(SetExprPy(SetExpression::Reference(
            ReferenceExpression::Variable(1),
        )));
        assert_eq!(
            expression.__rxor__(other),
            SetExprPy(SetExpression::SetOperation(
                SetOperator::Union,
                Box::new(SetExpression::SetOperation(
                    SetOperator::Difference,
                    Box::new(SetExpression::Reference(ReferenceExpression::Constant(
                        Set::with_capacity(10)
                    ))),
                    Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
                )),
                Box::new(SetExpression::SetOperation(
                    SetOperator::Difference,
                    Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
                    Box::new(SetExpression::Reference(ReferenceExpression::Constant(
                        Set::with_capacity(10)
                    ))),
                ))
            ))
        );
    }

    #[test]
    fn set_const_ror() {
        let expression = SetConstPy(Set::with_capacity(10));
        let other = SetUnion::Expr(SetExprPy(SetExpression::Reference(
            ReferenceExpression::Variable(1),
        )));
        assert_eq!(
            expression.__ror__(other),
            SetExprPy(SetExpression::SetOperation(
                SetOperator::Union,
                Box::new(SetExpression::Reference(ReferenceExpression::Constant(
                    Set::with_capacity(10)
                ))),
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
            ))
        );
    }

    #[test]
    fn set_const_add_element() {
        let expression = SetConstPy(Set::with_capacity(10));
        let element = ElementUnion::Const(0);
        assert_eq!(
            expression.add(element),
            SetExprPy(SetExpression::SetElementOperation(
                SetElementOperator::Add,
                ElementExpression::Constant(0),
                Box::new(SetExpression::Reference(ReferenceExpression::Constant(
                    Set::with_capacity(10)
                ))),
            ))
        );
    }

    #[test]
    fn set_const_remove_element() {
        let expression = SetConstPy(Set::with_capacity(10));
        let element = ElementUnion::Const(0);
        assert_eq!(
            expression.remove(element),
            SetExprPy(SetExpression::SetElementOperation(
                SetElementOperator::Remove,
                ElementExpression::Constant(0),
                Box::new(SetExpression::Reference(ReferenceExpression::Constant(
                    Set::with_capacity(10)
                ))),
            ))
        );
    }

    #[test]
    fn set_const_difference_ok() {
        let expression = SetConstPy(Set::with_capacity(10));
        let other = SetUnion::Expr(SetExprPy(SetExpression::Reference(
            ReferenceExpression::Variable(1),
        )));
        assert_eq!(
            expression.difference(other),
            SetExprPy(SetExpression::SetOperation(
                SetOperator::Difference,
                Box::new(SetExpression::Reference(ReferenceExpression::Constant(
                    Set::with_capacity(10)
                ))),
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
            ))
        );
    }

    #[test]
    fn set_const_intersection_ok() {
        let expression = SetConstPy(Set::with_capacity(10));
        let other = SetUnion::Expr(SetExprPy(SetExpression::Reference(
            ReferenceExpression::Variable(1),
        )));
        assert_eq!(
            expression.intersection(other),
            SetExprPy(SetExpression::SetOperation(
                SetOperator::Intersection,
                Box::new(SetExpression::Reference(ReferenceExpression::Constant(
                    Set::with_capacity(10)
                ))),
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
            ))
        );
    }

    #[test]
    fn set_const_symmetric_difference_ok() {
        let expression = SetConstPy(Set::with_capacity(10));
        let other = SetUnion::Expr(SetExprPy(SetExpression::Reference(
            ReferenceExpression::Variable(1),
        )));
        assert_eq!(
            expression.symmetric_difference(other),
            SetExprPy(SetExpression::SetOperation(
                SetOperator::Union,
                Box::new(SetExpression::SetOperation(
                    SetOperator::Difference,
                    Box::new(SetExpression::Reference(ReferenceExpression::Constant(
                        Set::with_capacity(10)
                    ))),
                    Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
                )),
                Box::new(SetExpression::SetOperation(
                    SetOperator::Difference,
                    Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
                    Box::new(SetExpression::Reference(ReferenceExpression::Constant(
                        Set::with_capacity(10)
                    ))),
                ))
            ))
        );
    }

    #[test]
    fn set_const_union() {
        let expression = SetConstPy(Set::with_capacity(10));
        let other = SetUnion::Expr(SetExprPy(SetExpression::Reference(
            ReferenceExpression::Variable(1),
        )));
        assert_eq!(
            expression.union(other),
            SetExprPy(SetExpression::SetOperation(
                SetOperator::Union,
                Box::new(SetExpression::Reference(ReferenceExpression::Constant(
                    Set::with_capacity(10)
                ))),
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
            ))
        );
    }

    #[test]
    fn set_const_contains() {
        let expression = SetConstPy(Set::with_capacity(10));
        let element = ElementUnion::Const(0);
        assert_eq!(
            expression.contains(element),
            ConditionPy(Condition::Set(Box::new(SetCondition::IsIn(
                ElementExpression::Constant(0),
                SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(10))),
            ))))
        );
    }

    #[test]
    fn set_const_len() {
        let expression = SetConstPy(Set::with_capacity(10));
        assert_eq!(
            expression.len(),
            IntExprPy(IntegerExpression::Cardinality(SetExpression::Reference(
                ReferenceExpression::Constant(Set::with_capacity(10))
            )))
        );
    }

    #[test]
    fn set_const_is_empty() {
        let expression = SetConstPy(Set::with_capacity(10));
        assert_eq!(
            expression.is_empty(),
            ConditionPy(Condition::Set(Box::new(SetCondition::IsEmpty(
                SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(10)))
            ))))
        );
    }

    #[test]
    fn set_const_complement() {
        let expression = SetConstPy(Set::with_capacity(10));
        assert_eq!(
            expression.complement(),
            SetExprPy(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(10)))
            )))
        );
    }

    #[test]
    fn integer_expression_from_expr() {
        let int = IntUnion::Expr(IntExprPy(IntegerExpression::Constant(0)));
        assert_eq!(IntegerExpression::from(int), IntegerExpression::Constant(0));
    }

    #[test]
    fn integer_expression_from_var() {
        let mut model = Model::default();
        let v = model.add_integer_variable("v", 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let int = IntUnion::Var(IntVarPy(v));
        assert_eq!(
            IntegerExpression::from(int),
            IntegerExpression::Variable(v.id())
        );
    }

    #[test]
    fn integer_expression_from_resource_var() {
        let mut model = Model::default();
        let v = model.add_integer_resource_variable("v", false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let int = IntUnion::ResourceVar(IntResourceVarPy(v));
        assert_eq!(
            IntegerExpression::from(int),
            IntegerExpression::ResourceVariable(v.id())
        );
    }

    #[test]
    fn integer_expression_from_const() {
        let int = IntUnion::Const(0);
        assert_eq!(IntegerExpression::from(int), IntegerExpression::Constant(0));
    }

    #[test]
    fn int_expr_to_expression() {
        let expression = IntExprPy(IntegerExpression::Constant(0));
        assert_eq!(
            IntegerExpression::from(expression),
            IntegerExpression::Constant(0)
        );
    }

    #[test]
    fn int_expr_new() {
        assert_eq!(
            IntExprPy::from(IntegerExpression::Constant(0)),
            IntExprPy(IntegerExpression::Constant(0))
        );
    }

    #[test]
    fn int_expr_new_py() {
        assert_eq!(
            IntExprPy::new_py(0),
            IntExprPy(IntegerExpression::Constant(0))
        );
    }

    #[test]
    fn int_expr_cost() {
        assert_eq!(IntExprPy::state_cost(), IntExprPy(IntegerExpression::Cost));
    }

    #[test]
    fn int_expr_lt_int() {
        let expression = IntExprPy(IntegerExpression::Constant(0));
        let other = IntOrFloatUnion::Int(IntUnion::Const(1));
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Lt),
            ConditionPy(Condition::ComparisonI(
                ComparisonOperator::Lt,
                Box::new(IntegerExpression::Constant(0)),
                Box::new(IntegerExpression::Constant(1))
            ))
        );
    }

    #[test]
    fn int_expr_lt_float() {
        let expression = IntExprPy(IntegerExpression::Constant(0));
        let other = IntOrFloatUnion::Float(FloatUnion::Const(1.0));
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Lt),
            ConditionPy(Condition::ComparisonC(
                ComparisonOperator::Lt,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(0)
                ))),
                Box::new(ContinuousExpression::Constant(1.0))
            ))
        );
    }

    #[test]
    fn int_expr_le_int() {
        let expression = IntExprPy(IntegerExpression::Constant(0));
        let other = IntOrFloatUnion::Int(IntUnion::Const(1));
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Le),
            ConditionPy(Condition::ComparisonI(
                ComparisonOperator::Le,
                Box::new(IntegerExpression::Constant(0)),
                Box::new(IntegerExpression::Constant(1))
            ))
        );
    }

    #[test]
    fn int_expr_le_float() {
        let expression = IntExprPy(IntegerExpression::Constant(0));
        let other = IntOrFloatUnion::Float(FloatUnion::Const(1.0));
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Le),
            ConditionPy(Condition::ComparisonC(
                ComparisonOperator::Le,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(0)
                ))),
                Box::new(ContinuousExpression::Constant(1.0))
            ))
        );
    }

    #[test]
    fn int_expr_eq_int() {
        let expression = IntExprPy(IntegerExpression::Constant(0));
        let other = IntOrFloatUnion::Int(IntUnion::Const(1));
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Eq),
            ConditionPy(Condition::ComparisonI(
                ComparisonOperator::Eq,
                Box::new(IntegerExpression::Constant(0)),
                Box::new(IntegerExpression::Constant(1))
            ))
        );
    }

    #[test]
    fn int_expr_eq_float() {
        let expression = IntExprPy(IntegerExpression::Constant(0));
        let other = IntOrFloatUnion::Float(FloatUnion::Const(1.0));
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Eq),
            ConditionPy(Condition::ComparisonC(
                ComparisonOperator::Eq,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(0)
                ))),
                Box::new(ContinuousExpression::Constant(1.0))
            ))
        );
    }

    #[test]
    fn int_expr_ne_int() {
        let expression = IntExprPy(IntegerExpression::Constant(0));
        let other = IntOrFloatUnion::Int(IntUnion::Const(1));
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Ne),
            ConditionPy(Condition::ComparisonI(
                ComparisonOperator::Ne,
                Box::new(IntegerExpression::Constant(0)),
                Box::new(IntegerExpression::Constant(1))
            ))
        );
    }

    #[test]
    fn int_expr_ne_float() {
        let expression = IntExprPy(IntegerExpression::Constant(0));
        let other = IntOrFloatUnion::Float(FloatUnion::Const(1.0));
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Ne),
            ConditionPy(Condition::ComparisonC(
                ComparisonOperator::Ne,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(0)
                ))),
                Box::new(ContinuousExpression::Constant(1.0))
            ))
        );
    }

    #[test]
    fn int_expr_gt_int() {
        let expression = IntExprPy(IntegerExpression::Constant(0));
        let other = IntOrFloatUnion::Int(IntUnion::Const(1));
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Gt),
            ConditionPy(Condition::ComparisonI(
                ComparisonOperator::Gt,
                Box::new(IntegerExpression::Constant(0)),
                Box::new(IntegerExpression::Constant(1))
            ))
        );
    }

    #[test]
    fn int_expr_gt_float() {
        let expression = IntExprPy(IntegerExpression::Constant(0));
        let other = IntOrFloatUnion::Float(FloatUnion::Const(1.0));
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Gt),
            ConditionPy(Condition::ComparisonC(
                ComparisonOperator::Gt,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(0)
                ))),
                Box::new(ContinuousExpression::Constant(1.0))
            ))
        );
    }

    #[test]
    fn int_expr_ge_int() {
        let expression = IntExprPy(IntegerExpression::Constant(0));
        let other = IntOrFloatUnion::Int(IntUnion::Const(1));
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Ge),
            ConditionPy(Condition::ComparisonI(
                ComparisonOperator::Ge,
                Box::new(IntegerExpression::Constant(0)),
                Box::new(IntegerExpression::Constant(1))
            ))
        );
    }

    #[test]
    fn int_expr_ge_float() {
        let expression = IntExprPy(IntegerExpression::Constant(0));
        let other = IntOrFloatUnion::Float(FloatUnion::Const(1.0));
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Ge),
            ConditionPy(Condition::ComparisonC(
                ComparisonOperator::Ge,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(0)
                ))),
                Box::new(ContinuousExpression::Constant(1.0))
            ))
        );
    }

    #[test]
    fn int_expr_abs() {
        let expression = IntExprPy(IntegerExpression::Constant(-1));
        assert_eq!(
            expression.__abs__(),
            IntExprPy(IntegerExpression::UnaryOperation(
                UnaryOperator::Abs,
                Box::new(IntegerExpression::Constant(-1))
            ))
        );
    }

    #[test]
    fn int_exp_neg() {
        let expression = IntExprPy(IntegerExpression::Constant(-1));
        assert_eq!(
            expression.__neg__(),
            IntExprPy(IntegerExpression::UnaryOperation(
                UnaryOperator::Neg,
                Box::new(IntegerExpression::Constant(-1))
            ))
        );
    }

    #[test]
    fn int_expr_add_int() {
        let expression = IntExprPy(IntegerExpression::Constant(0));
        let other = IntOrFloatUnion::Int(IntUnion::Const(1));
        assert_eq!(
            expression.__add__(other),
            IntOrFloatExpr::Int(IntExprPy(IntegerExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(IntegerExpression::Constant(0)),
                Box::new(IntegerExpression::Constant(1)),
            )))
        );
    }

    #[test]
    fn int_expr_add_float() {
        let expression = IntExprPy(IntegerExpression::Constant(0));
        let other = IntOrFloatUnion::Float(FloatUnion::Const(1.0));
        assert_eq!(
            expression.__add__(other),
            IntOrFloatExpr::Float(FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(0)
                ))),
                Box::new(ContinuousExpression::Constant(1.0)),
            )))
        );
    }

    #[test]
    fn int_expr_sub_int() {
        let expression = IntExprPy(IntegerExpression::Constant(0));
        let other = IntOrFloatUnion::Int(IntUnion::Const(1));
        assert_eq!(
            expression.__sub__(other),
            IntOrFloatExpr::Int(IntExprPy(IntegerExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(IntegerExpression::Constant(0)),
                Box::new(IntegerExpression::Constant(1)),
            )))
        );
    }

    #[test]
    fn int_expr_sub_float() {
        let expression = IntExprPy(IntegerExpression::Constant(0));
        let other = IntOrFloatUnion::Float(FloatUnion::Const(1.0));
        assert_eq!(
            expression.__sub__(other),
            IntOrFloatExpr::Float(FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(0)
                ))),
                Box::new(ContinuousExpression::Constant(1.0))
            )))
        );
    }

    #[test]
    fn int_expr_mul_int() {
        let expression = IntExprPy(IntegerExpression::Constant(0));
        let other = IntOrFloatUnion::Int(IntUnion::Const(1));
        assert_eq!(
            expression.__mul__(other),
            IntOrFloatExpr::Int(IntExprPy(IntegerExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(IntegerExpression::Constant(0)),
                Box::new(IntegerExpression::Constant(1)),
            )))
        );
    }

    #[test]
    fn int_expr_mul_float() {
        let expression = IntExprPy(IntegerExpression::Constant(0));
        let other = IntOrFloatUnion::Float(FloatUnion::Const(1.0));
        assert_eq!(
            expression.__mul__(other),
            IntOrFloatExpr::Float(FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(0)
                ))),
                Box::new(ContinuousExpression::Constant(1.0)),
            )))
        );
    }

    #[test]
    fn int_expr_truediv() {
        let expression = IntExprPy(IntegerExpression::Constant(0));
        let other = FloatUnion::IntConst(1);
        assert_eq!(
            expression.__truediv__(other),
            FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(0)
                ))),
                Box::new(ContinuousExpression::Constant(1.0))
            ))
        );
    }

    #[test]
    fn int_expr_floordiv_int() {
        let expression = IntExprPy(IntegerExpression::Constant(0));
        let other = IntOrFloatUnion::Int(IntUnion::Const(1));
        assert_eq!(
            expression.__floordiv__(other),
            IntOrFloatExpr::Int(IntExprPy(IntegerExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(IntegerExpression::Constant(0)),
                Box::new(IntegerExpression::Constant(1)),
            )))
        );
    }

    #[test]
    fn int_expr_floordiv_float() {
        let expression = IntExprPy(IntegerExpression::Constant(0));
        let other = IntOrFloatUnion::Float(FloatUnion::Const(1.0));
        assert_eq!(
            expression.__floordiv__(other),
            IntOrFloatExpr::Float(FloatExprPy(ContinuousExpression::Round(
                CastOperator::Floor,
                Box::new(ContinuousExpression::BinaryOperation(
                    BinaryOperator::Div,
                    Box::new(ContinuousExpression::FromInteger(Box::new(
                        IntegerExpression::Constant(0)
                    ))),
                    Box::new(ContinuousExpression::Constant(1.0)),
                ))
            )))
        );
    }

    #[test]
    fn int_expr_mod_int() {
        let expression = IntExprPy(IntegerExpression::Constant(0));
        let other = IntOrFloatUnion::Int(IntUnion::Const(1));
        assert_eq!(
            expression.__mod__(other),
            IntOrFloatExpr::Int(IntExprPy(IntegerExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(IntegerExpression::Constant(0)),
                Box::new(IntegerExpression::Constant(1)),
            )))
        );
    }

    #[test]
    fn int_expr_mod_float() {
        let expression = IntExprPy(IntegerExpression::Constant(0));
        let other = IntOrFloatUnion::Float(FloatUnion::Const(1.0));
        assert_eq!(
            expression.__mod__(other),
            IntOrFloatExpr::Float(FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(0)
                ))),
                Box::new(ContinuousExpression::Constant(1.0)),
            )))
        );
    }

    #[test]
    fn int_expr_pow() {
        let expression = IntExprPy(IntegerExpression::Constant(0));
        let other = FloatUnion::IntConst(1);
        assert_eq!(
            expression.__pow__(other, None),
            FloatExprPy(ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Pow,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(0)
                ))),
                Box::new(ContinuousExpression::Constant(1.0))
            ))
        );
    }

    #[test]
    fn int_expr_pow_with_modulo() {
        let expression = IntExprPy(IntegerExpression::Constant(0));
        let other = FloatUnion::IntConst(1);
        let modulo = FloatUnion::IntConst(2);
        assert_eq!(
            expression.__pow__(other, Some(modulo)),
            FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::ContinuousBinaryOperation(
                    ContinuousBinaryOperator::Pow,
                    Box::new(ContinuousExpression::FromInteger(Box::new(
                        IntegerExpression::Constant(0)
                    ))),
                    Box::new(ContinuousExpression::Constant(1.0))
                )),
                Box::new(ContinuousExpression::Constant(2.0))
            ))
        );
    }

    #[test]
    fn int_expr_radd_int() {
        let expression = IntExprPy(IntegerExpression::Constant(0));
        let other = IntOrFloatUnion::Int(IntUnion::Const(1));
        assert_eq!(
            expression.__radd__(other),
            IntOrFloatExpr::Int(IntExprPy(IntegerExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(IntegerExpression::Constant(0)),
                Box::new(IntegerExpression::Constant(1)),
            )))
        );
    }

    #[test]
    fn int_expr_radd_float() {
        let expression = IntExprPy(IntegerExpression::Constant(0));
        let other = IntOrFloatUnion::Float(FloatUnion::Const(1.0));
        assert_eq!(
            expression.__radd__(other),
            IntOrFloatExpr::Float(FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(0)
                ))),
                Box::new(ContinuousExpression::Constant(1.0)),
            )))
        );
    }

    #[test]
    fn int_expr_rsub_int() {
        let expression = IntExprPy(IntegerExpression::Constant(0));
        let other = IntOrFloatUnion::Int(IntUnion::Const(1));
        assert_eq!(
            expression.__rsub__(other),
            IntOrFloatExpr::Int(IntExprPy(IntegerExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::Constant(0)),
            )))
        );
    }

    #[test]
    fn int_expr_rsub_float() {
        let expression = IntExprPy(IntegerExpression::Constant(0));
        let other = IntOrFloatUnion::Float(FloatUnion::Const(1.0));
        assert_eq!(
            expression.__rsub__(other),
            IntOrFloatExpr::Float(FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(0)
                ))),
            )))
        );
    }

    #[test]
    fn int_expr_rmul_int() {
        let expression = IntExprPy(IntegerExpression::Constant(0));
        let other = IntOrFloatUnion::Int(IntUnion::Const(1));
        assert_eq!(
            expression.__rmul__(other),
            IntOrFloatExpr::Int(IntExprPy(IntegerExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(IntegerExpression::Constant(0)),
                Box::new(IntegerExpression::Constant(1)),
            )))
        );
    }

    #[test]
    fn int_expr_rmul_float() {
        let expression = IntExprPy(IntegerExpression::Constant(0));
        let other = IntOrFloatUnion::Float(FloatUnion::Const(1.0));
        assert_eq!(
            expression.__rmul__(other),
            IntOrFloatExpr::Float(FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(0)
                ))),
                Box::new(ContinuousExpression::Constant(1.0)),
            )))
        );
    }

    #[test]
    fn int_expr_rtruediv() {
        let expression = IntExprPy(IntegerExpression::Constant(0));
        let other = FloatUnion::IntConst(1);
        assert_eq!(
            expression.__rtruediv__(other),
            FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(0)
                ))),
            ))
        );
    }

    #[test]
    fn int_expr_rfloordiv_int() {
        let expression = IntExprPy(IntegerExpression::Constant(0));
        let other = IntOrFloatUnion::Int(IntUnion::Const(1));
        assert_eq!(
            expression.__rfloordiv__(other),
            IntOrFloatExpr::Int(IntExprPy(IntegerExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::Constant(0)),
            )))
        );
    }

    #[test]
    fn int_expr_rfloordiv_float() {
        let expression = IntExprPy(IntegerExpression::Constant(0));
        let other = IntOrFloatUnion::Float(FloatUnion::Const(1.0));
        assert_eq!(
            expression.__rfloordiv__(other),
            IntOrFloatExpr::Float(FloatExprPy(ContinuousExpression::Round(
                CastOperator::Floor,
                Box::new(ContinuousExpression::BinaryOperation(
                    BinaryOperator::Div,
                    Box::new(ContinuousExpression::Constant(1.0)),
                    Box::new(ContinuousExpression::FromInteger(Box::new(
                        IntegerExpression::Constant(0)
                    ))),
                ))
            )))
        );
    }

    #[test]
    fn int_expr_rmod_int() {
        let expression = IntExprPy(IntegerExpression::Constant(0));
        let other = IntOrFloatUnion::Int(IntUnion::Const(1));
        assert_eq!(
            expression.__rmod__(other),
            IntOrFloatExpr::Int(IntExprPy(IntegerExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::Constant(0)),
            )))
        );
    }

    #[test]
    fn int_expr_rmod_float() {
        let expression = IntExprPy(IntegerExpression::Constant(0));
        let other = IntOrFloatUnion::Float(FloatUnion::Const(1.0));
        assert_eq!(
            expression.__rmod__(other),
            IntOrFloatExpr::Float(FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(0)
                ))),
            )))
        );
    }

    #[test]
    fn int_expr_rpow() {
        let expression = IntExprPy(IntegerExpression::Constant(0));
        let other = FloatUnion::IntConst(1);
        assert_eq!(
            expression.__rpow__(other, None),
            FloatExprPy(ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Pow,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(0)
                ))),
            ))
        );
    }

    #[test]
    fn int_expr_rpow_with_modulo() {
        let expression = IntExprPy(IntegerExpression::Constant(0));
        let other = FloatUnion::IntConst(1);
        let modulo = FloatUnion::IntConst(2);
        assert_eq!(
            expression.__rpow__(other, Some(modulo)),
            FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::ContinuousBinaryOperation(
                    ContinuousBinaryOperator::Pow,
                    Box::new(ContinuousExpression::Constant(1.0)),
                    Box::new(ContinuousExpression::FromInteger(Box::new(
                        IntegerExpression::Constant(0)
                    ))),
                )),
                Box::new(ContinuousExpression::Constant(2.0))
            ))
        );
    }

    #[test]
    fn int_var_to_variable() {
        let mut model = Model::default();
        let v = model.add_integer_variable("v", 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let v_py = IntVarPy(v);
        assert_eq!(IntegerVariable::from(v_py), v);
    }

    #[test]
    fn int_var_to_expression() {
        let mut model = Model::default();
        let v = model.add_integer_variable("v", 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let v_py = IntVarPy(v);
        assert_eq!(
            IntegerExpression::from(v_py),
            IntegerExpression::Variable(v.id())
        );
    }

    #[test]
    fn int_var_new() {
        let mut model = Model::default();
        let v = model.add_integer_variable("v", 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        assert_eq!(IntVarPy::from(v), IntVarPy(v));
    }

    #[test]
    fn int_var_lt_int() {
        let mut model = Model::default();
        let v = model.add_integer_variable("v", 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntVarPy(v);

        let other = IntOrFloatUnion::Int(IntUnion::Const(1));
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Lt),
            ConditionPy(Condition::ComparisonI(
                ComparisonOperator::Lt,
                Box::new(IntegerExpression::Variable(v.id())),
                Box::new(IntegerExpression::Constant(1)),
            ))
        );
    }

    #[test]
    fn int_var_lt_float() {
        let mut model = Model::default();
        let v = model.add_integer_variable("v", 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntVarPy(v);

        let other = IntOrFloatUnion::Float(FloatUnion::Const(1.0));
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Lt),
            ConditionPy(Condition::ComparisonC(
                ComparisonOperator::Lt,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(v.id())
                ))),
                Box::new(ContinuousExpression::Constant(1.0)),
            ))
        );
    }

    #[test]
    fn int_var_le_int() {
        let mut model = Model::default();
        let v = model.add_integer_variable("v", 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntVarPy(v);

        let other = IntOrFloatUnion::Int(IntUnion::Const(1));
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Le),
            ConditionPy(Condition::ComparisonI(
                ComparisonOperator::Le,
                Box::new(IntegerExpression::Variable(v.id())),
                Box::new(IntegerExpression::Constant(1)),
            ))
        );
    }

    #[test]
    fn int_var_le_float() {
        let mut model = Model::default();
        let v = model.add_integer_variable("v", 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntVarPy(v);

        let other = IntOrFloatUnion::Float(FloatUnion::Const(1.0));
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Le),
            ConditionPy(Condition::ComparisonC(
                ComparisonOperator::Le,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(v.id())
                ))),
                Box::new(ContinuousExpression::Constant(1.0)),
            ))
        );
    }

    #[test]
    fn int_var_eq_int() {
        let mut model = Model::default();
        let v = model.add_integer_variable("v", 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntVarPy(v);

        let other = IntOrFloatUnion::Int(IntUnion::Const(1));
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Eq),
            ConditionPy(Condition::ComparisonI(
                ComparisonOperator::Eq,
                Box::new(IntegerExpression::Variable(v.id())),
                Box::new(IntegerExpression::Constant(1)),
            ))
        );
    }

    #[test]
    fn int_var_eq_float() {
        let mut model = Model::default();
        let v = model.add_integer_variable("v", 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntVarPy(v);

        let other = IntOrFloatUnion::Float(FloatUnion::Const(1.0));
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Eq),
            ConditionPy(Condition::ComparisonC(
                ComparisonOperator::Eq,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(v.id())
                ))),
                Box::new(ContinuousExpression::Constant(1.0)),
            ))
        );
    }

    #[test]
    fn int_var_ne_int() {
        let mut model = Model::default();
        let v = model.add_integer_variable("v", 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntVarPy(v);

        let other = IntOrFloatUnion::Int(IntUnion::Const(1));
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Ne),
            ConditionPy(Condition::ComparisonI(
                ComparisonOperator::Ne,
                Box::new(IntegerExpression::Variable(v.id())),
                Box::new(IntegerExpression::Constant(1)),
            ))
        );
    }

    #[test]
    fn int_var_ne_float() {
        let mut model = Model::default();
        let v = model.add_integer_variable("v", 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntVarPy(v);

        let other = IntOrFloatUnion::Float(FloatUnion::Const(1.0));
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Ne),
            ConditionPy(Condition::ComparisonC(
                ComparisonOperator::Ne,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(v.id())
                ))),
                Box::new(ContinuousExpression::Constant(1.0)),
            ))
        );
    }

    #[test]
    fn int_var_gt_int() {
        let mut model = Model::default();
        let v = model.add_integer_variable("v", 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntVarPy(v);

        let other = IntOrFloatUnion::Int(IntUnion::Const(1));
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Gt),
            ConditionPy(Condition::ComparisonI(
                ComparisonOperator::Gt,
                Box::new(IntegerExpression::Variable(v.id())),
                Box::new(IntegerExpression::Constant(1)),
            ))
        );
    }

    #[test]
    fn int_var_gt_float() {
        let mut model = Model::default();
        let v = model.add_integer_variable("v", 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntVarPy(v);

        let other = IntOrFloatUnion::Float(FloatUnion::Const(1.0));
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Gt),
            ConditionPy(Condition::ComparisonC(
                ComparisonOperator::Gt,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(v.id())
                ))),
                Box::new(ContinuousExpression::Constant(1.0)),
            ))
        );
    }

    #[test]
    fn int_var_ge_int() {
        let mut model = Model::default();
        let v = model.add_integer_variable("v", 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntVarPy(v);

        let other = IntOrFloatUnion::Int(IntUnion::Const(1));
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Ge),
            ConditionPy(Condition::ComparisonI(
                ComparisonOperator::Ge,
                Box::new(IntegerExpression::Variable(v.id())),
                Box::new(IntegerExpression::Constant(1)),
            ))
        );
    }

    #[test]
    fn int_var_ge_float() {
        let mut model = Model::default();
        let v = model.add_integer_variable("v", 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntVarPy(v);

        let other = IntOrFloatUnion::Float(FloatUnion::Const(1.0));
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Ge),
            ConditionPy(Condition::ComparisonC(
                ComparisonOperator::Ge,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(v.id())
                ))),
                Box::new(ContinuousExpression::Constant(1.0)),
            ))
        );
    }

    #[test]
    fn int_var_abs() {
        let mut model = Model::default();
        let v = model.add_integer_variable("v", 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntVarPy(v);
        assert_eq!(
            expression.__abs__(),
            IntExprPy(IntegerExpression::UnaryOperation(
                UnaryOperator::Abs,
                Box::new(IntegerExpression::Variable(v.id()))
            ))
        );
    }

    #[test]
    fn int_var_neg() {
        let mut model = Model::default();
        let v = model.add_integer_variable("v", 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntVarPy(v);
        assert_eq!(
            expression.__neg__(),
            IntExprPy(IntegerExpression::UnaryOperation(
                UnaryOperator::Neg,
                Box::new(IntegerExpression::Variable(v.id()))
            ))
        );
    }

    #[test]
    fn int_var_add_int() {
        let mut model = Model::default();
        let v = model.add_integer_variable("v", 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntVarPy(v);

        let other = IntOrFloatUnion::Int(IntUnion::Const(1));
        assert_eq!(
            expression.__add__(other),
            IntOrFloatExpr::Int(IntExprPy(IntegerExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(IntegerExpression::Variable(v.id())),
                Box::new(IntegerExpression::Constant(1)),
            )))
        );
    }

    #[test]
    fn int_var_add_float() {
        let mut model = Model::default();
        let v = model.add_integer_variable("v", 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntVarPy(v);

        let other = IntOrFloatUnion::Float(FloatUnion::Const(1.0));
        assert_eq!(
            expression.__add__(other),
            IntOrFloatExpr::Float(FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(v.id())
                ))),
                Box::new(ContinuousExpression::Constant(1.0)),
            )))
        );
    }

    #[test]
    fn int_var_sub_int() {
        let mut model = Model::default();
        let v = model.add_integer_variable("v", 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntVarPy(v);

        let other = IntOrFloatUnion::Int(IntUnion::Const(1));
        assert_eq!(
            expression.__sub__(other),
            IntOrFloatExpr::Int(IntExprPy(IntegerExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(IntegerExpression::Variable(v.id())),
                Box::new(IntegerExpression::Constant(1)),
            )))
        );
    }

    #[test]
    fn int_var_sub_float() {
        let mut model = Model::default();
        let v = model.add_integer_variable("v", 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntVarPy(v);

        let other = IntOrFloatUnion::Float(FloatUnion::Const(1.0));
        assert_eq!(
            expression.__sub__(other),
            IntOrFloatExpr::Float(FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(v.id())
                ))),
                Box::new(ContinuousExpression::Constant(1.0)),
            )))
        );
    }

    #[test]
    fn int_var_mul_int() {
        let mut model = Model::default();
        let v = model.add_integer_variable("v", 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntVarPy(v);

        let other = IntOrFloatUnion::Int(IntUnion::Const(1));
        assert_eq!(
            expression.__mul__(other),
            IntOrFloatExpr::Int(IntExprPy(IntegerExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(IntegerExpression::Variable(v.id())),
                Box::new(IntegerExpression::Constant(1)),
            )))
        );
    }

    #[test]
    fn int_var_mul_float() {
        let mut model = Model::default();
        let v = model.add_integer_variable("v", 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntVarPy(v);

        let other = IntOrFloatUnion::Float(FloatUnion::Const(1.0));
        assert_eq!(
            expression.__mul__(other),
            IntOrFloatExpr::Float(FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(v.id())
                ))),
                Box::new(ContinuousExpression::Constant(1.0)),
            )))
        );
    }

    #[test]
    fn int_var_truediv() {
        let mut model = Model::default();
        let v = model.add_integer_variable("v", 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntVarPy(v);

        let other = FloatUnion::IntConst(1);
        assert_eq!(
            expression.__truediv__(other),
            FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(v.id())
                ))),
                Box::new(ContinuousExpression::Constant(1.0)),
            ))
        );
    }

    #[test]
    fn int_var_floordiv_int() {
        let mut model = Model::default();
        let v = model.add_integer_variable("v", 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntVarPy(v);

        let other = IntOrFloatUnion::Int(IntUnion::Const(1));
        assert_eq!(
            expression.__floordiv__(other),
            IntOrFloatExpr::Int(IntExprPy(IntegerExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(IntegerExpression::Variable(v.id())),
                Box::new(IntegerExpression::Constant(1)),
            )))
        );
    }

    #[test]
    fn int_var_floordiv_float() {
        let mut model = Model::default();
        let v = model.add_integer_variable("v", 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntVarPy(v);

        let other = IntOrFloatUnion::Float(FloatUnion::Const(1.0));
        assert_eq!(
            expression.__floordiv__(other),
            IntOrFloatExpr::Float(FloatExprPy(ContinuousExpression::Round(
                CastOperator::Floor,
                Box::new(ContinuousExpression::BinaryOperation(
                    BinaryOperator::Div,
                    Box::new(ContinuousExpression::FromInteger(Box::new(
                        IntegerExpression::Variable(v.id())
                    ))),
                    Box::new(ContinuousExpression::Constant(1.0)),
                ))
            )))
        );
    }

    #[test]
    fn int_var_mod_int() {
        let mut model = Model::default();
        let v = model.add_integer_variable("v", 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntVarPy(v);

        let other = IntOrFloatUnion::Int(IntUnion::Const(1));
        assert_eq!(
            expression.__mod__(other),
            IntOrFloatExpr::Int(IntExprPy(IntegerExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(IntegerExpression::Variable(v.id())),
                Box::new(IntegerExpression::Constant(1)),
            )))
        );
    }

    #[test]
    fn int_var_mod_float() {
        let mut model = Model::default();
        let v = model.add_integer_variable("v", 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntVarPy(v);

        let other = IntOrFloatUnion::Float(FloatUnion::Const(1.0));
        assert_eq!(
            expression.__mod__(other),
            IntOrFloatExpr::Float(FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(v.id())
                ))),
                Box::new(ContinuousExpression::Constant(1.0)),
            )))
        );
    }

    #[test]
    fn int_var_pow() {
        let mut model = Model::default();
        let v = model.add_integer_variable("v", 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntVarPy(v);

        let other = FloatUnion::IntConst(1);
        assert_eq!(
            expression.__pow__(other, None),
            FloatExprPy(ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Pow,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(v.id())
                ))),
                Box::new(ContinuousExpression::Constant(1.0))
            ))
        );
    }

    #[test]
    fn int_var_pow_with_modulo() {
        let mut model = Model::default();
        let v = model.add_integer_variable("v", 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntVarPy(v);

        let other = FloatUnion::IntConst(1);
        let modulo = FloatUnion::IntConst(2);
        assert_eq!(
            expression.__pow__(other, Some(modulo)),
            FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::ContinuousBinaryOperation(
                    ContinuousBinaryOperator::Pow,
                    Box::new(ContinuousExpression::FromInteger(Box::new(
                        IntegerExpression::Variable(v.id())
                    ))),
                    Box::new(ContinuousExpression::Constant(1.0))
                )),
                Box::new(ContinuousExpression::Constant(2.0))
            ))
        );
    }

    #[test]
    fn int_var_radd_int() {
        let mut model = Model::default();
        let v = model.add_integer_variable("v", 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntVarPy(v);

        let other = IntOrFloatUnion::Int(IntUnion::Const(1));
        assert_eq!(
            expression.__radd__(other),
            IntOrFloatExpr::Int(IntExprPy(IntegerExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(IntegerExpression::Variable(v.id())),
                Box::new(IntegerExpression::Constant(1)),
            )))
        );
    }

    #[test]
    fn int_var_radd_float() {
        let mut model = Model::default();
        let v = model.add_integer_variable("v", 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntVarPy(v);

        let other = IntOrFloatUnion::Float(FloatUnion::Const(1.0));
        assert_eq!(
            expression.__radd__(other),
            IntOrFloatExpr::Float(FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(v.id())
                ))),
                Box::new(ContinuousExpression::Constant(1.0)),
            )))
        );
    }

    #[test]
    fn int_var_rsub_int() {
        let mut model = Model::default();
        let v = model.add_integer_variable("v", 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntVarPy(v);

        let other = IntOrFloatUnion::Int(IntUnion::Const(1));
        assert_eq!(
            expression.__rsub__(other),
            IntOrFloatExpr::Int(IntExprPy(IntegerExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::Variable(v.id())),
            )))
        );
    }

    #[test]
    fn int_var_rsub_float() {
        let mut model = Model::default();
        let v = model.add_integer_variable("v", 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntVarPy(v);

        let other = IntOrFloatUnion::Float(FloatUnion::Const(1.0));
        assert_eq!(
            expression.__rsub__(other),
            IntOrFloatExpr::Float(FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(v.id())
                ))),
            )))
        );
    }

    #[test]
    fn int_var_rmul_int() {
        let mut model = Model::default();
        let v = model.add_integer_variable("v", 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntVarPy(v);

        let other = IntOrFloatUnion::Int(IntUnion::Const(1));
        assert_eq!(
            expression.__rmul__(other),
            IntOrFloatExpr::Int(IntExprPy(IntegerExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(IntegerExpression::Variable(v.id())),
                Box::new(IntegerExpression::Constant(1)),
            )))
        );
    }

    #[test]
    fn int_var_rmul_float() {
        let mut model = Model::default();
        let v = model.add_integer_variable("v", 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntVarPy(v);

        let other = IntOrFloatUnion::Float(FloatUnion::Const(1.0));
        assert_eq!(
            expression.__rmul__(other),
            IntOrFloatExpr::Float(FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(v.id())
                ))),
                Box::new(ContinuousExpression::Constant(1.0)),
            )))
        );
    }

    #[test]
    fn int_var_rtruediv() {
        let mut model = Model::default();
        let v = model.add_integer_variable("v", 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntVarPy(v);

        let other = FloatUnion::IntConst(1);
        assert_eq!(
            expression.__rtruediv__(other),
            FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(v.id())
                ))),
            ))
        );
    }

    #[test]
    fn int_var_rfloordiv_int() {
        let mut model = Model::default();
        let v = model.add_integer_variable("v", 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntVarPy(v);

        let other = IntOrFloatUnion::Int(IntUnion::Const(1));
        assert_eq!(
            expression.__rfloordiv__(other),
            IntOrFloatExpr::Int(IntExprPy(IntegerExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::Variable(v.id())),
            )))
        );
    }

    #[test]
    fn int_var_rfloordiv_float() {
        let mut model = Model::default();
        let v = model.add_integer_variable("v", 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntVarPy(v);

        let other = IntOrFloatUnion::Float(FloatUnion::Const(1.0));
        assert_eq!(
            expression.__rfloordiv__(other),
            IntOrFloatExpr::Float(FloatExprPy(ContinuousExpression::Round(
                CastOperator::Floor,
                Box::new(ContinuousExpression::BinaryOperation(
                    BinaryOperator::Div,
                    Box::new(ContinuousExpression::Constant(1.0)),
                    Box::new(ContinuousExpression::FromInteger(Box::new(
                        IntegerExpression::Variable(v.id())
                    ))),
                ))
            )))
        );
    }

    #[test]
    fn int_var_rmod_int() {
        let mut model = Model::default();
        let v = model.add_integer_variable("v", 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntVarPy(v);

        let other = IntOrFloatUnion::Int(IntUnion::Const(1));
        assert_eq!(
            expression.__rmod__(other),
            IntOrFloatExpr::Int(IntExprPy(IntegerExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::Variable(v.id())),
            )))
        );
    }

    #[test]
    fn int_var_rmod_float() {
        let mut model = Model::default();
        let v = model.add_integer_variable("v", 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntVarPy(v);

        let other = IntOrFloatUnion::Float(FloatUnion::Const(1.0));
        assert_eq!(
            expression.__rmod__(other),
            IntOrFloatExpr::Float(FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(v.id())
                ))),
            )))
        );
    }

    #[test]
    fn int_var_rpow() {
        let mut model = Model::default();
        let v = model.add_integer_variable("v", 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntVarPy(v);

        let other = FloatUnion::IntConst(1);
        assert_eq!(
            expression.__rpow__(other, None),
            FloatExprPy(ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Pow,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(v.id())
                ))),
            ))
        );
    }

    #[test]
    fn int_var_rpow_with_modulo() {
        let mut model = Model::default();
        let v = model.add_integer_variable("v", 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntVarPy(v);

        let other = FloatUnion::IntConst(1);
        let modulo = FloatUnion::IntConst(2);
        assert_eq!(
            expression.__rpow__(other, Some(modulo)),
            FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::ContinuousBinaryOperation(
                    ContinuousBinaryOperator::Pow,
                    Box::new(ContinuousExpression::Constant(1.0)),
                    Box::new(ContinuousExpression::FromInteger(Box::new(
                        IntegerExpression::Variable(v.id())
                    ))),
                )),
                Box::new(ContinuousExpression::Constant(2.0))
            ))
        );
    }

    #[test]
    fn int_resource_var_to_resource_variable() {
        let mut model = Model::default();
        let v = model.add_integer_resource_variable("v", false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let v_py = IntResourceVarPy(v);
        assert_eq!(IntegerResourceVariable::from(v_py), v);
    }

    #[test]
    fn int_resource_var_to_expression() {
        let mut model = Model::default();
        let v = model.add_integer_resource_variable("v", false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let v_py = IntResourceVarPy(v);
        assert_eq!(
            IntegerExpression::from(v_py),
            IntegerExpression::ResourceVariable(v.id())
        );
    }

    #[test]
    fn int_resource_var_new() {
        let mut model = Model::default();
        let v = model.add_integer_resource_variable("v", false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        assert_eq!(IntResourceVarPy::from(v), IntResourceVarPy(v));
    }

    #[test]
    fn int_resource_var_lt_int() {
        let mut model = Model::default();
        let v = model.add_integer_resource_variable("v", false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntResourceVarPy(v);

        let other = IntOrFloatUnion::Int(IntUnion::Const(1));
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Lt),
            ConditionPy(Condition::ComparisonI(
                ComparisonOperator::Lt,
                Box::new(IntegerExpression::ResourceVariable(v.id())),
                Box::new(IntegerExpression::Constant(1)),
            ))
        );
    }

    #[test]
    fn int_resource_var_lt_float() {
        let mut model = Model::default();
        let v = model.add_integer_resource_variable("v", false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntResourceVarPy(v);

        let other = IntOrFloatUnion::Float(FloatUnion::Const(1.0));
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Lt),
            ConditionPy(Condition::ComparisonC(
                ComparisonOperator::Lt,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(v.id())
                ))),
                Box::new(ContinuousExpression::Constant(1.0)),
            ))
        );
    }

    #[test]
    fn int_resource_var_le_int() {
        let mut model = Model::default();
        let v = model.add_integer_resource_variable("v", false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntResourceVarPy(v);

        let other = IntOrFloatUnion::Int(IntUnion::Const(1));
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Le),
            ConditionPy(Condition::ComparisonI(
                ComparisonOperator::Le,
                Box::new(IntegerExpression::ResourceVariable(v.id())),
                Box::new(IntegerExpression::Constant(1)),
            ))
        );
    }

    #[test]
    fn int_resource_var_le_float() {
        let mut model = Model::default();
        let v = model.add_integer_resource_variable("v", false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntResourceVarPy(v);

        let other = IntOrFloatUnion::Float(FloatUnion::Const(1.0));
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Le),
            ConditionPy(Condition::ComparisonC(
                ComparisonOperator::Le,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(v.id())
                ))),
                Box::new(ContinuousExpression::Constant(1.0)),
            ))
        );
    }

    #[test]
    fn int_resource_var_eq_int() {
        let mut model = Model::default();
        let v = model.add_integer_resource_variable("v", false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntResourceVarPy(v);

        let other = IntOrFloatUnion::Int(IntUnion::Const(1));
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Eq),
            ConditionPy(Condition::ComparisonI(
                ComparisonOperator::Eq,
                Box::new(IntegerExpression::ResourceVariable(v.id())),
                Box::new(IntegerExpression::Constant(1)),
            ))
        );
    }

    #[test]
    fn int_resource_var_eq_float() {
        let mut model = Model::default();
        let v = model.add_integer_resource_variable("v", false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntResourceVarPy(v);

        let other = IntOrFloatUnion::Float(FloatUnion::Const(1.0));
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Eq),
            ConditionPy(Condition::ComparisonC(
                ComparisonOperator::Eq,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(v.id())
                ))),
                Box::new(ContinuousExpression::Constant(1.0)),
            ))
        );
    }

    #[test]
    fn int_resource_var_ne_int() {
        let mut model = Model::default();
        let v = model.add_integer_resource_variable("v", false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntResourceVarPy(v);

        let other = IntOrFloatUnion::Int(IntUnion::Const(1));
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Ne),
            ConditionPy(Condition::ComparisonI(
                ComparisonOperator::Ne,
                Box::new(IntegerExpression::ResourceVariable(v.id())),
                Box::new(IntegerExpression::Constant(1)),
            ))
        );
    }

    #[test]
    fn int_resource_var_ne_float() {
        let mut model = Model::default();
        let v = model.add_integer_resource_variable("v", false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntResourceVarPy(v);

        let other = IntOrFloatUnion::Float(FloatUnion::Const(1.0));
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Ne),
            ConditionPy(Condition::ComparisonC(
                ComparisonOperator::Ne,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(v.id())
                ))),
                Box::new(ContinuousExpression::Constant(1.0)),
            ))
        );
    }

    #[test]
    fn int_resource_var_gt_int() {
        let mut model = Model::default();
        let v = model.add_integer_resource_variable("v", false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntResourceVarPy(v);

        let other = IntOrFloatUnion::Int(IntUnion::Const(1));
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Gt),
            ConditionPy(Condition::ComparisonI(
                ComparisonOperator::Gt,
                Box::new(IntegerExpression::ResourceVariable(v.id())),
                Box::new(IntegerExpression::Constant(1)),
            ))
        );
    }

    #[test]
    fn int_resource_var_gt_float() {
        let mut model = Model::default();
        let v = model.add_integer_resource_variable("v", false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntResourceVarPy(v);

        let other = IntOrFloatUnion::Float(FloatUnion::Const(1.0));
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Gt),
            ConditionPy(Condition::ComparisonC(
                ComparisonOperator::Gt,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(v.id())
                ))),
                Box::new(ContinuousExpression::Constant(1.0)),
            ))
        );
    }

    #[test]
    fn int_resource_var_ge_int() {
        let mut model = Model::default();
        let v = model.add_integer_resource_variable("v", false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntResourceVarPy(v);

        let other = IntOrFloatUnion::Int(IntUnion::Const(1));
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Ge),
            ConditionPy(Condition::ComparisonI(
                ComparisonOperator::Ge,
                Box::new(IntegerExpression::ResourceVariable(v.id())),
                Box::new(IntegerExpression::Constant(1)),
            ))
        );
    }

    #[test]
    fn int_resource_var_ge_float() {
        let mut model = Model::default();
        let v = model.add_integer_resource_variable("v", false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntResourceVarPy(v);

        let other = IntOrFloatUnion::Float(FloatUnion::Const(1.0));
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Ge),
            ConditionPy(Condition::ComparisonC(
                ComparisonOperator::Ge,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(v.id())
                ))),
                Box::new(ContinuousExpression::Constant(1.0)),
            ))
        );
    }

    #[test]
    fn int_resource_var_abs() {
        let mut model = Model::default();
        let v = model.add_integer_resource_variable("v", false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntResourceVarPy(v);
        assert_eq!(
            expression.__abs__(),
            IntExprPy(IntegerExpression::UnaryOperation(
                UnaryOperator::Abs,
                Box::new(IntegerExpression::ResourceVariable(v.id()))
            ))
        );
    }

    #[test]
    fn int_resource_var_neg() {
        let mut model = Model::default();
        let v = model.add_integer_resource_variable("v", false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntResourceVarPy(v);
        assert_eq!(
            expression.__neg__(),
            IntExprPy(IntegerExpression::UnaryOperation(
                UnaryOperator::Neg,
                Box::new(IntegerExpression::ResourceVariable(v.id()))
            ))
        );
    }

    #[test]
    fn int_resource_var_add_int() {
        let mut model = Model::default();
        let v = model.add_integer_resource_variable("v", false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntResourceVarPy(v);

        let other = IntOrFloatUnion::Int(IntUnion::Const(1));
        assert_eq!(
            expression.__add__(other),
            IntOrFloatExpr::Int(IntExprPy(IntegerExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(IntegerExpression::ResourceVariable(v.id())),
                Box::new(IntegerExpression::Constant(1)),
            )))
        );
    }

    #[test]
    fn int_resource_var_add_float() {
        let mut model = Model::default();
        let v = model.add_integer_resource_variable("v", false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntResourceVarPy(v);

        let other = IntOrFloatUnion::Float(FloatUnion::Const(1.0));
        assert_eq!(
            expression.__add__(other),
            IntOrFloatExpr::Float(FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(v.id())
                ))),
                Box::new(ContinuousExpression::Constant(1.0)),
            )))
        );
    }

    #[test]
    fn int_resource_var_sub_int() {
        let mut model = Model::default();
        let v = model.add_integer_resource_variable("v", false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntResourceVarPy(v);

        let other = IntOrFloatUnion::Int(IntUnion::Const(1));
        assert_eq!(
            expression.__sub__(other),
            IntOrFloatExpr::Int(IntExprPy(IntegerExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(IntegerExpression::ResourceVariable(v.id())),
                Box::new(IntegerExpression::Constant(1)),
            )))
        );
    }

    #[test]
    fn int_resource_var_sub_float() {
        let mut model = Model::default();
        let v = model.add_integer_resource_variable("v", false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntResourceVarPy(v);

        let other = IntOrFloatUnion::Float(FloatUnion::Const(1.0));
        assert_eq!(
            expression.__sub__(other),
            IntOrFloatExpr::Float(FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(v.id())
                ))),
                Box::new(ContinuousExpression::Constant(1.0)),
            )))
        );
    }

    #[test]
    fn int_resource_var_mul_int() {
        let mut model = Model::default();
        let v = model.add_integer_resource_variable("v", false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntResourceVarPy(v);

        let other = IntOrFloatUnion::Int(IntUnion::Const(1));
        assert_eq!(
            expression.__mul__(other),
            IntOrFloatExpr::Int(IntExprPy(IntegerExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(IntegerExpression::ResourceVariable(v.id())),
                Box::new(IntegerExpression::Constant(1)),
            )))
        );
    }

    #[test]
    fn int_resource_var_mul_float() {
        let mut model = Model::default();
        let v = model.add_integer_resource_variable("v", false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntResourceVarPy(v);

        let other = IntOrFloatUnion::Float(FloatUnion::Const(1.0));
        assert_eq!(
            expression.__mul__(other),
            IntOrFloatExpr::Float(FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(v.id())
                ))),
                Box::new(ContinuousExpression::Constant(1.0)),
            )))
        );
    }

    #[test]
    fn int_resource_var_truediv() {
        let mut model = Model::default();
        let v = model.add_integer_resource_variable("v", false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntResourceVarPy(v);

        let other = FloatUnion::IntConst(1);
        assert_eq!(
            expression.__truediv__(other),
            FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(v.id())
                ))),
                Box::new(ContinuousExpression::Constant(1.0)),
            ))
        );
    }

    #[test]
    fn int_resource_var_floordiv_int() {
        let mut model = Model::default();
        let v = model.add_integer_resource_variable("v", false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntResourceVarPy(v);

        let other = IntOrFloatUnion::Int(IntUnion::Const(1));
        assert_eq!(
            expression.__floordiv__(other),
            IntOrFloatExpr::Int(IntExprPy(IntegerExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(IntegerExpression::ResourceVariable(v.id())),
                Box::new(IntegerExpression::Constant(1)),
            )))
        );
    }

    #[test]
    fn int_resource_var_floordiv_float() {
        let mut model = Model::default();
        let v = model.add_integer_resource_variable("v", false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntResourceVarPy(v);

        let other = IntOrFloatUnion::Float(FloatUnion::Const(1.0));
        assert_eq!(
            expression.__floordiv__(other),
            IntOrFloatExpr::Float(FloatExprPy(ContinuousExpression::Round(
                CastOperator::Floor,
                Box::new(ContinuousExpression::BinaryOperation(
                    BinaryOperator::Div,
                    Box::new(ContinuousExpression::FromInteger(Box::new(
                        IntegerExpression::ResourceVariable(v.id())
                    ))),
                    Box::new(ContinuousExpression::Constant(1.0)),
                ))
            )))
        );
    }

    #[test]
    fn int_resource_var_mod_int() {
        let mut model = Model::default();
        let v = model.add_integer_resource_variable("v", false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntResourceVarPy(v);

        let other = IntOrFloatUnion::Int(IntUnion::Const(1));
        assert_eq!(
            expression.__mod__(other),
            IntOrFloatExpr::Int(IntExprPy(IntegerExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(IntegerExpression::ResourceVariable(v.id())),
                Box::new(IntegerExpression::Constant(1)),
            )))
        );
    }

    #[test]
    fn int_resource_var_mod_float() {
        let mut model = Model::default();
        let v = model.add_integer_resource_variable("v", false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntResourceVarPy(v);

        let other = IntOrFloatUnion::Float(FloatUnion::Const(1.0));
        assert_eq!(
            expression.__mod__(other),
            IntOrFloatExpr::Float(FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(v.id())
                ))),
                Box::new(ContinuousExpression::Constant(1.0)),
            )))
        );
    }

    #[test]
    fn int_resource_var_pow() {
        let mut model = Model::default();
        let v = model.add_integer_resource_variable("v", false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntResourceVarPy(v);

        let other = FloatUnion::IntConst(1);
        assert_eq!(
            expression.__pow__(other, None),
            FloatExprPy(ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Pow,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(v.id())
                ))),
                Box::new(ContinuousExpression::Constant(1.0))
            ))
        );
    }

    #[test]
    fn int_resource_var_pow_with_modulo() {
        let mut model = Model::default();
        let v = model.add_integer_resource_variable("v", false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntResourceVarPy(v);

        let other = FloatUnion::IntConst(1);
        let modulo = FloatUnion::IntConst(2);
        assert_eq!(
            expression.__pow__(other, Some(modulo)),
            FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::ContinuousBinaryOperation(
                    ContinuousBinaryOperator::Pow,
                    Box::new(ContinuousExpression::FromInteger(Box::new(
                        IntegerExpression::ResourceVariable(v.id())
                    ))),
                    Box::new(ContinuousExpression::Constant(1.0))
                )),
                Box::new(ContinuousExpression::Constant(2.0))
            ))
        );
    }

    #[test]
    fn int_resource_var_radd_int() {
        let mut model = Model::default();
        let v = model.add_integer_resource_variable("v", false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntResourceVarPy(v);

        let other = IntOrFloatUnion::Int(IntUnion::Const(1));
        assert_eq!(
            expression.__radd__(other),
            IntOrFloatExpr::Int(IntExprPy(IntegerExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(IntegerExpression::ResourceVariable(v.id())),
                Box::new(IntegerExpression::Constant(1)),
            )))
        );
    }

    #[test]
    fn int_resource_var_radd_float() {
        let mut model = Model::default();
        let v = model.add_integer_resource_variable("v", false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntResourceVarPy(v);

        let other = IntOrFloatUnion::Float(FloatUnion::Const(1.0));
        assert_eq!(
            expression.__radd__(other),
            IntOrFloatExpr::Float(FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(v.id())
                ))),
                Box::new(ContinuousExpression::Constant(1.0)),
            )))
        );
    }

    #[test]
    fn int_resource_var_rsub_int() {
        let mut model = Model::default();
        let v = model.add_integer_resource_variable("v", false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntResourceVarPy(v);

        let other = IntOrFloatUnion::Int(IntUnion::Const(1));
        assert_eq!(
            expression.__rsub__(other),
            IntOrFloatExpr::Int(IntExprPy(IntegerExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::ResourceVariable(v.id())),
            )))
        );
    }

    #[test]
    fn int_resource_var_rsub_float() {
        let mut model = Model::default();
        let v = model.add_integer_resource_variable("v", false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntResourceVarPy(v);

        let other = IntOrFloatUnion::Float(FloatUnion::Const(1.0));
        assert_eq!(
            expression.__rsub__(other),
            IntOrFloatExpr::Float(FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(v.id())
                ))),
            )))
        );
    }

    #[test]
    fn int_resource_var_rmul_int() {
        let mut model = Model::default();
        let v = model.add_integer_resource_variable("v", false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntResourceVarPy(v);

        let other = IntOrFloatUnion::Int(IntUnion::Const(1));
        assert_eq!(
            expression.__rmul__(other),
            IntOrFloatExpr::Int(IntExprPy(IntegerExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(IntegerExpression::ResourceVariable(v.id())),
                Box::new(IntegerExpression::Constant(1)),
            )))
        );
    }

    #[test]
    fn int_resource_var_rmul_float() {
        let mut model = Model::default();
        let v = model.add_integer_resource_variable("v", false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntResourceVarPy(v);

        let other = IntOrFloatUnion::Float(FloatUnion::Const(1.0));
        assert_eq!(
            expression.__rmul__(other),
            IntOrFloatExpr::Float(FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(v.id())
                ))),
                Box::new(ContinuousExpression::Constant(1.0)),
            )))
        );
    }

    #[test]
    fn int_resource_var_rtruediv() {
        let mut model = Model::default();
        let v = model.add_integer_resource_variable("v", false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntResourceVarPy(v);

        let other = FloatUnion::IntConst(1);
        assert_eq!(
            expression.__rtruediv__(other),
            FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(v.id())
                ))),
            ))
        );
    }

    #[test]
    fn int_resource_var_rfloordiv_int() {
        let mut model = Model::default();
        let v = model.add_integer_resource_variable("v", false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntResourceVarPy(v);

        let other = IntOrFloatUnion::Int(IntUnion::Const(1));
        assert_eq!(
            expression.__rfloordiv__(other),
            IntOrFloatExpr::Int(IntExprPy(IntegerExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::ResourceVariable(v.id())),
            )))
        );
    }

    #[test]
    fn int_resource_var_rfloordiv_float() {
        let mut model = Model::default();
        let v = model.add_integer_resource_variable("v", false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntResourceVarPy(v);

        let other = IntOrFloatUnion::Float(FloatUnion::Const(1.0));
        assert_eq!(
            expression.__rfloordiv__(other),
            IntOrFloatExpr::Float(FloatExprPy(ContinuousExpression::Round(
                CastOperator::Floor,
                Box::new(ContinuousExpression::BinaryOperation(
                    BinaryOperator::Div,
                    Box::new(ContinuousExpression::Constant(1.0)),
                    Box::new(ContinuousExpression::FromInteger(Box::new(
                        IntegerExpression::ResourceVariable(v.id())
                    ))),
                ))
            )))
        );
    }

    #[test]
    fn int_resource_var_rmod_int() {
        let mut model = Model::default();
        let v = model.add_integer_resource_variable("v", false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntResourceVarPy(v);

        let other = IntOrFloatUnion::Int(IntUnion::Const(1));
        assert_eq!(
            expression.__rmod__(other),
            IntOrFloatExpr::Int(IntExprPy(IntegerExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::ResourceVariable(v.id())),
            )))
        );
    }

    #[test]
    fn int_resource_var_rmod_float() {
        let mut model = Model::default();
        let v = model.add_integer_resource_variable("v", false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntResourceVarPy(v);

        let other = IntOrFloatUnion::Float(FloatUnion::Const(1.0));
        assert_eq!(
            expression.__rmod__(other),
            IntOrFloatExpr::Float(FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(v.id())
                ))),
            )))
        );
    }

    #[test]
    fn int_resource_var_rpow() {
        let mut model = Model::default();
        let v = model.add_integer_resource_variable("v", false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntResourceVarPy(v);

        let other = FloatUnion::IntConst(1);
        assert_eq!(
            expression.__rpow__(other, None),
            FloatExprPy(ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Pow,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(v.id())
                ))),
            ))
        );
    }

    #[test]
    fn int_resource_var_rpow_with_modulo() {
        let mut model = Model::default();
        let v = model.add_integer_resource_variable("v", false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = IntResourceVarPy(v);

        let other = FloatUnion::IntConst(1);
        let modulo = FloatUnion::IntConst(2);
        assert_eq!(
            expression.__rpow__(other, Some(modulo)),
            FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::ContinuousBinaryOperation(
                    ContinuousBinaryOperator::Pow,
                    Box::new(ContinuousExpression::Constant(1.0)),
                    Box::new(ContinuousExpression::FromInteger(Box::new(
                        IntegerExpression::ResourceVariable(v.id())
                    ))),
                )),
                Box::new(ContinuousExpression::Constant(2.0))
            ))
        );
    }

    #[test]
    fn continuous_expression_from_expr() {
        let float = FloatUnion::Expr(FloatExprPy(ContinuousExpression::Constant(0.0)));
        assert_eq!(
            ContinuousExpression::from(float),
            ContinuousExpression::Constant(0.0)
        );
    }

    #[test]
    fn continuous_expression_from_int_expr() {
        let float = FloatUnion::IntExpr(IntExprPy(IntegerExpression::Constant(0)));
        assert_eq!(
            ContinuousExpression::from(float),
            ContinuousExpression::FromInteger(Box::new(IntegerExpression::Constant(0)))
        );
    }

    #[test]
    fn continuous_expression_from_var() {
        let mut model = Model::default();
        let v = model.add_continuous_variable("v", 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let float = FloatUnion::Var(FloatVarPy(v));
        assert_eq!(
            ContinuousExpression::from(float),
            ContinuousExpression::Variable(v.id())
        );
    }

    #[test]
    fn continuous_expression_from_resource_var() {
        let mut model = Model::default();
        let v = model.add_continuous_resource_variable("v", false, 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let float = FloatUnion::ResourceVar(FloatResourceVarPy(v));
        assert_eq!(
            ContinuousExpression::from(float),
            ContinuousExpression::ResourceVariable(v.id())
        );
    }

    #[test]
    fn continuous_expression_from_int_var() {
        let mut model = Model::default();
        let v = model.add_integer_variable("v", 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let float = FloatUnion::IntVar(IntVarPy(v));
        assert_eq!(
            ContinuousExpression::from(float),
            ContinuousExpression::FromInteger(Box::new(IntegerExpression::Variable(v.id())))
        );
    }

    #[test]
    fn continuous_expression_from_int_resource_var() {
        let mut model = Model::default();
        let v = model.add_integer_resource_variable("v", false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let float = FloatUnion::IntResourceVar(IntResourceVarPy(v));
        assert_eq!(
            ContinuousExpression::from(float),
            ContinuousExpression::FromInteger(Box::new(IntegerExpression::ResourceVariable(
                v.id()
            )))
        );
    }

    #[test]
    fn continuous_expression_from_const() {
        let float = FloatUnion::Const(0.0);
        assert_eq!(
            ContinuousExpression::from(float),
            ContinuousExpression::Constant(0.0)
        );
    }

    #[test]
    fn continuous_expression_from_int_const() {
        let float = FloatUnion::IntConst(0);
        assert_eq!(
            ContinuousExpression::from(float),
            ContinuousExpression::Constant(0.0)
        );
    }

    #[test]
    fn float_expr_to_expression() {
        let expression = FloatExprPy(ContinuousExpression::Constant(0.0));
        assert_eq!(
            ContinuousExpression::from(expression),
            ContinuousExpression::Constant(0.0)
        );
    }

    #[test]
    fn float_expr_new() {
        assert_eq!(
            FloatExprPy::from(ContinuousExpression::Constant(0.0)),
            FloatExprPy(ContinuousExpression::Constant(0.0))
        );
    }

    #[test]
    fn float_expr_new_py() {
        assert_eq!(
            FloatExprPy::new_py(0.0),
            FloatExprPy(ContinuousExpression::Constant(0.0))
        );
    }

    #[test]
    fn float_expr_cost() {
        assert_eq!(
            FloatExprPy::state_cost(),
            FloatExprPy(ContinuousExpression::Cost)
        );
    }

    #[test]
    fn float_expr_lt() {
        let expression = FloatExprPy(ContinuousExpression::Constant(0.0));
        let other = FloatUnion::Const(1.0);
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Lt),
            ConditionPy(Condition::ComparisonC(
                ComparisonOperator::Lt,
                Box::new(ContinuousExpression::Constant(0.0)),
                Box::new(ContinuousExpression::Constant(1.0))
            ))
        );
    }

    #[test]
    fn float_expr_le() {
        let expression = FloatExprPy(ContinuousExpression::Constant(0.0));
        let other = FloatUnion::Const(1.0);
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Le),
            ConditionPy(Condition::ComparisonC(
                ComparisonOperator::Le,
                Box::new(ContinuousExpression::Constant(0.0)),
                Box::new(ContinuousExpression::Constant(1.0))
            ))
        );
    }

    #[test]
    fn float_expr_eq() {
        let expression = FloatExprPy(ContinuousExpression::Constant(0.0));
        let other = FloatUnion::Const(1.0);
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Eq),
            ConditionPy(Condition::ComparisonC(
                ComparisonOperator::Eq,
                Box::new(ContinuousExpression::Constant(0.0)),
                Box::new(ContinuousExpression::Constant(1.0))
            ))
        );
    }

    #[test]
    fn float_expr_ne() {
        let expression = FloatExprPy(ContinuousExpression::Constant(0.0));
        let other = FloatUnion::Const(1.0);
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Ne),
            ConditionPy(Condition::ComparisonC(
                ComparisonOperator::Ne,
                Box::new(ContinuousExpression::Constant(0.0)),
                Box::new(ContinuousExpression::Constant(1.0))
            ))
        );
    }

    #[test]
    fn float_expr_gt() {
        let expression = FloatExprPy(ContinuousExpression::Constant(0.0));
        let other = FloatUnion::Const(1.0);
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Gt),
            ConditionPy(Condition::ComparisonC(
                ComparisonOperator::Gt,
                Box::new(ContinuousExpression::Constant(0.0)),
                Box::new(ContinuousExpression::Constant(1.0))
            ))
        );
    }

    #[test]
    fn float_expr_ge() {
        let expression = FloatExprPy(ContinuousExpression::Constant(0.0));
        let other = FloatUnion::Const(1.0);
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Ge),
            ConditionPy(Condition::ComparisonC(
                ComparisonOperator::Ge,
                Box::new(ContinuousExpression::Constant(0.0)),
                Box::new(ContinuousExpression::Constant(1.0))
            ))
        );
    }

    #[test]
    fn float_expr_abs() {
        let expression = FloatExprPy(ContinuousExpression::Constant(-1.0));
        assert_eq!(
            expression.__abs__(),
            FloatExprPy(ContinuousExpression::UnaryOperation(
                UnaryOperator::Abs,
                Box::new(ContinuousExpression::Constant(-1.0))
            ))
        );
    }

    #[test]
    fn float_expr_neg() {
        let expression = FloatExprPy(ContinuousExpression::Constant(-1.0));
        assert_eq!(
            expression.__neg__(),
            FloatExprPy(ContinuousExpression::UnaryOperation(
                UnaryOperator::Neg,
                Box::new(ContinuousExpression::Constant(-1.0))
            ))
        );
    }

    #[test]
    fn float_expr_add() {
        let expression = FloatExprPy(ContinuousExpression::Constant(0.0));
        let other = FloatUnion::Const(1.0);
        assert_eq!(
            expression.__add__(other),
            FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::Constant(0.0)),
                Box::new(ContinuousExpression::Constant(1.0)),
            ))
        );
    }

    #[test]
    fn float_expr_sub() {
        let expression = FloatExprPy(ContinuousExpression::Constant(0.0));
        let other = FloatUnion::Const(1.0);
        assert_eq!(
            expression.__sub__(other),
            FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ContinuousExpression::Constant(0.0)),
                Box::new(ContinuousExpression::Constant(1.0)),
            ))
        );
    }

    #[test]
    fn float_expr_mul() {
        let expression = FloatExprPy(ContinuousExpression::Constant(0.0));
        let other = FloatUnion::Const(1.0);
        assert_eq!(
            expression.__mul__(other),
            FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ContinuousExpression::Constant(0.0)),
                Box::new(ContinuousExpression::Constant(1.0)),
            ))
        );
    }

    #[test]
    fn float_expr_truediv() {
        let expression = FloatExprPy(ContinuousExpression::Constant(0.0));
        let other = FloatUnion::Const(1.0);
        assert_eq!(
            expression.__truediv__(other),
            FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ContinuousExpression::Constant(0.0)),
                Box::new(ContinuousExpression::Constant(1.0)),
            ))
        );
    }

    #[test]
    fn float_expr_floordiv_ok() {
        let expression = FloatExprPy(ContinuousExpression::Constant(0.0));
        let other = FloatUnion::Const(1.0);
        assert_eq!(
            expression.__floordiv__(other),
            FloatExprPy(ContinuousExpression::Round(
                CastOperator::Floor,
                Box::new(ContinuousExpression::BinaryOperation(
                    BinaryOperator::Div,
                    Box::new(ContinuousExpression::Constant(0.0)),
                    Box::new(ContinuousExpression::Constant(1.0))
                ))
            ))
        );
    }

    #[test]
    fn float_expr_mod() {
        let expression = FloatExprPy(ContinuousExpression::Constant(0.0));
        let other = FloatUnion::Const(1.0);
        assert_eq!(
            expression.__mod__(other),
            FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::Constant(0.0)),
                Box::new(ContinuousExpression::Constant(1.0)),
            ))
        );
    }

    #[test]
    fn float_expr_pow() {
        let expression = FloatExprPy(ContinuousExpression::Constant(0.0));
        let other = FloatUnion::Const(1.0);
        assert_eq!(
            expression.__pow__(other, None),
            FloatExprPy(ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Pow,
                Box::new(ContinuousExpression::Constant(0.0)),
                Box::new(ContinuousExpression::Constant(1.0))
            ))
        );
    }

    #[test]
    fn float_expr_pow_with_modulo() {
        let expression = FloatExprPy(ContinuousExpression::Constant(0.0));
        let other = FloatUnion::Const(1.0);
        let modulo = FloatUnion::Const(2.0);
        assert_eq!(
            expression.__pow__(other, Some(modulo)),
            FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::ContinuousBinaryOperation(
                    ContinuousBinaryOperator::Pow,
                    Box::new(ContinuousExpression::Constant(0.0)),
                    Box::new(ContinuousExpression::Constant(1.0))
                )),
                Box::new(ContinuousExpression::Constant(2.0))
            ))
        );
    }

    #[test]
    fn float_expr_radd() {
        let expression = FloatExprPy(ContinuousExpression::Constant(0.0));
        let other = FloatUnion::Const(1.0);
        assert_eq!(
            expression.__radd__(other),
            FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::Constant(0.0)),
                Box::new(ContinuousExpression::Constant(1.0)),
            ))
        );
    }

    #[test]
    fn float_expr_rsub() {
        let expression = FloatExprPy(ContinuousExpression::Constant(0.0));
        let other = FloatUnion::Const(1.0);
        assert_eq!(
            expression.__rsub__(other),
            FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Constant(0.0)),
            ))
        );
    }

    #[test]
    fn float_expr_rmul() {
        let expression = FloatExprPy(ContinuousExpression::Constant(0.0));
        let other = FloatUnion::Const(1.0);
        assert_eq!(
            expression.__rmul__(other),
            FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ContinuousExpression::Constant(0.0)),
                Box::new(ContinuousExpression::Constant(1.0)),
            ))
        );
    }

    #[test]
    fn float_expr_rtruediv() {
        let expression = FloatExprPy(ContinuousExpression::Constant(0.0));
        let other = FloatUnion::Const(1.0);
        assert_eq!(
            expression.__rtruediv__(other),
            FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Constant(0.0)),
            ))
        );
    }

    #[test]
    fn float_expr_rfloordiv_ok() {
        let expression = FloatExprPy(ContinuousExpression::Constant(0.0));
        let other = FloatUnion::Const(1.0);
        assert_eq!(
            expression.__rfloordiv__(other),
            FloatExprPy(ContinuousExpression::Round(
                CastOperator::Floor,
                Box::new(ContinuousExpression::BinaryOperation(
                    BinaryOperator::Div,
                    Box::new(ContinuousExpression::Constant(1.0)),
                    Box::new(ContinuousExpression::Constant(0.0)),
                ))
            ))
        );
    }

    #[test]
    fn float_expr_rmod() {
        let expression = FloatExprPy(ContinuousExpression::Constant(0.0));
        let other = FloatUnion::Const(1.0);
        assert_eq!(
            expression.__rmod__(other),
            FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Constant(0.0)),
            ))
        );
    }

    #[test]
    fn float_expr_rpow() {
        let expression = FloatExprPy(ContinuousExpression::Constant(0.0));
        let other = FloatUnion::Const(1.0);
        assert_eq!(
            expression.__rpow__(other, None),
            FloatExprPy(ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Pow,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Constant(0.0)),
            ))
        );
    }

    #[test]
    fn float_expr_rpow_with_modulo() {
        let expression = FloatExprPy(ContinuousExpression::Constant(0.0));
        let other = FloatUnion::Const(1.0);
        let modulo = FloatUnion::Const(2.0);
        assert_eq!(
            expression.__rpow__(other, Some(modulo)),
            FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::ContinuousBinaryOperation(
                    ContinuousBinaryOperator::Pow,
                    Box::new(ContinuousExpression::Constant(1.0)),
                    Box::new(ContinuousExpression::Constant(0.0)),
                )),
                Box::new(ContinuousExpression::Constant(2.0))
            ))
        );
    }

    #[test]
    fn float_exp_round() {
        let expression = FloatExprPy(ContinuousExpression::Constant(0.0));
        assert_eq!(
            expression.__round__(),
            IntExprPy(IntegerExpression::FromContinuous(
                CastOperator::Round,
                Box::new(ContinuousExpression::Constant(0.0))
            ))
        );
    }

    #[test]
    fn float_exp_trunc() {
        let expression = FloatExprPy(ContinuousExpression::Constant(0.0));
        assert_eq!(
            expression.__trunc__(),
            IntExprPy(IntegerExpression::FromContinuous(
                CastOperator::Trunc,
                Box::new(ContinuousExpression::Constant(0.0))
            ))
        );
    }

    #[test]
    fn float_exp_floor() {
        let expression = FloatExprPy(ContinuousExpression::Constant(0.0));
        assert_eq!(
            expression.__floor__(),
            IntExprPy(IntegerExpression::FromContinuous(
                CastOperator::Floor,
                Box::new(ContinuousExpression::Constant(0.0))
            ))
        );
    }

    #[test]
    fn float_exp_ceil() {
        let expression = FloatExprPy(ContinuousExpression::Constant(0.0));
        assert_eq!(
            expression.__ceil__(),
            IntExprPy(IntegerExpression::FromContinuous(
                CastOperator::Ceil,
                Box::new(ContinuousExpression::Constant(0.0))
            ))
        );
    }

    #[test]
    fn float_var_to_variable() {
        let mut model = Model::default();
        let v = model.add_continuous_variable("v", 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let v_py = FloatVarPy(v);
        assert_eq!(ContinuousVariable::from(v_py), v);
    }

    #[test]
    fn float_var_to_expression() {
        let mut model = Model::default();
        let v = model.add_continuous_variable("v", 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let v_py = FloatVarPy(v);
        assert_eq!(
            ContinuousExpression::from(v_py),
            ContinuousExpression::Variable(v.id())
        );
    }

    #[test]
    fn float_var_new() {
        let mut model = Model::default();
        let v = model.add_continuous_variable("v", 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();

        assert_eq!(FloatVarPy::from(v), FloatVarPy(v));
    }

    #[test]
    fn float_var_lt() {
        let mut model = Model::default();
        let v = model.add_continuous_variable("v", 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = FloatVarPy(v);

        let other = FloatUnion::Const(1.0);
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Lt),
            ConditionPy(Condition::ComparisonC(
                ComparisonOperator::Lt,
                Box::new(ContinuousExpression::Variable(v.id())),
                Box::new(ContinuousExpression::Constant(1.0))
            ))
        );
    }

    #[test]
    fn float_var_le() {
        let mut model = Model::default();
        let v = model.add_continuous_variable("v", 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = FloatVarPy(v);

        let other = FloatUnion::Const(1.0);
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Le),
            ConditionPy(Condition::ComparisonC(
                ComparisonOperator::Le,
                Box::new(ContinuousExpression::Variable(v.id())),
                Box::new(ContinuousExpression::Constant(1.0))
            ))
        );
    }

    #[test]
    fn float_var_eq() {
        let mut model = Model::default();
        let v = model.add_continuous_variable("v", 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = FloatVarPy(v);

        let other = FloatUnion::Const(1.0);
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Eq),
            ConditionPy(Condition::ComparisonC(
                ComparisonOperator::Eq,
                Box::new(ContinuousExpression::Variable(v.id())),
                Box::new(ContinuousExpression::Constant(1.0))
            ))
        );
    }

    #[test]
    fn float_var_ne() {
        let mut model = Model::default();
        let v = model.add_continuous_variable("v", 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = FloatVarPy(v);

        let other = FloatUnion::Const(1.0);
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Ne),
            ConditionPy(Condition::ComparisonC(
                ComparisonOperator::Ne,
                Box::new(ContinuousExpression::Variable(v.id())),
                Box::new(ContinuousExpression::Constant(1.0))
            ))
        );
    }

    #[test]
    fn float_var_gt() {
        let mut model = Model::default();
        let v = model.add_continuous_variable("v", 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = FloatVarPy(v);

        let other = FloatUnion::Const(1.0);
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Gt),
            ConditionPy(Condition::ComparisonC(
                ComparisonOperator::Gt,
                Box::new(ContinuousExpression::Variable(v.id())),
                Box::new(ContinuousExpression::Constant(1.0))
            ))
        );
    }

    #[test]
    fn float_var_ge() {
        let mut model = Model::default();
        let v = model.add_continuous_variable("v", 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = FloatVarPy(v);

        let other = FloatUnion::Const(1.0);
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Ge),
            ConditionPy(Condition::ComparisonC(
                ComparisonOperator::Ge,
                Box::new(ContinuousExpression::Variable(v.id())),
                Box::new(ContinuousExpression::Constant(1.0))
            ))
        );
    }

    #[test]
    fn float_var_abs() {
        let mut model = Model::default();
        let v = model.add_continuous_variable("v", 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = FloatVarPy(v);
        assert_eq!(
            expression.__abs__(),
            FloatExprPy(ContinuousExpression::UnaryOperation(
                UnaryOperator::Abs,
                Box::new(ContinuousExpression::Variable(v.id()))
            ))
        );
    }

    #[test]
    fn float_var_neg() {
        let mut model = Model::default();
        let v = model.add_continuous_variable("v", 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = FloatVarPy(v);
        assert_eq!(
            expression.__neg__(),
            FloatExprPy(ContinuousExpression::UnaryOperation(
                UnaryOperator::Neg,
                Box::new(ContinuousExpression::Variable(v.id()))
            ))
        );
    }

    #[test]
    fn float_var_add() {
        let mut model = Model::default();
        let v = model.add_continuous_variable("v", 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = FloatVarPy(v);

        let other = FloatUnion::Const(1.0);
        assert_eq!(
            expression.__add__(other),
            FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::Variable(v.id())),
                Box::new(ContinuousExpression::Constant(1.0)),
            ))
        );
    }

    #[test]
    fn float_var_sub() {
        let mut model = Model::default();
        let v = model.add_continuous_variable("v", 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = FloatVarPy(v);

        let other = FloatUnion::Const(1.0);
        assert_eq!(
            expression.__sub__(other),
            FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ContinuousExpression::Variable(v.id())),
                Box::new(ContinuousExpression::Constant(1.0)),
            ))
        );
    }

    #[test]
    fn float_var_mul() {
        let mut model = Model::default();
        let v = model.add_continuous_variable("v", 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = FloatVarPy(v);

        let other = FloatUnion::Const(1.0);
        assert_eq!(
            expression.__mul__(other),
            FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ContinuousExpression::Variable(v.id())),
                Box::new(ContinuousExpression::Constant(1.0)),
            ))
        );
    }

    #[test]
    fn float_var_truediv() {
        let mut model = Model::default();
        let v = model.add_continuous_variable("v", 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = FloatVarPy(v);

        let other = FloatUnion::Const(1.0);
        assert_eq!(
            expression.__truediv__(other),
            FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ContinuousExpression::Variable(v.id())),
                Box::new(ContinuousExpression::Constant(1.0)),
            ))
        );
    }

    #[test]
    fn float_var_floordiv() {
        let mut model = Model::default();
        let v = model.add_continuous_variable("v", 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = FloatVarPy(v);

        let other = FloatUnion::Const(1.0);
        assert_eq!(
            expression.__floordiv__(other),
            FloatExprPy(ContinuousExpression::Round(
                CastOperator::Floor,
                Box::new(ContinuousExpression::BinaryOperation(
                    BinaryOperator::Div,
                    Box::new(ContinuousExpression::Variable(v.id())),
                    Box::new(ContinuousExpression::Constant(1.0))
                ))
            ))
        );
    }

    #[test]
    fn float_var_mod() {
        let mut model = Model::default();
        let v = model.add_continuous_variable("v", 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = FloatVarPy(v);

        let other = FloatUnion::Const(1.0);
        assert_eq!(
            expression.__mod__(other),
            FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::Variable(v.id())),
                Box::new(ContinuousExpression::Constant(1.0)),
            ))
        );
    }

    #[test]
    fn float_var_pow() {
        let mut model = Model::default();
        let v = model.add_continuous_variable("v", 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = FloatVarPy(v);

        let other = FloatUnion::Const(1.0);
        assert_eq!(
            expression.__pow__(other, None),
            FloatExprPy(ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Pow,
                Box::new(ContinuousExpression::Variable(v.id())),
                Box::new(ContinuousExpression::Constant(1.0))
            ))
        );
    }

    #[test]
    fn float_var_pow_with_modulo() {
        let mut model = Model::default();
        let v = model.add_continuous_variable("v", 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = FloatVarPy(v);

        let other = FloatUnion::Const(1.0);
        let modulo = FloatUnion::Const(2.0);
        assert_eq!(
            expression.__pow__(other, Some(modulo)),
            FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::ContinuousBinaryOperation(
                    ContinuousBinaryOperator::Pow,
                    Box::new(ContinuousExpression::Variable(v.id())),
                    Box::new(ContinuousExpression::Constant(1.0))
                )),
                Box::new(ContinuousExpression::Constant(2.0))
            ))
        );
    }

    #[test]
    fn float_var_radd() {
        let mut model = Model::default();
        let v = model.add_continuous_variable("v", 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = FloatVarPy(v);

        let other = FloatUnion::Const(1.0);
        assert_eq!(
            expression.__radd__(other),
            FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::Variable(v.id())),
                Box::new(ContinuousExpression::Constant(1.0)),
            ))
        );
    }

    #[test]
    fn float_var_rsub() {
        let mut model = Model::default();
        let v = model.add_continuous_variable("v", 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = FloatVarPy(v);

        let other = FloatUnion::Const(1.0);
        assert_eq!(
            expression.__rsub__(other),
            FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Variable(v.id())),
            ))
        );
    }

    #[test]
    fn float_var_rmul() {
        let mut model = Model::default();
        let v = model.add_continuous_variable("v", 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = FloatVarPy(v);

        let other = FloatUnion::Const(1.0);
        assert_eq!(
            expression.__rmul__(other),
            FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ContinuousExpression::Variable(v.id())),
                Box::new(ContinuousExpression::Constant(1.0)),
            ))
        );
    }

    #[test]
    fn float_var_rtruediv() {
        let mut model = Model::default();
        let v = model.add_continuous_variable("v", 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = FloatVarPy(v);

        let other = FloatUnion::Const(1.0);
        assert_eq!(
            expression.__rtruediv__(other),
            FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Variable(v.id())),
            ))
        );
    }

    #[test]
    fn float_var_rfloordiv() {
        let mut model = Model::default();
        let v = model.add_continuous_variable("v", 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = FloatVarPy(v);

        let other = FloatUnion::Const(1.0);
        assert_eq!(
            expression.__rfloordiv__(other),
            FloatExprPy(ContinuousExpression::Round(
                CastOperator::Floor,
                Box::new(ContinuousExpression::BinaryOperation(
                    BinaryOperator::Div,
                    Box::new(ContinuousExpression::Constant(1.0)),
                    Box::new(ContinuousExpression::Variable(v.id())),
                ))
            ))
        );
    }

    #[test]
    fn float_var_rmod() {
        let mut model = Model::default();
        let v = model.add_continuous_variable("v", 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = FloatVarPy(v);

        let other = FloatUnion::Const(1.0);
        assert_eq!(
            expression.__rmod__(other),
            FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Variable(v.id())),
            ))
        );
    }

    #[test]
    fn float_var_rpow() {
        let mut model = Model::default();
        let v = model.add_continuous_variable("v", 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = FloatVarPy(v);

        let other = FloatUnion::Const(1.0);
        assert_eq!(
            expression.__rpow__(other, None),
            FloatExprPy(ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Pow,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Variable(v.id())),
            ))
        );
    }

    #[test]
    fn float_var_rpow_with_modulo() {
        let mut model = Model::default();
        let v = model.add_continuous_variable("v", 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = FloatVarPy(v);

        let other = FloatUnion::Const(1.0);
        let modulo = FloatUnion::Const(2.0);
        assert_eq!(
            expression.__rpow__(other, Some(modulo)),
            FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::ContinuousBinaryOperation(
                    ContinuousBinaryOperator::Pow,
                    Box::new(ContinuousExpression::Constant(1.0)),
                    Box::new(ContinuousExpression::Variable(v.id())),
                )),
                Box::new(ContinuousExpression::Constant(2.0))
            ))
        );
    }

    #[test]
    fn float_var_round() {
        let mut model = Model::default();
        let v = model.add_continuous_variable("v", 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = FloatVarPy(v);
        assert_eq!(
            expression.__round__(),
            IntExprPy(IntegerExpression::FromContinuous(
                CastOperator::Round,
                Box::new(ContinuousExpression::Variable(0))
            ))
        );
    }

    #[test]
    fn float_var_trunc() {
        let mut model = Model::default();
        let v = model.add_continuous_variable("v", 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = FloatVarPy(v);
        assert_eq!(
            expression.__trunc__(),
            IntExprPy(IntegerExpression::FromContinuous(
                CastOperator::Trunc,
                Box::new(ContinuousExpression::Variable(0))
            ))
        );
    }

    #[test]
    fn float_var_floor() {
        let mut model = Model::default();
        let v = model.add_continuous_variable("v", 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = FloatVarPy(v);
        assert_eq!(
            expression.__floor__(),
            IntExprPy(IntegerExpression::FromContinuous(
                CastOperator::Floor,
                Box::new(ContinuousExpression::Variable(0))
            ))
        );
    }

    #[test]
    fn float_var_ceil() {
        let mut model = Model::default();
        let v = model.add_continuous_variable("v", 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = FloatVarPy(v);
        assert_eq!(
            expression.__ceil__(),
            IntExprPy(IntegerExpression::FromContinuous(
                CastOperator::Ceil,
                Box::new(ContinuousExpression::Variable(0))
            ))
        );
    }

    #[test]
    fn float_resource_var_to_resource_variable() {
        let mut model = Model::default();
        let v = model.add_continuous_resource_variable("v", false, 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let v_py = FloatResourceVarPy(v);
        assert_eq!(ContinuousResourceVariable::from(v_py), v);
    }

    #[test]
    fn float_resource_var_to_expression() {
        let mut model = Model::default();
        let v = model.add_continuous_resource_variable("v", false, 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let v_py = FloatResourceVarPy(v);
        assert_eq!(
            ContinuousExpression::from(v_py),
            ContinuousExpression::ResourceVariable(v.id())
        );
    }

    #[test]
    fn float_resource_var_new() {
        let mut model = Model::default();
        let v = model.add_continuous_resource_variable("v", false, 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();

        assert_eq!(FloatResourceVarPy::from(v), FloatResourceVarPy(v));
    }

    #[test]
    fn float_resource_var_lt() {
        let mut model = Model::default();
        let v = model.add_continuous_resource_variable("v", false, 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = FloatResourceVarPy(v);

        let other = FloatUnion::Const(1.0);
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Lt),
            ConditionPy(Condition::ComparisonC(
                ComparisonOperator::Lt,
                Box::new(ContinuousExpression::ResourceVariable(v.id())),
                Box::new(ContinuousExpression::Constant(1.0))
            ))
        );
    }

    #[test]
    fn float_resource_var_le() {
        let mut model = Model::default();
        let v = model.add_continuous_resource_variable("v", false, 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = FloatResourceVarPy(v);

        let other = FloatUnion::Const(1.0);
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Le),
            ConditionPy(Condition::ComparisonC(
                ComparisonOperator::Le,
                Box::new(ContinuousExpression::ResourceVariable(v.id())),
                Box::new(ContinuousExpression::Constant(1.0))
            ))
        );
    }

    #[test]
    fn float_resource_var_eq() {
        let mut model = Model::default();
        let v = model.add_continuous_resource_variable("v", false, 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = FloatResourceVarPy(v);

        let other = FloatUnion::Const(1.0);
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Eq),
            ConditionPy(Condition::ComparisonC(
                ComparisonOperator::Eq,
                Box::new(ContinuousExpression::ResourceVariable(v.id())),
                Box::new(ContinuousExpression::Constant(1.0))
            ))
        );
    }

    #[test]
    fn float_resource_var_ne() {
        let mut model = Model::default();
        let v = model.add_continuous_resource_variable("v", false, 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = FloatResourceVarPy(v);

        let other = FloatUnion::Const(1.0);
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Ne),
            ConditionPy(Condition::ComparisonC(
                ComparisonOperator::Ne,
                Box::new(ContinuousExpression::ResourceVariable(v.id())),
                Box::new(ContinuousExpression::Constant(1.0))
            ))
        );
    }

    #[test]
    fn float_resource_var_gt() {
        let mut model = Model::default();
        let v = model.add_continuous_resource_variable("v", false, 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = FloatResourceVarPy(v);

        let other = FloatUnion::Const(1.0);
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Gt),
            ConditionPy(Condition::ComparisonC(
                ComparisonOperator::Gt,
                Box::new(ContinuousExpression::ResourceVariable(v.id())),
                Box::new(ContinuousExpression::Constant(1.0))
            ))
        );
    }

    #[test]
    fn float_resource_var_ge() {
        let mut model = Model::default();
        let v = model.add_continuous_resource_variable("v", false, 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = FloatResourceVarPy(v);

        let other = FloatUnion::Const(1.0);
        assert_eq!(
            expression.__richcmp__(other, CompareOp::Ge),
            ConditionPy(Condition::ComparisonC(
                ComparisonOperator::Ge,
                Box::new(ContinuousExpression::ResourceVariable(v.id())),
                Box::new(ContinuousExpression::Constant(1.0))
            ))
        );
    }

    #[test]
    fn float_resource_var_abs() {
        let mut model = Model::default();
        let v = model.add_continuous_resource_variable("v", false, 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = FloatResourceVarPy(v);
        assert_eq!(
            expression.__abs__(),
            FloatExprPy(ContinuousExpression::UnaryOperation(
                UnaryOperator::Abs,
                Box::new(ContinuousExpression::ResourceVariable(v.id()))
            ))
        );
    }

    #[test]
    fn float_resource_var_neg() {
        let mut model = Model::default();
        let v = model.add_continuous_resource_variable("v", false, 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = FloatResourceVarPy(v);
        assert_eq!(
            expression.__neg__(),
            FloatExprPy(ContinuousExpression::UnaryOperation(
                UnaryOperator::Neg,
                Box::new(ContinuousExpression::ResourceVariable(v.id()))
            ))
        );
    }

    #[test]
    fn float_resource_var_add() {
        let mut model = Model::default();
        let v = model.add_continuous_resource_variable("v", false, 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = FloatResourceVarPy(v);

        let other = FloatUnion::Const(1.0);
        assert_eq!(
            expression.__add__(other),
            FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::ResourceVariable(v.id())),
                Box::new(ContinuousExpression::Constant(1.0)),
            ))
        );
    }

    #[test]
    fn float_resource_var_sub() {
        let mut model = Model::default();
        let v = model.add_continuous_resource_variable("v", false, 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = FloatResourceVarPy(v);

        let other = FloatUnion::Const(1.0);
        assert_eq!(
            expression.__sub__(other),
            FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ContinuousExpression::ResourceVariable(v.id())),
                Box::new(ContinuousExpression::Constant(1.0)),
            ))
        );
    }

    #[test]
    fn float_resource_var_mul() {
        let mut model = Model::default();
        let v = model.add_continuous_resource_variable("v", false, 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = FloatResourceVarPy(v);

        let other = FloatUnion::Const(1.0);
        assert_eq!(
            expression.__mul__(other),
            FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ContinuousExpression::ResourceVariable(v.id())),
                Box::new(ContinuousExpression::Constant(1.0)),
            ))
        );
    }

    #[test]
    fn float_resource_var_truediv() {
        let mut model = Model::default();
        let v = model.add_continuous_resource_variable("v", false, 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = FloatResourceVarPy(v);

        let other = FloatUnion::Const(1.0);
        assert_eq!(
            expression.__truediv__(other),
            FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ContinuousExpression::ResourceVariable(v.id())),
                Box::new(ContinuousExpression::Constant(1.0)),
            ))
        );
    }

    #[test]
    fn float_resource_var_floordiv() {
        let mut model = Model::default();
        let v = model.add_continuous_resource_variable("v", false, 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = FloatResourceVarPy(v);

        let other = FloatUnion::Const(1.0);
        assert_eq!(
            expression.__floordiv__(other),
            FloatExprPy(ContinuousExpression::Round(
                CastOperator::Floor,
                Box::new(ContinuousExpression::BinaryOperation(
                    BinaryOperator::Div,
                    Box::new(ContinuousExpression::ResourceVariable(v.id())),
                    Box::new(ContinuousExpression::Constant(1.0))
                ))
            ))
        );
    }

    #[test]
    fn float_resource_var_mod() {
        let mut model = Model::default();
        let v = model.add_continuous_resource_variable("v", false, 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = FloatResourceVarPy(v);

        let other = FloatUnion::Const(1.0);
        assert_eq!(
            expression.__mod__(other),
            FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::ResourceVariable(v.id())),
                Box::new(ContinuousExpression::Constant(1.0)),
            ))
        );
    }

    #[test]
    fn float_resource_var_pow() {
        let mut model = Model::default();
        let v = model.add_continuous_resource_variable("v", false, 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = FloatResourceVarPy(v);

        let other = FloatUnion::Const(1.0);
        assert_eq!(
            expression.__pow__(other, None),
            FloatExprPy(ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Pow,
                Box::new(ContinuousExpression::ResourceVariable(v.id())),
                Box::new(ContinuousExpression::Constant(1.0))
            ))
        );
    }

    #[test]
    fn float_resource_var_pow_with_modulo() {
        let mut model = Model::default();
        let v = model.add_continuous_resource_variable("v", false, 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = FloatResourceVarPy(v);

        let other = FloatUnion::Const(1.0);
        let modulo = FloatUnion::Const(2.0);
        assert_eq!(
            expression.__pow__(other, Some(modulo)),
            FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::ContinuousBinaryOperation(
                    ContinuousBinaryOperator::Pow,
                    Box::new(ContinuousExpression::ResourceVariable(v.id())),
                    Box::new(ContinuousExpression::Constant(1.0))
                )),
                Box::new(ContinuousExpression::Constant(2.0))
            ))
        );
    }

    #[test]
    fn float_resource_var_radd() {
        let mut model = Model::default();
        let v = model.add_continuous_resource_variable("v", false, 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = FloatResourceVarPy(v);

        let other = FloatUnion::Const(1.0);
        assert_eq!(
            expression.__radd__(other),
            FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::ResourceVariable(v.id())),
                Box::new(ContinuousExpression::Constant(1.0)),
            ))
        );
    }

    #[test]
    fn float_resource_var_rsub() {
        let mut model = Model::default();
        let v = model.add_continuous_resource_variable("v", false, 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = FloatResourceVarPy(v);

        let other = FloatUnion::Const(1.0);
        assert_eq!(
            expression.__rsub__(other),
            FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::ResourceVariable(v.id())),
            ))
        );
    }

    #[test]
    fn float_resource_var_rmul() {
        let mut model = Model::default();
        let v = model.add_continuous_resource_variable("v", false, 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = FloatResourceVarPy(v);

        let other = FloatUnion::Const(1.0);
        assert_eq!(
            expression.__rmul__(other),
            FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ContinuousExpression::ResourceVariable(v.id())),
                Box::new(ContinuousExpression::Constant(1.0)),
            ))
        );
    }

    #[test]
    fn float_resource_var_rtruediv() {
        let mut model = Model::default();
        let v = model.add_continuous_resource_variable("v", false, 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = FloatResourceVarPy(v);

        let other = FloatUnion::Const(1.0);
        assert_eq!(
            expression.__rtruediv__(other),
            FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::ResourceVariable(v.id())),
            ))
        );
    }

    #[test]
    fn float_resource_var_rfloordiv() {
        let mut model = Model::default();
        let v = model.add_continuous_resource_variable("v", false, 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = FloatResourceVarPy(v);

        let other = FloatUnion::Const(1.0);
        assert_eq!(
            expression.__rfloordiv__(other),
            FloatExprPy(ContinuousExpression::Round(
                CastOperator::Floor,
                Box::new(ContinuousExpression::BinaryOperation(
                    BinaryOperator::Div,
                    Box::new(ContinuousExpression::Constant(1.0)),
                    Box::new(ContinuousExpression::ResourceVariable(v.id())),
                ))
            ))
        );
    }

    #[test]
    fn float_resource_var_rmod() {
        let mut model = Model::default();
        let v = model.add_continuous_resource_variable("v", false, 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = FloatResourceVarPy(v);

        let other = FloatUnion::Const(1.0);
        assert_eq!(
            expression.__rmod__(other),
            FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::ResourceVariable(v.id())),
            ))
        );
    }

    #[test]
    fn float_resource_var_rpow() {
        let mut model = Model::default();
        let v = model.add_continuous_resource_variable("v", false, 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = FloatResourceVarPy(v);

        let other = FloatUnion::Const(1.0);
        assert_eq!(
            expression.__rpow__(other, None),
            FloatExprPy(ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Pow,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::ResourceVariable(v.id())),
            ))
        );
    }

    #[test]
    fn float_resource_var_rpow_with_modulo() {
        let mut model = Model::default();
        let v = model.add_continuous_resource_variable("v", false, 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = FloatResourceVarPy(v);

        let other = FloatUnion::Const(1.0);
        let modulo = FloatUnion::Const(2.0);
        assert_eq!(
            expression.__rpow__(other, Some(modulo)),
            FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::ContinuousBinaryOperation(
                    ContinuousBinaryOperator::Pow,
                    Box::new(ContinuousExpression::Constant(1.0)),
                    Box::new(ContinuousExpression::ResourceVariable(v.id())),
                )),
                Box::new(ContinuousExpression::Constant(2.0))
            ))
        );
    }

    #[test]
    fn float_resource_var_round() {
        let mut model = Model::default();
        let v = model.add_continuous_resource_variable("v", false, 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = FloatResourceVarPy(v);
        assert_eq!(
            expression.__round__(),
            IntExprPy(IntegerExpression::FromContinuous(
                CastOperator::Round,
                Box::new(ContinuousExpression::ResourceVariable(0))
            ))
        );
    }

    #[test]
    fn float_resource_var_trunc() {
        let mut model = Model::default();
        let v = model.add_continuous_resource_variable("v", false, 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = FloatResourceVarPy(v);
        assert_eq!(
            expression.__trunc__(),
            IntExprPy(IntegerExpression::FromContinuous(
                CastOperator::Trunc,
                Box::new(ContinuousExpression::ResourceVariable(0))
            ))
        );
    }

    #[test]
    fn float_resource_var_floor() {
        let mut model = Model::default();
        let v = model.add_continuous_resource_variable("v", false, 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = FloatResourceVarPy(v);
        assert_eq!(
            expression.__floor__(),
            IntExprPy(IntegerExpression::FromContinuous(
                CastOperator::Floor,
                Box::new(ContinuousExpression::ResourceVariable(0))
            ))
        );
    }

    #[test]
    fn float_resource_var_ceil() {
        let mut model = Model::default();
        let v = model.add_continuous_resource_variable("v", false, 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let expression = FloatResourceVarPy(v);
        assert_eq!(
            expression.__ceil__(),
            IntExprPy(IntegerExpression::FromContinuous(
                CastOperator::Ceil,
                Box::new(ContinuousExpression::ResourceVariable(0))
            ))
        );
    }

    #[test]
    fn dp_sqrt() {
        let x = FloatUnion::Const(4.0);
        assert_eq!(
            sqrt(x),
            FloatExprPy(ContinuousExpression::ContinuousUnaryOperation(
                ContinuousUnaryOperator::Sqrt,
                Box::new(ContinuousExpression::Constant(4.0))
            ))
        );
    }

    #[test]
    fn dp_log() {
        let x = FloatUnion::Const(4.0);
        let y = FloatUnion::Const(2.0);
        assert_eq!(
            log(x, y),
            FloatExprPy(ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Log,
                Box::new(ContinuousExpression::Constant(4.0)),
                Box::new(ContinuousExpression::Constant(2.0)),
            ))
        );
    }

    #[test]
    fn dp_float() {
        let x = IntUnion::Const(4);
        assert_eq!(
            float(x),
            FloatExprPy(ContinuousExpression::FromInteger(Box::new(
                IntegerExpression::Constant(4)
            )))
        );
    }

    #[test]
    fn max_int_ok() {
        pyo3::prepare_freethreaded_python();

        let x = IntExprPy(IntegerExpression::Constant(4));
        let y = IntExprPy(IntegerExpression::Constant(2));
        let result = Python::with_gil(|py| {
            let x = x.into_py(py);
            let y = y.into_py(py);
            max(x.into_bound(py), y.into_bound(py))
        });
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            ExprUnion::Int(IntExprPy(IntegerExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(IntegerExpression::Constant(4)),
                Box::new(IntegerExpression::Constant(2)),
            )))
        );
    }

    #[test]
    fn max_float_ok() {
        pyo3::prepare_freethreaded_python();

        let x = FloatExprPy(ContinuousExpression::Constant(4.0));
        let y = FloatExprPy(ContinuousExpression::Constant(2.0));
        let result = Python::with_gil(|py| {
            let x = x.into_py(py);
            let y = y.into_py(py);
            max(x.into_bound(py), y.into_bound(py))
        });
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            ExprUnion::Float(FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ContinuousExpression::Constant(4.0)),
                Box::new(ContinuousExpression::Constant(2.0)),
            )))
        );
    }

    #[test]
    fn max_element_ok() {
        pyo3::prepare_freethreaded_python();

        let x = ElementExprPy(ElementExpression::Constant(4));
        let y = ElementExprPy(ElementExpression::Constant(2));
        let result = Python::with_gil(|py| {
            let x = x.into_py(py);
            let y = y.into_py(py);
            max(x.into_bound(py), y.into_bound(py))
        });
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            ExprUnion::Element(ElementExprPy(ElementExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ElementExpression::Constant(4)),
                Box::new(ElementExpression::Constant(2)),
            )))
        );
    }

    #[test]
    fn max_err() {
        pyo3::prepare_freethreaded_python();

        let x = ElementExprPy(ElementExpression::Constant(4));
        let y = FloatExprPy(ContinuousExpression::Constant(2.0));
        let result = Python::with_gil(|py| {
            let x = x.into_py(py);
            let y = y.into_py(py);
            min(x.into_bound(py), y.into_bound(py))
        });
        assert!(result.is_err());
    }

    #[test]
    fn min_int_ok() {
        pyo3::prepare_freethreaded_python();

        let x = IntExprPy(IntegerExpression::Constant(4));
        let y = IntExprPy(IntegerExpression::Constant(2));
        let result = Python::with_gil(|py| {
            let x = x.into_py(py);
            let y = y.into_py(py);
            min(x.into_bound(py), y.into_bound(py))
        });
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            ExprUnion::Int(IntExprPy(IntegerExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(IntegerExpression::Constant(4)),
                Box::new(IntegerExpression::Constant(2)),
            )))
        );
    }

    #[test]
    fn min_float_ok() {
        pyo3::prepare_freethreaded_python();

        let x = FloatExprPy(ContinuousExpression::Constant(4.0));
        let y = FloatExprPy(ContinuousExpression::Constant(2.0));
        let result = Python::with_gil(|py| {
            let x = x.into_py(py);
            let y = y.into_py(py);
            min(x.into_bound(py), y.into_bound(py))
        });
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            ExprUnion::Float(FloatExprPy(ContinuousExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ContinuousExpression::Constant(4.0)),
                Box::new(ContinuousExpression::Constant(2.0)),
            )))
        );
    }

    #[test]
    fn min_element_ok() {
        pyo3::prepare_freethreaded_python();

        let x = ElementExprPy(ElementExpression::Constant(4));
        let y = ElementExprPy(ElementExpression::Constant(2));
        let result = Python::with_gil(|py| {
            let x = x.into_py(py);
            let y = y.into_py(py);
            min(x.into_bound(py), y.into_bound(py))
        });
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            ExprUnion::Element(ElementExprPy(ElementExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ElementExpression::Constant(4)),
                Box::new(ElementExpression::Constant(2)),
            )))
        );
    }

    #[test]
    fn min_err() {
        pyo3::prepare_freethreaded_python();

        let x = ElementExprPy(ElementExpression::Constant(4));
        let y = FloatExprPy(ContinuousExpression::Constant(2.0));
        let result = Python::with_gil(|py| {
            let x = x.into_py(py);
            let y = y.into_py(py);
            min(x.into_bound(py), y.into_bound(py))
        });
        assert!(result.is_err());
    }

    #[test]
    fn condition_to_condition() {
        let condition = ConditionPy(Condition::Constant(true));
        assert_eq!(Condition::from(condition), Condition::Constant(true));
    }

    #[test]
    fn condition_new() {
        assert_eq!(
            ConditionPy::from(Condition::Constant(true)),
            ConditionPy(Condition::Constant(true))
        );
    }

    #[test]
    fn condition_invert() {
        assert_eq!(
            ConditionPy(Condition::Constant(true)).__invert__(),
            ConditionPy(Condition::Not(Box::new(Condition::Constant(true))))
        );
    }

    #[test]
    fn condiiton_and() {
        let x = ConditionPy(Condition::Constant(true));
        let y = ConditionPy(Condition::Constant(false));
        assert_eq!(
            x.__and__(&y),
            ConditionPy(Condition::And(
                Box::new(Condition::Constant(true)),
                Box::new(Condition::Constant(false))
            ))
        );
    }

    #[test]
    fn condiiton_or() {
        let x = ConditionPy(Condition::Constant(true));
        let y = ConditionPy(Condition::Constant(false));
        assert_eq!(
            x.__or__(&y),
            ConditionPy(Condition::Or(
                Box::new(Condition::Constant(true)),
                Box::new(Condition::Constant(false))
            ))
        );
    }

    #[test]
    fn if_then_else_int_ok() {
        pyo3::prepare_freethreaded_python();

        let x = IntExprPy(IntegerExpression::Constant(0));
        let y = IntExprPy(IntegerExpression::Constant(1));
        let condition = ConditionPy(Condition::Constant(true));
        let result = Python::with_gil(|py| {
            let x = x.into_py(py);
            let y = y.into_py(py);
            condition.if_then_else(x.into_bound(py), y.into_bound(py))
        });
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            ExprUnion::Int(IntExprPy(IntegerExpression::If(
                Box::new(Condition::Constant(true)),
                Box::new(IntegerExpression::Constant(0)),
                Box::new(IntegerExpression::Constant(1))
            )))
        );
    }

    #[test]
    fn if_then_else_float_ok() {
        pyo3::prepare_freethreaded_python();

        let x = FloatExprPy(ContinuousExpression::Constant(0.0));
        let y = FloatExprPy(ContinuousExpression::Constant(1.0));
        let condition = ConditionPy(Condition::Constant(true));
        let result = Python::with_gil(|py| {
            let x = x.into_py(py);
            let y = y.into_py(py);
            condition.if_then_else(x.into_bound(py), y.into_bound(py))
        });
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            ExprUnion::Float(FloatExprPy(ContinuousExpression::If(
                Box::new(Condition::Constant(true)),
                Box::new(ContinuousExpression::Constant(0.0)),
                Box::new(ContinuousExpression::Constant(1.0))
            )))
        );
    }

    #[test]
    fn if_then_else_element_ok() {
        pyo3::prepare_freethreaded_python();

        let x = ElementExprPy(ElementExpression::Constant(0));
        let y = ElementExprPy(ElementExpression::Constant(1));
        let condition = ConditionPy(Condition::Constant(true));
        let result = Python::with_gil(|py| {
            let x = x.into_py(py);
            let y = y.into_py(py);
            condition.if_then_else(x.into_bound(py), y.into_bound(py))
        });
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            ExprUnion::Element(ElementExprPy(ElementExpression::If(
                Box::new(Condition::Constant(true)),
                Box::new(ElementExpression::Constant(0)),
                Box::new(ElementExpression::Constant(1))
            )))
        );
    }

    #[test]
    fn if_then_else_err() {
        pyo3::prepare_freethreaded_python();

        let x = ElementExprPy(ElementExpression::Constant(0));
        let y = FloatExprPy(ContinuousExpression::Constant(1.0));
        let condition = ConditionPy(Condition::Constant(true));
        let result = Python::with_gil(|py| {
            let x = x.into_py(py);
            let y = y.into_py(py);
            condition.if_then_else(x.into_bound(py), y.into_bound(py))
        });
        assert!(result.is_err());
    }

    #[test]
    fn int_or_float_expr_from_cost_expression_int() {
        let expression = CostExpression::from(0);
        assert_eq!(
            IntOrFloatExpr::from(expression),
            IntOrFloatExpr::Int(IntExprPy::from(IntegerExpression::Constant(0)))
        )
    }

    #[test]
    fn int_or_float_expr_from_cost_expression_float() {
        let expression = CostExpression::from(0.0);
        assert_eq!(
            IntOrFloatExpr::from(expression),
            IntOrFloatExpr::Float(FloatExprPy::from(ContinuousExpression::Constant(0.0)))
        )
    }
}
