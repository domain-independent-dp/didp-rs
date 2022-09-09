use dypdl::prelude::*;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::pyclass::CompareOp;

#[derive(FromPyObject, Debug, PartialEq, Clone, Copy)]
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

#[derive(FromPyObject, Debug, PartialEq, Clone, Copy)]
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

/// A class representing an element expression.
///
/// If a comparison operator (`<`, `<=`, `==`, `!=`, `>`, `>=`) with an `ElementExpr`, `ElementVar`, `ElementResourceVar`, or `int` is applied, a condition is returned.
/// If an arithmetic operator (`+`, `-`, `*`, `/`, `//`, `%`) with an `ElementExpr`, `ElementVar`, `ElementResourceVar`, or `int` is applied, a new `ElementExpr` is returned.
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
#[pyclass(name = "ElementExpr")]
#[derive(Debug, PartialEq, Clone)]
pub struct ElementExprPy(ElementExpression);

impl From<ElementExprPy> for ElementExpression {
    fn from(expression: ElementExprPy) -> Self {
        expression.0
    }
}

impl ElementExprPy {
    pub fn new(expr: ElementExpression) -> Self {
        Self(expr)
    }
}

#[pymethods]
impl ElementExprPy {
    #[new]
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
}

/// A class representing an element variable.
///
/// If a comparison operator (`<`, `<=`, `==`, `!=`, `>`, `>=`) with an `ElementExpr`, `ElementVar`, `ElementResourceVar`, or `int` is applied, a condition is returned.
/// If an arithmetic operator (`+`, `-`, `*`, `/`, `//`, `%`) with an `ElementExpr`, `ElementVar`, `ElementResourceVar`, or `int` is applied, a new `ElementExpr` is returned.
#[pyclass(name = "ElementVar")]
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct ElementVarPy(ElementVariable);

impl From<ElementVarPy> for ElementVariable {
    fn from(v: ElementVarPy) -> Self {
        v.0
    }
}

impl From<ElementVarPy> for ElementExpression {
    fn from(v: ElementVarPy) -> Self {
        v.0.into()
    }
}

impl ElementVarPy {
    pub fn new(v: ElementVariable) -> ElementVarPy {
        ElementVarPy(v)
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
}

/// A class representing an element resource variable.
///
/// If a comparison operator (`<`, `<=`, `==`, `!=`, `>`, `>=`) with an `ElementExpr`, `ElementVar`, `ElementResourceVar`, or `int` is applied, a condition is returned.
/// If an arithmetic operator (`+`, `-`, `*`, `/`, `//`, `%`) with an `ElementExpr`, `ElementVar`, `ElementResourceVar`, or `int` is applied, a new `ElementExpr` is returned.
#[pyclass(name = "ElementResourceVar")]
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct ElementResourceVarPy(ElementResourceVariable);

impl From<ElementResourceVarPy> for ElementResourceVariable {
    fn from(v: ElementResourceVarPy) -> Self {
        v.0
    }
}

impl From<ElementResourceVarPy> for ElementExpression {
    fn from(v: ElementResourceVarPy) -> Self {
        v.0.into()
    }
}

impl ElementResourceVarPy {
    pub fn new(v: ElementResourceVariable) -> ElementResourceVarPy {
        ElementResourceVarPy(v)
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
/// A class representing a set expression.
///
/// If a comparison operator (`<`, `<=`, `==`, `!=`, `>`, `>=`) with a `SetExpr`, `SetVar`, or `SetConst` is applied, a condition is returned.
/// If an operator (`+`, `-`, `&`, `^`, `|`) with a `SetExpr`, `SetVar`, or `SetConst` is applied, a new `SetExpr` is returned.
#[pyclass(name = "SetExpr")]
#[derive(Debug, PartialEq, Clone)]
pub struct SetExprPy(SetExpression);

impl From<SetExprPy> for SetExpression {
    fn from(expression: SetExprPy) -> Self {
        expression.0
    }
}

impl SetExprPy {
    pub fn new(expression: SetExpression) -> SetExprPy {
        SetExprPy(expression)
    }
}

#[pymethods]
impl SetExprPy {
    fn __richcmp__(&self, other: SetUnion, op: CompareOp) -> ConditionPy {
        let lhs = self.clone().0;
        let rhs = SetExpression::from(other);
        let condition = match op {
            CompareOp::Lt => lhs.clone().is_subset(rhs.clone()) & !rhs.is_subset(lhs),
            CompareOp::Le => lhs.is_subset(rhs),
            CompareOp::Eq => lhs.clone().is_subset(rhs.clone()) & rhs.is_subset(lhs),
            CompareOp::Ne => !lhs.clone().is_subset(rhs.clone()) | !rhs.is_subset(lhs),
            CompareOp::Ge => rhs.is_subset(lhs),
            CompareOp::Gt => rhs.clone().is_subset(lhs.clone()) & !lhs.is_subset(rhs),
        };
        ConditionPy(condition)
    }

    fn __add__(&self, other: SetUnion) -> SetExprPy {
        self.__or__(other)
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

    fn __radd__(&self, other: SetUnion) -> SetExprPy {
        self.__add__(other)
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
    ///     If `element` is `int` and negative.
    #[pyo3(text_signature = "(element)")]
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
    ///     If `element` is `int` and negative.
    #[pyo3(text_signature = "(element)")]
    fn remove(&self, element: ElementUnion) -> SetExprPy {
        let element = ElementExpression::from(element);
        SetExprPy(self.clone().0.remove(element))
    }

    /// difference(other)
    ///
    /// Returns a set where all elements in an input set are removed.
    ///
    /// This method is the same as `-` operation.
    /// This method does not change the instance itself.
    ///
    /// Parameters
    /// ----------
    /// other: SetExpr, SetVar, or SetConst
    ///     Set to remove.
    /// Returns
    /// -------
    /// SetExpr
    ///     The set where all elements in `other` are removed.
    #[pyo3(text_signature = "(other)")]
    fn difference(&self, other: SetUnion) -> SetExprPy {
        self.__sub__(other)
    }

    /// intersection(other)
    ///
    /// Returns the intersection with another set.
    ///
    /// This method is the same as `&` operation.
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
    #[pyo3(text_signature = "(other)")]
    fn intersection(&self, other: SetUnion) -> SetExprPy {
        self.__and__(other)
    }

    /// symmetric_difference(other)
    ///
    /// Returns a set which only contains elements included in either of two sets but not in both.
    ///
    /// This method is the same as `^` operation.
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
    ///     The symemtric difference set.
    #[pyo3(text_signature = "(other)")]
    fn symmetric_difference(&self, other: SetUnion) -> SetExprPy {
        self.__xor__(other)
    }

    /// union(other)
    ///
    /// Returns the union of two sets.
    ///
    /// This method is the same as `\|` operation.
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
    #[pyo3(text_signature = "(other)")]
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
    ///     If `element` is `int` and negative.
    #[pyo3(text_signature = "(element)")]
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
    #[pyo3(text_signature = "()")]
    pub fn len(&self) -> IntExprPy {
        IntExprPy(self.clone().0.len())
    }

    /// Returns a condition checking if the set is empty.
    ///
    /// Returns
    /// -------
    /// Condition
    ///     The condition checking if the set is empty.
    #[pyo3(text_signature = "()")]
    pub fn is_empty(&self) -> ConditionPy {
        ConditionPy(self.clone().0.is_empty())
    }

    /// Returns the comeplement set.
    ///
    /// Returns
    /// -------
    /// SetExpr
    ///     The complement set.
    #[pyo3(text_signature = "()")]
    pub fn complement(&self) -> SetExprPy {
        SetExprPy(!self.clone().0)
    }
}

/// A class representing a set variable.
///
/// If a comparison operator (`<`, `<=`, `==`, `!=`, `>`, `>=`) with a `SetExpr`, `SetVar`, or `SetConst` is applied, a condition is returned.
/// If an operator (`+`, `-`, `&`, `^`, `|`) with a `SetExpr`, `SetVar`, or `SetConst` is applied, a new `SetExpr` is returned.
#[pyclass(name = "SetVar")]
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct SetVarPy(SetVariable);

impl From<SetVarPy> for SetVariable {
    fn from(v: SetVarPy) -> Self {
        v.0
    }
}

impl From<SetVarPy> for SetExpression {
    fn from(v: SetVarPy) -> Self {
        v.0.into()
    }
}

impl SetVarPy {
    pub fn new(v: SetVariable) -> SetVarPy {
        SetVarPy(v)
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
            CompareOp::Eq => lhs.is_subset(rhs.clone()) & rhs.is_subset(lhs),
            CompareOp::Ne => !lhs.is_subset(rhs.clone()) | !rhs.is_subset(lhs),
            CompareOp::Ge => rhs.is_subset(lhs),
            CompareOp::Gt => rhs.clone().is_subset(lhs) & !lhs.is_subset(rhs),
        };
        ConditionPy(condition)
    }

    fn __add__(&self, other: SetUnion) -> SetExprPy {
        self.__or__(other)
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

    fn __radd__(&self, other: SetUnion) -> SetExprPy {
        self.__add__(other)
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
    ///     If `element` is `int` and negative.
    #[pyo3(text_signature = "(element)")]
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
    ///     If `element` is `int` and negative.
    #[pyo3(text_signature = "(element)")]
    fn remove(&self, element: ElementUnion) -> SetExprPy {
        let element = ElementExpression::from(element);
        SetExprPy(self.0.remove(element))
    }

    /// difference(other)
    ///
    /// Returns a set where all elements in an input set are removed.
    ///
    /// This method is the same as `-` operation.
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
    ///     The set where all elements in `other` are removed.
    #[pyo3(text_signature = "(other)")]
    fn difference(&self, other: SetUnion) -> SetExprPy {
        self.__sub__(other)
    }

    /// intersection(other)
    ///
    /// Returns the intersection with another set.
    ///
    /// This method is the same as `&` operation.
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
    fn intersection(&self, other: SetUnion) -> SetExprPy {
        self.__and__(other)
    }

    /// symmetric_difference(other)
    ///
    /// Returns a set which only contains elements included in either of two sets but not in both.
    ///
    /// This method is the same as `^` operation.
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
    ///     The symemtric difference set.
    #[pyo3(text_signature = "(other)")]
    fn symmetric_difference(&self, other: SetUnion) -> SetExprPy {
        self.__xor__(other)
    }

    /// union(other)
    ///
    /// Returns the union of two sets.
    ///
    /// This method is the same as `\|` operation.
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
    #[pyo3(text_signature = "(other)")]
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
    ///     If `element` is `int` and negative.
    #[pyo3(text_signature = "(element)")]
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
    #[pyo3(text_signature = "()")]
    fn len(&self) -> IntExprPy {
        IntExprPy(self.0.len())
    }

    /// Returns a condition checking if the set is empty.
    ///
    /// Returns
    /// -------
    /// Condition
    ///     The condition checking if the set is empty.
    #[pyo3(text_signature = "()")]
    fn is_empty(&self) -> ConditionPy {
        ConditionPy(self.0.is_empty())
    }

    /// Returns the comeplement set.
    ///
    /// Returns
    /// -------
    /// SetExpr
    ///     The complement set.
    #[pyo3(text_signature = "()")]
    fn complement(&self) -> SetExprPy {
        SetExprPy(!self.0)
    }
}

#[pyclass(name = "SetConst")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetConstPy(Set);

impl From<SetConstPy> for Set {
    fn from(set: SetConstPy) -> Self {
        set.0
    }
}

impl From<SetConstPy> for SetExpression {
    fn from(set: SetConstPy) -> Self {
        set.0.into()
    }
}

impl SetConstPy {
    pub fn new(set: Set) -> SetConstPy {
        SetConstPy(set)
    }
}

/// A class representing a set constant.
///
/// If a comparison operator (`<`, `<=`, `==`, `!=`, `>`, `>=`) with a `SetExpr`, `SetVar`, or `SetConst` is applied, a condition is returned.
/// If an operator (`+`, `-`, `&`, `^`, `|`) with a `SetExpr`, `SetVar`, or `SetConst` is applied, a new `SetExpr` is returned.
#[pymethods]
impl SetConstPy {
    fn __richcmp__(&self, other: SetUnion, op: CompareOp) -> ConditionPy {
        let lhs = SetExpression::from(self.clone());
        let rhs = SetExpression::from(other);
        let condition = match op {
            CompareOp::Lt => lhs.clone().is_subset(rhs.clone()) & !rhs.is_subset(lhs),
            CompareOp::Le => lhs.is_subset(rhs),
            CompareOp::Eq => lhs.clone().is_subset(rhs.clone()) & rhs.is_subset(lhs),
            CompareOp::Ne => !lhs.clone().is_subset(rhs.clone()) | !rhs.is_subset(lhs),
            CompareOp::Ge => rhs.is_subset(lhs),
            CompareOp::Gt => rhs.clone().is_subset(lhs.clone()) & !lhs.is_subset(rhs),
        };
        ConditionPy(condition)
    }

    fn __add__(&self, other: SetUnion) -> SetExprPy {
        self.__or__(other)
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

    fn __radd__(&self, other: SetUnion) -> SetExprPy {
        self.__add__(other)
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
    #[pyo3(text_signature = "(element)")]
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
    ///     If `element` is `int` and negative.
    #[pyo3(text_signature = "(element)")]
    fn remove(&self, element: ElementUnion) -> SetExprPy {
        let set = SetExpression::from(self.clone());
        let element = ElementExpression::from(element);
        SetExprPy(set.remove(element))
    }

    /// difference(other)
    ///
    /// Returns a set where all elements in an input set are removed.
    ///
    /// This method is the same as `-` operation.
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
    ///     The set where all elements in `other` are removed.
    ///
    /// Raises
    /// ------
    /// OverflowError
    ///     If `element` is `int` and negative.
    #[pyo3(text_signature = "(other)")]
    fn difference(&self, other: SetUnion) -> SetExprPy {
        self.__sub__(other)
    }

    /// intersection(other)
    ///
    /// Returns the intersection with another set.
    ///
    /// This method is the same as `&` operation.
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
    #[pyo3(text_signature = "(other)")]
    fn intersection(&self, other: SetUnion) -> SetExprPy {
        self.__and__(other)
    }

    /// symmetric_difference(other)
    ///
    /// Returns a set which only contains elements included in either of two sets but not in both.
    ///
    /// This method is the same as `^` operation.
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
    ///     The symemtric difference set.
    #[pyo3(text_signature = "(other)")]
    fn symmetric_difference(&self, other: SetUnion) -> SetExprPy {
        self.__xor__(other)
    }

    /// union(other)
    ///
    /// Returns the union of two sets.
    ///
    /// This method is the same as `\|` operation.
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
    #[pyo3(text_signature = "(other)")]
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
    ///     If `element` is `int` and negative.
    #[pyo3(text_signature = "(element)")]
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
    #[pyo3(text_signature = "()")]
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
    #[pyo3(text_signature = "()")]
    fn is_empty(&self) -> ConditionPy {
        let set = SetExpression::from(self.clone());
        ConditionPy(set.is_empty())
    }

    /// Returns the comeplement set.
    ///
    /// Returns
    /// -------
    /// SetExpr
    ///     The complement set.
    #[pyo3(text_signature = "()")]
    fn complement(&self) -> SetExprPy {
        let set = SetExpression::from(self.clone());
        SetExprPy(!set)
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

/// A class representing an integer expression.
///
/// If a comparison operator (`<`, `<=`, `==`, `!=`, `>`, `>=`) with an `IntExpr`, `IntVar`, `IntResourceVar`, `FloatExpr`, `FloatVar`, `FloatResourceVar`, `int`, or `float` is applied a condition is returned.
/// If an arithmetic operator (`+`, `-`, `*`, `/`, `%`) with an `IntExpr`, `IntVar`, `IntResourceVar`, or `int` is applied, a new `IntExpr` is returned.
/// For division (`/`) and power (`**`), a `FloatExpr` is returned.
/// If an arithmetic operator with an `FloatExpr`, `FloatVar`, `FloatResourceVar`, or `float` is applied, a `FloatExpr` is returned.
///
/// Parameters
/// ----------
/// value : int
///     A value from which a constant expression is created.
#[pyclass(name = "IntExpr")]
#[derive(Debug, PartialEq, Clone)]
pub struct IntExprPy(IntegerExpression);

impl From<IntExprPy> for IntegerExpression {
    fn from(expression: IntExprPy) -> Self {
        expression.0
    }
}

impl IntExprPy {
    pub fn new(expr: IntegerExpression) -> Self {
        Self(expr)
    }
}

#[pymethods]
impl IntExprPy {
    #[new]
    fn new_py(value: Integer) -> Self {
        Self(IntegerExpression::from(value))
    }

    /// Returns the cost of the transitioned state, which can be used in a cost expression.
    ///
    /// Returns
    /// -------
    /// IntExpr
    ///     The cost of the transitioned state.
    #[pyo3(text_signature = "()")]
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
}

/// A class representing an integer variable.
///
/// If a comparison operator (`<`, `<=`, `==`, `!=`, `>`, `>=`) with an `IntExpr`, `IntVar`, `IntResourceVar`, `FloatExpr`, `FloatVar`, `FloatResourceVar`, `int`, or `float` is applied a condition is returned.
/// If an arithmetic operator (`+`, `-`, `*`, `//`, `%`) with an `IntExpr`, `IntVar`, `IntResourceVar`, or `int` is applied, a new `IntExpr` is returned.
/// For division (`/`) and power (`**`), a `FloatExpr` is returned.
/// If an arithmetic operator with an `FloatExpr`, `FloatVar`, `FloatResourceVar`, or `float` is applied, a `FloatExpr` is returned.
#[pyclass(name = "IntVar")]
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct IntVarPy(IntegerVariable);

impl From<IntVarPy> for IntegerVariable {
    fn from(v: IntVarPy) -> Self {
        v.0
    }
}

impl From<IntVarPy> for IntegerExpression {
    fn from(v: IntVarPy) -> Self {
        v.0.into()
    }
}

impl IntVarPy {
    pub fn new(v: IntegerVariable) -> IntVarPy {
        IntVarPy(v)
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
}

/// A class representing an integer resource variable.
///
/// If a comparison operator (`<`, `<=`, `==`, `!=`, `>`, `>=`) with an `IntExpr`, `IntVar`, `IntResourceVar`, `FloatExpr`, `FloatVar`, `FloatResourceVar`, `int`, or `float` is applied a condition is returned.
/// If an arithmetic operator (`+`, `-`, `*`, `//`, `%`) with an `IntExpr`, `IntVar`, `IntResourceVar`, or `int` is applied, a new `IntExpr` is returned.
/// For division (`/`) and power (`**`), a `FloatExpr` is returned.
/// If an arithmetic operator with an `FloatExpr`, `FloatVar`, `FloatResourceVar`, or `float` is applied, a `FloatExpr` is returned.
#[pyclass(name = "IntResourceVar")]
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct IntResourceVarPy(IntegerResourceVariable);

impl From<IntResourceVarPy> for IntegerResourceVariable {
    fn from(v: IntResourceVarPy) -> Self {
        v.0
    }
}

impl From<IntResourceVarPy> for IntegerExpression {
    fn from(v: IntResourceVarPy) -> Self {
        v.0.into()
    }
}

impl IntResourceVarPy {
    pub fn new(v: IntegerResourceVariable) -> IntResourceVarPy {
        IntResourceVarPy(v)
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

/// A class representing a continuous expression.
///
/// If a comparison operator (`<`, `<=`, `==`, `!=`, `>`, `>=`) with an `IntExpr`, `IntVar`, `IntResourceVar`, `FloatExpr`, `FloatVar`, `FloatResourceVar`, `int`, or `float` is applied a condition is returned.
/// If an arithmetic operator (`+`, `-`, `*`, `/`, `//`, `%`, `**`) with an `IntExpr`, `IntVar`, `IntResourceVar`, `FloatExpr`, `FloatVar`, `FloatResourceVar`, `int`, or `float` is applied, a new `FloatExpr` is returned.
/// `round`, `trunc`, `floor`, and `ceil` return an `IntExpr`.
///
/// Parameters
/// ----------
/// value : float
///     A value from which a constant expression is created.
#[pyclass(name = "FloatExpr")]
#[derive(Debug, PartialEq, Clone)]
pub struct FloatExprPy(ContinuousExpression);

impl From<FloatExprPy> for ContinuousExpression {
    fn from(expression: FloatExprPy) -> Self {
        expression.0
    }
}

impl FloatExprPy {
    pub fn new(expr: ContinuousExpression) -> Self {
        Self(expr)
    }
}

#[pymethods]
impl FloatExprPy {
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
    #[pyo3(text_signature = "()")]
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
}

/// A class representing a continuous variable.
///
/// If a comparison operator (`<`, `<=`, `==`, `!=`, `>`, `>=`) with an `IntExpr`, `IntVar`, `IntResourceVar`, `FloatExpr`, `FloatVar`, `FloatResourceVar`, `int`, or `float` is applied a condition is returned.
/// If an arithmetic operator (`+`, `-`, `*`, `/`, `//`, `%`, `**`) with an `IntExpr`, `IntVar`, `IntResourceVar`, `FloatExpr`, `FloatVar`, `FloatResourceVar`, `int`, or `float` is applied, a new `FloatExpr` is returned.
/// `round`, `trunc`, `floor`, and `ceil` return an `IntExpr`.
#[pyclass(name = "FloatVar")]
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct FloatVarPy(ContinuousVariable);

impl From<FloatVarPy> for ContinuousVariable {
    fn from(v: FloatVarPy) -> Self {
        v.0
    }
}

impl From<FloatVarPy> for ContinuousExpression {
    fn from(v: FloatVarPy) -> Self {
        v.0.into()
    }
}

impl FloatVarPy {
    pub fn new(v: ContinuousVariable) -> FloatVarPy {
        FloatVarPy(v)
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
}

/// A class representing a continuous resource variable.
///
/// If a comparison operator (`<`, `<=`, `==`, `!=`, `>`, `>=`) with an `IntExpr`, `IntVar`, `IntResourceVar`, `FloatExpr`, `FloatVar`, `FloatResourceVar`, `int`, or `float` is applied a condition is returned.
/// If an arithmetic operator (`+`, `-`, `*`, `/`, `//`, `%`, `**`) with an `IntExpr`, `IntVar`, `IntResourceVar`, `FloatExpr`, `FloatVar`, `FloatResourceVar`, `int`, or `float` is applied, a new `FloatExpr` is returned.
/// `round`, `trunc`, `floor`, and `ceil` return an `IntExpr`.
#[pyclass(name = "FloatResourceVar")]
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct FloatResourceVarPy(ContinuousResourceVariable);

impl From<FloatResourceVarPy> for ContinuousResourceVariable {
    fn from(v: FloatResourceVarPy) -> Self {
        v.0
    }
}

impl From<FloatResourceVarPy> for ContinuousExpression {
    fn from(v: FloatResourceVarPy) -> Self {
        v.0.into()
    }
}

impl FloatResourceVarPy {
    pub fn new(v: ContinuousResourceVariable) -> FloatResourceVarPy {
        FloatResourceVarPy(v)
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
}

/// Returns the square root.
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
#[pyfunction]
#[pyo3(text_signature = "(x)")]
pub fn sqrt(x: FloatUnion) -> FloatExprPy {
    FloatExprPy(ContinuousExpression::from(x).sqrt())
}

/// Returns the logarithm of `x` using y `as` a base.
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
#[pyfunction]
#[pyo3(text_signature = "(x, y)")]
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
#[pyfunction]
#[pyo3(text_signature = "(x)")]
pub fn float(x: IntUnion) -> FloatExprPy {
    FloatExprPy(ContinuousExpression::from(IntegerExpression::from(x)))
}

#[derive(FromPyObject, Debug, PartialEq, Clone)]
pub enum NumericArgUnion {
    #[pyo3(transparent)]
    Element(ElementUnion),
    #[pyo3(transparent)]
    Int(IntUnion),
    #[pyo3(transparent)]
    Float(FloatUnion),
}

#[derive(Debug, PartialEq, Clone)]
pub enum NumericReturnUnion {
    Element(ElementExprPy),
    Int(IntExprPy),
    Float(FloatExprPy),
}

impl IntoPy<Py<PyAny>> for NumericReturnUnion {
    fn into_py(self, py: Python<'_>) -> Py<PyAny> {
        match self {
            Self::Element(expr) => expr.into_py(py),
            Self::Int(expr) => expr.into_py(py),
            Self::Float(expr) => expr.into_py(py),
        }
    }
}

/// Returns the greater value.
///
/// Parameters
/// ----------
/// x: ElementExpr, ElementVar, ElementResourceVar, IntExpr, IntVar, IntResourceVar, FloatExpr, FloatVar, FloatResourceVar, int, or float
///     First input.
/// y: ElementExpr, ElementVar, ElementResourceVar, IntExpr, IntVar, IntResourceVar, FloatExpr, FloatVar, FloatResourceVar, int, or float
///     Second input.
///
/// Returns
/// -------
/// ElementExpr, IntExpr, or FloatExpr
///     The greater value.
///
/// Raises
/// ------
/// TypeError
///     If the types of `x` and `y` mismatch.
#[pyfunction]
#[pyo3(text_signature = "(x, y)")]
pub fn max(x: &PyAny, y: &PyAny) -> PyResult<NumericReturnUnion> {
    let result: (PyResult<IntUnion>, PyResult<IntUnion>) = (x.extract(), y.extract());
    if let (Ok(x), Ok(y)) = result {
        let x = IntegerExpression::from(x);
        let y = IntegerExpression::from(y);
        return Ok(NumericReturnUnion::Int(IntExprPy(x.max(y))));
    }
    let result: (PyResult<FloatUnion>, PyResult<FloatUnion>) = (x.extract(), y.extract());
    if let (Ok(x), Ok(y)) = result {
        let x = ContinuousExpression::from(x);
        let y = ContinuousExpression::from(y);
        return Ok(NumericReturnUnion::Float(FloatExprPy(x.max(y))));
    }
    let result: (PyResult<ElementUnion>, PyResult<ElementUnion>) = (x.extract(), y.extract());
    if let (Ok(x), Ok(y)) = result {
        let x = ElementExpression::from(x);
        let y = ElementExpression::from(y);
        return Ok(NumericReturnUnion::Element(ElementExprPy(x.max(y))));
    }
    Err(PyTypeError::new_err("arguments ('x', 'y') failed to extract (IntExpr, IntExpr), (FloatExpr, FloatExpr), or (ElementExpr, ElementExpr)"))
}

/// Returns the smaller value.
///
/// Parameters
/// ----------
/// x: ElementExpr, ElementVar, ElementResourceVar, IntExpr, IntVar, IntResourceVar, FloatExpr, FloatVar, FloatResourceVar, int, or float
///     First input.
/// y: ElementExpr, ElementVar, ElementResourceVar, IntExpr, IntVar, IntResourceVar, FloatExpr, FloatVar, FloatResourceVar, int, or float
///     Second input.
///
/// Returns
/// -------
/// ElementExpr, IntExpr, or FloatExpr
///     The smaller value.
///
/// Raises
/// ------
/// TypeError
///     If the types of `x` and `y` mismatch.
#[pyfunction]
#[pyo3(text_signature = "(x, y)")]
pub fn min(x: &PyAny, y: &PyAny) -> PyResult<NumericReturnUnion> {
    let result: (PyResult<IntUnion>, PyResult<IntUnion>) = (x.extract(), y.extract());
    if let (Ok(x), Ok(y)) = result {
        let x = IntegerExpression::from(x);
        let y = IntegerExpression::from(y);
        return Ok(NumericReturnUnion::Int(IntExprPy(x.min(y))));
    }
    let result: (PyResult<FloatUnion>, PyResult<FloatUnion>) = (x.extract(), y.extract());
    if let (Ok(x), Ok(y)) = result {
        let x = ContinuousExpression::from(x);
        let y = ContinuousExpression::from(y);
        return Ok(NumericReturnUnion::Float(FloatExprPy(x.min(y))));
    }
    let result: (PyResult<ElementUnion>, PyResult<ElementUnion>) = (x.extract(), y.extract());
    if let (Ok(x), Ok(y)) = result {
        let x = ElementExpression::from(x);
        let y = ElementExpression::from(y);
        return Ok(NumericReturnUnion::Element(ElementExprPy(x.min(y))));
    }
    Err(PyTypeError::new_err("arguments ('x', 'y') failed to extract (IntExpr, IntExpr), (FloatExpr, FloatExpr), or (ElementExpr, ElementExpr)"))
}

#[pyclass(name = "Condition")]
#[derive(Debug, PartialEq, Clone)]
pub struct ConditionPy(Condition);

impl ConditionPy {
    pub fn new(condition: Condition) -> ConditionPy {
        ConditionPy(condition)
    }
}

impl From<ConditionPy> for Condition {
    fn from(condition: ConditionPy) -> Self {
        condition.0
    }
}

/// A class representin a conditon.
///
/// The neation of a condition can be crated by `~x`.
/// The conjunction of two conditions can be crated by `x & y`.
/// The disjunction of two conditions can be crated by `x | y`.
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
    ///     The type of the return value is determined according to the types of `x` and `y`.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If the types of `x` and `y` mismatch.
    #[pyo3(text_signature = "(x, y)")]
    fn if_then_else(&self, x: &PyAny, y: &PyAny) -> PyResult<NumericReturnUnion> {
        let result: (PyResult<IntUnion>, PyResult<IntUnion>) = (x.extract(), y.extract());
        if let (Ok(x), Ok(y)) = result {
            let x = IntegerExpression::from(x);
            let y = IntegerExpression::from(y);
            return Ok(NumericReturnUnion::Int(IntExprPy(
                self.clone().0.if_then_else(x, y),
            )));
        }
        let result: (PyResult<FloatUnion>, PyResult<FloatUnion>) = (x.extract(), y.extract());
        if let (Ok(x), Ok(y)) = result {
            let x = ContinuousExpression::from(x);
            let y = ContinuousExpression::from(y);
            return Ok(NumericReturnUnion::Float(FloatExprPy(
                self.clone().0.if_then_else(x, y),
            )));
        }
        let result: (PyResult<ElementUnion>, PyResult<ElementUnion>) = (x.extract(), y.extract());
        if let (Ok(x), Ok(y)) = result {
            let x = ElementExpression::from(x);
            let y = ElementExpression::from(y);
            return Ok(NumericReturnUnion::Element(ElementExprPy(
                self.clone().0.if_then_else(x, y),
            )));
        }
        Err(PyTypeError::new_err("arguments ('x', 'y') failed to extract (IntExpr, IntExpr), (FloatExpr, FloatExpr), or (ElementExpr, ElementExpr)"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dypdl::expression::*;

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
            ElementExprPy::new(ElementExpression::Constant(0)),
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
        assert_eq!(ElementVarPy::new(v), ElementVarPy(v));
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
        assert_eq!(ElementResourceVarPy::new(v), ElementResourceVarPy(v));
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
            SetExprPy::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
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
            ConditionPy(Condition::And(
                Box::new(Condition::Set(Box::new(SetCondition::IsSubset(
                    SetExpression::Reference(ReferenceExpression::Variable(0)),
                    SetExpression::Reference(ReferenceExpression::Variable(1)),
                )))),
                Box::new(Condition::Set(Box::new(SetCondition::IsSubset(
                    SetExpression::Reference(ReferenceExpression::Variable(1)),
                    SetExpression::Reference(ReferenceExpression::Variable(0)),
                ))))
            ))
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
            ConditionPy(Condition::Or(
                Box::new(Condition::Not(Box::new(Condition::Set(Box::new(
                    SetCondition::IsSubset(
                        SetExpression::Reference(ReferenceExpression::Variable(0)),
                        SetExpression::Reference(ReferenceExpression::Variable(1)),
                    )
                ))))),
                Box::new(Condition::Not(Box::new(Condition::Set(Box::new(
                    SetCondition::IsSubset(
                        SetExpression::Reference(ReferenceExpression::Variable(1)),
                        SetExpression::Reference(ReferenceExpression::Variable(0)),
                    )
                )))))
            ))
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
    fn set_expr_add() {
        let expression = SetExprPy(SetExpression::Reference(ReferenceExpression::Variable(0)));
        let other = SetUnion::Expr(SetExprPy(SetExpression::Reference(
            ReferenceExpression::Variable(1),
        )));
        assert_eq!(
            expression.__add__(other),
            SetExprPy(SetExpression::SetOperation(
                SetOperator::Union,
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
            ))
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
    fn set_expr_radd() {
        let expression = SetExprPy(SetExpression::Reference(ReferenceExpression::Variable(0)));
        let other = SetUnion::Expr(SetExprPy(SetExpression::Reference(
            ReferenceExpression::Variable(1),
        )));
        assert_eq!(
            expression.__radd__(other),
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
        assert_eq!(SetVarPy::new(v), SetVarPy(v));
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
            ConditionPy(Condition::And(
                Box::new(Condition::Set(Box::new(SetCondition::IsSubset(
                    SetExpression::Reference(ReferenceExpression::Variable(v.id())),
                    SetExpression::Reference(ReferenceExpression::Variable(1)),
                )))),
                Box::new(Condition::Set(Box::new(SetCondition::IsSubset(
                    SetExpression::Reference(ReferenceExpression::Variable(1)),
                    SetExpression::Reference(ReferenceExpression::Variable(v.id())),
                ))))
            ))
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
            ConditionPy(Condition::Or(
                Box::new(Condition::Not(Box::new(Condition::Set(Box::new(
                    SetCondition::IsSubset(
                        SetExpression::Reference(ReferenceExpression::Variable(v.id())),
                        SetExpression::Reference(ReferenceExpression::Variable(1)),
                    )
                ))))),
                Box::new(Condition::Not(Box::new(Condition::Set(Box::new(
                    SetCondition::IsSubset(
                        SetExpression::Reference(ReferenceExpression::Variable(1)),
                        SetExpression::Reference(ReferenceExpression::Variable(v.id())),
                    )
                )))))
            ))
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
    fn set_var_add() {
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
            expression.__add__(other),
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
    fn set_var_radd() {
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
            expression.__radd__(other),
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
            SetConstPy::new(Set::with_capacity(10)),
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
            ConditionPy(Condition::And(
                Box::new(Condition::Set(Box::new(SetCondition::IsSubset(
                    SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(10))),
                    SetExpression::Reference(ReferenceExpression::Variable(1)),
                )))),
                Box::new(Condition::Set(Box::new(SetCondition::IsSubset(
                    SetExpression::Reference(ReferenceExpression::Variable(1)),
                    SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(10))),
                ))))
            ))
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
            ConditionPy(Condition::Or(
                Box::new(Condition::Not(Box::new(Condition::Set(Box::new(
                    SetCondition::IsSubset(
                        SetExpression::Reference(ReferenceExpression::Constant(
                            Set::with_capacity(10)
                        )),
                        SetExpression::Reference(ReferenceExpression::Variable(1)),
                    )
                ))))),
                Box::new(Condition::Not(Box::new(Condition::Set(Box::new(
                    SetCondition::IsSubset(
                        SetExpression::Reference(ReferenceExpression::Variable(1)),
                        SetExpression::Reference(ReferenceExpression::Constant(
                            Set::with_capacity(10)
                        )),
                    )
                )))))
            ))
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
    fn set_const_add() {
        let expression = SetConstPy(Set::with_capacity(10));
        let other = SetUnion::Expr(SetExprPy(SetExpression::Reference(
            ReferenceExpression::Variable(1),
        )));
        assert_eq!(
            expression.__add__(other),
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
    fn set_const_radd() {
        let expression = SetConstPy(Set::with_capacity(10));
        let other = SetUnion::Expr(SetExprPy(SetExpression::Reference(
            ReferenceExpression::Variable(1),
        )));
        assert_eq!(
            expression.__radd__(other),
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
            IntExprPy::new(IntegerExpression::Constant(0)),
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
        assert_eq!(IntVarPy::new(v), IntVarPy(v));
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
        assert_eq!(IntResourceVarPy::new(v), IntResourceVarPy(v));
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
            FloatExprPy::new(ContinuousExpression::Constant(0.0)),
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

        assert_eq!(FloatVarPy::new(v), FloatVarPy(v));
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

        assert_eq!(FloatResourceVarPy::new(v), FloatResourceVarPy(v));
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
            max(x.as_ref(py), y.as_ref(py))
        });
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            NumericReturnUnion::Int(IntExprPy(IntegerExpression::BinaryOperation(
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
            max(x.as_ref(py), y.as_ref(py))
        });
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            NumericReturnUnion::Float(FloatExprPy(ContinuousExpression::BinaryOperation(
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
            max(x.as_ref(py), y.as_ref(py))
        });
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            NumericReturnUnion::Element(ElementExprPy(ElementExpression::BinaryOperation(
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
            min(x.as_ref(py), y.as_ref(py))
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
            min(x.as_ref(py), y.as_ref(py))
        });
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            NumericReturnUnion::Int(IntExprPy(IntegerExpression::BinaryOperation(
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
            min(x.as_ref(py), y.as_ref(py))
        });
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            NumericReturnUnion::Float(FloatExprPy(ContinuousExpression::BinaryOperation(
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
            min(x.as_ref(py), y.as_ref(py))
        });
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            NumericReturnUnion::Element(ElementExprPy(ElementExpression::BinaryOperation(
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
            min(x.as_ref(py), y.as_ref(py))
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
            ConditionPy::new(Condition::Constant(true)),
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
            condition.if_then_else(x.as_ref(py), y.as_ref(py))
        });
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            NumericReturnUnion::Int(IntExprPy(IntegerExpression::If(
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
            condition.if_then_else(x.as_ref(py), y.as_ref(py))
        });
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            NumericReturnUnion::Float(FloatExprPy(ContinuousExpression::If(
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
            condition.if_then_else(x.as_ref(py), y.as_ref(py))
        });
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            NumericReturnUnion::Element(ElementExprPy(ElementExpression::If(
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
            condition.if_then_else(x.as_ref(py), y.as_ref(py))
        });
        assert!(result.is_err());
    }
}
