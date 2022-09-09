use super::continuous_expression::ContinuousExpression;
use super::element_expression::ElementExpression;
use super::integer_expression::IntegerExpression;
use super::set_expression::SetExpression;
use super::table_expression::TableExpression;
use super::{set_condition, SetCondition};
use crate::state::{DPState, SetVariable};
use crate::table_data::{Table1DHandle, Table2DHandle, Table3DHandle, TableHandle};
use crate::table_registry::TableRegistry;
use std::ops;

/// Operator for arithmetic comparison.
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum ComparisonOperator {
    /// The equal to operator.
    Eq,
    /// The not equal to operator.
    Ne,
    /// The greater than or equal to operator.
    Ge,
    /// The greater than operator.
    Gt,
    /// The less than operator.
    Le,
    /// The less than or equal to operator.
    Lt,
}

impl ComparisonOperator {
    /// Evaluate comparison.
    pub fn eval<T: PartialOrd>(&self, x: T, y: T) -> bool {
        match self {
            Self::Eq => x == y,
            Self::Ne => x != y,
            Self::Ge => x >= y,
            Self::Gt => x > y,
            Self::Le => x <= y,
            Self::Lt => x < y,
        }
    }
}

/// Condition.
#[derive(Debug, PartialEq, Clone)]
pub enum Condition {
    /// Constant.
    Constant(bool),
    /// Not x.
    Not(Box<Condition>),
    /// x and b.
    And(Box<Condition>, Box<Condition>),
    /// x or b.
    Or(Box<Condition>, Box<Condition>),
    /// Comparing two element expressions.
    ComparisonE(
        ComparisonOperator,
        Box<ElementExpression>,
        Box<ElementExpression>,
    ),
    /// Comparing two integer expressions.
    ComparisonI(
        ComparisonOperator,
        Box<IntegerExpression>,
        Box<IntegerExpression>,
    ),
    /// Comparing two continuous expressions.
    ComparisonC(
        ComparisonOperator,
        Box<ContinuousExpression>,
        Box<ContinuousExpression>,
    ),
    /// Set condition.
    Set(Box<set_condition::SetCondition>),
    /// A constant in a boolean table.
    Table(Box<TableExpression<bool>>),
}

/// A trait to produce if-then-else expression.
pub trait IfThenElse<T> {
    /// Returns an if-then-else expression, which retunrs a if this condition holds and b otherwise.
    fn if_then_else<U, V>(self, a: U, b: V) -> T
    where
        T: From<U> + From<V>;
}

impl Default for Condition {
    fn default() -> Condition {
        Self::Constant(false)
    }
}
impl ops::Not for Condition {
    type Output = Condition;

    /// Returns the negation of this condition.
    #[inline]
    fn not(self) -> Self::Output {
        Self::Not(Box::new(self))
    }
}

impl ops::BitAnd for Condition {
    type Output = Condition;

    /// Returns the conjuction of this condition and rhs.
    #[inline]
    fn bitand(self, rhs: Self) -> Self::Output {
        Self::And(Box::new(self), Box::new(rhs))
    }
}

impl ops::BitOr for Condition {
    type Output = Condition;

    /// Returns the disjunction of this condition and rhs.
    #[inline]
    fn bitor(self, rhs: Self) -> Self::Output {
        Self::Or(Box::new(self), Box::new(rhs))
    }
}

impl Condition {
    /// Returns a condition comparing two element expressions.
    #[inline]
    pub fn comparison_e<T, U>(op: ComparisonOperator, lhs: T, rhs: U) -> Self
    where
        ElementExpression: From<T>,
        ElementExpression: From<U>,
    {
        Self::ComparisonE(
            op,
            Box::new(ElementExpression::from(lhs)),
            Box::new(ElementExpression::from(rhs)),
        )
    }

    #[inline]
    /// Returns a condition comparing two integer expressions.
    pub fn comparison_i<T, U>(op: ComparisonOperator, lhs: T, rhs: U) -> Self
    where
        IntegerExpression: From<T>,
        IntegerExpression: From<U>,
    {
        Self::ComparisonI(
            op,
            Box::new(IntegerExpression::from(lhs)),
            Box::new(IntegerExpression::from(rhs)),
        )
    }

    #[inline]
    /// Returns a condition comparing two continuous expressions.
    pub fn comparison_c<T, U>(op: ComparisonOperator, lhs: T, rhs: U) -> Self
    where
        ContinuousExpression: From<T>,
        ContinuousExpression: From<U>,
    {
        Self::ComparisonC(
            op,
            Box::new(ContinuousExpression::from(lhs)),
            Box::new(ContinuousExpression::from(rhs)),
        )
    }
}

impl SetExpression {
    /// Returns a condition checking if an element is included in this set.
    #[inline]
    pub fn contains<T>(self, element: T) -> Condition
    where
        ElementExpression: From<T>,
    {
        Condition::Set(Box::new(SetCondition::IsIn(
            ElementExpression::from(element),
            self,
        )))
    }

    /// Returns a condition checking if this set is a subset of the other.
    #[inline]
    pub fn is_subset<T>(self, set: T) -> Condition
    where
        SetExpression: From<T>,
    {
        Condition::Set(Box::new(SetCondition::IsSubset(
            self,
            SetExpression::from(set),
        )))
    }

    /// Returns a condition checking if this set is empty.
    #[inline]
    pub fn is_empty(self) -> Condition {
        Condition::Set(Box::new(SetCondition::IsEmpty(self)))
    }
}

impl SetVariable {
    /// Returns a condition checking if an element is included in this set.
    #[inline]
    pub fn contains<T>(self, element: T) -> Condition
    where
        ElementExpression: From<T>,
    {
        Condition::Set(Box::new(SetCondition::IsIn(
            ElementExpression::from(element),
            SetExpression::from(self),
        )))
    }

    /// Returns a condition checking if this set is a subset of the other.
    #[inline]
    pub fn is_subset<T>(self, set: T) -> Condition
    where
        SetExpression: From<T>,
    {
        Condition::Set(Box::new(SetCondition::IsSubset(
            From::from(self),
            SetExpression::from(set),
        )))
    }

    /// Returns a condition checking if this set is empty.
    #[inline]
    pub fn is_empty(self) -> Condition {
        Condition::Set(Box::new(SetCondition::IsEmpty(SetExpression::from(self))))
    }
}

impl Table1DHandle<bool> {
    /// Returns a condition referring to a value in a 1D boolean table.
    #[inline]
    pub fn element<T>(&self, x: T) -> Condition
    where
        ElementExpression: From<T>,
    {
        Condition::Table(Box::new(TableExpression::Table1D(
            self.id(),
            ElementExpression::from(x),
        )))
    }
}

impl Table2DHandle<bool> {
    /// Returns a condition referring to a value in a 2D boolean table.
    #[inline]
    pub fn element<T, U>(&self, x: T, y: U) -> Condition
    where
        ElementExpression: From<T>,
        ElementExpression: From<U>,
    {
        Condition::Table(Box::new(TableExpression::Table2D(
            self.id(),
            ElementExpression::from(x),
            ElementExpression::from(y),
        )))
    }
}

impl Table3DHandle<bool> {
    /// Returns a condition referring to a value in a 3D boolean table.
    #[inline]
    pub fn element<T, U, V>(&self, x: T, y: U, z: V) -> Condition
    where
        ElementExpression: From<T>,
        ElementExpression: From<U>,
        ElementExpression: From<V>,
    {
        Condition::Table(Box::new(TableExpression::Table3D(
            self.id(),
            ElementExpression::from(x),
            ElementExpression::from(y),
            ElementExpression::from(z),
        )))
    }
}

impl TableHandle<bool> {
    /// Returns a condition referring to a value in a boolean table.
    #[inline]
    pub fn element<T>(&self, indices: Vec<T>) -> Condition
    where
        ElementExpression: From<T>,
    {
        let indices = indices.into_iter().map(ElementExpression::from).collect();
        Condition::Table(Box::new(TableExpression::Table(self.id(), indices)))
    }
}

impl Condition {
    /// Returns the evaluation result.
    ///
    /// # Panics
    ///
    /// if the cost of the transition state is used or a min/max reduce operation is performed on an empty set or vector.
    pub fn eval<T: DPState>(&self, state: &T, registry: &TableRegistry) -> bool {
        match self {
            Self::Constant(value) => *value,
            Self::Not(condition) => !condition.eval(state, registry),
            Self::And(x, y) => x.eval(state, registry) && y.eval(state, registry),
            Self::Or(x, y) => x.eval(state, registry) || y.eval(state, registry),
            Self::ComparisonE(op, x, y) => {
                op.eval(x.eval(state, registry), y.eval(state, registry))
            }
            Self::ComparisonI(op, x, y) => {
                op.eval(x.eval(state, registry), y.eval(state, registry))
            }
            Self::ComparisonC(op, x, y) => {
                op.eval(x.eval(state, registry), y.eval(state, registry))
            }
            Self::Set(set) => set.eval(state, registry),
            Self::Table(table) => *table.eval(state, registry, &registry.bool_tables),
        }
    }

    /// Returns a simplified version by precomputation.
    ///
    /// # Panics
    ///
    /// if a min/max reduce operation is performed on an empty set or vector.
    pub fn simplify(&self, registry: &TableRegistry) -> Condition {
        match self {
            Self::Not(c) => match c.simplify(registry) {
                Self::Constant(value) => Self::Constant(!value),
                c => Self::Not(Box::new(c)),
            },
            Self::And(x, y) => match (x.simplify(registry), y.simplify(registry)) {
                (x, Self::Constant(true)) => x,
                (Self::Constant(true), y) => y,
                (Self::Constant(false), _) | (_, Self::Constant(false)) => Self::Constant(false),
                (x, y) if x == y => x,
                (x, y) => Self::And(Box::new(x), Box::new(y)),
            },
            Self::Or(x, y) => match (x.simplify(registry), y.simplify(registry)) {
                (x, Self::Constant(false)) => x,
                (Self::Constant(false), y) => y,
                (Self::Constant(true), _) | (_, Self::Constant(true)) => Self::Constant(true),
                (x, y) if x == y => x,
                (x, y) => Self::Or(Box::new(x), Box::new(y)),
            },
            Self::ComparisonE(op, x, y) => match (op, x.simplify(registry), y.simplify(registry)) {
                (op, ElementExpression::Constant(x), ElementExpression::Constant(y)) => {
                    Self::Constant(op.eval(x, y))
                }
                (
                    ComparisonOperator::Eq,
                    ElementExpression::Variable(x),
                    ElementExpression::Variable(y),
                )
                | (
                    ComparisonOperator::Ge,
                    ElementExpression::Variable(x),
                    ElementExpression::Variable(y),
                )
                | (
                    ComparisonOperator::Le,
                    ElementExpression::Variable(x),
                    ElementExpression::Variable(y),
                )
                | (
                    ComparisonOperator::Eq,
                    ElementExpression::ResourceVariable(x),
                    ElementExpression::ResourceVariable(y),
                )
                | (
                    ComparisonOperator::Ge,
                    ElementExpression::ResourceVariable(x),
                    ElementExpression::ResourceVariable(y),
                )
                | (
                    ComparisonOperator::Le,
                    ElementExpression::ResourceVariable(x),
                    ElementExpression::ResourceVariable(y),
                ) if x == y => Self::Constant(true),
                (op, x, y) => Self::ComparisonE(op.clone(), Box::new(x), Box::new(y)),
            },
            Self::ComparisonI(op, x, y) => match (op, x.simplify(registry), y.simplify(registry)) {
                (op, IntegerExpression::Constant(x), IntegerExpression::Constant(y)) => {
                    Self::Constant(op.eval(x, y))
                }
                (
                    ComparisonOperator::Eq,
                    IntegerExpression::Variable(x),
                    IntegerExpression::Variable(y),
                )
                | (
                    ComparisonOperator::Ge,
                    IntegerExpression::Variable(x),
                    IntegerExpression::Variable(y),
                )
                | (
                    ComparisonOperator::Le,
                    IntegerExpression::Variable(x),
                    IntegerExpression::Variable(y),
                )
                | (
                    ComparisonOperator::Eq,
                    IntegerExpression::ResourceVariable(x),
                    IntegerExpression::ResourceVariable(y),
                )
                | (
                    ComparisonOperator::Ge,
                    IntegerExpression::ResourceVariable(x),
                    IntegerExpression::ResourceVariable(y),
                )
                | (
                    ComparisonOperator::Le,
                    IntegerExpression::ResourceVariable(x),
                    IntegerExpression::ResourceVariable(y),
                ) if x == y => Self::Constant(true),
                (op, x, y) => Self::ComparisonI(op.clone(), Box::new(x), Box::new(y)),
            },
            Self::ComparisonC(op, x, y) => match (op, x.simplify(registry), y.simplify(registry)) {
                (op, ContinuousExpression::Constant(x), ContinuousExpression::Constant(y)) => {
                    Self::Constant(op.eval(x, y))
                }
                (
                    ComparisonOperator::Eq,
                    ContinuousExpression::Variable(x),
                    ContinuousExpression::Variable(y),
                )
                | (
                    ComparisonOperator::Ge,
                    ContinuousExpression::Variable(x),
                    ContinuousExpression::Variable(y),
                )
                | (
                    ComparisonOperator::Le,
                    ContinuousExpression::Variable(x),
                    ContinuousExpression::Variable(y),
                )
                | (
                    ComparisonOperator::Eq,
                    ContinuousExpression::ResourceVariable(x),
                    ContinuousExpression::ResourceVariable(y),
                )
                | (
                    ComparisonOperator::Ge,
                    ContinuousExpression::ResourceVariable(x),
                    ContinuousExpression::ResourceVariable(y),
                )
                | (
                    ComparisonOperator::Le,
                    ContinuousExpression::ResourceVariable(x),
                    ContinuousExpression::ResourceVariable(y),
                ) if x == y => Self::Constant(true),
                (op, x, y) => Self::ComparisonC(op.clone(), Box::new(x), Box::new(y)),
            },
            Self::Set(condition) => match condition.simplify(registry) {
                set_condition::SetCondition::Constant(value) => Self::Constant(value),
                condition => Self::Set(Box::new(condition)),
            },
            Self::Table(condition) => match condition.simplify(registry, &registry.bool_tables) {
                TableExpression::Constant(value) => Self::Constant(value),
                condition => Self::Table(Box::new(condition)),
            },
            _ => self.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::reference_expression::ReferenceExpression;
    use super::*;
    use crate::state::*;
    use crate::table_data::TableInterface;
    use crate::variable_type::Set;
    use rustc_hash::FxHashMap;

    #[test]
    fn comparison_op_eval() {
        let op = ComparisonOperator::Eq;
        assert!(op.eval(1, 1));
        assert!(!op.eval(1, 2));

        let op = ComparisonOperator::Ne;
        assert!(!op.eval(1, 1));
        assert!(op.eval(1, 2));

        let op = ComparisonOperator::Ge;
        assert!(op.eval(1, 1));
        assert!(op.eval(2, 1));
        assert!(!op.eval(1, 2));

        let op = ComparisonOperator::Gt;
        assert!(!op.eval(1, 1));
        assert!(op.eval(2, 1));
        assert!(!op.eval(1, 2));

        let op = ComparisonOperator::Le;
        assert!(op.eval(1, 1));
        assert!(!op.eval(2, 1));
        assert!(op.eval(1, 2));

        let op = ComparisonOperator::Lt;
        assert!(!op.eval(1, 1));
        assert!(!op.eval(2, 1));
        assert!(op.eval(1, 2));
    }

    #[test]
    fn default() {
        let condition = Condition::default();
        assert_eq!(condition, Condition::Constant(false));
    }

    #[test]
    fn not() {
        let condition = Condition::Constant(false);
        assert_eq!(
            !condition,
            Condition::Not(Box::new(Condition::Constant(false)))
        )
    }

    #[test]
    fn bitand() {
        let condition1 = Condition::Constant(false);
        let condition2 = Condition::Constant(true);
        assert_eq!(
            condition1.clone() & condition2.clone(),
            Condition::And(Box::new(condition1), Box::new(condition2))
        );
    }

    #[test]
    fn bitor() {
        let condition1 = Condition::Constant(false);
        let condition2 = Condition::Constant(true);
        assert_eq!(
            condition1.clone() | condition2.clone(),
            Condition::Or(Box::new(condition1), Box::new(condition2))
        );
    }

    #[test]
    fn comparison_e() {
        assert_eq!(
            Condition::comparison_e(ComparisonOperator::Eq, 0, 1),
            Condition::ComparisonE(
                ComparisonOperator::Eq,
                Box::new(ElementExpression::Constant(0)),
                Box::new(ElementExpression::Constant(1))
            )
        );
    }

    #[test]
    fn comparison_i() {
        assert_eq!(
            Condition::comparison_i(ComparisonOperator::Eq, 0, 1),
            Condition::ComparisonI(
                ComparisonOperator::Eq,
                Box::new(IntegerExpression::Constant(0)),
                Box::new(IntegerExpression::Constant(1))
            )
        );
    }

    #[test]
    fn comparison_c() {
        assert_eq!(
            Condition::comparison_c(ComparisonOperator::Eq, 0, 1),
            Condition::ComparisonC(
                ComparisonOperator::Eq,
                Box::new(ContinuousExpression::Constant(0.0)),
                Box::new(ContinuousExpression::Constant(1.0))
            )
        );
    }

    #[test]
    fn contains() {
        let expression = SetExpression::Reference(ReferenceExpression::Constant(Set::default()));
        assert_eq!(
            expression.contains(0),
            Condition::Set(Box::new(SetCondition::IsIn(
                ElementExpression::Constant(0),
                SetExpression::Reference(ReferenceExpression::Constant(Set::default()))
            )))
        );

        let mut metadata = StateMetadata::default();
        let ob = metadata.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = metadata.add_set_variable(String::from("sv"), ob);
        assert!(v.is_ok());
        let v = v.unwrap();
        assert_eq!(
            v.contains(0),
            Condition::Set(Box::new(SetCondition::IsIn(
                ElementExpression::Constant(0),
                SetExpression::Reference(ReferenceExpression::Variable(v.id()))
            )))
        );
    }

    #[test]
    fn is_subset() {
        let expression = SetExpression::Reference(ReferenceExpression::Constant(Set::default()));

        assert_eq!(
            expression.is_subset(Set::default()),
            Condition::Set(Box::new(SetCondition::IsSubset(
                SetExpression::Reference(ReferenceExpression::Constant(Set::default())),
                SetExpression::Reference(ReferenceExpression::Constant(Set::default()))
            )))
        );

        let mut metadata = StateMetadata::default();
        let ob = metadata.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = metadata.add_set_variable(String::from("sv"), ob);
        assert!(v.is_ok());
        let v = v.unwrap();
        assert_eq!(
            v.is_subset(Set::default()),
            Condition::Set(Box::new(SetCondition::IsSubset(
                SetExpression::Reference(ReferenceExpression::Variable(v.id())),
                SetExpression::Reference(ReferenceExpression::Constant(Set::default()))
            )))
        );
    }

    #[test]
    fn is_empty() {
        let expression = SetExpression::Reference(ReferenceExpression::Constant(Set::default()));

        assert_eq!(
            expression.is_empty(),
            Condition::Set(Box::new(SetCondition::IsEmpty(SetExpression::Reference(
                ReferenceExpression::Constant(Set::default())
            ))))
        );

        let mut metadata = StateMetadata::default();
        let ob = metadata.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = metadata.add_set_variable(String::from("sv"), ob);
        assert!(v.is_ok());
        let v = v.unwrap();
        assert_eq!(
            v.is_empty(),
            Condition::Set(Box::new(SetCondition::IsEmpty(SetExpression::Reference(
                ReferenceExpression::Variable(v.id())
            ),)))
        );
    }

    #[test]
    fn table_element() {
        let mut registry = TableRegistry::default();

        let t = registry.add_table_1d(String::from("t"), vec![false, true]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(
            Table1DHandle::<bool>::element(&t, 0),
            Condition::Table(Box::new(TableExpression::Table1D(
                t.id(),
                ElementExpression::Constant(0)
            )))
        );

        let t = registry.add_table_2d(String::from("t"), vec![vec![false, true]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(
            Table2DHandle::<bool>::element(&t, 0, 0),
            Condition::Table(Box::new(TableExpression::Table2D(
                t.id(),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
            )))
        );

        let t = registry.add_table_3d(String::from("t"), vec![vec![vec![false, true]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(
            Table3DHandle::<bool>::element(&t, 0, 0, 0),
            Condition::Table(Box::new(TableExpression::Table3D(
                t.id(),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
            )))
        );

        let t = registry.add_table(String::from("t"), FxHashMap::default(), false);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(
            TableHandle::<bool>::element(&t, vec![0, 0, 0, 0]),
            Condition::Table(Box::new(TableExpression::Table(
                t.id(),
                vec![
                    ElementExpression::Constant(0),
                    ElementExpression::Constant(0),
                    ElementExpression::Constant(0),
                    ElementExpression::Constant(0)
                ],
            )))
        );
    }

    #[test]
    fn constant_eval() {
        let state = State::default();
        let registry = TableRegistry::default();
        let expression = Condition::Constant(true);
        assert!(expression.eval(&state, &registry));
        let expression = Condition::Constant(false);
        assert!(!expression.eval(&state, &registry));
    }

    #[test]
    fn not_eval() {
        let state = State::default();
        let registry = TableRegistry::default();
        let expression = Condition::Not(Box::new(Condition::Constant(false)));
        assert!(expression.eval(&state, &registry));
        let expression = Condition::Not(Box::new(Condition::Constant(true)));
        assert!(!expression.eval(&state, &registry));
    }

    #[test]
    fn and_eval() {
        let state = State::default();
        let registry = TableRegistry::default();
        let expression = Condition::And(
            Box::new(Condition::Constant(true)),
            Box::new(Condition::Constant(true)),
        );
        assert!(expression.eval(&state, &registry));
        let expression = Condition::And(
            Box::new(Condition::Constant(true)),
            Box::new(Condition::Constant(false)),
        );
        assert!(!expression.eval(&state, &registry));
        let expression = Condition::And(
            Box::new(Condition::Constant(false)),
            Box::new(Condition::Constant(true)),
        );
        assert!(!expression.eval(&state, &registry));
        let expression = Condition::And(
            Box::new(Condition::Constant(false)),
            Box::new(Condition::Constant(false)),
        );
        assert!(!expression.eval(&state, &registry));
    }

    #[test]
    fn or_eval() {
        let state = State::default();
        let registry = TableRegistry::default();
        let expression = Condition::Or(
            Box::new(Condition::Constant(true)),
            Box::new(Condition::Constant(true)),
        );
        assert!(expression.eval(&state, &registry));
        let expression = Condition::Or(
            Box::new(Condition::Constant(true)),
            Box::new(Condition::Constant(false)),
        );
        assert!(expression.eval(&state, &registry));
        let expression = Condition::Or(
            Box::new(Condition::Constant(false)),
            Box::new(Condition::Constant(true)),
        );
        assert!(expression.eval(&state, &registry));
        let expression = Condition::Or(
            Box::new(Condition::Constant(false)),
            Box::new(Condition::Constant(false)),
        );
        assert!(!expression.eval(&state, &registry));
    }

    #[test]
    fn comparison_e_eval() {
        let state = State::default();
        let registry = TableRegistry::default();
        let expression = Condition::ComparisonE(
            ComparisonOperator::Eq,
            Box::new(ElementExpression::Constant(0)),
            Box::new(ElementExpression::Constant(0)),
        );
        assert!(expression.eval(&state, &registry));
    }

    #[test]
    fn comparison_i_eval() {
        let state = State::default();
        let registry = TableRegistry::default();
        let expression = Condition::ComparisonI(
            ComparisonOperator::Eq,
            Box::new(IntegerExpression::Constant(0)),
            Box::new(IntegerExpression::Constant(0)),
        );
        assert!(expression.eval(&state, &registry));
    }

    #[test]
    fn comparison_c_eval() {
        let state = State::default();
        let registry = TableRegistry::default();
        let expression = Condition::ComparisonC(
            ComparisonOperator::Eq,
            Box::new(ContinuousExpression::Constant(0.0)),
            Box::new(ContinuousExpression::Constant(0.0)),
        );
        assert!(expression.eval(&state, &registry));
    }

    #[test]
    fn set_eval() {
        let state = State::default();
        let registry = TableRegistry::default();
        let expression = Condition::Set(Box::new(SetCondition::Constant(true)));
        assert!(expression.eval(&state, &registry));
    }

    #[test]
    fn table_eval() {
        let state = State::default();
        let registry = TableRegistry::default();
        let expression = Condition::Table(Box::new(TableExpression::Constant(true)));
        assert!(expression.eval(&state, &registry));
    }

    #[test]
    fn constant_simplify() {
        let registry = TableRegistry::default();
        let expression = Condition::Constant(true);
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn not_simplify() {
        let registry = TableRegistry::default();
        let expression = Condition::Not(Box::new(Condition::Constant(false)));
        assert_eq!(expression.simplify(&registry), Condition::Constant(true));
        let expression = Condition::Not(Box::new(Condition::Table(Box::new(
            TableExpression::Table1D(0, ElementExpression::Variable(0)),
        ))));
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn and_simplify() {
        let registry = TableRegistry::default();
        let expression = Condition::And(
            Box::new(Condition::Constant(true)),
            Box::new(Condition::Constant(true)),
        );
        assert_eq!(expression.simplify(&registry), Condition::Constant(true));
        let expression = Condition::And(
            Box::new(Condition::Constant(true)),
            Box::new(Condition::Constant(false)),
        );
        assert_eq!(expression.simplify(&registry), Condition::Constant(false));
        let expression = Condition::And(
            Box::new(Condition::Constant(false)),
            Box::new(Condition::Constant(true)),
        );
        assert_eq!(expression.simplify(&registry), Condition::Constant(false));
        let expression = Condition::And(
            Box::new(Condition::Constant(false)),
            Box::new(Condition::Constant(false)),
        );
        assert_eq!(expression.simplify(&registry), Condition::Constant(false));
        let expression = Condition::And(
            Box::new(Condition::Constant(false)),
            Box::new(Condition::Table(Box::new(TableExpression::Table1D(
                0,
                ElementExpression::Variable(0),
            )))),
        );
        assert_eq!(expression.simplify(&registry), Condition::Constant(false));
        let expression = Condition::And(
            Box::new(Condition::Table(Box::new(TableExpression::Table1D(
                0,
                ElementExpression::Variable(0),
            )))),
            Box::new(Condition::Constant(false)),
        );
        assert_eq!(expression.simplify(&registry), Condition::Constant(false));
        let expression = Condition::And(
            Box::new(Condition::Constant(true)),
            Box::new(Condition::Table(Box::new(TableExpression::Table1D(
                0,
                ElementExpression::Variable(0),
            )))),
        );
        assert_eq!(
            expression.simplify(&registry),
            Condition::Table(Box::new(TableExpression::Table1D(
                0,
                ElementExpression::Variable(0),
            ))),
        );
        let expression = Condition::And(
            Box::new(Condition::Table(Box::new(TableExpression::Table1D(
                0,
                ElementExpression::Variable(0),
            )))),
            Box::new(Condition::Constant(true)),
        );
        assert_eq!(
            expression.simplify(&registry),
            Condition::Table(Box::new(TableExpression::Table1D(
                0,
                ElementExpression::Variable(0),
            ))),
        );
        let expression = Condition::And(
            Box::new(Condition::Table(Box::new(TableExpression::Table1D(
                0,
                ElementExpression::Variable(0),
            )))),
            Box::new(Condition::Table(Box::new(TableExpression::Table1D(
                0,
                ElementExpression::Variable(0),
            )))),
        );
        assert_eq!(
            expression.simplify(&registry),
            Condition::Table(Box::new(TableExpression::Table1D(
                0,
                ElementExpression::Variable(0),
            ))),
        );
        let expression = Condition::And(
            Box::new(Condition::Table(Box::new(TableExpression::Table1D(
                0,
                ElementExpression::Variable(0),
            )))),
            Box::new(Condition::Table(Box::new(TableExpression::Table1D(
                1,
                ElementExpression::Variable(0),
            )))),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn or_simplify() {
        let registry = TableRegistry::default();
        let expression = Condition::Or(
            Box::new(Condition::Constant(true)),
            Box::new(Condition::Constant(true)),
        );
        assert_eq!(expression.simplify(&registry), Condition::Constant(true));
        let expression = Condition::Or(
            Box::new(Condition::Constant(true)),
            Box::new(Condition::Constant(false)),
        );
        assert_eq!(expression.simplify(&registry), Condition::Constant(true));
        let expression = Condition::Or(
            Box::new(Condition::Constant(false)),
            Box::new(Condition::Constant(true)),
        );
        assert_eq!(expression.simplify(&registry), Condition::Constant(true));
        let expression = Condition::Or(
            Box::new(Condition::Constant(false)),
            Box::new(Condition::Constant(false)),
        );
        assert_eq!(expression.simplify(&registry), Condition::Constant(false));
        let expression = Condition::Or(
            Box::new(Condition::Constant(false)),
            Box::new(Condition::Table(Box::new(TableExpression::Table1D(
                0,
                ElementExpression::Variable(0),
            )))),
        );
        assert_eq!(
            expression.simplify(&registry),
            Condition::Table(Box::new(TableExpression::Table1D(
                0,
                ElementExpression::Variable(0),
            ))),
        );
        let expression = Condition::Or(
            Box::new(Condition::Table(Box::new(TableExpression::Table1D(
                0,
                ElementExpression::Variable(0),
            )))),
            Box::new(Condition::Constant(false)),
        );
        assert_eq!(
            expression.simplify(&registry),
            Condition::Table(Box::new(TableExpression::Table1D(
                0,
                ElementExpression::Variable(0),
            ))),
        );
        let expression = Condition::Or(
            Box::new(Condition::Constant(true)),
            Box::new(Condition::Table(Box::new(TableExpression::Table1D(
                0,
                ElementExpression::Variable(0),
            )))),
        );
        assert_eq!(expression.simplify(&registry), Condition::Constant(true));
        let expression = Condition::Or(
            Box::new(Condition::Table(Box::new(TableExpression::Table1D(
                0,
                ElementExpression::Variable(0),
            )))),
            Box::new(Condition::Constant(true)),
        );
        assert_eq!(expression.simplify(&registry), Condition::Constant(true));
        let expression = Condition::Or(
            Box::new(Condition::Table(Box::new(TableExpression::Table1D(
                0,
                ElementExpression::Variable(0),
            )))),
            Box::new(Condition::Table(Box::new(TableExpression::Table1D(
                0,
                ElementExpression::Variable(0),
            )))),
        );
        assert_eq!(
            expression.simplify(&registry),
            Condition::Table(Box::new(TableExpression::Table1D(
                0,
                ElementExpression::Variable(0),
            ))),
        );
        let expression = Condition::Or(
            Box::new(Condition::Table(Box::new(TableExpression::Table1D(
                0,
                ElementExpression::Variable(0),
            )))),
            Box::new(Condition::Table(Box::new(TableExpression::Table1D(
                1,
                ElementExpression::Variable(0),
            )))),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn comparison_e_simplify() {
        let registry = TableRegistry::default();
        let expression = Condition::ComparisonE(
            ComparisonOperator::Lt,
            Box::new(ElementExpression::Constant(0)),
            Box::new(ElementExpression::Constant(1)),
        );
        assert_eq!(expression.simplify(&registry), Condition::Constant(true));
        let expression = Condition::ComparisonE(
            ComparisonOperator::Eq,
            Box::new(ElementExpression::Variable(0)),
            Box::new(ElementExpression::Variable(0)),
        );
        assert_eq!(expression.simplify(&registry), Condition::Constant(true));
        let expression = Condition::ComparisonE(
            ComparisonOperator::Ge,
            Box::new(ElementExpression::Variable(0)),
            Box::new(ElementExpression::Variable(0)),
        );
        assert_eq!(expression.simplify(&registry), Condition::Constant(true));
        let expression = Condition::ComparisonE(
            ComparisonOperator::Le,
            Box::new(ElementExpression::Variable(0)),
            Box::new(ElementExpression::Variable(0)),
        );
        assert_eq!(expression.simplify(&registry), Condition::Constant(true));
        let expression = Condition::ComparisonE(
            ComparisonOperator::Eq,
            Box::new(ElementExpression::ResourceVariable(0)),
            Box::new(ElementExpression::ResourceVariable(0)),
        );
        assert_eq!(expression.simplify(&registry), Condition::Constant(true));
        let expression = Condition::ComparisonE(
            ComparisonOperator::Ge,
            Box::new(ElementExpression::ResourceVariable(0)),
            Box::new(ElementExpression::ResourceVariable(0)),
        );
        assert_eq!(expression.simplify(&registry), Condition::Constant(true));
        let expression = Condition::ComparisonE(
            ComparisonOperator::Le,
            Box::new(ElementExpression::ResourceVariable(0)),
            Box::new(ElementExpression::ResourceVariable(0)),
        );
        assert_eq!(expression.simplify(&registry), Condition::Constant(true));
        let expression = Condition::ComparisonE(
            ComparisonOperator::Le,
            Box::new(ElementExpression::ResourceVariable(0)),
            Box::new(ElementExpression::ResourceVariable(1)),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn comparison_i_simplify() {
        let registry = TableRegistry::default();
        let expression = Condition::ComparisonI(
            ComparisonOperator::Lt,
            Box::new(IntegerExpression::Constant(0)),
            Box::new(IntegerExpression::Constant(1)),
        );
        assert_eq!(expression.simplify(&registry), Condition::Constant(true));
        let expression = Condition::ComparisonI(
            ComparisonOperator::Eq,
            Box::new(IntegerExpression::Variable(0)),
            Box::new(IntegerExpression::Variable(0)),
        );
        assert_eq!(expression.simplify(&registry), Condition::Constant(true));
        let expression = Condition::ComparisonI(
            ComparisonOperator::Ge,
            Box::new(IntegerExpression::Variable(0)),
            Box::new(IntegerExpression::Variable(0)),
        );
        assert_eq!(expression.simplify(&registry), Condition::Constant(true));
        let expression = Condition::ComparisonI(
            ComparisonOperator::Le,
            Box::new(IntegerExpression::Variable(0)),
            Box::new(IntegerExpression::Variable(0)),
        );
        assert_eq!(expression.simplify(&registry), Condition::Constant(true));
        let expression = Condition::ComparisonI(
            ComparisonOperator::Eq,
            Box::new(IntegerExpression::ResourceVariable(0)),
            Box::new(IntegerExpression::ResourceVariable(0)),
        );
        assert_eq!(expression.simplify(&registry), Condition::Constant(true));
        let expression = Condition::ComparisonI(
            ComparisonOperator::Ge,
            Box::new(IntegerExpression::ResourceVariable(0)),
            Box::new(IntegerExpression::ResourceVariable(0)),
        );
        assert_eq!(expression.simplify(&registry), Condition::Constant(true));
        let expression = Condition::ComparisonI(
            ComparisonOperator::Le,
            Box::new(IntegerExpression::ResourceVariable(0)),
            Box::new(IntegerExpression::ResourceVariable(0)),
        );
        assert_eq!(expression.simplify(&registry), Condition::Constant(true));
        let expression = Condition::ComparisonI(
            ComparisonOperator::Le,
            Box::new(IntegerExpression::ResourceVariable(0)),
            Box::new(IntegerExpression::ResourceVariable(1)),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn comparison_c_simplify() {
        let registry = TableRegistry::default();
        let expression = Condition::ComparisonC(
            ComparisonOperator::Lt,
            Box::new(ContinuousExpression::Constant(0.0)),
            Box::new(ContinuousExpression::Constant(1.0)),
        );
        assert_eq!(expression.simplify(&registry), Condition::Constant(true));
        let expression = Condition::ComparisonC(
            ComparisonOperator::Eq,
            Box::new(ContinuousExpression::Variable(0)),
            Box::new(ContinuousExpression::Variable(0)),
        );
        assert_eq!(expression.simplify(&registry), Condition::Constant(true));
        let expression = Condition::ComparisonC(
            ComparisonOperator::Ge,
            Box::new(ContinuousExpression::Variable(0)),
            Box::new(ContinuousExpression::Variable(0)),
        );
        assert_eq!(expression.simplify(&registry), Condition::Constant(true));
        let expression = Condition::ComparisonC(
            ComparisonOperator::Le,
            Box::new(ContinuousExpression::Variable(0)),
            Box::new(ContinuousExpression::Variable(0)),
        );
        assert_eq!(expression.simplify(&registry), Condition::Constant(true));
        let expression = Condition::ComparisonC(
            ComparisonOperator::Eq,
            Box::new(ContinuousExpression::ResourceVariable(0)),
            Box::new(ContinuousExpression::ResourceVariable(0)),
        );
        assert_eq!(expression.simplify(&registry), Condition::Constant(true));
        let expression = Condition::ComparisonC(
            ComparisonOperator::Ge,
            Box::new(ContinuousExpression::ResourceVariable(0)),
            Box::new(ContinuousExpression::ResourceVariable(0)),
        );
        assert_eq!(expression.simplify(&registry), Condition::Constant(true));
        let expression = Condition::ComparisonC(
            ComparisonOperator::Le,
            Box::new(ContinuousExpression::ResourceVariable(0)),
            Box::new(ContinuousExpression::ResourceVariable(0)),
        );
        assert_eq!(expression.simplify(&registry), Condition::Constant(true));
        let expression = Condition::ComparisonC(
            ComparisonOperator::Le,
            Box::new(ContinuousExpression::ResourceVariable(0)),
            Box::new(ContinuousExpression::ResourceVariable(1)),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn set_simplify() {
        let registry = TableRegistry::default();
        let expression = Condition::Set(Box::new(SetCondition::Constant(true)));
        assert_eq!(expression.simplify(&registry), Condition::Constant(true));
    }

    #[test]
    fn table_simplify() {
        let registry = TableRegistry::default();
        let expression = Condition::Table(Box::new(TableExpression::Constant(true)));
        assert_eq!(expression.simplify(&registry), Condition::Constant(true));
    }
}
