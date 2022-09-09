use super::condition::{Condition, IfThenElse};
use super::numeric_operator::{BinaryOperator, MaxMin};
use super::reference_expression::ReferenceExpression;
use super::table_expression::TableExpression;
use super::vector_expression::VectorExpression;
use crate::state::{DPState, ElementResourceVariable, ElementVariable};
use crate::table_data::{Table1DHandle, Table2DHandle, Table3DHandle, TableHandle};
use crate::table_registry::TableRegistry;
use crate::variable_type::Element;
use std::ops;

/// Element expression .
#[derive(Debug, PartialEq, Clone)]
pub enum ElementExpression {
    /// Constant.
    Constant(Element),
    /// Variable index.
    Variable(usize),
    /// Resource variable index.
    ResourceVariable(usize),
    /// Binary arithmetic operation.
    BinaryOperation(
        BinaryOperator,
        Box<ElementExpression>,
        Box<ElementExpression>,
    ),
    /// The last value of a vector expression.
    Last(Box<VectorExpression>),
    /// An item in a vector expression.
    At(Box<VectorExpression>, Box<ElementExpression>),
    /// A constant in a element table.
    Table(Box<TableExpression<Element>>),
    /// If-then-else expression, which returns the first one if the condition holds and the second one otherwise.
    If(
        Box<Condition>,
        Box<ElementExpression>,
        Box<ElementExpression>,
    ),
}

impl Default for ElementExpression {
    #[inline]
    fn default() -> Self {
        Self::Constant(0)
    }
}

impl From<Element> for ElementExpression {
    #[inline]
    fn from(e: Element) -> ElementExpression {
        ElementExpression::Constant(e)
    }
}

impl From<ElementVariable> for ElementExpression {
    #[inline]
    fn from(v: ElementVariable) -> ElementExpression {
        ElementExpression::Variable(v.id())
    }
}

impl From<ElementResourceVariable> for ElementExpression {
    #[inline]
    fn from(v: ElementResourceVariable) -> ElementExpression {
        ElementExpression::ResourceVariable(v.id())
    }
}

impl ops::Add for ElementExpression {
    type Output = ElementExpression;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        ElementExpression::BinaryOperation(BinaryOperator::Add, Box::new(self), Box::new(rhs))
    }
}

impl ops::Sub for ElementExpression {
    type Output = ElementExpression;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        ElementExpression::BinaryOperation(BinaryOperator::Sub, Box::new(self), Box::new(rhs))
    }
}

impl ops::Mul for ElementExpression {
    type Output = ElementExpression;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        ElementExpression::BinaryOperation(BinaryOperator::Mul, Box::new(self), Box::new(rhs))
    }
}

impl ops::Div for ElementExpression {
    type Output = ElementExpression;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        ElementExpression::BinaryOperation(BinaryOperator::Div, Box::new(self), Box::new(rhs))
    }
}

impl ops::Rem for ElementExpression {
    type Output = ElementExpression;

    #[inline]
    fn rem(self, rhs: Self) -> Self::Output {
        ElementExpression::BinaryOperation(BinaryOperator::Rem, Box::new(self), Box::new(rhs))
    }
}

impl MaxMin for ElementExpression {
    type Output = ElementExpression;

    #[inline]
    fn max(self, rhs: Self) -> Self::Output {
        ElementExpression::BinaryOperation(BinaryOperator::Max, Box::new(self), Box::new(rhs))
    }

    #[inline]
    fn min(self, rhs: Self) -> Self::Output {
        ElementExpression::BinaryOperation(BinaryOperator::Min, Box::new(self), Box::new(rhs))
    }
}

impl Table1DHandle<Element> {
    /// Returns a constant in a 1D element table.
    #[inline]
    pub fn element<T>(&self, x: T) -> ElementExpression
    where
        ElementExpression: From<T>,
    {
        ElementExpression::Table(Box::new(TableExpression::Table1D(
            self.id(),
            ElementExpression::from(x),
        )))
    }
}

impl Table2DHandle<Element> {
    /// Returns a constant in a 2D element table.
    #[inline]
    pub fn element<T, U>(&self, x: T, y: U) -> ElementExpression
    where
        ElementExpression: From<T>,
        ElementExpression: From<U>,
    {
        ElementExpression::Table(Box::new(TableExpression::Table2D(
            self.id(),
            ElementExpression::from(x),
            ElementExpression::from(y),
        )))
    }
}

impl Table3DHandle<Element> {
    /// Returns a constant in a 3D element table.
    #[inline]
    pub fn element<T, U, V>(&self, x: T, y: U, z: V) -> ElementExpression
    where
        ElementExpression: From<T>,
        ElementExpression: From<U>,
        ElementExpression: From<V>,
    {
        ElementExpression::Table(Box::new(TableExpression::Table3D(
            self.id(),
            ElementExpression::from(x),
            ElementExpression::from(y),
            ElementExpression::from(z),
        )))
    }
}

impl TableHandle<Element> {
    /// Returns a constant in an element table.
    #[inline]
    pub fn element<T>(&self, indices: Vec<T>) -> ElementExpression
    where
        ElementExpression: From<T>,
    {
        let indices = indices.into_iter().map(ElementExpression::from).collect();
        ElementExpression::Table(Box::new(TableExpression::Table(self.id(), indices)))
    }
}

impl IfThenElse<ElementExpression> for Condition {
    #[inline]
    fn if_then_else<U, V>(self, lhs: U, rhs: V) -> ElementExpression
    where
        ElementExpression: From<U> + From<V>,
    {
        ElementExpression::If(
            Box::new(self),
            Box::new(ElementExpression::from(lhs)),
            Box::new(ElementExpression::from(rhs)),
        )
    }
}

macro_rules! impl_binary_ops {
    ($T:ty,$U:ty) => {
        impl ops::Add<$U> for $T {
            type Output = ElementExpression;

            #[inline]
            fn add(self, rhs: $U) -> ElementExpression {
                ElementExpression::from(self) + ElementExpression::from(rhs)
            }
        }

        impl ops::Sub<$U> for $T {
            type Output = ElementExpression;

            #[inline]
            fn sub(self, rhs: $U) -> ElementExpression {
                ElementExpression::from(self) - ElementExpression::from(rhs)
            }
        }

        impl ops::Mul<$U> for $T {
            type Output = ElementExpression;

            #[inline]
            fn mul(self, rhs: $U) -> ElementExpression {
                ElementExpression::from(self) * ElementExpression::from(rhs)
            }
        }

        impl ops::Div<$U> for $T {
            type Output = ElementExpression;

            #[inline]
            fn div(self, rhs: $U) -> ElementExpression {
                ElementExpression::from(self) / ElementExpression::from(rhs)
            }
        }

        impl ops::Rem<$U> for $T {
            type Output = ElementExpression;

            #[inline]
            fn rem(self, rhs: $U) -> ElementExpression {
                ElementExpression::from(self) % ElementExpression::from(rhs)
            }
        }

        impl MaxMin<$U> for $T {
            type Output = ElementExpression;

            #[inline]
            fn max(self, rhs: $U) -> ElementExpression {
                ElementExpression::from(self).max(ElementExpression::from(rhs))
            }

            #[inline]
            fn min(self, rhs: $U) -> ElementExpression {
                ElementExpression::from(self).min(ElementExpression::from(rhs))
            }
        }
    };
}

impl_binary_ops!(ElementExpression, Element);
impl_binary_ops!(ElementExpression, ElementVariable);
impl_binary_ops!(ElementExpression, ElementResourceVariable);
impl_binary_ops!(Element, ElementExpression);
impl_binary_ops!(Element, ElementVariable);
impl_binary_ops!(Element, ElementResourceVariable);
impl_binary_ops!(ElementVariable, ElementExpression);
impl_binary_ops!(ElementVariable, Element);
impl_binary_ops!(ElementVariable, ElementVariable);
impl_binary_ops!(ElementVariable, ElementResourceVariable);
impl_binary_ops!(ElementResourceVariable, ElementExpression);
impl_binary_ops!(ElementResourceVariable, Element);
impl_binary_ops!(ElementResourceVariable, ElementVariable);
impl_binary_ops!(ElementResourceVariable, ElementResourceVariable);

impl ElementExpression {
    /// Returns the evaluation result.
    ///
    /// # Panics
    ///
    /// if the cost of the transition state is used or a min/max reduce operation is performed on an empty set or vector.
    pub fn eval<T: DPState>(&self, state: &T, registry: &TableRegistry) -> Element {
        match self {
            Self::Constant(x) => *x,
            Self::Variable(i) => state.get_element_variable(*i),
            Self::ResourceVariable(i) => state.get_element_resource_variable(*i),
            Self::BinaryOperation(op, x, y) => {
                op.eval(x.eval(state, registry), y.eval(state, registry))
            }
            Self::Last(vector) => match vector.as_ref() {
                VectorExpression::Reference(vector) => {
                    let f = |i| state.get_vector_variable(i);
                    *vector
                        .eval(state, registry, &f, &registry.vector_tables)
                        .last()
                        .unwrap()
                }
                vector => *vector.eval(state, registry).last().unwrap(),
            },
            Self::At(vector, i) => match vector.as_ref() {
                VectorExpression::Reference(vector) => {
                    let f = |i| state.get_vector_variable(i);
                    vector.eval(state, registry, &f, &registry.vector_tables)
                        [i.eval(state, registry)]
                }
                vector => vector.eval(state, registry)[i.eval(state, registry)],
            },
            Self::Table(table) => *table.eval(state, registry, &registry.element_tables),
            Self::If(condition, x, y) => {
                if condition.eval(state, registry) {
                    x.eval(state, registry)
                } else {
                    y.eval(state, registry)
                }
            }
        }
    }

    /// Returns a simplified version by precomputation.
    ///
    /// # Panics
    ///
    /// if a min/max reduce operation is performed on an empty set or vector.
    pub fn simplify(&self, registry: &TableRegistry) -> ElementExpression {
        match self {
            Self::Last(vector) => match vector.simplify(registry) {
                VectorExpression::Reference(ReferenceExpression::Constant(vector)) => {
                    Self::Constant(*vector.last().unwrap())
                }
                vector => Self::Last(Box::new(vector)),
            },
            Self::At(vector, i) => match (vector.simplify(registry), i.simplify(registry)) {
                (
                    VectorExpression::Reference(ReferenceExpression::Constant(vector)),
                    Self::Constant(i),
                ) => Self::Constant(vector[i]),
                (vector, i) => Self::At(Box::new(vector), Box::new(i)),
            },
            Self::BinaryOperation(op, x, y) => match (x.simplify(registry), y.simplify(registry)) {
                (Self::Constant(x), Self::Constant(y)) => Self::Constant(op.eval(x, y)),
                (x, y) => Self::BinaryOperation(op.clone(), Box::new(x), Box::new(y)),
            },
            Self::Table(table) => match table.simplify(registry, &registry.element_tables) {
                TableExpression::Constant(value) => Self::Constant(value),
                expression => Self::Table(Box::new(expression)),
            },
            Self::If(condition, x, y) => match condition.simplify(registry) {
                Condition::Constant(true) => x.simplify(registry),
                Condition::Constant(false) => y.simplify(registry),
                condition => Self::If(
                    Box::new(condition),
                    Box::new(x.simplify(registry)),
                    Box::new(y.simplify(registry)),
                ),
            },
            _ => self.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::condition::ComparisonOperator;
    use super::super::integer_expression::IntegerExpression;
    use super::*;
    use crate::state::*;
    use crate::table::*;
    use crate::table_data::{TableData, TableInterface};
    use crate::variable_type::*;
    use rustc_hash::FxHashMap;

    fn generate_registry() -> TableRegistry {
        let mut name_to_constant = FxHashMap::default();
        name_to_constant.insert(String::from("f0"), 1);

        let tables_1d = vec![Table1D::new(vec![1, 0])];
        let mut name_to_table_1d = FxHashMap::default();
        name_to_table_1d.insert(String::from("f1"), 0);

        let tables_2d = vec![Table2D::new(vec![vec![1, 0]])];
        let mut name_to_table_2d = FxHashMap::default();
        name_to_table_2d.insert(String::from("f2"), 0);

        let tables_3d = vec![Table3D::new(vec![vec![vec![1, 0]]])];
        let mut name_to_table_3d = FxHashMap::default();
        name_to_table_3d.insert(String::from("f3"), 0);

        let mut map = FxHashMap::default();
        let key = vec![0, 0, 0, 0];
        map.insert(key, 1);
        let key = vec![0, 0, 0, 1];
        map.insert(key, 0);
        let tables = vec![Table::new(map, 0)];
        let mut name_to_table = FxHashMap::default();
        name_to_table.insert(String::from("f4"), 0);

        let element_tables = TableData {
            name_to_constant,
            tables_1d,
            name_to_table_1d,
            tables_2d,
            name_to_table_2d,
            tables_3d,
            name_to_table_3d,
            tables,
            name_to_table,
        };

        let mut name_to_table_1d = FxHashMap::default();
        name_to_table_1d.insert(String::from("t1"), 0);
        let vector_tables = TableData {
            tables_1d: vec![Table1D::new(vec![vec![0, 1]])],
            name_to_table_1d,
            ..Default::default()
        };

        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(2);
        let default = Set::with_capacity(3);
        let tables_1d = vec![Table1D::new(vec![set, default.clone(), default])];
        let mut name_to_table_1d = FxHashMap::default();
        name_to_table_1d.insert(String::from("s1"), 0);
        let set_tables = TableData {
            tables_1d,
            name_to_table_1d,
            ..Default::default()
        };

        TableRegistry {
            element_tables,
            set_tables,
            vector_tables,
            ..Default::default()
        }
    }

    fn generate_state() -> State {
        let mut set1 = Set::with_capacity(3);
        set1.insert(0);
        set1.insert(2);
        let mut set2 = Set::with_capacity(3);
        set2.insert(0);
        set2.insert(1);
        State {
            signature_variables: SignatureVariables {
                set_variables: vec![set1, set2],
                vector_variables: vec![vec![0, 2]],
                element_variables: vec![1],
                ..Default::default()
            },
            resource_variables: ResourceVariables {
                element_variables: vec![2],
                ..Default::default()
            },
        }
    }

    #[test]
    fn elment_default() {
        assert_eq!(ElementExpression::default(), ElementExpression::Constant(0));
    }

    #[test]
    fn element_from() {
        assert_eq!(ElementExpression::from(1), ElementExpression::Constant(1));

        let mut metadata = StateMetadata::default();
        let ob = metadata.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let v = metadata.add_element_variable(String::from("ev"), ob);
        assert!(v.is_ok());
        let v = v.unwrap();
        assert_eq!(
            ElementExpression::from(v),
            ElementExpression::Variable(v.id())
        );

        let v = metadata.add_element_resource_variable(String::from("erv"), ob, true);
        assert!(v.is_ok());
        let v = v.unwrap();
        assert_eq!(
            ElementExpression::from(v),
            ElementExpression::ResourceVariable(v.id())
        );
    }

    #[test]
    fn element_add() {
        let expression1 = ElementExpression::Constant(0);
        let expression2 = ElementExpression::Constant(1);
        assert_eq!(
            expression1.clone() + expression2.clone(),
            ElementExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ElementExpression::Constant(0)),
                Box::new(ElementExpression::Constant(1))
            )
        );
        assert_eq!(
            expression2 + expression1,
            ElementExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ElementExpression::Constant(1)),
                Box::new(ElementExpression::Constant(0)),
            )
        );

        let expression1 = ElementExpression::Constant(0);
        let expression2 = 1;
        assert_eq!(
            expression1.clone() + expression2,
            ElementExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ElementExpression::Constant(0)),
                Box::new(ElementExpression::Constant(1))
            )
        );
        assert_eq!(
            expression2 + expression1,
            ElementExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ElementExpression::Constant(1)),
                Box::new(ElementExpression::Constant(0)),
            )
        );

        let mut metadata = StateMetadata::default();
        let ob = metadata.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let v = metadata.add_element_variable(String::from("ev"), ob);
        assert!(v.is_ok());
        let expression1 = v.unwrap();
        let expression2 = ElementExpression::Constant(0);
        assert_eq!(
            expression1 + expression2.clone(),
            ElementExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ElementExpression::Variable(expression1.id())),
                Box::new(ElementExpression::Constant(0)),
            )
        );
        assert_eq!(
            expression2 + expression1,
            ElementExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ElementExpression::Constant(0)),
                Box::new(ElementExpression::Variable(expression1.id())),
            )
        );

        let expression2 = 1;
        assert_eq!(
            expression1 + expression2,
            ElementExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ElementExpression::Variable(expression1.id())),
                Box::new(ElementExpression::Constant(1)),
            )
        );
        assert_eq!(
            expression2 + expression1,
            ElementExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ElementExpression::Constant(1)),
                Box::new(ElementExpression::Variable(expression1.id())),
            )
        );

        let v = metadata.add_element_variable(String::from("ev2"), ob);
        assert!(v.is_ok());
        let expression2 = v.unwrap();
        assert_eq!(
            expression1 + expression2,
            ElementExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ElementExpression::Variable(expression1.id())),
                Box::new(ElementExpression::Variable(expression2.id())),
            )
        );
        assert_eq!(
            expression2 + expression1,
            ElementExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ElementExpression::Variable(expression2.id())),
                Box::new(ElementExpression::Variable(expression1.id())),
            )
        );

        let v = metadata.add_element_resource_variable(String::from("erv"), ob, true);
        assert!(v.is_ok());
        let expression2 = v.unwrap();
        assert_eq!(
            expression1 + expression2,
            ElementExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ElementExpression::Variable(expression1.id())),
                Box::new(ElementExpression::ResourceVariable(expression2.id())),
            )
        );
        assert_eq!(
            expression2 + expression1,
            ElementExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ElementExpression::ResourceVariable(expression2.id())),
                Box::new(ElementExpression::Variable(expression1.id())),
            )
        );

        let expression1 = ElementExpression::Constant(0);
        assert_eq!(
            expression1.clone() + expression2,
            ElementExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ElementExpression::Constant(0)),
                Box::new(ElementExpression::ResourceVariable(expression2.id())),
            )
        );
        assert_eq!(
            expression2 + expression1,
            ElementExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ElementExpression::ResourceVariable(expression2.id())),
                Box::new(ElementExpression::Constant(0)),
            )
        );

        let expression1 = 1;
        assert_eq!(
            expression1 + expression2,
            ElementExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ElementExpression::Constant(1)),
                Box::new(ElementExpression::ResourceVariable(expression2.id())),
            )
        );
        assert_eq!(
            expression2 + expression1,
            ElementExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ElementExpression::ResourceVariable(expression2.id())),
                Box::new(ElementExpression::Constant(1)),
            )
        );

        let v = metadata.add_element_resource_variable(String::from("erv2"), ob, true);
        assert!(v.is_ok());
        let expression1 = v.unwrap();
        assert_eq!(
            expression1 + expression2,
            ElementExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ElementExpression::ResourceVariable(expression1.id())),
                Box::new(ElementExpression::ResourceVariable(expression2.id())),
            )
        );
        assert_eq!(
            expression2 + expression1,
            ElementExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ElementExpression::ResourceVariable(expression2.id())),
                Box::new(ElementExpression::ResourceVariable(expression1.id())),
            )
        );
    }

    #[test]
    fn element_sub() {
        let expression1 = ElementExpression::Constant(0);
        let expression2 = ElementExpression::Constant(1);
        assert_eq!(
            expression1.clone() - expression2.clone(),
            ElementExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ElementExpression::Constant(0)),
                Box::new(ElementExpression::Constant(1))
            )
        );
        assert_eq!(
            expression2 - expression1,
            ElementExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ElementExpression::Constant(1)),
                Box::new(ElementExpression::Constant(0)),
            )
        );

        let expression1 = ElementExpression::Constant(0);
        let expression2 = 1;
        assert_eq!(
            expression1.clone() - expression2,
            ElementExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ElementExpression::Constant(0)),
                Box::new(ElementExpression::Constant(1))
            )
        );
        assert_eq!(
            expression2 - expression1,
            ElementExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ElementExpression::Constant(1)),
                Box::new(ElementExpression::Constant(0)),
            )
        );

        let mut metadata = StateMetadata::default();
        let ob = metadata.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let v = metadata.add_element_variable(String::from("ev"), ob);
        assert!(v.is_ok());
        let expression1 = v.unwrap();
        let expression2 = ElementExpression::Constant(0);
        assert_eq!(
            expression1 - expression2.clone(),
            ElementExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ElementExpression::Variable(expression1.id())),
                Box::new(ElementExpression::Constant(0)),
            )
        );
        assert_eq!(
            expression2 - expression1,
            ElementExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ElementExpression::Constant(0)),
                Box::new(ElementExpression::Variable(expression1.id())),
            )
        );

        let expression2 = 1;
        assert_eq!(
            expression1 - expression2,
            ElementExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ElementExpression::Variable(expression1.id())),
                Box::new(ElementExpression::Constant(1)),
            )
        );
        assert_eq!(
            expression2 - expression1,
            ElementExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ElementExpression::Constant(1)),
                Box::new(ElementExpression::Variable(expression1.id())),
            )
        );

        let v = metadata.add_element_variable(String::from("ev2"), ob);
        assert!(v.is_ok());
        let expression2 = v.unwrap();
        assert_eq!(
            expression1 - expression2,
            ElementExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ElementExpression::Variable(expression1.id())),
                Box::new(ElementExpression::Variable(expression2.id())),
            )
        );
        assert_eq!(
            expression2 - expression1,
            ElementExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ElementExpression::Variable(expression2.id())),
                Box::new(ElementExpression::Variable(expression1.id())),
            )
        );

        let v = metadata.add_element_resource_variable(String::from("erv"), ob, true);
        assert!(v.is_ok());
        let expression2 = v.unwrap();
        assert_eq!(
            expression1 - expression2,
            ElementExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ElementExpression::Variable(expression1.id())),
                Box::new(ElementExpression::ResourceVariable(expression2.id())),
            )
        );
        assert_eq!(
            expression2 - expression1,
            ElementExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ElementExpression::ResourceVariable(expression2.id())),
                Box::new(ElementExpression::Variable(expression1.id())),
            )
        );

        let expression1 = ElementExpression::Constant(0);
        assert_eq!(
            expression1.clone() - expression2,
            ElementExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ElementExpression::Constant(0)),
                Box::new(ElementExpression::ResourceVariable(expression2.id())),
            )
        );
        assert_eq!(
            expression2 - expression1,
            ElementExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ElementExpression::ResourceVariable(expression2.id())),
                Box::new(ElementExpression::Constant(0)),
            )
        );

        let expression1 = 1;
        assert_eq!(
            expression1 - expression2,
            ElementExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ElementExpression::Constant(1)),
                Box::new(ElementExpression::ResourceVariable(expression2.id())),
            )
        );
        assert_eq!(
            expression2 - expression1,
            ElementExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ElementExpression::ResourceVariable(expression2.id())),
                Box::new(ElementExpression::Constant(1)),
            )
        );

        let v = metadata.add_element_resource_variable(String::from("erv2"), ob, true);
        assert!(v.is_ok());
        let expression1 = v.unwrap();
        assert_eq!(
            expression1 - expression2,
            ElementExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ElementExpression::ResourceVariable(expression1.id())),
                Box::new(ElementExpression::ResourceVariable(expression2.id())),
            )
        );
        assert_eq!(
            expression2 - expression1,
            ElementExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ElementExpression::ResourceVariable(expression2.id())),
                Box::new(ElementExpression::ResourceVariable(expression1.id())),
            )
        );
    }

    #[test]
    fn element_mul() {
        let expression1 = ElementExpression::Constant(0);
        let expression2 = ElementExpression::Constant(1);
        assert_eq!(
            expression1.clone() * expression2.clone(),
            ElementExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ElementExpression::Constant(0)),
                Box::new(ElementExpression::Constant(1))
            )
        );
        assert_eq!(
            expression2 * expression1,
            ElementExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ElementExpression::Constant(1)),
                Box::new(ElementExpression::Constant(0)),
            )
        );

        let expression1 = ElementExpression::Constant(0);
        let expression2 = 1;
        assert_eq!(
            expression1.clone() * expression2,
            ElementExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ElementExpression::Constant(0)),
                Box::new(ElementExpression::Constant(1))
            )
        );
        assert_eq!(
            expression2 * expression1,
            ElementExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ElementExpression::Constant(1)),
                Box::new(ElementExpression::Constant(0)),
            )
        );

        let mut metadata = StateMetadata::default();
        let ob = metadata.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let v = metadata.add_element_variable(String::from("ev"), ob);
        assert!(v.is_ok());
        let expression1 = v.unwrap();
        let expression2 = ElementExpression::Constant(0);
        assert_eq!(
            expression1 * expression2.clone(),
            ElementExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ElementExpression::Variable(expression1.id())),
                Box::new(ElementExpression::Constant(0)),
            )
        );
        assert_eq!(
            expression2 * expression1,
            ElementExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ElementExpression::Constant(0)),
                Box::new(ElementExpression::Variable(expression1.id())),
            )
        );

        let expression2 = 1;
        assert_eq!(
            expression1 * expression2,
            ElementExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ElementExpression::Variable(expression1.id())),
                Box::new(ElementExpression::Constant(1)),
            )
        );
        assert_eq!(
            expression2 * expression1,
            ElementExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ElementExpression::Constant(1)),
                Box::new(ElementExpression::Variable(expression1.id())),
            )
        );

        let v = metadata.add_element_variable(String::from("ev2"), ob);
        assert!(v.is_ok());
        let expression2 = v.unwrap();
        assert_eq!(
            expression1 * expression2,
            ElementExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ElementExpression::Variable(expression1.id())),
                Box::new(ElementExpression::Variable(expression2.id())),
            )
        );
        assert_eq!(
            expression2 * expression1,
            ElementExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ElementExpression::Variable(expression2.id())),
                Box::new(ElementExpression::Variable(expression1.id())),
            )
        );

        let v = metadata.add_element_resource_variable(String::from("erv"), ob, true);
        assert!(v.is_ok());
        let expression2 = v.unwrap();
        assert_eq!(
            expression1 * expression2,
            ElementExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ElementExpression::Variable(expression1.id())),
                Box::new(ElementExpression::ResourceVariable(expression2.id())),
            )
        );
        assert_eq!(
            expression2 * expression1,
            ElementExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ElementExpression::ResourceVariable(expression2.id())),
                Box::new(ElementExpression::Variable(expression1.id())),
            )
        );

        let expression1 = ElementExpression::Constant(0);
        assert_eq!(
            expression1.clone() * expression2,
            ElementExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ElementExpression::Constant(0)),
                Box::new(ElementExpression::ResourceVariable(expression2.id())),
            )
        );
        assert_eq!(
            expression2 * expression1,
            ElementExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ElementExpression::ResourceVariable(expression2.id())),
                Box::new(ElementExpression::Constant(0)),
            )
        );

        let expression1 = 1;
        assert_eq!(
            expression1 * expression2,
            ElementExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ElementExpression::Constant(1)),
                Box::new(ElementExpression::ResourceVariable(expression2.id())),
            )
        );
        assert_eq!(
            expression2 * expression1,
            ElementExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ElementExpression::ResourceVariable(expression2.id())),
                Box::new(ElementExpression::Constant(1)),
            )
        );

        let v = metadata.add_element_resource_variable(String::from("erv2"), ob, true);
        assert!(v.is_ok());
        let expression1 = v.unwrap();
        assert_eq!(
            expression1 * expression2,
            ElementExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ElementExpression::ResourceVariable(expression1.id())),
                Box::new(ElementExpression::ResourceVariable(expression2.id())),
            )
        );
        assert_eq!(
            expression2 * expression1,
            ElementExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ElementExpression::ResourceVariable(expression2.id())),
                Box::new(ElementExpression::ResourceVariable(expression1.id())),
            )
        );
    }

    #[test]
    fn element_div() {
        let expression1 = ElementExpression::Constant(0);
        let expression2 = ElementExpression::Constant(1);
        assert_eq!(
            expression1.clone() / expression2.clone(),
            ElementExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ElementExpression::Constant(0)),
                Box::new(ElementExpression::Constant(1))
            )
        );
        assert_eq!(
            expression2 / expression1,
            ElementExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ElementExpression::Constant(1)),
                Box::new(ElementExpression::Constant(0)),
            )
        );

        let expression1 = ElementExpression::Constant(0);
        let expression2 = 1;
        assert_eq!(
            expression1.clone() / expression2,
            ElementExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ElementExpression::Constant(0)),
                Box::new(ElementExpression::Constant(1))
            )
        );
        assert_eq!(
            expression2 / expression1,
            ElementExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ElementExpression::Constant(1)),
                Box::new(ElementExpression::Constant(0)),
            )
        );

        let mut metadata = StateMetadata::default();
        let ob = metadata.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let v = metadata.add_element_variable(String::from("ev"), ob);
        assert!(v.is_ok());
        let expression1 = v.unwrap();
        let expression2 = ElementExpression::Constant(0);
        assert_eq!(
            expression1 / expression2.clone(),
            ElementExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ElementExpression::Variable(expression1.id())),
                Box::new(ElementExpression::Constant(0)),
            )
        );
        assert_eq!(
            expression2 / expression1,
            ElementExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ElementExpression::Constant(0)),
                Box::new(ElementExpression::Variable(expression1.id())),
            )
        );

        let expression2 = 1;
        assert_eq!(
            expression1 / expression2,
            ElementExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ElementExpression::Variable(expression1.id())),
                Box::new(ElementExpression::Constant(1)),
            )
        );
        assert_eq!(
            expression2 / expression1,
            ElementExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ElementExpression::Constant(1)),
                Box::new(ElementExpression::Variable(expression1.id())),
            )
        );

        let v = metadata.add_element_variable(String::from("ev2"), ob);
        assert!(v.is_ok());
        let expression2 = v.unwrap();
        assert_eq!(
            expression1 / expression2,
            ElementExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ElementExpression::Variable(expression1.id())),
                Box::new(ElementExpression::Variable(expression2.id())),
            )
        );
        assert_eq!(
            expression2 / expression1,
            ElementExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ElementExpression::Variable(expression2.id())),
                Box::new(ElementExpression::Variable(expression1.id())),
            )
        );

        let v = metadata.add_element_resource_variable(String::from("erv"), ob, true);
        assert!(v.is_ok());
        let expression2 = v.unwrap();
        assert_eq!(
            expression1 / expression2,
            ElementExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ElementExpression::Variable(expression1.id())),
                Box::new(ElementExpression::ResourceVariable(expression2.id())),
            )
        );
        assert_eq!(
            expression2 / expression1,
            ElementExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ElementExpression::ResourceVariable(expression2.id())),
                Box::new(ElementExpression::Variable(expression1.id())),
            )
        );

        let expression1 = ElementExpression::Constant(0);
        assert_eq!(
            expression1.clone() / expression2,
            ElementExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ElementExpression::Constant(0)),
                Box::new(ElementExpression::ResourceVariable(expression2.id())),
            )
        );
        assert_eq!(
            expression2 / expression1,
            ElementExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ElementExpression::ResourceVariable(expression2.id())),
                Box::new(ElementExpression::Constant(0)),
            )
        );

        let expression1 = 1;
        assert_eq!(
            expression1 / expression2,
            ElementExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ElementExpression::Constant(1)),
                Box::new(ElementExpression::ResourceVariable(expression2.id())),
            )
        );
        assert_eq!(
            expression2 / expression1,
            ElementExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ElementExpression::ResourceVariable(expression2.id())),
                Box::new(ElementExpression::Constant(1)),
            )
        );

        let v = metadata.add_element_resource_variable(String::from("erv2"), ob, true);
        assert!(v.is_ok());
        let expression1 = v.unwrap();
        assert_eq!(
            expression1 / expression2,
            ElementExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ElementExpression::ResourceVariable(expression1.id())),
                Box::new(ElementExpression::ResourceVariable(expression2.id())),
            )
        );
        assert_eq!(
            expression2 / expression1,
            ElementExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ElementExpression::ResourceVariable(expression2.id())),
                Box::new(ElementExpression::ResourceVariable(expression1.id())),
            )
        );
    }

    #[test]
    fn element_rem() {
        let expression1 = ElementExpression::Constant(0);
        let expression2 = ElementExpression::Constant(1);
        assert_eq!(
            expression1.clone() % expression2.clone(),
            ElementExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ElementExpression::Constant(0)),
                Box::new(ElementExpression::Constant(1))
            )
        );
        assert_eq!(
            expression2 % expression1,
            ElementExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ElementExpression::Constant(1)),
                Box::new(ElementExpression::Constant(0)),
            )
        );

        let expression1 = ElementExpression::Constant(0);
        let expression2 = 1;
        assert_eq!(
            expression1.clone() % expression2,
            ElementExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ElementExpression::Constant(0)),
                Box::new(ElementExpression::Constant(1))
            )
        );
        assert_eq!(
            expression2 % expression1,
            ElementExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ElementExpression::Constant(1)),
                Box::new(ElementExpression::Constant(0)),
            )
        );

        let mut metadata = StateMetadata::default();
        let ob = metadata.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let v = metadata.add_element_variable(String::from("ev"), ob);
        assert!(v.is_ok());
        let expression1 = v.unwrap();
        let expression2 = ElementExpression::Constant(0);
        assert_eq!(
            expression1 % expression2.clone(),
            ElementExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ElementExpression::Variable(expression1.id())),
                Box::new(ElementExpression::Constant(0)),
            )
        );
        assert_eq!(
            expression2 % expression1,
            ElementExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ElementExpression::Constant(0)),
                Box::new(ElementExpression::Variable(expression1.id())),
            )
        );

        let expression2 = 1;
        assert_eq!(
            expression1 % expression2,
            ElementExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ElementExpression::Variable(expression1.id())),
                Box::new(ElementExpression::Constant(1)),
            )
        );
        assert_eq!(
            expression2 % expression1,
            ElementExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ElementExpression::Constant(1)),
                Box::new(ElementExpression::Variable(expression1.id())),
            )
        );

        let v = metadata.add_element_variable(String::from("ev2"), ob);
        assert!(v.is_ok());
        let expression2 = v.unwrap();
        assert_eq!(
            expression1 % expression2,
            ElementExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ElementExpression::Variable(expression1.id())),
                Box::new(ElementExpression::Variable(expression2.id())),
            )
        );
        assert_eq!(
            expression2 % expression1,
            ElementExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ElementExpression::Variable(expression2.id())),
                Box::new(ElementExpression::Variable(expression1.id())),
            )
        );

        let v = metadata.add_element_resource_variable(String::from("erv"), ob, true);
        assert!(v.is_ok());
        let expression2 = v.unwrap();
        assert_eq!(
            expression1 % expression2,
            ElementExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ElementExpression::Variable(expression1.id())),
                Box::new(ElementExpression::ResourceVariable(expression2.id())),
            )
        );
        assert_eq!(
            expression2 % expression1,
            ElementExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ElementExpression::ResourceVariable(expression2.id())),
                Box::new(ElementExpression::Variable(expression1.id())),
            )
        );

        let expression1 = ElementExpression::Constant(0);
        assert_eq!(
            expression1.clone() % expression2,
            ElementExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ElementExpression::Constant(0)),
                Box::new(ElementExpression::ResourceVariable(expression2.id())),
            )
        );
        assert_eq!(
            expression2 % expression1,
            ElementExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ElementExpression::ResourceVariable(expression2.id())),
                Box::new(ElementExpression::Constant(0)),
            )
        );

        let expression1 = 1;
        assert_eq!(
            expression1 % expression2,
            ElementExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ElementExpression::Constant(1)),
                Box::new(ElementExpression::ResourceVariable(expression2.id())),
            )
        );
        assert_eq!(
            expression2 % expression1,
            ElementExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ElementExpression::ResourceVariable(expression2.id())),
                Box::new(ElementExpression::Constant(1)),
            )
        );

        let v = metadata.add_element_resource_variable(String::from("erv2"), ob, true);
        assert!(v.is_ok());
        let expression1 = v.unwrap();
        assert_eq!(
            expression1 % expression2,
            ElementExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ElementExpression::ResourceVariable(expression1.id())),
                Box::new(ElementExpression::ResourceVariable(expression2.id())),
            )
        );
        assert_eq!(
            expression2 % expression1,
            ElementExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ElementExpression::ResourceVariable(expression2.id())),
                Box::new(ElementExpression::ResourceVariable(expression1.id())),
            )
        );
    }

    #[test]
    fn element_max() {
        let expression1 = ElementExpression::Constant(0);
        let expression2 = ElementExpression::Constant(1);
        assert_eq!(
            expression1.clone().max(expression2.clone()),
            ElementExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ElementExpression::Constant(0)),
                Box::new(ElementExpression::Constant(1))
            )
        );
        assert_eq!(
            expression2.max(expression1),
            ElementExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ElementExpression::Constant(1)),
                Box::new(ElementExpression::Constant(0)),
            )
        );

        let expression1 = ElementExpression::Constant(0);
        let expression2 = 1;
        assert_eq!(
            expression1.clone().max(expression2),
            ElementExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ElementExpression::Constant(0)),
                Box::new(ElementExpression::Constant(1))
            )
        );
        assert_eq!(
            MaxMin::max(expression2, expression1),
            ElementExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ElementExpression::Constant(1)),
                Box::new(ElementExpression::Constant(0)),
            )
        );

        let mut metadata = StateMetadata::default();
        let ob = metadata.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let v = metadata.add_element_variable(String::from("ev"), ob);
        assert!(v.is_ok());
        let expression1 = v.unwrap();
        let expression2 = ElementExpression::Constant(0);
        assert_eq!(
            expression1.max(expression2.clone()),
            ElementExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ElementExpression::Variable(expression1.id())),
                Box::new(ElementExpression::Constant(0)),
            )
        );
        assert_eq!(
            expression2.max(expression1),
            ElementExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ElementExpression::Constant(0)),
                Box::new(ElementExpression::Variable(expression1.id())),
            )
        );

        let expression2 = 1;
        assert_eq!(
            expression1.max(expression2),
            ElementExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ElementExpression::Variable(expression1.id())),
                Box::new(ElementExpression::Constant(1)),
            )
        );
        assert_eq!(
            MaxMin::max(expression2, expression1),
            ElementExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ElementExpression::Constant(1)),
                Box::new(ElementExpression::Variable(expression1.id())),
            )
        );

        let v = metadata.add_element_variable(String::from("ev2"), ob);
        assert!(v.is_ok());
        let expression2 = v.unwrap();
        assert_eq!(
            expression1.max(expression2),
            ElementExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ElementExpression::Variable(expression1.id())),
                Box::new(ElementExpression::Variable(expression2.id())),
            )
        );
        assert_eq!(
            expression2.max(expression1),
            ElementExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ElementExpression::Variable(expression2.id())),
                Box::new(ElementExpression::Variable(expression1.id())),
            )
        );

        let v = metadata.add_element_resource_variable(String::from("erv"), ob, true);
        assert!(v.is_ok());
        let expression2 = v.unwrap();
        assert_eq!(
            expression1.max(expression2),
            ElementExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ElementExpression::Variable(expression1.id())),
                Box::new(ElementExpression::ResourceVariable(expression2.id())),
            )
        );
        assert_eq!(
            expression2.max(expression1),
            ElementExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ElementExpression::ResourceVariable(expression2.id())),
                Box::new(ElementExpression::Variable(expression1.id())),
            )
        );

        let expression1 = ElementExpression::Constant(0);
        assert_eq!(
            expression1.clone().max(expression2),
            ElementExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ElementExpression::Constant(0)),
                Box::new(ElementExpression::ResourceVariable(expression2.id())),
            )
        );
        assert_eq!(
            expression2.max(expression1),
            ElementExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ElementExpression::ResourceVariable(expression2.id())),
                Box::new(ElementExpression::Constant(0)),
            )
        );

        let expression1 = 1;
        assert_eq!(
            MaxMin::max(expression1, expression2),
            ElementExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ElementExpression::Constant(1)),
                Box::new(ElementExpression::ResourceVariable(expression2.id())),
            )
        );
        assert_eq!(
            expression2.max(expression1),
            ElementExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ElementExpression::ResourceVariable(expression2.id())),
                Box::new(ElementExpression::Constant(1)),
            )
        );

        let v = metadata.add_element_resource_variable(String::from("erv2"), ob, true);
        assert!(v.is_ok());
        let expression1 = v.unwrap();
        assert_eq!(
            expression1.max(expression2),
            ElementExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ElementExpression::ResourceVariable(expression1.id())),
                Box::new(ElementExpression::ResourceVariable(expression2.id())),
            )
        );
        assert_eq!(
            expression2.max(expression1),
            ElementExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ElementExpression::ResourceVariable(expression2.id())),
                Box::new(ElementExpression::ResourceVariable(expression1.id())),
            )
        );
    }

    #[test]
    fn element_min() {
        let expression1 = ElementExpression::Constant(0);
        let expression2 = ElementExpression::Constant(1);
        assert_eq!(
            expression1.clone().min(expression2.clone()),
            ElementExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ElementExpression::Constant(0)),
                Box::new(ElementExpression::Constant(1))
            )
        );
        assert_eq!(
            expression2.min(expression1),
            ElementExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ElementExpression::Constant(1)),
                Box::new(ElementExpression::Constant(0)),
            )
        );

        let expression1 = ElementExpression::Constant(0);
        let expression2 = 1;
        assert_eq!(
            expression1.clone().min(expression2),
            ElementExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ElementExpression::Constant(0)),
                Box::new(ElementExpression::Constant(1))
            )
        );
        assert_eq!(
            MaxMin::min(expression2, expression1),
            ElementExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ElementExpression::Constant(1)),
                Box::new(ElementExpression::Constant(0)),
            )
        );

        let mut metadata = StateMetadata::default();
        let ob = metadata.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let v = metadata.add_element_variable(String::from("ev"), ob);
        assert!(v.is_ok());
        let expression1 = v.unwrap();
        let expression2 = ElementExpression::Constant(0);
        assert_eq!(
            expression1.min(expression2.clone()),
            ElementExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ElementExpression::Variable(expression1.id())),
                Box::new(ElementExpression::Constant(0)),
            )
        );
        assert_eq!(
            expression2.min(expression1),
            ElementExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ElementExpression::Constant(0)),
                Box::new(ElementExpression::Variable(expression1.id())),
            )
        );

        let expression2 = 1;
        assert_eq!(
            expression1.min(expression2),
            ElementExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ElementExpression::Variable(expression1.id())),
                Box::new(ElementExpression::Constant(1)),
            )
        );
        assert_eq!(
            MaxMin::min(expression2, expression1),
            ElementExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ElementExpression::Constant(1)),
                Box::new(ElementExpression::Variable(expression1.id())),
            )
        );

        let v = metadata.add_element_variable(String::from("ev2"), ob);
        assert!(v.is_ok());
        let expression2 = v.unwrap();
        assert_eq!(
            expression1.min(expression2),
            ElementExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ElementExpression::Variable(expression1.id())),
                Box::new(ElementExpression::Variable(expression2.id())),
            )
        );
        assert_eq!(
            expression2.min(expression1),
            ElementExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ElementExpression::Variable(expression2.id())),
                Box::new(ElementExpression::Variable(expression1.id())),
            )
        );

        let v = metadata.add_element_resource_variable(String::from("erv"), ob, true);
        assert!(v.is_ok());
        let expression2 = v.unwrap();
        assert_eq!(
            expression1.min(expression2),
            ElementExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ElementExpression::Variable(expression1.id())),
                Box::new(ElementExpression::ResourceVariable(expression2.id())),
            )
        );
        assert_eq!(
            expression2.min(expression1),
            ElementExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ElementExpression::ResourceVariable(expression2.id())),
                Box::new(ElementExpression::Variable(expression1.id())),
            )
        );

        let expression1 = ElementExpression::Constant(0);
        assert_eq!(
            expression1.clone().min(expression2),
            ElementExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ElementExpression::Constant(0)),
                Box::new(ElementExpression::ResourceVariable(expression2.id())),
            )
        );
        assert_eq!(
            expression2.min(expression1),
            ElementExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ElementExpression::ResourceVariable(expression2.id())),
                Box::new(ElementExpression::Constant(0)),
            )
        );

        let expression1 = 1;
        assert_eq!(
            MaxMin::min(expression1, expression2),
            ElementExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ElementExpression::Constant(1)),
                Box::new(ElementExpression::ResourceVariable(expression2.id())),
            )
        );
        assert_eq!(
            expression2.min(expression1),
            ElementExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ElementExpression::ResourceVariable(expression2.id())),
                Box::new(ElementExpression::Constant(1)),
            )
        );

        let v = metadata.add_element_resource_variable(String::from("erv2"), ob, true);
        assert!(v.is_ok());
        let expression1 = v.unwrap();
        assert_eq!(
            expression1.min(expression2),
            ElementExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ElementExpression::ResourceVariable(expression1.id())),
                Box::new(ElementExpression::ResourceVariable(expression2.id())),
            )
        );
        assert_eq!(
            expression2.min(expression1),
            ElementExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ElementExpression::ResourceVariable(expression2.id())),
                Box::new(ElementExpression::ResourceVariable(expression1.id())),
            )
        );
    }

    #[test]
    fn element_table_element() {
        let mut registry = TableRegistry::default();

        let t = registry.add_table_1d(String::from("t"), vec![0, 1]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(
            Table1DHandle::<Element>::element(&t, 0),
            ElementExpression::Table(Box::new(TableExpression::Table1D(
                t.id(),
                ElementExpression::Constant(0)
            )))
        );

        let t = registry.add_table_2d(String::from("t"), vec![vec![0, 1]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(
            Table2DHandle::<Element>::element(&t, 0, 0),
            ElementExpression::Table(Box::new(TableExpression::Table2D(
                t.id(),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
            )))
        );

        let t = registry.add_table_3d(String::from("t"), vec![vec![vec![0, 1]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(
            Table3DHandle::<Element>::element(&t, 0, 0, 0),
            ElementExpression::Table(Box::new(TableExpression::Table3D(
                t.id(),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
            )))
        );

        let t = registry.add_table(String::from("t"), FxHashMap::default(), 0);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(
            TableHandle::<Element>::element(&t, vec![0, 0, 0, 0]),
            ElementExpression::Table(Box::new(TableExpression::Table(
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
    fn element_if_then_else() {
        let condition = Condition::Constant(true);

        let expression1 = ElementExpression::Constant(0);
        let expression2 = ElementExpression::Constant(1);
        assert_eq!(
            IfThenElse::<ElementExpression>::if_then_else(
                condition.clone(),
                expression1.clone(),
                expression2.clone(),
            ),
            ElementExpression::If(
                Box::new(condition.clone()),
                Box::new(ElementExpression::Constant(0)),
                Box::new(ElementExpression::Constant(1))
            )
        );
        assert_eq!(
            IfThenElse::<ElementExpression>::if_then_else(
                condition.clone(),
                expression2,
                expression1
            ),
            ElementExpression::If(
                Box::new(condition.clone()),
                Box::new(ElementExpression::Constant(1)),
                Box::new(ElementExpression::Constant(0)),
            )
        );

        let expression1 = ElementExpression::Constant(0);
        let expression2 = 1;
        assert_eq!(
            IfThenElse::<ElementExpression>::if_then_else(
                condition.clone(),
                expression1.clone(),
                expression2,
            ),
            ElementExpression::If(
                Box::new(condition.clone()),
                Box::new(ElementExpression::Constant(0)),
                Box::new(ElementExpression::Constant(1))
            )
        );
        assert_eq!(
            IfThenElse::<ElementExpression>::if_then_else(
                condition.clone(),
                expression2,
                expression1
            ),
            ElementExpression::If(
                Box::new(condition.clone()),
                Box::new(ElementExpression::Constant(1)),
                Box::new(ElementExpression::Constant(0)),
            )
        );

        let mut metadata = StateMetadata::default();
        let ob = metadata.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let v = metadata.add_element_variable(String::from("ev"), ob);
        assert!(v.is_ok());
        let expression1 = v.unwrap();
        let expression2 = ElementExpression::Constant(0);
        assert_eq!(
            IfThenElse::<ElementExpression>::if_then_else(
                condition.clone(),
                expression1,
                expression2.clone(),
            ),
            ElementExpression::If(
                Box::new(condition.clone()),
                Box::new(ElementExpression::Variable(expression1.id())),
                Box::new(ElementExpression::Constant(0)),
            )
        );
        assert_eq!(
            IfThenElse::<ElementExpression>::if_then_else(
                condition.clone(),
                expression2,
                expression1
            ),
            ElementExpression::If(
                Box::new(condition.clone()),
                Box::new(ElementExpression::Constant(0)),
                Box::new(ElementExpression::Variable(expression1.id())),
            )
        );

        let expression2 = 1;
        assert_eq!(
            IfThenElse::<ElementExpression>::if_then_else(
                condition.clone(),
                expression1,
                expression2
            ),
            ElementExpression::If(
                Box::new(condition.clone()),
                Box::new(ElementExpression::Variable(expression1.id())),
                Box::new(ElementExpression::Constant(1)),
            )
        );
        assert_eq!(
            IfThenElse::<ElementExpression>::if_then_else(
                condition.clone(),
                expression2,
                expression1
            ),
            ElementExpression::If(
                Box::new(condition.clone()),
                Box::new(ElementExpression::Constant(1)),
                Box::new(ElementExpression::Variable(expression1.id())),
            )
        );

        let v = metadata.add_element_variable(String::from("ev2"), ob);
        assert!(v.is_ok());
        let expression2 = v.unwrap();
        assert_eq!(
            IfThenElse::<ElementExpression>::if_then_else(
                condition.clone(),
                expression1,
                expression2
            ),
            ElementExpression::If(
                Box::new(condition.clone()),
                Box::new(ElementExpression::Variable(expression1.id())),
                Box::new(ElementExpression::Variable(expression2.id())),
            )
        );
        assert_eq!(
            IfThenElse::<ElementExpression>::if_then_else(
                condition.clone(),
                expression2,
                expression1
            ),
            ElementExpression::If(
                Box::new(condition.clone()),
                Box::new(ElementExpression::Variable(expression2.id())),
                Box::new(ElementExpression::Variable(expression1.id())),
            )
        );

        let v = metadata.add_element_resource_variable(String::from("erv"), ob, true);
        assert!(v.is_ok());
        let expression2 = v.unwrap();
        assert_eq!(
            IfThenElse::<ElementExpression>::if_then_else(
                condition.clone(),
                expression1,
                expression2
            ),
            ElementExpression::If(
                Box::new(condition.clone()),
                Box::new(ElementExpression::Variable(expression1.id())),
                Box::new(ElementExpression::ResourceVariable(expression2.id())),
            )
        );
        assert_eq!(
            IfThenElse::<ElementExpression>::if_then_else(
                condition.clone(),
                expression2,
                expression1
            ),
            ElementExpression::If(
                Box::new(condition.clone()),
                Box::new(ElementExpression::ResourceVariable(expression2.id())),
                Box::new(ElementExpression::Variable(expression1.id())),
            )
        );

        let expression1 = ElementExpression::Constant(0);
        assert_eq!(
            IfThenElse::<ElementExpression>::if_then_else(
                condition.clone(),
                expression1.clone(),
                expression2,
            ),
            ElementExpression::If(
                Box::new(condition.clone()),
                Box::new(ElementExpression::Constant(0)),
                Box::new(ElementExpression::ResourceVariable(expression2.id())),
            )
        );
        assert_eq!(
            IfThenElse::<ElementExpression>::if_then_else(
                condition.clone(),
                expression2,
                expression1
            ),
            ElementExpression::If(
                Box::new(condition.clone()),
                Box::new(ElementExpression::ResourceVariable(expression2.id())),
                Box::new(ElementExpression::Constant(0)),
            )
        );

        let expression1 = 1;
        assert_eq!(
            IfThenElse::<ElementExpression>::if_then_else(
                condition.clone(),
                expression1,
                expression2
            ),
            ElementExpression::If(
                Box::new(condition.clone()),
                Box::new(ElementExpression::Constant(1)),
                Box::new(ElementExpression::ResourceVariable(expression2.id())),
            )
        );
        assert_eq!(
            IfThenElse::<ElementExpression>::if_then_else(
                condition.clone(),
                expression2,
                expression1
            ),
            ElementExpression::If(
                Box::new(condition.clone()),
                Box::new(ElementExpression::ResourceVariable(expression2.id())),
                Box::new(ElementExpression::Constant(1)),
            )
        );

        let v = metadata.add_element_resource_variable(String::from("erv2"), ob, true);
        assert!(v.is_ok());
        let expression1 = v.unwrap();
        assert_eq!(
            IfThenElse::<ElementExpression>::if_then_else(
                condition.clone(),
                expression1,
                expression2
            ),
            ElementExpression::If(
                Box::new(condition.clone()),
                Box::new(ElementExpression::ResourceVariable(expression1.id())),
                Box::new(ElementExpression::ResourceVariable(expression2.id())),
            )
        );
        assert_eq!(
            IfThenElse::<ElementExpression>::if_then_else(
                condition.clone(),
                expression2,
                expression1
            ),
            ElementExpression::If(
                Box::new(condition),
                Box::new(ElementExpression::ResourceVariable(expression2.id())),
                Box::new(ElementExpression::ResourceVariable(expression1.id())),
            )
        );
    }

    #[test]
    fn element_constant_eval() {
        let state = generate_state();
        let registry = generate_registry();
        let expression = ElementExpression::Constant(2);
        assert_eq!(expression.eval(&state, &registry), 2);
    }

    #[test]
    fn element_variable_eval() {
        let state = generate_state();
        let registry = generate_registry();
        let expression = ElementExpression::Variable(0);
        assert_eq!(expression.eval(&state, &registry), 1);
    }

    #[test]
    fn element_resource_variable_eval() {
        let state = generate_state();
        let registry = generate_registry();
        let expression = ElementExpression::ResourceVariable(0);
        assert_eq!(expression.eval(&state, &registry), 2);
    }

    #[test]
    fn element_numeric_operation_eval() {
        let state = generate_state();
        let registry = generate_registry();
        let expression = ElementExpression::BinaryOperation(
            BinaryOperator::Add,
            Box::new(ElementExpression::Variable(0)),
            Box::new(ElementExpression::Constant(1)),
        );
        assert_eq!(expression.eval(&state, &registry), 2);
    }

    #[test]
    fn element_last_eval() {
        let state = generate_state();
        let registry = generate_registry();
        let expression = ElementExpression::Last(Box::new(VectorExpression::Reference(
            ReferenceExpression::Constant(vec![0, 1]),
        )));
        assert_eq!(expression.eval(&state, &registry), 1);
    }

    #[test]
    fn element_at_eval() {
        let state = generate_state();
        let registry = generate_registry();
        let expression = ElementExpression::At(
            Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![0, 1],
            ))),
            Box::new(ElementExpression::Constant(0)),
        );
        assert_eq!(expression.eval(&state, &registry), 0);
    }

    #[test]
    fn element_table_eval() {
        let state = generate_state();
        let registry = generate_registry();
        let expression = ElementExpression::Table(Box::new(TableExpression::Constant(0)));
        assert_eq!(expression.eval(&state, &registry), 0);
    }

    #[test]
    fn element_if_eval() {
        let state = generate_state();
        let registry = generate_registry();
        let expression = ElementExpression::If(
            Box::new(Condition::Constant(true)),
            Box::new(ElementExpression::Constant(1)),
            Box::new(ElementExpression::Constant(0)),
        );
        assert_eq!(expression.eval(&state, &registry), 1);
        let expression = ElementExpression::If(
            Box::new(Condition::Constant(false)),
            Box::new(ElementExpression::Constant(1)),
            Box::new(ElementExpression::Constant(0)),
        );
        assert_eq!(expression.eval(&state, &registry), 0);
    }

    #[test]
    fn element_constant_simplify() {
        let registry = generate_registry();
        let expression = ElementExpression::Constant(2);
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn element_variable_simplify() {
        let registry = generate_registry();
        let expression = ElementExpression::Variable(0);
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn element_resource_variable_simplify() {
        let registry = generate_registry();
        let expression = ElementExpression::ResourceVariable(0);
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn element_numeric_operation_simplify() {
        let registry = generate_registry();
        let expression = ElementExpression::BinaryOperation(
            BinaryOperator::Add,
            Box::new(ElementExpression::Constant(0)),
            Box::new(ElementExpression::Constant(1)),
        );
        assert_eq!(
            expression.simplify(&registry),
            ElementExpression::Constant(1)
        );
        let expression = ElementExpression::BinaryOperation(
            BinaryOperator::Add,
            Box::new(ElementExpression::Variable(0)),
            Box::new(ElementExpression::Constant(1)),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn element_last_simplify() {
        let registry = generate_registry();
        let expression = ElementExpression::Last(Box::new(VectorExpression::Reference(
            ReferenceExpression::Constant(vec![0, 1]),
        )));
        assert_eq!(
            expression.simplify(&registry),
            ElementExpression::Constant(1)
        );
        let expression = ElementExpression::Last(Box::new(VectorExpression::Reference(
            ReferenceExpression::Variable(0),
        )));
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn element_at_simplify() {
        let registry = generate_registry();
        let expression = ElementExpression::At(
            Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![0, 1],
            ))),
            Box::new(ElementExpression::Constant(0)),
        );
        assert_eq!(
            expression.simplify(&registry),
            ElementExpression::Constant(0)
        );
        let expression = ElementExpression::At(
            Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![0, 1],
            ))),
            Box::new(ElementExpression::Variable(0)),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn element_table_simplify() {
        let registry = generate_registry();
        let expression = ElementExpression::Table(Box::new(TableExpression::Constant(0)));
        assert_eq!(
            expression.simplify(&registry),
            ElementExpression::Constant(0)
        );
        let expression = ElementExpression::Table(Box::new(TableExpression::Table1D(
            0,
            ElementExpression::Variable(0),
        )));
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn element_if_simplify() {
        let registry = generate_registry();
        let expression = ElementExpression::If(
            Box::new(Condition::Constant(true)),
            Box::new(ElementExpression::Constant(1)),
            Box::new(ElementExpression::Constant(0)),
        );
        assert_eq!(
            expression.simplify(&registry),
            ElementExpression::Constant(1)
        );
        let expression = ElementExpression::If(
            Box::new(Condition::Constant(false)),
            Box::new(ElementExpression::Constant(1)),
            Box::new(ElementExpression::Constant(0)),
        );
        assert_eq!(
            expression.simplify(&registry),
            ElementExpression::Constant(0)
        );
        let expression = ElementExpression::If(
            Box::new(Condition::ComparisonI(
                ComparisonOperator::Gt,
                Box::new(IntegerExpression::Variable(0)),
                Box::new(IntegerExpression::Constant(1)),
            )),
            Box::new(ElementExpression::Constant(1)),
            Box::new(ElementExpression::Constant(0)),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }
}
