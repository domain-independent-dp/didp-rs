use super::argument_expression::ArgumentExpression;
use super::condition::{Condition, IfThenElse};
use super::continuous_expression::ContinuousExpression;
use super::element_expression::ElementExpression;
use super::integer_vector_expression::IntegerVectorExpression;
use super::numeric_operator::{
    BinaryOperator, CastOperator, MaxMin, ReduceOperator, UnaryOperator,
};
use super::numeric_table_expression::NumericTableExpression;
use super::reference_expression::ReferenceExpression;
use super::set_expression::SetExpression;
use super::vector_expression::VectorExpression;
use crate::state::{IntegerResourceVariable, IntegerVariable, SetVariable, StateInterface};
use crate::table_data::{Table1DHandle, Table2DHandle, Table3DHandle, TableHandle};
use crate::table_registry::TableRegistry;
use crate::variable_type::{Continuous, Integer};
use std::boxed::Box;
use std::ops;

/// Integer numeric expression.
#[derive(Debug, PartialEq, Clone)]
pub enum IntegerExpression {
    /// Constant.
    Constant(Integer),
    /// Variable index.
    Variable(usize),
    /// Resource variable index.
    ResourceVariable(usize),
    /// The cost of the transitioned state.
    Cost,
    /// Unary arithmetic operation.
    UnaryOperation(UnaryOperator, Box<IntegerExpression>),
    /// Binary arithmetic operation.
    BinaryOperation(
        BinaryOperator,
        Box<IntegerExpression>,
        Box<IntegerExpression>,
    ),
    /// The cardinality of a set expression.
    Cardinality(SetExpression),
    /// The length of a set expression.
    Length(VectorExpression),
    /// A constant in an integer table.
    Table(Box<NumericTableExpression<Integer>>),
    /// If-then-else expression, which returns the first one if the condition holds and the second one otherwise.
    If(
        Box<Condition>,
        Box<IntegerExpression>,
        Box<IntegerExpression>,
    ),
    /// Conversion from a continuous expression.
    FromContinuous(CastOperator, Box<ContinuousExpression>),
    /// The last value in an integer vector.
    Last(Box<IntegerVectorExpression>),
    /// An item in an integer vector.
    At(Box<IntegerVectorExpression>, ElementExpression),
    /// Reduce operation on an integer vector expression.
    Reduce(ReduceOperator, Box<IntegerVectorExpression>),
}

impl Default for IntegerExpression {
    /// Returns an expression representing a constant zero.
    #[inline]
    fn default() -> Self {
        Self::Constant(0)
    }
}

impl From<Integer> for IntegerExpression {
    /// Returns an expression representing the constant.
    #[inline]
    fn from(v: Integer) -> Self {
        Self::Constant(v)
    }
}

impl From<IntegerVariable> for IntegerExpression {
    /// Returns an expression representing the variable.
    #[inline]
    fn from(v: IntegerVariable) -> Self {
        Self::Variable(v.id())
    }
}

impl From<IntegerResourceVariable> for IntegerExpression {
    /// Returns an expression representing the resource variable.
    #[inline]
    fn from(v: IntegerResourceVariable) -> Self {
        Self::ResourceVariable(v.id())
    }
}

impl IntegerExpression {
    /// Returns an expression representing the abstract value.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let model = Model::default();
    /// let state = model.target.clone();
    /// let expression = IntegerExpression::from(-1);
    /// let expression = expression.abs();
    ///
    /// assert_eq!(expression.eval(&state, &model.table_registry), 1);
    /// ```
    #[inline]
    pub fn abs(self) -> IntegerExpression {
        Self::UnaryOperation(UnaryOperator::Abs, Box::new(self))
    }
}

impl ops::Neg for IntegerExpression {
    type Output = IntegerExpression;

    /// Returns an expression representing the negative value.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let model = Model::default();
    /// let state = model.target.clone();
    /// let expression = IntegerExpression::from(1);
    /// let expression = -expression;
    ///
    /// assert_eq!(expression.eval(&state, &model.table_registry), -1);
    /// ```
    #[inline]
    fn neg(self) -> Self::Output {
        Self::UnaryOperation(UnaryOperator::Neg, Box::new(self))
    }
}

impl ops::Add for IntegerExpression {
    type Output = IntegerExpression;

    /// Returns an expression representing the addition.
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let model = Model::default();
    /// let state = model.target.clone();
    /// let a = IntegerExpression::from(2);
    /// let b = IntegerExpression::from(3);
    /// let expression = a + b;
    ///
    /// assert_eq!(expression.eval(&state, &model.table_registry), 5);
    /// ```
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        IntegerExpression::BinaryOperation(BinaryOperator::Add, Box::new(self), Box::new(rhs))
    }
}

impl ops::Sub for IntegerExpression {
    type Output = IntegerExpression;

    /// Returns an expression representing the subtraction.
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let model = Model::default();
    /// let state = model.target.clone();
    /// let a = IntegerExpression::from(2);
    /// let b = IntegerExpression::from(3);
    /// let expression = a - b;
    ///
    /// assert_eq!(expression.eval(&state, &model.table_registry), -1);
    /// ```
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        IntegerExpression::BinaryOperation(BinaryOperator::Sub, Box::new(self), Box::new(rhs))
    }
}

impl ops::Mul for IntegerExpression {
    type Output = IntegerExpression;

    /// Returns an expression representing the multiplication.
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let model = Model::default();
    /// let state = model.target.clone();
    /// let a = IntegerExpression::from(2);
    /// let b = IntegerExpression::from(3);
    /// let expression = a * b;
    ///
    /// assert_eq!(expression.eval(&state, &model.table_registry), 6);
    /// ```
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        IntegerExpression::BinaryOperation(BinaryOperator::Mul, Box::new(self), Box::new(rhs))
    }
}

impl ops::Div for IntegerExpression {
    type Output = IntegerExpression;

    /// Returns an expression representing the division.
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let model = Model::default();
    /// let state = model.target.clone();
    /// let a = IntegerExpression::from(2);
    /// let b = IntegerExpression::from(3);
    /// let expression = a / b;
    ///
    /// assert_eq!(expression.eval(&state, &model.table_registry), 0);
    /// ```
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        IntegerExpression::BinaryOperation(BinaryOperator::Div, Box::new(self), Box::new(rhs))
    }
}

impl ops::Rem for IntegerExpression {
    type Output = IntegerExpression;

    /// Returns an expression representing the remainder.
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let model = Model::default();
    /// let state = model.target.clone();
    /// let a = IntegerExpression::from(2);
    /// let b = IntegerExpression::from(3);
    /// let expression = a % b;
    ///
    /// assert_eq!(expression.eval(&state, &model.table_registry), 2);
    /// ```
    #[inline]
    fn rem(self, rhs: Self) -> Self::Output {
        IntegerExpression::BinaryOperation(BinaryOperator::Rem, Box::new(self), Box::new(rhs))
    }
}

impl MaxMin for IntegerExpression {
    type Output = IntegerExpression;

    /// Returns an expression representing the maximum.
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let model = Model::default();
    /// let state = model.target.clone();
    /// let a = IntegerExpression::from(2);
    /// let b = IntegerExpression::from(3);
    /// let expression = a.max(b);
    ///
    /// assert_eq!(expression.eval(&state, &model.table_registry), 3);
    /// ```
    #[inline]
    fn max(self, rhs: Self) -> Self::Output {
        IntegerExpression::BinaryOperation(BinaryOperator::Max, Box::new(self), Box::new(rhs))
    }

    /// Returns an expression representing the minimum.
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let model = Model::default();
    /// let state = model.target.clone();
    /// let a = IntegerExpression::from(2);
    /// let b = IntegerExpression::from(3);
    /// let expression = a.min(b);
    ///
    /// assert_eq!(expression.eval(&state, &model.table_registry), 2);
    /// ```
    #[inline]
    fn min(self, rhs: Self) -> Self::Output {
        IntegerExpression::BinaryOperation(BinaryOperator::Min, Box::new(self), Box::new(rhs))
    }
}

impl SetExpression {
    /// Returns an expression representing the length of a set.
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let object_type = model.add_object_type("object", 4).unwrap();
    /// let set = model.create_set(object_type, &[0, 1]).unwrap();
    /// let state = model.target.clone();
    ///
    /// let expression = SetExpression::from(set);
    /// let expression = expression.len();
    ///
    /// assert_eq!(expression.eval(&state, &model.table_registry), 2);
    /// ```
    #[inline]
    pub fn len(self) -> IntegerExpression {
        IntegerExpression::Cardinality(self)
    }
}

impl SetVariable {
    /// Returns an expression representing the length of a set.
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let object_type = model.add_object_type("object", 4).unwrap();
    /// let set = model.create_set(object_type, &[0, 1]).unwrap();
    /// let variable = model.add_set_variable("variable", object_type, set).unwrap();
    /// let state = model.target.clone();
    ///
    /// let expression = variable.len();
    ///
    /// assert_eq!(expression.eval(&state, &model.table_registry), 2);
    /// ```
    #[inline]
    pub fn len(self) -> IntegerExpression {
        IntegerExpression::Cardinality(SetExpression::from(self))
    }
}

impl Table1DHandle<Integer> {
    /// Returns a constant in a 1D integer table.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let table = model.add_table_1d("table", vec![2, 3]).unwrap();
    /// let object_type = model.add_object_type("object", 2).unwrap();
    /// let variable = model.add_element_variable("variable", object_type, 0).unwrap();
    /// let state = model.target.clone();
    ///
    /// let expression = Table1DHandle::<Integer>::element(&table, variable);
    /// assert_eq!(expression.eval(&state, &model.table_registry), 2);
    /// ```
    #[inline]
    pub fn element<T>(&self, x: T) -> IntegerExpression
    where
        ElementExpression: From<T>,
    {
        IntegerExpression::Table(Box::new(NumericTableExpression::Table1D(
            self.id(),
            ElementExpression::from(x),
        )))
    }

    /// Returns the sum of constants over a set expression in a 1D integer table.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let table = model.add_table_1d("table", vec![2, 3]).unwrap();
    /// let object_type = model.add_object_type("object", 2).unwrap();
    /// let set = model.create_set(object_type, &[0, 1]).unwrap();
    /// let variable = model.add_set_variable("variable", object_type, set).unwrap();
    /// let state = model.target.clone();
    ///
    /// let expression = table.sum(variable);
    /// assert_eq!(expression.eval(&state, &model.table_registry), 5);
    /// ```
    #[inline]
    pub fn sum<T>(&self, x: T) -> IntegerExpression
    where
        SetExpression: From<T>,
    {
        IntegerExpression::Table(Box::new(NumericTableExpression::Table1DReduce(
            ReduceOperator::Sum,
            self.id(),
            SetExpression::from(x),
        )))
    }

    /// Returns the product of constants over a set expression in a 1D integer table.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let table = model.add_table_1d("table", vec![2, 3]).unwrap();
    /// let object_type = model.add_object_type("object", 2).unwrap();
    /// let set = model.create_set(object_type, &[0, 1]).unwrap();
    /// let variable = model.add_set_variable("variable", object_type, set).unwrap();
    /// let state = model.target.clone();
    ///
    /// let expression = table.product(variable);
    /// assert_eq!(expression.eval(&state, &model.table_registry), 6);
    /// ```
    #[inline]
    pub fn product<T>(&self, x: T) -> IntegerExpression
    where
        SetExpression: From<T>,
    {
        IntegerExpression::Table(Box::new(NumericTableExpression::Table1DReduce(
            ReduceOperator::Product,
            self.id(),
            SetExpression::from(x),
        )))
    }

    /// Returns the maximum of constants over a set expression in a 1D integer table.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let table = model.add_table_1d("table", vec![2, 3]).unwrap();
    /// let object_type = model.add_object_type("object", 2).unwrap();
    /// let set = model.create_set(object_type, &[0, 1]).unwrap();
    /// let variable = model.add_set_variable("variable", object_type, set).unwrap();
    /// let state = model.target.clone();
    ///
    /// let expression = table.max(variable);
    /// assert_eq!(expression.eval(&state, &model.table_registry), 3);
    /// ```
    #[inline]
    pub fn max<T>(&self, x: T) -> IntegerExpression
    where
        SetExpression: From<T>,
    {
        IntegerExpression::Table(Box::new(NumericTableExpression::Table1DReduce(
            ReduceOperator::Max,
            self.id(),
            SetExpression::from(x),
        )))
    }

    /// Returns the minimum of constants over a set expression in a 1D integer table.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let table = model.add_table_1d("table", vec![2, 3]).unwrap();
    /// let object_type = model.add_object_type("object", 2).unwrap();
    /// let set = model.create_set(object_type, &[0, 1]).unwrap();
    /// let variable = model.add_set_variable("variable", object_type, set).unwrap();
    /// let state = model.target.clone();
    ///
    /// let expression = table.min(variable);
    /// assert_eq!(expression.eval(&state, &model.table_registry), 2);
    /// ```
    #[inline]
    pub fn min<T>(&self, x: T) -> IntegerExpression
    where
        SetExpression: From<T>,
    {
        IntegerExpression::Table(Box::new(NumericTableExpression::Table1DReduce(
            ReduceOperator::Min,
            self.id(),
            SetExpression::from(x),
        )))
    }
}

impl Table2DHandle<Integer> {
    /// Returns a constant in a 2D integer table.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let table = model.add_table_2d("table", vec![vec![2, 3], vec![4, 5]]).unwrap();
    /// let object_type = model.add_object_type("object", 2).unwrap();
    /// let variable = model.add_element_variable("variable", object_type, 0).unwrap();
    /// let state = model.target.clone();
    ///
    /// let expression = Table2DHandle::<Integer>::element(&table, variable, 1);
    /// assert_eq!(expression.eval(&state, &model.table_registry), 3);
    /// ```
    #[inline]
    pub fn element<T, U>(&self, x: T, y: U) -> IntegerExpression
    where
        ElementExpression: From<T>,
        ElementExpression: From<U>,
    {
        IntegerExpression::Table(Box::new(NumericTableExpression::Table2D(
            self.id(),
            ElementExpression::from(x),
            ElementExpression::from(y),
        )))
    }

    /// Returns the sum of constants over a set expression in a 2D integer table.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let table = model.add_table_2d("table", vec![vec![2, 3], vec![4, 5]]).unwrap();
    /// let object_type = model.add_object_type("object", 2).unwrap();
    /// let set = model.create_set(object_type, &[0, 1]).unwrap();
    /// let x = model.add_set_variable("x", object_type, set).unwrap();
    /// let y = model.add_element_variable("y", object_type, 0).unwrap();
    /// let state = model.target.clone();
    ///
    /// let expression = table.sum_x(x, y);
    /// assert_eq!(expression.eval(&state, &model.table_registry), 6);
    /// ```
    #[inline]
    pub fn sum_x<T, U>(&self, x: T, y: U) -> IntegerExpression
    where
        SetExpression: From<T>,
        ElementExpression: From<U>,
    {
        IntegerExpression::Table(Box::new(NumericTableExpression::Table2DReduceX(
            ReduceOperator::Sum,
            self.id(),
            SetExpression::from(x),
            ElementExpression::from(y),
        )))
    }

    /// Returns the sum of constants over a set expression in a 2D integer table.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let table = model.add_table_2d("table", vec![vec![2, 3], vec![4, 5]]).unwrap();
    /// let object_type = model.add_object_type("object", 2).unwrap();
    /// let x = model.add_element_variable("x", object_type, 0).unwrap();
    /// let set = model.create_set(object_type, &[0, 1]).unwrap();
    /// let y = model.add_set_variable("y", object_type, set).unwrap();
    /// let state = model.target.clone();
    ///
    /// let expression = table.sum_y(x, y);
    /// assert_eq!(expression.eval(&state, &model.table_registry), 5);
    /// ```
    #[inline]
    pub fn sum_y<T, U>(&self, x: T, y: U) -> IntegerExpression
    where
        ElementExpression: From<T>,
        SetExpression: From<U>,
    {
        IntegerExpression::Table(Box::new(NumericTableExpression::Table2DReduceY(
            ReduceOperator::Sum,
            self.id(),
            ElementExpression::from(x),
            SetExpression::from(y),
        )))
    }

    /// Returns the sum of constants over two set expressions in a 2D integer table.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let table = model.add_table_2d("table", vec![vec![2, 3], vec![4, 5]]).unwrap();
    /// let object_type = model.add_object_type("object", 2).unwrap();
    /// let x = model.create_set(object_type, &[0, 1]).unwrap();
    /// let y = model.add_set_variable("y", object_type, x.clone()).unwrap();
    /// let state = model.target.clone();
    ///
    /// let expression = table.sum(x, y);
    /// assert_eq!(expression.eval(&state, &model.table_registry), 14);
    /// ```
    #[inline]
    pub fn sum<T, U>(&self, x: T, y: U) -> IntegerExpression
    where
        SetExpression: From<T>,
        SetExpression: From<U>,
    {
        IntegerExpression::Table(Box::new(NumericTableExpression::Table2DReduce(
            ReduceOperator::Sum,
            self.id(),
            SetExpression::from(x),
            SetExpression::from(y),
        )))
    }

    /// Returns the product of constants over a set expression in a 2D integer table.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let table = model.add_table_2d("table", vec![vec![2, 3], vec![4, 5]]).unwrap();
    /// let object_type = model.add_object_type("object", 2).unwrap();
    /// let set = model.create_set(object_type, &[0, 1]).unwrap();
    /// let x = model.add_set_variable("x", object_type, set).unwrap();
    /// let y = model.add_element_variable("y", object_type, 0).unwrap();
    /// let state = model.target.clone();
    ///
    /// let expression = table.product_x(x, y);
    /// assert_eq!(expression.eval(&state, &model.table_registry), 8);
    /// ```
    #[inline]
    pub fn product_x<T, U>(&self, x: T, y: U) -> IntegerExpression
    where
        SetExpression: From<T>,
        ElementExpression: From<U>,
    {
        IntegerExpression::Table(Box::new(NumericTableExpression::Table2DReduceX(
            ReduceOperator::Product,
            self.id(),
            SetExpression::from(x),
            ElementExpression::from(y),
        )))
    }

    /// Returns the product of constants over a set expression in a 2D integer table.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let table = model.add_table_2d("table", vec![vec![2, 3], vec![4, 5]]).unwrap();
    /// let object_type = model.add_object_type("object", 2).unwrap();
    /// let x = model.add_element_variable("x", object_type, 0).unwrap();
    /// let set = model.create_set(object_type, &[0, 1]).unwrap();
    /// let y = model.add_set_variable("y", object_type, set).unwrap();
    /// let state = model.target.clone();
    ///
    /// let expression = table.product_y(x, y);
    /// assert_eq!(expression.eval(&state, &model.table_registry), 6);
    /// ```
    #[inline]
    pub fn product_y<T, U>(&self, x: T, y: U) -> IntegerExpression
    where
        ElementExpression: From<T>,
        SetExpression: From<U>,
    {
        IntegerExpression::Table(Box::new(NumericTableExpression::Table2DReduceY(
            ReduceOperator::Product,
            self.id(),
            ElementExpression::from(x),
            SetExpression::from(y),
        )))
    }

    /// Returns the product of constants over two set expressions in a 2D integer table.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let table = model.add_table_2d("table", vec![vec![2, 3], vec![4, 5]]).unwrap();
    /// let object_type = model.add_object_type("object", 2).unwrap();
    /// let x = model.create_set(object_type, &[0, 1]).unwrap();
    /// let y = model.add_set_variable("y", object_type, x.clone()).unwrap();
    /// let state = model.target.clone();
    ///
    /// let expression = table.product(x, y);
    /// assert_eq!(expression.eval(&state, &model.table_registry), 120);
    /// ```
    #[inline]
    pub fn product<T, U>(&self, x: T, y: U) -> IntegerExpression
    where
        SetExpression: From<T>,
        SetExpression: From<U>,
    {
        IntegerExpression::Table(Box::new(NumericTableExpression::Table2DReduce(
            ReduceOperator::Product,
            self.id(),
            SetExpression::from(x),
            SetExpression::from(y),
        )))
    }

    /// Returns the maximum of constants over a set expression in a 2D integer table.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let table = model.add_table_2d("table", vec![vec![2, 3], vec![4, 5]]).unwrap();
    /// let object_type = model.add_object_type("object", 2).unwrap();
    /// let set = model.create_set(object_type, &[0, 1]).unwrap();
    /// let x = model.add_set_variable("x", object_type, set).unwrap();
    /// let y = model.add_element_variable("y", object_type, 0).unwrap();
    /// let state = model.target.clone();
    ///
    /// let expression = table.max_x(x, y);
    /// assert_eq!(expression.eval(&state, &model.table_registry), 4);
    #[inline]
    pub fn max_x<T, U>(&self, x: T, y: U) -> IntegerExpression
    where
        SetExpression: From<T>,
        ElementExpression: From<U>,
    {
        IntegerExpression::Table(Box::new(NumericTableExpression::Table2DReduceX(
            ReduceOperator::Max,
            self.id(),
            SetExpression::from(x),
            ElementExpression::from(y),
        )))
    }

    /// Returns the maximum of constants over a set expression in a 2D integer table.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let table = model.add_table_2d("table", vec![vec![2, 3], vec![4, 5]]).unwrap();
    /// let object_type = model.add_object_type("object", 2).unwrap();
    /// let x = model.add_element_variable("x", object_type, 0).unwrap();
    /// let set = model.create_set(object_type, &[0, 1]).unwrap();
    /// let y = model.add_set_variable("y", object_type, set).unwrap();
    /// let state = model.target.clone();
    ///
    /// let expression = table.max_y(x, y);
    /// assert_eq!(expression.eval(&state, &model.table_registry), 3);
    #[inline]
    pub fn max_y<T, U>(&self, x: T, y: U) -> IntegerExpression
    where
        ElementExpression: From<T>,
        SetExpression: From<U>,
    {
        IntegerExpression::Table(Box::new(NumericTableExpression::Table2DReduceY(
            ReduceOperator::Max,
            self.id(),
            ElementExpression::from(x),
            SetExpression::from(y),
        )))
    }

    /// Returns the maximum of constants over two set expressions in a 2D integer table.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let table = model.add_table_2d("table", vec![vec![2, 3], vec![4, 5]]).unwrap();
    /// let object_type = model.add_object_type("object", 2).unwrap();
    /// let x = model.create_set(object_type, &[0, 1]).unwrap();
    /// let y = model.add_set_variable("y", object_type, x.clone()).unwrap();
    /// let state = model.target.clone();
    ///
    /// let expression = table.max(x, y);
    /// assert_eq!(expression.eval(&state, &model.table_registry), 5);
    /// ```
    #[inline]
    pub fn max<T, U>(&self, x: T, y: U) -> IntegerExpression
    where
        SetExpression: From<T>,
        SetExpression: From<U>,
    {
        IntegerExpression::Table(Box::new(NumericTableExpression::Table2DReduce(
            ReduceOperator::Max,
            self.id(),
            SetExpression::from(x),
            SetExpression::from(y),
        )))
    }

    /// Returns the minimum of constants over a set expression in a 2D integer table.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let table = model.add_table_2d("table", vec![vec![2, 3], vec![4, 5]]).unwrap();
    /// let object_type = model.add_object_type("object", 2).unwrap();
    /// let set = model.create_set(object_type, &[0, 1]).unwrap();
    /// let x = model.add_set_variable("x", object_type, set).unwrap();
    /// let y = model.add_element_variable("y", object_type, 0).unwrap();
    /// let state = model.target.clone();
    ///
    /// let expression = table.min_x(x, y);
    /// assert_eq!(expression.eval(&state, &model.table_registry), 2);
    #[inline]
    pub fn min_x<T, U>(&self, x: T, y: U) -> IntegerExpression
    where
        SetExpression: From<T>,
        ElementExpression: From<U>,
    {
        IntegerExpression::Table(Box::new(NumericTableExpression::Table2DReduceX(
            ReduceOperator::Min,
            self.id(),
            SetExpression::from(x),
            ElementExpression::from(y),
        )))
    }

    /// Returns the minimum of constants over a set expression in a 2D integer table.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let table = model.add_table_2d("table", vec![vec![2, 3], vec![4, 5]]).unwrap();
    /// let object_type = model.add_object_type("object", 2).unwrap();
    /// let x = model.add_element_variable("x", object_type, 0).unwrap();
    /// let set = model.create_set(object_type, &[0, 1]).unwrap();
    /// let y = model.add_set_variable("y", object_type, set).unwrap();
    /// let state = model.target.clone();
    ///
    /// let expression = table.min_y(x, y);
    /// assert_eq!(expression.eval(&state, &model.table_registry), 2);
    #[inline]
    pub fn min_y<T, U>(&self, x: T, y: U) -> IntegerExpression
    where
        ElementExpression: From<T>,
        SetExpression: From<U>,
    {
        IntegerExpression::Table(Box::new(NumericTableExpression::Table2DReduceY(
            ReduceOperator::Min,
            self.id(),
            ElementExpression::from(x),
            SetExpression::from(y),
        )))
    }

    /// Returns the minimum of constants over two set expressions in a 2D integer table.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let table = model.add_table_2d("table", vec![vec![2, 3], vec![4, 5]]).unwrap();
    /// let object_type = model.add_object_type("object", 2).unwrap();
    /// let x = model.create_set(object_type, &[0, 1]).unwrap();
    /// let y = model.add_set_variable("y", object_type, x.clone()).unwrap();
    /// let state = model.target.clone();
    ///
    /// let expression = table.min(x, y);
    /// assert_eq!(expression.eval(&state, &model.table_registry), 2);
    /// ```
    #[inline]
    pub fn min<T, U>(&self, x: T, y: U) -> IntegerExpression
    where
        SetExpression: From<T>,
        SetExpression: From<U>,
    {
        IntegerExpression::Table(Box::new(NumericTableExpression::Table2DReduce(
            ReduceOperator::Min,
            self.id(),
            SetExpression::from(x),
            SetExpression::from(y),
        )))
    }
}

impl Table3DHandle<Integer> {
    #[inline]
    /// Returns a constant in a 3D integer table.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let table = model.add_table_3d(
    ///     "table",
    ///     vec![vec![vec![2, 3], vec![4, 5]], vec![vec![6, 7], vec![8, 9]]]
    /// ).unwrap();
    /// let object_type = model.add_object_type("object", 2).unwrap();
    /// let variable = model.add_element_variable("variable", object_type, 0).unwrap();
    /// let state = model.target.clone();
    ///
    /// let expression = Table3DHandle::<Integer>::element(&table, variable, variable + 1, 1);
    /// assert_eq!(expression.eval(&state, &model.table_registry), 5);
    /// ```
    pub fn element<T, U, V>(&self, x: T, y: U, z: V) -> IntegerExpression
    where
        ElementExpression: From<T>,
        ElementExpression: From<U>,
        ElementExpression: From<V>,
    {
        IntegerExpression::Table(Box::new(NumericTableExpression::Table3D(
            self.id(),
            ElementExpression::from(x),
            ElementExpression::from(y),
            ElementExpression::from(z),
        )))
    }

    /// Returns the sum of constants over set expressions in a 3D integer table.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let table = model.add_table_3d(
    ///     "table",
    ///     vec![vec![vec![2, 3], vec![4, 5]], vec![vec![6, 7], vec![8, 9]]]
    /// ).unwrap();
    /// let object_type = model.add_object_type("object", 2).unwrap();
    /// let set = model.create_set(object_type, &[0, 1]).unwrap();
    /// let set_variable = model.add_set_variable("set", object_type, set.clone()).unwrap();
    /// let element_variable = model.add_element_variable("element", object_type, 0).unwrap();
    /// let state = model.target.clone();
    ///
    /// let expression = table.sum(set_variable, element_variable, 1);
    /// assert_eq!(expression.eval(&state, &model.table_registry), 10);
    ///
    /// let expression = table.sum(set, set_variable, set_variable);
    /// assert_eq!(expression.eval(&state, &model.table_registry), 44);
    /// ```
    #[inline]
    pub fn sum<T, U, V>(&self, x: T, y: U, z: V) -> IntegerExpression
    where
        ArgumentExpression: From<T>,
        ArgumentExpression: From<U>,
        ArgumentExpression: From<V>,
    {
        IntegerExpression::Table(Box::new(NumericTableExpression::Table3DReduce(
            ReduceOperator::Sum,
            self.id(),
            ArgumentExpression::from(x),
            ArgumentExpression::from(y),
            ArgumentExpression::from(z),
        )))
    }

    /// Returns the product of constants over set expressions in a 3D integer table.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let table = model.add_table_3d(
    ///     "table",
    ///     vec![vec![vec![2, 3], vec![4, 5]], vec![vec![6, 7], vec![8, 9]]]
    /// ).unwrap();
    /// let object_type = model.add_object_type("object", 2).unwrap();
    /// let set = model.create_set(object_type, &[0, 1]).unwrap();
    /// let set_variable = model.add_set_variable("set", object_type, set.clone()).unwrap();
    /// let element_variable = model.add_element_variable("element", object_type, 0).unwrap();
    /// let state = model.target.clone();
    ///
    /// let expression = table.product(set_variable, element_variable, 1);
    /// assert_eq!(expression.eval(&state, &model.table_registry), 21);
    ///
    /// let expression = table.product(set, set_variable, set_variable);
    /// assert_eq!(expression.eval(&state, &model.table_registry), 362880);
    /// ```
    #[inline]
    pub fn product<T, U, V>(&self, x: T, y: U, z: V) -> IntegerExpression
    where
        ArgumentExpression: From<T>,
        ArgumentExpression: From<U>,
        ArgumentExpression: From<V>,
    {
        IntegerExpression::Table(Box::new(NumericTableExpression::Table3DReduce(
            ReduceOperator::Product,
            self.id(),
            ArgumentExpression::from(x),
            ArgumentExpression::from(y),
            ArgumentExpression::from(z),
        )))
    }

    /// Returns the maximum of constants over set expressions in a 3D integer table.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let table = model.add_table_3d(
    ///     "table",
    ///     vec![vec![vec![2, 3], vec![4, 5]], vec![vec![6, 7], vec![8, 9]]]
    /// ).unwrap();
    /// let object_type = model.add_object_type("object", 2).unwrap();
    /// let set = model.create_set(object_type, &[0, 1]).unwrap();
    /// let set_variable = model.add_set_variable("set", object_type, set.clone()).unwrap();
    /// let element_variable = model.add_element_variable("element", object_type, 0).unwrap();
    /// let state = model.target.clone();
    ///
    /// let expression = table.max(set_variable, element_variable, 1);
    /// assert_eq!(expression.eval(&state, &model.table_registry), 7);
    ///
    /// let expression = table.max(set, set_variable, set_variable);
    /// assert_eq!(expression.eval(&state, &model.table_registry), 9);
    /// ```
    #[inline]
    pub fn max<T, U, V>(&self, x: T, y: U, z: V) -> IntegerExpression
    where
        ArgumentExpression: From<T>,
        ArgumentExpression: From<U>,
        ArgumentExpression: From<V>,
    {
        IntegerExpression::Table(Box::new(NumericTableExpression::Table3DReduce(
            ReduceOperator::Max,
            self.id(),
            ArgumentExpression::from(x),
            ArgumentExpression::from(y),
            ArgumentExpression::from(z),
        )))
    }

    /// Returns the minimum of constants over set expressions in a 3D integer table.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let table = model.add_table_3d(
    ///     "table",
    ///     vec![vec![vec![2, 3], vec![4, 5]], vec![vec![6, 7], vec![8, 9]]]
    /// ).unwrap();
    /// let object_type = model.add_object_type("object", 2).unwrap();
    /// let set = model.create_set(object_type, &[0, 1]).unwrap();
    /// let set_variable = model.add_set_variable("set", object_type, set.clone()).unwrap();
    /// let element_variable = model.add_element_variable("element", object_type, 0).unwrap();
    /// let state = model.target.clone();
    ///
    /// let expression = table.min(set_variable, element_variable, 1);
    /// assert_eq!(expression.eval(&state, &model.table_registry), 3);
    ///
    /// let expression = table.min(set, set_variable, set_variable);
    /// assert_eq!(expression.eval(&state, &model.table_registry), 2);
    /// ```
    #[inline]
    pub fn min<T, U, V>(&self, x: T, y: U, z: V) -> IntegerExpression
    where
        ArgumentExpression: From<T>,
        ArgumentExpression: From<U>,
        ArgumentExpression: From<V>,
    {
        IntegerExpression::Table(Box::new(NumericTableExpression::Table3DReduce(
            ReduceOperator::Min,
            self.id(),
            ArgumentExpression::from(x),
            ArgumentExpression::from(y),
            ArgumentExpression::from(z),
        )))
    }
}

impl TableHandle<Integer> {
    /// Returns a constant in an integer table.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    /// use rustc_hash::FxHashMap;
    ///
    /// let mut model = Model::default();
    /// let map = FxHashMap::from_iter(vec![(vec![0, 0, 0, 0], 1), (vec![1, 1, 1, 1], 2)]);
    /// let table = model.add_table("table", map, 0).unwrap();
    /// let object_type = model.add_object_type("object", 2).unwrap();
    /// let variable = model.add_element_variable("variable", object_type, 0).unwrap();
    /// let state = model.target.clone();
    ///
    /// let indices = vec![
    ///     ElementExpression::from(variable),
    ///     ElementExpression::from(0),
    ///     ElementExpression::from(0),
    ///     ElementExpression::from(0),
    /// ];
    /// let expression = TableHandle::<Integer>::element(&table, indices);
    /// assert_eq!(expression.eval(&state, &model.table_registry), 1);
    /// ```
    #[inline]
    pub fn element<T>(&self, indices: Vec<T>) -> IntegerExpression
    where
        ElementExpression: From<T>,
    {
        let indices = indices.into_iter().map(ElementExpression::from).collect();
        IntegerExpression::Table(Box::new(NumericTableExpression::Table(self.id(), indices)))
    }

    /// Returns the sum of constants over set expressions in an integer table.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    /// use dypdl::expression::*;
    /// use rustc_hash::FxHashMap;
    ///
    /// let mut model = Model::default();
    /// let map = FxHashMap::from_iter(vec![(vec![0, 0, 0, 0], 1), (vec![1, 1, 1, 1], 2)]);
    /// let table = model.add_table("table", map, 0).unwrap();
    /// let object_type = model.add_object_type("object", 2).unwrap();
    /// let set = model.create_set(object_type, &[0, 1]).unwrap();
    /// let set_variable = model.add_set_variable("set", object_type, set.clone()).unwrap();
    /// let element_variable = model.add_element_variable("element", object_type, 0).unwrap();
    /// let state = model.target.clone();
    ///
    /// let indices = vec![
    ///     ArgumentExpression::from(set),
    ///     ArgumentExpression::from(set_variable),
    ///     ArgumentExpression::from(element_variable),
    ///     ArgumentExpression::from(0),
    /// ];
    /// let expression = table.sum(indices);
    /// assert_eq!(expression.eval(&state, &model.table_registry), 1);
    /// ```
    #[inline]
    pub fn sum<T>(&self, indices: Vec<T>) -> IntegerExpression
    where
        ArgumentExpression: From<T>,
    {
        let indices = indices.into_iter().map(ArgumentExpression::from).collect();
        IntegerExpression::Table(Box::new(NumericTableExpression::TableReduce(
            ReduceOperator::Sum,
            self.id(),
            indices,
        )))
    }

    /// Returns the product of constants over set expressions in an integer table.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    /// use dypdl::expression::*;
    /// use rustc_hash::FxHashMap;
    ///
    /// let mut model = Model::default();
    /// let map = FxHashMap::from_iter(vec![(vec![0, 0, 0, 0], 1), (vec![1, 1, 1, 1], 2)]);
    /// let table = model.add_table("table", map, 0).unwrap();
    /// let object_type = model.add_object_type("object", 2).unwrap();
    /// let set = model.create_set(object_type, &[0, 1]).unwrap();
    /// let set_variable = model.add_set_variable("set", object_type, set.clone()).unwrap();
    /// let element_variable = model.add_element_variable("element", object_type, 0).unwrap();
    /// let state = model.target.clone();
    ///
    /// let indices = vec![
    ///     ArgumentExpression::from(set),
    ///     ArgumentExpression::from(set_variable),
    ///     ArgumentExpression::from(element_variable),
    ///     ArgumentExpression::from(0),
    /// ];
    /// let expression = table.product(indices);
    /// assert_eq!(expression.eval(&state, &model.table_registry), 0);
    /// ```
    #[inline]
    pub fn product<T>(&self, indices: Vec<T>) -> IntegerExpression
    where
        ArgumentExpression: From<T>,
    {
        let indices = indices.into_iter().map(ArgumentExpression::from).collect();
        IntegerExpression::Table(Box::new(NumericTableExpression::TableReduce(
            ReduceOperator::Product,
            self.id(),
            indices,
        )))
    }

    /// Returns the maximum of constants over set expressions in an integer table.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    /// use dypdl::expression::*;
    /// use rustc_hash::FxHashMap;
    ///
    /// let mut model = Model::default();
    /// let map = FxHashMap::from_iter(vec![(vec![0, 0, 0, 0], 1), (vec![1, 1, 1, 1], 2)]);
    /// let table = model.add_table("table", map, 0).unwrap();
    /// let object_type = model.add_object_type("object", 2).unwrap();
    /// let set = model.create_set(object_type, &[0, 1]).unwrap();
    /// let set_variable = model.add_set_variable("set", object_type, set.clone()).unwrap();
    /// let element_variable = model.add_element_variable("element", object_type, 0).unwrap();
    /// let state = model.target.clone();
    ///
    /// let indices = vec![
    ///     ArgumentExpression::from(set),
    ///     ArgumentExpression::from(set_variable),
    ///     ArgumentExpression::from(element_variable),
    ///     ArgumentExpression::from(0),
    /// ];
    /// let expression = table.max(indices);
    /// assert_eq!(expression.eval(&state, &model.table_registry), 1);
    /// ```
    #[inline]
    pub fn max<T>(&self, indices: Vec<T>) -> IntegerExpression
    where
        ArgumentExpression: From<T>,
    {
        let indices = indices.into_iter().map(ArgumentExpression::from).collect();
        IntegerExpression::Table(Box::new(NumericTableExpression::TableReduce(
            ReduceOperator::Max,
            self.id(),
            indices,
        )))
    }

    /// Returns the minimum of constants over set expressions in an integer table.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    /// use dypdl::expression::*;
    /// use rustc_hash::FxHashMap;
    ///
    /// let mut model = Model::default();
    /// let map = FxHashMap::from_iter(vec![(vec![0, 0, 0, 0], 1), (vec![1, 1, 1, 1], 2)]);
    /// let table = model.add_table("table", map, 0).unwrap();
    /// let object_type = model.add_object_type("object", 2).unwrap();
    /// let set = model.create_set(object_type, &[0, 1]).unwrap();
    /// let set_variable = model.add_set_variable("set", object_type, set.clone()).unwrap();
    /// let element_variable = model.add_element_variable("element", object_type, 0).unwrap();
    /// let state = model.target.clone();
    ///
    /// let indices = vec![
    ///     ArgumentExpression::from(set),
    ///     ArgumentExpression::from(set_variable),
    ///     ArgumentExpression::from(element_variable),
    ///     ArgumentExpression::from(0),
    /// ];
    /// let expression = table.min(indices);
    /// assert_eq!(expression.eval(&state, &model.table_registry), 0);
    /// ```
    #[inline]
    pub fn min<T>(&self, indices: Vec<T>) -> IntegerExpression
    where
        ArgumentExpression: From<T>,
    {
        let indices = indices.into_iter().map(ArgumentExpression::from).collect();
        IntegerExpression::Table(Box::new(NumericTableExpression::TableReduce(
            ReduceOperator::Min,
            self.id(),
            indices,
        )))
    }
}

impl IfThenElse<IntegerExpression> for Condition {
    #[inline]
    fn if_then_else<U, V>(self, lhs: U, rhs: V) -> IntegerExpression
    where
        IntegerExpression: From<U> + From<V>,
    {
        IntegerExpression::If(
            Box::new(self),
            Box::new(IntegerExpression::from(lhs)),
            Box::new(IntegerExpression::from(rhs)),
        )
    }
}

macro_rules! impl_unary_ops {
    ($T:ty) => {
        impl $T {
            /// Returns an expression representing the absolute value
            #[inline]
            pub fn abs(self) -> IntegerExpression {
                IntegerExpression::from(self).abs()
            }
        }

        impl ops::Neg for $T {
            type Output = IntegerExpression;

            /// Returns an expression representing the negative value
            #[inline]
            fn neg(self) -> Self::Output {
                -IntegerExpression::from(self)
            }
        }
    };
}

macro_rules! impl_binary_ops {
    ($T:ty,$U:ty) => {
        impl ops::Add<$U> for $T {
            type Output = IntegerExpression;

            /// Returns an expression representing the addition.
            #[inline]
            fn add(self, rhs: $U) -> IntegerExpression {
                IntegerExpression::from(self) + IntegerExpression::from(rhs)
            }
        }

        impl ops::Sub<$U> for $T {
            type Output = IntegerExpression;

            /// Returns an expression representing the subtraction.
            #[inline]
            fn sub(self, rhs: $U) -> IntegerExpression {
                IntegerExpression::from(self) - IntegerExpression::from(rhs)
            }
        }

        impl ops::Mul<$U> for $T {
            type Output = IntegerExpression;

            /// Returns an expression representing the multiplication.
            #[inline]
            fn mul(self, rhs: $U) -> IntegerExpression {
                IntegerExpression::from(self) * IntegerExpression::from(rhs)
            }
        }

        impl ops::Div<$U> for $T {
            type Output = IntegerExpression;

            /// Returns an expression representing the division.
            #[inline]
            fn div(self, rhs: $U) -> IntegerExpression {
                IntegerExpression::from(self) / IntegerExpression::from(rhs)
            }
        }

        impl ops::Rem<$U> for $T {
            type Output = IntegerExpression;

            /// Returns an expression representing the remainder.
            #[inline]
            fn rem(self, rhs: $U) -> IntegerExpression {
                IntegerExpression::from(self) % IntegerExpression::from(rhs)
            }
        }

        impl MaxMin<$U> for $T {
            type Output = IntegerExpression;

            #[inline]
            fn max(self, rhs: $U) -> IntegerExpression {
                IntegerExpression::from(self).max(IntegerExpression::from(rhs))
            }

            #[inline]
            fn min(self, rhs: $U) -> IntegerExpression {
                IntegerExpression::from(self).min(IntegerExpression::from(rhs))
            }
        }
    };
}

impl_binary_ops!(IntegerExpression, Integer);
impl_binary_ops!(IntegerExpression, IntegerVariable);
impl_binary_ops!(IntegerExpression, IntegerResourceVariable);
impl_binary_ops!(Integer, IntegerExpression);
impl_binary_ops!(Integer, IntegerVariable);
impl_binary_ops!(Integer, IntegerResourceVariable);
impl_unary_ops!(IntegerVariable);
impl_binary_ops!(IntegerVariable, IntegerExpression);
impl_binary_ops!(IntegerVariable, Integer);
impl_binary_ops!(IntegerVariable, IntegerVariable);
impl_binary_ops!(IntegerVariable, IntegerResourceVariable);
impl_unary_ops!(IntegerResourceVariable);
impl_binary_ops!(IntegerResourceVariable, IntegerExpression);
impl_binary_ops!(IntegerResourceVariable, Integer);
impl_binary_ops!(IntegerResourceVariable, IntegerVariable);
impl_binary_ops!(IntegerResourceVariable, IntegerResourceVariable);

impl IntegerExpression {
    /// Returns an integer expression by taking the floor of the continuous expression.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let model = Model::default();
    /// let state = model.target.clone();
    ///
    /// let expression = ContinuousExpression::from(1.5);
    /// let expression = IntegerExpression::floor(expression);
    ///
    /// assert_eq!(expression.eval(&state, &model.table_registry), 1);
    /// ```
    pub fn floor<T>(x: T) -> IntegerExpression
    where
        ContinuousExpression: From<T>,
    {
        Self::FromContinuous(CastOperator::Floor, Box::new(ContinuousExpression::from(x)))
    }

    /// Returns an integer expression by taking the ceiling of the continuous expression.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let model = Model::default();
    /// let state = model.target.clone();
    ///
    /// let expression = ContinuousExpression::from(1.5);
    /// let expression = IntegerExpression::ceil(expression);
    ///
    /// assert_eq!(expression.eval(&state, &model.table_registry), 2);
    /// ```
    pub fn ceil<T>(x: T) -> IntegerExpression
    where
        ContinuousExpression: From<T>,
    {
        Self::FromContinuous(CastOperator::Ceil, Box::new(ContinuousExpression::from(x)))
    }

    /// Returns an integer expression by rounding the continuous expression.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let model = Model::default();
    /// let state = model.target.clone();
    ///
    /// let expression = ContinuousExpression::from(1.5);
    /// let expression = IntegerExpression::round(expression);
    ///
    /// assert_eq!(expression.eval(&state, &model.table_registry), 2);
    /// ```
    pub fn round<T>(x: T) -> IntegerExpression
    where
        ContinuousExpression: From<T>,
    {
        Self::FromContinuous(CastOperator::Round, Box::new(ContinuousExpression::from(x)))
    }

    /// Returns an integer expression by truncating the continuous expression.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let model = Model::default();
    /// let state = model.target.clone();
    ///
    /// let expression = ContinuousExpression::from(1.5);
    /// let expression = IntegerExpression::trunc(expression);
    ///
    /// assert_eq!(expression.eval(&state, &model.table_registry), 1);
    /// ```
    pub fn trunc<T>(x: T) -> IntegerExpression
    where
        ContinuousExpression: From<T>,
    {
        Self::FromContinuous(CastOperator::Trunc, Box::new(ContinuousExpression::from(x)))
    }
}

impl IntegerExpression {
    /// Returns the evaluation result.
    ///
    /// # Panics
    ///
    /// Panics if the cost of the transition state is used or a min/max reduce operation is performed on an empty set or vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let model = Model::default();
    /// let state = model.target.clone();
    ///
    /// let expression = IntegerExpression::from(1);
    /// assert_eq!(expression.eval(&state, &model.table_registry), 1);
    /// ```
    #[inline]
    pub fn eval<U: StateInterface>(&self, state: &U, registry: &TableRegistry) -> Integer {
        self.eval_inner(None, state, registry)
    }

    /// Returns the evaluation result of a cost expression.
    ///
    /// # Panics
    ///
    /// Panics if a min/max reduce operation is performed on an empty set or vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let model = Model::default();
    /// let state = model.target.clone();
    ///
    /// let expression = IntegerExpression::Cost + 1;
    /// assert_eq!(expression.eval_cost(1, &state, &model.table_registry), 2);
    /// ```
    #[inline]
    pub fn eval_cost<U: StateInterface>(
        &self,
        cost: Integer,
        state: &U,
        registry: &TableRegistry,
    ) -> Integer {
        self.eval_inner(Some(cost), state, registry)
    }

    fn eval_inner<U: StateInterface>(
        &self,
        cost: Option<Integer>,
        state: &U,
        registry: &TableRegistry,
    ) -> Integer {
        match self {
            Self::Constant(x) => *x,
            Self::Variable(i) => state.get_integer_variable(*i),
            Self::ResourceVariable(i) => state.get_integer_resource_variable(*i),
            Self::Cost => cost.unwrap(),
            Self::UnaryOperation(op, x) => op.eval(x.eval_inner(cost, state, registry)),
            Self::BinaryOperation(op, a, b) => {
                let a = a.eval_inner(cost, state, registry);
                let b = b.eval_inner(cost, state, registry);
                op.eval(a, b)
            }
            Self::Cardinality(SetExpression::Reference(expression)) => {
                let f = |i| state.get_set_variable(i);
                let set = expression.eval(state, registry, &f, &registry.set_tables);
                set.count_ones(..) as Integer
            }
            Self::Cardinality(set) => set.eval(state, registry).count_ones(..) as Integer,
            Self::Table(t) => t.eval(state, registry, &registry.integer_tables),
            Self::If(condition, x, y) => {
                if condition.eval(state, registry) {
                    x.eval_inner(cost, state, registry)
                } else {
                    y.eval_inner(cost, state, registry)
                }
            }
            Self::FromContinuous(op, x) => op.eval(cost.map_or_else(
                || x.eval(state, registry),
                |cost| x.eval_cost(Continuous::from(cost), state, registry),
            )) as Integer,
            Self::Length(VectorExpression::Reference(expression)) => {
                let f = |i| state.get_vector_variable(i);
                let vector = expression.eval(state, registry, &f, &registry.vector_tables);
                vector.len() as Integer
            }
            Self::Length(vector) => vector.eval(state, registry).len() as Integer,
            Self::Last(vector) => match vector.as_ref() {
                IntegerVectorExpression::Constant(vector) => *vector.last().unwrap(),
                vector => *vector.eval_inner(cost, state, registry).last().unwrap(),
            },
            Self::At(vector, i) => match vector.as_ref() {
                IntegerVectorExpression::Constant(vector) => vector[i.eval(state, registry)],
                vector => vector.eval_inner(cost, state, registry)[i.eval(state, registry)],
            },
            Self::Reduce(op, vector) => match vector.as_ref() {
                IntegerVectorExpression::Constant(vector) => op.eval(vector),
                vector => op.eval(&vector.eval_inner(cost, state, registry)),
            },
        }
    }

    /// Returns a simplified version by precomputation.
    ///
    /// # Panics
    ///
    /// Panics if a min/max reduce operation is performed on an empty set or vector.
    pub fn simplify(&self, registry: &TableRegistry) -> IntegerExpression {
        match self {
            Self::UnaryOperation(op, x) => match x.simplify(registry) {
                Self::Constant(x) => Self::Constant(op.eval(x)),
                x => Self::UnaryOperation(op.clone(), Box::new(x)),
            },
            Self::BinaryOperation(op, a, b) => match (a.simplify(registry), b.simplify(registry)) {
                (Self::Constant(a), Self::Constant(b)) => Self::Constant(op.eval(a, b)),
                (a, b) => Self::BinaryOperation(op.clone(), Box::new(a), Box::new(b)),
            },
            Self::Cardinality(expression) => match expression.simplify(registry) {
                SetExpression::Reference(ReferenceExpression::Constant(set)) => {
                    Self::Constant(set.count_ones(..) as Integer)
                }
                expression => Self::Cardinality(expression),
            },
            Self::Length(expression) => match expression.simplify(registry) {
                VectorExpression::Reference(ReferenceExpression::Constant(vector)) => {
                    Self::Constant(vector.len() as Integer)
                }
                expression => Self::Length(expression),
            },
            Self::Table(expression) => {
                match expression.simplify(registry, &registry.integer_tables) {
                    NumericTableExpression::Constant(value) => Self::Constant(value),
                    expression => Self::Table(Box::new(expression)),
                }
            }
            Self::If(condition, x, y) => match condition.simplify(registry) {
                Condition::Constant(true) => x.simplify(registry),
                Condition::Constant(false) => y.simplify(registry),
                condition => Self::If(
                    Box::new(condition),
                    Box::new(x.simplify(registry)),
                    Box::new(y.simplify(registry)),
                ),
            },
            Self::FromContinuous(op, x) => match x.simplify(registry) {
                ContinuousExpression::Constant(x) => Self::Constant(op.eval(x) as Integer),
                x => Self::FromContinuous(op.clone(), Box::new(x)),
            },
            Self::Last(vector) => match vector.simplify(registry) {
                IntegerVectorExpression::Constant(vector) => {
                    Self::Constant(*vector.last().unwrap())
                }
                vector => Self::Last(Box::new(vector)),
            },
            Self::At(vector, i) => match (vector.simplify(registry), i.simplify(registry)) {
                (IntegerVectorExpression::Constant(vector), ElementExpression::Constant(i)) => {
                    Self::Constant(vector[i])
                }
                (vector, i) => Self::At(Box::new(vector), i),
            },
            Self::Reduce(op, vector) => match vector.simplify(registry) {
                IntegerVectorExpression::Constant(vector) => Self::Constant(op.eval(&vector)),
                vector => Self::Reduce(op.clone(), Box::new(vector)),
            },
            _ => self.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::table_expression::TableExpression;
    use super::super::table_vector_expression::TableVectorExpression;
    use super::*;
    use crate::state::*;
    use crate::table_data::TableInterface;
    use crate::variable_type::*;
    use rustc_hash::FxHashMap;

    #[test]
    fn default() {
        assert_eq!(IntegerExpression::default(), IntegerExpression::Constant(0));
    }

    #[test]
    fn from() {
        assert_eq!(IntegerExpression::from(1), IntegerExpression::Constant(1));

        let mut metadata = StateMetadata::default();

        let v = metadata.add_integer_variable(String::from("iv"));
        assert!(v.is_ok());
        let v = v.unwrap();
        assert_eq!(
            IntegerExpression::from(v),
            IntegerExpression::Variable(v.id())
        );

        let v = metadata.add_integer_resource_variable(String::from("irv"), true);
        assert!(v.is_ok());
        let v = v.unwrap();
        assert_eq!(
            IntegerExpression::from(v),
            IntegerExpression::ResourceVariable(v.id())
        );
    }

    #[test]
    fn abs() {
        let expression = IntegerExpression::Constant(1);
        assert_eq!(
            expression.abs(),
            IntegerExpression::UnaryOperation(
                UnaryOperator::Abs,
                Box::new(IntegerExpression::Constant(1))
            )
        );

        let mut metadata = StateMetadata::default();

        let v = metadata.add_integer_variable(String::from("iv"));
        assert!(v.is_ok());
        let v = v.unwrap();
        assert_eq!(
            v.abs(),
            IntegerExpression::UnaryOperation(
                UnaryOperator::Abs,
                Box::new(IntegerExpression::Variable(0))
            )
        );

        let v = metadata.add_integer_resource_variable(String::from("irv"), true);
        assert!(v.is_ok());
        let v = v.unwrap();
        assert_eq!(
            v.abs(),
            IntegerExpression::UnaryOperation(
                UnaryOperator::Abs,
                Box::new(IntegerExpression::ResourceVariable(0))
            )
        );
    }

    #[test]
    fn neg() {
        let expression = IntegerExpression::Constant(1);
        assert_eq!(
            -expression,
            IntegerExpression::UnaryOperation(
                UnaryOperator::Neg,
                Box::new(IntegerExpression::Constant(1))
            )
        );

        let mut metadata = StateMetadata::default();

        let v = metadata.add_integer_variable(String::from("iv"));
        assert!(v.is_ok());
        let v = v.unwrap();
        assert_eq!(
            -v,
            IntegerExpression::UnaryOperation(
                UnaryOperator::Neg,
                Box::new(IntegerExpression::Variable(0))
            )
        );

        let v = metadata.add_integer_resource_variable(String::from("irv"), true);
        assert!(v.is_ok());
        let v = v.unwrap();
        assert_eq!(
            -v,
            IntegerExpression::UnaryOperation(
                UnaryOperator::Neg,
                Box::new(IntegerExpression::ResourceVariable(0))
            )
        );
    }

    #[test]
    fn add() {
        let mut metadata = StateMetadata::default();
        let v1 = metadata.add_integer_variable(String::from("iv1"));
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let v2 = metadata.add_integer_variable(String::from("iv2"));
        assert!(v2.is_ok());
        let v2 = v2.unwrap();
        let rv1 = metadata.add_integer_resource_variable(String::from("irv1"), true);
        assert!(rv1.is_ok());
        let rv1 = rv1.unwrap();
        let rv2 = metadata.add_integer_resource_variable(String::from("irv2"), false);
        assert!(rv2.is_ok());
        let rv2 = rv2.unwrap();

        let expression1 = IntegerExpression::Constant(1);
        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            expression1 + expression2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::Constant(2))
            )
        );

        let expression1 = IntegerExpression::Constant(1);
        assert_eq!(
            expression1 + 2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::Constant(2))
            )
        );

        let expression1 = IntegerExpression::Constant(1);
        assert_eq!(
            expression1 + v2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::Variable(v2.id()))
            )
        );

        let expression1 = IntegerExpression::Constant(1);
        assert_eq!(
            expression1 + rv2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::ResourceVariable(rv2.id()))
            )
        );

        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            1 + expression2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::Constant(2)),
            )
        );

        assert_eq!(
            1 + v2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::Variable(v2.id())),
            )
        );

        assert_eq!(
            1 + rv2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::ResourceVariable(rv2.id())),
            )
        );

        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            v1 + expression2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(IntegerExpression::Variable(v1.id())),
                Box::new(IntegerExpression::Constant(2)),
            )
        );

        assert_eq!(
            v1 + 2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(IntegerExpression::Variable(v1.id())),
                Box::new(IntegerExpression::Constant(2)),
            )
        );

        assert_eq!(
            v1 + v2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(IntegerExpression::Variable(v1.id())),
                Box::new(IntegerExpression::Variable(v2.id())),
            )
        );

        assert_eq!(
            v1 + rv2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(IntegerExpression::Variable(v1.id())),
                Box::new(IntegerExpression::ResourceVariable(rv2.id())),
            )
        );

        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            rv1 + expression2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(IntegerExpression::ResourceVariable(rv1.id())),
                Box::new(IntegerExpression::Constant(2)),
            )
        );

        assert_eq!(
            rv1 + 2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(IntegerExpression::ResourceVariable(rv1.id())),
                Box::new(IntegerExpression::Constant(2)),
            )
        );

        assert_eq!(
            rv1 + v2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(IntegerExpression::ResourceVariable(rv1.id())),
                Box::new(IntegerExpression::Variable(v2.id())),
            )
        );

        assert_eq!(
            rv1 + rv2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(IntegerExpression::ResourceVariable(rv1.id())),
                Box::new(IntegerExpression::ResourceVariable(rv2.id())),
            )
        );
    }

    #[test]
    fn sub() {
        let mut metadata = StateMetadata::default();
        let v1 = metadata.add_integer_variable(String::from("iv1"));
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let v2 = metadata.add_integer_variable(String::from("iv2"));
        assert!(v2.is_ok());
        let v2 = v2.unwrap();
        let rv1 = metadata.add_integer_resource_variable(String::from("irv1"), true);
        assert!(rv1.is_ok());
        let rv1 = rv1.unwrap();
        let rv2 = metadata.add_integer_resource_variable(String::from("irv2"), false);
        assert!(rv2.is_ok());
        let rv2 = rv2.unwrap();

        let expression1 = IntegerExpression::Constant(1);
        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            expression1 - expression2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::Constant(2))
            )
        );

        let expression1 = IntegerExpression::Constant(1);
        assert_eq!(
            expression1 - 2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::Constant(2))
            )
        );

        let expression1 = IntegerExpression::Constant(1);
        assert_eq!(
            expression1 - v2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::Variable(v2.id()))
            )
        );

        let expression1 = IntegerExpression::Constant(1);
        assert_eq!(
            expression1 - rv2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::ResourceVariable(rv2.id()))
            )
        );

        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            1 - expression2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::Constant(2)),
            )
        );

        assert_eq!(
            1 - v2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::Variable(v2.id())),
            )
        );

        assert_eq!(
            1 - rv2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::ResourceVariable(rv2.id())),
            )
        );

        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            v1 - expression2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(IntegerExpression::Variable(v1.id())),
                Box::new(IntegerExpression::Constant(2)),
            )
        );

        assert_eq!(
            v1 - 2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(IntegerExpression::Variable(v1.id())),
                Box::new(IntegerExpression::Constant(2)),
            )
        );

        assert_eq!(
            v1 - v2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(IntegerExpression::Variable(v1.id())),
                Box::new(IntegerExpression::Variable(v2.id())),
            )
        );

        assert_eq!(
            v1 - rv2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(IntegerExpression::Variable(v1.id())),
                Box::new(IntegerExpression::ResourceVariable(rv2.id())),
            )
        );

        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            rv1 - expression2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(IntegerExpression::ResourceVariable(rv1.id())),
                Box::new(IntegerExpression::Constant(2)),
            )
        );

        assert_eq!(
            rv1 - 2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(IntegerExpression::ResourceVariable(rv1.id())),
                Box::new(IntegerExpression::Constant(2)),
            )
        );

        assert_eq!(
            rv1 - v2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(IntegerExpression::ResourceVariable(rv1.id())),
                Box::new(IntegerExpression::Variable(v2.id())),
            )
        );

        assert_eq!(
            rv1 - rv2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(IntegerExpression::ResourceVariable(rv1.id())),
                Box::new(IntegerExpression::ResourceVariable(rv2.id())),
            )
        );
    }

    #[test]
    fn mul() {
        let mut metadata = StateMetadata::default();
        let v1 = metadata.add_integer_variable(String::from("iv1"));
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let v2 = metadata.add_integer_variable(String::from("iv2"));
        assert!(v2.is_ok());
        let v2 = v2.unwrap();
        let rv1 = metadata.add_integer_resource_variable(String::from("irv1"), true);
        assert!(rv1.is_ok());
        let rv1 = rv1.unwrap();
        let rv2 = metadata.add_integer_resource_variable(String::from("irv2"), false);
        assert!(rv2.is_ok());
        let rv2 = rv2.unwrap();

        let expression1 = IntegerExpression::Constant(1);
        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            expression1 * expression2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::Constant(2))
            )
        );

        let expression1 = IntegerExpression::Constant(1);
        assert_eq!(
            expression1 * 2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::Constant(2))
            )
        );

        let expression1 = IntegerExpression::Constant(1);
        assert_eq!(
            expression1 * v2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::Variable(v2.id()))
            )
        );

        let expression1 = IntegerExpression::Constant(1);
        assert_eq!(
            expression1 * rv2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::ResourceVariable(rv2.id()))
            )
        );

        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            1 * expression2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::Constant(2)),
            )
        );

        assert_eq!(
            1 * v2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::Variable(v2.id())),
            )
        );

        assert_eq!(
            1 * rv2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::ResourceVariable(rv2.id())),
            )
        );

        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            v1 * expression2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(IntegerExpression::Variable(v1.id())),
                Box::new(IntegerExpression::Constant(2)),
            )
        );

        assert_eq!(
            v1 * 2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(IntegerExpression::Variable(v1.id())),
                Box::new(IntegerExpression::Constant(2)),
            )
        );

        assert_eq!(
            v1 * v2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(IntegerExpression::Variable(v1.id())),
                Box::new(IntegerExpression::Variable(v2.id())),
            )
        );

        assert_eq!(
            v1 * rv2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(IntegerExpression::Variable(v1.id())),
                Box::new(IntegerExpression::ResourceVariable(rv2.id())),
            )
        );

        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            rv1 * expression2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(IntegerExpression::ResourceVariable(rv1.id())),
                Box::new(IntegerExpression::Constant(2)),
            )
        );

        assert_eq!(
            rv1 * 2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(IntegerExpression::ResourceVariable(rv1.id())),
                Box::new(IntegerExpression::Constant(2)),
            )
        );

        assert_eq!(
            rv1 * v2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(IntegerExpression::ResourceVariable(rv1.id())),
                Box::new(IntegerExpression::Variable(v2.id())),
            )
        );

        assert_eq!(
            rv1 * rv2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(IntegerExpression::ResourceVariable(rv1.id())),
                Box::new(IntegerExpression::ResourceVariable(rv2.id())),
            )
        );
    }

    #[test]
    fn div() {
        let mut metadata = StateMetadata::default();
        let v1 = metadata.add_integer_variable(String::from("iv1"));
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let v2 = metadata.add_integer_variable(String::from("iv2"));
        assert!(v2.is_ok());
        let v2 = v2.unwrap();
        let rv1 = metadata.add_integer_resource_variable(String::from("irv1"), true);
        assert!(rv1.is_ok());
        let rv1 = rv1.unwrap();
        let rv2 = metadata.add_integer_resource_variable(String::from("irv2"), false);
        assert!(rv2.is_ok());
        let rv2 = rv2.unwrap();

        let expression1 = IntegerExpression::Constant(1);
        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            expression1 / expression2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::Constant(2))
            )
        );

        let expression1 = IntegerExpression::Constant(1);
        assert_eq!(
            expression1 / 2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::Constant(2))
            )
        );

        let expression1 = IntegerExpression::Constant(1);
        assert_eq!(
            expression1 / v2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::Variable(v2.id()))
            )
        );

        let expression1 = IntegerExpression::Constant(1);
        assert_eq!(
            expression1 / rv2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::ResourceVariable(rv2.id()))
            )
        );

        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            1 / expression2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::Constant(2)),
            )
        );

        assert_eq!(
            1 / v2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::Variable(v2.id())),
            )
        );

        assert_eq!(
            1 / rv2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::ResourceVariable(rv2.id())),
            )
        );

        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            v1 / expression2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(IntegerExpression::Variable(v1.id())),
                Box::new(IntegerExpression::Constant(2)),
            )
        );

        assert_eq!(
            v1 / 2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(IntegerExpression::Variable(v1.id())),
                Box::new(IntegerExpression::Constant(2)),
            )
        );

        assert_eq!(
            v1 / v2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(IntegerExpression::Variable(v1.id())),
                Box::new(IntegerExpression::Variable(v2.id())),
            )
        );

        assert_eq!(
            v1 / rv2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(IntegerExpression::Variable(v1.id())),
                Box::new(IntegerExpression::ResourceVariable(rv2.id())),
            )
        );

        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            rv1 / expression2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(IntegerExpression::ResourceVariable(rv1.id())),
                Box::new(IntegerExpression::Constant(2)),
            )
        );

        assert_eq!(
            rv1 / 2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(IntegerExpression::ResourceVariable(rv1.id())),
                Box::new(IntegerExpression::Constant(2)),
            )
        );

        assert_eq!(
            rv1 / v2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(IntegerExpression::ResourceVariable(rv1.id())),
                Box::new(IntegerExpression::Variable(v2.id())),
            )
        );

        assert_eq!(
            rv1 / rv2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(IntegerExpression::ResourceVariable(rv1.id())),
                Box::new(IntegerExpression::ResourceVariable(rv2.id())),
            )
        );
    }

    #[test]
    fn rem() {
        let mut metadata = StateMetadata::default();
        let v1 = metadata.add_integer_variable(String::from("iv1"));
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let v2 = metadata.add_integer_variable(String::from("iv2"));
        assert!(v2.is_ok());
        let v2 = v2.unwrap();
        let rv1 = metadata.add_integer_resource_variable(String::from("irv1"), true);
        assert!(rv1.is_ok());
        let rv1 = rv1.unwrap();
        let rv2 = metadata.add_integer_resource_variable(String::from("irv2"), false);
        assert!(rv2.is_ok());
        let rv2 = rv2.unwrap();

        let expression1 = IntegerExpression::Constant(1);
        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            expression1 % expression2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::Constant(2))
            )
        );

        let expression1 = IntegerExpression::Constant(1);
        assert_eq!(
            expression1 % 2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::Constant(2))
            )
        );

        let expression1 = IntegerExpression::Constant(1);
        assert_eq!(
            expression1 % v2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::Variable(v2.id()))
            )
        );

        let expression1 = IntegerExpression::Constant(1);
        assert_eq!(
            expression1 % rv2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::ResourceVariable(rv2.id()))
            )
        );

        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            1 % expression2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::Constant(2)),
            )
        );

        assert_eq!(
            1 % v2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::Variable(v2.id())),
            )
        );

        assert_eq!(
            1 % rv2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::ResourceVariable(rv2.id())),
            )
        );

        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            v1 % expression2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(IntegerExpression::Variable(v1.id())),
                Box::new(IntegerExpression::Constant(2)),
            )
        );

        assert_eq!(
            v1 % 2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(IntegerExpression::Variable(v1.id())),
                Box::new(IntegerExpression::Constant(2)),
            )
        );

        assert_eq!(
            v1 % v2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(IntegerExpression::Variable(v1.id())),
                Box::new(IntegerExpression::Variable(v2.id())),
            )
        );

        assert_eq!(
            v1 % rv2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(IntegerExpression::Variable(v1.id())),
                Box::new(IntegerExpression::ResourceVariable(rv2.id())),
            )
        );

        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            rv1 % expression2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(IntegerExpression::ResourceVariable(rv1.id())),
                Box::new(IntegerExpression::Constant(2)),
            )
        );

        assert_eq!(
            rv1 % 2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(IntegerExpression::ResourceVariable(rv1.id())),
                Box::new(IntegerExpression::Constant(2)),
            )
        );

        assert_eq!(
            rv1 % v2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(IntegerExpression::ResourceVariable(rv1.id())),
                Box::new(IntegerExpression::Variable(v2.id())),
            )
        );

        assert_eq!(
            rv1 % rv2,
            IntegerExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(IntegerExpression::ResourceVariable(rv1.id())),
                Box::new(IntegerExpression::ResourceVariable(rv2.id())),
            )
        );
    }

    #[test]
    fn max() {
        let mut metadata = StateMetadata::default();
        let v1 = metadata.add_integer_variable(String::from("iv1"));
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let v2 = metadata.add_integer_variable(String::from("iv2"));
        assert!(v2.is_ok());
        let v2 = v2.unwrap();
        let rv1 = metadata.add_integer_resource_variable(String::from("irv1"), true);
        assert!(rv1.is_ok());
        let rv1 = rv1.unwrap();
        let rv2 = metadata.add_integer_resource_variable(String::from("irv2"), false);
        assert!(rv2.is_ok());
        let rv2 = rv2.unwrap();

        let expression1 = IntegerExpression::Constant(1);
        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            expression1.max(expression2),
            IntegerExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::Constant(2))
            )
        );

        let expression1 = IntegerExpression::Constant(1);
        assert_eq!(
            expression1.max(2),
            IntegerExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::Constant(2))
            )
        );

        let expression1 = IntegerExpression::Constant(1);
        assert_eq!(
            expression1.max(v2),
            IntegerExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::Variable(v2.id()))
            )
        );

        let expression1 = IntegerExpression::Constant(1);
        assert_eq!(
            expression1.max(rv2),
            IntegerExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::ResourceVariable(rv2.id()))
            )
        );

        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            MaxMin::max(1, expression2),
            IntegerExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::Constant(2)),
            )
        );

        assert_eq!(
            MaxMin::max(1, v2),
            IntegerExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::Variable(v2.id())),
            )
        );

        assert_eq!(
            MaxMin::max(1, rv2),
            IntegerExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::ResourceVariable(rv2.id())),
            )
        );

        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            v1.max(expression2),
            IntegerExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(IntegerExpression::Variable(v1.id())),
                Box::new(IntegerExpression::Constant(2)),
            )
        );

        assert_eq!(
            v1.max(2),
            IntegerExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(IntegerExpression::Variable(v1.id())),
                Box::new(IntegerExpression::Constant(2)),
            )
        );

        assert_eq!(
            v1.max(v2),
            IntegerExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(IntegerExpression::Variable(v1.id())),
                Box::new(IntegerExpression::Variable(v2.id())),
            )
        );

        assert_eq!(
            v1.max(rv2),
            IntegerExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(IntegerExpression::Variable(v1.id())),
                Box::new(IntegerExpression::ResourceVariable(rv2.id())),
            )
        );

        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            rv1.max(expression2),
            IntegerExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(IntegerExpression::ResourceVariable(rv1.id())),
                Box::new(IntegerExpression::Constant(2)),
            )
        );

        assert_eq!(
            rv1.max(2),
            IntegerExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(IntegerExpression::ResourceVariable(rv1.id())),
                Box::new(IntegerExpression::Constant(2)),
            )
        );

        assert_eq!(
            rv1.max(v2),
            IntegerExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(IntegerExpression::ResourceVariable(rv1.id())),
                Box::new(IntegerExpression::Variable(v2.id())),
            )
        );

        assert_eq!(
            rv1.max(rv2),
            IntegerExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(IntegerExpression::ResourceVariable(rv1.id())),
                Box::new(IntegerExpression::ResourceVariable(rv2.id())),
            )
        );
    }

    #[test]
    fn min() {
        let mut metadata = StateMetadata::default();
        let v1 = metadata.add_integer_variable(String::from("iv1"));
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let v2 = metadata.add_integer_variable(String::from("iv2"));
        assert!(v2.is_ok());
        let v2 = v2.unwrap();
        let rv1 = metadata.add_integer_resource_variable(String::from("irv1"), true);
        assert!(rv1.is_ok());
        let rv1 = rv1.unwrap();
        let rv2 = metadata.add_integer_resource_variable(String::from("irv2"), false);
        assert!(rv2.is_ok());
        let rv2 = rv2.unwrap();

        let expression1 = IntegerExpression::Constant(1);
        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            expression1.min(expression2),
            IntegerExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::Constant(2))
            )
        );

        let expression1 = IntegerExpression::Constant(1);
        assert_eq!(
            expression1.min(2),
            IntegerExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::Constant(2))
            )
        );

        let expression1 = IntegerExpression::Constant(1);
        assert_eq!(
            expression1.min(v2),
            IntegerExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::Variable(v2.id()))
            )
        );

        let expression1 = IntegerExpression::Constant(1);
        assert_eq!(
            expression1.min(rv2),
            IntegerExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::ResourceVariable(rv2.id()))
            )
        );

        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            MaxMin::min(1, expression2),
            IntegerExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::Constant(2)),
            )
        );

        assert_eq!(
            MaxMin::min(1, v2),
            IntegerExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::Variable(v2.id())),
            )
        );

        assert_eq!(
            MaxMin::min(1, rv2),
            IntegerExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::ResourceVariable(rv2.id())),
            )
        );

        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            v1.min(expression2),
            IntegerExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(IntegerExpression::Variable(v1.id())),
                Box::new(IntegerExpression::Constant(2)),
            )
        );

        assert_eq!(
            v1.min(2),
            IntegerExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(IntegerExpression::Variable(v1.id())),
                Box::new(IntegerExpression::Constant(2)),
            )
        );

        assert_eq!(
            v1.min(v2),
            IntegerExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(IntegerExpression::Variable(v1.id())),
                Box::new(IntegerExpression::Variable(v2.id())),
            )
        );

        assert_eq!(
            v1.min(rv2),
            IntegerExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(IntegerExpression::Variable(v1.id())),
                Box::new(IntegerExpression::ResourceVariable(rv2.id())),
            )
        );

        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            rv1.min(expression2),
            IntegerExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(IntegerExpression::ResourceVariable(rv1.id())),
                Box::new(IntegerExpression::Constant(2)),
            )
        );

        assert_eq!(
            rv1.min(2),
            IntegerExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(IntegerExpression::ResourceVariable(rv1.id())),
                Box::new(IntegerExpression::Constant(2)),
            )
        );

        assert_eq!(
            rv1.min(v2),
            IntegerExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(IntegerExpression::ResourceVariable(rv1.id())),
                Box::new(IntegerExpression::Variable(v2.id())),
            )
        );

        assert_eq!(
            rv1.min(rv2),
            IntegerExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(IntegerExpression::ResourceVariable(rv1.id())),
                Box::new(IntegerExpression::ResourceVariable(rv2.id())),
            )
        );
    }

    #[test]
    fn set_len() {
        let expression = SetExpression::Reference(ReferenceExpression::Constant(Set::default()));
        assert_eq!(
            expression.clone().len(),
            IntegerExpression::Cardinality(expression)
        );

        let mut metadata = StateMetadata::default();
        let ob = metadata.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = metadata.add_set_variable(String::from("sv"), ob);
        assert!(v.is_ok());
        let v = v.unwrap();
        assert_eq!(
            v.len(),
            IntegerExpression::Cardinality(SetExpression::Reference(
                ReferenceExpression::Variable(v.id())
            ))
        );
    }

    #[test]
    fn table_element() {
        let mut registry = TableRegistry::default();

        let t = registry.add_table_1d(String::from("t1"), vec![0, 1]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(
            Table1DHandle::<Integer>::element(&t, 0),
            IntegerExpression::Table(Box::new(NumericTableExpression::Table1D(
                t.id(),
                ElementExpression::Constant(0)
            )))
        );
        assert_eq!(
            t.sum(SetExpression::default()),
            IntegerExpression::Table(Box::new(NumericTableExpression::Table1DReduce(
                ReduceOperator::Sum,
                t.id(),
                SetExpression::default(),
            )))
        );
        assert_eq!(
            t.product(SetExpression::default()),
            IntegerExpression::Table(Box::new(NumericTableExpression::Table1DReduce(
                ReduceOperator::Product,
                t.id(),
                SetExpression::default(),
            )))
        );
        assert_eq!(
            t.max(SetExpression::default()),
            IntegerExpression::Table(Box::new(NumericTableExpression::Table1DReduce(
                ReduceOperator::Max,
                t.id(),
                SetExpression::default(),
            )))
        );
        assert_eq!(
            t.min(SetExpression::default()),
            IntegerExpression::Table(Box::new(NumericTableExpression::Table1DReduce(
                ReduceOperator::Min,
                t.id(),
                SetExpression::default(),
            )))
        );

        let t = registry.add_table_2d(String::from("t2"), vec![vec![0, 1]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(
            Table2DHandle::<Integer>::element(&t, 0, 0),
            IntegerExpression::Table(Box::new(NumericTableExpression::Table2D(
                t.id(),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
            )))
        );
        assert_eq!(
            t.sum_x(SetExpression::default(), 0),
            IntegerExpression::Table(Box::new(NumericTableExpression::Table2DReduceX(
                ReduceOperator::Sum,
                t.id(),
                SetExpression::default(),
                ElementExpression::Constant(0),
            )))
        );
        assert_eq!(
            t.sum_y(0, SetExpression::default()),
            IntegerExpression::Table(Box::new(NumericTableExpression::Table2DReduceY(
                ReduceOperator::Sum,
                t.id(),
                ElementExpression::Constant(0),
                SetExpression::default(),
            )))
        );
        assert_eq!(
            t.sum(SetExpression::default(), SetExpression::default()),
            IntegerExpression::Table(Box::new(NumericTableExpression::Table2DReduce(
                ReduceOperator::Sum,
                t.id(),
                SetExpression::default(),
                SetExpression::default(),
            )))
        );
        assert_eq!(
            t.product_x(SetExpression::default(), 0),
            IntegerExpression::Table(Box::new(NumericTableExpression::Table2DReduceX(
                ReduceOperator::Product,
                t.id(),
                SetExpression::default(),
                ElementExpression::Constant(0),
            )))
        );
        assert_eq!(
            t.product_y(0, SetExpression::default()),
            IntegerExpression::Table(Box::new(NumericTableExpression::Table2DReduceY(
                ReduceOperator::Product,
                t.id(),
                ElementExpression::Constant(0),
                SetExpression::default(),
            )))
        );
        assert_eq!(
            t.product(SetExpression::default(), SetExpression::default()),
            IntegerExpression::Table(Box::new(NumericTableExpression::Table2DReduce(
                ReduceOperator::Product,
                t.id(),
                SetExpression::default(),
                SetExpression::default(),
            )))
        );
        assert_eq!(
            t.max_x(SetExpression::default(), 0),
            IntegerExpression::Table(Box::new(NumericTableExpression::Table2DReduceX(
                ReduceOperator::Max,
                t.id(),
                SetExpression::default(),
                ElementExpression::Constant(0),
            )))
        );
        assert_eq!(
            t.max_y(0, SetExpression::default()),
            IntegerExpression::Table(Box::new(NumericTableExpression::Table2DReduceY(
                ReduceOperator::Max,
                t.id(),
                ElementExpression::Constant(0),
                SetExpression::default(),
            )))
        );
        assert_eq!(
            t.max(SetExpression::default(), SetExpression::default()),
            IntegerExpression::Table(Box::new(NumericTableExpression::Table2DReduce(
                ReduceOperator::Max,
                t.id(),
                SetExpression::default(),
                SetExpression::default(),
            )))
        );
        assert_eq!(
            t.min_x(SetExpression::default(), 0),
            IntegerExpression::Table(Box::new(NumericTableExpression::Table2DReduceX(
                ReduceOperator::Min,
                t.id(),
                SetExpression::default(),
                ElementExpression::Constant(0),
            )))
        );
        assert_eq!(
            t.min_y(0, SetExpression::default()),
            IntegerExpression::Table(Box::new(NumericTableExpression::Table2DReduceY(
                ReduceOperator::Min,
                t.id(),
                ElementExpression::Constant(0),
                SetExpression::default(),
            )))
        );
        assert_eq!(
            t.min(SetExpression::default(), SetExpression::default()),
            IntegerExpression::Table(Box::new(NumericTableExpression::Table2DReduce(
                ReduceOperator::Min,
                t.id(),
                SetExpression::default(),
                SetExpression::default(),
            )))
        );

        let t = registry.add_table_3d(String::from("t3"), vec![vec![vec![0, 1]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(
            Table3DHandle::<Integer>::element(&t, 0, 0, 0),
            IntegerExpression::Table(Box::new(NumericTableExpression::Table3D(
                t.id(),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
            )))
        );
        assert_eq!(
            Table3DHandle::<Integer>::sum(&t, 0, 0, SetExpression::default()),
            IntegerExpression::Table(Box::new(NumericTableExpression::Table3DReduce(
                ReduceOperator::Sum,
                t.id(),
                ArgumentExpression::Element(ElementExpression::Constant(0)),
                ArgumentExpression::Element(ElementExpression::Constant(0)),
                ArgumentExpression::Set(SetExpression::default()),
            )))
        );
        assert_eq!(
            Table3DHandle::<Integer>::product(&t, 0, 0, SetExpression::default()),
            IntegerExpression::Table(Box::new(NumericTableExpression::Table3DReduce(
                ReduceOperator::Product,
                t.id(),
                ArgumentExpression::Element(ElementExpression::Constant(0)),
                ArgumentExpression::Element(ElementExpression::Constant(0)),
                ArgumentExpression::Set(SetExpression::default()),
            )))
        );
        assert_eq!(
            Table3DHandle::<Integer>::max(&t, 0, 0, SetExpression::default()),
            IntegerExpression::Table(Box::new(NumericTableExpression::Table3DReduce(
                ReduceOperator::Max,
                t.id(),
                ArgumentExpression::Element(ElementExpression::Constant(0)),
                ArgumentExpression::Element(ElementExpression::Constant(0)),
                ArgumentExpression::Set(SetExpression::default()),
            )))
        );
        assert_eq!(
            Table3DHandle::<Integer>::min(&t, 0, 0, SetExpression::default()),
            IntegerExpression::Table(Box::new(NumericTableExpression::Table3DReduce(
                ReduceOperator::Min,
                t.id(),
                ArgumentExpression::Element(ElementExpression::Constant(0)),
                ArgumentExpression::Element(ElementExpression::Constant(0)),
                ArgumentExpression::Set(SetExpression::default()),
            )))
        );

        let mut map = FxHashMap::default();
        map.insert(vec![0, 0, 0, 0], 1);
        let t = registry.add_table(String::from("t"), map, 0);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(
            TableHandle::<Integer>::element(&t, vec![0, 0, 0, 0]),
            IntegerExpression::Table(Box::new(NumericTableExpression::Table(
                t.id(),
                vec![
                    ElementExpression::Constant(0),
                    ElementExpression::Constant(0),
                    ElementExpression::Constant(0),
                    ElementExpression::Constant(0),
                ],
            )))
        );
        assert_eq!(
            TableHandle::<Integer>::sum(
                &t,
                vec![
                    ArgumentExpression::from(0),
                    ArgumentExpression::from(0),
                    ArgumentExpression::from(0),
                    ArgumentExpression::from(Set::default())
                ]
            ),
            IntegerExpression::Table(Box::new(NumericTableExpression::TableReduce(
                ReduceOperator::Sum,
                t.id(),
                vec![
                    ArgumentExpression::Element(ElementExpression::Constant(0)),
                    ArgumentExpression::Element(ElementExpression::Constant(0)),
                    ArgumentExpression::Element(ElementExpression::Constant(0)),
                    ArgumentExpression::Set(SetExpression::default()),
                ]
            )))
        );
        assert_eq!(
            TableHandle::<Integer>::product(
                &t,
                vec![
                    ArgumentExpression::from(0),
                    ArgumentExpression::from(0),
                    ArgumentExpression::from(0),
                    ArgumentExpression::from(Set::default())
                ]
            ),
            IntegerExpression::Table(Box::new(NumericTableExpression::TableReduce(
                ReduceOperator::Product,
                t.id(),
                vec![
                    ArgumentExpression::Element(ElementExpression::Constant(0)),
                    ArgumentExpression::Element(ElementExpression::Constant(0)),
                    ArgumentExpression::Element(ElementExpression::Constant(0)),
                    ArgumentExpression::Set(SetExpression::default()),
                ]
            )))
        );
        assert_eq!(
            TableHandle::<Integer>::max(
                &t,
                vec![
                    ArgumentExpression::from(0),
                    ArgumentExpression::from(0),
                    ArgumentExpression::from(0),
                    ArgumentExpression::from(Set::default())
                ]
            ),
            IntegerExpression::Table(Box::new(NumericTableExpression::TableReduce(
                ReduceOperator::Max,
                t.id(),
                vec![
                    ArgumentExpression::Element(ElementExpression::Constant(0)),
                    ArgumentExpression::Element(ElementExpression::Constant(0)),
                    ArgumentExpression::Element(ElementExpression::Constant(0)),
                    ArgumentExpression::Set(SetExpression::default()),
                ]
            )))
        );
        assert_eq!(
            TableHandle::<Integer>::min(
                &t,
                vec![
                    ArgumentExpression::from(0),
                    ArgumentExpression::from(0),
                    ArgumentExpression::from(0),
                    ArgumentExpression::from(Set::default())
                ]
            ),
            IntegerExpression::Table(Box::new(NumericTableExpression::TableReduce(
                ReduceOperator::Min,
                t.id(),
                vec![
                    ArgumentExpression::Element(ElementExpression::Constant(0)),
                    ArgumentExpression::Element(ElementExpression::Constant(0)),
                    ArgumentExpression::Element(ElementExpression::Constant(0)),
                    ArgumentExpression::Set(SetExpression::default()),
                ]
            )))
        );
    }

    #[test]
    fn if_then_else() {
        let mut metadata = StateMetadata::default();
        let v1 = metadata.add_integer_variable(String::from("iv1"));
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let v2 = metadata.add_integer_variable(String::from("iv2"));
        assert!(v2.is_ok());
        let v2 = v2.unwrap();
        let rv1 = metadata.add_integer_resource_variable(String::from("irv1"), true);
        assert!(rv1.is_ok());
        let rv1 = rv1.unwrap();
        let rv2 = metadata.add_integer_resource_variable(String::from("irv2"), false);
        assert!(rv2.is_ok());
        let rv2 = rv2.unwrap();

        let condition = Condition::Constant(true);
        let expression1 = IntegerExpression::Constant(1);
        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            IfThenElse::<IntegerExpression>::if_then_else(
                condition.clone(),
                expression1,
                expression2
            ),
            IntegerExpression::If(
                Box::new(condition),
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::Constant(2))
            )
        );

        let condition = Condition::Constant(true);
        let expression1 = IntegerExpression::Constant(1);
        assert_eq!(
            IfThenElse::<IntegerExpression>::if_then_else(condition.clone(), expression1, 2),
            IntegerExpression::If(
                Box::new(condition),
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::Constant(2))
            )
        );

        let condition = Condition::Constant(true);
        let expression1 = IntegerExpression::Constant(1);
        assert_eq!(
            IfThenElse::<IntegerExpression>::if_then_else(condition.clone(), expression1, v2),
            IntegerExpression::If(
                Box::new(condition),
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::Variable(v2.id()))
            )
        );

        let condition = Condition::Constant(true);
        let expression1 = IntegerExpression::Constant(1);
        assert_eq!(
            IfThenElse::<IntegerExpression>::if_then_else(condition.clone(), expression1, rv2),
            IntegerExpression::If(
                Box::new(condition),
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::ResourceVariable(rv2.id()))
            )
        );

        let condition = Condition::Constant(true);
        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            IfThenElse::<IntegerExpression>::if_then_else(condition.clone(), 1, expression2),
            IntegerExpression::If(
                Box::new(condition),
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::Constant(2)),
            )
        );

        let condition = Condition::Constant(true);
        assert_eq!(
            IfThenElse::<IntegerExpression>::if_then_else(condition.clone(), 1, v2),
            IntegerExpression::If(
                Box::new(condition),
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::Variable(v2.id())),
            )
        );

        let condition = Condition::Constant(true);
        assert_eq!(
            IfThenElse::<IntegerExpression>::if_then_else(condition.clone(), 1, rv2),
            IntegerExpression::If(
                Box::new(condition),
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::ResourceVariable(rv2.id())),
            )
        );

        let condition = Condition::Constant(true);
        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            IfThenElse::<IntegerExpression>::if_then_else(condition.clone(), v1, expression2),
            IntegerExpression::If(
                Box::new(condition),
                Box::new(IntegerExpression::Variable(v1.id())),
                Box::new(IntegerExpression::Constant(2)),
            )
        );

        let condition = Condition::Constant(true);
        assert_eq!(
            IfThenElse::<IntegerExpression>::if_then_else(condition.clone(), v1, 2),
            IntegerExpression::If(
                Box::new(condition),
                Box::new(IntegerExpression::Variable(v1.id())),
                Box::new(IntegerExpression::Constant(2)),
            )
        );

        let condition = Condition::Constant(true);
        assert_eq!(
            IfThenElse::<IntegerExpression>::if_then_else(condition.clone(), v1, v2),
            IntegerExpression::If(
                Box::new(condition),
                Box::new(IntegerExpression::Variable(v1.id())),
                Box::new(IntegerExpression::Variable(v2.id())),
            )
        );

        let condition = Condition::Constant(true);
        assert_eq!(
            IfThenElse::<IntegerExpression>::if_then_else(condition.clone(), v1, rv2),
            IntegerExpression::If(
                Box::new(condition),
                Box::new(IntegerExpression::Variable(v1.id())),
                Box::new(IntegerExpression::ResourceVariable(rv2.id())),
            )
        );

        let condition = Condition::Constant(true);
        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            IfThenElse::<IntegerExpression>::if_then_else(condition.clone(), rv1, expression2),
            IntegerExpression::If(
                Box::new(condition),
                Box::new(IntegerExpression::ResourceVariable(rv1.id())),
                Box::new(IntegerExpression::Constant(2)),
            )
        );

        let condition = Condition::Constant(true);
        assert_eq!(
            IfThenElse::<IntegerExpression>::if_then_else(condition.clone(), rv1, 2),
            IntegerExpression::If(
                Box::new(condition),
                Box::new(IntegerExpression::ResourceVariable(rv1.id())),
                Box::new(IntegerExpression::Constant(2)),
            )
        );

        let condition = Condition::Constant(true);
        assert_eq!(
            IfThenElse::<IntegerExpression>::if_then_else(condition.clone(), rv1, v2),
            IntegerExpression::If(
                Box::new(condition),
                Box::new(IntegerExpression::ResourceVariable(rv1.id())),
                Box::new(IntegerExpression::Variable(v2.id())),
            )
        );

        let condition = Condition::Constant(true);
        assert_eq!(
            IfThenElse::<IntegerExpression>::if_then_else(condition.clone(), rv1, rv2),
            IntegerExpression::If(
                Box::new(condition),
                Box::new(IntegerExpression::ResourceVariable(rv1.id())),
                Box::new(IntegerExpression::ResourceVariable(rv2.id())),
            )
        );
    }

    #[test]
    fn from_continuous() {
        assert_eq!(
            IntegerExpression::floor(2.5),
            IntegerExpression::FromContinuous(
                CastOperator::Floor,
                Box::new(ContinuousExpression::Constant(2.5))
            )
        );
        assert_eq!(
            IntegerExpression::ceil(2.5),
            IntegerExpression::FromContinuous(
                CastOperator::Ceil,
                Box::new(ContinuousExpression::Constant(2.5))
            )
        );
        assert_eq!(
            IntegerExpression::round(2.5),
            IntegerExpression::FromContinuous(
                CastOperator::Round,
                Box::new(ContinuousExpression::Constant(2.5))
            )
        );
        assert_eq!(
            IntegerExpression::trunc(2.5),
            IntegerExpression::FromContinuous(
                CastOperator::Trunc,
                Box::new(ContinuousExpression::Constant(2.5))
            )
        );
    }

    #[test]
    fn constant_eval() {
        let state = State::default();
        let registry = TableRegistry::default();
        let expression = IntegerExpression::Constant(0);
        assert_eq!(expression.eval(&state, &registry), 0);
    }

    #[test]
    fn variable_eval() {
        let state = State {
            signature_variables: SignatureVariables {
                integer_variables: vec![0],
                ..Default::default()
            },
            ..Default::default()
        };
        let registry = TableRegistry::default();
        let expression = IntegerExpression::Variable(0);
        assert_eq!(expression.eval(&state, &registry), 0);
    }

    #[test]
    fn resource_variable_eval() {
        let state = State {
            resource_variables: ResourceVariables {
                integer_variables: vec![0],
                ..Default::default()
            },
            ..Default::default()
        };
        let registry = TableRegistry::default();
        let expression = IntegerExpression::ResourceVariable(0);
        assert_eq!(expression.eval(&state, &registry), 0);
    }

    #[test]
    fn eval_cost() {
        let state = State::default();
        let registry = TableRegistry::default();
        let expression = IntegerExpression::Cost;
        assert_eq!(expression.eval_cost(10, &state, &registry), 10);
    }

    #[test]
    #[should_panic]
    fn eval_cost_panic() {
        let state = State::default();
        let registry = TableRegistry::default();
        let expression = IntegerExpression::Cost;
        expression.eval(&state, &registry);
    }

    #[test]
    fn unary_operation_eval() {
        let state = State::default();
        let registry = TableRegistry::default();
        let expression = IntegerExpression::UnaryOperation(
            UnaryOperator::Abs,
            Box::new(IntegerExpression::Constant(-1)),
        );
        assert_eq!(expression.eval(&state, &registry), 1);
    }

    #[test]
    fn binary_operation_eval() {
        let state = State::default();
        let registry = TableRegistry::default();
        let expression = IntegerExpression::BinaryOperation(
            BinaryOperator::Add,
            Box::new(IntegerExpression::Constant(1)),
            Box::new(IntegerExpression::Constant(2)),
        );
        assert_eq!(expression.eval(&state, &registry), 3);
    }

    #[test]
    fn cardinality_eval() {
        let state = State::default();
        let registry = TableRegistry::default();
        let mut set = Set::with_capacity(5);
        set.insert(1);
        set.insert(4);
        let expression = IntegerExpression::Cardinality(SetExpression::Reference(
            ReferenceExpression::Constant(set),
        ));
        assert_eq!(expression.eval(&state, &registry), 2);
    }

    #[test]
    fn length_eval() {
        let state = State::default();
        let registry = TableRegistry::default();
        let expression = IntegerExpression::Length(VectorExpression::Reference(
            ReferenceExpression::Constant(vec![1, 4]),
        ));
        assert_eq!(expression.eval(&state, &registry), 2);
    }

    #[test]
    fn table_eval() {
        let state = State::default();
        let registry = TableRegistry::default();
        let expression = IntegerExpression::Table(Box::new(NumericTableExpression::Constant(0)));
        assert_eq!(expression.eval(&state, &registry), 0);
    }

    #[test]
    fn if_eval() {
        let state = State::default();
        let registry = TableRegistry::default();

        let expression = IntegerExpression::If(
            Box::new(Condition::Constant(true)),
            Box::new(IntegerExpression::Constant(1)),
            Box::new(IntegerExpression::Constant(2)),
        );
        assert_eq!(expression.eval(&state, &registry), 1);

        let expression = IntegerExpression::If(
            Box::new(Condition::Constant(false)),
            Box::new(IntegerExpression::Constant(1)),
            Box::new(IntegerExpression::Constant(2)),
        );
        assert_eq!(expression.eval(&state, &registry), 2);
    }

    #[test]
    fn from_continuous_eval() {
        let state = State::default();
        let registry = TableRegistry::default();

        let expression = IntegerExpression::FromContinuous(
            CastOperator::Floor,
            Box::new(ContinuousExpression::Constant(1.5)),
        );
        assert_eq!(expression.eval(&state, &registry), 1);
    }

    #[test]
    fn last_eval() {
        let state = State::default();
        let registry = TableRegistry::default();

        let expression =
            IntegerExpression::Last(Box::new(IntegerVectorExpression::Constant(vec![1, 2, 3])));
        assert_eq!(expression.eval(&state, &registry), 3);
    }

    #[test]
    fn at_eval() {
        let state = State::default();
        let registry = TableRegistry::default();

        let expression = IntegerExpression::At(
            Box::new(IntegerVectorExpression::Constant(vec![1, 2, 3])),
            ElementExpression::Constant(0),
        );
        assert_eq!(expression.eval(&state, &registry), 1);
    }

    #[test]
    fn reduce_eval() {
        let state = State::default();
        let registry = TableRegistry::default();

        let expression = IntegerExpression::Reduce(
            ReduceOperator::Sum,
            Box::new(IntegerVectorExpression::Constant(vec![1, 2, 3])),
        );
        assert_eq!(expression.eval(&state, &registry), 6);
    }

    #[test]
    fn constant_simplify() {
        let registry = TableRegistry::default();
        let expression = IntegerExpression::Constant(1);
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn variable_simplify() {
        let registry = TableRegistry::default();
        let expression = IntegerExpression::Variable(0);
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn resource_variable_simplify() {
        let registry = TableRegistry::default();
        let expression = IntegerExpression::ResourceVariable(0);
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn cost_simplify() {
        let registry = TableRegistry::default();
        let expression = IntegerExpression::Cost;
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn unary_operation_simplify() {
        let registry = TableRegistry::default();
        let expression = IntegerExpression::UnaryOperation(
            UnaryOperator::Abs,
            Box::new(IntegerExpression::Constant(-1)),
        );
        assert_eq!(
            expression.simplify(&registry),
            IntegerExpression::Constant(1)
        );
        let expression = IntegerExpression::UnaryOperation(
            UnaryOperator::Abs,
            Box::new(IntegerExpression::Variable(0)),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn binary_operation_simplify() {
        let registry = TableRegistry::default();
        let expression = IntegerExpression::BinaryOperation(
            BinaryOperator::Add,
            Box::new(IntegerExpression::Constant(1)),
            Box::new(IntegerExpression::Constant(2)),
        );
        assert_eq!(
            expression.simplify(&registry),
            IntegerExpression::Constant(3)
        );
        let expression = IntegerExpression::BinaryOperation(
            BinaryOperator::Add,
            Box::new(IntegerExpression::Variable(0)),
            Box::new(IntegerExpression::Constant(2)),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn cardinality_simplify() {
        let registry = TableRegistry::default();
        let mut set = Set::with_capacity(5);
        set.insert(1);
        set.insert(4);
        let expression = IntegerExpression::Cardinality(SetExpression::Reference(
            ReferenceExpression::Constant(set),
        ));
        assert_eq!(
            expression.simplify(&registry),
            IntegerExpression::Constant(2)
        );
        let expression = IntegerExpression::Cardinality(SetExpression::Reference(
            ReferenceExpression::Variable(0),
        ));
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn length_simplify() {
        let registry = TableRegistry::default();
        let expression = IntegerExpression::Length(VectorExpression::Reference(
            ReferenceExpression::Constant(vec![1, 4]),
        ));
        assert_eq!(
            expression.simplify(&registry),
            IntegerExpression::Constant(2)
        );
        let expression = IntegerExpression::Length(VectorExpression::Reference(
            ReferenceExpression::Variable(0),
        ));
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn table_simplify() {
        let registry = TableRegistry::default();
        let expression = IntegerExpression::Table(Box::new(NumericTableExpression::Constant(0)));
        assert_eq!(
            expression.simplify(&registry),
            IntegerExpression::Constant(0)
        );
        let expression = IntegerExpression::Table(Box::new(NumericTableExpression::Table1D(
            0,
            ElementExpression::Variable(0),
        )));
        assert_eq!(expression.simplify(&registry), expression,);
    }

    #[test]
    fn if_simplify() {
        let registry = TableRegistry::default();

        let expression = IntegerExpression::If(
            Box::new(Condition::Constant(true)),
            Box::new(IntegerExpression::Constant(1)),
            Box::new(IntegerExpression::Constant(2)),
        );
        assert_eq!(
            expression.simplify(&registry),
            IntegerExpression::Constant(1)
        );

        let expression = IntegerExpression::If(
            Box::new(Condition::Constant(false)),
            Box::new(IntegerExpression::Constant(1)),
            Box::new(IntegerExpression::Constant(2)),
        );
        assert_eq!(
            expression.simplify(&registry),
            IntegerExpression::Constant(2)
        );

        let expression = IntegerExpression::If(
            Box::new(Condition::Table(Box::new(TableExpression::Table1D(
                0,
                ElementExpression::Variable(0),
            )))),
            Box::new(IntegerExpression::Constant(1)),
            Box::new(IntegerExpression::Constant(2)),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn from_continuous_simplify() {
        let registry = TableRegistry::default();

        let expression = IntegerExpression::FromContinuous(
            CastOperator::Floor,
            Box::new(ContinuousExpression::Constant(1.5)),
        );
        assert_eq!(
            expression.simplify(&registry),
            IntegerExpression::Constant(1)
        );

        let expression = IntegerExpression::FromContinuous(
            CastOperator::Floor,
            Box::new(ContinuousExpression::Variable(0)),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn last_simplify() {
        let registry = TableRegistry::default();

        let expression =
            IntegerExpression::Last(Box::new(IntegerVectorExpression::Constant(vec![1, 2, 3])));
        assert_eq!(
            expression.simplify(&registry),
            IntegerExpression::Constant(3)
        );

        let expression = IntegerExpression::Last(Box::new(IntegerVectorExpression::Table(
            Box::new(TableVectorExpression::Table1D(
                0,
                VectorExpression::Reference(ReferenceExpression::Variable(0)),
            )),
        )));
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn at_simplify() {
        let registry = TableRegistry::default();

        let expression = IntegerExpression::At(
            Box::new(IntegerVectorExpression::Constant(vec![1, 2, 3])),
            ElementExpression::Constant(0),
        );
        assert_eq!(
            expression.simplify(&registry),
            IntegerExpression::Constant(1)
        );

        let expression = IntegerExpression::At(
            Box::new(IntegerVectorExpression::Table(Box::new(
                TableVectorExpression::Table1D(
                    0,
                    VectorExpression::Reference(ReferenceExpression::Variable(0)),
                ),
            ))),
            ElementExpression::Constant(0),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn reduce_simplify() {
        let registry = TableRegistry::default();

        let expression = IntegerExpression::Reduce(
            ReduceOperator::Sum,
            Box::new(IntegerVectorExpression::Constant(vec![1, 2, 3])),
        );
        assert_eq!(
            expression.simplify(&registry),
            IntegerExpression::Constant(6)
        );

        let expression = IntegerExpression::Reduce(
            ReduceOperator::Sum,
            Box::new(IntegerVectorExpression::Table(Box::new(
                TableVectorExpression::Table1D(
                    0,
                    VectorExpression::Reference(ReferenceExpression::Variable(0)),
                ),
            ))),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }
}
