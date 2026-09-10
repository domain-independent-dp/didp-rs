use super::algorithms;
use super::argument_expression::ArgumentExpression;
use super::condition::{Condition, IfThenElse};
use super::continuous_expression::ContinuousExpression;
use super::element_expression::ElementExpression;
use super::numeric_operator::{
    BinaryOperator, CastOperator, MaxMin, ReduceOperator, UnaryOperator,
};
use super::numeric_table_expression::NumericTableExpression;
use super::reference_expression::ReferenceExpression;
use super::set_expression::SetExpression;
use super::substitute_local_variable::SubstituteLocalVariable;
use super::LocalEnvironment;
use crate::local_variable::LocalVariable;
use crate::state::{
    IntegerResourceVariable, IntegerVariable, SetResourceVariable, SetVariable, StateInterface,
};
use crate::state_functions::{StateFunctionCache, StateFunctions};
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
    /// State function index.
    StateFunction(usize),
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
    /// A constant in an integer table.
    Table(Box<NumericTableExpression<Integer>>),
    /// The minimum spanning tree cost over a set expression using a 2D integer table as edge costs.
    MinimumSpanningTree(Box<SetExpression>, usize),
    /// The minimum spanning tree cost over a set expression using a 2D integer table as edge costs and a 2D boolean table as edge connectivity.
    MinimumSpanningTreeWithConnectivity(Box<SetExpression>, usize, usize),
    /// The minimum spanning tree cost over a set expression using an explicit list of edges,
    /// each given as a node pair and a cost expression.
    MinimumSpanningTreeWithEdges(Box<SetExpression>, Vec<(usize, usize, IntegerExpression)>),
    /// The minimum spanning tree cost over a set expression using an explicit list of edges,
    /// each given as a node pair, a cost expression, and a condition for the edge to be present.
    MinimumSpanningTreeWithEdgesAndConnectivity(
        Box<SetExpression>,
        Vec<(usize, usize, IntegerExpression, Condition)>,
    ),
    /// The minimum spanning tree cost over a set expression using pre-sorted edge costs.
    MinimumSpanningTreeWithSortedEdges(Box<SetExpression>, Vec<(usize, usize, Integer)>),
    /// If-then-else expression, which returns the first one if the condition holds and the second one otherwise.
    If(
        Box<Condition>,
        Box<IntegerExpression>,
        Box<IntegerExpression>,
    ),
    /// Conversion from a continuous expression.
    FromContinuous(CastOperator, Box<ContinuousExpression>),
    /// Reduce operation over a set expression.
    Reduce(
        ReduceOperator,
        Box<SetExpression>,
        usize,
        Box<IntegerExpression>,
    ),
    /// Reduce operation over a set expression filtered with a condition.
    #[doc(hidden)]
    FilterReduce(
        ReduceOperator,
        Box<SetExpression>,
        usize,
        usize,
        Box<Condition>,
        Box<IntegerExpression>,
    ),
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
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    /// let expression = IntegerExpression::from(-1);
    /// let expression = expression.abs();
    ///
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     1,
    /// );
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
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    /// let expression = IntegerExpression::from(1);
    /// let expression = -expression;
    ///
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     -1,
    /// );
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
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let model = Model::default();
    /// let state = model.target.clone();
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    /// let a = IntegerExpression::from(2);
    /// let b = IntegerExpression::from(3);
    /// let expression = a + b;
    ///
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     5,
    /// );
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
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let model = Model::default();
    /// let state = model.target.clone();
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    /// let a = IntegerExpression::from(2);
    /// let b = IntegerExpression::from(3);
    /// let expression = a - b;
    ///
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     -1,
    /// );
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
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let model = Model::default();
    /// let state = model.target.clone();
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    /// let a = IntegerExpression::from(2);
    /// let b = IntegerExpression::from(3);
    /// let expression = a * b;
    ///
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     6,
    /// );
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
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let model = Model::default();
    /// let state = model.target.clone();
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    /// let a = IntegerExpression::from(2);
    /// let b = IntegerExpression::from(3);
    /// let expression = a / b;
    ///
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     0,
    /// );
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
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let model = Model::default();
    /// let state = model.target.clone();
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    /// let a = IntegerExpression::from(2);
    /// let b = IntegerExpression::from(3);
    /// let expression = a % b;
    ///
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     2,
    /// );
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
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let model = Model::default();
    /// let state = model.target.clone();
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    /// let a = IntegerExpression::from(2);
    /// let b = IntegerExpression::from(3);
    /// let expression = a.max(b);
    ///
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     3,
    /// );
    /// ```
    #[inline]
    fn max(self, rhs: Self) -> Self::Output {
        IntegerExpression::BinaryOperation(BinaryOperator::Max, Box::new(self), Box::new(rhs))
    }

    /// Returns an expression representing the minimum.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let model = Model::default();
    /// let state = model.target.clone();
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    /// let a = IntegerExpression::from(2);
    /// let b = IntegerExpression::from(3);
    /// let expression = a.min(b);
    ///
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     2,
    /// );
    /// ```
    #[inline]
    fn min(self, rhs: Self) -> Self::Output {
        IntegerExpression::BinaryOperation(BinaryOperator::Min, Box::new(self), Box::new(rhs))
    }
}

impl SetExpression {
    /// Returns an expression representing the cardinality of a set.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let object_type = model.add_object_type("object", 4).unwrap();
    /// let set = model.create_set(object_type, &[0, 1]).unwrap();
    /// let state = model.target.clone();
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let expression = SetExpression::from(set);
    /// let expression = expression.len();
    ///
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     2,
    /// );
    /// ```
    #[inline]
    pub fn len(self) -> IntegerExpression {
        IntegerExpression::Cardinality(self)
    }

    /// Returns an expression representing the sum over a set.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let object_type = model.add_object_type("object", 4).unwrap();
    /// let set = model.create_set(object_type, &[0, 1]).unwrap();
    /// let x = model.add_local_variable("x").unwrap();
    /// let table = model.add_table_1d("table", vec![2, 3]).unwrap();
    /// let state = model.target.clone();
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let set = SetExpression::from(set);
    /// let expression = set.sum(
    ///     x,
    ///     Table1DHandle::<Integer>::element(&table, x) * Table1DHandle::<Integer>::element(&table, x),
    /// );
    ///
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     13,
    /// );
    /// ```
    #[inline]
    pub fn sum(self, x: LocalVariable, f: IntegerExpression) -> IntegerExpression {
        IntegerExpression::Reduce(ReduceOperator::Sum, Box::new(self), x.id(), Box::new(f))
    }

    /// Returns an expression representing the product over a set.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let object_type = model.add_object_type("object", 4).unwrap();
    /// let set = model.create_set(object_type, &[0, 1]).unwrap();
    /// let x = model.add_local_variable("x").unwrap();
    /// let table = model.add_table_1d("table", vec![2, 3]).unwrap();
    /// let state = model.target.clone();
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let set = SetExpression::from(set);
    /// let expression = set.product(
    ///     x,
    ///     Table1DHandle::<Integer>::element(&table, x) * Table1DHandle::<Integer>::element(&table, x),
    /// );
    ///
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     36,
    /// );
    /// ```
    #[inline]
    pub fn product(self, x: LocalVariable, f: IntegerExpression) -> IntegerExpression {
        IntegerExpression::Reduce(ReduceOperator::Product, Box::new(self), x.id(), Box::new(f))
    }

    /// Returns an expression representing the maximum over a set.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let object_type = model.add_object_type("object", 4).unwrap();
    /// let set = model.create_set(object_type, &[0, 1]).unwrap();
    /// let x = model.add_local_variable("x").unwrap();
    /// let table = model.add_table_1d("table", vec![2, 3]).unwrap();
    /// let state = model.target.clone();
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let set = SetExpression::from(set);
    /// let expression = set.max(
    ///     x,
    ///     Table1DHandle::<Integer>::element(&table, x) * Table1DHandle::<Integer>::element(&table, x),
    /// );
    ///
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     9,
    /// );
    /// ```
    #[inline]
    pub fn max(self, x: LocalVariable, f: IntegerExpression) -> IntegerExpression {
        IntegerExpression::Reduce(ReduceOperator::Max, Box::new(self), x.id(), Box::new(f))
    }

    /// Returns an expression representing the minimum over a set.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let object_type = model.add_object_type("object", 4).unwrap();
    /// let set = model.create_set(object_type, &[0, 1]).unwrap();
    /// let x = model.add_local_variable("x").unwrap();
    /// let table = model.add_table_1d("table", vec![2, 3]).unwrap();
    /// let state = model.target.clone();
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let set = SetExpression::from(set);
    /// let expression = set.min(
    ///     x,
    ///     Table1DHandle::<Integer>::element(&table, x) * Table1DHandle::<Integer>::element(&table, x),
    /// );
    ///
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     4,
    /// );
    /// ```
    #[inline]
    pub fn min(self, x: LocalVariable, f: IntegerExpression) -> IntegerExpression {
        IntegerExpression::Reduce(ReduceOperator::Min, Box::new(self), x.id(), Box::new(f))
    }
}

impl SetVariable {
    /// Returns an expression representing the cardinality of a set.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let object_type = model.add_object_type("object", 4).unwrap();
    /// let set = model.create_set(object_type, &[0, 1]).unwrap();
    /// let variable = model.add_set_variable("variable", object_type, set).unwrap();
    /// let state = model.target.clone();
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let expression = variable.len();
    ///
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     2,
    /// );
    /// ```
    #[inline]
    pub fn len(self) -> IntegerExpression {
        IntegerExpression::Cardinality(SetExpression::from(self))
    }

    /// Returns an expression representing the sum over a set.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let object_type = model.add_object_type("object", 4).unwrap();
    /// let set = model.create_set(object_type, &[0, 1]).unwrap();
    /// let variable = model.add_set_variable("variable", object_type, set).unwrap();
    /// let x = model.add_local_variable("x").unwrap();
    /// let table = model.add_table_1d("table", vec![2, 3]).unwrap();
    /// let state = model.target.clone();
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let expression = variable.sum(
    ///     x,
    ///     Table1DHandle::<Integer>::element(&table, x) * Table1DHandle::<Integer>::element(&table, x),
    /// );
    ///
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     13,
    /// );
    /// ```
    #[inline]
    pub fn sum(self, x: LocalVariable, g: IntegerExpression) -> IntegerExpression {
        SetExpression::from(self).sum(x, g)
    }

    /// Returns an expression representing the product over a set.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let object_type = model.add_object_type("object", 4).unwrap();
    /// let set = model.create_set(object_type, &[0, 1]).unwrap();
    /// let variable = model.add_set_variable("variable", object_type, set).unwrap();
    /// let x = model.add_local_variable("x").unwrap();
    /// let table = model.add_table_1d("table", vec![2, 3]).unwrap();
    /// let state = model.target.clone();
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let expression = variable.product(
    ///     x,
    ///     Table1DHandle::<Integer>::element(&table, x) * Table1DHandle::<Integer>::element(&table, x),
    /// );
    ///
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     36,
    /// );
    /// ```
    #[inline]
    pub fn product(self, x: LocalVariable, f: IntegerExpression) -> IntegerExpression {
        SetExpression::from(self).product(x, f)
    }

    /// Returns an expression representing the maximum over a set.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let object_type = model.add_object_type("object", 4).unwrap();
    /// let set = model.create_set(object_type, &[0, 1]).unwrap();
    /// let x = model.add_local_variable("x").unwrap();
    /// let variable = model.add_set_variable("variable", object_type, set).unwrap();
    /// let table = model.add_table_1d("table", vec![2, 3]).unwrap();
    /// let state = model.target.clone();
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let expression = variable.max(
    ///     x,
    ///     Table1DHandle::<Integer>::element(&table, x) * Table1DHandle::<Integer>::element(&table, x),
    /// );
    ///
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     9,
    /// );
    /// ```
    #[inline]
    pub fn max(self, x: LocalVariable, f: IntegerExpression) -> IntegerExpression {
        SetExpression::from(self).max(x, f)
    }

    /// Returns an expression representing the minimum over a set.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let object_type = model.add_object_type("object", 4).unwrap();
    /// let set = model.create_set(object_type, &[0, 1]).unwrap();
    /// let variable = model.add_set_variable("variable", object_type, set).unwrap();
    /// let x = model.add_local_variable("x").unwrap();
    /// let table = model.add_table_1d("table", vec![2, 3]).unwrap();
    /// let state = model.target.clone();
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let expression = variable.min(
    ///     x,
    ///     Table1DHandle::<Integer>::element(&table, x) * Table1DHandle::<Integer>::element(&table, x),
    /// );
    ///
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     4,
    /// );
    /// ```
    #[inline]
    pub fn min(self, x: LocalVariable, f: IntegerExpression) -> IntegerExpression {
        SetExpression::from(self).min(x, f)
    }
}

impl SetResourceVariable {
    /// Returns an expression representing the cardinality of a set.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let object_type = model.add_object_type("object", 4).unwrap();
    /// let set = model.create_set(object_type, &[0, 1]).unwrap();
    /// let variable = model.add_set_resource_variable("variable", object_type, false, set).unwrap();
    /// let state = model.target.clone();
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let expression = variable.len();
    ///
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     2,
    /// );
    /// ```
    #[inline]
    pub fn len(self) -> IntegerExpression {
        IntegerExpression::Cardinality(SetExpression::from(self))
    }

    /// Returns an expression representing the sum over a set.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let object_type = model.add_object_type("object", 4).unwrap();
    /// let set = model.create_set(object_type, &[0, 1]).unwrap();
    /// let variable = model.add_set_resource_variable("variable", object_type, true, set).unwrap();
    /// let x = model.add_local_variable("x").unwrap();
    /// let table = model.add_table_1d("table", vec![2, 3]).unwrap();
    /// let state = model.target.clone();
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let expression = variable.sum(
    ///     x,
    ///     Table1DHandle::<Integer>::element(&table, x) * Table1DHandle::<Integer>::element(&table, x),
    /// );
    ///
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     13,
    /// );
    /// ```
    #[inline]
    pub fn sum(self, x: LocalVariable, f: IntegerExpression) -> IntegerExpression {
        SetExpression::from(self).sum(x, f)
    }

    /// Returns an expression representing the product over a set.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let object_type = model.add_object_type("object", 4).unwrap();
    /// let set = model.create_set(object_type, &[0, 1]).unwrap();
    /// let variable = model.add_set_resource_variable("variable", object_type, true, set).unwrap();
    /// let x = model.add_local_variable("x").unwrap();
    /// let table = model.add_table_1d("table", vec![2, 3]).unwrap();
    /// let state = model.target.clone();
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let expression = variable.product(
    ///     x,
    ///     Table1DHandle::<Integer>::element(&table, x) * Table1DHandle::<Integer>::element(&table, x),
    /// );
    ///
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     36,
    /// );
    /// ```
    #[inline]
    pub fn product(self, x: LocalVariable, f: IntegerExpression) -> IntegerExpression {
        SetExpression::from(self).product(x, f)
    }

    /// Returns an expression representing the maximum over a set.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let object_type = model.add_object_type("object", 4).unwrap();
    /// let set = model.create_set(object_type, &[0, 1]).unwrap();
    /// let x = model.add_local_variable("x").unwrap();
    /// let variable = model.add_set_resource_variable("variable", object_type, true, set).unwrap();
    /// let table = model.add_table_1d("table", vec![2, 3]).unwrap();
    /// let state = model.target.clone();
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let expression = variable.max(
    ///     x,
    ///     Table1DHandle::<Integer>::element(&table, x) * Table1DHandle::<Integer>::element(&table, x),
    /// );
    ///
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     9,
    /// );
    /// ```
    #[inline]
    pub fn max(self, x: LocalVariable, f: IntegerExpression) -> IntegerExpression {
        SetExpression::from(self).max(x, f)
    }

    /// Returns an expression representing the minimum over a set.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let object_type = model.add_object_type("object", 4).unwrap();
    /// let set = model.create_set(object_type, &[0, 1]).unwrap();
    /// let variable = model.add_set_resource_variable("variable", object_type, true, set).unwrap();
    /// let x = model.add_local_variable("x").unwrap();
    /// let table = model.add_table_1d("table", vec![2, 3]).unwrap();
    /// let state = model.target.clone();
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let expression = variable.min(
    ///     x,
    ///     Table1DHandle::<Integer>::element(&table, x) * Table1DHandle::<Integer>::element(&table, x),
    /// );
    ///
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     4,
    /// );
    /// ```
    #[inline]
    pub fn min(self, x: LocalVariable, f: IntegerExpression) -> IntegerExpression {
        SetExpression::from(self).min(x, f)
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
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let expression = Table1DHandle::<Integer>::element(&table, variable);
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     2,
    /// );
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
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let expression = table.sum(variable);
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     5,
    /// );
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
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let expression = table.product(variable);
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     6,
    /// );
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
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let expression = table.max(variable);
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     3,
    /// );
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
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let expression = table.min(variable);
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     2,
    /// );
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
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let expression = Table2DHandle::<Integer>::element(&table, variable, 1);
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     3,
    /// );
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

    /// Returns the cost of the minimum spanning tree over a set expression.
    ///
    /// The 2D table is interpreted as the complete graph edge-cost matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let table = model.add_table_2d(
    ///     "table",
    ///     vec![
    ///         vec![0, 1, 4, 3],
    ///         vec![1, 0, 2, 5],
    ///         vec![4, 2, 0, 6],
    ///         vec![3, 5, 6, 0],
    ///     ],
    /// ).unwrap();
    /// let object_type = model.add_object_type("object", 4).unwrap();
    /// let set = model.create_set(object_type, &[0, 1, 2, 3]).unwrap();
    /// let variable = model.add_set_variable("variable", object_type, set).unwrap();
    /// let state = model.target.clone();
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let expression = table.minimum_spanning_tree(variable);
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     6,
    /// );
    /// ```
    #[inline]
    pub fn minimum_spanning_tree<T>(&self, nodes: T) -> IntegerExpression
    where
        SetExpression: From<T>,
    {
        IntegerExpression::MinimumSpanningTree(Box::new(SetExpression::from(nodes)), self.id())
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
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let expression = table.sum_x(x, y);
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     6,
    /// );
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
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let expression = table.sum_y(x, y);
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,    
    ///     ),
    ///     5,
    /// );
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
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let expression = table.sum(x, y);
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     14,
    /// );
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
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let expression = table.product_x(x, y);
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     8,
    /// );
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
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let expression = table.product_y(x, y);
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     6,
    /// );
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
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let expression = table.product(x, y);
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     120,
    /// );
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
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let expression = table.max_x(x, y);
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     4,
    /// );
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
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let expression = table.max_y(x, y);
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     3,
    /// );
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
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let expression = table.max(x, y);
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     5,
    /// );
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
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let expression = table.min_x(x, y);
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     2,
    /// );
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
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let expression = table.min_y(x, y);
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,    
    ///     ),
    ///     2,
    /// );
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
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let expression = table.min(x, y);
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     2,
    /// );
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
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let expression = Table3DHandle::<Integer>::element(&table, variable, variable + 1, 1);
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     5,
    /// );
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
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let expression = table.sum(set_variable, element_variable, 1);
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     10,
    /// );
    ///
    /// let expression = table.sum(set, set_variable, set_variable);
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     44,
    /// );
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
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let expression = table.product(set_variable, element_variable, 1);
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     21,
    /// );
    ///
    /// let expression = table.product(set, set_variable, set_variable);
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     362880,
    /// );
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
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let expression = table.max(set_variable, element_variable, 1);
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     7,
    /// );
    ///
    /// let expression = table.max(set, set_variable, set_variable);
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     9,
    /// );
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
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let expression = table.min(set_variable, element_variable, 1);
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     3,
    /// );
    ///
    /// let expression = table.min(set, set_variable, set_variable);
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     2,
    /// );
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
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let indices = vec![
    ///     ElementExpression::from(variable),
    ///     ElementExpression::from(0),
    ///     ElementExpression::from(0),
    ///     ElementExpression::from(0),
    /// ];
    /// let expression = TableHandle::<Integer>::element(&table, indices);
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     1,
    /// );
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
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let indices = vec![
    ///     ArgumentExpression::from(set),
    ///     ArgumentExpression::from(set_variable),
    ///     ArgumentExpression::from(element_variable),
    ///     ArgumentExpression::from(0),
    /// ];
    /// let expression = table.sum(indices);
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     1,
    /// );
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
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let indices = vec![
    ///     ArgumentExpression::from(set),
    ///     ArgumentExpression::from(set_variable),
    ///     ArgumentExpression::from(element_variable),
    ///     ArgumentExpression::from(0),
    /// ];
    /// let expression = table.product(indices);
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     0,
    /// );
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
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let indices = vec![
    ///     ArgumentExpression::from(set),
    ///     ArgumentExpression::from(set_variable),
    ///     ArgumentExpression::from(element_variable),
    ///     ArgumentExpression::from(0),
    /// ];
    /// let expression = table.max(indices);
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     1,
    /// );
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
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let indices = vec![
    ///     ArgumentExpression::from(set),
    ///     ArgumentExpression::from(set_variable),
    ///     ArgumentExpression::from(element_variable),
    ///     ArgumentExpression::from(0),
    /// ];
    /// let expression = table.min(indices);
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     0,
    /// );
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
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let expression = ContinuousExpression::from(1.5);
    /// let expression = IntegerExpression::floor(expression);
    ///
    /// assert_eq!(
    ///     expression.eval(    
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    /// 1);
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
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let expression = ContinuousExpression::from(1.5);
    /// let expression = IntegerExpression::ceil(expression);
    ///
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     2,
    /// );
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
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let expression = ContinuousExpression::from(1.5);
    /// let expression = IntegerExpression::round(expression);
    ///
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     2,
    /// );
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
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let expression = ContinuousExpression::from(1.5);
    /// let expression = IntegerExpression::trunc(expression);
    ///
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     1,
    /// );
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
    /// Panics if the cost of the transition state is used, a min/max reduce operation is
    /// performed on an empty set, or a minimum spanning tree expression is evaluated on a
    /// disconnected graph.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let variable = model.add_integer_variable("variable", 1).unwrap();
    /// let state = model.target.clone();
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let expression = IntegerExpression::from(variable);
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,    
    ///     ),
    ///     1,
    /// );
    /// ```
    #[inline]
    pub fn eval<U: StateInterface>(
        &self,
        state: &U,
        function_cache: &mut StateFunctionCache,
        state_functions: &StateFunctions,
        registry: &TableRegistry,
    ) -> Integer {
        let mut local_environment = LocalEnvironment::default();

        self.eval_with_local_environment(
            state,
            function_cache,
            &mut local_environment,
            state_functions,
            registry,
        )
    }

    #[inline]
    pub fn eval_with_local_environment<U: StateInterface>(
        &self,
        state: &U,
        function_cache: &mut StateFunctionCache,
        local_environment: &mut LocalEnvironment,
        state_functions: &StateFunctions,
        registry: &TableRegistry,
    ) -> Integer {
        self.eval_inner(
            None,
            state,
            function_cache,
            local_environment,
            state_functions,
            registry,
        )
    }

    /// Returns the evaluation result of a cost expression.
    ///
    /// # Panics
    ///
    /// Panics if a min/max reduce operation is performed on an empty set or a minimum spanning
    /// tree expression is evaluated on a disconnected graph.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let model = Model::default();
    /// let state = model.target.clone();
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let expression = IntegerExpression::Cost + 1;
    /// assert_eq!(
    ///     expression.eval_cost(
    ///         1, &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     2,
    /// );
    /// ```
    #[inline]
    pub fn eval_cost<U: StateInterface>(
        &self,
        cost: Integer,
        state: &U,
        function_cache: &mut StateFunctionCache,
        state_functions: &StateFunctions,
        registry: &TableRegistry,
    ) -> Integer {
        let mut local_environment = LocalEnvironment::default();

        self.eval_cost_with_local_environment(
            cost,
            state,
            function_cache,
            &mut local_environment,
            state_functions,
            registry,
        )
    }

    #[inline]
    pub fn eval_cost_with_local_environment<U: StateInterface>(
        &self,
        cost: Integer,
        state: &U,
        function_cache: &mut StateFunctionCache,
        local_environment: &mut LocalEnvironment,
        state_functions: &StateFunctions,
        registry: &TableRegistry,
    ) -> Integer {
        self.eval_inner(
            Some(cost),
            state,
            function_cache,
            local_environment,
            state_functions,
            registry,
        )
    }

    fn eval_inner<U: StateInterface>(
        &self,
        cost: Option<Integer>,
        state: &U,
        function_cache: &mut StateFunctionCache,
        local_environment: &mut LocalEnvironment,
        state_functions: &StateFunctions,
        registry: &TableRegistry,
    ) -> Integer {
        match self {
            Self::Constant(x) => *x,
            Self::Variable(i) => state.get_integer_variable(*i),
            Self::ResourceVariable(i) => state.get_integer_resource_variable(*i),
            Self::StateFunction(i) => function_cache.get_integer_value(
                *i,
                state,
                local_environment,
                state_functions,
                registry,
            ),
            Self::Cost => cost.unwrap(),
            Self::UnaryOperation(op, x) => op.eval(x.eval_inner(
                cost,
                state,
                function_cache,
                local_environment,
                state_functions,
                registry,
            )),
            Self::BinaryOperation(op, a, b) => {
                let a = a.eval_inner(
                    cost,
                    state,
                    function_cache,
                    local_environment,
                    state_functions,
                    registry,
                );
                let b = b.eval_inner(
                    cost,
                    state,
                    function_cache,
                    local_environment,
                    state_functions,
                    registry,
                );
                op.eval(a, b)
            }
            Self::Cardinality(SetExpression::Reference(expression)) => {
                let set = expression.eval(
                    state,
                    function_cache,
                    local_environment,
                    state_functions,
                    registry,
                );
                set.count_ones(..) as Integer
            }
            Self::Cardinality(SetExpression::StateFunction(i)) => {
                let set = function_cache.get_set_value(
                    *i,
                    state,
                    local_environment,
                    state_functions,
                    registry,
                );
                set.count_ones(..) as Integer
            }
            Self::Cardinality(set) => set
                .eval_with_local_environment(
                    state,
                    function_cache,
                    local_environment,
                    state_functions,
                    registry,
                )
                .count_ones(..) as Integer,
            Self::Table(t) => t.eval(
                state,
                function_cache,
                local_environment,
                state_functions,
                registry,
                &registry.integer_tables,
            ),
            Self::MinimumSpanningTree(set, table) => {
                let set = match set.as_ref() {
                    SetExpression::Reference(expression) => expression.eval(
                        state,
                        function_cache,
                        local_environment,
                        state_functions,
                        registry,
                    ),
                    SetExpression::StateFunction(i) => function_cache.get_set_value(
                        *i,
                        state,
                        local_environment,
                        state_functions,
                        registry,
                    ),
                    set => &set.eval_with_local_environment(
                        state,
                        function_cache,
                        local_environment,
                        state_functions,
                        registry,
                    ),
                };
                let table = &registry.integer_tables.tables_2d[*table];

                algorithms::compute_minimum_spanning_tree_with_connectivity(
                    set,
                    |i, j| table.eval(i, j),
                    |_, _| true,
                    |weight| weight,
                )
            }
            Self::MinimumSpanningTreeWithConnectivity(set, table, connectivity_table) => {
                let set = match set.as_ref() {
                    SetExpression::Reference(expression) => expression.eval(
                        state,
                        function_cache,
                        local_environment,
                        state_functions,
                        registry,
                    ),
                    SetExpression::StateFunction(i) => function_cache.get_set_value(
                        *i,
                        state,
                        local_environment,
                        state_functions,
                        registry,
                    ),
                    set => &set.eval_with_local_environment(
                        state,
                        function_cache,
                        local_environment,
                        state_functions,
                        registry,
                    ),
                };
                let table = &registry.integer_tables.tables_2d[*table];
                let connectivity_table = &registry.bool_tables.tables_2d[*connectivity_table];

                algorithms::compute_minimum_spanning_tree_with_connectivity(
                    set,
                    |i, j| table.eval(i, j),
                    |i, j| connectivity_table.eval(i, j),
                    |weight| weight,
                )
            }
            Self::MinimumSpanningTreeWithEdges(set, edges) => {
                let set = set.eval_with_local_environment(
                    state,
                    function_cache,
                    local_environment,
                    state_functions,
                    registry,
                );

                let edges = edges.iter().map(|(i, j, weight)| {
                    (
                        *i,
                        *j,
                        weight.eval_inner(
                            cost,
                            state,
                            function_cache,
                            local_environment,
                            state_functions,
                            registry,
                        ),
                    )
                });

                algorithms::compute_minimum_spanning_tree_from_sorted_edges(
                    &set,
                    algorithms::sort_minimum_spanning_tree_edges(edges, |weight| weight),
                )
            }
            Self::MinimumSpanningTreeWithEdgesAndConnectivity(set, edges) => {
                let set = set.eval_with_local_environment(
                    state,
                    function_cache,
                    local_environment,
                    state_functions,
                    registry,
                );

                let edges = edges.iter().filter_map(|(i, j, weight, condition)| {
                    condition
                        .eval_with_local_environment(
                            state,
                            function_cache,
                            local_environment,
                            state_functions,
                            registry,
                        )
                        .then(|| {
                            (
                                *i,
                                *j,
                                weight.eval_inner(
                                    cost,
                                    state,
                                    function_cache,
                                    local_environment,
                                    state_functions,
                                    registry,
                                ),
                            )
                        })
                });

                algorithms::compute_minimum_spanning_tree_from_sorted_edges(
                    &set,
                    algorithms::sort_minimum_spanning_tree_edges(edges, |weight| weight),
                )
            }
            Self::MinimumSpanningTreeWithSortedEdges(set, sorted_edges) => {
                let set = match set.as_ref() {
                    SetExpression::Reference(expression) => expression.eval(
                        state,
                        function_cache,
                        local_environment,
                        state_functions,
                        registry,
                    ),
                    SetExpression::StateFunction(i) => function_cache.get_set_value(
                        *i,
                        state,
                        local_environment,
                        state_functions,
                        registry,
                    ),
                    set => &set.eval_with_local_environment(
                        state,
                        function_cache,
                        local_environment,
                        state_functions,
                        registry,
                    ),
                };

                algorithms::compute_minimum_spanning_tree_from_sorted_edges(
                    set,
                    sorted_edges.iter().copied(),
                )
            }
            Self::If(condition, x, y) => {
                if condition.eval_with_local_environment(
                    state,
                    function_cache,
                    local_environment,
                    state_functions,
                    registry,
                ) {
                    x.eval_inner(
                        cost,
                        state,
                        function_cache,
                        local_environment,
                        state_functions,
                        registry,
                    )
                } else {
                    y.eval_inner(
                        cost,
                        state,
                        function_cache,
                        local_environment,
                        state_functions,
                        registry,
                    )
                }
            }
            Self::FromContinuous(op, x) => op.eval(if let Some(cost) = cost {
                x.eval_cost_with_local_environment(
                    Continuous::from(cost),
                    state,
                    function_cache,
                    local_environment,
                    state_functions,
                    registry,
                )
            } else {
                x.eval_with_local_environment(
                    state,
                    function_cache,
                    local_environment,
                    state_functions,
                    registry,
                )
            }) as Integer,
            Self::Reduce(op, set, id, expression) => {
                let set = match set.as_ref() {
                    SetExpression::Reference(expression) => expression.eval(
                        state,
                        function_cache,
                        local_environment,
                        state_functions,
                        registry,
                    ),
                    set => &set.eval_with_local_environment(
                        state,
                        function_cache,
                        local_environment,
                        state_functions,
                        registry,
                    ),
                };
                let before = local_environment.get(*id);
                let result = op
                    .eval_iter(set.ones().map(|e| {
                        local_environment.set(*id, e);

                        expression.eval_inner(
                            cost,
                            state,
                            function_cache,
                            local_environment,
                            state_functions,
                            registry,
                        )
                    }))
                    .expect("`max`/`min` reduce performed on an empty set");

                if let Some(before) = before {
                    local_environment.set(*id, before);
                } else {
                    local_environment.unset(*id);
                }

                result
            }
            Self::FilterReduce(op, set, filter_id, id, condition, expression) => {
                let set = match set.as_ref() {
                    SetExpression::Reference(expression) => expression.eval(
                        state,
                        function_cache,
                        local_environment,
                        state_functions,
                        registry,
                    ),
                    set => &set.eval_with_local_environment(
                        state,
                        function_cache,
                        local_environment,
                        state_functions,
                        registry,
                    ),
                };
                let filter_before = local_environment.get(*filter_id);
                let reduce_before = local_environment.get(*id);
                let result = op
                    .eval_iter(set.ones().filter_map(|e| {
                        local_environment.set(*filter_id, e);
                        let passes = condition.eval_with_local_environment(
                            state,
                            function_cache,
                            local_environment,
                            state_functions,
                            registry,
                        );
                        if let Some(before) = filter_before {
                            local_environment.set(*filter_id, before);
                        } else {
                            local_environment.unset(*filter_id);
                        }

                        passes.then(|| {
                            local_environment.set(*id, e);
                            let result = expression.eval_inner(
                                cost,
                                state,
                                function_cache,
                                local_environment,
                                state_functions,
                                registry,
                            );
                            if let Some(before) = reduce_before {
                                local_environment.set(*id, before);
                            } else {
                                local_environment.unset(*id);
                            }
                            result
                        })
                    }))
                    .expect("`max`/`min` reduce performed on an empty filtered set");

                result
            }
        }
    }

    /// Returns a simplified version by precomputation.
    ///
    /// # Panics
    ///
    /// Panics if a min/max reduce operation is performed on an empty set or a constant minimum
    /// spanning tree expression contains a disconnected graph.
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
            Self::Table(expression) => {
                match expression.simplify(registry, &registry.integer_tables) {
                    NumericTableExpression::Constant(value) => Self::Constant(value),
                    expression => Self::Table(Box::new(expression)),
                }
            }
            Self::MinimumSpanningTree(set, table) => {
                let set = set.simplify(registry);
                let table =
                    registry.integer_tables.tables_2d.get(*table).expect(
                        "minimum spanning tree edge-weight table is not in the table registry",
                    );
                let sorted_edges = algorithms::sort_minimum_spanning_tree_edges_with_connectivity(
                    &table.0,
                    |_, _| true,
                    |weight| weight,
                );

                if let SetExpression::Reference(ReferenceExpression::Constant(set)) = &set {
                    return Self::Constant(
                        algorithms::compute_minimum_spanning_tree_from_sorted_edges(
                            set,
                            sorted_edges,
                        ),
                    );
                }

                Self::MinimumSpanningTreeWithSortedEdges(Box::new(set), sorted_edges)
            }
            Self::MinimumSpanningTreeWithConnectivity(set, table, connectivity_table) => {
                let set = set.simplify(registry);
                let table =
                    registry.integer_tables.tables_2d.get(*table).expect(
                        "minimum spanning tree edge-weight table is not in the table registry",
                    );
                let connectivity_table = registry
                    .bool_tables
                    .tables_2d
                    .get(*connectivity_table)
                    .expect(
                        "minimum spanning tree connectivity table is not in the table registry",
                    );
                let sorted_edges = algorithms::sort_minimum_spanning_tree_edges_with_connectivity(
                    &table.0,
                    |i, j| connectivity_table.0[i][j],
                    |weight| weight,
                );

                if let SetExpression::Reference(ReferenceExpression::Constant(set)) = &set {
                    return Self::Constant(
                        algorithms::compute_minimum_spanning_tree_from_sorted_edges(
                            set,
                            sorted_edges,
                        ),
                    );
                }

                Self::MinimumSpanningTreeWithSortedEdges(Box::new(set), sorted_edges)
            }
            Self::MinimumSpanningTreeWithEdges(set, edges) => {
                let set = set.simplify(registry);
                let edges = edges
                    .iter()
                    .map(|(i, j, weight)| (*i, *j, weight.simplify(registry)))
                    .collect::<Vec<_>>();
                let constant_edges = edges
                    .iter()
                    .map(|(i, j, weight)| match weight {
                        Self::Constant(value) => Some((*i, *j, *value)),
                        _ => None,
                    })
                    .collect::<Option<Vec<_>>>();

                if let Some(constant_edges) = constant_edges {
                    let sorted_edges =
                        algorithms::sort_minimum_spanning_tree_edges(constant_edges, |weight| {
                            weight
                        });

                    if let SetExpression::Reference(ReferenceExpression::Constant(set)) = &set {
                        return Self::Constant(
                            algorithms::compute_minimum_spanning_tree_from_sorted_edges(
                                set,
                                sorted_edges,
                            ),
                        );
                    }

                    return Self::MinimumSpanningTreeWithSortedEdges(Box::new(set), sorted_edges);
                }

                Self::MinimumSpanningTreeWithEdges(Box::new(set), edges)
            }
            Self::MinimumSpanningTreeWithEdgesAndConnectivity(set, edges) => {
                let set = set.simplify(registry);
                let edges = edges
                    .iter()
                    .map(|(i, j, weight, condition)| {
                        (
                            *i,
                            *j,
                            weight.simplify(registry),
                            condition.simplify(registry),
                        )
                    })
                    .collect::<Vec<_>>();
                let constant_edges = edges
                    .iter()
                    .map(|(i, j, weight, condition)| match (weight, condition) {
                        (_, Condition::Constant(false)) => Some(None),
                        (Self::Constant(value), Condition::Constant(true)) => {
                            Some(Some((*i, *j, *value)))
                        }
                        _ => None,
                    })
                    .collect::<Option<Vec<_>>>();

                if let Some(edges) = constant_edges {
                    let sorted_edges = algorithms::sort_minimum_spanning_tree_edges(
                        edges.into_iter().flatten(),
                        |weight| weight,
                    );

                    if let SetExpression::Reference(ReferenceExpression::Constant(set)) = &set {
                        return Self::Constant(
                            algorithms::compute_minimum_spanning_tree_from_sorted_edges(
                                set,
                                sorted_edges,
                            ),
                        );
                    }

                    return Self::MinimumSpanningTreeWithSortedEdges(Box::new(set), sorted_edges);
                }

                Self::MinimumSpanningTreeWithEdgesAndConnectivity(Box::new(set), edges)
            }
            Self::MinimumSpanningTreeWithSortedEdges(set, sorted_edges) => {
                let set = set.simplify(registry);

                if let SetExpression::Reference(ReferenceExpression::Constant(set)) = &set {
                    return Self::Constant(
                        algorithms::compute_minimum_spanning_tree_from_sorted_edges(
                            set,
                            sorted_edges.iter().copied(),
                        ),
                    );
                }

                Self::MinimumSpanningTreeWithSortedEdges(Box::new(set), sorted_edges.clone())
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
            Self::Reduce(op, set, id, expression) => {
                let set = set.simplify(registry);
                let expression = expression.simplify(registry);

                if let SetExpression::Reference(ReferenceExpression::Constant(set)) = &set {
                    let values = set
                        .ones()
                        .map(|element| {
                            match expression
                                .substitute_local_variable(*id, element)
                                .simplify(registry)
                            {
                                Self::Constant(value) => Some(value),
                                _ => None,
                            }
                        })
                        .collect::<Option<Vec<_>>>();
                    if let Some(values) = values {
                        return Self::Constant(
                            op.eval_iter(values.into_iter())
                                .expect("`max`/`min` reduce performed on an empty constant set"),
                        );
                    }
                }

                if let SetExpression::Filter(set, filter_id, condition) = set {
                    return Self::FilterReduce(
                        op.clone(),
                        set,
                        filter_id,
                        *id,
                        condition,
                        Box::new(expression),
                    );
                }

                if let Self::Table(table) = &expression {
                    if let Some(table) = table.reduce(op, &set, *id) {
                        return Self::Table(Box::new(table));
                    }
                }

                Self::Reduce(op.clone(), Box::new(set), *id, Box::new(expression))
            }
            Self::FilterReduce(op, set, filter_id, id, condition, expression) => {
                let set = set.simplify(registry);
                let expression = expression.simplify(registry);
                let condition = condition.simplify(registry);

                if let SetExpression::Reference(ReferenceExpression::Constant(set)) = &set {
                    let mut values = Vec::new();
                    let mut is_constant = true;

                    for element in set.ones() {
                        let condition = condition
                            .substitute_local_variable(*filter_id, element)
                            .simplify(registry);
                        match condition {
                            Condition::Constant(true) => {
                                match expression
                                    .substitute_local_variable(*id, element)
                                    .simplify(registry)
                                {
                                    Self::Constant(value) => values.push(value),
                                    _ => {
                                        is_constant = false;
                                        break;
                                    }
                                }
                            }
                            Condition::Constant(false) => {}
                            _ => {
                                is_constant = false;
                                break;
                            }
                        }
                    }

                    if is_constant {
                        return Self::Constant(
                            op.eval_iter(values.into_iter())
                                .expect("`max`/`min` reduce performed on an empty filtered set"),
                        );
                    }
                }

                match (op, &condition) {
                    (ReduceOperator::Sum, Condition::Constant(false)) => Self::Constant(0),
                    (ReduceOperator::Product, Condition::Constant(false)) => Self::Constant(1),
                    _ => Self::FilterReduce(
                        op.clone(),
                        Box::new(set),
                        *filter_id,
                        *id,
                        Box::new(condition),
                        Box::new(expression),
                    ),
                }
            }
            _ => self.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::condition::ComparisonOperator;
    use super::super::table_expression::TableExpression;
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

        let v = metadata.add_set_resource_variable(String::from("srv"), ob, false);
        assert!(v.is_ok());
        let v = v.unwrap();
        assert_eq!(
            v.len(),
            IntegerExpression::Cardinality(SetExpression::Reference(
                ReferenceExpression::ResourceVariable(v.id())
            ))
        );
    }

    #[test]
    fn set_reduce() {
        let mut local_variable_data = crate::LocalVariableData::default();
        let x = local_variable_data.add("x").unwrap();
        let set = SetExpression::Reference(ReferenceExpression::Variable(0));
        let value = IntegerExpression::Constant(2);

        assert_eq!(
            set.clone().sum(x, value.clone()),
            IntegerExpression::Reduce(
                ReduceOperator::Sum,
                Box::new(set.clone()),
                x.id(),
                Box::new(value.clone())
            )
        );

        let condition = Condition::comparison_e(ComparisonOperator::Ge, x, 1);
        assert_eq!(
            set.clone()
                .filter(x, condition.clone())
                .sum(x, value.clone()),
            IntegerExpression::Reduce(
                ReduceOperator::Sum,
                Box::new(SetExpression::Filter(
                    Box::new(set),
                    x.id(),
                    Box::new(condition),
                )),
                x.id(),
                Box::new(value)
            )
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
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();
        let expression = IntegerExpression::Constant(0);
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            0
        );
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
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();
        let expression = IntegerExpression::Variable(0);
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            0
        );
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
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();
        let expression = IntegerExpression::ResourceVariable(0);
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            0
        );
    }

    #[test]
    fn state_function_eval() {
        let mut state_metadata = StateMetadata::default();
        let v = state_metadata.add_integer_variable("v");
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_integer_function("f", v + 1);
        assert!(f.is_ok());
        let f = f.unwrap();
        let g = state_functions.add_integer_function("g", v + 2);
        assert!(g.is_ok());
        let g = g.unwrap();

        let state = State {
            signature_variables: SignatureVariables {
                integer_variables: vec![0],
                ..Default::default()
            },
            ..Default::default()
        };

        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        assert_eq!(
            f.eval(&state, &mut function_cache, &state_functions, &registry),
            1
        );

        assert_eq!(
            g.eval(&state, &mut function_cache, &state_functions, &registry),
            2
        );

        assert_eq!(
            f.eval(&state, &mut function_cache, &state_functions, &registry),
            1
        );
        assert_eq!(
            g.eval(&state, &mut function_cache, &state_functions, &registry),
            2
        );
    }

    #[test]
    fn eval_cost() {
        let state = State::default();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();
        let expression = IntegerExpression::Cost;
        assert_eq!(
            expression.eval_cost(10, &state, &mut function_cache, &state_functions, &registry),
            10
        );
    }

    #[test]
    #[should_panic]
    fn eval_cost_panic() {
        let state = State::default();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();
        let expression = IntegerExpression::Cost;
        expression.eval(&state, &mut function_cache, &state_functions, &registry);
    }

    #[test]
    fn unary_operation_eval() {
        let state = State::default();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();
        let expression = IntegerExpression::UnaryOperation(
            UnaryOperator::Abs,
            Box::new(IntegerExpression::Constant(-1)),
        );
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            1
        );
    }

    #[test]
    fn binary_operation_eval() {
        let state = State::default();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();
        let expression = IntegerExpression::BinaryOperation(
            BinaryOperator::Add,
            Box::new(IntegerExpression::Constant(1)),
            Box::new(IntegerExpression::Constant(2)),
        );
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            3
        );
    }

    #[test]
    fn cardinality_eval() {
        let state = State::default();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();
        let mut set = Set::with_capacity(5);
        set.insert(1);
        set.insert(4);
        let expression = IntegerExpression::Cardinality(SetExpression::Reference(
            ReferenceExpression::Constant(set),
        ));
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            2
        );
    }

    #[test]
    fn table_eval() {
        let state = State::default();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();
        let expression = IntegerExpression::Table(Box::new(NumericTableExpression::Constant(0)));
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            0
        );
    }

    #[test]
    fn minimum_spanning_tree_eval() {
        let state = State::default();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry {
            integer_tables: crate::table_data::TableData {
                tables_2d: vec![crate::table::Table2D::new(vec![
                    vec![0, 1, 4, 3],
                    vec![1, 0, 2, 5],
                    vec![4, 2, 0, 6],
                    vec![3, 5, 6, 0],
                ])],
                ..Default::default()
            },
            ..Default::default()
        };
        let mut set = Set::with_capacity(4);
        set.insert(0);
        set.insert(1);
        set.insert(2);
        set.insert(3);
        let expression = IntegerExpression::MinimumSpanningTree(
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(set))),
            0,
        );
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            6
        );
    }

    #[test]
    fn minimum_spanning_tree_with_sorted_edges_eval() {
        let state = State::default();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();
        let mut set = Set::with_capacity(4);
        set.insert(0);
        set.insert(1);
        set.insert(3);
        let expression = IntegerExpression::MinimumSpanningTreeWithSortedEdges(
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(set))),
            vec![
                (0, 1, 1),
                (1, 2, 2),
                (0, 3, 3),
                (0, 2, 4),
                (1, 3, 5),
                (2, 3, 6),
            ],
        );
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            4
        );
    }

    #[test]
    fn minimum_spanning_tree_with_edges_eval() {
        let state = State::default();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();
        let mut set = Set::with_capacity(4);
        set.insert(0);
        set.insert(1);
        set.insert(2);
        set.insert(3);
        // Edges are deliberately given out of weight order to confirm they are sorted before
        // running Kruskal's algorithm.
        let expression = IntegerExpression::MinimumSpanningTreeWithEdges(
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(set))),
            vec![
                (2, 3, IntegerExpression::Constant(6)),
                (1, 3, IntegerExpression::Constant(5)),
                (0, 3, IntegerExpression::Constant(3)),
                (1, 2, IntegerExpression::Constant(2)),
                (0, 2, IntegerExpression::Constant(4)),
                (0, 1, IntegerExpression::Constant(1)),
            ],
        );
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            6
        );
    }

    #[test]
    fn minimum_spanning_tree_with_edges_and_connectivity_eval() {
        let state = State::default();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();
        let mut set = Set::with_capacity(4);
        set.insert(0);
        set.insert(1);
        set.insert(2);
        set.insert(3);
        // The cheap edge (0, 1) is not present, so the tree must route through more expensive edges.
        let expression = IntegerExpression::MinimumSpanningTreeWithEdgesAndConnectivity(
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(set))),
            vec![
                (
                    0,
                    1,
                    IntegerExpression::Constant(1),
                    Condition::Constant(false),
                ),
                (
                    0,
                    2,
                    IntegerExpression::Constant(4),
                    Condition::Constant(true),
                ),
                (
                    0,
                    3,
                    IntegerExpression::Constant(3),
                    Condition::Constant(true),
                ),
                (
                    1,
                    2,
                    IntegerExpression::Constant(2),
                    Condition::Constant(true),
                ),
                (
                    1,
                    3,
                    IntegerExpression::Constant(5),
                    Condition::Constant(true),
                ),
                (
                    2,
                    3,
                    IntegerExpression::Constant(6),
                    Condition::Constant(true),
                ),
            ],
        );
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            9
        );
    }

    #[test]
    fn if_eval() {
        let state = State::default();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        let expression = IntegerExpression::If(
            Box::new(Condition::Constant(true)),
            Box::new(IntegerExpression::Constant(1)),
            Box::new(IntegerExpression::Constant(2)),
        );
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            1
        );

        let expression = IntegerExpression::If(
            Box::new(Condition::Constant(false)),
            Box::new(IntegerExpression::Constant(1)),
            Box::new(IntegerExpression::Constant(2)),
        );
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            2
        );
    }

    #[test]
    fn from_continuous_eval() {
        let state = State::default();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        let expression = IntegerExpression::FromContinuous(
            CastOperator::Floor,
            Box::new(ContinuousExpression::Constant(1.5)),
        );
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            1
        );
    }

    #[test]
    fn reduce_eval_restores_local_environment() {
        let state = State::default();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let mut registry = TableRegistry::default();
        let mut local_variable_data = crate::LocalVariableData::default();
        let x = local_variable_data.add("x").unwrap();
        let table = registry.add_table_1d("values", vec![10, 20, 30]).unwrap();
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(2);
        let expression =
            SetExpression::from(set).sum(x, Table1DHandle::<Integer>::element(&table, x));
        let mut local_environment = LocalEnvironment::default();
        local_environment.set(x.id(), 1);

        assert_eq!(
            expression.eval_with_local_environment(
                &state,
                &mut function_cache,
                &mut local_environment,
                &state_functions,
                &registry
            ),
            40
        );
        assert_eq!(local_environment.get(x.id()), Some(1));

        let mut local_environment = LocalEnvironment::default();
        let _ = expression.eval_with_local_environment(
            &state,
            &mut function_cache,
            &mut local_environment,
            &state_functions,
            &registry,
        );
        assert_eq!(local_environment.get(x.id()), None);
    }

    #[test]
    fn filter_reduce_eval_restores_local_environment() {
        let state = State::default();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let mut registry = TableRegistry::default();
        let mut local_variable_data = crate::LocalVariableData::default();
        let x = local_variable_data.add("x").unwrap();
        let y = local_variable_data.add("y").unwrap();
        let table = registry.add_table_1d("values", vec![10, 20, 30]).unwrap();
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(1);
        set.insert(2);
        let expression = IntegerExpression::FilterReduce(
            ReduceOperator::Sum,
            Box::new(SetExpression::from(set)),
            x.id(),
            y.id(),
            Box::new(Condition::comparison_e(ComparisonOperator::Ge, x, 1)),
            Box::new(Table1DHandle::<Integer>::element(&table, y)),
        );
        let mut local_environment = LocalEnvironment::default();
        local_environment.set(x.id(), 0);
        local_environment.set(y.id(), 2);

        assert_eq!(
            expression.eval_with_local_environment(
                &state,
                &mut function_cache,
                &mut local_environment,
                &state_functions,
                &registry
            ),
            50
        );
        assert_eq!(local_environment.get(x.id()), Some(0));
        assert_eq!(local_environment.get(y.id()), Some(2));

        let mut local_environment = LocalEnvironment::default();
        let _ = expression.eval_with_local_environment(
            &state,
            &mut function_cache,
            &mut local_environment,
            &state_functions,
            &registry,
        );
        assert_eq!(local_environment.get(x.id()), None);
        assert_eq!(local_environment.get(y.id()), None);
    }

    #[test]
    fn reduce_simplify() {
        let mut registry = TableRegistry::default();
        let mut local_variable_data = crate::LocalVariableData::default();
        let x = local_variable_data.add("x").unwrap();
        let y = local_variable_data.add("y").unwrap();
        let table = registry.add_table_1d("values", vec![1, 2, 3]).unwrap();
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(1);
        set.insert(2);
        let expression = SetExpression::from(set.clone()).sum(x, IntegerExpression::Constant(2));

        assert_eq!(
            expression.simplify(&registry),
            IntegerExpression::Constant(6)
        );

        let expression =
            SetExpression::from(set.clone()).sum(x, Table1DHandle::<Integer>::element(&table, x));
        assert_eq!(
            expression.simplify(&registry),
            IntegerExpression::Constant(6)
        );

        let expression = SetExpression::from(set).sum(x, IntegerExpression::Variable(0));
        assert_eq!(expression.simplify(&registry), expression);

        let expression = SetExpression::Reference(ReferenceExpression::Variable(0))
            .sum(x, IntegerExpression::Variable(0));
        assert_eq!(expression.simplify(&registry), expression);

        let set = SetExpression::Reference(ReferenceExpression::Variable(0));
        let condition = Condition::comparison_e(ComparisonOperator::Ge, x, 1);
        let expression = set
            .clone()
            .filter(x, condition.clone())
            .sum(y, Table1DHandle::<Integer>::element(&table, y));
        assert_eq!(
            expression.simplify(&registry),
            IntegerExpression::FilterReduce(
                ReduceOperator::Sum,
                Box::new(set.clone()),
                x.id(),
                y.id(),
                Box::new(condition),
                Box::new(Table1DHandle::<Integer>::element(&table, y)),
            )
        );

        let expression = set
            .clone()
            .sum(x, Table1DHandle::<Integer>::element(&table, x));
        assert_eq!(
            expression.simplify(&registry),
            IntegerExpression::Table(Box::new(NumericTableExpression::Table1DReduce(
                ReduceOperator::Sum,
                table.id(),
                set,
            )))
        );
    }

    #[test]
    fn filter_reduce_simplify() {
        let mut registry = TableRegistry::default();
        let mut local_variable_data = crate::LocalVariableData::default();
        let x = local_variable_data.add("x").unwrap();
        let table = registry.add_table_1d("values", vec![1, 2, 3]).unwrap();
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(1);
        set.insert(2);
        let expression = SetExpression::from(set.clone())
            .filter(x, Condition::Constant(true))
            .sum(x, IntegerExpression::Constant(2));

        assert_eq!(
            expression.simplify(&registry),
            IntegerExpression::Constant(6)
        );

        let expression = SetExpression::from(set.clone())
            .filter(x, Condition::Constant(false))
            .sum(x, IntegerExpression::Constant(2));
        assert_eq!(
            expression.simplify(&registry),
            IntegerExpression::Constant(0)
        );

        let expression = SetExpression::from(set.clone())
            .filter(x, Condition::comparison_e(ComparisonOperator::Ge, x, 1))
            .sum(x, Table1DHandle::<Integer>::element(&table, x));
        assert_eq!(
            expression.simplify(&registry),
            IntegerExpression::Constant(5)
        );

        let expression = SetExpression::from(set.clone())
            .filter(x, Condition::Constant(true))
            .sum(x, IntegerExpression::Variable(0));
        assert_eq!(
            expression.simplify(&registry),
            SetExpression::from(set).sum(x, IntegerExpression::Variable(0))
        );
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
    fn minimum_spanning_tree_simplify() {
        let registry = TableRegistry {
            integer_tables: crate::table_data::TableData {
                tables_2d: vec![crate::table::Table2D::new(vec![
                    vec![0, 1, 4, 3],
                    vec![1, 0, 2, 5],
                    vec![4, 2, 0, 6],
                    vec![3, 5, 6, 0],
                ])],
                ..Default::default()
            },
            ..Default::default()
        };
        let mut set = Set::with_capacity(4);
        set.insert(0);
        set.insert(1);
        set.insert(2);
        set.insert(3);
        let expression = IntegerExpression::MinimumSpanningTree(
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(set))),
            0,
        );
        assert_eq!(
            expression.simplify(&registry),
            IntegerExpression::Constant(6)
        );

        let expression = IntegerExpression::MinimumSpanningTree(
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
            0,
        );
        assert_eq!(
            expression.simplify(&registry),
            IntegerExpression::MinimumSpanningTreeWithSortedEdges(
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
                vec![
                    (0, 1, 1),
                    (1, 2, 2),
                    (0, 3, 3),
                    (0, 2, 4),
                    (1, 3, 5),
                    (2, 3, 6),
                ]
            )
        );
    }

    #[test]
    fn minimum_spanning_tree_with_edges_simplify() {
        let registry = TableRegistry::default();
        let edges = vec![
            (0, 2, IntegerExpression::Constant(4)),
            (1, 2, IntegerExpression::Constant(2)),
            (0, 1, IntegerExpression::Constant(1)),
        ];

        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(1);
        set.insert(2);
        let expression = IntegerExpression::MinimumSpanningTreeWithEdges(
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(set))),
            edges.clone(),
        );
        assert_eq!(
            expression.simplify(&registry),
            IntegerExpression::Constant(3)
        );

        let expression = IntegerExpression::MinimumSpanningTreeWithEdges(
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
            edges,
        );
        assert_eq!(
            expression.simplify(&registry),
            IntegerExpression::MinimumSpanningTreeWithSortedEdges(
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
                vec![(0, 1, 1), (1, 2, 2), (0, 2, 4)]
            )
        );

        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(1);
        set.insert(2);
        let edges = vec![
            (0, 1, IntegerExpression::Variable(0)),
            (0, 2, IntegerExpression::Constant(4)),
            (1, 2, IntegerExpression::Constant(2)),
        ];
        let expression = IntegerExpression::MinimumSpanningTreeWithEdges(
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(set))),
            edges,
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn minimum_spanning_tree_with_edges_and_connectivity_simplify() {
        let registry = TableRegistry::default();
        let edges = vec![
            (
                0,
                1,
                IntegerExpression::Constant(1),
                Condition::Constant(false),
            ),
            (
                0,
                2,
                IntegerExpression::Constant(4),
                Condition::Constant(true),
            ),
            (
                0,
                3,
                IntegerExpression::Constant(3),
                Condition::Constant(true),
            ),
            (
                1,
                2,
                IntegerExpression::Constant(2),
                Condition::Constant(true),
            ),
            (
                1,
                3,
                IntegerExpression::Constant(5),
                Condition::Constant(true),
            ),
            (
                2,
                3,
                IntegerExpression::Constant(6),
                Condition::Constant(true),
            ),
        ];

        let mut set = Set::with_capacity(4);
        set.insert(0);
        set.insert(1);
        set.insert(2);
        set.insert(3);
        let expression = IntegerExpression::MinimumSpanningTreeWithEdgesAndConnectivity(
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(set))),
            edges.clone(),
        );
        assert_eq!(
            expression.simplify(&registry),
            IntegerExpression::Constant(9)
        );

        let expression = IntegerExpression::MinimumSpanningTreeWithEdgesAndConnectivity(
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
            edges,
        );
        assert_eq!(
            expression.simplify(&registry),
            IntegerExpression::MinimumSpanningTreeWithSortedEdges(
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
                vec![(1, 2, 2), (0, 3, 3), (0, 2, 4), (1, 3, 5), (2, 3, 6)]
            )
        );

        let mut set = Set::with_capacity(4);
        set.insert(0);
        set.insert(1);
        set.insert(2);
        set.insert(3);
        let edges = vec![
            (
                0,
                1,
                IntegerExpression::Constant(1),
                Condition::Table(Box::new(TableExpression::Table1D(
                    0,
                    ElementExpression::Variable(0),
                ))),
            ),
            (
                0,
                2,
                IntegerExpression::Constant(4),
                Condition::Constant(true),
            ),
        ];
        let expression = IntegerExpression::MinimumSpanningTreeWithEdgesAndConnectivity(
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(set))),
            edges,
        );
        assert_eq!(expression.simplify(&registry), expression);
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
}
