use super::argument_expression::ArgumentExpression;
use super::condition::{Condition, IfThenElse};
use super::element_expression::ElementExpression;
use super::reference_expression::ReferenceExpression;
use super::set_reduce_expression::{SetReduceExpression, SetReduceOperator};
use super::table_expression::TableExpression;
use super::vector_expression::VectorExpression;
use crate::state::{ElementResourceVariable, ElementVariable, SetVariable, StateInterface};
use crate::state_functions::{StateFunctionCache, StateFunctions};
use crate::table_data::{Table1DHandle, Table2DHandle, Table3DHandle, TableHandle};
use crate::table_registry::TableRegistry;
use crate::variable_type::{Element, Set};
use std::ops;

/// Set expression.
#[derive(Debug, PartialEq, Clone)]
pub enum SetExpression {
    /// Reference to a constant or a variable.
    Reference(ReferenceExpression<Set>),
    /// State function index.
    StateFunction(usize),
    /// Complement set.
    Complement(Box<SetExpression>),
    /// Operation on two sets.
    SetOperation(SetOperator, Box<SetExpression>, Box<SetExpression>),
    /// Operation on an element and a set.
    SetElementOperation(SetElementOperator, ElementExpression, Box<SetExpression>),
    /// Reduce operation on a table of sets.
    Reduce(SetReduceExpression),
    /// Conversion from a vector.
    FromVector(usize, Box<VectorExpression>),
    /// If-then-else expression, which returns the first one if the condition holds and the second one otherwise.
    If(Box<Condition>, Box<SetExpression>, Box<SetExpression>),
}

/// Operator on an two sets.
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum SetOperator {
    /// Union.
    Union,
    /// Difference.
    Difference,
    /// Intersection.
    Intersection,
}

/// Operator on an elment and a set.
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum SetElementOperator {
    /// Add an element.
    Add,
    /// Remove an element.
    Remove,
}

impl Default for SetExpression {
    #[inline]
    fn default() -> Self {
        SetExpression::Reference(ReferenceExpression::Constant(Set::default()))
    }
}

impl From<Set> for SetExpression {
    #[inline]
    fn from(s: Set) -> Self {
        SetExpression::Reference(ReferenceExpression::Constant(s))
    }
}

impl From<SetVariable> for SetExpression {
    #[inline]
    fn from(v: SetVariable) -> Self {
        SetExpression::Reference(ReferenceExpression::Variable(v.id()))
    }
}

impl ops::Not for SetExpression {
    type Output = SetExpression;

    /// Returns an expression representing the complement.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let object_type = model.add_object_type("object", 4).unwrap();
    /// let state = model.target.clone();
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    /// let set = model.create_set(object_type, &[0, 1]).unwrap();
    ///
    /// let expression = SetExpression::from(set);
    /// let expression = !expression;
    /// let expected = model.create_set(object_type, &[2, 3]).unwrap();
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     expected,
    /// );
    /// ```
    #[inline]
    fn not(self) -> Self::Output {
        SetExpression::Complement(Box::new(self))
    }
}

impl ops::Not for SetVariable {
    type Output = SetExpression;

    /// Returns an expression representing the complement.
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
    /// let expression = !variable;
    /// let expected = model.create_set(object_type, &[2, 3]).unwrap();
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     expected,
    /// );
    /// ```
    #[inline]
    fn not(self) -> Self::Output {
        SetExpression::Complement(Box::new(SetExpression::from(self)))
    }
}

impl ops::BitOr for SetExpression {
    type Output = SetExpression;

    /// Returns an expression representing the union.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let object_type = model.add_object_type("object", 4).unwrap();
    /// let state = model.target.clone();
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let a = model.create_set(object_type, &[0, 1]).unwrap();
    /// let a = SetExpression::from(a);
    /// let b = model.create_set(object_type, &[1, 2]).unwrap();
    /// let b = SetExpression::from(b);
    /// let expression = a | b;
    /// let expected = model.create_set(object_type, &[0, 1, 2]).unwrap();
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     expected,
    /// );
    /// ```
    #[inline]
    fn bitor(self, rhs: Self) -> Self::Output {
        SetExpression::SetOperation(SetOperator::Union, Box::new(self), Box::new(rhs))
    }
}

impl ops::Sub for SetExpression {
    type Output = SetExpression;

    /// Returns an expression representing the difference.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let object_type = model.add_object_type("object", 4).unwrap();
    /// let state = model.target.clone();
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let a = model.create_set(object_type, &[0, 1]).unwrap();
    /// let a = SetExpression::from(a);
    /// let b = model.create_set(object_type, &[1, 2]).unwrap();
    /// let b = SetExpression::from(b);
    /// let expression = a - b;
    /// let expected = model.create_set(object_type, &[0]).unwrap();
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     expected,
    /// );
    /// ```
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        SetExpression::SetOperation(SetOperator::Difference, Box::new(self), Box::new(rhs))
    }
}

impl ops::BitAnd for SetExpression {
    type Output = SetExpression;

    /// Returns an expression representing the intersection.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let object_type = model.add_object_type("object", 4).unwrap();
    /// let state = model.target.clone();
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let a = model.create_set(object_type, &[0, 1]).unwrap();
    /// let a = SetExpression::from(a);
    /// let b = model.create_set(object_type, &[1, 2]).unwrap();
    /// let b = SetExpression::from(b);
    /// let expression = a & b;
    /// let expected = model.create_set(object_type, &[1]).unwrap();
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     expected,
    /// );
    /// ```
    #[inline]
    fn bitand(self, rhs: Self) -> Self::Output {
        SetExpression::SetOperation(SetOperator::Intersection, Box::new(self), Box::new(rhs))
    }
}

/// A trait for operations on an element and a set.
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
/// let expression = variable.add(2);
/// let expected = model.create_set(object_type, &[0, 1, 2]).unwrap();
/// assert_eq!(expression.eval(&state, &mut function_cache, &model.state_functions, &model.table_registry), expected);
///
/// let expression = variable.remove(1);
/// let expected = model.create_set(object_type, &[0]).unwrap();
/// assert_eq!(
///     expression.eval(
///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
///     ),
///     expected,
/// );
/// ```
pub trait SetElementOperation<Rhs> {
    /// Returns a set expression, where an element is added.
    fn add(self, rhs: Rhs) -> SetExpression;
    /// Returns a set expression, where an element is removed.
    fn remove(self, rhs: Rhs) -> SetExpression;
}

impl Table1DHandle<Set> {
    /// Returns a constant in a 1D set table.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let object_type = model.add_object_type("object", 2).unwrap();
    /// let a = model.create_set(object_type, &[1]).unwrap();
    /// let b = model.create_set(object_type, &[0]).unwrap();
    /// let table = model.add_table_1d("table", vec![a.clone(), b]).unwrap();
    /// let variable = model.add_element_variable("variable", object_type, 0).unwrap();
    /// let state = model.target.clone();
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let expression = table.element(variable);
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     a,
    /// );
    /// ```
    #[inline]
    pub fn element<T>(&self, x: T) -> SetExpression
    where
        ElementExpression: std::convert::From<T>,
    {
        SetExpression::Reference(ReferenceExpression::Table(TableExpression::Table1D(
            self.id(),
            ElementExpression::from(x),
        )))
    }

    /// Returns the union of sets in a 1D table.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let capacity = 2;
    /// let object_type = model.add_object_type("object", capacity).unwrap();
    /// let a = model.create_set(object_type, &[0, 1]).unwrap();
    /// let b = model.create_set(object_type, &[0]).unwrap();
    /// let table = model.add_table_1d("table", vec![a, b]).unwrap();
    /// let set = model.create_set(object_type, &[0, 1]).unwrap();
    /// let variable = model.add_set_variable("variable", object_type, set).unwrap();
    /// let state = model.target.clone();
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let expression = table.union(capacity, variable);
    /// let expected = model.create_set(object_type, &[0, 1]).unwrap();
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     expected,
    /// );
    /// ```
    #[inline]
    pub fn union<T>(&self, capacity: usize, x: T) -> SetExpression
    where
        ArgumentExpression: From<T>,
    {
        SetExpression::Reduce(SetReduceExpression::Table1D(
            SetReduceOperator::Union,
            capacity,
            self.id(),
            Box::new(ArgumentExpression::from(x)),
        ))
    }

    /// Returns the intersection of sets in a 1D table.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let capacity = 2;
    /// let object_type = model.add_object_type("object", capacity).unwrap();
    /// let a = model.create_set(object_type, &[0, 1]).unwrap();
    /// let b = model.create_set(object_type, &[0]).unwrap();
    /// let table = model.add_table_1d("table", vec![a, b]).unwrap();
    /// let set = model.create_set(object_type, &[0, 1]).unwrap();
    /// let variable = model.add_set_variable("variable", object_type, set).unwrap();
    /// let state = model.target.clone();
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let expression = table.intersection(capacity, variable);
    /// let expected = model.create_set(object_type, &[0]).unwrap();
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     expected,
    /// );
    /// ```
    #[inline]
    pub fn intersection<T>(&self, capacity: usize, x: T) -> SetExpression
    where
        ArgumentExpression: From<T>,
    {
        SetExpression::Reduce(SetReduceExpression::Table1D(
            SetReduceOperator::Intersection,
            capacity,
            self.id(),
            Box::new(ArgumentExpression::from(x)),
        ))
    }

    /// Returns the symmetric difference (disjunctive union) of sets in a 1D table.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let capacity = 2;
    /// let object_type = model.add_object_type("object", capacity).unwrap();
    /// let a = model.create_set(object_type, &[1]).unwrap();
    /// let b = model.create_set(object_type, &[0]).unwrap();
    /// let table = model.add_table_1d("table", vec![a, b]).unwrap();
    /// let set = model.create_set(object_type, &[0, 1]).unwrap();
    /// let variable = model.add_set_variable("variable", object_type, set).unwrap();
    /// let state = model.target.clone();
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let expression = table.symmetric_difference(capacity, variable);
    /// let expected = model.create_set(object_type, &[0, 1]).unwrap();
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     expected,
    /// );
    #[inline]
    pub fn symmetric_difference<T>(&self, capacity: usize, x: T) -> SetExpression
    where
        ArgumentExpression: From<T>,
    {
        SetExpression::Reduce(SetReduceExpression::Table1D(
            SetReduceOperator::SymmetricDifference,
            capacity,
            self.id(),
            Box::new(ArgumentExpression::from(x)),
        ))
    }
}

impl Table2DHandle<Set> {
    /// Returns a constant in a 2D set table.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let object_type = model.add_object_type("object", 2).unwrap();
    /// let a = model.create_set(object_type, &[1]).unwrap();
    /// let b = model.create_set(object_type, &[0]).unwrap();
    /// let table = model.add_table_2d("table", vec![vec![a.clone(), b.clone()], vec![b, a.clone()]]).unwrap();
    /// let variable = model.add_element_variable("variable", object_type, 0).unwrap();
    /// let state = model.target.clone();
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let expression = table.element(variable, 0);
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     a,
    /// );
    /// ```
    #[inline]
    pub fn element<T, U>(&self, x: T, y: U) -> SetExpression
    where
        ElementExpression: std::convert::From<T>,
        ElementExpression: std::convert::From<U>,
    {
        SetExpression::Reference(ReferenceExpression::Table(TableExpression::Table2D(
            self.id(),
            ElementExpression::from(x),
            ElementExpression::from(y),
        )))
    }

    /// Returns the union of sets in a 2D table.
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let capacity = 2;
    /// let object_type = model.add_object_type("object", capacity).unwrap();
    /// let a = model.create_set(object_type, &[1]).unwrap();
    /// let b = model.create_set(object_type, &[0]).unwrap();
    /// let table = model.add_table_2d("table", vec![vec![a.clone(), b.clone()], vec![a, b]]).unwrap();
    /// let set = model.create_set(object_type, &[0, 1]).unwrap();
    /// let variable = model.add_set_variable("variable", object_type, set).unwrap();
    /// let state = model.target.clone();
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let expression = table.union(capacity, variable, 0);
    /// let expected = model.create_set(object_type, &[1]).unwrap();
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     expected,
    /// );
    ///
    /// let expression = table.union(capacity, variable, variable);
    /// let expected = model.create_set(object_type, &[0, 1]).unwrap();
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     expected,
    /// );
    /// ```
    #[inline]
    pub fn union<T, U>(&self, capacity: usize, x: T, y: U) -> SetExpression
    where
        ArgumentExpression: From<T>,
        ArgumentExpression: From<U>,
    {
        SetExpression::Reduce(SetReduceExpression::Table2D(
            SetReduceOperator::Union,
            capacity,
            self.id(),
            Box::new(ArgumentExpression::from(x)),
            Box::new(ArgumentExpression::from(y)),
        ))
    }

    /// Returns the intersection of sets in a 2D table.
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let capacity = 2;
    /// let object_type = model.add_object_type("object", capacity).unwrap();
    /// let a = model.create_set(object_type, &[1]).unwrap();
    /// let b = model.create_set(object_type, &[0]).unwrap();
    /// let table = model.add_table_2d("table", vec![vec![a.clone(), b.clone()], vec![a, b]]).unwrap();
    /// let set = model.create_set(object_type, &[0, 1]).unwrap();
    /// let variable = model.add_set_variable("variable", object_type, set).unwrap();
    /// let state = model.target.clone();
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let expression = table.intersection(capacity, variable, 0);
    /// let expected = model.create_set(object_type, &[1]).unwrap();
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     expected,
    /// );
    ///
    /// let expression = table.intersection(capacity, variable, variable);
    /// let expected = model.create_set(object_type, &[]).unwrap();
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     expected,
    /// );
    /// ```
    #[inline]
    pub fn intersection<T, U>(&self, capacity: usize, x: T, y: U) -> SetExpression
    where
        ArgumentExpression: From<T>,
        ArgumentExpression: From<U>,
    {
        SetExpression::Reduce(SetReduceExpression::Table2D(
            SetReduceOperator::Intersection,
            capacity,
            self.id(),
            Box::new(ArgumentExpression::from(x)),
            Box::new(ArgumentExpression::from(y)),
        ))
    }

    /// Returns the symmetric difference (disjunctive union) of sets in a 2D table.
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let capacity = 2;
    /// let object_type = model.add_object_type("object", capacity).unwrap();
    /// let a = model.create_set(object_type, &[1]).unwrap();
    /// let b = model.create_set(object_type, &[0]).unwrap();
    /// let table = model.add_table_2d("table", vec![vec![a.clone(), b.clone()], vec![a, b]]).unwrap();
    /// let set = model.create_set(object_type, &[0, 1]).unwrap();
    /// let variable = model.add_set_variable("variable", object_type, set).unwrap();
    /// let state = model.target.clone();
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let expression = table.symmetric_difference(capacity, variable, 0);
    /// let expected = model.create_set(object_type, &[]).unwrap();
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     expected,
    /// );
    ///
    /// let expression = table.symmetric_difference(capacity, variable, variable);
    /// let expected = model.create_set(object_type, &[]).unwrap();
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     expected,
    /// );
    /// ```
    #[inline]
    pub fn symmetric_difference<T, U>(&self, capacity: usize, x: T, y: U) -> SetExpression
    where
        ArgumentExpression: From<T>,
        ArgumentExpression: From<U>,
    {
        SetExpression::Reduce(SetReduceExpression::Table2D(
            SetReduceOperator::SymmetricDifference,
            capacity,
            self.id(),
            Box::new(ArgumentExpression::from(x)),
            Box::new(ArgumentExpression::from(y)),
        ))
    }
}

impl Table3DHandle<Set> {
    /// Returns a constant in a 3D set table.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let object_type = model.add_object_type("object", 2).unwrap();
    /// let a = model.create_set(object_type, &[1]).unwrap();
    /// let b = model.create_set(object_type, &[0]).unwrap();
    /// let table = model.add_table_3d(
    ///     "table",
    ///     vec![
    ///         vec![vec![a.clone(), b.clone()], vec![b.clone(), a.clone()]],
    ///         vec![vec![a.clone(), b.clone()], vec![b, a.clone()]],
    ///     ]
    /// ).unwrap();
    /// let variable = model.add_element_variable("variable", object_type, 0).unwrap();
    /// let state = model.target.clone();
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let expression = table.element(variable, variable + 1, 1);
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     a,
    /// );
    /// ```
    #[inline]
    pub fn element<T, U, V>(&self, x: T, y: U, z: V) -> SetExpression
    where
        ElementExpression: std::convert::From<T>,
        ElementExpression: std::convert::From<U>,
        ElementExpression: std::convert::From<V>,
    {
        SetExpression::Reference(ReferenceExpression::Table(TableExpression::Table3D(
            self.id(),
            ElementExpression::from(x),
            ElementExpression::from(y),
            ElementExpression::from(z),
        )))
    }

    /// Returns the union of sets in a 3D table.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let capacity = 2;
    /// let object_type = model.add_object_type("object", capacity).unwrap();
    /// let a = model.create_set(object_type, &[1]).unwrap();
    /// let b = model.create_set(object_type, &[0]).unwrap();
    /// let table = model.add_table_3d(
    ///     "table",
    ///     vec![
    ///         vec![vec![a.clone(), b.clone()], vec![b.clone(), a.clone()]],
    ///         vec![vec![a.clone(), b.clone()], vec![b, a.clone()]],
    ///     ]
    /// ).unwrap();
    /// let set = model.create_set(object_type, &[0, 1]).unwrap();
    /// let variable = model.add_set_variable("variable", object_type, set).unwrap();
    /// let state = model.target.clone();
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let expression = table.union(capacity, variable, 0, 0);
    /// let expected = model.create_set(object_type, &[1]).unwrap();
    /// assert_eq!(
    ///     expression.eval(    
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     expected,
    /// );
    ///
    /// let expression = table.union(capacity, variable, variable, variable);
    /// let expected = model.create_set(object_type, &[0, 1]).unwrap();
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     expected,
    /// );
    /// ```
    #[inline]
    pub fn union<T, U, V>(&self, capacity: usize, x: T, y: U, z: V) -> SetExpression
    where
        ArgumentExpression: From<T>,
        ArgumentExpression: From<U>,
        ArgumentExpression: From<V>,
    {
        SetExpression::Reduce(SetReduceExpression::Table3D(
            SetReduceOperator::Union,
            capacity,
            self.id(),
            Box::new(ArgumentExpression::from(x)),
            Box::new(ArgumentExpression::from(y)),
            Box::new(ArgumentExpression::from(z)),
        ))
    }

    /// Returns the intersection of sets in a 3D table.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let capacity = 2;
    /// let object_type = model.add_object_type("object", capacity).unwrap();
    /// let a = model.create_set(object_type, &[1]).unwrap();
    /// let b = model.create_set(object_type, &[0]).unwrap();
    /// let table = model.add_table_3d(
    ///     "table",
    ///     vec![
    ///         vec![vec![a.clone(), b.clone()], vec![b.clone(), a.clone()]],
    ///         vec![vec![a.clone(), b.clone()], vec![b, a.clone()]],
    ///     ]
    /// ).unwrap();
    /// let set = model.create_set(object_type, &[0, 1]).unwrap();
    /// let variable = model.add_set_variable("variable", object_type, set).unwrap();
    /// let state = model.target.clone();
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let expression = table.intersection(capacity, variable, 0, 0);
    /// let expected = model.create_set(object_type, &[1]).unwrap();
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     expected,
    /// );
    ///
    /// let expression = table.intersection(capacity, variable, variable, variable);
    /// let expected = model.create_set(object_type, &[]).unwrap();
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),  
    ///     expected,
    /// );
    /// ```
    #[inline]
    pub fn intersection<T, U, V>(&self, capacity: usize, x: T, y: U, z: V) -> SetExpression
    where
        ArgumentExpression: From<T>,
        ArgumentExpression: From<U>,
        ArgumentExpression: From<V>,
    {
        SetExpression::Reduce(SetReduceExpression::Table3D(
            SetReduceOperator::Intersection,
            capacity,
            self.id(),
            Box::new(ArgumentExpression::from(x)),
            Box::new(ArgumentExpression::from(y)),
            Box::new(ArgumentExpression::from(z)),
        ))
    }

    /// Returns the symmetric difference (disjunctive union) of sets in a 3D table.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let capacity = 2;
    /// let object_type = model.add_object_type("object", capacity).unwrap();
    /// let a = model.create_set(object_type, &[1]).unwrap();
    /// let b = model.create_set(object_type, &[0]).unwrap();
    /// let table = model.add_table_3d(
    ///     "table",
    ///     vec![
    ///         vec![vec![a.clone(), b.clone()], vec![b.clone(), a.clone()]],
    ///         vec![vec![a.clone(), b.clone()], vec![b, a.clone()]],
    ///     ]
    /// ).unwrap();
    /// let set = model.create_set(object_type, &[0, 1]).unwrap();
    /// let variable = model.add_set_variable("variable", object_type, set).unwrap();
    /// let state = model.target.clone();
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let expression = table.symmetric_difference(capacity, variable, 0, 0);
    /// let expected = model.create_set(object_type, &[]).unwrap();
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     expected,
    /// );
    ///
    /// let expression = table.symmetric_difference(capacity, variable, variable, variable);
    /// let expected = model.create_set(object_type, &[]).unwrap();
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     expected,
    /// );
    /// ```
    #[inline]
    pub fn symmetric_difference<T, U, V>(&self, capacity: usize, x: T, y: U, z: V) -> SetExpression
    where
        ArgumentExpression: From<T>,
        ArgumentExpression: From<U>,
        ArgumentExpression: From<V>,
    {
        SetExpression::Reduce(SetReduceExpression::Table3D(
            SetReduceOperator::SymmetricDifference,
            capacity,
            self.id(),
            Box::new(ArgumentExpression::from(x)),
            Box::new(ArgumentExpression::from(y)),
            Box::new(ArgumentExpression::from(z)),
        ))
    }
}

impl TableHandle<Set> {
    /// Returns a constant in a set table.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    /// use rustc_hash::FxHashMap;
    ///
    /// let mut model = Model::default();
    /// let object_type = model.add_object_type("object", 2).unwrap();
    /// let a = model.create_set(object_type, &[1]).unwrap();
    /// let b = model.create_set(object_type, &[0]).unwrap();
    /// let map = FxHashMap::from_iter(
    ///     vec![(vec![0, 0, 0, 0], a.clone()), (vec![1, 1, 1, 1], a.clone())]
    /// );
    /// let table = model.add_table("table", map, b).unwrap();
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
    /// let expression = table.element(indices);
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     a,
    /// );
    /// ```
    #[inline]
    pub fn element<T>(&self, indices: Vec<T>) -> SetExpression
    where
        ElementExpression: std::convert::From<T>,
    {
        let indices = indices.into_iter().map(ElementExpression::from).collect();
        SetExpression::Reference(ReferenceExpression::Table(TableExpression::Table(
            self.id(),
            indices,
        )))
    }

    /// Returns the union of sets in a table.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    /// use dypdl::expression::*;
    /// use rustc_hash::FxHashMap;
    ///
    /// let capacity = 2;
    /// let mut model = Model::default();
    /// let object_type = model.add_object_type("object", capacity).unwrap();
    /// let a = model.create_set(object_type, &[1]).unwrap();
    /// let b = model.create_set(object_type, &[0]).unwrap();
    /// let map = FxHashMap::from_iter(
    ///     vec![(vec![0, 0, 0, 0], a.clone()), (vec![1, 1, 1, 1], a)]
    /// );
    /// let table = model.add_table("table", map, b).unwrap();
    /// let set = model.create_set(object_type, &[0, 1]).unwrap();
    /// let variable = model.add_set_variable("variable", object_type, set).unwrap();
    /// let state = model.target.clone();
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let indices = vec![
    ///     ArgumentExpression::from(variable),
    ///     ArgumentExpression::from(0),
    ///     ArgumentExpression::from(0),
    ///     ArgumentExpression::from(0),
    /// ];
    /// let expression = table.union(capacity, indices);
    /// let expected = model.create_set(object_type, &[0, 1]).unwrap();
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     expected,
    /// );
    ///
    /// let indices = vec![
    ///     ArgumentExpression::from(variable),
    ///     ArgumentExpression::from(variable),
    ///     ArgumentExpression::from(variable),
    ///     ArgumentExpression::from(variable),
    /// ];
    /// let expression = table.union(capacity, indices);
    /// let expected = model.create_set(object_type, &[0, 1]).unwrap();
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     expected,
    /// );
    /// ```
    #[inline]
    pub fn union<T>(&self, capacity: usize, indices: Vec<T>) -> SetExpression
    where
        ArgumentExpression: From<T>,
    {
        let indices = indices.into_iter().map(ArgumentExpression::from).collect();
        SetExpression::Reduce(SetReduceExpression::Table(
            SetReduceOperator::Union,
            capacity,
            self.id(),
            indices,
        ))
    }

    /// Returns the intersection of sets in a table.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    /// use dypdl::expression::*;
    /// use rustc_hash::FxHashMap;
    ///
    /// let capacity = 2;
    /// let mut model = Model::default();
    /// let object_type = model.add_object_type("object", capacity).unwrap();
    /// let a = model.create_set(object_type, &[1]).unwrap();
    /// let b = model.create_set(object_type, &[0]).unwrap();
    /// let map = FxHashMap::from_iter(
    ///     vec![(vec![0, 0, 0, 0], a.clone()), (vec![1, 1, 1, 1], a)]
    /// );
    /// let table = model.add_table("table", map, b).unwrap();
    /// let set = model.create_set(object_type, &[0, 1]).unwrap();
    /// let variable = model.add_set_variable("variable", object_type, set).unwrap();
    /// let state = model.target.clone();
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let indices = vec![
    ///     ArgumentExpression::from(variable),
    ///     ArgumentExpression::from(0),
    ///     ArgumentExpression::from(0),
    ///     ArgumentExpression::from(0),
    /// ];
    /// let expression = table.intersection(capacity, indices);
    /// let expected = model.create_set(object_type, &[]).unwrap();
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     expected,
    /// );
    ///
    /// let indices = vec![
    ///     ArgumentExpression::from(variable),
    ///     ArgumentExpression::from(variable),
    ///     ArgumentExpression::from(variable),
    ///     ArgumentExpression::from(variable),
    /// ];
    /// let expression = table.intersection(capacity, indices);
    /// let expected = model.create_set(object_type, &[]).unwrap();
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     expected,
    /// );
    /// ```
    #[inline]
    pub fn intersection<T>(&self, capacity: usize, indices: Vec<T>) -> SetExpression
    where
        ArgumentExpression: From<T>,
    {
        let indices = indices.into_iter().map(ArgumentExpression::from).collect();
        SetExpression::Reduce(SetReduceExpression::Table(
            SetReduceOperator::Intersection,
            capacity,
            self.id(),
            indices,
        ))
    }

    /// Returns the symmetric difference (disjunctive union) of sets in a table.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    /// use dypdl::expression::*;
    /// use rustc_hash::FxHashMap;
    ///
    /// let capacity = 2;
    /// let mut model = Model::default();
    /// let object_type = model.add_object_type("object", capacity).unwrap();
    /// let a = model.create_set(object_type, &[1]).unwrap();
    /// let b = model.create_set(object_type, &[0]).unwrap();
    /// let map = FxHashMap::from_iter(
    ///     vec![(vec![0, 0, 0, 0], a.clone()), (vec![1, 1, 1, 1], a)]
    /// );
    /// let table = model.add_table("table", map, b).unwrap();
    /// let set = model.create_set(object_type, &[0, 1]).unwrap();
    /// let variable = model.add_set_variable("variable", object_type, set).unwrap();
    /// let state = model.target.clone();
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let indices = vec![
    ///     ArgumentExpression::from(variable),
    ///     ArgumentExpression::from(0),
    ///     ArgumentExpression::from(0),
    ///     ArgumentExpression::from(0),
    /// ];
    /// let expression = table.symmetric_difference(capacity, indices);
    /// let expected = model.create_set(object_type, &[0, 1]).unwrap();
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     expected,
    /// );
    ///
    /// let indices = vec![
    ///     ArgumentExpression::from(variable),
    ///     ArgumentExpression::from(variable),
    ///     ArgumentExpression::from(variable),
    ///     ArgumentExpression::from(variable),
    /// ];
    /// let expression = table.symmetric_difference(capacity, indices);
    /// let expected = model.create_set(object_type, &[]).unwrap();
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     expected,
    /// );
    /// ```
    #[inline]
    pub fn symmetric_difference<T>(&self, capacity: usize, indices: Vec<T>) -> SetExpression
    where
        ArgumentExpression: From<T>,
    {
        let indices = indices.into_iter().map(ArgumentExpression::from).collect();
        SetExpression::Reduce(SetReduceExpression::Table(
            SetReduceOperator::SymmetricDifference,
            capacity,
            self.id(),
            indices,
        ))
    }
}

impl IfThenElse<SetExpression> for Condition {
    #[inline]
    fn if_then_else<U, V>(self, lhs: U, rhs: V) -> SetExpression
    where
        SetExpression: From<U> + From<V>,
    {
        SetExpression::If(
            Box::new(self),
            Box::new(SetExpression::from(lhs)),
            Box::new(SetExpression::from(rhs)),
        )
    }
}

macro_rules! impl_set_ops {
    ($T:ty,$U:ty) => {
        impl ops::BitOr<$U> for $T {
            type Output = SetExpression;

            /// Returns an expression representing the union.
            #[inline]
            fn bitor(self, rhs: $U) -> SetExpression {
                SetExpression::from(self) | SetExpression::from(rhs)
            }
        }

        impl ops::Sub<$U> for $T {
            type Output = SetExpression;

            /// Returns an expression representing the difference.
            #[inline]
            fn sub(self, rhs: $U) -> SetExpression {
                SetExpression::from(self) - SetExpression::from(rhs)
            }
        }

        impl ops::BitAnd<$U> for $T {
            type Output = SetExpression;

            /// Returns an expression representing the intersection.
            #[inline]
            fn bitand(self, rhs: $U) -> SetExpression {
                SetExpression::from(self) & SetExpression::from(rhs)
            }
        }
    };
}

macro_rules! impl_set_element_ops {
    ($T:ty,$U:ty) => {
        impl SetElementOperation<$U> for $T {
            #[inline]
            fn add(self, rhs: $U) -> SetExpression {
                SetExpression::SetElementOperation(
                    SetElementOperator::Add,
                    ElementExpression::from(rhs),
                    Box::new(SetExpression::from(self)),
                )
            }

            #[inline]
            fn remove(self, rhs: $U) -> SetExpression {
                SetExpression::SetElementOperation(
                    SetElementOperator::Remove,
                    ElementExpression::from(rhs),
                    Box::new(SetExpression::from(self)),
                )
            }
        }
    };
}

impl_set_ops!(SetExpression, Set);
impl_set_ops!(SetExpression, SetVariable);
impl_set_element_ops!(SetExpression, ElementExpression);
impl_set_element_ops!(SetExpression, Element);
impl_set_element_ops!(SetExpression, ElementVariable);
impl_set_element_ops!(SetExpression, ElementResourceVariable);
impl_set_ops!(Set, SetExpression);
impl_set_ops!(Set, SetVariable);
impl_set_element_ops!(Set, ElementExpression);
impl_set_element_ops!(Set, ElementVariable);
impl_set_element_ops!(Set, ElementResourceVariable);
impl_set_ops!(SetVariable, SetExpression);
impl_set_ops!(SetVariable, Set);
impl_set_ops!(SetVariable, SetVariable);
impl_set_element_ops!(SetVariable, ElementExpression);
impl_set_element_ops!(SetVariable, Element);
impl_set_element_ops!(SetVariable, ElementVariable);
impl_set_element_ops!(SetVariable, ElementResourceVariable);

impl SetExpression {
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
    /// let mut model = Model::default();
    /// let object_type = model.add_object_type("object_type", 2).unwrap();
    /// let set = model.create_set(object_type, &[0, 1]).unwrap();
    /// let variable = model.add_set_variable("variable", object_type, set.clone()).unwrap();
    /// let state = model.target.clone();
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    ///
    /// let expression = SetExpression::from(variable);
    /// assert_eq!(
    ///     expression.eval(
    ///         &state, &mut function_cache, &model.state_functions, &model.table_registry,
    ///     ),
    ///     set,
    /// );
    /// ```
    pub fn eval<T: StateInterface>(
        &self,
        state: &T,
        function_cache: &mut StateFunctionCache,
        state_functions: &StateFunctions,
        registry: &TableRegistry,
    ) -> Set {
        match self {
            Self::Reference(expression) => expression
                .eval(state, function_cache, state_functions, registry)
                .clone(),
            Self::StateFunction(i) => function_cache
                .get_set_value(*i, state, state_functions, registry)
                .clone(),
            Self::Complement(set) => {
                let mut set = set.eval(state, function_cache, state_functions, registry);
                set.toggle_range(..);
                set
            }
            Self::SetOperation(op, x, y) => match (op, x.as_ref(), y.as_ref()) {
                (op, x, SetExpression::Reference(y)) => {
                    let x = x.eval(state, function_cache, state_functions, registry);
                    let y = y.eval(state, function_cache, state_functions, registry);
                    Self::eval_set_operation(op, x, y)
                }
                (SetOperator::Intersection, SetExpression::Reference(x), y)
                | (SetOperator::Union, SetExpression::Reference(x), y) => {
                    let y = y.eval(state, function_cache, state_functions, registry);
                    let x = x.eval(state, function_cache, state_functions, registry);
                    Self::eval_set_operation(op, y, x)
                }
                (op, x, SetExpression::StateFunction(i)) => {
                    let x = x.eval(state, function_cache, state_functions, registry);
                    let y = function_cache.get_set_value(*i, state, state_functions, registry);
                    Self::eval_set_operation(op, x, y)
                }
                (SetOperator::Intersection, SetExpression::StateFunction(i), y)
                | (SetOperator::Union, SetExpression::StateFunction(i), y) => {
                    let y = y.eval(state, function_cache, state_functions, registry);
                    let x = function_cache.get_set_value(*i, state, state_functions, registry);
                    Self::eval_set_operation(op, y, x)
                }
                (op, x, y) => {
                    let x = x.eval(state, function_cache, state_functions, registry);
                    let y = y.eval(state, function_cache, state_functions, registry);
                    Self::eval_set_operation(op, x, &y)
                }
            },
            Self::SetElementOperation(op, element, set) => {
                let set = set.eval(state, function_cache, state_functions, registry);
                let element = element.eval(state, function_cache, state_functions, registry);
                Self::eval_set_element_operation(op, element, set)
            }
            Self::Reduce(expression) => {
                expression.eval(state, function_cache, state_functions, registry)
            }
            Self::FromVector(capacity, vector) => match vector.as_ref() {
                VectorExpression::Reference(ReferenceExpression::Constant(vector)) => {
                    let mut set = Set::with_capacity(*capacity);
                    vector.iter().for_each(|v| set.insert(*v));
                    set
                }
                vector => {
                    let mut set = Set::with_capacity(*capacity);
                    vector
                        .eval(state, function_cache, state_functions, registry)
                        .into_iter()
                        .for_each(|v| set.insert(v));
                    set
                }
            },
            Self::If(condition, x, y) => {
                if condition.eval(state, function_cache, state_functions, registry) {
                    x.eval(state, function_cache, state_functions, registry)
                } else {
                    y.eval(state, function_cache, state_functions, registry)
                }
            }
        }
    }

    /// Returns a simplified version by precomputation.
    ///
    /// # Panics
    ///
    /// Panics if a min/max reduce operation is performed on an empty set or vector.
    pub fn simplify(&self, registry: &TableRegistry) -> SetExpression {
        match self {
            Self::Reference(expression) => {
                Self::Reference(expression.simplify(registry, &registry.set_tables))
            }
            Self::Complement(expression) => match expression.simplify(registry) {
                Self::Reference(ReferenceExpression::Constant(mut set)) => {
                    set.toggle_range(..);
                    Self::Reference(ReferenceExpression::Constant(set))
                }
                Self::Complement(expression) => *expression,
                expression => Self::Complement(Box::new(expression)),
            },
            Self::SetOperation(op, x, y) => {
                match (op, x.simplify(registry), y.simplify(registry)) {
                    (
                        op,
                        Self::Reference(ReferenceExpression::Constant(x)),
                        Self::Reference(ReferenceExpression::Constant(y)),
                    ) => Self::Reference(ReferenceExpression::Constant(Self::eval_set_operation(
                        op, x, &y,
                    ))),
                    (
                        SetOperator::Union,
                        Self::Reference(ReferenceExpression::Variable(x)),
                        Self::Reference(ReferenceExpression::Variable(y)),
                    )
                    | (
                        SetOperator::Intersection,
                        Self::Reference(ReferenceExpression::Variable(x)),
                        Self::Reference(ReferenceExpression::Variable(y)),
                    ) if x == y => Self::Reference(ReferenceExpression::Variable(x)),
                    (SetOperator::Union, Self::StateFunction(x), Self::StateFunction(y))
                    | (SetOperator::Intersection, Self::StateFunction(x), Self::StateFunction(y))
                        if x == y =>
                    {
                        Self::StateFunction(x)
                    }
                    (op, x, y) => Self::SetOperation(op.clone(), Box::new(x), Box::new(y)),
                }
            }
            Self::SetElementOperation(op, element, set) => {
                match (set.simplify(registry), element.simplify(registry)) {
                    (
                        Self::Reference(ReferenceExpression::Constant(set)),
                        ElementExpression::Constant(element),
                    ) => Self::Reference(ReferenceExpression::Constant(
                        Self::eval_set_element_operation(op, element, set),
                    )),
                    (set, element) => Self::SetElementOperation(op.clone(), element, Box::new(set)),
                }
            }
            Self::Reduce(expression) => match expression.simplify(registry) {
                SetReduceExpression::Constant(set) => {
                    Self::Reference(ReferenceExpression::Constant(set))
                }
                expression => Self::Reduce(expression),
            },
            Self::FromVector(capacity, vector) => match vector.simplify(registry) {
                VectorExpression::Reference(ReferenceExpression::Constant(vector)) => {
                    let mut set = Set::with_capacity(*capacity);
                    vector.into_iter().for_each(|v| set.insert(v));
                    Self::Reference(ReferenceExpression::Constant(set))
                }
                vector => Self::FromVector(*capacity, Box::new(vector)),
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

    fn eval_set_operation(op: &SetOperator, mut x: Set, y: &Set) -> Set {
        match op {
            SetOperator::Union => {
                x.union_with(y);
                x
            }
            SetOperator::Difference => {
                x.difference_with(y);
                x
            }
            SetOperator::Intersection => {
                x.intersect_with(y);
                x
            }
        }
    }

    fn eval_set_element_operation(op: &SetElementOperator, element: Element, mut set: Set) -> Set {
        match op {
            SetElementOperator::Add => {
                set.insert(element);
                set
            }
            SetElementOperator::Remove => {
                set.set(element, false);
                set
            }
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
    fn set_default() {
        assert_eq!(
            SetExpression::default(),
            SetExpression::Reference(ReferenceExpression::Constant(Set::default()))
        );
    }

    #[test]
    fn set_from() {
        let mut metadata = StateMetadata::default();
        let ob = metadata.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let s = metadata.create_set(ob, &[1, 2, 3]);
        assert!(s.is_ok());
        let s = s.unwrap();
        assert_eq!(
            SetExpression::from(s.clone()),
            SetExpression::Reference(ReferenceExpression::Constant(s))
        );

        let v = metadata.add_set_variable(String::from("sv"), ob);
        assert!(v.is_ok());
        let v = v.unwrap();
        assert_eq!(
            SetExpression::from(v),
            SetExpression::Reference(ReferenceExpression::Variable(v.id()))
        );
    }

    #[test]
    fn set_not() {
        let expression = SetExpression::Reference(ReferenceExpression::Variable(0));
        assert_eq!(
            !expression,
            SetExpression::Complement(Box::new(SetExpression::Reference(
                ReferenceExpression::Variable(0)
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
            !v,
            SetExpression::Complement(Box::new(SetExpression::Reference(
                ReferenceExpression::Variable(v.id())
            )))
        );
    }

    #[test]
    fn set_bitor() {
        let expression1 = SetExpression::Reference(ReferenceExpression::Variable(0));
        let expression2 = SetExpression::Reference(ReferenceExpression::Variable(1));
        assert_eq!(
            expression1.clone() | expression2.clone(),
            SetExpression::SetOperation(
                SetOperator::Union,
                Box::new(expression1),
                Box::new(expression2),
            )
        );

        let mut metadata = StateMetadata::default();
        let ob = metadata.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let expression1 = SetExpression::Reference(ReferenceExpression::Variable(0));
        let s = metadata.create_set(ob, &[0, 1, 2]);
        assert!(s.is_ok());
        let s = s.unwrap();
        assert_eq!(
            expression1.clone() | s.clone(),
            SetExpression::SetOperation(
                SetOperator::Union,
                Box::new(expression1.clone()),
                Box::new(SetExpression::Reference(ReferenceExpression::Constant(
                    s.clone()
                ))),
            )
        );
        assert_eq!(
            s.clone() | expression1.clone(),
            SetExpression::SetOperation(
                SetOperator::Union,
                Box::new(SetExpression::Reference(ReferenceExpression::Constant(
                    s.clone()
                ))),
                Box::new(expression1),
            )
        );

        let expression1 = SetExpression::Reference(ReferenceExpression::Variable(0));
        let v = metadata.add_set_variable(String::from("sv"), ob);
        assert!(v.is_ok());
        let v = v.unwrap();
        assert_eq!(
            expression1.clone() | v,
            SetExpression::SetOperation(
                SetOperator::Union,
                Box::new(expression1.clone()),
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                    v.id()
                ))),
            )
        );
        assert_eq!(
            v | expression1.clone(),
            SetExpression::SetOperation(
                SetOperator::Union,
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                    v.id()
                ))),
                Box::new(expression1),
            )
        );

        assert_eq!(
            s.clone() | v,
            SetExpression::SetOperation(
                SetOperator::Union,
                Box::new(SetExpression::Reference(ReferenceExpression::Constant(
                    s.clone()
                ))),
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                    v.id()
                ))),
            )
        );
        assert_eq!(
            v | s.clone(),
            SetExpression::SetOperation(
                SetOperator::Union,
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                    v.id()
                ))),
                Box::new(SetExpression::Reference(ReferenceExpression::Constant(s))),
            )
        );
    }

    #[test]
    fn set_sub() {
        let expression1 = SetExpression::Reference(ReferenceExpression::Variable(0));
        let expression2 = SetExpression::Reference(ReferenceExpression::Variable(1));
        assert_eq!(
            expression1.clone() - expression2.clone(),
            SetExpression::SetOperation(
                SetOperator::Difference,
                Box::new(expression1),
                Box::new(expression2),
            )
        );

        let mut metadata = StateMetadata::default();
        let ob = metadata.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let expression1 = SetExpression::Reference(ReferenceExpression::Variable(0));
        let s = metadata.create_set(ob, &[0, 1, 2]);
        assert!(s.is_ok());
        let s = s.unwrap();
        assert_eq!(
            expression1.clone() - s.clone(),
            SetExpression::SetOperation(
                SetOperator::Difference,
                Box::new(expression1.clone()),
                Box::new(SetExpression::Reference(ReferenceExpression::Constant(
                    s.clone()
                ))),
            )
        );
        assert_eq!(
            s.clone() - expression1.clone(),
            SetExpression::SetOperation(
                SetOperator::Difference,
                Box::new(SetExpression::Reference(ReferenceExpression::Constant(
                    s.clone()
                ))),
                Box::new(expression1),
            )
        );

        let expression1 = SetExpression::Reference(ReferenceExpression::Variable(0));
        let v = metadata.add_set_variable(String::from("sv"), ob);
        assert!(v.is_ok());
        let v = v.unwrap();
        assert_eq!(
            expression1.clone() - v,
            SetExpression::SetOperation(
                SetOperator::Difference,
                Box::new(expression1.clone()),
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                    v.id()
                ))),
            )
        );
        assert_eq!(
            v - expression1.clone(),
            SetExpression::SetOperation(
                SetOperator::Difference,
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                    v.id()
                ))),
                Box::new(expression1),
            )
        );

        assert_eq!(
            s.clone() - v,
            SetExpression::SetOperation(
                SetOperator::Difference,
                Box::new(SetExpression::Reference(ReferenceExpression::Constant(
                    s.clone()
                ))),
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                    v.id()
                ))),
            )
        );
        assert_eq!(
            v - s.clone(),
            SetExpression::SetOperation(
                SetOperator::Difference,
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                    v.id()
                ))),
                Box::new(SetExpression::Reference(ReferenceExpression::Constant(s))),
            )
        );
    }

    #[test]
    fn set_bitand() {
        let expression1 = SetExpression::Reference(ReferenceExpression::Variable(0));
        let expression2 = SetExpression::Reference(ReferenceExpression::Variable(1));
        assert_eq!(
            expression1.clone() & expression2.clone(),
            SetExpression::SetOperation(
                SetOperator::Intersection,
                Box::new(expression1),
                Box::new(expression2),
            )
        );

        let mut metadata = StateMetadata::default();
        let ob = metadata.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let expression1 = SetExpression::Reference(ReferenceExpression::Variable(0));
        let s = metadata.create_set(ob, &[0, 1, 2]);
        assert!(s.is_ok());
        let s = s.unwrap();
        assert_eq!(
            expression1.clone() & s.clone(),
            SetExpression::SetOperation(
                SetOperator::Intersection,
                Box::new(expression1.clone()),
                Box::new(SetExpression::Reference(ReferenceExpression::Constant(
                    s.clone()
                ))),
            )
        );
        assert_eq!(
            s.clone() & expression1.clone(),
            SetExpression::SetOperation(
                SetOperator::Intersection,
                Box::new(SetExpression::Reference(ReferenceExpression::Constant(
                    s.clone()
                ))),
                Box::new(expression1),
            )
        );

        let expression1 = SetExpression::Reference(ReferenceExpression::Variable(0));
        let v = metadata.add_set_variable(String::from("sv"), ob);
        assert!(v.is_ok());
        let v = v.unwrap();
        assert_eq!(
            expression1.clone() & v,
            SetExpression::SetOperation(
                SetOperator::Intersection,
                Box::new(expression1.clone()),
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                    v.id()
                ))),
            )
        );
        assert_eq!(
            v & expression1.clone(),
            SetExpression::SetOperation(
                SetOperator::Intersection,
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                    v.id()
                ))),
                Box::new(expression1),
            )
        );

        assert_eq!(
            s.clone() & v,
            SetExpression::SetOperation(
                SetOperator::Intersection,
                Box::new(SetExpression::Reference(ReferenceExpression::Constant(
                    s.clone()
                ))),
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                    v.id()
                ))),
            )
        );
        assert_eq!(
            v & s.clone(),
            SetExpression::SetOperation(
                SetOperator::Intersection,
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                    v.id()
                ))),
                Box::new(SetExpression::Reference(ReferenceExpression::Constant(s))),
            )
        );
    }

    #[test]
    fn set_add() {
        let expression1 = SetExpression::Reference(ReferenceExpression::Variable(0));
        let expression2 = ElementExpression::Constant(0);
        assert_eq!(
            expression1.clone().add(expression2.clone()),
            SetExpression::SetElementOperation(
                SetElementOperator::Add,
                expression2.clone(),
                Box::new(expression1.clone()),
            )
        );
        assert_eq!(
            expression1.clone().add(1),
            SetExpression::SetElementOperation(
                SetElementOperator::Add,
                ElementExpression::Constant(1),
                Box::new(expression1.clone()),
            )
        );

        let mut metadata = StateMetadata::default();
        let ob = metadata.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let ev = metadata.add_element_variable(String::from("ev"), ob);
        assert!(ev.is_ok());
        let ev = ev.unwrap();
        assert_eq!(
            expression1.clone().add(ev),
            SetExpression::SetElementOperation(
                SetElementOperator::Add,
                ElementExpression::Variable(ev.id()),
                Box::new(expression1.clone()),
            )
        );

        let erv = metadata.add_element_resource_variable(String::from("ev"), ob, true);
        assert!(erv.is_ok());
        let erv = erv.unwrap();
        assert_eq!(
            expression1.clone().add(erv),
            SetExpression::SetElementOperation(
                SetElementOperator::Add,
                ElementExpression::ResourceVariable(erv.id()),
                Box::new(expression1),
            )
        );

        let sv = metadata.add_set_variable(String::from("sv"), ob);
        assert!(sv.is_ok());
        let sv = sv.unwrap();
        assert_eq!(
            sv.add(expression2.clone()),
            SetExpression::SetElementOperation(
                SetElementOperator::Add,
                expression2,
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                    sv.id()
                ))),
            )
        );
        assert_eq!(
            sv.add(1),
            SetExpression::SetElementOperation(
                SetElementOperator::Add,
                ElementExpression::Constant(1),
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                    sv.id()
                ))),
            )
        );
        assert_eq!(
            sv.add(ev),
            SetExpression::SetElementOperation(
                SetElementOperator::Add,
                ElementExpression::Variable(ev.id()),
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                    sv.id()
                ))),
            )
        );
        assert_eq!(
            sv.add(erv),
            SetExpression::SetElementOperation(
                SetElementOperator::Add,
                ElementExpression::ResourceVariable(erv.id()),
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                    sv.id()
                ))),
            )
        );
    }

    #[test]
    fn set_remove() {
        let expression1 = SetExpression::Reference(ReferenceExpression::Variable(0));
        let expression2 = ElementExpression::Constant(0);
        assert_eq!(
            expression1.clone().remove(expression2.clone()),
            SetExpression::SetElementOperation(
                SetElementOperator::Remove,
                expression2.clone(),
                Box::new(expression1.clone()),
            )
        );
        assert_eq!(
            expression1.clone().remove(1),
            SetExpression::SetElementOperation(
                SetElementOperator::Remove,
                ElementExpression::Constant(1),
                Box::new(expression1.clone()),
            )
        );

        let mut metadata = StateMetadata::default();
        let ob = metadata.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let ev = metadata.add_element_variable(String::from("ev"), ob);
        assert!(ev.is_ok());
        let ev = ev.unwrap();
        assert_eq!(
            expression1.clone().remove(ev),
            SetExpression::SetElementOperation(
                SetElementOperator::Remove,
                ElementExpression::Variable(ev.id()),
                Box::new(expression1.clone()),
            )
        );

        let erv = metadata.add_element_resource_variable(String::from("ev"), ob, true);
        assert!(erv.is_ok());
        let erv = erv.unwrap();
        assert_eq!(
            expression1.clone().remove(erv),
            SetExpression::SetElementOperation(
                SetElementOperator::Remove,
                ElementExpression::ResourceVariable(erv.id()),
                Box::new(expression1),
            )
        );

        let sv = metadata.add_set_variable(String::from("sv"), ob);
        assert!(sv.is_ok());
        let sv = sv.unwrap();
        assert_eq!(
            sv.remove(expression2.clone()),
            SetExpression::SetElementOperation(
                SetElementOperator::Remove,
                expression2,
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                    sv.id()
                ))),
            )
        );
        assert_eq!(
            sv.remove(1),
            SetExpression::SetElementOperation(
                SetElementOperator::Remove,
                ElementExpression::Constant(1),
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                    sv.id()
                ))),
            )
        );
        assert_eq!(
            sv.remove(ev),
            SetExpression::SetElementOperation(
                SetElementOperator::Remove,
                ElementExpression::Variable(ev.id()),
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                    sv.id()
                ))),
            )
        );
        assert_eq!(
            sv.remove(erv),
            SetExpression::SetElementOperation(
                SetElementOperator::Remove,
                ElementExpression::ResourceVariable(erv.id()),
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                    sv.id()
                ))),
            )
        );
    }

    #[test]
    fn set_table_1d_element() {
        let mut registry = TableRegistry::default();

        let t = registry.add_table_1d(String::from("t"), vec![Set::default()]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(
            t.element(0),
            SetExpression::Reference(ReferenceExpression::Table(TableExpression::Table1D(
                t.id(),
                ElementExpression::Constant(0)
            )))
        );
    }

    #[test]
    fn set_table_1d_union() {
        let mut registry = TableRegistry::default();

        let t = registry.add_table_1d(
            String::from("t"),
            vec![
                {
                    let mut set = Set::with_capacity(3);
                    set.insert(0);
                    set.insert(1);
                    set
                },
                {
                    let mut set = Set::with_capacity(3);
                    set.insert(1);
                    set.insert(2);
                    set
                },
            ],
        );
        assert!(t.is_ok());
        let t = t.unwrap();
        let x = SetExpression::Reference(ReferenceExpression::Constant({
            let mut set = Set::with_capacity(2);
            set.insert(0);
            set.insert(1);
            set
        }));
        assert_eq!(
            t.union(3, x.clone()),
            SetExpression::Reduce(SetReduceExpression::Table1D(
                SetReduceOperator::Union,
                3,
                t.id(),
                Box::new(ArgumentExpression::Set(x))
            ))
        );
    }

    #[test]
    fn set_table_1d_intersection() {
        let mut registry = TableRegistry::default();

        let t = registry.add_table_1d(
            String::from("t"),
            vec![
                {
                    let mut set = Set::with_capacity(3);
                    set.insert(0);
                    set.insert(1);
                    set
                },
                {
                    let mut set = Set::with_capacity(3);
                    set.insert(1);
                    set.insert(2);
                    set
                },
            ],
        );
        assert!(t.is_ok());
        let t = t.unwrap();
        let x = SetExpression::Reference(ReferenceExpression::Constant({
            let mut set = Set::with_capacity(2);
            set.insert(0);
            set.insert(1);
            set
        }));
        assert_eq!(
            t.intersection(3, x.clone()),
            SetExpression::Reduce(SetReduceExpression::Table1D(
                SetReduceOperator::Intersection,
                3,
                t.id(),
                Box::new(ArgumentExpression::Set(x))
            ))
        );
    }

    #[test]
    fn set_table_1d_symmetric_difference() {
        let mut registry = TableRegistry::default();

        let t = registry.add_table_1d(
            String::from("t"),
            vec![
                {
                    let mut set = Set::with_capacity(3);
                    set.insert(0);
                    set.insert(1);
                    set
                },
                {
                    let mut set = Set::with_capacity(3);
                    set.insert(1);
                    set.insert(2);
                    set
                },
            ],
        );
        assert!(t.is_ok());
        let t = t.unwrap();
        let x = SetExpression::Reference(ReferenceExpression::Constant({
            let mut set = Set::with_capacity(2);
            set.insert(0);
            set.insert(1);
            set
        }));
        assert_eq!(
            t.symmetric_difference(3, x.clone()),
            SetExpression::Reduce(SetReduceExpression::Table1D(
                SetReduceOperator::SymmetricDifference,
                3,
                t.id(),
                Box::new(ArgumentExpression::Set(x))
            ))
        );
    }

    #[test]
    fn set_table_2d_element() {
        let mut registry = TableRegistry::default();

        let t = registry.add_table_2d(String::from("t"), vec![vec![Set::default()]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(
            t.element(0, 0),
            SetExpression::Reference(ReferenceExpression::Table(TableExpression::Table2D(
                t.id(),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
            )))
        );
    }

    #[test]
    fn set_table_2d_union() {
        let mut registry = TableRegistry::default();

        let t = registry.add_table_2d(
            String::from("t"),
            vec![vec![
                {
                    let mut set = Set::with_capacity(3);
                    set.insert(0);
                    set.insert(1);
                    set
                },
                {
                    let mut set = Set::with_capacity(3);
                    set.insert(1);
                    set.insert(2);
                    set
                },
            ]],
        );
        assert!(t.is_ok());
        let t = t.unwrap();
        let y = SetExpression::Reference(ReferenceExpression::Constant({
            let mut set = Set::with_capacity(2);
            set.insert(0);
            set.insert(1);
            set
        }));
        assert_eq!(
            t.union(3, 0, y.clone()),
            SetExpression::Reduce(SetReduceExpression::Table2D(
                SetReduceOperator::Union,
                3,
                t.id(),
                Box::new(ArgumentExpression::Element(ElementExpression::Constant(0))),
                Box::new(ArgumentExpression::Set(y))
            ))
        );
    }

    #[test]
    fn set_table_2d_intersection() {
        let mut registry = TableRegistry::default();

        let t = registry.add_table_2d(
            String::from("t"),
            vec![vec![
                {
                    let mut set = Set::with_capacity(3);
                    set.insert(0);
                    set.insert(1);
                    set
                },
                {
                    let mut set = Set::with_capacity(3);
                    set.insert(1);
                    set.insert(2);
                    set
                },
            ]],
        );
        assert!(t.is_ok());
        let t = t.unwrap();
        let y = SetExpression::Reference(ReferenceExpression::Constant({
            let mut set = Set::with_capacity(2);
            set.insert(0);
            set.insert(1);
            set
        }));
        assert_eq!(
            t.intersection(3, 0, y.clone()),
            SetExpression::Reduce(SetReduceExpression::Table2D(
                SetReduceOperator::Intersection,
                3,
                t.id(),
                Box::new(ArgumentExpression::Element(ElementExpression::Constant(0))),
                Box::new(ArgumentExpression::Set(y))
            ))
        );
    }

    #[test]
    fn set_table_2d_symmetric_difference() {
        let mut registry = TableRegistry::default();

        let t = registry.add_table_2d(
            String::from("t"),
            vec![vec![
                {
                    let mut set = Set::with_capacity(3);
                    set.insert(0);
                    set.insert(1);
                    set
                },
                {
                    let mut set = Set::with_capacity(3);
                    set.insert(1);
                    set.insert(2);
                    set
                },
            ]],
        );
        assert!(t.is_ok());
        let t = t.unwrap();
        let y = SetExpression::Reference(ReferenceExpression::Constant({
            let mut set = Set::with_capacity(2);
            set.insert(0);
            set.insert(1);
            set
        }));
        assert_eq!(
            t.symmetric_difference(3, 0, y.clone()),
            SetExpression::Reduce(SetReduceExpression::Table2D(
                SetReduceOperator::SymmetricDifference,
                3,
                t.id(),
                Box::new(ArgumentExpression::Element(ElementExpression::Constant(0))),
                Box::new(ArgumentExpression::Set(y))
            ))
        );
    }

    #[test]
    fn set_table_3d_element() {
        let mut registry = TableRegistry::default();

        let t = registry.add_table_3d(String::from("t"), vec![vec![vec![Set::default()]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(
            t.element(0, 0, 0),
            SetExpression::Reference(ReferenceExpression::Table(TableExpression::Table3D(
                t.id(),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
            )))
        );
    }

    #[test]
    fn set_table_3d_union() {
        let mut registry = TableRegistry::default();

        let t = registry.add_table_3d(
            String::from("t"),
            vec![vec![vec![
                {
                    let mut set = Set::with_capacity(3);
                    set.insert(0);
                    set.insert(1);
                    set
                },
                {
                    let mut set = Set::with_capacity(3);
                    set.insert(1);
                    set.insert(2);
                    set
                },
            ]]],
        );
        assert!(t.is_ok());
        let t = t.unwrap();
        let z = SetExpression::Reference(ReferenceExpression::Constant({
            let mut set = Set::with_capacity(2);
            set.insert(0);
            set.insert(1);
            set
        }));
        assert_eq!(
            t.union(3, 0, 0, z.clone()),
            SetExpression::Reduce(SetReduceExpression::Table3D(
                SetReduceOperator::Union,
                3,
                t.id(),
                Box::new(ArgumentExpression::Element(ElementExpression::Constant(0))),
                Box::new(ArgumentExpression::Element(ElementExpression::Constant(0))),
                Box::new(ArgumentExpression::Set(z))
            ))
        );
    }

    #[test]
    fn set_table_3d_intersection() {
        let mut registry = TableRegistry::default();

        let t = registry.add_table_3d(
            String::from("t"),
            vec![vec![vec![
                {
                    let mut set = Set::with_capacity(3);
                    set.insert(0);
                    set.insert(1);
                    set
                },
                {
                    let mut set = Set::with_capacity(3);
                    set.insert(1);
                    set.insert(2);
                    set
                },
            ]]],
        );
        assert!(t.is_ok());
        let t = t.unwrap();
        let z = SetExpression::Reference(ReferenceExpression::Constant({
            let mut set = Set::with_capacity(2);
            set.insert(0);
            set.insert(1);
            set
        }));
        assert_eq!(
            t.intersection(3, 0, 0, z.clone()),
            SetExpression::Reduce(SetReduceExpression::Table3D(
                SetReduceOperator::Intersection,
                3,
                t.id(),
                Box::new(ArgumentExpression::Element(ElementExpression::Constant(0))),
                Box::new(ArgumentExpression::Element(ElementExpression::Constant(0))),
                Box::new(ArgumentExpression::Set(z))
            ))
        );
    }

    #[test]
    fn set_table_3d_symmetric_difference() {
        let mut registry = TableRegistry::default();

        let t = registry.add_table_3d(
            String::from("t"),
            vec![vec![vec![
                {
                    let mut set = Set::with_capacity(3);
                    set.insert(0);
                    set.insert(1);
                    set
                },
                {
                    let mut set = Set::with_capacity(3);
                    set.insert(1);
                    set.insert(2);
                    set
                },
            ]]],
        );
        assert!(t.is_ok());
        let t = t.unwrap();
        let z = SetExpression::Reference(ReferenceExpression::Constant({
            let mut set = Set::with_capacity(2);
            set.insert(0);
            set.insert(1);
            set
        }));
        assert_eq!(
            t.symmetric_difference(3, 0, 0, z.clone()),
            SetExpression::Reduce(SetReduceExpression::Table3D(
                SetReduceOperator::SymmetricDifference,
                3,
                t.id(),
                Box::new(ArgumentExpression::Element(ElementExpression::Constant(0))),
                Box::new(ArgumentExpression::Element(ElementExpression::Constant(0))),
                Box::new(ArgumentExpression::Set(z))
            ))
        );
    }

    #[test]
    fn set_table_element() {
        let mut registry = TableRegistry::default();

        let t = registry.add_table(String::from("t"), FxHashMap::default(), Set::default());
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(
            t.element(vec![0, 0, 0, 0]),
            SetExpression::Reference(ReferenceExpression::Table(TableExpression::Table(
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
    fn set_table_union() {
        let mut registry = TableRegistry::default();

        let t = registry.add_table(
            String::from("t"),
            {
                let mut map = FxHashMap::default();
                let mut set = Set::with_capacity(3);
                set.insert(0);
                set.insert(1);
                map.insert(vec![0, 0, 0, 0], set);
                map
            },
            {
                let mut set = Set::with_capacity(3);
                set.insert(1);
                set.insert(2);
                set
            },
        );
        assert!(t.is_ok());
        let t = t.unwrap();
        let args = vec![
            ArgumentExpression::Element(ElementExpression::Constant(0)),
            ArgumentExpression::Element(ElementExpression::Constant(0)),
            ArgumentExpression::Element(ElementExpression::Constant(0)),
            ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Constant({
                let mut set = Set::with_capacity(2);
                set.insert(0);
                set.insert(1);
                set
            }))),
        ];
        assert_eq!(
            t.union(3, args.clone()),
            SetExpression::Reduce(SetReduceExpression::Table(
                SetReduceOperator::Union,
                3,
                t.id(),
                args
            ))
        );
    }

    #[test]
    fn set_table_intersection() {
        let mut registry = TableRegistry::default();

        let t = registry.add_table(
            String::from("t"),
            {
                let mut map = FxHashMap::default();
                let mut set = Set::with_capacity(3);
                set.insert(0);
                set.insert(1);
                map.insert(vec![0, 0, 0, 0], set);
                map
            },
            {
                let mut set = Set::with_capacity(3);
                set.insert(1);
                set.insert(2);
                set
            },
        );
        assert!(t.is_ok());
        let t = t.unwrap();
        let args = vec![
            ArgumentExpression::Element(ElementExpression::Constant(0)),
            ArgumentExpression::Element(ElementExpression::Constant(0)),
            ArgumentExpression::Element(ElementExpression::Constant(0)),
            ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Constant({
                let mut set = Set::with_capacity(2);
                set.insert(0);
                set.insert(1);
                set
            }))),
        ];
        assert_eq!(
            t.intersection(3, args.clone()),
            SetExpression::Reduce(SetReduceExpression::Table(
                SetReduceOperator::Intersection,
                3,
                t.id(),
                args
            ))
        );
    }

    #[test]
    fn set_table_symmetric_difference() {
        let mut registry = TableRegistry::default();

        let t = registry.add_table(
            String::from("t"),
            {
                let mut map = FxHashMap::default();
                let mut set = Set::with_capacity(3);
                set.insert(0);
                set.insert(1);
                map.insert(vec![0, 0, 0, 0], set);
                map
            },
            {
                let mut set = Set::with_capacity(3);
                set.insert(1);
                set.insert(2);
                set
            },
        );
        assert!(t.is_ok());
        let t = t.unwrap();
        let args = vec![
            ArgumentExpression::Element(ElementExpression::Constant(0)),
            ArgumentExpression::Element(ElementExpression::Constant(0)),
            ArgumentExpression::Element(ElementExpression::Constant(0)),
            ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Constant({
                let mut set = Set::with_capacity(2);
                set.insert(0);
                set.insert(1);
                set
            }))),
        ];
        assert_eq!(
            t.symmetric_difference(3, args.clone()),
            SetExpression::Reduce(SetReduceExpression::Table(
                SetReduceOperator::SymmetricDifference,
                3,
                t.id(),
                args
            ))
        );
    }

    #[test]
    fn set_if_then_else() {
        let condition = Condition::Constant(true);

        let expression1 = SetExpression::Reference(ReferenceExpression::Variable(0));
        let expression2 = SetExpression::Reference(ReferenceExpression::Variable(1));
        assert_eq!(
            IfThenElse::<SetExpression>::if_then_else(
                condition.clone(),
                expression1.clone(),
                expression2.clone()
            ),
            SetExpression::If(
                Box::new(condition.clone()),
                Box::new(expression1),
                Box::new(expression2),
            )
        );

        let mut metadata = StateMetadata::default();
        let ob = metadata.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let expression1 = SetExpression::Reference(ReferenceExpression::Variable(0));
        let s = metadata.create_set(ob, &[0, 1, 2]);
        assert!(s.is_ok());
        let s = s.unwrap();
        assert_eq!(
            IfThenElse::<SetExpression>::if_then_else(
                condition.clone(),
                expression1.clone(),
                s.clone()
            ),
            SetExpression::If(
                Box::new(condition.clone()),
                Box::new(expression1.clone()),
                Box::new(SetExpression::Reference(ReferenceExpression::Constant(
                    s.clone()
                ))),
            )
        );
        assert_eq!(
            IfThenElse::<SetExpression>::if_then_else(
                condition.clone(),
                s.clone(),
                expression1.clone(),
            ),
            SetExpression::If(
                Box::new(condition.clone()),
                Box::new(SetExpression::Reference(ReferenceExpression::Constant(
                    s.clone()
                ))),
                Box::new(expression1),
            )
        );

        let expression1 = SetExpression::Reference(ReferenceExpression::Variable(0));
        let v = metadata.add_set_variable(String::from("sv"), ob);
        assert!(v.is_ok());
        let v = v.unwrap();
        assert_eq!(
            IfThenElse::<SetExpression>::if_then_else(condition.clone(), expression1.clone(), v,),
            SetExpression::If(
                Box::new(condition.clone()),
                Box::new(expression1.clone()),
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                    v.id()
                ))),
            )
        );
        assert_eq!(
            IfThenElse::<SetExpression>::if_then_else(condition.clone(), v, expression1.clone()),
            SetExpression::If(
                Box::new(condition.clone()),
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                    v.id()
                ))),
                Box::new(expression1),
            )
        );

        assert_eq!(
            IfThenElse::<SetExpression>::if_then_else(condition.clone(), s.clone(), v),
            SetExpression::If(
                Box::new(condition.clone()),
                Box::new(SetExpression::Reference(ReferenceExpression::Constant(
                    s.clone()
                ))),
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                    v.id()
                ))),
            )
        );
        assert_eq!(
            IfThenElse::<SetExpression>::if_then_else(condition.clone(), v, s.clone()),
            SetExpression::If(
                Box::new(condition),
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                    v.id()
                ))),
                Box::new(SetExpression::Reference(ReferenceExpression::Constant(s))),
            )
        );
    }

    #[test]
    fn set_if_eval() {
        let state = generate_state();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = generate_registry();
        let mut s1 = Set::with_capacity(3);
        s1.insert(1);
        let mut s0 = Set::with_capacity(3);
        s0.insert(0);
        let expression = SetExpression::If(
            Box::new(Condition::Constant(true)),
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(
                s1.clone(),
            ))),
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(
                s0.clone(),
            ))),
        );
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            s1
        );
        let expression = SetExpression::If(
            Box::new(Condition::Constant(false)),
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(s1))),
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(
                s0.clone(),
            ))),
        );
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            s0
        );
    }

    #[test]
    fn set_reference_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(2);
        let expression = SetExpression::Reference(ReferenceExpression::Constant(set.clone()));
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            set
        );
        let expression = SetExpression::Reference(ReferenceExpression::Variable(0));
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            state.signature_variables.set_variables[0]
        );
        let expression = SetExpression::Reference(ReferenceExpression::Table(
            TableExpression::Table1D(0, ElementExpression::Constant(0)),
        ));
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            set
        );
    }

    #[test]
    fn set_state_function_eval() {
        let mut state_metadata = StateMetadata::default();
        let ob = state_metadata.add_object_type("ob", 3);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = state_metadata.add_set_variable("v", ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_set_function("f", v.add(1));
        assert!(f.is_ok());
        let f = f.unwrap();
        let g = state_functions.add_set_function("g", v.add(2));
        assert!(g.is_ok());
        let g = g.unwrap();

        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![Set::with_capacity(3)],
                ..Default::default()
            },
            ..Default::default()
        };

        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        let mut expected1 = Set::with_capacity(3);
        expected1.insert(1);
        assert_eq!(
            f.eval(&state, &mut function_cache, &state_functions, &registry),
            expected1,
        );

        let mut expected2 = Set::with_capacity(3);
        expected2.insert(2);
        assert_eq!(
            g.eval(&state, &mut function_cache, &state_functions, &registry),
            expected2,
        );

        assert_eq!(
            f.eval(&state, &mut function_cache, &state_functions, &registry),
            expected1,
        );
        assert_eq!(
            g.eval(&state, &mut function_cache, &state_functions, &registry),
            expected2,
        );
    }

    #[test]
    fn set_complement_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let expression = SetExpression::Complement(Box::new(SetExpression::Reference(
            ReferenceExpression::Variable(0),
        )));
        let mut set = Set::with_capacity(3);
        set.insert(1);
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            set
        );
    }

    #[test]
    fn set_union_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let expression = SetExpression::SetOperation(
            SetOperator::Union,
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
        );
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(1);
        set.insert(2);
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            set
        );
        let expression = SetExpression::SetOperation(
            SetOperator::Union,
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
        );
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            state.signature_variables.set_variables[0]
        );
    }

    #[test]
    fn set_union_state_variable_x_eval() {
        let mut state_metadata = StateMetadata::default();
        let ob = state_metadata.add_object_type("ob", 3);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = state_metadata.add_set_variable("v", ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_set_function("f", v.add(1));
        assert!(f.is_ok());
        let f = f.unwrap();

        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![Set::with_capacity(3)],
                ..Default::default()
            },
            ..Default::default()
        };

        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        let expression = f.clone() | !f;

        let mut expected = Set::with_capacity(3);
        expected.insert(0);
        expected.insert(1);
        expected.insert(2);
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            expected,
        );
    }

    #[test]
    fn set_union_state_variable_y_eval() {
        let mut state_metadata = StateMetadata::default();
        let ob = state_metadata.add_object_type("ob", 3);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = state_metadata.add_set_variable("v", ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_set_function("f", v.add(1));
        assert!(f.is_ok());
        let f = f.unwrap();

        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![Set::with_capacity(3)],
                ..Default::default()
            },
            ..Default::default()
        };

        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        let expression = !f.clone() | f;

        let mut expected = Set::with_capacity(3);
        expected.insert(0);
        expected.insert(1);
        expected.insert(2);
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            expected,
        );
    }

    #[test]
    fn set_difference_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let expression = SetExpression::SetOperation(
            SetOperator::Difference,
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
        );
        let mut set = Set::with_capacity(3);
        set.insert(2);
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            set
        );
        let expression = SetExpression::SetOperation(
            SetOperator::Difference,
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
        );
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            Set::with_capacity(3)
        );
    }

    #[test]
    fn set_difference_state_variable_y_eval() {
        let mut state_metadata = StateMetadata::default();
        let ob = state_metadata.add_object_type("ob", 3);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = state_metadata.add_set_variable("v", ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_set_function("f", v.add(1));
        assert!(f.is_ok());
        let f = f.unwrap();

        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![Set::with_capacity(3)],
                ..Default::default()
            },
            ..Default::default()
        };

        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        let expression = f.add(0).add(1) - v.add(0);

        let mut expected = Set::with_capacity(3);
        expected.insert(1);
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            expected,
        );
    }

    #[test]
    fn set_intersect_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let expression = SetExpression::SetOperation(
            SetOperator::Intersection,
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
        );
        let mut set = Set::with_capacity(3);
        set.insert(0);
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            set
        );
        let expression = SetExpression::SetOperation(
            SetOperator::Intersection,
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
        );
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            state.signature_variables.set_variables[0]
        );
    }

    #[test]
    fn set_intersect_state_variable_x_eval() {
        let mut state_metadata = StateMetadata::default();
        let ob = state_metadata.add_object_type("ob", 3);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = state_metadata.add_set_variable("v", ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_set_function("f", v.add(1).add(2));
        assert!(f.is_ok());
        let f = f.unwrap();

        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![Set::with_capacity(3)],
                ..Default::default()
            },
            ..Default::default()
        };

        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        let expression = f.clone() & v.add(0).add(1);

        let mut expected = Set::with_capacity(3);
        expected.insert(1);
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            expected,
        );
    }

    #[test]
    fn set_intersect_state_variable_y_eval() {
        let mut state_metadata = StateMetadata::default();
        let ob = state_metadata.add_object_type("ob", 3);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = state_metadata.add_set_variable("v", ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_set_function("f", v.add(1).add(2));
        assert!(f.is_ok());
        let f = f.unwrap();

        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![Set::with_capacity(3)],
                ..Default::default()
            },
            ..Default::default()
        };

        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        let expression = f & v.add(0).add(1);

        let mut expected = Set::with_capacity(3);
        expected.insert(1);
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            expected
        )
    }

    #[test]
    fn set_add_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let expression = SetExpression::SetElementOperation(
            SetElementOperator::Add,
            ElementExpression::Constant(1),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
        );
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(1);
        set.insert(2);
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            set
        );
        let expression = SetExpression::SetElementOperation(
            SetElementOperator::Add,
            ElementExpression::Constant(0),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
        );
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            state.signature_variables.set_variables[0]
        );
    }

    #[test]
    fn set_remove_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let expression = SetExpression::SetElementOperation(
            SetElementOperator::Remove,
            ElementExpression::Constant(2),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
        );
        let mut set = Set::with_capacity(3);
        set.insert(0);
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            set
        );
        let expression = SetExpression::SetElementOperation(
            SetElementOperator::Remove,
            ElementExpression::Constant(1),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
        );
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            state.signature_variables.set_variables[0]
        );
    }

    #[test]
    fn set_reduce_eval() {
        let state = State::default();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry {
            set_tables: TableData {
                tables: vec![Table::new(
                    {
                        let mut map = FxHashMap::default();
                        map.insert(vec![0, 0, 0, 0], {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(1);
                            set
                        });
                        map.insert(vec![0, 0, 0, 1], {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(2);
                            set
                        });
                        map.insert(vec![0, 0, 1, 0], {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(3);
                            set
                        });
                        map
                    },
                    {
                        let mut set = Set::with_capacity(5);
                        set.insert(0);
                        set.insert(4);
                        set
                    },
                )],
                ..Default::default()
            },
            ..Default::default()
        };
        let expression = SetExpression::Reduce(SetReduceExpression::Table(
            SetReduceOperator::Union,
            5,
            0,
            vec![
                ArgumentExpression::Element(ElementExpression::Constant(0)),
                ArgumentExpression::Element(ElementExpression::Constant(0)),
                ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Constant({
                    let mut set = Set::with_capacity(2);
                    set.insert(0);
                    set.insert(1);
                    set
                }))),
                ArgumentExpression::Vector(VectorExpression::Reference(
                    ReferenceExpression::Constant(vec![0, 1]),
                )),
            ],
        ));
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            {
                let mut set = Set::with_capacity(5);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set.insert(3);
                set.insert(4);
                set
            }
        );
    }

    #[test]
    fn set_from_vector_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let expression = SetExpression::FromVector(
            3,
            Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![0, 1],
            ))),
        );
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(1);
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            set
        );
        let expression = SetExpression::FromVector(
            3,
            Box::new(VectorExpression::Reference(ReferenceExpression::Variable(
                0,
            ))),
        );
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(2);
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            set
        );
    }

    #[test]
    fn set_reference_simplify() {
        let registry = generate_registry();
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(2);
        let expression = SetExpression::Reference(ReferenceExpression::Constant(set.clone()));
        assert_eq!(expression.simplify(&registry), expression);
        let expression = SetExpression::Reference(ReferenceExpression::Variable(0));
        assert_eq!(expression.simplify(&registry), expression);
        let expression = SetExpression::Reference(ReferenceExpression::Table(
            TableExpression::Table1D(0, ElementExpression::Constant(0)),
        ));
        assert_eq!(
            expression.simplify(&registry),
            SetExpression::Reference(ReferenceExpression::Constant(set))
        );
    }

    #[test]
    fn set_complement_simplify() {
        let registry = generate_registry();
        let expression = SetExpression::Complement(Box::new(SetExpression::Reference(
            ReferenceExpression::Variable(0),
        )));
        assert_eq!(expression.simplify(&registry), expression);
        let mut set = Set::with_capacity(3);
        set.insert(1);
        let expression = SetExpression::Complement(Box::new(SetExpression::Reference(
            ReferenceExpression::Constant(set),
        )));
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(2);
        assert_eq!(
            expression.simplify(&registry),
            SetExpression::Reference(ReferenceExpression::Constant(set))
        );
        let expression = SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        ))));
        assert_eq!(
            expression.simplify(&registry),
            SetExpression::Reference(ReferenceExpression::Variable(0))
        );
    }

    #[test]
    fn set_union_simplify() {
        let registry = generate_registry();
        let expression = SetExpression::SetOperation(
            SetOperator::Union,
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
        );
        assert_eq!(expression.simplify(&registry), expression);
        let mut x = Set::with_capacity(3);
        x.insert(0);
        x.insert(2);
        let mut y = Set::with_capacity(3);
        y.insert(0);
        y.insert(1);
        let expression = SetExpression::SetOperation(
            SetOperator::Union,
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(x))),
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(y))),
        );
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(1);
        set.insert(2);
        assert_eq!(
            expression.simplify(&registry),
            SetExpression::Reference(ReferenceExpression::Constant(set))
        );
        let expression = SetExpression::SetOperation(
            SetOperator::Union,
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
        );
        assert_eq!(
            expression.simplify(&registry),
            SetExpression::Reference(ReferenceExpression::Variable(0))
        );
    }

    #[test]
    fn set_difference_simplify() {
        let registry = generate_registry();
        let expression = SetExpression::SetOperation(
            SetOperator::Difference,
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
        );
        assert_eq!(expression.simplify(&registry), expression);
        let mut x = Set::with_capacity(3);
        x.insert(0);
        x.insert(2);
        let mut y = Set::with_capacity(3);
        y.insert(0);
        y.insert(1);
        let expression = SetExpression::SetOperation(
            SetOperator::Difference,
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(x))),
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(y))),
        );
        let mut set = Set::with_capacity(3);
        set.insert(2);
        assert_eq!(
            expression.simplify(&registry),
            SetExpression::Reference(ReferenceExpression::Constant(set))
        );
    }

    #[test]
    fn set_intersect_simplify() {
        let registry = generate_registry();
        let expression = SetExpression::SetOperation(
            SetOperator::Intersection,
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
        );
        assert_eq!(expression.simplify(&registry), expression);
        let mut x = Set::with_capacity(3);
        x.insert(0);
        x.insert(2);
        let mut y = Set::with_capacity(3);
        y.insert(0);
        y.insert(1);
        let expression = SetExpression::SetOperation(
            SetOperator::Intersection,
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(x))),
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(y))),
        );
        let mut set = Set::with_capacity(3);
        set.insert(0);
        assert_eq!(
            expression.simplify(&registry),
            SetExpression::Reference(ReferenceExpression::Constant(set))
        );
        let expression = SetExpression::SetOperation(
            SetOperator::Union,
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
        );
        assert_eq!(
            expression.simplify(&registry),
            SetExpression::Reference(ReferenceExpression::Variable(0))
        );
    }

    #[test]
    fn set_add_simplify() {
        let registry = generate_registry();
        let expression = SetExpression::SetElementOperation(
            SetElementOperator::Add,
            ElementExpression::Constant(1),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
        );
        assert_eq!(expression.simplify(&registry), expression);
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(2);
        let expression = SetExpression::SetElementOperation(
            SetElementOperator::Add,
            ElementExpression::Constant(1),
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(set))),
        );
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(1);
        set.insert(2);
        assert_eq!(
            expression.simplify(&registry),
            SetExpression::Reference(ReferenceExpression::Constant(set))
        );
    }

    #[test]
    fn set_remove_simplify() {
        let registry = generate_registry();
        let expression = SetExpression::SetElementOperation(
            SetElementOperator::Remove,
            ElementExpression::Constant(2),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
        );
        assert_eq!(expression.simplify(&registry), expression);
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(2);
        let expression = SetExpression::SetElementOperation(
            SetElementOperator::Remove,
            ElementExpression::Constant(2),
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(set))),
        );
        let mut set = Set::with_capacity(3);
        set.insert(0);
        assert_eq!(
            expression.simplify(&registry),
            SetExpression::Reference(ReferenceExpression::Constant(set))
        );
    }

    #[test]
    fn set_reduce_constant_simplify() {
        let registry = TableRegistry {
            set_tables: TableData {
                tables: vec![Table::new(
                    {
                        let mut map = FxHashMap::default();
                        map.insert(vec![0, 0, 0, 0], {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(1);
                            set
                        });
                        map.insert(vec![0, 0, 0, 1], {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(2);
                            set
                        });
                        map.insert(vec![0, 0, 1, 0], {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(3);
                            set
                        });
                        map
                    },
                    {
                        let mut set = Set::with_capacity(5);
                        set.insert(0);
                        set.insert(4);
                        set
                    },
                )],
                ..Default::default()
            },
            ..Default::default()
        };
        let expression = SetExpression::Reduce(SetReduceExpression::Table(
            SetReduceOperator::Union,
            5,
            0,
            vec![
                ArgumentExpression::Element(ElementExpression::Constant(0)),
                ArgumentExpression::Element(ElementExpression::Constant(0)),
                ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Constant({
                    let mut set = Set::with_capacity(2);
                    set.insert(0);
                    set.insert(1);
                    set
                }))),
                ArgumentExpression::Vector(VectorExpression::Reference(
                    ReferenceExpression::Constant(vec![0, 1]),
                )),
            ],
        ));
        assert_eq!(
            expression.simplify(&registry),
            SetExpression::Reference(ReferenceExpression::Constant({
                let mut set = Set::with_capacity(5);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set.insert(3);
                set.insert(4);
                set
            }))
        );
    }

    #[test]
    fn set_reduce_simplify() {
        let registry = TableRegistry {
            set_tables: TableData {
                tables: vec![Table::new(
                    {
                        let mut map = FxHashMap::default();
                        map.insert(vec![0, 0, 0, 0], {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(1);
                            set
                        });
                        map.insert(vec![0, 0, 0, 1], {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(2);
                            set
                        });
                        map.insert(vec![0, 0, 1, 0], {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(3);
                            set
                        });
                        map
                    },
                    {
                        let mut set = Set::with_capacity(5);
                        set.insert(0);
                        set.insert(4);
                        set
                    },
                )],
                ..Default::default()
            },
            ..Default::default()
        };
        let expression = SetExpression::Reduce(SetReduceExpression::Table(
            SetReduceOperator::Union,
            5,
            0,
            vec![
                ArgumentExpression::Element(ElementExpression::If(
                    Box::new(Condition::Constant(true)),
                    Box::new(ElementExpression::Variable(0)),
                    Box::new(ElementExpression::Constant(0)),
                )),
                ArgumentExpression::Element(ElementExpression::If(
                    Box::new(Condition::Constant(true)),
                    Box::new(ElementExpression::Variable(0)),
                    Box::new(ElementExpression::Constant(0)),
                )),
                ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Constant({
                    let mut set = Set::with_capacity(2);
                    set.insert(0);
                    set.insert(1);
                    set
                }))),
                ArgumentExpression::Vector(VectorExpression::Reference(
                    ReferenceExpression::Constant(vec![0, 1]),
                )),
            ],
        ));
        assert_eq!(
            expression.simplify(&registry),
            SetExpression::Reduce(SetReduceExpression::Table(
                SetReduceOperator::Union,
                5,
                0,
                vec![
                    ArgumentExpression::Element(ElementExpression::Variable(0),),
                    ArgumentExpression::Element(ElementExpression::Variable(0),),
                    ArgumentExpression::Set(SetExpression::Reference(
                        ReferenceExpression::Constant({
                            let mut set = Set::with_capacity(2);
                            set.insert(0);
                            set.insert(1);
                            set
                        })
                    )),
                    ArgumentExpression::Vector(VectorExpression::Reference(
                        ReferenceExpression::Constant(vec![0, 1]),
                    )),
                ],
            ))
        );
    }

    #[test]
    fn set_from_vector_simplify() {
        let registry = generate_registry();
        let expression = SetExpression::FromVector(
            3,
            Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![0, 1],
            ))),
        );
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(1);
        assert_eq!(
            expression.simplify(&registry),
            SetExpression::Reference(ReferenceExpression::Constant(set))
        );
        let expression = SetExpression::FromVector(
            3,
            Box::new(VectorExpression::Reference(ReferenceExpression::Variable(
                0,
            ))),
        );
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(2);
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn set_if_simplify() {
        let registry = generate_registry();
        let mut s1 = Set::with_capacity(3);
        s1.insert(1);
        let mut s0 = Set::with_capacity(3);
        s0.insert(0);
        let expression = SetExpression::If(
            Box::new(Condition::Constant(true)),
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(
                s1.clone(),
            ))),
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(
                s0.clone(),
            ))),
        );
        assert_eq!(
            expression.simplify(&registry),
            SetExpression::Reference(ReferenceExpression::Constant(s1.clone()))
        );
        let expression = SetExpression::If(
            Box::new(Condition::Constant(false)),
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(
                s1.clone(),
            ))),
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(
                s0.clone(),
            ))),
        );
        assert_eq!(
            expression.simplify(&registry),
            SetExpression::Reference(ReferenceExpression::Constant(s0.clone()))
        );
        let expression = SetExpression::If(
            Box::new(Condition::ComparisonI(
                ComparisonOperator::Gt,
                Box::new(IntegerExpression::Variable(0)),
                Box::new(IntegerExpression::Constant(1)),
            )),
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(s1))),
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(s0))),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }
}
