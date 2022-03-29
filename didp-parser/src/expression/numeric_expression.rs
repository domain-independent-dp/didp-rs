use super::element_expression::{ElementExpression, SetExpression, VectorExpression};
use super::numeric_table_expression::{ArgumentExpression, NumericTableExpression};
use super::reference_expression::ReferenceExpression;
use super::util;
use crate::state::State;
use crate::table_data::TableData;
use crate::table_registry::TableRegistry;
use crate::variable::{Continuous, FromNumeric, Integer, Numeric};
use std::boxed::Box;

#[derive(Debug, PartialEq, Clone)]
pub enum NumericExpression<T: Numeric> {
    Constant(T),
    IntegerVariable(usize),
    ContinuousVariable(usize),
    IntegerResourceVariable(usize),
    ContinuousResourceVariable(usize),
    Cost,
    NumericOperation(
        NumericOperator,
        Box<NumericExpression<T>>,
        Box<NumericExpression<T>>,
    ),
    Cardinality(SetExpression),
    Length(VectorExpression),
    IntegerTable(NumericTableExpression<Integer>),
    ContinuousTable(NumericTableExpression<Continuous>),
    IntegerLast(Box<NumericVectorExpression<Integer>>),
    ContinuousLast(Box<NumericVectorExpression<Continuous>>),
    IntegerAt(Box<NumericVectorExpression<Integer>>, ElementExpression),
    ContinuousAt(Box<NumericVectorExpression<Continuous>>, ElementExpression),
    IntegerReduceSum(Box<NumericVectorExpression<Integer>>),
    ContinuousReduceSum(Box<NumericVectorExpression<Continuous>>),
    IntegerReduceProduct(Box<NumericVectorExpression<Integer>>),
    ContinuousReduceProduct(Box<NumericVectorExpression<Continuous>>),
}

#[derive(Debug, PartialEq, Clone)]
pub enum NumericOperator {
    Add,
    Subtract,
    Multiply,
    Divide,
    Max,
    Min,
}

impl<T: Numeric> Default for NumericExpression<T> {
    fn default() -> NumericExpression<T> {
        NumericExpression::Constant(T::zero())
    }
}

impl<T: Numeric> NumericExpression<T> {
    pub fn eval(&self, state: &State, registry: &TableRegistry) -> T {
        self.eval_inner(None, state, registry)
    }

    pub fn eval_cost(&self, cost: T, state: &State, registry: &TableRegistry) -> T {
        self.eval_inner(Some(cost), state, registry)
    }

    fn eval_inner(&self, cost: Option<T>, state: &State, registry: &TableRegistry) -> T {
        match self {
            Self::Constant(x) => *x,
            Self::IntegerVariable(i) => {
                T::from_integer(state.signature_variables.integer_variables[*i])
            }
            Self::IntegerResourceVariable(i) => {
                T::from_integer(state.resource_variables.integer_variables[*i])
            }
            Self::ContinuousVariable(i) => {
                T::from_continuous(state.signature_variables.continuous_variables[*i].into_inner())
            }
            Self::ContinuousResourceVariable(i) => {
                T::from_continuous(state.resource_variables.continuous_variables[*i])
            }
            Self::Cost => cost.unwrap(),
            Self::NumericOperation(op, a, b) => {
                let a = a.eval_inner(cost, state, registry);
                let b = b.eval_inner(cost, state, registry);
                Self::eval_operation(op, a, b)
            }
            Self::Cardinality(SetExpression::Reference(expression)) => {
                let set = expression.eval(
                    state,
                    registry,
                    &state.signature_variables.set_variables,
                    &registry.set_tables,
                );
                FromNumeric::from_usize(set.count_ones(..))
            }
            Self::Cardinality(set) => {
                FromNumeric::from_usize(set.eval(state, registry).count_ones(..))
            }
            Self::Length(VectorExpression::Reference(expression)) => {
                let vector = expression.eval(
                    state,
                    registry,
                    &state.signature_variables.vector_variables,
                    &registry.vector_tables,
                );
                FromNumeric::from_usize(vector.len())
            }
            Self::Length(vector) => FromNumeric::from_usize(vector.eval(state, registry).len()),
            Self::IntegerTable(t) => {
                T::from_integer(t.eval(state, registry, &registry.integer_tables))
            }
            Self::ContinuousTable(t) => {
                T::from_continuous(t.eval(state, registry, &registry.continuous_tables))
            }
            Self::IntegerLast(vector) => T::from_integer(match vector.as_ref() {
                NumericVectorExpression::Constant(vector) => *vector.last().unwrap(),
                vector => *vector
                    .eval_inner(
                        cost.map(|x| x.to_integer()),
                        state,
                        registry,
                        &registry.integer_tables,
                    )
                    .last()
                    .unwrap(),
            }),
            Self::ContinuousLast(vector) => T::from_continuous(match vector.as_ref() {
                NumericVectorExpression::Constant(vector) => *vector.last().unwrap(),
                vector => *vector
                    .eval_inner(
                        cost.map(|x| x.to_continuous()),
                        state,
                        registry,
                        &registry.continuous_tables,
                    )
                    .last()
                    .unwrap(),
            }),
            Self::IntegerAt(vector, i) => T::from_integer(match vector.as_ref() {
                NumericVectorExpression::Constant(vector) => vector[i.eval(state, registry)],
                vector => vector.eval_inner(
                    cost.map(|x| x.to_integer()),
                    state,
                    registry,
                    &registry.integer_tables,
                )[i.eval(state, registry)],
            }),
            Self::ContinuousAt(vector, i) => T::from_continuous(match vector.as_ref() {
                NumericVectorExpression::Constant(vector) => vector[i.eval(state, registry)],
                vector => vector.eval_inner(
                    cost.map(|x| x.to_continuous()),
                    state,
                    registry,
                    &registry.continuous_tables,
                )[i.eval(state, registry)],
            }),
            Self::IntegerReduceSum(vector) => T::from_integer(match vector.as_ref() {
                NumericVectorExpression::Constant(vector) => vector.iter().copied().sum(),
                vector => vector
                    .eval_inner(
                        cost.map(|x| x.to_integer()),
                        state,
                        registry,
                        &registry.integer_tables,
                    )
                    .into_iter()
                    .sum(),
            }),
            Self::ContinuousReduceSum(vector) => T::from_continuous(match vector.as_ref() {
                NumericVectorExpression::Constant(vector) => vector.iter().copied().sum(),
                vector => vector
                    .eval_inner(
                        cost.map(|x| x.to_continuous()),
                        state,
                        registry,
                        &registry.continuous_tables,
                    )
                    .into_iter()
                    .sum(),
            }),
            Self::IntegerReduceProduct(vector) => T::from_integer(match vector.as_ref() {
                NumericVectorExpression::Constant(vector) => vector.iter().copied().product(),
                vector => vector
                    .eval_inner(
                        cost.map(|x| x.to_integer()),
                        state,
                        registry,
                        &registry.integer_tables,
                    )
                    .into_iter()
                    .product(),
            }),
            Self::ContinuousReduceProduct(vector) => T::from_continuous(match vector.as_ref() {
                NumericVectorExpression::Constant(vector) => vector.iter().copied().product(),
                vector => vector
                    .eval_inner(
                        cost.map(|x| x.to_continuous()),
                        state,
                        registry,
                        &registry.continuous_tables,
                    )
                    .into_iter()
                    .product(),
            }),
        }
    }

    pub fn simplify(&self, registry: &TableRegistry) -> NumericExpression<T> {
        match self {
            Self::NumericOperation(op, a, b) => {
                match (a.simplify(registry), b.simplify(registry)) {
                    (Self::Constant(a), Self::Constant(b)) => {
                        Self::Constant(Self::eval_operation(op, a, b))
                    }
                    (a, b) => Self::NumericOperation(op.clone(), Box::new(a), Box::new(b)),
                }
            }
            Self::Cardinality(expression) => match expression.simplify(registry) {
                SetExpression::Reference(ReferenceExpression::Constant(set)) => {
                    Self::Constant(FromNumeric::from_usize(set.count_ones(..)))
                }
                expression => Self::Cardinality(expression),
            },
            Self::Length(expression) => match expression.simplify(registry) {
                VectorExpression::Reference(ReferenceExpression::Constant(vector)) => {
                    Self::Constant(FromNumeric::from_usize(vector.len()))
                }
                expression => Self::Length(expression),
            },
            Self::IntegerTable(expression) => {
                match expression.simplify(registry, &registry.integer_tables) {
                    NumericTableExpression::Constant(value) => {
                        Self::Constant(T::from_integer(value))
                    }
                    expression => Self::IntegerTable(expression),
                }
            }
            Self::ContinuousTable(expression) => {
                match expression.simplify(registry, &registry.continuous_tables) {
                    NumericTableExpression::Constant(value) => {
                        Self::Constant(T::from_continuous(value))
                    }
                    expression => Self::ContinuousTable(expression),
                }
            }
            Self::IntegerLast(vector) => {
                match vector.simplify(registry, &registry.integer_tables) {
                    NumericVectorExpression::Constant(vector) => {
                        Self::Constant(T::from_integer(*vector.last().unwrap()))
                    }
                    vector => Self::IntegerLast(Box::new(vector)),
                }
            }
            Self::ContinuousLast(vector) => {
                match vector.simplify(registry, &registry.continuous_tables) {
                    NumericVectorExpression::Constant(vector) => {
                        Self::Constant(T::from_continuous(*vector.last().unwrap()))
                    }
                    vector => Self::ContinuousLast(Box::new(vector)),
                }
            }
            Self::IntegerAt(vector, i) => {
                match (
                    vector.simplify(registry, &registry.integer_tables),
                    i.simplify(registry),
                ) {
                    (NumericVectorExpression::Constant(vector), ElementExpression::Constant(i)) => {
                        Self::Constant(T::from_integer(vector[i]))
                    }
                    (vector, i) => Self::IntegerAt(Box::new(vector), i),
                }
            }
            Self::ContinuousAt(vector, i) => {
                match (
                    vector.simplify(registry, &registry.continuous_tables),
                    i.simplify(registry),
                ) {
                    (NumericVectorExpression::Constant(vector), ElementExpression::Constant(i)) => {
                        Self::Constant(T::from_continuous(vector[i]))
                    }
                    (vector, i) => Self::ContinuousAt(Box::new(vector), i),
                }
            }
            Self::IntegerReduceSum(vector) => {
                match vector.simplify(registry, &registry.integer_tables) {
                    NumericVectorExpression::Constant(vector) => {
                        Self::Constant(T::from_integer(vector.into_iter().sum()))
                    }
                    vector => Self::IntegerReduceSum(Box::new(vector)),
                }
            }
            Self::ContinuousReduceSum(vector) => {
                match vector.simplify(registry, &registry.continuous_tables) {
                    NumericVectorExpression::Constant(vector) => {
                        Self::Constant(T::from_continuous(vector.into_iter().sum()))
                    }
                    vector => Self::ContinuousReduceSum(Box::new(vector)),
                }
            }
            Self::IntegerReduceProduct(vector) => {
                match vector.simplify(registry, &registry.integer_tables) {
                    NumericVectorExpression::Constant(vector) => {
                        Self::Constant(T::from_integer(vector.into_iter().product()))
                    }
                    vector => Self::IntegerReduceProduct(Box::new(vector)),
                }
            }
            Self::ContinuousReduceProduct(vector) => {
                match vector.simplify(registry, &registry.continuous_tables) {
                    NumericVectorExpression::Constant(vector) => {
                        Self::Constant(T::from_continuous(vector.into_iter().product()))
                    }
                    vector => Self::ContinuousReduceProduct(Box::new(vector)),
                }
            }
            _ => self.clone(),
        }
    }

    fn eval_operation(op: &NumericOperator, a: T, b: T) -> T {
        match op {
            NumericOperator::Add => a + b,
            NumericOperator::Subtract => a - b,
            NumericOperator::Multiply => a * b,
            NumericOperator::Divide => a / b,
            NumericOperator::Max => {
                if a > b {
                    a
                } else {
                    b
                }
            }
            NumericOperator::Min => {
                if a < b {
                    a
                } else {
                    b
                }
            }
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum NumericVectorExpression<T: Numeric> {
    Constant(Vec<T>),
    Reverse(Box<NumericVectorExpression<T>>),
    Push(NumericExpression<T>, Box<NumericVectorExpression<T>>),
    Pop(Box<NumericVectorExpression<T>>),
    Set(
        NumericExpression<T>,
        Box<NumericVectorExpression<T>>,
        ElementExpression,
    ),
    NumericOperation(
        NumericOperator,
        NumericExpression<T>,
        Box<NumericVectorExpression<T>>,
    ),
    VectorOperation(
        NumericOperator,
        Box<NumericVectorExpression<T>>,
        Box<NumericVectorExpression<T>>,
    ),
    Table(usize, Vec<VectorOrElementExpression>),
    TableSum(usize, Vec<ArgumentExpression>),
    Table1D(usize, VectorExpression),
    Table2D(usize, VectorExpression, VectorExpression),
    Table2DX(usize, VectorExpression, ElementExpression),
    Table2DY(usize, ElementExpression, VectorExpression),
    Table2DXSum(usize, VectorExpression, SetExpression),
    Table2DYSum(usize, SetExpression, VectorExpression),
}

impl<T: Numeric> NumericVectorExpression<T> {
    pub fn eval(&self, state: &State, registry: &TableRegistry, tables: &TableData<T>) -> Vec<T> {
        self.eval_inner(None, state, registry, tables)
    }

    pub fn eval_cost(
        &self,
        cost: T,
        state: &State,
        registry: &TableRegistry,
        tables: &TableData<T>,
    ) -> Vec<T> {
        self.eval_inner(Some(cost), state, registry, tables)
    }

    fn eval_inner(
        &self,
        cost: Option<T>,
        state: &State,
        registry: &TableRegistry,
        tables: &TableData<T>,
    ) -> Vec<T> {
        let vector_variables = &state.signature_variables.vector_variables;
        let vector_tables = &registry.vector_tables;
        let set_variables = &state.signature_variables.set_variables;
        let set_tables = &registry.set_tables;
        match self {
            Self::Constant(vector) => vector.clone(),
            Self::Reverse(vector) => {
                let mut vector = vector.eval_inner(cost, state, registry, tables);
                vector.reverse();
                vector
            }
            Self::Push(value, vector) => {
                let mut vector = vector.eval_inner(cost, state, registry, tables);
                vector.push(value.eval(state, registry));
                vector
            }
            Self::Pop(vector) => {
                let mut vector = vector.eval_inner(cost, state, registry, tables);
                vector.pop();
                vector
            }
            Self::Set(value, vector, i) => {
                let mut vector = vector.eval_inner(cost, state, registry, tables);
                vector[i.eval(state, registry)] = value.eval(state, registry);
                vector
            }
            Self::NumericOperation(op, x, y) => Self::eval_operation(
                op,
                x.eval_inner(cost, state, registry),
                y.eval_inner(cost, state, registry, tables),
            ),
            Self::VectorOperation(op, x, y) => match (x.as_ref(), y.as_ref()) {
                (Self::Constant(x), y) => Self::eval_vector_operation_in_y(
                    op,
                    x,
                    y.eval_inner(cost, state, registry, tables),
                ),
                (x, Self::Constant(y)) => Self::eval_vector_operation_in_x(
                    op,
                    x.eval_inner(cost, state, registry, tables),
                    y,
                ),
                (x, y) => Self::eval_vector_operation_in_y(
                    op,
                    &x.eval_inner(cost, state, registry, tables),
                    y.eval_inner(cost, state, registry, tables),
                ),
            },
            Self::Table(i, args) => Self::eval_table(*i, args, state, registry, tables),
            Self::TableSum(i, args) => Self::eval_table_sum(*i, args, state, registry, tables),
            Self::Table1D(i, VectorExpression::Reference(x)) => x
                .eval(state, registry, vector_variables, vector_tables)
                .iter()
                .map(|x| tables.tables_1d[*i].eval(*x))
                .collect(),
            Self::Table1D(i, x) => x
                .eval(state, registry)
                .into_iter()
                .map(|x| tables.tables_1d[*i].eval(x))
                .collect(),
            Self::Table2D(i, VectorExpression::Reference(x), VectorExpression::Reference(y)) => x
                .eval(state, registry, vector_variables, vector_tables)
                .iter()
                .zip(y.eval(state, registry, vector_variables, vector_tables))
                .map(|(x, y)| tables.tables_2d[*i].eval(*x, *y))
                .collect(),
            Self::Table2D(i, VectorExpression::Reference(x), y) => x
                .eval(state, registry, vector_variables, vector_tables)
                .iter()
                .zip(y.eval(state, registry))
                .map(|(x, y)| tables.tables_2d[*i].eval(*x, y))
                .collect(),
            Self::Table2D(i, x, VectorExpression::Reference(y)) => x
                .eval(state, registry)
                .into_iter()
                .zip(y.eval(state, registry, vector_variables, vector_tables))
                .map(|(x, y)| tables.tables_2d[*i].eval(x, *y))
                .collect(),
            Self::Table2D(i, x, y) => x
                .eval(state, registry)
                .into_iter()
                .zip(y.eval(state, registry))
                .map(|(x, y)| tables.tables_2d[*i].eval(x, y))
                .collect(),
            Self::Table2DX(i, VectorExpression::Reference(x), y) => {
                let y = y.eval(state, registry);
                x.eval(state, registry, vector_variables, vector_tables)
                    .iter()
                    .map(|x| tables.tables_2d[*i].eval(*x, y))
                    .collect()
            }
            Self::Table2DX(i, x, y) => {
                let y = y.eval(state, registry);
                x.eval(state, registry)
                    .into_iter()
                    .map(|x| tables.tables_2d[*i].eval(x, y))
                    .collect()
            }
            Self::Table2DY(i, x, VectorExpression::Reference(y)) => {
                let x = x.eval(state, registry);
                y.eval(state, registry, vector_variables, vector_tables)
                    .iter()
                    .map(|y| tables.tables_2d[*i].eval(x, *y))
                    .collect()
            }
            Self::Table2DY(i, x, y) => {
                let x = x.eval(state, registry);
                y.eval(state, registry)
                    .into_iter()
                    .map(|y| tables.tables_2d[*i].eval(x, y))
                    .collect()
            }
            Self::Table2DXSum(i, VectorExpression::Reference(x), SetExpression::Reference(y)) => {
                let y = y.eval(state, registry, set_variables, set_tables);
                x.eval(state, registry, vector_variables, vector_tables)
                    .iter()
                    .map(|x| tables.tables_2d[*i].sum_y(*x, y.ones()))
                    .collect()
            }
            Self::Table2DXSum(i, VectorExpression::Reference(x), y) => {
                let y = y.eval(state, registry);
                x.eval(state, registry, vector_variables, vector_tables)
                    .iter()
                    .map(|x| tables.tables_2d[*i].sum_y(*x, y.ones()))
                    .collect()
            }
            Self::Table2DXSum(i, x, SetExpression::Reference(y)) => {
                let y = y.eval(state, registry, set_variables, set_tables);
                x.eval(state, registry)
                    .into_iter()
                    .map(|x| tables.tables_2d[*i].sum_y(x, y.ones()))
                    .collect()
            }
            Self::Table2DXSum(i, x, y) => {
                let y = y.eval(state, registry);
                x.eval(state, registry)
                    .into_iter()
                    .map(|x| tables.tables_2d[*i].sum_y(x, y.ones()))
                    .collect()
            }
            Self::Table2DYSum(i, SetExpression::Reference(x), VectorExpression::Reference(y)) => {
                let x = x.eval(state, registry, set_variables, set_tables);
                y.eval(state, registry, vector_variables, vector_tables)
                    .iter()
                    .map(|y| tables.tables_2d[*i].sum_x(x.ones(), *y))
                    .collect()
            }
            Self::Table2DYSum(i, SetExpression::Reference(x), y) => {
                let x = x.eval(state, registry, set_variables, set_tables);
                y.eval(state, registry)
                    .into_iter()
                    .map(|y| tables.tables_2d[*i].sum_x(x.ones(), y))
                    .collect()
            }
            Self::Table2DYSum(i, x, VectorExpression::Reference(y)) => {
                let x = x.eval(state, registry);
                y.eval(state, registry, vector_variables, vector_tables)
                    .iter()
                    .map(|y| tables.tables_2d[*i].sum_x(x.ones(), *y))
                    .collect()
            }
            Self::Table2DYSum(i, x, y) => {
                let x = x.eval(state, registry);
                y.eval(state, registry)
                    .into_iter()
                    .map(|y| tables.tables_2d[*i].sum_x(x.ones(), y))
                    .collect()
            }
        }
    }

    pub fn simplify(
        &self,
        registry: &TableRegistry,
        tables: &TableData<T>,
    ) -> NumericVectorExpression<T> {
        match self {
            Self::Reverse(vector) => match vector.simplify(registry, tables) {
                Self::Constant(mut vector) => {
                    vector.reverse();
                    Self::Constant(vector)
                }
                vector => Self::Reverse(Box::new(vector)),
            },
            Self::Push(value, vector) => {
                match (value.simplify(registry), vector.simplify(registry, tables)) {
                    (NumericExpression::Constant(value), Self::Constant(mut vector)) => {
                        vector.push(value);
                        Self::Constant(vector)
                    }
                    (value, vector) => Self::Push(value, Box::new(vector)),
                }
            }
            Self::Pop(vector) => match vector.simplify(registry, tables) {
                Self::Constant(mut vector) => {
                    vector.pop();
                    Self::Constant(vector)
                }
                vector => Self::Pop(Box::new(vector)),
            },
            Self::Set(value, vector, i) => match (
                value.simplify(registry),
                vector.simplify(registry, tables),
                i.simplify(registry),
            ) {
                (
                    NumericExpression::Constant(value),
                    Self::Constant(mut vector),
                    ElementExpression::Constant(i),
                ) => {
                    vector[i] = value;
                    Self::Constant(vector)
                }
                (value, vector, i) => Self::Set(value, Box::new(vector), i),
            },
            Self::NumericOperation(op, x, y) => {
                match (x.simplify(registry), y.simplify(registry, tables)) {
                    (NumericExpression::Constant(x), Self::Constant(y)) => {
                        Self::Constant(Self::eval_operation(op, x, y))
                    }
                    (x, y) => Self::NumericOperation(op.clone(), x, Box::new(y)),
                }
            }
            Self::VectorOperation(op, x, y) => {
                match (x.simplify(registry, tables), y.simplify(registry, tables)) {
                    (Self::Constant(x), Self::Constant(y)) => {
                        Self::Constant(Self::eval_vector_operation_in_y(op, &x, y))
                    }
                    (x, y) => Self::VectorOperation(op.clone(), Box::new(x), Box::new(y)),
                }
            }
            Self::Table(i, args) => Self::simplify_table(*i, args, registry, tables),
            Self::TableSum(i, args) => Self::simplify_table_sum(*i, args, registry, tables),
            Self::Table1D(i, x) => match x.simplify(registry) {
                VectorExpression::Reference(ReferenceExpression::Constant(x)) => {
                    Self::Constant(x.iter().map(|x| tables.tables_1d[*i].eval(*x)).collect())
                }
                x => Self::Table1D(*i, x),
            },
            Self::Table2D(i, x, y) => match (x.simplify(registry), y.simplify(registry)) {
                (
                    VectorExpression::Reference(ReferenceExpression::Constant(x)),
                    VectorExpression::Reference(ReferenceExpression::Constant(y)),
                ) => Self::Constant(
                    x.into_iter()
                        .zip(y)
                        .map(|(x, y)| tables.tables_2d[*i].eval(x, y))
                        .collect(),
                ),
                (x, y) => Self::Table2D(*i, x, y),
            },
            Self::Table2DX(i, x, y) => match (x.simplify(registry), y.simplify(registry)) {
                (
                    VectorExpression::Reference(ReferenceExpression::Constant(x)),
                    ElementExpression::Constant(y),
                ) => Self::Constant(
                    x.into_iter()
                        .map(|x| tables.tables_2d[*i].eval(x, y))
                        .collect(),
                ),
                (x, y) => Self::Table2DX(*i, x, y),
            },
            Self::Table2DY(i, x, y) => match (x.simplify(registry), y.simplify(registry)) {
                (
                    ElementExpression::Constant(x),
                    VectorExpression::Reference(ReferenceExpression::Constant(y)),
                ) => Self::Constant(
                    y.into_iter()
                        .map(|y| tables.tables_2d[*i].eval(x, y))
                        .collect(),
                ),
                (x, y) => Self::Table2DY(*i, x, y),
            },
            Self::Table2DXSum(i, x, y) => match (x.simplify(registry), y.simplify(registry)) {
                (
                    VectorExpression::Reference(ReferenceExpression::Constant(x)),
                    SetExpression::Reference(ReferenceExpression::Constant(y)),
                ) => Self::Constant(
                    x.into_iter()
                        .map(|x| tables.tables_2d[*i].sum_y(x, y.ones()))
                        .collect(),
                ),
                (x, y) => Self::Table2DXSum(*i, x, y),
            },
            Self::Table2DYSum(i, x, y) => match (x.simplify(registry), y.simplify(registry)) {
                (
                    SetExpression::Reference(ReferenceExpression::Constant(x)),
                    VectorExpression::Reference(ReferenceExpression::Constant(y)),
                ) => Self::Constant(
                    y.into_iter()
                        .map(|y| tables.tables_2d[*i].sum_x(x.ones(), y))
                        .collect(),
                ),
                (x, y) => Self::Table2DYSum(*i, x, y),
            },
            _ => self.clone(),
        }
    }

    fn eval_table(
        i: usize,
        args: &[VectorOrElementExpression],
        state: &State,
        registry: &TableRegistry,
        tables: &TableData<T>,
    ) -> Vec<T> {
        let mut result = vec![vec![]];
        let mut vector_mode = false;
        for arg in args {
            match arg {
                VectorOrElementExpression::Element(element) => {
                    let element = element.eval(state, registry);
                    result.iter_mut().for_each(|r| r.push(element));
                }
                VectorOrElementExpression::Vector(vector) => match vector {
                    VectorExpression::Reference(vector) => {
                        let vector = vector.eval(
                            state,
                            registry,
                            &state.signature_variables.vector_variables,
                            &registry.vector_tables,
                        );
                        if vector_mode {
                            result.iter_mut().zip(vector).for_each(|(r, v)| r.push(*v));
                        } else {
                            result = vector
                                .iter()
                                .map(|v| {
                                    let mut r = result[0].clone();
                                    r.push(*v);
                                    r
                                })
                                .collect();
                            vector_mode = true;
                        }
                    }
                    vector => {
                        let vector = vector.eval(state, registry);
                        if vector_mode {
                            result.iter_mut().zip(vector).for_each(|(r, v)| r.push(v));
                        } else {
                            result = vector
                                .into_iter()
                                .map(|v| {
                                    let mut r = result[0].clone();
                                    r.push(v);
                                    r
                                })
                                .collect();
                            vector_mode = true;
                        }
                    }
                },
            }
        }
        match args.len() {
            1 => result
                .into_iter()
                .map(|r| tables.tables_1d[i].eval(r[0]))
                .collect(),
            2 => result
                .into_iter()
                .map(|r| tables.tables_2d[i].eval(r[0], r[1]))
                .collect(),
            3 => result
                .into_iter()
                .map(|r| tables.tables_3d[i].eval(r[0], r[1], r[2]))
                .collect(),
            _ => result
                .into_iter()
                .map(|r| tables.tables[i].eval(&r))
                .collect(),
        }
    }

    fn simplify_table(
        i: usize,
        args: &[VectorOrElementExpression],
        registry: &TableRegistry,
        tables: &TableData<T>,
    ) -> NumericVectorExpression<T> {
        let args: Vec<VectorOrElementExpression> =
            args.iter().map(|x| x.simplify(registry)).collect();
        let mut simplified_args = vec![vec![]];
        let mut vector_mode = false;
        for arg in &args {
            match arg {
                VectorOrElementExpression::Element(ElementExpression::Constant(element)) => {
                    simplified_args.iter_mut().for_each(|r| r.push(*element));
                }
                VectorOrElementExpression::Vector(VectorExpression::Reference(
                    ReferenceExpression::Constant(vector),
                )) => {
                    if vector_mode {
                        simplified_args
                            .iter_mut()
                            .zip(vector)
                            .for_each(|(r, v)| r.push(*v));
                    } else {
                        simplified_args = vector
                            .iter()
                            .map(|v| {
                                let mut r = simplified_args[0].clone();
                                r.push(*v);
                                r
                            })
                            .collect();
                        vector_mode = true;
                    }
                }
                _ => return NumericVectorExpression::Table(i, args),
            }
        }
        Self::Constant(match args.len() {
            1 => simplified_args
                .into_iter()
                .map(|r| tables.tables_1d[i].eval(r[0]))
                .collect(),
            2 => simplified_args
                .into_iter()
                .map(|r| tables.tables_2d[i].eval(r[0], r[1]))
                .collect(),
            3 => simplified_args
                .into_iter()
                .map(|r| tables.tables_3d[i].eval(r[0], r[1], r[2]))
                .collect(),
            _ => simplified_args
                .into_iter()
                .map(|r| tables.tables[i].eval(&r))
                .collect(),
        })
    }

    fn eval_table_sum(
        i: usize,
        args: &[ArgumentExpression],
        state: &State,
        registry: &TableRegistry,
        tables: &TableData<T>,
    ) -> Vec<T> {
        let mut result = vec![vec![vec![]]];
        let mut vector_mode = false;
        for arg in args {
            match arg {
                ArgumentExpression::Element(element) => {
                    let element = element.eval(state, registry);
                    result
                        .iter_mut()
                        .for_each(|rr| rr.iter_mut().for_each(|r| r.push(element)));
                }
                ArgumentExpression::Set(set) => {
                    result = match set {
                        SetExpression::Reference(set) => {
                            let set = set.eval(
                                state,
                                registry,
                                &state.signature_variables.set_variables,
                                &registry.set_tables,
                            );
                            result
                                .into_iter()
                                .map(|rr| util::expand_vector_with_set(rr, set))
                                .collect()
                        }
                        _ => {
                            let set = &set.eval(state, registry);
                            result
                                .into_iter()
                                .map(|rr| util::expand_vector_with_set(rr, &set))
                                .collect()
                        }
                    };
                }
                ArgumentExpression::Vector(vector) => {
                    if vector_mode {
                        match vector {
                            VectorExpression::Reference(vector) => result
                                .iter_mut()
                                .zip(vector.eval(
                                    state,
                                    registry,
                                    &state.signature_variables.vector_variables,
                                    &registry.vector_tables,
                                ))
                                .for_each(|(rr, v)| rr.iter_mut().for_each(|r| r.push(*v))),
                            _ => result
                                .iter_mut()
                                .zip(vector.eval(state, registry))
                                .for_each(|(rr, v)| rr.iter_mut().for_each(|r| r.push(v))),
                        }
                    } else {
                        result = match vector {
                            VectorExpression::Reference(vector) => vector
                                .eval(
                                    state,
                                    registry,
                                    &state.signature_variables.vector_variables,
                                    &registry.vector_tables,
                                )
                                .iter()
                                .map(|v| {
                                    let mut rr = result[0].clone();
                                    rr.iter_mut().for_each(|r| r.push(*v));
                                    rr
                                })
                                .collect(),
                            _ => vector
                                .eval(state, registry)
                                .into_iter()
                                .map(|v| {
                                    let mut rr = result[0].clone();
                                    rr.iter_mut().for_each(|r| r.push(v));
                                    rr
                                })
                                .collect(),
                        };
                        vector_mode = true;
                    }
                }
            }
        }
        match args.len() {
            1 => result
                .into_iter()
                .map(|rr| rr.into_iter().map(|r| tables.tables_1d[i].eval(r[0])).sum())
                .collect(),
            2 => result
                .into_iter()
                .map(|rr| {
                    rr.into_iter()
                        .map(|r| tables.tables_2d[i].eval(r[0], r[1]))
                        .sum()
                })
                .collect(),
            3 => result
                .into_iter()
                .map(|rr| {
                    rr.into_iter()
                        .map(|r| tables.tables_3d[i].eval(r[0], r[1], r[2]))
                        .sum()
                })
                .collect(),
            _ => result
                .into_iter()
                .map(|rr| rr.into_iter().map(|r| tables.tables[i].eval(&r)).sum())
                .collect(),
        }
    }

    fn simplify_table_sum(
        i: usize,
        args: &[ArgumentExpression],
        registry: &TableRegistry,
        tables: &TableData<T>,
    ) -> NumericVectorExpression<T> {
        let args: Vec<ArgumentExpression> = args.iter().map(|x| x.simplify(registry)).collect();
        let mut simplified_args = vec![vec![vec![]]];
        let mut vector_mode = false;
        for arg in &args {
            match arg {
                ArgumentExpression::Element(ElementExpression::Constant(element)) => {
                    simplified_args
                        .iter_mut()
                        .for_each(|rr| rr.iter_mut().for_each(|r| r.push(*element)));
                }
                ArgumentExpression::Set(SetExpression::Reference(
                    ReferenceExpression::Constant(set),
                )) => {
                    simplified_args = simplified_args
                        .into_iter()
                        .map(|rr| util::expand_vector_with_set(rr, set))
                        .collect();
                }
                ArgumentExpression::Vector(VectorExpression::Reference(
                    ReferenceExpression::Constant(vector),
                )) => {
                    if vector_mode {
                        simplified_args
                            .iter_mut()
                            .zip(vector)
                            .for_each(|(rr, v)| rr.iter_mut().for_each(|r| r.push(*v)));
                    } else {
                        simplified_args = vector
                            .iter()
                            .map(|v| {
                                let mut rr = simplified_args[0].clone();
                                rr.iter_mut().for_each(|r| r.push(*v));
                                rr
                            })
                            .collect();
                        vector_mode = true;
                    }
                }
                _ => return Self::TableSum(i, args),
            }
        }
        Self::Constant(match args.len() {
            1 => simplified_args
                .into_iter()
                .map(|rr| rr.into_iter().map(|r| tables.tables_1d[i].eval(r[0])).sum())
                .collect(),
            2 => simplified_args
                .into_iter()
                .map(|rr| {
                    rr.into_iter()
                        .map(|r| tables.tables_2d[i].eval(r[0], r[1]))
                        .sum()
                })
                .collect(),
            3 => simplified_args
                .into_iter()
                .map(|rr| {
                    rr.into_iter()
                        .map(|r| tables.tables_3d[i].eval(r[0], r[1], r[2]))
                        .sum()
                })
                .collect(),
            _ => simplified_args
                .into_iter()
                .map(|rr| rr.into_iter().map(|r| tables.tables[i].eval(&r)).sum())
                .collect(),
        })
    }

    fn eval_operation(op: &NumericOperator, x: T, mut y: Vec<T>) -> Vec<T> {
        match op {
            NumericOperator::Add => y.iter_mut().for_each(|y| *y = *y + x),
            NumericOperator::Subtract => y.iter_mut().for_each(|y| *y = *y - x),
            NumericOperator::Multiply => y.iter_mut().for_each(|y| *y = *y * x),
            NumericOperator::Divide => y.iter_mut().for_each(|y| *y = *y / x),
            NumericOperator::Max => y.iter_mut().for_each(|y| {
                if x > *y {
                    *y = x
                }
            }),
            NumericOperator::Min => y.iter_mut().for_each(|y| {
                if x < *y {
                    *y = x
                }
            }),
        }
        y
    }

    fn eval_vector_operation_in_x(op: &NumericOperator, mut x: Vec<T>, y: &[T]) -> Vec<T> {
        x.truncate(y.len());
        match op {
            NumericOperator::Add => x.iter_mut().zip(y).for_each(|(x, y)| *x = *x + *y),
            NumericOperator::Subtract => x.iter_mut().zip(y).for_each(|(x, y)| *x = *x - *y),
            NumericOperator::Multiply => x.iter_mut().zip(y).for_each(|(x, y)| *x = *x * *y),
            NumericOperator::Divide => x.iter_mut().zip(y).for_each(|(x, y)| *x = *x / *y),
            NumericOperator::Max => x.iter_mut().zip(y).for_each(|(x, y)| {
                if *y > *x {
                    *x = *y
                }
            }),
            NumericOperator::Min => x.iter_mut().zip(y).for_each(|(x, y)| {
                if *y < *x {
                    *x = *y
                }
            }),
        }
        x
    }

    fn eval_vector_operation_in_y(op: &NumericOperator, x: &[T], mut y: Vec<T>) -> Vec<T> {
        y.truncate(x.len());
        match op {
            NumericOperator::Add => y.iter_mut().zip(x).for_each(|(y, x)| *y = *x + *y),
            NumericOperator::Subtract => y.iter_mut().zip(x).for_each(|(y, x)| *y = *x - *y),
            NumericOperator::Multiply => y.iter_mut().zip(x).for_each(|(y, x)| *y = *x * *y),
            NumericOperator::Divide => y.iter_mut().zip(x).for_each(|(y, x)| *y = *x / *y),
            NumericOperator::Max => y.iter_mut().zip(x).for_each(|(y, x)| {
                if *x > *y {
                    *y = *x
                }
            }),
            NumericOperator::Min => y.iter_mut().zip(x).for_each(|(y, x)| {
                if *x < *y {
                    *y = *x
                }
            }),
        }
        y
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum VectorOrElementExpression {
    Vector(VectorExpression),
    Element(ElementExpression),
}

impl VectorOrElementExpression {
    pub fn simplify(&self, registry: &TableRegistry) -> VectorOrElementExpression {
        match self {
            VectorOrElementExpression::Vector(vector) => {
                VectorOrElementExpression::Vector(vector.simplify(registry))
            }
            VectorOrElementExpression::Element(element) => {
                VectorOrElementExpression::Element(element.simplify(registry))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::element_expression;
    use super::super::reference_expression::ReferenceExpression;
    use super::*;
    use crate::state;
    use crate::table;
    use crate::table_data;
    use crate::variable::*;
    use approx::assert_relative_eq;
    use ordered_float::OrderedFloat;
    use rustc_hash::FxHashMap;
    use std::rc::Rc;

    fn generate_state() -> state::State {
        let mut set1 = Set::with_capacity(3);
        set1.insert(0);
        set1.insert(2);
        let mut set2 = Set::with_capacity(3);
        set2.insert(0);
        set2.insert(1);
        state::State {
            signature_variables: Rc::new(state::SignatureVariables {
                set_variables: vec![set1, set2],
                vector_variables: vec![vec![0, 2]],
                element_variables: vec![1],
                integer_variables: vec![1, 2, 3],
                continuous_variables: vec![OrderedFloat(1.0), OrderedFloat(2.0), OrderedFloat(3.0)],
            }),
            resource_variables: state::ResourceVariables {
                integer_variables: vec![4, 5, 6],
                continuous_variables: vec![4.0, 5.0, 6.0],
            },
            stage: 0,
        }
    }

    fn generate_registry() -> TableRegistry {
        let mut name_to_constant = FxHashMap::default();
        name_to_constant.insert(String::from("f0"), 0);

        let tables_1d = vec![table::Table1D::new(vec![10, 20, 30])];
        let mut name_to_table_1d = FxHashMap::default();
        name_to_table_1d.insert(String::from("f1"), 0);

        let tables_2d = vec![table::Table2D::new(vec![
            vec![10, 20, 30],
            vec![40, 50, 60],
            vec![70, 80, 90],
        ])];
        let mut name_to_table_2d = FxHashMap::default();
        name_to_table_2d.insert(String::from("f2"), 0);

        let tables_3d = vec![table::Table3D::new(vec![
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
        ])];
        let mut name_to_table_3d = FxHashMap::default();
        name_to_table_3d.insert(String::from("f3"), 0);

        let mut map = FxHashMap::default();
        let key = vec![0, 1, 0, 0];
        map.insert(key, 100);
        let key = vec![0, 1, 0, 1];
        map.insert(key, 200);
        let key = vec![0, 1, 2, 0];
        map.insert(key, 300);
        let key = vec![0, 1, 2, 1];
        map.insert(key, 400);
        let tables = vec![table::Table::new(map, 0)];
        let mut name_to_table = FxHashMap::default();
        name_to_table.insert(String::from("f4"), 0);

        let integer_tables = table_data::TableData {
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

        let mut name_to_constant = FxHashMap::default();
        name_to_constant.insert(String::from("cf0"), 0.0);

        let tables_1d = vec![table::Table1D::new(vec![10.0, 20.0, 30.0])];
        let mut name_to_table_1d = FxHashMap::default();
        name_to_table_1d.insert(String::from("cf1"), 0);

        let tables_2d = vec![table::Table2D::new(vec![
            vec![10.0, 20.0, 30.0],
            vec![40.0, 50.0, 60.0],
            vec![70.0, 80.0, 90.0],
        ])];
        let mut name_to_table_2d = FxHashMap::default();
        name_to_table_2d.insert(String::from("cf2"), 0);

        let tables_3d = vec![table::Table3D::new(vec![
            vec![
                vec![10.0, 20.0, 30.0],
                vec![40.0, 50.0, 60.0],
                vec![70.0, 80.0, 90.0],
            ],
            vec![
                vec![10.0, 20.0, 30.0],
                vec![40.0, 50.0, 60.0],
                vec![70.0, 80.0, 90.0],
            ],
            vec![
                vec![10.0, 20.0, 30.0],
                vec![40.0, 50.0, 60.0],
                vec![70.0, 80.0, 90.0],
            ],
        ])];
        let mut name_to_table_3d = FxHashMap::default();
        name_to_table_3d.insert(String::from("cf3"), 0);

        let mut map = FxHashMap::default();
        let key = vec![0, 1, 0, 0];
        map.insert(key, 100.0);
        let key = vec![0, 1, 0, 1];
        map.insert(key, 200.0);
        let key = vec![0, 1, 2, 0];
        map.insert(key, 300.0);
        let key = vec![0, 1, 2, 1];
        map.insert(key, 400.0);
        let tables = vec![table::Table::new(map, 0.0)];
        let mut name_to_table = FxHashMap::default();
        name_to_table.insert(String::from("cf4"), 0);

        let continuous_tables = table_data::TableData {
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

        TableRegistry {
            integer_tables,
            continuous_tables,
            ..Default::default()
        }
    }

    #[test]
    fn constant_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let expression = NumericExpression::Constant(2);
        assert_eq!(expression.eval(&state, &registry), 2);
    }

    #[test]
    fn integer_variable_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let expression = NumericExpression::<Integer>::IntegerVariable(0);
        assert_eq!(expression.eval(&state, &registry), 1);
        let expression = NumericExpression::<Integer>::IntegerVariable(1);
        assert_eq!(expression.eval(&state, &registry), 2);
        let expression = NumericExpression::<Integer>::IntegerVariable(2);
        assert_eq!(expression.eval(&state, &registry), 3);
    }

    #[test]
    fn integer_resource_variable_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let expression = NumericExpression::<Integer>::IntegerResourceVariable(0);
        assert_eq!(expression.eval(&state, &registry), 4);
        let expression = NumericExpression::<Integer>::IntegerResourceVariable(1);
        assert_eq!(expression.eval(&state, &registry), 5);
        let expression = NumericExpression::<Integer>::IntegerResourceVariable(2);
        assert_eq!(expression.eval(&state, &registry), 6);
    }

    #[test]
    fn continuous_variable_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let expression = NumericExpression::<Continuous>::ContinuousVariable(0);
        assert_relative_eq!(expression.eval(&state, &registry), 1.0);
        let expression = NumericExpression::<Continuous>::ContinuousVariable(1);
        assert_relative_eq!(expression.eval(&state, &registry), 2.0);
        let expression = NumericExpression::<Continuous>::ContinuousVariable(2);
        assert_relative_eq!(expression.eval(&state, &registry), 3.0);
    }

    #[test]
    fn continuous_resource_variable_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let expression = NumericExpression::<Continuous>::ContinuousResourceVariable(0);
        assert_relative_eq!(expression.eval(&state, &registry), 4.0);
        let expression = NumericExpression::<Continuous>::ContinuousResourceVariable(1);
        assert_relative_eq!(expression.eval(&state, &registry), 5.0);
        let expression = NumericExpression::<Continuous>::ContinuousResourceVariable(2);
        assert_relative_eq!(expression.eval(&state, &registry), 6.0);
    }

    #[test]
    fn cost_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let expression: NumericExpression<Integer> = NumericExpression::Cost;
        assert_eq!(expression.eval_cost(0, &state, &registry), 0);
        let expression: NumericExpression<Integer> = NumericExpression::NumericOperation(
            NumericOperator::Add,
            Box::new(NumericExpression::Cost),
            Box::new(NumericExpression::Constant(1)),
        );
        assert_eq!(expression.eval_cost(0, &state, &registry), 1);
    }

    #[test]
    #[should_panic]
    fn cost_eval_panic() {
        let registry = generate_registry();
        let state = generate_state();
        let expression: NumericExpression<Integer> = NumericExpression::Cost;
        assert_eq!(expression.eval(&state, &registry), 0);
    }

    #[test]
    fn add_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let expression: NumericExpression<Integer> = NumericExpression::NumericOperation(
            NumericOperator::Add,
            Box::new(NumericExpression::Constant(3)),
            Box::new(NumericExpression::Constant(2)),
        );
        assert_eq!(expression.eval(&state, &registry), 5);
    }

    #[test]
    fn subtract_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let expression: NumericExpression<Integer> = NumericExpression::NumericOperation(
            NumericOperator::Subtract,
            Box::new(NumericExpression::Constant(3)),
            Box::new(NumericExpression::Constant(2)),
        );
        assert_eq!(expression.eval(&state, &registry), 1);
    }

    #[test]
    fn multiply_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let expression: NumericExpression<Integer> = NumericExpression::NumericOperation(
            NumericOperator::Multiply,
            Box::new(NumericExpression::Constant(3)),
            Box::new(NumericExpression::Constant(2)),
        );
        assert_eq!(expression.eval(&state, &registry), 6);
    }

    #[test]
    fn divide_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let expression: NumericExpression<Integer> = NumericExpression::NumericOperation(
            NumericOperator::Divide,
            Box::new(NumericExpression::Constant(3)),
            Box::new(NumericExpression::Constant(2)),
        );
        assert_eq!(expression.eval(&state, &registry), 1);
    }

    #[test]
    fn max_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let expression: NumericExpression<Integer> = NumericExpression::NumericOperation(
            NumericOperator::Max,
            Box::new(NumericExpression::Constant(3)),
            Box::new(NumericExpression::Constant(2)),
        );
        assert_eq!(expression.eval(&state, &registry), 3);
    }

    #[test]
    fn min_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let expression: NumericExpression<Integer> = NumericExpression::NumericOperation(
            NumericOperator::Min,
            Box::new(NumericExpression::Constant(3)),
            Box::new(NumericExpression::Constant(2)),
        );
        assert_eq!(expression.eval(&state, &registry), 2);
    }

    #[test]
    fn cardinality_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let expression = NumericExpression::<Integer>::Cardinality(SetExpression::Reference(
            ReferenceExpression::Variable(0),
        ));
        assert_eq!(expression.eval(&state, &registry), 2);
        let expression = NumericExpression::<Integer>::Cardinality(SetExpression::Complement(
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
        ));
        assert_eq!(expression.eval(&state, &registry), 1);
    }

    #[test]
    fn length_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let expression = NumericExpression::<Integer>::Length(VectorExpression::Reference(
            ReferenceExpression::Variable(0),
        ));
        assert_eq!(expression.eval(&state, &registry), 2);
        let expression = NumericExpression::<Integer>::Length(VectorExpression::Reverse(Box::new(
            VectorExpression::Reference(ReferenceExpression::Variable(0)),
        )));
        assert_eq!(expression.eval(&state, &registry), 2);
    }

    #[test]
    fn integer_table_eval() {
        let registry = generate_registry();
        let state = generate_state();

        let expression = NumericExpression::<Integer>::IntegerTable(NumericTableExpression::Table(
            0,
            vec![
                element_expression::ElementExpression::Constant(0),
                element_expression::ElementExpression::Constant(1),
                element_expression::ElementExpression::Constant(0),
                element_expression::ElementExpression::Constant(0),
            ],
        ));
        assert_eq!(expression.eval(&state, &registry), 100);
        let expression = NumericExpression::<Integer>::IntegerTable(NumericTableExpression::Table(
            0,
            vec![
                element_expression::ElementExpression::Constant(0),
                element_expression::ElementExpression::Constant(1),
                element_expression::ElementExpression::Constant(0),
                element_expression::ElementExpression::Constant(1),
            ],
        ));
        assert_eq!(expression.eval(&state, &registry), 200);
        let expression = NumericExpression::<Integer>::IntegerTable(NumericTableExpression::Table(
            0,
            vec![
                element_expression::ElementExpression::Constant(0),
                element_expression::ElementExpression::Constant(1),
                element_expression::ElementExpression::Constant(2),
                element_expression::ElementExpression::Constant(0),
            ],
        ));
        assert_eq!(expression.eval(&state, &registry), 300);
        let expression = NumericExpression::<Integer>::IntegerTable(NumericTableExpression::Table(
            0,
            vec![
                element_expression::ElementExpression::Constant(0),
                element_expression::ElementExpression::Constant(1),
                element_expression::ElementExpression::Constant(2),
                element_expression::ElementExpression::Constant(1),
            ],
        ));
        assert_eq!(expression.eval(&state, &registry), 400);
    }

    #[test]
    fn continuous_table_eval() {
        let registry = generate_registry();
        let state = generate_state();

        let expression =
            NumericExpression::<Continuous>::ContinuousTable(NumericTableExpression::Table(
                0,
                vec![
                    element_expression::ElementExpression::Constant(0),
                    element_expression::ElementExpression::Constant(1),
                    element_expression::ElementExpression::Constant(0),
                    element_expression::ElementExpression::Constant(0),
                ],
            ));
        assert_relative_eq!(expression.eval(&state, &registry), 100.0);
        let expression =
            NumericExpression::<Continuous>::ContinuousTable(NumericTableExpression::Table(
                0,
                vec![
                    element_expression::ElementExpression::Constant(0),
                    element_expression::ElementExpression::Constant(1),
                    element_expression::ElementExpression::Constant(0),
                    element_expression::ElementExpression::Constant(1),
                ],
            ));
        assert_relative_eq!(expression.eval(&state, &registry), 200.0);
        let expression =
            NumericExpression::<Continuous>::ContinuousTable(NumericTableExpression::Table(
                0,
                vec![
                    element_expression::ElementExpression::Constant(0),
                    element_expression::ElementExpression::Constant(1),
                    element_expression::ElementExpression::Constant(2),
                    element_expression::ElementExpression::Constant(0),
                ],
            ));
        assert_relative_eq!(expression.eval(&state, &registry), 300.0);
        let expression =
            NumericExpression::<Continuous>::ContinuousTable(NumericTableExpression::Table(
                0,
                vec![
                    element_expression::ElementExpression::Constant(0),
                    element_expression::ElementExpression::Constant(1),
                    element_expression::ElementExpression::Constant(2),
                    element_expression::ElementExpression::Constant(1),
                ],
            ));
        assert_relative_eq!(expression.eval(&state, &registry), 400.0);
    }

    #[test]
    fn integer_last_eval() {
        let state = generate_state();
        let registry = generate_registry();
        let expression = NumericExpression::<Integer>::IntegerLast(Box::new(
            NumericVectorExpression::Constant(vec![0, 1]),
        ));
        assert_eq!(expression.eval(&state, &registry), 1);
        let expression =
            NumericExpression::<Integer>::IntegerLast(Box::new(NumericVectorExpression::Reverse(
                Box::new(NumericVectorExpression::Constant(vec![0, 1])),
            )));
        assert_eq!(expression.eval(&state, &registry), 0);
    }

    #[test]
    fn continuous_last_eval() {
        let state = generate_state();
        let registry = generate_registry();
        let expression = NumericExpression::<Continuous>::ContinuousLast(Box::new(
            NumericVectorExpression::Constant(vec![0.0, 1.0]),
        ));
        assert_relative_eq!(expression.eval(&state, &registry), 1.0);
        let expression = NumericExpression::<Continuous>::ContinuousLast(Box::new(
            NumericVectorExpression::Reverse(Box::new(NumericVectorExpression::Constant(vec![
                0.0, 1.0,
            ]))),
        ));
        assert_relative_eq!(expression.eval(&state, &registry), 0.0);
    }

    #[test]
    fn integer_at_eval() {
        let state = generate_state();
        let registry = generate_registry();
        let expression = NumericExpression::<Integer>::IntegerAt(
            Box::new(NumericVectorExpression::Constant(vec![2, 1])),
            ElementExpression::Constant(0),
        );
        assert_eq!(expression.eval(&state, &registry), 2);
        let expression = NumericExpression::<Integer>::IntegerAt(
            Box::new(NumericVectorExpression::Reverse(Box::new(
                NumericVectorExpression::Constant(vec![2, 1]),
            ))),
            ElementExpression::Constant(0),
        );
        assert_eq!(expression.eval(&state, &registry), 1);
    }

    #[test]
    fn continuous_at_eval() {
        let state = generate_state();
        let registry = generate_registry();
        let expression = NumericExpression::<Continuous>::ContinuousAt(
            Box::new(NumericVectorExpression::Constant(vec![2.0, 1.0])),
            ElementExpression::Constant(0),
        );
        assert_relative_eq!(expression.eval(&state, &registry), 2.0);
        let expression = NumericExpression::<Continuous>::ContinuousAt(
            Box::new(NumericVectorExpression::Reverse(Box::new(
                NumericVectorExpression::Constant(vec![2.0, 1.0]),
            ))),
            ElementExpression::Constant(0),
        );
        assert_relative_eq!(expression.eval(&state, &registry), 1.0);
    }

    #[test]
    fn integer_reduce_sum_eval() {
        let state = generate_state();
        let registry = generate_registry();
        let expression = NumericExpression::<Integer>::IntegerReduceSum(Box::new(
            NumericVectorExpression::Constant(vec![2, 1]),
        ));
        assert_eq!(expression.eval(&state, &registry), 3);
        let expression = NumericExpression::<Integer>::IntegerReduceSum(Box::new(
            NumericVectorExpression::Reverse(Box::new(NumericVectorExpression::Constant(vec![
                2, 1,
            ]))),
        ));
        assert_eq!(expression.eval(&state, &registry), 3);
    }

    #[test]
    fn continuous_reduce_sum_eval() {
        let state = generate_state();
        let registry = generate_registry();
        let expression = NumericExpression::<Continuous>::ContinuousReduceSum(Box::new(
            NumericVectorExpression::Constant(vec![2.0, 1.0]),
        ));
        assert_relative_eq!(expression.eval(&state, &registry), 3.0);
        let expression = NumericExpression::<Continuous>::ContinuousReduceSum(Box::new(
            NumericVectorExpression::Reverse(Box::new(NumericVectorExpression::Constant(vec![
                2.0, 1.0,
            ]))),
        ));
        assert_relative_eq!(expression.eval(&state, &registry), 3.0);
    }

    #[test]
    fn integer_reduce_product_eval() {
        let state = generate_state();
        let registry = generate_registry();
        let expression = NumericExpression::<Integer>::IntegerReduceProduct(Box::new(
            NumericVectorExpression::Constant(vec![2, 1]),
        ));
        assert_eq!(expression.eval(&state, &registry), 2);
        let expression = NumericExpression::<Integer>::IntegerReduceProduct(Box::new(
            NumericVectorExpression::Reverse(Box::new(NumericVectorExpression::Constant(vec![
                2, 1,
            ]))),
        ));
        assert_eq!(expression.eval(&state, &registry), 2);
    }

    #[test]
    fn continuous_reduce_product_eval() {
        let state = generate_state();
        let registry = generate_registry();
        let expression = NumericExpression::<Continuous>::ContinuousReduceProduct(Box::new(
            NumericVectorExpression::Constant(vec![2.0, 1.0]),
        ));
        assert_relative_eq!(expression.eval(&state, &registry), 2.0);
        let expression = NumericExpression::<Continuous>::ContinuousReduceProduct(Box::new(
            NumericVectorExpression::Reverse(Box::new(NumericVectorExpression::Constant(vec![
                2.0, 1.0,
            ]))),
        ));
        assert_relative_eq!(expression.eval(&state, &registry), 2.0);
    }

    #[test]
    fn constant_simplify() {
        let registry = generate_registry();
        let expression = NumericExpression::Constant(2);
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn integer_variable_simplify() {
        let registry = generate_registry();
        let expression = NumericExpression::<Integer>::IntegerVariable(0);
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn continuous_variable_simplify() {
        let registry = generate_registry();
        let expression = NumericExpression::<Integer>::ContinuousVariable(0);
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn integer_resource_variable_simplify() {
        let registry = generate_registry();
        let expression = NumericExpression::<Integer>::IntegerResourceVariable(0);
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn continuous_resource_variable_simplify() {
        let registry = generate_registry();
        let expression = NumericExpression::<Integer>::ContinuousResourceVariable(0);
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn cost_simplify() {
        let registry = generate_registry();
        let expression: NumericExpression<Integer> = NumericExpression::Cost {};
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn add_simplify() {
        let registry = generate_registry();

        let expression: NumericExpression<Integer> = NumericExpression::NumericOperation(
            NumericOperator::Add,
            Box::new(NumericExpression::Constant(3)),
            Box::new(NumericExpression::Constant(2)),
        );
        assert_eq!(
            expression.simplify(&registry),
            NumericExpression::Constant(5)
        );

        let expression: NumericExpression<Integer> = NumericExpression::NumericOperation(
            NumericOperator::Add,
            Box::new(NumericExpression::IntegerVariable(0)),
            Box::new(NumericExpression::Constant(2)),
        );
        let simplified = expression.simplify(&registry);
        assert_eq!(simplified, expression);
    }

    #[test]
    fn subtract_simplify() {
        let registry = generate_registry();
        let expression: NumericExpression<Integer> = NumericExpression::NumericOperation(
            NumericOperator::Subtract,
            Box::new(NumericExpression::Constant(3)),
            Box::new(NumericExpression::Constant(2)),
        );
        assert_eq!(
            expression.simplify(&registry),
            NumericExpression::Constant(1)
        );

        let expression: NumericExpression<Integer> = NumericExpression::NumericOperation(
            NumericOperator::Subtract,
            Box::new(NumericExpression::IntegerVariable(0)),
            Box::new(NumericExpression::Constant(2)),
        );
        assert_eq!(expression.simplify(&registry), expression,);
    }

    #[test]
    fn multiply_simplify() {
        let registry = generate_registry();
        let expression: NumericExpression<Integer> = NumericExpression::NumericOperation(
            NumericOperator::Multiply,
            Box::new(NumericExpression::Constant(3)),
            Box::new(NumericExpression::Constant(2)),
        );
        assert_eq!(
            expression.simplify(&registry),
            NumericExpression::Constant(6)
        );

        let expression: NumericExpression<Integer> = NumericExpression::NumericOperation(
            NumericOperator::Multiply,
            Box::new(NumericExpression::IntegerVariable(0)),
            Box::new(NumericExpression::Constant(2)),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn divide_simplify() {
        let registry = generate_registry();
        let expression: NumericExpression<Integer> = NumericExpression::NumericOperation(
            NumericOperator::Divide,
            Box::new(NumericExpression::Constant(3)),
            Box::new(NumericExpression::Constant(2)),
        );
        assert_eq!(
            expression.simplify(&registry),
            NumericExpression::Constant(1)
        );

        let expression: NumericExpression<Integer> = NumericExpression::NumericOperation(
            NumericOperator::Divide,
            Box::new(NumericExpression::IntegerVariable(0)),
            Box::new(NumericExpression::Constant(2)),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn max_simplify() {
        let registry = generate_registry();
        let expression: NumericExpression<Integer> = NumericExpression::NumericOperation(
            NumericOperator::Max,
            Box::new(NumericExpression::Constant(3)),
            Box::new(NumericExpression::Constant(2)),
        );
        assert_eq!(
            expression.simplify(&registry),
            NumericExpression::Constant(3)
        );

        let expression: NumericExpression<Integer> = NumericExpression::NumericOperation(
            NumericOperator::Max,
            Box::new(NumericExpression::IntegerVariable(0)),
            Box::new(NumericExpression::Constant(2)),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn min_simplify() {
        let registry = generate_registry();
        let expression: NumericExpression<Integer> = NumericExpression::NumericOperation(
            NumericOperator::Min,
            Box::new(NumericExpression::Constant(3)),
            Box::new(NumericExpression::Constant(2)),
        );
        assert_eq!(
            expression.simplify(&registry),
            NumericExpression::Constant(2)
        );

        let expression: NumericExpression<Integer> = NumericExpression::NumericOperation(
            NumericOperator::Min,
            Box::new(NumericExpression::IntegerVariable(0)),
            Box::new(NumericExpression::Constant(2)),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn cardinality_simplify() {
        let registry = generate_registry();
        let expression = NumericExpression::<Integer>::Cardinality(SetExpression::Reference(
            ReferenceExpression::Variable(0),
        ));
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn integer_table_simplify() {
        let registry = generate_registry();

        let expression = NumericExpression::IntegerTable(NumericTableExpression::Table1D(
            0,
            element_expression::ElementExpression::Constant(0),
        ));
        assert_eq!(
            expression.simplify(&registry),
            NumericExpression::Constant(10)
        );

        let expression = NumericExpression::<Integer>::IntegerTable(
            NumericTableExpression::Table1D(0, element_expression::ElementExpression::Variable(0)),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn continuous_table_simplify() {
        let registry = generate_registry();

        let expression = NumericExpression::ContinuousTable(NumericTableExpression::Table1D(
            0,
            element_expression::ElementExpression::Constant(0),
        ));
        assert_eq!(
            expression.simplify(&registry),
            NumericExpression::Constant(10.0)
        );

        let expression = NumericExpression::<Continuous>::ContinuousTable(
            NumericTableExpression::Table1D(0, element_expression::ElementExpression::Variable(0)),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]

    fn integer_last_simplify() {
        let registry = generate_registry();
        let expression = NumericExpression::<Integer>::IntegerLast(Box::new(
            NumericVectorExpression::Constant(vec![0, 1]),
        ));
        assert_eq!(
            expression.simplify(&registry),
            NumericExpression::Constant(1)
        );
        let expression =
            NumericExpression::<Integer>::IntegerLast(Box::new(NumericVectorExpression::Table2DX(
                0,
                VectorExpression::Reference(ReferenceExpression::Variable(0)),
                ElementExpression::Variable(0),
            )));
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn continuous_last_simplify() {
        let registry = generate_registry();
        let expression = NumericExpression::<Continuous>::ContinuousLast(Box::new(
            NumericVectorExpression::Constant(vec![0.0, 1.0]),
        ));
        assert_eq!(
            expression.simplify(&registry),
            NumericExpression::Constant(1.0)
        );
        let expression = NumericExpression::<Continuous>::ContinuousLast(Box::new(
            NumericVectorExpression::Table2DX(
                0,
                VectorExpression::Reference(ReferenceExpression::Variable(0)),
                ElementExpression::Variable(0),
            ),
        ));
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn integer_at_simplify() {
        let registry = generate_registry();
        let expression = NumericExpression::<Integer>::IntegerAt(
            Box::new(NumericVectorExpression::Constant(vec![2, 1])),
            ElementExpression::Constant(0),
        );
        assert_eq!(
            expression.simplify(&registry),
            NumericExpression::Constant(2)
        );
        let expression = NumericExpression::<Integer>::IntegerAt(
            Box::new(NumericVectorExpression::Constant(vec![2, 1])),
            ElementExpression::Variable(0),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn continuous_at_simplify() {
        let registry = generate_registry();
        let expression = NumericExpression::<Continuous>::ContinuousAt(
            Box::new(NumericVectorExpression::Constant(vec![2.0, 1.0])),
            ElementExpression::Constant(0),
        );
        assert_eq!(
            expression.simplify(&registry),
            NumericExpression::Constant(2.0)
        );
        let expression = NumericExpression::<Continuous>::ContinuousAt(
            Box::new(NumericVectorExpression::Constant(vec![2.0, 1.0])),
            ElementExpression::Variable(0),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn integer_reduce_sum_simplify() {
        let registry = generate_registry();
        let expression = NumericExpression::<Integer>::IntegerReduceSum(Box::new(
            NumericVectorExpression::Constant(vec![2, 1]),
        ));
        assert_eq!(
            expression.simplify(&registry),
            NumericExpression::Constant(3)
        );
        let expression = NumericExpression::<Integer>::IntegerReduceSum(Box::new(
            NumericVectorExpression::Table2DX(
                0,
                VectorExpression::Reference(ReferenceExpression::Variable(0)),
                ElementExpression::Variable(0),
            ),
        ));
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn continuous_reduce_sum_simplify() {
        let registry = generate_registry();
        let expression = NumericExpression::<Continuous>::ContinuousReduceSum(Box::new(
            NumericVectorExpression::Constant(vec![2.0, 1.0]),
        ));
        assert_eq!(
            expression.simplify(&registry),
            NumericExpression::Constant(3.0)
        );
        let expression = NumericExpression::<Continuous>::ContinuousReduceSum(Box::new(
            NumericVectorExpression::Table2DX(
                0,
                VectorExpression::Reference(ReferenceExpression::Variable(0)),
                ElementExpression::Variable(0),
            ),
        ));
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn integer_reduce_product_simplify() {
        let registry = generate_registry();
        let expression = NumericExpression::<Integer>::IntegerReduceProduct(Box::new(
            NumericVectorExpression::Constant(vec![2, 1]),
        ));
        assert_eq!(
            expression.simplify(&registry),
            NumericExpression::Constant(2)
        );
        let expression = NumericExpression::<Integer>::IntegerReduceProduct(Box::new(
            NumericVectorExpression::Table2DX(
                0,
                VectorExpression::Reference(ReferenceExpression::Variable(0)),
                ElementExpression::Variable(0),
            ),
        ));
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn continuous_reduce_product_simplify() {
        let registry = generate_registry();
        let expression = NumericExpression::<Continuous>::ContinuousReduceProduct(Box::new(
            NumericVectorExpression::Constant(vec![2.0, 1.0]),
        ));
        assert_eq!(
            expression.simplify(&registry),
            NumericExpression::Constant(2.0)
        );
        let expression = NumericExpression::<Continuous>::ContinuousReduceProduct(Box::new(
            NumericVectorExpression::Table2DX(
                0,
                VectorExpression::Reference(ReferenceExpression::Variable(0)),
                ElementExpression::Variable(0),
            ),
        ));
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn vector_constant_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let tables = &registry.integer_tables;
        let expression = NumericVectorExpression::Constant(vec![0, 1]);
        assert_eq!(expression.eval(&state, &registry, tables), vec![0, 1]);
    }

    #[test]
    fn vector_reverse_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let tables = &registry.integer_tables;
        let expression =
            NumericVectorExpression::Reverse(Box::new(NumericVectorExpression::Constant(vec![
                0, 1,
            ])));
        assert_eq!(expression.eval(&state, &registry, tables), vec![1, 0]);
    }

    #[test]
    fn vector_push_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let tables = &registry.integer_tables;
        let expression = NumericVectorExpression::Push(
            NumericExpression::Constant(0),
            Box::new(NumericVectorExpression::Constant(vec![0, 1])),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![0, 1, 0]);
    }

    #[test]
    fn vector_pop_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let tables = &registry.integer_tables;
        let expression =
            NumericVectorExpression::Pop(Box::new(NumericVectorExpression::Constant(vec![0, 1])));
        assert_eq!(expression.eval(&state, &registry, tables), vec![0]);
    }

    #[test]
    fn vector_set_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let tables = &registry.integer_tables;
        let expression = NumericVectorExpression::Set(
            NumericExpression::Constant(0),
            Box::new(NumericVectorExpression::Constant(vec![0, 1])),
            ElementExpression::Constant(1),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![0, 0]);
    }

    #[test]
    fn vector_numeric_add_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let tables = &registry.integer_tables;
        let expression = NumericVectorExpression::NumericOperation(
            NumericOperator::Add,
            NumericExpression::Constant(2),
            Box::new(NumericVectorExpression::Constant(vec![0, 1])),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![2, 3]);
    }

    #[test]
    fn vector_numeric_subtract_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let tables = &registry.integer_tables;
        let expression = NumericVectorExpression::NumericOperation(
            NumericOperator::Subtract,
            NumericExpression::Constant(2),
            Box::new(NumericVectorExpression::Constant(vec![0, 1])),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![-2, -1]);
    }

    #[test]
    fn vector_numeric_multiply_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let tables = &registry.integer_tables;
        let expression = NumericVectorExpression::NumericOperation(
            NumericOperator::Multiply,
            NumericExpression::Constant(2),
            Box::new(NumericVectorExpression::Constant(vec![0, 1])),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![0, 2]);
    }

    #[test]
    fn vector_numeric_divide_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let tables = &registry.integer_tables;
        let expression = NumericVectorExpression::NumericOperation(
            NumericOperator::Divide,
            NumericExpression::Constant(2),
            Box::new(NumericVectorExpression::Constant(vec![0, 4])),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![0, 2]);
    }

    #[test]
    fn vector_numeric_max_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let tables = &registry.integer_tables;
        let expression = NumericVectorExpression::NumericOperation(
            NumericOperator::Max,
            NumericExpression::Constant(2),
            Box::new(NumericVectorExpression::Constant(vec![0, 4])),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![2, 4]);
    }

    #[test]
    fn vector_numeric_min_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let tables = &registry.integer_tables;
        let expression = NumericVectorExpression::NumericOperation(
            NumericOperator::Min,
            NumericExpression::Constant(2),
            Box::new(NumericVectorExpression::Constant(vec![0, 4])),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![0, 2]);
    }

    #[test]
    fn vector_add_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let tables = &registry.integer_tables;
        let expression = NumericVectorExpression::VectorOperation(
            NumericOperator::Add,
            Box::new(NumericVectorExpression::Constant(vec![0, 1, 3])),
            Box::new(NumericVectorExpression::Constant(vec![2, 3])),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![2, 4]);
        let expression = NumericVectorExpression::VectorOperation(
            NumericOperator::Add,
            Box::new(NumericVectorExpression::Reverse(Box::new(
                NumericVectorExpression::Constant(vec![0, 1, 3]),
            ))),
            Box::new(NumericVectorExpression::Constant(vec![2, 3])),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![5, 4]);
        let expression = NumericVectorExpression::VectorOperation(
            NumericOperator::Add,
            Box::new(NumericVectorExpression::Constant(vec![0, 1, 3])),
            Box::new(NumericVectorExpression::Reverse(Box::new(
                NumericVectorExpression::Constant(vec![2, 3]),
            ))),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![3, 3]);
        let expression = NumericVectorExpression::VectorOperation(
            NumericOperator::Add,
            Box::new(NumericVectorExpression::Reverse(Box::new(
                NumericVectorExpression::Constant(vec![0, 1, 3]),
            ))),
            Box::new(NumericVectorExpression::Reverse(Box::new(
                NumericVectorExpression::Constant(vec![2, 3]),
            ))),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![6, 3]);
    }

    #[test]
    fn vector_subtract_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let tables = &registry.integer_tables;
        let expression = NumericVectorExpression::VectorOperation(
            NumericOperator::Subtract,
            Box::new(NumericVectorExpression::Constant(vec![0, 1])),
            Box::new(NumericVectorExpression::Constant(vec![2, 3, 2])),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![-2, -2]);
        let expression = NumericVectorExpression::VectorOperation(
            NumericOperator::Subtract,
            Box::new(NumericVectorExpression::Reverse(Box::new(
                NumericVectorExpression::Constant(vec![0, 1, 3]),
            ))),
            Box::new(NumericVectorExpression::Constant(vec![2, 3])),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![1, -2]);
        let expression = NumericVectorExpression::VectorOperation(
            NumericOperator::Subtract,
            Box::new(NumericVectorExpression::Constant(vec![0, 1, 3])),
            Box::new(NumericVectorExpression::Reverse(Box::new(
                NumericVectorExpression::Constant(vec![2, 3]),
            ))),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![-3, -1]);
        let expression = NumericVectorExpression::VectorOperation(
            NumericOperator::Subtract,
            Box::new(NumericVectorExpression::Reverse(Box::new(
                NumericVectorExpression::Constant(vec![0, 1, 3]),
            ))),
            Box::new(NumericVectorExpression::Reverse(Box::new(
                NumericVectorExpression::Constant(vec![2, 3]),
            ))),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![0, -1]);
    }

    #[test]
    fn vector_multiply_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let tables = &registry.integer_tables;
        let expression = NumericVectorExpression::VectorOperation(
            NumericOperator::Multiply,
            Box::new(NumericVectorExpression::Constant(vec![0, 1, 5])),
            Box::new(NumericVectorExpression::Constant(vec![2, 3])),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![0, 3]);
        let expression = NumericVectorExpression::VectorOperation(
            NumericOperator::Multiply,
            Box::new(NumericVectorExpression::Reverse(Box::new(
                NumericVectorExpression::Constant(vec![0, 1, 3]),
            ))),
            Box::new(NumericVectorExpression::Constant(vec![2, 3])),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![6, 3]);
        let expression = NumericVectorExpression::VectorOperation(
            NumericOperator::Multiply,
            Box::new(NumericVectorExpression::Constant(vec![0, 1, 3])),
            Box::new(NumericVectorExpression::Reverse(Box::new(
                NumericVectorExpression::Constant(vec![2, 3]),
            ))),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![0, 2]);
        let expression = NumericVectorExpression::VectorOperation(
            NumericOperator::Multiply,
            Box::new(NumericVectorExpression::Reverse(Box::new(
                NumericVectorExpression::Constant(vec![0, 1, 3]),
            ))),
            Box::new(NumericVectorExpression::Reverse(Box::new(
                NumericVectorExpression::Constant(vec![2, 3]),
            ))),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![9, 2]);
    }

    #[test]
    fn vector_divide_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let tables = &registry.integer_tables;
        let expression = NumericVectorExpression::VectorOperation(
            NumericOperator::Divide,
            Box::new(NumericVectorExpression::Constant(vec![0, 6])),
            Box::new(NumericVectorExpression::Constant(vec![2, 3, 0])),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![0, 2]);
        let expression = NumericVectorExpression::VectorOperation(
            NumericOperator::Divide,
            Box::new(NumericVectorExpression::Reverse(Box::new(
                NumericVectorExpression::Constant(vec![0, 1, 3]),
            ))),
            Box::new(NumericVectorExpression::Constant(vec![2, 3])),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![1, 0]);
        let expression = NumericVectorExpression::VectorOperation(
            NumericOperator::Divide,
            Box::new(NumericVectorExpression::Constant(vec![0, 1, 3])),
            Box::new(NumericVectorExpression::Reverse(Box::new(
                NumericVectorExpression::Constant(vec![2, 3]),
            ))),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![0, 0]);
        let expression = NumericVectorExpression::VectorOperation(
            NumericOperator::Divide,
            Box::new(NumericVectorExpression::Reverse(Box::new(
                NumericVectorExpression::Constant(vec![0, 1, 3]),
            ))),
            Box::new(NumericVectorExpression::Reverse(Box::new(
                NumericVectorExpression::Constant(vec![2, 3]),
            ))),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![1, 0]);
    }

    #[test]
    fn vector_max_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let tables = &registry.integer_tables;
        let expression = NumericVectorExpression::VectorOperation(
            NumericOperator::Max,
            Box::new(NumericVectorExpression::Constant(vec![0, 6, 9])),
            Box::new(NumericVectorExpression::Constant(vec![2, 3])),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![2, 6]);
        let expression = NumericVectorExpression::VectorOperation(
            NumericOperator::Max,
            Box::new(NumericVectorExpression::Reverse(Box::new(
                NumericVectorExpression::Constant(vec![0, 1, 3]),
            ))),
            Box::new(NumericVectorExpression::Constant(vec![2, 3])),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![3, 3]);
        let expression = NumericVectorExpression::VectorOperation(
            NumericOperator::Max,
            Box::new(NumericVectorExpression::Constant(vec![0, 1, 3])),
            Box::new(NumericVectorExpression::Reverse(Box::new(
                NumericVectorExpression::Constant(vec![2, 3]),
            ))),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![3, 2]);
        let expression = NumericVectorExpression::VectorOperation(
            NumericOperator::Max,
            Box::new(NumericVectorExpression::Reverse(Box::new(
                NumericVectorExpression::Constant(vec![0, 1, 3]),
            ))),
            Box::new(NumericVectorExpression::Reverse(Box::new(
                NumericVectorExpression::Constant(vec![2, 3]),
            ))),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![3, 2]);
    }

    #[test]
    fn vector_min_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let tables = &registry.integer_tables;
        let expression = NumericVectorExpression::VectorOperation(
            NumericOperator::Min,
            Box::new(NumericVectorExpression::Constant(vec![0, 6])),
            Box::new(NumericVectorExpression::Constant(vec![2, 3, 2])),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![0, 3]);
        let expression = NumericVectorExpression::VectorOperation(
            NumericOperator::Min,
            Box::new(NumericVectorExpression::Reverse(Box::new(
                NumericVectorExpression::Constant(vec![0, 1, 3]),
            ))),
            Box::new(NumericVectorExpression::Constant(vec![2, 3])),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![2, 1]);
        let expression = NumericVectorExpression::VectorOperation(
            NumericOperator::Min,
            Box::new(NumericVectorExpression::Constant(vec![0, 1, 3])),
            Box::new(NumericVectorExpression::Reverse(Box::new(
                NumericVectorExpression::Constant(vec![2, 3]),
            ))),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![0, 1]);
        let expression = NumericVectorExpression::VectorOperation(
            NumericOperator::Min,
            Box::new(NumericVectorExpression::Reverse(Box::new(
                NumericVectorExpression::Constant(vec![0, 1, 3]),
            ))),
            Box::new(NumericVectorExpression::Reverse(Box::new(
                NumericVectorExpression::Constant(vec![2, 3]),
            ))),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![3, 1]);
    }

    #[test]
    fn vector_eval_cost() {
        let registry = generate_registry();
        let state = generate_state();
        let tables = &registry.integer_tables;
        let expression = NumericVectorExpression::NumericOperation(
            NumericOperator::Add,
            NumericExpression::Cost,
            Box::new(NumericVectorExpression::Constant(vec![2, 3])),
        );
        assert_eq!(
            expression.eval_cost(1, &state, &registry, tables),
            vec![3, 4]
        );
    }

    #[test]
    #[should_panic]
    fn vector_eval_cost_panic() {
        let registry = generate_registry();
        let state = generate_state();
        let tables = &registry.integer_tables;
        let expression = NumericVectorExpression::NumericOperation(
            NumericOperator::Add,
            NumericExpression::Cost,
            Box::new(NumericVectorExpression::Constant(vec![2, 3])),
        );
        expression.eval(&state, &registry, tables);
    }

    #[test]
    fn vector_table_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let tables = &registry.integer_tables;
        let expression = NumericVectorExpression::Table(
            0,
            vec![
                VectorOrElementExpression::Element(ElementExpression::Constant(0)),
                VectorOrElementExpression::Element(ElementExpression::Constant(1)),
                VectorOrElementExpression::Vector(VectorExpression::Reference(
                    ReferenceExpression::Constant(vec![0, 2]),
                )),
                VectorOrElementExpression::Element(ElementExpression::Constant(0)),
            ],
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![100, 300]);
    }

    #[test]
    fn vector_table_sum_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let tables = &registry.integer_tables;
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(1);
        let expression = NumericVectorExpression::TableSum(
            0,
            vec![
                ArgumentExpression::Element(ElementExpression::Constant(0)),
                ArgumentExpression::Element(ElementExpression::Constant(1)),
                ArgumentExpression::Vector(VectorExpression::Reference(
                    ReferenceExpression::Constant(vec![0, 2]),
                )),
                ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Constant(
                    set,
                ))),
            ],
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![300, 700]);
        let mut set = Set::with_capacity(3);
        set.insert(2);
        let expression = NumericVectorExpression::TableSum(
            0,
            vec![
                ArgumentExpression::Element(ElementExpression::Constant(0)),
                ArgumentExpression::Element(ElementExpression::Constant(1)),
                ArgumentExpression::Vector(VectorExpression::Reverse(Box::new(
                    VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 2])),
                ))),
                ArgumentExpression::Set(SetExpression::Complement(Box::new(
                    SetExpression::Reference(ReferenceExpression::Constant(set)),
                ))),
            ],
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![700, 300]);
    }

    #[test]
    fn vector_table_1d_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let tables = &registry.integer_tables;
        let expression = NumericVectorExpression::Table1D(
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![10, 20]);
        let expression = NumericVectorExpression::Table1D(
            0,
            VectorExpression::Reverse(Box::new(VectorExpression::Reference(
                ReferenceExpression::Constant(vec![0, 1]),
            ))),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![20, 10]);
    }

    #[test]
    fn vector_table_2d_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let tables = &registry.integer_tables;
        let expression = NumericVectorExpression::Table2D(
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![10, 50]);
        let expression = NumericVectorExpression::Table2D(
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            VectorExpression::Reverse(Box::new(VectorExpression::Reference(
                ReferenceExpression::Constant(vec![0, 1]),
            ))),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![20, 40]);
        let expression = NumericVectorExpression::Table2D(
            0,
            VectorExpression::Reverse(Box::new(VectorExpression::Reference(
                ReferenceExpression::Constant(vec![0, 1]),
            ))),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![40, 20]);
        let expression = NumericVectorExpression::Table2D(
            0,
            VectorExpression::Reverse(Box::new(VectorExpression::Reference(
                ReferenceExpression::Constant(vec![0, 1]),
            ))),
            VectorExpression::Reverse(Box::new(VectorExpression::Reference(
                ReferenceExpression::Constant(vec![0, 1]),
            ))),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![50, 10]);
    }

    #[test]
    fn vector_table_2d_x_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let tables = &registry.integer_tables;
        let expression = NumericVectorExpression::Table2DX(
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            ElementExpression::Constant(0),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![10, 40]);
        let expression = NumericVectorExpression::Table2DX(
            0,
            VectorExpression::Reverse(Box::new(VectorExpression::Reference(
                ReferenceExpression::Constant(vec![0, 1]),
            ))),
            ElementExpression::Constant(0),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![40, 10]);
    }

    #[test]
    fn vector_table_2d_y_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let tables = &registry.integer_tables;
        let expression = NumericVectorExpression::Table2DY(
            0,
            ElementExpression::Constant(0),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![10, 20]);
        let expression = NumericVectorExpression::Table2DY(
            0,
            ElementExpression::Constant(0),
            VectorExpression::Reverse(Box::new(VectorExpression::Reference(
                ReferenceExpression::Constant(vec![0, 1]),
            ))),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![20, 10]);
    }

    #[test]
    fn vector_table_2d_x_sum_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let tables = &registry.integer_tables;
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(1);
        let expression = NumericVectorExpression::Table2DXSum(
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            SetExpression::Reference(ReferenceExpression::Constant(set.clone())),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![30, 90]);
        let expression = NumericVectorExpression::Table2DXSum(
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            SetExpression::Complement(Box::new(SetExpression::Reference(
                ReferenceExpression::Constant(set.clone()),
            ))),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![30, 60]);
        let expression = NumericVectorExpression::Table2DXSum(
            0,
            VectorExpression::Reverse(Box::new(VectorExpression::Reference(
                ReferenceExpression::Constant(vec![0, 1]),
            ))),
            SetExpression::Reference(ReferenceExpression::Constant(set.clone())),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![90, 30]);
        let expression = NumericVectorExpression::Table2DXSum(
            0,
            VectorExpression::Reverse(Box::new(VectorExpression::Reference(
                ReferenceExpression::Constant(vec![0, 1]),
            ))),
            SetExpression::Complement(Box::new(SetExpression::Reference(
                ReferenceExpression::Constant(set),
            ))),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![60, 30]);
    }

    #[test]
    fn vector_table_2d_y_sum_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let tables = &registry.integer_tables;
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(1);
        let expression = NumericVectorExpression::Table2DYSum(
            0,
            SetExpression::Reference(ReferenceExpression::Constant(set.clone())),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![50, 70]);
        let expression = NumericVectorExpression::Table2DYSum(
            0,
            SetExpression::Complement(Box::new(SetExpression::Reference(
                ReferenceExpression::Constant(set.clone()),
            ))),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![70, 80]);
        let expression = NumericVectorExpression::Table2DYSum(
            0,
            SetExpression::Reference(ReferenceExpression::Constant(set.clone())),
            VectorExpression::Reverse(Box::new(VectorExpression::Reference(
                ReferenceExpression::Constant(vec![0, 1]),
            ))),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![70, 50]);
        let expression = NumericVectorExpression::Table2DYSum(
            0,
            SetExpression::Complement(Box::new(SetExpression::Reference(
                ReferenceExpression::Constant(set),
            ))),
            VectorExpression::Reverse(Box::new(VectorExpression::Reference(
                ReferenceExpression::Constant(vec![0, 1]),
            ))),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![80, 70]);
    }

    #[test]
    fn vector_constant_simplify() {
        let registry = generate_registry();
        let tables = &registry.integer_tables;
        let expression = NumericVectorExpression::Constant(vec![0, 1]);
        assert_eq!(expression.simplify(&registry, tables), expression);
    }

    #[test]
    fn vector_reverse_simplify() {
        let registry = generate_registry();
        let tables = &registry.integer_tables;
        let expression =
            NumericVectorExpression::Reverse(Box::new(NumericVectorExpression::Constant(vec![
                0, 1,
            ])));
        assert_eq!(
            expression.simplify(&registry, tables),
            NumericVectorExpression::Constant(vec![1, 0])
        );
        let expression =
            NumericVectorExpression::Reverse(Box::new(NumericVectorExpression::Table1D(
                0,
                VectorExpression::Reference(ReferenceExpression::Variable(0)),
            )));
        assert_eq!(expression.simplify(&registry, tables), expression);
    }

    #[test]
    fn vector_push_simplify() {
        let registry = generate_registry();
        let tables = &registry.integer_tables;
        let expression = NumericVectorExpression::Push(
            NumericExpression::Constant(0),
            Box::new(NumericVectorExpression::Constant(vec![0, 1])),
        );
        assert_eq!(
            expression.simplify(&registry, tables),
            NumericVectorExpression::Constant(vec![0, 1, 0])
        );
        let expression = NumericVectorExpression::Push(
            NumericExpression::Constant(0),
            Box::new(NumericVectorExpression::Table1D(
                0,
                VectorExpression::Reference(ReferenceExpression::Variable(0)),
            )),
        );
        assert_eq!(expression.simplify(&registry, tables), expression);
    }

    #[test]
    fn vector_pop_simplify() {
        let registry = generate_registry();
        let tables = &registry.integer_tables;
        let expression =
            NumericVectorExpression::Pop(Box::new(NumericVectorExpression::Constant(vec![0, 1])));
        assert_eq!(
            expression.simplify(&registry, tables),
            NumericVectorExpression::Constant(vec![0])
        );
        let expression = NumericVectorExpression::Pop(Box::new(NumericVectorExpression::Table1D(
            0,
            VectorExpression::Reference(ReferenceExpression::Variable(0)),
        )));
        assert_eq!(expression.simplify(&registry, tables), expression);
    }

    #[test]
    fn vector_set_simplify() {
        let registry = generate_registry();
        let tables = &registry.integer_tables;
        let expression = NumericVectorExpression::Set(
            NumericExpression::Constant(0),
            Box::new(NumericVectorExpression::Constant(vec![0, 1])),
            ElementExpression::Constant(1),
        );
        assert_eq!(
            expression.simplify(&registry, tables),
            NumericVectorExpression::Constant(vec![0, 0])
        );
        let expression = NumericVectorExpression::Set(
            NumericExpression::Constant(0),
            Box::new(NumericVectorExpression::Constant(vec![0, 1])),
            ElementExpression::Variable(1),
        );
        assert_eq!(expression.simplify(&registry, tables), expression);
    }

    #[test]
    fn vector_numeric_add_simplify() {
        let registry = generate_registry();
        let tables = &registry.integer_tables;
        let expression = NumericVectorExpression::NumericOperation(
            NumericOperator::Add,
            NumericExpression::Constant(2),
            Box::new(NumericVectorExpression::Constant(vec![0, 1])),
        );
        assert_eq!(
            expression.simplify(&registry, tables),
            NumericVectorExpression::Constant(vec![2, 3])
        );
        let expression = NumericVectorExpression::NumericOperation(
            NumericOperator::Add,
            NumericExpression::IntegerVariable(0),
            Box::new(NumericVectorExpression::Constant(vec![0, 1])),
        );
        assert_eq!(expression.simplify(&registry, tables), expression);
    }

    #[test]
    fn vector_numeric_subtract_simplify() {
        let registry = generate_registry();
        let tables = &registry.integer_tables;
        let expression = NumericVectorExpression::NumericOperation(
            NumericOperator::Subtract,
            NumericExpression::Constant(2),
            Box::new(NumericVectorExpression::Constant(vec![0, 1])),
        );
        assert_eq!(
            expression.simplify(&registry, tables),
            NumericVectorExpression::Constant(vec![-2, -1])
        );
        let expression = NumericVectorExpression::NumericOperation(
            NumericOperator::Subtract,
            NumericExpression::IntegerVariable(0),
            Box::new(NumericVectorExpression::Constant(vec![0, 1])),
        );
        assert_eq!(expression.simplify(&registry, tables), expression);
    }

    #[test]
    fn vector_numeric_multiply_simplify() {
        let registry = generate_registry();
        let tables = &registry.integer_tables;
        let expression = NumericVectorExpression::NumericOperation(
            NumericOperator::Multiply,
            NumericExpression::Constant(2),
            Box::new(NumericVectorExpression::Constant(vec![0, 1])),
        );
        assert_eq!(
            expression.simplify(&registry, tables),
            NumericVectorExpression::Constant(vec![0, 2])
        );
        let expression = NumericVectorExpression::NumericOperation(
            NumericOperator::Multiply,
            NumericExpression::IntegerVariable(0),
            Box::new(NumericVectorExpression::Constant(vec![0, 1])),
        );
        assert_eq!(expression.simplify(&registry, tables), expression);
    }

    #[test]
    fn vector_numeric_divide_simplify() {
        let registry = generate_registry();
        let tables = &registry.integer_tables;
        let expression = NumericVectorExpression::NumericOperation(
            NumericOperator::Divide,
            NumericExpression::Constant(2),
            Box::new(NumericVectorExpression::Constant(vec![0, 4])),
        );
        assert_eq!(
            expression.simplify(&registry, tables),
            NumericVectorExpression::Constant(vec![0, 2])
        );
        let expression = NumericVectorExpression::NumericOperation(
            NumericOperator::Divide,
            NumericExpression::IntegerVariable(0),
            Box::new(NumericVectorExpression::Constant(vec![0, 1])),
        );
        assert_eq!(expression.simplify(&registry, tables), expression);
    }

    #[test]
    fn vector_numeric_max_simplify() {
        let registry = generate_registry();
        let tables = &registry.integer_tables;
        let expression = NumericVectorExpression::NumericOperation(
            NumericOperator::Max,
            NumericExpression::Constant(2),
            Box::new(NumericVectorExpression::Constant(vec![0, 4])),
        );
        assert_eq!(
            expression.simplify(&registry, tables),
            NumericVectorExpression::Constant(vec![2, 4])
        );
        let expression = NumericVectorExpression::NumericOperation(
            NumericOperator::Max,
            NumericExpression::IntegerVariable(0),
            Box::new(NumericVectorExpression::Constant(vec![0, 1])),
        );
        assert_eq!(expression.simplify(&registry, tables), expression);
    }

    #[test]
    fn vector_numeric_min_simplify() {
        let registry = generate_registry();
        let tables = &registry.integer_tables;
        let expression = NumericVectorExpression::NumericOperation(
            NumericOperator::Min,
            NumericExpression::Constant(2),
            Box::new(NumericVectorExpression::Constant(vec![0, 4])),
        );
        assert_eq!(
            expression.simplify(&registry, tables),
            NumericVectorExpression::Constant(vec![0, 2])
        );
        let expression = NumericVectorExpression::NumericOperation(
            NumericOperator::Min,
            NumericExpression::IntegerVariable(0),
            Box::new(NumericVectorExpression::Constant(vec![0, 1])),
        );
        assert_eq!(expression.simplify(&registry, tables), expression);
    }

    #[test]
    fn vector_add_simplify() {
        let registry = generate_registry();
        let tables = &registry.integer_tables;
        let expression = NumericVectorExpression::VectorOperation(
            NumericOperator::Add,
            Box::new(NumericVectorExpression::Constant(vec![0, 1])),
            Box::new(NumericVectorExpression::Constant(vec![2, 3])),
        );
        assert_eq!(
            expression.simplify(&registry, tables),
            NumericVectorExpression::Constant(vec![2, 4])
        );
        let expression = NumericVectorExpression::VectorOperation(
            NumericOperator::Add,
            Box::new(NumericVectorExpression::Table1D(
                0,
                VectorExpression::Reference(ReferenceExpression::Variable(0)),
            )),
            Box::new(NumericVectorExpression::Constant(vec![2, 3])),
        );
        assert_eq!(expression.simplify(&registry, tables), expression);
    }

    #[test]
    fn vector_subtract_simplify() {
        let registry = generate_registry();
        let tables = &registry.integer_tables;
        let expression = NumericVectorExpression::VectorOperation(
            NumericOperator::Subtract,
            Box::new(NumericVectorExpression::Constant(vec![0, 1])),
            Box::new(NumericVectorExpression::Constant(vec![2, 3])),
        );
        assert_eq!(
            expression.simplify(&registry, tables),
            NumericVectorExpression::Constant(vec![-2, -2])
        );
        let expression = NumericVectorExpression::VectorOperation(
            NumericOperator::Subtract,
            Box::new(NumericVectorExpression::Table1D(
                0,
                VectorExpression::Reference(ReferenceExpression::Variable(0)),
            )),
            Box::new(NumericVectorExpression::Constant(vec![2, 3])),
        );
        assert_eq!(expression.simplify(&registry, tables), expression);
    }

    #[test]
    fn vector_multiply_simplify() {
        let registry = generate_registry();
        let tables = &registry.integer_tables;
        let expression = NumericVectorExpression::VectorOperation(
            NumericOperator::Multiply,
            Box::new(NumericVectorExpression::Constant(vec![0, 1])),
            Box::new(NumericVectorExpression::Constant(vec![2, 3])),
        );
        assert_eq!(
            expression.simplify(&registry, tables),
            NumericVectorExpression::Constant(vec![0, 3])
        );
        let expression = NumericVectorExpression::VectorOperation(
            NumericOperator::Multiply,
            Box::new(NumericVectorExpression::Table1D(
                0,
                VectorExpression::Reference(ReferenceExpression::Variable(0)),
            )),
            Box::new(NumericVectorExpression::Constant(vec![2, 3])),
        );
        assert_eq!(expression.simplify(&registry, tables), expression);
    }

    #[test]
    fn vector_divide_simplify() {
        let registry = generate_registry();
        let tables = &registry.integer_tables;
        let expression = NumericVectorExpression::VectorOperation(
            NumericOperator::Divide,
            Box::new(NumericVectorExpression::Constant(vec![0, 6])),
            Box::new(NumericVectorExpression::Constant(vec![2, 3])),
        );
        assert_eq!(
            expression.simplify(&registry, tables),
            NumericVectorExpression::Constant(vec![0, 2])
        );
        let expression = NumericVectorExpression::VectorOperation(
            NumericOperator::Divide,
            Box::new(NumericVectorExpression::Table1D(
                0,
                VectorExpression::Reference(ReferenceExpression::Variable(0)),
            )),
            Box::new(NumericVectorExpression::Constant(vec![2, 3])),
        );
        assert_eq!(expression.simplify(&registry, tables), expression);
    }

    #[test]
    fn vector_max_simplify() {
        let registry = generate_registry();
        let tables = &registry.integer_tables;
        let expression = NumericVectorExpression::VectorOperation(
            NumericOperator::Max,
            Box::new(NumericVectorExpression::Constant(vec![0, 6])),
            Box::new(NumericVectorExpression::Constant(vec![2, 3])),
        );
        assert_eq!(
            expression.simplify(&registry, tables),
            NumericVectorExpression::Constant(vec![2, 6])
        );
        let expression = NumericVectorExpression::VectorOperation(
            NumericOperator::Max,
            Box::new(NumericVectorExpression::Table1D(
                0,
                VectorExpression::Reference(ReferenceExpression::Variable(0)),
            )),
            Box::new(NumericVectorExpression::Constant(vec![2, 3])),
        );
        assert_eq!(expression.simplify(&registry, tables), expression);
    }

    #[test]
    fn vector_min_simplify() {
        let registry = generate_registry();
        let tables = &registry.integer_tables;
        let expression = NumericVectorExpression::VectorOperation(
            NumericOperator::Min,
            Box::new(NumericVectorExpression::Constant(vec![0, 6])),
            Box::new(NumericVectorExpression::Constant(vec![2, 3])),
        );
        assert_eq!(
            expression.simplify(&registry, tables),
            NumericVectorExpression::Constant(vec![0, 3])
        );
        let expression = NumericVectorExpression::VectorOperation(
            NumericOperator::Min,
            Box::new(NumericVectorExpression::Table1D(
                0,
                VectorExpression::Reference(ReferenceExpression::Variable(0)),
            )),
            Box::new(NumericVectorExpression::Constant(vec![2, 3])),
        );
        assert_eq!(expression.simplify(&registry, tables), expression);
    }

    #[test]
    fn vector_table_simplify() {
        let registry = generate_registry();
        let tables = &registry.integer_tables;
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(1);
        let expression = NumericVectorExpression::Table(
            0,
            vec![
                VectorOrElementExpression::Element(ElementExpression::Constant(0)),
                VectorOrElementExpression::Element(ElementExpression::Constant(1)),
                VectorOrElementExpression::Vector(VectorExpression::Reference(
                    ReferenceExpression::Constant(vec![0, 2]),
                )),
                VectorOrElementExpression::Element(ElementExpression::Constant(0)),
            ],
        );
        assert_eq!(
            expression.simplify(&registry, tables),
            NumericVectorExpression::Constant(vec![100, 300])
        );
        let mut set = Set::with_capacity(3);
        set.insert(2);
        let expression = NumericVectorExpression::Table(
            0,
            vec![
                VectorOrElementExpression::Element(ElementExpression::Variable(0)),
                VectorOrElementExpression::Element(ElementExpression::Constant(1)),
                VectorOrElementExpression::Vector(VectorExpression::Reference(
                    ReferenceExpression::Constant(vec![0, 2]),
                )),
                VectorOrElementExpression::Element(ElementExpression::Variable(0)),
            ],
        );
        assert_eq!(expression.simplify(&registry, tables), expression);
    }

    #[test]
    fn vector_table_sum_simplify() {
        let registry = generate_registry();
        let tables = &registry.integer_tables;
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(1);
        let expression = NumericVectorExpression::TableSum(
            0,
            vec![
                ArgumentExpression::Element(ElementExpression::Constant(0)),
                ArgumentExpression::Element(ElementExpression::Constant(1)),
                ArgumentExpression::Vector(VectorExpression::Reference(
                    ReferenceExpression::Constant(vec![0, 2]),
                )),
                ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Constant(
                    set,
                ))),
            ],
        );
        assert_eq!(
            expression.simplify(&registry, tables),
            NumericVectorExpression::Constant(vec![300, 700])
        );
        let mut set = Set::with_capacity(3);
        set.insert(2);
        let expression = NumericVectorExpression::TableSum(
            0,
            vec![
                ArgumentExpression::Element(ElementExpression::Variable(0)),
                ArgumentExpression::Element(ElementExpression::Constant(1)),
                ArgumentExpression::Vector(VectorExpression::Reference(
                    ReferenceExpression::Constant(vec![0, 2]),
                )),
                ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Constant(
                    set,
                ))),
            ],
        );
        assert_eq!(expression.simplify(&registry, tables), expression);
    }

    #[test]
    fn vector_table_1d_simplify() {
        let registry = generate_registry();
        let tables = &registry.integer_tables;
        let expression = NumericVectorExpression::Table1D(
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert_eq!(
            expression.simplify(&registry, tables),
            NumericVectorExpression::Constant(vec![10, 20])
        );
        let expression = NumericVectorExpression::Table1D(
            0,
            VectorExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert_eq!(expression.simplify(&registry, tables), expression);
    }

    #[test]
    fn vector_table_2d_simplify() {
        let registry = generate_registry();
        let tables = &registry.integer_tables;
        let expression = NumericVectorExpression::Table2D(
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert_eq!(
            expression.simplify(&registry, tables),
            NumericVectorExpression::Constant(vec![10, 50])
        );
        let expression = NumericVectorExpression::Table2D(
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            VectorExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert_eq!(expression.simplify(&registry, tables), expression);
    }

    #[test]
    fn vector_table_2d_x_simplify() {
        let registry = generate_registry();
        let tables = &registry.integer_tables;
        let expression = NumericVectorExpression::Table2DX(
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            ElementExpression::Constant(0),
        );
        assert_eq!(
            expression.simplify(&registry, tables),
            NumericVectorExpression::Constant(vec![10, 40])
        );
        let expression = NumericVectorExpression::Table2DX(
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            ElementExpression::Variable(0),
        );
        assert_eq!(expression.simplify(&registry, tables), expression);
    }

    #[test]
    fn vector_table_2d_y_simplify() {
        let registry = generate_registry();
        let tables = &registry.integer_tables;
        let expression = NumericVectorExpression::Table2DY(
            0,
            ElementExpression::Constant(0),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert_eq!(
            expression.simplify(&registry, tables),
            NumericVectorExpression::Constant(vec![10, 20])
        );
        let expression = NumericVectorExpression::Table2DY(
            0,
            ElementExpression::Variable(0),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert_eq!(expression.simplify(&registry, tables), expression);
    }

    #[test]
    fn vector_table_2d_x_sum_simplify() {
        let registry = generate_registry();
        let tables = &registry.integer_tables;
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(1);
        let expression = NumericVectorExpression::Table2DXSum(
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            SetExpression::Reference(ReferenceExpression::Constant(set)),
        );
        assert_eq!(
            expression.simplify(&registry, tables),
            NumericVectorExpression::Constant(vec![30, 90])
        );
        let expression = NumericVectorExpression::Table2DXSum(
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert_eq!(expression.simplify(&registry, tables), expression);
    }

    #[test]
    fn vector_table_2d_y_sum_simplify() {
        let registry = generate_registry();
        let tables = &registry.integer_tables;
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(1);
        let expression = NumericVectorExpression::Table2DYSum(
            0,
            SetExpression::Reference(ReferenceExpression::Constant(set)),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert_eq!(
            expression.simplify(&registry, tables),
            NumericVectorExpression::Constant(vec![50, 70])
        );
        let expression = NumericVectorExpression::Table2DYSum(
            0,
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert_eq!(expression.simplify(&registry, tables), expression);
    }
}
