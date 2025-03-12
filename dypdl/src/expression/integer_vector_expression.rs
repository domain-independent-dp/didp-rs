use super::condition::Condition;
use super::continuous_vector_expression::ContinuousVectorExpression;
use super::element_expression::ElementExpression;
use super::integer_expression::IntegerExpression;
use super::numeric_operator::{BinaryOperator, CastOperator, UnaryOperator};
use super::table_vector_expression::TableVectorExpression;
use crate::state_functions::{StateFunctionCache, StateFunctions};
use crate::state::StateInterface;
use crate::table_registry::TableRegistry;
use crate::variable_type::{Continuous, Integer};
use std::boxed::Box;

/// Expression representing a vector of numeric values with the integer value type.
#[derive(Debug, PartialEq, Clone)]
pub enum IntegerVectorExpression {
    /// Constant.
    Constant(Vec<Integer>),
    /// Reverses an integer vector.
    Reverse(Box<IntegerVectorExpression>),
    /// Pushes an element to an integer vector.
    Push(IntegerExpression, Box<IntegerVectorExpression>),
    /// Pops an element from an integer vector.
    Pop(Box<IntegerVectorExpression>),
    /// Sets an element in an integer vector.
    Set(
        IntegerExpression,
        Box<IntegerVectorExpression>,
        ElementExpression,
    ),
    /// Elementwise unary arithmetic operation.
    UnaryOperation(UnaryOperator, Box<IntegerVectorExpression>),
    /// Elementwise binary arithmetic operation.
    BinaryOperationX(
        BinaryOperator,
        IntegerExpression,
        Box<IntegerVectorExpression>,
    ),
    /// Elementwise binary arithmetic operation.
    BinaryOperationY(
        BinaryOperator,
        Box<IntegerVectorExpression>,
        IntegerExpression,
    ),
    /// Elementwise binary arithmetic operation.
    VectorOperation(
        BinaryOperator,
        Box<IntegerVectorExpression>,
        Box<IntegerVectorExpression>,
    ),
    /// A vector of constants in an integer table.
    Table(Box<TableVectorExpression<Integer>>),
    /// If-then-else expression, which returns the first one if the condition holds and the second one otherwise.
    If(
        Box<Condition>,
        Box<IntegerVectorExpression>,
        Box<IntegerVectorExpression>,
    ),
    /// Conversion from a continuous vector.
    FromContinuous(CastOperator, Box<ContinuousVectorExpression>),
}

impl IntegerVectorExpression {
    /// Returns the evaluation result.
    ///
    /// # Panics
    ///
    /// Panics if the cost of the transition state is used or a min/max reduce operation is performed on an empty set or vector.
    pub fn eval<U: StateInterface>(
        &self,
        state: &U,
        function_cache: &mut StateFunctionCache,
        state_functions: &StateFunctions,
        registry: &TableRegistry,
    ) -> Vec<Integer> {
        self.eval_inner(None, state, function_cache, state_functions, registry)
    }

    /// Returns the evaluation result of a cost expression.
    ///
    /// # Panics
    ///
    /// Panics if a min/max reduce operation is performed on an empty set or vector.
    pub fn eval_cost<U: StateInterface>(
        &self,
        cost: Integer,
        state: &U,
        function_cache: &mut StateFunctionCache,
        state_functions: &StateFunctions,
        registry: &TableRegistry,
    ) -> Vec<Integer> {
        self.eval_inner(
            Some(cost),
            state,
            function_cache,
            state_functions,
            registry,
        )
    }

    pub fn eval_inner<U: StateInterface>(
        &self,
        cost: Option<Integer>,
        state: &U,
        function_cache: &mut StateFunctionCache,
        state_functions: &StateFunctions,
        registry: &TableRegistry,
    ) -> Vec<Integer> {
        match self {
            Self::Constant(vector) => vector.clone(),
            Self::Reverse(vector) => {
                let mut vector =
                    vector.eval_inner(cost, state, function_cache, state_functions, registry);
                vector.reverse();
                vector
            }
            Self::Push(value, vector) => {
                let mut vector =
                    vector.eval_inner(cost, state, function_cache, state_functions, registry);
                vector.push(value.eval(state, function_cache, state_functions, registry));
                vector
            }
            Self::Pop(vector) => {
                let mut vector =
                    vector.eval_inner(cost, state, function_cache, state_functions, registry);
                vector.pop();
                vector
            }
            Self::Set(value, vector, i) => {
                let mut vector =
                    vector.eval_inner(cost, state, function_cache, state_functions, registry);
                vector[i.eval(state, function_cache, state_functions, registry)] =
                    value.eval(state, function_cache, state_functions, registry);
                vector
            }
            Self::UnaryOperation(op, x) => op.eval_vector(x.eval_inner(
                cost,
                state,
                function_cache,
                state_functions,
                registry,
            )),
            Self::BinaryOperationX(op, x, y) => op.eval_operation_x(
                if let Some(cost) = cost {
                    x.eval_cost(cost, state, function_cache, state_functions, registry)
                } else {
                    x.eval(state, function_cache, state_functions, registry)
                },
                y.eval_inner(cost, state, function_cache, state_functions, registry),
            ),
            Self::BinaryOperationY(op, x, y) => op.eval_operation_y(
                x.eval_inner(cost, state, function_cache, state_functions, registry),
                if let Some(cost) = cost {
                    y.eval_cost(cost, state, function_cache, state_functions, registry)
                } else {
                    y.eval(state, function_cache, state_functions, registry)
                },
            ),
            Self::VectorOperation(op, x, y) => match (x.as_ref(), y.as_ref()) {
                (Self::Constant(x), y) => op.eval_vector_operation_in_y(
                    x,
                    y.eval_inner(cost, state, function_cache, state_functions, registry),
                ),
                (x, Self::Constant(y)) => op.eval_vector_operation_in_x(
                    x.eval_inner(cost, state, function_cache, state_functions, registry),
                    y,
                ),
                (x, y) => op.eval_vector_operation_in_y(
                    &x.eval_inner(cost, state, function_cache, state_functions, registry),
                    y.eval_inner(cost, state, function_cache, state_functions, registry),
                ),
            },
            Self::Table(expression) => expression.eval(
                state,
                function_cache,
                state_functions,
                registry,
                &registry.integer_tables,
            ),
            Self::If(condition, x, y) => {
                if condition.eval(state, function_cache, state_functions, registry) {
                    x.eval_inner(cost, state, function_cache, state_functions, registry)
                } else {
                    y.eval_inner(cost, state, function_cache, state_functions, registry)
                }
            }
            Self::FromContinuous(op, x) => op
                .eval_vector(x.eval_inner(
                    cost.map(Continuous::from),
                    state,
                    function_cache,
                    state_functions,
                    registry,
                ))
                .into_iter()
                .map(|x| x as Integer)
                .collect(),
        }
    }

    /// Returns a simplified version by precomputation.
    pub fn simplify(&self, registry: &TableRegistry) -> IntegerVectorExpression {
        match self {
            Self::Reverse(vector) => match vector.simplify(registry) {
                Self::Constant(mut vector) => {
                    vector.reverse();
                    Self::Constant(vector)
                }
                vector => Self::Reverse(Box::new(vector)),
            },
            Self::Push(value, vector) => {
                match (value.simplify(registry), vector.simplify(registry)) {
                    (IntegerExpression::Constant(value), Self::Constant(mut vector)) => {
                        vector.push(value);
                        Self::Constant(vector)
                    }
                    (value, vector) => Self::Push(value, Box::new(vector)),
                }
            }
            Self::Pop(vector) => match vector.simplify(registry) {
                Self::Constant(mut vector) => {
                    vector.pop();
                    Self::Constant(vector)
                }
                vector => Self::Pop(Box::new(vector)),
            },
            Self::Set(value, vector, i) => match (
                value.simplify(registry),
                vector.simplify(registry),
                i.simplify(registry),
            ) {
                (
                    IntegerExpression::Constant(value),
                    Self::Constant(mut vector),
                    ElementExpression::Constant(i),
                ) => {
                    vector[i] = value;
                    Self::Constant(vector)
                }
                (value, vector, i) => Self::Set(value, Box::new(vector), i),
            },
            Self::UnaryOperation(op, x) => match x.simplify(registry) {
                IntegerVectorExpression::Constant(x) => Self::Constant(op.eval_vector(x)),
                x => Self::UnaryOperation(op.clone(), Box::new(x)),
            },
            Self::BinaryOperationX(op, x, y) => {
                match (x.simplify(registry), y.simplify(registry)) {
                    (IntegerExpression::Constant(x), Self::Constant(y)) => {
                        Self::Constant(op.eval_operation_x(x, y))
                    }
                    (x, y) => Self::BinaryOperationX(op.clone(), x, Box::new(y)),
                }
            }
            Self::BinaryOperationY(op, x, y) => {
                match (x.simplify(registry), y.simplify(registry)) {
                    (Self::Constant(x), IntegerExpression::Constant(y)) => {
                        Self::Constant(op.eval_operation_y(x, y))
                    }
                    (x, y) => Self::BinaryOperationY(op.clone(), Box::new(x), y),
                }
            }
            Self::VectorOperation(op, x, y) => match (x.simplify(registry), y.simplify(registry)) {
                (Self::Constant(x), Self::Constant(y)) => {
                    Self::Constant(op.eval_vector_operation_in_y(&x, y))
                }
                (x, y) => Self::VectorOperation(op.clone(), Box::new(x), Box::new(y)),
            },
            Self::Table(expression) => {
                match expression.simplify(registry, &registry.integer_tables) {
                    TableVectorExpression::Constant(value) => Self::Constant(value),
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
            Self::FromContinuous(op, expression) => match expression.simplify(registry) {
                ContinuousVectorExpression::Constant(value) => Self::Constant(
                    op.eval_vector(value)
                        .into_iter()
                        .map(|x| x as Integer)
                        .collect(),
                ),
                expression => Self::FromContinuous(op.clone(), Box::new(expression)),
            },
            _ => self.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::condition::ComparisonOperator;
    use super::super::continuous_expression::ContinuousExpression;
    use super::super::element_expression::ElementExpression;
    use super::super::reference_expression::ReferenceExpression;
    use super::super::table_vector_expression::TableVectorExpression;
    use super::super::vector_expression::VectorExpression;
    use super::*;
    use crate::state::*;

    #[test]
    fn constant_eval() {
        let state = State::default();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();
        let expression = IntegerVectorExpression::Constant(vec![0, 1]);
        assert_eq!(
            expression.eval(
                &state,
                &mut function_cache,
                &state_functions,
                &registry
            ),
            vec![0, 1]
        );
    }

    #[test]
    fn reverse_eval() {
        let state = State::default();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();
        let expression =
            IntegerVectorExpression::Reverse(Box::new(IntegerVectorExpression::Constant(vec![
                0, 1,
            ])));
        assert_eq!(
            expression.eval(
                &state,
                &mut function_cache,
                &state_functions,
                &registry
            ),
            vec![1, 0]
        );
    }

    #[test]
    fn push_eval() {
        let state = State::default();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();
        let expression = IntegerVectorExpression::Push(
            IntegerExpression::Constant(2),
            Box::new(IntegerVectorExpression::Constant(vec![0, 1])),
        );
        assert_eq!(
            expression.eval(
                &state,
                &mut function_cache,
                &state_functions,
                &registry
            ),
            vec![0, 1, 2]
        );
    }

    #[test]
    fn pop_eval() {
        let state = State::default();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();
        let expression =
            IntegerVectorExpression::Pop(Box::new(IntegerVectorExpression::Constant(vec![0, 1])));
        assert_eq!(
            expression.eval(
                &state,
                &mut function_cache,
                &state_functions,
                &registry
            ),
            vec![0]
        );
    }

    #[test]
    fn set_eval() {
        let state = State::default();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();
        let expression = IntegerVectorExpression::Set(
            IntegerExpression::Constant(2),
            Box::new(IntegerVectorExpression::Constant(vec![0, 1])),
            ElementExpression::Constant(1),
        );
        assert_eq!(
            expression.eval(
                &state,
                &mut function_cache,
                &state_functions,
                &registry
            ),
            vec![0, 2]
        );
    }

    #[test]
    fn unary_operation_eval() {
        let state = State::default();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();
        let expression = IntegerVectorExpression::UnaryOperation(
            UnaryOperator::Abs,
            Box::new(IntegerVectorExpression::Constant(vec![1, -1])),
        );
        assert_eq!(
            expression.eval(
                &state,
                &mut function_cache,
                &state_functions,
                &registry
            ),
            vec![1, 1]
        );
    }

    #[test]
    fn binary_operation_x_eval() {
        let state = State::default();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();
        let expression = IntegerVectorExpression::BinaryOperationX(
            BinaryOperator::Add,
            IntegerExpression::Constant(2),
            Box::new(IntegerVectorExpression::Constant(vec![0, 1])),
        );
        assert_eq!(
            expression.eval(
                &state,
                &mut function_cache,
                &state_functions,
                &registry
            ),
            vec![2, 3]
        );
    }

    #[test]
    fn binary_operation_y_eval() {
        let state = State::default();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();
        let expression = IntegerVectorExpression::BinaryOperationY(
            BinaryOperator::Add,
            Box::new(IntegerVectorExpression::Constant(vec![0, 1])),
            IntegerExpression::Constant(2),
        );
        assert_eq!(
            expression.eval(
                &state,
                &mut function_cache,
                &state_functions,
                &registry
            ),
            vec![2, 3]
        );
    }

    #[test]
    fn vector_operation_eval() {
        let state = State::default();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();
        let expression = IntegerVectorExpression::VectorOperation(
            BinaryOperator::Add,
            Box::new(IntegerVectorExpression::Constant(vec![0, 1])),
            Box::new(IntegerVectorExpression::Constant(vec![2, 3])),
        );
        assert_eq!(
            expression.eval(
                &state,
                &mut function_cache,
                &state_functions,
                &registry
            ),
            vec![2, 4]
        );
        let expression = IntegerVectorExpression::VectorOperation(
            BinaryOperator::Add,
            Box::new(IntegerVectorExpression::Constant(vec![0])),
            Box::new(IntegerVectorExpression::Pop(Box::new(
                IntegerVectorExpression::Constant(vec![2, 3]),
            ))),
        );
        assert_eq!(
            expression.eval(
                &state,
                &mut function_cache,
                &state_functions,
                &registry
            ),
            vec![2]
        );
        let expression = IntegerVectorExpression::VectorOperation(
            BinaryOperator::Add,
            Box::new(IntegerVectorExpression::Pop(Box::new(
                IntegerVectorExpression::Constant(vec![0, 1]),
            ))),
            Box::new(IntegerVectorExpression::Pop(Box::new(
                IntegerVectorExpression::Constant(vec![2, 3]),
            ))),
        );
        assert_eq!(
            expression.eval(
                &state,
                &mut function_cache,
                &state_functions,
                &registry
            ),
            vec![2]
        );
    }

    #[test]
    fn table_eval() {
        let state = State::default();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();
        let expression =
            IntegerVectorExpression::Table(Box::new(TableVectorExpression::Constant(vec![0, 1])));
        assert_eq!(
            expression.eval(
                &state,
                &mut function_cache,
                &state_functions,
                &registry
            ),
            vec![0, 1]
        );
    }

    #[test]
    fn if_eval() {
        let state = State::default();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();
        let expression = IntegerVectorExpression::If(
            Box::new(Condition::Constant(true)),
            Box::new(IntegerVectorExpression::Constant(vec![0, 1])),
            Box::new(IntegerVectorExpression::Constant(vec![2, 3])),
        );
        assert_eq!(
            expression.eval(
                &state,
                &mut function_cache,
                &state_functions,
                &registry
            ),
            vec![0, 1]
        );
        let expression = IntegerVectorExpression::If(
            Box::new(Condition::Constant(false)),
            Box::new(IntegerVectorExpression::Constant(vec![0, 1])),
            Box::new(IntegerVectorExpression::Constant(vec![2, 3])),
        );
        assert_eq!(
            expression.eval(
                &state,
                &mut function_cache,
                &state_functions,
                &registry
            ),
            vec![2, 3]
        );
    }

    #[test]
    fn from_continuous_eval() {
        let state = State::default();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();
        let expression = IntegerVectorExpression::FromContinuous(
            CastOperator::Ceil,
            Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
        );
        assert_eq!(
            expression.eval(
                &state,
                &mut function_cache,
                &state_functions,
                &registry
            ),
            vec![0, 1]
        );
    }

    #[test]
    fn constant_simplify() {
        let registry = TableRegistry::default();
        let expression = IntegerVectorExpression::Constant(vec![0, 1]);
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn reverse_simplify() {
        let registry = TableRegistry::default();
        let expression =
            IntegerVectorExpression::Reverse(Box::new(IntegerVectorExpression::Constant(vec![
                0, 1,
            ])));
        assert_eq!(
            expression.simplify(&registry),
            IntegerVectorExpression::Constant(vec![1, 0])
        );
        let expression = IntegerVectorExpression::Reverse(Box::new(IntegerVectorExpression::Push(
            IntegerExpression::Variable(0),
            Box::new(IntegerVectorExpression::Constant(vec![0, 1])),
        )));
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn push_simplify() {
        let registry = TableRegistry::default();
        let expression = IntegerVectorExpression::Push(
            IntegerExpression::Constant(2),
            Box::new(IntegerVectorExpression::Constant(vec![0, 1])),
        );
        assert_eq!(
            expression.simplify(&registry),
            IntegerVectorExpression::Constant(vec![0, 1, 2])
        );
        let expression = IntegerVectorExpression::Push(
            IntegerExpression::Variable(0),
            Box::new(IntegerVectorExpression::Constant(vec![0, 1])),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn pop_simplify() {
        let registry = TableRegistry::default();
        let expression =
            IntegerVectorExpression::Pop(Box::new(IntegerVectorExpression::Constant(vec![0, 1])));
        assert_eq!(
            expression.simplify(&registry),
            IntegerVectorExpression::Constant(vec![0])
        );
        let expression = IntegerVectorExpression::Pop(Box::new(IntegerVectorExpression::Push(
            IntegerExpression::Variable(0),
            Box::new(IntegerVectorExpression::Constant(vec![0, 1])),
        )));
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn set_simplify() {
        let registry = TableRegistry::default();
        let expression = IntegerVectorExpression::Set(
            IntegerExpression::Constant(2),
            Box::new(IntegerVectorExpression::Constant(vec![0, 1])),
            ElementExpression::Constant(1),
        );
        assert_eq!(
            expression.simplify(&registry),
            IntegerVectorExpression::Constant(vec![0, 2])
        );
        let expression = IntegerVectorExpression::Set(
            IntegerExpression::Constant(2),
            Box::new(IntegerVectorExpression::Constant(vec![0, 1])),
            ElementExpression::Variable(0),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn unary_operation_simplify() {
        let registry = TableRegistry::default();
        let expression = IntegerVectorExpression::UnaryOperation(
            UnaryOperator::Abs,
            Box::new(IntegerVectorExpression::Constant(vec![1, -1])),
        );
        assert_eq!(
            expression.simplify(&registry),
            IntegerVectorExpression::Constant(vec![1, 1])
        );
        let expression = IntegerVectorExpression::UnaryOperation(
            UnaryOperator::Abs,
            Box::new(IntegerVectorExpression::Push(
                IntegerExpression::Variable(0),
                Box::new(IntegerVectorExpression::Constant(vec![0, 1])),
            )),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn binary_operation_x_simplify() {
        let registry = TableRegistry::default();
        let expression = IntegerVectorExpression::BinaryOperationX(
            BinaryOperator::Add,
            IntegerExpression::Constant(2),
            Box::new(IntegerVectorExpression::Constant(vec![0, 1])),
        );
        assert_eq!(
            expression.simplify(&registry),
            IntegerVectorExpression::Constant(vec![2, 3])
        );
        let expression = IntegerVectorExpression::BinaryOperationX(
            BinaryOperator::Add,
            IntegerExpression::Variable(0),
            Box::new(IntegerVectorExpression::Constant(vec![0, 1])),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn binary_operation_y_simplify() {
        let registry = TableRegistry::default();
        let expression = IntegerVectorExpression::BinaryOperationY(
            BinaryOperator::Add,
            Box::new(IntegerVectorExpression::Constant(vec![0, 1])),
            IntegerExpression::Constant(2),
        );
        assert_eq!(
            expression.simplify(&registry),
            IntegerVectorExpression::Constant(vec![2, 3])
        );
        let expression = IntegerVectorExpression::BinaryOperationY(
            BinaryOperator::Add,
            Box::new(IntegerVectorExpression::Constant(vec![0, 1])),
            IntegerExpression::Variable(0),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn vector_operation_simplify() {
        let registry = TableRegistry::default();
        let expression = IntegerVectorExpression::VectorOperation(
            BinaryOperator::Add,
            Box::new(IntegerVectorExpression::Constant(vec![0, 1])),
            Box::new(IntegerVectorExpression::Constant(vec![2, 3])),
        );
        assert_eq!(
            expression.simplify(&registry),
            IntegerVectorExpression::Constant(vec![2, 4])
        );
        let expression = IntegerVectorExpression::VectorOperation(
            BinaryOperator::Add,
            Box::new(IntegerVectorExpression::Constant(vec![0, 1])),
            Box::new(IntegerVectorExpression::Push(
                IntegerExpression::Variable(0),
                Box::new(IntegerVectorExpression::Constant(vec![0, 1])),
            )),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn table_simplify() {
        let registry = TableRegistry::default();
        let expression =
            IntegerVectorExpression::Table(Box::new(TableVectorExpression::Constant(vec![0, 1])));
        assert_eq!(
            expression.simplify(&registry),
            IntegerVectorExpression::Constant(vec![0, 1])
        );
        let expression = IntegerVectorExpression::Table(Box::new(TableVectorExpression::Table1D(
            0,
            VectorExpression::Reference(ReferenceExpression::Variable(0)),
        )));
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn if_simplify() {
        let registry = TableRegistry::default();
        let expression = IntegerVectorExpression::If(
            Box::new(Condition::Constant(true)),
            Box::new(IntegerVectorExpression::Constant(vec![0, 1])),
            Box::new(IntegerVectorExpression::Constant(vec![2, 3])),
        );
        assert_eq!(
            expression.simplify(&registry),
            IntegerVectorExpression::Constant(vec![0, 1])
        );
        let expression = IntegerVectorExpression::If(
            Box::new(Condition::Constant(false)),
            Box::new(IntegerVectorExpression::Constant(vec![0, 1])),
            Box::new(IntegerVectorExpression::Constant(vec![2, 3])),
        );
        assert_eq!(
            expression.simplify(&registry),
            IntegerVectorExpression::Constant(vec![2, 3])
        );
        let expression = IntegerVectorExpression::If(
            Box::new(Condition::ComparisonI(
                ComparisonOperator::Gt,
                Box::new(IntegerExpression::Constant(0)),
                Box::new(IntegerExpression::Variable(0)),
            )),
            Box::new(IntegerVectorExpression::Constant(vec![0, 1])),
            Box::new(IntegerVectorExpression::Constant(vec![2, 3])),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn from_continuous_simplify() {
        let registry = TableRegistry::default();
        let expression = IntegerVectorExpression::FromContinuous(
            CastOperator::Ceil,
            Box::new(ContinuousVectorExpression::Constant(vec![0.0, 0.5])),
        );
        assert_eq!(
            expression.simplify(&registry),
            IntegerVectorExpression::Constant(vec![0, 1])
        );
        let expression = IntegerVectorExpression::FromContinuous(
            CastOperator::Ceil,
            Box::new(ContinuousVectorExpression::Push(
                ContinuousExpression::Variable(0),
                Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
            )),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }
}
