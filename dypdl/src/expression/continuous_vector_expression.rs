use super::condition::Condition;
use super::continuous_expression::ContinuousExpression;
use super::element_expression::ElementExpression;
use super::integer_vector_expression::IntegerVectorExpression;
use super::numeric_operator::{
    BinaryOperator, CastOperator, ContinuousBinaryOperator, ContinuousUnaryOperator, UnaryOperator,
};
use super::table_vector_expression::TableVectorExpression;
use crate::state::StateInterface;
use crate::table_registry::TableRegistry;
use crate::variable_type::Continuous;
use std::boxed::Box;

/// Expression representing a vector of numeric values with the continuous value type.
#[derive(Debug, PartialEq, Clone)]
pub enum ContinuousVectorExpression {
    /// Constant.
    Constant(Vec<Continuous>),
    /// Reverses a continuous vector.
    Reverse(Box<ContinuousVectorExpression>),
    /// Pushes an element to a continuous vector.
    Push(ContinuousExpression, Box<ContinuousVectorExpression>),
    /// Pops an element from a continuous vector.
    Pop(Box<ContinuousVectorExpression>),
    /// Set an element in a continuous vector.
    Set(
        ContinuousExpression,
        Box<ContinuousVectorExpression>,
        ElementExpression,
    ),
    /// Elementwise unary arithmetic operation.
    UnaryOperation(UnaryOperator, Box<ContinuousVectorExpression>),
    /// Elementwise unary arithmetic operation specific to continuous values.
    ContinuousUnaryOperation(ContinuousUnaryOperator, Box<ContinuousVectorExpression>),
    /// Elementwise rounding operation.
    Round(CastOperator, Box<ContinuousVectorExpression>),
    /// Elementwise binary arithmetic operation.
    BinaryOperationX(
        BinaryOperator,
        ContinuousExpression,
        Box<ContinuousVectorExpression>,
    ),
    /// Elementwise binary arithmetic operation.
    BinaryOperationY(
        BinaryOperator,
        Box<ContinuousVectorExpression>,
        ContinuousExpression,
    ),
    /// Elementwise binary arithmetic operation.
    VectorOperation(
        BinaryOperator,
        Box<ContinuousVectorExpression>,
        Box<ContinuousVectorExpression>,
    ),
    /// Elementwise binary arithmetic operation specific to continuous value.
    ContinuousBinaryOperationX(
        ContinuousBinaryOperator,
        ContinuousExpression,
        Box<ContinuousVectorExpression>,
    ),
    /// Elementwise binary arithmetic operation specific to continuous value.
    ContinuousBinaryOperationY(
        ContinuousBinaryOperator,
        Box<ContinuousVectorExpression>,
        ContinuousExpression,
    ),
    /// Elementwise binary arithmetic operation specific to continuous value.
    ContinuousVectorOperation(
        ContinuousBinaryOperator,
        Box<ContinuousVectorExpression>,
        Box<ContinuousVectorExpression>,
    ),
    /// A vector of constants in a continuous table.
    Table(Box<TableVectorExpression<Continuous>>),
    /// If-then-else expression, which returns the first one if the condition holds and the second one otherwise.
    If(
        Box<Condition>,
        Box<ContinuousVectorExpression>,
        Box<ContinuousVectorExpression>,
    ),
    /// Conversion from an integer vector.
    FromInteger(Box<IntegerVectorExpression>),
}

impl ContinuousVectorExpression {
    /// Returns the evaluation result.
    ///
    /// # Panics
    ///
    /// Panics if the cost of the transition state is used or a min/max reduce operation is performed on an empty set or vector.
    pub fn eval<U: StateInterface>(&self, state: &U, registry: &TableRegistry) -> Vec<Continuous> {
        self.eval_inner(None, state, registry)
    }

    /// Returns the evaluation result of a cost expression.
    ///
    /// # Panics
    ///
    /// Panics if a min/max reduce operation is performed on an empty set or vector.
    pub fn eval_cost<U: StateInterface>(
        &self,
        cost: Continuous,
        state: &U,
        registry: &TableRegistry,
    ) -> Vec<Continuous> {
        self.eval_inner(Some(cost), state, registry)
    }

    pub fn eval_inner<U: StateInterface>(
        &self,
        cost: Option<Continuous>,
        state: &U,
        registry: &TableRegistry,
    ) -> Vec<Continuous> {
        match self {
            Self::Constant(vector) => vector.clone(),
            Self::Reverse(vector) => {
                let mut vector = vector.eval_inner(cost, state, registry);
                vector.reverse();
                vector
            }
            Self::Push(value, vector) => {
                let mut vector = vector.eval_inner(cost, state, registry);
                vector.push(value.eval(state, registry));
                vector
            }
            Self::Pop(vector) => {
                let mut vector = vector.eval_inner(cost, state, registry);
                vector.pop();
                vector
            }
            Self::Set(value, vector, i) => {
                let mut vector = vector.eval_inner(cost, state, registry);
                vector[i.eval(state, registry)] = value.eval(state, registry);
                vector
            }
            Self::UnaryOperation(op, x) => op.eval_vector(x.eval_inner(cost, state, registry)),
            Self::ContinuousUnaryOperation(op, x) => {
                op.eval_vector(x.eval_inner(cost, state, registry))
            }
            Self::Round(op, x) => op.eval_vector(x.eval_inner(cost, state, registry)),
            Self::BinaryOperationX(op, x, y) => op.eval_operation_x(
                cost.map_or_else(
                    || x.eval(state, registry),
                    |cost| x.eval_cost(cost, state, registry),
                ),
                y.eval_inner(cost, state, registry),
            ),
            Self::BinaryOperationY(op, x, y) => op.eval_operation_y(
                x.eval_inner(cost, state, registry),
                cost.map_or_else(
                    || y.eval(state, registry),
                    |cost| y.eval_cost(cost, state, registry),
                ),
            ),
            Self::VectorOperation(op, x, y) => match (x.as_ref(), y.as_ref()) {
                (Self::Constant(x), y) => {
                    op.eval_vector_operation_in_y(x, y.eval_inner(cost, state, registry))
                }
                (x, Self::Constant(y)) => {
                    op.eval_vector_operation_in_x(x.eval_inner(cost, state, registry), y)
                }
                (x, y) => op.eval_vector_operation_in_y(
                    &x.eval_inner(cost, state, registry),
                    y.eval_inner(cost, state, registry),
                ),
            },
            Self::ContinuousBinaryOperationX(op, x, y) => op.eval_operation_x(
                cost.map_or_else(
                    || x.eval(state, registry),
                    |cost| x.eval_cost(cost, state, registry),
                ),
                y.eval_inner(cost, state, registry),
            ),
            Self::ContinuousBinaryOperationY(op, x, y) => op.eval_operation_y(
                x.eval_inner(cost, state, registry),
                cost.map_or_else(
                    || y.eval(state, registry),
                    |cost| y.eval_cost(cost, state, registry),
                ),
            ),
            Self::ContinuousVectorOperation(op, x, y) => match (x.as_ref(), y.as_ref()) {
                (Self::Constant(x), y) => {
                    op.eval_vector_operation_in_y(x, y.eval_inner(cost, state, registry))
                }
                (x, Self::Constant(y)) => {
                    op.eval_vector_operation_in_x(x.eval_inner(cost, state, registry), y)
                }
                (x, y) => op.eval_vector_operation_in_y(
                    &x.eval_inner(cost, state, registry),
                    y.eval_inner(cost, state, registry),
                ),
            },
            Self::Table(expression) => {
                expression.eval(state, registry, &registry.continuous_tables)
            }
            Self::If(condition, x, y) => {
                if condition.eval(state, registry) {
                    x.eval_inner(cost, state, registry)
                } else {
                    y.eval_inner(cost, state, registry)
                }
            }
            Self::FromInteger(x) => x
                .eval(state, registry)
                .into_iter()
                .map(Continuous::from)
                .collect(),
        }
    }

    /// Returns a simplified version by precomputation.
    ///
    /// # Panics
    ///
    /// Panics if a min/max reduce operation is performed on an empty set or vector.
    pub fn simplify(&self, registry: &TableRegistry) -> ContinuousVectorExpression {
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
                    (ContinuousExpression::Constant(value), Self::Constant(mut vector)) => {
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
                    ContinuousExpression::Constant(value),
                    Self::Constant(mut vector),
                    ElementExpression::Constant(i),
                ) => {
                    vector[i] = value;
                    Self::Constant(vector)
                }
                (value, vector, i) => Self::Set(value, Box::new(vector), i),
            },
            Self::UnaryOperation(op, x) => match x.simplify(registry) {
                Self::Constant(x) => Self::Constant(op.eval_vector(x)),
                x => Self::UnaryOperation(op.clone(), Box::new(x)),
            },
            Self::ContinuousUnaryOperation(op, x) => match x.simplify(registry) {
                Self::Constant(x) => Self::Constant(op.eval_vector(x)),
                x => Self::ContinuousUnaryOperation(op.clone(), Box::new(x)),
            },
            Self::Round(op, x) => match x.simplify(registry) {
                Self::Constant(x) => Self::Constant(op.eval_vector(x)),
                x => Self::Round(op.clone(), Box::new(x)),
            },
            Self::BinaryOperationX(op, x, y) => {
                match (x.simplify(registry), y.simplify(registry)) {
                    (ContinuousExpression::Constant(x), Self::Constant(y)) => {
                        Self::Constant(op.eval_operation_x(x, y))
                    }
                    (x, y) => Self::BinaryOperationX(op.clone(), x, Box::new(y)),
                }
            }
            Self::BinaryOperationY(op, x, y) => {
                match (x.simplify(registry), y.simplify(registry)) {
                    (Self::Constant(x), ContinuousExpression::Constant(y)) => {
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
            Self::ContinuousBinaryOperationX(op, x, y) => {
                match (x.simplify(registry), y.simplify(registry)) {
                    (ContinuousExpression::Constant(x), Self::Constant(y)) => {
                        Self::Constant(op.eval_operation_x(x, y))
                    }
                    (x, y) => Self::ContinuousBinaryOperationX(op.clone(), x, Box::new(y)),
                }
            }
            Self::ContinuousBinaryOperationY(op, x, y) => {
                match (x.simplify(registry), y.simplify(registry)) {
                    (Self::Constant(x), ContinuousExpression::Constant(y)) => {
                        Self::Constant(op.eval_operation_y(x, y))
                    }
                    (x, y) => Self::ContinuousBinaryOperationY(op.clone(), Box::new(x), y),
                }
            }
            Self::ContinuousVectorOperation(op, x, y) => {
                match (x.simplify(registry), y.simplify(registry)) {
                    (Self::Constant(x), Self::Constant(y)) => {
                        Self::Constant(op.eval_vector_operation_in_y(&x, y))
                    }
                    (x, y) => Self::ContinuousVectorOperation(op.clone(), Box::new(x), Box::new(y)),
                }
            }
            Self::Table(expression) => {
                match expression.simplify(registry, &registry.continuous_tables) {
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
            Self::FromInteger(expression) => match expression.simplify(registry) {
                IntegerVectorExpression::Constant(value) => {
                    Self::Constant(value.into_iter().map(Continuous::from).collect())
                }
                expression => Self::FromInteger(Box::new(expression)),
            },
            _ => self.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::condition::ComparisonOperator;
    use super::super::element_expression::ElementExpression;
    use super::super::integer_expression::IntegerExpression;
    use super::super::reference_expression::ReferenceExpression;
    use super::super::table_vector_expression::TableVectorExpression;
    use super::super::vector_expression::VectorExpression;
    use super::*;
    use crate::state::*;

    #[test]
    fn constant_eval() {
        let state = State::default();
        let registry = TableRegistry::default();
        let expression = ContinuousVectorExpression::Constant(vec![0.0, 1.0]);
        assert_eq!(expression.eval(&state, &registry), vec![0.0, 1.0]);
    }

    #[test]
    fn reverse_eval() {
        let state = State::default();
        let registry = TableRegistry::default();
        let expression = ContinuousVectorExpression::Reverse(Box::new(
            ContinuousVectorExpression::Constant(vec![0.0, 1.0]),
        ));
        assert_eq!(expression.eval(&state, &registry), vec![1.0, 0.0]);
    }

    #[test]
    fn push_eval() {
        let state = State::default();
        let registry = TableRegistry::default();
        let expression = ContinuousVectorExpression::Push(
            ContinuousExpression::Constant(2.0),
            Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
        );
        assert_eq!(expression.eval(&state, &registry), vec![0.0, 1.0, 2.0]);
    }

    #[test]
    fn pop_eval() {
        let state = State::default();
        let registry = TableRegistry::default();
        let expression =
            ContinuousVectorExpression::Pop(Box::new(ContinuousVectorExpression::Constant(vec![
                0.0, 1.0,
            ])));
        assert_eq!(expression.eval(&state, &registry), vec![0.0]);
    }

    #[test]
    fn set_eval() {
        let state = State::default();
        let registry = TableRegistry::default();
        let expression = ContinuousVectorExpression::Set(
            ContinuousExpression::Constant(2.0),
            Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
            ElementExpression::Constant(1),
        );
        assert_eq!(expression.eval(&state, &registry), vec![0.0, 2.0]);
    }

    #[test]
    fn unary_operation_eval() {
        let state = State::default();
        let registry = TableRegistry::default();
        let expression = ContinuousVectorExpression::UnaryOperation(
            UnaryOperator::Abs,
            Box::new(ContinuousVectorExpression::Constant(vec![1.0, -1.0])),
        );
        assert_eq!(expression.eval(&state, &registry), vec![1.0, 1.0]);
    }

    #[test]
    fn continuous_unary_operation_eval() {
        let state = State::default();
        let registry = TableRegistry::default();
        let expression = ContinuousVectorExpression::ContinuousUnaryOperation(
            ContinuousUnaryOperator::Sqrt,
            Box::new(ContinuousVectorExpression::Constant(vec![4.0, 9.0])),
        );
        assert_eq!(expression.eval(&state, &registry), vec![2.0, 3.0]);
    }

    #[test]
    fn binary_operation_x_eval() {
        let state = State::default();
        let registry = TableRegistry::default();
        let expression = ContinuousVectorExpression::BinaryOperationX(
            BinaryOperator::Add,
            ContinuousExpression::Constant(2.0),
            Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
        );
        assert_eq!(expression.eval(&state, &registry), vec![2.0, 3.0]);
    }

    #[test]
    fn binary_operation_y_eval() {
        let state = State::default();
        let registry = TableRegistry::default();
        let expression = ContinuousVectorExpression::BinaryOperationY(
            BinaryOperator::Add,
            Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
            ContinuousExpression::Constant(2.0),
        );
        assert_eq!(expression.eval(&state, &registry), vec![2.0, 3.0]);
    }

    #[test]
    fn vector_operation_eval() {
        let state = State::default();
        let registry = TableRegistry::default();
        let expression = ContinuousVectorExpression::VectorOperation(
            BinaryOperator::Add,
            Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
            Box::new(ContinuousVectorExpression::Constant(vec![2.0, 3.0])),
        );
        assert_eq!(expression.eval(&state, &registry), vec![2.0, 4.0]);
        let expression = ContinuousVectorExpression::VectorOperation(
            BinaryOperator::Add,
            Box::new(ContinuousVectorExpression::Constant(vec![0.0])),
            Box::new(ContinuousVectorExpression::Pop(Box::new(
                ContinuousVectorExpression::Constant(vec![2.0, 3.0]),
            ))),
        );
        assert_eq!(expression.eval(&state, &registry), vec![2.0]);
        let expression = ContinuousVectorExpression::VectorOperation(
            BinaryOperator::Add,
            Box::new(ContinuousVectorExpression::Pop(Box::new(
                ContinuousVectorExpression::Constant(vec![0.0, 1.0]),
            ))),
            Box::new(ContinuousVectorExpression::Pop(Box::new(
                ContinuousVectorExpression::Constant(vec![2.0, 3.0]),
            ))),
        );
        assert_eq!(expression.eval(&state, &registry), vec![2.0]);
    }

    #[test]
    fn continuous_binary_operation_x_eval() {
        let state = State::default();
        let registry = TableRegistry::default();
        let expression = ContinuousVectorExpression::ContinuousBinaryOperationX(
            ContinuousBinaryOperator::Pow,
            ContinuousExpression::Constant(2.0),
            Box::new(ContinuousVectorExpression::Constant(vec![2.0, 3.0])),
        );
        assert_eq!(expression.eval(&state, &registry), vec![4.0, 8.0]);
    }

    #[test]
    fn continuous_binary_operation_y_eval() {
        let state = State::default();
        let registry = TableRegistry::default();
        let expression = ContinuousVectorExpression::ContinuousBinaryOperationY(
            ContinuousBinaryOperator::Pow,
            Box::new(ContinuousVectorExpression::Constant(vec![2.0, 3.0])),
            ContinuousExpression::Constant(2.0),
        );
        assert_eq!(expression.eval(&state, &registry), vec![4.0, 9.0]);
    }

    #[test]
    fn continuous_vector_operation_eval() {
        let state = State::default();
        let registry = TableRegistry::default();
        let expression = ContinuousVectorExpression::ContinuousVectorOperation(
            ContinuousBinaryOperator::Pow,
            Box::new(ContinuousVectorExpression::Constant(vec![2.0, 3.0])),
            Box::new(ContinuousVectorExpression::Constant(vec![2.0, 3.0])),
        );
        assert_eq!(expression.eval(&state, &registry), vec![4.0, 27.0]);
        let expression = ContinuousVectorExpression::ContinuousVectorOperation(
            ContinuousBinaryOperator::Pow,
            Box::new(ContinuousVectorExpression::Constant(vec![2.0])),
            Box::new(ContinuousVectorExpression::Pop(Box::new(
                ContinuousVectorExpression::Constant(vec![2.0, 3.0]),
            ))),
        );
        assert_eq!(expression.eval(&state, &registry), vec![4.0]);
        let expression = ContinuousVectorExpression::ContinuousVectorOperation(
            ContinuousBinaryOperator::Pow,
            Box::new(ContinuousVectorExpression::Pop(Box::new(
                ContinuousVectorExpression::Constant(vec![2.0, 1.0]),
            ))),
            Box::new(ContinuousVectorExpression::Pop(Box::new(
                ContinuousVectorExpression::Constant(vec![2.0, 3.0]),
            ))),
        );
        assert_eq!(expression.eval(&state, &registry), vec![4.0]);
    }

    #[test]
    fn table_eval() {
        let state = State::default();
        let registry = TableRegistry::default();
        let expression =
            ContinuousVectorExpression::Table(Box::new(TableVectorExpression::Constant(vec![
                0.0, 1.0,
            ])));
        assert_eq!(expression.eval(&state, &registry), vec![0.0, 1.0]);
    }

    #[test]
    fn if_eval() {
        let state = State::default();
        let registry = TableRegistry::default();
        let expression = ContinuousVectorExpression::If(
            Box::new(Condition::Constant(true)),
            Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
            Box::new(ContinuousVectorExpression::Constant(vec![2.0, 3.0])),
        );
        assert_eq!(expression.eval(&state, &registry), vec![0.0, 1.0]);
        let expression = ContinuousVectorExpression::If(
            Box::new(Condition::Constant(false)),
            Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
            Box::new(ContinuousVectorExpression::Constant(vec![2.0, 3.0])),
        );
        assert_eq!(expression.eval(&state, &registry), vec![2.0, 3.0]);
    }

    #[test]
    fn from_integer_eval() {
        let state = State::default();
        let registry = TableRegistry::default();
        let expression = ContinuousVectorExpression::FromInteger(Box::new(
            IntegerVectorExpression::Constant(vec![0, 1]),
        ));
        assert_eq!(expression.eval(&state, &registry), vec![0.0, 1.0]);
    }

    #[test]
    fn constant_simplify() {
        let registry = TableRegistry::default();
        let expression = ContinuousVectorExpression::Constant(vec![0.0, 1.0]);
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn reverse_simplify() {
        let registry = TableRegistry::default();
        let expression = ContinuousVectorExpression::Reverse(Box::new(
            ContinuousVectorExpression::Constant(vec![0.0, 1.0]),
        ));
        assert_eq!(
            expression.simplify(&registry),
            ContinuousVectorExpression::Constant(vec![1.0, 0.0])
        );
        let expression =
            ContinuousVectorExpression::Reverse(Box::new(ContinuousVectorExpression::Push(
                ContinuousExpression::Variable(0),
                Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
            )));
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn push_simplify() {
        let registry = TableRegistry::default();
        let expression = ContinuousVectorExpression::Push(
            ContinuousExpression::Constant(2.0),
            Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
        );
        assert_eq!(
            expression.simplify(&registry),
            ContinuousVectorExpression::Constant(vec![0.0, 1.0, 2.0])
        );
        let expression = ContinuousVectorExpression::Push(
            ContinuousExpression::Variable(0),
            Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn pop_simplify() {
        let registry = TableRegistry::default();
        let expression =
            ContinuousVectorExpression::Pop(Box::new(ContinuousVectorExpression::Constant(vec![
                0.0, 1.0,
            ])));
        assert_eq!(
            expression.simplify(&registry),
            ContinuousVectorExpression::Constant(vec![0.0])
        );
        let expression =
            ContinuousVectorExpression::Pop(Box::new(ContinuousVectorExpression::Push(
                ContinuousExpression::Variable(0),
                Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
            )));
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn set_simplify() {
        let registry = TableRegistry::default();
        let expression = ContinuousVectorExpression::Set(
            ContinuousExpression::Constant(2.0),
            Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
            ElementExpression::Constant(1),
        );
        assert_eq!(
            expression.simplify(&registry),
            ContinuousVectorExpression::Constant(vec![0.0, 2.0])
        );
        let expression = ContinuousVectorExpression::Set(
            ContinuousExpression::Constant(2.0),
            Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
            ElementExpression::Variable(0),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn unary_operation_simplify() {
        let registry = TableRegistry::default();
        let expression = ContinuousVectorExpression::UnaryOperation(
            UnaryOperator::Abs,
            Box::new(ContinuousVectorExpression::Constant(vec![1.0, -1.0])),
        );
        assert_eq!(
            expression.simplify(&registry),
            ContinuousVectorExpression::Constant(vec![1.0, 1.0])
        );
        let expression = ContinuousVectorExpression::UnaryOperation(
            UnaryOperator::Abs,
            Box::new(ContinuousVectorExpression::Push(
                ContinuousExpression::Variable(0),
                Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
            )),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn continuous_unary_operation_simplify() {
        let registry = TableRegistry::default();
        let expression = ContinuousVectorExpression::ContinuousUnaryOperation(
            ContinuousUnaryOperator::Sqrt,
            Box::new(ContinuousVectorExpression::Constant(vec![4.0, 9.0])),
        );
        assert_eq!(
            expression.simplify(&registry),
            ContinuousVectorExpression::Constant(vec![2.0, 3.0])
        );
        let expression = ContinuousVectorExpression::ContinuousUnaryOperation(
            ContinuousUnaryOperator::Sqrt,
            Box::new(ContinuousVectorExpression::Push(
                ContinuousExpression::Variable(0),
                Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
            )),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn binary_operation_x_simplify() {
        let registry = TableRegistry::default();
        let expression = ContinuousVectorExpression::BinaryOperationX(
            BinaryOperator::Add,
            ContinuousExpression::Constant(2.0),
            Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
        );
        assert_eq!(
            expression.simplify(&registry),
            ContinuousVectorExpression::Constant(vec![2.0, 3.0])
        );
        let expression = ContinuousVectorExpression::BinaryOperationX(
            BinaryOperator::Add,
            ContinuousExpression::Variable(0),
            Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn binary_operation_y_simplify() {
        let registry = TableRegistry::default();
        let expression = ContinuousVectorExpression::BinaryOperationY(
            BinaryOperator::Add,
            Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
            ContinuousExpression::Constant(2.0),
        );
        assert_eq!(
            expression.simplify(&registry),
            ContinuousVectorExpression::Constant(vec![2.0, 3.0])
        );
        let expression = ContinuousVectorExpression::BinaryOperationY(
            BinaryOperator::Add,
            Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
            ContinuousExpression::Variable(0),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn vector_operation_simplify() {
        let registry = TableRegistry::default();
        let expression = ContinuousVectorExpression::VectorOperation(
            BinaryOperator::Add,
            Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
            Box::new(ContinuousVectorExpression::Constant(vec![2.0, 3.0])),
        );
        assert_eq!(
            expression.simplify(&registry),
            ContinuousVectorExpression::Constant(vec![2.0, 4.0])
        );
        let expression = ContinuousVectorExpression::VectorOperation(
            BinaryOperator::Add,
            Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
            Box::new(ContinuousVectorExpression::Push(
                ContinuousExpression::Variable(0),
                Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
            )),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn continuous_binary_operation_x_simplify() {
        let registry = TableRegistry::default();
        let expression = ContinuousVectorExpression::ContinuousBinaryOperationX(
            ContinuousBinaryOperator::Pow,
            ContinuousExpression::Constant(2.0),
            Box::new(ContinuousVectorExpression::Constant(vec![2.0, 3.0])),
        );
        assert_eq!(
            expression.simplify(&registry),
            ContinuousVectorExpression::Constant(vec![4.0, 8.0])
        );
        let expression = ContinuousVectorExpression::ContinuousBinaryOperationX(
            ContinuousBinaryOperator::Pow,
            ContinuousExpression::Variable(0),
            Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn continuous_binary_operation_y_simplify() {
        let registry = TableRegistry::default();
        let expression = ContinuousVectorExpression::ContinuousBinaryOperationY(
            ContinuousBinaryOperator::Pow,
            Box::new(ContinuousVectorExpression::Constant(vec![2.0, 3.0])),
            ContinuousExpression::Constant(2.0),
        );
        assert_eq!(
            expression.simplify(&registry),
            ContinuousVectorExpression::Constant(vec![4.0, 9.0])
        );
        let expression = ContinuousVectorExpression::ContinuousBinaryOperationY(
            ContinuousBinaryOperator::Pow,
            Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
            ContinuousExpression::Variable(0),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn continuous_vector_operation_simplify() {
        let registry = TableRegistry::default();
        let expression = ContinuousVectorExpression::ContinuousVectorOperation(
            ContinuousBinaryOperator::Pow,
            Box::new(ContinuousVectorExpression::Constant(vec![2.0, 3.0])),
            Box::new(ContinuousVectorExpression::Constant(vec![2.0, 3.0])),
        );
        assert_eq!(
            expression.simplify(&registry),
            ContinuousVectorExpression::Constant(vec![4.0, 27.0])
        );
        let expression = ContinuousVectorExpression::ContinuousVectorOperation(
            ContinuousBinaryOperator::Pow,
            Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
            Box::new(ContinuousVectorExpression::Push(
                ContinuousExpression::Variable(0),
                Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
            )),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn table_simplify() {
        let registry = TableRegistry::default();
        let expression =
            ContinuousVectorExpression::Table(Box::new(TableVectorExpression::Constant(vec![
                0.0, 1.0,
            ])));
        assert_eq!(
            expression.simplify(&registry),
            ContinuousVectorExpression::Constant(vec![0.0, 1.0])
        );
        let expression =
            ContinuousVectorExpression::Table(Box::new(TableVectorExpression::Table1D(
                0,
                VectorExpression::Reference(ReferenceExpression::Variable(0)),
            )));
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn if_simplify() {
        let registry = TableRegistry::default();
        let expression = ContinuousVectorExpression::If(
            Box::new(Condition::Constant(true)),
            Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
            Box::new(ContinuousVectorExpression::Constant(vec![2.0, 3.0])),
        );
        assert_eq!(
            expression.simplify(&registry),
            ContinuousVectorExpression::Constant(vec![0.0, 1.0])
        );
        let expression = ContinuousVectorExpression::If(
            Box::new(Condition::Constant(false)),
            Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
            Box::new(ContinuousVectorExpression::Constant(vec![2.0, 3.0])),
        );
        assert_eq!(
            expression.simplify(&registry),
            ContinuousVectorExpression::Constant(vec![2.0, 3.0])
        );
        let expression = ContinuousVectorExpression::If(
            Box::new(Condition::ComparisonC(
                ComparisonOperator::Gt,
                Box::new(ContinuousExpression::Constant(0.0)),
                Box::new(ContinuousExpression::Variable(0)),
            )),
            Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
            Box::new(ContinuousVectorExpression::Constant(vec![2.0, 3.0])),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn from_integer_simplify() {
        let registry = TableRegistry::default();
        let expression = ContinuousVectorExpression::FromInteger(Box::new(
            IntegerVectorExpression::Constant(vec![0, 1]),
        ));
        assert_eq!(
            expression.simplify(&registry),
            ContinuousVectorExpression::Constant(vec![0.0, 1.0])
        );
        let expression =
            ContinuousVectorExpression::FromInteger(Box::new(IntegerVectorExpression::Push(
                IntegerExpression::Variable(0),
                Box::new(IntegerVectorExpression::Constant(vec![0, 1])),
            )));
        assert_eq!(expression.simplify(&registry), expression);
    }
}
