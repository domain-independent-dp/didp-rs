use super::numeric_table_expression::NumericTableExpression;
use super::reference_expression::ReferenceExpression;
use super::set_expression::SetExpression;
use super::vector_expression::VectorExpression;
use crate::state::State;
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
            Self::Cost => panic!(String::from("cost cannot be accessd from eval function")),
            Self::NumericOperation(op, a, b) => {
                let a = a.eval(state, registry);
                let b = b.eval(state, registry);
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
        }
    }

    pub fn eval_cost(&self, cost: T, state: &State, registry: &TableRegistry) -> T {
        match self {
            Self::Cost => cost,
            Self::NumericOperation(op, a, b) => {
                let a = a.eval_cost(cost, state, registry);
                let b = b.eval_cost(cost, state, registry);
                Self::eval_operation(op, a, b)
            }
            _ => self.eval(state, registry),
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

#[cfg(test)]
mod tests {
    use super::super::element_expression;
    use super::super::reference_expression::ReferenceExpression;
    use super::*;
    use crate::state;
    use crate::table;
    use crate::table_data;
    use crate::variable::*;
    use ordered_float::OrderedFloat;
    use std::collections::HashMap;
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
        let mut name_to_constant = HashMap::new();
        name_to_constant.insert(String::from("f0"), 0);

        let tables_1d = vec![table::Table1D::new(vec![10, 20, 30])];
        let mut name_to_table_1d = HashMap::new();
        name_to_table_1d.insert(String::from("f1"), 0);

        let tables_2d = vec![table::Table2D::new(vec![
            vec![10, 20, 30],
            vec![40, 50, 60],
            vec![70, 80, 90],
        ])];
        let mut name_to_table_2d = HashMap::new();
        name_to_table_2d.insert(String::from("f2"), 0);

        let tables_3d = vec![table::Table3D::new(vec![
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
        ])];
        let mut name_to_table_3d = HashMap::new();
        name_to_table_3d.insert(String::from("f3"), 0);

        let mut map = HashMap::new();
        let key = vec![0, 1, 0, 0];
        map.insert(key, 100);
        let key = vec![0, 1, 0, 1];
        map.insert(key, 200);
        let key = vec![0, 1, 2, 0];
        map.insert(key, 300);
        let key = vec![0, 1, 2, 1];
        map.insert(key, 400);
        let tables = vec![table::Table::new(map, 0)];
        let mut name_to_table = HashMap::new();
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

        let mut name_to_constant = HashMap::new();
        name_to_constant.insert(String::from("cf0"), 0.0);

        let tables_1d = vec![table::Table1D::new(vec![10.0, 20.0, 30.0])];
        let mut name_to_table_1d = HashMap::new();
        name_to_table_1d.insert(String::from("cf1"), 0);

        let tables_2d = vec![table::Table2D::new(vec![
            vec![10.0, 20.0, 30.0],
            vec![40.0, 50.0, 60.0],
            vec![70.0, 80.0, 90.0],
        ])];
        let mut name_to_table_2d = HashMap::new();
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
        let mut name_to_table_3d = HashMap::new();
        name_to_table_3d.insert(String::from("cf3"), 0);

        let mut map = HashMap::new();
        let key = vec![0, 1, 0, 0];
        map.insert(key, 100.0);
        let key = vec![0, 1, 0, 1];
        map.insert(key, 200.0);
        let key = vec![0, 1, 2, 0];
        map.insert(key, 300.0);
        let key = vec![0, 1, 2, 1];
        map.insert(key, 400.0);
        let tables = vec![table::Table::new(map, 0.0)];
        let mut name_to_table = HashMap::new();
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
    fn number_eval() {
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
        assert_eq!(expression.eval(&state, &registry), 1.0);
        let expression = NumericExpression::<Continuous>::ContinuousVariable(1);
        assert_eq!(expression.eval(&state, &registry), 2.0);
        let expression = NumericExpression::<Continuous>::ContinuousVariable(2);
        assert_eq!(expression.eval(&state, &registry), 3.0);
    }

    #[test]
    fn continuous_resource_variable_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let expression = NumericExpression::<Continuous>::ContinuousResourceVariable(0);
        assert_eq!(expression.eval(&state, &registry), 4.0);
        let expression = NumericExpression::<Continuous>::ContinuousResourceVariable(1);
        assert_eq!(expression.eval(&state, &registry), 5.0);
        let expression = NumericExpression::<Continuous>::ContinuousResourceVariable(2);
        assert_eq!(expression.eval(&state, &registry), 6.0);
    }

    #[test]
    fn cost_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let expression: NumericExpression<Integer> = NumericExpression::Cost {};
        assert_eq!(expression.eval_cost(0, &state, &registry), 0);
    }

    #[test]
    #[should_panic]
    fn cost_eval_panic() {
        let registry = generate_registry();
        let state = generate_state();
        let expression: NumericExpression<Integer> = NumericExpression::Cost {};
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
    }

    #[test]
    fn table_eval() {
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
        assert_eq!(expression.eval(&state, &registry), 100.0);
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
        assert_eq!(expression.eval(&state, &registry), 200.0);
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
        assert_eq!(expression.eval(&state, &registry), 300.0);
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
        assert_eq!(expression.eval(&state, &registry), 400.0);
    }

    #[test]
    fn number_simplify() {
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
}
