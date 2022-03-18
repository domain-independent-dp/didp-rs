use super::numeric_table_expression;
use super::set_expression;
use crate::state;
use crate::table_registry;
use crate::variable;
use std::boxed::Box;
use std::cmp;

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum NumericExpression<T: variable::Numeric> {
    Constant(T),
    Variable(usize),
    ResourceVariable(usize),
    Cost,
    NumericOperation(
        NumericOperator,
        Box<NumericExpression<T>>,
        Box<NumericExpression<T>>,
    ),
    Cardinality(set_expression::SetExpression),
    Table(numeric_table_expression::NumericTableExpression<T>),
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum NumericOperator {
    Add,
    Subtract,
    Multiply,
    Divide,
    Max,
    Min,
}

impl<T: variable::Numeric> NumericExpression<T> {
    pub fn eval(
        &self,
        state: &state::State<T>,
        metadata: &state::StateMetadata,
        registry: &table_registry::TableRegistry<T>,
    ) -> T {
        match self {
            Self::Constant(x) => *x,
            Self::Variable(i) => state.signature_variables.numeric_variables[*i],
            Self::ResourceVariable(i) => state.resource_variables[*i],
            Self::Cost => state.cost,
            Self::NumericOperation(op, a, b) => {
                let a = a.eval(state, metadata, registry);
                let b = b.eval(state, metadata, registry);
                Self::eval_operation(op, a, b)
            }
            Self::Cardinality(set_expression::SetExpression::SetVariable(i)) => {
                let set = &state.signature_variables.set_variables[*i];
                T::from(set.count_ones(..)).unwrap()
            }
            Self::Cardinality(set_expression::SetExpression::PermutationVariable(i)) => {
                let set = &state.signature_variables.permutation_variables[*i];
                T::from(set.len()).unwrap()
            }
            Self::Cardinality(set) => T::from(set.eval(state, metadata).count_ones(..)).unwrap(),
            Self::Table(t) => t.eval(state, metadata, registry),
        }
    }

    pub fn simplify(&self, registry: &table_registry::TableRegistry<T>) -> NumericExpression<T> {
        match self {
            Self::NumericOperation(op, a, b) => {
                match (a.simplify(registry), b.simplify(registry)) {
                    (NumericExpression::Constant(a), NumericExpression::Constant(b)) => {
                        NumericExpression::Constant(Self::eval_operation(op, a, b))
                    }
                    (a, b) => Self::NumericOperation(op.clone(), Box::new(a), Box::new(b)),
                }
            }
            Self::Table(expression) => match expression.simplify(registry) {
                numeric_table_expression::NumericTableExpression::Constant(value) => {
                    Self::Constant(value)
                }
                expression => Self::Table(expression),
            },
            _ => self.clone(),
        }
    }

    fn eval_operation(op: &NumericOperator, a: T, b: T) -> T {
        match op {
            NumericOperator::Add => a + b,
            NumericOperator::Subtract => a - b,
            NumericOperator::Multiply => a * b,
            NumericOperator::Divide => a / b,
            NumericOperator::Max => cmp::max(a, b),
            NumericOperator::Min => cmp::min(a, b),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state;
    use crate::table;
    use std::collections::HashMap;
    use std::rc::Rc;

    fn generate_metadata() -> state::StateMetadata {
        let object_names = vec!["object".to_string()];
        let object_numbers = vec![10];
        let mut name_to_object = HashMap::new();
        name_to_object.insert("object".to_string(), 0);

        let set_variable_names = vec![
            "s0".to_string(),
            "s1".to_string(),
            "s2".to_string(),
            "s3".to_string(),
        ];
        let mut name_to_set_variable = HashMap::new();
        name_to_set_variable.insert("s0".to_string(), 0);
        name_to_set_variable.insert("s1".to_string(), 1);
        name_to_set_variable.insert("s2".to_string(), 2);
        name_to_set_variable.insert("s3".to_string(), 3);
        let set_variable_to_object = vec![0, 0, 0, 0];

        let permutation_variable_names = vec![
            "p0".to_string(),
            "p1".to_string(),
            "p2".to_string(),
            "p3".to_string(),
        ];
        let mut name_to_permutation_variable = HashMap::new();
        name_to_permutation_variable.insert("p0".to_string(), 0);
        name_to_permutation_variable.insert("p1".to_string(), 1);
        name_to_permutation_variable.insert("p2".to_string(), 2);
        name_to_permutation_variable.insert("p3".to_string(), 3);
        let permutation_variable_to_object = vec![0, 0, 0, 0];

        let element_variable_names = vec![
            "e0".to_string(),
            "e1".to_string(),
            "e2".to_string(),
            "e3".to_string(),
        ];
        let mut name_to_element_variable = HashMap::new();
        name_to_element_variable.insert("e0".to_string(), 0);
        name_to_element_variable.insert("e1".to_string(), 1);
        name_to_element_variable.insert("e2".to_string(), 2);
        name_to_element_variable.insert("e3".to_string(), 3);
        let element_variable_to_object = vec![0, 0, 0, 0];

        let numeric_variable_names = vec![
            "n0".to_string(),
            "n1".to_string(),
            "n2".to_string(),
            "n3".to_string(),
        ];
        let mut name_to_numeric_variable = HashMap::new();
        name_to_numeric_variable.insert("n0".to_string(), 0);
        name_to_numeric_variable.insert("n1".to_string(), 1);
        name_to_numeric_variable.insert("n2".to_string(), 2);
        name_to_numeric_variable.insert("n3".to_string(), 3);

        let resource_variable_names = vec![
            "r0".to_string(),
            "r1".to_string(),
            "r2".to_string(),
            "r3".to_string(),
        ];
        let mut name_to_resource_variable = HashMap::new();
        name_to_resource_variable.insert("r0".to_string(), 0);
        name_to_resource_variable.insert("r1".to_string(), 1);
        name_to_resource_variable.insert("r2".to_string(), 2);
        name_to_resource_variable.insert("r3".to_string(), 3);

        state::StateMetadata {
            object_names,
            name_to_object,
            object_numbers,
            set_variable_names,
            name_to_set_variable,
            set_variable_to_object,
            permutation_variable_names,
            name_to_permutation_variable,
            permutation_variable_to_object,
            element_variable_names,
            name_to_element_variable,
            element_variable_to_object,
            numeric_variable_names,
            name_to_numeric_variable,
            resource_variable_names,
            name_to_resource_variable,
            less_is_better: vec![false, false, true, false],
        }
    }

    fn generate_state() -> state::State<variable::IntegerVariable> {
        let mut set1 = variable::SetVariable::with_capacity(3);
        set1.insert(0);
        set1.insert(2);
        let mut set2 = variable::SetVariable::with_capacity(3);
        set2.insert(0);
        set2.insert(1);
        state::State {
            signature_variables: Rc::new(state::SignatureVariables {
                set_variables: vec![set1, set2],
                permutation_variables: vec![vec![0, 2]],
                element_variables: vec![1],
                numeric_variables: vec![1, 2, 3],
            }),
            resource_variables: vec![4, 5, 6],
            stage: 0,
            cost: 0,
        }
    }

    fn generate_registry() -> table_registry::TableRegistry<variable::IntegerVariable> {
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

        table_registry::TableRegistry {
            numeric_tables: table_registry::TableData {
                tables_1d,
                name_to_table_1d,
                tables_2d,
                name_to_table_2d,
                tables_3d,
                name_to_table_3d,
                tables,
                name_to_table,
            },
            ..Default::default()
        }
    }

    #[test]
    fn number_eval() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let state = generate_state();
        let expression = NumericExpression::Constant(2);
        assert_eq!(expression.eval(&state, &metadata, &registry), 2);
    }

    #[test]
    fn numeric_variable_eval() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let state = generate_state();
        let expression = NumericExpression::Variable(0);
        assert_eq!(expression.eval(&state, &metadata, &registry), 1);
        let expression = NumericExpression::Variable(1);
        assert_eq!(expression.eval(&state, &metadata, &registry), 2);
        let expression = NumericExpression::Variable(2);
        assert_eq!(expression.eval(&state, &metadata, &registry), 3);
    }

    #[test]
    fn resource_variable_eval() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let state = generate_state();
        let expression = NumericExpression::ResourceVariable(0);
        assert_eq!(expression.eval(&state, &metadata, &registry), 4);
        let expression = NumericExpression::ResourceVariable(1);
        assert_eq!(expression.eval(&state, &metadata, &registry), 5);
        let expression = NumericExpression::ResourceVariable(2);
        assert_eq!(expression.eval(&state, &metadata, &registry), 6);
    }

    #[test]
    fn cost_eval() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let state = generate_state();
        let expression: NumericExpression<variable::IntegerVariable> = NumericExpression::Cost {};
        assert_eq!(expression.eval(&state, &metadata, &registry), 0);
    }

    #[test]
    fn add_eval() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let state = generate_state();
        let expression: NumericExpression<variable::IntegerVariable> =
            NumericExpression::NumericOperation(
                NumericOperator::Add,
                Box::new(NumericExpression::Constant(3)),
                Box::new(NumericExpression::Constant(2)),
            );
        assert_eq!(expression.eval(&state, &metadata, &registry), 5);
    }

    #[test]
    fn subtract_eval() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let state = generate_state();
        let expression: NumericExpression<variable::IntegerVariable> =
            NumericExpression::NumericOperation(
                NumericOperator::Subtract,
                Box::new(NumericExpression::Constant(3)),
                Box::new(NumericExpression::Constant(2)),
            );
        assert_eq!(expression.eval(&state, &metadata, &registry), 1);
    }

    #[test]
    fn multiply_eval() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let state = generate_state();
        let expression: NumericExpression<variable::IntegerVariable> =
            NumericExpression::NumericOperation(
                NumericOperator::Multiply,
                Box::new(NumericExpression::Constant(3)),
                Box::new(NumericExpression::Constant(2)),
            );
        assert_eq!(expression.eval(&state, &metadata, &registry), 6);
    }

    #[test]
    fn divide_eval() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let state = generate_state();
        let expression: NumericExpression<variable::IntegerVariable> =
            NumericExpression::NumericOperation(
                NumericOperator::Divide,
                Box::new(NumericExpression::Constant(3)),
                Box::new(NumericExpression::Constant(2)),
            );
        assert_eq!(expression.eval(&state, &metadata, &registry), 1);
    }

    #[test]
    fn max_eval() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let state = generate_state();
        let expression: NumericExpression<variable::IntegerVariable> =
            NumericExpression::NumericOperation(
                NumericOperator::Max,
                Box::new(NumericExpression::Constant(3)),
                Box::new(NumericExpression::Constant(2)),
            );
        assert_eq!(expression.eval(&state, &metadata, &registry), 3);
    }

    #[test]
    fn min_eval() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let state = generate_state();
        let expression: NumericExpression<variable::IntegerVariable> =
            NumericExpression::NumericOperation(
                NumericOperator::Min,
                Box::new(NumericExpression::Constant(3)),
                Box::new(NumericExpression::Constant(2)),
            );
        assert_eq!(expression.eval(&state, &metadata, &registry), 2);
    }

    #[test]
    fn cardinality_eval() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let state = generate_state();
        let expression =
            NumericExpression::Cardinality(set_expression::SetExpression::SetVariable(0));
        assert_eq!(expression.eval(&state, &metadata, &registry), 2);
        let expression =
            NumericExpression::Cardinality(set_expression::SetExpression::PermutationVariable(0));
        assert_eq!(expression.eval(&state, &metadata, &registry), 2);
        let expression = NumericExpression::Cardinality(set_expression::SetExpression::Complement(
            Box::new(set_expression::SetExpression::SetVariable(0)),
        ));
        assert_eq!(expression.eval(&state, &metadata, &registry), 1);
    }

    #[test]
    fn table_eval() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let state = generate_state();
        let expression =
            NumericExpression::Table(numeric_table_expression::NumericTableExpression::Table(
                0,
                vec![
                    set_expression::ElementExpression::Constant(0),
                    set_expression::ElementExpression::Constant(1),
                    set_expression::ElementExpression::Constant(0),
                    set_expression::ElementExpression::Constant(0),
                ],
            ));
        assert_eq!(expression.eval(&state, &metadata, &registry), 100);
        let expression =
            NumericExpression::Table(numeric_table_expression::NumericTableExpression::Table(
                0,
                vec![
                    set_expression::ElementExpression::Constant(0),
                    set_expression::ElementExpression::Constant(1),
                    set_expression::ElementExpression::Constant(0),
                    set_expression::ElementExpression::Constant(1),
                ],
            ));
        assert_eq!(expression.eval(&state, &metadata, &registry), 200);
        let expression =
            NumericExpression::Table(numeric_table_expression::NumericTableExpression::Table(
                0,
                vec![
                    set_expression::ElementExpression::Constant(0),
                    set_expression::ElementExpression::Constant(1),
                    set_expression::ElementExpression::Constant(2),
                    set_expression::ElementExpression::Constant(0),
                ],
            ));
        assert_eq!(expression.eval(&state, &metadata, &registry), 300);
        let expression =
            NumericExpression::Table(numeric_table_expression::NumericTableExpression::Table(
                0,
                vec![
                    set_expression::ElementExpression::Constant(0),
                    set_expression::ElementExpression::Constant(1),
                    set_expression::ElementExpression::Constant(2),
                    set_expression::ElementExpression::Constant(1),
                ],
            ));
        assert_eq!(expression.eval(&state, &metadata, &registry), 400);
    }

    #[test]
    fn number_simplify() {
        let registry = generate_registry();
        let expression = NumericExpression::Constant(2);
        assert!(matches!(
            expression.simplify(&registry),
            NumericExpression::Constant(2)
        ));
    }

    #[test]
    fn numeric_variable_simplify() {
        let registry = generate_registry();
        let expression = NumericExpression::Variable(0);
        assert!(matches!(
            expression.simplify(&registry),
            NumericExpression::Variable(0)
        ));
    }

    #[test]
    fn resource_variable_simplify() {
        let registry = generate_registry();
        let expression = NumericExpression::ResourceVariable(0);
        assert!(matches!(
            expression.simplify(&registry),
            NumericExpression::ResourceVariable(0)
        ));
    }

    #[test]
    fn cost_simplify() {
        let registry = generate_registry();
        let expression: NumericExpression<variable::IntegerVariable> = NumericExpression::Cost {};
        assert!(matches!(
            expression.simplify(&registry),
            NumericExpression::Cost
        ));
    }

    #[test]
    fn add_simplify() {
        let registry = generate_registry();

        let expression: NumericExpression<variable::IntegerVariable> =
            NumericExpression::NumericOperation(
                NumericOperator::Add,
                Box::new(NumericExpression::Constant(3)),
                Box::new(NumericExpression::Constant(2)),
            );
        assert!(matches!(
            expression.simplify(&registry),
            NumericExpression::Constant(5)
        ));

        let expression: NumericExpression<variable::IntegerVariable> =
            NumericExpression::NumericOperation(
                NumericOperator::Add,
                Box::new(NumericExpression::Variable(0)),
                Box::new(NumericExpression::Constant(2)),
            );
        let simplified = expression.simplify(&registry);
        assert!(matches!(
            simplified,
            NumericExpression::NumericOperation(NumericOperator::Add, _, _)
        ));
        if let NumericExpression::NumericOperation(_, a, b) = simplified {
            assert!(matches!(*a, NumericExpression::Variable(0)));
            assert!(matches!(*b, NumericExpression::Constant(2)));
        }
    }

    #[test]
    fn subtract_simplify() {
        let registry = generate_registry();
        let expression: NumericExpression<variable::IntegerVariable> =
            NumericExpression::NumericOperation(
                NumericOperator::Subtract,
                Box::new(NumericExpression::Constant(3)),
                Box::new(NumericExpression::Constant(2)),
            );
        assert!(matches!(
            expression.simplify(&registry),
            NumericExpression::Constant(1)
        ));

        let expression: NumericExpression<variable::IntegerVariable> =
            NumericExpression::NumericOperation(
                NumericOperator::Subtract,
                Box::new(NumericExpression::Variable(0)),
                Box::new(NumericExpression::Constant(2)),
            );
        let simplified = expression.simplify(&registry);
        assert!(matches!(
            simplified,
            NumericExpression::NumericOperation(NumericOperator::Subtract, _, _)
        ));
        if let NumericExpression::NumericOperation(_, a, b) = simplified {
            assert!(matches!(*a, NumericExpression::Variable(0)));
            assert!(matches!(*b, NumericExpression::Constant(2)));
        }
    }

    #[test]
    fn multiply_simplify() {
        let registry = generate_registry();
        let expression: NumericExpression<variable::IntegerVariable> =
            NumericExpression::NumericOperation(
                NumericOperator::Multiply,
                Box::new(NumericExpression::Constant(3)),
                Box::new(NumericExpression::Constant(2)),
            );
        assert!(matches!(
            expression.simplify(&registry),
            NumericExpression::Constant(6)
        ));

        let expression: NumericExpression<variable::IntegerVariable> =
            NumericExpression::NumericOperation(
                NumericOperator::Multiply,
                Box::new(NumericExpression::Variable(0)),
                Box::new(NumericExpression::Constant(2)),
            );
        let simplified = expression.simplify(&registry);
        assert!(matches!(
            simplified,
            NumericExpression::NumericOperation(NumericOperator::Multiply, _, _)
        ));
        if let NumericExpression::NumericOperation(_, a, b) = simplified {
            assert!(matches!(*a, NumericExpression::Variable(0)));
            assert!(matches!(*b, NumericExpression::Constant(2)));
        }
    }

    #[test]
    fn divide_simplify() {
        let registry = generate_registry();
        let expression: NumericExpression<variable::IntegerVariable> =
            NumericExpression::NumericOperation(
                NumericOperator::Divide,
                Box::new(NumericExpression::Constant(3)),
                Box::new(NumericExpression::Constant(2)),
            );
        assert!(matches!(
            expression.simplify(&registry),
            NumericExpression::Constant(1)
        ));

        let expression: NumericExpression<variable::IntegerVariable> =
            NumericExpression::NumericOperation(
                NumericOperator::Divide,
                Box::new(NumericExpression::Variable(0)),
                Box::new(NumericExpression::Constant(2)),
            );
        let simplified = expression.simplify(&registry);
        assert!(matches!(
            simplified,
            NumericExpression::NumericOperation(NumericOperator::Divide, _, _)
        ));
        if let NumericExpression::NumericOperation(_, a, b) = simplified {
            assert!(matches!(*a, NumericExpression::Variable(0)));
            assert!(matches!(*b, NumericExpression::Constant(2)));
        }
    }

    #[test]
    fn max_simplify() {
        let registry = generate_registry();
        let expression: NumericExpression<variable::IntegerVariable> =
            NumericExpression::NumericOperation(
                NumericOperator::Max,
                Box::new(NumericExpression::Constant(3)),
                Box::new(NumericExpression::Constant(2)),
            );
        assert!(matches!(
            expression.simplify(&registry),
            NumericExpression::Constant(3)
        ));

        let expression: NumericExpression<variable::IntegerVariable> =
            NumericExpression::NumericOperation(
                NumericOperator::Max,
                Box::new(NumericExpression::Variable(0)),
                Box::new(NumericExpression::Constant(2)),
            );
        let simplified = expression.simplify(&registry);
        assert!(matches!(
            simplified,
            NumericExpression::NumericOperation(NumericOperator::Max, _, _)
        ));
        if let NumericExpression::NumericOperation(_, a, b) = simplified {
            assert!(matches!(*a, NumericExpression::Variable(0)));
            assert!(matches!(*b, NumericExpression::Constant(2)));
        }
    }

    #[test]
    fn min_simplify() {
        let registry = generate_registry();
        let expression: NumericExpression<variable::IntegerVariable> =
            NumericExpression::NumericOperation(
                NumericOperator::Min,
                Box::new(NumericExpression::Constant(3)),
                Box::new(NumericExpression::Constant(2)),
            );
        assert!(matches!(
            expression.simplify(&registry),
            NumericExpression::Constant(2)
        ));

        let expression: NumericExpression<variable::IntegerVariable> =
            NumericExpression::NumericOperation(
                NumericOperator::Min,
                Box::new(NumericExpression::Variable(0)),
                Box::new(NumericExpression::Constant(2)),
            );
        let simplified = expression.simplify(&registry);
        assert!(matches!(
            simplified,
            NumericExpression::NumericOperation(NumericOperator::Min, _, _)
        ));
        if let NumericExpression::NumericOperation(_, a, b) = simplified {
            assert!(matches!(*a, NumericExpression::Variable(0)));
            assert!(matches!(*b, NumericExpression::Constant(2)));
        }
    }

    #[test]
    fn cardinality_simplify() {
        let registry = generate_registry();
        let expression =
            NumericExpression::Cardinality(set_expression::SetExpression::SetVariable(0));
        assert!(matches!(
            expression.simplify(&registry),
            NumericExpression::Cardinality(set_expression::SetExpression::SetVariable(0))
        ));
    }

    #[test]
    fn table_1d_simplify() {
        let registry = generate_registry();

        let expression =
            NumericExpression::Table(numeric_table_expression::NumericTableExpression::Table1D(
                0,
                set_expression::ElementExpression::Constant(0),
            ));
        assert!(matches!(
            expression.simplify(&registry),
            NumericExpression::Constant(10)
        ));

        let expression =
            NumericExpression::Table(numeric_table_expression::NumericTableExpression::Table1D(
                0,
                set_expression::ElementExpression::Variable(0),
            ));
        assert!(matches!(
            expression.simplify(&registry),
            NumericExpression::Table(numeric_table_expression::NumericTableExpression::Table1D(
                0,
                set_expression::ElementExpression::Variable(0)
            ))
        ));
    }

    #[test]
    fn table_1d_sum_simplify() {
        let registry = generate_registry();

        let expression = NumericExpression::Table(
            numeric_table_expression::NumericTableExpression::Table1DSum(
                0,
                set_expression::SetExpression::SetVariable(0),
            ),
        );
        assert!(matches!(
            expression.simplify(&registry),
            NumericExpression::Table(
                numeric_table_expression::NumericTableExpression::Table1DSum(
                    0,
                    set_expression::SetExpression::SetVariable(0),
                )
            )
        ));
    }

    #[test]
    fn table_2d_simplify() {
        let registry = generate_registry();

        let expression =
            NumericExpression::Table(numeric_table_expression::NumericTableExpression::Table2D(
                0,
                set_expression::ElementExpression::Constant(0),
                set_expression::ElementExpression::Constant(0),
            ));
        assert!(matches!(
            expression.simplify(&registry),
            NumericExpression::Constant(10)
        ));

        let expression =
            NumericExpression::Table(numeric_table_expression::NumericTableExpression::Table2D(
                0,
                set_expression::ElementExpression::Constant(0),
                set_expression::ElementExpression::Variable(0),
            ));
        assert!(matches!(
            expression.simplify(&registry),
            NumericExpression::Table(numeric_table_expression::NumericTableExpression::Table2D(
                0,
                set_expression::ElementExpression::Constant(0),
                set_expression::ElementExpression::Variable(0)
            ))
        ));
    }

    #[test]
    fn table_2d_sum_simplify() {
        let registry = generate_registry();

        let expression = NumericExpression::Table(
            numeric_table_expression::NumericTableExpression::Table2DSum(
                0,
                set_expression::SetExpression::SetVariable(0),
                set_expression::SetExpression::SetVariable(1),
            ),
        );
        assert!(matches!(
            expression.simplify(&registry),
            NumericExpression::Table(
                numeric_table_expression::NumericTableExpression::Table2DSum(
                    0,
                    set_expression::SetExpression::SetVariable(0),
                    set_expression::SetExpression::SetVariable(1),
                )
            )
        ));
    }

    #[test]
    fn table_2d_sum_x_simplify() {
        let registry = generate_registry();

        let expression = NumericExpression::Table(
            numeric_table_expression::NumericTableExpression::Table2DSumX(
                0,
                set_expression::SetExpression::SetVariable(0),
                set_expression::ElementExpression::Constant(0),
            ),
        );
        assert!(matches!(
            expression.simplify(&registry),
            NumericExpression::Table(
                numeric_table_expression::NumericTableExpression::Table2DSumX(
                    0,
                    set_expression::SetExpression::SetVariable(0),
                    set_expression::ElementExpression::Constant(0),
                )
            )
        ));
    }

    #[test]
    fn table_2d_sum_y_simplify() {
        let registry = generate_registry();

        let expression = NumericExpression::Table(
            numeric_table_expression::NumericTableExpression::Table2DSumY(
                0,
                set_expression::ElementExpression::Constant(0),
                set_expression::SetExpression::SetVariable(0),
            ),
        );
        assert!(matches!(
            expression.simplify(&registry),
            NumericExpression::Table(
                numeric_table_expression::NumericTableExpression::Table2DSumY(
                    0,
                    set_expression::ElementExpression::Constant(0),
                    set_expression::SetExpression::SetVariable(0),
                )
            )
        ));
    }

    #[test]
    fn table_3d_simplify() {
        let registry = generate_registry();

        let expression =
            NumericExpression::Table(numeric_table_expression::NumericTableExpression::Table3D(
                0,
                set_expression::ElementExpression::Constant(0),
                set_expression::ElementExpression::Constant(0),
                set_expression::ElementExpression::Constant(0),
            ));
        assert!(matches!(
            expression.simplify(&registry),
            NumericExpression::Constant(10)
        ));

        let expression =
            NumericExpression::Table(numeric_table_expression::NumericTableExpression::Table3D(
                0,
                set_expression::ElementExpression::Constant(0),
                set_expression::ElementExpression::Constant(0),
                set_expression::ElementExpression::Variable(0),
            ));
        assert!(matches!(
            expression.simplify(&registry),
            NumericExpression::Table(numeric_table_expression::NumericTableExpression::Table3D(
                0,
                set_expression::ElementExpression::Constant(0),
                set_expression::ElementExpression::Constant(0),
                set_expression::ElementExpression::Variable(0)
            ))
        ));
    }

    #[test]
    fn table_3d_sum_simplify() {
        let registry = generate_registry();

        let expression = NumericExpression::Table(
            numeric_table_expression::NumericTableExpression::Table3DSum(
                0,
                set_expression::SetExpression::SetVariable(0),
                set_expression::SetExpression::SetVariable(1),
                set_expression::SetExpression::SetVariable(2),
            ),
        );
        assert!(matches!(
            expression.simplify(&registry),
            NumericExpression::Table(
                numeric_table_expression::NumericTableExpression::Table3DSum(
                    0,
                    set_expression::SetExpression::SetVariable(0),
                    set_expression::SetExpression::SetVariable(1),
                    set_expression::SetExpression::SetVariable(2),
                )
            )
        ));
    }

    #[test]
    fn table_3d_sum_x_simplify() {
        let registry = generate_registry();

        let expression = NumericExpression::Table(
            numeric_table_expression::NumericTableExpression::Table3DSumX(
                0,
                set_expression::SetExpression::SetVariable(0),
                set_expression::ElementExpression::Constant(0),
                set_expression::ElementExpression::Constant(0),
            ),
        );
        assert!(matches!(
            expression.simplify(&registry),
            NumericExpression::Table(
                numeric_table_expression::NumericTableExpression::Table3DSumX(
                    0,
                    set_expression::SetExpression::SetVariable(0),
                    set_expression::ElementExpression::Constant(0),
                    set_expression::ElementExpression::Constant(0),
                )
            )
        ));
    }

    #[test]
    fn table_3d_sum_y_simplify() {
        let registry = generate_registry();

        let expression = NumericExpression::Table(
            numeric_table_expression::NumericTableExpression::Table3DSumY(
                0,
                set_expression::ElementExpression::Constant(0),
                set_expression::SetExpression::SetVariable(0),
                set_expression::ElementExpression::Constant(0),
            ),
        );
        assert!(matches!(
            expression.simplify(&registry),
            NumericExpression::Table(
                numeric_table_expression::NumericTableExpression::Table3DSumY(
                    0,
                    set_expression::ElementExpression::Constant(0),
                    set_expression::SetExpression::SetVariable(0),
                    set_expression::ElementExpression::Constant(0),
                )
            )
        ));
    }

    #[test]
    fn table_3d_sum_z_simplify() {
        let registry = generate_registry();

        let expression = NumericExpression::Table(
            numeric_table_expression::NumericTableExpression::Table3DSumZ(
                0,
                set_expression::ElementExpression::Constant(0),
                set_expression::ElementExpression::Constant(0),
                set_expression::SetExpression::SetVariable(0),
            ),
        );
        assert!(matches!(
            expression.simplify(&registry),
            NumericExpression::Table(
                numeric_table_expression::NumericTableExpression::Table3DSumZ(
                    0,
                    set_expression::ElementExpression::Constant(0),
                    set_expression::ElementExpression::Constant(0),
                    set_expression::SetExpression::SetVariable(0),
                )
            )
        ));
    }

    #[test]
    fn table_3d_sum_xy_simplify() {
        let registry = generate_registry();

        let expression = NumericExpression::Table(
            numeric_table_expression::NumericTableExpression::Table3DSumXY(
                0,
                set_expression::SetExpression::SetVariable(0),
                set_expression::SetExpression::SetVariable(0),
                set_expression::ElementExpression::Constant(0),
            ),
        );
        assert!(matches!(
            expression.simplify(&registry),
            NumericExpression::Table(
                numeric_table_expression::NumericTableExpression::Table3DSumXY(
                    0,
                    set_expression::SetExpression::SetVariable(0),
                    set_expression::SetExpression::SetVariable(0),
                    set_expression::ElementExpression::Constant(0),
                )
            )
        ));
    }

    #[test]
    fn table_3d_sum_xz_simplify() {
        let registry = generate_registry();

        let expression = NumericExpression::Table(
            numeric_table_expression::NumericTableExpression::Table3DSumXZ(
                0,
                set_expression::SetExpression::SetVariable(0),
                set_expression::ElementExpression::Constant(0),
                set_expression::SetExpression::SetVariable(0),
            ),
        );
        assert!(matches!(
            expression.simplify(&registry),
            NumericExpression::Table(
                numeric_table_expression::NumericTableExpression::Table3DSumXZ(
                    0,
                    set_expression::SetExpression::SetVariable(0),
                    set_expression::ElementExpression::Constant(0),
                    set_expression::SetExpression::SetVariable(0),
                )
            )
        ));
    }

    #[test]
    fn table_3d_sum_yz_simplify() {
        let registry = generate_registry();

        let expression = NumericExpression::Table(
            numeric_table_expression::NumericTableExpression::Table3DSumYZ(
                0,
                set_expression::ElementExpression::Constant(0),
                set_expression::SetExpression::SetVariable(0),
                set_expression::SetExpression::SetVariable(0),
            ),
        );
        assert!(matches!(
            expression.simplify(&registry),
            NumericExpression::Table(
                numeric_table_expression::NumericTableExpression::Table3DSumYZ(
                    0,
                    set_expression::ElementExpression::Constant(0),
                    set_expression::SetExpression::SetVariable(0),
                    set_expression::SetExpression::SetVariable(0),
                )
            )
        ));
    }

    #[test]
    fn table_simplify() {
        let registry = generate_registry();

        let expression =
            NumericExpression::Table(numeric_table_expression::NumericTableExpression::Table(
                0,
                vec![
                    set_expression::ElementExpression::Constant(0),
                    set_expression::ElementExpression::Constant(1),
                    set_expression::ElementExpression::Constant(0),
                    set_expression::ElementExpression::Constant(0),
                ],
            ));
        assert!(matches!(
            expression.simplify(&registry),
            NumericExpression::Constant(100)
        ));

        let expression =
            NumericExpression::Table(numeric_table_expression::NumericTableExpression::Table(
                0,
                vec![
                    set_expression::ElementExpression::Constant(0),
                    set_expression::ElementExpression::Constant(1),
                    set_expression::ElementExpression::Constant(0),
                    set_expression::ElementExpression::Variable(0),
                ],
            ));
        let simplified = expression.simplify(&registry);
        assert!(matches!(
            simplified,
            NumericExpression::Table(numeric_table_expression::NumericTableExpression::Table(0, _))
        ));
        if let NumericExpression::Table(numeric_table_expression::NumericTableExpression::Table(
            _,
            args,
        )) = simplified
        {
            assert_eq!(args.len(), 4);
            assert!(matches!(
                args[0],
                set_expression::ElementExpression::Constant(0)
            ));
            assert!(matches!(
                args[1],
                set_expression::ElementExpression::Constant(1)
            ));
            assert!(matches!(
                args[2],
                set_expression::ElementExpression::Constant(0)
            ));
            assert!(matches!(
                args[3],
                set_expression::ElementExpression::Variable(0)
            ));
        }
    }
    #[test]
    fn table_sum_simplify() {
        let registry = generate_registry();

        let expression =
            NumericExpression::Table(numeric_table_expression::NumericTableExpression::TableSum(
                0,
                vec![
                    set_expression::ArgumentExpression::Element(
                        set_expression::ElementExpression::Constant(0),
                    ),
                    set_expression::ArgumentExpression::Element(
                        set_expression::ElementExpression::Constant(1),
                    ),
                    set_expression::ArgumentExpression::Element(
                        set_expression::ElementExpression::Constant(0),
                    ),
                    set_expression::ArgumentExpression::Element(
                        set_expression::ElementExpression::Constant(0),
                    ),
                ],
            ));
        assert!(matches!(
            expression.simplify(&registry),
            NumericExpression::Constant(100)
        ));

        let expression =
            NumericExpression::Table(numeric_table_expression::NumericTableExpression::TableSum(
                0,
                vec![
                    set_expression::ArgumentExpression::Element(
                        set_expression::ElementExpression::Constant(0),
                    ),
                    set_expression::ArgumentExpression::Element(
                        set_expression::ElementExpression::Constant(1),
                    ),
                    set_expression::ArgumentExpression::Element(
                        set_expression::ElementExpression::Constant(0),
                    ),
                    set_expression::ArgumentExpression::Element(
                        set_expression::ElementExpression::Variable(0),
                    ),
                ],
            ));
        let simplified = expression.simplify(&registry);
        assert!(matches!(
            simplified,
            NumericExpression::Table(numeric_table_expression::NumericTableExpression::TableSum(
                0,
                _,
            ))
        ));
        if let NumericExpression::Table(
            numeric_table_expression::NumericTableExpression::TableSum(_, args),
        ) = simplified
        {
            assert_eq!(args.len(), 4);
            assert!(matches!(
                args[0],
                set_expression::ArgumentExpression::Element(
                    set_expression::ElementExpression::Constant(0)
                )
            ));
            assert!(matches!(
                args[1],
                set_expression::ArgumentExpression::Element(
                    set_expression::ElementExpression::Constant(1)
                )
            ));
            assert!(matches!(
                args[2],
                set_expression::ArgumentExpression::Element(
                    set_expression::ElementExpression::Constant(0)
                )
            ));
            assert!(matches!(
                args[3],
                set_expression::ArgumentExpression::Element(
                    set_expression::ElementExpression::Variable(0)
                )
            ));
        }
    }
}
