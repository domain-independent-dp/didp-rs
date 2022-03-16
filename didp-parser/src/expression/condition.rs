use super::numeric_expression::NumericExpression;
use super::set_condition;
use crate::state;
use crate::table_registry;
use crate::variable;

#[derive(Debug)]
pub enum Condition<T: variable::Numeric> {
    Not(Box<Condition<T>>),
    And(Box<Condition<T>>, Box<Condition<T>>),
    Or(Box<Condition<T>>, Box<Condition<T>>),
    Comparison(
        ComparisonOperator,
        NumericExpression<T>,
        NumericExpression<T>,
    ),
    Set(set_condition::SetCondition),
}

#[derive(Debug)]
pub enum ComparisonOperator {
    Eq,
    Ne,
    Ge,
    Gt,
    Le,
    Lt,
}

impl<T: variable::Numeric> Condition<T> {
    pub fn eval(
        &self,
        state: &state::State<T>,
        metadata: &state::StateMetadata,
        registry: &table_registry::TableRegistry<T>,
    ) -> bool {
        match self {
            Condition::Not(c) => !c.eval(state, metadata, registry),
            Condition::And(x, y) => {
                x.eval(state, metadata, registry) && y.eval(state, metadata, registry)
            }
            Condition::Or(x, y) => {
                x.eval(state, metadata, registry) || y.eval(state, metadata, registry)
            }
            Condition::Comparison(op, x, y) => Self::eval_comparison(
                op,
                x.eval(state, metadata, registry),
                y.eval(state, metadata, registry),
            ),
            Condition::Set(c) => c.eval(state, metadata),
        }
    }

    fn eval_comparison(op: &ComparisonOperator, x: T, y: T) -> bool {
        match op {
            ComparisonOperator::Eq => x == y,
            ComparisonOperator::Ne => x != y,
            ComparisonOperator::Ge => x >= y,
            ComparisonOperator::Gt => x > y,
            ComparisonOperator::Le => x <= y,
            ComparisonOperator::Lt => x < y,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

    fn generate_registry() -> table_registry::TableRegistry<variable::IntegerVariable> {
        let tables_1d = vec![table::Table1D::new(Vec::new())];
        let mut name_to_table_1d = HashMap::new();
        name_to_table_1d.insert(String::from("f1"), 0);

        let tables_2d = vec![table::Table2D::new(Vec::new())];
        let mut name_to_table_2d = HashMap::new();
        name_to_table_2d.insert(String::from("f2"), 0);

        let tables_3d = vec![table::Table3D::new(Vec::new())];
        let mut name_to_table_3d = HashMap::new();
        name_to_table_3d.insert(String::from("f3"), 0);

        let tables = vec![table::Table::new(HashMap::new(), 0)];
        let mut name_to_table = HashMap::new();
        name_to_table.insert(String::from("f4"), 0);

        table_registry::TableRegistry {
            tables_1d,
            name_to_table_1d,
            tables_2d,
            name_to_table_2d,
            tables_3d,
            name_to_table_3d,
            tables,
            name_to_table,
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

    #[test]
    fn eval_eq() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let state = generate_state();

        let expression = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        assert!(expression.eval(&state, &metadata, &registry));

        let expression = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        assert!(!expression.eval(&state, &metadata, &registry));
    }

    #[test]
    fn eval_neq() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let state = generate_state();

        let expression = Condition::Comparison(
            ComparisonOperator::Ne,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        assert!(!expression.eval(&state, &metadata, &registry));

        let expression = Condition::Comparison(
            ComparisonOperator::Ne,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        assert!(expression.eval(&state, &metadata, &registry));
    }

    #[test]
    fn eval_ge() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let state = generate_state();

        let expression = Condition::Comparison(
            ComparisonOperator::Ge,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        assert!(expression.eval(&state, &metadata, &registry));

        let expression = Condition::Comparison(
            ComparisonOperator::Ge,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        assert!(!expression.eval(&state, &metadata, &registry));

        let expression = Condition::Comparison(
            ComparisonOperator::Ge,
            NumericExpression::Constant(1),
            NumericExpression::Constant(0),
        );
        assert!(expression.eval(&state, &metadata, &registry));
    }

    #[test]
    fn eval_gt() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let state = generate_state();

        let expression = Condition::Comparison(
            ComparisonOperator::Gt,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        assert!(!expression.eval(&state, &metadata, &registry));

        let expression = Condition::Comparison(
            ComparisonOperator::Gt,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        assert!(!expression.eval(&state, &metadata, &registry));

        let expression = Condition::Comparison(
            ComparisonOperator::Gt,
            NumericExpression::Constant(1),
            NumericExpression::Constant(0),
        );
        assert!(expression.eval(&state, &metadata, &registry));
    }

    #[test]
    fn eval_le() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let state = generate_state();

        let expression = Condition::Comparison(
            ComparisonOperator::Le,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        assert!(expression.eval(&state, &metadata, &registry));

        let expression = Condition::Comparison(
            ComparisonOperator::Le,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        assert!(expression.eval(&state, &metadata, &registry));

        let expression = Condition::Comparison(
            ComparisonOperator::Le,
            NumericExpression::Constant(1),
            NumericExpression::Constant(0),
        );
        assert!(!expression.eval(&state, &metadata, &registry));
    }

    #[test]
    fn eval_lt() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let state = generate_state();

        let expression = Condition::Comparison(
            ComparisonOperator::Lt,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        assert!(!expression.eval(&state, &metadata, &registry));

        let expression = Condition::Comparison(
            ComparisonOperator::Lt,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        assert!(expression.eval(&state, &metadata, &registry));

        let expression = Condition::Comparison(
            ComparisonOperator::Lt,
            NumericExpression::Constant(1),
            NumericExpression::Constant(0),
        );
        assert!(!expression.eval(&state, &metadata, &registry));
    }

    #[test]
    fn eval_not() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let state = generate_state();

        let expression = Condition::Not(Box::new(Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        )));
        assert!(!expression.eval(&state, &metadata, &registry));

        let expression = Condition::Not(Box::new(Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        )));
        assert!(expression.eval(&state, &metadata, &registry));
    }

    #[test]
    fn eval_and() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let state = generate_state();

        let x = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        let y = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        let expression = Condition::And(Box::new(x), Box::new(y));
        assert!(expression.eval(&state, &metadata, &registry));

        let x = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        let y = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        let expression = Condition::And(Box::new(x), Box::new(y));
        assert!(!expression.eval(&state, &metadata, &registry));

        let x = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        let y = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        let expression = Condition::And(Box::new(x), Box::new(y));
        assert!(!expression.eval(&state, &metadata, &registry));
    }

    #[test]
    fn eval_or() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let state = generate_state();

        let x = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        let y = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        let expression = Condition::Or(Box::new(x), Box::new(y));
        assert!(expression.eval(&state, &metadata, &registry));

        let x = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        let y = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        let expression = Condition::Or(Box::new(x), Box::new(y));
        assert!(expression.eval(&state, &metadata, &registry));

        let x = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        let y = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        let expression = Condition::Or(Box::new(x), Box::new(y));
        assert!(!expression.eval(&state, &metadata, &registry));
    }
}
