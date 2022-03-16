use super::numeric_table_expression;
use super::set_expression;
use crate::state;
use crate::table_registry;
use crate::variable;
use std::boxed::Box;
use std::cmp;

#[derive(Debug)]
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
    NumericTable(numeric_table_expression::NumericTableExpression),
}

#[derive(Debug)]
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
            NumericExpression::Constant(x) => *x,
            NumericExpression::Variable(i) => state.signature_variables.numeric_variables[*i],
            NumericExpression::ResourceVariable(i) => state.resource_variables[*i],
            NumericExpression::Cost => state.cost,
            NumericExpression::NumericOperation(op, a, b) => {
                let a = a.eval(state, metadata, registry);
                let b = b.eval(state, metadata, registry);
                match op {
                    NumericOperator::Add => a + b,
                    NumericOperator::Subtract => a - b,
                    NumericOperator::Multiply => a * b,
                    NumericOperator::Divide => a / b,
                    NumericOperator::Max => cmp::max(a, b),
                    NumericOperator::Min => cmp::min(a, b),
                }
            }
            NumericExpression::Cardinality(set) => {
                T::from(set.eval(state, metadata).count_ones(..)).unwrap()
            }
            NumericExpression::NumericTable(f) => f.eval(state, metadata, registry),
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
            NumericExpression::Cardinality(set_expression::SetExpression::SetVariable(1));
        assert_eq!(expression.eval(&state, &metadata, &registry), 2);
    }
}
