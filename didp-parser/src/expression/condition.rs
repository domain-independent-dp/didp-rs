use super::numeric_expression::NumericExpression;
use super::set_condition;
use crate::state;
use crate::variable;

#[derive(Debug)]
pub enum Condition<'a, T: variable::Numeric> {
    Not(Box<Condition<'a, T>>),
    And(Box<Condition<'a, T>>, Box<Condition<'a, T>>),
    Or(Box<Condition<'a, T>>, Box<Condition<'a, T>>),
    Comparison(
        ComparisonOperator,
        NumericExpression<'a, T>,
        NumericExpression<'a, T>,
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

impl<'a, T: variable::Numeric> Condition<'a, T> {
    pub fn eval(&self, state: &state::State<T>, metadata: &state::StateMetadata) -> bool {
        match self {
            Condition::Not(c) => !c.eval(state, metadata),
            Condition::And(x, y) => x.eval(state, metadata) && y.eval(state, metadata),
            Condition::Or(x, y) => x.eval(state, metadata) || y.eval(state, metadata),
            Condition::Comparison(op, x, y) => {
                Self::eval_comparison(op, x.eval(state, metadata), y.eval(state, metadata))
            }
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
    use crate::state;
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

    #[test]
    fn eval_eq() {
        let metadata = generate_metadata();
        let state = generate_state();

        let expression = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        assert!(expression.eval(&state, &metadata));

        let expression = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        assert!(!expression.eval(&state, &metadata));
    }

    #[test]
    fn eval_neq() {
        let metadata = generate_metadata();
        let state = generate_state();

        let expression = Condition::Comparison(
            ComparisonOperator::Ne,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        assert!(!expression.eval(&state, &metadata));

        let expression = Condition::Comparison(
            ComparisonOperator::Ne,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        assert!(expression.eval(&state, &metadata));
    }

    #[test]
    fn eval_ge() {
        let metadata = generate_metadata();
        let state = generate_state();

        let expression = Condition::Comparison(
            ComparisonOperator::Ge,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        assert!(expression.eval(&state, &metadata));

        let expression = Condition::Comparison(
            ComparisonOperator::Ge,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        assert!(!expression.eval(&state, &metadata));

        let expression = Condition::Comparison(
            ComparisonOperator::Ge,
            NumericExpression::Constant(1),
            NumericExpression::Constant(0),
        );
        assert!(expression.eval(&state, &metadata));
    }

    #[test]
    fn eval_gt() {
        let metadata = generate_metadata();
        let state = generate_state();

        let expression = Condition::Comparison(
            ComparisonOperator::Gt,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        assert!(!expression.eval(&state, &metadata));

        let expression = Condition::Comparison(
            ComparisonOperator::Gt,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        assert!(!expression.eval(&state, &metadata));

        let expression = Condition::Comparison(
            ComparisonOperator::Gt,
            NumericExpression::Constant(1),
            NumericExpression::Constant(0),
        );
        assert!(expression.eval(&state, &metadata));
    }

    #[test]
    fn eval_le() {
        let metadata = generate_metadata();
        let state = generate_state();

        let expression = Condition::Comparison(
            ComparisonOperator::Le,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        assert!(expression.eval(&state, &metadata));

        let expression = Condition::Comparison(
            ComparisonOperator::Le,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        assert!(expression.eval(&state, &metadata));

        let expression = Condition::Comparison(
            ComparisonOperator::Le,
            NumericExpression::Constant(1),
            NumericExpression::Constant(0),
        );
        assert!(!expression.eval(&state, &metadata));
    }

    #[test]
    fn eval_lt() {
        let metadata = generate_metadata();
        let state = generate_state();

        let expression = Condition::Comparison(
            ComparisonOperator::Lt,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        assert!(!expression.eval(&state, &metadata));

        let expression = Condition::Comparison(
            ComparisonOperator::Lt,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        assert!(expression.eval(&state, &metadata));

        let expression = Condition::Comparison(
            ComparisonOperator::Lt,
            NumericExpression::Constant(1),
            NumericExpression::Constant(0),
        );
        assert!(!expression.eval(&state, &metadata));
    }

    #[test]
    fn eval_not() {
        let metadata = generate_metadata();
        let state = generate_state();

        let expression = Condition::Not(Box::new(Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        )));
        assert!(!expression.eval(&state, &metadata));

        let expression = Condition::Not(Box::new(Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        )));
        assert!(expression.eval(&state, &metadata));
    }

    #[test]
    fn eval_and() {
        let metadata = generate_metadata();
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
        assert!(expression.eval(&state, &metadata));

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
        assert!(!expression.eval(&state, &metadata));

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
        assert!(!expression.eval(&state, &metadata));
    }

    #[test]
    fn eval_or() {
        let metadata = generate_metadata();
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
        assert!(expression.eval(&state, &metadata));

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
        assert!(expression.eval(&state, &metadata));

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
        assert!(!expression.eval(&state, &metadata));
    }
}
