use crate::expression;
use crate::state;
use crate::variable;
use std::rc::Rc;

pub struct Operator<'a, T: variable::Numeric> {
    pub preconditions: Vec<expression::Condition<'a, T>>,
    pub set_effects: Vec<(usize, expression::SetExpression)>,
    pub permutation_effects: Vec<(usize, expression::ElementExpression)>,
    pub element_effects: Vec<(usize, expression::ElementExpression)>,
    pub numeric_effects: Vec<(usize, expression::NumericExpression<'a, T>)>,
    pub resource_effects: Vec<(usize, expression::NumericExpression<'a, T>)>,
    pub cost: expression::NumericExpression<'a, T>,
}

impl<'a, T: variable::Numeric> Operator<'a, T> {
    pub fn is_applicable(&self, state: &state::State<T>, metadata: &state::StateMetadata) -> bool {
        for c in &self.preconditions {
            if !c.eval(state, metadata) {
                return false;
            }
        }
        true
    }

    pub fn apply_effects(
        &self,
        state: &state::State<T>,
        metadata: &state::StateMetadata,
    ) -> state::State<T> {
        let len = state.signature_variables.set_variables.len();
        let mut set_variables = Vec::with_capacity(len);
        let mut i = 0;

        for e in &self.set_effects {
            while i < e.0 {
                set_variables.push(state.signature_variables.set_variables[i].clone());
                i += 1;
            }
            set_variables.push(e.1.eval(state, metadata));
            i += 1;
        }
        while i < len {
            set_variables.push(state.signature_variables.set_variables[i].clone());
            i += 1;
        }

        let mut permutation_variables = state.signature_variables.permutation_variables.clone();
        for e in &self.permutation_effects {
            permutation_variables[e.0].push(e.1.eval(state));
        }

        let mut element_variables = state.signature_variables.element_variables.clone();
        for e in &self.element_effects {
            element_variables[e.0] = e.1.eval(state);
        }

        let mut numeric_variables = state.signature_variables.numeric_variables.clone();
        for e in &self.numeric_effects {
            numeric_variables[e.0] = e.1.eval(state, metadata);
        }

        let mut resource_variables = state.resource_variables.clone();
        for e in &self.resource_effects {
            resource_variables[e.0] = e.1.eval(state, metadata);
        }

        let stage = state.stage + 1;
        let cost = self.cost.eval(state, metadata);

        state::State {
            signature_variables: {
                Rc::new(state::SignatureVariables {
                    set_variables,
                    permutation_variables,
                    element_variables,
                    numeric_variables,
                })
            },
            resource_variables,
            stage,
            cost,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use expression::*;
    use std::collections::HashMap;

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
                permutation_variables: vec![vec![0, 2], vec![1, 2]],
                element_variables: vec![1, 2],
                numeric_variables: vec![1, 2, 3],
            }),
            resource_variables: vec![4, 5, 6],
            stage: 0,
            cost: 0,
        }
    }

    #[test]
    fn applicable() {
        let state = generate_state();
        let metadata = generate_metadata();
        let set_condition = Condition::Set(SetCondition::IsIn(
            ElementExpression::Constant(0),
            SetExpression::SetVariable(0),
        ));
        let numeric_condition = Condition::Comparison(
            ComparisonOperator::Ge,
            NumericExpression::Variable(0),
            NumericExpression::Constant(1),
        );
        let operator = Operator {
            preconditions: vec![set_condition, numeric_condition],
            set_effects: Vec::new(),
            permutation_effects: Vec::new(),
            element_effects: Vec::new(),
            numeric_effects: Vec::new(),
            resource_effects: Vec::new(),
            cost: NumericExpression::Constant(0),
        };
        assert!(operator.is_applicable(&state, &metadata));
    }

    #[test]
    fn not_applicable() {
        let state = generate_state();
        let metadata = generate_metadata();
        let set_condition = Condition::Set(SetCondition::IsIn(
            ElementExpression::Constant(0),
            SetExpression::SetVariable(0),
        ));
        let numeric_condition = Condition::Comparison(
            ComparisonOperator::Le,
            NumericExpression::Variable(0),
            NumericExpression::Constant(1),
        );
        let operator = Operator {
            preconditions: vec![set_condition, numeric_condition],
            set_effects: Vec::new(),
            permutation_effects: Vec::new(),
            element_effects: Vec::new(),
            numeric_effects: Vec::new(),
            resource_effects: Vec::new(),
            cost: NumericExpression::Constant(0),
        };
        assert!(operator.is_applicable(&state, &metadata));
    }

    #[test]
    fn appy_effects() {
        let state = generate_state();
        let metadata = generate_metadata();
        let set_effect1 = SetExpression::SetElementOperation(
            SetElementOperator::Add,
            Box::new(SetExpression::SetVariable(0)),
            ElementExpression::Constant(1),
        );
        let set_effect2 = SetExpression::SetElementOperation(
            SetElementOperator::Remove,
            Box::new(SetExpression::SetVariable(1)),
            ElementExpression::Constant(0),
        );
        let permutation_effect1 = ElementExpression::Constant(1);
        let permutation_effect2 = ElementExpression::Constant(0);
        let element_effect1 = ElementExpression::Constant(2);
        let element_effect2 = ElementExpression::Constant(1);
        let numeric_effect1 = NumericExpression::NumericOperation(
            NumericOperator::Subtract,
            Box::new(NumericExpression::Variable(0)),
            Box::new(NumericExpression::Constant(1)),
        );
        let numeric_effect2 = NumericExpression::NumericOperation(
            NumericOperator::Multiply,
            Box::new(NumericExpression::Variable(1)),
            Box::new(NumericExpression::Constant(2)),
        );
        let resource_effect1 = NumericExpression::NumericOperation(
            NumericOperator::Add,
            Box::new(NumericExpression::ResourceVariable(0)),
            Box::new(NumericExpression::Constant(1)),
        );
        let resource_effect2 = NumericExpression::NumericOperation(
            NumericOperator::Divide,
            Box::new(NumericExpression::ResourceVariable(1)),
            Box::new(NumericExpression::Constant(2)),
        );
        let operator = Operator {
            preconditions: Vec::new(),
            set_effects: vec![(0, set_effect1), (1, set_effect2)],
            permutation_effects: vec![(0, permutation_effect1), (1, permutation_effect2)],
            element_effects: vec![(0, element_effect1), (1, element_effect2)],
            numeric_effects: vec![(0, numeric_effect1), (1, numeric_effect2)],
            resource_effects: vec![(0, resource_effect1), (1, resource_effect2)],
            cost: NumericExpression::NumericOperation(
                NumericOperator::Add,
                Box::new(NumericExpression::Cost),
                Box::new(NumericExpression::Constant(1)),
            ),
        };

        let mut set1 = variable::SetVariable::with_capacity(3);
        set1.insert(0);
        set1.insert(1);
        set1.insert(2);
        let mut set2 = variable::SetVariable::with_capacity(3);
        set2.insert(1);
        let expected = state::State {
            signature_variables: Rc::new(state::SignatureVariables {
                set_variables: vec![set1, set2],
                permutation_variables: vec![vec![0, 2, 1], vec![1, 2, 0]],
                element_variables: vec![2, 1],
                numeric_variables: vec![0, 4, 3],
            }),
            resource_variables: vec![5, 2, 6],
            stage: 1,
            cost: 1,
        };
        let successor = operator.apply_effects(&state, &metadata);
        assert_eq!(successor, expected);
    }
}
