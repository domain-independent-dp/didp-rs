use crate::expression;
use crate::problem;
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
    pub fn is_applicable(&self, state: &state::State<T>, problem: &problem::Problem) -> bool {
        for c in &self.preconditions {
            if !c.eval(state, problem) {
                return false;
            }
        }
        true
    }

    pub fn apply_effects(
        &self,
        state: &state::State<T>,
        problem: &problem::Problem,
    ) -> state::State<T> {
        let len = state.signature_variables.set_variables.len();
        let mut set_variables = Vec::with_capacity(len);
        let mut i = 0;

        for e in &self.set_effects {
            while i < e.0 {
                set_variables.push(state.signature_variables.set_variables[i].clone());
                i += 1;
            }
            set_variables.push(e.1.eval(state, problem));
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
            numeric_variables[e.0] = e.1.eval(state, problem);
        }

        let mut resource_variables = state.resource_variables.clone();
        for e in &self.resource_effects {
            resource_variables[e.0] = e.1.eval(state, problem);
        }

        let cost = self.cost.eval(state, problem);

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
            cost,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use expression::*;

    fn generate_problem() -> problem::Problem {
        problem::Problem {
            set_variable_to_max_size: vec![3, 3],
            permutation_variable_to_max_length: vec![3, 3],
            element_to_set: vec![0, 2],
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
            cost: 0,
        }
    }

    #[test]
    fn applicable() {
        let state = generate_state();
        let problem = generate_problem();
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
        assert!(operator.is_applicable(&state, &problem));
    }

    #[test]
    fn not_applicable() {
        let state = generate_state();
        let problem = generate_problem();
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
        assert!(operator.is_applicable(&state, &problem));
    }

    #[test]
    fn appy_effects() {
        let state = generate_state();
        let problem = generate_problem();
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
            cost: 1,
        };
        let successor = operator.apply_effects(&state, &problem);
        assert_eq!(successor, expected);
    }
}
