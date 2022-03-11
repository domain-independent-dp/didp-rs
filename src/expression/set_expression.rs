use crate::problem;
use crate::state;
use crate::variable;

#[derive(Debug)]
pub enum SetExpression {
    SetVariable(usize),
    PermutationVariable(usize),
    Complement(Box<SetExpression>),
    SetOperation(SetOperator, Box<SetExpression>, Box<SetExpression>),
    SetElementOperation(SetElementOperator, Box<SetExpression>, ElementExpression),
}

#[derive(Debug)]
pub enum SetOperator {
    Union,
    Difference,
    Intersect,
}

#[derive(Debug)]
pub enum SetElementOperator {
    Add,
    Remove,
}

impl SetExpression {
    pub fn eval<T: variable::Numeric>(
        &self,
        state: &state::State<T>,
        problem: &problem::Problem,
    ) -> variable::SetVariable {
        match self {
            SetExpression::SetVariable(i) => state.signature_variables.set_variables[*i].clone(),
            SetExpression::PermutationVariable(i) => {
                let mut set = variable::SetVariable::with_capacity(
                    problem.permutation_variable_to_max_length[*i],
                );
                for v in &state.signature_variables.permutation_variables[*i] {
                    set.insert(*v);
                }
                set
            }
            SetExpression::Complement(s) => {
                let mut s = s.eval(&state, problem);
                s.toggle_range(..);
                s
            }
            SetExpression::SetOperation(op, a, b) => {
                let mut a = a.eval(&state, problem);
                let b = b.eval(&state, problem);
                match op {
                    SetOperator::Union => {
                        a.union_with(&b);
                        a
                    }
                    SetOperator::Difference => {
                        a.difference_with(&b);
                        a
                    }
                    SetOperator::Intersect => {
                        a.intersect_with(&b);
                        a
                    }
                }
            }
            SetExpression::SetElementOperation(op, s, e) => {
                let mut s = s.eval(&state, problem);
                let e = e.eval(&state);
                match op {
                    SetElementOperator::Add => {
                        s.insert(e);
                        s
                    }
                    SetElementOperator::Remove => {
                        s.set(e, false);
                        s
                    }
                }
            }
        }
    }
}

#[derive(Debug)]
pub enum ElementExpression {
    Constant(variable::ElementVariable),
    Variable(usize),
}

impl ElementExpression {
    pub fn eval<T: variable::Numeric>(&self, state: &state::State<T>) -> variable::ElementVariable {
        match self {
            ElementExpression::Constant(x) => *x,
            ElementExpression::Variable(i) => state.signature_variables.element_variables[*i],
        }
    }
}

#[derive(Debug)]
pub enum ArgumentExpression {
    Set(SetExpression),
    Element(ElementExpression),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::rc::Rc;

    fn generate_problem() -> problem::Problem {
        problem::Problem {
            set_variable_to_max_size: vec![3],
            permutation_variable_to_max_length: vec![3],
            element_to_set: vec![0],
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
            resource_variables: state::ResourceVariables {
                numeric_variables: vec![4, 5, 6],
            },
            cost: 0,
        }
    }
    #[test]
    fn element_number_eval() {
        let state = generate_state();
        let expression = ElementExpression::Constant(2);
        assert_eq!(expression.eval(&state), 2);
    }

    #[test]
    fn element_variable_eval() {
        let state = generate_state();
        let expression = ElementExpression::Variable(0);
        assert_eq!(expression.eval(&state), 1);
    }

    #[test]
    fn set_variable_eval() {
        let problem = generate_problem();
        let state = generate_state();
        let expression = SetExpression::SetVariable(0);
        assert_eq!(
            expression.eval(&state, &problem),
            state.signature_variables.set_variables[0]
        );
        let expression = SetExpression::SetVariable(1);
        assert_eq!(
            expression.eval(&state, &problem),
            state.signature_variables.set_variables[1]
        );
    }

    #[test]
    fn permutation_variable_eval() {
        let problem = generate_problem();
        let state = generate_state();
        let expression = SetExpression::PermutationVariable(0);
        let mut set = variable::SetVariable::with_capacity(3);
        set.insert(0);
        set.insert(2);
        assert_eq!(expression.eval(&state, &problem), set);
    }

    #[test]
    fn complement_eval() {
        let problem = generate_problem();
        let state = generate_state();
        let expression = SetExpression::Complement(Box::new(SetExpression::SetVariable(0)));
        let mut set = variable::SetVariable::with_capacity(3);
        set.insert(1);
        assert_eq!(expression.eval(&state, &problem), set);
    }

    #[test]
    fn union_eval() {
        let problem = generate_problem();
        let state = generate_state();
        let expression = SetExpression::SetOperation(
            SetOperator::Union,
            Box::new(SetExpression::SetVariable(0)),
            Box::new(SetExpression::SetVariable(1)),
        );
        let mut set = variable::SetVariable::with_capacity(3);
        set.insert(0);
        set.insert(1);
        set.insert(2);
        assert_eq!(expression.eval(&state, &problem), set);
        let expression = SetExpression::SetOperation(
            SetOperator::Union,
            Box::new(SetExpression::SetVariable(0)),
            Box::new(SetExpression::SetVariable(0)),
        );
        assert_eq!(
            expression.eval(&state, &problem),
            state.signature_variables.set_variables[0]
        );
    }

    #[test]
    fn difference_eval() {
        let problem = generate_problem();
        let state = generate_state();
        let expression = SetExpression::SetOperation(
            SetOperator::Difference,
            Box::new(SetExpression::SetVariable(0)),
            Box::new(SetExpression::SetVariable(1)),
        );
        let mut set = variable::SetVariable::with_capacity(3);
        set.insert(2);
        assert_eq!(expression.eval(&state, &problem), set);
        let expression = SetExpression::SetOperation(
            SetOperator::Difference,
            Box::new(SetExpression::SetVariable(0)),
            Box::new(SetExpression::SetVariable(0)),
        );
        assert_eq!(
            expression.eval(&state, &problem),
            variable::SetVariable::with_capacity(3)
        );
    }

    #[test]
    fn intersect_eval() {
        let problem = generate_problem();
        let state = generate_state();
        let expression = SetExpression::SetOperation(
            SetOperator::Intersect,
            Box::new(SetExpression::SetVariable(0)),
            Box::new(SetExpression::SetVariable(1)),
        );
        let mut set = variable::SetVariable::with_capacity(3);
        set.insert(0);
        assert_eq!(expression.eval(&state, &problem), set);
        let expression = SetExpression::SetOperation(
            SetOperator::Intersect,
            Box::new(SetExpression::SetVariable(0)),
            Box::new(SetExpression::SetVariable(0)),
        );
        assert_eq!(
            expression.eval(&state, &problem),
            state.signature_variables.set_variables[0]
        );
    }

    #[test]
    fn set_add_eval() {
        let problem = generate_problem();
        let state = generate_state();
        let expression = SetExpression::SetElementOperation(
            SetElementOperator::Add,
            Box::new(SetExpression::SetVariable(0)),
            ElementExpression::Constant(1),
        );
        let mut set = variable::SetVariable::with_capacity(3);
        set.insert(0);
        set.insert(1);
        set.insert(2);
        assert_eq!(expression.eval(&state, &problem), set);
        let expression = SetExpression::SetElementOperation(
            SetElementOperator::Add,
            Box::new(SetExpression::SetVariable(0)),
            ElementExpression::Constant(0),
        );
        assert_eq!(
            expression.eval(&state, &problem),
            state.signature_variables.set_variables[0]
        );
    }

    #[test]
    fn set_remove_eval() {
        let problem = generate_problem();
        let state = generate_state();
        let expression = SetExpression::SetElementOperation(
            SetElementOperator::Remove,
            Box::new(SetExpression::SetVariable(0)),
            ElementExpression::Constant(2),
        );
        let mut set = variable::SetVariable::with_capacity(3);
        set.insert(0);
        assert_eq!(expression.eval(&state, &problem), set);
        let expression = SetExpression::SetElementOperation(
            SetElementOperator::Remove,
            Box::new(SetExpression::SetVariable(0)),
            ElementExpression::Constant(1),
        );
        assert_eq!(
            expression.eval(&state, &problem),
            state.signature_variables.set_variables[0]
        );
    }
}
