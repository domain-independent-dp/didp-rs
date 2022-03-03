use super::set_expression::ElementExpression;
use super::set_expression::SetExpression;
use crate::problem;
use crate::state;
use crate::variable;

#[derive(Debug)]
pub enum SetCondition {
    Eq(ElementExpression, ElementExpression),
    Ne(ElementExpression, ElementExpression),
    IsIn(ElementExpression, SetExpression),
    IsSubset(SetExpression, SetExpression),
    IsEmpty(SetExpression),
}

impl SetCondition {
    pub fn eval<T: variable::Numeric>(
        &self,
        state: &state::State<T>,
        problem: &problem::Problem<T>,
    ) -> bool {
        match self {
            SetCondition::Eq(x, y) => {
                let x = x.eval(state);
                let y = y.eval(state);
                x == y
            }
            SetCondition::Ne(x, y) => {
                let x = x.eval(state);
                let y = y.eval(state);
                x != y
            }
            SetCondition::IsIn(e, s) => {
                let e = e.eval(state);
                let s = s.eval(state, problem);
                s.contains(e)
            }
            SetCondition::IsSubset(x, y) => {
                let x = x.eval(state, problem);
                let y = y.eval(state, problem);
                x.is_subset(&y)
            }
            SetCondition::IsEmpty(s) => {
                let s = s.eval(state, problem);
                s.count_ones(..) == 0
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::rc::Rc;

    fn generate_problem() -> problem::Problem<variable::IntegerVariable> {
        problem::Problem {
            set_variable_to_max_size: vec![3],
            permutation_variable_to_max_length: vec![3],
            element_to_set: vec![0],
            functions_1d: HashMap::new(),
            functions_2d: HashMap::new(),
            functions_3d: HashMap::new(),
            functions: HashMap::new(),
        }
    }

    fn generate_state() -> state::State<variable::IntegerVariable> {
        let mut set1 = variable::SetVariable::with_capacity(3);
        set1.insert(0);
        set1.insert(2);
        let mut set2 = variable::SetVariable::with_capacity(3);
        set2.insert(0);
        let mut set3 = variable::SetVariable::with_capacity(3);
        set3.insert(0);
        set3.insert(1);
        let set4 = variable::SetVariable::with_capacity(3);
        state::State {
            signature_variables: Rc::new(state::SignatureVariables {
                set_variables: vec![set1, set2, set3, set4],
                permutation_variables: vec![vec![0, 2]],
                element_variables: vec![1],
                numeric_variables: vec![1, 2, 3],
            }),
            resource_variables: state::ResourceVariables {
                numeric_variables: vec![4, 5, 6],
            },
        }
    }

    #[test]
    fn element_eq() {
        let problem = generate_problem();
        let state = generate_state();

        let expression = SetCondition::Eq(
            ElementExpression::Constant(1),
            ElementExpression::Constant(1),
        );
        assert!(expression.eval(&state, &problem));

        let expression = SetCondition::Eq(
            ElementExpression::Constant(0),
            ElementExpression::Constant(1),
        );
        assert!(!expression.eval(&state, &problem));

        let expression = SetCondition::Eq(
            ElementExpression::Constant(1),
            ElementExpression::Variable(0),
        );
        assert!(expression.eval(&state, &problem));

        let expression = SetCondition::Eq(
            ElementExpression::Constant(0),
            ElementExpression::Variable(0),
        );
        assert!(!expression.eval(&state, &problem));
    }

    #[test]
    fn element_ne() {
        let problem = generate_problem();
        let state = generate_state();

        let expression = SetCondition::Ne(
            ElementExpression::Constant(1),
            ElementExpression::Constant(1),
        );
        assert!(!expression.eval(&state, &problem));

        let expression = SetCondition::Ne(
            ElementExpression::Constant(0),
            ElementExpression::Constant(1),
        );
        assert!(expression.eval(&state, &problem));

        let expression = SetCondition::Ne(
            ElementExpression::Constant(1),
            ElementExpression::Variable(0),
        );
        assert!(!expression.eval(&state, &problem));

        let expression = SetCondition::Ne(
            ElementExpression::Constant(0),
            ElementExpression::Variable(0),
        );
        assert!(expression.eval(&state, &problem));
    }

    #[test]
    fn element_in_set() {
        let problem = generate_problem();
        let state = generate_state();

        let expression = SetCondition::IsIn(
            ElementExpression::Constant(0),
            SetExpression::SetVariable(0),
        );
        assert!(expression.eval(&state, &problem));

        let expression = SetCondition::IsIn(
            ElementExpression::Constant(1),
            SetExpression::SetVariable(0),
        );
        assert!(!expression.eval(&state, &problem));

        let expression = SetCondition::IsIn(
            ElementExpression::Constant(2),
            SetExpression::SetVariable(0),
        );
        assert!(expression.eval(&state, &problem));
    }

    #[test]
    fn subset_of() {
        let problem = generate_problem();
        let state = generate_state();

        let expression =
            SetCondition::IsSubset(SetExpression::SetVariable(0), SetExpression::SetVariable(0));
        assert!(expression.eval(&state, &problem));

        let expression =
            SetCondition::IsSubset(SetExpression::SetVariable(0), SetExpression::SetVariable(1));
        assert!(!expression.eval(&state, &problem));

        let expression =
            SetCondition::IsSubset(SetExpression::SetVariable(0), SetExpression::SetVariable(2));
        assert!(!expression.eval(&state, &problem));

        let expression =
            SetCondition::IsSubset(SetExpression::SetVariable(0), SetExpression::SetVariable(3));
        assert!(!expression.eval(&state, &problem));

        let expression =
            SetCondition::IsSubset(SetExpression::SetVariable(1), SetExpression::SetVariable(0));
        assert!(expression.eval(&state, &problem));

        let expression =
            SetCondition::IsSubset(SetExpression::SetVariable(2), SetExpression::SetVariable(0));
        assert!(!expression.eval(&state, &problem));

        let expression =
            SetCondition::IsSubset(SetExpression::SetVariable(3), SetExpression::SetVariable(1));
        assert!(expression.eval(&state, &problem));
    }

    #[test]
    fn is_empty() {
        let problem = generate_problem();
        let state = generate_state();

        let expression = SetCondition::IsEmpty(SetExpression::SetVariable(0));
        assert!(!expression.eval(&state, &problem));

        let expression = SetCondition::IsEmpty(SetExpression::SetVariable(3));
        assert!(expression.eval(&state, &problem));
    }
}
