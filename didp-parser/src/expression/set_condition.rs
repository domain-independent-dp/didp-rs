use super::set_expression::ElementExpression;
use super::set_expression::SetExpression;
use crate::state;
use crate::variable;

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum SetCondition {
    Constant(bool),
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
        metadata: &state::StateMetadata,
    ) -> bool {
        match self {
            Self::Constant(value) => *value,
            Self::Eq(x, y) => x.eval(state) == y.eval(state),
            Self::Ne(x, y) => x.eval(state) != y.eval(state),
            Self::IsIn(e, SetExpression::SetVariable(i)) => {
                state.signature_variables.set_variables[*i].contains(e.eval(state))
            }
            Self::IsIn(e, SetExpression::PermutationVariable(i)) => {
                state.signature_variables.permutation_variables[*i].contains(&e.eval(state))
            }
            Self::IsIn(e, s) => s.eval(state, metadata).contains(e.eval(state)),
            Self::IsSubset(SetExpression::SetVariable(i), SetExpression::SetVariable(j)) => {
                let x = &state.signature_variables.set_variables[*i];
                let y = &state.signature_variables.set_variables[*j];
                x.is_subset(y)
            }
            Self::IsSubset(x, SetExpression::SetVariable(j)) => {
                let y = &state.signature_variables.set_variables[*j];
                x.eval(state, metadata).is_subset(y)
            }
            Self::IsSubset(SetExpression::SetVariable(i), y) => {
                let x = &state.signature_variables.set_variables[*i];
                x.is_subset(&y.eval(state, metadata))
            }
            Self::IsSubset(x, y) => x.eval(state, metadata).is_subset(&y.eval(state, metadata)),
            Self::IsEmpty(SetExpression::SetVariable(i)) => {
                state.signature_variables.set_variables[*i].count_ones(..) == 0
            }
            Self::IsEmpty(SetExpression::PermutationVariable(i)) => {
                state.signature_variables.permutation_variables[*i].is_empty()
            }
            Self::IsEmpty(s) => s.eval(state, metadata).count_ones(..) == 0,
        }
    }

    pub fn simplify(&self) -> SetCondition {
        match self {
            Self::Eq(ElementExpression::Constant(x), ElementExpression::Constant(y)) => {
                Self::Constant(x == y)
            }
            Self::Eq(ElementExpression::Variable(x), ElementExpression::Variable(y)) if x == y => {
                Self::Constant(true)
            }
            Self::Ne(ElementExpression::Constant(x), ElementExpression::Constant(y)) => {
                Self::Constant(x != y)
            }
            Self::Ne(ElementExpression::Variable(x), ElementExpression::Variable(y)) if x == y => {
                Self::Constant(false)
            }
            Self::IsSubset(SetExpression::SetVariable(x), SetExpression::SetVariable(y))
            | Self::IsSubset(
                SetExpression::PermutationVariable(x),
                SetExpression::PermutationVariable(y),
            ) if x == y => Self::Constant(true),
            _ => self.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::set_expression::SetOperator;
    use super::*;
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
        let mut set3 = variable::SetVariable::with_capacity(3);
        set3.insert(0);
        set3.insert(1);
        let set4 = variable::SetVariable::with_capacity(3);
        state::State {
            signature_variables: Rc::new(state::SignatureVariables {
                set_variables: vec![set1, set2, set3, set4],
                permutation_variables: vec![vec![0, 2], vec![], vec![], vec![]],
                element_variables: vec![1, 2, 3, 4],
                numeric_variables: vec![1, 2, 3, 4],
            }),
            resource_variables: vec![4, 5, 6, 7],
            stage: 0,
            cost: 0,
        }
    }

    #[test]
    fn constant_eval() {
        let metadata = generate_metadata();
        let state = generate_state();

        let expression = SetCondition::Constant(true);
        assert!(expression.eval(&state, &metadata));

        let expression = SetCondition::Constant(false);
        assert!(!expression.eval(&state, &metadata));
    }

    #[test]
    fn eq_eval() {
        let metadata = generate_metadata();
        let state = generate_state();

        let expression = SetCondition::Eq(
            ElementExpression::Constant(1),
            ElementExpression::Constant(1),
        );
        assert!(expression.eval(&state, &metadata));

        let expression = SetCondition::Eq(
            ElementExpression::Constant(0),
            ElementExpression::Constant(1),
        );
        assert!(!expression.eval(&state, &metadata));

        let expression = SetCondition::Eq(
            ElementExpression::Constant(1),
            ElementExpression::Variable(0),
        );
        assert!(expression.eval(&state, &metadata));

        let expression = SetCondition::Eq(
            ElementExpression::Constant(0),
            ElementExpression::Variable(0),
        );
        assert!(!expression.eval(&state, &metadata));
    }

    #[test]
    fn ne_eval() {
        let metadata = generate_metadata();
        let state = generate_state();

        let expression = SetCondition::Ne(
            ElementExpression::Constant(1),
            ElementExpression::Constant(1),
        );
        assert!(!expression.eval(&state, &metadata));

        let expression = SetCondition::Ne(
            ElementExpression::Constant(0),
            ElementExpression::Constant(1),
        );
        assert!(expression.eval(&state, &metadata));

        let expression = SetCondition::Ne(
            ElementExpression::Constant(1),
            ElementExpression::Variable(0),
        );
        assert!(!expression.eval(&state, &metadata));

        let expression = SetCondition::Ne(
            ElementExpression::Constant(0),
            ElementExpression::Variable(0),
        );
        assert!(expression.eval(&state, &metadata));
    }

    #[test]
    fn is_in_eval() {
        let metadata = generate_metadata();
        let state = generate_state();

        let expression = SetCondition::IsIn(
            ElementExpression::Constant(0),
            SetExpression::SetVariable(0),
        );
        assert!(expression.eval(&state, &metadata));

        let expression = SetCondition::IsIn(
            ElementExpression::Constant(1),
            SetExpression::SetVariable(0),
        );
        assert!(!expression.eval(&state, &metadata));

        let expression = SetCondition::IsIn(
            ElementExpression::Constant(2),
            SetExpression::SetVariable(0),
        );
        assert!(expression.eval(&state, &metadata));

        let expression = SetCondition::IsIn(
            ElementExpression::Constant(0),
            SetExpression::PermutationVariable(0),
        );
        assert!(expression.eval(&state, &metadata));

        let expression = SetCondition::IsIn(
            ElementExpression::Constant(1),
            SetExpression::PermutationVariable(0),
        );
        assert!(!expression.eval(&state, &metadata));

        let expression = SetCondition::IsIn(
            ElementExpression::Constant(2),
            SetExpression::PermutationVariable(0),
        );
        assert!(expression.eval(&state, &metadata));

        let expression = SetCondition::IsIn(
            ElementExpression::Constant(0),
            SetExpression::Complement(Box::new(SetExpression::SetVariable(0))),
        );
        assert!(!expression.eval(&state, &metadata));

        let expression = SetCondition::IsIn(
            ElementExpression::Constant(1),
            SetExpression::Complement(Box::new(SetExpression::SetVariable(0))),
        );
        assert!(expression.eval(&state, &metadata));

        let expression = SetCondition::IsIn(
            ElementExpression::Constant(2),
            SetExpression::Complement(Box::new(SetExpression::SetVariable(0))),
        );
        assert!(!expression.eval(&state, &metadata));
    }

    #[test]
    fn is_subset_eval() {
        let metadata = generate_metadata();
        let state = generate_state();

        let expression =
            SetCondition::IsSubset(SetExpression::SetVariable(0), SetExpression::SetVariable(0));
        assert!(expression.eval(&state, &metadata));

        let expression =
            SetCondition::IsSubset(SetExpression::SetVariable(0), SetExpression::SetVariable(1));
        assert!(!expression.eval(&state, &metadata));

        let expression =
            SetCondition::IsSubset(SetExpression::SetVariable(0), SetExpression::SetVariable(2));
        assert!(!expression.eval(&state, &metadata));

        let expression =
            SetCondition::IsSubset(SetExpression::SetVariable(0), SetExpression::SetVariable(3));
        assert!(!expression.eval(&state, &metadata));

        let expression =
            SetCondition::IsSubset(SetExpression::SetVariable(1), SetExpression::SetVariable(0));
        assert!(expression.eval(&state, &metadata));

        let expression =
            SetCondition::IsSubset(SetExpression::SetVariable(2), SetExpression::SetVariable(0));
        assert!(!expression.eval(&state, &metadata));

        let expression =
            SetCondition::IsSubset(SetExpression::SetVariable(3), SetExpression::SetVariable(1));
        assert!(expression.eval(&state, &metadata));

        let expression = SetCondition::IsSubset(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::SetVariable(0),
            )))),
            SetExpression::SetVariable(0),
        );
        assert!(expression.eval(&state, &metadata));

        let expression = SetCondition::IsSubset(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::SetVariable(0),
            )))),
            SetExpression::SetVariable(1),
        );
        assert!(!expression.eval(&state, &metadata));

        let expression = SetCondition::IsSubset(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::SetVariable(0),
            )))),
            SetExpression::SetVariable(2),
        );
        assert!(!expression.eval(&state, &metadata));

        let expression = SetCondition::IsSubset(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::SetVariable(0),
            )))),
            SetExpression::SetVariable(3),
        );
        assert!(!expression.eval(&state, &metadata));

        let expression = SetCondition::IsSubset(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::SetVariable(1),
            )))),
            SetExpression::SetVariable(0),
        );
        assert!(expression.eval(&state, &metadata));

        let expression = SetCondition::IsSubset(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::SetVariable(2),
            )))),
            SetExpression::SetVariable(0),
        );
        assert!(!expression.eval(&state, &metadata));

        let expression = SetCondition::IsSubset(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::SetVariable(3),
            )))),
            SetExpression::SetVariable(1),
        );
        assert!(expression.eval(&state, &metadata));

        let expression = SetCondition::IsSubset(
            SetExpression::SetVariable(0),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::SetVariable(0),
            )))),
        );
        assert!(expression.eval(&state, &metadata));

        let expression = SetCondition::IsSubset(
            SetExpression::SetVariable(0),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::SetVariable(1),
            )))),
        );
        assert!(!expression.eval(&state, &metadata));

        let expression = SetCondition::IsSubset(
            SetExpression::SetVariable(0),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::SetVariable(2),
            )))),
        );
        assert!(!expression.eval(&state, &metadata));

        let expression = SetCondition::IsSubset(
            SetExpression::SetVariable(0),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::SetVariable(3),
            )))),
        );
        assert!(!expression.eval(&state, &metadata));

        let expression = SetCondition::IsSubset(
            SetExpression::SetVariable(1),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::SetVariable(0),
            )))),
        );
        assert!(expression.eval(&state, &metadata));

        let expression = SetCondition::IsSubset(
            SetExpression::SetVariable(2),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::SetVariable(0),
            )))),
        );
        assert!(!expression.eval(&state, &metadata));

        let expression = SetCondition::IsSubset(
            SetExpression::SetVariable(3),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::SetVariable(1),
            )))),
        );
        assert!(expression.eval(&state, &metadata));

        let expression = SetCondition::IsSubset(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::SetVariable(0),
            )))),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::SetVariable(0),
            )))),
        );
        assert!(expression.eval(&state, &metadata));

        let expression = SetCondition::IsSubset(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::SetVariable(0),
            )))),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::SetVariable(1),
            )))),
        );
        assert!(!expression.eval(&state, &metadata));

        let expression = SetCondition::IsSubset(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::SetVariable(0),
            )))),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::SetVariable(2),
            )))),
        );
        assert!(!expression.eval(&state, &metadata));

        let expression = SetCondition::IsSubset(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::SetVariable(0),
            )))),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::SetVariable(3),
            )))),
        );
        assert!(!expression.eval(&state, &metadata));

        let expression = SetCondition::IsSubset(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::SetVariable(1),
            )))),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::SetVariable(0),
            )))),
        );
        assert!(expression.eval(&state, &metadata));

        let expression = SetCondition::IsSubset(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::SetVariable(2),
            )))),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::SetVariable(0),
            )))),
        );
        assert!(!expression.eval(&state, &metadata));

        let expression = SetCondition::IsSubset(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::SetVariable(3),
            )))),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::SetVariable(1),
            )))),
        );
        assert!(expression.eval(&state, &metadata));
    }

    #[test]
    fn is_empty_eval() {
        let metadata = generate_metadata();
        let state = generate_state();

        let expression = SetCondition::IsEmpty(SetExpression::SetVariable(0));
        assert!(!expression.eval(&state, &metadata));

        let expression = SetCondition::IsEmpty(SetExpression::SetVariable(3));
        assert!(expression.eval(&state, &metadata));

        let expression = SetCondition::IsEmpty(SetExpression::PermutationVariable(0));
        assert!(!expression.eval(&state, &metadata));

        let expression = SetCondition::IsEmpty(SetExpression::PermutationVariable(1));
        assert!(expression.eval(&state, &metadata));

        let expression = SetCondition::IsEmpty(SetExpression::Complement(Box::new(
            SetExpression::SetVariable(0),
        )));
        assert!(!expression.eval(&state, &metadata));

        let expression = SetCondition::IsEmpty(SetExpression::Complement(Box::new(
            SetExpression::SetOperation(
                SetOperator::Union,
                Box::new(SetExpression::SetVariable(0)),
                Box::new(SetExpression::SetVariable(2)),
            ),
        )));
        assert!(expression.eval(&state, &metadata));
    }

    #[test]
    fn constant_simplify() {
        let expression = SetCondition::Constant(true);
        assert!(matches!(
            expression.simplify(),
            SetCondition::Constant(true)
        ));

        let expression = SetCondition::Constant(false);
        assert!(matches!(
            expression.simplify(),
            SetCondition::Constant(false)
        ));
    }

    #[test]
    fn eq_simplify() {
        let expression = SetCondition::Eq(
            ElementExpression::Constant(1),
            ElementExpression::Constant(1),
        );
        assert!(matches!(
            expression.simplify(),
            SetCondition::Constant(true)
        ));

        let expression = SetCondition::Eq(
            ElementExpression::Constant(0),
            ElementExpression::Constant(1),
        );
        assert!(matches!(
            expression.simplify(),
            SetCondition::Constant(false)
        ));

        let expression = SetCondition::Eq(
            ElementExpression::Variable(1),
            ElementExpression::Variable(1),
        );
        assert!(matches!(
            expression.simplify(),
            SetCondition::Constant(true)
        ));

        let expression = SetCondition::Eq(
            ElementExpression::Variable(0),
            ElementExpression::Variable(1),
        );
        assert!(matches!(
            expression.simplify(),
            SetCondition::Eq(
                ElementExpression::Variable(0),
                ElementExpression::Variable(1),
            )
        ));

        let expression = SetCondition::Eq(
            ElementExpression::Constant(1),
            ElementExpression::Variable(0),
        );
        assert!(matches!(
            expression.simplify(),
            SetCondition::Eq(
                ElementExpression::Constant(1),
                ElementExpression::Variable(0),
            )
        ));
    }

    #[test]
    fn ne_simplify() {
        let expression = SetCondition::Ne(
            ElementExpression::Constant(1),
            ElementExpression::Constant(1),
        );
        assert!(matches!(
            expression.simplify(),
            SetCondition::Constant(false)
        ));

        let expression = SetCondition::Ne(
            ElementExpression::Constant(0),
            ElementExpression::Constant(1),
        );
        assert!(matches!(
            expression.simplify(),
            SetCondition::Constant(true)
        ));

        let expression = SetCondition::Eq(
            ElementExpression::Variable(1),
            ElementExpression::Variable(1),
        );
        assert!(matches!(
            expression.simplify(),
            SetCondition::Constant(true)
        ));

        let expression = SetCondition::Eq(
            ElementExpression::Variable(0),
            ElementExpression::Variable(1),
        );
        assert!(matches!(
            expression.simplify(),
            SetCondition::Eq(
                ElementExpression::Variable(0),
                ElementExpression::Variable(1),
            )
        ));

        let expression = SetCondition::Eq(
            ElementExpression::Constant(1),
            ElementExpression::Variable(0),
        );
        assert!(matches!(
            expression.simplify(),
            SetCondition::Eq(
                ElementExpression::Constant(1),
                ElementExpression::Variable(0),
            )
        ));
    }

    #[test]
    fn is_in_simplify() {
        let expression = SetCondition::IsIn(
            ElementExpression::Constant(0),
            SetExpression::SetVariable(0),
        );
        assert!(matches!(
            expression.simplify(),
            SetCondition::IsIn(
                ElementExpression::Constant(0),
                SetExpression::SetVariable(0),
            )
        ));
    }

    #[test]
    fn is_subset_simplify() {
        let expression =
            SetCondition::IsSubset(SetExpression::SetVariable(0), SetExpression::SetVariable(0));
        assert!(matches!(
            expression.simplify(),
            SetCondition::Constant(true)
        ));

        let expression = SetCondition::IsSubset(
            SetExpression::PermutationVariable(0),
            SetExpression::PermutationVariable(0),
        );
        assert!(matches!(
            expression.simplify(),
            SetCondition::Constant(true)
        ));

        let expression =
            SetCondition::IsSubset(SetExpression::SetVariable(0), SetExpression::SetVariable(1));
        assert!(matches!(
            expression.simplify(),
            SetCondition::IsSubset(SetExpression::SetVariable(0), SetExpression::SetVariable(1))
        ));
    }

    #[test]
    fn is_empty_simplify() {
        let expression = SetCondition::IsEmpty(SetExpression::SetVariable(0));
        assert!(matches!(
            expression.simplify(),
            SetCondition::IsEmpty(SetExpression::SetVariable(0))
        ));
    }
}
