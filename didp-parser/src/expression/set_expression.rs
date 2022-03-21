use crate::state;
use crate::variable;

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum SetExpression {
    SetVariable(usize),
    PermutationVariable(usize),
    Complement(Box<SetExpression>),
    SetOperation(SetOperator, Box<SetExpression>, Box<SetExpression>),
    SetElementOperation(SetElementOperator, Box<SetExpression>, ElementExpression),
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum SetOperator {
    Union,
    Difference,
    Intersect,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum SetElementOperator {
    Add,
    Remove,
}

impl SetExpression {
    pub fn eval(&self, state: &state::State, metadata: &state::StateMetadata) -> variable::Set {
        match self {
            SetExpression::SetVariable(i) => state.signature_variables.set_variables[*i].clone(),
            SetExpression::PermutationVariable(i) => {
                let mut set =
                    variable::Set::with_capacity(metadata.permutation_variable_capacity(*i));
                for v in &state.signature_variables.permutation_variables[*i] {
                    set.insert(*v);
                }
                set
            }
            SetExpression::Complement(s) => {
                let mut s = s.eval(&state, metadata);
                s.toggle_range(..);
                s
            }
            SetExpression::SetOperation(op, a, b) => {
                let mut a = a.eval(&state, metadata);
                let b = b.eval(&state, metadata);
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
                let mut s = s.eval(&state, metadata);
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

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum ElementExpression {
    Stage,
    Constant(variable::Element),
    Variable(usize),
}

impl ElementExpression {
    pub fn eval(&self, state: &state::State) -> variable::Element {
        match self {
            ElementExpression::Stage => state.stage,
            ElementExpression::Constant(x) => *x,
            ElementExpression::Variable(i) => state.signature_variables.element_variables[*i],
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum ArgumentExpression {
    Set(SetExpression),
    Element(ElementExpression),
}

#[cfg(test)]
mod tests {
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
            ..Default::default()
        }
    }

    fn generate_state() -> state::State {
        let mut set1 = variable::Set::with_capacity(3);
        set1.insert(0);
        set1.insert(2);
        let mut set2 = variable::Set::with_capacity(3);
        set2.insert(0);
        set2.insert(1);
        state::State {
            signature_variables: Rc::new(state::SignatureVariables {
                set_variables: vec![set1, set2],
                permutation_variables: vec![vec![0, 2]],
                element_variables: vec![1],
                ..Default::default()
            }),
            ..Default::default()
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
        let metadata = generate_metadata();
        let state = generate_state();
        let expression = SetExpression::SetVariable(0);
        assert_eq!(
            expression.eval(&state, &metadata),
            state.signature_variables.set_variables[0]
        );
        let expression = SetExpression::SetVariable(1);
        assert_eq!(
            expression.eval(&state, &metadata),
            state.signature_variables.set_variables[1]
        );
    }

    #[test]
    fn permutation_variable_eval() {
        let metadata = generate_metadata();
        let state = generate_state();
        let expression = SetExpression::PermutationVariable(0);
        let mut set = variable::Set::with_capacity(10);
        set.insert(0);
        set.insert(2);
        assert_eq!(expression.eval(&state, &metadata), set);
    }

    #[test]
    fn complement_eval() {
        let metadata = generate_metadata();
        let state = generate_state();
        let expression = SetExpression::Complement(Box::new(SetExpression::SetVariable(0)));
        let mut set = variable::Set::with_capacity(3);
        set.insert(1);
        assert_eq!(expression.eval(&state, &metadata), set);
    }

    #[test]
    fn union_eval() {
        let metadata = generate_metadata();
        let state = generate_state();
        let expression = SetExpression::SetOperation(
            SetOperator::Union,
            Box::new(SetExpression::SetVariable(0)),
            Box::new(SetExpression::SetVariable(1)),
        );
        let mut set = variable::Set::with_capacity(3);
        set.insert(0);
        set.insert(1);
        set.insert(2);
        assert_eq!(expression.eval(&state, &metadata), set);
        let expression = SetExpression::SetOperation(
            SetOperator::Union,
            Box::new(SetExpression::SetVariable(0)),
            Box::new(SetExpression::SetVariable(0)),
        );
        assert_eq!(
            expression.eval(&state, &metadata),
            state.signature_variables.set_variables[0]
        );
    }

    #[test]
    fn difference_eval() {
        let metadata = generate_metadata();
        let state = generate_state();
        let expression = SetExpression::SetOperation(
            SetOperator::Difference,
            Box::new(SetExpression::SetVariable(0)),
            Box::new(SetExpression::SetVariable(1)),
        );
        let mut set = variable::Set::with_capacity(3);
        set.insert(2);
        assert_eq!(expression.eval(&state, &metadata), set);
        let expression = SetExpression::SetOperation(
            SetOperator::Difference,
            Box::new(SetExpression::SetVariable(0)),
            Box::new(SetExpression::SetVariable(0)),
        );
        assert_eq!(
            expression.eval(&state, &metadata),
            variable::Set::with_capacity(3)
        );
    }

    #[test]
    fn intersect_eval() {
        let metadata = generate_metadata();
        let state = generate_state();
        let expression = SetExpression::SetOperation(
            SetOperator::Intersect,
            Box::new(SetExpression::SetVariable(0)),
            Box::new(SetExpression::SetVariable(1)),
        );
        let mut set = variable::Set::with_capacity(3);
        set.insert(0);
        assert_eq!(expression.eval(&state, &metadata), set);
        let expression = SetExpression::SetOperation(
            SetOperator::Intersect,
            Box::new(SetExpression::SetVariable(0)),
            Box::new(SetExpression::SetVariable(0)),
        );
        assert_eq!(
            expression.eval(&state, &metadata),
            state.signature_variables.set_variables[0]
        );
    }

    #[test]
    fn set_add_eval() {
        let metadata = generate_metadata();
        let state = generate_state();
        let expression = SetExpression::SetElementOperation(
            SetElementOperator::Add,
            Box::new(SetExpression::SetVariable(0)),
            ElementExpression::Constant(1),
        );
        let mut set = variable::Set::with_capacity(3);
        set.insert(0);
        set.insert(1);
        set.insert(2);
        assert_eq!(expression.eval(&state, &metadata), set);
        let expression = SetExpression::SetElementOperation(
            SetElementOperator::Add,
            Box::new(SetExpression::SetVariable(0)),
            ElementExpression::Constant(0),
        );
        assert_eq!(
            expression.eval(&state, &metadata),
            state.signature_variables.set_variables[0]
        );
    }

    #[test]
    fn set_remove_eval() {
        let metadata = generate_metadata();
        let state = generate_state();
        let expression = SetExpression::SetElementOperation(
            SetElementOperator::Remove,
            Box::new(SetExpression::SetVariable(0)),
            ElementExpression::Constant(2),
        );
        let mut set = variable::Set::with_capacity(3);
        set.insert(0);
        assert_eq!(expression.eval(&state, &metadata), set);
        let expression = SetExpression::SetElementOperation(
            SetElementOperator::Remove,
            Box::new(SetExpression::SetVariable(0)),
            ElementExpression::Constant(1),
        );
        assert_eq!(
            expression.eval(&state, &metadata),
            state.signature_variables.set_variables[0]
        );
    }
}
