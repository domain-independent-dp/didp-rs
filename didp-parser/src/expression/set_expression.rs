use super::element_expression::{ElementExpression, TableExpression};
use crate::state;
use crate::table_registry;
use crate::variable;

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum SetExpression {
    Constant(variable::Set),
    SetVariable(usize),
    VectorVariable(usize),
    Complement(Box<SetExpression>),
    SetOperation(SetOperator, Box<SetExpression>, Box<SetExpression>),
    SetElementOperation(SetElementOperator, Box<SetExpression>, ElementExpression),
    Table(TableExpression<variable::Set>),
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
    pub fn eval(
        &self,
        state: &state::State,
        metadata: &state::StateMetadata,
        registry: &table_registry::TableRegistry,
    ) -> variable::Set {
        match self {
            SetExpression::Constant(value) => value.clone(),
            SetExpression::SetVariable(i) => state.signature_variables.set_variables[*i].clone(),
            SetExpression::VectorVariable(i) => {
                let mut set = variable::Set::with_capacity(metadata.vector_variable_capacity(*i));
                for v in &state.signature_variables.vector_variables[*i] {
                    set.insert(*v);
                }
                set
            }
            SetExpression::Complement(s) => {
                let mut s = s.eval(&state, metadata, registry);
                s.toggle_range(..);
                s
            }
            SetExpression::SetOperation(op, a, b) => {
                let mut a = a.eval(&state, metadata, registry);
                let b = b.eval(&state, metadata, registry);
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
                let mut s = s.eval(&state, metadata, registry);
                let e = e.eval(&state, &registry.element_tables);
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
            SetExpression::Table(table) => {
                table.eval(&state, &registry.element_tables, &registry.set_tables)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::table;
    use crate::table_data;
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

        let vector_variable_names = vec![
            "p0".to_string(),
            "p1".to_string(),
            "p2".to_string(),
            "p3".to_string(),
        ];
        let mut name_to_vector_variable = HashMap::new();
        name_to_vector_variable.insert("p0".to_string(), 0);
        name_to_vector_variable.insert("p1".to_string(), 1);
        name_to_vector_variable.insert("p2".to_string(), 2);
        name_to_vector_variable.insert("p3".to_string(), 3);
        let vector_variable_to_object = vec![0, 0, 0, 0];

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
            vector_variable_names,
            name_to_vector_variable,
            vector_variable_to_object,
            element_variable_names,
            name_to_element_variable,
            element_variable_to_object,
            ..Default::default()
        }
    }

    fn generate_registry() -> table_registry::TableRegistry {
        let mut set = variable::Set::with_capacity(3);
        set.insert(0);
        set.insert(2);
        let default = variable::Set::with_capacity(3);
        let tables_1d = vec![table::Table1D::new(vec![
            set.clone(),
            default.clone(),
            default.clone(),
        ])];
        let mut name_to_table_1d = HashMap::new();
        name_to_table_1d.insert(String::from("s1"), 0);
        table_registry::TableRegistry {
            set_tables: table_data::TableData {
                tables_1d,
                name_to_table_1d,
                ..Default::default()
            },
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
                vector_variables: vec![vec![0, 2]],
                element_variables: vec![1],
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[test]
    fn set_variable_eval() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let state = generate_state();
        let expression = SetExpression::SetVariable(0);
        assert_eq!(
            expression.eval(&state, &metadata, &registry),
            state.signature_variables.set_variables[0]
        );
        let expression = SetExpression::SetVariable(1);
        assert_eq!(
            expression.eval(&state, &metadata, &registry),
            state.signature_variables.set_variables[1]
        );
    }

    #[test]
    fn vector_variable_eval() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let state = generate_state();
        let expression = SetExpression::VectorVariable(0);
        let mut set = variable::Set::with_capacity(10);
        set.insert(0);
        set.insert(2);
        assert_eq!(expression.eval(&state, &metadata, &registry), set);
    }

    #[test]
    fn complement_eval() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let state = generate_state();
        let expression = SetExpression::Complement(Box::new(SetExpression::SetVariable(0)));
        let mut set = variable::Set::with_capacity(3);
        set.insert(1);
        assert_eq!(expression.eval(&state, &metadata, &registry), set);
    }

    #[test]
    fn union_eval() {
        let metadata = generate_metadata();
        let registry = generate_registry();
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
        assert_eq!(expression.eval(&state, &metadata, &registry), set);
        let expression = SetExpression::SetOperation(
            SetOperator::Union,
            Box::new(SetExpression::SetVariable(0)),
            Box::new(SetExpression::SetVariable(0)),
        );
        assert_eq!(
            expression.eval(&state, &metadata, &registry),
            state.signature_variables.set_variables[0]
        );
    }

    #[test]
    fn difference_eval() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let state = generate_state();
        let expression = SetExpression::SetOperation(
            SetOperator::Difference,
            Box::new(SetExpression::SetVariable(0)),
            Box::new(SetExpression::SetVariable(1)),
        );
        let mut set = variable::Set::with_capacity(3);
        set.insert(2);
        assert_eq!(expression.eval(&state, &metadata, &registry), set);
        let expression = SetExpression::SetOperation(
            SetOperator::Difference,
            Box::new(SetExpression::SetVariable(0)),
            Box::new(SetExpression::SetVariable(0)),
        );
        assert_eq!(
            expression.eval(&state, &metadata, &registry),
            variable::Set::with_capacity(3)
        );
    }

    #[test]
    fn intersect_eval() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let state = generate_state();
        let expression = SetExpression::SetOperation(
            SetOperator::Intersect,
            Box::new(SetExpression::SetVariable(0)),
            Box::new(SetExpression::SetVariable(1)),
        );
        let mut set = variable::Set::with_capacity(3);
        set.insert(0);
        assert_eq!(expression.eval(&state, &metadata, &registry), set);
        let expression = SetExpression::SetOperation(
            SetOperator::Intersect,
            Box::new(SetExpression::SetVariable(0)),
            Box::new(SetExpression::SetVariable(0)),
        );
        assert_eq!(
            expression.eval(&state, &metadata, &registry),
            state.signature_variables.set_variables[0]
        );
    }

    #[test]
    fn set_add_eval() {
        let metadata = generate_metadata();
        let registry = generate_registry();
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
        assert_eq!(expression.eval(&state, &metadata, &registry), set);
        let expression = SetExpression::SetElementOperation(
            SetElementOperator::Add,
            Box::new(SetExpression::SetVariable(0)),
            ElementExpression::Constant(0),
        );
        assert_eq!(
            expression.eval(&state, &metadata, &registry),
            state.signature_variables.set_variables[0]
        );
    }

    #[test]
    fn set_remove_eval() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let state = generate_state();
        let expression = SetExpression::SetElementOperation(
            SetElementOperator::Remove,
            Box::new(SetExpression::SetVariable(0)),
            ElementExpression::Constant(2),
        );
        let mut set = variable::Set::with_capacity(3);
        set.insert(0);
        assert_eq!(expression.eval(&state, &metadata, &registry), set);
        let expression = SetExpression::SetElementOperation(
            SetElementOperator::Remove,
            Box::new(SetExpression::SetVariable(0)),
            ElementExpression::Constant(1),
        );
        assert_eq!(
            expression.eval(&state, &metadata, &registry),
            state.signature_variables.set_variables[0]
        );
    }
}
