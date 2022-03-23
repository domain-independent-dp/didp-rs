use super::element_expression::{ElementExpression, TableExpression};
use crate::state::State;
use crate::table_registry::TableRegistry;
use crate::variable::Vector;

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum VectorExpression {
    Constant(Vector),
    Variable(usize),
    Push(ElementExpression, Box<VectorExpression>),
    Pop(Box<VectorExpression>),
    Table(TableExpression<Vector>),
}

impl VectorExpression {
    pub fn eval(&self, state: &State, registry: &TableRegistry) -> Vector {
        match self {
            Self::Constant(value) => value.clone(),
            Self::Variable(i) => state.signature_variables.vector_variables[*i].clone(),
            Self::Push(element, vector) => {
                let element = element.eval(state, registry);
                let mut vector = vector.eval(state, registry);
                vector.push(element);
                vector
            }
            Self::Pop(vector) => {
                let mut vector = vector.eval(state, registry);
                vector.pop();
                vector
            }
            Self::Table(table) => table.eval(state, registry, &registry.vector_tables),
        }
    }

    pub fn simplify(&self, registry: &TableRegistry) -> VectorExpression {
        match self {
            Self::Push(element, vector) => {
                match (element.simplify(registry), vector.simplify(registry)) {
                    (
                        ElementExpression::Constant(element),
                        VectorExpression::Constant(mut vector),
                    ) => {
                        vector.push(element);
                        Self::Constant(vector)
                    }
                    (element, vector) => Self::Push(element, Box::new(vector)),
                }
            }
            Self::Pop(vector) => match vector.simplify(registry) {
                VectorExpression::Constant(mut vector) => {
                    vector.pop();
                    Self::Constant(vector)
                }
                vector => Self::Pop(Box::new(vector)),
            },
            Self::Table(table) => match table.simplify(registry, &registry.vector_tables) {
                TableExpression::Constant(value) => Self::Constant(value),
                expression => Self::Table(expression),
            },
            _ => self.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::*;
    use crate::table::*;
    use crate::table_data::TableData;
    use std::collections::HashMap;
    use std::rc::Rc;

    fn generate_registry() -> TableRegistry {
        let mut name_to_table_1d = HashMap::new();
        name_to_table_1d.insert(String::from("t1"), 0);
        TableRegistry {
            vector_tables: TableData {
                tables_1d: vec![Table1D::new(vec![vec![0, 1]])],
                name_to_table_1d,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn generate_state() -> State {
        State {
            signature_variables: Rc::new(SignatureVariables {
                vector_variables: vec![vec![0, 2]],
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[test]
    fn constant_eval() {
        let state = generate_state();
        let registry = generate_registry();
        let expression = VectorExpression::Constant(vec![0, 1, 2]);
        assert_eq!(expression.eval(&state, &registry), vec![0, 1, 2]);
    }

    #[test]
    fn variable_eval() {
        let state = generate_state();
        let registry = generate_registry();
        let expression = VectorExpression::Variable(0);
        assert_eq!(expression.eval(&state, &registry), vec![0, 2]);
    }

    #[test]
    fn push_eval() {
        let state = generate_state();
        let registry = generate_registry();
        let expression = VectorExpression::Push(
            ElementExpression::Constant(0),
            Box::new(VectorExpression::Constant(vec![1, 2])),
        );
        assert_eq!(expression.eval(&state, &registry), vec![1, 2, 0]);
    }

    #[test]
    fn pop_eval() {
        let state = generate_state();
        let registry = generate_registry();
        let expression = VectorExpression::Pop(Box::new(VectorExpression::Constant(vec![1, 2])));
        assert_eq!(expression.eval(&state, &registry), vec![1]);
    }

    #[test]
    fn table_eval() {
        let state = generate_state();
        let registry = generate_registry();
        let expression =
            VectorExpression::Table(TableExpression::Table1D(0, ElementExpression::Constant(0)));
        assert_eq!(expression.eval(&state, &registry), vec![0, 1]);
    }

    #[test]
    fn constant_simplify() {
        let registry = generate_registry();
        let expression = VectorExpression::Constant(vec![0, 1, 2]);
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn variable_simplify() {
        let registry = generate_registry();
        let expression = VectorExpression::Variable(0);
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn push_simplify() {
        let registry = generate_registry();
        let expression = VectorExpression::Push(
            ElementExpression::Constant(0),
            Box::new(VectorExpression::Constant(vec![1, 2])),
        );
        assert_eq!(
            expression.simplify(&registry),
            VectorExpression::Constant(vec![1, 2, 0])
        );
        let expression = VectorExpression::Push(
            ElementExpression::Constant(0),
            Box::new(VectorExpression::Variable(0)),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn pop_simplify() {
        let registry = generate_registry();
        let expression = VectorExpression::Pop(Box::new(VectorExpression::Constant(vec![1, 2])));
        assert_eq!(
            expression.simplify(&registry),
            VectorExpression::Constant(vec![1])
        );
        let expression = VectorExpression::Pop(Box::new(VectorExpression::Variable(0)));
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn table_simplify() {
        let registry = generate_registry();
        let expression =
            VectorExpression::Table(TableExpression::Table1D(0, ElementExpression::Constant(0)));
        assert_eq!(
            expression.simplify(&registry),
            VectorExpression::Constant(vec![0, 1])
        );
        let expression =
            VectorExpression::Table(TableExpression::Table1D(0, ElementExpression::Variable(0)));
        assert_eq!(expression.simplify(&registry), expression);
    }
}
