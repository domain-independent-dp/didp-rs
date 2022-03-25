use super::element_expression::ElementExpression;
use super::reference_expression::ReferenceExpression;
use crate::state::State;
use crate::table_registry::TableRegistry;
use crate::variable::Vector;

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum VectorExpression {
    Indices(Box<VectorExpression>),
    Reference(ReferenceExpression<Vector>),
    Push(ElementExpression, Box<VectorExpression>),
    Pop(Box<VectorExpression>),
}

impl VectorExpression {
    pub fn eval(&self, state: &State, registry: &TableRegistry) -> Vector {
        match self {
            Self::Indices(expression) => {
                let mut vector = expression.eval(state, registry);
                for i in 0..vector.len() {
                    vector[i] = i;
                }
                vector
            }
            Self::Reference(expression) => expression
                .eval(
                    state,
                    registry,
                    &state.signature_variables.vector_variables,
                    &registry.vector_tables,
                )
                .clone(),
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
        }
    }

    pub fn simplify(&self, registry: &TableRegistry) -> VectorExpression {
        match self {
            Self::Indices(expression) => match expression.simplify(registry) {
                VectorExpression::Reference(ReferenceExpression::Constant(mut vector)) => {
                    for i in 0..vector.len() {
                        vector[i] = i;
                    }
                    Self::Reference(ReferenceExpression::Constant(vector))
                }
                expression => Self::Indices(Box::new(expression)),
            },
            Self::Reference(expression) => {
                Self::Reference(expression.simplify(registry, &registry.vector_tables))
            }
            Self::Push(element, vector) => {
                match (element.simplify(registry), vector.simplify(registry)) {
                    (
                        ElementExpression::Constant(element),
                        VectorExpression::Reference(ReferenceExpression::Constant(mut vector)),
                    ) => {
                        vector.push(element);
                        Self::Reference(ReferenceExpression::Constant(vector))
                    }
                    (element, vector) => Self::Push(element, Box::new(vector)),
                }
            }
            Self::Pop(vector) => match vector.simplify(registry) {
                VectorExpression::Reference(ReferenceExpression::Constant(mut vector)) => {
                    vector.pop();
                    Self::Reference(ReferenceExpression::Constant(vector))
                }
                vector => Self::Pop(Box::new(vector)),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::element_expression::*;
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
    fn indices_eval() {
        let state = generate_state();
        let registry = generate_registry();
        let expression = VectorExpression::Indices(Box::new(VectorExpression::Reference(
            ReferenceExpression::Constant(vec![1, 2]),
        )));
        assert_eq!(expression.eval(&state, &registry), vec![0, 1]);
    }

    #[test]
    fn reference_eval() {
        let state = generate_state();
        let registry = generate_registry();
        let expression = VectorExpression::Reference(ReferenceExpression::Constant(vec![1, 2]));
        assert_eq!(expression.eval(&state, &registry), vec![1, 2]);
    }

    #[test]
    fn push_eval() {
        let state = generate_state();
        let registry = generate_registry();
        let expression = VectorExpression::Push(
            ElementExpression::Constant(0),
            Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![1, 2],
            ))),
        );
        assert_eq!(expression.eval(&state, &registry), vec![1, 2, 0]);
    }

    #[test]
    fn pop_eval() {
        let state = generate_state();
        let registry = generate_registry();
        let expression = VectorExpression::Pop(Box::new(VectorExpression::Reference(
            ReferenceExpression::Constant(vec![1, 2]),
        )));
        assert_eq!(expression.eval(&state, &registry), vec![1]);
    }

    #[test]
    fn indices_simplify() {
        let registry = generate_registry();

        let expression = VectorExpression::Indices(Box::new(VectorExpression::Reference(
            ReferenceExpression::Variable(0),
        )));
        assert_eq!(expression.simplify(&registry), expression);

        let expression = VectorExpression::Indices(Box::new(VectorExpression::Reference(
            ReferenceExpression::Constant(vec![1, 2]),
        )));
        assert_eq!(
            expression.simplify(&registry),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1]))
        );
    }

    #[test]
    fn reference_simplify() {
        let registry = generate_registry();
        let expression = VectorExpression::Reference(ReferenceExpression::Constant(vec![1, 2]));
        assert_eq!(expression.simplify(&registry), expression);
        let expression = VectorExpression::Reference(ReferenceExpression::Table(
            TableExpression::Table1D(0, ElementExpression::Constant(0)),
        ));
        assert_eq!(
            expression.simplify(&registry),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1]))
        );
    }

    #[test]
    fn push_simplify() {
        let registry = generate_registry();
        let expression = VectorExpression::Push(
            ElementExpression::Constant(0),
            Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![1, 2],
            ))),
        );
        assert_eq!(
            expression.simplify(&registry),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![1, 2, 0]))
        );
        let expression = VectorExpression::Push(
            ElementExpression::Constant(0),
            Box::new(VectorExpression::Reference(ReferenceExpression::Variable(
                0,
            ))),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn pop_simplify() {
        let registry = generate_registry();
        let expression = VectorExpression::Pop(Box::new(VectorExpression::Reference(
            ReferenceExpression::Constant(vec![1, 2]),
        )));
        assert_eq!(
            expression.simplify(&registry),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![1]))
        );
        let expression = VectorExpression::Pop(Box::new(VectorExpression::Reference(
            ReferenceExpression::Variable(0),
        )));
        assert_eq!(expression.simplify(&registry), expression);
    }
}
