use super::element_expression::ElementExpression;
use super::reference_expression::ReferenceExpression;
use crate::state::State;
use crate::table_registry::TableRegistry;
use crate::variable::{Element, Set};

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum SetExpression {
    Reference(ReferenceExpression<Set>),
    Complement(Box<SetExpression>),
    SetOperation(SetOperator, Box<SetExpression>, Box<SetExpression>),
    SetElementOperation(SetElementOperator, ElementExpression, Box<SetExpression>),
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum SetOperator {
    Union,
    Difference,
    Intersection,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum SetElementOperator {
    Add,
    Remove,
}

impl SetExpression {
    pub fn eval(&self, state: &State, registry: &TableRegistry) -> Set {
        match self {
            SetExpression::Reference(expression) => expression
                .eval(
                    state,
                    registry,
                    &state.signature_variables.set_variables,
                    &registry.set_tables,
                )
                .clone(),
            SetExpression::Complement(set) => {
                let mut set = set.eval(state, registry);
                set.toggle_range(..);
                set
            }
            SetExpression::SetOperation(op, x, y) => {
                let x = x.eval(state, registry);
                let y = y.eval(state, registry);
                Self::eval_set_operation(op, x, y)
            }
            SetExpression::SetElementOperation(op, element, set) => {
                let set = set.eval(state, registry);
                let element = element.eval(state, registry);
                Self::eval_set_element_operation(op, element, set)
            }
        }
    }

    pub fn simplify(&self, registry: &TableRegistry) -> SetExpression {
        match self {
            Self::Reference(expression) => {
                Self::Reference(expression.simplify(registry, &registry.set_tables))
            }
            Self::Complement(expression) => match expression.simplify(registry) {
                Self::Reference(ReferenceExpression::Constant(mut set)) => {
                    set.toggle_range(..);
                    Self::Reference(ReferenceExpression::Constant(set))
                }
                Self::Complement(expression) => *expression,
                expression => Self::Complement(Box::new(expression)),
            },
            Self::SetOperation(op, x, y) => match (x.simplify(registry), y.simplify(registry)) {
                (
                    Self::Reference(ReferenceExpression::Constant(x)),
                    Self::Reference(ReferenceExpression::Constant(y)),
                ) => Self::Reference(ReferenceExpression::Constant(Self::eval_set_operation(
                    op, x, y,
                ))),
                (
                    Self::Reference(ReferenceExpression::Variable(x)),
                    Self::Reference(ReferenceExpression::Variable(y)),
                ) if x == y => match op {
                    SetOperator::Union | SetOperator::Intersection => {
                        Self::Reference(ReferenceExpression::Variable(x))
                    }
                    op => Self::SetOperation(
                        op.clone(),
                        Box::new(Self::Reference(ReferenceExpression::Variable(x))),
                        Box::new(Self::Reference(ReferenceExpression::Variable(y))),
                    ),
                },
                (x, y) => Self::SetOperation(op.clone(), Box::new(x), Box::new(y)),
            },
            Self::SetElementOperation(op, element, set) => {
                match (set.simplify(registry), element.simplify(registry)) {
                    (
                        Self::Reference(ReferenceExpression::Constant(set)),
                        ElementExpression::Constant(element),
                    ) => Self::Reference(ReferenceExpression::Constant(
                        Self::eval_set_element_operation(op, element, set),
                    )),
                    (set, element) => Self::SetElementOperation(op.clone(), element, Box::new(set)),
                }
            }
        }
    }

    fn eval_set_operation(op: &SetOperator, mut x: Set, y: Set) -> Set {
        match op {
            SetOperator::Union => {
                x.union_with(&y);
                x
            }
            SetOperator::Difference => {
                x.difference_with(&y);
                x
            }
            SetOperator::Intersection => {
                x.intersect_with(&y);
                x
            }
        }
    }

    fn eval_set_element_operation(op: &SetElementOperator, element: Element, mut set: Set) -> Set {
        match op {
            SetElementOperator::Add => {
                set.insert(element);
                set
            }
            SetElementOperator::Remove => {
                set.set(element, false);
                set
            }
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
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(2);
        let default = Set::with_capacity(3);
        let tables_1d = vec![Table1D::new(vec![set, default.clone(), default])];
        let mut name_to_table_1d = HashMap::new();
        name_to_table_1d.insert(String::from("s1"), 0);
        TableRegistry {
            set_tables: TableData {
                tables_1d,
                name_to_table_1d,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn generate_state() -> State {
        let mut set1 = Set::with_capacity(3);
        set1.insert(0);
        set1.insert(2);
        let mut set2 = Set::with_capacity(3);
        set2.insert(0);
        set2.insert(1);
        State {
            signature_variables: Rc::new(SignatureVariables {
                set_variables: vec![set1, set2],
                vector_variables: vec![vec![0, 2]],
                element_variables: vec![1],
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[test]
    fn reference_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(2);
        let expression = SetExpression::Reference(ReferenceExpression::Constant(set.clone()));
        assert_eq!(expression.eval(&state, &registry), set);
        let expression = SetExpression::Reference(ReferenceExpression::Variable(0));
        assert_eq!(
            expression.eval(&state, &registry),
            state.signature_variables.set_variables[0]
        );
        let expression = SetExpression::Reference(ReferenceExpression::Table(
            TableExpression::Table1D(0, ElementExpression::Constant(0)),
        ));
        assert_eq!(expression.eval(&state, &registry), set);
    }

    #[test]
    fn complement_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let expression = SetExpression::Complement(Box::new(SetExpression::Reference(
            ReferenceExpression::Variable(0),
        )));
        let mut set = Set::with_capacity(3);
        set.insert(1);
        assert_eq!(expression.eval(&state, &registry), set);
    }

    #[test]
    fn union_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let expression = SetExpression::SetOperation(
            SetOperator::Union,
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
        );
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(1);
        set.insert(2);
        assert_eq!(expression.eval(&state, &registry), set);
        let expression = SetExpression::SetOperation(
            SetOperator::Union,
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
        );
        assert_eq!(
            expression.eval(&state, &registry),
            state.signature_variables.set_variables[0]
        );
    }

    #[test]
    fn difference_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let expression = SetExpression::SetOperation(
            SetOperator::Difference,
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
        );
        let mut set = Set::with_capacity(3);
        set.insert(2);
        assert_eq!(expression.eval(&state, &registry), set);
        let expression = SetExpression::SetOperation(
            SetOperator::Difference,
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
        );
        assert_eq!(expression.eval(&state, &registry), Set::with_capacity(3));
    }

    #[test]
    fn intersect_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let expression = SetExpression::SetOperation(
            SetOperator::Intersection,
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
        );
        let mut set = Set::with_capacity(3);
        set.insert(0);
        assert_eq!(expression.eval(&state, &registry), set);
        let expression = SetExpression::SetOperation(
            SetOperator::Intersection,
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
        );
        assert_eq!(
            expression.eval(&state, &registry),
            state.signature_variables.set_variables[0]
        );
    }

    #[test]
    fn set_add_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let expression = SetExpression::SetElementOperation(
            SetElementOperator::Add,
            ElementExpression::Constant(1),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
        );
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(1);
        set.insert(2);
        assert_eq!(expression.eval(&state, &registry), set);
        let expression = SetExpression::SetElementOperation(
            SetElementOperator::Add,
            ElementExpression::Constant(0),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
        );
        assert_eq!(
            expression.eval(&state, &registry),
            state.signature_variables.set_variables[0]
        );
    }

    #[test]
    fn set_remove_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let expression = SetExpression::SetElementOperation(
            SetElementOperator::Remove,
            ElementExpression::Constant(2),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
        );
        let mut set = Set::with_capacity(3);
        set.insert(0);
        assert_eq!(expression.eval(&state, &registry), set);
        let expression = SetExpression::SetElementOperation(
            SetElementOperator::Remove,
            ElementExpression::Constant(1),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
        );
        assert_eq!(
            expression.eval(&state, &registry),
            state.signature_variables.set_variables[0]
        );
    }

    #[test]
    fn reference_simplify() {
        let registry = generate_registry();
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(2);
        let expression = SetExpression::Reference(ReferenceExpression::Constant(set.clone()));
        assert_eq!(expression.simplify(&registry), expression);
        let expression = SetExpression::Reference(ReferenceExpression::Variable(0));
        assert_eq!(expression.simplify(&registry), expression);
        let expression = SetExpression::Reference(ReferenceExpression::Table(
            TableExpression::Table1D(0, ElementExpression::Constant(0)),
        ));
        assert_eq!(
            expression.simplify(&registry),
            SetExpression::Reference(ReferenceExpression::Constant(set))
        );
    }

    #[test]
    fn complement_simplify() {
        let registry = generate_registry();
        let expression = SetExpression::Complement(Box::new(SetExpression::Reference(
            ReferenceExpression::Variable(0),
        )));
        assert_eq!(expression.simplify(&registry), expression);
        let mut set = Set::with_capacity(3);
        set.insert(1);
        let expression = SetExpression::Complement(Box::new(SetExpression::Reference(
            ReferenceExpression::Constant(set),
        )));
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(2);
        assert_eq!(
            expression.simplify(&registry),
            SetExpression::Reference(ReferenceExpression::Constant(set))
        );
        let expression = SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        ))));
        assert_eq!(
            expression.simplify(&registry),
            SetExpression::Reference(ReferenceExpression::Variable(0))
        );
    }

    #[test]
    fn union_simplify() {
        let registry = generate_registry();
        let expression = SetExpression::SetOperation(
            SetOperator::Union,
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
        );
        assert_eq!(expression.simplify(&registry), expression);
        let mut x = Set::with_capacity(3);
        x.insert(0);
        x.insert(2);
        let mut y = Set::with_capacity(3);
        y.insert(0);
        y.insert(1);
        let expression = SetExpression::SetOperation(
            SetOperator::Union,
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(x))),
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(y))),
        );
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(1);
        set.insert(2);
        assert_eq!(
            expression.simplify(&registry),
            SetExpression::Reference(ReferenceExpression::Constant(set))
        );
        let expression = SetExpression::SetOperation(
            SetOperator::Union,
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
        );
        assert_eq!(
            expression.simplify(&registry),
            SetExpression::Reference(ReferenceExpression::Variable(0))
        );
    }

    #[test]
    fn difference_simplify() {
        let registry = generate_registry();
        let expression = SetExpression::SetOperation(
            SetOperator::Difference,
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
        );
        assert_eq!(expression.simplify(&registry), expression);
        let mut x = Set::with_capacity(3);
        x.insert(0);
        x.insert(2);
        let mut y = Set::with_capacity(3);
        y.insert(0);
        y.insert(1);
        let expression = SetExpression::SetOperation(
            SetOperator::Difference,
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(x))),
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(y))),
        );
        let mut set = Set::with_capacity(3);
        set.insert(2);
        assert_eq!(
            expression.simplify(&registry),
            SetExpression::Reference(ReferenceExpression::Constant(set))
        );
    }

    #[test]
    fn intersect_simplify() {
        let registry = generate_registry();
        let expression = SetExpression::SetOperation(
            SetOperator::Intersection,
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
        );
        assert_eq!(expression.simplify(&registry), expression);
        let mut x = Set::with_capacity(3);
        x.insert(0);
        x.insert(2);
        let mut y = Set::with_capacity(3);
        y.insert(0);
        y.insert(1);
        let expression = SetExpression::SetOperation(
            SetOperator::Intersection,
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(x))),
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(y))),
        );
        let mut set = Set::with_capacity(3);
        set.insert(0);
        assert_eq!(
            expression.simplify(&registry),
            SetExpression::Reference(ReferenceExpression::Constant(set))
        );
        let expression = SetExpression::SetOperation(
            SetOperator::Union,
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
        );
        assert_eq!(
            expression.simplify(&registry),
            SetExpression::Reference(ReferenceExpression::Variable(0))
        );
    }

    #[test]
    fn set_add_simplify() {
        let registry = generate_registry();
        let expression = SetExpression::SetElementOperation(
            SetElementOperator::Add,
            ElementExpression::Constant(1),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
        );
        assert_eq!(expression.simplify(&registry), expression);
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(2);
        let expression = SetExpression::SetElementOperation(
            SetElementOperator::Add,
            ElementExpression::Constant(1),
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(set))),
        );
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(1);
        set.insert(2);
        assert_eq!(
            expression.simplify(&registry),
            SetExpression::Reference(ReferenceExpression::Constant(set))
        );
    }

    #[test]
    fn set_remove_simplify() {
        let registry = generate_registry();
        let expression = SetExpression::SetElementOperation(
            SetElementOperator::Remove,
            ElementExpression::Constant(2),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
        );
        assert_eq!(expression.simplify(&registry), expression);
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(2);
        let expression = SetExpression::SetElementOperation(
            SetElementOperator::Remove,
            ElementExpression::Constant(2),
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(set))),
        );
        let mut set = Set::with_capacity(3);
        set.insert(0);
        assert_eq!(
            expression.simplify(&registry),
            SetExpression::Reference(ReferenceExpression::Constant(set))
        );
    }
}
