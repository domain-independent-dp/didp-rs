use super::condition::Condition;
use super::element_expression::ElementExpression;
use super::reference_expression::ReferenceExpression;
use super::set_expression::SetExpression;
use crate::state::StateInterface;
use crate::table_registry::TableRegistry;
use crate::variable_type::Vector;

/// Vector expression.
#[derive(Debug, PartialEq, Clone)]
pub enum VectorExpression {
    /// Reference to a constant or a variable.
    Reference(ReferenceExpression<Vector>),
    /// Indices of a vector.
    Indices(Box<VectorExpression>),
    /// Reverse a vector.
    Reverse(Box<VectorExpression>),
    /// Set an element in a vector.
    Set(ElementExpression, Box<VectorExpression>, ElementExpression),
    /// Push an element to a vector.
    Push(ElementExpression, Box<VectorExpression>),
    /// Pop an element from a vector.
    Pop(Box<VectorExpression>),
    /// Conversion from a set.
    FromSet(Box<SetExpression>),
    /// If-then-else expression, which returns the first one if the condition holds and the second one otherwise.
    If(Box<Condition>, Box<VectorExpression>, Box<VectorExpression>),
}

impl VectorExpression {
    /// Returns the evaluation result.
    ///
    /// # Panics
    ///
    /// Panics if the cost of the transition state is used or a min/max reduce operation is performed on an empty set or vector.
    pub fn eval<T: StateInterface>(&self, state: &T, registry: &TableRegistry) -> Vector {
        match self {
            Self::Reference(expression) => {
                let f = |i| state.get_vector_variable(i);
                expression
                    .eval(state, registry, &f, &registry.vector_tables)
                    .clone()
            }
            Self::Indices(vector) => {
                let mut vector = vector.eval(state, registry);
                vector.iter_mut().enumerate().for_each(|(i, v)| *v = i);
                vector
            }
            Self::Reverse(vector) => {
                let mut vector = vector.eval(state, registry);
                vector.reverse();
                vector
            }
            Self::Set(element, vector, i) => {
                let mut vector = vector.eval(state, registry);
                vector[i.eval(state, registry)] = element.eval(state, registry);
                vector
            }
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
            Self::FromSet(set) => match set.as_ref() {
                SetExpression::Reference(set) => {
                    let f = |i| state.get_set_variable(i);
                    set.eval(state, registry, &f, &registry.set_tables)
                        .ones()
                        .collect()
                }
                set => set.eval(state, registry).ones().collect(),
            },
            Self::If(condition, x, y) => {
                if condition.eval(state, registry) {
                    x.eval(state, registry)
                } else {
                    y.eval(state, registry)
                }
            }
        }
    }

    /// Returns a simplified version by precomputation.
    ///
    /// # Panics
    ///
    /// Panics if the cost of the transition state is used or a min/max reduce operation is performed on an empty set or vector.
    pub fn simplify(&self, registry: &TableRegistry) -> VectorExpression {
        match self {
            Self::Reference(vector) => {
                Self::Reference(vector.simplify(registry, &registry.vector_tables))
            }
            Self::Indices(vector) => match vector.simplify(registry) {
                VectorExpression::Reference(ReferenceExpression::Constant(mut vector)) => {
                    vector.iter_mut().enumerate().for_each(|(i, v)| *v = i);
                    Self::Reference(ReferenceExpression::Constant(vector))
                }
                vector => Self::Indices(Box::new(vector)),
            },
            Self::Reverse(vector) => match vector.simplify(registry) {
                VectorExpression::Reference(ReferenceExpression::Constant(mut vector)) => {
                    vector.reverse();
                    Self::Reference(ReferenceExpression::Constant(vector))
                }
                vector => Self::Reverse(Box::new(vector)),
            },
            Self::Set(element, vector, i) => match (
                element.simplify(registry),
                vector.simplify(registry),
                i.simplify(registry),
            ) {
                (
                    ElementExpression::Constant(element),
                    VectorExpression::Reference(ReferenceExpression::Constant(mut vector)),
                    ElementExpression::Constant(i),
                ) => {
                    vector[i] = element;
                    Self::Reference(ReferenceExpression::Constant(vector))
                }
                (element, vector, i) => Self::Set(element, Box::new(vector), i),
            },
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
            Self::FromSet(set) => match set.simplify(registry) {
                SetExpression::Reference(ReferenceExpression::Constant(set)) => {
                    Self::Reference(ReferenceExpression::Constant(set.ones().collect()))
                }
                set => Self::FromSet(Box::new(set)),
            },
            Self::If(condition, x, y) => match condition.simplify(registry) {
                Condition::Constant(true) => x.simplify(registry),
                Condition::Constant(false) => y.simplify(registry),
                condition => Self::If(
                    Box::new(condition),
                    Box::new(x.simplify(registry)),
                    Box::new(y.simplify(registry)),
                ),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::condition::ComparisonOperator;
    use super::super::integer_expression::IntegerExpression;
    use super::super::table_expression::TableExpression;
    use super::*;
    use crate::state::*;
    use crate::table::*;
    use crate::table_data::*;
    use crate::variable_type::Set;
    use rustc_hash::FxHashMap;

    fn generate_registry() -> TableRegistry {
        let mut name_to_constant = FxHashMap::default();
        name_to_constant.insert(String::from("f0"), 1);

        let tables_1d = vec![Table1D::new(vec![1, 0])];
        let mut name_to_table_1d = FxHashMap::default();
        name_to_table_1d.insert(String::from("f1"), 0);

        let tables_2d = vec![Table2D::new(vec![vec![1, 0]])];
        let mut name_to_table_2d = FxHashMap::default();
        name_to_table_2d.insert(String::from("f2"), 0);

        let tables_3d = vec![Table3D::new(vec![vec![vec![1, 0]]])];
        let mut name_to_table_3d = FxHashMap::default();
        name_to_table_3d.insert(String::from("f3"), 0);

        let mut map = FxHashMap::default();
        let key = vec![0, 0, 0, 0];
        map.insert(key, 1);
        let key = vec![0, 0, 0, 1];
        map.insert(key, 0);
        let tables = vec![Table::new(map, 0)];
        let mut name_to_table = FxHashMap::default();
        name_to_table.insert(String::from("f4"), 0);

        let element_tables = TableData {
            name_to_constant,
            tables_1d,
            name_to_table_1d,
            tables_2d,
            name_to_table_2d,
            tables_3d,
            name_to_table_3d,
            tables,
            name_to_table,
        };

        let mut name_to_table_1d = FxHashMap::default();
        name_to_table_1d.insert(String::from("t1"), 0);
        let vector_tables = TableData {
            tables_1d: vec![Table1D::new(vec![vec![0, 1]])],
            name_to_table_1d,
            ..Default::default()
        };

        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(2);
        let default = Set::with_capacity(3);
        let tables_1d = vec![Table1D::new(vec![set, default.clone(), default])];
        let mut name_to_table_1d = FxHashMap::default();
        name_to_table_1d.insert(String::from("s1"), 0);
        let set_tables = TableData {
            tables_1d,
            name_to_table_1d,
            ..Default::default()
        };

        TableRegistry {
            element_tables,
            set_tables,
            vector_tables,
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
            signature_variables: SignatureVariables {
                set_variables: vec![set1, set2],
                vector_variables: vec![vec![0, 2]],
                element_variables: vec![1],
                ..Default::default()
            },
            resource_variables: ResourceVariables {
                element_variables: vec![2],
                ..Default::default()
            },
        }
    }

    #[test]
    fn vector_reference_eval() {
        let state = generate_state();
        let registry = generate_registry();
        let expression = VectorExpression::Reference(ReferenceExpression::Constant(vec![1, 2]));
        assert_eq!(expression.eval(&state, &registry), vec![1, 2]);
    }

    #[test]
    fn vector_indices_eval() {
        let state = generate_state();
        let registry = generate_registry();
        let expression = VectorExpression::Indices(Box::new(VectorExpression::Reference(
            ReferenceExpression::Constant(vec![1, 2]),
        )));
        assert_eq!(expression.eval(&state, &registry), vec![0, 1]);
    }

    #[test]
    fn vector_reverse_eval() {
        let state = generate_state();
        let registry = generate_registry();
        let expression = VectorExpression::Reverse(Box::new(VectorExpression::Reference(
            ReferenceExpression::Constant(vec![1, 2]),
        )));
        assert_eq!(expression.eval(&state, &registry), vec![2, 1]);
    }

    #[test]
    fn vector_set_eval() {
        let state = generate_state();
        let registry = generate_registry();
        let expression = VectorExpression::Set(
            ElementExpression::Constant(3),
            Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![1, 2],
            ))),
            ElementExpression::Constant(0),
        );
        assert_eq!(expression.eval(&state, &registry), vec![3, 2]);
    }

    #[test]
    fn vector_push_eval() {
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
    fn vector_pop_eval() {
        let state = generate_state();
        let registry = generate_registry();
        let expression = VectorExpression::Pop(Box::new(VectorExpression::Reference(
            ReferenceExpression::Constant(vec![1, 2]),
        )));
        assert_eq!(expression.eval(&state, &registry), vec![1]);
    }

    #[test]
    fn vector_from_set_eval() {
        let state = generate_state();
        let registry = generate_registry();
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(1);
        let expression = VectorExpression::FromSet(Box::new(SetExpression::Reference(
            ReferenceExpression::Constant(set),
        )));
        assert_eq!(expression.eval(&state, &registry), vec![0, 1]);
        let expression = VectorExpression::FromSet(Box::new(SetExpression::Reference(
            ReferenceExpression::Variable(0),
        )));
        assert_eq!(expression.eval(&state, &registry), vec![0, 2]);
    }

    #[test]
    fn vector_if_eval() {
        let state = generate_state();
        let registry = generate_registry();
        let expression = VectorExpression::If(
            Box::new(Condition::Constant(true)),
            Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![0, 1],
            ))),
            Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![1, 0],
            ))),
        );
        assert_eq!(expression.eval(&state, &registry), vec![0, 1]);
        let expression = VectorExpression::If(
            Box::new(Condition::Constant(false)),
            Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![0, 1],
            ))),
            Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![1, 0],
            ))),
        );
        assert_eq!(expression.eval(&state, &registry), vec![1, 0]);
    }

    #[test]
    fn vector_reference_simplify() {
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
    fn vector_indices_simplify() {
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
    fn vector_push_simplify() {
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
    fn vector_pop_simplify() {
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

    #[test]
    fn vector_set_simplify() {
        let registry = generate_registry();
        let expression = VectorExpression::Set(
            ElementExpression::Constant(0),
            Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![1, 2],
            ))),
            ElementExpression::Constant(0),
        );
        assert_eq!(
            expression.simplify(&registry),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 2]))
        );
        let expression = VectorExpression::Set(
            ElementExpression::Constant(0),
            Box::new(VectorExpression::Reference(ReferenceExpression::Variable(
                0,
            ))),
            ElementExpression::Variable(0),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn vector_from_set_simplify() {
        let registry = generate_registry();
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(1);
        let expression = VectorExpression::FromSet(Box::new(SetExpression::Reference(
            ReferenceExpression::Constant(set),
        )));
        assert_eq!(
            expression.simplify(&registry),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1]))
        );
        let expression = VectorExpression::FromSet(Box::new(SetExpression::Reference(
            ReferenceExpression::Variable(0),
        )));
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn vector_if_simplify() {
        let registry = generate_registry();
        let expression = VectorExpression::If(
            Box::new(Condition::Constant(true)),
            Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![0, 1],
            ))),
            Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![1, 0],
            ))),
        );
        assert_eq!(
            expression.simplify(&registry),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1]))
        );
        let expression = VectorExpression::If(
            Box::new(Condition::Constant(false)),
            Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![0, 1],
            ))),
            Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![1, 0],
            ))),
        );
        assert_eq!(
            expression.simplify(&registry),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![1, 0]))
        );
        let expression = VectorExpression::If(
            Box::new(Condition::ComparisonI(
                ComparisonOperator::Gt,
                Box::new(IntegerExpression::Variable(0)),
                Box::new(IntegerExpression::Constant(1)),
            )),
            Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![0, 1],
            ))),
            Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![1, 0],
            ))),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }
}
