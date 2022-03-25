use super::reference_expression::ReferenceExpression;
use crate::state::State;
use crate::table_data::TableData;
use crate::table_registry::TableRegistry;
use crate::variable::{Element, Vector};

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum ElementExpression {
    Stage,
    Constant(Element),
    Variable(usize),
    Last(Box<VectorExpression>),
    At(Box<VectorExpression>, Box<ElementExpression>),
    Table(Box<TableExpression<Element>>),
}

impl ElementExpression {
    pub fn eval(&self, state: &State, registry: &TableRegistry) -> Element {
        match self {
            Self::Stage => state.stage,
            Self::Constant(x) => *x,
            Self::Variable(i) => state.signature_variables.element_variables[*i],
            Self::Last(vector) => match vector.as_ref() {
                VectorExpression::Reference(vector) => *vector
                    .eval(
                        state,
                        registry,
                        &state.signature_variables.vector_variables,
                        &registry.vector_tables,
                    )
                    .last()
                    .unwrap(),
                vector => *vector.eval(state, registry).last().unwrap(),
            },
            Self::At(vector, i) => match vector.as_ref() {
                VectorExpression::Reference(vector) => vector.eval(
                    state,
                    registry,
                    &state.signature_variables.vector_variables,
                    &registry.vector_tables,
                )[i.eval(state, registry)],
                vector => vector.eval(state, registry)[i.eval(state, registry)],
            },
            Self::Table(table) => *table.eval(state, registry, &registry.element_tables),
        }
    }

    pub fn simplify(&self, registry: &TableRegistry) -> ElementExpression {
        match self {
            Self::Last(vector) => match vector.simplify(registry) {
                VectorExpression::Reference(ReferenceExpression::Constant(vector)) => {
                    Self::Constant(*vector.last().unwrap())
                }
                vector => Self::Last(Box::new(vector)),
            },
            Self::At(vector, i) => match (vector.simplify(registry), i.simplify(registry)) {
                (
                    VectorExpression::Reference(ReferenceExpression::Constant(vector)),
                    ElementExpression::Constant(i),
                ) => Self::Constant(vector[i]),
                (vector, i) => Self::At(Box::new(vector), Box::new(i)),
            },
            Self::Table(table) => match table.simplify(registry, &registry.element_tables) {
                TableExpression::Constant(value) => Self::Constant(value),
                expression => Self::Table(Box::new(expression)),
            },
            _ => self.clone(),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum VectorExpression {
    Reference(ReferenceExpression<Vector>),
    Indices(Box<VectorExpression>),
    Reverse(Box<VectorExpression>),
    Set(ElementExpression, Box<VectorExpression>, ElementExpression),
    Push(ElementExpression, Box<VectorExpression>),
    Pop(Box<VectorExpression>),
}

impl VectorExpression {
    pub fn eval(&self, state: &State, registry: &TableRegistry) -> Vector {
        match self {
            Self::Reference(expression) => expression
                .eval(
                    state,
                    registry,
                    &state.signature_variables.vector_variables,
                    &registry.vector_tables,
                )
                .clone(),
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
        }
    }

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
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum TableExpression<T: Clone> {
    Constant(T),
    Table1D(usize, ElementExpression),
    Table2D(usize, ElementExpression, ElementExpression),
    Table3D(
        usize,
        ElementExpression,
        ElementExpression,
        ElementExpression,
    ),
    Table(usize, Vec<ElementExpression>),
}

impl<T: Clone> TableExpression<T> {
    pub fn eval<'a>(
        &'a self,
        state: &State,
        registry: &'a TableRegistry,
        tables: &'a TableData<T>,
    ) -> &'a T {
        match self {
            Self::Constant(value) => value,
            Self::Table1D(i, x) => tables.tables_1d[*i].get(x.eval(state, registry)),
            Self::Table2D(i, x, y) => {
                tables.tables_2d[*i].get(x.eval(state, registry), y.eval(state, registry))
            }
            Self::Table3D(i, x, y, z) => tables.tables_3d[*i].get(
                x.eval(state, registry),
                y.eval(state, registry),
                z.eval(state, registry),
            ),
            Self::Table(i, args) => {
                let args: Vec<Element> = args.iter().map(|x| x.eval(state, registry)).collect();
                tables.tables[*i].get(&args)
            }
        }
    }

    pub fn simplify(&self, registry: &TableRegistry, tables: &TableData<T>) -> TableExpression<T> {
        match self {
            Self::Table1D(i, x) => match x.simplify(registry) {
                ElementExpression::Constant(x) => {
                    Self::Constant(tables.tables_1d[*i].get(x).clone())
                }
                x => Self::Table1D(*i, x),
            },
            Self::Table2D(i, x, y) => match (x.simplify(registry), y.simplify(registry)) {
                (ElementExpression::Constant(x), ElementExpression::Constant(y)) => {
                    Self::Constant(tables.tables_2d[*i].get(x, y).clone())
                }
                (x, y) => Self::Table2D(*i, x, y),
            },
            Self::Table3D(i, x, y, z) => match (
                x.simplify(registry),
                y.simplify(registry),
                z.simplify(registry),
            ) {
                (
                    ElementExpression::Constant(x),
                    ElementExpression::Constant(y),
                    ElementExpression::Constant(z),
                ) => Self::Constant(tables.tables_3d[*i].get(x, y, z).clone()),
                (x, y, z) => Self::Table3D(*i, x, y, z),
            },
            Self::Table(i, args) => {
                let mut simplified_args = Vec::with_capacity(args.len());
                for arg in args {
                    match arg.simplify(registry) {
                        ElementExpression::Constant(arg) => simplified_args.push(arg),
                        _ => return self.clone(),
                    }
                }
                Self::Constant(tables.tables[*i].get(&simplified_args).clone())
            }
            _ => self.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::*;
    use crate::table::*;
    use crate::variable::*;
    use std::collections::HashMap;
    use std::rc::Rc;

    fn generate_registry() -> TableRegistry {
        let mut name_to_constant = HashMap::new();
        name_to_constant.insert(String::from("f0"), 1);

        let tables_1d = vec![Table1D::new(vec![1, 0])];
        let mut name_to_table_1d = HashMap::new();
        name_to_table_1d.insert(String::from("f1"), 0);

        let tables_2d = vec![Table2D::new(vec![vec![1, 0]])];
        let mut name_to_table_2d = HashMap::new();
        name_to_table_2d.insert(String::from("f2"), 0);

        let tables_3d = vec![Table3D::new(vec![vec![vec![1, 0]]])];
        let mut name_to_table_3d = HashMap::new();
        name_to_table_3d.insert(String::from("f3"), 0);

        let mut map = HashMap::new();
        let key = vec![0, 0, 0, 0];
        map.insert(key, 1);
        let key = vec![0, 0, 0, 1];
        map.insert(key, 0);
        let tables = vec![Table::new(map, 0)];
        let mut name_to_table = HashMap::new();
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

        let mut name_to_table_1d = HashMap::new();
        name_to_table_1d.insert(String::from("t1"), 0);
        let vector_tables = TableData {
            tables_1d: vec![Table1D::new(vec![vec![0, 1]])],
            name_to_table_1d,
            ..Default::default()
        };

        TableRegistry {
            element_tables,
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
            signature_variables: Rc::new(SignatureVariables {
                set_variables: vec![set1, set2],
                vector_variables: vec![vec![0, 2]],
                element_variables: vec![1],
                ..Default::default()
            }),
            stage: 1,
            ..Default::default()
        }
    }

    #[test]
    fn stage_eval() {
        let state = generate_state();
        let registry = generate_registry();
        let expression = ElementExpression::Stage;
        assert_eq!(expression.eval(&state, &registry), 1);
    }

    #[test]
    fn element_constant_eval() {
        let state = generate_state();
        let registry = generate_registry();
        let expression = ElementExpression::Constant(2);
        assert_eq!(expression.eval(&state, &registry), 2);
    }

    #[test]
    fn element_variable_eval() {
        let state = generate_state();
        let registry = generate_registry();
        let expression = ElementExpression::Variable(0);
        assert_eq!(expression.eval(&state, &registry), 1);
    }

    #[test]
    fn element_last_eval() {
        let state = generate_state();
        let registry = generate_registry();
        let expression = ElementExpression::Last(Box::new(VectorExpression::Reference(
            ReferenceExpression::Constant(vec![0, 1]),
        )));
        assert_eq!(expression.eval(&state, &registry), 1);
    }

    #[test]
    fn element_at_eval() {
        let state = generate_state();
        let registry = generate_registry();
        let expression = ElementExpression::At(
            Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![0, 1],
            ))),
            Box::new(ElementExpression::Constant(0)),
        );
        assert_eq!(expression.eval(&state, &registry), 0);
    }

    #[test]
    fn element_table_eval() {
        let state = generate_state();
        let registry = generate_registry();
        let expression = ElementExpression::Table(Box::new(TableExpression::Constant(0)));
        assert_eq!(expression.eval(&state, &registry), 0);
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
    fn table_constant_eval() {
        let registry = generate_registry();
        let state = generate_state();

        let expression = TableExpression::Constant(1);
        assert_eq!(
            *expression.eval(&state, &registry, &registry.element_tables),
            1
        );
    }

    #[test]
    fn table_1d_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let expression = TableExpression::Table1D(0, ElementExpression::Constant(0));
        assert_eq!(
            *expression.eval(&state, &registry, &registry.element_tables),
            1
        );
        let expression = TableExpression::Table1D(0, ElementExpression::Constant(1));
        assert_eq!(
            *expression.eval(&state, &registry, &registry.element_tables),
            0
        );
    }

    #[test]
    fn table_2d_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let expression = TableExpression::Table2D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
        );
        assert_eq!(
            *expression.eval(&state, &registry, &registry.element_tables),
            1
        );
        let expression = TableExpression::Table2D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(1),
        );
        assert_eq!(
            *expression.eval(&state, &registry, &registry.element_tables),
            0
        );
    }

    #[test]
    fn table_3d_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let expression = TableExpression::Table3D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
        );
        assert_eq!(
            *expression.eval(&state, &registry, &registry.element_tables),
            1
        );
        let expression = TableExpression::Table3D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
            ElementExpression::Constant(1),
        );
        assert_eq!(
            *expression.eval(&state, &registry, &registry.element_tables),
            0
        );
    }

    #[test]
    fn table_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let expression = TableExpression::Table(
            0,
            vec![
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
            ],
        );
        assert_eq!(
            *expression.eval(&state, &registry, &registry.element_tables),
            1
        );
        let expression = TableExpression::Table(
            0,
            vec![
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(1),
            ],
        );
        assert_eq!(
            *expression.eval(&state, &registry, &registry.element_tables),
            0
        );
        let expression = TableExpression::Table(
            0,
            vec![
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(2),
            ],
        );
        assert_eq!(
            *expression.eval(&state, &registry, &registry.element_tables),
            0
        );
    }

    #[test]
    fn element_simplify() {
        let registry = generate_registry();
        let expression = ElementExpression::Stage;
        assert_eq!(expression.simplify(&registry), expression);
        let expression = ElementExpression::Constant(0);
        assert_eq!(expression.simplify(&registry), expression);
        let expression = ElementExpression::Variable(0);
        assert_eq!(expression.simplify(&registry), expression);
        let expression = ElementExpression::Table(Box::new(TableExpression::Constant(1)));
        assert_eq!(
            expression.simplify(&registry),
            ElementExpression::Constant(1)
        );
    }

    #[test]
    fn constant_simplify() {
        let registry = generate_registry();
        let expression = TableExpression::Constant(1);
        assert_eq!(
            expression.simplify(&registry, &registry.element_tables),
            expression
        );
    }

    #[test]
    fn table_1d_simplify() {
        let registry = generate_registry();

        let expression = TableExpression::Table1D(0, ElementExpression::Constant(0));
        assert_eq!(
            expression.simplify(&registry, &registry.element_tables),
            TableExpression::Constant(1)
        );

        let expression = TableExpression::Table1D(0, ElementExpression::Constant(1));
        assert_eq!(
            expression.simplify(&registry, &registry.element_tables),
            TableExpression::Constant(0)
        );

        let expression = TableExpression::Table1D(0, ElementExpression::Variable(0));
        assert_eq!(
            expression.simplify(&registry, &registry.element_tables),
            expression
        );
    }

    #[test]
    fn table_2d_simplify() {
        let registry = generate_registry();

        let expression = TableExpression::Table2D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
        );
        assert_eq!(
            expression.simplify(&registry, &registry.element_tables),
            TableExpression::Constant(1)
        );

        let expression = TableExpression::Table2D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(1),
        );
        assert_eq!(
            expression.simplify(&registry, &registry.element_tables),
            TableExpression::Constant(0)
        );

        let expression = TableExpression::Table2D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Variable(0),
        );
        assert_eq!(
            expression.simplify(&registry, &registry.element_tables),
            expression
        );
    }

    #[test]
    fn table_3d_simplify() {
        let registry = generate_registry();

        let expression = TableExpression::Table3D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
        );
        assert_eq!(
            expression.simplify(&registry, &registry.element_tables),
            TableExpression::Constant(1)
        );

        let expression = TableExpression::Table3D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
            ElementExpression::Constant(1),
        );
        assert_eq!(
            expression.simplify(&registry, &registry.element_tables),
            TableExpression::Constant(0)
        );

        let expression = TableExpression::Table3D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
            ElementExpression::Variable(0),
        );
        assert_eq!(
            expression.simplify(&registry, &registry.element_tables),
            expression
        );
    }

    #[test]
    fn table_simplify() {
        let registry = generate_registry();

        let expression = TableExpression::Table(
            0,
            vec![
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
            ],
        );
        assert_eq!(
            expression.simplify(&registry, &registry.element_tables),
            TableExpression::Constant(1)
        );

        let expression = TableExpression::Table(
            0,
            vec![
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(1),
            ],
        );
        assert_eq!(
            expression.simplify(&registry, &registry.element_tables),
            TableExpression::Constant(0)
        );

        let expression = TableExpression::Table(
            0,
            vec![
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(2),
            ],
        );
        assert_eq!(
            expression.simplify(&registry, &registry.element_tables),
            TableExpression::Constant(0)
        );

        let expression = TableExpression::Table(
            0,
            vec![
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Variable(0),
            ],
        );
        assert_eq!(
            expression.simplify(&registry, &registry.element_tables),
            expression
        );
    }
}
