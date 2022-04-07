use super::element_expression::{ElementExpression, SetExpression, VectorExpression};
use super::reference_expression::ReferenceExpression;
use super::util;
use crate::state::State;
use crate::table_data::TableData;
use crate::table_registry::TableRegistry;
use crate::variable::{Element, Numeric};

#[derive(Debug, PartialEq, Clone)]
pub enum ArgumentExpression {
    Set(SetExpression),
    Vector(VectorExpression),
    Element(ElementExpression),
}

impl ArgumentExpression {
    pub fn simplify(&self, registry: &TableRegistry) -> ArgumentExpression {
        match self {
            Self::Set(expression) => ArgumentExpression::Set(expression.simplify(registry)),
            Self::Vector(expression) => ArgumentExpression::Vector(expression.simplify(registry)),
            Self::Element(expression) => ArgumentExpression::Element(expression.simplify(registry)),
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum NumericTableExpression<T: Numeric> {
    Constant(T),
    Table(usize, Vec<ElementExpression>),
    TableSum(usize, Vec<ArgumentExpression>),
    Table1D(usize, ElementExpression),
    Table2D(usize, ElementExpression, ElementExpression),
    Table1DSum(usize, SetExpression),
    Table1DVectorSum(usize, VectorExpression),
    Table2DSum(usize, SetExpression, SetExpression),
    Table2DVectorSum(usize, VectorExpression, VectorExpression),
    Table2DSetVectorSum(usize, SetExpression, VectorExpression),
    Table2DVectorSetSum(usize, VectorExpression, SetExpression),
    Table2DSumX(usize, SetExpression, ElementExpression),
    Table2DSumY(usize, ElementExpression, SetExpression),
    Table2DVectorSumX(usize, VectorExpression, ElementExpression),
    Table2DVectorSumY(usize, ElementExpression, VectorExpression),
}

impl<T: Numeric> NumericTableExpression<T> {
    pub fn eval(&self, state: &State, registry: &TableRegistry, tables: &TableData<T>) -> T {
        let set_variables = &state.signature_variables.set_variables;
        let set_tables = &registry.set_tables;
        let vector_variables = &state.signature_variables.vector_variables;
        let vector_tables = &registry.vector_tables;
        match self {
            Self::Constant(value) => *value,
            Self::Table(i, args) => {
                let args: Vec<Element> = args.iter().map(|x| x.eval(state, registry)).collect();
                match args.len() {
                    1 => tables.tables_1d[*i].eval(args[0]),
                    2 => tables.tables_2d[*i].eval(args[0], args[1]),
                    3 => tables.tables_3d[*i].eval(args[0], args[1], args[2]),
                    _ => tables.tables[*i].eval(&args),
                }
            }
            Self::TableSum(i, args) => Self::sum_table(*i, args, state, registry, tables),
            Self::Table1D(i, x) => tables.tables_1d[*i].eval(x.eval(state, registry)),
            Self::Table2D(i, x, y) => {
                tables.tables_2d[*i].eval(x.eval(state, registry), y.eval(state, registry))
            }
            Self::Table1DSum(i, SetExpression::Reference(x)) => {
                tables.tables_1d[*i].sum(x.eval(state, registry, set_variables, set_tables).ones())
            }
            Self::Table1DSum(i, x) => tables.tables_1d[*i].sum(x.eval(state, registry).ones()),
            Self::Table1DVectorSum(i, VectorExpression::Reference(x)) => tables.tables_1d[*i].sum(
                x.eval(state, registry, vector_variables, vector_tables)
                    .iter()
                    .copied(),
            ),
            Self::Table1DVectorSum(i, x) => {
                tables.tables_1d[*i].sum(x.eval(state, registry).into_iter())
            }
            Self::Table2DSum(i, SetExpression::Reference(x), SetExpression::Reference(y)) => {
                let y = y.eval(state, registry, set_variables, set_tables);
                x.eval(state, registry, set_variables, set_tables)
                    .ones()
                    .map(|x| tables.tables_2d[*i].sum_y(x, y.ones()))
                    .sum()
            }
            Self::Table2DSum(i, SetExpression::Reference(x), y) => {
                let y = y.eval(state, registry);
                x.eval(state, registry, set_variables, set_tables)
                    .ones()
                    .map(|x| tables.tables_2d[*i].sum_y(x, y.ones()))
                    .sum()
            }
            Self::Table2DSum(i, x, SetExpression::Reference(y)) => {
                let y = y.eval(state, registry, set_variables, set_tables);
                x.eval(state, registry)
                    .ones()
                    .map(|x| tables.tables_2d[*i].sum_y(x, y.ones()))
                    .sum()
            }
            Self::Table2DSum(i, x, y) => {
                let y = y.eval(state, registry);
                x.eval(state, registry)
                    .ones()
                    .map(|x| tables.tables_2d[*i].sum_y(x, y.ones()))
                    .sum()
            }
            Self::Table2DVectorSum(
                i,
                VectorExpression::Reference(x),
                VectorExpression::Reference(y),
            ) => tables.tables_2d[*i].sum(
                x.eval(state, registry, vector_variables, vector_tables)
                    .iter()
                    .copied(),
                y.eval(state, registry, vector_variables, vector_tables)
                    .iter()
                    .copied(),
            ),
            Self::Table2DVectorSum(i, VectorExpression::Reference(x), y) => tables.tables_2d[*i]
                .sum(
                    x.eval(state, registry, vector_variables, vector_tables)
                        .iter()
                        .copied(),
                    y.eval(state, registry).into_iter(),
                ),
            Self::Table2DVectorSum(i, x, VectorExpression::Reference(y)) => tables.tables_2d[*i]
                .sum(
                    x.eval(state, registry).into_iter(),
                    y.eval(state, registry, vector_variables, vector_tables)
                        .iter()
                        .copied(),
                ),
            Self::Table2DVectorSum(i, x, y) => tables.tables_2d[*i].sum(
                x.eval(state, registry).into_iter(),
                y.eval(state, registry).into_iter(),
            ),
            Self::Table2DSetVectorSum(
                i,
                SetExpression::Reference(x),
                VectorExpression::Reference(y),
            ) => tables.tables_2d[*i].sum(
                x.eval(state, registry, set_variables, set_tables).ones(),
                y.eval(state, registry, vector_variables, vector_tables)
                    .iter()
                    .copied(),
            ),
            Self::Table2DSetVectorSum(i, SetExpression::Reference(x), y) => tables.tables_2d[*i]
                .sum(
                    x.eval(state, registry, set_variables, set_tables).ones(),
                    y.eval(state, registry).into_iter(),
                ),
            Self::Table2DSetVectorSum(i, x, VectorExpression::Reference(y)) => tables.tables_2d[*i]
                .sum(
                    x.eval(state, registry).ones(),
                    y.eval(state, registry, vector_variables, vector_tables)
                        .iter()
                        .copied(),
                ),
            Self::Table2DSetVectorSum(i, x, y) => tables.tables_2d[*i].sum(
                x.eval(state, registry).ones(),
                y.eval(state, registry).into_iter(),
            ),
            Self::Table2DVectorSetSum(
                i,
                VectorExpression::Reference(x),
                SetExpression::Reference(y),
            ) => {
                let x = x.eval(state, registry, vector_variables, vector_tables);
                y.eval(state, registry, set_variables, set_tables)
                    .ones()
                    .map(|y| tables.tables_2d[*i].sum_x(x.iter().copied(), y))
                    .sum()
            }
            Self::Table2DVectorSetSum(i, x, SetExpression::Reference(y)) => {
                let x = x.eval(state, registry);
                y.eval(state, registry, set_variables, set_tables)
                    .ones()
                    .map(|y| tables.tables_2d[*i].sum_x(x.iter().copied(), y))
                    .sum()
            }
            Self::Table2DVectorSetSum(i, VectorExpression::Reference(x), y) => {
                let x = x.eval(state, registry, vector_variables, vector_tables);
                y.eval(state, registry)
                    .ones()
                    .map(|y| tables.tables_2d[*i].sum_x(x.iter().copied(), y))
                    .sum()
            }
            Self::Table2DVectorSetSum(i, x, y) => {
                let x = x.eval(state, registry);
                y.eval(state, registry)
                    .ones()
                    .map(|y| tables.tables_2d[*i].sum_x(x.iter().copied(), y))
                    .sum()
            }
            Self::Table2DSumX(i, SetExpression::Reference(x), y) => tables.tables_2d[*i].sum_x(
                x.eval(state, registry, set_variables, set_tables).ones(),
                y.eval(state, registry),
            ),
            Self::Table2DSumX(i, x, y) => {
                tables.tables_2d[*i].sum_x(x.eval(state, registry).ones(), y.eval(state, registry))
            }
            Self::Table2DSumY(i, x, SetExpression::Reference(y)) => tables.tables_2d[*i].sum_y(
                x.eval(state, registry),
                y.eval(state, registry, set_variables, set_tables).ones(),
            ),
            Self::Table2DSumY(i, x, y) => {
                tables.tables_2d[*i].sum_y(x.eval(state, registry), y.eval(state, registry).ones())
            }
            Self::Table2DVectorSumX(i, VectorExpression::Reference(x), y) => tables.tables_2d[*i]
                .sum_x(
                    x.eval(state, registry, vector_variables, vector_tables)
                        .iter()
                        .copied(),
                    y.eval(state, registry),
                ),
            Self::Table2DVectorSumX(i, x, y) => tables.tables_2d[*i]
                .sum_x(x.eval(state, registry).into_iter(), y.eval(state, registry)),
            Self::Table2DVectorSumY(i, x, VectorExpression::Reference(y)) => tables.tables_2d[*i]
                .sum_y(
                    x.eval(state, registry),
                    y.eval(state, registry, vector_variables, vector_tables)
                        .iter()
                        .copied(),
                ),
            Self::Table2DVectorSumY(i, x, y) => tables.tables_2d[*i]
                .sum_y(x.eval(state, registry), y.eval(state, registry).into_iter()),
        }
    }

    pub fn simplify(
        &self,
        registry: &TableRegistry,
        tables: &TableData<T>,
    ) -> NumericTableExpression<T> {
        match self {
            Self::Table(i, args) => {
                let args: Vec<ElementExpression> =
                    args.iter().map(|x| x.simplify(registry)).collect();
                let mut simplified_args = Vec::with_capacity(args.len());
                for arg in &args {
                    match arg {
                        ElementExpression::Constant(arg) => {
                            simplified_args.push(*arg);
                        }
                        _ => return Self::Table(*i, args),
                    }
                }
                Self::Constant(match args.len() {
                    1 => tables.tables_1d[*i].eval(simplified_args[0]),
                    2 => tables.tables_2d[*i].eval(simplified_args[0], simplified_args[1]),
                    3 => tables.tables_3d[*i].eval(
                        simplified_args[0],
                        simplified_args[1],
                        simplified_args[2],
                    ),
                    _ => tables.tables[*i].eval(&simplified_args),
                })
            }
            Self::TableSum(i, args) => Self::simplify_sum_table(*i, args, registry, tables),
            Self::Table1D(i, x) => match x.simplify(registry) {
                ElementExpression::Constant(x) => Self::Constant(tables.tables_1d[*i].eval(x)),
                x => Self::Table1D(*i, x),
            },
            Self::Table2D(i, x, y) => match (x.simplify(registry), y.simplify(registry)) {
                (ElementExpression::Constant(x), ElementExpression::Constant(y)) => {
                    Self::Constant(tables.tables_2d[*i].eval(x, y))
                }
                (x, y) => Self::Table2D(*i, x, y),
            },
            Self::Table1DSum(i, x) => match x.simplify(registry) {
                SetExpression::Reference(ReferenceExpression::Constant(x)) => {
                    Self::Constant(tables.tables_1d[*i].sum(x.ones()))
                }
                x => Self::Table1DSum(*i, x),
            },
            Self::Table1DVectorSum(i, x) => match x.simplify(registry) {
                VectorExpression::Reference(ReferenceExpression::Constant(x)) => {
                    Self::Constant(tables.tables_1d[*i].sum(x.into_iter()))
                }
                x => Self::Table1DVectorSum(*i, x),
            },
            Self::Table2DSum(i, x, y) => match (x.simplify(registry), y.simplify(registry)) {
                (
                    SetExpression::Reference(ReferenceExpression::Constant(x)),
                    SetExpression::Reference(ReferenceExpression::Constant(y)),
                ) => Self::Constant(
                    x.ones()
                        .map(|x| tables.tables_2d[*i].sum_y(x, y.ones()))
                        .sum(),
                ),
                (x, y) => Self::Table2DSum(*i, x, y),
            },
            Self::Table2DVectorSum(i, x, y) => match (x.simplify(registry), y.simplify(registry)) {
                (
                    VectorExpression::Reference(ReferenceExpression::Constant(x)),
                    VectorExpression::Reference(ReferenceExpression::Constant(y)),
                ) => Self::Constant(tables.tables_2d[*i].sum(x.into_iter(), y.into_iter())),
                (x, y) => Self::Table2DVectorSum(*i, x, y),
            },
            Self::Table2DSetVectorSum(i, x, y) => {
                match (x.simplify(registry), y.simplify(registry)) {
                    (
                        SetExpression::Reference(ReferenceExpression::Constant(x)),
                        VectorExpression::Reference(ReferenceExpression::Constant(y)),
                    ) => Self::Constant(tables.tables_2d[*i].sum(x.ones(), y.into_iter())),
                    (x, y) => Self::Table2DSetVectorSum(*i, x, y),
                }
            }
            Self::Table2DVectorSetSum(i, x, y) => {
                match (x.simplify(registry), y.simplify(registry)) {
                    (
                        VectorExpression::Reference(ReferenceExpression::Constant(x)),
                        SetExpression::Reference(ReferenceExpression::Constant(y)),
                    ) => Self::Constant(
                        y.ones()
                            .map(|y| tables.tables_2d[*i].sum_x(x.iter().copied(), y))
                            .sum(),
                    ),
                    (x, y) => Self::Table2DVectorSetSum(*i, x, y),
                }
            }
            Self::Table2DSumX(i, x, y) => match (x.simplify(registry), y.simplify(registry)) {
                (
                    SetExpression::Reference(ReferenceExpression::Constant(x)),
                    ElementExpression::Constant(y),
                ) => Self::Constant(tables.tables_2d[*i].sum_x(x.ones(), y)),
                (x, y) => Self::Table2DSumX(*i, x, y),
            },
            Self::Table2DSumY(i, x, y) => match (x.simplify(registry), y.simplify(registry)) {
                (
                    ElementExpression::Constant(x),
                    SetExpression::Reference(ReferenceExpression::Constant(y)),
                ) => Self::Constant(tables.tables_2d[*i].sum_y(x, y.ones())),
                (x, y) => Self::Table2DSumY(*i, x, y),
            },
            Self::Table2DVectorSumX(i, x, y) => {
                match (x.simplify(registry), y.simplify(registry)) {
                    (
                        VectorExpression::Reference(ReferenceExpression::Constant(x)),
                        ElementExpression::Constant(y),
                    ) => Self::Constant(tables.tables_2d[*i].sum_x(x.into_iter(), y)),
                    (x, y) => Self::Table2DVectorSumX(*i, x, y),
                }
            }
            Self::Table2DVectorSumY(i, x, y) => {
                match (x.simplify(registry), y.simplify(registry)) {
                    (
                        ElementExpression::Constant(x),
                        VectorExpression::Reference(ReferenceExpression::Constant(y)),
                    ) => Self::Constant(tables.tables_2d[*i].sum_y(x, y.into_iter())),
                    (x, y) => Self::Table2DVectorSumY(*i, x, y),
                }
            }
            _ => self.clone(),
        }
    }

    fn sum_table(
        i: usize,
        args: &[ArgumentExpression],
        state: &State,
        registry: &TableRegistry,
        tables: &TableData<T>,
    ) -> T {
        let mut result = vec![vec![]];
        for v in args {
            match v {
                ArgumentExpression::Set(set) => {
                    result = match set {
                        SetExpression::Reference(set) => {
                            let set = set.eval(
                                state,
                                registry,
                                &state.signature_variables.set_variables,
                                &registry.set_tables,
                            );
                            util::expand_vector_with_set(result, set)
                        }
                        _ => util::expand_vector_with_set(result, &set.eval(state, registry)),
                    };
                }
                ArgumentExpression::Vector(vector) => {
                    result = match vector {
                        VectorExpression::Reference(vector) => {
                            let vector = vector.eval(
                                state,
                                registry,
                                &state.signature_variables.vector_variables,
                                &registry.vector_tables,
                            );
                            util::expand_vector_with_slice(result, vector)
                        }
                        _ => util::expand_vector_with_slice(result, &vector.eval(state, registry)),
                    };
                }
                ArgumentExpression::Element(element) => {
                    let element = element.eval(state, registry);
                    result.iter_mut().for_each(|r| r.push(element));
                }
            }
        }
        match args.len() {
            1 => result
                .into_iter()
                .map(|r| tables.tables_1d[i].eval(r[0]))
                .sum(),
            2 => result
                .into_iter()
                .map(|r| tables.tables_2d[i].eval(r[0], r[1]))
                .sum(),
            3 => result
                .into_iter()
                .map(|r| tables.tables_3d[i].eval(r[0], r[1], r[2]))
                .sum(),
            _ => result.into_iter().map(|r| tables.tables[i].eval(&r)).sum(),
        }
    }

    fn simplify_sum_table(
        i: usize,
        args: &[ArgumentExpression],
        registry: &TableRegistry,
        tables: &TableData<T>,
    ) -> NumericTableExpression<T> {
        let args: Vec<ArgumentExpression> = args.iter().map(|x| x.simplify(registry)).collect();
        let mut simplified_args = vec![vec![]];
        for arg in &args {
            match arg {
                ArgumentExpression::Element(ElementExpression::Constant(element)) => {
                    simplified_args.iter_mut().for_each(|r| r.push(*element));
                }
                ArgumentExpression::Set(SetExpression::Reference(
                    ReferenceExpression::Constant(set),
                )) => {
                    simplified_args = util::expand_vector_with_set(simplified_args, set);
                }
                ArgumentExpression::Vector(VectorExpression::Reference(
                    ReferenceExpression::Constant(vector),
                )) => {
                    simplified_args = util::expand_vector_with_slice(simplified_args, vector);
                }
                _ => return Self::TableSum(i, args),
            }
        }
        Self::Constant(match args.len() {
            1 => simplified_args
                .into_iter()
                .map(|r| tables.tables_1d[i].eval(r[0]))
                .sum(),
            2 => simplified_args
                .into_iter()
                .map(|r| tables.tables_2d[i].eval(r[0], r[1]))
                .sum(),
            3 => simplified_args
                .into_iter()
                .map(|r| tables.tables_3d[i].eval(r[0], r[1], r[2]))
                .sum(),
            _ => simplified_args
                .into_iter()
                .map(|r| tables.tables[i].eval(&r))
                .sum(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::*;
    use crate::table;
    use crate::variable::*;
    use rustc_hash::FxHashMap;
    use std::rc::Rc;

    fn generate_registry() -> TableRegistry {
        let mut name_to_constant = FxHashMap::default();
        name_to_constant.insert(String::from("f0"), 0);

        let tables_1d = vec![table::Table1D::new(vec![10, 20, 30])];
        let mut name_to_table_1d = FxHashMap::default();
        name_to_table_1d.insert(String::from("f1"), 0);

        let tables_2d = vec![table::Table2D::new(vec![
            vec![10, 20, 30],
            vec![40, 50, 60],
            vec![70, 80, 90],
        ])];
        let mut name_to_table_2d = FxHashMap::default();
        name_to_table_2d.insert(String::from("f2"), 0);

        let tables_3d = vec![table::Table3D::new(vec![
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
        ])];
        let mut name_to_table_3d = FxHashMap::default();
        name_to_table_3d.insert(String::from("f3"), 0);

        let mut map = FxHashMap::default();
        let key = vec![0, 1, 0, 0];
        map.insert(key, 100);
        let key = vec![0, 1, 0, 1];
        map.insert(key, 200);
        let key = vec![0, 1, 2, 0];
        map.insert(key, 300);
        let key = vec![0, 1, 2, 1];
        map.insert(key, 400);
        let tables = vec![table::Table::new(map, 0)];
        let mut name_to_table = FxHashMap::default();
        name_to_table.insert(String::from("f4"), 0);

        TableRegistry {
            integer_tables: TableData {
                name_to_constant,
                tables_1d,
                name_to_table_1d,
                tables_2d,
                name_to_table_2d,
                tables_3d,
                name_to_table_3d,
                tables,
                name_to_table,
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
                set_variables: vec![set1, set2, Set::with_capacity(3), Set::with_capacity(3)],
                vector_variables: vec![vec![0, 2], vec![0, 1], vec![], vec![]],
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[test]
    fn constant_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let expression = NumericTableExpression::Constant(10);
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            10
        );
    }

    #[test]
    fn table_1d_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let expression = NumericTableExpression::Table1D(0, ElementExpression::Constant(0));
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            10
        );
        let expression = NumericTableExpression::Table1D(0, ElementExpression::Constant(1));
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            20
        );
        let expression = NumericTableExpression::Table1D(0, ElementExpression::Constant(2));
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            30
        );
    }

    #[test]
    fn table_1d_sum_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let expression = NumericTableExpression::Table1DSum(
            0,
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            40
        );
        let expression = NumericTableExpression::Table1DSum(
            0,
            SetExpression::Reference(ReferenceExpression::Variable(1)),
        );
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            30
        );
        let expression = NumericTableExpression::Table1DSum(
            0,
            SetExpression::Complement(Box::new(SetExpression::Reference(
                ReferenceExpression::Variable(0),
            ))),
        );
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            20
        );
    }

    #[test]
    fn table_1d_vector_sum_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let expression = NumericTableExpression::Table1DVectorSum(
            0,
            VectorExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            40
        );
        let expression = NumericTableExpression::Table1DVectorSum(
            0,
            VectorExpression::Reverse(Box::new(VectorExpression::Reference(
                ReferenceExpression::Variable(0),
            ))),
        );
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            40
        );
    }

    #[test]
    fn table_2d_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let expression = NumericTableExpression::Table2D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(1),
        );
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            20
        );
    }

    #[test]
    fn table_2d_sum_eval() {
        let registry = generate_registry();
        let state = generate_state();

        let expression = NumericTableExpression::Table2DSum(
            0,
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Reference(ReferenceExpression::Variable(1)),
        );
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            180
        );

        let expression = NumericTableExpression::Table2DSum(
            0,
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(1)),
            )))),
        );
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            180
        );

        let expression = NumericTableExpression::Table2DSum(
            0,
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
            SetExpression::Reference(ReferenceExpression::Variable(1)),
        );
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            180
        );

        let expression = NumericTableExpression::Table2DSum(
            0,
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(1)),
            )))),
        );
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            180
        );
    }

    #[test]
    fn table_2d_vector_sum_eval() {
        let registry = generate_registry();
        let state = generate_state();

        let expression = NumericTableExpression::Table2DVectorSum(
            0,
            VectorExpression::Reference(ReferenceExpression::Variable(0)),
            VectorExpression::Reference(ReferenceExpression::Variable(1)),
        );
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            180
        );

        let expression = NumericTableExpression::Table2DVectorSum(
            0,
            VectorExpression::Reference(ReferenceExpression::Variable(0)),
            VectorExpression::Reverse(Box::new(VectorExpression::Reference(
                ReferenceExpression::Variable(1),
            ))),
        );
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            180
        );

        let expression = NumericTableExpression::Table2DVectorSum(
            0,
            VectorExpression::Reverse(Box::new(VectorExpression::Reference(
                ReferenceExpression::Variable(0),
            ))),
            VectorExpression::Reference(ReferenceExpression::Variable(1)),
        );
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            180
        );

        let expression = NumericTableExpression::Table2DVectorSum(
            0,
            VectorExpression::Reverse(Box::new(VectorExpression::Reference(
                ReferenceExpression::Variable(0),
            ))),
            VectorExpression::Reverse(Box::new(VectorExpression::Reference(
                ReferenceExpression::Variable(1),
            ))),
        );
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            180
        );
    }

    #[test]
    fn table_2d_set_vector_sum_eval() {
        let registry = generate_registry();
        let state = generate_state();

        let expression = NumericTableExpression::Table2DSetVectorSum(
            0,
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            VectorExpression::Reference(ReferenceExpression::Variable(1)),
        );
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            180
        );

        let expression = NumericTableExpression::Table2DSetVectorSum(
            0,
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            VectorExpression::Reverse(Box::new(VectorExpression::Reference(
                ReferenceExpression::Variable(1),
            ))),
        );
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            180
        );

        let expression = NumericTableExpression::Table2DSetVectorSum(
            0,
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
            VectorExpression::Reference(ReferenceExpression::Variable(1)),
        );
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            180
        );

        let expression = NumericTableExpression::Table2DSetVectorSum(
            0,
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
            VectorExpression::Reverse(Box::new(VectorExpression::Reference(
                ReferenceExpression::Variable(1),
            ))),
        );
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            180
        );
    }

    #[test]
    fn table_2d_vector_set_sum_eval() {
        let registry = generate_registry();
        let state = generate_state();

        let expression = NumericTableExpression::Table2DVectorSetSum(
            0,
            VectorExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Reference(ReferenceExpression::Variable(1)),
        );
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            180
        );

        let expression = NumericTableExpression::Table2DVectorSetSum(
            0,
            VectorExpression::Reverse(Box::new(VectorExpression::Reference(
                ReferenceExpression::Variable(0),
            ))),
            SetExpression::Reference(ReferenceExpression::Variable(1)),
        );
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            180
        );

        let expression = NumericTableExpression::Table2DVectorSetSum(
            0,
            VectorExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(1)),
            )))),
        );
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            180
        );

        let expression = NumericTableExpression::Table2DVectorSetSum(
            0,
            VectorExpression::Reverse(Box::new(VectorExpression::Reference(
                ReferenceExpression::Variable(0),
            ))),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(1)),
            )))),
        );
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            180
        );
    }

    #[test]
    fn table_2d_sum_x_eval() {
        let registry = generate_registry();
        let state = generate_state();

        let expression = NumericTableExpression::Table2DSumX(
            0,
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            ElementExpression::Constant(0),
        );
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            80
        );

        let expression = NumericTableExpression::Table2DSumX(
            0,
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
            ElementExpression::Constant(0),
        );
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            80
        );
    }

    #[test]
    fn table_2d_vector_sum_x_eval() {
        let registry = generate_registry();
        let state = generate_state();

        let expression = NumericTableExpression::Table2DVectorSumX(
            0,
            VectorExpression::Reference(ReferenceExpression::Variable(0)),
            ElementExpression::Constant(0),
        );
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            80
        );

        let expression = NumericTableExpression::Table2DVectorSumX(
            0,
            VectorExpression::Reverse(Box::new(VectorExpression::Reference(
                ReferenceExpression::Variable(0),
            ))),
            ElementExpression::Constant(0),
        );
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            80
        );
    }

    #[test]
    fn table_2d_sum_y_eval() {
        let registry = generate_registry();
        let state = generate_state();

        let expression = NumericTableExpression::Table2DSumY(
            0,
            ElementExpression::Constant(0),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            40
        );

        let expression = NumericTableExpression::Table2DSumY(
            0,
            ElementExpression::Constant(0),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
        );
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            40
        );
    }

    #[test]
    fn table_2d_vector_sum_y_eval() {
        let registry = generate_registry();
        let state = generate_state();

        let expression = NumericTableExpression::Table2DVectorSumY(
            0,
            ElementExpression::Constant(0),
            VectorExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            40
        );

        let expression = NumericTableExpression::Table2DVectorSumY(
            0,
            ElementExpression::Constant(0),
            VectorExpression::Reverse(Box::new(VectorExpression::Reference(
                ReferenceExpression::Variable(0),
            ))),
        );
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            40
        );
    }

    #[test]
    fn table_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let expression = NumericTableExpression::Table(
            0,
            vec![
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
            ],
        );
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            10
        );
        let expression = NumericTableExpression::Table(
            0,
            vec![
                ElementExpression::Constant(0),
                ElementExpression::Constant(1),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
            ],
        );
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            100
        );
        let expression = NumericTableExpression::Table(
            0,
            vec![
                ElementExpression::Constant(0),
                ElementExpression::Constant(1),
                ElementExpression::Constant(0),
                ElementExpression::Constant(1),
            ],
        );
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            200
        );
        let expression = NumericTableExpression::Table(
            0,
            vec![
                ElementExpression::Constant(0),
                ElementExpression::Constant(1),
                ElementExpression::Constant(2),
                ElementExpression::Constant(0),
            ],
        );
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            300
        );
        let expression = NumericTableExpression::Table(
            0,
            vec![
                ElementExpression::Constant(0),
                ElementExpression::Constant(1),
                ElementExpression::Constant(2),
                ElementExpression::Constant(1),
            ],
        );
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            400
        );
    }

    #[test]
    fn table_sum_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let expression = NumericTableExpression::TableSum(
            0,
            vec![
                ArgumentExpression::Element(ElementExpression::Constant(0)),
                ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Variable(0))),
                ArgumentExpression::Vector(VectorExpression::Reference(
                    ReferenceExpression::Variable(1),
                )),
            ],
        );
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            180
        );
        let expression = NumericTableExpression::TableSum(
            0,
            vec![
                ArgumentExpression::Element(ElementExpression::Constant(0)),
                ArgumentExpression::Element(ElementExpression::Constant(1)),
                ArgumentExpression::Set(SetExpression::Complement(Box::new(
                    SetExpression::Complement(Box::new(SetExpression::Reference(
                        ReferenceExpression::Variable(0),
                    ))),
                ))),
                ArgumentExpression::Vector(VectorExpression::Reverse(Box::new(
                    VectorExpression::Reference(ReferenceExpression::Variable(1)),
                ))),
            ],
        );
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            1000
        );
    }

    #[test]
    fn constant_simplify() {
        let registry = generate_registry();
        let expression = NumericTableExpression::Constant(0);
        assert_eq!(
            expression.simplify(&registry, &registry.integer_tables),
            expression
        );
    }

    #[test]
    fn table_1d_simplify() {
        let registry = generate_registry();

        let expression = NumericTableExpression::Table1D(0, ElementExpression::Constant(0));
        assert_eq!(
            expression.simplify(&registry, &registry.integer_tables),
            NumericTableExpression::Constant(10)
        );

        let expression = NumericTableExpression::Table1D(0, ElementExpression::Variable(0));
        assert_eq!(
            expression.simplify(&registry, &registry.integer_tables),
            expression
        );
    }

    #[test]
    fn table_1d_sum_simplify() {
        let registry = generate_registry();

        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(1);
        let expression = NumericTableExpression::Table1DSum(
            0,
            SetExpression::Reference(ReferenceExpression::Constant(set)),
        );
        assert_eq!(
            expression.simplify(&registry, &registry.integer_tables),
            NumericTableExpression::Constant(30)
        );

        let expression = NumericTableExpression::Table1DSum(
            0,
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert_eq!(
            expression.simplify(&registry, &registry.integer_tables),
            expression
        );
    }

    #[test]
    fn table_1d_vector_sum_simplify() {
        let registry = generate_registry();

        let expression = NumericTableExpression::Table1DVectorSum(
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert_eq!(
            expression.simplify(&registry, &registry.integer_tables),
            NumericTableExpression::Constant(30)
        );

        let expression = NumericTableExpression::Table1DVectorSum(
            0,
            VectorExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert_eq!(
            expression.simplify(&registry, &registry.integer_tables),
            expression
        );
    }

    #[test]
    fn table_2d_simplify() {
        let registry = generate_registry();

        let expression = NumericTableExpression::Table2D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
        );
        assert_eq!(
            expression.simplify(&registry, &registry.integer_tables),
            NumericTableExpression::Constant(10)
        );

        let expression = NumericTableExpression::Table2D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Variable(0),
        );
        assert_eq!(
            expression.simplify(&registry, &registry.integer_tables),
            expression
        );
    }

    #[test]
    fn table_2d_sum_simplify() {
        let registry = generate_registry();

        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(1);
        let expression = NumericTableExpression::Table2DSum(
            0,
            SetExpression::Reference(ReferenceExpression::Constant(set.clone())),
            SetExpression::Reference(ReferenceExpression::Constant(set)),
        );
        assert_eq!(
            expression.simplify(&registry, &registry.integer_tables),
            NumericTableExpression::Constant(120)
        );

        let expression = NumericTableExpression::Table2DSum(
            0,
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Reference(ReferenceExpression::Variable(1)),
        );
        assert_eq!(
            expression.simplify(&registry, &registry.integer_tables),
            expression
        );
    }

    #[test]
    fn table_2d_vector_sum_simplify() {
        let registry = generate_registry();

        let expression = NumericTableExpression::Table2DVectorSum(
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert_eq!(
            expression.simplify(&registry, &registry.integer_tables),
            NumericTableExpression::Constant(120)
        );

        let expression = NumericTableExpression::Table2DVectorSum(
            0,
            VectorExpression::Reference(ReferenceExpression::Variable(0)),
            VectorExpression::Reference(ReferenceExpression::Variable(1)),
        );
        assert_eq!(
            expression.simplify(&registry, &registry.integer_tables),
            expression
        );
    }

    #[test]
    fn table_2d_set_vector_sum_simplify() {
        let registry = generate_registry();

        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(1);
        let expression = NumericTableExpression::Table2DSetVectorSum(
            0,
            SetExpression::Reference(ReferenceExpression::Constant(set)),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert_eq!(
            expression.simplify(&registry, &registry.integer_tables),
            NumericTableExpression::Constant(120)
        );

        let expression = NumericTableExpression::Table2DSetVectorSum(
            0,
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            VectorExpression::Reference(ReferenceExpression::Variable(1)),
        );
        assert_eq!(
            expression.simplify(&registry, &registry.integer_tables),
            expression
        );
    }

    #[test]
    fn table_2d_vector_set_sum_simplify() {
        let registry = generate_registry();

        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(1);
        let expression = NumericTableExpression::Table2DVectorSetSum(
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            SetExpression::Reference(ReferenceExpression::Constant(set)),
        );
        assert_eq!(
            expression.simplify(&registry, &registry.integer_tables),
            NumericTableExpression::Constant(120)
        );

        let expression = NumericTableExpression::Table2DVectorSetSum(
            0,
            VectorExpression::Reference(ReferenceExpression::Variable(1)),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert_eq!(
            expression.simplify(&registry, &registry.integer_tables),
            expression
        );
    }

    #[test]
    fn table_2d_sum_x_simplify() {
        let registry = generate_registry();

        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(1);
        let expression = NumericTableExpression::Table2DSumX(
            0,
            SetExpression::Reference(ReferenceExpression::Constant(set)),
            ElementExpression::Constant(0),
        );
        assert_eq!(
            expression.simplify(&registry, &registry.integer_tables),
            NumericTableExpression::Constant(50)
        );

        let expression = NumericTableExpression::Table2DSumX(
            0,
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            ElementExpression::Constant(0),
        );
        assert_eq!(
            expression.simplify(&registry, &registry.integer_tables),
            expression
        );
    }

    #[test]
    fn table_2d_sum_y_simplify() {
        let registry = generate_registry();

        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(1);
        let expression = NumericTableExpression::Table2DSumY(
            0,
            ElementExpression::Constant(0),
            SetExpression::Reference(ReferenceExpression::Constant(set)),
        );
        assert_eq!(
            expression.simplify(&registry, &registry.integer_tables),
            NumericTableExpression::Constant(30)
        );

        let expression = NumericTableExpression::Table2DSumY(
            0,
            ElementExpression::Constant(0),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert_eq!(
            expression.simplify(&registry, &registry.integer_tables),
            expression
        );
    }

    #[test]
    fn table_2d_vector_sum_x_simplify() {
        let registry = generate_registry();

        let expression = NumericTableExpression::Table2DVectorSumX(
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            ElementExpression::Constant(0),
        );
        assert_eq!(
            expression.simplify(&registry, &registry.integer_tables),
            NumericTableExpression::Constant(50)
        );

        let expression = NumericTableExpression::Table2DVectorSumX(
            0,
            VectorExpression::Reference(ReferenceExpression::Variable(0)),
            ElementExpression::Constant(0),
        );
        assert_eq!(
            expression.simplify(&registry, &registry.integer_tables),
            expression
        );
    }

    #[test]
    fn table_2d_vector_sum_y_simplify() {
        let registry = generate_registry();

        let expression = NumericTableExpression::Table2DVectorSumY(
            0,
            ElementExpression::Constant(0),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert_eq!(
            expression.simplify(&registry, &registry.integer_tables),
            NumericTableExpression::Constant(30)
        );

        let expression = NumericTableExpression::Table2DVectorSumY(
            0,
            ElementExpression::Constant(0),
            VectorExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert_eq!(
            expression.simplify(&registry, &registry.integer_tables),
            expression
        );
    }

    #[test]
    fn table_simplify() {
        let registry = generate_registry();

        let expression = NumericTableExpression::Table(
            0,
            vec![
                ElementExpression::Constant(0),
                ElementExpression::Constant(1),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
            ],
        );
        assert_eq!(
            expression.simplify(&registry, &registry.integer_tables),
            NumericTableExpression::Constant(100)
        );

        let expression = NumericTableExpression::Table(
            0,
            vec![
                ElementExpression::Constant(0),
                ElementExpression::Constant(1),
                ElementExpression::Constant(0),
                ElementExpression::Variable(0),
            ],
        );
        assert_eq!(
            expression.simplify(&registry, &registry.integer_tables),
            expression
        );
    }

    #[test]
    fn table_sum_simplify() {
        let registry = generate_registry();

        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(1);
        let expression = NumericTableExpression::TableSum(
            0,
            vec![
                ArgumentExpression::Vector(VectorExpression::Reference(
                    ReferenceExpression::Constant(vec![0, 1]),
                )),
                ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Constant(
                    set,
                ))),
                ArgumentExpression::Element(ElementExpression::Constant(0)),
            ],
        );
        assert_eq!(
            expression.simplify(&registry, &registry.integer_tables),
            NumericTableExpression::Constant(100)
        );

        let expression = NumericTableExpression::TableSum(
            0,
            vec![
                ArgumentExpression::Element(ElementExpression::Constant(0)),
                ArgumentExpression::Element(ElementExpression::Constant(1)),
                ArgumentExpression::Element(ElementExpression::Constant(0)),
                ArgumentExpression::Vector(VectorExpression::Reference(
                    ReferenceExpression::Variable(0),
                )),
            ],
        );
        assert_eq!(
            expression.simplify(&registry, &registry.integer_tables),
            expression
        );
    }
}
