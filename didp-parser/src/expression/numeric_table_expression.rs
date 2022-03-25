use super::element_expression::{ElementExpression, VectorExpression};
use super::reference_expression::ReferenceExpression;
use super::set_expression::SetExpression;
use super::util;
use crate::state::State;
use crate::table;
use crate::table_data::TableData;
use crate::table_registry::TableRegistry;
use crate::variable::{Element, Numeric};

#[derive(Debug, PartialEq, Eq, Clone)]
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

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum NumericTableExpression<T: Numeric> {
    Constant(T),
    Table(usize, Vec<ElementExpression>),
    TableSum(usize, Vec<ArgumentExpression>),
    TableZipSum(usize, Vec<ArgumentExpression>),
    Table1D(usize, ElementExpression),
    Table2D(usize, ElementExpression, ElementExpression),
    Table3D(
        usize,
        ElementExpression,
        ElementExpression,
        ElementExpression,
    ),
    Table1DSum(usize, SetExpression),
    Table1DVectorSum(usize, VectorExpression),
    Table2DSum(usize, SetExpression, SetExpression),
    Table2DZipSum(usize, VectorExpression, VectorExpression),
    Table2DSumX(usize, SetExpression, ElementExpression),
    Table2DSumY(usize, ElementExpression, SetExpression),
    Table2DVectorSumX(usize, VectorExpression, ElementExpression),
    Table2DVectorSumY(usize, ElementExpression, VectorExpression),
    Table3DSum(usize, SetExpression, SetExpression, SetExpression),
    Table3DZipSum(usize, VectorExpression, VectorExpression, VectorExpression),
    Table3DSumX(usize, SetExpression, ElementExpression, ElementExpression),
    Table3DSumY(usize, ElementExpression, SetExpression, ElementExpression),
    Table3DSumZ(usize, ElementExpression, ElementExpression, SetExpression),
    Table3DVectorSumX(
        usize,
        VectorExpression,
        ElementExpression,
        ElementExpression,
    ),
    Table3DVectorSumY(
        usize,
        ElementExpression,
        VectorExpression,
        ElementExpression,
    ),
    Table3DVectorSumZ(
        usize,
        ElementExpression,
        ElementExpression,
        VectorExpression,
    ),
    Table3DSumXY(usize, SetExpression, SetExpression, ElementExpression),
    Table3DSumXZ(usize, SetExpression, ElementExpression, SetExpression),
    Table3DSumYZ(usize, ElementExpression, SetExpression, SetExpression),
    Table3DZipSumXY(usize, VectorExpression, VectorExpression, ElementExpression),
    Table3DZipSumXZ(usize, VectorExpression, ElementExpression, VectorExpression),
    Table3DZipSumYZ(usize, ElementExpression, VectorExpression, VectorExpression),
}

impl<T: Numeric> NumericTableExpression<T> {
    pub fn eval(&self, state: &State, registry: &TableRegistry, tables: &TableData<T>) -> T {
        match self {
            Self::Constant(value) => *value,
            Self::Table(i, args) => {
                let args: Vec<Element> = args.iter().map(|x| x.eval(state, registry)).collect();
                tables.tables[*i].eval(&args)
            }
            Self::TableSum(i, args) => Self::sum_table(&tables.tables[*i], args, state, registry),
            Self::TableZipSum(i, args) => {
                Self::zip_sum_table(&tables.tables[*i], args, state, registry)
            }
            Self::Table1D(i, x) => tables.tables_1d[*i].eval(x.eval(state, registry)),
            Self::Table2D(i, x, y) => {
                tables.tables_2d[*i].eval(x.eval(state, registry), y.eval(state, registry))
            }
            Self::Table3D(i, x, y, z) => tables.tables_3d[*i].eval(
                x.eval(state, registry),
                y.eval(state, registry),
                z.eval(state, registry),
            ),
            Self::Table1DSum(i, SetExpression::Reference(x)) => x
                .eval(
                    state,
                    registry,
                    &state.signature_variables.set_variables,
                    &registry.set_tables,
                )
                .ones()
                .map(|x| tables.tables_1d[*i].eval(x))
                .sum(),
            Self::Table1DSum(i, x) => x
                .eval(state, registry)
                .ones()
                .map(|x| tables.tables_1d[*i].eval(x))
                .sum(),
            Self::Table1DVectorSum(i, VectorExpression::Reference(x)) => x
                .eval(
                    state,
                    registry,
                    &state.signature_variables.vector_variables,
                    &registry.vector_tables,
                )
                .iter()
                .map(|x| tables.tables_1d[*i].eval(*x))
                .sum(),
            Self::Table1DVectorSum(i, x) => x
                .eval(state, registry)
                .into_iter()
                .map(|x| tables.tables_1d[*i].eval(x))
                .sum(),
            Self::Table2DSum(i, x, y) => {
                let x = x.eval(state, registry);
                let y = y.eval(state, registry);
                x.ones()
                    .map(|x| y.ones().map(|y| tables.tables_2d[*i].eval(x, y)).sum())
                    .sum()
            }
            Self::Table2DZipSum(i, x, y) => x
                .eval(state, registry)
                .into_iter()
                .zip(y.eval(state, registry).into_iter())
                .map(|(x, y)| tables.tables_2d[*i].eval(x, y))
                .sum(),
            Self::Table2DSumX(i, x, y) => {
                let y = y.eval(state, registry);
                x.eval(state, registry)
                    .ones()
                    .map(|x| tables.tables_2d[*i].eval(x, y))
                    .sum()
            }
            Self::Table2DSumY(i, x, y) => {
                let x = x.eval(state, registry);
                y.eval(state, registry)
                    .ones()
                    .map(|y| tables.tables_2d[*i].eval(x, y))
                    .sum()
            }
            Self::Table2DVectorSumX(i, x, y) => {
                let y = y.eval(state, registry);
                x.eval(state, registry)
                    .into_iter()
                    .map(|x| tables.tables_2d[*i].eval(x, y))
                    .sum()
            }
            Self::Table2DVectorSumY(i, x, y) => {
                let x = x.eval(state, registry);
                y.eval(state, registry)
                    .into_iter()
                    .map(|y| tables.tables_2d[*i].eval(x, y))
                    .sum()
            }
            Self::Table3DSum(i, x, y, z) => {
                let y = y.eval(state, registry);
                let z = z.eval(state, registry);
                x.eval(state, registry)
                    .ones()
                    .map(|x| {
                        y.ones()
                            .map(|y| z.ones().map(|z| tables.tables_3d[*i].eval(x, y, z)).sum())
                            .sum()
                    })
                    .sum()
            }
            Self::Table3DZipSum(i, x, y, z) => x
                .eval(state, registry)
                .into_iter()
                .zip(y.eval(state, registry).into_iter())
                .zip(z.eval(state, registry).into_iter())
                .map(|((x, y), z)| tables.tables_3d[*i].eval(x, y, z))
                .sum(),
            Self::Table3DSumX(i, x, y, z) => {
                let y = y.eval(state, registry);
                let z = z.eval(state, registry);
                x.eval(state, registry)
                    .ones()
                    .map(|x| tables.tables_3d[*i].eval(x, y, z))
                    .sum()
            }
            Self::Table3DSumY(i, x, y, z) => {
                let x = x.eval(state, registry);
                let z = z.eval(state, registry);
                y.eval(state, registry)
                    .ones()
                    .map(|y| tables.tables_3d[*i].eval(x, y, z))
                    .sum()
            }
            Self::Table3DSumZ(i, x, y, z) => {
                let x = x.eval(state, registry);
                let y = y.eval(state, registry);
                z.eval(state, registry)
                    .ones()
                    .map(|z| tables.tables_3d[*i].eval(x, y, z))
                    .sum()
            }
            Self::Table3DVectorSumX(i, x, y, z) => {
                let y = y.eval(state, registry);
                let z = z.eval(state, registry);
                x.eval(state, registry)
                    .into_iter()
                    .map(|x| tables.tables_3d[*i].eval(x, y, z))
                    .sum()
            }
            Self::Table3DVectorSumY(i, x, y, z) => {
                let x = x.eval(state, registry);
                let z = z.eval(state, registry);
                y.eval(state, registry)
                    .into_iter()
                    .map(|y| tables.tables_3d[*i].eval(x, y, z))
                    .sum()
            }
            Self::Table3DVectorSumZ(i, x, y, z) => {
                let x = x.eval(state, registry);
                let y = y.eval(state, registry);
                z.eval(state, registry)
                    .into_iter()
                    .map(|z| tables.tables_3d[*i].eval(x, y, z))
                    .sum()
            }
            Self::Table3DSumXY(i, x, y, z) => {
                let y = y.eval(state, registry);
                let z = z.eval(state, registry);
                x.eval(state, registry)
                    .ones()
                    .map(|x| y.ones().map(|y| tables.tables_3d[*i].eval(x, y, z)).sum())
                    .sum()
            }
            Self::Table3DSumXZ(i, x, y, z) => {
                let y = y.eval(state, registry);
                let z = z.eval(state, registry);
                x.eval(state, registry)
                    .ones()
                    .map(|x| z.ones().map(|z| tables.tables_3d[*i].eval(x, y, z)).sum())
                    .sum()
            }
            Self::Table3DSumYZ(i, x, y, z) => {
                let x = x.eval(state, registry);
                let z = z.eval(state, registry);
                y.eval(state, registry)
                    .ones()
                    .map(|y| z.ones().map(|z| tables.tables_3d[*i].eval(x, y, z)).sum())
                    .sum()
            }
            Self::Table3DZipSumXY(i, x, y, z) => {
                let z = z.eval(state, registry);
                x.eval(state, registry)
                    .into_iter()
                    .zip(y.eval(state, registry).into_iter())
                    .map(|(x, y)| tables.tables_3d[*i].eval(x, y, z))
                    .sum()
            }
            Self::Table3DZipSumXZ(i, x, y, z) => {
                let y = y.eval(state, registry);
                x.eval(state, registry)
                    .into_iter()
                    .zip(z.eval(state, registry).into_iter())
                    .map(|(x, z)| tables.tables_3d[*i].eval(x, y, z))
                    .sum()
            }
            Self::Table3DZipSumYZ(i, x, y, z) => {
                let x = x.eval(state, registry);
                y.eval(state, registry)
                    .into_iter()
                    .zip(z.eval(state, registry).into_iter())
                    .map(|(y, z)| tables.tables_3d[*i].eval(x, y, z))
                    .sum()
            }
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
                Self::Constant(tables.tables[*i].eval(&simplified_args))
            }
            Self::Table1D(i, ElementExpression::Constant(x)) => {
                Self::Constant(tables.tables_1d[*i].eval(*x))
            }
            Self::Table2D(i, ElementExpression::Constant(x), ElementExpression::Constant(y)) => {
                Self::Constant(tables.tables_2d[*i].eval(*x, *y))
            }
            Self::Table3D(
                i,
                ElementExpression::Constant(x),
                ElementExpression::Constant(y),
                ElementExpression::Constant(z),
            ) => Self::Constant(tables.tables_3d[*i].eval(*x, *y, *z)),
            Self::Table1DSum(i, SetExpression::Reference(ReferenceExpression::Constant(x))) => {
                Self::Constant(x.ones().map(|x| tables.tables_1d[*i].eval(x)).sum())
            }
            Self::Table1DVectorSum(
                i,
                VectorExpression::Reference(ReferenceExpression::Constant(x)),
            ) => Self::Constant(x.iter().map(|x| tables.tables_1d[*i].eval(*x)).sum()),
            Self::Table2DSum(
                i,
                SetExpression::Reference(ReferenceExpression::Constant(x)),
                SetExpression::Reference(ReferenceExpression::Constant(y)),
            ) => Self::Constant(
                x.ones()
                    .map(|x| y.ones().map(|y| tables.tables_2d[*i].eval(x, y)).sum())
                    .sum(),
            ),
            Self::Table2DZipSum(
                i,
                VectorExpression::Reference(ReferenceExpression::Constant(x)),
                VectorExpression::Reference(ReferenceExpression::Constant(y)),
            ) => Self::Constant(
                x.iter()
                    .zip(y.iter())
                    .map(|(x, y)| tables.tables_2d[*i].eval(*x, *y))
                    .sum(),
            ),
            Self::Table3DSum(
                i,
                SetExpression::Reference(ReferenceExpression::Constant(x)),
                SetExpression::Reference(ReferenceExpression::Constant(y)),
                SetExpression::Reference(ReferenceExpression::Constant(z)),
            ) => Self::Constant(
                x.ones()
                    .map(|x| {
                        y.ones()
                            .map(|y| z.ones().map(|z| tables.tables_3d[*i].eval(x, y, z)).sum())
                            .sum()
                    })
                    .sum(),
            ),
            Self::Table3DZipSum(
                i,
                VectorExpression::Reference(ReferenceExpression::Constant(x)),
                VectorExpression::Reference(ReferenceExpression::Constant(y)),
                VectorExpression::Reference(ReferenceExpression::Constant(z)),
            ) => Self::Constant(
                x.iter()
                    .zip(y.iter())
                    .zip(z.iter())
                    .map(|((x, y), z)| tables.tables_3d[*i].eval(*x, *y, *z))
                    .sum(),
            ),
            _ => self.clone(),
        }
    }

    fn sum_table(
        f: &table::Table<T>,
        args: &[ArgumentExpression],
        state: &State,
        registry: &TableRegistry,
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
        result.into_iter().map(|x| f.eval(&x)).sum()
    }

    fn zip_sum_table(
        f: &table::Table<T>,
        args: &[ArgumentExpression],
        state: &State,
        registry: &TableRegistry,
    ) -> T {
        let mut result = vec![vec![vec![]]];
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
                            result
                                .into_iter()
                                .map(|rr| util::expand_vector_with_set(rr, set))
                                .collect::<Vec<Vec<Vec<Element>>>>()
                        }
                        _ => {
                            let set = &set.eval(state, registry);
                            result
                                .into_iter()
                                .map(|rr| util::expand_vector_with_set(rr, &set))
                                .collect::<Vec<Vec<Vec<Element>>>>()
                        }
                    };
                }
                ArgumentExpression::Vector(vector) => {
                    match vector {
                        VectorExpression::Reference(vector) => {
                            let vector = vector.eval(
                                state,
                                registry,
                                &state.signature_variables.vector_variables,
                                &registry.vector_tables,
                            );
                            result
                                .iter_mut()
                                .zip(vector.iter())
                                .for_each(|(rr, e)| rr.iter_mut().for_each(|r| r.push(*e)));
                        }
                        _ => {
                            let vector = vector.eval(state, registry);
                            result
                                .iter_mut()
                                .zip(vector.into_iter())
                                .for_each(|(rr, e)| rr.iter_mut().for_each(|r| r.push(e)));
                        }
                    };
                }
                ArgumentExpression::Element(element) => {
                    let element = element.eval(state, registry);
                    result
                        .iter_mut()
                        .for_each(|rr| rr.iter_mut().for_each(|r| r.push(element)));
                }
            }
        }
        result.into_iter().flatten().map(|x| f.eval(&x)).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::*;
    use crate::variable::*;
    use std::collections::HashMap;
    use std::rc::Rc;

    fn generate_registry() -> TableRegistry {
        let mut name_to_constant = HashMap::new();
        name_to_constant.insert(String::from("f0"), 0);

        let tables_1d = vec![table::Table1D::new(vec![10, 20, 30])];
        let mut name_to_table_1d = HashMap::new();
        name_to_table_1d.insert(String::from("f1"), 0);

        let tables_2d = vec![table::Table2D::new(vec![
            vec![10, 20, 30],
            vec![40, 50, 60],
            vec![70, 80, 90],
        ])];
        let mut name_to_table_2d = HashMap::new();
        name_to_table_2d.insert(String::from("f2"), 0);

        let tables_3d = vec![table::Table3D::new(vec![
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
        ])];
        let mut name_to_table_3d = HashMap::new();
        name_to_table_3d.insert(String::from("f3"), 0);

        let mut map = HashMap::new();
        let key = vec![0, 1, 0, 0];
        map.insert(key, 100);
        let key = vec![0, 1, 0, 1];
        map.insert(key, 200);
        let key = vec![0, 1, 2, 0];
        map.insert(key, 300);
        let key = vec![0, 1, 2, 1];
        map.insert(key, 400);
        let tables = vec![table::Table::new(map, 0)];
        let mut name_to_table = HashMap::new();
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
    fn table_2d_zip_sum_eval() {}

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
    }

    #[test]
    fn table_3d_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let expression = NumericTableExpression::Table3D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(1),
            ElementExpression::Constant(2),
        );
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            60
        );
    }

    #[test]
    fn table_3d_sum_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let expression = NumericTableExpression::Table3DSum(
            0,
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Reference(ReferenceExpression::Variable(1)),
            SetExpression::Reference(ReferenceExpression::Variable(1)),
        );
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            240
        );
    }

    #[test]
    fn table_3d_sum_x_eval() {
        let registry = generate_registry();
        let state = generate_state();

        let expression = NumericTableExpression::Table3DSumX(
            0,
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            ElementExpression::Constant(1),
            ElementExpression::Constant(2),
        );
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            120
        );

        let expression = NumericTableExpression::Table3DSumX(
            0,
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
            ElementExpression::Constant(1),
            ElementExpression::Constant(2),
        );
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            120
        );
    }

    #[test]
    fn table_3d_vector_sum_x_eval() {
        let registry = generate_registry();
        let state = generate_state();

        let expression = NumericTableExpression::Table3DVectorSumX(
            0,
            VectorExpression::Reference(ReferenceExpression::Variable(0)),
            ElementExpression::Constant(1),
            ElementExpression::Constant(2),
        );
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            120
        );
    }

    #[test]
    fn table_3d_sum_y_eval() {
        let registry = generate_registry();
        let state = generate_state();

        let expression = NumericTableExpression::Table3DSumY(
            0,
            ElementExpression::Constant(1),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            ElementExpression::Constant(2),
        );
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            120
        );

        let expression = NumericTableExpression::Table3DSumY(
            0,
            ElementExpression::Constant(1),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
            ElementExpression::Constant(2),
        );
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            120
        );
    }

    #[test]
    fn table_3d_vector_sum_y_eval() {
        let registry = generate_registry();
        let state = generate_state();

        let expression = NumericTableExpression::Table3DVectorSumY(
            0,
            ElementExpression::Constant(1),
            VectorExpression::Reference(ReferenceExpression::Variable(0)),
            ElementExpression::Constant(2),
        );
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            120
        );
    }

    #[test]
    fn table_3d_sum_z_eval() {
        let registry = generate_registry();
        let state = generate_state();

        let expression = NumericTableExpression::Table3DSumZ(
            0,
            ElementExpression::Constant(1),
            ElementExpression::Constant(2),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            160
        );

        let expression = NumericTableExpression::Table3DSumZ(
            0,
            ElementExpression::Constant(1),
            ElementExpression::Constant(2),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
        );
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            160
        );
    }

    #[test]
    fn table_3d_vector_sum_z_eval() {
        let registry = generate_registry();
        let state = generate_state();

        let expression = NumericTableExpression::Table3DVectorSumZ(
            0,
            ElementExpression::Constant(1),
            ElementExpression::Constant(2),
            VectorExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            160
        );
    }

    #[test]
    fn table_3d_sum_xy_eval() {
        let registry = generate_registry();
        let state = generate_state();

        let expression = NumericTableExpression::Table3DSumXY(
            0,
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Reference(ReferenceExpression::Variable(1)),
            ElementExpression::Constant(2),
        );
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            180
        );
    }

    #[test]
    fn table_3d_sum_xz_eval() {
        let registry = generate_registry();
        let state = generate_state();

        let expression = NumericTableExpression::Table3DSumXZ(
            0,
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            ElementExpression::Constant(2),
            SetExpression::Reference(ReferenceExpression::Variable(1)),
        );
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            300
        );
    }

    #[test]
    fn table_3d_sum_yz_eval() {
        let registry = generate_registry();
        let state = generate_state();

        let expression = NumericTableExpression::Table3DSumYZ(
            0,
            ElementExpression::Constant(2),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Reference(ReferenceExpression::Variable(1)),
        );
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            180
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
                ArgumentExpression::Element(ElementExpression::Constant(1)),
                ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Variable(0))),
                ArgumentExpression::Vector(VectorExpression::Reference(
                    ReferenceExpression::Variable(1),
                )),
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
    fn table_2d_sum_x_simplify() {
        let registry = generate_registry();

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
    fn table_3d_simplify() {
        let registry = generate_registry();

        let expression = NumericTableExpression::Table3D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
        );
        assert_eq!(
            expression.simplify(&registry, &registry.integer_tables),
            NumericTableExpression::Constant(10)
        );

        let expression = NumericTableExpression::Table3D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
            ElementExpression::Variable(0),
        );
        assert_eq!(
            expression.simplify(&registry, &registry.integer_tables),
            expression
        );
    }

    #[test]
    fn table_3d_sum_simplify() {
        let registry = generate_registry();

        let expression = NumericTableExpression::Table3DSum(
            0,
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Reference(ReferenceExpression::Variable(1)),
            SetExpression::Reference(ReferenceExpression::Variable(2)),
        );
        assert_eq!(
            expression.simplify(&registry, &registry.integer_tables),
            expression
        );
    }

    #[test]
    fn table_3d_sum_x_simplify() {
        let registry = generate_registry();

        let expression = NumericTableExpression::Table3DSumX(
            0,
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
        );
        assert_eq!(
            expression.simplify(&registry, &registry.integer_tables),
            expression
        );
    }

    #[test]
    fn table_3d_sum_y_simplify() {
        let registry = generate_registry();

        let expression = NumericTableExpression::Table3DSumY(
            0,
            ElementExpression::Constant(0),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            ElementExpression::Constant(0),
        );
        assert_eq!(
            expression.simplify(&registry, &registry.integer_tables),
            expression
        );
    }

    #[test]
    fn table_3d_sum_z_simplify() {
        let registry = generate_registry();

        let expression = NumericTableExpression::Table3DSumZ(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert_eq!(
            expression.simplify(&registry, &registry.integer_tables),
            expression
        );
    }

    #[test]
    fn table_3d_sum_xy_simplify() {
        let registry = generate_registry();

        let expression = NumericTableExpression::Table3DSumXY(
            0,
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            ElementExpression::Constant(0),
        );
        assert_eq!(
            expression.simplify(&registry, &registry.integer_tables),
            expression,
        );
    }

    #[test]
    fn table_3d_sum_xz_simplify() {
        let registry = generate_registry();

        let expression = NumericTableExpression::Table3DSumXZ(
            0,
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            ElementExpression::Constant(0),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert_eq!(
            expression.simplify(&registry, &registry.integer_tables),
            expression
        );
    }

    #[test]
    fn table_3d_sum_yz_simplify() {
        let registry = generate_registry();

        let expression = NumericTableExpression::Table3DSumYZ(
            0,
            ElementExpression::Constant(0),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
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

        let expression = NumericTableExpression::TableSum(
            0,
            vec![
                ArgumentExpression::Element(ElementExpression::Constant(0)),
                ArgumentExpression::Element(ElementExpression::Constant(1)),
                ArgumentExpression::Element(ElementExpression::Constant(0)),
                ArgumentExpression::Element(ElementExpression::Variable(0)),
            ],
        );
        assert_eq!(
            expression.simplify(&registry, &registry.integer_tables),
            expression
        );
    }
}
