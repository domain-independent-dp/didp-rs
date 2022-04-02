use super::element_expression::{ElementExpression, SetExpression, VectorExpression};
use super::numeric_table_expression::ArgumentExpression;
use super::reference_expression::ReferenceExpression;
use super::util;
use crate::state::State;
use crate::table_data::TableData;
use crate::table_registry::TableRegistry;
use crate::variable::Numeric;

#[derive(Debug, PartialEq, Clone)]
pub enum TableVectorExpression<T: Numeric> {
    Constant(Vec<T>),
    Table(usize, Vec<VectorOrElementExpression>),
    TableSum(usize, Vec<ArgumentExpression>),
    Table1D(usize, VectorExpression),
    Table2D(usize, VectorExpression, VectorExpression),
    Table2DX(usize, VectorExpression, ElementExpression),
    Table2DY(usize, ElementExpression, VectorExpression),
    Table2DXSum(usize, VectorExpression, SetExpression),
    Table2DYSum(usize, SetExpression, VectorExpression),
}

impl<T: Numeric> TableVectorExpression<T> {
    pub fn eval(&self, state: &State, registry: &TableRegistry, tables: &TableData<T>) -> Vec<T> {
        let vector_variables = &state.signature_variables.vector_variables;
        let vector_tables = &registry.vector_tables;
        let set_variables = &state.signature_variables.set_variables;
        let set_tables = &registry.set_tables;
        match self {
            Self::Constant(vector) => vector.clone(),
            Self::Table(i, args) => Self::eval_table(*i, args, state, registry, tables),
            Self::TableSum(i, args) => Self::eval_table_sum(*i, args, state, registry, tables),
            Self::Table1D(i, VectorExpression::Reference(x)) => x
                .eval(state, registry, vector_variables, vector_tables)
                .iter()
                .map(|x| tables.tables_1d[*i].eval(*x))
                .collect(),
            Self::Table1D(i, x) => x
                .eval(state, registry)
                .into_iter()
                .map(|x| tables.tables_1d[*i].eval(x))
                .collect(),
            Self::Table2D(i, VectorExpression::Reference(x), VectorExpression::Reference(y)) => x
                .eval(state, registry, vector_variables, vector_tables)
                .iter()
                .zip(y.eval(state, registry, vector_variables, vector_tables))
                .map(|(x, y)| tables.tables_2d[*i].eval(*x, *y))
                .collect(),
            Self::Table2D(i, VectorExpression::Reference(x), y) => x
                .eval(state, registry, vector_variables, vector_tables)
                .iter()
                .zip(y.eval(state, registry))
                .map(|(x, y)| tables.tables_2d[*i].eval(*x, y))
                .collect(),
            Self::Table2D(i, x, VectorExpression::Reference(y)) => x
                .eval(state, registry)
                .into_iter()
                .zip(y.eval(state, registry, vector_variables, vector_tables))
                .map(|(x, y)| tables.tables_2d[*i].eval(x, *y))
                .collect(),
            Self::Table2D(i, x, y) => x
                .eval(state, registry)
                .into_iter()
                .zip(y.eval(state, registry))
                .map(|(x, y)| tables.tables_2d[*i].eval(x, y))
                .collect(),
            Self::Table2DX(i, VectorExpression::Reference(x), y) => {
                let y = y.eval(state, registry);
                x.eval(state, registry, vector_variables, vector_tables)
                    .iter()
                    .map(|x| tables.tables_2d[*i].eval(*x, y))
                    .collect()
            }
            Self::Table2DX(i, x, y) => {
                let y = y.eval(state, registry);
                x.eval(state, registry)
                    .into_iter()
                    .map(|x| tables.tables_2d[*i].eval(x, y))
                    .collect()
            }
            Self::Table2DY(i, x, VectorExpression::Reference(y)) => {
                let x = x.eval(state, registry);
                y.eval(state, registry, vector_variables, vector_tables)
                    .iter()
                    .map(|y| tables.tables_2d[*i].eval(x, *y))
                    .collect()
            }
            Self::Table2DY(i, x, y) => {
                let x = x.eval(state, registry);
                y.eval(state, registry)
                    .into_iter()
                    .map(|y| tables.tables_2d[*i].eval(x, y))
                    .collect()
            }
            Self::Table2DXSum(i, VectorExpression::Reference(x), SetExpression::Reference(y)) => {
                let y = y.eval(state, registry, set_variables, set_tables);
                x.eval(state, registry, vector_variables, vector_tables)
                    .iter()
                    .map(|x| tables.tables_2d[*i].sum_y(*x, y.ones()))
                    .collect()
            }
            Self::Table2DXSum(i, VectorExpression::Reference(x), y) => {
                let y = y.eval(state, registry);
                x.eval(state, registry, vector_variables, vector_tables)
                    .iter()
                    .map(|x| tables.tables_2d[*i].sum_y(*x, y.ones()))
                    .collect()
            }
            Self::Table2DXSum(i, x, SetExpression::Reference(y)) => {
                let y = y.eval(state, registry, set_variables, set_tables);
                x.eval(state, registry)
                    .into_iter()
                    .map(|x| tables.tables_2d[*i].sum_y(x, y.ones()))
                    .collect()
            }
            Self::Table2DXSum(i, x, y) => {
                let y = y.eval(state, registry);
                x.eval(state, registry)
                    .into_iter()
                    .map(|x| tables.tables_2d[*i].sum_y(x, y.ones()))
                    .collect()
            }
            Self::Table2DYSum(i, SetExpression::Reference(x), VectorExpression::Reference(y)) => {
                let x = x.eval(state, registry, set_variables, set_tables);
                y.eval(state, registry, vector_variables, vector_tables)
                    .iter()
                    .map(|y| tables.tables_2d[*i].sum_x(x.ones(), *y))
                    .collect()
            }
            Self::Table2DYSum(i, SetExpression::Reference(x), y) => {
                let x = x.eval(state, registry, set_variables, set_tables);
                y.eval(state, registry)
                    .into_iter()
                    .map(|y| tables.tables_2d[*i].sum_x(x.ones(), y))
                    .collect()
            }
            Self::Table2DYSum(i, x, VectorExpression::Reference(y)) => {
                let x = x.eval(state, registry);
                y.eval(state, registry, vector_variables, vector_tables)
                    .iter()
                    .map(|y| tables.tables_2d[*i].sum_x(x.ones(), *y))
                    .collect()
            }
            Self::Table2DYSum(i, x, y) => {
                let x = x.eval(state, registry);
                y.eval(state, registry)
                    .into_iter()
                    .map(|y| tables.tables_2d[*i].sum_x(x.ones(), y))
                    .collect()
            }
        }
    }

    pub fn simplify(
        &self,
        registry: &TableRegistry,
        tables: &TableData<T>,
    ) -> TableVectorExpression<T> {
        match self {
            Self::Table(i, args) => Self::simplify_table(*i, args, registry, tables),
            Self::TableSum(i, args) => Self::simplify_table_sum(*i, args, registry, tables),
            Self::Table1D(i, x) => match x.simplify(registry) {
                VectorExpression::Reference(ReferenceExpression::Constant(x)) => {
                    Self::Constant(x.iter().map(|x| tables.tables_1d[*i].eval(*x)).collect())
                }
                x => Self::Table1D(*i, x),
            },
            Self::Table2D(i, x, y) => match (x.simplify(registry), y.simplify(registry)) {
                (
                    VectorExpression::Reference(ReferenceExpression::Constant(x)),
                    VectorExpression::Reference(ReferenceExpression::Constant(y)),
                ) => Self::Constant(
                    x.into_iter()
                        .zip(y)
                        .map(|(x, y)| tables.tables_2d[*i].eval(x, y))
                        .collect(),
                ),
                (x, y) => Self::Table2D(*i, x, y),
            },
            Self::Table2DX(i, x, y) => match (x.simplify(registry), y.simplify(registry)) {
                (
                    VectorExpression::Reference(ReferenceExpression::Constant(x)),
                    ElementExpression::Constant(y),
                ) => Self::Constant(
                    x.into_iter()
                        .map(|x| tables.tables_2d[*i].eval(x, y))
                        .collect(),
                ),
                (x, y) => Self::Table2DX(*i, x, y),
            },
            Self::Table2DY(i, x, y) => match (x.simplify(registry), y.simplify(registry)) {
                (
                    ElementExpression::Constant(x),
                    VectorExpression::Reference(ReferenceExpression::Constant(y)),
                ) => Self::Constant(
                    y.into_iter()
                        .map(|y| tables.tables_2d[*i].eval(x, y))
                        .collect(),
                ),
                (x, y) => Self::Table2DY(*i, x, y),
            },
            Self::Table2DXSum(i, x, y) => match (x.simplify(registry), y.simplify(registry)) {
                (
                    VectorExpression::Reference(ReferenceExpression::Constant(x)),
                    SetExpression::Reference(ReferenceExpression::Constant(y)),
                ) => Self::Constant(
                    x.into_iter()
                        .map(|x| tables.tables_2d[*i].sum_y(x, y.ones()))
                        .collect(),
                ),
                (x, y) => Self::Table2DXSum(*i, x, y),
            },
            Self::Table2DYSum(i, x, y) => match (x.simplify(registry), y.simplify(registry)) {
                (
                    SetExpression::Reference(ReferenceExpression::Constant(x)),
                    VectorExpression::Reference(ReferenceExpression::Constant(y)),
                ) => Self::Constant(
                    y.into_iter()
                        .map(|y| tables.tables_2d[*i].sum_x(x.ones(), y))
                        .collect(),
                ),
                (x, y) => Self::Table2DYSum(*i, x, y),
            },
            _ => self.clone(),
        }
    }

    fn eval_table(
        i: usize,
        args: &[VectorOrElementExpression],
        state: &State,
        registry: &TableRegistry,
        tables: &TableData<T>,
    ) -> Vec<T> {
        let mut result = vec![vec![]];
        let mut vector_mode = false;
        for arg in args {
            match arg {
                VectorOrElementExpression::Element(element) => {
                    let element = element.eval(state, registry);
                    result.iter_mut().for_each(|r| r.push(element));
                }
                VectorOrElementExpression::Vector(vector) => match vector {
                    VectorExpression::Reference(vector) => {
                        let vector = vector.eval(
                            state,
                            registry,
                            &state.signature_variables.vector_variables,
                            &registry.vector_tables,
                        );
                        if vector_mode {
                            result.iter_mut().zip(vector).for_each(|(r, v)| r.push(*v));
                        } else {
                            result = vector
                                .iter()
                                .map(|v| {
                                    let mut r = result[0].clone();
                                    r.push(*v);
                                    r
                                })
                                .collect();
                            vector_mode = true;
                        }
                    }
                    vector => {
                        let vector = vector.eval(state, registry);
                        if vector_mode {
                            result.iter_mut().zip(vector).for_each(|(r, v)| r.push(v));
                        } else {
                            result = vector
                                .into_iter()
                                .map(|v| {
                                    let mut r = result[0].clone();
                                    r.push(v);
                                    r
                                })
                                .collect();
                            vector_mode = true;
                        }
                    }
                },
            }
        }
        match args.len() {
            1 => result
                .into_iter()
                .map(|r| tables.tables_1d[i].eval(r[0]))
                .collect(),
            2 => result
                .into_iter()
                .map(|r| tables.tables_2d[i].eval(r[0], r[1]))
                .collect(),
            3 => result
                .into_iter()
                .map(|r| tables.tables_3d[i].eval(r[0], r[1], r[2]))
                .collect(),
            _ => result
                .into_iter()
                .map(|r| tables.tables[i].eval(&r))
                .collect(),
        }
    }

    fn simplify_table(
        i: usize,
        args: &[VectorOrElementExpression],
        registry: &TableRegistry,
        tables: &TableData<T>,
    ) -> TableVectorExpression<T> {
        let args: Vec<VectorOrElementExpression> =
            args.iter().map(|x| x.simplify(registry)).collect();
        let mut simplified_args = vec![vec![]];
        let mut vector_mode = false;
        for arg in &args {
            match arg {
                VectorOrElementExpression::Element(ElementExpression::Constant(element)) => {
                    simplified_args.iter_mut().for_each(|r| r.push(*element));
                }
                VectorOrElementExpression::Vector(VectorExpression::Reference(
                    ReferenceExpression::Constant(vector),
                )) => {
                    if vector_mode {
                        simplified_args
                            .iter_mut()
                            .zip(vector)
                            .for_each(|(r, v)| r.push(*v));
                    } else {
                        simplified_args = vector
                            .iter()
                            .map(|v| {
                                let mut r = simplified_args[0].clone();
                                r.push(*v);
                                r
                            })
                            .collect();
                        vector_mode = true;
                    }
                }
                _ => return Self::Table(i, args),
            }
        }
        Self::Constant(match args.len() {
            1 => simplified_args
                .into_iter()
                .map(|r| tables.tables_1d[i].eval(r[0]))
                .collect(),
            2 => simplified_args
                .into_iter()
                .map(|r| tables.tables_2d[i].eval(r[0], r[1]))
                .collect(),
            3 => simplified_args
                .into_iter()
                .map(|r| tables.tables_3d[i].eval(r[0], r[1], r[2]))
                .collect(),
            _ => simplified_args
                .into_iter()
                .map(|r| tables.tables[i].eval(&r))
                .collect(),
        })
    }

    fn eval_table_sum(
        i: usize,
        args: &[ArgumentExpression],
        state: &State,
        registry: &TableRegistry,
        tables: &TableData<T>,
    ) -> Vec<T> {
        let mut result = vec![vec![vec![]]];
        let mut vector_mode = false;
        for arg in args {
            match arg {
                ArgumentExpression::Element(element) => {
                    let element = element.eval(state, registry);
                    result
                        .iter_mut()
                        .for_each(|rr| rr.iter_mut().for_each(|r| r.push(element)));
                }
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
                                .collect()
                        }
                        _ => {
                            let set = &set.eval(state, registry);
                            result
                                .into_iter()
                                .map(|rr| util::expand_vector_with_set(rr, &set))
                                .collect()
                        }
                    };
                }
                ArgumentExpression::Vector(vector) => {
                    if vector_mode {
                        match vector {
                            VectorExpression::Reference(vector) => result
                                .iter_mut()
                                .zip(vector.eval(
                                    state,
                                    registry,
                                    &state.signature_variables.vector_variables,
                                    &registry.vector_tables,
                                ))
                                .for_each(|(rr, v)| rr.iter_mut().for_each(|r| r.push(*v))),
                            _ => result
                                .iter_mut()
                                .zip(vector.eval(state, registry))
                                .for_each(|(rr, v)| rr.iter_mut().for_each(|r| r.push(v))),
                        }
                    } else {
                        result = match vector {
                            VectorExpression::Reference(vector) => vector
                                .eval(
                                    state,
                                    registry,
                                    &state.signature_variables.vector_variables,
                                    &registry.vector_tables,
                                )
                                .iter()
                                .map(|v| {
                                    let mut rr = result[0].clone();
                                    rr.iter_mut().for_each(|r| r.push(*v));
                                    rr
                                })
                                .collect(),
                            _ => vector
                                .eval(state, registry)
                                .into_iter()
                                .map(|v| {
                                    let mut rr = result[0].clone();
                                    rr.iter_mut().for_each(|r| r.push(v));
                                    rr
                                })
                                .collect(),
                        };
                        vector_mode = true;
                    }
                }
            }
        }
        match args.len() {
            1 => result
                .into_iter()
                .map(|rr| rr.into_iter().map(|r| tables.tables_1d[i].eval(r[0])).sum())
                .collect(),
            2 => result
                .into_iter()
                .map(|rr| {
                    rr.into_iter()
                        .map(|r| tables.tables_2d[i].eval(r[0], r[1]))
                        .sum()
                })
                .collect(),
            3 => result
                .into_iter()
                .map(|rr| {
                    rr.into_iter()
                        .map(|r| tables.tables_3d[i].eval(r[0], r[1], r[2]))
                        .sum()
                })
                .collect(),
            _ => result
                .into_iter()
                .map(|rr| rr.into_iter().map(|r| tables.tables[i].eval(&r)).sum())
                .collect(),
        }
    }

    fn simplify_table_sum(
        i: usize,
        args: &[ArgumentExpression],
        registry: &TableRegistry,
        tables: &TableData<T>,
    ) -> TableVectorExpression<T> {
        let args: Vec<ArgumentExpression> = args.iter().map(|x| x.simplify(registry)).collect();
        let mut simplified_args = vec![vec![vec![]]];
        let mut vector_mode = false;
        for arg in &args {
            match arg {
                ArgumentExpression::Element(ElementExpression::Constant(element)) => {
                    simplified_args
                        .iter_mut()
                        .for_each(|rr| rr.iter_mut().for_each(|r| r.push(*element)));
                }
                ArgumentExpression::Set(SetExpression::Reference(
                    ReferenceExpression::Constant(set),
                )) => {
                    simplified_args = simplified_args
                        .into_iter()
                        .map(|rr| util::expand_vector_with_set(rr, set))
                        .collect();
                }
                ArgumentExpression::Vector(VectorExpression::Reference(
                    ReferenceExpression::Constant(vector),
                )) => {
                    if vector_mode {
                        simplified_args
                            .iter_mut()
                            .zip(vector)
                            .for_each(|(rr, v)| rr.iter_mut().for_each(|r| r.push(*v)));
                    } else {
                        simplified_args = vector
                            .iter()
                            .map(|v| {
                                let mut rr = simplified_args[0].clone();
                                rr.iter_mut().for_each(|r| r.push(*v));
                                rr
                            })
                            .collect();
                        vector_mode = true;
                    }
                }
                _ => return Self::TableSum(i, args),
            }
        }
        Self::Constant(match args.len() {
            1 => simplified_args
                .into_iter()
                .map(|rr| rr.into_iter().map(|r| tables.tables_1d[i].eval(r[0])).sum())
                .collect(),
            2 => simplified_args
                .into_iter()
                .map(|rr| {
                    rr.into_iter()
                        .map(|r| tables.tables_2d[i].eval(r[0], r[1]))
                        .sum()
                })
                .collect(),
            3 => simplified_args
                .into_iter()
                .map(|rr| {
                    rr.into_iter()
                        .map(|r| tables.tables_3d[i].eval(r[0], r[1], r[2]))
                        .sum()
                })
                .collect(),
            _ => simplified_args
                .into_iter()
                .map(|rr| rr.into_iter().map(|r| tables.tables[i].eval(&r)).sum())
                .collect(),
        })
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum VectorOrElementExpression {
    Vector(VectorExpression),
    Element(ElementExpression),
}

impl VectorOrElementExpression {
    pub fn simplify(&self, registry: &TableRegistry) -> VectorOrElementExpression {
        match self {
            VectorOrElementExpression::Vector(vector) => {
                VectorOrElementExpression::Vector(vector.simplify(registry))
            }
            VectorOrElementExpression::Element(element) => {
                VectorOrElementExpression::Element(element.simplify(registry))
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::state::*;
    use crate::table;
    use crate::table_data;
    use crate::variable::*;
    use ordered_float::OrderedFloat;
    use rustc_hash::FxHashMap;
    use std::rc::Rc;

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
                integer_variables: vec![1, 2, 3],
                continuous_variables: vec![OrderedFloat(1.0), OrderedFloat(2.0), OrderedFloat(3.0)],
            }),
            resource_variables: ResourceVariables {
                integer_variables: vec![4, 5, 6],
                continuous_variables: vec![OrderedFloat(4.0), OrderedFloat(5.0), OrderedFloat(6.0)],
            },
        }
    }

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

        let integer_tables = table_data::TableData {
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

        let mut name_to_constant = FxHashMap::default();
        name_to_constant.insert(String::from("cf0"), 0.0);

        let tables_1d = vec![table::Table1D::new(vec![10.0, 20.0, 30.0])];
        let mut name_to_table_1d = FxHashMap::default();
        name_to_table_1d.insert(String::from("cf1"), 0);

        let tables_2d = vec![table::Table2D::new(vec![
            vec![10.0, 20.0, 30.0],
            vec![40.0, 50.0, 60.0],
            vec![70.0, 80.0, 90.0],
        ])];
        let mut name_to_table_2d = FxHashMap::default();
        name_to_table_2d.insert(String::from("cf2"), 0);

        let tables_3d = vec![table::Table3D::new(vec![
            vec![
                vec![10.0, 20.0, 30.0],
                vec![40.0, 50.0, 60.0],
                vec![70.0, 80.0, 90.0],
            ],
            vec![
                vec![10.0, 20.0, 30.0],
                vec![40.0, 50.0, 60.0],
                vec![70.0, 80.0, 90.0],
            ],
            vec![
                vec![10.0, 20.0, 30.0],
                vec![40.0, 50.0, 60.0],
                vec![70.0, 80.0, 90.0],
            ],
        ])];
        let mut name_to_table_3d = FxHashMap::default();
        name_to_table_3d.insert(String::from("cf3"), 0);

        let mut map = FxHashMap::default();
        let key = vec![0, 1, 0, 0];
        map.insert(key, 100.0);
        let key = vec![0, 1, 0, 1];
        map.insert(key, 200.0);
        let key = vec![0, 1, 2, 0];
        map.insert(key, 300.0);
        let key = vec![0, 1, 2, 1];
        map.insert(key, 400.0);
        let tables = vec![table::Table::new(map, 0.0)];
        let mut name_to_table = FxHashMap::default();
        name_to_table.insert(String::from("cf4"), 0);

        let continuous_tables = table_data::TableData {
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

        TableRegistry {
            integer_tables,
            continuous_tables,
            ..Default::default()
        }
    }
    #[test]
    fn vector_table_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let tables = &registry.integer_tables;
        let expression = TableVectorExpression::Table(
            0,
            vec![
                VectorOrElementExpression::Element(ElementExpression::Constant(0)),
                VectorOrElementExpression::Element(ElementExpression::Constant(1)),
                VectorOrElementExpression::Vector(VectorExpression::Reference(
                    ReferenceExpression::Constant(vec![0, 2]),
                )),
                VectorOrElementExpression::Element(ElementExpression::Constant(0)),
            ],
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![100, 300]);
    }

    #[test]
    fn vector_table_sum_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let tables = &registry.integer_tables;
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(1);
        let expression = TableVectorExpression::TableSum(
            0,
            vec![
                ArgumentExpression::Element(ElementExpression::Constant(0)),
                ArgumentExpression::Element(ElementExpression::Constant(1)),
                ArgumentExpression::Vector(VectorExpression::Reference(
                    ReferenceExpression::Constant(vec![0, 2]),
                )),
                ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Constant(
                    set,
                ))),
            ],
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![300, 700]);
        let mut set = Set::with_capacity(3);
        set.insert(2);
        let expression = TableVectorExpression::TableSum(
            0,
            vec![
                ArgumentExpression::Element(ElementExpression::Constant(0)),
                ArgumentExpression::Element(ElementExpression::Constant(1)),
                ArgumentExpression::Vector(VectorExpression::Reverse(Box::new(
                    VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 2])),
                ))),
                ArgumentExpression::Set(SetExpression::Complement(Box::new(
                    SetExpression::Reference(ReferenceExpression::Constant(set)),
                ))),
            ],
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![700, 300]);
    }

    #[test]
    fn vector_table_1d_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let tables = &registry.integer_tables;
        let expression = TableVectorExpression::Table1D(
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![10, 20]);
        let expression = TableVectorExpression::Table1D(
            0,
            VectorExpression::Reverse(Box::new(VectorExpression::Reference(
                ReferenceExpression::Constant(vec![0, 1]),
            ))),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![20, 10]);
    }

    #[test]
    fn vector_table_2d_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let tables = &registry.integer_tables;
        let expression = TableVectorExpression::Table2D(
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![10, 50]);
        let expression = TableVectorExpression::Table2D(
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            VectorExpression::Reverse(Box::new(VectorExpression::Reference(
                ReferenceExpression::Constant(vec![0, 1]),
            ))),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![20, 40]);
        let expression = TableVectorExpression::Table2D(
            0,
            VectorExpression::Reverse(Box::new(VectorExpression::Reference(
                ReferenceExpression::Constant(vec![0, 1]),
            ))),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![40, 20]);
        let expression = TableVectorExpression::Table2D(
            0,
            VectorExpression::Reverse(Box::new(VectorExpression::Reference(
                ReferenceExpression::Constant(vec![0, 1]),
            ))),
            VectorExpression::Reverse(Box::new(VectorExpression::Reference(
                ReferenceExpression::Constant(vec![0, 1]),
            ))),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![50, 10]);
    }

    #[test]
    fn vector_table_2d_x_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let tables = &registry.integer_tables;
        let expression = TableVectorExpression::Table2DX(
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            ElementExpression::Constant(0),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![10, 40]);
        let expression = TableVectorExpression::Table2DX(
            0,
            VectorExpression::Reverse(Box::new(VectorExpression::Reference(
                ReferenceExpression::Constant(vec![0, 1]),
            ))),
            ElementExpression::Constant(0),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![40, 10]);
    }

    #[test]
    fn vector_table_2d_y_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let tables = &registry.integer_tables;
        let expression = TableVectorExpression::Table2DY(
            0,
            ElementExpression::Constant(0),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![10, 20]);
        let expression = TableVectorExpression::Table2DY(
            0,
            ElementExpression::Constant(0),
            VectorExpression::Reverse(Box::new(VectorExpression::Reference(
                ReferenceExpression::Constant(vec![0, 1]),
            ))),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![20, 10]);
    }

    #[test]
    fn vector_table_2d_x_sum_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let tables = &registry.integer_tables;
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(1);
        let expression = TableVectorExpression::Table2DXSum(
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            SetExpression::Reference(ReferenceExpression::Constant(set.clone())),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![30, 90]);
        let expression = TableVectorExpression::Table2DXSum(
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            SetExpression::Complement(Box::new(SetExpression::Reference(
                ReferenceExpression::Constant(set.clone()),
            ))),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![30, 60]);
        let expression = TableVectorExpression::Table2DXSum(
            0,
            VectorExpression::Reverse(Box::new(VectorExpression::Reference(
                ReferenceExpression::Constant(vec![0, 1]),
            ))),
            SetExpression::Reference(ReferenceExpression::Constant(set.clone())),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![90, 30]);
        let expression = TableVectorExpression::Table2DXSum(
            0,
            VectorExpression::Reverse(Box::new(VectorExpression::Reference(
                ReferenceExpression::Constant(vec![0, 1]),
            ))),
            SetExpression::Complement(Box::new(SetExpression::Reference(
                ReferenceExpression::Constant(set),
            ))),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![60, 30]);
    }

    #[test]
    fn vector_table_2d_y_sum_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let tables = &registry.integer_tables;
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(1);
        let expression = TableVectorExpression::Table2DYSum(
            0,
            SetExpression::Reference(ReferenceExpression::Constant(set.clone())),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![50, 70]);
        let expression = TableVectorExpression::Table2DYSum(
            0,
            SetExpression::Complement(Box::new(SetExpression::Reference(
                ReferenceExpression::Constant(set.clone()),
            ))),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![70, 80]);
        let expression = TableVectorExpression::Table2DYSum(
            0,
            SetExpression::Reference(ReferenceExpression::Constant(set.clone())),
            VectorExpression::Reverse(Box::new(VectorExpression::Reference(
                ReferenceExpression::Constant(vec![0, 1]),
            ))),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![70, 50]);
        let expression = TableVectorExpression::Table2DYSum(
            0,
            SetExpression::Complement(Box::new(SetExpression::Reference(
                ReferenceExpression::Constant(set),
            ))),
            VectorExpression::Reverse(Box::new(VectorExpression::Reference(
                ReferenceExpression::Constant(vec![0, 1]),
            ))),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![80, 70]);
    }

    #[test]
    fn vector_table_simplify() {
        let registry = generate_registry();
        let tables = &registry.integer_tables;
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(1);
        let expression = TableVectorExpression::Table(
            0,
            vec![
                VectorOrElementExpression::Element(ElementExpression::Constant(0)),
                VectorOrElementExpression::Element(ElementExpression::Constant(1)),
                VectorOrElementExpression::Vector(VectorExpression::Reference(
                    ReferenceExpression::Constant(vec![0, 2]),
                )),
                VectorOrElementExpression::Element(ElementExpression::Constant(0)),
            ],
        );
        assert_eq!(
            expression.simplify(&registry, tables),
            TableVectorExpression::Constant(vec![100, 300])
        );
        let mut set = Set::with_capacity(3);
        set.insert(2);
        let expression = TableVectorExpression::Table(
            0,
            vec![
                VectorOrElementExpression::Element(ElementExpression::Variable(0)),
                VectorOrElementExpression::Element(ElementExpression::Constant(1)),
                VectorOrElementExpression::Vector(VectorExpression::Reference(
                    ReferenceExpression::Constant(vec![0, 2]),
                )),
                VectorOrElementExpression::Element(ElementExpression::Variable(0)),
            ],
        );
        assert_eq!(expression.simplify(&registry, tables), expression);
    }

    #[test]
    fn vector_table_sum_simplify() {
        let registry = generate_registry();
        let tables = &registry.integer_tables;
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(1);
        let expression = TableVectorExpression::TableSum(
            0,
            vec![
                ArgumentExpression::Element(ElementExpression::Constant(0)),
                ArgumentExpression::Element(ElementExpression::Constant(1)),
                ArgumentExpression::Vector(VectorExpression::Reference(
                    ReferenceExpression::Constant(vec![0, 2]),
                )),
                ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Constant(
                    set,
                ))),
            ],
        );
        assert_eq!(
            expression.simplify(&registry, tables),
            TableVectorExpression::Constant(vec![300, 700])
        );
        let mut set = Set::with_capacity(3);
        set.insert(2);
        let expression = TableVectorExpression::TableSum(
            0,
            vec![
                ArgumentExpression::Element(ElementExpression::Variable(0)),
                ArgumentExpression::Element(ElementExpression::Constant(1)),
                ArgumentExpression::Vector(VectorExpression::Reference(
                    ReferenceExpression::Constant(vec![0, 2]),
                )),
                ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Constant(
                    set,
                ))),
            ],
        );
        assert_eq!(expression.simplify(&registry, tables), expression);
    }

    #[test]
    fn vector_table_1d_simplify() {
        let registry = generate_registry();
        let tables = &registry.integer_tables;
        let expression = TableVectorExpression::Table1D(
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert_eq!(
            expression.simplify(&registry, tables),
            TableVectorExpression::Constant(vec![10, 20])
        );
        let expression = TableVectorExpression::Table1D(
            0,
            VectorExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert_eq!(expression.simplify(&registry, tables), expression);
    }

    #[test]
    fn vector_table_2d_simplify() {
        let registry = generate_registry();
        let tables = &registry.integer_tables;
        let expression = TableVectorExpression::Table2D(
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert_eq!(
            expression.simplify(&registry, tables),
            TableVectorExpression::Constant(vec![10, 50])
        );
        let expression = TableVectorExpression::Table2D(
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            VectorExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert_eq!(expression.simplify(&registry, tables), expression);
    }

    #[test]
    fn vector_table_2d_x_simplify() {
        let registry = generate_registry();
        let tables = &registry.integer_tables;
        let expression = TableVectorExpression::Table2DX(
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            ElementExpression::Constant(0),
        );
        assert_eq!(
            expression.simplify(&registry, tables),
            TableVectorExpression::Constant(vec![10, 40])
        );
        let expression = TableVectorExpression::Table2DX(
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            ElementExpression::Variable(0),
        );
        assert_eq!(expression.simplify(&registry, tables), expression);
    }

    #[test]
    fn vector_table_2d_y_simplify() {
        let registry = generate_registry();
        let tables = &registry.integer_tables;
        let expression = TableVectorExpression::Table2DY(
            0,
            ElementExpression::Constant(0),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert_eq!(
            expression.simplify(&registry, tables),
            TableVectorExpression::Constant(vec![10, 20])
        );
        let expression = TableVectorExpression::Table2DY(
            0,
            ElementExpression::Variable(0),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert_eq!(expression.simplify(&registry, tables), expression);
    }

    #[test]
    fn vector_table_2d_x_sum_simplify() {
        let registry = generate_registry();
        let tables = &registry.integer_tables;
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(1);
        let expression = TableVectorExpression::Table2DXSum(
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            SetExpression::Reference(ReferenceExpression::Constant(set)),
        );
        assert_eq!(
            expression.simplify(&registry, tables),
            TableVectorExpression::Constant(vec![30, 90])
        );
        let expression = TableVectorExpression::Table2DXSum(
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert_eq!(expression.simplify(&registry, tables), expression);
    }

    #[test]
    fn vector_table_2d_y_sum_simplify() {
        let registry = generate_registry();
        let tables = &registry.integer_tables;
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(1);
        let expression = TableVectorExpression::Table2DYSum(
            0,
            SetExpression::Reference(ReferenceExpression::Constant(set)),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert_eq!(
            expression.simplify(&registry, tables),
            TableVectorExpression::Constant(vec![50, 70])
        );
        let expression = TableVectorExpression::Table2DYSum(
            0,
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert_eq!(expression.simplify(&registry, tables), expression);
    }
}
