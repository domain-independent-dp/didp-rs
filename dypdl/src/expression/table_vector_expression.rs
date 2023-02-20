use super::argument_expression::ArgumentExpression;
use super::element_expression::ElementExpression;
use super::numeric_operator::ReduceOperator;
use super::reference_expression::ReferenceExpression;
use super::set_expression::SetExpression;
use super::util;
use super::vector_expression::VectorExpression;
use crate::state::StateInterface;
use crate::table::Table2D;
use crate::table_data::TableData;
use crate::table_registry::TableRegistry;
use crate::variable_type::{Element, Numeric, Set};

/// Expression representing a vector of numeric values constructed from a table.
#[derive(Debug, PartialEq, Clone)]
pub enum TableVectorExpression<T: Numeric> {
    /// Constant.
    Constant(Vec<T>),
    /// A vector of numeric values in a table.
    Table(usize, Vec<VectorOrElementExpression>),
    /// A vector of the sum of numeric values in a table.
    ///
    /// If the length of input vectors are different, the longer ones are truncated.
    TableReduce(ReduceOperator, usize, Vec<ArgumentExpression>),
    /// A vector of numeric values in a 1D table.
    ///
    /// If the length of input vectors are different, the longer ones are truncated.
    Table1D(usize, VectorExpression),
    /// A vector of numeric values in a 2D table.
    ///
    /// Given table t and vector x, construct a numeric vector \[t\[x_0\], ..., t\[x_{n-1}\]\].
    Table2D(usize, VectorExpression, VectorExpression),
    /// A vector of numeric values in a 2D table.
    ///
    /// Given table t and vectors x and y, construct a numeric vector \[t\[x_0, y_0\], ..., t\[x_{n-1}, y_{n-1}\]\].
    /// If the length of two input vectors are different, the longer one is truncated.
    Table2DX(usize, VectorExpression, ElementExpression),
    /// A vector of numeric values in a 2D table.
    ///
    /// Given table t, vector x, and constant y, construct a numeric vector \[t\[x_0, y\], ..., t\[x_{n-1}, y\]\].
    Table2DY(usize, ElementExpression, VectorExpression),
    /// A vector of numeric values in a 3D table.
    ///
    /// Given table t, vectors x, y, and z, construct a numeric vector \[t\[x_0, y_0, z_0\], ..., t\[x_{n-1}, y_{n-1}, z_{n-1}\]\].
    Table3D(
        usize,
        VectorOrElementExpression,
        VectorOrElementExpression,
        VectorOrElementExpression,
    ),
    /// A vector constructed by taking the sum of numeric values over a set in a 2D table.
    //
    /// Given table t, vector x, and set y, construct a numeric vector \[\sum_{i \in s} t\[x_0, i\], ..., \sum_{i \in s} t\[x_{n-1}, i\]\].
    /// If the length of input vectors are different, the longer ones are truncated.
    Table2DXReduce(ReduceOperator, usize, VectorExpression, SetExpression),
    /// A vector constructed by taking the sum of numeric values over a set in a 2D table.
    ///
    /// Given table t, set x, and vector y, construct a numeric vector \[\sum_{i \in s} t\[i, y_0\], ..., \sum_{i \in s} t\[i, y_{n-1}\]\].
    Table2DYReduce(ReduceOperator, usize, SetExpression, VectorExpression),
    /// A vector constructed by taking the sum of numeric values over sets in a 3D table.
    ///
    /// If the length of input vectors are different, the longer ones are truncated.
    Table3DReduce(
        ReduceOperator,
        usize,
        ArgumentExpression,
        ArgumentExpression,
        ArgumentExpression,
    ),
}

impl<T: Numeric> TableVectorExpression<T> {
    /// Returns the evaluation result.
    ///
    /// # Panics
    ///
    /// Panics if the cost of the transition state is used or a min/max reduce operation is performed on an empty set or vector.
    pub fn eval<U: StateInterface>(
        &self,
        state: &U,
        registry: &TableRegistry,
        tables: &TableData<T>,
    ) -> Vec<T> {
        let vector_f = |i| state.get_vector_variable(i);
        let vector_tables = &registry.vector_tables;
        let set_f = |i| state.get_set_variable(i);
        let set_tables = &registry.set_tables;
        match self {
            Self::Constant(vector) => vector.clone(),
            Self::Table(i, args) => {
                let args = Self::eval_args(args.iter(), state, registry);
                args.into_iter()
                    .map(|args| tables.tables[*i].eval(&args))
                    .collect()
            }
            Self::TableReduce(op, i, args) => {
                let args = Self::eval_sum_args(args.iter(), state, registry);
                args.into_iter()
                    .map(|args| {
                        op.eval_iter(args.into_iter().map(|args| tables.tables[*i].eval(&args)))
                            .unwrap()
                    })
                    .collect()
            }
            Self::Table1D(i, VectorExpression::Reference(x)) => x
                .eval(state, registry, &vector_f, vector_tables)
                .iter()
                .map(|x| tables.tables_1d[*i].eval(*x))
                .collect(),
            Self::Table1D(i, x) => x
                .eval(state, registry)
                .into_iter()
                .map(|x| tables.tables_1d[*i].eval(x))
                .collect(),
            Self::Table2D(i, VectorExpression::Reference(x), VectorExpression::Reference(y)) => {
                let x = x
                    .eval(state, registry, &vector_f, vector_tables)
                    .iter()
                    .copied();
                let y = y
                    .eval(state, registry, &vector_f, vector_tables)
                    .iter()
                    .copied();
                Self::table_2d(&tables.tables_2d[*i], x, y)
            }
            Self::Table2D(i, VectorExpression::Reference(x), y) => {
                let x = x
                    .eval(state, registry, &vector_f, vector_tables)
                    .iter()
                    .copied();
                let y = y.eval(state, registry).into_iter();
                Self::table_2d(&tables.tables_2d[*i], x, y)
            }
            Self::Table2D(i, x, VectorExpression::Reference(y)) => {
                let x = x.eval(state, registry).into_iter();
                let y = y
                    .eval(state, registry, &vector_f, vector_tables)
                    .iter()
                    .copied();
                Self::table_2d(&tables.tables_2d[*i], x, y)
            }
            Self::Table2D(i, x, y) => {
                let x = x.eval(state, registry).into_iter();
                let y = y.eval(state, registry).into_iter();
                Self::table_2d(&tables.tables_2d[*i], x, y)
            }
            Self::Table2DX(i, VectorExpression::Reference(x), y) => {
                let y = y.eval(state, registry);
                x.eval(state, registry, &vector_f, vector_tables)
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
                y.eval(state, registry, &vector_f, vector_tables)
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
            Self::Table3D(i, x, y, z) => {
                let args = Self::eval_args([x, y, z].into_iter(), state, registry);
                args.into_iter()
                    .map(|args| tables.tables_3d[*i].eval(args[0], args[1], args[2]))
                    .collect()
            }
            Self::Table2DXReduce(
                op,
                i,
                VectorExpression::Reference(x),
                SetExpression::Reference(y),
            ) => {
                let x = x
                    .eval(state, registry, &vector_f, vector_tables)
                    .iter()
                    .copied();
                let y = y.eval(state, registry, &set_f, set_tables);
                Self::x_reduce_table_2d(op, &tables.tables_2d[*i], x, y)
            }
            Self::Table2DXReduce(op, i, VectorExpression::Reference(x), y) => {
                let x = x
                    .eval(state, registry, &vector_f, vector_tables)
                    .iter()
                    .copied();
                let y = y.eval(state, registry);
                Self::x_reduce_table_2d(op, &tables.tables_2d[*i], x, &y)
            }
            Self::Table2DXReduce(op, i, x, SetExpression::Reference(y)) => {
                let x = x.eval(state, registry).into_iter();
                let y = y.eval(state, registry, &set_f, set_tables);
                Self::x_reduce_table_2d(op, &tables.tables_2d[*i], x, y)
            }
            Self::Table2DXReduce(op, i, x, y) => {
                let x = x.eval(state, registry).into_iter();
                let y = y.eval(state, registry);
                Self::x_reduce_table_2d(op, &tables.tables_2d[*i], x, &y)
            }
            Self::Table2DYReduce(
                op,
                i,
                SetExpression::Reference(x),
                VectorExpression::Reference(y),
            ) => {
                let x = x.eval(state, registry, &set_f, set_tables);
                let y = y
                    .eval(state, registry, &vector_f, vector_tables)
                    .iter()
                    .copied();
                Self::y_reduce_table_2d(op, &tables.tables_2d[*i], x, y)
            }
            Self::Table2DYReduce(op, i, SetExpression::Reference(x), y) => {
                let x = x.eval(state, registry, &set_f, set_tables);
                let y = y.eval(state, registry).into_iter();
                Self::y_reduce_table_2d(op, &tables.tables_2d[*i], x, y)
            }
            Self::Table2DYReduce(op, i, x, VectorExpression::Reference(y)) => {
                let x = x.eval(state, registry);
                let y = y
                    .eval(state, registry, &vector_f, vector_tables)
                    .iter()
                    .copied();
                Self::y_reduce_table_2d(op, &tables.tables_2d[*i], &x, y)
            }
            Self::Table2DYReduce(op, i, x, y) => {
                let x = x.eval(state, registry);
                let y = y.eval(state, registry).into_iter();
                Self::y_reduce_table_2d(op, &tables.tables_2d[*i], &x, y)
            }
            Self::Table3DReduce(op, i, x, y, z) => {
                let args = Self::eval_sum_args([x, y, z].into_iter(), state, registry);
                args.into_iter()
                    .map(|args| {
                        op.eval_iter(
                            args.into_iter()
                                .map(|args| tables.tables_3d[*i].eval(args[0], args[1], args[2])),
                        )
                        .unwrap()
                    })
                    .collect()
            }
        }
    }

    /// Returns a simplified version by precomputation.
    ///
    /// # Panics
    ///
    /// Panics if a min/max reduce operation is performed on an empty set or vector.
    pub fn simplify(
        &self,
        registry: &TableRegistry,
        tables: &TableData<T>,
    ) -> TableVectorExpression<T> {
        match self {
            Self::Table(i, args) => {
                let args: Vec<VectorOrElementExpression> =
                    args.iter().map(|x| x.simplify(registry)).collect();
                if let Some(args) = Self::simplify_args(args.iter()) {
                    Self::Constant(
                        args.into_iter()
                            .map(|args| tables.tables[*i].eval(&args))
                            .collect(),
                    )
                } else {
                    Self::Table(*i, args)
                }
            }
            Self::TableReduce(op, i, args) => {
                let args: Vec<ArgumentExpression> =
                    args.iter().map(|x| x.simplify(registry)).collect();
                if let Some(args) = Self::simplify_sum_args(args.iter()) {
                    Self::Constant(
                        args.into_iter()
                            .map(|args| {
                                args.into_iter()
                                    .map(|args| tables.tables[*i].eval(&args))
                                    .sum()
                            })
                            .collect(),
                    )
                } else {
                    Self::TableReduce(op.clone(), *i, args)
                }
            }
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
                ) => Self::Constant(Self::table_2d(
                    &tables.tables_2d[*i],
                    x.into_iter(),
                    y.into_iter(),
                )),
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
            Self::Table3D(i, x, y, z) => {
                let x = x.simplify(registry);
                let y = y.simplify(registry);
                let z = z.simplify(registry);
                if let Some(args) = Self::simplify_args([&x, &y, &z].into_iter()) {
                    Self::Constant(
                        args.into_iter()
                            .map(|args| tables.tables_3d[*i].eval(args[0], args[1], args[2]))
                            .collect(),
                    )
                } else {
                    Self::Table3D(*i, x, y, z)
                }
            }
            Self::Table2DXReduce(op, i, x, y) => match (x.simplify(registry), y.simplify(registry))
            {
                (
                    VectorExpression::Reference(ReferenceExpression::Constant(x)),
                    SetExpression::Reference(ReferenceExpression::Constant(y)),
                ) => Self::Constant(Self::x_reduce_table_2d(
                    op,
                    &tables.tables_2d[*i],
                    x.into_iter(),
                    &y,
                )),
                (x, y) => Self::Table2DXReduce(op.clone(), *i, x, y),
            },
            Self::Table2DYReduce(op, i, x, y) => match (x.simplify(registry), y.simplify(registry))
            {
                (
                    SetExpression::Reference(ReferenceExpression::Constant(x)),
                    VectorExpression::Reference(ReferenceExpression::Constant(y)),
                ) => Self::Constant(Self::y_reduce_table_2d(
                    op,
                    &tables.tables_2d[*i],
                    &x,
                    y.into_iter(),
                )),
                (x, y) => Self::Table2DYReduce(op.clone(), *i, x, y),
            },
            Self::Table3DReduce(op, i, x, y, z) => {
                let x = x.simplify(registry);
                let y = y.simplify(registry);
                let z = z.simplify(registry);
                if let Some(args) = Self::simplify_sum_args([&x, &y, &z].into_iter()) {
                    Self::Constant(
                        args.into_iter()
                            .map(|args| {
                                op.eval_iter(args.into_iter().map(|args| {
                                    tables.tables_3d[*i].eval(args[0], args[1], args[2])
                                }))
                                .unwrap()
                            })
                            .collect(),
                    )
                } else {
                    Self::Table3DReduce(op.clone(), *i, x, y, z)
                }
            }
            _ => self.clone(),
        }
    }

    fn eval_args<'a, I, U: StateInterface>(
        args: I,
        state: &U,
        registry: &TableRegistry,
    ) -> Vec<Vec<Element>>
    where
        I: Iterator<Item = &'a VectorOrElementExpression>,
    {
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
                        let f = |i| state.get_vector_variable(i);
                        let vector = vector.eval(state, registry, &f, &registry.vector_tables);
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
        result
    }

    fn simplify_args<'a, I>(args: I) -> Option<Vec<Vec<Element>>>
    where
        I: Iterator<Item = &'a VectorOrElementExpression>,
    {
        let mut simplified_args = vec![vec![]];
        let mut vector_mode = false;
        for arg in args {
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
                _ => return None,
            }
        }
        Some(simplified_args)
    }

    fn eval_sum_args<'a, I, U: StateInterface>(
        args: I,
        state: &U,
        registry: &TableRegistry,
    ) -> Vec<Vec<Vec<Element>>>
    where
        I: Iterator<Item = &'a ArgumentExpression>,
    {
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
                            let f = |i| state.get_set_variable(i);
                            let set = set.eval(state, registry, &f, &registry.set_tables);
                            result
                                .into_iter()
                                .map(|rr| util::expand_vector_with_set(rr, set))
                                .collect()
                        }
                        _ => {
                            let set = &set.eval(state, registry);
                            result
                                .into_iter()
                                .map(|rr| util::expand_vector_with_set(rr, set))
                                .collect()
                        }
                    };
                }
                ArgumentExpression::Vector(vector) => {
                    if vector_mode {
                        match vector {
                            VectorExpression::Reference(vector) => {
                                let f = |i| state.get_vector_variable(i);
                                result
                                    .iter_mut()
                                    .zip(vector.eval(state, registry, &f, &registry.vector_tables))
                                    .for_each(|(rr, v)| rr.iter_mut().for_each(|r| r.push(*v)))
                            }
                            _ => result
                                .iter_mut()
                                .zip(vector.eval(state, registry))
                                .for_each(|(rr, v)| rr.iter_mut().for_each(|r| r.push(v))),
                        }
                    } else {
                        result = match vector {
                            VectorExpression::Reference(vector) => {
                                let f = |i| state.get_vector_variable(i);
                                vector
                                    .eval(state, registry, &f, &registry.vector_tables)
                                    .iter()
                                    .map(|v| {
                                        let mut rr = result[0].clone();
                                        rr.iter_mut().for_each(|r| r.push(*v));
                                        rr
                                    })
                                    .collect()
                            }
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
        result
    }

    fn table_2d<I, J>(table: &Table2D<T>, x: I, y: J) -> Vec<T>
    where
        I: Iterator<Item = Element>,
        J: Iterator<Item = Element>,
    {
        x.zip(y).map(|(x, y)| table.eval(x, y)).collect()
    }

    fn x_reduce_table_2d<I>(op: &ReduceOperator, table: &Table2D<T>, x: I, y: &Set) -> Vec<T>
    where
        I: Iterator<Item = Element>,
    {
        x.map(|x| op.eval_iter(y.ones().map(|y| table.eval(x, y))).unwrap())
            .collect()
    }

    fn y_reduce_table_2d<I>(op: &ReduceOperator, table: &Table2D<T>, x: &Set, y: I) -> Vec<T>
    where
        I: Iterator<Item = Element>,
    {
        y.map(|y| op.eval_iter(x.ones().map(|x| table.eval(x, y))).unwrap())
            .collect()
    }

    fn simplify_sum_args<'a, I>(args: I) -> Option<Vec<Vec<Vec<Element>>>>
    where
        I: Iterator<Item = &'a ArgumentExpression>,
    {
        let mut simplified_args = vec![vec![vec![]]];
        let mut vector_mode = false;
        for arg in args {
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
                _ => return None,
            }
        }
        Some(simplified_args)
    }
}

/// An enum used to construct a numeric vector from a table.
#[derive(Debug, PartialEq, Clone)]
pub enum VectorOrElementExpression {
    Vector(VectorExpression),
    Element(ElementExpression),
}

impl VectorOrElementExpression {
    /// Returns a simplifeid version by precompuation.
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
    use rustc_hash::FxHashMap;

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
                integer_variables: vec![1, 2, 3],
                continuous_variables: vec![1.0, 2.0, 3.0],
            },
            resource_variables: ResourceVariables {
                element_variables: vec![],
                integer_variables: vec![4, 5, 6],
                continuous_variables: vec![4.0, 5.0, 6.0],
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
        let expression = TableVectorExpression::TableReduce(
            ReduceOperator::Sum,
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
        let expression = TableVectorExpression::TableReduce(
            ReduceOperator::Sum,
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
        let expression = TableVectorExpression::Table2DXReduce(
            ReduceOperator::Sum,
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            SetExpression::Reference(ReferenceExpression::Constant(set.clone())),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![30, 90]);
        let expression = TableVectorExpression::Table2DXReduce(
            ReduceOperator::Sum,
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            SetExpression::Complement(Box::new(SetExpression::Reference(
                ReferenceExpression::Constant(set.clone()),
            ))),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![30, 60]);
        let expression = TableVectorExpression::Table2DXReduce(
            ReduceOperator::Sum,
            0,
            VectorExpression::Reverse(Box::new(VectorExpression::Reference(
                ReferenceExpression::Constant(vec![0, 1]),
            ))),
            SetExpression::Reference(ReferenceExpression::Constant(set.clone())),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![90, 30]);
        let expression = TableVectorExpression::Table2DXReduce(
            ReduceOperator::Sum,
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
        let expression = TableVectorExpression::Table2DYReduce(
            ReduceOperator::Sum,
            0,
            SetExpression::Reference(ReferenceExpression::Constant(set.clone())),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![50, 70]);
        let expression = TableVectorExpression::Table2DYReduce(
            ReduceOperator::Sum,
            0,
            SetExpression::Complement(Box::new(SetExpression::Reference(
                ReferenceExpression::Constant(set.clone()),
            ))),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![70, 80]);
        let expression = TableVectorExpression::Table2DYReduce(
            ReduceOperator::Sum,
            0,
            SetExpression::Reference(ReferenceExpression::Constant(set.clone())),
            VectorExpression::Reverse(Box::new(VectorExpression::Reference(
                ReferenceExpression::Constant(vec![0, 1]),
            ))),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![70, 50]);
        let expression = TableVectorExpression::Table2DYReduce(
            ReduceOperator::Sum,
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
    fn vector_table_3d_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let tables = &registry.integer_tables;
        let expression = TableVectorExpression::Table3D(
            0,
            VectorOrElementExpression::Element(ElementExpression::Constant(0)),
            VectorOrElementExpression::Element(ElementExpression::Constant(0)),
            VectorOrElementExpression::Vector(VectorExpression::Reference(
                ReferenceExpression::Constant(vec![0, 2]),
            )),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![10, 30]);
    }

    #[test]
    fn vector_table_3d_sum_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let tables = &registry.integer_tables;
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(1);
        let expression = TableVectorExpression::Table3DReduce(
            ReduceOperator::Sum,
            0,
            ArgumentExpression::Element(ElementExpression::Constant(0)),
            ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Constant(set))),
            ArgumentExpression::Vector(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![0, 2],
            ))),
        );
        assert_eq!(expression.eval(&state, &registry, tables), vec![50, 90]);
    }

    #[test]
    fn vector_table_sum_simplify() {
        let registry = generate_registry();
        let tables = &registry.integer_tables;
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(1);
        let expression = TableVectorExpression::TableReduce(
            ReduceOperator::Sum,
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
        let expression = TableVectorExpression::TableReduce(
            ReduceOperator::Sum,
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
        let expression = TableVectorExpression::Table2DXReduce(
            ReduceOperator::Sum,
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            SetExpression::Reference(ReferenceExpression::Constant(set)),
        );
        assert_eq!(
            expression.simplify(&registry, tables),
            TableVectorExpression::Constant(vec![30, 90])
        );
        let expression = TableVectorExpression::Table2DXReduce(
            ReduceOperator::Sum,
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
        let expression = TableVectorExpression::Table2DYReduce(
            ReduceOperator::Sum,
            0,
            SetExpression::Reference(ReferenceExpression::Constant(set)),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert_eq!(
            expression.simplify(&registry, tables),
            TableVectorExpression::Constant(vec![50, 70])
        );
        let expression = TableVectorExpression::Table2DYReduce(
            ReduceOperator::Sum,
            0,
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert_eq!(expression.simplify(&registry, tables), expression);
    }

    #[test]
    fn vector_table_3d_simplify() {
        let registry = generate_registry();
        let tables = &registry.integer_tables;
        let expression = TableVectorExpression::Table3D(
            0,
            VectorOrElementExpression::Element(ElementExpression::Constant(0)),
            VectorOrElementExpression::Element(ElementExpression::Constant(0)),
            VectorOrElementExpression::Vector(VectorExpression::Reference(
                ReferenceExpression::Constant(vec![0, 2]),
            )),
        );
        assert_eq!(
            expression.simplify(&registry, tables),
            TableVectorExpression::Constant(vec![10, 30])
        );
        let expression = TableVectorExpression::Table3D(
            0,
            VectorOrElementExpression::Element(ElementExpression::Constant(0)),
            VectorOrElementExpression::Element(ElementExpression::Variable(0)),
            VectorOrElementExpression::Vector(VectorExpression::Reference(
                ReferenceExpression::Constant(vec![0, 2]),
            )),
        );
        assert_eq!(expression.simplify(&registry, tables), expression);
    }

    #[test]
    fn vector_table_3d_sum_simplify() {
        let registry = generate_registry();
        let tables = &registry.integer_tables;
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(1);
        let expression = TableVectorExpression::Table3DReduce(
            ReduceOperator::Sum,
            0,
            ArgumentExpression::Element(ElementExpression::Constant(0)),
            ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Constant(
                set.clone(),
            ))),
            ArgumentExpression::Vector(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![0, 2],
            ))),
        );
        assert_eq!(
            expression.simplify(&registry, tables),
            TableVectorExpression::Constant(vec![50, 90])
        );
        let expression = TableVectorExpression::Table3DReduce(
            ReduceOperator::Sum,
            0,
            ArgumentExpression::Element(ElementExpression::Variable(0)),
            ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Constant(set))),
            ArgumentExpression::Vector(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![0, 2],
            ))),
        );
        assert_eq!(expression.simplify(&registry, tables), expression);
    }
}
