use super::argument_expression::ArgumentExpression;
use super::element_expression::ElementExpression;
use super::numeric_operator::ReduceOperator;
use super::reference_expression::ReferenceExpression;
use super::set_expression::SetExpression;
use super::vector_expression::VectorExpression;
use crate::state::DPState;
use crate::table::{Table1D, Table2D};
use crate::table_data::TableData;
use crate::table_registry::TableRegistry;
use crate::variable_type::{Element, Numeric, Set};
use num_traits::Num;
use std::iter::{Product, Sum};

/// Expression referring to a numeric table.
#[derive(Debug, PartialEq, Clone)]
pub enum NumericTableExpression<T: Numeric> {
    /// Constant.
    Constant(T),
    /// Constant in a table.
    Table(usize, Vec<ElementExpression>),
    /// Reduce constants over sets and vectors in a table.
    TableReduce(ReduceOperator, usize, Vec<ArgumentExpression>),
    /// Constant in a 1D table.
    Table1D(usize, ElementExpression),
    /// Constant in a 2D table.
    Table2D(usize, ElementExpression, ElementExpression),
    /// Constant in a 3D table.
    Table3D(
        usize,
        ElementExpression,
        ElementExpression,
        ElementExpression,
    ),
    /// Reduce constants over a set in a 1D table.
    Table1DReduce(ReduceOperator, usize, SetExpression),
    /// Reduce constants over a vector in a 1D table.
    Table1DVectorReduce(ReduceOperator, usize, VectorExpression),
    /// Reduce constants over two sets in a 2D table.
    Table2DReduce(ReduceOperator, usize, SetExpression, SetExpression),
    /// Reduce constants over two vectors in a 2D table.
    Table2DVectorReduce(ReduceOperator, usize, VectorExpression, VectorExpression),
    /// Reduce constants over a set and a vector in a 2D table.
    Table2DSetVectorReduce(ReduceOperator, usize, SetExpression, VectorExpression),
    /// Reduce constants over a vector and a set in a 2D table.
    Table2DVectorSetReduce(ReduceOperator, usize, VectorExpression, SetExpression),
    /// Reduce constants over a set in a 2D table.
    Table2DReduceX(ReduceOperator, usize, SetExpression, ElementExpression),
    /// Reduce constants over a set in a 2D table.
    Table2DReduceY(ReduceOperator, usize, ElementExpression, SetExpression),
    /// Reduce constants over a vector in a 2D table.
    Table2DVectorReduceX(ReduceOperator, usize, VectorExpression, ElementExpression),
    /// Reduce constants over a vector in a 2D table.
    Table2DVectorReduceY(ReduceOperator, usize, ElementExpression, VectorExpression),
    /// Reduce constants over sets and vectors in a 3D table.
    Table3DReduce(
        ReduceOperator,
        usize,
        ArgumentExpression,
        ArgumentExpression,
        ArgumentExpression,
    ),
}

impl<T: Numeric> NumericTableExpression<T> {
    /// Returns the evaluation result.
    ///
    /// # Panics
    ///
    /// if the cost of the transitioned state is used or an empty set or vector is passed to a reduce operation or a min/max reduce operation is performed on an empty set or vector.
    pub fn eval<U: DPState>(
        &self,
        state: &U,
        registry: &TableRegistry,
        tables: &TableData<T>,
    ) -> T {
        let set_f = |i| state.get_set_variable(i);
        let set_tables = &registry.set_tables;
        let vector_f = |i| state.get_vector_variable(i);
        let vector_tables = &registry.vector_tables;
        match self {
            Self::Constant(value) => *value,
            Self::Table(i, args) => {
                let args: Vec<Element> = args.iter().map(|x| x.eval(state, registry)).collect();
                tables.tables[*i].eval(&args)
            }
            Self::TableReduce(op, i, args) => {
                let args = ArgumentExpression::eval_args(args.iter(), state, registry);
                op.eval_iter(args.into_iter().map(|args| tables.tables[*i].eval(&args)))
                    .unwrap()
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
            Self::Table1DReduce(op, i, SetExpression::Reference(x)) => Self::reduce_table_1d(
                op,
                &tables.tables_1d[*i],
                x.eval(state, registry, &set_f, set_tables).ones(),
            ),
            Self::Table1DReduce(op, i, x) => {
                Self::reduce_table_1d(op, &tables.tables_1d[*i], x.eval(state, registry).ones())
            }
            Self::Table1DVectorReduce(op, i, VectorExpression::Reference(x)) => {
                Self::reduce_table_1d(
                    op,
                    &tables.tables_1d[*i],
                    x.eval(state, registry, &vector_f, vector_tables)
                        .iter()
                        .copied(),
                )
            }
            Self::Table1DVectorReduce(op, i, x) => Self::reduce_table_1d(
                op,
                &tables.tables_1d[*i],
                x.eval(state, registry).into_iter(),
            ),
            Self::Table2DReduce(
                op,
                i,
                SetExpression::Reference(x),
                SetExpression::Reference(y),
            ) => {
                let x = x.eval(state, registry, &set_f, set_tables).ones();
                let y = y.eval(state, registry, &set_f, set_tables);
                Self::reduce_table_2d_set_y(op, &tables.tables_2d[*i], x, y)
            }
            Self::Table2DReduce(op, i, SetExpression::Reference(x), y) => {
                let x = x.eval(state, registry, &set_f, set_tables).ones();
                let y = y.eval(state, registry);
                Self::reduce_table_2d_set_y(op, &tables.tables_2d[*i], x, &y)
            }
            Self::Table2DReduce(op, i, x, SetExpression::Reference(y)) => {
                let y = y.eval(state, registry, &set_f, set_tables);
                Self::reduce_table_2d_set_y(
                    op,
                    &tables.tables_2d[*i],
                    x.eval(state, registry).ones(),
                    y,
                )
            }
            Self::Table2DReduce(op, i, x, y) => {
                let y = y.eval(state, registry);
                Self::reduce_table_2d_set_y(
                    op,
                    &tables.tables_2d[*i],
                    x.eval(state, registry).ones(),
                    &y,
                )
            }
            Self::Table2DVectorReduce(
                op,
                i,
                VectorExpression::Reference(x),
                VectorExpression::Reference(y),
            ) => {
                let x = x
                    .eval(state, registry, &vector_f, vector_tables)
                    .iter()
                    .copied();
                let y = y
                    .eval(state, registry, &vector_f, vector_tables)
                    .iter()
                    .copied();
                Self::reduce_table_2d(op, &tables.tables_2d[*i], x, y)
            }
            Self::Table2DVectorReduce(op, i, VectorExpression::Reference(x), y) => {
                let x = x
                    .eval(state, registry, &vector_f, vector_tables)
                    .iter()
                    .copied();
                let y = y.eval(state, registry).into_iter();
                Self::reduce_table_2d(op, &tables.tables_2d[*i], x, y)
            }
            Self::Table2DVectorReduce(op, i, x, VectorExpression::Reference(y)) => {
                let x = x.eval(state, registry).into_iter();
                let y = y
                    .eval(state, registry, &vector_f, vector_tables)
                    .iter()
                    .copied();
                Self::reduce_table_2d(op, &tables.tables_2d[*i], x, y)
            }
            Self::Table2DVectorReduce(op, i, x, y) => {
                let x = x.eval(state, registry).into_iter();
                let y = y.eval(state, registry).into_iter();
                Self::reduce_table_2d(op, &tables.tables_2d[*i], x, y)
            }
            Self::Table2DSetVectorReduce(
                op,
                i,
                SetExpression::Reference(x),
                VectorExpression::Reference(y),
            ) => {
                let x = x.eval(state, registry, &set_f, set_tables).ones();
                let y = y
                    .eval(state, registry, &vector_f, vector_tables)
                    .iter()
                    .copied();
                Self::reduce_table_2d(op, &tables.tables_2d[*i], x, y)
            }
            Self::Table2DSetVectorReduce(op, i, SetExpression::Reference(x), y) => {
                let y = y.eval(state, registry).into_iter();
                Self::reduce_table_2d(
                    op,
                    &tables.tables_2d[*i],
                    x.eval(state, registry, &set_f, set_tables).ones(),
                    y,
                )
            }
            Self::Table2DSetVectorReduce(op, i, x, VectorExpression::Reference(y)) => {
                let y = y
                    .eval(state, registry, &vector_f, vector_tables)
                    .iter()
                    .copied();
                Self::reduce_table_2d(op, &tables.tables_2d[*i], x.eval(state, registry).ones(), y)
            }
            Self::Table2DSetVectorReduce(op, i, x, y) => {
                let y = y.eval(state, registry).into_iter();
                Self::reduce_table_2d(op, &tables.tables_2d[*i], x.eval(state, registry).ones(), y)
            }
            Self::Table2DVectorSetReduce(
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
                Self::reduce_table_2d_set_y(op, &tables.tables_2d[*i], x, y)
            }
            Self::Table2DVectorSetReduce(op, i, x, SetExpression::Reference(y)) => {
                let x = x.eval(state, registry).into_iter();
                let y = y.eval(state, registry, &set_f, set_tables);
                Self::reduce_table_2d_set_y(op, &tables.tables_2d[*i], x, y)
            }
            Self::Table2DVectorSetReduce(op, i, VectorExpression::Reference(x), y) => {
                let x = x
                    .eval(state, registry, &vector_f, vector_tables)
                    .iter()
                    .copied();
                let y = y.eval(state, registry);
                Self::reduce_table_2d_set_y(op, &tables.tables_2d[*i], x, &y)
            }
            Self::Table2DVectorSetReduce(op, i, x, y) => {
                let x = x.eval(state, registry).into_iter();
                let y = y.eval(state, registry);
                Self::reduce_table_2d_set_y(op, &tables.tables_2d[*i], x, &y)
            }
            Self::Table2DReduceX(op, i, SetExpression::Reference(x), y) => {
                let x = x.eval(state, registry, &set_f, set_tables).ones();
                let y = y.eval(state, registry);
                Self::reduce_table_2d_x(op, &tables.tables_2d[*i], x, y)
            }
            Self::Table2DReduceX(op, i, x, y) => {
                let y = y.eval(state, registry);
                Self::reduce_table_2d_x(
                    op,
                    &tables.tables_2d[*i],
                    x.eval(state, registry).ones(),
                    y,
                )
            }
            Self::Table2DReduceY(op, i, x, SetExpression::Reference(y)) => {
                let x = x.eval(state, registry);
                let y = y.eval(state, registry, &set_f, set_tables).ones();
                Self::reduce_table_2d_y(op, &tables.tables_2d[*i], x, y)
            }
            Self::Table2DReduceY(op, i, x, y) => {
                let x = x.eval(state, registry);
                Self::reduce_table_2d_y(
                    op,
                    &tables.tables_2d[*i],
                    x,
                    y.eval(state, registry).ones(),
                )
            }
            Self::Table2DVectorReduceX(op, i, VectorExpression::Reference(x), y) => {
                let x = x
                    .eval(state, registry, &vector_f, vector_tables)
                    .iter()
                    .copied();
                let y = y.eval(state, registry);
                Self::reduce_table_2d_x(op, &tables.tables_2d[*i], x, y)
            }
            Self::Table2DVectorReduceX(op, i, x, y) => {
                let x = x.eval(state, registry).into_iter();
                let y = y.eval(state, registry);
                Self::reduce_table_2d_x(op, &tables.tables_2d[*i], x, y)
            }
            Self::Table2DVectorReduceY(op, i, x, VectorExpression::Reference(y)) => {
                let x = x.eval(state, registry);
                let y = y
                    .eval(state, registry, &vector_f, vector_tables)
                    .iter()
                    .copied();
                Self::reduce_table_2d_y(op, &tables.tables_2d[*i], x, y)
            }
            Self::Table2DVectorReduceY(op, i, x, y) => {
                let x = x.eval(state, registry);
                let y = y.eval(state, registry).into_iter();
                Self::reduce_table_2d_y(op, &tables.tables_2d[*i], x, y)
            }
            Self::Table3DReduce(op, i, x, y, z) => {
                let args = ArgumentExpression::eval_args([x, y, z].into_iter(), state, registry);
                op.eval_iter(
                    args.into_iter()
                        .map(|args| tables.tables_3d[*i].eval(args[0], args[1], args[2])),
                )
                .unwrap()
            }
        }
    }

    /// Returns a simplified version by precomputation.
    ///
    /// # Panics
    ///
    /// if a min/max reduce operation is performed on an empty set or vector.
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
            Self::TableReduce(op, i, args) => {
                let args: Vec<ArgumentExpression> =
                    args.iter().map(|x| x.simplify(registry)).collect();
                if let Some(args) = ArgumentExpression::simplify_args(args.iter()) {
                    Self::Constant(
                        op.eval_iter(args.into_iter().map(|args| tables.tables[*i].eval(&args)))
                            .unwrap(),
                    )
                } else {
                    Self::TableReduce(op.clone(), *i, args)
                }
            }
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
            Self::Table3D(i, x, y, z) => match (
                x.simplify(registry),
                y.simplify(registry),
                z.simplify(registry),
            ) {
                (
                    ElementExpression::Constant(x),
                    ElementExpression::Constant(y),
                    ElementExpression::Constant(z),
                ) => Self::Constant(tables.tables_3d[*i].eval(x, y, z)),
                (x, y, z) => Self::Table3D(*i, x, y, z),
            },
            Self::Table1DReduce(op, i, x) => match x.simplify(registry) {
                SetExpression::Reference(ReferenceExpression::Constant(x)) => {
                    Self::Constant(Self::reduce_table_1d(op, &tables.tables_1d[*i], x.ones()))
                }
                x => Self::Table1DReduce(op.clone(), *i, x),
            },
            Self::Table1DVectorReduce(op, i, x) => match x.simplify(registry) {
                VectorExpression::Reference(ReferenceExpression::Constant(x)) => Self::Constant(
                    Self::reduce_table_1d(op, &tables.tables_1d[*i], x.into_iter()),
                ),
                x => Self::Table1DVectorReduce(op.clone(), *i, x),
            },
            Self::Table2DReduce(op, i, x, y) => {
                match (x.simplify(registry), y.simplify(registry)) {
                    (
                        SetExpression::Reference(ReferenceExpression::Constant(x)),
                        SetExpression::Reference(ReferenceExpression::Constant(y)),
                    ) => Self::Constant(Self::reduce_table_2d_set_y(
                        op,
                        &tables.tables_2d[*i],
                        x.ones(),
                        &y,
                    )),
                    (x, y) => Self::Table2DReduce(op.clone(), *i, x, y),
                }
            }
            Self::Table2DVectorReduce(op, i, x, y) => {
                match (x.simplify(registry), y.simplify(registry)) {
                    (
                        VectorExpression::Reference(ReferenceExpression::Constant(x)),
                        VectorExpression::Reference(ReferenceExpression::Constant(y)),
                    ) => Self::Constant(Self::reduce_table_2d(
                        op,
                        &tables.tables_2d[*i],
                        x.into_iter(),
                        y.into_iter(),
                    )),
                    (x, y) => Self::Table2DVectorReduce(op.clone(), *i, x, y),
                }
            }
            Self::Table2DSetVectorReduce(op, i, x, y) => {
                match (x.simplify(registry), y.simplify(registry)) {
                    (
                        SetExpression::Reference(ReferenceExpression::Constant(x)),
                        VectorExpression::Reference(ReferenceExpression::Constant(y)),
                    ) => Self::Constant(Self::reduce_table_2d(
                        op,
                        &tables.tables_2d[*i],
                        x.ones(),
                        y.into_iter(),
                    )),
                    (x, y) => Self::Table2DSetVectorReduce(op.clone(), *i, x, y),
                }
            }
            Self::Table2DVectorSetReduce(op, i, x, y) => {
                match (x.simplify(registry), y.simplify(registry)) {
                    (
                        VectorExpression::Reference(ReferenceExpression::Constant(x)),
                        SetExpression::Reference(ReferenceExpression::Constant(y)),
                    ) => Self::Constant(Self::reduce_table_2d_set_y(
                        op,
                        &tables.tables_2d[*i],
                        x.into_iter(),
                        &y,
                    )),
                    (x, y) => Self::Table2DVectorSetReduce(op.clone(), *i, x, y),
                }
            }
            Self::Table2DReduceX(op, i, x, y) => match (x.simplify(registry), y.simplify(registry))
            {
                (
                    SetExpression::Reference(ReferenceExpression::Constant(x)),
                    ElementExpression::Constant(y),
                ) => Self::Constant(Self::reduce_table_2d_x(
                    op,
                    &tables.tables_2d[*i],
                    x.ones(),
                    y,
                )),
                (x, y) => Self::Table2DReduceX(op.clone(), *i, x, y),
            },
            Self::Table2DReduceY(op, i, x, y) => match (x.simplify(registry), y.simplify(registry))
            {
                (
                    ElementExpression::Constant(x),
                    SetExpression::Reference(ReferenceExpression::Constant(y)),
                ) => Self::Constant(Self::reduce_table_2d_y(
                    op,
                    &tables.tables_2d[*i],
                    x,
                    y.ones(),
                )),
                (x, y) => Self::Table2DReduceY(op.clone(), *i, x, y),
            },
            Self::Table2DVectorReduceX(op, i, x, y) => {
                match (x.simplify(registry), y.simplify(registry)) {
                    (
                        VectorExpression::Reference(ReferenceExpression::Constant(x)),
                        ElementExpression::Constant(y),
                    ) => Self::Constant(Self::reduce_table_2d_x(
                        op,
                        &tables.tables_2d[*i],
                        x.into_iter(),
                        y,
                    )),
                    (x, y) => Self::Table2DVectorReduceX(op.clone(), *i, x, y),
                }
            }
            Self::Table2DVectorReduceY(op, i, x, y) => {
                match (x.simplify(registry), y.simplify(registry)) {
                    (
                        ElementExpression::Constant(x),
                        VectorExpression::Reference(ReferenceExpression::Constant(y)),
                    ) => Self::Constant(Self::reduce_table_2d_y(
                        op,
                        &tables.tables_2d[*i],
                        x,
                        y.into_iter(),
                    )),
                    (x, y) => Self::Table2DVectorReduceY(op.clone(), *i, x, y),
                }
            }
            Self::Table3DReduce(op, i, x, y, z) => {
                let x = x.simplify(registry);
                let y = y.simplify(registry);
                let z = z.simplify(registry);
                if let Some(args) = ArgumentExpression::simplify_args([&x, &y, &z].into_iter()) {
                    Self::Constant(
                        op.eval_iter(
                            args.into_iter()
                                .map(|args| tables.tables_3d[*i].eval(args[0], args[1], args[2])),
                        )
                        .unwrap(),
                    )
                } else {
                    Self::Table3DReduce(op.clone(), *i, x, y, z)
                }
            }
            _ => self.clone(),
        }
    }

    fn reduce_table_1d<I>(op: &ReduceOperator, table: &Table1D<T>, x: I) -> T
    where
        T: Num + PartialOrd + Sum + Product,
        I: Iterator<Item = Element>,
    {
        op.eval_iter(x.map(|x| table.eval(x))).unwrap()
    }

    fn reduce_table_2d<I, J>(op: &ReduceOperator, table: &Table2D<T>, x: I, y: J) -> T
    where
        T: Num + PartialOrd + Sum + Product,
        I: Iterator<Item = Element>,
        J: Iterator<Item = Element> + Clone,
    {
        op.eval_iter(x.map(|x| op.eval_iter(y.clone().map(|y| table.eval(x, y))).unwrap()))
            .unwrap()
    }

    fn reduce_table_2d_set_y<I>(op: &ReduceOperator, table: &Table2D<T>, x: I, y: &Set) -> T
    where
        T: Num + PartialOrd + Sum + Product,
        I: Iterator<Item = Element>,
    {
        op.eval_iter(x.map(|x| op.eval_iter(y.ones().map(|y| table.eval(x, y))).unwrap()))
            .unwrap()
    }

    fn reduce_table_2d_x<I>(op: &ReduceOperator, table: &Table2D<T>, x: I, y: Element) -> T
    where
        T: Num + PartialOrd + Sum + Product,
        I: Iterator<Item = Element>,
    {
        op.eval_iter(x.map(|x| table.eval(x, y))).unwrap()
    }

    fn reduce_table_2d_y<I>(op: &ReduceOperator, table: &Table2D<T>, x: Element, y: I) -> T
    where
        T: Num + PartialOrd + Sum + Product,
        I: Iterator<Item = Element>,
    {
        op.eval_iter(y.map(|y| table.eval(x, y))).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::*;
    use crate::table;
    use rustc_hash::FxHashMap;

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
            signature_variables: SignatureVariables {
                set_variables: vec![set1, set2, Set::with_capacity(3), Set::with_capacity(3)],
                vector_variables: vec![vec![0, 2], vec![0, 1], vec![], vec![]],
                ..Default::default()
            },
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
        let expression = NumericTableExpression::Table1DReduce(
            ReduceOperator::Sum,
            0,
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            40
        );
        let expression = NumericTableExpression::Table1DReduce(
            ReduceOperator::Sum,
            0,
            SetExpression::Reference(ReferenceExpression::Variable(1)),
        );
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            30
        );
        let expression = NumericTableExpression::Table1DReduce(
            ReduceOperator::Sum,
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
        let expression = NumericTableExpression::Table1DVectorReduce(
            ReduceOperator::Sum,
            0,
            VectorExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            40
        );
        let expression = NumericTableExpression::Table1DVectorReduce(
            ReduceOperator::Sum,
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

        let expression = NumericTableExpression::Table2DReduce(
            ReduceOperator::Sum,
            0,
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Reference(ReferenceExpression::Variable(1)),
        );
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            180
        );

        let expression = NumericTableExpression::Table2DReduce(
            ReduceOperator::Sum,
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

        let expression = NumericTableExpression::Table2DReduce(
            ReduceOperator::Sum,
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

        let expression = NumericTableExpression::Table2DReduce(
            ReduceOperator::Sum,
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

        let expression = NumericTableExpression::Table2DVectorReduce(
            ReduceOperator::Sum,
            0,
            VectorExpression::Reference(ReferenceExpression::Variable(0)),
            VectorExpression::Reference(ReferenceExpression::Variable(1)),
        );
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            180
        );

        let expression = NumericTableExpression::Table2DVectorReduce(
            ReduceOperator::Sum,
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

        let expression = NumericTableExpression::Table2DVectorReduce(
            ReduceOperator::Sum,
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

        let expression = NumericTableExpression::Table2DVectorReduce(
            ReduceOperator::Sum,
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

        let expression = NumericTableExpression::Table2DSetVectorReduce(
            ReduceOperator::Sum,
            0,
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            VectorExpression::Reference(ReferenceExpression::Variable(1)),
        );
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            180
        );

        let expression = NumericTableExpression::Table2DSetVectorReduce(
            ReduceOperator::Sum,
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

        let expression = NumericTableExpression::Table2DSetVectorReduce(
            ReduceOperator::Sum,
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

        let expression = NumericTableExpression::Table2DSetVectorReduce(
            ReduceOperator::Sum,
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

        let expression = NumericTableExpression::Table2DVectorSetReduce(
            ReduceOperator::Sum,
            0,
            VectorExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Reference(ReferenceExpression::Variable(1)),
        );
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            180
        );

        let expression = NumericTableExpression::Table2DVectorSetReduce(
            ReduceOperator::Sum,
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

        let expression = NumericTableExpression::Table2DVectorSetReduce(
            ReduceOperator::Sum,
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

        let expression = NumericTableExpression::Table2DVectorSetReduce(
            ReduceOperator::Sum,
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

        let expression = NumericTableExpression::Table2DReduceX(
            ReduceOperator::Sum,
            0,
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            ElementExpression::Constant(0),
        );
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            80
        );

        let expression = NumericTableExpression::Table2DReduceX(
            ReduceOperator::Sum,
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

        let expression = NumericTableExpression::Table2DVectorReduceX(
            ReduceOperator::Sum,
            0,
            VectorExpression::Reference(ReferenceExpression::Variable(0)),
            ElementExpression::Constant(0),
        );
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            80
        );

        let expression = NumericTableExpression::Table2DVectorReduceX(
            ReduceOperator::Sum,
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

        let expression = NumericTableExpression::Table2DReduceY(
            ReduceOperator::Sum,
            0,
            ElementExpression::Constant(0),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            40
        );

        let expression = NumericTableExpression::Table2DReduceY(
            ReduceOperator::Sum,
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

        let expression = NumericTableExpression::Table2DVectorReduceY(
            ReduceOperator::Sum,
            0,
            ElementExpression::Constant(0),
            VectorExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            40
        );

        let expression = NumericTableExpression::Table2DVectorReduceY(
            ReduceOperator::Sum,
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
    fn table_3d_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let expression = NumericTableExpression::Table3D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
        );
        assert_eq!(
            expression.eval(&state, &registry, &registry.integer_tables),
            10
        );
    }

    #[test]
    fn table_3d_sum_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let expression = NumericTableExpression::Table3DReduce(
            ReduceOperator::Sum,
            0,
            ArgumentExpression::Element(ElementExpression::Constant(0)),
            ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Variable(0))),
            ArgumentExpression::Vector(VectorExpression::Reference(ReferenceExpression::Variable(
                1,
            ))),
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
        let expression = NumericTableExpression::TableReduce(
            ReduceOperator::Sum,
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
        let expression = NumericTableExpression::Table1DReduce(
            ReduceOperator::Sum,
            0,
            SetExpression::Reference(ReferenceExpression::Constant(set)),
        );
        assert_eq!(
            expression.simplify(&registry, &registry.integer_tables),
            NumericTableExpression::Constant(30)
        );

        let expression = NumericTableExpression::Table1DReduce(
            ReduceOperator::Sum,
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

        let expression = NumericTableExpression::Table1DVectorReduce(
            ReduceOperator::Sum,
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert_eq!(
            expression.simplify(&registry, &registry.integer_tables),
            NumericTableExpression::Constant(30)
        );

        let expression = NumericTableExpression::Table1DVectorReduce(
            ReduceOperator::Sum,
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
        let expression = NumericTableExpression::Table2DReduce(
            ReduceOperator::Sum,
            0,
            SetExpression::Reference(ReferenceExpression::Constant(set.clone())),
            SetExpression::Reference(ReferenceExpression::Constant(set)),
        );
        assert_eq!(
            expression.simplify(&registry, &registry.integer_tables),
            NumericTableExpression::Constant(120)
        );

        let expression = NumericTableExpression::Table2DReduce(
            ReduceOperator::Sum,
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

        let expression = NumericTableExpression::Table2DVectorReduce(
            ReduceOperator::Sum,
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert_eq!(
            expression.simplify(&registry, &registry.integer_tables),
            NumericTableExpression::Constant(120)
        );

        let expression = NumericTableExpression::Table2DVectorReduce(
            ReduceOperator::Sum,
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
        let expression = NumericTableExpression::Table2DSetVectorReduce(
            ReduceOperator::Sum,
            0,
            SetExpression::Reference(ReferenceExpression::Constant(set)),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert_eq!(
            expression.simplify(&registry, &registry.integer_tables),
            NumericTableExpression::Constant(120)
        );

        let expression = NumericTableExpression::Table2DSetVectorReduce(
            ReduceOperator::Sum,
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
        let expression = NumericTableExpression::Table2DVectorSetReduce(
            ReduceOperator::Sum,
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            SetExpression::Reference(ReferenceExpression::Constant(set)),
        );
        assert_eq!(
            expression.simplify(&registry, &registry.integer_tables),
            NumericTableExpression::Constant(120)
        );

        let expression = NumericTableExpression::Table2DVectorSetReduce(
            ReduceOperator::Sum,
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
        let expression = NumericTableExpression::Table2DReduceX(
            ReduceOperator::Sum,
            0,
            SetExpression::Reference(ReferenceExpression::Constant(set)),
            ElementExpression::Constant(0),
        );
        assert_eq!(
            expression.simplify(&registry, &registry.integer_tables),
            NumericTableExpression::Constant(50)
        );

        let expression = NumericTableExpression::Table2DReduceX(
            ReduceOperator::Sum,
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
        let expression = NumericTableExpression::Table2DReduceY(
            ReduceOperator::Sum,
            0,
            ElementExpression::Constant(0),
            SetExpression::Reference(ReferenceExpression::Constant(set)),
        );
        assert_eq!(
            expression.simplify(&registry, &registry.integer_tables),
            NumericTableExpression::Constant(30)
        );

        let expression = NumericTableExpression::Table2DReduceY(
            ReduceOperator::Sum,
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

        let expression = NumericTableExpression::Table2DVectorReduceX(
            ReduceOperator::Sum,
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            ElementExpression::Constant(0),
        );
        assert_eq!(
            expression.simplify(&registry, &registry.integer_tables),
            NumericTableExpression::Constant(50)
        );

        let expression = NumericTableExpression::Table2DVectorReduceX(
            ReduceOperator::Sum,
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

        let expression = NumericTableExpression::Table2DVectorReduceY(
            ReduceOperator::Sum,
            0,
            ElementExpression::Constant(0),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert_eq!(
            expression.simplify(&registry, &registry.integer_tables),
            NumericTableExpression::Constant(30)
        );

        let expression = NumericTableExpression::Table2DVectorReduceY(
            ReduceOperator::Sum,
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

        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(2);
        let expression = NumericTableExpression::Table3DReduce(
            ReduceOperator::Sum,
            0,
            ArgumentExpression::Element(ElementExpression::Constant(0)),
            ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Constant(set))),
            ArgumentExpression::Vector(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![0, 1],
            ))),
        );
        assert_eq!(
            expression.simplify(&registry, &registry.integer_tables),
            NumericTableExpression::Constant(180)
        );

        let expression = NumericTableExpression::Table3DReduce(
            ReduceOperator::Sum,
            0,
            ArgumentExpression::Element(ElementExpression::Constant(0)),
            ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Variable(0))),
            ArgumentExpression::Vector(VectorExpression::Reference(ReferenceExpression::Variable(
                1,
            ))),
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
        set.insert(2);
        let expression = NumericTableExpression::TableReduce(
            ReduceOperator::Sum,
            0,
            vec![
                ArgumentExpression::Element(ElementExpression::Constant(0)),
                ArgumentExpression::Element(ElementExpression::Constant(1)),
                ArgumentExpression::Set(SetExpression::Complement(Box::new(
                    SetExpression::Complement(Box::new(SetExpression::Reference(
                        ReferenceExpression::Constant(set),
                    ))),
                ))),
                ArgumentExpression::Vector(VectorExpression::Reverse(Box::new(
                    VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
                ))),
            ],
        );
        assert_eq!(
            expression.simplify(&registry, &registry.integer_tables),
            NumericTableExpression::Constant(1000)
        );

        let expression = NumericTableExpression::TableReduce(
            ReduceOperator::Sum,
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
