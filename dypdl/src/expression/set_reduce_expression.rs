use super::argument_expression::ArgumentExpression;
use super::element_expression::ElementExpression;
use super::reference_expression::ReferenceExpression;
use super::set_expression::SetExpression;
use super::vector_expression::VectorExpression;
use crate::state::StateInterface;
use crate::state_functions::{StateFunctionCache, StateFunctions};
use crate::table::{Table1D, Table2D};
use crate::table_registry::TableRegistry;
use crate::variable_type::{Element, Set};

/// Operator performing a reduce operation on an iterator of sets.
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum SetReduceOperator {
    /// Union.
    Union,
    /// Intersection.
    Intersection,
    /// Symmetric difference (disjunctive union).
    SymmetricDifference,
}

impl SetReduceOperator {
    /// Returns the evaluation result.
    ///
    /// The maximum number of elements (capacity) must be given.
    pub fn eval<'a, I: Iterator<Item = &'a Set>>(&self, mut iter: I, capacity: usize) -> Set {
        if let Some(first) = iter.next() {
            match self {
                Self::Union => iter.fold(first.clone(), |mut x, y| {
                    x.union_with(y);
                    x
                }),
                Self::Intersection => iter.fold(first.clone(), |mut x, y| {
                    x.intersect_with(y);
                    x
                }),
                Self::SymmetricDifference => iter.fold(first.clone(), |mut x, y| {
                    x.symmetric_difference_with(y);
                    x
                }),
            }
        } else {
            Set::with_capacity(capacity)
        }
    }
}

/// Expression performing a reduce operation on a tables of sets.
#[derive(Debug, PartialEq, Clone)]
pub enum SetReduceExpression {
    /// Constant.
    Constant(Set),
    /// Reduce operation over a 1D table.
    Table1D(SetReduceOperator, usize, usize, Box<ArgumentExpression>),
    /// Reduce operation over a 2D table.
    Table2D(
        SetReduceOperator,
        usize,
        usize,
        Box<ArgumentExpression>,
        Box<ArgumentExpression>,
    ),
    /// Reduce operation over a 3D table.
    Table3D(
        SetReduceOperator,
        usize,
        usize,
        Box<ArgumentExpression>,
        Box<ArgumentExpression>,
        Box<ArgumentExpression>,
    ),
    /// Reduce operation over a table.
    Table(SetReduceOperator, usize, usize, Vec<ArgumentExpression>),
}

impl SetReduceExpression {
    /// Returns the evaluation result.
    ///
    /// # Panics
    ///
    /// Panics if the cost of the transitioned state is used or an empty set or vector is passed to a reduce operation or a min/max reduce operation is performed on an empty set or vector.
    pub fn eval<U: StateInterface>(
        &self,
        state: &U,
        function_cache: &mut StateFunctionCache,
        state_functions: &StateFunctions,
        registry: &TableRegistry,
    ) -> Set {
        let set_tables = &registry.set_tables;

        match self {
            Self::Constant(set) => set.clone(),
            Self::Table1D(op, capacity, i, x) => match x.as_ref() {
                ArgumentExpression::Element(x) => {
                    let x = x.eval(state, function_cache, state_functions, registry);
                    set_tables.tables_1d[*i].get(x).clone()
                }
                ArgumentExpression::Set(SetExpression::Reference(x)) => {
                    let x = x.eval(state, function_cache, state_functions, registry);
                    Self::reduce_table_1d(op, *capacity, &set_tables.tables_1d[*i], x.ones())
                }
                ArgumentExpression::Set(SetExpression::StateFunction(x)) => {
                    let x = function_cache.get_set_value(*x, state, state_functions, registry);
                    Self::reduce_table_1d(op, *capacity, &set_tables.tables_1d[*i], x.ones())
                }
                ArgumentExpression::Set(x) => {
                    let x = x.eval(state, function_cache, state_functions, registry);
                    Self::reduce_table_1d(op, *capacity, &set_tables.tables_1d[*i], x.ones())
                }
                ArgumentExpression::Vector(VectorExpression::Reference(x)) => {
                    let x = x
                        .eval(state, function_cache, state_functions, registry)
                        .iter()
                        .copied();
                    Self::reduce_table_1d(op, *capacity, &set_tables.tables_1d[*i], x)
                }
                ArgumentExpression::Vector(x) => {
                    let x = x
                        .eval(state, function_cache, state_functions, registry)
                        .into_iter();
                    Self::reduce_table_1d(op, *capacity, &set_tables.tables_1d[*i], x)
                }
            },
            Self::Table2D(op, capacity, i, x, y) => match (x.as_ref(), y.as_ref()) {
                (ArgumentExpression::Element(x), ArgumentExpression::Element(y)) => {
                    let x = x.eval(state, function_cache, state_functions, registry);
                    let y = y.eval(state, function_cache, state_functions, registry);
                    set_tables.tables_2d[*i].get(x, y).clone()
                }
                (
                    ArgumentExpression::Element(x),
                    ArgumentExpression::Set(SetExpression::Reference(y)),
                ) => {
                    let x = x.eval(state, function_cache, state_functions, registry);
                    let y = y.eval(state, function_cache, state_functions, registry);
                    Self::reduce_table_2d_y(op, *capacity, &set_tables.tables_2d[*i], x, y.ones())
                }
                (
                    ArgumentExpression::Element(x),
                    ArgumentExpression::Set(SetExpression::StateFunction(y)),
                ) => {
                    let x = x.eval(state, function_cache, state_functions, registry);
                    let y = function_cache.get_set_value(*y, state, state_functions, registry);
                    Self::reduce_table_2d_y(op, *capacity, &set_tables.tables_2d[*i], x, y.ones())
                }
                (ArgumentExpression::Element(x), ArgumentExpression::Set(y)) => {
                    let x = x.eval(state, function_cache, state_functions, registry);
                    let y = y.eval(state, function_cache, state_functions, registry);
                    Self::reduce_table_2d_y(op, *capacity, &set_tables.tables_2d[*i], x, y.ones())
                }
                (
                    ArgumentExpression::Element(x),
                    ArgumentExpression::Vector(VectorExpression::Reference(y)),
                ) => {
                    let x = x.eval(state, function_cache, state_functions, registry);
                    let y = y
                        .eval(state, function_cache, state_functions, registry)
                        .iter()
                        .copied();
                    Self::reduce_table_2d_y(op, *capacity, &set_tables.tables_2d[*i], x, y)
                }
                (ArgumentExpression::Element(x), ArgumentExpression::Vector(y)) => {
                    let x = x.eval(state, function_cache, state_functions, registry);
                    let y = y
                        .eval(state, function_cache, state_functions, registry)
                        .into_iter();
                    Self::reduce_table_2d_y(op, *capacity, &set_tables.tables_2d[*i], x, y)
                }
                (
                    ArgumentExpression::Set(SetExpression::Reference(x)),
                    ArgumentExpression::Element(y),
                ) => {
                    let y = y.eval(state, function_cache, state_functions, registry);
                    let x = x.eval(state, function_cache, state_functions, registry);
                    Self::reduce_table_2d_x(op, *capacity, &set_tables.tables_2d[*i], x.ones(), y)
                }
                (
                    ArgumentExpression::Set(SetExpression::StateFunction(x)),
                    ArgumentExpression::Element(y),
                ) => {
                    let y = y.eval(state, function_cache, state_functions, registry);
                    let x = function_cache.get_set_value(*x, state, state_functions, registry);
                    Self::reduce_table_2d_x(op, *capacity, &set_tables.tables_2d[*i], x.ones(), y)
                }
                (ArgumentExpression::Set(x), ArgumentExpression::Element(y)) => {
                    let x = x.eval(state, function_cache, state_functions, registry);
                    let y = y.eval(state, function_cache, state_functions, registry);
                    Self::reduce_table_2d_x(op, *capacity, &set_tables.tables_2d[*i], x.ones(), y)
                }
                (
                    ArgumentExpression::Vector(VectorExpression::Reference(x)),
                    ArgumentExpression::Element(y),
                ) => {
                    let x = x
                        .eval(state, function_cache, state_functions, registry)
                        .iter()
                        .copied();
                    let y = y.eval(state, function_cache, state_functions, registry);
                    Self::reduce_table_2d_x(op, *capacity, &set_tables.tables_2d[*i], x, y)
                }
                (ArgumentExpression::Vector(x), ArgumentExpression::Element(y)) => {
                    let x = x
                        .eval(state, function_cache, state_functions, registry)
                        .into_iter();
                    let y = y.eval(state, function_cache, state_functions, registry);
                    Self::reduce_table_2d_x(op, *capacity, &set_tables.tables_2d[*i], x, y)
                }
                (
                    ArgumentExpression::Set(SetExpression::StateFunction(x)),
                    ArgumentExpression::Set(SetExpression::StateFunction(y)),
                ) => {
                    let (x, y) =
                        function_cache.get_set_value_pair(*x, *y, state, state_functions, registry);
                    Self::reduce_table_2d_set_y(
                        op,
                        *capacity,
                        &set_tables.tables_2d[*i],
                        x.ones(),
                        y,
                    )
                }
                (
                    ArgumentExpression::Set(SetExpression::StateFunction(x)),
                    ArgumentExpression::Set(SetExpression::Reference(y)),
                ) => {
                    let y = y.eval(state, function_cache, state_functions, registry);
                    let x = function_cache.get_set_value(*x, state, state_functions, registry);
                    Self::reduce_table_2d_set_y(
                        op,
                        *capacity,
                        &set_tables.tables_2d[*i],
                        x.ones(),
                        y,
                    )
                }
                (
                    ArgumentExpression::Set(SetExpression::Reference(x)),
                    ArgumentExpression::Set(SetExpression::StateFunction(y)),
                ) => {
                    let x = x.eval(state, function_cache, state_functions, registry);
                    let y = function_cache.get_set_value(*y, state, state_functions, registry);
                    Self::reduce_table_2d_set_y(
                        op,
                        *capacity,
                        &set_tables.tables_2d[*i],
                        x.ones(),
                        y,
                    )
                }
                (
                    ArgumentExpression::Set(SetExpression::StateFunction(x)),
                    ArgumentExpression::Set(y),
                ) => {
                    let y = y.eval(state, function_cache, state_functions, registry);
                    let x = function_cache.get_set_value(*x, state, state_functions, registry);
                    Self::reduce_table_2d_set_y(
                        op,
                        *capacity,
                        &set_tables.tables_2d[*i],
                        x.ones(),
                        &y,
                    )
                }
                (
                    ArgumentExpression::Set(x),
                    ArgumentExpression::Set(SetExpression::StateFunction(y)),
                ) => {
                    let x = x.eval(state, function_cache, state_functions, registry);
                    let y = function_cache.get_set_value(*y, state, state_functions, registry);
                    Self::reduce_table_2d_set_y(
                        op,
                        *capacity,
                        &set_tables.tables_2d[*i],
                        x.ones(),
                        y,
                    )
                }
                (
                    ArgumentExpression::Set(SetExpression::Reference(x)),
                    ArgumentExpression::Set(SetExpression::Reference(y)),
                ) => {
                    let x = x.eval(state, function_cache, state_functions, registry);
                    let y = y.eval(state, function_cache, state_functions, registry);
                    Self::reduce_table_2d_set_y(
                        op,
                        *capacity,
                        &set_tables.tables_2d[*i],
                        x.ones(),
                        y,
                    )
                }
                (
                    ArgumentExpression::Set(x),
                    ArgumentExpression::Set(SetExpression::Reference(y)),
                ) => {
                    let x = x.eval(state, function_cache, state_functions, registry);
                    let y = y.eval(state, function_cache, state_functions, registry);
                    Self::reduce_table_2d_set_y(
                        op,
                        *capacity,
                        &set_tables.tables_2d[*i],
                        x.ones(),
                        y,
                    )
                }
                (
                    ArgumentExpression::Set(SetExpression::Reference(x)),
                    ArgumentExpression::Set(y),
                ) => {
                    let y = y.eval(state, function_cache, state_functions, registry);
                    let x = x.eval(state, function_cache, state_functions, registry);
                    Self::reduce_table_2d_set_y(
                        op,
                        *capacity,
                        &set_tables.tables_2d[*i],
                        x.ones(),
                        &y,
                    )
                }
                (ArgumentExpression::Set(x), ArgumentExpression::Set(y)) => {
                    let x = x.eval(state, function_cache, state_functions, registry);
                    let y = y.eval(state, function_cache, state_functions, registry);
                    Self::reduce_table_2d_set_y(
                        op,
                        *capacity,
                        &set_tables.tables_2d[*i],
                        x.ones(),
                        &y,
                    )
                }
                (
                    ArgumentExpression::Set(SetExpression::Reference(x)),
                    ArgumentExpression::Vector(VectorExpression::Reference(y)),
                ) => {
                    let y = y
                        .eval(state, function_cache, state_functions, registry)
                        .iter()
                        .copied();
                    let x = x.eval(state, function_cache, state_functions, registry);
                    Self::reduce_table_2d(op, *capacity, &set_tables.tables_2d[*i], x.ones(), y)
                }
                (
                    ArgumentExpression::Set(x),
                    ArgumentExpression::Vector(VectorExpression::Reference(y)),
                ) => {
                    let x = x.eval(state, function_cache, state_functions, registry);
                    let y = y
                        .eval(state, function_cache, state_functions, registry)
                        .iter()
                        .copied();
                    Self::reduce_table_2d(op, *capacity, &set_tables.tables_2d[*i], x.ones(), y)
                }
                (
                    ArgumentExpression::Set(SetExpression::Reference(x)),
                    ArgumentExpression::Vector(y),
                ) => {
                    let y = y
                        .eval(state, function_cache, state_functions, registry)
                        .into_iter();
                    let x = x.eval(state, function_cache, state_functions, registry);
                    Self::reduce_table_2d(op, *capacity, &set_tables.tables_2d[*i], x.ones(), y)
                }
                (ArgumentExpression::Set(x), ArgumentExpression::Vector(y)) => {
                    let x = x.eval(state, function_cache, state_functions, registry);
                    let y = y
                        .eval(state, function_cache, state_functions, registry)
                        .into_iter();
                    Self::reduce_table_2d(op, *capacity, &set_tables.tables_2d[*i], x.ones(), y)
                }
                (
                    ArgumentExpression::Vector(VectorExpression::Reference(x)),
                    ArgumentExpression::Set(SetExpression::Reference(y)),
                ) => {
                    let x = x
                        .eval(state, function_cache, state_functions, registry)
                        .iter()
                        .copied();
                    let y = y.eval(state, function_cache, state_functions, registry);
                    Self::reduce_table_2d_set_y(op, *capacity, &set_tables.tables_2d[*i], x, y)
                }
                (
                    ArgumentExpression::Vector(x),
                    ArgumentExpression::Set(SetExpression::Reference(y)),
                ) => {
                    let x = x
                        .eval(state, function_cache, state_functions, registry)
                        .into_iter();
                    let y = y.eval(state, function_cache, state_functions, registry);
                    Self::reduce_table_2d_set_y(op, *capacity, &set_tables.tables_2d[*i], x, y)
                }
                (
                    ArgumentExpression::Vector(VectorExpression::Reference(x)),
                    ArgumentExpression::Set(y),
                ) => {
                    let x = x
                        .eval(state, function_cache, state_functions, registry)
                        .iter()
                        .copied();
                    let y = y.eval(state, function_cache, state_functions, registry);
                    Self::reduce_table_2d_set_y(op, *capacity, &set_tables.tables_2d[*i], x, &y)
                }
                (ArgumentExpression::Vector(x), ArgumentExpression::Set(y)) => {
                    let x = x
                        .eval(state, function_cache, state_functions, registry)
                        .into_iter();
                    let y = y.eval(state, function_cache, state_functions, registry);
                    Self::reduce_table_2d_set_y(op, *capacity, &set_tables.tables_2d[*i], x, &y)
                }
                (
                    ArgumentExpression::Vector(VectorExpression::Reference(x)),
                    ArgumentExpression::Vector(VectorExpression::Reference(y)),
                ) => {
                    let x = x
                        .eval(state, function_cache, state_functions, registry)
                        .iter()
                        .copied();
                    let y = y
                        .eval(state, function_cache, state_functions, registry)
                        .iter()
                        .copied();
                    Self::reduce_table_2d(op, *capacity, &set_tables.tables_2d[*i], x, y)
                }
                (
                    ArgumentExpression::Vector(x),
                    ArgumentExpression::Vector(VectorExpression::Reference(y)),
                ) => {
                    let x = x
                        .eval(state, function_cache, state_functions, registry)
                        .into_iter();
                    let y = y
                        .eval(state, function_cache, state_functions, registry)
                        .iter()
                        .copied();
                    Self::reduce_table_2d(op, *capacity, &set_tables.tables_2d[*i], x, y)
                }
                (
                    ArgumentExpression::Vector(VectorExpression::Reference(x)),
                    ArgumentExpression::Vector(y),
                ) => {
                    let x = x
                        .eval(state, function_cache, state_functions, registry)
                        .iter()
                        .copied();
                    let y = y
                        .eval(state, function_cache, state_functions, registry)
                        .into_iter();
                    Self::reduce_table_2d(op, *capacity, &set_tables.tables_2d[*i], x, y)
                }
                (ArgumentExpression::Vector(x), ArgumentExpression::Vector(y)) => {
                    let x = x
                        .eval(state, function_cache, state_functions, registry)
                        .into_iter();
                    let y = y
                        .eval(state, function_cache, state_functions, registry)
                        .into_iter();
                    Self::reduce_table_2d(op, *capacity, &set_tables.tables_2d[*i], x, y)
                }
            },
            Self::Table3D(op, capacity, i, x, y, z) => {
                let args = ArgumentExpression::eval_args(
                    [x.as_ref(), y.as_ref(), z.as_ref()].into_iter(),
                    state,
                    function_cache,
                    state_functions,
                    registry,
                );
                let iter = args
                    .into_iter()
                    .map(|args| set_tables.tables_3d[*i].get(args[0], args[1], args[2]));
                op.eval(iter, *capacity)
            }
            Self::Table(op, capacity, i, args) => {
                let args = ArgumentExpression::eval_args(
                    args.iter(),
                    state,
                    function_cache,
                    state_functions,
                    registry,
                );
                let iter = args
                    .into_iter()
                    .map(|args| set_tables.tables[*i].get(&args));
                op.eval(iter, *capacity)
            }
        }
    }

    /// Returns a simplified version by precomputation.
    ///
    /// # Panics
    ///
    /// Panics if a min/max reduce operation is performed on an empty set or vector.
    pub fn simplify(&self, registry: &TableRegistry) -> Self {
        let set_tables = &registry.set_tables;
        match self {
            Self::Table1D(op, capacity, i, x) => match x.simplify(registry) {
                ArgumentExpression::Element(ElementExpression::Constant(x)) => {
                    Self::Constant(set_tables.tables_1d[*i].get(x).clone())
                }
                ArgumentExpression::Set(SetExpression::Reference(
                    ReferenceExpression::Constant(x),
                )) => Self::Constant(Self::reduce_table_1d(
                    op,
                    *capacity,
                    &set_tables.tables_1d[*i],
                    x.ones(),
                )),
                ArgumentExpression::Vector(VectorExpression::Reference(
                    ReferenceExpression::Constant(x),
                )) => Self::Constant(Self::reduce_table_1d(
                    op,
                    *capacity,
                    &set_tables.tables_1d[*i],
                    x.into_iter(),
                )),
                x => Self::Table1D(op.clone(), *capacity, *i, Box::new(x)),
            },
            Self::Table2D(op, capacity, i, x, y) => {
                let table = &set_tables.tables_2d[*i];
                match (x.simplify(registry), y.simplify(registry)) {
                    (
                        ArgumentExpression::Element(ElementExpression::Constant(x)),
                        ArgumentExpression::Element(ElementExpression::Constant(y)),
                    ) => Self::Constant(table.get(x, y).clone()),
                    (
                        ArgumentExpression::Element(ElementExpression::Constant(x)),
                        ArgumentExpression::Set(SetExpression::Reference(
                            ReferenceExpression::Constant(y),
                        )),
                    ) => Self::Constant(Self::reduce_table_2d_y(op, *capacity, table, x, y.ones())),
                    (
                        ArgumentExpression::Element(ElementExpression::Constant(x)),
                        ArgumentExpression::Vector(VectorExpression::Reference(
                            ReferenceExpression::Constant(y),
                        )),
                    ) => Self::Constant(Self::reduce_table_2d_y(
                        op,
                        *capacity,
                        table,
                        x,
                        y.into_iter(),
                    )),
                    (
                        ArgumentExpression::Set(SetExpression::Reference(
                            ReferenceExpression::Constant(x),
                        )),
                        ArgumentExpression::Element(ElementExpression::Constant(y)),
                    ) => Self::Constant(Self::reduce_table_2d_x(op, *capacity, table, x.ones(), y)),
                    (
                        ArgumentExpression::Vector(VectorExpression::Reference(
                            ReferenceExpression::Constant(x),
                        )),
                        ArgumentExpression::Element(ElementExpression::Constant(y)),
                    ) => Self::Constant(Self::reduce_table_2d_x(
                        op,
                        *capacity,
                        table,
                        x.into_iter(),
                        y,
                    )),
                    (
                        ArgumentExpression::Set(SetExpression::Reference(
                            ReferenceExpression::Constant(x),
                        )),
                        ArgumentExpression::Set(SetExpression::Reference(
                            ReferenceExpression::Constant(y),
                        )),
                    ) => Self::Constant(Self::reduce_table_2d_set_y(
                        op,
                        *capacity,
                        table,
                        x.ones(),
                        &y,
                    )),
                    (
                        ArgumentExpression::Set(SetExpression::Reference(
                            ReferenceExpression::Constant(x),
                        )),
                        ArgumentExpression::Vector(VectorExpression::Reference(
                            ReferenceExpression::Constant(y),
                        )),
                    ) => Self::Constant(Self::reduce_table_2d(
                        op,
                        *capacity,
                        table,
                        x.ones(),
                        y.into_iter(),
                    )),
                    (
                        ArgumentExpression::Vector(VectorExpression::Reference(
                            ReferenceExpression::Constant(x),
                        )),
                        ArgumentExpression::Set(SetExpression::Reference(
                            ReferenceExpression::Constant(y),
                        )),
                    ) => Self::Constant(Self::reduce_table_2d_set_y(
                        op,
                        *capacity,
                        table,
                        x.into_iter(),
                        &y,
                    )),
                    (
                        ArgumentExpression::Vector(VectorExpression::Reference(
                            ReferenceExpression::Constant(x),
                        )),
                        ArgumentExpression::Vector(VectorExpression::Reference(
                            ReferenceExpression::Constant(y),
                        )),
                    ) => Self::Constant(Self::reduce_table_2d(
                        op,
                        *capacity,
                        table,
                        x.into_iter(),
                        y.into_iter(),
                    )),
                    (x, y) => Self::Table2D(op.clone(), *capacity, *i, Box::new(x), Box::new(y)),
                }
            }
            Self::Table3D(op, capacity, i, x, y, z) => {
                let x = x.simplify(registry);
                let y = y.simplify(registry);
                let z = z.simplify(registry);
                if let Some(args) = ArgumentExpression::simplify_args([&x, &y, &z].into_iter()) {
                    let iter = args
                        .into_iter()
                        .map(|args| set_tables.tables_3d[*i].get(args[0], args[1], args[2]));
                    Self::Constant(op.eval(iter, *capacity))
                } else {
                    Self::Table3D(
                        op.clone(),
                        *capacity,
                        *i,
                        Box::new(x),
                        Box::new(y),
                        Box::new(z),
                    )
                }
            }
            Self::Table(op, capacity, i, args) => {
                let args: Vec<ArgumentExpression> =
                    args.iter().map(|x| x.simplify(registry)).collect();
                if let Some(args) = ArgumentExpression::simplify_args(args.iter()) {
                    let iter = args
                        .into_iter()
                        .map(|args| set_tables.tables[*i].get(&args));
                    Self::Constant(op.eval(iter, *capacity))
                } else {
                    Self::Table(op.clone(), *capacity, *i, args)
                }
            }
            _ => self.clone(),
        }
    }

    fn reduce_table_1d<I>(
        op: &SetReduceOperator,
        capacity: usize,
        table: &Table1D<Set>,
        x: I,
    ) -> Set
    where
        I: Iterator<Item = Element>,
    {
        let iter = x.map(|x| table.get(x));
        op.eval(iter, capacity)
    }

    fn reduce_table_2d<I, J>(
        op: &SetReduceOperator,
        capacity: usize,
        table: &Table2D<Set>,
        x: I,
        y: J,
    ) -> Set
    where
        I: Iterator<Item = Element>,
        J: Iterator<Item = Element> + Clone,
    {
        let iter = x.flat_map(|x| y.clone().map(move |y| table.get(x, y)));
        op.eval(iter, capacity)
    }

    fn reduce_table_2d_set_y<I>(
        op: &SetReduceOperator,
        capacity: usize,
        table: &Table2D<Set>,
        x: I,
        y: &Set,
    ) -> Set
    where
        I: Iterator<Item = Element>,
    {
        let iter = x.flat_map(|x| y.ones().map(move |y| table.get(x, y)));
        op.eval(iter, capacity)
    }

    fn reduce_table_2d_x<I>(
        op: &SetReduceOperator,
        capacity: usize,
        table: &Table2D<Set>,
        x: I,
        y: Element,
    ) -> Set
    where
        I: Iterator<Item = Element>,
    {
        let iter = x.map(|x| table.get(x, y));
        op.eval(iter, capacity)
    }

    fn reduce_table_2d_y<I>(
        op: &SetReduceOperator,
        capacity: usize,
        table: &Table2D<Set>,
        x: Element,
        y: I,
    ) -> Set
    where
        I: Iterator<Item = Element>,
    {
        let iter = y.map(|y| table.get(x, y));
        op.eval(iter, capacity)
    }
}

#[cfg(test)]
mod tests {
    use super::super::condition::Condition;
    use super::super::set_expression::SetElementOperation;
    use super::*;
    use crate::state::{SignatureVariables, State, StateMetadata};
    use crate::table::{Table, Table3D};
    use crate::table_data::{TableData, TableInterface};
    use rustc_hash::FxHashMap;

    #[test]
    fn empty_eval() {
        let op = SetReduceOperator::Union;
        let vector = [];
        assert_eq!(op.eval(vector.iter(), 3), Set::with_capacity(3));
    }

    #[test]
    fn union_eval() {
        let op = SetReduceOperator::Union;
        let vector = [
            {
                let mut set = Set::with_capacity(3);
                set.insert(0);
                set.insert(1);
                set
            },
            {
                let mut set = Set::with_capacity(3);
                set.insert(1);
                set.insert(2);
                set
            },
        ];
        assert_eq!(op.eval(vector.iter(), 3), {
            let mut set = Set::with_capacity(3);
            set.insert(0);
            set.insert(1);
            set.insert(2);
            set
        });
    }

    #[test]
    fn intersection_eval() {
        let op = SetReduceOperator::Intersection;
        let vector = [
            {
                let mut set = Set::with_capacity(3);
                set.insert(0);
                set.insert(1);
                set
            },
            {
                let mut set = Set::with_capacity(3);
                set.insert(1);
                set.insert(2);
                set
            },
        ];
        assert_eq!(op.eval(vector.iter(), 3), {
            let mut set = Set::with_capacity(3);
            set.insert(1);
            set
        });
    }

    #[test]
    fn symmetric_difference_eval() {
        let op = SetReduceOperator::SymmetricDifference;
        let vector = [
            {
                let mut set = Set::with_capacity(3);
                set.insert(0);
                set.insert(1);
                set
            },
            {
                let mut set = Set::with_capacity(3);
                set.insert(1);
                set.insert(2);
                set
            },
        ];
        assert_eq!(op.eval(vector.iter(), 3), {
            let mut set = Set::with_capacity(3);
            set.insert(0);
            set.insert(2);
            set
        });
    }

    #[test]
    fn constant_eval() {
        let state = State::default();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();
        let expression = SetReduceExpression::Constant(Set::with_capacity(3));
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            Set::with_capacity(3)
        );
    }

    #[test]
    fn table_1d_element_eval() {
        let state = State::default();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry {
            set_tables: TableData {
                tables_1d: vec![Table1D::new(vec![
                    {
                        let mut set = Set::with_capacity(3);
                        set.insert(0);
                        set.insert(1);
                        set
                    },
                    {
                        let mut set = Set::with_capacity(3);
                        set.insert(1);
                        set.insert(2);
                        set
                    },
                ])],
                ..Default::default()
            },
            ..Default::default()
        };
        let expression = SetReduceExpression::Table1D(
            SetReduceOperator::Intersection,
            3,
            0,
            Box::new(ArgumentExpression::Element(ElementExpression::Constant(0))),
        );
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            {
                let mut set = Set::with_capacity(3);
                set.insert(0);
                set.insert(1);
                set
            }
        );
    }

    #[test]
    fn table_1d_set_reference_eval() {
        let state = State::default();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry {
            set_tables: TableData {
                tables_1d: vec![Table1D::new(vec![
                    {
                        let mut set = Set::with_capacity(3);
                        set.insert(0);
                        set.insert(1);
                        set
                    },
                    {
                        let mut set = Set::with_capacity(3);
                        set.insert(1);
                        set.insert(2);
                        set
                    },
                ])],
                ..Default::default()
            },
            ..Default::default()
        };
        let expression = SetReduceExpression::Table1D(
            SetReduceOperator::Intersection,
            3,
            0,
            Box::new(ArgumentExpression::Set(SetExpression::Reference(
                ReferenceExpression::Constant({
                    let mut set = Set::with_capacity(2);
                    set.insert(0);
                    set.insert(1);
                    set
                }),
            ))),
        );
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            {
                let mut set = Set::with_capacity(3);
                set.insert(1);
                set
            }
        );
    }

    #[test]
    fn table_1d_set_eval() {
        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![Set::with_capacity(2)],
                ..Default::default()
            },
            ..Default::default()
        };
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry {
            set_tables: TableData {
                tables_1d: vec![Table1D::new(vec![
                    {
                        let mut set = Set::with_capacity(3);
                        set.insert(0);
                        set.insert(1);
                        set
                    },
                    {
                        let mut set = Set::with_capacity(3);
                        set.insert(1);
                        set.insert(2);
                        set
                    },
                ])],
                ..Default::default()
            },
            ..Default::default()
        };
        let expression = SetReduceExpression::Table1D(
            SetReduceOperator::Intersection,
            3,
            0,
            Box::new(ArgumentExpression::Set(SetExpression::Complement(
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
            ))),
        );
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            {
                let mut set = Set::with_capacity(3);
                set.insert(1);
                set
            }
        );
    }

    #[test]
    fn table_1d_vector_reference_eval() {
        let state = State::default();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry {
            set_tables: TableData {
                tables_1d: vec![Table1D::new(vec![
                    {
                        let mut set = Set::with_capacity(3);
                        set.insert(0);
                        set.insert(1);
                        set
                    },
                    {
                        let mut set = Set::with_capacity(3);
                        set.insert(1);
                        set.insert(2);
                        set
                    },
                ])],
                ..Default::default()
            },
            ..Default::default()
        };
        let expression = SetReduceExpression::Table1D(
            SetReduceOperator::Intersection,
            3,
            0,
            Box::new(ArgumentExpression::Vector(VectorExpression::Reference(
                ReferenceExpression::Constant(vec![0, 1]),
            ))),
        );
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            {
                let mut set = Set::with_capacity(3);
                set.insert(1);
                set
            }
        );
    }

    #[test]
    fn table_1d_vector_eval() {
        let state = State {
            signature_variables: SignatureVariables {
                vector_variables: vec![vec![1, 0]],
                ..Default::default()
            },
            ..Default::default()
        };
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry {
            set_tables: TableData {
                tables_1d: vec![Table1D::new(vec![
                    {
                        let mut set = Set::with_capacity(3);
                        set.insert(0);
                        set.insert(1);
                        set
                    },
                    {
                        let mut set = Set::with_capacity(3);
                        set.insert(1);
                        set.insert(2);
                        set
                    },
                ])],
                ..Default::default()
            },
            ..Default::default()
        };
        let expression = SetReduceExpression::Table1D(
            SetReduceOperator::Intersection,
            3,
            0,
            Box::new(ArgumentExpression::Vector(VectorExpression::Reverse(
                Box::new(VectorExpression::Reference(ReferenceExpression::Variable(
                    0,
                ))),
            ))),
        );
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            {
                let mut set = Set::with_capacity(3);
                set.insert(1);
                set
            }
        );
    }

    #[test]
    fn table_1d_set_state_function_eval() {
        let mut state_metadata = StateMetadata::default();
        let ob = state_metadata.add_object_type("ob", 3);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = state_metadata.add_set_variable("v", ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_set_function("f", v.add(0).add(1));
        assert!(f.is_ok());
        let f = f.unwrap();

        let set1 = state_metadata.create_set(ob, &[0, 1]);
        assert!(set1.is_ok());
        let set1 = set1.unwrap();
        let set2 = state_metadata.create_set(ob, &[1, 2]);
        assert!(set2.is_ok());
        let set2 = set2.unwrap();

        let mut registry = TableRegistry::default();
        let t = registry.add_table_1d("t", vec![set1.clone(), set2.clone()]);
        assert!(t.is_ok());
        let t = t.unwrap();

        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![Set::with_capacity(3)],
                ..Default::default()
            },
            ..Default::default()
        };

        let expression = SetReduceExpression::Table1D(
            SetReduceOperator::Intersection,
            3,
            t.id(),
            Box::new(f.into()),
        );

        let mut expected = set1.clone();
        expected.intersect_with(&set2);

        assert_eq!(
            expression.eval(
                &state,
                &mut StateFunctionCache::new(&state_functions),
                &state_functions,
                &registry
            ),
            expected,
        );
    }

    #[test]
    fn table_2d_element_eval() {
        let state = State::default();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry {
            set_tables: TableData {
                tables_2d: vec![Table2D::new(vec![
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(1);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(2);
                            set
                        },
                    ],
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(3);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(4);
                            set
                        },
                    ],
                ])],
                ..Default::default()
            },
            ..Default::default()
        };
        let expression = SetReduceExpression::Table2D(
            SetReduceOperator::Union,
            5,
            0,
            Box::new(ArgumentExpression::Element(ElementExpression::Constant(0))),
            Box::new(ArgumentExpression::Element(ElementExpression::Constant(0))),
        );
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            {
                let mut set = Set::with_capacity(5);
                set.insert(0);
                set.insert(1);
                set
            }
        );
    }

    #[test]
    fn table_2d_element_set_reference_eval() {
        let state = State::default();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry {
            set_tables: TableData {
                tables_2d: vec![Table2D::new(vec![
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(1);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(2);
                            set
                        },
                    ],
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(3);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(4);
                            set
                        },
                    ],
                ])],
                ..Default::default()
            },
            ..Default::default()
        };
        let expression = SetReduceExpression::Table2D(
            SetReduceOperator::Union,
            5,
            0,
            Box::new(ArgumentExpression::Element(ElementExpression::Constant(0))),
            Box::new(ArgumentExpression::Set(SetExpression::Reference(
                ReferenceExpression::Constant({
                    let mut set = Set::with_capacity(2);
                    set.insert(0);
                    set.insert(1);
                    set
                }),
            ))),
        );
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            {
                let mut set = Set::with_capacity(5);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set
            }
        );
    }

    #[test]
    fn table_2d_element_set_eval() {
        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![Set::with_capacity(2)],
                ..Default::default()
            },
            ..Default::default()
        };
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry {
            set_tables: TableData {
                tables_2d: vec![Table2D::new(vec![
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(1);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(2);
                            set
                        },
                    ],
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(3);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(4);
                            set
                        },
                    ],
                ])],
                ..Default::default()
            },
            ..Default::default()
        };
        let expression = SetReduceExpression::Table2D(
            SetReduceOperator::Union,
            5,
            0,
            Box::new(ArgumentExpression::Element(ElementExpression::Constant(0))),
            Box::new(ArgumentExpression::Set(SetExpression::Complement(
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
            ))),
        );
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            {
                let mut set = Set::with_capacity(5);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set
            }
        );
    }

    #[test]
    fn table_2d_element_vector_reference_eval() {
        let state = State::default();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry {
            set_tables: TableData {
                tables_2d: vec![Table2D::new(vec![
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(1);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(2);
                            set
                        },
                    ],
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(3);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(4);
                            set
                        },
                    ],
                ])],
                ..Default::default()
            },
            ..Default::default()
        };
        let expression = SetReduceExpression::Table2D(
            SetReduceOperator::Union,
            5,
            0,
            Box::new(ArgumentExpression::Element(ElementExpression::Constant(0))),
            Box::new(ArgumentExpression::Vector(VectorExpression::Reference(
                ReferenceExpression::Constant(vec![0, 1]),
            ))),
        );
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            {
                let mut set = Set::with_capacity(5);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set
            }
        );
    }

    #[test]
    fn table_2d_element_vector_eval() {
        let state = State {
            signature_variables: SignatureVariables {
                vector_variables: vec![vec![1, 0]],
                ..Default::default()
            },
            ..Default::default()
        };
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry {
            set_tables: TableData {
                tables_2d: vec![Table2D::new(vec![
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(1);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(2);
                            set
                        },
                    ],
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(3);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(4);
                            set
                        },
                    ],
                ])],
                ..Default::default()
            },
            ..Default::default()
        };
        let expression = SetReduceExpression::Table2D(
            SetReduceOperator::Union,
            5,
            0,
            Box::new(ArgumentExpression::Element(ElementExpression::Constant(0))),
            Box::new(ArgumentExpression::Vector(VectorExpression::Reverse(
                Box::new(VectorExpression::Reference(ReferenceExpression::Variable(
                    0,
                ))),
            ))),
        );
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            {
                let mut set = Set::with_capacity(5);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set
            }
        );
    }

    #[test]
    fn table_2d_set_reference_element_eval() {
        let state = State::default();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry {
            set_tables: TableData {
                tables_2d: vec![Table2D::new(vec![
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(1);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(2);
                            set
                        },
                    ],
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(3);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(4);
                            set
                        },
                    ],
                ])],
                ..Default::default()
            },
            ..Default::default()
        };
        let expression = SetReduceExpression::Table2D(
            SetReduceOperator::Union,
            5,
            0,
            Box::new(ArgumentExpression::Set(SetExpression::Reference(
                ReferenceExpression::Constant({
                    let mut set = Set::with_capacity(2);
                    set.insert(0);
                    set.insert(1);
                    set
                }),
            ))),
            Box::new(ArgumentExpression::Element(ElementExpression::Constant(0))),
        );
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            {
                let mut set = Set::with_capacity(5);
                set.insert(0);
                set.insert(1);
                set.insert(3);
                set
            }
        );
    }

    #[test]
    fn table_2d_set_element_eval() {
        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![Set::with_capacity(2)],
                ..Default::default()
            },
            ..Default::default()
        };
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry {
            set_tables: TableData {
                tables_2d: vec![Table2D::new(vec![
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(1);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(2);
                            set
                        },
                    ],
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(3);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(4);
                            set
                        },
                    ],
                ])],
                ..Default::default()
            },
            ..Default::default()
        };
        let expression = SetReduceExpression::Table2D(
            SetReduceOperator::Union,
            5,
            0,
            Box::new(ArgumentExpression::Set(SetExpression::Complement(
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
            ))),
            Box::new(ArgumentExpression::Element(ElementExpression::Constant(0))),
        );
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            {
                let mut set = Set::with_capacity(5);
                set.insert(0);
                set.insert(1);
                set.insert(3);
                set
            }
        );
    }

    #[test]
    fn table_2d_vector_reference_element_eval() {
        let state = State::default();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry {
            set_tables: TableData {
                tables_2d: vec![Table2D::new(vec![
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(1);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(2);
                            set
                        },
                    ],
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(3);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(4);
                            set
                        },
                    ],
                ])],
                ..Default::default()
            },
            ..Default::default()
        };
        let expression = SetReduceExpression::Table2D(
            SetReduceOperator::Union,
            5,
            0,
            Box::new(ArgumentExpression::Vector(VectorExpression::Reference(
                ReferenceExpression::Constant(vec![0, 1]),
            ))),
            Box::new(ArgumentExpression::Element(ElementExpression::Constant(0))),
        );
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            {
                let mut set = Set::with_capacity(5);
                set.insert(0);
                set.insert(1);
                set.insert(3);
                set
            }
        );
    }

    #[test]
    fn table_2d_vector_element_eval() {
        let state = State {
            signature_variables: SignatureVariables {
                vector_variables: vec![vec![1, 0]],
                ..Default::default()
            },
            ..Default::default()
        };
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry {
            set_tables: TableData {
                tables_2d: vec![Table2D::new(vec![
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(1);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(2);
                            set
                        },
                    ],
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(3);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(4);
                            set
                        },
                    ],
                ])],
                ..Default::default()
            },
            ..Default::default()
        };
        let expression = SetReduceExpression::Table2D(
            SetReduceOperator::Union,
            5,
            0,
            Box::new(ArgumentExpression::Vector(VectorExpression::Reverse(
                Box::new(VectorExpression::Reference(ReferenceExpression::Variable(
                    0,
                ))),
            ))),
            Box::new(ArgumentExpression::Element(ElementExpression::Constant(0))),
        );
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            {
                let mut set = Set::with_capacity(5);
                set.insert(0);
                set.insert(1);
                set.insert(3);
                set
            }
        );
    }

    #[test]
    fn table_2d_set_reference_eval() {
        let state = State::default();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry {
            set_tables: TableData {
                tables_2d: vec![Table2D::new(vec![
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(1);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(2);
                            set
                        },
                    ],
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(3);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(4);
                            set
                        },
                    ],
                ])],
                ..Default::default()
            },
            ..Default::default()
        };
        let expression = SetReduceExpression::Table2D(
            SetReduceOperator::Union,
            5,
            0,
            Box::new(ArgumentExpression::Set(SetExpression::Reference(
                ReferenceExpression::Constant({
                    let mut set = Set::with_capacity(2);
                    set.insert(0);
                    set.insert(1);
                    set
                }),
            ))),
            Box::new(ArgumentExpression::Set(SetExpression::Reference(
                ReferenceExpression::Constant({
                    let mut set = Set::with_capacity(2);
                    set.insert(0);
                    set.insert(1);
                    set
                }),
            ))),
        );
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            {
                let mut set = Set::with_capacity(5);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set.insert(3);
                set.insert(4);
                set
            }
        );
    }

    #[test]
    fn table_2d_set_set_reference_eval() {
        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![Set::with_capacity(2)],
                ..Default::default()
            },
            ..Default::default()
        };
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry {
            set_tables: TableData {
                tables_2d: vec![Table2D::new(vec![
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(1);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(2);
                            set
                        },
                    ],
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(3);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(4);
                            set
                        },
                    ],
                ])],
                ..Default::default()
            },
            ..Default::default()
        };
        let expression = SetReduceExpression::Table2D(
            SetReduceOperator::Union,
            5,
            0,
            Box::new(ArgumentExpression::Set(SetExpression::Complement(
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
            ))),
            Box::new(ArgumentExpression::Set(SetExpression::Reference(
                ReferenceExpression::Constant({
                    let mut set = Set::with_capacity(2);
                    set.insert(0);
                    set.insert(1);
                    set
                }),
            ))),
        );
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            {
                let mut set = Set::with_capacity(5);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set.insert(3);
                set.insert(4);
                set
            }
        );
    }

    #[test]
    fn table_2d_set_reference_set_eval() {
        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![Set::with_capacity(2)],
                ..Default::default()
            },
            ..Default::default()
        };
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry {
            set_tables: TableData {
                tables_2d: vec![Table2D::new(vec![
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(1);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(2);
                            set
                        },
                    ],
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(3);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(4);
                            set
                        },
                    ],
                ])],
                ..Default::default()
            },
            ..Default::default()
        };
        let expression = SetReduceExpression::Table2D(
            SetReduceOperator::Union,
            5,
            0,
            Box::new(ArgumentExpression::Set(SetExpression::Reference(
                ReferenceExpression::Constant({
                    let mut set = Set::with_capacity(2);
                    set.insert(0);
                    set.insert(1);
                    set
                }),
            ))),
            Box::new(ArgumentExpression::Set(SetExpression::Complement(
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
            ))),
        );
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            {
                let mut set = Set::with_capacity(5);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set.insert(3);
                set.insert(4);
                set
            }
        );
    }

    #[test]
    fn table_2d_set_eval() {
        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![Set::with_capacity(2)],
                ..Default::default()
            },
            ..Default::default()
        };
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry {
            set_tables: TableData {
                tables_2d: vec![Table2D::new(vec![
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(1);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(2);
                            set
                        },
                    ],
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(3);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(4);
                            set
                        },
                    ],
                ])],
                ..Default::default()
            },
            ..Default::default()
        };
        let expression = SetReduceExpression::Table2D(
            SetReduceOperator::Union,
            5,
            0,
            Box::new(ArgumentExpression::Set(SetExpression::Complement(
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
            ))),
            Box::new(ArgumentExpression::Set(SetExpression::Complement(
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
            ))),
        );
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            {
                let mut set = Set::with_capacity(5);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set.insert(3);
                set.insert(4);
                set
            }
        );
    }

    #[test]
    fn table_2d_set_reference_vector_reference_eval() {
        let state = State::default();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry {
            set_tables: TableData {
                tables_2d: vec![Table2D::new(vec![
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(1);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(2);
                            set
                        },
                    ],
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(3);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(4);
                            set
                        },
                    ],
                ])],
                ..Default::default()
            },
            ..Default::default()
        };
        let expression = SetReduceExpression::Table2D(
            SetReduceOperator::Union,
            5,
            0,
            Box::new(ArgumentExpression::Set(SetExpression::Reference(
                ReferenceExpression::Constant({
                    let mut set = Set::with_capacity(2);
                    set.insert(0);
                    set.insert(1);
                    set
                }),
            ))),
            Box::new(ArgumentExpression::Vector(VectorExpression::Reference(
                ReferenceExpression::Constant(vec![0, 1]),
            ))),
        );
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            {
                let mut set = Set::with_capacity(5);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set.insert(3);
                set.insert(4);
                set
            }
        );
    }

    #[test]
    fn table_2d_set_vector_reference_eval() {
        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![Set::with_capacity(2)],
                ..Default::default()
            },
            ..Default::default()
        };
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry {
            set_tables: TableData {
                tables_2d: vec![Table2D::new(vec![
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(1);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(2);
                            set
                        },
                    ],
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(3);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(4);
                            set
                        },
                    ],
                ])],
                ..Default::default()
            },
            ..Default::default()
        };
        let expression = SetReduceExpression::Table2D(
            SetReduceOperator::Union,
            5,
            0,
            Box::new(ArgumentExpression::Set(SetExpression::Complement(
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
            ))),
            Box::new(ArgumentExpression::Vector(VectorExpression::Reference(
                ReferenceExpression::Constant(vec![0, 1]),
            ))),
        );
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            {
                let mut set = Set::with_capacity(5);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set.insert(3);
                set.insert(4);
                set
            }
        );
    }

    #[test]
    fn table_2d_set_reference_vector_eval() {
        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![Set::with_capacity(2)],
                ..Default::default()
            },
            ..Default::default()
        };
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry {
            set_tables: TableData {
                tables_2d: vec![Table2D::new(vec![
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(1);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(2);
                            set
                        },
                    ],
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(3);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(4);
                            set
                        },
                    ],
                ])],
                ..Default::default()
            },
            ..Default::default()
        };
        let expression = SetReduceExpression::Table2D(
            SetReduceOperator::Union,
            5,
            0,
            Box::new(ArgumentExpression::Set(SetExpression::Reference(
                ReferenceExpression::Constant({
                    let mut set = Set::with_capacity(2);
                    set.insert(0);
                    set.insert(1);
                    set
                }),
            ))),
            Box::new(ArgumentExpression::Vector(VectorExpression::Reverse(
                Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                    vec![1, 0],
                ))),
            ))),
        );
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            {
                let mut set = Set::with_capacity(5);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set.insert(3);
                set.insert(4);
                set
            }
        );
    }

    #[test]
    fn table_2d_set_vector_eval() {
        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![Set::with_capacity(2)],
                ..Default::default()
            },
            ..Default::default()
        };
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry {
            set_tables: TableData {
                tables_2d: vec![Table2D::new(vec![
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(1);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(2);
                            set
                        },
                    ],
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(3);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(4);
                            set
                        },
                    ],
                ])],
                ..Default::default()
            },
            ..Default::default()
        };
        let expression = SetReduceExpression::Table2D(
            SetReduceOperator::Union,
            5,
            0,
            Box::new(ArgumentExpression::Set(SetExpression::Complement(
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
            ))),
            Box::new(ArgumentExpression::Vector(VectorExpression::Reverse(
                Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                    vec![1, 0],
                ))),
            ))),
        );
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            {
                let mut set = Set::with_capacity(5);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set.insert(3);
                set.insert(4);
                set
            }
        );
    }

    #[test]
    fn table_2d_vector_reference_set_reference_eval() {
        let state = State::default();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry {
            set_tables: TableData {
                tables_2d: vec![Table2D::new(vec![
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(1);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(2);
                            set
                        },
                    ],
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(3);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(4);
                            set
                        },
                    ],
                ])],
                ..Default::default()
            },
            ..Default::default()
        };
        let expression = SetReduceExpression::Table2D(
            SetReduceOperator::Union,
            5,
            0,
            Box::new(ArgumentExpression::Vector(VectorExpression::Reference(
                ReferenceExpression::Constant(vec![0, 1]),
            ))),
            Box::new(ArgumentExpression::Set(SetExpression::Reference(
                ReferenceExpression::Constant({
                    let mut set = Set::with_capacity(2);
                    set.insert(0);
                    set.insert(1);
                    set
                }),
            ))),
        );
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            {
                let mut set = Set::with_capacity(5);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set.insert(3);
                set.insert(4);
                set
            }
        );
    }

    #[test]
    fn table_2d_vector_reference_set_eval() {
        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![Set::with_capacity(2)],
                ..Default::default()
            },
            ..Default::default()
        };
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry {
            set_tables: TableData {
                tables_2d: vec![Table2D::new(vec![
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(1);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(2);
                            set
                        },
                    ],
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(3);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(4);
                            set
                        },
                    ],
                ])],
                ..Default::default()
            },
            ..Default::default()
        };
        let expression = SetReduceExpression::Table2D(
            SetReduceOperator::Union,
            5,
            0,
            Box::new(ArgumentExpression::Vector(VectorExpression::Reference(
                ReferenceExpression::Constant(vec![0, 1]),
            ))),
            Box::new(ArgumentExpression::Set(SetExpression::Complement(
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
            ))),
        );
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            {
                let mut set = Set::with_capacity(5);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set.insert(3);
                set.insert(4);
                set
            }
        );
    }

    #[test]
    fn table_2d_vector_set_reference_eval() {
        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![Set::with_capacity(2)],
                ..Default::default()
            },
            ..Default::default()
        };
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry {
            set_tables: TableData {
                tables_2d: vec![Table2D::new(vec![
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(1);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(2);
                            set
                        },
                    ],
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(3);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(4);
                            set
                        },
                    ],
                ])],
                ..Default::default()
            },
            ..Default::default()
        };
        let expression = SetReduceExpression::Table2D(
            SetReduceOperator::Union,
            5,
            0,
            Box::new(ArgumentExpression::Vector(VectorExpression::Reverse(
                Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                    vec![1, 0],
                ))),
            ))),
            Box::new(ArgumentExpression::Set(SetExpression::Reference(
                ReferenceExpression::Constant({
                    let mut set = Set::with_capacity(2);
                    set.insert(0);
                    set.insert(1);
                    set
                }),
            ))),
        );
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            {
                let mut set = Set::with_capacity(5);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set.insert(3);
                set.insert(4);
                set
            }
        );
    }

    #[test]
    fn table_2d_vector_set_eval() {
        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![Set::with_capacity(2)],
                ..Default::default()
            },
            ..Default::default()
        };
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry {
            set_tables: TableData {
                tables_2d: vec![Table2D::new(vec![
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(1);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(2);
                            set
                        },
                    ],
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(3);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(4);
                            set
                        },
                    ],
                ])],
                ..Default::default()
            },
            ..Default::default()
        };
        let expression = SetReduceExpression::Table2D(
            SetReduceOperator::Union,
            5,
            0,
            Box::new(ArgumentExpression::Vector(VectorExpression::Reverse(
                Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                    vec![1, 0],
                ))),
            ))),
            Box::new(ArgumentExpression::Set(SetExpression::Complement(
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
            ))),
        );
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            {
                let mut set = Set::with_capacity(5);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set.insert(3);
                set.insert(4);
                set
            }
        );
    }

    #[test]
    fn table_2d_vector_reference_eval() {
        let state = State::default();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry {
            set_tables: TableData {
                tables_2d: vec![Table2D::new(vec![
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(1);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(2);
                            set
                        },
                    ],
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(3);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(4);
                            set
                        },
                    ],
                ])],
                ..Default::default()
            },
            ..Default::default()
        };
        let expression = SetReduceExpression::Table2D(
            SetReduceOperator::Union,
            5,
            0,
            Box::new(ArgumentExpression::Vector(VectorExpression::Reference(
                ReferenceExpression::Constant(vec![0, 1]),
            ))),
            Box::new(ArgumentExpression::Vector(VectorExpression::Reference(
                ReferenceExpression::Constant(vec![0, 1]),
            ))),
        );
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            {
                let mut set = Set::with_capacity(5);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set.insert(3);
                set.insert(4);
                set
            }
        );
    }

    #[test]
    fn table_2d_vector_reference_vector_eval() {
        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![Set::with_capacity(2)],
                ..Default::default()
            },
            ..Default::default()
        };
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry {
            set_tables: TableData {
                tables_2d: vec![Table2D::new(vec![
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(1);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(2);
                            set
                        },
                    ],
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(3);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(4);
                            set
                        },
                    ],
                ])],
                ..Default::default()
            },
            ..Default::default()
        };
        let expression = SetReduceExpression::Table2D(
            SetReduceOperator::Union,
            5,
            0,
            Box::new(ArgumentExpression::Vector(VectorExpression::Reference(
                ReferenceExpression::Constant(vec![0, 1]),
            ))),
            Box::new(ArgumentExpression::Vector(VectorExpression::Reverse(
                Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                    vec![1, 0],
                ))),
            ))),
        );
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            {
                let mut set = Set::with_capacity(5);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set.insert(3);
                set.insert(4);
                set
            }
        );
    }

    #[test]
    fn table_2d_vector_vector_reference_eval() {
        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![Set::with_capacity(2)],
                ..Default::default()
            },
            ..Default::default()
        };
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry {
            set_tables: TableData {
                tables_2d: vec![Table2D::new(vec![
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(1);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(2);
                            set
                        },
                    ],
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(3);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(4);
                            set
                        },
                    ],
                ])],
                ..Default::default()
            },
            ..Default::default()
        };
        let expression = SetReduceExpression::Table2D(
            SetReduceOperator::Union,
            5,
            0,
            Box::new(ArgumentExpression::Vector(VectorExpression::Reverse(
                Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                    vec![1, 0],
                ))),
            ))),
            Box::new(ArgumentExpression::Vector(VectorExpression::Reference(
                ReferenceExpression::Constant(vec![0, 1]),
            ))),
        );
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            {
                let mut set = Set::with_capacity(5);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set.insert(3);
                set.insert(4);
                set
            }
        );
    }

    #[test]
    fn table_2d_vector_eval() {
        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![Set::with_capacity(2)],
                ..Default::default()
            },
            ..Default::default()
        };
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry {
            set_tables: TableData {
                tables_2d: vec![Table2D::new(vec![
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(1);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(2);
                            set
                        },
                    ],
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(3);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(4);
                            set
                        },
                    ],
                ])],
                ..Default::default()
            },
            ..Default::default()
        };
        let expression = SetReduceExpression::Table2D(
            SetReduceOperator::Union,
            5,
            0,
            Box::new(ArgumentExpression::Vector(VectorExpression::Reverse(
                Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                    vec![1, 0],
                ))),
            ))),
            Box::new(ArgumentExpression::Vector(VectorExpression::Reverse(
                Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                    vec![1, 0],
                ))),
            ))),
        );
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            {
                let mut set = Set::with_capacity(5);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set.insert(3);
                set.insert(4);
                set
            }
        );
    }

    #[test]
    fn table_2d_set_state_function_element_eval() {
        let mut state_metadata = StateMetadata::default();
        let ob = state_metadata.add_object_type("ob", 5);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = state_metadata.add_set_variable("v", ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_set_function("f", v.add(0).add(1));
        assert!(f.is_ok());
        let f = f.unwrap();

        let set1 = state_metadata.create_set(ob, &[0, 1]);
        assert!(set1.is_ok());
        let set1 = set1.unwrap();
        let set2 = state_metadata.create_set(ob, &[0, 2]);
        assert!(set2.is_ok());
        let set2 = set2.unwrap();
        let set3 = state_metadata.create_set(ob, &[0, 3]);
        assert!(set3.is_ok());
        let set3 = set3.unwrap();
        let set4 = state_metadata.create_set(ob, &[0, 4]);
        assert!(set4.is_ok());
        let set4 = set4.unwrap();

        let mut registry = TableRegistry::default();
        let t = registry.add_table_2d(
            "t",
            vec![
                vec![set1.clone(), set2.clone()],
                vec![set3.clone(), set4.clone()],
            ],
        );
        assert!(t.is_ok());
        let t = t.unwrap();

        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![set1.clone()],
                ..Default::default()
            },
            ..Default::default()
        };

        let expression = SetReduceExpression::Table2D(
            SetReduceOperator::Union,
            3,
            t.id(),
            Box::new(f.into()),
            Box::new(1.into()),
        );

        let mut expected = set2.clone();
        expected.union_with(&set4);

        assert_eq!(
            expression.eval(
                &state,
                &mut StateFunctionCache::new(&state_functions),
                &state_functions,
                &registry
            ),
            expected,
        );
    }

    #[test]
    fn table_2d_element_set_state_function_eval() {
        let mut state_metadata = StateMetadata::default();
        let ob = state_metadata.add_object_type("ob", 5);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = state_metadata.add_set_variable("v", ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_set_function("f", v.add(0).add(1));
        assert!(f.is_ok());
        let f = f.unwrap();

        let set1 = state_metadata.create_set(ob, &[0, 1]);
        assert!(set1.is_ok());
        let set1 = set1.unwrap();
        let set2 = state_metadata.create_set(ob, &[0, 2]);
        assert!(set2.is_ok());
        let set2 = set2.unwrap();
        let set3 = state_metadata.create_set(ob, &[0, 3]);
        assert!(set3.is_ok());
        let set3 = set3.unwrap();
        let set4 = state_metadata.create_set(ob, &[0, 4]);
        assert!(set4.is_ok());
        let set4 = set4.unwrap();

        let mut registry = TableRegistry::default();
        let t = registry.add_table_2d(
            "t",
            vec![
                vec![set1.clone(), set2.clone()],
                vec![set3.clone(), set4.clone()],
            ],
        );
        assert!(t.is_ok());
        let t = t.unwrap();

        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![set1.clone()],
                ..Default::default()
            },
            ..Default::default()
        };

        let expression = SetReduceExpression::Table2D(
            SetReduceOperator::Union,
            3,
            t.id(),
            Box::new(1.into()),
            Box::new(f.into()),
        );

        let mut expected = set3.clone();
        expected.union_with(&set4);

        assert_eq!(
            expression.eval(
                &state,
                &mut StateFunctionCache::new(&state_functions),
                &state_functions,
                &registry
            ),
            expected,
        );
    }

    #[test]
    fn table_2d_set_state_function_set_state_function_eval() {
        let mut state_metadata = StateMetadata::default();
        let ob = state_metadata.add_object_type("ob", 5);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = state_metadata.add_set_variable("v", ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_set_function("f", v.add(0).add(1));
        assert!(f.is_ok());
        let f = f.unwrap();
        let g = state_functions.add_set_function("g", v.add(0).add(1));
        assert!(g.is_ok());
        let g = g.unwrap();

        let set1 = state_metadata.create_set(ob, &[0, 1]);
        assert!(set1.is_ok());
        let set1 = set1.unwrap();
        let set2 = state_metadata.create_set(ob, &[0, 2]);
        assert!(set2.is_ok());
        let set2 = set2.unwrap();
        let set3 = state_metadata.create_set(ob, &[0, 3]);
        assert!(set3.is_ok());
        let set3 = set3.unwrap();
        let set4 = state_metadata.create_set(ob, &[0, 4]);
        assert!(set4.is_ok());
        let set4 = set4.unwrap();

        let mut registry = TableRegistry::default();
        let t = registry.add_table_2d(
            "t",
            vec![
                vec![set1.clone(), set2.clone()],
                vec![set3.clone(), set4.clone()],
            ],
        );
        assert!(t.is_ok());
        let t = t.unwrap();

        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![Set::with_capacity(3)],
                ..Default::default()
            },
            ..Default::default()
        };

        let expression = SetReduceExpression::Table2D(
            SetReduceOperator::Union,
            3,
            t.id(),
            Box::new(f.into()),
            Box::new(g.into()),
        );

        let mut expected = set1.clone();
        expected.union_with(&set2);
        expected.union_with(&set3);
        expected.union_with(&set4);

        assert_eq!(
            expression.eval(
                &state,
                &mut StateFunctionCache::new(&state_functions),
                &state_functions,
                &registry
            ),
            expected,
        );
    }

    #[test]
    fn table_2d_set_state_function_set_reference_eval() {
        let mut state_metadata = StateMetadata::default();
        let ob = state_metadata.add_object_type("ob", 5);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = state_metadata.add_set_variable("v", ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_set_function("f", v.add(0).add(1));
        assert!(f.is_ok());
        let f = f.unwrap();

        let set1 = state_metadata.create_set(ob, &[0, 1]);
        assert!(set1.is_ok());
        let set1 = set1.unwrap();
        let set2 = state_metadata.create_set(ob, &[0, 2]);
        assert!(set2.is_ok());
        let set2 = set2.unwrap();
        let set3 = state_metadata.create_set(ob, &[0, 3]);
        assert!(set3.is_ok());
        let set3 = set3.unwrap();
        let set4 = state_metadata.create_set(ob, &[0, 4]);
        assert!(set4.is_ok());
        let set4 = set4.unwrap();

        let mut registry = TableRegistry::default();
        let t = registry.add_table_2d(
            "t",
            vec![
                vec![set1.clone(), set2.clone()],
                vec![set3.clone(), set4.clone()],
            ],
        );
        assert!(t.is_ok());
        let t = t.unwrap();

        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![set1.clone()],
                ..Default::default()
            },
            ..Default::default()
        };

        let expression = SetReduceExpression::Table2D(
            SetReduceOperator::Union,
            3,
            t.id(),
            Box::new(f.into()),
            Box::new(v.into()),
        );

        let mut expected = set1.clone();
        expected.union_with(&set2);
        expected.union_with(&set3);
        expected.union_with(&set4);

        assert_eq!(
            expression.eval(
                &state,
                &mut StateFunctionCache::new(&state_functions),
                &state_functions,
                &registry
            ),
            expected,
        );
    }

    #[test]
    fn table_2d_set_reference_set_state_function_eval() {
        let mut state_metadata = StateMetadata::default();
        let ob = state_metadata.add_object_type("ob", 5);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = state_metadata.add_set_variable("v", ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_set_function("f", v.add(0).add(1));
        assert!(f.is_ok());
        let f = f.unwrap();

        let set1 = state_metadata.create_set(ob, &[0, 1]);
        assert!(set1.is_ok());
        let set1 = set1.unwrap();
        let set2 = state_metadata.create_set(ob, &[0, 2]);
        assert!(set2.is_ok());
        let set2 = set2.unwrap();
        let set3 = state_metadata.create_set(ob, &[0, 3]);
        assert!(set3.is_ok());
        let set3 = set3.unwrap();
        let set4 = state_metadata.create_set(ob, &[0, 4]);
        assert!(set4.is_ok());
        let set4 = set4.unwrap();

        let mut registry = TableRegistry::default();
        let t = registry.add_table_2d(
            "t",
            vec![
                vec![set1.clone(), set2.clone()],
                vec![set3.clone(), set4.clone()],
            ],
        );
        assert!(t.is_ok());
        let t = t.unwrap();

        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![set1.clone()],
                ..Default::default()
            },
            ..Default::default()
        };

        let expression = SetReduceExpression::Table2D(
            SetReduceOperator::Union,
            3,
            t.id(),
            Box::new(v.into()),
            Box::new(f.into()),
        );

        let mut expected = set1.clone();
        expected.union_with(&set2);
        expected.union_with(&set3);
        expected.union_with(&set4);

        assert_eq!(
            expression.eval(
                &state,
                &mut StateFunctionCache::new(&state_functions),
                &state_functions,
                &registry
            ),
            expected,
        );
    }

    #[test]
    fn table_3d_eval() {
        let state = State::default();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry {
            set_tables: TableData {
                tables_3d: vec![Table3D::new(vec![vec![
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(1);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(2);
                            set
                        },
                    ],
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(3);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(4);
                            set
                        },
                    ],
                ]])],
                ..Default::default()
            },
            ..Default::default()
        };
        let expression = SetReduceExpression::Table3D(
            SetReduceOperator::Union,
            5,
            0,
            Box::new(ArgumentExpression::Element(ElementExpression::Constant(0))),
            Box::new(ArgumentExpression::Set(SetExpression::Reference(
                ReferenceExpression::Constant({
                    let mut set = Set::with_capacity(2);
                    set.insert(0);
                    set.insert(1);
                    set
                }),
            ))),
            Box::new(ArgumentExpression::Vector(VectorExpression::Reference(
                ReferenceExpression::Constant(vec![0, 1]),
            ))),
        );
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            {
                let mut set = Set::with_capacity(5);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set.insert(3);
                set.insert(4);
                set
            }
        );
    }

    #[test]
    fn table_eval() {
        let state = State::default();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry {
            set_tables: TableData {
                tables: vec![Table::new(
                    {
                        let mut map = FxHashMap::default();
                        map.insert(vec![0, 0, 0, 0], {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(1);
                            set
                        });
                        map.insert(vec![0, 0, 0, 1], {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(2);
                            set
                        });
                        map.insert(vec![0, 0, 1, 0], {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(3);
                            set
                        });
                        map
                    },
                    {
                        let mut set = Set::with_capacity(5);
                        set.insert(0);
                        set.insert(4);
                        set
                    },
                )],
                ..Default::default()
            },
            ..Default::default()
        };
        let expression = SetReduceExpression::Table(
            SetReduceOperator::Union,
            5,
            0,
            vec![
                ArgumentExpression::Element(ElementExpression::Constant(0)),
                ArgumentExpression::Element(ElementExpression::Constant(0)),
                ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Constant({
                    let mut set = Set::with_capacity(2);
                    set.insert(0);
                    set.insert(1);
                    set
                }))),
                ArgumentExpression::Vector(VectorExpression::Reference(
                    ReferenceExpression::Constant(vec![0, 1]),
                )),
            ],
        );
        assert_eq!(
            expression.eval(&state, &mut function_cache, &state_functions, &registry),
            {
                let mut set = Set::with_capacity(5);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set.insert(3);
                set.insert(4);
                set
            }
        );
    }

    #[test]
    fn constant_simplify() {
        let registry = TableRegistry::default();
        let expression = SetReduceExpression::Constant(Set::with_capacity(3));
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn table_1d_element_constant_simplify() {
        let registry = TableRegistry {
            set_tables: TableData {
                tables_1d: vec![Table1D::new(vec![
                    {
                        let mut set = Set::with_capacity(3);
                        set.insert(0);
                        set.insert(1);
                        set
                    },
                    {
                        let mut set = Set::with_capacity(3);
                        set.insert(1);
                        set.insert(2);
                        set
                    },
                ])],
                ..Default::default()
            },
            ..Default::default()
        };
        let expression = SetReduceExpression::Table1D(
            SetReduceOperator::Intersection,
            3,
            0,
            Box::new(ArgumentExpression::Element(ElementExpression::Constant(0))),
        );
        assert_eq!(
            expression.simplify(&registry),
            SetReduceExpression::Constant({
                let mut set = Set::with_capacity(3);
                set.insert(0);
                set.insert(1);
                set
            })
        );
    }

    #[test]
    fn table_1d_set_constant_simplify() {
        let registry = TableRegistry {
            set_tables: TableData {
                tables_1d: vec![Table1D::new(vec![
                    {
                        let mut set = Set::with_capacity(3);
                        set.insert(0);
                        set.insert(1);
                        set
                    },
                    {
                        let mut set = Set::with_capacity(3);
                        set.insert(1);
                        set.insert(2);
                        set
                    },
                ])],
                ..Default::default()
            },
            ..Default::default()
        };
        let expression = SetReduceExpression::Table1D(
            SetReduceOperator::Intersection,
            3,
            0,
            Box::new(ArgumentExpression::Set(SetExpression::Reference(
                ReferenceExpression::Constant({
                    let mut set = Set::with_capacity(2);
                    set.insert(0);
                    set.insert(1);
                    set
                }),
            ))),
        );
        assert_eq!(
            expression.simplify(&registry),
            SetReduceExpression::Constant({
                let mut set = Set::with_capacity(3);
                set.insert(1);
                set
            })
        );
    }

    #[test]
    fn table_1d_vector_constant_simplify() {
        let registry = TableRegistry {
            set_tables: TableData {
                tables_1d: vec![Table1D::new(vec![
                    {
                        let mut set = Set::with_capacity(3);
                        set.insert(0);
                        set.insert(1);
                        set
                    },
                    {
                        let mut set = Set::with_capacity(3);
                        set.insert(1);
                        set.insert(2);
                        set
                    },
                ])],
                ..Default::default()
            },
            ..Default::default()
        };
        let expression = SetReduceExpression::Table1D(
            SetReduceOperator::Intersection,
            3,
            0,
            Box::new(ArgumentExpression::Vector(VectorExpression::Reference(
                ReferenceExpression::Constant(vec![0, 1]),
            ))),
        );
        assert_eq!(
            expression.simplify(&registry),
            SetReduceExpression::Constant({
                let mut set = Set::with_capacity(3);
                set.insert(1);
                set
            })
        );
    }

    #[test]
    fn table_1d_simplify() {
        let registry = TableRegistry {
            set_tables: TableData {
                tables_1d: vec![Table1D::new(vec![
                    {
                        let mut set = Set::with_capacity(3);
                        set.insert(0);
                        set.insert(1);
                        set
                    },
                    {
                        let mut set = Set::with_capacity(3);
                        set.insert(1);
                        set.insert(2);
                        set
                    },
                ])],
                ..Default::default()
            },
            ..Default::default()
        };
        let expression = SetReduceExpression::Table1D(
            SetReduceOperator::Intersection,
            3,
            0,
            Box::new(ArgumentExpression::Element(ElementExpression::If(
                Box::new(Condition::Constant(true)),
                Box::new(ElementExpression::Variable(0)),
                Box::new(ElementExpression::Constant(0)),
            ))),
        );
        assert_eq!(
            expression.simplify(&registry),
            SetReduceExpression::Table1D(
                SetReduceOperator::Intersection,
                3,
                0,
                Box::new(ArgumentExpression::Element(ElementExpression::Variable(0))),
            )
        );
    }

    #[test]
    fn table_2d_element_constant_simplify() {
        let registry = TableRegistry {
            set_tables: TableData {
                tables_2d: vec![Table2D::new(vec![
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(1);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(2);
                            set
                        },
                    ],
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(3);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(4);
                            set
                        },
                    ],
                ])],
                ..Default::default()
            },
            ..Default::default()
        };
        let expression = SetReduceExpression::Table2D(
            SetReduceOperator::Union,
            5,
            0,
            Box::new(ArgumentExpression::Element(ElementExpression::Constant(0))),
            Box::new(ArgumentExpression::Element(ElementExpression::Constant(0))),
        );
        assert_eq!(
            expression.simplify(&registry),
            SetReduceExpression::Constant({
                let mut set = Set::with_capacity(5);
                set.insert(0);
                set.insert(1);
                set
            })
        );
    }

    #[test]
    fn table_2d_element_set_constant_simplify() {
        let registry = TableRegistry {
            set_tables: TableData {
                tables_2d: vec![Table2D::new(vec![
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(1);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(2);
                            set
                        },
                    ],
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(3);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(4);
                            set
                        },
                    ],
                ])],
                ..Default::default()
            },
            ..Default::default()
        };
        let expression = SetReduceExpression::Table2D(
            SetReduceOperator::Union,
            5,
            0,
            Box::new(ArgumentExpression::Element(ElementExpression::Constant(0))),
            Box::new(ArgumentExpression::Set(SetExpression::Reference(
                ReferenceExpression::Constant({
                    let mut set = Set::with_capacity(2);
                    set.insert(0);
                    set.insert(1);
                    set
                }),
            ))),
        );
        assert_eq!(
            expression.simplify(&registry),
            SetReduceExpression::Constant({
                let mut set = Set::with_capacity(5);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set
            })
        );
    }

    #[test]
    fn table_2d_element_vector_constant_simplify() {
        let registry = TableRegistry {
            set_tables: TableData {
                tables_2d: vec![Table2D::new(vec![
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(1);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(2);
                            set
                        },
                    ],
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(3);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(4);
                            set
                        },
                    ],
                ])],
                ..Default::default()
            },
            ..Default::default()
        };
        let expression = SetReduceExpression::Table2D(
            SetReduceOperator::Union,
            5,
            0,
            Box::new(ArgumentExpression::Element(ElementExpression::Constant(0))),
            Box::new(ArgumentExpression::Vector(VectorExpression::Reference(
                ReferenceExpression::Constant(vec![0, 1]),
            ))),
        );
        assert_eq!(
            expression.simplify(&registry),
            SetReduceExpression::Constant({
                let mut set = Set::with_capacity(5);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set
            })
        );
    }

    #[test]
    fn table_2d_set_element_constant_simplify() {
        let registry = TableRegistry {
            set_tables: TableData {
                tables_2d: vec![Table2D::new(vec![
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(1);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(2);
                            set
                        },
                    ],
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(3);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(4);
                            set
                        },
                    ],
                ])],
                ..Default::default()
            },
            ..Default::default()
        };
        let expression = SetReduceExpression::Table2D(
            SetReduceOperator::Union,
            5,
            0,
            Box::new(ArgumentExpression::Set(SetExpression::Reference(
                ReferenceExpression::Constant({
                    let mut set = Set::with_capacity(2);
                    set.insert(0);
                    set.insert(1);
                    set
                }),
            ))),
            Box::new(ArgumentExpression::Element(ElementExpression::Constant(0))),
        );
        assert_eq!(
            expression.simplify(&registry),
            SetReduceExpression::Constant({
                let mut set = Set::with_capacity(5);
                set.insert(0);
                set.insert(1);
                set.insert(3);
                set
            })
        );
    }

    #[test]
    fn table_2d_vector_element_constant_simplify() {
        let registry = TableRegistry {
            set_tables: TableData {
                tables_2d: vec![Table2D::new(vec![
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(1);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(2);
                            set
                        },
                    ],
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(3);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(4);
                            set
                        },
                    ],
                ])],
                ..Default::default()
            },
            ..Default::default()
        };
        let expression = SetReduceExpression::Table2D(
            SetReduceOperator::Union,
            5,
            0,
            Box::new(ArgumentExpression::Vector(VectorExpression::Reference(
                ReferenceExpression::Constant(vec![0, 1]),
            ))),
            Box::new(ArgumentExpression::Element(ElementExpression::Constant(0))),
        );
        assert_eq!(
            expression.simplify(&registry),
            SetReduceExpression::Constant({
                let mut set = Set::with_capacity(5);
                set.insert(0);
                set.insert(1);
                set.insert(3);
                set
            })
        );
    }

    #[test]
    fn table_2d_set_constant_simplify() {
        let registry = TableRegistry {
            set_tables: TableData {
                tables_2d: vec![Table2D::new(vec![
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(1);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(2);
                            set
                        },
                    ],
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(3);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(4);
                            set
                        },
                    ],
                ])],
                ..Default::default()
            },
            ..Default::default()
        };
        let expression = SetReduceExpression::Table2D(
            SetReduceOperator::Union,
            5,
            0,
            Box::new(ArgumentExpression::Set(SetExpression::Reference(
                ReferenceExpression::Constant({
                    let mut set = Set::with_capacity(2);
                    set.insert(0);
                    set.insert(1);
                    set
                }),
            ))),
            Box::new(ArgumentExpression::Set(SetExpression::Reference(
                ReferenceExpression::Constant({
                    let mut set = Set::with_capacity(2);
                    set.insert(0);
                    set.insert(1);
                    set
                }),
            ))),
        );
        assert_eq!(
            expression.simplify(&registry),
            SetReduceExpression::Constant({
                let mut set = Set::with_capacity(5);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set.insert(3);
                set.insert(4);
                set
            })
        );
    }

    #[test]
    fn table_2d_set_vector_constant_simplify() {
        let registry = TableRegistry {
            set_tables: TableData {
                tables_2d: vec![Table2D::new(vec![
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(1);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(2);
                            set
                        },
                    ],
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(3);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(4);
                            set
                        },
                    ],
                ])],
                ..Default::default()
            },
            ..Default::default()
        };
        let expression = SetReduceExpression::Table2D(
            SetReduceOperator::Union,
            5,
            0,
            Box::new(ArgumentExpression::Set(SetExpression::Reference(
                ReferenceExpression::Constant({
                    let mut set = Set::with_capacity(2);
                    set.insert(0);
                    set.insert(1);
                    set
                }),
            ))),
            Box::new(ArgumentExpression::Vector(VectorExpression::Reference(
                ReferenceExpression::Constant(vec![0, 1]),
            ))),
        );
        assert_eq!(
            expression.simplify(&registry),
            SetReduceExpression::Constant({
                let mut set = Set::with_capacity(5);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set.insert(3);
                set.insert(4);
                set
            })
        );
    }

    #[test]
    fn table_2d_vector_set_constant_simplify() {
        let registry = TableRegistry {
            set_tables: TableData {
                tables_2d: vec![Table2D::new(vec![
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(1);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(2);
                            set
                        },
                    ],
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(3);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(4);
                            set
                        },
                    ],
                ])],
                ..Default::default()
            },
            ..Default::default()
        };
        let expression = SetReduceExpression::Table2D(
            SetReduceOperator::Union,
            5,
            0,
            Box::new(ArgumentExpression::Vector(VectorExpression::Reference(
                ReferenceExpression::Constant(vec![0, 1]),
            ))),
            Box::new(ArgumentExpression::Set(SetExpression::Reference(
                ReferenceExpression::Constant({
                    let mut set = Set::with_capacity(2);
                    set.insert(0);
                    set.insert(1);
                    set
                }),
            ))),
        );
        assert_eq!(
            expression.simplify(&registry),
            SetReduceExpression::Constant({
                let mut set = Set::with_capacity(5);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set.insert(3);
                set.insert(4);
                set
            })
        );
    }

    #[test]
    fn table_2d_vector_constant_simplify() {
        let registry = TableRegistry {
            set_tables: TableData {
                tables_2d: vec![Table2D::new(vec![
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(1);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(2);
                            set
                        },
                    ],
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(3);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(4);
                            set
                        },
                    ],
                ])],
                ..Default::default()
            },
            ..Default::default()
        };
        let expression = SetReduceExpression::Table2D(
            SetReduceOperator::Union,
            5,
            0,
            Box::new(ArgumentExpression::Vector(VectorExpression::Reference(
                ReferenceExpression::Constant(vec![0, 1]),
            ))),
            Box::new(ArgumentExpression::Vector(VectorExpression::Reference(
                ReferenceExpression::Constant(vec![0, 1]),
            ))),
        );
        assert_eq!(
            expression.simplify(&registry),
            SetReduceExpression::Constant({
                let mut set = Set::with_capacity(5);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set.insert(3);
                set.insert(4);
                set
            })
        );
    }

    #[test]
    fn table_2d_simplify() {
        let registry = TableRegistry {
            set_tables: TableData {
                tables_2d: vec![Table2D::new(vec![
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(1);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(2);
                            set
                        },
                    ],
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(3);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(4);
                            set
                        },
                    ],
                ])],
                ..Default::default()
            },
            ..Default::default()
        };
        let expression = SetReduceExpression::Table2D(
            SetReduceOperator::Union,
            5,
            0,
            Box::new(ArgumentExpression::Element(ElementExpression::If(
                Box::new(Condition::Constant(true)),
                Box::new(ElementExpression::Variable(0)),
                Box::new(ElementExpression::Constant(0)),
            ))),
            Box::new(ArgumentExpression::Element(ElementExpression::If(
                Box::new(Condition::Constant(true)),
                Box::new(ElementExpression::Variable(0)),
                Box::new(ElementExpression::Constant(0)),
            ))),
        );
        assert_eq!(
            expression.simplify(&registry),
            SetReduceExpression::Table2D(
                SetReduceOperator::Union,
                5,
                0,
                Box::new(ArgumentExpression::Element(ElementExpression::Variable(0))),
                Box::new(ArgumentExpression::Element(ElementExpression::Variable(0)))
            )
        );
    }

    #[test]
    fn table_3d_constant_simplify() {
        let registry = TableRegistry {
            set_tables: TableData {
                tables_3d: vec![Table3D::new(vec![vec![
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(1);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(2);
                            set
                        },
                    ],
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(3);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(4);
                            set
                        },
                    ],
                ]])],
                ..Default::default()
            },
            ..Default::default()
        };
        let expression = SetReduceExpression::Table3D(
            SetReduceOperator::Union,
            5,
            0,
            Box::new(ArgumentExpression::Element(ElementExpression::Constant(0))),
            Box::new(ArgumentExpression::Set(SetExpression::Reference(
                ReferenceExpression::Constant({
                    let mut set = Set::with_capacity(2);
                    set.insert(0);
                    set.insert(1);
                    set
                }),
            ))),
            Box::new(ArgumentExpression::Vector(VectorExpression::Reference(
                ReferenceExpression::Constant(vec![0, 1]),
            ))),
        );
        assert_eq!(
            expression.simplify(&registry),
            SetReduceExpression::Constant({
                let mut set = Set::with_capacity(5);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set.insert(3);
                set.insert(4);
                set
            })
        );
    }

    #[test]
    fn table_3d_simplify() {
        let registry = TableRegistry {
            set_tables: TableData {
                tables_3d: vec![Table3D::new(vec![vec![
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(1);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(2);
                            set
                        },
                    ],
                    vec![
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(3);
                            set
                        },
                        {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(4);
                            set
                        },
                    ],
                ]])],
                ..Default::default()
            },
            ..Default::default()
        };
        let expression = SetReduceExpression::Table3D(
            SetReduceOperator::Union,
            5,
            0,
            Box::new(ArgumentExpression::Element(ElementExpression::If(
                Box::new(Condition::Constant(true)),
                Box::new(ElementExpression::Variable(0)),
                Box::new(ElementExpression::Constant(0)),
            ))),
            Box::new(ArgumentExpression::Set(SetExpression::Reference(
                ReferenceExpression::Constant({
                    let mut set = Set::with_capacity(2);
                    set.insert(0);
                    set.insert(1);
                    set
                }),
            ))),
            Box::new(ArgumentExpression::Vector(VectorExpression::Reference(
                ReferenceExpression::Constant(vec![0, 1]),
            ))),
        );
        assert_eq!(
            expression.simplify(&registry),
            SetReduceExpression::Table3D(
                SetReduceOperator::Union,
                5,
                0,
                Box::new(ArgumentExpression::Element(ElementExpression::Variable(0),)),
                Box::new(ArgumentExpression::Set(SetExpression::Reference(
                    ReferenceExpression::Constant({
                        let mut set = Set::with_capacity(2);
                        set.insert(0);
                        set.insert(1);
                        set
                    }),
                ))),
                Box::new(ArgumentExpression::Vector(VectorExpression::Reference(
                    ReferenceExpression::Constant(vec![0, 1]),
                ))),
            )
        );
    }

    #[test]
    fn table_constant_simplify() {
        let registry = TableRegistry {
            set_tables: TableData {
                tables: vec![Table::new(
                    {
                        let mut map = FxHashMap::default();
                        map.insert(vec![0, 0, 0, 0], {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(1);
                            set
                        });
                        map.insert(vec![0, 0, 0, 1], {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(2);
                            set
                        });
                        map.insert(vec![0, 0, 1, 0], {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(3);
                            set
                        });
                        map
                    },
                    {
                        let mut set = Set::with_capacity(5);
                        set.insert(0);
                        set.insert(4);
                        set
                    },
                )],
                ..Default::default()
            },
            ..Default::default()
        };
        let expression = SetReduceExpression::Table(
            SetReduceOperator::Union,
            5,
            0,
            vec![
                ArgumentExpression::Element(ElementExpression::Constant(0)),
                ArgumentExpression::Element(ElementExpression::Constant(0)),
                ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Constant({
                    let mut set = Set::with_capacity(2);
                    set.insert(0);
                    set.insert(1);
                    set
                }))),
                ArgumentExpression::Vector(VectorExpression::Reference(
                    ReferenceExpression::Constant(vec![0, 1]),
                )),
            ],
        );
        assert_eq!(
            expression.simplify(&registry),
            SetReduceExpression::Constant({
                let mut set = Set::with_capacity(5);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set.insert(3);
                set.insert(4);
                set
            })
        );
    }

    #[test]
    fn table_simplify() {
        let registry = TableRegistry {
            set_tables: TableData {
                tables: vec![Table::new(
                    {
                        let mut map = FxHashMap::default();
                        map.insert(vec![0, 0, 0, 0], {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(1);
                            set
                        });
                        map.insert(vec![0, 0, 0, 1], {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(2);
                            set
                        });
                        map.insert(vec![0, 0, 1, 0], {
                            let mut set = Set::with_capacity(5);
                            set.insert(0);
                            set.insert(3);
                            set
                        });
                        map
                    },
                    {
                        let mut set = Set::with_capacity(5);
                        set.insert(0);
                        set.insert(4);
                        set
                    },
                )],
                ..Default::default()
            },
            ..Default::default()
        };
        let expression = SetReduceExpression::Table(
            SetReduceOperator::Union,
            5,
            0,
            vec![
                ArgumentExpression::Element(ElementExpression::If(
                    Box::new(Condition::Constant(true)),
                    Box::new(ElementExpression::Variable(0)),
                    Box::new(ElementExpression::Constant(0)),
                )),
                ArgumentExpression::Element(ElementExpression::If(
                    Box::new(Condition::Constant(true)),
                    Box::new(ElementExpression::Variable(0)),
                    Box::new(ElementExpression::Constant(0)),
                )),
                ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Constant({
                    let mut set = Set::with_capacity(2);
                    set.insert(0);
                    set.insert(1);
                    set
                }))),
                ArgumentExpression::Vector(VectorExpression::Reference(
                    ReferenceExpression::Constant(vec![0, 1]),
                )),
            ],
        );
        assert_eq!(
            expression.simplify(&registry),
            SetReduceExpression::Table(
                SetReduceOperator::Union,
                5,
                0,
                vec![
                    ArgumentExpression::Element(ElementExpression::Variable(0),),
                    ArgumentExpression::Element(ElementExpression::Variable(0),),
                    ArgumentExpression::Set(SetExpression::Reference(
                        ReferenceExpression::Constant({
                            let mut set = Set::with_capacity(2);
                            set.insert(0);
                            set.insert(1);
                            set
                        })
                    )),
                    ArgumentExpression::Vector(VectorExpression::Reference(
                        ReferenceExpression::Constant(vec![0, 1]),
                    )),
                ],
            )
        );
    }
}
