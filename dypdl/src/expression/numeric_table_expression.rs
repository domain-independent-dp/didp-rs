use super::argument_expression::ArgumentExpression;
use super::element_expression::ElementExpression;
use super::local_environment::LocalEnvironment;
use super::numeric_operator::ReduceOperator;
use super::reference_expression::ReferenceExpression;
use super::set_expression::SetExpression;
use super::substitute_local_variable::SubstituteLocalVariable;
use crate::state::StateInterface;
use crate::state_functions::{StateFunctionCache, StateFunctions};
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
    /// Reduce constants over sets in a table.
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
    /// Reduce constants over two sets in a 2D table.
    Table2DReduce(ReduceOperator, usize, SetExpression, SetExpression),
    /// Reduce constants over a set in a 2D table.
    Table2DReduceX(ReduceOperator, usize, SetExpression, ElementExpression),
    /// Reduce constants over a set in a 2D table.
    Table2DReduceY(ReduceOperator, usize, ElementExpression, SetExpression),
    /// Reduce constants over sets in a 3D table.
    Table3DReduce(
        ReduceOperator,
        usize,
        ArgumentExpression,
        ArgumentExpression,
        ArgumentExpression,
    ),
}

impl<T: Numeric> NumericTableExpression<T> {
    /// Converts a reduce whose body is a direct table lookup into a table reduce.
    ///
    /// Returns `None` when the local variable is not used as exactly one complete
    /// table index. In particular, repeated uses such as `table[x, x]` cannot be
    /// represented by the Cartesian-product semantics of table reductions.
    pub(super) fn reduce(
        &self,
        op: &ReduceOperator,
        set: &SetExpression,
        id: usize,
    ) -> Option<Self> {
        fn contains_local_variable(expression: &ElementExpression, id: usize) -> bool {
            expression.substitute_local_variable(id, usize::MAX) != *expression
        }

        fn argument(
            expression: &ElementExpression,
            set: &SetExpression,
            id: usize,
        ) -> Option<(ArgumentExpression, bool)> {
            if *expression == ElementExpression::LocalVariable(id) {
                Some((ArgumentExpression::Set(set.clone()), true))
            } else if contains_local_variable(expression, id) {
                None
            } else {
                Some((ArgumentExpression::Element(expression.clone()), false))
            }
        }

        match self {
            Self::Table(i, args) => {
                let mut found = false;
                let mut reduced_args = Vec::with_capacity(args.len());
                for expression in args {
                    let (arg, replaced) = argument(expression, set, id)?;
                    if replaced && found {
                        return None;
                    }
                    found |= replaced;
                    reduced_args.push(arg);
                }
                found.then(|| Self::TableReduce(op.clone(), *i, reduced_args))
            }
            Self::Table1D(i, x) if *x == ElementExpression::LocalVariable(id) => {
                Some(Self::Table1DReduce(op.clone(), *i, set.clone()))
            }
            Self::Table2D(i, x, y) => {
                let x_is_local = *x == ElementExpression::LocalVariable(id);
                let y_is_local = *y == ElementExpression::LocalVariable(id);
                match (x_is_local, y_is_local) {
                    (true, false) if !contains_local_variable(y, id) => {
                        Some(Self::Table2DReduceX(op.clone(), *i, set.clone(), y.clone()))
                    }
                    (false, true) if !contains_local_variable(x, id) => {
                        Some(Self::Table2DReduceY(op.clone(), *i, x.clone(), set.clone()))
                    }
                    _ => None,
                }
            }
            Self::Table3D(i, x, y, z) => {
                let mut found = false;
                let mut args = Vec::with_capacity(3);
                for expression in [x, y, z] {
                    let (arg, replaced) = argument(expression, set, id)?;
                    if replaced && found {
                        return None;
                    }
                    found |= replaced;
                    args.push(arg);
                }
                found.then(|| {
                    Self::Table3DReduce(
                        op.clone(),
                        *i,
                        args[0].clone(),
                        args[1].clone(),
                        args[2].clone(),
                    )
                })
            }
            _ => None,
        }
    }

    /// Returns the evaluation result.
    ///
    /// # Panics
    ///
    /// Panics if the cost of the transitioned state is used or an empty set is passed to a reduce operation or a min/max reduce operation is performed on an empty set.
    pub fn eval<U: StateInterface>(
        &self,
        state: &U,
        function_cache: &mut StateFunctionCache,
        local_environment: &mut LocalEnvironment,
        state_functions: &StateFunctions,
        registry: &TableRegistry,
        tables: &TableData<T>,
    ) -> T {
        match self {
            Self::Constant(value) => *value,
            Self::Table(i, args) => {
                let args: Vec<Element> = args
                    .iter()
                    .map(|x| {
                        x.eval_with_local_environment(
                            state,
                            function_cache,
                            local_environment,
                            state_functions,
                            registry,
                        )
                    })
                    .collect();
                tables.tables[*i].eval(&args)
            }
            Self::TableReduce(op, i, args) => {
                let args = ArgumentExpression::eval_args(
                    args.iter(),
                    state,
                    function_cache,
                    local_environment,
                    state_functions,
                    registry,
                );
                op.eval_iter(args.into_iter().map(|args| tables.tables[*i].eval(&args)))
                    .unwrap()
            }
            Self::Table1D(i, x) => tables.tables_1d[*i].eval(x.eval_with_local_environment(
                state,
                function_cache,
                local_environment,
                state_functions,
                registry,
            )),
            Self::Table2D(i, x, y) => tables.tables_2d[*i].eval(
                x.eval_with_local_environment(
                    state,
                    function_cache,
                    local_environment,
                    state_functions,
                    registry,
                ),
                y.eval_with_local_environment(
                    state,
                    function_cache,
                    local_environment,
                    state_functions,
                    registry,
                ),
            ),
            Self::Table3D(i, x, y, z) => tables.tables_3d[*i].eval(
                x.eval_with_local_environment(
                    state,
                    function_cache,
                    local_environment,
                    state_functions,
                    registry,
                ),
                y.eval_with_local_environment(
                    state,
                    function_cache,
                    local_environment,
                    state_functions,
                    registry,
                ),
                z.eval_with_local_environment(
                    state,
                    function_cache,
                    local_environment,
                    state_functions,
                    registry,
                ),
            ),
            Self::Table1DReduce(op, i, SetExpression::Reference(x)) => Self::reduce_table_1d(
                op,
                &tables.tables_1d[*i],
                x.eval(
                    state,
                    function_cache,
                    local_environment,
                    state_functions,
                    registry,
                )
                .ones(),
            ),
            Self::Table1DReduce(op, i, SetExpression::StateFunction(x)) => Self::reduce_table_1d(
                op,
                &tables.tables_1d[*i],
                function_cache
                    .get_set_value(*x, state, local_environment, state_functions, registry)
                    .ones(),
            ),
            Self::Table1DReduce(op, i, x) => Self::reduce_table_1d(
                op,
                &tables.tables_1d[*i],
                x.eval_with_local_environment(
                    state,
                    function_cache,
                    local_environment,
                    state_functions,
                    registry,
                )
                .ones(),
            ),
            Self::Table2DReduce(
                op,
                i,
                SetExpression::StateFunction(x),
                SetExpression::StateFunction(y),
            ) => {
                let (x, y) = function_cache.get_set_value_pair(
                    *x,
                    *y,
                    state,
                    local_environment,
                    state_functions,
                    registry,
                );
                Self::reduce_table_2d_set_y(op, &tables.tables_2d[*i], x.ones(), y)
            }
            Self::Table2DReduce(
                op,
                i,
                SetExpression::StateFunction(x),
                SetExpression::Reference(y),
            ) => {
                let y = y.eval(
                    state,
                    function_cache,
                    local_environment,
                    state_functions,
                    registry,
                );
                let x = function_cache
                    .get_set_value(*x, state, local_environment, state_functions, registry)
                    .ones();
                Self::reduce_table_2d_set_y(op, &tables.tables_2d[*i], x, y)
            }
            Self::Table2DReduce(
                op,
                i,
                SetExpression::Reference(x),
                SetExpression::StateFunction(y),
            ) => {
                let x = x
                    .eval(
                        state,
                        function_cache,
                        local_environment,
                        state_functions,
                        registry,
                    )
                    .ones();
                let y = function_cache.get_set_value(
                    *y,
                    state,
                    local_environment,
                    state_functions,
                    registry,
                );
                Self::reduce_table_2d_set_y(op, &tables.tables_2d[*i], x, y)
            }
            Self::Table2DReduce(op, i, SetExpression::StateFunction(x), y) => {
                let y = y.eval_with_local_environment(
                    state,
                    function_cache,
                    local_environment,
                    state_functions,
                    registry,
                );
                let x = function_cache
                    .get_set_value(*x, state, local_environment, state_functions, registry)
                    .ones();
                Self::reduce_table_2d_set_y(op, &tables.tables_2d[*i], x, &y)
            }
            Self::Table2DReduce(op, i, x, SetExpression::StateFunction(y)) => {
                let x = x.eval_with_local_environment(
                    state,
                    function_cache,
                    local_environment,
                    state_functions,
                    registry,
                );
                let y = function_cache.get_set_value(
                    *y,
                    state,
                    local_environment,
                    state_functions,
                    registry,
                );
                Self::reduce_table_2d_set_y(op, &tables.tables_2d[*i], x.ones(), y)
            }
            Self::Table2DReduce(
                op,
                i,
                SetExpression::Reference(x),
                SetExpression::Reference(y),
            ) => {
                let x = x
                    .eval(
                        state,
                        function_cache,
                        local_environment,
                        state_functions,
                        registry,
                    )
                    .ones();
                let y = y.eval(
                    state,
                    function_cache,
                    local_environment,
                    state_functions,
                    registry,
                );
                Self::reduce_table_2d_set_y(op, &tables.tables_2d[*i], x, y)
            }
            Self::Table2DReduce(op, i, SetExpression::Reference(x), y) => {
                let y = y.eval_with_local_environment(
                    state,
                    function_cache,
                    local_environment,
                    state_functions,
                    registry,
                );
                let x = x
                    .eval(
                        state,
                        function_cache,
                        local_environment,
                        state_functions,
                        registry,
                    )
                    .ones();
                Self::reduce_table_2d_set_y(op, &tables.tables_2d[*i], x, &y)
            }
            Self::Table2DReduce(op, i, x, SetExpression::Reference(y)) => {
                let x = x.eval_with_local_environment(
                    state,
                    function_cache,
                    local_environment,
                    state_functions,
                    registry,
                );
                let y = y.eval(
                    state,
                    function_cache,
                    local_environment,
                    state_functions,
                    registry,
                );
                Self::reduce_table_2d_set_x(op, &tables.tables_2d[*i], &x, y.ones())
            }
            Self::Table2DReduce(op, i, x, y) => {
                let y = y.eval_with_local_environment(
                    state,
                    function_cache,
                    local_environment,
                    state_functions,
                    registry,
                );
                Self::reduce_table_2d_set_y(
                    op,
                    &tables.tables_2d[*i],
                    x.eval_with_local_environment(
                        state,
                        function_cache,
                        local_environment,
                        state_functions,
                        registry,
                    )
                    .ones(),
                    &y,
                )
            }
            Self::Table2DReduceX(op, i, SetExpression::Reference(x), y) => {
                let y = y.eval_with_local_environment(
                    state,
                    function_cache,
                    local_environment,
                    state_functions,
                    registry,
                );
                let x = x
                    .eval(
                        state,
                        function_cache,
                        local_environment,
                        state_functions,
                        registry,
                    )
                    .ones();
                Self::reduce_table_2d_x(op, &tables.tables_2d[*i], x, y)
            }
            Self::Table2DReduceX(op, i, x, y) => {
                let y = y.eval_with_local_environment(
                    state,
                    function_cache,
                    local_environment,
                    state_functions,
                    registry,
                );
                Self::reduce_table_2d_x(
                    op,
                    &tables.tables_2d[*i],
                    x.eval_with_local_environment(
                        state,
                        function_cache,
                        local_environment,
                        state_functions,
                        registry,
                    )
                    .ones(),
                    y,
                )
            }
            Self::Table2DReduceY(op, i, x, SetExpression::Reference(y)) => {
                let x = x.eval_with_local_environment(
                    state,
                    function_cache,
                    local_environment,
                    state_functions,
                    registry,
                );
                let y = y
                    .eval(
                        state,
                        function_cache,
                        local_environment,
                        state_functions,
                        registry,
                    )
                    .ones();
                Self::reduce_table_2d_y(op, &tables.tables_2d[*i], x, y)
            }
            Self::Table2DReduceY(op, i, x, y) => {
                let x = x.eval_with_local_environment(
                    state,
                    function_cache,
                    local_environment,
                    state_functions,
                    registry,
                );
                Self::reduce_table_2d_y(
                    op,
                    &tables.tables_2d[*i],
                    x,
                    y.eval_with_local_environment(
                        state,
                        function_cache,
                        local_environment,
                        state_functions,
                        registry,
                    )
                    .ones(),
                )
            }
            Self::Table3DReduce(op, i, x, y, z) => {
                let args = ArgumentExpression::eval_args(
                    [x, y, z].into_iter(),
                    state,
                    function_cache,
                    local_environment,
                    state_functions,
                    registry,
                );
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
    /// Panics if a min/max reduce operation is performed on an empty set.
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

    fn reduce_table_2d_set_x<I>(op: &ReduceOperator, table: &Table2D<T>, x: &Set, y: I) -> T
    where
        T: Num + PartialOrd + Sum + Product,
        I: Iterator<Item = Element>,
    {
        op.eval_iter(y.map(|y| op.eval_iter(x.ones().map(|x| table.eval(x, y))).unwrap()))
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
    use super::super::set_expression::SetElementOperation;
    use super::*;
    use crate::state::*;
    use crate::table;
    use crate::table_data::TableInterface;
    use crate::variable_type::Integer;
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
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn constant_eval() {
        let registry = generate_registry();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let mut local_environment = LocalEnvironment::default();
        let state = generate_state();
        let expression = NumericTableExpression::Constant(10);
        assert_eq!(
            expression.eval(
                &state,
                &mut function_cache,
                &mut local_environment,
                &state_functions,
                &registry,
                &registry.integer_tables
            ),
            10
        );
    }

    #[test]
    fn table_1d_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let mut local_environment = LocalEnvironment::default();
        let expression = NumericTableExpression::Table1D(0, ElementExpression::Constant(0));
        assert_eq!(
            expression.eval(
                &state,
                &mut function_cache,
                &mut local_environment,
                &state_functions,
                &registry,
                &registry.integer_tables
            ),
            10
        );
        let expression = NumericTableExpression::Table1D(0, ElementExpression::Constant(1));
        assert_eq!(
            expression.eval(
                &state,
                &mut function_cache,
                &mut local_environment,
                &state_functions,
                &registry,
                &registry.integer_tables
            ),
            20
        );
        let expression = NumericTableExpression::Table1D(0, ElementExpression::Constant(2));
        assert_eq!(
            expression.eval(
                &state,
                &mut function_cache,
                &mut local_environment,
                &state_functions,
                &registry,
                &registry.integer_tables
            ),
            30
        );
    }

    #[test]
    fn table_1d_sum_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let mut local_environment = LocalEnvironment::default();
        let expression = NumericTableExpression::Table1DReduce(
            ReduceOperator::Sum,
            0,
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert_eq!(
            expression.eval(
                &state,
                &mut function_cache,
                &mut local_environment,
                &state_functions,
                &registry,
                &registry.integer_tables
            ),
            40
        );
        let expression = NumericTableExpression::Table1DReduce(
            ReduceOperator::Sum,
            0,
            SetExpression::Reference(ReferenceExpression::Variable(1)),
        );
        assert_eq!(
            expression.eval(
                &state,
                &mut function_cache,
                &mut local_environment,
                &state_functions,
                &registry,
                &registry.integer_tables
            ),
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
            expression.eval(
                &state,
                &mut function_cache,
                &mut local_environment,
                &state_functions,
                &registry,
                &registry.integer_tables
            ),
            20
        );
    }

    #[test]
    fn table_1d_set_state_function_reduce_eval() {
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

        let mut registry = TableRegistry::default();
        let t = registry.add_table_1d("t", vec![1, 2]);
        assert!(t.is_ok());
        let t = t.unwrap();

        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![Set::with_capacity(3)],
                ..Default::default()
            },
            ..Default::default()
        };

        let expression = NumericTableExpression::Table1DReduce(ReduceOperator::Sum, t.id(), f);

        assert_eq!(
            expression.eval(
                &state,
                &mut StateFunctionCache::new(&state_functions),
                &mut LocalEnvironment::default(),
                &state_functions,
                &registry,
                &registry.integer_tables,
            ),
            3,
        );
    }

    #[test]
    fn table_2d_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let mut local_environment = LocalEnvironment::default();
        let expression = NumericTableExpression::Table2D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(1),
        );
        assert_eq!(
            expression.eval(
                &state,
                &mut function_cache,
                &mut local_environment,
                &state_functions,
                &registry,
                &registry.integer_tables
            ),
            20
        );
    }

    #[test]
    fn table_2d_sum_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let mut local_environment = LocalEnvironment::default();

        let expression = NumericTableExpression::Table2DReduce(
            ReduceOperator::Sum,
            0,
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Reference(ReferenceExpression::Variable(1)),
        );
        assert_eq!(
            expression.eval(
                &state,
                &mut function_cache,
                &mut local_environment,
                &state_functions,
                &registry,
                &registry.integer_tables
            ),
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
            expression.eval(
                &state,
                &mut function_cache,
                &mut local_environment,
                &state_functions,
                &registry,
                &registry.integer_tables
            ),
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
            expression.eval(
                &state,
                &mut function_cache,
                &mut local_environment,
                &state_functions,
                &registry,
                &registry.integer_tables
            ),
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
            expression.eval(
                &state,
                &mut function_cache,
                &mut local_environment,
                &state_functions,
                &registry,
                &registry.integer_tables
            ),
            180
        );
    }

    #[test]
    fn table_2d_sum_x_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let mut local_environment = LocalEnvironment::default();

        let expression = NumericTableExpression::Table2DReduceX(
            ReduceOperator::Sum,
            0,
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            ElementExpression::Constant(0),
        );
        assert_eq!(
            expression.eval(
                &state,
                &mut function_cache,
                &mut local_environment,
                &state_functions,
                &registry,
                &registry.integer_tables
            ),
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
            expression.eval(
                &state,
                &mut function_cache,
                &mut local_environment,
                &state_functions,
                &registry,
                &registry.integer_tables
            ),
            80
        );
    }

    #[test]
    fn table_2d_sum_y_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let mut local_environment = LocalEnvironment::default();

        let expression = NumericTableExpression::Table2DReduceY(
            ReduceOperator::Sum,
            0,
            ElementExpression::Constant(0),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert_eq!(
            expression.eval(
                &state,
                &mut function_cache,
                &mut local_environment,
                &state_functions,
                &registry,
                &registry.integer_tables
            ),
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
            expression.eval(
                &state,
                &mut function_cache,
                &mut local_environment,
                &state_functions,
                &registry,
                &registry.integer_tables
            ),
            40
        );
    }

    #[test]
    fn table_2d_set_state_function_element_reduce_x_eval() {
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

        let mut registry = TableRegistry::default();
        let t = registry.add_table_2d("t", vec![vec![1, 2], vec![3, 4]]);
        assert!(t.is_ok());
        let t = t.unwrap();

        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![Set::with_capacity(3)],
                ..Default::default()
            },
            ..Default::default()
        };

        let expression =
            NumericTableExpression::Table2DReduceX(ReduceOperator::Sum, t.id(), f, 1.into());

        assert_eq!(
            expression.eval(
                &state,
                &mut StateFunctionCache::new(&state_functions),
                &mut LocalEnvironment::default(),
                &state_functions,
                &registry,
                &registry.integer_tables,
            ),
            6,
        );
    }

    #[test]
    fn table_2d_element_set_state_function_reduce_y_eval() {
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

        let mut registry = TableRegistry::default();
        let t = registry.add_table_2d("t", vec![vec![1, 2], vec![3, 4]]);
        assert!(t.is_ok());
        let t = t.unwrap();

        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![Set::with_capacity(3)],
                ..Default::default()
            },
            ..Default::default()
        };

        let expression =
            NumericTableExpression::Table2DReduceY(ReduceOperator::Sum, t.id(), 1.into(), f);

        assert_eq!(
            expression.eval(
                &state,
                &mut StateFunctionCache::new(&state_functions),
                &mut LocalEnvironment::default(),
                &state_functions,
                &registry,
                &registry.integer_tables,
            ),
            7,
        );
    }

    #[test]
    fn table_2d_set_state_function_set_state_function_reduce_eval() {
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
        let g = state_functions.add_set_function("g", v.add(0).add(1));
        assert!(g.is_ok());
        let g = g.unwrap();

        let mut registry = TableRegistry::default();
        let t = registry.add_table_2d("t", vec![vec![1, 2], vec![3, 4]]);
        assert!(t.is_ok());
        let t = t.unwrap();

        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![Set::with_capacity(3)],
                ..Default::default()
            },
            ..Default::default()
        };

        let expression = NumericTableExpression::Table2DReduce(ReduceOperator::Sum, t.id(), f, g);

        assert_eq!(
            expression.eval(
                &state,
                &mut StateFunctionCache::new(&state_functions),
                &mut LocalEnvironment::default(),
                &state_functions,
                &registry,
                &registry.integer_tables,
            ),
            10,
        );
    }

    #[test]
    fn table_2d_set_state_function_set_reference_reduce_eval() {
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

        let mut registry = TableRegistry::default();
        let t = registry.add_table_2d("t", vec![vec![1, 2], vec![3, 4]]);
        assert!(t.is_ok());
        let t = t.unwrap();

        let set = state_metadata.create_set(ob, &[0, 1]);
        assert!(set.is_ok());
        let set = set.unwrap();

        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![Set::with_capacity(3)],
                ..Default::default()
            },
            ..Default::default()
        };

        let expression =
            NumericTableExpression::Table2DReduce(ReduceOperator::Sum, t.id(), f, set.into());

        assert_eq!(
            expression.eval(
                &state,
                &mut StateFunctionCache::new(&state_functions),
                &mut LocalEnvironment::default(),
                &state_functions,
                &registry,
                &registry.integer_tables,
            ),
            10,
        );
    }

    #[test]
    fn table_2d_set_reference_set_state_function_reduce_eval() {
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

        let mut registry = TableRegistry::default();
        let t = registry.add_table_2d("t", vec![vec![1, 2], vec![3, 4]]);
        assert!(t.is_ok());
        let t = t.unwrap();

        let set = state_metadata.create_set(ob, &[0, 1]);
        assert!(set.is_ok());
        let set = set.unwrap();

        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![Set::with_capacity(3)],
                ..Default::default()
            },
            ..Default::default()
        };

        let expression =
            NumericTableExpression::Table2DReduce(ReduceOperator::Sum, t.id(), set.into(), f);

        assert_eq!(
            expression.eval(
                &state,
                &mut StateFunctionCache::new(&state_functions),
                &mut LocalEnvironment::default(),
                &state_functions,
                &registry,
                &registry.integer_tables,
            ),
            10,
        );
    }

    #[test]
    fn table_3d_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let mut local_environment = LocalEnvironment::default();
        let expression = NumericTableExpression::Table3D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
        );
        assert_eq!(
            expression.eval(
                &state,
                &mut function_cache,
                &mut local_environment,
                &state_functions,
                &registry,
                &registry.integer_tables
            ),
            10
        );
    }

    #[test]
    fn table_3d_sum_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let mut local_environment = LocalEnvironment::default();
        let expression = NumericTableExpression::Table3DReduce(
            ReduceOperator::Sum,
            0,
            ArgumentExpression::Element(ElementExpression::Constant(0)),
            ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Variable(0))),
            ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Variable(1))),
        );
        assert_eq!(
            expression.eval(
                &state,
                &mut function_cache,
                &mut local_environment,
                &state_functions,
                &registry,
                &registry.integer_tables
            ),
            180
        );
    }

    #[test]
    fn table_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let mut local_environment = LocalEnvironment::default();
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
            expression.eval(
                &state,
                &mut function_cache,
                &mut local_environment,
                &state_functions,
                &registry,
                &registry.integer_tables
            ),
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
            expression.eval(
                &state,
                &mut function_cache,
                &mut local_environment,
                &state_functions,
                &registry,
                &registry.integer_tables
            ),
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
            expression.eval(
                &state,
                &mut function_cache,
                &mut local_environment,
                &state_functions,
                &registry,
                &registry.integer_tables
            ),
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
            expression.eval(
                &state,
                &mut function_cache,
                &mut local_environment,
                &state_functions,
                &registry,
                &registry.integer_tables
            ),
            400
        );
    }

    #[test]
    fn table_sum_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let mut local_environment = LocalEnvironment::default();
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
                ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Variable(1))),
            ],
        );
        assert_eq!(
            expression.eval(
                &state,
                &mut function_cache,
                &mut local_environment,
                &state_functions,
                &registry,
                &registry.integer_tables
            ),
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
            ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Constant({
                let mut set = Set::with_capacity(3);
                set.insert(0);
                set.insert(1);
                set
            }))),
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
            ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Variable(1))),
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
                ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Constant({
                    let mut set = Set::with_capacity(3);
                    set.insert(0);
                    set.insert(1);
                    set
                }))),
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
                ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Variable(0))),
            ],
        );
        assert_eq!(
            expression.simplify(&registry, &registry.integer_tables),
            expression
        );
    }

    #[test]
    fn reduce_direct_table_lookup() {
        let set = SetExpression::Reference(ReferenceExpression::Variable(0));
        let x = ElementExpression::LocalVariable(1);
        let op = ReduceOperator::Sum;

        let expression = NumericTableExpression::<Integer>::Table1D(2, x.clone());
        assert_eq!(
            expression.reduce(&op, &set, 1),
            Some(NumericTableExpression::Table1DReduce(
                op.clone(),
                2,
                set.clone(),
            ))
        );

        let expression = NumericTableExpression::<Integer>::Table2D(
            3,
            ElementExpression::Constant(0),
            x.clone(),
        );
        assert_eq!(
            expression.reduce(&op, &set, 1),
            Some(NumericTableExpression::Table2DReduceY(
                op.clone(),
                3,
                ElementExpression::Constant(0),
                set.clone(),
            ))
        );

        let expression = NumericTableExpression::<Integer>::Table3D(
            4,
            ElementExpression::Constant(0),
            x.clone(),
            ElementExpression::Constant(2),
        );
        assert_eq!(
            expression.reduce(&op, &set, 1),
            Some(NumericTableExpression::Table3DReduce(
                op.clone(),
                4,
                ArgumentExpression::Element(ElementExpression::Constant(0)),
                ArgumentExpression::Set(set.clone()),
                ArgumentExpression::Element(ElementExpression::Constant(2)),
            ))
        );

        let expression = NumericTableExpression::<Integer>::Table(
            5,
            vec![ElementExpression::Constant(0), x.clone()],
        );
        assert_eq!(
            expression.reduce(&op, &set, 1),
            Some(NumericTableExpression::TableReduce(
                op.clone(),
                5,
                vec![
                    ArgumentExpression::Element(ElementExpression::Constant(0)),
                    ArgumentExpression::Set(set.clone()),
                ],
            ))
        );

        let expression = NumericTableExpression::<Integer>::Table2D(6, x.clone(), x);
        assert_eq!(expression.reduce(&op, &set, 1), None);
    }
}
