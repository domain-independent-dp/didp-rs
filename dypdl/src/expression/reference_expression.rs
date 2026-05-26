use super::table_expression::TableExpression;
use crate::state::StateInterface;
use crate::state_functions::{StateFunctionCache, StateFunctions};
use crate::table_data::TableData;
use crate::table_registry::TableRegistry;
use crate::variable_type::Set;

/// Expression referring to a constant or a variable.
#[derive(Debug, PartialEq, Clone)]
pub enum ReferenceExpression<T: Clone> {
    /// Constant.
    Constant(T),
    /// Variable index.
    Variable(usize),
    /// Constant in a table.
    Table(TableExpression<T>),
}

impl<T: Clone> ReferenceExpression<T> {
    /// Returns a simplified version by precomputation.
    ///
    /// # Panics
    ///
    /// Panics if a min/max reduce operation is performed on an empty set or vector.
    pub fn simplify(
        &self,
        registry: &TableRegistry,
        tables: &TableData<T>,
    ) -> ReferenceExpression<T> {
        match self {
            Self::Table(table) => match table.simplify(registry, tables) {
                TableExpression::Constant(value) => Self::Constant(value),
                expression => Self::Table(expression),
            },
            _ => self.clone(),
        }
    }
}

impl ReferenceExpression<Set> {
    /// Returns the evaluation result.
    ///
    /// # Panics
    ///
    /// Panics if the cost of the transition state is used or a min/max reduce operation is performed on an empty set or vector.
    pub fn eval<'a, S: StateInterface>(
        &'a self,
        state: &'a S,
        function_cache: &mut StateFunctionCache,
        state_functions: &StateFunctions,
        registry: &'a TableRegistry,
    ) -> &'a Set {
        match self {
            Self::Constant(value) => value,
            Self::Variable(i) => state.get_set_variable(*i),
            Self::Table(table) => table.eval(
                state,
                function_cache,
                state_functions,
                registry,
                &registry.set_tables,
            ),
        }
    }
}
