use super::table_expression::TableExpression;
use crate::state::StateInterface;
use crate::state_functions::{StateFunctionCache, StateFunctions};
use crate::table_data::TableData;
use crate::table_registry::TableRegistry;
use crate::variable_type::{Set, Vector};

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

impl ReferenceExpression<Vector> {
    /// Returns the evaluation result.
    ///
    /// # Panics
    ///
    /// Panics if the cost of the transition state is used or a min/max reduce operation is performed on an empty set or vector.
    pub fn eval<'a, U: StateInterface>(
        &'a self,
        state: &'a U,
        function_cache: &mut StateFunctionCache,
        state_functions: &StateFunctions,
        registry: &'a TableRegistry,
    ) -> &'a Vector {
        match self {
            Self::Constant(value) => value,
            Self::Variable(i) => state.get_vector_variable(*i),
            Self::Table(table) => table.eval(
                state,
                function_cache,
                state_functions,
                registry,
                &registry.vector_tables,
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::element_expression::*;
    use super::*;
    use crate::state::*;
    use crate::table::*;
    use crate::table_data::TableData;
    use rustc_hash::FxHashMap;

    fn generate_registry() -> TableRegistry {
        let mut name_to_table_1d = FxHashMap::default();
        name_to_table_1d.insert(String::from("t1"), 0);
        TableRegistry {
            vector_tables: TableData {
                tables_1d: vec![Table1D::new(vec![vec![0, 1]])],
                name_to_table_1d,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn generate_state() -> State {
        State {
            signature_variables: SignatureVariables {
                vector_variables: vec![vec![0, 2]],
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn constant_eval() {
        let state = generate_state();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = generate_registry();
        let expression = ReferenceExpression::Constant(vec![0, 1, 2]);
        assert_eq!(
            *expression.eval(&state, &mut function_cache, &state_functions, &registry,),
            vec![0, 1, 2]
        );
    }

    #[test]
    fn variable_eval() {
        let state = generate_state();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = generate_registry();
        let expression: ReferenceExpression<Vector> = ReferenceExpression::Variable(0);
        assert_eq!(
            *expression.eval(&state, &mut function_cache, &state_functions, &registry,),
            vec![0, 2]
        );
    }

    #[test]
    fn table_eval() {
        let state = generate_state();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = generate_registry();
        let expression: ReferenceExpression<Vector> =
            ReferenceExpression::Table(TableExpression::Table1D(0, ElementExpression::Constant(0)));
        assert_eq!(
            *expression.eval(&state, &mut function_cache, &state_functions, &registry,),
            vec![0, 1]
        );
    }

    #[test]
    fn constant_simplify() {
        let registry = generate_registry();
        let expression: ReferenceExpression<Vector> = ReferenceExpression::Constant(vec![0, 1, 2]);
        assert_eq!(
            expression.simplify(&registry, &registry.vector_tables),
            expression
        );
    }

    #[test]
    fn variable_simplify() {
        let registry = generate_registry();
        let expression: ReferenceExpression<Vector> = ReferenceExpression::Variable(0);
        assert_eq!(
            expression.simplify(&registry, &registry.vector_tables),
            expression
        );
    }

    #[test]
    fn table_simplify() {
        let registry = generate_registry();
        let expression =
            ReferenceExpression::Table(TableExpression::Table1D(0, ElementExpression::Constant(0)));
        assert_eq!(
            expression.simplify(&registry, &registry.vector_tables),
            ReferenceExpression::Constant(vec![0, 1])
        );
        let expression =
            ReferenceExpression::Table(TableExpression::Table1D(0, ElementExpression::Variable(0)));
        assert_eq!(
            expression.simplify(&registry, &registry.vector_tables),
            expression
        );
    }
}
