use super::local_environment::LocalEnvironment;
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
    /// Resource variable index.
    ResourceVariable(usize),
    /// Constant in a table.
    Table(TableExpression<T>),
}

impl<T: Clone> ReferenceExpression<T> {
    /// Returns a simplified version by precomputation.
    ///
    /// # Panics
    ///
    /// Panics if a min/max reduce operation is performed on an empty set.
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
    /// Panics if the cost of the transition state is used or a min/max reduce operation is performed on an empty set.
    pub fn eval<'a, S: StateInterface>(
        &'a self,
        state: &'a S,
        function_cache: &mut StateFunctionCache,
        local_environment: &mut LocalEnvironment,
        state_functions: &StateFunctions,
        registry: &'a TableRegistry,
    ) -> &'a Set {
        match self {
            Self::Constant(value) => value,
            Self::Variable(i) => state.get_set_variable(*i),
            Self::ResourceVariable(i) => state.get_set_resource_variable(*i),
            Self::Table(table) => table.eval(
                state,
                function_cache,
                local_environment,
                state_functions,
                registry,
                &registry.set_tables,
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
            set_tables: TableData {
                tables_1d: vec![Table1D::new(vec![{
                    let mut set = Set::with_capacity(3);
                    set.insert(0);
                    set.insert(1);
                    set
                }])],
                name_to_table_1d,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn generate_state() -> State {
        State {
            signature_variables: SignatureVariables {
                set_variables: vec![{
                    let mut set = Set::with_capacity(3);
                    set.insert(0);
                    set.insert(2);
                    set
                }],
                ..Default::default()
            },
            resource_variables: ResourceVariables {
                set_variables: vec![{
                    let mut set = Set::with_capacity(3);
                    set.insert(1);
                    set.insert(2);
                    set
                }],
                ..Default::default()
            },
        }
    }

    #[test]
    fn constant_eval() {
        let state = generate_state();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let mut local_environment = LocalEnvironment::default();
        let registry = generate_registry();
        let expression = ReferenceExpression::Constant({
            let mut set = Set::with_capacity(3);
            set.insert(0);
            set
        });
        let mut expected = Set::with_capacity(3);
        expected.insert(0);
        assert_eq!(
            *expression.eval(
                &state,
                &mut function_cache,
                &mut local_environment,
                &state_functions,
                &registry,
            ),
            expected
        );
    }

    #[test]
    fn variable_eval() {
        let state = generate_state();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let mut local_environment = LocalEnvironment::default();
        let registry = generate_registry();
        let expression = ReferenceExpression::Variable(0);
        let mut expected = Set::with_capacity(3);
        expected.insert(0);
        expected.insert(2);
        assert_eq!(
            *expression.eval(
                &state,
                &mut function_cache,
                &mut local_environment,
                &state_functions,
                &registry,
            ),
            expected
        );
    }

    #[test]
    fn resource_variable_eval() {
        let state = generate_state();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let mut local_environment = LocalEnvironment::default();
        let registry = generate_registry();
        let expression = ReferenceExpression::ResourceVariable(0);
        let mut expected = Set::with_capacity(3);
        expected.insert(1);
        expected.insert(2);
        assert_eq!(
            *expression.eval(
                &state,
                &mut function_cache,
                &mut local_environment,
                &state_functions,
                &registry,
            ),
            expected
        );
    }

    #[test]
    fn table_eval() {
        let state = generate_state();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let mut local_environment = LocalEnvironment::default();
        let registry = generate_registry();
        let expression =
            ReferenceExpression::Table(TableExpression::Table1D(0, ElementExpression::Constant(0)));
        let mut expected = Set::with_capacity(3);
        expected.insert(0);
        expected.insert(1);
        assert_eq!(
            *expression.eval(
                &state,
                &mut function_cache,
                &mut local_environment,
                &state_functions,
                &registry,
            ),
            expected
        );
    }

    #[test]
    fn constant_simplify() {
        let registry = generate_registry();
        let expression = ReferenceExpression::Constant({
            let mut set = Set::with_capacity(3);
            set.insert(0);
            set
        });
        assert_eq!(
            expression.simplify(&registry, &registry.set_tables),
            expression
        );
    }

    #[test]
    fn variable_simplify() {
        let registry = generate_registry();
        let expression = ReferenceExpression::Variable(0);
        assert_eq!(
            expression.simplify(&registry, &registry.set_tables),
            expression
        );
    }

    #[test]
    fn resource_variable_simplify() {
        let registry = generate_registry();
        let expression = ReferenceExpression::ResourceVariable(0);
        assert_eq!(
            expression.simplify(&registry, &registry.set_tables),
            expression
        );
    }

    #[test]
    fn table_simplify_constant() {
        let registry = generate_registry();
        let expression = ReferenceExpression::<Set>::Table(TableExpression::Table1D(
            0,
            ElementExpression::Constant(0),
        ));
        let expected = ReferenceExpression::Constant({
            let mut set = Set::with_capacity(3);
            set.insert(0);
            set.insert(1);
            set
        });
        assert_eq!(
            expression.simplify(&registry, &registry.set_tables),
            expected
        );
    }

    #[test]
    fn table_simplify_variable() {
        let registry = generate_registry();
        let expression =
            ReferenceExpression::Table(TableExpression::Table1D(0, ElementExpression::Variable(0)));
        let expected =
            ReferenceExpression::Table(TableExpression::Table1D(0, ElementExpression::Variable(0)));
        assert_eq!(
            expression.simplify(&registry, &registry.set_tables),
            expected
        );
    }
}
