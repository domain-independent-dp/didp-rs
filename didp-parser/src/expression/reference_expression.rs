use super::element_expression::TableExpression;
use crate::state::State;
use crate::table_data::TableData;
use crate::table_registry::TableRegistry;

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum ReferenceExpression<T: Clone> {
    Constant(T),
    Variable(usize),
    Table(TableExpression<T>),
}

impl<T: Clone> ReferenceExpression<T> {
    pub fn eval<'a>(
        &'a self,
        state: &'a State,
        registry: &'a TableRegistry,
        variables: &'a [T],
        tables: &'a TableData<T>,
    ) -> &'a T {
        match self {
            Self::Constant(value) => value,
            Self::Variable(i) => &variables[*i],
            Self::Table(table) => table.eval(state, registry, tables),
        }
    }

    pub fn simplify(
        &self,
        registry: &TableRegistry,
        tables: &TableData<T>,
    ) -> ReferenceExpression<T> {
        match self {
            Self::Table(table) => match table.simplify(registry, &tables) {
                TableExpression::Constant(value) => Self::Constant(value),
                expression => Self::Table(expression),
            },
            _ => self.clone(),
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
    use std::collections::HashMap;
    use std::rc::Rc;

    fn generate_registry() -> TableRegistry {
        let mut name_to_table_1d = HashMap::new();
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
            signature_variables: Rc::new(SignatureVariables {
                vector_variables: vec![vec![0, 2]],
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[test]
    fn constant_eval() {
        let state = generate_state();
        let registry = generate_registry();
        let expression = ReferenceExpression::Constant(vec![0, 1, 2]);
        assert_eq!(
            *expression.eval(
                &state,
                &registry,
                &state.signature_variables.vector_variables,
                &registry.vector_tables
            ),
            vec![0, 1, 2]
        );
    }

    #[test]
    fn variable_eval() {
        let state = generate_state();
        let registry = generate_registry();
        let expression = ReferenceExpression::Variable(0);
        assert_eq!(
            *expression.eval(
                &state,
                &registry,
                &state.signature_variables.vector_variables,
                &registry.vector_tables
            ),
            vec![0, 2]
        );
    }

    #[test]
    fn table_eval() {
        let state = generate_state();
        let registry = generate_registry();
        let expression =
            ReferenceExpression::Table(TableExpression::Table1D(0, ElementExpression::Constant(0)));
        assert_eq!(
            *expression.eval(
                &state,
                &registry,
                &state.signature_variables.vector_variables,
                &registry.vector_tables
            ),
            vec![0, 1]
        );
    }

    #[test]
    fn constant_simplify() {
        let registry = generate_registry();
        let expression = ReferenceExpression::Constant(vec![0, 1, 2]);
        assert_eq!(
            expression.simplify(&registry, &registry.vector_tables),
            expression
        );
    }

    #[test]
    fn variable_simplify() {
        let registry = generate_registry();
        let expression = ReferenceExpression::Variable(0);
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
