use crate::expression;
use crate::state::DPState;
use crate::table_registry;

/// Condition with element parameters.
#[derive(Debug, PartialEq, Clone, Default)]
pub struct GroundedCondition {
    /// Pairs of parameters and indices of set variables.
    pub elements_in_set_variable: Vec<(usize, usize)>,
    /// Pairs of parameters and indices of vector variables.
    pub elements_in_vector_variable: Vec<(usize, usize)>,
    /// Condition
    pub condition: expression::Condition,
}

impl GroundedCondition {
    /// Returns true if the condition is satisfied, false if the condition is not satisifed, and None if an parameter is not included in the corresponding set or vector variable.
    ///
    /// # Panics
    ///
    /// if the cost of the transition state is used or a min/max reduce operation is performed on an empty set or vector.
    pub fn is_satisfied<U: DPState>(
        &self,
        state: &U,
        registry: &table_registry::TableRegistry,
    ) -> Option<bool> {
        for (i, v) in &self.elements_in_set_variable {
            if !state.get_set_variable(*i).contains(*v) {
                return None;
            }
        }
        for (i, v) in &self.elements_in_vector_variable {
            if !state.get_vector_variable(*i).contains(v) {
                return None;
            }
        }
        Some(self.condition.eval(state, registry))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expression::*;
    use crate::state;
    use crate::table;
    use crate::table_data;
    use crate::variable_type;
    use rustc_hash::FxHashMap;

    fn generate_registry() -> table_registry::TableRegistry {
        let tables_1d = vec![table::Table1D::new(vec![true, false])];
        let mut name_to_table_1d = FxHashMap::default();
        name_to_table_1d.insert(String::from("b1"), 0);

        let tables_2d = vec![table::Table2D::new(vec![
            vec![false, true],
            vec![true, false],
        ])];
        let mut name_to_table_2d = FxHashMap::default();
        name_to_table_2d.insert(String::from("b2"), 0);

        table_registry::TableRegistry {
            bool_tables: table_data::TableData {
                tables_1d,
                name_to_table_1d,
                tables_2d,
                name_to_table_2d,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn is_satisfied_test() {
        let registry = generate_registry();
        let mut s0 = variable_type::Set::with_capacity(2);
        s0.insert(0);
        let state = state::State {
            signature_variables: state::SignatureVariables {
                set_variables: vec![s0],
                vector_variables: vec![vec![1]],
                element_variables: vec![0],
                ..Default::default()
            },
            ..Default::default()
        };

        let condition = GroundedCondition {
            condition: Condition::Set(Box::new(SetCondition::IsIn(
                ElementExpression::Variable(0),
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            ))),
            ..Default::default()
        };
        assert_eq!(condition.is_satisfied(&state, &registry), Some(true));

        let condition = GroundedCondition {
            condition: Condition::Set(Box::new(SetCondition::IsIn(
                ElementExpression::Constant(0),
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            ))),
            elements_in_set_variable: vec![],
            elements_in_vector_variable: vec![(0, 0)],
        };
        assert!(condition.is_satisfied(&state, &registry).is_none());

        let condition = GroundedCondition {
            condition: Condition::Set(Box::new(SetCondition::IsIn(
                ElementExpression::Constant(1),
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            ))),
            elements_in_set_variable: vec![],
            elements_in_vector_variable: vec![(0, 1)],
        };
        assert_eq!(condition.is_satisfied(&state, &registry), Some(false));
    }
}
