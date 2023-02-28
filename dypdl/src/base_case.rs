use crate::grounded_condition::GroundedCondition;
use crate::state;
use crate::table_registry;

/// Base case.
#[derive(Debug, PartialEq, Clone, Default)]
pub struct BaseCase(Vec<GroundedCondition>);

impl From<Vec<GroundedCondition>> for BaseCase {
    #[inline]
    fn from(conditions: Vec<GroundedCondition>) -> Self {
        Self(conditions)
    }
}

impl From<BaseCase> for Vec<GroundedCondition> {
    #[inline]
    fn from(base_case: BaseCase) -> Self {
        base_case.0
    }
}

impl BaseCase {
    /// Creates a new base case.
    #[inline]
    pub fn new(conditions: Vec<GroundedCondition>) -> BaseCase {
        BaseCase(conditions)
    }

    /// Returns true if the base case is satisfied and false otherwise.
    ///
    /// # Panics
    ///
    /// Panics if the cost of the transition state is used or a min/max reduce operation is performed on an empty set or vector.
    #[inline]
    pub fn is_satisfied<U: state::StateInterface>(
        &self,
        state: &U,
        registry: &table_registry::TableRegistry,
    ) -> bool {
        self.0
            .iter()
            .all(|x| x.is_satisfied(state, registry).unwrap_or(true))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expression::*;
    use crate::variable_type;

    #[test]
    fn to_base_case() {
        let conditions = vec![
            GroundedCondition {
                condition: Condition::Constant(true),
                ..Default::default()
            },
            GroundedCondition {
                condition: Condition::Constant(false),
                ..Default::default()
            },
        ];
        let base_case = BaseCase::from(conditions.clone());
        assert_eq!(base_case.0, conditions);
    }

    #[test]
    fn from_base_case() {
        let conditions = vec![
            GroundedCondition {
                condition: Condition::Constant(true),
                ..Default::default()
            },
            GroundedCondition {
                condition: Condition::Constant(false),
                ..Default::default()
            },
        ];
        let base_case = BaseCase::from(conditions.clone());
        assert_eq!(Vec::<GroundedCondition>::from(base_case), conditions);
    }

    #[test]
    fn is_satisfied() {
        let mut s0 = variable_type::Set::with_capacity(2);
        s0.insert(0);
        s0.insert(1);
        let state = state::State {
            signature_variables: state::SignatureVariables {
                set_variables: vec![s0],
                integer_variables: vec![1],
                ..Default::default()
            },
            ..Default::default()
        };

        let registry = table_registry::TableRegistry::default();

        let base_case = BaseCase::new(vec![GroundedCondition {
            condition: Condition::Constant(true),
            ..Default::default()
        }]);
        assert!(base_case.is_satisfied(&state, &registry));

        let base_case = BaseCase::new(vec![
            GroundedCondition {
                condition: Condition::Constant(true),
                ..Default::default()
            },
            GroundedCondition {
                condition: Condition::Constant(false),
                ..Default::default()
            },
        ]);
        assert!(!base_case.is_satisfied(&state, &registry));
    }
}
