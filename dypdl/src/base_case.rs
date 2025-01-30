use crate::expression::Condition;
use crate::grounded_condition::GroundedCondition;
use crate::state::StateInterface;
use crate::state_functions::{StateFunctionCache, StateFunctions};
use crate::table_registry::TableRegistry;
use crate::transition::CostExpression;
use crate::variable_type::Numeric;

/// Base case.
#[derive(Debug, PartialEq, Clone, Default)]
pub struct BaseCase {
    pub conditions: Vec<GroundedCondition>,
    pub cost: Option<CostExpression>,
}

impl From<Vec<GroundedCondition>> for BaseCase {
    #[inline]
    fn from(conditions: Vec<GroundedCondition>) -> Self {
        Self {
            conditions,
            cost: None,
        }
    }
}

impl From<BaseCase> for Vec<Condition> {
    #[inline]
    fn from(base_case: BaseCase) -> Self {
        base_case
            .conditions
            .into_iter()
            .map(Condition::from)
            .collect()
    }
}

impl From<BaseCase> for (Vec<Condition>, Option<CostExpression>) {
    #[inline]
    fn from(base_case: BaseCase) -> Self {
        let conditions = base_case
            .conditions
            .into_iter()
            .map(Condition::from)
            .collect();
        (conditions, base_case.cost)
    }
}

impl BaseCase {
    /// Creates a new base case given conditions and a cost expression.
    #[inline]
    pub fn with_cost<T>(conditions: Vec<GroundedCondition>, cost: T) -> Self
    where
        CostExpression: From<T>,
    {
        Self {
            conditions,
            cost: Some(CostExpression::from(cost)),
        }
    }

    /// Returns true if the base case is satisfied and false otherwise.
    ///
    /// # Panics
    ///
    /// Panics if the cost of the transition state is used or a min/max reduce operation is performed on an empty set or vector.
    #[inline]
    pub fn is_satisfied<S: StateInterface>(
        &self,
        state: &S,
        function_cache: &mut StateFunctionCache,
        state_functions: &StateFunctions,
        registry: &TableRegistry,
    ) -> bool {
        self.conditions
            .iter()
            .all(|x| x.is_satisfied(state, function_cache, state_functions, registry))
    }

    /// Returns the cost of the base case if it is satisfied and None otherwise.
    ///
    /// # Panics
    ///
    /// Panics if the cost of the transition state is used or a min/max reduce operation is performed on an empty set or vector.
    pub fn eval_cost<S: StateInterface, T: Numeric>(
        &self,
        state: &S,
        function_cache: &mut StateFunctionCache,
        state_functions: &StateFunctions,
        registry: &TableRegistry,
    ) -> Option<T> {
        if self.is_satisfied(state, function_cache, state_functions, registry) {
            Some(self.cost.as_ref().map_or_else(T::zero, |cost| {
                cost.eval(state, function_cache, state_functions, registry)
            }))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expression::*;
    use crate::state::{SignatureVariables, State};
    use crate::variable_type::{Integer, Set};

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
        assert_eq!(base_case.conditions, conditions);
    }

    #[test]
    fn from_base_case_to_conditions() {
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
        let base_case = BaseCase::from(conditions);
        assert_eq!(
            Vec::<Condition>::from(base_case),
            vec![Condition::Constant(true), Condition::Constant(false),]
        );
    }

    #[test]
    fn from_base_case_to_conditions_and_cost() {
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
        let base_case = BaseCase::with_cost(conditions, 1);
        let (conditions, cost) = base_case.into();
        assert_eq!(
            conditions,
            vec![Condition::Constant(true), Condition::Constant(false)]
        );
        assert_eq!(
            cost,
            Some(CostExpression::from(IntegerExpression::Constant(1))),
        );
    }

    #[test]
    fn with_cost() {
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
        let cost = IntegerExpression::Constant(1);
        let base_case = BaseCase::with_cost(conditions.clone(), cost.clone());
        assert_eq!(base_case.conditions, conditions,);
        assert_eq!(base_case.cost, Some(CostExpression::from(cost)));
    }

    #[test]
    fn is_satisfied() {
        let mut s0 = Set::with_capacity(2);
        s0.insert(0);
        s0.insert(1);
        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![s0],
                integer_variables: vec![1],
                ..Default::default()
            },
            ..Default::default()
        };
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);

        let registry = TableRegistry::default();

        let base_case = BaseCase::from(vec![GroundedCondition {
            condition: Condition::Constant(true),
            ..Default::default()
        }]);
        assert!(base_case.is_satisfied(&state, &mut function_cache, &state_functions, &registry));
    }

    #[test]
    fn is_not_satisfied() {
        let mut s0 = Set::with_capacity(2);
        s0.insert(0);
        s0.insert(1);
        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![s0],
                integer_variables: vec![1],
                ..Default::default()
            },
            ..Default::default()
        };
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);

        let registry = TableRegistry::default();

        let base_case = BaseCase::from(vec![
            GroundedCondition {
                condition: Condition::Constant(true),
                ..Default::default()
            },
            GroundedCondition {
                condition: Condition::Constant(false),
                ..Default::default()
            },
        ]);
        assert!(!base_case.is_satisfied(&state, &mut function_cache, &state_functions, &registry));
    }

    #[test]
    fn eval_cost_zero() {
        let mut s0 = Set::with_capacity(2);
        s0.insert(0);
        s0.insert(1);
        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![s0],
                integer_variables: vec![1],
                ..Default::default()
            },
            ..Default::default()
        };
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);

        let registry = TableRegistry::default();

        let base_case = BaseCase::from(vec![GroundedCondition {
            condition: Condition::Constant(true),
            ..Default::default()
        }]);
        assert_eq!(
            base_case.eval_cost(&state, &mut function_cache, &state_functions, &registry),
            Some(0)
        );
    }

    #[test]
    fn eval_cost_some() {
        let mut s0 = Set::with_capacity(2);
        s0.insert(0);
        s0.insert(1);
        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![s0],
                integer_variables: vec![1],
                ..Default::default()
            },
            ..Default::default()
        };
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);

        let registry = TableRegistry::default();

        let base_case = BaseCase::with_cost(
            vec![GroundedCondition {
                condition: Condition::Constant(true),
                ..Default::default()
            }],
            IntegerExpression::Constant(1),
        );
        assert_eq!(
            base_case.eval_cost(&state, &mut function_cache, &state_functions, &registry),
            Some(1)
        );
    }

    #[test]
    fn eval_cost_none() {
        let mut s0 = Set::with_capacity(2);
        s0.insert(0);
        s0.insert(1);
        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![s0],
                integer_variables: vec![1],
                ..Default::default()
            },
            ..Default::default()
        };
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);

        let registry = TableRegistry::default();

        let base_case = BaseCase::with_cost(
            vec![
                GroundedCondition {
                    condition: Condition::Constant(true),
                    ..Default::default()
                },
                GroundedCondition {
                    condition: Condition::Constant(false),
                    ..Default::default()
                },
            ],
            IntegerExpression::Constant(1),
        );
        assert_eq!(
            base_case.eval_cost::<_, Integer>(
                &state,
                &mut function_cache,
                &state_functions,
                &registry
            ),
            None
        );
    }
}
