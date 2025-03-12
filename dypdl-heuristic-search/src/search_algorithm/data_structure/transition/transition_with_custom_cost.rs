use super::super::successor_generator::SuccessorGenerator;
use core::ops::Deref;
use dypdl::variable_type::Numeric;
use dypdl::{CostExpression, StateFunctionCache, StateFunctions, Transition, TransitionInterface};
use std::fmt::Debug;

/// Transition with a customized cost expression.
#[derive(Debug, PartialEq, Clone, Default)]
pub struct TransitionWithCustomCost {
    /// Transition.
    pub transition: Transition,
    /// Customized cost expression.
    pub custom_cost: CostExpression,
}

impl TransitionInterface for TransitionWithCustomCost {
    #[inline]
    fn is_applicable<S: dypdl::StateInterface>(
        &self,
        state: &S,
        function_cache: &mut StateFunctionCache,
        state_functions: &StateFunctions,
        registry: &dypdl::TableRegistry,
    ) -> bool {
        self.transition
            .is_applicable(state, function_cache, state_functions, registry)
    }

    #[inline]
    fn apply<S: dypdl::StateInterface, T: From<dypdl::State>>(
        &self,
        state: &S,
        function_cache: &mut StateFunctionCache,
        state_functions: &StateFunctions,
        registry: &dypdl::TableRegistry,
    ) -> T {
        self.transition
            .apply(state, function_cache, state_functions, registry)
    }

    #[inline]
    fn eval_cost<U: Numeric, T: dypdl::StateInterface>(
        &self,
        cost: U,
        state: &T,
        function_cache: &mut StateFunctionCache,
        state_functions: &StateFunctions,
        registry: &dypdl::TableRegistry,
    ) -> U {
        self.transition
            .eval_cost(cost, state, function_cache, state_functions, registry)
    }
}

impl From<TransitionWithCustomCost> for Transition {
    fn from(transition: TransitionWithCustomCost) -> Self {
        transition.transition
    }
}

impl<U, R> SuccessorGenerator<TransitionWithCustomCost, U, R>
where
    U: Deref<Target = TransitionWithCustomCost> + Clone + From<TransitionWithCustomCost>,
    R: Deref<Target = dypdl::Model>,
{
    /// Returns a successor generator returning applicable transitions with customized cost expressions.
    pub fn from_model_with_custom_costs(
        model: R,
        custom_costs: &[CostExpression],
        forced_custom_costs: &[CostExpression],
        backward: bool,
    ) -> Self {
        let forced_transitions = if backward {
            &model.backward_forced_transitions
        } else {
            &model.forward_forced_transitions
        };
        let forced_transitions = forced_transitions
            .iter()
            .zip(forced_custom_costs)
            .map(|(t, c)| {
                U::from(TransitionWithCustomCost {
                    transition: t.clone(),
                    custom_cost: c.simplify(&model.table_registry),
                })
            })
            .collect();

        let transitions = if backward {
            &model.backward_transitions
        } else {
            &model.forward_transitions
        };
        let transitions = transitions
            .iter()
            .zip(custom_costs)
            .map(|(t, c)| {
                U::from(TransitionWithCustomCost {
                    transition: t.clone(),
                    custom_cost: c.simplify(&model.table_registry),
                })
            })
            .collect();

        SuccessorGenerator::new(forced_transitions, transitions, backward, model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dypdl::expression::*;
    use dypdl::prelude::*;
    use std::rc::Rc;

    #[test]
    fn transition_with_custom_cost_to_transition() {
        let mut transition = Transition::new("transition");
        transition.set_cost(IntegerExpression::Cost + 1);
        let transition_with_custom_cost = TransitionWithCustomCost {
            transition: transition.clone(),
            custom_cost: CostExpression::Integer(IntegerExpression::Cost + 2),
        };
        assert_eq!(Transition::from(transition_with_custom_cost), transition);
    }

    #[test]
    fn is_applicable() {
        let mut model = Model::default();
        let var = model.add_integer_variable("v", 0);
        assert!(var.is_ok());
        let var = var.unwrap();

        let mut transition = Transition::new("transition");
        transition.add_precondition(Condition::comparison_i(ComparisonOperator::Le, var, 1));
        let transition = TransitionWithCustomCost {
            transition,
            custom_cost: CostExpression::Integer(IntegerExpression::Cost + 2),
        };
        let state = model.target;
        let mut function_cache = StateFunctionCache::new(&model.state_functions);
        assert!(transition.is_applicable(
            &state,
            &mut function_cache,
            &model.state_functions,
            &model.table_registry
        ));
    }

    #[test]
    fn is_not_applicable() {
        let mut model = Model::default();
        let var = model.add_integer_variable("v", 0);
        assert!(var.is_ok());
        let var = var.unwrap();

        let mut transition = Transition::new("transition");
        transition.add_precondition(Condition::comparison_i(ComparisonOperator::Le, var, 0));
        let transition = TransitionWithCustomCost {
            transition,
            custom_cost: CostExpression::Integer(IntegerExpression::Cost + 2),
        };
        let mut function_cache = StateFunctionCache::new(&model.state_functions);
        assert!(transition.is_applicable(
            &model.target,
            &mut function_cache,
            &model.state_functions,
            &model.table_registry
        ));
    }

    #[test]
    fn apply() {
        let mut model = Model::default();
        let var1 = model.add_integer_variable("var1", 0);
        assert!(var1.is_ok());
        let var1 = var1.unwrap();
        let var2 = model.add_integer_variable("var2", 0);
        assert!(var2.is_ok());

        let mut transition = Transition::new("transition");
        let result = transition.add_effect(var1, var1 + 1);
        assert!(result.is_ok());
        let transition = TransitionWithCustomCost {
            transition,
            custom_cost: CostExpression::Integer(IntegerExpression::Cost + 2),
        };

        let mut function_cache = StateFunctionCache::new(&model.state_functions);
        let state: State = transition.apply(
            &model.target,
            &mut function_cache,
            &model.state_functions,
            &model.table_registry,
        );
        assert_eq!(state.get_integer_variable(0), 1);
        assert_eq!(state.get_integer_variable(1), 0);
    }

    #[test]
    fn eval_cost() {
        let model = Model::default();

        let mut transition = Transition::new("transition");
        transition.set_cost(IntegerExpression::Cost + 1);
        let transition = TransitionWithCustomCost {
            transition,
            custom_cost: CostExpression::Integer(IntegerExpression::Cost + 2),
        };
        let mut function_cache = StateFunctionCache::new(&model.state_functions);
        let cost = transition.eval_cost(
            0,
            &model.target,
            &mut function_cache,
            &model.state_functions,
            &model.table_registry,
        );
        assert_eq!(cost, 1);
    }

    #[test]
    fn from_model_with_custom_costs_forward() {
        let mut model = Model::default();
        let mut transition1 = Transition::new("transition1");
        transition1.set_cost(IntegerExpression::Cost + 1);
        let result = model.add_forward_transition(transition1.clone());
        assert!(result.is_ok());
        let mut transition2 = Transition::new("transition2");
        transition2.set_cost(IntegerExpression::Cost + 2);
        let result = model.add_forward_transition(transition2.clone());
        assert!(result.is_ok());
        let mut transition3 = Transition::new("transition3");
        transition3.set_cost(IntegerExpression::Cost + 3);
        let result = model.add_forward_forced_transition(transition3.clone());
        assert!(result.is_ok());
        let mut transition4 = Transition::new("transition4");
        transition4.set_cost(IntegerExpression::Cost + 4);
        let result = model.add_forward_forced_transition(transition4.clone());
        assert!(result.is_ok());
        let mut transition5 = Transition::new("transition5");
        transition5.set_cost(IntegerExpression::Cost + 5);
        let result = model.add_backward_transition(transition5.clone());
        assert!(result.is_ok());
        let mut transition6 = Transition::new("transition6");
        transition6.set_cost(IntegerExpression::Cost + 6);
        let result = model.add_backward_forced_transition(transition6.clone());
        assert!(result.is_ok());
        let model = Rc::new(model);

        let custom_costs = [
            CostExpression::Integer(IntegerExpression::Cost + 7),
            CostExpression::Integer(IntegerExpression::Cost + 8),
        ];
        let forced_custom_costs = [
            CostExpression::Integer(IntegerExpression::Cost + 9),
            CostExpression::Integer(IntegerExpression::Cost + 10),
        ];
        let generator = SuccessorGenerator::<_>::from_model_with_custom_costs(
            model.clone(),
            &custom_costs,
            &forced_custom_costs,
            false,
        );

        assert_eq!(generator.model, model);
        assert_eq!(
            generator.transitions,
            vec![
                Rc::new(TransitionWithCustomCost {
                    transition: transition1,
                    custom_cost: CostExpression::Integer(IntegerExpression::Cost + 7),
                }),
                Rc::new(TransitionWithCustomCost {
                    transition: transition2,
                    custom_cost: CostExpression::Integer(IntegerExpression::Cost + 8),
                }),
            ]
        );
        assert_eq!(
            generator.forced_transitions,
            vec![
                Rc::new(TransitionWithCustomCost {
                    transition: transition3,
                    custom_cost: CostExpression::Integer(IntegerExpression::Cost + 9),
                }),
                Rc::new(TransitionWithCustomCost {
                    transition: transition4,
                    custom_cost: CostExpression::Integer(IntegerExpression::Cost + 10),
                }),
            ]
        );
    }

    #[test]
    fn from_model_with_custom_costs_backward() {
        let mut model = Model::default();
        let mut transition1 = Transition::new("transition1");
        transition1.set_cost(IntegerExpression::Cost + 1);
        let result = model.add_backward_transition(transition1.clone());
        assert!(result.is_ok());
        let mut transition2 = Transition::new("transition2");
        transition2.set_cost(IntegerExpression::Cost + 2);
        let result = model.add_backward_transition(transition2.clone());
        assert!(result.is_ok());
        let mut transition3 = Transition::new("transition3");
        transition3.set_cost(IntegerExpression::Cost + 3);
        let result = model.add_backward_forced_transition(transition3.clone());
        assert!(result.is_ok());
        let mut transition4 = Transition::new("transition4");
        transition4.set_cost(IntegerExpression::Cost + 4);
        let result = model.add_backward_forced_transition(transition4.clone());
        assert!(result.is_ok());
        let mut transition5 = Transition::new("transition5");
        transition5.set_cost(IntegerExpression::Cost + 5);
        let result = model.add_forward_transition(transition5.clone());
        assert!(result.is_ok());
        let mut transition6 = Transition::new("transition6");
        transition6.set_cost(IntegerExpression::Cost + 6);
        let result = model.add_forward_forced_transition(transition6.clone());
        assert!(result.is_ok());
        let model = Rc::new(model);

        let custom_costs = [
            CostExpression::Integer(IntegerExpression::Cost + 7),
            CostExpression::Integer(IntegerExpression::Cost + 8),
        ];
        let forced_custom_costs = [
            CostExpression::Integer(IntegerExpression::Cost + 9),
            CostExpression::Integer(IntegerExpression::Cost + 10),
        ];
        let generator = SuccessorGenerator::<_>::from_model_with_custom_costs(
            model.clone(),
            &custom_costs,
            &forced_custom_costs,
            true,
        );

        assert_eq!(generator.model, model);
        assert_eq!(
            generator.transitions,
            vec![
                Rc::new(TransitionWithCustomCost {
                    transition: transition1,
                    custom_cost: CostExpression::Integer(IntegerExpression::Cost + 7),
                }),
                Rc::new(TransitionWithCustomCost {
                    transition: transition2,
                    custom_cost: CostExpression::Integer(IntegerExpression::Cost + 8),
                }),
            ]
        );
        assert_eq!(
            generator.forced_transitions,
            vec![
                Rc::new(TransitionWithCustomCost {
                    transition: transition3,
                    custom_cost: CostExpression::Integer(IntegerExpression::Cost + 9),
                }),
                Rc::new(TransitionWithCustomCost {
                    transition: transition4,
                    custom_cost: CostExpression::Integer(IntegerExpression::Cost + 10),
                }),
            ]
        );
    }
}
