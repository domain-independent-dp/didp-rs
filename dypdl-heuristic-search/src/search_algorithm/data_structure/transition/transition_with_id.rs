use dypdl::{
    variable_type::Numeric, StateFunctionCache, StateFunctions, Transition, TransitionInterface,
};

#[derive(Debug, PartialEq, Clone, Default)]
pub struct TransitionWithId<T = Transition>
where
    T: TransitionInterface,
{
    /// Transition.
    pub transition: T,
    /// If forced.
    pub forced: bool,
    /// ID.
    pub id: usize,
}

impl<T: TransitionInterface> TransitionInterface for TransitionWithId<T> {
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
    fn apply<S: dypdl::StateInterface, U: From<dypdl::State>>(
        &self,
        state: &S,
        function_cache: &mut StateFunctionCache,
        state_functions: &StateFunctions,
        registry: &dypdl::TableRegistry,
    ) -> U {
        self.transition
            .apply(state, function_cache, state_functions, registry)
    }

    #[inline]
    fn eval_cost<U: Numeric, S: dypdl::StateInterface>(
        &self,
        cost: U,
        state: &S,
        function_cache: &mut StateFunctionCache,
        state_functions: &StateFunctions,
        registry: &dypdl::TableRegistry,
    ) -> U {
        self.transition
            .eval_cost(cost, state, function_cache, state_functions, registry)
    }
}

impl<T> From<TransitionWithId<T>> for Transition
where
    T: TransitionInterface,
    Transition: From<T>,
{
    fn from(transition: TransitionWithId<T>) -> Self {
        transition.transition.into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dypdl::expression::*;
    use dypdl::prelude::*;

    #[test]
    fn transition_with_custom_cost_to_transition() {
        let mut transition = Transition::new("transition");
        transition.set_cost(IntegerExpression::Cost + 1);
        let transition_with_custom_cost = TransitionWithId {
            transition: transition.clone(),
            forced: false,
            id: 0,
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
        let transition = TransitionWithId {
            transition,
            forced: false,
            id: 0,
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
        let transition = TransitionWithId {
            transition,
            forced: false,
            id: 0,
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
        let transition = TransitionWithId {
            transition,
            id: 0,
            forced: false,
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
        let transition = TransitionWithId {
            transition,
            id: 0,
            forced: false,
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
}
