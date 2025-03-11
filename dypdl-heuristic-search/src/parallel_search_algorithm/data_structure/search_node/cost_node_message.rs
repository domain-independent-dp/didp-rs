use super::super::arc_chain::ArcChain;
use super::search_node_message::SearchNodeMessage;
use crate::search_algorithm::data_structure::{
    CreateTransitionChain, StateInformation, StateWithHashableSignatureVariables,
};
use crate::search_algorithm::CostNode;
use dypdl::variable_type::Numeric;
use dypdl::{Model, ReduceFunction, StateFunctionCache, Transition, TransitionInterface};
use rustc_hash::FxHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// Node ordered by the cost to be sent to another thread via message passing.
#[derive(Debug, Clone, PartialEq)]
pub struct CostNodeMessage<T, V = Transition>
where
    V: TransitionInterface + Clone,
    Transition: From<V>,
{
    /// State.
    pub state: StateWithHashableSignatureVariables,
    /// Priority
    pub priority: T,
    /// Transitions applied before.
    pub transitions: Option<Arc<ArcChain<V>>>,
}

impl<T, V> From<CostNodeMessage<T, V>> for CostNode<T, V, Arc<V>, ArcChain<V>, Arc<ArcChain<V>>>
where
    T: Numeric + Ord,
    V: TransitionInterface + Clone,
    Transition: From<V>,
{
    fn from(node: CostNodeMessage<T, V>) -> Self {
        Self::with_state_and_priority_and_transition_chain(
            node.state.into(),
            node.priority,
            node.transitions,
        )
    }
}

impl<T, V> CostNodeMessage<T, V>
where
    T: Numeric + Send + Sync,
    V: TransitionInterface + Clone + Send + Sync,
    Transition: From<V>,
{
    /// Creates a new search node given a state, its cost, a DyPDL model, and a transition chain.
    ///
    /// If the model is minimizing, the priority becomes the negative of the cost.
    /// If the model is maximizing, the priority becomes the cost.
    pub fn new(
        state: StateWithHashableSignatureVariables,
        cost: T,
        model: &Model,
        transitions: Option<Arc<ArcChain<V>>>,
    ) -> Self {
        let priority = if model.reduce_function == ReduceFunction::Max {
            cost
        } else if cost == T::min_value() {
            T::max_value()
        } else if cost == T::max_value() {
            T::min_value()
        } else {
            -cost
        };

        Self {
            state,
            priority,
            transitions,
        }
    }

    /// Generates a root search node given a state, its cost, and a DyPDL model.
    ///
    /// If the model is minimizing, the priority becomes the negative of the cost.
    /// If the model is maximizing, the priority becomes the cost.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    /// use dypdl_heuristic_search::CostNodeMessage;
    /// use dypdl_heuristic_search::search_algorithm::{CostNode, StateInRegistry};
    /// use dypdl_heuristic_search::search_algorithm::data_structure::{
    ///     GetTransitions, StateInformation,
    /// };
    ///
    /// let mut model = Model::default();
    /// model.add_integer_variable("variable", 0).unwrap();
    ///
    /// let state = model.target.clone();
    /// let cost = 0;
    /// let node = CostNodeMessage::<_>::generate_root_node(state, cost, &model);
    /// let node = CostNode::from(node);
    /// assert_eq!(node.state(), &StateInRegistry::from(model.target.clone()));
    /// assert_eq!(node.cost(&model), cost);
    /// assert!(!node.is_closed());
    /// assert_eq!(node.transitions(), vec![]);
    /// ```
    pub fn generate_root_node<S>(state: S, cost: T, model: &Model) -> Self
    where
        StateWithHashableSignatureVariables: From<S>,
    {
        let state = StateWithHashableSignatureVariables::from(state);

        Self::new(state, cost, model, None)
    }
}

impl<T, V> CostNode<T, V, Arc<V>, ArcChain<V>, Arc<ArcChain<V>>>
where
    T: Numeric + Ord + Send + Sync,
    V: TransitionInterface + Clone + Send + Sync,
    Transition: From<V>,
{
    /// Generates a sendable successor node message given a transition and a DyPDL model.
    ///
    /// `function_cache` is not cleared and updated by this node.
    ///
    /// Returns `None` if the successor state is pruned by a state constraint.
    ///
    /// # Panics
    ///
    /// If an expression used in the transition is invalid.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    /// use dypdl_heuristic_search::search_algorithm::{CostNode, StateInRegistry};
    /// use dypdl_heuristic_search::search_algorithm::data_structure::{
    ///     GetTransitions, StateInformation,
    /// };
    /// use std::sync::Arc;
    ///
    /// let mut model = Model::default();
    /// let variable = model.add_integer_variable("variable", 0).unwrap();
    ///
    /// let state = model.target.clone();
    /// let cost = 0;
    /// let node = CostNode::<_, _, _, _, _>::generate_root_node(state, cost, &model);
    ///
    /// let mut transition = Transition::new("transition");
    /// transition.set_cost(IntegerExpression::Cost + 1);
    /// transition.add_effect(variable, variable + 1).unwrap();
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    /// let expected_state: StateInRegistry = transition.apply(
    ///     &model.target, &mut function_cache, &model.state_functions, &model.table_registry
    /// );
    ///
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    /// let node = node.generate_sendable_successor_node(
    ///     Arc::new(transition.clone()), &mut function_cache, &model,
    /// );
    /// assert!(node.is_some());
    /// let node = CostNode::from(node.unwrap());
    /// assert_eq!(node.state(), &expected_state);
    /// assert_eq!(node.cost(&model), 1);
    /// assert!(!node.is_closed());
    /// assert_eq!(node.transitions(), vec![transition]);
    /// ```
    pub fn generate_sendable_successor_node(
        &self,
        transition: Arc<V>,
        function_cache: &mut StateFunctionCache,
        model: &Model,
    ) -> Option<CostNodeMessage<T, V>> {
        let cost = self.cost(model);
        let (state, cost) = model.generate_successor_state(
            self.state(),
            function_cache,
            cost,
            transition.as_ref(),
            None,
        )?;
        let transitions = Arc::new(ArcChain::new(self.transition_chain(), transition));

        Some(CostNodeMessage::new(state, cost, model, Some(transitions)))
    }
}

impl<T, V> SearchNodeMessage for CostNodeMessage<T, V>
where
    T: Numeric + Send + Sync,
    V: TransitionInterface + Clone + Send + Sync,
    Transition: From<V>,
{
    fn assign_thread(&self, threads: usize) -> usize {
        const SEED: u32 = 0x5583c24d;

        let mut hasher = FxHasher::default();
        hasher.write_u32(SEED);
        self.state.signature_variables.hash(&mut hasher);
        hasher.finish() as usize % threads
    }
}

#[cfg(test)]
mod tests {
    use crate::search_algorithm::StateInRegistry;

    use super::*;
    use crate::search_algorithm::data_structure::GetTransitions;
    use dypdl::expression::*;
    use dypdl::prelude::*;

    #[test]
    fn test_from_message() {
        let mut model = Model::default();
        let v1 = model.add_integer_resource_variable("v1", true, 0);
        assert!(v1.is_ok());
        let v2 = model.add_integer_resource_variable("v2", false, 0);
        assert!(v2.is_ok());

        let state = model.target.clone();
        let message = CostNodeMessage::<_> {
            state: state.clone().into(),
            priority: -1,
            transitions: None,
        };

        let mut node = CostNode::from(message);
        assert_eq!(node.priority, -1);
        assert_eq!(node.state(), &StateInRegistry::from(state.clone()));
        assert_eq!(node.state_mut(), &mut StateInRegistry::from(state.clone()));
        assert_eq!(node.cost(&model), 1);
        assert_eq!(node.bound(&model), None);
        assert!(!node.is_closed());
        assert_eq!(node.transitions(), vec![]);
        assert_eq!(node.last(), None);
        assert_eq!(node.transition_chain(), None);
    }

    #[test]
    fn test_new_node_min() {
        let mut model = dypdl::Model::default();
        model.set_minimize();
        let variable = model.add_integer_variable("variable", 0);
        assert!(variable.is_ok());
        let state = StateWithHashableSignatureVariables::from(model.target.clone());
        let expected_state = state.clone();
        let node = CostNodeMessage::<_>::new(state, 1, &model, None);
        assert_eq!(node.state, expected_state);
        assert_eq!(node.priority, -1);
        assert_eq!(node.transitions, None);
    }

    #[test]
    fn test_new_node_min_cost_max() {
        let mut model = dypdl::Model::default();
        model.set_minimize();
        let variable = model.add_integer_variable("variable", 0);
        assert!(variable.is_ok());
        let state = StateWithHashableSignatureVariables::from(model.target.clone());
        let expected_state = state.clone();
        let node = CostNodeMessage::<_>::new(state, Integer::MAX, &model, None);
        assert_eq!(node.state, expected_state);
        assert_eq!(node.priority, Integer::MIN);
        assert_eq!(node.transitions, None);
    }

    #[test]
    fn test_new_node_min_cost_min() {
        let mut model = dypdl::Model::default();
        model.set_minimize();
        let variable = model.add_integer_variable("variable", 0);
        assert!(variable.is_ok());
        let state = StateWithHashableSignatureVariables::from(model.target.clone());
        let expected_state = state.clone();
        let node = CostNodeMessage::<_>::new(state, Integer::MIN, &model, None);
        assert_eq!(node.state, expected_state);
        assert_eq!(node.priority, Integer::MAX);
        assert_eq!(node.transitions, None);
    }

    #[test]
    fn test_new_node_max() {
        let mut model = dypdl::Model::default();
        model.set_maximize();
        let variable = model.add_integer_variable("variable", 0);
        assert!(variable.is_ok());
        let state = StateWithHashableSignatureVariables::from(model.target.clone());
        let expected_state = state.clone();
        let node = CostNodeMessage::<_>::new(state, 1, &model, None);
        assert_eq!(node.state, expected_state);
        assert_eq!(node.priority, 1);
        assert_eq!(node.transitions, None);
    }

    #[test]
    fn test_new_node_max_cost_max() {
        let mut model = dypdl::Model::default();
        model.set_maximize();
        let variable = model.add_integer_variable("variable", 0);
        assert!(variable.is_ok());
        let state = StateWithHashableSignatureVariables::from(model.target.clone());
        let expected_state = state.clone();
        let node = CostNodeMessage::<_>::new(state, Integer::MAX, &model, None);
        assert_eq!(node.state, expected_state);
        assert_eq!(node.priority, Integer::MAX);
        assert_eq!(node.transitions, None);
    }

    #[test]
    fn test_new_node_max_cost_min() {
        let mut model = dypdl::Model::default();
        model.set_maximize();
        let variable = model.add_integer_variable("variable", 0);
        assert!(variable.is_ok());
        let state = StateWithHashableSignatureVariables::from(model.target.clone());
        let expected_state = state.clone();
        let node = CostNodeMessage::<_>::new(state, Integer::MIN, &model, None);
        assert_eq!(node.state, expected_state);
        assert_eq!(node.priority, Integer::MIN);
        assert_eq!(node.transitions, None);
    }

    #[test]
    fn test_generate_root_node_min() {
        let mut model = dypdl::Model::default();
        model.set_minimize();
        let variable = model.add_integer_variable("variable", 0);
        assert!(variable.is_ok());
        let state = model.target.clone();
        let expected_state = StateWithHashableSignatureVariables::from(state.clone());
        let node = CostNodeMessage::<_>::generate_root_node(state, 1, &model);
        assert_eq!(node.state, expected_state);
        assert_eq!(node.priority, -1);
        assert_eq!(node.transitions, None);
    }

    #[test]
    fn test_generate_root_node_max() {
        let mut model = dypdl::Model::default();
        model.set_maximize();
        let variable = model.add_integer_variable("variable", 0);
        assert!(variable.is_ok());
        let state = model.target.clone();
        let expected_state = StateWithHashableSignatureVariables::from(state.clone());
        let node = CostNodeMessage::<_>::generate_root_node(state, 1, &model);
        assert_eq!(node.state, expected_state);
        assert_eq!(node.priority, 1);
        assert_eq!(node.transitions, None);
    }

    #[test]
    fn test_generate_sendable_successor_some_min() {
        let mut model = Model::default();
        model.set_minimize();
        let v1 = model.add_integer_resource_variable("v1", true, 0);
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let v2 = model.add_integer_resource_variable("v2", false, 0);
        assert!(v2.is_ok());
        let v2 = v2.unwrap();

        let mut transition = Transition::default();
        let result = transition.add_effect(v1, v1 + 1);
        assert!(result.is_ok());
        let result = transition.add_effect(v2, v2 + 1);
        assert!(result.is_ok());
        transition.set_cost(IntegerExpression::Cost + 1);

        let state = model.target.clone();
        let mut function_cache = StateFunctionCache::new(&model.state_functions);
        let expected_state = transition.apply(
            &state,
            &mut function_cache,
            &model.state_functions,
            &model.table_registry,
        );

        let node = CostNode::generate_root_node(state, 0, &model);
        let mut function_cache = StateFunctionCache::new(&model.state_functions);
        let successor = node.generate_sendable_successor_node(
            Arc::new(transition.clone()),
            &mut function_cache,
            &model,
        );
        assert!(successor.is_some());
        let successor = successor.unwrap();

        assert_eq!(&successor.state, &expected_state);
        assert_eq!(successor.priority, -1);
        assert_eq!(
            successor.transitions,
            Some(Arc::new(ArcChain::new(None, Arc::new(transition))))
        );
    }

    #[test]
    fn test_generate_sendable_successor_some_max() {
        let mut model = Model::default();
        model.set_maximize();
        let v1 = model.add_integer_resource_variable("v1", true, 0);
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let v2 = model.add_integer_resource_variable("v2", false, 0);
        assert!(v2.is_ok());
        let v2 = v2.unwrap();

        let mut transition = Transition::default();
        let result = transition.add_effect(v1, v1 + 1);
        assert!(result.is_ok());
        let result = transition.add_effect(v2, v2 + 1);
        assert!(result.is_ok());
        transition.set_cost(IntegerExpression::Cost + 1);

        let state = model.target.clone();
        let mut function_cache = StateFunctionCache::new(&model.state_functions);
        let expected_state = transition.apply(
            &state,
            &mut function_cache,
            &model.state_functions,
            &model.table_registry,
        );

        let node = CostNode::generate_root_node(state, 0, &model);
        let mut function_cache = StateFunctionCache::new(&model.state_functions);
        let successor = node.generate_sendable_successor_node(
            Arc::new(transition.clone()),
            &mut function_cache,
            &model,
        );
        assert!(successor.is_some());
        let successor = successor.unwrap();

        assert_eq!(&successor.state, &expected_state);
        assert_eq!(successor.priority, 1);
        assert_eq!(
            successor.transitions,
            Some(Arc::new(ArcChain::new(None, Arc::new(transition))))
        );
    }

    #[test]
    fn test_generate_sendable_successor_pruned_by_constraint() {
        let mut model = Model::default();
        let v1 = model.add_integer_resource_variable("v1", true, 0);
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let v2 = model.add_integer_resource_variable("v2", false, 0);
        assert!(v2.is_ok());
        let v2 = v2.unwrap();
        let result =
            model.add_state_constraint(Condition::comparison_i(ComparisonOperator::Le, v1, 0));
        assert!(result.is_ok());

        let state = model.target.clone();
        let node = CostNode::<_, _, _, ArcChain<_>, _>::generate_root_node(state, 0, &model);

        let mut transition = Transition::default();
        let result = transition.add_effect(v1, v1 + 1);
        assert!(result.is_ok());
        let result = transition.add_effect(v2, v2 + 1);
        assert!(result.is_ok());
        transition.set_cost(IntegerExpression::Cost + 1);

        let mut function_cache = StateFunctionCache::new(&model.state_functions);
        let result = node.generate_sendable_successor_node(
            Arc::new(transition),
            &mut function_cache,
            &model,
        );
        assert_eq!(result, None);
    }
}
