use super::super::arc_chain::ArcChain;
use super::cost_node_message::CostNodeMessage;
use super::search_node_message::SearchNodeMessage;
use crate::search_algorithm::data_structure::{
    CreateTransitionChain, StateInformation, StateWithHashableSignatureVariables,
};
use crate::search_algorithm::FNode;
use dypdl::variable_type::Numeric;
use dypdl::{Model, Transition, TransitionInterface};
use rustc_hash::FxHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// Node ordered by the f-value to be sent to another thread via message passing.
#[derive(Debug, Clone, PartialEq)]
pub struct FNodeMessage<T, V = Transition>
where
    V: TransitionInterface + Clone,
    Transition: From<V>,
{
    /// Node message.
    pub node: CostNodeMessage<T, V>,
    /// h-value.
    pub h: T,
    /// f-value.
    pub f: T,
}

impl<T, V> From<FNodeMessage<T, V>> for FNode<T, V, Arc<V>, ArcChain<V>, Arc<ArcChain<V>>>
where
    T: Numeric + Ord,
    V: TransitionInterface + Clone,
    Transition: From<V>,
{
    fn from(value: FNodeMessage<T, V>) -> Self {
        Self::with_node_and_h_and_f(value.node.into(), value.h, value.f)
    }
}

impl<T, V> FNodeMessage<T, V>
where
    T: Numeric + Ord + Send + Sync,
    V: TransitionInterface + Clone + Send + Sync,
    Transition: From<V>,
{
    /// Generates a root search node given a state, its cost, a DyPDL model, h- and f-evaluators,
    /// and a primal bound on the solution cost.
    ///
    /// `h_evaluator` is a function that takes a state and returns the dual bound of the cost from the state.
    /// If `h_evaluator` returns `None`, the state is a dead-end, so the node is not generated.
    /// `f_evaluator` is a function that takes `g`, the return value by `h_evaluator`, and the state and returns the dual bound of the cost of a solution extending the path to this node.
    ///
    /// Returns `None` if the state is a dead-end, or the return value by `f_evaluator` exceeds the primal bound.
    ///
    /// If the model is minimizing, the h- and f-values become the negatives of the values returned by the evaluators.
    /// If the model is maximizing, the h- and f-values become the values returned by the evaluators.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    /// use dypdl_heuristic_search::FNodeMessage;
    /// use dypdl_heuristic_search::search_algorithm::{FNode, StateInRegistry};
    /// use dypdl_heuristic_search::search_algorithm::data_structure::{
    ///     GetTransitions, StateInformation,
    /// };
    ///
    /// let mut model = Model::default();
    /// model.add_integer_variable("variable", 0).unwrap();
    ///
    /// let state = model.target.clone();
    /// let cost = 0;
    /// let h_evaluator = |_: &_| Some(0);
    /// let f_evaluator = |g, h, _: &_| g + h;
    /// let node = FNodeMessage::<_>::generate_root_node(
    ///     state, cost, &model, &h_evaluator, &f_evaluator, None,
    /// );
    /// assert!(node.is_some());
    /// let node = FNode::from(node.unwrap());
    /// assert_eq!(node.state(), &StateInRegistry::from(model.target.clone()));
    /// assert_eq!(node.cost(&model), cost);
    /// assert_eq!(node.bound(&model), Some(0));
    /// assert!(!node.is_closed());
    /// assert_eq!(node.transitions(), vec![]);
    /// ```
    pub fn generate_root_node<S, H, F>(
        state: S,
        cost: T,
        model: &Model,
        h_evaluator: H,
        f_evaluator: F,
        primal_bound: Option<T>,
    ) -> Option<Self>
    where
        StateWithHashableSignatureVariables: From<S>,
        H: FnOnce(&StateWithHashableSignatureVariables) -> Option<T>,
        F: FnOnce(T, T, &StateWithHashableSignatureVariables) -> T,
    {
        let state = StateWithHashableSignatureVariables::from(state);
        let (h, f) = FNode::<T, V, Arc<V>, ArcChain<V>, Arc<ArcChain<V>>>::evaluate_state(
            &state,
            cost,
            model,
            h_evaluator,
            f_evaluator,
            primal_bound,
            None,
        )?;
        let node = CostNodeMessage::<T, V>::generate_root_node::<StateWithHashableSignatureVariables>(
            state, cost, model,
        );

        Some(Self { node, h, f })
    }
}

impl<T, V> FNode<T, V, Arc<V>, ArcChain<V>, Arc<ArcChain<V>>>
where
    T: Numeric + Ord + Send + Sync,
    V: TransitionInterface + Clone + Send + Sync,
    Transition: From<V>,
{
    /// Generates a sendable successor node message given a transition, a DyPDL model, h- and f-evaluators,
    /// and a primal bound on the solution cost.
    ///
    /// `h_evaluator` is a function that takes a state and returns the dual bound of the cost from the state.
    /// If `h_evaluator` returns `None`, the state is a dead-end, so the node is not generated.
    /// `f_evaluator` is a function that takes `cost`, the return value by `h_evaluator`, and the state and returns the dual bound of the cost of a solution extending the path to this node.
    ///
    /// Returns `None` if the state is a dead-end, or the return value by `f_evaluator` exceeds the primal bound.
    ///
    /// If the model is minimizing, the h- and f-values become the negatives of the values returned by the evaluators.
    /// If the model is maximizing, the h- and f-values become the values returned by the evaluators.
    ///
    /// # Panics
    ///
    /// If an expression used in the transition is invalid.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    /// use dypdl_heuristic_search::search_algorithm::{FNode, StateInRegistry};
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
    /// let h_evaluator = |_: &_| Some(0);
    /// let f_evaluator = |g, h, _: &_| g + h;
    /// let node = FNode::<_, _, _, _, _>::generate_root_node(
    ///     state, cost, &model, &h_evaluator, &f_evaluator, None,
    /// ).unwrap();
    ///
    /// let mut transition = Transition::new("transition");
    /// transition.set_cost(IntegerExpression::Cost + 1);
    /// transition.add_effect(variable, variable + 1).unwrap();
    /// let expected_state: StateInRegistry = transition.apply(
    ///     &model.target, &model.table_registry,
    /// );
    ///
    /// let h_evaluator = |_: &_| Some(0);
    /// let f_evaluator = |g, h, _: &_| g + h;
    /// let node = node.generate_sendable_successor_node(
    ///     Arc::new(transition.clone()), &model, &h_evaluator, &f_evaluator, None,
    /// );
    /// assert!(node.is_some());
    /// let node = FNode::from(node.unwrap());
    /// assert_eq!(node.state(), &expected_state);
    /// assert_eq!(node.cost(&model), 1);
    /// assert_eq!(node.bound(&model), Some(1));
    /// assert!(!node.is_closed());
    /// assert_eq!(node.transitions(), vec![transition]);
    /// ```
    pub fn generate_sendable_successor_node<H, F>(
        &self,
        transition: Arc<V>,
        model: &Model,
        h_evaluator: H,
        f_evaluator: F,
        primal_bound: Option<T>,
    ) -> Option<FNodeMessage<T, V>>
    where
        H: FnOnce(&StateWithHashableSignatureVariables) -> Option<T>,
        F: FnOnce(T, T, &StateWithHashableSignatureVariables) -> T,
    {
        let (state, g) = model.generate_successor_state(
            self.state(),
            self.cost(model),
            transition.as_ref(),
            None,
        )?;
        let (h, f) = Self::evaluate_state(
            &state,
            g,
            model,
            h_evaluator,
            f_evaluator,
            primal_bound,
            None,
        )?;

        let transition_chain = Arc::new(ArcChain::new(self.transition_chain(), transition));
        let node = CostNodeMessage::new(state, g, model, Some(transition_chain));

        Some(FNodeMessage { node, h, f })
    }
}

impl<T, V> SearchNodeMessage for FNodeMessage<T, V>
where
    T: Numeric + Send + Sync,
    V: TransitionInterface + Clone + Send + Sync,
    Transition: From<V>,
{
    fn assign_thread(&self, threads: usize) -> usize {
        const SEED: u32 = 0x5583c24d;

        let mut hasher = FxHasher::default();
        hasher.write_u32(SEED);
        self.node.state.signature_variables.hash(&mut hasher);
        hasher.finish() as usize % threads
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search_algorithm::data_structure::{GetTransitions, StateInRegistry};
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
        let message = FNodeMessage {
            node: message,
            h: -2,
            f: -3,
        };

        let mut node = FNode::from(message);
        assert_eq!(node.h, -2);
        assert_eq!(node.f, -3);
        assert_eq!(node.state(), &StateInRegistry::from(state.clone()));
        assert_eq!(node.state_mut(), &mut StateInRegistry::from(state.clone()));
        assert_eq!(node.cost(&model), 1);
        assert_eq!(node.bound(&model), Some(3));
        assert!(!node.is_closed());
        assert_eq!(node.transitions(), vec![]);
        assert_eq!(node.last(), None);
        assert_eq!(node.transition_chain(), None);
    }

    #[test]
    fn test_generate_root_node_some_min() {
        let mut model = dypdl::Model::default();
        model.set_minimize();
        let variable = model.add_integer_variable("variable", 0);
        assert!(variable.is_ok());
        let state = model.target.clone();
        let expected_state = StateWithHashableSignatureVariables::from(state.clone());
        let h_evaluator = |_: &_| Some(0);
        let f_evaluator = |g, h, _: &_| g + h;
        let node = FNodeMessage::<_>::generate_root_node(
            state,
            1,
            &model,
            &h_evaluator,
            &f_evaluator,
            None,
        );
        assert!(node.is_some());
        let node = node.unwrap();
        assert_eq!(node.node.state, expected_state);
        assert_eq!(node.node.priority, -1);
        assert_eq!(node.h, 0);
        assert_eq!(node.f, -1);
        assert_eq!(node.node.transitions, None);
    }

    #[test]
    fn test_generate_root_node_some_max() {
        let mut model = dypdl::Model::default();
        model.set_maximize();
        let variable = model.add_integer_variable("variable", 0);
        assert!(variable.is_ok());
        let state = model.target.clone();
        let expected_state = StateWithHashableSignatureVariables::from(state.clone());
        let h_evaluator = |_: &_| Some(0);
        let f_evaluator = |g, h, _: &_| g + h;
        let node = FNodeMessage::<_>::generate_root_node(
            state,
            1,
            &model,
            &h_evaluator,
            &f_evaluator,
            None,
        );
        assert!(node.is_some());
        let node = node.unwrap();
        assert_eq!(node.node.state, expected_state);
        assert_eq!(node.node.priority, 1);
        assert_eq!(node.h, 0);
        assert_eq!(node.f, 1);
        assert_eq!(node.node.transitions, None);
    }

    #[test]
    fn test_generate_root_node_pruned_by_bound_min() {
        let mut model = dypdl::Model::default();
        model.set_minimize();
        let variable = model.add_integer_variable("variable", 0);
        assert!(variable.is_ok());
        let state = model.target.clone();
        let h_evaluator = |_: &_| Some(1);
        let f_evaluator = |g, h, _: &_| g + h;
        let node = FNodeMessage::<_>::generate_root_node(
            state,
            0,
            &model,
            &h_evaluator,
            &f_evaluator,
            Some(0),
        );
        assert!(node.is_none());
    }

    #[test]
    fn test_generate_root_node_pruned_by_bound_max() {
        let mut model = dypdl::Model::default();
        model.set_maximize();
        let variable = model.add_integer_variable("variable", 0);
        assert!(variable.is_ok());
        let state = model.target.clone();
        let h_evaluator = |_: &_| Some(1);
        let f_evaluator = |g, h, _: &_| g + h;
        let node = FNodeMessage::<_>::generate_root_node(
            state,
            0,
            &model,
            &h_evaluator,
            &f_evaluator,
            Some(2),
        );
        assert!(node.is_none());
    }

    #[test]
    fn test_generate_root_node_pruned_by_h() {
        let mut model = dypdl::Model::default();
        let variable = model.add_integer_variable("variable", 0);
        assert!(variable.is_ok());
        let state = model.target.clone();
        let h_evaluator = |_: &_| None;
        let f_evaluator = |g, h, _: &_| g + h;
        let node = FNodeMessage::<_>::generate_root_node(
            state,
            0,
            &model,
            &h_evaluator,
            &f_evaluator,
            None,
        );
        assert!(node.is_none());
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
        let expected_state = transition.apply(&state, &model.table_registry);
        let h_evaluator = |_: &_| Some(0);
        let f_evaluator = |g, h, _: &_| g + h;
        let node = FNode::generate_root_node(state, 0, &model, &h_evaluator, &f_evaluator, None);
        assert!(node.is_some());
        let node = node.unwrap();

        let h_evaluator = |_: &_| Some(0);
        let f_evaluator = |g, h, _: &_| g + h;
        let successor = node.generate_sendable_successor_node(
            Arc::new(transition.clone()),
            &model,
            &h_evaluator,
            &f_evaluator,
            None,
        );
        assert!(successor.is_some());
        let successor = successor.unwrap();

        assert_eq!(&successor.node.state, &expected_state);
        assert_eq!(successor.node.priority, -1);
        assert_eq!(
            successor.node.transitions,
            Some(Arc::new(ArcChain::new(None, Arc::new(transition))))
        );
        assert_eq!(successor.h, 0);
        assert_eq!(successor.f, -1);
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
        let expected_state = transition.apply(&state, &model.table_registry);
        let h_evaluator = |_: &_| Some(0);
        let f_evaluator = |g, h, _: &_| g + h;
        let node = FNode::generate_root_node(state, 0, &model, &h_evaluator, &f_evaluator, None);
        assert!(node.is_some());
        let node = node.unwrap();

        let h_evaluator = |_: &_| Some(0);
        let f_evaluator = |g, h, _: &_| g + h;
        let successor = node.generate_sendable_successor_node(
            Arc::new(transition.clone()),
            &model,
            &h_evaluator,
            &f_evaluator,
            None,
        );
        assert!(successor.is_some());
        let successor = successor.unwrap();

        assert_eq!(&successor.node.state, &expected_state);
        assert_eq!(successor.node.priority, 1);
        assert_eq!(
            successor.node.transitions,
            Some(Arc::new(ArcChain::new(None, Arc::new(transition))))
        );
        assert_eq!(successor.h, 0);
        assert_eq!(successor.f, 1);
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
        let h_evaluator = |_: &_| Some(0);
        let f_evaluator = |g, h, _: &_| g + h;
        let node = FNode::generate_root_node(state, 0, &model, &h_evaluator, &f_evaluator, None);
        assert!(node.is_some());
        let node = node.unwrap();

        let mut transition = Transition::default();
        let result = transition.add_effect(v1, v1 + 1);
        assert!(result.is_ok());
        let result = transition.add_effect(v2, v2 + 1);
        assert!(result.is_ok());
        transition.set_cost(IntegerExpression::Cost + 1);

        let h_evaluator = |_: &_| Some(0);
        let f_evaluator = |g, h, _: &_| g + h;
        let result = node.generate_sendable_successor_node(
            Arc::new(transition),
            &model,
            &h_evaluator,
            &f_evaluator,
            None,
        );
        assert_eq!(result, None);
    }

    #[test]
    fn test_generate_sendable_successor_pruned_by_bound_min() {
        let mut model = Model::default();
        model.set_minimize();
        let v1 = model.add_integer_resource_variable("v1", true, 0);
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let v2 = model.add_integer_resource_variable("v2", false, 0);
        assert!(v2.is_ok());
        let v2 = v2.unwrap();

        let state = model.target.clone();
        let h_evaluator = |_: &_| Some(0);
        let f_evaluator = |g, h, _: &_| g + h;
        let node = FNode::<_, _, _, ArcChain<_>, _>::generate_root_node(
            state,
            0,
            &model,
            &h_evaluator,
            &f_evaluator,
            None,
        );
        assert!(node.is_some());
        let node = node.unwrap();

        let mut transition = Transition::default();
        let result = transition.add_effect(v1, v1 + 1);
        assert!(result.is_ok());
        let result = transition.add_effect(v2, v2 + 1);
        assert!(result.is_ok());
        transition.set_cost(IntegerExpression::Cost + 1);

        let result = node.generate_successor_node(
            Arc::new(transition),
            &model,
            &h_evaluator,
            &f_evaluator,
            Some(0),
        );
        assert_eq!(result, None);
    }

    #[test]
    fn test_generate_sendable_successor_pruned_by_bound_max() {
        let mut model = Model::default();
        model.set_maximize();
        let v1 = model.add_integer_resource_variable("v1", true, 0);
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let v2 = model.add_integer_resource_variable("v2", false, 0);
        assert!(v2.is_ok());
        let v2 = v2.unwrap();

        let state = model.target.clone();
        let h_evaluator = |_: &_| Some(0);
        let f_evaluator = |g, h, _: &_| g + h;
        let node = FNode::generate_root_node(state, 0, &model, &h_evaluator, &f_evaluator, None);
        assert!(node.is_some());
        let node = node.unwrap();

        let mut transition = Transition::default();
        let result = transition.add_effect(v1, v1 + 1);
        assert!(result.is_ok());
        let result = transition.add_effect(v2, v2 + 1);
        assert!(result.is_ok());
        transition.set_cost(IntegerExpression::Cost + 1);

        let h_evaluator = |_: &_| Some(0);
        let f_evaluator = |g, h, _: &_| g + h;
        let result = node.generate_sendable_successor_node(
            Arc::new(transition),
            &model,
            &h_evaluator,
            &f_evaluator,
            Some(2),
        );
        assert_eq!(result, None);
    }

    #[test]
    fn test_generate_sendable_successor_pruned_by_h() {
        let mut model = Model::default();
        let v1 = model.add_integer_resource_variable("v1", true, 0);
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let v2 = model.add_integer_resource_variable("v2", false, 0);
        assert!(v2.is_ok());
        let v2 = v2.unwrap();

        let state = model.target.clone();
        let h_evaluator = |_: &_| Some(0);
        let f_evaluator = |g, h, _: &_| g + h;
        let node = FNode::generate_root_node(state, 0, &model, &h_evaluator, &f_evaluator, None);
        assert!(node.is_some());
        let node = node.unwrap();

        let mut transition = Transition::default();
        let result = transition.add_effect(v1, v1 + 1);
        assert!(result.is_ok());
        let result = transition.add_effect(v2, v2 + 1);
        assert!(result.is_ok());
        transition.set_cost(IntegerExpression::Cost + 1);

        let h_evaluator = |_: &_| None;
        let f_evaluator = |g, h, _: &_| g + h;
        let result = node.generate_sendable_successor_node(
            Arc::new(transition),
            &model,
            &h_evaluator,
            &f_evaluator,
            None,
        );
        assert_eq!(result, None);
    }
}
