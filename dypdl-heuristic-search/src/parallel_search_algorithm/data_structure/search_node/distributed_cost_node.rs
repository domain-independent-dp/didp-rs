use super::super::arc_chain::ArcChain;
use super::cost_node_message::CostNodeMessage;
use crate::search_algorithm::data_structure::{
    GetTransitions, HashableSignatureVariables, StateInformation, TransitionChain,
};
use crate::search_algorithm::{BfsNode, StateInRegistry, StateRegistry};
use dypdl::variable_type::Numeric;
use dypdl::{Model, ReduceFunction, Transition, TransitionInterface};
use std::cell::Cell;
use std::cmp::Ordering;
use std::fmt::{Debug, Display};
use std::ops::Deref;
use std::rc::Rc;
use std::sync::Arc;

/// Node ordered by its cost.
///
/// This node is expected be used in a distributed parallel search algorithm,
/// where a state belongs to a unique thread, but the transition chain is shared among threads.
///
/// In minimization, a node having a lower cost is `Greater` in `Ord`.
/// In maximization , a node having a higher cost is `Greater` in `Ord`.
///
/// This node does not have information about a bound.
#[derive(Debug, Clone)]
pub struct DistributedCostNode<T, V = Transition>
where
    T: Numeric,
    V: TransitionInterface + Clone,
    Transition: From<V>,
{
    state: StateInRegistry<Rc<HashableSignatureVariables>>,
    priority: T,
    closed: Cell<bool>,
    transitions: Option<Arc<ArcChain<V>>>,
}

impl<V, T> From<CostNodeMessage<T, V>> for DistributedCostNode<T, V>
where
    T: Numeric,
    V: TransitionInterface + Clone,
    Transition: From<V>,
{
    fn from(message: CostNodeMessage<T, V>) -> Self {
        DistributedCostNode {
            state: message.state.into(),
            priority: message.priority,
            closed: Cell::new(false),
            transitions: message.transitions,
        }
    }
}

impl<T, V> DistributedCostNode<T, V>
where
    T: Numeric + Ord,
    V: TransitionInterface + Clone,
    Transition: From<V>,
{
    fn new(
        state: StateInRegistry,
        priority: T,
        parent: Option<&Self>,
        transition: Option<Arc<V>>,
    ) -> Self {
        let transitions = transition.map(|transition| {
            Arc::new(ArcChain::new(
                parent.and_then(|parent| parent.transitions.clone()),
                transition,
            ))
        });

        DistributedCostNode {
            state,
            priority,
            closed: Cell::new(false),
            transitions,
        }
    }

    /// Generates a root search node given a state, its cost, and a DyPDL model.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    /// use dypdl_heuristic_search::DistributedCostNode;
    /// use dypdl_heuristic_search::search_algorithm::StateInRegistry;
    /// use dypdl_heuristic_search::search_algorithm::data_structure::{
    ///     GetTransitions, StateInformation,
    /// };
    ///
    /// let mut model = Model::default();
    /// model.add_integer_variable("variable", 0).unwrap();
    ///
    /// let state = model.target.clone();
    /// let cost = 0;
    /// let node = DistributedCostNode::<_>::generate_root_node(state, cost, &model);
    /// assert_eq!(node.state(), &StateInRegistry::from(model.target.clone()));
    /// assert_eq!(node.cost(&model), cost);
    /// assert!(!node.is_closed());
    /// assert_eq!(node.transitions(), vec![]);
    /// ```
    pub fn generate_root_node<S>(state: S, cost: T, model: &Model) -> Self
    where
        StateInRegistry: From<S>,
    {
        let state = StateInRegistry::from(state);
        let priority = if model.reduce_function == ReduceFunction::Max {
            cost
        } else {
            -cost
        };

        DistributedCostNode::new(state, priority, None, None)
    }

    /// Generates a successor node given a transition and a DyPDL model.
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
    /// use dypdl_heuristic_search::DistributedCostNode;
    /// use dypdl_heuristic_search::search_algorithm::StateInRegistry;
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
    /// let node = DistributedCostNode::<_>::generate_root_node(state, cost, &model);
    ///
    /// let mut transition = Transition::new("transition");
    /// transition.set_cost(IntegerExpression::Cost + 1);
    /// transition.add_effect(variable, variable + 1).unwrap();
    /// let expected_state: StateInRegistry = transition.apply(&model.target, &model.table_registry);
    ///
    /// let node = node.generate_successor_node(Arc::new(transition.clone()), &model);
    /// assert!(node.is_some());
    /// let node = node.unwrap();
    /// assert_eq!(node.state(), &expected_state);
    /// assert_eq!(node.cost(&model), 1);
    /// assert!(!node.is_closed());
    /// assert_eq!(node.transitions(), vec![transition]);
    /// ```
    pub fn generate_successor_node(&self, transition: Arc<V>, model: &Model) -> Option<Self> {
        let cost = self.cost(model);
        let (state, cost) =
            model.generate_successor_state(&self.state, cost, transition.as_ref(), None)?;
        let priority = if model.reduce_function == ReduceFunction::Max {
            cost
        } else {
            -cost
        };

        Some(DistributedCostNode::new(
            state,
            priority,
            Some(self),
            Some(transition),
        ))
    }

    /// Generates a sendable successor node message given a transition and a DyPDL model.
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
    /// use dypdl_heuristic_search::DistributedCostNode;
    /// use dypdl_heuristic_search::search_algorithm::StateInRegistry;
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
    /// let node = DistributedCostNode::<_>::generate_root_node(state, cost, &model);
    ///
    /// let mut transition = Transition::new("transition");
    /// transition.set_cost(IntegerExpression::Cost + 1);
    /// transition.add_effect(variable, variable + 1).unwrap();
    /// let expected_state: StateInRegistry = transition.apply(&model.target, &model.table_registry);
    ///
    /// let node = node.generate_sendable_successor_node(Arc::new(transition.clone()), &model);
    /// assert!(node.is_some());
    /// let node = DistributedCostNode::from(node.unwrap());
    /// assert_eq!(node.state(), &expected_state);
    /// assert_eq!(node.cost(&model), 1);
    /// assert!(!node.is_closed());
    /// assert_eq!(node.transitions(), vec![transition]);
    /// ```
    pub fn generate_sendable_successor_node(
        &self,
        transition: Arc<V>,
        model: &Model,
    ) -> Option<CostNodeMessage<T, V>> {
        let cost = self.cost(model);
        let (state, cost) =
            model.generate_successor_state(&self.state, cost, transition.as_ref(), None)?;
        let priority = if model.reduce_function == ReduceFunction::Max {
            cost
        } else {
            -cost
        };

        let transitions = Arc::new(ArcChain::new(self.transitions.clone(), transition));

        Some(CostNodeMessage {
            state,
            priority,
            transitions: Some(transitions),
        })
    }

    /// Generates a successor node given a transition and inserts it into a state registry.
    ///
    /// Returns the successor node and whether a new entry is generated or not.
    /// If the successor node dominates an existing non-closed node in the registry, the second return value is `false`.
    /// Returns `None` if the successor state is pruned by a state constraint, or the successor node is dominated.
    ///
    /// # Panics
    ///
    /// If an expression used in the transition is invalid.
    /// Generates a successor node given a transition and inserts it into a state registry.
    ///
    /// Returns the successor node and whether a new entry is generated or not.
    /// If the successor node dominates an existing non-closed node in the registry, the second return value is `false`.
    /// Returns `None` if the successor state is pruned by a state constraint, or the successor node is dominated.
    ///
    /// # Panics
    ///
    /// If an expression used in the transition is invalid.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    /// use dypdl_heuristic_search::DistributedCostNode;
    /// use dypdl_heuristic_search::search_algorithm::{
    ///     StateInRegistry, StateRegistry,
    /// };
    /// use dypdl_heuristic_search::search_algorithm::data_structure::{
    ///     GetTransitions, StateInformation,
    /// };
    /// use std::sync::Arc;
    ///
    /// let mut model = Model::default();
    /// let variable = model.add_integer_variable("variable", 0).unwrap();
    /// let mut registry = StateRegistry::<_, _, Arc<_>, _, _>::new(Arc::new(model.clone()));
    ///
    /// let state = model.target.clone();
    /// let cost = 0;
    /// let node = DistributedCostNode::<_>::generate_root_node(state, cost, &model);
    ///
    /// let mut transition = Transition::new("transition");
    /// transition.set_cost(IntegerExpression::Cost + 1);
    /// transition.add_effect(variable, variable + 1).unwrap();
    /// let expected_state: StateInRegistry = transition.apply(
    ///     &model.target, &model.table_registry,
    /// );
    ///
    /// let result = node.insert_successor_node(Arc::new(transition.clone()), &mut registry);
    /// assert!(result.is_some());
    /// let (node, generated) = result.unwrap();
    /// assert!(generated);
    /// assert_eq!(node.state(), &expected_state);
    /// assert_eq!(node.cost(&model), 1);
    /// assert!(!node.is_closed());
    /// assert_eq!(node.transitions(), vec![transition]);
    /// ```
    pub fn insert_successor_node<N, M>(
        &self,
        transition: Arc<V>,
        registry: &mut StateRegistry<T, Self, N, Rc<HashableSignatureVariables>, M>,
    ) -> Option<(N, bool)>
    where
        N: Deref<Target = Self> + From<Self> + Clone,
        M: Deref<Target = Model> + Clone,
    {
        let model = registry.model();
        let (state, cost) = model.generate_successor_state(
            self.state(),
            self.cost(model),
            transition.as_ref(),
            None,
        )?;
        let priority = if model.reduce_function == ReduceFunction::Max {
            cost
        } else {
            -cost
        };
        let constructor = |state: StateInRegistry, _: T, _: Option<&DistributedCostNode<T, V>>| {
            Some(DistributedCostNode::new(
                state,
                priority,
                Some(self),
                Some(transition),
            ))
        };

        if let Some((successor, dominated)) = registry.insert_with(state, cost, constructor) {
            let mut generated = true;

            if let Some(dominated) = dominated {
                if !dominated.is_closed() {
                    dominated.close();
                    generated = false;
                }
            }

            Some((successor, generated))
        } else {
            None
        }
    }
}

impl<T, V> PartialEq for DistributedCostNode<T, V>
where
    T: Numeric + PartialOrd,
    V: TransitionInterface + Clone,
    Transition: From<V>,
{
    /// Nodes are compared by their costs.
    /// This does not mean that the nodes are the same.
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

impl<T, V> Eq for DistributedCostNode<T, V>
where
    T: Numeric + Ord,
    V: TransitionInterface + Clone,
    Transition: From<V>,
{
}

impl<T, V> Ord for DistributedCostNode<T, V>
where
    T: Numeric + Ord,
    V: TransitionInterface + Clone,
    Transition: From<V>,
{
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority.cmp(&other.priority)
    }
}

impl<T, V> PartialOrd for DistributedCostNode<T, V>
where
    T: Numeric + Ord,
    V: TransitionInterface + Clone,
    Transition: From<V>,
{
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T, V> StateInformation<T> for DistributedCostNode<T, V>
where
    T: Numeric,
    V: TransitionInterface + Clone,
    Transition: From<V>,
{
    #[inline]
    fn state(&self) -> &StateInRegistry {
        &self.state
    }

    #[inline]
    fn state_mut(&mut self) -> &mut StateInRegistry {
        &mut self.state
    }

    #[inline]
    fn cost(&self, model: &Model) -> T {
        if model.reduce_function == ReduceFunction::Max {
            self.priority
        } else {
            -self.priority
        }
    }

    #[inline]
    fn bound(&self, _: &Model) -> Option<T> {
        None
    }

    #[inline]
    fn is_closed(&self) -> bool {
        self.closed.get()
    }

    #[inline]
    fn close(&self) {
        self.closed.set(true);
    }
}

impl<T, V> GetTransitions<V> for DistributedCostNode<T, V>
where
    T: Numeric,
    V: TransitionInterface + Clone,
    Transition: From<V>,
{
    #[inline]
    fn transitions(&self) -> Vec<V> {
        self.transitions
            .as_ref()
            .map_or_else(Vec::default, |transitions| transitions.transitions())
    }
}

impl<T, V> BfsNode<T, V> for DistributedCostNode<T, V>
where
    T: Numeric + Ord + Display,
    V: TransitionInterface + Clone,
    Transition: From<V>,
{
    #[inline]
    fn ordered_by_bound() -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dypdl::expression::*;
    use dypdl::prelude::*;

    #[test]
    fn ordered_by_bound() {
        assert!(!DistributedCostNode::<Integer>::ordered_by_bound());
    }

    #[test]
    fn generate_root_node_min() {
        let mut model = dypdl::Model::default();
        model.set_minimize();
        let variable = model.add_integer_variable("variable", 0);
        assert!(variable.is_ok());
        let state = model.target.clone();
        let mut expected_state = StateInRegistry::from(state.clone());
        let mut node = DistributedCostNode::<_>::generate_root_node(state, 1, &model);
        assert_eq!(node.state(), &expected_state);
        assert_eq!(node.state_mut(), &mut expected_state);
        assert_eq!(node.cost(&model), 1);
        assert_eq!(node.bound(&model), None);
        assert!(!node.is_closed());
        assert_eq!(node.transitions(), vec![]);
    }

    #[test]
    fn generate_root_node_max() {
        let mut model = dypdl::Model::default();
        model.set_maximize();
        let variable = model.add_integer_variable("variable", 0);
        assert!(variable.is_ok());
        let state = model.target.clone();
        let mut expected_state = StateInRegistry::from(state.clone());
        let mut node = DistributedCostNode::<_>::generate_root_node(state, 1, &model);
        assert_eq!(node.state(), &expected_state);
        assert_eq!(node.state_mut(), &mut expected_state);
        assert_eq!(node.cost(&model), 1);
        assert_eq!(node.bound(&model), None);
        assert!(!node.is_closed());
        assert_eq!(node.transitions(), vec![]);
    }

    #[test]
    fn close() {
        let model = dypdl::Model::default();
        let state = model.target.clone();
        let node = DistributedCostNode::<_>::generate_root_node(state, 0, &model);
        assert!(!node.is_closed());
        node.close();
        assert!(node.is_closed());
    }

    #[test]
    fn generate_successor_some_min() {
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
        let mut expected_state: StateInRegistry = transition.apply(&state, &model.table_registry);
        let node = DistributedCostNode::generate_root_node(state, 0, &model);

        let successor = node.generate_successor_node(Arc::new(transition.clone()), &model);
        assert!(successor.is_some());
        let mut successor = successor.unwrap();
        assert_eq!(successor.state(), &expected_state);
        assert_eq!(successor.state_mut(), &mut expected_state);
        assert_eq!(successor.cost(&model), 1);
        assert_eq!(successor.bound(&model), None);
        assert!(!successor.is_closed());
        assert_eq!(successor.transitions(), vec![transition]);
    }

    #[test]
    fn generate_successor_some_max() {
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
        let mut expected_state: StateInRegistry = transition.apply(&state, &model.table_registry);
        let node = DistributedCostNode::generate_root_node(state, 0, &model);

        let successor = node.generate_successor_node(Arc::new(transition.clone()), &model);
        assert!(successor.is_some());
        let mut successor = successor.unwrap();
        assert_eq!(successor.state(), &expected_state);
        assert_eq!(successor.state_mut(), &mut expected_state);
        assert_eq!(successor.cost(&model), 1);
        assert_eq!(successor.bound(&model), None);
        assert!(!successor.is_closed());
        assert_eq!(successor.transitions(), vec![transition]);
    }

    #[test]
    fn generate_successor_pruned_by_constraint() {
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
        let node = DistributedCostNode::generate_root_node(state, 0, &model);

        let mut transition = Transition::default();
        let result = transition.add_effect(v1, v1 + 1);
        assert!(result.is_ok());
        let result = transition.add_effect(v2, v2 + 1);
        assert!(result.is_ok());
        transition.set_cost(IntegerExpression::Cost + 1);

        let result = node.generate_successor_node(Arc::new(transition), &model);
        assert_eq!(result, None);
    }

    #[test]
    fn generate_sendable_successor_some_min() {
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
        let mut expected_state: StateInRegistry = transition.apply(&state, &model.table_registry);
        let node = DistributedCostNode::generate_root_node(state, 0, &model);

        let successor = node.generate_sendable_successor_node(Arc::new(transition.clone()), &model);
        assert!(successor.is_some());
        let mut successor = DistributedCostNode::from(successor.unwrap());
        assert_eq!(successor.state(), &expected_state);
        assert_eq!(successor.state_mut(), &mut expected_state);
        assert_eq!(successor.cost(&model), 1);
        assert_eq!(successor.bound(&model), None);
        assert!(!successor.is_closed());
        assert_eq!(successor.transitions(), vec![transition]);
    }

    #[test]
    fn generate_sendable_successor_some_max() {
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
        let mut expected_state: StateInRegistry = transition.apply(&state, &model.table_registry);
        let node = DistributedCostNode::generate_root_node(state, 0, &model);

        let successor = node.generate_sendable_successor_node(Arc::new(transition.clone()), &model);
        assert!(successor.is_some());
        let mut successor = DistributedCostNode::from(successor.unwrap());
        assert_eq!(successor.state(), &expected_state);
        assert_eq!(successor.state_mut(), &mut expected_state);
        assert_eq!(successor.cost(&model), 1);
        assert_eq!(successor.bound(&model), None);
        assert!(!successor.is_closed());
        assert_eq!(successor.transitions(), vec![transition]);
    }

    #[test]
    fn generate_sendable_successor_pruned_by_constraint() {
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
        let node = DistributedCostNode::generate_root_node(state, 0, &model);

        let mut transition = Transition::default();
        let result = transition.add_effect(v1, v1 + 1);
        assert!(result.is_ok());
        let result = transition.add_effect(v2, v2 + 1);
        assert!(result.is_ok());
        transition.set_cost(IntegerExpression::Cost + 1);

        let result = node.generate_successor_node(Arc::new(transition), &model);
        assert_eq!(result, None);
    }

    #[test]
    fn insert_successor_non_dominance_min() {
        let mut model = Model::default();
        model.set_minimize();
        let v1 = model.add_integer_resource_variable("v1", true, 0);
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let v2 = model.add_integer_resource_variable("v2", false, 0);
        assert!(v2.is_ok());
        let v2 = v2.unwrap();
        let model = Rc::new(model);

        let state = StateInRegistry::from(model.target.clone());
        let mut registry = StateRegistry::<_, DistributedCostNode<_>>::new(model.clone());

        let mut transition = Transition::default();
        let result = transition.add_effect(v1, v1 + 1);
        assert!(result.is_ok());
        let result = transition.add_effect(v2, v2 + 1);
        assert!(result.is_ok());
        transition.set_cost(IntegerExpression::Cost + 1);

        let expected_state: StateInRegistry = transition.apply(&state, &model.table_registry);
        let node = DistributedCostNode::generate_root_node(state, 0, &model);
        let result = registry.insert(node.clone());
        assert!(result.is_some());

        let result = node.insert_successor_node(Arc::new(transition.clone()), &mut registry);
        assert!(result.is_some());
        let (successor, generated) = result.unwrap();
        assert_eq!(successor.state(), &expected_state);
        assert_eq!(successor.cost(&model), 1);
        assert_eq!(successor.bound(&model), None);
        assert!(!successor.is_closed());
        assert_eq!(successor.transitions(), vec![transition],);
        assert!(generated);
        assert!(!node.is_closed());
    }

    #[test]
    fn insert_successor_non_dominance_max() {
        let mut model = Model::default();
        model.set_maximize();
        let v1 = model.add_integer_resource_variable("v1", true, 0);
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let v2 = model.add_integer_resource_variable("v2", false, 0);
        assert!(v2.is_ok());
        let v2 = v2.unwrap();
        let model = Rc::new(model);

        let state = StateInRegistry::from(model.target.clone());
        let mut registry = StateRegistry::<_, DistributedCostNode<_>>::new(model.clone());

        let mut transition = Transition::default();
        let result = transition.add_effect(v1, v1 + 1);
        assert!(result.is_ok());
        let result = transition.add_effect(v2, v2 + 1);
        assert!(result.is_ok());
        transition.set_cost(IntegerExpression::Cost + 1);

        let expected_state: StateInRegistry = transition.apply(&state, &model.table_registry);
        let node = DistributedCostNode::generate_root_node(state, 0, &model);
        let result = registry.insert(node.clone());
        assert!(result.is_some());

        let result = node.insert_successor_node(Arc::new(transition.clone()), &mut registry);
        assert!(result.is_some());
        let (successor, generated) = result.unwrap();
        assert_eq!(successor.state(), &expected_state);
        assert_eq!(successor.cost(&model), 1);
        assert_eq!(successor.bound(&model), None);
        assert!(!successor.is_closed());
        assert_eq!(successor.transitions(), vec![transition],);
        assert!(generated);
        assert!(!node.is_closed());
    }

    #[test]
    fn insert_successor_pruned_by_constraint() {
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

        let state = StateInRegistry::from(model.target.clone());
        let mut registry = StateRegistry::<_, DistributedCostNode<_>>::new(Rc::new(model.clone()));

        let node = DistributedCostNode::generate_root_node(state, 0, &model);

        let mut transition = Transition::default();
        let result = transition.add_effect(v1, v1 + 1);
        assert!(result.is_ok());
        let result = transition.add_effect(v2, v2 + 1);
        assert!(result.is_ok());
        transition.set_cost(IntegerExpression::Cost + 1);

        let result = node.insert_successor_node(Arc::new(transition), &mut registry);
        assert_eq!(result, None);
        assert!(!node.is_closed());
    }

    #[test]
    fn insert_successor_dominating_min() {
        let mut model = Model::default();
        model.set_minimize();
        let v1 = model.add_integer_resource_variable("v1", false, 0);
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

        let state = StateInRegistry::from(model.target.clone());
        let expected_state: StateInRegistry = transition.apply(&state, &model.table_registry);
        let node = DistributedCostNode::generate_root_node(state, 0, &model);
        let mut registry = StateRegistry::<_, DistributedCostNode<_>>::new(Rc::new(model.clone()));
        let result = registry.insert(node);
        assert!(result.is_some());
        let (node, dominated) = result.unwrap();
        assert!(dominated.is_none());

        let result = node.insert_successor_node(Arc::new(transition.clone()), &mut registry);
        assert!(result.is_some());
        let (successor, generated) = result.unwrap();
        assert_eq!(successor.state(), &expected_state);
        assert_eq!(successor.cost(&model), 0);
        assert_eq!(successor.bound(&model), None);
        assert!(!successor.is_closed());
        assert_eq!(successor.transitions(), vec![transition]);
        assert!(!generated);
        assert!(node.is_closed());
    }

    #[test]
    fn insert_successor_dominating_max() {
        let mut model = Model::default();
        model.set_maximize();
        let v1 = model.add_integer_resource_variable("v1", false, 0);
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

        let state = StateInRegistry::from(model.target.clone());
        let expected_state: StateInRegistry = transition.apply(&state, &model.table_registry);
        let node = DistributedCostNode::generate_root_node(state, 0, &model);
        let mut registry = StateRegistry::<_, DistributedCostNode<_>>::new(Rc::new(model.clone()));
        let result = registry.insert(node);
        assert!(result.is_some());
        let (node, dominated) = result.unwrap();
        assert!(dominated.is_none());

        let result = node.insert_successor_node(Arc::new(transition.clone()), &mut registry);
        assert!(result.is_some());
        let (successor, generated) = result.unwrap();
        assert_eq!(successor.state(), &expected_state);
        assert_eq!(successor.cost(&model), 1);
        assert_eq!(successor.bound(&model), None);
        assert!(!successor.is_closed());
        assert_eq!(successor.transitions(), vec![transition]);
        assert!(!generated);
        assert!(node.is_closed());
    }

    #[test]
    fn insert_successor_dominated_min() {
        let mut model = Model::default();
        model.set_minimize();
        let v1 = model.add_integer_resource_variable("v1", true, 0);
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let v2 = model.add_integer_resource_variable("v2", true, 0);
        assert!(v2.is_ok());
        let v2 = v2.unwrap();

        let mut transition = Transition::default();
        let result = transition.add_effect(v1, v1 + 1);
        assert!(result.is_ok());
        let result = transition.add_effect(v2, v2 + 1);
        assert!(result.is_ok());

        let state = StateInRegistry::from(model.target.clone());
        let node = DistributedCostNode::generate_root_node(state, 0, &model);
        let mut registry = StateRegistry::<_, DistributedCostNode<_>>::new(Rc::new(model.clone()));
        let result = registry.insert(node);
        assert!(result.is_some());
        let (node, dominated) = result.unwrap();
        assert!(dominated.is_none());

        let result = node.insert_successor_node(Arc::new(transition.clone()), &mut registry);
        assert!(result.is_none());
    }

    #[test]
    fn insert_successor_dominated_max() {
        let mut model = Model::default();
        model.set_maximize();
        let v1 = model.add_integer_resource_variable("v1", true, 0);
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let v2 = model.add_integer_resource_variable("v2", true, 0);
        assert!(v2.is_ok());
        let v2 = v2.unwrap();

        let mut transition = Transition::default();
        let result = transition.add_effect(v1, v1 + 1);
        assert!(result.is_ok());
        let result = transition.add_effect(v2, v2 + 1);
        assert!(result.is_ok());

        let state = StateInRegistry::from(model.target.clone());
        let node = DistributedCostNode::generate_root_node(state, 0, &model);
        let mut registry = StateRegistry::<_, DistributedCostNode<_>>::new(Rc::new(model.clone()));
        let result = registry.insert(node);
        assert!(result.is_some());
        let (node, dominated) = result.unwrap();
        assert!(dominated.is_none());

        let result = node.insert_successor_node(Arc::new(transition.clone()), &mut registry);
        assert!(result.is_none());
    }

    #[test]
    fn cmp_min() {
        let mut model = Model::default();
        model.set_minimize();
        let v1 = model.add_integer_resource_variable("v1", true, 0);
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let v2 = model.add_integer_resource_variable("v2", false, 0);
        assert!(v2.is_ok());
        let v2 = v2.unwrap();

        let state = model.target.clone();
        let node1 = DistributedCostNode::<_>::generate_root_node(state, 0, &model);
        let node1 = Rc::new(node1);

        let mut transition = Transition::default();
        let result = transition.add_effect(v1, v1 + 1);
        assert!(result.is_ok());
        let result = transition.add_effect(v2, v2 + 1);
        assert!(result.is_ok());
        let node2 = node1.generate_successor_node(Arc::new(transition), &model);
        assert!(node2.is_some());
        let node2 = Rc::new(node2.unwrap());

        let mut transition = Transition::default();
        transition.set_cost(IntegerExpression::Cost + 1);
        let mut registry = StateRegistry::<_, DistributedCostNode<_>>::new(Rc::new(model));
        let result = node1.insert_successor_node(Arc::new(transition), &mut registry);
        assert!(result.is_some());
        let (node3, _) = result.unwrap();

        assert!(node1 == node1);
        assert!(node1 >= node1);
        assert!(node1 == node2);
        assert!(node1 >= node2);
        assert!(node1 <= node2);
        assert!(node1 > node3);
        assert!(node1 >= node3);
        assert!(node1 != node3);
    }

    #[test]
    fn cmp_max() {
        let mut model = Model::default();
        model.set_maximize();
        let v1 = model.add_integer_resource_variable("v1", true, 0);
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let v2 = model.add_integer_resource_variable("v2", false, 0);
        assert!(v2.is_ok());
        let v2 = v2.unwrap();

        let state = model.target.clone();
        let node1 = DistributedCostNode::<_>::generate_root_node(state, 0, &model);
        let node1 = Rc::new(node1);

        let mut transition = Transition::default();
        let result = transition.add_effect(v1, v1 + 1);
        assert!(result.is_ok());
        let result = transition.add_effect(v2, v2 + 1);
        assert!(result.is_ok());
        let node2 = node1.generate_successor_node(Arc::new(transition), &model);
        assert!(node2.is_some());
        let node2 = Rc::new(node2.unwrap());

        let mut transition = Transition::default();
        transition.set_cost(IntegerExpression::Cost + 1);
        let mut registry = StateRegistry::<_, DistributedCostNode<_>>::new(Rc::new(model));
        let result = node1.insert_successor_node(Arc::new(transition), &mut registry);
        assert!(result.is_some());
        let (node3, _) = result.unwrap();

        assert!(node1 == node1);
        assert!(node1 >= node1);
        assert!(node1 == node2);
        assert!(node1 >= node2);
        assert!(node1 <= node2);
        assert!(node1 < node3);
        assert!(node1 <= node3);
        assert!(node1 != node3);
    }
}
