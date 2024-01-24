use super::super::state_registry::{StateInRegistry, StateInformation, StateRegistry};
use super::super::transition_chain::{CreateTransitionChain, GetTransitions, RcChain};
use super::BfsNode;
use dypdl::variable_type::Numeric;
use dypdl::{Model, ReduceFunction, Transition, TransitionInterface};
use std::cell::Cell;
use std::cmp::Ordering;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use std::ops::Deref;
use std::rc::Rc;

/// Node ordered by its priority.
///
/// A node having a higher priority is `Greater` in `Ord`.
#[derive(Clone, Debug)]
pub struct CostNode<T, V = Transition, R = Rc<V>, C = RcChain<V>, P = Rc<C>> {
    state: StateInRegistry,
    /// Priority.
    pub priority: T,
    closed: Cell<bool>,
    transition_chain: Option<P>,
    _phantom: PhantomData<(V, R, C)>,
}

impl<T, V, R, C, P> PartialEq for CostNode<T, V, R, C, P>
where
    T: PartialEq,
{
    /// Nodes are compared by their priorities.
    /// This does not mean that the nodes are the same.
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

impl<T, V, R, C, P> Eq for CostNode<T, V, R, C, P> where T: Eq {}

impl<T, V, R, C, P> Ord for CostNode<T, V, R, C, P>
where
    T: Ord,
{
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority.cmp(&other.priority)
    }
}

impl<T, V, R, C, P> PartialOrd for CostNode<T, V, R, C, P>
where
    T: Ord,
{
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T, V, R, C, P> StateInformation<T> for CostNode<T, V, R, C, P>
where
    T: Numeric,
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

impl<T, V, R, C, P> GetTransitions<V> for CostNode<T, V, R, C, P>
where
    V: TransitionInterface + Clone,
    Transition: From<V>,
    C: GetTransitions<V>,
    P: Deref<Target = C>,
{
    #[inline]
    fn transitions(&self) -> Vec<V> {
        self.transition_chain
            .as_ref()
            .map_or_else(Vec::default, |transitions| transitions.transitions())
    }

    #[inline]
    fn last(&self) -> Option<&V> {
        self.transition_chain
            .as_ref()
            .and_then(|transitions| transitions.last())
    }
}

impl<T, V, R, C, P> BfsNode<T, V> for CostNode<T, V, R, C, P>
where
    T: Numeric + Ord + Display,
    V: TransitionInterface + Clone,
    Transition: From<V>,
    C: GetTransitions<V>,
    P: Deref<Target = C>,
{
    #[inline]
    fn ordered_by_bound() -> bool {
        false
    }
}

impl<T, V, R, C, P> CostNode<T, V, R, C, P>
where
    T: Numeric + Ord,
    V: TransitionInterface + Clone,
    Transition: From<V>,
    R: Deref<Target = V>,
    C: CreateTransitionChain<R, P>,
    P: From<C> + Clone,
{
    /// Generate a new node given a state, its cost, a DyPDL model, and a transition chain.
    ///
    /// If the model is minimizing, the priority becomes the negative of the cost.
    /// If the model is maximizing, the priority becomes the cost.
    pub fn new(
        state: StateInRegistry,
        cost: T,
        model: &Model,
        transition_chain: Option<P>,
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

        Self::with_state_and_priority_and_transition_chain(state, priority, transition_chain)
    }

    /// Generate a new node given a state, its priority, and a transition chain.
    pub fn with_state_and_priority_and_transition_chain(
        state: StateInRegistry,
        priority: T,
        transition_chain: Option<P>,
    ) -> Self {
        CostNode {
            state,
            priority,
            closed: Cell::new(false),
            transition_chain,
            _phantom: PhantomData,
        }
    }

    /// Get the transition chain.
    pub fn transition_chain(&self) -> Option<P> {
        self.transition_chain.clone()
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
    /// let node = CostNode::<_>::generate_root_node(state, cost, &model);
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
        CostNode::new(state, cost, model, None)
    }

    /// Generates a successor node given a transition and a DyPDL model.
    ///
    /// Returns `None` if the successor state is pruned by a state constraint.
    ///
    /// If the model is minimizing, the priority becomes the negative of the cost.
    /// If the model is maximizing, the priority becomes the cost.
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
    /// use std::rc::Rc;
    ///
    /// let mut model = Model::default();
    /// let variable = model.add_integer_variable("variable", 0).unwrap();
    ///
    /// let state = model.target.clone();
    /// let cost = 0;
    /// let node = CostNode::<_>::generate_root_node(state, cost, &model);
    ///
    /// let mut transition = Transition::new("transition");
    /// transition.set_cost(IntegerExpression::Cost + 1);
    /// transition.add_effect(variable, variable + 1).unwrap();
    /// let expected_state: StateInRegistry = transition.apply(&model.target, &model.table_registry);
    ///
    /// let node = node.generate_successor_node(Rc::new(transition.clone()), &model);
    /// assert!(node.is_some());
    /// let node = node.unwrap();
    /// assert_eq!(node.state(), &expected_state);
    /// assert_eq!(node.cost(&model), 1);
    /// assert!(!node.is_closed());
    /// assert_eq!(node.transitions(), vec![transition]);
    /// ```
    pub fn generate_successor_node(&self, transition: R, model: &Model) -> Option<Self> {
        let cost = self.cost(model);
        let (state, cost) =
            model.generate_successor_state(&self.state, cost, transition.deref(), None)?;
        let transitions = P::from(C::new(self.transition_chain.clone(), transition));

        Some(CostNode::new(state, cost, model, Some(transitions)))
    }

    /// Generates a successor node given a transition and inserts it into a state registry.
    ///
    /// Returns the successor node and whether a new entry is generated or not.
    /// If the successor node dominates an existing non-closed node in the registry, the second return value is `false`.
    /// Returns `None` if the successor state is pruned by a state constraint, or the successor node is dominated.
    ///
    /// If the model is minimizing, the priority becomes the negative of the cost.
    /// If the model is maximizing, the priority becomes the cost.
    ///
    /// # Panics
    ///
    /// If an expression used in the transition is invalid.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    /// use dypdl_heuristic_search::search_algorithm::{
    ///     CostNode, StateInRegistry, StateRegistry,
    /// };
    /// use dypdl_heuristic_search::search_algorithm::data_structure::{
    ///     GetTransitions, StateInformation,
    /// };
    /// use std::rc::Rc;
    ///
    /// let mut model = Model::default();
    /// let variable = model.add_integer_variable("variable", 0).unwrap();
    /// let mut registry = StateRegistry::<_, CostNode<_>>::new(Rc::new(model.clone()));
    ///
    /// let state = model.target.clone();
    /// let cost = 0;
    /// let node = CostNode::<_>::generate_root_node(state, cost, &model);
    ///
    /// let mut transition = Transition::new("transition");
    /// transition.set_cost(IntegerExpression::Cost + 1);
    /// transition.add_effect(variable, variable + 1).unwrap();
    /// let expected_state: StateInRegistry = transition.apply(
    ///     &model.target, &model.table_registry,
    /// );
    ///
    /// let result = node.insert_successor_node(Rc::new(transition.clone()), &mut registry);
    /// assert!(result.is_some());
    /// let (node, generated) = result.unwrap();
    /// assert!(generated);
    /// assert_eq!(node.state(), &expected_state);
    /// assert_eq!(node.cost(&model), 1);
    /// assert!(!node.is_closed());
    /// assert_eq!(node.transitions(), vec![transition]);
    /// ```
    pub fn insert_successor_node<M>(
        &self,
        transition: R,
        registry: &mut StateRegistry<T, Self, M>,
    ) -> Option<(Rc<Self>, bool)>
    where
        M: Deref<Target = Model> + Clone,
    {
        let model = registry.model().clone();
        let (state, cost) = model.generate_successor_state(
            self.state(),
            self.cost(&model),
            transition.deref(),
            None,
        )?;
        let constructor = |state: StateInRegistry, _: T, _: Option<&Self>| {
            let transitions = P::from(C::new(self.transition_chain(), transition));
            Some(CostNode::new(state, cost, &model, Some(transitions)))
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

#[cfg(test)]
mod tests {
    use super::*;
    use dypdl::expression::*;
    use dypdl::prelude::*;

    #[test]
    fn test_new_min() {
        let mut model = Model::default();
        model.set_minimize();
        let v1 = model.add_integer_resource_variable("v1", true, 0);
        assert!(v1.is_ok());
        let v2 = model.add_integer_resource_variable("v2", false, 0);
        assert!(v2.is_ok());

        let mut state = StateInRegistry::from(model.target.clone());
        let cost = 10;
        let transition_chain = None;

        let mut node = CostNode::<_>::new(state.clone(), cost, &model, transition_chain);
        assert_eq!(node.priority, -10);
        assert_eq!(node.state(), &state);
        assert_eq!(node.state_mut(), &mut state);
        assert_eq!(node.cost(&model), 10);
        assert_eq!(node.bound(&model), None);
        assert!(!node.is_closed());
        assert_eq!(node.transitions(), vec![]);
        assert_eq!(node.last(), None);
        assert_eq!(node.transition_chain(), None);
    }

    #[test]
    fn test_new_max() {
        let mut model = Model::default();
        model.set_maximize();
        let v1 = model.add_integer_resource_variable("v1", true, 0);
        assert!(v1.is_ok());
        let v2 = model.add_integer_resource_variable("v2", false, 0);
        assert!(v2.is_ok());

        let mut state = StateInRegistry::from(model.target.clone());
        let cost = 10;
        let transition_chain = None;

        let mut node = CostNode::<_>::new(state.clone(), cost, &model, transition_chain);
        assert_eq!(node.priority, 10);
        assert_eq!(node.state(), &state);
        assert_eq!(node.state_mut(), &mut state);
        assert_eq!(node.cost(&model), 10);
        assert_eq!(node.bound(&model), None);
        assert!(!node.is_closed());
        assert_eq!(node.transitions(), vec![]);
        assert_eq!(node.last(), None);
        assert_eq!(node.transition_chain(), None);
    }

    #[test]
    fn test_with_state_and_priority_and_transition_chain() {
        let mut model = Model::default();
        model.set_maximize();
        let v1 = model.add_integer_resource_variable("v1", true, 0);
        assert!(v1.is_ok());
        let v2 = model.add_integer_resource_variable("v2", false, 0);
        assert!(v2.is_ok());

        let mut state = StateInRegistry::from(model.target.clone());
        let priority = 10;
        let transition_chain = None;

        let mut node = CostNode::<_>::with_state_and_priority_and_transition_chain(
            state.clone(),
            priority,
            transition_chain,
        );
        assert_eq!(node.priority, 10);
        assert_eq!(node.state(), &state);
        assert_eq!(node.state_mut(), &mut state);
        assert_eq!(node.cost(&model), 10);
        assert_eq!(node.bound(&model), None);
        assert!(!node.is_closed());
        assert_eq!(node.transitions(), vec![]);
        assert_eq!(node.last(), None);
        assert_eq!(node.transition_chain(), None);
    }

    #[test]
    fn test_close() {
        let model = Model::default();
        let state = StateInRegistry::from(model.target.clone());
        let cost = 10;
        let transition_chain = None;

        let node = CostNode::<_>::new(state.clone(), cost, &model, transition_chain);
        assert!(!node.is_closed());
        node.close();
        assert!(node.is_closed());
    }

    #[test]
    fn test_get_transitions() {
        let model = Model::default();
        let state = StateInRegistry::from(model.target.clone());
        let transition1 = Transition::new("t1");
        let transition2 = Transition::new("t2");
        let cost = 10;
        let transition_chain = Some(Rc::new(RcChain::new(None, Rc::new(transition1.clone()))));
        let transition_chain = Some(Rc::new(RcChain::new(
            transition_chain,
            Rc::new(transition2.clone()),
        )));
        let node = CostNode::<_>::new(state, cost, &model, transition_chain.clone());

        assert_eq!(node.last(), Some(&transition2));
        assert_eq!(node.transitions(), vec![transition1, transition2]);
        assert_eq!(node.transition_chain(), transition_chain);
    }

    #[test]
    fn test_ord() {
        let model = Model::default();

        let node1 = CostNode::<_>::with_state_and_priority_and_transition_chain(
            StateInRegistry::from(model.target.clone()),
            10,
            None,
        );
        let node2 = CostNode::<_>::with_state_and_priority_and_transition_chain(
            StateInRegistry::from(model.target.clone()),
            5,
            None,
        );

        assert!(node1 == node1);
        assert!(node1 <= node1);
        assert!(node1 >= node1);
        assert!(node1 != node2);
        assert!(node1 >= node1);
        assert!(node1 > node2);
        assert!(node2 <= node1);
        assert!(node2 < node1);
    }

    #[test]
    fn test_ordered_by_bound() {
        assert!(!CostNode::<Integer>::ordered_by_bound());
    }

    #[test]
    fn test_generate_root_node_min() {
        let mut model = dypdl::Model::default();
        model.set_minimize();
        let variable = model.add_integer_variable("variable", 0);
        assert!(variable.is_ok());
        let state = model.target.clone();
        let mut expected_state = StateInRegistry::from(state.clone());
        let mut node = CostNode::<_>::generate_root_node(state, 1, &model);
        assert_eq!(node.priority, -1);
        assert_eq!(node.state(), &expected_state);
        assert_eq!(node.state_mut(), &mut expected_state);
        assert_eq!(node.cost(&model), 1);
        assert_eq!(node.bound(&model), None);
        assert!(!node.is_closed());
        assert_eq!(node.transitions(), vec![]);
        assert_eq!(node.last(), None);
        assert_eq!(node.transition_chain(), None);
    }

    #[test]
    fn test_generate_root_node_max() {
        let mut model = dypdl::Model::default();
        model.set_maximize();
        let variable = model.add_integer_variable("variable", 0);
        assert!(variable.is_ok());
        let state = model.target.clone();
        let mut expected_state = StateInRegistry::from(state.clone());
        let mut node = CostNode::<_>::generate_root_node(state, 1, &model);
        assert_eq!(node.priority, 1);
        assert_eq!(node.state(), &expected_state);
        assert_eq!(node.state_mut(), &mut expected_state);
        assert_eq!(node.cost(&model), 1);
        assert_eq!(node.bound(&model), None);
        assert!(!node.is_closed());
        assert_eq!(node.transitions(), vec![]);
        assert_eq!(node.last(), None);
        assert_eq!(node.transition_chain(), None);
    }

    #[test]
    fn test_generate_successor_some_min() {
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
        let node = CostNode::<_>::generate_root_node(state, 0, &model);

        let successor = node.generate_successor_node(Rc::new(transition.clone()), &model);
        assert!(successor.is_some());
        let mut successor = successor.unwrap();
        assert_eq!(successor.priority, -1);
        assert_eq!(successor.state(), &expected_state);
        assert_eq!(successor.state_mut(), &mut expected_state);
        assert_eq!(successor.cost(&model), 1);
        assert_eq!(successor.bound(&model), None);
        assert!(!successor.is_closed());
        assert_eq!(successor.last(), Some(&transition));
        assert_eq!(successor.transitions(), vec![transition.clone()]);
        assert_eq!(
            successor.transition_chain(),
            Some(Rc::new(RcChain::new(None, Rc::new(transition))))
        );
    }

    #[test]
    fn test_generate_successor_some_max() {
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
        let mut expected_state: StateInRegistry = transition.apply(&state, &model.table_registry);
        let node = CostNode::<_>::generate_root_node(state, 0, &model);

        let successor = node.generate_successor_node(Rc::new(transition.clone()), &model);
        assert!(successor.is_some());
        let mut successor = successor.unwrap();
        assert_eq!(successor.priority, 1);
        assert_eq!(successor.state(), &expected_state);
        assert_eq!(successor.state_mut(), &mut expected_state);
        assert_eq!(successor.cost(&model), 1);
        assert_eq!(successor.bound(&model), None);
        assert!(!successor.is_closed());
        assert_eq!(successor.last(), Some(&transition));
        assert_eq!(successor.transitions(), vec![transition.clone()]);
        assert_eq!(
            successor.transition_chain(),
            Some(Rc::new(RcChain::new(None, Rc::new(transition))))
        );
    }

    #[test]
    fn test_generate_successor_pruned_by_constraint() {
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
        let node = CostNode::<_, _>::generate_root_node(state, 0, &model);

        let mut transition = Transition::default();
        let result = transition.add_effect(v1, v1 + 1);
        assert!(result.is_ok());
        let result = transition.add_effect(v2, v2 + 1);
        assert!(result.is_ok());
        transition.set_cost(IntegerExpression::Cost + 1);

        let result = node.generate_successor_node(Rc::new(transition), &model);
        assert_eq!(result, None);
    }

    #[test]
    fn test_insert_successor_non_dominance_min() {
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
        let mut registry = StateRegistry::<_, CostNode<_>>::new(model.clone());

        let mut transition = Transition::default();
        let result = transition.add_effect(v1, v1 + 1);
        assert!(result.is_ok());
        let result = transition.add_effect(v2, v2 + 1);
        assert!(result.is_ok());
        transition.set_cost(IntegerExpression::Cost + 1);

        let expected_state: StateInRegistry = transition.apply(&state, &model.table_registry);
        let node = CostNode::generate_root_node(state, 0, &model);
        let result = registry.insert(node.clone());
        assert!(result.is_some());

        let result = node.insert_successor_node(Rc::new(transition.clone()), &mut registry);
        assert!(result.is_some());
        let (successor, generated) = result.unwrap();
        assert_eq!(successor.priority, -1);
        assert_eq!(successor.state(), &expected_state);
        assert_eq!(successor.cost(&model), 1);
        assert_eq!(successor.bound(&model), None);
        assert!(!successor.is_closed());
        assert_eq!(successor.last(), Some(&transition));
        assert_eq!(successor.transitions(), vec![transition.clone()],);
        assert_eq!(
            successor.transition_chain(),
            Some(Rc::new(RcChain::new(None, Rc::new(transition))))
        );
        assert!(generated);
        assert!(!node.is_closed());
    }

    #[test]
    fn test_insert_successor_non_dominance_max() {
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
        let mut registry = StateRegistry::<_, CostNode<_>>::new(model.clone());

        let mut transition = Transition::default();
        let result = transition.add_effect(v1, v1 + 1);
        assert!(result.is_ok());
        let result = transition.add_effect(v2, v2 + 1);
        assert!(result.is_ok());
        transition.set_cost(IntegerExpression::Cost + 1);

        let expected_state: StateInRegistry = transition.apply(&state, &model.table_registry);
        let node = CostNode::generate_root_node(state, 0, &model);
        let result = registry.insert(node.clone());
        assert!(result.is_some());

        let result = node.insert_successor_node(Rc::new(transition.clone()), &mut registry);
        assert!(result.is_some());
        let (successor, generated) = result.unwrap();
        assert_eq!(successor.priority, 1);
        assert_eq!(successor.state(), &expected_state);
        assert_eq!(successor.cost(&model), 1);
        assert_eq!(successor.bound(&model), None);
        assert!(!successor.is_closed());
        assert_eq!(successor.last(), Some(&transition));
        assert_eq!(successor.transitions(), vec![transition.clone()]);
        assert_eq!(
            successor.transition_chain(),
            Some(Rc::new(RcChain::new(None, Rc::new(transition))))
        );
        assert!(generated);
        assert!(!node.is_closed());
    }

    #[test]
    fn test_insert_successor_pruned_by_constraint() {
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
        let mut registry = StateRegistry::<_, CostNode<_>>::new(Rc::new(model.clone()));

        let node = CostNode::generate_root_node(state, 0, &model);

        let mut transition = Transition::default();
        let result = transition.add_effect(v1, v1 + 1);
        assert!(result.is_ok());
        let result = transition.add_effect(v2, v2 + 1);
        assert!(result.is_ok());
        transition.set_cost(IntegerExpression::Cost + 1);

        let result = node.insert_successor_node(Rc::new(transition), &mut registry);
        assert_eq!(result, None);
        assert!(!node.is_closed());
    }

    #[test]
    fn test_insert_successor_dominating_min() {
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
        let node = CostNode::generate_root_node(state, 0, &model);
        let mut registry = StateRegistry::<_, CostNode<_>>::new(Rc::new(model.clone()));
        let result = registry.insert(node);
        assert!(result.is_some());
        let (node, dominated) = result.unwrap();
        assert!(dominated.is_none());

        let result = node.insert_successor_node(Rc::new(transition.clone()), &mut registry);
        assert!(result.is_some());
        let (successor, generated) = result.unwrap();
        assert_eq!(successor.priority, 0);
        assert_eq!(successor.state(), &expected_state);
        assert_eq!(successor.cost(&model), 0);
        assert_eq!(successor.bound(&model), None);
        assert!(!successor.is_closed());
        assert_eq!(successor.last(), Some(&transition));
        assert_eq!(successor.transitions(), vec![transition.clone()]);
        assert_eq!(
            successor.transition_chain(),
            Some(Rc::new(RcChain::new(None, Rc::new(transition))))
        );
        assert!(!generated);
        assert!(node.is_closed());
    }

    #[test]
    fn test_insert_successor_dominating_max() {
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
        let node = CostNode::generate_root_node(state, 0, &model);
        let mut registry = StateRegistry::<_, CostNode<_>>::new(Rc::new(model.clone()));
        let result = registry.insert(node);
        assert!(result.is_some());
        let (node, dominated) = result.unwrap();
        assert!(dominated.is_none());

        let result = node.insert_successor_node(Rc::new(transition.clone()), &mut registry);
        assert!(result.is_some());
        let (successor, generated) = result.unwrap();
        assert_eq!(successor.priority, 1);
        assert_eq!(successor.state(), &expected_state);
        assert_eq!(successor.cost(&model), 1);
        assert_eq!(successor.bound(&model), None);
        assert!(!successor.is_closed());
        assert_eq!(successor.last(), Some(&transition));
        assert_eq!(successor.transitions(), vec![transition.clone()]);
        assert_eq!(
            successor.transition_chain(),
            Some(Rc::new(RcChain::new(None, Rc::new(transition))))
        );
        assert!(!generated);
        assert!(node.is_closed());
    }

    #[test]
    fn test_insert_successor_dominated_min() {
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
        let node = CostNode::generate_root_node(state, 0, &model);
        let mut registry = StateRegistry::<_, CostNode<_>>::new(Rc::new(model.clone()));
        let result = registry.insert(node);
        assert!(result.is_some());
        let (node, dominated) = result.unwrap();
        assert!(dominated.is_none());

        let result = node.insert_successor_node(Rc::new(transition.clone()), &mut registry);
        assert!(result.is_none());
    }

    #[test]
    fn test_insert_successor_dominated_max() {
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
        let node = CostNode::generate_root_node(state, 0, &model);
        let mut registry = StateRegistry::<_, CostNode<_>>::new(Rc::new(model.clone()));
        let result = registry.insert(node);
        assert!(result.is_some());
        let (node, dominated) = result.unwrap();
        assert!(dominated.is_none());

        let result = node.insert_successor_node(Rc::new(transition.clone()), &mut registry);
        assert!(result.is_none());
    }
}
