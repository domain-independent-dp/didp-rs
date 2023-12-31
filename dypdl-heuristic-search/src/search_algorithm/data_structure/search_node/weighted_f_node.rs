use super::super::state_registry::{StateInRegistry, StateInformation, StateRegistry};
use super::super::transition_chain::{CreateTransitionChain, GetTransitions, RcChain};
use super::f_node::FNode;
use super::{BfsNode, CostNode};
use dypdl::variable_type::Numeric;
use dypdl::{Model, ReduceFunction, Transition, TransitionInterface};
use std::cmp::Ordering;
use std::fmt::Display;
use std::ops::Deref;
use std::rc::Rc;

/// Node ordered by the f-value, and tie is broken by the h-value.
/// A node having a higher f- and h-values is `Greater` in `Ord`.
///
/// In minimization, the negative of the h-value should be a lower bound on the cost of a solution from the state (dual bound).
/// In maximization, the h-value should be an upper bound on the solution of a solution from the state (dual bound).
#[derive(Debug, Clone)]
pub struct WeightedFNode<T, U, V = Transition, R = Rc<V>, C = RcChain<V>, P = Rc<C>> {
    node: FNode<T, V, R, C, P>,
    /// f-value.
    pub f: U,
}

impl<T, U, V, R, C, P> PartialEq for WeightedFNode<T, U, V, R, C, P>
where
    T: PartialOrd,
    U: PartialOrd,
{
    /// Nodes are compared by their f- and h-values.
    /// This does not mean that the nodes are the same.
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.f == other.f && self.node.h == other.node.h
    }
}

impl<T, U, V, R, C, P> Eq for WeightedFNode<T, U, V, R, C, P>
where
    T: Ord,
    U: Ord,
{
}

impl<T, U, V, R, C, P> Ord for WeightedFNode<T, U, V, R, C, P>
where
    T: Ord,
    U: Ord,
{
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        match self.f.cmp(&other.f) {
            Ordering::Equal => self.node.h.cmp(&other.node.h),
            result => result,
        }
    }
}

impl<T, U, V, R, C, P> PartialOrd for WeightedFNode<T, U, V, R, C, P>
where
    T: Ord,
    U: Ord,
{
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T, U, V, R, C, P> StateInformation<T> for WeightedFNode<T, U, V, R, C, P>
where
    T: Numeric,
{
    #[inline]
    fn state(&self) -> &StateInRegistry {
        self.node.state()
    }

    #[inline]
    fn state_mut(&mut self) -> &mut StateInRegistry {
        self.node.state_mut()
    }

    #[inline]
    fn cost(&self, model: &Model) -> T {
        self.node.cost(model)
    }

    #[inline]
    fn bound(&self, model: &Model) -> Option<T> {
        self.node.bound(model)
    }

    #[inline]
    fn is_closed(&self) -> bool {
        self.node.is_closed()
    }

    #[inline]
    fn close(&self) {
        self.node.close()
    }
}

impl<T, U, V, R, C, P> GetTransitions<V> for WeightedFNode<T, U, V, R, C, P>
where
    V: TransitionInterface + Clone,
    Transition: From<V>,
    C: GetTransitions<V>,
    P: Deref<Target = C>,
{
    #[inline]
    fn transitions(&self) -> Vec<V> {
        self.node.transitions()
    }

    #[inline]
    fn last(&self) -> Option<&V> {
        self.node.last()
    }
}

impl<T, U, V, R, C, P> BfsNode<T, V> for WeightedFNode<T, U, V, R, C, P>
where
    T: Numeric + Ord + Display,
    U: Numeric + Ord,
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

impl<T, U, V, R, C, P> WeightedFNode<T, U, V, R, C, P>
where
    T: Numeric + Ord,
    U: Numeric + Ord,
    V: TransitionInterface + Clone,
    Transition: From<V>,
    R: Deref<Target = V>,
    C: CreateTransitionChain<R, P>,
    P: From<C> + Clone,
{
    /// Creates a new node given a DyPDL node, a DyPDL model, and an f-evaluator.
    pub fn new<F>(node: FNode<T, V, R, C, P>, model: &Model, f_evaluator: F) -> Self
    where
        F: FnOnce(T, T, &StateInRegistry) -> U,
    {
        let f = if model.reduce_function == ReduceFunction::Max {
            f_evaluator(node.cost(model), node.h, node.state())
        } else {
            -f_evaluator(node.cost(model), -node.h, node.state())
        };
        Self { node, f }
    }

    /// Get the transition chain.
    pub fn transition_chain(&self) -> Option<P> {
        self.node.transition_chain()
    }

    /// Generates a root search node given a state, its cost, a DyPDL model, h-, bound, and f-evaluators,
    /// and a primal bound on the solution cost.
    ///
    /// `h_evaluator` is a function that takes a state and returns the dual bound of the cost from the state.
    /// If `h_evaluator` returns `None`, the state is a dead-end, so the node is not generated.
    /// `bound_evaluator` is a function that takes `cost`, the return value by `h_evaluator`, and the state and returns the dual bound of the cost of a solution extending the path to this node.
    /// `f_evaluator` is a function that takes `cost`, the return value by `h_evaluator`, and the state and returns the priority.
    ///
    /// Returns `None` if the state is a dead-end, or the dual bound exceeds the primal bound.
    ///
    /// If the model is minimizing, the h- and f-values become the negatives of the values returned by the evaluators.
    /// If the model is maximizing, the h- and f-values become the values returned by the evaluators.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    /// use dypdl_heuristic_search::search_algorithm::{WeightedFNode, StateInRegistry};
    /// use dypdl_heuristic_search::search_algorithm::data_structure::{
    ///     GetTransitions, StateInformation,
    /// };
    ///
    /// let mut model = Model::default();
    /// model.add_integer_variable("variable", 0).unwrap();
    ///
    /// let state = model.target.clone();
    /// let cost = 0;
    /// let h_evaluator = |_: &StateInRegistry| Some(1);
    /// let bound_evaluator = |g, h, _: &StateInRegistry| g + h;
    /// let f_evaluator = |g, h, _: &StateInRegistry| g + 2 * h;
    /// let node = WeightedFNode::<_, _>::generate_root_node(
    ///     state, cost, &model, &h_evaluator, &bound_evaluator, &f_evaluator, None,
    /// );
    /// assert!(node.is_some());
    /// let node = node.unwrap();
    /// assert_eq!(node.state(), &StateInRegistry::from(model.target.clone()));
    /// assert_eq!(node.cost(&model), cost);
    /// assert_eq!(node.bound(&model), Some(1));
    /// assert!(!node.is_closed());
    /// assert_eq!(node.transitions(), vec![]);
    /// ```
    pub fn generate_root_node<S, H, B, F>(
        state: S,
        cost: T,
        model: &Model,
        h_evaluator: H,
        bound_evaluator: B,
        f_evaluator: F,
        primal_bound: Option<T>,
    ) -> Option<Self>
    where
        StateInRegistry: From<S>,
        H: FnOnce(&StateInRegistry) -> Option<T>,
        B: FnOnce(T, T, &StateInRegistry) -> T,
        F: FnOnce(T, T, &StateInRegistry) -> U,
    {
        let node = FNode::generate_root_node(
            state,
            cost,
            model,
            h_evaluator,
            bound_evaluator,
            primal_bound,
        )?;

        Some(WeightedFNode::new(node, model, f_evaluator))
    }

    /// Generates a successor node given a transition, a DyPDL model, h-, bound, and f-evaluators,
    /// and a primal bound on the solution cost.
    ///
    /// `h_evaluator` is a function that takes a state and returns the dual bound of the cost from the state.
    /// If `h_evaluator` returns `None`, the state is a dead-end, so the node is not generated.
    /// `bound_evaluator` is a function that takes the cost, the return value by `h_evaluator`, and the state and returns the dual bound of the cost of a solution extending the path to this node.
    /// `f_evaluator` is a function that takes the cost, the return value by `h_evaluator`, and the state and returns the priority.
    ///
    /// Returns `None` if the state is a dead-end, or the dual bound exceeds the primal bound.
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
    /// use dypdl_heuristic_search::search_algorithm::{WeightedFNode, StateInRegistry};
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
    /// let h_evaluator = |_: &StateInRegistry| Some(1);
    /// let bound_evaluator = |g, h, _: &StateInRegistry| g + h;
    /// let f_evaluator = |g, h, _: &StateInRegistry| g + 2 * h;
    /// let node = WeightedFNode::<_, _>::generate_root_node(
    ///     state, cost, &model, &h_evaluator, &bound_evaluator, &f_evaluator, None,
    /// ).unwrap();
    ///
    /// let mut transition = Transition::new("transition");
    /// transition.set_cost(IntegerExpression::Cost + 1);
    /// transition.add_effect(variable, variable + 1).unwrap();
    /// let expected_state: StateInRegistry = transition.apply(
    ///     &model.target, &model.table_registry,
    /// );
    ///
    /// let node = node.generate_successor_node(
    ///     Rc::new(transition.clone()), &model, &h_evaluator, &bound_evaluator, &f_evaluator,
    ///     None,
    /// );
    /// assert!(node.is_some());
    /// let node = node.unwrap();
    /// assert_eq!(node.state(), &expected_state);
    /// assert_eq!(node.cost(&model), 1);
    /// assert_eq!(node.bound(&model), Some(2));
    /// assert!(!node.is_closed());
    /// assert_eq!(node.transitions(), vec![transition]);
    /// ```
    pub fn generate_successor_node<H, B, F>(
        &self,
        transition: R,
        model: &Model,
        h_evaluator: H,
        bound_evaluator: B,
        f_evaluator: F,
        primal_bound: Option<T>,
    ) -> Option<Self>
    where
        H: FnOnce(&StateInRegistry) -> Option<T>,
        B: FnOnce(T, T, &StateInRegistry) -> T,
        F: FnOnce(T, T, &StateInRegistry) -> U,
    {
        let node = self.node.generate_successor_node(
            transition,
            model,
            h_evaluator,
            bound_evaluator,
            primal_bound,
        )?;

        Some(WeightedFNode::new(node, model, f_evaluator))
    }

    /// Generates a successor node given a transition, h-, bound, and f- evaluators,
    /// and a primal bound on the solution cost, and inserts it into a state registry.
    ///
    /// `h_evaluator` is a function that takes a state and returns the dual bound of the cost from the state.
    /// If `h_evaluator` returns `None`, the state is a dead-end, so the node is not generated.
    /// `bound_evaluator` is a function that takes the cost, the return value by `h_evaluator`, and the state and returns the dual bound of the cost of a solution extending the path to this node.
    /// `f_evaluator` is a function that takes the cost, the return value by `h_evaluator`, and the state and returns the priority.
    ///
    /// Returns `None` if the successor state is pruned by a state constraint or a dead-end,
    /// the dual bound exceeds the primal bound, or the successor node is dominated.
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
    /// use dypdl_heuristic_search::search_algorithm::{
    ///     WeightedFNode, StateInRegistry, StateRegistry,
    /// };
    /// use dypdl_heuristic_search::search_algorithm::data_structure::{
    ///     GetTransitions, StateInformation,
    /// };
    /// use std::rc::Rc;
    ///
    /// let mut model = Model::default();
    /// let variable = model.add_integer_variable("variable", 0).unwrap();
    /// let mut registry = StateRegistry::<_, WeightedFNode<_, _>>::new(
    ///     Rc::new(model.clone()),
    /// );
    ///
    /// let state = model.target.clone();
    /// let cost = 0;
    /// let h_evaluator = |_: &StateInRegistry| Some(1);
    /// let bound_evaluator = |g, h, _: &StateInRegistry| g + h;
    /// let f_evaluator = |g, h, _: &StateInRegistry| g + 2 * h;
    /// let node = WeightedFNode::<_, _>::generate_root_node(
    ///     state, cost, &model, &h_evaluator, &bound_evaluator, &f_evaluator, None,
    /// ).unwrap();
    ///
    /// let mut transition = Transition::new("transition");
    /// transition.set_cost(IntegerExpression::Cost + 1);
    /// transition.add_effect(variable, variable + 1).unwrap();
    /// let expected_state: StateInRegistry = transition.apply(
    ///     &model.target, &model.table_registry,
    /// );
    ///
    /// let result = node.insert_successor_node(
    ///     Rc::new(transition.clone()), &mut registry, &h_evaluator, &bound_evaluator,
    ///     &f_evaluator, None,
    /// );
    /// assert!(result.is_some());
    /// let (node, generated) = result.unwrap();
    /// assert!(generated);
    /// assert_eq!(node.state(), &expected_state);
    /// assert_eq!(node.cost(&model), 1);
    /// assert_eq!(node.bound(&model), Some(2));
    /// assert!(!node.is_closed());
    /// assert_eq!(node.transitions(), vec![transition]);
    /// ```
    pub fn insert_successor_node<H, B, F, M>(
        &self,
        transition: R,
        registry: &mut StateRegistry<T, Self, M>,
        h_evaluator: H,
        bound_evaluator: B,
        f_evaluator: F,
        primal_bound: Option<T>,
    ) -> Option<(Rc<Self>, bool)>
    where
        H: FnOnce(&StateInRegistry) -> Option<T>,
        B: FnOnce(T, T, &StateInRegistry) -> T,
        F: FnOnce(T, T, &StateInRegistry) -> U,
        M: Deref<Target = Model> + Clone,
    {
        let (state, g) = registry.model().generate_successor_state(
            self.state(),
            self.cost(registry.model()),
            transition.deref(),
            None,
        )?;

        let model = registry.model().clone();

        let constructor = |state, g, other: Option<&Self>| {
            let (h, f) = FNode::evaluate_state(
                &state,
                g,
                &model,
                h_evaluator,
                bound_evaluator,
                primal_bound,
                other.map(|node| &node.node),
            )?;
            let transition_chain = P::from(C::new(self.transition_chain(), transition));
            let node = CostNode::new(state, g, &model, Some(transition_chain));
            let node = FNode::with_node_and_h_and_f(node, h, f);
            Some(Self::new(node, &model, f_evaluator))
        };

        let (successor, dominated) = registry.insert_with(state, g, constructor)?;

        let mut generated = true;

        if let Some(dominated) = dominated {
            if !dominated.is_closed() {
                dominated.close();
                generated = false;
            }
        }

        Some((successor, generated))
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

        let node = CostNode::<_>::new(state.clone(), cost, &model, transition_chain);
        let node = FNode::<_>::with_node_and_h_and_f(node, -5, -15);
        let mut node = WeightedFNode::new(node, &model, |g, h, _| g + 2 * h);

        assert_eq!(node.f, -20);
        assert_eq!(node.state(), &state);
        assert_eq!(node.state_mut(), &mut state);
        assert_eq!(node.cost(&model), 10);
        assert_eq!(node.bound(&model), Some(15));
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

        let node = CostNode::<_>::new(state.clone(), cost, &model, transition_chain);
        let node = FNode::<_>::with_node_and_h_and_f(node, 5, 15);
        let mut node = WeightedFNode::new(node, &model, |g, h, _| g + 2 * h);

        assert_eq!(node.f, 20);
        assert_eq!(node.state(), &state);
        assert_eq!(node.state_mut(), &mut state);
        assert_eq!(node.cost(&model), 10);
        assert_eq!(node.bound(&model), Some(15));
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
        let node = FNode::<_>::with_node_and_h_and_f(node, 5, 15);
        let node = WeightedFNode::new(node, &model, |g, h, _| g + 2 * h);
        assert!(!node.is_closed());
        node.close();
        assert!(node.is_closed());
    }

    #[test]
    fn test_get_transition() {
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
        let node = FNode::<_>::with_node_and_h_and_f(node, 5, 15);
        let node = WeightedFNode::new(node, &model, |g, h, _| g + 2 * h);

        assert_eq!(node.last(), Some(&transition2));
        assert_eq!(node.transitions(), vec![transition1, transition2]);
        assert_eq!(node.transition_chain(), transition_chain);
    }

    #[test]
    fn test_ord() {
        let model = Model::default();
        let f_evaluator = |g, h, _: &StateInRegistry| g + 2 * h;

        let node1 = CostNode::<_>::new(
            StateInRegistry::from(model.target.clone()),
            10,
            &model,
            None,
        );
        let node1 = FNode::<_>::with_node_and_h_and_f(node1, -5, -15);
        let node1 = WeightedFNode::new(node1, &model, &f_evaluator);
        let node2 =
            CostNode::<_>::new(StateInRegistry::from(model.target.clone()), 6, &model, None);
        let node2 = FNode::<_>::with_node_and_h_and_f(node2, -8, -14);
        let node2 = WeightedFNode::new(node2, &model, &f_evaluator);
        let node3 =
            CostNode::<_>::new(StateInRegistry::from(model.target.clone()), 8, &model, None);
        let node3 = FNode::<_>::with_node_and_h_and_f(node3, -6, -14);
        let node3 = WeightedFNode::new(node3, &model, &f_evaluator);

        assert!(node1 == node1);
        assert!(node1 <= node1);
        assert!(node1 >= node1);
        assert!(node1 != node2);
        assert!(node1 >= node2);
        assert!(node1 > node2);
        assert!(node2 <= node1);
        assert!(node2 < node1);
        assert!(node1 != node3);
        assert!(node1 >= node3);
        assert!(node1 > node3);
        assert!(node3 <= node1);
        assert!(node3 < node1);
    }

    #[test]
    fn test_ordered_by_bound() {
        assert!(!WeightedFNode::<Integer, Integer>::ordered_by_bound());
    }

    #[test]
    fn test_generate_root_node_some_min() {
        let mut model = dypdl::Model::default();
        model.set_minimize();
        let variable = model.add_integer_variable("variable", 0);
        assert!(variable.is_ok());
        let state = model.target.clone();
        let mut expected_state = StateInRegistry::from(state.clone());
        let h_evaluator = |_: &StateInRegistry| Some(1);
        let bound_evaluator = |g, h, _: &StateInRegistry| g + h;
        let f_evaluator = |g, h, _: &StateInRegistry| g + 2 * h;
        let node = WeightedFNode::<_, _>::generate_root_node(
            state,
            1,
            &model,
            &h_evaluator,
            &bound_evaluator,
            &f_evaluator,
            None,
        );
        assert!(node.is_some());
        let mut node = node.unwrap();
        assert_eq!(node.f, -3);
        assert_eq!(node.state(), &expected_state);
        assert_eq!(node.state_mut(), &mut expected_state);
        assert_eq!(node.cost(&model), 1);
        assert_eq!(node.bound(&model), Some(2));
        assert!(!node.is_closed());
        assert_eq!(node.last(), None);
        assert_eq!(node.transitions(), vec![]);
        assert_eq!(node.transition_chain(), None);
    }

    #[test]
    fn test_generate_root_node_some_max() {
        let mut model = dypdl::Model::default();
        model.set_maximize();
        let variable = model.add_integer_variable("variable", 0);
        assert!(variable.is_ok());
        let state = model.target.clone();
        let mut expected_state = StateInRegistry::from(state.clone());
        let h_evaluator = |_: &StateInRegistry| Some(1);
        let bound_evaluator = |g, h, _: &StateInRegistry| g + h;
        let f_evaluator = |g, h, _: &StateInRegistry| g + 2 * h;
        let node = WeightedFNode::<_, _>::generate_root_node(
            state,
            1,
            &model,
            &h_evaluator,
            &bound_evaluator,
            &f_evaluator,
            None,
        );
        assert!(node.is_some());
        let mut node = node.unwrap();
        assert_eq!(node.f, 3);
        assert_eq!(node.state(), &expected_state);
        assert_eq!(node.state_mut(), &mut expected_state);
        assert_eq!(node.cost(&model), 1);
        assert_eq!(node.bound(&model), Some(2));
        assert!(!node.is_closed());
        assert_eq!(node.last(), None);
        assert_eq!(node.transitions(), vec![]);
        assert_eq!(node.transition_chain(), None);
    }

    #[test]
    fn test_generate_root_node_pruned_by_bound_min() {
        let mut model = dypdl::Model::default();
        model.set_minimize();
        let variable = model.add_integer_variable("variable", 0);
        assert!(variable.is_ok());
        let state = model.target.clone();
        let h_evaluator = |_: &StateInRegistry| Some(1);
        let bound_evaluator = |g, h, _: &StateInRegistry| g + h;
        let f_evaluator = |g, h, _: &StateInRegistry| g + 2 * h;
        let node = WeightedFNode::<_, _>::generate_root_node(
            state,
            0,
            &model,
            &h_evaluator,
            &bound_evaluator,
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
        let h_evaluator = |_: &StateInRegistry| Some(1);
        let bound_evaluator = |g, h, _: &StateInRegistry| g + h;
        let f_evaluator = |g, h, _: &StateInRegistry| g + 2 * h;
        let node = WeightedFNode::<_, _>::generate_root_node(
            state,
            0,
            &model,
            &h_evaluator,
            &bound_evaluator,
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
        let h_evaluator = |_: &StateInRegistry| None;
        let bound_evaluator = |g, h, _: &StateInRegistry| g + h;
        let f_evaluator = |g, h, _: &StateInRegistry| g + 2 * h;
        let node = WeightedFNode::<_, _>::generate_root_node(
            state,
            0,
            &model,
            &h_evaluator,
            &bound_evaluator,
            &f_evaluator,
            None,
        );
        assert!(node.is_none());
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
        let h_evaluator = |_: &StateInRegistry| Some(1);
        let bound_evaluator = |g, h, _: &StateInRegistry| g + h;
        let f_evaluator = |g, h, _: &StateInRegistry| g + 2 * h;
        let node = WeightedFNode::<_, _>::generate_root_node(
            state,
            0,
            &model,
            &h_evaluator,
            &bound_evaluator,
            &f_evaluator,
            None,
        );
        assert!(node.is_some());
        let node = node.unwrap();

        let successor = node.generate_successor_node(
            Rc::new(transition.clone()),
            &model,
            &h_evaluator,
            &bound_evaluator,
            &f_evaluator,
            None,
        );
        assert!(successor.is_some());
        let mut successor = successor.unwrap();
        assert_eq!(successor.f, -3);
        assert_eq!(successor.state(), &expected_state);
        assert_eq!(successor.state_mut(), &mut expected_state);
        assert_eq!(successor.cost(&model), 1);
        assert_eq!(successor.bound(&model), Some(2));
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
        let h_evaluator = |_: &StateInRegistry| Some(1);
        let bound_evaluator = |g, h, _: &StateInRegistry| g + h;
        let f_evaluator = |g, h, _: &StateInRegistry| g + 2 * h;
        let node = WeightedFNode::<_, _>::generate_root_node(
            state,
            0,
            &model,
            &h_evaluator,
            &bound_evaluator,
            &f_evaluator,
            None,
        );
        assert!(node.is_some());
        let node = node.unwrap();

        let successor = node.generate_successor_node(
            Rc::new(transition.clone()),
            &model,
            &h_evaluator,
            &bound_evaluator,
            &f_evaluator,
            None,
        );
        assert!(successor.is_some());
        let mut successor = successor.unwrap();
        assert_eq!(successor.f, 3);
        assert_eq!(successor.state(), &expected_state);
        assert_eq!(successor.state_mut(), &mut expected_state);
        assert_eq!(successor.cost(&model), 1);
        assert_eq!(successor.bound(&model), Some(2));
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
        let h_evaluator = |_: &StateInRegistry| Some(0);
        let bound_evaluator = |g, h, _: &StateInRegistry| g + h;
        let f_evaluator = |g, h, _: &StateInRegistry| g + 2 * h;
        let node = WeightedFNode::<_, _>::generate_root_node(
            state,
            0,
            &model,
            &h_evaluator,
            &bound_evaluator,
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
            Rc::new(transition),
            &model,
            &h_evaluator,
            &bound_evaluator,
            &f_evaluator,
            None,
        );
        assert_eq!(result, None);
    }

    #[test]
    fn test_generate_successor_pruned_by_bound_min() {
        let mut model = Model::default();
        model.set_minimize();
        let v1 = model.add_integer_resource_variable("v1", true, 0);
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let v2 = model.add_integer_resource_variable("v2", false, 0);
        assert!(v2.is_ok());
        let v2 = v2.unwrap();

        let state = model.target.clone();
        let h_evaluator = |_: &StateInRegistry| Some(0);
        let bound_evaluator = |g, h, _: &StateInRegistry| g + h;
        let f_evaluator = |g, h, _: &StateInRegistry| g + 2 * h;
        let node = WeightedFNode::<_, _>::generate_root_node(
            state,
            0,
            &model,
            &h_evaluator,
            &bound_evaluator,
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
            Rc::new(transition),
            &model,
            &h_evaluator,
            &bound_evaluator,
            &f_evaluator,
            Some(0),
        );
        assert_eq!(result, None);
    }

    #[test]
    fn test_generate_successor_pruned_by_bound_max() {
        let mut model = Model::default();
        model.set_maximize();
        let v1 = model.add_integer_resource_variable("v1", true, 0);
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let v2 = model.add_integer_resource_variable("v2", false, 0);
        assert!(v2.is_ok());
        let v2 = v2.unwrap();

        let state = model.target.clone();
        let h_evaluator = |_: &StateInRegistry| Some(0);
        let bound_evaluator = |g, h, _: &StateInRegistry| g + h;
        let f_evaluator = |g, h, _: &StateInRegistry| g + 2 * h;
        let node = WeightedFNode::<_, _>::generate_root_node(
            state,
            0,
            &model,
            &h_evaluator,
            &bound_evaluator,
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
            Rc::new(transition),
            &model,
            &h_evaluator,
            &bound_evaluator,
            &f_evaluator,
            Some(2),
        );
        assert_eq!(result, None);
    }

    #[test]
    fn test_generate_successor_pruned_by_h() {
        let mut model = Model::default();
        let v1 = model.add_integer_resource_variable("v1", true, 0);
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let v2 = model.add_integer_resource_variable("v2", false, 0);
        assert!(v2.is_ok());
        let v2 = v2.unwrap();

        let state = model.target.clone();
        let h_evaluator = |_: &StateInRegistry| Some(0);
        let bound_evaluator = |g, h, _: &StateInRegistry| g + h;
        let f_evaluator = |g, h, _: &StateInRegistry| g + 2 * h;
        let node = WeightedFNode::<_, _>::generate_root_node(
            state,
            0,
            &model,
            &h_evaluator,
            &bound_evaluator,
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

        let h_evaluator = |_: &StateInRegistry| None;
        let result = node.generate_successor_node(
            Rc::new(transition),
            &model,
            &h_evaluator,
            &bound_evaluator,
            &f_evaluator,
            None,
        );
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
        let mut registry = StateRegistry::<_, WeightedFNode<_, _>>::new(model.clone());

        let mut transition = Transition::default();
        let result = transition.add_effect(v1, v1 + 1);
        assert!(result.is_ok());
        let result = transition.add_effect(v2, v2 + 1);
        assert!(result.is_ok());
        transition.set_cost(IntegerExpression::Cost + 1);

        let expected_state: StateInRegistry = transition.apply(&state, &model.table_registry);
        let h_evaluator = |_: &StateInRegistry| Some(1);
        let bound_evaluator = |g, h, _: &StateInRegistry| g + h;
        let f_evaluator = |g, h, _: &StateInRegistry| g + 2 * h;
        let node = WeightedFNode::generate_root_node(
            state,
            0,
            &model,
            &h_evaluator,
            &bound_evaluator,
            &f_evaluator,
            None,
        );
        assert!(node.is_some());
        let node = node.unwrap();
        let result = registry.insert(node.clone());
        assert!(result.is_some());

        let result = node.insert_successor_node(
            Rc::new(transition.clone()),
            &mut registry,
            &h_evaluator,
            &bound_evaluator,
            &f_evaluator,
            None,
        );
        assert!(result.is_some());
        let (successor, generated) = result.unwrap();
        assert_eq!(successor.f, -3);
        assert_eq!(successor.state(), &expected_state);
        assert_eq!(successor.cost(&model), 1);
        assert_eq!(successor.bound(&model), Some(2));
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
        let mut registry = StateRegistry::<_, WeightedFNode<_, _>>::new(model.clone());

        let mut transition = Transition::default();
        let result = transition.add_effect(v1, v1 + 1);
        assert!(result.is_ok());
        let result = transition.add_effect(v2, v2 + 1);
        assert!(result.is_ok());
        transition.set_cost(IntegerExpression::Cost + 1);

        let expected_state: StateInRegistry = transition.apply(&state, &model.table_registry);
        let h_evaluator = |_: &StateInRegistry| Some(1);
        let bound_evaluator = |g, h, _: &StateInRegistry| g + h;
        let f_evaluator = |g, h, _: &StateInRegistry| g + 2 * h;
        let node = WeightedFNode::generate_root_node(
            state,
            0,
            &model,
            &h_evaluator,
            &bound_evaluator,
            &f_evaluator,
            None,
        );
        assert!(node.is_some());
        let node = node.unwrap();
        let result = registry.insert(node.clone());
        assert!(result.is_some());

        let result = node.insert_successor_node(
            Rc::new(transition.clone()),
            &mut registry,
            &h_evaluator,
            &bound_evaluator,
            &f_evaluator,
            None,
        );
        assert!(result.is_some());
        let (successor, generated) = result.unwrap();
        assert_eq!(successor.state(), &expected_state);
        assert_eq!(successor.f, 3);
        assert_eq!(successor.cost(&model), 1);
        assert_eq!(successor.bound(&model), Some(2));
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
        let mut registry = StateRegistry::<_, WeightedFNode<_, _>>::new(Rc::new(model.clone()));

        let h_evaluator = |_: &StateInRegistry| Some(0);
        let bound_evaluator = |g, h, _: &StateInRegistry| g + h;
        let f_evaluator = |g, h, _: &StateInRegistry| g + 2 * h;
        let node = WeightedFNode::generate_root_node(
            state,
            0,
            &model,
            &h_evaluator,
            &bound_evaluator,
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

        let result = node.insert_successor_node(
            Rc::new(transition),
            &mut registry,
            &h_evaluator,
            &bound_evaluator,
            &f_evaluator,
            None,
        );
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
        let h_evaluator = |_: &StateInRegistry| Some(1);
        let bound_evaluator = |g, h, _: &StateInRegistry| g + h;
        let f_evaluator = |g, h, _: &StateInRegistry| g + 2 * h;
        let node = WeightedFNode::generate_root_node(
            state,
            0,
            &model,
            &h_evaluator,
            &bound_evaluator,
            &f_evaluator,
            None,
        );
        assert!(node.is_some());
        let node = node.unwrap();
        let mut registry = StateRegistry::<_, WeightedFNode<_, _>>::new(Rc::new(model.clone()));
        let result = registry.insert(node);
        assert!(result.is_some());
        let (node, dominated) = result.unwrap();
        assert!(dominated.is_none());

        let result = node.insert_successor_node(
            Rc::new(transition.clone()),
            &mut registry,
            &h_evaluator,
            &bound_evaluator,
            &f_evaluator,
            None,
        );
        assert!(result.is_some());
        let (successor, generated) = result.unwrap();
        assert_eq!(successor.f, -2);
        assert_eq!(successor.state(), &expected_state);
        assert_eq!(successor.cost(&model), 0);
        assert_eq!(successor.bound(&model), Some(1));
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
        let h_evaluator = |_: &StateInRegistry| Some(0);
        let bound_evaluator = |g, h, _: &StateInRegistry| g + h;
        let f_evaluator = |g, h, _: &StateInRegistry| g + 2 * h;
        let node = WeightedFNode::generate_root_node(
            state,
            0,
            &model,
            &h_evaluator,
            &bound_evaluator,
            &f_evaluator,
            None,
        );
        assert!(node.is_some());
        let node = node.unwrap();
        let mut registry = StateRegistry::<_, WeightedFNode<_, _>>::new(Rc::new(model.clone()));
        let result = registry.insert(node);
        assert!(result.is_some());
        let (node, dominated) = result.unwrap();
        assert!(dominated.is_none());

        let result = node.insert_successor_node(
            Rc::new(transition.clone()),
            &mut registry,
            &h_evaluator,
            &bound_evaluator,
            &f_evaluator,
            None,
        );
        assert!(result.is_some());
        let (successor, generated) = result.unwrap();
        assert_eq!(successor.f, 1);
        assert_eq!(successor.state(), &expected_state);
        assert_eq!(successor.cost(&model), 1);
        assert_eq!(successor.bound(&model), Some(1));
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
    fn test_insert_successor_pruned_by_bound_min() {
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
        assert!(result.is_ok());
        transition.set_cost(IntegerExpression::Cost + 1);

        let state = StateInRegistry::from(model.target.clone());
        let h_evaluator = |_: &StateInRegistry| Some(0);
        let bound_evaluator = |g, h, _: &StateInRegistry| g + h;
        let f_evaluator = |g, h, _: &StateInRegistry| g + 2 * h;
        let node = WeightedFNode::generate_root_node(
            state,
            0,
            &model,
            &h_evaluator,
            &bound_evaluator,
            &f_evaluator,
            None,
        );
        assert!(node.is_some());
        let node = node.unwrap();
        let mut registry = StateRegistry::<_, WeightedFNode<_, _>>::new(Rc::new(model.clone()));
        let result = registry.insert(node);
        assert!(result.is_some());
        let (node, dominated) = result.unwrap();
        assert!(dominated.is_none());

        let result = node.insert_successor_node(
            Rc::new(transition.clone()),
            &mut registry,
            &h_evaluator,
            &bound_evaluator,
            &f_evaluator,
            Some(0),
        );
        assert!(result.is_none());
    }

    #[test]
    fn test_insert_successor_pruned_by_bound_max() {
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
        assert!(result.is_ok());
        transition.set_cost(IntegerExpression::Cost + 1);

        let state = StateInRegistry::from(model.target.clone());
        let h_evaluator = |_: &StateInRegistry| Some(0);
        let bound_evaluator = |g, h, _: &StateInRegistry| g + h;
        let f_evaluator = |g, h, _: &StateInRegistry| g + 2 * h;
        let node = WeightedFNode::generate_root_node(
            state,
            0,
            &model,
            &h_evaluator,
            &bound_evaluator,
            &f_evaluator,
            None,
        );
        assert!(node.is_some());
        let node = node.unwrap();
        let mut registry = StateRegistry::<_, WeightedFNode<_, _>>::new(Rc::new(model.clone()));
        let result = registry.insert(node);
        assert!(result.is_some());
        let (node, dominated) = result.unwrap();
        assert!(dominated.is_none());

        let result = node.insert_successor_node(
            Rc::new(transition.clone()),
            &mut registry,
            &h_evaluator,
            &bound_evaluator,
            &f_evaluator,
            Some(2),
        );
        assert!(result.is_none());
    }

    #[test]
    fn test_insert_successor_pruned_by_h() {
        let mut model = Model::default();
        let v1 = model.add_integer_resource_variable("v1", true, 0);
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let v2 = model.add_integer_resource_variable("v2", false, 0);
        assert!(v2.is_ok());
        let v2 = v2.unwrap();

        let state = StateInRegistry::from(model.target.clone());
        let mut registry = StateRegistry::<_, WeightedFNode<_, _>>::new(Rc::new(model.clone()));

        let h_evaluator = |_: &StateInRegistry| Some(0);
        let bound_evaluator = |g, h, _: &StateInRegistry| g + h;
        let f_evaluator = |g, h, _: &StateInRegistry| g + 2 * h;
        let node = WeightedFNode::generate_root_node(
            state,
            0,
            &model,
            &h_evaluator,
            &bound_evaluator,
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

        let h_evaluator = |_: &StateInRegistry| None;
        let result = node.insert_successor_node(
            Rc::new(transition),
            &mut registry,
            &h_evaluator,
            &bound_evaluator,
            &f_evaluator,
            None,
        );
        assert_eq!(result, None);
        assert!(!node.is_closed());
    }
}
