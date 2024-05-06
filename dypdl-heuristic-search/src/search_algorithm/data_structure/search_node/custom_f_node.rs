use super::super::state_registry::{StateInRegistry, StateInformation, StateRegistry};
use super::super::transition::TransitionWithCustomCost;
use super::super::transition_chain::{CreateTransitionChain, GetTransitions};
use super::super::RcChain;
use super::{BfsNode, CostNode};
use dypdl::variable_type::Numeric;
use dypdl::Model;
use std::cmp::Ordering;
use std::fmt::{Debug, Display};
use std::ops::Deref;
use std::rc::Rc;

/// Node ordered by the f-value, and tie is broken by the h-value.
/// A node having a higher f- and h-values is `Greater` in `Ord`.
#[derive(Debug, Clone)]
pub struct CustomFNode<
    T,
    U,
    R = Rc<TransitionWithCustomCost>,
    C = RcChain<TransitionWithCustomCost>,
    P = Rc<C>,
> {
    node: CostNode<T, TransitionWithCustomCost, R, C, P>,
    pub g: U,
    pub h: U,
    pub f: U,
}

impl<T, U, R, C, P> PartialEq for CustomFNode<T, U, R, C, P>
where
    U: PartialEq,
{
    /// Nodes are compared by their f- and h-values.
    /// This does not mean that the nodes are the same.
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.f == other.f && self.h == other.h
    }
}

impl<T, U, R, C, P> Eq for CustomFNode<T, U, R, C, P> where U: Eq {}

impl<T, U, R, C, P> Ord for CustomFNode<T, U, R, C, P>
where
    U: Ord,
{
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        match self.f.cmp(&other.f) {
            Ordering::Equal => self.h.cmp(&other.h),
            result => result,
        }
    }
}

impl<T, U, R, C, P> PartialOrd for CustomFNode<T, U, R, C, P>
where
    U: Ord,
{
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T, U, R, C, P> StateInformation<T> for CustomFNode<T, U, R, C, P>
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
    fn cost(&self, model: &dypdl::Model) -> T {
        self.node.cost(model)
    }

    #[inline]
    fn bound(&self, model: &dypdl::Model) -> Option<T> {
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

impl<T: Numeric, U: Numeric, R, C, P> GetTransitions<TransitionWithCustomCost>
    for CustomFNode<T, U, R, C, P>
where
    C: GetTransitions<TransitionWithCustomCost>,
    P: Deref<Target = C>,
{
    #[inline]
    fn transitions(&self) -> Vec<TransitionWithCustomCost> {
        self.node.transitions()
    }

    #[inline]
    fn last(&self) -> Option<&TransitionWithCustomCost> {
        self.node.last()
    }
}

impl<T, U, R, C, P> BfsNode<T, TransitionWithCustomCost> for CustomFNode<T, U, R, C, P>
where
    T: Numeric + Display,
    U: Numeric + Ord,
    C: GetTransitions<TransitionWithCustomCost>,
    P: Deref<Target = C>,
{
    #[inline]
    fn ordered_by_bound() -> bool {
        false
    }
}

impl<T, U, R, C, P> CustomFNode<T, U, R, C, P>
where
    T: Numeric + Ord,
    U: Numeric + Ord,
    R: Deref<Target = TransitionWithCustomCost>,
    C: CreateTransitionChain<R, P>,
    P: From<C> + Clone,
{
    /// Generate a new node.
    pub fn new(node: CostNode<T, TransitionWithCustomCost, R, C, P>, g: U, h: U, f: U) -> Self {
        CustomFNode { node, g, h, f }
    }

    /// Get the transition chain.
    pub fn transition_chain(&self) -> Option<P> {
        self.node.transition_chain()
    }

    /// Generates a root search node given a state, its cost, its g-value, and h- and f-evaluators.
    ///
    /// Returns `None` if the node is a dead-end.
    ///
    /// `h_evaluator` is a function that takes a state and returns the h-value.
    /// If `h_evaluator` returns `None`, the state is a dead-end, so the node is not generated.
    /// `f_evaluator` is a function that takes `g`, the h-value, and the state and returns the f-value.
    /// If not `maximize`, the h- and f-values are negated.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    /// use dypdl_heuristic_search::search_algorithm::{CustomFNode, StateInRegistry};
    /// use dypdl_heuristic_search::search_algorithm::data_structure::{
    ///     GetTransitions, StateInformation,
    /// };
    ///
    /// let mut model = Model::default();
    /// model.add_integer_variable("variable", 0).unwrap();
    ///
    /// let state = model.target.clone();
    /// let cost = 0;
    /// let g = 1;
    /// let h_evaluator = |_: &StateInRegistry| Some(1);
    /// let f_evaluator = |g, h, _: &StateInRegistry| g + h;
    /// let node = CustomFNode::<_, _>::generate_root_node(
    ///     state, cost, &model, g, &h_evaluator, &f_evaluator, false,
    /// );
    /// assert!(node.is_some());
    /// let node = node.unwrap();
    /// assert_eq!(node.state(), &StateInRegistry::from(model.target.clone()));
    /// assert_eq!(node.cost(&model), cost);
    /// assert_eq!(node.bound(&model), None);
    /// assert!(!node.is_closed());
    /// assert_eq!(node.transitions(), vec![]);
    /// ```
    pub fn generate_root_node<S, H, F>(
        state: S,
        cost: T,
        model: &Model,
        g: U,
        h_evaluator: H,
        f_evaluator: F,
        maximize: bool,
    ) -> Option<Self>
    where
        StateInRegistry: From<S>,
        H: FnOnce(&StateInRegistry) -> Option<U>,
        F: FnOnce(U, U, &StateInRegistry) -> U,
    {
        let node = CostNode::<T, TransitionWithCustomCost, R, C, P>::generate_root_node(
            state, cost, model,
        );
        let h = h_evaluator(node.state())?;
        let f = f_evaluator(g, h, node.state());
        let (h, f) = if maximize { (h, f) } else { (-h, -f) };

        Some(CustomFNode::new(node, g, h, f))
    }

    /// Generates a successor node given a transition, a DyPDL model, and h- and f-evaluators.
    ///
    /// Returns `None` if the successor state is pruned by a state constraint or a dead-end.
    ///
    /// The g-value is the cost to reach this node, computed by the custom cost of the transition.
    /// `h_evaluator` is a function that takes a state and returns the h-value.
    /// If `h_evaluator` returns `None`, the state is a dead-end, so the node is not generated.
    /// `f_evaluator` is a function that takes g- and h-values and the state and returns the f-value.
    /// If not `maximize`, the h- and f-values are negated.
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
    ///     CustomFNode, StateInRegistry, TransitionWithCustomCost,
    /// };
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
    /// let g = 1;
    /// let h_evaluator = |_: &StateInRegistry| Some(0);
    /// let f_evaluator = |g, h, _: &StateInRegistry| g + h;
    /// let node = CustomFNode::<_, _>::generate_root_node(
    ///     state, cost, &model, g, &h_evaluator, &f_evaluator, false,
    /// ).unwrap();
    ///
    /// let mut transition = Transition::new("transition");
    /// transition.set_cost(IntegerExpression::Cost + 1);
    /// transition.add_effect(variable, variable + 1).unwrap();
    /// let transition = TransitionWithCustomCost {
    ///     transition,
    ///     custom_cost: CostExpression::from(IntegerExpression::Cost + 2),
    /// };
    /// let expected_state: StateInRegistry = transition.apply(
    ///     &model.target, &model.table_registry,
    /// );
    ///
    /// let node = node.generate_successor_node(
    ///     Rc::new(transition.clone()), &model, &h_evaluator, &f_evaluator, false,
    /// );
    /// assert!(node.is_some());
    /// let node = node.unwrap();
    /// assert_eq!(node.state(), &expected_state);
    /// assert_eq!(node.cost(&model), 1);
    /// assert_eq!(node.bound(&model), None);
    /// assert!(!node.is_closed());
    /// assert_eq!(node.transitions(), vec![transition]);
    /// ```
    pub fn generate_successor_node<H, F>(
        &self,
        transition: R,
        model: &Model,
        h_evaluator: H,
        f_evaluator: F,
        maximize: bool,
    ) -> Option<Self>
    where
        H: FnOnce(&StateInRegistry) -> Option<U>,
        F: FnOnce(U, U, &StateInRegistry) -> U,
    {
        let g = transition
            .custom_cost
            .eval_cost(self.g, self.state(), &model.table_registry);
        let node = self.node.generate_successor_node(transition, model)?;
        let h = h_evaluator(node.state())?;
        let f = f_evaluator(g, h, node.state());
        let (h, f) = if maximize { (h, f) } else { (-h, -f) };

        Some(CustomFNode::new(node, g, h, f))
    }

    /// Generates a successor node given a transition, a DyPDL model, and h- and f-evaluators.
    /// and inserts it into a state registry.
    ///
    /// Returns the successor node and whether a new entry is generated or not.
    /// If the successor node dominates an existing non-closed node in the registry, the second return value is `false`.
    ///
    /// Returns `None` if the successor state is pruned by a state constraint or a dead-end
    /// or the successor node is dominated.
    ///
    /// The g-value is the cost to reach this node, computed by the custom cost of the transition.
    /// `h_evaluator` is a function that takes a state and returns the h-value.
    /// If `h_evaluator` returns `None`, the state is a dead-end, so the node is not generated.
    /// `f_evaluator` is a function that takes g- and h-values and the state and returns the f-value.
    /// If not `maximize`, the h- and f-values are negated.
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
    ///     CustomFNode, StateInRegistry, StateRegistry, TransitionWithCustomCost,
    /// };
    /// use dypdl_heuristic_search::search_algorithm::data_structure::{
    ///     GetTransitions, StateInformation,
    /// };
    /// use std::rc::Rc;
    ///
    /// let mut model = Model::default();
    /// let variable = model.add_integer_variable("variable", 0).unwrap();
    /// let mut registry = StateRegistry::<_, CustomFNode<_, _>>::new(Rc::new(model.clone()));
    ///
    /// let state = model.target.clone();
    /// let cost = 0;
    /// let g = 1;
    /// let h_evaluator = |_: &StateInRegistry| Some(0);
    /// let f_evaluator = |g, h, _: &StateInRegistry| g + h;
    /// let node = CustomFNode::<_, _>::generate_root_node(
    ///     state, cost, &model, g, &h_evaluator, &f_evaluator, false,
    /// ).unwrap();
    ///
    /// let mut transition = Transition::new("transition");
    /// transition.set_cost(IntegerExpression::Cost + 1);
    /// transition.add_effect(variable, variable + 1).unwrap();
    /// let transition = TransitionWithCustomCost {
    ///     transition,
    ///     custom_cost: CostExpression::from(IntegerExpression::Cost + 2),
    /// };
    /// let expected_state: StateInRegistry = transition.apply(
    ///     &model.target, &model.table_registry,
    /// );
    ///
    /// let result = node.insert_successor_node(
    ///     Rc::new(transition.clone()), &mut registry, &h_evaluator, &f_evaluator, false,
    /// );
    /// assert!(result.is_some());
    /// let (node, generated) = result.unwrap();
    /// assert!(generated);
    /// assert_eq!(node.state(), &expected_state);
    /// assert_eq!(node.cost(&model), 1);
    /// assert_eq!(node.bound(&model), None);
    /// assert!(!node.is_closed());
    /// assert_eq!(node.transitions(), vec![transition]);
    /// ```
    pub fn insert_successor_node<H, F, M>(
        &self,
        transition: R,
        registry: &mut StateRegistry<T, Self, M>,
        h_evaluator: H,
        f_evaluator: F,
        maximize: bool,
    ) -> Option<(Rc<Self>, bool)>
    where
        H: FnOnce(&StateInRegistry) -> Option<U>,
        F: FnOnce(U, U, &StateInRegistry) -> U,
        M: Deref<Target = Model> + Clone,
    {
        let (state, cost) = registry.model().generate_successor_state(
            self.state(),
            self.cost(registry.model()),
            transition.deref(),
            None,
        )?;

        let model = registry.model().clone();

        let constructor = |state, cost, other: Option<&Self>| {
            let h = if let Some(other) = other {
                if maximize {
                    other.h
                } else {
                    -other.h
                }
            } else {
                h_evaluator(&state)?
            };
            let g = transition
                .custom_cost
                .eval_cost(self.g, self.state(), &model.table_registry);
            let f = f_evaluator(g, h, &state);

            let (h, f) = if maximize { (h, f) } else { (-h, -f) };

            let transitions = P::from(C::new(self.transition_chain(), transition));
            let node = CostNode::new(state, cost, &model, Some(transitions));

            Some(CustomFNode::new(node, g, h, f))
        };

        let result = registry.insert_with(state, cost, constructor);

        for d in result.dominated.iter() {
            if !d.is_closed() {
                d.close();
            }
        }

        let node = result.information?;

        Some((node, result.dominated.is_empty()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dypdl::{expression::*, prelude::*};
    use smallvec::SmallVec;

    #[test]
    fn test_new() {
        let mut model = Model::default();
        model.set_minimize();
        let v1 = model.add_integer_resource_variable("v1", true, 0);
        assert!(v1.is_ok());
        let v2 = model.add_integer_resource_variable("v2", false, 0);
        assert!(v2.is_ok());

        let mut state = StateInRegistry::from(model.target.clone());
        let cost = 10;
        let transition_chain = None;

        let node = CostNode::<_, _>::new(state.clone(), cost, &model, transition_chain);
        let mut node = CustomFNode::new(node, 5, 10, 15);
        assert_eq!(node.g, 5);
        assert_eq!(node.h, 10);
        assert_eq!(node.f, 15);
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

        let node = CostNode::<_, _>::new(state.clone(), cost, &model, transition_chain);
        let node = CustomFNode::new(node, 5, 10, 15);
        assert!(!node.is_closed());
        node.close();
        assert!(node.is_closed());
    }

    #[test]
    fn test_get_transitions() {
        let model = Model::default();
        let state = StateInRegistry::from(model.target.clone());
        let transition1 = TransitionWithCustomCost {
            transition: Transition::new("t1"),
            custom_cost: CostExpression::from(IntegerExpression::Cost + 1),
        };
        let transition2 = TransitionWithCustomCost {
            transition: Transition::new("t2"),
            custom_cost: CostExpression::from(IntegerExpression::Cost + 1),
        };
        let cost = 10;
        let transition_chain = Some(Rc::new(RcChain::new(None, Rc::new(transition1.clone()))));
        let transition_chain = Some(Rc::new(RcChain::new(
            transition_chain,
            Rc::new(transition2.clone()),
        )));
        let node = CostNode::<_, _>::new(state, cost, &model, transition_chain.clone());
        let node = CustomFNode::new(node, 5, 10, 15);

        assert_eq!(node.last(), Some(&transition2));
        assert_eq!(node.transitions(), vec![transition1, transition2]);
        assert_eq!(node.transition_chain(), transition_chain);
    }

    #[test]
    fn test_ord() {
        let model = Model::default();

        let node1 = CostNode::<_, _>::new(
            StateInRegistry::from(model.target.clone()),
            10,
            &model,
            None,
        );
        let node1 = CustomFNode::new(node1, -5, -10, -15);
        let node2 = CostNode::<_, _>::new(
            StateInRegistry::from(model.target.clone()),
            10,
            &model,
            None,
        );
        let node2 = CustomFNode::new(node2, -7, -9, -16);
        let node3 = CostNode::<_, _>::new(
            StateInRegistry::from(model.target.clone()),
            10,
            &model,
            None,
        );
        let node3 = CustomFNode::new(node3, -4, -11, -15);

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
        assert!(!CustomFNode::<Integer, Integer>::ordered_by_bound());
    }

    #[test]
    fn test_generate_root_node() {
        let mut model = dypdl::Model::default();
        model.set_minimize();
        let variable = model.add_integer_variable("variable", 0);
        assert!(variable.is_ok());
        let state = model.target.clone();
        let mut expected_state = StateInRegistry::from(state.clone());
        let h_evaluator = |_: &StateInRegistry| Some(0);
        let f_evaluator = |g, h, _: &StateInRegistry| g + h;
        let node = CustomFNode::<_, _>::generate_root_node(
            state,
            1,
            &model,
            2,
            &h_evaluator,
            &f_evaluator,
            false,
        );
        assert!(node.is_some());
        let mut node = node.unwrap();
        assert_eq!(node.g, 2);
        assert_eq!(node.h, 0);
        assert_eq!(node.f, -2);
        assert_eq!(node.state(), &expected_state);
        assert_eq!(node.state_mut(), &mut expected_state);
        assert_eq!(node.cost(&model), 1);
        assert_eq!(node.bound(&model), None);
        assert!(!node.is_closed());
        assert_eq!(node.last(), None);
        assert_eq!(node.transitions(), vec![]);
        assert_eq!(node.transition_chain(), None);
    }

    #[test]
    fn test_generate_root_node_pruned_by_h() {
        let mut model = dypdl::Model::default();
        let variable = model.add_integer_variable("variable", 0);
        assert!(variable.is_ok());
        let state = model.target.clone();
        let h_evaluator = |_: &StateInRegistry| None;
        let f_evaluator = |g, h, _: &StateInRegistry| g + h;
        let node = CustomFNode::<_, _>::generate_root_node(
            state,
            0,
            &model,
            1,
            &h_evaluator,
            &f_evaluator,
            false,
        );
        assert!(node.is_none());
    }

    #[test]
    fn test_generate_successor_some() {
        let mut model = Model::default();
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
        let transition = TransitionWithCustomCost {
            transition,
            custom_cost: CostExpression::from(IntegerExpression::Cost + 2),
        };

        let state = model.target.clone();
        let mut expected_state: StateInRegistry = transition.apply(&state, &model.table_registry);
        let h_evaluator = |_: &StateInRegistry| Some(0);
        let f_evaluator = |g, h, _: &StateInRegistry| g + h;
        let node = CustomFNode::<_, _>::generate_root_node(
            state,
            0,
            &model,
            0,
            &h_evaluator,
            &f_evaluator,
            false,
        );
        assert!(node.is_some());
        let node = node.unwrap();

        let successor = node.generate_successor_node(
            Rc::new(transition.clone()),
            &model,
            &h_evaluator,
            &f_evaluator,
            false,
        );
        assert!(successor.is_some());
        let mut successor = successor.unwrap();
        assert_eq!(successor.g, 2);
        assert_eq!(successor.h, 0);
        assert_eq!(successor.f, -2);
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
        let h_evaluator = |_: &StateInRegistry| Some(0);
        let f_evaluator = |g, h, _: &StateInRegistry| g + h;
        let node = CustomFNode::<_, _>::generate_root_node(
            state,
            0,
            &model,
            0,
            &h_evaluator,
            &f_evaluator,
            false,
        );
        assert!(node.is_some());
        let node = node.unwrap();

        let mut transition = Transition::default();
        let result = transition.add_effect(v1, v1 + 1);
        assert!(result.is_ok());
        let result = transition.add_effect(v2, v2 + 1);
        assert!(result.is_ok());
        transition.set_cost(IntegerExpression::Cost + 1);
        let transition = TransitionWithCustomCost {
            transition,
            custom_cost: CostExpression::from(IntegerExpression::Cost + 2),
        };

        let result = node.generate_successor_node(
            Rc::new(transition),
            &model,
            &h_evaluator,
            &f_evaluator,
            false,
        );
        assert_eq!(result, None);
    }

    #[test]
    fn test_insert_successor_non_dominance() {
        let mut model = Model::default();
        let v1 = model.add_integer_resource_variable("v1", true, 0);
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let v2 = model.add_integer_resource_variable("v2", false, 0);
        assert!(v2.is_ok());
        let v2 = v2.unwrap();
        let model = Rc::new(model);

        let state = StateInRegistry::from(model.target.clone());
        let mut registry = StateRegistry::<_, CustomFNode<_, _>>::new(model.clone());

        let mut transition = Transition::default();
        let result = transition.add_effect(v1, v1 + 1);
        assert!(result.is_ok());
        let result = transition.add_effect(v2, v2 + 1);
        assert!(result.is_ok());
        transition.set_cost(IntegerExpression::Cost + 1);
        let transition = TransitionWithCustomCost {
            transition,
            custom_cost: CostExpression::from(IntegerExpression::Cost + 2),
        };

        let expected_state: StateInRegistry = transition.apply(&state, &model.table_registry);
        let h_evaluator = |_: &StateInRegistry| Some(0);
        let f_evaluator = |g, h, _: &StateInRegistry| g + h;
        let node =
            CustomFNode::generate_root_node(state, 0, &model, 1, &h_evaluator, &f_evaluator, false);
        assert!(node.is_some());
        let node = node.unwrap();
        let result = registry.insert(node.clone());
        assert!(result.information.is_some());

        let result = node.insert_successor_node(
            Rc::new(transition.clone()),
            &mut registry,
            &h_evaluator,
            &f_evaluator,
            false,
        );
        assert!(result.is_some());
        let (successor, generated) = result.unwrap();
        assert_eq!(successor.g, 3);
        assert_eq!(successor.h, 0);
        assert_eq!(successor.f, -3);
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
        let mut registry = StateRegistry::<_, CustomFNode<_, _>>::new(Rc::new(model.clone()));

        let h_evaluator = |_: &StateInRegistry| Some(0);
        let f_evaluator = |g, h, _: &StateInRegistry| g + h;
        let node =
            CustomFNode::generate_root_node(state, 0, &model, 1, &h_evaluator, &f_evaluator, false);
        assert!(node.is_some());
        let node = node.unwrap();

        let mut transition = Transition::default();
        let result = transition.add_effect(v1, v1 + 1);
        assert!(result.is_ok());
        let result = transition.add_effect(v2, v2 + 1);
        assert!(result.is_ok());
        transition.set_cost(IntegerExpression::Cost + 1);
        let transition = TransitionWithCustomCost {
            transition,
            custom_cost: CostExpression::from(IntegerExpression::Cost + 2),
        };

        let result = node.insert_successor_node(
            Rc::new(transition),
            &mut registry,
            &h_evaluator,
            &f_evaluator,
            false,
        );
        assert_eq!(result, None);
        assert!(!node.is_closed());
    }

    #[test]
    fn test_insert_successor_dominating() {
        let mut model = Model::default();
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
        let transition = TransitionWithCustomCost {
            transition,
            custom_cost: CostExpression::from(IntegerExpression::Cost + 2),
        };

        let state = StateInRegistry::from(model.target.clone());
        let expected_state: StateInRegistry = transition.apply(&state, &model.table_registry);
        let h_evaluator = |_: &StateInRegistry| Some(0);
        let f_evaluator = |g, h, _: &StateInRegistry| g + h;
        let node =
            CustomFNode::generate_root_node(state, 0, &model, 1, &h_evaluator, &f_evaluator, false);
        assert!(node.is_some());
        let node = node.unwrap();
        let mut registry = StateRegistry::<_, CustomFNode<_, _>>::new(Rc::new(model.clone()));
        let result = registry.insert(node);
        assert!(result.information.is_some());
        let node = result.information.unwrap();
        let dominated = result.dominated;
        assert_eq!(dominated, SmallVec::<[_; 1]>::new());

        let result = node.insert_successor_node(
            Rc::new(transition.clone()),
            &mut registry,
            &h_evaluator,
            &f_evaluator,
            false,
        );
        assert!(result.is_some());
        let (successor, generated) = result.unwrap();
        assert_eq!(successor.g, 3);
        assert_eq!(successor.h, 0);
        assert_eq!(successor.f, -3);
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
    fn test_insert_successor_dominated() {
        let mut model = Model::default();
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
        let transition = TransitionWithCustomCost {
            transition,
            custom_cost: CostExpression::from(IntegerExpression::Cost + 2),
        };

        let state = StateInRegistry::from(model.target.clone());
        let h_evaluator = |_: &StateInRegistry| Some(0);
        let f_evaluator = |g, h, _: &StateInRegistry| g + h;
        let node =
            CustomFNode::generate_root_node(state, 0, &model, 1, &h_evaluator, &f_evaluator, false);
        assert!(node.is_some());
        let node = node.unwrap();
        let mut registry = StateRegistry::<_, CustomFNode<_, _>>::new(Rc::new(model.clone()));
        let result = registry.insert(node);
        assert!(result.information.is_some());
        let node = result.information.unwrap();
        let dominated = result.dominated;
        assert_eq!(dominated, SmallVec::<[_; 1]>::new());

        let result = node.insert_successor_node(
            Rc::new(transition),
            &mut registry,
            &h_evaluator,
            &f_evaluator,
            false,
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
        let mut registry = StateRegistry::<_, CustomFNode<_, _>>::new(Rc::new(model.clone()));

        let h_evaluator = |_: &StateInRegistry| Some(0);
        let f_evaluator = |g, h, _: &StateInRegistry| g + h;
        let node =
            CustomFNode::generate_root_node(state, 0, &model, 1, &h_evaluator, &f_evaluator, false);
        assert!(node.is_some());
        let node = node.unwrap();

        let mut transition = Transition::default();
        let result = transition.add_effect(v1, v1 + 1);
        assert!(result.is_ok());
        let result = transition.add_effect(v2, v2 + 1);
        assert!(result.is_ok());
        transition.set_cost(IntegerExpression::Cost + 1);
        let transition = TransitionWithCustomCost {
            transition,
            custom_cost: CostExpression::from(IntegerExpression::Cost + 2),
        };

        let h_evaluator = |_: &StateInRegistry| None;
        let result = node.insert_successor_node(
            Rc::new(transition),
            &mut registry,
            &h_evaluator,
            &f_evaluator,
            false,
        );
        assert_eq!(result, None);
        assert!(!node.is_closed());
    }
}
