use super::super::state_registry::{StateInRegistry, StateInformation, StateRegistry};
use super::super::transition::TransitionWithCustomCost;
use super::super::transition_chain::{GetTransitions, TransitionChain};
use super::super::{HashableSignatureVariables, RcChain};
use super::BfsNode;
use dypdl::variable_type::Numeric;
use dypdl::Model;
use std::cell::Cell;
use std::cmp::Ordering;
use std::fmt::{Debug, Display};
use std::ops::Deref;
use std::rc::Rc;

/// Node ordered by the f-value, and tie is broken by the h-value.
///
/// The g-value, which is the cost to reach this node, is computed by customized cost expressions
/// and does not necessarily correspond to the original cost.
/// The h-value is the priority of the state, and the f-value is the priority of this node computed from
/// the g- and h-values.
///
/// In minimization, a node having a lower f-value is `Greater` in `Ord`.
/// In maximization , a node having a higher f-value is `Greater` in `Ord`.
///
/// This node does not have information about a bound.
#[derive(Debug, Clone)]
pub struct CustomFNode<T, U>
where
    T: Numeric,
    U: Numeric,
{
    state: StateInRegistry,
    cost: T,
    g: U,
    h: U,
    f: U,
    closed: Cell<bool>,
    transitions: Option<Rc<RcChain<TransitionWithCustomCost>>>,
}

impl<T, U> CustomFNode<T, U>
where
    T: Numeric,
    U: Numeric,
{
    fn new(
        state: StateInRegistry,
        cost: T,
        g: U,
        h: U,
        f: U,
        parent: Option<&Self>,
        transition: Option<Rc<TransitionWithCustomCost>>,
    ) -> Self {
        let transitions = transition.map(|transition| {
            Rc::new(RcChain::new(
                parent.and_then(|parent| parent.transitions.clone()),
                transition,
            ))
        });

        CustomFNode {
            state,
            cost,
            g,
            h,
            f,
            closed: Cell::new(false),
            transitions,
        }
    }

    /// Generates a root search node given a state, its cost, its g-value, and h- and f-evaluators.
    ///
    /// Returns `None` if the node is a dead-end.
    ///
    /// The g-value is the cost to reach this node.
    /// `h_evaluator` is a function that takes a state and returns the h-value.
    /// If `h_evaluator` returns `None`, the state is a dead-end, so the node is not generated.
    /// `f_evaluator` is a function that takes g- and h-values and the state and returns the f-value.
    /// `maximize` is a flag that indicates whether the f-value should be maximized.
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
    ///     state, cost, g, &h_evaluator, &f_evaluator, false,
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
        let state = StateInRegistry::from(state);
        let h = h_evaluator(&state)?;
        let f = f_evaluator(g, h, &state);
        let (h, f) = if maximize { (h, f) } else { (-h, -f) };

        Some(CustomFNode::new(state, cost, g, h, f, None, None))
    }

    /// Generates a successor node given a transition, a DyPDL model, and h- and f-evaluators.
    ///
    /// Returns `None` if the successor state is pruned by a state constraint or a dead-end.
    ///
    /// `h_evaluator` is a function that takes a state and returns the h-value.
    /// If `h_evaluator` returns `None`, the state is a dead-end, so the node is not generated.
    /// `f_evaluator` is a function that takes g- and h-values and the state and returns the f-value.
    /// `maximize` is a flag that indicates whether the f-value should be maximized.
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
    ///     state, cost, g, &h_evaluator, &f_evaluator, false,
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
        transition: Rc<TransitionWithCustomCost>,
        model: &Model,
        h_evaluator: H,
        f_evaluator: F,
        maximize: bool,
    ) -> Option<Self>
    where
        H: FnOnce(&StateInRegistry) -> Option<U>,
        F: FnOnce(U, U, &StateInRegistry) -> U,
    {
        let (state, cost) =
            model.generate_successor_state(&self.state, self.cost, transition.as_ref(), None)?;
        let g = transition
            .custom_cost
            .eval_cost(self.g, &state, &model.table_registry);
        let h = h_evaluator(&state)?;
        let f = f_evaluator(g, h, &state);
        let (h, f) = if maximize { (h, f) } else { (-h, -f) };

        Some(CustomFNode::new(
            state,
            cost,
            g,
            h,
            f,
            Some(self),
            Some(transition),
        ))
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
    ///     state, cost, g, &h_evaluator, &f_evaluator, false,
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
    pub fn insert_successor_node<H, F, N, M>(
        &self,
        transition: Rc<TransitionWithCustomCost>,
        registry: &mut StateRegistry<T, Self, N, Rc<HashableSignatureVariables>, M>,
        h_evaluator: H,
        f_evaluator: F,
        maximize: bool,
    ) -> Option<(N, bool)>
    where
        H: FnOnce(&StateInRegistry) -> Option<U>,
        F: FnOnce(U, U, &StateInRegistry) -> U,
        N: Deref<Target = Self> + From<Self> + Clone,
        M: Deref<Target = Model> + Clone,
    {
        let (state, cost) = registry.model().generate_successor_state(
            &self.state,
            self.cost,
            transition.as_ref(),
            None,
        )?;

        let model = registry.model().clone();

        let constructor = |state, cost, other: Option<&CustomFNode<T, U>>| {
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
                .eval_cost(self.g, &state, &model.table_registry);
            let f = f_evaluator(g, h, &state);

            let (h, f) = if maximize { (h, f) } else { (-h, -f) };

            Some(CustomFNode::new(
                state,
                cost,
                g,
                h,
                f,
                Some(self),
                Some(transition),
            ))
        };

        let (successor, dominated) = registry.insert_with(state, cost, constructor)?;

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

impl<T, U> PartialEq for CustomFNode<T, U>
where
    T: Numeric,
    U: Numeric,
{
    /// Nodes are compared by their f- and h-values.
    /// This does not mean that the nodes are the same.
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.f == other.f && self.h == other.h
    }
}

impl<T, U> Eq for CustomFNode<T, U>
where
    T: Numeric,
    U: Numeric,
{
}

impl<T, U> Ord for CustomFNode<T, U>
where
    T: Numeric,
    U: Numeric + Ord,
{
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        match self.f.cmp(&other.f) {
            Ordering::Equal => self.h.cmp(&other.h),
            result => result,
        }
    }
}

impl<T, U> PartialOrd for CustomFNode<T, U>
where
    T: Numeric,
    U: Numeric + Ord,
{
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T, U> StateInformation<T> for CustomFNode<T, U>
where
    T: Numeric,
    U: Numeric,
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
    fn cost(&self, _: &dypdl::Model) -> T {
        self.cost
    }

    #[inline]
    fn bound(&self, _: &dypdl::Model) -> Option<T> {
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

impl<T: Numeric, U: Numeric> GetTransitions<TransitionWithCustomCost> for CustomFNode<T, U> {
    #[inline]
    fn transitions(&self) -> Vec<TransitionWithCustomCost> {
        self.transitions
            .as_ref()
            .map_or_else(Vec::new, |transitions| transitions.transitions())
    }
}

impl<T, U> BfsNode<T, TransitionWithCustomCost> for CustomFNode<T, U>
where
    T: Numeric + Display,
    U: Numeric + Ord,
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
        assert!(!CustomFNode::<Integer, Integer>::ordered_by_bound());
    }

    #[test]
    fn generate_root_node() {
        let mut model = dypdl::Model::default();
        model.set_minimize();
        let variable = model.add_integer_variable("variable", 0);
        assert!(variable.is_ok());
        let state = model.target.clone();
        let mut expected_state = StateInRegistry::from(state.clone());
        let h_evaluator = |_: &StateInRegistry| Some(0);
        let f_evaluator = |g, h, _: &StateInRegistry| g + h;
        let node =
            CustomFNode::<_, _>::generate_root_node(state, 1, 2, &h_evaluator, &f_evaluator, false);
        assert!(node.is_some());
        let mut node = node.unwrap();
        assert_eq!(node.state(), &expected_state);
        assert_eq!(node.state_mut(), &mut expected_state);
        assert_eq!(node.cost(&model), 1);
        assert_eq!(node.bound(&model), None);
        assert!(!node.is_closed());
        assert_eq!(node.transitions(), vec![]);
    }

    #[test]
    fn generate_root_node_pruned_by_h() {
        let mut model = dypdl::Model::default();
        let variable = model.add_integer_variable("variable", 0);
        assert!(variable.is_ok());
        let state = model.target.clone();
        let h_evaluator = |_: &StateInRegistry| None;
        let f_evaluator = |g, h, _: &StateInRegistry| g + h;
        let node =
            CustomFNode::<_, _>::generate_root_node(state, 0, 1, &h_evaluator, &f_evaluator, false);
        assert!(node.is_none());
    }

    #[test]
    fn close() {
        let model = dypdl::Model::default();
        let state = model.target;
        let h_evaluator = |_: &StateInRegistry| Some(0);
        let f_evaluator = |g, h, _: &StateInRegistry| g + h;
        let node =
            CustomFNode::<_, _>::generate_root_node(state, 0, 1, &h_evaluator, &f_evaluator, false);
        assert!(node.is_some());
        let node = node.unwrap();
        assert!(!node.is_closed());
        node.close();
        assert!(node.is_closed());
    }

    #[test]
    fn generate_successor_some() {
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
        let node = CustomFNode::generate_root_node(state, 0, 0, &h_evaluator, &f_evaluator, false);
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
        let h_evaluator = |_: &StateInRegistry| Some(0);
        let f_evaluator = |g, h, _: &StateInRegistry| g + h;
        let node = CustomFNode::generate_root_node(state, 0, 0, &h_evaluator, &f_evaluator, false);
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
    fn insert_successor_non_dominance() {
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
        let node = CustomFNode::generate_root_node(state, 0, 1, &h_evaluator, &f_evaluator, false);
        assert!(node.is_some());
        let node = node.unwrap();
        let result = registry.insert(node.clone());
        assert!(result.is_some());

        let result = node.insert_successor_node(
            Rc::new(transition.clone()),
            &mut registry,
            &h_evaluator,
            &f_evaluator,
            false,
        );
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
        let mut registry = StateRegistry::<_, CustomFNode<_, _>>::new(Rc::new(model.clone()));

        let h_evaluator = |_: &StateInRegistry| Some(0);
        let f_evaluator = |g, h, _: &StateInRegistry| g + h;
        let node = CustomFNode::generate_root_node(state, 0, 1, &h_evaluator, &f_evaluator, false);
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
    fn insert_successor_dominating() {
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
        let node = CustomFNode::generate_root_node(state, 0, 1, &h_evaluator, &f_evaluator, false);
        assert!(node.is_some());
        let node = node.unwrap();
        let mut registry = StateRegistry::<_, CustomFNode<_, _>>::new(Rc::new(model.clone()));
        let result = registry.insert(node);
        assert!(result.is_some());
        let (node, dominated) = result.unwrap();
        assert!(dominated.is_none());

        let result = node.insert_successor_node(
            Rc::new(transition.clone()),
            &mut registry,
            &h_evaluator,
            &f_evaluator,
            false,
        );
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
    fn insert_successor_dominated() {
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
        let node = CustomFNode::generate_root_node(state, 0, 1, &h_evaluator, &f_evaluator, false);
        assert!(node.is_some());
        let node = node.unwrap();
        let mut registry = StateRegistry::<_, CustomFNode<_, _>>::new(Rc::new(model.clone()));
        let result = registry.insert(node);
        assert!(result.is_some());
        let (node, dominated) = result.unwrap();
        assert!(dominated.is_none());

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
    fn insert_successor_pruned_by_h() {
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
        let node = CustomFNode::generate_root_node(state, 0, 1, &h_evaluator, &f_evaluator, false);
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

    #[test]
    fn cmp_min() {
        let mut model = Model::default();
        model.set_maximize();
        let v1 = model.add_integer_resource_variable("v1", true, 0);
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let v2 = model.add_integer_resource_variable("v2", false, 0);
        assert!(v2.is_ok());
        let v2 = v2.unwrap();
        let model = Rc::new(model);

        let state = model.target.clone();
        let h_evaluator_0 = |_: &StateInRegistry| Some(0);
        let f_evaluator = |g, h, _: &StateInRegistry| g + h;
        let node1 = CustomFNode::<_, _>::generate_root_node(
            state.clone(),
            0,
            0,
            &h_evaluator_0,
            &f_evaluator,
            false,
        );
        assert!(node1.is_some());
        let node1 = Rc::new(node1.unwrap());

        let mut transition = Transition::default();
        let result = transition.add_effect(v1, v1 + 1);
        assert!(result.is_ok());
        let result = transition.add_effect(v2, v2 + 1);
        assert!(result.is_ok());
        let transition = TransitionWithCustomCost {
            transition,
            custom_cost: CostExpression::from(IntegerExpression::Cost),
        };
        let node2 = node1.generate_successor_node(
            Rc::new(transition),
            &model,
            &h_evaluator_0,
            &f_evaluator,
            false,
        );
        assert!(node2.is_some());
        let node2 = Rc::new(node2.unwrap());

        let transition = Transition::default();
        let transition = TransitionWithCustomCost {
            transition,
            custom_cost: CostExpression::from(IntegerExpression::Cost + 1),
        };
        let mut registry = StateRegistry::<_, CustomFNode<_, _>>::new(model);
        let node3 = node1.insert_successor_node(
            Rc::new(transition),
            &mut registry,
            &h_evaluator_0,
            &f_evaluator,
            false,
        );
        assert!(node3.is_some());
        let (node3, _) = node3.unwrap();

        let h_evaluator_1 = |_: &StateInRegistry| Some(1);
        let node4 = CustomFNode::<_, _>::generate_root_node(
            state,
            0,
            0,
            &h_evaluator_1,
            &f_evaluator,
            false,
        );
        assert!(node4.is_some());
        let node4 = Rc::new(node4.unwrap());

        assert!(node1 == node1);
        assert!(node1 >= node1);
        assert!(node1 == node2);
        assert!(node1 >= node2);
        assert!(node1 <= node2);
        assert!(node1 >= node3);
        assert!(node1 > node3);
        assert!(node1 != node3);
        assert!(node1 >= node4);
        assert!(node1 > node4);
        assert!(node1 != node4);
        assert!(node3 >= node4);
        assert!(node3 > node4);
        assert!(node3 != node4);
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
        let model = Rc::new(model);

        let state = model.target.clone();
        let h_evaluator_0 = |_: &StateInRegistry| Some(0);
        let f_evaluator = |g, h, _: &StateInRegistry| g + h;
        let node1 = CustomFNode::<_, _>::generate_root_node(
            state.clone(),
            0,
            0,
            &h_evaluator_0,
            &f_evaluator,
            true,
        );
        assert!(node1.is_some());
        let node1 = Rc::new(node1.unwrap());

        let mut transition = Transition::default();
        let result = transition.add_effect(v1, v1 + 1);
        assert!(result.is_ok());
        let result = transition.add_effect(v2, v2 + 1);
        assert!(result.is_ok());
        let transition = TransitionWithCustomCost {
            transition,
            custom_cost: CostExpression::from(IntegerExpression::Cost),
        };
        let node2 = node1.generate_successor_node(
            Rc::new(transition),
            &model,
            &h_evaluator_0,
            &f_evaluator,
            true,
        );
        assert!(node2.is_some());
        let node2 = Rc::new(node2.unwrap());

        let transition = Transition::default();
        let transition = TransitionWithCustomCost {
            transition,
            custom_cost: CostExpression::from(IntegerExpression::Cost + 1),
        };
        let mut registry = StateRegistry::<_, CustomFNode<_, _>>::new(model);
        let node3 = node1.insert_successor_node(
            Rc::new(transition),
            &mut registry,
            &h_evaluator_0,
            &f_evaluator,
            true,
        );
        assert!(node3.is_some());
        let (node3, _) = node3.unwrap();

        let h_evaluator_1 = |_: &StateInRegistry| Some(1);
        let node4 = CustomFNode::<_, _>::generate_root_node(
            state,
            0,
            0,
            &h_evaluator_1,
            &f_evaluator,
            true,
        );
        assert!(node4.is_some());
        let node4 = Rc::new(node4.unwrap());

        assert!(node1 == node1);
        assert!(node1 >= node1);
        assert!(node1 == node2);
        assert!(node1 >= node2);
        assert!(node1 <= node2);
        assert!(node1 <= node3);
        assert!(node1 < node3);
        assert!(node1 != node3);
        assert!(node1 <= node4);
        assert!(node1 < node4);
        assert!(node1 != node4);
        assert!(node3 <= node4);
        assert!(node3 < node4);
        assert!(node3 != node4);
    }
}
