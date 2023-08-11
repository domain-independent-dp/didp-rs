use super::super::hashable_state::HashableSignatureVariables;
use super::super::state_registry::{StateInRegistry, StateInformation, StateRegistry};
use super::super::transition_chain::{GetTransitions, RcChain, TransitionChain};
use super::super::util::exceed_bound;
use super::BfsNode;
use dypdl::variable_type::Numeric;
use dypdl::{Model, ReduceFunction, Transition, TransitionInterface};
use std::cell::Cell;
use std::cmp::Ordering;
use std::fmt::Display;
use std::ops::Deref;
use std::rc::Rc;

/// Node ordered by the f-value, and tie is broken by the h-value.
///
/// The f-value is the priority of the node and can be different from a dual bound of the node,
/// which is the dual bound of the path cost from the target state to a base state via this node.
/// The f-value is computed from the dual bound of the node, e.g., by an weighted sum of the g- and h-values.
/// The g-value is the path cost from the target state to this state,
/// and the h-value is a dual bound on the path cost from this node to a base state.
///
/// In minimization, a node having a lower f-value is `Greater` in `Ord`.
/// In maximization , a node having a higher f-value is `Greater` in `Ord`.
#[derive(Debug, Clone)]
pub struct WeightedFNode<T, U, V = Transition>
where
    T: Numeric,
    U: Numeric,
    V: TransitionInterface + Clone,
    Transition: From<V>,
{
    state: StateInRegistry<Rc<HashableSignatureVariables>>,
    g: T,
    h: T,
    bound: T,
    f: U,
    closed: Cell<bool>,
    transitions: Option<Rc<RcChain<V>>>,
}

impl<T, U, V> WeightedFNode<T, U, V>
where
    T: Numeric,
    U: Numeric,
    V: TransitionInterface + Clone,
    Transition: From<V>,
{
    fn new(
        state: StateInRegistry,
        cost: T,
        h: T,
        bound: T,
        f: U,
        parent: Option<&Self>,
        transition: Option<Rc<V>>,
    ) -> Self {
        let transitions = transition.map(|transition| {
            Rc::new(RcChain::new(
                parent.and_then(|parent| parent.transitions.clone()),
                transition,
            ))
        });

        WeightedFNode {
            state,
            g: cost,
            h,
            bound,
            f,
            closed: Cell::new(false),
            transitions,
        }
    }

    /// Generates a root search node given a state, its cost, a DyPDL model, h-, bound, and f-evaluators,
    /// and a primal bound on the solution cost.
    ///
    /// Returns `None` if the node is a dead-end, or the f-value exceeds the primal bound.
    ///
    /// `h_evaluator` is a function that takes a state and returns the dual bound (the h-value).
    /// If `h_evaluator` returns `None`, the state is a dead-end, so the node is not generated.
    /// `bound_evaluator` is a function that takes g- and h-values and the state and returns the dual bound of this node.
    /// `f_evaluator`  is a function that takes g- and h-values and the state and returns the f-value.
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
        let state = StateInRegistry::from(state);
        let h = h_evaluator(&state)?;
        let bound = bound_evaluator(cost, h, &state);

        if exceed_bound(model, bound, primal_bound) {
            return None;
        }

        let f = f_evaluator(cost, h, &state);
        let (h, f) = if model.reduce_function == ReduceFunction::Max {
            (h, f)
        } else {
            (-h, -f)
        };
        Some(WeightedFNode::new(state, cost, h, bound, f, None, None))
    }

    /// Generates a successor node given a transition, a DyPDL model, h-, bound, and f-evaluators,
    /// and a primal bound on the solution cost.
    ///
    /// Returns `None` if the successor state is pruned by a state constraint or a dead-end,
    /// or the dual bound exceeds the primal bound.
    ///
    /// `h_evaluator` is a function that takes a state and returns the dual bound (the h-value).
    /// If `h_evaluator` returns `None`, the state is a dead-end, so the node is not generated.
    /// `bound_evaluator` is a function that takes g- and h-values and the state and returns the dual bound of this node.
    /// `f_evaluator`  is a function that takes g- and h-values and the state and returns the f-value.
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
        transition: Rc<V>,
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
        let (state, g) =
            model.generate_successor_state(&self.state, self.g, transition.as_ref(), None)?;
        let h = h_evaluator(&state)?;
        let bound = bound_evaluator(g, h, &state);

        if exceed_bound(model, bound, primal_bound) {
            return None;
        }

        let f = f_evaluator(g, h, &state);
        let (h, f) = if model.reduce_function == ReduceFunction::Max {
            (h, f)
        } else {
            (-h, -f)
        };

        Some(WeightedFNode::new(
            state,
            g,
            h,
            bound,
            f,
            Some(self),
            Some(transition),
        ))
    }

    /// Generates a successor node given a transition, h-, bound, and f- evaluators,
    /// and a primal bound on the solution cost, and inserts it into a state registry.
    ///
    /// Returns the successor node and whether a new entry is generated or not.
    /// If the successor node dominates an existing non-closed node in the registry, the second return value is `false`.
    ///
    /// `h_evaluator` is a function that takes a state and returns the dual bound (the h-value).
    /// If `h_evaluator` returns `None`, the state is a dead-end, so the node is not generated.
    /// `bound_evaluator` is a function that takes g- and h-values and the state and returns the dual bound of this node.
    /// `f_evaluator`  is a function that takes g- and h-values and the state and returns the f-value.
    ///
    /// Returns `None` if the successor state is pruned by a state constraint or a dead-end,
    /// the f-value exceeds the primal bound, or the successor node is dominated.
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
    pub fn insert_successor_node<H, B, F, N, M>(
        &self,
        transition: Rc<V>,
        registry: &mut StateRegistry<T, Self, N, Rc<HashableSignatureVariables>, M>,
        h_evaluator: H,
        bound_evaluator: B,
        f_evaluator: F,
        primal_bound: Option<T>,
    ) -> Option<(N, bool)>
    where
        H: FnOnce(&StateInRegistry) -> Option<T>,
        B: FnOnce(T, T, &StateInRegistry) -> T,
        F: FnOnce(T, T, &StateInRegistry) -> U,
        N: Deref<Target = Self> + From<Self> + Clone,
        M: Deref<Target = Model> + Clone,
    {
        let (state, g) = registry.model().generate_successor_state(
            &self.state,
            self.g,
            transition.as_ref(),
            None,
        )?;

        let model = registry.model().clone();
        let maximize = model.reduce_function == ReduceFunction::Max;

        let constructor = |state, g, other: Option<&WeightedFNode<T, U, V>>| {
            let h = if let Some(other) = other {
                if maximize {
                    other.h
                } else {
                    -other.h
                }
            } else {
                h_evaluator(&state)?
            };
            let bound = bound_evaluator(g, h, &state);

            if exceed_bound(&model, bound, primal_bound) {
                return None;
            }

            let f = f_evaluator(g, h, &state);
            let (h, f) = if maximize { (h, f) } else { (-h, -f) };

            Some(WeightedFNode::new(
                state,
                g,
                h,
                bound,
                f,
                Some(self),
                Some(transition),
            ))
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

impl<T, U, V> PartialEq for WeightedFNode<T, U, V>
where
    T: Numeric + PartialOrd,
    U: Numeric + PartialOrd,
    V: TransitionInterface + Clone,
    Transition: From<V>,
{
    /// Nodes are compared by their f- and h-values.
    /// This does not mean that the nodes are the same.
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.f == other.f && self.h == other.h
    }
}

impl<T, U, V> Eq for WeightedFNode<T, U, V>
where
    T: Numeric + Ord,
    U: Numeric + Ord,
    V: TransitionInterface + Clone,
    Transition: From<V>,
{
}

impl<T, U, V> Ord for WeightedFNode<T, U, V>
where
    T: Numeric + Ord,
    U: Numeric + Ord,
    V: TransitionInterface + Clone,
    Transition: From<V>,
{
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        match self.f.cmp(&other.f) {
            Ordering::Equal => self.h.cmp(&other.h),
            result => result,
        }
    }
}

impl<T, U, V> PartialOrd for WeightedFNode<T, U, V>
where
    T: Numeric + Ord,
    U: Numeric + Ord,
    V: TransitionInterface + Clone,
    Transition: From<V>,
{
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T, U, V> StateInformation<T, Rc<HashableSignatureVariables>> for WeightedFNode<T, U, V>
where
    T: Numeric,
    U: Numeric,
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
    fn cost(&self, _: &Model) -> T {
        self.g
    }

    #[inline]
    fn bound(&self, _: &Model) -> Option<T> {
        Some(self.bound)
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

impl<T, U, V> GetTransitions<V> for WeightedFNode<T, U, V>
where
    T: Numeric + Ord,
    U: Numeric + Ord,
    V: TransitionInterface + Clone,
    Transition: From<V>,
{
    #[inline]
    fn transitions(&self) -> Vec<V> {
        self.transitions
            .as_ref()
            .map_or_else(Vec::new, |transitions| transitions.transitions())
    }

    #[inline]
    fn last(&self) -> Option<&V> {
        self.transitions
            .as_ref()
            .map(|transitions| transitions.last())
    }
}

impl<T, U, V> BfsNode<T, V> for WeightedFNode<T, U, V>
where
    T: Numeric + Ord + Display,
    U: Numeric + Ord,
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
        assert!(!WeightedFNode::<Integer, Integer>::ordered_by_bound());
    }

    #[test]
    fn generate_root_node_some_min() {
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
        assert_eq!(node.state(), &expected_state);
        assert_eq!(node.state_mut(), &mut expected_state);
        assert_eq!(node.cost(&model), 1);
        assert_eq!(node.bound(&model), Some(2));
        assert!(!node.is_closed());
        assert_eq!(node.last(), None);
        assert_eq!(node.transitions(), vec![]);
    }

    #[test]
    fn generate_root_node_some_max() {
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
        assert_eq!(node.state(), &expected_state);
        assert_eq!(node.state_mut(), &mut expected_state);
        assert_eq!(node.cost(&model), 1);
        assert_eq!(node.bound(&model), Some(2));
        assert!(!node.is_closed());
        assert_eq!(node.last(), None);
        assert_eq!(node.transitions(), vec![]);
    }

    #[test]
    fn generate_root_node_pruned_by_bound_min() {
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
    fn generate_root_node_pruned_by_bound_max() {
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
    fn generate_root_node_pruned_by_h() {
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
    fn close() {
        let model = dypdl::Model::default();
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
        assert_eq!(successor.state(), &expected_state);
        assert_eq!(successor.state_mut(), &mut expected_state);
        assert_eq!(successor.cost(&model), 1);
        assert_eq!(successor.bound(&model), Some(2));
        assert!(!successor.is_closed());
        assert_eq!(successor.last(), Some(&transition));
        assert_eq!(successor.transitions(), vec![transition]);
    }

    #[test]
    fn generate_successor_some_max() {
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
        assert_eq!(successor.state(), &expected_state);
        assert_eq!(successor.state_mut(), &mut expected_state);
        assert_eq!(successor.cost(&model), 1);
        assert_eq!(successor.bound(&model), Some(2));
        assert!(!successor.is_closed());
        assert_eq!(successor.last(), Some(&transition));
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
    fn generate_successor_pruned_by_bound_min() {
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
    fn generate_successor_pruned_by_bound_max() {
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
    fn generate_successor_pruned_by_h() {
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
        assert_eq!(successor.cost(&model), 1);
        assert_eq!(successor.bound(&model), Some(2));
        assert!(!successor.is_closed());
        assert_eq!(successor.last(), Some(&transition));
        assert_eq!(successor.transitions(), vec![transition],);
        assert!(generated);
        assert!(!node.is_closed());
    }

    #[test]
    fn insert_successor_non_dominance_max() {
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
        assert_eq!(successor.state(), &expected_state);
        assert_eq!(successor.cost(&model), 1);
        assert_eq!(successor.bound(&model), Some(2));
        assert!(!successor.is_closed());
        assert_eq!(successor.last(), Some(&transition));
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
        assert_eq!(successor.state(), &expected_state);
        assert_eq!(successor.cost(&model), 0);
        assert_eq!(successor.bound(&model), Some(1));
        assert!(!successor.is_closed());
        assert_eq!(successor.last(), Some(&transition));
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
        assert_eq!(successor.state(), &expected_state);
        assert_eq!(successor.cost(&model), 1);
        assert_eq!(successor.bound(&model), Some(1));
        assert!(!successor.is_closed());
        assert_eq!(successor.last(), Some(&transition));
        assert_eq!(successor.transitions(), vec![transition]);
        assert!(!generated);
        assert!(node.is_closed());
    }

    #[test]
    fn insert_successor_pruned_by_bound_min() {
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
    fn insert_successor_pruned_by_bound_max() {
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
    fn insert_successor_pruned_by_h() {
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
        let model = Rc::new(model);

        let state = model.target.clone();
        let h_evaluator_1 = |_: &StateInRegistry| Some(1);
        let bound_evaluator = |g, h, _: &StateInRegistry| g + h;
        let f_evaluator = |g, h, _: &StateInRegistry| g + 2 * h;
        let node1 = WeightedFNode::<_, _>::generate_root_node(
            state.clone(),
            1,
            &model,
            &h_evaluator_1,
            &bound_evaluator,
            &f_evaluator,
            None,
        );
        assert!(node1.is_some());
        let node1 = Rc::new(node1.unwrap());

        let mut transition = Transition::default();
        let result = transition.add_effect(v1, v1 + 1);
        assert!(result.is_ok());
        let result = transition.add_effect(v2, v2 + 1);
        assert!(result.is_ok());
        let node2 = node1.generate_successor_node(
            Rc::new(transition),
            &model,
            &h_evaluator_1,
            &bound_evaluator,
            &f_evaluator,
            None,
        );
        assert!(node2.is_some());
        let node2 = Rc::new(node2.unwrap());

        let mut transition = Transition::default();
        transition.set_cost(IntegerExpression::Cost + 1);
        let mut registry = StateRegistry::<_, WeightedFNode<_, _>>::new(model.clone());
        let node3 = node1.insert_successor_node(
            Rc::new(transition),
            &mut registry,
            &h_evaluator_1,
            &bound_evaluator,
            &f_evaluator,
            None,
        );
        assert!(node3.is_some());
        let (node3, _) = node3.unwrap();

        let h_evaluator_2 = |_: &StateInRegistry| Some(2);
        let node4 = WeightedFNode::<_, _>::generate_root_node(
            state,
            0,
            &model,
            &h_evaluator_2,
            &bound_evaluator,
            &f_evaluator,
            None,
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
        let h_evaluator_1 = |_: &StateInRegistry| Some(1);
        let bound_evaluator = |g, h, _: &StateInRegistry| g + h;
        let f_evaluator = |g, h, _: &StateInRegistry| g + 2 * h;
        let node1 = WeightedFNode::<_, _>::generate_root_node(
            state.clone(),
            1,
            &model,
            &h_evaluator_1,
            &bound_evaluator,
            &f_evaluator,
            None,
        );
        assert!(node1.is_some());
        let node1 = Rc::new(node1.unwrap());

        let mut transition = Transition::default();
        let result = transition.add_effect(v1, v1 + 1);
        assert!(result.is_ok());
        let result = transition.add_effect(v2, v2 + 1);
        assert!(result.is_ok());
        let node2 = node1.generate_successor_node(
            Rc::new(transition),
            &model,
            &h_evaluator_1,
            &bound_evaluator,
            &f_evaluator,
            None,
        );
        assert!(node2.is_some());
        let node2 = Rc::new(node2.unwrap());

        let mut transition = Transition::default();
        transition.set_cost(IntegerExpression::Cost + 1);
        let mut registry = StateRegistry::<_, WeightedFNode<_, _>>::new(model.clone());
        let node3 = node1.insert_successor_node(
            Rc::new(transition),
            &mut registry,
            &h_evaluator_1,
            &bound_evaluator,
            &f_evaluator,
            None,
        );
        assert!(node3.is_some());
        let (node3, _) = node3.unwrap();

        let h_evaluator_2 = |_: &StateInRegistry| Some(2);
        let node4 = WeightedFNode::<_, _>::generate_root_node(
            state,
            0,
            &model,
            &h_evaluator_2,
            &bound_evaluator,
            &f_evaluator,
            None,
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
