use super::super::arc_chain::ArcChain;
use super::super::concurrent_state_registry::ConcurrentStateRegistry;
use crate::search_algorithm::data_structure::{
    exceed_bound, CreateTransitionChain, GetTransitions, HashableSignatureVariables,
    ParentAndChildStateFunctionCache, StateInformation,
};
use crate::search_algorithm::{BfsNode, StateInRegistry};
use dypdl::variable_type::Numeric;
use dypdl::{StateFunctionCache, Model, ReduceFunction, Transition, TransitionInterface};
use std::cmp::Ordering;
use std::fmt::Display;
use std::sync::atomic;
use std::sync::Arc;

/// Node ordered by the f-value, and tie is broken by the h-value.
///
/// This struct is sendable.
///
/// The f-value is a dual bound on the path cost from the target state to a base state via this node.
/// The h-value is a dual bound on the path cost from this node to a base state.
///
/// In minimization, a node having a lower f-value is `Greater` in `Ord`.
/// In maximization , a node having a higher f-value is `Greater` in `Ord`.
#[derive(Debug)]
pub struct SendableFNode<T, V = Transition>
where
    T: Numeric,
    V: TransitionInterface + Clone,
    Transition: From<V>,
{
    state: StateInRegistry<Arc<HashableSignatureVariables>>,
    g: T,
    h: T,
    f: T,
    closed: atomic::AtomicBool,
    transitions: Option<Arc<ArcChain<V>>>,
}

impl<T, V> SendableFNode<T, V>
where
    T: Numeric,
    V: TransitionInterface + Clone,
    Transition: From<V>,
{
    fn new(
        state: StateInRegistry<Arc<HashableSignatureVariables>>,
        cost: T,
        h: T,
        f: T,
        parent: Option<&Self>,
        transition: Option<Arc<V>>,
    ) -> Self {
        let transitions = transition.map(|transition| {
            Arc::new(ArcChain::new(
                parent.and_then(|parent| parent.transitions.clone()),
                transition,
            ))
        });

        SendableFNode {
            state,
            g: cost,
            h,
            f,
            closed: atomic::AtomicBool::new(false),
            transitions,
        }
    }

    /// Generates a root search node given a state, its cost, a DyPDL model, h- and f-evaluators,
    /// and a primal bound on the solution cost.
    ///
    /// Returns `None` if the node is a dead-end, or the f-value exceeds the primal bound.
    ///
    /// `h_evaluator` is a function that takes a state and returns the dual bound (the h-value).
    /// If `h_evaluator` returns `None`, the state is a dead-end, so the node is not generated.
    /// `f_evaluator` is a function that takes g- and h-values and the state and returns the f-value.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    /// use dypdl_heuristic_search::SendableFNode;
    /// use dypdl_heuristic_search::search_algorithm::StateInRegistry;
    /// use dypdl_heuristic_search::search_algorithm::data_structure::{
    ///     GetTransitions, StateInformation,
    /// };
    ///
    /// let mut model = Model::default();
    /// model.add_integer_variable("variable", 0).unwrap();
    ///
    /// let state = model.target.clone();
    /// let mut function_cache = StateFunctionCache::new(&model.state_functions);
    /// let cost = 0;
    /// let h_evaluator = |_: &_, _: &mut _| Some(0);
    /// let f_evaluator = |g, h, _: &_| g + h;
    /// let node = SendableFNode::<_>::generate_root_node(
    ///     state, &mut function_cache, cost, &model, &h_evaluator, &f_evaluator, None,
    /// );
    /// assert!(node.is_some());
    /// let node = node.unwrap();
    /// assert_eq!(node.state(), &StateInRegistry::from(model.target.clone()));
    /// assert_eq!(node.cost(&model), cost);
    /// assert_eq!(node.bound(&model), Some(0));
    /// assert!(!node.is_closed());
    /// assert_eq!(node.transitions(), vec![]);
    /// ```
    pub fn generate_root_node<S, H, F>(
        state: S,
        function_cache: &mut StateFunctionCache,
        cost: T,
        model: &Model,
        h_evaluator: H,
        f_evaluator: F,
        primal_bound: Option<T>,
    ) -> Option<Self>
    where
        StateInRegistry<Arc<HashableSignatureVariables>>: From<S>,
        H: FnOnce(
            &StateInRegistry<Arc<HashableSignatureVariables>>,
            &mut StateFunctionCache,
        ) -> Option<T>,
        F: FnOnce(T, T, &StateInRegistry<Arc<HashableSignatureVariables>>) -> T,
    {
        let state = StateInRegistry::from(state);
        let h = h_evaluator(&state, function_cache)?;
        let f = f_evaluator(cost, h, &state);

        if exceed_bound(model, f, primal_bound) {
            return None;
        }

        let (h, f) = if model.reduce_function == ReduceFunction::Max {
            (h, f)
        } else {
            (-h, -f)
        };

        Some(SendableFNode::new(state, cost, h, f, None, None))
    }

    /// Generates a successor node given a transition, a DyPDL model, h- and f-evaluators,
    /// and a primal bound on the solution cost.
    ///
    /// `function_cache.parent` is not cleared and updated by this node while `function_cache.child` is cleared and updated by the successor node if generated.
    ///
    /// Returns `None` if the successor state is pruned by a state constraint or a dead-end,
    /// or the f-value exceeds the primal bound.
    ///
    /// `h_evaluator` is a function that takes a state and returns the dual bound (the h-value).
    /// If `h_evaluator` returns `None`, the state is a dead-end, so the node is not generated.
    /// `f_evaluator` is a function that takes g- and h-values and the state and returns the f-value.
    ///
    /// # Panics
    ///
    /// If an expression used in the transition is invalid.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    /// use dypdl_heuristic_search::SendableFNode;
    /// use dypdl_heuristic_search::search_algorithm::StateInRegistry;
    /// use dypdl_heuristic_search::search_algorithm::data_structure::{
    ///     GetTransitions, StateInformation, ParentAndChildStateFunctionCache,
    /// };
    /// use std::sync::Arc;
    ///
    /// let mut model = Model::default();
    /// let variable = model.add_integer_variable("variable", 0).unwrap();
    ///
    /// let state = model.target.clone();
    /// let mut function_cache = ParentAndChildStateFunctionCache::new(&model.state_functions);
    /// let cost = 0;
    /// let h_evaluator = |_: &_, _: &mut _| Some(0);
    /// let f_evaluator = |g, h, _: &_| g + h;
    /// let node = SendableFNode::<_>::generate_root_node(
    ///     state, &mut function_cache.parent, cost, &model, &h_evaluator, &f_evaluator, None,
    /// ).unwrap();
    ///
    /// let mut transition = Transition::new("transition");
    /// transition.set_cost(IntegerExpression::Cost + 1);
    /// transition.add_effect(variable, variable + 1).unwrap();
    /// let mut function_cache_for_expected = StateFunctionCache::new(&model.state_functions);
    /// let expected_state: StateInRegistry<_> = transition.apply(
    ///     &model.target, &mut function_cache_for_expected, &model.state_functions, &model.table_registry,
    /// );
    ///
    /// let node = node.generate_successor_node(
    ///     Arc::new(transition.clone()),
    ///     &mut function_cache,
    ///     &model,
    ///     &h_evaluator,
    ///     &f_evaluator,
    ///     None,
    /// );
    /// assert!(node.is_some());
    /// let node = node.unwrap();
    /// assert_eq!(node.state(), &expected_state);
    /// assert_eq!(node.cost(&model), 1);
    /// assert_eq!(node.bound(&model), Some(1));
    /// assert!(!node.is_closed());
    /// assert_eq!(node.transitions(), vec![transition]);
    /// ```
    pub fn generate_successor_node<H, F>(
        &self,
        transition: Arc<V>,
        function_cache: &mut ParentAndChildStateFunctionCache,
        model: &Model,
        h_evaluator: H,
        f_evaluator: F,
        primal_bound: Option<T>,
    ) -> Option<Self>
    where
        H: FnOnce(
            &StateInRegistry<Arc<HashableSignatureVariables>>,
            &mut StateFunctionCache,
        ) -> Option<T>,
        F: FnOnce(T, T, &StateInRegistry<Arc<HashableSignatureVariables>>) -> T,
    {
        let (state, g) = model.generate_successor_state(
            &self.state,
            &mut function_cache.parent,
            self.g,
            transition.as_ref(),
            None,
        )?;
        function_cache.child.clear();
        let h = h_evaluator(&state, &mut function_cache.child)?;
        let f = f_evaluator(g, h, &state);

        if exceed_bound(model, f, primal_bound) {
            return None;
        }

        let (h, f) = if model.reduce_function == ReduceFunction::Max {
            (h, f)
        } else {
            (-h, -f)
        };

        Some(SendableFNode::new(
            state,
            g,
            h,
            f,
            Some(self),
            Some(transition),
        ))
    }

    /// Generates a successor node given a transition, h- and f- evaluators, and a primal bound on the solution cost,
    /// and inserts it into a state registry.
    ///
    /// `function_cache.parent` is not cleared and updated by this node while `function_cache.child` is cleared and updated by the successor node if generated.
    ///
    /// Returns the successor node and whether a new entry is generated or not.
    /// If the successor node dominates an existing non-closed node in the registry, the second return value is `false`.
    ///
    /// `h_evaluator` is a function that takes a state and returns the dual bound (the h-value).
    /// If `h_evaluator` returns `None`, the state is a dead-end, so the node is not generated.
    /// `f_evaluator` is a function that takes g- and h-values and the state and returns the f-value.
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
    /// use dypdl_heuristic_search::SendableFNode;
    /// use dypdl_heuristic_search::parallel_search_algorithm::ConcurrentStateRegistry;
    /// use dypdl_heuristic_search::search_algorithm::StateInRegistry;
    /// use dypdl_heuristic_search::search_algorithm::data_structure::{
    ///     GetTransitions, StateInformation, ParentAndChildStateFunctionCache,
    /// };
    /// use std::sync::Arc;
    ///
    /// let mut model = Model::default();
    /// let variable = model.add_integer_variable("variable", 0).unwrap();
    /// let registry = ConcurrentStateRegistry::new(Arc::new(model.clone()));
    ///
    /// let state = model.target.clone();
    /// let mut function_cache = ParentAndChildStateFunctionCache::new(&model.state_functions);
    /// let cost = 0;
    /// let h_evaluator = |_: &_, _: &mut _| Some(0);
    /// let f_evaluator = |g, h, _: &_| g + h;
    /// let node = SendableFNode::<_>::generate_root_node(
    ///     state, &mut function_cache.parent, cost, &model, &h_evaluator, &f_evaluator, None,
    /// ).unwrap();
    ///
    /// let mut transition = Transition::new("transition");
    /// transition.set_cost(IntegerExpression::Cost + 1);
    /// transition.add_effect(variable, variable + 1).unwrap();
    /// let mut function_cache_for_expected = StateFunctionCache::new(&model.state_functions);
    /// let expected_state: StateInRegistry<_> = transition.apply(
    ///     &model.target, &mut function_cache_for_expected, &model.state_functions, &model.table_registry,
    /// );
    ///
    /// let result = node.insert_successor_node(
    ///     Arc::new(transition.clone()),
    ///     &mut function_cache,
    ///     &registry,
    ///     &h_evaluator,
    ///     &f_evaluator, None,
    /// );
    /// assert!(result.is_some());
    /// let (node, generated) = result.unwrap();
    /// assert!(generated);
    /// assert_eq!(node.state(), &expected_state);
    /// assert_eq!(node.cost(&model), 1);
    /// assert_eq!(node.bound(&model), Some(1));
    /// assert!(!node.is_closed());
    /// assert_eq!(node.transitions(), vec![transition]);
    /// ```
    pub fn insert_successor_node<H, F>(
        &self,
        transition: Arc<V>,
        function_cache: &mut ParentAndChildStateFunctionCache,
        registry: &ConcurrentStateRegistry<T, Self>,
        h_evaluator: H,
        f_evaluator: F,
        primal_bound: Option<T>,
    ) -> Option<(Arc<Self>, bool)>
    where
        H: FnOnce(
            &StateInRegistry<Arc<HashableSignatureVariables>>,
            &mut StateFunctionCache,
        ) -> Option<T>,
        F: FnOnce(T, T, &StateInRegistry<Arc<HashableSignatureVariables>>) -> T,
    {
        let (state, g) = registry.model().generate_successor_state(
            &self.state,
            &mut function_cache.parent,
            self.g,
            transition.as_ref(),
            None,
        )?;

        let model = registry.model().clone();
        let maximize = model.reduce_function == ReduceFunction::Max;

        let constructor = |state, g, other: Option<&SendableFNode<T, V>>| {
            let h = if let Some(other) = other {
                if maximize {
                    other.h
                } else {
                    -other.h
                }
            } else {
                function_cache.child.clear();
                h_evaluator(&state, &mut function_cache.child)?
            };
            let f = f_evaluator(g, h, &state);

            if exceed_bound(&model, f, primal_bound) {
                return None;
            }

            let (h, f) = if maximize { (h, f) } else { (-h, -f) };

            Some(SendableFNode::new(
                state,
                g,
                h,
                f,
                Some(self),
                Some(transition),
            ))
        };

        let result = registry.insert_with(state, g, constructor);

        for d in result.dominated.iter() {
            if !d.is_closed() {
                d.close()
            }
        }

        let node = result.information?;

        Some((node, result.dominated.is_empty()))
    }
}

impl<T, V> Clone for SendableFNode<T, V>
where
    T: Numeric + PartialOrd,
    V: TransitionInterface + Clone,
    Transition: From<V>,
{
    fn clone(&self) -> Self {
        SendableFNode {
            state: self.state.clone(),
            g: self.g,
            h: self.h,
            f: self.f,
            closed: atomic::AtomicBool::new(self.closed.load(atomic::Ordering::Relaxed)),
            transitions: self.transitions.clone(),
        }
    }
}

impl<T, V> PartialEq for SendableFNode<T, V>
where
    T: Numeric + PartialOrd,
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

impl<T, V> Eq for SendableFNode<T, V>
where
    T: Numeric + Ord,
    V: TransitionInterface + Clone,
    Transition: From<V>,
{
}

impl<T, V> Ord for SendableFNode<T, V>
where
    T: Numeric + Ord,
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

impl<T, V> PartialOrd for SendableFNode<T, V>
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

impl<T, V> StateInformation<T, Arc<HashableSignatureVariables>> for SendableFNode<T, V>
where
    T: Numeric,
    V: TransitionInterface + Clone,
    Transition: From<V>,
{
    #[inline]
    fn state(&self) -> &StateInRegistry<Arc<HashableSignatureVariables>> {
        &self.state
    }

    #[inline]
    fn state_mut(&mut self) -> &mut StateInRegistry<Arc<HashableSignatureVariables>> {
        &mut self.state
    }

    #[inline]
    fn cost(&self, _: &Model) -> T {
        self.g
    }

    #[inline]
    fn bound(&self, model: &Model) -> Option<T> {
        if model.reduce_function == ReduceFunction::Min {
            Some(-self.f)
        } else {
            Some(self.f)
        }
    }

    #[inline]
    fn is_closed(&self) -> bool {
        self.closed.load(atomic::Ordering::Relaxed)
    }

    #[inline]
    fn close(&self) {
        self.closed.store(true, atomic::Ordering::Relaxed);
    }
}

impl<T, V> GetTransitions<V> for SendableFNode<T, V>
where
    T: Numeric + Ord,
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
            .and_then(|transitions| transitions.last())
    }
}

impl<T, V> BfsNode<T, V, Arc<HashableSignatureVariables>> for SendableFNode<T, V>
where
    T: Numeric + Ord + Display,
    V: TransitionInterface + Clone,
    Transition: From<V>,
{
    #[inline]
    fn ordered_by_bound() -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dypdl::{expression::*, prelude::*};
    use smallvec::SmallVec;

    #[test]
    fn ordered_by_bound() {
        assert!(SendableFNode::<Integer>::ordered_by_bound());
    }

    #[test]
    fn clone() {
        let mut model = dypdl::Model::default();
        model.set_minimize();
        let variable = model.add_integer_variable("variable", 0);
        assert!(variable.is_ok());
        let state = model.target.clone();
        let mut function_cache = StateFunctionCache::new(&model.state_functions);
        let h_evaluator = |_: &_, _: &mut _| Some(0);
        let f_evaluator = |g, h, _: &_| g + h;
        let node = SendableFNode::<_>::generate_root_node(
            state,
            &mut function_cache,
            1,
            &model,
            &h_evaluator,
            &f_evaluator,
            None,
        );
        assert!(node.is_some());
        let mut node = node.unwrap();
        let mut another = node.clone();
        assert_eq!(node.state(), another.state());
        assert_eq!(node.state_mut(), another.state_mut());
        assert_eq!(node.cost(&model), another.cost(&model));
        assert_eq!(node.bound(&model), another.bound(&model));
        assert_eq!(node.is_closed(), another.is_closed());
        assert_eq!(node.transitions(), another.transitions());
    }

    #[test]
    fn generate_root_node_some_min() {
        let mut model = dypdl::Model::default();
        model.set_minimize();
        let variable = model.add_integer_variable("variable", 0);
        assert!(variable.is_ok());
        let state = model.target.clone();
        let mut expected_state = StateInRegistry::from(state.clone());

        let mut function_cache = StateFunctionCache::new(&model.state_functions);
        let h_evaluator = |_: &_, _: &mut _| Some(0);
        let f_evaluator = |g, h, _: &_| g + h;
        let node = SendableFNode::<_>::generate_root_node(
            state,
            &mut function_cache,
            1,
            &model,
            &h_evaluator,
            &f_evaluator,
            None,
        );
        assert!(node.is_some());
        let mut node = node.unwrap();
        assert_eq!(node.state(), &expected_state);
        assert_eq!(node.state_mut(), &mut expected_state);
        assert_eq!(node.cost(&model), 1);
        assert_eq!(node.bound(&model), Some(1));
        assert!(!node.is_closed());
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

        let mut function_cache = StateFunctionCache::new(&model.state_functions);
        let h_evaluator = |_: &_, _: &mut _| Some(0);
        let f_evaluator = |g, h, _: &_| g + h;
        let node = SendableFNode::<_>::generate_root_node(
            state,
            &mut function_cache,
            1,
            &model,
            &h_evaluator,
            &f_evaluator,
            None,
        );
        assert!(node.is_some());
        let mut node = node.unwrap();
        assert_eq!(node.state(), &expected_state);
        assert_eq!(node.state_mut(), &mut expected_state);
        assert_eq!(node.cost(&model), 1);
        assert_eq!(node.bound(&model), Some(1));
        assert!(!node.is_closed());
        assert_eq!(node.transitions(), vec![]);
    }

    #[test]
    fn generate_root_node_pruned_by_bound_min() {
        let mut model = dypdl::Model::default();
        model.set_minimize();
        let variable = model.add_integer_variable("variable", 0);
        assert!(variable.is_ok());
        let state = model.target.clone();
        let mut function_cache = StateFunctionCache::new(&model.state_functions);
        let h_evaluator = |_: &_, _: &mut _| Some(1);
        let f_evaluator = |g, h, _: &_| g + h;
        let node = SendableFNode::<_>::generate_root_node(
            state,
            &mut function_cache,
            0,
            &model,
            &h_evaluator,
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
        let mut function_cache = StateFunctionCache::new(&model.state_functions);
        let h_evaluator = |_: &_, _: &mut _| Some(1);
        let f_evaluator = |g, h, _: &_| g + h;
        let node = SendableFNode::<_>::generate_root_node(
            state,
            &mut function_cache,
            0,
            &model,
            &h_evaluator,
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
        let mut function_cache = StateFunctionCache::new(&model.state_functions);
        let h_evaluator = |_: &_, _: &mut _| None;
        let f_evaluator = |g, h, _: &_| g + h;
        let node = SendableFNode::<_>::generate_root_node(
            state,
            &mut function_cache,
            0,
            &model,
            &h_evaluator,
            &f_evaluator,
            None,
        );
        assert!(node.is_none());
    }

    #[test]
    fn close() {
        let model = dypdl::Model::default();
        let state = model.target.clone();
        let mut function_cache = StateFunctionCache::new(&model.state_functions);
        let h_evaluator = |_: &_, _: &mut _| Some(0);
        let f_evaluator = |g, h, _: &_| g + h;
        let node = SendableFNode::<_>::generate_root_node(
            state,
            &mut function_cache,
            0,
            &model,
            &h_evaluator,
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
        let mut function_cache = StateFunctionCache::new(&model.state_functions);
        let mut expected_state: StateInRegistry<_> = transition.apply(
            &state,
            &mut function_cache,
            &model.state_functions,
            &model.table_registry,
        );

        let mut function_cache =
            ParentAndChildStateFunctionCache::new(&model.state_functions);
        let h_evaluator = |_: &_, _: &mut _| Some(0);
        let f_evaluator = |g, h, _: &_| g + h;
        let node = SendableFNode::generate_root_node(
            state,
            &mut function_cache.parent,
            0,
            &model,
            &h_evaluator,
            &f_evaluator,
            None,
        );
        assert!(node.is_some());
        let node = node.unwrap();

        let successor = node.generate_successor_node(
            Arc::new(transition.clone()),
            &mut function_cache,
            &model,
            &h_evaluator,
            &f_evaluator,
            None,
        );
        assert!(successor.is_some());
        let mut successor = successor.unwrap();
        assert_eq!(successor.state(), &expected_state);
        assert_eq!(successor.state_mut(), &mut expected_state);
        assert_eq!(successor.cost(&model), 1);
        assert_eq!(successor.bound(&model), Some(1));
        assert!(!successor.is_closed());
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
        let mut function_cache = StateFunctionCache::new(&model.state_functions);
        let mut expected_state: StateInRegistry<_> = transition.apply(
            &state,
            &mut function_cache,
            &model.state_functions,
            &model.table_registry,
        );

        let mut function_cache =
            ParentAndChildStateFunctionCache::new(&model.state_functions);
        let h_evaluator = |_: &_, _: &mut _| Some(0);
        let f_evaluator = |g, h, _: &_| g + h;
        let node = SendableFNode::generate_root_node(
            state,
            &mut function_cache.parent,
            0,
            &model,
            &h_evaluator,
            &f_evaluator,
            None,
        );
        assert!(node.is_some());
        let node = node.unwrap();

        let successor = node.generate_successor_node(
            Arc::new(transition.clone()),
            &mut function_cache,
            &model,
            &h_evaluator,
            &f_evaluator,
            None,
        );
        assert!(successor.is_some());
        let mut successor = successor.unwrap();
        assert_eq!(successor.state(), &expected_state);
        assert_eq!(successor.state_mut(), &mut expected_state);
        assert_eq!(successor.cost(&model), 1);
        assert_eq!(successor.bound(&model), Some(1));
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
        let mut function_cache =
            ParentAndChildStateFunctionCache::new(&model.state_functions);
        let h_evaluator = |_: &_, _: &mut _| Some(0);
        let f_evaluator = |g, h, _: &_| g + h;
        let node = SendableFNode::generate_root_node(
            state,
            &mut function_cache.parent,
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
            &mut function_cache,
            &model,
            &h_evaluator,
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
        let mut function_cache =
            ParentAndChildStateFunctionCache::new(&model.state_functions);
        let h_evaluator = |_: &_, _: &mut _| Some(0);
        let f_evaluator = |g, h, _: &_| g + h;
        let node = SendableFNode::generate_root_node(
            state,
            &mut function_cache.parent,
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
            &mut function_cache,
            &model,
            &h_evaluator,
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
        let mut function_cache =
            ParentAndChildStateFunctionCache::new(&model.state_functions);
        let h_evaluator = |_: &_, _: &mut _| Some(0);
        let f_evaluator = |g, h, _: &_| g + h;
        let node = SendableFNode::generate_root_node(
            state,
            &mut function_cache.parent,
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
            &mut function_cache,
            &model,
            &h_evaluator,
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
        let mut function_cache =
            ParentAndChildStateFunctionCache::new(&model.state_functions);
        let h_evaluator = |_: &_, _: &mut _| Some(0);
        let f_evaluator = |g, h, _: &_| g + h;
        let node = SendableFNode::generate_root_node(
            state,
            &mut function_cache.parent,
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

        let h_evaluator = |_: &_, _: &mut _| None;
        let result = node.generate_successor_node(
            Arc::new(transition),
            &mut function_cache,
            &model,
            &h_evaluator,
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
        let model = Arc::new(model);

        let state = StateInRegistry::from(model.target.clone());
        let registry = ConcurrentStateRegistry::<_, SendableFNode<_>>::new(model.clone());

        let mut transition = Transition::default();
        let result = transition.add_effect(v1, v1 + 1);
        assert!(result.is_ok());
        let result = transition.add_effect(v2, v2 + 1);
        assert!(result.is_ok());
        transition.set_cost(IntegerExpression::Cost + 1);

        let mut function_cache = StateFunctionCache::new(&model.state_functions);
        let expected_state: StateInRegistry<_> = transition.apply(
            &state,
            &mut function_cache,
            &model.state_functions,
            &model.table_registry,
        );

        let mut function_cache =
            ParentAndChildStateFunctionCache::new(&model.state_functions);
        let h_evaluator = |_: &_, _: &mut _| Some(0);
        let f_evaluator = |g, h, _: &_| g + h;
        let node = SendableFNode::generate_root_node(
            state,
            &mut function_cache.parent,
            0,
            &model,
            &h_evaluator,
            &f_evaluator,
            None,
        );
        assert!(node.is_some());
        let node = node.unwrap();
        let result = registry.insert(node);
        assert!(result.information.is_some());
        let node = result.information.unwrap();

        let result = node.insert_successor_node(
            Arc::new(transition.clone()),
            &mut function_cache,
            &registry,
            &h_evaluator,
            &f_evaluator,
            None,
        );
        assert!(result.is_some());
        let (successor, generated) = result.unwrap();
        assert_eq!(successor.state(), &expected_state);
        assert_eq!(successor.cost(&model), 1);
        assert_eq!(successor.bound(&model), Some(1));
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
        let model = Arc::new(model);

        let state = StateInRegistry::from(model.target.clone());
        let registry = ConcurrentStateRegistry::<_, SendableFNode<_>>::new(model.clone());

        let mut transition = Transition::default();
        let result = transition.add_effect(v1, v1 + 1);
        assert!(result.is_ok());
        let result = transition.add_effect(v2, v2 + 1);
        assert!(result.is_ok());
        transition.set_cost(IntegerExpression::Cost + 1);

        let mut function_cache = StateFunctionCache::new(&model.state_functions);
        let expected_state: StateInRegistry<_> = transition.apply(
            &state,
            &mut function_cache,
            &model.state_functions,
            &model.table_registry,
        );

        let mut function_cache =
            ParentAndChildStateFunctionCache::new(&model.state_functions);
        let h_evaluator = |_: &_, _: &mut _| Some(0);
        let f_evaluator = |g, h, _: &_| g + h;
        let node = SendableFNode::generate_root_node(
            state,
            &mut function_cache.parent,
            0,
            &model,
            &h_evaluator,
            &f_evaluator,
            None,
        );
        assert!(node.is_some());
        let node = node.unwrap();
        let result = registry.insert(node);
        assert!(result.information.is_some());
        let node = result.information.unwrap();

        let result = node.insert_successor_node(
            Arc::new(transition.clone()),
            &mut function_cache,
            &registry,
            &h_evaluator,
            &f_evaluator,
            None,
        );
        assert!(result.is_some());
        let (successor, generated) = result.unwrap();
        assert_eq!(successor.state(), &expected_state);
        assert_eq!(successor.cost(&model), 1);
        assert_eq!(successor.bound(&model), Some(1));
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
        let mut function_cache =
            ParentAndChildStateFunctionCache::new(&model.state_functions);
        let registry = ConcurrentStateRegistry::<_, SendableFNode<_>>::new(Arc::new(model.clone()));

        let h_evaluator = |_: &_, _: &mut _| Some(0);
        let f_evaluator = |g, h, _: &_| g + h;
        let node = SendableFNode::generate_root_node(
            state,
            &mut function_cache.parent,
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

        let result = node.insert_successor_node(
            Arc::new(transition),
            &mut function_cache,
            &registry,
            &h_evaluator,
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
        let mut function_cache = StateFunctionCache::new(&model.state_functions);
        let expected_state: StateInRegistry<_> = transition.apply(
            &state,
            &mut function_cache,
            &model.state_functions,
            &model.table_registry,
        );

        let mut function_cache =
            ParentAndChildStateFunctionCache::new(&model.state_functions);
        let h_evaluator = |_: &_, _: &mut _| Some(0);
        let f_evaluator = |g, h, _: &_| g + h;
        let node = SendableFNode::generate_root_node(
            state,
            &mut function_cache.parent,
            0,
            &model,
            &h_evaluator,
            &f_evaluator,
            None,
        );
        assert!(node.is_some());
        let node = node.unwrap();
        let registry = ConcurrentStateRegistry::<_, SendableFNode<_>>::new(Arc::new(model.clone()));
        let result = registry.insert(node);
        assert!(result.information.is_some());
        let node = result.information.unwrap();
        let dominated = result.dominated;
        assert_eq!(dominated, SmallVec::<[_; 1]>::new());

        let result = node.insert_successor_node(
            Arc::new(transition.clone()),
            &mut function_cache,
            &registry,
            &h_evaluator,
            &f_evaluator,
            None,
        );
        assert!(result.is_some());
        let (successor, generated) = result.unwrap();
        assert_eq!(successor.state(), &expected_state);
        assert_eq!(successor.cost(&model), 0);
        assert_eq!(successor.bound(&model), Some(0));
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
        let mut function_cache = StateFunctionCache::new(&model.state_functions);
        let expected_state: StateInRegistry<_> = transition.apply(
            &state,
            &mut function_cache,
            &model.state_functions,
            &model.table_registry,
        );

        let mut function_cache =
            ParentAndChildStateFunctionCache::new(&model.state_functions);
        let h_evaluator = |_: &_, _: &mut _| Some(0);
        let f_evaluator = |g, h, _: &_| g + h;
        let node = SendableFNode::generate_root_node(
            state,
            &mut function_cache.parent,
            0,
            &model,
            &h_evaluator,
            &f_evaluator,
            None,
        );
        assert!(node.is_some());
        let node = node.unwrap();
        let registry = ConcurrentStateRegistry::<_, SendableFNode<_>>::new(Arc::new(model.clone()));
        let result = registry.insert(node);
        assert!(result.information.is_some());
        let node = result.information.unwrap();
        let dominated = result.dominated;
        assert_eq!(dominated, SmallVec::<[_; 1]>::new());

        let result = node.insert_successor_node(
            Arc::new(transition.clone()),
            &mut function_cache,
            &registry,
            &h_evaluator,
            &f_evaluator,
            None,
        );
        assert!(result.is_some());
        let (successor, generated) = result.unwrap();
        assert_eq!(successor.state(), &expected_state);
        assert_eq!(successor.cost(&model), 1);
        assert_eq!(successor.bound(&model), Some(1));
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
        let mut function_cache =
            ParentAndChildStateFunctionCache::new(&model.state_functions);
        let h_evaluator = |_: &_, _: &mut _| Some(0);
        let f_evaluator = |g, h, _: &_| g + h;
        let node = SendableFNode::generate_root_node(
            state,
            &mut function_cache.parent,
            0,
            &model,
            &h_evaluator,
            &f_evaluator,
            None,
        );
        assert!(node.is_some());
        let node = node.unwrap();
        let registry = ConcurrentStateRegistry::<_, SendableFNode<_>>::new(Arc::new(model.clone()));
        let result = registry.insert(node);
        assert!(result.information.is_some());
        let node = result.information.unwrap();
        let dominated = result.dominated;
        assert_eq!(dominated, SmallVec::<[_; 1]>::new());

        let result = node.insert_successor_node(
            Arc::new(transition.clone()),
            &mut function_cache,
            &registry,
            &h_evaluator,
            &f_evaluator,
            None,
        );
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
        let mut function_cache =
            ParentAndChildStateFunctionCache::new(&model.state_functions);
        let h_evaluator = |_: &_, _: &mut _| Some(0);
        let f_evaluator = |g, h, _: &_| g + h;
        let node = SendableFNode::generate_root_node(
            state,
            &mut function_cache.parent,
            0,
            &model,
            &h_evaluator,
            &f_evaluator,
            None,
        );
        assert!(node.is_some());
        let node = node.unwrap();
        let registry = ConcurrentStateRegistry::<_, SendableFNode<_>>::new(Arc::new(model.clone()));
        let result = registry.insert(node);
        assert!(result.information.is_some());
        let node = result.information.unwrap();
        let dominated = result.dominated;
        assert_eq!(dominated, SmallVec::<[_; 1]>::new());

        let result = node.insert_successor_node(
            Arc::new(transition.clone()),
            &mut function_cache,
            &registry,
            &h_evaluator,
            &f_evaluator,
            None,
        );
        assert!(result.is_none());
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
        let mut function_cache =
            ParentAndChildStateFunctionCache::new(&model.state_functions);
        let h_evaluator = |_: &_, _: &mut _| Some(0);
        let f_evaluator = |g, h, _: &_| g + h;
        let node = SendableFNode::generate_root_node(
            state,
            &mut function_cache.parent,
            0,
            &model,
            &h_evaluator,
            &f_evaluator,
            None,
        );
        assert!(node.is_some());
        let node = node.unwrap();
        let registry = ConcurrentStateRegistry::<_, SendableFNode<_>>::new(Arc::new(model.clone()));
        let result = registry.insert(node);
        assert!(result.information.is_some());
        let node = result.information.unwrap();
        let dominated = result.dominated;
        assert_eq!(dominated, SmallVec::<[_; 1]>::new());

        let result = node.insert_successor_node(
            Arc::new(transition.clone()),
            &mut function_cache,
            &registry,
            &h_evaluator,
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
        let h_evaluator = |_: &_, _: &mut _| Some(0);
        let f_evaluator = |g, h, _: &_| g + h;
        let mut function_cache =
            ParentAndChildStateFunctionCache::new(&model.state_functions);
        let node = SendableFNode::generate_root_node(
            state,
            &mut function_cache.parent,
            0,
            &model,
            &h_evaluator,
            &f_evaluator,
            None,
        );
        assert!(node.is_some());
        let node = node.unwrap();
        let registry = ConcurrentStateRegistry::<_, SendableFNode<_>>::new(Arc::new(model.clone()));
        let result = registry.insert(node);
        assert!(result.information.is_some());
        let node = result.information.unwrap();
        let dominated = result.dominated;
        assert_eq!(dominated, SmallVec::<[_; 1]>::new());

        let result = node.insert_successor_node(
            Arc::new(transition.clone()),
            &mut function_cache,
            &registry,
            &h_evaluator,
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
        let registry = ConcurrentStateRegistry::<_, SendableFNode<_>>::new(Arc::new(model.clone()));

        let h_evaluator = |_: &_, _: &mut _| Some(0);
        let f_evaluator = |g, h, _: &_| g + h;
        let mut function_cache =
            ParentAndChildStateFunctionCache::new(&model.state_functions);
        let node = SendableFNode::generate_root_node(
            state,
            &mut function_cache.parent,
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

        let h_evaluator = |_: &_, _: &mut _| None;
        let result = node.insert_successor_node(
            Arc::new(transition),
            &mut function_cache,
            &registry,
            &h_evaluator,
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
        let model = Arc::new(model);

        let state = model.target.clone();
        let mut function_cache =
            ParentAndChildStateFunctionCache::new(&model.state_functions);
        let h_evaluator_0 = |_: &_, _: &mut _| Some(0);
        let f_evaluator = |g, h, _: &_| g + h;
        let node1 = SendableFNode::<_>::generate_root_node(
            state.clone(),
            &mut function_cache.parent,
            0,
            &model,
            &h_evaluator_0,
            &f_evaluator,
            None,
        );
        assert!(node1.is_some());
        let node1 = Arc::new(node1.unwrap());

        let mut transition = Transition::default();
        let result = transition.add_effect(v1, v1 + 1);
        assert!(result.is_ok());
        let result = transition.add_effect(v2, v2 + 1);
        assert!(result.is_ok());
        let node2 = node1.generate_successor_node(
            Arc::new(transition),
            &mut function_cache,
            &model,
            &h_evaluator_0,
            &f_evaluator,
            None,
        );
        assert!(node2.is_some());
        let node2 = Arc::new(node2.unwrap());

        let mut transition = Transition::default();
        transition.set_cost(IntegerExpression::Cost + 1);
        let registry = ConcurrentStateRegistry::<_, SendableFNode<_>>::new(model.clone());
        let node3 = node1.insert_successor_node(
            Arc::new(transition),
            &mut function_cache,
            &registry,
            &h_evaluator_0,
            &f_evaluator,
            None,
        );
        assert!(node3.is_some());
        let (node3, _) = node3.unwrap();

        let h_evaluator_1 = |_: &_, _: &mut _| Some(1);
        function_cache.parent.clear();
        let node4 = SendableFNode::<_>::generate_root_node(
            state,
            &mut function_cache.parent,
            0,
            &model,
            &h_evaluator_1,
            &f_evaluator,
            None,
        );
        assert!(node4.is_some());
        let node4 = Arc::new(node4.unwrap());

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
        let model = Arc::new(model);

        let state = model.target.clone();
        let mut function_cache =
            ParentAndChildStateFunctionCache::new(&model.state_functions);
        let h_evaluator_0 = |_: &_, _: &mut _| Some(0);
        let f_evaluator = |g, h, _: &_| g + h;
        let node1 = SendableFNode::<_>::generate_root_node(
            state.clone(),
            &mut function_cache.parent,
            0,
            &model,
            &h_evaluator_0,
            &f_evaluator,
            None,
        );
        assert!(node1.is_some());
        let node1 = Arc::new(node1.unwrap());

        let mut transition = Transition::default();
        let result = transition.add_effect(v1, v1 + 1);
        assert!(result.is_ok());
        let result = transition.add_effect(v2, v2 + 1);
        assert!(result.is_ok());
        let node2 = node1.generate_successor_node(
            Arc::new(transition),
            &mut function_cache,
            &model,
            &h_evaluator_0,
            &f_evaluator,
            None,
        );
        assert!(node2.is_some());
        let node2 = Arc::new(node2.unwrap());

        let mut transition = Transition::default();
        transition.set_cost(IntegerExpression::Cost + 1);
        let registry = ConcurrentStateRegistry::<_, SendableFNode<_>>::new(model.clone());
        let node3 = node1.insert_successor_node(
            Arc::new(transition),
            &mut function_cache,
            &registry,
            &h_evaluator_0,
            &f_evaluator,
            None,
        );
        assert!(node3.is_some());
        let (node3, _) = node3.unwrap();

        let h_evaluator_1 = |_: &_, _: &mut _| Some(1);
        function_cache.parent.clear();
        let node4 = SendableFNode::<_>::generate_root_node(
            state,
            &mut function_cache.parent,
            0,
            &model,
            &h_evaluator_1,
            &f_evaluator,
            None,
        );
        assert!(node4.is_some());
        let node4 = Arc::new(node4.unwrap());

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
