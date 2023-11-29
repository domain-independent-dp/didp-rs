use super::super::arc_chain::ArcChain;
use super::search_node_message::SearchNodeMessage;
use crate::search_algorithm::data_structure::{exceed_bound, StateWithHashableSignatureVariables};
use dypdl::variable_type::Numeric;
use dypdl::{Model, ReduceFunction, Transition, TransitionInterface};
use rustc_hash::FxHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// Node ordered by the f-value to be sent to another thread via message passing.
#[derive(Debug, Clone, PartialEq)]
pub struct FNodeMessage<T, V = Transition>
where
    T: Numeric,
    V: TransitionInterface + Clone,
    Transition: From<V>,
{
    /// State.
    pub state: StateWithHashableSignatureVariables,
    /// g-value.
    pub g: T,
    /// h-value.
    pub h: T,
    /// f-value.
    pub f: T,
    /// Transitions applied before.
    pub transitions: Option<Arc<ArcChain<V>>>,
}

impl<T, V> FNodeMessage<T, V>
where
    T: Numeric + Send + Sync,
    V: TransitionInterface + Clone + Send + Sync,
    Transition: From<V>,
{
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
    /// use dypdl_heuristic_search::{DistributedFNode, FNodeMessage};
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
    /// let h_evaluator = |_: &_| Some(0);
    /// let f_evaluator = |g, h, _: &_| g + h;
    /// let node = FNodeMessage::<_>::generate_root_node(
    ///     state, cost, &model, &h_evaluator, &f_evaluator, None,
    /// );
    /// assert!(node.is_some());
    /// let node = DistributedFNode::from(node.unwrap());
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
        let h = h_evaluator(&state)?;
        let f = f_evaluator(cost, h, &state);

        if exceed_bound(model, f, primal_bound) {
            return None;
        }

        let (h, f) = if model.reduce_function == ReduceFunction::Max {
            (h, f)
        } else {
            (-h, -f)
        };

        Some(Self {
            state,
            g: cost,
            h,
            f,
            transitions: None,
        })
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
        self.state.signature_variables.hash(&mut hasher);
        hasher.finish() as usize % threads
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_root_node_some_min() {
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
        assert_eq!(node.state, expected_state);
        assert_eq!(node.g, 1);
        assert_eq!(node.h, 0);
        assert_eq!(node.f, -1);
        assert_eq!(node.transitions, None);
    }

    #[test]
    fn generate_root_node_some_max() {
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
        assert_eq!(node.state, expected_state);
        assert_eq!(node.g, 1);
        assert_eq!(node.h, 0);
        assert_eq!(node.f, 1);
        assert_eq!(node.transitions, None);
    }

    #[test]
    fn generate_root_node_pruned_by_bound_min() {
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
    fn generate_root_node_pruned_by_bound_max() {
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
    fn generate_root_node_pruned_by_h() {
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
}
