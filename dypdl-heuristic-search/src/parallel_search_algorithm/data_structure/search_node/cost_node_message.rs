use super::super::arc_chain::ArcChain;
use super::search_node_message::SearchNodeMessage;
use crate::search_algorithm::data_structure::StateWithHashableSignatureVariables;
use dypdl::variable_type::Numeric;
use dypdl::{Model, ReduceFunction, Transition, TransitionInterface};
use rustc_hash::FxHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// Node ordered by the cost to be sent to another thread via message passing.
#[derive(Debug, Clone, PartialEq)]
pub struct CostNodeMessage<T, V = Transition>
where
    T: Numeric,
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

impl<T, V> CostNodeMessage<T, V>
where
    T: Numeric + Send + Sync,
    V: TransitionInterface + Clone + Send + Sync,
    Transition: From<V>,
{
    /// Generates a root search node given a state, its cost, and a DyPDL model.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    /// use dypdl_heuristic_search::{CostNodeMessage, DistributedCostNode};
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
    /// let node = CostNodeMessage::<_>::generate_root_node(state, cost, &model);
    /// let node = DistributedCostNode::from(node);
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

        let priority = if model.reduce_function == ReduceFunction::Max {
            cost
        } else {
            -cost
        };

        Self {
            state,
            priority,
            transitions: None,
        }
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
    use super::*;

    #[test]
    fn generate_root_node_min() {
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
    fn generate_root_node_max() {
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
}
