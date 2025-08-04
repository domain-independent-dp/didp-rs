use super::super::state_registry::StateInformation;
use super::super::transition_chain::GetTransitions;
use crate::search_algorithm::data_structure::{HashableSignatureVariables, TransitionWithId};
use dypdl::variable_type::Numeric;
use dypdl::TransitionInterface;
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::rc::Rc;

/// Trait for best-first search node.
pub trait BfsNode<T, V = TransitionWithId, K = Rc<HashableSignatureVariables>>:
    Ord + StateInformation<T, K> + GetTransitions<V>
where
    T: Numeric + Display,
    K: Hash + Eq + Clone + Debug,
    V: TransitionInterface + Clone,
{
    /// Returns whether nodes are ordered by their dual bounds.
    ///
    /// A dual bound of the node is the dual bound on the path cost from the target state
    /// to a base state via the node.
    fn ordered_by_bound() -> bool;
}
