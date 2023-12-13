use super::super::state_registry::StateInformation;
use super::super::transition_id_chain::GetTransitionIds;
use crate::search_algorithm::data_structure::HashableSignatureVariables;
use dypdl::variable_type::Numeric;
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::rc::Rc;

/// Trait for best-first search node where transitions are represented by their ids.
pub trait BfsNodeWithTransitionIds<T, K = Rc<HashableSignatureVariables>>:
    Ord + StateInformation<T, K> + GetTransitionIds
where
    T: Numeric + Display,
    K: Hash + Eq + Clone + Debug,
{
    /// Returns whether nodes are ordered by their dual bounds.
    ///
    /// A dual bound of the node is the dual bound on the path cost from the target state
    /// to a base state via the node.
    fn ordered_by_bound() -> bool;
}
