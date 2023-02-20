use super::hashable_state::HashableSignatureVariables;
use super::state_registry::{StateInRegistry, StateInformation, StateRegistry};
use dypdl::variable_type::Numeric;
use std::rc::Rc;

/// Trait representing a node used in best-first search.
pub trait BfsNodeInterface<T: Numeric>: Sized + StateInformation<T> + Ord {
    /// Returns the initial node.
    fn generate_initial_node<H, F>(
        registry: &mut StateRegistry<T, Self>,
        h_evaluator: H,
        f_evaluator: F,
    ) -> Option<(Rc<Self>, T, T)>
    where
        H: Fn(&StateInRegistry<Rc<HashableSignatureVariables>>, &dypdl::Model) -> Option<T>,
        F: Fn(T, T, &StateInRegistry<Rc<HashableSignatureVariables>>, &dypdl::Model) -> T;

    /// Returns a successor node and its h- and f-values if it it is not dominated and its f-value does not exceed the given primal bound.
    /// The last value returned indicates if a new search node is generated without dominating another open node.
    fn generate_successor<H, F>(
        &self,
        transition: Rc<dypdl::Transition>,
        registry: &mut StateRegistry<T, Self>,
        h_evaluator: H,
        f_evaluator: F,
        primal_bound: Option<T>,
    ) -> Option<(Rc<Self>, T, T, bool)>
    where
        H: Fn(&StateInRegistry<Rc<HashableSignatureVariables>>, &dypdl::Model) -> Option<T>,
        F: Fn(T, T, &StateInRegistry<Rc<HashableSignatureVariables>>, &dypdl::Model) -> T;

    /// Returns if the node is closed.
    fn closed(&self) -> bool;

    /// Close the node.
    fn close(&self);

    /// Returns the dual bound of a solution through this node.
    fn get_bound(&self, model: &dypdl::Model) -> T;

    /// Returns transitions to reach this node.
    fn transitions(&self) -> Vec<dypdl::Transition>;
}
