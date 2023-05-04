use super::hashable_state::HashableSignatureVariables;
use super::state_registry::{StateInRegistry, StateInformation, StateRegistry};
use dypdl::variable_type::Numeric;
use std::rc::Rc;

/// Trait representing a node used in best-first search.
///
/// `h_evaluator` returns the h-value of a state, which is a dual bound for the state.
/// If the h-value is `None`, the state is pruned.
///
/// `f_evaluator` takes the g-value and h-value and returns the f-value, which should be the dual bound
/// for a solution that passes through the state.
pub trait BfsNodeInterface<T: Numeric>: Sized + StateInformation<T> + Ord {
    /// Returns the initial node and its h- and f-values.
    /// Returns `None` if the initial state is pruned.
    fn generate_initial_node<H, F>(
        registry: &mut StateRegistry<T, Self>,
        h_evaluator: H,
        f_evaluator: F,
    ) -> Option<(Rc<Self>, T, T)>
    where
        H: FnOnce(&StateInRegistry<Rc<HashableSignatureVariables>>) -> Option<T>,
        F: FnOnce(T, T, &StateInRegistry<Rc<HashableSignatureVariables>>) -> T;

    /// Returns a successor node and its h- and f-values if it it is not dominated by existing node and its f-value does not exceed the given primal bound.
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
        H: FnOnce(&StateInRegistry<Rc<HashableSignatureVariables>>) -> Option<T>,
        F: FnOnce(T, T, &StateInRegistry<Rc<HashableSignatureVariables>>) -> T;

    /// Returns if the node is closed.
    fn closed(&self) -> bool;

    /// Close the node.
    fn close(&self);

    /// Returns the dual bound of a solution through this node.
    fn get_bound(&self, model: &dypdl::Model) -> T;

    /// Returns transitions to reach this node.
    fn transitions(&self) -> Vec<dypdl::Transition>;
}
