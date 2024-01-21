use super::data_structure::SuccessorGenerator;
use super::search::Solution;
use dypdl::{
    variable_type::Numeric, Model, State, StateInterface, Transition, TransitionInterface,
};
use std::{marker, ops::Deref, rc::Rc};

/// Input of a heuristic search solver.
#[derive(Debug, PartialEq, Clone)]
pub struct NeighborhoodSearchInput<
    T,
    N,
    G,
    S = State,
    V = Transition,
    D = Rc<V>,
    R = Rc<dypdl::Model>,
> where
    T: Numeric,
    G: FnMut(S, T) -> Option<N>,
    S: StateInterface,
    V: TransitionInterface,
    D: Deref<Target = V> + Clone,
    R: Deref<Target = Model>,
{
    /// Cost of the root node.
    pub root_cost: T,
    /// Function to generate a node given a state and its cost.
    pub node_generator: G,
    /// Successor generator.
    pub successor_generator: SuccessorGenerator<V, D, R>,
    /// Initial feasible solution.
    pub solution: Solution<T, V>,
    /// Phantom data.
    pub phantom: marker::PhantomData<S>,
}
