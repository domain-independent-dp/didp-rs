use super::state_registry::StateInRegistry;
use super::successor_generator::SuccessorGenerator;
use super::transition_with_custom_cost::TransitionWithCustomCost;
use dypdl::variable_type::Numeric;
use dypdl::{Model, StateInterface, TransitionInterface};
use std::ops::Deref;
use std::rc::Rc;

/// Input problem instance for beam search.
#[derive(Debug, PartialEq, Clone)]
pub struct BeamSearchProblemInstance<
    'a,
    T,
    U,
    S = StateInRegistry,
    V = TransitionWithCustomCost,
    D = Rc<V>,
    R = Rc<Model>,
> where
    T: Numeric,
    U: Numeric,
    S: StateInterface,
    V: TransitionInterface,
    D: Deref<Target = V> + Clone,
    R: Deref<Target = Model>,
{
    /// Target state.
    pub target: S,
    /// Successor generator.
    pub generator: SuccessorGenerator<V, D, R>,
    /// Cost.
    pub cost: T,
    /// g-value.
    pub g: U,
    /// Suffix of the solution.
    pub solution_suffix: &'a [V],
}
