use crate::solver;
use crate::successor_generator::SuccessorGenerator;
use dypdl::variable_type;

/// Common parameters for forward search solver.
pub struct ForwardSearchParameters<'a, T: variable_type::Numeric> {
    /// Successor generator.
    pub generator: SuccessorGenerator<'a, dypdl::Transition>,
    /// Common parameters.
    pub parameters: solver::SolverParameters<T>,
    /// Initial registry capacity.
    pub initial_registry_capacity: Option<usize>,
}
