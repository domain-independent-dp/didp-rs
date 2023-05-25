use super::data_structure::SuccessorGenerator;
use dypdl::{variable_type::Numeric, Model, Transition, TransitionInterface};
use std::error::Error;
use std::{ops::Deref, rc::Rc};

/// Input of a heuristic search solver.
#[derive(Debug, PartialEq, Clone)]
pub struct SearchInput<'a, N, V = Transition, D = Rc<V>, R = Rc<dypdl::Model>>
where
    V: TransitionInterface,
    D: Deref<Target = V> + Clone,
    R: Deref<Target = Model>,
{
    /// Root node.
    pub node: Option<N>,
    /// Successor generator.
    pub generator: SuccessorGenerator<V, D, R>,
    /// Suffix of the solution.
    pub solution_suffix: &'a [V],
}

/// Common parameters for heuristic search solvers.
#[derive(Debug, PartialEq, Clone, Copy, Default)]
pub struct Parameters<T> {
    /// Primal bound.
    pub primal_bound: Option<T>,
    /// Time limit.
    pub time_limit: Option<f64>,
    /// Returns all feasible solutions found.
    pub get_all_solutions: bool,
    /// Initial capacity of a data structure saving all states.
    pub initial_registry_capacity: Option<usize>,
    /// Suppress log output or not.
    pub quiet: bool,
}

/// Information about a solution.
#[derive(Debug, Default, PartialEq, Clone)]
pub struct Solution<T: Numeric, V = Transition> {
    /// Solution cost.
    /// `None` if the model is infeasible.
    pub cost: Option<T>,
    /// Best dual bound.
    pub best_bound: Option<T>,
    /// Solved to optimality or not.
    pub is_optimal: bool,
    /// Infeasible model or not.
    pub is_infeasible: bool,
    /// Transitions corresponding to the solution.
    pub transitions: Vec<V>,
    /// Number of expanded nodes.
    pub expanded: usize,
    /// Number of generated nodes.
    pub generated: usize,
    /// Elapsed time in seconds.
    pub time: f64,
    /// Whether to exceed the time limit
    pub time_out: bool,
}

impl<T: Numeric, V> Solution<T, V> {
    /// Returns whether the solution would never be improved.
    pub fn is_terminated(&self) -> bool {
        self.is_optimal || self.is_infeasible || self.time_out
    }
}

/// Trait representing an anytime search algorithm.
pub trait Search<T: Numeric> {
    /// Searches for the best solution.
    ///
    /// # Errors
    /// If an error occurs during the search.
    fn search(&mut self) -> Result<Solution<T>, Box<dyn Error>> {
        loop {
            let (solution, terminated) = self.search_next()?;

            if terminated {
                return Ok(solution);
            }
        }
    }

    /// Searches for the next solution.
    ///
    /// # Errors
    /// If an error occurs during the search.
    fn search_next(&mut self) -> Result<(Solution<T>, bool), Box<dyn Error>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use dypdl::Integer;

    struct MockSearch {
        counter: Integer,
    }

    impl Search<Integer> for MockSearch {
        fn search_next(&mut self) -> Result<(Solution<Integer>, bool), Box<dyn Error>> {
            if self.counter > 0 {
                self.counter -= 1;
                Ok((
                    Solution {
                        cost: Some(self.counter),
                        ..Default::default()
                    },
                    false,
                ))
            } else {
                Ok((
                    Solution {
                        cost: Some(0),
                        ..Default::default()
                    },
                    true,
                ))
            }
        }
    }

    #[test]
    fn solution_is_terminated_optimal() {
        let solution: Solution<Integer> = Solution {
            is_optimal: true,
            ..Default::default()
        };
        assert!(solution.is_terminated());
    }

    #[test]
    fn solution_is_terminated_infeasible() {
        let solution: Solution<Integer> = Solution {
            is_infeasible: true,
            ..Default::default()
        };
        assert!(solution.is_terminated());
    }

    #[test]
    fn solution_is_terminated_time_out() {
        let solution: Solution<Integer> = Solution {
            time_out: true,
            ..Default::default()
        };
        assert!(solution.is_terminated());
    }

    #[test]
    fn solution_is_not_terminated() {
        let solution: Solution<Integer> = Solution {
            is_optimal: false,
            is_infeasible: false,
            time_out: false,
            ..Default::default()
        };
        assert!(!solution.is_terminated());
    }

    #[test]
    fn search() {
        let mut solver = MockSearch { counter: 3 };
        let solution = solver.search();
        assert!(solution.is_ok());
        assert_eq!(
            solution.unwrap(),
            Solution {
                cost: Some(0),
                ..Default::default()
            }
        );
    }
}
