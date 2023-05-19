//! A module for utility.

use dypdl::{variable_type::Numeric, Model, Transition, TransitionInterface};

use super::{
    data_structure::{exceed_bound, GetTransitions},
    Solution,
};
use std::time;

/// Data structure to maintain the elapsed time.
///
/// The time keeper starts when it is created.
/// It can be stopped and started again.
pub struct TimeKeeper {
    time_limit: Option<time::Duration>,
    elapsed_time: time::Duration,
    start: time::Instant,
}

impl Default for TimeKeeper {
    fn default() -> TimeKeeper {
        TimeKeeper {
            time_limit: None,
            elapsed_time: time::Duration::from_secs(0),
            start: time::Instant::now(),
        }
    }
}

impl TimeKeeper {
    /// Returns a time keeper with the given time limit.
    pub fn with_time_limit(time_limit: f64) -> TimeKeeper {
        TimeKeeper {
            time_limit: Some(time::Duration::from_secs_f64(time_limit)),
            elapsed_time: time::Duration::from_secs(0),
            start: time::Instant::now(),
        }
    }

    /// Starts the time keeper.
    pub fn start(&mut self) {
        self.start = time::Instant::now()
    }

    /// Stops the time keeper.
    pub fn stop(&mut self) {
        self.elapsed_time += time::Instant::now() - self.start;
    }

    /// Returns the elapsed time.
    pub fn elapsed_time(&self) -> f64 {
        (self.elapsed_time + (time::Instant::now() - self.start)).as_secs_f64()
    }

    /// Returns the remaining time.
    pub fn remaining_time_limit(&self) -> Option<f64> {
        let elapsed_time = self.elapsed_time + (time::Instant::now() - self.start);
        self.time_limit.map(|time_limit| {
            if elapsed_time > time_limit {
                0.0
            } else {
                (time_limit - elapsed_time).as_secs_f64()
            }
        })
    }

    /// Returns whether the time limit is reached.
    pub fn check_time_limit(&self, quiet: bool) -> bool {
        if let Some(remaining) = self.remaining_time_limit() {
            if remaining <= 0.0 {
                if !quiet {
                    println!("Reached time limit.");
                }

                true
            } else {
                false
            }
        } else {
            false
        }
    }
}

/// Prints the current primal bound.
pub fn print_primal_bound<T, V>(solution: &Solution<T, V>)
where
    T: Numeric + std::fmt::Display,
    V: TransitionInterface + Clone,
{
    println!(
        "New primal bound: {}, expanded: {}, elapsed time: {}",
        solution.cost.unwrap(),
        solution.expanded,
        solution.time,
    );
}

/// Prints the current dual bound.
pub fn print_dual_bound<T, V>(solution: &Solution<T, V>)
where
    T: Numeric + std::fmt::Display,
    V: TransitionInterface + Clone,
{
    if let Some(bound) = solution.best_bound {
        println!(
            "New dual bound: {}, expanded: {}, elapsed time: {}",
            bound, solution.expanded, solution.time,
        );
    }
}

/// Updates the current solution given a search node, the solution cost, the suffix, and the time.
///
/// The cost should be the entire cost instead of the cost of the node.
///
/// # Examples
///
/// ```
/// use dypdl::prelude::*;
/// use dypdl_heuristic_search::Solution;
/// use dypdl_heuristic_search::search_algorithm::{
///     FNode, StateInRegistry, get_solution_cost_and_suffix,
/// };
/// use dypdl_heuristic_search::search_algorithm::data_structure::GetTransitions;
/// use dypdl_heuristic_search::search_algorithm::util::update_solution;
/// use std::rc::Rc;
///
/// let mut model = Model::default();
/// let var = model.add_integer_variable("var", 0).unwrap();
/// model.add_base_case(vec![Condition::comparison_i(ComparisonOperator::Ge, var, 3)]).unwrap();
///
/// let mut transition = Transition::new("transition");
/// transition.add_effect(var, var + 1).unwrap();
/// transition.set_cost(IntegerExpression::Cost + 1);
///
/// let h_evaluator = |_: &StateInRegistry| Some(0);
/// let f_evaluator = |g, h, _: &StateInRegistry| g + h;
/// let node = FNode::generate_root_node(
///     model.target.clone(), 0, &model, &h_evaluator, &f_evaluator, None,
/// ).unwrap();
/// let node = node.generate_successor_node(
///     Rc::new(transition.clone()), &model, &h_evaluator, &f_evaluator, None,
/// ).unwrap();
///
/// let suffix = [transition.clone(), transition.clone()];
/// let (cost, suffix) = get_solution_cost_and_suffix(&model, &node, &suffix).unwrap();
/// let time = 0.0;
///
/// let mut solution = Solution::default();
/// update_solution(&mut solution, &node, cost, suffix, time, true);
///
/// assert_eq!(solution.cost, Some(3));
/// assert_eq!(
///     solution.transitions,
///     [transition.clone(), transition.clone(), transition],
/// );
/// ```
pub fn update_solution<T, N, V>(
    solution: &mut Solution<T>,
    node: &N,
    cost: T,
    suffix: &[V],
    time: f64,
    quiet: bool,
) where
    T: Numeric + std::fmt::Display,
    N: GetTransitions<V>,
    V: TransitionInterface + Clone,
    Transition: From<V>,
{
    solution.cost = Some(cost);
    let mut transitions = node.transitions();
    transitions.extend_from_slice(suffix);
    solution.transitions = transitions.into_iter().map(Transition::from).collect();

    if let Some(best_bound) = solution.best_bound {
        solution.is_optimal = cost == best_bound;
    }

    solution.time = time;

    if !quiet {
        print_primal_bound(solution);
    }
}

/// Updates the best dual bound if better.
///
/// # Examples
///
/// ```
/// use dypdl::prelude::*;
/// use dypdl_heuristic_search::Solution;
/// use dypdl_heuristic_search::search_algorithm::util::update_bound_if_better;
/// use std::rc::Rc;
///
/// let mut model = Model::default();
/// model.set_minimize();
///
/// let mut solution = Solution::<_> {
///     best_bound: Some(0),
///     ..Default::default()
/// };
/// update_bound_if_better(&mut solution, 1, &model, true);
/// assert_eq!(solution.best_bound, Some(1));
/// ```
pub fn update_bound_if_better<T, V>(
    solution: &mut Solution<T, V>,
    bound: T,
    model: &Model,
    quiet: bool,
) where
    T: Numeric + std::fmt::Display,
    V: TransitionInterface + Clone,
{
    if solution.best_bound.map_or(true, |best_bound| {
        !exceed_bound(model, best_bound, Some(bound))
    }) {
        solution.best_bound = Some(bound);

        if !quiet {
            print_dual_bound(solution)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::data_structure::{FNode, StateInRegistry};
    use super::*;
    use dypdl::prelude::*;
    use std::rc::Rc;

    #[test]
    fn update_solution_not_optimal() {
        let mut model = Model::default();
        let var = model.add_integer_variable("var", 0).unwrap();
        model
            .add_base_case(vec![Condition::comparison_i(
                ComparisonOperator::Ge,
                var,
                3,
            )])
            .unwrap();

        let mut transition = Transition::new("transition");
        transition.add_effect(var, var + 1).unwrap();
        transition.set_cost(IntegerExpression::Cost + 1);

        let h_evaluator = |_: &StateInRegistry| Some(0);
        let f_evaluator = |g, h, _: &StateInRegistry| g + h;
        let node = FNode::generate_root_node(
            model.target.clone(),
            0,
            &model,
            &h_evaluator,
            &f_evaluator,
            None,
        )
        .unwrap();
        let node = node
            .generate_successor_node(
                Rc::new(transition.clone()),
                &model,
                &h_evaluator,
                &f_evaluator,
                None,
            )
            .unwrap();

        let suffix = &[transition.clone(), transition.clone()];
        let cost = 3;

        let mut solution = Solution::default();
        update_solution(&mut solution, &node, cost, suffix, 0.0, true);

        assert_eq!(solution.cost, Some(3));
        assert_eq!(
            solution.transitions,
            [transition.clone(), transition.clone(), transition],
        );
        assert_eq!(solution.best_bound, None);
        assert!(!solution.is_optimal);
        assert!(!solution.is_infeasible);
        assert_eq!(solution.time, 0.0);
        assert!(!solution.time_out);
        assert_eq!(solution.expanded, 0);
        assert_eq!(solution.generated, 0);
    }

    #[test]
    fn update_solution_optimal() {
        let mut model = Model::default();
        let var = model.add_integer_variable("var", 0).unwrap();
        model
            .add_base_case(vec![Condition::comparison_i(
                ComparisonOperator::Ge,
                var,
                3,
            )])
            .unwrap();

        let mut transition = Transition::new("transition");
        transition.add_effect(var, var + 1).unwrap();
        transition.set_cost(IntegerExpression::Cost + 1);

        let h_evaluator = |_: &StateInRegistry| Some(0);
        let f_evaluator = |g, h, _: &StateInRegistry| g + h;
        let node = FNode::generate_root_node(
            model.target.clone(),
            0,
            &model,
            &h_evaluator,
            &f_evaluator,
            None,
        )
        .unwrap();
        let node = node
            .generate_successor_node(
                Rc::new(transition.clone()),
                &model,
                &h_evaluator,
                &f_evaluator,
                None,
            )
            .unwrap();

        let suffix = &[transition.clone(), transition.clone()];
        let cost = 3;

        let mut solution = Solution {
            best_bound: Some(3),
            ..Default::default()
        };
        update_solution(&mut solution, &node, cost, suffix, 0.0, true);

        assert_eq!(solution.cost, Some(3));
        assert_eq!(
            solution.transitions,
            [transition.clone(), transition.clone(), transition],
        );
        assert_eq!(solution.best_bound, Some(3));
        assert!(solution.is_optimal);
        assert!(!solution.is_infeasible);
        assert_eq!(solution.time, 0.0);
        assert!(!solution.time_out);
        assert_eq!(solution.expanded, 0);
        assert_eq!(solution.generated, 0);
    }

    #[test]
    fn update_bound() {
        let mut model = Model::default();
        model.set_minimize();

        let mut solution = Solution::<_> {
            best_bound: Some(0),
            ..Default::default()
        };
        update_bound_if_better(&mut solution, 1, &model, true);
        assert_eq!(solution.best_bound, Some(1));
    }

    #[test]
    fn update_none_bound() {
        let mut model = Model::default();
        model.set_minimize();

        let mut solution = Solution::<_> {
            best_bound: None,
            ..Default::default()
        };
        update_bound_if_better(&mut solution, 1, &model, true);
        assert_eq!(solution.best_bound, Some(1));
    }

    #[test]
    fn not_update_bound() {
        let mut model = Model::default();
        model.set_minimize();

        let mut solution = Solution::<_> {
            best_bound: Some(2),
            ..Default::default()
        };
        update_bound_if_better(&mut solution, 1, &model, true);
        assert_eq!(solution.best_bound, Some(2));
    }
}
