use super::data_structure::state_registry::{StateInRegistry, StateRegistry};
use super::data_structure::{exceed_bound, BfsNodeInterface, SuccessorGenerator};
use super::search::{Search, Solution};
use super::util;
use dypdl::variable_type;
use std::error::Error;
use std::fmt;
use std::rc::Rc;

/// Depth-First Branch-and-Bound which expands child nodes in the best-first order (DFBBBFS).
///
/// This solver uses forward search based on the shortest path problem.
/// It only works with problems where the cost expressions are in the form of `cost + w`, `cost * w`, `max(cost, w)`, or `min(cost, w)`
/// where `cost` is `IntegerExpression::Cost`or `ContinuousExpression::Cost` and `w` is a numeric expression independent of `cost`.
///
/// It uses `h_evaluator` and `f_evaluator` for pruning.
/// If `h_evaluator` returns `None`, the state is pruned.
/// If `f_pruning` and `f_evaluator` returns a value that exceeds the primal bound, the state is pruned.
///
/// # References
///
/// Ryo Kuroiwa and J. Christopher Beck. "Solving Domain-Independent Dynamic Programming with Anytime Heuristic Search,"
/// Proceedings of the 33rd International Conference on Automated Planning and Scheduling (ICAPS), 2023.
///
/// # Examples
///
/// ```
/// use dypdl::prelude::*;
/// use dypdl_heuristic_search::search_algorithm::{DfbbBfs, Search};
/// use dypdl_heuristic_search::search_algorithm::data_structure::FNode;
/// use dypdl_heuristic_search::search_algorithm::data_structure::successor_generator::{
///     SuccessorGenerator
/// };
/// use dypdl_heuristic_search::search_algorithm::util::{
///     ForwardSearchParameters, Parameters, ProgressiveSearchParameters,
/// };
/// use std::rc::Rc;
///
/// let mut model = Model::default();
/// let variable = model.add_integer_variable("variable", 0).unwrap();
/// model.add_base_case(
///     vec![Condition::comparison_i(ComparisonOperator::Ge, variable, 1)]
/// ).unwrap();
/// let mut increment = Transition::new("increment");
/// increment.set_cost(IntegerExpression::Cost + 1);
/// increment.add_effect(variable, variable + 1).unwrap();
/// model.add_forward_transition(increment.clone()).unwrap();
///
/// let h_evaluator = |_: &_| Some(0);
/// let f_evaluator = |g, h, _: &_| g + h;
///
/// let model = Rc::new(model);
/// let generator = SuccessorGenerator::from_model(model.clone(), false);
/// let parameters = ForwardSearchParameters {
///     generator,
///     parameters: Parameters::default(),
///     initial_registry_capacity: None
/// };
///
/// let mut solver = DfbbBfs::<_, FNode<_>, _, _>::new(model, h_evaluator, f_evaluator, true, parameters);
/// let solution = solver.search().unwrap();
/// assert_eq!(solution.cost, Some(1));
/// assert_eq!(solution.transitions, vec![increment]);
/// assert!(!solution.is_infeasible);
/// ```
pub struct DfbbBfs<T, N, H, F>
where
    T: variable_type::Numeric + Ord + fmt::Display,
    N: BfsNodeInterface<T>,
    H: Fn(&StateInRegistry) -> Option<T>,
    F: Fn(T, T, &StateInRegistry) -> T,
{
    generator: SuccessorGenerator,
    h_evaluator: H,
    f_evaluator: F,
    f_pruning: bool,
    primal_bound: Option<T>,
    get_all_solutions: bool,
    quiet: bool,
    open: Vec<Rc<N>>,
    registry: StateRegistry<T, N>,
    time_keeper: util::TimeKeeper,
    solution: Solution<T>,
}

impl<T, N, H, F> DfbbBfs<T, N, H, F>
where
    T: variable_type::Numeric + Ord + fmt::Display,
    N: BfsNodeInterface<T>,
    H: Fn(&StateInRegistry) -> Option<T>,
    F: Fn(T, T, &StateInRegistry) -> T,
{
    /// Create a new DFBBBFS solver.
    pub fn new(
        model: Rc<dypdl::Model>,
        h_evaluator: H,
        f_evaluator: F,
        f_pruning: bool,
        parameters: util::ForwardSearchParameters<T>,
    ) -> DfbbBfs<T, N, H, F> {
        let time_keeper = parameters
            .parameters
            .time_limit
            .map_or_else(util::TimeKeeper::default, util::TimeKeeper::with_time_limit);
        let primal_bound = parameters.parameters.primal_bound;
        let get_all_solutions = parameters.parameters.get_all_solutions;
        let quiet = parameters.parameters.quiet;

        let mut open = Vec::new();
        let mut registry = StateRegistry::new(model);

        if let Some(capacity) = parameters.initial_registry_capacity {
            registry.reserve(capacity);
        }

        let mut solution = Solution::default();

        if let Some((node, h, f)) =
            N::generate_initial_node(&mut registry, &h_evaluator, &f_evaluator)
        {
            open.push(node);
            solution.generated += 1;

            if !quiet {
                println!("Initial h = {}", h);
            }

            if f_pruning {
                solution.best_bound = Some(f);
            }
        } else {
            solution.is_infeasible = true;
        }

        DfbbBfs {
            generator: parameters.generator,
            h_evaluator,
            f_evaluator,
            f_pruning,
            primal_bound,
            get_all_solutions,
            quiet,
            open,
            registry,
            time_keeper,
            solution,
        }
    }
}

impl<T, N, H, F> Search<T> for DfbbBfs<T, N, H, F>
where
    T: variable_type::Numeric + Ord + fmt::Display,
    N: BfsNodeInterface<T>,
    H: Fn(&StateInRegistry) -> Option<T>,
    F: Fn(T, T, &StateInRegistry) -> T,
{
    fn search_next(&mut self) -> Result<(Solution<T>, bool), Box<dyn Error>> {
        while let Some(node) = self.open.pop() {
            if node.closed() {
                continue;
            }

            node.close();

            let f = node.get_bound(self.registry.model());

            if self.f_pruning && exceed_bound(self.registry.model(), f, self.primal_bound) {
                continue;
            }

            if self.registry.model().is_base(node.state()) {
                if exceed_bound(self.registry.model(), node.cost(), self.primal_bound) {
                    if self.get_all_solutions {
                        let mut solution = self.solution.clone();
                        solution.cost = Some(node.cost());
                        solution.transitions = node.transitions();
                        solution.time = self.time_keeper.elapsed_time();

                        return Ok((solution, false));
                    }

                    continue;
                } else {
                    if !self.quiet {
                        println!(
                            "New primal bound: {}, expanded: {}",
                            node.cost(),
                            self.solution.expanded
                        );
                    }

                    let cost = node.cost();
                    self.solution.cost = Some(cost);
                    self.solution.transitions = node.transitions();
                    self.primal_bound = Some(cost);

                    if let Some(bound) = self.solution.best_bound {
                        self.solution.is_optimal = cost == bound;
                    }

                    self.solution.time = self.time_keeper.elapsed_time();

                    return Ok((self.solution.clone(), self.solution.is_optimal));
                }
            }

            if self.time_keeper.check_time_limit() {
                if !self.quiet {
                    println!("Reached time limit.");
                    println!("Expanded: {}", self.solution.expanded);
                }

                self.solution.time_out = true;
                self.solution.time = self.time_keeper.elapsed_time();

                return Ok((self.solution.clone(), true));
            }

            self.solution.expanded += 1;

            let primal_bound = if self.f_pruning {
                self.primal_bound
            } else {
                None
            };
            let mut successors = vec![];

            for transition in self.generator.applicable_transitions(node.state()) {
                let successor = node.generate_successor(
                    transition,
                    &mut self.registry,
                    &self.h_evaluator,
                    &self.f_evaluator,
                    primal_bound,
                );

                if let Some((successor, _, _, new_generated)) = successor {
                    successors.push(successor);

                    if new_generated {
                        self.solution.generated += 1;
                    }
                }
            }

            // reverse sort
            successors.sort_by(|a, b| b.cmp(a));
            self.open.append(&mut successors);
        }

        self.solution.is_infeasible = self.solution.cost.is_none();
        self.solution.is_optimal = self.solution.cost.is_some();
        self.solution.time = self.time_keeper.elapsed_time();
        Ok((self.solution.clone(), true))
    }
}
