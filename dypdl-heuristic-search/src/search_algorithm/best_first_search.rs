use super::data_structure::state_registry::{StateInRegistry, StateRegistry};
use super::data_structure::{
    exceed_bound, BfsNodeInterface, HashableSignatureVariables, SuccessorGenerator,
};
use super::search::{Search, Solution};
use super::util;
use dypdl::{variable_type, ReduceFunction};
use std::collections::BinaryHeap;
use std::error::Error;
use std::fmt;
use std::marker::PhantomData;
use std::rc::Rc;

/// Best-first search solver.
///
/// This solver uses forward search based on the shortest path problem.
/// It only works with problems where the cost expressions are in the form of `cost + w`, `cost * w`, `max(cost, w)`, or `min(cost, w)`
/// where `cost` is `IntegerExpression::Cost`or `ContinuousExpression::Cost` and `w` is a numeric expression independent of `cost`.
///
/// It uses `h_evaluator` and `f_evaluator` for pruning.
/// If `h_evaluator` returns `None`, the state is pruned.
/// If `f_pruning` and `f_evaluator` returns a value that exceeds the primal bound, the state is pruned.
///
/// `ordered_by_f` indicates if the open list is ordered by the f-value.
///
/// `is_optimal` indicates if the optimality is guaranteed.
///
/// # References
///
/// Ryo Kuroiwa and J. Christopher Beck. "Domain-Independent Dynamic Programming: Generic State Space Search for Combinatorial Optimization,"
/// Proceedings of the 33rd International Conference on Automated Planning and Scheduling (ICAPS), 2023.
///
/// Stephen Edelkamp, Shahid Jabbar, Alberto Lluch Lafuente. "Cost-Algebraic Heuristic Search,"
/// Proceedings of the 20th National Conference on Artificial Intelligence (AAAI), pp. 1362-1367, 2005.
///
/// Peter E. Hart, Nills J. Nilsson, Bertram Raphael. "A Formal Basis for the Heuristic Determination of Minimum Cost Paths",
/// IEEE Transactions of Systems Science and Cybernetics, vol. SSC-4(2), pp. 100-107, 1968.
///
/// # Examples
///
/// ```
/// use dypdl::prelude::*;
/// use dypdl_heuristic_search::search_algorithm::{BestFirstSearch, Search};
/// use dypdl_heuristic_search::search_algorithm::data_structure::FNode;
/// use dypdl_heuristic_search::search_algorithm::data_structure::successor_generator::{
///     SuccessorGenerator
/// };
/// use dypdl_heuristic_search::search_algorithm::util::{
///     ForwardSearchParameters, Parameters,
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
/// let h_evaluator = |_: &_, _: &_| Some(0);
/// let f_evaluator = |g, h, _: &_, _: &_| g + h;
///
/// let model = Rc::new(model);
/// let generator = SuccessorGenerator::from_model(model.clone(), false);
/// let parameters = ForwardSearchParameters {
///     generator,
///     parameters: Parameters::default(),
///     initial_registry_capacity: None
/// };
///
/// let mut solver = BestFirstSearch::<_, FNode<_>, _, _>::new(
///     model, h_evaluator, f_evaluator, true, true, true, parameters
/// );
/// let solution = solver.search().unwrap();
/// assert_eq!(solution.cost, Some(1));
/// assert_eq!(solution.transitions, vec![increment]);
/// assert!(!solution.is_infeasible);
/// ```
pub struct BestFirstSearch<T, N, H, F>
where
    T: variable_type::Numeric + Ord + fmt::Display,
    N: BfsNodeInterface<T>,
    H: Fn(&StateInRegistry, &dypdl::Model) -> Option<T>,
    F: Fn(T, T, &StateInRegistry, &dypdl::Model) -> T,
{
    generator: SuccessorGenerator,
    h_evaluator: H,
    f_evaluator: F,
    f_pruning: bool,
    ordered_by_f: bool,
    is_optimal: bool,
    primal_bound: Option<T>,
    get_all_solutions: bool,
    quiet: bool,
    open: BinaryHeap<Rc<N>>,
    registry: StateRegistry<T, N>,
    time_keeper: util::TimeKeeper,
    solution: Solution<T>,
    phantom: PhantomData<N>,
}

impl<T, N, H, F> BestFirstSearch<T, N, H, F>
where
    T: variable_type::Numeric + Ord + fmt::Display,
    N: BfsNodeInterface<T>,
    H: Fn(&StateInRegistry, &dypdl::Model) -> Option<T>,
    F: Fn(T, T, &StateInRegistry, &dypdl::Model) -> T,
{
    /// Create a new best-first search solver.
    pub fn new(
        model: Rc<dypdl::Model>,
        h_evaluator: H,
        f_evaluator: F,
        f_pruning: bool,
        ordered_by_f: bool,
        is_optimal: bool,
        parameters: util::ForwardSearchParameters<T>,
    ) -> BestFirstSearch<T, N, H, F> {
        let time_keeper = parameters
            .parameters
            .time_limit
            .map_or_else(util::TimeKeeper::default, util::TimeKeeper::with_time_limit);
        let primal_bound = parameters.parameters.primal_bound;
        let get_all_solutions = parameters.parameters.get_all_solutions;
        let quiet = parameters.parameters.quiet;

        let mut open = BinaryHeap::new();
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

            if f_pruning && is_optimal {
                solution.best_bound = Some(f);
            }
        } else {
            solution.is_infeasible = true;
        }

        BestFirstSearch {
            generator: parameters.generator,
            h_evaluator,
            f_evaluator,
            f_pruning,
            ordered_by_f,
            is_optimal,
            primal_bound,
            get_all_solutions,
            quiet,
            open,
            registry,
            time_keeper,
            solution,
            phantom: PhantomData::default(),
        }
    }
}

impl<T, N, H, F> Search<T> for BestFirstSearch<T, N, H, F>
where
    T: variable_type::Numeric + Ord + fmt::Display,
    N: BfsNodeInterface<T>,
    H: Fn(&StateInRegistry<Rc<HashableSignatureVariables>>, &dypdl::Model) -> Option<T>,
    F: Fn(T, T, &StateInRegistry<Rc<HashableSignatureVariables>>, &dypdl::Model) -> T,
{
    fn search_next(&mut self) -> Result<(Solution<T>, bool), Box<dyn Error>> {
        let minimize = self.registry.model().reduce_function == ReduceFunction::Min;

        while let Some(node) = self.open.pop() {
            if node.closed() {
                continue;
            }

            node.close();

            let f = node.get_bound(self.registry.model());

            if self.f_pruning
                && self.ordered_by_f
                && exceed_bound(self.registry.model(), f, self.primal_bound)
            {
                self.open.clear();
                break;
            }

            if let Some(best_bound) = self.solution.best_bound {
                if (minimize && f > best_bound) || (!minimize && f < best_bound) {
                    self.solution.best_bound = Some(f);

                    if !self.quiet {
                        println!("f = {}, expanded: {}", f, self.solution.expanded);
                    }
                }
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
                    let cost = node.cost();
                    self.solution.cost = Some(cost);
                    self.primal_bound = Some(cost);
                    self.solution.transitions = node.transitions();

                    if let Some(best_bound) = self.solution.best_bound {
                        self.solution.is_optimal = cost == best_bound;
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

            for transition in self.generator.applicable_transitions(node.state()) {
                let successor = node.generate_successor(
                    transition,
                    &mut self.registry,
                    &self.h_evaluator,
                    &self.f_evaluator,
                    primal_bound,
                );

                if let Some((successor, _, _, new_generated)) = successor {
                    self.open.push(successor);

                    if new_generated {
                        self.solution.generated += 1;
                    }
                }
            }
        }

        self.solution.is_infeasible = self.solution.cost.is_none();
        self.solution.is_optimal = self.is_optimal && self.solution.cost.is_some();
        self.solution.time = self.time_keeper.elapsed_time();
        Ok((self.solution.clone(), true))
    }
}
