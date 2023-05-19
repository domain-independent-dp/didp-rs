use super::data_structure::{exceed_bound, BfsNode, StateRegistry};
use super::rollout::get_solution_cost_and_suffix;
use super::search::{Parameters, Search, SearchInput, Solution};
use super::util::{print_dual_bound, update_solution, TimeKeeper};
use super::SuccessorGenerator;
use dypdl::{variable_type, Transition, TransitionInterface};
use std::collections;
use std::error::Error;
use std::fmt;
use std::rc::Rc;

/// Cyclic Best-First Search (CBFS).
///
/// This solver uses forward search based on the shortest path problem.
/// It only works with problems where the cost expressions are in the form of `cost + w`, `cost * w`, `max(cost, w)`, or `min(cost, w)`
/// where `cost` is `IntegerExpression::Cost`or `ContinuousExpression::Cost` and `w` is a numeric expression independent of `cost`.
///
/// It expands the best node from the current layer and then goes to the next layer.
///
/// Type parameter `N` is a node type that implements `BfsNode`.
/// Type parameter `E` is a type of a function that evaluates a transition and insert a successor node into a state registry.
/// The last argument of the function is the primal bound of the solution cost.
///
/// # References
///
/// Ryo Kuroiwa and J. Christopher Beck. "Solving Domain-Independent Dynamic Programming with Anytime Heuristic Search,"
/// Proceedings of the 33rd International Conference on Automated Planning and Scheduling (ICAPS), 2023.
///
/// Gio K. Kao, Edward C. Sewell, and Sheldom H. Jacobson. "A Branch, Bound and Remember Algorithm for the 1|r_i|Σt_i scheduling problem,"
/// Journal of Scheduling, vol. 12(2), pp. 163-175, 2009.
///
/// # Examples
///
/// ```
/// use dypdl::prelude::*;
/// use dypdl_heuristic_search::{Parameters, Search};
/// use dypdl_heuristic_search::search_algorithm::{
///     Cbfs, FNode, SearchInput, SuccessorGenerator,
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
/// let model = Rc::new(model);
///
/// let state = model.target.clone();
/// let cost = 0;
/// let h_evaluator = |_: &_| Some(0);
/// let f_evaluator = |g, h, _: &_| g + h;
/// let primal_bound = None;
/// let node = FNode::generate_root_node(
///     state,
///     cost,
///     &model,
///     &h_evaluator,
///     &f_evaluator,
///     primal_bound,
/// );
/// let generator = SuccessorGenerator::<Transition>::from_model(model.clone(), false);
/// let input = SearchInput {
///     node,
///     generator,
///     solution_suffix: &[],
/// };
/// let transition_evaluator =
///     move |node: &FNode<_>, transition, registry: &mut _, primal_bound| {
///         node.insert_successor_node(
///             transition,
///             registry,
///             &h_evaluator,
///             &f_evaluator,
///             primal_bound,
///         )
///     };
/// let parameters = Parameters::default();
///
/// let mut solver = Cbfs::<_, FNode<_>, _>::new(input, transition_evaluator, parameters);
/// let solution = solver.search().unwrap();
/// assert_eq!(solution.cost, Some(1));
/// assert_eq!(solution.transitions, vec![increment]);
/// assert!(!solution.is_infeasible);
/// ```
pub struct Cbfs<'a, T, N, E, V = Transition>
where
    T: variable_type::Numeric + Ord + fmt::Display,
    N: BfsNode<T, V>,
    E: Fn(&N, Rc<V>, &mut StateRegistry<T, N>, Option<T>) -> Option<(Rc<N>, bool)>,
    V: TransitionInterface + Clone + Default,
    Transition: From<V>,
{
    generator: SuccessorGenerator<V>,
    suffix: &'a [V],
    transition_evaluator: E,
    primal_bound: Option<T>,
    get_all_solutions: bool,
    quiet: bool,
    open: Vec<collections::BinaryHeap<Rc<N>>>,
    registry: StateRegistry<T, N>,
    time_keeper: TimeKeeper,
    solution: Solution<T>,
}

impl<'a, T, N, E, V> Cbfs<'a, T, N, E, V>
where
    T: variable_type::Numeric + Ord + fmt::Display,
    N: BfsNode<T, V>,
    E: Fn(&N, Rc<V>, &mut StateRegistry<T, N>, Option<T>) -> Option<(Rc<N>, bool)>,
    V: TransitionInterface + Clone + Default,
    Transition: From<V>,
{
    /// Create a new CBFS solver.
    pub fn new(
        input: SearchInput<'a, N, V>,
        transition_evaluator: E,
        parameters: Parameters<T>,
    ) -> Cbfs<'a, T, N, E, V> {
        let time_keeper = parameters
            .time_limit
            .map_or_else(TimeKeeper::default, TimeKeeper::with_time_limit);
        let primal_bound = parameters.primal_bound;
        let get_all_solutions = parameters.get_all_solutions;
        let quiet = parameters.quiet;

        let mut open = vec![collections::BinaryHeap::new()];
        let mut registry = StateRegistry::<_, _>::new(input.generator.model.clone());

        if let Some(capacity) = parameters.initial_registry_capacity {
            registry.reserve(capacity);
        }

        let mut solution = Solution::default();

        if let Some(node) = input.node {
            let (node, _) = registry.insert(node).unwrap();
            solution.generated += 1;
            solution.best_bound = node.bound(&input.generator.model);
            open[0].push(node);

            if !quiet {
                solution.time = time_keeper.elapsed_time();
                print_dual_bound(&solution);
            }
        } else {
            solution.is_infeasible = true;
        }

        Cbfs {
            generator: input.generator,
            suffix: input.solution_suffix,
            transition_evaluator,
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

impl<'a, T, N, E, V> Search<T> for Cbfs<'a, T, N, E, V>
where
    T: variable_type::Numeric + Ord + fmt::Display,
    N: BfsNode<T, V>,
    E: Fn(&N, Rc<V>, &mut StateRegistry<T, N>, Option<T>) -> Option<(Rc<N>, bool)>,
    V: TransitionInterface + Clone + Default,
    Transition: From<V>,
{
    fn search_next(&mut self) -> Result<(Solution<T>, bool), Box<dyn Error>> {
        if self.solution.is_terminated() {
            return Ok((self.solution.clone(), true));
        }

        self.time_keeper.start();
        let model = &self.generator.model;
        let mut i = 0;
        let mut no_node = true;

        loop {
            if let Some(node) = self.open[i].pop() {
                if node.is_closed() {
                    continue;
                }
                node.close();

                if let Some(dual_bound) = node.bound(model) {
                    if exceed_bound(model, dual_bound, self.primal_bound) {
                        if N::ordered_by_bound() {
                            self.open[i].clear();
                        }
                        continue;
                    }
                }

                if no_node {
                    no_node = false;
                }

                if let Some((cost, suffix)) =
                    get_solution_cost_and_suffix(model, &*node, self.suffix)
                {
                    if !exceed_bound(model, cost, self.primal_bound) {
                        self.primal_bound = Some(cost);
                        let time = self.time_keeper.elapsed_time();
                        update_solution(&mut self.solution, &*node, cost, suffix, time, self.quiet);
                        self.time_keeper.stop();

                        return Ok((self.solution.clone(), self.solution.is_optimal));
                    } else if self.get_all_solutions {
                        let mut solution = self.solution.clone();
                        let time = self.time_keeper.elapsed_time();
                        update_solution(&mut solution, &*node, cost, suffix, time, true);
                        self.time_keeper.stop();

                        return Ok((solution, false));
                    }
                    continue;
                }

                if self.time_keeper.check_time_limit(self.quiet) {
                    self.solution.time_out = true;
                    self.solution.time = self.time_keeper.elapsed_time();
                    self.time_keeper.stop();

                    return Ok((self.solution.clone(), true));
                }

                self.solution.expanded += 1;

                for transition in self.generator.applicable_transitions(node.state()) {
                    if let Some((successor, new_generated)) = (self.transition_evaluator)(
                        &node,
                        transition,
                        &mut self.registry,
                        self.primal_bound,
                    ) {
                        while i + 1 >= self.open.len() {
                            self.open.push(collections::BinaryHeap::new());
                        }

                        self.open[i + 1].push(successor);

                        if new_generated {
                            self.solution.generated += 1;
                        }
                    }
                }
            }

            if no_node && i + 1 == self.open.len() {
                break;
            } else if i + 1 == self.open.len() {
                i = 0;
                no_node = true;
            } else {
                i += 1;
            }
        }

        self.solution.is_infeasible = self.solution.cost.is_none();
        self.solution.is_optimal = self.solution.cost.is_some();
        self.solution.time = self.time_keeper.elapsed_time();
        self.time_keeper.stop();
        Ok((self.solution.clone(), true))
    }
}
