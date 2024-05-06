use super::data_structure::{exceed_bound, BfsNode, StateRegistry, SuccessorGenerator};
use super::rollout::get_solution_cost_and_suffix;
use super::search::{Parameters, Search, SearchInput, Solution};
use super::util::{self, print_dual_bound, update_bound_if_better, update_solution};
use dypdl::{variable_type, Transition, TransitionInterface};
use std::error::Error;
use std::fmt;
use std::rc::Rc;

/// Depth-First Branch-and-Bound (DFBB).
///
/// This solver uses forward search based on the shortest path problem.
/// It only works with problems where the cost expressions are in the form of `cost + w`, `cost * w`, `max(cost, w)`, or `min(cost, w)`
/// where `cost` is `IntegerExpression::Cost`or `ContinuousExpression::Cost` and `w` is a numeric expression independent of `cost`.
///
/// It expands nodes in the depth-first order, but successor nodes are ordered by the best-first order.
///
/// Type parameter `N` is a node type that implements `BfsNode`.
/// Type parameter `E` is a type of a function that evaluates a transition and insert a successor node into a state registry.
/// The last argument of the function is the primal bound of the solution cost.
/// Type parameter `B` is a type of a function that combines the g-value (the cost to a state) and the base cost.
/// It should be the same function as the cost expression, e.g., `cost + base_cost` for `cost + w`.
///
/// # References
///
/// Ryo Kuroiwa and J. Christopher Beck. "Solving Domain-Independent Dynamic Programming with Anytime Heuristic Search,"
/// Proceedings of the 33rd International Conference on Automated Planning and Scheduling (ICAPS), pp. 245-253, 2023.
///
/// # Examples
///
/// ```
/// use dypdl::prelude::*;
/// use dypdl_heuristic_search::{Parameters, Search};
/// use dypdl_heuristic_search::search_algorithm::{
///     Dfbb, FNode, SearchInput, SuccessorGenerator,
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
/// let base_cost_evaluator = |cost, base_cost| cost + base_cost;
/// let parameters = Parameters::default();
///
/// let mut solver = Dfbb::<_, FNode<_>, _, _>::new(
///     input, transition_evaluator, base_cost_evaluator, parameters,
/// );
/// let solution = solver.search().unwrap();
/// assert_eq!(solution.cost, Some(1));
/// assert_eq!(solution.transitions, vec![increment]);
/// assert!(!solution.is_infeasible);
/// ```
pub struct Dfbb<'a, T, N, E, B, V = Transition>
where
    T: variable_type::Numeric + Ord + fmt::Display,
    N: BfsNode<T, V>,
    E: FnMut(&N, Rc<V>, &mut StateRegistry<T, N>, Option<T>) -> Option<(Rc<N>, bool)>,
    B: FnMut(T, T) -> T,
    V: TransitionInterface + Clone + Default,
    Transition: From<V>,
{
    generator: SuccessorGenerator<V>,
    suffix: &'a [V],
    transition_evaluator: E,
    base_cost_evaluator: B,
    primal_bound: Option<T>,
    get_all_solutions: bool,
    quiet: bool,
    open: Vec<Rc<N>>,
    registry: StateRegistry<T, N>,
    depth_dual_bounds: Vec<Option<T>>,
    n_siblings: Vec<usize>,
    time_keeper: util::TimeKeeper,
    solution: Solution<T>,
}

impl<'a, T, N, E, B, V> Dfbb<'a, T, N, E, B, V>
where
    T: variable_type::Numeric + Ord + fmt::Display,
    N: BfsNode<T, V>,
    E: FnMut(&N, Rc<V>, &mut StateRegistry<T, N>, Option<T>) -> Option<(Rc<N>, bool)>,
    B: FnMut(T, T) -> T,
    V: TransitionInterface + Clone + Default,
    Transition: From<V>,
{
    /// Create a new DFBB solver.
    pub fn new(
        input: SearchInput<'a, N, V>,
        transition_evaluator: E,
        base_cost_evaluator: B,
        parameters: Parameters<T>,
    ) -> Dfbb<'a, T, N, E, B, V> {
        let mut time_keeper = parameters
            .time_limit
            .map_or_else(util::TimeKeeper::default, util::TimeKeeper::with_time_limit);
        let primal_bound = parameters.primal_bound;
        let get_all_solutions = parameters.get_all_solutions;
        let quiet = parameters.quiet;

        let mut open = Vec::new();
        let mut registry = StateRegistry::<_, _>::new(input.generator.model.clone());

        if let Some(capacity) = parameters.initial_registry_capacity {
            registry.reserve(capacity);
        }

        let mut depth_dual_bounds = Vec::default();
        let mut n_siblings = Vec::default();
        let mut solution = Solution::default();

        if let Some(node) = input.node {
            let result = registry.insert(node);
            let node = result.information.unwrap();
            solution.best_bound = node.bound(&input.generator.model);
            open.push(node);
            depth_dual_bounds.push(None);
            n_siblings.push(0);
            solution.generated += 1;

            if !quiet {
                solution.time = time_keeper.elapsed_time();
                print_dual_bound(&solution);
            }
        } else {
            solution.is_infeasible = true;
        }

        time_keeper.stop();

        Dfbb {
            generator: input.generator,
            suffix: input.solution_suffix,
            transition_evaluator,
            base_cost_evaluator,
            primal_bound,
            get_all_solutions,
            quiet,
            open,
            registry,
            depth_dual_bounds,
            n_siblings,
            time_keeper,
            solution,
        }
    }

    fn backtrack(n_siblings: &mut Vec<usize>, depth_dual_bounds: &mut Vec<Option<T>>) {
        while let Some(n) = n_siblings.pop() {
            if n == 0 {
                depth_dual_bounds.pop();
            } else {
                n_siblings.push(n - 1);
                break;
            }
        }
    }
}

impl<'a, T, N, E, B, V> Search<T> for Dfbb<'a, T, N, E, B, V>
where
    T: variable_type::Numeric + Ord + fmt::Display,
    N: BfsNode<T, V>,
    E: FnMut(&N, Rc<V>, &mut StateRegistry<T, N>, Option<T>) -> Option<(Rc<N>, bool)>,
    B: FnMut(T, T) -> T,
    V: TransitionInterface + Clone + Default,
    Transition: From<V>,
{
    fn search_next(&mut self) -> Result<(Solution<T>, bool), Box<dyn Error>> {
        self.time_keeper.start();
        let model = &self.generator.model;

        while let Some(node) = self.open.pop() {
            if node.is_closed() {
                Self::backtrack(&mut self.n_siblings, &mut self.depth_dual_bounds);
                continue;
            }
            node.close();

            if let Some(dual_bound) = node.bound(model) {
                if exceed_bound(model, dual_bound, self.primal_bound) {
                    Self::backtrack(&mut self.n_siblings, &mut self.depth_dual_bounds);
                    continue;
                }
            }

            if let Some((cost, suffix)) = get_solution_cost_and_suffix(
                model,
                &*node,
                self.suffix,
                &mut self.base_cost_evaluator,
            ) {
                Self::backtrack(&mut self.n_siblings, &mut self.depth_dual_bounds);

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

                return Ok((self.solution.clone(), true));
            }

            self.solution.expanded += 1;

            let mut successors = vec![];

            for transition in self.generator.applicable_transitions(node.state()) {
                if let Some((successor, new_generated)) = (self.transition_evaluator)(
                    &node,
                    transition,
                    &mut self.registry,
                    self.primal_bound,
                ) {
                    successors.push(successor);

                    if new_generated {
                        self.solution.generated += 1;
                    }
                }
            }

            successors.sort();

            if successors.is_empty() {
                Self::backtrack(&mut self.n_siblings, &mut self.depth_dual_bounds);
            } else {
                self.n_siblings.push(successors.len() - 1);
                self.depth_dual_bounds.push(None);
            }

            self.open.append(&mut successors);

            if N::ordered_by_bound() {
                let open_len = self.open.len();

                if open_len > 0 {
                    let bound = self.open[open_len - 1].bound(model).unwrap();
                    let depth = self.depth_dual_bounds.len() - 1;
                    let bound_up_to_depth = if depth < 1 {
                        None
                    } else {
                        self.depth_dual_bounds[depth - 1]
                    };
                    let dual_bound = if !exceed_bound(model, bound, bound_up_to_depth) {
                        bound
                    } else {
                        bound_up_to_depth.unwrap()
                    };
                    self.solution.time = self.time_keeper.elapsed_time();

                    if exceed_bound(model, dual_bound, self.primal_bound) {
                        self.open.clear();
                        break;
                    }

                    update_bound_if_better(&mut self.solution, dual_bound, model, self.quiet);

                    if self.n_siblings[depth] == 0 {
                        self.depth_dual_bounds[depth] = bound_up_to_depth;
                    } else {
                        let bound = self.open[open_len - 2].bound(model).unwrap();

                        if !exceed_bound(model, bound, bound_up_to_depth) {
                            self.depth_dual_bounds[depth] = Some(bound);
                        } else {
                            self.depth_dual_bounds[depth] = bound_up_to_depth;
                        }
                    }
                }
            }
        }

        self.solution.is_infeasible = self.solution.cost.is_none();
        self.solution.is_optimal = self.solution.cost.is_some();
        self.solution.best_bound = self.solution.cost;
        self.solution.time = self.time_keeper.elapsed_time();
        self.time_keeper.stop();
        Ok((self.solution.clone(), true))
    }
}
