use super::data_structure::{exceed_bound, BfsNode, StateRegistry, SuccessorGenerator};
use super::rollout::get_solution_cost_and_suffix;
use super::search::{Parameters, Search, SearchInput, Solution};
use super::util::{print_dual_bound, update_solution, TimeKeeper};
use dypdl::{variable_type, Transition, TransitionInterface};
use std::error::Error;
use std::fmt;
use std::mem;
use std::rc::Rc;

/// Parameters for best-first search.
#[derive(Debug, PartialEq, Clone, Copy)]
pub struct DbdfsParameters<T> {
    /// Width of the discrepancy to search.
    pub width: usize,
    /// Common parameters.
    pub parameters: Parameters<T>,
}

impl<T: Default> Default for DbdfsParameters<T> {
    fn default() -> Self {
        Self {
            width: 1,
            parameters: Parameters::default(),
        }
    }
}

/// Discrepancy-Bounded Depth-First Search (DBDFS).
///
/// This solver uses forward search based on the shortest path problem.
/// It only works with problems where the cost expressions are in the form of `cost + w`, `cost * w`, `max(cost, w)`, or `min(cost, w)`
/// where `cost` is `IntegerExpression::Cost`or `ContinuousExpression::Cost` and `w` is a numeric expression independent of `cost`.
///
/// It performs depth-first search where the discrepancy of searched nodes is bounded.
/// When a node has the discrepancy of `d`, its best successor has the discrepancy of `d`,
/// and other.successors have the discrepancy of `d + 1`.
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
/// Proceedings of the 33rd International Conference on Automated Planning and Scheduling (ICAPS), 2023.
///
/// J. Christopher Beck. "Discrepancy-Bounded Depth First Search,"
/// Second International Workshop on Integration of AI and OR Technologies in Constraint Programming for Combinatorial Optimization Problems (CPAIOR), 2000.
///
/// # Examples
///
/// ```
/// use dypdl::prelude::*;
/// use dypdl_heuristic_search::{DbdfsParameters, Parameters, Search};
/// use dypdl_heuristic_search::search_algorithm::{
///     Dbdfs, FNode, SearchInput, SuccessorGenerator,
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
/// let parameters = DbdfsParameters::default();
///
/// let mut solver = Dbdfs::<_, FNode<_>, _, _>::new(
///     input, transition_evaluator, base_cost_evaluator, parameters,
/// );
/// let solution = solver.search().unwrap();
/// assert_eq!(solution.cost, Some(1));
/// assert_eq!(solution.transitions, vec![increment]);
/// assert!(!solution.is_infeasible);
/// ```
pub struct Dbdfs<'a, T, N, E, B, V = Transition>
where
    T: variable_type::Numeric + Ord + fmt::Display,
    N: BfsNode<T, V>,
    E: Fn(&N, Rc<V>, &mut StateRegistry<T, N>, Option<T>) -> Option<(Rc<N>, bool)>,
    B: Fn(T, T) -> T,
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
    width: usize,
    discrepancy_limit: usize,
    open: Vec<(Rc<N>, usize)>,
    next_open: Vec<(Rc<N>, usize)>,
    registry: StateRegistry<T, N>,
    time_keeper: TimeKeeper,
    solution: Solution<T>,
}

impl<'a, T, N, E, B, V> Dbdfs<'a, T, N, E, B, V>
where
    T: variable_type::Numeric + Ord + fmt::Display,
    N: BfsNode<T, V>,
    E: Fn(&N, Rc<V>, &mut StateRegistry<T, N>, Option<T>) -> Option<(Rc<N>, bool)>,
    B: Fn(T, T) -> T,
    V: TransitionInterface + Clone + Default,
    Transition: From<V>,
{
    /// Create a new DBDFS solver.
    pub fn new(
        input: SearchInput<'a, N, V>,
        transition_evaluator: E,
        base_cost_evaluator: B,
        parameters: DbdfsParameters<T>,
    ) -> Dbdfs<'a, T, N, E, B, V> {
        let mut time_keeper = parameters
            .parameters
            .time_limit
            .map_or_else(TimeKeeper::default, TimeKeeper::with_time_limit);
        let primal_bound = parameters.parameters.primal_bound;
        let get_all_solutions = parameters.parameters.get_all_solutions;
        let quiet = parameters.parameters.quiet;

        let mut open = Vec::new();
        let next_open = Vec::new();
        let mut registry = StateRegistry::<_, _>::new(input.generator.model.clone());

        if let Some(capacity) = parameters.parameters.initial_registry_capacity {
            registry.reserve(capacity);
        }

        let mut solution = Solution::default();

        if let Some(node) = input.node {
            let (node, _) = registry.insert(node).unwrap();
            solution.best_bound = node.bound(&input.generator.model);
            open.push((node, 0));
            solution.generated += 1;

            if !quiet {
                solution.time = time_keeper.elapsed_time();
                print_dual_bound(&solution);
            }
        } else {
            solution.is_infeasible = true;
        }

        let discrepancy_limit = parameters.width - 1;

        if !quiet {
            println!("Initial discrepancy limit: {}", discrepancy_limit);
        }

        time_keeper.stop();

        Dbdfs {
            generator: input.generator,
            suffix: input.solution_suffix,
            transition_evaluator,
            base_cost_evaluator,
            primal_bound,
            get_all_solutions,
            quiet,
            width: parameters.width,
            discrepancy_limit,
            open,
            next_open,
            registry,
            time_keeper,
            solution,
        }
    }
}

impl<'a, T, N, E, B, V> Search<T> for Dbdfs<'a, T, N, E, B, V>
where
    T: variable_type::Numeric + Ord + fmt::Display,
    N: BfsNode<T, V>,
    E: Fn(&N, Rc<V>, &mut StateRegistry<T, N>, Option<T>) -> Option<(Rc<N>, bool)>,
    B: Fn(T, T) -> T,
    V: TransitionInterface + Clone + Default,
    Transition: From<V>,
{
    fn search_next(&mut self) -> Result<(Solution<T>, bool), Box<dyn Error>> {
        self.time_keeper.start();
        let model = &self.generator.model;

        while !self.open.is_empty() || !self.next_open.is_empty() {
            if self.open.is_empty() {
                mem::swap(&mut self.open, &mut self.next_open);
                self.discrepancy_limit += self.width;

                if !self.quiet {
                    println!(
                        "New discrepancy limit: {}, expanded: {}, elapsed time: {}",
                        self.discrepancy_limit,
                        self.solution.expanded,
                        self.time_keeper.elapsed_time()
                    );
                }
            }

            let (node, discrepancy) = self.open.pop().unwrap();

            if node.is_closed() {
                continue;
            }
            node.close();

            if let Some(dual_bound) = node.bound(model) {
                if exceed_bound(model, dual_bound, self.primal_bound) {
                    continue;
                }
            }

            if let Some((cost, suffix)) =
                get_solution_cost_and_suffix(model, &*node, self.suffix, &self.base_cost_evaluator)
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

            if let Some(best) = successors.pop() {
                let mut successors = successors
                    .into_iter()
                    .map(|x| (x, discrepancy + 1))
                    .collect();

                if discrepancy < self.discrepancy_limit {
                    self.open.append(&mut successors);
                } else {
                    self.next_open.append(&mut successors);
                }

                self.open.push((best, discrepancy));
            }
        }

        self.solution.is_infeasible = self.solution.cost.is_none();
        self.solution.is_optimal = self.solution.cost.is_some();
        self.solution.time = self.time_keeper.elapsed_time();
        self.time_keeper.stop();
        Ok((self.solution.clone(), true))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dypdl::Integer;

    #[test]
    fn parameters_default() {
        let parameters = DbdfsParameters::<Integer>::default();
        assert_eq!(parameters.width, 1);
        assert_eq!(parameters.parameters, Parameters::default());
    }
}
