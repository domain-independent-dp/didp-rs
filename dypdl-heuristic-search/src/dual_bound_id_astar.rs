use crate::search_algorithm::data_structure::ParentAndChildStateFunctionCache;

use super::f_evaluator_type::FEvaluatorType;
use super::search_algorithm::{FNode, Parameters, Search, SearchInput, SuccessorGenerator};
use dypdl::{variable_type, StateFunctionCache, Transition};
use std::fmt;
use std::rc::Rc;
use std::str;

/// Creates an Iterative Deepening A* (IDA*) solver using the dual bound as a heuristic function.
///
/// This solver uses forward search based on the shortest path problem.
/// It only works with problems where the cost expressions are in the form of `cost + w`, `cost * w`, `max(cost, w)`, or `min(cost, w)`
/// where `cost` is `IntegerExpression::Cost`or `ContinuousExpression::Cost` and `w` is a numeric expression independent of `cost`.
/// `f_evaluator_type` must be specified appropriately according to the cost expressions.
///
/// IDA* combines the optimality of A* with the memory efficiency of depth-first search.
/// It performs a series of depth-first searches with increasing f-cost thresholds.
/// Each iteration explores all nodes with f-cost up to the current threshold.
/// The threshold for the next iteration is set to the minimum f-cost of nodes that exceeded the current threshold.
///
/// # References
///
/// Richard E. Korf. "Depth-First Iterative-Deepening: An Optimal Admissible Tree Search,"
/// Artificial Intelligence, vol. 27(1), pp. 97-109, 1985.
///
/// Stuart Russell and Peter Norvig. "Artificial Intelligence: A Modern Approach" (4th Edition),
/// Pearson, 2020.
///
/// # Examples
///
/// ```
/// use dypdl::prelude::*;
/// use dypdl_heuristic_search::{create_dual_bound_id_astar, FEvaluatorType, Parameters};
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
/// model.add_dual_bound(IntegerExpression::from(0)).unwrap();
///
/// let model = Rc::new(model);
/// let parameters = Parameters::default();
/// let f_evaluator_type = FEvaluatorType::Plus;
///
/// let mut solver = create_dual_bound_id_astar(model, parameters, f_evaluator_type);
/// let solution = solver.search().unwrap();
/// assert_eq!(solution.cost, Some(1));
/// assert_eq!(solution.transitions, vec![increment]);
/// assert!(!solution.is_infeasible);
/// ```
pub fn create_dual_bound_id_astar<T>(
    model: Rc<dypdl::Model>,
    parameters: Parameters<T>,
    f_evaluator_type: FEvaluatorType,
) -> Box<dyn Search<T>>
where
    T: variable_type::Numeric + fmt::Display + Ord + 'static,
    <T as str::FromStr>::Err: fmt::Debug,
{
    let generator = SuccessorGenerator::<Transition>::from_model(model.clone(), false);
    let base_cost_evaluator = move |cost, base_cost| f_evaluator_type.eval(cost, base_cost);
    let cost = match f_evaluator_type {
        FEvaluatorType::Plus => T::zero(),
        FEvaluatorType::Product => T::one(),
        FEvaluatorType::Max => T::min_value(),
        FEvaluatorType::Min => T::max_value(),
        FEvaluatorType::Overwrite => T::zero(),
    };

    let state = model.target.clone();
    let mut cache = StateFunctionCache::new(&model.state_functions);
    let h_evaluator = move |state: &_, cache: &mut _| {
        if model.has_dual_bounds() {
            model.eval_dual_bound(state, cache)
        } else {
            // Without dual bounds, h = 0 (f = g, iterative deepening on cost alone)
            Some(T::zero())
        }
    };
    let f_evaluator = move |g, h, _: &_| f_evaluator_type.eval(g, h);
    let node = FNode::generate_root_node(
        state,
        &mut cache,
        cost,
        &generator.model,
        &h_evaluator,
        &f_evaluator,
        parameters.primal_bound,
    );
    let input = SearchInput {
        node,
        generator,
        solution_suffix: &[],
    };
    let transition_evaluator =
        move |node: &FNode<_>, transition, cache: &mut _, registry: &mut _, primal_bound| {
            node.insert_successor_node(
                transition,
                cache,
                registry,
                &h_evaluator,
                &f_evaluator,
                primal_bound,
            )
        };

    Box::new(IdAstar::<_, FNode<_>, _, _, _>::new(
        input,
        transition_evaluator,
        base_cost_evaluator,
        parameters,
    ))
}

use super::search_algorithm::{
    data_structure::{exceed_bound, BfsNode, StateRegistry, TransitionWithId},
    get_solution_cost_and_suffix,
    util::{print_dual_bound, update_bound_if_better, update_solution, TimeKeeper},
    Solution,
};
use dypdl::TransitionInterface;
use std::error::Error;

pub struct IdAstar<'a, T, N, E, B, V = Transition>
where
    T: variable_type::Numeric + Ord + fmt::Display,
    N: BfsNode<T, TransitionWithId<V>>,
    E: FnMut(
        &N,
        Rc<TransitionWithId<V>>,
        &mut ParentAndChildStateFunctionCache,
        &mut StateRegistry<T, N>,
        Option<T>,
    ) -> Option<(Rc<N>, bool)>,
    B: FnMut(T, T) -> T,
    V: TransitionInterface + Clone + Default,
    Transition: From<TransitionWithId<V>>,
{
    generator: SuccessorGenerator<V>,
    suffix: &'a [TransitionWithId<V>],
    transition_evaluator: E,
    base_cost_evaluator: B,
    primal_bound: Option<T>,
    quiet: bool,
    threshold: Option<T>,
    next_threshold: Option<T>,
    root_node: Option<Rc<N>>,
    registry: StateRegistry<T, N>,
    function_cache: ParentAndChildStateFunctionCache,
    applicable_transitions: Vec<Rc<TransitionWithId<V>>>,
    time_keeper: TimeKeeper,
    solution: Solution<T>,
}

impl<'a, T, N, E, B, V> IdAstar<'a, T, N, E, B, V>
where
    T: variable_type::Numeric + Ord + fmt::Display,
    N: BfsNode<T, TransitionWithId<V>>,
    E: FnMut(
        &N,
        Rc<TransitionWithId<V>>,
        &mut ParentAndChildStateFunctionCache,
        &mut StateRegistry<T, N>,
        Option<T>,
    ) -> Option<(Rc<N>, bool)>,
    B: FnMut(T, T) -> T,
    V: TransitionInterface + Clone + Default,
    Transition: From<TransitionWithId<V>>,
{
    /// Create a new IDA* solver.
    pub fn new(
        input: SearchInput<'a, N, V>,
        transition_evaluator: E,
        base_cost_evaluator: B,
        parameters: Parameters<T>,
    ) -> IdAstar<'a, T, N, E, B, V> {
        let mut time_keeper = parameters
            .time_limit
            .map_or_else(TimeKeeper::default, TimeKeeper::with_time_limit);
        let primal_bound = parameters.primal_bound;
        let quiet = parameters.quiet;

        let mut registry = StateRegistry::<_, _>::new(input.generator.model.clone());

        if let Some(capacity) = parameters.initial_registry_capacity {
            registry.reserve(capacity);
        }

        let mut solution = Solution::default();
        let mut threshold = None;
        let mut root_node = None;

        if let Some(node) = input.node {
            let result = registry.insert(node);
            let node = result.information.unwrap();
            solution.best_bound = node.bound(&input.generator.model);
            threshold = node.bound(&input.generator.model);
            root_node = Some(node);
            solution.generated += 1;

            if !quiet {
                solution.time = time_keeper.elapsed_time();
                print_dual_bound(&solution);
                if let Some(t) = threshold {
                    println!("Initial f-threshold: {}", t);
                }
            }
        } else {
            solution.is_infeasible = true;
        }

        let function_cache =
            ParentAndChildStateFunctionCache::new(&input.generator.model.state_functions);

        time_keeper.stop();

        IdAstar {
            generator: input.generator,
            suffix: input.solution_suffix,
            transition_evaluator,
            base_cost_evaluator,
            primal_bound,
            quiet,
            threshold,
            next_threshold: None,
            root_node,
            registry,
            function_cache,
            applicable_transitions: Vec::new(),
            time_keeper,
            solution,
        }
    }

    /// Perform depth-first search with f-cost threshold.
    /// Returns Some(cost) if solution found, None if no solution within threshold.
    /// Updates next_threshold with minimum f-cost that exceeded current threshold.
    fn dfs_threshold(&mut self, node: Rc<N>, threshold: T) -> Option<T> {
        let model = self.generator.model.clone();

        // Check if node's f-cost exceeds threshold or primal bound
        if let Some(f_cost) = node.bound(&model) {
            // Check if f-cost exceeds the threshold
            // For minimization, we prune nodes with f_cost > threshold (higher cost => worse)
            // For maximization, we prune nodes with f_cost < threshold (lower cost => worse)
            let exceeds_threshold = if model.reduce_function == dypdl::ReduceFunction::Min {
                f_cost > threshold
            } else {
                f_cost < threshold
            };

            if exceeds_threshold {
                // Update next threshold with minimum exceeding f-cost
                // For minimization, we want the minimum (smallest) of exceeding bounds
                // For maximization, we want the maximum (largest) of exceeding bounds
                let should_update = if model.reduce_function == dypdl::ReduceFunction::Min {
                    self.next_threshold.is_none() || f_cost < self.next_threshold.unwrap()
                } else {
                    self.next_threshold.is_none() || f_cost > self.next_threshold.unwrap()
                };

                if should_update {
                    self.next_threshold = Some(f_cost);
                }
                return None;
            }

            if exceed_bound(&model, f_cost, self.primal_bound) {
                return None;
            }
        }

        self.function_cache.parent.clear();

        // Check if this is a goal state
        if let Some((cost, suffix)) = get_solution_cost_and_suffix(
            &model,
            &*node,
            self.suffix,
            &mut self.base_cost_evaluator,
            &mut self.function_cache,
        ) {
            if !exceed_bound(&model, cost, self.primal_bound) {
                self.primal_bound = Some(cost);
                let time = self.time_keeper.elapsed_time();
                update_solution(&mut self.solution, &*node, cost, suffix, time, self.quiet);
                return Some(cost);
            }
            return None;
        }

        if self.time_keeper.check_time_limit(self.quiet) {
            self.solution.time_out = true;
            self.solution.time = self.time_keeper.elapsed_time();
            return None;
        }

        self.solution.expanded += 1;

        // Generate successors
        self.generator.generate_applicable_transitions(
            node.state(),
            &mut self.function_cache.parent,
            &mut self.applicable_transitions,
        );

        let mut successors = Vec::new();
        for transition in self.applicable_transitions.drain(..) {
            if let Some((successor, new_generated)) = (self.transition_evaluator)(
                &node,
                transition,
                &mut self.function_cache,
                &mut self.registry,
                self.primal_bound,
            ) {
                successors.push(successor);

                if new_generated {
                    self.solution.generated += 1;
                }
            }
        }
        // Sort successors by f-cost for efficiency
        successors.sort();
        for successor in successors {
            if let Some(cost) = self.dfs_threshold(successor, threshold) {
                return Some(cost);
            }
        }

        None
    }
}

impl<T, N, E, B, V> Search<T> for IdAstar<'_, T, N, E, B, V>
where
    T: variable_type::Numeric + Ord + fmt::Display,
    N: BfsNode<T, TransitionWithId<V>>,
    E: FnMut(
        &N,
        Rc<TransitionWithId<V>>,
        &mut ParentAndChildStateFunctionCache,
        &mut StateRegistry<T, N>,
        Option<T>,
    ) -> Option<(Rc<N>, bool)>,
    B: FnMut(T, T) -> T,
    V: TransitionInterface + Clone + Default,
    Transition: From<TransitionWithId<V>>,
{
    fn search_next(&mut self) -> Result<(Solution<T>, bool), Box<dyn Error>> {
        self.time_keeper.start();

        if self.solution.is_infeasible {
            self.time_keeper.stop();
            return Ok((self.solution.clone(), true));
        }

        let mut iteration = 0;

        while let Some(current_threshold) = self.threshold {
            iteration += 1;
            self.next_threshold = None;

            if !self.quiet {
                println!(
                    "IDA* iteration {}: f-threshold = {}, expanded: {}, generated: {}, elapsed time: {}",
                    iteration,
                    current_threshold,
                    self.solution.expanded,
                    self.solution.generated,
                    self.time_keeper.elapsed_time()
                );
            }

            // Clear the registry for the next iteration while keeping capacity - could be improved for memory usage
            self.registry.clear();

            if let Some(root) = &self.root_node {
                if let Some(_cost) = self.dfs_threshold(root.clone(), current_threshold) {
                    // Solution found
                    self.solution.is_optimal = true;
                    self.solution.time = self.time_keeper.elapsed_time();
                    self.time_keeper.stop();
                    return Ok((self.solution.clone(), true));
                }
            }

            // Check if we should continue with next threshold
            if let Some(next_t) = self.next_threshold {
                if exceed_bound(&self.generator.model, next_t, self.primal_bound) {
                    break;
                }
                self.threshold = Some(next_t);

                self.solution.time = self.time_keeper.elapsed_time();
                update_bound_if_better(
                    &mut self.solution,
                    next_t,
                    &self.generator.model,
                    self.quiet,
                );
            } else {
                // No more nodes
                break;
            }

            if self.time_keeper.check_time_limit(self.quiet) {
                self.solution.time_out = true;
                break;
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
