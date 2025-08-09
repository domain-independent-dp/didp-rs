use super::acps::ProgressiveSearchParameters;
use super::data_structure::{
    exceed_bound, BfsNode, ParentAndChildStateFunctionCache, StateRegistry, SuccessorGenerator,
    TransitionWithId,
};
use super::rollout::get_solution_cost_and_suffix;
use super::util::{print_dual_bound, update_bound_if_better, update_solution, TimeKeeper};
use super::{Parameters, Search, SearchInput, Solution};
use dypdl::{variable_type, Transition, TransitionInterface};
use std::collections;
use std::error::Error;
use std::fmt;
use std::rc::Rc;

/// Anytime Pack Progressive Search (APPS).
///
/// This solver uses forward search based on the shortest path problem.
/// It only works with problems where the cost expressions are in the form of `cost + w`, `cost * w`, `max(cost, w)`, or `min(cost, w)`
/// where `cost` is `IntegerExpression::Cost`or `ContinuousExpression::Cost` and `w` is a numeric expression independent of `cost`.
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
/// Sataya Gautam Vadlamudi, Sandip Aine, Partha Pratim Chakrabarti. "Anytime Pack Search,"
/// Natural Computing, vol. 15(3), pp. 395-414, 2016
///
/// # Examples
///
/// ```
/// use dypdl::prelude::*;
/// use dypdl_heuristic_search::{Parameters, ProgressiveSearchParameters, Search};
/// use dypdl_heuristic_search::search_algorithm::{
///     Apps, FNode, SearchInput, SuccessorGenerator,
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
/// let mut function_cache = StateFunctionCache::new(&model.state_functions);
/// let cost = 0;
/// let h_evaluator = |_: &_, _: &mut _| Some(0);
/// let f_evaluator = |g, h, _: &_| g + h;
/// let primal_bound = None;
/// let node = FNode::generate_root_node(
///     state,
///     &mut function_cache,
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
///     move |node: &FNode<_>, transition, cache: &mut _, registry: &mut _, primal_bound| {
///         node.insert_successor_node(
///             transition,
///             cache,
///             registry,
///             &h_evaluator,
///             &f_evaluator,
///             primal_bound,
///         )
///     };
/// let base_cost_evaluator = |cost, base_cost| cost + base_cost;
/// let parameters = Parameters::default();
/// let progressive_parameters = ProgressiveSearchParameters::default();
///
/// let mut solver = Apps::<_, FNode<_>, _, _>::new(
///     input, transition_evaluator, base_cost_evaluator, parameters, progressive_parameters,
/// );
/// let solution = solver.search().unwrap();
/// assert_eq!(solution.cost, Some(1));
/// assert_eq!(solution.transitions, vec![increment]);
/// assert!(!solution.is_infeasible);
/// ```
pub struct Apps<'a, T, N, E, B, V = Transition>
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
    progressive_parameters: ProgressiveSearchParameters,
    primal_bound: Option<T>,
    get_all_solutions: bool,
    quiet: bool,
    width: usize,
    open: collections::BinaryHeap<Rc<N>>,
    children: collections::BinaryHeap<Rc<N>>,
    suspend: collections::BinaryHeap<Rc<N>>,
    registry: StateRegistry<T, N>,
    function_cache: ParentAndChildStateFunctionCache,
    applicable_transitions: Vec<Rc<TransitionWithId<V>>>,
    goal_found: bool,
    time_keeper: TimeKeeper,
    solution: Solution<T>,
}

impl<'a, T, N, E, B, V> Apps<'a, T, N, E, B, V>
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
    /// Creates a new APPS solver.
    pub fn new(
        input: SearchInput<'a, N, V>,
        transition_evaluator: E,
        base_cost_evaluator: B,
        parameters: Parameters<T>,
        progressive_parameters: ProgressiveSearchParameters,
    ) -> Apps<'a, T, N, E, B, V> {
        let mut time_keeper = parameters
            .time_limit
            .map_or_else(TimeKeeper::default, TimeKeeper::with_time_limit);
        let primal_bound = parameters.primal_bound;
        let get_all_solutions = parameters.get_all_solutions;
        let quiet = parameters.quiet;

        let mut open = collections::BinaryHeap::new();
        let children = collections::BinaryHeap::new();
        let suspend = collections::BinaryHeap::new();
        let mut registry = StateRegistry::<_, _>::new(input.generator.model.clone());

        if let Some(capacity) = parameters.initial_registry_capacity {
            registry.reserve(capacity);
        }

        let mut solution = Solution::default();

        if let Some(node) = input.node {
            let result = registry.insert(node);
            let node = result.information.unwrap();
            solution.generated += 1;
            solution.best_bound = node.bound(&input.generator.model);
            open.push(node);

            if !quiet {
                solution.time = time_keeper.elapsed_time();
                print_dual_bound(&solution);
            }
        } else {
            solution.is_infeasible = true;
        }

        let function_cache =
            ParentAndChildStateFunctionCache::new(&input.generator.model.state_functions);
        time_keeper.stop();

        Apps {
            generator: input.generator,
            suffix: input.solution_suffix,
            transition_evaluator,
            base_cost_evaluator,
            progressive_parameters,
            primal_bound,
            get_all_solutions,
            quiet,
            width: progressive_parameters.init,
            open,
            children,
            suspend,
            registry,
            function_cache,
            applicable_transitions: Vec::new(),
            goal_found: false,
            time_keeper,
            solution,
        }
    }
}

impl<T, N, E, B, V> Search<T> for Apps<'_, T, N, E, B, V>
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
        if self.solution.is_terminated() {
            return Ok((self.solution.clone(), true));
        }

        self.time_keeper.start();
        let model = self.generator.model.clone();
        let suffix = self.suffix;

        while !self.open.is_empty() || !self.children.is_empty() || !self.suspend.is_empty() {
            if self.open.is_empty() {
                while self.open.len() < self.width {
                    if let Some(node) = self.children.pop() {
                        if node.is_closed() {
                            continue;
                        }
                        if let Some(bound) = node.bound(&model) {
                            if exceed_bound(&model, bound, self.primal_bound) {
                                if N::ordered_by_bound() {
                                    self.children.clear();
                                }
                                continue;
                            }
                        }
                        self.open.push(node);
                    } else {
                        break;
                    }
                }

                while let Some(node) = self.children.pop() {
                    if node.is_closed() {
                        continue;
                    }
                    if let Some(bound) = node.bound(&model) {
                        if exceed_bound(&model, bound, self.primal_bound) {
                            if N::ordered_by_bound() {
                                self.children.clear();
                            }
                            continue;
                        }
                    }
                    self.suspend.push(node);
                }

                // Run out current candidates
                if self.open.is_empty() {
                    if self.progressive_parameters.reset && self.goal_found {
                        self.width = self.progressive_parameters.init;
                    } else {
                        self.width = self.progressive_parameters.increase_width(self.width);
                    }

                    if N::ordered_by_bound() {
                        if let Some(bound) =
                            self.suspend.peek().map(|node| node.bound(&model).unwrap())
                        {
                            if exceed_bound(&model, bound, self.primal_bound) {
                                self.suspend.clear();
                            } else {
                                self.solution.time = self.time_keeper.elapsed_time();
                                update_bound_if_better(
                                    &mut self.solution,
                                    bound,
                                    &model,
                                    self.quiet,
                                );
                            }
                        }
                    }

                    while self.open.len() < self.width {
                        if let Some(node) = self.suspend.pop() {
                            if node.is_closed() {
                                continue;
                            }
                            if let Some(bound) = node.bound(&model) {
                                if exceed_bound(&model, bound, self.primal_bound) {
                                    if N::ordered_by_bound() {
                                        self.suspend.clear();
                                    }
                                    continue;
                                }
                            }
                            self.open.push(node);
                        } else {
                            break;
                        }
                    }

                    self.goal_found = false;
                }
            }

            while let Some(node) = self.open.pop() {
                if node.is_closed() {
                    continue;
                }
                node.close();

                if let Some(dual_bound) = node.bound(&model) {
                    if exceed_bound(&model, dual_bound, self.primal_bound) {
                        if N::ordered_by_bound() {
                            self.open.clear();
                        }
                        continue;
                    }
                }

                self.function_cache.parent.clear();

                if let Some((cost, suffix)) = get_solution_cost_and_suffix(
                    &model,
                    &*node,
                    suffix,
                    &mut self.base_cost_evaluator,
                    &mut self.function_cache,
                ) {
                    if !exceed_bound(&model, cost, self.primal_bound) {
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
                self.generator.generate_applicable_transitions(
                    node.state(),
                    &mut self.function_cache.parent,
                    &mut self.applicable_transitions,
                );

                for transition in self.applicable_transitions.drain(..) {
                    if let Some((successor, new_generated)) = (self.transition_evaluator)(
                        &node,
                        transition,
                        &mut self.function_cache,
                        &mut self.registry,
                        self.primal_bound,
                    ) {
                        self.children.push(successor);

                        if new_generated {
                            self.solution.generated += 1;
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
