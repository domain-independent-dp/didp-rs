use super::data_structure::{
    exceed_bound, BfsNode, ParentAndChildStateFunctionCache, StateRegistry,
};
use super::rollout::get_solution_cost_and_suffix;
use super::search::{Parameters, Search, SearchInput, Solution};
use super::util::{print_dual_bound, update_bound_if_better, update_solution, TimeKeeper};
use super::{SuccessorGeneratorWithDominance, TransitionWithId};
use dypdl::{variable_type, Transition, TransitionInterface};
use std::cmp;
use std::collections::BinaryHeap;
use std::error::Error;
use std::fmt;
use std::rc::Rc;

/// Parameters for progressive search.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct ProgressiveSearchParameters {
    /// The initial width.
    pub init: usize,
    /// The amount of increase.
    pub step: usize,
    /// The maximum value of the width.
    pub bound: Option<usize>,
    /// Whether reset the bound when a better solution is found.
    pub reset: bool,
}

impl ProgressiveSearchParameters {
    /// Returns the increased width.
    pub fn increase_width(&self, width: usize) -> usize {
        if let Some(bound) = self.bound {
            cmp::min(width + self.step, bound)
        } else {
            width + self.step
        }
    }
}

impl Default for ProgressiveSearchParameters {
    /// Returns parameters where the initial width is 1, the step is 1, no bound, and no reset.
    fn default() -> Self {
        Self {
            init: 1,
            step: 1,
            bound: None,
            reset: false,
        }
    }
}

/// Anytime Column Progressive Search (ACPS).
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
/// Ryo Kuroiwa and J. Christopher Beck."Solving Domain-Independent Dynamic Programming with Anytime Heuristic Search,"
/// Proceedings of the 33rd International Conference on Automated Planning and Scheduling (ICAPS), pp. 245-253, 2023.
///
/// Sataya Gautam Vadlamudi, Piyush Gaurav, Sandip Aine, and Partha Pratim Chakrabarti. "Anytime Column Search,"
/// Proceedings of AI 2012: Advances in Artificial Intelligence, pp. 254-255, 2012.
///
/// # Examples
///
/// ```
/// use dypdl::prelude::*;
/// use dypdl_heuristic_search::{Parameters, ProgressiveSearchParameters, Search};
/// use dypdl_heuristic_search::search_algorithm::{
///     Acps, FNode, SearchInput, SuccessorGenerator,
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
/// let mut solver = Acps::<_, FNode<_>, _, _>::new(
///     input, transition_evaluator, base_cost_evaluator, parameters, progressive_parameters,
/// );
/// let solution = solver.search().unwrap();
/// assert_eq!(solution.cost, Some(1));
/// assert_eq!(solution.transitions, vec![increment]);
/// assert!(!solution.is_infeasible);
/// ```
pub struct Acps<'a, T, N, E, B, V = Transition>
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
    Transition: From<V>,
{
    generator: SuccessorGeneratorWithDominance<V>,
    suffix: &'a [TransitionWithId<V>],
    transition_evaluator: E,
    base_cost_evaluator: B,
    progressive_parameters: ProgressiveSearchParameters,
    primal_bound: Option<T>,
    get_all_solutions: bool,
    quiet: bool,
    width: usize,
    open: Vec<BinaryHeap<Rc<N>>>,
    registry: StateRegistry<T, N>,
    function_cache: ParentAndChildStateFunctionCache,
    applicable_transitions: Vec<Rc<TransitionWithId<V>>>,
    layer_index: usize,
    node_index: usize,
    no_node: bool,
    goal_found: bool,
    dual_bound_candidate: Option<T>,
    time_keeper: TimeKeeper,
    solution: Solution<T>,
}

impl<'a, T, N, E, B, V> Acps<'a, T, N, E, B, V>
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
    Transition: From<V>,
{
    /// Creates a new ACPS solver.
    pub fn new(
        input: SearchInput<'a, N, TransitionWithId<V>>,
        transition_evaluator: E,
        base_cost_evaluator: B,
        parameters: Parameters<T>,
        progressive_parameters: ProgressiveSearchParameters,
    ) -> Acps<'a, T, N, E, B, V> {
        let mut time_keeper = parameters
            .time_limit
            .map_or_else(TimeKeeper::default, TimeKeeper::with_time_limit);
        let primal_bound = parameters.primal_bound;
        let quiet = parameters.quiet;

        let mut open = vec![BinaryHeap::new()];
        let mut registry = StateRegistry::new(input.generator.model.clone());

        if let Some(capacity) = parameters.initial_registry_capacity {
            registry.reserve(capacity);
        }

        let mut solution = Solution::default();

        if let Some(node) = input.node {
            let result = registry.insert(node);
            let node = result.information.unwrap();
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

        let function_cache =
            ParentAndChildStateFunctionCache::new(&input.generator.model.state_functions);

        time_keeper.stop();

        let generator = SuccessorGeneratorWithDominance::from(input.generator);

        Acps {
            generator: generator,
            suffix: input.solution_suffix,
            transition_evaluator,
            base_cost_evaluator,
            progressive_parameters,
            primal_bound,
            get_all_solutions: parameters.get_all_solutions,
            quiet,
            width: progressive_parameters.init,
            open,
            registry,
            function_cache,
            applicable_transitions: Vec::new(),
            layer_index: 0,
            node_index: 0,
            no_node: true,
            goal_found: false,
            dual_bound_candidate: None,
            time_keeper,
            solution,
        }
    }
}

impl<'a, T, N, E, B, V> Search<T> for Acps<'a, T, N, E, B, V>
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
    Transition: From<V>,
{
    fn search_next(&mut self) -> Result<(Solution<T>, bool), Box<dyn Error>> {
        if self.solution.is_terminated() {
            return Ok((self.solution.clone(), true));
        }

        self.time_keeper.start();
        let model = self.generator.get_model().clone();
        let suffix = self.suffix;

        loop {
            while self.node_index < self.width && !self.open[self.layer_index].is_empty() {
                let node = self.open[self.layer_index].pop().unwrap();

                if node.is_closed() {
                    continue;
                }
                node.close();

                if let Some(dual_bound) = node.bound(&model) {
                    if exceed_bound(&model, dual_bound, self.primal_bound) {
                        if N::ordered_by_bound() {
                            self.open[self.layer_index].clear();
                        }
                        continue;
                    }
                }

                if self.no_node {
                    self.no_node = false;
                }

                self.function_cache.parent.clear();

                if let Some((cost, suffix)) = get_solution_cost_and_suffix(
                    &model,
                    &*node,
                    suffix,
                    &mut self.base_cost_evaluator,
                    &mut self.function_cache,
                ) {
                    self.node_index += 1;

                    if !exceed_bound(&model, cost, self.primal_bound) {
                        if !self.goal_found {
                            self.goal_found = true;
                        }

                        self.primal_bound = Some(cost);
                        let time = self.time_keeper.elapsed_time();
                        update_solution::<_, _, TransitionWithId<V>>(
                            &mut self.solution,
                            &*node,
                            cost,
                            suffix,
                            time,
                            self.quiet,
                        );
                        self.time_keeper.stop();

                        return Ok((self.solution.clone(), self.solution.is_optimal));
                    } else if self.get_all_solutions {
                        let mut solution = self.solution.clone();
                        let time = self.time_keeper.elapsed_time();
                        update_solution::<_, _, TransitionWithId<V>>(
                            &mut solution,
                            &*node,
                            cost,
                            suffix,
                            time,
                            true,
                        );
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
                self.generator
                    .generate_applicable_transitions_with_dominance(
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
                        while self.layer_index + 1 >= self.open.len() {
                            self.open.push(BinaryHeap::new());
                        }

                        self.open[self.layer_index + 1].push(successor);

                        if new_generated {
                            self.solution.generated += 1;
                        }
                    }
                }

                self.node_index += 1;
            }

            if N::ordered_by_bound() {
                if let Some(bound) = self.open[self.layer_index]
                    .peek()
                    .map(|node| node.bound(&model).unwrap())
                {
                    if !exceed_bound(&model, bound, self.dual_bound_candidate) {
                        self.dual_bound_candidate = Some(bound);
                    }
                }
            }

            self.node_index = 0;

            if self.goal_found {
                self.dual_bound_candidate = None;
                self.layer_index = 0;
                self.no_node = true;
                self.goal_found = false;

                if self.progressive_parameters.reset {
                    self.width = self.progressive_parameters.init;
                } else {
                    self.width = self.progressive_parameters.increase_width(self.width);
                }
            } else if self.layer_index + 1 == self.open.len() {
                if self.no_node {
                    break;
                }

                if let Some(dual_bound) = self.dual_bound_candidate {
                    if exceed_bound(&model, dual_bound, self.primal_bound) {
                        break;
                    } else {
                        self.solution.time = self.time_keeper.elapsed_time();
                        update_bound_if_better(&mut self.solution, dual_bound, &model, self.quiet);
                        self.dual_bound_candidate = None;
                    }
                }

                self.layer_index = 0;
                self.no_node = true;
                self.width = self.progressive_parameters.increase_width(self.width);
            } else {
                self.layer_index += 1;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn progressive_increase_width() {
        let parameters = ProgressiveSearchParameters {
            init: 1,
            step: 2,
            bound: None,
            reset: false,
        };
        assert_eq!(parameters.increase_width(3), 5);
    }

    #[test]
    fn progressive_increase_width_bonded() {
        let parameters = ProgressiveSearchParameters {
            init: 1,
            step: 2,
            bound: Some(4),
            reset: false,
        };
        assert_eq!(parameters.increase_width(3), 4);
    }

    #[test]
    fn parameters_default() {
        let parameters = ProgressiveSearchParameters::default();
        assert_eq!(parameters.init, 1);
        assert_eq!(parameters.step, 1);
        assert_eq!(parameters.bound, None);
        assert!(!parameters.reset);
    }
}
