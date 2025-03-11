use super::data_structure::{
    exceed_bound, BfsNode, ParentAndChildStateFunctionCache, StateRegistry, SuccessorGenerator,
};
use super::rollout::get_solution_cost_and_suffix;
use super::search::{Parameters, Search, SearchInput, Solution};
use super::util::{update_bound_if_better, update_solution, TimeKeeper};
use dypdl::{variable_type, Transition, TransitionInterface};
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
/// It searches a node in the best-first order.
///
/// Type parameter `N` is a node type that implements `BfsNode`.
/// Type parameter `E` is a type of a function that evaluates a transition and insert a successor node into a state registry.
/// The last argument of the function is the primal bound of the solution cost.
/// Type parameter `B` is a type of a function that combines the g-value (the cost to a state) and the base cost.
/// It should be the same function as the cost expression, e.g., `cost + base_cost` for `cost + w`.
///
/// # References
///
/// Ryo Kuroiwa and J. Christopher Beck. "Domain-Independent Dynamic Programming: Generic State Space Search for Combinatorial Optimization,"
/// Proceedings of the 33rd International Conference on Automated Planning and Scheduling (ICAPS), pp. 236-244, 2023.
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
/// use dypdl_heuristic_search::{Parameters, Search};
/// use dypdl_heuristic_search::search_algorithm::{
///     BestFirstSearch, FNode, SearchInput, SuccessorGenerator,
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
///
/// let mut solver = BestFirstSearch::<_, FNode<_>, _, _>::new(
///     input, transition_evaluator, base_cost_evaluator, parameters,
/// );
/// let solution = solver.search().unwrap();
/// assert_eq!(solution.cost, Some(1));
/// assert_eq!(solution.transitions, vec![increment]);
/// assert!(!solution.is_infeasible);
/// ```
pub struct BestFirstSearch<'a, T, N, E, B, V = Transition>
where
    T: variable_type::Numeric + Ord + fmt::Display,
    N: BfsNode<T, V>,
    E: FnMut(
        &N,
        Rc<V>,
        &mut ParentAndChildStateFunctionCache,
        &mut StateRegistry<T, N>,
        Option<T>,
    ) -> Option<(Rc<N>, bool)>,
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
    open: BinaryHeap<Rc<N>>,
    registry: StateRegistry<T, N>,
    function_cache: ParentAndChildStateFunctionCache,
    applicable_transitions: Vec<Rc<V>>,
    time_keeper: TimeKeeper,
    solution: Solution<T>,
    phantom: PhantomData<N>,
}

impl<T, N, E, B, V> BestFirstSearch<'_, T, N, E, B, V>
where
    T: variable_type::Numeric + Ord + fmt::Display,
    N: BfsNode<T, V>,
    E: FnMut(
        &N,
        Rc<V>,
        &mut ParentAndChildStateFunctionCache,
        &mut StateRegistry<T, N>,
        Option<T>,
    ) -> Option<(Rc<N>, bool)>,
    B: FnMut(T, T) -> T,
    V: TransitionInterface + Clone + Default,
    Transition: From<V>,
{
    /// Creates a new best-first search solver.
    pub fn new(
        input: SearchInput<N, V>,
        transition_evaluator: E,
        base_cost_evaluator: B,
        parameters: Parameters<T>,
    ) -> BestFirstSearch<T, N, E, B, V> {
        let mut time_keeper = parameters
            .time_limit
            .map_or_else(TimeKeeper::default, TimeKeeper::with_time_limit);
        let primal_bound = parameters.primal_bound;
        let get_all_solutions = parameters.get_all_solutions;
        let quiet = parameters.quiet;

        let mut open = BinaryHeap::new();
        let mut registry = StateRegistry::new(input.generator.model.clone());

        if let Some(capacity) = parameters.initial_registry_capacity {
            registry.reserve(capacity);
        }

        let mut solution = Solution::default();
        if let Some(node) = input.node {
            let result = registry.insert(node);
            open.push(result.information.unwrap());
            solution.generated += 1;
        } else {
            solution.is_infeasible = true;
        }

        time_keeper.stop();

        let function_cache =
            ParentAndChildStateFunctionCache::new(&input.generator.model.state_functions);

        BestFirstSearch {
            generator: input.generator,
            suffix: input.solution_suffix,
            transition_evaluator,
            base_cost_evaluator,
            primal_bound,
            get_all_solutions,
            quiet,
            open,
            registry,
            function_cache,
            applicable_transitions: Vec::new(),
            time_keeper,
            solution,
            phantom: PhantomData,
        }
    }
}

impl<T, N, E, B, V> Search<T> for BestFirstSearch<'_, T, N, E, B, V>
where
    T: variable_type::Numeric + Ord + fmt::Display,
    N: BfsNode<T, V>,
    E: FnMut(
        &N,
        Rc<V>,
        &mut ParentAndChildStateFunctionCache,
        &mut StateRegistry<T, N>,
        Option<T>,
    ) -> Option<(Rc<N>, bool)>,
    B: FnMut(T, T) -> T,
    V: TransitionInterface + Clone + Default,
    Transition: From<V>,
{
    fn search_next(&mut self) -> Result<(Solution<T>, bool), Box<dyn Error>> {
        self.time_keeper.start();
        let model = &self.generator.model;

        while let Some(node) = self.open.pop() {
            if node.is_closed() {
                continue;
            }
            node.close();

            if let Some(dual_bound) = node.bound(model) {
                if exceed_bound(model, dual_bound, self.primal_bound) {
                    if N::ordered_by_bound() {
                        self.open.clear();
                    }
                    continue;
                }

                if N::ordered_by_bound() || self.open.is_empty() {
                    self.solution.time = self.time_keeper.elapsed_time();
                    update_bound_if_better(&mut self.solution, dual_bound, model, self.quiet);
                }
            }

            self.function_cache.parent.clear();

            if let Some((cost, suffix)) = get_solution_cost_and_suffix(
                model,
                &*node,
                self.suffix,
                &mut self.base_cost_evaluator,
                &mut self.function_cache,
            ) {
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
                    self.open.push(successor);

                    if new_generated {
                        self.solution.generated += 1;
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
