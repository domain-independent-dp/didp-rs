use super::data_structure::{exceed_bound, BfsNode, StateRegistry, SuccessorGenerator};
use super::rollout::get_solution_cost_and_suffix;
use super::search::{Parameters, Search, SearchInput, Solution};
use super::util::{update_bound_if_better, update_solution, TimeKeeper};
use dypdl::{variable_type, Transition, TransitionInterface};
use std::collections::VecDeque;
use std::error::Error;
use std::fmt;
use std::mem;
use std::rc::Rc;

/// Parameters for breadth-first search (BrFS).
#[derive(Debug, PartialEq, Clone, Copy, Default)]
pub struct BrfsParameters<T> {
    /// Keep nodes in all layers for duplicate detection.
    ///
    /// BrFS searches layer by layer, where the i th layer contains states that can be reached with i transitions.
    /// By default, BrFS only keeps states in the current layer to check for duplicates.
    /// If `keep_all_layers` is `true`, BrFS keeps states in all layers to check for duplicates.
    pub keep_all_layers: bool,
    /// Common parameters.
    pub parameters: Parameters<T>,
}

/// Breadth-first search solver.
///
/// This solver uses forward search based on the shortest path problem.
/// It only works with problems where the cost expressions are in the form of `cost + w`, `cost * w`, `max(cost, w)`, or `min(cost, w)`
/// where `cost` is `IntegerExpression::Cost`or `ContinuousExpression::Cost` and `w` is a numeric expression independent of `cost`.
///
/// It searches nodes in the breadth-first order.
///
/// Type parameter `N` is a node type that implements `BfsNode`.
/// Type parameter `E` is a type of a function that evaluates a transition and insert a successor node into a state registry.
/// The last argument of the function is the primal bound of the solution cost.
///
/// # Examples
///
/// ```
/// use dypdl::prelude::*;
/// use dypdl_heuristic_search::{BrfsParameters, Search};
/// use dypdl_heuristic_search::search_algorithm::{
///     BreadthFirstSearch, FNode, SearchInput, SuccessorGenerator,
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
/// let parameters = BrfsParameters::default();
///
/// let mut solver = BreadthFirstSearch::<_, FNode<_>, _>::new(
///     input, transition_evaluator, parameters,
/// );
/// let solution = solver.search().unwrap();
/// assert_eq!(solution.cost, Some(1));
/// assert_eq!(solution.transitions, vec![increment]);
/// assert!(!solution.is_infeasible);
/// ```
pub struct BreadthFirstSearch<'a, T, N, E, V = Transition>
where
    T: variable_type::Numeric + fmt::Display + Ord + 'static,
    N: BfsNode<T, V>,
    E: Fn(&N, Rc<V>, &mut StateRegistry<T, N>, Option<T>) -> Option<(Rc<N>, bool)>,
    V: TransitionInterface + Clone + Default,
    Transition: From<V>,
{
    generator: SuccessorGenerator<V>,
    suffix: &'a [V],
    transition_evaluator: E,
    keep_all_layers: bool,
    primal_bound: Option<T>,
    get_all_solutions: bool,
    quiet: bool,
    open: VecDeque<Rc<N>>,
    next_open: VecDeque<Rc<N>>,
    registry: StateRegistry<T, N>,
    layer_index: usize,
    layer_dual_bound: Option<T>,
    time_keeper: TimeKeeper,
    solution: Solution<T>,
}

impl<'a, T, N, E, V> BreadthFirstSearch<'a, T, N, E, V>
where
    T: variable_type::Numeric + fmt::Display + Ord + 'static,
    N: BfsNode<T, V> + Clone,
    E: Fn(&N, Rc<V>, &mut StateRegistry<T, N>, Option<T>) -> Option<(Rc<N>, bool)>,
    V: TransitionInterface + Clone + Default,
    Transition: From<V>,
{
    /// Create a new breadth-first search solver.
    pub fn new(
        input: SearchInput<'a, N, V>,
        transition_evaluator: E,
        parameters: BrfsParameters<T>,
    ) -> BreadthFirstSearch<'a, T, N, E, V> {
        let mut time_keeper = parameters
            .parameters
            .time_limit
            .map_or_else(TimeKeeper::default, TimeKeeper::with_time_limit);
        let get_all_solutions = parameters.parameters.get_all_solutions;
        let primal_bound = parameters.parameters.primal_bound;
        let quiet = parameters.parameters.quiet;
        let mut open = VecDeque::new();
        let next_open = VecDeque::new();
        let mut registry = StateRegistry::new(input.generator.model.clone());

        if let Some(capacity) = parameters.parameters.initial_registry_capacity {
            registry.reserve(capacity);
        }

        let mut solution = Solution::default();
        if let Some(node) = input.node.clone() {
            let (node, _) = registry.insert(node).unwrap();
            open.push_back(node);
            solution.generated += 1;
        } else {
            solution.is_infeasible = true;
        }

        time_keeper.stop();

        BreadthFirstSearch {
            generator: input.generator,
            suffix: input.solution_suffix,
            transition_evaluator,
            keep_all_layers: parameters.keep_all_layers,
            primal_bound,
            get_all_solutions,
            quiet,
            open,
            next_open,
            registry,
            layer_index: 0,
            layer_dual_bound: None,
            time_keeper,
            solution,
        }
    }
}

impl<'a, T, N, E, V> Search<T> for BreadthFirstSearch<'a, T, N, E, V>
where
    T: variable_type::Numeric + fmt::Display + Ord + 'static,
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

        loop {
            if self.open.is_empty() {
                if let Some(bound) = self.layer_dual_bound {
                    self.solution.time = self.time_keeper.elapsed_time();
                    update_bound_if_better(&mut self.solution, bound, model, self.quiet);
                }

                if !self.quiet {
                    println!(
                        "Searched layer: {}, expanded: {}, elapsed time: {}",
                        self.layer_index,
                        self.solution.expanded,
                        self.time_keeper.elapsed_time()
                    );
                }

                if self.next_open.is_empty() {
                    break;
                }

                mem::swap(&mut self.open, &mut self.next_open);

                if !self.keep_all_layers {
                    self.registry.clear();
                }

                self.layer_index += 1;
                self.layer_dual_bound = None;
            }

            while let Some(node) = self.open.pop_front() {
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
                        if let Some(bound) = successor.bound(model) {
                            if !exceed_bound(model, bound, self.layer_dual_bound) {
                                self.layer_dual_bound = Some(bound);
                            }
                        }

                        self.next_open.push_back(successor);

                        if new_generated {
                            self.solution.generated += 1;
                        }
                    }
                }
            }
        }

        self.solution.is_infeasible = self.solution.cost.is_none();
        self.solution.is_optimal = self.solution.cost.is_some();
        self.solution.time = self.time_keeper.elapsed_time();
        self.time_keeper.stop();
        Ok((self.solution.clone(), true))
    }
}
