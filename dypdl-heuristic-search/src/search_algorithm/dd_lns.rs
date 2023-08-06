use super::beam_search::BeamSearchParameters;
use super::data_structure::{
    exceed_bound, BfsNode, StateInRegistry, StateRegistry, SuccessorGenerator, TransitionMutex,
    TransitionWithId,
};
use super::randomized_restricted_dd::{randomized_restricted_dd, RandomizedRestrictedDDParameters};
use super::rollout::get_trace;
use super::search::{Parameters, Search, SearchInput, Solution};
use super::util::{print_primal_bound, update_bound_if_better, TimeKeeper};
use dypdl::{variable_type, Transition, TransitionInterface};
use rand::prelude::*;
use rand_pcg::Pcg64Mcg;
use std::error::Error;
use std::fmt::{Debug, Display};
use std::rc::Rc;

/// Parameters for DD-LNS
#[derive(Clone, PartialEq, Copy, Debug, Default)]
pub struct DdLnsParameters<T> {
    /// Probability to keep a non-best state.
    pub keep_probability: f64,
    /// Random seed.
    pub seed: u64,
    /// Parameters for beam search.
    pub beam_search_parameters: BeamSearchParameters<T>,
}

/// Large Neighborhood Search with Decision Diagrams (DD-LNS).
///
/// It performs LNS by constructing restricted multi-valued decision diagrams (MDD).
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
/// Type parameter `G` is a type of a function that generates a root node given a state and its cost.
///
/// # References
///
/// Xavier Gillard and Pierre Schaus. "Large Neighborhood Search with Decision Diagrams,"
/// Proceedings of the 31st International Joint Conference on Artificial Intelligence (IJCAI), pp. 4754-4760, 2022.
///
/// # Examples
///
/// ```
/// use dypdl::prelude::*;
/// use dypdl_heuristic_search::{Parameters, Search, Solution};
/// use dypdl_heuristic_search::search_algorithm::{
///     BeamSearchParameters, DdLns, DdLnsParameters, FNode, SuccessorGenerator,
///     TransitionWithId,
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
/// let h_evaluator = |_: &_| Some(0);
/// let f_evaluator = |g, h, _: &_| g + h;
/// let primal_bound = None;
/// let generator = SuccessorGenerator::<TransitionWithId>::from_model(model.clone(), false);
/// let transition_evaluator =
///     move |node: &FNode<_, _>, transition, registry: &mut _, primal_bound| {
///         node.insert_successor_node(
///             transition,
///             registry,
///             &h_evaluator,
///             &f_evaluator,
///             primal_bound,
///         )
///     };
/// let base_cost_evaluator = |cost, base_cost| cost + base_cost;
/// let root_generator = |state, cost| {
///     FNode::generate_root_node(
///         state,
///         cost,
///         &model,
///         &h_evaluator,
///         &f_evaluator,
///         primal_bound,
///     )
/// };
///
/// let solution = Solution::<_, _> {
///     cost: Some(1),
///     transitions: vec![TransitionWithId {
///         transition: increment.clone(),
///         id: 0,
///         forced: false,
///     }],
///     ..Default::default()
/// };
///
/// let parameters = DdLnsParameters {
///     keep_probability: 0.1,
///     seed: 0,
///     beam_search_parameters: BeamSearchParameters {
///         beam_size: 10000,
///         parameters: Parameters {
///             time_limit: Some(1800.0),
///             ..Default::default()
///         },
///         ..Default::default()
///     },
/// };
///
/// let mut solver = DdLns::new(
///     generator, transition_evaluator, base_cost_evaluator, root_generator, solution,
///     parameters,
/// );
/// let solution = solver.search().unwrap();
/// assert_eq!(solution.cost, Some(1));
/// assert_eq!(solution.transitions, vec![increment]);
/// assert!(!solution.is_infeasible);
/// ```
pub struct DdLns<T, N, E, B, G, V = Transition>
where
    T: variable_type::Numeric + Ord + Display,
    N: BfsNode<T, TransitionWithId<V>>,
    E: Fn(
        &N,
        Rc<TransitionWithId<V>>,
        &mut StateRegistry<T, N>,
        Option<T>,
    ) -> Option<(Rc<N>, bool)>,
    B: Fn(T, T) -> T,
    G: Fn(StateInRegistry, T) -> Option<N>,
    V: TransitionInterface + Clone + Default,
{
    generator: SuccessorGenerator<TransitionWithId<V>>,
    transition_evaluator: E,
    base_cost_evaluator: B,
    root_generator: G,
    beam_size: usize,
    keep_all_layers: bool,
    keep_probability: f64,
    quiet: bool,
    transition_mutex: TransitionMutex,
    rng: Pcg64Mcg,
    time_keeper: TimeKeeper,
    solution: Solution<T, TransitionWithId<V>>,
    first_call: bool,
}

impl<T, N, E, B, G, V> DdLns<T, N, E, B, G, V>
where
    T: variable_type::Numeric + Ord + Display,
    N: BfsNode<T, TransitionWithId<V>> + Clone,
    E: Fn(
        &N,
        Rc<TransitionWithId<V>>,
        &mut StateRegistry<T, N>,
        Option<T>,
    ) -> Option<(Rc<N>, bool)>,
    B: Fn(T, T) -> T,
    G: Fn(StateInRegistry, T) -> Option<N>,
    V: TransitionInterface + Clone + Default,
{
    /// Create a new DD-LNS solver.
    pub fn new(
        generator: SuccessorGenerator<TransitionWithId<V>>,
        transition_evaluator: E,
        base_cost_evaluator: B,
        root_generator: G,
        solution: Solution<T, TransitionWithId<V>>,
        parameters: DdLnsParameters<T>,
    ) -> DdLns<T, N, E, B, G, V> {
        let mut time_keeper = parameters
            .beam_search_parameters
            .parameters
            .time_limit
            .map_or_else(TimeKeeper::default, TimeKeeper::with_time_limit);
        let transition_mutex = TransitionMutex::new(&generator.model, generator.backward);
        let seed = parameters.seed;
        time_keeper.stop();

        DdLns {
            generator,
            transition_evaluator,
            base_cost_evaluator,
            root_generator,
            beam_size: parameters.beam_search_parameters.beam_size,
            keep_all_layers: parameters.beam_search_parameters.keep_all_layers,
            keep_probability: parameters.keep_probability,
            quiet: parameters.beam_search_parameters.parameters.quiet,
            transition_mutex,
            rng: Pcg64Mcg::seed_from_u64(seed),
            time_keeper,
            solution,
            first_call: true,
        }
    }

    //// Search for the next solution, returning the solution using `TransitionWithId`.
    pub fn search_inner(&mut self) -> (Solution<T, TransitionWithId<V>>, bool) {
        if self.solution.is_terminated() || self.solution.transitions.len() < 2 {
            return (self.solution.clone(), true);
        }

        self.time_keeper.start();

        if self.first_call {
            self.first_call = false;
            self.time_keeper.stop();

            return (self.solution.clone(), false);
        }

        let max_depth = self.solution.transitions.len() - 2;
        let mut d = max_depth;

        let (states, costs): (Vec<_>, Vec<_>) = get_trace(
            &self.generator.model.target,
            T::zero(),
            &self.solution.transitions,
            &self.generator.model,
        )
        .unzip();

        loop {
            let prefix = &self.solution.transitions[..d];
            let prefix_cost = if d == 0 { T::zero() } else { costs[d - 1] };
            let target = StateInRegistry::from(if d == 0 {
                self.generator.model.target.clone()
            } else {
                states[d - 1].clone()
            });
            let node = (self.root_generator)(target, prefix_cost);
            let suffix = &[];
            let generator =
                self.transition_mutex
                    .filter_successor_generator(&self.generator, prefix, suffix);
            let input = SearchInput {
                node,
                generator,
                solution_suffix: suffix,
            };

            let parameters = RandomizedRestrictedDDParameters {
                keep_probability: self.keep_probability,
                best_solution: Some(&self.solution.transitions),
                beam_search_parameters: BeamSearchParameters {
                    beam_size: self.beam_size,
                    keep_all_layers: self.keep_all_layers,
                    parameters: Parameters {
                        primal_bound: self.solution.cost,
                        get_all_solutions: false,
                        quiet: true,
                        time_limit: self.time_keeper.remaining_time_limit(),
                        ..Default::default()
                    },
                },
            };

            let solution = randomized_restricted_dd(
                &input,
                &self.transition_evaluator,
                &self.base_cost_evaluator,
                parameters,
                &mut self.rng,
            );

            self.solution.expanded += solution.expanded;
            self.solution.generated += solution.generated;

            if let Some(cost) = solution.cost {
                if !exceed_bound(&self.generator.model, cost, self.solution.cost) {
                    if let Some(best_bound) = self.solution.best_bound {
                        if cost == best_bound {
                            self.solution.is_optimal = true;
                        }
                    }

                    if Some(cost) == self.solution.best_bound {
                        self.solution.is_optimal = true;
                    }

                    if d == 0 && solution.is_optimal {
                        self.solution.is_optimal = true;
                        self.solution.best_bound = solution.cost;
                    }

                    let mut transitions = prefix.to_vec();
                    transitions.extend(solution.transitions.into_iter());
                    self.solution.transitions = transitions;

                    self.solution.cost = Some(cost);
                    self.solution.time = self.time_keeper.elapsed_time();

                    if !self.quiet {
                        print_primal_bound(&self.solution);
                    }

                    self.time_keeper.stop();

                    return (self.solution.clone(), self.solution.is_optimal);
                }
            }

            if d == 0 {
                if let Some(bound) = solution.best_bound {
                    update_bound_if_better(
                        &mut self.solution,
                        bound,
                        &self.generator.model,
                        self.quiet,
                    );
                }

                if solution.is_infeasible {
                    self.solution.is_optimal = self.solution.cost.is_some();
                    self.solution.is_infeasible = self.solution.cost.is_none();

                    if self.solution.is_optimal {
                        self.solution.best_bound = self.solution.cost;
                    }

                    self.time_keeper.stop();

                    return (self.solution.clone(), true);
                }

                d = max_depth;
            } else {
                d -= 1;
            }

            if solution.time_out {
                self.solution.time = self.time_keeper.elapsed_time();

                self.time_keeper.stop();

                return (self.solution.clone(), true);
            }
        }
    }
}

impl<T, N, E, B, G, V> Search<T> for DdLns<T, N, E, B, G, V>
where
    T: variable_type::Numeric + Ord + Display,
    N: BfsNode<T, TransitionWithId<V>> + Clone,
    E: Fn(
        &N,
        Rc<TransitionWithId<V>>,
        &mut StateRegistry<T, N>,
        Option<T>,
    ) -> Option<(Rc<N>, bool)>,
    B: Fn(T, T) -> T,
    G: Fn(StateInRegistry, T) -> Option<N>,
    V: TransitionInterface + Clone + Default,
    Transition: From<TransitionWithId<V>>,
{
    fn search_next(&mut self) -> Result<(Solution<T>, bool), Box<dyn Error>> {
        let (solution, is_terminated) = self.search_inner();
        let solution = Solution {
            cost: solution.cost,
            best_bound: solution.best_bound,
            is_optimal: solution.is_optimal,
            is_infeasible: solution.is_infeasible,
            transitions: solution
                .transitions
                .into_iter()
                .map(dypdl::Transition::from)
                .collect(),
            expanded: solution.expanded,
            generated: solution.generated,
            time: solution.time,
            time_out: solution.time_out,
        };

        Ok((solution, is_terminated))
    }
}
