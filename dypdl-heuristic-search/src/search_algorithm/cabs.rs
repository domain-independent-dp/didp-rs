use super::beam_search::BeamSearchParameters;
use super::data_structure::{exceed_bound, HashableSignatureVariables};
use super::search::{Parameters, Search, SearchInput, Solution};
use super::util::print_primal_bound;
use super::util::{update_bound_if_better, TimeKeeper};
use dypdl::{variable_type, Model, Transition, TransitionInterface};
use std::error::Error;
use std::fmt;
use std::fmt::Debug;
use std::hash::Hash;
use std::marker::PhantomData;
use std::ops::Deref;
use std::rc::Rc;
use std::str;

/// Parameters for CABS.
#[derive(Debug, PartialEq, Clone, Copy, Default)]
pub struct CabsParameters<T> {
    /// Maximum beam size.
    pub max_beam_size: Option<usize>,
    /// Parameters for beam search.
    pub beam_search_parameters: BeamSearchParameters<T>,
}

/// Complete Anytime Beam Search (CABS).
///
/// It iterates beam search with exponentially increasing beam width.
///
/// This solver uses forward search based on the shortest path problem.
/// It only works with problems where the cost expressions are in the form of `cost + w`, `cost * w`, `max(cost, w)`, or `min(cost, w)`
/// where `cost` is `IntegerExpression::Cost`or `ContinuousExpression::Cost` and `w` is a numeric expression independent of `cost`.
///
/// Type parameter `B` is a type of a function that performs beam search.
/// The function takes a `SearchInput` and `BeamSearchParameters` and returns a `Solution`.
///
/// # References
///
/// Ryo Kuroiwa and J. Christopher Beck. "Solving Domain-Independent Dynamic Programming with Anytime Heuristic Search,"
/// Proceedings of the 33rd International Conference on Automated Planning and Scheduling (ICAPS), pp. 245-253, 2023.
///
/// Weixiong Zhang. "Complete Anytime Beam Search,"
/// Proceedings of the 15th National Conference on Artificial Intelligence/Innovative Applications of Artificial Intelligence (AAAI/IAAI), pp. 425-430, 1998.
///
/// # Examples
///
/// ```
/// use dypdl::prelude::*;
/// use dypdl_heuristic_search::Search;
/// use dypdl_heuristic_search::search_algorithm::{
///     beam_search, BeamSearchParameters, Cabs, CabsParameters, FNode, SearchInput,
///     SuccessorGenerator,
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
/// let transition_evaluator = move |node: &FNode<_>, transition, primal_bound| {
///     node.generate_successor_node(
///         transition,
///         &model,
///         &h_evaluator,
///         &f_evaluator,
///         primal_bound,
///     )
/// };
/// let base_cost_evaluator = |cost, base_cost| cost + base_cost;
/// let beam_search = move |input: &SearchInput<_, _>, parameters| {
///     beam_search(input, &transition_evaluator, base_cost_evaluator, parameters)
/// };
/// let parameters = CabsParameters::default();
/// let input = SearchInput {
///     node,
///     generator,
///     solution_suffix: &[],
/// };
///
/// let mut solver = Cabs::<_, FNode<_>, _>::new(input, beam_search, parameters);
/// let solution = solver.search().unwrap();
/// assert_eq!(solution.cost, Some(1));
/// assert_eq!(solution.transitions, vec![increment]);
/// assert!(!solution.is_infeasible);
/// ```
pub struct Cabs<
    'a,
    T,
    N,
    B,
    V = Transition,
    D = Rc<V>,
    R = Rc<Model>,
    K = Rc<HashableSignatureVariables>,
> where
    T: variable_type::Numeric + fmt::Display,
    <T as str::FromStr>::Err: fmt::Debug,
    B: Fn(&SearchInput<N, V, D, R>, BeamSearchParameters<T>) -> Solution<T, V>,
    V: TransitionInterface + Clone + Default,
    Transition: From<V>,
    D: Deref<Target = V> + Clone,
    R: Deref<Target = Model> + Clone,
    K: Hash + Eq + Clone + Debug,
{
    input: SearchInput<'a, N, V, D, R>,
    beam_search: B,
    keep_all_layers: bool,
    primal_bound: Option<T>,
    quiet: bool,
    beam_size: usize,
    max_beam_size: Option<usize>,
    time_keeper: TimeKeeper,
    solution: Solution<T, V>,
    phantom: PhantomData<K>,
}

impl<'a, T, N, B, V, D, R, K> Cabs<'a, T, N, B, V, D, R, K>
where
    T: variable_type::Numeric + fmt::Display,
    <T as str::FromStr>::Err: fmt::Debug,
    B: Fn(&SearchInput<N, V, D, R>, BeamSearchParameters<T>) -> Solution<T, V>,
    V: TransitionInterface + Clone + Default,
    Transition: From<V>,
    D: Deref<Target = V> + Clone,
    R: Deref<Target = Model> + Clone,
    K: Hash + Eq + Clone + Debug,
{
    /// Create a new CABS solver.
    pub fn new(
        input: SearchInput<'a, N, V, D, R>,
        beam_search: B,
        parameters: CabsParameters<T>,
    ) -> Cabs<'a, T, N, B, V, D, R, K> {
        let mut time_keeper = parameters
            .beam_search_parameters
            .parameters
            .time_limit
            .map_or_else(TimeKeeper::default, TimeKeeper::with_time_limit);
        time_keeper.stop();

        Cabs {
            input,
            beam_search,
            keep_all_layers: parameters.beam_search_parameters.keep_all_layers,
            primal_bound: parameters.beam_search_parameters.parameters.primal_bound,
            quiet: parameters.beam_search_parameters.parameters.quiet,
            beam_size: parameters.beam_search_parameters.beam_size,
            max_beam_size: parameters.max_beam_size,
            time_keeper,
            solution: Solution::default(),
            phantom: PhantomData,
        }
    }

    //// Search for the next solution, returning the solution without converting it into `Transition`.
    pub fn search_inner(&mut self) -> (Solution<T, V>, bool) {
        self.time_keeper.start();
        let model = &self.input.generator.model;

        while !self.solution.is_terminated() {
            let last = self.max_beam_size.map_or(false, |max_beam_size| {
                if self.beam_size >= max_beam_size {
                    self.beam_size = max_beam_size;

                    if !self.quiet {
                        println!("Reached the maximum beam size.");
                    }

                    true
                } else {
                    false
                }
            });

            let parameters = BeamSearchParameters {
                beam_size: self.beam_size,
                keep_all_layers: self.keep_all_layers,
                parameters: Parameters {
                    primal_bound: self.primal_bound,
                    get_all_solutions: false,
                    quiet: true,
                    time_limit: self.time_keeper.remaining_time_limit(),
                    ..Default::default()
                },
            };
            let result = (self.beam_search)(&self.input, parameters);
            self.solution.expanded += result.expanded;
            self.solution.generated += result.generated;

            if !self.quiet {
                println!(
                    "Searched with beam size: {}, expanded: {}, elapsed time: {}",
                    self.beam_size,
                    self.solution.expanded,
                    self.time_keeper.elapsed_time()
                );
            }

            self.beam_size *= 2;

            if let Some(bound) = result.best_bound {
                self.solution.time = self.time_keeper.elapsed_time();
                update_bound_if_better(&mut self.solution, bound, model, self.quiet);
            }

            if let Some(cost) = result.cost {
                if !exceed_bound(model, cost, self.primal_bound) {
                    self.primal_bound = Some(cost);
                    self.solution.cost = Some(cost);
                    self.solution.transitions = result.transitions;
                    self.solution.is_optimal = result.is_optimal;
                    self.solution.time = self.time_keeper.elapsed_time();

                    if self.solution.is_optimal {
                        self.solution.best_bound = Some(cost);
                    }

                    if !self.quiet {
                        print_primal_bound(&self.solution);
                    }

                    self.time_keeper.stop();

                    return (self.solution.clone(), self.solution.is_optimal || last);
                }
            } else if result.is_infeasible {
                self.solution.is_optimal = self.solution.cost.is_some();
                self.solution.is_infeasible = self.solution.cost.is_none();
                self.solution.best_bound = self.solution.cost;
            }

            if last {
                break;
            }

            if result.time_out {
                if !self.quiet {
                    println!("Reached time limit.");
                }

                self.solution.time_out = true;
            }
        }

        self.solution.time = self.time_keeper.elapsed_time();
        self.time_keeper.stop();
        (self.solution.clone(), true)
    }
}

impl<'a, T, N, B, V, D, R, K> Search<T> for Cabs<'a, T, N, B, V, D, R, K>
where
    T: variable_type::Numeric + fmt::Display + Ord,
    <T as str::FromStr>::Err: fmt::Debug,
    B: Fn(&SearchInput<N, V, D, R>, BeamSearchParameters<T>) -> Solution<T, V>,
    V: TransitionInterface + Clone + Default,
    Transition: From<V>,
    D: Deref<Target = V> + Clone,
    R: Deref<Target = Model> + Clone,
    K: Hash + Eq + Clone + Debug,
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
