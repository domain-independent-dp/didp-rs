use super::beam_search::BeamSearchParameters;
use super::cabs::Cabs;
use super::data_structure::beam::{BeamInterface, InformationInBeam};
use super::data_structure::state_registry::StateInRegistry;
use super::data_structure::{
    exceed_bound, BeamSearchProblemInstance, CustomCostNodeInterface, SuccessorGenerator,
    TransitionConstraints, TransitionWithCustomCost,
};
use super::restricted_dd::restricted_dd;
use super::rollout::get_trace;
use super::search::{Search, Solution};
use super::util;
use crate::Parameters;
use dypdl::variable_type;
use rand::prelude::*;
use rand_pcg::Pcg64Mcg;
use std::error::Error;
use std::fmt;
use std::marker::PhantomData;
use std::rc::Rc;
use std::str;

/// Parameters for DD-LNS
#[derive(Clone, PartialEq, Debug)]
pub struct DdLnsParameters {
    /// Beam size.
    pub beam_size: usize,
    /// Probability to keep a non-best state.
    pub keep_probability: f64,
    /// Random seed.
    pub seed: u64,
}

/// Large Neighborhood Search with Decision Diagrams (DD-LNS).
///
/// It performs LNS by constructing restricted multi-valued decision diagrams (MDD).
///
/// This solver uses forward search based on the shortest path problem.
/// It only works with problems where the cost expressions are in the form of `cost + w`, `cost * w`, `max(cost, w)`, or `min(cost, w)`
/// where `cost` is `IntegerExpression::Cost`or `ContinuousExpression::Cost` and `w` is a numeric expression independent of `cost`.
///
/// It uses `h_evaluator` and `f_evaluator` for pruning.
/// If `h_evaluator` returns `None`, the state is pruned.
/// If `parameters.f_pruning` and `f_evaluator` returns a value that exceeds the f bound, the state is pruned.
///
/// Beam search searches layer by layer, where the i th layer contains states that can be reached with i transitions.
/// By default, this solver only keeps states in the current layer to check for duplicates.
/// If `parameters.keep_all_layers` is `true`, this solver keeps states in all layers to check for duplicates.
///
/// # References
///
/// Xavier Gillard and Pierre Schaus. "Solving Domain-Independent Dynamic Programming with Anytime Heuristic Search,"
/// Proceedings of the 31st International Joint Conference on Artificial Intelligence (IJCAI), 2022.
///
/// # Examples
///
/// ```
/// use dypdl::prelude::*;
/// use dypdl_heuristic_search::search_algorithm::data_structure::beam::Beam;
/// use dypdl_heuristic_search::search_algorithm::{
///     BeamSearchParameters, DdLns, DdLnsParameters, Search,
/// };
/// use dypdl_heuristic_search::search_algorithm::data_structure::BeamSearchNode;
/// use dypdl_heuristic_search::search_algorithm::data_structure::successor_generator::{
///     SuccessorGenerator
/// };
/// use dypdl_heuristic_search::Parameters;
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
///
/// let h_evaluator = |_: &_, _: &_| Some(0);
/// let f_evaluator = |g, h, _: &_, _: &_| g + h;
///
/// let model = Rc::new(model);
/// let generator = SuccessorGenerator::from_model_without_custom_cost(model.clone(), false);
/// let beam_constructor = |beam_size| Beam::<_, _, BeamSearchNode<_, _>>::new(beam_size);
/// let dd_lns_parameters = DdLnsParameters {
///     beam_size: 10000,
///     keep_probability: 0.1,
///     seed: 0,
/// };
/// let parameters = BeamSearchParameters {
///     beam_size: 1,
///     parameters: Parameters {
///         time_limit: Some(1800.0),
///         ..Default::default()
///     },
///     ..Default::default()
/// };
///
/// let mut solver = DdLns::new(
///     generator, h_evaluator, f_evaluator, beam_constructor, dd_lns_parameters, parameters
/// );
/// let solution = solver.search().unwrap();
/// assert_eq!(solution.cost, Some(1));
/// assert_eq!(solution.transitions, vec![increment]);
/// assert!(!solution.is_infeasible);
/// ```
pub struct DdLns<T, I, B, C, H, F>
where
    T: variable_type::Numeric + fmt::Display + Ord,
    <T as str::FromStr>::Err: fmt::Debug,
    I: InformationInBeam<T, T> + CustomCostNodeInterface<T, T>,
    B: BeamInterface<T, T, I>,
    C: Fn(usize) -> B,
    H: Fn(&StateInRegistry, &dypdl::Model) -> Option<T>,
    F: Fn(T, T, &StateInRegistry, &dypdl::Model) -> T,
{
    generator: SuccessorGenerator<TransitionWithCustomCost, Rc<TransitionWithCustomCost>>,
    transition_constraints: TransitionConstraints,
    h_evaluator: H,
    f_evaluator: F,
    beam_constructor: C,
    parameters: DdLnsParameters,
    maximize: bool,
    f_pruning: bool,
    keep_all_layers: bool,
    primal_bound: Option<T>,
    quiet: bool,
    beam_size: usize,
    rng: Pcg64Mcg,
    time_keeper: util::TimeKeeper,
    solution: Solution<T, TransitionWithCustomCost>,
    phantom: PhantomData<I>,
}

impl<T, I, B, C, H, F> DdLns<T, I, B, C, H, F>
where
    T: variable_type::Numeric + fmt::Display + Ord,
    <T as str::FromStr>::Err: fmt::Debug,
    I: InformationInBeam<T, T> + CustomCostNodeInterface<T, T>,
    B: BeamInterface<T, T, I>,
    C: Fn(usize) -> B,
    H: Fn(&StateInRegistry, &dypdl::Model) -> Option<T>,
    F: Fn(T, T, &StateInRegistry, &dypdl::Model) -> T,
{
    /// Create a new DD-LNS solver.
    pub fn new(
        generator: SuccessorGenerator<TransitionWithCustomCost, Rc<TransitionWithCustomCost>>,
        h_evaluator: H,
        f_evaluator: F,
        beam_constructor: C,
        dd_lns_parameters: DdLnsParameters,
        parameters: BeamSearchParameters<T, T>,
    ) -> DdLns<T, I, B, C, H, F> {
        let time_keeper = parameters
            .parameters
            .time_limit
            .map_or_else(util::TimeKeeper::default, util::TimeKeeper::with_time_limit);
        let transition_constraints =
            TransitionConstraints::new(&generator.model, generator.backward);
        let seed = dd_lns_parameters.seed;
        DdLns {
            generator,
            transition_constraints,
            h_evaluator,
            f_evaluator,
            beam_constructor,
            parameters: dd_lns_parameters,
            maximize: parameters.maximize,
            f_pruning: parameters.f_pruning,
            keep_all_layers: parameters.keep_all_layers,
            primal_bound: parameters.parameters.primal_bound,
            quiet: parameters.parameters.quiet,
            beam_size: parameters.beam_size,
            rng: Pcg64Mcg::seed_from_u64(seed),
            time_keeper,
            solution: Solution::default(),
            phantom: PhantomData::default(),
        }
    }

    fn search_initial_solution(&mut self) -> (Solution<T, TransitionWithCustomCost>, bool) {
        let parameters = BeamSearchParameters {
            maximize: self.maximize,
            f_pruning: self.f_pruning,
            f_bound: self.primal_bound,
            keep_all_layers: self.keep_all_layers,
            beam_size: self.beam_size,
            parameters: Parameters {
                primal_bound: self.primal_bound,
                get_all_solutions: false,
                quiet: true,
                time_limit: self.time_keeper.remaining_time_limit(),
            },
        };

        let mut cabs = Cabs::new(
            self.generator.clone(),
            &self.h_evaluator,
            &self.f_evaluator,
            &self.beam_constructor,
            parameters,
        );

        let (solution, is_terminated) = cabs.search_inner();
        self.solution = solution;

        if let Some(cost) = self.solution.cost {
            if !self.quiet {
                println!("Initial primal bound: {}", cost);
            }
        }

        (self.solution.clone(), is_terminated)
    }

    //// Search for the next solution, returning the solution using `TransitionWithCustomCost`.
    pub fn search_inner(&mut self) -> (Solution<T, TransitionWithCustomCost>, bool) {
        if self.solution.is_terminated() {
            return (self.solution.clone(), true);
        }

        if self.solution.cost.is_none() {
            return self.search_initial_solution();
        }

        if self.solution.transitions.len() < 2 {
            return (self.solution.clone(), true);
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
            let suffix = &[];
            let generator = self.transition_constraints.filter_successor_generator(
                &self.generator,
                prefix,
                suffix,
            );
            let problem = BeamSearchProblemInstance {
                target,
                generator,
                cost: prefix_cost,
                g: prefix_cost,
                solution_suffix: suffix,
            };

            let parameters = BeamSearchParameters {
                beam_size: self.parameters.beam_size,
                maximize: self.maximize,
                f_pruning: self.f_pruning,
                f_bound: self.solution.cost,
                keep_all_layers: self.keep_all_layers,
                parameters: Parameters {
                    primal_bound: self.solution.cost,
                    get_all_solutions: false,
                    quiet: true,
                    time_limit: self.time_keeper.remaining_time_limit(),
                },
            };

            let solution = restricted_dd(
                &problem,
                &self.h_evaluator,
                &self.f_evaluator,
                self.parameters.keep_probability,
                Some(&self.solution.transitions),
                &mut self.rng,
                parameters,
            );

            self.solution.expanded += solution.expanded;
            self.solution.generated += solution.generated;

            if let Some(cost) = solution.cost {
                if !exceed_bound(&self.generator.model, cost, self.solution.cost) {
                    if !self.quiet {
                        println!("New primal bound: {}, d: {}", cost, d);
                    }

                    if d == 0 {
                        self.solution.is_optimal = solution.is_optimal
                    }

                    let mut transitions = prefix.to_vec();
                    transitions.extend(solution.transitions.into_iter());
                    self.solution.transitions = transitions;

                    self.solution.cost = Some(cost);
                    self.solution.time = self.time_keeper.elapsed_time();

                    return (self.solution.clone(), self.solution.is_optimal);
                }
            }

            if d == 0 {
                if solution.is_infeasible {
                    self.solution.is_optimal = self.solution.cost.is_some();
                    self.solution.is_infeasible = self.solution.cost.is_none();

                    return (self.solution.clone(), true);
                }

                d = max_depth;
            } else {
                d -= 1;
            }

            if solution.time_out {
                self.solution.time = self.time_keeper.elapsed_time();

                return (self.solution.clone(), true);
            }
        }
    }
}

impl<T, I, B, C, H, F> Search<T> for DdLns<T, I, B, C, H, F>
where
    T: variable_type::Numeric + fmt::Display + Ord,
    <T as str::FromStr>::Err: fmt::Debug,
    I: InformationInBeam<T, T> + CustomCostNodeInterface<T, T>,
    B: BeamInterface<T, T, I>,
    C: Fn(usize) -> B,
    H: Fn(&StateInRegistry, &dypdl::Model) -> Option<T>,
    F: Fn(T, T, &StateInRegistry, &dypdl::Model) -> T,
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
