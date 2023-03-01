use super::beam_search::{beam_search, BeamSearchParameters};
use super::data_structure::beam::{BeamInterface, InformationInBeam};
use super::data_structure::state_registry::StateInRegistry;
use super::data_structure::{
    BeamSearchProblemInstance, CustomCostNodeInterface, SuccessorGenerator,
    TransitionWithCustomCost,
};
use super::search::{Search, Solution};
use super::util;
use dypdl::variable_type;
use std::error::Error;
use std::fmt;
use std::marker::PhantomData;
use std::str;

/// Complete Anytime Beam Search (CABS).
///
/// It iterates beam search with exponentially increasing beam width.
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
/// Ryo Kuroiwa and J. Christopher Beck. "Solving Domain-Independent Dynamic Programming with Anytime Heuristic Search,""
/// Proceedings of the 33rd International Conference on Automated Planning and Scheduling (ICAPS), 2023.
///
/// Weixiong Zhang. "Complete Anytime Beam Search,"
/// Proceedings of the 15th National Conference on Artificial Intelligence/Innovative Applications of Artificial Intelligence (AAAI/IAAI), pp. 425-430, 1998.
///
/// # Examples
///
/// ```
/// use dypdl::prelude::*;
/// use dypdl_heuristic_search::search_algorithm::data_structure::beam::Beam;
/// use dypdl_heuristic_search::search_algorithm::{BeamSearchParameters, Cabs, Search};
/// use dypdl_heuristic_search::search_algorithm::data_structure::BeamSearchNode;
/// use dypdl_heuristic_search::search_algorithm::data_structure::successor_generator::{
///     SuccessorGenerator
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
///
/// let h_evaluator = |_: &_, _: &_| Some(0);
/// let f_evaluator = |g, h, _: &_, _: &_| g + h;
///
/// let model = Rc::new(model);
/// let generator = SuccessorGenerator::from_model_without_custom_cost(model.clone(), false);
/// let beam_constructor = |beam_size| Beam::<_, _, BeamSearchNode<_, _>>::new(beam_size);
/// let parameters = BeamSearchParameters { beam_size: 1, ..Default::default() };
///
/// let mut solver = Cabs::new(
///     generator, h_evaluator, f_evaluator, beam_constructor, parameters
/// );
/// let solution = solver.search().unwrap();
/// assert_eq!(solution.cost, Some(1));
/// assert_eq!(solution.transitions, vec![increment]);
/// assert!(!solution.is_infeasible);
/// ```
pub struct Cabs<'a, T, I, B, C, H, F>
where
    T: variable_type::Numeric + fmt::Display + Ord,
    <T as str::FromStr>::Err: fmt::Debug,
    I: InformationInBeam<T, T> + CustomCostNodeInterface<T, T>,
    B: BeamInterface<T, T, I>,
    C: Fn(usize) -> B,
    H: Fn(&StateInRegistry, &dypdl::Model) -> Option<T>,
    F: Fn(T, T, &StateInRegistry, &dypdl::Model) -> T,
{
    problem: BeamSearchProblemInstance<'a, T, T>,
    h_evaluator: H,
    f_evaluator: F,
    beam_constructor: C,
    maximize: bool,
    f_pruning: bool,
    keep_all_layers: bool,
    primal_bound: Option<T>,
    quiet: bool,
    beam_size: usize,
    time_keeper: util::TimeKeeper,
    solution: Solution<T, TransitionWithCustomCost>,
    phantom: PhantomData<I>,
}

impl<'a, T, I, B, C, H, F> Cabs<'a, T, I, B, C, H, F>
where
    T: variable_type::Numeric + fmt::Display + Ord,
    <T as str::FromStr>::Err: fmt::Debug,
    I: InformationInBeam<T, T> + CustomCostNodeInterface<T, T>,
    B: BeamInterface<T, T, I>,
    C: Fn(usize) -> B,
    H: Fn(&StateInRegistry, &dypdl::Model) -> Option<T>,
    F: Fn(T, T, &StateInRegistry, &dypdl::Model) -> T,
{
    /// Create a new CABS solver.
    pub fn new(
        generator: SuccessorGenerator<TransitionWithCustomCost>,
        h_evaluator: H,
        f_evaluator: F,
        beam_constructor: C,
        parameters: BeamSearchParameters<T, T>,
    ) -> Cabs<'a, T, I, B, C, H, F> {
        let time_keeper = parameters
            .parameters
            .time_limit
            .map_or_else(util::TimeKeeper::default, util::TimeKeeper::with_time_limit);
        let target = StateInRegistry::from(generator.model.target.clone());
        let problem = BeamSearchProblemInstance {
            generator,
            target,
            cost: T::zero(),
            g: T::zero(),
            solution_suffix: &[],
        };
        Cabs {
            problem,
            h_evaluator,
            f_evaluator,
            beam_constructor,
            maximize: parameters.maximize,
            f_pruning: parameters.f_pruning,
            keep_all_layers: parameters.keep_all_layers,
            primal_bound: parameters.parameters.primal_bound,
            quiet: parameters.parameters.quiet,
            beam_size: parameters.beam_size,
            time_keeper,
            solution: Solution::default(),
            phantom: PhantomData::default(),
        }
    }

    //// Search for the next solution, returning the solution using `TransitionWithCustomCost`.
    pub fn search_inner(&mut self) -> (Solution<T, TransitionWithCustomCost>, bool) {
        while !self.solution.is_terminated() {
            if !self.quiet {
                println!(
                    "Beam size: {}, expanded: {}",
                    self.beam_size, self.solution.expanded
                );
            }

            let beam_constructor = |beam_size| (self.beam_constructor)(beam_size);
            let parameters = BeamSearchParameters {
                beam_size: self.beam_size,
                maximize: self.maximize,
                keep_all_layers: self.keep_all_layers,
                f_pruning: self.f_pruning,
                f_bound: self.primal_bound,
                parameters: util::Parameters {
                    primal_bound: self.primal_bound,
                    get_all_solutions: false,
                    quiet: true,
                    time_limit: self.time_keeper.remaining_time_limit(),
                },
            };
            let result = beam_search(
                &self.problem,
                &beam_constructor,
                &self.h_evaluator,
                &self.f_evaluator,
                parameters,
            );
            self.solution.expanded += result.expanded;
            self.solution.generated += result.generated;
            self.beam_size *= 2;

            match result.cost {
                Some(new_cost) => {
                    let found_better = self.solution.cost.map_or(true, |current_cost| {
                        match self.problem.generator.model.reduce_function {
                            dypdl::ReduceFunction::Max => new_cost > current_cost,
                            dypdl::ReduceFunction::Min => new_cost < current_cost,
                            _ => false,
                        }
                    });

                    if found_better || result.is_optimal {
                        self.solution.transitions = result.transitions;
                        self.solution.cost = Some(new_cost);
                        self.primal_bound = Some(new_cost);
                        self.solution.is_optimal = result.is_optimal;
                        self.solution.time = self.time_keeper.elapsed_time();

                        if !self.quiet {
                            println!(
                                "New primal bound: {}, expanded: {}",
                                new_cost, self.solution.expanded
                            );
                        }

                        return (self.solution.clone(), result.is_optimal);
                    }
                }
                _ => {
                    if !self.quiet {
                        println!("Failed to find a solution.");
                    }

                    if result.is_infeasible {
                        self.solution.is_optimal = self.solution.cost.is_some();
                        self.solution.is_infeasible = self.solution.cost.is_none();
                        self.solution.time = self.time_keeper.elapsed_time();
                    }
                }
            }

            if result.time_out {
                if !self.quiet {
                    println!("Reached time limit.");
                }

                self.solution.time = self.time_keeper.elapsed_time();
                self.solution.time_out = true;
            }
        }

        (self.solution.clone(), true)
    }
}

impl<'a, T, I, B, C, H, F> Search<T> for Cabs<'a, T, I, B, C, H, F>
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
