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

/// Complete Anytime Beam Search.
/// It iterates beam search with exponentially increasing beam width.
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
