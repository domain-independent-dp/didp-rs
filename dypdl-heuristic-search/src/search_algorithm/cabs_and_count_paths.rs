use crate::Parameters;

use super::beam_search::BeamSearchParameters;
use super::cabs::Cabs;
use super::count_paths::{count_paths, sample_paths};
use super::data_structure::beam::{BeamInterface, InformationInBeam};
use super::data_structure::state_registry::StateInRegistry;
use super::data_structure::{
    BeamSearchProblemInstance, CustomCostNodeInterface, SuccessorGenerator, TransitionConstraints,
    TransitionWithCustomCost,
};
use super::rollout::get_trace;
use super::search::{Search, Solution};
use super::util;
use dypdl::variable_type;
use rand::prelude::*;
use rand_pcg::Pcg64Mcg;
use std::error::Error;
use std::fmt;
use std::fs::OpenOptions;
use std::io::Write;
use std::marker::PhantomData;
use std::rc::Rc;
use std::str;

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct CountParameters {
    pub depth: usize,
    pub sample_size: usize,
    pub seed: u64,
    pub filename: String,
}

pub struct CabsAndCountPaths<T, I, B, C, H, F>
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
    maximize: bool,
    f_pruning: bool,
    keep_all_layers: bool,
    primal_bound: Option<T>,
    quiet: bool,
    beam_size: usize,
    rng: Pcg64Mcg,
    filename: String,
    depth: usize,
    sample_size: usize,
    time_keeper: util::TimeKeeper,
    solution: Solution<T, TransitionWithCustomCost>,
    phantom: PhantomData<I>,
}

impl<T, I, B, C, H, F> CabsAndCountPaths<T, I, B, C, H, F>
where
    T: variable_type::Numeric + fmt::Display + Ord,
    <T as str::FromStr>::Err: fmt::Debug,
    I: InformationInBeam<T, T> + CustomCostNodeInterface<T, T>,
    B: BeamInterface<T, T, I>,
    C: Fn(usize) -> B,
    H: Fn(&StateInRegistry, &dypdl::Model) -> Option<T>,
    F: Fn(T, T, &StateInRegistry, &dypdl::Model) -> T,
{
    /// Create a new LNBS solver.
    pub fn new(
        generator: SuccessorGenerator<TransitionWithCustomCost, Rc<TransitionWithCustomCost>>,
        h_evaluator: H,
        f_evaluator: F,
        beam_constructor: C,
        count_parameters: CountParameters,
        parameters: BeamSearchParameters<T, T>,
    ) -> CabsAndCountPaths<T, I, B, C, H, F> {
        let time_keeper = parameters
            .parameters
            .time_limit
            .map_or_else(util::TimeKeeper::default, |time_limit| {
                util::TimeKeeper::with_time_limit(time_limit)
            });
        let transition_constraints =
            TransitionConstraints::new(&generator.model, generator.backward);
        CabsAndCountPaths {
            generator,
            transition_constraints,
            h_evaluator,
            f_evaluator,
            beam_constructor,
            maximize: parameters.maximize,
            f_pruning: parameters.f_pruning,
            keep_all_layers: parameters.keep_all_layers,
            primal_bound: parameters.parameters.primal_bound,
            quiet: parameters.parameters.quiet,
            beam_size: parameters.beam_size,
            rng: Pcg64Mcg::seed_from_u64(count_parameters.seed),
            filename: count_parameters.filename,
            depth: count_parameters.depth,
            sample_size: count_parameters.sample_size,
            time_keeper,
            solution: Solution::default(),
            phantom: PhantomData::default(),
        }
    }

    pub fn search_inner(&mut self) -> (Solution<T, TransitionWithCustomCost>, bool) {
        if self.solution.is_terminated() {
            return (self.solution.clone(), true);
        }

        self.search_initial_solution()
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

        let (solution, _) = cabs.search_inner();
        self.solution = solution;

        if let Some(cost) = self.solution.cost {
            if !self.quiet {
                println!("Initial primal bound: {}", cost);
            }

            let max_depth = self.solution.transitions.len();

            let mut file = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&self.filename)
                .unwrap();

            file.write_all(format!("{} {}\n", max_depth, cost).as_bytes())
                .unwrap();

            let (states, costs): (Vec<_>, Vec<_>) = get_trace(
                &self.generator.model.target,
                T::zero(),
                &self.solution.transitions,
                &self.generator.model,
            )
            .unzip();

            let start = 0;
            let depth = std::cmp::min(self.depth, max_depth - start);

            let prefix = &self.solution.transitions[..start];
            let prefix_cost = if start == 0 {
                T::zero()
            } else {
                costs[start - 1]
            };
            let target = StateInRegistry::from(if start == 0 {
                self.generator.model.target.clone()
            } else {
                states[start - 1].clone()
            });
            let suffix = &self.solution.transitions[start + depth..];
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

            let (result, infeasible) = if depth <= 10 {
                count_paths(&problem)
            } else {
                sample_paths(&problem, self.sample_size, &mut self.rng)
            };

            file.write_all(format!("{} {}", depth, infeasible).as_bytes())
                .unwrap();

            for (cost, count) in result {
                file.write_all(format!(" {} {}", cost, count).as_bytes())
                    .unwrap();
            }

            file.write_all("\n".as_bytes()).unwrap();
        }

        (self.solution.clone(), true)
    }
}

impl<T, I, B, C, H, F> Search<T> for CabsAndCountPaths<T, I, B, C, H, F>
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
