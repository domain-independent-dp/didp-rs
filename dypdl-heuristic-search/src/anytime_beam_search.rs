use super::beam_search::{beam_search, BeamSearchParameters, EvaluatorsForBeamSearch};
use crate::beam::{Beam, InBeam, PrioritizedNode};
use crate::evaluator;
use crate::search_node::DPSearchNode;
use crate::solver;
use crate::solver::Solution;
use crate::state_registry::{StateInRegistry, StateInformation};
use crate::successor_generator::SuccessorGenerator;
use crate::transition_with_custom_cost::TransitionWithCustomCost;
use dypdl::{variable_type, ReduceFunction};
use std::fmt;
use std::str;

/// Performs multiple iterations of forward beam search using given beam sizes.
pub fn anytime_beam_search<'a, T, V, B, C, H, F>(
    model: &'a dypdl::Model,
    generator: &'a SuccessorGenerator<'a, TransitionWithCustomCost>,
    evaluators: &EvaluatorsForBeamSearch<H, F>,
    beam_constructor: &C,
    pruning: bool,
    callback: &mut Box<solver::Callback<T>>,
    parameters: solver::SolverParameters<T>,
) -> Solution<T>
where
    T: variable_type::Numeric + fmt::Display + Ord,
    <T as str::FromStr>::Err: fmt::Debug,
    V: DPSearchNode<T> + InBeam + Ord + StateInformation<T> + PrioritizedNode<T>,
    B: Beam<T, T, V>,
    C: Fn(usize) -> B,
    H: evaluator::Evaluator,
    F: Fn(T, T, &StateInRegistry, &dypdl::Model) -> T,
{
    let time_keeper = parameters.time_limit.map_or_else(
        solver::TimeKeeper::default,
        solver::TimeKeeper::with_time_limit,
    );
    let maximize = model.reduce_function == ReduceFunction::Max;
    let quiet = parameters.quiet;
    let mut solution = Solution::default();
    let mut beam_size = 1;
    loop {
        if !quiet {
            println!("Beam size: {}", beam_size);
        }
        let beam_constructor = || beam_constructor(beam_size);
        let parameters = BeamSearchParameters {
            maximize,
            primal_bound: if pruning { solution.cost } else { None },
            quiet: true,
        };
        let (result, time_out) = beam_search(
            model,
            generator,
            &beam_constructor,
            evaluators,
            parameters,
            &time_keeper,
        );
        solution.expanded += result.expanded;
        solution.generated += result.expanded;
        match result.cost {
            Some(new_cost) => {
                if let Some(current_cost) = solution.cost {
                    match model.reduce_function {
                        dypdl::ReduceFunction::Max if new_cost > current_cost => {
                            solution.transitions = result.transitions;
                            solution.cost = Some(new_cost);
                            (callback)(&solution);
                            if !quiet {
                                println!("New primal bound: {}", new_cost);
                            }
                        }
                        dypdl::ReduceFunction::Min if new_cost < current_cost => {
                            solution.transitions = result.transitions;
                            solution.cost = Some(new_cost);
                            (callback)(&solution);
                            if !quiet {
                                println!("New primal bound: {}", new_cost);
                            }
                        }
                        _ => {}
                    }
                } else {
                    solution.transitions = result.transitions;
                    solution.cost = Some(new_cost);
                    (callback)(&solution);
                    if !quiet {
                        println!("New primal bound: {}", new_cost);
                    }
                }
            }
            _ => {
                if !quiet {
                    println!("Failed to find a solution.")
                }
            }
        }
        if result.is_infeasible {
            solution.is_optimal = solution.cost.is_some();
            solution.is_infeasible = solution.cost.is_none();
            solution.time = time_keeper.elapsed_time();
            return solution;
        }
        if time_out {
            solution.time = time_keeper.elapsed_time();
            return solution;
        }
        beam_size *= 2;
    }
}
