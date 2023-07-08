use super::data_structure::BeamSearchProblemInstance;
use super::rollout::rollout;
use dypdl::variable_type::Numeric;
use dypdl::TransitionInterface;
use rand::distributions::Uniform;
use rand::prelude::*;
use rand_pcg::Pcg64Mcg;
use std::collections::BTreeMap;

pub fn count_paths<T, U>(problem: &BeamSearchProblemInstance<T, U>) -> (BTreeMap<U, i32>, usize)
where
    T: Numeric,
    U: Numeric + Ord,
{
    let model = &problem.generator.model;
    let initial_state = problem.target.clone();
    let cost = U::zero();
    let generator = &problem.generator;
    let mut stack = vec![(initial_state, cost)];
    let mut result = BTreeMap::default();
    let mut infeasible = 0;

    while let Some((state, cost)) = stack.pop() {
        if let Some(rollout_result) = rollout(&state, cost, problem.solution_suffix, model) {
            if rollout_result.is_base {
                result
                    .entry(rollout_result.cost)
                    .and_modify(|e| *e += 1)
                    .or_insert(1);
                continue;
            }
        }

        let mut dead_end = true;

        for transition in generator.applicable_transitions(&state) {
            if dead_end {
                dead_end = false;
            }

            let next_state = transition.apply(&state, &model.table_registry);

            if !model.check_constraints(&next_state) {
                infeasible += 1;
                continue;
            }

            let cost = transition.eval_cost(cost, &state, &model.table_registry);
            stack.push((next_state, cost));
        }

        if dead_end {
            infeasible += 1;
        }
    }

    (result, infeasible)
}

pub fn sample_paths<T, U>(
    problem: &BeamSearchProblemInstance<T, U>,
    sample_size: usize,
    rng: &mut Pcg64Mcg,
) -> (BTreeMap<U, i32>, usize)
where
    T: Numeric,
    U: Numeric + Ord,
{
    let mut result = BTreeMap::default();
    let mut infeasible = 0;

    for _ in 0..sample_size {
        if let Some(cost) = sample_path(problem, rng) {
            result.entry(cost).and_modify(|e| *e += 1).or_insert(1);
        } else {
            infeasible += 1;
        }
    }

    (result, infeasible)
}

pub fn sample_path<T, U>(problem: &BeamSearchProblemInstance<T, U>, rng: &mut Pcg64Mcg) -> Option<U>
where
    T: Numeric,
    U: Numeric + Ord,
{
    let model = &problem.generator.model;
    let generator = &problem.generator;
    let mut state = problem.target.clone();
    let mut cost = U::zero();

    loop {
        if !model.check_constraints(&state) {
            return None;
        }

        if let Some(result) = rollout(&state, cost, problem.solution_suffix, model) {
            if result.is_base {
                return Some(result.cost);
            }
        }

        let transitions: Vec<_> = generator.applicable_transitions(&state).collect();

        if transitions.is_empty() {
            return None;
        }

        let index = Uniform::from(0..transitions.len()).sample(rng);
        let transition = &transitions[index];

        cost = transition.eval_cost(cost, &state, &model.table_registry);
        state = transition.apply(&state, &model.table_registry);
    }
}
