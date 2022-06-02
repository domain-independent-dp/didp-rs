use crate::hashable_state;
use crate::solver;
use crate::successor_generator::SuccessorGenerator;
use didp_parser::variable;
use didp_parser::Transition;
use rustc_hash::FxHashMap;
use std::fmt;

#[derive(Clone, PartialEq, Default)]
pub struct DFSNode<T: variable::Numeric> {
    pub state: hashable_state::HashableState,
    pub cost: T,
}

pub fn forward_iterative_exist_dfs<T>(
    model: &didp_parser::Model<T>,
    generator: &SuccessorGenerator<Transition<T>>,
    mut primal_bound: Option<T>,
    maximize: bool,
    capacity: Option<usize>,
) -> solver::Solution<T>
where
    T: variable::Numeric + fmt::Display,
{
    let mut expanded = 0;
    let mut prob = FxHashMap::default();
    if let Some(capacity) = capacity {
        prob.reserve(capacity);
    };
    let mut incumbent = Vec::new();
    let node = DFSNode {
        state: hashable_state::HashableState::new(&model.target),
        cost: T::zero(),
    };
    while let Some((new_cost, transitions)) = exist_dfs(
        node.clone(),
        model,
        generator,
        &mut prob,
        primal_bound,
        maximize,
        &mut expanded,
    ) {
        if let Some(current_cost) = primal_bound {
            match model.reduce_function {
                didp_parser::ReduceFunction::Max if new_cost > current_cost => {
                    primal_bound = Some(new_cost);
                    incumbent = transitions;
                    println!("New primal bound: {}, expanded: {}", new_cost, expanded);
                }
                didp_parser::ReduceFunction::Min if new_cost < current_cost => {
                    primal_bound = Some(new_cost);
                    incumbent = transitions;
                    println!("New primal bound: {}, expanded: {}", new_cost, expanded);
                }
                _ => {}
            }
        } else {
            primal_bound = Some(new_cost);
            incumbent = transitions;
            println!("New primal bound: {}, expanded: {}", new_cost, expanded);
        }
    }
    println!("Expanded: {}", expanded);
    if let Some(cost) = primal_bound {
        incumbent.reverse();
        Some((cost, incumbent))
    } else {
        None
    }
}

pub fn exist_dfs<T: variable::Numeric>(
    node: DFSNode<T>,
    model: &didp_parser::Model<T>,
    generator: &SuccessorGenerator<Transition<T>>,
    prob: &mut FxHashMap<hashable_state::HashableState, T>,
    primal_bound: Option<T>,
    maximize: bool,
    expanded: &mut u32,
) -> solver::Solution<T> {
    let state = node.state;
    let cost = node.cost;
    *expanded += 1;
    if model.is_goal(&state) {
        if maximize && primal_bound.is_some() && cost <= primal_bound.unwrap() {
            return None;
        }
        return Some((cost, Vec::new()));
    }
    if let Some(other_cost) = prob.get(&state) {
        if (maximize && cost <= *other_cost) || (!maximize && cost >= *other_cost) {
            return None;
        } else {
            prob.remove(&state);
        }
    }
    for transition in generator.applicable_transitions(&state) {
        let cost = transition.eval_cost(cost, &state, &model.table_registry);
        if !maximize && primal_bound.is_some() && cost >= primal_bound.unwrap() {
            continue;
        }
        let successor = transition.apply(&state, &model.table_registry);
        if model.check_constraints(&successor) {
            let node = DFSNode {
                state: successor,
                cost,
            };
            let result = exist_dfs(
                node,
                model,
                generator,
                prob,
                primal_bound,
                maximize,
                expanded,
            );
            if let Some((cost, mut transitions)) = result {
                transitions.push(transition);
                return Some((cost, transitions));
            }
        }
    }
    prob.insert(state, cost);
    None
}
