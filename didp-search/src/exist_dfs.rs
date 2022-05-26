use crate::hashable_state;
use crate::solver;
use crate::successor_generator::SuccessorGenerator;
use didp_parser::variable;
use didp_parser::Transition;
use rustc_hash::FxHashMap;
use std::fmt;
use std::rc::Rc;

#[derive(Clone, PartialEq, Default)]
pub struct DFSNode<U: variable::Numeric> {
    pub state: hashable_state::HashableState,
    pub g: U,
}

pub fn forward_iterative_exist_dfs<T>(
    model: &didp_parser::Model<T>,
    generator: &SuccessorGenerator<didp_parser::Transition<T>>,
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
    let mut cost = None;
    let node = DFSNode {
        state: hashable_state::HashableState::new(&model.target),
        g: T::zero(),
    };
    while let Some((new_cost, transitions)) = exist_dfs(
        node.clone(),
        model,
        &generator,
        &mut prob,
        primal_bound,
        maximize,
        &mut expanded,
    ) {
        if let Some(current_cost) = cost {
            match model.reduce_function {
                didp_parser::ReduceFunction::Max if new_cost > current_cost => {
                    cost = Some(new_cost);
                    incumbent = transitions;
                    println!("New primal bound: {}, expanded: {}", new_cost, expanded);
                }
                didp_parser::ReduceFunction::Min if new_cost < current_cost => {
                    cost = Some(new_cost);
                    incumbent = transitions;
                    println!("New primal bound: {}, expanded: {}", new_cost, expanded);
                }
                _ => {}
            }
        } else {
            cost = Some(new_cost);
            incumbent = transitions;
            println!("New primal bound: {}, expanded: {}", new_cost, expanded);
        }
    }
    println!("Expanded: {}", expanded);
    if let Some(cost) = cost {
        incumbent.reverse();
        Some((cost, incumbent))
    } else {
        None
    }
}

pub type ExistDFSSolution<T> = Option<(T, Vec<Rc<Transition<T>>>)>;

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
    let g = node.g;
    *expanded += 1;
    if model.get_base_cost(&state).is_some() {
        if maximize && primal_bound.is_some() && g <= primal_bound.unwrap() {
            return None;
        }
        return Some((g, Vec::new()));
    }
    if let Some(other_g) = prob.get(&state) {
        if (maximize && g <= *other_g) || (!maximize && g >= *other_g) {
            return None;
        } else {
            prob.remove(&state);
        }
    }
    for transition in generator.applicable_transitions(&state) {
        let g = transition.eval_cost(g, &state, &model.table_registry);
        if !maximize && primal_bound.is_some() && g >= primal_bound.unwrap() {
            continue;
        }
        let mut successor = transition.apply(&state, &model.table_registry);
        if model.check_constraints(&successor) {
            let g = model.apply_forced_transitions_in_place(&mut successor, g, false);
            let node = DFSNode {
                state: successor,
                g,
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
            if let Some((g, mut transitions)) = result {
                transitions.push(transition);
                return Some((g, transitions));
            }
        }
    }
    prob.insert(state, g);
    None
}
