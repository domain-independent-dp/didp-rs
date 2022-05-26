use crate::bfs_node::TransitionWithG;
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

pub fn forward_iterative_exist_dfs<T, U>(
    model: &didp_parser::Model<T>,
    generator: &SuccessorGenerator<TransitionWithG<T, U>>,
    mut g_bound: Option<U>,
    maximize: bool,
    capacity: Option<usize>,
) -> solver::Solution<T>
where
    T: variable::Numeric + fmt::Display,
    U: variable::Numeric + fmt::Display,
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
        g: U::zero(),
    };
    while let Some((new_g_bound, new_cost, transitions)) = exist_dfs(
        node.clone(),
        model,
        &generator,
        &mut prob,
        g_bound,
        maximize,
        &mut expanded,
    ) {
        let mut transitions: Vec<Rc<Transition<T>>> = transitions
            .into_iter()
            .map(|t| Rc::new(t.transition.clone()))
            .collect();
        transitions.reverse();
        let new_cost = solver::compute_solution_cost(new_cost, &transitions, &model.target, &model);
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
        g_bound = Some(new_g_bound);
        println!("New g bound: {}, expanded: {}", new_g_bound, expanded);
    }
    println!("Expanded: {}", expanded);
    if let Some(cost) = cost {
        incumbent.reverse();
        Some((cost, incumbent))
    } else {
        None
    }
}

pub type ExistDFSSolution<T, U> = Option<(U, T, Vec<Rc<TransitionWithG<T, U>>>)>;

pub fn exist_dfs<T: variable::Numeric, U: variable::Numeric>(
    node: DFSNode<U>,
    model: &didp_parser::Model<T>,
    generator: &SuccessorGenerator<TransitionWithG<T, U>>,
    prob: &mut FxHashMap<hashable_state::HashableState, U>,
    g_bound: Option<U>,
    maximize: bool,
    expanded: &mut u32,
) -> ExistDFSSolution<T, U> {
    let state = node.state;
    let g = node.g;
    *expanded += 1;
    if let Some(base_cost) = model.get_base_cost(&state) {
        if maximize && g_bound.is_some() && g <= g_bound.unwrap() {
            return None;
        }
        return Some((g, base_cost, Vec::new()));
    }
    if let Some(other_g) = prob.get(&state) {
        if (maximize && g <= *other_g) || (!maximize && g >= *other_g) {
            return None;
        } else {
            prob.remove(&state);
        }
    }
    for transition in generator.applicable_transitions(&state) {
        let g = transition.g.eval_cost(g, &state, &model.table_registry);
        if !maximize && g_bound.is_some() && g >= g_bound.unwrap() {
            continue;
        }
        let successor = transition.transition.apply(&state, &model.table_registry);
        if model.check_constraints(&successor) {
            let node = DFSNode {
                state: successor,
                g,
            };
            let result = exist_dfs(node, model, generator, prob, g_bound, maximize, expanded);
            if let Some((g, base_cost, mut transitions)) = result {
                transitions.push(transition);
                return Some((g, base_cost, transitions));
            }
        }
    }
    prob.insert(state, g);
    None
}
