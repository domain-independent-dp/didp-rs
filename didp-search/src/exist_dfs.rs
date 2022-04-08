use crate::hashable_state;
use crate::search_node::TransitionWithG;
use crate::solver;
use crate::successor_generator::SuccessorGenerator;
use didp_parser::variable;
use didp_parser::Transition;
use rustc_hash::FxHashMap;
use std::fmt;
use std::rc::Rc;

#[derive(Clone, PartialEq, Default)]
pub struct DFSNode<T: variable::Numeric> {
    pub state: hashable_state::HashableState,
    pub g: T,
}

pub fn forward_iterative_exist_dfs<T: variable::Numeric + fmt::Display>(
    model: &didp_parser::Model<T>,
    generator: &SuccessorGenerator<TransitionWithG<T>>,
    mut primal_bound: Option<T>,
    maximize: bool,
    capacity: Option<usize>,
) -> solver::Solution<T> {
    let mut expanded = 0;
    let mut prob = FxHashMap::default();
    if let Some(capacity) = capacity {
        prob.reserve(capacity);
    };
    let mut incumbent = Vec::new();
    let node = DFSNode {
        state: hashable_state::HashableState::new(&model.target),
        g: T::zero(),
    };
    while let Some((cost, transitions)) = exist_dfs(
        node.clone(),
        model,
        &generator,
        &mut prob,
        primal_bound,
        maximize,
        &mut expanded,
    ) {
        let mut transitions: Vec<Rc<Transition<T>>> = transitions
            .into_iter()
            .map(|t| Rc::new(t.transition.clone()))
            .collect();
        transitions.reverse();
        let cost = solver::compute_solution_cost(cost, &transitions, &model.target, &model);
        println!("New primal bound: {}, expanded: {}", cost, expanded);
        primal_bound = Some(cost);
        incumbent = transitions;
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
    generator: &SuccessorGenerator<TransitionWithG<T>>,
    prob: &mut FxHashMap<hashable_state::HashableState, T>,
    primal_bound: Option<T>,
    maximize: bool,
    expanded: &mut u32,
) -> Option<(T, Vec<Rc<TransitionWithG<T>>>)> {
    let state = node.state;
    let g = node.g;
    *expanded += 1;
    if let Some(cost) = model.get_base_cost(&state) {
        return Some((cost, Vec::new()));
    }
    if let Some(other_g) = prob.get(&state) {
        if g >= *other_g {
            return None;
        } else {
            prob.remove(&state);
        }
    }
    for transition in generator.applicable_transitions(&state) {
        let g = transition.g.eval_cost(g, &state, &model.table_registry);
        if let Some(bound) = primal_bound {
            if (maximize && g <= bound) || (!maximize && g >= bound) {
                continue;
            }
        }
        let successor = transition.transition.apply(&state, &model.table_registry);
        if model.check_constraints(&successor) {
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
            if let Some((cost, mut transitions)) = result {
                transitions.push(transition);
                return Some((cost, transitions));
            }
        }
    }
    prob.insert(state, g);
    None
}
