use crate::bfs_node::{BFSNodeRegistry, TransitionWithG};
use crate::evaluator;
use crate::priority_queue;
use crate::search_node::StateForSearchNode;
use crate::solver;
use crate::successor_generator::SuccessorGenerator;
use didp_parser::variable;
use std::fmt;
use std::rc::Rc;

pub struct BFSEvaluators<'a, T: variable::Numeric, U: variable::Numeric, H, F> {
    pub generator: SuccessorGenerator<'a, TransitionWithG<T, U>>,
    pub h_evaluator: H,
    pub f_evaluator: F,
}

pub fn forward_bfs<'a, T, U, H, F>(
    model: &'a didp_parser::Model<T>,
    evaluators: &BFSEvaluators<'a, T, U, H, F>,
    g_bound: Option<U>,
    registry_capacity: Option<usize>,
) -> solver::Solution<T>
where
    T: variable::Numeric,
    U: variable::Numeric + Ord + fmt::Display,
    H: evaluator::Evaluator<U>,
    F: Fn(U, U, &StateForSearchNode, &didp_parser::Model<T>) -> U,
{
    let mut open = priority_queue::PriorityQueue::new(true);
    let mut registry = BFSNodeRegistry::new(model);
    if let Some(capacity) = registry_capacity {
        registry.reserve(capacity);
    }

    let cost = T::zero();
    let g = U::zero();
    let initial_state = StateForSearchNode::new(&model.target);
    let initial_node = match registry.get_node(initial_state, cost, g, None, None) {
        Some(node) => node,
        None => return None,
    };
    let h = evaluators.h_evaluator.eval(&initial_node.state, model)?;
    let f = (evaluators.f_evaluator)(g, h, &initial_node.state, model);
    *initial_node.h.borrow_mut() = Some(h);
    *initial_node.f.borrow_mut() = Some(f);
    open.push(initial_node);
    let mut expanded = 0;
    let mut f_max = f;

    while let Some(node) = open.pop() {
        if *node.closed.borrow() {
            continue;
        }
        *node.closed.borrow_mut() = true;
        expanded += 1;
        let f = (*node.f.borrow()).unwrap();
        if f > f_max {
            f_max = f;
            println!("f = {}, expanded: {}", f, expanded);
        }
        if let Some(cost) = model.get_base_cost(&node.state) {
            println!("Expanded: {}", expanded);
            let (cost, transitions) = node.trace_transitions(cost, model);
            let transitions = transitions
                .into_iter()
                .map(|t| Rc::new(t.transition.clone()))
                .collect();
            return Some((cost, transitions));
        }
        for transition in evaluators.generator.applicable_transitions(&node.state) {
            let g = transition
                .g
                .eval_cost(node.g, &node.state, &model.table_registry);
            let cost = transition.eval_cost(node.cost, &node.state, &model.table_registry);
            if g_bound.is_some() && g >= g_bound.unwrap() {
                continue;
            }
            let state = transition
                .transition
                .apply(&node.state, &model.table_registry);
            if model.check_constraints(&state) {
                if let Some(successor) =
                    registry.get_node(state, cost, g, Some(transition), Some(node.clone()))
                {
                    let h = *successor.h.borrow();
                    let h = match h {
                        Some(h) => Some(h),
                        None => {
                            let h = evaluators.h_evaluator.eval(&successor.state, model);
                            *successor.h.borrow_mut() = h;
                            h
                        }
                    };
                    if let Some(h) = h {
                        let f = (evaluators.f_evaluator)(g, h, &successor.state, model);
                        *successor.f.borrow_mut() = Some(f);
                        if g_bound.is_none() || f < g_bound.unwrap() {
                            open.push(successor);
                        }
                    }
                }
            }
        }
    }
    println!("Expanded: {}", expanded);
    None
}
