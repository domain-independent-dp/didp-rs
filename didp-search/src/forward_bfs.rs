use crate::evaluator;
use crate::priority_queue;
use crate::search_node;
use crate::solver;
use crate::successor_generator;
use didp_parser::variable;
use std::fmt;

pub fn forward_bfs<T, H, F>(
    model: &didp_parser::Model<T>,
    h_function: &H,
    f_function: &F,
    ub: Option<T>,
    registry_capacity: Option<usize>,
) -> solver::Solution<T>
where
    T: variable::Numeric + Ord + fmt::Display,
    H: evaluator::Evaluator<T>,
    F: Fn(T, T, &search_node::StateForSearchNode, &didp_parser::Model<T>) -> T,
{
    let mut open = priority_queue::PriorityQueue::new(true);
    let mut registry = search_node::SearchNodeRegistry::new(model);
    if let Some(capacity) = registry_capacity {
        registry.reserve(capacity);
    }
    let generator = successor_generator::SuccessorGenerator::new(model, false);

    let g = T::zero();
    let initial_state = search_node::StateForSearchNode::new(&model.target);
    let initial_node = match registry.get_node(initial_state, g, None, None) {
        Some(node) => node,
        None => return None,
    };
    let h = h_function.eval(&initial_node.state, model)?;
    let f = f_function(g, h, &initial_node.state, model);
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
            return Some(node.trace_transitions(cost, model));
        }
        for transition in generator.applicable_transitions(&node.state) {
            let g = transition.eval_cost(node.g, &node.state, &model.table_registry);
            if ub.is_some() && g >= ub.unwrap() {
                continue;
            }
            let state = transition.apply(&node.state, &model.table_registry);
            if let Some(successor) =
                registry.get_node(state, g, Some(transition), Some(node.clone()))
            {
                if model.check_constraints(&successor.state) {
                    let h = *successor.h.borrow();
                    let h = match h {
                        Some(h) => Some(h),
                        None => {
                            let h = h_function.eval(&node.state, model);
                            *successor.h.borrow_mut() = h;
                            h
                        }
                    };
                    if let Some(h) = h {
                        let f = f_function(g, h, &node.state, model);
                        *successor.f.borrow_mut() = Some(f);
                        if ub.is_none() || f < ub.unwrap() {
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
