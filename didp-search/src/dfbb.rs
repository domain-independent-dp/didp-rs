use crate::evaluator;
use crate::search_node::{SearchNodeRegistry, StateForSearchNode};
use crate::solver;
use crate::successor_generator::SuccessorGenerator;
use didp_parser::variable;
use didp_parser::ReduceFunction;
use std::fmt;

pub fn dfbb<'a, T, H, F>(
    model: &'a didp_parser::Model<T>,
    generator: SuccessorGenerator<didp_parser::Transition<T>>,
    h_evaluator: H,
    f_evaluator: F,
    mut primal_bound: Option<T>,
    registry_capacity: Option<usize>,
) -> solver::Solution<T>
where
    T: variable::Numeric + Ord + fmt::Display,
    H: evaluator::Evaluator<T>,
    F: Fn(T, T, &StateForSearchNode, &didp_parser::Model<T>) -> T,
{
    let mut open = Vec::new();
    let mut registry = SearchNodeRegistry::new(model);
    if let Some(capacity) = registry_capacity {
        registry.reserve(capacity);
    }

    let cost = T::zero();
    let initial_state = StateForSearchNode::new(&model.target);
    let initial_node = match registry.get_node(initial_state, cost, None, None) {
        Some(node) => node,
        None => return None,
    };
    open.push(initial_node);
    let mut expanded = 0;
    let mut incumbent = None;

    while let Some(node) = open.pop() {
        if *node.closed.borrow() {
            continue;
        }
        *node.closed.borrow_mut() = true;
        expanded += 1;
        if model.get_base_cost(&node.state).is_some()
            && (primal_bound.is_none()
                || ((model.reduce_function == ReduceFunction::Min
                    && node.cost < primal_bound.unwrap())
                    || (model.reduce_function == ReduceFunction::Max
                        && node.cost > primal_bound.unwrap())))
        {
            println!("New primal bound: {}", node.cost);
            primal_bound = Some(node.cost);
            incumbent = Some(node);
            continue;
        }
        if let Some(bound) = primal_bound {
            let h = h_evaluator.eval(&node.state, model);
            if let Some(h) = h {
                let f = (f_evaluator)(node.cost, h, &node.state, model);
                if (model.reduce_function == ReduceFunction::Min && f >= bound)
                    || (model.reduce_function == ReduceFunction::Max && f <= bound)
                {
                    continue;
                }
            } else {
                continue;
            }
        }
        for transition in generator.applicable_transitions(&node.state) {
            let cost = transition.eval_cost(node.cost, &node.state, &model.table_registry);
            let mut state = transition.apply(&node.state, &model.table_registry);
            if model.check_constraints(&state) {
                let cost = model.apply_forced_transitions_in_place(&mut state, cost, false);
                if let Some(successor) =
                    registry.get_node(state, cost, Some(transition), Some(node.clone()))
                {
                    open.push(successor);
                }
            }
        }
    }
    println!("Expanded: {}", expanded);
    if let Some(node) = incumbent {
        Some(node.trace_transitions(cost, model))
    } else {
        None
    }
}
