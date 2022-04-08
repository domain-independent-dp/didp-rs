use crate::evaluator;
use crate::forward_bfs::BFSEvaluators;
use crate::priority_queue;
use crate::search_node::{SearchNodeRegistry, StateForSearchNode};
use crate::solver;
use didp_parser::variable;
use std::fmt;
use std::mem;
use std::rc::Rc;
use std::str;

pub fn iterative_forward_beam_search<'a, T, H, F>(
    model: &'a didp_parser::Model<T>,
    evaluators: &BFSEvaluators<'a, T, H, F>,
    beams: &[usize],
    maximize: bool,
    mut primal_bound: Option<T>,
    registry_capacity: Option<usize>,
) -> solver::Solution<T>
where
    T: variable::Numeric + Ord + fmt::Display,
    <T as str::FromStr>::Err: fmt::Debug,
    H: evaluator::Evaluator<T>,
    F: Fn(T, T, &StateForSearchNode, &didp_parser::Model<T>) -> T,
{
    let mut incumbent = Vec::new();
    for beam in beams {
        let result = forward_beam_search(
            model,
            evaluators,
            *beam,
            maximize,
            primal_bound,
            registry_capacity,
        );
        if let Some((new_primal_bound, new_incumbent)) = result {
            primal_bound = Some(new_primal_bound);
            incumbent = new_incumbent;
            println!("New primal bound: {}", new_primal_bound);
        } else {
            println!("Failed to find a solution");
        }
    }
    primal_bound.map(|b| (b, incumbent))
}

pub fn forward_beam_search<'a, T, H, F>(
    model: &'a didp_parser::Model<T>,
    evaluators: &BFSEvaluators<'a, T, H, F>,
    beam: usize,
    maximize: bool,
    primal_bound: Option<T>,
    registry_capacity: Option<usize>,
) -> solver::Solution<T>
where
    T: variable::Numeric + Ord + fmt::Display,
    <T as str::FromStr>::Err: fmt::Debug,
    H: evaluator::Evaluator<T>,
    F: Fn(T, T, &StateForSearchNode, &didp_parser::Model<T>) -> T,
{
    let mut open = priority_queue::PriorityQueue::new(!maximize);
    let mut registry = SearchNodeRegistry::new(model);
    if let Some(capacity) = registry_capacity {
        registry.reserve(capacity);
    }

    let g = T::zero();
    let initial_state = StateForSearchNode::new(&model.target);
    let initial_node = match registry.get_node(initial_state, g, None, None) {
        Some(node) => node,
        None => return None,
    };
    let h = evaluators.h_evaluator.eval(&initial_node.state, model)?;
    let f = (evaluators.f_evaluator)(g, h, &initial_node.state, model);
    *initial_node.h.borrow_mut() = Some(h);
    *initial_node.f.borrow_mut() = Some(f);
    open.push(initial_node);
    let mut expanded = 0;
    let mut new_open = priority_queue::PriorityQueue::new(true);

    loop {
        let mut i = 0;
        let mut incumbent = None;
        while !open.is_empty() && i < beam {
            let node = open.pop().unwrap();
            if *node.closed.borrow() {
                continue;
            }
            *node.closed.borrow_mut() = true;
            expanded += 1;
            i += 1;
            if let Some(cost) = model.get_base_cost(&node.state) {
                if !maximize || primal_bound.is_none() || node.g <= primal_bound.unwrap() {
                    let solution = node.trace_transitions(cost, model);
                    if let Some((incumbent_cost, _)) = incumbent {
                        if (maximize && solution.0 > incumbent_cost)
                            || (!maximize && solution.0 < incumbent_cost)
                        {
                            incumbent = Some(solution);
                        }
                    } else {
                        incumbent = Some(solution);
                    }
                }
            }
            for transition in evaluators.generator.applicable_transitions(&node.state) {
                let g = transition
                    .g
                    .eval_cost(node.g, &node.state, &model.table_registry);
                if !maximize && primal_bound.is_some() && g >= primal_bound.unwrap() {
                    continue;
                }
                let state = transition
                    .transition
                    .apply(&node.state, &model.table_registry);
                if let Some(successor) =
                    registry.get_node(state, g, Some(transition), Some(node.clone()))
                {
                    if model.check_constraints(&successor.state) {
                        let h = *successor.h.borrow();
                        let h = match h {
                            Some(h) => Some(h),
                            None => {
                                let h = evaluators.h_evaluator.eval(&node.state, model);
                                *successor.h.borrow_mut() = h;
                                h
                            }
                        };
                        if let Some(h) = h {
                            let f = (evaluators.f_evaluator)(g, h, &node.state, model);
                            if let Some(bound) = primal_bound {
                                if (maximize && f <= bound) || (!maximize && f >= bound) {
                                    continue;
                                }
                            }
                            *successor.f.borrow_mut() = Some(f);
                            new_open.push(successor);
                        }
                    }
                }
            }
        }
        if let Some((cost, transitions)) = incumbent {
            println!("Expanded: {}", expanded);
            let transitions = transitions
                .into_iter()
                .map(|t| Rc::new(t.transition.clone()))
                .collect();
            return Some((cost, transitions));
        }
        if new_open.is_empty() {
            println!("Expanded: {}", expanded);
            return None;
        }
        registry.clear();
        open.clear();
        mem::swap(&mut open, &mut new_open);
    }
}
