use crate::bfs_node::{BFSNode, BFSNodeRegistry};
use crate::evaluator;
use crate::forward_bfs::BFSEvaluators;
use crate::priority_queue;
use crate::search_node::StateForSearchNode;
use crate::solver;
use didp_parser::variable;
use std::fmt;
use std::mem;
use std::rc::Rc;
use std::str;

pub fn iterative_forward_beam_search<'a, T, U, H, F>(
    model: &'a didp_parser::Model<T>,
    evaluators: &BFSEvaluators<'a, T, U, H, F>,
    beams: &[usize],
    maximize: bool,
    mut g_bound: Option<U>,
    discard_g_bound: bool,
    registry_capacity: Option<usize>,
) -> solver::Solution<T>
where
    T: variable::Numeric + fmt::Display,
    U: variable::Numeric + Ord + fmt::Display,
    <U as str::FromStr>::Err: fmt::Debug,
    H: evaluator::Evaluator<U>,
    F: Fn(U, U, &StateForSearchNode, &didp_parser::Model<T>) -> U,
{
    let mut incumbent = Vec::new();
    let mut cost = None;
    for beam in beams {
        let result = forward_beam_search(
            model,
            evaluators,
            *beam,
            maximize,
            g_bound,
            registry_capacity,
        );
        if let Some((new_g_bound, new_cost, new_incumbent)) = result {
            if let Some(current_cost) = cost {
                match model.reduce_function {
                    didp_parser::ReduceFunction::Max if new_cost > current_cost => {
                        incumbent = new_incumbent;
                        cost = Some(new_cost);
                        println!("New primal bound: {}", new_cost);
                    }
                    didp_parser::ReduceFunction::Min if new_cost < current_cost => {
                        incumbent = new_incumbent;
                        cost = Some(new_cost);
                        println!("New primal bound: {}", new_cost);
                    }
                    _ => {}
                }
            } else {
                incumbent = new_incumbent;
                cost = Some(new_cost);
                println!("New primal bound: {}", new_cost);
            }
            if !discard_g_bound {
                if let Some(current_bound) = g_bound {
                    if (maximize && new_g_bound > current_bound)
                        || (!maximize && new_g_bound < current_bound)
                    {
                        g_bound = Some(new_g_bound);
                        println!("New g bound: {}", new_g_bound);
                    }
                } else {
                    g_bound = Some(new_g_bound);
                    println!("New g bound: {}", new_g_bound);
                }
            }
        } else {
            println!("Failed to find a solution");
        }
    }
    cost.map(|cost| (cost, incumbent))
}

pub type BeamSearchSolution<T, U> = Option<(U, T, Vec<Rc<didp_parser::Transition<T>>>)>;

pub fn forward_beam_search<'a, T, U, H, F>(
    model: &'a didp_parser::Model<T>,
    evaluators: &BFSEvaluators<'a, T, U, H, F>,
    beam: usize,
    maximize: bool,
    g_bound: Option<U>,
    registry_capacity: Option<usize>,
) -> BeamSearchSolution<T, U>
where
    T: variable::Numeric,
    U: variable::Numeric + Ord + fmt::Display,
    H: evaluator::Evaluator<U>,
    F: Fn(U, U, &StateForSearchNode, &didp_parser::Model<T>) -> U,
{
    let mut open = priority_queue::PriorityQueue::new(maximize);
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
    let mut new_open = priority_queue::PriorityQueue::<Rc<BFSNode<T, U>>>::new(maximize);

    loop {
        let mut incumbent = None;
        while !open.is_empty() {
            let node = open.pop().unwrap();
            if *node.closed.borrow() {
                continue;
            }
            *node.closed.borrow_mut() = true;
            expanded += 1;
            if let Some(cost) = model.get_base_cost(&node.state) {
                if !maximize || g_bound.is_none() || node.g > g_bound.unwrap() {
                    let (cost, solution) = node.trace_transitions(cost, model);
                    if let Some((_, incumbent_cost, _)) = incumbent {
                        match model.reduce_function {
                            didp_parser::ReduceFunction::Max if cost > incumbent_cost => {
                                incumbent = Some((node.g, cost, solution));
                            }
                            didp_parser::ReduceFunction::Min if cost < incumbent_cost => {
                                incumbent = Some((node.g, cost, solution));
                            }
                            _ => {}
                        }
                    } else {
                        incumbent = Some((node.g, cost, solution));
                    }
                }
            }
            for transition in evaluators.generator.applicable_transitions(&node.state) {
                let g = transition
                    .g
                    .eval_cost(node.g, &node.state, &model.table_registry);
                let cost = transition.eval_cost(node.cost, &node.state, &model.table_registry);
                if !maximize && g_bound.is_some() && g >= g_bound.unwrap() {
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
                            if let Some(bound) = g_bound {
                                if !maximize && f >= bound {
                                    continue;
                                }
                            }
                            if new_open.len() < beam {
                                *successor.f.borrow_mut() = Some(f);
                                new_open.push(successor);
                            } else if let Some(peek) = new_open.peek() {
                                if (maximize && f > peek.f.borrow().unwrap())
                                    || (!maximize && f < peek.f.borrow().unwrap())
                                {
                                    new_open.pop();
                                    *successor.f.borrow_mut() = Some(f);
                                    new_open.push(successor);
                                }
                            }
                        }
                    }
                }
            }
        }
        println!("Expanded: {}", expanded);
        if let Some((g, cost, transitions)) = incumbent {
            let transitions = transitions
                .into_iter()
                .map(|t| Rc::new(t.transition.clone()))
                .collect();
            return Some((g, cost, transitions));
        }
        if new_open.is_empty() {
            return None;
        }
        registry.clear();
        mem::swap(&mut open, &mut new_open);
    }
}
