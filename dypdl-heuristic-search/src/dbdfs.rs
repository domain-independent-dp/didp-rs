use crate::bfs_node::BFSNode;
use crate::evaluator;
use crate::search_node::trace_transitions;
use crate::solver;
use crate::state_registry::{StateInRegistry, StateInformation, StateRegistry};
use crate::util::ForwardSearchParameters;
use dypdl::variable_type;
use std::cell::RefCell;
use std::fmt;
use std::mem;
use std::rc::Rc;

/// Performs discrepancy-bounded depth-first search which expands child nodes in the best-first order.
///
/// The f-value, the priority of a node, is computed by f_evaluator, which is a function of the g-value, the h-value, and the state.
/// The h-value is computed by h_evaluator.
/// The f-value is used as a dual bound to prune a node.
pub fn dbdfs<T, H, F>(
    model: &dypdl::Model,
    h_evaluator: &H,
    f_evaluator: F,
    width: usize,
    callback: &mut Box<solver::Callback<T>>,
    parameters: ForwardSearchParameters<T>,
) -> solver::Solution<T>
where
    T: variable_type::Numeric + Ord + fmt::Display,
    H: evaluator::Evaluator,
    F: Fn(T, T, &StateInRegistry, &dypdl::Model) -> T,
{
    let time_keeper = parameters.parameters.time_limit.map_or_else(
        solver::TimeKeeper::default,
        solver::TimeKeeper::with_time_limit,
    );
    let mut primal_bound = parameters.parameters.primal_bound;
    let quiet = parameters.parameters.quiet;
    let generator = parameters.generator;
    let mut open = Vec::new();
    let mut next_open = Vec::new();
    let mut registry = StateRegistry::new(model);
    if let Some(capacity) = parameters.initial_registry_capacity {
        registry.reserve(capacity);
    }

    let g = T::zero();
    let initial_state = StateInRegistry::new(&model.target);
    let h = h_evaluator.eval(&initial_state, model);
    if h.is_none() {
        return solver::Solution {
            is_infeasible: true,
            ..Default::default()
        };
    }
    let h = h.unwrap();
    if !quiet {
        println!("Initial h = {}", h);
    }
    let f = f_evaluator(g, h, &initial_state, model);
    let constructor = |state: StateInRegistry, g: T, _: Option<&Rc<BFSNode<T>>>| {
        Some(Rc::new(BFSNode {
            state,
            g,
            h: RefCell::new(Some(h)),
            f: RefCell::new(Some(f)),
            ..Default::default()
        }))
    };
    let (initial_node, _) = registry.insert(initial_state, g, constructor).unwrap();
    open.push((initial_node, 0));
    let mut expanded = 0;
    let mut generated = 0;
    let best_bound = f;
    let mut solution = solver::Solution {
        best_bound: Some(f),
        ..Default::default()
    };
    let mut discrepancy_limit = width - 1;
    if !quiet {
        println!("Initial discrepancy limit: {}", discrepancy_limit);
    }

    while !open.is_empty() || !next_open.is_empty() {
        if open.is_empty() {
            mem::swap(&mut open, &mut next_open);
            discrepancy_limit += width;
            if !quiet {
                println!("New discrepancy limit: {}", discrepancy_limit);
            }
        }

        let (node, discrepancy) = open.pop().unwrap();
        if *node.closed.borrow() {
            continue;
        }
        *node.closed.borrow_mut() = true;
        expanded += 1;

        let f = node.f.borrow().unwrap();
        if primal_bound.is_some() && f >= primal_bound.unwrap() {
            continue;
        }

        if model.is_goal(node.state()) {
            if !quiet {
                println!("New primal bound: {}, expanded: {}", node.cost(), expanded);
            }
            let cost = node.cost();
            solution.cost = Some(cost);
            solution.expanded = expanded;
            solution.generated = generated;
            solution.time = time_keeper.elapsed_time();
            solution.transitions = trace_transitions(node);
            primal_bound = Some(cost);
            (callback)(&solution);
            if cost == best_bound {
                solution.is_optimal = true;
                return solution;
            }
            continue;
        }

        if time_keeper.check_time_limit() {
            if !quiet {
                println!("Expanded: {}", expanded);
            }
            solution.expanded = expanded;
            solution.generated = generated;
            solution.time = time_keeper.elapsed_time();
            return solution;
        }

        let mut successors = vec![];
        for transition in generator.applicable_transitions(node.state()) {
            let g = transition.eval_cost(node.g, node.state(), &model.table_registry);
            if primal_bound.is_some() && g >= primal_bound.unwrap() {
                continue;
            }
            let state = transition.apply(node.state(), &model.table_registry);
            if model.check_constraints(&state) {
                let constructor = |state: StateInRegistry, g: T, other: Option<&Rc<BFSNode<T>>>| {
                    // use a cached h-value
                    let h = if let Some(other) = other {
                        *other.h.borrow()
                    } else {
                        None
                    };
                    Some(Rc::new(BFSNode {
                        state,
                        g,
                        h: RefCell::new(h),
                        f: RefCell::new(None),
                        parent: Some(node.clone()),
                        operator: Some(transition),
                        closed: RefCell::new(false),
                    }))
                };
                if let Some((successor, dominated)) = registry.insert(state, g, constructor) {
                    if let Some(dominated) = dominated {
                        if !*dominated.closed.borrow() {
                            *dominated.closed.borrow_mut() = true;
                        }
                    }
                    let h = match *successor.h.borrow() {
                        None => h_evaluator.eval(successor.state(), model),
                        h => h,
                    };
                    if let Some(h) = h {
                        let f = f_evaluator(g, h, successor.state(), model);
                        if primal_bound.is_some() && f >= primal_bound.unwrap() {
                            continue;
                        }
                        *successor.h.borrow_mut() = Some(h);
                        *successor.f.borrow_mut() = Some(f);
                        successors.push(successor);
                        generated += 1;
                    }
                }
            }
        }
        // reverse sort
        successors.sort_by(|a, b| b.cmp(a));
        if let Some(best) = successors.pop() {
            open.push((best, discrepancy));
            let mut successors = successors
                .into_iter()
                .map(|x| (x, discrepancy + 1))
                .collect();
            if discrepancy < discrepancy_limit {
                open.append(&mut successors);
            } else {
                next_open.append(&mut successors);
            }
        }
    }
    if solution.cost.is_none() {
        solution.is_infeasible = true;
    } else {
        solution.is_optimal = true;
    }
    solution.expanded = expanded;
    solution.generated = generated;
    solution.time = time_keeper.elapsed_time();
    solution
}
