use crate::bfs_node::BFSNode;
use crate::evaluator;
use crate::search_node::trace_transitions;
use crate::solver;
use crate::state_registry::{StateInRegistry, StateInformation, StateRegistry};
use crate::successor_generator::SuccessorGenerator;
use dypdl::variable_type;
use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections;
use std::fmt;
use std::rc::Rc;

/// Performs cyclic best-first search.
///
/// The f-value, the priority of a node, is computed by f_evaluator, which is a function of the g-value, the h-value, and the state.
/// The h-value is computed by h_evaluator.
pub fn cyclic_bfs<T, H, F>(
    model: &dypdl::Model,
    generator: SuccessorGenerator<dypdl::Transition>,
    h_evaluator: &H,
    f_evaluator: F,
    callback: &mut Box<solver::Callback<T>>,
    parameters: solver::SolverParameters<T>,
    initial_registry_capacity: Option<usize>,
) -> solver::Solution<T>
where
    T: variable_type::Numeric + Ord + fmt::Display,
    H: evaluator::Evaluator,
    F: Fn(T, T, &StateInRegistry, &dypdl::Model) -> T,
{
    let time_keeper = parameters.time_limit.map_or_else(
        solver::TimeKeeper::default,
        solver::TimeKeeper::with_time_limit,
    );
    let mut primal_bound = parameters.primal_bound;
    let mut open = vec![collections::BinaryHeap::new()];
    let mut registry = StateRegistry::new(model);
    if let Some(capacity) = initial_registry_capacity {
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
    if !parameters.quiet {
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
    open[0].push(Reverse(initial_node));
    let mut expanded = 0;
    let mut generated = 0;
    let best_bound = f;
    let mut solution = solver::Solution {
        best_bound: Some(f),
        ..Default::default()
    };

    let mut i = 0;
    let mut no_node = true;
    loop {
        if let Some(Reverse(node)) = open[i].pop() {
            if *node.closed.borrow() {
                continue;
            }
            *node.closed.borrow_mut() = true;
            expanded += 1;
            let f = node.f.borrow().unwrap();
            if primal_bound.is_some() && f >= primal_bound.unwrap() {
                open[i].clear();
            } else {
                if no_node {
                    no_node = false;
                }
                if model.is_goal(node.state()) {
                    if !parameters.quiet {
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
                    i = 0;
                    no_node = true;
                    continue;
                }

                if time_keeper.check_time_limit() {
                    if !parameters.quiet {
                        println!("Expanded: {}", expanded);
                    }
                    solution.expanded = expanded;
                    solution.generated = generated;
                    solution.time = time_keeper.elapsed_time();
                    return solution;
                }

                for transition in generator.applicable_transitions(node.state()) {
                    let g = transition.eval_cost(node.g, node.state(), &model.table_registry);
                    if primal_bound.is_some() && g >= primal_bound.unwrap() {
                        continue;
                    }
                    let state = transition.apply(node.state(), &model.table_registry);
                    if model.check_constraints(&state) {
                        let constructor =
                            |state: StateInRegistry, g: T, other: Option<&Rc<BFSNode<T>>>| {
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
                        if let Some((successor, dominated)) = registry.insert(state, g, constructor)
                        {
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
                                while i + 1 >= open.len() {
                                    open.push(collections::BinaryHeap::new());
                                }
                                open[i + 1].push(Reverse(successor));
                                generated += 1;
                            }
                        }
                    }
                }
            }
        }
        if no_node && i + 1 == open.len() {
            break;
        } else if i + 1 == open.len() {
            i = 0;
            no_node = true;
        } else {
            i += 1;
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