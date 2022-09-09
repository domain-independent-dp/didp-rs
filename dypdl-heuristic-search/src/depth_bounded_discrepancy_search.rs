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
use std::mem;
use std::rc::Rc;

/// Performs depth-bounded discrepancy search.
///
/// The f-value, the priority of a node, is computed by f_evaluator, which is a function of the g-value, the h-value, and the state.
/// The h-value is computed by h_evaluator.
/// The f-value is used as a dual bound to prune a node.
pub fn depth_bounded_discrepancy_search<T, H, F>(
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
    let mut open: Vec<collections::BinaryHeap<Reverse<Rc<BFSNode<T>>>>> = Vec::new();
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
    let mut current_node = Some(initial_node);
    let mut expanded = 0;
    let mut generated = 0;
    let best_bound = f;
    let mut solution = solver::Solution {
        best_bound: Some(f),
        ..Default::default()
    };

    let mut depth = 0;
    let mut depth_bound = 0;
    loop {
        let mut from_open = false;
        let mut tmp = None;
        mem::swap(&mut tmp, &mut current_node);
        let node = if let Some(node) = tmp {
            node
        } else {
            let mut change = false;
            while depth_bound < open.len() && open[depth_bound].is_empty() {
                depth_bound += 1;
                if !change {
                    change = true;
                }
            }
            if depth_bound == open.len() {
                break;
            }
            if change && !parameters.quiet {
                println!("New depth bound: {}, expanded: {}", depth_bound, expanded);
            }
            depth = depth_bound + 1;
            from_open = true;
            open[depth_bound].pop().unwrap().0
        };
        if *node.closed.borrow() {
            continue;
        }
        *node.closed.borrow_mut() = true;
        expanded += 1;
        let f = node.f.borrow().unwrap();
        if primal_bound.is_some() && f >= primal_bound.unwrap() {
            if from_open {
                open[depth_bound].clear();
            }
            continue;
        }

        if model.is_base(node.state()) {
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

        let mut successors = vec![];
        let mut best_f = None;
        let mut arg_best_f = None;
        for (i, transition) in generator.applicable_transitions(node.state()).enumerate() {
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
                        if best_f.is_none() || f < best_f.unwrap() {
                            best_f = Some(f);
                            arg_best_f = Some(i);
                        }
                        *successor.h.borrow_mut() = Some(h);
                        *successor.f.borrow_mut() = Some(f);
                        successors.push(successor);
                        generated += 1;
                    }
                }
            }
        }
        for (i, successor) in successors.into_iter().enumerate() {
            if i == arg_best_f.unwrap() {
                current_node = Some(successor);
            } else {
                while depth >= open.len() {
                    open.push(collections::BinaryHeap::new());
                }
                open[depth].push(Reverse(successor));
            }
        }
        depth += 1;
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
