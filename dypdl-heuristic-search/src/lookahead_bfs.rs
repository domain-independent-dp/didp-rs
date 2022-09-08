use crate::bfs_node::BFSNode;
use crate::evaluator;
use crate::search_node::trace_transitions;
use crate::solver;
use crate::state_registry::{StateInRegistry, StateInformation, StateRegistry};
use crate::successor_generator::SuccessorGenerator;
use dypdl::{variable_type, Continuous};
use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fmt;
use std::rc::Rc;

/// Performs best-first search with bounded depth-first lookahead.
///
/// The f-value, the priority of a node, is computed by f_evaluator, which is a function of the g-value, the h-value, and the state.
/// The h-value is computed by h_evaluator.
/// A node minimizes the f-value is expanded at each step.
pub fn lookahead_bfs<T, H, F>(
    model: &dypdl::Model,
    generator: SuccessorGenerator<dypdl::Transition>,
    h_evaluator: &H,
    f_evaluator: F,
    bound_ratio: Continuous,
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
    let g_bound = parameters.primal_bound;
    let mut open = BinaryHeap::default();
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
    open.push(Reverse(initial_node));
    let mut dfs_open = Vec::<Rc<BFSNode<T>>>::new();
    let mut expanded = 0;
    let mut generated = 0;
    let mut best_bound = f;
    let mut weighted_bound = T::from_continuous(best_bound.to_continuous() * bound_ratio);

    while let Some(Reverse(peek)) = open.peek() {
        if *peek.closed.borrow() {
            open.pop();
            continue;
        }

        let f = peek.f.borrow().unwrap();
        if f > best_bound {
            best_bound = f;
            weighted_bound = T::from_continuous(best_bound.to_continuous() * bound_ratio);
            if !parameters.quiet {
                println!("Best bound: {}, expanded: {}", f, expanded);
                println!("Weighted bound: {}", weighted_bound);
            }
        }

        let node = if let Some(node) = dfs_open.pop() {
            if *node.closed.borrow() {
                continue;
            }
            node
        } else {
            open.pop().unwrap().0
        };
        *node.closed.borrow_mut() = true;
        expanded += 1;

        if model.is_goal(node.state()) {
            let is_optimal = node.g <= best_bound;
            return solver::Solution {
                cost: Some(node.g),
                is_optimal,
                transitions: trace_transitions(node),
                expanded,
                generated,
                time: time_keeper.elapsed_time(),
                ..Default::default()
            };
        }
        if time_keeper.check_time_limit() {
            return solver::Solution {
                best_bound: Some(best_bound),
                expanded,
                generated,
                time: time_keeper.elapsed_time(),
                ..Default::default()
            };
        }

        let mut dfs_successors = Vec::new();
        for transition in generator.applicable_transitions(node.state()) {
            let g = transition.eval_cost(node.g, node.state(), &model.table_registry);
            if g_bound.is_some() && g >= g_bound.unwrap() {
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
                        if g_bound.is_some() && f >= g_bound.unwrap() {
                            continue;
                        }
                        *successor.h.borrow_mut() = Some(h);
                        *successor.f.borrow_mut() = Some(f);
                        if f <= weighted_bound {
                            dfs_successors.push(successor.clone());
                        }
                        open.push(Reverse(successor));
                        generated += 1;
                    }
                }
            }
        }
        dfs_successors.sort_by(|a, b| b.cmp(a));
        dfs_open.append(&mut dfs_successors);
    }
    solver::Solution {
        is_infeasible: true,
        expanded,
        generated,
        time: time_keeper.elapsed_time(),
        ..Default::default()
    }
}
