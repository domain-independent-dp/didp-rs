use crate::bfs_lifo_open_list::BFSLIFOOpenList;
use crate::bfs_node::BFSNode;
use crate::evaluator;
use crate::search_node::trace_transitions;
use crate::solver;
use crate::state_registry::{StateInRegistry, StateInformation, StateRegistry};
use crate::successor_generator::SuccessorGenerator;
use dypdl::variable_type;
use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;

/// Performs best-first search.
///
/// The f-value, the priority of a node, is computed by f_evaluator, which is a function of the g-value, the h-value, and the state.
/// The h-value is computed by h_evaluator.
/// A node minimizes the f-value is expanded at each step.
pub fn forward_bfs<T, H, F>(
    model: &dypdl::Model,
    generator: SuccessorGenerator<dypdl::Transition>,
    h_evaluator: &H,
    f_evaluator: F,
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
    let mut open = BFSLIFOOpenList::default();
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
    open.push((f, h), initial_node);
    let mut expanded = 0;
    let mut generated = 0;
    let mut f_max = f;

    while let Some(node) = open.pop() {
        if *node.closed.borrow() {
            continue;
        }
        *node.closed.borrow_mut() = true;
        expanded += 1;
        let f = node.f.borrow().unwrap();
        if f > f_max {
            f_max = f;
            if !parameters.quiet {
                println!("f = {}, expanded: {}", f, expanded);
            }
        }
        if model.is_goal(node.state()) {
            return solver::Solution {
                cost: Some(node.g),
                is_optimal: true,
                transitions: trace_transitions(node),
                expanded,
                generated,
                time: time_keeper.elapsed_time(),
                ..Default::default()
            };
        }
        if time_keeper.check_time_limit() {
            return solver::Solution {
                best_bound: Some(f_max),
                expanded,
                generated,
                time: time_keeper.elapsed_time(),
                ..Default::default()
            };
        }
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
                        open.push((f, h), successor);
                        generated += 1;
                    }
                }
            }
        }
    }
    solver::Solution {
        is_infeasible: true,
        expanded,
        generated,
        time: time_keeper.elapsed_time(),
        ..Default::default()
    }
}
