use crate::evaluator;
use crate::search_node::{trace_transitions, SearchNode};
use crate::solver;
use crate::state_registry::{StateInRegistry, StateInformation, StateRegistry};
use crate::successor_generator::SuccessorGenerator;
use dypdl::variable_type;
use dypdl::ReduceFunction;
use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;

/// Performs depth-first branch-and-bound.
///
/// The f-value, the priority of a node, is computed by f_evaluator, which is a function of the g-value, the h-value, and the state.
/// The h-value is computed by h_evaluator.
/// The f-value is used as a dual bound to prune a node.
pub fn dfbb<T, H, F>(
    model: &dypdl::Model,
    generator: SuccessorGenerator<dypdl::Transition>,
    h_evaluator: &Option<H>,
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
    let mut open = Vec::new();
    let mut registry = StateRegistry::new(model);
    if let Some(capacity) = initial_registry_capacity {
        registry.reserve(capacity);
    }

    let cost = T::zero();
    let initial_state = StateInRegistry::new(&model.target);
    let constructor = |state: StateInRegistry, cost: T, _: Option<&Rc<SearchNode<T>>>| {
        Some(Rc::new(SearchNode {
            state,
            cost,
            ..Default::default()
        }))
    };
    let (initial_node, _) = registry.insert(initial_state, cost, constructor).unwrap();
    open.push(initial_node);
    let mut expanded = 0;
    let mut generated = 0;
    let mut solution = solver::Solution::default();
    let mut best_bound = None;

    while let Some(node) = open.pop() {
        if *node.closed.borrow() {
            continue;
        }
        *node.closed.borrow_mut() = true;
        expanded += 1;

        if model.is_goal(node.state())
            && (primal_bound.is_none()
                || ((model.reduce_function == ReduceFunction::Min
                    && node.cost() < primal_bound.unwrap())
                    || (model.reduce_function == ReduceFunction::Max
                        && node.cost() > primal_bound.unwrap())))
        {
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
            continue;
        }

        if let (Some(bound), Some(h_evaluator)) = (primal_bound, h_evaluator.as_ref()) {
            let h = h_evaluator.eval(node.state(), model);
            if let Some(h) = h {
                let f = f_evaluator(node.cost(), h, node.state(), model);
                if (model.reduce_function == ReduceFunction::Min && f >= bound)
                    || (model.reduce_function == ReduceFunction::Max && f <= bound)
                {
                    continue;
                } else if best_bound.map_or(false, |dual_bound| {
                    (model.reduce_function == ReduceFunction::Min && f > dual_bound)
                        || (model.reduce_function == ReduceFunction::Max && f < dual_bound)
                }) {
                    best_bound = Some(f);
                    solution.best_bound = Some(f);
                    if f == bound {
                        return solution;
                    }
                }
            } else {
                continue;
            }
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
            let cost = transition.eval_cost(node.cost(), node.state(), &model.table_registry);
            let state = transition.apply(node.state(), &model.table_registry);
            if model.check_constraints(&state) {
                let constructor =
                    |state: StateInRegistry, cost: T, _: Option<&Rc<SearchNode<T>>>| {
                        Some(Rc::new(SearchNode {
                            state,
                            cost,
                            parent: Some(node.clone()),
                            operator: Some(transition),
                            closed: RefCell::new(false),
                        }))
                    };
                if let Some((successor, dominated)) = registry.insert(state, cost, constructor) {
                    if let Some(dominated) = dominated {
                        if !*dominated.closed.borrow() {
                            *dominated.closed.borrow_mut() = true;
                        }
                    }
                    open.push(successor);
                    generated += 1;
                }
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
