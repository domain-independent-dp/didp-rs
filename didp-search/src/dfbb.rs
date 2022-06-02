use crate::evaluator;
use crate::search_node::{trace_transitions, SearchNode};
use crate::solver;
use crate::state_registry::{StateInRegistry, StateInformation, StateRegistry};
use crate::successor_generator::SuccessorGenerator;
use didp_parser::variable;
use didp_parser::ReduceFunction;
use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;

pub fn dfbb<T, H, F>(
    model: &didp_parser::Model<T>,
    generator: SuccessorGenerator<didp_parser::Transition<T>>,
    h_evaluator: H,
    f_evaluator: F,
    mut primal_bound: Option<T>,
    registry_capacity: Option<usize>,
) -> solver::Solution<T>
where
    T: variable::Numeric + Ord + fmt::Display,
    H: evaluator::Evaluator<T>,
    F: Fn(T, T, &StateInRegistry, &didp_parser::Model<T>) -> T,
{
    let mut open = Vec::new();
    let mut registry = StateRegistry::new(model);
    if let Some(capacity) = registry_capacity {
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
    let initial_node = match registry.insert(initial_state, cost, constructor) {
        Some((node, _)) => node,
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
        if model.get_base_cost(node.state()).is_some()
            && (primal_bound.is_none()
                || ((model.reduce_function == ReduceFunction::Min
                    && node.cost() < primal_bound.unwrap())
                    || (model.reduce_function == ReduceFunction::Max
                        && node.cost() > primal_bound.unwrap())))
        {
            println!("New primal bound: {}", node.cost());
            primal_bound = Some(node.cost());
            incumbent = Some(node);
            continue;
        }
        if let Some(bound) = primal_bound {
            let h = h_evaluator.eval(node.state(), model);
            if let Some(h) = h {
                let f = f_evaluator(node.cost(), h, node.state(), model);
                if (model.reduce_function == ReduceFunction::Min && f >= bound)
                    || (model.reduce_function == ReduceFunction::Max && f <= bound)
                {
                    continue;
                }
            } else {
                continue;
            }
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
                }
            }
        }
    }
    println!("Expanded: {}", expanded);
    incumbent.map(|node| (node.cost(), trace_transitions(node)))
}
