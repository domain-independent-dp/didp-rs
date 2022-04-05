use crate::successor_generator;
use crate::util;
use didp_parser::variable;
use rustc_hash::FxHashMap;
use std::error::Error;
use std::rc::Rc;

type StateMemo<T> =
    FxHashMap<didp_parser::State, (Option<T>, Option<Rc<didp_parser::Transition<T>>>)>;

pub fn start_forward_recursion<T: variable::Numeric>(
    model: &didp_parser::Model<T>,
    config: &yaml_rust::Yaml,
) -> Result<util::Solution<T>, Box<dyn Error>> {
    let map = match config {
        yaml_rust::Yaml::Hash(map) => map,
        _ => {
            return Err(
                util::ConfigErr::new(format!("expected Hash, but found `{:?}`", config)).into(),
            )
        }
    };
    let memo_capacity = match map.get(&yaml_rust::Yaml::from_str("capacity")) {
        Some(yaml_rust::Yaml::Integer(value)) => Some(*value as usize),
        None => Some(1000000),
        value => {
            return Err(
                util::ConfigErr::new(format!("expected Integer, but found `{:?}`", value)).into(),
            )
        }
    };

    let generator = successor_generator::SuccessorGenerator::new(model, false);
    let mut memo = FxHashMap::default();
    if let Some(capacity) = memo_capacity {
        memo.reserve(capacity);
    }
    let mut expanded = 0;
    let cost = forward_recursion(
        model.target.clone(),
        model,
        &generator,
        &mut memo,
        &mut expanded,
    );
    let mut transitions = Vec::new();
    match model.reduce_function {
        didp_parser::ReduceFunction::Max | didp_parser::ReduceFunction::Min if cost.is_some() => {
            let mut state = model.target.clone();
            while let Some((_, Some(transition))) = memo.get(&state) {
                let transition = transition.as_ref().clone();
                state = transition.apply_effects(&state, &model.table_registry);
                transitions.push(transition);
            }
        }
        _ => {}
    }
    println!("Expanded: {}", expanded);
    if let Some(cost) = cost {
        Ok(Some((cost, transitions)))
    } else {
        Ok(None)
    }
}

pub fn forward_recursion<T: variable::Numeric>(
    state: didp_parser::State,
    model: &didp_parser::Model<T>,
    generator: &successor_generator::SuccessorGenerator<T>,
    memo: &mut StateMemo<T>,
    expanded: &mut i32,
) -> Option<T> {
    *expanded += 1;
    if let Some(cost) = model.get_base_cost(&state) {
        return Some(cost);
    }
    if let Some((cost, _)) = memo.get(&state) {
        return *cost;
    }
    let mut cost = None;
    let mut best_transition = None;
    for transition in generator.applicable_transitions(&state) {
        let successor = transition.apply_effects(&state, &model.table_registry);
        if model.check_constraints(&successor) {
            let successor_cost = forward_recursion(successor, model, generator, memo, expanded);
            if let Some(successor_cost) = successor_cost {
                let current_cost =
                    transition.eval_cost(successor_cost, &state, &model.table_registry);
                if cost.is_none() {
                    cost = Some(current_cost);
                    match model.reduce_function {
                        didp_parser::ReduceFunction::Min | didp_parser::ReduceFunction::Max => {
                            best_transition = Some(transition);
                        }
                        _ => {}
                    }
                } else {
                    match model.reduce_function {
                        didp_parser::ReduceFunction::Min => {
                            if current_cost < cost.unwrap() {
                                cost = Some(current_cost);
                                best_transition = Some(transition);
                            }
                        }
                        didp_parser::ReduceFunction::Max => {
                            if current_cost > cost.unwrap() {
                                cost = Some(current_cost);
                                best_transition = Some(transition);
                            }
                        }
                        didp_parser::ReduceFunction::Sum => {
                            cost = Some(cost.unwrap() + current_cost);
                        }
                        didp_parser::ReduceFunction::Product => {
                            cost = Some(cost.unwrap() * current_cost);
                        }
                    }
                }
            }
        }
    }
    memo.insert(state, (cost, best_transition));
    cost
}
