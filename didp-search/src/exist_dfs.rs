use crate::successor_generator;
use crate::util;
use didp_parser::variable;
use rustc_hash::FxHashSet;
use std::error::Error;
use std::fmt;
use std::rc::Rc;
use std::str;

pub fn run_forward_iterative_exist_dfs<T: variable::Numeric + fmt::Display>(
    model: &didp_parser::Model<T>,
    config: &yaml_rust::Yaml,
) -> Result<util::Solution<T>, Box<dyn Error>>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    let map = match config {
        yaml_rust::Yaml::Hash(map) => map,
        _ => {
            return Err(
                util::ConfigErr::new(format!("expected Hash, but found `{:?}`", config)).into(),
            )
        }
    };
    let capacity = match map.get(&yaml_rust::Yaml::from_str("capacity")) {
        Some(yaml_rust::Yaml::Integer(value)) => Some(*value as usize),
        None => Some(1000000),
        value => {
            return Err(
                util::ConfigErr::new(format!("expected Integer, but found `{:?}`", value)).into(),
            )
        }
    };
    Ok(forward_iterative_exist_dfs(model, capacity))
}

pub fn forward_iterative_exist_dfs<T: variable::Numeric + fmt::Display>(
    model: &didp_parser::Model<T>,
    capacity: Option<usize>,
) -> util::Solution<T> {
    let mut nodes = 0;
    let generator = successor_generator::SuccessorGenerator::new(model, false);
    let mut prob = FxHashSet::default();
    if let Some(capacity) = capacity {
        prob.reserve(capacity);
    };
    let mut incumbent = Vec::new();
    let mut ub = None;
    while let Some((cost, transitions)) = exist_dfs(
        model.target.clone(),
        T::zero(),
        model,
        &generator,
        &mut prob,
        ub,
        &mut nodes,
    ) {
        ub = Some(cost - T::one());
        incumbent = transitions;
        println!("New UB: {}, expanded: {}", cost - T::one(), nodes);
    }
    if let Some(cost) = ub {
        incumbent.reverse();
        let transitions = incumbent.into_iter().map(|t| t.as_ref().clone()).collect();
        Some((cost + T::one(), transitions))
    } else {
        None
    }
}

pub fn exist_dfs<T: variable::Numeric>(
    state: didp_parser::State,
    cost: T,
    model: &didp_parser::Model<T>,
    generator: &successor_generator::SuccessorGenerator<T>,
    prob: &mut FxHashSet<didp_parser::State>,
    ub: Option<T>,
    nodes: &mut u32,
) -> Option<(T, Vec<Rc<didp_parser::Transition<T>>>)> {
    *nodes += 1;
    if model.get_base_cost(&state).is_some() {
        return Some((cost, Vec::new()));
    }
    if prob.contains(&state) {
        return None;
    }
    for transition in generator.applicable_transitions(&state) {
        let cost = transition.eval_cost(cost, &state, &model.table_registry);
        if ub.is_none() || cost <= ub.unwrap() {
            let successor = transition.apply_effects(&state, &model.table_registry);
            if model.check_constraints(&successor) {
                let result = exist_dfs(successor, cost, model, generator, prob, ub, nodes);
                if let Some((cost, mut transitions)) = result {
                    transitions.push(transition);
                    return Some((cost, transitions));
                }
            }
        }
    }
    prob.insert(state);
    None
}
