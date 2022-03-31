use crate::priority_queue;
use crate::search_node;
use crate::successor_generator;
use didp_parser::expression;
use didp_parser::expression_parser;
use didp_parser::variable;
use rustc_hash::FxHashMap;
use std::error::Error;
use std::fmt;
use std::str;

#[derive(Debug, Clone)]
pub struct ConfigErr(String);

impl ConfigErr {
    pub fn new(message: String) -> ConfigErr {
        ConfigErr(format!("Error in config: {}", message))
    }
}

impl fmt::Display for ConfigErr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Error for ConfigErr {}

type Solution<T> = Option<(T, Vec<didp_parser::Transition<T>>)>;

pub fn forward_bfs<T: variable::Numeric + Ord>(
    model: &didp_parser::Model<T>,
    config: &yaml_rust::Yaml,
) -> Result<Solution<T>, Box<dyn Error>>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    let map = match config {
        yaml_rust::Yaml::Hash(map) => map,
        _ => return Err(ConfigErr::new(format!("expected Hash, but found `{:?}`", config)).into()),
    };
    let parameters = FxHashMap::default();
    let h_expression = match map.get(&yaml_rust::Yaml::from_str("h")) {
        Some(yaml_rust::Yaml::String(string)) => expression_parser::parse_numeric(
            string.clone(),
            &model.state_metadata,
            &model.table_registry,
            &parameters,
        )?,
        _ => expression::NumericExpression::Constant(T::zero()),
    };
    let f_expression = match map.get(&yaml_rust::Yaml::from_str("f")) {
        Some(yaml_rust::Yaml::String(string)) => expression_parser::parse_numeric(
            string.clone(),
            &model.state_metadata,
            &model.table_registry,
            &parameters,
        )?,
        _ => expression::NumericExpression::Cost,
    };

    let mut open = priority_queue::PriorityQueue::new(true);
    let mut registry = search_node::SearchNodeRegistry::new(model);
    let generator = successor_generator::SuccessorGenerator::new(model, false);

    let g = T::zero();
    let initial_node = match registry.get_node(model.target.clone(), g, None, None) {
        Some(node) => node,
        None => return Ok(None),
    };
    let h = h_expression.eval_cost(g, &model.target, &model.table_registry);
    let f = f_expression.eval_cost(h, &model.target, &model.table_registry);
    *initial_node.h.borrow_mut() = Some(h);
    *initial_node.f.borrow_mut() = Some(f);
    open.push(initial_node);

    while let Some(node) = open.pop() {
        if *node.closed.borrow() {
            continue;
        }
        *node.closed.borrow_mut() = true;
        if let Some(cost) = model.get_base_cost(&node.state) {
            return Ok(Some((node.g + cost, node.trace_transitions())));
        }
        for transition in generator.applicable_transitions(&node.state) {
            let state = transition.apply_effects(&node.state, &model.table_registry);
            let g = transition.eval_cost(node.g, &node.state, &model.table_registry);
            if let Some(successor) = registry.get_node(state, g, None, Some(node.clone())) {
                if model.check_constraints(&successor.state) {
                    let h = match *successor.h.borrow() {
                        Some(h) => h,
                        None => {
                            let h = h_expression.eval_cost(g, &node.state, &model.table_registry);
                            *successor.h.borrow_mut() = Some(h);
                            h
                        }
                    };
                    let f = f_expression.eval_cost(h, &node.state, &model.table_registry);
                    *successor.f.borrow_mut() = Some(f);
                    open.push(successor);
                }
            }
        }
    }
    Ok(None)
}
