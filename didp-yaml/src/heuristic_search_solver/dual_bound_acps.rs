use super::callback::get_callback;
use super::solution::CostToDump;
use super::solver_parameters;
use crate::util;
use dypdl::variable_type::Numeric;
use dypdl_heuristic_search::ProgressiveSearchParameters;
use dypdl_heuristic_search::{DualBoundACPS, FEvaluatorType};
use std::error::Error;
use std::fmt;
use std::str;

pub fn load_from_yaml<T>(config: &yaml_rust::Yaml) -> Result<DualBoundACPS<T>, Box<dyn Error>>
where
    T: Numeric + Ord + fmt::Display,
    <T as str::FromStr>::Err: fmt::Debug,
    CostToDump: From<T>,
{
    let map = match config {
        yaml_rust::Yaml::Hash(map) => map,
        _ => {
            return Err(util::YamlContentErr::new(format!(
                "expected Hash, but found `{:?}`",
                config
            ))
            .into())
        }
    };
    let f_evaluator_type = match map.get(&yaml_rust::Yaml::from_str("f")) {
        Some(yaml_rust::Yaml::String(string)) => match &string[..] {
            "+" => FEvaluatorType::Plus,
            "max" => FEvaluatorType::Max,
            "min" => FEvaluatorType::Min,
            "h" => FEvaluatorType::Overwrite,
            op => {
                return Err(util::YamlContentErr::new(format!(
                    "unexpected operator for f function `{}`",
                    op
                ))
                .into())
            }
        },
        None => FEvaluatorType::default(),
        value => {
            return Err(util::YamlContentErr::new(format!(
                "expected String, but found `{:?}`",
                value
            ))
            .into())
        }
    };
    let init = match map.get(&yaml_rust::Yaml::from_str("init")) {
        Some(yaml_rust::Yaml::Integer(value)) => *value as usize,
        Some(value) => {
            return Err(util::YamlContentErr::new(format!(
                "expected Integer, but found `{:?}`",
                value
            ))
            .into())
        }
        None => 1,
    };
    let step = match map.get(&yaml_rust::Yaml::from_str("step")) {
        Some(yaml_rust::Yaml::Integer(value)) => *value as usize,
        Some(value) => {
            return Err(util::YamlContentErr::new(format!(
                "expected Integer, but found `{:?}`",
                value
            ))
            .into())
        }
        None => 1,
    };
    let bound = match map.get(&yaml_rust::Yaml::from_str("bound")) {
        Some(yaml_rust::Yaml::Integer(value)) => Some(*value as usize),
        Some(value) => {
            return Err(util::YamlContentErr::new(format!(
                "expected Integer, but found `{:?}`",
                value
            ))
            .into())
        }
        None => None,
    };
    let progressive_parameters = ProgressiveSearchParameters { init, step, bound };
    let parameters = solver_parameters::parse_from_map(map)?;
    let initial_registry_capacity =
        match map.get(&yaml_rust::Yaml::from_str("initial_registry_capacity")) {
            Some(yaml_rust::Yaml::Integer(value)) => Some(*value as usize),
            None => Some(1000000),
            value => {
                return Err(util::YamlContentErr::new(format!(
                    "expected Integer, but found `{:?}`",
                    value
                ))
                .into())
            }
        };
    let callback = get_callback(map)?;
    Ok(DualBoundACPS {
        f_evaluator_type,
        progressive_parameters,
        callback,
        parameters,
        initial_registry_capacity,
    })
}
