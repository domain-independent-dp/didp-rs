use super::callback::get_callback;
use super::solution::CostToDump;
use super::solver_parameters;
use crate::util;
use dypdl::variable_type::Numeric;
use dypdl_heuristic_search::{DualBoundRRCBFS, FEvaluatorType, SolverParameters};
use std::error::Error;
use std::fmt;
use std::str;

pub fn load_from_yaml<T>(config: &yaml_rust::Yaml) -> Result<DualBoundRRCBFS<T>, Box<dyn Error>>
where
    T: Numeric + Ord + fmt::Display,
    <T as str::FromStr>::Err: fmt::Debug,
    CostToDump: From<T>,
{
    let map = match config {
        yaml_rust::Yaml::Hash(map) => map,
        yaml_rust::Yaml::Null => {
            return Ok(DualBoundRRCBFS {
                f_evaluator_type: FEvaluatorType::default(),
                callback: Box::new(|_| {}),
                parameters: SolverParameters::default(),
                initial_registry_capacity: Some(1000000),
            })
        }
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
    Ok(DualBoundRRCBFS {
        f_evaluator_type,
        callback,
        parameters,
        initial_registry_capacity,
    })
}
