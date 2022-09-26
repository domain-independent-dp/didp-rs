use super::callback::get_callback;
use super::solution::CostToDump;
use super::solver_parameters;
use crate::util;
use dypdl::variable_type::Numeric;
use dypdl_heuristic_search::{DualBoundAnytimeBeamSearch, FEvaluatorType, SolverParameters};
use std::error::Error;
use std::fmt;
use std::str;

pub fn load_from_yaml<T>(
    config: &yaml_rust::Yaml,
) -> Result<DualBoundAnytimeBeamSearch<T>, Box<dyn Error>>
where
    T: Numeric + Ord + fmt::Display,
    <T as str::FromStr>::Err: fmt::Debug,
    CostToDump: From<T>,
{
    let map = match config {
        yaml_rust::Yaml::Hash(map) => map,
        yaml_rust::Yaml::Null => {
            return Ok(DualBoundAnytimeBeamSearch {
                f_evaluator_type: FEvaluatorType::default(),
                callback: Box::new(|_| {}),
                pruning: true,
                parameters: SolverParameters::default(),
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
    let pruning = match map.get(&yaml_rust::Yaml::from_str("no_pruning")) {
        Some(yaml_rust::Yaml::Boolean(value)) => !(*value),
        None => true,
        value => {
            return Err(util::YamlContentErr::new(format!(
                "expected Boolean, but found `{:?}`",
                value
            ))
            .into())
        }
    };
    let parameters = solver_parameters::parse_from_map(map)?;
    let callback = get_callback(map)?;
    Ok(DualBoundAnytimeBeamSearch {
        f_evaluator_type,
        pruning,
        callback,
        parameters,
    })
}
