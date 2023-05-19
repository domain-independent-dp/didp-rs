use super::solver_parameters;
use crate::util;
use dypdl::variable_type::{Continuous, Numeric};
use dypdl_heuristic_search::{
    create_dual_bound_weighted_astar, FEvaluatorType, Parameters, Search,
};
use std::error::Error;
use std::fmt;
use std::rc::Rc;
use std::str;

pub fn load_from_yaml<T>(
    model: dypdl::Model,
    config: &yaml_rust::Yaml,
) -> Result<Box<dyn Search<T>>, Box<dyn Error>>
where
    T: Numeric + Ord + fmt::Display + 'static,
    <T as str::FromStr>::Err: fmt::Debug,
{
    let map = match config {
        yaml_rust::Yaml::Hash(map) => map,
        yaml_rust::Yaml::Null => {
            return Ok(create_dual_bound_weighted_astar(
                Rc::new(model),
                {
                    Parameters {
                        initial_registry_capacity: Some(1000000),
                        ..Default::default()
                    }
                },
                FEvaluatorType::Plus,
                1.0,
            ))
        }
        _ => {
            return Err(util::YamlContentErr::new(format!(
                "expected Hash for the solver config, but found `{:?}`",
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
                    "unexpected operator for `{}` for `f`",
                    op
                ))
                .into())
            }
        },
        None => FEvaluatorType::default(),
        value => {
            return Err(util::YamlContentErr::new(format!(
                "expected String for `f`, but found `{:?}`",
                value
            ))
            .into())
        }
    };
    let parameters = solver_parameters::parse_from_map(map)?;
    let weight = match map.get(&yaml_rust::Yaml::from_str("weight")) {
        Some(yaml_rust::Yaml::Integer(value)) => *value as Continuous,
        Some(yaml_rust::Yaml::Real(value)) => value.parse::<Continuous>()?,
        Some(value) => {
            return Err(util::YamlContentErr::new(format!(
                "expected Integer or Float for `weight`, but found `{:?}`",
                value
            ))
            .into())
        }
        None => 1.0,
    };
    Ok(create_dual_bound_weighted_astar(
        Rc::new(model),
        parameters,
        f_evaluator_type,
        weight,
    ))
}
