use super::solver_parameters;
use crate::util;
use dypdl::variable_type::Numeric;
use dypdl_heuristic_search::{Dijkstra, Parameters, Search};
use std::error::Error;
use std::fmt;
use std::rc::Rc;
use std::str;

pub fn load_from_yaml<T: Numeric>(
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
            return Ok(Box::new(Dijkstra::new(
                Rc::new(model),
                Parameters::default(),
                false,
                Some(1000000),
            )))
        }
        _ => {
            return Err(util::YamlContentErr::new(format!(
                "expected Hash, but found `{:?}`",
                config
            ))
            .into())
        }
    };
    let parameters = solver_parameters::parse_from_map(map)?;
    let lazy = match map.get(&yaml_rust::Yaml::from_str("lazy")) {
        Some(yaml_rust::Yaml::Boolean(value)) => *value,
        None => false,
        value => {
            return Err(util::YamlContentErr::new(format!(
                "expected Boolean for `lazy`, but found `{:?}`",
                value
            ))
            .into())
        }
    };
    let initial_registry_capacity =
        match map.get(&yaml_rust::Yaml::from_str("initial_registry_capacity")) {
            Some(yaml_rust::Yaml::Integer(value)) => Some(*value as usize),
            None => Some(1000000),
            value => {
                return Err(util::YamlContentErr::new(format!(
                    "expected Integer for `initial_registry_capacity`, but found `{:?}`",
                    value
                ))
                .into())
            }
        };
    Ok(Box::new(Dijkstra::new(
        Rc::new(model),
        parameters,
        lazy,
        initial_registry_capacity,
    )))
}
