use super::solver_parameters;
use crate::util;
use dypdl::variable_type::Numeric;
use dypdl_heuristic_search::{ForwardRecursion, SolverParameters};
use std::fmt;
use std::str;

pub fn load_from_yaml<T: Numeric>(
    config: &yaml_rust::Yaml,
) -> Result<ForwardRecursion<T>, util::YamlContentErr>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    let map = match config {
        yaml_rust::Yaml::Hash(map) => map,
        yaml_rust::Yaml::Null => {
            return Ok(ForwardRecursion {
                initial_registry_capacity: Some(1000000),
                parameters: SolverParameters::default(),
            })
        }
        _ => {
            return Err(util::YamlContentErr::new(format!(
                "expected Hash, but found `{:?}`",
                config
            )))
        }
    };
    let initial_registry_capacity =
        match map.get(&yaml_rust::Yaml::from_str("initial_registry_capacity")) {
            Some(yaml_rust::Yaml::Integer(value)) => Some(*value as usize),
            None => Some(1000000),
            value => {
                return Err(util::YamlContentErr::new(format!(
                    "expected Integer, but found `{:?}`",
                    value
                )))
            }
        };
    let parameters = solver_parameters::parse_from_map(map)?;
    Ok(ForwardRecursion {
        initial_registry_capacity,
        parameters,
    })
}
