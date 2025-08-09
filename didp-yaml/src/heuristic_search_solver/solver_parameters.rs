use crate::util;
use dypdl::variable_type::{Integer, Numeric};
use dypdl_heuristic_search::Parameters;
use std::fmt;
use std::str;

pub fn parse_from_map<T: Numeric>(
    map: &linked_hash_map::LinkedHashMap<yaml_rust::Yaml, yaml_rust::Yaml>,
) -> Result<Parameters<T>, util::YamlContentErr>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    let primal_bound = match map.get(&yaml_rust::Yaml::from_str("primal_bound")) {
        Some(yaml_rust::Yaml::Integer(value)) => Some(T::from_integer(*value as Integer)),
        Some(yaml_rust::Yaml::Real(value)) => Some(value.parse().map_err(|e| {
            util::YamlContentErr::new(format!("could not parse {value} as a number: {e:?}"))
        })?),
        None => None,
        value => {
            return Err(util::YamlContentErr::new(format!(
                "expected Integer or Real, but found `{value:?}`",
            )))
        }
    };
    let time_limit = match map.get(&yaml_rust::Yaml::from_str("time_limit")) {
        Some(value) => Some(util::get_numeric(value)?),
        None => None,
    };
    let quiet = match map.get(&yaml_rust::Yaml::from_str("quiet")) {
        Some(yaml_rust::Yaml::Boolean(value)) => *value,
        None => false,
        value => {
            return Err(util::YamlContentErr::new(format!(
                "expected Boolean, but found `{value:?}`",
            )))
        }
    };
    let get_all_solutions = match map.get(&yaml_rust::Yaml::from_str("get_all_solutions")) {
        Some(yaml_rust::Yaml::Boolean(value)) => *value,
        None => false,
        value => {
            return Err(util::YamlContentErr::new(format!(
                "expected Boolean, but found `{value:?}`",
            )))
        }
    };
    let initial_registry_capacity =
        match map.get(&yaml_rust::Yaml::from_str("initial_registry_capacity")) {
            Some(yaml_rust::Yaml::Integer(value)) => Some(*value as usize),
            None => Some(1000000),
            value => {
                return Err(util::YamlContentErr::new(format!(
                    "expected Integer for `initial_registry_capacity`, but found `{value:?}`",
                )))
            }
        };
    Ok(Parameters {
        primal_bound,
        time_limit,
        get_all_solutions,
        quiet,
        initial_registry_capacity,
    })
}
