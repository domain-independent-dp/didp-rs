use crate::util;
use dypdl::variable_type::{Integer, Numeric};
use dypdl_heuristic_search::SolverParameters;
use std::fmt;
use std::str;

pub fn parse_from_map<T: Numeric>(
    map: &linked_hash_map::LinkedHashMap<yaml_rust::Yaml, yaml_rust::Yaml>,
) -> Result<SolverParameters<T>, util::YamlContentErr>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    let primal_bound = match map.get(&yaml_rust::Yaml::from_str("primal_bound")) {
        Some(yaml_rust::Yaml::Integer(value)) => Some(T::from_integer(*value as Integer)),
        Some(yaml_rust::Yaml::Real(value)) => Some(value.parse().map_err(|e| {
            util::YamlContentErr::new(format!("could not parse {} as a number: {:?}", value, e))
        })?),
        None => None,
        value => {
            return Err(util::YamlContentErr::new(format!(
                "expected Integer or Real, but found `{:?}`",
                value
            )))
        }
    };
    let time_limit = match map.get(&yaml_rust::Yaml::from_str("time_limit")) {
        Some(yaml_rust::Yaml::Integer(value)) => Some(*value as u64),
        None => None,
        value => {
            return Err(util::YamlContentErr::new(format!(
                "expected Integer, but found `{:?}`",
                value
            )))
        }
    };
    let quiet = match map.get(&yaml_rust::Yaml::from_str("quiet")) {
        Some(yaml_rust::Yaml::Boolean(value)) => *value,
        None => false,
        value => {
            return Err(util::YamlContentErr::new(format!(
                "expected Boolean, but found `{:?}`",
                value
            )))
        }
    };
    Ok(SolverParameters {
        primal_bound,
        time_limit,
        quiet,
    })
}
