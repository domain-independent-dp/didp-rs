use super::solver_parameters;
use crate::util;
use dypdl::variable_type::Numeric;
use dypdl_heuristic_search::LookaheadLazyDijkstra;
use std::error::Error;
use std::fmt;
use std::str;

pub fn load_from_yaml<T: Numeric>(
    config: &yaml_rust::Yaml,
) -> Result<LookaheadLazyDijkstra<T>, Box<dyn Error>>
where
    <T as str::FromStr>::Err: fmt::Debug,
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
    let bound_ratio = util::get_numeric_by_key(map, "bound_ratio")?;
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
    Ok(LookaheadLazyDijkstra {
        bound_ratio,
        parameters,
        initial_registry_capacity,
    })
}
