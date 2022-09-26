use super::solution::CostToDump;
use super::solver_parameters;
use super::SolverFactory;
use crate::util;
use dypdl::variable_type::Numeric;
use dypdl_heuristic_search::IterativeSearch;
use std::error::Error;
use std::fmt;
use std::str;

pub fn load_from_yaml<T>(
    config: &yaml_rust::Yaml,
    model: &dypdl::Model,
) -> Result<IterativeSearch<T>, Box<dyn Error>>
where
    T: Numeric + Ord + fmt::Display + 'static,
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
    let parameters = solver_parameters::parse_from_map(map)?;
    let solver_configs = match map.get(&yaml_rust::Yaml::from_str("solvers")) {
        Some(yaml_rust::Yaml::Array(array)) => array,
        value => {
            return Err(util::YamlContentErr::new(format!(
                "expected Array, but found `{:?}`",
                value
            ))
            .into())
        }
    };
    let factory = SolverFactory::default();
    let mut solvers = Vec::new();
    for config in solver_configs {
        solvers.push(factory.create(config, model)?);
    }
    Ok(IterativeSearch {
        solvers,
        parameters,
    })
}
