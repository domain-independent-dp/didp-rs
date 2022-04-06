use crate::astar;
use crate::exist_dfs;
use crate::forward_recursion;
use crate::util;
use didp_parser::variable;
use std::error::Error;
use std::fmt;
use std::process;
use std::str;

pub fn solver_factory<T: 'static + variable::Numeric + Ord + fmt::Display>(
    config: &yaml_rust::Yaml,
) -> Box<
    dyn Fn(&didp_parser::Model<T>, &yaml_rust::Yaml) -> Result<util::Solution<T>, Box<dyn Error>>,
>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    let map = match config {
        yaml_rust::Yaml::Hash(map) => map,
        _ => {
            eprintln!("expected Hash, but found {:?}", config);
            process::exit(1);
        }
    };
    let config = match map.get(&yaml_rust::Yaml::from_str("config")) {
        Some(value) => value.clone(),
        None => yaml_rust::Yaml::Null,
    };
    match map.get(&yaml_rust::Yaml::from_str("solver")) {
        Some(yaml_rust::Yaml::String(string)) => match &string[..] {
            "astar" => Box::new(astar::astar),
            "beam_astar" => Box::new(astar::beam_astar),
            "exist_dfs" => Box::new(exist_dfs::run_forward_iterative_exist_dfs),
            "forward_recursion" => Box::new(forward_recursion::start_forward_recursion),
            value => {
                eprintln!("no such solver {:?}", value);
                process::exit(1);
            }
        },
        Some(value) => {
            eprintln!("expected String, but found {:?}", value);
            process::exit(1);
        }
        None => Box::new(forward_recursion::start_forward_recursion),
    }
}

pub fn iterative_search<T: variable::Numeric + Ord + fmt::Display, H, F>(
    model: &didp_parser::Model<T>,
    config: &yaml_rust::Yaml,
) -> Result<util::Solution<T>, Box<dyn Error>>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    let mut cost = None;
    let mut incumbent = Vec::new();
    let map = match config {
        yaml_rust::Yaml::Hash(map) => map,
        _ => {
            return Err(
                util::ConfigErr::new(format!("expected Hash, but found `{:?}`", config)).into(),
            )
        }
    };
    let ub = match map.get(&yaml_rust::Yaml::from_str("ub")) {
        Some(yaml_rust::Yaml::Integer(value)) => Some(T::from_integer(*value as variable::Integer)),
        Some(yaml_rust::Yaml::Real(value)) => Some(value.parse().map_err(|e| {
            util::ConfigErr::new(format!("could not parse {} as a number: {:?}", value, e))
        })?),
        None => None,
        value => {
            return Err(
                util::ConfigErr::new(format!("expected Integer, but found `{:?}`", value)).into(),
            )
        }
    };
    let solver_configs = match map.get(&yaml_rust::Yaml::from_str("solvers")) {
        Some(yaml_rust::Yaml::Array(array)) => array,
        value => {
            return Err(
                util::ConfigErr::new(format!("expected String, but found `{:?}`", value)).into(),
            )
        }
    };
    for config in solver_configs {}

    Ok(cost.map(|cost| (cost, incumbent)))
}
