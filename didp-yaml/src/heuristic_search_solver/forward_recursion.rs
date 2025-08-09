use super::solver_parameters;
use crate::util;
use dypdl::variable_type::Numeric;
use dypdl_heuristic_search::{ForwardRecursion, Parameters, Search};
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
            return Ok(Box::new(ForwardRecursion::new(Rc::new(model), {
                Parameters {
                    initial_registry_capacity: Some(1000000),
                    ..Default::default()
                }
            })))
        }
        _ => {
            return Err(util::YamlContentErr::new(
                format!("expected Hash, but found `{config:?}`",),
            )
            .into())
        }
    };
    let parameters = solver_parameters::parse_from_map(map)?;
    Ok(Box::new(ForwardRecursion::new(Rc::new(model), parameters)))
}
