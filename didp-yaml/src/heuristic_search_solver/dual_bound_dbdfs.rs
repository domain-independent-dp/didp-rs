use super::solver_parameters;
use crate::util;
use dypdl::variable_type::Numeric;
use dypdl_heuristic_search::{
    create_dual_bound_dbdfs, DbdfsParameters, FEvaluatorType, Parameters, Search,
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
            return Ok(create_dual_bound_dbdfs(
                Rc::new(model),
                DbdfsParameters {
                    parameters: Parameters {
                        initial_registry_capacity: Some(1000000),
                        ..Default::default()
                    },
                    ..Default::default()
                },
                FEvaluatorType::Plus,
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
    let width = match map.get(&yaml_rust::Yaml::from_str("width")) {
        Some(yaml_rust::Yaml::Integer(value)) => *value as usize,
        Some(value) => {
            return Err(util::YamlContentErr::new(format!(
                "expected Integer for `width`, but found `{:?}`",
                value
            ))
            .into())
        }
        None => 1,
    };
    let parameters = solver_parameters::parse_from_map(map)?;
    let parameters = DbdfsParameters { width, parameters };
    Ok(create_dual_bound_dbdfs(
        Rc::new(model),
        parameters,
        f_evaluator_type,
    ))
}
