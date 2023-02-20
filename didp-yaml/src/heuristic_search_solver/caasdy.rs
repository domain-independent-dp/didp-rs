use super::solver_parameters;
use crate::util;
use dypdl::variable_type::Numeric;
use dypdl_heuristic_search::{create_caasdy, FEvaluatorType, Parameters, Search};
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
    let (parameters, f_evaluator_type, initial_registry_capacity) = match config {
        yaml_rust::Yaml::Hash(map) => {
            let f_evaluator_type = match map.get(&yaml_rust::Yaml::from_str("f")) {
                Some(yaml_rust::Yaml::String(string)) => match &string[..] {
                    "+" => FEvaluatorType::Plus,
                    "max" => FEvaluatorType::Max,
                    "min" => FEvaluatorType::Min,
                    "h" => FEvaluatorType::Overwrite,
                    op => {
                        return Err(util::YamlContentErr::new(format!(
                            "unexpected operator `{}` for `f`",
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
            (parameters, f_evaluator_type, initial_registry_capacity)
        }
        yaml_rust::Yaml::Null => (
            Parameters::default(),
            FEvaluatorType::default(),
            Some(1000000),
        ),
        _ => {
            return Err(util::YamlContentErr::new(format!(
                "expected Hash for the solver config, but found `{:?}`",
                config
            ))
            .into())
        }
    };
    Ok(create_caasdy(
        Rc::new(model),
        parameters,
        f_evaluator_type,
        initial_registry_capacity,
    ))
}
