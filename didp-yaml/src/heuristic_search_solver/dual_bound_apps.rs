use super::solver_parameters;
use crate::util;
use dypdl::variable_type::Numeric;
use dypdl_heuristic_search::{
    create_dual_bound_apps, FEvaluatorType, Parameters, ProgressiveSearchParameters, Search,
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
            return Ok(create_dual_bound_apps(
                Rc::new(model),
                {
                    Parameters {
                        initial_registry_capacity: Some(1000000),
                        ..Default::default()
                    }
                },
                FEvaluatorType::Plus,
                ProgressiveSearchParameters::default(),
            ))
        }
        _ => {
            return Err(util::YamlContentErr::new(format!(
                "expected Hash for the solver config, but found `{config:?}`",
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
                    "unexpected operator for `{op}` for `f`",
                ))
                .into())
            }
        },
        None => FEvaluatorType::default(),
        value => {
            return Err(util::YamlContentErr::new(format!(
                "expected String for `f`, but found `{value:?}`",
            ))
            .into())
        }
    };
    let init = match map.get(&yaml_rust::Yaml::from_str("init")) {
        Some(yaml_rust::Yaml::Integer(value)) => *value as usize,
        Some(value) => {
            return Err(util::YamlContentErr::new(format!(
                "expected Integer for `init`, but found `{value:?}`",
            ))
            .into())
        }
        None => 1,
    };
    let step = match map.get(&yaml_rust::Yaml::from_str("step")) {
        Some(yaml_rust::Yaml::Integer(value)) => *value as usize,
        Some(value) => {
            return Err(util::YamlContentErr::new(format!(
                "expected Integer for `step`, but found `{value:?}`",
            ))
            .into())
        }
        None => 1,
    };
    let bound = match map.get(&yaml_rust::Yaml::from_str("width_bound")) {
        Some(yaml_rust::Yaml::Integer(value)) => Some(*value as usize),
        Some(value) => {
            return Err(util::YamlContentErr::new(format!(
                "expected Integer for `width_bound`, but found `{value:?}`",
            ))
            .into())
        }
        None => None,
    };
    let reset = match map.get(&yaml_rust::Yaml::from_str("reset")) {
        Some(yaml_rust::Yaml::Boolean(value)) => !(*value),
        None => false,
        value => {
            return Err(util::YamlContentErr::new(format!(
                "expected Boolean for `reset`, but found `{value:?}`",
            ))
            .into())
        }
    };
    let progressive_parameters = ProgressiveSearchParameters {
        init,
        step,
        bound,
        reset,
    };
    let parameters = solver_parameters::parse_from_map(map)?;
    Ok(create_dual_bound_apps(
        Rc::new(model),
        parameters,
        f_evaluator_type,
        progressive_parameters,
    ))
}
