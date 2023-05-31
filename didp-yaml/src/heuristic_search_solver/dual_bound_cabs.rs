use super::solver_parameters;
use crate::util;
use dypdl::variable_type::Numeric;
use dypdl_heuristic_search::{
    create_dual_bound_cabs, create_dual_bound_hd_cabs, create_dual_bound_hd_sync_cabs,
    create_dual_bound_shared_memory_cabs, BeamSearchParameters, CabsParameters, FEvaluatorType,
    Search,
};
use std::error::Error;
use std::fmt;
use std::rc::Rc;
use std::str;
use std::sync::Arc;

pub fn load_from_yaml<T>(
    model: dypdl::Model,
    config: &yaml_rust::Yaml,
) -> Result<Box<dyn Search<T>>, Box<dyn Error>>
where
    T: Numeric + Ord + fmt::Display + Send + Sync + 'static,
    <T as str::FromStr>::Err: fmt::Debug,
{
    let map = match config {
        yaml_rust::Yaml::Hash(map) => map,
        yaml_rust::Yaml::Null => {
            return Ok(create_dual_bound_cabs(
                Rc::new(model),
                CabsParameters::default(),
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
    let beam_size = match map.get(&yaml_rust::Yaml::from_str("initial_beam_size")) {
        Some(yaml_rust::Yaml::Integer(value)) => *value as usize,
        Some(value) => {
            return Err(util::YamlContentErr::new(format!(
                "expected Integer for `initial_beam_size`, but found `{:?}`",
                value
            ))
            .into())
        }
        None => 1,
    };
    let keep_all_layers = match map.get(&yaml_rust::Yaml::from_str("keep_all_layers")) {
        Some(yaml_rust::Yaml::Boolean(value)) => *value,
        None => false,
        value => {
            return Err(util::YamlContentErr::new(format!(
                "expected Boolean for `keep_all_layers`, but found `{:?}`",
                value
            ))
            .into())
        }
    };
    let max_beam_size = match map.get(&yaml_rust::Yaml::from_str("max_beam_size")) {
        Some(yaml_rust::Yaml::Integer(value)) => Some(*value as usize),
        Some(value) => {
            return Err(util::YamlContentErr::new(format!(
                "expected Integer for `max_beam_size`, but found `{:?}`",
                value
            ))
            .into())
        }
        None => None,
    };
    let parameters = solver_parameters::parse_from_map(map)?;
    let beam_search_parameters = BeamSearchParameters {
        parameters,
        beam_size,
        keep_all_layers,
    };
    let parameters = CabsParameters {
        max_beam_size,
        beam_search_parameters,
    };
    let threads = match map.get(&yaml_rust::Yaml::from_str("threads")) {
        Some(yaml_rust::Yaml::Integer(value)) => *value as usize,
        Some(value) => {
            return Err(util::YamlContentErr::new(format!(
                "expected Integer for `threads`, but found `{:?}`",
                value
            ))
            .into())
        }
        None => 1,
    };
    let parallel_type = match map.get(&yaml_rust::Yaml::from_str("parallel_type")) {
        Some(yaml_rust::Yaml::String(value)) => match value.as_str() {
            "hd" => 0,
            "hd-sync" => 1,
            "sm" => 2,
            _ => {
                return Err(util::YamlContentErr::new(format!(
                    "unexpected value for `parallel_type`: `{}`",
                    value
                ))
                .into())
            }
        },
        Some(value) => {
            return Err(util::YamlContentErr::new(format!(
                "expected String for `parallel_type`, but found `{:?}`",
                value
            ))
            .into())
        }
        None => 1,
    };

    if threads > 1 {
        match parallel_type {
            0 => Ok(create_dual_bound_hd_cabs(
                Arc::new(model),
                parameters,
                f_evaluator_type,
                threads,
            )),
            1 => Ok(create_dual_bound_hd_sync_cabs(
                Arc::new(model),
                parameters,
                f_evaluator_type,
                threads,
            )),
            2 => Ok(create_dual_bound_shared_memory_cabs(
                Arc::new(model),
                parameters,
                f_evaluator_type,
                threads,
            )),
            _ => Err(util::YamlContentErr::new(format!(
                "unexpected value for `parallel_type`: `{}`",
                parallel_type
            ))
            .into()),
        }
    } else {
        Ok(create_dual_bound_cabs(
            Rc::new(model),
            parameters,
            f_evaluator_type,
        ))
    }
}
