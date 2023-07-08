use super::solver_parameters;
use crate::util;
use dypdl::variable_type::Numeric;
use dypdl_heuristic_search::{create_dual_bound_lnbs, FEvaluatorType, LnbsParameters, Search};
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
    let lns_beam_size = match map.get(&yaml_rust::Yaml::from_str("lns_initial_beam_size")) {
        Some(yaml_rust::Yaml::Integer(value)) => *value as usize,
        Some(value) => {
            return Err(util::YamlContentErr::new(format!(
                "expected Integer for `lns_initial_beam_size`, but found `{:?}`",
                value
            ))
            .into())
        }
        None => beam_size,
    };
    let no_cost_weight = match map.get(&yaml_rust::Yaml::from_str("no_cost_weight")) {
        Some(yaml_rust::Yaml::Boolean(value)) => *value,
        None => false,
        value => {
            return Err(util::YamlContentErr::new(format!(
                "expected Boolean for `no_cost_weight`, but found `{:?}`",
                value
            ))
            .into())
        }
    };
    let no_bandit = match map.get(&yaml_rust::Yaml::from_str("no_bandit")) {
        Some(yaml_rust::Yaml::Boolean(value)) => *value,
        None => false,
        value => {
            return Err(util::YamlContentErr::new(format!(
                "expected Boolean for `no_bandit`, but found `{:?}`",
                value
            ))
            .into())
        }
    };
    let no_transition_constraints =
        match map.get(&yaml_rust::Yaml::from_str("no_transition_constraints")) {
            Some(yaml_rust::Yaml::Boolean(value)) => *value,
            None => false,
            value => {
                return Err(util::YamlContentErr::new(format!(
                    "expected Boolean for `no_transition_constraints`, but found `{:?}`",
                    value
                ))
                .into())
            }
        };
    let search_zero_cost_neighbor =
        match map.get(&yaml_rust::Yaml::from_str("search_zero_cost_neighbor")) {
            Some(yaml_rust::Yaml::Boolean(value)) => *value,
            None => false,
            value => {
                return Err(util::YamlContentErr::new(format!(
                    "expected Boolean for `search_zero_cost_neighbor`, but found `{:?}`",
                    value
                ))
                .into())
            }
        };
    let seed = match map.get(&yaml_rust::Yaml::from_str("seed")) {
        Some(yaml_rust::Yaml::Integer(value)) => *value as u64,
        Some(value) => {
            return Err(util::YamlContentErr::new(format!(
                "expected Integer for `seed`, but found `{:?}`",
                value
            ))
            .into())
        }
        None => 2023,
    };
    let bandit_parameters = LnbsParameters {
        initial_beam_size: lns_beam_size,
        no_cost_weight,
        has_negative_cost: search_zero_cost_neighbor,
        no_bandit,
        no_transition_constraints,
        seed,
    };
    let parameters = solver_parameters::parse_from_map(map)?;
    Ok(create_dual_bound_lnbs(
        Rc::new(model),
        parameters,
        bandit_parameters,
        f_evaluator_type,
        beam_size,
        keep_all_layers,
    ))
}
