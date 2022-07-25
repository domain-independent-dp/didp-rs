use super::expression_evaluator;
use super::solver_parameters;
use super::transition_with_custom_cost;
use crate::util;
use dypdl::variable_type::Numeric;
use dypdl_heuristic_search::{
    ExpressionBeamSearch, ExpressionEvaluator, FEvaluatorType, SolverParameters,
};
use rustc_hash::FxHashMap;
use std::error::Error;
use std::fmt;
use std::str;

pub fn load_from_yaml<T>(
    config: &yaml_rust::Yaml,
    model: &dypdl::Model,
) -> Result<ExpressionBeamSearch<T>, Box<dyn Error>>
where
    T: Numeric + Ord + fmt::Display,
    <T as str::FromStr>::Err: fmt::Debug,
{
    let map = match config {
        yaml_rust::Yaml::Hash(map) => map,
        yaml_rust::Yaml::Null => {
            let custom_costs = model
                .forward_transitions
                .iter()
                .map(|t| t.cost.clone())
                .collect();
            let forced_custom_costs = model
                .forward_forced_transitions
                .iter()
                .map(|t| t.cost.clone())
                .collect();
            return Ok(ExpressionBeamSearch {
                custom_costs,
                forced_custom_costs,
                h_evaluator: ExpressionEvaluator::default(),
                f_evaluator_type: FEvaluatorType::default(),
                custom_cost_type: None,
                beam_sizes: vec![10000],
                maximize: false,
                parameters: SolverParameters::default(),
            });
        }
        _ => {
            return Err(util::YamlContentErr::new(format!(
                "expected Hash, but found `{:?}`",
                config
            ))
            .into())
        }
    };
    let custom_cost_type = match map.get(&yaml_rust::Yaml::from_str("cost_type")) {
        Some(yaml_rust::Yaml::String(string)) => match &string[..] {
            "integer" => Some(dypdl::CostType::Integer),
            "continuous" => Some(dypdl::CostType::Continuous),
            value => {
                return Err(
                    util::YamlContentErr::new(format!("unexpected cost type `{}`", value)).into(),
                )
            }
        },
        None => None,
        value => {
            return Err(util::YamlContentErr::new(format!(
                "expected String, but found `{:?}`",
                value
            ))
            .into())
        }
    };
    let (custom_costs, forced_custom_costs) = match map.get(&yaml_rust::Yaml::from_str("g")) {
        Some(yaml_rust::Yaml::Hash(map)) => {
            let mut g_expressions = FxHashMap::default();
            g_expressions.reserve(map.len());
            for (key, value) in map.iter() {
                match (key, value) {
                    (yaml_rust::Yaml::String(key), yaml_rust::Yaml::String(value)) => {
                        g_expressions.insert(key.clone(), value.clone());
                    }
                    _ => {
                        return Err(util::YamlContentErr::new(format!(
                            "expected (String, String), but found (`{:?}`, `{:?}`)",
                            key, value
                        ))
                        .into())
                    }
                }
            }
            transition_with_custom_cost::load_custom_cost_expressions(
                model,
                false,
                custom_cost_type.as_ref().unwrap_or(&model.cost_type),
                &g_expressions,
            )?
        }
        None => {
            let custom_costs = model
                .forward_transitions
                .iter()
                .map(|t| t.cost.clone())
                .collect();
            let forced_custom_costs = model
                .forward_forced_transitions
                .iter()
                .map(|t| t.cost.clone())
                .collect();
            (custom_costs, forced_custom_costs)
        }
        value => {
            return Err(util::YamlContentErr::new(format!(
                "expected Hash, but found `{:?}`",
                value
            ))
            .into())
        }
    };
    let h_evaluator = match map.get(&yaml_rust::Yaml::from_str("h")) {
        Some(yaml_rust::Yaml::String(value)) => expression_evaluator::load_from_string(
            value.clone(),
            model,
            custom_cost_type.as_ref().unwrap_or(&model.cost_type),
        )?,
        None => ExpressionEvaluator::default(),
        value => {
            return Err(util::YamlContentErr::new(format!(
                "expected String, but found `{:?}`",
                value
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
                    "unexpected operator for f function `{}`",
                    op
                ))
                .into())
            }
        },
        None => FEvaluatorType::default(),
        value => {
            return Err(util::YamlContentErr::new(format!(
                "expected String, but found `{:?}`",
                value
            ))
            .into())
        }
    };
    let beam_sizes = match map.get(&yaml_rust::Yaml::from_str("beam_size")) {
        Some(yaml_rust::Yaml::Integer(value)) => vec![*value as usize],
        Some(yaml_rust::Yaml::Array(array)) => {
            let mut beams = Vec::new();
            for v in array {
                match v {
                    yaml_rust::Yaml::Integer(value) => {
                        beams.push(*value as usize);
                    }
                    value => {
                        return Err(util::YamlContentErr::new(format!(
                            "expected Integer or Array, but found `{:?}`",
                            value
                        ))
                        .into())
                    }
                }
            }
            beams
        }
        None => vec![10000],
        value => {
            return Err(util::YamlContentErr::new(format!(
                "expected Integer or Array, but found `{:?}`",
                value
            ))
            .into())
        }
    };
    let maximize = match map.get(&yaml_rust::Yaml::from_str("maximize")) {
        Some(yaml_rust::Yaml::Boolean(value)) => *value,
        Some(value) => {
            return Err(util::YamlContentErr::new(format!(
                "expected Boolean, but found `{:?}`",
                value
            ))
            .into())
        }
        None => false,
    };
    let parameters = solver_parameters::parse_from_map(map)?;
    Ok(ExpressionBeamSearch {
        custom_costs,
        forced_custom_costs,
        h_evaluator,
        f_evaluator_type,
        custom_cost_type,
        beam_sizes,
        maximize,
        parameters,
    })
}
