use super::solver_parameters;
use super::transition_with_custom_cost;
use crate::dypdl_parser;
use crate::util;
use dypdl::variable_type::{Integer, Numeric, OrderedContinuous};
use dypdl::CostExpression;
use dypdl::CostType;
use dypdl_heuristic_search::search_algorithm::BeamSearchParameters;
use dypdl_heuristic_search::Search;
use dypdl_heuristic_search::{CustomExpressionParameters, ExpressionBeamSearch, FEvaluatorType};
use rustc_hash::FxHashMap;
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
    let custom_cost_type = match map.get(&yaml_rust::Yaml::from_str("cost_type")) {
        Some(yaml_rust::Yaml::String(string)) => match &string[..] {
            "integer" => dypdl::CostType::Integer,
            "continuous" => dypdl::CostType::Continuous,
            value => {
                return Err(
                    util::YamlContentErr::new(format!("unexpected cost type `{}`", value)).into(),
                )
            }
        },
        None => model.cost_type,
        value => {
            return Err(util::YamlContentErr::new(format!(
                "expected String for `custom_cost_type`, but found `{:?}`",
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
                            "expected (String, String) for `g`, but found (`{:?}`, `{:?}`)",
                            key, value
                        ))
                        .into())
                    }
                }
            }
            transition_with_custom_cost::load_custom_cost_expressions(
                &model,
                false,
                &custom_cost_type,
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

    let parameters = FxHashMap::default();
    let h_expression = match map.get(&yaml_rust::Yaml::from_str("h")) {
        Some(value) => match custom_cost_type {
            CostType::Integer => Some(CostExpression::from(
                dypdl_parser::load_integer_expression_from_yaml(value, &model, &parameters)?,
            )),
            CostType::Continuous => Some(CostExpression::from(
                dypdl_parser::load_continuous_expression_from_yaml(value, &model, &parameters)?,
            )),
        },
        None => None,
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
    let custom_f_evaluator_type = match map.get(&yaml_rust::Yaml::from_str("custom_f")) {
        Some(yaml_rust::Yaml::String(string)) => match &string[..] {
            "+" => FEvaluatorType::Plus,
            "max" => FEvaluatorType::Max,
            "min" => FEvaluatorType::Min,
            "h" => FEvaluatorType::Overwrite,
            op => {
                return Err(util::YamlContentErr::new(format!(
                    "unexpected operator for custom f function `{}`",
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
    let beam_size = match map.get(&yaml_rust::Yaml::from_str("beam_size")) {
        Some(yaml_rust::Yaml::Integer(value)) => *value as usize,
        value => {
            return Err(util::YamlContentErr::new(format!(
                "expected Integer, but found `{:?}`",
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

    match custom_cost_type {
        CostType::Integer => {
            let parameters = BeamSearchParameters {
                beam_size,
                keep_all_layers,
                parameters: solver_parameters::parse_from_map(map)?,
            };
            Ok(Box::new(ExpressionBeamSearch::<_, Integer>::new(
                Rc::new(model),
                parameters,
                f_evaluator_type,
                CustomExpressionParameters {
                    custom_costs,
                    forced_custom_costs,
                    h_expression,
                    f_evaluator_type: custom_f_evaluator_type,
                    custom_cost_type,
                    maximize,
                },
            )))
        }
        CostType::Continuous => {
            let parameters = BeamSearchParameters {
                beam_size,
                keep_all_layers,
                parameters: solver_parameters::parse_from_map(map)?,
            };
            Ok(Box::new(ExpressionBeamSearch::<_, OrderedContinuous>::new(
                Rc::new(model),
                parameters,
                f_evaluator_type,
                CustomExpressionParameters {
                    custom_costs,
                    forced_custom_costs,
                    h_expression,
                    f_evaluator_type: custom_f_evaluator_type,
                    custom_cost_type,
                    maximize,
                },
            )))
        }
    }
}
