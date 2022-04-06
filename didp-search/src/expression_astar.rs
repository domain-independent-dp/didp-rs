use crate::expression_evaluator;
use crate::forward_beam_search;
use crate::forward_bfs;
use crate::solver;
use crate::util;
use didp_parser::expression;
use didp_parser::expression_parser;
use didp_parser::variable;
use rustc_hash::FxHashMap;
use std::cmp;
use std::error::Error;
use std::fmt;
use std::str;

pub struct ExpressionAstar<T: variable::Numeric> {
    h_evaluator: expression_evaluator::ExpressionEvaluator<T>,
    f_evaluator_type: FEvaluatorType,
    ub: Option<T>,
    registry_capacity: Option<usize>,
}

enum FEvaluatorType {
    Plus,
    Max,
    Min,
    Overwrite,
}

impl<T: variable::Numeric + Ord + fmt::Display> solver::Solver<T> for ExpressionAstar<T> {
    fn solve(&mut self, model: &didp_parser::Model<T>) -> solver::Solution<T> {
        match self.f_evaluator_type {
            FEvaluatorType::Plus => {
                let f_evaluator =
                    Box::new(|g, h, _: &didp_parser::State, _: &didp_parser::Model<T>| g + h);
                forward_bfs::forward_bfs(
                    model,
                    &self.h_evaluator,
                    &f_evaluator,
                    self.ub,
                    self.registry_capacity,
                )
            }
            FEvaluatorType::Max => {
                let f_evaluator =
                    Box::new(|g, h, _: &didp_parser::State, _: &didp_parser::Model<T>| {
                        cmp::max(g, h)
                    });
                forward_bfs::forward_bfs(
                    model,
                    &self.h_evaluator,
                    &f_evaluator,
                    self.ub,
                    self.registry_capacity,
                )
            }
            FEvaluatorType::Min => {
                let f_evaluator =
                    Box::new(|g, h, _: &didp_parser::State, _: &didp_parser::Model<T>| {
                        cmp::min(g, h)
                    });
                forward_bfs::forward_bfs(
                    model,
                    &self.h_evaluator,
                    &f_evaluator,
                    self.ub,
                    self.registry_capacity,
                )
            }
            FEvaluatorType::Overwrite => {
                let f_evaluator =
                    Box::new(|_, h, _: &didp_parser::State, _: &didp_parser::Model<T>| h);
                forward_bfs::forward_bfs(
                    model,
                    &self.h_evaluator,
                    &f_evaluator,
                    self.ub,
                    self.registry_capacity,
                )
            }
        }
    }
}

impl<T: variable::Numeric + Ord> ExpressionAstar<T> {
    pub fn new(
        model: &didp_parser::Model<T>,
        config: &yaml_rust::Yaml,
    ) -> Result<ExpressionAstar<T>, Box<dyn Error>>
    where
        <T as str::FromStr>::Err: fmt::Debug,
    {
        let map = match config {
            yaml_rust::Yaml::Hash(map) => map,
            _ => {
                return Err(util::ConfigErr::new(format!(
                    "expected Hash, but found `{:?}`",
                    config
                ))
                .into())
            }
        };
        let h_evaluator = match map.get(&yaml_rust::Yaml::from_str("h")) {
            Some(value) => expression_evaluator::ExpressionEvaluator::load_from_yaml(value, model)?,
            None => expression_evaluator::ExpressionEvaluator::default(),
        };
        let registry_capacity = match map.get(&yaml_rust::Yaml::from_str("registry_capacity")) {
            Some(yaml_rust::Yaml::Integer(value)) => Some(*value as usize),
            None => Some(1000000),
            value => {
                return Err(util::ConfigErr::new(format!(
                    "expected Integer, but found `{:?}`",
                    value
                ))
                .into())
            }
        };
        let ub = match map.get(&yaml_rust::Yaml::from_str("ub")) {
            Some(yaml_rust::Yaml::Integer(value)) => {
                Some(T::from_integer(*value as variable::Integer))
            }
            Some(yaml_rust::Yaml::Real(value)) => Some(value.parse().map_err(|e| {
                util::ConfigErr::new(format!("could not parse {} as a number: {:?}", value, e))
            })?),
            None => None,
            value => {
                return Err(util::ConfigErr::new(format!(
                    "expected Integer, but found `{:?}`",
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
                    return Err(util::ConfigErr::new(format!(
                        "unexpected operator for f function `{}`",
                        op
                    ))
                    .into())
                }
            },
            None => FEvaluatorType::Plus,
            value => {
                return Err(util::ConfigErr::new(format!(
                    "expected String, but found `{:?}`",
                    value
                ))
                .into())
            }
        };
        Ok(ExpressionAstar {
            h_evaluator,
            f_evaluator_type,
            ub,
            registry_capacity,
        })
    }
}

pub fn beam_astar<T: variable::Numeric + Ord + fmt::Display>(
    model: &didp_parser::Model<T>,
    config: &yaml_rust::Yaml,
) -> Result<util::Solution<T>, Box<dyn Error>>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    let map = match config {
        yaml_rust::Yaml::Hash(map) => map,
        _ => {
            return Err(
                util::ConfigErr::new(format!("expected Hash, but found `{:?}`", config)).into(),
            )
        }
    };
    let parameters = FxHashMap::default();
    let h_expression = match map.get(&yaml_rust::Yaml::from_str("h")) {
        Some(yaml_rust::Yaml::String(string)) => expression_parser::parse_numeric(
            string.clone(),
            &model.state_metadata,
            &model.table_registry,
            &parameters,
        )?,
        None => expression::NumericExpression::Constant(T::zero()),
        value => {
            return Err(
                util::ConfigErr::new(format!("expected String, but found `{:?}`", value)).into(),
            )
        }
    };
    let h_function = |state: &didp_parser::State, model: &didp_parser::Model<T>| {
        Some(h_expression.eval(state, &model.table_registry))
    };
    let registry_capacity = match map.get(&yaml_rust::Yaml::from_str("registry_capacity")) {
        Some(yaml_rust::Yaml::Integer(value)) => Some(*value as usize),
        None => Some(1000000),
        value => {
            return Err(
                util::ConfigErr::new(format!("expected Integer, but found `{:?}`", value)).into(),
            )
        }
    };
    let beam = match map.get(&yaml_rust::Yaml::from_str("beam")) {
        Some(yaml_rust::Yaml::Integer(value)) => *value as usize,
        None => return Err(util::ConfigErr::new(String::from("option beam not found")).into()),
        value => {
            return Err(
                util::ConfigErr::new(format!("expected Integer, but found `{:?}`", value)).into(),
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
    match map.get(&yaml_rust::Yaml::from_str("f")) {
        Some(yaml_rust::Yaml::String(string)) => match &string[..] {
            "+" => {
                let f_function = |g, h, _: &didp_parser::State, _: &didp_parser::Model<T>| g + h;
                Ok(forward_beam_search::forward_beam_search(
                    model,
                    &h_function,
                    &f_function,
                    beam,
                    ub,
                    registry_capacity,
                ))
            }
            "max" => {
                let f_function =
                    |g, h, _: &didp_parser::State, _: &didp_parser::Model<T>| cmp::max(g, h);
                Ok(forward_beam_search::forward_beam_search(
                    model,
                    &h_function,
                    &f_function,
                    beam,
                    ub,
                    registry_capacity,
                ))
            }
            "min" => {
                let f_function =
                    |g, h, _: &didp_parser::State, _: &didp_parser::Model<T>| cmp::min(g, h);
                Ok(forward_beam_search::forward_beam_search(
                    model,
                    &h_function,
                    &f_function,
                    beam,
                    ub,
                    registry_capacity,
                ))
            }
            "h" => {
                let f_function = |_, h, _: &didp_parser::State, _: &didp_parser::Model<T>| h;
                Ok(forward_beam_search::forward_beam_search(
                    model,
                    &h_function,
                    &f_function,
                    beam,
                    ub,
                    registry_capacity,
                ))
            }
            op => Err(
                util::ConfigErr::new(format!("unexpected operator for f function `{}`", op)).into(),
            ),
        },
        None => {
            let f_function = |g, h, _: &didp_parser::State, _: &didp_parser::Model<T>| g + h;
            Ok(forward_beam_search::forward_beam_search(
                model,
                &h_function,
                &f_function,
                beam,
                ub,
                registry_capacity,
            ))
        }
        value => {
            Err(util::ConfigErr::new(format!("expected String, but found `{:?}`", value)).into())
        }
    }
}
