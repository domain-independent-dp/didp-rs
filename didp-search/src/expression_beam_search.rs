use super::expression_astar::FEvaluatorType;
use crate::expression_evaluator;
use crate::forward_beam_search;
use crate::search_node::StateForSearchNode;
use crate::solver;
use didp_parser::variable;
use std::cmp;
use std::error::Error;
use std::fmt;
use std::str;

pub struct ExpressionBeamSearch<T: variable::Numeric> {
    h_evaluator: expression_evaluator::ExpressionEvaluator<T>,
    f_evaluator_type: FEvaluatorType,
    beams: Vec<usize>,
    maximize: bool,
    primal_bound: Option<T>,
    registry_capacity: Option<usize>,
}

impl<T: variable::Numeric + Ord + fmt::Display> solver::Solver<T> for ExpressionBeamSearch<T> {
    #[inline]
    fn set_primal_bound(&mut self, bound: Option<T>) {
        self.primal_bound = bound;
    }

    fn solve(&mut self, model: &didp_parser::Model<T>) -> solver::Solution<T> {
        match self.f_evaluator_type {
            FEvaluatorType::Plus => {
                let f_evaluator =
                    Box::new(|g, h, _: &StateForSearchNode, _: &didp_parser::Model<T>| g + h);
                forward_beam_search::iterative_forward_beam_search(
                    model,
                    &self.h_evaluator,
                    &f_evaluator,
                    &self.beams,
                    self.maximize,
                    self.primal_bound,
                    self.registry_capacity,
                )
            }
            FEvaluatorType::Max => {
                let f_evaluator =
                    Box::new(|g, h, _: &StateForSearchNode, _: &didp_parser::Model<T>| {
                        cmp::max(g, h)
                    });
                forward_beam_search::iterative_forward_beam_search(
                    model,
                    &self.h_evaluator,
                    &f_evaluator,
                    &self.beams,
                    self.maximize,
                    self.primal_bound,
                    self.registry_capacity,
                )
            }
            FEvaluatorType::Min => {
                let f_evaluator =
                    Box::new(|g, h, _: &StateForSearchNode, _: &didp_parser::Model<T>| {
                        cmp::min(g, h)
                    });
                forward_beam_search::iterative_forward_beam_search(
                    model,
                    &self.h_evaluator,
                    &f_evaluator,
                    &self.beams,
                    self.maximize,
                    self.primal_bound,
                    self.registry_capacity,
                )
            }
            FEvaluatorType::Overwrite => {
                let f_evaluator =
                    Box::new(|_, h, _: &StateForSearchNode, _: &didp_parser::Model<T>| h);
                forward_beam_search::iterative_forward_beam_search(
                    model,
                    &self.h_evaluator,
                    &f_evaluator,
                    &self.beams,
                    self.maximize,
                    self.primal_bound,
                    self.registry_capacity,
                )
            }
        }
    }
}

impl<T: variable::Numeric + Ord> ExpressionBeamSearch<T> {
    pub fn new(
        model: &didp_parser::Model<T>,
        config: &yaml_rust::Yaml,
    ) -> Result<ExpressionBeamSearch<T>, Box<dyn Error>>
    where
        <T as str::FromStr>::Err: fmt::Debug,
    {
        let map = match config {
            yaml_rust::Yaml::Hash(map) => map,
            _ => {
                return Err(solver::ConfigErr::new(format!(
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
                return Err(solver::ConfigErr::new(format!(
                    "expected Integer, but found `{:?}`",
                    value
                ))
                .into())
            }
        };
        let primal_bound = match map.get(&yaml_rust::Yaml::from_str("primal_bound")) {
            Some(yaml_rust::Yaml::Integer(value)) => {
                Some(T::from_integer(*value as variable::Integer))
            }
            Some(yaml_rust::Yaml::Real(value)) => Some(value.parse().map_err(|e| {
                solver::ConfigErr::new(format!("could not parse {} as a number: {:?}", value, e))
            })?),
            None => None,
            value => {
                return Err(solver::ConfigErr::new(format!(
                    "expected Integer, but found `{:?}`",
                    value
                ))
                .into())
            }
        };
        let beams = match map.get(&yaml_rust::Yaml::from_str("beam")) {
            Some(yaml_rust::Yaml::Integer(value)) => vec![*value as usize],
            Some(yaml_rust::Yaml::Array(array)) => {
                let mut beams = Vec::new();
                for v in array {
                    match v {
                        yaml_rust::Yaml::Integer(value) => {
                            beams.push(*value as usize);
                        }
                        value => {
                            return Err(solver::ConfigErr::new(format!(
                                "expected Integer or Array, but found `{:?}`",
                                value
                            ))
                            .into())
                        }
                    }
                }
                beams
            }
            value => {
                return Err(solver::ConfigErr::new(format!(
                    "expected Integer or Array, but found `{:?}`",
                    value
                ))
                .into())
            }
        };
        let maximize = match map.get(&yaml_rust::Yaml::from_str("maximize")) {
            Some(yaml_rust::Yaml::Boolean(value)) => *value,
            Some(value) => {
                return Err(solver::ConfigErr::new(format!(
                    "expected Boolean, but found `{:?}`",
                    value
                ))
                .into())
            }
            None => model.reduce_function == didp_parser::ReduceFunction::Max,
        };
        let f_evaluator_type = match map.get(&yaml_rust::Yaml::from_str("f")) {
            Some(yaml_rust::Yaml::String(string)) => match &string[..] {
                "+" => FEvaluatorType::Plus,
                "max" => FEvaluatorType::Max,
                "min" => FEvaluatorType::Min,
                "h" => FEvaluatorType::Overwrite,
                op => {
                    return Err(solver::ConfigErr::new(format!(
                        "unexpected operator for f function `{}`",
                        op
                    ))
                    .into())
                }
            },
            None => FEvaluatorType::Plus,
            value => {
                return Err(solver::ConfigErr::new(format!(
                    "expected String, but found `{:?}`",
                    value
                ))
                .into())
            }
        };
        Ok(ExpressionBeamSearch {
            h_evaluator,
            f_evaluator_type,
            primal_bound,
            beams,
            maximize,
            registry_capacity,
        })
    }
}
