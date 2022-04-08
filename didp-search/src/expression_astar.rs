use crate::expression_evaluator;
use crate::forward_bfs;
use crate::search_node::StateForSearchNode;
use crate::solver;
use didp_parser::variable;
use std::cmp;
use std::error::Error;
use std::fmt;
use std::str;

#[derive(Default)]
pub struct ExpressionAstar<T: variable::Numeric> {
    h_evaluator: expression_evaluator::ExpressionEvaluator<T>,
    f_evaluator_type: FEvaluatorType,
    ub: Option<T>,
    registry_capacity: Option<usize>,
}

pub enum FEvaluatorType {
    Plus,
    Max,
    Min,
    Overwrite,
}

impl Default for FEvaluatorType {
    fn default() -> Self {
        Self::Plus
    }
}

impl<T: variable::Numeric + Ord + fmt::Display> solver::Solver<T> for ExpressionAstar<T> {
    #[inline]
    fn set_primal_bound(&mut self, ub: Option<T>) {
        self.ub = ub;
    }

    fn solve(&mut self, model: &didp_parser::Model<T>) -> solver::Solution<T> {
        match self.f_evaluator_type {
            FEvaluatorType::Plus => {
                let f_evaluator =
                    Box::new(|g, h, _: &StateForSearchNode, _: &didp_parser::Model<T>| g + h);
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
                    Box::new(|g, h, _: &StateForSearchNode, _: &didp_parser::Model<T>| {
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
                    Box::new(|g, h, _: &StateForSearchNode, _: &didp_parser::Model<T>| {
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
                    Box::new(|_, h, _: &StateForSearchNode, _: &didp_parser::Model<T>| h);
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
            yaml_rust::Yaml::Null => return Ok(ExpressionAstar::default()),
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
        let ub = match map.get(&yaml_rust::Yaml::from_str("ub")) {
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
        Ok(ExpressionAstar {
            h_evaluator,
            f_evaluator_type,
            ub,
            registry_capacity,
        })
    }
}
