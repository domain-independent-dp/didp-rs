use crate::expression_evaluator::ExpressionEvaluator;
use crate::forward_bfs;
use crate::solver;
use crate::state_registry::StateInRegistry;
use crate::successor_generator::SuccessorGenerator;
use didp_parser::expression_parser::ParseNumericExpression;
use didp_parser::variable;
use std::cmp;
use std::error::Error;
use std::fmt;
use std::str;

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

pub struct ExpressionAstar<T: variable::Numeric> {
    h_expression: Option<String>,
    f_evaluator_type: FEvaluatorType,
    parameters: solver::SolverParameters<T>,
    registry_capacity: Option<usize>,
}

impl<T: variable::Numeric> Default for ExpressionAstar<T> {
    fn default() -> Self {
        ExpressionAstar {
            h_expression: None,
            f_evaluator_type: FEvaluatorType::default(),
            parameters: solver::SolverParameters::default(),
            registry_capacity: Some(1000000),
        }
    }
}

impl<T> solver::Solver<T> for ExpressionAstar<T>
where
    T: variable::Numeric + ParseNumericExpression + Ord + fmt::Display,
    <T as str::FromStr>::Err: fmt::Debug,
{
    fn solve(
        &mut self,
        model: &didp_parser::Model<T>,
    ) -> Result<solver::Solution<T>, Box<dyn Error>> {
        let generator = SuccessorGenerator::<didp_parser::Transition<T>>::new(model, false);
        let h_evaluator = if let Some(h_expression) = self.h_expression.as_ref() {
            ExpressionEvaluator::new(h_expression.clone(), model)?
        } else {
            ExpressionEvaluator::default()
        };
        let solution = match self.f_evaluator_type {
            FEvaluatorType::Plus => {
                let f_evaluator =
                    Box::new(|g, h, _: &StateInRegistry, _: &didp_parser::Model<T>| g + h);
                forward_bfs::forward_bfs(
                    model,
                    generator,
                    h_evaluator,
                    f_evaluator,
                    self.parameters,
                    self.registry_capacity,
                )
            }
            FEvaluatorType::Max => {
                let f_evaluator =
                    Box::new(|g, h, _: &StateInRegistry, _: &didp_parser::Model<T>| cmp::max(g, h));
                forward_bfs::forward_bfs(
                    model,
                    generator,
                    h_evaluator,
                    f_evaluator,
                    self.parameters,
                    self.registry_capacity,
                )
            }
            FEvaluatorType::Min => {
                let f_evaluator =
                    Box::new(|g, h, _: &StateInRegistry, _: &didp_parser::Model<T>| cmp::min(g, h));
                forward_bfs::forward_bfs(
                    model,
                    generator,
                    h_evaluator,
                    f_evaluator,
                    self.parameters,
                    self.registry_capacity,
                )
            }
            FEvaluatorType::Overwrite => {
                let f_evaluator =
                    Box::new(|_, h, _: &StateInRegistry, _: &didp_parser::Model<T>| h);
                forward_bfs::forward_bfs(
                    model,
                    generator,
                    h_evaluator,
                    f_evaluator,
                    self.parameters,
                    self.registry_capacity,
                )
            }
        };
        Ok(solution)
    }

    #[inline]
    fn set_primal_bound(&mut self, primal_bound: T) {
        self.parameters.primal_bound = Some(primal_bound)
    }

    #[inline]
    fn set_time_limit(&mut self, time_limit: u64) {
        self.parameters.time_limit = Some(time_limit)
    }
}

impl<T: variable::Numeric> ExpressionAstar<T> {
    pub fn new(config: &yaml_rust::Yaml) -> Result<ExpressionAstar<T>, Box<dyn Error>>
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
        let h_expression = match map.get(&yaml_rust::Yaml::from_str("h")) {
            Some(yaml_rust::Yaml::String(value)) => Some(value.clone()),
            None => None,
            value => {
                return Err(solver::ConfigErr::new(format!(
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
                    return Err(solver::ConfigErr::new(format!(
                        "unexpected operator for f function `{}`",
                        op
                    ))
                    .into())
                }
            },
            None => FEvaluatorType::default(),
            value => {
                return Err(solver::ConfigErr::new(format!(
                    "expected String, but found `{:?}`",
                    value
                ))
                .into())
            }
        };
        let parameters = solver::SolverParameters::parse_from_map(map)?;
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
        Ok(ExpressionAstar {
            h_expression,
            f_evaluator_type,
            parameters,
            registry_capacity,
        })
    }
}
