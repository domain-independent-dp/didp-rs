use crate::bfs_node::TransitionWithG;
use crate::expression_evaluator::ExpressionEvaluator;
use crate::forward_bfs;
use crate::search_node::StateForSearchNode;
use crate::solver;
use crate::successor_generator::SuccessorGenerator;
use didp_parser::expression_parser::ParseNumericExpression;
use didp_parser::variable;
use rustc_hash::FxHashMap;
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
    g_expressions: Option<FxHashMap<String, String>>,
    h_expression: Option<String>,
    f_evaluator_type: FEvaluatorType,
    primal_bound: Option<T>,
    primal_bound_is_not_g_bound: bool,
    registry_capacity: Option<usize>,
}

impl<T: variable::Numeric> Default for ExpressionAstar<T> {
    fn default() -> Self {
        ExpressionAstar {
            g_expressions: None,
            h_expression: None,
            f_evaluator_type: FEvaluatorType::default(),
            primal_bound: None,
            primal_bound_is_not_g_bound: false,
            registry_capacity: Some(1000000),
        }
    }
}

impl<T> solver::Solver<T> for ExpressionAstar<T>
where
    T: variable::Numeric + ParseNumericExpression + Ord + fmt::Display,
    <T as str::FromStr>::Err: fmt::Debug,
{
    #[inline]
    fn set_primal_bound(&mut self, primal_bound: Option<T>) {
        self.primal_bound = primal_bound
    }

    fn solve(
        &mut self,
        model: &didp_parser::Model<T>,
    ) -> Result<solver::Solution<T>, Box<dyn Error>> {
        match self.g_expressions.as_ref() {
            Some(g_expressions) => {
                if let Ok(generator) =
                    SuccessorGenerator::<TransitionWithG<T, variable::Integer>>::with_expressions(
                        &model,
                        false,
                        g_expressions,
                    )
                {
                    self.solve_inner(model, generator)
                } else {
                    let generator = SuccessorGenerator::<
                        TransitionWithG<T, variable::OrderedContinuous>,
                    >::with_expressions(
                        &model, false, g_expressions
                    )?;
                    self.solve_inner(model, generator)
                }
            }
            None => {
                let generator = SuccessorGenerator::<TransitionWithG<T, T>>::new(&model, false);
                self.solve_inner(model, generator)
            }
        }
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
        let g_expressions = match map.get(&yaml_rust::Yaml::from_str("g")) {
            Some(yaml_rust::Yaml::Hash(map)) => {
                let mut g_expressions = FxHashMap::default();
                g_expressions.reserve(map.len());
                for (key, value) in map.iter() {
                    match (key, value) {
                        (yaml_rust::Yaml::String(key), yaml_rust::Yaml::String(value)) => {
                            g_expressions.insert(key.clone(), value.clone());
                        }
                        _ => {
                            return Err(solver::ConfigErr::new(format!(
                                "expected (String, String), but found (`{:?}`, `{:?}`)",
                                key, value
                            ))
                            .into())
                        }
                    }
                }
                Some(g_expressions)
            }
            None => None,
            value => {
                return Err(solver::ConfigErr::new(format!(
                    "expected Hash, but found `{:?}`",
                    value
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
        let primal_bound_is_not_g_bound =
            match map.get(&yaml_rust::Yaml::from_str("primal_bound_is_not_g_bound")) {
                Some(yaml_rust::Yaml::Boolean(value)) => *value,
                None => false,
                value => {
                    return Err(solver::ConfigErr::new(format!(
                        "expected Boolean, but found `{:?}`",
                        value
                    ))
                    .into())
                }
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
        Ok(ExpressionAstar {
            g_expressions,
            h_expression,
            f_evaluator_type,
            primal_bound,
            primal_bound_is_not_g_bound,
            registry_capacity,
        })
    }

    fn solve_inner<'a, U>(
        &self,
        model: &'a didp_parser::Model<T>,
        generator: SuccessorGenerator<'a, TransitionWithG<T, U>>,
    ) -> Result<solver::Solution<T>, Box<dyn Error>>
    where
        <U as str::FromStr>::Err: fmt::Debug,
        U: variable::Numeric + ParseNumericExpression + Ord + fmt::Display,
    {
        let h_evaluator = if let Some(h_expression) = self.h_expression.as_ref() {
            ExpressionEvaluator::new(h_expression.clone(), &model)?
        } else {
            ExpressionEvaluator::default()
        };
        let g_bound = if self.primal_bound_is_not_g_bound {
            None
        } else {
            self.primal_bound.map(U::from)
        };
        let solution = match self.f_evaluator_type {
            FEvaluatorType::Plus => {
                let f_evaluator =
                    Box::new(|g, h, _: &StateForSearchNode, _: &didp_parser::Model<T>| g + h);
                let evaluators = forward_bfs::BFSEvaluators {
                    generator,
                    h_evaluator,
                    f_evaluator,
                };
                forward_bfs::forward_bfs(model, &evaluators, g_bound, self.registry_capacity)
            }
            FEvaluatorType::Max => {
                let f_evaluator =
                    Box::new(|g, h, _: &StateForSearchNode, _: &didp_parser::Model<T>| {
                        cmp::max(g, h)
                    });
                let evaluators = forward_bfs::BFSEvaluators {
                    generator,
                    h_evaluator,
                    f_evaluator,
                };
                forward_bfs::forward_bfs(model, &evaluators, g_bound, self.registry_capacity)
            }
            FEvaluatorType::Min => {
                let f_evaluator =
                    Box::new(|g, h, _: &StateForSearchNode, _: &didp_parser::Model<T>| {
                        cmp::min(g, h)
                    });
                let evaluators = forward_bfs::BFSEvaluators {
                    generator,
                    h_evaluator,
                    f_evaluator,
                };
                forward_bfs::forward_bfs(model, &evaluators, g_bound, self.registry_capacity)
            }
            FEvaluatorType::Overwrite => {
                let f_evaluator =
                    Box::new(|_, h, _: &StateForSearchNode, _: &didp_parser::Model<T>| h);
                let evaluators = forward_bfs::BFSEvaluators {
                    generator,
                    h_evaluator,
                    f_evaluator,
                };
                forward_bfs::forward_bfs(model, &evaluators, g_bound, self.registry_capacity)
            }
        };
        Ok(solution)
    }
}
