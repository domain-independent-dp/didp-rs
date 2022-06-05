use super::expression_astar::FEvaluatorType;
use crate::expression_evaluator::ExpressionEvaluator;
use crate::forward_beam_search;
use crate::solver;
use crate::state_registry::StateInRegistry;
use crate::successor_generator::SuccessorGenerator;
use crate::transition_with_custom_cost::TransitionWithCustomCost;
use didp_parser::expression_parser::ParseNumericExpression;
use didp_parser::variable;
use rustc_hash::FxHashMap;
use std::cmp;
use std::error::Error;
use std::fmt;
use std::str;

pub struct ExpressionBeamSearch<T> {
    g_expressions: Option<FxHashMap<String, String>>,
    h_expression: Option<String>,
    f_evaluator_type: FEvaluatorType,
    beam_sizes: Vec<usize>,
    maximize: bool,
    parameters: solver::SolverParameters<T>,
}

impl<T: Default> Default for ExpressionBeamSearch<T> {
    fn default() -> Self {
        ExpressionBeamSearch {
            g_expressions: None,
            h_expression: None,
            f_evaluator_type: FEvaluatorType::default(),
            beam_sizes: vec![10000],
            maximize: false,
            parameters: solver::SolverParameters::default(),
        }
    }
}

impl<T> solver::Solver<T> for ExpressionBeamSearch<T>
where
    T: variable::Numeric + ParseNumericExpression + Ord + fmt::Display,
    <T as str::FromStr>::Err: fmt::Debug,
{
    fn solve(
        &mut self,
        model: &didp_parser::Model<T>,
    ) -> Result<solver::Solution<T>, Box<dyn Error>> {
        match self.g_expressions.as_ref() {
            Some(g_expressions) => {
                if let Ok(generator) = SuccessorGenerator::<
                    TransitionWithCustomCost<T, variable::Integer>,
                >::with_expressions(
                    model, false, g_expressions
                ) {
                    self.solve_inner(model, generator)
                } else {
                    let generator =
                        SuccessorGenerator::<
                            TransitionWithCustomCost<T, variable::OrderedContinuous>,
                        >::with_expressions(model, false, g_expressions)?;
                    self.solve_inner(model, generator)
                }
            }
            None => {
                let generator =
                    SuccessorGenerator::<TransitionWithCustomCost<T, T>>::new(model, false);
                self.solve_inner(model, generator)
            }
        }
    }

    #[inline]
    fn set_primal_bound(&mut self, primal_bound: T) {
        self.parameters.primal_bound = Some(primal_bound)
    }

    #[inline]
    fn set_time_limit(&mut self, time_limit: u64) {
        self.parameters.time_limit = Some(time_limit)
    }

    #[inline]
    fn get_primal_bound(&self) -> Option<T> {
        self.parameters.primal_bound
    }

    #[inline]
    fn get_time_limit(&self) -> Option<u64> {
        self.parameters.time_limit
    }
}

impl<T: variable::Numeric + fmt::Display> ExpressionBeamSearch<T> {
    pub fn new(config: &yaml_rust::Yaml) -> Result<ExpressionBeamSearch<T>, Box<dyn Error>>
    where
        <T as str::FromStr>::Err: fmt::Debug,
    {
        let map = match config {
            yaml_rust::Yaml::Hash(map) => map,
            yaml_rust::Yaml::Null => return Ok(ExpressionBeamSearch::default()),
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
            None => vec![10000],
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
            None => false,
        };
        let parameters = solver::SolverParameters::parse_from_map(map)?;
        Ok(ExpressionBeamSearch {
            g_expressions,
            h_expression,
            f_evaluator_type,
            beam_sizes,
            maximize,
            parameters,
        })
    }

    fn solve_inner<'a, U>(
        &self,
        model: &'a didp_parser::Model<T>,
        generator: SuccessorGenerator<'a, TransitionWithCustomCost<T, U>>,
    ) -> Result<solver::Solution<T>, Box<dyn Error>>
    where
        <U as str::FromStr>::Err: fmt::Debug,
        U: variable::Numeric + ParseNumericExpression + Ord + fmt::Display,
    {
        let h_evaluator = if let Some(h_expression) = self.h_expression.as_ref() {
            ExpressionEvaluator::new(h_expression.clone(), model)?
        } else {
            ExpressionEvaluator::default()
        };
        let solution = match self.f_evaluator_type {
            FEvaluatorType::Plus => {
                let f_evaluator =
                    Box::new(|g, h, _: &StateInRegistry, _: &didp_parser::Model<T>| g + h);
                forward_beam_search::iterative_forward_beam_search(
                    model,
                    generator,
                    h_evaluator,
                    f_evaluator,
                    &self.beam_sizes,
                    self.maximize,
                    self.parameters,
                )
            }
            FEvaluatorType::Max => {
                let f_evaluator =
                    Box::new(|g, h, _: &StateInRegistry, _: &didp_parser::Model<T>| cmp::max(g, h));
                forward_beam_search::iterative_forward_beam_search(
                    model,
                    generator,
                    h_evaluator,
                    f_evaluator,
                    &self.beam_sizes,
                    self.maximize,
                    self.parameters,
                )
            }
            FEvaluatorType::Min => {
                let f_evaluator =
                    Box::new(|g, h, _: &StateInRegistry, _: &didp_parser::Model<T>| cmp::min(g, h));
                forward_beam_search::iterative_forward_beam_search(
                    model,
                    generator,
                    h_evaluator,
                    f_evaluator,
                    &self.beam_sizes,
                    self.maximize,
                    self.parameters,
                )
            }
            FEvaluatorType::Overwrite => {
                let f_evaluator =
                    Box::new(|_, h, _: &StateInRegistry, _: &didp_parser::Model<T>| h);
                forward_beam_search::iterative_forward_beam_search(
                    model,
                    generator,
                    h_evaluator,
                    f_evaluator,
                    &self.beam_sizes,
                    self.maximize,
                    self.parameters,
                )
            }
        };
        Ok(solution)
    }
}
