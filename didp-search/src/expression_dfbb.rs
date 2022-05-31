use crate::dfbb;
use crate::expression_astar::FEvaluatorType;
use crate::expression_evaluator::ExpressionEvaluator;
use crate::search_node::StateForSearchNode;
use crate::solver;
use crate::successor_generator::SuccessorGenerator;
use didp_parser::expression_parser::ParseNumericExpression;
use didp_parser::variable;
use didp_parser::Transition;
use std::cmp;
use std::error::Error;
use std::fmt;
use std::str;

pub struct ExpressionDFBB<T: variable::Numeric> {
    h_expression: Option<String>,
    f_evaluator_type: FEvaluatorType,
    primal_bound: Option<T>,
    registry_capacity: Option<usize>,
}

impl<T: variable::Numeric> Default for ExpressionDFBB<T> {
    fn default() -> Self {
        ExpressionDFBB {
            h_expression: None,
            f_evaluator_type: FEvaluatorType::default(),
            primal_bound: None,
            registry_capacity: Some(1000000),
        }
    }
}

impl<T> solver::Solver<T> for ExpressionDFBB<T>
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
        let generator = SuccessorGenerator::<Transition<T>>::new(model, false);
        self.solve_inner(model, generator)
    }
}

impl<T: variable::Numeric> ExpressionDFBB<T> {
    pub fn new(config: &yaml_rust::Yaml) -> Result<ExpressionDFBB<T>, Box<dyn Error>>
    where
        <T as str::FromStr>::Err: fmt::Debug,
    {
        let map = match config {
            yaml_rust::Yaml::Hash(map) => map,
            yaml_rust::Yaml::Null => return Ok(ExpressionDFBB::default()),
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
        Ok(ExpressionDFBB {
            h_expression,
            f_evaluator_type,
            primal_bound,
            registry_capacity,
        })
    }

    fn solve_inner<'a>(
        &self,
        model: &'a didp_parser::Model<T>,
        generator: SuccessorGenerator<'a, Transition<T>>,
    ) -> Result<solver::Solution<T>, Box<dyn Error>>
    where
        <T as str::FromStr>::Err: fmt::Debug,
        T: variable::Numeric + ParseNumericExpression + Ord + fmt::Display,
    {
        let h_evaluator = if let Some(h_expression) = self.h_expression.as_ref() {
            ExpressionEvaluator::new(h_expression.clone(), model)?
        } else {
            ExpressionEvaluator::default()
        };
        let solution = match self.f_evaluator_type {
            FEvaluatorType::Plus => {
                let f_evaluator =
                    Box::new(|g, h, _: &StateForSearchNode, _: &didp_parser::Model<T>| g + h);
                dfbb::dfbb(
                    model,
                    generator,
                    h_evaluator,
                    f_evaluator,
                    self.primal_bound,
                    self.registry_capacity,
                )
            }
            FEvaluatorType::Max => {
                let f_evaluator =
                    Box::new(|g, h, _: &StateForSearchNode, _: &didp_parser::Model<T>| {
                        cmp::max(g, h)
                    });
                dfbb::dfbb(
                    model,
                    generator,
                    h_evaluator,
                    f_evaluator,
                    self.primal_bound,
                    self.registry_capacity,
                )
            }
            FEvaluatorType::Min => {
                let f_evaluator =
                    Box::new(|g, h, _: &StateForSearchNode, _: &didp_parser::Model<T>| {
                        cmp::min(g, h)
                    });
                dfbb::dfbb(
                    model,
                    generator,
                    h_evaluator,
                    f_evaluator,
                    self.primal_bound,
                    self.registry_capacity,
                )
            }
            FEvaluatorType::Overwrite => {
                let f_evaluator =
                    Box::new(|_, h, _: &StateForSearchNode, _: &didp_parser::Model<T>| h);
                dfbb::dfbb(
                    model,
                    generator,
                    h_evaluator,
                    f_evaluator,
                    self.primal_bound,
                    self.registry_capacity,
                )
            }
        };
        Ok(solution)
    }
}
