use crate::exist_dfs;
use crate::expression_astar;
use crate::forward_recursion;
use crate::solver;
use crate::util;
use didp_parser::variable;
use std::error::Error;
use std::fmt;
use std::str;

#[derive(Default)]
pub struct SolverFactory;

impl SolverFactory {
    pub fn create<'a, T: 'static + variable::Numeric + Ord + fmt::Display>(
        &self,
        model: &'a didp_parser::Model<T>,
        config: &yaml_rust::Yaml,
    ) -> Result<Box<dyn solver::Solver<T>>, Box<dyn Error>>
    where
        <T as str::FromStr>::Err: fmt::Debug,
    {
        let map = match config {
            yaml_rust::Yaml::Hash(map) => map,
            _ => {
                return Err(solver::ConfigErr::new(format!(
                    "expected Hash, but found {:?}",
                    config
                ))
                .into());
            }
        };
        let config = match map.get(&yaml_rust::Yaml::from_str("config")) {
            Some(value) => value.clone(),
            None => yaml_rust::Yaml::Null,
        };
        match map.get(&yaml_rust::Yaml::from_str("solver")) {
            Some(yaml_rust::Yaml::String(string)) => match &string[..] {
                "expression_astar" => Ok(Box::new(expression_astar::ExpressionAstar::new(
                    model, &config,
                )?)),
                "forward_recursion" => Ok(Box::new(forward_recursion::ForwardRecursion::new(
                    model, &config,
                )?)),
                value => Err(util::ConfigErr::new(format!("no such solver {:?}", value)).into()),
            },
            Some(value) => {
                Err(util::ConfigErr::new(format!("expected String, but found {:?}", value)).into())
            }
            None => Ok(Box::new(forward_recursion::ForwardRecursion::new(
                model, &config,
            )?)),
        }
    }
}
