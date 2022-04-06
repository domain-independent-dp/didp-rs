use crate::exist_dfs;
use crate::expression_astar;
use crate::expression_beam_search;
use crate::forward_recursion;
use crate::solver;
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
                "expression_beam" => Ok(Box::new(
                    expression_beam_search::ExpressionBeamSearch::new(model, &config)?,
                )),
                "iterative_exist_dfs" => {
                    Ok(Box::new(exist_dfs::IterativeForwardExistDfs::new(&config)?))
                }
                "forward_recursion" => {
                    Ok(Box::new(forward_recursion::ForwardRecursion::new(&config)?))
                }
                "iterative" => Ok(Box::new(IterativeSearch::new(model, &config)?)),
                value => Err(solver::ConfigErr::new(format!("no such solver {:?}", value)).into()),
            },
            Some(value) => Err(solver::ConfigErr::new(format!(
                "expected String, but found {:?}",
                value
            ))
            .into()),
            None => Ok(Box::new(forward_recursion::ForwardRecursion::new(&config)?)),
        }
    }
}

pub struct IterativeSearch<T: variable::Numeric> {
    solvers: Vec<Box<dyn solver::Solver<T>>>,
    ub: Option<T>,
}

impl<T: variable::Numeric + fmt::Display> solver::Solver<T> for IterativeSearch<T> {
    fn solve(&mut self, model: &didp_parser::Model<T>) -> solver::Solution<T> {
        let mut cost = self.ub;
        let mut transitions = Vec::new();
        for solver in &mut self.solvers {
            solver.set_ub(cost);
            let result = solver.solve(model);
            if let Some((ub, incumbent)) = result {
                println!("New UB: {}", ub);
                cost = Some(ub);
                transitions = incumbent;
            } else {
                println!("Failed to find a solution");
            }
        }
        cost.map(|cost| (cost, transitions))
    }
}

impl<T: 'static + variable::Numeric + Ord + fmt::Display> IterativeSearch<T> {
    pub fn new(
        model: &didp_parser::Model<T>,
        config: &yaml_rust::Yaml,
    ) -> Result<IterativeSearch<T>, Box<dyn Error>>
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
        let solver_configs = match map.get(&yaml_rust::Yaml::from_str("solvers")) {
            Some(yaml_rust::Yaml::Array(array)) => array,
            value => {
                return Err(solver::ConfigErr::new(format!(
                    "expected Array, but found `{:?}`",
                    value
                ))
                .into())
            }
        };
        let factory = SolverFactory::default();
        let mut solvers = Vec::new();
        for config in solver_configs {
            solvers.push(factory.create(model, &config)?);
        }
        Ok(IterativeSearch { solvers, ub })
    }
}
