use crate::dijkstra;
use crate::expression_astar;
use crate::expression_beam_search;
use crate::expression_dfbb;
use crate::expression_exist_dfs;
use crate::forward_recursion;
use crate::lazy_dijkstra;
use crate::solver;
use didp_parser::expression_parser::ParseNumericExpression;
use didp_parser::variable;
use std::error::Error;
use std::fmt;
use std::str;

#[derive(Default)]
pub struct SolverFactory;

impl SolverFactory {
    pub fn create<T>(
        &self,
        config: &yaml_rust::Yaml,
    ) -> Result<Box<dyn solver::Solver<T>>, Box<dyn Error>>
    where
        T: 'static + variable::Numeric + ParseNumericExpression + Ord + fmt::Display,
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
                "dijkstra" => Ok(Box::new(dijkstra::Dijkstra::new(&config)?)),
                "lazy_dijkstra" => Ok(Box::new(lazy_dijkstra::LazyDijkstra::new(&config)?)),
                "expression_astar" => {
                    Ok(Box::new(expression_astar::ExpressionAstar::new(&config)?))
                }
                "expression_beam" => Ok(Box::new(
                    expression_beam_search::ExpressionBeamSearch::new(&config)?,
                )),
                "expression_exist_dfs" => Ok(Box::new(
                    expression_exist_dfs::ExpressionExistDfs::new(&config)?,
                )),
                "expression_dfbb" => Ok(Box::new(expression_dfbb::ExpressionDFBB::new(&config)?)),
                "forward_recursion" => {
                    Ok(Box::new(forward_recursion::ForwardRecursion::new(&config)?))
                }
                "iterative" => Ok(Box::new(IterativeSearch::new(&config)?)),
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
    parameters: solver::SolverParameters<T>,
}

impl<T: variable::Numeric + fmt::Display> solver::Solver<T> for IterativeSearch<T> {
    fn solve(
        &mut self,
        model: &didp_parser::Model<T>,
    ) -> Result<solver::Solution<T>, Box<dyn Error>> {
        let time_keeper = self.parameters.time_limit.map(solver::TimeKeeper::new);
        let mut primal_bound = self.parameters.primal_bound;
        let mut solution = solver::Solution::default();
        for solver in &mut self.solvers {
            if let Some(bound) = primal_bound {
                solver.set_primal_bound(bound);
            }
            if let Some(time_keeper) = time_keeper.as_ref() {
                let time_limit = time_keeper.remaining_time_limit();
                solver.set_time_limit(time_limit.as_secs());
            }
            let result = solver.solve(model)?;
            if let Some(bound) = result.cost {
                println!("New primal bound: {}", bound);
                primal_bound = Some(bound);
                solution = result;
            } else {
                println!("Failed to find a solution");
            }
        }
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

impl<T> IterativeSearch<T>
where
    T: 'static + variable::Numeric + ParseNumericExpression + Ord + fmt::Display,
{
    pub fn new(config: &yaml_rust::Yaml) -> Result<IterativeSearch<T>, Box<dyn Error>>
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
        let parameters = solver::SolverParameters::parse_from_map(map)?;
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
            solvers.push(factory.create(config)?);
        }
        Ok(IterativeSearch {
            solvers,
            parameters,
        })
    }
}
