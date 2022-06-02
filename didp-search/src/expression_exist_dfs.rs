use crate::exist_dfs;
use crate::solver;
use crate::successor_generator::SuccessorGenerator;
use didp_parser::expression_parser::ParseNumericExpression;
use didp_parser::variable;
use std::error::Error;
use std::fmt;
use std::str;

#[derive(Default)]
pub struct ExpressionExistDfs<T: variable::Numeric> {
    maximize: bool,
    primal_bound: Option<T>,
    capacity: Option<usize>,
}

impl<'a, T> solver::Solver<T> for ExpressionExistDfs<T>
where
    T: variable::Numeric + ParseNumericExpression + Ord + fmt::Display,
    <T as str::FromStr>::Err: fmt::Debug,
{
    #[inline]
    fn set_primal_bound(&mut self, bound: Option<T>) {
        self.primal_bound = bound;
    }

    #[inline]
    fn solve(
        &mut self,
        model: &didp_parser::Model<T>,
    ) -> Result<solver::Solution<T>, Box<dyn Error>> {
        let generator = SuccessorGenerator::<didp_parser::Transition<T>>::new(model, false);
        Ok(exist_dfs::forward_iterative_exist_dfs(
            model,
            &generator,
            self.primal_bound,
            self.maximize,
            self.capacity,
        ))
    }
}

impl<T: variable::Numeric> ExpressionExistDfs<T> {
    pub fn new(config: &yaml_rust::Yaml) -> Result<ExpressionExistDfs<T>, solver::ConfigErr>
    where
        <T as str::FromStr>::Err: fmt::Debug,
    {
        let map = match config {
            yaml_rust::Yaml::Hash(map) => map,
            yaml_rust::Yaml::Null => return Ok(ExpressionExistDfs::default()),
            _ => {
                return Err(solver::ConfigErr::new(format!(
                    "expected Hash, but found `{:?}`",
                    config
                )))
            }
        };
        let capacity = match map.get(&yaml_rust::Yaml::from_str("capacity")) {
            Some(yaml_rust::Yaml::Integer(value)) => Some(*value as usize),
            None => Some(1000000),
            value => {
                return Err(solver::ConfigErr::new(format!(
                    "expected Integer, but found `{:?}`",
                    value
                )))
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
                )))
            }
        };
        let maximize = match map.get(&yaml_rust::Yaml::from_str("maximize")) {
            Some(yaml_rust::Yaml::Boolean(value)) => *value,
            Some(value) => {
                return Err(solver::ConfigErr::new(format!(
                    "expected Boolean, but found `{:?}`",
                    value
                )))
            }
            None => false,
        };
        Ok(ExpressionExistDfs {
            primal_bound,
            maximize,
            capacity,
        })
    }
}
