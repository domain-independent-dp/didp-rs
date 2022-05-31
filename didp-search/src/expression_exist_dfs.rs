use crate::bfs_node::TransitionWithG;
use crate::exist_dfs;
use crate::solver;
use crate::successor_generator::SuccessorGenerator;
use didp_parser::expression_parser::ParseNumericExpression;
use didp_parser::variable;
use rustc_hash::FxHashMap;
use std::error::Error;
use std::fmt;
use std::str;

#[derive(Default)]
pub struct ExpressionExistDfs<T: variable::Numeric> {
    g_expressions: Option<FxHashMap<String, String>>,
    maximize: bool,
    primal_bound: Option<T>,
    primal_bound_is_not_g_bound: bool,
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
        let solution = match self.g_expressions.as_ref() {
            Some(g_expressions) => {
                if let Ok(generator) =
                    SuccessorGenerator::<TransitionWithG<T, variable::Integer>>::with_expressions(
                        model,
                        false,
                        g_expressions,
                    )
                {
                    let g_bound = if self.primal_bound_is_not_g_bound {
                        None
                    } else {
                        self.primal_bound.map(|x| x.to_integer())
                    };
                    exist_dfs::forward_iterative_exist_dfs(
                        model,
                        &generator,
                        g_bound,
                        self.maximize,
                        self.capacity,
                    )
                } else {
                    let generator = SuccessorGenerator::<
                        TransitionWithG<T, variable::OrderedContinuous>,
                    >::with_expressions(
                        model, false, g_expressions
                    )?;
                    let g_bound = if self.primal_bound_is_not_g_bound {
                        None
                    } else {
                        self.primal_bound
                            .map(|x| ordered_float::OrderedFloat(x.to_continuous()))
                    };
                    exist_dfs::forward_iterative_exist_dfs(
                        model,
                        &generator,
                        g_bound,
                        self.maximize,
                        self.capacity,
                    )
                }
            }
            None => {
                let generator = SuccessorGenerator::<TransitionWithG<T, T>>::new(model, false);
                let g_bound = if self.primal_bound_is_not_g_bound {
                    None
                } else {
                    self.primal_bound.map(T::from)
                };
                exist_dfs::forward_iterative_exist_dfs(
                    model,
                    &generator,
                    g_bound,
                    self.maximize,
                    self.capacity,
                )
            }
        };
        Ok(solution)
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
                            )))
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
        let primal_bound_is_not_g_bound =
            match map.get(&yaml_rust::Yaml::from_str("primal_bound_is_not_g_bound")) {
                Some(yaml_rust::Yaml::Boolean(value)) => *value,
                None => false,
                value => {
                    return Err(solver::ConfigErr::new(format!(
                        "expected Boolean, but found `{:?}`",
                        value
                    )))
                }
            };
        Ok(ExpressionExistDfs {
            g_expressions,
            primal_bound,
            maximize,
            primal_bound_is_not_g_bound,
            capacity,
        })
    }
}
