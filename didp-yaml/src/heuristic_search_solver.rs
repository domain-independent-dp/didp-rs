//! A module for heuristic search solvers for DyPDL.

mod caasdy;
mod callback;
mod dijkstra;
mod dual_bound_anytime_beam_search;
mod dual_bound_bfdfbb;
mod dual_bound_cbfs;
mod dual_bound_chokudai_search;
mod dual_bound_dbdfs;
mod dual_bound_dds;
mod dual_bound_dfbb;
mod dual_bound_lookahead_bfs;
mod dual_bound_rcbfs;
mod dual_bound_weighted_astar;
mod expression_beam_search;
mod expression_evaluator;
mod forward_recursion;
mod ibdfs;
mod iterative_search;
mod lazy_dijkstra;
mod lookahead_lazy_dijkstra;
mod solver_parameters;
mod transition_with_custom_cost;

use crate::util;
use dypdl::variable_type;
use std::error::Error;
use std::fmt;
use std::str;

/// Factory of a heuristic search solver.
#[derive(Default)]
pub struct SolverFactory;

impl SolverFactory {
    /// Returns a heuristic search solver specified by a YAML configuration file.
    ///
    /// # Errors
    ///
    /// if the format is invalid.
    pub fn create<T>(
        &self,
        config: &yaml_rust::Yaml,
        model: &dypdl::Model,
    ) -> Result<Box<dyn dypdl_heuristic_search::Solver<T>>, Box<dyn Error>>
    where
        T: variable_type::Numeric + Ord + fmt::Display + 'static,
        <T as str::FromStr>::Err: fmt::Debug,
    {
        let map = match config {
            yaml_rust::Yaml::Hash(map) => map,
            _ => {
                return Err(util::YamlContentErr::new(format!(
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
                "dijkstra" => Ok(Box::new(dijkstra::load_from_yaml(&config)?)),
                "lazy_dijkstra" => Ok(Box::new(lazy_dijkstra::load_from_yaml(&config)?)),
                "caasdy" => Ok(Box::new(caasdy::load_from_yaml(&config)?)),
                "expression_beam" => Ok(Box::new(expression_beam_search::load_from_yaml(
                    &config, model,
                )?)),
                "ibdfs" => Ok(Box::new(ibdfs::load_from_yaml(&config)?)),
                "dual_bound_weighted_astar" => Ok(Box::new(
                    dual_bound_weighted_astar::load_from_yaml(&config)?,
                )),
                "dual_bound_dfbb" => Ok(Box::new(dual_bound_dfbb::load_from_yaml(&config)?)),
                "dual_bound_bfdfbb" => Ok(Box::new(dual_bound_bfdfbb::load_from_yaml(&config)?)),
                "dual_bound_cbfs" => Ok(Box::new(dual_bound_cbfs::load_from_yaml(&config)?)),
                "dual_bound_rrcbfs" => Ok(Box::new(dual_bound_rcbfs::load_from_yaml(&config)?)),
                "dual_bound_dds" => Ok(Box::new(dual_bound_dds::load_from_yaml(&config)?)),
                "dual_bound_dbdfs" => Ok(Box::new(dual_bound_dbdfs::load_from_yaml(&config)?)),
                "dual_bound_anytime_beam" => Ok(Box::new(
                    dual_bound_anytime_beam_search::load_from_yaml(&config)?,
                )),
                "dual_bound_chokudai" => Ok(Box::new(dual_bound_chokudai_search::load_from_yaml(
                    &config,
                )?)),
                "dual_bound_lookahead_bfs" => {
                    Ok(Box::new(dual_bound_lookahead_bfs::load_from_yaml(&config)?))
                }
                "lookahead_lazy_dijkstra" => {
                    Ok(Box::new(lookahead_lazy_dijkstra::load_from_yaml(&config)?))
                }
                "forward_recursion" => Ok(Box::new(forward_recursion::load_from_yaml(&config)?)),
                "iterative" => Ok(Box::new(iterative_search::load_from_yaml(&config, model)?)),
                value => {
                    Err(util::YamlContentErr::new(format!("no such solver {:?}", value,)).into())
                }
            },
            Some(value) => Err(util::YamlContentErr::new(format!(
                "expected String, but found {:?}",
                value
            ))
            .into()),
            None => Ok(Box::new(forward_recursion::load_from_yaml(&config)?)),
        }
    }
}
