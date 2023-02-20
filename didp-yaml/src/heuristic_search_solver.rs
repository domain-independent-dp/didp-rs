//! A module for heuristic search solvers for DyPDL.

mod caasdy;
mod dijkstra;
mod dual_bound_acps;
mod dual_bound_apps;
mod dual_bound_breadth_first_search;
mod dual_bound_cabs;
mod dual_bound_cbfs;
mod dual_bound_dbdfs;
mod dual_bound_dfbb;
mod dual_bound_weighted_astar;
mod expression_beam_search;
mod forward_recursion;
mod solution;
mod solver_parameters;
mod transition_with_custom_cost;

use crate::util;
use dypdl::variable_type;
use std::error::Error;
use std::fmt;
use std::fs::OpenOptions;
use std::io::Write;
use std::str;

pub use solution::{CostToDump, SolutionToDump};

/// Returns a heuristic search solver specified by a YAML configuration file.
///
/// # Errors
///
/// If the format is invalid.
pub fn create_solver<T>(
    solver_name: &str,
    config: &yaml_rust::Yaml,
    model: dypdl::Model,
) -> Result<Box<dyn dypdl_heuristic_search::Search<T>>, Box<dyn Error>>
where
    T: variable_type::Numeric + Ord + fmt::Display + 'static + Send + Sync,
    <T as str::FromStr>::Err: fmt::Debug,
    CostToDump: From<T>,
{
    match solver_name {
        "caasdy" => caasdy::load_from_yaml(model, config),
        "dual_bound_cabs" => dual_bound_cabs::load_from_yaml(model, config),
        "dual_bound_dfbb" => dual_bound_dfbb::load_from_yaml(model, config),
        "dual_bound_cbfs" => dual_bound_cbfs::load_from_yaml(model, config),
        "dual_bound_acps" => dual_bound_acps::load_from_yaml(model, config),
        "dual_bound_apps" => dual_bound_apps::load_from_yaml(model, config),
        "dual_bound_dbdfs" => dual_bound_dbdfs::load_from_yaml(model, config),
        "forward_recursion" => forward_recursion::load_from_yaml(model, config),
        "dijkstra" => dijkstra::load_from_yaml(model, config),
        "dual_bound_breadth_first_search" => {
            dual_bound_breadth_first_search::load_from_yaml(model, config)
        }
        "dual_bound_weighted_astar" => dual_bound_weighted_astar::load_from_yaml(model, config),
        "expression_beam_search" => expression_beam_search::load_from_yaml(model, config),
        _ => Err(util::YamlContentErr::new(format!("No such solver: {}", solver_name)).into()),
    }
}

/// Solve a problem and dump the history of the search.
///
/// # Errors
///
/// If the solver causes an error or files cannot be opened.
pub fn solve_and_dump_solutions<T>(
    mut solver: Box<dyn dypdl_heuristic_search::Search<T>>,
    history_filename: &str,
    solution_filename: &str,
) -> Result<dypdl_heuristic_search::Solution<T>, Box<dyn Error>>
where
    T: variable_type::Numeric + Ord + fmt::Display + 'static,
    <T as str::FromStr>::Err: fmt::Debug,
    CostToDump: From<T>,
{
    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(history_filename)?;

    loop {
        let (solution, terminated) = solver.search_next()?;

        if let Some(cost) = solution.cost {
            let line = format!(
                "{}, {}, {}, {}\n",
                solution.time, cost, solution.expanded, solution.generated
            );
            file.write_all(line.as_bytes())?;
            let solution_to_dump = SolutionToDump::from(solution.clone());
            solution_to_dump.dump_to_file(solution_filename)?;
        }

        if terminated {
            return Ok(solution);
        }
    }
}
