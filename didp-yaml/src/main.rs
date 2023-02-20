use didp_yaml::dypdl_parser;
use didp_yaml::heuristic_search_solver;
use didp_yaml::heuristic_search_solver::{CostToDump, SolutionToDump};
use didp_yaml::util;
use dypdl::variable_type;
use dypdl_heuristic_search::Search;
use std::env;
use std::error::Error;
use std::fmt;
use std::fs;
use std::process;
use std::str;
use std::time;

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

fn load_config(
    config: &yaml_rust::Yaml,
) -> Result<(String, yaml_rust::Yaml, Option<String>), Box<dyn Error>> {
    let map = match config {
        yaml_rust::Yaml::Hash(map) => map,
        _ => {
            return Err(util::YamlContentErr::new(format!(
                "expected Hash for the config file, but found {:?}",
                config
            ))
            .into());
        }
    };
    let solver_name = match map.get(&yaml_rust::Yaml::from_str("solver")) {
        Some(yaml_rust::Yaml::String(value)) => value.clone(),
        value => {
            return Err(util::YamlContentErr::new(format!(
                "expected String for `solver`, but found {:?}",
                value
            ))
            .into())
        }
    };
    let config = match map.get(&yaml_rust::Yaml::from_str("config")) {
        Some(value) => value.clone(),
        None => yaml_rust::Yaml::Null,
    };
    let dump_filename = match map.get(&yaml_rust::Yaml::from_str("dump_to")) {
        Some(yaml_rust::Yaml::String(value)) => Some(value.clone()),
        Some(value) => {
            return Err(util::YamlContentErr::new(format!(
                "expected String for `dump_to`, but found {:?}",
                value
            ))
            .into())
        }
        _ => None,
    };
    Ok((solver_name, config, dump_filename))
}

fn solve<T>(mut solver: Box<dyn Search<T>>, dump_filename: Option<&str>, solution_filename: &str)
where
    T: variable_type::Numeric + Ord + fmt::Display + 'static,
    <T as str::FromStr>::Err: fmt::Debug,
    CostToDump: From<T>,
{
    let solution = if let Some(dump_filename) = dump_filename {
        heuristic_search_solver::solve_and_dump_solutions(solver, dump_filename, solution_filename)
    } else {
        solver.search()
    };
    let solution = solution.unwrap_or_else(|e| {
        eprintln!("Failed to run the solver: {:}", e);
        process::exit(1);
    });

    let expanded = solution.expanded;
    let generated = solution.generated;
    let search_time = solution.time;

    if let Some(cost) = solution.cost {
        println!("transitions:");
        for transition in &solution.transitions {
            println!("{}", transition.get_full_name());
        }

        println!("cost: {}", cost);
        if solution.is_optimal {
            println!("optimal cost: {}", cost);
        } else if let Some(bound) = solution.best_bound {
            println!("best bound: {}", bound);
        }

        let solution = SolutionToDump::from(solution);
        solution
            .dump_to_file(solution_filename)
            .unwrap_or_else(|e| {
                eprintln!("Failed to dump a solution: {:?}", e);
                process::exit(1);
            });
    } else if solution.is_infeasible {
        println!("The problem is infeasible.");
    } else {
        println!("Could not find a solution.");

        if let Some(bound) = solution.best_bound {
            println!("best bound: {}", bound);
        }
    }

    println!("Expanded: {}", expanded);
    println!("Generated: {}", generated);
    println!("Search time: {}s", search_time);
}

fn main() {
    let start = time::Instant::now();

    let mut args = env::args();
    args.next();

    let domain = args.next().unwrap_or_else(|| {
        eprintln!("Didn't get a domain file name.");
        process::exit(1);
    });
    let problem = args.next().unwrap_or_else(|| {
        eprintln!("Didn't get a problem file name.");
        process::exit(1);
    });
    let config = args.next().unwrap_or_else(|| {
        eprintln!("Didn't get a config file name.");
        process::exit(1);
    });
    let domain = fs::read_to_string(domain).unwrap_or_else(|e| {
        eprintln!("Couldn't read a domain file: {:?}", e);
        process::exit(1);
    });
    let domain = yaml_rust::YamlLoader::load_from_str(&domain).unwrap_or_else(|e| {
        eprintln!("Couldn't read a domain file: {:?}", e);
        process::exit(1);
    });
    assert_eq!(domain.len(), 1);
    let domain = &domain[0];
    let problem = fs::read_to_string(problem).unwrap_or_else(|e| {
        eprintln!("Could'nt read a problem file: {:?}", e);
        process::exit(1);
    });
    let problem = yaml_rust::YamlLoader::load_from_str(&problem).unwrap_or_else(|e| {
        eprintln!("Couldn't read a problem file: {:?}", e);
        process::exit(1);
    });
    assert_eq!(problem.len(), 1);
    let problem = &problem[0];
    let config = fs::read_to_string(config).unwrap_or_else(|e| {
        eprintln!("Couldn't read a config file: {:?}", e);
        process::exit(1);
    });
    let config = yaml_rust::YamlLoader::load_from_str(&config).unwrap_or_else(|e| {
        eprintln!("Couldn't read a config file: {:?}", e);
        process::exit(1);
    });
    assert_eq!(config.len(), 1);
    let config = &config[0];
    let model = dypdl_parser::load_model_from_yaml(domain, problem).unwrap_or_else(|e| {
        eprintln!("Couldn't load a model: {:?}", e);
        process::exit(1);
    });

    let (solver_name, config, dump_filename) = load_config(config).unwrap_or_else(|e| {
        eprintln!("Couldn't load a config: {:?}", e);
        process::exit(1);
    });

    match model.cost_type {
        dypdl::CostType::Integer => {
            let solver: Box<dyn Search<variable_type::Integer>> =
                heuristic_search_solver::create_solver(&solver_name, &config, model)
                    .unwrap_or_else(|e| {
                        eprintln!("Couldn't load a solver: {:?}", e);
                        process::exit(1);
                    });
            let end = time::Instant::now();
            println!("Preparing time: {}s", (end - start).as_secs_f64());
            solve(solver, dump_filename.as_deref(), "solution.yaml");
        }
        dypdl::CostType::Continuous => {
            let solver: Box<dyn Search<variable_type::OrderedContinuous>> =
                heuristic_search_solver::create_solver(&solver_name, &config, model)
                    .unwrap_or_else(|e| {
                        eprintln!("Couldn't load a solver: {:?}", e);
                        process::exit(1);
                    });
            let end = time::Instant::now();
            println!("Preparing time: {}s", (end - start).as_secs_f64());
            solve(solver, dump_filename.as_deref(), "solution.yaml");
        }
    }
}
