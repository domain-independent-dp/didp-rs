use didp_yaml::dypdl_parser;
use didp_yaml::heuristic_search_solver;
use dypdl::variable_type;
use dypdl_heuristic_search::Solver;
use std::env;
use std::fmt;
use std::fs;
use std::process;
use std::str;

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

fn solve<T>(model: &dypdl::Model, config: &yaml_rust::Yaml)
where
    T: variable_type::Numeric + Ord + fmt::Display + 'static,
    <T as str::FromStr>::Err: fmt::Debug,
{
    let factory = heuristic_search_solver::SolverFactory::default();
    let mut solver: Box<dyn Solver<T>> = factory.create(config, model).unwrap_or_else(|e| {
        eprintln!("Error: {:?}", e);
        process::exit(1);
    });
    let solution = solver.solve(model).unwrap_or_else(|e| {
        eprintln!("Error: {:?}", e);
        process::exit(1);
    });
    if let Some(cost) = solution.cost {
        println!("transitions:");
        for transition in solution.transitions {
            println!("{}", transition.get_full_name());
        }
        println!("cost: {}", cost);
        if solution.is_optimal {
            println!("optimal cost: {}", cost);
        } else if let Some(bound) = solution.best_bound {
            println!("best bound: {}", bound);
        }
    } else if solution.is_infeasible {
        println!("The problem is infeasible.");
    } else {
        println!("Could not find a solution.");
        if let Some(bound) = solution.best_bound {
            println!("best bound: {}", bound);
        }
    }
    println!("Expanded: {}", solution.expanded);
    println!("Generated: {}", solution.generated);
    println!("Search time: {}", solution.time);
}

fn main() {
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
        eprintln!("Coundn't read a domain file: {:?}", e);
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
        eprintln!("Could'nt read a config file: {:?}", e);
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
    match model.cost_type {
        dypdl::CostType::Integer => solve::<variable_type::Integer>(&model, config),
        dypdl::CostType::Continuous => solve::<variable_type::OrderedContinuous>(&model, config),
    }
}
