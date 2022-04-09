use didp_parser::expression_parser::ParseNumericExpression;
use didp_parser::variable;
use didp_search::solver_factory;
use std::env;
use std::fmt;
use std::fs;
use std::process;
use std::str;
use std::time::Instant;

fn solve<T>(model: &didp_parser::Model<T>, config: &yaml_rust::Yaml)
where
    T: 'static + variable::Numeric + ParseNumericExpression + Ord + fmt::Display,
    <T as str::FromStr>::Err: fmt::Debug,
{
    let factory = solver_factory::SolverFactory::default();
    let mut solver = factory.create(config).unwrap_or_else(|e| {
        eprintln!("Error: {:?}", e);
        process::exit(1);
    });
    let solution = solver.solve(&model).unwrap_or_else(|e| {
        eprintln!("Error: {:?}", e);
        process::exit(1);
    });
    match solution {
        Some((cost, transitions)) => {
            println!("transitions:");
            for transition in transitions {
                println!("{}", transition.get_full_name());
            }
            println!("cost: {}", cost);
        }
        None => {
            println!("no solution");
        }
    }
}

fn main() {
    let start = Instant::now();
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
        eprintln!("Coundn't read a problem file: {:?}", e);
        process::exit(1);
    });
    assert_eq!(problem.len(), 1);
    let problem = &problem[0];
    let config = fs::read_to_string(config).unwrap_or_else(|e| {
        eprintln!("Could'nt read a config file: {:?}", e);
        process::exit(1);
    });
    let config = yaml_rust::YamlLoader::load_from_str(&config).unwrap_or_else(|e| {
        eprintln!("Coundn't read a config file: {:?}", e);
        process::exit(1);
    });
    assert_eq!(config.len(), 1);
    let config = &config[0];
    let model =
        didp_parser::CostWrappedModel::load_from_yaml(domain, problem).unwrap_or_else(|e| {
            eprintln!("Coundn't load a model: {:?}", e);
            process::exit(1);
        });
    match model {
        didp_parser::CostWrappedModel::Integer(model) => solve(&model, config),
        didp_parser::CostWrappedModel::OrderedContinuous(model) => solve(&model, config),
    }
    let stop = Instant::now();
    println!("Time: {:?}", stop.duration_since(start));
}
