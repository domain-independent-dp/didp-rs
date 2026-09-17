//! Command-line validation of a solution against a DyPDL model in YAML format.

use didp_yaml::dypdl_parser::load_model_from_yaml;
use didp_yaml::solution::{
    load_solution_from_file, validate_solution, CostTolerance, SolutionError,
};
use std::env;
use std::ffi::OsString;
use std::fs;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use yaml_rust::{Yaml, YamlLoader};

const USAGE: &str = "Usage: didp-yaml-validator [--rel-tol NUMBER] [--abs-tol NUMBER] DOMAIN PROBLEM SOLUTION\n\nValidate a forward solution and print its computed cost.\nCost comparison is exact by default; tolerances apply only to continuous costs.\nExit codes: 0 valid, 1 invalid solution or cost mismatch, 2 input or evaluation error.";

struct Arguments {
    paths: Vec<PathBuf>,
    tolerance: CostTolerance,
}

fn parse_arguments(mut args: impl Iterator<Item = OsString>) -> Result<Option<Arguments>, String> {
    let mut paths = Vec::new();
    let mut tolerance = CostTolerance::default();
    let mut positional = false;
    while let Some(arg) = args.next() {
        if !positional {
            if arg == "--help" || arg == "-h" {
                return Ok(None);
            }
            if arg == "--" {
                positional = true;
                continue;
            }
            if arg == "--rel-tol" || arg == "--abs-tol" {
                let value = args
                    .next()
                    .and_then(|value| value.into_string().ok())
                    .and_then(|value| value.parse::<f64>().ok())
                    .ok_or_else(|| format!("{} requires a number", arg.to_string_lossy()))?;
                if arg == "--rel-tol" {
                    tolerance.rel_tol = value;
                } else {
                    tolerance.abs_tol = value;
                }
                continue;
            }
            if arg.to_string_lossy().starts_with('-') {
                return Err(format!("Unknown option `{}`", arg.to_string_lossy()));
            }
        }
        paths.push(PathBuf::from(arg));
    }
    if paths.len() != 3 {
        return Err("Expected domain, problem, and solution file paths".to_owned());
    }
    tolerance.validate().map_err(|error| error.to_string())?;
    Ok(Some(Arguments { paths, tolerance }))
}

fn read_document(path: &Path, kind: &str) -> Result<Yaml, String> {
    let source = fs::read_to_string(path)
        .map_err(|error| format!("Could not read {kind} file `{}`: {error}", path.display()))?;
    let mut documents = YamlLoader::load_from_str(&source)
        .map_err(|error| format!("Invalid YAML in {kind} file `{}`: {error}", path.display()))?;
    if documents.len() != 1 {
        return Err(format!(
            "Expected exactly one YAML document in {kind} file `{}`, found {}",
            path.display(),
            documents.len()
        ));
    }
    Ok(documents.remove(0))
}

fn run() -> Result<(), (u8, String)> {
    let Some(arguments) = parse_arguments(env::args_os().skip(1))
        .map_err(|message| (2, format!("{message}\n\n{USAGE}")))?
    else {
        println!("{USAGE}");
        return Ok(());
    };
    let domain = read_document(&arguments.paths[0], "domain").map_err(|message| (2, message))?;
    let problem = read_document(&arguments.paths[1], "problem").map_err(|message| (2, message))?;
    let model = load_model_from_yaml(&domain, &problem)
        .map_err(|error| (2, format!("Could not load the model: {error}")))?;
    let path = &arguments.paths[2];
    let solution = load_solution_from_file(&model, path).map_err(|error| {
        (
            2,
            format!("Could not load solution `{}`: {error}", path.display()),
        )
    })?;
    let cost = validate_solution(&model, &solution, arguments.tolerance).map_err(|error| {
        let code = match &error {
            SolutionError::Validation(dypdl::SolutionValidationError::EvaluationError {
                ..
            }) => 2,
            SolutionError::Validation(_) | SolutionError::CostMismatch { .. } => 1,
            _ => 2,
        };
        (
            code,
            format!("Solution `{}` failed validation: {error}", path.display()),
        )
    })?;
    println!("Solution is valid.\ncost: {cost}");
    Ok(())
}

fn main() -> ExitCode {
    // This standalone executable reports caught model/evaluation failures itself.
    // Library users retain their own process-wide panic hook.
    std::panic::set_hook(Box::new(|_| {}));
    let result = catch_unwind(AssertUnwindSafe(run)).unwrap_or_else(|payload| {
        let message = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| payload.downcast_ref::<&str>().copied())
            .unwrap_or("expression evaluation failed");
        Err((2, format!("Could not evaluate the model: {message}")))
    });
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err((code, message)) => {
            eprintln!("{message}");
            ExitCode::from(code)
        }
    }
}
