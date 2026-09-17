use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

const DOMAIN: &str = "state_variables:\n  - {name: x, type: integer}\nbase_cases:\n  - conditions: ['(= x 0)']\ntransitions:\n  - name: step\n    preconditions: ['(> x 0)']\n    effect: {x: '(- x 1)'}\n    cost: '(+ cost 1)'\n";
const PROBLEM: &str = "target: {x: 2}\n";
const SOLUTION: &str = "cost: 2\ntransitions: [{name: step}, {name: step}]\n";

struct Fixture(PathBuf);

impl Fixture {
    fn new(domain: &str, problem: &str, solution: &str) -> Self {
        static NEXT_ID: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "didp-validator-test-{}-{}",
            std::process::id(),
            NEXT_ID.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        for (name, text) in [
            ("domain.yaml", domain),
            ("problem.yaml", problem),
            ("solution.yaml", solution),
        ] {
            fs::write(path.join(name), text).unwrap();
        }
        Self(path)
    }

    fn run(&self, args: &[&str]) -> Output {
        Command::new(env!("CARGO_BIN_EXE_didp-yaml-validator"))
            .args(args)
            .arg(self.0.join("domain.yaml"))
            .arg(self.0.join("problem.yaml"))
            .arg(self.0.join("solution.yaml"))
            .output()
            .unwrap()
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn valid_solution_reports_computed_cost_without_writing_files() {
    let fixture = Fixture::new(DOMAIN, PROBLEM, SOLUTION);
    let output = fixture.run(&[]);
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "Solution is valid.\ncost: 2\n"
    );
    assert!(output.stderr.is_empty());
    assert_eq!(
        fs::read_to_string(fixture.0.join("solution.yaml")).unwrap(),
        SOLUTION
    );
}

#[test]
fn invalid_solutions_and_declared_costs_exit_one() {
    for (solution, message) in [
        (
            "transitions: [{name: step}]",
            "does not satisfy any base case",
        ),
        (
            "transitions: [{name: step}, {name: step}, {name: step}]",
            "transitions remain",
        ),
        (
            "cost: 7\ntransitions: [{name: step}, {name: step}]",
            "declared cost 7 does not match the computed cost 2",
        ),
    ] {
        let output = Fixture::new(DOMAIN, PROBLEM, solution).run(&[]);
        assert_eq!(output.status.code(), Some(1));
        assert!(String::from_utf8_lossy(&output.stderr).contains(message));
        assert!(output.stdout.is_empty());
    }
}

#[test]
fn malformed_input_and_unresolved_transitions_exit_two() {
    for (domain, problem, solution, message) in [
        (
            DOMAIN,
            PROBLEM,
            "transitions: [{name: unknown}]",
            "no forward transition",
        ),
        (DOMAIN, PROBLEM, "[", "invalid YAML"),
        ("", PROBLEM, SOLUTION, "exactly one YAML document"),
        (
            DOMAIN,
            "---\ntarget: {x: 2}\n---\n{}",
            SOLUTION,
            "exactly one YAML document",
        ),
        ("{}", PROBLEM, SOLUTION, "Could not load the model"),
    ] {
        let output = Fixture::new(domain, problem, solution).run(&[]);
        assert_eq!(output.status.code(), Some(2));
        assert!(
            String::from_utf8_lossy(&output.stderr).contains(message),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn continuous_tolerance_is_explicit() {
    let domain = format!(
        "cost_type: continuous\n{}",
        DOMAIN.replace("(+ cost 1)", "(+ cost 0.1)")
    );
    let fixture = Fixture::new(
        &domain,
        "target: {x: 3}",
        "cost: 0.3\ntransitions: [{name: step}, {name: step}, {name: step}]",
    );
    assert_eq!(fixture.run(&[]).status.code(), Some(1));
    assert_eq!(fixture.run(&["--rel-tol", "1e-9"]).status.code(), Some(0));
    assert_eq!(fixture.run(&["--abs-tol", "1e-9"]).status.code(), Some(0));
    assert_eq!(fixture.run(&["--abs-tol", "-1"]).status.code(), Some(2));
}

#[test]
fn evaluation_failures_are_readable() {
    let domain = DOMAIN.replace("(+ cost 1)", "(/ cost (- x 1))");
    let output = Fixture::new(&domain, PROBLEM, SOLUTION).run(&[]);
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("cost of transition 2 (`step`)"), "{stderr}");
    assert!(!stderr.contains("panicked"));
}

#[test]
fn help_and_missing_arguments() {
    let executable = env!("CARGO_BIN_EXE_didp-yaml-validator");
    let output = Command::new(executable).arg("--help").output().unwrap();
    assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stdout).contains("Usage:"));
    let output = Command::new(executable).output().unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("Expected domain, problem, and solution")
    );
}
