use didp_yaml::solution::{
    load_solution_from_str, load_solution_from_yaml, validate_cost, validate_solution,
    CostTolerance, LoadedSolution, SolutionCost, SolutionError, SolutionToDump,
};
use dypdl::prelude::*;
use dypdl_heuristic_search::Solution;

fn model() -> Model {
    let mut model = Model::default();
    let x = model.add_integer_variable("x", 2).unwrap();
    model
        .add_base_case(vec![Condition::comparison_i(ComparisonOperator::Eq, x, 0)])
        .unwrap();
    let mut step = Transition::new("visit");
    step.parameter_names = vec!["z".to_owned(), "a".to_owned()];
    step.parameter_values = vec![1, 0];
    step.add_precondition(Condition::comparison_i(ComparisonOperator::Gt, x, 0));
    step.add_effect(x, x - 1).unwrap();
    step.set_cost(IntegerExpression::Cost + 1);
    model.add_forward_transition(step).unwrap();
    model
}

const TWO_STEPS: &str = "transitions:\n  - {name: visit, parameters: {a: 0, z: 1}}\n  - {name: visit, parameters: {z: 1, a: 0}}\n";

#[test]
fn load_and_validate_with_parameters_in_different_orders() {
    let model = model();
    let solution = load_solution_from_str(&model, &format!("cost: 2\n{TWO_STEPS}")).unwrap();
    assert_eq!(solution.cost, Some(SolutionCost::Integer(2)));
    assert_eq!(
        solution.transitions,
        vec![model.forward_transitions[0].clone(); 2]
    );
    assert_eq!(
        validate_solution(&model, &solution, CostTolerance::default()).unwrap(),
        SolutionCost::Integer(2)
    );
}

#[test]
fn loading_retains_declared_cost_without_validating() {
    let model = model();
    let solution = load_solution_from_str(&model, &format!("cost: 999\n{TWO_STEPS}")).unwrap();
    assert_eq!(solution.cost, Some(SolutionCost::Integer(999)));
    assert!(matches!(
        validate_solution(
            &model,
            &solution,
            CostTolerance {
                abs_tol: 1000.0,
                rel_tol: 1000.0
            }
        ),
        Err(SolutionError::CostMismatch {
            declared: SolutionCost::Integer(999),
            computed: SolutionCost::Integer(2)
        })
    ));
    for prefix in ["", "cost: null\n"] {
        let solution = load_solution_from_str(&model, &format!("{prefix}{TWO_STEPS}")).unwrap();
        assert_eq!(solution.cost, None);
        assert_eq!(
            validate_solution(&model, &solution, CostTolerance::default()).unwrap(),
            SolutionCost::Integer(2)
        );
    }
    let solution = load_solution_from_str(&model, "transitions: []").unwrap();
    assert!(matches!(
        validate_solution(&model, &solution, CostTolerance::default()),
        Err(SolutionError::Validation(
            dypdl::SolutionValidationError::NotBaseState { .. }
        ))
    ));
}

#[test]
fn malformed_documents_and_fields_have_paths() {
    let model = model();
    for (source, expected) in [
        ("", "exactly one YAML document"),
        (
            "---\ntransitions: []\n---\ntransitions: []",
            "exactly one YAML document",
        ),
        ("[", "invalid YAML"),
        ("[]", "solution: expected a mapping"),
        ("{}", "transitions: missing required field"),
        ("transitions: null", "transitions: expected a sequence"),
        ("transitions: [visit]", "transitions[0]: expected a mapping"),
        ("transitions: [{}]", "transitions[0].name"),
        ("transitions: [{name: true}]", "transitions[0].name"),
        (
            "transitions: [{name: visit, parameters: []}]",
            "transitions[0].parameters",
        ),
        (
            "transitions: [{name: visit, parameters: {a: -1, z: 1}}]",
            "transitions[0].parameters.a",
        ),
        (
            "transitions: [{name: visit, parameters: {a: 0.5, z: 1}}]",
            "transitions[0].parameters.a",
        ),
        (
            "transitions: [{name: visit, parameters: {0: 0}}]",
            "parameter names must be strings",
        ),
        (
            "transitions: []\ncost: 1.5",
            "cost: expected a 32-bit integer",
        ),
        (
            "transitions: []\ncost: 2147483648",
            "cost: expected a 32-bit integer",
        ),
        (
            "transitions: []\ncost: true",
            "cost: expected a 32-bit integer",
        ),
        ("transitions: []\ncots: 1", "solution.cots: unknown field"),
        (
            "transitions: [{name: visit, parameter: {a: 0}}]",
            "transitions[0].parameter: unknown field",
        ),
        ("transitions: []\ntransitions: []", "duplicated key"),
        (
            "transitions: [{name: visit, parameters: {a: 0, a: 1}}]",
            "duplicated key",
        ),
    ] {
        let error = load_solution_from_str(&model, source).unwrap_err();
        assert!(
            error.to_string().contains(expected),
            "source={source:?}, error={error}"
        );
    }
}

#[test]
fn only_exact_name_and_complete_parameter_maps_are_resolved() {
    let mut model = model();
    let mut backward = model.forward_transitions[0].clone();
    backward.name = "backward".to_owned();
    model.add_backward_transition(backward).unwrap();
    for transition in [
        "{name: missing, parameters: {a: 0, z: 1}}",
        "{name: visit, parameters: {a: 0}}",
        "{name: visit, parameters: {a: 0, z: 1, extra: 0}}",
        "{name: visit, parameters: {a: 0, z: 99}}",
        "{name: visit, parameters: {wrong: 0, z: 1}}",
        "{name: backward, parameters: {a: 0, z: 1}}",
        "{name: 'visit z:1 a:0'}",
    ] {
        assert!(
            load_solution_from_str(&model, &format!("transitions: [{transition}]"))
                .unwrap_err()
                .to_string()
                .contains("no forward transition")
        );
    }
}

#[test]
fn ambiguous_names_are_rejected_including_forced_collisions() {
    let mut model = model();
    model
        .add_forward_forced_transition(model.forward_transitions[0].clone())
        .unwrap();
    assert!(load_solution_from_str(&model, TWO_STEPS)
        .unwrap_err()
        .to_string()
        .contains("ambiguous transition"));
    model.forward_transitions.clear();
    let solution = load_solution_from_str(&model, TWO_STEPS).unwrap();
    assert_eq!(
        validate_solution(&model, &solution, CostTolerance::default()).unwrap(),
        SolutionCost::Integer(2)
    );
}

#[test]
fn raw_names_that_look_like_grounded_names_are_distinct() {
    let mut model = model();
    let mut step = model.forward_transitions[0].clone();
    step.name = "visit z:1 a:0".to_owned();
    step.parameter_names.clear();
    step.parameter_values.clear();
    model.add_forward_transition(step.clone()).unwrap();
    let loaded = load_solution_from_str(&model, "transitions: [{name: 'visit z:1 a:0'}]").unwrap();
    assert_eq!(loaded.transitions, vec![step]);
    assert_eq!(
        load_solution_from_str(&model, TWO_STEPS)
            .unwrap()
            .transitions[0],
        model.forward_transitions[0]
    );
}

#[test]
fn solver_dump_round_trips_including_yaml_like_names() {
    let mut model = model();
    model.forward_transitions[0].name = "true".to_owned();
    model.forward_transitions[0].parameter_names[0] = "false".to_owned();
    let transitions = vec![model.forward_transitions[0].clone(); 2];
    let dump = SolutionToDump::from(Solution {
        cost: Some(2),
        transitions: transitions.clone(),
        ..Default::default()
    })
    .dump_to_str()
    .unwrap();
    let loaded = load_solution_from_str(&model, &dump).unwrap();
    assert_eq!(
        loaded,
        LoadedSolution {
            cost: Some(SolutionCost::Integer(2)),
            transitions
        }
    );
    assert_eq!(
        validate_solution(&model, &loaded, CostTolerance::default()).unwrap(),
        SolutionCost::Integer(2)
    );
    let dump = SolutionToDump::from(loaded.clone()).dump_to_str().unwrap();
    assert_eq!(
        load_solution_from_yaml(
            &model,
            &yaml_rust::YamlLoader::load_from_str(&dump).unwrap()[0]
        )
        .unwrap(),
        loaded
    );
}

#[test]
fn continuous_costs_and_tolerances() {
    let mut model = model();
    model.cost_type = CostType::Continuous;
    model.forward_transitions[0].set_cost(ContinuousExpression::Cost + 0.1);
    let solution =
        load_solution_from_str(&model, &format!("cost: 0.2000000001\n{TWO_STEPS}")).unwrap();
    assert!(matches!(
        validate_solution(&model, &solution, CostTolerance::default()),
        Err(SolutionError::CostMismatch { .. })
    ));
    for tolerance in [
        CostTolerance {
            abs_tol: 1e-9,
            rel_tol: 0.0,
        },
        CostTolerance {
            abs_tol: 0.0,
            rel_tol: 1e-8,
        },
    ] {
        assert_eq!(
            validate_solution(&model, &solution, tolerance).unwrap(),
            SolutionCost::Continuous(0.2)
        );
    }
    for cost in [".nan", ".inf", "-.inf", "1e999", "true", "'0.2'"] {
        assert!(load_solution_from_str(&model, &format!("cost: {cost}\n{TWO_STEPS}")).is_err());
    }
    assert_eq!(
        load_solution_from_str(&model, &format!("cost: 2\n{TWO_STEPS}"))
            .unwrap()
            .cost,
        Some(SolutionCost::Continuous(2.0))
    );
    for (rel_tol, abs_tol) in [
        (-1.0, 0.0),
        (0.0, -1.0),
        (f64::NAN, 0.0),
        (0.0, f64::INFINITY),
    ] {
        assert!(matches!(
            validate_solution(&model, &solution, CostTolerance { rel_tol, abs_tol }),
            Err(SolutionError::InvalidTolerance)
        ));
    }
}

#[test]
fn cost_comparison_is_independent_of_solution_validation() {
    let tolerance = CostTolerance::default();
    assert!(validate_cost(
        SolutionCost::Integer(2),
        SolutionCost::Integer(2),
        tolerance
    )
    .is_ok());
    assert!(matches!(
        validate_cost(
            SolutionCost::Integer(2),
            SolutionCost::Integer(3),
            tolerance
        ),
        Err(SolutionError::CostMismatch {
            computed: SolutionCost::Integer(2),
            declared: SolutionCost::Integer(3),
        })
    ));
    for (computed, declared, expected) in [
        (
            SolutionCost::Integer(2),
            SolutionCost::Continuous(2.0),
            CostType::Integer,
        ),
        (
            SolutionCost::Continuous(2.0),
            SolutionCost::Integer(2),
            CostType::Continuous,
        ),
    ] {
        assert!(matches!(
            validate_cost(computed, declared, tolerance),
            Err(SolutionError::CostTypeMismatch { expected: actual }) if actual == expected
        ));
    }
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for (computed, declared) in [(value, 0.0), (0.0, value), (value, value)] {
            assert!(matches!(
                validate_cost(
                    SolutionCost::Continuous(computed),
                    SolutionCost::Continuous(declared),
                    tolerance,
                ),
                Err(SolutionError::CostMismatch { .. })
            ));
        }
    }
}
