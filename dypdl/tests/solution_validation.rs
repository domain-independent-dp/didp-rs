use dypdl::prelude::*;
use dypdl::variable_type::{Numeric, OrderedContinuous};
use dypdl::SolutionValidationError;

fn model_with_step() -> (Model, Transition) {
    let mut model = Model::default();
    let x = model.add_integer_variable("x", 2).unwrap();
    model
        .add_base_case(vec![Condition::comparison_i(ComparisonOperator::Eq, x, 0)])
        .unwrap();
    model
        .add_state_constraint(Condition::comparison_i(ComparisonOperator::Ge, x, 0))
        .unwrap();
    let mut step = Transition::new("step");
    step.add_precondition(Condition::comparison_i(ComparisonOperator::Gt, x, 0));
    step.add_effect(x, x - 1).unwrap();
    step.set_cost(IntegerExpression::Cost + x);
    model.add_forward_transition(step.clone()).unwrap();
    (model, step)
}

#[test]
fn costs_are_evaluated_backwards_in_the_original_states() {
    let (mut model, mut step) = model_with_step();
    let x = model.get_integer_variable("x").unwrap();
    model.base_cases[0].cost = Some(3.into());
    step.set_cost(IntegerExpression::Cost * x + 1);
    model.forward_transitions.clear();
    model.add_forward_transition(step.clone()).unwrap();
    // Last transition: 3 * 1 + 1 = 4; first: 4 * 2 + 1 = 9.
    assert_eq!(
        model
            .validate_solution::<Integer>(&[step.clone(), step])
            .unwrap(),
        9
    );
}

#[test]
fn original_unsimplified_transitions_are_accepted_but_modified_ones_are_not() {
    let (mut model, mut step) = model_with_step();
    step.set_cost(IntegerExpression::Cost + (IntegerExpression::Constant(1) + 2));
    model.forward_transitions.clear();
    model.add_forward_transition(step.clone()).unwrap();
    assert_ne!(step, model.forward_transitions[0]);
    assert_eq!(
        model
            .validate_solution::<Integer>(&[step.clone(), step.clone()])
            .unwrap(),
        6
    );
    step.set_cost(0);
    assert!(matches!(
        model.validate_solution::<Integer>(&[step]),
        Err(SolutionValidationError::UnknownTransition {
            transition_index: 0,
            ..
        })
    ));
}

#[test]
fn all_memberships_are_checked_before_feasibility() {
    let (mut model, step) = model_with_step();
    let mut blocked = step.clone();
    blocked.name = "blocked".to_owned();
    blocked.add_precondition(Condition::Constant(false));
    model.add_forward_transition(blocked.clone()).unwrap();
    let unknown = Transition::new("unknown");
    for transitions in [
        vec![blocked, unknown.clone()],
        vec![step.clone(), step, unknown.clone()],
    ] {
        // The unknown reference is reported before an earlier inapplicable
        // transition or an early base state.
        assert_eq!(
            model.validate_solution::<Integer>(&transitions),
            Err(SolutionValidationError::UnknownTransition {
                transition_index: transitions.len() - 1,
                transition_name: "unknown".to_owned(),
            })
        );
    }
    for target in [-1, 0] {
        model.target.signature_variables.integer_variables[0] = target;
        assert_eq!(
            model.validate_solution::<Integer>(&[unknown.clone(), Transition::new("later")]),
            Err(SolutionValidationError::UnknownTransition {
                transition_index: 0,
                transition_name: "unknown".to_owned(),
            })
        );
    }
}

#[test]
fn membership_preparation_retains_expression_error_context() {
    let (mut model, mut step) = model_with_step();
    let x = model.get_integer_variable("x").unwrap();
    // The target is already a base state. Invalid expressions must still be
    // diagnosed during input preparation, before the early-base-state check.
    model.target.signature_variables.integer_variables[0] = 0;
    let mut other = Model::default();
    other.add_integer_variable("first", 0).unwrap();
    let foreign = other.add_integer_variable("second", 0).unwrap();
    step.set_cost(IntegerExpression::Cost + foreign);
    assert!(matches!(
        model.validate_solution::<Integer>(&[step.clone()]),
        Err(SolutionValidationError::InvalidTransition {
            transition_index: 0,
            ..
        })
    ));

    // Simplification can panic even before replay. It must remain catchable.
    step.set_cost(IntegerExpression::Cost + (IntegerExpression::Constant(1) / 0));
    let error = model.validate_solution::<Integer>(&[step]).unwrap_err();
    assert!(matches!(
        error,
        SolutionValidationError::EvaluationError { state_index: 0, .. }
    ));
    assert!(error.to_string().contains("transition 1 (`step`)"));
    assert_eq!(model.target.signature_variables.integer_variables[0], 0);
    // Avoid changing the model's registered transition while preparing inputs.
    assert_eq!(
        model.forward_transitions[0].cost,
        (IntegerExpression::Cost + x).into()
    );
}

#[test]
fn final_and_early_base_states_and_empty_solutions() {
    let (mut model, step) = model_with_step();
    assert_eq!(
        model.validate_solution::<Integer>(&[]),
        Err(SolutionValidationError::NotBaseState { state_index: 0 })
    );
    assert_eq!(
        model.validate_solution::<Integer>(std::slice::from_ref(&step)),
        Err(SolutionValidationError::NotBaseState { state_index: 1 })
    );
    assert_eq!(
        model.validate_solution::<Integer>(&[step.clone(), step.clone(), step]),
        Err(SolutionValidationError::EarlyBaseState {
            state_index: 2,
            remaining_transitions: 1
        })
    );
    model.target.signature_variables.integer_variables[0] = 0;
    model.base_cases[0].cost = Some(7.into());
    assert_eq!(model.validate_solution::<Integer>(&[]).unwrap(), 7);
}

#[test]
fn target_and_successor_constraints_are_checked() {
    let (mut model, mut step) = model_with_step();
    model.target.signature_variables.integer_variables[0] = -1;
    assert_eq!(
        model.validate_solution::<Integer>(&[]),
        Err(SolutionValidationError::StateConstraintViolation {
            state_index: 0,
            constraint_index: 0
        })
    );
    model.target.signature_variables.integer_variables[0] = 2;
    let x = model.get_integer_variable("x").unwrap();
    step.effect.integer_effects.clear();
    step.add_effect(x, -1).unwrap();
    model.add_forward_transition(step.clone()).unwrap();
    assert_eq!(
        model.validate_solution::<Integer>(&[step]),
        Err(SolutionValidationError::StateConstraintViolation {
            state_index: 1,
            constraint_index: 0
        })
    );
}

#[test]
fn precondition_and_grounded_membership_failures_have_context() {
    let (mut model, mut step) = model_with_step();
    step.add_precondition(Condition::Constant(false));
    model.add_forward_transition(step.clone()).unwrap();
    let error = model.validate_solution::<Integer>(&[step]).unwrap_err();
    assert!(matches!(
        error,
        SolutionValidationError::TransitionNotApplicable {
            transition_index: 0,
            ..
        }
    ));
    assert!(error
        .to_string()
        .contains("Transition 1 (`step`) is not applicable in state 0"));

    let object = model.add_object_type("item", 2).unwrap();
    let set = model.create_set(object, &[0]).unwrap();
    model.add_set_variable("remaining", object, set).unwrap();
    let mut step = model.forward_transitions[0].clone();
    step.elements_in_set_variable.push((0, 1));
    model.add_forward_transition(step.clone()).unwrap();
    let error = model.validate_solution::<Integer>(&[step]).unwrap_err();
    assert!(error
        .to_string()
        .contains("parameter value 1 is not in set variable `remaining`"));
}

#[test]
fn forced_priority_is_not_a_feasibility_requirement_and_backward_is_rejected() {
    let (mut model, step) = model_with_step();
    let mut forced = step.clone();
    forced.name = "forced".to_owned();
    model.add_forward_forced_transition(forced.clone()).unwrap();
    assert_eq!(
        model
            .validate_solution::<Integer>(&[step.clone(), forced])
            .unwrap(),
        3
    );
    let mut backward = step.clone();
    backward.name = "backward".to_owned();
    model.add_backward_transition(backward.clone()).unwrap();
    assert!(matches!(
        model.validate_solution::<Integer>(&[backward]),
        Err(SolutionValidationError::UnknownTransition { .. })
    ));
}

#[test]
fn continuous_and_multiple_base_costs() {
    let (mut model, mut step) = model_with_step();
    model.cost_type = CostType::Continuous;
    let case = model.base_cases[0].clone();
    model.base_cases[0].cost = Some(2.5.into());
    model.base_cases.push(dypdl::BaseCase {
        cost: Some(1.5.into()),
        ..case
    });
    step.set_cost(ContinuousExpression::Cost + 0.25);
    model.forward_transitions.clear();
    model.add_forward_transition(step.clone()).unwrap();
    assert_eq!(
        model
            .validate_solution::<OrderedContinuous>(&[step.clone(), step.clone()])
            .unwrap()
            .into_inner(),
        2.0
    );
    model.set_maximize();
    assert_eq!(
        model
            .validate_solution::<OrderedContinuous>(&[step.clone(), step])
            .unwrap()
            .into_inner(),
        3.0
    );
}

#[test]
fn explicit_base_state_cost_is_used() {
    let (mut model, step) = model_with_step();
    model.base_cases.clear();
    let mut base = model.target.clone();
    base.signature_variables.integer_variables[0] = 0;
    model.base_states.push((base, Some(5.into())));
    assert_eq!(
        model
            .validate_solution::<Integer>(&[step.clone(), step])
            .unwrap(),
        8
    );
}

#[test]
fn evaluation_failure_is_an_error_with_transition_context() {
    let (mut model, mut step) = model_with_step();
    let x = model.get_integer_variable("x").unwrap();
    step.set_cost(IntegerExpression::Cost / (x - 1));
    model.forward_transitions.clear();
    model.add_forward_transition(step.clone()).unwrap();
    let error = model
        .validate_solution::<Integer>(&[step.clone(), step])
        .unwrap_err();
    assert!(matches!(
        error,
        SolutionValidationError::EvaluationError { state_index: 1, .. }
    ));
    assert!(error.to_string().contains("cost of transition 2 (`step`)"));
}

#[test]
fn legacy_validation_still_accepts_unregistered_transitions_and_wrong_cost() {
    let (mut model, step) = model_with_step();
    model.forward_transitions.clear();
    assert!(model.validate_forward(&[step.clone(), step], 999, false));
}

#[test]
fn legacy_validation_keeps_target_constraint_and_panic_behavior() {
    let (mut model, mut step) = model_with_step();
    let x = model.get_integer_variable("x").unwrap();
    model.target.signature_variables.integer_variables[0] = -1;
    step.preconditions.clear();
    step.effect.integer_effects.clear();
    step.add_effect(x, 0).unwrap();
    step.set_cost(1);
    // Neither membership nor target constraints are checked by the legacy API.
    assert!(model.validate_forward(&[step], 999, false));

    let (mut model, mut step) = model_with_step();
    let x = model.get_integer_variable("x").unwrap();
    step.set_cost(IntegerExpression::Cost / (x - 1));
    model.forward_transitions.clear();
    let transitions = [step.clone(), step];
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        model.validate_forward(&transitions, 0, false)
    }))
    .is_err());
}

#[test]
fn wrong_cost_type_and_non_finite_objectives_are_rejected() {
    let (mut model, step) = model_with_step();
    assert!(matches!(
        model.validate_solution::<OrderedContinuous>(&[]),
        Err(SolutionValidationError::CostTypeMismatch { .. })
    ));
    model.cost_type = CostType::Continuous;
    assert!(matches!(
        model.validate_solution::<Integer>(&[]),
        Err(SolutionValidationError::CostTypeMismatch { .. })
    ));
    model.base_cases[0].cost = Some(f64::INFINITY.into());
    assert!(matches!(
        model.validate_solution::<OrderedContinuous>(&[step.clone(), step]),
        Err(SolutionValidationError::EvaluationError { state_index: 2, .. })
    ));
}

fn assert_id_result<T: Numeric + Ord + 'static>(
    model: &Model,
    ids: &[TransitionId],
    expected: Result<T, SolutionValidationError>,
) {
    let transitions = ids
        .iter()
        .map(|id| model.get_transition(id).unwrap().clone())
        .collect::<Vec<_>>();
    assert_eq!(model.validate_solution::<T>(&transitions), expected);
    assert_eq!(model.validate_solution_with_ids::<T>(ids), expected);
}

#[test]
fn ids_use_registered_transitions_and_fold_nonadditive_costs() {
    let (mut model, mut step) = model_with_step();
    let x = model.get_integer_variable("x").unwrap();
    model.base_cases[0].cost = Some(3.into());
    step.set_cost(IntegerExpression::Cost * x + 1);
    let regular = model.add_forward_transition(step.clone()).unwrap();
    let forced = model.add_forward_forced_transition(step).unwrap();
    assert_id_result(&model, &[regular, forced], Ok(9));
}

#[test]
fn invalid_ids_report_their_position_and_flags() {
    let (mut model, step) = model_with_step();
    let regular = model.add_forward_transition(step.clone()).unwrap();
    let backward = model.add_backward_transition(step.clone()).unwrap();
    let backward_forced = model.add_backward_forced_transition(step).unwrap();
    for id in [
        backward,
        backward_forced,
        TransitionId {
            id: 2,
            forced: false,
            backward: false,
        },
        // Index 0 exists in the regular bucket but not the forced bucket.
        TransitionId {
            id: 0,
            forced: true,
            backward: false,
        },
    ] {
        let error = model
            .validate_solution_with_ids::<Integer>(&[regular.clone(), id.clone()])
            .unwrap_err();
        assert!(matches!(
            &error,
            SolutionValidationError::InvalidTransitionId {
                transition_index: 1,
                transition_id,
                ..
            } if transition_id == &id
        ));
        assert!(error.to_string().contains("Transition 2 has an invalid ID"));
        assert!(error.to_string().contains(&format!("index {}", id.id)));
        let reason = if id.backward {
            "backward IDs"
        } else {
            "no transition with this ID"
        };
        assert!(error.to_string().contains(reason));
    }
}

#[test]
fn ids_keep_target_constraints_and_base_state_checks() {
    let (mut model, step) = model_with_step();
    let id = model.add_forward_transition(step).unwrap();
    assert_id_result::<Integer>(
        &model,
        &[],
        Err(SolutionValidationError::NotBaseState { state_index: 0 }),
    );
    assert_id_result::<Integer>(
        &model,
        std::slice::from_ref(&id),
        Err(SolutionValidationError::NotBaseState { state_index: 1 }),
    );
    assert_id_result::<Integer>(
        &model,
        &[id.clone(), id.clone(), id.clone()],
        Err(SolutionValidationError::EarlyBaseState {
            state_index: 2,
            remaining_transitions: 1,
        }),
    );
    model.target.signature_variables.integer_variables[0] = -1;
    for ids in [vec![], vec![id]] {
        assert_id_result::<Integer>(
            &model,
            &ids,
            Err(SolutionValidationError::StateConstraintViolation {
                state_index: 0,
                constraint_index: 0,
            }),
        );
    }
    model.target.signature_variables.integer_variables[0] = 0;
    model.base_cases[0].cost = Some(7.into());
    assert_id_result(&model, &[], Ok(7));
}

#[test]
fn ids_keep_successor_constraints_preconditions_and_parameter_membership() {
    let (mut model, mut step) = model_with_step();
    let x = model.get_integer_variable("x").unwrap();
    step.effect.integer_effects.clear();
    step.add_effect(x, -1).unwrap();
    let id = model.add_forward_transition(step.clone()).unwrap();
    assert_id_result::<Integer>(
        &model,
        &[id],
        Err(SolutionValidationError::StateConstraintViolation {
            state_index: 1,
            constraint_index: 0,
        }),
    );
    step.add_precondition(Condition::Constant(false));
    let id = model.add_forward_transition(step.clone()).unwrap();
    let expected = model.validate_solution::<Integer>(&[step]);
    assert!(matches!(
        expected,
        Err(SolutionValidationError::TransitionNotApplicable { .. })
    ));
    assert_id_result(&model, &[id], expected);

    let object = model.add_object_type("item", 2).unwrap();
    let set = model.create_set(object, &[0]).unwrap();
    model.add_set_variable("remaining", object, set).unwrap();
    let mut step = model.forward_transitions[0].clone();
    step.elements_in_set_variable.push((0, 1));
    let id = model.add_forward_transition(step.clone()).unwrap();
    let expected = model.validate_solution::<Integer>(&[step]);
    assert!(expected
        .as_ref()
        .unwrap_err()
        .to_string()
        .contains("parameter value 1"));
    assert_id_result(&model, &[id], expected);
}

#[test]
fn ids_keep_evaluation_safeguards_and_cost_types() {
    let (mut model, mut step) = model_with_step();
    let x = model.get_integer_variable("x").unwrap();
    step.set_cost(IntegerExpression::Cost / (x - 1));
    let id = model.add_forward_transition(step.clone()).unwrap();
    let expected = model.validate_solution::<Integer>(&[step.clone(), step]);
    assert!(matches!(
        expected,
        Err(SolutionValidationError::EvaluationError { state_index: 1, .. })
    ));
    assert_id_result(&model, &[id.clone(), id], expected);
    assert_id_result::<OrderedContinuous>(
        &model,
        &[],
        Err(SolutionValidationError::CostTypeMismatch {
            expected: CostType::Integer,
        }),
    );

    model.cost_type = CostType::Continuous;
    let mut step = model.forward_transitions[0].clone();
    step.set_cost(ContinuousExpression::Cost + 0.25);
    let id = model.add_forward_transition(step).unwrap();
    assert_id_result(
        &model,
        &[id.clone(), id.clone()],
        Ok(OrderedContinuous::from(0.5)),
    );
    assert_id_result::<Integer>(
        &model,
        &[],
        Err(SolutionValidationError::CostTypeMismatch {
            expected: CostType::Continuous,
        }),
    );
    for value in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        model.base_cases[0].cost = Some(value.into());
        let step = model.get_transition(&id).unwrap().clone();
        let expected = model.validate_solution::<OrderedContinuous>(&[step.clone(), step]);
        assert!(matches!(
            expected,
            Err(SolutionValidationError::EvaluationError { state_index: 2, .. })
        ));
        assert_id_result(&model, &[id.clone(), id.clone()], expected);
    }
}

#[test]
fn ids_are_relative_to_the_supplied_model() {
    let (model, _) = model_with_step();
    let mut other = Model::default();
    let foreign_id = other
        .add_forward_transition(Transition::new("other"))
        .unwrap();
    assert_id_result(&model, &[foreign_id.clone(), foreign_id], Ok(3));
}
