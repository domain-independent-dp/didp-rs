import didppy as dp
import pytest


@pytest.fixture
def model_and_step():
    model = dp.Model()
    x = model.add_int_var(target=2, name="x")
    model.add_base_case([x == 0])
    model.add_state_constr(x >= 0)
    step = dp.Transition(
        name="step",
        cost=x + dp.IntExpr.state_cost(),
        preconditions=[x > 0],
        effects=[(x, x - 1)],
    )
    model.add_transition(step)
    return model, step, x


def test_sequence_and_solver_solution(model_and_step, capfd):
    model, step, _ = model_and_step
    assert model.validate_solution([step, step]) == 3
    assert type(model.validate_solution((step, step))) is int
    solution = dp.CABS(model, quiet=True).search()
    assert isinstance(solution, dp.Solution)
    cost = model.validate_solution(transitions=solution.transitions)
    assert cost == solution.cost == 3
    assert model.validate_cost(cost, solution.cost) is None
    with pytest.raises(TypeError):
        model.validate_solution(solution)
    with pytest.raises(TypeError):
        model.validate_solution([step, step], rel_tol=1e-9)
    captured = capfd.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_transition_id_sequences_use_registered_transitions(model_and_step, capfd):
    model, step, x = model_and_step
    step.cost = dp.IntExpr.state_cost() + (dp.IntExpr(1) + 2)
    step_id = model.add_transition(step)
    assert isinstance(step_id, dp.TransitionId)
    # Modifying the original object does not change the registered transition.
    step.cost = 999
    assert model.validate_solution([step_id, step_id]) == 6
    cost = model.validate_solution(transitions=(step_id, step_id))
    assert type(cost) is int
    assert model.validate_cost(cost, 6) is None
    forced = dp.Transition(
        name="forced", cost=dp.IntExpr.state_cost() + 1, effects=[(x, x - 1)]
    )
    forced_id = model.add_transition(forced, forced=True)
    assert model.validate_solution([step_id, forced_id]) == 4
    captured = capfd.readouterr()
    assert captured.out == captured.err == ""


def test_mixed_transition_and_id_sequences_are_rejected(model_and_step):
    model, step, _ = model_and_step
    step_id = model.add_transition(step)
    for candidate in ([step, step_id], [step_id, step], [step_id, 0], (0, step_id)):
        with pytest.raises(TypeError):
            model.validate_solution(candidate)


@pytest.mark.parametrize("forced", [False, True])
def test_backward_transition_ids_are_rejected(model_and_step, forced):
    model, step, _ = model_and_step
    step_id = model.add_transition(step)
    backward_id = model.add_transition(step, forced=forced, backward=True)
    with pytest.raises(
        dp.ValidationError, match="Transition 2 has an invalid ID.*backward IDs"
    ):
        model.validate_solution([step_id, backward_id])


@pytest.mark.parametrize("forced", [False, True])
def test_out_of_range_transition_ids_have_context(model_and_step, forced):
    model, step, _ = model_and_step
    step_id = model.add_transition(step)
    other = dp.Model()
    # Obtain an index that does not exist in the supplied model's bucket.
    for _ in range(3):
        missing_id = other.add_transition(dp.Transition(name="other"), forced=forced)
    with pytest.raises(
        dp.ValidationError,
        match="Transition 2 has an invalid ID.*index 2.*no transition",
    ):
        model.validate_solution([step_id, missing_id])


def test_transition_ids_are_interpreted_relative_to_the_model(model_and_step):
    model, _, _ = model_and_step
    other = dp.Model()
    foreign_id = other.add_transition(dp.Transition(name="other", cost=999))
    assert model.validate_solution([foreign_id, foreign_id]) == 3


def test_transition_ids_keep_feasibility_checks(model_and_step):
    model, step, x = model_and_step
    step_id = model.add_transition(step)
    with pytest.raises(dp.ValidationError, match="final state.*state 1"):
        model.validate_solution([step_id])
    with pytest.raises(dp.ValidationError, match="State 2.*1 transitions remain"):
        model.validate_solution([step_id] * 3)
    blocked_id = model.add_transition(
        dp.Transition(name="blocked", preconditions=[x < 0], effects=[(x, 0)])
    )
    with pytest.raises(
        dp.ValidationError, match=r"Transition 1 \(`blocked`\).*precondition 1"
    ):
        model.validate_solution([blocked_id])
    invalid_id = model.add_transition(dp.Transition(name="invalid", effects=[(x, -1)]))
    with pytest.raises(dp.ValidationError, match="State 1 violates state constraint 1"):
        model.validate_solution([invalid_id])
    model.set_target(x, -1)
    with pytest.raises(dp.ValidationError, match="State 0 violates state constraint 1"):
        model.validate_solution([step_id])
    model.set_target(x, 0)
    assert model.validate_solution(()) == 0


@pytest.mark.parametrize("use_ids", [False, True])
@pytest.mark.parametrize("target", [2, -1, 0])
def test_all_references_are_checked_before_feasibility(model_and_step, use_ids, target):
    model, _, x = model_and_step
    model.set_target(x, target)
    blocked = dp.Transition(name="blocked", preconditions=[x < 0], effects=[(x, 0)])
    blocked_id = model.add_transition(blocked)
    if use_ids:
        other = dp.Model()
        for _ in range(3):
            missing_id = other.add_transition(dp.Transition(name="other"))
        candidate = [blocked_id, missing_id]
        message = "Transition 2 has an invalid ID"
    else:
        candidate = [blocked, dp.Transition(name="unknown")]
        message = r"Transition 2 \(`unknown`\) is not a registered transition"
    # At target 2 the first precondition fails, at -1 the target constraint
    # fails, and at 0 the target is already a base state. Input errors win.
    with pytest.raises(dp.ValidationError, match=message):
        model.validate_solution(candidate)
    assert model.target_state[x] == target


def test_load_from_string_and_file(model_and_step, tmp_path):
    model, step, _ = model_and_step
    text = "cost: 3\ntransitions: [{name: step}, {name: step}]"
    path = tmp_path / "solution.yaml"
    path.write_text(text)
    for loaded in (
        model.load_solution_from_str(text),
        model.load_solution_from_file(path),
        model.load_solution_from_file(str(path)),
    ):
        assert type(loaded) is tuple
        assert len(loaded) == 2
        transitions, declared_cost = loaded
        assert type(transitions) is list
        assert all(isinstance(t, dp.Transition) for t in transitions)
        assert type(declared_cost) is int
        assert declared_cost == 3
        assert [t.name for t in transitions] == [step.name, step.name]
        cost = model.validate_solution(transitions)
        assert cost == 3
        assert model.validate_cost(cost, declared_cost) is None
    assert path.read_text() == text


@pytest.mark.parametrize("prefix", ["", "cost: null\n"])
def test_loading_without_declared_cost(model_and_step, prefix):
    model, _, _ = model_and_step
    transitions, declared_cost = model.load_solution_from_str(
        prefix + "transitions: [{name: step}, {name: step}]"
    )
    assert declared_cost is None
    assert model.validate_solution(transitions) == 3
    assert declared_cost is None


@pytest.mark.parametrize("float_cost", [False, True])
@pytest.mark.parametrize(
    "prefix, expected",
    [("", None), ("cost: null\n", None), ("cost: 0\n", 0), ("cost: 2\n", 2)],
)
def test_loaded_pair_shape_and_cost_type(float_cost, prefix, expected, tmp_path):
    model = dp.Model(float_cost=float_cost)
    text = prefix + "transitions: []"
    path = tmp_path / "solution.yaml"
    path.write_text(text)
    for loaded in (
        model.load_solution_from_str(text),
        model.load_solution_from_file(path),
    ):
        assert type(loaded) is tuple
        assert len(loaded) == 2
        transitions, declared_cost = loaded
        assert type(transitions) is list
        assert transitions == []
        if expected is None:
            assert declared_cost is None
        else:
            assert type(declared_cost) is (float if float_cost else int)
            assert declared_cost == expected


def test_loading_does_not_check_feasibility_or_evaluate_cost(model_and_step):
    model, _, x = model_and_step
    transitions, declared_cost = model.load_solution_from_str(
        "cost: 999\ntransitions: [{name: step}]"
    )
    assert len(transitions) == 1
    assert declared_cost == 999
    with pytest.raises(dp.ValidationError, match="final state.*state 1"):
        model.validate_solution(transitions)
    assert model.target_state[x] == 2


def test_wrong_declared_cost_is_a_validation_error(model_and_step):
    model, _, _ = model_and_step
    transitions, declared_cost = model.load_solution_from_str(
        "cost: 999\ntransitions: [{name: step}, {name: step}]"
    )
    assert declared_cost == 999
    cost = model.validate_solution(transitions)
    assert cost == 3
    assert issubclass(dp.ValidationError, ValueError)
    with pytest.raises(
        dp.ValidationError, match="declared cost 999 does not match the computed cost 3"
    ):
        model.validate_cost(cost, declared_cost, rel_tol=1000, abs_tol=1000)
    assert declared_cost == 999


def test_infeasible_paths_have_context(model_and_step):
    model, step, x = model_and_step
    with pytest.raises(dp.ValidationError, match="final state.*state 1"):
        model.validate_solution([step])
    with pytest.raises(dp.ValidationError, match="State 2.*1 transitions remain"):
        model.validate_solution([step, step, step])
    blocked = dp.Transition(name="blocked", preconditions=[x < 0], effects=[(x, 0)])
    model.add_transition(blocked)
    with pytest.raises(
        dp.ValidationError, match=r"Transition 1 \(`blocked`\).*state 0.*precondition 1"
    ):
        model.validate_solution([blocked])
    invalid = dp.Transition(name="invalid", effects=[(x, -1)])
    model.add_transition(invalid)
    with pytest.raises(dp.ValidationError, match="State 1 violates state constraint 1"):
        model.validate_solution([invalid])


def test_target_constraints_and_empty_base_solution(model_and_step):
    model, _, x = model_and_step
    with pytest.raises(dp.ValidationError, match="final state.*state 0"):
        model.validate_solution([])
    model.set_target(x, -1)
    with pytest.raises(dp.ValidationError, match="State 0 violates state constraint 1"):
        model.validate_solution([])
    model.set_target(x, 0)
    assert model.validate_solution([]) == 0
    transitions, declared_cost = model.load_solution_from_str("transitions: []")
    assert transitions == []
    assert declared_cost is None
    assert model.validate_solution(transitions) == 0


def test_original_transition_simplification_and_mutation(model_and_step, capfd):
    model, step, _ = model_and_step
    step.cost = dp.IntExpr.state_cost() + (dp.IntExpr(1) + 2)
    model.add_transition(step)
    capfd.readouterr()
    assert model.validate_solution([step, step]) == 6
    assert capfd.readouterr().err == ""
    step.cost = 999
    with pytest.raises(
        dp.ValidationError, match="not a registered transition in this model"
    ):
        model.validate_solution([step, step])


def test_forced_transition_loading_and_priority(model_and_step):
    model, step, x = model_and_step
    forced = dp.Transition(
        name="forced",
        preconditions=[x > 0],
        effects=[(x, x - 1)],
        cost=x + dp.IntExpr.state_cost(),
    )
    model.add_transition(forced, forced=True)
    assert model.validate_solution([step, forced]) == 3
    transitions, declared_cost = model.load_solution_from_str(
        "transitions: [{name: step}, {name: forced}]"
    )
    assert declared_cost is None
    assert model.validate_solution(transitions) == 3


def test_backward_and_ambiguous_references(model_and_step):
    model, _, x = model_and_step
    backward = dp.Transition(name="backward", effects=[(x, 0)])
    model.add_transition(backward, backward=True)
    with pytest.raises(ValueError, match="no forward transition"):
        model.load_solution_from_str("transitions: [{name: backward}]")
    with pytest.raises(
        dp.ValidationError, match="not a registered transition in this model"
    ):
        model.validate_solution([backward])
    model.add_transition(dp.Transition(name="step", effects=[(x, 0)]), forced=True)
    with pytest.raises(ValueError, match="ambiguous transition"):
        model.load_solution_from_str("transitions: [{name: step}]")


def test_float_cost_comparison_and_return_type():
    model = dp.Model(float_cost=True)
    x = model.add_int_var(target=3)
    model.add_base_case([x == 0])
    step = dp.Transition(
        name="step", cost=0.1 + dp.FloatExpr.state_cost(), effects=[(x, x - 1)]
    )
    step_id = model.add_transition(step)
    text = "cost: 0.3\ntransitions: [{name: step}, {name: step}, {name: step}]"
    transitions, declared_cost = model.load_solution_from_str(text)
    assert type(declared_cost) is float
    assert declared_cost == 0.3
    cost = model.validate_solution(transitions)
    assert type(cost) is float
    assert cost == 0.1 + 0.1 + 0.1
    assert type(model.validate_solution([step_id] * 3)) is float
    assert model.validate_solution([step_id] * 3) == cost
    with pytest.raises(dp.ValidationError, match="does not match the computed cost"):
        model.validate_cost(cost, declared_cost)
    for kwargs in ({"rel_tol": 1e-9}, {"abs_tol": 1e-9}):
        assert model.validate_cost(cost, declared_cost, **kwargs) is None
    assert model.validate_solution([step] * 3) == 0.1 + 0.1 + 0.1
    _, integer_declared_cost = model.load_solution_from_str(
        text.replace("cost: 0.3", "cost: 1")
    )
    assert type(integer_declared_cost) is float
    assert integer_declared_cost == 1.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rel_tol": -1},
        {"abs_tol": -1},
        {"rel_tol": float("nan")},
        {"abs_tol": float("inf")},
    ],
)
def test_invalid_tolerances(model_and_step, kwargs):
    model, _, _ = model_and_step
    with pytest.raises(ValueError, match="finite, non-negative"):
        model.validate_cost(3, 3, **kwargs)


def test_cost_comparison_does_not_validate_feasibility(model_and_step):
    model, _, _ = model_and_step
    assert model.validate_cost(computed_cost=3, declared_cost=3) is None
    assert model.validate_cost(-2, -2) is None
    continuous_model = dp.Model(float_cost=True)
    assert continuous_model.validate_cost(3, 3.0) is None
    assert continuous_model.validate_cost(0.0, -0.0) is None


@pytest.mark.parametrize(
    "computed, declared", [(3.0, 3), (3, 3.0), (None, 3), (3, None), ("3", 3), (3, "3")]
)
def test_cost_comparison_type_errors(model_and_step, computed, declared):
    model, _, _ = model_and_step
    with pytest.raises(TypeError):
        model.validate_cost(computed, declared)


@pytest.mark.parametrize("computed, declared", [(2147483648, 0), (0, -2147483649)])
def test_integer_cost_comparison_overflow(model_and_step, computed, declared):
    model, _, _ = model_and_step
    with pytest.raises(OverflowError):
        model.validate_cost(computed, declared)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_cost_comparison_rejects_nonfinite_values(value):
    model = dp.Model(float_cost=True)
    for computed, declared in [(value, 0.0), (0.0, value), (value, value)]:
        with pytest.raises(dp.ValidationError):
            model.validate_cost(computed, declared, rel_tol=1, abs_tol=1)


@pytest.mark.parametrize("value", [None, 3, "step", [3]])
def test_invalid_input_type(model_and_step, value):
    model, _, _ = model_and_step
    with pytest.raises(TypeError):
        model.validate_solution(value)


@pytest.mark.parametrize(
    "text, message",
    [
        ("", "exactly one YAML document"),
        ("---\ntransitions: []\n---\ntransitions: []", "exactly one YAML document"),
        ("[", "invalid YAML"),
        ("{}", "transitions: missing required field"),
        ("transitions: [step]", r"transitions\[0\]"),
        ("transitions: [{name: missing}]", "no forward transition"),
        (
            "transitions: [{name: step, parameters: {i: -1}}]",
            "non-negative element index",
        ),
        ("transitions: []\ncost: 1.5", "expected a 32-bit integer"),
        ("transitions: []\ncost: 2147483648", "expected a 32-bit integer"),
    ],
)
def test_loading_errors(model_and_step, text, message):
    model, _, _ = model_and_step
    with pytest.raises(ValueError, match=message):
        model.load_solution_from_str(text)


def test_missing_file_is_file_not_found(model_and_step, tmp_path):
    model, _, _ = model_and_step
    with pytest.raises(FileNotFoundError):
        model.load_solution_from_file(tmp_path / "missing.yaml")


def test_evaluation_error_can_be_caught(model_and_step):
    model, step, x = model_and_step
    step.cost = dp.IntExpr.state_cost() // (x - 1)
    step_id = model.add_transition(step)
    for candidate in ([step, step], [step_id, step_id]):
        with pytest.raises(
            dp.ValidationError, match=r"cost of transition 2 \(`step`\)"
        ):
            model.validate_solution(candidate)
    # An evaluation failure does not mutate the model or its target state.
    assert model.target_state[x] == 2


def test_yaml_model_grounded_parameters():
    domain = """
objects: [item]
state_variables:
  - {name: remaining, type: set, object: item}
base_cases:
  - ['(is_empty remaining)']
transitions:
  - name: visit
    parameters:
      - {name: i, object: remaining}
    effect: {remaining: '(remove i remaining)'}
    cost: '(+ cost 1)'
"""
    model = dp.Model.load_from_str(
        domain, "object_numbers: {item: 2}\ntarget: {remaining: [0, 1]}"
    )
    transitions, declared_cost = model.load_solution_from_str(
        "cost: 2\ntransitions: [{name: visit, parameters: {i: 1}}, "
        "{name: visit, parameters: {i: 0}}]"
    )
    assert [t.name for t in transitions] == ["visit i:1", "visit i:0"]
    assert declared_cost == 2
    assert model.validate_solution(transitions) == 2
