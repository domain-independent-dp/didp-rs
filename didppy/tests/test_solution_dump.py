import didppy as dp
import pytest


@pytest.fixture
def model_and_step():
    model = dp.Model()
    x = model.add_int_var(target=2, name="x")
    model.add_base_case([x == 0])
    step = dp.Transition(
        name="step",
        preconditions=[x > 0],
        effects=[(x, x - 1)],
        cost=1 + dp.IntExpr.state_cost(),
    )
    step_id = model.add_transition(step)
    return model, step, step_id, x


@pytest.mark.parametrize("use_ids", [False, True])
@pytest.mark.parametrize("cost", [None, 0, 2, 99])
def test_string_and_file_round_trip(model_and_step, use_ids, cost, tmp_path):
    model, step, step_id, x = model_and_step
    candidate = [step_id, step_id] if use_ids else [step, step]
    text = model.dump_solution_to_str(candidate, cost=cost)
    assert isinstance(text, str)
    assert model.dump_solution_to_str(tuple(candidate), cost) == text
    transitions, declared_cost = model.load_solution_from_str(text)
    assert [t.name for t in transitions] == ["step", "step"]
    assert declared_cost == cost
    assert model.validate_solution(transitions) == 2
    for path in (
        tmp_path / "solution.yaml",
        str(tmp_path / "solution-string-path.yaml"),
    ):
        assert model.dump_solution_to_file(candidate, path, cost=cost) is None
        restored, declared_cost = model.load_solution_from_file(path)
        assert declared_cost == cost
        assert model.validate_solution(restored) == 2
    assert (tmp_path / "solution.yaml").read_text() == text
    assert model.target_state[x] == 2


def test_omitted_cost_and_empty_sequence():
    model = dp.Model()
    text = model.dump_solution_to_str([])
    assert text == model.dump_solution_to_str((), cost=None)
    assert model.load_solution_from_str(text) == ([], None)
    text = model.dump_solution_to_str([], cost=0)
    assert model.load_solution_from_str(text) == ([], 0)


@pytest.mark.parametrize("use_ids", [False, True])
def test_dump_does_not_check_feasibility_or_evaluate_expressions(
    model_and_step, use_ids
):
    model, step, step_id, x = model_and_step
    candidate = [step_id] if use_ids else [step]
    text = model.dump_solution_to_str(candidate, cost=99)
    transitions, cost = model.load_solution_from_str(text)
    assert cost == 99
    with pytest.raises(dp.ValidationError, match="final state.*state 1"):
        model.validate_solution(transitions)

    step = dp.Transition(
        name="division",
        effects=[(x, x - 1)],
        cost=dp.IntExpr.state_cost() // (x - 1),
    )
    step_id = model.add_transition(step)
    candidate = [step_id, step_id] if use_ids else [step, step]
    text = model.dump_solution_to_str(candidate)
    transitions, cost = model.load_solution_from_str(text)
    assert cost is None
    with pytest.raises(dp.ValidationError, match="cost of transition 2"):
        model.validate_solution(transitions)


@pytest.mark.parametrize("cost", [0, 1, 0.2, -0.5])
def test_continuous_costs_are_preserved(cost):
    model = dp.Model(float_cost=True)
    text = model.dump_solution_to_str([], cost=cost)
    transitions, declared_cost = model.load_solution_from_str(text)
    assert transitions == []
    assert type(declared_cost) is float
    assert declared_cost == cost


@pytest.mark.parametrize("cost", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_costs_are_rejected(cost):
    model = dp.Model(float_cost=True)
    with pytest.raises(ValueError, match="cost must be finite"):
        model.dump_solution_to_str([], cost=cost)


@pytest.mark.parametrize("float_cost, cost", [(False, 1.5), (False, "2"), (True, "2")])
def test_cost_type_errors(float_cost, cost):
    model = dp.Model(float_cost=float_cost)
    with pytest.raises(TypeError):
        model.dump_solution_to_str([], cost=cost)


@pytest.mark.parametrize("cost", [2147483648, -2147483649])
def test_integer_cost_overflow(cost):
    with pytest.raises(OverflowError):
        dp.Model().dump_solution_to_str([], cost=cost)


@pytest.mark.parametrize("value", [None, 0, "step", [0]])
def test_invalid_sequence_types(model_and_step, value):
    model, _, _, _ = model_and_step
    with pytest.raises(TypeError):
        model.dump_solution_to_str(value)


def test_mixed_sequence_types(model_and_step):
    model, step, step_id, _ = model_and_step
    for candidate in ([step, step_id], [step_id, step]):
        with pytest.raises(TypeError):
            model.dump_solution_to_str(candidate)


@pytest.mark.parametrize("forced", [False, True])
def test_invalid_ids_report_position(model_and_step, forced):
    model, step, step_id, _ = model_and_step
    backward_id = model.add_transition(step, backward=True, forced=forced)
    with pytest.raises(ValueError, match=r"transitions\[1\].*backward IDs"):
        model.dump_solution_to_str([step_id, backward_id])
    other = dp.Model()
    for _ in range(2):
        missing_id = other.add_transition(dp.Transition(name="other"), forced=forced)
    with pytest.raises(ValueError, match=r"transitions\[1\].*No such transition"):
        model.dump_solution_to_str([step_id, missing_id])


def test_forced_and_model_relative_ids(model_and_step):
    model, step, step_id, x = model_and_step
    forced = dp.Transition(name="forced", effects=[(x, x - 1)], cost=step.cost)
    forced_id = model.add_transition(forced, forced=True)
    text = model.dump_solution_to_str([step_id, forced_id])
    transitions, _ = model.load_solution_from_str(text)
    assert [t.name for t in transitions] == ["step", "forced"]
    assert model.validate_solution(transitions) == 2
    other = dp.Model()
    foreign_id = other.add_transition(dp.Transition(name="other"))
    assert model.dump_solution_to_str([foreign_id]) == model.dump_solution_to_str(
        [step_id]
    )


@pytest.mark.parametrize("name", ["true", "null", "visit i:3", "quoted ' name"])
def test_literal_names_round_trip(name):
    model = dp.Model()
    step = dp.Transition(name=name)
    step_id = model.add_transition(step)
    for candidate in ([step], [step_id]):
        restored, cost = model.load_solution_from_str(
            model.dump_solution_to_str(candidate)
        )
        assert [t.name for t in restored] == [name]
        assert cost is None


def test_grounded_parameter_maps_round_trip():
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
    candidate = model.load_solution_from_str(
        "cost: 2\ntransitions: [{name: visit, parameters: {i: 1}}, {name: visit, parameters: {i: 0}}]"
    )
    text = model.dump_solution_to_str(*candidate)
    restored, cost = model.load_solution_from_str(text)
    assert [t.name for t in restored] == ["visit i:1", "visit i:0"]
    assert model.validate_solution(restored) == cost == 2


def test_dump_serializes_references_not_transition_definitions(model_and_step):
    model, step, step_id, _ = model_and_step
    step.cost = 999
    assert model.dump_solution_to_str([step]) == model.dump_solution_to_str([step_id])
    text = model.dump_solution_to_str([dp.Transition(name="unregistered")])
    with pytest.raises(ValueError, match="no forward transition"):
        model.load_solution_from_str(text)


def test_solver_result_can_be_dumped(model_and_step):
    model, _, _, _ = model_and_step
    solution = dp.CABS(model, quiet=True).search()
    text = model.dump_solution_to_str(solution.transitions, cost=solution.cost)
    restored, cost = model.load_solution_from_str(text)
    assert model.validate_solution(restored) == cost == 2


def test_file_overwrite_and_errors(model_and_step, tmp_path):
    model, step, step_id, _ = model_and_step
    path = tmp_path / "solution.yaml"
    path.write_text("keep this content")
    backward_id = model.add_transition(step, backward=True)
    for candidate, cost, error in [
        ([backward_id], None, ValueError),
        ([step_id, step], None, TypeError),
        ([step_id], "bad cost", TypeError),
    ]:
        with pytest.raises(error):
            model.dump_solution_to_file(candidate, path, cost=cost)
        assert path.read_text() == "keep this content"
    model.dump_solution_to_file([step_id, step_id], path, cost=2)
    assert path.read_text() == model.dump_solution_to_str([step_id, step_id], cost=2)
    with pytest.raises(FileNotFoundError):
        model.dump_solution_to_file([], tmp_path / "missing" / "solution.yaml")
    with pytest.raises(OSError):
        model.dump_solution_to_file([], tmp_path)
