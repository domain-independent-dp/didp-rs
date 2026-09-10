import didppy as dp
import pytest


error_cases = [
    ({"primal_bound": 1.5}, TypeError),
    ({"time_limit": -1}, BaseException),
    ({"initial_registry_capacity": -1}, OverflowError),
]


def resource_model():
    model = dp.Model()
    x = model.add_int_resource_var(target=1, less_is_better=True)
    model.add_base_case([x == 0])
    t = dp.Transition(
        name="decrement",
        cost=1 + dp.IntExpr.state_cost(),
        effects=[(x, x - 1)],
    )
    model.add_transition(t)
    model.add_dual_bound(x)
    return model, x


@pytest.mark.parametrize("kwargs, error", error_cases)
def test_error(kwargs, error):
    model, _ = resource_model()

    with pytest.raises(error):
        dp.Labeling(model, **kwargs)


def test_search():
    model, x = resource_model()
    solver = dp.Labeling(model, quiet=True)
    solution = solver.search()

    assert solution.cost == 1
    assert solution.state(model)[x] == 0
    assert model.validate_forward(solution.transitions, solution.cost, quiet=True)


def test_search_next():
    model, x = resource_model()
    solver = dp.Labeling(model, quiet=True)
    solution, terminated = solver.search_next()

    assert solution.cost == 1
    assert terminated
    assert model.validate_forward(solution.transitions, solution.cost, quiet=True)


def multi_category_resource_model():
    # Exercises node priority ordering across all four resource variable categories
    # (element, integer, continuous, set) at once, not just integer.
    model = dp.Model()
    obj = model.add_object_type(number=3)
    e = model.add_element_resource_var(object_type=obj, target=0, less_is_better=True)
    x = model.add_int_resource_var(target=1, less_is_better=True)
    c = model.add_float_resource_var(target=0.0, less_is_better=True)
    s = model.add_set_resource_var(object_type=obj, target={0, 1, 2}, less_is_better=True)
    model.add_base_case([x == 0])
    t = dp.Transition(
        name="decrement",
        cost=1 + dp.IntExpr.state_cost(),
        effects=[(x, x - 1)],
    )
    model.add_transition(t)
    model.add_dual_bound(x)
    return model, e, x, c, s


def test_search_with_multiple_resource_variable_categories():
    model, e, x, c, s = multi_category_resource_model()
    solver = dp.Labeling(model, quiet=True)
    solution = solver.search()

    assert solution.cost == 1
    state = solution.state(model)
    assert state[x] == 0
    assert state[e] == 0
    assert state[c] == 0.0
    assert state[s] == {0, 1, 2}
    assert model.validate_forward(solution.transitions, solution.cost, quiet=True)


def test_search_panic():
    model = dp.Model()
    obj = model.add_object_type(number=4)
    var = model.add_element_var(object_type=obj, target=3)
    table = model.add_int_table([1, 2, 3])
    model.add_base_case([var == 0])
    t = dp.Transition(
        name="decrement",
        cost=table[var + 1] + dp.IntExpr.state_cost(),
        effects=[(var, var - 1)],
    )
    model.add_transition(t)
    model.add_dual_bound(0)
    solver = dp.Labeling(model, quiet=True)

    with pytest.raises(BaseException):
        solver.search()


def test_search_next_panic():
    model = dp.Model()
    obj = model.add_object_type(number=4)
    var = model.add_element_var(object_type=obj, target=3)
    table = model.add_int_table([1, 2, 3])
    model.add_base_case([var == 0])
    t = dp.Transition(
        name="decrement",
        cost=table[var + 1] + dp.IntExpr.state_cost(),
        effects=[(var, var - 1)],
    )
    model.add_transition(t)
    model.add_dual_bound(0)
    solver = dp.Labeling(model, quiet=True)

    with pytest.raises(BaseException):
        solver.search_next()


def make_model(cost_fn):
    model = dp.Model()
    x = model.add_int_resource_var(target=1, less_is_better=True)
    model.add_base_case([x == 0])
    t = dp.Transition(name="decrement", cost=cost_fn(), effects=[(x, x - 1)])
    model.add_transition(t)
    model.add_dual_bound(x)
    return model, x


f_operator_cases = [
    (dp.FOperator.Plus, lambda: 1 + dp.IntExpr.state_cost(), 1),
    (dp.FOperator.Product, lambda: 2 * dp.IntExpr.state_cost(), 0),
    (dp.FOperator.Max, lambda: dp.max(1, dp.IntExpr.state_cost()), 1),
    (dp.FOperator.Min, lambda: dp.min(1, dp.IntExpr.state_cost()), 0),
]


@pytest.mark.parametrize("f_operator, cost_fn, expected", f_operator_cases)
def test_f_operator(f_operator, cost_fn, expected):
    model, _ = make_model(cost_fn)
    solver = dp.Labeling(model, f_operator=f_operator, quiet=True)
    solution = solver.search()

    assert solution.cost == expected
    assert model.validate_forward(solution.transitions, solution.cost, quiet=True)
