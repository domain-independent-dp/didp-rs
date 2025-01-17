import didppy as dp
import pytest


error_cases = [
    ({"primal_bound": 1.5}, TypeError),
    ({"time_limit": -1}, BaseException),
    ({"initial_registry_capacity": -1}, OverflowError),
    ({"width": -1}, OverflowError),
]


@pytest.mark.parametrize("kwargs, error", error_cases)
def test_error(kwargs, error):
    model = dp.Model()
    var = model.add_int_var(target=1)
    model.add_base_case([var == 0])
    t = dp.Transition(
        name="decrement", cost=dp.IntExpr.state_cost() + 1, effects=[(var, var - 1)]
    )
    model.add_transition(t)
    model.add_dual_bound(0)

    with pytest.raises(error):
        dp.DBDFS(model, **kwargs)


def test_search():
    model = dp.Model()
    var = model.add_int_var(target=1)
    model.add_base_case([var == 0])
    t = dp.Transition(
        name="decrement", cost=dp.IntExpr.state_cost() + 1, effects=[(var, var - 1)]
    )
    model.add_transition(t)
    model.add_dual_bound(0)
    solver = dp.DBDFS(model)
    solution = solver.search()

    assert solution.cost == 1
    assert solution.state(model)[var] == 0
    assert model.validate_forward(solution.transitions, solution.cost, quiet=True)


def test_search_panic():
    model = dp.Model()
    obj = model.add_object_type(number=4)
    var = model.add_element_var(object_type=obj, target=3)
    table = model.add_int_table([1, 2, 3])
    model.add_base_case([var == 0])
    t = dp.Transition(
        name="decrement",
        cost=dp.IntExpr.state_cost() + table[var + 1],
        effects=[(var, var - 1)],
    )
    model.add_transition(t)
    model.add_dual_bound(0)
    solver = dp.DBDFS(model)

    with pytest.raises(BaseException):
        solver.search()


def test_search_next():
    model = dp.Model()
    var = model.add_int_var(target=1)
    model.add_base_case([var == 0])
    t = dp.Transition(
        name="decrement",
        cost=dp.IntExpr.state_cost() + 1,
        effects=[(var, var - 1)],
    )
    model.add_transition(t)
    model.add_dual_bound(0)
    solver = dp.DBDFS(model)
    solution, _ = solver.search_next()

    assert solution.cost == 1
    assert model.validate_forward(solution.transitions, solution.cost, quiet=True)


def test_search_next_panic():
    model = dp.Model()
    obj = model.add_object_type(number=4)
    var = model.add_element_var(object_type=obj, target=3)
    table = model.add_int_table([1, 2, 3])
    model.add_base_case([var == 0])
    t = dp.Transition(
        name="decrement",
        cost=dp.IntExpr.state_cost() + table[var + 1],
        effects=[(var, var - 1)],
    )
    model.add_transition(t)
    model.add_dual_bound(0)
    solver = dp.DBDFS(model)

    with pytest.raises(BaseException):
        solver.search_next()
