import pytest

import didppy as dp

error_cases = [
    ({"time_limit": 1800, "primal_bound": 1.5}, TypeError),
    ({"time_limit": -1}, BaseException),
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
        dp.LNBS(model, **kwargs)


def test_panic():
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

    with pytest.raises(BaseException):
        dp.LNBS(model, time_limit=1800)


def test_search():
    model = dp.Model()
    var = model.add_int_var(target=1)
    model.add_base_case([var == 0])
    t = dp.Transition(
        name="decrement", cost=dp.IntExpr.state_cost() + 1, effects=[(var, var - 1)]
    )
    model.add_transition(t)
    model.add_dual_bound(0)
    solver = dp.LNBS(model, time_limit=1800)
    solution = solver.search()

    assert solution.cost == 1
    assert model.validate_forward(solution.transitions, solution.cost, quiet=True)


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
    solver = dp.LNBS(model, time_limit=1800)
    solution, _ = solver.search_next()

    assert solution.cost == 1
    assert model.validate_forward(solution.transitions, solution.cost, quiet=True)


def test_hot_start():
    model = dp.Model()
    var = model.add_int_var(target=1)
    model.add_base_case([var == 0])
    t = dp.Transition(
        name="decrement",
        cost=dp.IntExpr.state_cost() + 1,
        effects=[(var, var - 1)],
    )
    model.add_transition(t)
    t = dp.Transition(
        name="expensive decrement",
        cost=dp.IntExpr.state_cost() + 2,
        effects=[(var, var - 1)],
    )
    model.add_transition(t)
    model.add_dual_bound(0)
    solver = dp.LNBS(model, time_limit=1800, initial_solution=[t])
    solution = solver.search()

    assert solution.cost == 1
    assert model.validate_forward(solution.transitions, solution.cost, quiet=True)
