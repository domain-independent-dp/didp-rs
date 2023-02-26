import didppy as dp
import pytest


error_cases = [
    ({"beam_size": 100, "primal_bound": 1.5}, TypeError),
    ({"beam_size": -1}, OverflowError),
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
        dp.ExpressionBeamSearch(model, **kwargs)


def test_search():
    model = dp.Model()
    var = model.add_int_var(target=1)
    model.add_base_case([var == 0])
    t = dp.Transition(
        name="decrement", cost=dp.IntExpr.state_cost() + 1, effects=[(var, var - 1)]
    )
    model.add_transition(t)
    model.add_dual_bound(0)
    solver = dp.ExpressionBeamSearch(model, 100)
    solution = solver.search()

    assert solution.cost == 1
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
    solver = dp.ExpressionBeamSearch(model, 100)

    with pytest.raises(BaseException):
        solver.search()
