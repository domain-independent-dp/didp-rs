import didppy as dp
import pytest

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

    with pytest.raises(BaseException, match="index out of bounds"):
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
    assert solution.state(model)[var] == 0
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


@pytest.mark.parametrize("threads", [1, 2])
@pytest.mark.parametrize("float_cost", [False, True])
def test_transition_dominance_with_transition_mutex(threads, float_cost):
    model = dp.Model(float_cost=float_cost)
    obj = model.add_object_type(number=4)
    remaining = model.add_set_var(object_type=obj, target=[0, 1, 2, 3])
    model.add_base_case([remaining.is_empty()])
    state_cost = dp.FloatExpr.state_cost() if float_cost else dp.IntExpr.state_cost()
    transitions = []
    ids = []
    for i in range(4):
        transition = dp.Transition(
            name=f"remove {i}",
            cost=state_cost + 1,
            effects=[(remaining, remaining.remove(i))],
            preconditions=[remaining.contains(i)],
        )
        ids.append(model.add_transition(transition))
        transitions.append(transition)
    expensive = dp.Transition(
        name="expensive remove 3",
        cost=state_cost + 2,
        effects=[(remaining, remaining.remove(3))],
        preconditions=[remaining.contains(3)],
    )
    expensive_id = model.add_transition(expensive)
    model.add_transition_dominance(ids[3], expensive_id)
    model.add_dual_bound(0)

    # Start above the optimum so LNBS searches neighborhoods with fixed prefix
    # and suffix transitions, leaving gaps in the retained transition IDs.
    solution = dp.LNBS(
        model,
        time_limit=10,
        quiet=True,
        seed=2023,
        threads=threads,
        initial_solution=transitions[:3] + [expensive],
    ).search()

    assert solution.is_optimal
    assert solution.cost == 4
    assert solution.best_bound == 4
    assert model.validate_forward(solution.transitions, solution.cost, quiet=True)
