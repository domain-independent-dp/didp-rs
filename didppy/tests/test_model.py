import pytest

import didppy as dp


def test_default():
    model = dp.Model()

    assert not model.maximize
    assert not model.float_cost
    assert model.state_constrs == []
    assert model.base_cases == []
    assert model.dual_bounds == []


def test_init():
    model = dp.Model(maximize=True, float_cost=True)

    assert model.maximize
    assert model.float_cost
    assert model.state_constrs == []
    assert model.base_cases == []
    assert model.dual_bounds == []


def test_set_minimize():
    model = dp.Model(maximize=True)

    assert model.maximize

    model.maximize = False

    assert not model.maximize


def test_set_maximize():
    model = dp.Model(maximize=False)

    assert not model.maximize

    model.maximize = True

    assert model.maximize


def test_add_object_type():
    model = dp.Model()
    obj = model.add_object_type(number=4)

    assert model.get_number_of_object(obj) == 4


def test_add_object_type_with_name():
    model = dp.Model()
    model.add_object_type(number=4, name="obj")
    obj = model.get_object_type("obj")

    assert model.get_number_of_object(obj) == 4


def test_add_object_type_error():
    model = dp.Model()
    model.add_object_type(number=4, name="obj")

    with pytest.raises(RuntimeError):
        model.add_object_type(number=4, name="obj")


set_const_cases = [
    ([], set()),
    (set(), set()),
    ([1, 2, 3], {1, 2, 3}),
    ({1, 2, 3}, {1, 2, 3}),
]


@pytest.mark.parametrize("value, expected", set_const_cases)
def test_create_set_const(value, expected):
    model = dp.Model()
    obj = model.add_object_type(number=4)
    const = model.create_set_const(object_type=obj, value=value)

    assert const.eval() == expected


set_const_error_cases = [
    ({}, TypeError),
    ([1, 2, 3, 4], RuntimeError),
    ({1, 2, 3, 4}, RuntimeError),
    ([-1], TypeError),
    ({-1}, TypeError),
]


@pytest.mark.parametrize("value, error", set_const_error_cases)
def test_create_set_const_error(value, error):
    model = dp.Model()
    obj = model.add_object_type(number=4)

    with pytest.raises(error):
        model.create_set_const(object_type=obj, value=value)


def test_crate_set_const_not_included_error():
    model = dp.Model()
    obj = model.add_object_type(number=4)
    model = dp.Model()

    with pytest.raises(RuntimeError):
        model.create_set_const(object_type=obj, value=[])


def test_add_element_var():
    model = dp.Model()
    obj = model.add_object_type(number=4)
    var = model.add_element_var(object_type=obj, target=1)
    obj = model.get_object_type_of(var)

    assert model.get_number_of_object(obj) == 4
    assert model.get_target(var) == 1


def test_add_element_var_with_name():
    model = dp.Model()
    obj = model.add_object_type(number=4)
    model.add_element_var(object_type=obj, target=1, name="var")
    var = model.get_element_var("var")
    obj = model.get_object_type_of(var)

    assert model.get_number_of_object(obj) == 4
    assert model.get_target(var) == 1


element_var_error_cases = [(-1, OverflowError), (1.5, TypeError)]


@pytest.mark.parametrize("value, error", element_var_error_cases)
def test_add_element_var_error(value, error):
    model = dp.Model()
    obj = model.add_object_type(number=4)

    with pytest.raises(error):
        model.add_element_var(object_type=obj, target=value)


def test_add_element_var_name_error():
    model = dp.Model()
    obj = model.add_object_type(number=4)
    model.add_element_var(object_type=obj, target=1, name="var")

    with pytest.raises(RuntimeError):
        model.add_element_var(object_type=obj, target=1, name="var")


def test_add_element_var_not_included():
    model = dp.Model()
    obj = model.add_object_type(number=4)
    model = dp.Model()

    with pytest.raises(RuntimeError):
        model.add_element_var(object_type=obj, target=1)


def test_get_element_var_error():
    model = dp.Model()

    with pytest.raises(RuntimeError):
        model.get_element_var("var")


def test_add_element_resource_var_default():
    model = dp.Model()
    obj = model.add_object_type(number=4)
    var = model.add_element_resource_var(
        object_type=obj,
        target=1,
    )
    obj = model.get_object_type_of(var)

    assert model.get_number_of_object(obj) == 4
    assert model.get_target(var) == 1
    assert not model.get_preference(var)


preference_cases = [True, False]


@pytest.mark.parametrize("preference", preference_cases)
def test_add_element_resource_var(preference):
    model = dp.Model()
    obj = model.add_object_type(number=4)
    var = model.add_element_resource_var(
        object_type=obj, target=1, less_is_better=preference
    )
    obj = model.get_object_type_of(var)

    assert model.get_number_of_object(obj) == 4
    assert model.get_target(var) == 1
    assert model.get_preference(var) == preference


@pytest.mark.parametrize("preference,", preference_cases)
def test_add_element_resource_var_with_name(preference):
    model = dp.Model()
    obj = model.add_object_type(number=4)
    model.add_element_resource_var(
        object_type=obj, target=1, name="var", less_is_better=preference
    )
    var = model.get_element_resource_var("var")
    obj = model.get_object_type_of(var)

    assert model.get_number_of_object(obj) == 4
    assert model.get_target(var) == 1
    assert model.get_preference(var) == preference


@pytest.mark.parametrize("value, error", element_var_error_cases)
def test_add_element_resource_var_error(value, error):
    model = dp.Model()
    obj = model.add_object_type(number=4)

    with pytest.raises(error):
        model.add_element_resource_var(object_type=obj, target=value)


def test_add_element_resource_var_name_error():
    model = dp.Model()
    obj = model.add_object_type(number=4)
    model.add_element_resource_var(object_type=obj, target=1, name="var")

    with pytest.raises(RuntimeError):
        model.add_element_resource_var(object_type=obj, target=1, name="var")


def test_add_element_resource_var_not_included():
    model = dp.Model()
    obj = model.add_object_type(number=4)
    model = dp.Model()

    with pytest.raises(RuntimeError):
        model.add_element_resource_var(object_type=obj, target=1)


def test_get_element_resource_var_error():
    model = dp.Model()

    with pytest.raises(RuntimeError):
        model.get_element_resource_var("var")


set_var_cases = [
    ([], set()),
    (set(), set()),
    ([0, 1], {0, 1}),
    ({0, 1}, {0, 1}),
]


@pytest.mark.parametrize("target, expected", set_var_cases)
def test_add_set_var(target, expected):
    model = dp.Model()
    obj = model.add_object_type(number=4)
    var = model.add_set_var(object_type=obj, target=target)
    obj = model.get_object_type_of(var)

    assert model.get_number_of_object(obj) == 4
    assert model.get_target(var) == expected


def test_add_set_var_const():
    model = dp.Model()
    obj = model.add_object_type(number=4)
    const = model.create_set_const(object_type=obj, value={0, 1})
    var = model.add_set_var(object_type=obj, target=const)
    obj = model.get_object_type_of(var)

    assert model.get_number_of_object(obj) == 4
    assert model.get_target(var) == {0, 1}


set_var_error_cases = [
    ({}, TypeError),
    ([-1], TypeError),
    ({-1}, TypeError),
    ({1.5}, TypeError),
]


@pytest.mark.parametrize("target, error", set_var_error_cases)
def test_add_set_var_error(target, error):
    model = dp.Model()
    obj = model.add_object_type(number=4)

    with pytest.raises(error):
        model.add_set_var(object_type=obj, target=target)


def test_add_set_var_name_error():
    model = dp.Model()
    obj = model.add_object_type(number=4)
    model.add_set_var(object_type=obj, target={0, 1}, name="var")

    with pytest.raises(RuntimeError):
        model.add_set_var(object_type=obj, target={0, 1}, name="var")


def test_add_set_var_not_included():
    model = dp.Model()
    obj = model.add_object_type(number=4)
    model = dp.Model()

    with pytest.raises(RuntimeError):
        model.add_set_var(object_type=obj, target={0, 1})


int_var_cases = [1, -1]


@pytest.mark.parametrize("target", int_var_cases)
def test_add_int_var(target):
    model = dp.Model()
    var = model.add_int_var(target=target)

    assert model.get_target(var) == target


@pytest.mark.parametrize("target", int_var_cases)
def test_add_int_var_with_name(target):
    model = dp.Model()
    model.add_int_var(target=target, name="var")
    var = model.get_int_var("var")

    assert model.get_target(var) == target


def test_add_int_var_error():
    model = dp.Model()

    with pytest.raises(TypeError):
        model.add_int_var(target=1.5)


def test_add_int_var_name_error():
    model = dp.Model()
    model.add_int_var(target=1, name="var")

    with pytest.raises(RuntimeError):
        model.add_int_var(target=1, name="var")


def test_get_int_var_error():
    model = dp.Model()

    with pytest.raises(RuntimeError):
        model.get_int_var("var")


@pytest.mark.parametrize("target", int_var_cases)
def test_add_int_resource_var_default(target):
    model = dp.Model()
    var = model.add_int_resource_var(target=target)

    assert model.get_target(var) == target
    assert not model.get_preference(var)


@pytest.mark.parametrize("preference", preference_cases)
def test_add_int_resource_var(preference):
    model = dp.Model()
    var = model.add_int_resource_var(target=1, less_is_better=preference)

    assert model.get_target(var) == 1
    assert model.get_preference(var) == preference


@pytest.mark.parametrize("preference,", preference_cases)
def test_add_int_resource_var_with_name(preference):
    model = dp.Model()
    model.add_int_resource_var(target=1, name="var", less_is_better=preference)
    var = model.get_int_resource_var("var")

    assert model.get_target(var) == 1
    assert model.get_preference(var) == preference


def test_add_int_resource_var_error():
    model = dp.Model()

    with pytest.raises(TypeError):
        model.add_int_resource_var(target=1.5)


def test_add_int_resource_var_name_error():
    model = dp.Model()
    model.add_int_resource_var(target=1, name="var")

    with pytest.raises(RuntimeError):
        model.add_int_resource_var(target=1, name="var")


def test_get_int_resource_var_error():
    model = dp.Model()

    with pytest.raises(RuntimeError):
        model.get_int_resource_var("var")


float_var_cases = [
    (1, pytest.approx(1.0)),
    (-1, pytest.approx(-1.0)),
    (1.5, pytest.approx(1.5)),
    (-1.5, pytest.approx(-1.5)),
]


@pytest.mark.parametrize("target, expected", float_var_cases)
def test_add_float_var(target, expected):
    model = dp.Model()
    var = model.add_float_var(target=target)

    assert model.get_target(var) == expected


@pytest.mark.parametrize("target, expected", float_var_cases)
def test_add_float_var_with_name(target, expected):
    model = dp.Model()
    model.add_float_var(target=target, name="var")
    var = model.get_float_var("var")

    assert model.get_target(var) == expected


def test_add_float_var_error():
    model = dp.Model()

    with pytest.raises(TypeError):
        model.add_float_var(target=())


def test_add_float_var_name_error():
    model = dp.Model()
    model.add_float_var(target=1, name="var")

    with pytest.raises(RuntimeError):
        model.add_float_var(target=1, name="var")


def test_get_float_var_error():
    model = dp.Model()

    with pytest.raises(RuntimeError):
        model.get_float_var("var")


@pytest.mark.parametrize("target, expected", float_var_cases)
def test_add_float_resource_var_default(target, expected):
    model = dp.Model()
    var = model.add_float_resource_var(target=target)

    assert model.get_target(var) == expected
    assert not model.get_preference(var)


@pytest.mark.parametrize("preference", preference_cases)
def test_add_float_resource_var(preference):
    model = dp.Model()
    var = model.add_float_resource_var(target=1, less_is_better=preference)

    assert model.get_target(var) == pytest.approx(1.0)
    assert model.get_preference(var) == preference


@pytest.mark.parametrize("preference,", preference_cases)
def test_add_float_resource_var_with_name(preference):
    model = dp.Model()
    model.add_float_resource_var(target=1, name="var", less_is_better=preference)
    var = model.get_float_resource_var("var")

    assert model.get_target(var) == pytest.approx(1.0)
    assert model.get_preference(var) == preference


def test_add_float_resource_var_error():
    model = dp.Model()

    with pytest.raises(TypeError):
        model.add_float_resource_var(target=())


def test_add_float_resource_var_name_error():
    model = dp.Model()
    model.add_float_resource_var(target=1, name="var")

    with pytest.raises(RuntimeError):
        model.add_float_resource_var(target=1, name="var")


def test_get_float_resource_var_error():
    model = dp.Model()

    with pytest.raises(RuntimeError):
        model.get_float_resource_var("var")


class TestSetTarget:
    model = dp.Model()
    obj = model.add_object_type(number=4)
    element_var = model.add_element_var(object_type=obj, target=1)
    element_resource_var = model.add_element_resource_var(
        object_type=obj, target=2, less_is_better=True
    )
    set_var = model.add_set_var(object_type=obj, target=[0, 1])
    int_var = model.add_int_var(target=3)
    int_resource_var = model.add_int_resource_var(target=4, less_is_better=True)
    float_var = model.add_float_var(target=0.5)
    float_resource_var = model.add_float_resource_var(target=0.6, less_is_better=True)
    set_const = model.create_set_const(object_type=obj, value=[1, 2])

    cases = [
        (element_var, 2, 2),
        (element_resource_var, 3, 3),
        (set_var, set_const, {1, 2}),
        (set_var, [1, 2], {1, 2}),
        (set_var, {1, 2}, {1, 2}),
        (int_var, 4, 4),
        (int_resource_var, 5, 5),
        (float_var, 0.6, pytest.approx(0.6)),
        (float_var, 6, pytest.approx(6)),
        (float_resource_var, 0.7, pytest.approx(0.7)),
        (float_resource_var, 7, pytest.approx(7)),
    ]

    @pytest.mark.parametrize("var, target, expected", cases)
    def test(self, var, target, expected):
        self.model.set_target(var, target)

        assert self.model.get_target(var) == expected

    error_cases = [
        (element_var, -1, OverflowError),
        (element_var, 1.5, TypeError),
        (element_resource_var, -1, OverflowError),
        (element_resource_var, 1.5, TypeError),
        (set_var, {}, TypeError),
        (set_var, {-1}, TypeError),
        (set_var, {1.5}, TypeError),
        (set_var, {4}, RuntimeError),
        (int_var, 1.5, TypeError),
        (int_resource_var, 1.5, TypeError),
        (float_var, (), TypeError),
        (float_resource_var, (), TypeError),
    ]

    @pytest.mark.parametrize("var, target, error", error_cases)
    def test_error(self, var, target, error):
        with pytest.raises(error):
            self.model.set_target(var, target)


def test_set_preference_element_true():
    model = dp.Model()
    obj = model.add_object_type(number=4)
    var = model.add_element_resource_var(
        object_type=obj, target=2, less_is_better=False
    )
    model.set_preference(var, True)

    assert model.get_preference(var)


def test_set_preference_element_false():
    model = dp.Model()
    obj = model.add_object_type(number=4)
    var = model.add_element_resource_var(object_type=obj, target=2, less_is_better=True)
    model.set_preference(var, False)

    assert not model.get_preference(var)


def test_set_preference_int_true():
    model = dp.Model()
    var = model.add_int_resource_var(target=2, less_is_better=False)
    model.set_preference(var, True)

    assert model.get_preference(var)


def test_set_preference_int_false():
    model = dp.Model()
    var = model.add_int_resource_var(target=2, less_is_better=True)
    model.set_preference(var, False)

    assert not model.get_preference(var)


def test_set_preference_float_true():
    model = dp.Model()
    var = model.add_float_resource_var(target=2, less_is_better=False)
    model.set_preference(var, True)

    assert model.get_preference(var)


def test_set_preference_float_false():
    model = dp.Model()
    var = model.add_float_resource_var(target=2, less_is_better=True)
    model.set_preference(var, False)

    assert not model.get_preference(var)


def test_add_state_constr():
    model = dp.Model()
    obj = model.add_object_type(number=4)
    set_var = model.add_set_var(object_type=obj, target=[0, 1])
    int_var = model.add_int_var(target=3)
    state = model.target_state

    model.add_state_constr(~set_var.contains(2) | (int_var > 3))

    assert len(model.state_constrs) == 1
    assert model.state_constrs[0].eval(state, model)

    model.add_state_constr(int_var < 3)
    assert len(model.state_constrs) == 2
    assert model.state_constrs[0].eval(state, model)
    assert not model.state_constrs[1].eval(state, model)


def test_add_state_constr_error():
    model = dp.Model()

    with pytest.raises(RuntimeError):
        model.add_state_constr(dp.IntExpr.state_cost() > 0)


def test_add_state_constr_var_not_included_error():
    model = dp.Model()
    var = model.add_int_var(target=3)
    model = dp.Model()

    with pytest.raises(RuntimeError):
        model.add_state_constr(var > 0)


def test_add_state_constr_table_not_included_error():
    model = dp.Model()
    table = model.add_int_table([1, 2, 3])
    model = dp.Model()

    with pytest.raises(RuntimeError):
        model.add_state_constr(table[0] > 0)


def test_check_state_constr_true():
    model = dp.Model()
    obj = model.add_object_type(number=4)
    set_var = model.add_set_var(object_type=obj, target=[0, 1])
    int_var = model.add_int_var(target=3)
    model.add_state_constr(set_var.contains(0))
    model.add_state_constr(set_var.contains(1))
    model.add_state_constr(int_var > 0)
    model.add_state_constr(int_var < 4)
    state = model.target_state

    assert model.check_state_constr(state)


def test_check_state_constr_false():
    model = dp.Model()
    obj = model.add_object_type(number=4)
    set_var = model.add_set_var(object_type=obj, target=[0, 1])
    int_var = model.add_int_var(target=3)
    model.add_state_constr(set_var.contains(0))
    model.add_state_constr(set_var.contains(1))
    model.add_state_constr(set_var.contains(2))
    model.add_state_constr(int_var > 0)
    model.add_state_constr(int_var < 4)
    state = model.target_state

    assert not model.check_state_constr(state)


def test_check_state_constr_error():
    model = dp.Model()
    obj = model.add_object_type(number=4)
    var = model.add_element_var(object_type=obj, target=3)
    table = model.add_int_table([0, 1, 2, 3])
    model.add_state_constr(table[var + 1] > 0)

    with pytest.raises(BaseException):
        model.check_state_constr(model.target_state)


def test_add_state_constr_panic():
    model = dp.Model()
    table = model.add_int_table([1, 2, 3])

    with pytest.raises(BaseException):
        model.add_state_constr(table[4] > 0)


def test_add_base_case():
    model = dp.Model()
    obj = model.add_object_type(number=4)
    set_var = model.add_set_var(object_type=obj, target=[0, 1])
    int_var = model.add_int_var(target=3)
    state = model.target_state

    model.add_base_case([set_var.contains(0), set_var.contains(1)])

    assert len(model.base_cases) == 1
    assert len(model.base_cases[0]) == 2
    assert model.base_cases[0][0][0].eval(state, model)
    assert model.base_cases[0][0][1].eval(state, model)
    assert model.base_cases[0][1].eval(state, model) == 0

    model.add_base_case([(int_var < 3) | ~set_var.contains(1)])

    assert len(model.base_cases) == 2
    assert model.base_cases[0][0][0].eval(state, model)
    assert model.base_cases[0][0][1].eval(state, model)
    assert model.base_cases[0][1].eval(state, model) == 0
    assert not model.base_cases[1][0][0].eval(state, model)
    assert model.base_cases[1][1].eval(state, model) == 0


def test_add_base_case_with_cost():
    model = dp.Model()
    obj = model.add_object_type(number=4)
    set_var = model.add_set_var(object_type=obj, target=[0, 1])
    int_var = model.add_int_var(target=3)
    state = model.target_state

    model.add_base_case([set_var.contains(0), set_var.contains(1)], cost=int_var + 1)

    assert len(model.base_cases) == 1
    assert len(model.base_cases[0]) == 2
    assert model.base_cases[0][0][0].eval(state, model)
    assert model.base_cases[0][0][1].eval(state, model)
    assert model.base_cases[0][1].eval(state, model) == 4

    model.add_base_case([(int_var < 3) | ~set_var.contains(1)], cost=2)

    assert len(model.base_cases) == 2
    assert model.base_cases[0][0][0].eval(state, model)
    assert model.base_cases[0][0][1].eval(state, model)
    assert model.base_cases[0][1].eval(state, model) == 4
    assert not model.base_cases[1][0][0].eval(state, model)
    assert model.base_cases[1][1].eval(state, model) == 2


def test_add_base_case_error():
    model = dp.Model()

    with pytest.raises(RuntimeError):
        model.add_base_case([dp.IntExpr.state_cost() > 0])


def test_add_base_case_panic():
    model = dp.Model()
    int_var = model.add_int_var(target=3)

    with pytest.raises(BaseException):
        model.add_base_case([int_var > 0], cost=dp.IntExpr.state_cost())


def test_add_base_case_panic_cost():
    model = dp.Model()
    table = model.add_int_table([1, 2, 3])

    with pytest.raises(BaseException):
        model.add_base_case([table[4] > 0])


def test_add_base_case_var_not_included_error():
    model = dp.Model()
    var = model.add_int_var(target=3)
    model = dp.Model()

    with pytest.raises(RuntimeError):
        model.add_base_case([var > 0])


def test_add_base_case_table_not_included_error():
    model = dp.Model()
    table = model.add_int_table([1, 2, 3])
    model = dp.Model()

    with pytest.raises(RuntimeError):
        model.add_base_case([table[0] > 0])


def test_check_base_case_true():
    model = dp.Model()
    obj = model.add_object_type(number=4)
    set_var = model.add_set_var(object_type=obj, target=[0, 1])
    int_var = model.add_int_var(target=3)
    model.add_base_case([set_var.contains(0), set_var.contains(1)])
    model.add_base_case([int_var < 2, int_var > 0])
    state = model.target_state

    assert model.is_base(state)


def test_check_base_case_false():
    model = dp.Model()
    obj = model.add_object_type(number=4)
    set_var = model.add_set_var(object_type=obj, target=[0, 1])
    int_var = model.add_int_var(target=3)
    model.add_base_case(
        [
            set_var.contains(0),
            set_var.contains(1),
            set_var.contains(2),
        ]
    )
    model.add_base_case([int_var < 2, int_var > 0])
    state = model.target_state

    assert not model.is_base(state)


def test_check_base_case_error():
    model = dp.Model()
    obj = model.add_object_type(number=4)
    var = model.add_element_var(object_type=obj, target=3)
    table = model.add_int_table([0, 1, 2, 3])
    model.add_base_case([table[var + 1] > 0])

    with pytest.raises(BaseException):
        model.is_base(model.target_state)


def test_eval_base_cost_zero():
    model = dp.Model()
    obj = model.add_object_type(number=4)
    set_var = model.add_set_var(object_type=obj, target=[0, 1])
    int_var = model.add_int_var(target=3)
    model.add_base_case([set_var.contains(0), set_var.contains(1)])
    model.add_base_case([int_var < 2, int_var > 0])
    state = model.target_state

    assert model.eval_base_cost(state) == 0


def test_eval_base_cost_some():
    model = dp.Model()
    model.maximize = False
    obj = model.add_object_type(number=4)
    set_var = model.add_set_var(object_type=obj, target=[0, 1])
    int_var = model.add_int_var(target=3)
    model.add_base_case([set_var.contains(0), set_var.contains(1)], cost=int_var)
    model.add_base_case([int_var <= 3, int_var >= 2], cost=4)
    model.add_base_case([int_var < 2, int_var > 0], cost=2)
    state = model.target_state

    assert model.eval_base_cost(state) == 3


def test_eval_base_case_none():
    model = dp.Model()
    obj = model.add_object_type(number=4)
    set_var = model.add_set_var(object_type=obj, target=[0, 1])
    int_var = model.add_int_var(target=3)
    model.add_base_case(
        [
            set_var.contains(0),
            set_var.contains(1),
            set_var.contains(2),
        ],
        cost=2,
    )
    model.add_base_case([int_var < 2, int_var > 0])
    state = model.target_state

    assert model.eval_base_cost(state) is None


def test_add_transition():
    model = dp.Model()

    assert model.get_transitions() == []
    assert model.get_transitions(forced=False) == []
    assert model.get_transitions(forced=True) == []

    t1 = dp.Transition(name="t1")
    model.add_transition(t1)

    assert len(model.get_transitions()) == 1
    assert model.get_transitions()[0].name == "t1"
    assert len(model.get_transitions(forced=False)) == 1
    assert model.get_transitions(forced=False)[0].name == "t1"
    assert model.get_transitions(forced=True) == []

    t2 = dp.Transition(name="t2")
    model.add_transition(t2)

    assert len(model.get_transitions()) == 2
    assert model.get_transitions()[0].name == "t1"
    assert model.get_transitions()[1].name == "t2"
    assert len(model.get_transitions(forced=False)) == 2
    assert model.get_transitions(forced=False)[0].name == "t1"
    assert model.get_transitions(forced=False)[1].name == "t2"
    assert model.get_transitions(forced=True) == []

    ft1 = dp.Transition(name="ft1")
    model.add_transition(ft1, forced=True)

    assert len(model.get_transitions()) == 2
    assert model.get_transitions()[0].name == "t1"
    assert model.get_transitions()[1].name == "t2"
    assert len(model.get_transitions(forced=False)) == 2
    assert model.get_transitions(forced=False)[0].name == "t1"
    assert model.get_transitions(forced=False)[1].name == "t2"
    assert len(model.get_transitions(forced=True)) == 1
    assert model.get_transitions(forced=True)[0].name == "ft1"

    ft2 = dp.Transition(name="ft2")
    model.add_transition(ft2, forced=True)

    assert len(model.get_transitions()) == 2
    assert model.get_transitions()[0].name == "t1"
    assert model.get_transitions()[1].name == "t2"
    assert len(model.get_transitions(forced=False)) == 2
    assert model.get_transitions(forced=False)[0].name == "t1"
    assert model.get_transitions(forced=False)[1].name == "t2"
    assert len(model.get_transitions(forced=True)) == 2
    assert model.get_transitions(forced=True)[0].name == "ft1"
    assert model.get_transitions(forced=True)[1].name == "ft2"


class TestTransitionError:
    model = dp.Model()
    int_var = model.add_int_var(target=3)
    int_table = model.add_int_table([1, 2, 3, 4])

    model = dp.Model(float_cost=False)
    float_var = model.add_float_var(target=0.5)
    float_table = model.add_float_table([0.1, 0.2, 0.3, 0.4])

    cases = [
        (dp.Transition(name="t", cost=dp.FloatExpr.state_cost()), RuntimeError),
        (dp.Transition(name="t", cost=int_table[0]), RuntimeError),
        (dp.Transition(name="t", cost=float_table[4]), BaseException),
        (dp.Transition(name="t", effects=[(int_var, 0)]), RuntimeError),
        (
            dp.Transition(name="t", effects=[(float_var, dp.FloatExpr.state_cost())]),
            RuntimeError,
        ),
        (
            dp.Transition(name="t", effects=[(float_var, int_table[0])]),
            RuntimeError,
        ),
        (
            dp.Transition(name="t", effects=[(float_var, float_table[4])]),
            BaseException,
        ),
        (dp.Transition(name="t", preconditions=[int_var < 0]), RuntimeError),
        (dp.Transition(name="t", preconditions=[int_table[0] < 0]), RuntimeError),
        (dp.Transition(name="t", preconditions=[float_table[4] < 0]), BaseException),
    ]

    @pytest.mark.parametrize("transition, error", cases)
    def test(self, transition, error):
        with pytest.raises(error):
            self.model.add_transition(transition)

        with pytest.raises(error):
            self.model.add_transition(transition, forced=True)


def test_add_dual_bound_int():
    model = dp.Model(float_cost=False)
    var = model.add_int_var(target=0)
    resource_var = model.add_int_resource_var(target=1)
    state = model.target_state

    model.add_dual_bound(2)

    assert len(model.dual_bounds) == 1
    assert model.dual_bounds[0].eval(state, model) == 2

    model.add_dual_bound(dp.IntExpr(3))

    assert len(model.dual_bounds) == 2
    assert model.dual_bounds[0].eval(state, model) == 2
    assert model.dual_bounds[1].eval(state, model) == 3

    model.add_dual_bound(var)

    assert len(model.dual_bounds) == 3
    assert model.dual_bounds[0].eval(state, model) == 2
    assert model.dual_bounds[1].eval(state, model) == 3
    assert model.dual_bounds[2].eval(state, model) == 0

    model.add_dual_bound(resource_var)

    assert len(model.dual_bounds) == 4
    assert model.dual_bounds[0].eval(state, model) == 2
    assert model.dual_bounds[1].eval(state, model) == 3
    assert model.dual_bounds[2].eval(state, model) == 0
    assert model.dual_bounds[3].eval(state, model) == 1

    model.add_dual_bound(-1)

    assert len(model.dual_bounds) == 5
    assert model.dual_bounds[0].eval(state, model) == 2
    assert model.dual_bounds[1].eval(state, model) == 3
    assert model.dual_bounds[2].eval(state, model) == 0
    assert model.dual_bounds[3].eval(state, model) == 1
    assert model.dual_bounds[4].eval(state, model) == -1


def test_add_dual_bound_float():
    model = dp.Model(float_cost=True)
    int_var = model.add_int_var(target=0)
    int_resource_var = model.add_int_resource_var(target=1)
    float_var = model.add_float_var(target=0.5)
    float_resource_var = model.add_float_resource_var(target=0.6)
    state = model.target_state

    model.add_dual_bound(2)

    assert len(model.dual_bounds) == 1
    assert model.dual_bounds[0].eval(state, model) == 2

    model.add_dual_bound(dp.IntExpr(3))

    assert len(model.dual_bounds) == 2
    assert model.dual_bounds[0].eval(state, model) == 2
    assert model.dual_bounds[1].eval(state, model) == 3

    model.add_dual_bound(int_var)

    assert len(model.dual_bounds) == 3
    assert model.dual_bounds[0].eval(state, model) == 2
    assert model.dual_bounds[1].eval(state, model) == 3
    assert model.dual_bounds[2].eval(state, model) == 0

    model.add_dual_bound(int_resource_var)

    assert len(model.dual_bounds) == 4
    assert model.dual_bounds[0].eval(state, model) == 2
    assert model.dual_bounds[1].eval(state, model) == 3
    assert model.dual_bounds[2].eval(state, model) == 0
    assert model.dual_bounds[3].eval(state, model) == 1

    model.add_dual_bound(-1)

    assert len(model.dual_bounds) == 5
    assert model.dual_bounds[0].eval(state, model) == 2
    assert model.dual_bounds[1].eval(state, model) == 3
    assert model.dual_bounds[2].eval(state, model) == 0
    assert model.dual_bounds[3].eval(state, model) == 1
    assert model.dual_bounds[4].eval(state, model) == -1

    model.add_dual_bound(0.7)

    assert len(model.dual_bounds) == 6
    assert model.dual_bounds[0].eval(state, model) == 2
    assert model.dual_bounds[1].eval(state, model) == 3
    assert model.dual_bounds[2].eval(state, model) == 0
    assert model.dual_bounds[3].eval(state, model) == 1
    assert model.dual_bounds[4].eval(state, model) == -1
    assert model.dual_bounds[5].eval(state, model) == pytest.approx(0.7)

    model.add_dual_bound(float_var)

    assert len(model.dual_bounds) == 7
    assert model.dual_bounds[0].eval(state, model) == 2
    assert model.dual_bounds[1].eval(state, model) == 3
    assert model.dual_bounds[2].eval(state, model) == 0
    assert model.dual_bounds[3].eval(state, model) == 1
    assert model.dual_bounds[4].eval(state, model) == -1
    assert model.dual_bounds[5].eval(state, model) == pytest.approx(0.7)
    assert model.dual_bounds[6].eval(state, model) == pytest.approx(0.5)

    model.add_dual_bound(float_resource_var)

    assert len(model.dual_bounds) == 8
    assert model.dual_bounds[0].eval(state, model) == 2
    assert model.dual_bounds[1].eval(state, model) == 3
    assert model.dual_bounds[2].eval(state, model) == 0
    assert model.dual_bounds[3].eval(state, model) == 1
    assert model.dual_bounds[4].eval(state, model) == -1
    assert model.dual_bounds[5].eval(state, model) == pytest.approx(0.7)
    assert model.dual_bounds[6].eval(state, model) == pytest.approx(0.5)
    assert model.dual_bounds[7].eval(state, model) == pytest.approx(0.6)


class TestAddDualBoundError:
    model = dp.Model(float_cost=False)
    float_var = model.add_float_var(target=0.5)
    float_resource_var = model.add_float_resource_var(target=0.6)
    table = model.add_int_table([0, 1, 2])

    int_cases = [
        (1.5, RuntimeError),
        (dp.FloatExpr(1.5), RuntimeError),
        (float_var, RuntimeError),
        (float_resource_var, RuntimeError),
        (dp.IntExpr.state_cost(), RuntimeError),
        (table[3], BaseException),
    ]

    @pytest.mark.parametrize("value, error", int_cases)
    def test_int(self, value, error):
        with pytest.raises(error):
            self.model.add_dual_bound(value)

    float_cases = [
        (dp.FloatExpr.state_cost(), RuntimeError),
        (float_var, RuntimeError),
        (float_resource_var, RuntimeError),
        (table[0], RuntimeError),
    ]

    @pytest.mark.parametrize("value, error", float_cases)
    def test_not_included(self, value, error):
        model = dp.Model(float_cost=True)

        with pytest.raises(error):
            model.add_dual_bound(value)


def test_eval_dual_bound_int_minimize():
    model = dp.Model(float_cost=False, maximize=False)
    int_var = model.add_int_var(target=3)
    state = model.target_state

    assert model.eval_dual_bound(state) is None

    model.add_dual_bound(0)

    assert model.eval_dual_bound(state) == 0

    model.add_dual_bound(int_var)

    assert model.eval_dual_bound(state) == 3

    model.add_dual_bound(-1)

    assert model.eval_dual_bound(state) == 3


def test_eval_dual_bound_int_maximize():
    model = dp.Model(float_cost=False, maximize=True)
    var = model.add_int_var(target=0)
    state = model.target_state

    assert model.eval_dual_bound(state) is None

    model.add_dual_bound(2)

    assert model.eval_dual_bound(state) == 2

    model.add_dual_bound(var)

    assert model.eval_dual_bound(state) == 0

    model.add_dual_bound(3)

    assert model.eval_dual_bound(state) == 0


def test_eval_dual_bound_float_minimize():
    model = dp.Model(float_cost=True, maximize=False)
    int_var = model.add_int_var(target=2)
    int_resource_var = model.add_int_resource_var(target=3)
    float_var = model.add_float_var(target=3.5)
    state = model.target_state

    assert model.eval_dual_bound(state) is None

    model.add_dual_bound(0)

    assert model.eval_dual_bound(state) == pytest.approx(0.0)

    model.add_dual_bound(dp.IntExpr(1))

    assert model.eval_dual_bound(state) == pytest.approx(1.0)

    model.add_dual_bound(int_var)

    assert model.eval_dual_bound(state) == pytest.approx(2.0)

    model.add_dual_bound(int_resource_var)

    assert model.eval_dual_bound(state) == pytest.approx(3.0)

    model.add_dual_bound(float_var)

    assert model.eval_dual_bound(state) == pytest.approx(3.5)

    model.add_dual_bound(dp.FloatExpr(2.5))

    assert model.eval_dual_bound(state) == pytest.approx(3.5)


def test_eval_dual_bound_float_maximize():
    model = dp.Model(float_cost=True, maximize=True)
    int_var = model.add_int_var(target=4)
    int_resource_var = model.add_int_resource_var(target=3)
    float_var = model.add_float_var(target=1.5)
    state = model.target_state

    assert model.eval_dual_bound(state) is None

    model.add_dual_bound(6)

    assert model.eval_dual_bound(state) == pytest.approx(6.0)

    model.add_dual_bound(dp.IntExpr(5))

    assert model.eval_dual_bound(state) == pytest.approx(5.0)

    model.add_dual_bound(int_var)

    assert model.eval_dual_bound(state) == pytest.approx(4.0)

    model.add_dual_bound(int_resource_var)

    assert model.eval_dual_bound(state) == pytest.approx(3.0)

    model.add_dual_bound(float_var)

    assert model.eval_dual_bound(state) == pytest.approx(1.5)

    model.add_dual_bound(dp.FloatExpr(2.5))

    assert model.eval_dual_bound(state) == pytest.approx(1.5)


def test_eval_dual_bound_int_panic():
    model = dp.Model()
    obj = model.add_object_type(number=3)
    table = model.add_int_table([0, 1, 2])
    var = model.add_element_var(object_type=obj, target=2)
    state = model.target_state
    model.add_dual_bound(table[var + 1])

    with pytest.raises(BaseException):
        model.eval_dual_bound(state)


def test_eval_dual_bound_float_panic():
    model = dp.Model(float_cost=True)
    obj = model.add_object_type(number=3)
    table = model.add_float_table([0.0, 0.1, 0.2])
    var = model.add_element_var(object_type=obj, target=2)
    state = model.target_state
    model.add_dual_bound(table[var + 1])

    with pytest.raises(BaseException):
        model.eval_dual_bound(state)


class TestValidateForward:
    model = dp.Model()
    var = model.add_int_var(target=5)
    t1 = dp.Transition(
        name="decrement",
        cost=dp.IntExpr.state_cost() + 1,
        effects=[(var, var - 1)],
        preconditions=[var >= 1],
    )
    t2 = dp.Transition(
        name="decrement 2",
        cost=dp.IntExpr.state_cost() + 1,
        effects=[(var, var - 2)],
        preconditions=[var >= 2],
    )
    t3 = dp.Transition(
        name="decrement 3",
        cost=dp.IntExpr.state_cost() + 2,
        effects=[(var, var - 3)],
        preconditions=[var == 4],
    )
    model.add_transition(t1)
    model.add_transition(t2)
    model.add_transition(
        t3,
        forced=True,
    )
    model.add_state_constr(var != 3)
    model.add_base_case([var == 0])
    model.add_dual_bound(0)

    def test_true(self):
        transitions = [self.t1, self.t3, self.t1]
        cost = 4

        assert self.model.validate_forward(transitions, cost, quiet=True)

    def test_false_state_constr(self):
        transitions = [self.t2, self.t2, self.t1]
        cost = 3

        assert not self.model.validate_forward(transitions, cost, quiet=True)

    def test_false_transition(self):
        transitions = [self.t3, self.t1, self.t1]
        cost = 4

        assert not self.model.validate_forward(transitions, cost, quiet=True)

    def test_false_base_case(self):
        transitions = [self.t1, self.t3]
        cost = 4

        assert not self.model.validate_forward(transitions, cost, quiet=True)


element_table_cases = [
    ([1, 2, 3], 0, 1),
    ([[1, 2], [1, 2]], (0, 0), 1),
    ([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], (0, 0, 0), 1),
]


@pytest.mark.parametrize("value, index, expected", element_table_cases)
def test_add_element_table(value, index, expected):
    model = dp.Model()
    table = model.add_element_table(value)
    state = model.target_state

    assert table[index].eval(state, model) == expected


@pytest.mark.parametrize("value, index, expected", element_table_cases)
def test_add_element_table_with_name(value, index, expected):
    model = dp.Model()
    table = model.add_element_table(value, name="table")
    state = model.target_state

    assert table[index].eval(state, model) == expected


@pytest.mark.parametrize("value, _index, _expected", element_table_cases)
def test_add_element_table_with_name_error(value, _index, _expected):
    model = dp.Model()
    model.add_element_table(value, name="table")

    with pytest.raises(RuntimeError):
        model.add_element_table(value, name="table")


element_table_error_cases = [
    (0, TypeError),
    ([-1], TypeError),
    ([1.5], TypeError),
    ([[-1], [0]], TypeError),
    ([[1.5], [0]], TypeError),
    ([[[-1], [0]], [[0], [0]]], TypeError),
    ([[[1.5], [0]], [[0], [0]]], TypeError),
    ([[[[0], [0]], [[0], [0]]]], TypeError),
]


@pytest.mark.parametrize("value, error", element_table_error_cases)
def test_add_element_table_error(value, error):
    model = dp.Model()

    with pytest.raises(error):
        model.add_element_table(value)


def test_add_element_table_dict():
    model = dp.Model()
    table = model.add_element_table({(0, 0, 0, 0): 1}, default=0)
    state = model.target_state

    assert table[0, 0, 0, 0].eval(state, model) == 1


def test_add_element_table_dict_with_name():
    model = dp.Model()
    table = model.add_element_table({(0, 0, 0, 0): 1}, default=0, name="table")
    state = model.target_state

    assert table[0, 0, 0, 0].eval(state, model) == 1


def test_add_element_table_dict_with_name_error():
    model = dp.Model()
    model.add_element_table({(0, 0, 0, 0): 1}, default=0, name="table")

    with pytest.raises(RuntimeError):
        model.add_element_table({(0, 0, 0, 0): 1}, default=0, name="table")


def test_add_element_table_dict_without_default_error():
    model = dp.Model()

    with pytest.raises(TypeError):
        model.add_element_table({(0, 0, 0, 0): 1})


element_table_dict_error_cases = [
    (-1, 1, TypeError),
    (1.5, 1, TypeError),
    (1, -1, OverflowError),
    (1, 1.5, TypeError),
]


@pytest.mark.parametrize("value, default, error", element_table_dict_error_cases)
def test_add_element_table_dict_error(value, default, error):
    model = dp.Model()

    with pytest.raises(error):
        model.add_element_table({(0, 0, 0, 0): value}, default=default)


set_table_cases = [
    ([[0, 1], []], 0, {0, 1}),
    ([{0, 1}, set()], 0, {0, 1}),
    ([[[0, 1], [1, 2]], [[], [1, 2]]], (0, 0), {0, 1}),
    ([[{0, 1}, {1, 2}], [set(), {1, 2}]], (0, 0), {0, 1}),
    (
        [[[[0, 1], [0, 2]], [[], [0, 3]]], [[[2, 3], [1, 3]], [[], [0, 1]]]],
        (0, 0, 0),
        {0, 1},
    ),
    (
        [[[{0, 1}, {0, 2}], [set(), {0, 3}]], [[{2, 3}, {1, 3}], [set(), {0, 1}]]],
        (0, 0, 0),
        {0, 1},
    ),
]


@pytest.mark.parametrize("value, index, expected", set_table_cases)
def test_add_set_table(value, index, expected):
    model = dp.Model()
    obj = model.add_object_type(number=4)
    table = model.add_set_table(value, object_type=obj)
    state = model.target_state

    assert table[index].eval(state, model) == expected


def test_add_set_table_1d_const():
    model = dp.Model()
    obj = model.add_object_type(number=4)
    const = model.create_set_const(object_type=obj, value=[0, 1])
    table = model.add_set_table([const, const])
    state = model.target_state

    assert table[0].eval(state, model) == {0, 1}


def test_add_set_table_2d_const():
    model = dp.Model()
    obj = model.add_object_type(number=4)
    const = model.create_set_const(object_type=obj, value=[0, 1])
    table = model.add_set_table([[const, const], [const, const]])
    state = model.target_state

    assert table[0, 0].eval(state, model) == {0, 1}


def test_add_set_table_3d_const():
    model = dp.Model()
    obj = model.add_object_type(number=4)
    const = model.create_set_const(object_type=obj, value=[0, 1])
    table = model.add_set_table(
        [[[const, const], [const, const]], [[const, const], [const, const]]]
    )
    state = model.target_state

    assert table[0, 0, 0].eval(state, model) == {0, 1}


@pytest.mark.parametrize("value, _index, _expected", set_table_cases)
def test_add_set_table_no_object_error(value, _index, _expected):
    model = dp.Model()

    with pytest.raises(TypeError):
        model.add_set_table(value)


@pytest.mark.parametrize("value, index, expected", set_table_cases)
def test_add_set_table_with_name(value, index, expected):
    model = dp.Model()
    obj = model.add_object_type(number=4)
    table = model.add_set_table(value, name="table", object_type=obj)
    state = model.target_state

    assert table[index].eval(state, model) == expected


@pytest.mark.parametrize("value, _index, _expected", set_table_cases)
def test_add_set_table_with_name_error(value, _index, _expected):
    model = dp.Model()
    obj = model.add_object_type(number=4)
    model.add_set_table(value, name="table", object_type=obj)

    with pytest.raises(RuntimeError):
        model.add_set_table(value, name="table", object_type=obj)


set_table_error_cases = [
    ([0], TypeError),
    ({0}, TypeError),
    ([[-1], [0]], TypeError),
    ([{-1}, {0}], TypeError),
    ([[1.5], [0]], TypeError),
    ([{1.5}, {0}], TypeError),
    ([[[-1], [0]], [[0], [0]]], TypeError),
    ([[{-1}, {0}], [{0}, {0}]], TypeError),
    ([[[1.5], [0]], [[0], [0]]], TypeError),
    ([[{1.5}, {0}], [{0}, {0}]], TypeError),
    ([[[[-1], [0]], [[0], [0]]]], TypeError),
    ([[[{-1}, {0}], [{0}, {0}]]], TypeError),
    ([[[[1.5], [0]], [[0], [0]]]], TypeError),
    ([[[{1.5}, {0}], [{0}, {0}]]], TypeError),
    ([[[[[1], [0]], [[0], [0]]]]], TypeError),
    ([[[[{1}, {0}], [{0}, {0}]]]], TypeError),
]


@pytest.mark.parametrize("value, error", set_table_error_cases)
def test_add_set_table_error(value, error):
    model = dp.Model()

    with pytest.raises(error):
        model.add_set_table(value)


set_table_dict_cases = [
    ([0, 1], [], {0, 1}),
    ({0, 1}, set(), {0, 1}),
]


@pytest.mark.parametrize("value, default, expected", set_table_dict_cases)
def test_add_set_table_dict(value, default, expected):
    model = dp.Model()
    obj = model.add_object_type(number=4)
    table = model.add_set_table({(0, 0, 0, 0): value}, default=default, object_type=obj)
    state = model.target_state

    assert table[0, 0, 0, 0].eval(state, model) == expected


def test_add_set_table_dict_from_const():
    model = dp.Model()
    obj = model.add_object_type(number=4)
    const = model.create_set_const(object_type=obj, value=[0, 1])
    table = model.add_set_table({(0, 0, 0, 0): const}, default=const, object_type=obj)
    state = model.target_state

    assert table[0, 0, 0, 0].eval(state, model) == {0, 1}


@pytest.mark.parametrize("value, default, _expected", set_table_dict_cases)
def test_add_set_table_dict_no_object_error(value, default, _expected):
    model = dp.Model()

    with pytest.raises(TypeError):
        model.add_set_table({(0, 0, 0, 0): value}, default=default)


@pytest.mark.parametrize("value, default, expected", set_table_dict_cases)
def test_add_set_table_dict_with_name(value, default, expected):
    model = dp.Model()
    obj = model.add_object_type(number=4)
    table = model.add_set_table(
        {(0, 0, 0, 0): value}, default=default, object_type=obj, name="table"
    )
    state = model.target_state

    assert table[0, 0, 0, 0].eval(state, model) == expected


def test_add_set_table_dict_with_name_error():
    model = dp.Model()
    obj = model.add_object_type(number=4)
    model.add_set_table(
        {(0, 0, 0, 0): [0, 1]}, default=[], object_type=obj, name="table"
    )

    with pytest.raises(RuntimeError):
        model.add_set_table(
            {(0, 0, 0, 0): [0, 1]},
            default=[],
            object_type=obj,
            name="table",
        )


def test_add_set_table_dict_without_default_error():
    model = dp.Model()
    obj = model.add_object_type(number=4)

    with pytest.raises(TypeError):
        model.add_set_table({(0, 0, 0, 0): [0, 1]}, object_type=obj)


set_table_dict_error_cases = [
    ([-1], [1], TypeError),
    ({-1}, {1}, TypeError),
    ([1.5], [1], TypeError),
    ({1.5}, {1}, TypeError),
    ([1], [-1], TypeError),
    ({1}, {-1}, TypeError),
    ([1], [1.5], TypeError),
]


@pytest.mark.parametrize("value, default, error", set_table_dict_error_cases)
def test_add_set_table_dict_error(value, default, error):
    model = dp.Model()
    obj = model.add_object_type(number=4)

    with pytest.raises(error):
        model.add_set_table({(0, 0, 0, 0): value}, default=default, object_type=obj)


bool_table_cases = [
    ([True, False, False], 0, True),
    ([[True, False], [False, False]], (0, 0), True),
    (
        [[[True, False], [False, False]], [[False, False], [False, False]]],
        (0, 0, 0),
        True,
    ),
]


@pytest.mark.parametrize("value, index, expected", bool_table_cases)
def test_add_bool_table(value, index, expected):
    model = dp.Model()
    table = model.add_bool_table(value)
    state = model.target_state

    assert table[index].eval(state, model) == expected


@pytest.mark.parametrize("value, index, expected", bool_table_cases)
def test_add_bool_table_with_name(value, index, expected):
    model = dp.Model()
    table = model.add_bool_table(value, name="table")
    state = model.target_state

    assert table[index].eval(state, model) == expected


@pytest.mark.parametrize("value, _index, _expected", bool_table_cases)
def test_add_bool_table_with_name_error(value, _index, _expected):
    model = dp.Model()
    model.add_bool_table(value, name="table")

    with pytest.raises(RuntimeError):
        model.add_bool_table(value, name="table")


bool_table_error_cases = [
    (True, TypeError),
    ([-1], TypeError),
    ([1.5], TypeError),
    ([[-1], [0]], TypeError),
    ([[1.5], [0]], TypeError),
    ([[[-1], [0]], [[0], [0]]], TypeError),
    ([[[1.5], [0]], [[0], [0]]], TypeError),
    ([[[[True], [False]], [[True], [False]]]], TypeError),
]


@pytest.mark.parametrize("value, error", bool_table_error_cases)
def test_add_bool_table_error(value, error):
    model = dp.Model()

    with pytest.raises(error):
        model.add_bool_table(value)


def test_add_bool_table_dict():
    model = dp.Model()
    table = model.add_bool_table({(0, 0, 0, 0): True}, default=False)
    state = model.target_state

    assert table[0, 0, 0, 0].eval(state, model)


def test_add_bool_table_dict_with_name():
    model = dp.Model()
    table = model.add_bool_table({(0, 0, 0, 0): True}, default=False, name="table")
    state = model.target_state

    assert table[0, 0, 0, 0].eval(state, model)


def test_add_bool_table_dict_with_name_error():
    model = dp.Model()
    model.add_bool_table({(0, 0, 0, 0): True}, default=False, name="table")

    with pytest.raises(RuntimeError):
        model.add_bool_table({(0, 0, 0, 0): True}, default=False, name="table")


def test_add_bool_table_dict_without_default_error():
    model = dp.Model()

    with pytest.raises(TypeError):
        model.add_bool_table({(0, 0, 0, 0): True})


bool_table_dict_error_cases = [
    (-1, 1, TypeError),
    (1.5, 1, TypeError),
    (1, -1, TypeError),
    (1, 1.5, TypeError),
]


@pytest.mark.parametrize("value, default, error", bool_table_dict_error_cases)
def test_add_bool_table_dict_error(value, default, error):
    model = dp.Model()

    with pytest.raises(error):
        model.add_bool_table({(0, 0, 0, 0): value}, default=default)


int_table_cases = [
    ([1, 2, 3], 0, 1),
    ([-1, 2, 3], 0, -1),
    ([[1, 2], [1, 2]], (0, 0), 1),
    ([[-1, 2], [1, 2]], (0, 0), -1),
    ([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], (0, 0, 0), 1),
    ([[[-1, 2], [3, 4]], [[5, 6], [7, 8]]], (0, 0, 0), -1),
]


@pytest.mark.parametrize("value, index, expected", int_table_cases)
def test_add_int_table(value, index, expected):
    model = dp.Model()
    table = model.add_int_table(value)
    state = model.target_state

    assert table[index].eval(state, model) == expected


@pytest.mark.parametrize("value, index, expected", int_table_cases)
def test_add_int_table_with_name(value, index, expected):
    model = dp.Model()
    table = model.add_int_table(value, name="table")
    state = model.target_state

    assert table[index].eval(state, model) == expected


@pytest.mark.parametrize("value, _index, _expected", int_table_cases)
def test_add_int_table_with_name_error(value, _index, _expected):
    model = dp.Model()
    model.add_int_table(value, name="table")

    with pytest.raises(RuntimeError):
        model.add_int_table(value, name="table")


int_table_error_cases = [
    (0, TypeError),
    ([1.5], TypeError),
    ([[1.5], [0]], TypeError),
    ([[[1.5], [0]], [[0], [0]]], TypeError),
    ([[[[0], [0]], [[0], [0]]]], TypeError),
]


@pytest.mark.parametrize("value, error", int_table_error_cases)
def test_add_int_table_error(value, error):
    model = dp.Model()

    with pytest.raises(error):
        model.add_int_table(value)


def test_add_int_table_dict():
    model = dp.Model()
    table = model.add_int_table({(0, 0, 0, 0): 1}, default=0)
    state = model.target_state

    assert table[0, 0, 0, 0].eval(state, model) == 1


def test_add_int_table_dict_with_name():
    model = dp.Model()
    table = model.add_int_table({(0, 0, 0, 0): 1}, default=0, name="table")
    state = model.target_state

    assert table[0, 0, 0, 0].eval(state, model) == 1


def test_add_int_table_dict_with_name_error():
    model = dp.Model()
    model.add_int_table({(0, 0, 0, 0): 1}, default=0, name="table")

    with pytest.raises(RuntimeError):
        model.add_int_table({(0, 0, 0, 0): 1}, default=0, name="table")


def test_add_int_table_dict_without_default_error():
    model = dp.Model()

    with pytest.raises(TypeError):
        model.add_int_table({(0, 0, 0, 0): 1})


int_table_dict_error_cases = [
    (1.5, 1, TypeError),
    (1, 1.5, TypeError),
]


@pytest.mark.parametrize("value, default, error", int_table_dict_error_cases)
def test_add_int_table_dict_error(value, default, error):
    model = dp.Model()

    with pytest.raises(error):
        model.add_int_table({(0, 0, 0, 0): value}, default=default)


float_table_cases = [
    ([1, 2, 3], 0, pytest.approx(1.0)),
    ([1.5, 2.5, 3.5], 0, pytest.approx(1.5)),
    ([-1, 2, 3], 0, pytest.approx(-1.0)),
    ([[1, 2], [1, 2]], (0, 0), pytest.approx(1.0)),
    ([[1.5, 2.5], [1.5, 2.5]], (0, 0), pytest.approx(1.5)),
    ([[-1, 2], [1, 2]], (0, 0), pytest.approx(-1.0)),
    ([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], (0, 0, 0), pytest.approx(1.0)),
    (
        [[[1.5, 2.5], [3.5, 4.5]], [[5.5, 6.5], [7.5, 8.5]]],
        (0, 0, 0),
        pytest.approx(1.5),
    ),
    ([[[-1, 2], [3, 4]], [[5, 6], [7, 8]]], (0, 0, 0), pytest.approx(-1.0)),
]


@pytest.mark.parametrize("value, index, expected", float_table_cases)
def test_add_float_table(value, index, expected):
    model = dp.Model()
    table = model.add_float_table(value)
    state = model.target_state

    assert table[index].eval(state, model) == expected


@pytest.mark.parametrize("value, index, expected", float_table_cases)
def test_add_float_table_with_name(value, index, expected):
    model = dp.Model()
    table = model.add_float_table(value, name="table")
    state = model.target_state

    assert table[index].eval(state, model) == expected


@pytest.mark.parametrize("value, _index, _expected", float_table_cases)
def test_add_float_table_with_name_error(value, _index, _expected):
    model = dp.Model()
    model.add_float_table(value, name="table")

    with pytest.raises(RuntimeError):
        model.add_float_table(value, name="table")


float_table_error_cases = [
    (0, TypeError),
    ([[[[0], [0]], [[0], [0]]]], TypeError),
]


@pytest.mark.parametrize("value, error", float_table_error_cases)
def test_add_float_table_error(value, error):
    model = dp.Model()

    with pytest.raises(error):
        model.add_float_table(value)


float_table_dict_cases = [
    (-1, 1, pytest.approx(-1.0)),
    (1.5, 1, pytest.approx(1.5)),
    (1, -1, pytest.approx(1.0)),
]


@pytest.mark.parametrize("value, default, expected", float_table_dict_cases)
def test_add_float_table_dict(value, default, expected):
    model = dp.Model()
    table = model.add_float_table({(0, 0, 0, 0): value}, default=default)
    state = model.target_state

    assert table[0, 0, 0, 0].eval(state, model) == expected


@pytest.mark.parametrize("value, default, expected", float_table_dict_cases)
def test_add_float_table_dict_with_name(value, default, expected):
    model = dp.Model()
    table = model.add_float_table({(0, 0, 0, 0): value}, default=default, name="table")
    state = model.target_state

    assert table[0, 0, 0, 0].eval(state, model) == expected


def test_add_float_table_dict_with_name_error():
    model = dp.Model()
    model.add_float_table({(0, 0, 0, 0): 1}, default=0, name="table")

    with pytest.raises(RuntimeError):
        model.add_float_table({(0, 0, 0, 0): 1}, default=0, name="table")


def test_add_float_table_dict_without_default_error():
    model = dp.Model()

    with pytest.raises(TypeError):
        model.add_float_table({(0, 0, 0, 0): 1})


float_table_dict_error_cases = [
    ((), 1, TypeError),
    (1, (), TypeError),
]


@pytest.mark.parametrize("value, default, error", float_table_dict_error_cases)
def test_add_float_table_dict_error(value, default, error):
    model = dp.Model()

    with pytest.raises(error):
        model.add_float_table({(0, 0, 0, 0): value}, default=default)
