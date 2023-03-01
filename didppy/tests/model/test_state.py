import didppy as dp
import pytest


class TestStateElementVariable:
    model = dp.Model()
    obj = model.add_object_type(number=4)
    var1 = model.add_element_var(object_type=obj, target=2)
    var2 = model.add_element_var(object_type=obj, target=1)

    def test_get(self):
        state = self.model.target_state

        assert state[self.var1] == 2
        assert state[self.var2] == 1

    def test_get_error(self):
        model = dp.Model()
        state = model.target_state

        with pytest.raises(BaseException):
            state[self.var1]

    def test_set(self):
        state = self.model.target_state
        state[self.var1] = 3

        assert state[self.var1] == 3
        assert state[self.var2] == 1

    def test_set_error(self):
        state = self.model.target_state

        with pytest.raises(OverflowError):
            state[self.var1] = -1


class TestStateElementResourceVariable:
    model = dp.Model()
    obj = model.add_object_type(number=4)
    var1 = model.add_element_resource_var(
        object_type=obj, target=2, less_is_better=True
    )
    var2 = model.add_element_resource_var(
        object_type=obj, target=1, less_is_better=False
    )

    def test_get(self):
        state = self.model.target_state

        assert state[self.var1] == 2
        assert state[self.var2] == 1

    def test_get_error(self):
        model = dp.Model()
        state = model.target_state

        with pytest.raises(BaseException):
            state[self.var1]

    def test_set(self):
        state = self.model.target_state
        state[self.var1] = 3

        assert state[self.var1] == 3
        assert state[self.var2] == 1

    def test_set_error(self):
        state = self.model.target_state

        with pytest.raises(OverflowError):
            state[self.var1] = -1


class TestStateSetVariable:
    model = dp.Model()
    obj = model.add_object_type(number=4)
    var1 = model.add_set_var(object_type=obj, target=[0, 1])
    var2 = model.add_set_var(object_type=obj, target=[1, 2])

    def test_get(self):
        state = self.model.target_state

        assert state[self.var1] == {0, 1}
        assert state[self.var2] == {1, 2}

    def test_get_error(self):
        model = dp.Model()
        state = model.target_state

        with pytest.raises(BaseException):
            state[self.var1]

    def test_set(self):
        state = self.model.target_state
        state[self.var1] = self.model.create_set_const(
            object_type=self.obj, value=[0, 2]
        )

        assert state[self.var1] == {0, 2}
        assert state[self.var2] == {1, 2}


class TestStateIntVariable:
    model = dp.Model()
    var1 = model.add_int_var(target=2)
    var2 = model.add_int_var(target=1)

    def test_get(self):
        state = self.model.target_state

        assert state[self.var1] == 2
        assert state[self.var2] == 1

    def test_get_error(self):
        model = dp.Model()
        state = model.target_state

        with pytest.raises(BaseException):
            state[self.var1]

    set_cases = [3, -3]

    @pytest.mark.parametrize("value", set_cases)
    def test_set(self, value):
        state = self.model.target_state
        state[self.var1] = value

        assert state[self.var1] == value
        assert state[self.var2] == 1

    def test_set_error(self):
        state = self.model.target_state

        with pytest.raises(TypeError):
            state[self.var1] = 1.5


class TestStateIntResourceVariable:
    model = dp.Model()
    var1 = model.add_int_resource_var(target=2, less_is_better=True)
    var2 = model.add_int_resource_var(target=1, less_is_better=False)

    def test_get(self):
        state = self.model.target_state

        assert state[self.var1] == 2
        assert state[self.var2] == 1

    def test_get_error(self):
        model = dp.Model()
        state = model.target_state

        with pytest.raises(BaseException):
            state[self.var1]

    set_cases = [3, -3]

    @pytest.mark.parametrize("value", set_cases)
    def test_set(self, value):
        state = self.model.target_state
        state[self.var1] = value

        assert state[self.var1] == value
        assert state[self.var2] == 1

    def test_set_error(self):
        state = self.model.target_state

        with pytest.raises(TypeError):
            state[self.var1] = 1.5


class TestStateFloatVariable:
    model = dp.Model()
    var1 = model.add_float_var(target=2.5)
    var2 = model.add_float_var(target=1.5)

    def test_get(self):
        state = self.model.target_state

        assert state[self.var1] == pytest.approx(2.5)
        assert state[self.var2] == pytest.approx(1.5)

    def test_get_error(self):
        model = dp.Model()
        state = model.target_state

        with pytest.raises(BaseException):
            state[self.var1]

    set_cases = [
        (3.5, pytest.approx(3.5)),
        (-3.5, pytest.approx(-3.5)),
        (3, pytest.approx(3.0)),
        (-3, pytest.approx(-3.0)),
    ]

    @pytest.mark.parametrize("value, expected", set_cases)
    def test_set(self, value, expected):
        state = self.model.target_state
        state[self.var1] = value

        assert state[self.var1] == expected
        assert state[self.var2] == pytest.approx(1.5)


class TestStateFloatResourceVariable:
    model = dp.Model()
    var1 = model.add_float_resource_var(target=2.5, less_is_better=True)
    var2 = model.add_float_resource_var(target=1.5, less_is_better=False)

    def test_get(self):
        state = self.model.target_state

        assert state[self.var1] == pytest.approx(2.5)
        assert state[self.var2] == pytest.approx(1.5)

    def test_get_error(self):
        model = dp.Model()
        state = model.target_state

        with pytest.raises(BaseException):
            state[self.var1]

    set_cases = [
        (3.5, pytest.approx(3.5)),
        (-3.5, pytest.approx(-3.5)),
        (3, pytest.approx(3.0)),
        (-3, pytest.approx(-3.0)),
    ]

    @pytest.mark.parametrize("value, expected", set_cases)
    def test_set(self, value, expected):
        state = self.model.target_state
        state[self.var1] = value

        assert state[self.var1] == expected
        assert state[self.var2] == pytest.approx(1.5)


get_error_cases = [
    0,
    0.5,
    dp.ElementExpr(0),
    dp.IntExpr(0),
    dp.FloatExpr(0.5),
]


@pytest.mark.parametrize("value", get_error_cases)
def test_get_variable_error(value):
    model = dp.Model()
    state = model.target_state

    with pytest.raises(TypeError):
        state[value]
