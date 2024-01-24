import pytest

import didppy as dp


class TestTransition:
    model = dp.Model()
    obj = model.add_object_type(number=4)
    element_var = model.add_element_var(object_type=obj, target=1)
    element_resource_var = model.add_element_resource_var(
        object_type=obj, target=2, less_is_better=True
    )
    set_var1 = model.add_set_var(object_type=obj, target=[0, 1])
    set_var2 = model.add_set_var(object_type=obj, target=[1, 2])
    set_var3 = model.add_set_var(object_type=obj, target=[0])
    set_var4 = model.add_set_var(object_type=obj, target=[])
    int_var = model.add_int_var(target=3)
    int_resource_var = model.add_int_resource_var(target=4, less_is_better=True)
    float_var = model.add_float_var(target=0.5)
    float_resource_var = model.add_float_resource_var(target=0.6, less_is_better=True)
    set_const1 = model.create_set_const(object_type=obj, value=[1, 2])
    set_const2 = model.create_set_const(object_type=obj, value=[2])
    set_const3 = model.create_set_const(object_type=obj, value=[])
    set_const4 = model.create_set_const(object_type=obj, value=[1])

    def test_default(self):
        state = self.model.target_state
        transition = dp.Transition(name="test")

        assert transition.name == "test"
        assert transition.cost.eval_cost(0, state, self.model) == 0
        assert transition.preconditions == []
        assert transition[self.element_var].eval(state, self.model) == 1
        assert transition[self.element_resource_var].eval(state, self.model) == 2
        assert transition[self.set_var1].eval(state, self.model) == {0, 1}
        assert transition[self.set_var2].eval(state, self.model) == {1, 2}
        assert transition[self.set_var3].eval(state, self.model) == {0}
        assert transition[self.set_var4].eval(state, self.model) == set()
        assert transition[self.int_var].eval(state, self.model) == 3
        assert transition[self.int_resource_var].eval(state, self.model) == 4
        assert transition[self.float_var].eval(state, self.model) == 0.5
        assert transition[self.float_resource_var].eval(state, self.model) == 0.6

    init_cases = [
        (
            1,
            [
                (element_var, 2),
                (element_resource_var, 3),
                (set_var3, set_const3),
                (set_var1, set_const1),
                (set_var2, set_const2),
                (set_var4, set_const4),
                (int_var, 4),
                (int_resource_var, 5),
                (float_var, 0.6),
                (float_resource_var, 0.7),
            ],
            [set_var1.contains(0), int_var < 0],
            1,
            [
                2,
                3,
                {1, 2},
                {2},
                set(),
                {1},
                4,
                5,
                pytest.approx(0.6),
                pytest.approx(0.7),
            ],
            2,
        ),
        (
            dp.IntExpr.state_cost() + 1,
            [
                (element_var, dp.ElementExpr(2)),
                (element_resource_var, dp.ElementExpr(3)),
                (set_var1, dp.SetExpr(set_const1)),
                (set_var4, dp.SetExpr(set_const4)),
                (set_var3, dp.SetExpr(set_const3)),
                (int_var, dp.IntExpr(4)),
                (int_resource_var, dp.IntExpr(5)),
                (float_var, dp.FloatExpr(0.6)),
                (float_resource_var, dp.FloatExpr(0.7)),
            ],
            [set_var1.contains(0), int_var < 0],
            1,
            [
                2,
                3,
                {1, 2},
                {1, 2},
                set(),
                {1},
                4,
                5,
                pytest.approx(0.6),
                pytest.approx(0.7),
            ],
            2,
        ),
        (
            int_var,
            [
                (element_var, element_var),
                (element_resource_var, element_resource_var),
                (set_var1, set_var2),
                (set_var2, set_var1),
                (set_var3, set_var4),
                (set_var4, set_var3),
                (int_var, int_var),
                (int_resource_var, int_resource_var),
                (float_var, float_var),
                (float_resource_var, float_resource_var),
            ],
            [set_var1.contains(0), int_var < 0],
            3,
            [
                1,
                2,
                {1, 2},
                {0, 1},
                set(),
                {0},
                3,
                4,
                pytest.approx(0.5),
                pytest.approx(0.6),
            ],
            2,
        ),
        (
            int_resource_var,
            [
                (element_var, element_resource_var),
                (element_resource_var, element_var),
                (set_var1, set_var1),
                (int_var, int_resource_var),
                (int_resource_var, int_var),
                (float_var, float_resource_var),
                (float_resource_var, float_var),
            ],
            [set_var1.contains(0), int_var < 0],
            4,
            [
                2,
                1,
                {0, 1},
                {1, 2},
                {0},
                set(),
                4,
                3,
                pytest.approx(0.6),
                pytest.approx(0.5),
            ],
            2,
        ),
        (
            dp.FloatExpr.state_cost() + 1.5,
            [
                (element_var, 2),
                (element_resource_var, 3),
                (set_var1, set_const1),
                (int_var, 4),
                (int_resource_var, 5),
                (float_var, 6),
                (float_resource_var, 7),
            ],
            [set_var1.contains(0), int_var < 0],
            1.5,
            [
                2,
                3,
                {1, 2},
                {1, 2},
                {0},
                set(),
                4,
                5,
                pytest.approx(6.0),
                pytest.approx(7.0),
            ],
            2,
        ),
        (
            float_var,
            [
                (element_var, 2),
                (element_resource_var, 3),
                (set_var1, set_const1),
                (int_var, 4),
                (int_resource_var, 5),
                (float_var, int_var),
                (float_resource_var, int_resource_var),
            ],
            [set_var1.contains(0), int_var < 0],
            pytest.approx(0.5),
            [
                2,
                3,
                {1, 2},
                {1, 2},
                {0},
                set(),
                4,
                5,
                pytest.approx(3.0),
                pytest.approx(4.0),
            ],
            2,
        ),
        (
            float_resource_var,
            [
                (element_var, 2),
                (element_resource_var, 3),
                (set_var1, set_const1),
                (int_var, 4),
                (int_resource_var, 5),
                (float_var, int_resource_var),
                (float_resource_var, int_var),
            ],
            [set_var1.contains(0), int_var < 0],
            pytest.approx(0.6),
            [
                2,
                3,
                {1, 2},
                {1, 2},
                {0},
                set(),
                4,
                5,
                pytest.approx(4.0),
                pytest.approx(3.0),
            ],
            2,
        ),
    ]

    @pytest.mark.parametrize(
        "cost, effects, preconditions, expected_cost, expected_effects, expected_n_preconditions",  # noqa: E501
        init_cases,
    )
    def test_init(
        self,
        cost,
        effects,
        preconditions,
        expected_cost,
        expected_effects,
        expected_n_preconditions,
    ):
        state = self.model.target_state
        transition = dp.Transition(
            name="test", cost=cost, effects=effects, preconditions=preconditions
        )

        assert transition.name == "test"
        assert transition.cost.eval_cost(0, state, self.model) == expected_cost
        assert (
            transition[self.element_var].eval(state, self.model) == expected_effects[0]
        )
        assert (
            transition[self.element_resource_var].eval(state, self.model)
            == expected_effects[1]
        )
        assert transition[self.set_var1].eval(state, self.model) == expected_effects[2]
        assert transition[self.set_var2].eval(state, self.model) == expected_effects[3]
        assert transition[self.set_var3].eval(state, self.model) == expected_effects[4]
        assert transition[self.set_var4].eval(state, self.model) == expected_effects[5]
        assert transition[self.int_var].eval(state, self.model) == expected_effects[6]
        assert (
            transition[self.int_resource_var].eval(state, self.model)
            == expected_effects[7]
        )
        assert transition[self.float_var].eval(state, self.model) == expected_effects[8]
        assert (
            transition[self.float_resource_var].eval(state, self.model)
            == expected_effects[9]
        )
        assert len(transition.preconditions) == expected_n_preconditions

    init_error_cases = [
        (dp.ElementExpr(0), None, None),
        (0, [int_var > 0], None),
        (0, None, [(int_var, 0)]),
    ]

    @pytest.mark.parametrize(
        "cost, effects, preconditions",
        init_error_cases,
    )
    def test_init_error(
        self,
        cost,
        effects,
        preconditions,
    ):
        with pytest.raises(Exception):
            dp.Transition(
                name="test", cost=cost, effects=effects, preconditions=preconditions
            )

    def test_set_name(self):
        transition = dp.Transition(name="test")
        transition.name = "changed"

        assert transition.name == "changed"

    def test_set_name_error(self):
        transition = dp.Transition(name="test")

        with pytest.raises(TypeError):
            transition.name = 0

    set_cost_cases = [
        (1, 1),
        (dp.IntExpr.state_cost() + 1, 1),
        (int_var, 3),
        (int_resource_var, 4),
        (1.5, pytest.approx(1.5)),
        (dp.FloatExpr.state_cost() + 1.5, pytest.approx(1.5)),
        (float_var, pytest.approx(0.5)),
        (float_resource_var, pytest.approx(0.6)),
    ]

    @pytest.mark.parametrize("cost, expected", set_cost_cases)
    def test_set_cost(self, cost, expected):
        state = self.model.target_state
        transition = dp.Transition(name="test")
        transition.cost = cost

        assert transition.cost.eval_cost(0, state, self.model) == expected

    def test_set_cost_error(self):
        transition = dp.Transition(name="test")

        with pytest.raises(TypeError):
            transition.cost = dp.ElementExpr(0)

    def test_add_precondition(self):
        transition = dp.Transition(name="test")
        transition.add_precondition(self.int_var > 0)
        transition.add_precondition(self.float_var > 0)

        assert len(transition.preconditions) == 2

    def test_add_precondition_error(self):
        transition = dp.Transition(name="test")

        with pytest.raises(TypeError):
            transition.add_precondition(1 > 0)

    set_effect_cases = [
        (element_var, 2, 2),
        (element_var, dp.ElementExpr(2), 2),
        (element_var, element_var, 1),
        (element_var, element_resource_var, 2),
        (element_resource_var, 3, 3),
        (element_resource_var, dp.ElementExpr(3), 3),
        (element_resource_var, element_var, 1),
        (element_resource_var, element_resource_var, 2),
        (set_var1, set_const1, {1, 2}),
        (set_var1, set_var1, {0, 1}),
        (int_var, 4, 4),
        (int_var, dp.IntExpr(4), 4),
        (int_var, int_var, 3),
        (int_var, int_resource_var, 4),
        (int_resource_var, 5, 5),
        (int_resource_var, dp.IntExpr(5), 5),
        (int_resource_var, int_var, 3),
        (int_resource_var, int_resource_var, 4),
        (float_var, 0.6, pytest.approx(0.6)),
        (float_var, dp.FloatExpr(0.6), pytest.approx(0.6)),
        (float_var, float_var, pytest.approx(0.5)),
        (float_var, float_resource_var, pytest.approx(0.6)),
        (float_var, 4, 4),
        (float_var, dp.IntExpr(4), 4),
        (float_var, int_var, 3),
        (float_var, int_resource_var, 4),
        (float_resource_var, 0.7, pytest.approx(0.7)),
        (float_resource_var, dp.FloatExpr(0.7), pytest.approx(0.7)),
        (float_resource_var, float_var, pytest.approx(0.5)),
        (float_resource_var, float_resource_var, pytest.approx(0.6)),
        (float_resource_var, 4, 4),
        (float_resource_var, dp.IntExpr(4), 4),
        (float_resource_var, int_var, 3),
        (float_resource_var, int_resource_var, 4),
    ]

    @pytest.mark.parametrize("var, effect, expected", set_effect_cases)
    def test_set_effect(self, var, effect, expected):
        state = self.model.target_state
        transition = dp.Transition(name="test")
        transition[var] = effect

        assert transition[var].eval(state, self.model) == expected

    def test_set_multiple_set_effects(self):
        state = self.model.target_state
        transition = dp.Transition(name="test")
        transition[self.set_var3] = self.set_var1
        transition[self.set_var1] = self.set_var2
        transition[self.set_var2] = self.set_var4
        transition[self.set_var4] = self.set_var3
        state = transition.apply(state, self.model)

        assert state[self.set_var1] == {1, 2}
        assert state[self.set_var2] == set()
        assert state[self.set_var3] == {0, 1}
        assert state[self.set_var4] == {0}

    @pytest.mark.parametrize("var, effect, expected", set_effect_cases)
    def test_overwrite_effect(self, var, effect, expected):
        state = self.model.target_state
        transition = dp.Transition(
            name="test",
            effects=[
                (self.element_var, 0),
                (self.element_resource_var, 0),
                (self.set_var1, self.set_const1),
                (self.int_var, 0),
                (self.int_resource_var, 0),
                (self.float_var, 0),
                (self.float_resource_var, 0),
            ],
        )
        transition[var] = effect

        assert transition[var].eval(state, self.model) == expected

    def test_overwrite_multiple_set_effects(self):
        state = self.model.target_state
        transition = dp.Transition(
            name="test",
            effects=[
                (self.element_var, 0),
                (self.element_resource_var, 0),
                (self.set_var1, self.set_const1),
                (self.int_var, 0),
                (self.int_resource_var, 0),
                (self.float_var, 0),
                (self.float_resource_var, 0),
            ],
        )
        transition[self.set_var2] = self.set_var4
        transition[self.set_var1] = self.set_var2
        transition[self.set_var3] = self.set_var1

        assert transition[self.set_var2].eval(state, self.model) == set()
        assert transition[self.set_var1].eval(state, self.model) == {1, 2}
        assert transition[self.set_var3].eval(state, self.model) == {0, 1}

    set_effect_error_cases = [
        (element_var, -1),
        (element_resource_var, -1),
        (set_var1, {1, 2}),
        (int_var, 1.5),
        (int_resource_var, 1.5),
        (0, 0),
    ]

    @pytest.mark.parametrize("var, effect", set_effect_error_cases)
    def test_set_effect_error(self, var, effect):
        transition = dp.Transition(name="test")

        with pytest.raises(Exception):
            transition[var] = effect

    @pytest.mark.parametrize("var, effect, expected", set_effect_cases)
    def test_add_effect(self, var, effect, expected):
        state = self.model.target_state
        transition = dp.Transition(name="test")
        transition.add_effect(var, effect)

        assert transition[var].eval(state, self.model) == expected

    def test_add_multiple_set_effects(self):
        state = self.model.target_state
        transition = dp.Transition(name="test")
        transition.add_effect(self.set_var2, self.set_var4)
        transition.add_effect(self.set_var1, self.set_var2)
        transition.add_effect(self.set_var3, self.set_var1)
        transition.add_effect(self.set_var4, self.set_var3)
        state = transition.apply(state, self.model)

        assert state[self.set_var1] == {1, 2}
        assert state[self.set_var2] == set()
        assert state[self.set_var3] == {0, 1}
        assert state[self.set_var4] == {0}

    @pytest.mark.parametrize("var, effect, _", set_effect_cases)
    def test_add_effect_overwrite_error(self, var, effect, _):
        transition = dp.Transition(
            name="test",
            effects=[
                (self.element_var, 0),
                (self.element_resource_var, 0),
                (self.set_var1, self.set_const1),
                (self.int_var, 0),
                (self.int_resource_var, 0),
                (self.float_var, 0),
                (self.float_resource_var, 0),
            ],
        )

        with pytest.raises(RuntimeError):
            transition.add_effect(var, effect)

    @pytest.mark.parametrize("var, effect", set_effect_error_cases)
    def test_add_effect_error(self, var, effect):
        transition = dp.Transition(name="test")

        with pytest.raises(Exception):
            transition.add_effect(var, effect)

    def test_is_applicable(self):
        state = self.model.target_state
        transition = dp.Transition(
            name="test",
            preconditions=[
                self.element_var >= 1,
                self.element_resource_var >= 2,
                self.set_var1.contains(0),
                self.int_var >= 3,
                self.int_resource_var >= 0.4,
                self.float_var > 0.4,
                self.float_resource_var > 0.5,
            ],
        )

        assert transition.is_applicable(state, self.model)

        transition.add_precondition(self.set_var1.contains(1))

        assert transition.is_applicable(state, self.model)

        transition.add_precondition(self.int_var < 3)

        assert not transition.is_applicable(state, self.model)

    def test_is_applicable_error(self):
        state = self.model.target_state
        transition = dp.Transition(
            name="test", preconditions=[dp.IntExpr.state_cost() > 0]
        )

        with pytest.raises(BaseException):
            transition.is_applicable(state, self.model)

    apply_cases = [
        (
            [
                (element_var, 2),
                (element_resource_var, 3),
                (set_var3, set_const3),
                (set_var1, set_const1),
                (set_var2, set_const2),
                (set_var4, set_const4),
                (int_var, 4),
                (int_resource_var, 5),
                (float_var, 0.6),
                (float_resource_var, 0.7),
            ],
            [
                2,
                3,
                {1, 2},
                {2},
                set(),
                {1},
                4,
                5,
                pytest.approx(0.6),
                pytest.approx(0.7),
            ],
        ),
        (
            [
                (element_var, 2),
                (set_var1, set_const1),
                (int_var, 4),
                (float_var, 0.6),
            ],
            [
                2,
                2,
                {1, 2},
                {1, 2},
                {0},
                set(),
                4,
                4,
                pytest.approx(0.6),
                pytest.approx(0.6),
            ],
        ),
    ]

    @pytest.mark.parametrize("effects, expected", apply_cases)
    def test_apply(self, effects, expected):
        state = self.model.target_state
        transition = dp.Transition(name="test", effects=effects)
        state = transition.apply(state, self.model)

        assert state[self.element_var] == expected[0]
        assert state[self.element_resource_var] == expected[1]
        assert state[self.set_var1] == expected[2]
        assert state[self.set_var2] == expected[3]
        assert state[self.set_var3] == expected[4]
        assert state[self.set_var4] == expected[5]
        assert state[self.int_var] == expected[6]
        assert state[self.int_resource_var] == expected[7]
        assert state[self.float_var] == expected[8]
        assert state[self.float_resource_var] == expected[9]

    def test_apply_error(self):
        state = self.model.target_state
        transition = dp.Transition(
            name="test", effects=[(self.int_var, dp.IntExpr.state_cost())]
        )

        with pytest.raises(BaseException):
            transition.apply(state, self.model)


def test_eval_int_cost():
    model = dp.Model()
    state = model.target_state
    transition = dp.Transition(name="test", cost=dp.IntExpr.state_cost() + 1)

    assert transition.eval_cost(1, state, model) == 2


def test_float_eval_float_cost():
    model = dp.Model(float_cost=True)
    state = model.target_state
    transition = dp.Transition(name="test", cost=dp.FloatExpr.state_cost() + 1.5)

    assert transition.eval_cost(1.2, state, model) == pytest.approx(2.7)


def test_int_eval_float_cost_error():
    model = dp.Model()
    state = model.target_state
    transition = dp.Transition(name="test", cost=dp.IntExpr.state_cost() + 1)

    with pytest.raises(TypeError):
        transition.eval_cost(1.2, state, model)
