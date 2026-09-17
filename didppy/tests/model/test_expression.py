from typing import ClassVar

import didppy as dp
import pytest


def test_element_expr():
    model = dp.Model()
    state = model.target_state

    assert dp.ElementExpr(3).eval(state, model) == 3


def test_element_expr_raise():
    with pytest.raises(OverflowError):
        dp.ElementExpr(-1)


def test_element_expr_bool_raise():
    with pytest.raises(Exception, match="cannot be converted to bool"):
        bool(dp.ElementExpr(3))

    with pytest.raises(Exception, match="cannot be converted to bool"):
        if dp.ElementExpr(1):
            pass


def test_element_expr_eval_raise():
    model = dp.Model()
    state = model.target_state

    with pytest.raises(BaseException, match="called `Option::unwrap"):
        (dp.IntExpr.state_cost() > 0).if_then_else(
            dp.ElementExpr(0), dp.ElementExpr(1)
        ).eval(state, model)


def test_element_var():
    model = dp.Model()
    obj = model.add_object_type(number=4)
    var = model.add_element_var(object_type=obj, target=3)
    state = model.target_state

    assert state[var] == 3


def test_element_var_bool_raise():
    model = dp.Model()
    obj = model.add_object_type(number=4)
    var = model.add_element_var(object_type=obj, target=3)

    with pytest.raises(Exception, match="cannot be converted to bool"):
        bool(var)

    with pytest.raises(Exception, match="cannot be converted to bool"):
        if var:
            pass


def test_element_resource_var():
    model = dp.Model()
    obj = model.add_object_type(number=4)
    var = model.add_element_resource_var(object_type=obj, target=3, less_is_better=True)
    state = model.target_state

    assert state[var] == 3


def test_element_resource_var_bool_raise():
    model = dp.Model()
    obj = model.add_object_type(number=4)
    var = model.add_element_resource_var(object_type=obj, target=3)

    with pytest.raises(Exception, match="cannot be converted to bool"):
        bool(var)

    with pytest.raises(Exception, match="cannot be converted to bool"):
        if var:
            pass


class TestElementBinaryOperator:
    model = dp.Model()
    obj = model.add_object_type(number=4)

    three_expr = dp.ElementExpr(3)
    three_var = model.add_element_var(object_type=obj, target=3)
    three_resource_var = model.add_element_resource_var(
        object_type=obj, target=3, less_is_better=True
    )

    two_expr = dp.ElementExpr(2)
    two_var = model.add_element_var(object_type=obj, target=2)
    two_resource_var = model.add_element_resource_var(
        object_type=obj, target=2, less_is_better=False
    )

    state = model.target_state

    add_cases: ClassVar = []

    for i, three in enumerate([three_expr, three_var, three_resource_var, 3]):
        for j, two in enumerate([two_expr, two_var, two_resource_var, 2]):
            if i < 3 or j < 3:
                add_cases.append((three, two, 5))

    @pytest.mark.parametrize("lhs, rhs, expected", add_cases)
    def test_add(self, lhs, rhs, expected):
        assert (lhs + rhs).eval(self.state, self.model) == expected

    sub_cases: ClassVar = []

    for i, three in enumerate([three_expr, three_var, three_resource_var, 3]):
        for j, two in enumerate([two_expr, two_var, two_resource_var, 2]):
            if i < 3 or j < 3:
                sub_cases.append((three, two, 1))

    @pytest.mark.parametrize("lhs, rhs, expected", sub_cases)
    def test_sub(self, lhs, rhs, expected):
        assert (lhs - rhs).eval(self.state, self.model) == expected

    mul_cases: ClassVar = []

    for i, three in enumerate([three_expr, three_var, three_resource_var, 3]):
        for j, two in enumerate([two_expr, two_var, two_resource_var, 2]):
            if i < 3 or j < 3:
                mul_cases.append((three, two, 6))

    @pytest.mark.parametrize("lhs, rhs, expected", mul_cases)
    def test_mul(self, lhs, rhs, expected):
        assert (lhs * rhs).eval(self.state, self.model) == expected

    truediv_cases: ClassVar = []

    for i, three in enumerate([three_expr, three_var, three_resource_var, 3]):
        for j, two in enumerate([two_expr, two_var, two_resource_var, 2]):
            if i < 3 or j < 3:
                truediv_cases.append((three, two, 1))

    @pytest.mark.parametrize("lhs, rhs, expected", truediv_cases)
    def test_truediv(self, lhs, rhs, expected):
        assert (lhs / rhs).eval(self.state, self.model) == expected

    floordiv_cases: ClassVar = []

    for i, three in enumerate([three_expr, three_var, three_resource_var, 3]):
        for j, two in enumerate([two_expr, two_var, two_resource_var, 2]):
            if i < 3 or j < 3:
                floordiv_cases.append((three, two, 1))

    @pytest.mark.parametrize("lhs, rhs, expected", floordiv_cases)
    def test_floordiv(self, lhs, rhs, expected):
        assert (lhs // rhs).eval(self.state, self.model) == expected

    mod_cases: ClassVar = []

    for i, three in enumerate([three_expr, three_var, three_resource_var, 3]):
        for j, two in enumerate([two_expr, two_var, two_resource_var, 2]):
            if i < 3 or j < 3:
                mod_cases.append((three, two, 1))

    @pytest.mark.parametrize("lhs, rhs, expected", mod_cases)
    def test_mod(self, lhs, rhs, expected):
        assert (lhs % rhs).eval(self.state, self.model) == expected

    lt_cases: ClassVar = [
        (value, value, False) for value in [three_expr, three_var, three_resource_var]
    ]

    for i, three in enumerate([three_expr, three_var, three_resource_var, 3]):
        for j, two in enumerate([two_expr, two_var, two_resource_var, 2]):
            if i < 3 or j < 3:
                lt_cases.append((three, two, False))
                lt_cases.append((two, three, True))

    @pytest.mark.parametrize("lhs, rhs, expected", lt_cases)
    def test_lt(self, lhs, rhs, expected):
        assert (lhs < rhs).eval(self.state, self.model) == expected

    le_cases: ClassVar = [
        (value, value, True) for value in [three_expr, three_var, three_resource_var]
    ]

    for i, three in enumerate([three_expr, three_var, three_resource_var, 3]):
        for j, two in enumerate([two_expr, two_var, two_resource_var, 2]):
            if i < 3 or j < 3:
                le_cases.append((three, two, False))
                le_cases.append((two, three, True))

    @pytest.mark.parametrize("lhs, rhs, expected", le_cases)
    def test_le(self, lhs, rhs, expected):
        assert (lhs <= rhs).eval(self.state, self.model) == expected

    eq_cases: ClassVar = [
        (value, value, True) for value in [three_expr, three_var, three_resource_var]
    ]

    for i, three in enumerate([three_expr, three_var, three_resource_var, 3]):
        for j, two in enumerate([two_expr, two_var, two_resource_var, 2]):
            if i < 3 or j < 3:
                eq_cases.append((three, two, False))

    @pytest.mark.parametrize("lhs, rhs, expected", eq_cases)
    def test_eq(self, lhs, rhs, expected):
        assert (lhs == rhs).eval(self.state, self.model) == expected

    ne_cases: ClassVar = [
        (value, value, False) for value in [three_expr, three_var, three_resource_var]
    ]

    for i, three in enumerate([three_expr, three_var, three_resource_var, 3]):
        for j, two in enumerate([two_expr, two_var, two_resource_var, 2]):
            if i < 3 or j < 3:
                ne_cases.append((three, two, True))

    @pytest.mark.parametrize("lhs, rhs, expected", ne_cases)
    def test_ne(self, lhs, rhs, expected):
        assert (lhs != rhs).eval(self.state, self.model) == expected

    ge_cases: ClassVar = [
        (value, value, True) for value in [three_expr, three_var, three_resource_var]
    ]

    for i, three in enumerate([three_expr, three_var, three_resource_var, 3]):
        for j, two in enumerate([two_expr, two_var, two_resource_var, 2]):
            if i < 3 or j < 3:
                ge_cases.append((three, two, True))
                ge_cases.append((two, three, False))

    @pytest.mark.parametrize("lhs, rhs, expected", ge_cases)
    def test_ge(self, lhs, rhs, expected):
        assert (lhs >= rhs).eval(self.state, self.model) == expected

    gt_cases: ClassVar = [
        (value, value, False) for value in [three_expr, three_var, three_resource_var]
    ]

    for i, three in enumerate([three_expr, three_var, three_resource_var, 3]):
        for j, two in enumerate([two_expr, two_var, two_resource_var, 2]):
            if i < 3 or j < 3:
                gt_cases.append((three, two, True))
                gt_cases.append((two, three, False))

    @pytest.mark.parametrize("lhs, rhs, expected", gt_cases)
    def test_gt(self, lhs, rhs, expected):
        assert (lhs > rhs).eval(self.state, self.model) == expected


set_eval_cases = [
    ([], set()),
    ([0, 1], {0, 1}),
    ([2, 3], {2, 3}),
    (set(), set()),
    ({0, 1}, {0, 1}),
    ({2, 3}, {2, 3}),
]


@pytest.mark.parametrize("value, expected", set_eval_cases)
def test_set_expr(value, expected):
    model = dp.Model()
    obj = model.add_object_type(number=4)
    const = model.create_set_const(object_type=obj, value=value)
    expr = dp.SetExpr(const)
    state = model.target_state

    assert expr.eval(state, model) == expected


def test_set_expr_bool_raise():
    model = dp.Model()
    obj = model.add_object_type(number=4)
    const = model.create_set_const(object_type=obj, value=[])
    expr = dp.SetExpr(const)

    with pytest.raises(Exception, match="cannot be converted to bool"):
        bool(expr)

    with pytest.raises(Exception, match="cannot be converted to bool"):
        if expr:
            pass


def test_set_expr_eval_raise():
    model = dp.Model()
    obj = model.add_object_type(number=4)
    const = model.create_set_const(object_type=obj, value=[0, 1])
    state = model.target_state

    with pytest.raises(BaseException, match="called `Option::unwrap"):
        (dp.IntExpr.state_cost() > 0).if_then_else(
            dp.SetExpr(const), dp.SetExpr(const)
        ).eval(state, model)


@pytest.mark.parametrize("value, expected", set_eval_cases)
def test_set_const(value, expected):
    model = dp.Model()
    obj = model.add_object_type(number=4)
    const = model.create_set_const(object_type=obj, value=value)

    assert const.eval() == expected


def test_set_const_bool_raise():
    model = dp.Model()
    obj = model.add_object_type(number=4)
    const = model.create_set_const(object_type=obj, value=[])

    with pytest.raises(Exception, match="cannot be converted to bool"):
        bool(const)

    with pytest.raises(Exception, match="cannot be converted to bool"):
        if const:
            pass


@pytest.mark.parametrize("value, expected", set_eval_cases)
def test_set_var(value, expected):
    model = dp.Model()
    obj = model.add_object_type(number=4)
    var = model.add_set_var(object_type=obj, target=value)
    state = model.target_state

    assert state[var] == expected


def test_set_var_bool_raise():
    model = dp.Model()
    obj = model.add_object_type(number=4)
    var = model.add_set_var(object_type=obj, target=[])

    with pytest.raises(Exception, match="cannot be converted to bool"):
        bool(var)

    with pytest.raises(Exception, match="cannot be converted to bool"):
        if var:
            pass


@pytest.mark.parametrize("value, expected", set_eval_cases)
def test_set_resource_var(value, expected):
    model = dp.Model()
    obj = model.add_object_type(number=4)
    var = model.add_set_resource_var(
        object_type=obj, target=value, less_is_better=False
    )
    state = model.target_state

    assert state[var] == expected


def test_set_resource_var_bool_raise():
    model = dp.Model()
    obj = model.add_object_type(number=4)
    var = model.add_set_resource_var(object_type=obj, target=[], less_is_better=False)

    with pytest.raises(Exception, match="cannot be converted to bool"):
        bool(var)

    with pytest.raises(Exception, match="cannot be converted to bool"):
        if var:
            pass


set_len_cases = [
    ([], 0),
    ([0], 1),
    ([0, 1], 2),
    ([0, 2, 3], 3),
    ([0, 1, 2, 3], 4),
    (set(), 0),
    ({0}, 1),
    ({0}, 1),
    ({0, 1}, 2),
    ({0, 2, 3}, 3),
    ({0, 1, 2, 3}, 4),
]


@pytest.mark.parametrize("value, expected", set_len_cases)
def test_set_expr_len(value, expected):
    model = dp.Model()
    obj = model.add_object_type(number=4)
    const = model.create_set_const(object_type=obj, value=value)
    expr = dp.SetExpr(const)
    state = model.target_state

    assert expr.len().eval(state, model) == expected


@pytest.mark.parametrize("value, expected", set_len_cases)
def test_set_const_len(value, expected):
    model = dp.Model()
    obj = model.add_object_type(number=4)
    const = model.create_set_const(object_type=obj, value=value)
    state = model.target_state

    assert const.len().eval(state, model) == expected


@pytest.mark.parametrize("value, expected", set_len_cases)
def test_set_var_len(value, expected):
    model = dp.Model()
    obj = model.add_object_type(number=4)
    var = model.add_set_var(object_type=obj, target=value)
    state = model.target_state

    assert var.len().eval(state, model) == expected


@pytest.mark.parametrize("value, expected", set_len_cases)
def test_set_resource_var_len(value, expected):
    model = dp.Model()
    obj = model.add_object_type(number=4)
    var = model.add_set_resource_var(
        object_type=obj, target=value, less_is_better=False
    )
    state = model.target_state

    assert var.len().eval(state, model) == expected


set_empty_cases = [
    ([], True),
    ([0], False),
    ([1, 2, 3], False),
    (set(), True),
    ({0}, False),
    ({1, 2, 3}, False),
]


@pytest.mark.parametrize("value, expected", set_empty_cases)
def test_set_expr_empty(value, expected):
    model = dp.Model()
    obj = model.add_object_type(number=4)
    const = model.create_set_const(object_type=obj, value=value)
    expr = dp.SetExpr(const)
    state = model.target_state

    assert expr.is_empty().eval(state, model) == expected


@pytest.mark.parametrize("value, expected", set_empty_cases)
def test_set_const_empty(value, expected):
    model = dp.Model()
    obj = model.add_object_type(number=4)
    const = model.create_set_const(object_type=obj, value=value)
    state = model.target_state

    assert const.is_empty().eval(state, model) == expected


@pytest.mark.parametrize("value, expected", set_empty_cases)
def test_set_var_empty(value, expected):
    model = dp.Model()
    obj = model.add_object_type(number=4)
    var = model.add_set_var(object_type=obj, target=value)
    state = model.target_state

    assert var.is_empty().eval(state, model) == expected


@pytest.mark.parametrize("value, expected", set_empty_cases)
def test_set_resource_var_empty(value, expected):
    model = dp.Model()
    obj = model.add_object_type(number=4)
    var = model.add_set_resource_var(
        object_type=obj, target=value, less_is_better=False
    )
    state = model.target_state

    assert var.is_empty().eval(state, model) == expected


set_complement_cases = [
    ([], {0, 1, 2, 3}),
    ([0], {1, 2, 3}),
    ([0, 1], {2, 3}),
    ([0, 1, 2, 3], set()),
    (set(), {0, 1, 2, 3}),
    ({0}, {1, 2, 3}),
    ({0, 1}, {2, 3}),
    ({0, 1, 2, 3}, set()),
]


@pytest.mark.parametrize("value, expected", set_complement_cases)
def test_set_expr_complement(value, expected):
    model = dp.Model()
    obj = model.add_object_type(number=4)
    const = model.create_set_const(object_type=obj, value=value)
    expr = dp.SetExpr(const)
    state = model.target_state

    assert expr.complement().eval(state, model) == expected


@pytest.mark.parametrize("value, expected", set_complement_cases)
def test_set_const_complement(value, expected):
    model = dp.Model()
    obj = model.add_object_type(number=4)
    const = model.create_set_const(object_type=obj, value=value)
    state = model.target_state

    assert const.complement().eval(state, model) == expected


@pytest.mark.parametrize("value, expected", set_complement_cases)
def test_set_var_complement(value, expected):
    model = dp.Model()
    obj = model.add_object_type(number=4)
    var = model.add_set_var(object_type=obj, target=value)
    state = model.target_state

    assert var.complement().eval(state, model) == expected


@pytest.mark.parametrize("value, expected", set_complement_cases)
def test_set_resource_var_complement(value, expected):
    model = dp.Model()
    obj = model.add_object_type(number=4)
    var = model.add_set_resource_var(
        object_type=obj, target=value, less_is_better=False
    )
    state = model.target_state

    assert var.complement().eval(state, model) == expected


class TestSetBinaryOperator:
    model = dp.Model()
    obj = model.add_object_type(number=4)

    const = model.create_set_const(object_type=obj, value=[0, 1])
    expr = dp.SetExpr(const)
    var = model.add_set_var(object_type=obj, target=[0, 1])
    res_var = model.add_set_resource_var(
        object_type=obj, target=[0, 1], less_is_better=False
    )

    empty_const = model.create_set_const(object_type=obj, value=[])
    empty_expr = dp.SetExpr(empty_const)
    empty_var = model.add_set_var(object_type=obj, target=[])

    overlap_const = model.create_set_const(object_type=obj, value=[1, 2])
    overlap_expr = dp.SetExpr(overlap_const)
    overlap_var = model.add_set_var(object_type=obj, target=[1, 2])

    subset_const = model.create_set_const(object_type=obj, value=[0])
    subset_expr = dp.SetExpr(subset_const)
    subset_var = model.add_set_var(object_type=obj, target=[0])

    disjoint_const = model.create_set_const(object_type=obj, value=[2, 3])
    disjoint_expr = dp.SetExpr(disjoint_const)
    disjoint_var = model.add_set_var(object_type=obj, target=[2, 3])

    state = model.target_state

    or_cases: ClassVar = []

    for value in [const, expr, var, res_var]:
        for other in [empty_const, empty_expr, empty_var]:
            or_cases.append((value, other, {0, 1}))

        for other in [overlap_const, overlap_expr, overlap_var]:
            or_cases.append((value, other, {0, 1, 2}))

        for other in [subset_const, subset_expr, subset_var]:
            or_cases.append((value, other, {0, 1}))

        for other in [disjoint_const, disjoint_expr, disjoint_var]:
            or_cases.append((value, other, {0, 1, 2, 3}))

    @pytest.mark.parametrize("lhs, rhs, expected", or_cases)
    def test_or(self, lhs, rhs, expected):
        assert (lhs | rhs).eval(self.state, self.model) == expected

    @pytest.mark.parametrize("lhs, rhs, expected", or_cases)
    def test_union(self, lhs, rhs, expected):
        assert lhs.union(rhs).eval(self.state, self.model) == expected

    sub_cases: ClassVar = []

    for value in [const, expr, var, res_var]:
        for other in [empty_const, empty_expr, empty_var]:
            sub_cases.append((value, other, {0, 1}))

        for other in [overlap_const, overlap_expr, overlap_var]:
            sub_cases.append((value, other, {0}))

        for other in [subset_const, subset_expr, subset_var]:
            sub_cases.append((value, other, {1}))

        for other in [disjoint_const, disjoint_expr, disjoint_var]:
            sub_cases.append((value, other, {0, 1}))

    @pytest.mark.parametrize("lhs, rhs, expected", sub_cases)
    def test_sub(self, lhs, rhs, expected):
        assert (lhs - rhs).eval(self.state, self.model) == expected

    @pytest.mark.parametrize("lhs, rhs, expected", sub_cases)
    def test_difference(self, lhs, rhs, expected):
        assert lhs.difference(rhs).eval(self.state, self.model) == expected

    and_cases: ClassVar = []

    for value in [const, expr, var, res_var]:
        for other in [empty_const, empty_expr, empty_var]:
            and_cases.append((value, other, set()))

        for other in [overlap_const, overlap_expr, overlap_var]:
            and_cases.append((value, other, {1}))

        for other in [subset_const, subset_expr, subset_var]:
            and_cases.append((value, other, {0}))

        for other in [disjoint_const, disjoint_expr, disjoint_var]:
            and_cases.append((value, other, set()))

    @pytest.mark.parametrize("lhs, rhs, expected", and_cases)
    def test_and(self, lhs, rhs, expected):
        assert (lhs & rhs).eval(self.state, self.model) == expected

    @pytest.mark.parametrize("lhs, rhs, expected", and_cases)
    def test_intersection(self, lhs, rhs, expected):
        assert lhs.intersection(rhs).eval(self.state, self.model) == expected

    xor_cases: ClassVar = []

    for value in [const, expr, var, res_var]:
        for other in [empty_const, empty_expr, empty_var]:
            xor_cases.append((value, other, {0, 1}))

        for other in [overlap_const, overlap_expr, overlap_var]:
            xor_cases.append((value, other, {0, 2}))

        for other in [subset_const, subset_expr, subset_var]:
            xor_cases.append((value, other, {1}))

        for other in [disjoint_const, disjoint_expr, disjoint_var]:
            xor_cases.append((value, other, {0, 1, 2, 3}))

    @pytest.mark.parametrize("lhs, rhs, expected", xor_cases)
    def test_xor(self, lhs, rhs, expected):
        assert (lhs ^ rhs).eval(self.state, self.model) == expected

    @pytest.mark.parametrize("lhs, rhs, expected", xor_cases)
    def test_symmetric_difference(self, lhs, rhs, expected):
        assert lhs.symmetric_difference(rhs).eval(self.state, self.model) == expected

    lt_cases: ClassVar = [
        (value, value, False) for value in [const, expr, var, res_var]
    ]

    for value in [const, expr, var, res_var]:
        for other in [empty_const, empty_expr, empty_var]:
            lt_cases.append((value, other, False))
            lt_cases.append((other, value, True))

        for other in [overlap_const, overlap_expr, overlap_var]:
            lt_cases.append((value, other, False))
            lt_cases.append((other, other, False))

        for other in [subset_const, subset_expr, subset_var]:
            lt_cases.append((value, other, False))
            lt_cases.append((other, value, True))

        for other in [disjoint_const, disjoint_expr, disjoint_var]:
            lt_cases.append((value, other, False))
            lt_cases.append((other, value, False))

    @pytest.mark.parametrize("lhs, rhs, expected", lt_cases)
    def test_lt(self, lhs, rhs, expected):
        assert (lhs < rhs).eval(self.state, self.model) == expected

    le_cases: ClassVar = [(value, value, True) for value in [const, expr, var, res_var]]

    for value in [const, expr, var, res_var]:
        for other in [empty_const, empty_expr, empty_var]:
            le_cases.append((value, other, False))
            le_cases.append((other, value, True))

        for other in [overlap_const, overlap_expr, overlap_var]:
            le_cases.append((value, other, False))
            le_cases.append((other, value, False))

        for other in [subset_const, subset_expr, subset_var]:
            le_cases.append((value, other, False))
            le_cases.append((other, value, True))

        for other in [disjoint_const, disjoint_expr, disjoint_var]:
            le_cases.append((value, other, False))
            le_cases.append((other, value, False))

    @pytest.mark.parametrize("lhs, rhs, expected", le_cases)
    def test_le(self, lhs, rhs, expected):
        assert (lhs <= rhs).eval(self.state, self.model) == expected

    @pytest.mark.parametrize("lhs, rhs, expected", le_cases)
    def test_issubset(self, lhs, rhs, expected):
        assert lhs.issubset(rhs).eval(self.state, self.model) == expected

    eq_cases: ClassVar = [(value, value, True) for value in [const, expr, var, res_var]]

    for value in [const, expr, var, res_var]:
        for other in [empty_const, empty_expr, empty_var]:
            eq_cases.append((value, other, False))
            eq_cases.append((other, value, False))

        for other in [overlap_const, overlap_expr, overlap_var]:
            eq_cases.append((value, other, False))
            eq_cases.append((other, value, False))

        for other in [subset_const, subset_expr, subset_var]:
            eq_cases.append((value, other, False))
            eq_cases.append((other, value, False))

        for other in [disjoint_const, disjoint_expr, disjoint_var]:
            eq_cases.append((value, other, False))
            eq_cases.append((other, value, False))

    @pytest.mark.parametrize("lhs, rhs, expected", eq_cases)
    def test_eq(self, lhs, rhs, expected):
        assert (lhs == rhs).eval(self.state, self.model) == expected

    ne_cases: ClassVar = [
        (value, value, False) for value in [const, expr, var, res_var]
    ]

    for value in [const, expr, var, res_var]:
        for other in [empty_const, empty_expr, empty_var]:
            ne_cases.append((value, other, True))
            ne_cases.append((other, value, True))

        for other in [overlap_const, overlap_expr, overlap_var]:
            ne_cases.append((value, other, True))
            ne_cases.append((other, value, True))

        for other in [subset_const, subset_expr, subset_var]:
            ne_cases.append((value, other, True))
            ne_cases.append((other, value, True))

        for other in [disjoint_const, disjoint_expr, disjoint_var]:
            ne_cases.append((value, other, True))
            ne_cases.append((other, value, True))

    @pytest.mark.parametrize("lhs, rhs, expected", ne_cases)
    def test_ne(self, lhs, rhs, expected):
        assert (lhs != rhs).eval(self.state, self.model) == expected

    ge_cases: ClassVar = [(value, value, True) for value in [const, expr, var, res_var]]

    for value in [const, expr, var, res_var]:
        for other in [empty_const, empty_expr, empty_var]:
            ge_cases.append((value, other, True))
            ge_cases.append((other, value, False))

        for other in [overlap_const, overlap_expr, overlap_var]:
            ge_cases.append((value, other, False))
            ge_cases.append((other, value, False))

        for other in [subset_const, subset_expr, subset_var]:
            ge_cases.append((value, other, True))
            ge_cases.append((other, value, False))

        for other in [disjoint_const, disjoint_expr, disjoint_var]:
            ge_cases.append((value, other, False))
            ge_cases.append((other, value, False))

    @pytest.mark.parametrize("lhs, rhs, expected", ge_cases)
    def test_ge(self, lhs, rhs, expected):
        assert (lhs >= rhs).eval(self.state, self.model) == expected

    @pytest.mark.parametrize("lhs, rhs, expected", ge_cases)
    def test_issuperset(self, lhs, rhs, expected):
        assert lhs.issuperset(rhs).eval(self.state, self.model) == expected

    gt_cases: ClassVar = [
        (value, value, False) for value in [const, expr, var, res_var]
    ]

    for value in [const, expr, var, res_var]:
        for other in [empty_const, empty_expr, empty_var]:
            gt_cases.append((value, other, True))
            gt_cases.append((other, value, False))

        for other in [overlap_const, overlap_expr, overlap_var]:
            gt_cases.append((value, other, False))
            gt_cases.append((other, value, False))

        for other in [subset_const, subset_expr, subset_var]:
            gt_cases.append((value, other, True))
            gt_cases.append((other, value, False))

        for other in [disjoint_const, disjoint_expr, disjoint_var]:
            gt_cases.append((value, other, False))
            gt_cases.append((other, value, False))

    @pytest.mark.parametrize("lhs, rhs, expected", gt_cases)
    def test_gt(self, lhs, rhs, expected):
        assert (lhs > rhs).eval(self.state, self.model) == expected


class TestSetElementOperator:
    model = dp.Model()
    obj = model.add_object_type(number=4)

    const = model.create_set_const(object_type=obj, value=[0, 1])
    expr = dp.SetExpr(const)
    var = model.add_set_var(object_type=obj, target=[0, 1])
    res_var = model.add_set_resource_var(
        object_type=obj, target=[0, 1], less_is_better=False
    )

    empty_const = model.create_set_const(object_type=obj, value=[])
    empty_expr = dp.SetExpr(empty_const)
    empty_var = model.add_set_var(object_type=obj, target=[])

    one_expr = dp.ElementExpr(1)
    one_var = model.add_element_var(object_type=obj, target=1)
    one_resource_var = model.add_element_resource_var(
        object_type=obj, target=1, less_is_better=True
    )

    two_expr = dp.ElementExpr(2)
    two_var = model.add_element_var(object_type=obj, target=2)
    two_resource_var = model.add_element_resource_var(
        object_type=obj, target=2, less_is_better=True
    )

    state = model.target_state

    add_cases: ClassVar = []

    for value in [const, expr, var, res_var]:
        for one in [one_expr, one_var, one_resource_var, 1]:
            add_cases.append((value, one, {0, 1}))

        for two in [two_expr, two_var, two_resource_var, 2]:
            add_cases.append((value, two, {0, 1, 2}))

    for empty in [empty_const, empty_expr, empty_var]:
        for one in [one_expr, one_var, one_resource_var, 1]:
            add_cases.append((empty, one, {1}))

        for two in [two_expr, two_var, two_resource_var, 2]:
            add_cases.append((empty, two, {2}))

    @pytest.mark.parametrize("value, element, expected", add_cases)
    def test_add(self, value, element, expected):
        assert value.add(element).eval(self.state, self.model) == expected

    discard_cases: ClassVar = []

    for value in [const, expr, var, res_var]:
        for one in [one_expr, one_var, one_resource_var, 1]:
            discard_cases.append((value, one, {0}))

        for two in [two_expr, two_var, two_resource_var, 2]:
            discard_cases.append((value, two, {0, 1}))

    for empty in [empty_const, empty_expr, empty_var]:
        for one in [one_expr, one_var, one_resource_var, 1]:
            discard_cases.append((empty, one, set()))

        for two in [two_expr, two_var, two_resource_var, 2]:
            discard_cases.append((empty, two, set()))

    @pytest.mark.parametrize("value, element, expected", discard_cases)
    def test_discard(self, value, element, expected):
        assert value.discard(element).eval(self.state, self.model) == expected

    @pytest.mark.parametrize("value, element, expected", discard_cases)
    def test_remove(self, value, element, expected):
        assert value.remove(element).eval(self.state, self.model) == expected

    contains_cases: ClassVar = []

    for value in [const, expr, var, res_var]:
        for one in [one_expr, one_var, one_resource_var, 1]:
            contains_cases.append((value, one, True))

        for two in [two_expr, two_var, two_resource_var, 2]:
            contains_cases.append((value, two, False))

    for empty in [empty_const, empty_expr, empty_var]:
        for one in [one_expr, one_var, one_resource_var, 1]:
            contains_cases.append((empty, one, False))

        for two in [two_expr, two_var, two_resource_var, 2]:
            contains_cases.append((empty, two, False))

    @pytest.mark.parametrize("value, element, expected", contains_cases)
    def test_contains(self, value, element, expected):
        assert value.contains(element).eval(self.state, self.model) == expected


class TestSetIsdisjoint:
    model = dp.Model()
    obj = model.add_object_type(number=4)

    const = model.create_set_const(object_type=obj, value=[0, 1])
    expr = dp.SetExpr(const)
    var = model.add_set_var(object_type=obj, target=[0, 1])
    res_var = model.add_set_resource_var(
        object_type=obj, target=[0, 1], less_is_better=False
    )

    overlap_const = model.create_set_const(object_type=obj, value=[1, 2])
    disjoint_const = model.create_set_const(object_type=obj, value=[2, 3])

    state = model.target_state

    cases: ClassVar = []

    for value in [const, expr, var, res_var]:
        cases.append((value, overlap_const, False))
        cases.append((value, disjoint_const, True))

    @pytest.mark.parametrize("value, other, expected", cases)
    def test_isdisjoint(self, value, other, expected):
        assert value.isdisjoint(other).eval(self.state, self.model) == expected


class TestLocalVar:
    model = dp.Model()
    obj = model.add_object_type(number=4)
    var = model.add_set_var(object_type=obj, target=[0, 1, 2, 3])
    x = model.add_local_var()
    state = model.target_state

    def test_add_named(self):
        model = dp.Model()
        model.add_local_var("x")
        x = model.get_local_var("x")
        obj = model.add_object_type(number=4)
        var = model.add_set_var(object_type=obj, target=[0, 1, 2, 3])
        state = model.target_state

        assert var.filter(x, x > 1).eval(state, model) == {2, 3}

    def test_add_duplicate_name_raise(self):
        model = dp.Model()
        model.add_local_var("x")

        with pytest.raises(RuntimeError):
            model.add_local_var("x")

    def test_get_undefined_raise(self):
        model = dp.Model()

        with pytest.raises(RuntimeError):
            model.get_local_var("x")

    def test_bool_raise(self):
        with pytest.raises(Exception, match="cannot be converted to bool"):
            bool(self.x)

        with pytest.raises(Exception, match="cannot be converted to bool"):
            if self.x:
                pass

    def test_add(self):
        assert self.var.filter(self.x, (self.x + 1) > 1).eval(
            self.state, self.model
        ) == {1, 2, 3}

    def test_radd(self):
        assert self.var.filter(self.x, (1 + self.x) > 1).eval(
            self.state, self.model
        ) == {1, 2, 3}

    def test_sub(self):
        assert self.var.filter(self.x, (self.x - self.x) == 0).eval(
            self.state, self.model
        ) == {0, 1, 2, 3}

    def test_rsub(self):
        assert self.var.filter(self.x, (3 - self.x) >= 2).eval(
            self.state, self.model
        ) == {0, 1}

    def test_mul(self):
        assert self.var.filter(self.x, (self.x * 2) >= 4).eval(
            self.state, self.model
        ) == {2, 3}

    def test_rmul(self):
        assert self.var.filter(self.x, (2 * self.x) >= 4).eval(
            self.state, self.model
        ) == {2, 3}

    def test_floordiv(self):
        assert self.var.filter(self.x, (self.x // 2) >= 1).eval(
            self.state, self.model
        ) == {2, 3}

    def test_rfloordiv(self):
        assert self.var.filter(self.x, (6 // (self.x + 1)) >= 3).eval(
            self.state, self.model
        ) == {0, 1}

    def test_truediv(self):
        assert self.var.filter(self.x, (self.x / 2) >= 1).eval(
            self.state, self.model
        ) == {2, 3}

    def test_rtruediv(self):
        assert self.var.filter(self.x, (6 / (self.x + 1)) >= 3).eval(
            self.state, self.model
        ) == {0, 1}

    def test_mod(self):
        assert self.var.filter(self.x, (self.x % 2) == 0).eval(
            self.state, self.model
        ) == {0, 2}

    def test_rmod(self):
        assert self.var.filter(self.x, (5 % (self.x + 1)) == 0).eval(
            self.state, self.model
        ) == {0}

    def test_lt(self):
        assert self.var.filter(self.x, self.x < 2).eval(self.state, self.model) == {
            0,
            1,
        }

    def test_le(self):
        assert self.var.filter(self.x, self.x <= 2).eval(self.state, self.model) == {
            0,
            1,
            2,
        }

    def test_eq(self):
        assert self.var.filter(self.x, self.x == 2).eval(self.state, self.model) == {2}

    def test_ne(self):
        assert self.var.filter(self.x, self.x != 2).eval(self.state, self.model) == {
            0,
            1,
            3,
        }

    def test_gt(self):
        assert self.var.filter(self.x, self.x > 2).eval(self.state, self.model) == {3}

    def test_ge(self):
        assert self.var.filter(self.x, self.x >= 2).eval(self.state, self.model) == {
            2,
            3,
        }


class TestSetReduce:
    model = dp.Model()
    obj = model.add_object_type(number=4)

    const = model.create_set_const(object_type=obj, value=[0, 1, 2, 3])
    expr = dp.SetExpr(const)
    var = model.add_set_var(object_type=obj, target=[0, 1, 2, 3])
    res_var = model.add_set_resource_var(
        object_type=obj, target=[0, 1, 2, 3], less_is_better=False
    )

    int_table = model.add_int_table([0, 1, 2, 3])
    float_table = model.add_float_table([0.0, 1.0, 2.0, 3.0])

    x = model.add_local_var()
    state = model.target_state

    values: ClassVar = [const, expr, var, res_var]

    @pytest.mark.parametrize("value", values)
    def test_filter(self, value):
        assert value.filter(self.x, self.x > 1).eval(self.state, self.model) == {
            2,
            3,
        }

    @pytest.mark.parametrize("value", values)
    def test_any(self, value):
        assert value.any(self.x, self.x > 2).eval(self.state, self.model)
        assert not value.any(self.x, self.x > 3).eval(self.state, self.model)

    @pytest.mark.parametrize("value", values)
    def test_all(self, value):
        assert value.all(self.x, self.x < 4).eval(self.state, self.model)
        assert not value.all(self.x, self.x > 0).eval(self.state, self.model)

    def test_any_and_all_empty_set(self):
        empty = self.model.create_set_const(object_type=self.obj, value=[])
        assert not empty.any(self.x, self.x >= 0).eval(self.state, self.model)
        assert empty.all(self.x, self.x < 0).eval(self.state, self.model)

    @pytest.mark.parametrize("value", values)
    def test_sum_int(self, value):
        assert (
            value.sum(self.x, self.int_table[self.x]).eval(self.state, self.model) == 6
        )

    @pytest.mark.parametrize("value", values)
    def test_sum_float(self, value):
        assert (
            value.sum(self.x, self.float_table[self.x]).eval(self.state, self.model)
            == 6.0
        )

    @pytest.mark.parametrize("value", values)
    def test_product_int(self, value):
        assert (
            value.filter(self.x, self.x > 0)
            .product(self.x, self.int_table[self.x])
            .eval(self.state, self.model)
            == 6
        )

    @pytest.mark.parametrize("value", values)
    def test_product_float(self, value):
        assert (
            value.filter(self.x, self.x > 0)
            .product(self.x, self.float_table[self.x])
            .eval(self.state, self.model)
            == 6.0
        )

    @pytest.mark.parametrize("value", values)
    def test_max_int(self, value):
        assert (
            value.max(self.x, self.int_table[self.x]).eval(self.state, self.model) == 3
        )

    @pytest.mark.parametrize("value", values)
    def test_max_float(self, value):
        assert (
            value.max(self.x, self.float_table[self.x]).eval(self.state, self.model)
            == 3.0
        )

    @pytest.mark.parametrize("value", values)
    def test_min_int(self, value):
        assert (
            value.min(self.x, self.int_table[self.x]).eval(self.state, self.model) == 0
        )

    @pytest.mark.parametrize("value", values)
    def test_min_float(self, value):
        assert (
            value.min(self.x, self.float_table[self.x]).eval(self.state, self.model)
            == 0.0
        )

    @pytest.mark.parametrize("value", values)
    def test_filter_then_sum_int(self, value):
        assert (
            value.filter(self.x, self.x > 1)
            .sum(self.x, self.int_table[self.x])
            .eval(self.state, self.model)
            == 5
        )

    @pytest.mark.parametrize("value", values)
    def test_filter_then_sum_float(self, value):
        assert (
            value.filter(self.x, self.x > 1)
            .sum(self.x, self.float_table[self.x])
            .eval(self.state, self.model)
            == 5.0
        )

    @pytest.mark.parametrize("value", values)
    def test_filter_then_product_int(self, value):
        assert (
            value.filter(self.x, self.x > 1)
            .product(self.x, self.int_table[self.x])
            .eval(self.state, self.model)
            == 6
        )

    @pytest.mark.parametrize("value", values)
    def test_filter_then_product_float(self, value):
        assert (
            value.filter(self.x, self.x > 1)
            .product(self.x, self.float_table[self.x])
            .eval(self.state, self.model)
            == 6.0
        )

    @pytest.mark.parametrize("value", values)
    def test_filter_then_max_int(self, value):
        assert (
            value.filter(self.x, self.x > 1)
            .max(self.x, self.int_table[self.x])
            .eval(self.state, self.model)
            == 3
        )

    @pytest.mark.parametrize("value", values)
    def test_filter_then_max_float(self, value):
        assert (
            value.filter(self.x, self.x > 1)
            .max(self.x, self.float_table[self.x])
            .eval(self.state, self.model)
            == 3.0
        )

    @pytest.mark.parametrize("value", values)
    def test_filter_then_min_int(self, value):
        assert (
            value.filter(self.x, self.x > 1)
            .min(self.x, self.int_table[self.x])
            .eval(self.state, self.model)
            == 2
        )

    @pytest.mark.parametrize("value", values)
    def test_filter_then_min_float(self, value):
        assert (
            value.filter(self.x, self.x > 1)
            .min(self.x, self.float_table[self.x])
            .eval(self.state, self.model)
            == 2.0
        )

    def test_max_empty_set_raise(self):
        with pytest.raises(BaseException, match="reduce performed on an empty set"):
            self.var.filter(self.x, self.x > 100).max(
                self.x, self.int_table[self.x]
            ).eval(self.state, self.model)

    def test_min_empty_set_raise(self):
        with pytest.raises(BaseException, match="reduce performed on an empty set"):
            self.var.filter(self.x, self.x > 100).min(
                self.x, self.int_table[self.x]
            ).eval(self.state, self.model)


int_expr_cases = [(3, 3), (-3, -3)]


@pytest.mark.parametrize("value, expected", int_expr_cases)
def test_int_expr(value, expected):
    model = dp.Model()
    state = model.target_state
    expr = dp.IntExpr(value)

    assert expr.eval(state, model) == expected


def test_int_expr_bool_raise():
    expr = dp.IntExpr(1)

    with pytest.raises(Exception, match="cannot be converted to bool"):
        bool(expr)

    with pytest.raises(Exception, match="cannot be converted to bool"):
        if expr:
            pass


def test_int_var_bool_raise():
    model = dp.Model()
    var = model.add_int_var(target=1)

    with pytest.raises(Exception, match="cannot be converted to bool"):
        bool(var)

    with pytest.raises(Exception, match="cannot be converted to bool"):
        if var:
            pass


def test_int_resource_var_bool_raise():
    model = dp.Model()
    var = model.add_int_resource_var(target=1)

    with pytest.raises(Exception, match="cannot be converted to bool"):
        bool(var)

    with pytest.raises(Exception, match="cannot be converted to bool"):
        if var:
            pass


def test_int_expr_eval_raise():
    model = dp.Model()
    state = model.target_state

    with pytest.raises(BaseException, match="called `Option::unwrap"):
        dp.IntExpr.state_cost().eval(state, model)


def test_int_expr_float_raise():
    with pytest.raises(TypeError):
        dp.IntExpr(3.5)


def test_int_expr_eval_cost():
    model = dp.Model()
    state = model.target_state
    expr = dp.IntExpr.state_cost() + 2

    assert expr.eval_cost(1, state, model) == 3


int_abs_cases = [(3, 3), (-3, 3)]


@pytest.mark.parametrize("value, expected", int_abs_cases)
def test_int_expr_abs(value, expected):
    model = dp.Model()
    state = model.target_state
    expr = dp.IntExpr(value)

    assert abs(expr).eval(state, model) == expected


@pytest.mark.parametrize("value, expected", int_abs_cases)
def test_int_var_abs(value, expected):
    model = dp.Model()
    var = model.add_int_var(target=value)
    state = model.target_state

    assert abs(var).eval(state, model) == expected


@pytest.mark.parametrize("value, expected", int_abs_cases)
def test_int_resource_var_abs(value, expected):
    model = dp.Model()
    var = model.add_int_resource_var(target=value, less_is_better=True)
    state = model.target_state

    assert abs(var).eval(state, model) == expected


int_neg_cases = [(3, -3), (-3, 3)]


@pytest.mark.parametrize("value, expected", int_neg_cases)
def test_int_expr_neg(value, expected):
    model = dp.Model()
    state = model.target_state
    expr = dp.IntExpr(value)

    assert (-expr).eval(state, model) == expected


@pytest.mark.parametrize("value, expected", int_neg_cases)
def test_int_var_neg(value, expected):
    model = dp.Model()
    var = model.add_int_var(target=value)
    state = model.target_state

    assert (-var).eval(state, model) == expected


@pytest.mark.parametrize("value, expected", int_neg_cases)
def test_int_resource_var_neg(value, expected):
    model = dp.Model()
    var = model.add_int_resource_var(target=value, less_is_better=True)
    state = model.target_state

    assert (-var).eval(state, model) == expected


class TestIntBinaryOperator:
    model = dp.Model()

    three_expr = dp.IntExpr(3)
    three_var = model.add_int_var(target=3)
    three_resource_var = model.add_int_resource_var(target=3, less_is_better=True)

    two_expr = dp.IntExpr(2)
    two_var = model.add_int_var(target=2)
    two_resource_var = model.add_int_resource_var(target=2, less_is_better=True)

    modulo_expr = dp.FloatExpr(4.0)
    modulo_var = model.add_float_var(target=4.0)
    modulo_resource_var = model.add_float_resource_var(target=4.0, less_is_better=True)

    state = model.target_state

    add_cases: ClassVar = []

    for i, three in enumerate([three_expr, three_var, three_resource_var, 3]):
        for j, two in enumerate([two_expr, two_var, two_resource_var, 2]):
            if i < 3 or j < 3:
                add_cases.append((three, two, 5))

    @pytest.mark.parametrize("lhs, rhs, expected", add_cases)
    def test_add(self, lhs, rhs, expected):
        assert (lhs + rhs).eval(self.state, self.model) == expected

    sub_cases: ClassVar = []

    for i, three in enumerate([three_expr, three_var, three_resource_var, 3]):
        for j, two in enumerate([two_expr, two_var, two_resource_var, 2]):
            if i < 3 or j < 3:
                sub_cases.append((three, two, 1))

    @pytest.mark.parametrize("lhs, rhs, expected", sub_cases)
    def test_sub(self, lhs, rhs, expected):
        assert (lhs - rhs).eval(self.state, self.model) == expected

    mul_cases: ClassVar = []

    for i, three in enumerate([three_expr, three_var, three_resource_var, 3]):
        for j, two in enumerate([two_expr, two_var, two_resource_var, 2]):
            if i < 3 or j < 3:
                mul_cases.append((three, two, 6))

    @pytest.mark.parametrize("lhs, rhs, expected", mul_cases)
    def test_mul(self, lhs, rhs, expected):
        assert (lhs * rhs).eval(self.state, self.model) == expected

    mod_cases: ClassVar = []

    for i, three in enumerate([three_expr, three_var, three_resource_var, 3]):
        for j, two in enumerate([two_expr, two_var, two_resource_var, 2]):
            if i < 3 or j < 3:
                mod_cases.append((three, two, 1))

    @pytest.mark.parametrize("lhs, rhs, expected", mod_cases)
    def test_mod(self, lhs, rhs, expected):
        assert (lhs % rhs).eval(self.state, self.model) == expected

    truediv_cases: ClassVar = []

    for i, three in enumerate([three_expr, three_var, three_resource_var, 3]):
        for j, two in enumerate([two_expr, two_var, two_resource_var, 2]):
            if i < 3 or j < 3:
                truediv_cases.append((three, two, pytest.approx(1.5)))

    @pytest.mark.parametrize("lhs, rhs, expected", truediv_cases)
    def test_truediv(self, lhs, rhs, expected):
        assert (lhs / rhs).eval(self.state, self.model) == expected

    floordiv_cases: ClassVar = []

    for i, three in enumerate([three_expr, three_var, three_resource_var, 3]):
        for j, two in enumerate([two_expr, two_var, two_resource_var, 2]):
            if i < 3 or j < 3:
                floordiv_cases.append((three, two, 1))

    @pytest.mark.parametrize("lhs, rhs, expected", floordiv_cases)
    def test_floordiv(self, lhs, rhs, expected):
        assert (lhs // rhs).eval(self.state, self.model) == expected

    pow_cases: ClassVar = []

    for i, three in enumerate([three_expr, three_var, three_resource_var, 3]):
        for j, two in enumerate([two_expr, two_var, two_resource_var, 2]):
            if i < 3 or j < 3:
                pow_cases.append((three, two, pytest.approx(9.0)))

    @pytest.mark.parametrize("lhs, rhs, expected", pow_cases)
    def test_pow_op(self, lhs, rhs, expected):
        assert (lhs**rhs).eval(self.state, self.model) == expected

    @pytest.mark.parametrize("lhs, rhs, expected", pow_cases)
    def test_pow(self, lhs, rhs, expected):
        assert pow(lhs, rhs).eval(self.state, self.model) == expected

    pow_modulo_cases: ClassVar = []

    for i, three in enumerate([three_expr, three_var, three_resource_var, 3]):
        for j, two in enumerate([two_expr, two_var, two_resource_var, 2]):
            for k, modulo in enumerate(
                [modulo_expr, modulo_var, modulo_resource_var, 4.0]
            ):
                if i < 3 or j < 3:
                    pow_modulo_cases.append((three, two, modulo, pytest.approx(1.0)))

    @pytest.mark.parametrize("lhs, rhs, modulo, expected", pow_modulo_cases)
    def test_pow_modulo(self, lhs, rhs, modulo, expected):
        assert pow(lhs, rhs, modulo).eval(self.state, self.model) == expected

    lt_cases: ClassVar = [
        (value, value, False) for value in [three_expr, three_var, three_resource_var]
    ]

    for i, three in enumerate([three_expr, three_var, three_resource_var, 3]):
        for j, two in enumerate([two_expr, two_var, two_resource_var, 2]):
            if i < 3 or j < 3:
                lt_cases.append((three, two, False))
                lt_cases.append((two, three, True))

    @pytest.mark.parametrize("lhs, rhs, expected", lt_cases)
    def test_lt(self, lhs, rhs, expected):
        assert (lhs < rhs).eval(self.state, self.model) == expected

    le_cases: ClassVar = [
        (value, value, True) for value in [three_expr, three_var, three_resource_var]
    ]

    for i, three in enumerate([three_expr, three_var, three_resource_var, 3]):
        for j, two in enumerate([two_expr, two_var, two_resource_var, 2]):
            if i < 3 or j < 3:
                le_cases.append((three, two, False))
                le_cases.append((two, three, True))

    @pytest.mark.parametrize("lhs, rhs, expected", le_cases)
    def test_le(self, lhs, rhs, expected):
        assert (lhs <= rhs).eval(self.state, self.model) == expected

    eq_cases: ClassVar = [
        (value, value, True) for value in [three_expr, three_var, three_resource_var]
    ]

    for i, three in enumerate([three_expr, three_var, three_resource_var, 3]):
        for j, two in enumerate([two_expr, two_var, two_resource_var, 2]):
            if i < 3 or j < 3:
                eq_cases.append((three, two, False))

    @pytest.mark.parametrize("lhs, rhs, expected", eq_cases)
    def test_eq(self, lhs, rhs, expected):
        assert (lhs == rhs).eval(self.state, self.model) == expected

    ne_cases: ClassVar = [
        (value, value, False) for value in [three_expr, three_var, three_resource_var]
    ]

    for i, three in enumerate([three_expr, three_var, three_resource_var, 3]):
        for j, two in enumerate([two_expr, two_var, two_resource_var, 2]):
            if i < 3 or j < 3:
                ne_cases.append((three, two, True))

    @pytest.mark.parametrize("lhs, rhs, expected", ne_cases)
    def test_ne(self, lhs, rhs, expected):
        assert (lhs != rhs).eval(self.state, self.model) == expected

    ge_cases: ClassVar = [
        (value, value, True) for value in [three_expr, three_var, three_resource_var]
    ]

    for i, three in enumerate([three_expr, three_var, three_resource_var, 3]):
        for j, two in enumerate([two_expr, two_var, two_resource_var, 2]):
            if i < 3 or j < 3:
                ge_cases.append((three, two, True))
                ge_cases.append((two, three, False))

    @pytest.mark.parametrize("lhs, rhs, expected", ge_cases)
    def test_ge(self, lhs, rhs, expected):
        assert (lhs >= rhs).eval(self.state, self.model) == expected

    gt_cases: ClassVar = [
        (value, value, False) for value in [three_expr, three_var, three_resource_var]
    ]

    for i, three in enumerate([three_expr, three_var, three_resource_var, 3]):
        for j, two in enumerate([two_expr, two_var, two_resource_var, 2]):
            if i < 3 or j < 3:
                gt_cases.append((three, two, True))
                gt_cases.append((two, three, False))

    @pytest.mark.parametrize("lhs, rhs, expected", gt_cases)
    def test_gt(self, lhs, rhs, expected):
        assert (lhs > rhs).eval(self.state, self.model) == expected


float_expr_cases = [
    (3.5, pytest.approx(3.5)),
    (-3.5, pytest.approx(-3.5)),
    (3, pytest.approx(3.0)),
]


@pytest.mark.parametrize("value, expected", float_expr_cases)
def test_float_expr(value, expected):
    model = dp.Model()
    state = model.target_state
    expr = dp.FloatExpr(value)

    assert expr.eval(state, model) == expected


def test_float_expr_eval_raise():
    model = dp.Model()
    state = model.target_state

    with pytest.raises(BaseException, match="called `Option::unwrap"):
        dp.FloatExpr.state_cost().eval(state, model)


def test_float_expr_bool_raise():
    expr = dp.FloatExpr(1.5)

    with pytest.raises(Exception, match="cannot be converted to bool"):
        bool(expr)

    with pytest.raises(Exception, match="cannot be converted to bool"):
        if expr:
            pass


def test_float_var_bool_raise():
    model = dp.Model()
    var = model.add_float_var(target=1.5)

    with pytest.raises(Exception, match="cannot be converted to bool"):
        bool(var)

    with pytest.raises(Exception, match="cannot be converted to bool"):
        if var:
            pass


def test_float_resource_var_bool_raise():
    model = dp.Model()
    var = model.add_float_resource_var(target=1.5)

    with pytest.raises(Exception, match="cannot be converted to bool"):
        bool(var)

    with pytest.raises(Exception, match="cannot be converted to bool"):
        if var:
            pass


def test_float_expr_eval_cost():
    model = dp.Model()
    state = model.target_state
    expr = dp.FloatExpr.state_cost() + 2.5

    assert expr.eval_cost(1.5, state, model) == pytest.approx(4.0)


float_abs_cases = [(3.5, pytest.approx(3.5)), (-3.5, pytest.approx(3.5))]


@pytest.mark.parametrize("value, expected", float_abs_cases)
def test_float_expr_abs(value, expected):
    model = dp.Model()
    state = model.target_state
    expr = dp.FloatExpr(value)

    assert abs(expr).eval(state, model) == expected


@pytest.mark.parametrize("value, expected", float_abs_cases)
def test_float_var_abs(value, expected):
    model = dp.Model()
    var = model.add_float_var(target=value)
    state = model.target_state

    assert abs(var).eval(state, model) == expected


@pytest.mark.parametrize("value, expected", float_abs_cases)
def test_float_resource_var_abs(value, expected):
    model = dp.Model()
    var = model.add_float_resource_var(target=value, less_is_better=True)
    state = model.target_state

    assert abs(var).eval(state, model) == expected


float_neg_cases = [(3, -3), (-3, 3)]


@pytest.mark.parametrize("value, expected", float_neg_cases)
def test_float_expr_neg(value, expected):
    model = dp.Model()
    state = model.target_state
    expr = dp.FloatExpr(value)

    assert (-expr).eval(state, model) == expected


@pytest.mark.parametrize("value, expected", float_neg_cases)
def test_float_var_neg(value, expected):
    model = dp.Model()
    var = model.add_float_var(target=value)
    state = model.target_state

    assert (-var).eval(state, model) == expected


@pytest.mark.parametrize("value, expected", float_neg_cases)
def test_float_resource_var_neg(value, expected):
    model = dp.Model()
    var = model.add_float_resource_var(target=value, less_is_better=True)
    state = model.target_state

    assert (-var).eval(state, model) == expected


float_round_cases = [(3.5, 4), (3.4, 3), (-3.5, -4), (-3.4, -3)]


@pytest.mark.parametrize("value, expected", float_round_cases)
def test_float_expr_round(value, expected):
    model = dp.Model()
    state = model.target_state
    expr = dp.FloatExpr(value)

    assert round(expr).eval(state, model) == expected


@pytest.mark.parametrize("value, expected", float_round_cases)
def test_float_var_round(value, expected):
    model = dp.Model()
    var = model.add_float_var(target=value)
    state = model.target_state

    assert round(var).eval(state, model) == expected


@pytest.mark.parametrize("value, expected", float_round_cases)
def test_float_resource_var_round(value, expected):
    model = dp.Model()
    var = model.add_float_resource_var(target=value, less_is_better=True)
    state = model.target_state

    assert round(var).eval(state, model) == expected


float_trunc_cases = [(3.9, 3), (3.4, 3), (-3.9, -3), (-3.4, -3)]


@pytest.mark.parametrize("value, expected", float_trunc_cases)
def test_float_expr_trunc(value, expected):
    from math import trunc

    model = dp.Model()
    state = model.target_state
    expr = dp.FloatExpr(value)

    assert trunc(expr).eval(state, model) == expected


@pytest.mark.parametrize("value, expected", float_trunc_cases)
def test_float_var_trunc(value, expected):
    from math import trunc

    model = dp.Model()
    var = model.add_float_var(target=value)
    state = model.target_state

    assert trunc(var).eval(state, model) == expected


@pytest.mark.parametrize("value, expected", float_trunc_cases)
def test_float_resource_var_trunc(value, expected):
    from math import trunc

    model = dp.Model()
    var = model.add_float_resource_var(target=value, less_is_better=True)
    state = model.target_state

    assert trunc(var).eval(state, model) == expected


float_floor_cases = [(3.9, 3), (3.4, 3), (-3.9, -4), (-3.4, -4)]


@pytest.mark.parametrize("value, expected", float_floor_cases)
def test_float_expr_floor(value, expected):
    from math import floor

    model = dp.Model()
    state = model.target_state
    expr = dp.FloatExpr(value)

    assert floor(expr).eval(state, model) == expected


@pytest.mark.parametrize("value, expected", float_floor_cases)
def test_float_var_floor(value, expected):
    from math import floor

    model = dp.Model()
    var = model.add_float_var(target=value)
    state = model.target_state

    assert floor(var).eval(state, model) == expected


@pytest.mark.parametrize("value, expected", float_floor_cases)
def test_float_resource_var_floor(value, expected):
    from math import floor

    model = dp.Model()
    var = model.add_float_resource_var(target=value, less_is_better=True)
    state = model.target_state

    assert floor(var).eval(state, model) == expected


float_ceil_cases = [(3.9, 4), (3.4, 4), (-3.9, -3), (-3.4, -3)]


@pytest.mark.parametrize("value, expected", float_ceil_cases)
def test_float_expr_ceil(value, expected):
    from math import ceil

    model = dp.Model()
    state = model.target_state
    expr = dp.FloatExpr(value)

    assert ceil(expr).eval(state, model) == expected


@pytest.mark.parametrize("value, expected", float_ceil_cases)
def test_float_var_ceil(value, expected):
    from math import ceil

    model = dp.Model()
    var = model.add_float_var(target=value)
    state = model.target_state

    assert ceil(var).eval(state, model) == expected


@pytest.mark.parametrize("value, expected", float_ceil_cases)
def test_float_resource_var_ceil(value, expected):
    from math import ceil

    model = dp.Model()
    var = model.add_float_resource_var(target=value, less_is_better=True)
    state = model.target_state

    assert ceil(var).eval(state, model) == expected


class TestFloatBinaryOperator:
    model = dp.Model()

    three_expr = dp.FloatExpr(0.3)
    three_var = model.add_float_var(target=0.3)
    three_resource_var = model.add_float_resource_var(target=0.3, less_is_better=True)

    two_expr = dp.FloatExpr(0.2)
    two_var = model.add_float_var(target=0.2)
    two_resource_var = model.add_float_resource_var(target=0.2, less_is_better=True)

    modulo_expr = dp.FloatExpr(0.4)
    modulo_var = model.add_float_var(target=0.4)
    modulo_resource_var = model.add_float_resource_var(target=0.4, less_is_better=True)

    int_expr = dp.IntExpr(2)
    int_var = model.add_int_var(target=2)
    int_resource_var = model.add_int_resource_var(target=2, less_is_better=True)

    state = model.target_state

    add_cases: ClassVar = []

    for i, three in enumerate([three_expr, three_var, three_resource_var, 0.3]):
        for j, two in enumerate([two_expr, two_var, two_resource_var, 0.2]):
            if i < 3 or j < 3:
                add_cases.append((three, two, pytest.approx(0.5)))

    for i, float_value in enumerate([three_expr, three_var, three_resource_var, 0.3]):
        for j, int_value in enumerate([int_expr, int_var, int_resource_var, 2]):
            if i < 3 or j < 3:
                add_cases.append((float_value, int_value, pytest.approx(2.3)))
                add_cases.append((int_value, float_value, pytest.approx(2.3)))

    @pytest.mark.parametrize("lhs, rhs, expected", add_cases)
    def test_add(self, lhs, rhs, expected):
        assert (lhs + rhs).eval(self.state, self.model) == expected

    sub_cases: ClassVar = []

    for i, three in enumerate([three_expr, three_var, three_resource_var, 0.3]):
        for j, two in enumerate([two_expr, two_var, two_resource_var, 0.2]):
            if i < 3 or j < 3:
                sub_cases.append((three, two, pytest.approx(0.1)))

    for i, float_value in enumerate([three_expr, three_var, three_resource_var, 0.3]):
        for j, int_value in enumerate([int_expr, int_var, int_resource_var, 2]):
            if i < 3 or j < 3:
                sub_cases.append((float_value, int_value, pytest.approx(-1.7)))
                sub_cases.append((int_value, float_value, pytest.approx(1.7)))

    @pytest.mark.parametrize("lhs, rhs, expected", sub_cases)
    def test_sub(self, lhs, rhs, expected):
        assert (lhs - rhs).eval(self.state, self.model) == expected

    mul_cases: ClassVar = []

    for i, three in enumerate([three_expr, three_var, three_resource_var, 0.3]):
        for j, two in enumerate([two_expr, two_var, two_resource_var, 0.2]):
            if i < 3 or j < 3:
                mul_cases.append((three, two, pytest.approx(0.06)))

    for i, float_value in enumerate([three_expr, three_var, three_resource_var, 0.3]):
        for j, int_value in enumerate([int_expr, int_var, int_resource_var, 2]):
            if i < 3 or j < 3:
                mul_cases.append((float_value, int_value, pytest.approx(0.6)))
                mul_cases.append((int_value, float_value, pytest.approx(0.6)))

    @pytest.mark.parametrize("lhs, rhs, expected", mul_cases)
    def test_mul(self, lhs, rhs, expected):
        assert (lhs * rhs).eval(self.state, self.model) == expected

    mod_cases: ClassVar = []

    for i, three in enumerate([three_expr, three_var, three_resource_var, 0.3]):
        for j, two in enumerate([two_expr, two_var, two_resource_var, 0.2]):
            if i < 3 or j < 3:
                mod_cases.append((three, two, pytest.approx(0.1)))

    for i, float_value in enumerate([three_expr, three_var, three_resource_var, 0.3]):
        for j, int_value in enumerate([int_expr, int_var, int_resource_var, 2]):
            if i < 3 or j < 3:
                mod_cases.append(
                    (float_value, int_value, pytest.approx(pytest.approx(0.3)))
                )
                mod_cases.append(
                    (int_value, float_value, pytest.approx(pytest.approx(0.2)))
                )

    @pytest.mark.parametrize("lhs, rhs, expected", mod_cases)
    def test_mod(self, lhs, rhs, expected):
        assert (lhs % rhs).eval(self.state, self.model) == expected

    truediv_cases: ClassVar = []

    for i, three in enumerate([three_expr, three_var, three_resource_var, 0.3]):
        for j, two in enumerate([two_expr, two_var, two_resource_var, 0.2]):
            if i < 3 or j < 3:
                truediv_cases.append((three, two, pytest.approx(1.5)))

    for i, float_value in enumerate([three_expr, three_var, three_resource_var, 0.3]):
        for j, int_value in enumerate([int_expr, int_var, int_resource_var, 2]):
            if i < 3 or j < 3:
                truediv_cases.append(
                    (float_value, int_value, pytest.approx(pytest.approx(0.15)))
                )
                truediv_cases.append(
                    (int_value, float_value, pytest.approx(pytest.approx(2 / 0.3)))
                )

    @pytest.mark.parametrize("lhs, rhs, expected", truediv_cases)
    def test_truediv(self, lhs, rhs, expected):
        assert (lhs / rhs).eval(self.state, self.model) == expected

    floordiv_cases: ClassVar = []

    for i, three in enumerate([three_expr, three_var, three_resource_var, 0.3]):
        for j, two in enumerate([two_expr, two_var, two_resource_var, 0.2]):
            if i < 3 or j < 3:
                floordiv_cases.append((three, two, pytest.approx(1.0)))

    for i, float_value in enumerate([three_expr, three_var, three_resource_var, 0.3]):
        for j, int_value in enumerate([int_expr, int_var, int_resource_var, 2]):
            if i < 3 or j < 3:
                floordiv_cases.append(
                    (float_value, int_value, pytest.approx(pytest.approx(0.0)))
                )
                floordiv_cases.append(
                    (int_value, float_value, pytest.approx(pytest.approx(6.0)))
                )

    @pytest.mark.parametrize("lhs, rhs, expected", floordiv_cases)
    def test_floordiv(self, lhs, rhs, expected):
        assert (lhs // rhs).eval(self.state, self.model) == expected

    pow_cases: ClassVar = []

    for i, three in enumerate([three_expr, three_var, three_resource_var, 0.3]):
        for j, two in enumerate([two_expr, two_var, two_resource_var, 0.2]):
            if i < 3 or j < 3:
                pow_cases.append((three, two, pytest.approx(0.3**0.2)))

    for i, float_value in enumerate([three_expr, three_var, three_resource_var, 0.3]):
        for j, int_value in enumerate([int_expr, int_var, int_resource_var, 2]):
            if i < 3 or j < 3:
                pow_cases.append(
                    (float_value, int_value, pytest.approx(pytest.approx(0.09)))
                )
                pow_cases.append(
                    (int_value, float_value, pytest.approx(pytest.approx(2**0.3)))
                )

    @pytest.mark.parametrize("lhs, rhs, expected", pow_cases)
    def test_pow_op(self, lhs, rhs, expected):
        assert (lhs**rhs).eval(self.state, self.model) == expected

    @pytest.mark.parametrize("lhs, rhs, expected", pow_cases)
    def test_pow(self, lhs, rhs, expected):
        assert pow(lhs, rhs).eval(self.state, self.model) == expected

    pow_modulo_cases: ClassVar = []

    for i, three in enumerate([three_expr, three_var, three_resource_var]):
        for j, two in enumerate([two_expr, two_var, two_resource_var, 0.2]):
            for k, modulo in enumerate(
                [modulo_expr, modulo_var, modulo_resource_var, 0.4]
            ):
                pow_modulo_cases.append(
                    (three, two, modulo, pytest.approx(0.3**0.2 % 0.4))
                )

    @pytest.mark.parametrize("lhs, rhs, modulo, expected", pow_modulo_cases)
    def test_pow_modulo(self, lhs, rhs, modulo, expected):
        assert pow(lhs, rhs, modulo).eval(self.state, self.model) == expected

    lt_cases: ClassVar = [
        (value, value, False) for value in [three_expr, three_var, three_resource_var]
    ]

    for i, three in enumerate([three_expr, three_var, three_resource_var, 0.3]):
        for j, two in enumerate([two_expr, two_var, two_resource_var, 0.2]):
            if i < 3 or j < 3:
                lt_cases.append((three, two, False))
                lt_cases.append((two, three, True))

    for i, float_value in enumerate([three_expr, three_var, three_resource_var, 0.3]):
        for j, int_value in enumerate([int_expr, int_var, int_resource_var, 2]):
            if i < 3 or j < 3:
                lt_cases.append((float_value, int_value, True))
                lt_cases.append((int_value, float_value, False))

    @pytest.mark.parametrize("lhs, rhs, expected", lt_cases)
    def test_lt(self, lhs, rhs, expected):
        assert (lhs < rhs).eval(self.state, self.model) == expected

    le_cases: ClassVar = [
        (value, value, True) for value in [three_expr, three_var, three_resource_var]
    ]

    for i, three in enumerate([three_expr, three_var, three_resource_var, 0.3]):
        for j, two in enumerate([two_expr, two_var, two_resource_var, 0.2]):
            if i < 3 or j < 3:
                le_cases.append((three, two, False))
                le_cases.append((two, three, True))

    for i, float_value in enumerate([three_expr, three_var, three_resource_var, 0.3]):
        for j, int_value in enumerate([int_expr, int_var, int_resource_var, 2]):
            if i < 3 or j < 3:
                le_cases.append((float_value, int_value, True))
                le_cases.append((int_value, float_value, False))

    @pytest.mark.parametrize("lhs, rhs, expected", le_cases)
    def test_le(self, lhs, rhs, expected):
        assert (lhs <= rhs).eval(self.state, self.model) == expected

    eq_cases: ClassVar = [
        (value, value, True) for value in [three_expr, three_var, three_resource_var]
    ]

    for i, three in enumerate([three_expr, three_var, three_resource_var, 0.3]):
        for j, two in enumerate([two_expr, two_var, two_resource_var, 0.2]):
            if i < 3 or j < 3:
                eq_cases.append((three, two, False))

    for i, float_value in enumerate([three_expr, three_var, three_resource_var, 0.3]):
        for j, int_value in enumerate([int_expr, int_var, int_resource_var, 2]):
            if i < 3 or j < 3:
                eq_cases.append((float_value, int_value, False))
                eq_cases.append((int_value, float_value, False))

    @pytest.mark.parametrize("lhs, rhs, expected", eq_cases)
    def test_eq(self, lhs, rhs, expected):
        assert (lhs == rhs).eval(self.state, self.model) == expected

    ne_cases: ClassVar = [
        (value, value, False) for value in [three_expr, three_var, three_resource_var]
    ]

    for i, three in enumerate([three_expr, three_var, three_resource_var, 0.3]):
        for j, two in enumerate([two_expr, two_var, two_resource_var, 0.2]):
            if i < 3 or j < 3:
                ne_cases.append((three, two, True))

    for i, float_value in enumerate([three_expr, three_var, three_resource_var, 0.3]):
        for j, int_value in enumerate([int_expr, int_var, int_resource_var, 2]):
            if i < 3 or j < 3:
                ne_cases.append((float_value, int_value, True))
                ne_cases.append((int_value, float_value, True))

    @pytest.mark.parametrize("lhs, rhs, expected", ne_cases)
    def test_ne(self, lhs, rhs, expected):
        assert (lhs != rhs).eval(self.state, self.model) == expected

    ge_cases: ClassVar = [
        (value, value, True) for value in [three_expr, three_var, three_resource_var]
    ]

    for i, three in enumerate([three_expr, three_var, three_resource_var, 0.3]):
        for j, two in enumerate([two_expr, two_var, two_resource_var, 0.2]):
            if i < 3 or j < 3:
                ge_cases.append((three, two, True))
                ge_cases.append((two, three, False))

    for i, float_value in enumerate([three_expr, three_var, three_resource_var, 0.3]):
        for j, int_value in enumerate([int_expr, int_var, int_resource_var, 2]):
            if i < 3 or j < 3:
                ge_cases.append((float_value, int_value, False))
                ge_cases.append((int_value, float_value, True))

    @pytest.mark.parametrize("lhs, rhs, expected", ge_cases)
    def test_ge(self, lhs, rhs, expected):
        assert (lhs >= rhs).eval(self.state, self.model) == expected

    gt_cases: ClassVar = [
        (value, value, False) for value in [three_expr, three_var, three_resource_var]
    ]

    for i, three in enumerate([three_expr, three_var, three_resource_var, 0.3]):
        for j, two in enumerate([two_expr, two_var, two_resource_var, 0.2]):
            if i < 3 or j < 3:
                gt_cases.append((three, two, True))
                gt_cases.append((two, three, False))

    for i, float_value in enumerate([three_expr, three_var, three_resource_var, 0.3]):
        for j, int_value in enumerate([int_expr, int_var, int_resource_var, 2]):
            if i < 3 or j < 3:
                gt_cases.append((float_value, int_value, False))
                gt_cases.append((int_value, float_value, True))

    @pytest.mark.parametrize("lhs, rhs, expected", gt_cases)
    def test_gt(self, lhs, rhs, expected):
        assert (lhs > rhs).eval(self.state, self.model) == expected


class TestSqrt:
    model = dp.Model()

    int_expr = dp.IntExpr(4)
    int_var = model.add_int_var(target=4)
    int_resource_var = model.add_int_resource_var(target=4, less_is_better=True)

    float_expr = dp.FloatExpr(4.0)
    float_var = model.add_float_var(target=4.0)
    float_resource_var = model.add_float_resource_var(target=4.0, less_is_better=True)

    state = model.target_state

    cases: ClassVar = [
        (int_expr, pytest.approx(2.0)),
        (int_var, pytest.approx(2.0)),
        (int_resource_var, pytest.approx(2.0)),
        (4, pytest.approx(2.0)),
        (float_expr, pytest.approx(2.0)),
        (float_var, pytest.approx(2.0)),
        (float_resource_var, pytest.approx(2.0)),
        (4.0, pytest.approx(2.0)),
    ]

    @pytest.mark.parametrize("value, expected", cases)
    def test(self, value, expected):
        assert dp.sqrt(value).eval(self.state, self.model) == expected

    nan_cases: ClassVar = [-1, dp.IntExpr(-1), dp.FloatExpr(-1.0)]

    @pytest.mark.parametrize("value", nan_cases)
    def test_nan(self, value):
        import math

        assert math.isnan(dp.sqrt(value).eval(self.state, self.model))


class TestLog:
    model = dp.Model()

    three_expr = dp.FloatExpr(0.3)
    three_var = model.add_float_var(target=0.3)
    three_resource_var = model.add_float_resource_var(target=0.3, less_is_better=True)

    two_expr = dp.FloatExpr(0.2)
    two_var = model.add_float_var(target=0.2)
    two_resource_var = model.add_float_resource_var(target=0.2, less_is_better=True)

    int_expr = dp.IntExpr(2)
    int_var = model.add_int_var(target=2)
    int_resource_var = model.add_int_resource_var(target=2, less_is_better=True)

    state = model.target_state

    cases: ClassVar = []

    import math

    for i, three in enumerate([three_expr, three_var, three_resource_var, 0.3]):
        for j, two in enumerate([two_expr, two_var, two_resource_var, 0.2]):
            if i < 3 or j < 3:
                cases.append((three, two, pytest.approx(math.log(0.3, 0.2))))

    for i, float_value in enumerate([three_expr, three_var, three_resource_var, 0.3]):
        for j, int_value in enumerate([int_expr, int_var, int_resource_var, 2]):
            if i < 3 or j < 3:
                cases.append((float_value, int_value, pytest.approx(math.log2(0.3))))
                cases.append((int_value, float_value, pytest.approx(math.log(2, 0.3))))

    @pytest.mark.parametrize("lhs, rhs, expected", cases)
    def test(self, lhs, rhs, expected):
        assert dp.log(lhs, rhs).eval(self.state, self.model) == expected

    nan_cases: ClassVar = [
        (-2, 1),
        (1, -1),
        (dp.IntExpr(-2), 1),
        (1, dp.IntExpr(-1)),
        (dp.FloatExpr(-2), 1),
        (1, dp.FloatExpr(-1)),
    ]

    @pytest.mark.parametrize("lhs, rhs", nan_cases)
    def test_nan(self, lhs, rhs):
        import math

        assert math.isnan(dp.log(lhs, rhs).eval(self.state, self.model))


class TestFloat:
    model = dp.Model()

    int_expr = dp.IntExpr(2)
    int_var = model.add_int_var(target=2)
    int_resource_var = model.add_int_resource_var(target=2, less_is_better=True)

    float_expr = dp.FloatExpr(2.5)
    float_var = model.add_float_var(target=2.5)
    float_resource_var = model.add_float_resource_var(target=2.5, less_is_better=True)

    state = model.target_state

    cases: ClassVar = [
        (int_expr, pytest.approx(2.0)),
        (int_var, pytest.approx(2.0)),
        (int_resource_var, pytest.approx(2.0)),
        (2, pytest.approx(2.0)),
    ]

    @pytest.mark.parametrize("value, expected", cases)
    def test(self, value, expected):
        assert dp.float(value).eval(self.state, self.model) == expected

    error_cases: ClassVar = [float_expr, float_var, float_resource_var, 2.5]

    @pytest.mark.parametrize("value", error_cases)
    def test_error(self, value):
        with pytest.raises(TypeError):
            dp.float(value)


class TestElementMax:
    model = dp.Model()
    obj = model.add_object_type(number=4)

    three_expr = dp.ElementExpr(3)
    three_var = model.add_element_var(object_type=obj, target=3)
    three_resource_var = model.add_element_resource_var(
        object_type=obj, target=3, less_is_better=True
    )

    two_expr = dp.ElementExpr(2)
    two_var = model.add_element_var(object_type=obj, target=2)
    two_resource_var = model.add_element_resource_var(
        object_type=obj, target=2, less_is_better=False
    )

    state = model.target_state

    cases: ClassVar = [
        (three, three, 3) for three in [three_expr, three_var, three_resource_var, 3]
    ]

    for i, three in enumerate([three_expr, three_var, three_resource_var, 3]):
        for j, two in enumerate([two_expr, two_var, two_resource_var, 2]):
            cases.append((three, two, 3))
            cases.append((two, three, 3))

    @pytest.mark.parametrize("lhs, rhs, expected", cases)
    def test(self, lhs, rhs, expected):
        assert dp.max(lhs, rhs).eval(self.state, self.model) == expected


class TestSetMax:
    model = dp.Model()
    obj = model.add_object_type(number=4)

    const = model.create_set_const(object_type=obj, value=[0, 1])
    expr = dp.SetExpr(const)
    var = model.add_set_var(object_type=obj, target=[0, 1])

    empty_const = model.create_set_const(object_type=obj, value=[])
    empty_expr = dp.SetExpr(empty_const)
    empty_var = model.add_set_var(object_type=obj, target=[])

    overlap_const = model.create_set_const(object_type=obj, value=[1, 2])
    overlap_expr = dp.SetExpr(overlap_const)
    overlap_var = model.add_set_var(object_type=obj, target=[1, 2])

    subset_const = model.create_set_const(object_type=obj, value=[0])
    subset_expr = dp.SetExpr(subset_const)
    subset_var = model.add_set_var(object_type=obj, target=[0])

    disjoint_const = model.create_set_const(object_type=obj, value=[2, 3])
    disjoint_expr = dp.SetExpr(disjoint_const)
    disjoint_var = model.add_set_var(object_type=obj, target=[2, 3])

    state = model.target_state

    cases: ClassVar = [(value, value, {0, 1}) for value in [const, expr, var]]

    for value in [const, expr, var]:
        for other in [empty_const, empty_expr, empty_var]:
            cases.append((value, other, {0, 1}))
            cases.append((other, value, {0, 1}))

        for other in [overlap_const, overlap_expr, overlap_var]:
            cases.append((value, other, {0, 1}))
            cases.append((other, value, {1, 2}))

        for other in [subset_const, subset_expr, subset_var]:
            cases.append((value, other, {0, 1}))
            cases.append((other, value, {0, 1}))

        for other in [disjoint_const, disjoint_expr, disjoint_var]:
            cases.append((value, other, {0, 1}))
            cases.append((other, value, {2, 3}))

    @pytest.mark.parametrize("lhs, rhs, expected", cases)
    def test(self, lhs, rhs, expected):
        assert dp.max(lhs, rhs).eval(self.state, self.model) == expected


class TestIntMax:
    model = dp.Model()

    three_expr = dp.IntExpr(3)
    three_var = model.add_int_var(target=3)
    three_resource_var = model.add_int_resource_var(target=3, less_is_better=True)

    two_expr = dp.IntExpr(2)
    two_var = model.add_int_var(target=2)
    two_resource_var = model.add_int_resource_var(target=2, less_is_better=True)

    state = model.target_state

    cases: ClassVar = [
        (value, value, 3) for value in [three_expr, three_var, three_resource_var, 3]
    ]

    for i, three in enumerate([three_expr, three_var, three_resource_var, 3]):
        for j, two in enumerate([two_expr, two_var, two_resource_var, 2]):
            cases.append((three, two, 3))
            cases.append((two, three, 3))

    @pytest.mark.parametrize("lhs, rhs, expected", cases)
    def test(self, lhs, rhs, expected):
        assert dp.max(lhs, rhs).eval(self.state, self.model) == expected


class TestFloatMax:
    model = dp.Model()

    three_expr = dp.FloatExpr(0.3)
    three_var = model.add_float_var(target=0.3)
    three_resource_var = model.add_float_resource_var(target=0.3, less_is_better=True)

    two_expr = dp.FloatExpr(0.2)
    two_var = model.add_float_var(target=0.2)
    two_resource_var = model.add_float_resource_var(target=0.2, less_is_better=True)

    int_expr = dp.IntExpr(2)
    int_var = model.add_int_var(target=2)
    int_resource_var = model.add_int_resource_var(target=2, less_is_better=True)

    state = model.target_state

    cases: ClassVar = [
        (value, value, pytest.approx(0.3))
        for value in [three_expr, three_var, three_resource_var, 0.3]
    ]

    for i, three in enumerate([three_expr, three_var, three_resource_var, 0.3]):
        for j, two in enumerate([two_expr, two_var, two_resource_var, 0.2]):
            cases.append((three, two, pytest.approx(0.3)))

    for i, float_value in enumerate([three_expr, three_var, three_resource_var, 0.3]):
        for j, int_value in enumerate([int_expr, int_var, int_resource_var, 2]):
            cases.append((float_value, int_value, pytest.approx(2.0)))
            cases.append((int_value, float_value, pytest.approx(2.0)))

    @pytest.mark.parametrize("lhs, rhs, expected", cases)
    def test(self, lhs, rhs, expected):
        assert dp.max(lhs, rhs).eval(self.state, self.model) == expected


class TestElementMin:
    model = dp.Model()
    obj = model.add_object_type(number=4)

    three_expr = dp.ElementExpr(3)
    three_var = model.add_element_var(object_type=obj, target=3)
    three_resource_var = model.add_element_resource_var(
        object_type=obj, target=3, less_is_better=True
    )

    two_expr = dp.ElementExpr(2)
    two_var = model.add_element_var(object_type=obj, target=2)
    two_resource_var = model.add_element_resource_var(
        object_type=obj, target=2, less_is_better=False
    )

    state = model.target_state

    cases: ClassVar = [
        (three, three, 3) for three in [three_expr, three_var, three_resource_var, 3]
    ]

    for i, three in enumerate([three_expr, three_var, three_resource_var, 3]):
        for j, two in enumerate([two_expr, two_var, two_resource_var, 2]):
            cases.append((three, two, 2))
            cases.append((two, three, 2))

    @pytest.mark.parametrize("lhs, rhs, expected", cases)
    def test(self, lhs, rhs, expected):
        assert dp.min(lhs, rhs).eval(self.state, self.model) == expected


class TestSetMin:
    model = dp.Model()
    obj = model.add_object_type(number=4)

    const = model.create_set_const(object_type=obj, value=[0, 1])
    expr = dp.SetExpr(const)
    var = model.add_set_var(object_type=obj, target=[0, 1])

    empty_const = model.create_set_const(object_type=obj, value=[])
    empty_expr = dp.SetExpr(empty_const)
    empty_var = model.add_set_var(object_type=obj, target=[])

    overlap_const = model.create_set_const(object_type=obj, value=[1, 2])
    overlap_expr = dp.SetExpr(overlap_const)
    overlap_var = model.add_set_var(object_type=obj, target=[1, 2])

    subset_const = model.create_set_const(object_type=obj, value=[0])
    subset_expr = dp.SetExpr(subset_const)
    subset_var = model.add_set_var(object_type=obj, target=[0])

    disjoint_const = model.create_set_const(object_type=obj, value=[2, 3])
    disjoint_expr = dp.SetExpr(disjoint_const)
    disjoint_var = model.add_set_var(object_type=obj, target=[2, 3])

    state = model.target_state

    cases: ClassVar = [(value, value, {0, 1}) for value in [const, expr, var]]

    for value in [const, expr, var]:
        for other in [empty_const, empty_expr, empty_var]:
            cases.append((value, other, set()))
            cases.append((other, value, set()))

        for other in [overlap_const, overlap_expr, overlap_var]:
            cases.append((value, other, {0, 1}))
            cases.append((other, value, {1, 2}))

        for other in [subset_const, subset_expr, subset_var]:
            cases.append((value, other, {0}))
            cases.append((other, value, {0}))

        for other in [disjoint_const, disjoint_expr, disjoint_var]:
            cases.append((value, other, {0, 1}))
            cases.append((other, value, {2, 3}))

    @pytest.mark.parametrize("lhs, rhs, expected", cases)
    def test(self, lhs, rhs, expected):
        assert dp.min(lhs, rhs).eval(self.state, self.model) == expected


class TestIntMin:
    model = dp.Model()

    three_expr = dp.IntExpr(3)
    three_var = model.add_int_var(target=3)
    three_resource_var = model.add_int_resource_var(target=3, less_is_better=True)

    two_expr = dp.IntExpr(2)
    two_var = model.add_int_var(target=2)
    two_resource_var = model.add_int_resource_var(target=2, less_is_better=True)

    state = model.target_state

    cases: ClassVar = [
        (value, value, 3) for value in [three_expr, three_var, three_resource_var, 3]
    ]

    for i, three in enumerate([three_expr, three_var, three_resource_var, 3]):
        for j, two in enumerate([two_expr, two_var, two_resource_var, 2]):
            cases.append((three, two, 2))
            cases.append((two, three, 2))

    @pytest.mark.parametrize("lhs, rhs, expected", cases)
    def test(self, lhs, rhs, expected):
        assert dp.min(lhs, rhs).eval(self.state, self.model) == expected


class TestFloatMin:
    model = dp.Model()

    three_expr = dp.FloatExpr(0.3)
    three_var = model.add_float_var(target=0.3)
    three_resource_var = model.add_float_resource_var(target=0.3, less_is_better=True)

    two_expr = dp.FloatExpr(0.2)
    two_var = model.add_float_var(target=0.2)
    two_resource_var = model.add_float_resource_var(target=0.2, less_is_better=True)

    int_expr = dp.IntExpr(2)
    int_var = model.add_int_var(target=2)
    int_resource_var = model.add_int_resource_var(target=2, less_is_better=True)

    state = model.target_state

    cases: ClassVar = [
        (value, value, pytest.approx(0.3))
        for value in [three_expr, three_var, three_resource_var, 0.3]
    ]

    for i, three in enumerate([three_expr, three_var, three_resource_var, 0.3]):
        for j, two in enumerate([two_expr, two_var, two_resource_var, 0.2]):
            cases.append((three, two, pytest.approx(0.2)))

    for i, float_value in enumerate([three_expr, three_var, three_resource_var, 0.3]):
        for j, int_value in enumerate([int_expr, int_var, int_resource_var, 2]):
            cases.append((float_value, int_value, pytest.approx(0.3)))
            cases.append((int_value, float_value, pytest.approx(0.3)))

    @pytest.mark.parametrize("lhs, rhs, expected", cases)
    def test(self, lhs, rhs, expected):
        assert dp.min(lhs, rhs).eval(self.state, self.model) == expected


class TestMaxMinError:
    model = dp.Model()
    obj = model.add_object_type(number=4)

    element_expr = dp.ElementExpr(1)
    element_var = model.add_element_var(object_type=obj, target=1)
    element_resource_var = model.add_element_resource_var(
        object_type=obj, target=1, less_is_better=True
    )

    set_const = model.create_set_const(object_type=obj, value=[1, 2])
    set_expr = dp.SetExpr(set_const)
    set_var = model.add_set_var(object_type=obj, target=[1, 2])

    int_expr = dp.IntExpr(1)
    int_var = model.add_int_var(target=1)
    int_resource_var = model.add_int_resource_var(target=1, less_is_better=True)

    float_expr = dp.FloatExpr(1.0)
    float_var = model.add_float_var(target=1.0)
    float_resource_var = model.add_float_resource_var(target=1.0, less_is_better=True)

    state = model.target_state

    cases: ClassVar = []

    for set_value in [set_const, set_expr, set_var]:
        for element_value in [element_expr, element_var, element_resource_var, 1]:
            cases.append((set_value, element_value))
            cases.append((element_value, set_value))

        for int_value in [int_expr, int_var, int_resource_var]:
            cases.append((set_value, int_value))
            cases.append((int_value, set_value))

        for float_value in [float_expr, float_var, float_resource_var, 1.0]:
            cases.append((set_value, float_value))
            cases.append((float_value, set_value))

    for element_value in [element_expr, element_var, element_resource_var]:
        cases.append((element_var, -1))

        for int_value in [int_expr, int_var, int_resource_var]:
            cases.append((element_value, int_value))
            cases.append((int_value, element_value))

        for float_value in [float_expr, float_var, float_resource_var, 1.0]:
            cases.append((element_value, float_value))
            cases.append((float_value, element_value))

    @pytest.mark.parametrize("lhs, rhs", cases)
    def test_max(self, lhs, rhs):
        with pytest.raises(TypeError):
            dp.max(lhs, rhs)

    @pytest.mark.parametrize("lhs, rhs", cases)
    def test_min(self, lhs, rhs):
        with pytest.raises(TypeError):
            dp.min(lhs, rhs)


class TestFractionalKnapsack:
    model = dp.Model()
    obj = model.add_object_type(number=4)
    var = model.add_set_var(object_type=obj, target=[0, 1, 2])
    state = model.target_state

    capacity = 5
    values: ClassVar = [2, 3, 5, 10]
    weights: ClassVar = [1, 2, 4, 1]
    values_float: ClassVar = [2.0, 3.0, 5.0, 10.0]
    weights_float: ClassVar = [1.0, 2.0, 4.0, 1.0]

    int_values = model.add_int_table(values)
    int_weights = model.add_int_table(weights)
    float_values = model.add_float_table(values_float)
    float_weights = model.add_float_table(weights_float)

    def test_array_int(self):
        expr = dp.fractional_knapsack(
            self.var, self.capacity, self.values, self.weights
        )

        assert expr.eval(self.state, self.model) == 7.5

    def test_array_float(self):
        expr = dp.fractional_knapsack(
            self.var, self.capacity, self.values_float, self.weights_float
        )

        assert expr.eval(self.state, self.model) == 7.5

    def test_array_expr(self):
        values = [dp.IntExpr(v) for v in self.values]
        weights = [dp.IntExpr(w) for w in self.weights]
        expr = dp.fractional_knapsack(self.var, self.capacity, values, weights)

        assert expr.eval(self.state, self.model) == 7.5

    def test_int_tables(self):
        expr = dp.fractional_knapsack(
            self.var, self.capacity, self.int_values, self.int_weights
        )

        assert expr.eval(self.state, self.model) == 7.5

    def test_float_tables(self):
        expr = dp.fractional_knapsack(
            self.var, self.capacity, self.float_values, self.float_weights
        )

        assert expr.eval(self.state, self.model) == 7.5

    def test_int_value_float_weight_tables(self):
        expr = dp.fractional_knapsack(
            self.var, self.capacity, self.int_values, self.float_weights
        )

        assert expr.eval(self.state, self.model) == 7.5

    def test_float_value_int_weight_tables(self):
        expr = dp.fractional_knapsack(
            self.var, self.capacity, self.float_values, self.int_weights
        )

        assert expr.eval(self.state, self.model) == 7.5

    def test_mixed_array_and_table_raise(self):
        with pytest.raises(TypeError):
            dp.fractional_knapsack(
                self.var, self.capacity, self.int_values, self.weights
            )

        with pytest.raises(TypeError):
            dp.fractional_knapsack(
                self.var, self.capacity, self.values, self.int_weights
            )

    def test_table_argument_does_not_hang(self):
        # Regression test: `values`/`weights` used to be matched against a
        # plain `list` conversion before a 1D table, and `IntTable1D.__getitem__`
        # never raises `IndexError`, so Python's legacy sequence-iteration
        # protocol used to loop forever trying to read the table as a list.
        expr = dp.fractional_knapsack(
            self.var, self.capacity, self.int_values, self.int_weights
        )

        assert expr.eval(self.state, self.model) == 7.5


class TestMinimumSpanningTree:
    model = dp.Model()
    obj = model.add_object_type(number=4)
    nodes = model.create_set_const(object_type=obj, value=[0, 1, 2, 3])
    state = model.target_state

    edge_costs = model.add_int_table(
        [[0, 1, 4, 3], [1, 0, 2, 5], [4, 2, 0, 6], [3, 5, 6, 0]]
    )
    edge_costs_float = model.add_float_table(
        [
            [0.0, 1.5, 4.5, 3.5],
            [1.5, 0.0, 2.5, 5.5],
            [4.5, 2.5, 0.0, 6.5],
            [3.5, 5.5, 6.5, 0.0],
        ]
    )
    connected = model.add_bool_table(
        [
            [True, True, True, False],
            [True, True, True, True],
            [True, True, True, True],
            [False, True, True, True],
        ]
    )

    int_edges: ClassVar = [(0, 1, 1), (0, 2, 4), (0, 3, 3), (1, 2, 2), (2, 3, 6)]
    float_edges: ClassVar = [
        (0, 1, 1.5),
        (0, 2, 4.5),
        (0, 3, 3.5),
        (1, 2, 2.5),
        (2, 3, 6.5),
    ]
    int_edges_with_connectivity: ClassVar = [
        (0, 1, 1, True),
        (0, 2, 4, True),
        (0, 3, 3, True),
        (1, 2, 2, False),
        (2, 3, 6, True),
    ]
    float_edges_with_connectivity: ClassVar = [
        (0, 1, 1.5, True),
        (0, 2, 4.5, True),
        (0, 3, 3.5, True),
        (1, 2, 2.5, False),
        (2, 3, 6.5, True),
    ]

    def test_int_table(self):
        expr = dp.minimum_spanning_tree(self.nodes, self.edge_costs)

        assert expr.eval(self.state, self.model) == 6

    def test_float_table(self):
        expr = dp.minimum_spanning_tree(self.nodes, self.edge_costs_float)

        assert expr.eval(self.state, self.model) == 7.5

    def test_int_table_with_connected(self):
        expr = dp.minimum_spanning_tree(self.nodes, self.edge_costs, self.connected)

        assert expr.eval(self.state, self.model) == 8

    def test_float_table_with_connected(self):
        expr = dp.minimum_spanning_tree(
            self.nodes, self.edge_costs_float, self.connected
        )

        assert expr.eval(self.state, self.model) == 9.5

    def test_int_edges(self):
        expr = dp.minimum_spanning_tree(self.nodes, self.int_edges)

        assert expr.eval(self.state, self.model) == 6

    def test_float_edges(self):
        expr = dp.minimum_spanning_tree(self.nodes, self.float_edges)

        assert expr.eval(self.state, self.model) == 7.5

    def test_int_edges_with_connectivity(self):
        expr = dp.minimum_spanning_tree(self.nodes, self.int_edges_with_connectivity)

        assert expr.eval(self.state, self.model) == 8

    def test_float_edges_with_connectivity(self):
        expr = dp.minimum_spanning_tree(self.nodes, self.float_edges_with_connectivity)

        assert expr.eval(self.state, self.model) == 9.5

    def test_connected_with_edges_raise(self):
        with pytest.raises(TypeError):
            dp.minimum_spanning_tree(self.nodes, self.int_edges, self.connected)

    def test_mixed_edge_arity_raise(self):
        with pytest.raises(TypeError):
            dp.minimum_spanning_tree(self.nodes, [(0, 1, 1), (1, 2, 2, True)])


def test_condition_bool_error():
    condition = dp.IntExpr(2) > dp.IntExpr(1)

    with pytest.raises(Exception, match="cannot be converted to bool"):
        bool(condition)

    with pytest.raises(Exception, match="cannot be converted to bool"):
        if condition:
            pass


comparison_cases = [
    (dp.IntExpr(2), dp.IntExpr(1)),
    (dp.IntExpr(2), 1),
    (2, dp.IntExpr(1)),
    (dp.FloatExpr(2.5), dp.IntExpr(1)),
    (dp.IntExpr(2), dp.FloatExpr(1.5)),
    (dp.ElementExpr(2), dp.ElementExpr(1)),
]


@pytest.mark.parametrize("lhs, rhs", comparison_cases)
def test_max_bool_error(lhs, rhs):
    with pytest.raises(Exception, match="cannot be converted to bool"):
        max(lhs, rhs)


@pytest.mark.parametrize("lhs, rhs", comparison_cases)
def test_min_bool_error(lhs, rhs):
    with pytest.raises(Exception, match="cannot be converted to bool"):
        min(lhs, rhs)


condition_eval_cases = [
    (dp.IntExpr(2) > dp.IntExpr(1), True),
    (dp.IntExpr(2) < dp.IntExpr(1), False),
]


def test_condition_true():
    model = dp.Model()
    state = model.target_state
    condition = dp.Condition(True)

    assert condition.eval(state, model)


def test_condition_false():
    model = dp.Model()
    state = model.target_state
    condition = dp.Condition(False)

    assert not condition.eval(state, model)


@pytest.mark.parametrize("condition, expected", condition_eval_cases)
def test_condition_eval(condition, expected):
    model = dp.Model()
    state = model.target_state

    assert condition.eval(state, model) == expected


def test_condition_eval_error():
    model = dp.Model()
    state = model.target_state

    with pytest.raises(BaseException, match="called `Option::unwrap"):
        (dp.IntExpr.state_cost() > 0).eval(state, model)


condition_invert_cases = [
    (dp.IntExpr(2) > dp.IntExpr(1), False),
    (dp.IntExpr(2) < dp.IntExpr(1), True),
]


@pytest.mark.parametrize("condition, expected", condition_invert_cases)
def test_condition_invert(condition, expected):
    model = dp.Model()
    state = model.target_state

    assert (~condition).eval(state, model) == expected


condition_and_cases = [
    (dp.IntExpr(2) > dp.IntExpr(1), dp.IntExpr(3) > dp.IntExpr(2), True),
    (dp.IntExpr(2) < dp.IntExpr(1), dp.IntExpr(3) > dp.IntExpr(2), False),
    (dp.IntExpr(2) > dp.IntExpr(1), dp.IntExpr(3) < dp.IntExpr(2), False),
    (dp.IntExpr(2) < dp.IntExpr(1), dp.IntExpr(3) < dp.IntExpr(2), False),
]


@pytest.mark.parametrize("lhs, rhs, expected", condition_and_cases)
def test_condition_and(lhs, rhs, expected):
    model = dp.Model()
    state = model.target_state

    assert (lhs & rhs).eval(state, model) == expected


condition_or_cases = [
    (dp.IntExpr(2) > dp.IntExpr(1), dp.IntExpr(3) > dp.IntExpr(2), True),
    (dp.IntExpr(2) < dp.IntExpr(1), dp.IntExpr(3) > dp.IntExpr(2), True),
    (dp.IntExpr(2) > dp.IntExpr(1), dp.IntExpr(3) < dp.IntExpr(2), True),
    (dp.IntExpr(2) < dp.IntExpr(1), dp.IntExpr(3) < dp.IntExpr(2), False),
]


@pytest.mark.parametrize("lhs, rhs, expected", condition_or_cases)
def test_condition_or(lhs, rhs, expected):
    model = dp.Model()
    state = model.target_state

    assert (lhs | rhs).eval(state, model) == expected


class TestElementIfThenElse:
    model = dp.Model()
    obj = model.add_object_type(number=4)

    three_expr = dp.ElementExpr(3)
    three_var = model.add_element_var(object_type=obj, target=3)
    three_resource_var = model.add_element_resource_var(
        object_type=obj, target=3, less_is_better=True
    )

    two_expr = dp.ElementExpr(2)
    two_var = model.add_element_var(object_type=obj, target=2)
    two_resource_var = model.add_element_resource_var(
        object_type=obj, target=2, less_is_better=False
    )

    state = model.target_state

    cases: ClassVar = []

    for i, three in enumerate([three_expr, three_var, three_resource_var, 3]):
        for j, two in enumerate([two_expr, two_var, two_resource_var, 2]):
            cases.append((dp.IntExpr(2) > dp.IntExpr(1), three, two, 3))
            cases.append((dp.IntExpr(2) < dp.IntExpr(1), three, two, 2))

    @pytest.mark.parametrize("condition, lhs, rhs, expected", cases)
    def test(self, condition, lhs, rhs, expected):
        assert condition.if_then_else(lhs, rhs).eval(self.state, self.model) == expected


class TestSetIfThenElse:
    model = dp.Model()
    obj = model.add_object_type(number=4)

    const = model.create_set_const(object_type=obj, value=[0, 1])
    expr = dp.SetExpr(const)
    var = model.add_set_var(object_type=obj, target=[0, 1])

    empty_const = model.create_set_const(object_type=obj, value=[])
    empty_expr = dp.SetExpr(empty_const)
    empty_var = model.add_set_var(object_type=obj, target=[])

    state = model.target_state

    cases: ClassVar = []

    for value in [const, expr, var]:
        for other in [empty_const, empty_expr, empty_var]:
            cases.append((dp.IntExpr(2) > dp.IntExpr(1), value, other, {0, 1}))
            cases.append((dp.IntExpr(2) < dp.IntExpr(1), value, other, set()))

    @pytest.mark.parametrize("condition, lhs, rhs, expected", cases)
    def test(self, condition, lhs, rhs, expected):
        assert condition.if_then_else(lhs, rhs).eval(self.state, self.model) == expected


class TestIntIfThenElse:
    model = dp.Model()

    three_expr = dp.IntExpr(3)
    three_var = model.add_int_var(target=3)
    three_resource_var = model.add_int_resource_var(target=3, less_is_better=True)

    two_expr = dp.IntExpr(2)
    two_var = model.add_int_var(target=2)
    two_resource_var = model.add_int_resource_var(target=2, less_is_better=True)

    state = model.target_state

    cases: ClassVar = []

    for i, three in enumerate([three_expr, three_var, three_resource_var, 3]):
        for j, two in enumerate([two_expr, two_var, two_resource_var, 2]):
            cases.append((dp.IntExpr(2) > dp.IntExpr(1), three, two, 3))
            cases.append((dp.IntExpr(2) < dp.IntExpr(1), three, two, 2))

    @pytest.mark.parametrize("condition, lhs, rhs, expected", cases)
    def test(self, condition, lhs, rhs, expected):
        assert condition.if_then_else(lhs, rhs).eval(self.state, self.model) == expected


class TestFloatIfThenElse:
    model = dp.Model()

    three_expr = dp.FloatExpr(0.3)
    three_var = model.add_float_var(target=0.3)
    three_resource_var = model.add_float_resource_var(target=0.3, less_is_better=True)

    two_expr = dp.FloatExpr(0.2)
    two_var = model.add_float_var(target=0.2)
    two_resource_var = model.add_float_resource_var(target=0.2, less_is_better=True)

    int_expr = dp.IntExpr(2)
    int_var = model.add_int_var(target=2)
    int_resource_var = model.add_int_resource_var(target=2, less_is_better=True)

    state = model.target_state

    cases: ClassVar = []

    for i, three in enumerate([three_expr, three_var, three_resource_var, 0.3]):
        for j, two in enumerate([two_expr, two_var, two_resource_var, 0.2]):
            cases.append(
                (dp.IntExpr(2) > dp.IntExpr(1), three, two, pytest.approx(0.3))
            )
            cases.append(
                (dp.IntExpr(2) < dp.IntExpr(1), three, two, pytest.approx(0.2))
            )

    for i, float_value in enumerate([three_expr, three_var, three_resource_var, 0.3]):
        for j, int_value in enumerate([int_expr, int_var, int_resource_var, 2]):
            cases.append(
                (
                    dp.IntExpr(2) > dp.IntExpr(1),
                    float_value,
                    int_value,
                    pytest.approx(0.3),
                )
            )
            cases.append(
                (
                    dp.IntExpr(2) < dp.IntExpr(1),
                    float_value,
                    int_value,
                    pytest.approx(2.0),
                )
            )

    @pytest.mark.parametrize("condition, lhs, rhs, expected", cases)
    def test(self, condition, lhs, rhs, expected):
        assert condition.if_then_else(lhs, rhs).eval(self.state, self.model) == expected


class TestIfThenElseError:
    model = dp.Model()
    obj = model.add_object_type(number=4)

    element_expr = dp.ElementExpr(1)
    element_var = model.add_element_var(object_type=obj, target=1)
    element_resource_var = model.add_element_resource_var(
        object_type=obj, target=1, less_is_better=True
    )

    set_const = model.create_set_const(object_type=obj, value=[1, 2])
    set_expr = dp.SetExpr(set_const)
    set_var = model.add_set_var(object_type=obj, target=[1, 2])

    int_expr = dp.IntExpr(1)
    int_var = model.add_int_var(target=1)
    int_resource_var = model.add_int_resource_var(target=1, less_is_better=True)

    float_expr = dp.FloatExpr(1.0)
    float_var = model.add_float_var(target=1.0)
    float_resource_var = model.add_float_resource_var(target=1.0, less_is_better=True)

    state = model.target_state

    cases: ClassVar = []

    for set_value in [set_const, set_expr, set_var]:
        for element_value in [element_expr, element_var, element_resource_var, 1]:
            cases.append((set_value, element_value))
            cases.append((element_value, set_value))

        for int_value in [int_expr, int_var, int_resource_var]:
            cases.append((set_value, int_value))
            cases.append((int_value, set_value))

        for float_value in [float_expr, float_var, float_resource_var, 1.0]:
            cases.append((set_value, float_value))
            cases.append((float_value, set_value))

    for element_value in [element_expr, element_var, element_resource_var]:
        cases.append((element_var, -1))

        for int_value in [int_expr, int_var, int_resource_var]:
            cases.append((element_value, int_value))
            cases.append((int_value, element_value))

        for float_value in [float_expr, float_var, float_resource_var, 1.0]:
            cases.append((element_value, float_value))
            cases.append((float_value, element_value))

    @pytest.mark.parametrize("lhs, rhs", cases)
    def test(self, lhs, rhs):
        with pytest.raises(TypeError):
            (dp.IntExpr(2) > dp.IntExpr(1)).if_then_else(lhs, rhs)
