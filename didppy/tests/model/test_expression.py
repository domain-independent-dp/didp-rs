import didppy as dp
import pytest


def test_element_expr():
    model = dp.Model()
    state = model.target_state

    assert dp.ElementExpr(3).eval(state, model) == 3


def test_element_expr_raise():
    with pytest.raises(OverflowError):
        dp.ElementExpr(-1)


def test_element_var():
    model = dp.Model()
    obj = model.add_object_type(number=4)
    var = model.add_element_var(object_type=obj, target=3)
    state = model.target_state

    assert state[var] == 3


def test_element_resource_var():
    model = dp.Model()
    obj = model.add_object_type(number=4)
    var = model.add_element_resource_var(object_type=obj, target=3, less_is_better=True)
    state = model.target_state

    assert state[var] == 3


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

    add_cases = []

    for i, lhs in enumerate([three_expr, three_var, three_resource_var, 3]):
        for j, rhs in enumerate([two_expr, two_var, two_resource_var, 2]):
            if i < 3 or j < 3:
                add_cases.append((lhs, rhs, 5))

    @pytest.mark.parametrize("lhs, rhs, expected", add_cases)
    def test_add(self, lhs, rhs, expected):
        assert (lhs + rhs).eval(self.state, self.model) == expected

    sub_cases = []

    for i, lhs in enumerate([three_expr, three_var, three_resource_var, 3]):
        for j, rhs in enumerate([two_expr, two_var, two_resource_var, 2]):
            if i < 3 or j < 3:
                sub_cases.append((lhs, rhs, 1))

    @pytest.mark.parametrize("lhs, rhs, expected", sub_cases)
    def test_sub(self, lhs, rhs, expected):
        assert (lhs - rhs).eval(self.state, self.model) == expected

    mul_cases = []

    for i, lhs in enumerate([three_expr, three_var, three_resource_var, 3]):
        for j, rhs in enumerate([two_expr, two_var, two_resource_var, 2]):
            if i < 3 or j < 3:
                mul_cases.append((lhs, rhs, 6))

    @pytest.mark.parametrize("lhs, rhs, expected", mul_cases)
    def test_mul(self, lhs, rhs, expected):
        assert (lhs * rhs).eval(self.state, self.model) == expected

    truediv_cases = []

    for i, lhs in enumerate([three_expr, three_var, three_resource_var, 3]):
        for j, rhs in enumerate([two_expr, two_var, two_resource_var, 2]):
            if i < 3 or j < 3:
                truediv_cases.append((lhs, rhs, 1))

    @pytest.mark.parametrize("lhs, rhs, expected", truediv_cases)
    def test_truediv(self, lhs, rhs, expected):
        assert (lhs / rhs).eval(self.state, self.model) == expected

    floordiv_cases = []

    for i, lhs in enumerate([three_expr, three_var, three_resource_var, 3]):
        for j, rhs in enumerate([two_expr, two_var, two_resource_var, 2]):
            if i < 3 or j < 3:
                floordiv_cases.append((lhs, rhs, 1))

    @pytest.mark.parametrize("lhs, rhs, expected", floordiv_cases)
    def test_floordiv(self, lhs, rhs, expected):
        assert (lhs // rhs).eval(self.state, self.model) == expected

    mod_cases = []

    for i, lhs in enumerate([three_expr, three_var, three_resource_var, 3]):
        for j, rhs in enumerate([two_expr, two_var, two_resource_var, 2]):
            if i < 3 or j < 3:
                mod_cases.append((lhs, rhs, 1))

    @pytest.mark.parametrize("lhs, rhs, expected", mod_cases)
    def test_mod(self, lhs, rhs, expected):
        assert (lhs % rhs).eval(self.state, self.model) == expected

    lt_cases = [
        (value, value, False) for value in [three_expr, three_var, three_resource_var]
    ]

    for i, lhs in enumerate([three_expr, three_var, three_resource_var, 3]):
        for j, rhs in enumerate([two_expr, two_var, two_resource_var, 2]):
            if i < 3 or j < 3:
                lt_cases.append((lhs, rhs, False))

    for i, lhs in enumerate([two_expr, two_var, two_resource_var, 2]):
        for j, rhs in enumerate([three_expr, three_var, three_resource_var, 3]):
            if i < 3 or j < 3:
                lt_cases.append((lhs, rhs, True))

    @pytest.mark.parametrize("lhs, rhs, expected", lt_cases)
    def test_lt(self, lhs, rhs, expected):
        assert (lhs < rhs).eval(self.state, self.model) == expected

    le_cases = [
        (value, value, True) for value in [three_expr, three_var, three_resource_var]
    ]

    for i, lhs in enumerate([three_expr, three_var, three_resource_var, 3]):
        for j, rhs in enumerate([two_expr, two_var, two_resource_var, 2]):
            if i < 3 or j < 3:
                le_cases.append((lhs, rhs, False))

    for i, lhs in enumerate([two_expr, two_var, two_resource_var, 2]):
        for j, rhs in enumerate([three_expr, three_var, three_resource_var, 3]):
            if i < 3 or j < 3:
                le_cases.append((lhs, rhs, True))

    @pytest.mark.parametrize("lhs, rhs, expected", le_cases)
    def test_le(self, lhs, rhs, expected):
        assert (lhs <= rhs).eval(self.state, self.model) == expected

    eq_cases = [
        (value, value, True) for value in [three_expr, three_var, three_resource_var]
    ]

    for i, lhs in enumerate([three_expr, three_var, three_resource_var, 3]):
        for j, rhs in enumerate([two_expr, two_var, two_resource_var, 2]):
            if i < 3 or j < 3:
                eq_cases.append((lhs, rhs, False))

    @pytest.mark.parametrize("lhs, rhs, expected", eq_cases)
    def test_eq(self, lhs, rhs, expected):
        assert (lhs == rhs).eval(self.state, self.model) == expected

    ne_cases = [
        (value, value, False) for value in [three_expr, three_var, three_resource_var]
    ]

    for i, lhs in enumerate([three_expr, three_var, three_resource_var, 3]):
        for j, rhs in enumerate([two_expr, two_var, two_resource_var, 2]):
            if i < 3 or j < 3:
                ne_cases.append((lhs, rhs, True))

    @pytest.mark.parametrize("lhs, rhs, expected", ne_cases)
    def test_ne(self, lhs, rhs, expected):
        assert (lhs != rhs).eval(self.state, self.model) == expected

    ge_cases = [
        (value, value, True) for value in [three_expr, three_var, three_resource_var]
    ]

    for i, lhs in enumerate([three_expr, three_var, three_resource_var, 3]):
        for j, rhs in enumerate([two_expr, two_var, two_resource_var, 2]):
            if i < 3 or j < 3:
                ge_cases.append((lhs, rhs, True))

    for i, lhs in enumerate([two_expr, two_var, two_resource_var, 2]):
        for j, rhs in enumerate([three_expr, three_var, three_resource_var, 3]):
            if i < 3 or j < 3:
                ge_cases.append((lhs, rhs, False))

    @pytest.mark.parametrize("lhs, rhs, expected", ge_cases)
    def test_ge(self, lhs, rhs, expected):
        assert (lhs >= rhs).eval(self.state, self.model) == expected

    gt_cases = [
        (value, value, True) for value in [three_expr, three_var, three_resource_var]
    ]

    for i, lhs in enumerate([three_expr, three_var, three_resource_var, 3]):
        for j, rhs in enumerate([two_expr, two_var, two_resource_var, 2]):
            if i < 3 or j < 3:
                gt_cases.append((lhs, rhs, True))

    for i, lhs in enumerate([two_expr, two_var, two_resource_var, 2]):
        for j, rhs in enumerate([three_expr, three_var, three_resource_var, 3]):
            if i < 3 or j < 3:
                gt_cases.append((lhs, rhs, False))

    @pytest.mark.parametrize("lhs, rhs, expected", gt_cases)
    def test_gt(self, lhs, rhs, expected):
        assert (lhs >= rhs).eval(self.state, self.model) == expected


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


@pytest.mark.parametrize("value, expected", set_eval_cases)
def test_set_const(value, expected):
    model = dp.Model()
    obj = model.add_object_type(number=4)
    const = model.create_set_const(object_type=obj, value=value)

    assert const.eval() == expected


@pytest.mark.parametrize("value, expected", set_eval_cases)
def test_set_var(value, expected):
    model = dp.Model()
    obj = model.add_object_type(number=4)
    var = model.add_set_var(object_type=obj, target=value)
    state = model.target_state

    assert state[var] == expected


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


class TestSetBinaryOperator:
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

    or_cases = []

    for lhs in [const, expr, var]:
        for rhs in [empty_const, empty_expr, empty_var]:
            or_cases.append((lhs, rhs, {0, 1}))

    for lhs in [const, expr, var]:
        for rhs in [overlap_const, overlap_expr, overlap_var]:
            or_cases.append((lhs, rhs, {0, 1, 2}))

    for lhs in [const, expr, var]:
        for rhs in [subset_const, subset_expr, subset_var]:
            or_cases.append((lhs, rhs, {0, 1}))

    for lhs in [const, expr, var]:
        for rhs in [disjoint_const, disjoint_expr, disjoint_var]:
            or_cases.append((lhs, rhs, {0, 1, 2, 3}))

    @pytest.mark.parametrize("lhs, rhs, expected", or_cases)
    def test_or(self, lhs, rhs, expected):
        assert (lhs | rhs).eval(self.state, self.model) == expected

    @pytest.mark.parametrize("lhs, rhs, expected", or_cases)
    def test_union(self, lhs, rhs, expected):
        assert lhs.union(rhs).eval(self.state, self.model) == expected

    sub_cases = []

    for lhs in [const, expr, var]:
        for rhs in [empty_const, empty_expr, empty_var]:
            sub_cases.append((lhs, rhs, {0, 1}))

    for lhs in [const, expr, var]:
        for rhs in [overlap_const, overlap_expr, overlap_var]:
            sub_cases.append((lhs, rhs, {0}))

    for lhs in [const, expr, var]:
        for rhs in [subset_const, subset_expr, subset_var]:
            sub_cases.append((lhs, rhs, {1}))

    for lhs in [const, expr, var]:
        for rhs in [disjoint_const, disjoint_expr, disjoint_var]:
            sub_cases.append((lhs, rhs, {0, 1}))

    @pytest.mark.parametrize("lhs, rhs, expected", sub_cases)
    def test_sub(self, lhs, rhs, expected):
        assert (lhs - rhs).eval(self.state, self.model) == expected

    @pytest.mark.parametrize("lhs, rhs, expected", sub_cases)
    def test_difference(self, lhs, rhs, expected):
        assert lhs.difference(rhs).eval(self.state, self.model) == expected

    and_cases = []

    for lhs in [const, expr, var]:
        for rhs in [empty_const, empty_expr, empty_var]:
            and_cases.append((lhs, rhs, set()))

    for lhs in [const, expr, var]:
        for rhs in [overlap_const, overlap_expr, overlap_var]:
            and_cases.append((lhs, rhs, {1}))

    for lhs in [const, expr, var]:
        for rhs in [subset_const, subset_expr, subset_var]:
            and_cases.append((lhs, rhs, {0}))

    for lhs in [const, expr, var]:
        for rhs in [disjoint_const, disjoint_expr, disjoint_var]:
            and_cases.append((lhs, rhs, set()))

    @pytest.mark.parametrize("lhs, rhs, expected", and_cases)
    def test_and(self, lhs, rhs, expected):
        assert (lhs & rhs).eval(self.state, self.model) == expected

    @pytest.mark.parametrize("lhs, rhs, expected", and_cases)
    def test_intersection(self, lhs, rhs, expected):
        assert lhs.intersection(rhs).eval(self.state, self.model) == expected

    xor_cases = []

    for lhs in [const, expr, var]:
        for rhs in [empty_const, empty_expr, empty_var]:
            xor_cases.append((lhs, rhs, {0, 1}))

    for lhs in [const, expr, var]:
        for rhs in [overlap_const, overlap_expr, overlap_var]:
            xor_cases.append((lhs, rhs, {0, 2}))

    for lhs in [const, expr, var]:
        for rhs in [subset_const, subset_expr, subset_var]:
            xor_cases.append((lhs, rhs, {1}))

    for lhs in [const, expr, var]:
        for rhs in [disjoint_const, disjoint_expr, disjoint_var]:
            xor_cases.append((lhs, rhs, {0, 1, 2, 3}))

    @pytest.mark.parametrize("lhs, rhs, expected", xor_cases)
    def test_xor(self, lhs, rhs, expected):
        assert (lhs ^ rhs).eval(self.state, self.model) == expected

    @pytest.mark.parametrize("lhs, rhs, expected", xor_cases)
    def test_symmetric_difference(self, lhs, rhs, expected):
        assert lhs.symmetric_difference(rhs).eval(self.state, self.model) == expected

    lt_cases = [(value, value, False) for value in [const, expr, var]]

    for lhs in [const, expr, var]:
        for rhs in [empty_const, empty_expr, empty_var]:
            lt_cases.append((lhs, rhs, False))

    for lhs in [const, expr, var]:
        for rhs in [overlap_const, overlap_expr, overlap_var]:
            lt_cases.append((lhs, rhs, False))

    for lhs in [const, expr, var]:
        for rhs in [subset_const, subset_expr, subset_var]:
            lt_cases.append((lhs, rhs, False))

    for lhs in [const, expr, var]:
        for rhs in [disjoint_const, disjoint_expr, disjoint_var]:
            lt_cases.append((lhs, rhs, False))

    for lhs in [empty_const, empty_expr, empty_var]:
        for rhs in [const, expr, var]:
            lt_cases.append((lhs, rhs, True))

    for lhs in [overlap_const, overlap_expr, overlap_var]:
        for rhs in [const, expr, var]:
            lt_cases.append((lhs, rhs, False))

    for lhs in [subset_const, subset_expr, subset_var]:
        for rhs in [const, expr, var]:
            lt_cases.append((lhs, rhs, True))

    for rhs in [disjoint_const, disjoint_expr, disjoint_var]:
        for lhs in [const, expr, var]:
            lt_cases.append((lhs, rhs, False))

    @pytest.mark.parametrize("lhs, rhs, expected", lt_cases)
    def test_lt(self, lhs, rhs, expected):
        assert (lhs < rhs).eval(self.state, self.model) == expected

    le_cases = [(value, value, True) for value in [const, expr, var]]

    for lhs in [const, expr, var]:
        for rhs in [empty_const, empty_expr, empty_var]:
            le_cases.append((lhs, rhs, False))

    for lhs in [const, expr, var]:
        for rhs in [overlap_const, overlap_expr, overlap_var]:
            le_cases.append((lhs, rhs, False))

    for lhs in [const, expr, var]:
        for rhs in [subset_const, subset_expr, subset_var]:
            le_cases.append((lhs, rhs, False))

    for lhs in [const, expr, var]:
        for rhs in [disjoint_const, disjoint_expr, disjoint_var]:
            le_cases.append((lhs, rhs, False))

    for lhs in [empty_const, empty_expr, empty_var]:
        for rhs in [const, expr, var]:
            le_cases.append((lhs, rhs, True))

    for lhs in [overlap_const, overlap_expr, overlap_var]:
        for rhs in [const, expr, var]:
            le_cases.append((lhs, rhs, False))

    for lhs in [subset_const, subset_expr, subset_var]:
        for rhs in [const, expr, var]:
            le_cases.append((lhs, rhs, True))

    for rhs in [disjoint_const, disjoint_expr, disjoint_var]:
        for lhs in [const, expr, var]:
            le_cases.append((lhs, rhs, False))

    @pytest.mark.parametrize("lhs, rhs, expected", le_cases)
    def test_le(self, lhs, rhs, expected):
        assert (lhs <= rhs).eval(self.state, self.model) == expected

    @pytest.mark.parametrize("lhs, rhs, expected", le_cases)
    def test_issubset(self, lhs, rhs, expected):
        assert lhs.issubset(rhs).eval(self.state, self.model) == expected

    eq_cases = [(value, value, True) for value in [const, expr, var]]

    for lhs in [const, expr, var]:
        for rhs in [empty_const, empty_expr, empty_var]:
            eq_cases.append((lhs, rhs, False))

    for lhs in [const, expr, var]:
        for rhs in [overlap_const, overlap_expr, overlap_var]:
            eq_cases.append((lhs, rhs, False))

    for lhs in [const, expr, var]:
        for rhs in [subset_const, subset_expr, subset_var]:
            eq_cases.append((lhs, rhs, False))

    for lhs in [const, expr, var]:
        for rhs in [disjoint_const, disjoint_expr, disjoint_var]:
            eq_cases.append((lhs, rhs, False))

    for lhs in [empty_const, empty_expr, empty_var]:
        for rhs in [const, expr, var]:
            eq_cases.append((lhs, rhs, False))

    for lhs in [overlap_const, overlap_expr, overlap_var]:
        for rhs in [const, expr, var]:
            eq_cases.append((lhs, rhs, False))

    for lhs in [subset_const, subset_expr, subset_var]:
        for rhs in [const, expr, var]:
            eq_cases.append((lhs, rhs, False))

    for rhs in [disjoint_const, disjoint_expr, disjoint_var]:
        for lhs in [const, expr, var]:
            eq_cases.append((lhs, rhs, False))

    @pytest.mark.parametrize("lhs, rhs, expected", eq_cases)
    def test_eq(self, lhs, rhs, expected):
        assert (lhs == rhs).eval(self.state, self.model) == expected

    ne_cases = [(value, value, False) for value in [const, expr, var]]

    for lhs in [const, expr, var]:
        for rhs in [empty_const, empty_expr, empty_var]:
            ne_cases.append((lhs, rhs, True))

    for lhs in [const, expr, var]:
        for rhs in [overlap_const, overlap_expr, overlap_var]:
            ne_cases.append((lhs, rhs, True))

    for lhs in [const, expr, var]:
        for rhs in [subset_const, subset_expr, subset_var]:
            ne_cases.append((lhs, rhs, True))

    for lhs in [const, expr, var]:
        for rhs in [disjoint_const, disjoint_expr, disjoint_var]:
            ne_cases.append((lhs, rhs, True))

    for lhs in [empty_const, empty_expr, empty_var]:
        for rhs in [const, expr, var]:
            ne_cases.append((lhs, rhs, True))

    for lhs in [overlap_const, overlap_expr, overlap_var]:
        for rhs in [const, expr, var]:
            ne_cases.append((lhs, rhs, True))

    for lhs in [subset_const, subset_expr, subset_var]:
        for rhs in [const, expr, var]:
            ne_cases.append((lhs, rhs, True))

    for rhs in [disjoint_const, disjoint_expr, disjoint_var]:
        for lhs in [const, expr, var]:
            ne_cases.append((lhs, rhs, True))

    @pytest.mark.parametrize("lhs, rhs, expected", ne_cases)
    def test_ne(self, lhs, rhs, expected):
        assert (lhs != rhs).eval(self.state, self.model) == expected

    ge_cases = [(value, value, True) for value in [const, expr, var]]

    for lhs in [const, expr, var]:
        for rhs in [empty_const, empty_expr, empty_var]:
            ge_cases.append((lhs, rhs, True))

    for lhs in [const, expr, var]:
        for rhs in [overlap_const, overlap_expr, overlap_var]:
            ge_cases.append((lhs, rhs, False))

    for lhs in [const, expr, var]:
        for rhs in [subset_const, subset_expr, subset_var]:
            ge_cases.append((lhs, rhs, True))

    for lhs in [const, expr, var]:
        for rhs in [disjoint_const, disjoint_expr, disjoint_var]:
            ge_cases.append((lhs, rhs, False))

    for lhs in [empty_const, empty_expr, empty_var]:
        for rhs in [const, expr, var]:
            ge_cases.append((lhs, rhs, False))

    for lhs in [overlap_const, overlap_expr, overlap_var]:
        for rhs in [const, expr, var]:
            ge_cases.append((lhs, rhs, False))

    for lhs in [subset_const, subset_expr, subset_var]:
        for rhs in [const, expr, var]:
            ge_cases.append((lhs, rhs, False))

    for rhs in [disjoint_const, disjoint_expr, disjoint_var]:
        for lhs in [const, expr, var]:
            ge_cases.append((lhs, rhs, False))

    @pytest.mark.parametrize("lhs, rhs, expected", ge_cases)
    def test_ge(self, lhs, rhs, expected):
        assert (lhs >= rhs).eval(self.state, self.model) == expected

    @pytest.mark.parametrize("lhs, rhs, expected", ge_cases)
    def test_issuperset(self, lhs, rhs, expected):
        assert lhs.issuperset(rhs).eval(self.state, self.model) == expected

    gt_cases = [(value, value, False) for value in [const, expr, var]]

    for lhs in [const, expr, var]:
        for rhs in [empty_const, empty_expr, empty_var]:
            gt_cases.append((lhs, rhs, True))

    for lhs in [const, expr, var]:
        for rhs in [overlap_const, overlap_expr, overlap_var]:
            gt_cases.append((lhs, rhs, False))

    for lhs in [const, expr, var]:
        for rhs in [subset_const, subset_expr, subset_var]:
            gt_cases.append((lhs, rhs, True))

    for lhs in [const, expr, var]:
        for rhs in [disjoint_const, disjoint_expr, disjoint_var]:
            gt_cases.append((lhs, rhs, False))

    for lhs in [empty_const, empty_expr, empty_var]:
        for rhs in [const, expr, var]:
            gt_cases.append((lhs, rhs, False))

    for lhs in [overlap_const, overlap_expr, overlap_var]:
        for rhs in [const, expr, var]:
            gt_cases.append((lhs, rhs, False))

    for lhs in [subset_const, subset_expr, subset_var]:
        for rhs in [const, expr, var]:
            gt_cases.append((lhs, rhs, False))

    for rhs in [disjoint_const, disjoint_expr, disjoint_var]:
        for lhs in [const, expr, var]:
            gt_cases.append((lhs, rhs, False))

    @pytest.mark.parametrize("lhs, rhs, expected", gt_cases)
    def test_gt(self, lhs, rhs, expected):
        assert (lhs > rhs).eval(self.state, self.model) == expected


class TestSetElementOperator:
    model = dp.Model()
    obj = model.add_object_type(number=4)

    const = model.create_set_const(object_type=obj, value=[0, 1])
    expr = dp.SetExpr(const)
    var = model.add_set_var(object_type=obj, target=[0, 1])

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

    add_cases = []

    for value in [const, expr, var]:
        for one in [one_expr, one_var, one_resource_var, 1]:
            add_cases.append((value, one, {0, 1}))

    for value in [const, expr, var]:
        for two in [two_expr, two_var, two_resource_var, 2]:
            add_cases.append((value, two, {0, 1, 2}))

    for empty in [empty_const, empty_expr, empty_var]:
        for one in [one_expr, one_var, one_resource_var, 1]:
            add_cases.append((empty, one, {1}))

    for empty in [empty_const, empty_expr, empty_var]:
        for two in [two_expr, two_var, two_resource_var, 2]:
            add_cases.append((empty, two, {2}))

    @pytest.mark.parametrize("value, element, expected", add_cases)
    def test_add(self, value, element, expected):
        assert value.add(element).eval(self.state, self.model) == expected

    discard_cases = []

    for value in [const, expr, var]:
        for one in [one_expr, one_var, one_resource_var, 1]:
            discard_cases.append((value, one, {0}))

    for value in [const, expr, var]:
        for two in [two_expr, two_var, two_resource_var, 2]:
            discard_cases.append((value, two, {0, 1}))

    for empty in [empty_const, empty_expr, empty_var]:
        for one in [one_expr, one_var, one_resource_var, 1]:
            discard_cases.append((empty, one, set()))

    for empty in [empty_const, empty_expr, empty_var]:
        for two in [two_expr, two_var, two_resource_var, 2]:
            discard_cases.append((empty, two, set()))

    @pytest.mark.parametrize("value, element, expected", discard_cases)
    def test_discard(self, value, element, expected):
        assert value.discard(element).eval(self.state, self.model) == expected

    @pytest.mark.parametrize("value, element, expected", discard_cases)
    def test_remove(self, value, element, expected):
        assert value.remove(element).eval(self.state, self.model) == expected

    contains_cases = []

    for value in [const, expr, var]:
        for one in [one_expr, one_var, one_resource_var, 1]:
            contains_cases.append((value, one, True))

    for value in [const, expr, var]:
        for two in [two_expr, two_var, two_resource_var, 2]:
            contains_cases.append((value, two, False))

    for empty in [empty_const, empty_expr, empty_var]:
        for one in [one_expr, one_var, one_resource_var, 1]:
            contains_cases.append((empty, one, False))

    for empty in [empty_const, empty_expr, empty_var]:
        for two in [two_expr, two_var, two_resource_var, 2]:
            contains_cases.append((empty, two, False))

    @pytest.mark.parametrize("value, element, expected", contains_cases)
    def test_contains(self, value, element, expected):
        assert value.contains(element).eval(self.state, self.model) == expected


def test_int_expr():
    model = dp.Model()
    state = model.target_state
    expr = dp.IntExpr(3)

    assert expr.eval(state, model) == 3


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

    add_cases = []

    for i, lhs in enumerate([three_expr, three_var, three_resource_var, 3]):
        for j, rhs in enumerate([two_expr, two_var, two_resource_var, 2]):
            if i < 3 or j < 3:
                add_cases.append((lhs, rhs, 5))

    @pytest.mark.parametrize("lhs, rhs, expected", add_cases)
    def test_add(self, lhs, rhs, expected):
        assert (lhs + rhs).eval(self.state, self.model) == expected

    sub_cases = []

    for i, lhs in enumerate([three_expr, three_var, three_resource_var, 3]):
        for j, rhs in enumerate([two_expr, two_var, two_resource_var, 2]):
            if i < 3 or j < 3:
                sub_cases.append((lhs, rhs, 1))

    @pytest.mark.parametrize("lhs, rhs, expected", sub_cases)
    def test_sub(self, lhs, rhs, expected):
        assert (lhs - rhs).eval(self.state, self.model) == expected

    mul_cases = []

    for i, lhs in enumerate([three_expr, three_var, three_resource_var, 3]):
        for j, rhs in enumerate([two_expr, two_var, two_resource_var, 2]):
            if i < 3 or j < 3:
                mul_cases.append((lhs, rhs, 6))

    @pytest.mark.parametrize("lhs, rhs, expected", mul_cases)
    def test_mul(self, lhs, rhs, expected):
        assert (lhs * rhs).eval(self.state, self.model) == expected

    mod_cases = []

    for i, lhs in enumerate([three_expr, three_var, three_resource_var, 3]):
        for j, rhs in enumerate([two_expr, two_var, two_resource_var, 2]):
            if i < 3 or j < 3:
                mod_cases.append((lhs, rhs, 1))

    @pytest.mark.parametrize("lhs, rhs, expected", mod_cases)
    def test_mod(self, lhs, rhs, expected):
        assert (lhs % rhs).eval(self.state, self.model) == expected

    truediv_cases = []

    for i, lhs in enumerate([three_expr, three_var, three_resource_var, 3]):
        for j, rhs in enumerate([two_expr, two_var, two_resource_var, 2]):
            if i < 3 or j < 3:
                truediv_cases.append((lhs, rhs, pytest.approx(1.5)))

    @pytest.mark.parametrize("lhs, rhs, expected", truediv_cases)
    def test_truediv(self, lhs, rhs, expected):
        assert (lhs / rhs).eval(self.state, self.model) == expected

    floordiv_cases = []

    for i, lhs in enumerate([three_expr, three_var, three_resource_var, 3]):
        for j, rhs in enumerate([two_expr, two_var, two_resource_var, 2]):
            if i < 3 or j < 3:
                floordiv_cases.append((lhs, rhs, 1))

    @pytest.mark.parametrize("lhs, rhs, expected", floordiv_cases)
    def test_floordiv(self, lhs, rhs, expected):
        assert (lhs // rhs).eval(self.state, self.model) == expected

    pow_cases = []

    for i, lhs in enumerate([three_expr, three_var, three_resource_var, 3]):
        for j, rhs in enumerate([two_expr, two_var, two_resource_var, 2]):
            if i < 3 or j < 3:
                pow_cases.append((lhs, rhs, pytest.approx(9.0)))

    @pytest.mark.parametrize("lhs, rhs, expected", pow_cases)
    def test_pow_op(self, lhs, rhs, expected):
        assert (lhs**rhs).eval(self.state, self.model) == expected

    @pytest.mark.parametrize("lhs, rhs, expected", pow_cases)
    def test_pow(self, lhs, rhs, expected):
        assert pow(lhs, rhs).eval(self.state, self.model) == expected

    pow_modulo_cases = []

    for i, lhs in enumerate([three_expr, three_var, three_resource_var, 3]):
        for j, rhs in enumerate([two_expr, two_var, two_resource_var, 2]):
            for k, modulo in enumerate(
                [modulo_expr, modulo_var, modulo_resource_var, 4.0]
            ):
                if i < 3 or j < 3:
                    pow_modulo_cases.append((lhs, rhs, modulo, pytest.approx(1.0)))

    @pytest.mark.parametrize("lhs, rhs, modulo, expected", pow_modulo_cases)
    def test_pow_modulo(self, lhs, rhs, modulo, expected):
        assert pow(lhs, rhs, modulo).eval(self.state, self.model) == expected

    lt_cases = [
        (value, value, False) for value in [three_expr, three_var, three_resource_var]
    ]

    for i, lhs in enumerate([three_expr, three_var, three_resource_var, 3]):
        for j, rhs in enumerate([two_expr, two_var, two_resource_var, 2]):
            if i < 3 or j < 3:
                lt_cases.append((lhs, rhs, False))

    for i, lhs in enumerate([two_expr, two_var, two_resource_var, 2]):
        for j, rhs in enumerate([three_expr, three_var, three_resource_var, 3]):
            if i < 3 or j < 3:
                lt_cases.append((lhs, rhs, True))

    @pytest.mark.parametrize("lhs, rhs, expected", lt_cases)
    def test_lt(self, lhs, rhs, expected):
        assert (lhs < rhs).eval(self.state, self.model) == expected

    le_cases = [
        (value, value, True) for value in [three_expr, three_var, three_resource_var]
    ]

    for i, lhs in enumerate([three_expr, three_var, three_resource_var, 3]):
        for j, rhs in enumerate([two_expr, two_var, two_resource_var, 2]):
            if i < 3 or j < 3:
                le_cases.append((lhs, rhs, False))

    for i, lhs in enumerate([two_expr, two_var, two_resource_var, 2]):
        for j, rhs in enumerate([three_expr, three_var, three_resource_var, 3]):
            if i < 3 or j < 3:
                le_cases.append((lhs, rhs, True))

    @pytest.mark.parametrize("lhs, rhs, expected", le_cases)
    def test_le(self, lhs, rhs, expected):
        assert (lhs <= rhs).eval(self.state, self.model) == expected

    eq_cases = [
        (value, value, True) for value in [three_expr, three_var, three_resource_var]
    ]

    for i, lhs in enumerate([three_expr, three_var, three_resource_var, 3]):
        for j, rhs in enumerate([two_expr, two_var, two_resource_var, 2]):
            if i < 3 or j < 3:
                eq_cases.append((lhs, rhs, False))

    @pytest.mark.parametrize("lhs, rhs, expected", eq_cases)
    def test_eq(self, lhs, rhs, expected):
        assert (lhs == rhs).eval(self.state, self.model) == expected

    ne_cases = [
        (value, value, False) for value in [three_expr, three_var, three_resource_var]
    ]

    for i, lhs in enumerate([three_expr, three_var, three_resource_var, 3]):
        for j, rhs in enumerate([two_expr, two_var, two_resource_var, 2]):
            if i < 3 or j < 3:
                ne_cases.append((lhs, rhs, True))

    @pytest.mark.parametrize("lhs, rhs, expected", ne_cases)
    def test_ne(self, lhs, rhs, expected):
        assert (lhs != rhs).eval(self.state, self.model) == expected

    ge_cases = [
        (value, value, True) for value in [three_expr, three_var, three_resource_var]
    ]

    for i, lhs in enumerate([three_expr, three_var, three_resource_var, 3]):
        for j, rhs in enumerate([two_expr, two_var, two_resource_var, 2]):
            if i < 3 or j < 3:
                ge_cases.append((lhs, rhs, True))

    for i, lhs in enumerate([two_expr, two_var, two_resource_var, 2]):
        for j, rhs in enumerate([three_expr, three_var, three_resource_var, 3]):
            if i < 3 or j < 3:
                ge_cases.append((lhs, rhs, False))

    @pytest.mark.parametrize("lhs, rhs, expected", ge_cases)
    def test_ge(self, lhs, rhs, expected):
        assert (lhs >= rhs).eval(self.state, self.model) == expected

    gt_cases = [
        (value, value, True) for value in [three_expr, three_var, three_resource_var]
    ]

    for i, lhs in enumerate([three_expr, three_var, three_resource_var, 3]):
        for j, rhs in enumerate([two_expr, two_var, two_resource_var, 2]):
            if i < 3 or j < 3:
                gt_cases.append((lhs, rhs, True))

    for i, lhs in enumerate([two_expr, two_var, two_resource_var, 2]):
        for j, rhs in enumerate([three_expr, three_var, three_resource_var, 3]):
            if i < 3 or j < 3:
                gt_cases.append((lhs, rhs, False))

    @pytest.mark.parametrize("lhs, rhs, expected", gt_cases)
    def test_gt(self, lhs, rhs, expected):
        assert (lhs >= rhs).eval(self.state, self.model) == expected


def test_float_expr():
    model = dp.Model()
    state = model.target_state
    expr = dp.FloatExpr(3.5)

    assert expr.eval(state, model) == pytest.approx(3.5)


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
