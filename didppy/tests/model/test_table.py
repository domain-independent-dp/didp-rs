import didppy as dp
import pytest


class TestElementTable1D:
    model = dp.Model()
    obj = model.add_object_type(number=2)

    zero_expr = dp.ElementExpr(0)
    zero_var = model.add_element_var(object_type=obj, target=0)
    zero_resource_var = model.add_element_resource_var(
        object_type=obj, target=0, less_is_better=True
    )

    one_expr = dp.ElementExpr(1)
    one_var = model.add_element_var(object_type=obj, target=1)
    one_resource_var = model.add_element_resource_var(
        object_type=obj, target=1, less_is_better=True
    )

    table = model.add_element_table([1, 0])
    state = model.target_state

    cases = [
        (zero_expr, 1),
        (zero_var, 1),
        (zero_resource_var, 1),
        (0, 1),
        (one_expr, 0),
        (one_var, 0),
        (one_resource_var, 0),
        (1, 0),
    ]

    @pytest.mark.parametrize("x, expected", cases)
    def test(self, x, expected):
        assert self.table[x].eval(self.state, self.model) == expected

    error_cases = [-1, dp.IntExpr(0), 2]

    @pytest.mark.parametrize("index", error_cases)
    def test_error(self, index):
        with pytest.raises(BaseException):
            self.table[index].eval(self.state, self.model)


class TestElementTable2D:
    model = dp.Model()
    obj = model.add_object_type(number=2)

    zero_expr = dp.ElementExpr(0)
    zero_var = model.add_element_var(object_type=obj, target=0)
    zero_resource_var = model.add_element_resource_var(
        object_type=obj, target=0, less_is_better=True
    )

    one_expr = dp.ElementExpr(1)
    one_var = model.add_element_var(object_type=obj, target=1)
    one_resource_var = model.add_element_resource_var(
        object_type=obj, target=1, less_is_better=True
    )

    table = model.add_element_table([[1, 0], [0, 1]])
    state = model.target_state

    cases = [
        (zero, zero, 1) for zero in [zero_expr, zero_var, zero_resource_var, 0]
    ] + [(one, one, 1) for one in [one_expr, one_var, one_resource_var, 1]]

    for zero in [zero_expr, zero_var, zero_resource_var, 0]:
        for one in [one_expr, one_var, one_resource_var, 1]:
            cases.append((zero, one, 0))
            cases.append((one, zero, 0))

    @pytest.mark.parametrize("x, y, expected", cases)
    def test(self, x, y, expected):
        assert self.table[x, y].eval(self.state, self.model) == expected

    error_cases = [
        0,
        (0, -1),
        (-1, 0),
        (0, dp.IntExpr(1)),
        (dp.IntExpr(1), 0),
        (0, 2),
        (2, 0),
    ]

    @pytest.mark.parametrize("index", error_cases)
    def test_error(self, index):
        with pytest.raises(BaseException):
            self.table[index].eval(self.state, self.model)


class TestElementTable3D:
    model = dp.Model()
    obj = model.add_object_type(number=3)

    zero_expr = dp.ElementExpr(0)
    zero_var = model.add_element_var(object_type=obj, target=0)
    zero_resource_var = model.add_element_resource_var(
        object_type=obj, target=0, less_is_better=True
    )

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

    table = model.add_element_table(
        [
            [[1, 0, 2], [2, 0, 1], [1, 2, 0]],
            [[0, 2, 1], [0, 1, 2], [1, 0, 2]],
            [[1, 2, 0], [0, 1, 2], [0, 2, 1]],
        ]
    )
    state = model.target_state

    cases = (
        [(zero, zero, zero, 1) for zero in [zero_expr, zero_var, zero_resource_var, 0]]
        + [(one, one, one, 1) for one in [one_expr, one_var, one_resource_var, 1]]
        + [(two, two, two, 1) for two in [two_expr, two_var, two_resource_var, 2]]
    )

    for zero in [zero_expr, zero_var, zero_resource_var, 0]:
        for one in [one_expr, one_var, one_resource_var, 1]:
            for two in [two_expr, two_var, two_resource_var, 2]:
                cases.append((zero, one, two, 1))
                cases.append((zero, two, one, 2))
                cases.append((one, zero, two, 1))
                cases.append((one, two, zero, 1))
                cases.append((two, zero, one, 2))
                cases.append((two, one, zero, 0))

    for zero in [zero_expr, zero_var, zero_resource_var, 0]:
        for one in [one_expr, one_var, one_resource_var, 1]:
            cases.append((zero, zero, one, 0))
            cases.append((zero, one, zero, 2))
            cases.append((zero, one, one, 0))
            cases.append((one, zero, zero, 0))
            cases.append((one, zero, one, 2))
            cases.append((one, one, zero, 0))

    for zero in [zero_expr, zero_var, zero_resource_var, 0]:
        for two in [two_expr, two_var, two_resource_var, 2]:
            cases.append((zero, zero, two, 2))
            cases.append((zero, two, zero, 1))
            cases.append((zero, two, two, 0))
            cases.append((two, zero, zero, 1))
            cases.append((two, zero, two, 0))
            cases.append((two, two, zero, 0))

    for one in [one_expr, one_var, one_resource_var, 1]:
        for two in [two_expr, two_var, two_resource_var, 2]:
            cases.append((one, one, two, 2))
            cases.append((one, two, one, 0))
            cases.append((one, two, two, 2))
            cases.append((two, one, one, 1))
            cases.append((two, one, two, 2))
            cases.append((two, two, one, 2))

    @pytest.mark.parametrize("x, y, z, expected", cases)
    def test(self, x, y, z, expected):
        assert self.table[x, y, z].eval(self.state, self.model) == expected

    error_cases = [
        0,
        (0, 1),
        (-1, 0, 0),
        (0, -1, 0),
        (0, 0, -1),
        (dp.IntExpr(1), 0, 0),
        (0, dp.IntExpr(1), 0),
        (0, 0, dp.IntExpr(1)),
        (3, 0, 0),
        (0, 3, 0),
        (0, 0, 3),
    ]

    @pytest.mark.parametrize("index", error_cases)
    def test_error(self, index):
        with pytest.raises(BaseException):
            self.table[index].eval(self.state, self.model)


class TestElementTable:
    model = dp.Model()
    obj = model.add_object_type(number=3)

    zero_expr = dp.ElementExpr(0)
    zero_var = model.add_element_var(object_type=obj, target=0)
    zero_resource_var = model.add_element_resource_var(
        object_type=obj, target=0, less_is_better=True
    )

    one_expr = dp.ElementExpr(1)
    one_var = model.add_element_var(object_type=obj, target=1)
    one_resource_var = model.add_element_resource_var(
        object_type=obj, target=1, less_is_better=True
    )

    table = model.add_element_table({(0, 0, 0, 0): 1, (1, 1, 1, 1): 0}, default=2)
    state = model.target_state

    cases = (
        [
            ((zero, zero, zero, zero), 1)
            for zero in [zero_expr, zero_var, zero_resource_var, 0]
        ]
        + [
            ((one, one, one, one), 0)
            for one in [one_expr, one_var, one_resource_var, 1]
        ]
        + [((0, 1, 0, 1), 2)]
    )

    @pytest.mark.parametrize("index, expected", cases)
    def test(self, index, expected):
        assert self.table[index].eval(self.state, self.model) == expected

    error_cases = [
        (-1, 0, 0),
        (0, -1, 0),
        (0, 0, -1),
        (dp.IntExpr(1), 0, 0),
        (0, dp.IntExpr(1), 0),
        (0, 0, dp.IntExpr(1)),
    ]

    @pytest.mark.parametrize("index", error_cases)
    def test_error(self, index):
        with pytest.raises(BaseException):
            self.table[index].eval(self.state, self.model)


class TestSetTable1D:
    model = dp.Model()
    obj = model.add_object_type(number=2)

    zero_expr = dp.ElementExpr(0)
    zero_var = model.add_element_var(object_type=obj, target=0)
    zero_resource_var = model.add_element_resource_var(
        object_type=obj, target=0, less_is_better=True
    )

    one_expr = dp.ElementExpr(1)
    one_var = model.add_element_var(object_type=obj, target=1)
    one_resource_var = model.add_element_resource_var(
        object_type=obj, target=1, less_is_better=True
    )

    set_const = model.create_set_const(object_type=obj, value=[0, 1])
    set_expr = dp.SetExpr(set_const)
    set_var = model.add_set_var(object_type=obj, target=[0, 1])

    table = model.add_set_table([[1], [0]], object_type=obj)
    state = model.target_state

    cases = [
        (zero_expr, {1}),
        (zero_var, {1}),
        (zero_resource_var, {1}),
        (0, {1}),
        (one_expr, {0}),
        (one_var, {0}),
        (one_resource_var, {0}),
        (1, {0}),
    ]

    @pytest.mark.parametrize("x, expected", cases)
    def test(self, x, expected):
        assert self.table[x].eval(self.state, self.model) == expected

    union_cases = [
        (set_const, {0, 1}),
        (set_expr, {0, 1}),
        (set_var, {0, 1}),
    ]

    @pytest.mark.parametrize("x, expected", union_cases)
    def test_union(self, x, expected):
        assert self.table.union(x).eval(self.state, self.model) == expected

    intersection_cases = [
        (set_const, set()),
        (set_expr, set()),
        (set_var, set()),
    ]

    @pytest.mark.parametrize("x, expected", intersection_cases)
    def test_intersection(self, x, expected):
        assert self.table.intersection(x).eval(self.state, self.model) == expected

    symmetric_difference_cases = [
        (set_const, {0, 1}),
        (set_expr, {0, 1}),
        (set_var, {0, 1}),
    ]

    @pytest.mark.parametrize("x, expected", symmetric_difference_cases)
    def test_symmetric_difference(self, x, expected):
        assert (
            self.table.symmetric_difference(x).eval(self.state, self.model) == expected
        )

    error_cases = [-1, dp.IntExpr(0), 2]

    @pytest.mark.parametrize("index", error_cases)
    def test_error(self, index):
        with pytest.raises(BaseException):
            self.table[index].eval(self.state, self.model)


class TestSetTable2D:
    model = dp.Model()
    obj = model.add_object_type(number=2)

    zero_expr = dp.ElementExpr(0)
    zero_var = model.add_element_var(object_type=obj, target=0)
    zero_resource_var = model.add_element_resource_var(
        object_type=obj, target=0, less_is_better=True
    )

    one_expr = dp.ElementExpr(1)
    one_var = model.add_element_var(object_type=obj, target=1)
    one_resource_var = model.add_element_resource_var(
        object_type=obj, target=1, less_is_better=True
    )

    set_const = model.create_set_const(object_type=obj, value=[0, 1])
    set_expr = dp.SetExpr(set_const)
    set_var = model.add_set_var(object_type=obj, target=[0, 1])

    table = model.add_set_table([[[1], [0]], [[0], [1]]], object_type=obj)
    state = model.target_state

    cases = [
        (zero, zero, {1}) for zero in [zero_expr, zero_var, zero_resource_var, 0]
    ] + [(one, one, {1}) for one in [one_expr, one_var, one_resource_var, 1]]

    for zero in [zero_expr, zero_var, zero_resource_var, 0]:
        for one in [one_expr, one_var, one_resource_var, 1]:
            cases.append((zero, one, {0}))
            cases.append((one, zero, {0}))

    @pytest.mark.parametrize("x, y, expected", cases)
    def test(self, x, y, expected):
        assert self.table[x, y].eval(self.state, self.model) == expected

    union_cases = []

    for x in [set_const, set_expr, set_var]:
        for y in [set_const, set_expr, set_var]:
            union_cases.append((x, y, {0, 1}))

    for value in [set_const, set_expr, set_var]:
        for zero in [zero_expr, zero_var, zero_resource_var, 0]:
            union_cases.append((value, zero, {0, 1}))
            union_cases.append((zero, value, {0, 1}))

    @pytest.mark.parametrize("x, y, expected", union_cases)
    def test_union(self, x, y, expected):
        assert self.table.union(x, y).eval(self.state, self.model) == expected

    intersection_cases = []

    for x in [set_const, set_expr, set_var]:
        for y in [set_const, set_expr, set_var]:
            intersection_cases.append((x, y, set()))

    for value in [set_const, set_expr, set_var]:
        for zero in [zero_expr, zero_var, zero_resource_var, 0]:
            intersection_cases.append((value, zero, set()))
            intersection_cases.append((zero, value, set()))

    @pytest.mark.parametrize("x, y, expected", intersection_cases)
    def test_intersection(self, x, y, expected):
        assert self.table.intersection(x, y).eval(self.state, self.model) == expected

    symmetric_difference_cases = []

    for x in [set_const, set_expr, set_var]:
        for y in [set_const, set_expr, set_var]:
            symmetric_difference_cases.append((x, y, set()))

    for value in [set_const, set_expr, set_var]:
        for zero in [zero_expr, zero_var, zero_resource_var, 0]:
            symmetric_difference_cases.append((value, zero, {0, 1}))
            symmetric_difference_cases.append((zero, value, {0, 1}))

    @pytest.mark.parametrize("x, y, expected", symmetric_difference_cases)
    def test_symmetric_difference(self, x, y, expected):
        assert (
            self.table.symmetric_difference(x, y).eval(self.state, self.model)
            == expected
        )

    error_cases = [
        0,
        (-1, 0),
        (0, -1),
        (dp.IntExpr(0), 0),
        (0, dp.IntExpr(2)),
        (2, 0),
        (0, 2),
    ]

    @pytest.mark.parametrize("index", error_cases)
    def test_error(self, index):
        with pytest.raises(BaseException):
            self.table[index].eval(self.state, self.model)


class TestSetTable3D:
    model = dp.Model()
    obj = model.add_object_type(number=3)

    zero_expr = dp.ElementExpr(0)
    zero_var = model.add_element_var(object_type=obj, target=0)
    zero_resource_var = model.add_element_resource_var(
        object_type=obj, target=0, less_is_better=True
    )

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

    set_const = model.create_set_const(object_type=obj, value=[0, 1])
    set_expr = dp.SetExpr(set_const)
    set_var = model.add_set_var(object_type=obj, target=[0, 1])

    table = model.add_set_table(
        [
            [[[1], [0], [2]], [[2], [0], [1]], [[1], [2], [0]]],
            [[[0], [2], [1]], [[0], [1], [2]], [[1], [0], [2]]],
            [[[1], [2], [0]], [[0], [1], [2]], [[0], [2], [1]]],
        ],
        object_type=obj,
    )
    state = model.target_state

    cases = (
        [
            (zero, zero, zero, {1})
            for zero in [zero_expr, zero_var, zero_resource_var, 0]
        ]
        + [(one, one, one, {1}) for one in [one_expr, one_var, one_resource_var, 1]]
        + [(two, two, two, {1}) for two in [two_expr, two_var, two_resource_var, 2]]
    )

    for zero in [zero_expr, zero_var, zero_resource_var, 0]:
        for one in [one_expr, one_var, one_resource_var, 1]:
            for two in [two_expr, two_var, two_resource_var, 2]:
                cases.append((zero, one, two, {1}))
                cases.append((zero, two, one, {2}))
                cases.append((one, zero, two, {1}))
                cases.append((one, two, zero, {1}))
                cases.append((two, zero, one, {2}))
                cases.append((two, one, zero, {0}))

    for zero in [zero_expr, zero_var, zero_resource_var, 0]:
        for one in [one_expr, one_var, one_resource_var, 1]:
            cases.append((zero, zero, one, {0}))
            cases.append((zero, one, zero, {2}))
            cases.append((zero, one, one, {0}))
            cases.append((one, zero, zero, {0}))
            cases.append((one, zero, one, {2}))
            cases.append((one, one, zero, {0}))

    for zero in [zero_expr, zero_var, zero_resource_var, 0]:
        for two in [two_expr, two_var, two_resource_var, 2]:
            cases.append((zero, zero, two, {2}))
            cases.append((zero, two, zero, {1}))
            cases.append((zero, two, two, {0}))
            cases.append((two, zero, zero, {1}))
            cases.append((two, zero, two, {0}))
            cases.append((two, two, zero, {0}))

    for one in [one_expr, one_var, one_resource_var, 1]:
        for two in [two_expr, two_var, two_resource_var, 2]:
            cases.append((one, one, two, {2}))
            cases.append((one, two, one, {0}))
            cases.append((one, two, two, {2}))
            cases.append((two, one, one, {1}))
            cases.append((two, one, two, {2}))
            cases.append((two, two, one, {2}))

    @pytest.mark.parametrize("x, y, z, expected", cases)
    def test(self, x, y, z, expected):
        assert self.table[x, y, z].eval(self.state, self.model) == expected

    union_cases = []

    for x in [set_const, set_expr, set_var]:
        for y in [set_const, set_expr, set_var]:
            for z in [set_const, set_expr, set_var]:
                union_cases.append((x, y, z, {0, 1, 2}))

    for value in [set_const, set_expr, set_var]:
        for zero in [zero_expr, zero_var, zero_resource_var, 0]:
            union_cases.append((value, value, zero, {0, 1, 2}))
            union_cases.append((value, zero, value, {0, 1, 2}))
            union_cases.append((value, zero, zero, {0, 1}))
            union_cases.append((zero, value, value, {0, 1, 2}))
            union_cases.append((zero, value, zero, {1, 2}))
            union_cases.append((zero, zero, value, {0, 1}))

    @pytest.mark.parametrize("x, y, z, expected", union_cases)
    def test_union(self, x, y, z, expected):
        assert self.table.union(x, y, z).eval(self.state, self.model) == expected

    intersection_cases = []

    for x in [set_const, set_expr, set_var]:
        for y in [set_const, set_expr, set_var]:
            for z in [set_const, set_expr, set_var]:
                intersection_cases.append((x, y, z, set()))

    for value in [set_const, set_expr, set_var]:
        for two in [two_expr, two_var, two_resource_var, 2]:
            for zero in [zero_expr, zero_var, zero_resource_var, 0]:
                intersection_cases.append((value, two, zero, {1}))

    for value in [set_const, set_expr, set_var]:
        for zero in [zero_expr, zero_var, zero_resource_var, 0]:
            intersection_cases.append((value, value, zero, set()))
            intersection_cases.append((value, zero, value, set()))
            intersection_cases.append((value, zero, zero, set()))
            intersection_cases.append((zero, value, value, set()))
            intersection_cases.append((zero, value, zero, set()))
            intersection_cases.append((zero, zero, value, set()))

    @pytest.mark.parametrize("x, y, z, expected", intersection_cases)
    def test_intersection(self, x, y, z, expected):
        assert self.table.intersection(x, y, z).eval(self.state, self.model) == expected

    symmetric_difference_cases = []

    for x in [set_const, set_expr, set_var]:
        for y in [set_const, set_expr, set_var]:
            for z in [set_const, set_expr, set_var]:
                symmetric_difference_cases.append((x, y, z, set()))

    for value in [set_const, set_expr, set_var]:
        for zero in [zero_expr, zero_var, zero_resource_var, 0]:
            symmetric_difference_cases.append((value, value, zero, {1, 2}))
            symmetric_difference_cases.append((value, zero, value, {1, 2}))
            symmetric_difference_cases.append((value, zero, zero, {0, 1}))
            symmetric_difference_cases.append((zero, value, value, {1, 2}))
            symmetric_difference_cases.append((zero, value, zero, {1, 2}))
            symmetric_difference_cases.append((zero, zero, value, {0, 1}))

    @pytest.mark.parametrize("x, y, z, expected", symmetric_difference_cases)
    def test_symmetric_difference(self, x, y, z, expected):
        assert (
            self.table.symmetric_difference(x, y, z).eval(self.state, self.model)
            == expected
        )

    error_cases = [
        0,
        (0, 1),
        (-1, 0, 0),
        (0, -1, 0),
        (0, 0, -1),
        (dp.IntExpr(1), 0, 0),
        (0, dp.IntExpr(1), 0),
        (0, 0, dp.IntExpr(1)),
        (3, 0, 0),
        (0, 3, 0),
        (0, 0, 3),
    ]

    @pytest.mark.parametrize("index", error_cases)
    def test_error(self, index):
        with pytest.raises(BaseException):
            self.table[index].eval(self.state, self.model)


class TestSetTable:
    model = dp.Model()
    obj = model.add_object_type(number=3)

    zero_expr = dp.ElementExpr(0)
    zero_var = model.add_element_var(object_type=obj, target=0)
    zero_resource_var = model.add_element_resource_var(
        object_type=obj, target=0, less_is_better=True
    )

    one_expr = dp.ElementExpr(1)
    one_var = model.add_element_var(object_type=obj, target=1)
    one_resource_var = model.add_element_resource_var(
        object_type=obj, target=1, less_is_better=True
    )

    set_const = model.create_set_const(object_type=obj, value=[0, 1])
    set_expr = dp.SetExpr(set_const)
    set_var = model.add_set_var(object_type=obj, target=[0, 1])

    table = model.add_set_table(
        {(0, 0, 0, 0): [0, 1], (1, 1, 1, 1): [1, 2]}, default=[0, 2], object_type=obj
    )
    state = model.target_state

    cases = (
        [
            ((zero, zero, zero, zero), {0, 1})
            for zero in [zero_expr, zero_var, zero_resource_var, 0]
        ]
        + [
            ((one, one, one, one), {1, 2})
            for one in [one_expr, one_var, one_resource_var, 1]
        ]
        + [((0, 1, 0, 1), {0, 2})]
    )

    @pytest.mark.parametrize("index, expected", cases)
    def test(self, index, expected):
        assert self.table[index].eval(self.state, self.model) == expected

    union_cases = [
        ((value, value, value, value), {0, 1, 2})
        for value in [set_const, set_expr, set_var]
    ]

    for value in [set_const, set_expr, set_var]:
        for zero in [zero_expr, zero_var, zero_resource_var, 0]:
            union_cases.append(((zero, zero, zero, value), {0, 1, 2}))

    @pytest.mark.parametrize("index, expected", union_cases)
    def test_union(self, index, expected):
        assert self.table.union(index).eval(self.state, self.model) == expected

    intersection_cases = [
        ((value, value, value, value), set())
        for value in [set_const, set_expr, set_var]
    ]

    for value in [set_const, set_expr, set_var]:
        for zero in [zero_expr, zero_var, zero_resource_var, 0]:
            intersection_cases.append(((zero, zero, zero, value), {0}))

    @pytest.mark.parametrize("index, expected", intersection_cases)
    def test_intersection(self, index, expected):
        assert self.table.intersection(index).eval(self.state, self.model) == expected

    symmetric_difference_cases = [
        ((value, value, value, value), {0, 2})
        for value in [set_const, set_expr, set_var]
    ]

    for value in [set_const, set_expr, set_var]:
        for zero in [zero_expr, zero_var, zero_resource_var, 0]:
            symmetric_difference_cases.append(((zero, zero, zero, value), {1, 2}))

    @pytest.mark.parametrize("index, expected", symmetric_difference_cases)
    def test_symmetric_difference(self, index, expected):
        assert (
            self.table.symmetric_difference(index).eval(self.state, self.model)
            == expected
        )

    error_cases = [
        (-1, 0, 0),
        (0, -1, 0),
        (0, 0, -1),
        (dp.IntExpr(1), 0, 0),
        (0, dp.IntExpr(1), 0),
        (0, 0, dp.IntExpr(1)),
    ]

    @pytest.mark.parametrize("index", error_cases)
    def test_error(self, index):
        with pytest.raises(BaseException):
            self.table[index].eval(self.state, self.model)


class TestBoolTable1D:
    model = dp.Model()
    obj = model.add_object_type(number=2)

    zero_expr = dp.ElementExpr(0)
    zero_var = model.add_element_var(object_type=obj, target=0)
    zero_resource_var = model.add_element_resource_var(
        object_type=obj, target=0, less_is_better=True
    )

    one_expr = dp.ElementExpr(1)
    one_var = model.add_element_var(object_type=obj, target=1)
    one_resource_var = model.add_element_resource_var(
        object_type=obj, target=1, less_is_better=True
    )

    table = model.add_bool_table([True, False])
    state = model.target_state

    cases = [
        (zero_expr, True),
        (zero_var, True),
        (zero_resource_var, True),
        (0, True),
        (one_expr, False),
        (one_var, False),
        (one_resource_var, False),
        (1, False),
    ]

    @pytest.mark.parametrize("x, expected", cases)
    def test(self, x, expected):
        assert self.table[x].eval(self.state, self.model) == expected

    error_cases = [-1, dp.IntExpr(0), 2]

    @pytest.mark.parametrize("index", error_cases)
    def test_error(self, index):
        with pytest.raises(BaseException):
            self.table[index].eval(self.state, self.model)


class TestBoolTable2D:
    model = dp.Model()
    obj = model.add_object_type(number=2)

    zero_expr = dp.ElementExpr(0)
    zero_var = model.add_element_var(object_type=obj, target=0)
    zero_resource_var = model.add_element_resource_var(
        object_type=obj, target=0, less_is_better=True
    )

    one_expr = dp.ElementExpr(1)
    one_var = model.add_element_var(object_type=obj, target=1)
    one_resource_var = model.add_element_resource_var(
        object_type=obj, target=1, less_is_better=True
    )

    table = model.add_bool_table([[True, False], [False, True]])
    state = model.target_state

    cases = [
        (zero, zero, True) for zero in [zero_expr, zero_var, zero_resource_var, 0]
    ] + [(one, one, True) for one in [one_expr, one_var, one_resource_var, 1]]

    for zero in [zero_expr, zero_var, zero_resource_var, 0]:
        for one in [one_expr, one_var, one_resource_var, 1]:
            cases.append((zero, one, False))
            cases.append((one, zero, False))

    @pytest.mark.parametrize("x, y, expected", cases)
    def test(self, x, y, expected):
        assert self.table[x, y].eval(self.state, self.model) == expected

    error_cases = [
        0,
        (0, -1),
        (-1, 0),
        (0, dp.IntExpr(1)),
        (dp.IntExpr(1), 0),
        (0, 2),
        (2, 0),
    ]

    @pytest.mark.parametrize("index", error_cases)
    def test_error(self, index):
        with pytest.raises(BaseException):
            self.table[index].eval(self.state, self.model)


class TestBoolTable3D:
    model = dp.Model()
    obj = model.add_object_type(number=3)

    zero_expr = dp.ElementExpr(0)
    zero_var = model.add_element_var(object_type=obj, target=0)
    zero_resource_var = model.add_element_resource_var(
        object_type=obj, target=0, less_is_better=True
    )

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

    table = model.add_bool_table(
        [
            [[True, False, True], [True, False, True], [True, True, False]],
            [[False, True, True], [False, True, True], [True, False, True]],
            [[True, True, False], [False, True, True], [False, True, True]],
        ]
    )
    state = model.target_state

    cases = (
        [(zero, zero, zero, 1) for zero in [zero_expr, zero_var, zero_resource_var, 0]]
        + [(one, one, one, 1) for one in [one_expr, one_var, one_resource_var, 1]]
        + [(two, two, two, 1) for two in [two_expr, two_var, two_resource_var, 2]]
    )

    for zero in [zero_expr, zero_var, zero_resource_var, 0]:
        for one in [one_expr, one_var, one_resource_var, 1]:
            for two in [two_expr, two_var, two_resource_var, 2]:
                cases.append((zero, one, two, True))
                cases.append((zero, two, one, True))
                cases.append((one, zero, two, True))
                cases.append((one, two, zero, True))
                cases.append((two, zero, one, True))
                cases.append((two, one, zero, False))

    for zero in [zero_expr, zero_var, zero_resource_var, 0]:
        for one in [one_expr, one_var, one_resource_var, 1]:
            cases.append((zero, zero, one, False))
            cases.append((zero, one, zero, True))
            cases.append((zero, one, one, False))
            cases.append((one, zero, zero, False))
            cases.append((one, zero, one, True))
            cases.append((one, one, zero, False))

    for zero in [zero_expr, zero_var, zero_resource_var, 0]:
        for two in [two_expr, two_var, two_resource_var, 2]:
            cases.append((zero, zero, two, True))
            cases.append((zero, two, zero, True))
            cases.append((zero, two, two, False))
            cases.append((two, zero, zero, True))
            cases.append((two, zero, two, False))
            cases.append((two, two, zero, False))

    for one in [one_expr, one_var, one_resource_var, 1]:
        for two in [two_expr, two_var, two_resource_var, 2]:
            cases.append((one, one, two, True))
            cases.append((one, two, one, False))
            cases.append((one, two, two, True))
            cases.append((two, one, one, True))
            cases.append((two, one, two, True))
            cases.append((two, two, one, True))

    @pytest.mark.parametrize("x, y, z, expected", cases)
    def test(self, x, y, z, expected):
        assert self.table[x, y, z].eval(self.state, self.model) == expected

    error_cases = [
        0,
        (0, 1),
        (-1, 0, 0),
        (0, -1, 0),
        (0, 0, -1),
        (dp.IntExpr(1), 0, 0),
        (0, dp.IntExpr(1), 0),
        (0, 0, dp.IntExpr(1)),
        (3, 0, 0),
        (0, 3, 0),
        (0, 0, 3),
    ]

    @pytest.mark.parametrize("index", error_cases)
    def test_error(self, index):
        with pytest.raises(BaseException):
            self.table[index].eval(self.state, self.model)


class TestBoolTable:
    model = dp.Model()
    obj = model.add_object_type(number=3)

    zero_expr = dp.ElementExpr(0)
    zero_var = model.add_element_var(object_type=obj, target=0)
    zero_resource_var = model.add_element_resource_var(
        object_type=obj, target=0, less_is_better=True
    )

    one_expr = dp.ElementExpr(1)
    one_var = model.add_element_var(object_type=obj, target=1)
    one_resource_var = model.add_element_resource_var(
        object_type=obj, target=1, less_is_better=True
    )

    table = model.add_bool_table(
        {(0, 0, 0, 0): True, (1, 1, 1, 1): False}, default=True
    )
    state = model.target_state

    cases = (
        [
            ((zero, zero, zero, zero), True)
            for zero in [zero_expr, zero_var, zero_resource_var, 0]
        ]
        + [
            ((one, one, one, one), False)
            for one in [one_expr, one_var, one_resource_var, 1]
        ]
        + [((0, 1, 0, 1), True)]
    )

    @pytest.mark.parametrize("index, expected", cases)
    def test(self, index, expected):
        assert self.table[index].eval(self.state, self.model) == expected

    error_cases = [
        (-1, 0, 0),
        (0, -1, 0),
        (0, 0, -1),
        (dp.IntExpr(1), 0, 0),
        (0, dp.IntExpr(1), 0),
        (0, 0, dp.IntExpr(1)),
    ]

    @pytest.mark.parametrize("index", error_cases)
    def test_error(self, index):
        with pytest.raises(BaseException):
            self.table[index].eval(self.state, self.model)


class TestIntTable1D:
    model = dp.Model()
    obj = model.add_object_type(number=2)

    zero_expr = dp.ElementExpr(0)
    zero_var = model.add_element_var(object_type=obj, target=0)
    zero_resource_var = model.add_element_resource_var(
        object_type=obj, target=0, less_is_better=True
    )

    one_expr = dp.ElementExpr(1)
    one_var = model.add_element_var(object_type=obj, target=1)
    one_resource_var = model.add_element_resource_var(
        object_type=obj, target=1, less_is_better=True
    )

    set_const = model.create_set_const(object_type=obj, value=[0, 1])
    set_expr = dp.SetExpr(set_const)
    set_var = model.add_set_var(object_type=obj, target=[0, 1])

    table = model.add_int_table([3, 2])
    state = model.target_state

    cases = [
        (zero_expr, 3),
        (zero_var, 3),
        (zero_resource_var, 3),
        (0, 3),
        (one_expr, 2),
        (one_var, 2),
        (one_resource_var, 2),
        (1, 2),
        (set_const, 5),
        (set_expr, 5),
        (set_var, 5),
    ]

    @pytest.mark.parametrize("x, expected", cases)
    def test(self, x, expected):
        assert self.table[x].eval(self.state, self.model) == expected

    product_cases = [(set_const, 6), (set_expr, 6), (set_var, 6)]

    @pytest.mark.parametrize("x, expected", product_cases)
    def test_product(self, x, expected):
        assert self.table.product(x).eval(self.state, self.model) == expected

    max_cases = [(set_const, 3), (set_expr, 3), (set_var, 3)]

    @pytest.mark.parametrize("x, expected", max_cases)
    def test_max(self, x, expected):
        assert self.table.max(x).eval(self.state, self.model) == expected

    min_cases = [(set_const, 2), (set_expr, 2), (set_var, 2)]

    @pytest.mark.parametrize("x, expected", min_cases)
    def test_min(self, x, expected):
        assert self.table.min(x).eval(self.state, self.model) == expected

    error_cases = [-1, dp.IntExpr(0), 2]

    @pytest.mark.parametrize("index", error_cases)
    def test_error(self, index):
        with pytest.raises(BaseException):
            self.table[index].eval(self.state, self.model)


class TestIntTable2D:
    model = dp.Model()
    obj = model.add_object_type(number=2)

    zero_expr = dp.ElementExpr(0)
    zero_var = model.add_element_var(object_type=obj, target=0)
    zero_resource_var = model.add_element_resource_var(
        object_type=obj, target=0, less_is_better=True
    )

    one_expr = dp.ElementExpr(1)
    one_var = model.add_element_var(object_type=obj, target=1)
    one_resource_var = model.add_element_resource_var(
        object_type=obj, target=1, less_is_better=True
    )

    set_const = model.create_set_const(object_type=obj, value=[0, 1])
    set_expr = dp.SetExpr(set_const)
    set_var = model.add_set_var(object_type=obj, target=[0, 1])

    table = model.add_int_table([[3, 2], [2, 3]])
    state = model.target_state

    cases = [
        (zero, zero, 3) for zero in [zero_expr, zero_var, zero_resource_var, 0]
    ] + [(one, one, 3) for one in [one_expr, one_var, one_resource_var, 1]]

    for x in [set_const, set_expr, set_var]:
        for y in [set_const, set_expr, set_var]:
            cases.append((x, y, 10))

    for zero in [zero_expr, zero_var, zero_resource_var, 0]:
        for one in [one_expr, one_var, one_resource_var, 1]:
            cases.append((zero, one, 2))
            cases.append((one, zero, 2))

    for value in [set_const, set_expr, set_var]:
        for zero in [zero_expr, zero_var, zero_resource_var, 0]:
            cases.append((value, zero, 5))
            cases.append((zero, value, 5))

    @pytest.mark.parametrize("x, y, expected", cases)
    def test(self, x, y, expected):
        assert self.table[x, y].eval(self.state, self.model) == expected

    product_cases = []

    for x in [set_const, set_expr, set_var]:
        for y in [set_const, set_expr, set_var]:
            product_cases.append((x, y, 36))

    for value in [set_const, set_expr, set_var]:
        for zero in [zero_expr, zero_var, zero_resource_var, 0]:
            product_cases.append((value, zero, 6))
            product_cases.append((zero, value, 6))

    @pytest.mark.parametrize("x, y, expected", product_cases)
    def test_product(self, x, y, expected):
        assert self.table.product(x, y).eval(self.state, self.model) == expected

    max_cases = []

    for x in [set_const, set_expr, set_var]:
        for y in [set_const, set_expr, set_var]:
            max_cases.append((x, y, 3))

    for value in [set_const, set_expr, set_var]:
        for zero in [zero_expr, zero_var, zero_resource_var, 0]:
            max_cases.append((value, zero, 3))
            max_cases.append((zero, value, 3))

    @pytest.mark.parametrize("x, y, expected", max_cases)
    def test_max(self, x, y, expected):
        assert self.table.max(x, y).eval(self.state, self.model) == expected

    min_cases = [(value, value, 2) for value in [set_const, set_expr, set_var]]

    for x in [set_const, set_expr, set_var]:
        for y in [set_const, set_expr, set_var]:
            min_cases.append((x, y, 2))

    for value in [set_const, set_expr, set_var]:
        for zero in [zero_expr, zero_var, zero_resource_var, 0]:
            min_cases.append((value, zero, 2))
            min_cases.append((zero, value, 2))

    @pytest.mark.parametrize("x, y, expected", min_cases)
    def test_min(self, x, y, expected):
        assert self.table.min(x, y).eval(self.state, self.model) == expected

    error_cases = [
        0,
        (0, -1),
        (-1, 0),
        (0, dp.IntExpr(1)),
        (dp.IntExpr(1), 0),
        (0, 2),
        (2, 0),
    ]

    @pytest.mark.parametrize("index", error_cases)
    def test_error(self, index):
        with pytest.raises(BaseException):
            self.table[index].eval(self.state, self.model)


class TestIntTable3D:
    model = dp.Model()
    obj = model.add_object_type(number=3)

    zero_expr = dp.ElementExpr(0)
    zero_var = model.add_element_var(object_type=obj, target=0)
    zero_resource_var = model.add_element_resource_var(
        object_type=obj, target=0, less_is_better=True
    )

    one_expr = dp.ElementExpr(1)
    one_var = model.add_element_var(object_type=obj, target=1)
    one_resource_var = model.add_element_resource_var(
        object_type=obj, target=1, less_is_better=True
    )

    set_const = model.create_set_const(object_type=obj, value=[0, 1])
    set_expr = dp.SetExpr(set_const)
    set_var = model.add_set_var(object_type=obj, target=[0, 1])

    two_expr = dp.ElementExpr(2)
    two_var = model.add_element_var(object_type=obj, target=2)
    two_resource_var = model.add_element_resource_var(
        object_type=obj, target=2, less_is_better=True
    )

    table = model.add_int_table(
        [
            [[3, 2, 4], [4, 2, 3], [3, 4, 2]],
            [[2, 4, 3], [2, 3, 4], [3, 2, 4]],
            [[3, 4, 2], [2, 3, 4], [2, 4, 3]],
        ]
    )
    state = model.target_state

    cases = (
        [(zero, zero, zero, 3) for zero in [zero_expr, zero_var, zero_resource_var, 0]]
        + [(one, one, one, 3) for one in [one_expr, one_var, one_resource_var, 1]]
        + [(two, two, two, 3) for two in [two_expr, two_var, two_resource_var, 2]]
    )

    for x in [set_const, set_expr, set_var]:
        for y in [set_const, set_expr, set_var]:
            for z in [set_const, set_expr, set_var]:
                cases.append((x, y, z, 22))

    for zero in [zero_expr, zero_var, zero_resource_var, 0]:
        for one in [one_expr, one_var, one_resource_var, 1]:
            for two in [two_expr, two_var, two_resource_var, 2]:
                cases.append((zero, one, two, 3))
                cases.append((zero, two, one, 4))
                cases.append((one, zero, two, 3))
                cases.append((one, two, zero, 3))
                cases.append((two, zero, one, 4))
                cases.append((two, one, zero, 2))

    for zero in [zero_expr, zero_var, zero_resource_var, 0]:
        for one in [one_expr, one_var, one_resource_var, 1]:
            cases.append((zero, zero, one, 2))
            cases.append((zero, one, zero, 4))
            cases.append((zero, one, one, 2))
            cases.append((one, zero, zero, 2))
            cases.append((one, zero, one, 4))
            cases.append((one, one, zero, 2))

    for zero in [zero_expr, zero_var, zero_resource_var, 0]:
        for two in [two_expr, two_var, two_resource_var, 2]:
            cases.append((zero, zero, two, 4))
            cases.append((zero, two, zero, 3))
            cases.append((zero, two, two, 2))
            cases.append((two, zero, zero, 3))
            cases.append((two, zero, two, 2))
            cases.append((two, two, zero, 2))

    for one in [one_expr, one_var, one_resource_var, 1]:
        for two in [two_expr, two_var, two_resource_var, 2]:
            cases.append((one, one, two, 4))
            cases.append((one, two, one, 2))
            cases.append((one, two, two, 4))
            cases.append((two, one, one, 3))
            cases.append((two, one, two, 4))
            cases.append((two, two, one, 4))

    for value in [set_const, set_expr, set_var]:
        for zero in [zero_expr, zero_var, zero_resource_var, 0]:
            cases.append((value, value, zero, 11))
            cases.append((value, zero, value, 11))
            cases.append((value, zero, zero, 5))
            cases.append((zero, value, value, 11))
            cases.append((zero, value, zero, 7))
            cases.append((zero, zero, value, 5))

    @pytest.mark.parametrize("x, y, z, expected", cases)
    def test(self, x, y, z, expected):
        assert self.table[x, y, z].eval(self.state, self.model) == expected

    product_cases = []

    for x in [set_const, set_expr, set_var]:
        for y in [set_const, set_expr, set_var]:
            for z in [set_const, set_expr, set_var]:
                product_cases.append((x, y, z, 2304))

    for value in [set_const, set_expr, set_var]:
        for zero in [zero_expr, zero_var, zero_resource_var, 0]:
            product_cases.append((value, value, zero, 48))
            product_cases.append((value, zero, value, 48))
            product_cases.append((value, zero, zero, 6))
            product_cases.append((zero, value, value, 48))
            product_cases.append((zero, value, zero, 12))
            product_cases.append((zero, zero, value, 6))

    @pytest.mark.parametrize("x, y, z, expected", product_cases)
    def test_product(self, x, y, z, expected):
        assert self.table.product(x, y, z).eval(self.state, self.model) == expected

    max_cases = []

    for x in [set_const, set_expr, set_var]:
        for y in [set_const, set_expr, set_var]:
            for z in [set_const, set_expr, set_var]:
                max_cases.append((x, y, z, 4))

    for value in [set_const, set_expr, set_var]:
        for zero in [zero_expr, zero_var, zero_resource_var, 0]:
            max_cases.append((value, value, zero, 4))
            max_cases.append((value, zero, value, 4))
            max_cases.append((value, zero, zero, 3))
            max_cases.append((zero, value, value, 4))
            max_cases.append((zero, value, zero, 4))
            max_cases.append((zero, zero, value, 3))

    @pytest.mark.parametrize("x, y, z, expected", max_cases)
    def test_max(self, x, y, z, expected):
        assert self.table.max(x, y, z).eval(self.state, self.model) == expected

    min_cases = []

    for x in [set_const, set_expr, set_var]:
        for y in [set_const, set_expr, set_var]:
            for z in [set_const, set_expr, set_var]:
                min_cases.append((x, y, z, 2))

    for value in [set_const, set_expr, set_var]:
        for zero in [zero_expr, zero_var, zero_resource_var, 0]:
            min_cases.append((value, value, zero, 2))
            min_cases.append((value, zero, value, 2))
            min_cases.append((value, zero, zero, 2))
            min_cases.append((zero, value, value, 2))
            min_cases.append((zero, value, zero, 3))
            min_cases.append((zero, zero, value, 2))

    @pytest.mark.parametrize("x, y, z, expected", min_cases)
    def test_min(self, x, y, z, expected):
        assert self.table.min(x, y, z).eval(self.state, self.model) == expected

    error_cases = [
        0,
        (0, 1),
        (-1, 0, 0),
        (0, -1, 0),
        (0, 0, -1),
        (dp.IntExpr(1), 0, 0),
        (0, dp.IntExpr(1), 0),
        (0, 0, dp.IntExpr(1)),
        (3, 0, 0),
        (0, 3, 0),
        (0, 0, 3),
    ]

    @pytest.mark.parametrize("index", error_cases)
    def test_error(self, index):
        with pytest.raises(BaseException):
            self.table[index].eval(self.state, self.model)


class TestIntTable:
    model = dp.Model()
    obj = model.add_object_type(number=3)

    zero_expr = dp.ElementExpr(0)
    zero_var = model.add_element_var(object_type=obj, target=0)
    zero_resource_var = model.add_element_resource_var(
        object_type=obj, target=0, less_is_better=True
    )

    one_expr = dp.ElementExpr(1)
    one_var = model.add_element_var(object_type=obj, target=1)
    one_resource_var = model.add_element_resource_var(
        object_type=obj, target=1, less_is_better=True
    )

    set_const = model.create_set_const(object_type=obj, value=[0, 1])
    set_expr = dp.SetExpr(set_const)
    set_var = model.add_set_var(object_type=obj, target=[0, 1])

    table = model.add_int_table({(0, 0, 0, 0): 3, (1, 1, 1, 1): 2}, default=4)
    state = model.target_state

    cases = (
        [
            ((zero, zero, zero, zero), 3)
            for zero in [zero_expr, zero_var, zero_resource_var, 0]
        ]
        + [
            ((one, one, one, one), 2)
            for one in [one_expr, one_var, one_resource_var, 1]
        ]
        + [((0, 1, 0, 1), 4)]
        + [
            ((value, value, value, value), 61)
            for value in [set_const, set_expr, set_var]
        ]
    )

    for value in [set_const, set_expr, set_var]:
        for zero in [zero_expr, zero_var, zero_resource_var, 0]:
            cases.append(((zero, zero, zero, value), 7))

    @pytest.mark.parametrize("index, expected", cases)
    def test(self, index, expected):
        assert self.table[index].eval(self.state, self.model) == expected

    product_cases = [
        ((value, value, value, value), 1610612736)
        for value in [set_const, set_expr, set_var]
    ]

    for value in [set_const, set_expr, set_var]:
        for zero in [zero_expr, zero_var, zero_resource_var, 0]:
            product_cases.append(((zero, zero, zero, value), 12))

    @pytest.mark.parametrize("index, expected", product_cases)
    def test_product(self, index, expected):
        assert self.table.product(index).eval(self.state, self.model) == expected

    max_cases = [
        ((value, value, value, value), 4) for value in [set_const, set_expr, set_var]
    ]

    for value in [set_const, set_expr, set_var]:
        for zero in [zero_expr, zero_var, zero_resource_var, 0]:
            max_cases.append(((zero, zero, zero, value), 4))

    @pytest.mark.parametrize("index, expected", max_cases)
    def test_max(self, index, expected):
        assert self.table.max(index).eval(self.state, self.model) == expected

    min_cases = [
        ((value, value, value, value), 2) for value in [set_const, set_expr, set_var]
    ]

    for value in [set_const, set_expr, set_var]:
        for zero in [zero_expr, zero_var, zero_resource_var, 0]:
            min_cases.append(((zero, zero, zero, value), 3))

    @pytest.mark.parametrize("index, expected", min_cases)
    def test_min(self, index, expected):
        assert self.table.min(index).eval(self.state, self.model) == expected

    error_cases = [
        (-1, 0, 0),
        (0, -1, 0),
        (0, 0, -1),
        (dp.IntExpr(1), 0, 0),
        (0, dp.IntExpr(1), 0),
        (0, 0, dp.IntExpr(1)),
    ]

    @pytest.mark.parametrize("index", error_cases)
    def test_error(self, index):
        with pytest.raises(BaseException):
            self.table[index].eval(self.state, self.model)


class TestFloatTable1D:
    model = dp.Model()
    obj = model.add_object_type(number=2)

    zero_expr = dp.ElementExpr(0)
    zero_var = model.add_element_var(object_type=obj, target=0)
    zero_resource_var = model.add_element_resource_var(
        object_type=obj, target=0, less_is_better=True
    )

    one_expr = dp.ElementExpr(1)
    one_var = model.add_element_var(object_type=obj, target=1)
    one_resource_var = model.add_element_resource_var(
        object_type=obj, target=1, less_is_better=True
    )

    set_const = model.create_set_const(object_type=obj, value=[0, 1])
    set_expr = dp.SetExpr(set_const)
    set_var = model.add_set_var(object_type=obj, target=[0, 1])

    table = model.add_float_table([0.3, 0.2])
    state = model.target_state

    cases = [
        (zero_expr, pytest.approx(0.3)),
        (zero_var, pytest.approx(0.3)),
        (zero_resource_var, pytest.approx(0.3)),
        (0, pytest.approx(0.3)),
        (one_expr, pytest.approx(0.2)),
        (one_var, pytest.approx(0.2)),
        (one_resource_var, pytest.approx(0.2)),
        (1, pytest.approx(0.2)),
        (set_const, pytest.approx(0.5)),
        (set_expr, pytest.approx(0.5)),
        (set_var, pytest.approx(0.5)),
    ]

    @pytest.mark.parametrize("x, expected", cases)
    def test(self, x, expected):
        assert self.table[x].eval(self.state, self.model) == expected

    product_cases = [
        (set_const, pytest.approx(0.06)),
        (set_expr, pytest.approx(0.06)),
        (set_var, pytest.approx(0.06)),
    ]

    @pytest.mark.parametrize("x, expected", product_cases)
    def test_product(self, x, expected):
        assert self.table.product(x).eval(self.state, self.model) == expected

    max_cases = [
        (set_const, pytest.approx(0.3)),
        (set_expr, pytest.approx(0.3)),
        (set_var, pytest.approx(0.3)),
    ]

    @pytest.mark.parametrize("x, expected", max_cases)
    def test_max(self, x, expected):
        assert self.table.max(x).eval(self.state, self.model) == expected

    min_cases = [
        (set_const, pytest.approx(0.2)),
        (set_expr, pytest.approx(0.2)),
        (set_var, pytest.approx(0.2)),
    ]

    @pytest.mark.parametrize("x, expected", min_cases)
    def test_min(self, x, expected):
        assert self.table.min(x).eval(self.state, self.model) == expected

    error_cases = [-1, dp.FloatExpr(0), 2]

    @pytest.mark.parametrize("index", error_cases)
    def test_error(self, index):
        with pytest.raises(BaseException):
            self.table[index].eval(self.state, self.model)


class TestFloatTable2D:
    model = dp.Model()
    obj = model.add_object_type(number=2)

    zero_expr = dp.ElementExpr(0)
    zero_var = model.add_element_var(object_type=obj, target=0)
    zero_resource_var = model.add_element_resource_var(
        object_type=obj, target=0, less_is_better=True
    )

    one_expr = dp.ElementExpr(1)
    one_var = model.add_element_var(object_type=obj, target=1)
    one_resource_var = model.add_element_resource_var(
        object_type=obj, target=1, less_is_better=True
    )

    set_const = model.create_set_const(object_type=obj, value=[0, 1])
    set_expr = dp.SetExpr(set_const)
    set_var = model.add_set_var(object_type=obj, target=[0, 1])

    table = model.add_float_table([[0.3, 0.2], [0.2, 0.3]])
    state = model.target_state

    cases = [
        (zero, zero, pytest.approx(0.3))
        for zero in [zero_expr, zero_var, zero_resource_var, 0]
    ] + [
        (one, one, pytest.approx(0.3))
        for one in [one_expr, one_var, one_resource_var, 1]
    ]

    for x in [set_const, set_expr, set_var]:
        for y in [set_const, set_expr, set_var]:
            cases.append((x, y, pytest.approx(1.0)))

    for zero in [zero_expr, zero_var, zero_resource_var, 0]:
        for one in [one_expr, one_var, one_resource_var, 1]:
            cases.append((zero, one, pytest.approx(0.2)))
            cases.append((one, zero, pytest.approx(0.2)))

    for value in [set_const, set_expr, set_var]:
        for zero in [zero_expr, zero_var, zero_resource_var, 0]:
            cases.append((value, zero, pytest.approx(0.5)))
            cases.append((zero, value, pytest.approx(0.5)))

    @pytest.mark.parametrize("x, y, expected", cases)
    def test(self, x, y, expected):
        assert self.table[x, y].eval(self.state, self.model) == expected

    product_cases = []

    for x in [set_const, set_expr, set_var]:
        for y in [set_const, set_expr, set_var]:
            product_cases.append((x, y, pytest.approx(0.0036)))

    for value in [set_const, set_expr, set_var]:
        for zero in [zero_expr, zero_var, zero_resource_var, 0]:
            product_cases.append((value, zero, pytest.approx(0.06)))
            product_cases.append((zero, value, pytest.approx(0.06)))

    @pytest.mark.parametrize("x, y, expected", product_cases)
    def test_product(self, x, y, expected):
        assert self.table.product(x, y).eval(self.state, self.model) == expected

    max_cases = []

    for x in [set_const, set_expr, set_var]:
        for y in [set_const, set_expr, set_var]:
            max_cases.append((x, y, pytest.approx(0.3)))

    for value in [set_const, set_expr, set_var]:
        for zero in [zero_expr, zero_var, zero_resource_var, 0]:
            max_cases.append((value, zero, pytest.approx(0.3)))
            max_cases.append((zero, value, pytest.approx(0.3)))

    @pytest.mark.parametrize("x, y, expected", max_cases)
    def test_max(self, x, y, expected):
        assert self.table.max(x, y).eval(self.state, self.model) == expected

    min_cases = []

    for x in [set_const, set_expr, set_var]:
        for y in [set_const, set_expr, set_var]:
            min_cases.append((x, y, pytest.approx(0.2)))

    for value in [set_const, set_expr, set_var]:
        for zero in [zero_expr, zero_var, zero_resource_var, 0]:
            min_cases.append((value, zero, pytest.approx(0.2)))
            min_cases.append((zero, value, pytest.approx(0.2)))

    @pytest.mark.parametrize("x, y, expected", min_cases)
    def test_min(self, x, y, expected):
        assert self.table.min(x, y).eval(self.state, self.model) == expected

    error_cases = [
        0,
        (0, -1),
        (-1, 0),
        (0, dp.FloatExpr(1)),
        (dp.FloatExpr(1), 0),
        (0, 2),
        (2, 0),
    ]

    @pytest.mark.parametrize("index", error_cases)
    def test_error(self, index):
        with pytest.raises(BaseException):
            self.table[index].eval(self.state, self.model)


class TestFloatTable3D:
    model = dp.Model()
    obj = model.add_object_type(number=3)

    zero_expr = dp.ElementExpr(0)
    zero_var = model.add_element_var(object_type=obj, target=0)
    zero_resource_var = model.add_element_resource_var(
        object_type=obj, target=0, less_is_better=True
    )

    one_expr = dp.ElementExpr(1)
    one_var = model.add_element_var(object_type=obj, target=1)
    one_resource_var = model.add_element_resource_var(
        object_type=obj, target=1, less_is_better=True
    )

    set_const = model.create_set_const(object_type=obj, value=[0, 1])
    set_expr = dp.SetExpr(set_const)
    set_var = model.add_set_var(object_type=obj, target=[0, 1])

    two_expr = dp.ElementExpr(2)
    two_var = model.add_element_var(object_type=obj, target=2)
    two_resource_var = model.add_element_resource_var(
        object_type=obj, target=2, less_is_better=True
    )

    table = model.add_float_table(
        [
            [[0.3, 0.2, 0.4], [0.4, 0.2, 0.3], [0.3, 0.4, 0.2]],
            [[0.2, 0.4, 0.3], [0.2, 0.3, 0.4], [0.3, 0.2, 0.4]],
            [[0.3, 0.4, 0.2], [0.2, 0.3, 0.4], [0.2, 0.4, 0.3]],
        ]
    )
    state = model.target_state

    cases = (
        [
            (zero, zero, zero, pytest.approx(0.3))
            for zero in [zero_expr, zero_var, zero_resource_var, 0]
        ]
        + [
            (one, one, one, pytest.approx(0.3))
            for one in [one_expr, one_var, one_resource_var, 1]
        ]
        + [
            (two, two, two, pytest.approx(0.3))
            for two in [two_expr, two_var, two_resource_var, 2]
        ]
    )

    for x in [set_const, set_expr, set_var]:
        for y in [set_const, set_expr, set_var]:
            for z in [set_const, set_expr, set_var]:
                cases.append((x, y, z, pytest.approx(2.2)))

    for zero in [zero_expr, zero_var, zero_resource_var, 0]:
        for one in [one_expr, one_var, one_resource_var, 1]:
            for two in [two_expr, two_var, two_resource_var, 2]:
                cases.append((zero, one, two, pytest.approx(0.3)))
                cases.append((zero, two, one, pytest.approx(0.4)))
                cases.append((one, zero, two, pytest.approx(0.3)))
                cases.append((one, two, zero, pytest.approx(0.3)))
                cases.append((two, zero, one, pytest.approx(0.4)))
                cases.append((two, one, zero, pytest.approx(0.2)))

    for zero in [zero_expr, zero_var, zero_resource_var, 0]:
        for one in [one_expr, one_var, one_resource_var, 1]:
            cases.append((zero, zero, one, pytest.approx(0.2)))
            cases.append((zero, one, zero, pytest.approx(0.4)))
            cases.append((zero, one, one, pytest.approx(0.2)))
            cases.append((one, zero, zero, pytest.approx(0.2)))
            cases.append((one, zero, one, pytest.approx(0.4)))
            cases.append((one, one, zero, pytest.approx(0.2)))

    for zero in [zero_expr, zero_var, zero_resource_var, 0]:
        for two in [two_expr, two_var, two_resource_var, 2]:
            cases.append((zero, zero, two, pytest.approx(0.4)))
            cases.append((zero, two, zero, pytest.approx(0.3)))
            cases.append((zero, two, two, pytest.approx(0.2)))
            cases.append((two, zero, zero, pytest.approx(0.3)))
            cases.append((two, zero, two, pytest.approx(0.2)))
            cases.append((two, two, zero, pytest.approx(0.2)))

    for one in [one_expr, one_var, one_resource_var, 1]:
        for two in [two_expr, two_var, two_resource_var, 2]:
            cases.append((one, one, two, pytest.approx(0.4)))
            cases.append((one, two, one, pytest.approx(0.2)))
            cases.append((one, two, two, pytest.approx(0.4)))
            cases.append((two, one, one, pytest.approx(0.3)))
            cases.append((two, one, two, pytest.approx(0.4)))
            cases.append((two, two, one, pytest.approx(0.4)))

    for value in [set_const, set_expr, set_var]:
        for zero in [zero_expr, zero_var, zero_resource_var, 0]:
            cases.append((value, value, zero, pytest.approx(1.1)))
            cases.append((value, zero, value, pytest.approx(1.1)))
            cases.append((value, zero, zero, pytest.approx(0.5)))
            cases.append((zero, value, value, pytest.approx(1.1)))
            cases.append((zero, value, zero, pytest.approx(0.7)))
            cases.append((zero, zero, value, pytest.approx(0.5)))

    @pytest.mark.parametrize("x, y, z, expected", cases)
    def test(self, x, y, z, expected):
        assert self.table[x, y, z].eval(self.state, self.model) == expected

    product_cases = []

    for x in [set_const, set_expr, set_var]:
        for y in [set_const, set_expr, set_var]:
            for z in [set_const, set_expr, set_var]:
                product_cases.append((x, y, z, pytest.approx(2.304e-5)))

    for value in [set_const, set_expr, set_var]:
        for zero in [zero_expr, zero_var, zero_resource_var, 0]:
            product_cases.append((value, value, zero, pytest.approx(0.0048)))
            product_cases.append((value, zero, value, pytest.approx(0.0048)))
            product_cases.append((value, zero, zero, pytest.approx(0.06)))
            product_cases.append((zero, value, value, pytest.approx(0.0048)))
            product_cases.append((zero, value, zero, pytest.approx(0.12)))
            product_cases.append((zero, zero, value, pytest.approx(0.06)))

    @pytest.mark.parametrize("x, y, z, expected", product_cases)
    def test_product(self, x, y, z, expected):
        assert self.table.product(x, y, z).eval(self.state, self.model) == expected

    max_cases = []

    for x in [set_const, set_expr, set_var]:
        for y in [set_const, set_expr, set_var]:
            for z in [set_const, set_expr, set_var]:
                max_cases.append((x, y, z, pytest.approx(0.4)))

    for value in [set_const, set_expr, set_var]:
        for zero in [zero_expr, zero_var, zero_resource_var, 0]:
            max_cases.append((value, value, zero, pytest.approx(0.4)))
            max_cases.append((value, zero, value, pytest.approx(0.4)))
            max_cases.append((value, zero, zero, pytest.approx(0.3)))
            max_cases.append((zero, value, value, pytest.approx(0.4)))
            max_cases.append((zero, value, zero, pytest.approx(0.4)))
            max_cases.append((zero, zero, value, pytest.approx(0.3)))

    @pytest.mark.parametrize("x, y, z, expected", max_cases)
    def test_max(self, x, y, z, expected):
        assert self.table.max(x, y, z).eval(self.state, self.model) == expected

    min_cases = []

    for x in [set_const, set_expr, set_var]:
        for y in [set_const, set_expr, set_var]:
            for z in [set_const, set_expr, set_var]:
                min_cases.append((x, y, z, pytest.approx(0.2)))

    for value in [set_const, set_expr, set_var]:
        for zero in [zero_expr, zero_var, zero_resource_var, 0]:
            min_cases.append((value, value, zero, pytest.approx(0.2)))
            min_cases.append((value, zero, value, pytest.approx(0.2)))
            min_cases.append((value, zero, zero, pytest.approx(0.2)))
            min_cases.append((zero, value, value, pytest.approx(0.2)))
            min_cases.append((zero, value, zero, pytest.approx(0.3)))
            min_cases.append((zero, zero, value, pytest.approx(0.2)))

    @pytest.mark.parametrize("x, y, z, expected", min_cases)
    def test_min(self, x, y, z, expected):
        assert self.table.min(x, y, z).eval(self.state, self.model) == expected

    error_cases = [
        0,
        (0, 1),
        (-1, 0, 0),
        (0, -1, 0),
        (0, 0, -1),
        (dp.FloatExpr(1), 0, 0),
        (0, dp.FloatExpr(1), 0),
        (0, 0, dp.FloatExpr(1)),
        (3, 0, 0),
        (0, 3, 0),
        (0, 0, 3),
    ]

    @pytest.mark.parametrize("index", error_cases)
    def test_error(self, index):
        with pytest.raises(BaseException):
            self.table[index].eval(self.state, self.model)


class TestFloatTable:
    model = dp.Model()
    obj = model.add_object_type(number=3)

    zero_expr = dp.ElementExpr(0)
    zero_var = model.add_element_var(object_type=obj, target=0)
    zero_resource_var = model.add_element_resource_var(
        object_type=obj, target=0, less_is_better=True
    )

    one_expr = dp.ElementExpr(1)
    one_var = model.add_element_var(object_type=obj, target=1)
    one_resource_var = model.add_element_resource_var(
        object_type=obj, target=1, less_is_better=True
    )

    set_const = model.create_set_const(object_type=obj, value=[0, 1])
    set_expr = dp.SetExpr(set_const)
    set_var = model.add_set_var(object_type=obj, target=[0, 1])

    table = model.add_float_table({(0, 0, 0, 0): 0.3, (1, 1, 1, 1): 0.2}, default=0.4)
    state = model.target_state

    cases = (
        [
            ((zero, zero, zero, zero), pytest.approx(0.3))
            for zero in [zero_expr, zero_var, zero_resource_var, 0]
        ]
        + [
            ((one, one, one, one), pytest.approx(0.2))
            for one in [one_expr, one_var, one_resource_var, 1]
        ]
        + [((0, 1, 0, 1), pytest.approx(0.4))]
        + [
            ((value, value, value, value), pytest.approx(6.1))
            for value in [set_const, set_expr, set_var]
        ]
    )

    for value in [set_const, set_expr, set_var]:
        for zero in [zero_expr, zero_var, zero_resource_var, 0]:
            cases.append(((zero, zero, zero, value), pytest.approx(0.7)))

    @pytest.mark.parametrize("index, expected", cases)
    def test(self, index, expected):
        assert self.table[index].eval(self.state, self.model) == expected

    product_cases = [
        ((value, value, value, value), pytest.approx(1.61061273e-07))
        for value in [set_const, set_expr, set_var]
    ]

    for value in [set_const, set_expr, set_var]:
        for zero in [zero_expr, zero_var, zero_resource_var, 0]:
            product_cases.append(((zero, zero, zero, value), pytest.approx(0.12)))

    @pytest.mark.parametrize("index, expected", product_cases)
    def test_product(self, index, expected):
        assert self.table.product(index).eval(self.state, self.model) == expected

    max_cases = [
        ((value, value, value, value), pytest.approx(0.4))
        for value in [set_const, set_expr, set_var]
    ]

    for value in [set_const, set_expr, set_var]:
        for zero in [zero_expr, zero_var, zero_resource_var, 0]:
            max_cases.append(((zero, zero, zero, value), pytest.approx(0.4)))

    @pytest.mark.parametrize("index, expected", max_cases)
    def test_max(self, index, expected):
        assert self.table.max(index).eval(self.state, self.model) == expected

    min_cases = [
        ((value, value, value, value), pytest.approx(0.2))
        for value in [set_const, set_expr, set_var]
    ]

    for value in [set_const, set_expr, set_var]:
        for zero in [zero_expr, zero_var, zero_resource_var, 0]:
            min_cases.append(((zero, zero, zero, value), pytest.approx(0.3)))

    @pytest.mark.parametrize("index, expected", min_cases)
    def test_min(self, index, expected):
        assert self.table.min(index).eval(self.state, self.model) == expected

    error_cases = [
        (-1, 0, 0),
        (0, -1, 0),
        (0, 0, -1),
        (dp.FloatExpr(1), 0, 0),
        (0, dp.FloatExpr(1), 0),
        (0, 0, dp.FloatExpr(1)),
    ]

    @pytest.mark.parametrize("index", error_cases)
    def test_error(self, index):
        with pytest.raises(BaseException):
            self.table[index].eval(self.state, self.model)
