use super::expression::*;
use dypdl::expression::ArgumentExpression;
use dypdl::prelude::*;
use pyo3::prelude::*;

/// 1-dimensional table of element constants.
///
/// `t[x]` returns an element expression referring to an item where `t` is `ElementTable1D` and `x` is `ElementExpr`, `ElementVar`, `ElementResourceVar`, or `int`.
///
/// Examples
/// --------
/// >>> import didppy as dp
/// >>> model = dp.Model()
/// >>> obj = model.add_object_type(number=2)
/// >>> var = model.add_element_var(object_type=obj, target=1)
/// >>> table = model.add_element_table([2, 3])
/// >>> model.add_base_case([table[var] == 2])
#[pyclass(name = "ElementTable1D")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ElementTable1DPy(Table1DHandle<Element>);

impl From<ElementTable1DPy> for Table1DHandle<Element> {
    fn from(table: ElementTable1DPy) -> Self {
        table.0
    }
}

impl ElementTable1DPy {
    pub fn new(table: Table1DHandle<Element>) -> ElementTable1DPy {
        ElementTable1DPy(table)
    }
}

#[pymethods]
impl ElementTable1DPy {
    fn __getitem__(&self, i: ElementUnion) -> ElementExprPy {
        ElementExprPy::new(self.0.element(i))
    }
}

/// 2-dimensional table of element constants.
///
/// `t[x, y]` returns an element expression referring to an item where `t` is `ElementTable2D` and `x` and `y` are `ElementExpr`, `ElementVar`, `ElementResourceVar`, or `int`.
///
/// Examples
/// --------
/// >>> import didppy as dp
/// >>> model = dp.Model()
/// >>> obj = model.add_object_type(number=2)
/// >>> var = model.add_element_var(object_type=obj, target=1)
/// >>> table = model.add_element_table([[2, 3], [0, 1]])
/// >>> model.add_base_case([table[0, var] == 2])
#[pyclass(name = "ElementTable2D")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ElementTable2DPy(Table2DHandle<Element>);

impl From<ElementTable2DPy> for Table2DHandle<Element> {
    fn from(table: ElementTable2DPy) -> Self {
        table.0
    }
}

impl ElementTable2DPy {
    pub fn new(table: Table2DHandle<Element>) -> ElementTable2DPy {
        ElementTable2DPy(table)
    }
}

#[pymethods]
impl ElementTable2DPy {
    fn __getitem__(&self, index: (ElementUnion, ElementUnion)) -> ElementExprPy {
        let (x, y) = index;
        ElementExprPy::new(self.0.element(x, y))
    }
}

/// 3-dimensional table of element constants.
///
/// `t[x, y, z]` returns an element expression referring to an item where `t` is `ElementTable3D` and `x`, `y`, and `z` are `ElementExpr`, `ElementVar`, `ElementResourceVar`, or `int`.
///
/// Examples
/// --------
/// >>> import didppy as dp
/// >>> model = dp.Model()
/// >>> obj = model.add_object_type(number=2)
/// >>> var = model.add_element_var(object_type=obj, target=1)
/// >>> table = model.add_element_table([[[2, 3], [0, 1]], [[0, 1], [2, 2]]]])
/// >>> model.add_base_case([table[0, var, 1] == 2])
#[pyclass(name = "ElementTable3D")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ElementTable3DPy(Table3DHandle<Element>);

impl From<ElementTable3DPy> for Table3DHandle<Element> {
    fn from(table: ElementTable3DPy) -> Self {
        table.0
    }
}

impl ElementTable3DPy {
    pub fn new(table: Table3DHandle<Element>) -> ElementTable3DPy {
        ElementTable3DPy(table)
    }
}

#[pymethods]
impl ElementTable3DPy {
    fn __getitem__(&self, index: (ElementUnion, ElementUnion, ElementUnion)) -> ElementExprPy {
        let (x, y, z) = index;
        ElementExprPy::new(self.0.element(x, y, z))
    }
}

/// Table of element constants.
///
/// `t[index]` returns an element expression referring to an item where `t` is `ElementTable` and `index` is a sequence of `ElementExpr`, `ElementVar`, `ElementResourceVar`, or `int`.
///
/// Examples
/// --------
/// >>> import didppy as dp
/// >>> model = dp.Model()
/// >>> obj = model.add_object_type(number=2)
/// >>> var = model.add_element_var(object_type=obj, target=1)
/// >>> table = model.add_element_table({(0, 0, 0, 0): 1, (1, 1, 1, 1): 3}, default=2)
/// >>> model.add_base_case([table[0, var, 1, 0] == 2])
#[pyclass(name = "ElementTable")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ElementTablePy(TableHandle<Element>);

impl From<ElementTablePy> for TableHandle<Element> {
    fn from(table: ElementTablePy) -> Self {
        table.0
    }
}

impl ElementTablePy {
    pub fn new(table: TableHandle<Element>) -> ElementTablePy {
        ElementTablePy(table)
    }
}

#[pymethods]
impl ElementTablePy {
    fn __getitem__(&self, index: Vec<ElementUnion>) -> ElementExprPy {
        let index = index.into_iter().map(ElementExpression::from).collect();
        ElementExprPy::new(self.0.element(index))
    }
}

/// 1-dimensional table of set constants.
///
/// `t[x]` returns a set expression referring to an item where `t` is `SetTable1D` and `x` is `ElementExpr`, `ElementVar`, `ElementResourceVar`, or `int`.
///
/// Examples
/// --------
/// >>> import didppy as dp
/// >>> model = dp.Model()
/// >>> object1 = model.add_object_type(number=2)
/// >>> object2 = model.add_object_type(number=4)
/// >>> var = model.add_element_var(object_type=obj1, target=0)
/// >>> table = model.add_set_table([[2, 3], [1, 2]], object_type=obj2)
/// >>> model.add_base_case([table[var].contains(1)])
#[pyclass(name = "SetTable1D")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetTable1DPy(Table1DHandle<Set>, usize);

impl From<SetTable1DPy> for Table1DHandle<Set> {
    fn from(table: SetTable1DPy) -> Self {
        table.0
    }
}

impl SetTable1DPy {
    pub fn new(table: Table1DHandle<Set>, capacity: usize) -> SetTable1DPy {
        SetTable1DPy(table, capacity)
    }

    pub fn get_capacity_of_set(&self) -> usize {
        self.1
    }
}

#[pymethods]
impl SetTable1DPy {
    fn __getitem__(&self, i: ElementUnion) -> SetExprPy {
        SetExprPy::new(self.0.element(i))
    }

    /// union(x)
    ///
    /// Takes the union of set constants in a table over the set of indices.
    ///
    /// Parameters
    /// ----------
    /// x : int, ElementExpr, ElementVar, ElementResourceVar, SetExpr, SetVar, or SetConst
    ///     Set of indices for the first dimension.
    ///
    /// Returns
    /// -------
    /// SetExpr
    ///     The union.
    ///
    /// Raises
    /// ------
    /// OverflowError
    ///     If x is a negative integer.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> object1 = model.add_object_type(number=2)
    /// >>> object2 = model.add_object_type(number=6)
    /// >>> var = model.add_set_var(object_type=obj1, target=[0, 1])
    /// >>> table = model.add_set_table([[2, 3], [1, 2]], object_type=obj2)
    /// >>> model.add_base_case([table.union(var).contains(2)])
    #[pyo3(signature = (x))]
    fn union(&self, x: ArgumentUnion) -> SetExprPy {
        SetExprPy::new(self.0.union(self.1, x))
    }

    /// intersection(x)
    ///
    /// Takes the intersection of set constants in a table over the set of indices.
    ///
    /// Parameters
    /// ----------
    /// x : int, ElementExpr, ElementVar, ElementResourceVar, SetExpr, SetVar, or SetConst
    ///     Set of indices for the first dimension.
    ///
    /// Returns
    /// -------
    /// SetExpr
    ///     The intersection.
    ///
    /// Raises
    /// ------
    /// OverflowError
    ///     If x is a negative integer.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> object1 = model.add_object_type(number=2)
    /// >>> object2 = model.add_object_type(number=6)
    /// >>> var = model.add_set_var(object_type=obj1, target=[0, 1])
    /// >>> table = model.add_set_table([[2, 3], [1, 2]], object_type=obj2)
    /// >>> model.add_base_case([table.intersection(var).contains(2)])
    #[pyo3(signature = (x))]
    fn intersection(&self, x: ArgumentUnion) -> SetExprPy {
        SetExprPy::new(self.0.intersection(self.1, x))
    }

    /// symmetric_difference(x)
    ///
    /// Takes the symmetric difference of set constants in a table over the set of indices.
    ///
    /// Parameters
    /// ----------
    /// x : int, ElementExpr, ElementVar, ElementResourceVar, SetExpr, SetVar, or SetConst
    ///     Set of indices for the first dimension.
    ///
    /// Returns
    /// -------
    /// SetExpr
    ///     The symmetric difference.
    ///
    /// Raises
    /// ------
    /// OverflowError
    ///     If x is a negative integer.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> object1 = model.add_object_type(number=2)
    /// >>> object2 = model.add_object_type(number=6)
    /// >>> var = model.add_set_var(object_type=obj1, target=[0, 1])
    /// >>> table = model.add_set_table([[2, 3], [1, 2]], object_type=obj2)
    /// >>> model.add_base_case([table.symmetric_difference(var).contains(2)])
    #[pyo3(signature = (x))]
    fn symmetric_difference(&self, x: ArgumentUnion) -> SetExprPy {
        SetExprPy::new(self.0.symmetric_difference(self.1, x))
    }
}

/// 2-dimensional table of set constants.
///
/// `t[x, y]` returns a set expression referring to an item where `t` is `SetTable2D` and `x` and `y` are `ElementExpr`, `ElementVar`, `ElementResourceVar`, or `int`.
///
/// Examples
/// --------
/// >>> import didppy as dp
/// >>> model = dp.Model()
/// >>> object1 = model.add_object_type(number=2)
/// >>> object2 = model.add_object_type(number=4)
/// >>> var = model.add_element_var(object_type=obj1, target=0)
/// >>> table = model.add_set_table([[[2, 3], [1, 2]], [[1, 1], [0, 1]]], object_type=obj2)
/// >>> model.add_base_case([table[0, var].contains(1)])
#[pyclass(name = "SetTable2D")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetTable2DPy(Table2DHandle<Set>, usize);

impl SetTable2DPy {
    pub fn new(table: Table2DHandle<Set>, capacity: usize) -> SetTable2DPy {
        SetTable2DPy(table, capacity)
    }

    pub fn get_capacity_of_set(&self) -> usize {
        self.1
    }
}

impl From<SetTable2DPy> for Table2DHandle<Set> {
    fn from(table: SetTable2DPy) -> Self {
        table.0
    }
}

#[pymethods]
impl SetTable2DPy {
    fn __getitem__(&self, index: (ElementUnion, ElementUnion)) -> SetExprPy {
        let (x, y) = index;
        SetExprPy::new(self.0.element(x, y))
    }

    /// union(x, y)
    ///
    /// Takes the union of set constants in a table over the set of indices.
    ///
    /// Parameters
    /// ----------
    /// x : int, ElementExpr, ElementVar, ElementResourceVar, SetExpr, SetVar, or SetConst
    ///     Set of indices for the first dimension.
    /// y : int, ElementExpr, ElementVar, ElementResourceVar, SetExpr, SetVar, or SetConst
    ///     Set of indices for the second dimension.
    ///
    /// Returns
    /// -------
    /// SetExpr
    ///     The union.
    ///
    /// Raises
    /// ------
    /// OverflowError
    ///     If x or y is a negative integer.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> object1 = model.add_object_type(number=2)
    /// >>> object2 = model.add_object_type(number=4)
    /// >>> var = model.add_set_var(object_type=obj1, target=[0, 1])
    /// >>> table = model.add_set_table([[[2, 3], [1, 2]], [[1, 1], [0, 1]]], object_type=obj2)
    /// >>> model.add_base_case([table.union(0, var).contains(1)])
    #[pyo3(signature = (x, y))]
    fn union(&self, x: ArgumentUnion, y: ArgumentUnion) -> SetExprPy {
        SetExprPy::new(self.0.union(self.1, x, y))
    }

    /// intersection(x, y)
    ///
    /// Takes the intersection of set constants in a table over the set of indices.
    ///
    /// Parameters
    /// ----------
    /// x : int, ElementExpr, ElementVar, ElementResourceVar, SetExpr, SetVar, or SetConst
    ///     Set of indices for the first dimension.
    /// y : int, ElementExpr, ElementVar, ElementResourceVar, SetExpr, SetVar, or SetConst
    ///     Set of indices for the second dimension.
    ///
    /// Returns
    /// -------
    /// SetExpr
    ///     The intersection.
    ///
    /// Raises
    /// ------
    /// OverflowError
    ///     If x or y is a negative integer.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> object1 = model.add_object_type(number=2)
    /// >>> object2 = model.add_object_type(number=4)
    /// >>> var = model.add_set_var(object_type=x, target=[0, 1])
    /// >>> table = model.add_set_table([[[2, 3], [1, 2]], [[1, 1], [0, 1]]], object_type=obj2)
    /// >>> model.add_base_case([table.intersection(0, var).contains(1)])
    #[pyo3(signature = (x, y))]
    fn intersection(&self, x: ArgumentUnion, y: ArgumentUnion) -> SetExprPy {
        SetExprPy::new(self.0.intersection(self.1, x, y))
    }

    /// symmetric_difference(x, y)
    ///
    /// Takes the symmetric difference of set constants in a table over the set of indices.
    ///
    /// Parameters
    /// ----------
    /// x : int, ElementExpr, ElementVar, ElementResourceVar, SetExpr, SetVar, or SetConst
    ///     Set of indices for the first dimension.
    /// y : int, ElementExpr, ElementVar, ElementResourceVar, SetExpr, SetVar, or SetConst
    ///     Set of indices for the second dimension.
    ///
    /// Returns
    /// -------
    /// SetExpr
    ///     The symmetric difference.
    ///
    /// Raises
    /// ------
    /// OverflowError
    ///     If x or y is a negative integer.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> object1 = model.add_object_type(number=2)
    /// >>> object2 = model.add_object_type(number=4)
    /// >>> var = model.add_set_var(object_type=obj1, target=[0, 1])
    /// >>> table = model.add_set_table([[[2, 3], [1, 2]], [[1, 1], [0, 1]]], object_type=obj2)
    /// >>> model.add_base_case([table.symmetric_difference(0, var).contains(1)])
    #[pyo3(signature = (x, y))]
    fn symmetric_difference(&self, x: ArgumentUnion, y: ArgumentUnion) -> SetExprPy {
        SetExprPy::new(self.0.symmetric_difference(self.1, x, y))
    }
}

/// 3-dimensional table of set constants.
///
/// `t[x, y, z]` returns a set expression referring to an item where `t` is `SetTable3D` and `x`, `y`, and `z` are `ElementExpr`, `ElementVar`, `ElementResourceVar`, or `int`.
///
/// Examples
/// --------
/// >>> import didppy as dp
/// >>> model = dp.Model()
/// >>> object1 = model.add_object_type(number=2)
/// >>> object2 = model.add_object_type(number=4)
/// >>> var = model.add_element_var(object_type=obj1, target=0)
/// >>> table = model.add_set_table([[[[2, 3], [1, 2]], [[], [2]]], [[[1, 1], [2, 3]], [[], [2]]]], object_type=obj2)
/// >>> model.add_base_case([table[0, var, 1].contains(1)])
#[pyclass(name = "SetTable3D")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetTable3DPy(Table3DHandle<Set>, usize);

impl From<SetTable3DPy> for Table3DHandle<Set> {
    fn from(table: SetTable3DPy) -> Self {
        table.0
    }
}

impl SetTable3DPy {
    pub fn new(table: Table3DHandle<Set>, capacity: usize) -> SetTable3DPy {
        SetTable3DPy(table, capacity)
    }

    pub fn get_capacity_of_set(&self) -> usize {
        self.1
    }
}

#[pymethods]
impl SetTable3DPy {
    fn __getitem__(&self, index: (ElementUnion, ElementUnion, ElementUnion)) -> SetExprPy {
        let (x, y, z) = index;
        SetExprPy::new(self.0.element(x, y, z))
    }

    /// union(x, y, z)
    ///
    /// Takes the union of set constants in a table over the set of indices.
    ///
    /// Parameters
    /// ----------
    /// x : int, ElementExpr, ElementVar, ElementResourceVar, SetExpr, SetVar, or SetConst
    ///     Set of indices for the first dimension.
    /// y : int, ElementExpr, ElementVar, ElementResourceVar, SetExpr, SetVar, or SetConst
    ///     Set of indices for the second dimension.
    /// z : int, ElementExpr, ElementVar, ElementResourceVar, SetExpr, SetVar, or SetConst
    ///     Set of indices for the third dimension.
    ///
    /// Returns
    /// -------
    /// SetExpr
    ///     The union.
    ///
    /// Raises
    /// ------
    /// OverflowError
    ///     If x, y, or z is a negative integer.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> object1 = model.add_object_type(number=2)
    /// >>> object2 = model.add_object_type(number=4)
    /// >>> var = model.add_set_var(object_type=obj1, target=[0, 1])
    /// >>> table = model.add_set_table([[[[2, 3], [1, 2]], [[], [2]]], [[[1, 1], [2, 3]], [[], [2]]]], object_type=obj2)
    /// >>> model.add_base_case([table.union(0, var, 1).contains(1)])
    #[pyo3(signature = (x, y, z))]
    fn union(&self, x: ArgumentUnion, y: ArgumentUnion, z: ArgumentUnion) -> SetExprPy {
        SetExprPy::new(self.0.union(self.1, x, y, z))
    }

    /// intersection(x, y, z)
    ///
    /// Takes the intersection of set constants in a table over the set of indices.
    ///
    /// Parameters
    /// ----------
    /// x : int, ElementExpr, ElementVar, ElementResourceVar, SetExpr, SetVar, or SetConst
    ///     Set of indices for the first dimension.
    /// y : int, ElementExpr, ElementVar, ElementResourceVar, SetExpr, SetVar, or SetConst
    ///     Set of indices for the second dimension.
    /// z : int, ElementExpr, ElementVar, ElementResourceVar, SetExpr, SetVar, or SetConst
    ///     Set of indices for the third dimension.
    ///
    /// Returns
    /// -------
    /// SetExpr
    ///     The intersection.
    ///
    /// Raises
    /// ------
    /// OverflowError
    ///     If x, y, or z is a negative integer.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> object1 = model.add_object_type(number=2)
    /// >>> object2 = model.add_object_type(number=4)
    /// >>> var = model.add_set_var(object_type=obj1, target=[0, 1])
    /// >>> table = model.add_set_table([[[[2, 3], [1, 2]], [[], [2]]], [[[1, 1], [2, 3]], [[], [2]]]], object_type=obj2)
    /// >>> model.add_base_case([table.intersection(0, var, 1).contains(1)])
    #[pyo3(signature = (x, y, z))]
    fn intersection(&self, x: ArgumentUnion, y: ArgumentUnion, z: ArgumentUnion) -> SetExprPy {
        SetExprPy::new(self.0.intersection(self.1, x, y, z))
    }

    /// symmetric_difference(x, y, z)
    ///
    /// Takes the symmetric difference of set constants in a table over the set of indices.
    ///
    /// Parameters
    /// ----------
    /// x : int, ElementExpr, ElementVar, ElementResourceVar, SetExpr, SetVar, or SetConst
    ///     Set of indices for the first dimension.
    /// y : int, ElementExpr, ElementVar, ElementResourceVar, SetExpr, SetVar, or SetConst
    ///     Set of indices for the second dimension.
    /// z : int, ElementExpr, ElementVar, ElementResourceVar, SetExpr, SetVar, or SetConst
    ///     Set of indices for the third dimension.
    ///
    /// Returns
    /// -------
    /// SetExpr
    ///     The symmetric difference.
    ///
    /// Raises
    /// ------
    /// OverflowError
    ///     If x, y, or z is a negative integer.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> object1 = model.add_object_type(number=2)
    /// >>> object2 = model.add_object_type(number=4)
    /// >>> var = model.add_set_var(object_type=obj1, target=[0, 1])
    /// >>> table = model.add_set_table([[[[2, 3], [1, 2]], [[], [2]]], [[[1, 1], [2, 3]], [[], [2]]]], object_type=obj2)
    /// >>> model.add_base_case([table.symmetric_difference(0, var, 1).contains(1)])
    #[pyo3(signature = (x, y, z))]
    fn symmetric_difference(
        &self,
        x: ArgumentUnion,
        y: ArgumentUnion,
        z: ArgumentUnion,
    ) -> SetExprPy {
        SetExprPy::new(self.0.symmetric_difference(self.1, x, y, z))
    }
}

/// Table of set constants.
///
/// `t[index]` returns a set expression referring to an item where `t` is `SetTable` and `index` is a sequence of `ElementExpr`, `ElementVar`, `ElementResourceVar`, or `int`.
///
/// Examples
/// --------
/// >>> import didppy as dp
/// >>> model = dp.Model()
/// >>> object1 = model.add_object_type(number=2)
/// >>> object2 = model.add_object_type(number=4)
/// >>> var = model.add_element_var(object_type=obj1, target=0)
/// >>> table = model.add_set_table({(0, 0, 0, 0): [1, 2], (1, 1, 1, 1): [2, 1]}, default=[], object_type=obj2)
/// >>> model.add_base_case([table[0, var, 0, 1].contains(1)])
#[pyclass(name = "SetTable")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetTablePy(TableHandle<Set>, usize);

impl From<SetTablePy> for TableHandle<Set> {
    fn from(table: SetTablePy) -> Self {
        table.0
    }
}

impl SetTablePy {
    pub fn new(table: TableHandle<Set>, capacity: usize) -> SetTablePy {
        SetTablePy(table, capacity)
    }

    pub fn get_capacity_of_set(&self) -> usize {
        self.1
    }
}

#[pymethods]
impl SetTablePy {
    fn __getitem__(&self, index: Vec<ElementUnion>) -> SetExprPy {
        let index = index.into_iter().map(ElementExpression::from).collect();
        SetExprPy::new(self.0.element(index))
    }

    /// union(indices)
    ///
    /// Takes the union of set constants in a table over the set of indices.
    ///
    /// Parameters
    /// ----------
    /// indices : tuple of int, ElementExpr, ElementVar, ElementResourceVar, SetExpr, SetVar, or SetConst
    ///     Tuple of index sets.
    ///
    /// Returns
    /// -------
    /// SetExpr
    ///     The union.
    ///
    /// Raises
    /// ------
    /// OverflowError
    ///     If a negative integer is in `indices`.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> object1 = model.add_object_type(number=2)
    /// >>> object2 = model.add_object_type(number=4)
    /// >>> var = model.add_set_var(object_type=obj1, target=[0, 1])
    /// >>> table = model.add_set_table({(0, 0, 0, 0): [1, 2], (1, 1, 1, 1): [2, 1]}, default=[], object_type=obj2)
    /// >>> model.add_base_case([table.union((0, var, 0, 1)).contains(1)])
    #[pyo3(signature = (indices))]
    fn union(&self, indices: Vec<ArgumentUnion>) -> SetExprPy {
        SetExprPy::new(self.0.union(
            self.1,
            indices.into_iter().map(ArgumentExpression::from).collect(),
        ))
    }

    /// intersection(indices)
    ///
    /// Takes the intersection of set constants in a table over the set of indices.
    ///
    /// Parameters
    /// ----------
    /// indices : tuple of int, ElementExpr, ElementVar, ElementResourceVar, SetExpr, SetVar, or SetConst
    ///     Tuple of index sets.
    ///
    /// Returns
    /// -------
    /// SetExpr
    ///     The intersection.
    ///
    /// Raises
    /// ------
    /// OverflowError
    ///     If a negative integer is in `indices`.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> object1 = model.add_object_type(number=2)
    /// >>> object2 = model.add_object_type(number=4)
    /// >>> var = model.add_set_var(object_type=obj1, target=[0, 1])
    /// >>> table = model.add_set_table({(0, 0, 0, 0): [1, 2], (1, 1, 1, 1): [2, 1]}, default=[], object_type=obj2)
    /// >>> model.add_base_case([table.intersection((0, var, 0, 1)).contains(1)])
    #[pyo3(signature = (indices))]
    fn intersection(&self, indices: Vec<ArgumentUnion>) -> SetExprPy {
        SetExprPy::new(self.0.intersection(
            self.1,
            indices.into_iter().map(ArgumentExpression::from).collect(),
        ))
    }

    /// symmetric_difference(indices)
    ///
    /// Takes the symmetric difference of set constants in a table over the set of indices.
    ///
    /// Parameters
    /// ----------
    /// indices : tuple of int, ElementExpr, ElementVar, ElementResourceVar, SetExpr, SetVar, or SetConst
    ///     Tuple of index sets.
    ///
    /// Returns
    /// -------
    /// SetExpr
    ///     The symmetric difference.
    ///
    /// Raises
    /// ------
    /// OverflowError
    ///     If a negative integer is in `indices`.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> object1 = model.add_object_type(number=2)
    /// >>> object2 = model.add_object_type(number=4)
    /// >>> var = model.add_set_var(object_type=obj1, target=[0, 1])
    /// >>> table = model.add_set_table({(0, 0, 0, 0): [1, 2], (1, 1, 1, 1): [2, 1]}, default=[], object_type=obj2)
    /// >>> model.add_base_case([table.symmetric_difference((0, var, 0, 1)).contains(1)])
    #[pyo3(signature = (indices))]
    fn symmetric_difference(&self, indices: Vec<ArgumentUnion>) -> SetExprPy {
        SetExprPy::new(self.0.symmetric_difference(
            self.1,
            indices.into_iter().map(ArgumentExpression::from).collect(),
        ))
    }
}

/// 1-dimensional table of bool constants.
///
/// `t[x]` returns a condition referring to an item where `t` is `BoolTable1D` and `x` is `ElementExpr`, `ElementVar`, `ElementResourceVar`, or `int`.
///
/// Examples
/// --------
/// >>> import didppy as dp
/// >>> model = dp.Model()
/// >>> obj = model.add_object_type(number=2)
/// >>> var = model.add_element_var(object_type=obj, target=1)
/// >>> table = model.add_bool_table([True, False])
/// >>> model.add_base_case([table[var]])
#[pyclass(name = "BoolTable1D")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoolTable1DPy(Table1DHandle<bool>);

impl From<BoolTable1DPy> for Table1DHandle<bool> {
    fn from(table: BoolTable1DPy) -> Self {
        table.0
    }
}

impl BoolTable1DPy {
    pub fn new(table: Table1DHandle<bool>) -> BoolTable1DPy {
        BoolTable1DPy(table)
    }
}

#[pymethods]
impl BoolTable1DPy {
    fn __getitem__(&self, i: ElementUnion) -> ConditionPy {
        ConditionPy::new(self.0.element(i))
    }
}

/// 2-dimensional table of bool constants.
///
/// `t[x, y]` returns a condition referring to an item where `t` is `BoolTable2D` and `x` and `y` are `ElementExpr`, `ElementVar`, `ElementResourceVar`, or `int`.
///
/// Examples
/// --------
/// >>> import didppy as dp
/// >>> model = dp.Model()
/// >>> obj = model.add_object_type(number=2)
/// >>> var = model.add_element_var(object_type=obj, target=1)
/// >>> table = model.add_bool_table([[True, False], [False, True]])
/// >>> model.add_base_case([table[0, var]])
#[pyclass(name = "BoolTable2D")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoolTable2DPy(Table2DHandle<bool>);

impl From<BoolTable2DPy> for Table2DHandle<bool> {
    fn from(table: BoolTable2DPy) -> Self {
        table.0
    }
}

impl BoolTable2DPy {
    pub fn new(table: Table2DHandle<bool>) -> BoolTable2DPy {
        BoolTable2DPy(table)
    }
}

#[pymethods]
impl BoolTable2DPy {
    fn __getitem__(&self, index: (ElementUnion, ElementUnion)) -> ConditionPy {
        let (x, y) = index;
        ConditionPy::new(self.0.element(x, y))
    }
}

/// 3-dimensional table of bool constants.
///
/// `t[x, y, z]` returns a condition referring to an item where `t` is `BoolTable3D` and `x`, `y`, and `z` are `ElementExpr`, `ElementVar`, `ElementResourceVar`, or `int`.
///
/// Examples
/// --------
/// >>> import didppy as dp
/// >>> model = dp.Model()
/// >>> obj = model.add_object_type(number=2)
/// >>> var = model.add_element_var(object_type=obj, target=1)
/// >>> table = model.add_bool_table([[[True, False], [False, True]], [[False, False], [True, True]]])
/// >>> model.add_base_case([table[0, var, 1]])
#[pyclass(name = "BoolTable3D")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoolTable3DPy(Table3DHandle<bool>);

impl From<BoolTable3DPy> for Table3DHandle<bool> {
    fn from(table: BoolTable3DPy) -> Self {
        table.0
    }
}

impl BoolTable3DPy {
    pub fn new(table: Table3DHandle<bool>) -> BoolTable3DPy {
        BoolTable3DPy(table)
    }
}

#[pymethods]
impl BoolTable3DPy {
    fn __getitem__(&self, index: (ElementUnion, ElementUnion, ElementUnion)) -> ConditionPy {
        let (x, y, z) = index;
        ConditionPy::new(self.0.element(x, y, z))
    }
}

/// Table of bool constants.
///
/// `t[index]` returns a condition referring to an item where `t` is `BoolTable` and `index` is a sequence of `ElementExpr`, `ElementVar`, `ElementResourceVar`, or `int`.
///
/// Examples
/// --------
/// >>> import didppy as dp
/// >>> model = dp.Model()
/// >>> obj = model.add_object_type(number=2)
/// >>> var = model.add_element_var(object_type=obj, target=1)
/// >>> table = model.add_bool_table({(0, 0, 0, 0): False, (1, 1, 1, 1): True}, default=False)
/// >>> model.add_base_case([table[1, var, 1, 1]])
#[pyclass(name = "BoolTable")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoolTablePy(TableHandle<bool>);

impl From<BoolTablePy> for TableHandle<bool> {
    fn from(table: BoolTablePy) -> Self {
        table.0
    }
}

impl BoolTablePy {
    pub fn new(table: TableHandle<bool>) -> BoolTablePy {
        BoolTablePy(table)
    }
}

#[pymethods]
impl BoolTablePy {
    fn __getitem__(&self, index: Vec<ElementUnion>) -> ConditionPy {
        let index = index.into_iter().map(ElementExpression::from).collect();
        ConditionPy::new(self.0.element(index))
    }
}

#[derive(FromPyObject, Debug, PartialEq, Clone)]
pub enum ArgumentUnion {
    Element(ElementUnion),
    Set(SetUnion),
}

impl From<ArgumentUnion> for ArgumentExpression {
    fn from(index: ArgumentUnion) -> Self {
        match index {
            ArgumentUnion::Element(index) => {
                ArgumentExpression::from(ElementExpression::from(index))
            }
            ArgumentUnion::Set(index) => ArgumentExpression::from(SetExpression::from(index)),
        }
    }
}

/// 1-dimensional table of integer constants.
///
/// `t[x]` returns an integer expression referring to an item where `t` is `IntTable1D` and `x` is `ElementExpr`, `ElementVar`, `ElementResourceVar`, or `int`.
/// If `x` is `SetExpr`, `SetVar`, or `SetConst`, `t[x]` returns the sum of constants over `x`.
///
/// Examples
/// --------
/// >>> import didppy as dp
/// >>> model = dp.Model()
/// >>> obj = model.add_object_type(number=2)
/// >>> var = model.add_element_var(object_type=obj, target=1)
/// >>> table = model.add_int_table([2, 3])
/// >>> model.add_base_case([table[var] == 2])
/// >>> set_var = model.add_set_var(object_type=obj, target=[0, 1])
/// >>> model.add_base_case([table[set_var] == 5])
#[pyclass(name = "IntTable1D")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntTable1DPy(Table1DHandle<Integer>);

impl From<IntTable1DPy> for Table1DHandle<Integer> {
    fn from(table: IntTable1DPy) -> Self {
        table.0
    }
}

impl IntTable1DPy {
    pub fn new(table: Table1DHandle<Integer>) -> IntTable1DPy {
        IntTable1DPy(table)
    }
}

#[pymethods]
impl IntTable1DPy {
    fn __getitem__(&self, i: ArgumentUnion) -> IntExprPy {
        IntExprPy::new(match i {
            ArgumentUnion::Element(i) => self.0.element(i),
            ArgumentUnion::Set(i) => self.0.sum(i),
        })
    }

    /// product(set)
    ///
    /// Takes the product of constants in a table over the set of indices.
    ///
    /// Parameters
    /// ----------
    /// set : SetExpr, SetVar, or SetConst
    ///     Set of indices.
    ///
    /// Returns
    /// -------
    /// IntExpr
    ///     The product.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> obj = model.add_object_type(number=2)
    /// >>> var = model.add_set_var(object_type=obj, target=[0, 1])
    /// >>> table = model.add_int_table([2, 3])
    /// >>> model.add_base_case([table.product(var) == 2])
    #[pyo3(signature = (i))]
    fn product(&self, i: SetUnion) -> IntExprPy {
        IntExprPy::new(self.0.product(i))
    }

    /// max(set)
    ///
    /// Takes the maximum of constants in a table over the set of indices.
    ///
    /// Parameters
    /// ----------
    /// set : SetExpr, SetVar, or SetConst
    ///     Set of indices.
    ///
    /// Returns
    /// -------
    /// IntExpr
    ///     The maximum.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> obj = model.add_object_type(number=2)
    /// >>> var = model.add_set_var(object_type=obj, target=[0, 1])
    /// >>> table = model.add_int_table([2, 3])
    /// >>> model.add_base_case([table.max(var) == 2])
    #[pyo3(signature = (i))]
    fn max(&self, i: SetUnion) -> IntExprPy {
        IntExprPy::new(self.0.max(i))
    }

    /// min(set)
    ///
    /// Takes the minimum of constants in a table over the set of indices.
    ///
    /// Parameters
    /// ----------
    /// set : SetExpr, SetVar, or SetConst
    ///     Set of indices.
    ///
    /// Returns
    /// -------
    /// IntExpr
    ///     The minimum.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> obj = model.add_object_type(number=2)
    /// >>> var = model.add_set_var(object_type=obj, target=[0, 1])
    /// >>> table = model.add_int_table([2, 3])
    /// >>> model.add_base_case([table.min(var) == 2])
    #[pyo3(signature = (i))]
    fn min(&self, i: SetUnion) -> IntExprPy {
        IntExprPy::new(self.0.min(i))
    }
}

/// 2-dimensional table of integer constants.
///
/// `t[x, y]` returns an integer expression referring to an item where `t` is `IntTable2D` and `x` and `y` are `ElementExpr`, `ElementVar`, `ElementResourceVar`, or `int`.
/// If `x` and/or `y` are/is `SetExpr`, `SetVar`, or `SetConst`, `t[x, y]` returns the sum of constants over `x` and `y`.
///
/// Examples
/// --------
/// >>> import didppy as dp
/// >>> model = dp.Model()
/// >>> obj = model.add_object_type(number=2)
/// >>> var = model.add_element_var(object_type=obj, target=1)
/// >>> set_var = model.add_set_var(object_type=obj, target=[0, 1])
/// >>> table = model.add_int_table([[2, 3], [-1, 2]])
/// >>> model.add_base_case([table[var, set_var] == 2])
#[pyclass(name = "IntTable2D")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntTable2DPy(Table2DHandle<Integer>);

impl From<IntTable2DPy> for Table2DHandle<Integer> {
    fn from(table: IntTable2DPy) -> Self {
        table.0
    }
}

impl IntTable2DPy {
    pub fn new(table: Table2DHandle<Integer>) -> IntTable2DPy {
        IntTable2DPy(table)
    }
}

#[pymethods]
impl IntTable2DPy {
    fn __getitem__(&self, index: (ArgumentUnion, ArgumentUnion)) -> IntExprPy {
        let (x, y) = index;
        IntExprPy::new(match (x, y) {
            (ArgumentUnion::Element(x), ArgumentUnion::Element(y)) => self.0.element(x, y),
            (ArgumentUnion::Set(x), ArgumentUnion::Element(y)) => self.0.sum_x(x, y),
            (ArgumentUnion::Element(x), ArgumentUnion::Set(y)) => self.0.sum_y(x, y),
            (ArgumentUnion::Set(x), ArgumentUnion::Set(y)) => self.0.sum(x, y),
        })
    }

    /// product(x, y)
    ///
    /// Takes the product of constants in a table over the set of indices.
    ///
    /// Parameters
    /// ----------
    /// x: int, ElementExpr, ElementVar, ElementResourceVar, SetExpr, SetVar, or SetConst
    ///     Set of indices for the first dimension.
    /// y: int, ElementExpr, ElementVar, ElementResourceVar, SetExpr, SetVar, or SetConst
    ///     Set of indices for the second dimension.
    ///
    /// Returns
    /// -------
    /// IntExpr
    ///     The product.
    ///
    /// Raises
    /// ------
    /// OverflowError
    ///     If x or y is a negative integer.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> obj = model.add_object_type(number=2)
    /// >>> var = model.add_element_var(object_type=obj, target=1)
    /// >>> set_var = model.add_set_var(object_type=obj, target=[0, 1])
    /// >>> table = model.add_int_table([[2, 3], [-1, 2]])
    /// >>> model.add_base_case([table.product(var, set_var) == 2])
    #[pyo3(signature = (x, y))]
    fn product(&self, x: ArgumentUnion, y: ArgumentUnion) -> IntExprPy {
        IntExprPy::new(match (x, y) {
            (ArgumentUnion::Element(x), ArgumentUnion::Element(y)) => self.0.element(x, y),
            (ArgumentUnion::Set(x), ArgumentUnion::Element(y)) => self.0.product_x(x, y),
            (ArgumentUnion::Element(x), ArgumentUnion::Set(y)) => self.0.product_y(x, y),
            (ArgumentUnion::Set(x), ArgumentUnion::Set(y)) => self.0.product(x, y),
        })
    }

    /// max(x, y)
    ///
    /// Takes the maximum of constants in a table over the set of indices.
    ///
    /// Parameters
    /// ----------
    /// x: int, ElementExpr, ElementVar, ElementResourceVar, SetExpr, SetVar, or SetConst
    ///     Set of indices for the first dimension.
    /// y: int, ElementExpr, ElementVar, ElementResourceVar, SetExpr, SetVar, or SetConst
    ///     Set of indices for the second dimension.
    ///
    /// Returns
    /// -------
    /// IntExpr
    ///     The maximum.
    ///
    /// Raises
    /// ------
    /// OverflowError
    ///     If x or y is a negative integer.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> obj = model.add_object_type(number=2)
    /// >>> var = model.add_element_var(object_type=obj, target=1)
    /// >>> set_var = model.add_set_var(object_type=obj, target=[0, 1])
    /// >>> table = model.add_int_table([[2, 3], [-1, 2]])
    /// >>> model.add_base_case([table.max(var, set_var) == 2])
    #[pyo3(signature = (x, y))]
    fn max(&self, x: ArgumentUnion, y: ArgumentUnion) -> IntExprPy {
        IntExprPy::new(match (x, y) {
            (ArgumentUnion::Element(x), ArgumentUnion::Element(y)) => self.0.element(x, y),
            (ArgumentUnion::Set(x), ArgumentUnion::Element(y)) => self.0.max_x(x, y),
            (ArgumentUnion::Element(x), ArgumentUnion::Set(y)) => self.0.max_y(x, y),
            (ArgumentUnion::Set(x), ArgumentUnion::Set(y)) => self.0.max(x, y),
        })
    }

    /// min(x, y)
    ///
    /// Takes the minimum of constants in a table over the set of indices.
    ///
    /// Parameters
    /// ----------
    /// x: int, ElementExpr, ElementVar, ElementResourceVar, SetExpr, SetVar, or SetConst
    ///     Set of indices for the first dimension.
    /// y: int, ElementExpr, ElementVar, ElementResourceVar, SetExpr, SetVar, or SetConst
    ///     Set of indices for the second dimension.
    ///
    /// Returns
    /// -------
    /// IntExpr
    ///     The minimum.
    ///
    /// Raises
    /// ------
    /// OverflowError
    ///     If x or y is a negative integer.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> obj = model.add_object_type(number=2)
    /// >>> var = model.add_element_var(object_type=obj, target=1)
    /// >>> set_var = model.add_set_var(object_type=obj, target=[0, 1])
    /// >>> table = model.add_int_table([[2, 3], [-1, 2]])
    /// >>> model.add_base_case([table.min(var, set_var) == 2])
    #[pyo3(signature = (x, y))]
    fn min(&self, x: ArgumentUnion, y: ArgumentUnion) -> IntExprPy {
        IntExprPy::new(match (x, y) {
            (ArgumentUnion::Element(x), ArgumentUnion::Element(y)) => self.0.element(x, y),
            (ArgumentUnion::Set(x), ArgumentUnion::Element(y)) => self.0.min_x(x, y),
            (ArgumentUnion::Element(x), ArgumentUnion::Set(y)) => self.0.min_y(x, y),
            (ArgumentUnion::Set(x), ArgumentUnion::Set(y)) => self.0.min(x, y),
        })
    }
}

/// 3-dimensional table of integer constants.
///
/// `t[x, y, z]` returns an integer expression referring to an item where `t` is `IntTable3D` and `x`, `y`, and `z` are `ElementExpr`, `ElementVar`, `ElementResourceVar`, or `int`.
/// If `x`, `y`, and/or `z` are/is `SetExpr`, `SetVar`, or `SetConst`, `t[x, y, z]` returns the sum of constants over `x`, `y`, and `z`.
///
/// Examples
/// --------
/// >>> import didppy as dp
/// >>> model = dp.Model()
/// >>> obj = model.add_object_type(number=2)
/// >>> var = model.add_element_var(object_type=obj, target=1)
/// >>> set_var = model.add_set_var(object_type=obj, target=[0, 1])
/// >>> table = model.add_int_table([[[2, 3], [0, 1]], [[0, -1], [2, 2]]]])
/// >>> model.add_base_case([table[var, set_var, 1] == 2])
#[pyclass(name = "IntTable3D")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntTable3DPy(Table3DHandle<Integer>);

impl From<IntTable3DPy> for Table3DHandle<Integer> {
    fn from(table: IntTable3DPy) -> Self {
        table.0
    }
}

impl IntTable3DPy {
    pub fn new(table: Table3DHandle<Integer>) -> IntTable3DPy {
        IntTable3DPy(table)
    }
}

#[pymethods]
impl IntTable3DPy {
    fn __getitem__(&self, index: (ArgumentUnion, ArgumentUnion, ArgumentUnion)) -> IntExprPy {
        let (x, y, z) = index;
        IntExprPy::new(match (x, y, z) {
            (ArgumentUnion::Element(x), ArgumentUnion::Element(y), ArgumentUnion::Element(z)) => {
                self.0.element(x, y, z)
            }
            (x, y, z) => self.0.sum(x, y, z),
        })
    }

    /// product(x, y, z)
    ///
    /// Takes the product of constants in a table over the set of indices.
    ///
    /// Parameters
    /// ----------
    /// x : tuple of int, ElementExpr, ElementVar, ElementResourceVar, SetExpr, SetVar, or SetConst
    ///     Set of indices for the first dimension.
    /// y : tuple of int, ElementExpr, ElementVar, ElementResourceVar, SetExpr, SetVar, or SetConst
    ///     Set of indices for the second dimension.
    /// z : tuple of int, ElementExpr, ElementVar, ElementResourceVar, SetExpr, SetVar, or SetConst
    ///     Set of indices for the third dimension.
    ///
    /// Returns
    /// -------
    /// IntExpr
    ///     The product.
    ///
    /// Raises
    /// ------
    /// OverflowError
    ///     If x, y, or z is a negative integer.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> obj = model.add_object_type(number=2)
    /// >>> var = model.add_element_var(object_type=obj, target=1)
    /// >>> set_var = model.add_set_var(object_type=obj, target=[0, 1])
    /// >>> table = model.add_int_table([[[2, 3], [0, 1]], [[0, -1], [2, 2]]]])
    /// >>> model.add_base_case([table.product(var, set_var, 1) == 2])
    #[pyo3(signature = (x, y, z))]
    fn product(&self, x: ArgumentUnion, y: ArgumentUnion, z: ArgumentUnion) -> IntExprPy {
        IntExprPy::new(match (x, y, z) {
            (ArgumentUnion::Element(x), ArgumentUnion::Element(y), ArgumentUnion::Element(z)) => {
                self.0.element(x, y, z)
            }
            (x, y, z) => self.0.product(x, y, z),
        })
    }

    /// max(x, y, z)
    ///
    /// Takes the maximum of constants in a table over the set of indices.
    ///
    /// Parameters
    /// ----------
    /// x : tuple of int, ElementExpr, ElementVar, ElementResourceVar, SetExpr, SetVar, or SetConst
    ///     Set of indices for the first dimension.
    /// y : tuple of int, ElementExpr, ElementVar, ElementResourceVar, SetExpr, SetVar, or SetConst
    ///     Set of indices for the second dimension.
    /// z : tuple of int, ElementExpr, ElementVar, ElementResourceVar, SetExpr, SetVar, or SetConst
    ///     Set of indices for the third dimension.
    ///
    /// Returns
    /// -------
    /// IntExpr
    ///     The maximum.
    ///
    /// Raises
    /// ------
    /// OverflowError
    ///     If x, y, or z is a negative integer.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> obj = model.add_object_type(number=2)
    /// >>> var = model.add_element_var(object_type=obj, target=1)
    /// >>> set_var = model.add_set_var(object_type=obj, target=[0, 1])
    /// >>> table = model.add_int_table([[[2, 3], [0, 1]], [[0, -1], [2, 2]]]])
    /// >>> model.add_base_case([table.max(var, set_var, 1) == 2])
    #[pyo3(signature = (x, y, z))]
    fn max(&self, x: ArgumentUnion, y: ArgumentUnion, z: ArgumentUnion) -> IntExprPy {
        IntExprPy::new(match (x, y, z) {
            (ArgumentUnion::Element(x), ArgumentUnion::Element(y), ArgumentUnion::Element(z)) => {
                self.0.element(x, y, z)
            }
            (x, y, z) => self.0.max(x, y, z),
        })
    }

    /// min(x, y, z)
    ///
    /// Takes the minimum of constants in a table over the set of indices.
    ///
    /// Parameters
    /// ----------
    /// x : tuple of int, ElementExpr, ElementVar, ElementResourceVar, SetExpr, SetVar, or SetConst
    ///     Set of indices for the first dimension.
    /// y : tuple of int, ElementExpr, ElementVar, ElementResourceVar, SetExpr, SetVar, or SetConst
    ///     Set of indices for the second dimension.
    /// z : tuple of int, ElementExpr, ElementVar, ElementResourceVar, SetExpr, SetVar, or SetConst
    ///     Set of indices for the third dimension.
    ///
    /// Returns
    /// -------
    /// IntExpr
    ///     The minimum.
    ///
    /// Raises
    /// ------
    /// OverflowError
    ///     If x, y, or z is a negative integer.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> obj = model.add_object_type(number=2)
    /// >>> var = model.add_element_var(object_type=obj, target=1)
    /// >>> set_var = model.add_set_var(object_type=obj, target=[0, 1])
    /// >>> table = model.add_int_table([[[2, 3], [0, 1]], [[0, -1], [2, 2]]]])
    /// >>> model.add_base_case([table.min(var, set_var, 1) == 2])
    #[pyo3(signature = (x, y, z))]
    fn min(&self, x: ArgumentUnion, y: ArgumentUnion, z: ArgumentUnion) -> IntExprPy {
        IntExprPy::new(match (x, y, z) {
            (ArgumentUnion::Element(x), ArgumentUnion::Element(y), ArgumentUnion::Element(z)) => {
                self.0.element(x, y, z)
            }
            (x, y, z) => self.0.min(x, y, z),
        })
    }
}

/// Table of integer constants.
///
/// `t[index]` returns an integer expression referring to an item where `t` is `IntTable` and `index` is a sequence of `ElementExpr`, `ElementVar`, `ElementResourceVar`, or `int`.
/// If one of `index` is `SetExpr`, `SetVar`, or `SetConst`, `t[index]` returns the sum of constants.
///
/// Examples
/// --------
/// >>> import didppy as dp
/// >>> model = dp.Model()
/// >>> obj = model.add_object_type(number=2)
/// >>> var = model.add_element_var(object_type=obj, target=1)
/// >>> set_var = model.add_set_var(object_type=obj, target=[0, 1])
/// >>> table = model.add_int_table({(0, 0, 0, 0): -1, (1, 1, 1, 1): 3}, default=2)
/// >>> model.add_base_case([table[0, var, set_var, 0] == 2])
#[pyclass(name = "IntTable")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntTablePy(TableHandle<Integer>);

impl From<IntTablePy> for TableHandle<Integer> {
    fn from(table: IntTablePy) -> Self {
        table.0
    }
}

impl IntTablePy {
    pub fn new(table: TableHandle<Integer>) -> IntTablePy {
        IntTablePy(table)
    }
}

#[pymethods]
impl IntTablePy {
    fn __getitem__(&self, index: Vec<ArgumentUnion>) -> IntExprPy {
        let mut elements = Vec::with_capacity(index.len());
        for i in &index {
            match i {
                ArgumentUnion::Element(i) => elements.push(ElementExpression::from(i.clone())),
                _ => return IntExprPy::new(self.0.sum(index)),
            }
        }
        IntExprPy::new(self.0.element(elements))
    }

    /// product(indices)
    ///
    /// Takes the product of constants in a table over the set of indices.
    ///
    /// Parameters
    /// ----------
    /// indices : tuple of int, ElementExpr, ElementVar, ElementResourceVar, SetExpr, SetVar, or SetConst
    ///     Tuple of index sets.
    ///
    /// Returns
    /// -------
    /// IntExpr
    ///     The product.
    ///
    /// Raises
    /// ------
    /// OverflowError
    ///     If a negative integer is in `indices`.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> obj = model.add_object_type(number=2)
    /// >>> var = model.add_set_var(object_type=obj, target=[0, 1])
    /// >>> set_var = model.add_set_var(object_type=obj, target=[0, 1])
    /// >>> table = model.add_int_table({(0, 0, 0, 0): -1, (1, 1, 1, 1): 3}, default=2)
    /// >>> model.add_base_case([table.product(0, var, set_var, 0) == 2])
    #[pyo3(signature = (indices))]
    fn product(&self, indices: Vec<ArgumentUnion>) -> IntExprPy {
        let mut elements = Vec::with_capacity(indices.len());
        for i in &indices {
            match i {
                ArgumentUnion::Element(i) => elements.push(ElementExpression::from(i.clone())),
                _ => return IntExprPy::new(self.0.product(indices)),
            }
        }
        IntExprPy::new(self.0.element(elements))
    }

    /// max(indices)
    ///
    /// Takes the maximum of constants in a table over the set of indices.
    ///
    /// Parameters
    /// ----------
    /// indices : tuple of int, ElementExpr, ElementVar, ElementResourceVar, SetExpr, SetVar, or SetConst
    ///     Tuple of index sets.
    ///
    /// Returns
    /// -------
    /// IntExpr
    ///     The maximum.
    ///
    /// Raises
    /// ------
    /// OverflowError
    ///     If a negative integer is in `indices`.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> obj = model.add_object_type(number=2)
    /// >>> var = model.add_element_var(object_type=obj, target=1)
    /// >>> set_var = model.add_set_var(object_type=obj, target=[0, 1])
    /// >>> table = model.add_int_table({(0, 0, 0, 0): -1, (1, 1, 1, 1): 3}, default=2)
    /// >>> model.add_base_case([table.max(0, var, set_var, 0) == 2])
    #[pyo3(signature = (indices))]
    fn max(&self, indices: Vec<ArgumentUnion>) -> IntExprPy {
        let mut elements = Vec::with_capacity(indices.len());
        for i in &indices {
            match i {
                ArgumentUnion::Element(i) => elements.push(ElementExpression::from(i.clone())),
                _ => return IntExprPy::new(self.0.max(indices)),
            }
        }
        IntExprPy::new(self.0.element(elements))
    }

    /// min(indices)
    ///
    /// Takes the minimum of constants in a table over the set of indices.
    ///
    /// Parameters
    /// ----------
    /// indices : tuple of int, ElementExpr, ElementVar, ElementResourceVar, SetExpr, SetVar, or SetConst
    ///     Tuple of index sets.
    ///
    /// Returns
    /// -------
    /// IntExpr
    ///     The minimum.
    ///
    /// Raises
    /// ------
    /// OverflowError
    ///     If a negative integer is in `indices`.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> obj = model.add_object_type(number=2)
    /// >>> var = model.add_element_var(object_type=obj, target=1)
    /// >>> set_var = model.add_set_var(object_type=obj, target=[0, 1])
    /// >>> table = model.add_int_table({(0, 0, 0, 0): -1, (1, 1, 1, 1): 3}, default=2)
    /// >>> model.add_base_case([table.min(0, var, set_var, 0) == 2])
    #[pyo3(signature = (indices))]
    fn min(&self, indices: Vec<ArgumentUnion>) -> IntExprPy {
        let mut elements = Vec::with_capacity(indices.len());
        for i in &indices {
            match i {
                ArgumentUnion::Element(i) => elements.push(ElementExpression::from(i.clone())),
                _ => return IntExprPy::new(self.0.min(indices)),
            }
        }
        IntExprPy::new(self.0.element(elements))
    }
}

/// 1-dimensional table of continuous constants.
///
/// `t[x]` returns an continuous expression referring to an item where `t` is `FloatTable1D` and `x` is `ElementExpr`, `ElementVar`, `ElementResourceVar`, or `int`.
/// If `x` is `SetExpr`, `SetVar`, or `SetConst`, `t[x]` returns the sum of constants over `x`.
///
/// Examples
/// --------
/// >>> import didppy as dp
/// >>> model = dp.Model()
/// >>> obj = model.add_object_type(number=2)
/// >>> var = model.add_element_var(object_type=obj, target=1)
/// >>> table = model.add_float_table([2.5, 3.5])
/// >>> model.add_base_case([table[var] == 2.5])
/// >>> set_var = model.add_set_var(object_type=obj, target=[0, 1])
/// >>> model.add_base_case([table[set_var] == 2.5])
#[pyclass(name = "FloatTable1D")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FloatTable1DPy(Table1DHandle<Continuous>);

impl From<FloatTable1DPy> for Table1DHandle<Continuous> {
    fn from(table: FloatTable1DPy) -> Self {
        table.0
    }
}

impl FloatTable1DPy {
    pub fn new(table: Table1DHandle<Continuous>) -> FloatTable1DPy {
        FloatTable1DPy(table)
    }
}

#[pymethods]
impl FloatTable1DPy {
    fn __getitem__(&self, i: ArgumentUnion) -> FloatExprPy {
        FloatExprPy::new(match i {
            ArgumentUnion::Element(i) => self.0.element(i),
            ArgumentUnion::Set(i) => self.0.sum(i),
        })
    }

    /// product(set)
    ///
    /// Takes the product of constants in a table over the set of indices.
    ///
    /// Parameters
    /// ----------
    /// set : SetExpr, SetVar, or SetConst
    ///     Set of indices
    ///
    /// Returns
    /// -------
    /// FloatExpr
    ///     The product.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> obj = model.add_object_type(number=2)
    /// >>> var = model.add_set_var(object_type=obj, target=[0, 1])
    /// >>> table = model.add_float_table([2.5, 3.5])
    /// >>> model.add_base_case([table.product(var) == 2.5])
    #[pyo3(signature = (i))]
    fn product(&self, i: SetUnion) -> FloatExprPy {
        FloatExprPy::new(self.0.product(i))
    }

    /// max(set)
    ///
    /// Takes the maximum of constants in a table over the set of indices.
    ///
    /// Parameters
    /// ----------
    /// set : SetExpr, SetVar, or SetConst
    ///     Set of indices
    ///
    /// Returns
    /// -------
    /// FloatExpr
    ///     The maximum.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> obj = model.add_object_type(number=2)
    /// >>> var = model.add_set_var(object_type=obj, target=[0, 1])
    /// >>> table = model.add_float_table([2.5, 3.5])
    /// >>> model.add_base_case([table.max(var) == 2.5])
    #[pyo3(signature = (i))]
    fn max(&self, i: SetUnion) -> FloatExprPy {
        FloatExprPy::new(self.0.max(i))
    }

    /// min(set)
    ///
    /// Takes the minimum of constants in a table over the set of indices.
    ///
    /// Parameters
    /// ----------
    /// set : SetExpr, SetVar, or SetConst
    ///     Set of indices
    ///
    /// Returns
    /// -------
    /// FloatExpr
    ///     The minimum.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> obj = model.add_object_type(number=2)
    /// >>> var = model.add_set_var(object_type=obj, target=[0, 1])
    /// >>> table = model.add_float_table([2.5, 3.5])
    /// >>> model.add_base_case([table.min(var) == 2.5])
    #[pyo3(signature = (i))]
    fn min(&self, i: SetUnion) -> FloatExprPy {
        FloatExprPy::new(self.0.min(i))
    }
}

/// 2-dimensional table of continuous constants.
///
/// `t[x, y]` returns a continuous expression referring to an item where `t` is `FloatTable2D` and `x` and `y` are `ElementExpr`, `ElementVar`, `ElementResourceVar`, or `int`.
/// If `x` and/or `y` are/is `SetExpr`, `SetVar`, or `SetConst`, `t[x, y]` returns the sum of constants over `x` and `y`.
///
/// Examples
/// --------
/// >>> import didppy as dp
/// >>> model = dp.Model()
/// >>> obj = model.add_object_type(number=2)
/// >>> var = model.add_element_var(object_type=obj, target=1)
/// >>> set_var = model.add_set_var(object_type=obj, target=[0, 1])
/// >>> table = model.add_float_table([[2.5, 3.5], [-1.5, 2.5]])
/// >>> model.add_base_case([table[var, set_var] == 2.5])
#[pyclass(name = "FloatTable2D")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FloatTable2DPy(Table2DHandle<Continuous>);

impl From<FloatTable2DPy> for Table2DHandle<Continuous> {
    fn from(table: FloatTable2DPy) -> Self {
        table.0
    }
}

impl FloatTable2DPy {
    pub fn new(table: Table2DHandle<Continuous>) -> FloatTable2DPy {
        FloatTable2DPy(table)
    }
}

#[pymethods]
impl FloatTable2DPy {
    fn __getitem__(&self, index: (ArgumentUnion, ArgumentUnion)) -> FloatExprPy {
        let (x, y) = index;
        FloatExprPy::new(match (x, y) {
            (ArgumentUnion::Element(x), ArgumentUnion::Element(y)) => self.0.element(x, y),
            (ArgumentUnion::Set(x), ArgumentUnion::Element(y)) => self.0.sum_x(x, y),
            (ArgumentUnion::Element(x), ArgumentUnion::Set(y)) => self.0.sum_y(x, y),
            (ArgumentUnion::Set(x), ArgumentUnion::Set(y)) => self.0.sum(x, y),
        })
    }

    /// product(x, y)
    ///
    /// Takes the product of constants in a table over the set of indices.
    ///
    /// Parameters
    /// ----------
    /// x: int, ElementExpr, ElementVar, ElementResourceVar, SetExpr, SetVar, or SetConst
    ///     Set of indices for the first dimension.
    /// y: int, ElementExpr, ElementVar, ElementResourceVar, SetExpr, SetVar, or SetConst
    ///     Set of indices for the second dimension.
    ///
    /// Returns
    /// -------
    /// FloatExpr
    ///     The product.
    ///
    /// Raises
    /// ------
    /// OverflowError
    ///     If x or y is a negative integer.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> obj = model.add_object_type(number=2)
    /// >>> var = model.add_element_var(object_type=obj, target=1)
    /// >>> set_var = model.add_set_var(object_type=obj, target=[0, 1])
    /// >>> table = model.add_int_table([[2.5, 3.5], [-1.5, 2.5]])
    /// >>> model.add_base_case([table.product(var, set_var) == 2.5])
    #[pyo3(signature = (x, y))]
    fn product(&self, x: ArgumentUnion, y: ArgumentUnion) -> FloatExprPy {
        FloatExprPy::new(match (x, y) {
            (ArgumentUnion::Element(x), ArgumentUnion::Element(y)) => self.0.element(x, y),
            (ArgumentUnion::Set(x), ArgumentUnion::Element(y)) => self.0.product_x(x, y),
            (ArgumentUnion::Element(x), ArgumentUnion::Set(y)) => self.0.product_y(x, y),
            (ArgumentUnion::Set(x), ArgumentUnion::Set(y)) => self.0.product(x, y),
        })
    }

    /// max(x, y)
    ///
    /// Takes the maximum of constants in a table over the set of indices.
    ///
    /// Parameters
    /// ----------
    /// x: int, ElementExpr, ElementVar, ElementResourceVar, SetExpr, SetVar, or SetConst
    ///     Set of indices for the first dimension.
    /// y: int, ElementExpr, ElementVar, ElementResourceVar, SetExpr, SetVar, or SetConst
    ///     Set of indices for the second dimension.
    ///
    /// Returns
    /// -------
    /// FloatExpr
    ///     The maximum.
    ///
    /// Raises
    /// ------
    /// OverflowError
    ///     If x or y is a negative integer.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> obj = model.add_object_type(number=2)
    /// >>> var = model.add_element_var(object_type=obj, target=1)
    /// >>> set_var = model.add_set_var(object_type=obj, target=[0, 1])
    /// >>> table = model.add_int_table([[2.5, 3.5], [-1.5, 2.5]])
    /// >>> model.add_base_case([table.max(var, set_var) == 2.5])
    #[pyo3(signature = (x, y))]
    fn max(&self, x: ArgumentUnion, y: ArgumentUnion) -> FloatExprPy {
        FloatExprPy::new(match (x, y) {
            (ArgumentUnion::Element(x), ArgumentUnion::Element(y)) => self.0.element(x, y),
            (ArgumentUnion::Set(x), ArgumentUnion::Element(y)) => self.0.max_x(x, y),
            (ArgumentUnion::Element(x), ArgumentUnion::Set(y)) => self.0.max_y(x, y),
            (ArgumentUnion::Set(x), ArgumentUnion::Set(y)) => self.0.max(x, y),
        })
    }

    /// min(x, y)
    ///
    /// Takes the minimum of constants in a table over the set of indices.
    ///
    /// Parameters
    /// ----------
    /// x: int, ElementExpr, ElementVar, ElementResourceVar, SetExpr, SetVar, or SetConst
    ///     Set of indices for the first dimension.
    /// y: int, ElementExpr, ElementVar, ElementResourceVar, SetExpr, SetVar, or SetConst
    ///     Set of indices for the second dimension.
    ///
    /// Returns
    /// -------
    /// FloatExpr
    ///     The minimum.
    ///
    /// Raises
    /// ------
    /// OverflowError
    ///     If x or y is a negative integer.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> obj = model.add_object_type(number=2)
    /// >>> var = model.add_element_var(object_type=obj, target=1)
    /// >>> set_var = model.add_set_var(object_type=obj, target=[0, 1])
    /// >>> table = model.add_int_table([[2.5, 3.5], [-1.5, 2.5]])
    /// >>> model.add_base_case([table.min(var, set_var) == 2.5])
    #[pyo3(signature = (x, y))]
    fn min(&self, x: ArgumentUnion, y: ArgumentUnion) -> FloatExprPy {
        FloatExprPy::new(match (x, y) {
            (ArgumentUnion::Element(x), ArgumentUnion::Element(y)) => self.0.element(x, y),
            (ArgumentUnion::Set(x), ArgumentUnion::Element(y)) => self.0.min_x(x, y),
            (ArgumentUnion::Element(x), ArgumentUnion::Set(y)) => self.0.min_y(x, y),
            (ArgumentUnion::Set(x), ArgumentUnion::Set(y)) => self.0.min(x, y),
        })
    }
}

/// Table of continuous constants.
///
/// `t[x, y, z]` returns a continuous expression referring to an item where `t` is `FloatTable3D` and `x`, `y`, and `z` are `ElementExpr`, `ElementVar`, `ElementResourceVar`, or `int`.
/// If `x`, `y`, and/or `z` are/is `SetExpr`, `SetVar`, or `SetConst`, `t[x, y, z]` returns the sum of constants over `x`, `y`, and `z`.
///
/// Examples
/// --------
/// >>> import didppy as dp
/// >>> model = dp.Model()
/// >>> obj = model.add_object_type(number=2)
/// >>> var = model.add_element_var(object_type=obj, target=1)
/// >>> set_var = model.add_set_var(object_type=obj, target=[0, 1])
/// >>> table = model.add_float_table([[[2.5, 3.5], [0.5, 1.5]], [[0.5, -1.5], [2.5, 2.5]]]])
/// >>> model.add_base_case([table[var, set_var, 1] == 2.5])
#[pyclass(name = "FloatTable3D")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FloatTable3DPy(Table3DHandle<Continuous>);

impl From<FloatTable3DPy> for Table3DHandle<Continuous> {
    fn from(table: FloatTable3DPy) -> Self {
        table.0
    }
}

impl FloatTable3DPy {
    pub fn new(table: Table3DHandle<Continuous>) -> FloatTable3DPy {
        FloatTable3DPy(table)
    }
}

#[pymethods]
impl FloatTable3DPy {
    fn __getitem__(&self, index: (ArgumentUnion, ArgumentUnion, ArgumentUnion)) -> FloatExprPy {
        let (x, y, z) = index;
        FloatExprPy::new(match (x, y, z) {
            (ArgumentUnion::Element(x), ArgumentUnion::Element(y), ArgumentUnion::Element(z)) => {
                self.0.element(x, y, z)
            }
            (x, y, z) => self.0.sum(x, y, z),
        })
    }

    /// product(x, y, z)
    ///
    /// Takes the product of constants in a table over the set of indices.
    ///
    /// Parameters
    /// ----------
    /// x : tuple of int, ElementExpr, ElementVar, ElementResourceVar, SetExpr, SetVar, or SetConst
    ///     Set of indices for the first dimension.
    /// y : tuple of int, ElementExpr, ElementVar, ElementResourceVar, SetExpr, SetVar, or SetConst
    ///     Set of indices for the second dimension.
    /// z : tuple of int, ElementExpr, ElementVar, ElementResourceVar, SetExpr, SetVar, or SetConst
    ///     Set of indices for the third dimension.
    ///
    /// Returns
    /// -------
    /// FloatExpr
    ///     The product.
    ///
    /// Raises
    /// ------
    /// OverflowError
    ///     If x, y, or z is a negative integer.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> obj = model.add_object_type(number=2)
    /// >>> var = model.add_element_var(object_type=obj, target=1)
    /// >>> set_var = model.add_set_var(object_type=obj, target=[0, 1])
    /// >>> table = model.add_float_table([[[2.5, 3.5], [0.5, 1.5]], [[0.5, -1.5], [2.5, 2.5]]]])
    /// >>> model.add_base_case([table.product(var, set_var, 1) == 2.5])
    #[pyo3(signature = (x, y, z))]
    fn product(&self, x: ArgumentUnion, y: ArgumentUnion, z: ArgumentUnion) -> FloatExprPy {
        FloatExprPy::new(match (x, y, z) {
            (ArgumentUnion::Element(x), ArgumentUnion::Element(y), ArgumentUnion::Element(z)) => {
                self.0.element(x, y, z)
            }
            (x, y, z) => self.0.product(x, y, z),
        })
    }

    /// max(x, y, z)
    ///
    /// Takes the maximum of constants in a table over the set of indices.
    ///
    /// Parameters
    /// ----------
    /// x : tuple of int, ElementExpr, ElementVar, ElementResourceVar, SetExpr, SetVar, or SetConst
    ///     Set of indices for the first dimension.
    /// y : tuple of int, ElementExpr, ElementVar, ElementResourceVar, SetExpr, SetVar, or SetConst
    ///     Set of indices for the second dimension.
    /// z : tuple of int, ElementExpr, ElementVar, ElementResourceVar, SetExpr, SetVar, or SetConst
    ///     Set of indices for the third dimension.
    ///
    /// Returns
    /// -------
    /// FloatExpr
    ///     The maximum.
    ///
    /// Raises
    /// ------
    /// OverflowError
    ///     If x, y, or z is a negative integer.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> obj = model.add_object_type(number=2)
    /// >>> var = model.add_element_var(object_type=obj, target=1)
    /// >>> set_var = model.add_set_var(object_type=obj, target=[0, 1])
    /// >>> table = model.add_float_table([[[2.5, 3.5], [0.5, 1.5]], [[0.5, -1.5], [2.5, 2.5]]]])
    /// >>> model.add_base_case([table.max(var, set_var, 1) == 2.5])
    #[pyo3(signature = (x, y, z))]
    fn max(&self, x: ArgumentUnion, y: ArgumentUnion, z: ArgumentUnion) -> FloatExprPy {
        FloatExprPy::new(match (x, y, z) {
            (ArgumentUnion::Element(x), ArgumentUnion::Element(y), ArgumentUnion::Element(z)) => {
                self.0.element(x, y, z)
            }
            (x, y, z) => self.0.max(x, y, z),
        })
    }

    /// min(x, y, z)
    ///
    /// Takes the minimum of constants in a table over the set of indices.
    ///
    /// Parameters
    /// ----------
    /// x : tuple of int, ElementExpr, ElementVar, ElementResourceVar, SetExpr, SetVar, or SetConst
    ///     Set of indices for the first dimension.
    /// y : tuple of int, ElementExpr, ElementVar, ElementResourceVar, SetExpr, SetVar, or SetConst
    ///     Set of indices for the second dimension.
    /// z : tuple of int, ElementExpr, ElementVar, ElementResourceVar, SetExpr, SetVar, or SetConst
    ///     Set of indices for the third dimension.
    ///
    /// Returns
    /// -------
    /// FloatExpr
    ///     The minimum.
    ///
    /// Raises
    /// ------
    /// OverflowError
    ///     If x, y, or z is a negative integer.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> obj = model.add_object_type(number=2)
    /// >>> var = model.add_element_var(object_type=obj, target=1)
    /// >>> set_var = model.add_set_var(object_type=obj, target=[0, 1])
    /// >>> table = model.add_float_table([[[2.5, 3.5], [0.5, 1.5]], [[0.5, -1.5], [2.5, 2.5]]]])
    /// >>> model.add_base_case([table.min(var, set_var, 1) == 2.5])
    #[pyo3(signature = (x, y, z))]
    fn min(&self, x: ArgumentUnion, y: ArgumentUnion, z: ArgumentUnion) -> FloatExprPy {
        FloatExprPy::new(match (x, y, z) {
            (ArgumentUnion::Element(x), ArgumentUnion::Element(y), ArgumentUnion::Element(z)) => {
                self.0.element(x, y, z)
            }
            (x, y, z) => self.0.min(x, y, z),
        })
    }
}

/// Table of continuous constants.
///
/// `t[index]` returns a continuous expression referring to an item where `t` is `FloatTable` and `index` is a sequence of `ElementExpr`, `ElementVar`, `ElementResourceVar`, or `int`.
/// If one of `index` is `SetExpr`, `SetVar`, or `SetConst`, `t[index]` returns the sum of constants.
///
/// Examples
/// --------
/// >>> import didppy as dp
/// >>> model = dp.Model()
/// >>> obj = model.add_object_type(number=2)
/// >>> var = model.add_element_var(object_type=obj, target=1)
/// >>> set_var = model.add_set_var(object_type=obj, target=[0, 1])
/// >>> table = model.add_float_table({(0, 0, 0, 0): -1.5, (1, 1, 1, 1): 3.5}, default=2.5)
/// >>> model.add_base_case([table[0, var, set_var, 0] == 2.5])
#[pyclass(name = "FloatTable")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FloatTablePy(TableHandle<Continuous>);

impl From<FloatTablePy> for TableHandle<Continuous> {
    fn from(table: FloatTablePy) -> Self {
        table.0
    }
}

impl FloatTablePy {
    pub fn new(table: TableHandle<Continuous>) -> FloatTablePy {
        FloatTablePy(table)
    }
}

#[pymethods]
impl FloatTablePy {
    fn __getitem__(&self, index: Vec<ArgumentUnion>) -> FloatExprPy {
        let mut elements = Vec::with_capacity(index.len());
        for i in &index {
            match i {
                ArgumentUnion::Element(i) => elements.push(ElementExpression::from(i.clone())),
                _ => return FloatExprPy::new(self.0.sum(index)),
            }
        }
        FloatExprPy::new(self.0.element(elements))
    }

    /// product(index)
    ///
    /// Takes the product of constants in a table over the set of indices.
    ///
    /// Parameters
    /// ----------
    /// index : tuple of int, ElementExpr, ElementVar, ElementResourceVar, SetExpr, SetVar, or SetConst
    ///     Tuple of index sets
    ///
    /// Returns
    /// -------
    /// FloatExpr
    ///     The product.
    ///
    /// Raises
    /// ------
    /// OverflowError
    ///     If a negative integer is in `index`.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> obj = model.add_object_type(number=2)
    /// >>> var = model.add_set_var(object_type=obj, target=[0, 1])
    /// >>> set_var = model.add_set_var(object_type=obj, target=[0, 1])
    /// >>> table = model.add_float_table({(0, 0, 0, 0): -1.5, (1, 1, 1, 1): 3.5}, default=2.5)
    /// >>> model.add_base_case([table.product(0, var, set_var, 0) == 2.5])
    #[pyo3(signature = (index))]
    fn product(&self, index: Vec<ArgumentUnion>) -> FloatExprPy {
        let mut elements = Vec::with_capacity(index.len());
        for i in &index {
            match i {
                ArgumentUnion::Element(i) => elements.push(ElementExpression::from(i.clone())),
                _ => return FloatExprPy::new(self.0.product(index)),
            }
        }
        FloatExprPy::new(self.0.element(elements))
    }

    /// max(index)
    ///
    /// Takes the maximum of constants in a table over the set of indices.
    ///
    /// Parameters
    /// ----------
    /// index : tuple of int, ElementExpr, ElementVar, ElementResourceVar, SetExpr, SetVar, or SetConst
    ///     Tuple of index sets
    ///
    /// Returns
    /// -------
    /// FloatExpr
    ///     The maximum.
    ///
    /// Raises
    /// ------
    /// OverflowError
    ///     If a negative integer is in `index`.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> obj = model.add_object_type(number=2)
    /// >>> var = model.add_set_var(object_type=obj, target=[0, 1])
    /// >>> set_var = model.add_set_var(object_type=obj, target=[0, 1])
    /// >>> table = model.add_float_table({(0, 0, 0, 0): -1.5, (1, 1, 1, 1): 3.5}, default=2.5)
    /// >>> model.add_base_case([table.max(0, var, set_var, 0) == 2.5])
    #[pyo3(signature = (index))]
    fn max(&self, index: Vec<ArgumentUnion>) -> FloatExprPy {
        let mut elements = Vec::with_capacity(index.len());
        for i in &index {
            match i {
                ArgumentUnion::Element(i) => elements.push(ElementExpression::from(i.clone())),
                _ => return FloatExprPy::new(self.0.max(index)),
            }
        }
        FloatExprPy::new(self.0.element(elements))
    }

    /// min(index)
    ///
    /// Takes the minimum of constants in a table over the set of indices.
    ///
    /// Parameters
    /// ----------
    /// index : tuple of int, ElementExpr, ElementVar, ElementResourceVar, SetExpr, SetVar, or SetConst
    ///     Tuple of index sets
    ///
    /// Returns
    /// -------
    /// FloatExpr
    ///     The minimum.
    ///
    /// Raises
    /// ------
    /// OverflowError
    ///     If a negative integer is in `index`.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> obj = model.add_object_type(number=2)
    /// >>> var = model.add_set_var(object_type=obj, target=[0, 1])
    /// >>> set_var = model.add_set_var(object_type=obj, target=[0, 1])
    /// >>> table = model.add_float_table({(0, 0, 0, 0): -1.5, (1, 1, 1, 1): 3.5}, default=2.5)
    /// >>> model.add_base_case([table.min(0, var, set_var, 0) == 2.5])
    #[pyo3(signature = (index))]
    fn min(&self, index: Vec<ArgumentUnion>) -> FloatExprPy {
        let mut elements = Vec::with_capacity(index.len());
        for i in &index {
            match i {
                ArgumentUnion::Element(i) => elements.push(ElementExpression::from(i.clone())),
                _ => return FloatExprPy::new(self.0.min(index)),
            }
        }
        FloatExprPy::new(self.0.element(elements))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dypdl::expression::*;
    use rustc_hash::FxHashMap;

    #[test]
    fn element_table_1d_new() {
        let mut model = Model::default();
        let t = model.add_table_1d("t", vec![1]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(ElementTable1DPy::new(t), ElementTable1DPy(t));
    }

    #[test]
    fn element_table_1d_getitem() {
        let mut model = Model::default();
        let t = model.add_table_1d("t", vec![1usize]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = ElementTable1DPy(t);
        let i = ElementUnion::Const(0);
        assert_eq!(
            t_py.__getitem__(i),
            ElementExprPy::new(ElementExpression::Table(Box::new(
                TableExpression::Table1D(t.id(), ElementExpression::Constant(0))
            )))
        );
    }

    #[test]
    fn element_table_2d_new() {
        let mut model = Model::default();
        let t = model.add_table_2d("t", vec![vec![1]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(ElementTable2DPy::new(t), ElementTable2DPy(t));
    }

    #[test]
    fn element_table_2d_getitem() {
        let mut model = Model::default();
        let t = model.add_table_2d("t", vec![vec![1usize]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = ElementTable2DPy(t);
        let x = ElementUnion::Const(0);
        let y = ElementUnion::Const(0);
        assert_eq!(
            t_py.__getitem__((x, y)),
            ElementExprPy::new(ElementExpression::Table(Box::new(
                TableExpression::Table2D(
                    t.id(),
                    ElementExpression::Constant(0),
                    ElementExpression::Constant(0)
                )
            )))
        );
    }

    #[test]
    fn element_table_3d_new() {
        let mut model = Model::default();
        let t = model.add_table_3d("t", vec![vec![vec![1]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(ElementTable3DPy::new(t), ElementTable3DPy(t));
    }

    #[test]
    fn element_table_3d_getitem() {
        let mut model = Model::default();
        let t = model.add_table_3d("t", vec![vec![vec![1usize]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = ElementTable3DPy(t);
        let x = ElementUnion::Const(0);
        let y = ElementUnion::Const(0);
        let z = ElementUnion::Const(0);
        assert_eq!(
            t_py.__getitem__((x, y, z)),
            ElementExprPy::new(ElementExpression::Table(Box::new(
                TableExpression::Table3D(
                    t.id(),
                    ElementExpression::Constant(0),
                    ElementExpression::Constant(0),
                    ElementExpression::Constant(0)
                )
            )))
        );
    }

    #[test]
    fn element_table_new() {
        let mut model = Model::default();
        let t = model.add_table("t", FxHashMap::default(), 0usize);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(ElementTablePy::new(t), ElementTablePy(t));
    }

    #[test]
    fn element_table_getitem() {
        let mut model = Model::default();
        let t = model.add_table("t", FxHashMap::default(), 0usize);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = ElementTablePy(t);
        let index = vec![
            ElementUnion::Const(0),
            ElementUnion::Const(0),
            ElementUnion::Const(0),
            ElementUnion::Const(0),
        ];
        assert_eq!(
            t_py.__getitem__(index),
            ElementExprPy::new(ElementExpression::Table(Box::new(TableExpression::Table(
                t.id(),
                vec![
                    ElementExpression::Constant(0),
                    ElementExpression::Constant(0),
                    ElementExpression::Constant(0),
                    ElementExpression::Constant(0),
                ]
            ))))
        );
    }

    #[test]
    fn set_table_1d_new() {
        let mut model = Model::default();
        let t = model.add_table_1d("t", vec![Set::with_capacity(10)]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(SetTable1DPy::new(t.clone(), 10), SetTable1DPy(t, 10));
    }

    #[test]
    fn set_table_1d_get_capacity_of_set() {
        let mut model = Model::default();
        let t = model.add_table_1d("t", vec![Set::with_capacity(10)]);
        assert!(t.is_ok());
        let t = SetTable1DPy(t.unwrap(), 10);
        assert_eq!(t.get_capacity_of_set(), 10);
    }

    #[test]
    fn set_table_1d_getitem() {
        let mut model = Model::default();
        let t = model.add_table_1d("t", vec![Set::with_capacity(10)]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = SetTable1DPy(t.clone(), 10);
        let i = ElementUnion::Const(0);
        assert_eq!(
            t_py.__getitem__(i),
            SetExprPy::new(SetExpression::Reference(ReferenceExpression::Table(
                TableExpression::Table1D(t.id(), ElementExpression::Constant(0))
            )))
        );
    }

    #[test]
    fn set_table_1d_union() {
        let mut model = Model::default();
        let t = model.add_table_1d("t", vec![Set::with_capacity(10)]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = SetTable1DPy(t.clone(), 10);
        let x = ArgumentUnion::Element(ElementUnion::Const(0));
        assert_eq!(
            t_py.union(x),
            SetExprPy::new(SetExpression::Reduce(SetReduceExpression::Table1D(
                SetReduceOperator::Union,
                10,
                t.id(),
                Box::new(ArgumentExpression::Element(ElementExpression::Constant(0)))
            )))
        );
    }

    #[test]
    fn set_table_1d_intersection() {
        let mut model = Model::default();
        let t = model.add_table_1d("t", vec![Set::with_capacity(10)]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = SetTable1DPy(t.clone(), 10);
        let x = ArgumentUnion::Element(ElementUnion::Const(0));
        assert_eq!(
            t_py.intersection(x),
            SetExprPy::new(SetExpression::Reduce(SetReduceExpression::Table1D(
                SetReduceOperator::Intersection,
                10,
                t.id(),
                Box::new(ArgumentExpression::Element(ElementExpression::Constant(0)))
            )))
        );
    }

    #[test]
    fn set_table_1d_symmetric_difference() {
        let mut model = Model::default();
        let t = model.add_table_1d("t", vec![Set::with_capacity(10)]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = SetTable1DPy(t.clone(), 10);
        let x = ArgumentUnion::Element(ElementUnion::Const(0));
        assert_eq!(
            t_py.symmetric_difference(x),
            SetExprPy::new(SetExpression::Reduce(SetReduceExpression::Table1D(
                SetReduceOperator::SymmetricDifference,
                10,
                t.id(),
                Box::new(ArgumentExpression::Element(ElementExpression::Constant(0)))
            )))
        );
    }

    #[test]
    fn set_table_2d_new() {
        let mut model = Model::default();
        let t = model.add_table_2d("t", vec![vec![Set::with_capacity(10)]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(SetTable2DPy::new(t.clone(), 10), SetTable2DPy(t, 10));
    }

    #[test]
    fn set_table_2d_get_capacity_of_set() {
        let mut model = Model::default();
        let t = model.add_table_2d("t", vec![vec![Set::with_capacity(10)]]);
        assert!(t.is_ok());
        let t = SetTable2DPy(t.unwrap(), 10);
        assert_eq!(t.get_capacity_of_set(), 10);
    }

    #[test]
    fn set_table_2d_getitem() {
        let mut model = Model::default();
        let t = model.add_table_2d("t", vec![vec![Set::with_capacity(10)]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = SetTable2DPy(t.clone(), 10);
        let x = ElementUnion::Const(0);
        let y = ElementUnion::Const(0);
        assert_eq!(
            t_py.__getitem__((x, y)),
            SetExprPy::new(SetExpression::Reference(ReferenceExpression::Table(
                TableExpression::Table2D(
                    t.id(),
                    ElementExpression::Constant(0),
                    ElementExpression::Constant(0)
                )
            )))
        );
    }

    #[test]
    fn set_table_2d_union() {
        let mut model = Model::default();
        let t = model.add_table_2d("t", vec![vec![Set::with_capacity(10)]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = SetTable2DPy(t.clone(), 10);
        let x = ArgumentUnion::Element(ElementUnion::Const(0));
        let y = ArgumentUnion::Element(ElementUnion::Const(0));
        assert_eq!(
            t_py.union(x, y),
            SetExprPy::new(SetExpression::Reduce(SetReduceExpression::Table2D(
                SetReduceOperator::Union,
                10,
                t.id(),
                Box::new(ArgumentExpression::Element(ElementExpression::Constant(0))),
                Box::new(ArgumentExpression::Element(ElementExpression::Constant(0)))
            )))
        );
    }

    #[test]
    fn set_table_2d_intersection() {
        let mut model = Model::default();
        let t = model.add_table_2d("t", vec![vec![Set::with_capacity(10)]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = SetTable2DPy(t.clone(), 10);
        let x = ArgumentUnion::Element(ElementUnion::Const(0));
        let y = ArgumentUnion::Element(ElementUnion::Const(0));
        assert_eq!(
            t_py.intersection(x, y),
            SetExprPy::new(SetExpression::Reduce(SetReduceExpression::Table2D(
                SetReduceOperator::Intersection,
                10,
                t.id(),
                Box::new(ArgumentExpression::Element(ElementExpression::Constant(0))),
                Box::new(ArgumentExpression::Element(ElementExpression::Constant(0)))
            )))
        );
    }

    #[test]
    fn set_table_2d_symmetric_difference() {
        let mut model = Model::default();
        let t = model.add_table_2d("t", vec![vec![Set::with_capacity(10)]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = SetTable2DPy(t.clone(), 10);
        let x = ArgumentUnion::Element(ElementUnion::Const(0));
        let y = ArgumentUnion::Element(ElementUnion::Const(0));
        assert_eq!(
            t_py.symmetric_difference(x, y),
            SetExprPy::new(SetExpression::Reduce(SetReduceExpression::Table2D(
                SetReduceOperator::SymmetricDifference,
                10,
                t.id(),
                Box::new(ArgumentExpression::Element(ElementExpression::Constant(0))),
                Box::new(ArgumentExpression::Element(ElementExpression::Constant(0)))
            )))
        );
    }

    #[test]
    fn set_table_3d_new() {
        let mut model = Model::default();
        let t = model.add_table_3d("t", vec![vec![vec![Set::with_capacity(10)]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(SetTable3DPy::new(t.clone(), 10), SetTable3DPy(t, 10));
    }

    #[test]
    fn set_table_3d_get_capacity_of_set() {
        let mut model = Model::default();
        let t = model.add_table_3d("t", vec![vec![vec![Set::with_capacity(10)]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t = SetTable3DPy(t, 10);
        assert_eq!(t.get_capacity_of_set(), 10);
    }

    #[test]
    fn set_table_3d_getitem() {
        let mut model = Model::default();
        let t = model.add_table_3d("t", vec![vec![vec![Set::with_capacity(10)]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = SetTable3DPy(t.clone(), 10);
        let x = ElementUnion::Const(0);
        let y = ElementUnion::Const(0);
        let z = ElementUnion::Const(0);
        assert_eq!(
            t_py.__getitem__((x, y, z)),
            SetExprPy::new(SetExpression::Reference(ReferenceExpression::Table(
                TableExpression::Table3D(
                    t.id(),
                    ElementExpression::Constant(0),
                    ElementExpression::Constant(0),
                    ElementExpression::Constant(0)
                )
            )))
        );
    }

    #[test]
    fn set_table_3d_union() {
        let mut model = Model::default();
        let t = model.add_table_3d("t", vec![vec![vec![Set::with_capacity(10)]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = SetTable3DPy(t.clone(), 10);
        let x = ArgumentUnion::Element(ElementUnion::Const(0));
        let y = ArgumentUnion::Element(ElementUnion::Const(0));
        let z = ArgumentUnion::Element(ElementUnion::Const(0));
        assert_eq!(
            t_py.union(x, y, z),
            SetExprPy::new(SetExpression::Reduce(SetReduceExpression::Table3D(
                SetReduceOperator::Union,
                10,
                t.id(),
                Box::new(ArgumentExpression::Element(ElementExpression::Constant(0))),
                Box::new(ArgumentExpression::Element(ElementExpression::Constant(0))),
                Box::new(ArgumentExpression::Element(ElementExpression::Constant(0)))
            )))
        );
    }

    #[test]
    fn set_table_3d_intersection() {
        let mut model = Model::default();
        let t = model.add_table_3d("t", vec![vec![vec![Set::with_capacity(10)]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = SetTable3DPy(t.clone(), 10);
        let x = ArgumentUnion::Element(ElementUnion::Const(0));
        let y = ArgumentUnion::Element(ElementUnion::Const(0));
        let z = ArgumentUnion::Element(ElementUnion::Const(0));
        assert_eq!(
            t_py.intersection(x, y, z),
            SetExprPy::new(SetExpression::Reduce(SetReduceExpression::Table3D(
                SetReduceOperator::Intersection,
                10,
                t.id(),
                Box::new(ArgumentExpression::Element(ElementExpression::Constant(0))),
                Box::new(ArgumentExpression::Element(ElementExpression::Constant(0))),
                Box::new(ArgumentExpression::Element(ElementExpression::Constant(0)))
            )))
        );
    }

    #[test]
    fn set_table_3d_symmetric_difference() {
        let mut model = Model::default();
        let t = model.add_table_3d("t", vec![vec![vec![Set::with_capacity(10)]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = SetTable3DPy(t.clone(), 10);
        let x = ArgumentUnion::Element(ElementUnion::Const(0));
        let y = ArgumentUnion::Element(ElementUnion::Const(0));
        let z = ArgumentUnion::Element(ElementUnion::Const(0));
        assert_eq!(
            t_py.symmetric_difference(x, y, z),
            SetExprPy::new(SetExpression::Reduce(SetReduceExpression::Table3D(
                SetReduceOperator::SymmetricDifference,
                10,
                t.id(),
                Box::new(ArgumentExpression::Element(ElementExpression::Constant(0))),
                Box::new(ArgumentExpression::Element(ElementExpression::Constant(0))),
                Box::new(ArgumentExpression::Element(ElementExpression::Constant(0)))
            )))
        );
    }

    #[test]
    fn set_table_new() {
        let mut model = Model::default();
        let t = model.add_table("t", FxHashMap::default(), Set::with_capacity(10));
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(SetTablePy::new(t.clone(), 10), SetTablePy(t, 10));
    }

    #[test]
    fn set_table_get_capacity_of_set() {
        let mut model = Model::default();
        let t = model.add_table("t", FxHashMap::default(), Set::with_capacity(10));
        assert!(t.is_ok());
        let t = t.unwrap();
        let t = SetTablePy(t, 10);
        assert_eq!(t.get_capacity_of_set(), 10);
    }

    #[test]
    fn set_table_getitem() {
        let mut model = Model::default();
        let t = model.add_table("t", FxHashMap::default(), Set::with_capacity(10));
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = SetTablePy(t.clone(), 10);
        let index = vec![
            ElementUnion::Const(0),
            ElementUnion::Const(0),
            ElementUnion::Const(0),
            ElementUnion::Const(0),
        ];
        assert_eq!(
            t_py.__getitem__(index),
            SetExprPy::new(SetExpression::Reference(ReferenceExpression::Table(
                TableExpression::Table(
                    t.id(),
                    vec![
                        ElementExpression::Constant(0),
                        ElementExpression::Constant(0),
                        ElementExpression::Constant(0),
                        ElementExpression::Constant(0),
                    ]
                )
            )))
        );
    }

    #[test]
    fn set_table_union() {
        let mut model = Model::default();
        let t = model.add_table("t", FxHashMap::default(), Set::with_capacity(10));
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = SetTablePy(t.clone(), 10);
        let indices = vec![
            ArgumentUnion::Element(ElementUnion::Const(0)),
            ArgumentUnion::Element(ElementUnion::Const(0)),
            ArgumentUnion::Element(ElementUnion::Const(0)),
            ArgumentUnion::Element(ElementUnion::Const(0)),
        ];
        assert_eq!(
            t_py.union(indices),
            SetExprPy::new(SetExpression::Reduce(SetReduceExpression::Table(
                SetReduceOperator::Union,
                10,
                t.id(),
                vec![
                    ArgumentExpression::Element(ElementExpression::Constant(0)),
                    ArgumentExpression::Element(ElementExpression::Constant(0)),
                    ArgumentExpression::Element(ElementExpression::Constant(0)),
                    ArgumentExpression::Element(ElementExpression::Constant(0))
                ]
            )))
        );
    }

    #[test]
    fn set_table_intersection() {
        let mut model = Model::default();
        let t = model.add_table("t", FxHashMap::default(), Set::with_capacity(10));
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = SetTablePy(t.clone(), 10);
        let indices = vec![
            ArgumentUnion::Element(ElementUnion::Const(0)),
            ArgumentUnion::Element(ElementUnion::Const(0)),
            ArgumentUnion::Element(ElementUnion::Const(0)),
            ArgumentUnion::Element(ElementUnion::Const(0)),
        ];
        assert_eq!(
            t_py.intersection(indices),
            SetExprPy::new(SetExpression::Reduce(SetReduceExpression::Table(
                SetReduceOperator::Intersection,
                10,
                t.id(),
                vec![
                    ArgumentExpression::Element(ElementExpression::Constant(0)),
                    ArgumentExpression::Element(ElementExpression::Constant(0)),
                    ArgumentExpression::Element(ElementExpression::Constant(0)),
                    ArgumentExpression::Element(ElementExpression::Constant(0))
                ]
            )))
        );
    }

    #[test]
    fn set_table_symmetric_difference() {
        let mut model = Model::default();
        let t = model.add_table("t", FxHashMap::default(), Set::with_capacity(10));
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = SetTablePy(t.clone(), 10);
        let indices = vec![
            ArgumentUnion::Element(ElementUnion::Const(0)),
            ArgumentUnion::Element(ElementUnion::Const(0)),
            ArgumentUnion::Element(ElementUnion::Const(0)),
            ArgumentUnion::Element(ElementUnion::Const(0)),
        ];
        assert_eq!(
            t_py.symmetric_difference(indices),
            SetExprPy::new(SetExpression::Reduce(SetReduceExpression::Table(
                SetReduceOperator::SymmetricDifference,
                10,
                t.id(),
                vec![
                    ArgumentExpression::Element(ElementExpression::Constant(0)),
                    ArgumentExpression::Element(ElementExpression::Constant(0)),
                    ArgumentExpression::Element(ElementExpression::Constant(0)),
                    ArgumentExpression::Element(ElementExpression::Constant(0))
                ]
            )))
        );
    }

    #[test]
    fn bool_table_1d_new() {
        let mut model = Model::default();
        let t = model.add_table_1d("t", vec![true]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(BoolTable1DPy::new(t), BoolTable1DPy(t));
    }

    #[test]
    fn bool_table_1d_getitem() {
        let mut model = Model::default();
        let t = model.add_table_1d("t", vec![true]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = BoolTable1DPy(t);
        let i = ElementUnion::Const(0);
        assert_eq!(
            t_py.__getitem__(i),
            ConditionPy::new(Condition::Table(Box::new(TableExpression::Table1D(
                t.id(),
                ElementExpression::Constant(0)
            ))))
        );
    }

    #[test]
    fn bool_table_2d_new() {
        let mut model = Model::default();
        let t = model.add_table_2d("t", vec![vec![true]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(BoolTable2DPy::new(t), BoolTable2DPy(t));
    }

    #[test]
    fn bool_table_2d_getitem() {
        let mut model = Model::default();
        let t = model.add_table_2d("t", vec![vec![true]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = BoolTable2DPy(t);
        let x = ElementUnion::Const(0);
        let y = ElementUnion::Const(0);
        assert_eq!(
            t_py.__getitem__((x, y)),
            ConditionPy::new(Condition::Table(Box::new(TableExpression::Table2D(
                t.id(),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0)
            ))))
        );
    }

    #[test]
    fn bool_table_3d_new() {
        let mut model = Model::default();
        let t = model.add_table_3d("t", vec![vec![vec![true]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(BoolTable3DPy::new(t), BoolTable3DPy(t));
    }

    #[test]
    fn bool_table_3d_getitem() {
        let mut model = Model::default();
        let t = model.add_table_3d("t", vec![vec![vec![true]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = BoolTable3DPy(t);
        let x = ElementUnion::Const(0);
        let y = ElementUnion::Const(0);
        let z = ElementUnion::Const(0);
        assert_eq!(
            t_py.__getitem__((x, y, z)),
            ConditionPy::new(Condition::Table(Box::new(TableExpression::Table3D(
                t.id(),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0)
            ))))
        );
    }

    #[test]
    fn bool_table_new() {
        let mut model = Model::default();
        let t = model.add_table("t", FxHashMap::default(), false);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(BoolTablePy::new(t), BoolTablePy(t));
    }

    #[test]
    fn bool_table_getitem() {
        let mut model = Model::default();
        let t = model.add_table("t", FxHashMap::default(), false);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = BoolTablePy(t);
        let index = vec![
            ElementUnion::Const(0),
            ElementUnion::Const(0),
            ElementUnion::Const(0),
            ElementUnion::Const(0),
        ];
        assert_eq!(
            t_py.__getitem__(index),
            ConditionPy::new(Condition::Table(Box::new(TableExpression::Table(
                t.id(),
                vec![
                    ElementExpression::Constant(0),
                    ElementExpression::Constant(0),
                    ElementExpression::Constant(0),
                    ElementExpression::Constant(0),
                ]
            ))))
        );
    }

    #[test]
    fn argument_expression_from_table_index_element() {
        let i = ArgumentUnion::Element(ElementUnion::Const(0));
        assert_eq!(
            ArgumentExpression::from(i),
            ArgumentExpression::Element(ElementExpression::Constant(0))
        );
    }

    #[test]
    fn argument_expression_from_table_index_set() {
        let i = ArgumentUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10))));
        assert_eq!(
            ArgumentExpression::from(i),
            ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Constant(
                Set::with_capacity(10)
            )))
        );
    }

    #[test]
    fn int_table_1d_new() {
        let mut model = Model::default();
        let t = model.add_table_1d("t", vec![1]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(IntTable1DPy::new(t), IntTable1DPy(t));
    }

    #[test]
    fn int_table_1d_element() {
        let mut model = Model::default();
        let t = model.add_table_1d("t", vec![1]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = IntTable1DPy(t);
        let i = ArgumentUnion::Element(ElementUnion::Const(0));
        assert_eq!(
            t_py.__getitem__(i),
            IntExprPy::new(IntegerExpression::Table(Box::new(
                NumericTableExpression::Table1D(t.id(), ElementExpression::Constant(0))
            )))
        );
    }

    #[test]
    fn int_table_1d_sum() {
        let mut model = Model::default();
        let t = model.add_table_1d("t", vec![1]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = IntTable1DPy(t);
        let i = ArgumentUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10))));
        assert_eq!(
            t_py.__getitem__(i),
            IntExprPy::new(IntegerExpression::Table(Box::new(
                NumericTableExpression::Table1DReduce(
                    ReduceOperator::Sum,
                    t.id(),
                    SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(10)))
                )
            )))
        );
    }

    #[test]
    fn int_table_1d_product() {
        let mut model = Model::default();
        let t = model.add_table_1d("t", vec![1]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = IntTable1DPy(t);
        let i = SetUnion::Const(SetConstPy::new(Set::with_capacity(10)));
        assert_eq!(
            t_py.product(i),
            IntExprPy::new(IntegerExpression::Table(Box::new(
                NumericTableExpression::Table1DReduce(
                    ReduceOperator::Product,
                    t.id(),
                    SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(10)))
                )
            )))
        );
    }

    #[test]
    fn int_table_1d_max() {
        let mut model = Model::default();
        let t = model.add_table_1d("t", vec![1]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = IntTable1DPy(t);
        let i = SetUnion::Const(SetConstPy::new(Set::with_capacity(10)));
        assert_eq!(
            t_py.max(i),
            IntExprPy::new(IntegerExpression::Table(Box::new(
                NumericTableExpression::Table1DReduce(
                    ReduceOperator::Max,
                    t.id(),
                    SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(10)))
                )
            )))
        );
    }

    #[test]
    fn int_table_1d_min() {
        let mut model = Model::default();
        let t = model.add_table_1d("t", vec![1]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = IntTable1DPy(t);
        let i = SetUnion::Const(SetConstPy::new(Set::with_capacity(10)));
        assert_eq!(
            t_py.min(i),
            IntExprPy::new(IntegerExpression::Table(Box::new(
                NumericTableExpression::Table1DReduce(
                    ReduceOperator::Min,
                    t.id(),
                    SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(10)))
                )
            )))
        );
    }

    #[test]
    fn int_table_2d_new() {
        let mut model = Model::default();
        let t = model.add_table_2d("t", vec![vec![1]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(IntTable2DPy::new(t), IntTable2DPy(t));
    }

    #[test]
    fn int_table_2d_element() {
        let mut model = Model::default();
        let t = model.add_table_2d("t", vec![vec![1]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = IntTable2DPy(t);
        let x = ArgumentUnion::Element(ElementUnion::Const(0));
        let y = ArgumentUnion::Element(ElementUnion::Const(0));
        assert_eq!(
            t_py.__getitem__((x, y)),
            IntExprPy::new(IntegerExpression::Table(Box::new(
                NumericTableExpression::Table2D(
                    t.id(),
                    ElementExpression::Constant(0),
                    ElementExpression::Constant(0)
                )
            )))
        );
    }

    #[test]
    fn int_table_2d_sum_x() {
        let mut model = Model::default();
        let t = model.add_table_2d("t", vec![vec![1]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = IntTable2DPy(t);
        let x = ArgumentUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10))));
        let y = ArgumentUnion::Element(ElementUnion::Const(0));
        assert_eq!(
            t_py.__getitem__((x, y)),
            IntExprPy::new(IntegerExpression::Table(Box::new(
                NumericTableExpression::Table2DReduceX(
                    ReduceOperator::Sum,
                    t.id(),
                    SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(10))),
                    ElementExpression::Constant(0)
                )
            )))
        );
    }

    #[test]
    fn int_table_2d_sum_y() {
        let mut model = Model::default();
        let t = model.add_table_2d("t", vec![vec![1]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = IntTable2DPy(t);
        let x = ArgumentUnion::Element(ElementUnion::Const(0));
        let y = ArgumentUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10))));
        assert_eq!(
            t_py.__getitem__((x, y)),
            IntExprPy::new(IntegerExpression::Table(Box::new(
                NumericTableExpression::Table2DReduceY(
                    ReduceOperator::Sum,
                    t.id(),
                    ElementExpression::Constant(0),
                    SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(10))),
                )
            )))
        );
    }

    #[test]
    fn int_table_2d_sum() {
        let mut model = Model::default();
        let t = model.add_table_2d("t", vec![vec![1]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = IntTable2DPy(t);
        let x = ArgumentUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10))));
        let y = ArgumentUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10))));
        assert_eq!(
            t_py.__getitem__((x, y)),
            IntExprPy::new(IntegerExpression::Table(Box::new(
                NumericTableExpression::Table2DReduce(
                    ReduceOperator::Sum,
                    t.id(),
                    SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(10))),
                    SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(10))),
                )
            )))
        );
    }

    #[test]
    fn int_table_2d_product_element() {
        let mut model = Model::default();
        let t = model.add_table_2d("t", vec![vec![1]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = IntTable2DPy(t);
        let x = ArgumentUnion::Element(ElementUnion::Const(0));
        let y = ArgumentUnion::Element(ElementUnion::Const(0));
        assert_eq!(
            t_py.product(x, y),
            IntExprPy::new(IntegerExpression::Table(Box::new(
                NumericTableExpression::Table2D(
                    t.id(),
                    ElementExpression::Constant(0),
                    ElementExpression::Constant(0)
                )
            )))
        );
    }

    #[test]
    fn int_table_2d_product_x() {
        let mut model = Model::default();
        let t = model.add_table_2d("t", vec![vec![1]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = IntTable2DPy(t);
        let x = ArgumentUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10))));
        let y = ArgumentUnion::Element(ElementUnion::Const(0));
        assert_eq!(
            t_py.product(x, y),
            IntExprPy::new(IntegerExpression::Table(Box::new(
                NumericTableExpression::Table2DReduceX(
                    ReduceOperator::Product,
                    t.id(),
                    SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(10))),
                    ElementExpression::Constant(0)
                )
            )))
        );
    }

    #[test]
    fn int_table_2d_product_y() {
        let mut model = Model::default();
        let t = model.add_table_2d("t", vec![vec![1]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = IntTable2DPy(t);
        let x = ArgumentUnion::Element(ElementUnion::Const(0));
        let y = ArgumentUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10))));
        assert_eq!(
            t_py.product(x, y),
            IntExprPy::new(IntegerExpression::Table(Box::new(
                NumericTableExpression::Table2DReduceY(
                    ReduceOperator::Product,
                    t.id(),
                    ElementExpression::Constant(0),
                    SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(10))),
                )
            )))
        );
    }

    #[test]
    fn int_table_2d_product() {
        let mut model = Model::default();
        let t = model.add_table_2d("t", vec![vec![1]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = IntTable2DPy(t);
        let x = ArgumentUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10))));
        let y = ArgumentUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10))));
        assert_eq!(
            t_py.product(x, y),
            IntExprPy::new(IntegerExpression::Table(Box::new(
                NumericTableExpression::Table2DReduce(
                    ReduceOperator::Product,
                    t.id(),
                    SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(10))),
                    SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(10))),
                )
            )))
        );
    }

    #[test]
    fn int_table_2d_max_element() {
        let mut model = Model::default();
        let t = model.add_table_2d("t", vec![vec![1]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = IntTable2DPy(t);
        let x = ArgumentUnion::Element(ElementUnion::Const(0));
        let y = ArgumentUnion::Element(ElementUnion::Const(0));
        assert_eq!(
            t_py.max(x, y),
            IntExprPy::new(IntegerExpression::Table(Box::new(
                NumericTableExpression::Table2D(
                    t.id(),
                    ElementExpression::Constant(0),
                    ElementExpression::Constant(0)
                )
            )))
        );
    }

    #[test]
    fn int_table_2d_max_x() {
        let mut model = Model::default();
        let t = model.add_table_2d("t", vec![vec![1]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = IntTable2DPy(t);
        let x = ArgumentUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10))));
        let y = ArgumentUnion::Element(ElementUnion::Const(0));
        assert_eq!(
            t_py.max(x, y),
            IntExprPy::new(IntegerExpression::Table(Box::new(
                NumericTableExpression::Table2DReduceX(
                    ReduceOperator::Max,
                    t.id(),
                    SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(10))),
                    ElementExpression::Constant(0)
                )
            )))
        );
    }

    #[test]
    fn int_table_2d_max_y() {
        let mut model = Model::default();
        let t = model.add_table_2d("t", vec![vec![1]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = IntTable2DPy(t);
        let x = ArgumentUnion::Element(ElementUnion::Const(0));
        let y = ArgumentUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10))));
        assert_eq!(
            t_py.max(x, y),
            IntExprPy::new(IntegerExpression::Table(Box::new(
                NumericTableExpression::Table2DReduceY(
                    ReduceOperator::Max,
                    t.id(),
                    ElementExpression::Constant(0),
                    SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(10))),
                )
            )))
        );
    }

    #[test]
    fn int_table_2d_max() {
        let mut model = Model::default();
        let t = model.add_table_2d("t", vec![vec![1]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = IntTable2DPy(t);
        let x = ArgumentUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10))));
        let y = ArgumentUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10))));
        assert_eq!(
            t_py.max(x, y),
            IntExprPy::new(IntegerExpression::Table(Box::new(
                NumericTableExpression::Table2DReduce(
                    ReduceOperator::Max,
                    t.id(),
                    SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(10))),
                    SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(10))),
                )
            )))
        );
    }

    #[test]
    fn int_table_2d_min_element() {
        let mut model = Model::default();
        let t = model.add_table_2d("t", vec![vec![1]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = IntTable2DPy(t);
        let x = ArgumentUnion::Element(ElementUnion::Const(0));
        let y = ArgumentUnion::Element(ElementUnion::Const(0));
        assert_eq!(
            t_py.min(x, y),
            IntExprPy::new(IntegerExpression::Table(Box::new(
                NumericTableExpression::Table2D(
                    t.id(),
                    ElementExpression::Constant(0),
                    ElementExpression::Constant(0)
                )
            )))
        );
    }

    #[test]
    fn int_table_2d_min_x() {
        let mut model = Model::default();
        let t = model.add_table_2d("t", vec![vec![1]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = IntTable2DPy(t);
        let x = ArgumentUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10))));
        let y = ArgumentUnion::Element(ElementUnion::Const(0));
        assert_eq!(
            t_py.min(x, y),
            IntExprPy::new(IntegerExpression::Table(Box::new(
                NumericTableExpression::Table2DReduceX(
                    ReduceOperator::Min,
                    t.id(),
                    SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(10))),
                    ElementExpression::Constant(0)
                )
            )))
        );
    }

    #[test]
    fn int_table_2d_min_y() {
        let mut model = Model::default();
        let t = model.add_table_2d("t", vec![vec![1]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = IntTable2DPy(t);
        let x = ArgumentUnion::Element(ElementUnion::Const(0));
        let y = ArgumentUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10))));
        assert_eq!(
            t_py.min(x, y),
            IntExprPy::new(IntegerExpression::Table(Box::new(
                NumericTableExpression::Table2DReduceY(
                    ReduceOperator::Min,
                    t.id(),
                    ElementExpression::Constant(0),
                    SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(10))),
                )
            )))
        );
    }

    #[test]
    fn int_table_2d_min() {
        let mut model = Model::default();
        let t = model.add_table_2d("t", vec![vec![1]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = IntTable2DPy(t);
        let x = ArgumentUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10))));
        let y = ArgumentUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10))));
        assert_eq!(
            t_py.min(x, y),
            IntExprPy::new(IntegerExpression::Table(Box::new(
                NumericTableExpression::Table2DReduce(
                    ReduceOperator::Min,
                    t.id(),
                    SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(10))),
                    SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(10))),
                )
            )))
        );
    }

    #[test]
    fn int_table_3d_new() {
        let mut model = Model::default();
        let t = model.add_table_3d("t", vec![vec![vec![1]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(IntTable3DPy::new(t), IntTable3DPy(t));
    }

    #[test]
    fn int_table_3d_element() {
        let mut model = Model::default();
        let t = model.add_table_3d("t", vec![vec![vec![1]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = IntTable3DPy(t);
        let x = ArgumentUnion::Element(ElementUnion::Const(0));
        let y = ArgumentUnion::Element(ElementUnion::Const(0));
        let z = ArgumentUnion::Element(ElementUnion::Const(0));
        assert_eq!(
            t_py.__getitem__((x, y, z)),
            IntExprPy::new(IntegerExpression::Table(Box::new(
                NumericTableExpression::Table3D(
                    t.id(),
                    ElementExpression::Constant(0),
                    ElementExpression::Constant(0),
                    ElementExpression::Constant(0),
                )
            )))
        );
    }

    #[test]
    fn int_table_3d_sum() {
        let mut model = Model::default();
        let t = model.add_table_3d("t", vec![vec![vec![1]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = IntTable3DPy(t);
        let x = ArgumentUnion::Element(ElementUnion::Const(0));
        let y = ArgumentUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10))));
        let z = ArgumentUnion::Element(ElementUnion::Const(0));
        assert_eq!(
            t_py.__getitem__((x, y, z)),
            IntExprPy::new(IntegerExpression::Table(Box::new(
                NumericTableExpression::Table3DReduce(
                    ReduceOperator::Sum,
                    t.id(),
                    ArgumentExpression::Element(ElementExpression::Constant(0)),
                    ArgumentExpression::Set(SetExpression::Reference(
                        ReferenceExpression::Constant(Set::with_capacity(10))
                    )),
                    ArgumentExpression::Element(ElementExpression::Constant(0)),
                )
            )))
        );
    }

    #[test]
    fn int_table_3d_product_element() {
        let mut model = Model::default();
        let t = model.add_table_3d("t", vec![vec![vec![1]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = IntTable3DPy(t);
        let x = ArgumentUnion::Element(ElementUnion::Const(0));
        let y = ArgumentUnion::Element(ElementUnion::Const(0));
        let z = ArgumentUnion::Element(ElementUnion::Const(0));
        assert_eq!(
            t_py.product(x, y, z),
            IntExprPy::new(IntegerExpression::Table(Box::new(
                NumericTableExpression::Table3D(
                    t.id(),
                    ElementExpression::Constant(0),
                    ElementExpression::Constant(0),
                    ElementExpression::Constant(0),
                )
            )))
        );
    }

    #[test]
    fn int_table_3d_product() {
        let mut model = Model::default();
        let t = model.add_table_3d("t", vec![vec![vec![1]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = IntTable3DPy(t);
        let x = ArgumentUnion::Element(ElementUnion::Const(0));
        let y = ArgumentUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10))));
        let z = ArgumentUnion::Element(ElementUnion::Const(0));
        assert_eq!(
            t_py.product(x, y, z),
            IntExprPy::new(IntegerExpression::Table(Box::new(
                NumericTableExpression::Table3DReduce(
                    ReduceOperator::Product,
                    t.id(),
                    ArgumentExpression::Element(ElementExpression::Constant(0)),
                    ArgumentExpression::Set(SetExpression::Reference(
                        ReferenceExpression::Constant(Set::with_capacity(10))
                    )),
                    ArgumentExpression::Element(ElementExpression::Constant(0)),
                )
            )))
        );
    }

    #[test]
    fn int_table_3d_max_element() {
        let mut model = Model::default();
        let t = model.add_table_3d("t", vec![vec![vec![1]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = IntTable3DPy(t);
        let x = ArgumentUnion::Element(ElementUnion::Const(0));
        let y = ArgumentUnion::Element(ElementUnion::Const(0));
        let z = ArgumentUnion::Element(ElementUnion::Const(0));
        assert_eq!(
            t_py.max(x, y, z),
            IntExprPy::new(IntegerExpression::Table(Box::new(
                NumericTableExpression::Table3D(
                    t.id(),
                    ElementExpression::Constant(0),
                    ElementExpression::Constant(0),
                    ElementExpression::Constant(0),
                )
            )))
        );
    }

    #[test]
    fn int_table_3d_max() {
        let mut model = Model::default();
        let t = model.add_table_3d("t", vec![vec![vec![1]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = IntTable3DPy(t);
        let x = ArgumentUnion::Element(ElementUnion::Const(0));
        let y = ArgumentUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10))));
        let z = ArgumentUnion::Element(ElementUnion::Const(0));
        assert_eq!(
            t_py.max(x, y, z),
            IntExprPy::new(IntegerExpression::Table(Box::new(
                NumericTableExpression::Table3DReduce(
                    ReduceOperator::Max,
                    t.id(),
                    ArgumentExpression::Element(ElementExpression::Constant(0)),
                    ArgumentExpression::Set(SetExpression::Reference(
                        ReferenceExpression::Constant(Set::with_capacity(10))
                    )),
                    ArgumentExpression::Element(ElementExpression::Constant(0)),
                )
            )))
        );
    }

    #[test]
    fn int_table_3d_min_element() {
        let mut model = Model::default();
        let t = model.add_table_3d("t", vec![vec![vec![1]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = IntTable3DPy(t);
        let x = ArgumentUnion::Element(ElementUnion::Const(0));
        let y = ArgumentUnion::Element(ElementUnion::Const(0));
        let z = ArgumentUnion::Element(ElementUnion::Const(0));
        assert_eq!(
            t_py.min(x, y, z),
            IntExprPy::new(IntegerExpression::Table(Box::new(
                NumericTableExpression::Table3D(
                    t.id(),
                    ElementExpression::Constant(0),
                    ElementExpression::Constant(0),
                    ElementExpression::Constant(0),
                )
            )))
        );
    }

    #[test]
    fn int_table_3d_min() {
        let mut model = Model::default();
        let t = model.add_table_3d("t", vec![vec![vec![1]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = IntTable3DPy(t);
        let x = ArgumentUnion::Element(ElementUnion::Const(0));
        let y = ArgumentUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10))));
        let z = ArgumentUnion::Element(ElementUnion::Const(0));
        assert_eq!(
            t_py.min(x, y, z),
            IntExprPy::new(IntegerExpression::Table(Box::new(
                NumericTableExpression::Table3DReduce(
                    ReduceOperator::Min,
                    t.id(),
                    ArgumentExpression::Element(ElementExpression::Constant(0)),
                    ArgumentExpression::Set(SetExpression::Reference(
                        ReferenceExpression::Constant(Set::with_capacity(10))
                    )),
                    ArgumentExpression::Element(ElementExpression::Constant(0)),
                )
            )))
        );
    }

    #[test]
    fn int_table_new() {
        let mut model = Model::default();
        let t = model.add_table("t", FxHashMap::default(), 0);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(IntTablePy::new(t), IntTablePy(t));
    }

    #[test]
    fn int_table_element() {
        let mut model = Model::default();
        let t = model.add_table("t", FxHashMap::default(), 1);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = IntTablePy(t);
        let index = vec![
            ArgumentUnion::Element(ElementUnion::Const(0)),
            ArgumentUnion::Element(ElementUnion::Const(0)),
            ArgumentUnion::Element(ElementUnion::Const(0)),
            ArgumentUnion::Element(ElementUnion::Const(0)),
        ];
        assert_eq!(
            t_py.__getitem__(index),
            IntExprPy::new(IntegerExpression::Table(Box::new(
                NumericTableExpression::Table(
                    t.id(),
                    vec![
                        ElementExpression::Constant(0),
                        ElementExpression::Constant(0),
                        ElementExpression::Constant(0),
                        ElementExpression::Constant(0),
                    ]
                )
            )))
        );
    }

    #[test]
    fn int_table_sum() {
        let mut model = Model::default();
        let t = model.add_table("t", FxHashMap::default(), 0);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = IntTablePy(t);
        let index = vec![
            ArgumentUnion::Element(ElementUnion::Const(0)),
            ArgumentUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10)))),
            ArgumentUnion::Element(ElementUnion::Const(0)),
            ArgumentUnion::Element(ElementUnion::Const(0)),
        ];
        assert_eq!(
            t_py.__getitem__(index),
            IntExprPy::new(IntegerExpression::Table(Box::new(
                NumericTableExpression::TableReduce(
                    ReduceOperator::Sum,
                    t.id(),
                    vec![
                        ArgumentExpression::Element(ElementExpression::Constant(0)),
                        ArgumentExpression::Set(SetExpression::Reference(
                            ReferenceExpression::Constant(Set::with_capacity(10))
                        )),
                        ArgumentExpression::Element(ElementExpression::Constant(0)),
                        ArgumentExpression::Element(ElementExpression::Constant(0)),
                    ]
                )
            )))
        );
    }

    #[test]
    fn int_table_product_element() {
        let mut model = Model::default();
        let t = model.add_table("t", FxHashMap::default(), 1);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = IntTablePy(t);
        let index = vec![
            ArgumentUnion::Element(ElementUnion::Const(0)),
            ArgumentUnion::Element(ElementUnion::Const(0)),
            ArgumentUnion::Element(ElementUnion::Const(0)),
            ArgumentUnion::Element(ElementUnion::Const(0)),
        ];
        assert_eq!(
            t_py.product(index),
            IntExprPy::new(IntegerExpression::Table(Box::new(
                NumericTableExpression::Table(
                    t.id(),
                    vec![
                        ElementExpression::Constant(0),
                        ElementExpression::Constant(0),
                        ElementExpression::Constant(0),
                        ElementExpression::Constant(0),
                    ]
                )
            )))
        );
    }

    #[test]
    fn int_table_product() {
        let mut model = Model::default();
        let t = model.add_table("t", FxHashMap::default(), 0);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = IntTablePy(t);
        let index = vec![
            ArgumentUnion::Element(ElementUnion::Const(0)),
            ArgumentUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10)))),
            ArgumentUnion::Element(ElementUnion::Const(0)),
            ArgumentUnion::Element(ElementUnion::Const(0)),
        ];
        assert_eq!(
            t_py.product(index),
            IntExprPy::new(IntegerExpression::Table(Box::new(
                NumericTableExpression::TableReduce(
                    ReduceOperator::Product,
                    t.id(),
                    vec![
                        ArgumentExpression::Element(ElementExpression::Constant(0)),
                        ArgumentExpression::Set(SetExpression::Reference(
                            ReferenceExpression::Constant(Set::with_capacity(10))
                        )),
                        ArgumentExpression::Element(ElementExpression::Constant(0)),
                        ArgumentExpression::Element(ElementExpression::Constant(0)),
                    ]
                )
            )))
        );
    }

    #[test]
    fn int_table_max_element() {
        let mut model = Model::default();
        let t = model.add_table("t", FxHashMap::default(), 1);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = IntTablePy(t);
        let index = vec![
            ArgumentUnion::Element(ElementUnion::Const(0)),
            ArgumentUnion::Element(ElementUnion::Const(0)),
            ArgumentUnion::Element(ElementUnion::Const(0)),
            ArgumentUnion::Element(ElementUnion::Const(0)),
        ];
        assert_eq!(
            t_py.max(index),
            IntExprPy::new(IntegerExpression::Table(Box::new(
                NumericTableExpression::Table(
                    t.id(),
                    vec![
                        ElementExpression::Constant(0),
                        ElementExpression::Constant(0),
                        ElementExpression::Constant(0),
                        ElementExpression::Constant(0),
                    ]
                )
            )))
        );
    }

    #[test]
    fn int_table_max() {
        let mut model = Model::default();
        let t = model.add_table("t", FxHashMap::default(), 0);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = IntTablePy(t);
        let index = vec![
            ArgumentUnion::Element(ElementUnion::Const(0)),
            ArgumentUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10)))),
            ArgumentUnion::Element(ElementUnion::Const(0)),
            ArgumentUnion::Element(ElementUnion::Const(0)),
        ];
        assert_eq!(
            t_py.max(index),
            IntExprPy::new(IntegerExpression::Table(Box::new(
                NumericTableExpression::TableReduce(
                    ReduceOperator::Max,
                    t.id(),
                    vec![
                        ArgumentExpression::Element(ElementExpression::Constant(0)),
                        ArgumentExpression::Set(SetExpression::Reference(
                            ReferenceExpression::Constant(Set::with_capacity(10))
                        )),
                        ArgumentExpression::Element(ElementExpression::Constant(0)),
                        ArgumentExpression::Element(ElementExpression::Constant(0)),
                    ]
                )
            )))
        );
    }

    #[test]
    fn int_table_min_element() {
        let mut model = Model::default();
        let t = model.add_table("t", FxHashMap::default(), 1);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = IntTablePy(t);
        let index = vec![
            ArgumentUnion::Element(ElementUnion::Const(0)),
            ArgumentUnion::Element(ElementUnion::Const(0)),
            ArgumentUnion::Element(ElementUnion::Const(0)),
            ArgumentUnion::Element(ElementUnion::Const(0)),
        ];
        assert_eq!(
            t_py.min(index),
            IntExprPy::new(IntegerExpression::Table(Box::new(
                NumericTableExpression::Table(
                    t.id(),
                    vec![
                        ElementExpression::Constant(0),
                        ElementExpression::Constant(0),
                        ElementExpression::Constant(0),
                        ElementExpression::Constant(0),
                    ]
                )
            )))
        );
    }

    #[test]
    fn int_table_min() {
        let mut model = Model::default();
        let t = model.add_table("t", FxHashMap::default(), 0);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = IntTablePy(t);
        let index = vec![
            ArgumentUnion::Element(ElementUnion::Const(0)),
            ArgumentUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10)))),
            ArgumentUnion::Element(ElementUnion::Const(0)),
            ArgumentUnion::Element(ElementUnion::Const(0)),
        ];
        assert_eq!(
            t_py.min(index),
            IntExprPy::new(IntegerExpression::Table(Box::new(
                NumericTableExpression::TableReduce(
                    ReduceOperator::Min,
                    t.id(),
                    vec![
                        ArgumentExpression::Element(ElementExpression::Constant(0)),
                        ArgumentExpression::Set(SetExpression::Reference(
                            ReferenceExpression::Constant(Set::with_capacity(10))
                        )),
                        ArgumentExpression::Element(ElementExpression::Constant(0)),
                        ArgumentExpression::Element(ElementExpression::Constant(0)),
                    ]
                )
            )))
        );
    }

    #[test]
    fn float_table_1d_new() {
        let mut model = Model::default();
        let t = model.add_table_1d("t", vec![1.5]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(FloatTable1DPy::new(t), FloatTable1DPy(t));
    }

    #[test]
    fn float_table_1d_element() {
        let mut model = Model::default();
        let t = model.add_table_1d("t", vec![1.0]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = FloatTable1DPy(t);
        let i = ArgumentUnion::Element(ElementUnion::Const(0));
        assert_eq!(
            t_py.__getitem__(i),
            FloatExprPy::new(ContinuousExpression::Table(Box::new(
                NumericTableExpression::Table1D(t.id(), ElementExpression::Constant(0))
            )))
        );
    }

    #[test]
    fn float_table_1d_sum() {
        let mut model = Model::default();
        let t = model.add_table_1d("t", vec![1.0]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = FloatTable1DPy(t);
        let i = ArgumentUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10))));
        assert_eq!(
            t_py.__getitem__(i),
            FloatExprPy::new(ContinuousExpression::Table(Box::new(
                NumericTableExpression::Table1DReduce(
                    ReduceOperator::Sum,
                    t.id(),
                    SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(10)))
                )
            )))
        );
    }

    #[test]
    fn float_table_1d_product() {
        let mut model = Model::default();
        let t = model.add_table_1d("t", vec![1.0]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = FloatTable1DPy(t);
        let i = SetUnion::Const(SetConstPy::new(Set::with_capacity(10)));
        assert_eq!(
            t_py.product(i),
            FloatExprPy::new(ContinuousExpression::Table(Box::new(
                NumericTableExpression::Table1DReduce(
                    ReduceOperator::Product,
                    t.id(),
                    SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(10)))
                )
            )))
        );
    }

    #[test]
    fn float_table_1d_max() {
        let mut model = Model::default();
        let t = model.add_table_1d("t", vec![1.0]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = FloatTable1DPy(t);
        let i = SetUnion::Const(SetConstPy::new(Set::with_capacity(10)));
        assert_eq!(
            t_py.max(i),
            FloatExprPy::new(ContinuousExpression::Table(Box::new(
                NumericTableExpression::Table1DReduce(
                    ReduceOperator::Max,
                    t.id(),
                    SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(10)))
                )
            )))
        );
    }

    #[test]
    fn float_table_1d_min() {
        let mut model = Model::default();
        let t = model.add_table_1d("t", vec![1.0]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = FloatTable1DPy(t);
        let i = SetUnion::Const(SetConstPy::new(Set::with_capacity(10)));
        assert_eq!(
            t_py.min(i),
            FloatExprPy::new(ContinuousExpression::Table(Box::new(
                NumericTableExpression::Table1DReduce(
                    ReduceOperator::Min,
                    t.id(),
                    SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(10)))
                )
            )))
        );
    }

    #[test]
    fn float_table_2d_new() {
        let mut model = Model::default();
        let t = model.add_table_2d("t", vec![vec![1.5]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(FloatTable2DPy::new(t), FloatTable2DPy(t));
    }

    #[test]
    fn float_table_2d_element() {
        let mut model = Model::default();
        let t = model.add_table_2d("t", vec![vec![1.0]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = FloatTable2DPy(t);
        let x = ArgumentUnion::Element(ElementUnion::Const(0));
        let y = ArgumentUnion::Element(ElementUnion::Const(0));
        assert_eq!(
            t_py.__getitem__((x, y)),
            FloatExprPy::new(ContinuousExpression::Table(Box::new(
                NumericTableExpression::Table2D(
                    t.id(),
                    ElementExpression::Constant(0),
                    ElementExpression::Constant(0)
                )
            )))
        );
    }

    #[test]
    fn float_table_2d_sum_x() {
        let mut model = Model::default();
        let t = model.add_table_2d("t", vec![vec![1.0]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = FloatTable2DPy(t);
        let x = ArgumentUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10))));
        let y = ArgumentUnion::Element(ElementUnion::Const(0));
        assert_eq!(
            t_py.__getitem__((x, y)),
            FloatExprPy::new(ContinuousExpression::Table(Box::new(
                NumericTableExpression::Table2DReduceX(
                    ReduceOperator::Sum,
                    t.id(),
                    SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(10))),
                    ElementExpression::Constant(0)
                )
            )))
        );
    }

    #[test]
    fn float_table_2d_sum_y() {
        let mut model = Model::default();
        let t = model.add_table_2d("t", vec![vec![1.0]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = FloatTable2DPy(t);
        let x = ArgumentUnion::Element(ElementUnion::Const(0));
        let y = ArgumentUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10))));
        assert_eq!(
            t_py.__getitem__((x, y)),
            FloatExprPy::new(ContinuousExpression::Table(Box::new(
                NumericTableExpression::Table2DReduceY(
                    ReduceOperator::Sum,
                    t.id(),
                    ElementExpression::Constant(0),
                    SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(10))),
                )
            )))
        );
    }

    #[test]
    fn float_table_2d_sum() {
        let mut model = Model::default();
        let t = model.add_table_2d("t", vec![vec![1.0]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = FloatTable2DPy(t);
        let x = ArgumentUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10))));
        let y = ArgumentUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10))));
        assert_eq!(
            t_py.__getitem__((x, y)),
            FloatExprPy::new(ContinuousExpression::Table(Box::new(
                NumericTableExpression::Table2DReduce(
                    ReduceOperator::Sum,
                    t.id(),
                    SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(10))),
                    SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(10))),
                )
            )))
        );
    }

    #[test]
    fn float_table_2d_product_element() {
        let mut model = Model::default();
        let t = model.add_table_2d("t", vec![vec![1.0]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = FloatTable2DPy(t);
        let x = ArgumentUnion::Element(ElementUnion::Const(0));
        let y = ArgumentUnion::Element(ElementUnion::Const(0));
        assert_eq!(
            t_py.product(x, y),
            FloatExprPy::new(ContinuousExpression::Table(Box::new(
                NumericTableExpression::Table2D(
                    t.id(),
                    ElementExpression::Constant(0),
                    ElementExpression::Constant(0)
                )
            )))
        );
    }

    #[test]
    fn float_table_2d_product_x() {
        let mut model = Model::default();
        let t = model.add_table_2d("t", vec![vec![1.0]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = FloatTable2DPy(t);
        let x = ArgumentUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10))));
        let y = ArgumentUnion::Element(ElementUnion::Const(0));
        assert_eq!(
            t_py.product(x, y),
            FloatExprPy::new(ContinuousExpression::Table(Box::new(
                NumericTableExpression::Table2DReduceX(
                    ReduceOperator::Product,
                    t.id(),
                    SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(10))),
                    ElementExpression::Constant(0)
                )
            )))
        );
    }

    #[test]
    fn float_table_2d_product_y() {
        let mut model = Model::default();
        let t = model.add_table_2d("t", vec![vec![1.0]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = FloatTable2DPy(t);
        let x = ArgumentUnion::Element(ElementUnion::Const(0));
        let y = ArgumentUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10))));
        assert_eq!(
            t_py.product(x, y),
            FloatExprPy::new(ContinuousExpression::Table(Box::new(
                NumericTableExpression::Table2DReduceY(
                    ReduceOperator::Product,
                    t.id(),
                    ElementExpression::Constant(0),
                    SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(10))),
                )
            )))
        );
    }

    #[test]
    fn float_table_2d_product() {
        let mut model = Model::default();
        let t = model.add_table_2d("t", vec![vec![1.0]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = FloatTable2DPy(t);
        let x = ArgumentUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10))));
        let y = ArgumentUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10))));
        assert_eq!(
            t_py.product(x, y),
            FloatExprPy::new(ContinuousExpression::Table(Box::new(
                NumericTableExpression::Table2DReduce(
                    ReduceOperator::Product,
                    t.id(),
                    SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(10))),
                    SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(10))),
                )
            )))
        );
    }

    #[test]
    fn float_table_2d_max_element() {
        let mut model = Model::default();
        let t = model.add_table_2d("t", vec![vec![1.0]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = FloatTable2DPy(t);
        let x = ArgumentUnion::Element(ElementUnion::Const(0));
        let y = ArgumentUnion::Element(ElementUnion::Const(0));
        assert_eq!(
            t_py.max(x, y),
            FloatExprPy::new(ContinuousExpression::Table(Box::new(
                NumericTableExpression::Table2D(
                    t.id(),
                    ElementExpression::Constant(0),
                    ElementExpression::Constant(0)
                )
            )))
        );
    }

    #[test]
    fn float_table_2d_max_x() {
        let mut model = Model::default();
        let t = model.add_table_2d("t", vec![vec![1.0]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = FloatTable2DPy(t);
        let x = ArgumentUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10))));
        let y = ArgumentUnion::Element(ElementUnion::Const(0));
        assert_eq!(
            t_py.max(x, y),
            FloatExprPy::new(ContinuousExpression::Table(Box::new(
                NumericTableExpression::Table2DReduceX(
                    ReduceOperator::Max,
                    t.id(),
                    SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(10))),
                    ElementExpression::Constant(0)
                )
            )))
        );
    }

    #[test]
    fn float_table_2d_max_y() {
        let mut model = Model::default();
        let t = model.add_table_2d("t", vec![vec![1.0]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = FloatTable2DPy(t);
        let x = ArgumentUnion::Element(ElementUnion::Const(0));
        let y = ArgumentUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10))));
        assert_eq!(
            t_py.max(x, y),
            FloatExprPy::new(ContinuousExpression::Table(Box::new(
                NumericTableExpression::Table2DReduceY(
                    ReduceOperator::Max,
                    t.id(),
                    ElementExpression::Constant(0),
                    SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(10))),
                )
            )))
        );
    }

    #[test]
    fn float_table_2d_max() {
        let mut model = Model::default();
        let t = model.add_table_2d("t", vec![vec![1.0]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = FloatTable2DPy(t);
        let x = ArgumentUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10))));
        let y = ArgumentUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10))));
        assert_eq!(
            t_py.max(x, y),
            FloatExprPy::new(ContinuousExpression::Table(Box::new(
                NumericTableExpression::Table2DReduce(
                    ReduceOperator::Max,
                    t.id(),
                    SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(10))),
                    SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(10))),
                )
            )))
        );
    }

    #[test]
    fn float_table_2d_min_element() {
        let mut model = Model::default();
        let t = model.add_table_2d("t", vec![vec![1.0]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = FloatTable2DPy(t);
        let x = ArgumentUnion::Element(ElementUnion::Const(0));
        let y = ArgumentUnion::Element(ElementUnion::Const(0));
        assert_eq!(
            t_py.min(x, y),
            FloatExprPy::new(ContinuousExpression::Table(Box::new(
                NumericTableExpression::Table2D(
                    t.id(),
                    ElementExpression::Constant(0),
                    ElementExpression::Constant(0)
                )
            )))
        );
    }

    #[test]
    fn float_table_2d_min_x() {
        let mut model = Model::default();
        let t = model.add_table_2d("t", vec![vec![1.0]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = FloatTable2DPy(t);
        let x = ArgumentUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10))));
        let y = ArgumentUnion::Element(ElementUnion::Const(0));
        assert_eq!(
            t_py.min(x, y),
            FloatExprPy::new(ContinuousExpression::Table(Box::new(
                NumericTableExpression::Table2DReduceX(
                    ReduceOperator::Min,
                    t.id(),
                    SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(10))),
                    ElementExpression::Constant(0)
                )
            )))
        );
    }

    #[test]
    fn float_table_2d_min_y() {
        let mut model = Model::default();
        let t = model.add_table_2d("t", vec![vec![1.0]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = FloatTable2DPy(t);
        let x = ArgumentUnion::Element(ElementUnion::Const(0));
        let y = ArgumentUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10))));
        assert_eq!(
            t_py.min(x, y),
            FloatExprPy::new(ContinuousExpression::Table(Box::new(
                NumericTableExpression::Table2DReduceY(
                    ReduceOperator::Min,
                    t.id(),
                    ElementExpression::Constant(0),
                    SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(10))),
                )
            )))
        );
    }

    #[test]
    fn float_table_2d_min() {
        let mut model = Model::default();
        let t = model.add_table_2d("t", vec![vec![1.0]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = FloatTable2DPy(t);
        let x = ArgumentUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10))));
        let y = ArgumentUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10))));
        assert_eq!(
            t_py.min(x, y),
            FloatExprPy::new(ContinuousExpression::Table(Box::new(
                NumericTableExpression::Table2DReduce(
                    ReduceOperator::Min,
                    t.id(),
                    SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(10))),
                    SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(10))),
                )
            )))
        );
    }

    #[test]
    fn float_table_3d_new() {
        let mut model = Model::default();
        let t = model.add_table_3d("t", vec![vec![vec![1.5]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(FloatTable3DPy::new(t), FloatTable3DPy(t));
    }

    #[test]
    fn float_table_3d_element() {
        let mut model = Model::default();
        let t = model.add_table_3d("t", vec![vec![vec![1.0]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = FloatTable3DPy(t);
        let x = ArgumentUnion::Element(ElementUnion::Const(0));
        let y = ArgumentUnion::Element(ElementUnion::Const(0));
        let z = ArgumentUnion::Element(ElementUnion::Const(0));
        assert_eq!(
            t_py.__getitem__((x, y, z)),
            FloatExprPy::new(ContinuousExpression::Table(Box::new(
                NumericTableExpression::Table3D(
                    t.id(),
                    ElementExpression::Constant(0),
                    ElementExpression::Constant(0),
                    ElementExpression::Constant(0),
                )
            )))
        );
    }

    #[test]
    fn float_table_3d_sum() {
        let mut model = Model::default();
        let t = model.add_table_3d("t", vec![vec![vec![1.0]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = FloatTable3DPy(t);
        let x = ArgumentUnion::Element(ElementUnion::Const(0));
        let y = ArgumentUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10))));
        let z = ArgumentUnion::Element(ElementUnion::Const(0));
        assert_eq!(
            t_py.__getitem__((x, y, z)),
            FloatExprPy::new(ContinuousExpression::Table(Box::new(
                NumericTableExpression::Table3DReduce(
                    ReduceOperator::Sum,
                    t.id(),
                    ArgumentExpression::Element(ElementExpression::Constant(0)),
                    ArgumentExpression::Set(SetExpression::Reference(
                        ReferenceExpression::Constant(Set::with_capacity(10))
                    )),
                    ArgumentExpression::Element(ElementExpression::Constant(0)),
                )
            )))
        );
    }

    #[test]
    fn float_table_3d_product_element() {
        let mut model = Model::default();
        let t = model.add_table_3d("t", vec![vec![vec![1.0]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = FloatTable3DPy(t);
        let x = ArgumentUnion::Element(ElementUnion::Const(0));
        let y = ArgumentUnion::Element(ElementUnion::Const(0));
        let z = ArgumentUnion::Element(ElementUnion::Const(0));
        assert_eq!(
            t_py.product(x, y, z),
            FloatExprPy::new(ContinuousExpression::Table(Box::new(
                NumericTableExpression::Table3D(
                    t.id(),
                    ElementExpression::Constant(0),
                    ElementExpression::Constant(0),
                    ElementExpression::Constant(0),
                )
            )))
        );
    }

    #[test]
    fn float_table_3d_product() {
        let mut model = Model::default();
        let t = model.add_table_3d("t", vec![vec![vec![1.0]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = FloatTable3DPy(t);
        let x = ArgumentUnion::Element(ElementUnion::Const(0));
        let y = ArgumentUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10))));
        let z = ArgumentUnion::Element(ElementUnion::Const(0));
        assert_eq!(
            t_py.product(x, y, z),
            FloatExprPy::new(ContinuousExpression::Table(Box::new(
                NumericTableExpression::Table3DReduce(
                    ReduceOperator::Product,
                    t.id(),
                    ArgumentExpression::Element(ElementExpression::Constant(0)),
                    ArgumentExpression::Set(SetExpression::Reference(
                        ReferenceExpression::Constant(Set::with_capacity(10))
                    )),
                    ArgumentExpression::Element(ElementExpression::Constant(0)),
                )
            )))
        );
    }

    #[test]
    fn float_table_3d_max_element() {
        let mut model = Model::default();
        let t = model.add_table_3d("t", vec![vec![vec![1.0]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = FloatTable3DPy(t);
        let x = ArgumentUnion::Element(ElementUnion::Const(0));
        let y = ArgumentUnion::Element(ElementUnion::Const(0));
        let z = ArgumentUnion::Element(ElementUnion::Const(0));
        assert_eq!(
            t_py.max(x, y, z),
            FloatExprPy::new(ContinuousExpression::Table(Box::new(
                NumericTableExpression::Table3D(
                    t.id(),
                    ElementExpression::Constant(0),
                    ElementExpression::Constant(0),
                    ElementExpression::Constant(0),
                )
            )))
        );
    }

    #[test]
    fn float_table_3d_max() {
        let mut model = Model::default();
        let t = model.add_table_3d("t", vec![vec![vec![1.0]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = FloatTable3DPy(t);
        let x = ArgumentUnion::Element(ElementUnion::Const(0));
        let y = ArgumentUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10))));
        let z = ArgumentUnion::Element(ElementUnion::Const(0));
        assert_eq!(
            t_py.max(x, y, z),
            FloatExprPy::new(ContinuousExpression::Table(Box::new(
                NumericTableExpression::Table3DReduce(
                    ReduceOperator::Max,
                    t.id(),
                    ArgumentExpression::Element(ElementExpression::Constant(0)),
                    ArgumentExpression::Set(SetExpression::Reference(
                        ReferenceExpression::Constant(Set::with_capacity(10))
                    )),
                    ArgumentExpression::Element(ElementExpression::Constant(0)),
                )
            )))
        );
    }

    #[test]
    fn float_table_3d_min_element() {
        let mut model = Model::default();
        let t = model.add_table_3d("t", vec![vec![vec![1.0]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = FloatTable3DPy(t);
        let x = ArgumentUnion::Element(ElementUnion::Const(0));
        let y = ArgumentUnion::Element(ElementUnion::Const(0));
        let z = ArgumentUnion::Element(ElementUnion::Const(0));
        assert_eq!(
            t_py.min(x, y, z),
            FloatExprPy::new(ContinuousExpression::Table(Box::new(
                NumericTableExpression::Table3D(
                    t.id(),
                    ElementExpression::Constant(0),
                    ElementExpression::Constant(0),
                    ElementExpression::Constant(0),
                )
            )))
        );
    }

    #[test]
    fn float_table_3d_min() {
        let mut model = Model::default();
        let t = model.add_table_3d("t", vec![vec![vec![1.0]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = FloatTable3DPy(t);
        let x = ArgumentUnion::Element(ElementUnion::Const(0));
        let y = ArgumentUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10))));
        let z = ArgumentUnion::Element(ElementUnion::Const(0));
        assert_eq!(
            t_py.min(x, y, z),
            FloatExprPy::new(ContinuousExpression::Table(Box::new(
                NumericTableExpression::Table3DReduce(
                    ReduceOperator::Min,
                    t.id(),
                    ArgumentExpression::Element(ElementExpression::Constant(0)),
                    ArgumentExpression::Set(SetExpression::Reference(
                        ReferenceExpression::Constant(Set::with_capacity(10))
                    )),
                    ArgumentExpression::Element(ElementExpression::Constant(0)),
                )
            )))
        );
    }

    #[test]
    fn float_table_new() {
        let mut model = Model::default();
        let t = model.add_table("t", FxHashMap::default(), 0.0);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(FloatTablePy::new(t), FloatTablePy(t));
    }

    #[test]
    fn float_table_element() {
        let mut model = Model::default();
        let t = model.add_table("t", FxHashMap::default(), 1.0);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = FloatTablePy(t);
        let index = vec![
            ArgumentUnion::Element(ElementUnion::Const(0)),
            ArgumentUnion::Element(ElementUnion::Const(0)),
            ArgumentUnion::Element(ElementUnion::Const(0)),
            ArgumentUnion::Element(ElementUnion::Const(0)),
        ];
        assert_eq!(
            t_py.__getitem__(index),
            FloatExprPy::new(ContinuousExpression::Table(Box::new(
                NumericTableExpression::Table(
                    t.id(),
                    vec![
                        ElementExpression::Constant(0),
                        ElementExpression::Constant(0),
                        ElementExpression::Constant(0),
                        ElementExpression::Constant(0),
                    ]
                )
            )))
        );
    }

    #[test]
    fn float_table_sum() {
        let mut model = Model::default();
        let t = model.add_table("t", FxHashMap::default(), 0.0);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = FloatTablePy(t);
        let index = vec![
            ArgumentUnion::Element(ElementUnion::Const(0)),
            ArgumentUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10)))),
            ArgumentUnion::Element(ElementUnion::Const(0)),
            ArgumentUnion::Element(ElementUnion::Const(0)),
        ];
        assert_eq!(
            t_py.__getitem__(index),
            FloatExprPy::new(ContinuousExpression::Table(Box::new(
                NumericTableExpression::TableReduce(
                    ReduceOperator::Sum,
                    t.id(),
                    vec![
                        ArgumentExpression::Element(ElementExpression::Constant(0)),
                        ArgumentExpression::Set(SetExpression::Reference(
                            ReferenceExpression::Constant(Set::with_capacity(10))
                        )),
                        ArgumentExpression::Element(ElementExpression::Constant(0)),
                        ArgumentExpression::Element(ElementExpression::Constant(0)),
                    ]
                )
            )))
        );
    }

    #[test]
    fn float_table_product_element() {
        let mut model = Model::default();
        let t = model.add_table("t", FxHashMap::default(), 1.0);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = FloatTablePy(t);
        let index = vec![
            ArgumentUnion::Element(ElementUnion::Const(0)),
            ArgumentUnion::Element(ElementUnion::Const(0)),
            ArgumentUnion::Element(ElementUnion::Const(0)),
            ArgumentUnion::Element(ElementUnion::Const(0)),
        ];
        assert_eq!(
            t_py.product(index),
            FloatExprPy::new(ContinuousExpression::Table(Box::new(
                NumericTableExpression::Table(
                    t.id(),
                    vec![
                        ElementExpression::Constant(0),
                        ElementExpression::Constant(0),
                        ElementExpression::Constant(0),
                        ElementExpression::Constant(0),
                    ]
                )
            )))
        );
    }

    #[test]
    fn float_table_product() {
        let mut model = Model::default();
        let t = model.add_table("t", FxHashMap::default(), 0.0);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = FloatTablePy(t);
        let index = vec![
            ArgumentUnion::Element(ElementUnion::Const(0)),
            ArgumentUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10)))),
            ArgumentUnion::Element(ElementUnion::Const(0)),
            ArgumentUnion::Element(ElementUnion::Const(0)),
        ];
        assert_eq!(
            t_py.product(index),
            FloatExprPy::new(ContinuousExpression::Table(Box::new(
                NumericTableExpression::TableReduce(
                    ReduceOperator::Product,
                    t.id(),
                    vec![
                        ArgumentExpression::Element(ElementExpression::Constant(0)),
                        ArgumentExpression::Set(SetExpression::Reference(
                            ReferenceExpression::Constant(Set::with_capacity(10))
                        )),
                        ArgumentExpression::Element(ElementExpression::Constant(0)),
                        ArgumentExpression::Element(ElementExpression::Constant(0)),
                    ]
                )
            )))
        );
    }

    #[test]
    fn float_table_max_element() {
        let mut model = Model::default();
        let t = model.add_table("t", FxHashMap::default(), 1.0);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = FloatTablePy(t);
        let index = vec![
            ArgumentUnion::Element(ElementUnion::Const(0)),
            ArgumentUnion::Element(ElementUnion::Const(0)),
            ArgumentUnion::Element(ElementUnion::Const(0)),
            ArgumentUnion::Element(ElementUnion::Const(0)),
        ];
        assert_eq!(
            t_py.max(index),
            FloatExprPy::new(ContinuousExpression::Table(Box::new(
                NumericTableExpression::Table(
                    t.id(),
                    vec![
                        ElementExpression::Constant(0),
                        ElementExpression::Constant(0),
                        ElementExpression::Constant(0),
                        ElementExpression::Constant(0),
                    ]
                )
            )))
        );
    }

    #[test]
    fn float_table_max() {
        let mut model = Model::default();
        let t = model.add_table("t", FxHashMap::default(), 0.0);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = FloatTablePy(t);
        let index = vec![
            ArgumentUnion::Element(ElementUnion::Const(0)),
            ArgumentUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10)))),
            ArgumentUnion::Element(ElementUnion::Const(0)),
            ArgumentUnion::Element(ElementUnion::Const(0)),
        ];
        assert_eq!(
            t_py.max(index),
            FloatExprPy::new(ContinuousExpression::Table(Box::new(
                NumericTableExpression::TableReduce(
                    ReduceOperator::Max,
                    t.id(),
                    vec![
                        ArgumentExpression::Element(ElementExpression::Constant(0)),
                        ArgumentExpression::Set(SetExpression::Reference(
                            ReferenceExpression::Constant(Set::with_capacity(10))
                        )),
                        ArgumentExpression::Element(ElementExpression::Constant(0)),
                        ArgumentExpression::Element(ElementExpression::Constant(0)),
                    ]
                )
            )))
        );
    }

    #[test]
    fn float_table_min_element() {
        let mut model = Model::default();
        let t = model.add_table("t", FxHashMap::default(), 1.0);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = FloatTablePy(t);
        let index = vec![
            ArgumentUnion::Element(ElementUnion::Const(0)),
            ArgumentUnion::Element(ElementUnion::Const(0)),
            ArgumentUnion::Element(ElementUnion::Const(0)),
            ArgumentUnion::Element(ElementUnion::Const(0)),
        ];
        assert_eq!(
            t_py.min(index),
            FloatExprPy::new(ContinuousExpression::Table(Box::new(
                NumericTableExpression::Table(
                    t.id(),
                    vec![
                        ElementExpression::Constant(0),
                        ElementExpression::Constant(0),
                        ElementExpression::Constant(0),
                        ElementExpression::Constant(0),
                    ]
                )
            )))
        );
    }

    #[test]
    fn float_table_min() {
        let mut model = Model::default();
        let t = model.add_table("t", FxHashMap::default(), 0.0);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = FloatTablePy(t);
        let index = vec![
            ArgumentUnion::Element(ElementUnion::Const(0)),
            ArgumentUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10)))),
            ArgumentUnion::Element(ElementUnion::Const(0)),
            ArgumentUnion::Element(ElementUnion::Const(0)),
        ];
        assert_eq!(
            t_py.min(index),
            FloatExprPy::new(ContinuousExpression::Table(Box::new(
                NumericTableExpression::TableReduce(
                    ReduceOperator::Min,
                    t.id(),
                    vec![
                        ArgumentExpression::Element(ElementExpression::Constant(0)),
                        ArgumentExpression::Set(SetExpression::Reference(
                            ReferenceExpression::Constant(Set::with_capacity(10))
                        )),
                        ArgumentExpression::Element(ElementExpression::Constant(0)),
                        ArgumentExpression::Element(ElementExpression::Constant(0)),
                    ]
                )
            )))
        );
    }
}
