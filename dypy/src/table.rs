use super::expression::*;
use dypdl::expression::ArgumentExpression;
use dypdl::prelude::*;
use pyo3::prelude::*;

/// A class representing a 1-dimensional table of element constants.
///
/// `t[x]` returns an element expression referring to an item where `t` is `ElementTable1D` and `x` is `ElementExpr`, `ElementVar`, `ElementResourceVar`, or `int`.
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

/// A class representing a 2-dimensional table of element constants.
///
/// `t[x, y]` returns an element expression referring to an item where `t` is `ElementTable2D` and `x` and `y` are `ElementExpr`, `ElementVar`, `ElementResourceVar`, or `int`.
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

/// A class representing a 3-dimensional table of element constants.
///
/// `t[x, y, z]` returns an element expression referring to an item where `t` is `ElementTable3D` and `x`, `y`, and `z` are `ElementExpr`, `ElementVar`, `ElementResourceVar`, or `int`.
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

/// A class representing a table of element constants.
///
/// `t[index]` returns an element expression referring to an item where `t` is `ElementTable` and `index` is a sequence of `ElementExpr`, `ElementVar`, `ElementResourceVar`, or `int`.
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

/// A class representing a 1-dimensional table of set constants.
///
/// `t[x]` returns a set expression referring to an item where `t` is `SetTable1D` and `x` is `ElementExpr`, `ElementVar`, `ElementResourceVar`, or `int`.
#[pyclass(name = "SetTable1D")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetTable1DPy(Table1DHandle<Set>);

impl From<SetTable1DPy> for Table1DHandle<Set> {
    fn from(table: SetTable1DPy) -> Self {
        table.0
    }
}

impl SetTable1DPy {
    pub fn new(table: Table1DHandle<Set>) -> SetTable1DPy {
        SetTable1DPy(table)
    }
}

#[pymethods]
impl SetTable1DPy {
    fn __getitem__(&self, i: ElementUnion) -> SetExprPy {
        SetExprPy::new(self.0.element(i))
    }
}

/// A class representing a 2-dimensional table of set constants.
///
/// `t[x, y]` returns a set expression referring to an item where `t` is `SetTable2D` and `x` and `y` are `ElementExpr`, `ElementVar`, `ElementResourceVar`, or `int`.
#[pyclass(name = "SetTable2D")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetTable2DPy(Table2DHandle<Set>);

impl SetTable2DPy {
    pub fn new(table: Table2DHandle<Set>) -> SetTable2DPy {
        SetTable2DPy(table)
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
}

/// A class representing a 3-dimensional table of set constants.
///
/// `t[x, y, z]` returns a set expression referring to an item where `t` is `SetTable3D` and `x`, `y`, and `z` are `ElementExpr`, `ElementVar`, `ElementResourceVar`, or `int`.
#[pyclass(name = "SetTable3D")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetTable3DPy(Table3DHandle<Set>);

impl From<SetTable3DPy> for Table3DHandle<Set> {
    fn from(table: SetTable3DPy) -> Self {
        table.0
    }
}

impl SetTable3DPy {
    pub fn new(table: Table3DHandle<Set>) -> SetTable3DPy {
        SetTable3DPy(table)
    }
}

#[pymethods]
impl SetTable3DPy {
    fn __getitem__(&self, index: (ElementUnion, ElementUnion, ElementUnion)) -> SetExprPy {
        let (x, y, z) = index;
        SetExprPy::new(self.0.element(x, y, z))
    }
}

/// A class representing a table of set constants.
///
/// `t[index]` returns a set expression referring to an item where `t` is `SetTable` and `index` is a sequence of `ElementExpr`, `ElementVar`, `ElementResourceVar`, or `int`.
#[pyclass(name = "SetTable")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetTablePy(TableHandle<Set>);

impl From<SetTablePy> for TableHandle<Set> {
    fn from(table: SetTablePy) -> Self {
        table.0
    }
}

impl SetTablePy {
    pub fn new(table: TableHandle<Set>) -> SetTablePy {
        SetTablePy(table)
    }
}

#[pymethods]
impl SetTablePy {
    fn __getitem__(&self, index: Vec<ElementUnion>) -> SetExprPy {
        let index = index.into_iter().map(ElementExpression::from).collect();
        SetExprPy::new(self.0.element(index))
    }
}

/// A class representing a 1-dimensional table of bool constants.
///
/// `t[x]` returns a condition referring to an item where `t` is `BoolTable1D` and `x` is `ElementExpr`, `ElementVar`, `ElementResourceVar`, or `int`.
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

/// A class representing a 2-dimensional table of bool constants.
///
/// `t[x, y]` returns a condition referring to an item where `t` is `BoolTable2D` and `x` and `y` are `ElementExpr`, `ElementVar`, `ElementResourceVar`, or `int`.
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

/// A class representing a 3-dimensional table of bool constants.
///
/// `t[x, y, z]` returns a condition referring to an item where `t` is `BoolTable3D` and `x`, `y`, and `z` are `ElementExpr`, `ElementVar`, `ElementResourceVar`, or `int`.
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

/// A class representing a table of bool constants.
///
/// `t[index]` returns a condition referring to an item where `t` is `BoolTable` and `index` is a sequence of `ElementExpr`, `ElementVar`, `ElementResourceVar`, or `int`.
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
pub enum TableIndexUnion {
    Element(ElementUnion),
    Set(SetUnion),
}

impl From<TableIndexUnion> for ArgumentExpression {
    fn from(index: TableIndexUnion) -> Self {
        match index {
            TableIndexUnion::Element(index) => {
                ArgumentExpression::from(ElementExpression::from(index))
            }
            TableIndexUnion::Set(index) => ArgumentExpression::from(SetExpression::from(index)),
        }
    }
}

/// A class representing a 1-dimensional table of integer constants.
///
/// `t[x]` returns an integer expression referring to an item where `t` is `IntTable1D` and `x` is `ElementExpr`, `ElementVar`, `ElementResourceVar`, or `int`.
/// If `x` is `SetExpr`, `SetVar`, or `SetConst`, `t[x]` returns the sum of constants over `x`.
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
    fn __getitem__(&self, i: TableIndexUnion) -> IntExprPy {
        IntExprPy::new(match i {
            TableIndexUnion::Element(i) => self.0.element(i),
            TableIndexUnion::Set(i) => self.0.sum(i),
        })
    }
}

/// A class representing a 2-dimensional table of integer constants.
///
/// `t[x, y]` returns an integer expression referring to an item where `t` is `IntTable2D` and `x` and `y` are `ElementExpr`, `ElementVar`, `ElementResourceVar`, or `int`.
/// If `x` and/or `y` are/is `SetExpr`, `SetVar`, or `SetConst`, `t[x, y]` returns the sum of constants over `x` and `y`.
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
    fn __getitem__(&self, index: (TableIndexUnion, TableIndexUnion)) -> IntExprPy {
        let (x, y) = index;
        IntExprPy::new(match (x, y) {
            (TableIndexUnion::Element(x), TableIndexUnion::Element(y)) => self.0.element(x, y),
            (TableIndexUnion::Set(x), TableIndexUnion::Element(y)) => self.0.sum_x(x, y),
            (TableIndexUnion::Element(x), TableIndexUnion::Set(y)) => self.0.sum_y(x, y),
            (TableIndexUnion::Set(x), TableIndexUnion::Set(y)) => self.0.sum(x, y),
        })
    }
}

/// A class representing a 3-dimensional table of integer constants.
///
/// `t[x, y, z]` returns an integer expression referring to an item where `t` is `IntTable3D` and `x`, `y`, and `z` are `ElementExpr`, `ElementVar`, `ElementResourceVar`, or `int`.
/// If `x`, `y`, and/or `z` are/is `SetExpr`, `SetVar`, or `SetConst`, `t[x, y, z]` returns the sum of constants over `x`, `y`, and `z`.
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
    fn __getitem__(&self, index: (TableIndexUnion, TableIndexUnion, TableIndexUnion)) -> IntExprPy {
        let (x, y, z) = index;
        IntExprPy::new(match (x, y, z) {
            (
                TableIndexUnion::Element(x),
                TableIndexUnion::Element(y),
                TableIndexUnion::Element(z),
            ) => self.0.element(x, y, z),
            (x, y, z) => self.0.sum(x, y, z),
        })
    }
}

/// A class representing a table of integer constants.
///
/// `t[index]` returns an integer expression referring to an item where `t` is `IntTable` and `index` is a sequence of `ElementExpr`, `ElementVar`, `ElementResourceVar`, or `int`.
/// If one of `index` is `SetExpr`, `SetVar`, or `SetConst`, `t[index]` returns the sum of constants.
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
    fn __getitem__(&self, index: Vec<TableIndexUnion>) -> IntExprPy {
        let mut elements = Vec::with_capacity(index.len());
        for i in &index {
            match i {
                TableIndexUnion::Element(i) => elements.push(ElementExpression::from(i.clone())),
                _ => return IntExprPy::new(self.0.sum(index)),
            }
        }
        IntExprPy::new(self.0.element(elements))
    }
}

/// A class representing a 1-dimensional table of continuous constants.
///
/// `t[x]` returns an continuous expression referring to an item where `t` is `FloatTable1D` and `x` is `ElementExpr`, `ElementVar`, `ElementResourceVar`, or `int`.
/// If `x` is `SetExpr`, `SetVar`, or `SetConst`, `t[x]` returns the sum of constants over `x`.
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
    fn __getitem__(&self, i: TableIndexUnion) -> FloatExprPy {
        FloatExprPy::new(match i {
            TableIndexUnion::Element(i) => self.0.element(i),
            TableIndexUnion::Set(i) => self.0.sum(i),
        })
    }
}

/// A class representing a 2-dimensional table of continuous constants.
///
/// `t[x, y]` returns a continuous expression referring to an item where `t` is `FloatTable2D` and `x` and `y` are `ElementExpr`, `ElementVar`, `ElementResourceVar`, or `int`.
/// If `x` and/or `y` are/is `SetExpr`, `SetVar`, or `SetConst`, `t[x, y]` returns the sum of constants over `x` and `y`.
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
    fn __getitem__(&self, index: (TableIndexUnion, TableIndexUnion)) -> FloatExprPy {
        let (x, y) = index;
        FloatExprPy::new(match (x, y) {
            (TableIndexUnion::Element(x), TableIndexUnion::Element(y)) => self.0.element(x, y),
            (TableIndexUnion::Set(x), TableIndexUnion::Element(y)) => self.0.sum_x(x, y),
            (TableIndexUnion::Element(x), TableIndexUnion::Set(y)) => self.0.sum_y(x, y),
            (TableIndexUnion::Set(x), TableIndexUnion::Set(y)) => self.0.sum(x, y),
        })
    }
}

/// A class representing a table of continuous constants.
///
/// `t[x, y, z]` returns a continuous expression referring to an item where `t` is `FloatTable3D` and `x`, `y`, and `z` are `ElementExpr`, `ElementVar`, `ElementResourceVar`, or `int`.
/// If `x`, `y`, and/or `z` are/is `SetExpr`, `SetVar`, or `SetConst`, `t[x, y, z]` returns the sum of constants over `x`, `y`, and `z`.
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
    fn __getitem__(
        &self,
        index: (TableIndexUnion, TableIndexUnion, TableIndexUnion),
    ) -> FloatExprPy {
        let (x, y, z) = index;
        FloatExprPy::new(match (x, y, z) {
            (
                TableIndexUnion::Element(x),
                TableIndexUnion::Element(y),
                TableIndexUnion::Element(z),
            ) => self.0.element(x, y, z),
            (x, y, z) => self.0.sum(x, y, z),
        })
    }
}

/// A class representing a table of continuous constants.
///
/// `t[index]` returns a continuous expression referring to an item where `t` is `FloatTable` and `index` is a sequence of `ElementExpr`, `ElementVar`, `ElementResourceVar`, or `int`.
/// If one of `index` is `SetExpr`, `SetVar`, or `SetConst`, `t[index]` returns the sum of constants.
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
    fn __getitem__(&self, index: Vec<TableIndexUnion>) -> FloatExprPy {
        let mut elements = Vec::with_capacity(index.len());
        for i in &index {
            match i {
                TableIndexUnion::Element(i) => elements.push(ElementExpression::from(i.clone())),
                _ => return FloatExprPy::new(self.0.sum(index)),
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
        let t = model.add_table_1d("t", vec![]);
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
        let t = model.add_table_2d("t", vec![]);
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
        let t = model.add_table_3d("t", vec![]);
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
        let t = model.add_table_1d("t", vec![]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(SetTable1DPy::new(t.clone()), SetTable1DPy(t));
    }

    #[test]
    fn set_table_1d_getitem() {
        let mut model = Model::default();
        let t = model.add_table_1d("t", vec![Set::with_capacity(10)]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = SetTable1DPy(t.clone());
        let i = ElementUnion::Const(0);
        assert_eq!(
            t_py.__getitem__(i),
            SetExprPy::new(SetExpression::Reference(ReferenceExpression::Table(
                TableExpression::Table1D(t.id(), ElementExpression::Constant(0))
            )))
        );
    }

    #[test]
    fn set_table_2d_new() {
        let mut model = Model::default();
        let t = model.add_table_2d("t", vec![]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(SetTable2DPy::new(t.clone()), SetTable2DPy(t));
    }

    #[test]
    fn set_table_2d_getitem() {
        let mut model = Model::default();
        let t = model.add_table_2d("t", vec![vec![Set::with_capacity(10)]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = SetTable2DPy(t.clone());
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
    fn set_table_3d_new() {
        let mut model = Model::default();
        let t = model.add_table_3d("t", vec![]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(SetTable3DPy::new(t.clone()), SetTable3DPy(t));
    }

    #[test]
    fn set_table_3d_getitem() {
        let mut model = Model::default();
        let t = model.add_table_3d("t", vec![vec![vec![Set::with_capacity(10)]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = SetTable3DPy(t.clone());
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
    fn set_table_new() {
        let mut model = Model::default();
        let t = model.add_table("t", FxHashMap::default(), Set::with_capacity(10));
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(SetTablePy::new(t.clone()), SetTablePy(t));
    }

    #[test]
    fn set_table_getitem() {
        let mut model = Model::default();
        let t = model.add_table("t", FxHashMap::default(), Set::with_capacity(10));
        assert!(t.is_ok());
        let t = t.unwrap();
        let t_py = SetTablePy(t.clone());
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
    fn bool_table_1d_new() {
        let mut model = Model::default();
        let t = model.add_table_1d("t", vec![]);
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
        let t = model.add_table_2d("t", vec![]);
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
        let t = model.add_table_3d("t", vec![]);
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
        let i = TableIndexUnion::Element(ElementUnion::Const(0));
        assert_eq!(
            ArgumentExpression::from(i),
            ArgumentExpression::Element(ElementExpression::Constant(0))
        );
    }

    #[test]
    fn argument_expression_from_table_index_set() {
        let i = TableIndexUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10))));
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
        let t = model.add_table_1d("t", vec![]);
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
        let i = TableIndexUnion::Element(ElementUnion::Const(0));
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
        let i = TableIndexUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10))));
        assert_eq!(
            t_py.__getitem__(i),
            IntExprPy::new(IntegerExpression::Table(Box::new(
                NumericTableExpression::Table1DSum(
                    t.id(),
                    SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(10)))
                )
            )))
        );
    }

    #[test]
    fn int_table_2d_new() {
        let mut model = Model::default();
        let t = model.add_table_2d("t", vec![]);
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
        let x = TableIndexUnion::Element(ElementUnion::Const(0));
        let y = TableIndexUnion::Element(ElementUnion::Const(0));
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
        let x = TableIndexUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10))));
        let y = TableIndexUnion::Element(ElementUnion::Const(0));
        assert_eq!(
            t_py.__getitem__((x, y)),
            IntExprPy::new(IntegerExpression::Table(Box::new(
                NumericTableExpression::Table2DSumX(
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
        let x = TableIndexUnion::Element(ElementUnion::Const(0));
        let y = TableIndexUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10))));
        assert_eq!(
            t_py.__getitem__((x, y)),
            IntExprPy::new(IntegerExpression::Table(Box::new(
                NumericTableExpression::Table2DSumY(
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
        let x = TableIndexUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10))));
        let y = TableIndexUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10))));
        assert_eq!(
            t_py.__getitem__((x, y)),
            IntExprPy::new(IntegerExpression::Table(Box::new(
                NumericTableExpression::Table2DSum(
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
        let t = model.add_table_3d("t", vec![]);
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
        let x = TableIndexUnion::Element(ElementUnion::Const(0));
        let y = TableIndexUnion::Element(ElementUnion::Const(0));
        let z = TableIndexUnion::Element(ElementUnion::Const(0));
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
        let x = TableIndexUnion::Element(ElementUnion::Const(0));
        let y = TableIndexUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10))));
        let z = TableIndexUnion::Element(ElementUnion::Const(0));
        assert_eq!(
            t_py.__getitem__((x, y, z)),
            IntExprPy::new(IntegerExpression::Table(Box::new(
                NumericTableExpression::Table3DSum(
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
            TableIndexUnion::Element(ElementUnion::Const(0)),
            TableIndexUnion::Element(ElementUnion::Const(0)),
            TableIndexUnion::Element(ElementUnion::Const(0)),
            TableIndexUnion::Element(ElementUnion::Const(0)),
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
            TableIndexUnion::Element(ElementUnion::Const(0)),
            TableIndexUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10)))),
            TableIndexUnion::Element(ElementUnion::Const(0)),
            TableIndexUnion::Element(ElementUnion::Const(0)),
        ];
        assert_eq!(
            t_py.__getitem__(index),
            IntExprPy::new(IntegerExpression::Table(Box::new(
                NumericTableExpression::TableSum(
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
        let t = model.add_table_1d("t", vec![]);
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
        let i = TableIndexUnion::Element(ElementUnion::Const(0));
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
        let i = TableIndexUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10))));
        assert_eq!(
            t_py.__getitem__(i),
            FloatExprPy::new(ContinuousExpression::Table(Box::new(
                NumericTableExpression::Table1DSum(
                    t.id(),
                    SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(10)))
                )
            )))
        );
    }

    #[test]
    fn float_table_2d_new() {
        let mut model = Model::default();
        let t = model.add_table_2d("t", vec![]);
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
        let x = TableIndexUnion::Element(ElementUnion::Const(0));
        let y = TableIndexUnion::Element(ElementUnion::Const(0));
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
        let x = TableIndexUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10))));
        let y = TableIndexUnion::Element(ElementUnion::Const(0));
        assert_eq!(
            t_py.__getitem__((x, y)),
            FloatExprPy::new(ContinuousExpression::Table(Box::new(
                NumericTableExpression::Table2DSumX(
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
        let x = TableIndexUnion::Element(ElementUnion::Const(0));
        let y = TableIndexUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10))));
        assert_eq!(
            t_py.__getitem__((x, y)),
            FloatExprPy::new(ContinuousExpression::Table(Box::new(
                NumericTableExpression::Table2DSumY(
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
        let x = TableIndexUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10))));
        let y = TableIndexUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10))));
        assert_eq!(
            t_py.__getitem__((x, y)),
            FloatExprPy::new(ContinuousExpression::Table(Box::new(
                NumericTableExpression::Table2DSum(
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
        let t = model.add_table_3d("t", vec![]);
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
        let x = TableIndexUnion::Element(ElementUnion::Const(0));
        let y = TableIndexUnion::Element(ElementUnion::Const(0));
        let z = TableIndexUnion::Element(ElementUnion::Const(0));
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
        let x = TableIndexUnion::Element(ElementUnion::Const(0));
        let y = TableIndexUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10))));
        let z = TableIndexUnion::Element(ElementUnion::Const(0));
        assert_eq!(
            t_py.__getitem__((x, y, z)),
            FloatExprPy::new(ContinuousExpression::Table(Box::new(
                NumericTableExpression::Table3DSum(
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
            TableIndexUnion::Element(ElementUnion::Const(0)),
            TableIndexUnion::Element(ElementUnion::Const(0)),
            TableIndexUnion::Element(ElementUnion::Const(0)),
            TableIndexUnion::Element(ElementUnion::Const(0)),
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
            TableIndexUnion::Element(ElementUnion::Const(0)),
            TableIndexUnion::Set(SetUnion::Const(SetConstPy::new(Set::with_capacity(10)))),
            TableIndexUnion::Element(ElementUnion::Const(0)),
            TableIndexUnion::Element(ElementUnion::Const(0)),
        ];
        assert_eq!(
            t_py.__getitem__(index),
            FloatExprPy::new(ContinuousExpression::Table(Box::new(
                NumericTableExpression::TableSum(
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