//! A module for modeling.
mod expression;
mod state;
mod table;
mod transition;

use dypdl::prelude::*;
use dypdl::variable_type::{OrderedContinuous, ToNumeric};
pub use expression::*;
use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use rustc_hash::FxHashMap;
pub use state::StatePy;
use std::collections::HashSet;
pub use table::*;
pub use transition::{CostUnion, IntOrFloat, TransitionPy};

/// Object type.
/// This class is used to define :class:`ElementVar`, :class:`ElementResourceVar`, :class:`SetVar`, and :class:`SetConst`.
#[pyclass(name = "ObjectType")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObjectTypePy(ObjectType);

impl From<ObjectTypePy> for ObjectType {
    fn from(ob: ObjectTypePy) -> Self {
        ob.0
    }
}

impl From<ObjectType> for ObjectTypePy {
    fn from(ob: ObjectType) -> Self {
        Self(ob)
    }
}

#[derive(FromPyObject, Debug, Clone, PartialEq, Eq)]
pub enum CreateSetArgUnion {
    #[pyo3(transparent, annotation = "list[unsigned int]")]
    List(Vec<Element>),
    #[pyo3(transparent, annotation = "set[unsigned int]")]
    Set(HashSet<Element>),
}

#[derive(FromPyObject, Debug, Clone, PartialEq, Eq)]
pub enum TargetSetArgUnion {
    #[pyo3(transparent, annotation = "SetConst")]
    SetConst(SetConstPy),
    #[pyo3(transparent)]
    CreateSetArg(CreateSetArgUnion),
}

#[derive(FromPyObject, Debug, PartialEq, Clone)]
pub enum TargetArgUnion {
    #[pyo3(transparent, annotation = "unsigned int")]
    Element(Element),
    #[pyo3(transparent, annotation = "int")]
    Int(Integer),
    #[pyo3(transparent, annotation = "float")]
    Float(Continuous),
    #[pyo3(transparent)]
    Set(TargetSetArgUnion),
}

#[derive(FromPyObject, Debug, Clone, PartialEq, Eq)]
pub enum TableIndexUnion {
    #[pyo3(transparent, annotation = "unsigned int")]
    Table1D(Element),
    #[pyo3(transparent, annotation = "tuple[unsigned int, unsigned int]")]
    Table2D((Element, Element)),
    #[pyo3(
        transparent,
        annotation = "tuple[unsigned int, unsigned int, unsigned int]"
    )]
    Table3D((Element, Element, Element)),
    #[pyo3(transparent, annotation = "Sequence[unsigned int]")]
    Table(Vec<Element>),
}

#[derive(FromPyObject, Debug, Clone, PartialEq)]
pub enum TableArgUnion {
    Element(ElementTableArgUnion),
    Set(SetTableArgUnion),
    Bool(BoolTableArgUnion),
    Int(IntTableArgUnion),
    Float(FloatTableArgUnion),
}

#[derive(FromPyObject, Debug, Clone, PartialEq)]
pub enum TableUnion {
    #[pyo3(transparent)]
    Element(ElementTableUnion),
    #[pyo3(transparent)]
    Set(SetTableUnion),
    #[pyo3(transparent)]
    Bool(BoolTableUnion),
    #[pyo3(transparent)]
    Int(IntTableUnion),
    #[pyo3(transparent)]
    Float(FloatTableUnion),
}

#[derive(FromPyObject, Debug, Clone, PartialEq)]
pub enum SetDefaultArgUnion {
    #[pyo3(transparent)]
    Element(ElementTablePy),
    #[pyo3(transparent)]
    Set(SetTablePy),
    #[pyo3(transparent)]
    Bool(BoolTablePy),
    #[pyo3(transparent)]
    Int(IntTablePy),
    #[pyo3(transparent)]
    Float(FloatTablePy),
}

#[derive(FromPyObject, Debug, Clone, PartialEq, Eq)]
pub enum ElementTableArgUnion {
    #[pyo3(transparent, annotation = "list[unsigned int]")]
    Table1D(Vec<Element>),
    #[pyo3(transparent, annotation = "list[list[unsigned int]]")]
    Table2D(Vec<Vec<Element>>),
    #[pyo3(transparent, annotation = "list[list[list[unsigned int]]]")]
    Table3D(Vec<Vec<Vec<Element>>>),
    #[pyo3(transparent, annotation = "dict[list[unsigned int], unsigned int]")]
    Table(FxHashMap<Vec<Element>, Element>),
}

#[derive(FromPyObject, Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementTableUnion {
    #[pyo3(transparent, annotation = "ElementTable1D")]
    Table1D(ElementTable1DPy),
    #[pyo3(transparent, annotation = "ElementTable2D")]
    Table2D(ElementTable2DPy),
    #[pyo3(transparent, annotation = "ElementTable3D")]
    Table3D(ElementTable3DPy),
    #[pyo3(transparent, annotation = "ElementTable")]
    Table(ElementTablePy),
}

impl IntoPy<Py<PyAny>> for ElementTableUnion {
    fn into_py(self, py: Python<'_>) -> Py<PyAny> {
        match self {
            Self::Table1D(table) => table.into_py(py),
            Self::Table2D(table) => table.into_py(py),
            Self::Table3D(table) => table.into_py(py),
            Self::Table(table) => table.into_py(py),
        }
    }
}

#[derive(FromPyObject, Debug, Clone, PartialEq, Eq)]
pub enum SetTableArgUnion {
    #[pyo3(
        transparent,
        annotation = "list[Union[list[unsigned int], set[unsigned int], SetConst]]"
    )]
    Table1D(Vec<TargetSetArgUnion>),
    #[pyo3(
        transparent,
        annotation = "list[list[Union[list[unsigned int], set[unsigned int], SetConst]]]"
    )]
    Table2D(Vec<Vec<TargetSetArgUnion>>),
    #[pyo3(
        transparent,
        annotation = "list[list[list[Union[list[unsigned int], set[unsigned int], SetConst]]]]"
    )]
    Table3D(Vec<Vec<Vec<TargetSetArgUnion>>>),
    #[pyo3(
        transparent,
        annotation = "dict[list[unsigned int], Union[list[unsigned int], set[unsigned int], SetConst]]"
    )]
    Table(FxHashMap<Vec<Element>, TargetSetArgUnion>),
}

#[derive(FromPyObject, Debug, Clone, PartialEq, Eq)]
pub enum SetTableUnion {
    #[pyo3(transparent, annotation = "SetTable1D")]
    Table1D(SetTable1DPy),
    #[pyo3(transparent, annotation = "SetTable2D")]
    Table2D(SetTable2DPy),
    #[pyo3(transparent, annotation = "SetTable3D")]
    Table3D(SetTable3DPy),
    #[pyo3(transparent, annotation = "SetTable")]
    Table(SetTablePy),
}

impl IntoPy<Py<PyAny>> for SetTableUnion {
    fn into_py(self, py: Python<'_>) -> Py<PyAny> {
        match self {
            Self::Table1D(table) => table.into_py(py),
            Self::Table2D(table) => table.into_py(py),
            Self::Table3D(table) => table.into_py(py),
            Self::Table(table) => table.into_py(py),
        }
    }
}

#[derive(FromPyObject, Debug, Clone, PartialEq, Eq)]
pub enum BoolTableArgUnion {
    #[pyo3(transparent, annotation = "list[bool]")]
    Table1D(Vec<bool>),
    #[pyo3(transparent, annotation = "list[list[bool]]")]
    Table2D(Vec<Vec<bool>>),
    #[pyo3(transparent, annotation = "list[list[list[bool]]]")]
    Table3D(Vec<Vec<Vec<bool>>>),
    #[pyo3(transparent, annotation = "dict[list[unsigned int], bool]")]
    Table(FxHashMap<Vec<Element>, bool>),
}

#[derive(FromPyObject, Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoolTableUnion {
    #[pyo3(transparent, annotation = "BoolTable1D")]
    Table1D(BoolTable1DPy),
    #[pyo3(transparent, annotation = "BoolTable2D")]
    Table2D(BoolTable2DPy),
    #[pyo3(transparent, annotation = "BoolTable3D")]
    Table3D(BoolTable3DPy),
    #[pyo3(transparent, annotation = "BoolTable")]
    Table(BoolTablePy),
}

impl IntoPy<Py<PyAny>> for BoolTableUnion {
    fn into_py(self, py: Python<'_>) -> Py<PyAny> {
        match self {
            Self::Table1D(table) => table.into_py(py),
            Self::Table2D(table) => table.into_py(py),
            Self::Table3D(table) => table.into_py(py),
            Self::Table(table) => table.into_py(py),
        }
    }
}

#[derive(FromPyObject, Debug, Clone, PartialEq, Eq)]
pub enum IntTableArgUnion {
    #[pyo3(transparent, annotation = "list[unsigned int]")]
    Table1D(Vec<Integer>),
    #[pyo3(transparent, annotation = "list[list[unsigned int]]")]
    Table2D(Vec<Vec<Integer>>),
    #[pyo3(transparent, annotation = "list[list[list[unsigned int]]]")]
    Table3D(Vec<Vec<Vec<Integer>>>),
    #[pyo3(transparent, annotation = "dict[list[unsigned int], unsigned int]")]
    Table(FxHashMap<Vec<Element>, Integer>),
}

#[derive(FromPyObject, Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntTableUnion {
    #[pyo3(transparent, annotation = "IntTable1D")]
    Table1D(IntTable1DPy),
    #[pyo3(transparent, annotation = "IntTable2D")]
    Table2D(IntTable2DPy),
    #[pyo3(transparent, annotation = "IntTable3D")]
    Table3D(IntTable3DPy),
    #[pyo3(transparent, annotation = "IntTable")]
    Table(IntTablePy),
}

impl IntoPy<Py<PyAny>> for IntTableUnion {
    fn into_py(self, py: Python<'_>) -> Py<PyAny> {
        match self {
            Self::Table1D(table) => table.into_py(py),
            Self::Table2D(table) => table.into_py(py),
            Self::Table3D(table) => table.into_py(py),
            Self::Table(table) => table.into_py(py),
        }
    }
}

#[derive(FromPyObject, Debug, Clone, PartialEq)]
pub enum FloatTableArgUnion {
    #[pyo3(transparent, annotation = "list[unsigned int]")]
    Table1D(Vec<Continuous>),
    #[pyo3(transparent, annotation = "list[list[unsigned int]]")]
    Table2D(Vec<Vec<Continuous>>),
    #[pyo3(transparent, annotation = "list[list[list[unsigned int]]]")]
    Table3D(Vec<Vec<Vec<Continuous>>>),
    #[pyo3(transparent, annotation = "dict[list[unsigned int], unsigned int]")]
    Table(FxHashMap<Vec<Element>, Continuous>),
}

#[derive(FromPyObject, Debug, Clone, Copy, PartialEq)]
pub enum FloatTableUnion {
    #[pyo3(transparent, annotation = "FloatTable1D")]
    Table1D(FloatTable1DPy),
    #[pyo3(transparent, annotation = "FloatTable2D")]
    Table2D(FloatTable2DPy),
    #[pyo3(transparent, annotation = "FloatTable3D")]
    Table3D(FloatTable3DPy),
    #[pyo3(transparent, annotation = "FloatTable")]
    Table(FloatTablePy),
}

impl IntoPy<Py<PyAny>> for FloatTableUnion {
    fn into_py(self, py: Python<'_>) -> Py<PyAny> {
        match self {
            Self::Table1D(table) => table.into_py(py),
            Self::Table2D(table) => table.into_py(py),
            Self::Table3D(table) => table.into_py(py),
            Self::Table(table) => table.into_py(py),
        }
    }
}

#[allow(rustdoc::broken_intra_doc_links)]
/// DyPDL model.
///
/// Parameters
/// ----------
/// maximize: bool, default: false
///     Maximize the cost or not.
/// float_cost: bool, default: false
///     Use a continuous value to represent the cost or not.
///
/// Examples
/// --------
/// Create a model.
///
/// >>> import didppy as dp
/// >>> model = dp.Model(maximize=False, float_cost=False)
/// >>> model.maximize
/// False
/// >>> model.float_cost
/// False
///
/// Get and set the target state.
///
/// >>> import didppy as dp
/// >>> model = dp.Model()
/// >>> var = model.add_int_var(target=4)
/// >>> state = model.target_state
/// >>> state[var]
/// 4
/// >>> state[var] = 5
/// >>> model.target_state = state
/// >>> model.target_state[var]
/// 5
#[pyclass(name = "Model")]
#[derive(Debug, PartialEq, Clone, Default)]
pub struct ModelPy(Model);

impl From<ModelPy> for Model {
    fn from(model: ModelPy) -> Self {
        model.0
    }
}

impl From<Model> for ModelPy {
    fn from(model: Model) -> Self {
        Self(model)
    }
}

impl ModelPy {
    /// Get the model as reference.
    pub fn inner_as_ref(&self) -> &Model {
        &self.0
    }
}

#[pymethods]
impl ModelPy {
    #[new]
    #[pyo3(text_signature = "(maximize=False, float_cost=False)")]
    #[pyo3(signature = (maximize = false, float_cost = false))]
    fn new_py(maximize: bool, float_cost: bool) -> ModelPy {
        let mut model = ModelPy(if float_cost {
            Model::continuous_cost_model()
        } else {
            Model::integer_cost_model()
        });
        model.set_maximize(maximize);

        model
    }

    /// bool : Maximize the cost or not.
    #[getter]
    fn maximize(&self) -> bool {
        self.0.reduce_function == ReduceFunction::Max
    }

    // bool : Maximize the cost or not.
    #[setter]
    fn set_maximize(&mut self, maximize: bool) {
        if maximize {
            self.0.set_maximize();
        } else {
            self.0.set_minimize();
        }
    }

    /// bool : If the cost is represented by a continuous value or not.
    #[getter]
    pub fn float_cost(&self) -> bool {
        self.0.cost_type == CostType::Continuous
    }

    /// State : Target state.
    #[getter]
    fn target_state(&self) -> state::StatePy {
        self.0.target.clone().into()
    }

    #[setter]
    fn set_target_state(&mut self, state: state::StatePy) {
        self.0.target = state.into();
    }

    /// get_object_type(name)
    ///
    /// Gets the object type by a name.
    ///
    /// Parameters
    /// ----------
    /// name: str
    ///     Name of an object type.
    ///
    /// Returns
    /// -------
    /// ObjectType
    ///     The object type.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If no such object type.
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> model.add_object_type(number=4, name="obj")
    /// >>> obj = model.get_object_type("obj")
    /// >>> model.get_number_of_object(obj)
    /// 4
    #[pyo3(signature = (name))]
    fn get_object_type(&self, name: &str) -> PyResult<ObjectTypePy> {
        match self.0.get_object_type(name) {
            Ok(ob) => Ok(ObjectTypePy::from(ob)),
            Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
        }
    }

    /// get_number_of_object(object_type)
    ///
    /// Gets the number of objects associated with an object type.
    ///
    /// Parameters
    /// ----------
    /// object_type: ObjectType
    ///     Object type.
    ///
    /// Returns
    /// -------
    /// int
    ///     The number of objects.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If the object type is not included in the model.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> obj = model.add_object_type(number=4)
    /// >>> model.get_number_of_object(obj)
    /// 4
    #[pyo3(signature = (object_type))]
    fn get_number_of_object(&self, object_type: ObjectTypePy) -> PyResult<usize> {
        match self.0.get_number_of_objects(object_type.into()) {
            Ok(number) => Ok(number),
            Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
        }
    }

    /// add_object_type(number, name)
    ///
    /// Adds an object type to the model.
    ///
    /// Parameters
    /// ----------
    /// number: int
    ///     Number of objects.
    /// name: str or None, default: None
    ///     Name of the object type.
    ///
    /// Returns
    /// -------
    /// ObjectType
    ///     The object type.
    ///
    /// Raises
    /// ------
    /// OverflowError
    ///     if :code:`number` is negative.
    /// RuntimeError
    ///     If :code:`name` is already used.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> obj = model.add_object_type(number=4)
    /// >>> model.get_number_of_object(obj)
    /// 4
    #[pyo3(signature = (number, name = None))]
    #[pyo3(text_signature = "(number, name = None)")]
    fn add_object_type(&mut self, number: usize, name: Option<&str>) -> PyResult<ObjectTypePy> {
        let name = name.map_or_else(
            || {
                let n = self.0.state_metadata.number_of_object_types();
                format!("_object_{}", n)
            },
            String::from,
        );
        match self.0.add_object_type(name, number) {
            Ok(ob) => Ok(ObjectTypePy::from(ob)),
            Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
        }
    }

    // Disabled due to the conflict with :code:`create_set_const` and :code:`add_set_table`.
    // /// set_number_of_object(object_type, number)
    // ///
    // /// Sets the number of objects associated with an object type.
    // ///
    // /// Parameters
    // /// ----------
    // /// object_type: ObjectType
    // ///     Object type.
    // /// number: int
    // ///     The number of objects.
    // ///
    // /// Raises
    // /// ------
    // /// RuntimeError
    // ///     If the object type is not included in the model.
    // /// OverflowError
    // ///     If :code:`number` is negative.
    // ///
    // /// Examples
    // /// --------
    // /// >>> import didppy as dp
    // /// >>> model = dp.Model()
    // /// >>> obj = model.add_object_type(number=4)
    // /// >>> model.set_number_of_object(obj, 7)
    // /// >>> model.get_number_of_object(obj)
    // /// 7
    // #[pyo3(signature = (object_type, number))]
    // fn set_number_of_object(&mut self, object_type: ObjectTypePy, number: usize) -> PyResult<()> {
    //     match self.0.set_number_of_object(object_type.into(), number) {
    //         Ok(_) => Ok(()),
    //         Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
    //     }
    // }

    /// create_set_const(object_type, value)
    ///
    /// Creates a set constant given an object type.
    ///
    /// Parameters
    /// ----------
    /// object_type: ObjectType
    ///     Object type.
    /// value: list of int or set of int
    ///     The set of index of objects.
    ///
    /// Returns
    /// -------
    /// SetConst
    ///     The set constant.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If the object type is not included in the model.
    ///     If an element in :code:`value` is greater than or equal to the number of objects.
    /// TypeError
    ///     If an element in :code:`value` is negative.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> obj = model.add_object_type(number=4)
    /// >>> s = model.create_set_const(obj, [0, 1, 2])
    /// >>> var = model.add_element_var(object_type=obj, target=0)
    /// >>> s.contains(var).eval(model.target_state, model)
    /// True
    #[pyo3(signature = (object_type, value))]
    fn create_set_const(
        &self,
        object_type: ObjectTypePy,
        value: CreateSetArgUnion,
    ) -> PyResult<SetConstPy> {
        let array = match value {
            CreateSetArgUnion::List(value) => value,
            CreateSetArgUnion::Set(value) => value.into_iter().collect(),
        };
        match self.0.create_set(object_type.into(), &array) {
            Ok(set) => Ok(SetConstPy::from(set)),
            Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
        }
    }

    /// get_element_var(name)
    ///
    /// Gets an element variable by a name.
    ///
    /// Parameters
    /// ----------
    /// name: str
    ///     Name of a variable.
    ///
    /// Returns
    /// -------
    /// ElementVar
    ///     The variable.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If no such variable.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> obj = model.add_object_type(number=4)
    /// >>> model.add_element_var(object_type=obj, target=1, name="var")
    /// >>> var = model.get_element_var("var")
    /// >>> model.get_target(var)
    /// 1
    #[pyo3(signature = (name))]
    fn get_element_var(&self, name: &str) -> PyResult<ElementVarPy> {
        match self.0.get_element_variable(name) {
            Ok(var) => Ok(ElementVarPy::from(var)),
            Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
        }
    }

    /// add_element_var(object_type, target, name)
    ///
    /// Adds an element variable to the model.
    ///
    /// Parameters
    /// ----------
    /// object_type: ObjectType
    ///     Object type associated with the variable.
    /// target: int
    ///     Value of the variable in the target state.
    /// name: str or None, default: None
    ///     Name of the variable.
    ///     If None, :code:`__element_var_{id}` is used where :code:`{id}` is the id of the variable.
    ///
    /// Returns
    /// -------
    /// ElementVar
    ///     The variable.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If :code:`object_type` is not included in the model.
    ///     If :code:`name` is already used.
    /// OverflowError
    ///     If :code:`target` is negative.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> obj = model.add_object_type(number=4)
    /// >>> var = model.add_element_var(object_type=obj, target=1)
    /// >>> model.get_target(var)
    /// 1
    #[pyo3(signature = (object_type, target, name = None))]
    #[pyo3(text_signature = "(object_type, target, name=None)")]
    fn add_element_var(
        &mut self,
        object_type: ObjectTypePy,
        target: Element,
        name: Option<&str>,
    ) -> PyResult<ElementVarPy> {
        let name = name.map_or_else(
            || {
                let n = self.0.state_metadata.number_of_element_variables();
                format!("__element_var_{}", n)
            },
            String::from,
        );
        match self
            .0
            .add_element_variable(name, object_type.into(), target)
        {
            Ok(var) => Ok(ElementVarPy::from(var)),
            Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
        }
    }

    /// get_element_resource_var(name)
    ///
    /// Gets an element resource variable by a name.
    ///
    /// Parameters
    /// ----------
    /// name: str
    ///     Name of a variable.
    ///
    /// Returns
    /// -------
    /// ElementResourceVar
    ///     The variable.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If no such variable.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> obj = model.add_object_type(number=4)
    /// >>> model.add_element_resource_var(object_type=obj, target=1, less_is_better=True, name="var")
    /// >>> var = model.get_element_resource_var("var")
    /// >>> model.get_target(var)
    /// 1
    #[pyo3(signature = (name))]
    fn get_element_resource_var(&self, name: &str) -> PyResult<ElementResourceVarPy> {
        match self.0.get_element_resource_variable(name) {
            Ok(var) => Ok(ElementResourceVarPy::from(var)),
            Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
        }
    }

    /// add_element_resource_var(object_type, target, less_is_better, name)
    ///
    /// Adds an element resource variable to the model.
    ///
    /// Intuitively, with :code:`less_is_better=True`/:code:`less_is_better=False`, if everything else is the same, a state having a smaller/greater value is better.
    /// Formally, if the values of non-resource variables are the same, a state having equal or better resource variable values must lead to an equal or better solution that has equal or fewer transitions than the other.
    ///
    /// Parameters
    /// ----------
    /// object_type: ObjectType
    ///     Object type associated with the variable.
    /// target: int
    ///     Value of the variable in the target state.
    /// less_is_better: bool, default: False
    ///     Prefer a smaller value or not.
    /// name: str or None, default: None
    ///     Name of the variable.
    ///     If None, :code:`__element_resource_var_{id}` is used where :code:`{id}` is the id of the variable.
    ///
    /// Returns
    /// -------
    /// ElementResourceVar
    ///     The variable.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If :code:`object_type` is not included in the model.
    ///     If :code:`name` is already used.
    /// OverflowError
    ///     If :code:`target` is negative.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> obj = model.add_object_type(number=4)
    /// >>> var = model.add_element_resource_var(object_type=obj, target=1, less_is_better=True)
    /// >>> model.get_target(var)
    /// 1
    #[pyo3(signature = (object_type, target, less_is_better = false, name = None))]
    #[pyo3(text_signature = "(object_type, target, less_is_better=False, name=None)")]
    fn add_element_resource_var(
        &mut self,
        object_type: ObjectTypePy,
        target: Element,
        less_is_better: bool,
        name: Option<&str>,
    ) -> PyResult<ElementResourceVarPy> {
        let name = name.map_or_else(
            || {
                let n = self.0.state_metadata.number_of_element_resource_variables();
                format!("__element_resource_var_{}", n)
            },
            String::from,
        );
        match self
            .0
            .add_element_resource_variable(name, object_type.into(), less_is_better, target)
        {
            Ok(var) => Ok(ElementResourceVarPy::from(var)),
            Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
        }
    }

    /// get_set_var(name)
    ///
    /// Gets a set variable by a name.
    ///
    /// Parameters
    /// ----------
    /// name: str
    ///     Name of a variable.
    ///
    /// Returns
    /// -------
    /// SetVar
    ///     The variable.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If no such variable.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> obj = model.add_object_type(number=4)
    /// >>> model.add_set_var(object_type=obj, target=[1, 2, 3], name="var")
    /// >>> var = model.get_set_var("var")
    /// >>> var.contains(1).eval(model.target_state, model)
    /// True
    #[pyo3(signature = (name))]
    fn get_set_var(&self, name: &str) -> PyResult<SetVarPy> {
        match self.0.get_set_variable(name) {
            Ok(var) => Ok(SetVarPy::from(var)),
            Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
        }
    }

    /// add_set_var(object_type, target, name)
    ///
    /// Adds a set variable to the model.
    ///
    /// Parameters
    /// ----------
    /// object_type: ObjectType
    ///     Object type associated with the variable.
    /// target: SetConst, list of int, or set of int
    ///     Value of the variable in the target state.
    /// name: str or None, default: None
    ///     Name of the variable.
    ///     If None, :code:`__set_var_{id}` is used where :code:`{id}` is the id of the variable.
    ///
    /// Returns
    /// -------
    /// SetVar
    ///     The variable.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If :code:`object_type` is not included in the model.
    ///     If a value in :code:`target` is greater than or equal to the number of the objects.
    ///     If :code:`name` is already used.
    /// TypeError
    ///     If a value in :code:`target` is negative.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> obj = model.add_object_type(number=4)
    /// >>> var = model.add_set_var(object_type=obj, target=[1, 2, 3])
    /// >>> var.contains(1).eval(model.target_state, model)
    /// True
    #[pyo3(signature = (object_type, target, name = None))]
    #[pyo3(text_signature = "(object_type, target, name=None)")]
    fn add_set_var(
        &mut self,
        object_type: ObjectTypePy,
        target: TargetSetArgUnion,
        name: Option<&str>,
    ) -> PyResult<SetVarPy> {
        let target = self.convert_target_set_arg(Some(object_type), target)?;
        let name = name.map_or_else(
            || {
                let n = self.0.state_metadata.number_of_set_variables();
                format!("__set_var_{}", n)
            },
            String::from,
        );
        match self.0.add_set_variable(name, object_type.into(), target) {
            Ok(var) => Ok(SetVarPy::from(var)),
            Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
        }
    }

    /// get_int_var(name)
    ///
    /// Gets an integer variable by a name.
    ///
    /// Parameters
    /// ----------
    /// name: str
    ///     Name of a variable.
    ///
    /// Returns
    /// -------
    /// IntVar
    ///     The variable.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If no such variable.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> model.add_int_var(target=1, name="var")
    /// >>> var = model.get_int_var("var")
    /// >>> model.get_target(var)
    /// 1
    #[pyo3(signature = (name))]
    fn get_int_var(&self, name: &str) -> PyResult<IntVarPy> {
        match self.0.get_integer_variable(name) {
            Ok(var) => Ok(IntVarPy::from(var)),
            Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
        }
    }

    /// add_int_var(target, name)
    ///
    /// Adds an integer variable to the model.
    ///
    /// Parameters
    /// ----------
    /// target: int
    ///     Value of the variable in the target state.
    /// name: str or None, default: None
    ///     Name of the variable.
    ///     If None, :code:`__int_var_{id}` is used where :code:`{id}` is the id of the variable.
    ///
    /// Returns
    /// -------
    /// IntVar
    ///     The variable.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If :code:`name` is already used.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> var = model.add_int_var(target=1)
    /// 1
    #[pyo3(signature = (target, name = None))]
    #[pyo3(text_signature = "(target, name=None)")]
    fn add_int_var(&mut self, target: Integer, name: Option<&str>) -> PyResult<IntVarPy> {
        let name = name.map_or_else(
            || {
                let n = self.0.state_metadata.number_of_integer_variables();
                format!("__int_var_{}", n)
            },
            String::from,
        );
        match self.0.add_integer_variable(name, target) {
            Ok(var) => Ok(IntVarPy::from(var)),
            Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
        }
    }

    /// get_int_resource_var(name)
    ///
    /// Gets an integer resource variable by a name.
    ///
    /// Parameters
    /// ----------
    /// name: str
    ///     Name of a variable.
    ///
    /// Returns
    /// -------
    /// IntResourceVar
    ///     The variable.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If no such variable.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> model.add_int_resource_var(target=1, less_is_better=True, name="var")
    /// >>> var = model.get_int_resource_var("var")
    /// >>> model.get_target(var)
    /// 1
    #[pyo3(signature = (name))]
    fn get_int_resource_var(&self, name: &str) -> PyResult<IntResourceVarPy> {
        match self.0.get_integer_resource_variable(name) {
            Ok(var) => Ok(IntResourceVarPy::from(var)),
            Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
        }
    }

    /// add_int_resource_var(target, less_is_better, name)
    ///
    /// Adds an integer resource variable to the model.
    ///
    /// Intuitively, with :code:`less_is_better=True`/:code:`less_is_better=False`, if everything else is the same, a state having a smaller/greater value is better.
    /// Formally, if the values of non-resource variables are the same, a state having equal or better resource variable values must lead to an equal or better solution that has equal or fewer transitions than the other.
    ///
    /// Parameters
    /// ----------
    /// target: int
    ///     Value of the variable in the target state.
    /// less_is_better: bool, default: False
    ///     Prefer a smaller value or not.
    /// name: str or None, default: None
    ///     Name of the variable.
    ///     If None, :code:`__int_resource_var_{id}` is used where :code:`{id}` is the id of the variable.
    ///
    /// Returns
    /// -------
    /// IntResourceVar
    ///     The variable.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If :code:`name` is already used.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> var = model.add_int_resource_var(target=1, less_is_better=True)
    /// >>> model.get_target(var)
    /// 1
    #[pyo3(signature = (target, less_is_better = false, name = None))]
    #[pyo3(text_signature = "(target, less_is_better=False, name=None)")]
    fn add_int_resource_var(
        &mut self,
        target: Integer,
        less_is_better: bool,
        name: Option<&str>,
    ) -> PyResult<IntResourceVarPy> {
        let name = name.map_or_else(
            || {
                let n = self.0.state_metadata.number_of_integer_resource_variables();
                format!("__int_resource_var_{}", n)
            },
            String::from,
        );
        match self
            .0
            .add_integer_resource_variable(name, less_is_better, target)
        {
            Ok(var) => Ok(IntResourceVarPy::from(var)),
            Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
        }
    }

    /// get_float_var(name)
    ///
    /// Gets a continuous variable by a name.
    ///
    /// Parameters
    /// ----------
    /// name: str
    ///     Name of a variable.
    ///
    /// Returns
    /// -------
    /// FloatVar
    ///     The variable.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If no such variable.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> model.add_float_var(target=1.5, name="var")
    /// >>> var = model.get_float_var("var")
    /// >>> model.get_target(var)
    /// 1.5
    #[pyo3(signature = (name))]
    fn get_float_var(&self, name: &str) -> PyResult<FloatVarPy> {
        match self.0.get_continuous_variable(name) {
            Ok(var) => Ok(FloatVarPy::from(var)),
            Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
        }
    }

    /// add_float_var(target, name)
    ///
    /// Adds a continuous variable to the model.
    ///
    /// Parameters
    /// ----------
    /// target: float or int
    ///     Value of the variable in the target state.
    /// name: str or None, default: None
    ///     Name of the variable.
    ///     If None, :code:`__float_var_{id}` is used where :code:`{id}` is the id of the variable.
    ///
    /// Returns
    /// -------
    /// FloatVar
    ///     The variable.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If :code:`name` is already used.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> var = model.add_float_var(target=1.5)
    /// >>> model.get_target(var)
    /// 1.5
    #[pyo3(signature = (target, name = None))]
    #[pyo3(text_signature = "(target, name=None)")]
    fn add_float_var(&mut self, target: Continuous, name: Option<&str>) -> PyResult<FloatVarPy> {
        let name = name.map_or_else(
            || {
                let n = self.0.state_metadata.number_of_continuous_variables();
                format!("__float_var_{}", n)
            },
            String::from,
        );
        match self.0.add_continuous_variable(name, target) {
            Ok(var) => Ok(FloatVarPy::from(var)),
            Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
        }
    }

    /// get_float_resource_var(name)
    ///
    /// Gets a continuous resource variable by a name.
    ///
    /// Parameters
    /// ----------
    /// name: str
    ///     Name of a variable.
    ///
    /// Returns
    /// -------
    /// FloatResourceVar
    ///     The variable.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If no such variable.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> model.add_float_resource_var(target=1.5, less_is_better=True, name="var")
    /// >>> var = model.get_float_resource_var("var")
    /// >>> model.get_target(var)
    /// 1.5
    #[pyo3(signature = (name))]
    fn get_float_resource_var(&self, name: &str) -> PyResult<FloatResourceVarPy> {
        match self.0.get_continuous_resource_variable(name) {
            Ok(var) => Ok(FloatResourceVarPy::from(var)),
            Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
        }
    }

    /// add_float_resource_var(target, less_is_better, name)
    ///
    /// Adds a continuous resource variable to the model.
    ///
    /// Intuitively, with :code:`less_is_better=True`/:code:`less_is_better=False`, if everything else is the same, a state having a smaller/greater value is better.
    /// Formally, if the values of non-resource variables are the same, a state having equal or better resource variable values must lead to an equal or better solution that has equal or fewer transitions than the other.
    ///
    /// Parameters
    /// ----------
    /// target: float or int
    ///     Value of the variable in the target state.
    /// less_is_better: bool, default: False
    ///     Prefer a smaller value or not.
    /// name: str or None, default: None
    ///     Name of the variable.
    ///     If None, :code:`__float_resource_var_{id}` is used where :code:`{id}` is the id of the variable.
    ///
    /// Returns
    /// -------
    /// FloatResourceVar
    ///     The variable.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If :code:`name` is already used.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> var = model.add_float_resource_var(target=1.5, less_is_better=True)
    /// >>> model.get_target(var)
    /// 1.5
    #[pyo3(signature = (target, less_is_better = false, name = None))]
    #[pyo3(text_signature = "(target, less_is_better=False, name=None)")]
    fn add_float_resource_var(
        &mut self,
        target: Continuous,
        less_is_better: bool,
        name: Option<&str>,
    ) -> PyResult<FloatResourceVarPy> {
        let name = name.map_or_else(
            || {
                let n = self
                    .0
                    .state_metadata
                    .number_of_continuous_resource_variables();
                format!("__float_resource_var_{}", n)
            },
            String::from,
        );
        match self
            .0
            .add_continuous_resource_variable(name, less_is_better, target)
        {
            Ok(var) => Ok(FloatResourceVarPy::from(var)),
            Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
        }
    }

    /// get_object_type_of(var)
    ///
    /// Gets the object type associated with a variable.
    ///
    /// Parameters
    /// ----------
    /// var: ElementVar, ElementResourceVar, or SetVar
    ///     Variable.
    ///
    /// Returns
    /// -------
    /// ObjectType
    ///     The object type.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If the variable is not included in the model.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> obj = model.add_object_type(number=4)
    /// >>> var = model.add_element_var(object_type=obj, target=1)
    /// >>> obj = model.get_object_type_of(var)
    /// >>> model.get_number_of_objects(obj)
    /// 4
    #[pyo3(signature = (var))]
    fn get_object_type_of(&self, var: ObjectVarUnion) -> PyResult<ObjectTypePy> {
        let result = match var {
            ObjectVarUnion::Element(var) => self.0.get_object_type_of(ElementVariable::from(var)),
            ObjectVarUnion::ElementResource(var) => self
                .0
                .get_object_type_of(ElementResourceVariable::from(var)),
            ObjectVarUnion::Set(var) => self.0.get_object_type_of(SetVariable::from(var)),
        };
        match result {
            Ok(ob) => Ok(ObjectTypePy(ob)),
            Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
        }
    }

    /// get_target(var)
    ///
    /// Gets the value of a variable in the target state.
    ///
    /// Parameters
    /// ----------
    /// var: ElementVar, ElementResourceVar, SetVar, IntVar, IntResourceVar, FloatVar, or FloatResourceVar
    ///     Variable.
    ///
    /// Returns
    /// -------
    /// int, set, or float
    ///     The value in the target state.
    ///     For :class:`ElementVar`, :class:`ElementResourceVar`, :class:`IntVar`, and :class:`IntResourceVar`, :class:`int` is returned.
    ///     For :class:`SetVar`, :class:`set` is returned.
    ///     For :class:`FloatVar` and :class:`FloatResourceVar`, :class:`float` is returned.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If the variable is not included in the model.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> var = model.add_int_var(target=1)
    /// >>> model.get_target(var)
    /// 1
    #[pyo3(signature = (var))]
    fn get_target(&self, var: VarUnion) -> PyResult<state::VariableValueUnion> {
        match var {
            VarUnion::Element(var) => match self.0.get_target(ElementVariable::from(var)) {
                Ok(value) => Ok(state::VariableValueUnion::Element(value)),
                Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
            },
            VarUnion::ElementResource(var) => {
                match self.0.get_target(ElementResourceVariable::from(var)) {
                    Ok(value) => Ok(state::VariableValueUnion::Element(value)),
                    Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
                }
            }
            VarUnion::Set(var) => match self.0.get_target(SetVariable::from(var)) {
                Ok(value) => Ok(state::VariableValueUnion::Set(HashSet::from_iter(
                    value.ones(),
                ))),
                Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
            },
            VarUnion::Int(var) => match self.0.get_target(IntegerVariable::from(var)) {
                Ok(value) => Ok(state::VariableValueUnion::Int(value)),
                Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
            },
            VarUnion::IntResource(var) => {
                match self.0.get_target(IntegerResourceVariable::from(var)) {
                    Ok(value) => Ok(state::VariableValueUnion::Int(value)),
                    Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
                }
            }
            VarUnion::Float(var) => match self.0.get_target(ContinuousVariable::from(var)) {
                Ok(value) => Ok(state::VariableValueUnion::Float(value)),
                Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
            },
            VarUnion::FloatResource(var) => {
                match self.0.get_target(ContinuousResourceVariable::from(var)) {
                    Ok(value) => Ok(state::VariableValueUnion::Float(value)),
                    Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
                }
            }
        }
    }

    /// set_target(var, target)
    ///
    /// Sets the value of a variable in the target state.
    ///
    /// Parameters
    /// ----------
    /// var: ElementVar, ElementResourceVar, SetVar, IntVar, IntResourceVar, FloatVar, or FloatResourceVar
    ///     Variable.
    /// target: int, SetConst, list of int, set of int, or float
    ///     Value in the target state.
    ///     For :class:`ElementVar`, :class:`ElementResourceVar`, :class:`IntVar`, and :class:`IntResourceVar`, it should be :class:`int`.
    ///     For :class:`SetVar`, it should be :class:`SetConst`, :class:`list` of :class:`int`, or :class:`set` of :class:`int`.
    ///     For :class:`FloatVar` and :class:`FloatResourceVar`, it should be :class:`float`.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If the types of :code:`var` and :code:`target` mismatch.
    /// RuntimeError
    ///     If the variable is not included in the model.
    ///     If :code:`var` is :class:`SetVar` and a value in :code:`target` is greater than or equal to the number of the associated objects.
    /// OverflowError
    ///     If :code:`var` is :class:`ElementVar` or :class:`ElementResourceVar` and :code:`target` is negative.
    ///     If :code:`var` is :class:`SetVar` and a value in :code:`target` is negative.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> var = model.add_int_var(target=1)
    /// >>> model.set_target(var, 2)
    /// >>> model.get_target(var)
    /// 2
    #[pyo3(signature = (var, target))]
    fn set_target(&mut self, var: VarUnion, target: &PyAny) -> PyResult<()> {
        let result = match var {
            VarUnion::Element(var) => {
                let target = target.extract()?;
                self.0.set_target(ElementVariable::from(var), target)
            }
            VarUnion::ElementResource(var) => {
                let target = target.extract()?;
                self.0
                    .set_target(ElementResourceVariable::from(var), target)
            }
            VarUnion::Int(var) => {
                let target = target.extract()?;
                self.0.set_target(IntegerVariable::from(var), target)
            }
            VarUnion::IntResource(var) => {
                let target = target.extract()?;
                self.0
                    .set_target(IntegerResourceVariable::from(var), target)
            }
            VarUnion::Float(var) => {
                let target = target.extract()?;
                self.0.set_target(ContinuousVariable::from(var), target)
            }
            VarUnion::FloatResource(var) => {
                let target = target.extract()?;
                self.0
                    .set_target(ContinuousResourceVariable::from(var), target)
            }
            VarUnion::Set(var) => {
                let target = match target.extract()? {
                    TargetSetArgUnion::SetConst(target) => target,
                    TargetSetArgUnion::CreateSetArg(target) => {
                        let ob = self.get_object_type_of(ObjectVarUnion::Set(var))?;
                        self.create_set_const(ob, target)?
                    }
                }
                .into();
                self.0.set_target(SetVariable::from(var), target)
            }
        };
        match result {
            Ok(_) => Ok(()),
            Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
        }
    }

    /// get_preference(var)
    ///
    /// Gets the preference of a resource variable.
    ///
    /// Parameters
    /// ----------
    /// var: ElementResourceVar, IntResourceVar, or FloatResourceVar
    ///     Resource variable.
    ///
    /// Returns
    /// -------
    /// bool
    ///     :code:`True` if less is better and :code:`False` otherwise.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If the variable is not included in the model.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> var = model.add_int_resource_var(target=1, less_is_better=True)
    /// >>> model.get_preference(var)
    /// True
    #[pyo3(signature = (var))]
    fn get_preference(&self, var: ResourceVarUnion) -> PyResult<bool> {
        let result = match var {
            ResourceVarUnion::Element(var) => {
                self.0.get_preference(ElementResourceVariable::from(var))
            }
            ResourceVarUnion::Int(var) => self.0.get_preference(IntegerResourceVariable::from(var)),
            ResourceVarUnion::Float(var) => {
                self.0.get_preference(ContinuousResourceVariable::from(var))
            }
        };
        match result {
            Ok(value) => Ok(value),
            Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
        }
    }

    /// set_preference(var, less_is_better)
    ///
    /// Sets the preference of a resource variable.
    ///
    /// Parameters
    /// ----------
    /// var: ElementResourceVar, IntResourceVar, or FloatResourceVar
    ///     Resource variable.
    /// less_is_better: bool
    ///     Prefer a smaller value  or not.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If the variable is not included in the model.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> var = model.add_int_resource_var(target=1, less_is_better=True)
    /// >>> model.set_preference(var, False)
    /// >>> model.get_preference(var)
    /// False
    #[pyo3(signature = (var, less_is_better))]
    fn set_preference(&mut self, var: ResourceVarUnion, less_is_better: bool) -> PyResult<()> {
        let result = match var {
            ResourceVarUnion::Element(var) => self
                .0
                .set_preference(ElementResourceVariable::from(var), less_is_better),
            ResourceVarUnion::Int(var) => self
                .0
                .set_preference(IntegerResourceVariable::from(var), less_is_better),
            ResourceVarUnion::Float(var) => self
                .0
                .set_preference(ContinuousResourceVariable::from(var), less_is_better),
        };
        match result {
            Ok(_) => Ok(()),
            Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
        }
    }

    /// list of Condition : State constraints.   
    #[getter]
    fn state_constrs(&self) -> Vec<ConditionPy> {
        self.0
            .state_constraints
            .iter()
            .map(|constraint| ConditionPy::from(Condition::from(constraint.clone())))
            .collect()
    }

    /// add_state_constr(condition)
    ///
    /// Adds a state constraint to the model.
    ///
    /// Parameters
    /// ----------
    /// condition: Condition
    ///     State constraint.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If the condition is invalid.
    ///     E.g., it uses a variable not included in the model or the cost of the transitioned state.
    /// PanicException
    ///     If an index of a table is out of range.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> var = model.add_int_var(target=4)
    /// >>> model.add_state_constr(var >= 0)
    /// >>> model.check_state_constr(model.target_state)
    /// True
    #[pyo3(signature = (condition))]
    fn add_state_constr(&mut self, condition: ConditionPy) -> PyResult<()> {
        match self.0.add_state_constraint(condition.into()) {
            Ok(_) => Ok(()),
            Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
        }
    }

    /// check_state_constr(state)
    ///
    /// Checks if the state satisfies all the state constraints.
    ///
    /// Parameters
    /// ----------
    /// state: State
    ///     State to be checked.
    ///
    /// Returns
    /// -------
    /// bool
    ///     True if the state satisfies all the state constraints.
    ///
    /// Raises
    /// ------
    /// PanicException
    ///     If state constraints are invalid.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> var = model.add_int_var(target=4)
    /// >>> model.add_state_constr(var >= 0)
    /// >>> model.check_state_constr(model.target_state)
    /// True
    #[pyo3(signature = (state))]
    fn check_state_constr(&self, state: &state::StatePy) -> bool {
        self.0.check_constraints(state.inner_as_ref())
    }

    /// list of tuple of list of Conditions and IntExpr or FloatExpr : Base cases with their cost expressions.
    #[getter]
    fn base_cases(&self) -> Vec<(Vec<ConditionPy>, IntOrFloatExpr)> {
        self.0
            .base_cases
            .iter()
            .map(|base_case| {
                let (conditions, cost) = base_case.clone().into();
                (
                    conditions.into_iter().map(ConditionPy::from).collect(),
                    cost.map_or(
                        IntOrFloatExpr::Int(IntExprPy::from(IntegerExpression::from(0))),
                        IntOrFloatExpr::from,
                    ),
                )
            })
            .collect()
    }

    /// add_base_case(conditions, cost)
    ///
    /// Adds a base case to the model.
    ///
    /// Parameters
    /// ----------
    /// conditions: list of Condition
    ///     Base case.
    /// cost: IntExpr, IntVar, IntResourceVar, FloatExpr, FloatVar, FloatResourceVar, int, float, or None, default: None
    ///     Expression to compute the value of a base state.
    ///     This expression can use state variables in the base state.
    ///     :func:`IntExpr.state_cost()` and :func:`FloatExpr.state_cost()` are not allowed.
    ///     If None, the value of the base state is 0.
    ///     When a state satisfies multiple base cases, the minimum/maximum value is used in minimization/maximization.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If one of :code:`conditions` is invalid.
    ///     E.g., it uses a variable not included in the model or the cost of the transitioned state.
    /// PanicException
    ///     If an index of a table is out of range.
    ///     If :func:`IntExpr.state_cost()` or :func:`FloatExpr.state_cost()` is used in :code:`cost`.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> var = model.add_int_var(target=4)
    /// >>> model.add_base_case([var >= 2, var <= 5])
    /// >>> model.is_base(model.target_state)
    /// True
    /// >>> model.eval_base_cost(model.target_state)
    /// 0
    /// >>> model.add_base_case([var >= 3, var <= 4], cost=-var)
    /// >>> model.eval_base_cost(model.target_state)
    /// -4
    #[pyo3(signature = (conditions, cost = None))]
    #[pyo3(text_signature = "(conditions, cost=None)")]
    fn add_base_case(
        &mut self,
        conditions: Vec<ConditionPy>,
        cost: Option<CostUnion>,
    ) -> PyResult<()> {
        let conditions = conditions.into_iter().map(|x| x.into()).collect();

        let result = if let Some(cost) = cost {
            self.0.add_base_case_with_cost(conditions, cost)
        } else {
            self.0.add_base_case(conditions)
        };

        match result {
            Ok(_) => Ok(()),
            Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
        }
    }

    /// is_base(state)
    ///
    /// Checks if the state is a base state.
    ///
    /// Parameters
    /// ----------
    /// state: State
    ///     State to be checked.
    ///
    /// Returns
    /// -------
    /// bool
    ///     True if the state is a base state.
    ///
    /// Raises
    /// ------
    /// PanicException
    ///     If base cases are invalid.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> var = model.add_int_var(target=4)
    /// >>> model.add_base_case([var == 4])
    /// >>> model.is_base(model.target_state)
    /// True
    #[pyo3(signature = (state))]
    fn is_base(&self, state: &state::StatePy) -> bool {
        self.0.is_base(state.inner_as_ref())
    }

    /// is_base(state)
    ///
    /// Evaluates the cost of a base state.
    ///
    /// Parameters
    /// ----------
    /// state: State
    ///     State to be evaluated.
    ///
    /// Returns
    /// -------
    /// int, float, or None
    ///     None if the state is a base state.
    ///
    /// Raises
    /// ------
    /// PanicException
    ///     If base cases are invalid.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> var = model.add_int_var(target=4)
    /// >>> model.add_base_case([var == 4], cost=2)
    /// >>> model.eval_base_cost(model.target_state)
    /// 2
    #[pyo3(signature = (state))]
    fn eval_base_cost(&self, state: &state::StatePy) -> Option<IntOrFloat> {
        match self.0.cost_type {
            CostType::Integer => self
                .0
                .eval_base_cost(state.inner_as_ref())
                .map(IntOrFloat::Int),
            CostType::Continuous => self
                .0
                .eval_base_cost::<OrderedContinuous, _>(state.inner_as_ref())
                .map(|x| IntOrFloat::Float(x.to_continuous())),
        }
    }

    /// get_transitions(forced, backward)
    ///
    /// Returns the transitions of the model.
    ///
    /// Parameters
    /// ----------
    /// forced: bool, default: False
    ///     Get forced transitions or not.
    /// backward: bool, default: False
    ///     Get backward transitions or not.
    ///
    /// Returns
    /// -------
    /// list of Transition
    ///     Transitions.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> model.add_transition(dp.Transition(name="t1"))
    /// >>> model.add_transition(dp.Transition(name="t2"))
    /// >>> model.add_transition(dp.Transition(name="ft1"), forced=True)
    /// >>> model.add_transition(dp.Transition(name="ft2"), forced=True)
    /// >>> [t.name for t in model.get_transitions()]
    /// ['t1', 't2']
    /// >>> [t.name for t in model.get_transitions(forced=True)]
    /// ['ft1', 'ft2']
    #[pyo3(signature = (forced = false, backward = false))]
    fn get_transitions(&self, forced: bool, backward: bool) -> Vec<TransitionPy> {
        let transitions = if forced && backward {
            &self.0.backward_forced_transitions
        } else if forced {
            &self.0.forward_forced_transitions
        } else if backward {
            &self.0.backward_transitions
        } else {
            &self.0.forward_transitions
        };
        transitions
            .iter()
            .map(|x| TransitionPy::from(x.clone()))
            .collect()
    }

    /// add_transition(transition, forced, backward)
    ///
    /// Adds a transition to the model.
    ///
    /// Parameters
    /// ----------
    /// transition: Transition
    ///     Transition.
    /// forced: bool, default: False
    ///     If it is a forced transition or not.
    /// backward: bool, default: False
    ///     If it is a backward transition or not.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If an expression used in the transition is invalid.
    ///     E.g., it uses a variable not included in the model.
    ///     If the cost type of the model is integer and a transition with a continuous cost expression is added.
    /// PanicException
    ///     If an index of a table is out of range.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> var = model.add_int_var(target=4)
    /// >>> t = dp.Transition(name="t", cost=1 + dp.IntExpr.state_cost())
    /// >>> model.add_transition(t)
    /// >>> [t.name for t in model.get_transitions()]
    /// ['t']
    #[pyo3(signature = (transition, forced = false, backward = false))]
    #[pyo3(text_signature = "(transition, forced=False, backward=False)")]
    fn add_transition(
        &mut self,
        transition: TransitionPy,
        forced: bool,
        backward: bool,
    ) -> PyResult<()> {
        let result = if forced && backward {
            self.0.add_backward_forced_transition(transition.into())
        } else if forced {
            self.0.add_forward_forced_transition(transition.into())
        } else if backward {
            self.0.add_backward_transition(transition.into())
        } else {
            self.0.add_forward_transition(transition.into())
        };
        match result {
            Ok(_) => Ok(()),
            Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
        }
    }

    /// list of IntExpr or FloatExpr : Dual bounds.
    #[getter]
    fn dual_bounds(&self) -> Vec<IntOrFloatExpr> {
        self.0
            .dual_bounds
            .iter()
            .map(|x| match x.clone() {
                CostExpression::Integer(x) => IntOrFloatExpr::Int(IntExprPy::from(x)),
                CostExpression::Continuous(x) => IntOrFloatExpr::Float(FloatExprPy::from(x)),
            })
            .collect()
    }

    /// add_dual_bound(bound)
    ///
    /// Adds a dual bound to the model.
    ///
    /// Parameters
    /// ----------
    /// bound: IntExpr, IntVar, IntResourceVar, FloatExpr, FloatVar, FloatResourceVar, int, or float
    ///     Expression to compute a dual bound.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If :code:`bound` is invalid.
    ///     E.g., it uses a variable not included in the model or the cost of the transitioned state.
    ///     If the cost type of model is integer, and :code:`bound` is :class:`FloatExpr`, :class:`FloatVar`, :class:`FloatResourceVar`, or :class:`float`.
    /// PanicException
    ///     If an index of a table is out of range.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> var = model.add_int_var(target=4)
    /// >>> model.eval_dual_bound(model.target_state)
    /// >>> model.add_dual_bound(var)
    /// >>> model.eval_dual_bound(model.target_state)
    /// 4
    fn add_dual_bound(&mut self, bound: CostUnion) -> PyResult<()> {
        let result = match bound {
            CostUnion::Int(bound) => self.0.add_dual_bound(IntegerExpression::from(bound)),
            CostUnion::Float(bound) => self.0.add_dual_bound(ContinuousExpression::from(bound)),
        };
        match result {
            Ok(_) => Ok(()),
            Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
        }
    }

    /// eval_dual_bound(state)
    ///
    /// Evaluates the dual bound on the cost of the state.
    ///
    /// Parameters
    /// ----------
    /// state: State
    ///     State to be evaluated.
    ///
    /// Returns
    /// -------
    /// int, float, or None
    ///     The dual bound on the cost of the state.
    ///     None if no dual bound is defined.
    ///
    /// Raises
    /// ------
    /// PanicException
    ///     If dual bounds are invalid.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> var = model.add_int_var(target=4)
    /// >>> model.eval_dual_bound(model.target_state)
    /// >>> model.add_dual_bound(var)
    /// >>> model.eval_dual_bound(model.target_state)
    /// 4
    #[pyo3(signature = (state))]
    fn eval_dual_bound(&self, state: &state::StatePy) -> Option<IntOrFloat> {
        match self.0.cost_type {
            CostType::Integer => self
                .0
                .eval_dual_bound(state.inner_as_ref())
                .map(IntOrFloat::Int),
            CostType::Continuous => self
                .0
                .eval_dual_bound::<_, OrderedContinuous>(state.inner_as_ref())
                .map(|x| IntOrFloat::Float(x.to_continuous())),
        }
    }

    /// validate_forward(transitions, cost, quiet)
    ///
    /// Validates a solution consists of forward transitions.
    ///
    /// Parameters
    /// ----------
    /// transitions: list of Transition
    ///     Transitions in the solution.
    /// cost: int or float
    ///     Cost of the solution.
    /// quiet: bool, default: False
    ///     Suppress output messages.
    ///
    /// Returns
    /// -------
    /// bool
    ///     True if the solution is valid.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If the type of :code:`cost` and the cost type mismatches.
    /// PanicException
    ///     If expressions in the transitions are not valid.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> var = model.add_int_var(target=1)
    /// >>> model.add_base_case([var == 0])
    /// >>> t = dp.Transition(
    /// ...     name="t",
    /// ...     effects=[(var, var - 1)],
    /// ...     cost=dp.IntExpr.state_cost() + 1,
    /// ... )
    /// >>> model.add_transition(t)
    /// >>> model.validate_forward([t], 1)
    /// True
    #[pyo3(signature = (transitions, cost, quiet = false))]
    fn validate_forward(
        &self,
        transitions: Vec<TransitionPy>,
        cost: &PyAny,
        quiet: bool,
    ) -> PyResult<bool> {
        let transitions = transitions
            .iter()
            .map(|t| Transition::from(t.clone()))
            .collect::<Vec<_>>();
        if self.float_cost() {
            let cost: Continuous = cost.extract()?;
            Ok(self
                .0
                .validate_forward(&transitions, OrderedContinuous::from(cost), !quiet))
        } else {
            let cost: Integer = cost.extract()?;
            Ok(self.0.validate_forward(&transitions, cost, !quiet))
        }
    }

    /// add_element_table(table, default, name)
    ///
    /// Adds a table of element constants.
    /// Up to 3-dimensional tables can be added by passing a nested list.
    /// For more than 4-dimensional tables, use :class:`dict`.
    /// Values in the table can be accessed in expressions using tuples of :class:`ElementExpr`, :class:`ElementVar`, :class:`ElementResourceVar`, and :class:`int` as indices.
    ///
    /// Parameters
    /// ----------
    /// table: list of int, list of list of int, list of list of list of int, or dict[Sequence[int], int]
    ///     Table of constants.
    /// default: int or None, default: None
    ///     Default value.
    ///     Used only when :class:`dict` is given.
    ///     If a key not included in :class:`dict` is given, the table returns :code:`default`.
    /// name: str or None, default: None
    ///     Name of the table.
    ///     If :code:`None`, :code:`__element_table{dimension}_{id}` is used where :code:`{dimension}` is :code:`_1d`, :code:`_2d`, :code:`_3d`, or empty depending on the input and :code:`{id}` is the id of the table.
    ///
    /// Returns
    /// -------
    /// ElementTable1D, ElementTable2D, ElementTable3D, or ElementTable
    ///     :class:`ElementTable1D` is returned if :code:`table` is :class:`list` of :class:`int`.
    ///     :class:`ElementTable2D` is returned if :code:`table` is :class:`list` of :class:`list` of :class:`int`.
    ///     :class:`ElementTable3D` is returned if :code:`table` is :class:`list` of :class:`list` of :class:`list` of :class:`int`.
    ///     :class:`ElementTable` is returned if :class:`dict` is given.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If :code:`name` is already used.
    /// TypeError
    ///     If :code:`table` is :class:`dict` and :code:`default` is :code:`None`.
    ///     If :code:`table` is :class:`dict` and one of its keys contains a negative value.
    /// OverflowError
    ///     If a value in :code:`table` or :code:`default` is negative.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> state = model.target_state
    /// >>> table = model.add_element_table([[1, 2], [3, 4]])
    /// >>> table[0, 0].eval(state, model)
    /// 1
    #[pyo3(signature = (table, default = None, name = None))]
    #[pyo3(text_signature = "(table, default=None, name=None)")]
    fn add_element_table(
        &mut self,
        table: ElementTableArgUnion,
        default: Option<Element>,
        name: Option<&str>,
    ) -> PyResult<ElementTableUnion> {
        match table {
            ElementTableArgUnion::Table1D(table) => {
                let name = match name {
                    Some(name) => String::from(name),
                    None => {
                        let n = self.0.table_registry.element_tables.tables_1d.len();
                        format!("__element_table_1d_{}", n)
                    }
                };
                match self.0.add_table_1d(name, table) {
                    Ok(table) => Ok(ElementTableUnion::Table1D(ElementTable1DPy::from(table))),
                    Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
                }
            }
            ElementTableArgUnion::Table2D(table) => {
                let name = match name {
                    Some(name) => String::from(name),
                    None => {
                        let n = self.0.table_registry.element_tables.tables_2d.len();
                        format!("__element_table_2d_{}", n)
                    }
                };
                match self.0.add_table_2d(name, table) {
                    Ok(table) => Ok(ElementTableUnion::Table2D(ElementTable2DPy::from(table))),
                    Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
                }
            }
            ElementTableArgUnion::Table3D(table) => {
                let name = match name {
                    Some(name) => String::from(name),
                    None => {
                        let n = self.0.table_registry.element_tables.tables_3d.len();
                        format!("__element_table_3d_{}", n)
                    }
                };
                match self.0.add_table_3d(name, table) {
                    Ok(table) => Ok(ElementTableUnion::Table3D(ElementTable3DPy::from(table))),
                    Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
                }
            }
            ElementTableArgUnion::Table(table) => {
                let name = match name {
                    Some(name) => String::from(name),
                    None => {
                        let n = self.0.table_registry.element_tables.tables.len();
                        format!("__element_table_{}", n)
                    }
                };
                if let Some(default) = default {
                    match self.0.add_table(name, table, default) {
                        Ok(table) => Ok(ElementTableUnion::Table(ElementTablePy::from(table))),
                        Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
                    }
                } else {
                    Err(PyTypeError::new_err(
                        "argument 'default' must not be 'None'",
                    ))
                }
            }
        }
    }

    /// add_set_table(table, default, name, object_type)
    ///
    /// Adds a table of set constants.
    /// Up to 3-dimensional tables can be added by passing a nested list.
    /// For more than 4-dimensional tables, use :class:`dict`.
    /// Values in the table can be accessed in expressions using tuples of :class:`ElementExpr`, :class:`ElementVar`, :class:`ElementResourceVar`, and :class:`int` as indices.
    /// The union, intersection, and symmetric difference of sets in a table can be taken by using :class:`SetExpr`, :class:`SetVar`, and :class:`SetConst` in indices.
    ///
    /// Parameters
    /// ----------
    /// table: list of set values, list of list of set values, list of list of list of set values, or dict
    ///     Table of constants.
    ///     A set value can be :code:`SetConst`, :class:`list` of :class:`int`, or :class:`set` of :class:`int`.
    /// default: SetConst, list of int, set of int, or None, default: None
    ///     Default value.
    ///     Used only when :class:`dict` is given.
    ///     If a key not included in :class:`dict` is given, the table returns :code:`default`.
    /// name: str or None, default: None
    ///     Name of the table.
    ///     If :code:`None`, :code:`__set_table{dimension}_{id}` is used where :code:`{dimension}` is :code:`_1d`, :code:`_2d`, :code:`_3d`, or empty depending on the input and :code:`{id}` is the id of the table.
    /// object_type: ObjectType or None, default: None
    ///     Object type associated with constants.
    ///     Mandatory if :class:`list` of :class:`int` or :class:`set` of :class:`int` is used in :code:`table` or :code:`default`.
    ///     Otherwise, it is ignored.
    ///
    /// Returns
    /// -------
    /// SetTable1D, SetTable2D, SetTable3D, or SetTable
    ///     :class:`SetTable1D` is returned if :code:`table` is :class:`list`.
    ///     :class:`SetTable2D` is returned if :code:`table` is :class:`list` of :class:`list``.
    ///     :class:`SetTable3D` is returned if :code:`table` is :class:`list` of :class:`list` of :class:`list`.
    ///     :class:`SetTable` is returned if :class:`dict` is given.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If :code:`name` is already used.
    /// TypeError
    ///     If :code:`table` is :class:`dict` and :code:`default` is :code:`None`.
    ///     if :class:`list` of :class:`int` or :class:`set` of :class:`int` is used in :code:`table` and :code:`object_type` is :code:`None`.
    /// OverflowError
    ///     If a value in :code:`table` or :code:`default` is negative.
    ///     If :code:`table` is :class:`dict` and one of its keys contains a negative value.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> state = model.target_state
    /// >>> obj = model.add_object_type(number=4)
    /// >>> table = model.add_set_table([[[0, 1], [1, 2]], [[0, 2], [1, 3]]], object_type=obj)
    /// >>> table[0, 0].eval(state, model)
    /// {0, 1}
    /// >>> var = model.add_set_var(object_type=obj, target=[0, 1])
    /// >>> state = model.target_state
    /// >>> table.union(0, var).eval(state, model)
    /// {0, 1, 2}
    /// >>> table.intersection(0, var).eval(state, model)
    /// {1}
    /// >>> table.symmetric_difference(0, var).eval(state, model)
    /// {0, 2}
    #[pyo3(signature = (table, default = None, name = None, object_type = None))]
    #[pyo3(text_signature = "(table, default=None, name=None, object_type=None)")]
    fn add_set_table(
        &mut self,
        table: SetTableArgUnion,
        default: Option<TargetSetArgUnion>,
        name: Option<&str>,
        object_type: Option<ObjectTypePy>,
    ) -> PyResult<SetTableUnion> {
        match table {
            SetTableArgUnion::Table1D(table) => {
                let name = match name {
                    Some(name) => String::from(name),
                    None => {
                        let n = self.0.table_registry.set_tables.tables_1d.len();
                        format!("__set_table_1d_{}", n)
                    }
                };
                let mut set_table = Vec::with_capacity(table.len());
                for set in table {
                    set_table.push(self.convert_target_set_arg(object_type, set)?);
                }
                match self.0.add_table_1d(name, set_table) {
                    Ok(table) => {
                        let capacity = self
                            .0
                            .get_capacity_of_set_in_1d_table(table.clone())
                            .unwrap();
                        Ok(SetTableUnion::Table1D(SetTable1DPy::new(table, capacity)))
                    }
                    Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
                }
            }
            SetTableArgUnion::Table2D(table) => {
                let name = match name {
                    Some(name) => String::from(name),
                    None => {
                        let n = self.0.table_registry.set_tables.tables_2d.len();
                        format!("__set_table_2d_{}", n)
                    }
                };
                let mut set_table = Vec::with_capacity(table.len());
                for vector in table {
                    let mut set_vector = Vec::with_capacity(vector.len());
                    for set in vector {
                        set_vector.push(self.convert_target_set_arg(object_type, set)?);
                    }
                    set_table.push(set_vector);
                }
                match self.0.add_table_2d(name, set_table) {
                    Ok(table) => {
                        let capacity = self
                            .0
                            .get_capacity_of_set_in_2d_table(table.clone())
                            .unwrap();
                        Ok(SetTableUnion::Table2D(SetTable2DPy::new(table, capacity)))
                    }
                    Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
                }
            }
            SetTableArgUnion::Table3D(table) => {
                let name = match name {
                    Some(name) => String::from(name),
                    None => {
                        let n = self.0.table_registry.set_tables.tables_3d.len();
                        format!("__set_table_3d_{}", n)
                    }
                };
                let mut set_table = Vec::with_capacity(table.len());
                for vector2d in table {
                    let mut set_vector2d = Vec::with_capacity(vector2d.len());
                    for vector in vector2d {
                        let mut set_vector = Vec::with_capacity(vector.len());
                        for set in vector {
                            set_vector.push(self.convert_target_set_arg(object_type, set)?);
                        }
                        set_vector2d.push(set_vector);
                    }
                    set_table.push(set_vector2d);
                }
                match self.0.add_table_3d(name, set_table) {
                    Ok(table) => {
                        let capacity = self
                            .0
                            .get_capacity_of_set_in_3d_table(table.clone())
                            .unwrap();
                        Ok(SetTableUnion::Table3D(SetTable3DPy::new(table, capacity)))
                    }
                    Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
                }
            }
            SetTableArgUnion::Table(table) => {
                let name = match name {
                    Some(name) => String::from(name),
                    None => {
                        let n = self.0.table_registry.set_tables.tables.len();
                        format!("__set_table_{}", n)
                    }
                };
                let mut map = FxHashMap::default();
                for (k, v) in table {
                    map.insert(k, self.convert_target_set_arg(object_type, v)?);
                }
                if let Some(default) = default {
                    let default = self.convert_target_set_arg(object_type, default)?;
                    match self.0.add_table(name, map, default) {
                        Ok(table) => {
                            let capacity =
                                self.0.get_capacity_of_set_in_table(table.clone()).unwrap();
                            Ok(SetTableUnion::Table(SetTablePy::new(table, capacity)))
                        }
                        Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
                    }
                } else {
                    Err(PyTypeError::new_err(
                        "argument 'default' must not be 'None'",
                    ))
                }
            }
        }
    }

    /// add_bool_table(table, default, name)
    ///
    /// Adds a table of bool constants.
    /// Up to 3-dimensional tables can be added by passing a nested list.
    /// For more than 4-dimensional tables, use :class:`dict`.
    /// Values in the table can be accessed in expressions using tuples of :class:`ElementExpr`, :class:`ElementVar`, :class:`ElementResourceVar`, and :class:`int` as indices.
    ///
    /// Parameters
    /// ----------
    /// table: list of bool, list of list of bool, list of list of list of bool, or dict[Sequence[int], bool]
    ///     Table of constants.
    /// default: bool or None, default: None
    ///     Default value.
    ///     Used only when :class:`dict` is given.
    ///     If a key not included in :class:`dict` is given, the table returns :code:`default`.
    /// name: str or None, default: None
    ///     Name of the table.
    ///     If :code:`None`, :code:`__bool_table{dimension}_{id}` is used where :code:`{dimension}` is :code:`_1d`, :code:`_2d`, :code:`_3d`, or empty depending on the input and :code:`{id}` is the id of the table.
    ///
    /// Returns
    /// -------
    /// BoolTable1D, BoolTable2D, BoolTable3D, or BoolTable
    ///     :class:`BoolTable1D` is returned if :code:`table` is :class:`list` of :code:`bool`.
    ///     :class:`BoolTable2D` is returned if :code:`table` is :class:`list` of :class:`list` of :code:`bool`.
    ///     :class:`BoolTable3D` is returned if :code:`table` is :class:`list` of :class:`list` of :class:`list` of :code:`bool`.
    ///     :class:`BoolTable` is returned if :class:`dict` is given.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If :code:`name` is already used.
    /// TypeError
    ///     If :code:`table` is :class:`dict` and :code:`default` is :code:`None`.
    /// OverflowError
    ///     If :code:`table` is :class:`dict` and one of its keys contains a negative value.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> state = model.target_state
    /// >>> table = model.add_bool_table([[True, False], [False, True]])
    /// >>> table[0, 0].eval(state, model)
    /// True
    #[pyo3(signature = (table, default = None, name = None))]
    #[pyo3(text_signature = "(table, default=None, name=None)")]
    fn add_bool_table(
        &mut self,
        table: BoolTableArgUnion,
        default: Option<bool>,
        name: Option<&str>,
    ) -> PyResult<BoolTableUnion> {
        match table {
            BoolTableArgUnion::Table1D(table) => {
                let name = match name {
                    Some(name) => String::from(name),
                    None => {
                        let n = self.0.table_registry.bool_tables.tables_1d.len();
                        format!("__bool_table_1d_{}", n)
                    }
                };
                match self.0.add_table_1d(name, table) {
                    Ok(table) => Ok(BoolTableUnion::Table1D(BoolTable1DPy::from(table))),
                    Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
                }
            }
            BoolTableArgUnion::Table2D(table) => {
                let name = match name {
                    Some(name) => String::from(name),
                    None => {
                        let n = self.0.table_registry.bool_tables.tables_2d.len();
                        format!("__bool_table_2d_{}", n)
                    }
                };
                match self.0.add_table_2d(name, table) {
                    Ok(table) => Ok(BoolTableUnion::Table2D(BoolTable2DPy::from(table))),
                    Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
                }
            }
            BoolTableArgUnion::Table3D(table) => {
                let name = match name {
                    Some(name) => String::from(name),
                    None => {
                        let n = self.0.table_registry.bool_tables.tables_3d.len();
                        format!("__bool_table_3d_{}", n)
                    }
                };
                match self.0.add_table_3d(name, table) {
                    Ok(table) => Ok(BoolTableUnion::Table3D(BoolTable3DPy::from(table))),
                    Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
                }
            }
            BoolTableArgUnion::Table(table) => {
                let name = match name {
                    Some(name) => String::from(name),
                    None => {
                        let n = self.0.table_registry.bool_tables.tables.len();
                        format!("__bool_table_{}", n)
                    }
                };
                if let Some(default) = default {
                    match self.0.add_table(name, table, default) {
                        Ok(table) => Ok(BoolTableUnion::Table(BoolTablePy::from(table))),
                        Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
                    }
                } else {
                    Err(PyTypeError::new_err(
                        "argument 'default' must not be 'None'",
                    ))
                }
            }
        }
    }

    /// add_int_table(table, default, name)
    ///
    /// Adds a table of integer constants.
    /// Up to 3-dimensional tables can be added by passing a nested list.
    /// For more than 4-dimensional tables, use :class:`dict`.
    /// Values in the table can be accessed in expressions using tuples of :class:`ElementExpr`, :class:`ElementVar`, :class:`ElementResourceVar`, and :class:`int` as indices.
    /// The sum, product, maximum, and minimum of values in a table can be taken by using :class:`SetExpr`, :class:`SetVar`, and :class:`SetConst` in indices.
    ///
    /// Parameters
    /// ----------
    /// table: list of int, list of list of int, list of list of list of int, or dict[Sequence[int], int]
    ///     Table of constants.
    /// default: int or None, default: None
    ///     Default value.
    ///     Used only when :class:`dict` is given.
    ///     If a key not included in :class:`dict` is given, the table returns :code:`default`.
    /// name: str or None, default: None
    ///     Name of the table.
    ///     If :code:`None`, :code:`__int_table{dimension}_{id}` is used where :code:`{dimension}` is :code:`_1d`, :code:`_2d`, :code:`_3d`, or empty depending on the input and :code:`{id}` is the id of the table.
    ///
    /// Returns
    /// -------
    /// IntTable1D, IntTable2D, IntTable3D, or IntTable
    ///     :class:`IntTable1D` is returned if :code:`table` is :class:`list` of :class:`int`.
    ///     :class:`IntTable2D` is returned if :code:`table` is :class:`list` of :class:`list` of :class:`int`.
    ///     :class:`IntTable3D` is returned if :code:`table` is :class:`list` of :class:`list` of :class:`list` of :class:`int`.
    ///     :class:`IntTable` is returned if :class:`dict` is given.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If :code:`name` is already used.
    /// TypeError
    ///     If :code:`table` is :class:`dict` and :code:`default` is :code:`None`.
    /// OverflowError
    ///     If :code:`table` is :class:`dict` and one of its keys contains a negative value.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> state = model.target_state
    /// >>> obj = model.add_object_type(number=4)
    /// >>> table = model.add_int_table([[1, 2], [3, 4]])
    /// >>> table[0, 0].eval(state, model)
    /// 1
    /// >>> var = model.add_set_var(object_type=obj, target=[0, 1])
    /// >>> state = model.target_state
    /// >>> table[0, var].eval(state, model)
    /// 3
    /// >>> table.product(0, var).eval(state, model)
    /// 2
    /// >>> table.max(0, var).eval(state, model)
    /// 2
    /// >>> table.min(0, var).eval(state, model)
    /// 1
    #[pyo3(signature = (table, default = None, name = None))]
    #[pyo3(text_signature = "(table, default=None, name=None)")]
    fn add_int_table(
        &mut self,
        table: IntTableArgUnion,
        default: Option<Integer>,
        name: Option<&str>,
    ) -> PyResult<IntTableUnion> {
        match table {
            IntTableArgUnion::Table1D(table) => {
                let name = match name {
                    Some(name) => String::from(name),
                    None => {
                        let n = self.0.table_registry.integer_tables.tables_1d.len();
                        format!("__int_table_1d_{}", n)
                    }
                };
                match self.0.add_table_1d(name, table) {
                    Ok(table) => Ok(IntTableUnion::Table1D(IntTable1DPy::from(table))),
                    Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
                }
            }
            IntTableArgUnion::Table2D(table) => {
                let name = match name {
                    Some(name) => String::from(name),
                    None => {
                        let n = self.0.table_registry.integer_tables.tables_2d.len();
                        format!("__int_table_2d_{}", n)
                    }
                };
                match self.0.add_table_2d(name, table) {
                    Ok(table) => Ok(IntTableUnion::Table2D(IntTable2DPy::from(table))),
                    Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
                }
            }
            IntTableArgUnion::Table3D(table) => {
                let name = match name {
                    Some(name) => String::from(name),
                    None => {
                        let n = self.0.table_registry.integer_tables.tables_3d.len();
                        format!("__int_table_3d_{}", n)
                    }
                };
                match self.0.add_table_3d(name, table) {
                    Ok(table) => Ok(IntTableUnion::Table3D(IntTable3DPy::from(table))),
                    Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
                }
            }
            IntTableArgUnion::Table(table) => {
                let name = match name {
                    Some(name) => String::from(name),
                    None => {
                        let n = self.0.table_registry.integer_tables.tables.len();
                        format!("__int_table_{}", n)
                    }
                };
                if let Some(default) = default {
                    match self.0.add_table(name, table, default) {
                        Ok(table) => Ok(IntTableUnion::Table(IntTablePy::from(table))),
                        Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
                    }
                } else {
                    Err(PyTypeError::new_err(
                        "argument 'default' must not be 'None'",
                    ))
                }
            }
        }
    }

    /// add_float_table(table, default, name)
    ///
    /// Adds a table of continuous constants.
    /// Up to 3-dimensional tables can be added by passing a nested list.
    /// For more than 4-dimensional tables, use :class:`dict`.
    /// Values in the table can be accessed in expressions using tuples of :class:`ElementExpr`, :class:`ElementVar`, :class:`ElementResourceVar`, and :class:`int` as indices.
    /// The sum, product, maximum, and minimum of values in a table can be taken by using :class:`SetExpr`, :class:`SetVar`, and :class:`SetConst` in indices.
    ///
    /// Parameters
    /// ----------
    /// table: list of float or int, list of list of float or int, list of list of list of float or int, or dict[Sequence[int], Union[float|int]]
    ///     Table of constants.
    /// default: int or None, default: None
    ///     Default value.
    ///     Used only when :class:`dict` is given.
    ///     If a key not included in :class:`dict` is given, the table returns :code:`default`.
    /// name: str or None, default: None
    ///     Name of the table.
    ///     If :code:`None`, :code:`__float_table{dimension}_{id}` is used where :code:`{dimension}` is :code:`_1d`, :code:`_2d`, :code:`_3d`, or empty depending on the input and :code:`{id}` is the id of the table.
    ///
    /// Returns
    /// -------
    /// FloatTable1D, FloatTable2D, FloatTable3D, or FloatTable
    ///     :class:`FloatTable1D` is returned if :code:`table` is :class:`list` of :class:`int`.
    ///     :class:`FloatTable2D` is returned if :code:`table` is :class:`list` of :class:`list` of :class:`int`.
    ///     :class:`FloatTable3D` is returned if :code:`table` is :class:`list` of :class:`list` of :class:`list` of :class:`int`.
    ///     :class:`FloatTable` is returned if :class:`dict` is given.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If :code:`name` is already used.
    /// TypeError
    ///     If :code:`table` is :class:`dict` and :code:`default` is :code:`None`.
    /// OverflowError
    ///     If :code:`table` is :class:`dict` and one of its keys contains a negative value.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> state = model.target_state
    /// >>> obj = model.add_object_type(number=4)
    /// >>> table = model.add_float_table([[1.5, 2.5], [3.5, 4.5]])
    /// >>> table[0, 0].eval(state, model)
    /// 1.5
    /// >>> var = model.add_set_var(object_type=obj, target=[0, 1])
    /// >>> state = model.target_state
    /// >>> table[0, var].eval(state, model)
    /// 4.0
    /// >>> table.product(0, var).eval(state, model)
    /// 3.75
    /// >>> table.max(0, var).eval(state, model)
    /// 2.5
    /// >>> table.min(0, var).eval(state, model)
    /// 1.5
    #[pyo3(signature = (table, default = None, name = None))]
    #[pyo3(text_signature = "(table, default=None, name=None)")]
    fn add_float_table(
        &mut self,
        table: FloatTableArgUnion,
        default: Option<Continuous>,
        name: Option<&str>,
    ) -> PyResult<FloatTableUnion> {
        match table {
            FloatTableArgUnion::Table1D(table) => {
                let name = match name {
                    Some(name) => String::from(name),
                    None => {
                        let n = self.0.table_registry.continuous_tables.tables_1d.len();
                        format!("__float_table_1d_{}", n)
                    }
                };
                match self.0.add_table_1d(name, table) {
                    Ok(table) => Ok(FloatTableUnion::Table1D(FloatTable1DPy::from(table))),
                    Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
                }
            }
            FloatTableArgUnion::Table2D(table) => {
                let name = match name {
                    Some(name) => String::from(name),
                    None => {
                        let n = self.0.table_registry.continuous_tables.tables_2d.len();
                        format!("__float_table_2d_{}", n)
                    }
                };
                match self.0.add_table_2d(name, table) {
                    Ok(table) => Ok(FloatTableUnion::Table2D(FloatTable2DPy::from(table))),
                    Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
                }
            }
            FloatTableArgUnion::Table3D(table) => {
                let name = match name {
                    Some(name) => String::from(name),
                    None => {
                        let n = self.0.table_registry.continuous_tables.tables_3d.len();
                        format!("__float_table_3d_{}", n)
                    }
                };
                match self.0.add_table_3d(name, table) {
                    Ok(table) => Ok(FloatTableUnion::Table3D(FloatTable3DPy::from(table))),
                    Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
                }
            }
            FloatTableArgUnion::Table(table) => {
                let name = match name {
                    Some(name) => String::from(name),
                    None => {
                        let n = self.0.table_registry.continuous_tables.tables.len();
                        format!("__float_table_{}", n)
                    }
                };
                if let Some(default) = default {
                    match self.0.add_table(name, table, default) {
                        Ok(table) => Ok(FloatTableUnion::Table(FloatTablePy::from(table))),
                        Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
                    }
                } else {
                    Err(PyTypeError::new_err(
                        "argument 'default' must not be 'None'",
                    ))
                }
            }
        }
    }

    // Disabled this conflicts with the simplification of an expression.
    // /// set_table_item(table, index, value, object_type)
    // ///
    // /// Sets an item in a table of constants.
    // ///
    // /// Parameters
    // /// ----------
    // /// table: ElementTable1D, ElementTable2D, ElementTable3D, ElementTable, SetTable1D, SetTable2D, SetTable3D, SetTable, BoolTable1D, BoolTable2D, BoolTable3D, BoolTable, IntTable1D, IntTable2D, IntTable3D, IntTable, FloatTable1D, FloatTable2D, FloatTable3D, or FloatTable
    // ///     Table to update.
    // /// index: int or sequence of int
    // ///     Index to update.
    // ///     It should be :class:`int` for a 1D table.
    // ///     The length should be 2 and 3 for 2D and 3D tables, respectively.
    // /// value: int, SetConst, list of int, set of int, bool, or float
    // ///     Value to set.
    // ///
    // /// Raises
    // /// ------
    // /// RuntimeError
    // ///     If :code:`table` is not included in the model.
    // ///     If :code:`index` is out of bound of the table.
    // /// TypeError
    // ///     If :code:`table` and the shape of :code:`index` mismatch.
    // /// OverflowError
    // ///     If a value in :code:`index` is negative.
    // ///     If :code:`table` is a table of element constants and :code:`value` is negative.
    // ///     If :code:`table` is a table of set constants and a value in :code:`value` is negative.
    // #[pyo3(text_signature = "(table, index, value)")]
    // #[args(ob = "None")]
    // fn set_table_item(
    //     &mut self,
    //     table: TableUnion,
    //     index: TableIndexUnion,
    //     value: &PyAny,
    // ) -> PyResult<()> {
    //     match table {
    //         TableUnion::Element(table) => {
    //             let value = value.extract()?;
    //             self.set_element_table_item(table, index, value)
    //         }
    //         TableUnion::Set(table) => {
    //             let value = value.extract()?;
    //             self.set_set_table_item(table, index, value)
    //         }
    //         TableUnion::Bool(table) => {
    //             let value = value.extract()?;
    //             self.set_bool_table_item(table, index, value)
    //         }
    //         TableUnion::Int(table) => {
    //             let value = value.extract()?;
    //             self.set_int_table_item(table, index, value)
    //         }
    //         TableUnion::Float(table) => {
    //             let value = value.extract()?;
    //             self.set_float_table_item(table, index, value)
    //         }
    //     }
    // }

    // Disabled this conflicts with the simplification of an expression.
    // /// set_default(table, index, value, object_type)
    // ///
    // /// Sets a default value for a table of constants.
    // ///
    // /// Parameters
    // /// ----------
    // /// table: ElementTable, SetTable, BoolTable, IntTable, or FloatTable
    // ///     Table to update.
    // /// value: int, SetConst, list of int, set of int, bool, or float
    // ///     Default value to set.
    // ///
    // /// Raises
    // /// ------
    // /// RuntimeError
    // ///     If :code:`table` is not included in the model.
    // /// TypeError
    // ///     If :code:`table` and the shape of :code:`index` mismatch.
    // /// OverflowError
    // ///     If :code:`table` is a table of element constants and :code:`value` is negative.
    // ///     If :code:`table` is a table of set constants and a value in :code:`value` is negative.
    // #[pyo3(text_signature = "(table, value)")]
    // fn set_default(&mut self, table: SetDefaultArgUnion, value: &PyAny) -> PyResult<()> {
    //     let result = match table {
    //         SetDefaultArgUnion::Element(table) => {
    //             let value: Element = value.extract()?;
    //             self.0.set_default(table.into(), value)
    //         }
    //         SetDefaultArgUnion::Bool(table) => {
    //             let value: bool = value.extract()?;
    //             self.0.set_default(table.into(), value)
    //         }
    //         SetDefaultArgUnion::Int(table) => {
    //             let value: Integer = value.extract()?;
    //             self.0.set_default(table.into(), value)
    //         }
    //         SetDefaultArgUnion::Float(table) => {
    //             let value: Continuous = value.extract()?;
    //             self.0.set_default(table.into(), value)
    //         }
    //         SetDefaultArgUnion::Set(table) => {
    //             let value: TargetSetArgUnion = value.extract()?;
    //             let capacity = table.get_capacity_of_set();
    //             let value = Self::convert_target_set_arg_with_capacity(capacity, value)?;
    //             self.0.set_default(table.into(), value)
    //         }
    //     };
    //     match result {
    //         Ok(_) => Ok(()),
    //         Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
    //     }
    // }

    // Disabled this conflicts with the simplification of an expression.
    // /// update_table(table, default, object_type)
    // ///
    // /// Adds a table of set constants.
    // ///
    // /// Parameters
    // /// ----------
    // /// table: ElementTable1D, ElementTable2D, ElementTable3D, ElementTable, SetTable1D, SetTable2D, SetTable3D, SetTable, BoolTable1D, BoolTable2D, BoolTable3D, BoolTable, IntTable1D, IntTable2D, IntTable3D, IntTable, FloatTable1D, FloatTable2D, FloatTable3D, or FloatTable
    // ///     Table to update.
    // /// value
    // ///     Table of constants.
    // /// default: SetConst, list of int, set of int, or None, default: None
    // ///     Default value.
    // ///     Used only when :class:`dict` is given.
    // ///     If a key not included in :class:`dict` is given, the table returns :code:`default`.
    // ///
    // /// Raises
    // /// ------
    // /// RuntimeError
    // ///     If :code:`table` is not included in the model.
    // /// TypeError
    // ///     If :code:`table` is :class:`dict` and :code:`default` is :code:`None`.
    // /// OverflowError
    // ///     If :code:`table` is a table of element or set constants and a value in :code:`table` or :code:`default` is negative.
    // ///     If :code:`table` is :class:`dict` and one of its keys contains a negative value.
    // #[pyo3(text_signature = "(table, default=None, name=None)")]
    // #[args(default = "None")]
    // fn update_table(
    //     &mut self,
    //     table: TableUnion,
    //     value: &PyAny,
    //     default: Option<&PyAny>,
    // ) -> PyResult<()> {
    //     match table {
    //         TableUnion::Element(table) => {
    //             let value = value.extract()?;
    //             let default = if let Some(default) = default {
    //                 default.extract()?
    //             } else {
    //                 None
    //             };
    //             self.update_element_table(table, value, default)
    //         }
    //         TableUnion::Set(table) => {
    //             let value = value.extract()?;
    //             let default = if let Some(default) = default {
    //                 default.extract()?
    //             } else {
    //                 None
    //             };
    //             self.update_set_table(table, value, default)
    //         }
    //         TableUnion::Bool(table) => {
    //             let value = value.extract()?;
    //             let default = if let Some(default) = default {
    //                 default.extract()?
    //             } else {
    //                 None
    //             };
    //             self.update_bool_table(table, value, default)
    //         }
    //         TableUnion::Int(table) => {
    //             let value = value.extract()?;
    //             let default = if let Some(default) = default {
    //                 default.extract()?
    //             } else {
    //                 None
    //             };
    //             self.update_int_table(table, value, default)
    //         }
    //         TableUnion::Float(table) => {
    //             let value = value.extract()?;
    //             let default = if let Some(default) = default {
    //                 default.extract()?
    //             } else {
    //                 None
    //             };
    //             self.update_float_table(table, value, default)
    //         }
    //     }
    // }
}

impl ModelPy {
    fn convert_target_set_arg(
        &self,
        object_type: Option<ObjectTypePy>,
        target: TargetSetArgUnion,
    ) -> PyResult<Set> {
        match target {
            TargetSetArgUnion::SetConst(target) => Ok(Set::from(target)),
            TargetSetArgUnion::CreateSetArg(target) => match object_type {
                Some(ob) => Ok(Set::from(self.create_set_const(ob, target)?)),
                None => Err(PyTypeError::new_err("argument 'ob' must not be 'None'")),
            },
        }
    }

    // fn convert_target_set_arg_with_capacity(
    //     capacity: usize,
    //     target: TargetSetArgUnion,
    // ) -> PyResult<Set> {
    //     match target {
    //         TargetSetArgUnion::SetConst(target) => Ok(Set::from(target)),
    //         TargetSetArgUnion::CreateSetArg(CreateSetArgUnion::List(list)) => {
    //             let mut set = Set::with_capacity(capacity);
    //             for i in list {
    //                 if i >= capacity {
    //                     return Err(PyRuntimeError::new_err(format!(
    //                         "value `{}` >= capacity :code:`{} of a set`",
    //                         i, capacity
    //                     )));
    //                 }
    //                 set.insert(i);
    //             }
    //             Ok(set)
    //         }
    //         TargetSetArgUnion::CreateSetArg(CreateSetArgUnion::Set(hash_set)) => {
    //             let mut set = Set::with_capacity(capacity);
    //             for i in hash_set {
    //                 if i >= capacity {
    //                     return Err(PyRuntimeError::new_err(format!(
    //                         "value `{}` >= capacity :code:`{} of a set`",
    //                         i, capacity
    //                     )));
    //                 }
    //                 set.insert(i);
    //             }
    //             Ok(set)
    //         }
    //     }
    // }

    // fn set_element_table_item(
    //     &mut self,
    //     table: ElementTableUnion,
    //     index: TableIndexUnion,
    //     value: Element,
    // ) -> PyResult<()> {
    //     match (table, index) {
    //         (ElementTableUnion::Table1D(table), TableIndexUnion::Table1D(x)) => {
    //             match self.0.set_table_1d(table.into(), x, value) {
    //                 Ok(_) => Ok(()),
    //                 Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
    //             }
    //         }
    //         (ElementTableUnion::Table2D(table), TableIndexUnion::Table2D((x, y))) => {
    //             match self.0.set_table_2d(table.into(), x, y, value) {
    //                 Ok(_) => Ok(()),
    //                 Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
    //             }
    //         }
    //         (ElementTableUnion::Table3D(table), TableIndexUnion::Table3D((x, y, z))) => {
    //             match self.0.set_table_3d(table.into(), x, y, z, value) {
    //                 Ok(_) => Ok(()),
    //                 Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
    //             }
    //         }
    //         (ElementTableUnion::Table(table), TableIndexUnion::Table1D(x)) => {
    //             match self.0.set_table(table.into(), vec![x], value) {
    //                 Ok(_) => Ok(()),
    //                 Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
    //             }
    //         }
    //         (ElementTableUnion::Table(table), TableIndexUnion::Table2D((x, y))) => {
    //             match self.0.set_table(table.into(), vec![x, y], value) {
    //                 Ok(_) => Ok(()),
    //                 Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
    //             }
    //         }
    //         (ElementTableUnion::Table(table), TableIndexUnion::Table3D((x, y, z))) => {
    //             match self.0.set_table(table.into(), vec![x, y, z], value) {
    //                 Ok(_) => Ok(()),
    //                 Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
    //             }
    //         }
    //         (ElementTableUnion::Table(table), TableIndexUnion::Table(index)) => {
    //             match self.0.set_table(table.into(), index, value) {
    //                 Ok(_) => Ok(()),
    //                 Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
    //             }
    //         }
    //         (_, _) => {
    //             Err(PyTypeError::new_err("arguments ('table', 'index') failed to extract (ElementTable1D, unsigned int), (ElementTable2D, tuple[unsigned int, unsigned int]), (ElementTable3D, tuple[unsigned int, unsigned int, unsigned int]), (ElementTable, unsigned int), (ElementTable, tuple[unsigned int, unsigned int]), (ElementTable, tuple[unsigned int, unsigned int, unsigned int]), or (ElementTable, Sequence[unsigned int])"))
    //         }
    //     }
    // }

    // fn update_element_table(
    //     &mut self,
    //     table: ElementTableUnion,
    //     value: ElementTableArgUnion,
    //     default: Option<Element>,
    // ) -> PyResult<()> {
    //     match (table, value) {
    //         (ElementTableUnion::Table1D(table), ElementTableArgUnion::Table1D(value)) => {
    //             match self.0.update_table_1d(table.into(), value) {
    //                 Ok(_) => Ok(()),
    //                 Err(err) => Err(PyRuntimeError::new_err(err.to_string()))
    //             }
    //         }
    //         (ElementTableUnion::Table2D(table), ElementTableArgUnion::Table2D(value)) => {
    //             match self.0.update_table_2d(table.into(), value) {
    //                 Ok(_) => Ok(()),
    //                 Err(err) => Err(PyRuntimeError::new_err(err.to_string()))
    //             }
    //         }
    //         (ElementTableUnion::Table3D(table), ElementTableArgUnion::Table3D(value)) => {
    //             match self.0.update_table_3d(table.into(), value) {
    //                 Ok(_) => Ok(()),
    //                 Err(err) => Err(PyRuntimeError::new_err(err.to_string()))
    //             }
    //         }
    //         (ElementTableUnion::Table(table), ElementTableArgUnion::Table(value)) => {
    //             if let Some(default) = default {
    //                 match self.0.update_table(table.into(), value, default) {
    //                     Ok(_) => Ok(()),
    //                     Err(err) => Err(PyRuntimeError::new_err(err.to_string()))
    //                 }
    //             } else {
    //                 Err(PyTypeError::new_err("argument 'default' must not be 'None'"))
    //             }
    //         }
    //         _ => {
    //             Err(PyTypeError::new_err("arguments ('table', 'value') failed to extract (ElementTable1D, list[unsigned int]), (ElementTable2D, list[list[unsigned int]]), (ElementTable3D, list[list[list[unsigned int]]]), or (ElementTable, dict[Sequence[unsigned int], unsigned int])"))

    //         }
    //     }
    // }

    // fn set_set_table_item(
    //     &mut self,
    //     table: SetTableUnion,
    //     index: TableIndexUnion,
    //     value: TargetSetArgUnion,
    // ) -> PyResult<()> {
    //     match (table, index) {
    //         (SetTableUnion::Table1D(table), TableIndexUnion::Table1D(x)) => {
    //             let capacity = table.get_capacity_of_set();
    //             let value = Self::convert_target_set_arg_with_capacity(capacity, value)?;
    //             match self.0.set_table_1d(table.into(), x, value) {
    //                 Ok(_) => Ok(()),
    //                 Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
    //             }
    //         }
    //         (SetTableUnion::Table2D(table), TableIndexUnion::Table2D((x, y))) => {
    //             let capacity = table.get_capacity_of_set();
    //             let value = Self::convert_target_set_arg_with_capacity(capacity, value)?;
    //             match self.0.set_table_2d(table.into(), x, y, value) {
    //                 Ok(_) => Ok(()),
    //                 Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
    //             }
    //         }
    //         (SetTableUnion::Table3D(table), TableIndexUnion::Table3D((x, y, z))) => {
    //             let capacity = table.get_capacity_of_set();
    //             let value = Self::convert_target_set_arg_with_capacity(capacity, value)?;
    //             match self.0.set_table_3d(table.into(), x, y, z, value) {
    //                 Ok(_) => Ok(()),
    //                 Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
    //             }
    //         }
    //         (SetTableUnion::Table(table), TableIndexUnion::Table1D(x)) => {
    //             let capacity = table.get_capacity_of_set();
    //             let value = Self::convert_target_set_arg_with_capacity(capacity, value)?;
    //             match self.0.set_table(table.into(), vec![x], value) {
    //                 Ok(_) => Ok(()),
    //                 Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
    //             }
    //         }
    //         (SetTableUnion::Table(table), TableIndexUnion::Table2D((x, y))) => {
    //             let capacity = table.get_capacity_of_set();
    //             let value = Self::convert_target_set_arg_with_capacity(capacity, value)?;
    //             match self.0.set_table(table.into(), vec![x, y], value) {
    //                 Ok(_) => Ok(()),
    //                 Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
    //             }
    //         }
    //         (SetTableUnion::Table(table), TableIndexUnion::Table3D((x, y, z))) => {
    //             let capacity = table.get_capacity_of_set();
    //             let value = Self::convert_target_set_arg_with_capacity(capacity, value)?;
    //             match self.0.set_table(table.into(), vec![x, y, z], value) {
    //                 Ok(_) => Ok(()),
    //                 Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
    //             }
    //         }
    //         (SetTableUnion::Table(table), TableIndexUnion::Table(index)) => {
    //             let capacity = table.get_capacity_of_set();
    //             let value = Self::convert_target_set_arg_with_capacity(capacity, value)?;
    //             match self.0.set_table(table.into(), index, value) {
    //                 Ok(_) => Ok(()),
    //                 Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
    //             }
    //         }
    //         (_, _) => {
    //             Err(PyTypeError::new_err("arguments ('table', 'index') failed to extract (SetTable1D, unsigned int), (SetTable2D, tuple[unsigned int, unsigned int]), (SetTable3D, tuple[unsigned int, unsigned int, unsigned int]), (SetTable, unsigned int), (SetTable, tuple[unsigned int, unsigned int]), (SetTable, tuple[unsigned int, unsigned int, unsigned int]), or (SetTable, Sequence[unsigned int])"))
    //         }
    //     }
    // }

    // fn update_set_table(
    //     &mut self,
    //     table: SetTableUnion,
    //     value: SetTableArgUnion,
    //     default: Option<TargetSetArgUnion>,
    // ) -> PyResult<()> {
    //     match (table, value) {
    //         (SetTableUnion::Table1D(table), SetTableArgUnion::Table1D(value)) => {
    //             let capacity = table.get_capacity_of_set();
    //             let mut set_table = Vec::with_capacity(value.len());
    //             for set in value {
    //                 set_table.push(Self::convert_target_set_arg_with_capacity(capacity, set)?);
    //             }
    //             match self.0.update_table_1d(table.into(), set_table) {
    //                 Ok(_) => Ok(()),
    //                 Err(err) => Err(PyRuntimeError::new_err(err.to_string()))
    //             }
    //         }
    //         (SetTableUnion::Table2D(table), SetTableArgUnion::Table2D(value)) => {
    //             let capacity = table.get_capacity_of_set();
    //             let mut set_table = Vec::with_capacity(value.len());
    //             for vector in value {
    //                 let mut set_vector = Vec::with_capacity(vector.len());
    //                 for set in vector {
    //                     set_vector.push(Self::convert_target_set_arg_with_capacity(capacity, set)?);
    //                 }
    //                 set_table.push(set_vector);
    //             }
    //             match self.0.update_table_2d(table.into(), set_table) {
    //                 Ok(_) => Ok(()),
    //                 Err(err) => Err(PyRuntimeError::new_err(err.to_string()))
    //             }
    //         }
    //         (SetTableUnion::Table3D(table), SetTableArgUnion::Table3D(value)) => {
    //             let capacity = table.get_capacity_of_set();
    //             let mut set_table = Vec::with_capacity(value.len());
    //             for vector2 in value {
    //                 let mut set_vector2 = Vec::with_capacity(vector2.len());
    //                 for vector in vector2 {
    //                     let mut set_vector = Vec::with_capacity(vector.len());
    //                     for set in vector {
    //                         set_vector.push(Self::convert_target_set_arg_with_capacity(capacity, set)?);
    //                     }
    //                     set_vector2.push(set_vector);
    //                 }
    //                 set_table.push(set_vector2);
    //             }
    //             match self.0.update_table_3d(table.into(), set_table) {
    //                 Ok(_) => Ok(()),
    //                 Err(err) => Err(PyRuntimeError::new_err(err.to_string()))
    //             }
    //         }
    //         (SetTableUnion::Table(table), SetTableArgUnion::Table(value)) => {
    //             if let Some(default) = default {
    //             let capacity = table.get_capacity_of_set();
    //                 let default = Self::convert_target_set_arg_with_capacity(capacity, default)?;
    //                 let mut map = FxHashMap::default();
    //                 for (k, v) in value {
    //                     map.insert(k, Self::convert_target_set_arg_with_capacity(capacity, v)?);
    //                 }
    //                 match self.0.update_table(table.into(), map, default) {
    //                     Ok(_) => Ok(()),
    //                     Err(err) => Err(PyRuntimeError::new_err(err.to_string()))
    //                 }
    //             } else {
    //                 Err(PyTypeError::new_err("argument 'default' must not be 'None'"))
    //             }
    //         }
    //         _ => {
    //             Err(PyTypeError::new_err("arguments ('table', 'value') failed to extract (SetTable1D, list[Union[list[unsigned int], set[unsigned int], SetConst]]), (SetTable2D, list[list[Union[list[unsigned int], set[unsigned int], SetConst]]), (SetTable3D, list[list[list[Union[list[unsigned int], set[unsigned int], SetConst]]]), or (SetTable, dict[Sequence[unsigned int], Union[list[unsigned int], set[unsigned int], SetConst])"))
    //         }
    //     }
    // }

    // fn set_bool_table_item(
    //     &mut self,
    //     table: BoolTableUnion,
    //     index: TableIndexUnion,
    //     value: bool,
    // ) -> PyResult<()> {
    //     match (table, index) {
    //         (BoolTableUnion::Table1D(table), TableIndexUnion::Table1D(x)) => {
    //             match self.0.set_table_1d(table.into(), x, value) {
    //                 Ok(_) => Ok(()),
    //                 Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
    //             }
    //         }
    //         (BoolTableUnion::Table2D(table), TableIndexUnion::Table2D((x, y))) => {
    //             match self.0.set_table_2d(table.into(), x, y, value) {
    //                 Ok(_) => Ok(()),
    //                 Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
    //             }
    //         }
    //         (BoolTableUnion::Table3D(table), TableIndexUnion::Table3D((x, y, z))) => {
    //             match self.0.set_table_3d(table.into(), x, y, z, value) {
    //                 Ok(_) => Ok(()),
    //                 Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
    //             }
    //         }
    //         (BoolTableUnion::Table(table), TableIndexUnion::Table1D(x)) => {
    //             match self.0.set_table(table.into(), vec![x], value) {
    //                 Ok(_) => Ok(()),
    //                 Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
    //             }
    //         }
    //         (BoolTableUnion::Table(table), TableIndexUnion::Table2D((x, y))) => {
    //             match self.0.set_table(table.into(), vec![x, y], value) {
    //                 Ok(_) => Ok(()),
    //                 Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
    //             }
    //         }
    //         (BoolTableUnion::Table(table), TableIndexUnion::Table3D((x, y, z))) => {
    //             match self.0.set_table(table.into(), vec![x, y, z], value) {
    //                 Ok(_) => Ok(()),
    //                 Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
    //             }
    //         }
    //         (BoolTableUnion::Table(table), TableIndexUnion::Table(index)) => {
    //             match self.0.set_table(table.into(), index, value) {
    //                 Ok(_) => Ok(()),
    //                 Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
    //             }
    //         }
    //         (_, _) => {
    //             Err(PyTypeError::new_err("arguments ('table', 'index') failed to extract (BoolTable1D, unsigned int), (BoolTable2D, tuple[unsigned int, unsigned int]), (BoolTable3D, tuple[unsigned int, unsigned int, unsigned int]), (BoolTable, unsigned int), (BoolTable, tuple[unsigned int, unsigned int]), (BoolTable, tuple[unsigned int, unsigned int, unsigned int]), or (BoolTable, Sequence[unsigned int])"))
    //         }
    //     }
    // }

    // fn update_bool_table(
    //     &mut self,
    //     table: BoolTableUnion,
    //     value: BoolTableArgUnion,
    //     default: Option<bool>,
    // ) -> PyResult<()> {
    //     match (table, value) {
    //         (BoolTableUnion::Table1D(table), BoolTableArgUnion::Table1D(value)) => {
    //             match self.0.update_table_1d(table.into(), value) {
    //                 Ok(_) => Ok(()),
    //                 Err(err) => Err(PyRuntimeError::new_err(err.to_string()))
    //             }
    //         }
    //         (BoolTableUnion::Table2D(table), BoolTableArgUnion::Table2D(value)) => {
    //             match self.0.update_table_2d(table.into(), value) {
    //                 Ok(_) => Ok(()),
    //                 Err(err) => Err(PyRuntimeError::new_err(err.to_string()))
    //             }
    //         }
    //         (BoolTableUnion::Table3D(table), BoolTableArgUnion::Table3D(value)) => {
    //             match self.0.update_table_3d(table.into(), value) {
    //                 Ok(_) => Ok(()),
    //                 Err(err) => Err(PyRuntimeError::new_err(err.to_string()))
    //             }
    //         }
    //         (BoolTableUnion::Table(table), BoolTableArgUnion::Table(value)) => {
    //             if let Some(default) = default {
    //                 match self.0.update_table(table.into(), value, default) {
    //                     Ok(_) => Ok(()),
    //                     Err(err) => Err(PyRuntimeError::new_err(err.to_string()))
    //                 }
    //             } else {
    //                 Err(PyTypeError::new_err("argument 'default' must not be 'None'"))
    //             }
    //         }
    //         _ => {
    //             Err(PyTypeError::new_err("arguments ('table', 'value') failed to extract (BoolTable1D, list[bool]), (BoolTable2D, list[list[bool]]), (BoolTable3D, list[list[list[bool]]]), or (BoolTable, dict[Sequence[unsigned int], bool])"))

    //         }
    //     }
    // }

    // fn set_int_table_item(
    //     &mut self,
    //     table: IntTableUnion,
    //     index: TableIndexUnion,
    //     value: Integer,
    // ) -> PyResult<()> {
    //     match (table, index) {
    //         (IntTableUnion::Table1D(table), TableIndexUnion::Table1D(x)) => {
    //             match self.0.set_table_1d(table.into(), x, value) {
    //                 Ok(_) => Ok(()),
    //                 Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
    //             }
    //         }
    //         (IntTableUnion::Table2D(table), TableIndexUnion::Table2D((x, y))) => {
    //             match self.0.set_table_2d(table.into(), x, y, value) {
    //                 Ok(_) => Ok(()),
    //                 Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
    //             }
    //         }
    //         (IntTableUnion::Table3D(table), TableIndexUnion::Table3D((x, y, z))) => {
    //             match self.0.set_table_3d(table.into(), x, y, z, value) {
    //                 Ok(_) => Ok(()),
    //                 Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
    //             }
    //         }
    //         (IntTableUnion::Table(table), TableIndexUnion::Table1D(x)) => {
    //             match self.0.set_table(table.into(), vec![x], value) {
    //                 Ok(_) => Ok(()),
    //                 Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
    //             }
    //         }
    //         (IntTableUnion::Table(table), TableIndexUnion::Table2D((x, y))) => {
    //             match self.0.set_table(table.into(), vec![x, y], value) {
    //                 Ok(_) => Ok(()),
    //                 Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
    //             }
    //         }
    //         (IntTableUnion::Table(table), TableIndexUnion::Table3D((x, y, z))) => {
    //             match self.0.set_table(table.into(), vec![x, y, z], value) {
    //                 Ok(_) => Ok(()),
    //                 Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
    //             }
    //         }
    //         (IntTableUnion::Table(table), TableIndexUnion::Table(index)) => {
    //             match self.0.set_table(table.into(), index, value) {
    //                 Ok(_) => Ok(()),
    //                 Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
    //             }
    //         }
    //         (_, _) => {
    //             Err(PyTypeError::new_err("arguments ('table', 'index') failed to extract (IntTable1D, unsigned int), (IntTable2D, tuple[unsigned int, unsigned int]), (IntTable3D, tuple[unsigned int, unsigned int, unsigned int]), (IntTable, unsigned int), (IntTable, tuple[unsigned int, unsigned int]), (IntTable, tuple[unsigned int, unsigned int, unsigned int]), or (IntTable, Sequence[unsigned int])"))
    //         }
    //     }
    // }

    // fn update_int_table(
    //     &mut self,
    //     table: IntTableUnion,
    //     value: IntTableArgUnion,
    //     default: Option<Integer>,
    // ) -> PyResult<()> {
    //     match (table, value) {
    //         (IntTableUnion::Table1D(table), IntTableArgUnion::Table1D(value)) => {
    //             match self.0.update_table_1d(table.into(), value) {
    //                 Ok(_) => Ok(()),
    //                 Err(err) => Err(PyRuntimeError::new_err(err.to_string()))
    //             }
    //         }
    //         (IntTableUnion::Table2D(table), IntTableArgUnion::Table2D(value)) => {
    //             match self.0.update_table_2d(table.into(), value) {
    //                 Ok(_) => Ok(()),
    //                 Err(err) => Err(PyRuntimeError::new_err(err.to_string()))
    //             }
    //         }
    //         (IntTableUnion::Table3D(table), IntTableArgUnion::Table3D(value)) => {
    //             match self.0.update_table_3d(table.into(), value) {
    //                 Ok(_) => Ok(()),
    //                 Err(err) => Err(PyRuntimeError::new_err(err.to_string()))
    //             }
    //         }
    //         (IntTableUnion::Table(table), IntTableArgUnion::Table(value)) => {
    //             if let Some(default) = default {
    //                 match self.0.update_table(table.into(), value, default) {
    //                     Ok(_) => Ok(()),
    //                     Err(err) => Err(PyRuntimeError::new_err(err.to_string()))
    //                 }
    //             } else {
    //                 Err(PyTypeError::new_err("argument 'default' must not be 'None'"))
    //             }
    //         }
    //         _ => {
    //             Err(PyTypeError::new_err("arguments ('table', 'value') failed to extract (IntTable1D, list[int]), (IntTable2D, list[list[int]]), (IntTable3D, list[list[list[int]]]), or (IntTable, dict[Sequence[unsigned int], int])"))

    //         }
    //     }
    // }

    // fn set_float_table_item(
    //     &mut self,
    //     table: FloatTableUnion,
    //     index: TableIndexUnion,
    //     value: Continuous,
    // ) -> PyResult<()> {
    //     match (table, index) {
    //         (FloatTableUnion::Table1D(table), TableIndexUnion::Table1D(x)) => {
    //             match self.0.set_table_1d(table.into(), x, value) {
    //                 Ok(_) => Ok(()),
    //                 Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
    //             }
    //         }
    //         (FloatTableUnion::Table2D(table), TableIndexUnion::Table2D((x, y))) => {
    //             match self.0.set_table_2d(table.into(), x, y, value) {
    //                 Ok(_) => Ok(()),
    //                 Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
    //             }
    //         }
    //         (FloatTableUnion::Table3D(table), TableIndexUnion::Table3D((x, y, z))) => {
    //             match self.0.set_table_3d(table.into(), x, y, z, value) {
    //                 Ok(_) => Ok(()),
    //                 Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
    //             }
    //         }
    //         (FloatTableUnion::Table(table), TableIndexUnion::Table1D(x)) => {
    //             match self.0.set_table(table.into(), vec![x], value) {
    //                 Ok(_) => Ok(()),
    //                 Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
    //             }
    //         }
    //         (FloatTableUnion::Table(table), TableIndexUnion::Table2D((x, y))) => {
    //             match self.0.set_table(table.into(), vec![x, y], value) {
    //                 Ok(_) => Ok(()),
    //                 Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
    //             }
    //         }
    //         (FloatTableUnion::Table(table), TableIndexUnion::Table3D((x, y, z))) => {
    //             match self.0.set_table(table.into(), vec![x, y, z], value) {
    //                 Ok(_) => Ok(()),
    //                 Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
    //             }
    //         }
    //         (FloatTableUnion::Table(table), TableIndexUnion::Table(index)) => {
    //             match self.0.set_table(table.into(), index, value) {
    //                 Ok(_) => Ok(()),
    //                 Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
    //             }
    //         }
    //         (_, _) => {
    //             Err(PyTypeError::new_err("arguments ('table', 'index') failed to extract (FloatTable1D, unsigned int), (FloatTable2D, tuple[unsigned int, unsigned int]), (FloatTable3D, tuple[unsigned int, unsigned int, unsigned int]), (FloatTable, unsigned int), (FloatTable, tuple[unsigned int, unsigned int]), (FloatTable, tuple[unsigned int, unsigned int, unsigned int]), or (FloatTable, Sequence[unsigned int])"))
    //         }
    //     }
    // }

    // fn update_float_table(
    //     &mut self,
    //     table: FloatTableUnion,
    //     value: FloatTableArgUnion,
    //     default: Option<Continuous>,
    // ) -> PyResult<()> {
    //     match (table, value) {
    //         (FloatTableUnion::Table1D(table), FloatTableArgUnion::Table1D(value)) => {
    //             match self.0.update_table_1d(table.into(), value) {
    //                 Ok(_) => Ok(()),
    //                 Err(err) => Err(PyRuntimeError::new_err(err.to_string()))
    //             }
    //         }
    //         (FloatTableUnion::Table2D(table), FloatTableArgUnion::Table2D(value)) => {
    //             match self.0.update_table_2d(table.into(), value) {
    //                 Ok(_) => Ok(()),
    //                 Err(err) => Err(PyRuntimeError::new_err(err.to_string()))
    //             }
    //         }
    //         (FloatTableUnion::Table3D(table), FloatTableArgUnion::Table3D(value)) => {
    //             match self.0.update_table_3d(table.into(), value) {
    //                 Ok(_) => Ok(()),
    //                 Err(err) => Err(PyRuntimeError::new_err(err.to_string()))
    //             }
    //         }
    //         (FloatTableUnion::Table(table), FloatTableArgUnion::Table(value)) => {
    //             if let Some(default) = default {
    //                 match self.0.update_table(table.into(), value, default) {
    //                     Ok(_) => Ok(()),
    //                     Err(err) => Err(PyRuntimeError::new_err(err.to_string()))
    //                 }
    //             } else {
    //                 Err(PyTypeError::new_err("argument 'default' must not be 'None'"))
    //             }
    //         }
    //         _ => {
    //             Err(PyTypeError::new_err("arguments ('table', 'value') failed to extract (FloatTable1D, list[float]), (FloatTable2D, list[list[float]]), (FloatTable3D, list[list[list[float]]]), or (FloatTable, dict[Sequence[unsigned int], float])"))

    //         }
    //     }
    // }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dypdl::{BaseCase, GroundedCondition};

    #[test]
    fn object_from_py() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let ob_py = ObjectTypePy(ob);
        assert_eq!(ObjectType::from(ob_py), ob);
    }

    #[test]
    fn object_new() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        assert_eq!(ObjectTypePy::from(ob), ObjectTypePy(ob));
    }

    #[test]
    fn model_from_py() {
        let model = Model::default();
        let model_py = ModelPy(model.clone());
        assert_eq!(Model::from(model_py), model);
    }

    #[test]
    fn model_from() {
        assert_eq!(ModelPy::from(Model::default()), ModelPy(Model::default()));
    }

    #[test]
    fn model_inner_as_ref() {
        let model = Model::default();
        let model_py = ModelPy(model.clone());
        assert_eq!(model_py.inner_as_ref(), &model);
    }

    #[test]
    fn int_cost_model() {
        assert_eq!(
            ModelPy::new_py(false, false),
            ModelPy(Model {
                cost_type: CostType::Integer,
                reduce_function: ReduceFunction::Min,
                ..Default::default()
            })
        );
    }

    #[test]
    fn int_cost_model_maximize() {
        assert_eq!(
            ModelPy::new_py(true, false),
            ModelPy(Model {
                cost_type: CostType::Integer,
                reduce_function: ReduceFunction::Max,
                ..Default::default()
            })
        );
    }

    #[test]
    fn float_cost_model() {
        assert_eq!(
            ModelPy::new_py(false, true),
            ModelPy(Model {
                cost_type: CostType::Continuous,
                reduce_function: ReduceFunction::Min,
                ..Default::default()
            })
        );
    }

    #[test]
    fn float_cost_model_maximize() {
        assert_eq!(
            ModelPy::new_py(true, true),
            ModelPy(Model {
                cost_type: CostType::Continuous,
                reduce_function: ReduceFunction::Max,
                ..Default::default()
            })
        );
    }

    #[test]
    fn cost_type_int() {
        let model = ModelPy(Model {
            cost_type: CostType::Integer,
            ..Default::default()
        });
        assert!(!model.float_cost());
    }

    #[test]
    fn cost_type_float() {
        let model = ModelPy(Model {
            cost_type: CostType::Continuous,
            ..Default::default()
        });
        assert!(model.float_cost());
    }

    #[test]
    fn add_and_getobject_ok() {
        let mut model = ModelPy::default();

        let ob1 = model.add_object_type(10, None);
        assert!(ob1.is_ok());
        let ob1 = ob1.unwrap();
        let n = model.get_number_of_object(ob1);
        assert!(n.is_ok());
        assert_eq!(n.unwrap(), 10);

        let ob2 = model.add_object_type(5, None);
        assert!(ob2.is_ok());
        let ob2 = ob2.unwrap();
        let n = model.get_number_of_object(ob2);
        assert!(n.is_ok());
        assert_eq!(n.unwrap(), 5);

        assert_ne!(ob1, ob2);
    }

    #[test]
    fn add_and_getobject_with_name_ok() {
        let mut model = ModelPy::default();

        let ob1 = model.add_object_type(10, Some("ob1"));
        assert!(ob1.is_ok());
        let ob1 = ob1.unwrap();
        let n = model.get_number_of_object(ob1);
        assert!(n.is_ok());
        assert_eq!(n.unwrap(), 10);
        let ob = model.get_object_type("ob1");
        assert!(ob.is_ok());
        assert_eq!(ob.unwrap(), ob1);

        let ob2 = model.add_object_type(5, Some("ob2"));
        assert!(ob2.is_ok());
        let ob2 = ob2.unwrap();
        let n = model.get_number_of_object(ob2);
        assert!(n.is_ok());
        assert_eq!(n.unwrap(), 5);
        let ob = model.get_object_type("ob2");
        assert!(ob.is_ok());
        assert_eq!(ob.unwrap(), ob2);

        assert_ne!(ob1, ob2);
    }

    #[test]
    fn get_object_err() {
        let model = ModelPy::default();
        assert!(model.get_object_type("ob1").is_err());
    }

    #[test]
    fn get_number_of_object_err() {
        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let model = ModelPy::default();
        assert!(model.get_number_of_object(ob).is_err());
    }

    #[test]
    fn add_object_err() {
        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, Some("ob"));
        assert!(ob.is_ok());
        let snapshot = model.clone();
        let ob = model.add_object_type(10, Some("ob"));
        assert!(ob.is_err());
        assert_eq!(model, snapshot);
    }

    // #[test]
    // fn set_number_of_object_ok() {
    //     let mut model = ModelPy::default();
    //     let ob = model.add_object_type(10, None);
    //     assert!(ob.is_ok());
    //     let ob = ob.unwrap();
    //     let result = model.set_number_of_object(ob, 5);
    //     assert!(result.is_ok());
    //     let n = model.get_number_of_object(ob);
    //     assert!(n.is_ok());
    //     assert_eq!(n.unwrap(), 5);
    // }

    // #[test]
    // fn set_number_of_object_err() {
    //     let mut model = ModelPy::default();
    //     let ob = model.add_object_type(10, None);
    //     assert!(ob.is_ok());
    //     let ob = ob.unwrap();

    //     let mut model = ModelPy::default();
    //     let snapshot = model.clone();
    //     let result = model.set_number_of_object(ob, 5);
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    #[test]
    fn create_set_from_list_ok() {
        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let set = model.create_set_const(ob, CreateSetArgUnion::List(vec![0, 1, 2]));
        assert!(set.is_ok());
        assert_eq!(
            set.unwrap(),
            SetConstPy::from({
                let mut set = Set::with_capacity(10);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set
            })
        );
    }

    #[test]
    fn create_set_from_list_err() {
        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let set = model.create_set_const(ob, CreateSetArgUnion::List(vec![0, 1, 2, 10]));
        assert!(set.is_err());
    }

    #[test]
    fn create_set_from_set_ok() {
        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let mut set = HashSet::new();
        set.insert(0);
        set.insert(1);
        set.insert(2);
        let set = model.create_set_const(ob, CreateSetArgUnion::Set(set));
        assert!(set.is_ok());
        assert_eq!(
            set.unwrap(),
            SetConstPy::from({
                let mut set = Set::with_capacity(10);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set
            })
        );
    }

    #[test]
    fn create_set_from_set_err() {
        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let mut set = HashSet::new();
        set.insert(0);
        set.insert(1);
        set.insert(2);
        set.insert(10);
        let set = model.create_set_const(ob, CreateSetArgUnion::Set(set));
        assert!(set.is_err());
    }

    #[test]
    fn add_and_get_element_var_ok() {
        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let v1 = model.add_element_var(ob, 0, None);
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let n = model.get_target(VarUnion::Element(v1));
        assert!(n.is_ok());
        assert_eq!(n.unwrap(), state::VariableValueUnion::Element(0));
        let result = model.get_object_type_of(ObjectVarUnion::Element(v1));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ob);

        let v2 = model.add_element_var(ob, 1, None);
        assert!(v2.is_ok());
        let v2 = v2.unwrap();
        let n = model.get_target(VarUnion::Element(v2));
        assert!(n.is_ok());
        assert_eq!(n.unwrap(), state::VariableValueUnion::Element(1));
        let result = model.get_object_type_of(ObjectVarUnion::Element(v2));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ob);

        assert_ne!(v1, v2);
    }

    #[test]
    fn add_and_get_element_var_with_name_ok() {
        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let v1 = model.add_element_var(ob, 0, Some("v1"));
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let n = model.get_target(VarUnion::Element(v1));
        assert!(n.is_ok());
        assert_eq!(n.unwrap(), state::VariableValueUnion::Element(0));
        let v = model.get_element_var("v1");
        assert!(v.is_ok());
        assert_eq!(v.unwrap(), v1);
        let result = model.get_object_type_of(ObjectVarUnion::Element(v1));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ob);

        let v2 = model.add_element_var(ob, 1, Some("v2"));
        assert!(v2.is_ok());
        let v2 = v2.unwrap();
        let n = model.get_target(VarUnion::Element(v2));
        assert!(n.is_ok());
        assert_eq!(n.unwrap(), state::VariableValueUnion::Element(1));
        let v = model.get_element_var("v2");
        assert!(v.is_ok());
        assert_eq!(v.unwrap(), v2);
        let result = model.get_object_type_of(ObjectVarUnion::Element(v2));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ob);

        assert_ne!(v1, v2);
    }

    #[test]
    fn get_element_var_err() {
        let model = ModelPy::default();
        assert!(model.get_element_var("v").is_err());
    }

    #[test]
    fn add_element_var_err() {
        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let v = model.add_element_var(ob, 0, Some("v"));
        assert!(v.is_ok());
        let snapshot = model.clone();
        let v = model.add_element_var(ob, 0, Some("v"));
        assert!(v.is_err());
        assert_eq!(model, snapshot);
    }

    #[test]
    fn get_element_target_err() {
        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let v = model.add_element_var(ob, 0, None);
        assert!(v.is_ok());
        let v = v.unwrap();
        let model = ModelPy::default();
        assert!(model.get_target(VarUnion::Element(v)).is_err());
    }

    #[test]
    fn set_element_target_ok() {
        pyo3::prepare_freethreaded_python();

        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let v = model.add_element_var(ob, 0, None);
        assert!(v.is_ok());
        let v = v.unwrap();
        let result = Python::with_gil(|py| {
            let target = 1usize.into_py(py);
            model.set_target(VarUnion::Element(v), target.as_ref(py))
        });
        assert!(result.is_ok());
        let n = model.get_target(VarUnion::Element(v));
        assert!(n.is_ok());
        assert_eq!(n.unwrap(), state::VariableValueUnion::Element(1));
    }

    #[test]
    fn set_element_target_extract_err() {
        pyo3::prepare_freethreaded_python();

        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let v = model.add_element_var(ob, 0, None);
        assert!(v.is_ok());
        let v = v.unwrap();
        let snapshot = model.clone();
        let result = Python::with_gil(|py| {
            let target = 1.5f64.into_py(py);
            model.set_target(VarUnion::Element(v), target.as_ref(py))
        });
        assert!(result.is_err());
        assert_eq!(model, snapshot);
    }

    #[test]
    fn set_element_target_no_variable_err() {
        pyo3::prepare_freethreaded_python();

        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let v = model.add_element_var(ob, 0, None);
        assert!(v.is_ok());
        let v = v.unwrap();
        let mut model = ModelPy::default();
        let snapshot = model.clone();
        let result = Python::with_gil(|py| {
            let target = 1usize.into_py(py);
            model.set_target(VarUnion::Element(v), target.as_ref(py))
        });
        assert!(result.is_err());
        assert_eq!(model, snapshot);
    }

    #[test]
    fn get_object_type_of_element_var_err() {
        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let v = model.add_element_var(ob, 0, None);
        assert!(v.is_ok());
        let v = v.unwrap();
        let model = ModelPy::default();
        let result = model.get_object_type_of(ObjectVarUnion::Element(v));
        assert!(result.is_err());
    }

    #[test]
    fn add_and_get_element_resource_var_ok() {
        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let v1 = model.add_element_resource_var(ob, 0, false, None);
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let n = model.get_target(VarUnion::ElementResource(v1));
        assert!(n.is_ok());
        assert_eq!(n.unwrap(), state::VariableValueUnion::Element(0));
        let less_is_better = model.get_preference(ResourceVarUnion::Element(v1));
        assert!(less_is_better.is_ok());
        assert!(!less_is_better.unwrap());
        let result = model.get_object_type_of(ObjectVarUnion::ElementResource(v1));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ob);

        let v2 = model.add_element_resource_var(ob, 1, true, None);
        assert!(v2.is_ok());
        let v2 = v2.unwrap();
        let n = model.get_target(VarUnion::ElementResource(v2));
        assert!(n.is_ok());
        assert_eq!(n.unwrap(), state::VariableValueUnion::Element(1));
        let less_is_better = model.get_preference(ResourceVarUnion::Element(v2));
        assert!(less_is_better.is_ok());
        assert!(less_is_better.unwrap());
        let result = model.get_object_type_of(ObjectVarUnion::ElementResource(v2));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ob);

        assert_ne!(v1, v2);
    }

    #[test]
    fn add_and_get_element_resource_var_with_name_ok() {
        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let v1 = model.add_element_resource_var(ob, 0, false, Some("v1"));
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let n = model.get_target(VarUnion::ElementResource(v1));
        assert!(n.is_ok());
        assert_eq!(n.unwrap(), state::VariableValueUnion::Element(0));
        let v = model.get_element_resource_var("v1");
        assert!(v.is_ok());
        assert_eq!(v.unwrap(), v1);
        let less_is_better = model.get_preference(ResourceVarUnion::Element(v1));
        assert!(less_is_better.is_ok());
        assert!(!less_is_better.unwrap());
        let result = model.get_object_type_of(ObjectVarUnion::ElementResource(v1));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ob);

        let v2 = model.add_element_resource_var(ob, 1, true, Some("v2"));
        assert!(v2.is_ok());
        let v2 = v2.unwrap();
        let n = model.get_target(VarUnion::ElementResource(v2));
        assert!(n.is_ok());
        assert_eq!(n.unwrap(), state::VariableValueUnion::Element(1));
        let v = model.get_element_resource_var("v2");
        assert!(v.is_ok());
        assert_eq!(v.unwrap(), v2);
        let less_is_better = model.get_preference(ResourceVarUnion::Element(v2));
        assert!(less_is_better.is_ok());
        assert!(less_is_better.unwrap());
        let result = model.get_object_type_of(ObjectVarUnion::ElementResource(v2));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ob);

        assert_ne!(v1, v2);
    }

    #[test]
    fn get_element_resource_var_err() {
        let model = ModelPy::default();
        assert!(model.get_element_resource_var("v").is_err());
    }

    #[test]
    fn add_element_resource_var_err() {
        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let v = model.add_element_resource_var(ob, 0, false, Some("v"));
        assert!(v.is_ok());
        let snapshot = model.clone();
        let v = model.add_element_resource_var(ob, 0, false, Some("v"));
        assert!(v.is_err());
        assert_eq!(model, snapshot);
    }

    #[test]
    fn get_element_resource_target_err() {
        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let v = model.add_element_resource_var(ob, 0, false, None);
        assert!(v.is_ok());
        let v = v.unwrap();
        let model = ModelPy::default();
        assert!(model.get_target(VarUnion::ElementResource(v)).is_err());
    }

    #[test]
    fn set_element_resource_target_ok() {
        pyo3::prepare_freethreaded_python();

        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let v = model.add_element_resource_var(ob, 0, false, None);
        assert!(v.is_ok());
        let v = v.unwrap();
        let result = Python::with_gil(|py| {
            let target = 1usize.into_py(py);
            model.set_target(VarUnion::ElementResource(v), target.as_ref(py))
        });
        assert!(result.is_ok());
        let n = model.get_target(VarUnion::ElementResource(v));
        assert!(n.is_ok());
        assert_eq!(n.unwrap(), state::VariableValueUnion::Element(1));
    }

    #[test]
    fn set_element_resource_target_extract_err() {
        pyo3::prepare_freethreaded_python();

        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let v = model.add_element_resource_var(ob, 0, false, None);
        assert!(v.is_ok());
        let v = v.unwrap();
        let snapshot = model.clone();
        let result = Python::with_gil(|py| {
            let target = 1.5f64.into_py(py);
            model.set_target(VarUnion::ElementResource(v), target.as_ref(py))
        });
        assert!(result.is_err());
        assert_eq!(model, snapshot);
    }

    #[test]
    fn set_element_resource_target_no_variable_err() {
        pyo3::prepare_freethreaded_python();

        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let v = model.add_element_resource_var(ob, 0, false, None);
        assert!(v.is_ok());
        let v = v.unwrap();
        let mut model = ModelPy::default();
        let snapshot = model.clone();
        let result = Python::with_gil(|py| {
            let target = 1usize.into_py(py);
            model.set_target(VarUnion::ElementResource(v), target.as_ref(py))
        });
        assert!(result.is_err());
        assert_eq!(model, snapshot);
    }

    #[test]
    fn get_object_type_of_element_resource_var_err() {
        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let v = model.add_element_resource_var(ob, 0, false, None);
        assert!(v.is_ok());
        let v = v.unwrap();
        let model = ModelPy::default();
        let result = model.get_object_type_of(ObjectVarUnion::ElementResource(v));
        assert!(result.is_err());
    }

    #[test]
    fn get_element_preference_err() {
        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let v = model.add_element_resource_var(ob, 0, false, None);
        assert!(v.is_ok());
        let v = v.unwrap();
        let model = ModelPy::default();
        assert!(model.get_preference(ResourceVarUnion::Element(v)).is_err());
    }

    #[test]
    fn set_elemnet_preference_ok() {
        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let v = model.add_element_resource_var(ob, 0, false, None);
        assert!(v.is_ok());
        let v = v.unwrap();
        let result = model.set_preference(ResourceVarUnion::Element(v), true);
        assert!(result.is_ok());
        let less_is_better = model.get_preference(ResourceVarUnion::Element(v));
        assert!(less_is_better.is_ok());
        assert!(less_is_better.unwrap());
    }

    #[test]
    fn set_elemnet_preference_err() {
        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let v = model.add_element_resource_var(ob, 0, false, None);
        assert!(v.is_ok());
        let v = v.unwrap();
        let mut model = ModelPy::default();
        let snapshot = model.clone();
        let result = model.set_preference(ResourceVarUnion::Element(v), true);
        assert!(result.is_err());
        assert_eq!(model, snapshot);
    }

    #[test]
    fn add_from_list_and_get_set_var_ok() {
        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let target = TargetSetArgUnion::CreateSetArg(CreateSetArgUnion::List(vec![0, 1, 2]));
        let v1 = model.add_set_var(ob, target, None);
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let target = model.get_target(VarUnion::Set(v1));
        assert!(target.is_ok());
        assert_eq!(
            target.unwrap(),
            state::VariableValueUnion::Set(HashSet::from_iter(vec![0, 1, 2])),
        );
        let result = model.get_object_type_of(ObjectVarUnion::Set(v1));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ob);

        let target = TargetSetArgUnion::CreateSetArg(CreateSetArgUnion::List(vec![3, 4, 5]));
        let v2 = model.add_set_var(ob, target, None);
        assert!(v2.is_ok());
        let v2 = v2.unwrap();
        let target = model.get_target(VarUnion::Set(v2));
        assert!(target.is_ok());
        assert_eq!(
            target.unwrap(),
            state::VariableValueUnion::Set(HashSet::from_iter(vec![3, 4, 5])),
        );
        let result = model.get_object_type_of(ObjectVarUnion::Set(v2));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ob);

        assert_ne!(v1, v2);
    }

    #[test]
    fn add_from_list_and_get_set_var_with_name_ok() {
        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let target = TargetSetArgUnion::CreateSetArg(CreateSetArgUnion::List(vec![0, 1, 2]));
        let v1 = model.add_set_var(ob, target, Some("v1"));
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let target = model.get_target(VarUnion::Set(v1));
        assert!(target.is_ok());
        assert_eq!(
            target.unwrap(),
            state::VariableValueUnion::Set(HashSet::from_iter(vec![0, 1, 2])),
        );
        let v = model.get_set_var("v1");
        assert!(v.is_ok());
        assert_eq!(v.unwrap(), v1);
        let result = model.get_object_type_of(ObjectVarUnion::Set(v1));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ob);

        let target = TargetSetArgUnion::CreateSetArg(CreateSetArgUnion::List(vec![3, 4, 5]));
        let v2 = model.add_set_var(ob, target, Some("v2"));
        assert!(v2.is_ok());
        let v2 = v2.unwrap();
        let target = model.get_target(VarUnion::Set(v2));
        assert!(target.is_ok());
        assert_eq!(
            target.unwrap(),
            state::VariableValueUnion::Set(HashSet::from_iter(vec![3, 4, 5])),
        );
        let v = model.get_set_var("v2");
        assert!(v.is_ok());
        assert_eq!(v.unwrap(), v2);
        let result = model.get_object_type_of(ObjectVarUnion::Set(v2));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ob);

        assert_ne!(v1, v2);
    }

    #[test]
    fn add_set_var_from_list_err() {
        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let snapshot = model.clone();
        let target = TargetSetArgUnion::CreateSetArg(CreateSetArgUnion::List(vec![0, 1, 2, 10]));
        assert!(model.add_set_var(ob, target, None).is_err());
        assert_eq!(model, snapshot);
    }

    #[test]
    fn add_from_set_and_get_set_var_ok() {
        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let target = TargetSetArgUnion::CreateSetArg(CreateSetArgUnion::Set({
            let mut set = HashSet::default();
            set.insert(0);
            set.insert(1);
            set.insert(2);
            set
        }));
        let v1 = model.add_set_var(ob, target, None);
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let target = model.get_target(VarUnion::Set(v1));
        assert!(target.is_ok());
        assert_eq!(
            target.unwrap(),
            state::VariableValueUnion::Set(HashSet::from_iter(vec![0, 1, 2])),
        );
        let result = model.get_object_type_of(ObjectVarUnion::Set(v1));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ob);

        let target = TargetSetArgUnion::CreateSetArg(CreateSetArgUnion::Set({
            let mut set = HashSet::default();
            set.insert(3);
            set.insert(4);
            set.insert(5);
            set
        }));
        let v2 = model.add_set_var(ob, target, None);
        assert!(v2.is_ok());
        let v2 = v2.unwrap();
        let target = model.get_target(VarUnion::Set(v2));
        assert!(target.is_ok());
        assert_eq!(
            target.unwrap(),
            state::VariableValueUnion::Set(HashSet::from_iter(vec![3, 4, 5])),
        );
        let result = model.get_object_type_of(ObjectVarUnion::Set(v2));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ob);

        assert_ne!(v1, v2);
    }

    #[test]
    fn add_from_set_and_get_set_var_with_name_ok() {
        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let target = TargetSetArgUnion::CreateSetArg(CreateSetArgUnion::Set({
            let mut set = HashSet::default();
            set.insert(0);
            set.insert(1);
            set.insert(2);
            set
        }));
        let v1 = model.add_set_var(ob, target, Some("v1"));
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let target = model.get_target(VarUnion::Set(v1));
        assert!(target.is_ok());
        assert_eq!(
            target.unwrap(),
            state::VariableValueUnion::Set(HashSet::from_iter(vec![0, 1, 2])),
        );
        let v = model.get_set_var("v1");
        assert!(v.is_ok());
        assert_eq!(v.unwrap(), v1);
        let result = model.get_object_type_of(ObjectVarUnion::Set(v1));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ob);

        let target = TargetSetArgUnion::CreateSetArg(CreateSetArgUnion::Set({
            let mut set = HashSet::default();
            set.insert(3);
            set.insert(4);
            set.insert(5);
            set
        }));
        let v2 = model.add_set_var(ob, target, Some("v2"));
        assert!(v2.is_ok());
        let v2 = v2.unwrap();
        let target = model.get_target(VarUnion::Set(v2));
        assert!(target.is_ok());
        assert_eq!(
            target.unwrap(),
            state::VariableValueUnion::Set(HashSet::from_iter(vec![3, 4, 5])),
        );
        let v = model.get_set_var("v2");
        assert!(v.is_ok());
        assert_eq!(v.unwrap(), v2);
        let result = model.get_object_type_of(ObjectVarUnion::Set(v2));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ob);

        assert_ne!(v1, v2);
    }

    #[test]
    fn add_set_var_from_set_err() {
        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let snapshot = model.clone();
        let target = TargetSetArgUnion::CreateSetArg(CreateSetArgUnion::Set({
            let mut set = HashSet::default();
            set.insert(3);
            set.insert(4);
            set.insert(5);
            set.insert(10);
            set
        }));
        assert!(model.add_set_var(ob, target, None).is_err());
        assert_eq!(model, snapshot);
    }

    #[test]
    fn add_from_set_const_and_get_set_var_ok() {
        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let target = TargetSetArgUnion::SetConst(SetConstPy::from({
            let mut set = Set::with_capacity(10);
            set.insert(0);
            set.insert(1);
            set.insert(2);
            set
        }));
        let v1 = model.add_set_var(ob, target, None);
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let target = model.get_target(VarUnion::Set(v1));
        assert!(target.is_ok());
        assert_eq!(
            target.unwrap(),
            state::VariableValueUnion::Set(HashSet::from_iter(vec![0, 1, 2])),
        );
        let result = model.get_object_type_of(ObjectVarUnion::Set(v1));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ob);

        let target = TargetSetArgUnion::SetConst(SetConstPy::from({
            let mut set = Set::with_capacity(10);
            set.insert(3);
            set.insert(4);
            set.insert(5);
            set
        }));
        let v2 = model.add_set_var(ob, target, None);
        assert!(v2.is_ok());
        let v2 = v2.unwrap();
        let target = model.get_target(VarUnion::Set(v2));
        assert!(target.is_ok());
        assert_eq!(
            target.unwrap(),
            state::VariableValueUnion::Set(HashSet::from_iter(vec![3, 4, 5])),
        );
        let result = model.get_object_type_of(ObjectVarUnion::Set(v2));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ob);

        assert_ne!(v1, v2);
    }

    #[test]
    fn add_from_set_const_and_get_set_var_with_name_ok() {
        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let target = TargetSetArgUnion::SetConst(SetConstPy::from({
            let mut set = Set::with_capacity(10);
            set.insert(0);
            set.insert(1);
            set.insert(2);
            set
        }));
        let v1 = model.add_set_var(ob, target, Some("v1"));
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let target = model.get_target(VarUnion::Set(v1));
        assert!(target.is_ok());
        assert_eq!(
            target.unwrap(),
            state::VariableValueUnion::Set(HashSet::from_iter(vec![0, 1, 2])),
        );
        let v = model.get_set_var("v1");
        assert!(v.is_ok());
        assert_eq!(v.unwrap(), v1);
        let result = model.get_object_type_of(ObjectVarUnion::Set(v1));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ob);

        let target = TargetSetArgUnion::SetConst(SetConstPy::from({
            let mut set = Set::with_capacity(10);
            set.insert(3);
            set.insert(4);
            set.insert(5);
            set
        }));
        let v2 = model.add_set_var(ob, target, Some("v2"));
        assert!(v2.is_ok());
        let v2 = v2.unwrap();
        let target = model.get_target(VarUnion::Set(v2));
        assert!(target.is_ok());
        assert_eq!(
            target.unwrap(),
            state::VariableValueUnion::Set(HashSet::from_iter(vec![3, 4, 5])),
        );
        let v = model.get_set_var("v2");
        assert!(v.is_ok());
        assert_eq!(v.unwrap(), v2);
        let result = model.get_object_type_of(ObjectVarUnion::Set(v2));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ob);

        assert_ne!(v1, v2);
    }

    #[test]
    fn get_set_var_err() {
        let model = ModelPy::default();
        assert!(model.get_set_var("v").is_err());
    }

    #[test]
    fn add_set_var_duplicate_err() {
        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let target = TargetSetArgUnion::CreateSetArg(CreateSetArgUnion::List(vec![0, 1, 2]));
        assert!(model.add_set_var(ob, target, Some("v")).is_ok());
        let snapshot = model.clone();
        let target = TargetSetArgUnion::CreateSetArg(CreateSetArgUnion::Set({
            let mut set = HashSet::default();
            set.insert(3);
            set.insert(4);
            set.insert(5);
            set
        }));
        assert!(model.add_set_var(ob, target, Some("v")).is_err());
        assert_eq!(model, snapshot);
    }

    #[test]
    fn get_set_target_err() {
        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let target = TargetSetArgUnion::CreateSetArg(CreateSetArgUnion::List(vec![0, 1, 2]));
        let v = model.add_set_var(ob, target, None);
        assert!(v.is_ok());
        let v = v.unwrap();
        let model = ModelPy::default();
        assert!(model.get_target(VarUnion::Set(v)).is_err());
    }

    #[test]
    fn set_set_target_from_list_ok() {
        pyo3::prepare_freethreaded_python();

        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let target = TargetSetArgUnion::CreateSetArg(CreateSetArgUnion::List(vec![0, 1, 2]));
        let v = model.add_set_var(ob, target, None);
        assert!(v.is_ok());
        let v = v.unwrap();
        let result = Python::with_gil(|py| {
            let target = vec![3usize, 4usize, 5usize].into_py(py);
            model.set_target(VarUnion::Set(v), target.as_ref(py))
        });
        assert!(result.is_ok());
        let target = model.get_target(VarUnion::Set(v));
        assert!(target.is_ok());
        assert_eq!(
            target.unwrap(),
            state::VariableValueUnion::Set(HashSet::from_iter(vec![3, 4, 5])),
        );
    }

    #[test]
    fn set_set_target_from_list_err() {
        pyo3::prepare_freethreaded_python();

        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let target = TargetSetArgUnion::CreateSetArg(CreateSetArgUnion::List(vec![0, 1, 2]));
        let v = model.add_set_var(ob, target, None);
        assert!(v.is_ok());
        let v = v.unwrap();
        let snapshot = model.clone();
        let result = Python::with_gil(|py| {
            let target = vec![3usize, 4usize, 5usize, 10usize].into_py(py);
            model.set_target(VarUnion::Set(v), target.as_ref(py))
        });
        assert!(result.is_err());
        assert_eq!(model, snapshot);
    }

    #[test]
    fn set_set_target_from_set_ok() {
        pyo3::prepare_freethreaded_python();

        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let target = TargetSetArgUnion::CreateSetArg(CreateSetArgUnion::List(vec![0, 1, 2]));
        let v = model.add_set_var(ob, target, None);
        assert!(v.is_ok());
        let v = v.unwrap();
        let result = Python::with_gil(|py| {
            let target = {
                let mut set = HashSet::<Element>::default();
                set.insert(3);
                set.insert(4);
                set.insert(5);
                set
            }
            .into_py(py);
            model.set_target(VarUnion::Set(v), target.as_ref(py))
        });
        assert!(result.is_ok());
        let target = model.get_target(VarUnion::Set(v));
        assert!(target.is_ok());
        assert_eq!(
            target.unwrap(),
            state::VariableValueUnion::Set(HashSet::from_iter(vec![3, 4, 5])),
        );
    }

    #[test]
    fn set_set_target_from_set_err() {
        pyo3::prepare_freethreaded_python();

        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let target = TargetSetArgUnion::CreateSetArg(CreateSetArgUnion::List(vec![0, 1, 2]));
        let v = model.add_set_var(ob, target, None);
        assert!(v.is_ok());
        let v = v.unwrap();
        let snapshot = model.clone();
        let result = Python::with_gil(|py| {
            let target = {
                let mut set = HashSet::<Element>::default();
                set.insert(3);
                set.insert(4);
                set.insert(5);
                set.insert(10);
                set
            }
            .into_py(py);
            model.set_target(VarUnion::Set(v), target.as_ref(py))
        });
        assert!(result.is_err());
        assert_eq!(model, snapshot);
    }

    #[test]
    fn set_set_target_from_set_const_ok() {
        pyo3::prepare_freethreaded_python();

        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let target = TargetSetArgUnion::CreateSetArg(CreateSetArgUnion::List(vec![0, 1, 2]));
        let v = model.add_set_var(ob, target, None);
        assert!(v.is_ok());
        let v = v.unwrap();
        let result = Python::with_gil(|py| {
            let target = SetConstPy::from({
                let mut set = Set::with_capacity(10);
                set.insert(3);
                set.insert(4);
                set.insert(5);
                set
            })
            .into_py(py);
            model.set_target(VarUnion::Set(v), target.as_ref(py))
        });
        assert!(result.is_ok());
        let target = model.get_target(VarUnion::Set(v));
        assert!(target.is_ok());
        assert_eq!(
            target.unwrap(),
            state::VariableValueUnion::Set(HashSet::from_iter(vec![3, 4, 5])),
        );
    }

    #[test]
    fn set_set_target_extract_err() {
        pyo3::prepare_freethreaded_python();

        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let target = TargetSetArgUnion::CreateSetArg(CreateSetArgUnion::List(vec![0, 1, 2]));
        let v = model.add_set_var(ob, target, None);
        assert!(v.is_ok());
        let v = v.unwrap();
        let snapshot = model.clone();
        let result = Python::with_gil(|py| {
            let target = 1usize.into_py(py);
            model.set_target(VarUnion::Set(v), target.as_ref(py))
        });
        assert!(result.is_err());
        assert_eq!(model, snapshot);
    }

    #[test]
    fn set_set_target_no_variable_err() {
        pyo3::prepare_freethreaded_python();

        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let target = TargetSetArgUnion::CreateSetArg(CreateSetArgUnion::List(vec![0, 1, 2]));
        let v = model.add_set_var(ob, target, None);
        assert!(v.is_ok());
        let v = v.unwrap();
        let mut model = ModelPy::default();
        let snapshot = model.clone();
        let result = Python::with_gil(|py| {
            let target = vec![3usize, 4usize, 5usize].into_py(py);
            model.set_target(VarUnion::Set(v), target.as_ref(py))
        });
        assert!(result.is_err());
        assert_eq!(model, snapshot);
    }

    #[test]
    fn get_object_type_of_set_var_err() {
        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let target = TargetSetArgUnion::CreateSetArg(CreateSetArgUnion::List(vec![0, 1, 2]));
        let v = model.add_set_var(ob, target, None);
        assert!(v.is_ok());
        let v = v.unwrap();
        let model = ModelPy::default();
        let result = model.get_object_type_of(ObjectVarUnion::Set(v));
        assert!(result.is_err());
    }

    #[test]
    fn add_and_get_int_var_ok() {
        let mut model = ModelPy::default();

        let v1 = model.add_int_var(0, None);
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let target = model.get_target(VarUnion::Int(v1));
        assert!(target.is_ok());
        assert_eq!(target.unwrap(), state::VariableValueUnion::Int(0));

        let v2 = model.add_int_var(1, None);
        assert!(v2.is_ok());
        let v2 = v2.unwrap();
        let target = model.get_target(VarUnion::Int(v2));
        assert!(target.is_ok());
        assert_eq!(target.unwrap(), state::VariableValueUnion::Int(1));

        assert_ne!(v1, v2);
    }

    #[test]
    fn add_and_get_int_var_with_name_ok() {
        let mut model = ModelPy::default();

        let v1 = model.add_int_var(0, Some("v1"));
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let target = model.get_target(VarUnion::Int(v1));
        assert!(target.is_ok());
        assert_eq!(target.unwrap(), state::VariableValueUnion::Int(0));
        let v = model.get_int_var("v1");
        assert!(v.is_ok());
        assert_eq!(v.unwrap(), v1);

        let v2 = model.add_int_var(1, Some("v2"));
        assert!(v2.is_ok());
        let v2 = v2.unwrap();
        let target = model.get_target(VarUnion::Int(v2));
        assert!(target.is_ok());
        assert_eq!(target.unwrap(), state::VariableValueUnion::Int(1));
        let v = model.get_int_var("v2");
        assert!(v.is_ok());
        assert_eq!(v.unwrap(), v2);

        assert_ne!(v1, v2);
    }

    #[test]
    fn get_int_var_err() {
        let model = ModelPy::default();
        assert!(model.get_int_var("v").is_err());
    }

    #[test]
    fn add_int_var_err() {
        let mut model = ModelPy::default();

        let v = model.add_int_var(0, Some("v"));
        assert!(v.is_ok());
        let snapshot = model.clone();
        let v = model.add_int_var(1, Some("v"));
        assert!(v.is_err());
        assert_eq!(model, snapshot);
    }

    #[test]
    fn get_int_target_err() {
        let mut model = ModelPy::default();
        let v = model.add_int_var(0, None);
        assert!(v.is_ok());
        let v = v.unwrap();

        let model = ModelPy::default();
        assert!(model.get_target(VarUnion::Int(v)).is_err());
    }

    #[test]
    fn set_int_target_ok() {
        pyo3::prepare_freethreaded_python();

        let mut model = ModelPy::default();
        let v = model.add_int_var(0, None);
        assert!(v.is_ok());
        let v = v.unwrap();
        let result = Python::with_gil(|py| {
            let target = 1i32.into_py(py);
            model.set_target(VarUnion::Int(v), target.as_ref(py))
        });
        assert!(result.is_ok());
        let target = model.get_target(VarUnion::Int(v));
        assert!(target.is_ok());
        assert_eq!(target.unwrap(), state::VariableValueUnion::Int(1));
    }

    #[test]
    fn set_int_target_extract_err() {
        pyo3::prepare_freethreaded_python();

        let mut model = ModelPy::default();
        let v = model.add_int_var(0, None);
        assert!(v.is_ok());
        let v = v.unwrap();
        let snapshot = model.clone();
        let result = Python::with_gil(|py| {
            let target = 1.5f64.into_py(py);
            model.set_target(VarUnion::Int(v), target.as_ref(py))
        });
        assert!(result.is_err());
        assert_eq!(model, snapshot);
    }

    #[test]
    fn set_int_target_no_variable_err() {
        pyo3::prepare_freethreaded_python();

        let mut model = ModelPy::default();
        let v = model.add_int_var(0, None);
        assert!(v.is_ok());
        let v = v.unwrap();
        let mut model = ModelPy::default();
        let snapshot = model.clone();
        let result = Python::with_gil(|py| {
            let target = 1i32.into_py(py);
            model.set_target(VarUnion::Int(v), target.as_ref(py))
        });
        assert!(result.is_err());
        assert_eq!(model, snapshot);
    }

    #[test]
    fn add_and_get_int_resource_var_ok() {
        let mut model = ModelPy::default();

        let v1 = model.add_int_resource_var(0, false, None);
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let target = model.get_target(VarUnion::IntResource(v1));
        assert!(target.is_ok());
        assert_eq!(target.unwrap(), state::VariableValueUnion::Int(0));
        let less_is_better = model.get_preference(ResourceVarUnion::Int(v1));
        assert!(less_is_better.is_ok());
        assert!(!less_is_better.unwrap());

        let v2 = model.add_int_resource_var(1, true, None);
        assert!(v2.is_ok());
        let v2 = v2.unwrap();
        let target = model.get_target(VarUnion::IntResource(v2));
        assert!(target.is_ok());
        assert_eq!(target.unwrap(), state::VariableValueUnion::Int(1));
        let less_is_better = model.get_preference(ResourceVarUnion::Int(v2));
        assert!(less_is_better.is_ok());
        assert!(less_is_better.unwrap());

        assert_ne!(v1, v2);
    }

    #[test]
    fn add_and_get_int_resource_var_with_name_ok() {
        let mut model = ModelPy::default();

        let v1 = model.add_int_resource_var(0, false, Some("v1"));
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let target = model.get_target(VarUnion::IntResource(v1));
        assert!(target.is_ok());
        assert_eq!(target.unwrap(), state::VariableValueUnion::Int(0));
        let v = model.get_int_resource_var("v1");
        assert!(v.is_ok());
        assert_eq!(v.unwrap(), v1);
        let less_is_better = model.get_preference(ResourceVarUnion::Int(v1));
        assert!(less_is_better.is_ok());
        assert!(!less_is_better.unwrap());

        let v2 = model.add_int_resource_var(1, true, Some("v2"));
        assert!(v2.is_ok());
        let v2 = v2.unwrap();
        let target = model.get_target(VarUnion::IntResource(v2));
        assert!(target.is_ok());
        assert_eq!(target.unwrap(), state::VariableValueUnion::Int(1));
        let v = model.get_int_resource_var("v2");
        assert!(v.is_ok());
        assert_eq!(v.unwrap(), v2);
        let less_is_better = model.get_preference(ResourceVarUnion::Int(v2));
        assert!(less_is_better.is_ok());
        assert!(less_is_better.unwrap());

        assert_ne!(v1, v2);
    }

    #[test]
    fn get_int_resource_var_err() {
        let model = ModelPy::default();
        assert!(model.get_int_resource_var("v").is_err());
    }

    #[test]
    fn add_int_resource_var_err() {
        let mut model = ModelPy::default();

        let v = model.add_int_resource_var(0, false, Some("v"));
        assert!(v.is_ok());
        let snapshot = model.clone();
        let v = model.add_int_resource_var(1, true, Some("v"));
        assert!(v.is_err());
        assert_eq!(model, snapshot);
    }

    #[test]
    fn get_int_resource_target_err() {
        let mut model = ModelPy::default();
        let v = model.add_int_resource_var(0, false, None);
        assert!(v.is_ok());
        let v = v.unwrap();

        let model = ModelPy::default();
        assert!(model.get_target(VarUnion::IntResource(v)).is_err());
    }

    #[test]
    fn set_int_resource_target_ok() {
        pyo3::prepare_freethreaded_python();

        let mut model = ModelPy::default();
        let v = model.add_int_resource_var(0, false, None);
        assert!(v.is_ok());
        let v = v.unwrap();
        let result = Python::with_gil(|py| {
            let target = 1i32.into_py(py);
            model.set_target(VarUnion::IntResource(v), target.as_ref(py))
        });
        assert!(result.is_ok());
        let target = model.get_target(VarUnion::IntResource(v));
        assert!(target.is_ok());
        assert_eq!(target.unwrap(), state::VariableValueUnion::Int(1));
    }

    #[test]
    fn set_int_resource_target_extract_err() {
        pyo3::prepare_freethreaded_python();

        let mut model = ModelPy::default();
        let v = model.add_int_resource_var(0, false, None);
        assert!(v.is_ok());
        let v = v.unwrap();
        let snapshot = model.clone();
        let result = Python::with_gil(|py| {
            let target = 1.5f64.into_py(py);
            model.set_target(VarUnion::IntResource(v), target.as_ref(py))
        });
        assert!(result.is_err());
        assert_eq!(model, snapshot);
    }

    #[test]
    fn set_int_resource_target_no_variable_err() {
        pyo3::prepare_freethreaded_python();

        let mut model = ModelPy::default();
        let v = model.add_int_resource_var(0, false, None);
        assert!(v.is_ok());
        let v = v.unwrap();
        let mut model = ModelPy::default();
        let snapshot = model.clone();
        let result = Python::with_gil(|py| {
            let target = 1i32.into_py(py);
            model.set_target(VarUnion::IntResource(v), target.as_ref(py))
        });
        assert!(result.is_err());
        assert_eq!(model, snapshot);
    }

    #[test]
    fn get_int_preference_err() {
        let mut model = ModelPy::default();
        let v = model.add_int_resource_var(0, false, None);
        assert!(v.is_ok());
        let v = v.unwrap();

        let model = ModelPy::default();
        assert!(model.get_preference(ResourceVarUnion::Int(v)).is_err());
    }

    #[test]
    fn set_int_preference_ok() {
        let mut model = ModelPy::default();
        let v = model.add_int_resource_var(0, false, None);
        assert!(v.is_ok());
        let v = v.unwrap();
        let result = model.set_preference(ResourceVarUnion::Int(v), true);
        assert!(result.is_ok());
        let less_is_better = model.get_preference(ResourceVarUnion::Int(v));
        assert!(less_is_better.is_ok());
        assert!(less_is_better.unwrap());
    }

    #[test]
    fn set_int_preference_err() {
        let mut model = ModelPy::default();
        let v = model.add_int_resource_var(0, false, None);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut model = ModelPy::default();
        let snapshot = model.clone();
        assert!(model
            .set_preference(ResourceVarUnion::Int(v), true)
            .is_err());
        assert_eq!(model, snapshot);
    }

    #[test]
    fn add_and_get_float_var_ok() {
        let mut model = ModelPy::default();

        let v1 = model.add_float_var(0.0, None);
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let target = model.get_target(VarUnion::Float(v1));
        assert!(target.is_ok());
        assert_eq!(target.unwrap(), state::VariableValueUnion::Float(0.0));

        let v2 = model.add_float_var(1.0, None);
        assert!(v2.is_ok());
        let v2 = v2.unwrap();
        let target = model.get_target(VarUnion::Float(v2));
        assert!(target.is_ok());
        assert_eq!(target.unwrap(), state::VariableValueUnion::Float(1.0));

        assert_ne!(v1, v2);
    }

    #[test]
    fn add_and_get_float_var_with_name_ok() {
        let mut model = ModelPy::default();

        let v1 = model.add_float_var(0.0, Some("v1"));
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let target = model.get_target(VarUnion::Float(v1));
        assert!(target.is_ok());
        assert_eq!(target.unwrap(), state::VariableValueUnion::Float(0.0));
        let v = model.get_float_var("v1");
        assert!(v.is_ok());
        assert_eq!(v.unwrap(), v1);

        let v2 = model.add_float_var(1.0, Some("v2"));
        assert!(v2.is_ok());
        let v2 = v2.unwrap();
        let target = model.get_target(VarUnion::Float(v2));
        assert!(target.is_ok());
        assert_eq!(target.unwrap(), state::VariableValueUnion::Float(1.0));
        let v = model.get_float_var("v2");
        assert!(v.is_ok());
        assert_eq!(v.unwrap(), v2);

        assert_ne!(v1, v2);
    }

    #[test]
    fn get_float_var_err() {
        let model = ModelPy::default();
        assert!(model.get_float_var("v").is_err());
    }

    #[test]
    fn add_float_var_err() {
        let mut model = ModelPy::default();

        let v = model.add_float_var(0.0, Some("v"));
        assert!(v.is_ok());
        let snapshot = model.clone();
        let v = model.add_float_var(1.0, Some("v"));
        assert!(v.is_err());
        assert_eq!(model, snapshot);
    }

    #[test]
    fn get_float_target_err() {
        let mut model = ModelPy::default();
        let v = model.add_float_var(0.0, None);
        assert!(v.is_ok());
        let v = v.unwrap();

        let model = ModelPy::default();
        assert!(model.get_target(VarUnion::Float(v)).is_err());
    }

    #[test]
    fn set_float_target_ok() {
        pyo3::prepare_freethreaded_python();

        let mut model = ModelPy::default();
        let v = model.add_float_var(0.0, None);
        assert!(v.is_ok());
        let v = v.unwrap();
        let result = Python::with_gil(|py| {
            let target = 1i32.into_py(py);
            model.set_target(VarUnion::Float(v), target.as_ref(py))
        });
        assert!(result.is_ok());
        let target = model.get_target(VarUnion::Float(v));
        assert!(target.is_ok());
        assert_eq!(target.unwrap(), state::VariableValueUnion::Float(1.0));
    }

    #[test]
    fn set_float_target_extract_err() {
        pyo3::prepare_freethreaded_python();

        let mut model = ModelPy::default();
        let v = model.add_float_var(0.0, None);
        assert!(v.is_ok());
        let v = v.unwrap();
        let snapshot = model.clone();
        let result = Python::with_gil(|py| {
            let target: Py<PyAny> = (0, 1).into_py(py);
            model.set_target(VarUnion::Float(v), target.as_ref(py))
        });
        assert!(result.is_err());
        assert_eq!(model, snapshot);
    }

    #[test]
    fn set_float_target_no_variable_err() {
        pyo3::prepare_freethreaded_python();

        let mut model = ModelPy::default();
        let v = model.add_float_var(0.0, None);
        assert!(v.is_ok());
        let v = v.unwrap();
        let mut model = ModelPy::default();
        let snapshot = model.clone();
        let result = Python::with_gil(|py| {
            let target = 1i32.into_py(py);
            model.set_target(VarUnion::Float(v), target.as_ref(py))
        });
        assert!(result.is_err());
        assert_eq!(model, snapshot);
    }

    #[test]
    fn add_and_get_float_resource_var_ok() {
        let mut model = ModelPy::default();

        let v1 = model.add_float_resource_var(0.0, false, None);
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let target = model.get_target(VarUnion::FloatResource(v1));
        assert!(target.is_ok());
        assert_eq!(target.unwrap(), state::VariableValueUnion::Float(0.0));
        let less_is_better = model.get_preference(ResourceVarUnion::Float(v1));
        assert!(less_is_better.is_ok());
        assert!(!less_is_better.unwrap());

        let v2 = model.add_float_resource_var(1.0, true, None);
        assert!(v2.is_ok());
        let v2 = v2.unwrap();
        let target = model.get_target(VarUnion::FloatResource(v2));
        assert!(target.is_ok());
        assert_eq!(target.unwrap(), state::VariableValueUnion::Float(1.0));
        let less_is_better = model.get_preference(ResourceVarUnion::Float(v2));
        assert!(less_is_better.is_ok());
        assert!(less_is_better.unwrap());

        assert_ne!(v1, v2);
    }

    #[test]
    fn add_and_get_float_resource_var_with_name_ok() {
        let mut model = ModelPy::default();

        let v1 = model.add_float_resource_var(0.0, false, Some("v1"));
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let target = model.get_target(VarUnion::FloatResource(v1));
        assert!(target.is_ok());
        assert_eq!(target.unwrap(), state::VariableValueUnion::Float(0.0));
        let v = model.get_float_resource_var("v1");
        assert!(v.is_ok());
        assert_eq!(v.unwrap(), v1);
        let less_is_better = model.get_preference(ResourceVarUnion::Float(v1));
        assert!(less_is_better.is_ok());
        assert!(!less_is_better.unwrap());

        let v2 = model.add_float_resource_var(1.0, true, Some("v2"));
        assert!(v2.is_ok());
        let v2 = v2.unwrap();
        let target = model.get_target(VarUnion::FloatResource(v2));
        assert!(target.is_ok());
        assert_eq!(target.unwrap(), state::VariableValueUnion::Float(1.0));
        let v = model.get_float_resource_var("v2");
        assert!(v.is_ok());
        assert_eq!(v.unwrap(), v2);
        let less_is_better = model.get_preference(ResourceVarUnion::Float(v2));
        assert!(less_is_better.is_ok());
        assert!(less_is_better.unwrap());

        assert_ne!(v1, v2);
    }

    #[test]
    fn get_float_resource_var_err() {
        let model = ModelPy::default();
        assert!(model.get_float_resource_var("v").is_err());
    }

    #[test]
    fn add_float_resource_var_err() {
        let mut model = ModelPy::default();

        let v = model.add_float_resource_var(0.0, false, Some("v"));
        assert!(v.is_ok());
        let snapshot = model.clone();
        let v = model.add_float_resource_var(1.0, true, Some("v"));
        assert!(v.is_err());
        assert_eq!(model, snapshot);
    }

    #[test]
    fn get_float_resource_target_err() {
        let mut model = ModelPy::default();
        let v = model.add_float_resource_var(0.0, false, None);
        assert!(v.is_ok());
        let v = v.unwrap();

        let model = ModelPy::default();
        assert!(model.get_target(VarUnion::FloatResource(v)).is_err());
    }

    #[test]
    fn set_float_resource_target_ok() {
        pyo3::prepare_freethreaded_python();

        let mut model = ModelPy::default();
        let v = model.add_float_resource_var(0.0, false, None);
        assert!(v.is_ok());
        let v = v.unwrap();
        let result = Python::with_gil(|py| {
            let target = 1i32.into_py(py);
            model.set_target(VarUnion::FloatResource(v), target.as_ref(py))
        });
        assert!(result.is_ok());
        let target = model.get_target(VarUnion::FloatResource(v));
        assert!(target.is_ok());
        assert_eq!(target.unwrap(), state::VariableValueUnion::Float(1.0));
    }

    #[test]
    fn set_float_resource_target_extract_err() {
        pyo3::prepare_freethreaded_python();

        let mut model = ModelPy::default();
        let v = model.add_float_resource_var(0.0, false, None);
        assert!(v.is_ok());
        let v = v.unwrap();
        let snapshot = model.clone();
        let result = Python::with_gil(|py| {
            let target: Py<PyAny> = (0, 1).into_py(py);
            model.set_target(VarUnion::FloatResource(v), target.as_ref(py))
        });
        assert!(result.is_err());
        assert_eq!(model, snapshot);
    }

    #[test]
    fn set_float_resource_target_no_variable_err() {
        pyo3::prepare_freethreaded_python();

        let mut model = ModelPy::default();
        let v = model.add_float_resource_var(0.0, false, None);
        assert!(v.is_ok());
        let v = v.unwrap();
        let mut model = ModelPy::default();
        let snapshot = model.clone();
        let result = Python::with_gil(|py| {
            let target = 1i32.into_py(py);
            model.set_target(VarUnion::FloatResource(v), target.as_ref(py))
        });
        assert!(result.is_err());
        assert_eq!(model, snapshot);
    }

    #[test]
    fn get_float_preference_err() {
        let mut model = ModelPy::default();
        let v = model.add_float_resource_var(0.0, false, None);
        assert!(v.is_ok());
        let v = v.unwrap();

        let model = ModelPy::default();
        assert!(model.get_preference(ResourceVarUnion::Float(v)).is_err());
    }

    #[test]
    fn set_float_preference_ok() {
        let mut model = ModelPy::default();
        let v = model.add_float_resource_var(0.0, false, None);
        assert!(v.is_ok());
        let v = v.unwrap();
        let result = model.set_preference(ResourceVarUnion::Float(v), true);
        assert!(result.is_ok());
        let less_is_better = model.get_preference(ResourceVarUnion::Float(v));
        assert!(less_is_better.is_ok());
        assert!(less_is_better.unwrap());
    }

    #[test]
    fn set_float_preference_err() {
        let mut model = ModelPy::default();
        let v = model.add_float_resource_var(0.0, false, None);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut model = ModelPy::default();
        let snapshot = model.clone();
        assert!(model
            .set_preference(ResourceVarUnion::Float(v), true)
            .is_err());
        assert_eq!(model, snapshot);
    }

    #[test]
    fn add_constr_ok() {
        let mut model = ModelPy::default();
        let v = model.add_int_var(0, None);
        assert!(v.is_ok());
        let v = v.unwrap();
        let v_id = IntegerVariable::from(v).id();
        let result = model.add_state_constr(ConditionPy::from(Condition::ComparisonI(
            ComparisonOperator::Gt,
            Box::new(IntegerExpression::Variable(v_id)),
            Box::new(IntegerExpression::Variable(0)),
        )));
        assert!(result.is_ok());
        assert_eq!(
            model.0.state_constraints,
            vec![GroundedCondition {
                condition: Condition::ComparisonI(
                    ComparisonOperator::Gt,
                    Box::new(IntegerExpression::Variable(v_id)),
                    Box::new(IntegerExpression::Variable(0))
                ),
                ..Default::default()
            }],
        );
    }

    #[test]
    fn add_constr_err() {
        let mut model = ModelPy::default();
        let v = model.add_int_var(0, None);
        assert!(v.is_ok());
        let v = v.unwrap();
        let v_id = IntegerVariable::from(v).id();
        let mut model = ModelPy::default();
        let snapshot = model.clone();
        let result = model.add_state_constr(ConditionPy::from(Condition::ComparisonI(
            ComparisonOperator::Gt,
            Box::new(IntegerExpression::Variable(v_id)),
            Box::new(IntegerExpression::Variable(0)),
        )));
        assert!(result.is_err());
        assert_eq!(model, snapshot);
    }

    #[test]
    fn add_base_case_ok() {
        let mut model = ModelPy::default();
        let v = model.add_int_var(0, None);
        assert!(v.is_ok());
        let v = v.unwrap();
        let v_id = IntegerVariable::from(v).id();
        let result = model.add_base_case(
            vec![ConditionPy::from(Condition::ComparisonI(
                ComparisonOperator::Gt,
                Box::new(IntegerExpression::Variable(v_id)),
                Box::new(IntegerExpression::Variable(0)),
            ))],
            None,
        );
        assert!(result.is_ok());
        assert_eq!(
            model.0.base_cases,
            vec![BaseCase::from(vec![GroundedCondition {
                condition: Condition::ComparisonI(
                    ComparisonOperator::Gt,
                    Box::new(IntegerExpression::Variable(v_id)),
                    Box::new(IntegerExpression::Variable(0))
                ),
                ..Default::default()
            }])],
        );
    }

    #[test]
    fn add_base_case_err() {
        let mut model = ModelPy::default();
        let v = model.add_int_var(0, None);
        assert!(v.is_ok());
        let v = v.unwrap();
        let v_id = IntegerVariable::from(v).id();
        let mut model = ModelPy::default();
        let snapshot = model.clone();
        let result = model.add_base_case(
            vec![ConditionPy::from(Condition::ComparisonI(
                ComparisonOperator::Gt,
                Box::new(IntegerExpression::Variable(v_id)),
                Box::new(IntegerExpression::Variable(0)),
            ))],
            None,
        );
        assert!(result.is_err());
        assert_eq!(model, snapshot);
    }

    #[test]
    fn set_minimize() {
        let mut model = ModelPy(Model {
            reduce_function: ReduceFunction::Max,
            ..Default::default()
        });
        model.set_maximize(false);
        assert_eq!(
            model.0,
            Model {
                reduce_function: ReduceFunction::Min,
                ..Default::default()
            }
        );
    }

    #[test]
    fn set_maximize() {
        let mut model = ModelPy(Model {
            reduce_function: ReduceFunction::Min,
            ..Default::default()
        });
        model.set_maximize(true);
        assert_eq!(
            model.0,
            Model {
                reduce_function: ReduceFunction::Max,
                ..Default::default()
            }
        );
    }

    #[test]
    fn add_forward_transition_ok() {
        let mut model = ModelPy::default();
        let t = TransitionPy::default();
        let result = model.add_transition(t, false, false);
        assert!(result.is_ok());
        assert_eq!(
            model.0,
            Model {
                forward_transitions: vec![Transition::default()],
                ..Default::default()
            }
        );
    }

    #[test]
    fn add_forward_transition_err() {
        let mut model = ModelPy::new_py(false, false);
        let t = TransitionPy::new_py(
            "t",
            Some(CostUnion::Float(FloatUnion::Expr(FloatExprPy::from(
                ContinuousExpression::Cost,
            )))),
            None,
            None,
        );
        assert!(t.is_ok());
        let t = t.unwrap();
        let snapshot = model.clone();
        let result = model.add_transition(t, false, false);
        assert!(result.is_err());
        assert_eq!(model, snapshot);
    }

    #[test]
    fn add_forward_forced_transition_ok() {
        let mut model = ModelPy::default();
        let t = TransitionPy::default();
        let result = model.add_transition(t, true, false);
        assert!(result.is_ok());
        assert_eq!(
            model.0,
            Model {
                forward_forced_transitions: vec![Transition::default()],
                ..Default::default()
            }
        );
    }

    #[test]
    fn add_forward_forced_transition_err() {
        let mut model = ModelPy::new_py(false, false);
        let t = TransitionPy::new_py(
            "t",
            Some(CostUnion::Float(FloatUnion::Expr(FloatExprPy::from(
                ContinuousExpression::Cost,
            )))),
            None,
            None,
        );
        assert!(t.is_ok());
        let t = t.unwrap();
        let snapshot = model.clone();
        let result = model.add_transition(t, true, false);
        assert!(result.is_err());
        assert_eq!(model, snapshot);
    }

    #[test]
    fn add_backward_transition_ok() {
        let mut model = ModelPy::default();
        let t = TransitionPy::default();
        let result = model.add_transition(t, false, true);
        assert!(result.is_ok());
        assert_eq!(
            model.0,
            Model {
                backward_transitions: vec![Transition::default()],
                ..Default::default()
            }
        );
    }

    #[test]
    fn add_backward_transition_err() {
        let mut model = ModelPy::new_py(false, false);
        let t = TransitionPy::new_py(
            "t",
            Some(CostUnion::Float(FloatUnion::Expr(FloatExprPy::from(
                ContinuousExpression::Cost,
            )))),
            None,
            None,
        );
        assert!(t.is_ok());
        let t = t.unwrap();
        let snapshot = model.clone();
        let result = model.add_transition(t, false, true);
        assert!(result.is_err());
        assert_eq!(model, snapshot);
    }

    #[test]
    fn add_backward_forced_transition_ok() {
        let mut model = ModelPy::default();
        let t = TransitionPy::default();
        let result = model.add_transition(t, true, true);
        assert!(result.is_ok());
        assert_eq!(
            model.0,
            Model {
                backward_forced_transitions: vec![Transition::default()],
                ..Default::default()
            }
        );
    }

    #[test]
    fn add_backward_forced_transition_err() {
        let mut model = ModelPy::new_py(false, false);
        let t = TransitionPy::new_py(
            "t",
            Some(CostUnion::Float(FloatUnion::Expr(FloatExprPy::from(
                ContinuousExpression::Cost,
            )))),
            None,
            None,
        );
        assert!(t.is_ok());
        let t = t.unwrap();
        let snapshot = model.clone();
        let result = model.add_transition(t, true, true);
        assert!(result.is_err());
        assert_eq!(model, snapshot);
    }

    #[test]
    fn add_int_dual_bound_ok() {
        let mut model = ModelPy::new_py(false, false);
        assert!(model
            .add_dual_bound(CostUnion::Int(IntUnion::Const(0)))
            .is_ok());
        assert_eq!(
            model.0.dual_bounds,
            vec![CostExpression::Integer(IntegerExpression::Constant(0))]
        );
    }

    #[test]
    fn add_int_dual_bound_err() {
        let mut model = ModelPy::new_py(false, false);
        assert!(model
            .add_dual_bound(CostUnion::Float(FloatUnion::Const(1.5)))
            .is_err());
        assert_eq!(model.0.dual_bounds, vec![]);
    }

    #[test]
    fn add_float_dual_bound_ok() {
        let mut model = ModelPy::new_py(false, true);
        assert!(model
            .add_dual_bound(CostUnion::Float(FloatUnion::Const(1.5)))
            .is_ok());
        assert_eq!(
            model.0.dual_bounds,
            vec![CostExpression::Continuous(ContinuousExpression::Constant(
                1.5
            ))]
        );
    }

    #[test]
    fn add_float_dual_bound_err() {
        let mut model = ModelPy::new_py(false, false);
        assert!(model
            .add_dual_bound(CostUnion::Float(FloatUnion::Expr(FloatExprPy::from(
                ContinuousExpression::Cost
            ))))
            .is_err());
        assert_eq!(model.0.dual_bounds, vec![]);
    }

    #[test]
    fn add_element_table_1d_ok() {
        let mut model = ModelPy::default();

        let table = ElementTableArgUnion::Table1D(vec![1]);
        let t1 = model.add_element_table(table, None, None);
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        assert!(matches!(t1, ElementTableUnion::Table1D(_)));

        let table = ElementTableArgUnion::Table1D(vec![2]);
        let t2 = model.add_element_table(table, None, None);
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        assert!(matches!(t2, ElementTableUnion::Table1D(_)));

        assert_ne!(t1, t2);
    }

    #[test]
    fn add_element_table_1d_with_name_ok() {
        let mut model = ModelPy::default();

        let table = ElementTableArgUnion::Table1D(vec![1]);
        let t1 = model.add_element_table(table, None, Some("t1"));
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        assert!(matches!(t1, ElementTableUnion::Table1D(_)));

        let table = ElementTableArgUnion::Table1D(vec![2]);
        let t2 = model.add_element_table(table, None, Some("t2"));
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        assert!(matches!(t2, ElementTableUnion::Table1D(_)));

        assert_ne!(t1, t2);
    }

    #[test]
    fn add_element_table_1d_err() {
        let mut model = ModelPy::default();

        let table = ElementTableArgUnion::Table1D(vec![1]);
        let t = model.add_element_table(table, None, Some("t"));
        assert!(t.is_ok());
        let table = ElementTableArgUnion::Table1D(vec![1]);
        let t = model.add_element_table(table, None, Some("t"));
        assert!(t.is_err());
    }

    // #[test]
    // fn set_element_table_1d_item_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = ElementTableArgUnion::Table1D(vec![1]);
    //     let t = model.add_element_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let result = Python::with_gil(|py| {
    //         let value = 2usize.into_py(py);
    //         let index = TableIndexUnion::Table1D(0);
    //         model.set_table_item(TableUnion::Element(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_ok());
    // }

    // #[test]
    // fn set_element_table_1d_item_extract_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = ElementTableArgUnion::Table1D(vec![1]);
    //     let t = model.add_element_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = 1.5f64.into_py(py);
    //         let index = TableIndexUnion::Table1D(0);
    //         model.set_table_item(TableUnion::Element(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn set_element_table_1d_item_dimension_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = ElementTableArgUnion::Table1D(vec![1]);
    //     let t = model.add_element_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = 2usize.into_py(py);
    //         let index = TableIndexUnion::Table2D((0, 0));
    //         model.set_table_item(TableUnion::Element(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn set_element_table_1d_no_variable_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = ElementTableArgUnion::Table1D(vec![1]);
    //     let t = model.add_element_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let mut model = ModelPy::default();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = 2usize.into_py(py);
    //         let index = TableIndexUnion::Table1D(0);
    //         model.set_table_item(TableUnion::Element(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_element_table_1d_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = ElementTableArgUnion::Table1D(vec![1]);
    //     let t = model.add_element_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let result = Python::with_gil(|py| {
    //         let value: Py<PyAny> = vec![2usize].into_py(py);
    //         model.update_table(TableUnion::Element(t), value.as_ref(py), None)
    //     });
    //     assert!(result.is_ok());
    // }

    // #[test]
    // fn update_element_table_1d_value_extract_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = ElementTableArgUnion::Table1D(vec![1]);
    //     let t = model.add_element_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value: Py<PyAny> = vec![1.5f64].into_py(py);
    //         model.update_table(TableUnion::Element(t), value.as_ref(py), None)
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_element_table_1d_value_dimension_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = ElementTableArgUnion::Table1D(vec![1]);
    //     let t = model.add_element_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value: Py<PyAny> = vec![vec![2usize]].into_py(py);
    //         model.update_table(TableUnion::Element(t), value.as_ref(py), None)
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_element_table_1d_value_no_table_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = ElementTableArgUnion::Table1D(vec![1]);
    //     let t = model.add_element_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let mut model = ModelPy::default();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value: Py<PyAny> = vec![2usize].into_py(py);
    //         model.update_table(TableUnion::Element(t), value.as_ref(py), None)
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    #[test]
    fn add_element_table_2d_ok() {
        let mut model = ModelPy::default();

        let table = ElementTableArgUnion::Table2D(vec![vec![1]]);
        let t1 = model.add_element_table(table, None, None);
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        assert!(matches!(t1, ElementTableUnion::Table2D(_)));

        let table = ElementTableArgUnion::Table2D(vec![vec![2]]);
        let t2 = model.add_element_table(table, None, None);
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        assert!(matches!(t2, ElementTableUnion::Table2D(_)));

        assert_ne!(t1, t2);
    }

    #[test]
    fn add_element_table_2d_with_name_ok() {
        let mut model = ModelPy::default();

        let table = ElementTableArgUnion::Table2D(vec![vec![1]]);
        let t1 = model.add_element_table(table, None, Some("t1"));
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        assert!(matches!(t1, ElementTableUnion::Table2D(_)));

        let table = ElementTableArgUnion::Table2D(vec![vec![2]]);
        let t2 = model.add_element_table(table, None, Some("t2"));
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        assert!(matches!(t2, ElementTableUnion::Table2D(_)));

        assert_ne!(t1, t2);
    }

    #[test]
    fn add_element_table_2d_err() {
        let mut model = ModelPy::default();

        let table = ElementTableArgUnion::Table2D(vec![vec![1]]);
        let t = model.add_element_table(table, None, Some("t"));
        assert!(t.is_ok());
        let table = ElementTableArgUnion::Table2D(vec![vec![1]]);
        let t = model.add_element_table(table, None, Some("t"));
        assert!(t.is_err());
    }

    // #[test]
    // fn set_element_table_2d_item_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = ElementTableArgUnion::Table2D(vec![vec![1]]);
    //     let t = model.add_element_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let result = Python::with_gil(|py| {
    //         let value = 2usize.into_py(py);
    //         let index = TableIndexUnion::Table2D((0, 0));
    //         model.set_table_item(TableUnion::Element(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_ok());
    // }

    // #[test]
    // fn set_element_table_2d_item_extract_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = ElementTableArgUnion::Table2D(vec![vec![1]]);
    //     let t = model.add_element_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = 1.5f64.into_py(py);
    //         let index = TableIndexUnion::Table2D((0, 0));
    //         model.set_table_item(TableUnion::Element(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn set_element_table_2d_item_dimension_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = ElementTableArgUnion::Table2D(vec![vec![1]]);
    //     let t = model.add_element_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = 2usize.into_py(py);
    //         let index = TableIndexUnion::Table3D((0, 0, 0));
    //         model.set_table_item(TableUnion::Element(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn set_element_table_2d_no_variable_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = ElementTableArgUnion::Table2D(vec![vec![1]]);
    //     let t = model.add_element_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let mut model = ModelPy::default();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = 2usize.into_py(py);
    //         let index = TableIndexUnion::Table2D((0, 0));
    //         model.set_table_item(TableUnion::Element(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_element_table_2d_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = ElementTableArgUnion::Table2D(vec![vec![1]]);
    //     let t = model.add_element_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let result = Python::with_gil(|py| {
    //         let value: Py<PyAny> = vec![vec![2usize]].into_py(py);
    //         model.update_table(TableUnion::Element(t), value.as_ref(py), None)
    //     });
    //     assert!(result.is_ok());
    // }

    // #[test]
    // fn update_element_table_2d_value_extract_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = ElementTableArgUnion::Table2D(vec![vec![1]]);
    //     let t = model.add_element_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value: Py<PyAny> = vec![vec![1.5f64]].into_py(py);
    //         model.update_table(TableUnion::Element(t), value.as_ref(py), None)
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_element_table_2d_value_dimension_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = ElementTableArgUnion::Table2D(vec![vec![1]]);
    //     let t = model.add_element_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value: Py<PyAny> = vec![vec![vec![2usize]]].into_py(py);
    //         model.update_table(TableUnion::Element(t), value.as_ref(py), None)
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_element_table_2d_value_no_table_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = ElementTableArgUnion::Table2D(vec![vec![1]]);
    //     let t = model.add_element_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let mut model = ModelPy::default();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value: Py<PyAny> = vec![vec![2usize]].into_py(py);
    //         model.update_table(TableUnion::Element(t), value.as_ref(py), None)
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    #[test]
    fn add_element_table_3d_ok() {
        let mut model = ModelPy::default();

        let table = ElementTableArgUnion::Table3D(vec![vec![vec![1]]]);
        let t1 = model.add_element_table(table, None, None);
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        assert!(matches!(t1, ElementTableUnion::Table3D(_)));

        let table = ElementTableArgUnion::Table3D(vec![vec![vec![2]]]);
        let t2 = model.add_element_table(table, None, None);
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        assert!(matches!(t2, ElementTableUnion::Table3D(_)));

        assert_ne!(t1, t2);
    }

    #[test]
    fn add_element_table_3d_with_name_ok() {
        let mut model = ModelPy::default();

        let table = ElementTableArgUnion::Table3D(vec![vec![vec![1]]]);
        let t1 = model.add_element_table(table, None, Some("t1"));
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        assert!(matches!(t1, ElementTableUnion::Table3D(_)));

        let table = ElementTableArgUnion::Table3D(vec![vec![vec![2]]]);
        let t2 = model.add_element_table(table, None, Some("t2"));
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        assert!(matches!(t2, ElementTableUnion::Table3D(_)));

        assert_ne!(t1, t2);
    }

    #[test]
    fn add_element_table_3d_err() {
        let mut model = ModelPy::default();

        let table = ElementTableArgUnion::Table3D(vec![vec![vec![1]]]);
        let t = model.add_element_table(table, None, Some("t"));
        assert!(t.is_ok());
        let table = ElementTableArgUnion::Table3D(vec![vec![vec![1]]]);
        let t = model.add_element_table(table, None, Some("t"));
        assert!(t.is_err());
    }

    // #[test]
    // fn set_element_table_3d_item_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = ElementTableArgUnion::Table3D(vec![vec![vec![1]]]);
    //     let t = model.add_element_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let result = Python::with_gil(|py| {
    //         let value = 2usize.into_py(py);
    //         let index = TableIndexUnion::Table3D((0, 0, 0));
    //         model.set_table_item(TableUnion::Element(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_ok());
    // }

    // #[test]
    // fn set_element_table_3d_item_extract_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = ElementTableArgUnion::Table3D(vec![vec![vec![1]]]);
    //     let t = model.add_element_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = 1.5f64.into_py(py);
    //         let index = TableIndexUnion::Table3D((0, 0, 0));
    //         model.set_table_item(TableUnion::Element(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn set_element_table_3d_item_dimension_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = ElementTableArgUnion::Table3D(vec![vec![vec![1]]]);
    //     let t = model.add_element_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = 2usize.into_py(py);
    //         let index = TableIndexUnion::Table2D((0, 0));
    //         model.set_table_item(TableUnion::Element(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn set_element_table_3d_no_variable_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = ElementTableArgUnion::Table3D(vec![vec![vec![1]]]);
    //     let t = model.add_element_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let mut model = ModelPy::default();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = 2usize.into_py(py);
    //         let index = TableIndexUnion::Table3D((0, 0, 0));
    //         model.set_table_item(TableUnion::Element(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_element_table_3d_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = ElementTableArgUnion::Table3D(vec![vec![vec![1]]]);
    //     let t = model.add_element_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let result = Python::with_gil(|py| {
    //         let value: Py<PyAny> = vec![vec![vec![2usize]]].into_py(py);
    //         model.update_table(TableUnion::Element(t), value.as_ref(py), None)
    //     });
    //     assert!(result.is_ok());
    // }

    // #[test]
    // fn update_element_table_3d_value_extract_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = ElementTableArgUnion::Table3D(vec![vec![vec![1]]]);
    //     let t = model.add_element_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value: Py<PyAny> = vec![vec![vec![1.5f64]]].into_py(py);
    //         model.update_table(TableUnion::Element(t), value.as_ref(py), None)
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_element_table_3d_value_dimension_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = ElementTableArgUnion::Table3D(vec![vec![vec![1]]]);
    //     let t = model.add_element_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value: Py<PyAny> = vec![vec![2usize]].into_py(py);
    //         model.update_table(TableUnion::Element(t), value.as_ref(py), None)
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_element_table_3d_value_no_table_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = ElementTableArgUnion::Table3D(vec![vec![vec![1]]]);
    //     let t = model.add_element_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let mut model = ModelPy::default();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value: Py<PyAny> = vec![vec![vec![2usize]]].into_py(py);
    //         model.update_table(TableUnion::Element(t), value.as_ref(py), None)
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    #[test]
    fn add_element_table_ok() {
        let mut model = ModelPy::default();

        let table = ElementTableArgUnion::Table(FxHashMap::default());
        let t1 = model.add_element_table(table, Some(1usize), None);
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        assert!(matches!(t1, ElementTableUnion::Table(_)));

        let table = ElementTableArgUnion::Table(FxHashMap::default());
        let t2 = model.add_element_table(table, Some(2usize), None);
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        assert!(matches!(t2, ElementTableUnion::Table(_)));

        assert_ne!(t1, t2);
    }

    #[test]
    fn add_element_table_with_name_ok() {
        let mut model = ModelPy::default();

        let table = ElementTableArgUnion::Table(FxHashMap::default());
        let t1 = model.add_element_table(table, Some(1usize), Some("t1"));
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        assert!(matches!(t1, ElementTableUnion::Table(_)));

        let table = ElementTableArgUnion::Table(FxHashMap::default());
        let t2 = model.add_element_table(table, Some(2usize), Some("t2"));
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        assert!(matches!(t1, ElementTableUnion::Table(_)));

        assert_ne!(t1, t2);
    }

    #[test]
    fn add_element_table_duplicate_err() {
        let mut model = ModelPy::default();

        let table = ElementTableArgUnion::Table(FxHashMap::default());
        let t = model.add_element_table(table, Some(1usize), Some("t"));
        assert!(t.is_ok());
        let table = ElementTableArgUnion::Table(FxHashMap::default());
        let t = model.add_element_table(table, Some(1usize), Some("t"));
        assert!(t.is_err());
    }

    #[test]
    fn add_element_table_no_default_err() {
        let mut model = ModelPy::default();

        let table = ElementTableArgUnion::Table(FxHashMap::default());
        let t = model.add_element_table(table, None, None);
        assert!(t.is_err());
    }

    // #[test]
    // fn set_element_table_item_1d_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = ElementTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_element_table(table, Some(1usize), None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let result = Python::with_gil(|py| {
    //         let value = 2usize.into_py(py);
    //         let index = TableIndexUnion::Table1D(0);
    //         model.set_table_item(TableUnion::Element(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_ok());
    //     let t = match t {
    //         ElementTableUnion::Table(t) => t,
    //         _ => panic!("expected ElementTableUnion::Table but `{:?}`", t),
    //     };
    //     let t_id = TableHandle::<Element>::from(t).id();
    //     assert_eq!(
    //         *model.0.table_registry.element_tables.tables[t_id].get(&[0]),
    //         2
    //     );
    // }

    // #[test]
    // fn set_element_table_item_2d_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = ElementTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_element_table(table, Some(1usize), None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let result = Python::with_gil(|py| {
    //         let value = 2usize.into_py(py);
    //         let index = TableIndexUnion::Table2D((0, 0));
    //         model.set_table_item(TableUnion::Element(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_ok());
    //     let t = match t {
    //         ElementTableUnion::Table(t) => t,
    //         _ => panic!("expected ElementTableUnion::Table but `{:?}`", t),
    //     };
    //     let t_id = TableHandle::<Element>::from(t).id();
    //     assert_eq!(
    //         *model.0.table_registry.element_tables.tables[t_id].get(&[0, 0]),
    //         2
    //     );
    // }

    // #[test]
    // fn set_element_table_item_3d_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = ElementTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_element_table(table, Some(1usize), None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let result = Python::with_gil(|py| {
    //         let value = 2usize.into_py(py);
    //         let index = TableIndexUnion::Table3D((0, 0, 0));
    //         model.set_table_item(TableUnion::Element(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_ok());
    //     let t = match t {
    //         ElementTableUnion::Table(t) => t,
    //         _ => panic!("expected ElementTableUnion::Table but `{:?}`", t),
    //     };
    //     let t_id = TableHandle::<Element>::from(t).id();
    //     assert_eq!(
    //         *model.0.table_registry.element_tables.tables[t_id].get(&[0, 0, 0]),
    //         2
    //     );
    // }

    // #[test]
    // fn set_element_table_item_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = ElementTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_element_table(table, Some(1usize), None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let result = Python::with_gil(|py| {
    //         let value = 2usize.into_py(py);
    //         let index = TableIndexUnion::Table(vec![0, 0, 0, 0]);
    //         model.set_table_item(TableUnion::Element(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_ok());
    // }

    // #[test]
    // fn set_element_table_item_extract_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = ElementTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_element_table(table, Some(1usize), None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = 1.5f64.into_py(py);
    //         let index = TableIndexUnion::Table(vec![0, 0, 0, 0]);
    //         model.set_table_item(TableUnion::Element(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn set_element_table_item_no_table_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = ElementTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_element_table(table, Some(1usize), None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let mut model = ModelPy::default();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = 2usize.into_py(py);
    //         let index = TableIndexUnion::Table(vec![0, 0, 0, 0]);
    //         model.set_table_item(TableUnion::Element(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn set_element_default_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = ElementTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_element_table(table, Some(1usize), None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let t = match t {
    //         ElementTableUnion::Table(t) => t,
    //         _ => panic!("expected ElementTableUnion::Table but `{:?}`", t),
    //     };
    //     let result = Python::with_gil(|py| {
    //         let value = 2usize.into_py(py);
    //         model.set_default(SetDefaultArgUnion::Element(t), value.as_ref(py))
    //     });
    //     assert!(result.is_ok());
    // }

    // #[test]
    // fn set_element_default_extract_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = ElementTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_element_table(table, Some(1usize), None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let t = match t {
    //         ElementTableUnion::Table(t) => t,
    //         _ => panic!("expected ElementTableUnion::Table but `{:?}`", t),
    //     };
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = 1.5f64.into_py(py);
    //         model.set_default(SetDefaultArgUnion::Element(t), value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn set_element_default_no_table_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = ElementTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_element_table(table, Some(1usize), None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let t = match t {
    //         ElementTableUnion::Table(t) => t,
    //         _ => panic!("expected ElementTableUnion::Table but `{:?}`", t),
    //     };
    //     let mut model = ModelPy::default();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = 1.5f64.into_py(py);
    //         model.set_default(SetDefaultArgUnion::Element(t), value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_element_table_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = ElementTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_element_table(table, Some(1usize), None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let result = Python::with_gil(|py| {
    //         let value = {
    //             let mut map = FxHashMap::default();
    //             map.insert((0, 0, 0, 0), 2);
    //             map
    //         }
    //         .into_py(py);
    //         let default = 3usize.into_py(py);
    //         model.update_table(
    //             TableUnion::Element(t),
    //             value.as_ref(py),
    //             Some(default.as_ref(py)),
    //         )
    //     });
    //     assert!(result.is_ok());
    // }

    // #[test]
    // fn update_element_table_no_default_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = ElementTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_element_table(table, Some(1usize), None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = {
    //             let mut map = FxHashMap::default();
    //             map.insert((0, 0, 0, 0), 2);
    //             map
    //         }
    //         .into_py(py);
    //         model.update_table(TableUnion::Element(t), value.as_ref(py), None)
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_element_table_default_extract_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = ElementTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_element_table(table, Some(1usize), None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = {
    //             let mut map = FxHashMap::default();
    //             map.insert((0, 0, 0, 0), 2);
    //             map
    //         }
    //         .into_py(py);
    //         let default = 1.5f64.into_py(py);
    //         model.update_table(
    //             TableUnion::Element(t),
    //             value.as_ref(py),
    //             Some(default.as_ref(py)),
    //         )
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_element_table_value_extract_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = ElementTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_element_table(table, Some(1usize), None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = {
    //             let mut map = FxHashMap::default();
    //             map.insert((0, 0, 0, 0), 1.5f64);
    //             map
    //         }
    //         .into_py(py);
    //         let default = 1.5f64.into_py(py);
    //         model.update_table(
    //             TableUnion::Element(t),
    //             value.as_ref(py),
    //             Some(default.as_ref(py)),
    //         )
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_element_table_dimension_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = ElementTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_element_table(table, Some(1usize), None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = vec![2usize].into_py(py);
    //         let default = 3usize.into_py(py);
    //         model.update_table(
    //             TableUnion::Element(t),
    //             value.as_ref(py),
    //             Some(default.as_ref(py)),
    //         )
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    #[test]
    fn add_set_table_1d_from_set_const_ok() {
        let mut model = ModelPy::default();

        let table = SetTableArgUnion::Table1D(vec![
            TargetSetArgUnion::SetConst(SetConstPy::from(Set::with_capacity(10))),
            TargetSetArgUnion::SetConst(SetConstPy::from({
                let mut set = Set::with_capacity(10);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set
            })),
        ]);
        let t1 = model.add_set_table(table, None, None, None);
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        let t1 = match t1 {
            SetTableUnion::Table1D(t) => t,
            _ => panic!("expected SetTableUnion::Table1D but `{:?}`", t1),
        };
        assert_eq!(t1.get_capacity_of_set(), 10);
        let t_id = Table1DHandle::<Set>::from(t1).id();
        assert_eq!(
            model.0.table_registry.set_tables.tables_1d[t_id],
            dypdl::Table1D::new(vec![Set::with_capacity(10), {
                let mut set = Set::with_capacity(10);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set
            }])
        );

        let table = SetTableArgUnion::Table1D(vec![
            TargetSetArgUnion::SetConst(SetConstPy::from(Set::with_capacity(10))),
            TargetSetArgUnion::SetConst(SetConstPy::from({
                let mut set = Set::with_capacity(10);
                set.insert(3);
                set.insert(4);
                set.insert(5);
                set
            })),
        ]);
        let t2 = model.add_set_table(table, None, None, None);
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        let t2 = match t2 {
            SetTableUnion::Table1D(t) => t,
            _ => panic!("expected SetTableUnion::Table1D but `{:?}`", t2),
        };
        assert_eq!(t2.get_capacity_of_set(), 10);
        let t_id = Table1DHandle::<Set>::from(t2).id();
        assert_eq!(
            model.0.table_registry.set_tables.tables_1d[t_id],
            dypdl::Table1D::new(vec![Set::with_capacity(10), {
                let mut set = Set::with_capacity(10);
                set.insert(3);
                set.insert(4);
                set.insert(5);
                set
            }])
        );
    }

    #[test]
    fn add_set_table_1d_from_set_const_with_name_ok() {
        let mut model = ModelPy::default();

        let table = SetTableArgUnion::Table1D(vec![
            TargetSetArgUnion::SetConst(SetConstPy::from(Set::with_capacity(10))),
            TargetSetArgUnion::SetConst(SetConstPy::from({
                let mut set = Set::with_capacity(10);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set
            })),
        ]);
        let t1 = model.add_set_table(table, None, Some("t1"), None);
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        let t1 = match t1 {
            SetTableUnion::Table1D(t) => t,
            _ => panic!("expected SetTableUnion::Table1D but `{:?}`", t1),
        };
        assert_eq!(t1.get_capacity_of_set(), 10);
        let t_id = Table1DHandle::<Set>::from(t1).id();
        assert_eq!(
            model.0.table_registry.set_tables.tables_1d[t_id],
            dypdl::Table1D::new(vec![Set::with_capacity(10), {
                let mut set = Set::with_capacity(10);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set
            }])
        );
        assert_eq!(
            model.0.table_registry.set_tables.name_to_table_1d.get("t1"),
            Some(&t_id)
        );

        let table = SetTableArgUnion::Table1D(vec![
            TargetSetArgUnion::SetConst(SetConstPy::from(Set::with_capacity(10))),
            TargetSetArgUnion::SetConst(SetConstPy::from({
                let mut set = Set::with_capacity(10);
                set.insert(3);
                set.insert(4);
                set.insert(5);
                set
            })),
        ]);
        let t2 = model.add_set_table(table, None, Some("t2"), None);
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        let t2 = match t2 {
            SetTableUnion::Table1D(t) => t,
            _ => panic!("expected SetTableUnion::Table1D but `{:?}`", t2),
        };
        assert_eq!(t2.get_capacity_of_set(), 10);
        let t_id = Table1DHandle::<Set>::from(t2).id();
        assert_eq!(
            model.0.table_registry.set_tables.tables_1d[t_id],
            dypdl::Table1D::new(vec![Set::with_capacity(10), {
                let mut set = Set::with_capacity(10);
                set.insert(3);
                set.insert(4);
                set.insert(5);
                set
            }])
        );
        assert_eq!(
            model.0.table_registry.set_tables.name_to_table_1d.get("t2"),
            Some(&t_id)
        );
    }

    #[test]
    fn add_set_table_1d_from_set_const_duplicate_err() {
        let mut model = ModelPy::default();
        let table = SetTableArgUnion::Table1D(vec![
            TargetSetArgUnion::SetConst(SetConstPy::from(Set::with_capacity(10))),
            TargetSetArgUnion::SetConst(SetConstPy::from({
                let mut set = Set::with_capacity(10);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set
            })),
        ]);
        let t = model.add_set_table(table, None, Some("t1"), None);
        assert!(t.is_ok());
        let table = SetTableArgUnion::Table1D(vec![
            TargetSetArgUnion::SetConst(SetConstPy::from(Set::with_capacity(10))),
            TargetSetArgUnion::SetConst(SetConstPy::from({
                let mut set = Set::with_capacity(10);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set
            })),
        ]);
        let snapshot = model.clone();
        let t = model.add_set_table(table, None, Some("t1"), None);
        assert!(t.is_err());
        assert_eq!(model, snapshot);
    }

    #[test]
    fn add_set_table_1d_from_list_ok() {
        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());

        let table =
            SetTableArgUnion::Table1D(vec![TargetSetArgUnion::SetConst(SetConstPy::from({
                let mut set = Set::with_capacity(10);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set
            }))]);
        let t1 = model.add_set_table(table, None, None, None);
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        let t1 = match t1 {
            SetTableUnion::Table1D(t) => t,
            _ => panic!("expected SetTableUnion::Table1D but `{:?}`", t1),
        };
        assert_eq!(t1.get_capacity_of_set(), 10);

        let table =
            SetTableArgUnion::Table1D(vec![TargetSetArgUnion::SetConst(SetConstPy::from({
                let mut set = Set::with_capacity(10);
                set.insert(3);
                set.insert(4);
                set.insert(5);
                set
            }))]);
        let t2 = model.add_set_table(table, None, None, None);
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        let t2 = match t2 {
            SetTableUnion::Table1D(t) => t,
            _ => panic!("expected SetTableUnion::Table1D but `{:?}`", t2),
        };
        assert_eq!(t2.get_capacity_of_set(), 10);

        assert_ne!(t1, t2);

        let mut model2 = ModelPy::default();
        let ob = model2.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let table = SetTableArgUnion::Table1D(vec![TargetSetArgUnion::CreateSetArg(
            CreateSetArgUnion::List(vec![0, 1, 2]),
        )]);
        let t1 = model2.add_set_table(table, None, None, Some(ob));
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        let t1 = match t1 {
            SetTableUnion::Table1D(t) => t,
            _ => panic!("expected SetTableUnion::Table1D but `{:?}`", t1),
        };
        assert_eq!(t1.get_capacity_of_set(), 10);

        let table = SetTableArgUnion::Table1D(vec![TargetSetArgUnion::CreateSetArg(
            CreateSetArgUnion::List(vec![3, 4, 5]),
        )]);
        let t2 = model2.add_set_table(table, None, None, Some(ob));
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        let t2 = match t2 {
            SetTableUnion::Table1D(t) => t,
            _ => panic!("expected SetTableUnion::Table1D but `{:?}`", t2),
        };
        assert_eq!(t2.get_capacity_of_set(), 10);

        assert_ne!(t1, t2);

        assert_eq!(model2, model);
    }

    #[test]
    fn add_set_table_1d_from_list_with_name_ok() {
        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());

        let table =
            SetTableArgUnion::Table1D(vec![TargetSetArgUnion::SetConst(SetConstPy::from({
                let mut set = Set::with_capacity(10);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set
            }))]);
        let t1 = model.add_set_table(table, None, Some("t1"), None);
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        let t1 = match t1 {
            SetTableUnion::Table1D(t) => t,
            _ => panic!("expected SetTableUnion::Table1D but `{:?}`", t1),
        };
        assert_eq!(t1.get_capacity_of_set(), 10);

        let table =
            SetTableArgUnion::Table1D(vec![TargetSetArgUnion::SetConst(SetConstPy::from({
                let mut set = Set::with_capacity(10);
                set.insert(3);
                set.insert(4);
                set.insert(5);
                set
            }))]);
        let t2 = model.add_set_table(table, None, Some("t2"), None);
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        let t2 = match t2 {
            SetTableUnion::Table1D(t) => t,
            _ => panic!("expected SetTableUnion::Table1D but `{:?}`", t2),
        };
        assert_eq!(t2.get_capacity_of_set(), 10);

        assert_ne!(t1, t2);

        let mut model2 = ModelPy::default();
        let ob = model2.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let table = SetTableArgUnion::Table1D(vec![TargetSetArgUnion::CreateSetArg(
            CreateSetArgUnion::List(vec![0, 1, 2]),
        )]);
        let t1 = model2.add_set_table(table, None, Some("t1"), Some(ob));
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        let t1 = match t1 {
            SetTableUnion::Table1D(t) => t,
            _ => panic!("expected SetTableUnion::Table1D but `{:?}`", t1),
        };
        assert_eq!(t1.get_capacity_of_set(), 10);

        let table = SetTableArgUnion::Table1D(vec![TargetSetArgUnion::CreateSetArg(
            CreateSetArgUnion::List(vec![3, 4, 5]),
        )]);
        let t2 = model2.add_set_table(table, None, Some("t2"), Some(ob));
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        let t2 = match t2 {
            SetTableUnion::Table1D(t) => t,
            _ => panic!("expected SetTableUnion::Table1D but `{:?}`", t2),
        };
        assert_eq!(t2.get_capacity_of_set(), 10);

        assert_ne!(t1, t2);

        assert_eq!(model2, model);
    }

    #[test]
    fn add_set_table_1d_from_list_no_object_err() {
        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let table = SetTableArgUnion::Table1D(vec![TargetSetArgUnion::CreateSetArg(
            CreateSetArgUnion::List(vec![0, 1, 2]),
        )]);
        let mut model = ModelPy::default();
        let snapshot = model.clone();
        let t = model.add_set_table(table, None, None, Some(ob));
        assert!(t.is_err());
        assert_eq!(model, snapshot);
    }

    #[test]
    fn add_set_table_1d_from_list_none_object_err() {
        let mut model = ModelPy::default();

        let table = SetTableArgUnion::Table1D(vec![TargetSetArgUnion::CreateSetArg(
            CreateSetArgUnion::List(vec![0, 1, 2]),
        )]);
        let snapshot = model.clone();
        let t = model.add_set_table(table, None, None, None);
        assert!(t.is_err());
        assert_eq!(model, snapshot);
    }

    #[test]
    fn add_set_table_1d_from_list_table_err() {
        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let table = SetTableArgUnion::Table1D(vec![TargetSetArgUnion::CreateSetArg(
            CreateSetArgUnion::List(vec![0, 1, 2, 10]),
        )]);
        let snapshot = model.clone();
        let t = model.add_set_table(table, None, None, Some(ob));
        assert!(t.is_err());
        assert_eq!(model, snapshot);
    }

    #[test]
    fn add_set_table_1d_from_set_ok() {
        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());

        let table =
            SetTableArgUnion::Table1D(vec![TargetSetArgUnion::SetConst(SetConstPy::from({
                let mut set = Set::with_capacity(10);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set
            }))]);
        let t1 = model.add_set_table(table, None, None, None);
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        let t1 = match t1 {
            SetTableUnion::Table1D(t) => t,
            _ => panic!("expected SetTableUnion::Table1D but `{:?}`", t1),
        };
        assert_eq!(t1.get_capacity_of_set(), 10);

        let table =
            SetTableArgUnion::Table1D(vec![TargetSetArgUnion::SetConst(SetConstPy::from({
                let mut set = Set::with_capacity(10);
                set.insert(3);
                set.insert(4);
                set.insert(5);
                set
            }))]);
        let t2 = model.add_set_table(table, None, None, None);
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        let t2 = match t2 {
            SetTableUnion::Table1D(t) => t,
            _ => panic!("expected SetTableUnion::Table1D but `{:?}`", t2),
        };
        assert_eq!(t2.get_capacity_of_set(), 10);

        assert_ne!(t1, t2);

        let mut model2 = ModelPy::default();
        let ob = model2.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let table =
            SetTableArgUnion::Table1D(vec![TargetSetArgUnion::SetConst(SetConstPy::from({
                let mut set = Set::with_capacity(10);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set
            }))]);
        let t1 = model2.add_set_table(table, None, None, Some(ob));
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        let t1 = match t1 {
            SetTableUnion::Table1D(t) => t,
            _ => panic!("expected SetTableUnion::Table1D but `{:?}`", t1),
        };
        assert_eq!(t1.get_capacity_of_set(), 10);

        let table =
            SetTableArgUnion::Table1D(vec![TargetSetArgUnion::SetConst(SetConstPy::from({
                let mut set = Set::with_capacity(10);
                set.insert(3);
                set.insert(4);
                set.insert(5);
                set
            }))]);
        let t2 = model2.add_set_table(table, None, None, Some(ob));
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        let t2 = match t2 {
            SetTableUnion::Table1D(t) => t,
            _ => panic!("expected SetTableUnion::Table1D but `{:?}`", t2),
        };
        assert_eq!(t2.get_capacity_of_set(), 10);

        assert_ne!(t1, t2);

        assert_eq!(model2, model);
    }

    #[test]
    fn add_set_table_1d_from_set_with_name_ok() {
        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());

        let table =
            SetTableArgUnion::Table1D(vec![TargetSetArgUnion::SetConst(SetConstPy::from({
                let mut set = Set::with_capacity(10);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set
            }))]);
        let t1 = model.add_set_table(table, None, Some("t1"), None);
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        let t1 = match t1 {
            SetTableUnion::Table1D(t) => t,
            _ => panic!("expected SetTableUnion::Table1D but `{:?}`", t1),
        };
        assert_eq!(t1.get_capacity_of_set(), 10);

        let table =
            SetTableArgUnion::Table1D(vec![TargetSetArgUnion::SetConst(SetConstPy::from({
                let mut set = Set::with_capacity(10);
                set.insert(3);
                set.insert(4);
                set.insert(5);
                set
            }))]);
        let t2 = model.add_set_table(table, None, Some("t2"), None);
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        let t2 = match t2 {
            SetTableUnion::Table1D(t) => t,
            _ => panic!("expected SetTableUnion::Table1D but `{:?}`", t2),
        };
        assert_eq!(t2.get_capacity_of_set(), 10);

        assert_ne!(t1, t2);

        let mut model2 = ModelPy::default();
        let ob = model2.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let table = SetTableArgUnion::Table1D(vec![TargetSetArgUnion::CreateSetArg(
            CreateSetArgUnion::Set({
                let mut set = HashSet::new();
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set
            }),
        )]);
        let t1 = model2.add_set_table(table, None, Some("t1"), Some(ob));
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        let t1 = match t1 {
            SetTableUnion::Table1D(t) => t,
            _ => panic!("expected SetTableUnion::Table1D but `{:?}`", t1),
        };
        assert_eq!(t1.get_capacity_of_set(), 10);

        let table = SetTableArgUnion::Table1D(vec![TargetSetArgUnion::CreateSetArg(
            CreateSetArgUnion::Set({
                let mut set = HashSet::new();
                set.insert(3);
                set.insert(4);
                set.insert(5);
                set
            }),
        )]);
        let t2 = model2.add_set_table(table, None, Some("t2"), Some(ob));
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        let t2 = match t2 {
            SetTableUnion::Table1D(t) => t,
            _ => panic!("expected SetTableUnion::Table1D but `{:?}`", t2),
        };
        assert_eq!(t2.get_capacity_of_set(), 10);

        assert_ne!(t1, t2);

        assert_eq!(model2, model);
    }

    #[test]
    fn add_set_table_1d_from_set_no_object_err() {
        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let table = SetTableArgUnion::Table1D(vec![TargetSetArgUnion::CreateSetArg(
            CreateSetArgUnion::Set({
                let mut set = HashSet::new();
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set
            }),
        )]);
        let mut model = ModelPy::default();
        let snapshot = model.clone();
        let t = model.add_set_table(table, None, None, Some(ob));
        assert!(t.is_err());
        assert_eq!(model, snapshot);
    }

    #[test]
    fn add_set_table_1d_from_set_none_object_err() {
        let mut model = ModelPy::default();

        let table = SetTableArgUnion::Table1D(vec![TargetSetArgUnion::CreateSetArg(
            CreateSetArgUnion::Set({
                let mut set = HashSet::new();
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set
            }),
        )]);
        let snapshot = model.clone();
        let t = model.add_set_table(table, None, None, None);
        assert!(t.is_err());
        assert_eq!(model, snapshot);
    }

    #[test]
    fn add_set_table_1d_from_set_table_err() {
        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let table = SetTableArgUnion::Table1D(vec![TargetSetArgUnion::CreateSetArg(
            CreateSetArgUnion::Set({
                let mut set = HashSet::new();
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set.insert(10);
                set
            }),
        )]);
        let snapshot = model.clone();
        let t = model.add_set_table(table, None, None, Some(ob));
        assert!(t.is_err());
        assert_eq!(model, snapshot);
    }

    // #[test]
    // fn set_set_table_1d_item_from_set_const_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = SetTableArgUnion::Table1D(vec![TargetSetArgUnion::SetConst(SetConstPy::from(
    //         Set::with_capacity(10),
    //     ))]);
    //     let t = model.add_set_table(table, None, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let index = TableIndexUnion::Table1D(0);
    //     let result = Python::with_gil(|py| {
    //         let value = SetConstPy::from({
    //             let mut set = Set::with_capacity(10);
    //             set.insert(1);
    //             set
    //         })
    //         .into_py(py);
    //         model.set_table_item(TableUnion::Set(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_ok());
    // }

    // #[test]
    // fn set_set_table_1d_item_value_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = SetTableArgUnion::Table1D(vec![TargetSetArgUnion::SetConst(SetConstPy::from(
    //         Set::with_capacity(10),
    //     ))]);
    //     let t = model.add_set_table(table, None, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let index = TableIndexUnion::Table1D(0);
    //     let mut model = ModelPy::default();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = 1usize.into_py(py);
    //         model.set_table_item(TableUnion::Set(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn set_set_table_1d_item_no_table_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = SetTableArgUnion::Table1D(vec![TargetSetArgUnion::SetConst(SetConstPy::from(
    //         Set::with_capacity(10),
    //     ))]);
    //     let t = model.add_set_table(table, None, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let index = TableIndexUnion::Table1D(0);
    //     let mut model = ModelPy::default();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = SetConstPy::from({
    //             let mut set = Set::with_capacity(10);
    //             set.insert(1);
    //             set
    //         })
    //         .into_py(py);
    //         model.set_table_item(TableUnion::Set(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn set_set_table_1d_item_from_list_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let ob = model.add_object_type(10, None);
    //     assert!(ob.is_ok());
    //     let table = SetTableArgUnion::Table1D(vec![TargetSetArgUnion::SetConst(SetConstPy::from(
    //         Set::with_capacity(10),
    //     ))]);
    //     let t = model.add_set_table(table, None, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let index = TableIndexUnion::Table1D(0);
    //     let result = Python::with_gil(|py| {
    //         let value = vec![1].into_py(py);
    //         model.set_table_item(TableUnion::Set(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_ok());
    // }

    // #[test]
    // fn set_set_table_1d_item_from_list_out_of_bound_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let ob = model.add_object_type(10, None);
    //     assert!(ob.is_ok());
    //     let table = SetTableArgUnion::Table1D(vec![TargetSetArgUnion::SetConst(SetConstPy::from(
    //         Set::with_capacity(10),
    //     ))]);
    //     let t = model.add_set_table(table, None, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let index = TableIndexUnion::Table1D(0);
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = vec![10].into_py(py);
    //         model.set_table_item(TableUnion::Set(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn set_set_table_1d_item_from_set_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let ob = model.add_object_type(10, None);
    //     assert!(ob.is_ok());
    //     let table = SetTableArgUnion::Table1D(vec![TargetSetArgUnion::SetConst(SetConstPy::from(
    //         Set::with_capacity(10),
    //     ))]);
    //     let t = model.add_set_table(table, None, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let index = TableIndexUnion::Table1D(0);
    //     let result = Python::with_gil(|py| {
    //         let value = {
    //             let mut set = HashSet::new();
    //             set.insert(1);
    //             set
    //         }
    //         .into_py(py);
    //         model.set_table_item(TableUnion::Set(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_ok());
    // }

    // #[test]
    // fn set_set_table_1d_item_from_set_out_of_bound_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let ob = model.add_object_type(10, None);
    //     assert!(ob.is_ok());
    //     let table = SetTableArgUnion::Table1D(vec![TargetSetArgUnion::SetConst(SetConstPy::from(
    //         Set::with_capacity(10),
    //     ))]);
    //     let t = model.add_set_table(table, None, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let index = TableIndexUnion::Table1D(0);
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = {
    //             let mut set = HashSet::new();
    //             set.insert(10);
    //             set
    //         }
    //         .into_py(py);
    //         model.set_table_item(TableUnion::Set(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_set_table_1d_from_set_const_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();

    //     let table = SetTableArgUnion::Table1D(vec![TargetSetArgUnion::SetConst(SetConstPy::from(
    //         Set::with_capacity(10),
    //     ))]);
    //     let t1 = model.add_set_table(table, None, None, None);
    //     assert!(t1.is_ok());
    //     let t1 = t1.unwrap();
    //     let t1 = match t1 {
    //         SetTableUnion::Table1D(t) => t,
    //         _ => panic!("expected SetTableUnion::Table1D but `{:?}`", t1),
    //     };
    //     let table = vec![
    //         SetConstPy::from(Set::with_capacity(10)),
    //         SetConstPy::from({
    //             let mut set = Set::with_capacity(10);
    //             set.insert(0);
    //             set.insert(1);
    //             set.insert(2);
    //             set
    //         }),
    //     ];
    //     let result = Python::with_gil(|py| {
    //         let table = table.into_py(py);
    //         model.update_table(
    //             TableUnion::Set(SetTableUnion::Table1D(t1.clone())),
    //             table.as_ref(py),
    //             None,
    //         )
    //     });
    //     assert!(result.is_ok());
    //     let t_id = Table1DHandle::<Set>::from(t1).id();
    //     assert_eq!(
    //         model.0.table_registry.set_tables.tables_1d[t_id],
    //         dypdl::Table1D::new(vec![Set::with_capacity(10), {
    //             let mut set = Set::with_capacity(10);
    //             set.insert(0);
    //             set.insert(1);
    //             set.insert(2);
    //             set
    //         }])
    //     );
    // }

    // #[test]
    // fn update_set_table_1d_value_extract_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();

    //     let table = SetTableArgUnion::Table1D(vec![TargetSetArgUnion::SetConst(SetConstPy::from(
    //         Set::with_capacity(10),
    //     ))]);
    //     let t1 = model.add_set_table(table, None, None, None);
    //     assert!(t1.is_ok());
    //     let t1 = t1.unwrap();
    //     let t1 = match t1 {
    //         SetTableUnion::Table1D(t) => t,
    //         _ => panic!("expected SetTableUnion::Table1D but `{:?}`", t1),
    //     };
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let table = 1usize.into_py(py);
    //         model.update_table(
    //             TableUnion::Set(SetTableUnion::Table1D(t1.clone())),
    //             table.as_ref(py),
    //             None,
    //         )
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_set_table_1d_dimension_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();

    //     let table = SetTableArgUnion::Table1D(vec![TargetSetArgUnion::SetConst(SetConstPy::from(
    //         Set::with_capacity(10),
    //     ))]);
    //     let t1 = model.add_set_table(table, None, None, None);
    //     assert!(t1.is_ok());
    //     let t1 = t1.unwrap();
    //     let t1 = match t1 {
    //         SetTableUnion::Table1D(t) => t,
    //         _ => panic!("expected SetTableUnion::Table1D but `{:?}`", t1),
    //     };
    //     let snapshot = model.clone();
    //     let table = vec![vec![
    //         SetConstPy::from(Set::with_capacity(10)),
    //         SetConstPy::from({
    //             let mut set = Set::with_capacity(10);
    //             set.insert(0);
    //             set.insert(1);
    //             set.insert(2);
    //             set
    //         }),
    //     ]];
    //     let result = Python::with_gil(|py| {
    //         let table = table.into_py(py);
    //         model.update_table(
    //             TableUnion::Set(SetTableUnion::Table1D(t1.clone())),
    //             table.as_ref(py),
    //             None,
    //         )
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_set_table_1d_no_table_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();

    //     let table = SetTableArgUnion::Table1D(vec![TargetSetArgUnion::SetConst(SetConstPy::from(
    //         Set::with_capacity(10),
    //     ))]);
    //     let t1 = model.add_set_table(table, None, None, None);
    //     assert!(t1.is_ok());
    //     let t1 = t1.unwrap();
    //     let t1 = match t1 {
    //         SetTableUnion::Table1D(t) => t,
    //         _ => panic!("expected SetTableUnion::Table1D but `{:?}`", t1),
    //     };
    //     let table = vec![
    //         SetConstPy::from(Set::with_capacity(10)),
    //         SetConstPy::from({
    //             let mut set = Set::with_capacity(10);
    //             set.insert(0);
    //             set.insert(1);
    //             set.insert(2);
    //             set
    //         }),
    //     ];
    //     let mut model = ModelPy::default();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let table = table.into_py(py);
    //         model.update_table(
    //             TableUnion::Set(SetTableUnion::Table1D(t1.clone())),
    //             table.as_ref(py),
    //             None,
    //         )
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_set_table_1d_from_list_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let ob = model.add_object_type(10, None);
    //     assert!(ob.is_ok());

    //     let table = SetTableArgUnion::Table1D(vec![TargetSetArgUnion::SetConst(SetConstPy::from(
    //         Set::with_capacity(10),
    //     ))]);
    //     let t1 = model.add_set_table(table, None, None, None);
    //     assert!(t1.is_ok());
    //     let t1 = t1.unwrap();
    //     let t1 = match t1 {
    //         SetTableUnion::Table1D(t) => t,
    //         _ => panic!("expected SetTableUnion::Table1D but `{:?}`", t1),
    //     };
    //     let result = Python::with_gil(|py| {
    //         let table = vec![vec![], vec![0, 1, 2]].into_py(py);
    //         model.update_table(
    //             TableUnion::Set(SetTableUnion::Table1D(t1.clone())),
    //             table.as_ref(py),
    //             None,
    //         )
    //     });
    //     assert!(result.is_ok());
    //     let t_id = Table1DHandle::<Set>::from(t1).id();
    //     assert_eq!(
    //         model.0.table_registry.set_tables.tables_1d[t_id],
    //         dypdl::Table1D::new(vec![Set::with_capacity(10), {
    //             let mut set = Set::with_capacity(10);
    //             set.insert(0);
    //             set.insert(1);
    //             set.insert(2);
    //             set
    //         }])
    //     );
    // }

    // #[test]
    // fn update_set_table_1d_from_list_out_of_bound_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let ob = model.add_object_type(10, None);
    //     assert!(ob.is_ok());

    //     let table = SetTableArgUnion::Table1D(vec![TargetSetArgUnion::SetConst(SetConstPy::from(
    //         Set::with_capacity(10),
    //     ))]);
    //     let t1 = model.add_set_table(table, None, None, None);
    //     assert!(t1.is_ok());
    //     let t1 = t1.unwrap();
    //     let t1 = match t1 {
    //         SetTableUnion::Table1D(t) => t,
    //         _ => panic!("expected SetTableUnion::Table1D but `{:?}`", t1),
    //     };
    //     let result = Python::with_gil(|py| {
    //         let table = vec![0, 1, 2, 10].into_py(py);
    //         model.update_table(
    //             TableUnion::Set(SetTableUnion::Table1D(t1.clone())),
    //             table.as_ref(py),
    //             None,
    //         )
    //     });
    //     assert!(result.is_err());
    // }

    // #[test]
    // fn update_set_table_1d_from_set_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let ob = model.add_object_type(10, None);
    //     assert!(ob.is_ok());

    //     let table = SetTableArgUnion::Table1D(vec![TargetSetArgUnion::SetConst(SetConstPy::from(
    //         Set::with_capacity(10),
    //     ))]);
    //     let t1 = model.add_set_table(table, None, None, None);
    //     assert!(t1.is_ok());
    //     let t1 = t1.unwrap();
    //     let t1 = match t1 {
    //         SetTableUnion::Table1D(t) => t,
    //         _ => panic!("expected SetTableUnion::Table1D but `{:?}`", t1),
    //     };
    //     let result = Python::with_gil(|py| {
    //         let table = vec![HashSet::new(), {
    //             let mut set = HashSet::new();
    //             set.insert(0);
    //             set.insert(1);
    //             set.insert(2);
    //             set
    //         }]
    //         .into_py(py);
    //         model.update_table(
    //             TableUnion::Set(SetTableUnion::Table1D(t1.clone())),
    //             table.as_ref(py),
    //             None,
    //         )
    //     });
    //     assert!(result.is_ok());
    //     let t_id = Table1DHandle::<Set>::from(t1).id();
    //     assert_eq!(
    //         model.0.table_registry.set_tables.tables_1d[t_id],
    //         dypdl::Table1D::new(vec![Set::with_capacity(10), {
    //             let mut set = Set::with_capacity(10);
    //             set.insert(0);
    //             set.insert(1);
    //             set.insert(2);
    //             set
    //         }])
    //     );
    // }

    // #[test]
    // fn update_set_table_1d_from_set_out_of_bound_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let ob = model.add_object_type(10, None);
    //     assert!(ob.is_ok());

    //     let table = SetTableArgUnion::Table1D(vec![TargetSetArgUnion::SetConst(SetConstPy::from(
    //         Set::with_capacity(10),
    //     ))]);
    //     let t1 = model.add_set_table(table, None, None, None);
    //     assert!(t1.is_ok());
    //     let t1 = t1.unwrap();
    //     let t1 = match t1 {
    //         SetTableUnion::Table1D(t) => t,
    //         _ => panic!("expected SetTableUnion::Table1D but `{:?}`", t1),
    //     };
    //     let result = Python::with_gil(|py| {
    //         let table = vec![HashSet::new(), {
    //             let mut set = HashSet::new();
    //             set.insert(0);
    //             set.insert(1);
    //             set.insert(2);
    //             set.insert(10);
    //             set
    //         }]
    //         .into_py(py);
    //         model.update_table(
    //             TableUnion::Set(SetTableUnion::Table1D(t1.clone())),
    //             table.as_ref(py),
    //             None,
    //         )
    //     });
    //     assert!(result.is_err());
    // }

    #[test]
    fn add_set_table_2d_from_set_const_ok() {
        let mut model = ModelPy::default();

        let table = SetTableArgUnion::Table2D(vec![vec![
            TargetSetArgUnion::SetConst(SetConstPy::from(Set::with_capacity(10))),
            TargetSetArgUnion::SetConst(SetConstPy::from({
                let mut set = Set::with_capacity(10);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set
            })),
        ]]);
        let t1 = model.add_set_table(table, None, None, None);
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        let t1 = match t1 {
            SetTableUnion::Table2D(t) => t,
            _ => panic!("expected SetTableUnion::Table2D but `{:?}`", t1),
        };
        assert_eq!(t1.get_capacity_of_set(), 10);
        let t_id = Table2DHandle::<Set>::from(t1).id();
        assert_eq!(
            model.0.table_registry.set_tables.tables_2d[t_id],
            dypdl::Table2D::new(vec![vec![Set::with_capacity(10), {
                let mut set = Set::with_capacity(10);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set
            }]])
        );

        let table = SetTableArgUnion::Table2D(vec![vec![
            TargetSetArgUnion::SetConst(SetConstPy::from(Set::with_capacity(10))),
            TargetSetArgUnion::SetConst(SetConstPy::from({
                let mut set = Set::with_capacity(10);
                set.insert(3);
                set.insert(4);
                set.insert(5);
                set
            })),
        ]]);
        let t2 = model.add_set_table(table, None, None, None);
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        let t2 = match t2 {
            SetTableUnion::Table2D(t) => t,
            _ => panic!("expected SetTableUnion::Table2D but `{:?}`", t2),
        };
        assert_eq!(t2.get_capacity_of_set(), 10);
        let t_id = Table2DHandle::<Set>::from(t2).id();
        assert_eq!(
            model.0.table_registry.set_tables.tables_2d[t_id],
            dypdl::Table2D::new(vec![vec![Set::with_capacity(10), {
                let mut set = Set::with_capacity(10);
                set.insert(3);
                set.insert(4);
                set.insert(5);
                set
            }]])
        );
    }

    #[test]
    fn add_set_table_2d_from_set_const_with_name_ok() {
        let mut model = ModelPy::default();

        let table = SetTableArgUnion::Table2D(vec![vec![
            TargetSetArgUnion::SetConst(SetConstPy::from(Set::with_capacity(10))),
            TargetSetArgUnion::SetConst(SetConstPy::from({
                let mut set = Set::with_capacity(10);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set
            })),
        ]]);
        let t1 = model.add_set_table(table, None, Some("t1"), None);
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        let t1 = match t1 {
            SetTableUnion::Table2D(t) => t,
            _ => panic!("expected SetTableUnion::Table2D but `{:?}`", t1),
        };
        assert_eq!(t1.get_capacity_of_set(), 10);
        let t_id = Table2DHandle::<Set>::from(t1).id();
        assert_eq!(
            model.0.table_registry.set_tables.tables_2d[t_id],
            dypdl::Table2D::new(vec![vec![Set::with_capacity(10), {
                let mut set = Set::with_capacity(10);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set
            }]])
        );
        assert_eq!(
            model.0.table_registry.set_tables.name_to_table_2d.get("t1"),
            Some(&t_id)
        );

        let table = SetTableArgUnion::Table2D(vec![vec![
            TargetSetArgUnion::SetConst(SetConstPy::from(Set::with_capacity(10))),
            TargetSetArgUnion::SetConst(SetConstPy::from({
                let mut set = Set::with_capacity(10);
                set.insert(3);
                set.insert(4);
                set.insert(5);
                set
            })),
        ]]);
        let t2 = model.add_set_table(table, None, Some("t2"), None);
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        let t2 = match t2 {
            SetTableUnion::Table2D(t) => t,
            _ => panic!("expected SetTableUnion::Table2D but `{:?}`", t2),
        };
        assert_eq!(t2.get_capacity_of_set(), 10);
        let t_id = Table2DHandle::<Set>::from(t2).id();
        assert_eq!(
            model.0.table_registry.set_tables.tables_2d[t_id],
            dypdl::Table2D::new(vec![vec![Set::with_capacity(10), {
                let mut set = Set::with_capacity(10);
                set.insert(3);
                set.insert(4);
                set.insert(5);
                set
            }]])
        );
        assert_eq!(
            model.0.table_registry.set_tables.name_to_table_2d.get("t2"),
            Some(&t_id)
        );
    }

    #[test]
    fn add_set_table_2d_from_set_const_duplicate_err() {
        let mut model = ModelPy::default();
        let table = SetTableArgUnion::Table2D(vec![vec![
            TargetSetArgUnion::SetConst(SetConstPy::from(Set::with_capacity(10))),
            TargetSetArgUnion::SetConst(SetConstPy::from({
                let mut set = Set::with_capacity(10);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set
            })),
        ]]);
        let t = model.add_set_table(table, None, Some("t1"), None);
        assert!(t.is_ok());
        let table = SetTableArgUnion::Table2D(vec![vec![
            TargetSetArgUnion::SetConst(SetConstPy::from(Set::with_capacity(10))),
            TargetSetArgUnion::SetConst(SetConstPy::from({
                let mut set = Set::with_capacity(10);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set
            })),
        ]]);
        let snapshot = model.clone();
        let t = model.add_set_table(table, None, Some("t1"), None);
        assert!(t.is_err());
        assert_eq!(model, snapshot);
    }

    #[test]
    fn add_set_table_2d_from_list_ok() {
        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());

        let table =
            SetTableArgUnion::Table2D(vec![vec![TargetSetArgUnion::SetConst(SetConstPy::from({
                let mut set = Set::with_capacity(10);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set
            }))]]);
        let t1 = model.add_set_table(table, None, None, None);
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        let t1 = match t1 {
            SetTableUnion::Table2D(t) => t,
            _ => panic!("expected SetTableUnion::Table2D but `{:?}`", t1),
        };
        assert_eq!(t1.get_capacity_of_set(), 10);

        let table =
            SetTableArgUnion::Table2D(vec![vec![TargetSetArgUnion::SetConst(SetConstPy::from({
                let mut set = Set::with_capacity(10);
                set.insert(3);
                set.insert(4);
                set.insert(5);
                set
            }))]]);
        let t2 = model.add_set_table(table, None, None, None);
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        let t2 = match t2 {
            SetTableUnion::Table2D(t) => t,
            _ => panic!("expected SetTableUnion::Table2D but `{:?}`", t2),
        };
        assert_eq!(t2.get_capacity_of_set(), 10);

        assert_ne!(t1, t2);

        let mut model2 = ModelPy::default();
        let ob = model2.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let table = SetTableArgUnion::Table2D(vec![vec![TargetSetArgUnion::CreateSetArg(
            CreateSetArgUnion::List(vec![0, 1, 2]),
        )]]);
        let t1 = model2.add_set_table(table, None, None, Some(ob));
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        let t1 = match t1 {
            SetTableUnion::Table2D(t) => t,
            _ => panic!("expected SetTableUnion::Table2D but `{:?}`", t1),
        };
        assert_eq!(t1.get_capacity_of_set(), 10);

        let table = SetTableArgUnion::Table2D(vec![vec![TargetSetArgUnion::CreateSetArg(
            CreateSetArgUnion::List(vec![3, 4, 5]),
        )]]);
        let t2 = model2.add_set_table(table, None, None, Some(ob));
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        let t2 = match t2 {
            SetTableUnion::Table2D(t) => t,
            _ => panic!("expected SetTableUnion::Table2D but `{:?}`", t2),
        };
        assert_eq!(t2.get_capacity_of_set(), 10);

        assert_ne!(t1, t2);

        assert_eq!(model2, model);
    }

    #[test]
    fn add_set_table_2d_from_list_with_name_ok() {
        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());

        let table =
            SetTableArgUnion::Table2D(vec![vec![TargetSetArgUnion::SetConst(SetConstPy::from({
                let mut set = Set::with_capacity(10);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set
            }))]]);
        let t1 = model.add_set_table(table, None, Some("t1"), None);
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        let t1 = match t1 {
            SetTableUnion::Table2D(t) => t,
            _ => panic!("expected SetTableUnion::Table2D but `{:?}`", t1),
        };
        assert_eq!(t1.get_capacity_of_set(), 10);

        let table =
            SetTableArgUnion::Table2D(vec![vec![TargetSetArgUnion::SetConst(SetConstPy::from({
                let mut set = Set::with_capacity(10);
                set.insert(3);
                set.insert(4);
                set.insert(5);
                set
            }))]]);
        let t2 = model.add_set_table(table, None, Some("t2"), None);
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        let t2 = match t2 {
            SetTableUnion::Table2D(t) => t,
            _ => panic!("expected SetTableUnion::Table2D but `{:?}`", t2),
        };
        assert_eq!(t2.get_capacity_of_set(), 10);

        assert_ne!(t1, t2);

        let mut model2 = ModelPy::default();
        let ob = model2.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let table = SetTableArgUnion::Table2D(vec![vec![TargetSetArgUnion::CreateSetArg(
            CreateSetArgUnion::List(vec![0, 1, 2]),
        )]]);
        let t1 = model2.add_set_table(table, None, Some("t1"), Some(ob));
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        let t1 = match t1 {
            SetTableUnion::Table2D(t) => t,
            _ => panic!("expected SetTableUnion::Table2D but `{:?}`", t1),
        };
        assert_eq!(t1.get_capacity_of_set(), 10);

        let table = SetTableArgUnion::Table2D(vec![vec![TargetSetArgUnion::CreateSetArg(
            CreateSetArgUnion::List(vec![3, 4, 5]),
        )]]);
        let t2 = model2.add_set_table(table, None, Some("t2"), Some(ob));
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        let t2 = match t2 {
            SetTableUnion::Table2D(t) => t,
            _ => panic!("expected SetTableUnion::Table2D but `{:?}`", t2),
        };
        assert_eq!(t2.get_capacity_of_set(), 10);

        assert_ne!(t1, t2);

        assert_eq!(model2, model);
    }

    #[test]
    fn add_set_table_2d_from_list_no_object_err() {
        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let table = SetTableArgUnion::Table2D(vec![vec![TargetSetArgUnion::CreateSetArg(
            CreateSetArgUnion::List(vec![0, 1, 2]),
        )]]);
        let mut model = ModelPy::default();
        let snapshot = model.clone();
        let t = model.add_set_table(table, None, None, Some(ob));
        assert!(t.is_err());
        assert_eq!(model, snapshot);
    }

    #[test]
    fn add_set_table_2d_from_list_none_object_err() {
        let mut model = ModelPy::default();

        let table = SetTableArgUnion::Table2D(vec![vec![TargetSetArgUnion::CreateSetArg(
            CreateSetArgUnion::List(vec![0, 1, 2]),
        )]]);
        let snapshot = model.clone();
        let t = model.add_set_table(table, None, None, None);
        assert!(t.is_err());
        assert_eq!(model, snapshot);
    }

    #[test]
    fn add_set_table_2d_from_list_table_err() {
        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let table = SetTableArgUnion::Table2D(vec![vec![TargetSetArgUnion::CreateSetArg(
            CreateSetArgUnion::List(vec![0, 1, 2, 10]),
        )]]);
        let snapshot = model.clone();
        let t = model.add_set_table(table, None, None, Some(ob));
        assert!(t.is_err());
        assert_eq!(model, snapshot);
    }

    #[test]
    fn add_set_table_2d_from_set_ok() {
        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());

        let table =
            SetTableArgUnion::Table2D(vec![vec![TargetSetArgUnion::SetConst(SetConstPy::from({
                let mut set = Set::with_capacity(10);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set
            }))]]);
        let t1 = model.add_set_table(table, None, None, None);
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        let t1 = match t1 {
            SetTableUnion::Table2D(t) => t,
            _ => panic!("expected SetTableUnion::Table2D but `{:?}`", t1),
        };
        assert_eq!(t1.get_capacity_of_set(), 10);

        let table =
            SetTableArgUnion::Table2D(vec![vec![TargetSetArgUnion::SetConst(SetConstPy::from({
                let mut set = Set::with_capacity(10);
                set.insert(3);
                set.insert(4);
                set.insert(5);
                set
            }))]]);
        let t2 = model.add_set_table(table, None, None, None);
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        let t2 = match t2 {
            SetTableUnion::Table2D(t) => t,
            _ => panic!("expected SetTableUnion::Table2D but `{:?}`", t2),
        };
        assert_eq!(t2.get_capacity_of_set(), 10);

        assert_ne!(t1, t2);

        let mut model2 = ModelPy::default();
        let ob = model2.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let table =
            SetTableArgUnion::Table2D(vec![vec![TargetSetArgUnion::SetConst(SetConstPy::from({
                let mut set = Set::with_capacity(10);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set
            }))]]);
        let t1 = model2.add_set_table(table, None, None, Some(ob));
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        let t1 = match t1 {
            SetTableUnion::Table2D(t) => t,
            _ => panic!("expected SetTableUnion::Table2D but `{:?}`", t1),
        };
        assert_eq!(t1.get_capacity_of_set(), 10);

        let table =
            SetTableArgUnion::Table2D(vec![vec![TargetSetArgUnion::SetConst(SetConstPy::from({
                let mut set = Set::with_capacity(10);
                set.insert(3);
                set.insert(4);
                set.insert(5);
                set
            }))]]);
        let t2 = model2.add_set_table(table, None, None, Some(ob));
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        let t2 = match t2 {
            SetTableUnion::Table2D(t) => t,
            _ => panic!("expected SetTableUnion::Table2D but `{:?}`", t2),
        };
        assert_eq!(t2.get_capacity_of_set(), 10);

        assert_ne!(t1, t2);

        assert_eq!(model2, model);
    }

    #[test]
    fn add_set_table_2d_from_set_with_name_ok() {
        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());

        let table =
            SetTableArgUnion::Table2D(vec![vec![TargetSetArgUnion::SetConst(SetConstPy::from({
                let mut set = Set::with_capacity(10);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set
            }))]]);
        let t1 = model.add_set_table(table, None, Some("t1"), None);
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        let t1 = match t1 {
            SetTableUnion::Table2D(t) => t,
            _ => panic!("expected SetTableUnion::Table2D but `{:?}`", t1),
        };
        assert_eq!(t1.get_capacity_of_set(), 10);

        let table =
            SetTableArgUnion::Table2D(vec![vec![TargetSetArgUnion::SetConst(SetConstPy::from({
                let mut set = Set::with_capacity(10);
                set.insert(3);
                set.insert(4);
                set.insert(5);
                set
            }))]]);
        let t2 = model.add_set_table(table, None, Some("t2"), None);
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        let t2 = match t2 {
            SetTableUnion::Table2D(t) => t,
            _ => panic!("expected SetTableUnion::Table2D but `{:?}`", t2),
        };
        assert_eq!(t2.get_capacity_of_set(), 10);

        assert_ne!(t1, t2);

        let mut model2 = ModelPy::default();
        let ob = model2.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let table = SetTableArgUnion::Table2D(vec![vec![TargetSetArgUnion::CreateSetArg(
            CreateSetArgUnion::Set({
                let mut set = HashSet::new();
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set
            }),
        )]]);
        let t1 = model2.add_set_table(table, None, Some("t1"), Some(ob));
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        let t1 = match t1 {
            SetTableUnion::Table2D(t) => t,
            _ => panic!("expected SetTableUnion::Table2D but `{:?}`", t1),
        };
        assert_eq!(t1.get_capacity_of_set(), 10);

        let table = SetTableArgUnion::Table2D(vec![vec![TargetSetArgUnion::CreateSetArg(
            CreateSetArgUnion::Set({
                let mut set = HashSet::new();
                set.insert(3);
                set.insert(4);
                set.insert(5);
                set
            }),
        )]]);
        let t2 = model2.add_set_table(table, None, Some("t2"), Some(ob));
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        let t2 = match t2 {
            SetTableUnion::Table2D(t) => t,
            _ => panic!("expected SetTableUnion::Table2D but `{:?}`", t2),
        };
        assert_eq!(t2.get_capacity_of_set(), 10);

        assert_ne!(t1, t2);

        assert_eq!(model2, model);
    }

    #[test]
    fn add_set_table_2d_from_set_no_object_err() {
        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let table = SetTableArgUnion::Table2D(vec![vec![TargetSetArgUnion::CreateSetArg(
            CreateSetArgUnion::Set({
                let mut set = HashSet::new();
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set
            }),
        )]]);
        let mut model = ModelPy::default();
        let snapshot = model.clone();
        let t = model.add_set_table(table, None, None, Some(ob));
        assert!(t.is_err());
        assert_eq!(model, snapshot);
    }

    #[test]
    fn add_set_table_2d_from_set_none_object_err() {
        let mut model = ModelPy::default();

        let table = SetTableArgUnion::Table2D(vec![vec![TargetSetArgUnion::CreateSetArg(
            CreateSetArgUnion::Set({
                let mut set = HashSet::new();
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set
            }),
        )]]);
        let snapshot = model.clone();
        let t = model.add_set_table(table, None, None, None);
        assert!(t.is_err());
        assert_eq!(model, snapshot);
    }

    #[test]
    fn add_set_table_2d_from_set_table_err() {
        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let table = SetTableArgUnion::Table2D(vec![vec![TargetSetArgUnion::CreateSetArg(
            CreateSetArgUnion::Set({
                let mut set = HashSet::new();
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set.insert(10);
                set
            }),
        )]]);
        let snapshot = model.clone();
        let t = model.add_set_table(table, None, None, Some(ob));
        assert!(t.is_err());
        assert_eq!(model, snapshot);
    }

    // #[test]
    // fn set_set_table_2d_item_from_set_const_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = SetTableArgUnion::Table2D(vec![vec![TargetSetArgUnion::SetConst(
    //         SetConstPy::from(Set::with_capacity(10)),
    //     )]]);
    //     let t = model.add_set_table(table, None, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let index = TableIndexUnion::Table2D((0, 0));
    //     let result = Python::with_gil(|py| {
    //         let value = SetConstPy::from({
    //             let mut set = Set::with_capacity(10);
    //             set.insert(1);
    //             set
    //         })
    //         .into_py(py);
    //         model.set_table_item(TableUnion::Set(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_ok());
    // }

    // #[test]
    // fn set_set_table_2d_item_value_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = SetTableArgUnion::Table2D(vec![vec![TargetSetArgUnion::SetConst(
    //         SetConstPy::from(Set::with_capacity(10)),
    //     )]]);
    //     let t = model.add_set_table(table, None, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let index = TableIndexUnion::Table2D((0, 0));
    //     let mut model = ModelPy::default();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = 1usize.into_py(py);
    //         model.set_table_item(TableUnion::Set(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn set_set_table_2d_item_no_table_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = SetTableArgUnion::Table2D(vec![vec![TargetSetArgUnion::SetConst(
    //         SetConstPy::from(Set::with_capacity(10)),
    //     )]]);
    //     let t = model.add_set_table(table, None, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let index = TableIndexUnion::Table2D((0, 0));
    //     let mut model = ModelPy::default();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = SetConstPy::from({
    //             let mut set = Set::with_capacity(10);
    //             set.insert(1);
    //             set
    //         })
    //         .into_py(py);
    //         model.set_table_item(TableUnion::Set(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn set_set_table_2d_item_from_list_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let ob = model.add_object_type(10, None);
    //     assert!(ob.is_ok());
    //     let table = SetTableArgUnion::Table2D(vec![vec![TargetSetArgUnion::SetConst(
    //         SetConstPy::from(Set::with_capacity(10)),
    //     )]]);
    //     let t = model.add_set_table(table, None, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let index = TableIndexUnion::Table2D((0, 0));
    //     let result = Python::with_gil(|py| {
    //         let value = vec![1].into_py(py);
    //         model.set_table_item(TableUnion::Set(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_ok());
    // }

    // #[test]
    // fn set_set_table_2d_item_from_list_out_of_bound_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let ob = model.add_object_type(10, None);
    //     assert!(ob.is_ok());
    //     let table = SetTableArgUnion::Table2D(vec![vec![TargetSetArgUnion::SetConst(
    //         SetConstPy::from(Set::with_capacity(10)),
    //     )]]);
    //     let t = model.add_set_table(table, None, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let index = TableIndexUnion::Table2D((0, 0));
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = vec![10].into_py(py);
    //         model.set_table_item(TableUnion::Set(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn set_set_table_2d_item_from_set_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let ob = model.add_object_type(10, None);
    //     assert!(ob.is_ok());
    //     let table = SetTableArgUnion::Table2D(vec![vec![TargetSetArgUnion::SetConst(
    //         SetConstPy::from(Set::with_capacity(10)),
    //     )]]);
    //     let t = model.add_set_table(table, None, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let index = TableIndexUnion::Table2D((0, 0));
    //     let result = Python::with_gil(|py| {
    //         let value = {
    //             let mut set = HashSet::new();
    //             set.insert(1);
    //             set
    //         }
    //         .into_py(py);
    //         model.set_table_item(TableUnion::Set(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_ok());
    // }

    // #[test]
    // fn set_set_table_2d_item_from_set_out_of_bound_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let ob = model.add_object_type(10, None);
    //     assert!(ob.is_ok());
    //     let table = SetTableArgUnion::Table2D(vec![vec![TargetSetArgUnion::SetConst(
    //         SetConstPy::from(Set::with_capacity(10)),
    //     )]]);
    //     let t = model.add_set_table(table, None, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let index = TableIndexUnion::Table2D((0, 0));
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = {
    //             let mut set = HashSet::new();
    //             set.insert(10);
    //             set
    //         }
    //         .into_py(py);
    //         model.set_table_item(TableUnion::Set(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_set_table_2d_from_set_const_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();

    //     let table = SetTableArgUnion::Table2D(vec![vec![TargetSetArgUnion::SetConst(
    //         SetConstPy::from(Set::with_capacity(10)),
    //     )]]);
    //     let t1 = model.add_set_table(table, None, None, None);
    //     assert!(t1.is_ok());
    //     let t1 = t1.unwrap();
    //     let t1 = match t1 {
    //         SetTableUnion::Table2D(t) => t,
    //         _ => panic!("expected SetTableUnion::Table2D but `{:?}`", t1),
    //     };
    //     let table = vec![vec![
    //         SetConstPy::from(Set::with_capacity(10)),
    //         SetConstPy::from({
    //             let mut set = Set::with_capacity(10);
    //             set.insert(0);
    //             set.insert(1);
    //             set.insert(2);
    //             set
    //         }),
    //     ]];
    //     let result = Python::with_gil(|py| {
    //         let table = table.into_py(py);
    //         model.update_table(
    //             TableUnion::Set(SetTableUnion::Table2D(t1.clone())),
    //             table.as_ref(py),
    //             None,
    //         )
    //     });
    //     assert!(result.is_ok());
    //     let t_id = Table2DHandle::<Set>::from(t1).id();
    //     assert_eq!(
    //         model.0.table_registry.set_tables.tables_2d[t_id],
    //         dypdl::Table2D::new(vec![vec![Set::with_capacity(10), {
    //             let mut set = Set::with_capacity(10);
    //             set.insert(0);
    //             set.insert(1);
    //             set.insert(2);
    //             set
    //         }]])
    //     );
    // }

    // #[test]
    // fn update_set_table_2d_value_extract_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();

    //     let table = SetTableArgUnion::Table2D(vec![vec![TargetSetArgUnion::SetConst(
    //         SetConstPy::from(Set::with_capacity(10)),
    //     )]]);
    //     let t1 = model.add_set_table(table, None, None, None);
    //     assert!(t1.is_ok());
    //     let t1 = t1.unwrap();
    //     let t1 = match t1 {
    //         SetTableUnion::Table2D(t) => t,
    //         _ => panic!("expected SetTableUnion::Table2D but `{:?}`", t1),
    //     };
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let table = 1usize.into_py(py);
    //         model.update_table(
    //             TableUnion::Set(SetTableUnion::Table2D(t1.clone())),
    //             table.as_ref(py),
    //             None,
    //         )
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_set_table_2d_dimension_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();

    //     let table = SetTableArgUnion::Table2D(vec![vec![TargetSetArgUnion::SetConst(
    //         SetConstPy::from(Set::with_capacity(10)),
    //     )]]);
    //     let t1 = model.add_set_table(table, None, None, None);
    //     assert!(t1.is_ok());
    //     let t1 = t1.unwrap();
    //     let t1 = match t1 {
    //         SetTableUnion::Table2D(t) => t,
    //         _ => panic!("expected SetTableUnion::Table2D but `{:?}`", t1),
    //     };
    //     let snapshot = model.clone();
    //     let table = vec![vec![vec![
    //         SetConstPy::from(Set::with_capacity(10)),
    //         SetConstPy::from({
    //             let mut set = Set::with_capacity(10);
    //             set.insert(0);
    //             set.insert(1);
    //             set.insert(2);
    //             set
    //         }),
    //     ]]];
    //     let result = Python::with_gil(|py| {
    //         let table = table.into_py(py);
    //         model.update_table(
    //             TableUnion::Set(SetTableUnion::Table2D(t1.clone())),
    //             table.as_ref(py),
    //             None,
    //         )
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_set_table_2d_no_table_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();

    //     let table = SetTableArgUnion::Table2D(vec![vec![TargetSetArgUnion::SetConst(
    //         SetConstPy::from(Set::with_capacity(10)),
    //     )]]);
    //     let t1 = model.add_set_table(table, None, None, None);
    //     assert!(t1.is_ok());
    //     let t1 = t1.unwrap();
    //     let t1 = match t1 {
    //         SetTableUnion::Table2D(t) => t,
    //         _ => panic!("expected SetTableUnion::Table2D but `{:?}`", t1),
    //     };
    //     let table = vec![vec![
    //         SetConstPy::from(Set::with_capacity(10)),
    //         SetConstPy::from({
    //             let mut set = Set::with_capacity(10);
    //             set.insert(0);
    //             set.insert(1);
    //             set.insert(2);
    //             set
    //         }),
    //     ]];
    //     let mut model = ModelPy::default();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let table = table.into_py(py);
    //         model.update_table(
    //             TableUnion::Set(SetTableUnion::Table2D(t1.clone())),
    //             table.as_ref(py),
    //             None,
    //         )
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_set_table_2d_from_list_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let ob = model.add_object_type(10, None);
    //     assert!(ob.is_ok());

    //     let table = SetTableArgUnion::Table2D(vec![vec![TargetSetArgUnion::SetConst(
    //         SetConstPy::from(Set::with_capacity(10)),
    //     )]]);
    //     let t1 = model.add_set_table(table, None, None, None);
    //     assert!(t1.is_ok());
    //     let t1 = t1.unwrap();
    //     let t1 = match t1 {
    //         SetTableUnion::Table2D(t) => t,
    //         _ => panic!("expected SetTableUnion::Table2D but `{:?}`", t1),
    //     };
    //     let result = Python::with_gil(|py| {
    //         let table = vec![vec![vec![], vec![0, 1, 2]]].into_py(py);
    //         model.update_table(
    //             TableUnion::Set(SetTableUnion::Table2D(t1.clone())),
    //             table.as_ref(py),
    //             None,
    //         )
    //     });
    //     assert!(result.is_ok());
    //     let t_id = Table2DHandle::<Set>::from(t1).id();
    //     assert_eq!(
    //         model.0.table_registry.set_tables.tables_2d[t_id],
    //         dypdl::Table2D::new(vec![vec![Set::with_capacity(10), {
    //             let mut set = Set::with_capacity(10);
    //             set.insert(0);
    //             set.insert(1);
    //             set.insert(2);
    //             set
    //         }]])
    //     );
    // }

    // #[test]
    // fn update_set_table_2d_from_list_out_of_bound_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let ob = model.add_object_type(10, None);
    //     assert!(ob.is_ok());

    //     let table = SetTableArgUnion::Table2D(vec![vec![TargetSetArgUnion::SetConst(
    //         SetConstPy::from(Set::with_capacity(10)),
    //     )]]);
    //     let t1 = model.add_set_table(table, None, None, None);
    //     assert!(t1.is_ok());
    //     let t1 = t1.unwrap();
    //     let t1 = match t1 {
    //         SetTableUnion::Table2D(t) => t,
    //         _ => panic!("expected SetTableUnion::Table2D but `{:?}`", t1),
    //     };
    //     let result = Python::with_gil(|py| {
    //         let table = vec![vec![vec![0, 1, 2, 10]]].into_py(py);
    //         model.update_table(
    //             TableUnion::Set(SetTableUnion::Table2D(t1.clone())),
    //             table.as_ref(py),
    //             None,
    //         )
    //     });
    //     assert!(result.is_err());
    // }

    // #[test]
    // fn update_set_table_2d_from_set_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let ob = model.add_object_type(10, None);
    //     assert!(ob.is_ok());

    //     let table = SetTableArgUnion::Table2D(vec![vec![TargetSetArgUnion::SetConst(
    //         SetConstPy::from(Set::with_capacity(10)),
    //     )]]);
    //     let t1 = model.add_set_table(table, None, None, None);
    //     assert!(t1.is_ok());
    //     let t1 = t1.unwrap();
    //     let t1 = match t1 {
    //         SetTableUnion::Table2D(t) => t,
    //         _ => panic!("expected SetTableUnion::Table2D but `{:?}`", t1),
    //     };
    //     let result = Python::with_gil(|py| {
    //         let table = vec![vec![HashSet::new(), {
    //             let mut set = HashSet::new();
    //             set.insert(0);
    //             set.insert(1);
    //             set.insert(2);
    //             set
    //         }]]
    //         .into_py(py);
    //         model.update_table(
    //             TableUnion::Set(SetTableUnion::Table2D(t1.clone())),
    //             table.as_ref(py),
    //             None,
    //         )
    //     });
    //     assert!(result.is_ok());
    //     let t_id = Table2DHandle::<Set>::from(t1).id();
    //     assert_eq!(
    //         model.0.table_registry.set_tables.tables_2d[t_id],
    //         dypdl::Table2D::new(vec![vec![Set::with_capacity(10), {
    //             let mut set = Set::with_capacity(10);
    //             set.insert(0);
    //             set.insert(1);
    //             set.insert(2);
    //             set
    //         }]])
    //     );
    // }

    // #[test]
    // fn update_set_table_2d_from_set_out_of_bound_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let ob = model.add_object_type(10, None);
    //     assert!(ob.is_ok());

    //     let table = SetTableArgUnion::Table2D(vec![vec![TargetSetArgUnion::SetConst(
    //         SetConstPy::from(Set::with_capacity(10)),
    //     )]]);
    //     let t1 = model.add_set_table(table, None, None, None);
    //     assert!(t1.is_ok());
    //     let t1 = t1.unwrap();
    //     let t1 = match t1 {
    //         SetTableUnion::Table2D(t) => t,
    //         _ => panic!("expected SetTableUnion::Table2D but `{:?}`", t1),
    //     };
    //     let result = Python::with_gil(|py| {
    //         let table = vec![vec![HashSet::new(), {
    //             let mut set = HashSet::new();
    //             set.insert(0);
    //             set.insert(1);
    //             set.insert(2);
    //             set.insert(10);
    //             set
    //         }]]
    //         .into_py(py);
    //         model.update_table(
    //             TableUnion::Set(SetTableUnion::Table2D(t1.clone())),
    //             table.as_ref(py),
    //             None,
    //         )
    //     });
    //     assert!(result.is_err());
    // }

    #[test]
    fn add_set_table_3d_from_set_const_ok() {
        let mut model = ModelPy::default();

        let table = SetTableArgUnion::Table3D(vec![vec![vec![
            TargetSetArgUnion::SetConst(SetConstPy::from(Set::with_capacity(10))),
            TargetSetArgUnion::SetConst(SetConstPy::from({
                let mut set = Set::with_capacity(10);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set
            })),
        ]]]);
        let t1 = model.add_set_table(table, None, None, None);
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        let t1 = match t1 {
            SetTableUnion::Table3D(t) => t,
            _ => panic!("expected SetTableUnion::Table3D but `{:?}`", t1),
        };
        assert_eq!(t1.get_capacity_of_set(), 10);
        let t_id = Table3DHandle::<Set>::from(t1).id();
        assert_eq!(
            model.0.table_registry.set_tables.tables_3d[t_id],
            dypdl::Table3D::new(vec![vec![vec![Set::with_capacity(10), {
                let mut set = Set::with_capacity(10);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set
            }]]])
        );

        let table = SetTableArgUnion::Table3D(vec![vec![vec![
            TargetSetArgUnion::SetConst(SetConstPy::from(Set::with_capacity(10))),
            TargetSetArgUnion::SetConst(SetConstPy::from({
                let mut set = Set::with_capacity(10);
                set.insert(3);
                set.insert(4);
                set.insert(5);
                set
            })),
        ]]]);
        let t2 = model.add_set_table(table, None, None, None);
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        let t2 = match t2 {
            SetTableUnion::Table3D(t) => t,
            _ => panic!("expected SetTableUnion::Table3D but `{:?}`", t2),
        };
        assert_eq!(t2.get_capacity_of_set(), 10);
        let t_id = Table3DHandle::<Set>::from(t2).id();
        assert_eq!(
            model.0.table_registry.set_tables.tables_3d[t_id],
            dypdl::Table3D::new(vec![vec![vec![Set::with_capacity(10), {
                let mut set = Set::with_capacity(10);
                set.insert(3);
                set.insert(4);
                set.insert(5);
                set
            }]]])
        );
    }

    #[test]
    fn add_set_table_3d_from_set_const_with_name_ok() {
        let mut model = ModelPy::default();

        let table = SetTableArgUnion::Table3D(vec![vec![vec![
            TargetSetArgUnion::SetConst(SetConstPy::from(Set::with_capacity(10))),
            TargetSetArgUnion::SetConst(SetConstPy::from({
                let mut set = Set::with_capacity(10);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set
            })),
        ]]]);
        let t1 = model.add_set_table(table, None, Some("t1"), None);
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        let t1 = match t1 {
            SetTableUnion::Table3D(t) => t,
            _ => panic!("expected SetTableUnion::Table3D but `{:?}`", t1),
        };
        assert_eq!(t1.get_capacity_of_set(), 10);
        let t_id = Table3DHandle::<Set>::from(t1).id();
        assert_eq!(
            model.0.table_registry.set_tables.tables_3d[t_id],
            dypdl::Table3D::new(vec![vec![vec![Set::with_capacity(10), {
                let mut set = Set::with_capacity(10);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set
            }]]])
        );
        assert_eq!(
            model.0.table_registry.set_tables.name_to_table_3d.get("t1"),
            Some(&t_id)
        );

        let table = SetTableArgUnion::Table3D(vec![vec![vec![
            TargetSetArgUnion::SetConst(SetConstPy::from(Set::with_capacity(10))),
            TargetSetArgUnion::SetConst(SetConstPy::from({
                let mut set = Set::with_capacity(10);
                set.insert(3);
                set.insert(4);
                set.insert(5);
                set
            })),
        ]]]);
        let t2 = model.add_set_table(table, None, Some("t2"), None);
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        let t2 = match t2 {
            SetTableUnion::Table3D(t) => t,
            _ => panic!("expected SetTableUnion::Table3D but `{:?}`", t2),
        };
        assert_eq!(t2.get_capacity_of_set(), 10);
        let t_id = Table3DHandle::<Set>::from(t2).id();
        assert_eq!(
            model.0.table_registry.set_tables.tables_3d[t_id],
            dypdl::Table3D::new(vec![vec![vec![Set::with_capacity(10), {
                let mut set = Set::with_capacity(10);
                set.insert(3);
                set.insert(4);
                set.insert(5);
                set
            }]]])
        );
        assert_eq!(
            model.0.table_registry.set_tables.name_to_table_3d.get("t2"),
            Some(&t_id)
        );
    }

    #[test]
    fn add_set_table_3d_from_set_const_duplicate_err() {
        let mut model = ModelPy::default();
        let table = SetTableArgUnion::Table3D(vec![vec![vec![
            TargetSetArgUnion::SetConst(SetConstPy::from(Set::with_capacity(10))),
            TargetSetArgUnion::SetConst(SetConstPy::from({
                let mut set = Set::with_capacity(10);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set
            })),
        ]]]);
        let t = model.add_set_table(table, None, Some("t1"), None);
        assert!(t.is_ok());
        let table = SetTableArgUnion::Table3D(vec![vec![vec![
            TargetSetArgUnion::SetConst(SetConstPy::from(Set::with_capacity(10))),
            TargetSetArgUnion::SetConst(SetConstPy::from({
                let mut set = Set::with_capacity(10);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set
            })),
        ]]]);
        let snapshot = model.clone();
        let t = model.add_set_table(table, None, Some("t1"), None);
        assert!(t.is_err());
        assert_eq!(model, snapshot);
    }

    #[test]
    fn add_set_table_3d_from_list_ok() {
        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());

        let table = SetTableArgUnion::Table3D(vec![vec![vec![TargetSetArgUnion::SetConst(
            SetConstPy::from({
                let mut set = Set::with_capacity(10);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set
            }),
        )]]]);
        let t1 = model.add_set_table(table, None, None, None);
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        let t1 = match t1 {
            SetTableUnion::Table3D(t) => t,
            _ => panic!("expected SetTableUnion::Table3D but `{:?}`", t1),
        };
        assert_eq!(t1.get_capacity_of_set(), 10);

        let table = SetTableArgUnion::Table3D(vec![vec![vec![TargetSetArgUnion::SetConst(
            SetConstPy::from({
                let mut set = Set::with_capacity(10);
                set.insert(3);
                set.insert(4);
                set.insert(5);
                set
            }),
        )]]]);
        let t2 = model.add_set_table(table, None, None, None);
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        let t2 = match t2 {
            SetTableUnion::Table3D(t) => t,
            _ => panic!("expected SetTableUnion::Table3D but `{:?}`", t2),
        };
        assert_eq!(t2.get_capacity_of_set(), 10);

        assert_ne!(t1, t2);

        let mut model2 = ModelPy::default();
        let ob = model2.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let table = SetTableArgUnion::Table3D(vec![vec![vec![TargetSetArgUnion::CreateSetArg(
            CreateSetArgUnion::List(vec![0, 1, 2]),
        )]]]);
        let t1 = model2.add_set_table(table, None, None, Some(ob));
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        let t1 = match t1 {
            SetTableUnion::Table3D(t) => t,
            _ => panic!("expected SetTableUnion::Table3D but `{:?}`", t1),
        };
        assert_eq!(t1.get_capacity_of_set(), 10);

        let table = SetTableArgUnion::Table3D(vec![vec![vec![TargetSetArgUnion::CreateSetArg(
            CreateSetArgUnion::List(vec![3, 4, 5]),
        )]]]);
        let t2 = model2.add_set_table(table, None, None, Some(ob));
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        let t2 = match t2 {
            SetTableUnion::Table3D(t) => t,
            _ => panic!("expected SetTableUnion::Table3D but `{:?}`", t2),
        };
        assert_eq!(t2.get_capacity_of_set(), 10);

        assert_ne!(t1, t2);

        assert_eq!(model2, model);
    }

    #[test]
    fn add_set_table_3d_from_list_with_name_ok() {
        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());

        let table = SetTableArgUnion::Table3D(vec![vec![vec![TargetSetArgUnion::SetConst(
            SetConstPy::from({
                let mut set = Set::with_capacity(10);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set
            }),
        )]]]);
        let t1 = model.add_set_table(table, None, Some("t1"), None);
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        let t1 = match t1 {
            SetTableUnion::Table3D(t) => t,
            _ => panic!("expected SetTableUnion::Table3D but `{:?}`", t1),
        };
        assert_eq!(t1.get_capacity_of_set(), 10);

        let table = SetTableArgUnion::Table3D(vec![vec![vec![TargetSetArgUnion::SetConst(
            SetConstPy::from({
                let mut set = Set::with_capacity(10);
                set.insert(3);
                set.insert(4);
                set.insert(5);
                set
            }),
        )]]]);
        let t2 = model.add_set_table(table, None, Some("t2"), None);
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        let t2 = match t2 {
            SetTableUnion::Table3D(t) => t,
            _ => panic!("expected SetTableUnion::Table3D but `{:?}`", t2),
        };
        assert_eq!(t2.get_capacity_of_set(), 10);

        assert_ne!(t1, t2);

        let mut model2 = ModelPy::default();
        let ob = model2.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let table = SetTableArgUnion::Table3D(vec![vec![vec![TargetSetArgUnion::CreateSetArg(
            CreateSetArgUnion::List(vec![0, 1, 2]),
        )]]]);
        let t1 = model2.add_set_table(table, None, Some("t1"), Some(ob));
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        let t1 = match t1 {
            SetTableUnion::Table3D(t) => t,
            _ => panic!("expected SetTableUnion::Table3D but `{:?}`", t1),
        };
        assert_eq!(t1.get_capacity_of_set(), 10);

        let table = SetTableArgUnion::Table3D(vec![vec![vec![TargetSetArgUnion::CreateSetArg(
            CreateSetArgUnion::List(vec![3, 4, 5]),
        )]]]);
        let t2 = model2.add_set_table(table, None, Some("t2"), Some(ob));
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        let t2 = match t2 {
            SetTableUnion::Table3D(t) => t,
            _ => panic!("expected SetTableUnion::Table3D but `{:?}`", t2),
        };
        assert_eq!(t2.get_capacity_of_set(), 10);

        assert_ne!(t1, t2);

        assert_eq!(model2, model);
    }

    #[test]
    fn add_set_table_3d_from_list_no_object_err() {
        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let table = SetTableArgUnion::Table3D(vec![vec![vec![TargetSetArgUnion::CreateSetArg(
            CreateSetArgUnion::List(vec![0, 1, 2]),
        )]]]);
        let mut model = ModelPy::default();
        let snapshot = model.clone();
        let t = model.add_set_table(table, None, None, Some(ob));
        assert!(t.is_err());
        assert_eq!(model, snapshot);
    }

    #[test]
    fn add_set_table_3d_from_list_none_object_err() {
        let mut model = ModelPy::default();

        let table = SetTableArgUnion::Table3D(vec![vec![vec![TargetSetArgUnion::CreateSetArg(
            CreateSetArgUnion::List(vec![0, 1, 2]),
        )]]]);
        let snapshot = model.clone();
        let t = model.add_set_table(table, None, None, None);
        assert!(t.is_err());
        assert_eq!(model, snapshot);
    }

    #[test]
    fn add_set_table_3d_from_list_table_err() {
        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let table = SetTableArgUnion::Table3D(vec![vec![vec![TargetSetArgUnion::CreateSetArg(
            CreateSetArgUnion::List(vec![0, 1, 2, 10]),
        )]]]);
        let snapshot = model.clone();
        let t = model.add_set_table(table, None, None, Some(ob));
        assert!(t.is_err());
        assert_eq!(model, snapshot);
    }

    #[test]
    fn add_set_table_3d_from_set_ok() {
        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());

        let table = SetTableArgUnion::Table3D(vec![vec![vec![TargetSetArgUnion::SetConst(
            SetConstPy::from({
                let mut set = Set::with_capacity(10);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set
            }),
        )]]]);
        let t1 = model.add_set_table(table, None, None, None);
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        let t1 = match t1 {
            SetTableUnion::Table3D(t) => t,
            _ => panic!("expected SetTableUnion::Table3D but `{:?}`", t1),
        };
        assert_eq!(t1.get_capacity_of_set(), 10);

        let table = SetTableArgUnion::Table3D(vec![vec![vec![TargetSetArgUnion::SetConst(
            SetConstPy::from({
                let mut set = Set::with_capacity(10);
                set.insert(3);
                set.insert(4);
                set.insert(5);
                set
            }),
        )]]]);
        let t2 = model.add_set_table(table, None, None, None);
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        let t2 = match t2 {
            SetTableUnion::Table3D(t) => t,
            _ => panic!("expected SetTableUnion::Table3D but `{:?}`", t2),
        };
        assert_eq!(t2.get_capacity_of_set(), 10);

        assert_ne!(t1, t2);

        let mut model2 = ModelPy::default();
        let ob = model2.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let table = SetTableArgUnion::Table3D(vec![vec![vec![TargetSetArgUnion::SetConst(
            SetConstPy::from({
                let mut set = Set::with_capacity(10);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set
            }),
        )]]]);
        let t1 = model2.add_set_table(table, None, None, Some(ob));
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        let t1 = match t1 {
            SetTableUnion::Table3D(t) => t,
            _ => panic!("expected SetTableUnion::Table3D but `{:?}`", t1),
        };
        assert_eq!(t1.get_capacity_of_set(), 10);

        let table = SetTableArgUnion::Table3D(vec![vec![vec![TargetSetArgUnion::SetConst(
            SetConstPy::from({
                let mut set = Set::with_capacity(10);
                set.insert(3);
                set.insert(4);
                set.insert(5);
                set
            }),
        )]]]);
        let t2 = model2.add_set_table(table, None, None, Some(ob));
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        let t2 = match t2 {
            SetTableUnion::Table3D(t) => t,
            _ => panic!("expected SetTableUnion::Table3D but `{:?}`", t2),
        };
        assert_eq!(t2.get_capacity_of_set(), 10);

        assert_ne!(t1, t2);

        assert_eq!(model2, model);
    }

    #[test]
    fn add_set_table_3d_from_set_with_name_ok() {
        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());

        let table = SetTableArgUnion::Table3D(vec![vec![vec![TargetSetArgUnion::SetConst(
            SetConstPy::from({
                let mut set = Set::with_capacity(10);
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set
            }),
        )]]]);
        let t1 = model.add_set_table(table, None, Some("t1"), None);
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        let t1 = match t1 {
            SetTableUnion::Table3D(t) => t,
            _ => panic!("expected SetTableUnion::Table3D but `{:?}`", t1),
        };
        assert_eq!(t1.get_capacity_of_set(), 10);

        let table = SetTableArgUnion::Table3D(vec![vec![vec![TargetSetArgUnion::SetConst(
            SetConstPy::from({
                let mut set = Set::with_capacity(10);
                set.insert(3);
                set.insert(4);
                set.insert(5);
                set
            }),
        )]]]);
        let t2 = model.add_set_table(table, None, Some("t2"), None);
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        let t2 = match t2 {
            SetTableUnion::Table3D(t) => t,
            _ => panic!("expected SetTableUnion::Table3D but `{:?}`", t2),
        };
        assert_eq!(t2.get_capacity_of_set(), 10);

        assert_ne!(t1, t2);

        let mut model2 = ModelPy::default();
        let ob = model2.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let table = SetTableArgUnion::Table3D(vec![vec![vec![TargetSetArgUnion::CreateSetArg(
            CreateSetArgUnion::Set({
                let mut set = HashSet::new();
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set
            }),
        )]]]);
        let t1 = model2.add_set_table(table, None, Some("t1"), Some(ob));
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        let t1 = match t1 {
            SetTableUnion::Table3D(t) => t,
            _ => panic!("expected SetTableUnion::Table3D but `{:?}`", t1),
        };
        assert_eq!(t1.get_capacity_of_set(), 10);

        let table = SetTableArgUnion::Table3D(vec![vec![vec![TargetSetArgUnion::CreateSetArg(
            CreateSetArgUnion::Set({
                let mut set = HashSet::new();
                set.insert(3);
                set.insert(4);
                set.insert(5);
                set
            }),
        )]]]);
        let t2 = model2.add_set_table(table, None, Some("t2"), Some(ob));
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        let t2 = match t2 {
            SetTableUnion::Table3D(t) => t,
            _ => panic!("expected SetTableUnion::Table3D but `{:?}`", t2),
        };
        assert_eq!(t2.get_capacity_of_set(), 10);

        assert_ne!(t1, t2);

        assert_eq!(model2, model);
    }

    #[test]
    fn add_set_table_3d_from_set_no_object_err() {
        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let table = SetTableArgUnion::Table3D(vec![vec![vec![TargetSetArgUnion::CreateSetArg(
            CreateSetArgUnion::Set({
                let mut set = HashSet::new();
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set
            }),
        )]]]);
        let mut model = ModelPy::default();
        let snapshot = model.clone();
        let t = model.add_set_table(table, None, None, Some(ob));
        assert!(t.is_err());
        assert_eq!(model, snapshot);
    }

    #[test]
    fn add_set_table_3d_from_set_none_object_err() {
        let mut model = ModelPy::default();

        let table = SetTableArgUnion::Table3D(vec![vec![vec![TargetSetArgUnion::CreateSetArg(
            CreateSetArgUnion::Set({
                let mut set = HashSet::new();
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set
            }),
        )]]]);
        let snapshot = model.clone();
        let t = model.add_set_table(table, None, None, None);
        assert!(t.is_err());
        assert_eq!(model, snapshot);
    }

    #[test]
    fn add_set_table_3d_from_set_table_err() {
        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let table = SetTableArgUnion::Table3D(vec![vec![vec![TargetSetArgUnion::CreateSetArg(
            CreateSetArgUnion::Set({
                let mut set = HashSet::new();
                set.insert(0);
                set.insert(1);
                set.insert(2);
                set.insert(10);
                set
            }),
        )]]]);
        let snapshot = model.clone();
        let t = model.add_set_table(table, None, None, Some(ob));
        assert!(t.is_err());
        assert_eq!(model, snapshot);
    }

    // #[test]
    // fn set_set_table_3d_item_from_set_const_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = SetTableArgUnion::Table3D(vec![vec![vec![TargetSetArgUnion::SetConst(
    //         SetConstPy::from(Set::with_capacity(10)),
    //     )]]]);
    //     let t = model.add_set_table(table, None, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let index = TableIndexUnion::Table3D((0, 0, 0));
    //     let result = Python::with_gil(|py| {
    //         let value = SetConstPy::from({
    //             let mut set = Set::with_capacity(10);
    //             set.insert(1);
    //             set
    //         })
    //         .into_py(py);
    //         model.set_table_item(TableUnion::Set(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_ok());
    // }

    // #[test]
    // fn set_set_table_3d_item_value_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = SetTableArgUnion::Table3D(vec![vec![vec![TargetSetArgUnion::SetConst(
    //         SetConstPy::from(Set::with_capacity(10)),
    //     )]]]);
    //     let t = model.add_set_table(table, None, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let index = TableIndexUnion::Table3D((0, 0, 0));
    //     let mut model = ModelPy::default();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = 1usize.into_py(py);
    //         model.set_table_item(TableUnion::Set(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn set_set_table_3d_item_no_table_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = SetTableArgUnion::Table3D(vec![vec![vec![TargetSetArgUnion::SetConst(
    //         SetConstPy::from(Set::with_capacity(10)),
    //     )]]]);
    //     let t = model.add_set_table(table, None, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let index = TableIndexUnion::Table3D((0, 0, 0));
    //     let mut model = ModelPy::default();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = SetConstPy::from({
    //             let mut set = Set::with_capacity(10);
    //             set.insert(1);
    //             set
    //         })
    //         .into_py(py);
    //         model.set_table_item(TableUnion::Set(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn set_set_table_3d_item_from_list_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let ob = model.add_object_type(10, None);
    //     assert!(ob.is_ok());
    //     let table = SetTableArgUnion::Table3D(vec![vec![vec![TargetSetArgUnion::SetConst(
    //         SetConstPy::from(Set::with_capacity(10)),
    //     )]]]);
    //     let t = model.add_set_table(table, None, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let index = TableIndexUnion::Table3D((0, 0, 0));
    //     let result = Python::with_gil(|py| {
    //         let value = vec![1].into_py(py);
    //         model.set_table_item(TableUnion::Set(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_ok());
    // }

    // #[test]
    // fn set_set_table_3d_item_from_list_out_of_bound_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let ob = model.add_object_type(10, None);
    //     assert!(ob.is_ok());
    //     let table = SetTableArgUnion::Table3D(vec![vec![vec![TargetSetArgUnion::SetConst(
    //         SetConstPy::from(Set::with_capacity(10)),
    //     )]]]);
    //     let t = model.add_set_table(table, None, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let index = TableIndexUnion::Table3D((0, 0, 0));
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = vec![10].into_py(py);
    //         model.set_table_item(TableUnion::Set(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn set_set_table_3d_item_from_set_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let ob = model.add_object_type(10, None);
    //     assert!(ob.is_ok());
    //     let table = SetTableArgUnion::Table3D(vec![vec![vec![TargetSetArgUnion::SetConst(
    //         SetConstPy::from(Set::with_capacity(10)),
    //     )]]]);
    //     let t = model.add_set_table(table, None, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let index = TableIndexUnion::Table3D((0, 0, 0));
    //     let result = Python::with_gil(|py| {
    //         let value = {
    //             let mut set = HashSet::new();
    //             set.insert(1);
    //             set
    //         }
    //         .into_py(py);
    //         model.set_table_item(TableUnion::Set(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_ok());
    // }

    // #[test]
    // fn set_set_table_3d_item_from_set_out_of_bound_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let ob = model.add_object_type(10, None);
    //     assert!(ob.is_ok());
    //     let table = SetTableArgUnion::Table3D(vec![vec![vec![TargetSetArgUnion::SetConst(
    //         SetConstPy::from(Set::with_capacity(10)),
    //     )]]]);
    //     let t = model.add_set_table(table, None, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let index = TableIndexUnion::Table3D((0, 0, 0));
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = {
    //             let mut set = HashSet::new();
    //             set.insert(10);
    //             set
    //         }
    //         .into_py(py);
    //         model.set_table_item(TableUnion::Set(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_set_table_3d_from_set_const_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();

    //     let table = SetTableArgUnion::Table3D(vec![vec![vec![TargetSetArgUnion::SetConst(
    //         SetConstPy::from(Set::with_capacity(10)),
    //     )]]]);
    //     let t1 = model.add_set_table(table, None, None, None);
    //     assert!(t1.is_ok());
    //     let t1 = t1.unwrap();
    //     let t1 = match t1 {
    //         SetTableUnion::Table3D(t) => t,
    //         _ => panic!("expected SetTableUnion::Table3D but `{:?}`", t1),
    //     };
    //     let table = vec![vec![vec![
    //         SetConstPy::from(Set::with_capacity(10)),
    //         SetConstPy::from({
    //             let mut set = Set::with_capacity(10);
    //             set.insert(0);
    //             set.insert(1);
    //             set.insert(2);
    //             set
    //         }),
    //     ]]];
    //     let result = Python::with_gil(|py| {
    //         let table = table.into_py(py);
    //         model.update_table(
    //             TableUnion::Set(SetTableUnion::Table3D(t1.clone())),
    //             table.as_ref(py),
    //             None,
    //         )
    //     });
    //     assert!(result.is_ok());
    //     let t_id = Table3DHandle::<Set>::from(t1).id();
    //     assert_eq!(
    //         model.0.table_registry.set_tables.tables_3d[t_id],
    //         dypdl::Table3D::new(vec![vec![vec![Set::with_capacity(10), {
    //             let mut set = Set::with_capacity(10);
    //             set.insert(0);
    //             set.insert(1);
    //             set.insert(2);
    //             set
    //         }]]])
    //     );
    // }

    // #[test]
    // fn update_set_table_3d_value_extract_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();

    //     let table = SetTableArgUnion::Table3D(vec![vec![vec![TargetSetArgUnion::SetConst(
    //         SetConstPy::from(Set::with_capacity(10)),
    //     )]]]);
    //     let t1 = model.add_set_table(table, None, None, None);
    //     assert!(t1.is_ok());
    //     let t1 = t1.unwrap();
    //     let t1 = match t1 {
    //         SetTableUnion::Table3D(t) => t,
    //         _ => panic!("expected SetTableUnion::Table3D but `{:?}`", t1),
    //     };
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let table = 1usize.into_py(py);
    //         model.update_table(
    //             TableUnion::Set(SetTableUnion::Table3D(t1.clone())),
    //             table.as_ref(py),
    //             None,
    //         )
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_set_table_3d_dimension_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();

    //     let table = SetTableArgUnion::Table3D(vec![vec![vec![TargetSetArgUnion::SetConst(
    //         SetConstPy::from(Set::with_capacity(10)),
    //     )]]]);
    //     let t1 = model.add_set_table(table, None, None, None);
    //     assert!(t1.is_ok());
    //     let t1 = t1.unwrap();
    //     let t1 = match t1 {
    //         SetTableUnion::Table3D(t) => t,
    //         _ => panic!("expected SetTableUnion::Table3D but `{:?}`", t1),
    //     };
    //     let snapshot = model.clone();
    //     let table = vec![vec![
    //         SetConstPy::from(Set::with_capacity(10)),
    //         SetConstPy::from({
    //             let mut set = Set::with_capacity(10);
    //             set.insert(0);
    //             set.insert(1);
    //             set.insert(2);
    //             set
    //         }),
    //     ]];
    //     let result = Python::with_gil(|py| {
    //         let table = table.into_py(py);
    //         model.update_table(
    //             TableUnion::Set(SetTableUnion::Table3D(t1.clone())),
    //             table.as_ref(py),
    //             None,
    //         )
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_set_table_3d_no_table_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();

    //     let table = SetTableArgUnion::Table3D(vec![vec![vec![TargetSetArgUnion::SetConst(
    //         SetConstPy::from(Set::with_capacity(10)),
    //     )]]]);
    //     let t1 = model.add_set_table(table, None, None, None);
    //     assert!(t1.is_ok());
    //     let t1 = t1.unwrap();
    //     let t1 = match t1 {
    //         SetTableUnion::Table3D(t) => t,
    //         _ => panic!("expected SetTableUnion::Table3D but `{:?}`", t1),
    //     };
    //     let table = vec![vec![vec![
    //         SetConstPy::from(Set::with_capacity(10)),
    //         SetConstPy::from({
    //             let mut set = Set::with_capacity(10);
    //             set.insert(0);
    //             set.insert(1);
    //             set.insert(2);
    //             set
    //         }),
    //     ]]];
    //     let mut model = ModelPy::default();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let table = table.into_py(py);
    //         model.update_table(
    //             TableUnion::Set(SetTableUnion::Table3D(t1.clone())),
    //             table.as_ref(py),
    //             None,
    //         )
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_set_table_3d_from_list_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let ob = model.add_object_type(10, None);
    //     assert!(ob.is_ok());

    //     let table = SetTableArgUnion::Table3D(vec![vec![vec![TargetSetArgUnion::SetConst(
    //         SetConstPy::from(Set::with_capacity(10)),
    //     )]]]);
    //     let t1 = model.add_set_table(table, None, None, None);
    //     assert!(t1.is_ok());
    //     let t1 = t1.unwrap();
    //     let t1 = match t1 {
    //         SetTableUnion::Table3D(t) => t,
    //         _ => panic!("expected SetTableUnion::Table3D but `{:?}`", t1),
    //     };
    //     let result = Python::with_gil(|py| {
    //         let table = vec![vec![vec![vec![], vec![0, 1, 2]]]].into_py(py);
    //         model.update_table(
    //             TableUnion::Set(SetTableUnion::Table3D(t1.clone())),
    //             table.as_ref(py),
    //             None,
    //         )
    //     });
    //     assert!(result.is_ok());
    //     let t_id = Table3DHandle::<Set>::from(t1).id();
    //     assert_eq!(
    //         model.0.table_registry.set_tables.tables_3d[t_id],
    //         dypdl::Table3D::new(vec![vec![vec![Set::with_capacity(10), {
    //             let mut set = Set::with_capacity(10);
    //             set.insert(0);
    //             set.insert(1);
    //             set.insert(2);
    //             set
    //         }]]])
    //     );
    // }

    // #[test]
    // fn update_set_table_3d_from_list_out_of_bound_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let ob = model.add_object_type(10, None);
    //     assert!(ob.is_ok());

    //     let table = SetTableArgUnion::Table3D(vec![vec![vec![TargetSetArgUnion::SetConst(
    //         SetConstPy::from(Set::with_capacity(10)),
    //     )]]]);
    //     let t1 = model.add_set_table(table, None, None, None);
    //     assert!(t1.is_ok());
    //     let t1 = t1.unwrap();
    //     let t1 = match t1 {
    //         SetTableUnion::Table3D(t) => t,
    //         _ => panic!("expected SetTableUnion::Table3D but `{:?}`", t1),
    //     };
    //     let result = Python::with_gil(|py| {
    //         let table = vec![vec![vec![vec![0, 1, 2, 10]]]].into_py(py);
    //         model.update_table(
    //             TableUnion::Set(SetTableUnion::Table3D(t1.clone())),
    //             table.as_ref(py),
    //             None,
    //         )
    //     });
    //     assert!(result.is_err());
    // }

    // #[test]
    // fn update_set_table_3d_from_set_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let ob = model.add_object_type(10, None);
    //     assert!(ob.is_ok());

    //     let table = SetTableArgUnion::Table3D(vec![vec![vec![TargetSetArgUnion::SetConst(
    //         SetConstPy::from(Set::with_capacity(10)),
    //     )]]]);
    //     let t1 = model.add_set_table(table, None, None, None);
    //     assert!(t1.is_ok());
    //     let t1 = t1.unwrap();
    //     let t1 = match t1 {
    //         SetTableUnion::Table3D(t) => t,
    //         _ => panic!("expected SetTableUnion::Table3D but `{:?}`", t1),
    //     };
    //     let result = Python::with_gil(|py| {
    //         let table = vec![vec![vec![HashSet::new(), {
    //             let mut set = HashSet::new();
    //             set.insert(0);
    //             set.insert(1);
    //             set.insert(2);
    //             set
    //         }]]]
    //         .into_py(py);
    //         model.update_table(
    //             TableUnion::Set(SetTableUnion::Table3D(t1.clone())),
    //             table.as_ref(py),
    //             None,
    //         )
    //     });
    //     assert!(result.is_ok());
    //     let t_id = Table3DHandle::<Set>::from(t1).id();
    //     assert_eq!(
    //         model.0.table_registry.set_tables.tables_3d[t_id],
    //         dypdl::Table3D::new(vec![vec![vec![Set::with_capacity(10), {
    //             let mut set = Set::with_capacity(10);
    //             set.insert(0);
    //             set.insert(1);
    //             set.insert(2);
    //             set
    //         }]]])
    //     );
    // }

    // #[test]
    // fn update_set_table_3d_from_set_out_of_bound_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let ob = model.add_object_type(10, None);
    //     assert!(ob.is_ok());

    //     let table = SetTableArgUnion::Table3D(vec![vec![vec![TargetSetArgUnion::SetConst(
    //         SetConstPy::from(Set::with_capacity(10)),
    //     )]]]);
    //     let t1 = model.add_set_table(table, None, None, None);
    //     assert!(t1.is_ok());
    //     let t1 = t1.unwrap();
    //     let t1 = match t1 {
    //         SetTableUnion::Table3D(t) => t,
    //         _ => panic!("expected SetTableUnion::Table3D but `{:?}`", t1),
    //     };
    //     let result = Python::with_gil(|py| {
    //         let table = vec![vec![vec![HashSet::new(), {
    //             let mut set = HashSet::new();
    //             set.insert(0);
    //             set.insert(1);
    //             set.insert(2);
    //             set.insert(10);
    //             set
    //         }]]]
    //         .into_py(py);
    //         model.update_table(
    //             TableUnion::Set(SetTableUnion::Table3D(t1.clone())),
    //             table.as_ref(py),
    //             None,
    //         )
    //     });
    //     assert!(result.is_err());
    // }

    #[test]
    fn add_set_table_from_set_const_ok() {
        let mut model = ModelPy::default();

        let table = SetTableArgUnion::Table({
            let mut map = FxHashMap::default();
            map.insert(
                vec![0, 0, 0, 0],
                TargetSetArgUnion::SetConst(SetConstPy::from({
                    let mut set = Set::with_capacity(10);
                    set.insert(0);
                    set.insert(1);
                    set.insert(2);
                    set
                })),
            );
            map
        });
        let t1 = model.add_set_table(
            table,
            Some(TargetSetArgUnion::SetConst(SetConstPy::from(
                Set::with_capacity(10),
            ))),
            None,
            None,
        );
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        let t1 = match t1 {
            SetTableUnion::Table(t) => t,
            _ => panic!("expected SetTableUnion::Table but `{:?}`", t1),
        };
        assert_eq!(t1.get_capacity_of_set(), 10);
        let t_id = TableHandle::<Set>::from(t1).id();
        assert_eq!(
            model.0.table_registry.set_tables.tables[t_id],
            dypdl::Table::new(
                {
                    let mut map = FxHashMap::default();
                    let mut set = Set::with_capacity(10);
                    set.insert(0);
                    set.insert(1);
                    set.insert(2);
                    map.insert(vec![0, 0, 0, 0], set);
                    map
                },
                Set::with_capacity(10)
            )
        );

        let table = SetTableArgUnion::Table({
            let mut map = FxHashMap::default();
            map.insert(
                vec![0, 0, 0, 0],
                TargetSetArgUnion::SetConst(SetConstPy::from({
                    let mut set = Set::with_capacity(10);
                    set.insert(3);
                    set.insert(4);
                    set.insert(5);
                    set
                })),
            );
            map
        });
        let t2 = model.add_set_table(
            table,
            Some(TargetSetArgUnion::SetConst(SetConstPy::from(
                Set::with_capacity(10),
            ))),
            None,
            None,
        );
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        let t2 = match t2 {
            SetTableUnion::Table(t) => t,
            _ => panic!("expected SetTableUnion::Table but `{:?}`", t2),
        };
        assert_eq!(t2.get_capacity_of_set(), 10);
        let t_id = TableHandle::<Set>::from(t2).id();
        assert_eq!(
            model.0.table_registry.set_tables.tables[t_id],
            dypdl::Table::new(
                {
                    let mut map = FxHashMap::default();
                    let mut set = Set::with_capacity(10);
                    set.insert(3);
                    set.insert(4);
                    set.insert(5);
                    map.insert(vec![0, 0, 0, 0], set);
                    map
                },
                Set::with_capacity(10)
            )
        );
    }

    #[test]
    fn add_set_table_from_set_const_with_name_ok() {
        let mut model = ModelPy::default();

        let table = SetTableArgUnion::Table({
            let mut map = FxHashMap::default();
            map.insert(
                vec![0, 0, 0, 0],
                TargetSetArgUnion::SetConst(SetConstPy::from({
                    let mut set = Set::with_capacity(10);
                    set.insert(0);
                    set.insert(1);
                    set.insert(2);
                    set
                })),
            );
            map
        });
        let t1 = model.add_set_table(
            table,
            Some(TargetSetArgUnion::SetConst(SetConstPy::from(
                Set::with_capacity(10),
            ))),
            Some("t1"),
            None,
        );
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        let t1 = match t1 {
            SetTableUnion::Table(t) => t,
            _ => panic!("expected SetTableUnion::Table but `{:?}`", t1),
        };
        assert_eq!(t1.get_capacity_of_set(), 10);
        let t_id = TableHandle::<Set>::from(t1).id();
        assert_eq!(
            model.0.table_registry.set_tables.tables[t_id],
            dypdl::Table::new(
                {
                    let mut map = FxHashMap::default();
                    let mut set = Set::with_capacity(10);
                    set.insert(0);
                    set.insert(1);
                    set.insert(2);
                    map.insert(vec![0, 0, 0, 0], set);
                    map
                },
                Set::with_capacity(10)
            )
        );

        let table = SetTableArgUnion::Table({
            let mut map = FxHashMap::default();
            map.insert(
                vec![0, 0, 0, 0],
                TargetSetArgUnion::SetConst(SetConstPy::from({
                    let mut set = Set::with_capacity(10);
                    set.insert(3);
                    set.insert(4);
                    set.insert(5);
                    set
                })),
            );
            map
        });
        let t2 = model.add_set_table(
            table,
            Some(TargetSetArgUnion::SetConst(SetConstPy::from(
                Set::with_capacity(10),
            ))),
            Some("t2"),
            None,
        );
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        let t2 = match t2 {
            SetTableUnion::Table(t) => t,
            _ => panic!("expected SetTableUnion::Table but `{:?}`", t2),
        };
        assert_eq!(t2.get_capacity_of_set(), 10);
        let t_id = TableHandle::<Set>::from(t2).id();
        assert_eq!(
            model.0.table_registry.set_tables.tables[t_id],
            dypdl::Table::new(
                {
                    let mut map = FxHashMap::default();
                    let mut set = Set::with_capacity(10);
                    set.insert(3);
                    set.insert(4);
                    set.insert(5);
                    map.insert(vec![0, 0, 0, 0], set);
                    map
                },
                Set::with_capacity(10)
            )
        );
    }

    #[test]
    fn add_set_table_from_set_const_duplicate_err() {
        let mut model = ModelPy::default();
        let table = SetTableArgUnion::Table(FxHashMap::default());
        let t = model.add_set_table(
            table,
            Some(TargetSetArgUnion::SetConst(SetConstPy::from(
                Set::with_capacity(10),
            ))),
            Some("t1"),
            None,
        );
        assert!(t.is_ok());
        let table = SetTableArgUnion::Table(FxHashMap::default());
        let snapshot = model.clone();
        let t = model.add_set_table(
            table,
            Some(TargetSetArgUnion::SetConst(SetConstPy::from(
                Set::with_capacity(10),
            ))),
            Some("t1"),
            None,
        );
        assert!(t.is_err());
        assert_eq!(model, snapshot);
    }

    #[test]
    fn add_set_table_from_set_const_no_default_err() {
        let mut model = ModelPy::default();
        let table = SetTableArgUnion::Table(FxHashMap::default());
        let snapshot = model.clone();
        let t = model.add_set_table(table, None, Some("t1"), None);
        assert!(t.is_err());
        assert_eq!(model, snapshot);
    }

    #[test]
    fn add_set_table_from_list_ok() {
        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());

        let table = SetTableArgUnion::Table({
            let mut map = FxHashMap::default();
            map.insert(
                vec![0, 0, 0, 0],
                TargetSetArgUnion::SetConst(SetConstPy::from({
                    let mut set = Set::with_capacity(10);
                    set.insert(0);
                    set.insert(1);
                    set.insert(2);
                    set
                })),
            );
            map
        });
        let t1 = model.add_set_table(
            table,
            Some(TargetSetArgUnion::SetConst(SetConstPy::from(
                Set::with_capacity(10),
            ))),
            None,
            None,
        );
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        let t1 = match t1 {
            SetTableUnion::Table(t) => t,
            _ => panic!("expected SetTableUnion::Table but `{:?}`", t1),
        };
        assert_eq!(t1.get_capacity_of_set(), 10);

        let table = SetTableArgUnion::Table({
            let mut map = FxHashMap::default();
            map.insert(
                vec![0, 0, 0, 0],
                TargetSetArgUnion::SetConst(SetConstPy::from({
                    let mut set = Set::with_capacity(10);
                    set.insert(3);
                    set.insert(4);
                    set.insert(5);
                    set
                })),
            );
            map
        });
        let t2 = model.add_set_table(
            table,
            Some(TargetSetArgUnion::SetConst(SetConstPy::from(
                Set::with_capacity(10),
            ))),
            None,
            None,
        );
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        let t2 = match t2 {
            SetTableUnion::Table(t) => t,
            _ => panic!("expected SetTableUnion::Table but `{:?}`", t2),
        };
        assert_eq!(t2.get_capacity_of_set(), 10);

        assert_ne!(t1, t2);

        let mut model2 = ModelPy::default();
        let ob = model2.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let table = SetTableArgUnion::Table({
            let mut map = FxHashMap::default();
            map.insert(
                vec![0, 0, 0, 0],
                TargetSetArgUnion::CreateSetArg(CreateSetArgUnion::List(vec![0, 1, 2])),
            );
            map
        });
        let t1 = model2.add_set_table(
            table,
            Some(TargetSetArgUnion::SetConst(SetConstPy::from(
                Set::with_capacity(10),
            ))),
            None,
            Some(ob),
        );
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        let t1 = match t1 {
            SetTableUnion::Table(t) => t,
            _ => panic!("expected SetTableUnion::Table but `{:?}`", t1),
        };
        assert_eq!(t1.get_capacity_of_set(), 10);

        let table = SetTableArgUnion::Table({
            let mut map = FxHashMap::default();
            map.insert(
                vec![0, 0, 0, 0],
                TargetSetArgUnion::CreateSetArg(CreateSetArgUnion::List(vec![3, 4, 5])),
            );
            map
        });
        let t2 = model2.add_set_table(
            table,
            Some(TargetSetArgUnion::SetConst(SetConstPy::from(
                Set::with_capacity(10),
            ))),
            None,
            Some(ob),
        );
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        let t2 = match t2 {
            SetTableUnion::Table(t) => t,
            _ => panic!("expected SetTableUnion::Table but `{:?}`", t2),
        };
        assert_eq!(t2.get_capacity_of_set(), 10);

        assert_ne!(t1, t2);

        assert_eq!(model, model2);
    }

    #[test]
    fn add_set_table_from_list_with_name_ok() {
        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());

        let table = SetTableArgUnion::Table({
            let mut map = FxHashMap::default();
            map.insert(
                vec![0, 0, 0, 0],
                TargetSetArgUnion::SetConst(SetConstPy::from({
                    let mut set = Set::with_capacity(10);
                    set.insert(0);
                    set.insert(1);
                    set.insert(2);
                    set
                })),
            );
            map
        });
        let t1 = model.add_set_table(
            table,
            Some(TargetSetArgUnion::SetConst(SetConstPy::from(
                Set::with_capacity(10),
            ))),
            Some("t1"),
            None,
        );
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        let t1 = match t1 {
            SetTableUnion::Table(t) => t,
            _ => panic!("expected SetTableUnion::Table but `{:?}`", t1),
        };
        assert_eq!(t1.get_capacity_of_set(), 10);

        let table = SetTableArgUnion::Table({
            let mut map = FxHashMap::default();
            map.insert(
                vec![0, 0, 0, 0],
                TargetSetArgUnion::SetConst(SetConstPy::from({
                    let mut set = Set::with_capacity(10);
                    set.insert(3);
                    set.insert(4);
                    set.insert(5);
                    set
                })),
            );
            map
        });
        let t2 = model.add_set_table(
            table,
            Some(TargetSetArgUnion::SetConst(SetConstPy::from(
                Set::with_capacity(10),
            ))),
            Some("t2"),
            None,
        );
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        let t2 = match t2 {
            SetTableUnion::Table(t) => t,
            _ => panic!("expected SetTableUnion::Table but `{:?}`", t2),
        };
        assert_eq!(t2.get_capacity_of_set(), 10);

        assert_ne!(t1, t2);

        let mut model2 = ModelPy::default();
        let ob = model2.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let table = SetTableArgUnion::Table({
            let mut map = FxHashMap::default();
            map.insert(
                vec![0, 0, 0, 0],
                TargetSetArgUnion::CreateSetArg(CreateSetArgUnion::List(vec![0, 1, 2])),
            );
            map
        });
        let t1 = model2.add_set_table(
            table,
            Some(TargetSetArgUnion::SetConst(SetConstPy::from(
                Set::with_capacity(10),
            ))),
            Some("t1"),
            Some(ob),
        );
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        let t1 = match t1 {
            SetTableUnion::Table(t) => t,
            _ => panic!("expected SetTableUnion::Table but `{:?}`", t1),
        };
        assert_eq!(t1.get_capacity_of_set(), 10);

        let table = SetTableArgUnion::Table({
            let mut map = FxHashMap::default();
            map.insert(
                vec![0, 0, 0, 0],
                TargetSetArgUnion::CreateSetArg(CreateSetArgUnion::List(vec![3, 4, 5])),
            );
            map
        });
        let t2 = model2.add_set_table(
            table,
            Some(TargetSetArgUnion::SetConst(SetConstPy::from(
                Set::with_capacity(10),
            ))),
            Some("t2"),
            Some(ob),
        );
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        let t2 = match t2 {
            SetTableUnion::Table(t) => t,
            _ => panic!("expected SetTableUnion::Table but `{:?}`", t2),
        };
        assert_eq!(t2.get_capacity_of_set(), 10);

        assert_ne!(t1, t2);

        assert_eq!(model, model2);
    }

    #[test]
    fn add_set_table_from_list_out_of_bound_err() {
        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let table = SetTableArgUnion::Table({
            let mut map = FxHashMap::default();
            map.insert(
                vec![0, 0, 0, 0],
                TargetSetArgUnion::CreateSetArg(CreateSetArgUnion::List(vec![1, 2, 3, 10])),
            );
            map
        });
        let snapshot = model.clone();
        let t = model.add_set_table(table, None, None, Some(ob));
        assert!(t.is_err());
        assert_eq!(model, snapshot);
    }

    #[test]
    fn add_set_table_from_list_default_out_of_bound_err() {
        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let table = SetTableArgUnion::Table({
            let mut map = FxHashMap::default();
            map.insert(
                vec![0, 0, 0, 0],
                TargetSetArgUnion::CreateSetArg(CreateSetArgUnion::List(vec![1, 2, 3, 10])),
            );
            map
        });
        let default = TargetSetArgUnion::CreateSetArg(CreateSetArgUnion::List(vec![1, 2, 3]));
        let snapshot = model.clone();
        let t = model.add_set_table(table, Some(default), None, Some(ob));
        assert!(t.is_err());
        assert_eq!(model, snapshot);
    }

    #[test]
    fn add_set_table_from_list_no_object_err() {
        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let mut model = ModelPy::default();
        let table = SetTableArgUnion::Table({
            let mut map = FxHashMap::default();
            map.insert(
                vec![0, 0, 0, 0],
                TargetSetArgUnion::CreateSetArg(CreateSetArgUnion::List(vec![1, 2, 3])),
            );
            map
        });
        let default = TargetSetArgUnion::CreateSetArg(CreateSetArgUnion::List(vec![1, 2, 3]));
        let snapshot = model.clone();
        let t = model.add_set_table(table, Some(default), None, Some(ob));
        assert!(t.is_err());
        assert_eq!(model, snapshot);
    }

    #[test]
    fn add_set_table_from_list_none_object_err() {
        let mut model = ModelPy::default();
        let table = SetTableArgUnion::Table(FxHashMap::default());
        let default = TargetSetArgUnion::CreateSetArg(CreateSetArgUnion::List(vec![1, 2, 3]));
        let snapshot = model.clone();
        let t = model.add_set_table(table, Some(default), None, None);
        assert!(t.is_err());
        assert_eq!(model, snapshot);
    }

    #[test]
    fn add_set_table_from_set_ok() {
        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());

        let table = SetTableArgUnion::Table({
            let mut map = FxHashMap::default();
            map.insert(
                vec![0, 0, 0, 0],
                TargetSetArgUnion::SetConst(SetConstPy::from({
                    let mut set = Set::with_capacity(10);
                    set.insert(0);
                    set.insert(1);
                    set.insert(2);
                    set
                })),
            );
            map
        });
        let t1 = model.add_set_table(
            table,
            Some(TargetSetArgUnion::SetConst(SetConstPy::from(
                Set::with_capacity(10),
            ))),
            None,
            None,
        );
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        let t1 = match t1 {
            SetTableUnion::Table(t) => t,
            _ => panic!("expected SetTableUnion::Table but `{:?}`", t1),
        };
        assert_eq!(t1.get_capacity_of_set(), 10);

        let table = SetTableArgUnion::Table({
            let mut map = FxHashMap::default();
            map.insert(
                vec![0, 0, 0, 0],
                TargetSetArgUnion::SetConst(SetConstPy::from({
                    let mut set = Set::with_capacity(10);
                    set.insert(3);
                    set.insert(4);
                    set.insert(5);
                    set
                })),
            );
            map
        });
        let t2 = model.add_set_table(
            table,
            Some(TargetSetArgUnion::SetConst(SetConstPy::from(
                Set::with_capacity(10),
            ))),
            None,
            None,
        );
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        let t2 = match t2 {
            SetTableUnion::Table(t) => t,
            _ => panic!("expected SetTableUnion::Table but `{:?}`", t2),
        };
        assert_eq!(t2.get_capacity_of_set(), 10);

        assert_ne!(t1, t2);

        let mut model2 = ModelPy::default();
        let ob = model2.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let table = SetTableArgUnion::Table({
            let mut map = FxHashMap::default();
            map.insert(
                vec![0, 0, 0, 0],
                TargetSetArgUnion::CreateSetArg(CreateSetArgUnion::Set({
                    let mut set = HashSet::new();
                    set.insert(0);
                    set.insert(1);
                    set.insert(2);
                    set
                })),
            );
            map
        });
        let t1 = model2.add_set_table(
            table,
            Some(TargetSetArgUnion::SetConst(SetConstPy::from(
                Set::with_capacity(10),
            ))),
            None,
            Some(ob),
        );
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        let t1 = match t1 {
            SetTableUnion::Table(t) => t,
            _ => panic!("expected SetTableUnion::Table but `{:?}`", t1),
        };
        assert_eq!(t1.get_capacity_of_set(), 10);

        let table = SetTableArgUnion::Table({
            let mut map = FxHashMap::default();
            map.insert(
                vec![0, 0, 0, 0],
                TargetSetArgUnion::CreateSetArg(CreateSetArgUnion::Set({
                    let mut set = HashSet::new();
                    set.insert(3);
                    set.insert(4);
                    set.insert(5);
                    set
                })),
            );
            map
        });
        let t2 = model2.add_set_table(
            table,
            Some(TargetSetArgUnion::SetConst(SetConstPy::from(
                Set::with_capacity(10),
            ))),
            None,
            Some(ob),
        );
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        let t2 = match t2 {
            SetTableUnion::Table(t) => t,
            _ => panic!("expected SetTableUnion::Table but `{:?}`", t2),
        };
        assert_eq!(t2.get_capacity_of_set(), 10);

        assert_ne!(t1, t2);

        assert_eq!(model, model2);
    }

    #[test]
    fn add_set_table_from_set_with_name_ok() {
        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());

        let table = SetTableArgUnion::Table({
            let mut map = FxHashMap::default();
            map.insert(
                vec![0, 0, 0, 0],
                TargetSetArgUnion::SetConst(SetConstPy::from({
                    let mut set = Set::with_capacity(10);
                    set.insert(0);
                    set.insert(1);
                    set.insert(2);
                    set
                })),
            );
            map
        });
        let t1 = model.add_set_table(
            table,
            Some(TargetSetArgUnion::SetConst(SetConstPy::from(
                Set::with_capacity(10),
            ))),
            Some("t1"),
            None,
        );
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        let t1 = match t1 {
            SetTableUnion::Table(t) => t,
            _ => panic!("expected SetTableUnion::Table but `{:?}`", t1),
        };
        assert_eq!(t1.get_capacity_of_set(), 10);

        let table = SetTableArgUnion::Table({
            let mut map = FxHashMap::default();
            map.insert(
                vec![0, 0, 0, 0],
                TargetSetArgUnion::SetConst(SetConstPy::from({
                    let mut set = Set::with_capacity(10);
                    set.insert(3);
                    set.insert(4);
                    set.insert(5);
                    set
                })),
            );
            map
        });
        let t2 = model.add_set_table(
            table,
            Some(TargetSetArgUnion::SetConst(SetConstPy::from(
                Set::with_capacity(10),
            ))),
            Some("t2"),
            None,
        );
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        let t2 = match t2 {
            SetTableUnion::Table(t) => t,
            _ => panic!("expected SetTableUnion::Table but `{:?}`", t2),
        };
        assert_eq!(t2.get_capacity_of_set(), 10);

        assert_ne!(t1, t2);

        let mut model2 = ModelPy::default();
        let ob = model2.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let table = SetTableArgUnion::Table({
            let mut map = FxHashMap::default();
            map.insert(
                vec![0, 0, 0, 0],
                TargetSetArgUnion::CreateSetArg(CreateSetArgUnion::Set({
                    let mut set = HashSet::new();
                    set.insert(0);
                    set.insert(1);
                    set.insert(2);
                    set
                })),
            );
            map
        });
        let t1 = model2.add_set_table(
            table,
            Some(TargetSetArgUnion::SetConst(SetConstPy::from(
                Set::with_capacity(10),
            ))),
            Some("t1"),
            Some(ob),
        );
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        let t1 = match t1 {
            SetTableUnion::Table(t) => t,
            _ => panic!("expected SetTableUnion::Table but `{:?}`", t1),
        };
        assert_eq!(t1.get_capacity_of_set(), 10);

        let table = SetTableArgUnion::Table({
            let mut map = FxHashMap::default();
            map.insert(
                vec![0, 0, 0, 0],
                TargetSetArgUnion::CreateSetArg(CreateSetArgUnion::Set({
                    let mut set = HashSet::new();
                    set.insert(3);
                    set.insert(4);
                    set.insert(5);
                    set
                })),
            );
            map
        });
        let t2 = model2.add_set_table(
            table,
            Some(TargetSetArgUnion::SetConst(SetConstPy::from(
                Set::with_capacity(10),
            ))),
            Some("t2"),
            Some(ob),
        );
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        let t2 = match t2 {
            SetTableUnion::Table(t) => t,
            _ => panic!("expected SetTableUnion::Table but `{:?}`", t2),
        };
        assert_eq!(t2.get_capacity_of_set(), 10);

        assert_ne!(t1, t2);

        assert_eq!(model, model2);
    }

    #[test]
    fn add_set_table_from_set_out_of_bound_err() {
        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let table = SetTableArgUnion::Table({
            let mut map = FxHashMap::default();
            map.insert(
                vec![0, 0, 0, 0],
                TargetSetArgUnion::CreateSetArg(CreateSetArgUnion::Set({
                    let mut set = HashSet::new();
                    set.insert(0);
                    set.insert(1);
                    set.insert(2);
                    set.insert(10);
                    set
                })),
            );
            map
        });
        let snapshot = model.clone();
        let t = model.add_set_table(table, None, None, Some(ob));
        assert!(t.is_err());
        assert_eq!(model, snapshot);
    }

    #[test]
    fn add_set_table_from_set_default_out_of_bound_err() {
        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let table = SetTableArgUnion::Table({
            let mut map = FxHashMap::default();
            map.insert(
                vec![0, 0, 0, 0],
                TargetSetArgUnion::CreateSetArg(CreateSetArgUnion::Set({
                    let mut set = HashSet::new();
                    set.insert(0);
                    set.insert(1);
                    set.insert(2);
                    set
                })),
            );
            map
        });
        let default = TargetSetArgUnion::CreateSetArg(CreateSetArgUnion::Set({
            let mut set = HashSet::new();
            set.insert(0);
            set.insert(1);
            set.insert(2);
            set.insert(10);
            set
        }));
        let snapshot = model.clone();
        let t = model.add_set_table(table, Some(default), None, Some(ob));
        assert!(t.is_err());
        assert_eq!(model, snapshot);
    }

    #[test]
    fn add_set_table_from_set_no_object_err() {
        let mut model = ModelPy::default();
        let ob = model.add_object_type(10, None);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let mut model = ModelPy::default();
        let table = SetTableArgUnion::Table({
            let mut map = FxHashMap::default();
            map.insert(
                vec![0, 0, 0, 0],
                TargetSetArgUnion::CreateSetArg(CreateSetArgUnion::Set({
                    let mut set = HashSet::new();
                    set.insert(0);
                    set.insert(1);
                    set.insert(2);
                    set
                })),
            );
            map
        });
        let default = TargetSetArgUnion::CreateSetArg(CreateSetArgUnion::Set({
            let mut set = HashSet::new();
            set.insert(0);
            set.insert(1);
            set.insert(2);
            set
        }));
        let snapshot = model.clone();
        let t = model.add_set_table(table, Some(default), None, Some(ob));
        assert!(t.is_err());
        assert_eq!(model, snapshot);
    }

    #[test]
    fn add_set_table_from_set_none_object_err() {
        let mut model = ModelPy::default();
        let table = SetTableArgUnion::Table({
            let mut map = FxHashMap::default();
            map.insert(
                vec![0, 0, 0, 0],
                TargetSetArgUnion::CreateSetArg(CreateSetArgUnion::Set({
                    let mut set = HashSet::new();
                    set.insert(0);
                    set.insert(1);
                    set.insert(2);
                    set
                })),
            );
            map
        });
        let default = TargetSetArgUnion::CreateSetArg(CreateSetArgUnion::Set({
            let mut set = HashSet::new();
            set.insert(0);
            set.insert(1);
            set.insert(2);
            set
        }));
        let snapshot = model.clone();
        let t = model.add_set_table(table, Some(default), None, None);
        assert!(t.is_err());
        assert_eq!(model, snapshot);
    }

    // #[test]
    // fn set_set_table_item_1d_from_set_const_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = SetTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_set_table(
    //         table,
    //         Some(TargetSetArgUnion::SetConst(SetConstPy::from(
    //             Set::with_capacity(10),
    //         ))),
    //         None,
    //         None,
    //     );
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let t_id = match t.clone() {
    //         SetTableUnion::Table(t) => TableHandle::<Set>::from(t).id(),
    //         _ => panic!("Expected SetTableUnion::Table but `{:?}`", t),
    //     };
    //     let index = TableIndexUnion::Table1D(0);
    //     let result = Python::with_gil(|py| {
    //         let value = SetConstPy::from({
    //             let mut set = Set::with_capacity(10);
    //             set.insert(1);
    //             set
    //         })
    //         .into_py(py);
    //         model.set_table_item(TableUnion::Set(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_ok());
    //     assert_eq!(model.0.table_registry.set_tables.tables[t_id].get(&[0]), &{
    //         let mut set = Set::with_capacity(10);
    //         set.insert(1);
    //         set
    //     });
    // }

    // #[test]
    // fn set_set_table_item_2d_from_set_const_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = SetTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_set_table(
    //         table,
    //         Some(TargetSetArgUnion::SetConst(SetConstPy::from(
    //             Set::with_capacity(10),
    //         ))),
    //         None,
    //         None,
    //     );
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let t_id = match t.clone() {
    //         SetTableUnion::Table(t) => TableHandle::<Set>::from(t).id(),
    //         _ => panic!("Expected SetTableUnion::Table but `{:?}`", t),
    //     };
    //     let index = TableIndexUnion::Table2D((0, 0));
    //     let result = Python::with_gil(|py| {
    //         let value = SetConstPy::from({
    //             let mut set = Set::with_capacity(10);
    //             set.insert(1);
    //             set
    //         })
    //         .into_py(py);
    //         model.set_table_item(TableUnion::Set(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_ok());
    //     assert_eq!(
    //         model.0.table_registry.set_tables.tables[t_id].get(&[0, 0]),
    //         &{
    //             let mut set = Set::with_capacity(10);
    //             set.insert(1);
    //             set
    //         }
    //     );
    // }

    // #[test]
    // fn set_set_table_item_3d_from_set_const_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = SetTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_set_table(
    //         table,
    //         Some(TargetSetArgUnion::SetConst(SetConstPy::from(
    //             Set::with_capacity(10),
    //         ))),
    //         None,
    //         None,
    //     );
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let t_id = match t.clone() {
    //         SetTableUnion::Table(t) => TableHandle::<Set>::from(t).id(),
    //         _ => panic!("Expected SetTableUnion::Table but `{:?}`", t),
    //     };
    //     let index = TableIndexUnion::Table3D((0, 0, 0));
    //     let result = Python::with_gil(|py| {
    //         let value = SetConstPy::from({
    //             let mut set = Set::with_capacity(10);
    //             set.insert(1);
    //             set
    //         })
    //         .into_py(py);
    //         model.set_table_item(TableUnion::Set(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_ok());
    //     assert_eq!(
    //         model.0.table_registry.set_tables.tables[t_id].get(&[0, 0, 0]),
    //         &{
    //             let mut set = Set::with_capacity(10);
    //             set.insert(1);
    //             set
    //         }
    //     );
    // }

    // #[test]
    // fn set_set_table_item_from_set_const_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = SetTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_set_table(
    //         table,
    //         Some(TargetSetArgUnion::SetConst(SetConstPy::from(
    //             Set::with_capacity(10),
    //         ))),
    //         None,
    //         None,
    //     );
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let t_id = match t.clone() {
    //         SetTableUnion::Table(t) => TableHandle::<Set>::from(t).id(),
    //         _ => panic!("Expected SetTableUnion::Table but `{:?}`", t),
    //     };
    //     let index = TableIndexUnion::Table(vec![0, 0, 0, 0]);
    //     let result = Python::with_gil(|py| {
    //         let value = SetConstPy::from({
    //             let mut set = Set::with_capacity(10);
    //             set.insert(1);
    //             set
    //         })
    //         .into_py(py);
    //         model.set_table_item(TableUnion::Set(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_ok());
    //     assert_eq!(
    //         model.0.table_registry.set_tables.tables[t_id].get(&[0, 0, 0, 0]),
    //         &{
    //             let mut set = Set::with_capacity(10);
    //             set.insert(1);
    //             set
    //         }
    //     );
    // }

    // #[test]
    // fn set_set_table_item_value_extract_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = SetTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_set_table(
    //         table,
    //         Some(TargetSetArgUnion::SetConst(SetConstPy::from(
    //             Set::with_capacity(10),
    //         ))),
    //         None,
    //         None,
    //     );
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let index = TableIndexUnion::Table(vec![0, 0, 0, 0]);
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = 1usize.into_py(py);
    //         model.set_table_item(TableUnion::Set(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn set_set_table_item_no_table_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = SetTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_set_table(
    //         table,
    //         Some(TargetSetArgUnion::SetConst(SetConstPy::from(
    //             Set::with_capacity(10),
    //         ))),
    //         None,
    //         None,
    //     );
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let index = TableIndexUnion::Table(vec![0, 0, 0, 0]);
    //     let mut model = ModelPy::default();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = SetConstPy::from({
    //             let mut set = Set::with_capacity(10);
    //             set.insert(1);
    //             set
    //         })
    //         .into_py(py);
    //         model.set_table_item(TableUnion::Set(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn set_set_table_item_1d_from_list_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = SetTableArgUnion::Table(FxHashMap::default());
    //     let ob = model.add_object_type(10, None);
    //     assert!(ob.is_ok());
    //     let t = model.add_set_table(
    //         table,
    //         Some(TargetSetArgUnion::SetConst(SetConstPy::from(
    //             Set::with_capacity(10),
    //         ))),
    //         None,
    //         None,
    //     );
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let t_id = match t.clone() {
    //         SetTableUnion::Table(t) => TableHandle::<Set>::from(t).id(),
    //         _ => panic!("Expected SetTableUnion::Table but `{:?}`", t),
    //     };
    //     let index = TableIndexUnion::Table1D(0);
    //     let result = Python::with_gil(|py| {
    //         let value = vec![1].into_py(py);
    //         model.set_table_item(TableUnion::Set(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_ok());
    //     assert_eq!(model.0.table_registry.set_tables.tables[t_id].get(&[0]), &{
    //         let mut set = Set::with_capacity(10);
    //         set.insert(1);
    //         set
    //     });
    // }

    // #[test]
    // fn set_set_table_item_2d_from_list_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let ob = model.add_object_type(10, None);
    //     assert!(ob.is_ok());
    //     let table = SetTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_set_table(
    //         table,
    //         Some(TargetSetArgUnion::SetConst(SetConstPy::from(
    //             Set::with_capacity(10),
    //         ))),
    //         None,
    //         None,
    //     );
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let t_id = match t.clone() {
    //         SetTableUnion::Table(t) => TableHandle::<Set>::from(t).id(),
    //         _ => panic!("Expected SetTableUnion::Table but `{:?}`", t),
    //     };
    //     let index = TableIndexUnion::Table2D((0, 0));
    //     let result = Python::with_gil(|py| {
    //         let value = vec![1].into_py(py);
    //         model.set_table_item(TableUnion::Set(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_ok());
    //     assert_eq!(
    //         model.0.table_registry.set_tables.tables[t_id].get(&[0, 0]),
    //         &{
    //             let mut set = Set::with_capacity(10);
    //             set.insert(1);
    //             set
    //         }
    //     );
    // }

    // #[test]
    // fn set_set_table_item_3d_from_list_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let ob = model.add_object_type(10, None);
    //     assert!(ob.is_ok());
    //     let table = SetTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_set_table(
    //         table,
    //         Some(TargetSetArgUnion::SetConst(SetConstPy::from(
    //             Set::with_capacity(10),
    //         ))),
    //         None,
    //         None,
    //     );
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let t_id = match t.clone() {
    //         SetTableUnion::Table(t) => TableHandle::<Set>::from(t).id(),
    //         _ => panic!("Expected SetTableUnion::Table but `{:?}`", t),
    //     };
    //     let index = TableIndexUnion::Table3D((0, 0, 0));
    //     let result = Python::with_gil(|py| {
    //         let value = vec![1].into_py(py);
    //         model.set_table_item(TableUnion::Set(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_ok());
    //     assert_eq!(
    //         model.0.table_registry.set_tables.tables[t_id].get(&[0, 0, 0]),
    //         &{
    //             let mut set = Set::with_capacity(10);
    //             set.insert(1);
    //             set
    //         }
    //     );
    // }

    // #[test]
    // fn set_set_table_item_from_list_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let ob = model.add_object_type(10, None);
    //     assert!(ob.is_ok());
    //     let table = SetTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_set_table(
    //         table,
    //         Some(TargetSetArgUnion::SetConst(SetConstPy::from(
    //             Set::with_capacity(10),
    //         ))),
    //         None,
    //         None,
    //     );
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let t_id = match t.clone() {
    //         SetTableUnion::Table(t) => TableHandle::<Set>::from(t).id(),
    //         _ => panic!("Expected SetTableUnion::Table but `{:?}`", t),
    //     };
    //     let index = TableIndexUnion::Table(vec![0, 0, 0, 0]);
    //     let result = Python::with_gil(|py| {
    //         let value = vec![1].into_py(py);
    //         model.set_table_item(TableUnion::Set(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_ok());
    //     assert_eq!(
    //         model.0.table_registry.set_tables.tables[t_id].get(&[0, 0, 0, 0]),
    //         &{
    //             let mut set = Set::with_capacity(10);
    //             set.insert(1);
    //             set
    //         }
    //     );
    // }

    // #[test]
    // fn set_set_table_item_from_list_out_of_bound_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let ob = model.add_object_type(10, None);
    //     assert!(ob.is_ok());
    //     let table = SetTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_set_table(
    //         table,
    //         Some(TargetSetArgUnion::SetConst(SetConstPy::from(
    //             Set::with_capacity(10),
    //         ))),
    //         None,
    //         None,
    //     );
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let index = TableIndexUnion::Table(vec![0, 0, 0, 0]);
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = vec![10].into_py(py);
    //         model.set_table_item(TableUnion::Set(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn set_set_table_item_1d_from_set_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = SetTableArgUnion::Table(FxHashMap::default());
    //     let ob = model.add_object_type(10, None);
    //     assert!(ob.is_ok());
    //     let t = model.add_set_table(
    //         table,
    //         Some(TargetSetArgUnion::SetConst(SetConstPy::from(
    //             Set::with_capacity(10),
    //         ))),
    //         None,
    //         None,
    //     );
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let t_id = match t.clone() {
    //         SetTableUnion::Table(t) => TableHandle::<Set>::from(t).id(),
    //         _ => panic!("Expected SetTableUnion::Table but `{:?}`", t),
    //     };
    //     let index = TableIndexUnion::Table1D(0);
    //     let result = Python::with_gil(|py| {
    //         let value = {
    //             let mut set = HashSet::new();
    //             set.insert(1);
    //             set
    //         }
    //         .into_py(py);
    //         model.set_table_item(TableUnion::Set(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_ok());
    //     assert_eq!(model.0.table_registry.set_tables.tables[t_id].get(&[0]), &{
    //         let mut set = Set::with_capacity(10);
    //         set.insert(1);
    //         set
    //     });
    // }

    // #[test]
    // fn set_set_table_item_2d_from_set_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let ob = model.add_object_type(10, None);
    //     assert!(ob.is_ok());
    //     let table = SetTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_set_table(
    //         table,
    //         Some(TargetSetArgUnion::SetConst(SetConstPy::from(
    //             Set::with_capacity(10),
    //         ))),
    //         None,
    //         None,
    //     );
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let t_id = match t.clone() {
    //         SetTableUnion::Table(t) => TableHandle::<Set>::from(t).id(),
    //         _ => panic!("Expected SetTableUnion::Table but `{:?}`", t),
    //     };
    //     let index = TableIndexUnion::Table2D((0, 0));
    //     let result = Python::with_gil(|py| {
    //         let value = {
    //             let mut set = HashSet::new();
    //             set.insert(1);
    //             set
    //         }
    //         .into_py(py);
    //         model.set_table_item(TableUnion::Set(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_ok());
    //     assert_eq!(
    //         model.0.table_registry.set_tables.tables[t_id].get(&[0, 0]),
    //         &{
    //             let mut set = Set::with_capacity(10);
    //             set.insert(1);
    //             set
    //         }
    //     );
    // }

    // #[test]
    // fn set_set_table_item_3d_from_set_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let ob = model.add_object_type(10, None);
    //     assert!(ob.is_ok());
    //     let table = SetTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_set_table(
    //         table,
    //         Some(TargetSetArgUnion::SetConst(SetConstPy::from(
    //             Set::with_capacity(10),
    //         ))),
    //         None,
    //         None,
    //     );
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let t_id = match t.clone() {
    //         SetTableUnion::Table(t) => TableHandle::<Set>::from(t).id(),
    //         _ => panic!("Expected SetTableUnion::Table but `{:?}`", t),
    //     };
    //     let index = TableIndexUnion::Table3D((0, 0, 0));
    //     let result = Python::with_gil(|py| {
    //         let value = {
    //             let mut set = HashSet::new();
    //             set.insert(1);
    //             set
    //         }
    //         .into_py(py);
    //         model.set_table_item(TableUnion::Set(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_ok());
    //     assert_eq!(
    //         model.0.table_registry.set_tables.tables[t_id].get(&[0, 0, 0]),
    //         &{
    //             let mut set = Set::with_capacity(10);
    //             set.insert(1);
    //             set
    //         }
    //     );
    // }

    // #[test]
    // fn set_set_table_item_from_set_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let ob = model.add_object_type(10, None);
    //     assert!(ob.is_ok());
    //     let table = SetTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_set_table(
    //         table,
    //         Some(TargetSetArgUnion::SetConst(SetConstPy::from(
    //             Set::with_capacity(10),
    //         ))),
    //         None,
    //         None,
    //     );
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let t_id = match t.clone() {
    //         SetTableUnion::Table(t) => TableHandle::<Set>::from(t).id(),
    //         _ => panic!("Expected SetTableUnion::Table but `{:?}`", t),
    //     };
    //     let index = TableIndexUnion::Table(vec![0, 0, 0, 0]);
    //     let result = Python::with_gil(|py| {
    //         let value = {
    //             let mut set = HashSet::new();
    //             set.insert(1);
    //             set
    //         }
    //         .into_py(py);
    //         model.set_table_item(TableUnion::Set(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_ok());
    //     assert_eq!(
    //         model.0.table_registry.set_tables.tables[t_id].get(&[0, 0, 0, 0]),
    //         &{
    //             let mut set = Set::with_capacity(10);
    //             set.insert(1);
    //             set
    //         }
    //     );
    // }

    // #[test]
    // fn set_set_table_item_from_set_out_of_bound_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let ob = model.add_object_type(10, None);
    //     assert!(ob.is_ok());
    //     let table = SetTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_set_table(
    //         table,
    //         Some(TargetSetArgUnion::SetConst(SetConstPy::from(
    //             Set::with_capacity(10),
    //         ))),
    //         None,
    //         None,
    //     );
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let index = TableIndexUnion::Table(vec![0, 0, 0, 0]);
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = {
    //             let mut set = HashSet::new();
    //             set.insert(10);
    //             set
    //         }
    //         .into_py(py);
    //         model.set_table_item(TableUnion::Set(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn set_set_default_from_set_const_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = SetTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_set_table(
    //         table,
    //         Some(TargetSetArgUnion::SetConst(SetConstPy::from(
    //             Set::with_capacity(10),
    //         ))),
    //         None,
    //         None,
    //     );
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let t = match t {
    //         SetTableUnion::Table(t) => t,
    //         _ => panic!("Expected SetTableUnion::Table but `{:?}`", t),
    //     };
    //     let result = Python::with_gil(|py| {
    //         let value = SetConstPy::from(Set::with_capacity(10)).into_py(py);
    //         model.set_default(SetDefaultArgUnion::Set(t), value.as_ref(py))
    //     });
    //     assert!(result.is_ok());
    // }

    // #[test]
    // fn set_set_default_value_extract_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = SetTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_set_table(
    //         table,
    //         Some(TargetSetArgUnion::SetConst(SetConstPy::from(
    //             Set::with_capacity(10),
    //         ))),
    //         None,
    //         None,
    //     );
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let t = match t {
    //         SetTableUnion::Table(t) => t,
    //         _ => panic!("Expected SetTableUnion::Table but `{:?}`", t),
    //     };
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = 1usize.into_py(py);
    //         model.set_default(SetDefaultArgUnion::Set(t), value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn set_set_default_no_table_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = SetTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_set_table(
    //         table,
    //         Some(TargetSetArgUnion::SetConst(SetConstPy::from(
    //             Set::with_capacity(10),
    //         ))),
    //         None,
    //         None,
    //     );
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let t = match t {
    //         SetTableUnion::Table(t) => t,
    //         _ => panic!("Expected SetTableUnion::Table but `{:?}`", t),
    //     };
    //     let mut model = ModelPy::default();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = SetConstPy::from(Set::with_capacity(10)).into_py(py);
    //         model.set_default(SetDefaultArgUnion::Set(t), value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn set_set_default_from_list_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let ob = model.add_object_type(10, None);
    //     assert!(ob.is_ok());
    //     let table = SetTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_set_table(
    //         table,
    //         Some(TargetSetArgUnion::SetConst(SetConstPy::from(
    //             Set::with_capacity(10),
    //         ))),
    //         None,
    //         None,
    //     );
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let t = match t {
    //         SetTableUnion::Table(t) => t,
    //         _ => panic!("Expected SetTableUnion::Table but `{:?}`", t),
    //     };
    //     let result = Python::with_gil(|py| {
    //         let value = vec![1].into_py(py);
    //         model.set_default(SetDefaultArgUnion::Set(t), value.as_ref(py))
    //     });
    //     assert!(result.is_ok());
    // }

    // #[test]
    // fn set_set_default_from_list_out_of_bound_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let ob = model.add_object_type(10, None);
    //     assert!(ob.is_ok());
    //     let table = SetTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_set_table(
    //         table,
    //         Some(TargetSetArgUnion::SetConst(SetConstPy::from(
    //             Set::with_capacity(10),
    //         ))),
    //         None,
    //         None,
    //     );
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let t = match t {
    //         SetTableUnion::Table(t) => t,
    //         _ => panic!("Expected SetTableUnion::Table but `{:?}`", t),
    //     };
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = vec![10].into_py(py);
    //         model.set_default(SetDefaultArgUnion::Set(t), value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn set_set_default_from_set_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let ob = model.add_object_type(10, None);
    //     assert!(ob.is_ok());
    //     let table = SetTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_set_table(
    //         table,
    //         Some(TargetSetArgUnion::SetConst(SetConstPy::from(
    //             Set::with_capacity(10),
    //         ))),
    //         None,
    //         None,
    //     );
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let t = match t {
    //         SetTableUnion::Table(t) => t,
    //         _ => panic!("Expected SetTableUnion::Table but `{:?}`", t),
    //     };
    //     let result = Python::with_gil(|py| {
    //         let value = {
    //             let mut set = HashSet::new();
    //             set.insert(1);
    //             set
    //         }
    //         .into_py(py);
    //         model.set_default(SetDefaultArgUnion::Set(t), value.as_ref(py))
    //     });
    //     assert!(result.is_ok());
    // }

    // #[test]
    // fn set_set_default_from_set_out_of_bound_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let ob = model.add_object_type(10, None);
    //     assert!(ob.is_ok());
    //     let table = SetTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_set_table(
    //         table,
    //         Some(TargetSetArgUnion::SetConst(SetConstPy::from(
    //             Set::with_capacity(10),
    //         ))),
    //         None,
    //         None,
    //     );
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let t = match t {
    //         SetTableUnion::Table(t) => t,
    //         _ => panic!("Expected SetTableUnion::Table but `{:?}`", t),
    //     };
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = {
    //             let mut set = HashSet::new();
    //             set.insert(10);
    //             set
    //         }
    //         .into_py(py);
    //         model.set_default(SetDefaultArgUnion::Set(t), value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_set_table_from_set_const_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();

    //     let table = SetTableArgUnion::Table(FxHashMap::default());
    //     let t1 = model.add_set_table(
    //         table,
    //         Some(TargetSetArgUnion::SetConst(SetConstPy::from(
    //             Set::with_capacity(10),
    //         ))),
    //         None,
    //         None,
    //     );
    //     assert!(t1.is_ok());
    //     let t1 = t1.unwrap();
    //     let t1 = match t1 {
    //         SetTableUnion::Table(t) => t,
    //         _ => panic!("expected SetTableUnion::Table but `{:?}`", t1),
    //     };
    //     let table: FxHashMap<(Element, Element, Element, Element), SetConstPy> = {
    //         let mut map = FxHashMap::default();
    //         map.insert(
    //             (0, 0, 0, 0),
    //             SetConstPy::from({
    //                 let mut set = Set::with_capacity(10);
    //                 set.insert(0);
    //                 set.insert(1);
    //                 set.insert(2);
    //                 set
    //             }),
    //         );
    //         map
    //     };
    //     let result = Python::with_gil(|py| {
    //         let table = table.into_py(py);
    //         let default = SetConstPy::from(Set::with_capacity(10)).into_py(py);
    //         model.update_table(
    //             TableUnion::Set(SetTableUnion::Table(t1.clone())),
    //             table.as_ref(py),
    //             Some(default.as_ref(py)),
    //         )
    //     });
    //     assert!(result.is_ok());
    //     let t_id = TableHandle::<Set>::from(t1).id();
    //     assert_eq!(
    //         model.0.table_registry.set_tables.tables[t_id],
    //         dypdl::Table::new(
    //             {
    //                 let mut map = FxHashMap::default();
    //                 let mut set = Set::with_capacity(10);
    //                 set.insert(0);
    //                 set.insert(1);
    //                 set.insert(2);
    //                 map.insert(vec![0, 0, 0, 0], set);
    //                 map
    //             },
    //             Set::with_capacity(10)
    //         )
    //     );
    // }

    // #[test]
    // fn update_set_table_dimension_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();

    //     let table = SetTableArgUnion::Table(FxHashMap::default());
    //     let t1 = model.add_set_table(
    //         table,
    //         Some(TargetSetArgUnion::SetConst(SetConstPy::from(
    //             Set::with_capacity(10),
    //         ))),
    //         None,
    //         None,
    //     );
    //     assert!(t1.is_ok());
    //     let t1 = t1.unwrap();
    //     let t1 = match t1 {
    //         SetTableUnion::Table(t) => t,
    //         _ => panic!("expected SetTableUnion::Table but `{:?}`", t1),
    //     };
    //     let table = vec![SetConstPy::from({
    //         let mut set = Set::with_capacity(10);
    //         set.insert(0);
    //         set.insert(1);
    //         set.insert(2);
    //         set
    //     })];
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let table = table.into_py(py);
    //         let default = SetConstPy::from(Set::with_capacity(10)).into_py(py);
    //         model.update_table(
    //             TableUnion::Set(SetTableUnion::Table(t1.clone())),
    //             table.as_ref(py),
    //             Some(default.as_ref(py)),
    //         )
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_set_table_value_extract_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();

    //     let table = SetTableArgUnion::Table(FxHashMap::default());
    //     let t1 = model.add_set_table(
    //         table,
    //         Some(TargetSetArgUnion::SetConst(SetConstPy::from(
    //             Set::with_capacity(10),
    //         ))),
    //         None,
    //         None,
    //     );
    //     assert!(t1.is_ok());
    //     let t1 = t1.unwrap();
    //     let t1 = match t1 {
    //         SetTableUnion::Table(t) => t,
    //         _ => panic!("expected SetTableUnion::Table but `{:?}`", t1),
    //     };
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let table = 1usize.into_py(py);
    //         let default = SetConstPy::from(Set::with_capacity(10)).into_py(py);
    //         model.update_table(
    //             TableUnion::Set(SetTableUnion::Table(t1.clone())),
    //             table.as_ref(py),
    //             Some(default.as_ref(py)),
    //         )
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_set_table_default_extract_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();

    //     let table = SetTableArgUnion::Table(FxHashMap::default());
    //     let t1 = model.add_set_table(
    //         table,
    //         Some(TargetSetArgUnion::SetConst(SetConstPy::from(
    //             Set::with_capacity(10),
    //         ))),
    //         None,
    //         None,
    //     );
    //     assert!(t1.is_ok());
    //     let t1 = t1.unwrap();
    //     let t1 = match t1 {
    //         SetTableUnion::Table(t) => t,
    //         _ => panic!("expected SetTableUnion::Table but `{:?}`", t1),
    //     };
    //     let table: FxHashMap<(Element, Element, Element, Element), SetConstPy> = {
    //         let mut map = FxHashMap::default();
    //         map.insert(
    //             (0, 0, 0, 0),
    //             SetConstPy::from({
    //                 let mut set = Set::with_capacity(10);
    //                 set.insert(0);
    //                 set.insert(1);
    //                 set.insert(2);
    //                 set
    //             }),
    //         );
    //         map
    //     };
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let table = table.into_py(py);
    //         let default = 1usize.into_py(py);
    //         model.update_table(
    //             TableUnion::Set(SetTableUnion::Table(t1.clone())),
    //             table.as_ref(py),
    //             Some(default.as_ref(py)),
    //         )
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_set_table_no_table_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();

    //     let table = SetTableArgUnion::Table(FxHashMap::default());
    //     let t1 = model.add_set_table(
    //         table,
    //         Some(TargetSetArgUnion::SetConst(SetConstPy::from(
    //             Set::with_capacity(10),
    //         ))),
    //         None,
    //         None,
    //     );
    //     assert!(t1.is_ok());
    //     let t1 = t1.unwrap();
    //     let t1 = match t1 {
    //         SetTableUnion::Table(t) => t,
    //         _ => panic!("expected SetTableUnion::Table but `{:?}`", t1),
    //     };
    //     let table: FxHashMap<(Element, Element, Element, Element), SetConstPy> = {
    //         let mut map = FxHashMap::default();
    //         map.insert(
    //             (0, 0, 0, 0),
    //             SetConstPy::from({
    //                 let mut set = Set::with_capacity(10);
    //                 set.insert(0);
    //                 set.insert(1);
    //                 set.insert(2);
    //                 set
    //             }),
    //         );
    //         map
    //     };
    //     let mut model = ModelPy::default();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let table = table.into_py(py);
    //         let default = SetConstPy::from(Set::with_capacity(10)).into_py(py);
    //         model.update_table(
    //             TableUnion::Set(SetTableUnion::Table(t1.clone())),
    //             table.as_ref(py),
    //             Some(default.as_ref(py)),
    //         )
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_set_table_from_set_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let ob = model.add_object_type(10, None);
    //     assert!(ob.is_ok());

    //     let table = SetTableArgUnion::Table(FxHashMap::default());
    //     let t1 = model.add_set_table(
    //         table,
    //         Some(TargetSetArgUnion::SetConst(SetConstPy::from(
    //             Set::with_capacity(10),
    //         ))),
    //         None,
    //         None,
    //     );
    //     assert!(t1.is_ok());
    //     let t1 = t1.unwrap();
    //     let t1 = match t1 {
    //         SetTableUnion::Table(t) => t,
    //         _ => panic!("expected SetTableUnion::Table but `{:?}`", t1),
    //     };
    //     let table: FxHashMap<(Element, Element, Element, Element), HashSet<Element>> = {
    //         let mut map = FxHashMap::default();
    //         map.insert((0, 0, 0, 0), {
    //             let mut set = HashSet::new();
    //             set.insert(0);
    //             set.insert(1);
    //             set.insert(2);
    //             set
    //         });
    //         map
    //     };
    //     let result = Python::with_gil(|py| {
    //         let table = table.into_py(py);
    //         let default = HashSet::<Element>::new().into_py(py);
    //         model.update_table(
    //             TableUnion::Set(SetTableUnion::Table(t1.clone())),
    //             table.as_ref(py),
    //             Some(default.as_ref(py)),
    //         )
    //     });
    //     assert!(result.is_ok());
    //     let t_id = TableHandle::<Set>::from(t1).id();
    //     assert_eq!(
    //         model.0.table_registry.set_tables.tables[t_id],
    //         dypdl::Table::new(
    //             {
    //                 let mut map = FxHashMap::default();
    //                 let mut set = Set::with_capacity(10);
    //                 set.insert(0);
    //                 set.insert(1);
    //                 set.insert(2);
    //                 map.insert(vec![0, 0, 0, 0], set);
    //                 map
    //             },
    //             Set::with_capacity(10)
    //         )
    //     );
    // }

    // #[test]
    // fn update_set_table_from_set_out_of_bound_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let ob = model.add_object_type(10, None);
    //     assert!(ob.is_ok());

    //     let table = SetTableArgUnion::Table(FxHashMap::default());
    //     let t1 = model.add_set_table(
    //         table,
    //         Some(TargetSetArgUnion::SetConst(SetConstPy::from(
    //             Set::with_capacity(10),
    //         ))),
    //         None,
    //         None,
    //     );
    //     assert!(t1.is_ok());
    //     let t1 = t1.unwrap();
    //     let t1 = match t1 {
    //         SetTableUnion::Table(t) => t,
    //         _ => panic!("expected SetTableUnion::Table but `{:?}`", t1),
    //     };
    //     let table: FxHashMap<(Element, Element, Element, Element), HashSet<Element>> = {
    //         let mut map = FxHashMap::default();
    //         map.insert((0, 0, 0, 0), {
    //             let mut set = HashSet::new();
    //             set.insert(0);
    //             set.insert(1);
    //             set.insert(2);
    //             set.insert(10);
    //             set
    //         });
    //         map
    //     };
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let table = table.into_py(py);
    //         let default = HashSet::<Element>::new().into_py(py);
    //         model.update_table(
    //             TableUnion::Set(SetTableUnion::Table(t1.clone())),
    //             table.as_ref(py),
    //             Some(default.as_ref(py)),
    //         )
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_set_table_from_set_default_out_of_bound_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let ob = model.add_object_type(10, None);
    //     assert!(ob.is_ok());

    //     let table = SetTableArgUnion::Table(FxHashMap::default());
    //     let t1 = model.add_set_table(
    //         table,
    //         Some(TargetSetArgUnion::SetConst(SetConstPy::from(
    //             Set::with_capacity(10),
    //         ))),
    //         None,
    //         None,
    //     );
    //     assert!(t1.is_ok());
    //     let t1 = t1.unwrap();
    //     let t1 = match t1 {
    //         SetTableUnion::Table(t) => t,
    //         _ => panic!("expected SetTableUnion::Table but `{:?}`", t1),
    //     };
    //     let table: FxHashMap<(Element, Element, Element, Element), HashSet<Element>> = {
    //         let mut map = FxHashMap::default();
    //         map.insert((0, 0, 0, 0), {
    //             let mut set = HashSet::new();
    //             set.insert(0);
    //             set.insert(1);
    //             set.insert(2);
    //             set
    //         });
    //         map
    //     };
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let table = table.into_py(py);
    //         let default = {
    //             let mut set = HashSet::<Element>::new();
    //             set.insert(10);
    //             set
    //         }
    //         .into_py(py);
    //         model.update_table(
    //             TableUnion::Set(SetTableUnion::Table(t1.clone())),
    //             table.as_ref(py),
    //             Some(default.as_ref(py)),
    //         )
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_set_table_from_list_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let ob = model.add_object_type(10, None);
    //     assert!(ob.is_ok());

    //     let table = SetTableArgUnion::Table(FxHashMap::default());
    //     let t1 = model.add_set_table(
    //         table,
    //         Some(TargetSetArgUnion::SetConst(SetConstPy::from(
    //             Set::with_capacity(10),
    //         ))),
    //         None,
    //         None,
    //     );
    //     assert!(t1.is_ok());
    //     let t1 = t1.unwrap();
    //     let t1 = match t1 {
    //         SetTableUnion::Table(t) => t,
    //         _ => panic!("expected SetTableUnion::Table but `{:?}`", t1),
    //     };
    //     let table: FxHashMap<(Element, Element, Element, Element), Vec<Element>> = {
    //         let mut map = FxHashMap::default();
    //         map.insert((0, 0, 0, 0), vec![0, 1, 2]);
    //         map
    //     };
    //     let result = Python::with_gil(|py| {
    //         let table = table.into_py(py);
    //         let default = Vec::<Element>::new().into_py(py);
    //         model.update_table(
    //             TableUnion::Set(SetTableUnion::Table(t1.clone())),
    //             table.as_ref(py),
    //             Some(default.as_ref(py)),
    //         )
    //     });
    //     assert!(result.is_ok());
    //     let t_id = TableHandle::<Set>::from(t1).id();
    //     assert_eq!(
    //         model.0.table_registry.set_tables.tables[t_id],
    //         dypdl::Table::new(
    //             {
    //                 let mut map = FxHashMap::default();
    //                 let mut set = Set::with_capacity(10);
    //                 set.insert(0);
    //                 set.insert(1);
    //                 set.insert(2);
    //                 map.insert(vec![0, 0, 0, 0], set);
    //                 map
    //             },
    //             Set::with_capacity(10)
    //         )
    //     );
    // }

    // #[test]
    // fn update_set_table_from_list_out_of_bound_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let ob = model.add_object_type(10, None);
    //     assert!(ob.is_ok());

    //     let table = SetTableArgUnion::Table(FxHashMap::default());
    //     let t1 = model.add_set_table(
    //         table,
    //         Some(TargetSetArgUnion::SetConst(SetConstPy::from(
    //             Set::with_capacity(10),
    //         ))),
    //         None,
    //         None,
    //     );
    //     assert!(t1.is_ok());
    //     let t1 = t1.unwrap();
    //     let t1 = match t1 {
    //         SetTableUnion::Table(t) => t,
    //         _ => panic!("expected SetTableUnion::Table but `{:?}`", t1),
    //     };
    //     let table: FxHashMap<(Element, Element, Element, Element), Vec<Element>> = {
    //         let mut map = FxHashMap::default();
    //         map.insert((0, 0, 0, 0), vec![0, 1, 2, 10]);
    //         map
    //     };
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let table = table.into_py(py);
    //         let default = Vec::<Element>::new().into_py(py);
    //         model.update_table(
    //             TableUnion::Set(SetTableUnion::Table(t1.clone())),
    //             table.as_ref(py),
    //             Some(default.as_ref(py)),
    //         )
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_set_table_from_list_default_out_of_bound_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let ob = model.add_object_type(10, None);
    //     assert!(ob.is_ok());

    //     let table = SetTableArgUnion::Table(FxHashMap::default());
    //     let t1 = model.add_set_table(
    //         table,
    //         Some(TargetSetArgUnion::SetConst(SetConstPy::from(
    //             Set::with_capacity(10),
    //         ))),
    //         None,
    //         None,
    //     );
    //     assert!(t1.is_ok());
    //     let t1 = t1.unwrap();
    //     let t1 = match t1 {
    //         SetTableUnion::Table(t) => t,
    //         _ => panic!("expected SetTableUnion::Table but `{:?}`", t1),
    //     };
    //     let table: FxHashMap<(Element, Element, Element, Element), Vec<Element>> = {
    //         let mut map = FxHashMap::default();
    //         map.insert((0, 0, 0, 0), vec![0, 1, 2]);
    //         map
    //     };
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let table = table.into_py(py);
    //         let default = vec![10].into_py(py);
    //         model.update_table(
    //             TableUnion::Set(SetTableUnion::Table(t1.clone())),
    //             table.as_ref(py),
    //             Some(default.as_ref(py)),
    //         )
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    #[test]
    fn add_bool_table_1d_ok() {
        let mut model = ModelPy::default();

        let table = BoolTableArgUnion::Table1D(vec![true]);
        let t1 = model.add_bool_table(table, None, None);
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        assert!(matches!(t1, BoolTableUnion::Table1D(_)));

        let table = BoolTableArgUnion::Table1D(vec![false]);
        let t2 = model.add_bool_table(table, None, None);
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        assert!(matches!(t2, BoolTableUnion::Table1D(_)));

        assert_ne!(t1, t2);
    }

    #[test]
    fn add_bool_table_1d_with_name_ok() {
        let mut model = ModelPy::default();

        let table = BoolTableArgUnion::Table1D(vec![true]);
        let t1 = model.add_bool_table(table, None, Some("t1"));
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        assert!(matches!(t1, BoolTableUnion::Table1D(_)));

        let table = BoolTableArgUnion::Table1D(vec![false]);
        let t2 = model.add_bool_table(table, None, Some("t2"));
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        assert!(matches!(t2, BoolTableUnion::Table1D(_)));

        assert_ne!(t1, t2);
    }

    #[test]
    fn add_bool_table_1d_err() {
        let mut model = ModelPy::default();

        let table = BoolTableArgUnion::Table1D(vec![true]);
        let t = model.add_bool_table(table, None, Some("t"));
        assert!(t.is_ok());
        let table = BoolTableArgUnion::Table1D(vec![true]);
        let t = model.add_bool_table(table, None, Some("t"));
        assert!(t.is_err());
    }

    // #[test]
    // fn set_bool_table_1d_item_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = BoolTableArgUnion::Table1D(vec![true]);
    //     let t = model.add_bool_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let result = Python::with_gil(|py| {
    //         let value = true.into_py(py);
    //         let index = TableIndexUnion::Table1D(0);
    //         model.set_table_item(TableUnion::Bool(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_ok());
    // }

    // #[test]
    // fn set_bool_table_1d_item_extract_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = BoolTableArgUnion::Table1D(vec![true]);
    //     let t = model.add_bool_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = 1.5f64.into_py(py);
    //         let index = TableIndexUnion::Table1D(0);
    //         model.set_table_item(TableUnion::Bool(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn set_bool_table_1d_item_dimension_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = BoolTableArgUnion::Table1D(vec![true]);
    //     let t = model.add_bool_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = true.into_py(py);
    //         let index = TableIndexUnion::Table2D((0, 0));
    //         model.set_table_item(TableUnion::Bool(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn set_bool_table_1d_no_variable_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = BoolTableArgUnion::Table1D(vec![true]);
    //     let t = model.add_bool_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let mut model = ModelPy::default();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = true.into_py(py);
    //         let index = TableIndexUnion::Table1D(0);
    //         model.set_table_item(TableUnion::Bool(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_bool_table_1d_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = BoolTableArgUnion::Table1D(vec![true]);
    //     let t = model.add_bool_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let result = Python::with_gil(|py| {
    //         let value: Py<PyAny> = vec![true].into_py(py);
    //         model.update_table(TableUnion::Bool(t), value.as_ref(py), None)
    //     });
    //     assert!(result.is_ok());
    // }

    // #[test]
    // fn update_bool_table_1d_value_extract_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = BoolTableArgUnion::Table1D(vec![true]);
    //     let t = model.add_bool_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value: Py<PyAny> = vec![1.5f64].into_py(py);
    //         model.update_table(TableUnion::Bool(t), value.as_ref(py), None)
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_bool_table_1d_value_dimension_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = BoolTableArgUnion::Table1D(vec![true]);
    //     let t = model.add_bool_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value: Py<PyAny> = vec![vec![true]].into_py(py);
    //         model.update_table(TableUnion::Bool(t), value.as_ref(py), None)
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_bool_table_1d_value_no_table_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = BoolTableArgUnion::Table1D(vec![true]);
    //     let t = model.add_bool_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let mut model = ModelPy::default();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value: Py<PyAny> = vec![true].into_py(py);
    //         model.update_table(TableUnion::Bool(t), value.as_ref(py), None)
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    #[test]
    fn add_bool_table_2d_ok() {
        let mut model = ModelPy::default();

        let table = BoolTableArgUnion::Table2D(vec![vec![true]]);
        let t1 = model.add_bool_table(table, None, None);
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        assert!(matches!(t1, BoolTableUnion::Table2D(_)));

        let table = BoolTableArgUnion::Table2D(vec![vec![false]]);
        let t2 = model.add_bool_table(table, None, None);
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        assert!(matches!(t2, BoolTableUnion::Table2D(_)));

        assert_ne!(t1, t2);
    }

    #[test]
    fn add_bool_table_2d_with_name_ok() {
        let mut model = ModelPy::default();

        let table = BoolTableArgUnion::Table2D(vec![vec![true]]);
        let t1 = model.add_bool_table(table, None, Some("t1"));
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        assert!(matches!(t1, BoolTableUnion::Table2D(_)));

        let table = BoolTableArgUnion::Table2D(vec![vec![false]]);
        let t2 = model.add_bool_table(table, None, Some("t2"));
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        assert!(matches!(t2, BoolTableUnion::Table2D(_)));

        assert_ne!(t1, t2);
    }

    #[test]
    fn add_bool_table_2d_err() {
        let mut model = ModelPy::default();

        let table = BoolTableArgUnion::Table2D(vec![vec![true]]);
        let t = model.add_bool_table(table, None, Some("t"));
        assert!(t.is_ok());
        let table = BoolTableArgUnion::Table2D(vec![vec![true]]);
        let t = model.add_bool_table(table, None, Some("t"));
        assert!(t.is_err());
    }

    // #[test]
    // fn set_bool_table_2d_item_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = BoolTableArgUnion::Table2D(vec![vec![true]]);
    //     let t = model.add_bool_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let result = Python::with_gil(|py| {
    //         let value = true.into_py(py);
    //         let index = TableIndexUnion::Table2D((0, 0));
    //         model.set_table_item(TableUnion::Bool(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_ok());
    // }

    // #[test]
    // fn set_bool_table_2d_item_extract_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = BoolTableArgUnion::Table2D(vec![vec![true]]);
    //     let t = model.add_bool_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = 1.5f64.into_py(py);
    //         let index = TableIndexUnion::Table2D((0, 0));
    //         model.set_table_item(TableUnion::Bool(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn set_bool_table_2d_item_dimension_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = BoolTableArgUnion::Table2D(vec![vec![true]]);
    //     let t = model.add_bool_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = true.into_py(py);
    //         let index = TableIndexUnion::Table3D((0, 0, 0));
    //         model.set_table_item(TableUnion::Bool(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn set_bool_table_2d_no_variable_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = BoolTableArgUnion::Table2D(vec![vec![true]]);
    //     let t = model.add_bool_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let mut model = ModelPy::default();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = true.into_py(py);
    //         let index = TableIndexUnion::Table2D((0, 0));
    //         model.set_table_item(TableUnion::Bool(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_bool_table_2d_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = BoolTableArgUnion::Table2D(vec![vec![true]]);
    //     let t = model.add_bool_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let result = Python::with_gil(|py| {
    //         let value: Py<PyAny> = vec![vec![true]].into_py(py);
    //         model.update_table(TableUnion::Bool(t), value.as_ref(py), None)
    //     });
    //     assert!(result.is_ok());
    // }

    // #[test]
    // fn update_bool_table_2d_value_extract_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = BoolTableArgUnion::Table2D(vec![vec![true]]);
    //     let t = model.add_bool_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value: Py<PyAny> = vec![vec![1.5f64]].into_py(py);
    //         model.update_table(TableUnion::Bool(t), value.as_ref(py), None)
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_bool_table_2d_value_dimension_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = BoolTableArgUnion::Table2D(vec![vec![true]]);
    //     let t = model.add_bool_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value: Py<PyAny> = vec![vec![vec![true]]].into_py(py);
    //         model.update_table(TableUnion::Bool(t), value.as_ref(py), None)
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_bool_table_2d_value_no_table_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = BoolTableArgUnion::Table2D(vec![vec![true]]);
    //     let t = model.add_bool_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let mut model = ModelPy::default();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value: Py<PyAny> = vec![vec![true]].into_py(py);
    //         model.update_table(TableUnion::Bool(t), value.as_ref(py), None)
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    #[test]
    fn add_bool_table_3d_ok() {
        let mut model = ModelPy::default();

        let table = BoolTableArgUnion::Table3D(vec![vec![vec![true]]]);
        let t1 = model.add_bool_table(table, None, None);
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        assert!(matches!(t1, BoolTableUnion::Table3D(_)));

        let table = BoolTableArgUnion::Table3D(vec![vec![vec![false]]]);
        let t2 = model.add_bool_table(table, None, None);
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        assert!(matches!(t2, BoolTableUnion::Table3D(_)));

        assert_ne!(t1, t2);
    }

    #[test]
    fn add_bool_table_3d_with_name_ok() {
        let mut model = ModelPy::default();

        let table = BoolTableArgUnion::Table3D(vec![vec![vec![true]]]);
        let t1 = model.add_bool_table(table, None, Some("t1"));
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        assert!(matches!(t1, BoolTableUnion::Table3D(_)));

        let table = BoolTableArgUnion::Table3D(vec![vec![vec![false]]]);
        let t2 = model.add_bool_table(table, None, Some("t2"));
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        assert!(matches!(t2, BoolTableUnion::Table3D(_)));

        assert_ne!(t1, t2);
    }

    #[test]
    fn add_bool_table_3d_err() {
        let mut model = ModelPy::default();

        let table = BoolTableArgUnion::Table3D(vec![vec![vec![true]]]);
        let t = model.add_bool_table(table, None, Some("t"));
        assert!(t.is_ok());
        let table = BoolTableArgUnion::Table3D(vec![vec![vec![true]]]);
        let t = model.add_bool_table(table, None, Some("t"));
        assert!(t.is_err());
    }

    // #[test]
    // fn set_bool_table_3d_item_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = BoolTableArgUnion::Table3D(vec![vec![vec![true]]]);
    //     let t = model.add_bool_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let result = Python::with_gil(|py| {
    //         let value = true.into_py(py);
    //         let index = TableIndexUnion::Table3D((0, 0, 0));
    //         model.set_table_item(TableUnion::Bool(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_ok());
    // }

    // #[test]
    // fn set_bool_table_3d_item_extract_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = BoolTableArgUnion::Table3D(vec![vec![vec![true]]]);
    //     let t = model.add_bool_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = 1.5f64.into_py(py);
    //         let index = TableIndexUnion::Table3D((0, 0, 0));
    //         model.set_table_item(TableUnion::Bool(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn set_bool_table_3d_item_dimension_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = BoolTableArgUnion::Table3D(vec![vec![vec![true]]]);
    //     let t = model.add_bool_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = true.into_py(py);
    //         let index = TableIndexUnion::Table2D((0, 0));
    //         model.set_table_item(TableUnion::Bool(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn set_bool_table_3d_no_variable_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = BoolTableArgUnion::Table3D(vec![vec![vec![true]]]);
    //     let t = model.add_bool_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let mut model = ModelPy::default();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = true.into_py(py);
    //         let index = TableIndexUnion::Table3D((0, 0, 0));
    //         model.set_table_item(TableUnion::Bool(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_bool_table_3d_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = BoolTableArgUnion::Table3D(vec![vec![vec![true]]]);
    //     let t = model.add_bool_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let result = Python::with_gil(|py| {
    //         let value: Py<PyAny> = vec![vec![vec![true]]].into_py(py);
    //         model.update_table(TableUnion::Bool(t), value.as_ref(py), None)
    //     });
    //     assert!(result.is_ok());
    // }

    // #[test]
    // fn update_bool_table_3d_value_extract_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = BoolTableArgUnion::Table3D(vec![vec![vec![true]]]);
    //     let t = model.add_bool_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value: Py<PyAny> = vec![vec![vec![1.5f64]]].into_py(py);
    //         model.update_table(TableUnion::Bool(t), value.as_ref(py), None)
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_bool_table_3d_value_dimension_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = BoolTableArgUnion::Table3D(vec![vec![vec![true]]]);
    //     let t = model.add_bool_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value: Py<PyAny> = vec![vec![true]].into_py(py);
    //         model.update_table(TableUnion::Bool(t), value.as_ref(py), None)
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_bool_table_3d_value_no_table_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = BoolTableArgUnion::Table3D(vec![vec![vec![true]]]);
    //     let t = model.add_bool_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let mut model = ModelPy::default();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value: Py<PyAny> = vec![vec![vec![true]]].into_py(py);
    //         model.update_table(TableUnion::Bool(t), value.as_ref(py), None)
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    #[test]
    fn add_bool_table_ok() {
        let mut model = ModelPy::default();

        let table = BoolTableArgUnion::Table(FxHashMap::default());
        let t1 = model.add_bool_table(table, Some(true), None);
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        assert!(matches!(t1, BoolTableUnion::Table(_)));

        let table = BoolTableArgUnion::Table(FxHashMap::default());
        let t2 = model.add_bool_table(table, Some(true), None);
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        assert!(matches!(t2, BoolTableUnion::Table(_)));

        assert_ne!(t1, t2);
    }

    #[test]
    fn add_bool_table_with_name_ok() {
        let mut model = ModelPy::default();

        let table = BoolTableArgUnion::Table(FxHashMap::default());
        let t1 = model.add_bool_table(table, Some(true), Some("t1"));
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        assert!(matches!(t1, BoolTableUnion::Table(_)));

        let table = BoolTableArgUnion::Table(FxHashMap::default());
        let t2 = model.add_bool_table(table, Some(true), Some("t2"));
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        assert!(matches!(t1, BoolTableUnion::Table(_)));

        assert_ne!(t1, t2);
    }

    #[test]
    fn add_bool_table_duplicate_err() {
        let mut model = ModelPy::default();

        let table = BoolTableArgUnion::Table(FxHashMap::default());
        let t = model.add_bool_table(table, Some(true), Some("t"));
        assert!(t.is_ok());
        let table = BoolTableArgUnion::Table(FxHashMap::default());
        let t = model.add_bool_table(table, Some(true), Some("t"));
        assert!(t.is_err());
    }

    #[test]
    fn add_bool_table_no_default_err() {
        let mut model = ModelPy::default();

        let table = BoolTableArgUnion::Table(FxHashMap::default());
        let t = model.add_bool_table(table, None, None);
        assert!(t.is_err());
    }

    // #[test]
    // fn set_bool_table_item_1d_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = BoolTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_bool_table(table, Some(true), None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let result = Python::with_gil(|py| {
    //         let value = true.into_py(py);
    //         let index = TableIndexUnion::Table1D(0);
    //         model.set_table_item(TableUnion::Bool(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_ok());
    //     let t = match t {
    //         BoolTableUnion::Table(t) => t,
    //         _ => panic!("expected BoolTableUnion::Table but `{:?}`", t),
    //     };
    //     let t_id = TableHandle::<bool>::from(t).id();
    //     assert!(*model.0.table_registry.bool_tables.tables[t_id].get(&[0]),);
    // }

    // #[test]
    // fn set_bool_table_item_2d_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = BoolTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_bool_table(table, Some(true), None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let result = Python::with_gil(|py| {
    //         let value = true.into_py(py);
    //         let index = TableIndexUnion::Table2D((0, 0));
    //         model.set_table_item(TableUnion::Bool(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_ok());
    //     let t = match t {
    //         BoolTableUnion::Table(t) => t,
    //         _ => panic!("expected BoolTableUnion::Table but `{:?}`", t),
    //     };
    //     let t_id = TableHandle::<bool>::from(t).id();
    //     assert!(*model.0.table_registry.bool_tables.tables[t_id].get(&[0, 0]),);
    // }

    // #[test]
    // fn set_bool_table_item_3d_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = BoolTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_bool_table(table, Some(true), None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let result = Python::with_gil(|py| {
    //         let value = true.into_py(py);
    //         let index = TableIndexUnion::Table3D((0, 0, 0));
    //         model.set_table_item(TableUnion::Bool(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_ok());
    //     let t = match t {
    //         BoolTableUnion::Table(t) => t,
    //         _ => panic!("expected BoolTableUnion::Table but `{:?}`", t),
    //     };
    //     let t_id = TableHandle::<bool>::from(t).id();
    //     assert!(*model.0.table_registry.bool_tables.tables[t_id].get(&[0, 0, 0]),);
    // }

    // #[test]
    // fn set_bool_table_item_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = BoolTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_bool_table(table, Some(true), None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let result = Python::with_gil(|py| {
    //         let value = true.into_py(py);
    //         let index = TableIndexUnion::Table(vec![0, 0, 0, 0]);
    //         model.set_table_item(TableUnion::Bool(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_ok());
    // }

    // #[test]
    // fn set_bool_table_item_extract_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = BoolTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_bool_table(table, Some(true), None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = 1.5f64.into_py(py);
    //         let index = TableIndexUnion::Table(vec![0, 0, 0, 0]);
    //         model.set_table_item(TableUnion::Bool(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn set_bool_table_item_no_table_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = BoolTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_bool_table(table, Some(true), None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let mut model = ModelPy::default();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = true.into_py(py);
    //         let index = TableIndexUnion::Table(vec![0, 0, 0, 0]);
    //         model.set_table_item(TableUnion::Bool(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn set_bool_default_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = BoolTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_bool_table(table, Some(true), None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let t = match t {
    //         BoolTableUnion::Table(t) => t,
    //         _ => panic!("expected BoolTableUnion::Table but `{:?}`", t),
    //     };
    //     let result = Python::with_gil(|py| {
    //         let value = true.into_py(py);
    //         model.set_default(SetDefaultArgUnion::Bool(t), value.as_ref(py))
    //     });
    //     assert!(result.is_ok());
    // }

    // #[test]
    // fn set_bool_default_extract_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = BoolTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_bool_table(table, Some(true), None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let t = match t {
    //         BoolTableUnion::Table(t) => t,
    //         _ => panic!("expected BoolTableUnion::Table but `{:?}`", t),
    //     };
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = 1.5f64.into_py(py);
    //         model.set_default(SetDefaultArgUnion::Bool(t), value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn set_bool_default_no_table_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = BoolTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_bool_table(table, Some(true), None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let t = match t {
    //         BoolTableUnion::Table(t) => t,
    //         _ => panic!("expected BoolTableUnion::Table but `{:?}`", t),
    //     };
    //     let mut model = ModelPy::default();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = 1.5f64.into_py(py);
    //         model.set_default(SetDefaultArgUnion::Bool(t), value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_bool_table_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = BoolTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_bool_table(table, Some(true), None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let result = Python::with_gil(|py| {
    //         let value = {
    //             let mut map = FxHashMap::default();
    //             map.insert((0, 0, 0, 0), false);
    //             map
    //         }
    //         .into_py(py);
    //         let default = false.into_py(py);
    //         model.update_table(
    //             TableUnion::Bool(t),
    //             value.as_ref(py),
    //             Some(default.as_ref(py)),
    //         )
    //     });
    //     assert!(result.is_ok());
    // }

    // #[test]
    // fn update_bool_table_no_default_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = BoolTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_bool_table(table, Some(true), None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = {
    //             let mut map = FxHashMap::default();
    //             map.insert((0, 0, 0, 0), false);
    //             map
    //         }
    //         .into_py(py);
    //         model.update_table(TableUnion::Bool(t), value.as_ref(py), None)
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_bool_table_default_extract_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = BoolTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_bool_table(table, Some(true), None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = {
    //             let mut map = FxHashMap::default();
    //             map.insert((0, 0, 0, 0), false);
    //             map
    //         }
    //         .into_py(py);
    //         let default = 1.5f64.into_py(py);
    //         model.update_table(
    //             TableUnion::Bool(t),
    //             value.as_ref(py),
    //             Some(default.as_ref(py)),
    //         )
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_bool_table_value_extract_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = BoolTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_bool_table(table, Some(true), None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = {
    //             let mut map = FxHashMap::default();
    //             map.insert((0, 0, 0, 0), 1.5f64);
    //             map
    //         }
    //         .into_py(py);
    //         let default = 1.5f64.into_py(py);
    //         model.update_table(
    //             TableUnion::Bool(t),
    //             value.as_ref(py),
    //             Some(default.as_ref(py)),
    //         )
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_bool_table_dimension_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = BoolTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_bool_table(table, Some(true), None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = vec![true].into_py(py);
    //         let default = false.into_py(py);
    //         model.update_table(
    //             TableUnion::Bool(t),
    //             value.as_ref(py),
    //             Some(default.as_ref(py)),
    //         )
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    #[test]
    fn add_int_table_1d_ok() {
        let mut model = ModelPy::default();

        let table = IntTableArgUnion::Table1D(vec![1]);
        let t1 = model.add_int_table(table, None, None);
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        assert!(matches!(t1, IntTableUnion::Table1D(_)));

        let table = IntTableArgUnion::Table1D(vec![2]);
        let t2 = model.add_int_table(table, None, None);
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        assert!(matches!(t2, IntTableUnion::Table1D(_)));

        assert_ne!(t1, t2);
    }

    #[test]
    fn add_int_table_1d_with_name_ok() {
        let mut model = ModelPy::default();

        let table = IntTableArgUnion::Table1D(vec![1]);
        let t1 = model.add_int_table(table, None, Some("t1"));
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        assert!(matches!(t1, IntTableUnion::Table1D(_)));

        let table = IntTableArgUnion::Table1D(vec![2]);
        let t2 = model.add_int_table(table, None, Some("t2"));
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        assert!(matches!(t2, IntTableUnion::Table1D(_)));

        assert_ne!(t1, t2);
    }

    #[test]
    fn add_int_table_1d_err() {
        let mut model = ModelPy::default();

        let table = IntTableArgUnion::Table1D(vec![1]);
        let t = model.add_int_table(table, None, Some("t"));
        assert!(t.is_ok());
        let table = IntTableArgUnion::Table1D(vec![1]);
        let t = model.add_int_table(table, None, Some("t"));
        assert!(t.is_err());
    }

    // #[test]
    // fn set_int_table_1d_item_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = IntTableArgUnion::Table1D(vec![1]);
    //     let t = model.add_int_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let result = Python::with_gil(|py| {
    //         let value = 2i32.into_py(py);
    //         let index = TableIndexUnion::Table1D(0);
    //         model.set_table_item(TableUnion::Int(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_ok());
    // }

    // #[test]
    // fn set_int_table_1d_item_extract_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = IntTableArgUnion::Table1D(vec![1]);
    //     let t = model.add_int_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = 1.5f64.into_py(py);
    //         let index = TableIndexUnion::Table1D(0);
    //         model.set_table_item(TableUnion::Int(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn set_int_table_1d_item_dimension_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = IntTableArgUnion::Table1D(vec![1]);
    //     let t = model.add_int_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = 2i32.into_py(py);
    //         let index = TableIndexUnion::Table2D((0, 0));
    //         model.set_table_item(TableUnion::Int(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn set_int_table_1d_no_variable_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = IntTableArgUnion::Table1D(vec![1]);
    //     let t = model.add_int_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let mut model = ModelPy::default();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = 2i32.into_py(py);
    //         let index = TableIndexUnion::Table1D(0);
    //         model.set_table_item(TableUnion::Int(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_int_table_1d_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = IntTableArgUnion::Table1D(vec![1]);
    //     let t = model.add_int_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let result = Python::with_gil(|py| {
    //         let value: Py<PyAny> = vec![2i32].into_py(py);
    //         model.update_table(TableUnion::Int(t), value.as_ref(py), None)
    //     });
    //     assert!(result.is_ok());
    // }

    // #[test]
    // fn update_int_table_1d_value_extract_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = IntTableArgUnion::Table1D(vec![1]);
    //     let t = model.add_int_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value: Py<PyAny> = vec![1.5f64].into_py(py);
    //         model.update_table(TableUnion::Int(t), value.as_ref(py), None)
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_int_table_1d_value_dimension_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = IntTableArgUnion::Table1D(vec![1]);
    //     let t = model.add_int_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value: Py<PyAny> = vec![vec![2i32]].into_py(py);
    //         model.update_table(TableUnion::Int(t), value.as_ref(py), None)
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_int_table_1d_value_no_table_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = IntTableArgUnion::Table1D(vec![1]);
    //     let t = model.add_int_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let mut model = ModelPy::default();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value: Py<PyAny> = vec![2i32].into_py(py);
    //         model.update_table(TableUnion::Int(t), value.as_ref(py), None)
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    #[test]
    fn add_int_table_2d_ok() {
        let mut model = ModelPy::default();

        let table = IntTableArgUnion::Table2D(vec![vec![1]]);
        let t1 = model.add_int_table(table, None, None);
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        assert!(matches!(t1, IntTableUnion::Table2D(_)));

        let table = IntTableArgUnion::Table2D(vec![vec![2]]);
        let t2 = model.add_int_table(table, None, None);
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        assert!(matches!(t2, IntTableUnion::Table2D(_)));

        assert_ne!(t1, t2);
    }

    #[test]
    fn add_int_table_2d_with_name_ok() {
        let mut model = ModelPy::default();

        let table = IntTableArgUnion::Table2D(vec![vec![1]]);
        let t1 = model.add_int_table(table, None, Some("t1"));
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        assert!(matches!(t1, IntTableUnion::Table2D(_)));

        let table = IntTableArgUnion::Table2D(vec![vec![2]]);
        let t2 = model.add_int_table(table, None, Some("t2"));
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        assert!(matches!(t2, IntTableUnion::Table2D(_)));

        assert_ne!(t1, t2);
    }

    #[test]
    fn add_int_table_2d_err() {
        let mut model = ModelPy::default();

        let table = IntTableArgUnion::Table2D(vec![vec![1]]);
        let t = model.add_int_table(table, None, Some("t"));
        assert!(t.is_ok());
        let table = IntTableArgUnion::Table2D(vec![vec![1]]);
        let t = model.add_int_table(table, None, Some("t"));
        assert!(t.is_err());
    }

    // #[test]
    // fn set_int_table_2d_item_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = IntTableArgUnion::Table2D(vec![vec![1]]);
    //     let t = model.add_int_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let result = Python::with_gil(|py| {
    //         let value = 2i32.into_py(py);
    //         let index = TableIndexUnion::Table2D((0, 0));
    //         model.set_table_item(TableUnion::Int(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_ok());
    // }

    // #[test]
    // fn set_int_table_2d_item_extract_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = IntTableArgUnion::Table2D(vec![vec![1]]);
    //     let t = model.add_int_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = 1.5f64.into_py(py);
    //         let index = TableIndexUnion::Table2D((0, 0));
    //         model.set_table_item(TableUnion::Int(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn set_int_table_2d_item_dimension_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = IntTableArgUnion::Table2D(vec![vec![1]]);
    //     let t = model.add_int_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = 2i32.into_py(py);
    //         let index = TableIndexUnion::Table3D((0, 0, 0));
    //         model.set_table_item(TableUnion::Int(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn set_int_table_2d_no_variable_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = IntTableArgUnion::Table2D(vec![vec![1]]);
    //     let t = model.add_int_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let mut model = ModelPy::default();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = 2i32.into_py(py);
    //         let index = TableIndexUnion::Table2D((0, 0));
    //         model.set_table_item(TableUnion::Int(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_int_table_2d_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = IntTableArgUnion::Table2D(vec![vec![1]]);
    //     let t = model.add_int_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let result = Python::with_gil(|py| {
    //         let value: Py<PyAny> = vec![vec![2i32]].into_py(py);
    //         model.update_table(TableUnion::Int(t), value.as_ref(py), None)
    //     });
    //     assert!(result.is_ok());
    // }

    // #[test]
    // fn update_int_table_2d_value_extract_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = IntTableArgUnion::Table2D(vec![vec![1]]);
    //     let t = model.add_int_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value: Py<PyAny> = vec![vec![1.5f64]].into_py(py);
    //         model.update_table(TableUnion::Int(t), value.as_ref(py), None)
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_int_table_2d_value_dimension_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = IntTableArgUnion::Table2D(vec![vec![1]]);
    //     let t = model.add_int_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value: Py<PyAny> = vec![vec![vec![2i32]]].into_py(py);
    //         model.update_table(TableUnion::Int(t), value.as_ref(py), None)
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_int_table_2d_value_no_table_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = IntTableArgUnion::Table2D(vec![vec![1]]);
    //     let t = model.add_int_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let mut model = ModelPy::default();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value: Py<PyAny> = vec![vec![2i32]].into_py(py);
    //         model.update_table(TableUnion::Int(t), value.as_ref(py), None)
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    #[test]
    fn add_int_table_3d_ok() {
        let mut model = ModelPy::default();

        let table = IntTableArgUnion::Table3D(vec![vec![vec![1]]]);
        let t1 = model.add_int_table(table, None, None);
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        assert!(matches!(t1, IntTableUnion::Table3D(_)));

        let table = IntTableArgUnion::Table3D(vec![vec![vec![2]]]);
        let t2 = model.add_int_table(table, None, None);
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        assert!(matches!(t2, IntTableUnion::Table3D(_)));

        assert_ne!(t1, t2);
    }

    #[test]
    fn add_int_table_3d_with_name_ok() {
        let mut model = ModelPy::default();

        let table = IntTableArgUnion::Table3D(vec![vec![vec![1]]]);
        let t1 = model.add_int_table(table, None, Some("t1"));
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        assert!(matches!(t1, IntTableUnion::Table3D(_)));

        let table = IntTableArgUnion::Table3D(vec![vec![vec![2]]]);
        let t2 = model.add_int_table(table, None, Some("t2"));
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        assert!(matches!(t2, IntTableUnion::Table3D(_)));

        assert_ne!(t1, t2);
    }

    #[test]
    fn add_int_table_3d_err() {
        let mut model = ModelPy::default();

        let table = IntTableArgUnion::Table3D(vec![vec![vec![1]]]);
        let t = model.add_int_table(table, None, Some("t"));
        assert!(t.is_ok());
        let table = IntTableArgUnion::Table3D(vec![vec![vec![1]]]);
        let t = model.add_int_table(table, None, Some("t"));
        assert!(t.is_err());
    }

    // #[test]
    // fn set_int_table_3d_item_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = IntTableArgUnion::Table3D(vec![vec![vec![1]]]);
    //     let t = model.add_int_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let result = Python::with_gil(|py| {
    //         let value = 2i32.into_py(py);
    //         let index = TableIndexUnion::Table3D((0, 0, 0));
    //         model.set_table_item(TableUnion::Int(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_ok());
    // }

    // #[test]
    // fn set_int_table_3d_item_extract_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = IntTableArgUnion::Table3D(vec![vec![vec![1]]]);
    //     let t = model.add_int_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = 1.5f64.into_py(py);
    //         let index = TableIndexUnion::Table3D((0, 0, 0));
    //         model.set_table_item(TableUnion::Int(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn set_int_table_3d_item_dimension_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = IntTableArgUnion::Table3D(vec![vec![vec![1]]]);
    //     let t = model.add_int_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = 2i32.into_py(py);
    //         let index = TableIndexUnion::Table2D((0, 0));
    //         model.set_table_item(TableUnion::Int(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn set_int_table_3d_no_variable_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = IntTableArgUnion::Table3D(vec![vec![vec![1]]]);
    //     let t = model.add_int_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let mut model = ModelPy::default();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = 2i32.into_py(py);
    //         let index = TableIndexUnion::Table3D((0, 0, 0));
    //         model.set_table_item(TableUnion::Int(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_int_table_3d_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = IntTableArgUnion::Table3D(vec![vec![vec![1]]]);
    //     let t = model.add_int_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let result = Python::with_gil(|py| {
    //         let value: Py<PyAny> = vec![vec![vec![2i32]]].into_py(py);
    //         model.update_table(TableUnion::Int(t), value.as_ref(py), None)
    //     });
    //     assert!(result.is_ok());
    // }

    // #[test]
    // fn update_int_table_3d_value_extract_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = IntTableArgUnion::Table3D(vec![vec![vec![1]]]);
    //     let t = model.add_int_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value: Py<PyAny> = vec![vec![vec![1.5f64]]].into_py(py);
    //         model.update_table(TableUnion::Int(t), value.as_ref(py), None)
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_int_table_3d_value_dimension_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = IntTableArgUnion::Table3D(vec![vec![vec![1]]]);
    //     let t = model.add_int_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value: Py<PyAny> = vec![vec![2i32]].into_py(py);
    //         model.update_table(TableUnion::Int(t), value.as_ref(py), None)
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_int_table_3d_value_no_table_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = IntTableArgUnion::Table3D(vec![vec![vec![1]]]);
    //     let t = model.add_int_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let mut model = ModelPy::default();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value: Py<PyAny> = vec![vec![vec![2i32]]].into_py(py);
    //         model.update_table(TableUnion::Int(t), value.as_ref(py), None)
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    #[test]
    fn add_int_table_ok() {
        let mut model = ModelPy::default();

        let table = IntTableArgUnion::Table(FxHashMap::default());
        let t1 = model.add_int_table(table, Some(1i32), None);
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        assert!(matches!(t1, IntTableUnion::Table(_)));

        let table = IntTableArgUnion::Table(FxHashMap::default());
        let t2 = model.add_int_table(table, Some(2i32), None);
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        assert!(matches!(t2, IntTableUnion::Table(_)));

        assert_ne!(t1, t2);
    }

    #[test]
    fn add_int_table_with_name_ok() {
        let mut model = ModelPy::default();

        let table = IntTableArgUnion::Table(FxHashMap::default());
        let t1 = model.add_int_table(table, Some(1i32), Some("t1"));
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        assert!(matches!(t1, IntTableUnion::Table(_)));

        let table = IntTableArgUnion::Table(FxHashMap::default());
        let t2 = model.add_int_table(table, Some(2i32), Some("t2"));
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        assert!(matches!(t1, IntTableUnion::Table(_)));

        assert_ne!(t1, t2);
    }

    #[test]
    fn add_int_table_duplicate_err() {
        let mut model = ModelPy::default();

        let table = IntTableArgUnion::Table(FxHashMap::default());
        let t = model.add_int_table(table, Some(1i32), Some("t"));
        assert!(t.is_ok());
        let table = IntTableArgUnion::Table(FxHashMap::default());
        let t = model.add_int_table(table, Some(1i32), Some("t"));
        assert!(t.is_err());
    }

    #[test]
    fn add_int_table_no_default_err() {
        let mut model = ModelPy::default();

        let table = IntTableArgUnion::Table(FxHashMap::default());
        let t = model.add_int_table(table, None, None);
        assert!(t.is_err());
    }

    // #[test]
    // fn set_int_table_item_1d_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = IntTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_int_table(table, Some(1i32), None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let result = Python::with_gil(|py| {
    //         let value = 2i32.into_py(py);
    //         let index = TableIndexUnion::Table1D(0);
    //         model.set_table_item(TableUnion::Int(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_ok());
    //     let t = match t {
    //         IntTableUnion::Table(t) => t,
    //         _ => panic!("expected IntTableUnion::Table but `{:?}`", t),
    //     };
    //     let t_id = TableHandle::<Integer>::from(t).id();
    //     assert_eq!(
    //         *model.0.table_registry.integer_tables.tables[t_id].get(&[0]),
    //         2
    //     );
    // }

    // #[test]
    // fn set_int_table_item_2d_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = IntTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_int_table(table, Some(1i32), None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let result = Python::with_gil(|py| {
    //         let value = 2i32.into_py(py);
    //         let index = TableIndexUnion::Table2D((0, 0));
    //         model.set_table_item(TableUnion::Int(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_ok());
    //     let t = match t {
    //         IntTableUnion::Table(t) => t,
    //         _ => panic!("expected IntTableUnion::Table but `{:?}`", t),
    //     };
    //     let t_id = TableHandle::<Integer>::from(t).id();
    //     assert_eq!(
    //         *model.0.table_registry.integer_tables.tables[t_id].get(&[0, 0]),
    //         2
    //     );
    // }

    // #[test]
    // fn set_int_table_item_3d_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = IntTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_int_table(table, Some(1i32), None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let result = Python::with_gil(|py| {
    //         let value = 2i32.into_py(py);
    //         let index = TableIndexUnion::Table3D((0, 0, 0));
    //         model.set_table_item(TableUnion::Int(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_ok());
    //     let t = match t {
    //         IntTableUnion::Table(t) => t,
    //         _ => panic!("expected IntTableUnion::Table but `{:?}`", t),
    //     };
    //     let t_id = TableHandle::<Integer>::from(t).id();
    //     assert_eq!(
    //         *model.0.table_registry.integer_tables.tables[t_id].get(&[0, 0, 0]),
    //         2
    //     );
    // }

    // #[test]
    // fn set_int_table_item_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = IntTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_int_table(table, Some(1i32), None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let result = Python::with_gil(|py| {
    //         let value = 2i32.into_py(py);
    //         let index = TableIndexUnion::Table(vec![0, 0, 0, 0]);
    //         model.set_table_item(TableUnion::Int(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_ok());
    // }

    // #[test]
    // fn set_int_table_item_extract_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = IntTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_int_table(table, Some(1i32), None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = 1.5f64.into_py(py);
    //         let index = TableIndexUnion::Table(vec![0, 0, 0, 0]);
    //         model.set_table_item(TableUnion::Int(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn set_int_table_item_no_table_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = IntTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_int_table(table, Some(1i32), None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let mut model = ModelPy::default();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = 2i32.into_py(py);
    //         let index = TableIndexUnion::Table(vec![0, 0, 0, 0]);
    //         model.set_table_item(TableUnion::Int(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn set_int_default_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = IntTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_int_table(table, Some(1i32), None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let t = match t {
    //         IntTableUnion::Table(t) => t,
    //         _ => panic!("expected IntTableUnion::Table but `{:?}`", t),
    //     };
    //     let result = Python::with_gil(|py| {
    //         let value = 2i32.into_py(py);
    //         model.set_default(SetDefaultArgUnion::Int(t), value.as_ref(py))
    //     });
    //     assert!(result.is_ok());
    // }

    // #[test]
    // fn set_int_default_extract_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = IntTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_int_table(table, Some(1i32), None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let t = match t {
    //         IntTableUnion::Table(t) => t,
    //         _ => panic!("expected IntTableUnion::Table but `{:?}`", t),
    //     };
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = 1.5f64.into_py(py);
    //         model.set_default(SetDefaultArgUnion::Int(t), value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn set_int_default_no_table_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = IntTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_int_table(table, Some(1i32), None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let t = match t {
    //         IntTableUnion::Table(t) => t,
    //         _ => panic!("expected IntTableUnion::Table but `{:?}`", t),
    //     };
    //     let mut model = ModelPy::default();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = 1.5f64.into_py(py);
    //         model.set_default(SetDefaultArgUnion::Int(t), value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_int_table_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = IntTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_int_table(table, Some(1i32), None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let result = Python::with_gil(|py| {
    //         let value = {
    //             let mut map = FxHashMap::default();
    //             map.insert((0, 0, 0, 0), 2);
    //             map
    //         }
    //         .into_py(py);
    //         let default = 3i32.into_py(py);
    //         model.update_table(
    //             TableUnion::Int(t),
    //             value.as_ref(py),
    //             Some(default.as_ref(py)),
    //         )
    //     });
    //     assert!(result.is_ok());
    // }

    // #[test]
    // fn update_int_table_no_default_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = IntTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_int_table(table, Some(1i32), None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = {
    //             let mut map = FxHashMap::default();
    //             map.insert((0, 0, 0, 0), 2);
    //             map
    //         }
    //         .into_py(py);
    //         model.update_table(TableUnion::Int(t), value.as_ref(py), None)
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_int_table_default_extract_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = IntTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_int_table(table, Some(1i32), None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = {
    //             let mut map = FxHashMap::default();
    //             map.insert((0, 0, 0, 0), 2);
    //             map
    //         }
    //         .into_py(py);
    //         let default = 1.5f64.into_py(py);
    //         model.update_table(
    //             TableUnion::Int(t),
    //             value.as_ref(py),
    //             Some(default.as_ref(py)),
    //         )
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_int_table_value_extract_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = IntTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_int_table(table, Some(1i32), None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = {
    //             let mut map = FxHashMap::default();
    //             map.insert((0, 0, 0, 0), 1.5f64);
    //             map
    //         }
    //         .into_py(py);
    //         let default = 1.5f64.into_py(py);
    //         model.update_table(
    //             TableUnion::Int(t),
    //             value.as_ref(py),
    //             Some(default.as_ref(py)),
    //         )
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_int_table_dimension_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = IntTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_int_table(table, Some(1i32), None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = vec![2i32].into_py(py);
    //         let default = 3i32.into_py(py);
    //         model.update_table(
    //             TableUnion::Int(t),
    //             value.as_ref(py),
    //             Some(default.as_ref(py)),
    //         )
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    #[test]
    fn add_float_table_1d_ok() {
        let mut model = ModelPy::default();

        let table = FloatTableArgUnion::Table1D(vec![1.0]);
        let t1 = model.add_float_table(table, None, None);
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        assert!(matches!(t1, FloatTableUnion::Table1D(_)));

        let table = FloatTableArgUnion::Table1D(vec![2.0]);
        let t2 = model.add_float_table(table, None, None);
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        assert!(matches!(t2, FloatTableUnion::Table1D(_)));

        assert_ne!(t1, t2);
    }

    #[test]
    fn add_float_table_1d_with_name_ok() {
        let mut model = ModelPy::default();

        let table = FloatTableArgUnion::Table1D(vec![1.0]);
        let t1 = model.add_float_table(table, None, Some("t1"));
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        assert!(matches!(t1, FloatTableUnion::Table1D(_)));

        let table = FloatTableArgUnion::Table1D(vec![2.0]);
        let t2 = model.add_float_table(table, None, Some("t2"));
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        assert!(matches!(t2, FloatTableUnion::Table1D(_)));

        assert_ne!(t1, t2);
    }

    #[test]
    fn add_float_table_1d_err() {
        let mut model = ModelPy::default();

        let table = FloatTableArgUnion::Table1D(vec![1.0]);
        let t = model.add_float_table(table, None, Some("t"));
        assert!(t.is_ok());
        let table = FloatTableArgUnion::Table1D(vec![1.0]);
        let t = model.add_float_table(table, None, Some("t"));
        assert!(t.is_err());
    }

    // #[test]
    // fn set_float_table_1d_item_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = FloatTableArgUnion::Table1D(vec![1.0]);
    //     let t = model.add_float_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let result = Python::with_gil(|py| {
    //         let value = 2f64.into_py(py);
    //         let index = TableIndexUnion::Table1D(0);
    //         model.set_table_item(TableUnion::Float(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_ok());
    // }

    // #[test]
    // fn set_float_table_1d_item_extract_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = FloatTableArgUnion::Table1D(vec![1.0]);
    //     let t = model.add_float_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = vec![1.5f64].into_py(py);
    //         let index = TableIndexUnion::Table1D(0);
    //         model.set_table_item(TableUnion::Float(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn set_float_table_1d_item_dimension_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = FloatTableArgUnion::Table1D(vec![1.0]);
    //     let t = model.add_float_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = 2f64.into_py(py);
    //         let index = TableIndexUnion::Table2D((0, 0));
    //         model.set_table_item(TableUnion::Float(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn set_float_table_1d_no_variable_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = FloatTableArgUnion::Table1D(vec![1.0]);
    //     let t = model.add_float_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let mut model = ModelPy::default();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = 2f64.into_py(py);
    //         let index = TableIndexUnion::Table1D(0);
    //         model.set_table_item(TableUnion::Float(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_float_table_1d_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = FloatTableArgUnion::Table1D(vec![1.0]);
    //     let t = model.add_float_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let result = Python::with_gil(|py| {
    //         let value: Py<PyAny> = vec![2f64].into_py(py);
    //         model.update_table(TableUnion::Float(t), value.as_ref(py), None)
    //     });
    //     assert!(result.is_ok());
    // }

    // #[test]
    // fn update_float_table_1d_value_extract_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = FloatTableArgUnion::Table1D(vec![1.0]);
    //     let t = model.add_float_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value: Py<PyAny> = vec![vec![1.5f64]].into_py(py);
    //         model.update_table(TableUnion::Float(t), value.as_ref(py), None)
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_float_table_1d_value_dimension_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = FloatTableArgUnion::Table1D(vec![1.0]);
    //     let t = model.add_float_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value: Py<PyAny> = vec![vec![2f64]].into_py(py);
    //         model.update_table(TableUnion::Float(t), value.as_ref(py), None)
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_float_table_1d_value_no_table_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = FloatTableArgUnion::Table1D(vec![1.0]);
    //     let t = model.add_float_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let mut model = ModelPy::default();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value: Py<PyAny> = vec![2f64].into_py(py);
    //         model.update_table(TableUnion::Float(t), value.as_ref(py), None)
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    #[test]
    fn add_float_table_2d_ok() {
        let mut model = ModelPy::default();

        let table = FloatTableArgUnion::Table2D(vec![vec![1.0]]);
        let t1 = model.add_float_table(table, None, None);
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        assert!(matches!(t1, FloatTableUnion::Table2D(_)));

        let table = FloatTableArgUnion::Table2D(vec![vec![2.0]]);
        let t2 = model.add_float_table(table, None, None);
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        assert!(matches!(t2, FloatTableUnion::Table2D(_)));

        assert_ne!(t1, t2);
    }

    #[test]
    fn add_float_table_2d_with_name_ok() {
        let mut model = ModelPy::default();

        let table = FloatTableArgUnion::Table2D(vec![vec![1.0]]);
        let t1 = model.add_float_table(table, None, Some("t1"));
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        assert!(matches!(t1, FloatTableUnion::Table2D(_)));

        let table = FloatTableArgUnion::Table2D(vec![vec![2.0]]);
        let t2 = model.add_float_table(table, None, Some("t2"));
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        assert!(matches!(t2, FloatTableUnion::Table2D(_)));

        assert_ne!(t1, t2);
    }

    #[test]
    fn add_float_table_2d_err() {
        let mut model = ModelPy::default();

        let table = FloatTableArgUnion::Table2D(vec![vec![1.0]]);
        let t = model.add_float_table(table, None, Some("t"));
        assert!(t.is_ok());
        let table = FloatTableArgUnion::Table2D(vec![vec![1.0]]);
        let t = model.add_float_table(table, None, Some("t"));
        assert!(t.is_err());
    }

    // #[test]
    // fn set_float_table_2d_item_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = FloatTableArgUnion::Table2D(vec![vec![1.0]]);
    //     let t = model.add_float_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let result = Python::with_gil(|py| {
    //         let value = 2f64.into_py(py);
    //         let index = TableIndexUnion::Table2D((0, 0));
    //         model.set_table_item(TableUnion::Float(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_ok());
    // }

    // #[test]
    // fn set_float_table_2d_item_extract_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = FloatTableArgUnion::Table2D(vec![vec![1.0]]);
    //     let t = model.add_float_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = vec![1.5f64].into_py(py);
    //         let index = TableIndexUnion::Table2D((0, 0));
    //         model.set_table_item(TableUnion::Float(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn set_float_table_2d_item_dimension_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = FloatTableArgUnion::Table2D(vec![vec![1.0]]);
    //     let t = model.add_float_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = 2f64.into_py(py);
    //         let index = TableIndexUnion::Table3D((0, 0, 0));
    //         model.set_table_item(TableUnion::Float(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn set_float_table_2d_no_variable_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = FloatTableArgUnion::Table2D(vec![vec![1.0]]);
    //     let t = model.add_float_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let mut model = ModelPy::default();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = 2f64.into_py(py);
    //         let index = TableIndexUnion::Table2D((0, 0));
    //         model.set_table_item(TableUnion::Float(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_float_table_2d_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = FloatTableArgUnion::Table2D(vec![vec![1.0]]);
    //     let t = model.add_float_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let result = Python::with_gil(|py| {
    //         let value: Py<PyAny> = vec![vec![2f64]].into_py(py);
    //         model.update_table(TableUnion::Float(t), value.as_ref(py), None)
    //     });
    //     assert!(result.is_ok());
    // }

    // #[test]
    // fn update_float_table_2d_value_extract_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = FloatTableArgUnion::Table2D(vec![vec![1.0]]);
    //     let t = model.add_float_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value: Py<PyAny> = vec![vec![vec![1.5f64]]].into_py(py);
    //         model.update_table(TableUnion::Float(t), value.as_ref(py), None)
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_float_table_2d_value_dimension_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = FloatTableArgUnion::Table2D(vec![vec![1.0]]);
    //     let t = model.add_float_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value: Py<PyAny> = vec![vec![vec![2f64]]].into_py(py);
    //         model.update_table(TableUnion::Float(t), value.as_ref(py), None)
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_float_table_2d_value_no_table_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = FloatTableArgUnion::Table2D(vec![vec![1.0]]);
    //     let t = model.add_float_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let mut model = ModelPy::default();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value: Py<PyAny> = vec![vec![2f64]].into_py(py);
    //         model.update_table(TableUnion::Float(t), value.as_ref(py), None)
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    #[test]
    fn add_float_table_3d_ok() {
        let mut model = ModelPy::default();

        let table = FloatTableArgUnion::Table3D(vec![vec![vec![1.0]]]);
        let t1 = model.add_float_table(table, None, None);
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        assert!(matches!(t1, FloatTableUnion::Table3D(_)));

        let table = FloatTableArgUnion::Table3D(vec![vec![vec![2.0]]]);
        let t2 = model.add_float_table(table, None, None);
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        assert!(matches!(t2, FloatTableUnion::Table3D(_)));

        assert_ne!(t1, t2);
    }

    #[test]
    fn add_float_table_3d_with_name_ok() {
        let mut model = ModelPy::default();

        let table = FloatTableArgUnion::Table3D(vec![vec![vec![1.0]]]);
        let t1 = model.add_float_table(table, None, Some("t1"));
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        assert!(matches!(t1, FloatTableUnion::Table3D(_)));

        let table = FloatTableArgUnion::Table3D(vec![vec![vec![2.0]]]);
        let t2 = model.add_float_table(table, None, Some("t2"));
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        assert!(matches!(t2, FloatTableUnion::Table3D(_)));

        assert_ne!(t1, t2);
    }

    #[test]
    fn add_float_table_3d_err() {
        let mut model = ModelPy::default();

        let table = FloatTableArgUnion::Table3D(vec![vec![vec![1.0]]]);
        let t = model.add_float_table(table, None, Some("t"));
        assert!(t.is_ok());
        let table = FloatTableArgUnion::Table3D(vec![vec![vec![1.0]]]);
        let t = model.add_float_table(table, None, Some("t"));
        assert!(t.is_err());
    }

    // #[test]
    // fn set_float_table_3d_item_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = FloatTableArgUnion::Table3D(vec![vec![vec![1.0]]]);
    //     let t = model.add_float_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let result = Python::with_gil(|py| {
    //         let value = 2f64.into_py(py);
    //         let index = TableIndexUnion::Table3D((0, 0, 0));
    //         model.set_table_item(TableUnion::Float(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_ok());
    // }

    // #[test]
    // fn set_float_table_3d_item_extract_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = FloatTableArgUnion::Table3D(vec![vec![vec![1.0]]]);
    //     let t = model.add_float_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = vec![1.5f64].into_py(py);
    //         let index = TableIndexUnion::Table3D((0, 0, 0));
    //         model.set_table_item(TableUnion::Float(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn set_float_table_3d_item_dimension_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = FloatTableArgUnion::Table3D(vec![vec![vec![1.0]]]);
    //     let t = model.add_float_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = 2f64.into_py(py);
    //         let index = TableIndexUnion::Table2D((0, 0));
    //         model.set_table_item(TableUnion::Float(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn set_float_table_3d_no_variable_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = FloatTableArgUnion::Table3D(vec![vec![vec![1.0]]]);
    //     let t = model.add_float_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let mut model = ModelPy::default();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = 2f64.into_py(py);
    //         let index = TableIndexUnion::Table3D((0, 0, 0));
    //         model.set_table_item(TableUnion::Float(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_float_table_3d_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = FloatTableArgUnion::Table3D(vec![vec![vec![1.0]]]);
    //     let t = model.add_float_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let result = Python::with_gil(|py| {
    //         let value: Py<PyAny> = vec![vec![vec![2f64]]].into_py(py);
    //         model.update_table(TableUnion::Float(t), value.as_ref(py), None)
    //     });
    //     assert!(result.is_ok());
    // }

    // #[test]
    // fn update_float_table_3d_value_extract_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = FloatTableArgUnion::Table3D(vec![vec![vec![1.0]]]);
    //     let t = model.add_float_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value: Py<PyAny> = vec![vec![vec![vec![1.5f64]]]].into_py(py);
    //         model.update_table(TableUnion::Float(t), value.as_ref(py), None)
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_float_table_3d_value_dimension_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = FloatTableArgUnion::Table3D(vec![vec![vec![1.0]]]);
    //     let t = model.add_float_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value: Py<PyAny> = vec![vec![2f64]].into_py(py);
    //         model.update_table(TableUnion::Float(t), value.as_ref(py), None)
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_float_table_3d_value_no_table_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = FloatTableArgUnion::Table3D(vec![vec![vec![1.0]]]);
    //     let t = model.add_float_table(table, None, None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let mut model = ModelPy::default();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value: Py<PyAny> = vec![vec![vec![2f64]]].into_py(py);
    //         model.update_table(TableUnion::Float(t), value.as_ref(py), None)
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    #[test]
    fn add_float_table_ok() {
        let mut model = ModelPy::default();

        let table = FloatTableArgUnion::Table(FxHashMap::default());
        let t1 = model.add_float_table(table, Some(1f64), None);
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        assert!(matches!(t1, FloatTableUnion::Table(_)));

        let table = FloatTableArgUnion::Table(FxHashMap::default());
        let t2 = model.add_float_table(table, Some(2f64), None);
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        assert!(matches!(t2, FloatTableUnion::Table(_)));

        assert_ne!(t1, t2);
    }

    #[test]
    fn add_float_table_with_name_ok() {
        let mut model = ModelPy::default();

        let table = FloatTableArgUnion::Table(FxHashMap::default());
        let t1 = model.add_float_table(table, Some(1f64), Some("t1"));
        assert!(t1.is_ok());
        let t1 = t1.unwrap();
        assert!(matches!(t1, FloatTableUnion::Table(_)));

        let table = FloatTableArgUnion::Table(FxHashMap::default());
        let t2 = model.add_float_table(table, Some(2f64), Some("t2"));
        assert!(t2.is_ok());
        let t2 = t2.unwrap();
        assert!(matches!(t1, FloatTableUnion::Table(_)));

        assert_ne!(t1, t2);
    }

    #[test]
    fn add_float_table_duplicate_err() {
        let mut model = ModelPy::default();

        let table = FloatTableArgUnion::Table(FxHashMap::default());
        let t = model.add_float_table(table, Some(1f64), Some("t"));
        assert!(t.is_ok());
        let table = FloatTableArgUnion::Table(FxHashMap::default());
        let t = model.add_float_table(table, Some(1f64), Some("t"));
        assert!(t.is_err());
    }

    #[test]
    fn add_float_table_no_default_err() {
        let mut model = ModelPy::default();

        let table = FloatTableArgUnion::Table(FxHashMap::default());
        let t = model.add_float_table(table, None, None);
        assert!(t.is_err());
    }

    // #[test]
    // fn set_float_table_item_1d_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = FloatTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_float_table(table, Some(1f64), None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let result = Python::with_gil(|py| {
    //         let value = 2f64.into_py(py);
    //         let index = TableIndexUnion::Table1D(0);
    //         model.set_table_item(TableUnion::Float(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_ok());
    //     let t = match t {
    //         FloatTableUnion::Table(t) => t,
    //         _ => panic!("expected FloatTableUnion::Table but `{:?}`", t),
    //     };
    //     let t_id = TableHandle::<Continuous>::from(t).id();
    //     assert_eq!(
    //         *model.0.table_registry.continuous_tables.tables[t_id].get(&[0]),
    //         2.0
    //     );
    // }

    // #[test]
    // fn set_float_table_item_2d_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = FloatTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_float_table(table, Some(1f64), None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let result = Python::with_gil(|py| {
    //         let value = 2f64.into_py(py);
    //         let index = TableIndexUnion::Table2D((0, 0));
    //         model.set_table_item(TableUnion::Float(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_ok());
    //     let t = match t {
    //         FloatTableUnion::Table(t) => t,
    //         _ => panic!("expected FloatTableUnion::Table but `{:?}`", t),
    //     };
    //     let t_id = TableHandle::<Continuous>::from(t).id();
    //     assert_eq!(
    //         *model.0.table_registry.continuous_tables.tables[t_id].get(&[0, 0]),
    //         2.0
    //     );
    // }

    // #[test]
    // fn set_float_table_item_3d_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = FloatTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_float_table(table, Some(1f64), None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let result = Python::with_gil(|py| {
    //         let value = 2f64.into_py(py);
    //         let index = TableIndexUnion::Table3D((0, 0, 0));
    //         model.set_table_item(TableUnion::Float(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_ok());
    //     let t = match t {
    //         FloatTableUnion::Table(t) => t,
    //         _ => panic!("expected FloatTableUnion::Table but `{:?}`", t),
    //     };
    //     let t_id = TableHandle::<Continuous>::from(t).id();
    //     assert_eq!(
    //         *model.0.table_registry.continuous_tables.tables[t_id].get(&[0, 0, 0]),
    //         2.0
    //     );
    // }

    // #[test]
    // fn set_float_table_item_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = FloatTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_float_table(table, Some(1f64), None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let result = Python::with_gil(|py| {
    //         let value = 2f64.into_py(py);
    //         let index = TableIndexUnion::Table(vec![0, 0, 0, 0]);
    //         model.set_table_item(TableUnion::Float(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_ok());
    // }

    // #[test]
    // fn set_float_table_item_extract_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = FloatTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_float_table(table, Some(1f64), None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = vec![1.5f64].into_py(py);
    //         let index = TableIndexUnion::Table(vec![0, 0, 0, 0]);
    //         model.set_table_item(TableUnion::Float(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn set_float_table_item_no_table_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = FloatTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_float_table(table, Some(1f64), None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let mut model = ModelPy::default();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = 2f64.into_py(py);
    //         let index = TableIndexUnion::Table(vec![0, 0, 0, 0]);
    //         model.set_table_item(TableUnion::Float(t), index, value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn set_float_default_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = FloatTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_float_table(table, Some(1f64), None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let t = match t {
    //         FloatTableUnion::Table(t) => t,
    //         _ => panic!("expected FloatTableUnion::Table but `{:?}`", t),
    //     };
    //     let result = Python::with_gil(|py| {
    //         let value = 2f64.into_py(py);
    //         model.set_default(SetDefaultArgUnion::Float(t), value.as_ref(py))
    //     });
    //     assert!(result.is_ok());
    // }

    // #[test]
    // fn set_float_default_extract_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = FloatTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_float_table(table, Some(1f64), None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let t = match t {
    //         FloatTableUnion::Table(t) => t,
    //         _ => panic!("expected FloatTableUnion::Table but `{:?}`", t),
    //     };
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = vec![1.5f64].into_py(py);
    //         model.set_default(SetDefaultArgUnion::Float(t), value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn set_float_default_no_table_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = FloatTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_float_table(table, Some(1f64), None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let t = match t {
    //         FloatTableUnion::Table(t) => t,
    //         _ => panic!("expected FloatTableUnion::Table but `{:?}`", t),
    //     };
    //     let mut model = ModelPy::default();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = vec![1.5f64].into_py(py);
    //         model.set_default(SetDefaultArgUnion::Float(t), value.as_ref(py))
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_float_table_ok() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = FloatTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_float_table(table, Some(1f64), None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let result = Python::with_gil(|py| {
    //         let value = {
    //             let mut map = FxHashMap::default();
    //             map.insert((0, 0, 0, 0), 2);
    //             map
    //         }
    //         .into_py(py);
    //         let default = 3f64.into_py(py);
    //         model.update_table(
    //             TableUnion::Float(t),
    //             value.as_ref(py),
    //             Some(default.as_ref(py)),
    //         )
    //     });
    //     assert!(result.is_ok());
    // }

    // #[test]
    // fn update_float_table_no_default_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = FloatTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_float_table(table, Some(1f64), None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = {
    //             let mut map = FxHashMap::default();
    //             map.insert((0, 0, 0, 0), 2);
    //             map
    //         }
    //         .into_py(py);
    //         model.update_table(TableUnion::Float(t), value.as_ref(py), None)
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_float_table_default_extract_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = FloatTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_float_table(table, Some(1f64), None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = {
    //             let mut map = FxHashMap::default();
    //             map.insert((0, 0, 0, 0), 2);
    //             map
    //         }
    //         .into_py(py);
    //         let default = vec![1.5f64].into_py(py);
    //         model.update_table(
    //             TableUnion::Float(t),
    //             value.as_ref(py),
    //             Some(default.as_ref(py)),
    //         )
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_float_table_value_extract_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = FloatTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_float_table(table, Some(1f64), None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = {
    //             let mut map = FxHashMap::default();
    //             map.insert((0, 0, 0, 0), vec![1.5f64]);
    //             map
    //         }
    //         .into_py(py);
    //         let default = vec![1.5f64].into_py(py);
    //         model.update_table(
    //             TableUnion::Float(t),
    //             value.as_ref(py),
    //             Some(default.as_ref(py)),
    //         )
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }

    // #[test]
    // fn update_float_table_dimension_err() {
    //     pyo3::prepare_freethreaded_python();

    //     let mut model = ModelPy::default();
    //     let table = FloatTableArgUnion::Table(FxHashMap::default());
    //     let t = model.add_float_table(table, Some(1f64), None);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let snapshot = model.clone();
    //     let result = Python::with_gil(|py| {
    //         let value = vec![2f64].into_py(py);
    //         let default = 3f64.into_py(py);
    //         model.update_table(
    //             TableUnion::Float(t),
    //             value.as_ref(py),
    //             Some(default.as_ref(py)),
    //         )
    //     });
    //     assert!(result.is_err());
    //     assert_eq!(model, snapshot);
    // }
}
