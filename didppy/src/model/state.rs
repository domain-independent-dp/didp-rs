use super::expression::*;
use dypdl::prelude::*;
use dypdl::variable_type;
use dypdl::StateInterface;
use pyo3::prelude::*;
use std::collections::HashSet;

#[derive(FromPyObject, Debug, PartialEq, Clone, IntoPyObject)]
pub enum VariableValueUnion {
    #[pyo3(transparent)]
    Element(variable_type::Element),
    #[pyo3(transparent)]
    Set(HashSet<Element>),
    #[pyo3(transparent)]
    Int(variable_type::Integer),
    #[pyo3(transparent)]
    Float(variable_type::Continuous),
}

/// DyPDL state.
///
/// Values of state variables can be accessed by :code:`state[var]`, where :code:`state` is :class:`State` and
/// :code:`var` is either of :class:`ElementVar`, :class:`ElementResourceVar`, :class:`SetVar`, :class:`IntVar`, :class:`IntResourceVar`, :class:`FloatVar`, and :class:`FloatResourceVar`.
///
/// Examples
/// --------
/// >>> import didppy as dp
/// >>> model = dp.Model()
/// >>> var = model.add_int_var(target=4)
/// >>> state = model.target_state
/// >>> state[var]
/// 4
/// >>> state[var] = 5
/// >>> state[var]
/// 5
#[pyclass(name = "State")]
#[derive(Debug, PartialEq, Clone, Default)]
pub struct StatePy(State);

impl From<StatePy> for State {
    fn from(state: StatePy) -> Self {
        state.0
    }
}

impl From<State> for StatePy {
    fn from(state: State) -> Self {
        StatePy(state)
    }
}

impl StatePy {
    pub fn inner_as_ref(&self) -> &State {
        &self.0
    }
}

#[pymethods]
impl StatePy {
    fn __getitem__(&self, var: VarUnion) -> VariableValueUnion {
        match var {
            VarUnion::Element(var) => VariableValueUnion::Element(
                self.0.get_element_variable(ElementVariable::from(var).id()),
            ),
            VarUnion::ElementResource(var) => VariableValueUnion::Element(
                self.0
                    .get_element_resource_variable(ElementResourceVariable::from(var).id()),
            ),
            VarUnion::Set(var) => VariableValueUnion::Set(HashSet::from_iter(
                self.0.get_set_variable(SetVariable::from(var).id()).ones(),
            )),
            VarUnion::Int(var) => VariableValueUnion::Int(
                self.0.get_integer_variable(IntegerVariable::from(var).id()),
            ),
            VarUnion::IntResource(var) => VariableValueUnion::Int(
                self.0
                    .get_integer_resource_variable(IntegerResourceVariable::from(var).id()),
            ),
            VarUnion::Float(var) => VariableValueUnion::Float(
                self.0
                    .get_continuous_variable(ContinuousVariable::from(var).id()),
            ),
            VarUnion::FloatResource(var) => VariableValueUnion::Float(
                self.0
                    .get_continuous_resource_variable(ContinuousResourceVariable::from(var).id()),
            ),
        }
    }

    fn __setitem__(&mut self, var: VarUnion, value: Bound<'_, PyAny>) -> PyResult<()> {
        match var {
            VarUnion::Element(var) => {
                let var = ElementVariable::from(var);
                self.0.signature_variables.element_variables[var.id()] = value.extract()?;
            }
            VarUnion::ElementResource(var) => {
                let var = ElementResourceVariable::from(var);
                self.0.resource_variables.element_variables[var.id()] = value.extract()?;
            }
            VarUnion::Set(var) => {
                let var = SetVariable::from(var);
                let value = value.extract::<SetConstPy>()?.into();
                self.0.signature_variables.set_variables[var.id()] = value;
            }
            VarUnion::Int(var) => {
                let var = IntegerVariable::from(var);
                self.0.signature_variables.integer_variables[var.id()] = value.extract()?;
            }
            VarUnion::IntResource(var) => {
                let var = IntegerResourceVariable::from(var);
                self.0.resource_variables.integer_variables[var.id()] = value.extract()?;
            }
            VarUnion::Float(var) => {
                let var = ContinuousVariable::from(var);
                self.0.signature_variables.continuous_variables[var.id()] = value.extract()?;
            }
            VarUnion::FloatResource(var) => {
                let var = ContinuousResourceVariable::from(var);
                self.0.resource_variables.continuous_variables[var.id()] = value.extract()?;
            }
        };
        Ok(())
    }
}
