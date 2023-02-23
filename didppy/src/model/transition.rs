use crate::ModelPy;

use super::expression::*;
use super::state::StatePy;
use dypdl::expression::*;
use dypdl::prelude::*;
use dypdl::TransitionInterface;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

#[derive(FromPyObject, Debug, PartialEq, Clone)]
pub enum CostUnion {
    #[pyo3(transparent)]
    Int(IntUnion),
    #[pyo3(transparent)]
    Float(FloatUnion),
}

impl From<CostUnion> for CostExpression {
    fn from(cost: CostUnion) -> Self {
        match cost {
            CostUnion::Int(cost) => CostExpression::Integer(IntegerExpression::from(cost)),
            CostUnion::Float(cost) => CostExpression::Continuous(ContinuousExpression::from(cost)),
        }
    }
}

#[derive(FromPyObject, Debug, Clone, Copy, PartialEq)]
pub enum IntOrFloat {
    #[pyo3(transparent, annotation = "int")]
    Int(Integer),
    #[pyo3(transparent, annotation = "float")]
    Float(Continuous),
}

impl IntoPy<Py<PyAny>> for IntOrFloat {
    fn into_py(self, py: Python<'_>) -> Py<PyAny> {
        match self {
            Self::Int(int) => int.into_py(py),
            Self::Float(float) => float.into_py(py),
        }
    }
}
/// Transition.
///
/// An effect on a variable can be accessed by `transition[var]`, where `transition` is :class:`Transition` and
/// `var` is either of :class:`ElementVar`, :class:`ElementResourceVar`, :class:`SetVar`, :class:`IntVar`, :class:`IntResourceVar`, :class:`FloatVar`, and :class:`FloatResourceVar`.
///
/// Parameters
/// ----------
/// name: str
///     Name of the transition.
/// cost: IntExpr, IntVar, IntResourceVar, FloatExpr, FloatVar, FloatResourceVar, int, float, or None, default: None
///     Cost expression.
///     :func:`IntExpr.state_cost()` or :func:`FloatExpr.state_cost()` can be used to represent the cost of the transitioned state.
///     If `None`, :func:`IntExpr.state_cost()` is used.
/// preconditions: list of Condition or None, default: None
///     Preconditions, which must be satisfied by a state to be applicable.
/// effects: list of tuple of a variable and an expression or None, default: None
///     Effects, a sequence of tuple of a variable and an expression.
///     Instead of an expression, a variable or an immediate value can be used.
///
/// Raises
/// ------
/// RuntimeError
///     If multiple effects are defined for the same variable.
/// TypeError
///     If the types of a variable and an expression in `effects` mismatch.
///
/// Examples
/// --------
/// >>> import didppy as dp
/// >>> model = dp.Model()
/// >>> var = model.add_int_var(target=4)
/// >>> t = dp.Transition(
/// ...     name="t",
/// ...     cost=dp.IntExpr.state_cost() + 1,
/// ...     preconditions=[var >= 1],
/// ...     effects=[(var, var - 1)],
/// ... )
/// >>> state = model.target_state
/// >>> t.cost.eval_cost(0, state, model)
/// 1
/// >>> t.cost = dp.IntExpr.state_cost() + 2
/// >>> t.cost.eval_cost(0, state, model)
/// 2
/// >>> preconditions = t.preconditions
/// >>> preconditions[0].eval(state, model)
/// True
/// >>> t[var].eval(state, model)
/// 3
/// >>> t[var] = var + 1
/// >>> t[var].eval(state, model)
/// 5
#[pyclass(name = "Transition")]
#[pyo3(text_signature = "(name, cost=None, preconditions=None, effects=None)")]
#[derive(Debug, PartialEq, Clone, Default)]
pub struct TransitionPy(Transition);

impl From<TransitionPy> for Transition {
    fn from(transition: TransitionPy) -> Self {
        transition.0
    }
}

impl From<Transition> for TransitionPy {
    fn from(transition: Transition) -> Self {
        TransitionPy(transition)
    }
}

impl TransitionPy {
    pub fn inner_as_ref(&self) -> &Transition {
        &self.0
    }

    fn get_effect<T: Clone>(var_id: usize, effects: &[(usize, T)]) -> Option<T> {
        for (id, effect) in effects {
            if *id == var_id {
                return Some(effect.clone());
            }
        }
        None
    }

    fn set_effect<T: Clone>(var_id: usize, new_effect: T, effects: &mut Vec<(usize, T)>) {
        for (id, effect) in effects.iter_mut() {
            if *id == var_id {
                *effect = new_effect;
                return;
            }
        }

        effects.push((var_id, new_effect));
    }
}

#[pymethods]
impl TransitionPy {
    #[new]
    #[pyo3(signature = (name, cost = None, preconditions = vec![], effects = vec![]))]
    pub fn new_py(
        name: &str,
        cost: Option<CostUnion>,
        preconditions: Option<Vec<ConditionPy>>,
        effects: Option<Vec<(VarUnion, &PyAny)>>,
    ) -> PyResult<TransitionPy> {
        let mut transition = TransitionPy(Transition::new(name));
        if let Some(cost) = cost {
            transition.set_cost(cost);
        }
        if let Some(preconditions) = preconditions {
            for condition in preconditions {
                transition.add_precondition(condition);
            }
        }
        if let Some(effects) = effects {
            for (var, expr) in effects {
                transition.add_effect(var, expr)?;
            }
        }
        Ok(transition)
    }

    /// str : Name of the transition.
    #[getter]
    pub fn name(&self) -> String {
        self.0.get_full_name()
    }

    #[setter]
    pub fn set_name(&mut self, name: &str) {
        self.0.name = name.to_string();
    }

    /// IntExpr or FloatExpr : Cost expression.
    #[getter]
    pub fn cost(&self) -> IntOrFloatExpr {
        match self.0.cost {
            CostExpression::Integer(ref cost) => IntOrFloatExpr::Int(IntExprPy::from(cost.clone())),
            CostExpression::Continuous(ref cost) => {
                IntOrFloatExpr::Float(FloatExprPy::from(cost.clone()))
            }
        }
    }

    #[setter]
    fn set_cost(&mut self, cost: CostUnion) {
        match cost {
            CostUnion::Int(cost) => self.0.set_cost(IntegerExpression::from(cost)),
            CostUnion::Float(cost) => self.0.set_cost(ContinuousExpression::from(cost)),
        }
    }

    /// list of Condition : Preconditions.   
    /// Note that the order of preconditions may differ from the order in which they are added.
    #[getter]
    fn preconditions(&self) -> Vec<ConditionPy> {
        self.0
            .elements_in_set_variable
            .iter()
            .map(|(var, element)| {
                Condition::Set(Box::new(SetCondition::IsIn(
                    ElementExpression::Constant(*element),
                    SetExpression::Reference(ReferenceExpression::Variable(*var)),
                )))
            })
            .chain(
                self.0
                    .preconditions
                    .iter()
                    .map(|condition| condition.condition.clone()),
            )
            .map(ConditionPy::from)
            .collect()
    }

    /// add_precondition(condition)
    ///
    /// Adds a precondition to the transition.
    ///
    /// Parameters
    /// ----------
    /// condition: Condition
    ///     Precondition.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> var = model.add_int_var(target=4)
    /// >>> t = dp.Transition(name="t")
    /// >>> t.add_precondition(var >= 1)
    /// >>> t.preconditions[0].eval(model.target_state, model)
    /// True
    #[pyo3(signature = (condition))]
    fn add_precondition(&mut self, condition: ConditionPy) {
        self.0.add_precondition(condition.into())
    }

    fn __getitem__(&self, var: VarUnion) -> ExprUnion {
        match var {
            VarUnion::Element(var) => {
                let id = ElementVariable::from(var).id();
                let effect = Self::get_effect(id, &self.0.effect.element_effects);

                if let Some(effect) = effect {
                    ExprUnion::Element(ElementExprPy::from(effect))
                } else {
                    ExprUnion::Element(ElementExprPy::from(ElementExpression::from(var)))
                }
            }
            VarUnion::ElementResource(var) => {
                let id = ElementResourceVariable::from(var).id();
                let effect = Self::get_effect(id, &self.0.effect.element_resource_effects);

                if let Some(effect) = effect {
                    ExprUnion::Element(ElementExprPy::from(effect))
                } else {
                    ExprUnion::Element(ElementExprPy::from(ElementExpression::from(var)))
                }
            }
            VarUnion::Set(var) => {
                let id = SetVariable::from(var).id();
                let effect = Self::get_effect(id, &self.0.effect.set_effects);

                if let Some(effect) = effect {
                    ExprUnion::Set(SetExprPy::from(effect))
                } else {
                    ExprUnion::Set(SetExprPy::from(SetExpression::from(var)))
                }
            }
            VarUnion::Int(var) => {
                let id = IntegerVariable::from(var).id();
                let effect = Self::get_effect(id, &self.0.effect.integer_effects);

                if let Some(effect) = effect {
                    ExprUnion::Int(IntExprPy::from(effect))
                } else {
                    ExprUnion::Int(IntExprPy::from(IntegerExpression::from(var)))
                }
            }
            VarUnion::IntResource(var) => {
                let id = IntegerResourceVariable::from(var).id();
                let effect = Self::get_effect(id, &self.0.effect.integer_resource_effects);

                if let Some(effect) = effect {
                    ExprUnion::Int(IntExprPy::from(effect))
                } else {
                    ExprUnion::Int(IntExprPy::from(IntegerExpression::from(var)))
                }
            }
            VarUnion::Float(var) => {
                let id = ContinuousVariable::from(var).id();
                let effect = Self::get_effect(id, &self.0.effect.continuous_effects);

                if let Some(effect) = effect {
                    ExprUnion::Float(FloatExprPy::from(effect))
                } else {
                    ExprUnion::Float(FloatExprPy::from(ContinuousExpression::from(var)))
                }
            }
            VarUnion::FloatResource(var) => {
                let id = ContinuousResourceVariable::from(var).id();
                let effect = Self::get_effect(id, &self.0.effect.continuous_resource_effects);

                if let Some(effect) = effect {
                    ExprUnion::Float(FloatExprPy::from(effect))
                } else {
                    ExprUnion::Float(FloatExprPy::from(ContinuousExpression::from(var)))
                }
            }
        }
    }

    fn __setitem__(&mut self, var: VarUnion, expr: &PyAny) -> PyResult<()> {
        match var {
            VarUnion::Element(var) => {
                let var = ElementVariable::from(var);
                let expr: ElementUnion = expr.extract()?;
                let expr = ElementExpression::from(expr);
                Self::set_effect(var.id(), expr, &mut self.0.effect.element_effects);
            }
            VarUnion::ElementResource(var) => {
                let var = ElementResourceVariable::from(var);
                let expr: ElementUnion = expr.extract()?;
                let expr = ElementExpression::from(expr);
                Self::set_effect(var.id(), expr, &mut self.0.effect.element_resource_effects);
            }
            VarUnion::Set(var) => {
                let var = SetVariable::from(var);
                let expr: SetUnion = expr.extract()?;
                let expr = SetExpression::from(expr);
                Self::set_effect(var.id(), expr, &mut self.0.effect.set_effects);
            }
            VarUnion::Int(var) => {
                let var = IntegerVariable::from(var);
                let expr: IntUnion = expr.extract()?;
                let expr = IntegerExpression::from(expr);
                Self::set_effect(var.id(), expr, &mut self.0.effect.integer_effects);
            }
            VarUnion::IntResource(var) => {
                let var = IntegerResourceVariable::from(var);
                let expr: IntUnion = expr.extract()?;
                let expr = IntegerExpression::from(expr);
                Self::set_effect(var.id(), expr, &mut self.0.effect.integer_resource_effects);
            }
            VarUnion::Float(var) => {
                let var = ContinuousVariable::from(var);
                let expr: FloatUnion = expr.extract()?;
                let expr = ContinuousExpression::from(expr);
                Self::set_effect(var.id(), expr, &mut self.0.effect.continuous_effects);
            }
            VarUnion::FloatResource(var) => {
                let var = ContinuousResourceVariable::from(var);
                let expr: FloatUnion = expr.extract()?;
                let expr = ContinuousExpression::from(expr);
                Self::set_effect(
                    var.id(),
                    expr,
                    &mut self.0.effect.continuous_resource_effects,
                );
            }
        };
        Ok(())
    }

    /// add_effect(var, expr)
    ///
    /// Adds an effect to the transition.
    ///
    /// Parameters
    /// ----------
    /// var: ElementVar, ElementResourceVar, SetVar, IntVar, IntResourceVar, FloatVar, or FloatResourceVar
    ///     Variable to update.
    /// expr: ElementExpr, ElementVar, ElementResourceVar, SetExpr, SetVar, SetConst, IntExpr, IntVar, IntResourceVar, FloatExpr, FloatVar, FloatResourceVar, int, or float
    ///     Expression to update the variable.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If the types of `var` and `expr` mismatch.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> var = model.add_int_var(target=4)
    /// >>> t = dp.Transition(name="t")
    /// >>> t.add_effect(var, var + 1)
    /// >>> t[var].eval(model.target_state, model)
    /// 5
    #[pyo3(signature = (var, expr))]
    fn add_effect(&mut self, var: VarUnion, expr: &PyAny) -> PyResult<()> {
        let result = match var {
            VarUnion::Element(var) => {
                let expr: ElementUnion = expr.extract()?;
                self.0
                    .add_effect(ElementVariable::from(var), ElementExpression::from(expr))
            }
            VarUnion::ElementResource(var) => {
                let expr: ElementUnion = expr.extract()?;
                self.0.add_effect(
                    ElementResourceVariable::from(var),
                    ElementExpression::from(expr),
                )
            }
            VarUnion::Set(var) => {
                let expr: SetUnion = expr.extract()?;
                self.0
                    .add_effect(SetVariable::from(var), SetExpression::from(expr))
            }
            VarUnion::Int(var) => {
                let expr: IntUnion = expr.extract()?;
                self.0
                    .add_effect(IntegerVariable::from(var), IntegerExpression::from(expr))
            }
            VarUnion::IntResource(var) => {
                let expr: IntUnion = expr.extract()?;
                self.0.add_effect(
                    IntegerResourceVariable::from(var),
                    IntegerExpression::from(expr),
                )
            }
            VarUnion::Float(var) => {
                let expr: FloatUnion = expr.extract()?;
                self.0.add_effect(
                    ContinuousVariable::from(var),
                    ContinuousExpression::from(expr),
                )
            }
            VarUnion::FloatResource(var) => {
                let expr: FloatUnion = expr.extract()?;
                self.0.add_effect(
                    ContinuousResourceVariable::from(var),
                    ContinuousExpression::from(expr),
                )
            }
        };
        match result {
            Ok(()) => Ok(()),
            Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
        }
    }

    /// is_applicable(state, model)
    ///
    /// Checks if the transition is applicable in the given state.
    ///
    /// Parameters
    /// ----------
    /// state: State
    ///     State to check.
    /// model: Model
    ///     DyPDL model.
    ///
    /// Returns
    /// -------
    /// bool
    ///     True if the transition is applicable in the given state.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> var = model.add_int_var(target=4)
    /// >>> t = dp.Transition(name="t", preconditions=[var >= 0])
    /// >>> t.is_applicable(model.target_state, model)
    /// True
    #[pyo3(signature = (state, model))]
    fn is_applicable(&self, state: &StatePy, model: &ModelPy) -> bool {
        self.0
            .is_applicable(state.inner_as_ref(), &model.inner_as_ref().table_registry)
    }

    /// apply(state, model)
    ///
    /// Applies the transition to the given state.
    ///
    /// Parameters
    /// ----------
    /// state: State
    ///     State to apply the transition to.
    /// model: Model
    ///     DyPDL model.
    ///
    /// Returns
    /// -------
    /// State
    ///    State after applying the transition.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> var = model.add_int_var(target=4)
    /// >>> t = dp.Transition(name="t", effects=[(var, var + 1)])
    /// >>> state = t.apply(model.target_state, model)
    /// >>> state[var]
    /// 5
    #[pyo3(signature = (state, model))]
    fn apply(&self, state: &mut StatePy, model: &ModelPy) -> StatePy {
        self.0
            .apply(state.inner_as_ref(), &model.inner_as_ref().table_registry)
    }

    /// eval_cost(cost, state, model)
    ///
    /// Evaluates the cost of the transition in the given state.
    ///
    /// Parameters
    /// ----------
    /// cost: int or float
    ///     Cost of the next state.
    /// state: State
    ///     Current state.
    /// model: Model
    ///     DyPDL model.
    ///
    /// Returns
    /// -------
    /// int or float
    ///     Cost of the transition.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If the type of `cost` mismatches the cost type of `model`.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> var = model.add_int_var(target=4)
    /// >>> t = dp.Transition(name="t", cost=dp.IntExpr.state_cost() + 1)
    /// >>> t.eval_cost(1, model.target_state, model)
    /// 2
    #[pyo3(signature = (cost, state, model))]
    fn eval_cost(&self, cost: &PyAny, state: &StatePy, model: &ModelPy) -> PyResult<IntOrFloat> {
        if model.float_cost() {
            let cost = cost.extract()?;
            Ok(IntOrFloat::Float(self.0.eval_cost(
                cost,
                state.inner_as_ref(),
                &model.inner_as_ref().table_registry,
            )))
        } else {
            let cost = cost.extract()?;
            Ok(IntOrFloat::Int(self.0.eval_cost(
                cost,
                state.inner_as_ref(),
                &model.inner_as_ref().table_registry,
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dypdl::GroundedCondition;

    #[test]
    fn cost_expression_from_cost_union_int() {
        let cost = CostUnion::Int(IntUnion::Const(0));
        assert_eq!(
            CostExpression::from(cost),
            CostExpression::Integer(IntegerExpression::Constant(0))
        );
    }

    #[test]
    fn cost_expression_from_cost_union_float() {
        let cost = CostUnion::Float(FloatUnion::Const(0.0));
        assert_eq!(
            CostExpression::from(cost),
            CostExpression::Continuous(ContinuousExpression::Constant(0.0))
        );
    }

    #[test]
    fn transition_from() {
        let transition = TransitionPy(Transition::default());
        assert_eq!(Transition::from(transition), Transition::default());
    }

    #[test]
    fn new() {
        let transition = Transition::default();
        assert_eq!(
            TransitionPy::from(transition),
            TransitionPy(Transition::default())
        );
    }

    #[test]
    fn new_py_with_none_ok() {
        let result = TransitionPy::new_py("t", None, None, None);
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            TransitionPy(Transition {
                name: String::from("t"),
                preconditions: vec![],
                effect: dypdl::Effect::default(),
                ..Default::default()
            })
        );
    }

    #[test]
    fn new_py_ok() {
        pyo3::prepare_freethreaded_python();

        let mut model = Model::default();
        let v1 = model.add_integer_variable("v1", 0);
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let v2 = model.add_continuous_variable("v2", 0.0);
        assert!(v2.is_ok());
        let v2 = v2.unwrap();

        let preconditions = vec![
            ConditionPy::from(Condition::Constant(true)),
            ConditionPy::from(Condition::Constant(false)),
        ];
        let result = Python::with_gil(|py| {
            let v1 = VarUnion::Int(IntVarPy::from(v1));
            let v2 = VarUnion::Float(FloatVarPy::from(v2));
            let expr1 = IntExprPy::from(IntegerExpression::Constant(1)).into_py(py);
            let expr2 = FloatExprPy::from(ContinuousExpression::Constant(2.0)).into_py(py);
            let effects = vec![(v1, expr1.as_ref(py)), (v2, expr2.as_ref(py))];
            TransitionPy::new_py("t", None, Some(preconditions), Some(effects))
        });
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            TransitionPy(Transition {
                name: String::from("t"),
                preconditions: vec![
                    GroundedCondition {
                        condition: Condition::Constant(true),
                        ..Default::default()
                    },
                    GroundedCondition {
                        condition: Condition::Constant(false),
                        ..Default::default()
                    },
                ],
                effect: dypdl::Effect {
                    integer_effects: vec![(v1.id(), IntegerExpression::Constant(1))],
                    continuous_effects: vec![(v2.id(), ContinuousExpression::Constant(2.0))],
                    ..Default::default()
                },
                ..Default::default()
            })
        );
    }

    #[test]
    fn new_py_with_cost_ok() {
        pyo3::prepare_freethreaded_python();

        let mut model = Model::default();
        let v1 = model.add_integer_variable("v1", 0);
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let v2 = model.add_continuous_variable("v2", 0.0);
        assert!(v2.is_ok());
        let v2 = v2.unwrap();

        let cost = CostUnion::Int(IntUnion::Const(0));
        let preconditions = vec![
            ConditionPy::from(Condition::Constant(true)),
            ConditionPy::from(Condition::Constant(false)),
        ];
        let result = Python::with_gil(|py| {
            let v1 = VarUnion::Int(IntVarPy::from(v1));
            let expr1 = IntExprPy::from(IntegerExpression::Constant(1)).into_py(py);
            let v2 = VarUnion::Float(FloatVarPy::from(v2));
            let expr2 = FloatExprPy::from(ContinuousExpression::Constant(2.0)).into_py(py);
            let effects = vec![(v1, expr1.as_ref(py)), (v2, expr2.as_ref(py))];
            TransitionPy::new_py("t", Some(cost), Some(preconditions), Some(effects))
        });
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            TransitionPy(Transition {
                name: String::from("t"),
                cost: CostExpression::Integer(IntegerExpression::Constant(0)),
                preconditions: vec![
                    GroundedCondition {
                        condition: Condition::Constant(true),
                        ..Default::default()
                    },
                    GroundedCondition {
                        condition: Condition::Constant(false),
                        ..Default::default()
                    },
                ],
                effect: dypdl::Effect {
                    integer_effects: vec![(v1.id(), IntegerExpression::Constant(1))],
                    continuous_effects: vec![(v2.id(), ContinuousExpression::Constant(2.0))],
                    ..Default::default()
                },
                ..Default::default()
            })
        );
    }

    #[test]
    fn new_err() {
        let mut model = Model::default();
        let v1 = model.add_integer_variable("v1", 0);
        assert!(v1.is_ok());
        let v1 = v1.unwrap();

        let preconditions = vec![
            ConditionPy::from(Condition::Constant(true)),
            ConditionPy::from(Condition::Constant(false)),
        ];
        let result = Python::with_gil(|py| {
            let v1 = VarUnion::Int(IntVarPy::from(v1));
            let expr1 = ElementExprPy::from(ElementExpression::Constant(1)).into_py(py);
            let effects = vec![(v1, expr1.as_ref(py))];
            TransitionPy::new_py("t", None, Some(preconditions), Some(effects))
        });
        assert!(result.is_err());
    }

    #[test]
    fn name() {
        let transition = TransitionPy(Transition {
            name: String::from("t"),
            ..Default::default()
        });
        assert_eq!(transition.name(), String::from("t"));
    }

    #[test]
    fn set_cost_int() {
        let mut transition = TransitionPy(Transition::default());
        transition.set_cost(CostUnion::Int(IntUnion::Const(0)));
        assert_eq!(
            transition,
            TransitionPy(Transition {
                cost: CostExpression::Integer(IntegerExpression::Constant(0)),
                ..Default::default()
            })
        );
    }

    #[test]
    fn set_cost_float() {
        let mut transition = TransitionPy(Transition::default());
        transition.set_cost(CostUnion::Float(FloatUnion::Const(0.0)));
        assert_eq!(
            transition,
            TransitionPy(Transition {
                cost: CostExpression::Continuous(ContinuousExpression::Constant(0.0)),
                ..Default::default()
            })
        );
    }

    #[test]
    fn add_precondition() {
        let mut transition = TransitionPy(Transition::default());
        transition.add_precondition(ConditionPy::from(Condition::Constant(true)));
        assert_eq!(
            transition,
            TransitionPy(Transition {
                preconditions: vec![GroundedCondition {
                    condition: Condition::Constant(true),
                    ..Default::default()
                }],
                ..Default::default()
            })
        );
    }

    #[test]
    fn add_element_effect_ok() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_variable("v", ob, 0);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut transition = TransitionPy(Transition::default());
        let result = Python::with_gil(|py| {
            let v = VarUnion::Element(ElementVarPy::from(v));
            let expr = ElementExprPy::from(ElementExpression::Constant(0)).into_py(py);
            transition.add_effect(v, expr.as_ref(py))
        });
        assert!(result.is_ok());
        assert_eq!(
            transition,
            TransitionPy(Transition {
                effect: dypdl::Effect {
                    element_effects: vec![(v.id(), ElementExpression::Constant(0))],
                    ..Default::default()
                },
                ..Default::default()
            })
        );
    }

    #[test]
    fn add_element_effect_type_err() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_variable("v", ob, 0);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut transition = TransitionPy(Transition::default());
        let result = Python::with_gil(|py| {
            let v = VarUnion::Element(ElementVarPy::from(v));
            let expr = IntExprPy::from(IntegerExpression::Constant(0)).into_py(py);
            transition.add_effect(v, expr.as_ref(py))
        });
        assert!(result.is_err());
    }

    #[test]
    fn add_element_effect_duplicate_err() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_variable("v", ob, 0);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut transition = TransitionPy(Transition {
            effect: dypdl::Effect {
                element_effects: vec![(v.id(), ElementExpression::Constant(0))],
                ..Default::default()
            },
            ..Default::default()
        });
        let result = Python::with_gil(|py| {
            let v = VarUnion::Element(ElementVarPy::from(v));
            let expr = ElementExprPy::from(ElementExpression::Constant(0)).into_py(py);
            transition.add_effect(v, expr.as_ref(py))
        });
        assert!(result.is_err());
    }

    #[test]
    fn add_element_resource_effect_ok() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_resource_variable("v", ob, false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut transition = TransitionPy(Transition::default());
        let result = Python::with_gil(|py| {
            let v = VarUnion::ElementResource(ElementResourceVarPy::from(v));
            let expr = ElementExprPy::from(ElementExpression::Constant(0)).into_py(py);
            transition.add_effect(v, expr.as_ref(py))
        });
        assert!(result.is_ok());
        assert_eq!(
            transition,
            TransitionPy(Transition {
                effect: dypdl::Effect {
                    element_resource_effects: vec![(v.id(), ElementExpression::Constant(0))],
                    ..Default::default()
                },
                ..Default::default()
            })
        );
    }

    #[test]
    fn add_element_resource_effect_type_err() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_resource_variable("v", ob, false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut transition = TransitionPy(Transition::default());
        let result = Python::with_gil(|py| {
            let v = VarUnion::ElementResource(ElementResourceVarPy::from(v));
            let expr = IntExprPy::from(IntegerExpression::Constant(0)).into_py(py);
            transition.add_effect(v, expr.as_ref(py))
        });
        assert!(result.is_err());
    }

    #[test]
    fn add_element_resource_effect_duplicate_err() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_resource_variable("v", ob, false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut transition = TransitionPy(Transition {
            effect: dypdl::Effect {
                element_resource_effects: vec![(v.id(), ElementExpression::Constant(0))],
                ..Default::default()
            },
            ..Default::default()
        });
        let result = Python::with_gil(|py| {
            let v = VarUnion::ElementResource(ElementResourceVarPy::from(v));
            let expr = ElementExprPy::from(ElementExpression::Constant(0)).into_py(py);
            transition.add_effect(v, expr.as_ref(py))
        });
        assert!(result.is_err());
    }

    #[test]
    fn add_set_effect_ok() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_set_variable("v", ob, Set::with_capacity(10));
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut transition = TransitionPy(Transition::default());
        let result = Python::with_gil(|py| {
            let v = VarUnion::Set(SetVarPy::from(v));
            let expr = SetConstPy::from(Set::with_capacity(10)).into_py(py);
            transition.add_effect(v, expr.as_ref(py))
        });
        assert!(result.is_ok());
        assert_eq!(
            transition,
            TransitionPy(Transition {
                effect: dypdl::Effect {
                    set_effects: vec![(
                        v.id(),
                        SetExpression::Reference(ReferenceExpression::Constant(
                            Set::with_capacity(10)
                        ))
                    )],
                    ..Default::default()
                },
                ..Default::default()
            })
        );
    }

    #[test]
    fn add_set_effect_type_err() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_set_variable("v", ob, Set::with_capacity(10));
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut transition = TransitionPy(Transition::default());
        let result = Python::with_gil(|py| {
            let v = VarUnion::Set(SetVarPy::from(v));
            let expr = IntExprPy::from(IntegerExpression::Constant(0)).into_py(py);
            transition.add_effect(v, expr.as_ref(py))
        });
        assert!(result.is_err());
    }

    #[test]
    fn add_set_effect_duplicate_err() {
        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_set_variable("v", ob, Set::with_capacity(10));
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut transition = TransitionPy(Transition {
            effect: dypdl::Effect {
                set_effects: vec![(
                    v.id(),
                    SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(10))),
                )],
                ..Default::default()
            },
            ..Default::default()
        });
        let result = Python::with_gil(|py| {
            let v = VarUnion::Set(SetVarPy::from(v));
            let expr = SetConstPy::from(Set::with_capacity(10)).into_py(py);
            transition.add_effect(v, expr.as_ref(py))
        });
        assert!(result.is_err());
    }

    #[test]
    fn add_int_effect_ok() {
        let mut model = Model::default();
        let v = model.add_integer_variable("v", 0);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut transition = TransitionPy(Transition::default());
        let result = Python::with_gil(|py| {
            let v = VarUnion::Int(IntVarPy::from(v));
            let expr = IntExprPy::from(IntegerExpression::Constant(0)).into_py(py);
            transition.add_effect(v, expr.as_ref(py))
        });
        assert!(result.is_ok());
        assert_eq!(
            transition,
            TransitionPy(Transition {
                effect: dypdl::Effect {
                    integer_effects: vec![(v.id(), IntegerExpression::Constant(0))],
                    ..Default::default()
                },
                ..Default::default()
            })
        );
    }

    #[test]
    fn add_int_effect_type_err() {
        let mut model = Model::default();
        let v = model.add_integer_variable("v", 0);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut transition = TransitionPy(Transition::default());
        let result = Python::with_gil(|py| {
            let v = VarUnion::Int(IntVarPy::from(v));
            let expr = ElementExprPy::from(ElementExpression::Constant(0)).into_py(py);
            transition.add_effect(v, expr.as_ref(py))
        });
        assert!(result.is_err());
    }

    #[test]
    fn add_int_effect_duplicate_err() {
        let mut model = Model::default();
        let v = model.add_integer_variable("v", 0);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut transition = TransitionPy(Transition {
            effect: dypdl::Effect {
                integer_effects: vec![(v.id(), IntegerExpression::Constant(0))],
                ..Default::default()
            },
            ..Default::default()
        });
        let result = Python::with_gil(|py| {
            let v = VarUnion::Int(IntVarPy::from(v));
            let expr = IntExprPy::from(IntegerExpression::Constant(0)).into_py(py);
            transition.add_effect(v, expr.as_ref(py))
        });
        assert!(result.is_err());
    }

    #[test]
    fn add_int_resource_effect_ok() {
        let mut model = Model::default();
        let v = model.add_integer_resource_variable("v", false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut transition = TransitionPy(Transition::default());
        let result = Python::with_gil(|py| {
            let v = VarUnion::IntResource(IntResourceVarPy::from(v));
            let expr = IntExprPy::from(IntegerExpression::Constant(0)).into_py(py);
            transition.add_effect(v, expr.as_ref(py))
        });
        assert!(result.is_ok());
        assert_eq!(
            transition,
            TransitionPy(Transition {
                effect: dypdl::Effect {
                    integer_resource_effects: vec![(v.id(), IntegerExpression::Constant(0))],
                    ..Default::default()
                },
                ..Default::default()
            })
        );
    }

    #[test]
    fn add_int_resource_effect_type_err() {
        let mut model = Model::default();
        let v = model.add_integer_resource_variable("v", false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut transition = TransitionPy(Transition::default());
        let result = Python::with_gil(|py| {
            let v = VarUnion::IntResource(IntResourceVarPy::from(v));
            let expr = ElementExprPy::from(ElementExpression::Constant(0)).into_py(py);
            transition.add_effect(v, expr.as_ref(py))
        });
        assert!(result.is_err());
    }

    #[test]
    fn add_int_resource_effect_duplicate_err() {
        let mut model = Model::default();
        let v = model.add_integer_resource_variable("v", false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut transition = TransitionPy(Transition {
            effect: dypdl::Effect {
                integer_resource_effects: vec![(v.id(), IntegerExpression::Constant(0))],
                ..Default::default()
            },
            ..Default::default()
        });
        let result = Python::with_gil(|py| {
            let v = VarUnion::IntResource(IntResourceVarPy::from(v));
            let expr = IntExprPy::from(IntegerExpression::Constant(0)).into_py(py);
            transition.add_effect(v, expr.as_ref(py))
        });
        assert!(result.is_err());
    }

    #[test]
    fn add_float_effect_ok() {
        let mut model = Model::default();
        let v = model.add_continuous_variable("v", 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut transition = TransitionPy(Transition::default());
        let result = Python::with_gil(|py| {
            let v = VarUnion::Float(FloatVarPy::from(v));
            let expr = FloatExprPy::from(ContinuousExpression::Constant(0.0)).into_py(py);
            transition.add_effect(v, expr.as_ref(py))
        });
        assert!(result.is_ok());
        assert_eq!(
            transition,
            TransitionPy(Transition {
                effect: dypdl::Effect {
                    continuous_effects: vec![(v.id(), ContinuousExpression::Constant(0.0))],
                    ..Default::default()
                },
                ..Default::default()
            })
        );
    }

    #[test]
    fn add_float_effect_type_err() {
        let mut model = Model::default();
        let v = model.add_continuous_variable("v", 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut transition = TransitionPy(Transition::default());
        let result = Python::with_gil(|py| {
            let v = VarUnion::Float(FloatVarPy::from(v));
            let expr = ElementExprPy::from(ElementExpression::Constant(0)).into_py(py);
            transition.add_effect(v, expr.as_ref(py))
        });
        assert!(result.is_err());
    }

    #[test]
    fn add_float_effect_duplicate_err() {
        let mut model = Model::default();
        let v = model.add_continuous_variable("v", 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut transition = TransitionPy(Transition {
            effect: dypdl::Effect {
                continuous_effects: vec![(v.id(), ContinuousExpression::Constant(0.0))],
                ..Default::default()
            },
            ..Default::default()
        });
        let result = Python::with_gil(|py| {
            let v = VarUnion::Float(FloatVarPy::from(v));
            let expr = FloatExprPy::from(ContinuousExpression::Constant(0.0)).into_py(py);
            transition.add_effect(v, expr.as_ref(py))
        });
        assert!(result.is_err());
    }

    #[test]
    fn add_float_resource_effect_ok() {
        let mut model = Model::default();
        let v = model.add_continuous_resource_variable("v", false, 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut transition = TransitionPy(Transition::default());
        let result = Python::with_gil(|py| {
            let v = VarUnion::FloatResource(FloatResourceVarPy::from(v));
            let expr = FloatExprPy::from(ContinuousExpression::Constant(0.0)).into_py(py);
            transition.add_effect(v, expr.as_ref(py))
        });
        assert!(result.is_ok());
        assert_eq!(
            transition,
            TransitionPy(Transition {
                effect: dypdl::Effect {
                    continuous_resource_effects: vec![(
                        v.id(),
                        ContinuousExpression::Constant(0.0)
                    )],
                    ..Default::default()
                },
                ..Default::default()
            })
        );
    }

    #[test]
    fn add_float_resource_effect_type_err() {
        let mut model = Model::default();
        let v = model.add_continuous_resource_variable("v", false, 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut transition = TransitionPy(Transition::default());
        let result = Python::with_gil(|py| {
            let v = VarUnion::FloatResource(FloatResourceVarPy::from(v));
            let expr = ElementExprPy::from(ElementExpression::Constant(0)).into_py(py);
            transition.add_effect(v, expr.as_ref(py))
        });
        assert!(result.is_err());
    }

    #[test]
    fn add_float_resource_effect_duplicate_err() {
        let mut model = Model::default();
        let v = model.add_continuous_resource_variable("v", false, 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut transition = TransitionPy(Transition {
            effect: dypdl::Effect {
                continuous_resource_effects: vec![(v.id(), ContinuousExpression::Constant(0.0))],
                ..Default::default()
            },
            ..Default::default()
        });
        let result = Python::with_gil(|py| {
            let v = VarUnion::FloatResource(FloatResourceVarPy::from(v));
            let expr = FloatExprPy::from(ContinuousExpression::Constant(0.0)).into_py(py);
            transition.add_effect(v, expr.as_ref(py))
        });
        assert!(result.is_err());
    }
}
