use super::expression::*;
use dypdl::prelude::*;
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

/// A class representing a transition.
///
/// Parameters
/// ----------
/// name: str
///     Name of the transition.
/// cost: IntExpr, IntVar, IntResourceVar, FloatExpr, FloatVar, FloatResourceVar, int, float, or None, default: None
///     Cost expression.
///     `IntExpr.state_cost()` or `FloatExpr.state_cost()` can be used to reperesent the cost of the transitioned state.
///     If `None`, `IntExpr.state_cost()` is used.
/// preconditions: list of Condition or None, default: None
///     Preconditions, which must be satisfied by a state to be applicable.
/// effects: list of tuple of a variable and an expression or None, default: None
///     Effects, a sequence of tuple of a variable and an expression.
///     Instead of an exprssion, a variable or an immidiate value can be used.
///
/// Raises
/// ------
/// RuntimeError
///     If multiple effects are defined for the same variable.
/// TypeError
///     If the types of a variable and an expression in `effects` mismatch.
#[pyclass(name = "Transition")]
#[pyo3(text_signature = "(name, cost=None, preconditions=None, effects=None)")]
#[derive(Debug, PartialEq, Clone, Default)]
pub struct TransitionPy(Transition);

impl From<TransitionPy> for Transition {
    fn from(transition: TransitionPy) -> Self {
        transition.0
    }
}

impl TransitionPy {
    pub fn new(transition: Transition) -> TransitionPy {
        TransitionPy(transition)
    }
}

#[pymethods]
impl TransitionPy {
    #[new]
    #[args(cost = "None", preconditions = "vec![]", effects = "vec![]")]
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

    /// IntExpr or FloatExpr : Cost expression.
    /// `IntExpr.state_cost()` or `FloatExpr.state_cost()` can be used to reperesent the cost of the transitioned state.
    #[setter]
    fn set_cost(&mut self, cost: CostUnion) {
        match cost {
            CostUnion::Int(cost) => self.0.set_cost(IntegerExpression::from(cost)),
            CostUnion::Float(cost) => self.0.set_cost(ContinuousExpression::from(cost)),
        }
    }

    /// add_precondition(condition)
    ///
    /// Adds a precondition to the transition.
    ///
    /// Parameters
    /// ----------
    /// condition: Condition
    ///     Precondition.
    #[pyo3(text_signature = "(condition)")]
    fn add_precondition(&mut self, condition: ConditionPy) {
        self.0.add_precondition(condition.into())
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
    #[pyo3(text_signature = "(var, expr)")]
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use dypdl::expression::*;
    use dypdl::GroundedCondition;

    #[test]
    fn cost_exprssion_from_cost_union_int() {
        let cost = CostUnion::Int(IntUnion::Const(0));
        assert_eq!(
            CostExpression::from(cost),
            CostExpression::Integer(IntegerExpression::Constant(0))
        );
    }

    #[test]
    fn cost_exprssion_from_cost_union_float() {
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
            TransitionPy::new(transition),
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
            ConditionPy::new(Condition::Constant(true)),
            ConditionPy::new(Condition::Constant(false)),
        ];
        let result = Python::with_gil(|py| {
            let v1 = VarUnion::Int(IntVarPy::new(v1));
            let v2 = VarUnion::Float(FloatVarPy::new(v2));
            let expr1 = IntExprPy::new(IntegerExpression::Constant(1)).into_py(py);
            let expr2 = FloatExprPy::new(ContinuousExpression::Constant(2.0)).into_py(py);
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
            ConditionPy::new(Condition::Constant(true)),
            ConditionPy::new(Condition::Constant(false)),
        ];
        let result = Python::with_gil(|py| {
            let v1 = VarUnion::Int(IntVarPy::new(v1));
            let expr1 = IntExprPy::new(IntegerExpression::Constant(1)).into_py(py);
            let v2 = VarUnion::Float(FloatVarPy::new(v2));
            let expr2 = FloatExprPy::new(ContinuousExpression::Constant(2.0)).into_py(py);
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
            ConditionPy::new(Condition::Constant(true)),
            ConditionPy::new(Condition::Constant(false)),
        ];
        let result = Python::with_gil(|py| {
            let v1 = VarUnion::Int(IntVarPy::new(v1));
            let expr1 = ElementExprPy::new(ElementExpression::Constant(1)).into_py(py);
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
        transition.add_precondition(ConditionPy::new(Condition::Constant(true)));
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
            let v = VarUnion::Element(ElementVarPy::new(v));
            let expr = ElementExprPy::new(ElementExpression::Constant(0)).into_py(py);
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
            let v = VarUnion::Element(ElementVarPy::new(v));
            let expr = IntExprPy::new(IntegerExpression::Constant(0)).into_py(py);
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
            let v = VarUnion::Element(ElementVarPy::new(v));
            let expr = ElementExprPy::new(ElementExpression::Constant(0)).into_py(py);
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
            let v = VarUnion::ElementResource(ElementResourceVarPy::new(v));
            let expr = ElementExprPy::new(ElementExpression::Constant(0)).into_py(py);
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
            let v = VarUnion::ElementResource(ElementResourceVarPy::new(v));
            let expr = IntExprPy::new(IntegerExpression::Constant(0)).into_py(py);
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
            let v = VarUnion::ElementResource(ElementResourceVarPy::new(v));
            let expr = ElementExprPy::new(ElementExpression::Constant(0)).into_py(py);
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
            let v = VarUnion::Set(SetVarPy::new(v));
            let expr = SetConstPy::new(Set::with_capacity(10)).into_py(py);
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
            let v = VarUnion::Set(SetVarPy::new(v));
            let expr = IntExprPy::new(IntegerExpression::Constant(0)).into_py(py);
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
            let v = VarUnion::Set(SetVarPy::new(v));
            let expr = SetConstPy::new(Set::with_capacity(10)).into_py(py);
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
            let v = VarUnion::Int(IntVarPy::new(v));
            let expr = IntExprPy::new(IntegerExpression::Constant(0)).into_py(py);
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
            let v = VarUnion::Int(IntVarPy::new(v));
            let expr = ElementExprPy::new(ElementExpression::Constant(0)).into_py(py);
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
            let v = VarUnion::Int(IntVarPy::new(v));
            let expr = IntExprPy::new(IntegerExpression::Constant(0)).into_py(py);
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
            let v = VarUnion::IntResource(IntResourceVarPy::new(v));
            let expr = IntExprPy::new(IntegerExpression::Constant(0)).into_py(py);
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
            let v = VarUnion::IntResource(IntResourceVarPy::new(v));
            let expr = ElementExprPy::new(ElementExpression::Constant(0)).into_py(py);
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
            let v = VarUnion::IntResource(IntResourceVarPy::new(v));
            let expr = IntExprPy::new(IntegerExpression::Constant(0)).into_py(py);
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
            let v = VarUnion::Float(FloatVarPy::new(v));
            let expr = FloatExprPy::new(ContinuousExpression::Constant(0.0)).into_py(py);
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
            let v = VarUnion::Float(FloatVarPy::new(v));
            let expr = ElementExprPy::new(ElementExpression::Constant(0)).into_py(py);
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
            let v = VarUnion::Float(FloatVarPy::new(v));
            let expr = FloatExprPy::new(ContinuousExpression::Constant(0.0)).into_py(py);
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
            let v = VarUnion::FloatResource(FloatResourceVarPy::new(v));
            let expr = FloatExprPy::new(ContinuousExpression::Constant(0.0)).into_py(py);
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
            let v = VarUnion::FloatResource(FloatResourceVarPy::new(v));
            let expr = ElementExprPy::new(ElementExpression::Constant(0)).into_py(py);
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
            let v = VarUnion::FloatResource(FloatResourceVarPy::new(v));
            let expr = FloatExprPy::new(ContinuousExpression::Constant(0.0)).into_py(py);
            transition.add_effect(v, expr.as_ref(py))
        });
        assert!(result.is_err());
    }
}
