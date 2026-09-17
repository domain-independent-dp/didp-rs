use super::{IntOrFloat, ModelPy, TransitionIdPy, TransitionPy};
use didp_yaml::solution::{
    self, CostTolerance, LoadedSolution, SolutionCost, SolutionError, SolutionToDump,
};
use dypdl::variable_type::{Integer, Numeric, OrderedContinuous};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

pyo3::create_exception!(
    didppy,
    ValidationError,
    PyValueError,
    "A solution is infeasible, cannot be evaluated, or has an incorrect declared cost."
);

pub fn load_from_str(
    model: &ModelPy,
    source: &str,
) -> PyResult<(Vec<TransitionPy>, Option<IntOrFloat>)> {
    solution::load_solution_from_str(&model.0, source)
        .map(|solution| {
            let transitions = solution
                .transitions
                .into_iter()
                .map(TransitionPy::from)
                .collect();
            let cost = solution.cost.map(|cost| match cost {
                SolutionCost::Integer(cost) => IntOrFloat::Int(cost),
                SolutionCost::Continuous(cost) => IntOrFloat::Float(cost),
            });
            (transitions, cost)
        })
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

#[derive(FromPyObject)]
pub enum SolutionTransitions {
    #[pyo3(transparent, annotation = "Sequence[TransitionId]")]
    Ids(Vec<TransitionIdPy>),
    #[pyo3(transparent, annotation = "Sequence[Transition]")]
    Transitions(Vec<TransitionPy>),
}

impl SolutionTransitions {
    fn into_transitions(self, model: &ModelPy) -> PyResult<Vec<dypdl::Transition>> {
        match self {
            Self::Ids(ids) => ids
                .into_iter()
                .enumerate()
                .map(|(index, id)| {
                    if id.0.backward {
                        return Err(PyValueError::new_err(format!(
                            "transitions[{index}]: backward IDs cannot be dumped as a forward solution"
                        )));
                    }
                    model.0.get_transition(&id.0).cloned().map_err(|error| {
                        PyValueError::new_err(format!("transitions[{index}]: {error}"))
                    })
                })
                .collect(),
            Self::Transitions(transitions) => Ok(transitions
                .into_iter()
                .map(dypdl::Transition::from)
                .collect()),
        }
    }

    fn validate<T: Numeric + Ord + 'static>(
        self,
        model: &ModelPy,
    ) -> Result<T, dypdl::SolutionValidationError> {
        match self {
            Self::Ids(ids) => {
                let ids = ids.into_iter().map(|id| id.0).collect::<Vec<_>>();
                model.0.validate_solution_with_ids(&ids)
            }
            Self::Transitions(transitions) => {
                let transitions = transitions
                    .into_iter()
                    .map(dypdl::Transition::from)
                    .collect::<Vec<_>>();
                model.0.validate_solution(&transitions)
            }
        }
    }
}

pub fn dump_to_str(
    model: &ModelPy,
    transitions: SolutionTransitions,
    cost: Option<Bound<'_, PyAny>>,
) -> PyResult<String> {
    let cost = cost
        .map(|cost| {
            if model.float_cost() {
                let cost: f64 = cost.extract()?;
                if !cost.is_finite() {
                    return Err(PyValueError::new_err("cost must be finite"));
                }
                Ok(SolutionCost::Continuous(cost))
            } else {
                Ok(SolutionCost::Integer(cost.extract()?))
            }
        })
        .transpose()?;
    let transitions = transitions.into_transitions(model)?;
    SolutionToDump::from(LoadedSolution { cost, transitions })
        .dump_to_str()
        .map_err(|error| PyValueError::new_err(format!("Could not serialize solution: {error}")))
}

pub fn validate(model: &ModelPy, transitions: SolutionTransitions) -> PyResult<IntOrFloat> {
    if model.float_cost() {
        transitions
            .validate::<OrderedContinuous>(model)
            .map(|cost| IntOrFloat::Float(cost.into_inner()))
    } else {
        transitions.validate::<Integer>(model).map(IntOrFloat::Int)
    }
    .map_err(|error| ValidationError::new_err(error.to_string()))
}

pub fn validate_cost(
    model: &ModelPy,
    computed_cost: Bound<'_, PyAny>,
    declared_cost: Bound<'_, PyAny>,
    rel_tol: f64,
    abs_tol: f64,
) -> PyResult<()> {
    let (computed, declared) = if model.float_cost() {
        (
            SolutionCost::Continuous(computed_cost.extract()?),
            SolutionCost::Continuous(declared_cost.extract()?),
        )
    } else {
        (
            SolutionCost::Integer(computed_cost.extract()?),
            SolutionCost::Integer(declared_cost.extract()?),
        )
    };
    solution::validate_cost(computed, declared, CostTolerance { rel_tol, abs_tol }).map_err(
        |error| match error {
            SolutionError::InvalidTolerance => PyValueError::new_err(error.to_string()),
            SolutionError::CostTypeMismatch { .. } => PyTypeError::new_err(error.to_string()),
            _ => ValidationError::new_err(error.to_string()),
        },
    )
}
