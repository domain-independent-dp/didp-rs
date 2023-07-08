use super::f_operator::FOperator;
use super::wrapped_solver::{SolutionPy, WrappedSolver};
use crate::model::ModelPy;
use dypdl::prelude::*;
use dypdl::variable_type::OrderedContinuous;
use dypdl_heuristic_search::{
    create_dual_bound_cabs_and_count_paths, CountParameters, FEvaluatorType, Search,
};
use pyo3::prelude::*;
use std::rc::Rc;

#[pyclass(unsendable, name = "CountPaths")]
#[pyo3(
    text_signature = "(model, depth=8, sample_size=10000, filename='cost_distribution.txt', time_limit=None, f_operator=0, seed=2023, initial_beam_size=None, keep_all_layers=False, primal_bound=None, quiet=False)"
)]
pub struct CountPathsPy(
    WrappedSolver<Box<dyn Search<Integer>>, Box<dyn Search<OrderedContinuous>>>,
);

#[pymethods]
impl CountPathsPy {
    #[new]
    #[pyo3(signature = (
        model,
        depth  = 8,
        sample_size = 10000,
        filename = "cost_distribution.txt",
        time_limit = None,
        f_operator = FOperator::Plus,
        seed = 2023,
        initial_beam_size = None,
        keep_all_layers = false,
        primal_bound = None,
        quiet = false
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: &ModelPy,
        depth: usize,
        sample_size: usize,
        filename: &str,
        time_limit: Option<f64>,
        f_operator: FOperator,
        seed: u64,
        initial_beam_size: Option<usize>,
        keep_all_layers: bool,
        primal_bound: Option<&PyAny>,
        quiet: bool,
    ) -> PyResult<CountPathsPy> {
        let f_evaluator_type = FEvaluatorType::from(f_operator);
        let initial_beam_size = initial_beam_size.unwrap_or(1);

        if model.float_cost() {
            let primal_bound = if let Some(primal_bound) = primal_bound {
                Some(OrderedContinuous::from(
                    primal_bound.extract::<Continuous>()?,
                ))
            } else {
                None
            };
            let parameters = dypdl_heuristic_search::Parameters::<OrderedContinuous> {
                primal_bound,
                time_limit,
                get_all_solutions: false,
                quiet,
            };
            let count_parameters = CountParameters {
                depth,
                sample_size,
                seed,
                filename: String::from(filename),
            };
            let solver = create_dual_bound_cabs_and_count_paths::<OrderedContinuous>(
                Rc::new(model.inner_as_ref().clone()),
                parameters,
                count_parameters,
                f_evaluator_type,
                initial_beam_size,
                keep_all_layers,
            );
            Ok(CountPathsPy(WrappedSolver::Float(solver)))
        } else {
            let primal_bound = if let Some(primal_bound) = primal_bound {
                Some(primal_bound.extract::<Integer>()?)
            } else {
                None
            };
            let parameters = dypdl_heuristic_search::Parameters::<Integer> {
                primal_bound,
                time_limit,
                get_all_solutions: false,
                quiet,
            };
            let count_parameters = CountParameters {
                depth,
                sample_size,
                filename: String::from(filename),
                seed,
            };
            let solver = create_dual_bound_cabs_and_count_paths::<Integer>(
                Rc::new(model.inner_as_ref().clone()),
                parameters,
                count_parameters,
                f_evaluator_type,
                initial_beam_size,
                keep_all_layers,
            );
            Ok(CountPathsPy(WrappedSolver::Int(solver)))
        }
    }

    #[pyo3(signature = ())]
    fn search(&mut self) -> PyResult<SolutionPy> {
        self.0.search()
    }

    #[pyo3(signature = ())]
    fn search_next(&mut self) -> PyResult<(SolutionPy, bool)> {
        self.0.search_next()
    }
}
