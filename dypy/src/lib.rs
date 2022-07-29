use pyo3::prelude::*;

mod expression;
mod heuristic_search_solver;
mod model;
mod table;
mod transition;

/// DyPy -- DyPDL interface for Python
#[pymodule]
fn dypy(_: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<model::ObjectTypePy>()?;
    m.add_class::<model::ModelPy>()?;
    m.add_class::<transition::TransitionPy>()?;
    m.add_class::<table::ElementTable1DPy>()?;
    m.add_class::<table::ElementTable2DPy>()?;
    m.add_class::<table::ElementTable3DPy>()?;
    m.add_class::<table::ElementTablePy>()?;
    m.add_class::<table::SetTable1DPy>()?;
    m.add_class::<table::SetTable2DPy>()?;
    m.add_class::<table::SetTable3DPy>()?;
    m.add_class::<table::SetTablePy>()?;
    m.add_class::<table::BoolTable1DPy>()?;
    m.add_class::<table::BoolTable2DPy>()?;
    m.add_class::<table::BoolTable3DPy>()?;
    m.add_class::<table::BoolTablePy>()?;
    m.add_class::<table::IntTable1DPy>()?;
    m.add_class::<table::IntTable2DPy>()?;
    m.add_class::<table::IntTable3DPy>()?;
    m.add_class::<table::IntTablePy>()?;
    m.add_class::<table::FloatTable1DPy>()?;
    m.add_class::<table::FloatTable2DPy>()?;
    m.add_class::<table::FloatTable3DPy>()?;
    m.add_class::<table::FloatTablePy>()?;
    m.add_class::<expression::ElementExprPy>()?;
    m.add_class::<expression::ElementVarPy>()?;
    m.add_class::<expression::ElementResourceVarPy>()?;
    m.add_class::<expression::SetExprPy>()?;
    m.add_class::<expression::SetVarPy>()?;
    m.add_class::<expression::SetConstPy>()?;
    m.add_class::<expression::IntExprPy>()?;
    m.add_class::<expression::IntVarPy>()?;
    m.add_class::<expression::IntResourceVarPy>()?;
    m.add_class::<expression::FloatExprPy>()?;
    m.add_class::<expression::FloatVarPy>()?;
    m.add_class::<expression::FloatResourceVarPy>()?;
    m.add_class::<expression::ConditionPy>()?;
    m.add_function(wrap_pyfunction!(expression::sqrt, m)?)?;
    m.add_function(wrap_pyfunction!(expression::log, m)?)?;
    m.add_function(wrap_pyfunction!(expression::float, m)?)?;
    m.add_function(wrap_pyfunction!(expression::max, m)?)?;
    m.add_function(wrap_pyfunction!(expression::min, m)?)?;
    m.add_class::<heuristic_search_solver::SolutionPy>()?;
    m.add_class::<heuristic_search_solver::FOperator>()?;
    m.add_class::<heuristic_search_solver::CAASDyPy>()?;
    m.add_class::<heuristic_search_solver::DijkstraPy>()?;
    m.add_class::<heuristic_search_solver::LazyDijkstraPy>()?;
    m.add_class::<heuristic_search_solver::DualBoundDFBBPy>()?;
    m.add_class::<heuristic_search_solver::IBDFSPy>()?;
    m.add_class::<heuristic_search_solver::ForwardRecursionPy>()?;
    m.add_class::<heuristic_search_solver::ExpressionBeamSearchPy>()?;
    m.add_class::<heuristic_search_solver::ExpressionEpsilonBeamSearchPy>()?;
    Ok(())
}
