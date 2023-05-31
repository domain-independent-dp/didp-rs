use pyo3::prelude::*;

mod heuristic_search_solver;
mod model;

pub use model::ModelPy;

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

/// DIDPPy -- DyPDL interface for Python
#[pymodule]
fn didppy(_: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<model::ObjectTypePy>()?;
    m.add_class::<model::ModelPy>()?;
    m.add_class::<model::TransitionPy>()?;
    m.add_class::<model::StatePy>()?;
    m.add_class::<model::ElementTable1DPy>()?;
    m.add_class::<model::ElementTable2DPy>()?;
    m.add_class::<model::ElementTable3DPy>()?;
    m.add_class::<model::ElementTablePy>()?;
    m.add_class::<model::SetTable1DPy>()?;
    m.add_class::<model::SetTable2DPy>()?;
    m.add_class::<model::SetTable3DPy>()?;
    m.add_class::<model::SetTablePy>()?;
    m.add_class::<model::BoolTable1DPy>()?;
    m.add_class::<model::BoolTable2DPy>()?;
    m.add_class::<model::BoolTable3DPy>()?;
    m.add_class::<model::BoolTablePy>()?;
    m.add_class::<model::IntTable1DPy>()?;
    m.add_class::<model::IntTable2DPy>()?;
    m.add_class::<model::IntTable3DPy>()?;
    m.add_class::<model::IntTablePy>()?;
    m.add_class::<model::FloatTable1DPy>()?;
    m.add_class::<model::FloatTable2DPy>()?;
    m.add_class::<model::FloatTable3DPy>()?;
    m.add_class::<model::FloatTablePy>()?;
    m.add_class::<model::ElementExprPy>()?;
    m.add_class::<model::ElementVarPy>()?;
    m.add_class::<model::ElementResourceVarPy>()?;
    m.add_class::<model::SetExprPy>()?;
    m.add_class::<model::SetVarPy>()?;
    m.add_class::<model::SetConstPy>()?;
    m.add_class::<model::IntExprPy>()?;
    m.add_class::<model::IntVarPy>()?;
    m.add_class::<model::IntResourceVarPy>()?;
    m.add_class::<model::FloatExprPy>()?;
    m.add_class::<model::FloatVarPy>()?;
    m.add_class::<model::FloatResourceVarPy>()?;
    m.add_class::<model::ConditionPy>()?;
    m.add_function(wrap_pyfunction!(model::sqrt, m)?)?;
    m.add_function(wrap_pyfunction!(model::log, m)?)?;
    m.add_function(wrap_pyfunction!(model::float, m)?)?;
    m.add_function(wrap_pyfunction!(model::max, m)?)?;
    m.add_function(wrap_pyfunction!(model::min, m)?)?;
    m.add_class::<heuristic_search_solver::SolutionPy>()?;
    m.add_class::<heuristic_search_solver::FOperator>()?;
    m.add_class::<heuristic_search_solver::CaasdyPy>()?;
    m.add_class::<heuristic_search_solver::CabsPy>()?;
    m.add_class::<heuristic_search_solver::BeamParallelizationMethod>()?;
    m.add_class::<heuristic_search_solver::DfbbPy>()?;
    m.add_class::<heuristic_search_solver::CbfsPy>()?;
    m.add_class::<heuristic_search_solver::ExpressionBeamSearchPy>()?;
    m.add_class::<heuristic_search_solver::AcpsPy>()?;
    m.add_class::<heuristic_search_solver::AppsPy>()?;
    m.add_class::<heuristic_search_solver::DbdfsPy>()?;
    m.add_class::<heuristic_search_solver::ForwardRecursionPy>()?;
    m.add_class::<heuristic_search_solver::BreadthFirstSearchPy>()?;
    m.add_class::<heuristic_search_solver::WeightedAstarPy>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
