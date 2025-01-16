use pyo3::prelude::*;

/// An enum representing how to parallelize beam search.
///
/// :attr:`~BeamParallelizationMethod.Hdbs2`: Use HDBS2.
///
/// :attr:`~BeamParallelizationMethod.Hdbs1`: Use HDBS1.
///
/// :attr:`~BeamParallelizationMethod.Sbs`: Use SBS.
///
/// References
/// ----------
/// Ryo Kuroiwa and J. Christopher Beck. "Parallel Beam Search Algorithms for Domain-Independent Dynamic Programming,"
/// Proceedings of the 38th Annual AAAI Conference on Artificial Intelligence (AAAI), 2024.
#[pyclass(eq, eq_int, name = "BeamParallelizationMethod")]
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum BeamParallelizationMethod {
    /// Use HDBS2.
    Hdbs2 = 0,
    /// Use HDBS1.
    Hdbs1 = 1,
    /// Use SBS.
    Sbs = 2,
}
