use pyo3::prelude::*;

/// An enum representing how to parallelize beam search.
///
/// :attr:`~BeamParallelizationMethod.HD` (0): Use HDBS2.
///
/// :attr:`~BeamParallelizationMethod.HDSync` (1): Use HDBS1.
///
/// :attr:`~BeamParallelizationMethod.SharedMemory` (2): Use SMBS.
#[pyclass(name = "BeamParallelizationMethod")]
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum BeamParallelizationMethod {
    /// Use HDBS2.
    HD = 0,
    /// Use HDBS1.
    HDSync = 1,
    /// Use shared memory beam search.
    SharedMemory = 2,
}
