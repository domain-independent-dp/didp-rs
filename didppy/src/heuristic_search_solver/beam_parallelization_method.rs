use pyo3::prelude::*;

/// An enum representing how to parallelize beam search.
///
/// :attr:`~BeamParallelizationMethod.HDSync` (1): Use HD sync beam search.
///
/// :attr:`~BeamParallelizationMethod.SharedMemory` (2): Use shared memory beam search.
#[pyclass(name = "BeamParallelizationMethod")]
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum BeamParallelizationMethod {
    /// Use HD sync beam search.
    HDSync = 1,
    /// Use shared memory beam search.
    SharedMemory = 2,
}
