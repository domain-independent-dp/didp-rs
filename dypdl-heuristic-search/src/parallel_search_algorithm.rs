//! A module for parallel search algorithms.

pub mod data_structure;
mod shared_memory_beam_search;

pub use data_structure::{
    ConcurrentStateRegistry, DistributedFNode, SendableCostNode, SendableFNode,
};
pub use shared_memory_beam_search::shared_memory_beam_search;
