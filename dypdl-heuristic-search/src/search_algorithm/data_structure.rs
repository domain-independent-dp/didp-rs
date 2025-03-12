//! A module for data structures.

mod beam;
mod hashable_state;
mod parent_and_child_state_function_cache;
mod search_node;
mod state_registry;
mod successor_generator;
mod transition;
mod transition_chain;
mod transition_id_chain;
mod transition_mutex;
mod util;

pub use beam::{Beam, BeamDrain, BeamInsertionStatus};
pub use hashable_state::{
    HashableSignatureVariables, HashableState, StateWithHashableSignatureVariables,
};
pub use parent_and_child_state_function_cache::ParentAndChildStateFunctionCache;
pub use search_node::{
    BfsNode, BfsNodeWithTransitionIds, CostNode, CustomFNode, FNode, FNodeEvaluators, WeightedFNode,
};
pub use state_registry::{
    remove_dominated, InsertionResult, StateInRegistry, StateInformation, StateRegistry,
};
pub use successor_generator::{ApplicableTransitions, SuccessorGenerator};
pub use transition::{TransitionWithCustomCost, TransitionWithId};
pub use transition_chain::{CreateTransitionChain, GetTransitions, RcChain};
pub use transition_id_chain::GetTransitionIds;
pub use transition_mutex::TransitionMutex;
pub use util::exceed_bound;
