use dypdl::prelude::*;

/// Parent and child caches for state functions.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ParentAndChildStateFunctionCache {
    /// Parent.
    pub parent: StateFunctionCache,
    /// Child.
    pub child: StateFunctionCache,
}

impl ParentAndChildStateFunctionCache {
    /// Create a new parent and child cache.
    pub fn new(state_functions: &StateFunctions) -> Self {
        Self {
            parent: StateFunctionCache::new(state_functions),
            child: StateFunctionCache::new(state_functions),
        }
    }
}
