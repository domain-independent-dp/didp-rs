use crate::{StateFunctionCache, StateFunctions};

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expression::{Condition, IntegerExpression};

    #[test]
    fn new_creates_parent_and_child_caches() {
        let mut state_functions = StateFunctions::default();
        state_functions
            .add_integer_function("integer", IntegerExpression::Constant(1))
            .unwrap();
        state_functions
            .add_boolean_function("boolean", Condition::Constant(true))
            .unwrap();

        let cache = ParentAndChildStateFunctionCache::new(&state_functions);

        assert_eq!(cache.parent, StateFunctionCache::new(&state_functions));
        assert_eq!(cache.child, StateFunctionCache::new(&state_functions));
    }
}
