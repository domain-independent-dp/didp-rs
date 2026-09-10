use super::super::state_registry::{StateInRegistry, StateInformation};
use super::super::transition_chain::GetTransitions;
use super::super::util;
use super::BfsNode;
use dypdl::variable_type::Numeric;
use dypdl::{Model, TransitionInterface};
use std::cmp::Ordering;
use std::fmt::Display;
use std::marker::PhantomData;
use std::ops::Deref;
use std::rc::Rc;

/// A node wrapping another best-first search node, ordered by comparing resource variables
/// lexicographically before falling back to the wrapped node's own ordering.
///
/// This is used by the labeling algorithm, a resource-constrained shortest path search.
#[derive(Debug, Clone)]
pub struct ResourceLexicographicNode<T, N, M = Rc<Model>> {
    /// Wrapped node.
    pub node: N,
    model: M,
    _phantom: PhantomData<T>,
}

impl<T, N, M> ResourceLexicographicNode<T, N, M> {
    /// Creates a new resource-lexicographic node.
    pub fn new(node: N, model: M) -> Self {
        Self {
            node,
            model,
            _phantom: PhantomData,
        }
    }
}

impl<T, N, M> PartialEq for ResourceLexicographicNode<T, N, M>
where
    N: StateInformation<T> + PartialEq,
    T: Numeric,
{
    fn eq(&self, other: &Self) -> bool {
        self.node == other.node
            && self.node.state().resource_variables == other.node.state().resource_variables
    }
}

impl<T, N, M> Eq for ResourceLexicographicNode<T, N, M>
where
    N: StateInformation<T> + Eq,
    T: Numeric,
{
}

impl<T, N, M> Ord for ResourceLexicographicNode<T, N, M>
where
    N: StateInformation<T> + Ord,
    T: Numeric,
    M: Deref<Target = Model>,
{
    fn cmp(&self, other: &Self) -> Ordering {
        let result = util::lexicographic_resource_variables_cmp(
            &self.model.state_metadata,
            self.node.state(),
            other.node.state(),
        );

        if result != Ordering::Equal {
            return result;
        }

        self.node.cmp(&other.node)
    }
}

impl<T, N, M> PartialOrd for ResourceLexicographicNode<T, N, M>
where
    N: StateInformation<T> + Ord,
    T: Numeric,
    M: Deref<Target = Model>,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T, N, M> StateInformation<T> for ResourceLexicographicNode<T, N, M>
where
    N: StateInformation<T>,
    T: Numeric,
{
    #[inline]
    fn state(&self) -> &StateInRegistry {
        self.node.state()
    }

    #[inline]
    fn state_mut(&mut self) -> &mut StateInRegistry {
        self.node.state_mut()
    }

    #[inline]
    fn cost(&self, model: &Model) -> T {
        self.node.cost(model)
    }

    #[inline]
    fn bound(&self, model: &Model) -> Option<T> {
        self.node.bound(model)
    }

    #[inline]
    fn is_closed(&self) -> bool {
        self.node.is_closed()
    }

    #[inline]
    fn close(&self) {
        self.node.close()
    }
}

impl<T, N, V, M> GetTransitions<V> for ResourceLexicographicNode<T, N, M>
where
    N: GetTransitions<V>,
{
    #[inline]
    fn transitions(&self) -> Vec<V> {
        self.node.transitions()
    }

    #[inline]
    fn last(&self) -> Option<&V> {
        self.node.last()
    }
}

impl<T, N, V, M> BfsNode<T, V> for ResourceLexicographicNode<T, N, M>
where
    N: BfsNode<T, V>,
    T: Numeric + Display,
    V: TransitionInterface + Clone,
    M: Deref<Target = Model>,
{
    #[inline]
    fn ordered_by_bound() -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::super::CostNode;
    use super::super::FNode;
    use super::*;
    use dypdl::ResourceVariables;

    #[test]
    fn test_state_information() {
        let model = Model::default();
        let model = Rc::new(model);
        let node = ResourceLexicographicNode::<_, _>::new(
            CostNode::<_>::new(StateInRegistry::default(), 0, &model, None),
            model.clone(),
        );
        assert_eq!(node.state(), &StateInRegistry::default());
        assert_eq!(node.cost(&model), 0);
        assert_eq!(node.bound(&model), None);
        assert!(!node.is_closed());
        node.close();
        assert!(node.is_closed());
    }

    #[test]
    fn test_get_transitions() {
        let model = Model::default();
        let model = Rc::new(model);
        let node = ResourceLexicographicNode::<i32, CostNode<_>>::new(
            CostNode::<_>::new(StateInRegistry::default(), 0, &model, None),
            model,
        );
        assert_eq!(node.transitions(), vec![]);
        assert_eq!(node.last(), None);
    }

    #[test]
    fn test_ordering() {
        let mut model = Model::default();

        let iv1 = model.add_integer_resource_variable("iv1", false, 0);
        assert!(iv1.is_ok());

        let iv2 = model.add_integer_resource_variable("iv2", true, 0);
        assert!(iv2.is_ok());

        let model = Rc::new(model);

        let state1 = StateInRegistry {
            resource_variables: ResourceVariables {
                integer_variables: vec![1, -1],
                ..Default::default()
            },
            ..Default::default()
        };
        let node1 = ResourceLexicographicNode::<i32, FNode<_>>::new(
            FNode::with_node_and_h_and_f(CostNode::<_>::new(state1, 2, &model, None), 2, 4),
            model.clone(),
        );

        let state2 = StateInRegistry {
            resource_variables: ResourceVariables {
                integer_variables: vec![0, -2],
                ..Default::default()
            },
            ..Default::default()
        };
        let node2 = ResourceLexicographicNode::<i32, FNode<_>>::new(
            FNode::with_node_and_h_and_f(CostNode::<_>::new(state2, 0, &model, None), 0, 0),
            model.clone(),
        );

        let state3 = StateInRegistry {
            resource_variables: ResourceVariables {
                integer_variables: vec![1, -1],
                ..Default::default()
            },
            ..Default::default()
        };
        let node3 = ResourceLexicographicNode::<i32, FNode<_>>::new(
            FNode::with_node_and_h_and_f(CostNode::<_>::new(state3, 3, &model, None), 0, 3),
            model.clone(),
        );

        assert!(node1 == node1);
        assert!(node1 >= node2);
        assert!(node1 >= node3);
    }

    #[test]
    fn test_ordered_by_bound() {
        assert!(!ResourceLexicographicNode::<i32, CostNode<i32>>::ordered_by_bound(),);
    }
}
