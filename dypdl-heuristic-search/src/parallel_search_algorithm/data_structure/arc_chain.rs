use crate::search_algorithm::data_structure::TransitionChain;
use dypdl::{Transition, TransitionInterface};
use std::sync::Arc;

/// Chain of transitions implemented by a linked list of `Arc`.
#[derive(PartialEq, Debug)]
pub struct ArcChain<T = Transition>
where
    T: TransitionInterface + Clone,
    Transition: From<T>,
{
    parent: Option<Arc<Self>>,
    last: Arc<T>,
}

impl<T> TransitionChain<T, Arc<T>, Arc<Self>> for ArcChain<T>
where
    T: TransitionInterface + Clone,
    Transition: From<T>,
{
    fn new(parent: Option<Arc<Self>>, last: Arc<T>) -> Self {
        Self { parent, last }
    }

    fn last(&self) -> &T {
        &self.last
    }

    fn parent(&self) -> Option<&Arc<Self>> {
        self.parent.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_no_parent() {
        let op1 = Arc::new(dypdl::Transition {
            name: String::from("op1"),
            ..Default::default()
        });
        let chain = ArcChain::new(None, op1.clone());
        assert_eq!(chain.parent(), None);
        assert_eq!(chain.last(), &*op1);
    }

    #[test]
    fn new_with_parent() {
        let op1 = Arc::new(dypdl::Transition {
            name: String::from("op1"),
            ..Default::default()
        });
        let chain1 = Arc::new(ArcChain::new(None, op1));
        let op2 = Arc::new(dypdl::Transition {
            name: String::from("op2"),
            ..Default::default()
        });
        let chain2 = ArcChain::new(Some(chain1.clone()), op2.clone());
        assert_eq!(chain2.parent(), Some(&chain1));
        assert_eq!(chain2.last(), &*op2);
    }

    #[test]
    fn trace_transitions_test() {
        let op1 = dypdl::Transition {
            name: String::from("op1"),
            ..Default::default()
        };
        let chain1 = Arc::new(ArcChain::new(None, Arc::new(op1.clone())));
        let op2 = dypdl::Transition {
            name: String::from("op2"),
            ..Default::default()
        };
        let chain2 = Arc::new(ArcChain::new(Some(chain1), Arc::new(op2.clone())));
        let result = chain2.transitions();
        let expected = vec![op1, op2];
        assert_eq!(result, expected);
    }
}
