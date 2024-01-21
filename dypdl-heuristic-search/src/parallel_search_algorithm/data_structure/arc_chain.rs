use crate::search_algorithm::data_structure::{CreateTransitionChain, GetTransitions};
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

impl<T> CreateTransitionChain<Arc<T>, Arc<Self>> for ArcChain<T>
where
    T: TransitionInterface + Clone,
    Transition: From<T>,
{
    fn new(parent: Option<Arc<Self>>, last: Arc<T>) -> Self {
        Self { parent, last }
    }
}

impl<T> GetTransitions<T> for ArcChain<T>
where
    T: TransitionInterface + Clone,
    Transition: From<T>,
{
    fn transitions(&self) -> Vec<T> {
        let mut result = vec![(*self.last).clone()];
        let mut parent = &self.parent;

        while let Some(current) = parent {
            result.push((*current.last).clone());
            parent = &current.parent;
        }

        result.reverse();
        result
    }

    fn last(&self) -> Option<&T> {
        Some(&self.last)
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
        assert_eq!(chain.parent, None);
        assert_eq!(chain.last(), Some(&*op1));
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
        assert_eq!(chain2.parent, Some(chain1));
        assert_eq!(chain2.last(), Some(&*op2));
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
