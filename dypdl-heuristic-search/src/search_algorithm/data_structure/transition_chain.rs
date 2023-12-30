use dypdl::{Transition, TransitionInterface};
use std::rc::Rc;

/// Trait to get a sequence of transitions.
pub trait GetTransitions<V = Transition> {
    /// Returns transitions to reach this node.
    fn transitions(&self) -> Vec<V>;

    /// Returns the last transition to reach this node.
    fn last(&self) -> Option<&V>;
}

/// Trait to create a chain of transitions.
pub trait CreateTransitionChain<R, P> {
    /// Returns a new transition chain.
    fn new(parent: Option<P>, last: R) -> Self;
}

/// Chain of transitions implemented by a linked list of `Rc`.
#[derive(PartialEq, Debug, Clone)]
pub struct RcChain<T = Transition>
where
    T: TransitionInterface + Clone,
    Transition: From<T>,
{
    parent: Option<Rc<Self>>,
    last: Rc<T>,
}

impl<T> CreateTransitionChain<Rc<T>, Rc<Self>> for RcChain<T>
where
    T: TransitionInterface + Clone,
    Transition: From<T>,
{
    fn new(parent: Option<Rc<Self>>, last: Rc<T>) -> Self {
        Self { parent, last }
    }
}

impl<T> GetTransitions<T> for RcChain<T>
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
        let op1 = Rc::new(dypdl::Transition {
            name: String::from("op1"),
            ..Default::default()
        });
        let chain = RcChain::new(None, op1.clone());
        assert_eq!(chain.parent, None);
        assert_eq!(chain.last(), Some(&*op1));
    }

    #[test]
    fn new_with_parent() {
        let op1 = Rc::new(dypdl::Transition {
            name: String::from("op1"),
            ..Default::default()
        });
        let chain1 = Rc::new(RcChain::new(None, op1));
        let op2 = Rc::new(dypdl::Transition {
            name: String::from("op2"),
            ..Default::default()
        });
        let chain2 = RcChain::new(Some(chain1.clone()), op2.clone());
        assert_eq!(chain2.parent, Some(chain1));
        assert_eq!(chain2.last(), Some(&*op2));
    }

    #[test]
    fn trace_transitions_test() {
        let op1 = dypdl::Transition {
            name: String::from("op1"),
            ..Default::default()
        };
        let chain1 = Rc::new(RcChain::new(None, Rc::new(op1.clone())));
        let op2 = dypdl::Transition {
            name: String::from("op2"),
            ..Default::default()
        };
        let chain2 = Rc::new(RcChain::new(Some(chain1), Rc::new(op2.clone())));
        let result = chain2.transitions();
        let expected = vec![op1, op2];
        assert_eq!(result, expected);
    }
}
