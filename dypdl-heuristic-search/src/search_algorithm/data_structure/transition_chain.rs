use core::ops::Deref;
use dypdl::{Transition, TransitionInterface};
use std::rc::Rc;

/// Trait to get a sequence of transitions.
pub trait GetTransitions<V = Transition> {
    /// Returns transitions to reach this node.
    fn transitions(&self) -> Vec<V>;

    /// Returns the last transition to reach this node.
    fn last(&self) -> Option<&V>;
}

/// Chain of transitions.
///
/// # Examples
///
/// ```
/// use dypdl::prelude::*;
/// use dypdl_heuristic_search::search_algorithm::data_structure::{
///     RcChain, TransitionChain
/// };
/// use std::rc::Rc;
///
/// let t1 = Transition::new("t1");
/// let chain = Rc::new(RcChain::new(None, Rc::new(t1.clone())));
/// assert_eq!(chain.last(), &t1);
/// assert_eq!(chain.parent(), None);
/// assert_eq!(chain.transitions(), vec![t1.clone()]);
///
/// let t2 = Transition::new("t2");
/// let new_chain = Rc::new(RcChain::new(Some(chain.clone()), Rc::new(t2.clone())));
/// assert_eq!(new_chain.last(), &t2);
/// assert_eq!(new_chain.parent(), Some(&chain));
/// assert_eq!(new_chain.transitions(), vec![t1, t2]);
/// ```
pub trait TransitionChain<T = Transition, R = Rc<T>, P = Rc<Self>>
where
    T: TransitionInterface + Clone,
    R: Deref<Target = T>,
    P: Deref<Target = Self>,
    Transition: From<T>,
{
    /// Returns a new transition chain.
    fn new(parent: Option<P>, last: R) -> Self;

    /// Returns the last transition.
    fn last(&self) -> &T;

    /// Returns a parent transition chain.
    fn parent(&self) -> Option<&P>;

    /// Returns transitions as a vector.
    fn transitions(&self) -> Vec<T> {
        let mut result = vec![self.last().clone()];
        let mut parent = self.parent();

        while let Some(current) = parent {
            result.push(current.last().clone());
            parent = current.parent();
        }

        result.reverse();
        result
    }
}

/// Chain of transitions implemented by a linked list of `Rc`.
#[derive(PartialEq, Debug)]
pub struct RcChain<T = Transition>
where
    T: TransitionInterface + Clone,
    Transition: From<T>,
{
    parent: Option<Rc<Self>>,
    last: Rc<T>,
}

impl<T> TransitionChain<T> for RcChain<T>
where
    T: TransitionInterface + Clone,
    Transition: From<T>,
{
    fn new(parent: Option<Rc<Self>>, last: Rc<T>) -> Self {
        Self { parent, last }
    }

    fn last(&self) -> &T {
        &self.last
    }

    fn parent(&self) -> Option<&Rc<Self>> {
        self.parent.as_ref()
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
        assert_eq!(chain.parent(), None);
        assert_eq!(chain.last(), &*op1);
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
        assert_eq!(chain2.parent(), Some(&chain1));
        assert_eq!(chain2.last(), &*op2);
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
