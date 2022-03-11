use std::cmp::Ordering;
use std::rc::Rc;

use crate::variable;

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct State<T: variable::Numeric> {
    pub signature_variables: Rc<SignatureVariables<T>>,
    pub resource_variables: ResourceVariables<T>,
    pub cost: T,
}

#[derive(Debug, Hash, PartialEq, Eq, Clone)]
pub struct SignatureVariables<T: variable::Numeric> {
    pub set_variables: Vec<variable::SetVariable>,
    pub permutation_variables: Vec<variable::PermutationVariable>,
    pub element_variables: Vec<variable::ElementVariable>,
    pub numeric_variables: Vec<T>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct ResourceVariables<T: variable::Numeric> {
    pub numeric_variables: Vec<T>,
}

impl<T: variable::Numeric> PartialOrd for ResourceVariables<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let state = Some(Ordering::Equal);
        dominance(state, &self.numeric_variables, &other.numeric_variables)
    }
}

fn dominance<T: variable::Numeric>(state: Option<Ordering>, a: &[T], b: &[T]) -> Option<Ordering> {
    debug_assert!(a.len() == b.len());

    let mut result = match state {
        Some(_) => state,
        None => return None,
    };
    for (v1, v2) in a.iter().zip(b.iter()) {
        match result {
            Some(Ordering::Equal) => {
                if v1 < v2 {
                    result = Some(Ordering::Less);
                }
                if v1 > v2 {
                    result = Some(Ordering::Greater);
                }
            }
            Some(Ordering::Less) => {
                if v1 > v2 {
                    return None;
                }
            }
            Some(Ordering::Greater) => {
                if v1 < v2 {
                    return None;
                }
            }
            None => {}
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resource_variables_eq() {
        let a = ResourceVariables {
            numeric_variables: vec![1, 2],
        };
        let b = ResourceVariables {
            numeric_variables: vec![1, 2],
        };
        assert_eq!(a, b);
    }

    #[test]
    fn resource_variables_lt() {
        let a = ResourceVariables {
            numeric_variables: vec![1, 2],
        };
        let b = ResourceVariables {
            numeric_variables: vec![1, -2],
        };
        assert!(a >= b);
        assert!(a > b);
        assert!(b <= a);
        assert!(b < a);
    }

    #[test]
    fn resource_variables_neq() {
        let a = ResourceVariables {
            numeric_variables: vec![1, 2],
        };
        let b = ResourceVariables {
            numeric_variables: vec![3, 1],
        };
        assert_eq!(a.partial_cmp(&b), None);
    }

    #[test]
    #[should_panic]
    fn resource_variables_ilength_panic() {
        let a = ResourceVariables {
            numeric_variables: vec![1, 2],
        };
        let b = ResourceVariables {
            numeric_variables: vec![1],
        };
        let _ = a < b;
    }
}
