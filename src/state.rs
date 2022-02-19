use std::cmp::Ordering;
use std::rc::Rc;

use crate::variable;

#[derive(Debug, PartialEq, Eq)]
pub struct State {
    pub signature_variables: Rc<SignatureVariables>,
    pub resource_variables: ResourceVariables,
}

#[derive(Debug, Hash, PartialEq, Eq)]
pub struct SignatureVariables {
    pub set_variables: Vec<variable::SetVariable>,
    pub permutation_variables: Vec<variable::PermutationVariable>,
    pub element_variables: Vec<variable::ElementVariable>,
    pub integer_variables: Vec<variable::IntegerVariable>,
    pub continuous_variables: Vec<variable::ContinuousVariable>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct ResourceVariables {
    pub integer_variables: Vec<variable::IntegerVariable>,
    pub continuous_variables: Vec<variable::ContinuousVariable>,
}

impl PartialOrd for ResourceVariables {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let state = Some(Ordering::Equal);
        let state = dominance(state, &self.integer_variables, &other.integer_variables);
        dominance(
            state,
            &self.continuous_variables,
            &other.continuous_variables,
        )
    }
}

fn dominance<T: PartialOrd>(state: Option<Ordering>, a: &[T], b: &[T]) -> Option<Ordering> {
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
    use ordered_float::OrderedFloat;

    #[test]
    fn resource_variables_eq() {
        let a = ResourceVariables {
            integer_variables: vec![1, 2],
            continuous_variables: vec![OrderedFloat(1.0), OrderedFloat(2.0)],
        };
        let b = ResourceVariables {
            integer_variables: vec![1, 2],
            continuous_variables: vec![OrderedFloat(1.0), OrderedFloat(2.0)],
        };
        assert_eq!(a, b);
    }

    #[test]
    fn resource_variables_integer_lt() {
        let a = ResourceVariables {
            integer_variables: vec![1, 2],
            continuous_variables: vec![OrderedFloat(1.0), OrderedFloat(2.0)],
        };
        let b = ResourceVariables {
            integer_variables: vec![1, -2],
            continuous_variables: vec![OrderedFloat(1.0), OrderedFloat(2.0)],
        };
        assert!(a >= b);
        assert!(a > b);
        assert!(b <= a);
        assert!(b < a);
    }

    #[test]
    fn resource_variables_continuous_lt() {
        let a = ResourceVariables {
            integer_variables: vec![1, 2],
            continuous_variables: vec![OrderedFloat(1.0), OrderedFloat(2.0)],
        };
        let b = ResourceVariables {
            integer_variables: vec![1, 2],
            continuous_variables: vec![OrderedFloat(1.0), OrderedFloat(-1.0)],
        };
        assert!(a >= b);
        assert!(a > b);
        assert!(b <= a);
        assert!(b < a);
    }

    #[test]
    fn resource_variables_neq() {
        let a = ResourceVariables {
            integer_variables: vec![1, 2],
            continuous_variables: vec![OrderedFloat(1.0), OrderedFloat(2.0)],
        };
        let b = ResourceVariables {
            integer_variables: vec![3, 2],
            continuous_variables: vec![OrderedFloat(1.0), OrderedFloat(-1.0)],
        };
        assert_eq!(a.partial_cmp(&b), None);
        let b = ResourceVariables {
            integer_variables: vec![3, 1],
            continuous_variables: vec![OrderedFloat(1.0), OrderedFloat(-1.0)],
        };
        assert_eq!(a.partial_cmp(&b), None);
        let b = ResourceVariables {
            integer_variables: vec![1, 1],
            continuous_variables: vec![OrderedFloat(1.0), OrderedFloat(3.0)],
        };
        assert_eq!(a.partial_cmp(&b), None);
    }

    #[test]
    #[should_panic]
    fn resource_variables_integer_length_panic() {
        let a = ResourceVariables {
            integer_variables: vec![1, 2],
            continuous_variables: vec![OrderedFloat(1.0), OrderedFloat(2.0)],
        };
        let b = ResourceVariables {
            integer_variables: vec![1],
            continuous_variables: vec![OrderedFloat(1.0), OrderedFloat(2.0)],
        };
        let _ = a < b;
    }

    #[test]
    #[should_panic]
    fn resource_variables_continuous_length_panic() {
        let a = ResourceVariables {
            integer_variables: vec![1, 2],
            continuous_variables: vec![OrderedFloat(1.0), OrderedFloat(2.0)],
        };
        let b = ResourceVariables {
            integer_variables: vec![1, 2],
            continuous_variables: vec![OrderedFloat(1.0)],
        };
        let _ = a < b;
    }
}
