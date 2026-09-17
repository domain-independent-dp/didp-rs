use dypdl::variable_type::Numeric;
use dypdl::{Model, ReduceFunction, StateInterface, StateMetadata};
use std::cmp::Ordering;

/// Returns if the given value equals to or exceeds the given bound.
/// The comparison depends on if the model is for minimization or maximization.
///
/// # Examples
///
/// ```
/// use dypdl::prelude::*;
/// use dypdl_heuristic_search::search_algorithm::data_structure::exceed_bound;
///
/// let mut model = Model::default();
/// model.set_minimize();
///
/// assert!(!exceed_bound(&model, 1, None));
/// assert!(exceed_bound(&model, 1, Some(1)));
/// assert!(!exceed_bound(&model, 1, Some(2)));
/// ```
pub fn exceed_bound<T: Numeric + PartialOrd>(model: &Model, value: T, bound: Option<T>) -> bool {
    bound.is_some_and(|bound| match model.reduce_function {
        ReduceFunction::Max => value <= bound,
        ReduceFunction::Min => value >= bound,
        _ => false,
    })
}

fn lexicographic_cmp<T, F, G>(x: &F, y: &G, less_is_better: &[bool]) -> Ordering
where
    T: PartialOrd,
    F: Fn(usize) -> T,
    G: Fn(usize) -> T,
{
    for (i, l) in less_is_better.iter().enumerate() {
        let a = x(i);
        let b = y(i);

        if *l {
            if a < b {
                return Ordering::Greater;
            } else if a > b {
                return Ordering::Less;
            }
        } else if a < b {
            return Ordering::Less;
        } else if a > b {
            return Ordering::Greater;
        }
    }

    Ordering::Equal
}

/// Compares resource variables of two states lexicographically.
///
/// The order of comparison is element resource variables, integer resource variables,
/// continuous resource variables, and set resource variables, each in the order they were
/// added to the model. Whether a smaller or greater value is preferred for each resource
/// variable is determined by the model.
pub fn lexicographic_resource_variables_cmp<S: StateInterface>(
    metadata: &StateMetadata,
    a: &S,
    b: &S,
) -> Ordering {
    let x = |i| a.get_element_resource_variable(i);
    let y = |i| b.get_element_resource_variable(i);
    let result = lexicographic_cmp(&x, &y, &metadata.element_less_is_better);

    if result != Ordering::Equal {
        return result;
    }

    let x = |i| a.get_integer_resource_variable(i);
    let y = |i| b.get_integer_resource_variable(i);
    let result = lexicographic_cmp(&x, &y, &metadata.integer_less_is_better);

    if result != Ordering::Equal {
        return result;
    }

    let x = |i| a.get_continuous_resource_variable(i);
    let y = |i| b.get_continuous_resource_variable(i);
    let result = lexicographic_cmp(&x, &y, &metadata.continuous_less_is_better);

    if result != Ordering::Equal {
        return result;
    }

    let x = |i| a.get_set_resource_variable(i);
    let y = |i| b.get_set_resource_variable(i);

    lexicographic_cmp(&x, &y, &metadata.set_less_is_better)
}

#[cfg(test)]
mod tests {
    use dypdl::{variable_type::Set, ResourceVariables, State};

    use super::*;

    #[test]
    fn exceed_bound_none() {
        let model = Model::default();
        assert!(!exceed_bound(&model, 0, None));
    }

    #[test]
    fn exceed_bound_minimization() {
        let model = Model {
            reduce_function: ReduceFunction::Min,
            ..Default::default()
        };
        assert!(exceed_bound(&model, 0, Some(-1)));
    }

    #[test]
    fn not_exceed_bound_minimization() {
        let model = Model {
            reduce_function: ReduceFunction::Min,
            ..Default::default()
        };
        assert!(!exceed_bound(&model, 0, Some(1)));
    }

    #[test]
    fn exceed_bound_maximization() {
        let model = Model {
            reduce_function: ReduceFunction::Max,
            ..Default::default()
        };
        assert!(exceed_bound(&model, 0, Some(1)));
    }

    #[test]
    fn not_exceed_bound_maximization() {
        let model = Model {
            reduce_function: ReduceFunction::Max,
            ..Default::default()
        };
        assert!(!exceed_bound(&model, 0, Some(-1)));
    }

    fn create_metadata() -> StateMetadata {
        let mut metadata = StateMetadata::default();
        let result = metadata.add_object_type("object", 4);
        let ob = result.unwrap();
        let result = metadata.add_element_resource_variable("erv1", ob, false);
        assert!(result.is_ok());
        let result = metadata.add_element_resource_variable("erv2", ob, true);
        assert!(result.is_ok());
        let result = metadata.add_element_resource_variable("erv3", ob, true);
        assert!(result.is_ok());
        let result = metadata.add_set_resource_variable("srv1", ob, true);
        assert!(result.is_ok());
        let result = metadata.add_set_resource_variable("srv2", ob, false);
        assert!(result.is_ok());
        let result = metadata.add_set_resource_variable("srv3", ob, false);
        assert!(result.is_ok());
        let result = metadata.add_integer_resource_variable("irv1", false);
        assert!(result.is_ok());
        let result = metadata.add_integer_resource_variable("irv2", true);
        assert!(result.is_ok());
        let result = metadata.add_integer_resource_variable("irv3", true);
        assert!(result.is_ok());
        let result = metadata.add_continuous_resource_variable("crv1", true);
        assert!(result.is_ok());
        let result = metadata.add_continuous_resource_variable("crv2", false);
        assert!(result.is_ok());
        let result = metadata.add_continuous_resource_variable("crv3", false);
        assert!(result.is_ok());

        metadata
    }

    #[test]
    fn lexicographic_resource_variables_cmp_eq() {
        let metadata = create_metadata();

        let a = State {
            resource_variables: ResourceVariables {
                element_variables: vec![0, 1, 2],
                integer_variables: vec![-2, 1, 3],
                continuous_variables: vec![-5.0, 1.0, -2.0],
                set_variables: vec![
                    Set::with_capacity(4),
                    {
                        let mut set = Set::with_capacity(4);
                        set.insert(0);
                        set.insert(1);
                        set
                    },
                    {
                        let mut set = Set::with_capacity(4);
                        set.insert(1);
                        set.insert(2);
                        set
                    },
                ],
            },
            ..Default::default()
        };

        assert_eq!(
            lexicographic_resource_variables_cmp(&metadata, &a, &a),
            Ordering::Equal
        );
    }

    #[test]
    fn lexicographic_resource_variables_cmp_element() {
        let metadata = create_metadata();

        let a = State {
            resource_variables: ResourceVariables {
                element_variables: vec![0, 0, 1],
                integer_variables: vec![-2, 1, 3],
                continuous_variables: vec![-5.0, 1.0, -2.0],
                set_variables: vec![
                    Set::with_capacity(4),
                    {
                        let mut set = Set::with_capacity(4);
                        set.insert(0);
                        set.insert(1);
                        set
                    },
                    {
                        let mut set = Set::with_capacity(4);
                        set.insert(1);
                        set.insert(2);
                        set
                    },
                ],
            },
            ..Default::default()
        };

        let b = State {
            resource_variables: ResourceVariables {
                element_variables: vec![1, 1, 2],
                integer_variables: vec![1, 0, 4],
                continuous_variables: vec![-5.6, 1.5, -2.1],
                set_variables: vec![
                    Set::with_capacity(4),
                    {
                        let mut set = Set::with_capacity(4);
                        set.insert(0);
                        set
                    },
                    {
                        let mut set = Set::with_capacity(4);
                        set.insert(2);
                        set
                    },
                ],
            },
            ..Default::default()
        };

        assert_eq!(
            lexicographic_resource_variables_cmp(&metadata, &a, &b),
            Ordering::Less
        );
        assert_eq!(
            lexicographic_resource_variables_cmp(&metadata, &b, &a),
            Ordering::Greater
        );
    }

    #[test]
    fn lexicographic_resource_variables_cmp_element_less_is_better() {
        let metadata = create_metadata();

        let a = State {
            resource_variables: ResourceVariables {
                element_variables: vec![0, 1, 2],
                integer_variables: vec![-2, 1, 3],
                continuous_variables: vec![-5.0, 1.0, -2.0],
                set_variables: vec![
                    Set::with_capacity(4),
                    {
                        let mut set = Set::with_capacity(4);
                        set.insert(0);
                        set.insert(1);
                        set
                    },
                    {
                        let mut set = Set::with_capacity(4);
                        set.insert(1);
                        set.insert(2);
                        set
                    },
                ],
            },
            ..Default::default()
        };

        let b = State {
            resource_variables: ResourceVariables {
                element_variables: vec![0, 1, 1],
                integer_variables: vec![1, 0, 4],
                continuous_variables: vec![-5.6, 1.5, -2.1],
                set_variables: vec![
                    Set::with_capacity(4),
                    {
                        let mut set = Set::with_capacity(4);
                        set.insert(0);
                        set
                    },
                    {
                        let mut set = Set::with_capacity(4);
                        set.insert(2);
                        set
                    },
                ],
            },
            ..Default::default()
        };

        assert_eq!(
            lexicographic_resource_variables_cmp(&metadata, &a, &b),
            Ordering::Less
        );
        assert_eq!(
            lexicographic_resource_variables_cmp(&metadata, &b, &a),
            Ordering::Greater
        );
    }

    #[test]
    fn lexicographic_resource_variables_cmp_integer() {
        let metadata = create_metadata();

        let a = State {
            resource_variables: ResourceVariables {
                element_variables: vec![0, 1, 2],
                integer_variables: vec![-2, 0, -1],
                continuous_variables: vec![-5.0, 1.0, -2.0],
                set_variables: vec![
                    Set::with_capacity(4),
                    {
                        let mut set = Set::with_capacity(4);
                        set.insert(0);
                        set.insert(1);
                        set
                    },
                    {
                        let mut set = Set::with_capacity(4);
                        set.insert(1);
                        set.insert(2);
                        set
                    },
                ],
            },
            ..Default::default()
        };

        let b = State {
            resource_variables: ResourceVariables {
                element_variables: vec![0, 1, 2],
                integer_variables: vec![-1, 1, 3],
                continuous_variables: vec![-5.6, 1.2, -2.1],
                set_variables: vec![
                    Set::with_capacity(4),
                    {
                        let mut set = Set::with_capacity(4);
                        set.insert(0);
                        set
                    },
                    {
                        let mut set = Set::with_capacity(4);
                        set.insert(1);
                        set
                    },
                ],
            },
            ..Default::default()
        };

        assert_eq!(
            lexicographic_resource_variables_cmp(&metadata, &a, &b),
            Ordering::Less
        );
        assert_eq!(
            lexicographic_resource_variables_cmp(&metadata, &b, &a),
            Ordering::Greater
        );
    }

    #[test]
    fn lexicographic_resource_variables_cmp_integer_less_is_better() {
        let metadata = create_metadata();

        let a = State {
            resource_variables: ResourceVariables {
                element_variables: vec![0, 1, 2],
                integer_variables: vec![-2, 1, 2],
                continuous_variables: vec![-5.0, 1.0, -2.0],
                set_variables: vec![
                    Set::with_capacity(4),
                    {
                        let mut set = Set::with_capacity(4);
                        set.insert(0);
                        set.insert(1);
                        set
                    },
                    {
                        let mut set = Set::with_capacity(4);
                        set.insert(1);
                        set.insert(2);
                        set
                    },
                ],
            },
            ..Default::default()
        };

        let b = State {
            resource_variables: ResourceVariables {
                element_variables: vec![0, 1, 2],
                integer_variables: vec![-2, 0, 3],
                continuous_variables: vec![-5.6, 1.2, -2.1],
                set_variables: vec![
                    Set::with_capacity(4),
                    {
                        let mut set = Set::with_capacity(4);
                        set.insert(0);
                        set
                    },
                    {
                        let mut set = Set::with_capacity(4);
                        set.insert(1);
                        set
                    },
                ],
            },
            ..Default::default()
        };

        assert_eq!(
            lexicographic_resource_variables_cmp(&metadata, &a, &b),
            Ordering::Less
        );
        assert_eq!(
            lexicographic_resource_variables_cmp(&metadata, &b, &a),
            Ordering::Greater
        );
    }

    #[test]
    fn lexicographic_resource_variables_cmp_continuous() {
        let metadata = create_metadata();

        let a = State {
            resource_variables: ResourceVariables {
                element_variables: vec![0, 1, 2],
                integer_variables: vec![-2, 1, 3],
                continuous_variables: vec![-5.0, 1.0, -1.5],
                set_variables: vec![
                    Set::with_capacity(4),
                    {
                        let mut set = Set::with_capacity(4);
                        set.insert(0);
                        set.insert(1);
                        set
                    },
                    {
                        let mut set = Set::with_capacity(4);
                        set.insert(1);
                        set.insert(2);
                        set
                    },
                ],
            },
            ..Default::default()
        };

        let b = State {
            resource_variables: ResourceVariables {
                element_variables: vec![0, 1, 2],
                integer_variables: vec![-2, 1, 3],
                continuous_variables: vec![-5.0, 1.5, -2.0],
                set_variables: vec![
                    Set::with_capacity(4),
                    {
                        let mut set = Set::with_capacity(4);
                        set.insert(0);
                        set
                    },
                    {
                        let mut set = Set::with_capacity(4);
                        set.insert(1);
                        set
                    },
                ],
            },
            ..Default::default()
        };

        assert_eq!(
            lexicographic_resource_variables_cmp(&metadata, &a, &b),
            Ordering::Less
        );
        assert_eq!(
            lexicographic_resource_variables_cmp(&metadata, &b, &a),
            Ordering::Greater
        );
    }

    #[test]
    fn lexicographic_resource_variables_cmp_continuous_less_is_better() {
        let metadata = create_metadata();

        let a = State {
            resource_variables: ResourceVariables {
                element_variables: vec![0, 1, 2],
                integer_variables: vec![-2, 1, 3],
                continuous_variables: vec![-5.0, 0.0, -3.0],
                set_variables: vec![
                    Set::with_capacity(4),
                    {
                        let mut set = Set::with_capacity(4);
                        set.insert(0);
                        set.insert(1);
                        set
                    },
                    {
                        let mut set = Set::with_capacity(4);
                        set.insert(1);
                        set.insert(2);
                        set
                    },
                ],
            },
            ..Default::default()
        };

        let b = State {
            resource_variables: ResourceVariables {
                element_variables: vec![0, 1, 2],
                integer_variables: vec![-2, 1, 3],
                continuous_variables: vec![-5.5, 1.0, -2.0],
                set_variables: vec![
                    Set::with_capacity(4),
                    {
                        let mut set = Set::with_capacity(4);
                        set.insert(0);
                        set
                    },
                    {
                        let mut set = Set::with_capacity(4);
                        set.insert(1);
                        set
                    },
                ],
            },
            ..Default::default()
        };

        assert_eq!(
            lexicographic_resource_variables_cmp(&metadata, &a, &b),
            Ordering::Less
        );
        assert_eq!(
            lexicographic_resource_variables_cmp(&metadata, &b, &a),
            Ordering::Greater
        );
    }

    #[test]
    fn lexicographic_resource_variables_cmp_set() {
        let metadata = create_metadata();

        let a = State {
            resource_variables: ResourceVariables {
                element_variables: vec![0, 1, 2],
                integer_variables: vec![-2, 1, 3],
                continuous_variables: vec![-5.0, 1.0, -2.0],
                set_variables: vec![
                    Set::with_capacity(4),
                    {
                        let mut set = Set::with_capacity(4);
                        set.insert(0);
                        set.insert(1);
                        set
                    },
                    {
                        let mut set = Set::with_capacity(4);
                        set.insert(1);
                        set.insert(2);
                        set
                    },
                ],
            },
            ..Default::default()
        };

        let b = State {
            resource_variables: ResourceVariables {
                element_variables: vec![0, 1, 2],
                integer_variables: vec![-2, 1, 3],
                continuous_variables: vec![-5.0, 1.5, -2.0],
                set_variables: vec![
                    Set::with_capacity(4),
                    {
                        let mut set = Set::with_capacity(4);
                        set.insert(0);
                        set.insert(1);
                        set.insert(2);
                        set
                    },
                    {
                        let mut set = Set::with_capacity(4);
                        set.insert(1);
                        set
                    },
                ],
            },
            ..Default::default()
        };

        assert_eq!(
            lexicographic_resource_variables_cmp(&metadata, &a, &b),
            Ordering::Less
        );
        assert_eq!(
            lexicographic_resource_variables_cmp(&metadata, &b, &a),
            Ordering::Greater
        );
    }

    #[test]
    fn lexicographic_resource_variables_cmp_set_less_is_better() {
        let metadata = create_metadata();

        let a = State {
            resource_variables: ResourceVariables {
                element_variables: vec![0, 1, 2],
                integer_variables: vec![-2, 1, 3],
                continuous_variables: vec![-5.0, 1.0, -2.0],
                set_variables: vec![
                    {
                        let mut set = Set::with_capacity(4);
                        set.insert(0);
                        set
                    },
                    {
                        let mut set = Set::with_capacity(4);
                        set.insert(0);
                        set.insert(1);
                        set
                    },
                    {
                        let mut set = Set::with_capacity(4);
                        set.insert(1);
                        set.insert(2);
                        set
                    },
                ],
            },
            ..Default::default()
        };

        let b = State {
            resource_variables: ResourceVariables {
                element_variables: vec![0, 1, 2],
                integer_variables: vec![-2, 1, 3],
                continuous_variables: vec![-5.0, 1.5, -2.0],
                set_variables: vec![
                    Set::with_capacity(4),
                    {
                        let mut set = Set::with_capacity(4);
                        set.insert(0);
                        set
                    },
                    {
                        let mut set = Set::with_capacity(4);
                        set.insert(1);
                        set
                    },
                ],
            },
            ..Default::default()
        };

        assert_eq!(
            lexicographic_resource_variables_cmp(&metadata, &a, &b),
            Ordering::Less
        );
        assert_eq!(
            lexicographic_resource_variables_cmp(&metadata, &b, &a),
            Ordering::Greater
        );
    }
}
