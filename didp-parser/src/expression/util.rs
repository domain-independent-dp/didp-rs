use crate::variable::{Element, Set};
use std::iter;

pub fn expand_vector_with_set(vector: Vec<Vec<Element>>, set: &Set) -> Vec<Vec<Element>> {
    vector
        .into_iter()
        .flat_map(|r| {
            iter::repeat(r)
                .zip(set.ones())
                .map(|(mut r, e)| {
                    r.push(e);
                    r
                })
                .collect::<Vec<Vec<Element>>>()
        })
        .collect()
}

pub fn expand_vector_with_slice(vector: Vec<Vec<Element>>, slice: &[Element]) -> Vec<Vec<Element>> {
    vector
        .into_iter()
        .flat_map(|r| {
            iter::repeat(r)
                .zip(slice.iter())
                .map(|(mut r, e)| {
                    r.push(*e);
                    r
                })
                .collect::<Vec<Vec<Element>>>()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expand_vector_with_set() {
        let vector = vec![vec![0, 1, 2], vec![3, 4, 5]];
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(2);
        assert_eq!(
            expand_vector_with_set(vector, &set),
            vec![
                vec![0, 1, 2, 0],
                vec![0, 1, 2, 2],
                vec![3, 4, 5, 0],
                vec![3, 4, 5, 2]
            ]
        )
    }

    #[test]
    fn test_expand_vector_with_slice() {
        let vector = vec![vec![0, 1, 2], vec![3, 4, 5]];
        let vector2 = vec![2, 0];
        assert_eq!(
            expand_vector_with_slice(vector, &vector2),
            vec![
                vec![0, 1, 2, 2],
                vec![0, 1, 2, 0],
                vec![3, 4, 5, 2],
                vec![3, 4, 5, 0]
            ]
        )
    }
}
