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
