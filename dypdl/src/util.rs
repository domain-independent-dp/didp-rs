use std::collections::hash_map::Entry;
use std::error::Error;
use std::fmt;

use rustc_hash::FxHashMap;

/// Error in modeling.
#[derive(Debug)]
pub struct ModelErr(String);

impl ModelErr {
    /// Creates a new error.
    pub fn new(message: String) -> ModelErr {
        ModelErr(format!("Error in problem definition: {}", message))
    }
}

impl fmt::Display for ModelErr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Error for ModelErr {}

pub fn get_id(name: &str, name_to_id: &FxHashMap<String, usize>) -> Result<usize, ModelErr> {
    if let Some(id) = name_to_id.get(name) {
        Ok(*id)
    } else {
        Err(ModelErr::new(format!("no such name `{}`", name)))
    }
}

pub fn add_name<T>(
    name: T,
    names: &mut Vec<String>,
    name_to_id: &mut FxHashMap<String, usize>,
) -> Result<usize, ModelErr>
where
    String: From<T>,
{
    let name = String::from(name);

    match name_to_id.entry(name) {
        Entry::Vacant(e) => {
            let id = names.len();
            names.push(e.key().clone());
            e.insert(id);

            Ok(id)
        }
        Entry::Occupied(e) => Err(ModelErr::new(format!("name `{}` is already used", e.key()))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_id() {
        let name_to_id = FxHashMap::from(
            [(String::from("a"), 0), (String::from("b"), 1)]
                .into_iter()
                .collect(),
        );

        let result = get_id("a", &name_to_id);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0);

        let result = get_id("b", &name_to_id);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 1);

        let result = get_id("c", &name_to_id);
        assert!(result.is_err());
    }

    #[test]
    fn test_add_name() {
        let mut names = Vec::default();
        let mut name_to_id = FxHashMap::default();

        let result = add_name("a", &mut names, &mut name_to_id);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0);

        let result = add_name("b", &mut names, &mut name_to_id);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 1);

        let result = add_name("a", &mut names, &mut name_to_id);
        assert!(result.is_err());
    }
}
