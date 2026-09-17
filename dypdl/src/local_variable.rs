use crate::util;
use crate::ModelErr;
use rustc_hash::FxHashMap;

/// Local variable bound to an element in a set.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct LocalVariable(usize);

impl LocalVariable {
    /// Returns the index.
    #[inline]
    pub fn id(&self) -> usize {
        self.0
    }
}

/// Data for local variables.
#[derive(Debug, PartialEq, Clone, Default)]
pub struct LocalVariableData {
    /// Names of local variables.
    pub names: Vec<String>,
    /// Mapping from a local variable name to its ID.
    pub name_to_id: FxHashMap<String, usize>,
}

impl LocalVariableData {
    /// Adds a local variable.
    pub fn add<T>(&mut self, name: T) -> Result<LocalVariable, ModelErr>
    where
        String: From<T>,
    {
        let id = util::add_name(name, &mut self.names, &mut self.name_to_id)?;

        Ok(LocalVariable(id))
    }

    /// Returns a local variable given a name.
    pub fn get(&self, name: &str) -> Result<LocalVariable, ModelErr> {
        let id = util::get_id(name, &self.name_to_id)?;

        Ok(LocalVariable(id))
    }

    /// Returns the number of local variables.
    pub fn number_of_variables(&self) -> usize {
        self.names.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_variable_id() {
        let variable = LocalVariable(2);

        assert_eq!(variable.id(), 2);
    }

    #[test]
    fn add_get_and_count_variables() {
        let mut data = LocalVariableData::default();

        let x = data.add("x").unwrap();
        let y = data.add(String::from("y")).unwrap();

        assert_eq!(x.id(), 0);
        assert_eq!(y.id(), 1);
        assert_eq!(data.get("x").unwrap(), x);
        assert_eq!(data.get("y").unwrap(), y);
        assert_eq!(data.number_of_variables(), 2);
    }

    #[test]
    fn add_duplicate_name_err() {
        let mut data = LocalVariableData::default();

        assert!(data.add("x").is_ok());
        assert!(data.add("x").is_err());
    }

    #[test]
    fn get_missing_name_err() {
        let data = LocalVariableData::default();

        assert!(data.get("x").is_err());
    }
}
