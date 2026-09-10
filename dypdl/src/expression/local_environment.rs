/// A local environment mapping a local element variable index to a value.
#[derive(Debug, Default, Clone)]
pub struct LocalEnvironment {
    id_to_value: Vec<Option<usize>>,
}

impl LocalEnvironment {
    /// Sets the value of a local element variable.
    pub fn set(&mut self, id: usize, value: usize) -> Option<usize> {
        if id >= self.id_to_value.len() {
            self.id_to_value.resize(id + 1, None);
        }

        let before = self.id_to_value[id];
        self.id_to_value[id] = Some(value);

        before
    }

    /// Unsets the value of a local element variable.
    pub fn unset(&mut self, id: usize) -> Option<usize> {
        if id >= self.id_to_value.len() {
            None
        } else {
            let before = self.id_to_value[id];
            self.id_to_value[id] = None;

            before
        }
    }

    /// Gets the value of a local element variable.
    pub fn get(&self, id: usize) -> Option<usize> {
        if id >= self.id_to_value.len() {
            None
        } else {
            self.id_to_value[id]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_get_and_unset_return_none() {
        let mut environment = LocalEnvironment::default();

        assert_eq!(environment.get(0), None);
        assert_eq!(environment.get(10), None);
        assert_eq!(environment.unset(0), None);
        assert_eq!(environment.unset(10), None);
    }

    #[test]
    fn set_resizes_sparse_environment() {
        let mut environment = LocalEnvironment::default();

        assert_eq!(environment.set(2, 5), None);
        assert_eq!(environment.get(0), None);
        assert_eq!(environment.get(1), None);
        assert_eq!(environment.get(2), Some(5));
        assert_eq!(environment.get(3), None);
    }

    #[test]
    fn set_returns_previous_value_and_overwrites() {
        let mut environment = LocalEnvironment::default();

        assert_eq!(environment.set(1, 3), None);
        assert_eq!(environment.set(1, 7), Some(3));
        assert_eq!(environment.get(1), Some(7));
    }

    #[test]
    fn unset_returns_previous_value_and_clears() {
        let mut environment = LocalEnvironment::default();

        environment.set(1, 3);

        assert_eq!(environment.unset(1), Some(3));
        assert_eq!(environment.get(1), None);
        assert_eq!(environment.unset(1), None);
    }
}
