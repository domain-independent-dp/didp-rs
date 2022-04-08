use crate::state;
use crate::state::DPState;
use crate::variable::{Continuous, Numeric};
use crate::yaml_util;
use lazy_static::lazy_static;
use std::error::Error;
use std::fmt;
use std::str;

#[derive(Debug, PartialEq, Clone, Default)]
pub struct BaseState<T: Numeric> {
    pub state: state::State,
    pub cost: T,
}

impl<T: Numeric> BaseState<T> {
    pub fn get_cost<U: DPState>(&self, state: &U, metadata: &state::StateMetadata) -> Option<T> {
        for i in 0..metadata.number_of_element_variables() {
            if self.state.get_element_variable(i) != state.get_element_variable(i) {
                return None;
            }
        }
        for i in 0..metadata.number_of_integer_variables() {
            if self.state.get_integer_variable(i) != state.get_integer_variable(i) {
                return None;
            }
        }
        for i in 0..metadata.number_of_integer_resource_variables() {
            if self.state.get_integer_resource_variable(i) != state.get_integer_resource_variable(i)
            {
                return None;
            }
        }
        for i in 0..metadata.number_of_continuous_variables() {
            if (self.state.get_continuous_variable(i) - state.get_continuous_variable(i)).abs()
                > Continuous::EPSILON
            {
                return None;
            }
        }
        for i in 0..metadata.number_of_continuous_resource_variables() {
            if (self.state.get_continuous_resource_variable(i)
                - state.get_continuous_resource_variable(i))
            .abs()
                > Continuous::EPSILON
            {
                return None;
            }
        }
        for i in 0..metadata.number_of_set_variables() {
            if self.state.get_set_variable(i) != state.get_set_variable(i) {
                return None;
            }
        }
        for i in 0..metadata.number_of_vector_variables() {
            if self.state.get_vector_variable(i) != state.get_vector_variable(i) {
                return None;
            }
        }
        Some(self.cost)
    }

    pub fn load_from_yaml(
        value: &yaml_rust::Yaml,
        metadata: &state::StateMetadata,
    ) -> Result<BaseState<T>, Box<dyn Error>>
    where
        <T as str::FromStr>::Err: fmt::Debug,
    {
        lazy_static! {
            static ref COST_KEY: yaml_rust::Yaml = yaml_rust::Yaml::from_str("cost");
        }
        let map = yaml_util::get_map(value)?;
        let state = yaml_util::get_yaml_by_key(map, "state")?;
        let state = state::State::load_from_yaml(state, metadata)?;
        match map.get(&COST_KEY) {
            Some(cost) => {
                let cost = yaml_util::get_numeric(cost)?;
                Ok(BaseState { state, cost })
            }
            None => Ok(BaseState {
                state,
                cost: T::zero(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::variable;
    use rustc_hash::FxHashMap;

    fn generate_metadata() -> state::StateMetadata {
        let mut name_to_integer_variable = FxHashMap::default();
        name_to_integer_variable.insert(String::from("i0"), 0);
        state::StateMetadata {
            integer_variable_names: vec![String::from("i0")],
            name_to_integer_variable,
            ..Default::default()
        }
    }

    fn generate_state() -> state::State {
        state::State {
            signature_variables: state::SignatureVariables {
                integer_variables: vec![1],
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn get_cost() {
        let state = generate_state();
        let metadata = generate_metadata();

        let base_state = BaseState {
            state: state::State {
                signature_variables: state::SignatureVariables {
                    integer_variables: vec![1],
                    ..Default::default()
                },
                ..Default::default()
            },
            cost: 1,
        };
        assert_eq!(base_state.get_cost(&state, &metadata), Some(1));

        let base_state = BaseState {
            state: state::State {
                signature_variables: state::SignatureVariables {
                    integer_variables: vec![2],
                    ..Default::default()
                },
                ..Default::default()
            },
            cost: 1,
        };
        assert_eq!(base_state.get_cost(&state, &metadata), None);
    }

    #[test]
    fn load_from_yaml_ok() {
        let metadata = generate_metadata();

        let expected = BaseState {
            state: state::State {
                signature_variables: state::SignatureVariables {
                    integer_variables: vec![1],
                    ..Default::default()
                },
                ..Default::default()
            },
            cost: 0,
        };

        let base_state = yaml_rust::YamlLoader::load_from_str(
            r"
state: { i0: 1 }        
",
        );
        assert!(base_state.is_ok());
        let base_state = base_state.unwrap();
        assert_eq!(base_state.len(), 1);
        let base_state = BaseState::load_from_yaml(&base_state[0], &metadata);
        assert!(base_state.is_ok());
        assert_eq!(base_state.unwrap(), expected);

        let base_state = yaml_rust::YamlLoader::load_from_str(
            r"
state: { i0: 1 }        
cost: 0
",
        );
        assert!(base_state.is_ok());
        let base_state = base_state.unwrap();
        assert_eq!(base_state.len(), 1);
        let base_state = BaseState::load_from_yaml(&base_state[0], &metadata);
        assert!(base_state.is_ok());
        assert_eq!(base_state.unwrap(), expected);
    }

    #[test]
    fn load_from_yaml_err() {
        let metadata = generate_metadata();

        let base_state = yaml_rust::YamlLoader::load_from_str(
            r"
{ i0: 1 }        
",
        );
        assert!(base_state.is_ok());
        let base_state = base_state.unwrap();
        assert_eq!(base_state.len(), 1);
        let base_state = BaseState::<variable::Integer>::load_from_yaml(&base_state[0], &metadata);
        assert!(base_state.is_err());
    }
}
