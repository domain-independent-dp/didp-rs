use crate::state;
use crate::variable::Numeric;
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
    pub fn get_cost(&self, state: &state::State) -> Option<T> {
        if *state == self.state {
            Some(self.cost)
        } else {
            None
        }
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
    use std::rc::Rc;

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
            signature_variables: Rc::new(state::SignatureVariables {
                integer_variables: vec![1],
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[test]
    fn get_cost() {
        let state = generate_state();

        let base_state = BaseState {
            state: state::State {
                signature_variables: Rc::new(state::SignatureVariables {
                    integer_variables: vec![1],
                    ..Default::default()
                }),
                ..Default::default()
            },
            cost: 1,
        };
        assert_eq!(base_state.get_cost(&state), Some(1));

        let base_state = BaseState {
            state: state::State {
                signature_variables: Rc::new(state::SignatureVariables {
                    integer_variables: vec![2],
                    ..Default::default()
                }),
                ..Default::default()
            },
            cost: 1,
        };
        assert_eq!(base_state.get_cost(&state), None);
    }

    #[test]
    fn load_from_yaml_ok() {
        let metadata = generate_metadata();

        let expected = BaseState {
            state: state::State {
                signature_variables: Rc::new(state::SignatureVariables {
                    integer_variables: vec![1],
                    ..Default::default()
                }),
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
