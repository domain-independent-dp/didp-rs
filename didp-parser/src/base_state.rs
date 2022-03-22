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
