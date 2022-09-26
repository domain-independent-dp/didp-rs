use dypdl::variable_type::{Continuous, Element, Integer, Numeric, OrderedContinuous};
use dypdl::Transition;
use dypdl_heuristic_search::Solution;
use serde::Serialize;
use std::collections::BTreeMap;
use std::error::Error;
use std::fs;

#[derive(Debug, PartialEq, Serialize)]
struct TransitionToDump {
    name: String,
    parameters: BTreeMap<String, Element>,
}

impl From<Transition> for TransitionToDump {
    fn from(transition: Transition) -> Self {
        Self {
            name: transition.name,
            parameters: BTreeMap::from_iter(
                transition
                    .parameter_names
                    .into_iter()
                    .zip(transition.parameter_values),
            ),
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum CostToDump {
    Integer(Integer),
    Continuous(Continuous),
}

impl Serialize for CostToDump {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            Self::Integer(value) => serializer.serialize_i32(*value),
            Self::Continuous(value) => serializer.serialize_f64(*value),
        }
    }
}

impl From<Integer> for CostToDump {
    fn from(cost: Integer) -> Self {
        Self::Integer(cost)
    }
}

impl From<OrderedContinuous> for CostToDump {
    fn from(cost: OrderedContinuous) -> Self {
        Self::Continuous(cost.into_inner())
    }
}

#[derive(Debug, PartialEq, Serialize)]
pub struct SolutionToDump {
    cost: Option<CostToDump>,
    transitions: Vec<TransitionToDump>,
}

impl<T: Numeric> From<Solution<T>> for SolutionToDump
where
    CostToDump: From<T>,
{
    fn from(solution: Solution<T>) -> Self {
        Self {
            cost: solution.cost.map(CostToDump::from),
            transitions: solution
                .transitions
                .into_iter()
                .map(TransitionToDump::from)
                .collect(),
        }
    }
}

impl SolutionToDump {
    pub fn dump_to_file(&self, filename: &str) -> Result<(), Box<dyn Error>> {
        let solution = serde_yaml::to_string(self)?;
        fs::write(filename, solution)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cost_to_dump_from_integer() {
        assert_eq!(CostToDump::from(10), CostToDump::Integer(10));
    }

    #[test]
    fn cost_to_dump_from_ordered_continuous() {
        assert_eq!(
            CostToDump::from(OrderedContinuous::from(10.5)),
            CostToDump::Continuous(10.5)
        );
    }

    #[test]
    fn transition_from() {
        let transition = Transition {
            name: String::from("transition"),
            parameter_names: vec![String::from("p1"), String::from("p2")],
            parameter_values: vec![0, 1],
            ..Default::default()
        };
        assert_eq!(
            TransitionToDump::from(transition),
            TransitionToDump {
                name: String::from("transition"),
                parameters: BTreeMap::from([(String::from("p1"), 0), (String::from("p2"), 1)])
            }
        )
    }

    #[test]
    fn solution_from() {
        let solution = Solution {
            cost: Some(10),
            transitions: vec![
                Transition {
                    name: String::from("transition1"),
                    parameter_names: vec![String::from("p1"), String::from("p2")],
                    parameter_values: vec![0, 1],
                    ..Default::default()
                },
                Transition {
                    name: String::from("transition2"),
                    parameter_names: vec![String::from("p1"), String::from("p2")],
                    parameter_values: vec![1, 2],
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        assert_eq!(
            SolutionToDump::from(solution),
            SolutionToDump {
                cost: Some(CostToDump::Integer(10)),
                transitions: vec![
                    TransitionToDump {
                        name: String::from("transition1"),
                        parameters: BTreeMap::from([
                            (String::from("p1"), 0),
                            (String::from("p2"), 1)
                        ])
                    },
                    TransitionToDump {
                        name: String::from("transition2"),
                        parameters: BTreeMap::from([
                            (String::from("p1"), 1),
                            (String::from("p2"), 2)
                        ])
                    }
                ]
            }
        );
    }
}
