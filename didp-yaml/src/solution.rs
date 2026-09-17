//! Loading, validating, and dumping solutions in the solver's YAML format.

mod loader;
mod validation;

pub use loader::{load_solution_from_file, load_solution_from_str, load_solution_from_yaml};
pub use validation::{validate_cost, validate_solution, CostTolerance, SolutionError};

use dypdl::variable_type::{Continuous, Element, Integer, Numeric, OrderedContinuous};
use dypdl::Transition;
use dypdl_heuristic_search::Solution;
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::fs;
use yaml_rust::{yaml::Hash, Yaml, YamlEmitter};

#[derive(Debug, PartialEq)]
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

impl TransitionToDump {
    fn to_yaml(&self) -> Result<Yaml, Box<dyn Error>> {
        let mut parameters = Hash::new();

        for (name, value) in &self.parameters {
            parameters.insert(
                Yaml::String(name.clone()),
                Yaml::Integer(i64::try_from(*value)?),
            );
        }

        let mut yaml = Hash::new();
        yaml.insert(Yaml::from_str("name"), Yaml::String(self.name.clone()));
        yaml.insert(Yaml::from_str("parameters"), Yaml::Hash(parameters));
        Ok(Yaml::Hash(yaml))
    }
}

/// Cost of a solution that is serializable.
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum SolutionCost {
    /// Integer value.
    Integer(Integer),
    /// Continuous value.
    Continuous(Continuous),
}

/// Backward-compatible name for a serializable solution cost.
pub use SolutionCost as CostToDump;

impl fmt::Display for SolutionCost {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Integer(value) => write!(f, "{value}"),
            Self::Continuous(value) => write!(f, "{value}"),
        }
    }
}

/// A candidate solution resolved against an existing model.
///
/// Loading does not establish feasibility. `cost` is the optional declared cost;
/// [`validate_solution`] independently computes the cost and checks it if present.
#[derive(Debug, PartialEq, Clone)]
pub struct LoadedSolution {
    /// Declared objective value, or `None` when omitted or null in the YAML.
    pub cost: Option<SolutionCost>,
    /// Forward transitions from the model, in solution order.
    pub transitions: Vec<Transition>,
}

impl From<&CostToDump> for Yaml {
    fn from(cost: &CostToDump) -> Self {
        match cost {
            CostToDump::Integer(value) => Yaml::Integer(i64::from(*value)),
            CostToDump::Continuous(value) => Yaml::Real(value.to_string()),
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

/// Solution that is serializable.
#[derive(Debug, PartialEq)]
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
    fn to_yaml(&self) -> Result<Yaml, Box<dyn Error>> {
        let mut yaml = Hash::new();
        yaml.insert(
            Yaml::from_str("cost"),
            self.cost.as_ref().map(Yaml::from).unwrap_or(Yaml::Null),
        );
        let transitions = self
            .transitions
            .iter()
            .map(TransitionToDump::to_yaml)
            .collect::<Result<Vec<_>, _>>()?;
        yaml.insert(Yaml::from_str("transitions"), Yaml::Array(transitions));
        Ok(Yaml::Hash(yaml))
    }

    /// Output the solution to a file.
    pub fn dump_to_file(&self, filename: &str) -> Result<(), Box<dyn Error>> {
        fs::write(filename, self.dump_to_str()?)?;
        Ok(())
    }

    /// Returns a YAML document representing the solution.
    pub fn dump_to_str(&self) -> Result<String, Box<dyn Error>> {
        let mut solution = String::new();
        let mut emitter = YamlEmitter::new(&mut solution);
        emitter.dump(&self.to_yaml()?)?;
        Ok(solution)
    }
}

impl From<LoadedSolution> for SolutionToDump {
    fn from(solution: LoadedSolution) -> Self {
        Self {
            cost: solution.cost,
            transitions: solution
                .transitions
                .into_iter()
                .map(TransitionToDump::from)
                .collect(),
        }
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
