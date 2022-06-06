use rustc_hash::FxHashMap;
use std::error::Error;
use std::fmt;
use std::str;
use yaml_rust::Yaml;

mod base_case;
mod effect;
pub mod expression;
pub mod expression_parser;
mod grounded_condition;
mod parse_expression_from_yaml;
mod state;
pub mod table;
mod table_data;
mod table_registry;
mod transition;
mod util;
pub mod variable;
mod yaml_util;

pub use base_case::BaseCase;
pub use effect::Effect;
pub use expression_parser::ParseErr;
pub use grounded_condition::GroundedCondition;
pub use state::{DPState, ResourceVariables, SignatureVariables, State, StateMetadata};
pub use table::{Table, Table1D, Table2D, Table3D};
pub use table_data::TableData;
pub use table_registry::TableRegistry;
pub use transition::Transition;
pub use util::ModelErr;

#[derive(Debug, PartialEq, Clone)]
pub enum ReduceFunction {
    Min,
    Max,
    Sum,
    Product,
}

impl Default for ReduceFunction {
    fn default() -> Self {
        Self::Min
    }
}

impl ReduceFunction {
    pub fn load_from_yaml(value: &Yaml) -> Result<ReduceFunction, Box<dyn Error>> {
        let reduce_function = yaml_util::get_string(value)?;
        match &reduce_function[..] {
            "min" => Ok(Self::Min),
            "max" => Ok(Self::Max),
            "sum" => Ok(Self::Sum),
            "product" => Ok(Self::Product),
            _ => Err(yaml_util::YamlContentErr::new(format!(
                "no such reduce function `{}`",
                reduce_function
            ))
            .into()),
        }
    }
}

#[derive(Debug, PartialEq, Clone, Default)]
pub struct Model<T: variable::Numeric> {
    pub domain_name: String,
    pub problem_name: String,
    pub state_metadata: state::StateMetadata,
    pub target: state::State,
    pub table_registry: table_registry::TableRegistry,
    pub constraints: Vec<GroundedCondition>,
    pub base_cases: Vec<BaseCase>,
    pub base_states: Vec<State>,
    pub reduce_function: ReduceFunction,
    pub forward_transitions: Vec<Transition<T>>,
    pub forward_forced_transitions: Vec<Transition<T>>,
    pub backward_transitions: Vec<Transition<T>>,
    pub backward_forced_transitions: Vec<Transition<T>>,
}

impl<T: variable::Numeric> Model<T> {
    pub fn check_constraints<U: DPState>(&self, state: &U) -> bool {
        self.constraints.iter().all(|constraint| {
            constraint
                .is_satisfied(state, &self.table_registry)
                .unwrap_or(true)
        })
    }

    pub fn is_goal<U: DPState>(&self, state: &U) -> bool {
        self.base_cases
            .iter()
            .any(|case| case.is_satisfied(state, &self.table_registry))
            || self
                .base_states
                .iter()
                .any(|base| base.is_satisfied(state, &self.state_metadata))
    }
}

impl<T: variable::Numeric + parse_expression_from_yaml::ParesNumericExpressionFromYaml> Model<T> {
    pub fn load_from_yaml(domain: &Yaml, problem: &Yaml) -> Result<Model<T>, Box<dyn Error>>
    where
        <T as str::FromStr>::Err: fmt::Debug,
    {
        let domain = yaml_util::get_map(domain)?;
        let domain_name = yaml_util::get_string_by_key_or_default(domain, "domain", "")?;
        let problem = yaml_util::get_map(problem)?;
        let problem_name = yaml_util::get_string_by_key_or_default(problem, "problem", "")?;

        let variables = yaml_util::get_yaml_by_key(domain, "state_variables")?;
        let state_metadata = match (
            domain.get(&Yaml::from_str("objects")),
            problem.get(&Yaml::from_str("object_numbers")),
        ) {
            (Some(objects), Some(object_numbers)) => {
                state::StateMetadata::load_from_yaml(objects, variables, object_numbers)?
            }
            (None, None) => {
                let objects = yaml_rust::Yaml::Array(Vec::new());
                let object_numbers = yaml_rust::Yaml::Hash(linked_hash_map::LinkedHashMap::new());
                state::StateMetadata::load_from_yaml(&objects, variables, &object_numbers)?
            }
            (Some(_), None) => {
                return Err(ModelErr::new(String::from(
                    "key `object_numbers` not found while `objects` found ",
                ))
                .into())
            }
            (None, Some(_)) => {
                return Err(ModelErr::new(String::from(
                    "key `objects` not found while `object_numbers` found ",
                ))
                .into())
            }
        };

        let target = yaml_util::get_yaml_by_key(problem, "target")?;
        let target = state::State::load_from_yaml(target, &state_metadata)?;

        let table_registry = match (
            domain.get(&Yaml::from_str("tables")),
            problem.get(&Yaml::from_str("table_values")),
        ) {
            (Some(tables), Some(table_values)) => table_registry::TableRegistry::load_from_yaml(
                tables,
                table_values,
                &state_metadata,
            )?,
            (None, None) => TableRegistry {
                ..Default::default()
            },
            (Some(_), None) => {
                return Err(ModelErr::new(String::from(
                    "key `table_values` not found while `table` found ",
                ))
                .into())
            }
            (None, Some(_)) => {
                return Err(ModelErr::new(String::from(
                    "key `table` not found while `table_values` found ",
                ))
                .into())
            }
        };

        let mut constraints = Vec::new();
        if let Some(value) = domain.get(&Yaml::from_str("constraints")) {
            let array = yaml_util::get_array(value)?;
            let parameters = FxHashMap::default();
            for constraint in array {
                let conditions = GroundedCondition::load_grounded_conditions_from_yaml(
                    constraint,
                    &state_metadata,
                    &table_registry,
                    &parameters,
                )?;
                let conditions = Self::filter_constraints(conditions)?;
                constraints.extend(conditions);
            }
        }
        if let Some(value) = problem.get(&Yaml::from_str("constraints")) {
            let array = yaml_util::get_array(value)?;
            let parameters = FxHashMap::default();
            for constraint in array {
                let conditions = GroundedCondition::load_grounded_conditions_from_yaml(
                    constraint,
                    &state_metadata,
                    &table_registry,
                    &parameters,
                )?;
                let conditions = Self::filter_constraints(conditions)?;
                constraints.extend(conditions);
            }
        }

        let mut base_cases = Vec::new();
        if let Some(array) = domain.get(&yaml_rust::Yaml::from_str("base_cases")) {
            for base_case in yaml_util::get_array(array)? {
                let base_case =
                    BaseCase::load_from_yaml(base_case, &state_metadata, &table_registry)?;
                base_cases.push(base_case);
            }
        }
        if let Some(array) = problem.get(&yaml_rust::Yaml::from_str("base_cases")) {
            for base_case in yaml_util::get_array(array)? {
                let base_case =
                    BaseCase::load_from_yaml(base_case, &state_metadata, &table_registry)?;
                base_cases.push(base_case);
            }
        }
        let mut base_states = Vec::new();
        if let Some(array) = problem.get(&yaml_rust::Yaml::from_str("base_states")) {
            for base_state in yaml_util::get_array(array)? {
                let base_state = state::State::load_from_yaml(base_state, &state_metadata)?;
                base_states.push(base_state);
            }
        }
        if base_cases.is_empty() && base_states.is_empty() {
            return Err(ModelErr::new(String::from("no base case or condition")).into());
        }

        let reduce_function = yaml_util::get_yaml_by_key(domain, "reduce")?;
        let reduce_function = ReduceFunction::load_from_yaml(reduce_function)?;

        let mut forward_transitions = Vec::new();
        let mut forward_forced_transitions = Vec::new();
        let mut backward_transitions = Vec::new();
        let mut backward_forced_transitions = Vec::new();
        if let Some(array) = domain.get(&yaml_rust::Yaml::from_str("transitions")) {
            for transition in yaml_util::get_array(array)? {
                let (transition, forced, backward) = transition::load_transitions_from_yaml(
                    transition,
                    &state_metadata,
                    &table_registry,
                )?;
                if forced {
                    if backward {
                        backward_forced_transitions.extend(transition)
                    } else {
                        forward_forced_transitions.extend(transition)
                    }
                } else if backward {
                    backward_transitions.extend(transition)
                } else {
                    forward_transitions.extend(transition)
                }
            }
        }
        if let Some(array) = problem.get(&yaml_rust::Yaml::from_str("transitions")) {
            for transition in yaml_util::get_array(array)? {
                let (transition, forced, backward) = transition::load_transitions_from_yaml(
                    transition,
                    &state_metadata,
                    &table_registry,
                )?;
                if forced {
                    if backward {
                        backward_forced_transitions.extend(transition)
                    } else {
                        forward_forced_transitions.extend(transition)
                    }
                } else if backward {
                    backward_transitions.extend(transition)
                } else {
                    forward_transitions.extend(transition)
                }
            }
        }
        if forward_transitions.is_empty() && backward_transitions.is_empty() {
            return Err(ModelErr::new(String::from("no transitions")).into());
        }

        Ok(Model {
            domain_name,
            problem_name,
            state_metadata,
            target,
            table_registry,
            constraints,
            base_cases,
            base_states,
            reduce_function,
            forward_transitions,
            forward_forced_transitions,
            backward_transitions,
            backward_forced_transitions,
        })
    }

    fn filter_constraints(
        conditions: Vec<grounded_condition::GroundedCondition>,
    ) -> Result<Vec<grounded_condition::GroundedCondition>, ModelErr> {
        let mut result = Vec::new();
        for condition in conditions {
            match condition.condition {
                expression::Condition::Constant(true) => continue,
                expression::Condition::Constant(false)
                    if condition.elements_in_set_variable.is_empty()
                        && condition.elements_in_vector_variable.is_empty() =>
                {
                    return Err(ModelErr::new(String::from(
                        "model has a constraint never satisfied",
                    )))
                }
                _ => result.push(condition),
            }
        }
        Ok(result)
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum CostWrappedModel {
    Integer(Model<variable::Integer>),
    OrderedContinuous(Model<variable::OrderedContinuous>),
}

impl CostWrappedModel {
    pub fn load_from_yaml(
        domain: &Yaml,
        problem: &Yaml,
    ) -> Result<CostWrappedModel, Box<dyn Error>> {
        if let Ok(model) = Model::<variable::Integer>::load_from_yaml(domain, problem) {
            Ok(CostWrappedModel::Integer(model))
        } else {
            let model = Model::<variable::OrderedContinuous>::load_from_yaml(domain, problem)?;
            Ok(CostWrappedModel::OrderedContinuous(model))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use expression::*;

    #[test]
    fn reduce_function_load_from_yaml_ok() {
        let yaml = r"min";
        let yaml = yaml_rust::YamlLoader::load_from_str(yaml);
        assert!(yaml.is_ok());
        let yaml = yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let reduce = ReduceFunction::load_from_yaml(yaml);
        assert!(reduce.is_ok());
        assert_eq!(reduce.unwrap(), ReduceFunction::Min);
        let yaml = r"max";
        let yaml = yaml_rust::YamlLoader::load_from_str(yaml);
        assert!(yaml.is_ok());
        let yaml = yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let reduce = ReduceFunction::load_from_yaml(yaml);
        assert!(reduce.is_ok());
        assert_eq!(reduce.unwrap(), ReduceFunction::Max);
        let yaml = r"sum";
        let yaml = yaml_rust::YamlLoader::load_from_str(yaml);
        assert!(yaml.is_ok());
        let yaml = yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let reduce = ReduceFunction::load_from_yaml(yaml);
        assert!(reduce.is_ok());
        assert_eq!(reduce.unwrap(), ReduceFunction::Sum);
        let yaml = r"product";
        let yaml = yaml_rust::YamlLoader::load_from_str(yaml);
        assert!(yaml.is_ok());
        let yaml = yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let reduce = ReduceFunction::load_from_yaml(yaml);
        assert!(reduce.is_ok());
        assert_eq!(reduce.unwrap(), ReduceFunction::Product);
    }

    #[test]
    fn reduce_function_load_from_yaml_err() {
        let yaml = r"or";
        let yaml = yaml_rust::YamlLoader::load_from_str(yaml);
        assert!(yaml.is_ok());
        let yaml = yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let reduce = ReduceFunction::load_from_yaml(yaml);
        assert!(reduce.is_err());
    }

    #[test]
    fn check_constraints() {
        let state = state::State::default();
        let model = Model::<variable::Integer> {
            constraints: vec![GroundedCondition {
                condition: Condition::Constant(true),
                ..Default::default()
            }],
            ..Default::default()
        };
        assert!(model.check_constraints(&state));
        let model = Model::<variable::Integer> {
            constraints: vec![
                GroundedCondition {
                    condition: Condition::Constant(true),
                    ..Default::default()
                },
                GroundedCondition {
                    condition: Condition::Constant(false),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        assert!(!model.check_constraints(&state));
    }

    #[test]
    fn is_goal() {
        let state = state::State::default();
        let model = Model::<variable::Integer> {
            base_cases: vec![BaseCase::new(vec![GroundedCondition {
                condition: Condition::Constant(true),
                ..Default::default()
            }])],
            ..Default::default()
        };
        assert!(model.is_goal(&state));
        let model = Model::<variable::Integer> {
            base_cases: vec![BaseCase::new(vec![GroundedCondition {
                condition: Condition::Constant(false),
                ..Default::default()
            }])],
            ..Default::default()
        };
        assert!(!model.is_goal(&state));
        let model = Model::<variable::Integer> {
            base_cases: vec![BaseCase::new(vec![GroundedCondition {
                condition: Condition::Constant(false),
                ..Default::default()
            }])],
            base_states: vec![state::State::default()],
            ..Default::default()
        };
        assert!(model.is_goal(&state));
    }

    #[test]
    fn model_load_from_yaml_ok() {
        let domain = r"
domain: ADD
state_variables: [ {name: v1, type: integer}, {name: v2, type: integer} ]
reduce: min
base_cases:
        - [(>= v1 1)]
constraints:
        - (>= v1 0)
transitions:
        - name: add
          effect:
                v1: (+ v1 1)
          cost: (+ cost 1)
        - name: recover
          preconditions: [(< v1 0)] 
          effect:
                v1: '0'
          cost: cost
          forced: true
";
        let domain = yaml_rust::YamlLoader::load_from_str(domain);
        assert!(domain.is_ok());
        let domain = domain.unwrap();
        assert_eq!(domain.len(), 1);
        let domain = &domain[0];

        let problem = r"
domain: ADD
problem: one
target:
        v1: 0
        v2: 0
base_cases:
        - [(>= v2 1), (= 0 0)]
constraints:
        - (>= v2 0)
transitions:
        - name: addv2
          effect:
                v2: (+ v2 1)
          cost: (+ cost 1)
";
        let problem = yaml_rust::YamlLoader::load_from_str(problem);
        assert!(problem.is_ok());
        let problem = problem.unwrap();
        assert_eq!(problem.len(), 1);
        let problem = &problem[0];

        let model = Model::<variable::Integer>::load_from_yaml(domain, problem);
        assert!(model.is_ok());
        let model = model.unwrap();

        let mut name_to_integer_variable = FxHashMap::default();
        name_to_integer_variable.insert(String::from("v1"), 0);
        name_to_integer_variable.insert(String::from("v2"), 1);
        let expected = Model {
            domain_name: String::from("ADD"),
            problem_name: String::from("one"),
            state_metadata: state::StateMetadata {
                integer_variable_names: vec![String::from("v1"), String::from("v2")],
                name_to_integer_variable,
                ..Default::default()
            },
            target: state::State {
                signature_variables: SignatureVariables {
                    integer_variables: vec![0, 0],
                    ..Default::default()
                },
                ..Default::default()
            },
            constraints: vec![
                GroundedCondition {
                    condition: Condition::Comparison(Box::new(Comparison::ComparisonII(
                        ComparisonOperator::Ge,
                        NumericExpression::IntegerVariable(0),
                        NumericExpression::Constant(0),
                    ))),
                    ..Default::default()
                },
                GroundedCondition {
                    condition: Condition::Comparison(Box::new(Comparison::ComparisonII(
                        ComparisonOperator::Ge,
                        NumericExpression::IntegerVariable(1),
                        NumericExpression::Constant(0),
                    ))),
                    ..Default::default()
                },
            ],
            base_cases: vec![
                BaseCase::new(vec![GroundedCondition {
                    condition: Condition::Comparison(Box::new(Comparison::ComparisonII(
                        ComparisonOperator::Ge,
                        NumericExpression::IntegerVariable(0),
                        NumericExpression::Constant(1),
                    ))),
                    ..Default::default()
                }]),
                BaseCase::new(vec![GroundedCondition {
                    condition: Condition::Comparison(Box::new(Comparison::ComparisonII(
                        ComparisonOperator::Ge,
                        NumericExpression::IntegerVariable(1),
                        NumericExpression::Constant(1),
                    ))),
                    ..Default::default()
                }]),
            ],
            forward_transitions: vec![
                Transition {
                    name: String::from("add"),
                    effect: effect::Effect {
                        integer_effects: vec![(
                            0,
                            NumericExpression::NumericOperation(
                                NumericOperator::Add,
                                Box::new(NumericExpression::IntegerVariable(0)),
                                Box::new(NumericExpression::Constant(1)),
                            ),
                        )],
                        ..Default::default()
                    },
                    cost: NumericExpression::NumericOperation(
                        NumericOperator::Add,
                        Box::new(NumericExpression::Cost),
                        Box::new(NumericExpression::Constant(1)),
                    ),
                    ..Default::default()
                },
                Transition {
                    name: String::from("addv2"),
                    effect: effect::Effect {
                        integer_effects: vec![(
                            1,
                            NumericExpression::NumericOperation(
                                NumericOperator::Add,
                                Box::new(NumericExpression::IntegerVariable(1)),
                                Box::new(NumericExpression::Constant(1)),
                            ),
                        )],
                        ..Default::default()
                    },
                    cost: NumericExpression::NumericOperation(
                        NumericOperator::Add,
                        Box::new(NumericExpression::Cost),
                        Box::new(NumericExpression::Constant(1)),
                    ),
                    ..Default::default()
                },
            ],
            forward_forced_transitions: vec![Transition {
                name: String::from("recover"),
                preconditions: vec![GroundedCondition {
                    condition: Condition::Comparison(Box::new(Comparison::ComparisonII(
                        ComparisonOperator::Lt,
                        NumericExpression::IntegerVariable(0),
                        NumericExpression::Constant(0),
                    ))),
                    ..Default::default()
                }],
                effect: effect::Effect {
                    integer_effects: vec![(0, NumericExpression::Constant(0))],
                    ..Default::default()
                },
                cost: NumericExpression::Cost,
                ..Default::default()
            }],
            ..Default::default()
        };

        assert_eq!(model.domain_name, expected.domain_name);
        assert_eq!(model.problem_name, expected.problem_name);
        assert_eq!(model.state_metadata, expected.state_metadata);
        assert_eq!(model.target, expected.target);
        assert_eq!(model.table_registry, expected.table_registry);
        assert_eq!(model.constraints, expected.constraints);
        assert_eq!(model.base_cases, expected.base_cases);
        assert_eq!(model.forward_transitions, expected.forward_transitions);
        assert_eq!(model.backward_transitions, expected.backward_transitions);
        assert_eq!(model, expected);

        let domain = r"
domain: Fibonacci 
state_variables: [ {name: v, type: integer} ]
reduce: sum
transitions:
        - name: one
          direction: backward
          effect:
                v: (+ v 1)
        - name: two
          direction: backward
          effect:
                v: (+ v 2)
";
        let domain = yaml_rust::YamlLoader::load_from_str(domain);
        assert!(domain.is_ok());
        let domain = domain.unwrap();
        assert_eq!(domain.len(), 1);
        let domain = &domain[0];

        let problem = r"
domain: Fibonacci
problem: Fibonacci10
target:
        v: 10
base_states:
        - { v: 0 }
        - { v: 1 }
";
        let problem = yaml_rust::YamlLoader::load_from_str(problem);
        assert!(problem.is_ok());
        let problem = problem.unwrap();
        assert_eq!(problem.len(), 1);
        let problem = &problem[0];

        let model = Model::<variable::Integer>::load_from_yaml(domain, problem);
        assert!(model.is_ok());
        let model = model.unwrap();

        let mut name_to_integer_variable = FxHashMap::default();
        name_to_integer_variable.insert(String::from("v"), 0);
        let expected = Model {
            domain_name: String::from("Fibonacci"),
            problem_name: String::from("Fibonacci10"),
            state_metadata: state::StateMetadata {
                integer_variable_names: vec![String::from("v")],
                name_to_integer_variable,
                ..Default::default()
            },
            target: state::State {
                signature_variables: SignatureVariables {
                    integer_variables: vec![10],
                    ..Default::default()
                },
                ..Default::default()
            },
            base_states: vec![
                state::State {
                    signature_variables: SignatureVariables {
                        integer_variables: vec![0],
                        ..Default::default()
                    },
                    ..Default::default()
                },
                state::State {
                    signature_variables: SignatureVariables {
                        integer_variables: vec![1],
                        ..Default::default()
                    },
                    ..Default::default()
                },
            ],
            reduce_function: ReduceFunction::Sum,
            backward_transitions: vec![
                Transition {
                    name: String::from("one"),
                    effect: Effect {
                        integer_effects: vec![(
                            0,
                            NumericExpression::NumericOperation(
                                NumericOperator::Add,
                                Box::new(NumericExpression::IntegerVariable(0)),
                                Box::new(NumericExpression::Constant(1)),
                            ),
                        )],
                        ..Default::default()
                    },
                    cost: NumericExpression::Cost,
                    ..Default::default()
                },
                Transition {
                    name: String::from("two"),
                    effect: Effect {
                        integer_effects: vec![(
                            0,
                            NumericExpression::NumericOperation(
                                NumericOperator::Add,
                                Box::new(NumericExpression::IntegerVariable(0)),
                                Box::new(NumericExpression::Constant(2)),
                            ),
                        )],
                        ..Default::default()
                    },
                    cost: NumericExpression::Cost,
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        assert_eq!(model.domain_name, expected.domain_name);
        assert_eq!(model.problem_name, expected.problem_name);
        assert_eq!(model.state_metadata, expected.state_metadata);
        assert_eq!(model.target, expected.target);
        assert_eq!(model.table_registry, expected.table_registry);
        assert_eq!(model.constraints, expected.constraints);
        assert_eq!(model.base_cases, expected.base_cases);
        assert_eq!(model.base_states, expected.base_states);
        assert_eq!(model.reduce_function, expected.reduce_function);
        assert_eq!(model.forward_transitions, expected.forward_transitions);
        assert_eq!(model.backward_transitions, expected.backward_transitions);
        assert_eq!(model, expected);

        let domain = r"
domain: TSPTW
reduce: min
objects: [cities]
state_variables:
        - name: unvisited
          type: set
          object: cities
        - name: location
          type: element
          object: cities
        - name: time
          type: integer
          preference: less
tables:
        - name: ready_time
          type: integer
          args: [cities]
        - name: due_date 
          type: integer
          args: [cities]
        - name: distance 
          type: integer
          args: [cities, cities]
          default: 0
        - name: connected
          type: bool
          args: [cities, cities]
          default: true
constraints:
        - (<= time (due_date location))
        - (= 0 0)
reduce: min
transitions:
        - name: visit
          direction: forward
          parameters: [{ name: to, object: unvisited }]
          preconditions: [(connected location to)]
          effect:
                unvisited: (remove to unvisited)
                location: to
                time: (max (+ time (distance location to)) (ready_time to))
          cost: (+ cost (distance location to))
";
        let domain = yaml_rust::YamlLoader::load_from_str(domain);
        assert!(domain.is_ok());
        let domain = domain.unwrap();
        assert_eq!(domain.len(), 1);
        let domain = &domain[0];

        let problem = r"
domain: TSPTW
problem: test
numeric_type: integer
object_numbers: { cities: 3 }
target:
        unvisited: [0, 1, 2]
        location: 0
        time: 0
base_cases:
        - [(is_empty unvisited), (= location 0)]
table_values:
        ready_time: {0: 0, 1: 1, 2: 1}
        due_date: {0: 10000, 1: 2, 2: 2}
        distance: {[0, 1]: 1, [0, 2]: 1, [1, 0]: 1, [1, 2]: 1, [2, 0]: 1, [2, 1]: 1}
        connected: {[0, 0]: false, [1, 1]: false, [2, 2]: false}
";
        let problem = yaml_rust::YamlLoader::load_from_str(problem);
        assert!(problem.is_ok());
        let problem = problem.unwrap();
        assert_eq!(problem.len(), 1);
        let problem = &problem[0];

        let model = Model::<variable::Integer>::load_from_yaml(domain, problem);
        assert!(model.is_ok());
        let model = model.unwrap();

        let mut name_to_object = FxHashMap::default();
        name_to_object.insert(String::from("cities"), 0);
        let mut name_to_set_variable = FxHashMap::default();
        name_to_set_variable.insert(String::from("unvisited"), 0);
        let mut name_to_element_variable = FxHashMap::default();
        name_to_element_variable.insert(String::from("location"), 0);
        let mut name_to_integer_resource_variable = FxHashMap::default();
        name_to_integer_resource_variable.insert(String::from("time"), 0);
        let mut unvisited = variable::Set::with_capacity(3);
        unvisited.insert(0);
        unvisited.insert(1);
        unvisited.insert(2);
        let mut name_to_table_1d = FxHashMap::default();
        name_to_table_1d.insert(String::from("ready_time"), 0);
        name_to_table_1d.insert(String::from("due_date"), 1);
        let mut numeric_name_to_table_2d = FxHashMap::default();
        numeric_name_to_table_2d.insert(String::from("distance"), 0);
        let mut bool_name_to_table_2d = FxHashMap::default();
        bool_name_to_table_2d.insert(String::from("connected"), 0);
        let expected = Model {
            domain_name: String::from("TSPTW"),
            problem_name: String::from("test"),
            state_metadata: state::StateMetadata {
                object_names: vec![String::from("cities")],
                name_to_object,
                object_numbers: vec![3],
                set_variable_names: vec![String::from("unvisited")],
                name_to_set_variable,
                set_variable_to_object: vec![0],
                element_variable_names: vec![String::from("location")],
                name_to_element_variable,
                element_variable_to_object: vec![0],
                integer_resource_variable_names: vec![String::from("time")],
                name_to_integer_resource_variable,
                integer_less_is_better: vec![true],
                ..Default::default()
            },
            target: state::State {
                signature_variables: state::SignatureVariables {
                    set_variables: vec![unvisited],
                    element_variables: vec![0],
                    ..Default::default()
                },
                resource_variables: state::ResourceVariables {
                    integer_variables: vec![0],
                    ..Default::default()
                },
            },
            table_registry: table_registry::TableRegistry {
                integer_tables: table_data::TableData {
                    tables_1d: vec![
                        table::Table1D::new(vec![0, 1, 1]),
                        table::Table1D::new(vec![10000, 2, 2]),
                    ],
                    name_to_table_1d,
                    tables_2d: vec![table::Table2D::new(vec![
                        vec![0, 1, 1],
                        vec![1, 0, 1],
                        vec![1, 1, 0],
                    ])],
                    name_to_table_2d: numeric_name_to_table_2d,
                    ..Default::default()
                },
                bool_tables: table_data::TableData {
                    tables_2d: vec![table::Table2D::new(vec![
                        vec![false, true, true],
                        vec![true, false, true],
                        vec![true, true, false],
                    ])],
                    name_to_table_2d: bool_name_to_table_2d,
                    ..Default::default()
                },
                ..Default::default()
            },
            constraints: vec![GroundedCondition {
                condition: Condition::Comparison(Box::new(Comparison::ComparisonII(
                    ComparisonOperator::Le,
                    NumericExpression::IntegerResourceVariable(0),
                    NumericExpression::IntegerTable(NumericTableExpression::Table1D(
                        1,
                        ElementExpression::Variable(0),
                    )),
                ))),
                ..Default::default()
            }],
            base_cases: vec![BaseCase::new(vec![
                GroundedCondition {
                    condition: Condition::Set(Box::new(SetCondition::IsEmpty(
                        SetExpression::Reference(ReferenceExpression::Variable(0)),
                    ))),
                    ..Default::default()
                },
                GroundedCondition {
                    condition: Condition::Comparison(Box::new(Comparison::ComparisonEE(
                        ComparisonOperator::Eq,
                        ElementExpression::Variable(0),
                        ElementExpression::Constant(0),
                    ))),
                    ..Default::default()
                },
            ])],
            base_states: Vec::new(),
            reduce_function: ReduceFunction::Min,
            forward_transitions: vec![
                Transition {
                    name: String::from("visit"),
                    parameter_names: vec![String::from("to")],
                    parameter_values: vec![0],
                    elements_in_set_variable: vec![(0, 0)],
                    preconditions: vec![GroundedCondition {
                        condition: Condition::Table(TableExpression::Table2D(
                            0,
                            ElementExpression::Variable(0),
                            ElementExpression::Constant(0),
                        )),
                        ..Default::default()
                    }],
                    effect: Effect {
                        set_effects: vec![(
                            0,
                            SetExpression::SetElementOperation(
                                SetElementOperator::Remove,
                                ElementExpression::Constant(0),
                                Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                                    0,
                                ))),
                            ),
                        )],
                        element_effects: vec![(0, ElementExpression::Constant(0))],
                        integer_resource_effects: vec![(
                            0,
                            NumericExpression::NumericOperation(
                                NumericOperator::Max,
                                Box::new(NumericExpression::NumericOperation(
                                    NumericOperator::Add,
                                    Box::new(NumericExpression::IntegerResourceVariable(0)),
                                    Box::new(NumericExpression::IntegerTable(
                                        NumericTableExpression::Table2D(
                                            0,
                                            ElementExpression::Variable(0),
                                            ElementExpression::Constant(0),
                                        ),
                                    )),
                                )),
                                Box::new(NumericExpression::Constant(0)),
                            ),
                        )],
                        ..Default::default()
                    },
                    cost: NumericExpression::NumericOperation(
                        NumericOperator::Add,
                        Box::new(NumericExpression::Cost),
                        Box::new(NumericExpression::IntegerTable(
                            NumericTableExpression::Table2D(
                                0,
                                ElementExpression::Variable(0),
                                ElementExpression::Constant(0),
                            ),
                        )),
                    ),
                    ..Default::default()
                },
                Transition {
                    name: String::from("visit"),
                    parameter_names: vec![String::from("to")],
                    parameter_values: vec![1],
                    elements_in_set_variable: vec![(0, 1)],
                    preconditions: vec![GroundedCondition {
                        condition: Condition::Table(TableExpression::Table2D(
                            0,
                            ElementExpression::Variable(0),
                            ElementExpression::Constant(1),
                        )),
                        ..Default::default()
                    }],
                    effect: Effect {
                        set_effects: vec![(
                            0,
                            SetExpression::SetElementOperation(
                                SetElementOperator::Remove,
                                ElementExpression::Constant(1),
                                Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                                    0,
                                ))),
                            ),
                        )],
                        element_effects: vec![(0, ElementExpression::Constant(1))],
                        integer_resource_effects: vec![(
                            0,
                            NumericExpression::NumericOperation(
                                NumericOperator::Max,
                                Box::new(NumericExpression::NumericOperation(
                                    NumericOperator::Add,
                                    Box::new(NumericExpression::IntegerResourceVariable(0)),
                                    Box::new(NumericExpression::IntegerTable(
                                        NumericTableExpression::Table2D(
                                            0,
                                            ElementExpression::Variable(0),
                                            ElementExpression::Constant(1),
                                        ),
                                    )),
                                )),
                                Box::new(NumericExpression::Constant(1)),
                            ),
                        )],
                        ..Default::default()
                    },
                    cost: NumericExpression::NumericOperation(
                        NumericOperator::Add,
                        Box::new(NumericExpression::Cost),
                        Box::new(NumericExpression::IntegerTable(
                            NumericTableExpression::Table2D(
                                0,
                                ElementExpression::Variable(0),
                                ElementExpression::Constant(1),
                            ),
                        )),
                    ),
                    ..Default::default()
                },
                Transition {
                    name: String::from("visit"),
                    parameter_names: vec![String::from("to")],
                    parameter_values: vec![2],
                    elements_in_set_variable: vec![(0, 2)],
                    preconditions: vec![GroundedCondition {
                        condition: Condition::Table(TableExpression::Table2D(
                            0,
                            ElementExpression::Variable(0),
                            ElementExpression::Constant(2),
                        )),
                        ..Default::default()
                    }],
                    effect: effect::Effect {
                        set_effects: vec![(
                            0,
                            SetExpression::SetElementOperation(
                                SetElementOperator::Remove,
                                ElementExpression::Constant(2),
                                Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                                    0,
                                ))),
                            ),
                        )],
                        element_effects: vec![(0, ElementExpression::Constant(2))],
                        integer_resource_effects: vec![(
                            0,
                            NumericExpression::NumericOperation(
                                NumericOperator::Max,
                                Box::new(NumericExpression::NumericOperation(
                                    NumericOperator::Add,
                                    Box::new(NumericExpression::IntegerResourceVariable(0)),
                                    Box::new(NumericExpression::IntegerTable(
                                        NumericTableExpression::Table2D(
                                            0,
                                            ElementExpression::Variable(0),
                                            ElementExpression::Constant(2),
                                        ),
                                    )),
                                )),
                                Box::new(NumericExpression::Constant(1)),
                            ),
                        )],
                        ..Default::default()
                    },
                    cost: NumericExpression::NumericOperation(
                        NumericOperator::Add,
                        Box::new(NumericExpression::Cost),
                        Box::new(NumericExpression::IntegerTable(
                            NumericTableExpression::Table2D(
                                0,
                                ElementExpression::Variable(0),
                                ElementExpression::Constant(2),
                            ),
                        )),
                    ),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        assert_eq!(model.domain_name, expected.domain_name);
        assert_eq!(model.problem_name, expected.problem_name);
        assert_eq!(model.state_metadata, expected.state_metadata);
        assert_eq!(model.target, expected.target);
        assert_eq!(model.table_registry, expected.table_registry);
        assert_eq!(model.constraints, expected.constraints);
        assert_eq!(model.base_cases, expected.base_cases);
        assert_eq!(model.forward_transitions, expected.forward_transitions);
        assert_eq!(model.backward_transitions, expected.backward_transitions);
        assert_eq!(model, expected);
    }

    #[test]
    fn model_load_from_yaml_err() {
        let problem = r"
domain: TSPTW
problem: test
numeric_type: integer
object_numbers: { cities: 3 }
target:
        unvisited: [0, 1, 2]
        location: 0
        time: 0
base_cases:
        - condition: (is_empty unvisited)
        - condition: (is location 0)
table_values:
        ready_time: {0: 0, 1: 1, 2: 1}
        due_date: {0: 10000, 1: 2, 2: 2}
        distance: {[0, 1]: 1, [0, 2]: 1, [1, 0]: 1, [1, 2]: 1, [2, 0]: 1, [2, 1]: 1}
        connected: {[0, 0]: false, [1, 1]: false, [2, 2]: false}
";
        let problem = yaml_rust::YamlLoader::load_from_str(problem);
        assert!(problem.is_ok());
        let problem = problem.unwrap();
        assert_eq!(problem.len(), 1);
        let problem = &problem[0];

        let domain = r"
reduce: min
objects: [cities]
variables:
        - name: unvisited
          type: set
          object: cities
        - name: location
          type: element
          object: cities
        - name: time
          type: integer
          preference: less
tables:
        - name: ready_time
          type: integer
          args: [cities]
        - name: due_date 
          type: integer
          args: [cities]
        - name: distance 
          type: integer
          args: [cities, cities]
          default: 0
        - name: connected
          type: bool
          args: [cities, cities]
          default: true
constraints:
        - condition: (<= time (due_date location))
reduce: min
transitions:
        - name: visit
          parameters: [{ name: to, object: unvisited }]
          preconditions: [(connected location to)]
          effect:
                unvisited: (remove to unvisited)
                location: to
                time: (max (+ time (distance location to)) (ready_time to))
          cost: (+ cost (distance location to))
";
        let domain = yaml_rust::YamlLoader::load_from_str(domain);
        assert!(domain.is_ok());
        let domain = domain.unwrap();
        assert_eq!(domain.len(), 1);
        let domain = &domain[0];

        let model = Model::<variable::Integer>::load_from_yaml(domain, problem);
        assert!(model.is_err());

        let domain = r"
domain: TSPTW
reduce: min
state_variables:
        - name: unvisited
          type: set
          object: cities
        - name: location
          type: element
          object: cities
        - name: time
          type: integer
          preference: less
tables:
        - name: ready_time
          type: integer
          args: [cities]
        - name: due_date 
          type: integer
          args: [cities]
        - name: distance 
          type: integer
          args: [cities, cities]
          default: 0
        - name: connected
          type: bool
          args: [cities, cities]
          default: true
constraints:
        - condition: (<= time (due_date location))
reduce: min
transitions:
        - name: visit
          parameters: [{ name: to, object: unvisited }]
          preconditions: [(connected location to)]
          effect:
                unvisited: (remove to unvisited)
                location: to
                time: (max (+ time (distance location to)) (ready_time to))
          cost: (+ cost (distance location to))
";
        let domain = yaml_rust::YamlLoader::load_from_str(domain);
        assert!(domain.is_ok());
        let domain = domain.unwrap();
        assert_eq!(domain.len(), 1);
        let domain = &domain[0];

        let model = Model::<variable::Integer>::load_from_yaml(domain, problem);
        assert!(model.is_err());

        let domain = r"
domain: TSPTW
reduce: min
objects: [null]
tables:
        - name: ready_time
          type: integer
          args: [cities]
        - name: due_date 
          type: integer
          args: [cities]
        - name: distance 
          type: integer
          args: [cities, cities]
          default: 0
        - name: connected
          type: bool
          args: [cities, cities]
          default: true
constraints:
        - condition: (<= time (due_date location))
reduce: min
transitions:
        - name: visit
          parameters: [{ name: to, object: unvisited }]
          preconditions: [(connected location to)]
          effect:
                unvisited: (remove to unvisited)
                location: to
                time: (max (+ time (distance location to)) (ready_time to))
          cost: (+ cost (distance location to))
";
        let domain = yaml_rust::YamlLoader::load_from_str(domain);
        assert!(domain.is_ok());
        let domain = domain.unwrap();
        assert_eq!(domain.len(), 1);
        let domain = &domain[0];

        let model = Model::<variable::Integer>::load_from_yaml(domain, problem);
        assert!(model.is_err());

        let domain = r"
domain: TSPTW
reduce: min
objects: [cities]
state_variables:
        - name: unvisited
          type: set
          object: cities
        - name: location
          type: element
          object: cities
        - name: time
          type: integer
          preference: less
constraints:
        - condition: (<= time (due_date location))
reduce: min
transitions:
        - name: visit
          parameters: [{ name: to, object: unvisited }]
          preconditions: [(connected location to)]
          effect:
                unvisited: (remove to unvisited)
                location: to
                time: (max (+ time (distance location to)) (ready_time to))
          cost: (+ cost (distance location to))
";
        let domain = yaml_rust::YamlLoader::load_from_str(domain);
        assert!(domain.is_ok());
        let domain = domain.unwrap();
        assert_eq!(domain.len(), 1);
        let domain = &domain[0];

        let model = Model::<variable::Integer>::load_from_yaml(domain, problem);
        assert!(model.is_err());

        let domain = r"
reduce: min
objects: [cities]
state_variables:
        - name: unvisited
          type: set
          object: cities
        - name: location
          type: element
          object: cities
        - name: time
          type: integer
          preference: less
tables:
        - name: ready_time
          type: integer
          args: [cities]
        - name: due_date 
          type: integer
          args: [cities]
        - name: distance 
          type: integer
          args: [cities, cities]
          default: 0
        - name: connected
          type: bool
          args: [cities, cities]
          default: true
constraints:
        - condition: (<= time (due_date location))
";
        let domain = yaml_rust::YamlLoader::load_from_str(domain);
        assert!(domain.is_ok());
        let domain = domain.unwrap();
        assert_eq!(domain.len(), 1);
        let domain = &domain[0];

        let model = Model::<variable::Integer>::load_from_yaml(domain, problem);
        assert!(model.is_err());

        let domain = r"
domain: TSPTW
reduce: min
objects: [cities]
state_variables:
        - name: unvisited
          type: set
          object: cities
        - name: location
          type: element
          object: cities
        - name: time
          type: integer
          preference: less
tables:
        - name: ready_time
          type: integer
          args: [cities]
        - name: due_date 
          type: integer
          args: [cities]
        - name: distance 
          type: integer
          args: [cities, cities]
          default: 0
        - name: connected
          type: bool
          args: [cities, cities]
          default: true
constraints:
        - condition: (<= time (due_date location))
        - condition: (= 1 2)
reduce: min
transitions:
        - name: visit
          parameters: [{ name: to, object: unvisited }]
          preconditions: [(connected location to)]
          effect:
                unvisited: (remove to unvisited)
                location: to
                time: (max (+ time (distance location to)) (ready_time to))
          cost: (+ cost (distance location to))
";
        let domain = yaml_rust::YamlLoader::load_from_str(domain);
        assert!(domain.is_ok());
        let domain = domain.unwrap();
        assert_eq!(domain.len(), 1);
        let domain = &domain[0];

        let model = Model::<variable::Integer>::load_from_yaml(domain, problem);
        assert!(model.is_err());

        let problem = r"
problem: test
numeric_type: integer
object_numbers: { cities: 3 }
target:
        unvisited: [0, 1, 2]
        location: 0
        time: 0
base_cases:
        - [(is_empty unvisited), (is location 0)]
table_values:
        ready_time: {0: 0, 1: 1, 2: 1}
        due_date: {0: 10000, 1: 2, 2: 2}
        distance: {[0, 1]: 1, [0, 2]: 1, [1, 0]: 1, [1, 2]: 1, [2, 0]: 1, [2, 1]: 1}
        connected: {[0, 0]: false, [1, 1]: false, [2, 2]: false}
";
        let problem = yaml_rust::YamlLoader::load_from_str(problem);
        assert!(problem.is_ok());
        let problem = problem.unwrap();
        assert_eq!(problem.len(), 1);
        let problem = &problem[0];

        let model = Model::<variable::Integer>::load_from_yaml(domain, problem);
        assert!(model.is_err());

        let problem = r"
domain: TSP
problem: test
numeric_type: integer
object_numbers: { cities: 3 }
target:
        unvisited: [0, 1, 2]
        location: 0
        time: 0
base_cases:
        - [(is_empty unvisited) (is location 0)]
table_values:
        ready_time: {0: 0, 1: 1, 2: 1}
        due_date: {0: 10000, 1: 2, 2: 2}
        distance: {[0, 1]: 1, [0, 2]: 1, [1, 0]: 1, [1, 2]: 1, [2, 0]: 1, [2, 1]: 1}
        connected: {[0, 0]: false, [1, 1]: false, [2, 2]: false}
";
        let problem = yaml_rust::YamlLoader::load_from_str(problem);
        assert!(problem.is_ok());
        let problem = problem.unwrap();
        assert_eq!(problem.len(), 1);
        let problem = &problem[0];

        let model = Model::<variable::Integer>::load_from_yaml(domain, problem);
        assert!(model.is_err());

        let problem = r"
domain: TSPTW
problem: test
numeric_type: integer
target:
        unvisited: [0, 1, 2]
        location: 0
        time: 0
base_cases:
        - [(is_empty unvisited), (is location 0)]
table_values:
        ready_time: {0: 0, 1: 1, 2: 1}
        due_date: {0: 10000, 1: 2, 2: 2}
        distance: {[0, 1]: 1, [0, 2]: 1, [1, 0]: 1, [1, 2]: 1, [2, 0]: 1, [2, 1]: 1}
        connected: {[0, 0]: false, [1, 1]: false, [2, 2]: false}
";
        let problem = yaml_rust::YamlLoader::load_from_str(problem);
        assert!(problem.is_ok());
        let problem = problem.unwrap();
        assert_eq!(problem.len(), 1);
        let problem = &problem[0];

        let model = Model::<variable::Integer>::load_from_yaml(domain, problem);
        assert!(model.is_err());

        let problem = r"
domain: TSPTW
problem: test
numeric_type: integer
object_numbers: { cities: 3 }
base_cases:
        - [(is_empty unvisited), (is location 0)]
table_values:
        ready_time: {0: 0, 1: 1, 2: 1}
        due_date: {0: 10000, 1: 2, 2: 2}
        distance: {[0, 1]: 1, [0, 2]: 1, [1, 0]: 1, [1, 2]: 1, [2, 0]: 1, [2, 1]: 1}
        connected: {[0, 0]: false, [1, 1]: false, [2, 2]: false}
";
        let problem = yaml_rust::YamlLoader::load_from_str(problem);
        assert!(problem.is_ok());
        let problem = problem.unwrap();
        assert_eq!(problem.len(), 1);
        let problem = &problem[0];

        let model = Model::<variable::Integer>::load_from_yaml(domain, problem);
        assert!(model.is_err());

        let problem = r"
domain: TSPTW
problem: test
numeric_type: integer
object_numbers: { cities: 3 }
target:
        unvisited: [0, 1, 2]
        location: 0
        time: 0
table_values:
        ready_time: {0: 0, 1: 1, 2: 1}
        due_date: {0: 10000, 1: 2, 2: 2}
        distance: {[0, 1]: 1, [0, 2]: 1, [1, 0]: 1, [1, 2]: 1, [2, 0]: 1, [2, 1]: 1}
        connected: {[0, 0]: false, [1, 1]: false, [2, 2]: false}
";
        let problem = yaml_rust::YamlLoader::load_from_str(problem);
        assert!(problem.is_ok());
        let problem = problem.unwrap();
        assert_eq!(problem.len(), 1);
        let problem = &problem[0];

        let model = Model::<variable::Integer>::load_from_yaml(domain, problem);
        assert!(model.is_err());

        let problem = r"
domain: TSPTW
problem: test
numeric_type: integer
object_numbers: { cities: 3 }
target:
        unvisited: [0, 1, 2]
        location: 0
        time: 0
base_cases:
        - [(is_empty unvisited), (is location 0)]
";
        let problem = yaml_rust::YamlLoader::load_from_str(problem);
        assert!(problem.is_ok());
        let problem = problem.unwrap();
        assert_eq!(problem.len(), 1);
        let problem = &problem[0];

        let model = Model::<variable::Integer>::load_from_yaml(domain, problem);
        assert!(model.is_err());

        let problem = r"
domain: TSPTW
problem: test
numeric_type: integer
object_numbers: { cities: 3 }
target:
        unvisited: [0, 1, 2]
        location: 0
        time: 0
base_cases:
        - [(is_empty unvisited), (is location 0), (!= 0 0)]
table_values:
        ready_time: {0: 0, 1: 1, 2: 1}
        due_date: {0: 10000, 1: 2, 2: 2}
        distance: {[0, 1]: 1, [0, 2]: 1, [1, 0]: 1, [1, 2]: 1, [2, 0]: 1, [2, 1]: 1}
        connected: {[0, 0]: false, [1, 1]: false, [2, 2]: false}
";
        let problem = yaml_rust::YamlLoader::load_from_str(problem);
        assert!(problem.is_ok());
        let problem = problem.unwrap();
        assert_eq!(problem.len(), 1);
        let problem = &problem[0];

        let model = Model::<variable::Integer>::load_from_yaml(domain, problem);
        assert!(model.is_err());
    }
    #[test]
    fn cost_wrapped_model_load_from_yaml_ok() {
        let domain = r"
domain: ADD
state_variables: [ {name: v, type: integer} ]
reduce: min
transitions:
        - name: add
          effect:
                v: (+ v 1)
          cost: (+ cost 1)
";
        let domain = yaml_rust::YamlLoader::load_from_str(domain);
        assert!(domain.is_ok());
        let domain = domain.unwrap();
        assert_eq!(domain.len(), 1);
        let domain = &domain[0];

        let problem = r"
domain: ADD
problem: one
target:
        v: 0
base_cases:
        - [(>= v 1), (= 0 0)]
";
        let problem = yaml_rust::YamlLoader::load_from_str(problem);
        assert!(problem.is_ok());
        let problem = problem.unwrap();
        assert_eq!(problem.len(), 1);
        let problem = &problem[0];

        let model = CostWrappedModel::load_from_yaml(domain, problem);
        assert!(model.is_ok());
        let model = model.unwrap();

        let mut name_to_integer_variable = FxHashMap::default();
        name_to_integer_variable.insert(String::from("v"), 0);
        let expected = CostWrappedModel::Integer(Model {
            domain_name: String::from("ADD"),
            problem_name: String::from("one"),
            state_metadata: state::StateMetadata {
                integer_variable_names: vec![String::from("v")],
                name_to_integer_variable,
                ..Default::default()
            },
            target: state::State {
                signature_variables: SignatureVariables {
                    integer_variables: vec![0],
                    ..Default::default()
                },
                ..Default::default()
            },
            base_cases: vec![BaseCase::new(vec![GroundedCondition {
                condition: Condition::Comparison(Box::new(Comparison::ComparisonII(
                    ComparisonOperator::Ge,
                    NumericExpression::IntegerVariable(0),
                    NumericExpression::Constant(1),
                ))),
                ..Default::default()
            }])],
            forward_transitions: vec![Transition {
                name: String::from("add"),
                effect: effect::Effect {
                    integer_effects: vec![(
                        0,
                        NumericExpression::NumericOperation(
                            NumericOperator::Add,
                            Box::new(NumericExpression::IntegerVariable(0)),
                            Box::new(NumericExpression::Constant(1)),
                        ),
                    )],
                    ..Default::default()
                },
                cost: NumericExpression::NumericOperation(
                    NumericOperator::Add,
                    Box::new(NumericExpression::Cost),
                    Box::new(NumericExpression::Constant(1)),
                ),
                ..Default::default()
            }],
            ..Default::default()
        });
        assert_eq!(model, expected);

        let domain = r"
domain: ADD
state_variables: [ {name: v, type: integer} ]
reduce: min
transitions:
        - name: add
          effect:
                v: (+ v 1)
          cost: (+ cost 1.0)
";
        let domain = yaml_rust::YamlLoader::load_from_str(domain);
        assert!(domain.is_ok());
        let domain = domain.unwrap();
        assert_eq!(domain.len(), 1);
        let domain = &domain[0];

        let problem = r"
domain: ADD
problem: one
target:
        v: 0
base_cases:
        - [(>= v 1), (= 0 0)]
";
        let problem = yaml_rust::YamlLoader::load_from_str(problem);
        assert!(problem.is_ok());
        let problem = problem.unwrap();
        assert_eq!(problem.len(), 1);
        let problem = &problem[0];

        let model = CostWrappedModel::load_from_yaml(domain, problem);
        assert!(model.is_ok());
        let model = model.unwrap();

        let mut name_to_integer_variable = FxHashMap::default();
        name_to_integer_variable.insert(String::from("v"), 0);
        let expected = CostWrappedModel::OrderedContinuous(Model {
            domain_name: String::from("ADD"),
            problem_name: String::from("one"),
            state_metadata: state::StateMetadata {
                integer_variable_names: vec![String::from("v")],
                name_to_integer_variable,
                ..Default::default()
            },
            target: state::State {
                signature_variables: SignatureVariables {
                    integer_variables: vec![0],
                    ..Default::default()
                },
                ..Default::default()
            },
            base_cases: vec![BaseCase::new(vec![GroundedCondition {
                condition: Condition::Comparison(Box::new(Comparison::ComparisonII(
                    ComparisonOperator::Ge,
                    NumericExpression::IntegerVariable(0),
                    NumericExpression::Constant(1),
                ))),
                ..Default::default()
            }])],
            forward_transitions: vec![Transition {
                name: String::from("add"),
                effect: effect::Effect {
                    integer_effects: vec![(
                        0,
                        NumericExpression::NumericOperation(
                            NumericOperator::Add,
                            Box::new(NumericExpression::IntegerVariable(0)),
                            Box::new(NumericExpression::Constant(1)),
                        ),
                    )],
                    ..Default::default()
                },
                cost: NumericExpression::NumericOperation(
                    NumericOperator::Add,
                    Box::new(NumericExpression::Cost),
                    Box::new(NumericExpression::Constant(ordered_float::OrderedFloat(
                        1.0,
                    ))),
                ),
                ..Default::default()
            }],
            ..Default::default()
        });
        assert_eq!(model, expected);
    }
}
