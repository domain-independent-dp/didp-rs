use std::collections;
use std::error::Error;
use std::fmt;
use std::str;
use yaml_rust::Yaml;

mod base_case;
mod base_state;
pub mod expression;
pub mod expression_parser;
mod grounded_condition;
mod state;
pub mod table;
mod table_registry;
mod transition;
mod util;
pub mod variable;
mod yaml_util;

pub use base_case::BaseCase;
pub use base_state::BaseState;
pub use expression_parser::ParseErr;
pub use grounded_condition::GroundedCondition;
pub use state::{ResourceVariables, SignatureVariables, State, StateMetadata};
pub use table_registry::{TableData, TableRegistry};
pub use transition::Transition;
pub use util::ModelErr;

#[derive(Debug, PartialEq)]
pub enum CostType {
    Integer,
    Continuous,
}

impl CostType {
    pub fn load_from_yaml(value: &Yaml) -> Result<CostType, Box<dyn Error>> {
        let map = yaml_util::get_map(value)?;
        let numeric_type = yaml_util::get_string_by_key(&map, "cost_type")?;
        match &numeric_type[..] {
            "integer" => Ok(Self::Integer),
            "continuous" => Ok(Self::Continuous),
            _ => Err(yaml_util::YamlContentErr::new(format!(
                "no such numeric type `{}`",
                numeric_type
            ))
            .into()),
        }
    }
}

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
    pub base_cases: Vec<BaseCase<T>>,
    pub base_states: Vec<BaseState<T>>,
    pub reduce_function: ReduceFunction,
    pub forward_transitions: Vec<Transition<T>>,
    pub backward_transitions: Vec<Transition<T>>,
}

impl<T: variable::Numeric> Model<T> {
    pub fn check_constraints(&self, state: &state::State) -> bool {
        self.constraints.iter().all(|constraint| {
            constraint
                .is_satisfied(state, &self.state_metadata, &self.table_registry)
                .unwrap_or(true)
        })
    }

    pub fn get_base_cost(&self, state: &state::State) -> Option<T> {
        for base_state in &self.base_states {
            let cost = base_state.get_cost(state);
            if cost.is_some() {
                return cost;
            }
        }
        for base_case in &self.base_cases {
            let cost = base_case.get_cost(state, &self.state_metadata, &self.table_registry);
            if cost.is_some() {
                return cost;
            }
        }
        None
    }

    pub fn load_from_yaml(domain: &Yaml, problem: &Yaml) -> Result<Model<T>, Box<dyn Error>>
    where
        <T as str::FromStr>::Err: fmt::Debug,
    {
        let domain = yaml_util::get_map(&domain)?;
        let domain_name = yaml_util::get_string_by_key(&domain, "domain")?;
        let problem = yaml_util::get_map(&problem)?;
        let domain_name2 = yaml_util::get_string_by_key(&problem, "domain")?;
        if domain_name != domain_name2 {
            return Err(yaml_util::YamlContentErr::new(format!(
                "domain mismatch: expected `{}`, but is `{}`",
                domain_name, domain_name2
            ))
            .into());
        }
        let problem_name = yaml_util::get_string_by_key(&problem, "problem")?;

        let variables = yaml_util::get_yaml_by_key(&domain, "variables")?;
        let state_metadata = match (
            domain.get(&Yaml::from_str("objects")),
            problem.get(&Yaml::from_str("object_numbers")),
        ) {
            (Some(objects), Some(object_numbers)) => {
                state::StateMetadata::load_from_yaml(&objects, variables, object_numbers)?
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

        let target = yaml_util::get_yaml_by_key(&problem, "target")?;
        let target = state::State::load_from_yaml(target, &state_metadata)?;

        let table_registry = match (
            domain.get(&Yaml::from_str("tables")),
            problem.get(&Yaml::from_str("table_values")),
        ) {
            (Some(tables), Some(table_values)) => table_registry::TableRegistry::load_from_yaml(
                &tables,
                &table_values,
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
            let parameters = collections::HashMap::new();
            for constraint in array {
                let conditions = GroundedCondition::load_grounded_conditions_from_yaml(
                    &constraint,
                    &state_metadata,
                    &table_registry,
                    &parameters,
                )?;
                let conditions = Self::filter_constraints(conditions)?;
                constraints.extend(conditions);
            }
        }

        let mut base_cases = Vec::new();
        if let Some(array) = problem.get(&yaml_rust::Yaml::from_str("base_cases")) {
            for base_case in yaml_util::get_array(array)? {
                let base_case =
                    BaseCase::load_from_yaml(&base_case, &state_metadata, &table_registry)?;
                base_cases.push(base_case);
            }
        }
        let mut base_states = Vec::new();
        if let Some(array) = problem.get(&yaml_rust::Yaml::from_str("base_states")) {
            for base_state in yaml_util::get_array(array)? {
                let base_state = BaseState::load_from_yaml(&base_state, &state_metadata)?;
                base_states.push(base_state);
            }
        }
        if base_cases.is_empty() && base_states.is_empty() {
            return Err(ModelErr::new(String::from("no base case or condition")).into());
        }

        let reduce_function = yaml_util::get_yaml_by_key(&domain, "reduce")?;
        let reduce_function = ReduceFunction::load_from_yaml(reduce_function)?;

        let mut forward_transitions = Vec::new();
        let mut backward_transitions = Vec::new();
        for transition in yaml_util::get_array_by_key(&domain, "transitions")? {
            let (transition, backward) = transition::load_transitions_from_yaml(
                &transition,
                &state_metadata,
                &table_registry,
            )?;
            if backward {
                backward_transitions.extend(transition)
            } else {
                forward_transitions.extend(transition)
            }
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
            backward_transitions,
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
                        && condition.elements_in_permutation_variable.is_empty() =>
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

#[cfg(test)]
mod tests {
    use super::*;
    use expression::*;
    use std::collections::HashMap;
    use std::rc::Rc;

    #[test]
    fn cost_type_load_from_yaml_ok() {
        let yaml = r"cost_type: integer";
        let yaml = yaml_rust::YamlLoader::load_from_str(yaml);
        assert!(yaml.is_ok());
        let yaml = yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let numeric_type = CostType::load_from_yaml(yaml);
        assert!(numeric_type.is_ok());
        assert_eq!(numeric_type.unwrap(), CostType::Integer);
        let yaml = r"cost_type: continuous";
        let yaml = yaml_rust::YamlLoader::load_from_str(yaml);
        assert!(yaml.is_ok());
        let yaml = yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let numeric_type = CostType::load_from_yaml(yaml);
        assert!(numeric_type.is_ok());
        assert_eq!(numeric_type.unwrap(), CostType::Continuous);
    }

    #[test]
    fn cost_type_load_from_yaml_err() {
        let yaml = r"type: integer";
        let yaml = yaml_rust::YamlLoader::load_from_str(yaml);
        assert!(yaml.is_ok());
        let yaml = yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let numeric_type = CostType::load_from_yaml(yaml);
        assert!(numeric_type.is_err());
        let yaml = r"cost_type: bool";
        let yaml = yaml_rust::YamlLoader::load_from_str(yaml);
        assert!(yaml.is_ok());
        let yaml = yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let numeric_type = CostType::load_from_yaml(yaml);
        assert!(numeric_type.is_err());
    }

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
        let reduce = CostType::load_from_yaml(yaml);
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
    fn get_base_cost() {
        let state = state::State::default();
        let model = Model {
            base_cases: vec![BaseCase {
                conditions: vec![GroundedCondition {
                    condition: Condition::Constant(true),
                    ..Default::default()
                }],
                cost: NumericExpression::Constant(0),
            }],
            ..Default::default()
        };
        assert_eq!(model.get_base_cost(&state), Some(0));
        let model = Model {
            base_cases: vec![BaseCase {
                conditions: vec![GroundedCondition {
                    condition: Condition::Constant(false),
                    ..Default::default()
                }],
                cost: NumericExpression::Constant(0),
            }],
            ..Default::default()
        };
        assert_eq!(model.get_base_cost(&state), None);
        let model = Model {
            base_cases: vec![BaseCase {
                conditions: vec![GroundedCondition {
                    condition: Condition::Constant(false),
                    ..Default::default()
                }],
                cost: NumericExpression::Constant(0),
            }],
            base_states: vec![BaseState {
                state: state::State::default(),
                cost: 1,
            }],
            ..Default::default()
        };
        assert_eq!(model.get_base_cost(&state), Some(1));
    }

    #[test]
    fn model_load_from_yaml_ok() {
        let domain = r"
domain: ADD
variables: [ {name: v, type: integer} ]
reduce: min
transitions:
        - name: add
          effects:
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

        let model = Model::<variable::Integer>::load_from_yaml(domain, problem);
        assert!(model.is_ok());
        let model = model.unwrap();

        let mut name_to_integer_variable = HashMap::new();
        name_to_integer_variable.insert(String::from("v"), 0);
        let expected = Model {
            domain_name: String::from("ADD"),
            problem_name: String::from("one"),
            state_metadata: state::StateMetadata {
                integer_variable_names: vec![String::from("v")],
                name_to_integer_variable,
                ..Default::default()
            },
            target: state::State {
                signature_variables: Rc::new(SignatureVariables {
                    integer_variables: vec![0],
                    ..Default::default()
                }),
                stage: 0,
                ..Default::default()
            },
            base_cases: vec![BaseCase {
                cost: NumericExpression::Constant(0),
                conditions: vec![GroundedCondition {
                    condition: Condition::Comparison(Box::new(Comparison::ComparisonII(
                        ComparisonOperator::Ge,
                        NumericExpression::IntegerVariable(0),
                        NumericExpression::Constant(1),
                    ))),
                    ..Default::default()
                }],
            }],
            forward_transitions: vec![Transition {
                name: String::from("add"),
                integer_effects: vec![(
                    0,
                    NumericExpression::NumericOperation(
                        NumericOperator::Add,
                        Box::new(NumericExpression::IntegerVariable(0)),
                        Box::new(NumericExpression::Constant(1)),
                    ),
                )],
                cost: NumericExpression::NumericOperation(
                    NumericOperator::Add,
                    Box::new(NumericExpression::Cost),
                    Box::new(NumericExpression::Constant(1)),
                ),
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
variables: [ {name: v, type: integer} ]
reduce: sum
transitions:
        - name: one
          direction: backward
          effects:
                v: (+ v 1)
          cost: cost
        - name: two
          direction: backward
          effects:
                v: (+ v 2)
          cost: cost
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
        - state: { v: 0 }
        - state: { v: 1 }
";
        let problem = yaml_rust::YamlLoader::load_from_str(problem);
        assert!(problem.is_ok());
        let problem = problem.unwrap();
        assert_eq!(problem.len(), 1);
        let problem = &problem[0];

        let model = Model::<variable::Integer>::load_from_yaml(domain, problem);
        assert!(model.is_ok());
        let model = model.unwrap();

        let mut name_to_integer_variable = HashMap::new();
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
                signature_variables: Rc::new(SignatureVariables {
                    integer_variables: vec![10],
                    ..Default::default()
                }),
                stage: 0,
                ..Default::default()
            },
            base_states: vec![
                BaseState {
                    state: state::State {
                        signature_variables: Rc::new(SignatureVariables {
                            integer_variables: vec![0],
                            ..Default::default()
                        }),
                        stage: 0,
                        ..Default::default()
                    },
                    cost: 0,
                },
                BaseState {
                    state: state::State {
                        signature_variables: Rc::new(SignatureVariables {
                            integer_variables: vec![1],
                            ..Default::default()
                        }),
                        stage: 0,
                        ..Default::default()
                    },
                    cost: 0,
                },
            ],
            reduce_function: ReduceFunction::Sum,
            backward_transitions: vec![
                Transition {
                    name: String::from("one"),
                    integer_effects: vec![(
                        0,
                        NumericExpression::NumericOperation(
                            NumericOperator::Add,
                            Box::new(NumericExpression::IntegerVariable(0)),
                            Box::new(NumericExpression::Constant(1)),
                        ),
                    )],
                    cost: NumericExpression::Cost,
                    ..Default::default()
                },
                Transition {
                    name: String::from("two"),
                    integer_effects: vec![(
                        0,
                        NumericExpression::NumericOperation(
                            NumericOperator::Add,
                            Box::new(NumericExpression::IntegerVariable(0)),
                            Box::new(NumericExpression::Constant(2)),
                        ),
                    )],
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
        - (<= time (due_date location))
        - (= 0 0)
reduce: min
transitions:
        - name: visit
          direction: forward
          parameters: [{ name: to, object: unvisited }]
          preconditions: [(connected location to)]
          effects:
                unvisited: (- unvisited to)
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
        assert!(model.is_ok());
        let model = model.unwrap();

        let mut name_to_object = HashMap::new();
        name_to_object.insert(String::from("cities"), 0);
        let mut name_to_set_variable = HashMap::new();
        name_to_set_variable.insert(String::from("unvisited"), 0);
        let mut name_to_element_variable = HashMap::new();
        name_to_element_variable.insert(String::from("location"), 0);
        let mut name_to_integer_resource_variable = HashMap::new();
        name_to_integer_resource_variable.insert(String::from("time"), 0);
        let mut unvisited = variable::Set::with_capacity(3);
        unvisited.insert(0);
        unvisited.insert(1);
        unvisited.insert(2);
        let mut name_to_table_1d = HashMap::new();
        name_to_table_1d.insert(String::from("ready_time"), 0);
        name_to_table_1d.insert(String::from("due_date"), 1);
        let mut numeric_name_to_table_2d = HashMap::new();
        numeric_name_to_table_2d.insert(String::from("distance"), 0);
        let mut bool_name_to_table_2d = HashMap::new();
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
                signature_variables: Rc::new(state::SignatureVariables {
                    set_variables: vec![unvisited],
                    element_variables: vec![0],
                    ..Default::default()
                }),
                resource_variables: state::ResourceVariables {
                    integer_variables: vec![0],
                    ..Default::default()
                },
                stage: 0,
            },
            table_registry: table_registry::TableRegistry {
                integer_tables: table_registry::TableData {
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
                bool_tables: table_registry::TableData {
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
            base_cases: vec![BaseCase {
                cost: NumericExpression::Constant(0),
                conditions: vec![
                    GroundedCondition {
                        condition: Condition::Set(SetCondition::IsEmpty(
                            SetExpression::SetVariable(0),
                        )),
                        ..Default::default()
                    },
                    GroundedCondition {
                        condition: Condition::Set(SetCondition::Eq(
                            ElementExpression::Variable(0),
                            ElementExpression::Constant(0),
                        )),
                        ..Default::default()
                    },
                ],
            }],
            base_states: Vec::new(),
            reduce_function: ReduceFunction::Min,
            forward_transitions: vec![
                Transition {
                    name: String::from("visit to:0"),
                    elements_in_set_variable: vec![(0, 0)],
                    preconditions: vec![GroundedCondition {
                        condition: Condition::Table(BoolTableExpression::Table2D(
                            0,
                            ElementExpression::Variable(0),
                            ElementExpression::Constant(0),
                        )),
                        ..Default::default()
                    }],
                    set_effects: vec![(
                        0,
                        SetExpression::SetElementOperation(
                            SetElementOperator::Remove,
                            Box::new(SetExpression::SetVariable(0)),
                            ElementExpression::Constant(0),
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
                    name: String::from("visit to:1"),
                    elements_in_set_variable: vec![(0, 1)],
                    preconditions: vec![GroundedCondition {
                        condition: Condition::Table(BoolTableExpression::Table2D(
                            0,
                            ElementExpression::Variable(0),
                            ElementExpression::Constant(1),
                        )),
                        ..Default::default()
                    }],
                    set_effects: vec![(
                        0,
                        SetExpression::SetElementOperation(
                            SetElementOperator::Remove,
                            Box::new(SetExpression::SetVariable(0)),
                            ElementExpression::Constant(1),
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
                    name: String::from("visit to:2"),
                    elements_in_set_variable: vec![(0, 2)],
                    preconditions: vec![GroundedCondition {
                        condition: Condition::Table(BoolTableExpression::Table2D(
                            0,
                            ElementExpression::Variable(0),
                            ElementExpression::Constant(2),
                        )),
                        ..Default::default()
                    }],
                    set_effects: vec![(
                        0,
                        SetExpression::SetElementOperation(
                            SetElementOperator::Remove,
                            Box::new(SetExpression::SetVariable(0)),
                            ElementExpression::Constant(2),
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
            backward_transitions: Vec::new(),
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
          effects:
                unvisited: (- unvisited to)
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
          effects:
                unvisited: (- unvisited to)
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
          effects:
                unvisited: (- unvisited to)
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
constraints:
        - condition: (<= time (due_date location))
reduce: min
transitions:
        - name: visit
          parameters: [{ name: to, object: unvisited }]
          preconditions: [(connected location to)]
          effects:
                unvisited: (- unvisited to)
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
        - condition: (= 1 2)
reduce: min
transitions:
        - name: visit
          parameters: [{ name: to, object: unvisited }]
          preconditions: [(connected location to)]
          effects:
                unvisited: (- unvisited to)
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
}
