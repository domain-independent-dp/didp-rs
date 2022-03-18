use std::error::Error;
use std::fmt;
use std::str;
use yaml_rust::Yaml;

pub mod expression;
pub mod expression_parser;
mod grounded_condition;
mod operator;
mod state;
pub mod table;
mod table_registry;
pub mod variable;
mod yaml_util;

pub use grounded_condition::GroundedCondition;
pub use operator::Operator;
pub use state::{SignatureVariables, State, StateMetadata};
pub use table_registry::{TableData, TableRegistry};

#[derive(Debug)]
pub struct ProblemErr(String);

impl ProblemErr {
    pub fn new(message: String) -> ProblemErr {
        ProblemErr(format!("Error in problem definiton: {}", message))
    }
}

impl fmt::Display for ProblemErr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Error for ProblemErr {}

pub enum NumericType {
    Integer,
    Continuous,
}

impl NumericType {
    pub fn load_from_yaml(value: &Yaml) -> Result<NumericType, Box<dyn Error>> {
        let map = yaml_util::get_map(value)?;
        let numeric_type = yaml_util::get_string_by_key(&map, "numeric_type")?;
        match &numeric_type[..] {
            "integer" => Ok(Self::Integer),
            "continuous" => Ok(Self::Continuous),
            _ => Err(ProblemErr::new(format!("no such numeric type `{}`", numeric_type)).into()),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Default)]
pub struct Problem<T: variable::Numeric> {
    pub minimize: bool,
    pub domain_name: String,
    pub problem_name: String,
    pub state_metadata: state::StateMetadata,
    pub initial_state: state::State<T>,
    pub table_registry: table_registry::TableRegistry<T>,
    pub constraints: Vec<grounded_condition::GroundedCondition<T>>,
    pub goals: Vec<grounded_condition::GroundedCondition<T>>,
    pub operators: Vec<operator::Operator<T>>,
}

impl<T: variable::Numeric> Problem<T> {
    pub fn load_from_yaml(domain: &Yaml, problem: &Yaml) -> Result<Problem<T>, Box<dyn Error>>
    where
        <T as str::FromStr>::Err: fmt::Debug,
    {
        let domain = yaml_util::get_map(&domain)?;
        let minimize = match yaml_util::get_string_by_key(&domain, "metric") {
            Ok(value) => match &value[..] {
                "minimize" => true,
                "maximize" => false,
                _ => {
                    return Err(yaml_util::YamlContentErr::new(format!(
                        "expected `minimize` or `maximize`, but is {}",
                        value
                    ))
                    .into())
                }
            },
            Err(_) => true,
        };
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

        let objects = yaml_util::get_yaml_by_key(&domain, "objects")?;
        let variables = yaml_util::get_yaml_by_key(&domain, "variables")?;
        let object_numbers = yaml_util::get_yaml_by_key(&problem, "object_numbers")?;
        let state_metadata =
            state::StateMetadata::load_from_yaml(objects, variables, object_numbers)?;

        let initial_state = yaml_util::get_yaml_by_key(&problem, "initial_state")?;
        let initial_state = state::State::<T>::load_from_yaml(initial_state, &state_metadata)?;

        let table_registry = match (
            domain.get(&Yaml::from_str("tables")),
            problem.get(&Yaml::from_str("table_values")),
        ) {
            (Some(tables), Some(table_values)) => {
                table_registry::TableRegistry::<T>::load_from_yaml(
                    &tables,
                    &table_values,
                    &state_metadata,
                )?
            }
            (None, None) => TableRegistry {
                ..Default::default()
            },
            (None, Some(_)) => {
                return Err(ProblemErr::new(String::from(
                    "key `table_values` not found while `table` found ",
                ))
                .into())
            }
            (Some(_), None) => {
                return Err(ProblemErr::new(String::from(
                    "key `table` not found while `table_values` found ",
                ))
                .into())
            }
        };

        let mut constraints = Vec::new();
        if let Some(value) = domain.get(&Yaml::from_str("constraints")) {
            let array = yaml_util::get_array(value)?;
            for constraint in array {
                let conditions = grounded_condition::load_grounded_conditions_from_yaml(
                    &constraint,
                    &state_metadata,
                    &table_registry,
                )?;
                let conditions = Self::filiter_grounded_conditions(conditions)?;
                constraints.extend(conditions);
            }
        }

        let mut goals = Vec::new();
        for goal in yaml_util::get_array_by_key(&problem, "goals")? {
            let conditions = grounded_condition::load_grounded_conditions_from_yaml(
                &goal,
                &state_metadata,
                &table_registry,
            )?;
            let conditions = Self::filiter_grounded_conditions(conditions)?;
            goals.extend(conditions);
        }

        let mut operators = Vec::new();
        for operator in yaml_util::get_array_by_key(&domain, "operators")? {
            operators.extend(operator::load_operators_from_yaml(
                &operator,
                &state_metadata,
                &table_registry,
            )?);
        }

        Ok(Problem {
            minimize,
            domain_name,
            problem_name,
            state_metadata,
            initial_state,
            table_registry,
            constraints,
            goals,
            operators,
        })
    }

    fn filiter_grounded_conditions(
        conditions: Vec<grounded_condition::GroundedCondition<T>>,
    ) -> Result<Vec<grounded_condition::GroundedCondition<T>>, ProblemErr> {
        let mut result = Vec::with_capacity(conditions.len());
        for condition in conditions {
            match condition.condition {
                expression::Condition::Constant(true) => continue,
                expression::Condition::Constant(false) => {
                    return Err(ProblemErr::new(String::from(
                        "problem has a condition never satisfied",
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
    fn numeric_type_load_from_yaml_ok() {
        let yaml = r"numeric_type: integer";
        let yaml = yaml_rust::YamlLoader::load_from_str(yaml);
        assert!(yaml.is_ok());
        let yaml = yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let numeric_type = NumericType::load_from_yaml(yaml);
        assert!(numeric_type.is_ok());
        assert!(matches!(numeric_type.unwrap(), NumericType::Integer));
        let yaml = r"numeric_type: continuous";
        let yaml = yaml_rust::YamlLoader::load_from_str(yaml);
        assert!(yaml.is_ok());
        let yaml = yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let numeric_type = NumericType::load_from_yaml(yaml);
        assert!(numeric_type.is_ok());
        assert!(matches!(numeric_type.unwrap(), NumericType::Continuous));
    }

    #[test]
    fn numeric_type_load_from_yaml_err() {
        let yaml = r"type: integer";
        let yaml = yaml_rust::YamlLoader::load_from_str(yaml);
        assert!(yaml.is_ok());
        let yaml = yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let numeric_type = NumericType::load_from_yaml(yaml);
        assert!(numeric_type.is_err());
        let yaml = r"numeric_type: bool";
        let yaml = yaml_rust::YamlLoader::load_from_str(yaml);
        assert!(yaml.is_ok());
        let yaml = yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let numeric_type = NumericType::load_from_yaml(yaml);
        assert!(numeric_type.is_err());
    }

    #[test]
    fn problem_load_from_yaml_ok() {
        let domain = r"
domain: TSPTW
objects: [cities]
variables:
        - name: unvisited
          type: set
          object: cities
        - name: location
          type: element
          object: cities
        - name: time
          type: resource
          less_is_better: true
tables:
        - name: ready_time
          type: numeric
          args: [cities]
        - name: due_date 
          type: numeric
          args: [cities]
        - name: distance 
          type: numeric
          args: [cities, cities]
          default: 0
        - name: connected
          type: bool
          args: [cities, cities]
          default: true
constraints:
        - condition: (<= time (due_date location))
operators:
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

        let problem = r"
domain: TSPTW
problem: test
numeric_type: integer
object_numbers: { cities: 3 }
initial_state:
        unvisited: [0, 1, 2]
        location: 0
        time: 0
goals:
        - condition: (is_empty unvisited)
        - condition: (is location 0)
        - condition: (= 0 0)
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

        let problem = Problem::<variable::IntegerVariable>::load_from_yaml(domain, problem);
        assert!(problem.is_ok());
        let problem = problem.unwrap();

        let mut name_to_object = HashMap::new();
        name_to_object.insert(String::from("cities"), 0);
        let mut name_to_set_variable = HashMap::new();
        name_to_set_variable.insert(String::from("unvisited"), 0);
        let mut name_to_element_variable = HashMap::new();
        name_to_element_variable.insert(String::from("location"), 0);
        let mut name_to_resource_variable = HashMap::new();
        name_to_resource_variable.insert(String::from("time"), 0);
        let mut unvisited = variable::SetVariable::with_capacity(3);
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
        let expected = Problem {
            minimize: true,
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
                resource_variable_names: vec![String::from("time")],
                name_to_resource_variable,
                less_is_better: vec![true],
                ..Default::default()
            },
            initial_state: state::State {
                signature_variables: Rc::new(state::SignatureVariables {
                    set_variables: vec![unvisited],
                    element_variables: vec![0],
                    ..Default::default()
                }),
                resource_variables: vec![0],
                stage: 0,
                cost: 0,
            },
            table_registry: table_registry::TableRegistry {
                numeric_tables: table_registry::TableData {
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
            },
            constraints: vec![GroundedCondition {
                condition: Condition::Comparison(
                    ComparisonOperator::Le,
                    NumericExpression::ResourceVariable(0),
                    NumericExpression::Table(NumericTableExpression::Table1D(
                        1,
                        ElementExpression::Variable(0),
                    )),
                ),
                ..Default::default()
            }],
            goals: vec![
                GroundedCondition {
                    condition: Condition::Set(SetCondition::IsEmpty(SetExpression::SetVariable(0))),
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
            operators: vec![
                Operator {
                    name: String::from("visit to:0"),
                    elements_in_set_variable: vec![(0, 0)],
                    preconditions: vec![Condition::Table(BoolTableExpression::Table2D(
                        0,
                        ElementExpression::Variable(0),
                        ElementExpression::Constant(0),
                    ))],
                    set_effects: vec![(
                        0,
                        SetExpression::SetElementOperation(
                            SetElementOperator::Remove,
                            Box::new(SetExpression::SetVariable(0)),
                            ElementExpression::Constant(0),
                        ),
                    )],
                    element_effects: vec![(0, ElementExpression::Constant(0))],
                    resource_effects: vec![(
                        0,
                        NumericExpression::NumericOperation(
                            NumericOperator::Max,
                            Box::new(NumericExpression::NumericOperation(
                                NumericOperator::Add,
                                Box::new(NumericExpression::ResourceVariable(0)),
                                Box::new(NumericExpression::Table(
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
                        Box::new(NumericExpression::Table(NumericTableExpression::Table2D(
                            0,
                            ElementExpression::Variable(0),
                            ElementExpression::Constant(0),
                        ))),
                    ),
                    ..Default::default()
                },
                Operator {
                    name: String::from("visit to:1"),
                    elements_in_set_variable: vec![(0, 1)],
                    preconditions: vec![Condition::Table(BoolTableExpression::Table2D(
                        0,
                        ElementExpression::Variable(0),
                        ElementExpression::Constant(1),
                    ))],
                    set_effects: vec![(
                        0,
                        SetExpression::SetElementOperation(
                            SetElementOperator::Remove,
                            Box::new(SetExpression::SetVariable(0)),
                            ElementExpression::Constant(1),
                        ),
                    )],
                    element_effects: vec![(0, ElementExpression::Constant(1))],
                    resource_effects: vec![(
                        0,
                        NumericExpression::NumericOperation(
                            NumericOperator::Max,
                            Box::new(NumericExpression::NumericOperation(
                                NumericOperator::Add,
                                Box::new(NumericExpression::ResourceVariable(0)),
                                Box::new(NumericExpression::Table(
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
                        Box::new(NumericExpression::Table(NumericTableExpression::Table2D(
                            0,
                            ElementExpression::Variable(0),
                            ElementExpression::Constant(1),
                        ))),
                    ),
                    ..Default::default()
                },
                Operator {
                    name: String::from("visit to:2"),
                    elements_in_set_variable: vec![(0, 2)],
                    preconditions: vec![Condition::Table(BoolTableExpression::Table2D(
                        0,
                        ElementExpression::Variable(0),
                        ElementExpression::Constant(2),
                    ))],
                    set_effects: vec![(
                        0,
                        SetExpression::SetElementOperation(
                            SetElementOperator::Remove,
                            Box::new(SetExpression::SetVariable(0)),
                            ElementExpression::Constant(2),
                        ),
                    )],
                    element_effects: vec![(0, ElementExpression::Constant(2))],
                    resource_effects: vec![(
                        0,
                        NumericExpression::NumericOperation(
                            NumericOperator::Max,
                            Box::new(NumericExpression::NumericOperation(
                                NumericOperator::Add,
                                Box::new(NumericExpression::ResourceVariable(0)),
                                Box::new(NumericExpression::Table(
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
                        Box::new(NumericExpression::Table(NumericTableExpression::Table2D(
                            0,
                            ElementExpression::Variable(0),
                            ElementExpression::Constant(2),
                        ))),
                    ),
                    ..Default::default()
                },
            ],
        };
        assert_eq!(problem.minimize, expected.minimize);
        assert_eq!(problem.domain_name, expected.domain_name);
        assert_eq!(problem.problem_name, expected.problem_name);
        assert_eq!(problem.state_metadata, expected.state_metadata);
        assert_eq!(problem.initial_state, expected.initial_state);
        assert_eq!(problem.table_registry, expected.table_registry);
        assert_eq!(problem.constraints, expected.constraints);
        assert_eq!(problem.goals, expected.goals);
        assert_eq!(problem.operators[0], expected.operators[0]);
        assert_eq!(problem.operators[1], expected.operators[1]);
        assert_eq!(problem.operators[2], expected.operators[2]);
        assert_eq!(problem.operators, expected.operators);
        assert_eq!(problem, expected);
    }

    #[test]
    fn problem_load_from_yaml_err() {
        let problem = r"
domain: TSPTW
problem: test
numeric_type: integer
object_numbers: { cities: 3 }
initial_state:
        unvisited: [0, 1, 2]
        location: 0
        time: 0
goals:
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
        let problem_yaml = &problem[0];

        let domain = r"
metric: minimize
objects: [cities]
variables:
        - name: unvisited
          type: set
          object: cities
        - name: location
          type: element
          object: cities
        - name: time
          type: resource
          less_is_better: true
tables:
        - name: ready_time
          type: numeric
          args: [cities]
        - name: due_date 
          type: numeric
          args: [cities]
        - name: distance 
          type: numeric
          args: [cities, cities]
          default: 0
        - name: connected
          type: bool
          args: [cities, cities]
          default: true
constraints:
        - condition: (<= time (due_date location))
operators:
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

        let problem = Problem::<variable::IntegerVariable>::load_from_yaml(domain, problem_yaml);
        assert!(problem.is_err());

        let domain = r"
domain: TSPTW
metric: minimize
variables:
        - name: unvisited
          type: set
          object: cities
        - name: location
          type: element
          object: cities
        - name: time
          type: resource
          less_is_better: true
tables:
        - name: ready_time
          type: numeric
          args: [cities]
        - name: due_date 
          type: numeric
          args: [cities]
        - name: distance 
          type: numeric
          args: [cities, cities]
          default: 0
        - name: connected
          type: bool
          args: [cities, cities]
          default: true
constraints:
        - condition: (<= time (due_date location))
operators:
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

        let problem = Problem::<variable::IntegerVariable>::load_from_yaml(domain, problem_yaml);
        assert!(problem.is_err());

        let domain = r"
domain: TSPTW
metric: minimize 
objects: [null]
tables:
        - name: ready_time
          type: numeric
          args: [cities]
        - name: due_date 
          type: numeric
          args: [cities]
        - name: distance 
          type: numeric
          args: [cities, cities]
          default: 0
        - name: connected
          type: bool
          args: [cities, cities]
          default: true
constraints:
        - condition: (<= time (due_date location))
operators:
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

        let problem = Problem::<variable::IntegerVariable>::load_from_yaml(domain, problem_yaml);
        assert!(problem.is_err());

        let domain = r"
domain: TSPTW
metric: minimize
objects: [cities]
variables:
        - name: unvisited
          type: set
          object: cities
        - name: location
          type: element
          object: cities
        - name: time
          type: resource
          less_is_better: true
constraints:
        - condition: (<= time (due_date location))
operators:
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

        let problem = Problem::<variable::IntegerVariable>::load_from_yaml(domain, problem_yaml);
        assert!(problem.is_err());

        let domain = r"
metric: minimize
objects: [cities]
variables:
        - name: unvisited
          type: set
          object: cities
        - name: location
          type: element
          object: cities
        - name: time
          type: resource
          less_is_better: true
tables:
        - name: ready_time
          type: numeric
          args: [cities]
        - name: due_date 
          type: numeric
          args: [cities]
        - name: distance 
          type: numeric
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

        let problem = Problem::<variable::IntegerVariable>::load_from_yaml(domain, problem_yaml);
        assert!(problem.is_err());

        let domain = r"
domain: TSPTW
metric: minimize
objects: [cities]
variables:
        - name: unvisited
          type: set
          object: cities
        - name: location
          type: element
          object: cities
        - name: time
          type: resource
          less_is_better: true
tables:
        - name: ready_time
          type: numeric
          args: [cities]
        - name: due_date 
          type: numeric
          args: [cities]
        - name: distance 
          type: numeric
          args: [cities, cities]
          default: 0
        - name: connected
          type: bool
          args: [cities, cities]
          default: true
constraints:
        - condition: (<= time (due_date location))
        - condition: (= 1 2)
operators:
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

        let problem = Problem::<variable::IntegerVariable>::load_from_yaml(domain, problem_yaml);
        assert!(problem.is_err());

        let problem = r"
problem: test
numeric_type: integer
object_numbers: { cities: 3 }
initial_state:
        unvisited: [0, 1, 2]
        location: 0
        time: 0
goals:
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
        let problem_yaml = &problem[0];

        let problem = Problem::<variable::IntegerVariable>::load_from_yaml(domain, problem_yaml);
        assert!(problem.is_err());

        let problem = r"
domain: TSP
problem: test
numeric_type: integer
object_numbers: { cities: 3 }
initial_state:
        unvisited: [0, 1, 2]
        location: 0
        time: 0
goals:
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
        let problem_yaml = &problem[0];

        let problem = Problem::<variable::IntegerVariable>::load_from_yaml(domain, problem_yaml);
        assert!(problem.is_err());

        let problem = r"
domain: TSPTW
problem: test
numeric_type: integer
initial_state:
        unvisited: [0, 1, 2]
        location: 0
        time: 0
goals:
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
        let problem_yaml = &problem[0];

        let problem = Problem::<variable::IntegerVariable>::load_from_yaml(domain, problem_yaml);
        assert!(problem.is_err());

        let problem = r"
domain: TSPTW
problem: test
numeric_type: integer
object_numbers: { cities: 3 }
goals:
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
        let problem_yaml = &problem[0];

        let problem = Problem::<variable::IntegerVariable>::load_from_yaml(domain, problem_yaml);
        assert!(problem.is_err());

        let problem = r"
domain: TSPTW
problem: test
numeric_type: integer
object_numbers: { cities: 3 }
initial_state:
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
        let problem_yaml = &problem[0];

        let problem = Problem::<variable::IntegerVariable>::load_from_yaml(domain, problem_yaml);
        assert!(problem.is_err());

        let problem = r"
domain: TSPTW
problem: test
numeric_type: integer
object_numbers: { cities: 3 }
initial_state:
        unvisited: [0, 1, 2]
        location: 0
        time: 0
goals:
        - condition: (is_empty unvisited)
        - condition: (is location 0)
";
        let problem = yaml_rust::YamlLoader::load_from_str(problem);
        assert!(problem.is_ok());
        let problem = problem.unwrap();
        assert_eq!(problem.len(), 1);
        let problem_yaml = &problem[0];

        let problem = Problem::<variable::IntegerVariable>::load_from_yaml(domain, problem_yaml);
        assert!(problem.is_err());

        let problem = r"
domain: TSPTW
problem: test
numeric_type: integer
object_numbers: { cities: 3 }
initial_state:
        unvisited: [0, 1, 2]
        location: 0
        time: 0
goals:
        - condition: (is_empty unvisited)
        - condition: (is location 0)
        - condition: (!= 0 0)
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

        let problem = Problem::<variable::IntegerVariable>::load_from_yaml(domain, problem);
        assert!(problem.is_err());
    }
}
