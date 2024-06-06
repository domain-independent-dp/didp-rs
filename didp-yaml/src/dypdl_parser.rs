//! A module for a YAML based DyPDL parser.

mod base_case_parser;
pub mod expression_parser;
mod grounded_condition_parser;
mod parse_expression_from_yaml;
mod state_parser;
mod table_registry_parser;
mod transition_parser;

pub use base_case_parser::load_base_case_from_yaml;
pub use grounded_condition_parser::load_grounded_conditions_from_yaml;
pub use state_parser::load_state_from_yaml;
pub use table_registry_parser::load_table_registry_from_yaml;
pub use transition_parser::load_transitions_from_yaml;
use yaml_rust::yaml::{Array, Hash};

use crate::util;
use dypdl::expression::{
    Condition, ContinuousExpression, ElementExpression, IntegerExpression, SetExpression,
    VectorExpression,
};
use dypdl::variable_type::Element;
use dypdl::{CostType, GroundedCondition, Model, ModelErr, ReduceFunction};
use rustc_hash::FxHashMap;
use std::error::Error;
use yaml_rust::Yaml;

/// Returns an integer expression loaded from YAML.
///
/// `parameters` specify names and values of constants.
///
/// # Errors
///
/// If the format is invalid.
pub fn load_integer_expression_from_yaml(
    value: &Yaml,
    model: &Model,
    parameters: &FxHashMap<String, Element>,
) -> Result<IntegerExpression, Box<dyn Error>> {
    parse_expression_from_yaml::parse_integer_from_yaml(
        value,
        &model.state_metadata,
        &model.table_registry,
        parameters,
    )
}

/// Returns a continuous expression loaded from YAML.
///
/// `parameters` specify names and values of constants.
///
/// # Errors
///
/// If the format is invalid.
pub fn load_continuous_expression_from_yaml(
    value: &Yaml,
    model: &Model,
    parameters: &FxHashMap<String, Element>,
) -> Result<ContinuousExpression, Box<dyn Error>> {
    parse_expression_from_yaml::parse_continuous_from_yaml(
        value,
        &model.state_metadata,
        &model.table_registry,
        parameters,
    )
}

/// Returns an element expression loaded from YAML.
///
/// `parameters` specify names and values of constants.
///
/// # Errors
///
/// If the format is invalid.
pub fn load_element_expression_from_yaml(
    value: &Yaml,
    model: &Model,
    parameters: &FxHashMap<String, Element>,
) -> Result<ElementExpression, Box<dyn Error>> {
    parse_expression_from_yaml::parse_element_from_yaml(
        value,
        &model.state_metadata,
        &model.table_registry,
        parameters,
    )
}

/// Returns a set expression loaded from YAML.
///
/// `parameters` specify names and values of constants.
///
/// # Errors
///
/// If the format is invalid.
pub fn load_set_expression_from_yaml(
    value: &Yaml,
    model: &Model,
    parameters: &FxHashMap<String, Element>,
) -> Result<SetExpression, Box<dyn Error>> {
    parse_expression_from_yaml::parse_set_from_yaml(
        value,
        &model.state_metadata,
        &model.table_registry,
        parameters,
    )
}

/// Returns a vector expression loaded from YAML.
///
/// `parameters` specify names and values of constants.
///
/// # Errors
///
/// If the format is invalid.
pub fn load_vector_expression_from_yaml(
    value: &Yaml,
    model: &Model,
    parameters: &FxHashMap<String, Element>,
) -> Result<VectorExpression, Box<dyn Error>> {
    parse_expression_from_yaml::parse_vector_from_yaml(
        value,
        &model.state_metadata,
        &model.table_registry,
        parameters,
    )
}

/// Returns a DyPDL model loaded from YAML.
///
/// # Errors
///
/// If the format is invalid.
pub fn load_model_from_yaml(domain: &Yaml, problem: &Yaml) -> Result<Model, Box<dyn Error>> {
    let domain = util::get_map(domain)?;
    let problem = util::get_map(problem)?;

    let variables = util::get_yaml_by_key(domain, "state_variables")?;
    let state_metadata = match (
        domain.get(&Yaml::from_str("objects")),
        problem.get(&Yaml::from_str("object_numbers")),
    ) {
        (Some(objects), Some(object_numbers)) => {
            state_parser::load_metadata_from_yaml(objects, variables, object_numbers)?
        }
        (None, None) => {
            let objects = yaml_rust::Yaml::Array(Vec::new());
            let object_numbers = yaml_rust::Yaml::Hash(linked_hash_map::LinkedHashMap::new());
            state_parser::load_metadata_from_yaml(&objects, variables, &object_numbers)?
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

    let target = util::get_yaml_by_key(problem, "target")?;
    let target = load_state_from_yaml(target, &state_metadata)?;

    let empty_array = Yaml::Array(Array::new());
    let empty_hash = Yaml::Hash(Hash::new());
    let (tables, table_values) = match (
        domain.get(&Yaml::from_str("tables")),
        problem.get(&Yaml::from_str("table_values")),
    ) {
        (Some(tables), Some(table_values)) => (tables, table_values),
        (None, None) => (&empty_array, &empty_hash),
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

    let (dictionaries, dictionary_values) = match (
        domain.get(&Yaml::from_str("dictionaries")),
        problem.get(&Yaml::from_str("dictionary_values")),
    ) {
        (Some(dictionaries), Some(dictionary_values)) => (dictionaries, dictionary_values),
        (None, None) => (&empty_array, &empty_hash),
        (Some(_), None) => {
            return Err(ModelErr::new(String::from(
                "key `dictionary_values` not found while `dictionary` found ",
            ))
            .into())
        }
        (None, Some(_)) => {
            return Err(ModelErr::new(String::from(
                "key `dictionaries` not found while `dictionary_values` found ",
            ))
            .into())
        }
    };

    let table_registry = load_table_registry_from_yaml(
        tables,
        table_values,
        dictionaries,
        dictionary_values,
        &state_metadata,
    )?;

    let mut constraints = Vec::new();
    if let Some(value) = domain.get(&Yaml::from_str("constraints")) {
        let array = util::get_array(value)?;
        let parameters = FxHashMap::default();
        for constraint in array {
            let conditions = load_grounded_conditions_from_yaml(
                constraint,
                &state_metadata,
                &table_registry,
                &parameters,
            )?;
            let conditions = filter_constraints(conditions)?;
            constraints.extend(conditions);
        }
    }
    if let Some(value) = problem.get(&Yaml::from_str("constraints")) {
        let array = util::get_array(value)?;
        let parameters = FxHashMap::default();
        for constraint in array {
            let conditions = load_grounded_conditions_from_yaml(
                constraint,
                &state_metadata,
                &table_registry,
                &parameters,
            )?;
            let conditions = filter_constraints(conditions)?;
            constraints.extend(conditions);
        }
    }

    let cost_type = if let Ok(cost_type) = util::get_yaml_by_key(domain, "cost_type") {
        load_cost_type_from_yaml(cost_type)?
    } else {
        CostType::Integer
    };

    let mut base_cases = Vec::new();
    if let Some(array) = domain.get(&yaml_rust::Yaml::from_str("base_cases")) {
        for base_case in util::get_array(array)? {
            let base_case = base_case_parser::load_base_case_from_yaml(
                base_case,
                &state_metadata,
                &table_registry,
                &cost_type,
            )?;
            base_cases.push(base_case);
        }
    }
    if let Some(array) = problem.get(&yaml_rust::Yaml::from_str("base_cases")) {
        for base_case in util::get_array(array)? {
            let base_case = base_case_parser::load_base_case_from_yaml(
                base_case,
                &state_metadata,
                &table_registry,
                &cost_type,
            )?;
            base_cases.push(base_case);
        }
    }
    let mut base_states = Vec::new();
    if let Some(array) = problem.get(&yaml_rust::Yaml::from_str("base_states")) {
        for base_state in util::get_array(array)? {
            let base_state = base_case_parser::load_base_state_from_yaml(
                base_state,
                &state_metadata,
                &cost_type,
            )?;
            base_states.push(base_state);
        }
    }
    if base_cases.is_empty() && base_states.is_empty() {
        return Err(ModelErr::new(String::from("no base case or condition")).into());
    }

    let reduce_function = if let Ok(reduce_function) = util::get_yaml_by_key(domain, "reduce") {
        load_reduce_function_from_yaml(reduce_function)?
    } else {
        ReduceFunction::Min
    };

    let mut forward_transitions = Vec::new();
    let mut forward_forced_transitions = Vec::new();
    let mut backward_transitions = Vec::new();
    let mut backward_forced_transitions = Vec::new();
    if let Some(array) = domain.get(&yaml_rust::Yaml::from_str("transitions")) {
        for transition in util::get_array(array)? {
            let (transition, forced, backward) = load_transitions_from_yaml(
                transition,
                &state_metadata,
                &table_registry,
                &cost_type,
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
        for transition in util::get_array(array)? {
            let (transition, forced, backward) = load_transitions_from_yaml(
                transition,
                &state_metadata,
                &table_registry,
                &cost_type,
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
    let mut dual_bounds = vec![];
    if let Some(array) = domain.get(&yaml_rust::Yaml::from_str("dual_bounds")) {
        let parameters = FxHashMap::default();
        for bound in util::get_array(array)? {
            match cost_type {
                CostType::Integer => {
                    let expression = parse_expression_from_yaml::parse_integer_from_yaml(
                        bound,
                        &state_metadata,
                        &table_registry,
                        &parameters,
                    )?;
                    dual_bounds.push(dypdl::CostExpression::Integer(
                        expression.simplify(&table_registry),
                    ));
                }
                CostType::Continuous => {
                    let expression = parse_expression_from_yaml::parse_continuous_from_yaml(
                        bound,
                        &state_metadata,
                        &table_registry,
                        &parameters,
                    )?;
                    dual_bounds.push(dypdl::CostExpression::Continuous(
                        expression.simplify(&table_registry),
                    ));
                }
            }
        }
    }
    if let Some(array) = problem.get(&yaml_rust::Yaml::from_str("dual_bounds")) {
        let parameters = FxHashMap::default();
        for bound in util::get_array(array)? {
            match cost_type {
                CostType::Integer => {
                    let expression = parse_expression_from_yaml::parse_integer_from_yaml(
                        bound,
                        &state_metadata,
                        &table_registry,
                        &parameters,
                    )?;
                    dual_bounds.push(dypdl::CostExpression::Integer(
                        expression.simplify(&table_registry),
                    ));
                }
                CostType::Continuous => {
                    let expression = parse_expression_from_yaml::parse_continuous_from_yaml(
                        bound,
                        &state_metadata,
                        &table_registry,
                        &parameters,
                    )?;
                    dual_bounds.push(dypdl::CostExpression::Continuous(
                        expression.simplify(&table_registry),
                    ));
                }
            }
        }
    }

    Ok(Model {
        state_metadata,
        target,
        table_registry,
        state_constraints: constraints,
        base_cases,
        base_states,
        reduce_function,
        cost_type,
        forward_transitions,
        forward_forced_transitions,
        backward_transitions,
        backward_forced_transitions,
        dual_bounds,
    })
}

fn load_cost_type_from_yaml(value: &Yaml) -> Result<CostType, Box<dyn Error>> {
    let cost_type = util::get_string(value)?;
    match &cost_type[..] {
        "integer" => Ok(CostType::Integer),
        "continuous" => Ok(CostType::Continuous),
        _ => Err(util::YamlContentErr::new(format!("no such cost type `{}`", cost_type)).into()),
    }
}

fn load_reduce_function_from_yaml(value: &Yaml) -> Result<ReduceFunction, Box<dyn Error>> {
    let reduce_function = util::get_string(value)?;
    match &reduce_function[..] {
        "min" => Ok(ReduceFunction::Min),
        "max" => Ok(ReduceFunction::Max),
        "sum" => Ok(ReduceFunction::Sum),
        "product" => Ok(ReduceFunction::Product),
        _ => Err(util::YamlContentErr::new(format!(
            "no such reduce function `{}`",
            reduce_function
        ))
        .into()),
    }
}

fn filter_constraints(
    conditions: Vec<GroundedCondition>,
) -> Result<Vec<GroundedCondition>, ModelErr> {
    let mut result = Vec::new();
    for condition in conditions {
        match condition.condition {
            Condition::Constant(true) => continue,
            Condition::Constant(false)
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

#[cfg(test)]
mod tests {
    use super::*;
    use dypdl::expression::*;
    use dypdl::variable_type::*;
    use dypdl::Table;
    use dypdl::{
        BaseCase, CostExpression, Effect, ResourceVariables, SignatureVariables, State,
        StateMetadata, Table1D, Table2D, TableData, TableRegistry, Transition,
    };

    #[test]
    fn load_integer_expression_from_yaml_err() {
        let model = Model::default();
        let parameters = FxHashMap::default();

        let value = Yaml::String(String::from("(+ cost 1"));
        let result = load_integer_expression_from_yaml(&value, &model, &parameters);
        assert!(result.is_err());

        let value = Yaml::Real(String::from("1.2"));
        let result = load_integer_expression_from_yaml(&value, &model, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn load_continuous_expression_from_yaml_ok() {
        let model = Model::default();
        let parameters = FxHashMap::default();

        let value = Yaml::String(String::from("(+ cost 1)"));
        let result = load_continuous_expression_from_yaml(&value, &model, &parameters);
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::Cost),
                Box::new(ContinuousExpression::Constant(1.0))
            )
        );

        let value = Yaml::Integer(1);
        let result = load_continuous_expression_from_yaml(&value, &model, &parameters);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ContinuousExpression::Constant(1.0));

        let value = Yaml::Real(String::from("1.2"));
        let result = load_continuous_expression_from_yaml(&value, &model, &parameters);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ContinuousExpression::Constant(1.2));
    }

    #[test]
    fn load_continuous_expression_from_yaml_err() {
        let model = Model::default();
        let parameters = FxHashMap::default();

        let value = Yaml::String(String::from("(+ cost 1"));
        let result = load_continuous_expression_from_yaml(&value, &model, &parameters);
        assert!(result.is_err());

        let value = Yaml::Real(String::from("a"));
        let result = load_continuous_expression_from_yaml(&value, &model, &parameters);
        assert!(result.is_err());

        let value = Yaml::Boolean(true);
        let result = load_continuous_expression_from_yaml(&value, &model, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn load_element_expression_from_yaml_ok() {
        let model = Model::default();
        let parameters = FxHashMap::default();

        let value = Yaml::String(String::from("1"));
        let result = load_element_expression_from_yaml(&value, &model, &parameters);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ElementExpression::Constant(1));

        let value = Yaml::Integer(1);
        let result = load_element_expression_from_yaml(&value, &model, &parameters);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ElementExpression::Constant(1));
    }

    #[test]
    fn load_element_expression_from_yaml_err() {
        let model = Model::default();
        let parameters = FxHashMap::default();

        let value = Yaml::String(String::from("(+ cost 1"));
        let result = load_element_expression_from_yaml(&value, &model, &parameters);
        assert!(result.is_err());

        let value = Yaml::Real(String::from("1.2"));
        let result = load_element_expression_from_yaml(&value, &model, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn load_set_expression_from_yaml_ok() {
        let mut metadata = StateMetadata::default();
        let ob = metadata.add_object_type(String::from("something"), 3);
        assert!(ob.is_ok());
        let model = Model {
            state_metadata: metadata,
            ..Default::default()
        };
        let parameters = FxHashMap::default();

        let value = Yaml::String(String::from("(something 0 1)"));
        let result = load_set_expression_from_yaml(&value, &model, &parameters);
        assert!(result.is_ok());
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(1);
        assert_eq!(
            result.unwrap(),
            SetExpression::Reference(ReferenceExpression::Constant(set))
        );
    }

    #[test]
    fn load_set_expression_from_yaml_err() {
        let model = Model::default();
        let parameters = FxHashMap::default();

        let value = Yaml::Array(vec![Yaml::Integer(0), Yaml::Integer(1)]);
        let result = load_set_expression_from_yaml(&value, &model, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn load_vector_expression_from_yaml_ok() {
        let model = Model::default();
        let parameters = FxHashMap::default();

        let value = Yaml::String(String::from("(vector 0 1)"));
        let result = load_vector_expression_from_yaml(&value, &model, &parameters);
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1]))
        );

        let value = Yaml::Array(vec![Yaml::Integer(0), Yaml::Integer(1)]);
        let result = load_vector_expression_from_yaml(&value, &model, &parameters);
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1]))
        );
    }

    #[test]
    fn load_vector_expression_from_yaml_err() {
        let model = Model::default();
        let parameters = FxHashMap::default();

        let value = Yaml::Integer(0);
        let result = load_vector_expression_from_yaml(&value, &model, &parameters);
        assert!(result.is_err());

        let value = Yaml::Array(vec![Yaml::String(String::from("1")), Yaml::Integer(1)]);
        let result = load_vector_expression_from_yaml(&value, &model, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn reduce_function_load_from_yaml_ok() {
        let yaml = r"min";
        let yaml = yaml_rust::YamlLoader::load_from_str(yaml);
        assert!(yaml.is_ok());
        let yaml = yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let reduce = load_reduce_function_from_yaml(yaml);
        assert!(reduce.is_ok());
        assert_eq!(reduce.unwrap(), ReduceFunction::Min);
        let yaml = r"max";
        let yaml = yaml_rust::YamlLoader::load_from_str(yaml);
        assert!(yaml.is_ok());
        let yaml = yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let reduce = load_reduce_function_from_yaml(yaml);
        assert!(reduce.is_ok());
        assert_eq!(reduce.unwrap(), ReduceFunction::Max);
        let yaml = r"sum";
        let yaml = yaml_rust::YamlLoader::load_from_str(yaml);
        assert!(yaml.is_ok());
        let yaml = yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let reduce = load_reduce_function_from_yaml(yaml);
        assert!(reduce.is_ok());
        assert_eq!(reduce.unwrap(), ReduceFunction::Sum);
        let yaml = r"product";
        let yaml = yaml_rust::YamlLoader::load_from_str(yaml);
        assert!(yaml.is_ok());
        let yaml = yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let reduce = load_reduce_function_from_yaml(yaml);
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
        let reduce = load_reduce_function_from_yaml(yaml);
        assert!(reduce.is_err());
    }

    #[test]
    fn model_load_from_yaml_ok() {
        let domain = r"
domain: ADD
state_variables: [ {name: v1, type: integer}, {name: v2, type: integer} ]
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
dual_bounds:
        - 0
        - 1
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
dual_bounds:
        - 2
        - 3
";
        let problem = yaml_rust::YamlLoader::load_from_str(problem);
        assert!(problem.is_ok());
        let problem = problem.unwrap();
        assert_eq!(problem.len(), 1);
        let problem = &problem[0];

        let model = load_model_from_yaml(domain, problem);
        assert!(model.is_ok());
        let model = model.unwrap();

        let mut name_to_integer_variable = FxHashMap::default();
        name_to_integer_variable.insert(String::from("v1"), 0);
        name_to_integer_variable.insert(String::from("v2"), 1);
        let expected = Model {
            state_metadata: StateMetadata {
                integer_variable_names: vec![String::from("v1"), String::from("v2")],
                name_to_integer_variable,
                ..Default::default()
            },
            target: State {
                signature_variables: SignatureVariables {
                    integer_variables: vec![0, 0],
                    ..Default::default()
                },
                ..Default::default()
            },
            state_constraints: vec![
                GroundedCondition {
                    condition: Condition::ComparisonI(
                        ComparisonOperator::Ge,
                        Box::new(IntegerExpression::Variable(0)),
                        Box::new(IntegerExpression::Constant(0)),
                    ),
                    ..Default::default()
                },
                GroundedCondition {
                    condition: Condition::ComparisonI(
                        ComparisonOperator::Ge,
                        Box::new(IntegerExpression::Variable(1)),
                        Box::new(IntegerExpression::Constant(0)),
                    ),
                    ..Default::default()
                },
            ],
            base_cases: vec![
                BaseCase::from(vec![GroundedCondition {
                    condition: Condition::ComparisonI(
                        ComparisonOperator::Ge,
                        Box::new(IntegerExpression::Variable(0)),
                        Box::new(IntegerExpression::Constant(1)),
                    ),
                    ..Default::default()
                }]),
                BaseCase::from(vec![GroundedCondition {
                    condition: Condition::ComparisonI(
                        ComparisonOperator::Ge,
                        Box::new(IntegerExpression::Variable(1)),
                        Box::new(IntegerExpression::Constant(1)),
                    ),
                    ..Default::default()
                }]),
            ],
            forward_transitions: vec![
                Transition {
                    name: String::from("add"),
                    effect: Effect {
                        integer_effects: vec![(
                            0,
                            IntegerExpression::BinaryOperation(
                                BinaryOperator::Add,
                                Box::new(IntegerExpression::Variable(0)),
                                Box::new(IntegerExpression::Constant(1)),
                            ),
                        )],
                        ..Default::default()
                    },
                    cost: CostExpression::Integer(IntegerExpression::BinaryOperation(
                        BinaryOperator::Add,
                        Box::new(IntegerExpression::Cost),
                        Box::new(IntegerExpression::Constant(1)),
                    )),
                    ..Default::default()
                },
                Transition {
                    name: String::from("addv2"),
                    effect: Effect {
                        integer_effects: vec![(
                            1,
                            IntegerExpression::BinaryOperation(
                                BinaryOperator::Add,
                                Box::new(IntegerExpression::Variable(1)),
                                Box::new(IntegerExpression::Constant(1)),
                            ),
                        )],
                        ..Default::default()
                    },
                    cost: CostExpression::Integer(IntegerExpression::BinaryOperation(
                        BinaryOperator::Add,
                        Box::new(IntegerExpression::Cost),
                        Box::new(IntegerExpression::Constant(1)),
                    )),
                    ..Default::default()
                },
            ],
            forward_forced_transitions: vec![Transition {
                name: String::from("recover"),
                preconditions: vec![GroundedCondition {
                    condition: Condition::ComparisonI(
                        ComparisonOperator::Lt,
                        Box::new(IntegerExpression::Variable(0)),
                        Box::new(IntegerExpression::Constant(0)),
                    ),
                    ..Default::default()
                }],
                effect: Effect {
                    integer_effects: vec![(0, IntegerExpression::Constant(0))],
                    ..Default::default()
                },
                cost: CostExpression::Integer(IntegerExpression::Cost),
                ..Default::default()
            }],
            dual_bounds: vec![
                CostExpression::Integer(IntegerExpression::Constant(0)),
                CostExpression::Integer(IntegerExpression::Constant(1)),
                CostExpression::Integer(IntegerExpression::Constant(2)),
                CostExpression::Integer(IntegerExpression::Constant(3)),
            ],
            ..Default::default()
        };

        assert_eq!(model.state_metadata, expected.state_metadata);
        assert_eq!(model.target, expected.target);
        assert_eq!(model.table_registry, expected.table_registry);
        assert_eq!(model.state_constraints, expected.state_constraints);
        assert_eq!(model.base_cases, expected.base_cases);
        assert_eq!(model.reduce_function, expected.reduce_function);
        assert_eq!(model.cost_type, expected.cost_type);
        assert_eq!(model.forward_transitions, expected.forward_transitions);
        assert_eq!(model.backward_transitions, expected.backward_transitions);
        assert_eq!(model, expected);

        let domain = r"
domain: Fibonacci 
state_variables: [ {name: v, type: integer} ]
reduce: sum
cost_type: integer
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

        let model = load_model_from_yaml(domain, problem);
        assert!(model.is_ok());
        let model = model.unwrap();

        let mut name_to_integer_variable = FxHashMap::default();
        name_to_integer_variable.insert(String::from("v"), 0);
        let expected = Model {
            state_metadata: StateMetadata {
                integer_variable_names: vec![String::from("v")],
                name_to_integer_variable,
                ..Default::default()
            },
            target: State {
                signature_variables: SignatureVariables {
                    integer_variables: vec![10],
                    ..Default::default()
                },
                ..Default::default()
            },
            base_states: vec![
                (
                    State {
                        signature_variables: SignatureVariables {
                            integer_variables: vec![0],
                            ..Default::default()
                        },
                        ..Default::default()
                    },
                    None,
                ),
                (
                    State {
                        signature_variables: SignatureVariables {
                            integer_variables: vec![1],
                            ..Default::default()
                        },
                        ..Default::default()
                    },
                    None,
                ),
            ],
            reduce_function: ReduceFunction::Sum,
            cost_type: CostType::Integer,
            backward_transitions: vec![
                Transition {
                    name: String::from("one"),
                    effect: Effect {
                        integer_effects: vec![(
                            0,
                            IntegerExpression::BinaryOperation(
                                BinaryOperator::Add,
                                Box::new(IntegerExpression::Variable(0)),
                                Box::new(IntegerExpression::Constant(1)),
                            ),
                        )],
                        ..Default::default()
                    },
                    cost: CostExpression::Integer(IntegerExpression::Cost),
                    ..Default::default()
                },
                Transition {
                    name: String::from("two"),
                    effect: Effect {
                        integer_effects: vec![(
                            0,
                            IntegerExpression::BinaryOperation(
                                BinaryOperator::Add,
                                Box::new(IntegerExpression::Variable(0)),
                                Box::new(IntegerExpression::Constant(2)),
                            ),
                        )],
                        ..Default::default()
                    },
                    cost: CostExpression::Integer(IntegerExpression::Cost),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        assert_eq!(model.state_metadata, expected.state_metadata);
        assert_eq!(model.target, expected.target);
        assert_eq!(model.table_registry, expected.table_registry);
        assert_eq!(model.state_constraints, expected.state_constraints);
        assert_eq!(model.base_cases, expected.base_cases);
        assert_eq!(model.base_states, expected.base_states);
        assert_eq!(model.reduce_function, expected.reduce_function);
        assert_eq!(model.cost_type, expected.cost_type);
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
cost_type: continuous
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

        let model = load_model_from_yaml(domain, problem);
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
        let mut unvisited = Set::with_capacity(3);
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
            state_metadata: StateMetadata {
                object_type_names: vec![String::from("cities")],
                name_to_object_type: name_to_object,
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
            target: State {
                signature_variables: SignatureVariables {
                    set_variables: vec![unvisited],
                    element_variables: vec![0],
                    ..Default::default()
                },
                resource_variables: ResourceVariables {
                    integer_variables: vec![0],
                    ..Default::default()
                },
            },
            table_registry: TableRegistry {
                integer_tables: TableData {
                    tables_1d: vec![Table1D::new(vec![0, 1, 1]), Table1D::new(vec![10000, 2, 2])],
                    name_to_table_1d,
                    tables_2d: vec![Table2D::new(vec![
                        vec![0, 1, 1],
                        vec![1, 0, 1],
                        vec![1, 1, 0],
                    ])],
                    name_to_table_2d: numeric_name_to_table_2d,
                    ..Default::default()
                },
                bool_tables: TableData {
                    tables_2d: vec![Table2D::new(vec![
                        vec![false, true, true],
                        vec![true, false, true],
                        vec![true, true, false],
                    ])],
                    name_to_table_2d: bool_name_to_table_2d,
                    ..Default::default()
                },
                ..Default::default()
            },
            state_constraints: vec![GroundedCondition {
                condition: Condition::ComparisonI(
                    ComparisonOperator::Le,
                    Box::new(IntegerExpression::ResourceVariable(0)),
                    Box::new(IntegerExpression::Table(Box::new(
                        NumericTableExpression::Table1D(1, ElementExpression::Variable(0)),
                    ))),
                ),
                ..Default::default()
            }],
            base_cases: vec![BaseCase::from(vec![
                GroundedCondition {
                    condition: Condition::Set(Box::new(SetCondition::IsEmpty(
                        SetExpression::Reference(ReferenceExpression::Variable(0)),
                    ))),
                    ..Default::default()
                },
                GroundedCondition {
                    condition: Condition::ComparisonE(
                        ComparisonOperator::Eq,
                        Box::new(ElementExpression::Variable(0)),
                        Box::new(ElementExpression::Constant(0)),
                    ),
                    ..Default::default()
                },
            ])],
            base_states: Vec::new(),
            reduce_function: ReduceFunction::Min,
            cost_type: CostType::Continuous,
            forward_transitions: vec![
                Transition {
                    name: String::from("visit"),
                    parameter_names: vec![String::from("to")],
                    parameter_values: vec![0],
                    elements_in_set_variable: vec![(0, 0)],
                    preconditions: vec![GroundedCondition {
                        condition: Condition::Table(Box::new(TableExpression::Table2D(
                            0,
                            ElementExpression::Variable(0),
                            ElementExpression::Constant(0),
                        ))),
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
                            IntegerExpression::BinaryOperation(
                                BinaryOperator::Max,
                                Box::new(IntegerExpression::BinaryOperation(
                                    BinaryOperator::Add,
                                    Box::new(IntegerExpression::ResourceVariable(0)),
                                    Box::new(IntegerExpression::Table(Box::new(
                                        NumericTableExpression::Table2D(
                                            0,
                                            ElementExpression::Variable(0),
                                            ElementExpression::Constant(0),
                                        ),
                                    ))),
                                )),
                                Box::new(IntegerExpression::Constant(0)),
                            ),
                        )],
                        ..Default::default()
                    },
                    cost: CostExpression::Continuous(ContinuousExpression::BinaryOperation(
                        BinaryOperator::Add,
                        Box::new(ContinuousExpression::Cost),
                        Box::new(ContinuousExpression::FromInteger(Box::new(
                            IntegerExpression::Table(Box::new(NumericTableExpression::Table2D(
                                0,
                                ElementExpression::Variable(0),
                                ElementExpression::Constant(0),
                            ))),
                        ))),
                    )),
                    ..Default::default()
                },
                Transition {
                    name: String::from("visit"),
                    parameter_names: vec![String::from("to")],
                    parameter_values: vec![1],
                    elements_in_set_variable: vec![(0, 1)],
                    preconditions: vec![GroundedCondition {
                        condition: Condition::Table(Box::new(TableExpression::Table2D(
                            0,
                            ElementExpression::Variable(0),
                            ElementExpression::Constant(1),
                        ))),
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
                            IntegerExpression::BinaryOperation(
                                BinaryOperator::Max,
                                Box::new(IntegerExpression::BinaryOperation(
                                    BinaryOperator::Add,
                                    Box::new(IntegerExpression::ResourceVariable(0)),
                                    Box::new(IntegerExpression::Table(Box::new(
                                        NumericTableExpression::Table2D(
                                            0,
                                            ElementExpression::Variable(0),
                                            ElementExpression::Constant(1),
                                        ),
                                    ))),
                                )),
                                Box::new(IntegerExpression::Constant(1)),
                            ),
                        )],
                        ..Default::default()
                    },
                    cost: CostExpression::Continuous(ContinuousExpression::BinaryOperation(
                        BinaryOperator::Add,
                        Box::new(ContinuousExpression::Cost),
                        Box::new(ContinuousExpression::FromInteger(Box::new(
                            IntegerExpression::Table(Box::new(NumericTableExpression::Table2D(
                                0,
                                ElementExpression::Variable(0),
                                ElementExpression::Constant(1),
                            ))),
                        ))),
                    )),
                    ..Default::default()
                },
                Transition {
                    name: String::from("visit"),
                    parameter_names: vec![String::from("to")],
                    parameter_values: vec![2],
                    elements_in_set_variable: vec![(0, 2)],
                    preconditions: vec![GroundedCondition {
                        condition: Condition::Table(Box::new(TableExpression::Table2D(
                            0,
                            ElementExpression::Variable(0),
                            ElementExpression::Constant(2),
                        ))),
                        ..Default::default()
                    }],
                    effect: Effect {
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
                            IntegerExpression::BinaryOperation(
                                BinaryOperator::Max,
                                Box::new(IntegerExpression::BinaryOperation(
                                    BinaryOperator::Add,
                                    Box::new(IntegerExpression::ResourceVariable(0)),
                                    Box::new(IntegerExpression::Table(Box::new(
                                        NumericTableExpression::Table2D(
                                            0,
                                            ElementExpression::Variable(0),
                                            ElementExpression::Constant(2),
                                        ),
                                    ))),
                                )),
                                Box::new(IntegerExpression::Constant(1)),
                            ),
                        )],
                        ..Default::default()
                    },
                    cost: CostExpression::Continuous(ContinuousExpression::BinaryOperation(
                        BinaryOperator::Add,
                        Box::new(ContinuousExpression::Cost),
                        Box::new(ContinuousExpression::FromInteger(Box::new(
                            IntegerExpression::Table(Box::new(NumericTableExpression::Table2D(
                                0,
                                ElementExpression::Variable(0),
                                ElementExpression::Constant(2),
                            ))),
                        ))),
                    )),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        assert_eq!(model.state_metadata, expected.state_metadata);
        assert_eq!(model.target, expected.target);
        assert_eq!(model.table_registry, expected.table_registry);
        assert_eq!(model.state_constraints, expected.state_constraints);
        assert_eq!(model.base_cases, expected.base_cases);
        assert_eq!(model.reduce_function, expected.reduce_function);
        assert_eq!(model.cost_type, expected.cost_type);
        assert_eq!(model.forward_transitions, expected.forward_transitions);
        assert_eq!(model.backward_transitions, expected.backward_transitions);
        assert_eq!(model, expected);
    }

    #[test]
    fn model_load_from_yaml_with_dictionary_ok() {
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
dictionaries:
        - name: connected
          type: bool
          default: true
constraints:
        - (<= time (due_date location))
        - (= 0 0)
reduce: min
cost_type: continuous
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
dictionary_values:
        connected: {[0, 0]: false, [1, 1]: false, [2, 2]: false}
";
        let problem = yaml_rust::YamlLoader::load_from_str(problem);
        assert!(problem.is_ok());
        let problem = problem.unwrap();
        assert_eq!(problem.len(), 1);
        let problem = &problem[0];

        let model = load_model_from_yaml(domain, problem);
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
        let mut unvisited = Set::with_capacity(3);
        unvisited.insert(0);
        unvisited.insert(1);
        unvisited.insert(2);
        let mut name_to_table_1d = FxHashMap::default();
        name_to_table_1d.insert(String::from("ready_time"), 0);
        name_to_table_1d.insert(String::from("due_date"), 1);
        let mut numeric_name_to_table_2d = FxHashMap::default();
        numeric_name_to_table_2d.insert(String::from("distance"), 0);
        let mut bool_name_to_table = FxHashMap::default();
        bool_name_to_table.insert(String::from("connected"), 0);
        let mut bool_table = FxHashMap::<Vec<Element>, bool>::default();
        bool_table.insert(vec![0, 0], false);
        bool_table.insert(vec![1, 1], false);
        bool_table.insert(vec![2, 2], false);

        let expected = Model {
            state_metadata: StateMetadata {
                object_type_names: vec![String::from("cities")],
                name_to_object_type: name_to_object,
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
            target: State {
                signature_variables: SignatureVariables {
                    set_variables: vec![unvisited],
                    element_variables: vec![0],
                    ..Default::default()
                },
                resource_variables: ResourceVariables {
                    integer_variables: vec![0],
                    ..Default::default()
                },
            },
            table_registry: TableRegistry {
                integer_tables: TableData {
                    tables_1d: vec![Table1D::new(vec![0, 1, 1]), Table1D::new(vec![10000, 2, 2])],
                    name_to_table_1d,
                    tables_2d: vec![Table2D::new(vec![
                        vec![0, 1, 1],
                        vec![1, 0, 1],
                        vec![1, 1, 0],
                    ])],
                    name_to_table_2d: numeric_name_to_table_2d,
                    ..Default::default()
                },
                bool_tables: TableData {
                    tables: vec![Table::new(bool_table, true)],
                    name_to_table: bool_name_to_table,
                    ..Default::default()
                },
                ..Default::default()
            },
            state_constraints: vec![GroundedCondition {
                condition: Condition::ComparisonI(
                    ComparisonOperator::Le,
                    Box::new(IntegerExpression::ResourceVariable(0)),
                    Box::new(IntegerExpression::Table(Box::new(
                        NumericTableExpression::Table1D(1, ElementExpression::Variable(0)),
                    ))),
                ),
                ..Default::default()
            }],
            base_cases: vec![BaseCase::from(vec![
                GroundedCondition {
                    condition: Condition::Set(Box::new(SetCondition::IsEmpty(
                        SetExpression::Reference(ReferenceExpression::Variable(0)),
                    ))),
                    ..Default::default()
                },
                GroundedCondition {
                    condition: Condition::ComparisonE(
                        ComparisonOperator::Eq,
                        Box::new(ElementExpression::Variable(0)),
                        Box::new(ElementExpression::Constant(0)),
                    ),
                    ..Default::default()
                },
            ])],
            base_states: Vec::new(),
            reduce_function: ReduceFunction::Min,
            cost_type: CostType::Continuous,
            forward_transitions: vec![
                Transition {
                    name: String::from("visit"),
                    parameter_names: vec![String::from("to")],
                    parameter_values: vec![0],
                    elements_in_set_variable: vec![(0, 0)],
                    preconditions: vec![GroundedCondition {
                        condition: Condition::Table(Box::new(TableExpression::Table(
                            0,
                            vec![
                                ElementExpression::Variable(0),
                                ElementExpression::Constant(0),
                            ],
                        ))),
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
                            IntegerExpression::BinaryOperation(
                                BinaryOperator::Max,
                                Box::new(IntegerExpression::BinaryOperation(
                                    BinaryOperator::Add,
                                    Box::new(IntegerExpression::ResourceVariable(0)),
                                    Box::new(IntegerExpression::Table(Box::new(
                                        NumericTableExpression::Table2D(
                                            0,
                                            ElementExpression::Variable(0),
                                            ElementExpression::Constant(0),
                                        ),
                                    ))),
                                )),
                                Box::new(IntegerExpression::Constant(0)),
                            ),
                        )],
                        ..Default::default()
                    },
                    cost: CostExpression::Continuous(ContinuousExpression::BinaryOperation(
                        BinaryOperator::Add,
                        Box::new(ContinuousExpression::Cost),
                        Box::new(ContinuousExpression::FromInteger(Box::new(
                            IntegerExpression::Table(Box::new(NumericTableExpression::Table2D(
                                0,
                                ElementExpression::Variable(0),
                                ElementExpression::Constant(0),
                            ))),
                        ))),
                    )),
                    ..Default::default()
                },
                Transition {
                    name: String::from("visit"),
                    parameter_names: vec![String::from("to")],
                    parameter_values: vec![1],
                    elements_in_set_variable: vec![(0, 1)],
                    preconditions: vec![GroundedCondition {
                        condition: Condition::Table(Box::new(TableExpression::Table(
                            0,
                            vec![
                                ElementExpression::Variable(0),
                                ElementExpression::Constant(1),
                            ],
                        ))),
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
                            IntegerExpression::BinaryOperation(
                                BinaryOperator::Max,
                                Box::new(IntegerExpression::BinaryOperation(
                                    BinaryOperator::Add,
                                    Box::new(IntegerExpression::ResourceVariable(0)),
                                    Box::new(IntegerExpression::Table(Box::new(
                                        NumericTableExpression::Table2D(
                                            0,
                                            ElementExpression::Variable(0),
                                            ElementExpression::Constant(1),
                                        ),
                                    ))),
                                )),
                                Box::new(IntegerExpression::Constant(1)),
                            ),
                        )],
                        ..Default::default()
                    },
                    cost: CostExpression::Continuous(ContinuousExpression::BinaryOperation(
                        BinaryOperator::Add,
                        Box::new(ContinuousExpression::Cost),
                        Box::new(ContinuousExpression::FromInteger(Box::new(
                            IntegerExpression::Table(Box::new(NumericTableExpression::Table2D(
                                0,
                                ElementExpression::Variable(0),
                                ElementExpression::Constant(1),
                            ))),
                        ))),
                    )),
                    ..Default::default()
                },
                Transition {
                    name: String::from("visit"),
                    parameter_names: vec![String::from("to")],
                    parameter_values: vec![2],
                    elements_in_set_variable: vec![(0, 2)],
                    preconditions: vec![GroundedCondition {
                        condition: Condition::Table(Box::new(TableExpression::Table(
                            0,
                            vec![
                                ElementExpression::Variable(0),
                                ElementExpression::Constant(2),
                            ],
                        ))),
                        ..Default::default()
                    }],
                    effect: Effect {
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
                            IntegerExpression::BinaryOperation(
                                BinaryOperator::Max,
                                Box::new(IntegerExpression::BinaryOperation(
                                    BinaryOperator::Add,
                                    Box::new(IntegerExpression::ResourceVariable(0)),
                                    Box::new(IntegerExpression::Table(Box::new(
                                        NumericTableExpression::Table2D(
                                            0,
                                            ElementExpression::Variable(0),
                                            ElementExpression::Constant(2),
                                        ),
                                    ))),
                                )),
                                Box::new(IntegerExpression::Constant(1)),
                            ),
                        )],
                        ..Default::default()
                    },
                    cost: CostExpression::Continuous(ContinuousExpression::BinaryOperation(
                        BinaryOperator::Add,
                        Box::new(ContinuousExpression::Cost),
                        Box::new(ContinuousExpression::FromInteger(Box::new(
                            IntegerExpression::Table(Box::new(NumericTableExpression::Table2D(
                                0,
                                ElementExpression::Variable(0),
                                ElementExpression::Constant(2),
                            ))),
                        ))),
                    )),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        assert_eq!(model.state_metadata, expected.state_metadata);
        assert_eq!(model.target, expected.target);
        assert_eq!(model.table_registry, expected.table_registry);
        assert_eq!(model.state_constraints, expected.state_constraints);
        assert_eq!(model.base_cases, expected.base_cases);
        assert_eq!(model.reduce_function, expected.reduce_function);
        assert_eq!(model.cost_type, expected.cost_type);
        assert_eq!(model.forward_transitions, expected.forward_transitions);
        assert_eq!(model.backward_transitions, expected.backward_transitions);
        assert_eq!(model, expected);
    }

    #[test]
    fn model_load_from_yaml_err() {
        let domain = r"
domain: ADD
state_variables: [ {name: v1, type: integer}, {name: v2, type: integer} ]
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
dual_bounds:
        - 0
        - foo
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
dual_bounds:
        - 2
        - 3
";
        let problem = yaml_rust::YamlLoader::load_from_str(problem);
        assert!(problem.is_ok());
        let problem = problem.unwrap();
        assert_eq!(problem.len(), 1);
        let problem = &problem[0];

        let model = load_model_from_yaml(domain, problem);
        assert!(model.is_err());

        let domain = r"
domain: ADD
state_variables: [ {name: v1, type: integer}, {name: v2, type: integer} ]
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
dual_bounds:
        - 0
        - 1
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
dual_bounds:
        - 2
        - foo
";
        let problem = yaml_rust::YamlLoader::load_from_str(problem);
        assert!(problem.is_ok());
        let problem = problem.unwrap();
        assert_eq!(problem.len(), 1);
        let problem = &problem[0];

        let model = load_model_from_yaml(domain, problem);
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

        let model = load_model_from_yaml(domain, problem);
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

        let model = load_model_from_yaml(domain, problem);
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

        let model = load_model_from_yaml(domain, problem);
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

        let model = load_model_from_yaml(domain, problem);
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

        let model = load_model_from_yaml(domain, problem);
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

        let model = load_model_from_yaml(domain, problem);
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

        let model = load_model_from_yaml(domain, problem);
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

        let model = load_model_from_yaml(domain, problem);
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

        let model = load_model_from_yaml(domain, problem);
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

        let model = load_model_from_yaml(domain, problem);
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

        let model = load_model_from_yaml(domain, problem);
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

        let model = load_model_from_yaml(domain, problem);
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

        let model = load_model_from_yaml(domain, problem);
        assert!(model.is_err());
    }

    #[test]
    fn model_load_from_yaml_with_dictionary_err() {
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
dictionaries:
        - name: connected
          type: bool
          default: true
constraints:
        - (<= time (due_date location))
        - (= 0 0)
reduce: min
cost_type: continuous
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
";
        let problem = yaml_rust::YamlLoader::load_from_str(problem);
        assert!(problem.is_ok());
        let problem = problem.unwrap();
        assert_eq!(problem.len(), 1);
        let problem = &problem[0];

        let model = load_model_from_yaml(domain, problem);
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
constraints:
        - (<= time (due_date location))
        - (= 0 0)
reduce: min
cost_type: continuous
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
dictionary_values:
        connected: {[0, 0]: false, [1, 1]: false, [2, 2]: false}
";
        let problem = yaml_rust::YamlLoader::load_from_str(problem);
        assert!(problem.is_ok());
        let problem = problem.unwrap();
        assert_eq!(problem.len(), 1);
        let problem = &problem[0];

        let model = load_model_from_yaml(domain, problem);
        assert!(model.is_err());
    }
}
