use super::expression_parser;
use super::grounded_condition_parser;
use super::parse_expression_from_yaml::{
    parse_continuous_from_yaml, parse_element_from_yaml, parse_integer_from_yaml,
};
use super::state_parser::ground_parameters_from_yaml;
use crate::util;
use dypdl::expression;
use dypdl::variable_type::Element;
use dypdl::{CostExpression, CostType, Effect, StateMetadata, TableRegistry, Transition};
use lazy_static::lazy_static;
use rustc_hash::FxHashMap;
use std::error::Error;
use yaml_rust::Yaml;

type TransitionsWithFlags = (Vec<Transition>, bool, bool);

/// Returns transitoins of loaded from YAML
///
/// The second returned value indicates whether it is forced or not.
/// The third returned value indicates whether it is backward or not.
///
/// # Errors
///
/// if the format is invalid.
pub fn load_transitions_from_yaml(
    value: &Yaml,
    metadata: &StateMetadata,
    registry: &TableRegistry,
    cost_type: &CostType,
) -> Result<TransitionsWithFlags, Box<dyn Error>> {
    lazy_static! {
        static ref PARAMETERS_KEY: Yaml = Yaml::from_str("parameters");
        static ref PRECONDITIONS_KEY: Yaml = Yaml::from_str("preconditions");
        static ref DIRECTION_KEY: Yaml = Yaml::from_str("direction");
        static ref FORCED_KEY: Yaml = Yaml::from_str("forced");
    }
    let map = util::get_map(value)?;
    let lifted_name = util::get_string_by_key(map, "name")?;

    let (
        parameters_array,
        elements_in_set_variable_array,
        elements_in_vector_variable_array,
        parameter_names,
    ) = match map.get(&PARAMETERS_KEY) {
        Some(value) => {
            let result = ground_parameters_from_yaml(metadata, value)?;
            let array = util::get_array(value)?;
            let mut parameter_names = Vec::with_capacity(array.len());
            for map in array {
                let map = util::get_map(map)?;
                let value = util::get_string_by_key(map, "name")?;
                parameter_names.push(value);
            }
            (result.0, result.1, result.2, parameter_names)
        }
        None => (
            vec![FxHashMap::default()],
            vec![vec![]],
            vec![vec![]],
            vec![],
        ),
    };

    let effect = map.get(&Yaml::String(String::from("effect")));
    let lifted_cost = map.get(&Yaml::String(String::from("cost")));
    let lifted_preconditions = match map.get(&PRECONDITIONS_KEY) {
        Some(lifted_preconditions) => Some(util::get_array(lifted_preconditions)?),
        None => None,
    };
    let mut transitions = Vec::with_capacity(parameters_array.len());
    'outer: for ((parameters, elements_in_set_variable), elements_in_vector_variable) in
        parameters_array
            .into_iter()
            .zip(elements_in_set_variable_array.into_iter())
            .zip(elements_in_vector_variable_array.into_iter())
    {
        let parameter_values = parameter_names
            .iter()
            .map(|name| parameters[name])
            .collect();
        let preconditions = match lifted_preconditions {
            Some(lifted_preconditions) => {
                let mut preconditions = Vec::with_capacity(lifted_preconditions.len());
                for condition in lifted_preconditions {
                    let conditions = grounded_condition_parser::load_grounded_conditions_from_yaml(
                        condition,
                        metadata,
                        registry,
                        &parameters,
                    )?;
                    for condition in conditions {
                        match condition.condition {
                            expression::Condition::Constant(true) => continue,
                            expression::Condition::Constant(false)
                                if condition.elements_in_set_variable.is_empty()
                                    && condition.elements_in_vector_variable.is_empty() =>
                            {
                                continue 'outer
                            }
                            _ => preconditions.push(condition),
                        }
                    }
                }
                preconditions
            }
            None => Vec::new(),
        };
        let effect = match effect {
            Some(effect) => load_effect_from_yaml(effect, metadata, registry, &parameters)?,
            None => Effect::default(),
        };
        let cost = match cost_type {
            CostType::Integer => {
                let expression = match lifted_cost {
                    Some(cost) => parse_integer_from_yaml(cost, metadata, registry, &parameters)?,
                    None => expression::IntegerExpression::Cost,
                };
                CostExpression::Integer(expression.simplify(registry))
            }
            CostType::Continuous => {
                let expression = match lifted_cost {
                    Some(cost) => {
                        parse_continuous_from_yaml(cost, metadata, registry, &parameters)?
                    }
                    None => expression::ContinuousExpression::Cost,
                };
                CostExpression::Continuous(expression.simplify(registry))
            }
        };

        transitions.push(Transition {
            name: lifted_name.clone(),
            parameter_names: parameter_names.clone(),
            parameter_values,
            elements_in_set_variable,
            elements_in_vector_variable,
            preconditions,
            effect,
            cost,
        })
    }
    let backward = match map.get(&DIRECTION_KEY) {
        Some(direction) => {
            let direction = util::get_string(direction)?;
            match &direction[..] {
                "forward" => false,
                "backward" => true,
                _ => {
                    return Err(util::YamlContentErr::new(format!(
                        "no such direction `{}`",
                        direction
                    ))
                    .into())
                }
            }
        }
        None => false,
    };
    let forced = match map.get(&FORCED_KEY) {
        Some(forced) => util::get_bool(forced)?,
        None => false,
    };

    Ok((transitions, forced, backward))
}

fn load_effect_from_yaml(
    value: &Yaml,
    metadata: &StateMetadata,
    registry: &TableRegistry,
    parameters: &FxHashMap<String, Element>,
) -> Result<Effect, Box<dyn Error>> {
    let lifted_effects = util::get_map(value)?;
    let mut set_effects = Vec::new();
    let mut vector_effects = Vec::new();
    let mut element_effects = Vec::new();
    let mut integer_effects = Vec::new();
    let mut continuous_effects = Vec::new();
    let mut element_resource_effects = Vec::new();
    let mut integer_resource_effects = Vec::new();
    let mut continuous_resource_effects = Vec::new();
    for (variable, effect) in lifted_effects {
        let variable = util::get_string(variable)?;
        if let Some(i) = metadata.name_to_set_variable.get(&variable) {
            let effect = util::get_string(effect)?;
            let effect = expression_parser::parse_set(effect, metadata, registry, parameters)?;
            set_effects.push((*i, effect.simplify(registry)));
        } else if let Some(i) = metadata.name_to_vector_variable.get(&variable) {
            let effect = util::get_string(effect)?;
            let effect = expression_parser::parse_vector(effect, metadata, registry, parameters)?;
            vector_effects.push((*i, effect.simplify(registry)));
        } else if let Some(i) = metadata.name_to_element_variable.get(&variable) {
            let effect = parse_element_from_yaml(effect, metadata, registry, parameters)?;
            element_effects.push((*i, effect.simplify(registry)));
        } else if let Some(i) = metadata.name_to_element_resource_variable.get(&variable) {
            let effect = parse_element_from_yaml(effect, metadata, registry, parameters)?;
            element_resource_effects.push((*i, effect.simplify(registry)));
        } else if let Some(i) = metadata.name_to_integer_variable.get(&variable) {
            let effect = parse_integer_from_yaml(effect, metadata, registry, parameters)?;
            integer_effects.push((*i, effect.simplify(registry)));
        } else if let Some(i) = metadata.name_to_integer_resource_variable.get(&variable) {
            let effect = parse_integer_from_yaml(effect, metadata, registry, parameters)?;
            integer_resource_effects.push((*i, effect.simplify(registry)));
        } else if let Some(i) = metadata.name_to_continuous_variable.get(&variable) {
            let effect = parse_continuous_from_yaml(effect, metadata, registry, parameters)?;
            continuous_effects.push((*i, effect.simplify(registry)));
        } else if let Some(i) = metadata.name_to_continuous_resource_variable.get(&variable) {
            let effect = parse_continuous_from_yaml(effect, metadata, registry, parameters)?;
            continuous_resource_effects.push((*i, effect.simplify(registry)));
        } else {
            return Err(
                util::YamlContentErr::new(format!("no such variable `{}`", variable)).into(),
            );
        }
    }
    Ok(Effect {
        set_effects,
        vector_effects,
        element_effects,
        integer_effects,
        continuous_effects,
        element_resource_effects,
        integer_resource_effects,
        continuous_resource_effects,
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    use dypdl::expression::*;
    use dypdl::prelude::*;
    use dypdl::CostExpression;
    use dypdl::GroundedCondition;
    use rustc_hash::FxHashMap;

    #[test]
    fn load_transitions_from_yaml_ok() {
        let mut metadata = StateMetadata::default();
        let result = metadata.add_object_type(String::from("object"), 3);
        assert!(result.is_ok());
        let ob = result.unwrap();
        let result = metadata.add_set_variable(String::from("s0"), ob);
        assert!(result.is_ok());
        let result = metadata.add_set_variable(String::from("s1"), ob);
        assert!(result.is_ok());
        let result = metadata.add_set_variable(String::from("s2"), ob);
        assert!(result.is_ok());
        let result = metadata.add_set_variable(String::from("s3"), ob);
        assert!(result.is_ok());
        let result = metadata.add_vector_variable(String::from("p0"), ob);
        assert!(result.is_ok());
        let result = metadata.add_vector_variable(String::from("p1"), ob);
        assert!(result.is_ok());
        let result = metadata.add_vector_variable(String::from("p2"), ob);
        assert!(result.is_ok());
        let result = metadata.add_vector_variable(String::from("p3"), ob);
        assert!(result.is_ok());
        let result = metadata.add_element_variable(String::from("e0"), ob);
        assert!(result.is_ok());
        let result = metadata.add_element_variable(String::from("e1"), ob);
        assert!(result.is_ok());
        let result = metadata.add_element_variable(String::from("e2"), ob);
        assert!(result.is_ok());
        let result = metadata.add_element_variable(String::from("e3"), ob);
        assert!(result.is_ok());
        let result = metadata.add_element_resource_variable(String::from("er0"), ob, false);
        assert!(result.is_ok());
        let result = metadata.add_element_resource_variable(String::from("er1"), ob, false);
        assert!(result.is_ok());
        let result = metadata.add_element_resource_variable(String::from("er2"), ob, true);
        assert!(result.is_ok());
        let result = metadata.add_element_resource_variable(String::from("er3"), ob, false);
        assert!(result.is_ok());
        let result = metadata.add_integer_variable(String::from("i0"));
        assert!(result.is_ok());
        let result = metadata.add_integer_variable(String::from("i1"));
        assert!(result.is_ok());
        let result = metadata.add_integer_variable(String::from("i2"));
        assert!(result.is_ok());
        let result = metadata.add_integer_variable(String::from("i3"));
        assert!(result.is_ok());
        let result = metadata.add_continuous_variable(String::from("c0"));
        assert!(result.is_ok());
        let result = metadata.add_continuous_variable(String::from("c1"));
        assert!(result.is_ok());
        let result = metadata.add_continuous_variable(String::from("c2"));
        assert!(result.is_ok());
        let result = metadata.add_continuous_variable(String::from("c3"));
        assert!(result.is_ok());
        let result = metadata.add_integer_resource_variable(String::from("ir0"), false);
        assert!(result.is_ok());
        let result = metadata.add_integer_resource_variable(String::from("ir1"), false);
        assert!(result.is_ok());
        let result = metadata.add_integer_resource_variable(String::from("ir2"), true);
        assert!(result.is_ok());
        let result = metadata.add_integer_resource_variable(String::from("ir3"), false);
        assert!(result.is_ok());
        let result = metadata.add_continuous_resource_variable(String::from("cr0"), false);
        assert!(result.is_ok());
        let result = metadata.add_continuous_resource_variable(String::from("cr1"), false);
        assert!(result.is_ok());
        let result = metadata.add_continuous_resource_variable(String::from("cr2"), true);
        assert!(result.is_ok());
        let result = metadata.add_continuous_resource_variable(String::from("cr3"), false);
        assert!(result.is_ok());

        let mut registry = TableRegistry::default();
        let result = registry.add_table_1d(String::from("f1"), vec![10, 20, 30]);
        assert!(result.is_ok());
        let result =
            registry.add_table_2d(String::from("f2"), vec![vec![10, 20, 30], vec![40, 50, 60]]);
        assert!(result.is_ok());

        let cost_type = CostType::Integer;

        let transition = r"
name: transition
preconditions: [(>= (f2 0 1) 10)]
";
        let transition = yaml_rust::YamlLoader::load_from_str(transition);
        assert!(transition.is_ok());
        let transition = transition.unwrap();
        assert_eq!(transition.len(), 1);
        let transition = &transition[0];
        let transitions = load_transitions_from_yaml(transition, &metadata, &registry, &cost_type);
        let expected = vec![Transition {
            name: String::from("transition"),
            preconditions: Vec::new(),
            effect: Effect::default(),
            cost: CostExpression::Integer(IntegerExpression::Cost),
            ..Default::default()
        }];
        assert_eq!(transitions.unwrap(), (expected, false, false));

        let cost_type_continuous = CostType::Continuous;

        let transition = r"
name: transition
preconditions: [(>= (f2 0 1) 10)]
";
        let transition = yaml_rust::YamlLoader::load_from_str(transition);
        assert!(transition.is_ok());
        let transition = transition.unwrap();
        assert_eq!(transition.len(), 1);
        let transition = &transition[0];
        let transitions =
            load_transitions_from_yaml(transition, &metadata, &registry, &cost_type_continuous);
        let expected = vec![Transition {
            name: String::from("transition"),
            preconditions: Vec::new(),
            effect: Effect::default(),
            cost: CostExpression::Continuous(ContinuousExpression::Cost),
            ..Default::default()
        }];
        assert_eq!(transitions.unwrap(), (expected, false, false));

        let transition = r"
name: transition
preconditions: [(>= (f2 0 1) 10)]
effect: {e0: 0}
cost: 0
";
        let transition = yaml_rust::YamlLoader::load_from_str(transition);
        assert!(transition.is_ok());
        let transition = transition.unwrap();
        assert_eq!(transition.len(), 1);
        let transition = &transition[0];
        let transitions = load_transitions_from_yaml(transition, &metadata, &registry, &cost_type);
        let expected = vec![Transition {
            name: String::from("transition"),
            preconditions: Vec::new(),
            effect: Effect {
                element_effects: vec![(0, ElementExpression::Constant(0))],
                ..Default::default()
            },
            cost: CostExpression::Integer(IntegerExpression::Constant(0)),
            ..Default::default()
        }];
        assert_eq!(transitions.unwrap(), (expected, false, false));

        let transition = r"
name: transition
preconditions: [(>= (f2 0 1) 10)]
effect: {e0: 0}
cost: 0
";
        let transition = yaml_rust::YamlLoader::load_from_str(transition);
        assert!(transition.is_ok());
        let transition = transition.unwrap();
        assert_eq!(transition.len(), 1);
        let transition = &transition[0];
        let transitions =
            load_transitions_from_yaml(transition, &metadata, &registry, &cost_type_continuous);
        let expected = vec![Transition {
            name: String::from("transition"),
            preconditions: Vec::new(),
            effect: Effect {
                element_effects: vec![(0, ElementExpression::Constant(0))],
                ..Default::default()
            },
            cost: CostExpression::Continuous(ContinuousExpression::Constant(0.0)),
            ..Default::default()
        }];
        assert_eq!(transitions.unwrap(), (expected, false, false));

        let transition = r"
name: transition
effect: {e0: '0'}
cost: '0'
";
        let transition = yaml_rust::YamlLoader::load_from_str(transition);
        assert!(transition.is_ok());
        let transition = transition.unwrap();
        assert_eq!(transition.len(), 1);
        let transition = &transition[0];
        let transitions = load_transitions_from_yaml(transition, &metadata, &registry, &cost_type);
        assert!(transitions.is_ok());
        let expected = vec![Transition {
            name: String::from("transition"),
            preconditions: Vec::new(),
            effect: Effect {
                element_effects: vec![(0, ElementExpression::Constant(0))],
                ..Default::default()
            },
            cost: CostExpression::Integer(IntegerExpression::Constant(0)),
            ..Default::default()
        }];
        assert_eq!(transitions.unwrap(), (expected, false, false));

        let transition = r"
name: transition
effect: {e0: '0'}
cost: '0'
forced: false
";
        let transition = yaml_rust::YamlLoader::load_from_str(transition);
        assert!(transition.is_ok());
        let transition = transition.unwrap();
        assert_eq!(transition.len(), 1);
        let transition = &transition[0];
        let transitions = load_transitions_from_yaml(transition, &metadata, &registry, &cost_type);
        assert!(transitions.is_ok());
        let expected = vec![Transition {
            name: String::from("transition"),
            preconditions: Vec::new(),
            effect: Effect {
                element_effects: vec![(0, ElementExpression::Constant(0))],
                ..Default::default()
            },
            cost: CostExpression::Integer(IntegerExpression::Constant(0)),
            ..Default::default()
        }];
        assert_eq!(transitions.unwrap(), (expected, false, false));

        let transition = r"
name: transition
effect: {e0: '0'}
cost: '0'
forced: true
";
        let transition = yaml_rust::YamlLoader::load_from_str(transition);
        assert!(transition.is_ok());
        let transition = transition.unwrap();
        assert_eq!(transition.len(), 1);
        let transition = &transition[0];
        let transitions = load_transitions_from_yaml(transition, &metadata, &registry, &cost_type);
        assert!(transitions.is_ok());
        let expected = vec![Transition {
            name: String::from("transition"),
            preconditions: Vec::new(),
            effect: Effect {
                element_effects: vec![(0, ElementExpression::Constant(0))],
                ..Default::default()
            },
            cost: CostExpression::Integer(IntegerExpression::Constant(0)),
            ..Default::default()
        }];
        assert_eq!(transitions.unwrap(), (expected, true, false));

        let transition = r"
name: transition
effect: {e0: '0'}
cost: '0'
direction: forward
";
        let transition = yaml_rust::YamlLoader::load_from_str(transition);
        assert!(transition.is_ok());
        let transition = transition.unwrap();
        assert_eq!(transition.len(), 1);
        let transition = &transition[0];
        let transitions = load_transitions_from_yaml(transition, &metadata, &registry, &cost_type);
        assert!(transitions.is_ok());
        let expected = vec![Transition {
            name: String::from("transition"),
            preconditions: Vec::new(),
            effect: Effect {
                element_effects: vec![(0, ElementExpression::Constant(0))],
                ..Default::default()
            },
            cost: CostExpression::Integer(IntegerExpression::Constant(0)),
            ..Default::default()
        }];
        assert_eq!(transitions.unwrap(), (expected, false, false));

        let transition = r"
name: transition
effect: {e0: '0'}
cost: '0'
direction: backward
";
        let transition = yaml_rust::YamlLoader::load_from_str(transition);
        assert!(transition.is_ok());
        let transition = transition.unwrap();
        assert_eq!(transition.len(), 1);
        let transition = &transition[0];
        let transitions = load_transitions_from_yaml(transition, &metadata, &registry, &cost_type);
        assert!(transitions.is_ok());
        let expected = vec![Transition {
            name: String::from("transition"),
            preconditions: Vec::new(),
            effect: Effect {
                element_effects: vec![(0, ElementExpression::Constant(0))],
                ..Default::default()
            },
            cost: CostExpression::Integer(IntegerExpression::Constant(0)),
            ..Default::default()
        }];
        assert_eq!(transitions.unwrap(), (expected, false, true));

        let transition = r"
name: transition
parameters:
        - name: e
          object: s0
preconditions:
        - (>= (f2 e0 e) 10)
        - (!= e 2)
effect:
        e0: e
        s0: (add e s0)
        p0: (push e p0)
        i0: '1'
        ir0: '2'
cost: (+ cost (f1 e))
";
        let transition = yaml_rust::YamlLoader::load_from_str(transition);
        assert!(transition.is_ok());
        let transition = transition.unwrap();
        assert_eq!(transition.len(), 1);
        let transition = &transition[0];
        let transitions = load_transitions_from_yaml(transition, &metadata, &registry, &cost_type);
        assert!(transitions.is_ok());
        let expected = vec![
            Transition {
                name: String::from("transition"),
                parameter_names: vec![String::from("e")],
                parameter_values: vec![0],
                elements_in_set_variable: vec![(0, 0)],
                elements_in_vector_variable: Vec::new(),
                preconditions: vec![GroundedCondition {
                    condition: Condition::ComparisonI(
                        ComparisonOperator::Ge,
                        Box::new(IntegerExpression::Table(Box::new(
                            NumericTableExpression::Table2D(
                                0,
                                ElementExpression::Variable(0),
                                ElementExpression::Constant(0),
                            ),
                        ))),
                        Box::new(IntegerExpression::Constant(10)),
                    ),
                    ..Default::default()
                }],
                effect: Effect {
                    set_effects: vec![(
                        0,
                        SetExpression::SetElementOperation(
                            SetElementOperator::Add,
                            ElementExpression::Constant(0),
                            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
                        ),
                    )],
                    vector_effects: vec![(
                        0,
                        VectorExpression::Push(
                            ElementExpression::Constant(0),
                            Box::new(VectorExpression::Reference(ReferenceExpression::Variable(
                                0,
                            ))),
                        ),
                    )],
                    element_effects: vec![(0, ElementExpression::Constant(0))],
                    integer_effects: vec![(0, IntegerExpression::Constant(1))],
                    integer_resource_effects: vec![(0, IntegerExpression::Constant(2))],
                    ..Default::default()
                },
                cost: CostExpression::Integer(IntegerExpression::BinaryOperation(
                    BinaryOperator::Add,
                    Box::new(IntegerExpression::Cost),
                    Box::new(IntegerExpression::Constant(10)),
                )),
            },
            Transition {
                name: String::from("transition"),
                parameter_names: vec![String::from("e")],
                parameter_values: vec![1],
                elements_in_set_variable: vec![(0, 1)],
                elements_in_vector_variable: Vec::new(),
                preconditions: vec![GroundedCondition {
                    condition: Condition::ComparisonI(
                        ComparisonOperator::Ge,
                        Box::new(IntegerExpression::Table(Box::new(
                            NumericTableExpression::Table2D(
                                0,
                                ElementExpression::Variable(0),
                                ElementExpression::Constant(1),
                            ),
                        ))),
                        Box::new(IntegerExpression::Constant(10)),
                    ),
                    ..Default::default()
                }],
                effect: Effect {
                    set_effects: vec![(
                        0,
                        SetExpression::SetElementOperation(
                            SetElementOperator::Add,
                            ElementExpression::Constant(1),
                            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
                        ),
                    )],
                    vector_effects: vec![(
                        0,
                        VectorExpression::Push(
                            ElementExpression::Constant(1),
                            Box::new(VectorExpression::Reference(ReferenceExpression::Variable(
                                0,
                            ))),
                        ),
                    )],
                    element_effects: vec![(0, ElementExpression::Constant(1))],
                    integer_effects: vec![(0, IntegerExpression::Constant(1))],
                    integer_resource_effects: vec![(0, IntegerExpression::Constant(2))],
                    ..Default::default()
                },
                cost: CostExpression::Integer(IntegerExpression::BinaryOperation(
                    BinaryOperator::Add,
                    Box::new(IntegerExpression::Cost),
                    Box::new(IntegerExpression::Constant(20)),
                )),
            },
        ];
        assert_eq!(transitions.unwrap(), (expected, false, false));
    }

    #[test]
    fn load_transitions_from_yaml_err() {
        let mut metadata = StateMetadata::default();
        let result = metadata.add_object_type(String::from("object"), 3);
        assert!(result.is_ok());
        let ob = result.unwrap();
        let result = metadata.add_set_variable(String::from("s0"), ob);
        assert!(result.is_ok());
        let result = metadata.add_set_variable(String::from("s1"), ob);
        assert!(result.is_ok());
        let result = metadata.add_set_variable(String::from("s2"), ob);
        assert!(result.is_ok());
        let result = metadata.add_set_variable(String::from("s3"), ob);
        assert!(result.is_ok());
        let result = metadata.add_vector_variable(String::from("p0"), ob);
        assert!(result.is_ok());
        let result = metadata.add_vector_variable(String::from("p1"), ob);
        assert!(result.is_ok());
        let result = metadata.add_vector_variable(String::from("p2"), ob);
        assert!(result.is_ok());
        let result = metadata.add_vector_variable(String::from("p3"), ob);
        assert!(result.is_ok());
        let result = metadata.add_element_variable(String::from("e0"), ob);
        assert!(result.is_ok());
        let result = metadata.add_element_variable(String::from("e1"), ob);
        assert!(result.is_ok());
        let result = metadata.add_element_variable(String::from("e2"), ob);
        assert!(result.is_ok());
        let result = metadata.add_element_variable(String::from("e3"), ob);
        assert!(result.is_ok());
        let result = metadata.add_element_resource_variable(String::from("er0"), ob, false);
        assert!(result.is_ok());
        let result = metadata.add_element_resource_variable(String::from("er1"), ob, false);
        assert!(result.is_ok());
        let result = metadata.add_element_resource_variable(String::from("er2"), ob, true);
        assert!(result.is_ok());
        let result = metadata.add_element_resource_variable(String::from("er3"), ob, false);
        assert!(result.is_ok());
        let result = metadata.add_integer_variable(String::from("i0"));
        assert!(result.is_ok());
        let result = metadata.add_integer_variable(String::from("i1"));
        assert!(result.is_ok());
        let result = metadata.add_integer_variable(String::from("i2"));
        assert!(result.is_ok());
        let result = metadata.add_integer_variable(String::from("i3"));
        assert!(result.is_ok());
        let result = metadata.add_continuous_variable(String::from("c0"));
        assert!(result.is_ok());
        let result = metadata.add_continuous_variable(String::from("c1"));
        assert!(result.is_ok());
        let result = metadata.add_continuous_variable(String::from("c2"));
        assert!(result.is_ok());
        let result = metadata.add_continuous_variable(String::from("c3"));
        assert!(result.is_ok());
        let result = metadata.add_integer_resource_variable(String::from("ir0"), false);
        assert!(result.is_ok());
        let result = metadata.add_integer_resource_variable(String::from("ir1"), false);
        assert!(result.is_ok());
        let result = metadata.add_integer_resource_variable(String::from("ir2"), true);
        assert!(result.is_ok());
        let result = metadata.add_integer_resource_variable(String::from("ir3"), false);
        assert!(result.is_ok());
        let result = metadata.add_continuous_resource_variable(String::from("cr0"), false);
        assert!(result.is_ok());
        let result = metadata.add_continuous_resource_variable(String::from("cr1"), false);
        assert!(result.is_ok());
        let result = metadata.add_continuous_resource_variable(String::from("cr2"), true);
        assert!(result.is_ok());
        let result = metadata.add_continuous_resource_variable(String::from("cr3"), false);
        assert!(result.is_ok());

        let mut registry = TableRegistry::default();
        let result = registry.add_table_1d(String::from("f1"), vec![10, 20, 30]);
        assert!(result.is_ok());
        let result =
            registry.add_table_2d(String::from("f2"), vec![vec![10, 20, 30], vec![40, 50, 60]]);
        assert!(result.is_ok());

        let cost_type = CostType::Integer;

        let transition = r"
parameters:
        - name: e
          object: s0
preconditions:
        - (>= (f2 e0 e) 10)
effect:
        e0: e
        s0: (add e s0)
        p0: e
        i0: '1'
        ir0: '2'
cost: (+ cost (f1 e))
";
        let transition = yaml_rust::YamlLoader::load_from_str(transition);
        assert!(transition.is_ok());
        let transition = transition.unwrap();
        assert_eq!(transition.len(), 1);
        let transition = &transition[0];
        let transitions = load_transitions_from_yaml(transition, &metadata, &registry, &cost_type);
        assert!(transitions.is_err());

        let transition = r"
name: transition
preconditions:
        - (>= (f2 e0 e) 10)
effect:
        e0: e
        s0: (add e s0)
        p0: (push e p0)
        i0: '1'
        ir0: '2'
cost: (+ cost (f1 e))
";
        let transition = yaml_rust::YamlLoader::load_from_str(transition);
        assert!(transition.is_ok());
        let transition = transition.unwrap();
        assert_eq!(transition.len(), 1);
        let transition = &transition[0];
        let transitions = load_transitions_from_yaml(transition, &metadata, &registry, &cost_type);
        assert!(transitions.is_err());

        let transition = r"
parameters:
        - name: e
          object: s0
preconditions:
        - (>= (f2 e0 e) 10)
effect:
        e0: e
        s0: (add e s0)
        p0: (push e p0)
        i0: '1'
        ir0: '2'
        ir5: '5'
cost: (+ cost (f1 e))
";
        let transition = yaml_rust::YamlLoader::load_from_str(transition);
        assert!(transition.is_ok());
        let transition = transition.unwrap();
        assert_eq!(transition.len(), 1);
        let transition = &transition[0];
        let transitions = load_transitions_from_yaml(transition, &metadata, &registry, &cost_type);
        assert!(transitions.is_err());

        let transition = r"
name: transition
effect: {e0: '0'}
cost: '0'
forced: fasle
";
        let transition = yaml_rust::YamlLoader::load_from_str(transition);
        assert!(transition.is_ok());
        let transition = transition.unwrap();
        assert_eq!(transition.len(), 1);
        let transition = &transition[0];
        let transitions = load_transitions_from_yaml(transition, &metadata, &registry, &cost_type);
        assert!(transitions.is_err());

        let transition = r"
name: transition
effect: {e0: '0'}
cost: '0'
direction: both
";
        let transition = yaml_rust::YamlLoader::load_from_str(transition);
        assert!(transition.is_ok());
        let transition = transition.unwrap();
        assert_eq!(transition.len(), 1);
        let transition = &transition[0];
        let transitions = load_transitions_from_yaml(transition, &metadata, &registry, &cost_type);
        assert!(transitions.is_err());
    }

    #[test]
    fn load_effect_from_yaml_ok() {
        let mut metadata = StateMetadata::default();
        let result = metadata.add_object_type(String::from("object"), 3);
        assert!(result.is_ok());
        let ob = result.unwrap();
        let result = metadata.add_set_variable(String::from("s0"), ob);
        assert!(result.is_ok());
        let result = metadata.add_set_variable(String::from("s1"), ob);
        assert!(result.is_ok());
        let result = metadata.add_set_variable(String::from("s2"), ob);
        assert!(result.is_ok());
        let result = metadata.add_set_variable(String::from("s3"), ob);
        assert!(result.is_ok());
        let result = metadata.add_vector_variable(String::from("p0"), ob);
        assert!(result.is_ok());
        let result = metadata.add_vector_variable(String::from("p1"), ob);
        assert!(result.is_ok());
        let result = metadata.add_vector_variable(String::from("p2"), ob);
        assert!(result.is_ok());
        let result = metadata.add_vector_variable(String::from("p3"), ob);
        assert!(result.is_ok());
        let result = metadata.add_element_variable(String::from("e0"), ob);
        assert!(result.is_ok());
        let result = metadata.add_element_variable(String::from("e1"), ob);
        assert!(result.is_ok());
        let result = metadata.add_element_variable(String::from("e2"), ob);
        assert!(result.is_ok());
        let result = metadata.add_element_variable(String::from("e3"), ob);
        assert!(result.is_ok());
        let result = metadata.add_element_resource_variable(String::from("er0"), ob, false);
        assert!(result.is_ok());
        let result = metadata.add_element_resource_variable(String::from("er1"), ob, false);
        assert!(result.is_ok());
        let result = metadata.add_element_resource_variable(String::from("er2"), ob, true);
        assert!(result.is_ok());
        let result = metadata.add_element_resource_variable(String::from("er3"), ob, false);
        assert!(result.is_ok());
        let result = metadata.add_integer_variable(String::from("i0"));
        assert!(result.is_ok());
        let result = metadata.add_integer_variable(String::from("i1"));
        assert!(result.is_ok());
        let result = metadata.add_integer_variable(String::from("i2"));
        assert!(result.is_ok());
        let result = metadata.add_integer_variable(String::from("i3"));
        assert!(result.is_ok());
        let result = metadata.add_continuous_variable(String::from("c0"));
        assert!(result.is_ok());
        let result = metadata.add_continuous_variable(String::from("c1"));
        assert!(result.is_ok());
        let result = metadata.add_continuous_variable(String::from("c2"));
        assert!(result.is_ok());
        let result = metadata.add_continuous_variable(String::from("c3"));
        assert!(result.is_ok());
        let result = metadata.add_integer_resource_variable(String::from("ir0"), false);
        assert!(result.is_ok());
        let result = metadata.add_integer_resource_variable(String::from("ir1"), false);
        assert!(result.is_ok());
        let result = metadata.add_integer_resource_variable(String::from("ir2"), true);
        assert!(result.is_ok());
        let result = metadata.add_integer_resource_variable(String::from("ir3"), false);
        assert!(result.is_ok());
        let result = metadata.add_continuous_resource_variable(String::from("cr0"), false);
        assert!(result.is_ok());
        let result = metadata.add_continuous_resource_variable(String::from("cr1"), false);
        assert!(result.is_ok());
        let result = metadata.add_continuous_resource_variable(String::from("cr2"), true);
        assert!(result.is_ok());
        let result = metadata.add_continuous_resource_variable(String::from("cr3"), false);
        assert!(result.is_ok());

        let mut registry = TableRegistry::default();
        let result = registry.add_table_1d(String::from("f1"), vec![10, 20, 30]);
        assert!(result.is_ok());
        let result =
            registry.add_table_2d(String::from("f2"), vec![vec![10, 20, 30], vec![40, 50, 60]]);
        assert!(result.is_ok());

        let mut parameters = FxHashMap::default();
        parameters.insert(String::from("e"), 0);

        let effect = r"
 e0: e
 s0: (add e s0)
 p0: (push e p0)
 i0: 1
 er0: 1
 ir0: '2'
 c0: 1.0
 cr0: '2.0'
";
        let effect = yaml_rust::YamlLoader::load_from_str(effect);
        assert!(effect.is_ok());
        let effect = effect.unwrap();
        assert_eq!(effect.len(), 1);
        let effect = &effect[0];
        let effect = load_effect_from_yaml(effect, &metadata, &registry, &parameters);
        assert!(effect.is_ok());
        let expected = Effect {
            set_effects: vec![(
                0,
                SetExpression::SetElementOperation(
                    SetElementOperator::Add,
                    ElementExpression::Constant(0),
                    Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
                ),
            )],
            vector_effects: vec![(
                0,
                VectorExpression::Push(
                    ElementExpression::Constant(0),
                    Box::new(VectorExpression::Reference(ReferenceExpression::Variable(
                        0,
                    ))),
                ),
            )],
            element_effects: vec![(0, ElementExpression::Constant(0))],
            integer_effects: vec![(0, IntegerExpression::Constant(1))],
            element_resource_effects: vec![(0, ElementExpression::Constant(1))],
            integer_resource_effects: vec![(0, IntegerExpression::Constant(2))],
            continuous_effects: vec![(0, ContinuousExpression::Constant(1.0))],
            continuous_resource_effects: vec![(0, ContinuousExpression::Constant(2.0))],
        };
        assert_eq!(effect.unwrap(), expected);
    }

    #[test]
    fn load_effect_from_yaml_err() {
        let mut metadata = StateMetadata::default();
        let result = metadata.add_object_type(String::from("object"), 3);
        assert!(result.is_ok());
        let ob = result.unwrap();
        let result = metadata.add_set_variable(String::from("s0"), ob);
        assert!(result.is_ok());
        let result = metadata.add_set_variable(String::from("s1"), ob);
        assert!(result.is_ok());
        let result = metadata.add_set_variable(String::from("s2"), ob);
        assert!(result.is_ok());
        let result = metadata.add_set_variable(String::from("s3"), ob);
        assert!(result.is_ok());
        let result = metadata.add_vector_variable(String::from("p0"), ob);
        assert!(result.is_ok());
        let result = metadata.add_vector_variable(String::from("p1"), ob);
        assert!(result.is_ok());
        let result = metadata.add_vector_variable(String::from("p2"), ob);
        assert!(result.is_ok());
        let result = metadata.add_vector_variable(String::from("p3"), ob);
        assert!(result.is_ok());
        let result = metadata.add_element_variable(String::from("e0"), ob);
        assert!(result.is_ok());
        let result = metadata.add_element_variable(String::from("e1"), ob);
        assert!(result.is_ok());
        let result = metadata.add_element_variable(String::from("e2"), ob);
        assert!(result.is_ok());
        let result = metadata.add_element_variable(String::from("e3"), ob);
        assert!(result.is_ok());
        let result = metadata.add_element_resource_variable(String::from("er0"), ob, false);
        assert!(result.is_ok());
        let result = metadata.add_element_resource_variable(String::from("er1"), ob, false);
        assert!(result.is_ok());
        let result = metadata.add_element_resource_variable(String::from("er2"), ob, true);
        assert!(result.is_ok());
        let result = metadata.add_element_resource_variable(String::from("er3"), ob, false);
        assert!(result.is_ok());
        let result = metadata.add_integer_variable(String::from("i0"));
        assert!(result.is_ok());
        let result = metadata.add_integer_variable(String::from("i1"));
        assert!(result.is_ok());
        let result = metadata.add_integer_variable(String::from("i2"));
        assert!(result.is_ok());
        let result = metadata.add_integer_variable(String::from("i3"));
        assert!(result.is_ok());
        let result = metadata.add_continuous_variable(String::from("c0"));
        assert!(result.is_ok());
        let result = metadata.add_continuous_variable(String::from("c1"));
        assert!(result.is_ok());
        let result = metadata.add_continuous_variable(String::from("c2"));
        assert!(result.is_ok());
        let result = metadata.add_continuous_variable(String::from("c3"));
        assert!(result.is_ok());
        let result = metadata.add_integer_resource_variable(String::from("ir0"), false);
        assert!(result.is_ok());
        let result = metadata.add_integer_resource_variable(String::from("ir1"), false);
        assert!(result.is_ok());
        let result = metadata.add_integer_resource_variable(String::from("ir2"), true);
        assert!(result.is_ok());
        let result = metadata.add_integer_resource_variable(String::from("ir3"), false);
        assert!(result.is_ok());
        let result = metadata.add_continuous_resource_variable(String::from("cr0"), false);
        assert!(result.is_ok());
        let result = metadata.add_continuous_resource_variable(String::from("cr1"), false);
        assert!(result.is_ok());
        let result = metadata.add_continuous_resource_variable(String::from("cr2"), true);
        assert!(result.is_ok());
        let result = metadata.add_continuous_resource_variable(String::from("cr3"), false);
        assert!(result.is_ok());

        let mut registry = TableRegistry::default();
        let result = registry.add_table_1d(String::from("f1"), vec![10, 20, 30]);
        assert!(result.is_ok());
        let result =
            registry.add_table_2d(String::from("f2"), vec![vec![10, 20, 30], vec![40, 50, 60]]);
        assert!(result.is_ok());

        let mut parameters = FxHashMap::default();
        parameters.insert(String::from("e"), 0);

        let effect = r"
 e0: f
 s0: (add e s0)
 p0: (push e p0)
 i0: '1'
 er0: -1
 ir0: '2'
 c0: '1.0'
 cr0: '2.0'
";
        let effect = yaml_rust::YamlLoader::load_from_str(effect);
        assert!(effect.is_ok());
        let effect = effect.unwrap();
        assert_eq!(effect.len(), 1);
        let effect = &effect[0];
        let effect = load_effect_from_yaml(effect, &metadata, &registry, &parameters);
        assert!(effect.is_err());

        let mut parameters = FxHashMap::default();
        parameters.insert(String::from("e"), 0);

        let effect = r"
 e0: f
 s0: (add e s0)
 p0: (push e p0)
 i0: '1'
 ir0: '2'
 c0: '1.0'
 cr0: '2.0'
";
        let effect = yaml_rust::YamlLoader::load_from_str(effect);
        assert!(effect.is_ok());
        let effect = effect.unwrap();
        assert_eq!(effect.len(), 1);
        let effect = &effect[0];
        let effect = load_effect_from_yaml(effect, &metadata, &registry, &parameters);
        assert!(effect.is_err());

        let effect = r"
 e0: e
 e4: e
 s0: (add e s0)
 p0: (push e p0)
 i0: '1'
 ir0: '2'
 c0: '1.0'
 cr0: '2.0'
";
        let effect = yaml_rust::YamlLoader::load_from_str(effect);
        assert!(effect.is_ok());
        let effect = effect.unwrap();
        assert_eq!(effect.len(), 1);
        let effect = &effect[0];
        let effect = load_effect_from_yaml(effect, &metadata, &registry, &parameters);
        assert!(effect.is_err());

        let effect = r"
 - e0: e
 - s0: (add e s0)
 - p0: (push e p0)
 - i0: '1'
 - ir0: '2'
 - c0: '1.0'
 - cr0: '2.0'
";
        let effect = yaml_rust::YamlLoader::load_from_str(effect);
        assert!(effect.is_ok());
        let effect = effect.unwrap();
        assert_eq!(effect.len(), 1);
        let effect = &effect[0];
        let effect = load_effect_from_yaml(effect, &metadata, &registry, &parameters);
        assert!(effect.is_err());
    }
}
