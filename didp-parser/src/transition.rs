use crate::expression;
use crate::expression_parser;
use crate::grounded_condition;
use crate::state;
use crate::table_registry;
use crate::variable::{Continuous, Integer, Numeric};
use crate::yaml_util;
use lazy_static::lazy_static;
use ordered_float::OrderedFloat;
use std::collections;
use std::error;
use std::fmt;
use std::rc::Rc;
use std::str;

#[derive(Debug, PartialEq, Clone, Default)]
pub struct Transition<T: Numeric> {
    pub name: String,
    pub elements_in_set_variable: Vec<(usize, usize)>,
    pub elements_in_vector_variable: Vec<(usize, usize)>,
    pub preconditions: Vec<grounded_condition::GroundedCondition>,
    pub set_effects: Vec<(usize, expression::SetExpression)>,
    pub vector_effects: Vec<(usize, expression::ElementExpression)>,
    pub element_effects: Vec<(usize, expression::ElementExpression)>,
    pub integer_effects: Vec<(usize, expression::NumericExpression<Integer>)>,
    pub continuous_effects: Vec<(usize, expression::NumericExpression<Continuous>)>,
    pub integer_resource_effects: Vec<(usize, expression::NumericExpression<Integer>)>,
    pub continuous_resource_effects: Vec<(usize, expression::NumericExpression<Continuous>)>,
    pub cost: expression::NumericExpression<T>,
}

impl<T: Numeric> Transition<T> {
    pub fn is_applicable(
        &self,
        state: &state::State,
        metadata: &state::StateMetadata,
        registry: &table_registry::TableRegistry,
    ) -> bool {
        for (i, v) in &self.elements_in_set_variable {
            if !state.signature_variables.set_variables[*i].contains(*v) {
                return false;
            }
        }
        for (i, v) in &self.elements_in_vector_variable {
            if !state.signature_variables.vector_variables[*i].contains(v) {
                return false;
            }
        }
        self.preconditions
            .iter()
            .all(|c| c.is_satisfied(state, metadata, registry).unwrap_or(true))
    }

    pub fn apply_effects(
        &self,
        state: &state::State,
        metadata: &state::StateMetadata,
        registry: &table_registry::TableRegistry,
    ) -> state::State {
        let len = state.signature_variables.set_variables.len();
        let mut set_variables = Vec::with_capacity(len);
        let mut i = 0;

        for e in &self.set_effects {
            while i < e.0 {
                set_variables.push(state.signature_variables.set_variables[i].clone());
                i += 1;
            }
            set_variables.push(e.1.eval(state, metadata, registry));
            i += 1;
        }
        while i < len {
            set_variables.push(state.signature_variables.set_variables[i].clone());
            i += 1;
        }

        let mut vector_variables = state.signature_variables.vector_variables.clone();
        for e in &self.vector_effects {
            vector_variables[e.0].push(e.1.eval(state, &registry.element_tables));
        }

        let mut element_variables = state.signature_variables.element_variables.clone();
        for e in &self.element_effects {
            element_variables[e.0] = e.1.eval(state, &registry.element_tables);
        }

        let mut integer_variables = state.signature_variables.integer_variables.clone();
        for e in &self.integer_effects {
            integer_variables[e.0] = e.1.eval(state, metadata, registry);
        }

        let mut continuous_variables = state.signature_variables.continuous_variables.clone();
        for e in &self.continuous_effects {
            continuous_variables[e.0] = OrderedFloat(e.1.eval(state, metadata, registry));
        }

        let mut integer_resource_variables = state.resource_variables.integer_variables.clone();
        for e in &self.integer_resource_effects {
            integer_resource_variables[e.0] = e.1.eval(state, metadata, registry);
        }

        let mut continuous_resource_variables =
            state.resource_variables.continuous_variables.clone();
        for e in &self.continuous_resource_effects {
            continuous_resource_variables[e.0] = e.1.eval(state, metadata, registry);
        }

        let stage = state.stage + 1;

        state::State {
            signature_variables: {
                Rc::new(state::SignatureVariables {
                    set_variables,
                    vector_variables,
                    element_variables,
                    integer_variables,
                    continuous_variables,
                })
            },
            resource_variables: state::ResourceVariables {
                integer_variables: integer_resource_variables,
                continuous_variables: continuous_resource_variables,
            },
            stage,
        }
    }

    pub fn eval_cost(
        &self,
        cost: T,
        state: &state::State,
        metadata: &state::StateMetadata,
        registry: &table_registry::TableRegistry,
    ) -> T {
        self.cost.eval_cost(cost, state, metadata, registry)
    }
}

type TranstionsWithDirection<T> = (Vec<Transition<T>>, bool);

pub fn load_transitions_from_yaml<T: Numeric>(
    value: &yaml_rust::Yaml,
    metadata: &state::StateMetadata,
    registry: &table_registry::TableRegistry,
) -> Result<TranstionsWithDirection<T>, Box<dyn error::Error>>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    lazy_static! {
        static ref PARAMETERS_KEY: yaml_rust::Yaml = yaml_rust::Yaml::from_str("parameters");
        static ref PRECONDITIONS_KEY: yaml_rust::Yaml = yaml_rust::Yaml::from_str("preconditions");
        static ref DIRECTION_KEY: yaml_rust::Yaml = yaml_rust::Yaml::from_str("direction");
    }
    let map = yaml_util::get_map(value)?;
    let lifted_name = yaml_util::get_string_by_key(map, "name")?;

    let (
        parameters_array,
        elements_in_set_variable_array,
        elements_in_vector_variable_array,
        parameter_names,
    ) = match map.get(&PARAMETERS_KEY) {
        Some(value) => {
            let result = metadata.ground_parameters_from_yaml(value)?;
            let array = yaml_util::get_array(value)?;
            let mut parameter_names = Vec::with_capacity(array.len());
            for map in array {
                let map = yaml_util::get_map(map)?;
                let value = yaml_util::get_string_by_key(&map, "name")?;
                parameter_names.push(value);
            }
            (result.0, result.1, result.2, parameter_names)
        }
        None => (
            vec![collections::HashMap::new()],
            vec![vec![]],
            vec![vec![]],
            vec![],
        ),
    };

    let lifted_effects = yaml_util::get_map_by_key(map, "effects")?;
    let lifted_cost = yaml_util::get_string_by_key(map, "cost")?;
    let lifted_preconditions = match map.get(&PRECONDITIONS_KEY) {
        Some(lifted_preconditions) => Some(yaml_util::get_array(lifted_preconditions)?),
        None => None,
    };
    let mut transitions = Vec::with_capacity(parameters_array.len());
    'outer: for ((parameters, elements_in_set_variable), elements_in_vector_variable) in
        parameters_array
            .into_iter()
            .zip(elements_in_set_variable_array.into_iter())
            .zip(elements_in_vector_variable_array.into_iter())
    {
        let mut name = lifted_name.clone();
        for parameter_name in &parameter_names {
            name += format!(" {}:{}", parameter_name, parameters[parameter_name]).as_str();
        }
        let preconditions = match lifted_preconditions {
            Some(lifted_preconditions) => {
                let mut preconditions = Vec::with_capacity(lifted_preconditions.len());
                for condition in lifted_preconditions {
                    let conditions =
                        grounded_condition::GroundedCondition::load_grounded_conditions_from_yaml(
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
        let mut set_effects = Vec::new();
        let mut vector_effects = Vec::new();
        let mut element_effects = Vec::new();
        let mut integer_effects = Vec::new();
        let mut continuous_effects = Vec::new();
        let mut integer_resource_effects = Vec::new();
        let mut continuous_resource_effects = Vec::new();
        for (variable, effect) in lifted_effects {
            let effect = yaml_util::get_string(effect)?;
            let variable = yaml_util::get_string(variable)?;
            if let Some(i) = metadata.name_to_set_variable.get(&variable) {
                let effect = expression_parser::parse_set(effect, metadata, &parameters)?;
                set_effects.push((*i, effect));
            } else if let Some(i) = metadata.name_to_vector_variable.get(&variable) {
                let effect = expression_parser::parse_element(effect, metadata, &parameters)?;
                vector_effects.push((*i, effect));
            } else if let Some(i) = metadata.name_to_element_variable.get(&variable) {
                let effect = expression_parser::parse_element(effect, metadata, &parameters)?;
                element_effects.push((*i, effect));
            } else if let Some(i) = metadata.name_to_integer_variable.get(&variable) {
                let effect = expression_parser::parse_numeric::<Integer>(
                    effect,
                    metadata,
                    registry,
                    &parameters,
                )?;
                integer_effects.push((*i, effect.simplify(registry)));
            } else if let Some(i) = metadata.name_to_integer_resource_variable.get(&variable) {
                let effect = expression_parser::parse_numeric::<Integer>(
                    effect,
                    metadata,
                    registry,
                    &parameters,
                )?;
                integer_resource_effects.push((*i, effect.simplify(registry)));
            } else if let Some(i) = metadata.name_to_continuous_variable.get(&variable) {
                let effect = expression_parser::parse_numeric::<Continuous>(
                    effect,
                    metadata,
                    registry,
                    &parameters,
                )?;
                continuous_effects.push((*i, effect.simplify(registry)));
            } else if let Some(i) = metadata.name_to_continuous_resource_variable.get(&variable) {
                let effect = expression_parser::parse_numeric::<Continuous>(
                    effect,
                    metadata,
                    registry,
                    &parameters,
                )?;
                continuous_resource_effects.push((*i, effect.simplify(registry)));
            } else {
                return Err(yaml_util::YamlContentErr::new(format!(
                    "no such variable `{}`",
                    variable
                ))
                .into());
            }
        }
        let cost =
            expression_parser::parse_numeric(lifted_cost.clone(), metadata, registry, &parameters)?;
        let cost = cost.simplify(registry);

        transitions.push(Transition {
            name,
            elements_in_set_variable,
            elements_in_vector_variable,
            preconditions,
            set_effects,
            vector_effects,
            element_effects,
            integer_effects,
            continuous_effects,
            integer_resource_effects,
            continuous_resource_effects,
            cost,
        })
    }
    let backward = match map.get(&DIRECTION_KEY) {
        Some(direction) => {
            let direction = yaml_util::get_string(direction)?;
            match &direction[..] {
                "forward" => false,
                "backward" => true,
                _ => {
                    return Err(yaml_util::YamlContentErr::new(format!(
                        "no such direction `{}`",
                        direction
                    ))
                    .into())
                }
            }
        }
        None => false,
    };

    Ok((transitions, backward))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::table;
    use crate::table_data;
    use crate::variable::Set;
    use expression::*;
    use std::collections::HashMap;

    fn generate_metadata() -> state::StateMetadata {
        let object_names = vec![String::from("object")];
        let object_numbers = vec![3];
        let mut name_to_object = HashMap::new();
        name_to_object.insert("object".to_string(), 0);

        let set_variable_names = vec![
            String::from("s0"),
            String::from("s1"),
            String::from("s2"),
            String::from("s3"),
        ];
        let mut name_to_set_variable = HashMap::new();
        name_to_set_variable.insert(String::from("s0"), 0);
        name_to_set_variable.insert(String::from("s1"), 1);
        name_to_set_variable.insert(String::from("s2"), 2);
        name_to_set_variable.insert(String::from("s3"), 3);
        let set_variable_to_object = vec![0, 0, 0, 0];

        let vector_variable_names = vec![
            "p0".to_string(),
            "p1".to_string(),
            "p2".to_string(),
            "p3".to_string(),
        ];
        let mut name_to_vector_variable = HashMap::new();
        name_to_vector_variable.insert("p0".to_string(), 0);
        name_to_vector_variable.insert("p1".to_string(), 1);
        name_to_vector_variable.insert("p2".to_string(), 2);
        name_to_vector_variable.insert("p3".to_string(), 3);
        let vector_variable_to_object = vec![0, 0, 0, 0];

        let element_variable_names = vec![
            "e0".to_string(),
            "e1".to_string(),
            "e2".to_string(),
            "e3".to_string(),
        ];
        let mut name_to_element_variable = HashMap::new();
        name_to_element_variable.insert("e0".to_string(), 0);
        name_to_element_variable.insert("e1".to_string(), 1);
        name_to_element_variable.insert("e2".to_string(), 2);
        name_to_element_variable.insert("e3".to_string(), 3);
        let element_variable_to_object = vec![0, 0, 0, 0];

        let integer_variable_names = vec![
            "i0".to_string(),
            "i1".to_string(),
            "i2".to_string(),
            "i3".to_string(),
        ];
        let mut name_to_integer_variable = HashMap::new();
        name_to_integer_variable.insert("i0".to_string(), 0);
        name_to_integer_variable.insert("i1".to_string(), 1);
        name_to_integer_variable.insert("i2".to_string(), 2);
        name_to_integer_variable.insert("i3".to_string(), 3);

        let integer_resource_variable_names = vec![
            "ir0".to_string(),
            "ir1".to_string(),
            "ir2".to_string(),
            "ir3".to_string(),
        ];
        let mut name_to_integer_resource_variable = HashMap::new();
        name_to_integer_resource_variable.insert("ir0".to_string(), 0);
        name_to_integer_resource_variable.insert("ir1".to_string(), 1);
        name_to_integer_resource_variable.insert("ir2".to_string(), 2);
        name_to_integer_resource_variable.insert("ir3".to_string(), 3);

        let continuous_variable_names = vec![
            "c0".to_string(),
            "c1".to_string(),
            "c2".to_string(),
            "c3".to_string(),
        ];
        let mut name_to_continuous_variable = HashMap::new();
        name_to_continuous_variable.insert("c0".to_string(), 0);
        name_to_continuous_variable.insert("c1".to_string(), 1);
        name_to_continuous_variable.insert("c2".to_string(), 2);
        name_to_continuous_variable.insert("c3".to_string(), 3);

        let continuous_resource_variable_names = vec![
            "cr0".to_string(),
            "cr1".to_string(),
            "cr2".to_string(),
            "cr3".to_string(),
        ];
        let mut name_to_continuous_resource_variable = HashMap::new();
        name_to_continuous_resource_variable.insert("cr0".to_string(), 0);
        name_to_continuous_resource_variable.insert("cr1".to_string(), 1);
        name_to_continuous_resource_variable.insert("cr2".to_string(), 2);
        name_to_continuous_resource_variable.insert("cr3".to_string(), 3);

        state::StateMetadata {
            object_names,
            name_to_object,
            object_numbers,
            set_variable_names,
            name_to_set_variable,
            set_variable_to_object,
            vector_variable_names,
            name_to_vector_variable,
            vector_variable_to_object,
            element_variable_names,
            name_to_element_variable,
            element_variable_to_object,
            integer_variable_names,
            name_to_integer_variable,
            continuous_variable_names,
            name_to_continuous_variable,
            integer_resource_variable_names,
            name_to_integer_resource_variable,
            integer_less_is_better: vec![false, false, true, false],
            continuous_resource_variable_names,
            name_to_continuous_resource_variable,
            continuous_less_is_better: vec![false, false, true, false],
        }
    }

    fn generate_registry() -> table_registry::TableRegistry {
        let tables_1d = vec![table::Table1D::new(vec![10, 20, 30])];
        let mut name_to_table_1d = HashMap::new();
        name_to_table_1d.insert(String::from("f1"), 0);

        let tables_2d = vec![table::Table2D::new(vec![
            vec![10, 20, 30],
            vec![40, 50, 60],
        ])];
        let mut name_to_table_2d = HashMap::new();
        name_to_table_2d.insert(String::from("f2"), 0);

        table_registry::TableRegistry {
            integer_tables: table_data::TableData {
                tables_1d,
                name_to_table_1d,
                tables_2d,
                name_to_table_2d,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn generate_state() -> state::State {
        let mut set1 = Set::with_capacity(3);
        set1.insert(0);
        set1.insert(2);
        let mut set2 = Set::with_capacity(3);
        set2.insert(0);
        set2.insert(1);
        state::State {
            signature_variables: Rc::new(state::SignatureVariables {
                set_variables: vec![set1, set2],
                vector_variables: vec![vec![0, 2], vec![1, 2]],
                element_variables: vec![1, 2],
                integer_variables: vec![1, 2, 3],
                continuous_variables: vec![OrderedFloat(1.0), OrderedFloat(2.0), OrderedFloat(3.0)],
            }),
            resource_variables: state::ResourceVariables {
                integer_variables: vec![4, 5, 6],
                continuous_variables: vec![4.0, 5.0, 6.0],
            },
            stage: 0,
        }
    }

    #[test]
    fn applicable() {
        let state = generate_state();
        let registry = generate_registry();
        let metadata = generate_metadata();
        let set_condition = grounded_condition::GroundedCondition {
            condition: Condition::Set(SetCondition::IsIn(
                ElementExpression::Constant(0),
                SetExpression::SetVariable(0),
            )),
            ..Default::default()
        };
        let numeric_condition = grounded_condition::GroundedCondition {
            condition: Condition::Comparison(Box::new(Comparison::ComparisonII(
                ComparisonOperator::Ge,
                NumericExpression::IntegerVariable(0),
                NumericExpression::Constant(1),
            ))),
            ..Default::default()
        };

        let transition = Transition {
            name: String::from(""),
            preconditions: vec![set_condition, numeric_condition],
            cost: NumericExpression::Constant(0),
            ..Default::default()
        };
        assert!(transition.is_applicable(&state, &metadata, &registry));

        let transition = Transition {
            name: String::from(""),
            elements_in_set_variable: vec![(0, 0), (1, 1)],
            elements_in_vector_variable: vec![(0, 0), (1, 2)],
            cost: NumericExpression::Constant(0),
            ..Default::default()
        };
        assert!(transition.is_applicable(&state, &metadata, &registry));
    }

    #[test]
    fn not_applicable() {
        let state = generate_state();
        let registry = generate_registry();
        let metadata = generate_metadata();
        let set_condition = grounded_condition::GroundedCondition {
            condition: Condition::Set(SetCondition::IsIn(
                ElementExpression::Constant(0),
                SetExpression::SetVariable(0),
            )),
            ..Default::default()
        };
        let numeric_condition = grounded_condition::GroundedCondition {
            condition: Condition::Comparison(Box::new(Comparison::ComparisonII(
                ComparisonOperator::Le,
                NumericExpression::IntegerVariable(0),
                NumericExpression::Constant(1),
            ))),
            ..Default::default()
        };

        let transition = Transition {
            name: String::from(""),
            preconditions: vec![set_condition, numeric_condition],
            cost: NumericExpression::Constant(0),
            ..Default::default()
        };
        assert!(transition.is_applicable(&state, &metadata, &registry));

        let transition = Transition {
            name: String::from(""),
            elements_in_set_variable: vec![(0, 1), (1, 1)],
            elements_in_vector_variable: vec![(0, 0), (1, 2)],
            cost: NumericExpression::Constant(0),
            ..Default::default()
        };
        assert!(!transition.is_applicable(&state, &metadata, &registry));

        let transition = Transition {
            name: String::from(""),
            elements_in_set_variable: vec![(0, 1), (1, 1)],
            elements_in_vector_variable: vec![(0, 1), (1, 2)],
            cost: NumericExpression::Constant(0),
            ..Default::default()
        };
        assert!(!transition.is_applicable(&state, &metadata, &registry));
    }

    #[test]
    fn appy_effects() {
        let state = generate_state();
        let registry = generate_registry();
        let metadata = generate_metadata();
        let set_effect1 = SetExpression::SetElementOperation(
            SetElementOperator::Add,
            Box::new(SetExpression::SetVariable(0)),
            ElementExpression::Constant(1),
        );
        let set_effect2 = SetExpression::SetElementOperation(
            SetElementOperator::Remove,
            Box::new(SetExpression::SetVariable(1)),
            ElementExpression::Constant(0),
        );
        let vector_effect1 = ElementExpression::Constant(1);
        let vector_effect2 = ElementExpression::Constant(0);
        let element_effect1 = ElementExpression::Constant(2);
        let element_effect2 = ElementExpression::Constant(1);
        let integer_effect1 = NumericExpression::NumericOperation(
            NumericOperator::Subtract,
            Box::new(NumericExpression::IntegerVariable(0)),
            Box::new(NumericExpression::Constant(1)),
        );
        let integer_effect2 = NumericExpression::NumericOperation(
            NumericOperator::Multiply,
            Box::new(NumericExpression::IntegerVariable(1)),
            Box::new(NumericExpression::Constant(2)),
        );
        let continuous_effect1 = NumericExpression::NumericOperation(
            NumericOperator::Subtract,
            Box::new(NumericExpression::ContinuousVariable(0)),
            Box::new(NumericExpression::Constant(1.0)),
        );
        let continuous_effect2 = NumericExpression::NumericOperation(
            NumericOperator::Multiply,
            Box::new(NumericExpression::ContinuousVariable(1)),
            Box::new(NumericExpression::Constant(2.0)),
        );
        let integer_resource_effect1 = NumericExpression::NumericOperation(
            NumericOperator::Add,
            Box::new(NumericExpression::IntegerResourceVariable(0)),
            Box::new(NumericExpression::Constant(1)),
        );
        let integer_resource_effect2 = NumericExpression::NumericOperation(
            NumericOperator::Divide,
            Box::new(NumericExpression::IntegerResourceVariable(1)),
            Box::new(NumericExpression::Constant(2)),
        );
        let continuous_resource_effect1 = NumericExpression::NumericOperation(
            NumericOperator::Add,
            Box::new(NumericExpression::ContinuousResourceVariable(0)),
            Box::new(NumericExpression::Constant(1.0)),
        );
        let continuous_resource_effect2 = NumericExpression::NumericOperation(
            NumericOperator::Divide,
            Box::new(NumericExpression::ContinuousResourceVariable(1)),
            Box::new(NumericExpression::Constant(2.0)),
        );
        let transition = Transition {
            name: String::from(""),
            set_effects: vec![(0, set_effect1), (1, set_effect2)],
            vector_effects: vec![(0, vector_effect1), (1, vector_effect2)],
            element_effects: vec![(0, element_effect1), (1, element_effect2)],
            integer_effects: vec![(0, integer_effect1), (1, integer_effect2)],
            continuous_effects: vec![(0, continuous_effect1), (1, continuous_effect2)],
            integer_resource_effects: vec![
                (0, integer_resource_effect1),
                (1, integer_resource_effect2),
            ],
            continuous_resource_effects: vec![
                (0, continuous_resource_effect1),
                (1, continuous_resource_effect2),
            ],
            cost: NumericExpression::NumericOperation(
                NumericOperator::Add,
                Box::new(NumericExpression::Cost),
                Box::new(NumericExpression::Constant(1)),
            ),
            ..Default::default()
        };

        let mut set1 = Set::with_capacity(3);
        set1.insert(0);
        set1.insert(1);
        set1.insert(2);
        let mut set2 = Set::with_capacity(3);
        set2.insert(1);
        let expected = state::State {
            signature_variables: Rc::new(state::SignatureVariables {
                set_variables: vec![set1, set2],
                vector_variables: vec![vec![0, 2, 1], vec![1, 2, 0]],
                element_variables: vec![2, 1],
                integer_variables: vec![0, 4, 3],
                continuous_variables: vec![OrderedFloat(0.0), OrderedFloat(4.0), OrderedFloat(3.0)],
            }),
            resource_variables: state::ResourceVariables {
                integer_variables: vec![5, 2, 6],
                continuous_variables: vec![5.0, 2.5, 6.0],
            },
            stage: 1,
        };
        let successor = transition.apply_effects(&state, &metadata, &registry);
        assert_eq!(successor, expected);
    }

    #[test]
    fn eval_cost() {
        let state = generate_state();
        let registry = generate_registry();
        let metadata = generate_metadata();

        let transition = Transition {
            cost: NumericExpression::NumericOperation(
                NumericOperator::Add,
                Box::new(NumericExpression::Cost),
                Box::new(NumericExpression::Constant(1)),
            ),
            ..Default::default()
        };
        assert_eq!(transition.eval_cost(0, &state, &metadata, &registry), 1);
    }

    #[test]
    fn load_transitions_from_yaml_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();

        let transition = r"
name: transition
preconditions: [(>= (f2 0 1) 10)]
effects: {e0: '0'}
cost: '0'
";
        let transition = yaml_rust::YamlLoader::load_from_str(transition);
        assert!(transition.is_ok());
        let transition = transition.unwrap();
        assert_eq!(transition.len(), 1);
        let transition = &transition[0];
        let transitions = load_transitions_from_yaml(transition, &metadata, &registry);
        let expected = vec![Transition {
            name: String::from("transition"),
            preconditions: Vec::new(),
            element_effects: vec![(0, ElementExpression::Constant(0))],
            cost: NumericExpression::Constant(0),
            ..Default::default()
        }];
        assert_eq!(transitions.unwrap(), (expected, false));

        let transition = r"
name: transition
effects: {e0: '0'}
cost: '0'
";
        let transition = yaml_rust::YamlLoader::load_from_str(transition);
        assert!(transition.is_ok());
        let transition = transition.unwrap();
        assert_eq!(transition.len(), 1);
        let transition = &transition[0];
        let transitions = load_transitions_from_yaml(transition, &metadata, &registry);
        assert!(transitions.is_ok());
        let expected = vec![Transition {
            name: String::from("transition"),
            preconditions: Vec::new(),
            element_effects: vec![(0, ElementExpression::Constant(0))],
            cost: NumericExpression::Constant(0),
            ..Default::default()
        }];
        assert_eq!(transitions.unwrap(), (expected, false));

        let transition = r"
name: transition
parameters:
        - name: e
          object: s0
preconditions:
        - (>= (f2 e0 e) 10)
        - (is_not e 2)
effects:
        e0: e
        s0: (+ s0 e)
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
        let transitions = load_transitions_from_yaml(transition, &metadata, &registry);
        assert!(transitions.is_ok());
        let expected = vec![
            Transition {
                name: String::from("transition e:0"),
                elements_in_set_variable: vec![(0, 0)],
                elements_in_vector_variable: Vec::new(),
                preconditions: vec![grounded_condition::GroundedCondition {
                    condition: Condition::Comparison(Box::new(Comparison::ComparisonII(
                        ComparisonOperator::Ge,
                        NumericExpression::IntegerTable(NumericTableExpression::Table2D(
                            0,
                            ElementExpression::Variable(0),
                            ElementExpression::Constant(0),
                        )),
                        NumericExpression::Constant(10),
                    ))),
                    ..Default::default()
                }],
                set_effects: vec![(
                    0,
                    SetExpression::SetElementOperation(
                        SetElementOperator::Add,
                        Box::new(SetExpression::SetVariable(0)),
                        ElementExpression::Constant(0),
                    ),
                )],
                vector_effects: vec![(0, ElementExpression::Constant(0))],
                element_effects: vec![(0, ElementExpression::Constant(0))],
                integer_effects: vec![(0, NumericExpression::Constant(1))],
                integer_resource_effects: vec![(0, NumericExpression::Constant(2))],
                cost: NumericExpression::NumericOperation(
                    NumericOperator::Add,
                    Box::new(NumericExpression::Cost),
                    Box::new(NumericExpression::Constant(10)),
                ),
                ..Default::default()
            },
            Transition {
                name: String::from("transition e:1"),
                elements_in_set_variable: vec![(0, 1)],
                elements_in_vector_variable: Vec::new(),
                preconditions: vec![grounded_condition::GroundedCondition {
                    condition: Condition::Comparison(Box::new(Comparison::ComparisonII(
                        ComparisonOperator::Ge,
                        NumericExpression::IntegerTable(NumericTableExpression::Table2D(
                            0,
                            ElementExpression::Variable(0),
                            ElementExpression::Constant(1),
                        )),
                        NumericExpression::Constant(10),
                    ))),
                    ..Default::default()
                }],
                set_effects: vec![(
                    0,
                    SetExpression::SetElementOperation(
                        SetElementOperator::Add,
                        Box::new(SetExpression::SetVariable(0)),
                        ElementExpression::Constant(1),
                    ),
                )],
                vector_effects: vec![(0, ElementExpression::Constant(1))],
                element_effects: vec![(0, ElementExpression::Constant(1))],
                integer_effects: vec![(0, NumericExpression::Constant(1))],
                integer_resource_effects: vec![(0, NumericExpression::Constant(2))],
                cost: NumericExpression::NumericOperation(
                    NumericOperator::Add,
                    Box::new(NumericExpression::Cost),
                    Box::new(NumericExpression::Constant(20)),
                ),
                ..Default::default()
            },
        ];
        assert_eq!(transitions.unwrap(), (expected, false));
    }

    #[test]
    fn load_transitions_from_yaml_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();

        let transition = r"
parameters:
        - name: e
          object: s0
preconditions:
        - (>= (f2 e0 e) 10)
effects:
        e0: e
        s0: (+ s0 e)
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
        let transitions = load_transitions_from_yaml::<Integer>(transition, &metadata, &registry);
        assert!(transitions.is_err());

        let transition = r"
name: transition
parameters:
        - name: e
          object: s0
preconditions:
        - (>= (f2 e0 e) 10)
cost: (+ cost (f1 e))
";
        let transition = yaml_rust::YamlLoader::load_from_str(transition);
        assert!(transition.is_ok());
        let transition = transition.unwrap();
        assert_eq!(transition.len(), 1);
        let transition = &transition[0];
        let transitions = load_transitions_from_yaml::<Integer>(transition, &metadata, &registry);
        assert!(transitions.is_err());

        let transition = r"
name: transition
parameters:
        - name: e
          object: s0
preconditions:
        - (>= (f2 e0 e) 10)
effects:
        e0: e
        s0: (+ s0 e)
        p0: e
        i0: '1'
        ir0: '2'
";
        let transition = yaml_rust::YamlLoader::load_from_str(transition);
        assert!(transition.is_ok());
        let transition = transition.unwrap();
        assert_eq!(transition.len(), 1);
        let transition = &transition[0];
        let transitions = load_transitions_from_yaml::<Integer>(transition, &metadata, &registry);
        assert!(transitions.is_err());

        let transition = r"
name: transition
preconditions:
        - (>= (f2 e0 e) 10)
effects:
        e0: e
        s0: (+ s0 e)
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
        let transitions = load_transitions_from_yaml::<Integer>(transition, &metadata, &registry);
        assert!(transitions.is_err());

        let transition = r"
parameters:
        - name: e
          object: s0
preconditions:
        - (>= (f2 e0 e) 10)
effects:
        e0: e
        s0: (+ s0 e)
        p0: e
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
        let transitions = load_transitions_from_yaml::<Integer>(transition, &metadata, &registry);
        assert!(transitions.is_err());
    }
}
