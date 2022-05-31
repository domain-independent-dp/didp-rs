use crate::effect;
use crate::expression;
use crate::expression_parser::ParseNumericExpression;
use crate::grounded_condition;
use crate::state;
use crate::state::DPState;
use crate::table_registry;
use crate::variable::{Element, Numeric};
use crate::yaml_util;
use lazy_static::lazy_static;
use rustc_hash::FxHashMap;
use std::error;
use std::fmt;
use std::str;

#[derive(Debug, PartialEq, Clone, Default)]
pub struct Transition<T: Numeric> {
    pub name: String,
    pub parameter_names: Vec<String>,
    pub parameter_values: Vec<Element>,
    pub elements_in_set_variable: Vec<(usize, Element)>,
    pub elements_in_vector_variable: Vec<(usize, Element)>,
    pub preconditions: Vec<grounded_condition::GroundedCondition>,
    pub effect: effect::Effect,
    pub cost: expression::NumericExpression<T>,
}

impl<T: Numeric> Transition<T> {
    pub fn is_applicable<S: DPState>(
        &self,
        state: &S,
        registry: &table_registry::TableRegistry,
    ) -> bool {
        for (i, v) in &self.elements_in_set_variable {
            if !state.get_set_variable(*i).contains(*v) {
                return false;
            }
        }
        for (i, v) in &self.elements_in_vector_variable {
            if !state.get_vector_variable(*i).contains(v) {
                return false;
            }
        }
        self.preconditions
            .iter()
            .all(|c| c.is_satisfied(state, registry).unwrap_or(true))
    }

    pub fn apply<S: DPState>(&self, state: &S, registry: &table_registry::TableRegistry) -> S {
        state.apply_effect(&self.effect, registry)
    }

    pub fn apply_in_place<S: DPState>(
        &self,
        state: &mut S,
        registry: &table_registry::TableRegistry,
    ) {
        state.apply_effect_in_place(&self.effect, registry)
    }

    pub fn eval_cost<S: DPState>(
        &self,
        cost: T,
        state: &S,
        registry: &table_registry::TableRegistry,
    ) -> T {
        self.cost.eval_cost(cost, state, registry)
    }

    pub fn get_full_name(&self) -> String {
        let mut full_name = self.name.clone();
        for (name, value) in self
            .parameter_names
            .iter()
            .zip(self.parameter_values.iter())
        {
            full_name += format!(" {}:{}", name, value).as_str();
        }
        full_name
    }
}

type TransitionsWithFlags<T> = (Vec<Transition<T>>, bool, bool);

pub fn load_transitions_from_yaml<T: Numeric + ParseNumericExpression>(
    value: &yaml_rust::Yaml,
    metadata: &state::StateMetadata,
    registry: &table_registry::TableRegistry,
) -> Result<TransitionsWithFlags<T>, Box<dyn error::Error>>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    lazy_static! {
        static ref PARAMETERS_KEY: yaml_rust::Yaml = yaml_rust::Yaml::from_str("parameters");
        static ref PRECONDITIONS_KEY: yaml_rust::Yaml = yaml_rust::Yaml::from_str("preconditions");
        static ref DIRECTION_KEY: yaml_rust::Yaml = yaml_rust::Yaml::from_str("direction");
        static ref FORCED_KEY: yaml_rust::Yaml = yaml_rust::Yaml::from_str("forced");
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
                let value = yaml_util::get_string_by_key(map, "name")?;
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

    let effect = yaml_util::get_yaml_by_key(map, "effect")?;
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
        let parameter_values = parameter_names
            .iter()
            .map(|name| parameters[name])
            .collect();
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
        let effect = effect::Effect::load_from_yaml(effect, metadata, registry, &parameters)?;
        let cost = T::parse_expression(lifted_cost.clone(), metadata, registry, &parameters)?;
        let cost = cost.simplify(registry);

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
    let forced = match map.get(&FORCED_KEY) {
        Some(forced) => yaml_util::get_bool(forced)?,
        None => false,
    };

    Ok((transitions, forced, backward))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::table;
    use crate::table_data;
    use crate::variable::*;
    use expression::*;
    use rustc_hash::FxHashMap;

    fn generate_metadata() -> state::StateMetadata {
        let object_names = vec![String::from("object")];
        let object_numbers = vec![3];
        let mut name_to_object = FxHashMap::default();
        name_to_object.insert("object".to_string(), 0);

        let set_variable_names = vec![
            String::from("s0"),
            String::from("s1"),
            String::from("s2"),
            String::from("s3"),
        ];
        let mut name_to_set_variable = FxHashMap::default();
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
        let mut name_to_vector_variable = FxHashMap::default();
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
        let mut name_to_element_variable = FxHashMap::default();
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
        let mut name_to_integer_variable = FxHashMap::default();
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
        let mut name_to_integer_resource_variable = FxHashMap::default();
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
        let mut name_to_continuous_variable = FxHashMap::default();
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
        let mut name_to_continuous_resource_variable = FxHashMap::default();
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
        let mut name_to_table_1d = FxHashMap::default();
        name_to_table_1d.insert(String::from("f1"), 0);

        let tables_2d = vec![table::Table2D::new(vec![
            vec![10, 20, 30],
            vec![40, 50, 60],
        ])];
        let mut name_to_table_2d = FxHashMap::default();
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
            signature_variables: state::SignatureVariables {
                set_variables: vec![set1, set2],
                vector_variables: vec![vec![0, 2], vec![1, 2]],
                element_variables: vec![1, 2],
                integer_variables: vec![1, 2, 3],
                continuous_variables: vec![1.0, 2.0, 3.0],
            },
            resource_variables: state::ResourceVariables {
                integer_variables: vec![4, 5, 6],
                continuous_variables: vec![4.0, 5.0, 6.0],
            },
        }
    }

    #[test]
    fn applicable() {
        let state = generate_state();
        let registry = generate_registry();
        let set_condition = grounded_condition::GroundedCondition {
            condition: Condition::Set(Box::new(SetCondition::IsIn(
                ElementExpression::Constant(0),
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            ))),
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
        assert!(transition.is_applicable(&state, &registry));

        let transition = Transition {
            name: String::from(""),
            elements_in_set_variable: vec![(0, 0), (1, 1)],
            elements_in_vector_variable: vec![(0, 0), (1, 2)],
            cost: NumericExpression::Constant(0),
            ..Default::default()
        };
        assert!(transition.is_applicable(&state, &registry));
    }

    #[test]
    fn not_applicable() {
        let state = generate_state();
        let registry = generate_registry();
        let set_condition = grounded_condition::GroundedCondition {
            condition: Condition::Set(Box::new(SetCondition::IsIn(
                ElementExpression::Constant(0),
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            ))),
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
        assert!(transition.is_applicable(&state, &registry));

        let transition = Transition {
            name: String::from(""),
            elements_in_set_variable: vec![(0, 1), (1, 1)],
            elements_in_vector_variable: vec![(0, 0), (1, 2)],
            cost: NumericExpression::Constant(0),
            ..Default::default()
        };
        assert!(!transition.is_applicable(&state, &registry));

        let transition = Transition {
            name: String::from(""),
            elements_in_set_variable: vec![(0, 1), (1, 1)],
            elements_in_vector_variable: vec![(0, 1), (1, 2)],
            cost: NumericExpression::Constant(0),
            ..Default::default()
        };
        assert!(!transition.is_applicable(&state, &registry));
    }

    #[test]
    fn appy_effects() {
        let state = generate_state();
        let registry = generate_registry();
        let set_effect1 = SetExpression::SetElementOperation(
            SetElementOperator::Add,
            ElementExpression::Constant(1),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
        );
        let set_effect2 = SetExpression::SetElementOperation(
            SetElementOperator::Remove,
            ElementExpression::Constant(0),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
        );
        let vector_effect1 = VectorExpression::Push(
            ElementExpression::Constant(1),
            Box::new(VectorExpression::Reference(ReferenceExpression::Variable(
                0,
            ))),
        );
        let vector_effect2 = VectorExpression::Push(
            ElementExpression::Constant(0),
            Box::new(VectorExpression::Reference(ReferenceExpression::Variable(
                1,
            ))),
        );
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
            effect: effect::Effect {
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
            },
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
            signature_variables: state::SignatureVariables {
                set_variables: vec![set1, set2],
                vector_variables: vec![vec![0, 2, 1], vec![1, 2, 0]],
                element_variables: vec![2, 1],
                integer_variables: vec![0, 4, 3],
                continuous_variables: vec![0.0, 4.0, 3.0],
            },
            resource_variables: state::ResourceVariables {
                integer_variables: vec![5, 2, 6],
                continuous_variables: vec![5.0, 2.5, 6.0],
            },
        };
        let successor = transition.apply(&state, &registry);
        assert_eq!(successor, expected);
    }

    #[test]
    fn eval_cost() {
        let state = generate_state();
        let registry = generate_registry();

        let transition = Transition {
            cost: NumericExpression::NumericOperation(
                NumericOperator::Add,
                Box::new(NumericExpression::Cost),
                Box::new(NumericExpression::Constant(1)),
            ),
            ..Default::default()
        };
        assert_eq!(transition.eval_cost(0, &state, &registry), 1);
    }

    #[test]
    fn get_full_name() {
        let transition = Transition::<Integer> {
            name: String::from("transition"),
            parameter_names: vec![String::from("param1"), String::from("param2")],
            parameter_values: vec![0, 1],
            ..Default::default()
        };
        assert_eq!(
            transition.get_full_name(),
            String::from("transition param1:0 param2:1")
        );
    }

    #[test]
    fn load_transitions_from_yaml_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();

        let transition = r"
name: transition
preconditions: [(>= (f2 0 1) 10)]
effect: {e0: '0'}
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
            effect: effect::Effect {
                element_effects: vec![(0, ElementExpression::Constant(0))],
                ..Default::default()
            },
            cost: NumericExpression::Constant(0),
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
        let transitions = load_transitions_from_yaml(transition, &metadata, &registry);
        assert!(transitions.is_ok());
        let expected = vec![Transition {
            name: String::from("transition"),
            preconditions: Vec::new(),
            effect: effect::Effect {
                element_effects: vec![(0, ElementExpression::Constant(0))],
                ..Default::default()
            },
            cost: NumericExpression::Constant(0),
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
        let transitions = load_transitions_from_yaml(transition, &metadata, &registry);
        assert!(transitions.is_ok());
        let expected = vec![Transition {
            name: String::from("transition"),
            preconditions: Vec::new(),
            effect: effect::Effect {
                element_effects: vec![(0, ElementExpression::Constant(0))],
                ..Default::default()
            },
            cost: NumericExpression::Constant(0),
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
        let transitions = load_transitions_from_yaml(transition, &metadata, &registry);
        assert!(transitions.is_ok());
        let expected = vec![Transition {
            name: String::from("transition"),
            preconditions: Vec::new(),
            effect: effect::Effect {
                element_effects: vec![(0, ElementExpression::Constant(0))],
                ..Default::default()
            },
            cost: NumericExpression::Constant(0),
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
        let transitions = load_transitions_from_yaml(transition, &metadata, &registry);
        assert!(transitions.is_ok());
        let expected = vec![Transition {
            name: String::from("transition"),
            preconditions: Vec::new(),
            effect: effect::Effect {
                element_effects: vec![(0, ElementExpression::Constant(0))],
                ..Default::default()
            },
            cost: NumericExpression::Constant(0),
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
        let transitions = load_transitions_from_yaml(transition, &metadata, &registry);
        assert!(transitions.is_ok());
        let expected = vec![Transition {
            name: String::from("transition"),
            preconditions: Vec::new(),
            effect: effect::Effect {
                element_effects: vec![(0, ElementExpression::Constant(0))],
                ..Default::default()
            },
            cost: NumericExpression::Constant(0),
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
        - (is_not e 2)
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
        let transitions = load_transitions_from_yaml(transition, &metadata, &registry);
        assert!(transitions.is_ok());
        let expected = vec![
            Transition {
                name: String::from("transition"),
                parameter_names: vec![String::from("e")],
                parameter_values: vec![0],
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
                effect: effect::Effect {
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
                    integer_effects: vec![(0, NumericExpression::Constant(1))],
                    integer_resource_effects: vec![(0, NumericExpression::Constant(2))],
                    ..Default::default()
                },
                cost: NumericExpression::NumericOperation(
                    NumericOperator::Add,
                    Box::new(NumericExpression::Cost),
                    Box::new(NumericExpression::Constant(10)),
                ),
            },
            Transition {
                name: String::from("transition"),
                parameter_names: vec![String::from("e")],
                parameter_values: vec![1],
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
                effect: effect::Effect {
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
                    integer_effects: vec![(0, NumericExpression::Constant(1))],
                    integer_resource_effects: vec![(0, NumericExpression::Constant(2))],
                    ..Default::default()
                },
                cost: NumericExpression::NumericOperation(
                    NumericOperator::Add,
                    Box::new(NumericExpression::Cost),
                    Box::new(NumericExpression::Constant(20)),
                ),
            },
        ];
        assert_eq!(transitions.unwrap(), (expected, false, false));
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
effect:
        e0: e
        s0: (add e s0)
        p0: (push e p0)
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
        let transitions = load_transitions_from_yaml::<Integer>(transition, &metadata, &registry);
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
        let transitions = load_transitions_from_yaml::<Integer>(transition, &metadata, &registry);
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
        let transitions = load_transitions_from_yaml::<Integer>(transition, &metadata, &registry);
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
        let transitions = load_transitions_from_yaml::<Integer>(transition, &metadata, &registry);
        assert!(transitions.is_err());
    }
}
