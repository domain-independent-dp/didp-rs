use crate::expression;
use crate::expression_parser;
use crate::state;
use crate::table_registry;
use crate::variable::{Continuous, Element, Integer};
use crate::yaml_util;
use ordered_float::OrderedFloat;
use rustc_hash::FxHashMap;
use std::error::Error;
use std::rc::Rc;

#[derive(Debug, PartialEq, Clone, Default)]
pub struct Effect {
    pub set_effects: Vec<(usize, expression::SetExpression)>,
    pub vector_effects: Vec<(usize, expression::VectorExpression)>,
    pub element_effects: Vec<(usize, expression::ElementExpression)>,
    pub integer_effects: Vec<(usize, expression::NumericExpression<Integer>)>,
    pub continuous_effects: Vec<(usize, expression::NumericExpression<Continuous>)>,
    pub integer_resource_effects: Vec<(usize, expression::NumericExpression<Integer>)>,
    pub continuous_resource_effects: Vec<(usize, expression::NumericExpression<Continuous>)>,
}

impl Effect {
    pub fn apply(
        &self,
        state: &state::State,
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
            set_variables.push(e.1.eval(state, registry));
            i += 1;
        }
        while i < len {
            set_variables.push(state.signature_variables.set_variables[i].clone());
            i += 1;
        }

        let len = state.signature_variables.vector_variables.len();
        let mut vector_variables = Vec::with_capacity(len);
        for e in &self.vector_effects {
            while i < e.0 {
                vector_variables.push(state.signature_variables.vector_variables[i].clone());
                i += 1;
            }
            vector_variables.push(e.1.eval(state, registry));
            i += 1;
        }
        while i < len {
            vector_variables.push(state.signature_variables.vector_variables[i].clone());
            i += 1;
        }

        let mut element_variables = state.signature_variables.element_variables.clone();
        for e in &self.element_effects {
            element_variables[e.0] = e.1.eval(state, registry);
        }

        let mut integer_variables = state.signature_variables.integer_variables.clone();
        for e in &self.integer_effects {
            integer_variables[e.0] = e.1.eval(state, registry);
        }

        let mut continuous_variables = state.signature_variables.continuous_variables.clone();
        for e in &self.continuous_effects {
            continuous_variables[e.0] = OrderedFloat(e.1.eval(state, registry));
        }

        let mut integer_resource_variables = state.resource_variables.integer_variables.clone();
        for e in &self.integer_resource_effects {
            integer_resource_variables[e.0] = e.1.eval(state, registry);
        }

        let mut continuous_resource_variables =
            state.resource_variables.continuous_variables.clone();
        for e in &self.continuous_resource_effects {
            continuous_resource_variables[e.0] = OrderedFloat(e.1.eval(state, registry));
        }

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
        }
    }

    pub fn load_from_yaml(
        value: &yaml_rust::Yaml,
        metadata: &state::StateMetadata,
        registry: &table_registry::TableRegistry,
        parameters: &FxHashMap<String, Element>,
    ) -> Result<Effect, Box<dyn Error>> {
        let lifted_effects = yaml_util::get_map(value)?;
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
                let effect = expression_parser::parse_set(effect, metadata, registry, parameters)?;
                set_effects.push((*i, effect));
            } else if let Some(i) = metadata.name_to_vector_variable.get(&variable) {
                let effect =
                    expression_parser::parse_vector(effect, metadata, registry, parameters)?;
                vector_effects.push((*i, effect));
            } else if let Some(i) = metadata.name_to_element_variable.get(&variable) {
                let effect =
                    expression_parser::parse_element(effect, metadata, registry, parameters)?;
                element_effects.push((*i, effect));
            } else if let Some(i) = metadata.name_to_integer_variable.get(&variable) {
                let effect = expression_parser::parse_numeric::<Integer>(
                    effect, metadata, registry, parameters,
                )?;
                integer_effects.push((*i, effect.simplify(registry)));
            } else if let Some(i) = metadata.name_to_integer_resource_variable.get(&variable) {
                let effect = expression_parser::parse_numeric::<Integer>(
                    effect, metadata, registry, parameters,
                )?;
                integer_resource_effects.push((*i, effect.simplify(registry)));
            } else if let Some(i) = metadata.name_to_continuous_variable.get(&variable) {
                let effect = expression_parser::parse_numeric::<Continuous>(
                    effect, metadata, registry, parameters,
                )?;
                continuous_effects.push((*i, effect.simplify(registry)));
            } else if let Some(i) = metadata.name_to_continuous_resource_variable.get(&variable) {
                let effect = expression_parser::parse_numeric::<Continuous>(
                    effect, metadata, registry, parameters,
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
        Ok(Effect {
            set_effects,
            vector_effects,
            element_effects,
            integer_effects,
            continuous_effects,
            integer_resource_effects,
            continuous_resource_effects,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::table;
    use crate::table_data;
    use crate::variable::Set;
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
            signature_variables: Rc::new(state::SignatureVariables {
                set_variables: vec![set1, set2],
                vector_variables: vec![vec![0, 2], vec![1, 2]],
                element_variables: vec![1, 2],
                integer_variables: vec![1, 2, 3],
                continuous_variables: vec![OrderedFloat(1.0), OrderedFloat(2.0), OrderedFloat(3.0)],
            }),
            resource_variables: state::ResourceVariables {
                integer_variables: vec![4, 5, 6],
                continuous_variables: vec![OrderedFloat(4.0), OrderedFloat(5.0), OrderedFloat(6.0)],
            },
        }
    }

    #[test]
    fn appy() {
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
        let effect = Effect {
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
                continuous_variables: vec![OrderedFloat(5.0), OrderedFloat(2.5), OrderedFloat(6.0)],
            },
        };
        let successor = effect.apply(&state, &registry);
        assert_eq!(successor, expected);
    }

    #[test]
    fn load_from_yaml_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let mut parameters = FxHashMap::default();
        parameters.insert(String::from("e"), 0);

        let effect = r"
 e0: e
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
        let effect = Effect::load_from_yaml(effect, &metadata, &registry, &parameters);
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
            integer_effects: vec![(0, NumericExpression::Constant(1))],
            integer_resource_effects: vec![(0, NumericExpression::Constant(2))],
            continuous_effects: vec![(0, NumericExpression::Constant(1.0))],
            continuous_resource_effects: vec![(0, NumericExpression::Constant(2.0))],
        };
        assert_eq!(effect.unwrap(), expected);
    }

    #[test]
    fn load_from_yaml_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
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
        let effect = Effect::load_from_yaml(effect, &metadata, &registry, &parameters);
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
        let effect = Effect::load_from_yaml(effect, &metadata, &registry, &parameters);
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
        let effect = Effect::load_from_yaml(effect, &metadata, &registry, &parameters);
        assert!(effect.is_err());
    }
}
