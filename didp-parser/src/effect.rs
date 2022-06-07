use crate::expression;
use crate::expression_parser;
use crate::parse_expression_from_yaml::{parse_element_from_yaml, ParesNumericExpressionFromYaml};
use crate::state;
use crate::table_registry;
use crate::variable::{Continuous, Element, Integer};
use crate::yaml_util;
use rustc_hash::FxHashMap;
use std::error::Error;

#[derive(Debug, PartialEq, Clone, Default)]
pub struct Effect {
    pub set_effects: Vec<(usize, expression::SetExpression)>,
    pub vector_effects: Vec<(usize, expression::VectorExpression)>,
    pub element_effects: Vec<(usize, expression::ElementExpression)>,
    pub integer_effects: Vec<(usize, expression::NumericExpression<Integer>)>,
    pub continuous_effects: Vec<(usize, expression::NumericExpression<Continuous>)>,
    pub element_resource_effects: Vec<(usize, expression::ElementExpression)>,
    pub integer_resource_effects: Vec<(usize, expression::NumericExpression<Integer>)>,
    pub continuous_resource_effects: Vec<(usize, expression::NumericExpression<Continuous>)>,
}

impl Effect {
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
        let mut element_resource_effects = Vec::new();
        let mut integer_resource_effects = Vec::new();
        let mut continuous_resource_effects = Vec::new();
        for (variable, effect) in lifted_effects {
            let variable = yaml_util::get_string(variable)?;
            if let Some(i) = metadata.name_to_set_variable.get(&variable) {
                let effect = yaml_util::get_string(effect)?;
                let effect = expression_parser::parse_set(effect, metadata, registry, parameters)?;
                set_effects.push((*i, effect));
            } else if let Some(i) = metadata.name_to_vector_variable.get(&variable) {
                let effect = yaml_util::get_string(effect)?;
                let effect =
                    expression_parser::parse_vector(effect, metadata, registry, parameters)?;
                vector_effects.push((*i, effect));
            } else if let Some(i) = metadata.name_to_element_variable.get(&variable) {
                let effect = parse_element_from_yaml(effect, metadata, registry, parameters)?;
                element_effects.push((*i, effect));
            } else if let Some(i) = metadata.name_to_element_resource_variable.get(&variable) {
                let effect = parse_element_from_yaml(effect, metadata, registry, parameters)?;
                element_resource_effects.push((*i, effect));
            } else if let Some(i) = metadata.name_to_integer_variable.get(&variable) {
                let effect =
                    Integer::parse_expression_from_yaml(effect, metadata, registry, parameters)?;
                integer_effects.push((*i, effect.simplify(registry)));
            } else if let Some(i) = metadata.name_to_integer_resource_variable.get(&variable) {
                let effect =
                    Integer::parse_expression_from_yaml(effect, metadata, registry, parameters)?;
                integer_resource_effects.push((*i, effect.simplify(registry)));
            } else if let Some(i) = metadata.name_to_continuous_variable.get(&variable) {
                let effect =
                    Continuous::parse_expression_from_yaml(effect, metadata, registry, parameters)?;
                continuous_effects.push((*i, effect.simplify(registry)));
            } else if let Some(i) = metadata.name_to_continuous_resource_variable.get(&variable) {
                let effect =
                    Continuous::parse_expression_from_yaml(effect, metadata, registry, parameters)?;
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
            element_resource_effects,
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

        let element_resource_variable_names = vec![
            "er0".to_string(),
            "er1".to_string(),
            "er2".to_string(),
            "er3".to_string(),
        ];
        let mut name_to_element_resource_variable = FxHashMap::default();
        name_to_element_resource_variable.insert("er0".to_string(), 0);
        name_to_element_resource_variable.insert("er1".to_string(), 1);
        name_to_element_resource_variable.insert("er2".to_string(), 2);
        name_to_element_resource_variable.insert("er3".to_string(), 3);
        let element_resource_variable_to_object = vec![0, 0, 0, 0];

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
            element_resource_variable_names,
            name_to_element_resource_variable,
            element_resource_variable_to_object,
            element_less_is_better: vec![false, false, true, true],
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
            element_resource_effects: vec![(0, ElementExpression::Constant(1))],
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
        let effect = Effect::load_from_yaml(effect, &metadata, &registry, &parameters);
        assert!(effect.is_err());
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
