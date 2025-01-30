use super::ToYamlString;
use dypdl::{StateFunctions, StateMetadata, TableRegistry, Transition};
use yaml_rust::yaml::Array;
use yaml_rust::{yaml::Hash, Yaml};

pub fn transition_to_yaml(
    t: &Transition,
    forward: bool,
    forced: bool,
    state_metadata: &StateMetadata,
    state_functions: &StateFunctions,
    table_registry: &TableRegistry,
) -> Result<Yaml, &'static str> {
    let mut hash = Hash::new();

    // Insert transition name
    hash.insert(Yaml::from_str("name"), Yaml::String(t.get_full_name()));

    // Insert transition effects
    let mut effect_yaml_hash = Hash::new();
    let all_effects = &t.effect;

    macro_rules! insert_effects_to_yaml {
        ($($effect_field:ident).+, $($variable_field:ident).+) => {
            for (var_index, expr) in &all_effects . $( $effect_field ).+ {
                effect_yaml_hash.insert(
                    Yaml::from_str(&state_metadata . $( $variable_field ).+ [*var_index]),
                    Yaml::String(expr.to_yaml_string(state_metadata, state_functions, table_registry)?),
                );
            }
        }
    }
    insert_effects_to_yaml!(set_effects, set_variable_names);
    insert_effects_to_yaml!(element_effects, element_variable_names);
    insert_effects_to_yaml!(element_resource_effects, element_resource_variable_names);
    insert_effects_to_yaml!(integer_effects, integer_variable_names);
    insert_effects_to_yaml!(integer_resource_effects, integer_resource_variable_names);
    insert_effects_to_yaml!(continuous_effects, continuous_variable_names);
    insert_effects_to_yaml!(
        continuous_resource_effects,
        continuous_resource_variable_names
    );
    hash.insert(Yaml::from_str("effect"), Yaml::Hash(effect_yaml_hash));

    // Insert the transition cost.
    hash.insert(
        Yaml::from_str("cost"),
        Yaml::String(
            t.cost
                .to_yaml_string(state_metadata, state_functions, table_registry)?,
        ),
    );

    // Insert the transition preconditions.
    let mut preconditions = Array::new();
    for precond in &t.get_preconditions() {
        preconditions.push(Yaml::String(precond.clone().to_yaml_string(
            state_metadata,
            state_functions,
            table_registry,
        )?));
    }
    hash.insert(Yaml::from_str("preconditions"), Yaml::Array(preconditions));

    // Insert the transition forced flag.
    hash.insert(
        Yaml::from_str("forced"),
        if forced {
            Yaml::Boolean(true)
        } else {
            Yaml::Boolean(false)
        },
    );

    // Insert the transition direction.
    hash.insert(
        Yaml::from_str("direction"),
        if forward {
            Yaml::from_str("forward")
        } else {
            Yaml::from_str("backward")
        },
    );

    Ok(Yaml::Hash(hash))
}

#[cfg(test)]
mod tests {
    use dypdl::expression::{BinaryOperator, Condition, IntegerExpression};
    use dypdl::{Effect, GroundedCondition, StateFunctions, TableRegistry, Transition};
    use rustc_hash::FxHashMap;
    use yaml_rust::yaml::Hash;
    use yaml_rust::Yaml;

    use crate::dypdl_yaml_dumper::transition_to_yaml::transition_to_yaml;

    fn generate_metadata() -> dypdl::StateMetadata {
        let object_names = vec!["object".to_string()];
        let object_numbers = vec![10];
        let mut name_to_object = FxHashMap::default();
        name_to_object.insert("object".to_string(), 0);

        let set_variable_names = vec!["s0".to_string(), "s1".to_string()];
        let mut name_to_set_variable = FxHashMap::default();
        name_to_set_variable.insert("s0".to_string(), 0);
        name_to_set_variable.insert("s1".to_string(), 1);
        let set_variable_to_object = vec![0, 0];

        let element_variable_names = vec!["e0".to_string(), "e1".to_string()];
        let mut name_to_element_variable = FxHashMap::default();
        name_to_element_variable.insert("e0".to_string(), 0);
        name_to_element_variable.insert("e1".to_string(), 1);
        let element_variable_to_object = vec![0, 0];

        let element_resource_variable_names = vec!["er0".to_string(), "er1".to_string()];
        let mut name_to_element_resource_variable = FxHashMap::default();
        name_to_element_resource_variable.insert("er0".to_string(), 0);
        name_to_element_resource_variable.insert("er1".to_string(), 1);
        let element_resource_variable_to_object = vec![0, 0];

        let integer_variable_names = vec!["i0".to_string(), "i1".to_string()];
        let mut name_to_integer_variable = FxHashMap::default();
        name_to_integer_variable.insert("i0".to_string(), 0);
        name_to_integer_variable.insert("i1".to_string(), 1);

        let integer_resource_variable_names = vec!["ir0".to_string(), "ir1".to_string()];
        let mut name_to_integer_resource_variable = FxHashMap::default();
        name_to_integer_resource_variable.insert("ir0".to_string(), 0);
        name_to_integer_resource_variable.insert("ir1".to_string(), 1);

        let continuous_variable_names = vec!["c0".to_string(), "c1".to_string()];
        let mut name_to_continuous_variable = FxHashMap::default();
        name_to_continuous_variable.insert("c0".to_string(), 0);
        name_to_continuous_variable.insert("c1".to_string(), 1);

        let continuous_resource_variable_names = vec!["cr0".to_string(), "cr1".to_string()];
        let mut name_to_continuous_resource_variable = FxHashMap::default();
        name_to_continuous_resource_variable.insert("cr0".to_string(), 0);
        name_to_continuous_resource_variable.insert("cr1".to_string(), 1);

        dypdl::StateMetadata {
            object_type_names: object_names,
            name_to_object_type: name_to_object,
            object_numbers,
            set_variable_names,
            name_to_set_variable,
            set_variable_to_object,
            vector_variable_names: vec![],
            name_to_vector_variable: FxHashMap::default(),
            vector_variable_to_object: vec![],
            element_variable_names,
            name_to_element_variable,
            element_variable_to_object,
            element_resource_variable_names,
            name_to_element_resource_variable,
            element_less_is_better: vec![false, true],
            element_resource_variable_to_object,
            integer_variable_names,
            name_to_integer_variable,
            integer_resource_variable_names,
            name_to_integer_resource_variable,
            integer_less_is_better: vec![false, true],
            continuous_variable_names,
            name_to_continuous_variable,
            continuous_resource_variable_names,
            name_to_continuous_resource_variable,
            continuous_less_is_better: vec![false, true],
        }
    }

    #[test]
    fn transition_to_yaml_test() {
        let transition = Transition {
            name: "t1".to_owned(),
            parameter_names: vec![],
            parameter_values: vec![],
            elements_in_set_variable: vec![(0, 0)],
            elements_in_vector_variable: vec![],
            preconditions: vec![GroundedCondition {
                condition: Condition::And(
                    Condition::Constant(true).into(),
                    Condition::Constant(false).into(),
                ),
                ..Default::default()
            }],
            effect: Effect {
                integer_effects: vec![(0, IntegerExpression::Constant(3))],
                ..Default::default()
            },
            cost: dypdl::CostExpression::Integer(IntegerExpression::BinaryOperation(
                BinaryOperator::Add,
                IntegerExpression::Cost.into(),
                IntegerExpression::Constant(1).into(),
            )),
        };
        let state_metadata = generate_metadata();

        let mut expected_yaml = Hash::new();
        expected_yaml.insert(Yaml::from_str("name"), Yaml::from_str("t1"));

        let mut effect_hash = Hash::new();
        effect_hash.insert(Yaml::from_str("i0"), Yaml::String("3".to_owned()));
        expected_yaml.insert(Yaml::from_str("effect"), Yaml::Hash(effect_hash));

        expected_yaml.insert(Yaml::from_str("cost"), Yaml::from_str("(+ cost 1)"));

        expected_yaml.insert(
            Yaml::from_str("preconditions"),
            Yaml::Array(vec![
                Yaml::from_str("(is_in 0 s0)"),
                Yaml::from_str("(and true false)"),
            ]),
        );

        expected_yaml.insert(Yaml::from_str("forced"), Yaml::Boolean(false));
        expected_yaml.insert(Yaml::from_str("direction"), Yaml::from_str("forward"));

        let result_yaml = transition_to_yaml(
            &transition,
            true,
            false,
            &state_metadata,
            &StateFunctions::default(),
            &TableRegistry::default(),
        );
        assert!(result_yaml.is_ok());
        assert_eq!(result_yaml.unwrap(), Yaml::Hash(expected_yaml));
    }
}
