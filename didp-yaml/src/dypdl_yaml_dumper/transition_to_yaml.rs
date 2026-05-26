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
