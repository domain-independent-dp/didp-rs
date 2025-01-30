use super::expression_to_string::ToYamlString;
use dypdl::{StateFunctions, StateMetadata, TableRegistry};
use yaml_rust::{
    yaml::{Array, Hash},
    Yaml,
};

pub fn state_functions_to_yaml(
    metadata: &StateMetadata,
    functions: &StateFunctions,
    registry: &TableRegistry,
) -> Result<Option<Yaml>, &'static str> {
    let mut function_array = Array::new();

    add_typed_state_functions(
        metadata,
        functions,
        registry,
        &functions.integer_function_names,
        "integer",
        &functions.integer_functions,
        &mut function_array,
    )?;

    add_typed_state_functions(
        metadata,
        functions,
        registry,
        &functions.continuous_function_names,
        "continuous",
        &functions.continuous_functions,
        &mut function_array,
    )?;

    add_typed_state_functions(
        metadata,
        functions,
        registry,
        &functions.element_function_names,
        "element",
        &functions.element_functions,
        &mut function_array,
    )?;

    add_typed_state_functions(
        metadata,
        functions,
        registry,
        &functions.set_function_names,
        "set",
        &functions.set_functions,
        &mut function_array,
    )?;

    add_typed_state_functions(
        metadata,
        functions,
        registry,
        &functions.boolean_function_names,
        "bool",
        &functions.boolean_functions,
        &mut function_array,
    )?;

    if function_array.is_empty() {
        Ok(None)
    } else {
        Ok(Some(Yaml::Array(function_array)))
    }
}

fn add_typed_state_functions<E>(
    metadata: &StateMetadata,
    functions: &StateFunctions,
    registry: &TableRegistry,
    names: &[String],
    function_type: &str,
    expressions: &[E],
    function_array: &mut Array,
) -> Result<(), &'static str>
where
    E: ToYamlString + Clone,
{
    for (name, expression) in names.iter().zip(expressions.iter()) {
        function_array.push(state_function_to_yaml(
            metadata,
            functions,
            registry,
            name,
            function_type,
            expression.clone(),
        )?);
    }

    Ok(())
}

fn state_function_to_yaml<E>(
    metadata: &StateMetadata,
    functions: &StateFunctions,
    registry: &TableRegistry,
    name: &str,
    function_type: &str,
    expression: E,
) -> Result<Yaml, &'static str>
where
    E: ToYamlString,
{
    let mut function_hash = Hash::new();

    function_hash.insert(Yaml::from_str("name"), Yaml::from_str(name));
    function_hash.insert(Yaml::from_str("type"), Yaml::from_str(function_type));
    function_hash.insert(
        Yaml::from_str("expression"),
        Yaml::String(expression.to_yaml_string(metadata, functions, registry)?),
    );

    Ok(Yaml::Hash(function_hash))
}

#[cfg(test)]
mod tests {
    use super::*;
    use dypdl::prelude::*;

    #[test]
    fn test_empty_state_functions_to_yaml() {
        let metadata = StateMetadata::default();
        let functions = StateFunctions::default();
        let registry = TableRegistry::default();

        let yaml = state_functions_to_yaml(&metadata, &functions, &registry).unwrap();

        assert_eq!(yaml, None);
    }

    #[test]
    fn test_state_functions_to_yaml() {
        let registry = TableRegistry::default();

        let mut metadata = StateMetadata::default();
        let result = metadata.add_object_type("object", 4);
        assert!(result.is_ok());
        let object = result.unwrap();
        let result = metadata.add_set_variable("v", object);
        assert!(result.is_ok());
        let v = result.unwrap();

        let mut functions = StateFunctions::default();
        let f1 = ContinuousExpression::from(4.2);
        let result = functions.add_continuous_function("f1", f1.clone());
        assert!(result.is_ok());
        let f2 = IntegerExpression::from(42);
        let result = functions.add_integer_function("f2", f2.clone());
        assert!(result.is_ok());
        let f3 = SetExpression::from(v);
        let result = functions.add_set_function("f3", f3.clone());
        assert!(result.is_ok());
        let f4 = ElementExpression::from(3);
        let result = functions.add_element_function("f4", f4.clone());
        assert!(result.is_ok());
        let f5 = v.contains(3);
        let result = functions.add_boolean_function("f5", f5.clone());
        assert!(result.is_ok());

        let expected = Some(Yaml::Array(vec![
            Yaml::Hash({
                let mut hash = Hash::new();
                hash.insert(Yaml::from_str("name"), Yaml::from_str("f2"));
                hash.insert(Yaml::from_str("type"), Yaml::from_str("integer"));
                hash.insert(Yaml::from_str("expression"), Yaml::String("42".to_owned()));
                hash
            }),
            Yaml::Hash({
                let mut hash = Hash::new();
                hash.insert(Yaml::from_str("name"), Yaml::from_str("f1"));
                hash.insert(Yaml::from_str("type"), Yaml::from_str("continuous"));
                hash.insert(Yaml::from_str("expression"), Yaml::String("4.2".to_owned()));
                hash
            }),
            Yaml::Hash({
                let mut hash = Hash::new();
                hash.insert(Yaml::from_str("name"), Yaml::from_str("f4"));
                hash.insert(Yaml::from_str("type"), Yaml::from_str("element"));
                hash.insert(Yaml::from_str("expression"), Yaml::String("3".to_owned()));
                hash
            }),
            Yaml::Hash({
                let mut hash = Hash::new();
                hash.insert(Yaml::from_str("name"), Yaml::from_str("f3"));
                hash.insert(Yaml::from_str("type"), Yaml::from_str("set"));
                hash.insert(Yaml::from_str("expression"), Yaml::String("v".to_owned()));
                hash
            }),
            Yaml::Hash({
                let mut hash = Hash::new();
                hash.insert(Yaml::from_str("name"), Yaml::from_str("f5"));
                hash.insert(Yaml::from_str("type"), Yaml::from_str("bool"));
                hash.insert(
                    Yaml::from_str("expression"),
                    Yaml::String("(is_in 3 v)".to_owned()),
                );
                hash
            }),
        ]));

        let result = state_functions_to_yaml(&metadata, &functions, &registry);
        assert!(result.is_ok());
        let yaml = result.unwrap();
        assert_eq!(yaml, expected);
    }
}
