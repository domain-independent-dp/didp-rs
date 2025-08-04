use super::expression_to_string::ToYamlString;
use dypdl::{
    expression::Condition, StateFunctions, StateMetadata, TableRegistry, Transition,
    TransitionDominance,
};
use yaml_rust::{
    yaml::{Array, Hash},
    Yaml,
};

pub fn transition_dominance_to_yaml(
    metadata: &StateMetadata,
    functions: &StateFunctions,
    registry: &TableRegistry,
    transitions: &[Transition],
    dominance: &TransitionDominance,
) -> Result<Yaml, &'static str> {
    let mut hash = Hash::new();

    let dominating = transitions
        .get(dominance.dominating)
        .ok_or("Dominating transition not found")?;
    let dominated = transitions
        .get(dominance.dominated)
        .ok_or("Dominated transition not found")?;

    let mut dominating_hash = Hash::new();
    dominating_hash.insert(
        Yaml::from_str("name"),
        Yaml::String(dominating.get_full_name()),
    );
    hash.insert(Yaml::from_str("dominating"), Yaml::Hash(dominating_hash));

    let mut dominated_hash = Hash::new();
    dominated_hash.insert(
        Yaml::from_str("name"),
        Yaml::String(dominated.get_full_name()),
    );
    hash.insert(Yaml::from_str("dominated"), Yaml::Hash(dominated_hash));

    if !dominance.conditions.is_empty() {
        let conditions = dominance
            .conditions
            .iter()
            .map(|c| Condition::from(c.clone()).to_yaml_string(metadata, functions, registry))
            .collect::<Result<Vec<_>, _>>()?;

        hash.insert(
            Yaml::from_str("conditions"),
            Yaml::Array(Array::from(
                conditions.into_iter().map(Yaml::String).collect::<Vec<_>>(),
            )),
        );
    }

    Ok(Yaml::Hash(hash))
}

#[cfg(test)]
mod tests {
    use super::*;
    use dypdl::prelude::*;

    #[test]
    fn test_transition_dominance_to_yaml_with_ok() {
        let metadata = StateMetadata::default();
        let functions = StateFunctions::default();
        let registry = TableRegistry::default();

        let transitions = [
            Transition::new("transition1"),
            Transition::new("transition2"),
        ];

        let dominance = TransitionDominance {
            dominating: 0,
            dominated: 1,
            backward: true,
            conditions: vec![],
        };

        let expected = Yaml::Hash({
            let mut hash = Hash::new();
            let mut dominating_hash = Hash::new();
            dominating_hash.insert(Yaml::from_str("name"), Yaml::from_str("transition1"));
            hash.insert(Yaml::from_str("dominating"), Yaml::Hash(dominating_hash));
            let mut dominated_hash = Hash::new();
            dominated_hash.insert(Yaml::from_str("name"), Yaml::from_str("transition2"));
            hash.insert(Yaml::from_str("dominated"), Yaml::Hash(dominated_hash));

            hash
        });

        let result = transition_dominance_to_yaml(
            &metadata,
            &functions,
            &registry,
            &transitions,
            &dominance,
        );
        assert_eq!(result, Ok(expected));
    }

    #[test]
    fn test_transition_dominance_to_yaml_with_conditions_ok() {
        let functions = StateFunctions::default();
        let registry = TableRegistry::default();

        let mut metadata = StateMetadata::default();
        let result = metadata.add_integer_variable("v");
        assert!(result.is_ok());
        let v = result.unwrap();

        let transitions = [
            Transition::new("transition1"),
            Transition::new("transition2"),
        ];

        let dominance = TransitionDominance {
            dominating: 0,
            dominated: 1,
            backward: true,
            conditions: vec![
                Condition::comparison_i(ComparisonOperator::Ge, v, 0).into(),
                Condition::comparison_i(ComparisonOperator::Le, v, 10).into(),
            ],
        };

        let expected = Yaml::Hash({
            let mut hash = Hash::new();
            let mut dominating_hash = Hash::new();
            dominating_hash.insert(Yaml::from_str("name"), Yaml::from_str("transition1"));
            hash.insert(Yaml::from_str("dominating"), Yaml::Hash(dominating_hash));
            let mut dominated_hash = Hash::new();
            dominated_hash.insert(Yaml::from_str("name"), Yaml::from_str("transition2"));
            hash.insert(Yaml::from_str("dominated"), Yaml::Hash(dominated_hash));
            hash.insert(
                Yaml::from_str("conditions"),
                Yaml::Array(vec![
                    Yaml::from_str("(>= v 0)"),
                    Yaml::from_str("(<= v 10)"),
                ]),
            );

            hash
        });

        let result = transition_dominance_to_yaml(
            &metadata,
            &functions,
            &registry,
            &transitions,
            &dominance,
        );
        assert_eq!(result, Ok(expected));
    }

    #[test]
    fn test_transition_dominance_to_yaml_no_dominating_err() {
        let metadata = StateMetadata::default();
        let functions = StateFunctions::default();
        let registry = TableRegistry::default();

        let transitions = [
            Transition::new("transition1"),
            Transition::new("transition2"),
        ];

        let dominance = TransitionDominance {
            dominating: 2,
            dominated: 1,
            backward: false,
            conditions: vec![],
        };

        let result = transition_dominance_to_yaml(
            &metadata,
            &functions,
            &registry,
            &transitions,
            &dominance,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_transition_dominance_to_yaml_no_dominated_err() {
        let metadata = StateMetadata::default();
        let functions = StateFunctions::default();
        let registry = TableRegistry::default();

        let transitions = [
            Transition::new("transition1"),
            Transition::new("transition2"),
        ];

        let dominance = TransitionDominance {
            dominating: 0,
            dominated: 2,
            backward: false,
            conditions: vec![],
        };

        let result = transition_dominance_to_yaml(
            &metadata,
            &functions,
            &registry,
            &transitions,
            &dominance,
        );
        assert!(result.is_err());
    }
}
