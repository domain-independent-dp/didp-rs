use dypdl::expression::Condition;
use dypdl::{CostType, Model, ReduceFunction};
use state_to_yaml::state_to_yaml;
use std::error::Error;
use yaml_rust::yaml::Array;
use yaml_rust::YamlEmitter;
use yaml_rust::{yaml::Hash, Yaml};

mod base_case_to_yaml;
mod expression_to_string;
mod state_function_to_yaml;
mod state_to_yaml;
mod table_to_yaml;
mod to_yaml;
mod transition_dominance_to_yaml;
mod transition_to_yaml;
mod variable_to_yaml;

use base_case_to_yaml::base_case_to_yaml;
use expression_to_string::ToYamlString;
use table_to_yaml::{set_table_data_to_yaml, table_data_to_yaml};
use to_yaml::ToYaml;
use transition_dominance_to_yaml::transition_dominance_to_yaml;
use transition_to_yaml::transition_to_yaml;
use variable_to_yaml::*;

pub fn model_to_yaml(model: &Model) -> Result<(Yaml, Yaml), Box<dyn Error>> {
    let mut domain_hash = Hash::new();
    let mut problem_hash = Hash::new();

    let state_metadata = &model.state_metadata;
    let table_registry = &model.table_registry;

    let cost_type = match model.cost_type {
        CostType::Integer => "integer",
        CostType::Continuous => "continuous",
    };
    domain_hash.insert(Yaml::from_str("cost_type"), Yaml::from_str(cost_type));

    // The reduce field in the domain file
    let reduce_function = match model.reduce_function {
        ReduceFunction::Min => Yaml::from_str("min"),
        ReduceFunction::Max => Yaml::from_str("max"),
        ReduceFunction::Sum => Yaml::from_str("sum"),
        ReduceFunction::Product => Yaml::from_str("product"),
    };

    domain_hash.insert(Yaml::from_str("reduce"), reduce_function);

    // Objects field in the domain file
    let objects = state_metadata
        .object_type_names
        .iter()
        .map(|s: &String| Yaml::from_str(s))
        .collect();

    // Object_numbers field in the problem file
    let mut object_numbers = Hash::new();
    for i in 0..state_metadata.object_numbers.len() {
        object_numbers.insert(
            Yaml::from_str(&state_metadata.object_type_names[i]),
            state_metadata.object_numbers[i].to_yaml()?,
        );
    }

    domain_hash.insert(Yaml::from_str("objects"), Yaml::Array(objects));
    problem_hash.insert(Yaml::from_str("object_numbers"), Yaml::Hash(object_numbers));

    // State_variables field in the domain file
    let mut state_variables = Array::new();
    let number_of_integer_variables = state_metadata.number_of_integer_variables();
    if number_of_integer_variables > 0 {
        state_variables.extend(
            (0..number_of_integer_variables)
                .map(|i: usize| integer_variable_to_yaml(state_metadata, i)),
        );
    }

    let number_of_integer_resource_variables =
        state_metadata.number_of_integer_resource_variables();
    if number_of_integer_resource_variables > 0 {
        state_variables.extend(
            (0..number_of_integer_resource_variables)
                .map(|i: usize| integer_resource_variable_to_yaml(state_metadata, i)),
        );
    }

    let number_of_continuous_variables = state_metadata.number_of_continuous_variables();
    if number_of_continuous_variables > 0 {
        state_variables.extend(
            (0..number_of_continuous_variables)
                .map(|i: usize| continuous_variable_to_yaml(state_metadata, i)),
        );
    }

    let number_of_continuous_resource_variables =
        state_metadata.number_of_continuous_resource_variables();
    if number_of_continuous_resource_variables > 0 {
        state_variables.extend(
            (0..number_of_continuous_resource_variables)
                .map(|i: usize| continuous_resource_variable_to_yaml(state_metadata, i)),
        );
    }

    let number_of_element_variables = state_metadata.number_of_element_variables();
    if number_of_element_variables > 0 {
        state_variables.extend(
            (0..number_of_element_variables)
                .map(|i: usize| element_variable_to_yaml(state_metadata, i)),
        );
    }

    let number_of_element_resource_variables =
        state_metadata.number_of_element_resource_variables();
    if number_of_element_resource_variables > 0 {
        state_variables.extend(
            (0..number_of_element_resource_variables)
                .map(|i: usize| element_resource_variable_to_yaml(state_metadata, i)),
        );
    }

    let number_of_set_variables = state_metadata.number_of_set_variables();
    if number_of_set_variables > 0 {
        state_variables.extend(
            (0..number_of_set_variables).map(|i: usize| set_variable_to_yaml(state_metadata, i)),
        );
    }

    domain_hash.insert(
        Yaml::from_str("state_variables"),
        Yaml::Array(state_variables),
    );

    // The target field in the problem file.
    problem_hash.insert(
        Yaml::from_str("target"),
        state_to_yaml(&model.target, state_metadata)?,
    );

    let state_functions = &model.state_functions;

    if let Some(state_functions_yaml) = state_function_to_yaml::state_functions_to_yaml(
        state_metadata,
        state_functions,
        table_registry,
    )? {
        domain_hash.insert(Yaml::from_str("state_functions"), state_functions_yaml);
    }

    // The transitions field
    let mut transitions = Array::new();
    for t in &model.forward_transitions {
        transitions.push(transition_to_yaml(
            t,
            true,
            false,
            state_metadata,
            state_functions,
            table_registry,
        )?);
    }
    for t in &model.forward_forced_transitions {
        transitions.push(transition_to_yaml(
            t,
            true,
            true,
            state_metadata,
            state_functions,
            table_registry,
        )?);
    }
    for t in &model.backward_transitions {
        transitions.push(transition_to_yaml(
            t,
            false,
            false,
            state_metadata,
            state_functions,
            table_registry,
        )?);
    }
    for t in &model.backward_forced_transitions {
        transitions.push(transition_to_yaml(
            t,
            false,
            true,
            state_metadata,
            state_functions,
            table_registry,
        )?);
    }

    if !transitions.is_empty() {
        problem_hash.insert(Yaml::from_str("transitions"), Yaml::Array(transitions));
    }

    if !model.transition_dominance.is_empty() {
        let mut array = Array::new();

        for d in &model.transition_dominance {
            if d.backward {
                array.push(transition_dominance_to_yaml(
                    &model.state_metadata,
                    &model.state_functions,
                    &model.table_registry,
                    &model.backward_transitions,
                    d,
                )?);
            } else {
                array.push(transition_dominance_to_yaml(
                    &model.state_metadata,
                    &model.state_functions,
                    &model.table_registry,
                    &model.forward_transitions,
                    d,
                )?);
            }
        }

        problem_hash.insert(Yaml::from_str("transition_dominance"), Yaml::Array(array));
    }

    // The base cases field
    let mut base_cases = Array::new();
    for bc in &model.base_cases {
        base_cases.push(base_case_to_yaml(
            bc,
            state_metadata,
            state_functions,
            table_registry,
        )?);
    }
    if !base_cases.is_empty() {
        problem_hash.insert(Yaml::from_str("base_cases"), Yaml::Array(base_cases));
    }

    // The constraints field
    let mut constraints = Array::new();
    for c in &model.state_constraints {
        constraints.push(Yaml::String(Condition::from(c.clone()).to_yaml_string(
            state_metadata,
            state_functions,
            table_registry,
        )?));
    }
    if !constraints.is_empty() {
        problem_hash.insert(Yaml::from_str("constraints"), Yaml::Array(constraints));
    }

    // The dual bound field
    let mut dual_bounds = Array::new();
    for db in &model.dual_bounds {
        dual_bounds.push(Yaml::String(db.to_yaml_string(
            state_metadata,
            state_functions,
            table_registry,
        )?))
    }
    if !dual_bounds.is_empty() {
        problem_hash.insert(Yaml::from_str("dual_bounds"), Yaml::Array(dual_bounds));
    }

    // The tables field in the domain file
    // and the table_values field in the problem file
    let mut table_names = Array::new();
    let mut table_values = Hash::new();
    let mut dictionary_names = Array::new();
    let mut dictionary_values = Hash::new();
    table_data_to_yaml(
        &table_registry.integer_tables,
        "integer",
        &mut table_names,
        &mut table_values,
        &mut dictionary_names,
        &mut dictionary_values,
    )?;
    table_data_to_yaml(
        &table_registry.continuous_tables,
        "continuous",
        &mut table_names,
        &mut table_values,
        &mut dictionary_names,
        &mut dictionary_values,
    )?;
    table_data_to_yaml(
        &table_registry.element_tables,
        "element",
        &mut table_names,
        &mut table_values,
        &mut dictionary_names,
        &mut dictionary_values,
    )?;
    table_data_to_yaml(
        &table_registry.bool_tables,
        "bool",
        &mut table_names,
        &mut table_values,
        &mut dictionary_names,
        &mut dictionary_values,
    )?;
    set_table_data_to_yaml(
        &table_registry.set_tables,
        "set",
        &mut table_names,
        &mut table_values,
        &mut dictionary_names,
        &mut dictionary_values,
    )?;

    if !table_names.is_empty() {
        domain_hash.insert(Yaml::from_str("tables"), Yaml::Array(table_names));
    }
    if !table_values.is_empty() {
        problem_hash.insert(Yaml::from_str("table_values"), Yaml::Hash(table_values));
    }
    if !dictionary_names.is_empty() {
        domain_hash.insert(
            Yaml::from_str("dictionaries"),
            Yaml::Array(dictionary_names),
        );
    }
    if !dictionary_values.is_empty() {
        problem_hash.insert(
            Yaml::from_str("dictionary_values"),
            Yaml::Hash(dictionary_values),
        );
    }

    Ok((Yaml::Hash(domain_hash), Yaml::Hash(problem_hash)))
}

pub fn dump_model(model: &Model) -> Result<(String, String), Box<dyn Error>> {
    let (domain_yaml, problem_yaml) = model_to_yaml(model)?;

    let mut domain_str = String::new();
    let mut domain_emitter = YamlEmitter::new(&mut domain_str);
    domain_emitter.dump(&domain_yaml).unwrap();

    let mut problem_str = String::new();
    let mut problem_emitter = YamlEmitter::new(&mut problem_str);
    problem_emitter.dump(&problem_yaml).unwrap();

    Ok((domain_str, problem_str))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dypdl_parser;

    use dypdl::prelude::*;
    use dypdl::{Table, Table1D, Table2D, Table3D};
    use rustc_hash::FxHashMap;

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

        let integer_variable_names = vec!["n0".to_string(), "n1".to_string()];
        let mut name_to_integer_variable = FxHashMap::default();
        name_to_integer_variable.insert("n0".to_string(), 0);
        name_to_integer_variable.insert("n1".to_string(), 1);

        let integer_resource_variable_names = vec!["r0".to_string(), "r1".to_string()];
        let mut name_to_integer_resource_variable = FxHashMap::default();
        name_to_integer_resource_variable.insert("r0".to_string(), 0);
        name_to_integer_resource_variable.insert("r1".to_string(), 1);

        let continuous_variable_names = vec!["c0".to_string(), "c1".to_string()];
        let mut name_to_continuous_variable = FxHashMap::default();
        name_to_continuous_variable.insert("c0".to_string(), 0);
        name_to_continuous_variable.insert("c1".to_string(), 1);

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
            integer_variable_names,
            name_to_integer_variable,
            integer_resource_variable_names,
            name_to_integer_resource_variable,
            integer_less_is_better: vec![false, true],
            continuous_variable_names,
            name_to_continuous_variable,
            ..Default::default()
        }
    }

    fn generate_registry() -> dypdl::TableRegistry {
        let mut name_to_constant = FxHashMap::default();
        name_to_constant.insert(String::from("f0"), 0);

        let tables_1d = vec![Table1D::new(Vec::new())];
        let mut name_to_table_1d = FxHashMap::default();
        name_to_table_1d.insert(String::from("f1"), 0);

        let tables_2d = vec![Table2D::new(Vec::new())];
        let mut name_to_table_2d = FxHashMap::default();
        name_to_table_2d.insert(String::from("f2"), 0);

        let tables_3d = vec![Table3D::new(Vec::new())];
        let mut name_to_table_3d = FxHashMap::default();
        name_to_table_3d.insert(String::from("f3"), 0);

        let tables = vec![Table::new(FxHashMap::default(), 0)];
        let mut name_to_table = FxHashMap::default();
        name_to_table.insert(String::from("f4"), 0);

        dypdl::TableRegistry {
            integer_tables: dypdl::TableData {
                name_to_constant,
                tables_1d,
                name_to_table_1d,
                tables_2d,
                name_to_table_2d,
                tables_3d,
                name_to_table_3d,
                tables,
                name_to_table,
            },
            ..Default::default()
        }
    }

    fn generate_model() -> Model {
        let state_metadata = generate_metadata();
        let table_registry = generate_registry();

        Model {
            state_metadata,
            table_registry,
            cost_type: CostType::Integer,
            reduce_function: ReduceFunction::Min,
            ..Default::default()
        }
    }

    #[test]
    fn model_to_yaml_ok() {
        let model = generate_model();
        let yaml_result = model_to_yaml(&model);
        assert!(yaml_result.is_ok());

        let (_, problem_yaml) = yaml_result.unwrap();

        if let Yaml::Hash(hash) = problem_yaml {
            if let Some(Yaml::Hash(table_values)) = hash.get(&Yaml::from_str("tables")) {
                if let Some(value) = table_values.get(&Yaml::from_str("f0")) {
                    assert_eq!(value.clone(), Yaml::from_str("0"));
                }
            }
        }
    }

    #[test]
    fn model_with_state_functions_to_yaml_ok() {
        let mut model = Model::default();
        let result = model.add_integer_state_function("f1", IntegerExpression::from(0));
        assert!(result.is_ok());

        let yam_result = model_to_yaml(&model);
        assert!(yam_result.is_ok());
        let (domain_yaml, _) = yam_result.unwrap();

        let expected = Yaml::Array(vec![Yaml::Hash({
            let mut hash = Hash::new();
            hash.insert(Yaml::from_str("name"), Yaml::from_str("f1"));
            hash.insert(Yaml::from_str("type"), Yaml::from_str("integer"));
            hash.insert(Yaml::from_str("expression"), Yaml::String("0".to_owned()));
            hash
        })]);

        if let Yaml::Hash(hash) = domain_yaml {
            let result = hash.get(&Yaml::from_str("state_functions"));
            assert!(result.is_some());
            let state_functions = result.unwrap();
            assert_eq!(state_functions, &expected);
        } else {
            panic!("The domain yaml is not a hash");
        }
    }

    #[test]
    fn model_with_transition_dominance_to_yaml_ok() {
        let mut model = Model::default();
        let result = model.add_integer_variable("v", 0);
        assert!(result.is_ok());
        let v = result.unwrap();

        let transition1 = Transition::new("transition1");
        let result = model.add_forward_transition(transition1);
        assert!(result.is_ok());
        let id1 = result.unwrap();
        let transition2 = Transition::new("transition2");
        let result = model.add_forward_transition(transition2);
        assert!(result.is_ok());
        let id2 = result.unwrap();

        let result = model.add_transition_dominance_with_conditions(
            &id1,
            &id2,
            vec![
                Condition::comparison_i(ComparisonOperator::Ge, v, 0),
                Condition::comparison_i(ComparisonOperator::Le, v, 10),
            ],
        );
        assert!(result.is_ok());

        let transition3 = Transition::new("transition3");
        let result = model.add_backward_transition(transition3);
        assert!(result.is_ok());
        let id3 = result.unwrap();
        let transition4 = Transition::new("transition4");
        let result = model.add_backward_transition(transition4);
        assert!(result.is_ok());
        let id4 = result.unwrap();

        let result = model.add_transition_dominance(&id3, &id4);
        assert!(result.is_ok());

        let yam_result = model_to_yaml(&model);
        assert!(yam_result.is_ok());
        let (_, problem_yaml) = yam_result.unwrap();

        let expected = Yaml::Array(vec![
            Yaml::Hash({
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
                        Yaml::String("(>= v 0)".to_owned()),
                        Yaml::String("(<= v 10)".to_owned()),
                    ]),
                );
                hash
            }),
            Yaml::Hash({
                let mut hash = Hash::new();
                let mut dominating_hash = Hash::new();
                dominating_hash.insert(Yaml::from_str("name"), Yaml::from_str("transition3"));
                hash.insert(Yaml::from_str("dominating"), Yaml::Hash(dominating_hash));
                let mut dominated_hash = Hash::new();
                dominated_hash.insert(Yaml::from_str("name"), Yaml::from_str("transition4"));
                hash.insert(Yaml::from_str("dominated"), Yaml::Hash(dominated_hash));
                hash
            }),
        ]);

        if let Yaml::Hash(hash) = problem_yaml {
            let result = hash.get(&Yaml::from_str("transition_dominance"));
            assert_eq!(result, Some(&expected));
        } else {
            panic!("The problem yaml is not a hash");
        }
    }

    #[test]
    fn model_to_string_ok() {
        let model = generate_model();

        let result = dump_model(&model);
        assert!(result.is_ok());

        let (domain_str, problem_str) = result.unwrap();

        let domain_expected = r"---
cost_type: integer
reduce: min
objects:
  - object
state_variables:
  - name: n0
    type: integer
  - name: n1
    type: integer
  - name: r0
    type: integer
    preference: greater
  - name: r1
    type: integer
    preference: less
  - name: c0
    type: continuous
  - name: c1
    type: continuous
  - name: e0
    type: element
    object: object
  - name: e1
    type: element
    object: object
  - name: s0
    type: set
    object: object
  - name: s1
    type: set
    object: object
tables:
  - name: f0
    type: integer
  - name: f1
    type: integer
    args:
      - 0
  - name: f2
    type: integer
    args:
      - 0
      - 0
  - name: f3
    type: integer
    args:
      - 0
      - 0
      - 0
dictionaries:
  - name: f4
    type: integer
    default: 0";
        assert_eq!(domain_str, domain_expected.to_owned());

        let problem_expected = r"---
object_numbers:
  object: 10
target: {}
table_values:
  f0: 0
  f1: {}
  f2: {}
  f3: {}
dictionary_values:
  f4: {}";
        assert_eq!(problem_str, problem_expected.to_owned());
    }

    #[test]
    fn test_emit_ground_transitions() {
        let domain_str = r"---
cost_type: integer
reduce: min
objects:
    - object
base_cases:
    - - (>= n0 10)
      - (>= n1 10)
    - - (is_empty s0)
state_variables:
    - name: n0
      type: integer
    - name: n1
      type: integer
    - name: r0
      type: integer
      preference: greater
    - name: r1
      type: integer
      preference: less
    - name: c0
      type: continuous
    - name: c1
      type: continuous
    - name: e0
      type: element
      object: object
    - name: e1
      type: element
      object: object
    - name: s0
      type: set
      object: object
    - name: s1
      type: set
      object: object
transitions:
    - name: t1
      preconditions:
        - (<= n0 0)
        - (>= n1 1)
      effect:
        r0: (+ r1 1)
        r1: (+ r0 1)
      cost: (+ cost n1)
    - name: t2
      preconditions:
        - (<= c0 0)
        - (>= c1 1)
      effect:
        c0: (+ r1 1)
        c1: (+ r0 1)
      cost: (+ cost (ceil c1))
tables:
    - name: f0
      type: integer
    - name: f1
      type: integer
      args:
        - 3
    - name: f2
      type: integer
      args:
        - 3
        - 3
    - name: f3
      type: integer
      args:
        - 3
        - object
        - 3
    - name: f4
      type: integer
      args:
        - 3
      default: 0
    - name: f5
      type: set
      args:
        - 3
        - 3
      object: 4";
        let problem_str = r"---
object_numbers:
    object: 10
target: 
    n0: 0
    n1: 0
    r0: 0
    r1: 0
    c0: 0
    c1: 0
    e0: 0
    e1: 0
    s0: [0, 1, 2, 3]
    s1: [4, 5, 6, 7]
table_values:
    f0: 0
    f1: {0: 1}
    f2: {[0, 0]: 1}
    f3: {[0, 0, 0]: 1}
    f4: {0: 1}
    f5: {[0, 1]: [1, 2, 3]}
";
        let domain = yaml_rust::YamlLoader::load_from_str(domain_str)
            .unwrap_or_else(|_| panic!("Cannot load example domain"));
        assert_eq!(domain.len(), 1);
        let domain = &domain[0];

        let problem = yaml_rust::YamlLoader::load_from_str(problem_str)
            .unwrap_or_else(|_| panic!("Cannot load example domain"));
        assert_eq!(problem.len(), 1);
        let problem = &problem[0];

        let model = dypdl_parser::load_model_from_yaml(domain, problem)
            .unwrap_or_else(|err| panic!("Cannot load example model due to Error: {:?}", err));
        let (new_domain, new_problem) = dump_model(&model).unwrap_or_else(|e| {
            panic!("Couldn't dump the model: {:?}", e);
        });

        let new_domain = yaml_rust::YamlLoader::load_from_str(&new_domain)
            .unwrap_or_else(|_| panic!("Cannot load the new domain"));
        assert_eq!(new_domain.len(), 1);
        let new_domain = &new_domain[0];

        let new_problem = yaml_rust::YamlLoader::load_from_str(&new_problem)
            .unwrap_or_else(|_| panic!("Cannot load the new problem"));
        assert_eq!(new_problem.len(), 1);
        let new_problem = &new_problem[0];

        let new_model = dypdl_parser::load_model_from_yaml(new_domain, new_problem)
            .unwrap_or_else(|e| panic!("Cannot load the new model {:?}", e));

        assert_eq!(model.state_metadata, new_model.state_metadata);
        assert_eq!(model.target, new_model.target);
        assert_eq!(model.table_registry, new_model.table_registry);
        assert_eq!(model.state_constraints, new_model.state_constraints);
        assert_eq!(model.base_cases, new_model.base_cases);
        assert_eq!(model.base_states, new_model.base_states);
        assert_eq!(model.reduce_function, new_model.reduce_function);
        assert_eq!(model.cost_type, new_model.cost_type);
        assert_eq!(model.forward_transitions, new_model.forward_transitions);
        assert_eq!(
            model.forward_forced_transitions,
            new_model.forward_forced_transitions
        );
        assert_eq!(model.backward_transitions, new_model.backward_transitions);
        assert_eq!(
            model.backward_forced_transitions,
            new_model.backward_forced_transitions
        );
        assert_eq!(model.dual_bounds, new_model.dual_bounds);
    }

    #[test]
    fn test_emit_dictionaries() {
        let domain_str = r"---
cost_type: integer
reduce: min
objects:
    - object
base_cases:
    - - (>= n0 10)
      - (>= n1 10)
state_variables:
    - name: n0
      type: integer
    - name: n1
      type: integer
    - name: r0
      type: integer
      preference: greater
    - name: r1
      type: integer
      preference: less
    - name: c0
      type: continuous
    - name: c1
      type: continuous
transitions:
    - name: t1
      preconditions:
        - (<= n0 0)
        - (>= n1 1)
      effect:
        r0: (+ r1 1)
        r1: (+ r0 1)
      cost: (+ cost n1)
tables:
    - name: f0
      type: integer
    - name: f1
      type: integer
      args:
        - 3
    - name: f2
      type: integer
      args:
        - 3
        - 3
    - name: f3
      type: integer
      args:
        - 3
        - object
        - 3
        - object
dictionaries:
    - name: f4
      type: integer
      args:
        - 3
      default: 0";
        let problem_str = r"---
object_numbers:
    object: 10
target: 
    n0: 0
    n1: 0
    r0: 0
    r1: 0
    c0: 0
    c1: 0
table_values:
    f0: 0
    f1: {0: 1}
    f2: {[0, 0]: 1}
    f3: {[2, 0, 0, 9]: 1}
dictionary_values:
    f4: {[0]: 1}
";
        let domain = yaml_rust::YamlLoader::load_from_str(domain_str)
            .unwrap_or_else(|_| panic!("Cannot load example domain"));
        assert_eq!(domain.len(), 1);
        let domain = &domain[0];

        let problem = yaml_rust::YamlLoader::load_from_str(problem_str)
            .unwrap_or_else(|_| panic!("Cannot load example domain"));
        assert_eq!(problem.len(), 1);
        let problem = &problem[0];

        let model = dypdl_parser::load_model_from_yaml(domain, problem)
            .unwrap_or_else(|err| panic!("Cannot load example model due to Error: {:?}", err));
        let (new_domain, new_problem) = dump_model(&model).unwrap_or_else(|e| {
            panic!("Couldn't dump the model: {:?}", e);
        });

        let new_domain = yaml_rust::YamlLoader::load_from_str(&new_domain)
            .unwrap_or_else(|_| panic!("Cannot load the new domain"));
        assert_eq!(new_domain.len(), 1);
        let new_domain = &new_domain[0];

        let new_problem = yaml_rust::YamlLoader::load_from_str(&new_problem)
            .unwrap_or_else(|_| panic!("Cannot load the new problem"));
        assert_eq!(new_problem.len(), 1);
        let new_problem = &new_problem[0];

        let new_model = dypdl_parser::load_model_from_yaml(new_domain, new_problem)
            .unwrap_or_else(|e| panic!("Cannot load the new model {:?}", e));

        assert_eq!(model.state_metadata, new_model.state_metadata);
        assert_eq!(model.target, new_model.target);
        assert_eq!(model.table_registry, new_model.table_registry);
        assert_eq!(model.state_constraints, new_model.state_constraints);
        assert_eq!(model.base_cases, new_model.base_cases);
        assert_eq!(model.base_states, new_model.base_states);
        assert_eq!(model.reduce_function, new_model.reduce_function);
        assert_eq!(model.cost_type, new_model.cost_type);
        assert_eq!(model.forward_transitions, new_model.forward_transitions);
        assert_eq!(
            model.forward_forced_transitions,
            new_model.forward_forced_transitions
        );
        assert_eq!(model.backward_transitions, new_model.backward_transitions);
        assert_eq!(
            model.backward_forced_transitions,
            new_model.backward_forced_transitions
        );
        assert_eq!(model.dual_bounds, new_model.dual_bounds);
    }

    #[test]
    fn test_emit_base_case_with_cost() {
        let domain_str = r"---
cost_type: integer
reduce: min
objects:
    - object
base_cases:
    - cost: 1
      conditions:
        - (>= n0 10)
        - (>= n1 10)
state_variables:
    - name: n0
      type: integer
    - name: n1
      type: integer
    - name: r0
      type: integer
      preference: greater
    - name: r1
      type: integer
      preference: less
transitions:
    - name: t1
      preconditions:
        - (<= n0 0)
        - (>= n1 1)
      effect:
        r0: (+ r1 1)
        r1: (+ r0 1)
      cost: (+ cost n1)
tables:
    - name: f0
      type: integer
    - name: f1
      type: integer
      args:
        - 3
    - name: f2
      type: integer
      args:
        - 3
        - 3
";
        let problem_str = r"---
object_numbers:
    object: 10
target: 
    n0: 0
    n1: 0
    r0: 0
    r1: 0
table_values:
    f0: 0
    f1: {0: 1}
    f2: {[0, 0]: 1}
";
        let domain = yaml_rust::YamlLoader::load_from_str(domain_str)
            .unwrap_or_else(|_| panic!("Cannot load example domain"));
        assert_eq!(domain.len(), 1);
        let domain = &domain[0];

        let problem = yaml_rust::YamlLoader::load_from_str(problem_str)
            .unwrap_or_else(|_| panic!("Cannot load example domain"));
        assert_eq!(problem.len(), 1);
        let problem = &problem[0];

        let model = dypdl_parser::load_model_from_yaml(domain, problem)
            .unwrap_or_else(|err| panic!("Cannot load example model due to Error: {:?}", err));
        let (new_domain, new_problem) = dump_model(&model).unwrap_or_else(|e| {
            panic!("Couldn't dump the model: {:?}", e);
        });

        let new_domain = yaml_rust::YamlLoader::load_from_str(&new_domain)
            .unwrap_or_else(|_| panic!("Cannot load the new domain"));
        assert_eq!(new_domain.len(), 1);
        let new_domain = &new_domain[0];

        let new_problem = yaml_rust::YamlLoader::load_from_str(&new_problem)
            .unwrap_or_else(|_| panic!("Cannot load the new problem"));
        assert_eq!(new_problem.len(), 1);
        let new_problem = &new_problem[0];

        let new_model = dypdl_parser::load_model_from_yaml(new_domain, new_problem)
            .unwrap_or_else(|e| panic!("Cannot load the new model {:?}", e));

        assert_eq!(model.state_metadata, new_model.state_metadata);
        assert_eq!(model.target, new_model.target);
        assert_eq!(model.table_registry, new_model.table_registry);
        assert_eq!(model.state_constraints, new_model.state_constraints);
        assert_eq!(model.base_cases, new_model.base_cases);
        assert_eq!(model.base_states, new_model.base_states);
        assert_eq!(model.reduce_function, new_model.reduce_function);
        assert_eq!(model.cost_type, new_model.cost_type);
        assert_eq!(model.forward_transitions, new_model.forward_transitions);
        assert_eq!(
            model.forward_forced_transitions,
            new_model.forward_forced_transitions
        );
        assert_eq!(model.backward_transitions, new_model.backward_transitions);
        assert_eq!(
            model.backward_forced_transitions,
            new_model.backward_forced_transitions
        );
        assert_eq!(model.dual_bounds, new_model.dual_bounds);
    }
}
