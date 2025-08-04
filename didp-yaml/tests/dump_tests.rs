use didp_yaml::*;
use dypdl::{GroundedCondition, Transition};
use itertools::Itertools;
use std::fs;

fn transition_equality_ignoring_parameter(t1: Vec<Transition>, t2: Vec<Transition>) -> bool {
    if t1.len() != t2.len() {
        return false;
    }

    for (t1_element, t2_element) in t1.iter().zip(t2.iter()) {
        let filtered_t1 = Transition {
            name: t1_element.get_full_name(),
            parameter_names: vec![],
            parameter_values: vec![],
            elements_in_set_variable: vec![],
            elements_in_vector_variable: vec![],
            preconditions: t1_element
                .get_preconditions()
                .iter()
                .map(|cond| GroundedCondition::from(cond.clone()))
                .collect_vec(),
            ..t1_element.clone()
        };
        let filtered_t2 = Transition {
            name: t2_element.get_full_name(),
            parameter_names: vec![],
            parameter_values: vec![],
            elements_in_set_variable: vec![],
            elements_in_vector_variable: vec![],
            preconditions: t2_element
                .get_preconditions()
                .iter()
                .map(|cond| GroundedCondition::from(cond.clone()))
                .collect_vec(),
            ..t2_element.clone()
        };
        if filtered_t1 != filtered_t2 {
            println!("t1: {:?}\nt2: {:?}", filtered_t1, filtered_t2);
            return false;
        }
    }
    true
}

#[test]
fn test_emit_bpp_problem() {
    let domain = fs::read_to_string("tests/example_problems/bin-packing/bin-packing-domain.yaml")
        .unwrap_or_else(|e| panic!("Cannot read example domain {:?}", e));
    let domain = yaml_rust::YamlLoader::load_from_str(&domain)
        .unwrap_or_else(|e| panic!("Cannot load example domain {:?}", e));
    assert_eq!(domain.len(), 1);
    let domain = &domain[0];

    let problem = fs::read_to_string("tests/example_problems/bin-packing/bin-packing-problem.yaml")
        .unwrap_or_else(|e| panic!("Cannot read example problem {:?}", e));
    let problem = yaml_rust::YamlLoader::load_from_str(&problem)
        .unwrap_or_else(|e| panic!("Cannot load example problem {:?}", e));
    assert_eq!(problem.len(), 1);
    let problem = &problem[0];

    let model = dypdl_parser::load_model_from_yaml(domain, problem)
        .unwrap_or_else(|e| panic!("Cannot load example model due to {:?}", e));

    let (new_domain, new_problem) = dump_model(&model).unwrap_or_else(|e| {
        panic!("Couldn't dump the model: {:?}", e);
    });

    let new_domain = yaml_rust::YamlLoader::load_from_str(&new_domain)
        .unwrap_or_else(|e| panic!("Cannot load the new domain {:?}", e));
    assert_eq!(new_domain.len(), 1);
    let new_domain = &new_domain[0];

    let new_problem = yaml_rust::YamlLoader::load_from_str(&new_problem)
        .unwrap_or_else(|e| panic!("Cannot load the new problem {:?}", e));
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
    assert!(transition_equality_ignoring_parameter(
        model.forward_transitions,
        new_model.forward_transitions
    ));
    assert!(transition_equality_ignoring_parameter(
        model.forward_forced_transitions,
        new_model.forward_forced_transitions
    ));
    assert!(transition_equality_ignoring_parameter(
        model.backward_transitions,
        new_model.backward_transitions
    ));
    assert!(transition_equality_ignoring_parameter(
        model.backward_forced_transitions,
        new_model.backward_forced_transitions
    ));
    assert_eq!(model.dual_bounds, new_model.dual_bounds);
    assert_eq!(model.state_functions, new_model.state_functions);
    assert_eq!(model.transition_dominance, new_model.transition_dominance);
}

#[test]
fn test_emit_cvrp_problem() {
    let domain = fs::read_to_string("tests/example_problems/cvrp/cvrp-domain.yaml")
        .unwrap_or_else(|e| panic!("Cannot read example domain {:?}", e));
    let domain = yaml_rust::YamlLoader::load_from_str(&domain)
        .unwrap_or_else(|e| panic!("Cannot load example domain {:?}", e));
    assert_eq!(domain.len(), 1);
    let domain = &domain[0];

    let problem = fs::read_to_string("tests/example_problems/cvrp/cvrp-problem.yaml")
        .unwrap_or_else(|e| panic!("Cannot read example problem {:?}", e));
    let problem = yaml_rust::YamlLoader::load_from_str(&problem)
        .unwrap_or_else(|e| panic!("Cannot load example problem {:?}", e));
    assert_eq!(problem.len(), 1);
    let problem = &problem[0];

    let model = dypdl_parser::load_model_from_yaml(domain, problem)
        .unwrap_or_else(|e| panic!("Cannot load example model due to {:?}", e));

    let (new_domain, new_problem) = dump_model(&model).unwrap_or_else(|e| {
        panic!("Couldn't dump the model: {:?}", e);
    });

    let new_domain = yaml_rust::YamlLoader::load_from_str(&new_domain)
        .unwrap_or_else(|e| panic!("Cannot load the new domain {:?}", e));
    assert_eq!(new_domain.len(), 1);
    let new_domain = &new_domain[0];

    let new_problem = yaml_rust::YamlLoader::load_from_str(&new_problem)
        .unwrap_or_else(|e| panic!("Cannot load the new problem {:?}", e));
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
    assert!(transition_equality_ignoring_parameter(
        model.forward_transitions,
        new_model.forward_transitions
    ));
    assert!(transition_equality_ignoring_parameter(
        model.forward_forced_transitions,
        new_model.forward_forced_transitions
    ));
    assert!(transition_equality_ignoring_parameter(
        model.backward_transitions,
        new_model.backward_transitions
    ));
    assert!(transition_equality_ignoring_parameter(
        model.backward_forced_transitions,
        new_model.backward_forced_transitions
    ));
    assert_eq!(model.dual_bounds, new_model.dual_bounds);
    assert_eq!(model.state_functions, new_model.state_functions);
    assert_eq!(model.transition_dominance, new_model.transition_dominance);
}

#[test]
fn test_emit_graph_clear_problem() {
    let domain = fs::read_to_string("tests/example_problems/graph-clear/graph-clear-domain.yaml")
        .unwrap_or_else(|e| panic!("Cannot read example domain {:?}", e));
    let domain = yaml_rust::YamlLoader::load_from_str(&domain)
        .unwrap_or_else(|e| panic!("Cannot load example domain {:?}", e));
    assert_eq!(domain.len(), 1);
    let domain = &domain[0];

    let problem = fs::read_to_string("tests/example_problems/graph-clear/graph-clear-problem.yaml")
        .unwrap_or_else(|e| panic!("Cannot read example problem {:?}", e));
    let problem = yaml_rust::YamlLoader::load_from_str(&problem)
        .unwrap_or_else(|e| panic!("Cannot load example problem {:?}", e));
    assert_eq!(problem.len(), 1);
    let problem = &problem[0];

    let model = dypdl_parser::load_model_from_yaml(domain, problem)
        .unwrap_or_else(|e| panic!("Cannot load example model due to {:?}", e));

    let (new_domain, new_problem) = dump_model(&model).unwrap_or_else(|e| {
        panic!("Couldn't dump the model: {:?}", e);
    });

    let new_domain = yaml_rust::YamlLoader::load_from_str(&new_domain)
        .unwrap_or_else(|e| panic!("Cannot load the new domain {:?}", e));
    assert_eq!(new_domain.len(), 1);
    let new_domain = &new_domain[0];

    let new_problem = yaml_rust::YamlLoader::load_from_str(&new_problem)
        .unwrap_or_else(|e| panic!("Cannot load the new problem {:?}", e));
    assert_eq!(new_problem.len(), 1);
    let new_problem = &new_problem[0];

    let new_model = dypdl_parser::load_model_from_yaml(new_domain, new_problem)
        .unwrap_or_else(|e| panic!("Cannot load the new model {:?}", e));
    assert_eq!(model.state_metadata, new_model.state_metadata);
    assert_eq!(model.table_registry, new_model.table_registry);
    assert_eq!(model.target, new_model.target);
    assert_eq!(model.state_constraints, new_model.state_constraints);
    assert_eq!(model.base_cases, new_model.base_cases);
    assert_eq!(model.base_states, new_model.base_states);
    assert_eq!(model.reduce_function, new_model.reduce_function);
    assert_eq!(model.cost_type, new_model.cost_type);
    assert!(transition_equality_ignoring_parameter(
        model.forward_transitions,
        new_model.forward_transitions
    ));
    assert!(transition_equality_ignoring_parameter(
        model.forward_forced_transitions,
        new_model.forward_forced_transitions
    ));
    assert!(transition_equality_ignoring_parameter(
        model.backward_transitions,
        new_model.backward_transitions
    ));
    assert!(transition_equality_ignoring_parameter(
        model.backward_forced_transitions,
        new_model.backward_forced_transitions
    ));
    assert_eq!(model.dual_bounds, new_model.dual_bounds);
    assert_eq!(model.state_functions, new_model.state_functions);
    assert_eq!(model.transition_dominance, new_model.transition_dominance);
}

#[test]
fn test_emit_knapsack_problem() {
    let domain = fs::read_to_string("tests/example_problems/knapsack/knapsack-domain.yaml")
        .unwrap_or_else(|e| panic!("Cannot read example domain {:?}", e));
    let domain = yaml_rust::YamlLoader::load_from_str(&domain)
        .unwrap_or_else(|e| panic!("Cannot load example domain {:?}", e));
    assert_eq!(domain.len(), 1);
    let domain = &domain[0];

    let problem = fs::read_to_string("tests/example_problems/knapsack/knapsack-problem.yaml")
        .unwrap_or_else(|e| panic!("Cannot read example problem {:?}", e));
    let problem = yaml_rust::YamlLoader::load_from_str(&problem)
        .unwrap_or_else(|e| panic!("Cannot load example problem {:?}", e));
    assert_eq!(problem.len(), 1);
    let problem = &problem[0];

    let model = dypdl_parser::load_model_from_yaml(domain, problem)
        .unwrap_or_else(|e| panic!("Cannot load example model due to {:?}", e));

    let (new_domain, new_problem) = dump_model(&model).unwrap_or_else(|e| {
        panic!("Couldn't dump the model: {:?}", e);
    });

    let new_domain = yaml_rust::YamlLoader::load_from_str(&new_domain)
        .unwrap_or_else(|e| panic!("Cannot load the new domain {:?}", e));
    assert_eq!(new_domain.len(), 1);
    let new_domain = &new_domain[0];

    let new_problem = yaml_rust::YamlLoader::load_from_str(&new_problem)
        .unwrap_or_else(|e| panic!("Cannot load the new problem {:?}", e));
    assert_eq!(new_problem.len(), 1);
    let new_problem = &new_problem[0];

    let new_model = dypdl_parser::load_model_from_yaml(new_domain, new_problem)
        .unwrap_or_else(|e| panic!("Cannot load the new model {:?}", e));
    assert_eq!(model.state_metadata, new_model.state_metadata);
    assert_eq!(model.table_registry, new_model.table_registry);
    assert_eq!(model.target, new_model.target);
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
    assert_eq!(model.state_functions, new_model.state_functions);
    assert_eq!(model.transition_dominance, new_model.transition_dominance);
}

#[test]
fn test_emit_mpdtsp_problem() {
    let domain = fs::read_to_string("tests/example_problems/m-pdtsp/m-pdtsp-domain.yaml")
        .unwrap_or_else(|e| panic!("Cannot read example domain {:?}", e));
    let domain = yaml_rust::YamlLoader::load_from_str(&domain)
        .unwrap_or_else(|e| panic!("Cannot load example domain {:?}", e));
    assert_eq!(domain.len(), 1);
    let domain = &domain[0];

    let problem = fs::read_to_string("tests/example_problems/m-pdtsp/m-pdtsp-problem.yaml")
        .unwrap_or_else(|e| panic!("Cannot read example problem {:?}", e));
    let problem = yaml_rust::YamlLoader::load_from_str(&problem)
        .unwrap_or_else(|e| panic!("Cannot load example problem {:?}", e));
    assert_eq!(problem.len(), 1);
    let problem = &problem[0];

    let model = dypdl_parser::load_model_from_yaml(domain, problem)
        .unwrap_or_else(|e| panic!("Cannot load example model due to {:?}", e));

    let (new_domain, new_problem) = dump_model(&model).unwrap_or_else(|e| {
        panic!("Couldn't dump the model: {:?}", e);
    });

    let new_domain = yaml_rust::YamlLoader::load_from_str(&new_domain)
        .unwrap_or_else(|e| panic!("Cannot load the new domain {:?}", e));
    assert_eq!(new_domain.len(), 1);
    let new_domain = &new_domain[0];

    let new_problem = yaml_rust::YamlLoader::load_from_str(&new_problem)
        .unwrap_or_else(|e| panic!("Cannot load the new problem {:?}", e));
    assert_eq!(new_problem.len(), 1);
    let new_problem = &new_problem[0];

    let new_model = dypdl_parser::load_model_from_yaml(new_domain, new_problem)
        .unwrap_or_else(|e| panic!("Cannot load the new model {:?}", e));
    assert_eq!(model.state_metadata, new_model.state_metadata);
    assert_eq!(model.table_registry, new_model.table_registry);
    assert_eq!(model.target, new_model.target);
    assert_eq!(model.state_constraints, new_model.state_constraints);
    assert_eq!(model.base_cases, new_model.base_cases);
    assert_eq!(model.base_states, new_model.base_states);
    assert_eq!(model.reduce_function, new_model.reduce_function);
    assert_eq!(model.cost_type, new_model.cost_type);
    assert!(transition_equality_ignoring_parameter(
        model.forward_transitions,
        new_model.forward_transitions
    ));
    assert!(transition_equality_ignoring_parameter(
        model.forward_forced_transitions,
        new_model.forward_forced_transitions
    ));
    assert!(transition_equality_ignoring_parameter(
        model.backward_transitions,
        new_model.backward_transitions
    ));
    assert!(transition_equality_ignoring_parameter(
        model.backward_forced_transitions,
        new_model.backward_forced_transitions
    ));
    assert_eq!(model.dual_bounds, new_model.dual_bounds);
    assert_eq!(model.state_functions, new_model.state_functions);
    assert_eq!(model.transition_dominance, new_model.transition_dominance);
}
