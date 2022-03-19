use didp_parser::variable;
use didp_parser::Operator;
use std::collections;

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct SuccessorGenerator<'a, T: variable::Numeric> {
    relevant_set_variables: Vec<usize>,
    set_element_to_operators: Vec<Vec<Vec<Operator<T>>>>,
    relevant_permutation_variables: Vec<usize>,
    permutation_element_to_operators: Vec<Vec<Vec<Operator<T>>>>,
    global_operators: Vec<Operator<T>>,
    metadata: &'a didp_parser::StateMetadata,
    registry: &'a didp_parser::TableRegistry<T>,
}

impl<'a, T: variable::Numeric> SuccessorGenerator<'a, T> {
    pub fn new(
        operators: &[Operator<T>],
        metadata: &'a didp_parser::StateMetadata,
        registry: &'a didp_parser::TableRegistry<T>,
    ) -> SuccessorGenerator<'a, T> {
        let n = metadata.number_of_set_variables();
        let mut relevant_set_variables = collections::BTreeSet::new();
        let mut set_element_to_operators: Vec<Vec<Vec<Operator<T>>>> = (0..n)
            .map(|i| {
                let m = metadata.set_variable_capacity(i);
                (0..m).map(|_| Vec::new()).collect()
            })
            .collect();
        let n = metadata.number_of_permutation_variables();
        let mut relevant_permutation_variables = collections::BTreeSet::new();
        let mut permutation_element_to_operators: Vec<Vec<Vec<Operator<T>>>> = (0..n)
            .map(|i| {
                let m = metadata.permutation_variable_capacity(i);
                (0..m).map(|_| Vec::new()).collect()
            })
            .collect();
        let mut global_operators = Vec::new();
        for op in operators {
            if !op.elements_in_set_variable.is_empty() {
                let op = Operator {
                    name: op.name.clone(),
                    elements_in_set_variable: op.elements_in_set_variable[1..].to_vec(),
                    elements_in_permutation_variable: op.elements_in_permutation_variable.clone(),
                    preconditions: op.preconditions.clone(),
                    set_effects: op.set_effects.clone(),
                    permutation_effects: op.permutation_effects.clone(),
                    element_effects: op.element_effects.clone(),
                    numeric_effects: op.numeric_effects.clone(),
                    resource_effects: op.resource_effects.clone(),
                    cost: op.cost.clone(),
                };
                let (i, v) = op.elements_in_set_variable[0];
                set_element_to_operators[i][v].push(op);
                relevant_set_variables.insert(i);
            } else if !op.elements_in_permutation_variable.is_empty() {
                let op = Operator {
                    name: op.name.clone(),
                    elements_in_set_variable: op.elements_in_set_variable.clone(),
                    elements_in_permutation_variable: op.elements_in_permutation_variable[1..]
                        .to_vec(),
                    preconditions: op.preconditions.clone(),
                    set_effects: op.set_effects.clone(),
                    permutation_effects: op.permutation_effects.clone(),
                    element_effects: op.element_effects.clone(),
                    numeric_effects: op.numeric_effects.clone(),
                    resource_effects: op.resource_effects.clone(),
                    cost: op.cost.clone(),
                };
                let (i, v) = op.elements_in_permutation_variable[0];
                permutation_element_to_operators[i][v].push(op);
                relevant_permutation_variables.insert(i);
            } else {
                global_operators.push(op.clone());
            }
        }
        SuccessorGenerator {
            relevant_set_variables: relevant_set_variables.into_iter().collect(),
            set_element_to_operators,
            relevant_permutation_variables: relevant_permutation_variables.into_iter().collect(),
            permutation_element_to_operators,
            global_operators,
            metadata,
            registry,
        }
    }

    pub fn generate_applicable_operators<'b>(
        &'a self,
        state: &'b didp_parser::State<T>,
        mut result: Vec<&'a Operator<T>>,
    ) -> Vec<&'a Operator<T>> {
        result.clear();
        for i in &self.relevant_set_variables {
            for v in state.signature_variables.set_variables[*i].ones() {
                for op in &self.set_element_to_operators[*i][v] {
                    if op.is_applicable(state, self.metadata, self.registry) {
                        result.push(op);
                    }
                }
            }
        }
        for i in &self.relevant_permutation_variables {
            for v in &state.signature_variables.permutation_variables[*i] {
                for op in &self.permutation_element_to_operators[*i][*v] {
                    if op.is_applicable(state, self.metadata, self.registry) {
                        result.push(op);
                    }
                }
            }
        }
        for op in &self.global_operators {
            if op.is_applicable(state, self.metadata, self.registry) {
                result.push(op);
            }
        }
        result
    }
}
