use didp_parser::variable;
use didp_parser::Transition;
use std::collections;

#[derive(Debug, PartialEq, Clone)]
pub struct OneParameterSuccessorGenerator<'a, T: variable::Numeric> {
    relevant_set_variables: Vec<usize>,
    set_element_to_transitions: Vec<Vec<Vec<Transition<T>>>>,
    relevant_vector_variables: Vec<usize>,
    vector_element_to_transitions: Vec<Vec<Vec<Transition<T>>>>,
    global_transitions: Vec<&'a Transition<T>>,
    registry: &'a didp_parser::TableRegistry,
}

impl<'a, T: variable::Numeric> OneParameterSuccessorGenerator<'a, T> {
    pub fn new(
        model: &'a didp_parser::Model<T>,
        backward: bool,
    ) -> OneParameterSuccessorGenerator<'a, T> {
        let transitions = if backward {
            &model.backward_transitions
        } else {
            &model.forward_transitions
        };
        let n = model.state_metadata.number_of_set_variables();
        let mut relevant_set_variables = collections::BTreeSet::new();
        let mut set_element_to_transitions: Vec<Vec<Vec<Transition<T>>>> = (0..n)
            .map(|i| {
                let m = model.state_metadata.set_variable_capacity(i);
                (0..m).map(|_| Vec::new()).collect()
            })
            .collect();
        let n = model.state_metadata.number_of_vector_variables();
        let mut relevant_vector_variables = collections::BTreeSet::new();
        let mut vector_element_to_transitions: Vec<Vec<Vec<Transition<T>>>> = (0..n)
            .map(|i| {
                let m = model.state_metadata.vector_variable_capacity(i);
                (0..m).map(|_| Vec::new()).collect()
            })
            .collect();
        let mut global_transitions = Vec::new();
        for op in transitions {
            if !op.elements_in_set_variable.is_empty() {
                let op = Transition {
                    name: op.name.clone(),
                    elements_in_set_variable: op.elements_in_set_variable[1..].to_vec(),
                    elements_in_vector_variable: op.elements_in_vector_variable.clone(),
                    preconditions: op.preconditions.clone(),
                    set_effects: op.set_effects.clone(),
                    vector_effects: op.vector_effects.clone(),
                    element_effects: op.element_effects.clone(),
                    integer_effects: op.integer_effects.clone(),
                    integer_resource_effects: op.integer_resource_effects.clone(),
                    continuous_effects: op.continuous_effects.clone(),
                    continuous_resource_effects: op.continuous_resource_effects.clone(),
                    cost: op.cost.clone(),
                };
                let (i, v) = op.elements_in_set_variable[0];
                set_element_to_transitions[i][v].push(op);
                relevant_set_variables.insert(i);
            } else if !op.elements_in_vector_variable.is_empty() {
                let op = Transition {
                    name: op.name.clone(),
                    elements_in_set_variable: op.elements_in_set_variable.clone(),
                    elements_in_vector_variable: op.elements_in_vector_variable[1..].to_vec(),
                    preconditions: op.preconditions.clone(),
                    set_effects: op.set_effects.clone(),
                    vector_effects: op.vector_effects.clone(),
                    element_effects: op.element_effects.clone(),
                    integer_effects: op.integer_effects.clone(),
                    integer_resource_effects: op.integer_resource_effects.clone(),
                    continuous_effects: op.continuous_effects.clone(),
                    continuous_resource_effects: op.continuous_resource_effects.clone(),
                    cost: op.cost.clone(),
                };
                let (i, v) = op.elements_in_vector_variable[0];
                vector_element_to_transitions[i][v].push(op);
                relevant_vector_variables.insert(i);
            } else {
                global_transitions.push(op);
            }
        }
        OneParameterSuccessorGenerator {
            relevant_set_variables: relevant_set_variables.into_iter().collect(),
            set_element_to_transitions,
            relevant_vector_variables: relevant_vector_variables.into_iter().collect(),
            vector_element_to_transitions,
            global_transitions,
            registry: &model.table_registry,
        }
    }

    pub fn generate_applicable_transitions<'b>(
        &'a self,
        state: &'b didp_parser::State,
        mut result: Vec<&'a Transition<T>>,
    ) -> Vec<&'a Transition<T>> {
        result.clear();
        for op in &self.global_transitions {
            if op.is_applicable(state, self.registry) {
                result.push(op);
            }
        }
        for i in &self.relevant_set_variables {
            for v in state.signature_variables.set_variables[*i].ones() {
                for op in &self.set_element_to_transitions[*i][v] {
                    if op.is_applicable(state, self.registry) {
                        result.push(op);
                    }
                }
            }
        }
        for i in &self.relevant_vector_variables {
            for v in &state.signature_variables.vector_variables[*i] {
                for op in &self.vector_element_to_transitions[*i][*v] {
                    if op.is_applicable(state, self.registry) {
                        result.push(op);
                    }
                }
            }
        }
        result
    }

    pub fn applicable_transitions<'b>(
        &'a self,
        state: &'b didp_parser::State,
    ) -> OneParameterApplicableTransitions<'a, 'b, T> {
        OneParameterApplicableTransitions {
            state,
            generator: self,
            global_iter: self.global_transitions.iter(),
            relevant_iter: None,
            variable_index: 0,
            ones: None,
            vector_iter: None,
            iter: None,
        }
    }
}

pub struct OneParameterApplicableTransitions<'a, 'b, T: variable::Numeric> {
    state: &'b didp_parser::State,
    generator: &'a OneParameterSuccessorGenerator<'a, T>,
    global_iter: std::slice::Iter<'a, &'a Transition<T>>,
    relevant_iter: Option<std::slice::Iter<'a, usize>>,
    variable_index: usize,
    ones: Option<fixedbitset::Ones<'b>>,
    vector_iter: Option<std::slice::Iter<'b, usize>>,
    iter: Option<std::slice::Iter<'a, Transition<T>>>,
}

impl<'a, 'b, T: variable::Numeric> OneParameterApplicableTransitions<'a, 'b, T> {
    fn next_vector(&mut self) -> Option<&'a Transition<T>> {
        if let Some(vector_iter) = &mut self.vector_iter {
            if let Some(v) = vector_iter.next() {
                self.iter = Some(
                    self.generator.vector_element_to_transitions[self.variable_index][*v].iter(),
                );
                return self.next();
            }
        }
        match self.relevant_iter.as_mut().unwrap().next() {
            Some(v) => {
                self.vector_iter = Some(self.state.signature_variables.vector_variables[*v].iter());
                self.variable_index = *v;
                self.next()
            }
            None => None,
        }
    }

    fn next_set(&mut self) -> Option<&'a Transition<T>> {
        if self.vector_iter.is_some() {
            return self.next_vector();
        }

        if let Some(ones) = &mut self.ones {
            if let Some(v) = ones.next() {
                self.iter =
                    Some(self.generator.set_element_to_transitions[self.variable_index][v].iter());
                return self.next();
            }
        }
        match self.relevant_iter.as_mut().unwrap().next() {
            Some(v) => {
                self.ones = Some(self.state.signature_variables.set_variables[*v].ones());
                self.variable_index = *v;
                self.next()
            }
            None => {
                self.ones = None;
                self.relevant_iter = Some(self.generator.relevant_vector_variables.iter());
                self.next_vector()
            }
        }
    }
}

impl<'a, 'b, T: variable::Numeric> Iterator for OneParameterApplicableTransitions<'a, 'b, T> {
    type Item = &'a Transition<T>;

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.global_iter.next() {
            Some(op) => {
                if op.is_applicable(self.state, self.generator.registry) {
                    Some(op)
                } else {
                    self.next()
                }
            }
            None => match &mut self.iter {
                Some(iter) => match iter.next() {
                    Some(op) => {
                        if op.is_applicable(self.state, self.generator.registry) {
                            Some(op)
                        } else {
                            self.next()
                        }
                    }
                    None => self.next_set(),
                },
                None => {
                    self.relevant_iter = Some(self.generator.relevant_set_variables.iter());
                    self.next_set()
                }
            },
        }
    }
}