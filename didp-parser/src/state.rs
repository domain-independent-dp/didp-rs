use crate::variable;
use crate::yaml_util;
use std::cmp::Ordering;
use std::collections;
use std::fmt;
use std::rc::Rc;
use std::str;

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct State<T: variable::Numeric> {
    pub signature_variables: Rc<SignatureVariables<T>>,
    pub resource_variables: Vec<T>,
    pub stage: variable::ElementVariable,
    pub cost: T,
}

#[derive(Debug, Hash, PartialEq, Eq, Clone)]
pub struct SignatureVariables<T: variable::Numeric> {
    pub set_variables: Vec<variable::SetVariable>,
    pub permutation_variables: Vec<variable::PermutationVariable>,
    pub element_variables: Vec<variable::ElementVariable>,
    pub numeric_variables: Vec<T>,
}

impl<T: variable::Numeric> State<T> {
    pub fn load_from_yaml(
        value: &yaml_rust::Yaml,
        metadata: &StateMetadata,
    ) -> Result<State<T>, yaml_util::YamlContentErr>
    where
        <T as str::FromStr>::Err: fmt::Debug,
    {
        let value = yaml_util::get_map(value)?;
        let mut set_variables = Vec::with_capacity(metadata.set_variable_names.len());
        for name in &metadata.set_variable_names {
            let values = yaml_util::get_usize_array_by_key(value, name)?;
            let mut set = variable::SetVariable::with_capacity(
                metadata.get_set_variable_capacity_by_name(name),
            );
            for v in values {
                set.insert(v);
            }
            set_variables.push(set);
        }
        let mut permutation_variables =
            Vec::with_capacity(metadata.permutation_variable_names.len());
        for name in &metadata.permutation_variable_names {
            let permutation = yaml_util::get_usize_array_by_key(value, name)?;
            permutation_variables.push(permutation);
        }
        let mut element_variables = Vec::with_capacity(metadata.element_variable_names.len());
        for name in &metadata.element_variable_names {
            let element = yaml_util::get_usize_by_key(value, name)?;
            element_variables.push(element);
        }
        let mut numeric_variables = Vec::with_capacity(metadata.numeric_variable_names.len());
        for name in &metadata.numeric_variable_names {
            let value = yaml_util::get_numeric_by_key::<T>(value, name)?;
            numeric_variables.push(value);
        }
        let mut resource_variables = Vec::with_capacity(metadata.resource_variable_names.len());
        for name in &metadata.resource_variable_names {
            let value = yaml_util::get_numeric_by_key(value, name)?;
            resource_variables.push(value);
        }
        let stage = if let Ok(value) = yaml_util::get_usize_by_key(value, "stage") {
            value
        } else {
            0
        };
        let cost = if let Ok(value) = yaml_util::get_numeric_by_key(value, "cost") {
            value
        } else {
            T::zero()
        };
        Ok(State {
            signature_variables: Rc::new(SignatureVariables {
                set_variables,
                permutation_variables,
                element_variables,
                numeric_variables,
            }),
            resource_variables,
            stage,
            cost,
        })
    }
}

pub struct StateMetadata {
    pub object_names: Vec<String>,
    pub name_to_object: collections::HashMap<String, usize>,
    pub object_numbers: Vec<usize>,

    pub set_variable_names: Vec<String>,
    pub name_to_set_variable: collections::HashMap<String, usize>,
    pub set_variable_to_object: Vec<usize>,

    pub permutation_variable_names: Vec<String>,
    pub name_to_permutation_variable: collections::HashMap<String, usize>,
    pub permutation_variable_to_object: Vec<usize>,

    pub element_variable_names: Vec<String>,
    pub name_to_element_variable: collections::HashMap<String, usize>,
    pub element_variable_to_object: Vec<usize>,

    pub numeric_variable_names: Vec<String>,
    pub name_to_numeric_variable: collections::HashMap<String, usize>,

    pub resource_variable_names: Vec<String>,
    pub name_to_resource_variable: collections::HashMap<String, usize>,
    pub less_is_better: Vec<bool>,
}

impl StateMetadata {
    pub fn get_set_variable_capacity(&self, i: usize) -> usize {
        self.object_numbers[self.set_variable_to_object[i]]
    }

    pub fn get_set_variable_capacity_by_name(&self, name: &str) -> usize {
        self.object_numbers[self.set_variable_to_object[self.name_to_set_variable[name]]]
    }

    pub fn get_permutation_variable_capacity(&self, i: usize) -> usize {
        self.object_numbers[self.permutation_variable_to_object[i]]
    }

    pub fn get_permutation_variable_capacity_by_name(&self, name: &str) -> usize {
        self.object_numbers
            [self.permutation_variable_to_object[self.name_to_permutation_variable[name]]]
    }

    pub fn get_element_variable_capacity(&self, i: usize) -> usize {
        self.object_numbers[self.element_variable_to_object[i]]
    }

    pub fn get_element_variable_capacity_by_name(&self, name: &str) -> usize {
        self.object_numbers[self.element_variable_to_object[self.name_to_element_variable[name]]]
    }

    pub fn dominance<T: variable::Numeric>(&self, a: &[T], b: &[T]) -> Option<Ordering> {
        debug_assert!(a.len() == b.len());

        let mut result = Some(Ordering::Equal);
        for (i, (v1, v2)) in a.iter().zip(b.iter()).enumerate() {
            match result {
                Some(Ordering::Equal) => {
                    if v1 < v2 {
                        if self.less_is_better[i] {
                            result = Some(Ordering::Greater);
                        } else {
                            result = Some(Ordering::Less);
                        }
                    }
                    if v1 > v2 {
                        if self.less_is_better[i] {
                            result = Some(Ordering::Less);
                        } else {
                            result = Some(Ordering::Greater);
                        }
                    }
                }
                Some(Ordering::Less) => {
                    if v1 < v2 {
                        if self.less_is_better[i] {
                            return None;
                        }
                    } else if v1 > v2 && !self.less_is_better[i] {
                        return None;
                    }
                }
                Some(Ordering::Greater) => {
                    if v1 > v2 {
                        if self.less_is_better[i] {
                            return None;
                        }
                    } else if v1 < v2 && !self.less_is_better[i] {
                        return None;
                    }
                }
                None => {}
            }
        }
        result
    }

    pub fn get_grounded_parameter_set_from_yaml(
        &self,
        value: &yaml_rust::Yaml,
    ) -> Result<GroundedParameterTrpet, yaml_util::YamlContentErr> {
        let map = yaml_util::get_map(value)?;
        let mut parameters_set: Vec<collections::HashMap<String, usize>> =
            Vec::with_capacity(map.len());
        let mut elements_in_set_variable_set: Vec<Vec<(usize, usize)>> =
            Vec::with_capacity(map.len());
        let mut elements_in_permutation_variable_set: Vec<Vec<(usize, usize)>> =
            Vec::with_capacity(map.len());
        for (key, value) in map {
            let key = yaml_util::get_string(key)?;
            let value = yaml_util::get_string(value)?;
            let (n, set_index, permutation_index) = if let Some(i) = self.name_to_object.get(&value)
            {
                (self.object_numbers[*i], None, None)
            } else if let Some(i) = self.name_to_set_variable.get(&value) {
                (self.get_set_variable_capacity(*i), Some(*i), None)
            } else if let Some(i) = self.name_to_permutation_variable.get(&value) {
                (self.get_permutation_variable_capacity(*i), None, Some(*i))
            } else {
                return Err(yaml_util::YamlContentErr::new(format!(
                    "no such object, set variable, or permutation variable `{}`",
                    value
                )));
            };
            let mut new_parameteres_set = Vec::with_capacity(parameters_set.len() * n);
            let mut new_elements_in_set_variable_set =
                Vec::with_capacity(elements_in_set_variable_set.len() * n);
            let mut new_elements_in_permutation_variable_set =
                Vec::with_capacity(elements_in_permutation_variable_set.len() * n);
            for ((parameters, elements_in_set_variable), elements_in_permutation_variable) in
                parameters_set
                    .iter()
                    .zip(elements_in_set_variable_set.iter())
                    .zip(elements_in_permutation_variable_set.iter())
            {
                for i in 0..n {
                    let mut parameters = parameters.clone();
                    parameters.insert(key.clone(), i);
                    let mut elements_in_set_variable = elements_in_set_variable.clone();
                    if let Some(j) = set_index {
                        elements_in_set_variable.push((j, i));
                    }
                    let mut elements_in_permutation_variable =
                        elements_in_permutation_variable.clone();
                    if let Some(j) = permutation_index {
                        elements_in_permutation_variable.push((j, i));
                    }
                    new_parameteres_set.push(parameters);
                    new_elements_in_set_variable_set.push(elements_in_set_variable);
                    new_elements_in_permutation_variable_set.push(elements_in_permutation_variable);
                }
            }
            parameters_set = new_parameteres_set;
            elements_in_set_variable_set = new_elements_in_set_variable_set;
            elements_in_permutation_variable_set = new_elements_in_permutation_variable_set;
        }

        Ok((
            parameters_set,
            elements_in_set_variable_set,
            elements_in_permutation_variable_set,
        ))
    }

    pub fn load_from_yaml(
        objects: &yaml_rust::Yaml,
        variables: &yaml_rust::Yaml,
        object_numbers_yaml: &yaml_rust::Yaml,
    ) -> Result<StateMetadata, yaml_util::YamlContentErr> {
        let object_names = yaml_util::get_string_array(objects)?;
        let mut name_to_object = collections::HashMap::new();
        for (i, name) in object_names.iter().enumerate() {
            name_to_object.insert(name.clone(), i);
        }
        let object_numbers_yaml = yaml_util::get_map(object_numbers_yaml)?;
        let mut object_numbers = Vec::with_capacity(object_names.len());
        for (i, name) in object_names.iter().enumerate() {
            object_numbers[i] = yaml_util::get_usize_by_key(object_numbers_yaml, name)?;
        }

        let mut set_variable_names = Vec::new();
        let mut name_to_set_variable = collections::HashMap::new();
        let mut set_variable_to_object = Vec::new();
        let mut permutation_variable_names = Vec::new();
        let mut name_to_permutation_variable = collections::HashMap::new();
        let mut permutation_variable_to_object = Vec::new();
        let mut element_variable_names = Vec::new();
        let mut name_to_element_variable = collections::HashMap::new();
        let mut element_variable_to_object = Vec::new();
        let mut numeric_variable_names = Vec::new();
        let mut name_to_numeric_variable = collections::HashMap::new();
        let mut resource_variable_names = Vec::new();
        let mut name_to_resource_variable = collections::HashMap::new();
        let mut less_is_better = Vec::new();

        let variables = yaml_util::get_array(variables)?;
        for value in variables {
            let map = yaml_util::get_map(value)?;
            let name = yaml_util::get_string_by_key(map, "name")?;
            let variable_type = yaml_util::get_string_by_key(map, "type")?;
            match &variable_type[..] {
                "set" => {
                    let id = Self::get_object_id(map, &name_to_object)?;
                    set_variable_to_object.push(id);
                    name_to_set_variable.insert(name.clone(), set_variable_names.len());
                    set_variable_names.push(name);
                }
                "permutation" => {
                    let id = Self::get_object_id(map, &name_to_object)?;
                    permutation_variable_to_object.push(id);
                    name_to_permutation_variable.insert(name.clone(), set_variable_names.len());
                    permutation_variable_names.push(name);
                }
                "element" => {
                    let id = Self::get_object_id(map, &name_to_object)?;
                    element_variable_to_object.push(id);
                    name_to_element_variable.insert(name.clone(), set_variable_names.len());
                    element_variable_names.push(name);
                }
                "numeric" => {
                    name_to_numeric_variable.insert(name.clone(), numeric_variable_names.len());
                    numeric_variable_names.push(name);
                }
                "resource" => {
                    name_to_resource_variable.insert(name.clone(), numeric_variable_names.len());
                    resource_variable_names.push(name);
                    let preference = match yaml_util::get_bool_by_key(map, "less_is_better") {
                        Ok(value) => value,
                        _ => false,
                    };
                    less_is_better.push(preference);
                }
                value => {
                    return Err(yaml_util::YamlContentErr::new(format!(
                        "`{:?}` is not a variable type",
                        value
                    )))
                }
            }
        }

        Ok(StateMetadata {
            object_names,
            name_to_object,
            object_numbers,
            set_variable_names,
            name_to_set_variable,
            set_variable_to_object,
            permutation_variable_names,
            name_to_permutation_variable,
            permutation_variable_to_object,
            element_variable_names,
            name_to_element_variable,
            element_variable_to_object,
            numeric_variable_names,
            name_to_numeric_variable,
            resource_variable_names,
            name_to_resource_variable,
            less_is_better,
        })
    }

    fn get_object_id(
        map: &linked_hash_map::LinkedHashMap<yaml_rust::Yaml, yaml_rust::Yaml>,
        name_to_object: &collections::HashMap<String, usize>,
    ) -> Result<usize, yaml_util::YamlContentErr> {
        let name = yaml_util::get_string_by_key(map, "object")?;
        match name_to_object.get(&name) {
            Some(id) => Ok(*id),
            None => Err(yaml_util::YamlContentErr::new(format!(
                "object `{}` does not exist",
                name
            ))),
        }
    }
}

type GroundedParameterTrpet = (
    Vec<collections::HashMap<String, usize>>,
    Vec<Vec<(usize, usize)>>,
    Vec<Vec<(usize, usize)>>,
);

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_state_metadata() -> StateMetadata {
        let resource_variable_names = vec!["r1".to_string(), "r2".to_string(), "r3".to_string()];
        let mut name_to_resource_variable = collections::HashMap::new();
        name_to_resource_variable.insert("r1".to_string(), 0);
        name_to_resource_variable.insert("r2".to_string(), 1);
        name_to_resource_variable.insert("r3".to_string(), 2);

        StateMetadata {
            object_names: Vec::new(),
            name_to_object: collections::HashMap::new(),
            object_numbers: Vec::new(),
            set_variable_names: Vec::new(),
            name_to_set_variable: collections::HashMap::new(),
            set_variable_to_object: Vec::new(),
            permutation_variable_names: Vec::new(),
            name_to_permutation_variable: collections::HashMap::new(),
            permutation_variable_to_object: Vec::new(),
            element_variable_names: Vec::new(),
            name_to_element_variable: collections::HashMap::new(),
            element_variable_to_object: Vec::new(),
            numeric_variable_names: Vec::new(),
            name_to_numeric_variable: collections::HashMap::new(),
            resource_variable_names,
            name_to_resource_variable,
            less_is_better: vec![false, false, true],
        }
    }

    #[test]
    fn resource_variables_dominance() {
        let metadata = generate_state_metadata();
        let a = vec![1, 2, 2];
        let b = vec![1, 2, 2];
        assert!(matches!(metadata.dominance(&a, &b), Some(Ordering::Equal)));
        let b = vec![1, 1, 3];
        assert!(matches!(
            metadata.dominance(&a, &b),
            Some(Ordering::Greater)
        ));
        assert!(matches!(metadata.dominance(&b, &a), Some(Ordering::Less)));
        let b = vec![1, 3, 3];
        assert!(metadata.dominance(&b, &a).is_none());
    }

    #[test]
    #[should_panic]
    fn resource_variables_dominance_length_panic() {
        let metadata = generate_state_metadata();
        let a = vec![1, 2, 2];
        let b = vec![1, 2];
        metadata.dominance(&b, &a);
    }
}
