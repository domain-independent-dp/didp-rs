use crate::variable;
use crate::yaml_util;
use std::cmp::Ordering;
use std::collections;
use std::error::Error;
use std::rc::Rc;
use yaml_rust::Yaml;

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct State<T: variable::Numeric> {
    pub signature_variables: Rc<SignatureVariables<T>>,
    pub resource_variables: Vec<T>,
    pub cost: T,
}

#[derive(Debug, Hash, PartialEq, Eq, Clone)]
pub struct SignatureVariables<T: variable::Numeric> {
    pub set_variables: Vec<variable::SetVariable>,
    pub permutation_variables: Vec<variable::PermutationVariable>,
    pub element_variables: Vec<variable::ElementVariable>,
    pub numeric_variables: Vec<T>,
}

pub struct StateMetadata {
    pub object_names: Vec<String>,
    pub name_to_object: collections::HashMap<String, usize>,
    pub object_numbers: Vec<usize>,

    pub set_variable_to_name: Vec<String>,
    pub name_to_set_variable: collections::HashMap<String, usize>,
    pub set_variable_to_object: Vec<usize>,

    pub permutation_variable_to_name: Vec<String>,
    pub name_to_permutation_variable: collections::HashMap<String, usize>,
    pub permutation_variable_to_object: Vec<usize>,

    pub element_variable_to_name: Vec<String>,
    pub name_to_element_variable: collections::HashMap<String, usize>,
    pub element_variable_to_object: Vec<usize>,

    pub numeric_variable_to_name: Vec<String>,
    pub name_to_numeric_variable: collections::HashMap<String, usize>,

    pub resource_variable_to_name: Vec<String>,
    pub name_to_resource_variable: collections::HashMap<String, usize>,
    pub less_is_better: Vec<bool>,
}

impl StateMetadata {
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

    pub fn new(
        domain: &linked_hash_map::LinkedHashMap<Yaml, Yaml>,
        problem: &linked_hash_map::LinkedHashMap<Yaml, Yaml>,
    ) -> Result<StateMetadata, Box<dyn Error>> {
        let object_names = yaml_util::get_string_array(domain, "objects")?;
        let mut name_to_object = collections::HashMap::new();
        for (i, name) in object_names.iter().enumerate() {
            name_to_object.insert(name.clone(), i);
        }
        let object_numbers = Self::collect_object_numbers(problem, &name_to_object)?;

        let mut set_variable_to_name = Vec::new();
        let mut name_to_set_variable = collections::HashMap::new();
        let mut set_variable_to_object = Vec::new();
        let mut permutation_variable_to_name = Vec::new();
        let mut name_to_permutation_variable = collections::HashMap::new();
        let mut permutation_variable_to_object = Vec::new();
        let mut element_variable_to_name = Vec::new();
        let mut name_to_element_variable = collections::HashMap::new();
        let mut element_variable_to_object = Vec::new();
        let mut numeric_variable_to_name = Vec::new();
        let mut name_to_numeric_variable = collections::HashMap::new();
        let mut resource_variable_to_name = Vec::new();
        let mut name_to_resource_variable = collections::HashMap::new();
        let mut less_is_better = Vec::new();

        let variables = yaml_util::get_hash_value(problem, "variables")?;
        for (key, value) in variables {
            let name = yaml_util::get_string(key)?;
            let annotation = yaml_util::get_hash(value)?;
            let variable_type = yaml_util::get_string_value(annotation, "type")?;
            match &variable_type[..] {
                "set" => {
                    let id = Self::get_object_id(&annotation, &name_to_object)?;
                    set_variable_to_object.push(id);
                    name_to_set_variable.insert(name.clone(), set_variable_to_name.len());
                    set_variable_to_name.push(name.clone());
                }
                "permutation" => {
                    let id = Self::get_object_id(&annotation, &name_to_object)?;
                    permutation_variable_to_object.push(id);
                    name_to_permutation_variable.insert(name.clone(), set_variable_to_name.len());
                    permutation_variable_to_name.push(name.clone());
                }
                "element" => {
                    let id = Self::get_object_id(&annotation, &name_to_object)?;
                    element_variable_to_object.push(id);
                    name_to_element_variable.insert(name.clone(), set_variable_to_name.len());
                    element_variable_to_name.push(name.clone());
                }
                "numeric" => {
                    name_to_numeric_variable.insert(name.clone(), numeric_variable_to_name.len());
                    numeric_variable_to_name.push(name.clone());
                }
                "resource" => {
                    name_to_resource_variable.insert(name.clone(), numeric_variable_to_name.len());
                    resource_variable_to_name.push(name.clone());
                    let preference = match annotation
                        .get(&Yaml::String("less_is_better".to_string()))
                    {
                        Some(Yaml::Boolean(value)) => *value,
                        Some(value) => return Err(yaml_util::YamlContentErr::new(format!(
                            "key `less_is_better` found in the variable annotation but not Boolean, but {:?}", value),
                        ).into()),
                        None => false,
                    };
                    less_is_better.push(preference);
                }
                value => {
                    return Err(yaml_util::YamlContentErr::new(format!(
                        "`{:?}` is not a variable type",
                        value
                    ))
                    .into())
                }
            }
        }

        Ok(StateMetadata {
            object_names,
            name_to_object,
            object_numbers,
            set_variable_to_name,
            name_to_set_variable,
            set_variable_to_object,
            permutation_variable_to_name,
            name_to_permutation_variable,
            permutation_variable_to_object,
            element_variable_to_name,
            name_to_element_variable,
            element_variable_to_object,
            numeric_variable_to_name,
            name_to_numeric_variable,
            resource_variable_to_name,
            name_to_resource_variable,
            less_is_better,
        })
    }

    fn get_object_id(
        map: &linked_hash_map::LinkedHashMap<Yaml, Yaml>,
        name_to_object: &collections::HashMap<String, usize>,
    ) -> Result<usize, yaml_util::YamlContentErr> {
        let name = yaml_util::get_string_value(map, "object")?;
        match name_to_object.get(&name) {
            Some(id) => Ok(*id),
            None => Err(yaml_util::YamlContentErr::new(format!(
                "object `{}` does not exist",
                name
            ))),
        }
    }

    fn collect_object_numbers(
        map: &linked_hash_map::LinkedHashMap<Yaml, Yaml>,
        name_to_object: &collections::HashMap<String, usize>,
    ) -> Result<Vec<usize>, Box<dyn Error>> {
        let mut object_numbers: Vec<usize> = (0..name_to_object.len()).map(|_| 0).collect();
        for (key, value) in map {
            let key = yaml_util::get_string(key)?;
            let value = yaml_util::get_usize(value)?;
            match name_to_object.get(&key) {
                Some(i) => object_numbers[*i] = value,
                None => {
                    return Err(yaml_util::YamlContentErr::new(format!(
                        "object `{}` not found",
                        key
                    ))
                    .into())
                }
            }
        }
        Ok(object_numbers)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_state_metadata() -> StateMetadata {
        let mut name_to_resource_variable = collections::HashMap::new();
        name_to_resource_variable.insert("r1".to_string(), 0);
        name_to_resource_variable.insert("r2".to_string(), 1);
        name_to_resource_variable.insert("r3".to_string(), 2);

        StateMetadata {
            object_names: Vec::new(),
            name_to_object: collections::HashMap::new(),
            object_numbers: Vec::new(),
            set_variable_to_name: Vec::new(),
            name_to_set_variable: collections::HashMap::new(),
            set_variable_to_object: Vec::new(),
            permutation_variable_to_name: Vec::new(),
            name_to_permutation_variable: collections::HashMap::new(),
            permutation_variable_to_object: Vec::new(),
            element_variable_to_name: Vec::new(),
            name_to_element_variable: collections::HashMap::new(),
            element_variable_to_object: Vec::new(),
            numeric_variable_to_name: Vec::new(),
            name_to_numeric_variable: collections::HashMap::new(),
            resource_variable_to_name: vec!["r1".to_string(), "r2".to_string(), "r3".to_string()],
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
