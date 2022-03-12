use crate::errors::ProblemErr;
use crate::variable;
use std::cmp::Ordering;
use std::collections;
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

    pub fn from_yaml(data: &Yaml) -> Result<StateMetadata, ProblemErr> {
        let map = match data {
            Yaml::Hash(map) => map,
            _ => {
                return Err(ProblemErr::Reason(format!(
                    "the value is not Hash, but `{:?}`",
                    data
                )))
            }
        };
        let object_names = match map.get(&Yaml::String("objects".to_string())) {
            Some(Yaml::Array(array)) => Self::parse_string_array(array)?,
            Some(value) => {
                return Err(ProblemErr::Reason(format!(
                    "the value of key `objects` is not Array, but `{:?}`",
                    value
                )))
            }
            None => {
                return Err(ProblemErr::Reason(
                    "key `objects` not found in the domain yaml".to_string(),
                ))
            }
        };
        let mut name_to_object = collections::HashMap::new();
        for (i, name) in object_names.iter().enumerate() {
            name_to_object.insert(name.clone(), i);
        }

        let variables = match map.get(&Yaml::String("variables".to_string())) {
            Some(Yaml::Hash(value)) => value,
            Some(value) => {
                return Err(ProblemErr::Reason(format!(
                    "the value of key `variables` is not Array, but `{:?}`",
                    value
                )))
            }
            None => {
                return Err(ProblemErr::Reason(
                    "key `varaibles` not found in the domain yaml".to_string(),
                ))
            }
        };

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

        for (key, value) in variables {
            let name = match key {
                Yaml::String(name) => name,
                _ => {
                    return Err(ProblemErr::Reason(format!(
                        "variable name is not String but `{:?}`",
                        key
                    )))
                }
            };
            let annotation = match value {
                Yaml::Hash(value) => value,
                _ => {
                    return Err(ProblemErr::Reason(format!(
                        "expected Hash for variable annotation but `{:?}`",
                        value
                    )))
                }
            };
            let variable_type = match annotation.get(&Yaml::String("type".to_string())) {
                Some(Yaml::String(name)) => name,
                Some(_) => {
                    return Err(ProblemErr::Reason(
                        "key `type` found in the variable annotation but not String".to_string(),
                    ))
                }
                None => {
                    return Err(ProblemErr::Reason(
                        "key `type` not found in the variable annotaiton".to_string(),
                    ))
                }
            };
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
                        Some(value) => return Err(ProblemErr::Reason(format!(
                            "key `less_is_better` found in the variable annotation but not Boolean, but {:?}", value),
                        )),
                        None => false,
                    };
                    less_is_better.push(preference);
                }
                value => {
                    return Err(ProblemErr::Reason(format!(
                        "`{:?}` is not a variable type",
                        value
                    )))
                }
            }
        }

        Ok(StateMetadata {
            object_names,
            name_to_object,
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

    fn parse_string_array(array: &[Yaml]) -> Result<Vec<String>, ProblemErr> {
        let mut result = Vec::with_capacity(array.len());
        for v in array {
            match v {
                Yaml::String(string) => result.push(string.clone()),
                _ => {
                    return Err(ProblemErr::Reason(format!(
                        "expected String but is `{:?}`",
                        v
                    )))
                }
            }
        }
        Ok(result)
    }

    fn get_object_id(
        value: &linked_hash_map::LinkedHashMap<Yaml, Yaml>,
        name_to_object: &collections::HashMap<String, usize>,
    ) -> Result<usize, ProblemErr> {
        let name = match value.get(&Yaml::String("object".to_string())) {
            Some(Yaml::String(name)) => name,
            Some(_) => {
                return Err(ProblemErr::Reason(
                    "key `object` found in the variable annotation but not String".to_string(),
                ))
            }
            None => {
                return Err(ProblemErr::Reason(
                    "key `object` not found in the variable annotaiton".to_string(),
                ))
            }
        };
        match name_to_object.get(name) {
            Some(id) => Ok(*id),
            None => Err(ProblemErr::Reason(format!(
                "object `{}` does not exist",
                name
            ))),
        }
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
