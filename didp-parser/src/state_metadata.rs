use crate::effect;
use crate::table_registry;
use crate::variable::{Continuous, Element, Integer, Numeric, Set, Vector};
use crate::yaml_util;
use lazy_static::lazy_static;
use rustc_hash::{FxHashMap, FxHashSet};
use std::cmp::Ordering;

pub trait DPState {


#[derive(Debug, PartialEq, Clone, Eq, Default)]
pub struct StateMetadata {
    pub object_names: Vec<String>,
    pub name_to_object: FxHashMap<String, usize>,
    pub object_numbers: Vec<usize>,

    pub set_variable_names: Vec<String>,
    pub name_to_set_variable: FxHashMap<String, usize>,
    pub set_variable_to_object: Vec<usize>,

    pub vector_variable_names: Vec<String>,
    pub name_to_vector_variable: FxHashMap<String, usize>,
    pub vector_variable_to_object: Vec<usize>,

    pub element_variable_names: Vec<String>,
    pub name_to_element_variable: FxHashMap<String, usize>,
    pub element_variable_to_object: Vec<usize>,

    pub integer_variable_names: Vec<String>,
    pub name_to_integer_variable: FxHashMap<String, usize>,

    pub continuous_variable_names: Vec<String>,
    pub name_to_continuous_variable: FxHashMap<String, usize>,

    pub integer_resource_variable_names: Vec<String>,
    pub name_to_integer_resource_variable: FxHashMap<String, usize>,
    pub integer_less_is_better: Vec<bool>,

    pub continuous_resource_variable_names: Vec<String>,
    pub name_to_continuous_resource_variable: FxHashMap<String, usize>,
    pub continuous_less_is_better: Vec<bool>,
}

impl StateMetadata {
    pub fn number_of_objects(&self) -> usize {
        self.object_names.len()
    }

    pub fn number_of_set_variables(&self) -> usize {
        self.set_variable_names.len()
    }

    pub fn number_of_vector_variables(&self) -> usize {
        self.vector_variable_names.len()
    }

    pub fn number_of_element_variables(&self) -> usize {
        self.element_variable_names.len()
    }

    pub fn number_of_integer_variables(&self) -> usize {
        self.integer_variable_names.len()
    }

    pub fn number_of_continuous_variables(&self) -> usize {
        self.continuous_variable_names.len()
    }

    pub fn number_of_integer_resource_variables(&self) -> usize {
        self.integer_resource_variable_names.len()
    }

    pub fn number_of_continuous_resource_variables(&self) -> usize {
        self.continuous_resource_variable_names.len()
    }

    pub fn set_variable_capacity(&self, i: usize) -> usize {
        self.object_numbers[self.set_variable_to_object[i]]
    }

    pub fn set_variable_capacity_by_name(&self, name: &str) -> usize {
        self.object_numbers[self.set_variable_to_object[self.name_to_set_variable[name]]]
    }

    pub fn vector_variable_capacity(&self, i: usize) -> usize {
        self.object_numbers[self.vector_variable_to_object[i]]
    }

    pub fn vector_variable_capacity_by_name(&self, name: &str) -> usize {
        self.object_numbers[self.vector_variable_to_object[self.name_to_vector_variable[name]]]
    }

    pub fn element_variable_capacity(&self, i: usize) -> usize {
        self.object_numbers[self.element_variable_to_object[i]]
    }

    pub fn element_variable_capacity_by_name(&self, name: &str) -> usize {
        self.object_numbers[self.element_variable_to_object[self.name_to_element_variable[name]]]
    }

    pub fn get_name_set(&self) -> FxHashSet<String> {
        let mut name_set = FxHashSet::default();
        for name in &self.object_names {
            name_set.insert(name.clone());
        }
        for name in &self.set_variable_names {
            name_set.insert(name.clone());
        }
        for name in &self.vector_variable_names {
            name_set.insert(name.clone());
        }
        for name in &self.element_variable_names {
            name_set.insert(name.clone());
        }
        for name in &self.integer_variable_names {
            name_set.insert(name.clone());
        }
        for name in &self.continuous_variable_names {
            name_set.insert(name.clone());
        }
        for name in &self.integer_resource_variable_names {
            name_set.insert(name.clone());
        }
        for name in &self.continuous_resource_variable_names {
            name_set.insert(name.clone());
        }
        name_set
    }

    pub fn dominance<U: DPState>(&self, a: &U, b: &U) -> Option<Ordering> {
        let status = Some(Ordering::Equal);
        let x = |i| a.get_integer_resource_variable(i);
        let y = |i| b.get_integer_resource_variable(i);
        let status = Self::compare_resource_variables(&x, &y, &self.integer_less_is_better, status);
        status?;
        let x = |i| a.get_continuous_resource_variable(i);
        let y = |i| b.get_continuous_resource_variable(i);
        Self::compare_resource_variables(&x, &y, &self.continuous_less_is_better, status)
    }

    fn compare_resource_variables<T: Numeric, F, G>(
        x: &F,
        y: &G,
        less_is_better: &[bool],
        mut status: Option<Ordering>,
    ) -> Option<Ordering>
    where
        F: Fn(usize) -> T,
        G: Fn(usize) -> T,
    {
        for (i, less_is_better) in less_is_better.iter().enumerate() {
            let v1 = x(i);
            let v2 = y(i);
            match status {
                Some(Ordering::Equal) => {
                    if v1 < v2 {
                        if *less_is_better {
                            status = Some(Ordering::Greater);
                        } else {
                            status = Some(Ordering::Less);
                        }
                    }
                    if v1 > v2 {
                        if *less_is_better {
                            status = Some(Ordering::Less);
                        } else {
                            status = Some(Ordering::Greater);
                        }
                    }
                }
                Some(Ordering::Less) => {
                    if v1 < v2 {
                        if *less_is_better {
                            return None;
                        }
                    } else if v1 > v2 && !less_is_better {
                        return None;
                    }
                }
                Some(Ordering::Greater) => {
                    if v1 > v2 {
                        if *less_is_better {
                            return None;
                        }
                    } else if v1 < v2 && !less_is_better {
                        return None;
                    }
                }
                None => {}
            }
        }
        status
    }

    pub fn ground_parameters_from_yaml(
        &self,
        value: &yaml_rust::Yaml,
    ) -> Result<GroundedParameterTriplet, yaml_util::YamlContentErr> {
        let array = yaml_util::get_array(value)?;
        let mut parameters_array: Vec<FxHashMap<String, usize>> = Vec::with_capacity(array.len());
        parameters_array.push(FxHashMap::default());
        let mut elements_in_set_variable_array: Vec<Vec<(usize, usize)>> =
            Vec::with_capacity(array.len());
        elements_in_set_variable_array.push(vec![]);
        let mut elements_in_vector_variable_array: Vec<Vec<(usize, usize)>> =
            Vec::with_capacity(array.len());
        elements_in_vector_variable_array.push(vec![]);
        let mut reserved_names = FxHashSet::default();
        for value in array {
            let map = yaml_util::get_map(value)?;
            let name = yaml_util::get_string_by_key(&map, "name")?;
            if let Some(name) = reserved_names.get(&name) {
                return Err(yaml_util::YamlContentErr::new(format!(
                    "parameter name `{}` is already used",
                    name
                )));
            }
            reserved_names.insert(name.clone());
            let object = yaml_util::get_string_by_key(&map, "object")?;
            let (n, set_index, vector_index) = if let Some(i) = self.name_to_object.get(&object) {
                (self.object_numbers[*i], None, None)
            } else if let Some(i) = self.name_to_set_variable.get(&object) {
                (self.set_variable_capacity(*i), Some(*i), None)
            } else if let Some(i) = self.name_to_vector_variable.get(&object) {
                (self.vector_variable_capacity(*i), None, Some(*i))
            } else {
                return Err(yaml_util::YamlContentErr::new(format!(
                    "no such object, set variable, or vector variable `{}`",
                    object
                )));
            };
            let mut new_parameteres_set = Vec::with_capacity(parameters_array.len() * n);
            let mut new_elements_in_set_variable_array =
                Vec::with_capacity(elements_in_set_variable_array.len() * n);
            let mut new_elements_in_vector_variable_array =
                Vec::with_capacity(elements_in_vector_variable_array.len() * n);
            for ((parameters, elements_in_set_variable), elements_in_vector_variable) in
                parameters_array
                    .iter()
                    .zip(elements_in_set_variable_array.iter())
                    .zip(elements_in_vector_variable_array.iter())
            {
                for i in 0..n {
                    let mut parameters = parameters.clone();
                    parameters.insert(name.clone(), i);
                    let mut elements_in_set_variable = elements_in_set_variable.clone();
                    if let Some(j) = set_index {
                        elements_in_set_variable.push((j, i));
                    }
                    let mut elements_in_vector_variable = elements_in_vector_variable.clone();
                    if let Some(j) = vector_index {
                        elements_in_vector_variable.push((j, i));
                    }
                    new_parameteres_set.push(parameters);
                    new_elements_in_set_variable_array.push(elements_in_set_variable);
                    new_elements_in_vector_variable_array.push(elements_in_vector_variable);
                }
            }
            parameters_array = new_parameteres_set;
            elements_in_set_variable_array = new_elements_in_set_variable_array;
            elements_in_vector_variable_array = new_elements_in_vector_variable_array;
        }

        Ok((
            parameters_array,
            elements_in_set_variable_array,
            elements_in_vector_variable_array,
        ))
    }

    pub fn load_from_yaml(
        objects: &yaml_rust::Yaml,
        variables: &yaml_rust::Yaml,
        object_numbers_yaml: &yaml_rust::Yaml,
    ) -> Result<StateMetadata, yaml_util::YamlContentErr> {
        let mut reserved_names = FxHashSet::default();
        let object_names = yaml_util::get_string_array(objects)?;
        let mut name_to_object = FxHashMap::default();
        for (i, name) in object_names.iter().enumerate() {
            match reserved_names.get(name) {
                Some(name) => {
                    return Err(yaml_util::YamlContentErr::new(format!(
                        "object name `{}` is already used",
                        name
                    )))
                }
                None => {
                    reserved_names.insert(name.clone());
                    name_to_object.insert(name.clone(), i);
                }
            }
        }
        let object_numbers_yaml = yaml_util::get_map(object_numbers_yaml)?;
        let mut object_numbers: Vec<usize> = (0..object_names.len()).map(|_| 0).collect();
        for (i, name) in object_names.iter().enumerate() {
            object_numbers[i] = yaml_util::get_usize_by_key(object_numbers_yaml, name)?;
        }

        let mut set_variable_names = Vec::new();
        let mut name_to_set_variable = FxHashMap::default();
        let mut set_variable_to_object = Vec::new();
        let mut vector_variable_names = Vec::new();
        let mut name_to_vector_variable = FxHashMap::default();
        let mut vector_variable_to_object = Vec::new();
        let mut element_variable_names = Vec::new();
        let mut name_to_element_variable = FxHashMap::default();
        let mut element_variable_to_object = Vec::new();
        let mut integer_variable_names = Vec::new();
        let mut name_to_integer_variable = FxHashMap::default();
        let mut continuous_variable_names = Vec::new();
        let mut name_to_continuous_variable = FxHashMap::default();
        let mut integer_resource_variable_names = Vec::new();
        let mut name_to_integer_resource_variable = FxHashMap::default();
        let mut integer_less_is_better = Vec::new();
        let mut continuous_resource_variable_names = Vec::new();
        let mut name_to_continuous_resource_variable = FxHashMap::default();
        let mut continuous_less_is_better = Vec::new();

        let variables = yaml_util::get_array(variables)?;
        for value in variables {
            let map = yaml_util::get_map(value)?;
            let name = yaml_util::get_string_by_key(map, "name")?;
            if let Some(name) = reserved_names.get(&name) {
                return Err(yaml_util::YamlContentErr::new(format!(
                    "variable name `{}` is already used",
                    name
                )));
            }
            reserved_names.insert(name.clone());
            let variable_type = yaml_util::get_string_by_key(map, "type")?;
            match &variable_type[..] {
                "set" => {
                    let id = Self::get_object_id(map, &name_to_object)?;
                    set_variable_to_object.push(id);
                    name_to_set_variable.insert(name.clone(), set_variable_names.len());
                    set_variable_names.push(name);
                }
                "vector" => {
                    let id = Self::get_object_id(map, &name_to_object)?;
                    vector_variable_to_object.push(id);
                    name_to_vector_variable.insert(name.clone(), vector_variable_names.len());
                    vector_variable_names.push(name);
                }
                "element" => {
                    let id = Self::get_object_id(map, &name_to_object)?;
                    element_variable_to_object.push(id);
                    name_to_element_variable.insert(name.clone(), element_variable_names.len());
                    element_variable_names.push(name);
                }
                "integer" => match Self::get_less_is_better(map)? {
                    Some(value) => {
                        name_to_integer_resource_variable
                            .insert(name.clone(), integer_resource_variable_names.len());
                        integer_resource_variable_names.push(name);
                        integer_less_is_better.push(value);
                    }
                    None => {
                        name_to_integer_variable.insert(name.clone(), integer_variable_names.len());
                        integer_variable_names.push(name);
                    }
                },
                "continuous" => match Self::get_less_is_better(map)? {
                    Some(value) => {
                        name_to_continuous_resource_variable
                            .insert(name.clone(), continuous_resource_variable_names.len());
                        continuous_resource_variable_names.push(name);
                        continuous_less_is_better.push(value);
                    }
                    None => {
                        name_to_continuous_variable
                            .insert(name.clone(), continuous_variable_names.len());
                        continuous_variable_names.push(name);
                    }
                },
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
            vector_variable_names,
            name_to_vector_variable,
            vector_variable_to_object,
            element_variable_names,
            name_to_element_variable,
            element_variable_to_object,
            integer_variable_names,
            name_to_integer_variable,
            continuous_variable_names,
            name_to_continuous_variable,
            integer_resource_variable_names,
            name_to_integer_resource_variable,
            integer_less_is_better,
            continuous_resource_variable_names,
            name_to_continuous_resource_variable,
            continuous_less_is_better,
        })
    }

    fn get_object_id(
        map: &linked_hash_map::LinkedHashMap<yaml_rust::Yaml, yaml_rust::Yaml>,
        name_to_object: &FxHashMap<String, usize>,
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

    fn get_less_is_better(
        map: &linked_hash_map::LinkedHashMap<yaml_rust::Yaml, yaml_rust::Yaml>,
    ) -> Result<Option<bool>, yaml_util::YamlContentErr> {
        lazy_static! {
            static ref KEY: yaml_rust::Yaml = yaml_rust::Yaml::from_str("preference");
        }
        match map.get(&KEY) {
            Some(yaml_rust::Yaml::String(value)) if &value[..] == "greater" => Ok(Some(false)),
            Some(yaml_rust::Yaml::String(value)) if &value[..] == "less" => Ok(Some(true)),
            Some(value) => Err(yaml_util::YamlContentErr::new(format!(
                "expected `String(\"greater\")` or `String(\"less\")`, found `{:?}`",
                value
            ))),
            None => Ok(None),
        }
    }
}

type GroundedParameterTriplet = (
    Vec<FxHashMap<String, usize>>,
    Vec<Vec<(usize, usize)>>,
    Vec<Vec<(usize, usize)>>,
);